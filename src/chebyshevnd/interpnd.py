"""n-dimensional Chebyshev interpolation."""

import numbers

import numpy as np

from .backend import array_namespace, to_numpy
from .batch import ChebyshevBatch
from .batched import precontract_tables
from .batched import evaluate_batch as _evaluate_batch
from .derivatives import differentiate_coefficients
from .errors import OutOfDomainError, describe_out_of_domain
from .kernels import evaluate_point
from .utils import (
    chebyshev_gauss_lobatto_nodes,
    chebyshev_nodes,
    chebyshev_values_at_nodes_nd,
    compute_chebyshev_coefficients_nd,
    prune_trailing_slices_greedy_nd,
)

_NODE_FAMILIES = ("Chebyshev", "GaussLobatto")


class ChebyshevInterpolantND:
    """
    Chebyshev interpolant of a function of d variables on a box
    [a_1, b_1] x ... x [a_d, b_d], for any d >= 1.

    Coefficients are computed from values sampled on a tensor grid with a
    single d-dimensional DCT. Evaluation contracts the coefficient tensor
    one axis at a time with Chebyshev polynomial values at the point, in
    compiled code.

    Several functions sampled on the same grid can share one interpolant:
    give their values along a trailing function axis, and name them with
    ``output_names``. They are then evaluated together, at little more cost
    than one.

    :meth:`evaluate` handles one point on the CPU. :meth:`evaluate_batch`
    handles many points at once, on NumPy (CPU) or CuPy (GPU) arrays.

    Evaluation reuses scratch buffers held by the instance, so one instance
    must not be evaluated from several threads at once.
    """

    def __init__(self, vals, domains, nodes="Chebyshev", axis_names=None, output_names=None,
                 domain_tol=1e-12):
        """
        Construct an nD Chebyshev interpolant from values sampled on a tensor grid.

        Parameters
        ----------
        vals : array-like
            Real or complex function values sampled on the tensor grid of
            the chosen node family (``chebyshev_nodes`` per axis for
            'Chebyshev', ``chebyshev_gauss_lobatto_nodes`` for 'GaussLobatto')
            with ``np.meshgrid(..., indexing='ij')`` ordering, so that axis k
            of ``vals`` is variable k. Sample each axis at its nodes in the
            order returned (descending). Shape (N_1, ..., N_d) for one
            function, or (N_1, ..., N_d, F) for F functions on the same grid,
            in which case ``output_names`` is required.
        domains : sequence of (float, float)
            One (lower, upper) pair per axis of ``vals``, in axis order: the
            same pairs used to generate the nodes. Each lower bound must be
            smaller than its upper bound.
        nodes : str, optional
            Node family the values were sampled at: 'Chebyshev' (default) or
            'GaussLobatto'.
        axis_names : sequence of str, optional
            One name per axis, e.g. ``("nu", "u", "e")``. Defaults to
            ``("x0", "x1", ...)``.
        output_names : sequence of str, optional
            One name per function along the trailing function axis of
            ``vals``. Giving it is what declares that axis: without it,
            ``vals`` must have exactly one axis per domain, so a missing
            domain cannot be mistaken for a function axis.
        domain_tol : float, optional
            How far outside its domain a coordinate may lie before
            evaluation raises :class:`OutOfDomainError`, as a fraction of
            the axis half-width (b - a) / 2. The default 1e-12 only absorbs
            rounding at the boundaries: a Chebyshev series diverges quickly
            outside its domain, so extrapolation is never silent.
        """
        vals = np.asarray(vals)
        # Two dtypes only (design doc A4): complex input stays complex
        dtype = np.complex128 if np.iscomplexobj(vals) else np.float64
        vals = vals.astype(dtype, copy=False)

        a, b = _parse_domains(domains)
        d = a.shape[0]
        self._single = output_names is None
        if self._single and vals.ndim != d:
            hint = (" If its last axis holds several functions, pass output_names."
                    if vals.ndim == d + 1 else "")
            raise ValueError(
                f"vals has {vals.ndim} axes but {d} domains were given; give one "
                f"(lower, upper) pair per axis of vals.{hint}"
            )
        if not self._single and vals.ndim != d + 1:
            raise ValueError(
                f"output_names was given, so vals needs {d} grid axes plus a trailing "
                f"function axis ({d + 1} axes); it has {vals.ndim}."
            )
        grid_shape = vals.shape[:d]
        if nodes not in _NODE_FAMILIES:
            raise ValueError(f"nodes must be one of {_NODE_FAMILIES}, got {nodes!r}.")
        min_nodes = 2 if nodes == "GaussLobatto" else 1
        if any(n < min_nodes for n in grid_shape):
            raise ValueError(
                f"Every axis needs at least {min_nodes} node(s) for nodes={nodes!r}; "
                f"got grid shape {grid_shape}."
            )

        domain_tol = float(domain_tol)
        if not domain_tol >= 0.0:
            raise ValueError(f"domain_tol must be non-negative, got {domain_tol}.")
        self._domain_tol = domain_tol
        self._a, self._b = a, b
        self._bounds = np.stack([a, b])
        self._nodes = nodes
        self._axis_names = _parse_names(axis_names, d, "x", "axis_names")
        self._grid_shape = tuple(int(n) for n in grid_shape)
        n_out = 1 if self._single else vals.shape[-1]
        if n_out < 1:
            raise ValueError("The function axis of vals is empty.")
        self._output_names = () if self._single else _parse_names(output_names, n_out, "f", "output_names")
        self._output_index = {name: i for i, name in enumerate(self._output_names)}

        # Coefficients always carry a trailing output axis internally
        coeffs = compute_chebyshev_coefficients_nd(vals, nodes=nodes, axes=range(d))
        if self._single:
            coeffs = coeffs[..., np.newaxis]
        self._coeffs_full = np.ascontiguousarray(coeffs)

        abs_c = np.abs(self._coeffs_full)
        self._scale = abs_c.reshape(-1, n_out).max(axis=0)
        self._interp_error = _trailing_error_estimate(abs_c)
        self._discarded = np.zeros(n_out)
        self._x = np.empty(d)
        self._set_truncation(self._grid_shape)

    # ------------------------------------------------------------------ setup

    @classmethod
    def _from_tables(cls, coeffs, coeffs_full, a, b, *, nodes, axis_names, output_names,
                     domain_tol, abs_error, grid_shape):
        """
        Build an interpolant directly from coefficient tensors, e.g. the result
        of a precontraction.

        ``coeffs`` (the evaluation tensor) and ``coeffs_full`` (or None) carry
        a trailing output axis. ``output_names`` is empty for a single
        function. ``abs_error`` (one value per output) becomes the error
        estimate: it is inherited from the parent, whose bound still holds.
        """
        self = cls.__new__(cls)
        self._single = len(output_names) == 0
        self._domain_tol = domain_tol
        self._a, self._b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        self._bounds = np.stack([self._a, self._b])
        self._nodes = nodes
        self._axis_names = tuple(axis_names)
        self._grid_shape = tuple(int(n) for n in grid_shape)
        self._output_names = tuple(output_names)
        self._output_index = {name: i for i, name in enumerate(self._output_names)}
        self._coeffs_full = None if coeffs_full is None else np.ascontiguousarray(coeffs_full)
        reference = coeffs if coeffs_full is None else coeffs_full
        n_out = reference.shape[-1]
        self._scale = np.abs(reference).reshape(-1, n_out).max(axis=0)
        self._interp_error = np.array(abs_error, dtype=np.float64)
        self._discarded = np.zeros(n_out)
        self._x = np.empty(self._a.shape[0])
        self._set_compact(coeffs)
        return self

    def _set_truncation(self, kmax):
        """Keep the leading ``kmax`` block of the full tensor as the compact evaluation tensor (A2)."""
        self._set_compact(self._coeffs_full[tuple(slice(int(k)) for k in kmax)])

    def _set_compact(self, coeffs):
        """Make ``coeffs`` (trailing output axis included) the evaluation tensor."""
        self._coeffs = np.ascontiguousarray(coeffs)
        self._kmax = tuple(int(k) for k in self._coeffs.shape[:-1])
        self._flat = self._coeffs.reshape(-1)
        self._shape = np.array(self._kmax, dtype=np.int64)
        rest = self._flat.shape[0] // self._kmax[0]
        dtype = self._coeffs.dtype
        self._scratch = np.empty(len(self._kmax) + max(self._kmax))
        self._work = np.empty((2, rest), dtype=dtype)
        self._out = np.empty(self._coeffs.shape[-1], dtype=dtype)
        # Copies of the table on other devices, and copies with a set of
        # fixed axes moved to the front (for precontraction), made on first use
        self._device_tables = {}
        self._fixed_first = {}

    # ------------------------------------------------------------- properties

    @property
    def ndim(self):
        """Number of variables d."""
        return len(self._grid_shape)

    @property
    def grid_shape(self):
        """Number of nodes per axis, (N_1, ..., N_d)."""
        return self._grid_shape

    @property
    def kmax(self):
        """Number of coefficients used per axis after truncation, (k_1, ..., k_d)."""
        return self._kmax

    @property
    def domains(self):
        """The (lower, upper) bounds of each axis."""
        return tuple(zip(self._a.tolist(), self._b.tolist()))

    @property
    def axis_names(self):
        """The name of each axis."""
        return self._axis_names

    @property
    def n_outputs(self):
        """Number of functions F held by the interpolant (1 without a function axis)."""
        return self._coeffs.shape[-1]

    @property
    def output_names(self):
        """The name of each function; empty without a function axis."""
        return self._output_names

    @property
    def output_index(self):
        """Mapping from function name to its position in evaluation results."""
        return dict(self._output_index)

    @property
    def dtype(self):
        """Coefficient dtype: float64, or complex128 for complex values."""
        return self._coeffs.dtype

    @property
    def domain_tol(self):
        """Allowed overshoot outside the domain, as a fraction of the half-width."""
        return self._domain_tol

    @property
    def nodes(self):
        """Node family the values were sampled at."""
        return self._nodes

    # ------------------------------------------------------------- evaluation

    def evaluate(self, *xs):
        """
        Evaluate the interpolant at a single point.

        Parameters
        ----------
        *xs : float
            One coordinate per axis, in axis order.

        Returns
        -------
        float, complex or np.ndarray
            For a single function, its value (complex for complex values).
            With a function axis, an array of shape (F,).

        Raises
        ------
        OutOfDomainError
            If a coordinate lies outside its domain (beyond ``domain_tol``)
            or is NaN.
        """
        x = self._point(xs)
        status = evaluate_point(self._flat, self._shape, self._bounds, x, self._domain_tol,
                                self._scratch, self._work, self._out)
        if status:
            raise OutOfDomainError(
                "Point outside the interpolation domain: " + self._domain_message(x, status) + "."
            )
        return self._result(self._out)

    def evaluate_batch(self, points, return_status=False):
        """
        Evaluate the interpolant at many points at once.

        Runs on whichever library ``points`` comes from: NumPy arrays are
        evaluated on the CPU, CuPy arrays on the GPU. On the first CuPy call
        the coefficients are copied to the GPU once and kept there.

        Parameters
        ----------
        points : array, shape (B, d)
            One point per row, coordinates in axis order. For a
            1-dimensional interpolant a shape (B,) array is also accepted.
        return_status : bool, optional
            If True, return the per-point status instead of raising for
            points outside the domain.

        Returns
        -------
        values : array
            Shape (B,) for a single function, (B, F) with a function axis;
            on the same device as ``points``.
        status : int64 array, shape (B,)
            Only if ``return_status``. Bit i of ``status[j]`` is set when
            coordinate i of point j lies outside its domain or is NaN; the
            corresponding values are extrapolations and should not be used.

        Raises
        ------
        OutOfDomainError
            If any point is outside the domain and ``return_status`` is False.
        """
        xp = array_namespace(points)
        pts = xp.asarray(points, dtype=xp.float64)
        if pts.ndim == 1 and self.ndim == 1:
            pts = pts[:, None]
        if pts.ndim != 2 or pts.shape[1] != self.ndim:
            raise ValueError(
                f"points must have shape (B, {self.ndim}) for axes {self._axis_names}; "
                f"got {tuple(pts.shape)}."
            )
        coeffs, a, b = self._table_on(xp)
        values, status = _evaluate_batch(coeffs, a, b, pts, tol=self._domain_tol)
        if self._single:
            values = values[:, 0]
        if return_status:
            return values, status
        bad = status != 0
        if bool(xp.any(bad)):
            first = int(xp.flatnonzero(bad)[0])
            raise OutOfDomainError(
                f"{int(bad.sum())} of {pts.shape[0]} points are outside the interpolation domain; "
                f"the first is point {first}: "
                + self._domain_message(to_numpy(pts[first]), int(status[first])) + "."
            )
        return values

    def _table_on(self, xp):
        """The compact coefficients and bounds as arrays of module `xp`, cached per device."""
        if xp is np:
            return self._coeffs, self._a, self._b
        key = xp.__name__
        if key not in self._device_tables:
            self._device_tables[key] = (xp.asarray(self._coeffs), xp.asarray(self._a), xp.asarray(self._b))
        return self._device_tables[key]

    def _domain_message(self, x, status):
        """Describe which coordinates of `x` are outside the domain, from a status bitmask."""
        return describe_out_of_domain(self._axis_names, self._a, self._b, x, status)

    def _result(self, out):
        """Return the scalar for a single function, otherwise a copy of the array."""
        return out.item(0) if self._single else out.copy()

    def _point(self, xs):
        """Validate scalar coordinates and copy them into the instance's point buffer."""
        if len(xs) != self.ndim:
            raise TypeError(
                f"evaluate() takes {self.ndim} coordinates {self._axis_names}, got {len(xs)}."
            )
        try:
            self._x[:] = xs
        except (TypeError, ValueError):
            raise TypeError("evaluate() takes one scalar per axis.") from None
        return self._x

    # --------------------------------------------------------- precontraction

    def precontract(self, fixed):
        """
        Fix some axes at given values and return the interpolant over the others.

        This is the main speed-up when some variables stay constant over many
        evaluations (e.g. the mass ratio and spins of one waveform): the
        fixed axes are contracted once, and every later evaluation only
        works on the much smaller remaining tensor.

        Parameters
        ----------
        fixed : mapping
            Axis name (or index) -> value, e.g. ``{"nu": 0.2, "chi1": 0.5}``.
            At least one axis must remain. The cost is about one pass over
            the coefficient tensor, whichever axes are fixed.

        Returns
        -------
        ChebyshevInterpolantND
            An interpolant over the remaining axes, in their original order,
            with the same outputs, dtype and ``domain_tol``. It evaluates
            exactly like this interpolant with the fixed values inserted,
            including any truncation from :meth:`optimize`. If the full
            coefficient tensor is still held, it is precontracted too, so
            the result can be optimized again on its own. Its error
            estimate starts from this interpolant's, which still bounds it.

        Raises
        ------
        OutOfDomainError
            If a fixed value lies outside its axis's domain.

        See Also
        --------
        precontract_batch : the same with different values for each member
            of a batch.
        """
        axes, values = self._resolve_fixed(fixed)
        for value in values:
            if np.ndim(value) != 0:
                raise TypeError(
                    "precontract() takes one value per fixed axis; use precontract_batch() for arrays."
                )
        row = np.array([values], dtype=np.float64)
        # The batched routine with a batch of one: a single GEMM, whatever
        # the number and position of the fixed axes
        tables, status = self._precontract_tables(np, axes, row)
        if status[0]:
            x = np.full(self.ndim, np.nan)
            x[axes] = row[0]
            raise OutOfDomainError(
                "Fixed value outside the interpolation domain: "
                + self._domain_message(x, int(status[0])) + "."
            )
        coeffs = tables[0]
        if self._coeffs_full is None:
            full = None
        elif self._kmax == self._grid_shape:
            full = coeffs  # nothing was truncated, so the full tensor gives the same table
        else:
            full = self._precontract_tables(np, axes, row, full=True)[0][0]

        live = [i for i in range(self.ndim) if i not in axes]
        return ChebyshevInterpolantND._from_tables(
            coeffs, full, self._a[live], self._b[live],
            nodes=self._nodes,
            axis_names=[self._axis_names[i] for i in live],
            output_names=self._output_names,
            domain_tol=self._domain_tol,
            abs_error=self._interp_error + self._discarded,
            grid_shape=[self._grid_shape[i] for i in live],
        )

    def precontract_batch(self, fixed, return_status=False):
        """
        Fix some axes at different values for each member of a batch.

        The batch version of :meth:`precontract`: member j (e.g. one
        waveform) gets its own table, with the fixed axes at its own values.
        All tables are built with one matrix multiplication, on the device
        of the given arrays: NumPy on the CPU, CuPy on the GPU, where this
        interpolant's coefficients are copied once and cached.

        Parameters
        ----------
        fixed : mapping
            Axis name (or index) -> array of shape (B,) with one value per
            member, e.g. ``{"nu": nu_values, "chi1": chi1_values}``. Scalars
            are allowed alongside arrays and apply to every member. At least
            one axis must remain.
        return_status : bool, optional
            If True, return the per-member status of the fixed values
            instead of raising for values outside the domain.

        Returns
        -------
        ChebyshevBatch
            The per-member tables over the remaining axes.
        status : int64 array, shape (B,)
            Only if ``return_status``. Bit i of ``status[j]`` is set when
            member j's value for axis i (numbered as in this interpolant)
            lies outside its domain or is NaN.

        Raises
        ------
        OutOfDomainError
            If a fixed value lies outside its domain and ``return_status``
            is False.
        """
        axes, raw = self._resolve_fixed(fixed)
        xp = array_namespace(*raw)
        columns = [xp.asarray(v, dtype=xp.float64) for v in raw]
        if any(c.ndim > 1 for c in columns):
            raise ValueError("Each fixed value must be a scalar or a 1D array with one value per member.")
        lengths = {c.shape[0] for c in columns if c.ndim == 1}
        if not lengths:
            raise ValueError("precontract_batch() needs at least one array of values; "
                             "use precontract() for a single set of values.")
        if len(lengths) > 1:
            raise ValueError(f"The arrays of fixed values have different lengths: {sorted(lengths)}.")
        n_members = lengths.pop()
        values = xp.stack([xp.broadcast_to(c, (n_members,)) for c in columns], axis=1)

        _, a, b = self._table_on(xp)
        tables, status = self._precontract_tables(xp, axes, values)
        live = [i for i in range(self.ndim) if i not in axes]
        batch = ChebyshevBatch(
            tables, a[live], b[live],
            axis_names=[self._axis_names[i] for i in live],
            output_names=self._output_names,
            domain_tol=self._domain_tol,
        )
        if return_status:
            return batch, status
        bad = status != 0
        if bool(xp.any(bad)):
            first = int(xp.flatnonzero(bad)[0])
            x = np.full(self.ndim, np.nan)
            x[axes] = to_numpy(values[first])
            raise OutOfDomainError(
                f"{int(bad.sum())} of {n_members} members have fixed values outside the interpolation "
                f"domain; the first is member {first}: "
                + self._domain_message(x, int(status[first])) + "."
            )
        return batch

    def _precontract_tables(self, xp, axes, values, full=False):
        """
        Per-member tables with ``axes`` fixed at the rows of ``values``.

        The table with the fixed axes moved to the front is cached per set of
        axes (and device), so repeated precontractions skip the transpose.
        This costs one extra copy of the table per set of axes in use.
        """
        key = (xp.__name__, tuple(axes), full)
        if key not in self._fixed_first:
            coeffs, a, b = (self._coeffs_full, self._a, self._b) if full else self._table_on(xp)
            live = [i for i in range(self.ndim) if i not in axes]
            order = list(axes) + live
            table = xp.ascontiguousarray(xp.transpose(coeffs, order + [self.ndim]))
            self._fixed_first[key] = (table, a[order], b[order])
        table, a, b = self._fixed_first[key]
        return precontract_tables(table, a, b, range(len(axes)), values,
                                  tol=self._domain_tol, status_axes=axes)

    def _resolve_fixed(self, fixed):
        """Turn a mapping of axis name/index -> value into (axis indices, values)."""
        if not hasattr(fixed, "items"):
            raise TypeError("fixed must be a mapping from axis name (or index) to value.")
        axes, values = [], []
        for key, value in fixed.items():
            ax = self._axis_position(key)
            if ax in axes:
                raise ValueError(f"Axis {self._axis_names[ax]!r} is fixed more than once.")
            axes.append(ax)
            values.append(value)
        if not axes:
            raise ValueError("fixed must name at least one axis.")
        if len(axes) == self.ndim:
            raise ValueError("Fixing every axis leaves nothing to interpolate; use evaluate() instead.")
        return axes, values

    def _axis_position(self, key):
        """Index of the axis named `key`, or `key` itself if it is a valid integer index."""
        if isinstance(key, str):
            try:
                return self._axis_names.index(key)
            except ValueError:
                raise KeyError(f"No axis named {key!r}; the axes are {self._axis_names}.") from None
        if isinstance(key, numbers.Integral) and not isinstance(key, bool):
            if not -self.ndim <= key < self.ndim:
                raise KeyError(f"Axis index {key} out of range for {self.ndim} axes.")
            return int(key) % self.ndim
        raise TypeError(f"Axes are given by name or integer index, not {key!r}.")

    # ------------------------------------------------------------ derivatives

    def derivative(self, orders):
        """
        A partial derivative of the interpolant, as a new interpolant.

        Differentiation happens in coefficient space, so it is exact for the
        interpolating polynomial. The result composes with everything else:
        it can be evaluated, precontracted, optimized and differentiated
        again, and :meth:`values_at_nodes` gives its values on the sampling
        grid.

        Parameters
        ----------
        orders : sequence of int or mapping
            The derivative order per axis, either one non-negative integer
            per axis in axis order, e.g. ``(0, 1, 2)``, or a mapping from axis
            name (or index) to order, e.g. ``{"u": 1, "e": 1}``, with
            unlisted axes taken as 0.

        Returns
        -------
        ChebyshevInterpolantND
            The derivative with respect to the physical coordinates (the
            factor ``2 / (b - a)`` per order is included), with the same
            axes, outputs and domain. It is built from this interpolant's
            truncated series (so it is the exact derivative of what
            :meth:`evaluate` computes), and also from the full series if
            that is still held.

        Notes
        -----
        Each derivative loses roughly a factor N^2 of relative accuracy, for
        N nodes on the differentiated axis, because it amplifies the
        rounding noise in the coefficients. The error estimate therefore has
        three parts:

        - the trailing coefficients of the derivative itself;
        - an exact bound on what this interpolant's truncation contributes;
        - a rounding term, ``eps * max|c| * prod((N^2 * 2/(b - a))**order)``
          over the differentiated axes.

        The rounding term is deliberately conservative. It grows with N, so
        for high derivatives more nodes eventually make the result worse.
        Choose node counts for the highest derivative you need, and check
        the estimate.
        """
        orders = self._parse_orders(orders)
        compact = self._differentiate(self._coeffs, orders)
        full = None if self._coeffs_full is None else self._differentiate(self._coeffs_full, orders)

        # Differentiating removes the top `order` coefficients along each axis
        keep = tuple(slice(max(k - o, 1)) for k, o in zip(self._kmax, orders))
        if full is not None:
            meaningful = tuple(slice(max(n - o, 1)) for n, o in zip(self._grid_shape, orders))
            interp_error = _trailing_error_estimate(np.abs(full[meaningful]))
            padded = np.zeros_like(full)
            padded[tuple(slice(k) for k in self._kmax)] = compact
            # Exact bound on what the truncation contributes to the derivative
            truncation = np.abs(full - padded).reshape(-1, full.shape[-1]).sum(axis=0)
        else:
            interp_error = _trailing_error_estimate(np.abs(compact[keep]))
            # Without the full series, scale the discarded sum by Markov's
            # inequality, |p'| <= n^2 max|p| on [-1, 1] for degree n
            factor = 1.0
            for k, o, a, b in zip(self._kmax, orders, self._a, self._b):
                factor *= (max(k - 1, 1) ** 2 * 2.0 / (b - a)) ** o
            truncation = self._discarded * factor

        # Rounding noise in the coefficients, amplified by the same Markov
        # factor; it dominates once a function is resolved to rounding level
        sizes = self._grid_shape if full is not None else self._kmax
        amplification = 1.0
        for n, o, a, b in zip(sizes, orders, self._a, self._b):
            amplification *= (max(n - 1, 1) ** 2 * 2.0 / (b - a)) ** o
        rounding = np.finfo(np.float64).eps * self._scale * amplification if any(orders) else 0.0

        return ChebyshevInterpolantND._from_tables(
            compact[keep], full, self._a, self._b,
            nodes=self._nodes,
            axis_names=self._axis_names,
            output_names=self._output_names,
            domain_tol=self._domain_tol,
            abs_error=interp_error + truncation + rounding,
            grid_shape=self._grid_shape,
        )

    def values_at_nodes(self):
        """
        Values of the full series on the interpolant's own node grid.

        For an interpolant built from samples, this reproduces the samples
        (to rounding). For a derivative, it gives the derivative's values
        at the nodes where the original function was sampled. That is the
        build workflow for higher-order tables: fit F0, take
        ``F0.derivative(...).values_at_nodes()``, combine with other values
        at the same nodes (see :meth:`axis_nodes`), and fit the result.

        Uses the full (untruncated) series and one inverse DCT.

        Returns
        -------
        np.ndarray
            Shape ``grid_shape`` for a single function, ``grid_shape + (F,)``
            with a function axis, ordered like ``np.meshgrid(*axis_nodes(),
            indexing='ij')``.

        Raises
        ------
        RuntimeError
            If the full tensor was dropped by ``optimize(keep_full=False)``.
        """
        if self._coeffs_full is None:
            raise RuntimeError(
                "values_at_nodes() needs the full coefficient tensor, which was dropped by "
                "optimize(keep_full=False)."
            )
        vals = chebyshev_values_at_nodes_nd(self._coeffs_full, nodes=self._nodes, axes=range(self.ndim))
        return vals[..., 0] if self._single else vals

    def axis_nodes(self):
        """
        The node coordinates along each axis.

        Returns
        -------
        tuple of np.ndarray
            For each axis, its ``grid_shape`` nodes in the order the node
            functions return them (descending), so that
            ``np.meshgrid(*interp.axis_nodes(), indexing='ij')`` is the
            sampling grid.
        """
        node_fn = chebyshev_nodes if self._nodes == "Chebyshev" else chebyshev_gauss_lobatto_nodes
        return tuple(node_fn(n, a, b) for n, a, b in zip(self._grid_shape, self._a, self._b))

    def _differentiate(self, coeffs, orders):
        """Differentiate a coefficient tensor (trailing output axis included) by `orders`."""
        for ax, order in enumerate(orders):
            if order:
                scale = (2.0 / (self._b[ax] - self._a[ax])) ** order
                coeffs = differentiate_coefficients(coeffs, ax, order) * scale
        return np.ascontiguousarray(coeffs)

    def _parse_orders(self, orders):
        """Turn a sequence or mapping of derivative orders into one int per axis."""
        if hasattr(orders, "items"):
            parsed = [0] * self.ndim
            seen = set()
            for key, order in orders.items():
                ax = self._axis_position(key)
                if ax in seen:
                    raise ValueError(f"Axis {self._axis_names[ax]!r} appears more than once in orders.")
                seen.add(ax)
                parsed[ax] = order
        else:
            parsed = list(orders)
            if len(parsed) != self.ndim:
                raise ValueError(
                    f"orders needs one entry per axis {self._axis_names}; got {len(parsed)}."
                )
        for order in parsed:
            if not isinstance(order, numbers.Integral) or isinstance(order, bool) or order < 0:
                raise ValueError(f"Derivative orders must be non-negative integers, got {order!r}.")
        return tuple(int(o) for o in parsed)

    # ----------------------------------------------------- coefficients, errors

    def get_coefficients(self, full=False):
        """
        Get the Chebyshev coefficients of the interpolant.

        Parameters
        ----------
        full : bool, optional
            If False (default), return the truncated block used for
            evaluation, shape ``kmax``. If True, return the full tensor,
            shape ``grid_shape``.

        Returns
        -------
        np.ndarray
            The coefficient tensor, with the function axis last when there
            is one.
        """
        if full:
            if self._coeffs_full is None:
                raise RuntimeError("The full coefficient tensor was dropped by optimize(keep_full=False).")
            coeffs = self._coeffs_full
        else:
            coeffs = self._coeffs
        return coeffs[..., 0] if self._single else coeffs

    def absolute_error(self):
        """
        Estimate the absolute error of the interpolant.

        The estimate is the interpolation error, judged from the trailing
        coefficients of the full tensor along each axis, plus the sum of
        the magnitudes of any coefficients removed by :meth:`optimize`
        (an upper bound on the error the truncation introduces).

        Returns
        -------
        float or np.ndarray
            Absolute error estimate; an array of shape (F,), one per
            function, when there is a function axis.
        """
        err = self._interp_error + self._discarded
        return float(err[0]) if self._single else err

    def relative_error(self):
        """
        Estimate the relative error: :meth:`absolute_error` divided by the
        largest coefficient magnitude, per function.

        Returns
        -------
        float or np.ndarray
            Relative error estimate; an array of shape (F,) when there is a
            function axis.
        """
        err = self._interp_error + self._discarded
        safe = np.where(self._scale > 0, self._scale, 1.0)
        rel = np.where(self._scale > 0, err / safe, 0.0)
        return float(rel[0]) if self._single else rel

    # ------------------------------------------------------------- truncation

    def optimize(self, rel_tol=1e-8, keep_full=True):
        """
        Truncate the series by greedily removing trailing coefficient slices
        while the sum of the removed magnitudes stays below
        ``rel_tol`` times the largest coefficient magnitude. With a function
        axis, one truncation is shared by all functions and the bound holds
        for each function relative to its own largest coefficient.

        Truncation always starts from the full tensor, so ``optimize`` can
        be called again with a different tolerance.

        Parameters
        ----------
        rel_tol : float, optional
            Allowed relative error of the truncation, in (0, 1).
        keep_full : bool, optional
            If False, drop the full coefficient tensor to save memory. Later
            calls to ``optimize`` are then impossible.
        """
        if self._coeffs_full is None:
            raise RuntimeError(
                "optimize() needs the full coefficient tensor, which was dropped by "
                "an earlier optimize(keep_full=False)."
            )
        if not 0.0 < rel_tol < 1.0:
            raise ValueError(f"rel_tol must lie in (0, 1), got {rel_tol}.")

        abs_c = np.abs(self._coeffs_full)
        n_out = abs_c.shape[-1]
        # Normalise each output by its own scale; the greedy pruner then
        # bounds every output's discarded sum by rel_tol times its scale
        scale = np.where(self._scale > 0, self._scale, 1.0)
        magnitude = (abs_c / scale).max(axis=-1)
        kmax, _ = prune_trailing_slices_greedy_nd(magnitude, delta=rel_tol, scale=1.0)
        kmax = tuple(max(1, int(k)) for k in kmax)

        kept = abs_c[tuple(slice(k) for k in kmax)].reshape(-1, n_out).sum(axis=0)
        total = abs_c.reshape(-1, n_out).sum(axis=0)
        self._discarded = np.maximum(total - kept, 0.0)
        self._set_truncation(kmax)
        if not keep_full:
            self._coeffs_full = None

    def effective_grid(self):
        """
        Get the effective grid size after truncation.

        Returns
        -------
        tuple of int
            Per-axis truncation counts (``kmax``).
        """
        return self._kmax

    def __repr__(self):
        return (
            f"ChebyshevInterpolantND(axes={self._axis_names}, grid_shape={self._grid_shape}, "
            f"kmax={self._kmax}, outputs={self.n_outputs}, nodes={self._nodes!r}, "
            f"dtype={self.dtype})"
        )


def _parse_domains(domains):
    """Return the lower and upper bounds as float64 arrays, checking each pair."""
    pairs = list(domains)
    if not pairs:
        raise ValueError("domains must contain at least one (lower, upper) pair.")
    a = np.empty(len(pairs))
    b = np.empty(len(pairs))
    for i, pair in enumerate(pairs):
        try:
            lo, hi = pair
        except (TypeError, ValueError):
            raise ValueError(f"domains[{i}] must be a (lower, upper) pair, got {pair!r}.") from None
        lo, hi = float(lo), float(hi)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(f"domains[{i}] = {pair!r} must be finite.")
        if not lo < hi:
            raise ValueError(f"domains[{i}] = {pair!r}: the lower bound must be smaller than the upper bound.")
        a[i], b[i] = lo, hi
    return a, b


def _parse_names(names, count, prefix, argname):
    """Return `count` unique names, defaulting to prefix0, prefix1, ..."""
    if names is None:
        return tuple(f"{prefix}{i}" for i in range(count))
    names = tuple(str(n) for n in names)
    if len(names) != count:
        raise ValueError(f"{argname} has {len(names)} entries but {count} are needed.")
    if len(set(names)) != len(names):
        raise ValueError(f"{argname} must be unique, got {names}.")
    return names


def _trailing_error_estimate(abs_c):
    """
    Per-output error estimate from the trailing coefficients along each axis.

    ``abs_c`` has shape (N_1, ..., N_d, F). For every axis with at least two
    coefficients, the last and second-to-last slices are scaled by N and
    N - 1 respectively (the convention of the 2D-4D classes), and the
    largest result over all axes is taken.
    """
    d = abs_c.ndim - 1
    n_out = abs_c.shape[-1]
    err = np.zeros(n_out)
    for ax in range(d):
        n = abs_c.shape[ax]
        if n < 2:
            continue
        c = np.moveaxis(abs_c, ax, 0)
        last = c[-1].reshape(-1, n_out).max(axis=0) * n
        second = c[-2].reshape(-1, n_out).max(axis=0) * (n - 1)
        err = np.maximum(err, np.maximum(last, second))
    return err
