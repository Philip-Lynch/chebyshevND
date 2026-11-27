"""n-dimensional Chebyshev interpolation."""

import numbers

import numpy as np

from .backend import array_namespace, resolve_device, to_numpy
from .batch import ChebyshevBatch
from .batched import precontract_blocks, precontract_tables
from .batched import evaluate_batch as _evaluate_batch
from .batched import value_and_grad_batch as _value_and_grad_batch
from .blocks import (
    BlockedCoeffs,
    group_scale,
    outside_box_sum,
    pack_blocks,
    parse_groups,
    partition_outputs,
)
from .derivatives import differentiate_coefficients
from .errors import OutOfDomainError, describe_out_of_domain
from .kernels import (
    RawBlockedTable,
    RawTable,
    blocked_sizes,
    evaluate_blocks_point,
    evaluate_point,
    value_and_grad_blocks_point,
    value_and_grad_point,
)
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
    :meth:`value_and_grad` and :meth:`value_and_grad_batch` add the gradient
    at little extra cost, and :meth:`raw` exposes the table to the user's own
    Numba code (see :func:`chebyshevnd.eval_into`).

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
        self._scale_value = abs_c.reshape(-1, n_out).max(axis=0)
        self._interp_error = _trailing_error_estimate(abs_c)
        self._discarded = np.zeros(n_out)
        self._discarded_l2 = np.zeros(n_out)
        self._truncation = None
        self._x = np.empty(d)
        self._set_truncation(self._grid_shape)

    # ------------------------------------------------------------------ setup

    @classmethod
    def _from_tables(cls, coeffs, coeffs_full, a, b, *, nodes, axis_names, output_names,
                     domain_tol, abs_error, grid_shape, discarded=None, scale=None,
                     discarded_l2=None, truncation=None, dtype=None, flat=None):
        """
        Build an interpolant directly from coefficient tensors, e.g. the result
        of a precontraction, or a table read from a file.

        ``coeffs`` (the evaluation tensor) and ``coeffs_full`` (or None) carry
        a trailing output axis; for a table stored in blocks, ``coeffs`` is
        instead a sequence of ``(block, outputs)`` pairs (see
        :mod:`chebyshevnd.blocks`). ``output_names`` is empty for a single
        function. ``abs_error`` (one value per output) becomes the error
        estimate: it is inherited from the parent, whose bound still holds.
        ``discarded`` (the truncation part of the estimate, default zero),
        ``scale`` (the largest coefficient magnitude per output, default
        computed from the coefficients), ``discarded_l2`` (default zero) and
        ``truncation`` (the settings of the ``optimize`` that made the table)
        restore a saved table exactly. ``dtype`` is needed only for a blocked
        table with no blocks. ``flat``, for a blocked table, is a buffer that
        already holds its blocks one after the other (see ``pack_blocks``).
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
        n_out = max(len(self._output_names), 1)
        if isinstance(coeffs, np.ndarray):
            blocks = [(coeffs, np.arange(n_out, dtype=np.int64))]
        else:
            blocks = [(np.asarray(c), np.asarray(o, dtype=np.int64)) for c, o in coeffs]
        if blocks:
            dtype = blocks[0][0].dtype
        elif self._coeffs_full is not None:
            dtype = self._coeffs_full.dtype
        # Without a given scale, it is computed on first use (see _scale): a
        # pass over the coefficients that precontraction need not pay for
        self._scale_value = None if scale is None else np.array(scale, dtype=np.float64)
        self._interp_error = np.array(abs_error, dtype=np.float64)
        self._discarded = np.zeros(n_out) if discarded is None else np.array(discarded, dtype=np.float64)
        self._discarded_l2 = (np.zeros(n_out) if discarded_l2 is None
                              else np.array(discarded_l2, dtype=np.float64))
        self._truncation = truncation
        self._x = np.empty(self._a.shape[0])
        self._set_blocks(blocks, n_out, dtype, flat=flat)
        return self

    def _set_truncation(self, kmax):
        """Keep the leading ``kmax`` block of the full tensor as the compact evaluation tensor (A2)."""
        self._set_compact(self._coeffs_full[tuple(slice(int(k)) for k in kmax)])

    def _set_compact(self, coeffs):
        """Make ``coeffs`` (trailing output axis included) the evaluation tensor."""
        n_out = coeffs.shape[-1]
        self._set_blocks([(coeffs, np.arange(n_out, dtype=np.int64))], n_out, coeffs.dtype)

    def _set_blocks(self, blocks, n_out, dtype, flat=None):
        """
        Make ``blocks`` the evaluation tables: ``(coeffs, outputs)`` pairs,
        each ``coeffs`` of shape (k_1, ..., k_d, F_b) with the output index
        of each of its columns in ``outputs``.

        One block holding every output in order is a *dense* table, stored
        and evaluated as before blocks existed (A2). Anything else is a
        *blocked* table: every block becomes a view into one flat array, the
        layout of the blocked kernels, and outputs in no block are zero.
        ``flat``, if given, already holds the blocks one after the other and
        becomes that array, without a copy.
        """
        d = len(self._grid_shape)
        self._n_out = int(n_out)
        self._dtype = np.dtype(dtype)
        blocks = [(c, np.asarray(o, dtype=np.int64)) for c, o in blocks]
        self._dense = (len(blocks) == 1 and blocks[0][1].shape[0] == self._n_out
                       and bool(np.all(blocks[0][1] == np.arange(self._n_out))))
        if self._dense:
            self._coeffs = np.ascontiguousarray(blocks[0][0])
            self._blocks = ((self._coeffs, blocks[0][1]),)
            self._packed = None
            self._kmax = tuple(int(k) for k in self._coeffs.shape[:-1])
            self._flat = self._coeffs.reshape(-1)
            self._shape = np.array(self._kmax, dtype=np.int64)
            rest = self._flat.shape[0] // self._kmax[0]
            self._scratch = np.empty(len(self._kmax) + max(self._kmax))
            self._work = np.empty((2, rest), dtype=self._dtype)
        else:
            packed = pack_blocks(blocks, d, self._dtype, flat=flat)
            flat, offsets = packed[0], packed[1]
            self._blocks = tuple((flat[offsets[i]:offsets[i + 1]].reshape(c.shape), o)
                                 for i, (c, o) in enumerate(blocks))
            self._packed = packed
            self._coeffs = self._flat = self._shape = None
            self._kmax = tuple(max((int(c.shape[i]) for c, _ in blocks), default=0) for i in range(d))
            m, rest = blocked_sizes(packed[2], packed[4])
            # One workspace for both kernels: the gradient kernel needs
            # d + 2 max(k) scratch and 3 (d + 1) work rows
            self._scratch = np.empty(d + 2 * m)
            self._work = np.empty((3 * (d + 1), rest), dtype=self._dtype)
        self._out = np.empty(self._n_out, dtype=self._dtype)
        # Copies of the table on other devices, and copies with a set of
        # fixed axes moved to the front (for precontraction), made on first use
        self._device_tables = {}
        self._fixed_first = {}
        # Buffers for value_and_grad, allocated on first use
        self._grad_buffers = None

    @property
    def _scale(self):
        """
        The largest coefficient magnitude per output, from the full tensor
        if it is held, otherwise from the stored blocks (zero for outputs in
        none). Computed on first use. ``optimize`` reads it before it
        replaces the blocks or drops the full tensor, so the value never
        depends on when it is first needed.
        """
        if self._scale_value is None:
            if self._coeffs_full is not None:
                scale = np.abs(self._coeffs_full).reshape(-1, self._n_out).max(axis=0)
            else:
                scale = np.zeros(self._n_out)
                for c, o in self._blocks:
                    scale[o] = np.abs(c).reshape(-1, len(o)).max(axis=0)
            self._scale_value = scale
        return self._scale_value

    @property
    def _untruncated(self):
        """True if the evaluation tensor is the whole full tensor (never truncated)."""
        return self._dense and self._kmax == self._grid_shape

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
        """
        Number of coefficients used per axis after truncation, (k_1, ..., k_d).

        For a table stored in blocks (after ``optimize(per_output=True)``),
        the largest over the blocks: the shape of :meth:`get_coefficients`.
        :attr:`output_kmax` gives each output's own.
        """
        return self._kmax

    @property
    def output_kmax(self):
        """
        Number of coefficients stored per axis for each function.

        Returns
        -------
        np.ndarray
            int64, shape (F, d): row j is the box of the block that holds
            function j, and a row of zeros for a function stored in no block
            (one that evaluates to exactly zero). For a dense table, every
            row is :attr:`kmax`.
        """
        out = np.zeros((self._n_out, self.ndim), dtype=np.int64)
        for c, o in self._blocks:
            out[o] = c.shape[:-1]
        return out

    @property
    def blocks(self):
        """
        The stored coefficients, as ``(outputs, coeffs)`` pairs, one per block.

        A dense table is one block holding every function. After
        ``optimize(per_output=True)``, block b holds the functions
        ``outputs`` (int64 positions in evaluation results, ascending) in a
        dense tensor ``coeffs`` of shape (k_1, ..., k_d, F_b), trailing
        function axis included even for a single function. Functions in no
        block evaluate to exactly zero, as do their derivatives.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            Read-only views.
        """
        return tuple((_read_only(o), _read_only(c)) for c, o in self._blocks)

    @property
    def discarded_l2(self):
        """
        The root-sum-square of the coefficients removed by :meth:`optimize`.

        ``sqrt(sum of |discarded coefficient|^2)``, per function: the
        quantity ``optimize(norm="l2")`` keeps below its budget. Unlike the
        sum of magnitudes in :meth:`absolute_error`, it is not a bound on
        the error, but when the removed coefficients are noise it is a far
        better estimate of its typical size. Zero for a table that was
        never optimized. A precontraction inherits its parent's figure (its
        error at a point is the parent's error there); a derivative gets the
        figure of its own discarded coefficients if the full tensor is held,
        and otherwise its parent's, scaled like the truncation bound.

        Returns
        -------
        float or np.ndarray
            An array of shape (F,) when there is a function axis.
        """
        return float(self._discarded_l2[0]) if self._single else self._discarded_l2.copy()

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
        return self._n_out

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
        return self._dtype

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
        if self._dense:
            status = evaluate_point(self._flat, self._shape, self._bounds, x, self._domain_tol,
                                    self._scratch, self._work, self._out)
        else:
            status = evaluate_blocks_point(*self._packed, self._bounds, x, self._domain_tol,
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
        xp, pts = self._batch_points(points)
        coeffs, a, b = self._table_on(xp)
        values, status = _evaluate_batch(coeffs, a, b, pts, tol=self._domain_tol)
        if self._single:
            values = values[:, 0]
        if return_status:
            return values, status
        self._check_batch_status(xp, pts, status)
        return values

    def value_and_grad(self, *xs):
        """
        Evaluate the interpolant and its gradient at a single point.

        The gradient along every axis comes from the same pass over the
        coefficients as the value, at about 1.0-1.4x the cost of
        :meth:`evaluate`. It is the exact gradient of the interpolating
        polynomial (the same as :meth:`derivative` would give), so a vector
        field built from it, e.g. Hamilton's equations for an interpolated
        Hamiltonian, is exactly the one generated by the interpolant.

        Parameters
        ----------
        *xs : float
            One coordinate per axis, in axis order.

        Returns
        -------
        value : float, complex or np.ndarray
            As for :meth:`evaluate`.
        grad : np.ndarray
            Derivatives with respect to the coordinates (in the domain's
            units): shape (d,) for a single function, (d, F) with a function
            axis, so ``grad[i]`` is the derivative along axis i.

        Raises
        ------
        OutOfDomainError
            If a coordinate lies outside its domain (beyond ``domain_tol``)
            or is NaN.

        See Also
        --------
        raw : the same evaluation from inside the user's own Numba code.
        """
        x = self._point(xs, "value_and_grad")
        if self._grad_buffers is None:
            d = self.ndim
            if self._dense:
                self._grad_buffers = (
                    np.empty(d + 2 * max(self._kmax)),
                    np.empty((2 * (d + 1), self._work.shape[1]), dtype=self.dtype),
                    np.empty((d + 1, self.n_outputs), dtype=self.dtype),
                )
            else:
                # The blocked workspace already serves both kernels
                self._grad_buffers = (self._scratch, self._work,
                                      np.empty((d + 1, self.n_outputs), dtype=self.dtype))
        scratch, work, out = self._grad_buffers
        if self._dense:
            status = value_and_grad_point(self._flat, self._shape, self._bounds, x, self._domain_tol,
                                          scratch, work, out)
        else:
            status = value_and_grad_blocks_point(*self._packed, self._bounds, x, self._domain_tol,
                                                 scratch, work, out)
        if status:
            raise OutOfDomainError(
                "Point outside the interpolation domain: " + self._domain_message(x, status) + "."
            )
        grad = out[1:, 0].copy() if self._single else out[1:].copy()
        return self._result(out[0]), grad

    def value_and_grad_batch(self, points, return_status=False):
        """
        Evaluate the interpolant and its gradient at many points at once.

        The batch version of :meth:`value_and_grad`, on the CPU (NumPy) or
        the GPU (CuPy), like :meth:`evaluate_batch`.

        Parameters
        ----------
        points : array, shape (B, d)
            As for :meth:`evaluate_batch`.
        return_status : bool, optional
            If True, return the per-point status instead of raising for
            points outside the domain.

        Returns
        -------
        values : array
            Shape (B,) for a single function, (B, F) with a function axis.
        grads : array
            Shape (B, d) for a single function, (B, d, F) with a function
            axis: ``grads[j, i]`` is the derivative along axis i at point j.
        status : int64 array, shape (B,)
            Only if ``return_status``; as for :meth:`evaluate_batch`.

        Raises
        ------
        OutOfDomainError
            If any point is outside the domain and ``return_status`` is False.
        """
        xp, pts = self._batch_points(points)
        coeffs, a, b = self._table_on(xp)
        values, grads, status = _value_and_grad_batch(coeffs, a, b, pts, tol=self._domain_tol)
        if self._single:
            values, grads = values[:, 0], grads[..., 0]
        if return_status:
            return values, grads, status
        self._check_batch_status(xp, pts, status)
        return values, grads

    def raw(self, blocked=None):
        """
        The table as plain arrays, for evaluation inside the user's own Numba code.

        Precontract first, so the table is as small as possible. Then pass
        its fields into an ``@njit`` function and evaluate it there with
        :func:`chebyshevnd.eval_into` or :func:`chebyshevnd.eval_grad_into`,
        e.g. once per call of an ODE right-hand side::

            from chebyshevnd import eval_grad_into

            table = H.precontract({"nu": 0.2}).raw()
            ws = table.workspace()
            out = np.empty((table.ndim + 1, table.n_outputs), dtype=table.dtype)

            @njit
            def rhs(t, y, coeffs, shape, bounds, tol, scratch, work, out):
                status = eval_grad_into(coeffs, shape, bounds, tol, y, scratch, work, out)
                dH = out[1:, 0]  # derivative along each axis
                ...

            solve_ivp(lambda t, y: rhs(t, y, *table, *ws, out), ...)

        Pass the fields separately, as here, rather than the tuples: Numba
        handles array arguments in compiled code, but tuples in Python,
        which costs about 1.5 us more per call from Python.

        A table stored in blocks (after ``optimize(per_output=True)``)
        gives a :class:`~chebyshevnd.RawBlockedTable` instead, for
        :func:`chebyshevnd.eval_blocked_into` and
        :func:`chebyshevnd.eval_grad_blocked_into`.

        Parameters
        ----------
        blocked : bool, optional
            None (default): a ``RawTable`` for a dense table and a
            ``RawBlockedTable`` for a blocked one. True: a
            ``RawBlockedTable`` for either (one block for a dense table), so
            that one ``@njit`` function written for the blocked entry points
            serves every table. False: a ``RawTable``, for dense tables only.

        Returns
        -------
        RawTable or RawBlockedTable
            The compact (truncated) coefficients, the number of coefficients
            per axis, the domain bounds and ``domain_tol``, as read-only
            views. Later calls to :meth:`optimize` do not change it.

        Raises
        ------
        ValueError
            For ``blocked=False`` on a table stored in blocks.
        """
        if blocked is None:
            blocked = not self._dense
        if not blocked:
            if not self._dense:
                raise ValueError("This table is stored in blocks (see optimize(per_output=True)), so it "
                                 "has no RawTable; use raw() and eval_blocked_into.")
            return RawTable(_read_only(self._flat), _read_only(self._shape), _read_only(self._bounds),
                            self._domain_tol)
        packed = self._packed if self._packed is not None else pack_blocks(self._blocks, self.ndim,
                                                                           self._dtype)
        return RawBlockedTable(*(_read_only(a) for a in packed), self._n_out, _read_only(self._bounds),
                               self._domain_tol)

    def _batch_points(self, points):
        """The array module of `points`, and `points` checked as a (B, d) float64 array."""
        xp = array_namespace(points)
        pts = xp.asarray(points, dtype=xp.float64)
        if pts.ndim == 1 and self.ndim == 1:
            pts = pts[:, None]
        if pts.ndim != 2 or pts.shape[1] != self.ndim:
            raise ValueError(
                f"points must have shape (B, {self.ndim}) for axes {self._axis_names}; "
                f"got {tuple(pts.shape)}."
            )
        return xp, pts

    def _check_batch_status(self, xp, pts, status):
        """Raise OutOfDomainError describing the first point with a nonzero status, if any."""
        bad = status != 0
        if bool(xp.any(bad)):
            first = int(xp.flatnonzero(bad)[0])
            raise OutOfDomainError(
                f"{int(bad.sum())} of {pts.shape[0]} points are outside the interpolation domain; "
                f"the first is point {first}: "
                + self._domain_message(to_numpy(pts[first]), int(status[first])) + "."
            )

    def to_device(self, device="cuda"):
        """
        Copy the table to a device now, rather than on first use there.

        The interpolant always keeps its coefficients on the host (for
        :meth:`evaluate` and Numba). This copies them to the GPU once, so
        that the first batched call from an ODE loop does not pay for it.

        Parameters
        ----------
        device : {"cuda", "cpu"}
            "cuda" copies to CuPy's current device; "cpu" does nothing.

        Returns
        -------
        ChebyshevInterpolantND
            This interpolant.
        """
        xp = resolve_device(device)
        if xp is not np:
            self._table_on(xp)
        return self

    def _table_on(self, xp):
        """
        The compact coefficients and bounds as arrays of module `xp`, cached
        per device. For a table stored in blocks, the coefficients are a
        :class:`~chebyshevnd.blocks.BlockedCoeffs`.
        """
        if xp is np and self._dense:
            return self._coeffs, self._a, self._b
        key = xp.__name__
        if key not in self._device_tables:
            if self._dense:
                table = xp.asarray(self._coeffs)
            else:
                table = BlockedCoeffs([c for c, _ in self._blocks], [o for _, o in self._blocks],
                                      self._n_out, self._dtype, box=self._kmax).to(xp)
            self._device_tables[key] = (table, xp.asarray(self._a), xp.asarray(self._b))
        return self._device_tables[key]

    def _domain_message(self, x, status):
        """Describe which coordinates of `x` are outside the domain, from a status bitmask."""
        return describe_out_of_domain(self._axis_names, self._a, self._b, x, status)

    def _result(self, out):
        """Return the scalar for a single function, otherwise a copy of the array."""
        return out.item(0) if self._single else out.copy()

    def _point(self, xs, method="evaluate"):
        """Validate scalar coordinates and copy them into the instance's point buffer."""
        if len(xs) != self.ndim:
            raise TypeError(
                f"{method}() takes {self.ndim} coordinates {self._axis_names}, got {len(xs)}."
            )
        try:
            self._x[:] = xs
        except (TypeError, ValueError):
            raise TypeError(f"{method}() takes one scalar per axis.") from None
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
        flat = None
        if not self._dense:
            # Write the blocks of the result straight into the one array the
            # blocked kernels use, rather than copying them there afterwards
            shapes = [tuple(k for i, k in enumerate(c.shape[:-1]) if i not in axes) + (c.shape[-1],)
                      for c, _ in self._blocks]
            ends = np.cumsum([0] + [int(np.prod(shape)) for shape in shapes])
            flat = np.empty(int(ends[-1]), dtype=self._dtype)
            out = [flat[ends[i]:ends[i + 1]].reshape((1,) + shape) for i, shape in enumerate(shapes)]
        # The batched routine with a batch of one: a single GEMM, whatever
        # the number and position of the fixed axes
        tables, status = self._precontract_tables(np, axes, row, out=None if flat is None else out)
        if status[0]:
            x = np.full(self.ndim, np.nan)
            x[axes] = row[0]
            raise OutOfDomainError(
                "Fixed value outside the interpolation domain: "
                + self._domain_message(x, int(status[0])) + "."
            )
        if self._dense:
            coeffs = tables[0]
        else:
            coeffs = [(t[0], o) for t, o in zip(tables.blocks, tables.outputs)]
        if self._coeffs_full is None:
            full = None
        elif self._untruncated:
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
            discarded_l2=self._discarded_l2,
            truncation=self._truncation,
            dtype=self._dtype,
            flat=flat,
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
        xp, values = _member_values(raw)
        n_members = values.shape[0]

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

    def _precontract_tables(self, xp, axes, values, full=False, out=None):
        """
        Per-member tables with ``axes`` fixed at the rows of ``values``.

        The table with the fixed axes moved to the front is cached per set of
        axes (and device), so repeated precontractions skip the transpose.
        This costs one extra copy of the table per set of axes in use.
        For a blocked table, ``out`` may give the arrays to write each
        block's result into.
        """
        key = (xp.__name__, tuple(axes), full)
        if key not in self._fixed_first:
            coeffs, a, b = (self._coeffs_full, self._a, self._b) if full else self._table_on(xp)
            live = [i for i in range(self.ndim) if i not in axes]
            order = list(axes) + live
            perm = order + [self.ndim]
            if isinstance(coeffs, BlockedCoeffs):
                table = [xp.ascontiguousarray(xp.transpose(c, perm)) for c in coeffs.blocks]
            else:
                table = xp.ascontiguousarray(xp.transpose(coeffs, perm))
            self._fixed_first[key] = (table, a[order], b[order])
        table, a, b = self._fixed_first[key]
        if isinstance(table, list):
            # A blocked table: one GEMM per block, the outputs keep their blocks
            tables, status = precontract_blocks(table, a, b, len(axes), values, tol=self._domain_tol,
                                                status_axes=axes, out=out)
            box = tuple(k for i, k in enumerate(self._kmax) if i not in axes)
            return BlockedCoeffs(tables, self._table_on(xp)[0].outputs, self._n_out, self._dtype,
                                 box=box, members=values.shape[0], xp=xp), status
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

        See Also
        --------
        value_and_grad : the value and first derivatives at a point in one
            pass, without building derivative interpolants; for use at run
            time, e.g. in an ODE right-hand side.
        """
        orders = self._parse_orders(orders)
        derived = [(self._differentiate(c, orders), o) for c, o in self._blocks]
        full = None if self._coeffs_full is None else self._differentiate(self._coeffs_full, orders)

        # Differentiating removes the top `order` coefficients along each axis
        kept = [(c[tuple(slice(max(k - o, 1)) for k, o in zip(c.shape[:-1], orders))], idx)
                for c, idx in derived]
        n_out = self._n_out
        if full is not None:
            meaningful = tuple(slice(max(n - o, 1)) for n, o in zip(self._grid_shape, orders))
            interp_error = _trailing_error_estimate(np.abs(full[meaningful]))
            padded = np.zeros_like(full)
            for c, idx in derived:
                padded[tuple(slice(k) for k in c.shape[:-1]) + (idx,)] = c
            # Exact bound on what the truncation contributes to the derivative
            removed = np.abs(full - padded).reshape(-1, n_out)
            truncation = removed.sum(axis=0)
            discarded_l2 = np.sqrt((removed ** 2).sum(axis=0))
        else:
            interp_error = np.zeros(n_out)
            for c, idx in kept:
                interp_error[idx] = _trailing_error_estimate(np.abs(c))
            factor = self._markov_factor(orders)
            truncation = self._discarded * factor
            discarded_l2 = self._discarded_l2 * factor

        # Rounding noise in the coefficients, amplified by the same Markov
        # factor; it dominates once a function is resolved to rounding level
        sizes = self._grid_shape if full is not None else self._kmax
        amplification = 1.0
        for n, o, a, b in zip(sizes, orders, self._a, self._b):
            amplification *= (max(n - 1, 1) ** 2 * 2.0 / (b - a)) ** o
        rounding = np.finfo(np.float64).eps * self._scale * amplification if any(orders) else 0.0

        return ChebyshevInterpolantND._from_tables(
            kept[0][0] if self._dense else kept, full, self._a, self._b,
            nodes=self._nodes,
            axis_names=self._axis_names,
            output_names=self._output_names,
            domain_tol=self._domain_tol,
            abs_error=interp_error + truncation + rounding,
            grid_shape=self._grid_shape,
            discarded_l2=discarded_l2,
            truncation=self._truncation,
            dtype=self._dtype,
        )

    def check_derivatives(self, orders, points, step=None):
        """
        Compare an analytic derivative with finite differences of this interpolant.

        A routine sanity check for grid builds: the analytic derivative from
        :meth:`derivative` should agree with central finite differences of
        this interpolant, to the accuracy finite differences allow.

        Parameters
        ----------
        orders : sequence of int or mapping
            As for :meth:`derivative`. First derivatives, pure second
            derivatives and mixed second derivatives are supported.
        points : array-like, shape (n, d)
            Points inside the domain at which to compare.
        step : float, optional
            Finite-difference step as a fraction of each axis's width.
            Defaults to 1e-5 for first and 1e-4 for second derivatives,
            near the optimum for float64.

        Returns
        -------
        float or np.ndarray
            ``max |analytic - finite difference| / max |analytic|`` over the
            points (absolute if the derivative vanishes there), per function.
            For a well-resolved table, expect around 1e-10 for first and 1e-8
            for second derivatives (the limit of the finite differences);
            much larger values indicate a problem.
        """
        orders = self._parse_orders(orders)
        if sum(orders) not in (1, 2):
            raise ValueError("check_derivatives() supports first and second derivatives only.")
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if pts.shape[1] != self.ndim:
            raise ValueError(f"points must have shape (n, {self.ndim}); got {pts.shape}.")
        if step is None:
            step = 1e-5 if sum(orders) == 1 else 1e-4
        h = step * (self._b - self._a)

        axes = [ax for ax, o in enumerate(orders) for _ in range(o)]
        if len(axes) == 1 or axes[0] != axes[1]:
            # Product of central differences, one per axis
            stencil = [(1.0, [])]
            for ax in axes:
                stencil = [(w * s, shifts + [(ax, s)]) for w, shifts in stencil for s in (1.0, -1.0)]
            denominator = np.prod([2.0 * h[ax] for ax in axes])
        else:
            ax = axes[0]
            stencil = [(1.0, [(ax, 1.0)]), (-2.0, []), (1.0, [(ax, -1.0)])]
            denominator = h[ax] ** 2

        shifted = []
        for _, shifts in stencil:
            p = pts.copy()
            for ax, s in shifts:
                p[:, ax] += s * h[ax]
            shifted.append(p)
        # Evaluate without the domain check: stencils may step just outside
        values, _ = _evaluate_batch(self._table_on(np)[0], self._a, self._b, np.concatenate(shifted),
                                    tol=np.inf)
        values = values.reshape(len(stencil), len(pts), -1)
        finite_diff = sum(w * v for (w, _), v in zip(stencil, values)) / denominator

        analytic = self.derivative(orders).evaluate_batch(pts)
        analytic = analytic.reshape(len(pts), -1)
        scale = np.abs(analytic).max(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        discrepancy = np.abs(analytic - finite_diff).max(axis=0) / scale
        return float(discrepancy[0]) if self._single else discrepancy

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

    def _markov_factor(self, orders):
        """
        How much a derivative of order ``orders`` can amplify the discarded coefficients.

        The discarded coefficients have degree up to N_i - 1 on axis i (N_i
        nodes), so by Markov's inequality, ``|p'| <= n^2 max|p|`` on [-1, 1]
        for degree n, each order on axis i multiplies their bound by at most
        ``(N_i - 1)^2 * 2 / (b_i - a_i)``.
        """
        factor = 1.0
        for n, o, a, b in zip(self._grid_shape, orders, self._a, self._b):
            factor *= (max(n - 1, 1) ** 2 * 2.0 / (b - a)) ** o
        return factor

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
            evaluation, shape ``kmax``; for a table stored in blocks, a
            zero-filled dense copy of the blocks (see :attr:`blocks`). If
            True, return the full tensor, shape ``grid_shape``.

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
        elif self._dense:
            coeffs = self._coeffs
        else:
            coeffs = self._table_on(np)[0].to_dense()
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

        See Also
        --------
        error_by_axis : the estimate split by axis.
        """
        err = self._interp_error + self._discarded
        return float(err[0]) if self._single else err

    def error_by_axis(self, relative=False):
        """
        The trailing-coefficient error estimate, axis by axis.

        Shows which axis limits the accuracy, e.g. to choose the number of
        nodes per axis: an axis with an estimate far below the others can
        have fewer nodes, and the axis with the largest one needs more. The
        interpolation part of :meth:`absolute_error` is the largest entry
        (for an interpolant built from samples).

        The estimate is conservative on an axis where the function is an
        exact polynomial with only one or two nodes to spare: e.g. a
        quadratic sampled at 3 nodes, or an even quadratic at 4. Its top
        genuine coefficients then sit in the last slices and are reported
        as error. One or two more nodes on that axis show whether the
        series has really ended.

        Parameters
        ----------
        relative : bool, optional
            Divide by the largest coefficient magnitude of each function.

        Returns
        -------
        np.ndarray
            Shape (d,) for a single function, (d, F) with a function axis.
            Computed from the full coefficient tensor if it is held,
            otherwise from the truncated one.
        """
        if self._coeffs_full is not None or self._dense:
            coeffs = self._coeffs_full if self._coeffs_full is not None else self._coeffs
            err = _trailing_errors_by_axis(np.abs(coeffs))
        else:
            err = np.zeros((self.ndim, self._n_out))
            for c, idx in self._blocks:
                err[:, idx] = _trailing_errors_by_axis(np.abs(c))
        if relative:
            err = np.where(self._scale > 0, err / np.where(self._scale > 0, self._scale, 1.0), 0.0)
        return err[:, 0] if self._single else err

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

    def optimize(self, rel_tol=1e-8, keep_full=True, *, groups=None, norm="l1", per_output=False,
                 max_blocks=None):
        """
        Truncate the series by greedily removing trailing coefficient slices
        while the removed coefficients stay within a budget.

        By default, the budget is that the sum of the removed magnitudes stays
        below ``rel_tol`` times the largest coefficient magnitude. With a
        function axis, one truncation is shared by all functions and the
        bound holds for each function relative to its own largest
        coefficient.

        For tables of many functions of very different sizes (e.g. the
        Fourier harmonics of a few families of functions), three options
        remove far more:

        - ``groups`` measures each function against the largest coefficient
          of its group, so a negligible member of a family is not resolved
          to its own (noise) level;
        - ``norm="l2"`` budgets the root-sum-square of the removed
          coefficients instead of the sum of their magnitudes, which suits a
          tail of interpolation noise, where many tiny coefficients add up
          in magnitude but not in their typical effect;
        - ``per_output=True`` truncates each function on its own, and stores
          the functions in at most ``max_blocks`` dense blocks (see
          :attr:`blocks`). A function whose whole series fits in its budget
          is stored in no block and evaluates to exactly zero.

        Truncation always starts from the full tensor, so ``optimize`` can
        be called again with other settings.

        Parameters
        ----------
        rel_tol : float, optional
            Allowed relative error of the truncation, in (0, 1).
        keep_full : bool, optional
            If False, drop the full coefficient tensor to save memory. Later
            calls to ``optimize`` are then impossible.
        groups : sequence or mapping, optional
            The reference scale of each function's budget, ``rel_tol * S_g``,
            with S_g the largest coefficient magnitude over the functions of
            its group. A sequence of F hashable labels (one per function),
            or a mapping from label to a sequence of function names or
            indices that puts every function in exactly one group. By
            default every function is its own group.
        norm : {"l1", "l2"}, optional
            ``"l1"`` (default): the sum of the removed magnitudes stays below
            the budget, a rigorous bound on the error the truncation
            introduces anywhere in the domain. ``"l2"``: their root-sum-square
            does (see :attr:`discarded_l2`), an estimate of the typical error.
        per_output : bool, optional
            Truncate each function separately rather than all together.
        max_blocks : int, optional
            With ``per_output``, the most blocks to store the functions in:
            functions with similar truncations share a block, whose shape
            fits all of them. Fewer blocks store more coefficients (each
            block is a box) but cost less per evaluation in the batched
            NumPy/CuPy paths, which loop over blocks in Python. None
            (default) stores each function in a block of its own.

        Notes
        -----
        After truncation, :meth:`absolute_error` adds the sum of the removed
        magnitudes to the interpolation estimate, whatever ``norm`` is, and
        :attr:`discarded_l2` holds their root-sum-square. Both are computed
        from what is actually stored: a function's block may keep more of
        its coefficients than its own truncation would.
        """
        if self._coeffs_full is None:
            raise RuntimeError(
                "optimize() needs the full coefficient tensor, which was dropped by "
                "an earlier optimize(keep_full=False)."
            )
        if not 0.0 < rel_tol < 1.0:
            raise ValueError(f"rel_tol must lie in (0, 1), got {rel_tol}.")
        if norm not in ("l1", "l2"):
            raise ValueError(f"norm must be 'l1' or 'l2', got {norm!r}.")
        if max_blocks is not None:
            if not per_output:
                raise ValueError("max_blocks applies to per_output=True; without it the table is "
                                 "one block.")
            if (isinstance(max_blocks, bool) or not isinstance(max_blocks, numbers.Integral)
                    or max_blocks < 1):
                raise ValueError(f"max_blocks must be a positive integer or None, got {max_blocks!r}.")
            max_blocks = int(max_blocks)
        labels = parse_groups(groups, self._output_names, self._n_out)

        full = self._coeffs_full
        abs_c = np.abs(full)
        n_out = abs_c.shape[-1]
        squared = norm == "l2"
        delta = rel_tol ** 2 if squared else rel_tol
        # Each output's reference scale: its own largest coefficient, or its group's
        scale = self._scale if labels is None else group_scale(labels, self._scale)
        if not per_output:
            # Normalise each output by its scale; the greedy pruner then
            # bounds every output's discarded sum by rel_tol times its scale
            safe = np.where(scale > 0, scale, 1.0)
            magnitude = (abs_c / safe).max(axis=-1)
            if squared:
                magnitude = magnitude ** 2
            kmax, _ = prune_trailing_slices_greedy_nd(magnitude, delta=delta, scale=1.0)
            kmax = tuple(max(1, int(k)) for k in kmax)
            boxes = [(kmax, None)]
        else:
            own = np.zeros((n_out, self.ndim), dtype=np.int64)
            for j in range(n_out):
                if scale[j] == 0.0:
                    continue  # the whole group is zero: store nothing
                magnitude = abs_c[..., j] ** 2 if squared else abs_c[..., j]
                k, _ = prune_trailing_slices_greedy_nd(magnitude, delta=delta,
                                                       scale=float(scale[j]) ** (2 if squared else 1))
                if min(k) > 0:
                    own[j] = k
            stored = np.flatnonzero(own[:, 0] > 0)
            boxes = []
            for run in partition_outputs(own[stored], max_blocks):
                members = stored[run]
                boxes.append((tuple(int(k) for k in own[members].max(axis=0)), members))

        # What the stored boxes keep, per output: L1 as the total minus the
        # kept part; L2 directly from the removed part, which is far smaller
        # than the total
        squares = abs_c ** 2
        total = abs_c.reshape(-1, n_out).sum(axis=0)
        kept = np.zeros(n_out)
        removed_sq = squares.reshape(-1, n_out).sum(axis=0)  # everything, for outputs in no block
        blocks = []
        for box, members in boxes:
            cut = tuple(slice(k) for k in box)
            if members is None:
                kept = abs_c[cut].reshape(-1, n_out).sum(axis=0)
                removed_sq = outside_box_sum(squares, box)
                blocks.append((full[cut], np.arange(n_out, dtype=np.int64)))
            else:
                kept[members] = abs_c[cut + (members,)].reshape(-1, len(members)).sum(axis=0)
                removed_sq[members] = outside_box_sum(squares[..., members], box)
                blocks.append((full[cut + (members,)], members))
        self._discarded = np.maximum(total - kept, 0.0)
        self._discarded_l2 = np.sqrt(removed_sq)
        self._truncation = {"rel_tol": float(rel_tol), "norm": norm, "per_output": bool(per_output),
                            "max_blocks": max_blocks, "groups": labels}
        self._set_blocks(blocks, n_out, self._dtype)
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
        blocks = ""
        if not self._dense:
            stored = sum(c.size for c, _ in self._blocks) / (np.prod(self._grid_shape) * self._n_out)
            blocks = f"blocks={len(self._blocks)}, stored={stored:.1%}, "
        return (
            f"ChebyshevInterpolantND(axes={self._axis_names}, grid_shape={self._grid_shape}, "
            f"kmax={self._kmax}, outputs={self.n_outputs}, {blocks}nodes={self._nodes!r}, "
            f"dtype={self.dtype})"
        )


def _read_only(array):
    """A read-only view of `array`."""
    view = array.view()
    view.flags.writeable = False
    return view


def _member_values(raw):
    """
    Stack per-member fixed values into a (B, n_fixed) float64 array.

    ``raw`` holds one entry per fixed axis: an array of shape (B,), or a
    scalar that applies to every member. Returns the array module and the
    stacked values.
    """
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
    return xp, xp.stack([xp.broadcast_to(c, (n_members,)) for c in columns], axis=1)


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
    Per-output error estimate from the trailing coefficients: the largest
    of :func:`_trailing_errors_by_axis` over the axes.
    """
    return _trailing_errors_by_axis(abs_c).max(axis=0, initial=0.0)


def _trailing_errors_by_axis(abs_c):
    """
    Per-axis, per-output error estimates from the trailing coefficients.

    ``abs_c`` has shape (N_1, ..., N_d, F); the result has shape (d, F).
    Along each axis, take the largest magnitude in each slice; ignoring
    trailing slices that are exactly zero (a derivative's top
    coefficients), the last and second-to-last are scaled by the number of
    remaining slices n and by n - 1 (the convention of the 2D-4D classes),
    and the larger is the estimate. An axis with a single nonzero slice
    contributes 0.
    """
    d = abs_c.ndim - 1
    n_out = abs_c.shape[-1]
    err = np.zeros((d, n_out))
    for ax in range(d):
        slices = np.moveaxis(abs_c, ax, 0).reshape(abs_c.shape[ax], -1, n_out).max(axis=1)
        for f in range(n_out):
            nonzero = np.flatnonzero(slices[:, f])
            n = nonzero[-1] + 1 if nonzero.size else 0
            if n >= 2:
                err[ax, f] = max(slices[n - 1, f] * n, slices[n - 2, f] * (n - 1))
    return err
