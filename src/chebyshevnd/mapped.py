"""Interpolants evaluated in physical coordinates, through domain maps."""

import numpy as np

from .backend import array_namespace, resolve_device, to_numpy
from .batched import evaluate_batch as _evaluate_batch
from .batched import value_and_grad_batch as _value_and_grad_batch
from .errors import OutOfDomainError
from .interpnd import ChebyshevInterpolantND, _member_values
from .maps import (
    AxisMap,
    RawMappedTable,
    dependent_axes,
    eval_grad_physical_into,
    eval_physical_into,
    forward_map,
    inverse_map,
    pack_maps,
)


class MappedInterpolant:
    """
    A Chebyshev table evaluated in physical coordinates.

    The table (a :class:`ChebyshevInterpolantND`) lives on a box in table
    coordinates; each axis has an :class:`AxisMap` from its physical
    coordinate to its table coordinate. Maps may use *boundary functions*:
    the outputs of a boundary table, a real :class:`ChebyshevInterpolantND`
    over the physical coordinates of some of the other axes. This lets the
    table cover a region that is not a box in physical coordinates, e.g.
    eccentric orbits above a separatrix ``p_sep(e)``, or momenta near an
    ``r``-dependent reference value.

    Every method takes and returns physical coordinates, and derivatives are
    with respect to physical coordinates (the chain rule through the maps,
    including the boundary functions, is applied). Box-level operations
    (``optimize``, ``derivative``, ...) are done on :attr:`table` before
    wrapping it.

    Evaluation reuses buffers held by the instance, so one instance must not
    be evaluated from several threads at once.
    """

    def __init__(self, table, maps=None, boundary=None):
        """
        Wrap a table with maps for some of its axes.

        Parameters
        ----------
        table : ChebyshevInterpolantND
            The table in table coordinates. Its axes are named after the
            physical variables (e.g. ``"r"`` for an axis stored as ``1/r``).
        maps : mapping, optional
            Axis name (or index) -> :class:`AxisMap`. Unlisted axes are not
            mapped (table coordinate = physical coordinate).
        boundary : ChebyshevInterpolantND, optional
            The boundary table: real, with named outputs (the boundary
            functions that maps refer to by name), and axes named after
            physical axes of ``table`` whose own maps do not use the
            boundary. Its domain must contain the physical range of those
            axes.
        """
        if not isinstance(table, ChebyshevInterpolantND):
            raise TypeError("table must be a ChebyshevInterpolantND.")
        resolved = [AxisMap() for _ in table.axis_names]
        for key, axis_map in (maps or {}).items():
            resolved[table._axis_position(key)] = axis_map
        self._table = table
        self._maps, self._baxes = validate_maps(table.axis_names, resolved, boundary)
        self._boundary = boundary
        self._codes, self._params = pack_maps(self._maps, boundary.output_names if boundary else ())
        self._dependent = dependent_axes(self._codes)
        if boundary is not None:
            check_boundary_covers(table.axis_names, table.domains, self._codes, self._params,
                                  self._baxes, boundary)
        self._raw = None
        self._buffers = None
        self._device = {}

    # ------------------------------------------------------------- properties

    @property
    def table(self):
        """The underlying :class:`ChebyshevInterpolantND`, in table coordinates."""
        return self._table

    @property
    def boundary(self):
        """The boundary table, or None."""
        return self._boundary

    @property
    def maps(self):
        """Axis name -> :class:`AxisMap`, for every axis."""
        return dict(zip(self._table.axis_names, self._maps))

    @property
    def axis_names(self):
        """The name of each axis."""
        return self._table.axis_names

    @property
    def ndim(self):
        """Number of variables d."""
        return self._table.ndim

    @property
    def output_names(self):
        """The name of each function; empty without a function axis."""
        return self._table.output_names

    @property
    def output_index(self):
        """Mapping from function name to its position in evaluation results."""
        return self._table.output_index

    @property
    def n_outputs(self):
        """Number of functions F."""
        return self._table.n_outputs

    @property
    def dtype(self):
        """Coefficient dtype."""
        return self._table.dtype

    @property
    def domain_tol(self):
        """Allowed overshoot outside the table domain, as a fraction of the half-width."""
        return self._table.domain_tol

    # ----------------------------------------------------- coordinate changes

    def to_table(self, points):
        """
        Table coordinates of physical points.

        Parameters
        ----------
        points : array, shape (B, d)
            Physical points (NumPy or CuPy).

        Returns
        -------
        array, shape (B, d)
            NaN where a map is undefined (e.g. the log of a negative number).
            Points outside the domain are mapped without complaint.
        """
        xp, pts = self._batch_points(points)
        dev = self._on(xp)
        bvals = None
        if dev["boundary"] is not None:
            bvals, _ = _evaluate_batch(*dev["boundary"], _columns(pts, self._baxes, xp), tol=np.inf)
        return inverse_map(pts, self._codes, dev["params"], xp, bvals=bvals, baxes=self._baxes)

    def to_physical(self, table_points):
        """
        Physical coordinates of points given in table coordinates.

        Parameters
        ----------
        table_points : array, shape (B, d)

        Returns
        -------
        array, shape (B, d)
        """
        xp, pts = self._batch_points(table_points)
        dev = self._on(xp)

        def boundary(xb):
            return _evaluate_batch(*dev["boundary"], xb, tol=np.inf)[0]

        return forward_map(pts, self._codes, dev["params"], xp, boundary=boundary, baxes=self._baxes)

    # ------------------------------------------------------- single points

    def evaluate(self, *xs):
        """
        Evaluate at a single point in physical coordinates.

        Parameters
        ----------
        *xs : float
            One physical coordinate per axis, in axis order.

        Returns
        -------
        float, complex or np.ndarray
            For a single function, its value; with a function axis, an
            array of shape (F,).

        Raises
        ------
        OutOfDomainError
            If the point maps outside the table's domain (or the boundary
            table's), or a map is undefined there.
        """
        args, out, _ = self._single_buffers()
        x = self._point(xs, "evaluate")
        status = eval_physical_into(*args, out)
        if status:
            self._raise_point(x, status)
        return out.item(0) if not self.output_names else out.copy()

    def value_and_grad(self, *xs):
        """
        Evaluate and differentiate at a single point in physical coordinates.

        Parameters
        ----------
        *xs : float
            One physical coordinate per axis, in axis order.

        Returns
        -------
        value : float, complex or np.ndarray
            As for :meth:`evaluate`.
        grad : np.ndarray
            Derivatives with respect to the physical coordinates: shape
            (d,) for a single function, (d, F) with a function axis.

        Raises
        ------
        OutOfDomainError
            As for :meth:`evaluate`.
        """
        args, _, out = self._single_buffers()
        x = self._point(xs, "value_and_grad")
        status = eval_grad_physical_into(*args, out)
        if status:
            self._raise_point(x, status)
        if not self.output_names:
            return out[0, 0].item(), out[1:, 0].copy()
        return out[0].copy(), out[1:].copy()

    def raw(self):
        """
        The mapped table as plain arrays, for the user's own Numba code.

        Precontract first, then pass the fields to an ``@njit`` function
        and evaluate there with :func:`chebyshevnd.eval_physical_into` or
        :func:`chebyshevnd.eval_grad_physical_into`.

        Returns
        -------
        RawMappedTable
            Read-only views; a snapshot of the current tables.
        """
        if self._raw is None:
            t = self._table.raw()
            if self._boundary is None:
                bcoeffs, bshape = np.zeros(1), np.zeros(0, dtype=np.int64)
                bbounds, baxes = np.zeros((2, 0)), np.zeros(0, dtype=np.int64)
            else:
                b = self._boundary.raw()
                bcoeffs, bshape, bbounds = b.coeffs, b.shape, b.bounds
                baxes = np.array(self._baxes, dtype=np.int64)
            self._raw = RawMappedTable(t.coeffs, t.shape, t.bounds, t.tol, *map(_read_only, (
                self._codes, self._params, bcoeffs, bshape, bbounds, baxes)))
        return self._raw

    # ------------------------------------------------------------- batches

    def evaluate_batch(self, points, return_status=False):
        """
        Evaluate at many physical points at once (NumPy on the CPU, CuPy on the GPU).

        Parameters
        ----------
        points : array, shape (B, d)
            Physical points, one per row.
        return_status : bool, optional
            Return the per-point status instead of raising.

        Returns
        -------
        values : array
            Shape (B,) for a single function, (B, F) with a function axis.
        status : int64 array, shape (B,)
            Only if ``return_status``; see :func:`chebyshevnd.eval_physical_into`.
        """
        xp, pts = self._batch_points(points)
        dev = self._on(xp)
        values, status = _physical_batch(dev["coeffs"], dev["a"], dev["b"], self._codes,
                                         dev["params"], dev["boundary"], self._baxes, pts,
                                         self.domain_tol, 0, xp)
        if not self.output_names:
            values = values[:, 0]
        return self._finish(xp, pts, status, return_status, values)

    def value_and_grad_batch(self, points, return_status=False):
        """
        Evaluate and differentiate at many physical points at once.

        Parameters
        ----------
        points : array, shape (B, d)
            Physical points, one per row.
        return_status : bool, optional
            Return the per-point status instead of raising.

        Returns
        -------
        values : array
            Shape (B,) or (B, F).
        grads : array
            Shape (B, d) or (B, d, F): derivatives with respect to the
            physical coordinates.
        status : int64 array, shape (B,)
            Only if ``return_status``.
        """
        xp, pts = self._batch_points(points)
        dev = self._on(xp)
        values, grads, status = _physical_batch(dev["coeffs"], dev["a"], dev["b"], self._codes,
                                                dev["params"], dev["boundary"], self._baxes, pts,
                                                self.domain_tol, 1, xp)
        if not self.output_names:
            values, grads = values[:, 0], grads[..., 0]
        return self._finish(xp, pts, status, return_status, values, grads)

    def value_grad_hess_batch(self, points, return_status=False):
        """
        Values, gradients and Hessians in physical coordinates, at many points.

        Meant for building tables (e.g. higher-order NIT terms from
        derivatives of lower-order ones at the grid nodes), not for the
        right-hand side of an ODE: it builds the second-derivative tables
        of the table and the boundary on every call.

        The Hessian combines the table's Hessian with the map's Jacobian J
        and Hessian H: ``d2F/dx_i dx_j = sum_ab F_ab J_ai J_bj + sum_a F_a H_aij``.

        Parameters
        ----------
        points : array, shape (B, d)
            Physical points, one per row.
        return_status : bool, optional
            Return the per-point status instead of raising.

        Returns
        -------
        values : array
            Shape (B,) or (B, F).
        grads : array
            Shape (B, d) or (B, d, F).
        hessians : array
            Shape (B, d, d) or (B, d, d, F).
        status : int64 array, shape (B,)
            Only if ``return_status``.
        """
        xp, pts = self._batch_points(points)
        dev = self._on(xp)
        tol = self.domain_tol
        bvals = bgrads = bhess = None
        bits = xp.zeros(pts.shape[0], dtype=xp.int64)
        if self._boundary is not None:
            xb = _columns(pts, self._baxes, xp)
            bvals, bgrads, bstatus = _value_and_grad_batch(*dev["boundary"], xb, tol=tol)
            bits = _boundary_bits(bstatus, self._baxes, xp)
            bhess = _hessian_batch(self._boundary, xb, xp, tol)
        y, jac, map_hess = inverse_map(pts, self._codes, dev["params"], xp, bvals, bgrads, bhess,
                                       self._baxes, order=2)
        values, gy, status = _value_and_grad_batch(dev["coeffs"], dev["a"], dev["b"], y, tol=tol)
        gyy = _hessian_batch(self._table, y, xp, tol)
        jac = jac.astype(gy.dtype)
        grads = xp.einsum("naf,nai->nif", gy, jac)
        hess = (xp.einsum("nabf,nai,nbj->nijf", gyy, jac, jac)
                + xp.einsum("naf,naij->nijf", gy, map_hess.astype(gy.dtype)))
        status = status | bits
        if not self.output_names:
            values, grads, hess = values[:, 0], grads[..., 0], hess[..., 0]
        return self._finish(xp, pts, status, return_status, values, grads, hess)

    # ------------------------------------------------------- precontraction

    def precontract(self, fixed):
        """
        Fix some axes at physical values and return the mapped interpolant over the others.

        The fixed values are mapped to table coordinates and the table is
        precontracted there. Boundary functions follow: if the boundary
        table's inputs are all fixed, the boundary functions become numbers
        in the remaining maps; if some are, the boundary table is
        precontracted at them too.

        Parameters
        ----------
        fixed : mapping
            Axis name (or index) -> physical value. An axis whose map uses
            the boundary can only be fixed together with every input of the
            boundary table.

        Returns
        -------
        MappedInterpolant

        Raises
        ------
        OutOfDomainError
            If a fixed value lies outside the domain.
        """
        axes, values = self._table._resolve_fixed(fixed)
        for value in values:
            if np.ndim(value) != 0:
                raise TypeError("precontract() takes one value per fixed axis; use "
                                "precontract_batch() for arrays.")
        x = np.zeros((1, self.ndim))
        x[0, axes] = values
        y, bvals, status = self._fixed_table_values(np, axes, x)
        if status[0]:
            raise OutOfDomainError("Fixed value outside the interpolation domain: "
                                   + self._describe(x[0], int(status[0])) + ".")
        child = self._table.precontract({ax: float(y[0, i]) for i, ax in enumerate(axes)})

        names = self.axis_names
        live = [a for a in range(self.ndim) if a not in axes]
        maps = [self._maps[a] for a in live]
        boundary = self._boundary
        if boundary is not None:
            fixed_inputs = [j for j in self._baxes if j in axes]
            if len(fixed_inputs) == len(self._baxes):
                constants = dict(zip(boundary.output_names, bvals[0].tolist()))
                maps = [m.bind(constants) if m.references else m for m in maps]
                boundary = None
            elif fixed_inputs:
                boundary = boundary.precontract({names[j]: float(x[0, j]) for j in fixed_inputs})
        return MappedInterpolant(child, dict(zip([names[a] for a in live], maps)), boundary)

    def precontract_batch(self, fixed, return_status=False):
        """
        Fix some axes at different physical values for each member of a batch.

        The batch version of :meth:`precontract`, on the device of the
        given arrays.

        Parameters
        ----------
        fixed : mapping
            Axis name (or index) -> array of shape (B,) of physical values
            (scalars apply to every member).
        return_status : bool, optional
            Return the per-member status instead of raising.

        Returns
        -------
        MappedBatch
        status : int64 array, shape (B,)
            Only if ``return_status``.
        """
        axes, raw = self._table._resolve_fixed(fixed)
        xp, values = _member_values(raw)
        n_members = values.shape[0]
        x = xp.zeros((n_members, self.ndim), dtype=xp.float64)
        for i, ax in enumerate(axes):
            x[:, ax] = values[:, i]
        y, bvals, bits = self._fixed_table_values(xp, axes, x)
        batch, status = self._table.precontract_batch({ax: y[:, i] for i, ax in enumerate(axes)},
                                                      return_status=True)
        status = status | bits

        names = self.axis_names
        live = [a for a in range(self.ndim) if a not in axes]
        codes = self._codes[live].copy()
        params = xp.ascontiguousarray(xp.broadcast_to(self._on(xp)["params"][xp.asarray(live)],
                                                      (n_members, len(live), 4)))
        boundary, baxes = None, []
        if self._boundary is not None:
            fixed_inputs = [j for j in self._baxes if j in axes]
            if len(fixed_inputs) == len(self._baxes):
                # The boundary functions become per-member numbers
                for i in range(len(live)):
                    for col in (1, 2):
                        src = int(codes[i, col])
                        if src >= 0:
                            params[:, i, col - 1] = bvals[:, src]
                            codes[i, col] = -1
            else:
                if fixed_inputs:
                    bbatch, bstatus = self._boundary.precontract_batch(
                        {names[j]: x[:, j] for j in fixed_inputs}, return_status=True)
                    status = status | _boundary_bits(bstatus, self._baxes, xp)
                    boundary = (bbatch.tables, xp.asarray(bbatch.domains).T[0],
                                xp.asarray(bbatch.domains).T[1])
                    remaining = bbatch.axis_names
                else:
                    boundary = self._on(xp)["boundary"]
                    remaining = self._boundary.axis_names
                baxes = [live.index(names.index(n)) for n in remaining]

        result = MappedBatch(batch.tables, *(xp.asarray(v) for v in zip(*batch.domains)), codes,
                             params, boundary, baxes, axis_names=[names[a] for a in live],
                             output_names=self.output_names, domain_tol=self.domain_tol,
                             maps=[self._maps[a] for a in live])
        if return_status:
            return result, status
        bad = status != 0
        if bool(xp.any(bad)):
            first = int(xp.flatnonzero(bad)[0])
            point = np.full(self.ndim, np.nan)
            point[axes] = to_numpy(values[first])
            raise OutOfDomainError(
                f"{int(bad.sum())} of {n_members} members have fixed values outside the "
                f"interpolation domain; the first is member {first}: "
                + self._describe(point, int(status[first])) + ".")
        return result

    def _fixed_table_values(self, xp, axes, x):
        """
        Table coordinates of the fixed axes of the points ``x`` (only their
        columns are used), the boundary values if the boundary was needed,
        and the status bits of the boundary evaluation.
        """
        names = self.axis_names
        needs_boundary = [a for a in axes if a in self._dependent]
        bvals = None
        bits = xp.zeros(x.shape[0], dtype=xp.int64)
        if self._boundary is not None and set(self._baxes) <= set(axes):
            bvals, bstatus = _evaluate_batch(*self._on(xp)["boundary"], _columns(x, self._baxes, xp),
                                             tol=self.domain_tol)
            bits = _boundary_bits(bstatus, self._baxes, xp)
        elif needs_boundary:
            missing = [names[j] for j in self._baxes if j not in axes]
            raise ValueError(
                f"The maps of {[names[a] for a in needs_boundary]} use the boundary table, so they "
                f"can only be fixed together with all of its inputs; {missing} are not fixed."
            )
        y = inverse_map(x, self._codes, self._on(xp)["params"], xp, bvals=bvals, baxes=self._baxes,
                        axes=axes)
        # Status of the fixed values in the table's domain
        a = xp.asarray([self._table.domains[ax][0] for ax in axes])
        b = xp.asarray([self._table.domains[ax][1] for ax in axes])
        t = (2.0 * y - a - b) / (b - a)
        outside = ~(xp.abs(t) <= 1.0 + self.domain_tol)
        weights = xp.asarray([1 << ax for ax in axes], dtype=xp.int64)
        return y, bvals, bits | (outside * weights).sum(axis=1)

    def to_device(self, device="cuda"):
        """
        Copy the table, the boundary table and the maps to a device now.

        Parameters
        ----------
        device : {"cuda", "cpu"}
            "cuda" copies to CuPy's current device; "cpu" does nothing.

        Returns
        -------
        MappedInterpolant
            This interpolant.
        """
        xp = resolve_device(device)
        if xp is not np:
            self._on(xp)
        return self

    # -------------------------------------------------------------- helpers

    def _on(self, xp):
        """The tables and packed parameters as arrays of module ``xp``, cached per device."""
        key = xp.__name__
        if key not in self._device:
            coeffs, a, b = self._table._table_on(xp)
            boundary = None if self._boundary is None else self._boundary._table_on(xp)
            self._device[key] = {"coeffs": coeffs, "a": a, "b": b, "boundary": boundary,
                                 "params": xp.asarray(self._params)}
        return self._device[key]

    def _single_buffers(self):
        """
        Kernel arguments up to (not including) ``out``, with the point buffer
        ``self._x`` in place of x, and the output arrays for values and for
        values + gradients. Built once, so a call only fills ``self._x``.
        """
        if self._buffers is None:
            raw = self.raw()
            # The same arrays as raw(), but writable: Numba dispatches
            # read-only arrays more slowly
            arrays = [np.array(a) if isinstance(a, np.ndarray) else a for a in raw]
            self._x = np.empty(self.ndim)
            self._buffers = ((*arrays, self._x, *raw.workspace()),
                             np.empty(self.n_outputs, dtype=self.dtype),
                             np.empty((self.ndim + 1, self.n_outputs), dtype=self.dtype))
        return self._buffers

    def _point(self, xs, method):
        """Validate scalar coordinates and copy them into the point buffer."""
        if len(xs) != self.ndim:
            raise TypeError(f"{method}() takes {self.ndim} coordinates {self.axis_names}, got {len(xs)}.")
        try:
            self._x[:] = xs
        except (TypeError, ValueError):
            raise TypeError(f"{method}() takes one scalar per axis.") from None
        return self._x

    def _batch_points(self, points):
        xp = array_namespace(points)
        pts = xp.asarray(points, dtype=xp.float64)
        if pts.ndim == 1 and self.ndim == 1:
            pts = pts[:, None]
        if pts.ndim != 2 or pts.shape[1] != self.ndim:
            raise ValueError(f"points must have shape (B, {self.ndim}) for axes {self.axis_names}; "
                             f"got {tuple(pts.shape)}.")
        return xp, pts

    def _finish(self, xp, pts, status, return_status, *results):
        if return_status:
            return (*results, status)
        bad = status != 0
        if bool(xp.any(bad)):
            first = int(xp.flatnonzero(bad)[0])
            raise OutOfDomainError(
                f"{int(bad.sum())} of {pts.shape[0]} points are outside the interpolation domain; "
                f"the first is point {first}: "
                + self._describe(to_numpy(pts[first]), int(status[first])) + ".")
        return results[0] if len(results) == 1 else results

    def _raise_point(self, x, status):
        raise OutOfDomainError("Point outside the interpolation domain: "
                               + self._describe(x, status) + ".")

    def _describe(self, x, status):
        """Describe the flagged coordinates of physical point ``x`` (NaN where unknown)."""
        x = np.asarray(x, dtype=np.float64)
        known = np.where(np.isnan(x), 0.0, x)
        y = to_numpy(self.to_table(known[None]))[0]
        parts = []
        for i, name in enumerate(self.axis_names):
            if not status >> i & 1:
                continue
            a, b = self._table.domains[i]
            if self._maps[i].is_identity:
                parts.append(f"{name} = {float(x[i])!r} not in [{a!r}, {b!r}]")
            elif np.isnan(y[i]) or not a <= y[i] <= b:
                parts.append(f"{name} = {float(x[i])!r} maps to {float(y[i])!r}, outside the "
                             f"table domain [{a!r}, {b!r}]")
            else:
                parts.append(f"{name} = {float(x[i])!r} is outside the boundary table's domain")
        return "; ".join(parts)

    def __repr__(self):
        mapped = {n: m for n, m in self.maps.items() if not m.is_identity}
        return (f"MappedInterpolant(axes={self.axis_names}, maps={mapped}, "
                f"boundary={None if self._boundary is None else self._boundary.output_names}, "
                f"table={self._table!r})")


class MappedBatch:
    """
    One mapped table per batch member, all on the same device.

    Made by :meth:`MappedInterpolant.precontract_batch`. Member j evaluates
    its own table, with its own boundary functions, at its own physical
    point.
    """

    def __init__(self, tables, a, b, codes, params, boundary, baxes, *, axis_names, output_names,
                 domain_tol, maps):
        self._tables = tables
        self._a, self._b = a, b
        self._codes = codes
        self._params = params
        self._boundary = boundary
        self._baxes = list(baxes)
        self._axis_names = tuple(axis_names)
        self._output_names = tuple(output_names)
        self._domain_tol = domain_tol
        self._maps = tuple(maps)

    @property
    def n_members(self):
        """Number of batch members B."""
        return self._tables.shape[0]

    @property
    def ndim(self):
        """Number of remaining axes d."""
        return len(self._axis_names)

    @property
    def axis_names(self):
        """Names of the remaining axes."""
        return self._axis_names

    @property
    def output_names(self):
        """The name of each function; empty without a function axis."""
        return self._output_names

    @property
    def output_index(self):
        """Mapping from function name to its position in evaluation results."""
        return {name: i for i, name in enumerate(self._output_names)}

    @property
    def n_outputs(self):
        """Number of functions F."""
        return self._tables.shape[-1]

    @property
    def dtype(self):
        """Coefficient dtype."""
        return self._tables.dtype

    def evaluate(self, points, return_status=False):
        """
        Evaluate each member at its own physical point.

        Parameters
        ----------
        points : array, shape (B, d)
            Row j is member j's point.
        return_status : bool, optional
            Return the per-member status instead of raising.

        Returns
        -------
        values : array, shape (B,) or (B, F)
        status : int64 array, shape (B,)
            Only if ``return_status``.
        """
        xp, pts = self._points(points)
        values, status = _physical_batch(self._tables, self._a, self._b, self._codes, self._params,
                                         self._boundary, self._baxes, pts, self._domain_tol, 0, xp)
        if not self._output_names:
            values = values[:, 0]
        return self._finish(xp, pts, status, return_status, values)

    def value_and_grad(self, points, return_status=False):
        """
        Evaluate each member and its physical gradient at its own point.

        Returns
        -------
        values : array, shape (B,) or (B, F)
        grads : array, shape (B, d) or (B, d, F)
        status : int64 array, shape (B,)
            Only if ``return_status``.
        """
        xp, pts = self._points(points)
        values, grads, status = _physical_batch(self._tables, self._a, self._b, self._codes,
                                                self._params, self._boundary, self._baxes, pts,
                                                self._domain_tol, 1, xp)
        if not self._output_names:
            values, grads = values[:, 0], grads[..., 0]
        return self._finish(xp, pts, status, return_status, values, grads)

    def _points(self, points):
        xp = array_namespace(self._tables)
        pts = xp.asarray(points, dtype=xp.float64)
        if pts.ndim == 1 and self.ndim == 1:
            pts = pts[:, None]
        if pts.ndim != 2 or pts.shape != (self.n_members, self.ndim):
            raise ValueError(f"points must have shape ({self.n_members}, {self.ndim}): one point per "
                             f"member, axes {self._axis_names}; got {tuple(pts.shape)}.")
        return xp, pts

    def _finish(self, xp, pts, status, return_status, *results):
        if return_status:
            return (*results, status)
        bad = status != 0
        if bool(xp.any(bad)):
            first = int(xp.flatnonzero(bad)[0])
            flagged = [self._axis_names[i] for i in range(self.ndim) if int(status[first]) >> i & 1]
            point = to_numpy(pts[first])
            raise OutOfDomainError(
                f"{int(bad.sum())} of {self.n_members} members are outside the interpolation "
                f"domain; the first is member {first}, at "
                + ", ".join(f"{n} = {float(v)!r}" for n, v in zip(self._axis_names, point))
                + f" (outside on {flagged}).")
        return results[0] if len(results) == 1 else results

    def __repr__(self):
        return (f"MappedBatch(members={self.n_members}, axes={self._axis_names}, "
                f"outputs={self.n_outputs}, dtype={self.dtype})")


# ------------------------------------------------------------------ helpers


def validate_maps(names, maps, boundary):
    """
    Check one map per axis against the boundary table.

    Parameters
    ----------
    names : sequence of str
        Axis names.
    maps : sequence of AxisMap
        One map per axis.
    boundary : ChebyshevInterpolantND or None

    Returns
    -------
    maps : tuple of AxisMap
    baxes : list of int
        For each boundary input, the index of that axis.
    """
    names = tuple(names)
    for name, axis_map in zip(names, maps):
        if not isinstance(axis_map, AxisMap):
            raise TypeError(f"The map for axis {name!r} must be an AxisMap, got {axis_map!r}.")
    references = sorted({r for m in maps for r in m.references})
    if boundary is None:
        if references:
            raise ValueError(f"The maps use boundary outputs {references}, but no boundary table "
                             "was given.")
        return tuple(maps), []
    if not isinstance(boundary, ChebyshevInterpolantND):
        raise TypeError("boundary must be a ChebyshevInterpolantND.")
    if boundary.dtype.kind == "c":
        raise ValueError("The boundary table must be real.")
    if not boundary.output_names:
        raise ValueError("The boundary table needs output_names, so maps can refer to them.")
    missing = [r for r in references if r not in boundary.output_index]
    if missing:
        raise ValueError(f"The maps use boundary outputs {missing}, which the boundary table does "
                         f"not have; its outputs are {boundary.output_names}.")
    unknown = [n for n in boundary.axis_names if n not in names]
    if unknown:
        raise ValueError(f"The boundary table's axes {unknown} are not axes of the table {names}.")
    baxes = [names.index(n) for n in boundary.axis_names]
    bad = [names[j] for j in baxes if maps[j].references]
    if bad:
        raise ValueError(f"Axes {bad} are inputs of the boundary table, so their own maps cannot "
                         "use the boundary.")
    return tuple(maps), baxes


def check_boundary_covers(names, domains, codes, params, baxes, boundary):
    """Raise unless the boundary table is defined over the physical range of its inputs."""
    for i, j in enumerate(baxes):
        a, b = domains[j]
        ends = forward_map(np.array([[a], [b]], dtype=np.float64), codes[j:j + 1], params[j:j + 1], np)
        lo, hi = float(np.min(ends)), float(np.max(ends))
        blo, bhi = boundary.domains[i]
        slack = 1e-9 * (bhi - blo)
        if not (blo - slack <= lo and hi <= bhi + slack):
            raise ValueError(
                f"The boundary table covers {names[j]} in [{blo!r}, {bhi!r}], but the table's "
                f"domain on that axis reaches {names[j]} in [{lo!r}, {hi!r}]."
            )


def _read_only(array):
    view = np.ascontiguousarray(array).view()
    view.flags.writeable = False
    return view


def _columns(points, idx, xp):
    """points[:, idx] for a list of column indices, valid for NumPy and CuPy."""
    return xp.stack([points[:, j] for j in idx], axis=1)


def _boundary_bits(bstatus, baxes, xp):
    """Boundary status bits, renumbered to the axes of the main table."""
    bits = xp.zeros_like(bstatus)
    for jb, j in enumerate(baxes):
        bits = bits | (((bstatus >> jb) & 1) << j)
    return bits


def _physical_batch(coeffs, a, b, codes, params, boundary, baxes, pts, tol, order, xp):
    """
    Batched evaluation in physical coordinates: the boundary, the inverse
    map, the table, and (``order == 1``) the gradient chain rule.

    ``coeffs`` and ``boundary[0]`` may be shared or per-member tables.
    Returns ``(values, status)`` or ``(values, grads, status)``.
    """
    bvals = bgrads = None
    bits = xp.zeros(pts.shape[0], dtype=xp.int64)
    if boundary is not None:
        xb = _columns(pts, baxes, xp)
        if order:
            bvals, bgrads, bstatus = _value_and_grad_batch(*boundary, xb, tol=tol)
        else:
            bvals, bstatus = _evaluate_batch(*boundary, xb, tol=tol)
        bits = _boundary_bits(bstatus, baxes, xp)
    if order == 0:
        y = inverse_map(pts, codes, params, xp, bvals=bvals, baxes=baxes)
        values, status = _evaluate_batch(coeffs, a, b, y, tol=tol)
        return values, status | bits
    y, jac = inverse_map(pts, codes, params, xp, bvals=bvals, bgrads=bgrads, baxes=baxes, order=1)
    values, gy, status = _value_and_grad_batch(coeffs, a, b, y, tol=tol)
    # dF/dx_i = sum_a dF/dy_a dy_a/dx_i
    grads = xp.matmul(xp.swapaxes(jac, 1, 2).astype(gy.dtype), gy)
    return values, grads, status | bits


def _hessian_batch(interp, points, xp, tol):
    """Second derivatives of ``interp`` (table coordinates) at ``points``: shape (B, d, d, F)."""
    d = interp.ndim
    out = None
    for i in range(d):
        for j in range(i, d):
            orders = [0] * d
            orders[i] += 1
            orders[j] += 1
            coeffs, a, b = interp.derivative(orders)._table_on(xp)
            vals, _ = _evaluate_batch(coeffs, a, b, points, tol=np.inf)
            if out is None:
                out = xp.empty((points.shape[0], d, d, vals.shape[1]), dtype=vals.dtype)
            out[:, i, j] = vals
            out[:, j, i] = vals
    return out
