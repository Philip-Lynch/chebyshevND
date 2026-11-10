"""
Domain maps between physical coordinates and table coordinates.

A table is a tensor Chebyshev series over a box in *table coordinates* y.
The physical variables x the user works with need not fill a box: an orbit's
angular momentum follows a value that depends on its radius, and the
separatrix bounds the semi-latus rectum from below at a value that depends
on eccentricity and spins. A map per axis turns the physical region into the
table's box:

    y_a = f_a(s_a),    s_a = (x_a - L_a) / W_a,

where the transform f_a is one of

    identity    f(s) = s
    log         f(s) = log(s + offset)
    power       f(s) = (s + offset) ** power      (e.g. power=-1 for u = 1/r)

and the window (L_a, W_a) is either constant, or given by *boundary
functions* of other physical coordinates: outputs of one real
:class:`~chebyshevnd.ChebyshevInterpolantND` (the boundary table) over some
of the other axes. W_a is either a width, or ``upper - L_a`` for an upper
bound. Axes the boundary table takes as inputs must have constant windows
themselves, so the map from physical to table coordinates needs a single
pass.

The module holds the map specification (:class:`AxisMap`), its packed
numeric form (``codes``, ``params``), the inverse map with its Jacobian and
Hessian written once against an array module ``xp`` (NumPy or CuPy), the
forward map, and the Numba kernels that evaluate a mapped table in
physical coordinates (:func:`eval_physical_into`,
:func:`eval_grad_physical_into`).
"""

import math
import numbers
from typing import NamedTuple

import numpy as np
from numba import njit

from .kernels import Workspace, evaluate_point, value_and_grad_point

#: Transform codes, in the packed ``codes[:, 0]`` column
TRANSFORMS = {"identity": 0, "log": 1, "power": 2}
_TRANSFORM_NAMES = {code: name for name, code in TRANSFORMS.items()}

#: ``codes[:, 3]``: how the scale entry defines the window width
SCALE_IS_WIDTH, SCALE_IS_UPPER = 0, 1


class AxisMap:
    """
    The map from one physical coordinate x to its table coordinate y.

    ``y = f((x - lower) / W)``, with ``W = width``, or ``W = upper - lower``
    if ``upper`` is given, and ``f`` the transform:

    - ``"identity"``: ``f(s) = s``;
    - ``"log"``: ``f(s) = log(s + offset)``;
    - ``"power"``: ``f(s) = (s + offset) ** power``.

    ``lower``, ``upper`` and ``width`` are numbers, or the name of an output
    of the interpolant's boundary table, a function of other physical
    coordinates. For example, with ``r`` and ``p_phi`` axes:

    - ``AxisMap("power", power=-1)`` stores the ``r`` axis as ``u = 1/r``;
    - ``AxisMap(lower="p_ref", width="p_width")`` stores ``p_phi`` relative to
      a reference ``p_ref(r)``, in units of ``p_width(r)``;
    - ``AxisMap("log", lower="p_sep", offset=1e-3)`` stores ``p`` as
      ``log(p - p_sep(e) + 1e-3)``;
    - ``AxisMap(lower="p_sep", upper=20.0)`` maps ``[p_sep(e), 20]`` to
      ``[0, 1]``.

    The table's domain on that axis is given in table coordinates y. For a
    map with a constant window, :meth:`table_domain` converts a physical
    interval.

    Parameters
    ----------
    transform : {"identity", "log", "power"}
        The transform f.
    lower : float or str, optional
        L, the lower end of the window. Default 0.
    upper : float or str, optional
        Upper end of the window; ``W = upper - lower``. Not with ``width``.
    width : float or str, optional
        The window width W. Default 1 when ``upper`` is not given.
    offset : float, optional
        Added to s before a log or power transform; not used by identity.
    power : float, optional
        The exponent of a power transform (nonzero).
    """

    def __init__(self, transform="identity", *, lower=0.0, upper=None, width=None, offset=0.0,
                 power=None):
        if transform not in TRANSFORMS:
            raise ValueError(f"transform must be one of {tuple(TRANSFORMS)}, got {transform!r}.")
        if upper is not None and width is not None:
            raise ValueError("Give either upper or width, not both.")
        offset = _finite(offset, "offset")
        if transform == "identity" and offset != 0.0:
            raise ValueError("offset only applies to the log and power transforms.")
        if transform == "power":
            if power is None:
                raise ValueError("A power transform needs power=.")
            power = _finite(power, "power")
            if power == 0.0:
                raise ValueError("power must be nonzero.")
        elif power is not None:
            raise ValueError("power only applies to the power transform.")
        self._transform = transform
        self._lower = _source(lower, "lower")
        if upper is None:
            self._upper = None
            self._width = _source(1.0 if width is None else width, "width")
            if self._width == 0.0:
                raise ValueError("width must be nonzero.")
        else:
            self._upper = _source(upper, "upper")
            self._width = None
            if self._upper == self._lower:
                raise ValueError("upper must differ from lower.")
        self._offset = offset
        self._power = power

    # ------------------------------------------------------------- properties

    @property
    def transform(self):
        """The transform name."""
        return self._transform

    @property
    def lower(self):
        """L: a number, or the name of a boundary output."""
        return self._lower

    @property
    def upper(self):
        """The upper end of the window, or None if the window is given by its width."""
        return self._upper

    @property
    def width(self):
        """The window width, or None if the window is given by its upper end."""
        return self._width

    @property
    def offset(self):
        """Offset added to s before a log or power transform."""
        return self._offset

    @property
    def power(self):
        """Exponent of a power transform, else None."""
        return self._power

    @property
    def references(self):
        """Names of the boundary outputs this map uses."""
        return tuple(v for v in (self._lower, self._upper, self._width) if isinstance(v, str))

    @property
    def is_identity(self):
        """True if y = x."""
        return (self._transform == "identity" and self._lower == 0.0 and self._upper is None
                and self._width == 1.0)

    # ------------------------------------------------------------ conversions

    def table_domain(self, x_lower, x_upper):
        """
        The table-coordinate interval of a physical interval.

        Only for maps with a constant window, whose y is a monotone function
        of x alone.

        Returns
        -------
        (float, float)
            The interval's image, lower end first.
        """
        if self.references:
            raise ValueError("table_domain() needs a map with a constant window; this one uses "
                             f"the boundary outputs {self.references}.")
        codes, params = _pack_one(self, {})
        ys = inverse_map(np.array([[x_lower], [x_upper]], dtype=np.float64), codes[None],
                         params[None], np)[:, 0]
        if not np.all(np.isfinite(ys)):
            raise ValueError(f"[{x_lower}, {x_upper}] is not inside the domain of this map.")
        return float(ys.min()), float(ys.max())

    def bind(self, values):
        """
        This map with boundary outputs replaced by numbers.

        Parameters
        ----------
        values : mapping
            Boundary output name -> number, for (at least) every name in
            :attr:`references`.
        """
        def resolve(v):
            return float(values[v]) if isinstance(v, str) else v

        return AxisMap(self._transform, lower=resolve(self._lower),
                       upper=None if self._upper is None else resolve(self._upper),
                       width=None if self._width is None else resolve(self._width),
                       offset=self._offset, power=self._power)

    def to_dict(self):
        """A JSON-compatible description, for hashing and storage."""
        return {"transform": self._transform, "lower": self._lower, "upper": self._upper,
                "width": self._width, "offset": self._offset, "power": self._power}

    @classmethod
    def from_dict(cls, spec):
        """Inverse of :meth:`to_dict`."""
        return cls(spec["transform"], lower=spec["lower"], upper=spec["upper"],
                   width=spec["width"], offset=spec["offset"], power=spec["power"])

    def __eq__(self, other):
        return isinstance(other, AxisMap) and self.to_dict() == other.to_dict()

    def __hash__(self):
        return hash(tuple(self.to_dict().items()))

    def __repr__(self):
        parts = [repr(self._transform)]
        if self._lower != 0.0:
            parts.append(f"lower={self._lower!r}")
        if self._upper is not None:
            parts.append(f"upper={self._upper!r}")
        elif self._width != 1.0:
            parts.append(f"width={self._width!r}")
        if self._offset:
            parts.append(f"offset={self._offset!r}")
        if self._power is not None:
            parts.append(f"power={self._power!r}")
        return f"AxisMap({', '.join(parts)})"


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    return float(value)


def _source(value, name):
    """A window entry: a boundary output name, or a finite number."""
    if isinstance(value, str):
        return value
    return _finite(value, name)


# ------------------------------------------------------------------ packing


def _pack_one(axis_map, output_index):
    """codes (4,) and params (4,) of one map; ``output_index`` maps boundary output names to columns."""
    def entry(v):
        return (output_index[v], 0.0) if isinstance(v, str) else (-1, v)

    lo_src, lo_val = entry(axis_map.lower)
    if axis_map.upper is not None:
        sc_src, sc_val = entry(axis_map.upper)
        mode = SCALE_IS_UPPER
    else:
        sc_src, sc_val = entry(axis_map.width)
        mode = SCALE_IS_WIDTH
    codes = np.array([TRANSFORMS[axis_map.transform], lo_src, sc_src, mode], dtype=np.int64)
    params = np.array([lo_val, sc_val, axis_map.offset,
                       0.0 if axis_map.power is None else axis_map.power], dtype=np.float64)
    return codes, params


def pack_maps(maps, boundary_outputs=()):
    """
    The packed form of a list of maps, one per axis.

    Parameters
    ----------
    maps : sequence of AxisMap
    boundary_outputs : sequence of str
        The boundary table's output names, in order.

    Returns
    -------
    codes : int64 array, shape (d, 4)
        Per axis: transform code, column of the boundary output giving L
        (-1 for a constant), column giving the scale entry (-1 for a
        constant), and whether the scale entry is the width (0) or the upper
        end (1).
    params : float64 array, shape (d, 4)
        Per axis: constant L, constant scale entry, offset, power.
    """
    index = {name: i for i, name in enumerate(boundary_outputs)}
    packed = [_pack_one(m, index) for m in maps]
    return (np.stack([c for c, _ in packed]).reshape(-1, 4),
            np.stack([p for _, p in packed]).reshape(-1, 4))


def dependent_axes(codes):
    """Axes whose window uses the boundary table."""
    return [a for a in range(codes.shape[0]) if codes[a, 1] >= 0 or codes[a, 2] >= 0]


# --------------------------------------------------------------- xp versions


def inverse_map(x, codes, params, xp, bvals=None, bgrads=None, bhess=None, baxes=(), order=0,
                axes=None):
    """
    Table coordinates of physical points, with derivatives of the map.

    Written against the array module ``xp`` (NumPy or CuPy).

    Parameters
    ----------
    x : array, shape (B, d)
        Physical points.
    codes : NumPy int array, shape (d, 4)
        From :func:`pack_maps` (always on the host: it sets the structure).
    params : array, shape (d, 4) or (B, d, 4)
        From :func:`pack_maps`, or one row per point.
    xp : module
        ``numpy`` or ``cupy``.
    bvals, bgrads, bhess : arrays, optional
        The boundary table at ``x[:, baxes]``: values (B, Fb), gradients
        (B, db, Fb) for ``order >= 1``, Hessians (B, db, db, Fb) for
        ``order >= 2``. Needed if any computed axis uses the boundary.
    baxes : sequence of int
        The axes the boundary table takes as inputs, in its own order.
    order : {0, 1, 2}
        Also return the Jacobian (1) and Hessian (2) of the map.
    axes : sequence of int, optional
        Compute only these axes (default all).

    Returns
    -------
    y : array, shape (B, len(axes))
    jac : array, shape (B, len(axes), d)
        Only if ``order >= 1``: ``jac[:, i, j] = dy_axes[i] / dx_j``.
    hess : array, shape (B, len(axes), d, d)
        Only if ``order >= 2``: ``hess[:, i, j, k] = d^2 y_axes[i] / dx_j dx_k``.
    """
    n, d = x.shape
    axes = list(range(d)) if axes is None else [int(a) for a in axes]
    baxes = [int(j) for j in baxes]
    params = xp.broadcast_to(xp.asarray(params, dtype=xp.float64), (n, d, 4))
    y = xp.empty((n, len(axes)), dtype=xp.float64)
    jac = xp.zeros((n, len(axes), d), dtype=xp.float64) if order >= 1 else None
    hess = xp.zeros((n, len(axes), d, d), dtype=xp.float64) if order >= 2 else None

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for i, a in enumerate(axes):
            kind, lo_src, sc_src, mode = (int(c) for c in codes[a])
            L, dL, ddL = _window_entry(params[:, a, 0], lo_src, bvals, bgrads, bhess, order)
            S, dS, ddS = _window_entry(params[:, a, 1], sc_src, bvals, bgrads, bhess, order)
            if mode == SCALE_IS_UPPER:
                W = S - L
                dW = _difference(dS, dL)
                ddW = _difference(ddS, ddL)
            else:
                W, dW, ddW = S, dS, ddS
            s = (x[:, a] - L) / W
            f, f1, f2 = _transform(kind, s, params[:, a, 2], params[:, a, 3], xp, order)
            y[:, i] = f
            if order == 0:
                continue

            # ds[:, j] = ds/dx_j, nonzero for j = a and the boundary inputs.
            # The boundary has few inputs, so they are indexed one at a time
            # (plain integer indices work the same in NumPy and CuPy).
            dependent = dL is not None or dW is not None
            ds = xp.zeros((n, d), dtype=xp.float64)
            ds[:, a] = 1.0 / W
            if dependent:
                dL_ = 0.0 if dL is None else dL
                dW_ = 0.0 if dW is None else dW
                dsb = -(dL_ + s[:, None] * dW_) / W[:, None]  # (B, db)
                for p, j in enumerate(baxes):
                    ds[:, j] = dsb[:, p]
            jac[:, i] = f1[:, None] * ds
            if order == 1:
                continue

            # dds[:, j, k] = d^2 s / dx_j dx_k
            dds = xp.zeros((n, d, d), dtype=xp.float64)
            if dependent:
                inner = xp.zeros((n, len(baxes), len(baxes)), dtype=xp.float64)
                if ddL is not None:
                    inner = inner + ddL
                if ddW is not None:
                    inner = inner + s[:, None, None] * ddW
                if dW is not None:
                    inner = inner + dsb[:, :, None] * dW[:, None, :] + dW[:, :, None] * dsb[:, None, :]
                    cross = -dW / (W**2)[:, None]
                    for p, j in enumerate(baxes):
                        dds[:, a, j] = cross[:, p]
                        dds[:, j, a] = cross[:, p]
                block = -inner / W[:, None, None]
                for p, j in enumerate(baxes):
                    for q, k in enumerate(baxes):
                        dds[:, j, k] = block[:, p, q]
            hess[:, i] = f2[:, None, None] * ds[:, :, None] * ds[:, None, :] + f1[:, None, None] * dds

    out = (y,)
    if order >= 1:
        out += (jac,)
    if order >= 2:
        out += (hess,)
    return out if order else y


def _window_entry(const, src, bvals, bgrads, bhess, order):
    """Value, gradient (B, db) and Hessian (B, db, db) of a window entry; None for zero derivatives."""
    if src < 0:
        return const, None, None
    if bvals is None:
        raise ValueError("This map uses the boundary table, but no boundary values were given.")
    grad = bgrads[:, :, src] if order >= 1 else None
    hess = bhess[:, :, :, src] if order >= 2 else None
    return bvals[:, src], grad, hess


def _difference(a, b):
    """a - b for derivative arrays where None means zero."""
    if b is None:
        return a
    if a is None:
        return -b
    return a - b


def _transform(kind, s, offset, power, xp, order):
    """f(s), f'(s), f''(s) for a transform code (derivatives None below ``order``)."""
    if kind == TRANSFORMS["identity"]:
        f1 = xp.ones_like(s) if order >= 1 else None
        f2 = xp.zeros_like(s) if order >= 2 else None
        return s, f1, f2
    u = s + offset
    if kind == TRANSFORMS["log"]:
        f = xp.log(u)
        f1 = 1.0 / u if order >= 1 else None
        f2 = -1.0 / u**2 if order >= 2 else None
        return f, f1, f2
    f = u**power
    f1 = power * u ** (power - 1.0) if order >= 1 else None
    f2 = power * (power - 1.0) * u ** (power - 2.0) if order >= 2 else None
    return f, f1, f2


def forward_map(y, codes, params, xp, boundary=None, baxes=()):
    """
    Physical coordinates of table-coordinate points: the inverse of :func:`inverse_map`.

    Parameters
    ----------
    y : array, shape (B, d)
        Points in table coordinates.
    codes, params, xp, baxes
        As for :func:`inverse_map`.
    boundary : callable, optional
        ``boundary(x_b) -> values (B, Fb)``, the boundary table evaluated at
        physical points ``x_b`` of shape (B, db). Needed if any axis uses
        the boundary.

    Returns
    -------
    array, shape (B, d)
    """
    n, d = y.shape
    params = xp.broadcast_to(xp.asarray(params, dtype=xp.float64), (n, d, 4))
    x = xp.empty((n, d), dtype=xp.float64)
    dependent = dependent_axes(codes)
    independent = [a for a in range(d) if a not in dependent]
    if dependent and boundary is None:
        raise ValueError("These maps use the boundary table, but none was given.")
    bvals = None
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        # Axes with constant windows first: the boundary takes them as inputs
        for a in independent + dependent:
            if dependent and a == dependent[0]:
                bvals = boundary(xp.stack([x[:, j] for j in baxes], axis=1))
            kind, lo_src, sc_src, mode = (int(c) for c in codes[a])
            L = params[:, a, 0] if lo_src < 0 else bvals[:, lo_src]
            S = params[:, a, 1] if sc_src < 0 else bvals[:, sc_src]
            W = S - L if mode == SCALE_IS_UPPER else S
            v = y[:, a]
            if kind == TRANSFORMS["identity"]:
                s = v
            elif kind == TRANSFORMS["log"]:
                s = xp.exp(v) - params[:, a, 2]
            else:
                s = v ** (1.0 / params[:, a, 3]) - params[:, a, 2]
            x[:, a] = L + W * s
    return x


# ----------------------------------------------------------- Numba versions


@njit(cache=True)
def _inverse_map_point(codes, params, bout, baxes, x, y, jac, gradient):
    """
    Single-point :func:`inverse_map`: fill ``y`` and, if ``gradient``, the
    dense Jacobian ``jac[a, i] = dy_a / dx_i``. ``bout`` holds the boundary
    values in row 0 and their gradients in rows 1..db.
    """
    d = codes.shape[0]
    db = baxes.shape[0]
    for a in range(d):
        kind = codes[a, 0]
        lo = codes[a, 1]
        sc = codes[a, 2]
        L = params[a, 0] if lo < 0 else bout[0, lo]
        S = params[a, 1] if sc < 0 else bout[0, sc]
        W = S - L if codes[a, 3] == SCALE_IS_UPPER else S
        s = (x[a] - L) / W
        if kind == 0:
            y[a] = s
            f1 = 1.0
        elif kind == 1:
            u = s + params[a, 2]
            y[a] = np.log(u)
            f1 = 1.0 / u
        else:
            u = s + params[a, 2]
            p = params[a, 3]
            y[a] = np.power(u, p)
            f1 = p * np.power(u, p - 1.0)
        if gradient:
            for i in range(d):
                jac[a, i] = 0.0
            jac[a, a] = f1 / W
            if lo >= 0 or sc >= 0:
                for jb in range(db):
                    dL = bout[1 + jb, lo] if lo >= 0 else 0.0
                    dS = bout[1 + jb, sc] if sc >= 0 else 0.0
                    dW = dS - dL if codes[a, 3] == SCALE_IS_UPPER else dS
                    jac[a, baxes[jb]] = -f1 * (dL + s * dW) / W


@njit(cache=True)
def mapped_scratch_size(shape, bcoeffs, bshape):
    """Length of the float64 scratch array the mapped kernels need."""
    d = shape.shape[0]
    m = 0
    for i in range(d):
        m = max(m, shape[i])
    size = d + d * d + d + 2 * m
    db = bshape.shape[0]
    if db > 0:
        mb = 0
        bsize = 1
        for i in range(db):
            mb = max(mb, bshape[i])
            bsize *= bshape[i]
        n_out = bcoeffs.shape[0] // bsize
        rest = bcoeffs.shape[0] // bshape[0]
        size += db + (db + 2 * mb) + (db + 1) * n_out + 2 * (db + 1) * rest
    return size


@njit(cache=True)
def _map_point(codes, params, bcoeffs, bshape, bbounds, baxes, tol, x, scratch, gradient):
    """
    Evaluate the boundary table (if any) and the inverse map at ``x``.

    Fills ``scratch[:d]`` with y and ``scratch[d:d + d*d]`` with the
    Jacobian. Returns the boundary's status, with bits in the main table's
    axis numbering, and the offset of the free part of ``scratch``.
    """
    d = codes.shape[0]
    y = scratch[:d]
    jac = scratch[d:d + d * d].reshape((d, d))
    off = d + d * d
    db = bshape.shape[0]
    status = 0
    if db > 0:
        mb = 0
        bsize = 1
        for i in range(db):
            mb = max(mb, bshape[i])
            bsize *= bshape[i]
        n_out = bcoeffs.shape[0] // bsize
        rest = bcoeffs.shape[0] // bshape[0]
        xb = scratch[off:off + db]
        off += db
        for jb in range(db):
            xb[jb] = x[baxes[jb]]
        bscratch = scratch[off:off + db + 2 * mb]
        off += db + 2 * mb
        bout = scratch[off:off + (db + 1) * n_out].reshape((db + 1, n_out))
        off += (db + 1) * n_out
        bwork = scratch[off:off + 2 * (db + 1) * rest].reshape((2 * (db + 1), rest))
        off += 2 * (db + 1) * rest
        if gradient:
            bstatus = value_and_grad_point(bcoeffs, bshape, bbounds, xb, tol, bscratch, bwork, bout)
        else:
            bstatus = evaluate_point(bcoeffs, bshape, bbounds, xb, tol, bscratch, bwork, bout[0])
        for jb in range(db):
            if bstatus >> jb & 1:
                status |= 1 << baxes[jb]
        _inverse_map_point(codes, params, bout, baxes, x, y, jac, gradient)
    else:
        _inverse_map_point(codes, params, scratch[:0].reshape((1, 0)), baxes, x, y, jac, gradient)
    return status, off


@njit(cache=True)
def _check_mapped_call(coeffs, shape, codes, bcoeffs, bshape, x, scratch, work, rows, gradient):
    """Raise if the arrays passed to the mapped kernels do not fit the table."""
    d = shape.shape[0]
    if x.shape[0] != d or codes.shape[0] != d:
        raise ValueError("x and the maps must have one entry per axis of the table")
    size = 1
    for i in range(d):
        size *= shape[i]
    if rows != coeffs.shape[0] // size:
        raise ValueError("out does not have one entry per output of the table")
    need_rows = 2 * (d + 1) if gradient else 2
    if (scratch.shape[0] < mapped_scratch_size(shape, bcoeffs, bshape) or work.shape[0] < need_rows
            or work.shape[1] < coeffs.shape[0] // shape[0]):
        raise ValueError("workspace too small for this table; make it with table.workspace()")


@njit(cache=True)
def eval_physical_into(coeffs, shape, bounds, tol, codes, params, bcoeffs, bshape, bbounds, baxes,
                       x, scratch, work, out):
    """
    Evaluate a mapped table at a point in physical coordinates, into ``out``.

    The Numba entry point for a :class:`~chebyshevnd.MappedInterpolant`: it
    evaluates the boundary table, maps ``x`` to table coordinates and
    evaluates the table there, without allocating. Call it from your own
    ``@njit`` code with the fields of ``interp.raw()``, as for
    :func:`~chebyshevnd.eval_into`::

        table = H.precontract({"nu": 0.2}).raw()
        ws = table.workspace()

        @njit
        def rhs(t, y, coeffs, shape, bounds, tol, codes, params,
                bcoeffs, bshape, bbounds, baxes, scratch, work, out):
            status = eval_physical_into(coeffs, shape, bounds, tol, codes, params,
                                        bcoeffs, bshape, bbounds, baxes, y, scratch, work, out)
            ...

        solve_ivp(lambda t, y: rhs(t, y, *table, *ws, out), ...)

    Parameters
    ----------
    coeffs, shape, bounds, tol, codes, params, bcoeffs, bshape, bbounds, baxes
        The fields of a :class:`RawMappedTable`.
    x : 1D float64 array
        The point in physical coordinates.
    scratch, work
        The fields of a :class:`~chebyshevnd.Workspace` from
        ``table.workspace()``.
    out : 1D array, shape (F,)
        Receives the outputs; same dtype as the table.

    Returns
    -------
    int
        Status bitmask: bit i is set when the table coordinate of axis i
        lies outside the table's domain (beyond ``tol``) or is NaN (e.g. the
        log of a negative number), or when coordinate i lies outside the
        boundary table's domain. 0 means inside.
    """
    _check_mapped_call(coeffs, shape, codes, bcoeffs, bshape, x, scratch, work, out.shape[0], False)
    status, off = _map_point(codes, params, bcoeffs, bshape, bbounds, baxes, tol, x, scratch, False)
    d = shape.shape[0]
    status |= evaluate_point(coeffs, shape, bounds, scratch[:d], tol, scratch[off:], work, out)
    return status


@njit(cache=True)
def eval_grad_physical_into(coeffs, shape, bounds, tol, codes, params, bcoeffs, bshape, bbounds,
                            baxes, x, scratch, work, out):
    """
    Evaluate a mapped table and its gradient in physical coordinates, into ``out``.

    As :func:`eval_physical_into`, plus the gradient with respect to the
    physical coordinates: the table's gradient (from
    :func:`~chebyshevnd.eval_grad_into`) times the Jacobian of the map,
    including the dependence of the boundary functions on other axes.

    Parameters
    ----------
    coeffs, shape, bounds, tol, codes, params, bcoeffs, bshape, bbounds, baxes, x, scratch, work
        As for :func:`eval_physical_into`.
    out : 2D array, shape (d + 1, F)
        Row 0 receives the outputs, row i + 1 their derivatives with respect
        to physical coordinate i.

    Returns
    -------
    int
        Status bitmask, as for :func:`eval_physical_into`.
    """
    if out.shape[0] != shape.shape[0] + 1:
        raise ValueError("out must have shape (d + 1, F)")
    _check_mapped_call(coeffs, shape, codes, bcoeffs, bshape, x, scratch, work, out.shape[1], True)
    status, off = _map_point(codes, params, bcoeffs, bshape, bbounds, baxes, tol, x, scratch, True)
    d = shape.shape[0]
    status |= value_and_grad_point(coeffs, shape, bounds, scratch[:d], tol, scratch[off:], work, out)

    # Chain rule in place. The Jacobian is diagonal except for the columns of
    # the boundary inputs, which have constant windows themselves: update
    # those rows first (they read the dependent rows), then scale the rest.
    jac = scratch[d:d + d * d].reshape((d, d))
    n_out = out.shape[1]
    for i in range(d):
        if codes[i, 1] < 0 and codes[i, 2] < 0:
            for f in range(n_out):
                g = out[1 + i, f] * jac[i, i]
                for a in range(d):
                    if a != i and jac[a, i] != 0.0:
                        g += out[1 + a, f] * jac[a, i]
                out[1 + i, f] = g
    for a in range(d):
        if codes[a, 1] >= 0 or codes[a, 2] >= 0:
            for f in range(n_out):
                out[1 + a, f] *= jac[a, a]
    return status


class RawMappedTable(NamedTuple):
    """
    A mapped interpolant as plain arrays, for compiled code.

    Returned by :meth:`MappedInterpolant.raw`. The fields are the first ten
    arguments of :func:`eval_physical_into` and
    :func:`eval_grad_physical_into`, in order, so from Python
    ``eval_physical_into(*table, x, *ws, out)`` works. As for
    :class:`~chebyshevnd.RawTable`, pass the fields to your own ``@njit``
    functions as separate arguments.

    Attributes
    ----------
    coeffs, shape, bounds, tol
        The table, as in :class:`~chebyshevnd.RawTable`.
    codes, params : np.ndarray
        The packed maps, shape (d, 4) each (see :func:`pack_maps`).
    bcoeffs, bshape, bbounds : np.ndarray
        The boundary table, as in :class:`~chebyshevnd.RawTable`; ``bshape``
        is empty when there is none.
    baxes : np.ndarray
        For each boundary input, the index of the corresponding axis.
    """

    coeffs: np.ndarray
    shape: np.ndarray
    bounds: np.ndarray
    tol: float
    codes: np.ndarray
    params: np.ndarray
    bcoeffs: np.ndarray
    bshape: np.ndarray
    bbounds: np.ndarray
    baxes: np.ndarray

    @property
    def ndim(self):
        """Number of axes d."""
        return int(self.shape.shape[0])

    @property
    def n_outputs(self):
        """Number of outputs F."""
        return int(self.coeffs.shape[0] // np.prod(self.shape))

    @property
    def dtype(self):
        """Coefficient dtype, and the dtype of the ``out`` arrays."""
        return self.coeffs.dtype

    def workspace(self):
        """
        Allocate scratch space for evaluating this table.

        Returns
        -------
        Workspace
            Serves both :func:`eval_physical_into` and
            :func:`eval_grad_physical_into`; one per thread.
        """
        d = self.ndim
        rest = self.coeffs.shape[0] // int(self.shape[0])
        return Workspace(np.empty(mapped_scratch_size(self.shape, self.bcoeffs, self.bshape)),
                         np.empty((2 * (d + 1), rest), dtype=self.coeffs.dtype))
