"""
Numba kernels for evaluating compact tensor-product Chebyshev series.

A table is the C-contiguous coefficient array of shape (k_1, ..., k_d, F),
passed flattened, holding F independent outputs along the trailing axis.
Evaluation contracts the leading axis first: each coefficient slice is added
to a running buffer with one axpy, so memory is read contiguously and the
output axis is never contracted. Each contraction shrinks the data by a
factor k_i, so the first one dominates the cost.

The public entry points for user code are :func:`eval_into` and
:func:`eval_grad_into`, which can be called from inside the user's own
``@njit`` functions, e.g. an ODE right-hand side. They take plain arrays: the
fields of a :class:`RawTable` (from ``ChebyshevInterpolantND.raw()``), the
point, the fields of a :class:`Workspace`, and the output array.
"""

from typing import NamedTuple

import numpy as np
from numba import njit


@njit(cache=True)
def chebyshev_basis(n, t, out):
    """
    Fill ``out[:n]`` with T_0(t), ..., T_{n-1}(t) using the three-term recurrence.

    The recurrence is stable on [-1, 1]. Unlike ``cos(k * arccos(t))``, it
    keeps full accuracy near t = +-1.
    """
    out[0] = 1.0
    if n > 1:
        out[1] = t
        for k in range(2, n):
            out[k] = 2.0 * t * out[k - 1] - out[k - 2]


@njit(cache=True)
def chebyshev_basis_and_derivative(n, t, basis, deriv):
    """
    Fill ``basis[:n]`` with T_k(t) and ``deriv[:n]`` with T'_k(t), k < n.

    Uses ``T'_{k+1} = 2 T_k + 2 t T'_k - T'_{k-1}`` alongside the recurrence
    for T_k, so there is no division and t = +-1 needs no special case.
    """
    basis[0] = 1.0
    deriv[0] = 0.0
    if n > 1:
        basis[1] = t
        deriv[1] = 1.0
        for k in range(2, n):
            basis[k] = 2.0 * t * basis[k - 1] - basis[k - 2]
            deriv[k] = 2.0 * basis[k - 1] + 2.0 * t * deriv[k - 1] - deriv[k - 2]


@njit(cache=True)
def _contract_axis(src, n, rest, basis, dst):
    """
    ``dst[:rest] = sum_k basis[k] * src[k*rest:(k+1)*rest]`` for k < n.

    Used only on the small partial sums of the gradient kernel: inlining the
    same loop by hand is ~10% faster on the large first pass, so the hot
    loops in :func:`contract` and :func:`value_and_grad_point` are written out.
    """
    for j in range(rest):
        dst[j] = src[j]
    for k in range(1, n):
        tk = basis[k]
        off = k * rest
        for j in range(rest):
            dst[j] += tk * src[off + j]


@njit(cache=True)
def _map_to_box(bounds, xs, tol, ts):
    """Map ``xs`` from its box to [-1, 1]^d in ``ts``; return the status bitmask."""
    status = 0
    for i in range(ts.shape[0]):
        a = bounds[0, i]
        b = bounds[1, i]
        t = (2.0 * xs[i] - a - b) / (b - a)
        if not abs(t) <= 1.0 + tol:
            status |= 1 << i
        ts[i] = t
    return status


@njit(cache=True)
def contract(coeffs, shape, ts, basis, work_a, work_b, out):
    """
    Evaluate a flattened (k_1, ..., k_d, F) table at a point in [-1, 1]^d.

    Parameters
    ----------
    coeffs : 1D array
        The C-contiguous table, flattened; float64 or complex128.
    shape : 1D int64 array
        (k_1, ..., k_d). F is inferred as ``coeffs.size // prod(shape)``.
    ts : 1D float64 array
        The point, one coordinate per axis, already mapped to [-1, 1].
    basis : 1D float64 array
        Scratch space of length >= max(shape).
    work_a, work_b : 1D arrays
        Scratch space, same dtype as ``coeffs``, each of length
        >= k_2 * ... * k_d * F. They alternate as source and destination of
        successive contractions; keeping them as separate arrays (rather than
        halves of one) lets the compiler vectorise the inner loop.
    out : 1D array
        Receives the F output values; same dtype as ``coeffs``.
    """
    d = shape.shape[0]
    rest = coeffs.shape[0]
    src = coeffs
    for ax in range(d):
        n = shape[ax]
        rest //= n
        chebyshev_basis(n, ts[ax], basis)
        # Alternate buffers so the source of one pass is never its destination
        dst = work_a if ax % 2 == 0 else work_b
        for j in range(rest):
            dst[j] = src[j]
        for k in range(1, n):
            tk = basis[k]
            off = k * rest
            for j in range(rest):
                dst[j] += tk * src[off + j]
        src = dst
    for f in range(out.shape[0]):
        out[f] = src[f]


@njit(cache=True)
def evaluate_point(coeffs, shape, bounds, xs, tol, scratch, work, out):
    """
    Map ``xs`` from its box to [-1, 1]^d and evaluate the table there.

    Parameters
    ----------
    coeffs, shape, out
        As for :func:`contract`.
    work : 2D array, shape (2, k_2 * ... * k_d * F)
        Scratch space, same dtype as ``coeffs``; its rows are
        :func:`contract`'s two work buffers.
    bounds : 2D float64 array, shape (2, d)
        Lower bounds in row 0, upper bounds in row 1.
    xs : 1D float64 array
        The point, one coordinate per axis.
    tol : float
        Allowed overshoot outside the domain, as a fraction of each axis's
        half-width.
    scratch : 1D float64 array
        Scratch space of length >= d + max(shape).

    Returns
    -------
    int
        Status bitmask: bit i is set when coordinate i lies outside its
        domain by more than ``tol`` (in units of the half-width), or is NaN.
        0 means the point is inside the domain. The output is computed
        either way (it extrapolates outside the domain); it is up to the
        caller to act on the status.
    """
    d = shape.shape[0]
    ts = scratch[:d]
    status = _map_to_box(bounds, xs, tol, ts)
    contract(coeffs, shape, ts, scratch[d:], work[0], work[1], out)
    return status


@njit(cache=True)
def value_and_grad_point(coeffs, shape, bounds, xs, tol, scratch, work, out):
    """
    Evaluate a table and its gradient at one point, in a single pass over it.

    Parameters
    ----------
    coeffs, shape, bounds, xs, tol
        As for :func:`evaluate_point`.
    scratch : 1D float64 array
        Scratch space of length >= d + 2 * max(shape).
    work : 2D array, shape (2 * (d + 1), k_2 * ... * k_d * F)
        Scratch space, same dtype as ``coeffs``.
    out : 2D array, shape (d + 1, F)
        Receives the values in row 0 and the derivative with respect to
        coordinate i (in the domain's units) in row i + 1.

    Returns
    -------
    int
        Status bitmask, as for :func:`evaluate_point`.

    Notes
    -----
    The contraction keeps one partial sum per "branch": the value, and the
    derivative along each axis contracted so far. Contracting axis i
    turns the value branch into two (with T and with T'), in one pass over
    its data, and contracts every existing derivative branch with T. Axis 0
    is therefore read once, with two accumulators, and the extra branches
    only touch the much smaller partial sums. The cost is about 1.0-1.4x a
    value evaluation.
    """
    d = shape.shape[0]
    m = 0
    for i in range(d):
        m = max(m, shape[i])
    ts = scratch[:d]
    basis = scratch[d:d + m]
    deriv = scratch[d + m:d + 2 * m]
    status = _map_to_box(bounds, xs, tol, ts)

    # Branch b (0 = value, i + 1 = derivative along axis i) alternates
    # between work rows 2b and 2b + 1, by the parity of the axis
    rest = coeffs.shape[0]
    for ax in range(d):
        n = shape[ax]
        rest //= n
        chebyshev_basis_and_derivative(n, ts[ax], basis, deriv)
        p = ax % 2
        q = 1 - p
        for branch in range(1, ax + 1):
            _contract_axis(work[2 * branch + q], n, rest, basis, work[2 * branch + p])
        src = coeffs if ax == 0 else work[q]
        dst = work[p]
        dst_deriv = work[2 * (ax + 1) + p]
        for j in range(rest):
            v = src[j]
            dst[j] = v
            dst_deriv[j] = 0.0 * v  # T'_0 = 0
        for k in range(1, n):
            tk = basis[k]
            dk = deriv[k]
            off = k * rest
            for j in range(rest):
                v = src[off + j]
                dst[j] += tk * v
                dst_deriv[j] += dk * v

    p = (d - 1) % 2
    n_out = out.shape[1]
    for f in range(n_out):
        out[0, f] = work[p, f]
    for i in range(d):
        scale = 2.0 / (bounds[1, i] - bounds[0, i])
        row = work[2 * (i + 1) + p]
        for f in range(n_out):
            out[i + 1, f] = scale * row[f]
    return status


# ---------------------------------------------------------------- public API


class RawTable(NamedTuple):
    """
    An interpolant's compact table as plain arrays, for compiled code.

    Returned by :meth:`ChebyshevInterpolantND.raw`. Its fields are the first
    four arguments of :func:`eval_into` and :func:`eval_grad_into`, in
    order, so from Python ``eval_into(*table, x, *ws, out)`` works.

    Pass the fields to your own ``@njit`` functions as separate arguments,
    not the tuple itself: Numba checks the types of array arguments in
    compiled code, but of tuples in Python, which adds about 1.5 us to every
    call from Python (e.g. from ``scipy.integrate.solve_ivp``). Star-unpacking
    a ``RawTable`` inside ``@njit`` code is not supported either.

    The arrays are read-only views of the interpolant's arrays, taken at the
    time of the call: a later ``optimize()`` on the interpolant does not
    change them.

    Attributes
    ----------
    coeffs : np.ndarray
        The C-contiguous coefficient tensor of shape (k_1, ..., k_d, F),
        flattened; float64 or complex128.
    shape : np.ndarray
        (k_1, ..., k_d), int64.
    bounds : np.ndarray
        Shape (2, d): lower bounds in row 0, upper bounds in row 1.
    tol : float
        Allowed overshoot outside the domain, as a fraction of each axis's
        half-width, before a coordinate is flagged in the status.
    """

    coeffs: np.ndarray
    shape: np.ndarray
    bounds: np.ndarray
    tol: float

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

        One workspace serves both :func:`eval_into` and
        :func:`eval_grad_into`. Each thread evaluating at the same time needs
        its own.

        Returns
        -------
        Workspace
        """
        d = self.ndim
        rest = self.coeffs.shape[0] // int(self.shape[0])
        return Workspace(np.empty(d + 2 * int(self.shape.max())),
                         np.empty((2 * (d + 1), rest), dtype=self.coeffs.dtype))


class Workspace(NamedTuple):
    """
    Scratch space for :func:`eval_into` and :func:`eval_grad_into`.

    Made by :meth:`RawTable.workspace`, which sizes it for its table. Its
    fields are the two arguments after ``x``.

    Attributes
    ----------
    scratch : np.ndarray
        float64, length >= d + 2 * max(k).
    work : np.ndarray
        Shape (2 * (d + 1), k_2 * ... * k_d * F), the table's dtype.
    """

    scratch: np.ndarray
    work: np.ndarray


@njit(cache=True)
def _check_call(coeffs, shape, x, scratch, work, rows, gradient):
    """Raise if the arrays passed to eval_into / eval_grad_into do not fit the table."""
    d = shape.shape[0]
    if x.shape[0] != d:
        raise ValueError("x must have one coordinate per axis of the table")
    m = 0
    size = 1
    for i in range(d):
        m = max(m, shape[i])
        size *= shape[i]
    if rows != coeffs.shape[0] // size:
        raise ValueError("out does not have one entry per output of the table")
    need_rows = 2 * (d + 1) if gradient else 2
    need_scratch = d + 2 * m if gradient else d + m
    if (scratch.shape[0] < need_scratch or work.shape[0] < need_rows
            or work.shape[1] < coeffs.shape[0] // shape[0]):
        raise ValueError("workspace too small for this table; make it with table.workspace()")


@njit(cache=True)
def eval_into(coeffs, shape, bounds, tol, x, scratch, work, out):
    """
    Evaluate a raw table at one point, writing the outputs into ``out``.

    Compiled with Numba, so it can be called from the user's own ``@njit``
    code, such as an ODE right-hand side, without returning to Python. It
    allocates nothing. For example::

        table = interp.precontract({"nu": 0.2}).raw()
        ws = table.workspace()
        out = np.empty(table.n_outputs, dtype=table.dtype)

        @njit
        def rhs(t, y, coeffs, shape, bounds, tol, scratch, work, out):
            status = eval_into(coeffs, shape, bounds, tol, y, scratch, work, out)
            ...

        solve_ivp(lambda t, y: rhs(t, y, *table, *ws, out), ...)

    Parameters
    ----------
    coeffs, shape, bounds, tol
        The fields of a :class:`RawTable`, from
        :meth:`ChebyshevInterpolantND.raw`.
    x : 1D float64 array
        The point, one coordinate per axis, in the domain's units.
    scratch, work
        The fields of a :class:`Workspace`, from ``table.workspace()``.
    out : 1D array, shape (F,)
        Receives the outputs; same dtype as the table (complex128 for a
        complex table).

    Returns
    -------
    int
        Status bitmask: bit i is set when coordinate i lies outside its
        domain (beyond ``tol``) or is NaN; 0 means inside. ``out`` is filled
        either way, with an extrapolation for points outside, so act on the
        status rather than the values.

    Raises
    ------
    ValueError
        If ``x``, the workspace or ``out`` does not have the right size for
        the table.
    """
    _check_call(coeffs, shape, x, scratch, work, out.shape[0], False)
    return evaluate_point(coeffs, shape, bounds, x, tol, scratch, work, out)


@njit(cache=True)
def eval_grad_into(coeffs, shape, bounds, tol, x, scratch, work, out):
    """
    Evaluate a raw table and its gradient at one point, into ``out``.

    The gradient along every axis comes from the same pass over the table
    as the value, at about 1.0-1.4x the cost of :func:`eval_into`. It is the
    exact gradient of the interpolating polynomial, so a vector field built
    from it (e.g. Hamilton's equations for an interpolated Hamiltonian) is
    exactly the one generated by the interpolant.

    Parameters
    ----------
    coeffs, shape, bounds, tol, x, scratch, work
        As for :func:`eval_into`.
    out : 2D array, shape (d + 1, F)
        Row 0 receives the outputs, row i + 1 their derivatives with respect
        to coordinate i (in the domain's units, not [-1, 1]). Same dtype as
        the table.

    Returns
    -------
    int
        Status bitmask, as for :func:`eval_into`.

    Raises
    ------
    ValueError
        If ``x``, the workspace or ``out`` does not have the right size for
        the table.
    """
    if out.shape[0] != shape.shape[0] + 1:
        raise ValueError("out must have shape (d + 1, F)")
    _check_call(coeffs, shape, x, scratch, work, out.shape[1], True)
    return value_and_grad_point(coeffs, shape, bounds, x, tol, scratch, work, out)
