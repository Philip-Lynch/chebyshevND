"""
Numba kernels for evaluating compact tensor-product Chebyshev series.

A table is the C-contiguous coefficient array of shape (k_1, ..., k_d, F),
passed flattened, holding F independent outputs along the trailing axis.
Evaluation contracts the leading axis first: each coefficient slice is added
to a running buffer with one axpy, so memory is read contiguously and the
output axis is never contracted. Each contraction shrinks the data by a
factor k_i, so the first one dominates the cost.
"""

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
    status = 0
    for i in range(d):
        a = bounds[0, i]
        b = bounds[1, i]
        t = (2.0 * xs[i] - a - b) / (b - a)
        if not abs(t) <= 1.0 + tol:
            status |= 1 << i
        ts[i] = t
    contract(coeffs, shape, ts, scratch[d:], work[0], work[1], out)
    return status
