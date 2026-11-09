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
        >= k_2 * ... * k_d * F.
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
def evaluate_point(coeffs, shape, a, b, xs, ts, basis, work_a, work_b, out):
    """
    Map ``xs`` from the box [a, b] to [-1, 1]^d and evaluate the table there.

    ``ts`` is scratch space of length d; the other arguments are as for
    :func:`contract`.
    """
    d = shape.shape[0]
    for i in range(d):
        ts[i] = (2.0 * xs[i] - a[i] - b[i]) / (b[i] - a[i])
    contract(coeffs, shape, ts, basis, work_a, work_b, out)
