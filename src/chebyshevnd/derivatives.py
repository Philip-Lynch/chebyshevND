"""Differentiation of tensor Chebyshev series in coefficient space."""

import numpy as np


def differentiate_coefficients(coeffs, axis, order=1):
    """
    Coefficients of the derivative of a Chebyshev series along one axis.

    The derivative is with respect to the coordinate t in [-1, 1]. To
    differentiate with respect to x in [a, b], multiply the result by
    ``(2 / (b - a)) ** order``.

    Uses the backward recurrence ``c'_{k-1} = c'_{k+1} + 2 k c_k`` (with
    ``c'_{n-1} = c'_n = 0``) and halves ``c'_0``, applied to every fibre along
    ``axis`` at once. Differentiation lowers the degree by one, so the result
    keeps the input's shape with its last ``order`` coefficients along
    ``axis`` equal to zero.

    Parameters
    ----------
    coeffs : array-like
        Chebyshev coefficients, real or complex, any number of axes.
    axis : int
        Axis to differentiate along.
    order : int, optional
        Number of derivatives to take (>= 0).

    Returns
    -------
    np.ndarray
        Coefficients of the derivative, same shape as ``coeffs``.
    """
    order = int(order)
    if order < 0:
        raise ValueError(f"order must be non-negative, got {order}.")
    c = np.moveaxis(np.asarray(coeffs), axis, 0)
    c = c.astype(np.result_type(c, np.float64), copy=True)
    for _ in range(order):
        c = _differentiate_leading_axis(c)
    return np.moveaxis(c, 0, axis)


def _differentiate_leading_axis(c):
    """One derivative along axis 0 of a coefficient array."""
    n = c.shape[0]
    d = np.zeros_like(c)
    if n < 2:
        return d
    d[n - 2] = 2 * (n - 1) * c[n - 1]
    for k in range(n - 2, 0, -1):
        d[k - 1] = d[k + 1] + 2 * k * c[k]
    d[0] *= 0.5
    return d
