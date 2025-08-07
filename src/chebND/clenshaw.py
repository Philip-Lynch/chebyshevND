
import numpy as np
from numba import njit

@njit
def clenshaw_1d(coeffs, x,  k_max=None, a=-1.0, b=1.0):
    """
    Evaluate Chebyshev series at a single point x using Clenshaw's algorithm.
    
    Parameters:
        coeffs: array of Chebyshev coefficients
        x: point of evaluation in [a, b]
        a, b: domain bounds (default [-1, 1])
        k_max : int, optional
            Number of coefficients to use (truncate at this order).
            Defaults to len(coeffs).
    
    Returns:
        interpolated value at x
    """
    if k_max is None:
        k_max = len(coeffs)
    
    
    # Map x from [a, b] to [-1, 1]
    xt = (2.0 * x - a - b) / (b - a)
    
    b_k1 = 0.0
    b_k2 = 0.0
    for k in range(k_max-1, 0, -1):
        b_k = 2 * xt * b_k1 - b_k2 + coeffs[k]
        b_k2 = b_k1
        b_k1 = b_k
    return coeffs[0] + xt * b_k1 - b_k2

@njit
def clenshaw_2d(coeffs, x, y, NyMaxes = None, ax = -1., bx=1, ay=-1., by=1):
    """
    Evaluate a 2D Chebyshev interpolant using the Clenshaw algorithm.

    Parameters:

        coeffs : 2D np.ndarray
            Matrix of Chebyshev coefficients [Nx, Ny].
        x, y : float
            Point to evaluate.
        Nx : int, optional
            Number of Chebyshev points in x-direction. If None, defaults to shape(coeffs)[0].
        Ny : int, optional
            Number of Chebyshev points in y-direction. If None, defaults to shape(coeffs)[1].
        ax, bx : float
            Domain bounds in x-direction.
        ay, by : float
            Domain bounds in y-direction.

    Returns:
        float
            Value of interpolant at (x, y)
    """
    if NyMaxes is None:
        NyMaxes = np.ones(coeffs.shape[0], dtype=np.int64) * coeffs.shape[1]

    # Map x, y from [ax, bx] and [ay, by] to [-1, 1]
    tx = (2 * x - (bx + ax)) / (bx - ax)
    ty = (2 * y - (by + ay)) / (by - ay)

    # Evaluate in y-direction first for each fixed x
    temp = np.array([clenshaw_1d(coeffs[i, :], ty,NyMaxes[i]) for i in range(NyMaxes.shape[0])])
    
    # Then evaluate resulting 1D Chebyshev series in x-direction
    return clenshaw_1d(temp, tx,NyMaxes.shape[0])