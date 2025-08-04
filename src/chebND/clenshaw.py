
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
    for k in range(k_max, 0, -1):
        b_k = 2 * xt * b_k1 - b_k2 + coeffs[k]
        b_k2 = b_k1
        b_k1 = b_k
    return coeffs[0] + xt * b_k1 - b_k2