import numpy as np

def chebyshev_gauss_lobatto_nodes(N,a,b):
    """
    Generate Chebyshev-Gauss-Lobatto nodes for N points.
    This includes the endpoints a and b.
    Parameters:
        N: int
            Number of Chebyshev points.
        a, b: float
            Domain bounds [a, b].
    
    Returns:
        x: np.ndarray
            Chebyshev-Gauss-Lobatto nodes in the interval [a, b].
    """
    # Chebyshev-Gauss-Lobatto nodes on [-1, 1]
    x_cheb = -np.cos(np.pi * np.arange(N) / (N - 1))
    # Rescale to [a, b]
    x = 0.5 * (b - a) * x_cheb + 0.5 * (a + b)
    return x

def chebyshev_nodes(N,a,b):
    """
    Generate Chebyshev nodes for N points.
    This excludes the endpoints a and b.
    Parameters:
        N: int
            Number of Chebyshev points.
        a, b: float
            Domain bounds [a, b].
    
    Returns:
        x: np.ndarray
            Chebyshev nodes in the interval [a, b].
    """
    # Chebyshev nodes on [-1, 1]
    x_cheb = np.cos(np.pi * (2*np.arange(N)+1) / (2*N))
    # Rescale to [a, b]
    x = 0.5 * (b - a) * x_cheb + 0.5 * (a + b)
    return x


# Adaptive Clenshaw algorithm for 1D Chebyshev polynomial evaluation
# Eliminating insignificant coefficients to speed up the evaluation


def find_significant_coeffs_1d(coefficients, rel_tol=1e-8, max_coeff=None, x_max_minus_i=1.):

    if max_coeff is None:
        max_coeff = np.max(np.abs(coefficients))

    nymax = 0
    # Find the last significant coefficient
    n = len(coefficients)
    for i in range(n - 1, -1, -1):
        if abs(coefficients[i]) > (rel_tol * max_coeff) /x_max_minus_i/(n):
            nymax = i
            break
    else:
        return 0  # If all coefficients are below tol, return 0

    return nymax


# Adaptive Clenshaw algorithm for 2D Chebyshev polynomial evaluation

def find_significant_coeffs_2d(coefficients, rel_tol=1e-8,max_coeff=None, x_max_minus_i=1.):

    nx, ny = coefficients.shape
    tmp = np.zeros(nx)
    nyMax = np.zeros(nx)

    if coefficients.ndim != 2:
        raise ValueError("Input coefficients must be a 2D array.")

    if rel_tol <= 0:
        raise ValueError("Relative tolerance must be positive.")
    if max_coeff is None:
        max_coeff = np.max(np.abs(coefficients))


    # Evaluate in the y-direction first
    for i in range(nx):
        nyMax[i] = find_significant_coeffs_1d(coefficients[i, :], rel_tol / x_max_minus_i ,max_coeff=max_coeff, x_max_minus_i = nx)

    # Trim off the trailing zeros
    last_nz = np.max(np.nonzero(nyMax))
    nyMaxTrimmed = nyMax[:last_nz + 1]

    return nyMaxTrimmed.astype(np.int64)