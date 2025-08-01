import numpy as np

def sort_2d_by_first_column_desc(arr):
    """
    Sort a 2D numpy array by the first column in descending order.
    """
    return arr[arr[:, 0].argsort()[::-1]]

def read_1d_data_from_file(filename):
    """
    Read a 2-column .dat file (x, y) into a 2D numpy array.
    """
    try:
        data = np.loadtxt(filename)
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        return data
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return np.array([])

def find_significant_coeffs_1d(coeffs, tol=1e-12):
    """
    Find the index of the last significant Chebyshev coefficient.
    """
    for i in range(len(coeffs)-1, -1, -1):
        if abs(coeffs[i]) > tol:
            return i
    return 0

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
    x_cheb = - np.cos(np.pi * (2*np.arange(N)+1) / (2*N))
    # Rescale to [a, b]
    x = 0.5 * (b - a) * x_cheb + 0.5 * (a + b)
    return x