import numpy as np
from scipy.fft import dct
from .clenshaw import clenshaw_1d
from .utils import chebyshev_nodes
def compute_chebyshev_coefficients_1d(func_or_vals, a, b, N, kind='dct'):
    """
    Compute the Chebyshev coefficients of a function or discrete values on [-1, 1] scaled to [a, b].
    
    Parameters:
        func_or_vals: Callable or np.ndarray
            Either a function f(x), or an array of y-values defined at Chebyshev points.
        a, b: float
            Domain bounds [a, b] over which interpolation is defined.
        N: int
            Number of Chebyshev points.
        kind: str
            'dct' (default) uses SciPy DCT-I to compute coefficients.

    Returns:
        coeffs: np.ndarray
            Chebyshev coefficients.
    """

    if callable(func_or_vals):
        x= chebyshev_nodes(N,a,b)
        y = func_or_vals(x)
    else:
        y = np.asarray(func_or_vals)

    # Compute Chebyshev coefficients using DCT-I (type 1)
    coeffs = dct(y, type=2) / N 
    coeffs[0] *= 0.5

    return coeffs


class ChebyshevInterpolant1D:
    def __init__(self, func_or_vals, a, b, N, nodes = 'Chebyshev'):
        """
        Construct a 1D Chebyshev interpolant.

        Parameters:
            func_or_vals: Callable or array
                Function f(x) or array of values at Chebyshev nodes.
            a, b: float
                Domain bounds [a, b].
            N: int
                Number of Chebyshev points.
        """
        self.a = a
        self.b = b
        self.N = N

        if callable(func_or_vals):
            self.x_nodes = chebyshev_nodes(N, a, b)
            y = func_or_vals(self.x_nodes)
        else:
            y = np.asarray(func_or_vals)

        # To make this work with end points, use type=1, divide by N-1 and 0.5*coeff[-1]
        if nodes == 'Chebyshev':
        
            self.coeffs = dct(y, type=2) / N
            self.coeffs[0] *= 0.5
        elif nodes == 'GaussLobatto':
            self.coeffs = dct(y, type=1, norm = None) /(N-1)
            self.coeffs[0] *= 0.5
            self.coeffs[-1] *= 0.5
        else:
            ValueError('Incorrect node type. Please choose either Chebyshev or GaussLobatto')

        self.absErr = max(abs(self.coeffs[-1]), abs(self.coeffs[-2]))
        self.maxCoeff = max(abs(self.coeffs))
        self.relErr = self.absErr/self.maxCoeff
        self.k_max = len(self.coeffs) - 1

    def get_coefficients(self):
        """
        Get the Chebyshev coefficients of the interpolant.
        
        Returns:
            np.ndarray: Chebyshev coefficients.
        """
        return self.coeffs
    def absolute_error(self):
        """
        Compute the absolute error of the interpolant at point x.
        
        Returns:
            float: Absolute error at the Chebyshev nodes.
        """ 


        return self.absErr

    def relative_error(self):
        """
        Compute the relative error of the interpolant at point x.
        
        Returns:
            float: Relative error at the Chebyshev nodes.
        """
        return self.relErr
        

    def evaluate(self, x):
        """Evaluate the interpolant at x (scalar or array)."""
        x = np.atleast_1d(x)
        return np.array([clenshaw_1d(self.coeffs,xi,self.k_max, self.a, self.b) for xi in x])
    
    def optimize(self, rel_tol=1e-6):
        """
        Optimize the Chebyshev coefficients by removing small ones.
        
        Parameters:
            rel_tol: float
                Relative tolerance for coefficient significance.
        
        Returns:
            np.ndarray: Positions of significant coefficients.
        """
        if rel_tol < self.relErr:
            raise ValueError("Relative tolerance must be greater than the relative error of the interpolant.")
        
        k_max = len(self.coeffs) - 1
        for k in range(len(self.coeffs) - 1, -1, -1):
            if abs(self.coeffs[k]) > rel_tol * self.maxCoeff:
                k_max = k
                break
                
        self.k_max = k_max