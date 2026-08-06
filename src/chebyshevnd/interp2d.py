import numpy as np
from .clenshaw import clenshaw_1d, clenshaw_2d
from .utils import find_significant_coeffs_2d, compute_chebyshev_coefficients_nd

class ChebyshevInterpolant2D:
    def __init__(self, vals, ax = -1., bx = 1., ay = -1., by = 1., nodes = 'Chebyshev'):
        """
        Initialize a 2D Chebyshev interpolant.

        Parameters:
            func_or_vals : callable or array-like
                Function f(x, y) to interpolate, or 2D array of values at Chebyshev nodes.
            ax, bx : float
                Domain bounds in x.
            ay, by : float
                Domain bounds in y.
            Nx, Ny : int
                Number of Chebyshev points in x and y.
        """
        self.Nx, self.Ny = vals.shape
        self.ax, self.bx = ax, bx
        self.ay, self.by = ay, by

        

        # Calculate the coefficients and store them
        self.coeffs = compute_chebyshev_coefficients_nd(vals,nodes = nodes)

        # Calculate absolute and relative errors
        
        self.abs_error = self.absolute_error()
        self.max = np.max([np.abs(self.coeffs)])
        self.rel_error = self.abs_error / self.max
        self.NyMaxes = np.ones(self.Nx, dtype=np.int64) * self.Ny

    def evaluate(self, x, y):

        return clenshaw_2d(self.coeffs, x, y, NyMaxes =self.NyMaxes, ax =self.ax, bx =self.bx, ay =self.ay, by =self.by)
    
    def get_coefficients(self):
        """
        Get the Chebyshev coefficients of the interpolant.

        Returns:
            np.ndarray: 2D array of Chebyshev coefficients.
        """
        return self.coeffs
    
    def absolute_error(self):
        """
        Compute the absolute error of the interpolant at the Chebyshev nodes.

        Returns:
            float: Absolute error at the Chebyshev nodes.
        """
        err = 0.0
        for axis, N in enumerate(self.coeffs.shape):
            c = np.moveaxis(self.coeffs, axis, 0)
            err = max(err, np.max(np.abs(c[-2])) * (N - 1), np.max(np.abs(c[-1])) * N)
        return err

    def relative_error(self):
        """
        Compute the relative error of the interpolant at the Chebyshev nodes.

        Returns:
            float: Relative error at the Chebyshev nodes.
        """
        return self.rel_error
    
    def optimize(self, rel_tol=1e-8):

        self.NyMaxes = find_significant_coeffs_2d(self.coeffs, rel_tol=rel_tol,max_coeff=self.max)

    def effective_grid(self):
        """
        Get the effective grid size based on significant coefficients.

        Returns:
            np.ndarray: Array of significant NyMaxes for each x.
        """
        return [len(self.NyMaxes),max(self.NyMaxes).item()]

