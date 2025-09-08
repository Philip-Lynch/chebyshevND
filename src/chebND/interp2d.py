import numpy as np
from scipy.fft import dctn
from chebND.clenshaw import clenshaw_1d, clenshaw_2d
from chebND.utils import chebyshev_nodes, find_significant_coeffs_2d, compute_chebyshev_coefficients_nd

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

    def compute_coefficients(self,func_or_vals,Nx, Ny, ax, bx, ay, by, nodes='Chebyshev'):
        """
        Compute the 2D Chebyshev coefficients of a function or discrete values
        over [ax, bx] × [ay, by].

        Parameters:
            func_or_vals : callable or 2D np.ndarray
                Either a function f(x, y), or a 2D array of values sampled at Chebyshev points.
            ax, bx : float
                Domain bounds in x-direction.
            ay, by : float
                Domain bounds in y-direction.
            Nx, Ny : int
                Number of Chebyshev points in x and y directions.
            kind : str
                Currently only 'dct' is supported, using DCT-I.

        Returns:
            coeffs : 2D np.ndarray
                Array of Chebyshev coefficients.
        """

        # Generate Chebyshev nodes in x and y
        x = chebyshev_nodes(Nx, ax, bx)
        y = chebyshev_nodes(Ny, ay, by)

        if callable(func_or_vals):
            X, Y = np.meshgrid(x, y, indexing='ij')
            vals = func_or_vals(X, Y)
        else:
            vals = np.asarray(func_or_vals)
            assert vals.shape == (Nx, Ny), "Input array shape must match (Nx, Ny)"

        if kind == 'dct':
            # Compute DCT-I in both dimensions
            coeffs = dctn(vals, type=2, norm=None)
            # Normalize to match Chebyshev projection
            coeffs *= 1 / ((Nx) * (Ny))
            coeffs[0, :] *= 0.5
            coeffs[:, 0] *= 0.5
        else:
            raise ValueError("Only 'dct' is supported for now.")

        return coeffs
    
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
        y_max = np.max([np.abs(self.coeffs[:, -2])*(self.Nx - 1),(np.abs(self.coeffs[:, -1]) * (self.Nx))])
        x_max = np.max([np.abs(self.coeffs[-2, :])*(self.Ny - 1),(np.abs(self.coeffs[-1, :]) * (self.Ny))])

        abs_error = np.max([y_max, x_max])
        return abs_error

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

