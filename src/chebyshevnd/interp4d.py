import numpy as np
from scipy.fft import dctn
from .clenshaw import clenshaw_1d, clenshaw_2d,clenshaw_3d,clenshaw_4d, partial_clenshaw_4d
from .utils import(compute_chebyshev_coefficients_nd,
                         prune_trailing_slices_greedy_nd,
                         rescale_variable
                         )

from numba import njit

class ChebyshevInterpolant4D:
    def __init__(self, vals,
                  ax = -1., bx = 1.,
                  ay = -1., by = 1.,
                  az = -1., bz = 1.,
                  aw = -1., bw = 1.,nodes = 'Chebyshev'):
        """
        Initialize a 4D Chebyshev interpolant.

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
        self.ax, self.bx = ax, bx
        self.ay, self.by = ay, by
        self.az, self.bz = az, bz
        self.aw, self.bw = aw, bw
        Nx, Ny, Nz, Nw = vals.shape
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Nw = Nw
        self.precomp_x = None
        self.kmax = (Nx,Ny,Nz,Nw)
        self.kmax_precomp = (self.kmax[1],self.kmax[2],self.kmax[3])
        

        # Calculate the coefficients and store them
        self.coeffs = compute_chebyshev_coefficients_nd(vals,nodes = nodes)

        # Calculate absolute and relative errors
        
        self.abs_error = self.absolute_error()
        self.max = np.max([np.abs(self.coeffs)])
        self.rel_error = self.abs_error / self.max
        
    
    def evaluate(self, x, y,z,w):
        coeffs = self.coeffs
        kmax = self.kmax
        ax, bx = self.ax, self.bx
        ay, by = self.ay, self.by
        az, bz = self.az, self.bz
        aw, bw = self.aw, self.bw
        return clenshaw_4d(coeffs, x, y, z, w, 
                           ax=ax, bx=bx, 
                           ay=ay, by=by, 
                           az=az, bz=bz, 
                           aw=aw, bw=bw,
                           kmax=kmax)
    
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

        self.kmax = prune_trailing_slices_greedy_nd(self.coeffs, delta=rel_tol, scale='max')[0]
        self.kmax_precomp = (self.kmax[1],self.kmax[2],self.kmax[3])

    def effective_grid(self):
        """
        Get the effective grid size based on significant coefficients.

        Returns:
            np.ndarray: Array of significant NyMaxes for each x.
        """
        return self.kmax
    
    
    
    def precompute_x(self, x_value):
        self.precomp_x = partial_clenshaw_4d(self.coeffs, x_value, a=self.ax, b=self.bx, kmax = self.kmax)
        self.kmax_precomp = (self.kmax[1],self.kmax[2],self.kmax[3])


    def evaluate_yzw(self, y, z,w):
        """
        Given precomputed x contribution (precomp_x),
        evaluate only in (y, z,w).
        """
        if self.precomp_x is None:
            raise ValueError("precompute_x must be called before evaluate_yz.")
        
        
        return clenshaw_3d(
            self.precomp_x, y, z, w,
            ax=self.ay, bx=self.by,
            ay=self.az, by=self.bz,
            az=self.aw, bz=self.bw,
            kmax=self.kmax_precomp
        )


