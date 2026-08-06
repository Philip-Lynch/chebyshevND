import numpy as np
from scipy.fft import dctn
from .clenshaw import clenshaw_1d, clenshaw_2d,clenshaw_3d
from .utils import(compute_chebyshev_coefficients_nd,
                         prune_trailing_slices_greedy_nd,
                         rescale_variable
                         )

class ChebyshevInterpolant3D:
    def __init__(self, vals,
                  ax = -1., bx = 1.,
                  ay = -1., by = 1.,
                  az = -1., bz = 1., nodes = 'Chebyshev'):
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
        self.ax, self.bx = ax, bx
        self.ay, self.by = ay, by
        self.az, self.bz = az, bz
        self.Nx, self.Ny, self.Nz = vals.shape
        self.precomp_x = None
        

        # Calculate the coefficients and store them
        self.coeffs = compute_chebyshev_coefficients_nd(vals,nodes = nodes)

        # Calculate absolute and relative errors
        
        self.abs_error = self.absolute_error()
        self.max = np.max([np.abs(self.coeffs)])
        self.rel_error = self.abs_error / self.max
        self.kmax = (self.Nx,self.Ny,self.Nz)
        self.NyMaxes = np.ones(self.Nx, dtype=np.int64) * self.Nz
    
    def evaluate(self, x, y,z):
        coeffs = self.coeffs
        kmax = self.kmax
        ax, bx = self.ax, self.bx
        ay, by = self.ay, self.by
        az, bz = self.az, self.bz
        return clenshaw_3d(coeffs, x, y, z, ax=ax, bx=bx, ay=ay, by=by, az=az, bz=bz, kmax=kmax)
    
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
        y_max = np.max([np.abs(self.coeffs[:, -2,:])*(self.Nx - 1),(np.abs(self.coeffs[:, -1,:]) * (self.Nx))])
        x_max = np.max([np.abs(self.coeffs[-2, :,:])*(self.Ny - 1),(np.abs(self.coeffs[-1, :,:]) * (self.Ny))])
        z_max = np.max([np.abs(self.coeffs[:, :, -2])*(self.Nz - 1),(np.abs(self.coeffs[:, :, -1]) * (self.Nz))])
        abs_error = np.max([y_max, x_max,z_max])
        return abs_error

    def relative_error(self):
        """
        Compute the relative error of the interpolant at the Chebyshev nodes.

        Returns:
            float: Relative error at the Chebyshev nodes.
        """
        return self.rel_error
    
    def optimize(self, rel_tol=1e-8):

        self.kmax = prune_trailing_slices_greedy_nd(self.coeffs, delta=rel_tol, scale='max')[0]
        self.NyMaxes = np.ones(self.kmax[1], dtype=np.int64) * self.kmax[2]

    def effective_grid(self):
        """
        Get the effective grid size based on significant coefficients.

        Returns:
            np.ndarray: Array of significant NyMaxes for each x.
        """
        return self.kmax
    
    

    def precompute_x(self, x_value):
        """
        Precompute the contribution along x at a fixed x_value.
        Returns a 2D array of effective coefficients in (y, z).
        """
        tx = (2 * x_value - (self.bx + self.ax)) / (self.bx - self.ax)

        # Collapse x-direction using Clenshaw recurrence for each (j, k)
        Ny, Nz = self.Ny, self.Nz
        temp_yz = np.empty((Ny, Nz))

        for j in range(self.kmax[1]):
            for k in range(self.kmax[2]):
                # pull the 1D coeffs in x for this (j, k)
                coeffs_x = self.coeffs[:, j, k]
                temp_yz[j, k] = clenshaw_1d(coeffs_x, tx)

        self.precomp_x = temp_yz

    def evaluate_yz(self, y, z):
        """
        Given precomputed x contribution (precomp_yz),
        evaluate only in (y, z).
        """
        if self.precomp_x is None:
            raise ValueError("precompute_x must be called before evaluate_yz.")
        
        #NyMaxes = np.ones(self.kmax[1], dtype=np.int64) * self.kmax[2]
        return clenshaw_2d(
            self.precomp_x, y, z,
            ax=self.ay, bx=self.by,
            ay=self.az, by=self.bz,
            NyMaxes=self.NyMaxes
        )
