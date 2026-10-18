"""4D Chebyshev interpolation."""

import numpy as np
from .clenshaw import clenshaw_1d, clenshaw_2d,clenshaw_3d,clenshaw_4d, partial_clenshaw_4d
from .utils import(compute_chebyshev_coefficients_nd,
                         prune_trailing_slices_greedy_nd,
                         rescale_variable
                         )


class ChebyshevInterpolant4D:
    """
    Chebyshev interpolant of a function of four variables on
    [ax, bx] x [ay, by] x [az, bz] x [aw, bw].

    Coefficients are computed from values sampled on a tensor grid with a
    single 4D DCT, and evaluation uses nested Clenshaw recursions.
    """

    def __init__(self, vals,
                  ax = -1., bx = 1.,
                  ay = -1., by = 1.,
                  az = -1., bz = 1.,
                  aw = -1., bw = 1.,nodes = 'Chebyshev'):
        """
        Construct a 4D Chebyshev interpolant from values sampled on a tensor grid.

        Parameters
        ----------
        vals : np.ndarray
            4D array of function values, shape (Nx, Ny, Nz, Nw), sampled on
            the tensor grid of the chosen node family (``chebyshev_nodes`` per
            axis for 'Chebyshev', ``chebyshev_gauss_lobatto_nodes`` for
            'GaussLobatto') with ``np.meshgrid(..., indexing='ij')`` ordering.
        ax, bx : float, optional
            Domain bounds in x. Default [-1, 1].
        ay, by : float, optional
            Domain bounds in y. Default [-1, 1].
        az, bz : float, optional
            Domain bounds in z. Default [-1, 1].
        aw, bw : float, optional
            Domain bounds in w. Default [-1, 1].
        nodes : str, optional
            Node family the values were sampled at: 'Chebyshev' (default) or
            'GaussLobatto'.
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
        """
        Evaluate the interpolant at a single point.

        Parameters
        ----------
        x, y, z, w : float
            Evaluation point in [ax, bx] x [ay, by] x [az, bz] x [aw, bw].

        Returns
        -------
        float
            Value of the interpolant at (x, y, z, w).
        """
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

        Returns
        -------
        np.ndarray
            4D array of Chebyshev coefficients.
        """
        return self.coeffs

    def absolute_error(self):
        """
        Estimate the absolute error of the interpolant from the trailing
        coefficients along each axis.

        Returns
        -------
        float
            Absolute error estimate.
        """
        err = 0.0
        for axis, N in enumerate(self.coeffs.shape):
            c = np.moveaxis(self.coeffs, axis, 0)
            err = max(err, np.max(np.abs(c[-2])) * (N - 1), np.max(np.abs(c[-1])) * N)
        return err

    def relative_error(self):
        """
        Estimate the relative error of the interpolant from the trailing
        coefficients along each axis.

        Returns
        -------
        float
            Relative error estimate.
        """
        return self.rel_error

    def optimize(self, rel_tol=1e-8):
        """
        Truncate the Chebyshev series by greedily pruning trailing coefficient
        slices while keeping the global relative error below `rel_tol`.

        Sets ``self.kmax``, the per-axis truncation counts that ``evaluate()``
        uses to shorten the Clenshaw recursions. An x axis collapsed earlier
        by ``precompute_x`` keeps its old truncation; call ``precompute_x``
        again to use the new one.

        Parameters
        ----------
        rel_tol : float, optional
            Allowed global relative error of the truncation.
        """
        self.kmax = prune_trailing_slices_greedy_nd(self.coeffs, delta=rel_tol, scale='max')[0]

    def effective_grid(self):
        """
        Get the effective grid size after truncation.

        Returns
        -------
        tuple of int
            Per-axis truncation counts (kmax).
        """
        return self.kmax

    def precompute_x(self, x_value):
        """
        Collapse the x axis at a fixed x value.

        Stores an effective 3D coefficient array in (y, z, w) so that repeated
        evaluations at this x via ``evaluate_yzw`` are cheaper than full 4D
        evaluations.

        Parameters
        ----------
        x_value : float
            The fixed x value.
        """
        self.precomp_x = partial_clenshaw_4d(self.coeffs, x_value, a=self.ax, b=self.bx, kmax = self.kmax)
        self.kmax_precomp = (self.kmax[1],self.kmax[2],self.kmax[3])

    def evaluate_yzw(self, y, z,w):
        """
        Evaluate at (y, z, w) with the x contribution fixed by ``precompute_x``.

        Parameters
        ----------
        y, z, w : float
            Evaluation point in [ay, by] x [az, bz] x [aw, bw].

        Returns
        -------
        float
            Value of the interpolant at the precomputed x and (y, z, w).
        """
        if self.precomp_x is None:
            raise ValueError("precompute_x must be called before evaluate_yzw.")

        return clenshaw_3d(
            self.precomp_x, y, z, w,
            ax=self.ay, bx=self.by,
            ay=self.az, by=self.bz,
            az=self.aw, bz=self.bw,
            kmax=self.kmax_precomp
        )
