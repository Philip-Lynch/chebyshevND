"""3D Chebyshev interpolation."""

import numpy as np
from .clenshaw import clenshaw_1d, clenshaw_2d,clenshaw_3d
from .utils import(compute_chebyshev_coefficients_nd,
                         prune_trailing_slices_greedy_nd,
                         rescale_variable
                         )


class ChebyshevInterpolant3D:
    """
    Chebyshev interpolant of a function of three variables on
    [ax, bx] x [ay, by] x [az, bz].

    Coefficients are computed from values sampled on a tensor grid with a
    single 3D DCT, and evaluation uses nested Clenshaw recursions.
    """

    def __init__(self, vals,
                  ax = -1., bx = 1.,
                  ay = -1., by = 1.,
                  az = -1., bz = 1., nodes = 'Chebyshev'):
        """
        Construct a 3D Chebyshev interpolant from values sampled on a tensor grid.

        Parameters
        ----------
        vals : np.ndarray
            3D array of function values, shape (Nx, Ny, Nz), sampled on the
            tensor grid of the chosen node family (``chebyshev_nodes`` per
            axis for 'Chebyshev', ``chebyshev_gauss_lobatto_nodes`` for
            'GaussLobatto') with ``np.meshgrid(..., indexing='ij')`` ordering.
            Axis 0 is x, axis 1 is y and axis 2 is z, matching the order of
            the bound arguments; pass the bounds by keyword to avoid mixing
            them up.
        ax, bx : float, optional
            Lower and upper bounds of x (axis 0 of ``vals``).
            Default [-1, 1].
        ay, by : float, optional
            Lower and upper bounds of y (axis 1 of ``vals``).
            Default [-1, 1].
        az, bz : float, optional
            Lower and upper bounds of z (axis 2 of ``vals``).
            Default [-1, 1].
        nodes : str, optional
            Node family the values were sampled at: 'Chebyshev' (default) or
            'GaussLobatto'.
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
        self.NyMaxes = np.ones(self.Ny, dtype=np.int64) * self.Nz

    def evaluate(self, x, y,z):
        """
        Evaluate the interpolant at a single point.

        Parameters
        ----------
        x, y, z : float
            Evaluation point in [ax, bx] x [ay, by] x [az, bz].

        Returns
        -------
        float
            Value of the interpolant at (x, y, z).
        """
        coeffs = self.coeffs
        kmax = self.kmax
        ax, bx = self.ax, self.bx
        ay, by = self.ay, self.by
        az, bz = self.az, self.bz
        return clenshaw_3d(coeffs, x, y, z, ax=ax, bx=bx, ay=ay, by=by, az=az, bz=bz, kmax=kmax)

    def get_coefficients(self):
        """
        Get the Chebyshev coefficients of the interpolant.

        Returns
        -------
        np.ndarray
            3D array of Chebyshev coefficients.
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

        Stores an effective 2D coefficient array in (y, z) so that repeated
        evaluations at this x via ``evaluate_yz`` are cheaper than full 3D
        evaluations.

        Parameters
        ----------
        x_value : float
            The fixed x value.
        """
        tx = (2 * x_value - (self.bx + self.ax)) / (self.bx - self.ax)
        ix, iy, iz = self.kmax

        # Collapse x-direction using Clenshaw recurrence for each (j, k)
        temp_yz = np.empty((iy, iz))

        for j in range(iy):
            for k in range(iz):
                # pull the 1D coeffs in x for this (j, k)
                coeffs_x = self.coeffs[:ix, j, k]
                temp_yz[j, k] = clenshaw_1d(coeffs_x, tx)

        self.precomp_x = temp_yz
        # Truncation used by evaluate_yz, tied to the block filled above so a
        # later optimize() cannot make it read entries that were never set
        self.NyMaxes = np.ones(iy, dtype=np.int64) * iz

    def evaluate_yz(self, y, z):
        """
        Evaluate at (y, z) with the x contribution fixed by ``precompute_x``.

        Parameters
        ----------
        y, z : float
            Evaluation point in [ay, by] x [az, bz].

        Returns
        -------
        float
            Value of the interpolant at the precomputed x and (y, z).
        """
        if self.precomp_x is None:
            raise ValueError("precompute_x must be called before evaluate_yz.")

        return clenshaw_2d(
            self.precomp_x, y, z,
            ax=self.ay, bx=self.by,
            ay=self.az, by=self.bz,
            NyMaxes=self.NyMaxes
        )
