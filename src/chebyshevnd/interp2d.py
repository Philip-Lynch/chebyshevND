"""2D Chebyshev interpolation."""

import numpy as np
from .clenshaw import clenshaw_1d, clenshaw_2d
from .utils import find_significant_coeffs_2d, compute_chebyshev_coefficients_nd


class ChebyshevInterpolant2D:
    """
    Chebyshev interpolant of a function of two variables on [ax, bx] x [ay, by].

    Coefficients are computed from values sampled on a tensor grid with a
    single 2D DCT, and evaluation uses nested Clenshaw recursions.
    """

    def __init__(self, vals, ax = -1., bx = 1., ay = -1., by = 1., nodes = 'Chebyshev'):
        """
        Construct a 2D Chebyshev interpolant from values sampled on a tensor grid.

        Parameters
        ----------
        vals : np.ndarray
            2D array of function values, shape (Nx, Ny), sampled on the tensor
            grid of the chosen node family (``chebyshev_nodes`` per axis for
            'Chebyshev', ``chebyshev_gauss_lobatto_nodes`` for 'GaussLobatto')
            with ``np.meshgrid(..., indexing='ij')`` ordering.
        ax, bx : float, optional
            Domain bounds in x. Default [-1, 1].
        ay, by : float, optional
            Domain bounds in y. Default [-1, 1].
        nodes : str, optional
            Node family the values were sampled at: 'Chebyshev' (default) or
            'GaussLobatto'.
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
        """
        Evaluate the interpolant at a single point.

        Parameters
        ----------
        x, y : float
            Evaluation point in [ax, bx] x [ay, by].

        Returns
        -------
        float
            Value of the interpolant at (x, y).
        """
        return clenshaw_2d(self.coeffs, x, y, NyMaxes =self.NyMaxes, ax =self.ax, bx =self.bx, ay =self.ay, by =self.by)

    def get_coefficients(self):
        """
        Get the Chebyshev coefficients of the interpolant.

        Returns
        -------
        np.ndarray
            2D array of Chebyshev coefficients.
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
        Truncate the Chebyshev series by dropping insignificant coefficients.

        Computes per-row truncation counts that ``evaluate()`` uses to shorten
        the Clenshaw recursions.

        Parameters
        ----------
        rel_tol : float, optional
            Relative tolerance for coefficient significance.
        """
        self.NyMaxes = find_significant_coeffs_2d(self.coeffs, rel_tol=rel_tol,max_coeff=self.max)

    def effective_grid(self):
        """
        Get the effective grid size after truncation.

        Returns
        -------
        list of int
            ``[number of rows kept, largest per-row coefficient count]``.
        """
        return [len(self.NyMaxes),max(self.NyMaxes).item()]
