import numpy as np
from scipy.fft import dct
from .clenshaw import clenshaw_1d


class ChebyshevInterpolant1D:
    def __init__(self, vals, a=-1., b=1., nodes='Chebyshev'):
        """
        Construct a 1D Chebyshev interpolant from values sampled on a Chebyshev grid.

        Parameters:
            vals : array-like
                Function values sampled at the nodes of [a, b]:
                chebyshev_nodes(N, a, b) for 'Chebyshev', or
                chebyshev_gauss_lobatto_nodes(N, a, b) for 'GaussLobatto'.
            a, b : float
                Domain bounds [a, b].
            nodes : str
                Node family the values were sampled at: 'Chebyshev' or 'GaussLobatto'.
        """
        self.a = a
        self.b = b

        y = np.asarray(vals)
        self.N = len(y)

        # To make this work with end points, use type=1, divide by N-1 and 0.5*coeff[-1]
        if nodes == 'Chebyshev':
            self.coeffs = dct(y, type=2) / self.N
            self.coeffs[0] *= 0.5
        elif nodes == 'GaussLobatto':
            self.coeffs = dct(y, type=1, norm = None) / (self.N - 1)
            self.coeffs[0] *= 0.5
            self.coeffs[-1] *= 0.5
        else:
            raise ValueError('Incorrect node type. Please choose either Chebyshev or GaussLobatto')

        self.absErr = max(abs(self.coeffs[-1]), abs(self.coeffs[-2]))
        self.maxCoeff = max(abs(self.coeffs))
        self.relErr = self.absErr/self.maxCoeff
        # Number of leading coefficients used by evaluate(); a count, not an index
        self.k_max = len(self.coeffs)

    def get_coefficients(self):
        """
        Get the Chebyshev coefficients of the interpolant.

        Returns:
            np.ndarray: Chebyshev coefficients.
        """
        return self.coeffs

    def absolute_error(self):
        """
        Estimate the absolute error of the interpolant from its trailing coefficients.

        Returns:
            float: Absolute error estimate.
        """
        return self.absErr

    def relative_error(self):
        """
        Estimate the relative error of the interpolant from its trailing coefficients.

        Returns:
            float: Relative error estimate.
        """
        return self.relErr

    def evaluate(self, x):
        """Evaluate the interpolant at x: a scalar returns a float, an array returns an array."""
        x_arr = np.atleast_1d(x)
        result = np.array([clenshaw_1d(self.coeffs, xi, self.k_max, self.a, self.b) for xi in x_arr])
        if np.ndim(x) == 0:
            return float(result[0])
        return result

    def optimize(self, rel_tol=1e-6):
        """
        Truncate the Chebyshev series by dropping insignificant trailing coefficients.

        Parameters:
            rel_tol: float
                Relative tolerance for coefficient significance.

        Sets self.k_max to the number of leading coefficients that evaluate() uses.
        """
        if rel_tol < self.relErr:
            raise ValueError("Relative tolerance must be greater than the relative error of the interpolant.")

        k_max = 1
        for k in range(len(self.coeffs) - 1, -1, -1):
            if abs(self.coeffs[k]) > rel_tol * self.maxCoeff:
                k_max = k + 1
                break

        self.k_max = k_max
