"""
Smoke tests: build an interpolant in each dimension and check it reproduces
smooth functions at off-grid points. These protect the package rename;
the full unit-test suite is added separately.
"""

import numpy as np
import pytest

from chebyshevnd import (
    ChebyshevInterpolant1D,
    ChebyshevInterpolant2D,
    ChebyshevInterpolant3D,
    ChebyshevInterpolant4D,
    clenshaw_1d,
    chebyshev_nodes,
)
from chebyshevnd.utils import chebyshev_gauss_lobatto_nodes


def cheb_grid_values(f, domains, Ns):
    """Sample f on a tensor grid of Chebyshev nodes, one (a, b) per axis."""
    axes = [chebyshev_nodes(N, a, b) for (a, b), N in zip(domains, Ns)]
    grids = np.meshgrid(*axes, indexing="ij")
    return f(*grids)


def test_import_surface():
    import chebyshevnd

    for name in chebyshevnd.__all__:
        assert hasattr(chebyshevnd, name), f"chebyshevnd.{name} listed in __all__ but missing"


def test_clenshaw_1d_matches_numpy_chebval():
    rng = np.random.default_rng(42)
    coeffs = rng.standard_normal(20) * 0.5 ** np.arange(20)
    a, b = 0.5, 3.0
    for x in np.linspace(a, b, 7):
        xt = (2 * x - a - b) / (b - a)
        expected = np.polynomial.chebyshev.chebval(xt, coeffs)
        result = clenshaw_1d(coeffs, x, len(coeffs), a, b)
        assert result == pytest.approx(expected, rel=1e-12)


def test_1d_interpolant_from_function():
    f = lambda x: np.exp(x) * np.sin(2 * x)
    a, b = 0.0, 2.0
    interp = ChebyshevInterpolant1D(f, a, b, N=32)
    x_test = np.linspace(a, b, 17)
    err = np.max(np.abs(interp.evaluate(x_test) - f(x_test)))
    assert err < 1e-8


def test_1d_interpolant_from_values_matches_function_input():
    f = lambda x: 1.0 / (1.0 + x**2)
    a, b, N = -1.0, 1.0, 40
    vals = f(chebyshev_nodes(N, a, b))
    from_func = ChebyshevInterpolant1D(f, a, b, N=N)
    from_vals = ChebyshevInterpolant1D(vals, a, b, N=N)
    np.testing.assert_allclose(from_vals.get_coefficients(), from_func.get_coefficients())


def test_2d_interpolant():
    f = lambda x, y: np.sin(x) * np.cos(y)
    domains = [(-1.0, 2.0), (0.0, 3.0)]
    vals = cheb_grid_values(f, domains, (24, 24))
    interp = ChebyshevInterpolant2D(vals, *domains[0], *domains[1])
    for x in np.linspace(*domains[0], 5):
        for y in np.linspace(*domains[1], 5):
            assert interp.evaluate(x, y) == pytest.approx(f(x, y), abs=1e-8)


def test_2d_interpolant_gauss_lobatto_nodes():
    f = lambda x, y: np.exp(-(x**2)) * np.cos(y)
    domains = [(-1.0, 1.0), (0.0, 2.0)]
    axes = [chebyshev_gauss_lobatto_nodes(24, a, b) for a, b in domains]
    X, Y = np.meshgrid(*axes, indexing="ij")
    interp = ChebyshevInterpolant2D(f(X, Y), *domains[0], *domains[1], nodes="GaussLobatto")
    for x in np.linspace(*domains[0], 5):
        for y in np.linspace(*domains[1], 5):
            assert interp.evaluate(x, y) == pytest.approx(f(x, y), abs=1e-8)


def test_3d_interpolant():
    f = lambda x, y, z: np.sin(x) * np.cos(y) * np.exp(-z)
    domains = [(0.0, 2.0), (-1.0, 3.0), (1.0, 4.0)]
    vals = cheb_grid_values(f, domains, (16, 16, 16))
    interp = ChebyshevInterpolant3D(vals, *domains[0], *domains[1], *domains[2])
    for x in np.linspace(*domains[0], 3):
        for y in np.linspace(*domains[1], 3):
            for z in np.linspace(*domains[2], 3):
                assert interp.evaluate(x, y, z) == pytest.approx(f(x, y, z), abs=1e-7)


def test_4d_interpolant():
    f = lambda x, y, z, w: np.sin(x) * np.cos(y) * np.exp(-z) * (w**3 + w)
    domains = [(0.0, 2.0), (-1.0, 3.0), (1.0, 4.0), (-2.0, 0.0)]
    vals = cheb_grid_values(f, domains, (12, 12, 12, 12))
    interp = ChebyshevInterpolant4D(
        vals, *domains[0], *domains[1], *domains[2], *domains[3]
    )
    for x in np.linspace(*domains[0], 3):
        for y in np.linspace(*domains[1], 3):
            for z in np.linspace(*domains[2], 3):
                for w in np.linspace(*domains[3], 3):
                    assert interp.evaluate(x, y, z, w) == pytest.approx(f(x, y, z, w), abs=1e-6)
