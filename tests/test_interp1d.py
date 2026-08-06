"""Tests for ChebyshevInterpolant1D."""

import numpy as np
import pytest

from chebyshevnd import ChebyshevInterpolant1D, compute_chebyshev_coefficients_1d
from chebyshevnd.utils import chebyshev_gauss_lobatto_nodes


def test_chebyshev_polynomial_gives_delta_coefficients():
    # Interpolating T5 itself must give coeffs = e_5 exactly
    f = lambda x: np.cos(5 * np.arccos(x))
    interp = ChebyshevInterpolant1D(f, -1.0, 1.0, N=12)
    expected = np.zeros(12)
    expected[5] = 1.0
    np.testing.assert_allclose(interp.get_coefficients(), expected, atol=1e-12)


def test_interpolates_its_own_nodes_exactly():
    # The interpolant must reproduce the sampled values at the nodes to machine
    # precision — this catches any off-by-one in the number of coefficients used.
    f = np.exp
    interp = ChebyshevInterpolant1D(f, 0.0, 2.0, N=8)
    np.testing.assert_allclose(interp.evaluate(interp.x_nodes), f(interp.x_nodes), atol=1e-12)


def test_interpolates_gauss_lobatto_nodes_exactly():
    f = np.exp
    interp = ChebyshevInterpolant1D(f, 0.0, 2.0, N=8, nodes="GaussLobatto")
    np.testing.assert_allclose(interp.evaluate(interp.x_nodes), f(interp.x_nodes), atol=1e-12)


def test_polynomial_is_reproduced_exactly():
    p = lambda x: 3 * x**5 - 2 * x**3 + x - 7
    interp = ChebyshevInterpolant1D(p, -2.0, 1.0, N=10)
    # Coefficients above the polynomial degree vanish
    assert np.max(np.abs(interp.get_coefficients()[6:])) < 1e-11
    x = np.linspace(-2.0, 1.0, 13)
    np.testing.assert_allclose(interp.evaluate(x), p(x), rtol=1e-11, atol=1e-11)


def test_spectral_convergence_on_runge_function():
    f = lambda x: 1.0 / (1.0 + 25 * x**2)
    errors = []
    x = np.linspace(-1.0, 1.0, 101)
    # Poles at ±i/5 give a convergence rate of only ~1.22^-N, so machine
    # precision needs N ≈ 180
    for N in (20, 60, 180):
        interp = ChebyshevInterpolant1D(f, -1.0, 1.0, N=N)
        errors.append(np.max(np.abs(interp.evaluate(x) - f(x))))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-12


def test_gauss_lobatto_callable_matches_values_input():
    # Passing a callable with nodes='GaussLobatto' must sample at GL nodes,
    # i.e. agree with passing the GL-sampled values directly.
    f = np.exp
    a, b, N = 0.0, 2.0, 16
    vals = f(chebyshev_gauss_lobatto_nodes(N, a, b))
    from_callable = ChebyshevInterpolant1D(f, a, b, N=N, nodes="GaussLobatto")
    from_values = ChebyshevInterpolant1D(vals, a, b, N=N, nodes="GaussLobatto")
    np.testing.assert_allclose(
        from_callable.get_coefficients(), from_values.get_coefficients(), rtol=1e-13
    )


def test_invalid_node_type_raises():
    with pytest.raises(ValueError):
        ChebyshevInterpolant1D(np.ones(8), 0.0, 1.0, N=8, nodes="bogus")


def test_compute_chebyshev_coefficients_1d_matches_class():
    f = np.sin
    a, b, N = -1.0, 2.0, 20
    coeffs = compute_chebyshev_coefficients_1d(f, a, b, N)
    interp = ChebyshevInterpolant1D(f, a, b, N=N)
    np.testing.assert_allclose(coeffs, interp.get_coefficients(), rtol=1e-13)


def test_optimize_truncates_and_preserves_accuracy():
    f = np.exp
    a, b = 0.0, 2.0
    interp = ChebyshevInterpolant1D(f, a, b, N=40)
    x = np.linspace(a, b, 33)
    before = interp.evaluate(x)

    rel_tol = 1e-8
    interp.optimize(rel_tol=rel_tol)
    assert interp.k_max < 40  # exp needs far fewer than 40 coefficients at 1e-8
    after = interp.evaluate(x)
    # Dropping sub-threshold coefficients moves values by at most N * tol * max|c|
    assert np.max(np.abs(after - before)) <= 40 * rel_tol * interp.maxCoeff
    np.testing.assert_allclose(after, f(x), atol=1e-6)


def test_optimize_rejects_tolerance_below_resolution():
    interp = ChebyshevInterpolant1D(np.exp, 0.0, 2.0, N=40)
    with pytest.raises(ValueError):
        interp.optimize(rel_tol=1e-30)


def test_error_estimates_are_consistent():
    interp = ChebyshevInterpolant1D(np.exp, 0.0, 2.0, N=20)
    c = interp.get_coefficients()
    assert interp.absolute_error() == pytest.approx(max(abs(c[-1]), abs(c[-2])))
    assert interp.relative_error() == pytest.approx(
        interp.absolute_error() / np.max(np.abs(c))
    )


def test_evaluate_accepts_scalar_and_array():
    interp = ChebyshevInterpolant1D(np.exp, 0.0, 2.0, N=20)
    scalar_result = interp.evaluate(1.0)
    assert scalar_result.shape == (1,)
    array_result = interp.evaluate(np.array([0.5, 1.0, 1.5]))
    assert array_result.shape == (3,)
