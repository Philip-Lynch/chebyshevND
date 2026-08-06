"""Tests for ChebyshevInterpolant1D."""

import numpy as np
import pytest

from chebyshevnd import (
    ChebyshevInterpolant1D,
    chebyshev_gauss_lobatto_nodes,
    chebyshev_nodes,
    compute_chebyshev_coefficients_nd,
)


def build(f, a, b, N, nodes="Chebyshev"):
    node_fn = chebyshev_nodes if nodes == "Chebyshev" else chebyshev_gauss_lobatto_nodes
    x = node_fn(N, a, b)
    return x, ChebyshevInterpolant1D(f(x), a, b, nodes=nodes)


def test_chebyshev_polynomial_gives_delta_coefficients():
    # Interpolating T5 itself must give coeffs = e_5 exactly
    f = lambda x: np.cos(5 * np.arccos(x))
    _, interp = build(f, -1.0, 1.0, 12)
    expected = np.zeros(12)
    expected[5] = 1.0
    np.testing.assert_allclose(interp.get_coefficients(), expected, atol=1e-12)


def test_interpolates_its_own_nodes_exactly():
    # The interpolant must reproduce the sampled values at the nodes to machine
    # precision — this catches any off-by-one in the number of coefficients used.
    x, interp = build(np.exp, 0.0, 2.0, 8)
    np.testing.assert_allclose(interp.evaluate(x), np.exp(x), atol=1e-12)


def test_interpolates_gauss_lobatto_nodes_exactly():
    x, interp = build(np.exp, 0.0, 2.0, 8, nodes="GaussLobatto")
    np.testing.assert_allclose(interp.evaluate(x), np.exp(x), atol=1e-12)


def test_node_families_agree_off_grid():
    # A well-resolved function interpolates to the same thing on either grid
    x_test = np.linspace(0.0, 2.0, 21)
    _, cheb = build(np.exp, 0.0, 2.0, 20)
    _, gl = build(np.exp, 0.0, 2.0, 20, nodes="GaussLobatto")
    np.testing.assert_allclose(cheb.evaluate(x_test), gl.evaluate(x_test), atol=1e-13)


def test_polynomial_is_reproduced_exactly():
    p = lambda x: 3 * x**5 - 2 * x**3 + x - 7
    _, interp = build(p, -2.0, 1.0, 10)
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
        _, interp = build(f, -1.0, 1.0, N)
        errors.append(np.max(np.abs(interp.evaluate(x) - f(x))))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-12


def test_invalid_node_type_raises():
    with pytest.raises(ValueError):
        ChebyshevInterpolant1D(np.ones(8), 0.0, 1.0, nodes="bogus")


def test_coefficients_match_nd_function():
    # The 1D class and the generic nD coefficient routine implement the same transform
    vals = np.sin(chebyshev_nodes(20, -1.0, 2.0))
    interp = ChebyshevInterpolant1D(vals, -1.0, 2.0)
    np.testing.assert_allclose(
        compute_chebyshev_coefficients_nd(vals),
        interp.get_coefficients(),
        rtol=1e-13,
        atol=1e-15,
    )


def test_optimize_truncates_and_preserves_accuracy():
    a, b = 0.0, 2.0
    _, interp = build(np.exp, a, b, 40)
    x = np.linspace(a, b, 33)
    before = interp.evaluate(x)

    rel_tol = 1e-8
    interp.optimize(rel_tol=rel_tol)
    assert interp.k_max < 40  # exp needs far fewer than 40 coefficients at 1e-8
    after = interp.evaluate(x)
    # Dropping sub-threshold coefficients moves values by at most N * tol * max|c|
    assert np.max(np.abs(after - before)) <= 40 * rel_tol * interp.maxCoeff
    np.testing.assert_allclose(after, np.exp(x), atol=1e-6)


def test_optimize_rejects_tolerance_below_resolution():
    _, interp = build(np.exp, 0.0, 2.0, 40)
    with pytest.raises(ValueError):
        interp.optimize(rel_tol=1e-30)


def test_error_estimates_are_consistent():
    _, interp = build(np.exp, 0.0, 2.0, 20)
    c = interp.get_coefficients()
    assert interp.absolute_error() == pytest.approx(max(abs(c[-1]), abs(c[-2])))
    assert interp.relative_error() == pytest.approx(
        interp.absolute_error() / np.max(np.abs(c))
    )


def test_evaluate_scalar_returns_float_array_returns_array():
    _, interp = build(np.exp, 0.0, 2.0, 20)
    scalar_result = interp.evaluate(1.0)
    assert isinstance(scalar_result, float)
    assert scalar_result == pytest.approx(np.e)
    array_result = interp.evaluate(np.array([0.5, 1.0, 1.5]))
    assert array_result.shape == (3,)
