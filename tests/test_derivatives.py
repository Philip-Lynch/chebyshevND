"""Tests for analytic derivatives of ChebyshevInterpolantND."""

import numpy as np
import pytest
from numpy.polynomial import chebyshev as npcheb

from chebyshevnd import ChebyshevInterpolantND, differentiate_coefficients
from helpers import decaying_coeffs, sample_on_grid

DOMAINS = [(0.0, 1.5), (-1.0, 1.0), (0.5, 2.0)]
NS = (20, 20, 20)
NAMES = ("x", "y", "z")


def f(x, y, z):
    return np.sin(2 * x) * np.exp(y) * np.cos(z)


# Analytic partial derivatives of f, keyed by (order_x, order_y, order_z)
EXACT = {
    (1, 0, 0): lambda x, y, z: 2 * np.cos(2 * x) * np.exp(y) * np.cos(z),
    (0, 1, 0): f,
    (0, 0, 1): lambda x, y, z: -np.sin(2 * x) * np.exp(y) * np.sin(z),
    (2, 0, 0): lambda x, y, z: -4 * f(x, y, z),
    (0, 0, 2): lambda x, y, z: -f(x, y, z),
    (1, 1, 0): lambda x, y, z: 2 * np.cos(2 * x) * np.exp(y) * np.cos(z),
    (1, 0, 1): lambda x, y, z: -2 * np.cos(2 * x) * np.exp(y) * np.sin(z),
    (0, 1, 1): lambda x, y, z: -np.sin(2 * x) * np.exp(y) * np.sin(z),
}


def build(nodes="Chebyshev"):
    _, vals = sample_on_grid(f, DOMAINS, NS, nodes=nodes)
    return ChebyshevInterpolantND(vals, DOMAINS, nodes=nodes, axis_names=NAMES)


def random_points(n, seed=4, domains=DOMAINS):
    rng = np.random.default_rng(seed)
    lo, hi = np.array(domains).T
    return rng.uniform(lo, hi, size=(n, len(domains)))


# ------------------------------------------------- coefficient recurrence

@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_recurrence_matches_numpy_chebder(axis, order):
    rng = np.random.default_rng(1)
    coeffs = decaying_coeffs(rng, (7, 6, 5)) + 1j * decaying_coeffs(rng, (7, 6, 5))
    got = differentiate_coefficients(coeffs, axis, order)
    expected = npcheb.chebder(coeffs, m=order, axis=axis)
    n = coeffs.shape[axis]
    assert got.shape == coeffs.shape
    np.testing.assert_allclose(np.take(got, range(n - order), axis=axis), expected, rtol=1e-13, atol=1e-15)
    np.testing.assert_array_equal(np.take(got, range(n - order, n), axis=axis), 0)


def test_recurrence_edge_cases():
    np.testing.assert_array_equal(differentiate_coefficients(np.array([3.0]), 0), [0.0])
    np.testing.assert_array_equal(differentiate_coefficients(np.array([3.0, 5.0]), 0), [5.0, 0.0])
    np.testing.assert_array_equal(differentiate_coefficients(np.arange(4.0), 0, order=0), np.arange(4.0))
    assert differentiate_coefficients(np.arange(4), 0).dtype == np.float64
    with pytest.raises(ValueError):
        differentiate_coefficients(np.ones(3), 0, order=-1)


# ---------------------------------------------------------------- accuracy

@pytest.mark.parametrize("orders", list(EXACT), ids=str)
def test_derivatives_of_separable_function(orders):
    # 20 nodes per axis: ~1e-10 for first and ~1e-8 for second derivatives
    # is the design target; the function is resolved well beyond that
    nd = build()
    deriv = nd.derivative(orders)
    tol = 1e-10 if sum(orders) == 1 else 1e-8
    for p in random_points(30):
        assert deriv.evaluate(*p) == pytest.approx(EXACT[orders](*p), abs=tol)


def test_mixed_partials_commute():
    nd = build()
    joint = nd.derivative({"x": 1, "z": 1})
    xz = nd.derivative({"x": 1}).derivative({"z": 1})
    zx = nd.derivative({"z": 1}).derivative({"x": 1})
    np.testing.assert_allclose(xz.get_coefficients(), joint.get_coefficients(), rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(zx.get_coefficients(), joint.get_coefficients(), rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("domain, n", [((0.0, 1.0), 24), ((-2.0, 3.0), 60), ((10.0, 10.5), 16)])
def test_chain_rule_under_domain_rescaling(domain, n):
    # The same function on domains of different widths: the 2/(b - a)
    # factor must give the derivative in the physical coordinate
    g = lambda x: np.exp(np.sin(x))
    dg = lambda x: np.cos(x) * g(x)
    d2g = lambda x: (np.cos(x) ** 2 - np.sin(x)) * g(x)
    _, vals = sample_on_grid(g, [domain], (n,))
    nd = ChebyshevInterpolantND(vals, [domain])
    first, second = nd.derivative([1]), nd.derivative([2])
    for x in np.linspace(*domain, 9):
        assert first.evaluate(x) == pytest.approx(dg(x), abs=1e-10)
        assert second.evaluate(x) == pytest.approx(d2g(x), abs=1e-8)


@pytest.mark.parametrize("nodes", ["Chebyshev", "GaussLobatto"])
def test_both_node_families(nodes):
    deriv = build(nodes).derivative({"y": 1, "z": 1})
    for p in random_points(10):
        assert deriv.evaluate(*p) == pytest.approx(EXACT[(0, 1, 1)](*p), abs=1e-10)


def test_complex_multi_output_matches_single_outputs():
    funcs = [lambda x, y, z: np.exp(1j * (x + y)) * z, lambda x, y, z: np.cos(x * y) + 1j * z**2]
    axes = sample_on_grid(lambda *x: x[0], DOMAINS, (14, 12, 10))[0]
    grids = np.meshgrid(*axes, indexing="ij")
    vals = np.stack([g(*grids) for g in funcs], axis=-1)
    multi = ChebyshevInterpolantND(vals, DOMAINS, axis_names=NAMES, output_names=["a", "b"]).derivative({"x": 1, "y": 1})
    singles = [ChebyshevInterpolantND(vals[..., i], DOMAINS).derivative((1, 1, 0)) for i in range(2)]
    for p in random_points(8):
        np.testing.assert_allclose(multi.evaluate(*p), [s.evaluate(*p) for s in singles], rtol=1e-13, atol=1e-15)


def test_derivative_commutes_with_precontraction():
    nd = build()
    for p in random_points(6):
        a = nd.derivative({"y": 1, "z": 1}).precontract({"x": p[0]})
        b = nd.precontract({"x": p[0]}).derivative({"y": 1, "z": 1})
        assert a.evaluate(*p[1:]) == pytest.approx(b.evaluate(*p[1:]), rel=1e-13, abs=1e-14)


# --------------------------------------------- truncation and error estimates

def test_derivative_of_optimized_series_is_exact_for_truncation():
    nd = build()
    nd.optimize(1e-8)
    deriv = nd.derivative({"x": 1})
    expected_coeffs = npcheb.chebder(nd.get_coefficients(), axis=0) * 2.0 / (DOMAINS[0][1] - DOMAINS[0][0])
    assert deriv.kmax == (nd.kmax[0] - 1,) + nd.kmax[1:]
    np.testing.assert_allclose(deriv.get_coefficients(), expected_coeffs, rtol=1e-13, atol=1e-15)
    # Truncating the function to 1e-8 shows up in the derivative's estimate
    points = random_points(40)
    actual = max(abs(deriv.evaluate(*p) - EXACT[(1, 0, 0)](*p)) for p in points)
    assert actual <= deriv.absolute_error()


@pytest.mark.parametrize("orders", [(1, 0, 0), (0, 0, 2), (1, 1, 0)], ids=str)
def test_error_estimate_bounds_actual_error(orders):
    nd = build()
    deriv = nd.derivative(orders)
    actual = max(abs(deriv.evaluate(*p) - EXACT[orders](*p)) for p in random_points(200))
    assert actual <= deriv.absolute_error()
    # Differentiation amplifies errors, and the estimate reflects that
    assert deriv.absolute_error() > nd.absolute_error()


def test_derivative_without_full_tensor_uses_markov_bound():
    nd = build()
    nd.optimize(1e-8, keep_full=False)
    deriv = nd.derivative({"x": 1})
    with pytest.raises(RuntimeError):
        deriv.get_coefficients(full=True)
    actual = max(abs(deriv.evaluate(*p) - EXACT[(1, 0, 0)](*p)) for p in random_points(40))
    assert actual <= deriv.absolute_error()


def test_order_beyond_degree_gives_zero():
    _, vals = sample_on_grid(lambda x, y: x**2 + y, [(0, 1), (0, 1)], (3, 4))
    nd = ChebyshevInterpolantND(vals, [(0, 1), (0, 1)])
    assert nd.derivative((3, 0)).evaluate(0.3, 0.4) == pytest.approx(0.0, abs=1e-13)
    assert nd.derivative((2, 0)).evaluate(0.3, 0.4) == pytest.approx(2.0, abs=1e-12)


def test_zero_orders_reproduce_the_interpolant():
    nd = build()
    same = nd.derivative((0, 0, 0))
    p = random_points(1)[0]
    assert same.evaluate(*p) == nd.evaluate(*p)


def test_metadata_is_kept():
    nd = build()
    deriv = nd.derivative({"z": 1})
    assert deriv.axis_names == NAMES
    assert deriv.domains == nd.domains
    assert deriv.grid_shape == nd.grid_shape
    assert deriv.nodes == nd.nodes


@pytest.mark.parametrize("orders, error", [
    ((1, 0), ValueError),
    ((1, -1, 0), ValueError),
    ((1.5, 0, 0), ValueError),
    ((True, 0, 0), ValueError),
    ({"w": 1}, KeyError),
    ({"x": 1, 0: 2}, ValueError),
])
def test_bad_orders(orders, error):
    with pytest.raises(error):
        build().derivative(orders)
