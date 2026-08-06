"""Tests for node generation, coefficient computation, and pruning utilities."""

import numpy as np
import pytest

from chebyshevnd import (
    chebyshev_nodes,
    clenshaw_3d,
    compute_chebyshev_coefficients_nd,
    find_significant_coeffs_1d,
    find_significant_coeffs_2d,
    prune_trailing_slices_greedy_nd,
    renormalize_coefficients_GL,
)
from chebyshevnd.utils import chebyshev_gauss_lobatto_nodes, rescale_variable
from helpers import decaying_coeffs, direct_eval, sample_on_grid


def test_chebyshev_nodes_analytic_formula():
    N, a, b = 9, -2.0, 3.0
    nodes = chebyshev_nodes(N, a, b)
    k = np.arange(N)
    expected = 0.5 * (b - a) * np.cos(np.pi * (2 * k + 1) / (2 * N)) + 0.5 * (a + b)
    np.testing.assert_allclose(nodes, expected, rtol=1e-14)
    # Root nodes exclude the endpoints and come out in descending order
    assert np.all(nodes > a) and np.all(nodes < b)
    assert np.all(np.diff(nodes) < 0)


def test_gauss_lobatto_nodes_include_endpoints():
    N, a, b = 8, 0.0, 5.0
    nodes = chebyshev_gauss_lobatto_nodes(N, a, b)
    assert nodes[0] == pytest.approx(b)
    assert nodes[-1] == pytest.approx(a)
    j = np.arange(N)
    expected = 0.5 * (b - a) * np.cos(np.pi * j / (N - 1)) + 0.5 * (a + b)
    np.testing.assert_allclose(nodes, expected, rtol=1e-14)


def test_rescale_variable_maps_domain_to_unit():
    a, b = 2.0, 6.0
    assert rescale_variable(a, a, b) == pytest.approx(-1.0)
    assert rescale_variable(b, a, b) == pytest.approx(1.0)
    assert rescale_variable(4.0, a, b) == pytest.approx(0.0)


@pytest.mark.parametrize("nodes", ["Chebyshev", "GaussLobatto"])
def test_coefficients_of_chebyshev_product_are_a_delta(nodes):
    # f(x, y) = T2(x) * T3(y) must produce coeffs[2, 3] = 1 and zeros elsewhere,
    # which pins down the DCT normalization exactly.
    f = lambda x, y: np.cos(2 * np.arccos(x)) * np.cos(3 * np.arccos(y))
    _, vals = sample_on_grid(f, [(-1.0, 1.0), (-1.0, 1.0)], (8, 9), nodes=nodes)
    coeffs = compute_chebyshev_coefficients_nd(vals, nodes=nodes)
    expected = np.zeros((8, 9))
    expected[2, 3] = 1.0
    np.testing.assert_allclose(coeffs, expected, atol=1e-12)


def test_compute_coefficients_invalid_nodes_raises():
    with pytest.raises(ValueError):
        compute_chebyshev_coefficients_nd(np.ones((4, 4)), nodes="bogus")


def test_renormalize_gl_rejects_short_axes():
    with pytest.raises(ValueError):
        renormalize_coefficients_GL(np.ones((1, 4)))


def test_find_significant_coeffs_1d_returns_count():
    coeffs = np.array([1.0, 1e-3, 1e-12, 0.0])
    # Threshold is rel_tol * max / n = 1e-6 / 4; last coefficient above it is index 1
    assert find_significant_coeffs_1d(coeffs, rel_tol=1e-6) == 2
    # With an external max_coeff, everything can be insignificant
    tiny = np.array([1e-9, 1e-9, 1e-9])
    assert find_significant_coeffs_1d(tiny, rel_tol=1e-3, max_coeff=1.0) == 0


def test_find_significant_coeffs_2d_counts_are_sane():
    rng = np.random.default_rng(3)
    coeffs = decaying_coeffs(rng, (10, 10), base=0.3)
    counts = find_significant_coeffs_2d(coeffs, rel_tol=1e-6)
    assert counts.dtype == np.int64
    assert 1 <= len(counts) <= 10
    assert np.all(counts >= 0) and np.all(counts <= 10)
    # Coefficients decay along both axes, so later rows keep no more than the first
    assert counts[-1] <= counts[0]


def test_prune_respects_error_budget():
    rng = np.random.default_rng(11)
    coeffs = decaying_coeffs(rng, (10, 9, 8), base=0.35)
    delta = 1e-6
    kmax, discarded = prune_trailing_slices_greedy_nd(coeffs, delta=delta, scale="max")
    assert discarded <= delta * np.max(np.abs(coeffs))
    assert all(1 <= k <= n for k, n in zip(kmax, coeffs.shape))
    # sum(|discarded coeffs|) bounds the uniform error of the trimmed series
    trimmed = coeffs[: kmax[0], : kmax[1], : kmax[2]]
    for _ in range(5):
        ts = rng.uniform(-1.0, 1.0, size=3)
        full = direct_eval(coeffs, ts)
        cut = direct_eval(trimmed, ts)
        assert abs(full - cut) <= discarded + 1e-12


def test_prune_keeps_everything_when_budget_is_zero():
    rng = np.random.default_rng(5)
    coeffs = decaying_coeffs(rng, (5, 5))
    kmax, discarded = prune_trailing_slices_greedy_nd(coeffs, delta=0.0)
    assert kmax == coeffs.shape
    assert discarded == 0.0


def test_prune_all_zero_coefficients():
    kmax, discarded = prune_trailing_slices_greedy_nd(np.zeros((4, 4)), delta=1e-8)
    assert kmax == (0, 0)
    assert discarded == 0.0
