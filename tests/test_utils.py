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


@pytest.mark.parametrize("nodes", ["Chebyshev", "GaussLobatto"])
def test_coefficients_over_subset_of_axes_match_per_function(nodes):
    # A trailing axis of independent functions must not be transformed: the
    # result equals fitting each function on its own.
    rng = np.random.default_rng(3)
    vals = rng.standard_normal((7, 5, 4)) + 1j * rng.standard_normal((7, 5, 4))
    together = compute_chebyshev_coefficients_nd(vals, nodes=nodes, axes=(0, 1))
    for f in range(vals.shape[-1]):
        alone = compute_chebyshev_coefficients_nd(vals[..., f], nodes=nodes)
        np.testing.assert_allclose(together[..., f], alone, rtol=1e-13, atol=1e-15)


def test_coefficients_axes_default_is_all_axes():
    rng = np.random.default_rng(4)
    vals = rng.standard_normal((6, 5, 3))
    np.testing.assert_array_equal(
        compute_chebyshev_coefficients_nd(vals),
        compute_chebyshev_coefficients_nd(vals, axes=(0, 1, 2)),
    )


def test_renormalize_gl_only_checks_transformed_axes():
    # A length-1 function axis is fine as long as it is not transformed
    coeffs = np.ones((4, 5, 1))
    renormalize_coefficients_GL(coeffs, axes=(0, 1))
    with pytest.raises(ValueError):
        renormalize_coefficients_GL(coeffs)


def test_repeated_axes_rejected():
    with pytest.raises(ValueError):
        compute_chebyshev_coefficients_nd(np.ones((4, 4)), axes=(0, -2))


@pytest.mark.parametrize("nodes", ["Chebyshev", "GaussLobatto"])
def test_values_at_nodes_inverts_coefficients(nodes):
    from chebyshevnd import chebyshev_values_at_nodes_nd
    rng = np.random.default_rng(8)
    vals = rng.standard_normal((7, 5, 6)) + 1j * rng.standard_normal((7, 5, 6))
    coeffs = compute_chebyshev_coefficients_nd(vals, nodes=nodes)
    np.testing.assert_allclose(chebyshev_values_at_nodes_nd(coeffs, nodes=nodes), vals, rtol=1e-13, atol=1e-14)
    # Only the chosen axes are transformed
    partial = compute_chebyshev_coefficients_nd(vals, nodes=nodes, axes=(0, 2))
    np.testing.assert_allclose(chebyshev_values_at_nodes_nd(partial, nodes=nodes, axes=(0, 2)), vals,
                               rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("nodes", ["Chebyshev", "GaussLobatto"])
def test_values_at_nodes_match_direct_evaluation(nodes):
    # Independent check: evaluate the series at the actual node positions
    from chebyshevnd import chebyshev_values_at_nodes_nd
    rng = np.random.default_rng(9)
    coeffs = decaying_coeffs(rng, (6, 5))
    node_fn = chebyshev_nodes if nodes == "Chebyshev" else chebyshev_gauss_lobatto_nodes
    tx, ty = node_fn(6, -1.0, 1.0), node_fn(5, -1.0, 1.0)
    expected = np.array([[direct_eval(coeffs, [x, y]) for y in ty] for x in tx])
    np.testing.assert_allclose(chebyshev_values_at_nodes_nd(coeffs, nodes=nodes), expected, rtol=1e-13, atol=1e-14)


def test_values_at_nodes_checks_arguments():
    from chebyshevnd import chebyshev_values_at_nodes_nd
    with pytest.raises(ValueError):
        chebyshev_values_at_nodes_nd(np.ones((1, 4)), nodes="GaussLobatto")
    with pytest.raises(ValueError):
        chebyshev_values_at_nodes_nd(np.ones((4, 4)), nodes="bogus")
