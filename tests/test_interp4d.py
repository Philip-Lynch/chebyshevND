"""Tests for ChebyshevInterpolant4D."""

import numpy as np
import pytest

from chebyshevnd import ChebyshevInterpolant4D
from helpers import sample_on_grid

F = lambda x, y, z, w: np.sin(x) * np.cos(y) * np.exp(-z) * (w**3 + w)
DOMAINS = [(0.0, 2.0), (-1.0, 3.0), (1.0, 4.0), (-2.0, 0.0)]


def build(Ns=(10, 10, 10, 10), nodes="Chebyshev"):
    axes, vals = sample_on_grid(F, DOMAINS, Ns, nodes=nodes)
    interp = ChebyshevInterpolant4D(
        vals, *DOMAINS[0], *DOMAINS[1], *DOMAINS[2], *DOMAINS[3], nodes=nodes
    )
    return axes, vals, interp


def sample_points(n):
    rng = np.random.default_rng(23)
    return [tuple(rng.uniform(a, b) for a, b in DOMAINS) for _ in range(n)]


@pytest.mark.parametrize("nodes", ["Chebyshev", "GaussLobatto"])
def test_reproduces_grid_values_exactly(nodes):
    axes, vals, interp = build(Ns=(7, 7, 7, 7), nodes=nodes)
    for idx in [(0, 0, 0, 0), (3, 4, 2, 5), (6, 6, 6, 6), (1, 5, 3, 2)]:
        point = tuple(axes[d][i] for d, i in enumerate(idx))
        assert interp.evaluate(*point) == pytest.approx(vals[idx], abs=1e-10)


def test_tight_round_trip_off_grid():
    # cos(y) on a half-width-2 domain is the slowest-converging factor;
    # 14 coefficients push its truncation error below 1e-8
    _, _, interp = build(Ns=(12, 14, 12, 8))
    for p in sample_points(20):
        assert interp.evaluate(*p) == pytest.approx(F(*p), abs=1e-7)


def test_optimize_reduces_work_and_preserves_accuracy():
    _, _, interp = build(Ns=(12, 12, 12, 12))
    points = sample_points(8)
    before = [interp.evaluate(*p) for p in points]

    interp.optimize(rel_tol=1e-8)
    assert interp.kmax < (12, 12, 12, 12)  # at least one axis must shrink
    after = [interp.evaluate(*p) for p in points]
    np.testing.assert_allclose(after, before, atol=1e-6)
    for p, val in zip(points, after):
        assert val == pytest.approx(F(*p), abs=1e-6)


def test_precompute_x_matches_full_evaluate():
    _, _, interp = build(Ns=(8, 8, 8, 8))
    x0 = 1.1
    interp.precompute_x(x0)
    for _, y, z, w in sample_points(6):
        assert interp.evaluate_yzw(y, z, w) == pytest.approx(
            interp.evaluate(x0, y, z, w), abs=1e-10
        )


def test_precompute_x_matches_evaluate_after_optimize():
    _, _, interp = build(Ns=(12, 12, 12, 12))
    interp.optimize(rel_tol=1e-8)
    x0 = 0.6
    interp.precompute_x(x0)
    # Both paths use the same truncated series, so they agree to rounding. A
    # looser tolerance here used to hide precompute_x ignoring kmax[0].
    for _, y, z, w in sample_points(6):
        assert interp.evaluate_yzw(y, z, w) == pytest.approx(
            interp.evaluate(x0, y, z, w), abs=1e-13
        )


def test_evaluate_yzw_requires_precompute():
    _, _, interp = build(Ns=(6, 6, 6, 6))
    with pytest.raises(ValueError):
        interp.evaluate_yzw(0.0, 2.0, -1.0)


def test_absolute_error_includes_w_axis():
    # White-box: a trailing coefficient along the w-axis must contribute to the
    # error estimate (it was silently ignored before).
    _, _, interp = build(Ns=(6, 6, 6, 6))
    c = np.zeros((6, 6, 6, 6))
    c[0, 0, 0, -1] = 7.0
    interp.coeffs = c
    assert interp.absolute_error() == pytest.approx(7.0 * 6)
