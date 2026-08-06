"""Tests for ChebyshevInterpolant3D."""

import numpy as np
import pytest

from chebyshevnd import ChebyshevInterpolant3D
from helpers import sample_on_grid

F = lambda x, y, z: np.sin(x) * np.cos(y) * np.exp(-z)
DOMAINS = [(0.0, 2.0), (-1.0, 3.0), (1.0, 4.0)]


def build(Ns=(20, 20, 20), nodes="Chebyshev"):
    axes, vals = sample_on_grid(F, DOMAINS, Ns, nodes=nodes)
    interp = ChebyshevInterpolant3D(vals, *DOMAINS[0], *DOMAINS[1], *DOMAINS[2], nodes=nodes)
    return axes, vals, interp


@pytest.mark.parametrize("nodes", ["Chebyshev", "GaussLobatto"])
def test_reproduces_grid_values_exactly(nodes):
    axes, vals, interp = build(Ns=(8, 9, 8), nodes=nodes)
    for i in (0, 3, 7):
        for j in (0, 4, 8):
            for k in (0, 3, 7):
                assert interp.evaluate(axes[0][i], axes[1][j], axes[2][k]) == pytest.approx(
                    vals[i, j, k], abs=1e-11
                )


def test_tight_round_trip_off_grid():
    _, _, interp = build()
    for x in np.linspace(*DOMAINS[0], 4):
        for y in np.linspace(*DOMAINS[1], 4):
            for z in np.linspace(*DOMAINS[2], 4):
                assert interp.evaluate(x, y, z) == pytest.approx(F(x, y, z), abs=1e-11)


def sample_points(n):
    rng = np.random.default_rng(19)
    return [
        tuple(rng.uniform(a, b) for a, b in DOMAINS)
        for _ in range(n)
    ]


def test_optimize_reduces_work_and_preserves_accuracy():
    _, _, interp = build()
    points = sample_points(10)
    before = [interp.evaluate(*p) for p in points]

    interp.optimize(rel_tol=1e-8)
    assert interp.kmax < (20, 20, 20)  # at least one axis must shrink
    after = [interp.evaluate(*p) for p in points]
    np.testing.assert_allclose(after, before, atol=1e-6)
    for p, val in zip(points, after):
        assert val == pytest.approx(F(*p), abs=1e-6)


def test_precompute_x_matches_full_evaluate():
    _, _, interp = build(Ns=(12, 12, 12))
    x0 = 1.3
    interp.precompute_x(x0)
    for _, y, z in sample_points(6):
        assert interp.evaluate_yz(y, z) == pytest.approx(interp.evaluate(x0, y, z), abs=1e-11)


def test_precompute_x_matches_evaluate_after_optimize():
    _, _, interp = build()
    interp.optimize(rel_tol=1e-8)
    x0 = 0.4
    interp.precompute_x(x0)
    for _, y, z in sample_points(6):
        assert interp.evaluate_yz(y, z) == pytest.approx(interp.evaluate(x0, y, z), abs=1e-9)


def test_evaluate_yz_requires_precompute():
    _, _, interp = build(Ns=(8, 8, 8))
    with pytest.raises(ValueError):
        interp.evaluate_yz(0.0, 2.0)


def test_absolute_error_scales_with_matching_axis():
    # White-box: a trailing coefficient along the y-axis must be scaled by Ny.
    _, _, interp = build(Ns=(4, 9, 5))
    c = np.zeros((4, 9, 5))
    c[0, -1, 0] = 5.0
    interp.coeffs = c
    assert interp.absolute_error() == pytest.approx(5.0 * 9)
