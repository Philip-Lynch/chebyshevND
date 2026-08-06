"""Tests for ChebyshevInterpolant2D."""

import numpy as np
import pytest

from chebyshevnd import ChebyshevInterpolant2D
from helpers import sample_on_grid

F = lambda x, y: np.exp(-(x**2)) * np.cos(2 * y)
DOMAINS = [(0.0, 2.0), (-1.0, 1.0)]


def build(Ns=(30, 32), nodes="Chebyshev"):
    axes, vals = sample_on_grid(F, DOMAINS, Ns, nodes=nodes)
    interp = ChebyshevInterpolant2D(vals, *DOMAINS[0], *DOMAINS[1], nodes=nodes)
    return axes, vals, interp


@pytest.mark.parametrize("nodes", ["Chebyshev", "GaussLobatto"])
def test_reproduces_grid_values_exactly(nodes):
    axes, vals, interp = build(Ns=(10, 12), nodes=nodes)
    for i, x in enumerate(axes[0]):
        for j, y in enumerate(axes[1]):
            assert interp.evaluate(x, y) == pytest.approx(vals[i, j], abs=1e-11)


def test_tight_round_trip_off_grid():
    _, _, interp = build()
    for x in np.linspace(*DOMAINS[0], 7):
        for y in np.linspace(*DOMAINS[1], 7):
            assert interp.evaluate(x, y) == pytest.approx(F(x, y), abs=1e-12)


def test_optimize_reduces_work_and_preserves_accuracy():
    _, _, interp = build()
    points = [(x, y) for x in np.linspace(*DOMAINS[0], 5) for y in np.linspace(*DOMAINS[1], 5)]
    before = [interp.evaluate(x, y) for x, y in points]

    interp.optimize(rel_tol=1e-8)
    n_rows, max_count = interp.effective_grid()
    assert n_rows < 30 or max_count < 32  # smooth function: truncation must bite
    after = [interp.evaluate(x, y) for x, y in points]
    np.testing.assert_allclose(after, before, atol=1e-6)
    for (x, y), val in zip(points, after):
        assert val == pytest.approx(F(x, y), abs=1e-6)


def test_effective_grid_bounds():
    _, _, interp = build(Ns=(20, 24))
    interp.optimize(rel_tol=1e-8)
    n_rows, max_count = interp.effective_grid()
    assert 1 <= n_rows <= 20
    assert 1 <= max_count <= 24


def test_absolute_error_scales_with_matching_axis():
    # White-box: a single trailing coefficient along the y-axis must be scaled
    # by Ny, not Nx.
    _, _, interp = build(Ns=(4, 8))
    c = np.zeros((4, 8))
    c[0, -1] = 3.0
    interp.coeffs = c
    assert interp.absolute_error() == pytest.approx(3.0 * 8)


def test_error_estimate_bounds_true_error_for_smooth_function():
    _, _, interp = build(Ns=(12, 12))
    true_err = max(
        abs(interp.evaluate(x, y) - F(x, y))
        for x in np.linspace(*DOMAINS[0], 9)
        for y in np.linspace(*DOMAINS[1], 9)
    )
    assert true_err <= interp.absolute_error() * 10
