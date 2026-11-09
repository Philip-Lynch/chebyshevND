"""Tests for ChebyshevInterpolantND."""

import numpy as np
import pytest

from chebyshevnd import (
    ChebyshevInterpolant1D,
    ChebyshevInterpolant2D,
    ChebyshevInterpolant3D,
    ChebyshevInterpolant4D,
    ChebyshevInterpolantND,
)
from helpers import sample_on_grid

# One smooth test function per dimension, with distinct domains and node
# counts on every axis so that axis or bound mix-ups cannot cancel out
CASES = {
    1: (lambda x: np.exp(x) * np.sin(2 * x), [(0.0, 2.0)], (30,)),
    2: (lambda x, y: np.sin(x) * np.cos(y), [(-1.0, 2.0), (0.0, 3.0)], (24, 18)),
    3: (lambda x, y, z: np.sin(x) * np.cos(y) * np.exp(-z),
        [(0.0, 2.0), (-1.0, 3.0), (1.0, 4.0)], (22, 16, 12)),
    4: (lambda x, y, z, w: np.sin(x) * np.cos(y) * np.exp(-z) * (w**3 + w),
        [(0.0, 2.0), (-1.0, 3.0), (1.0, 4.0), (-2.0, 0.0)], (13, 14, 12, 8)),
}
V01_CLASSES = {1: ChebyshevInterpolant1D, 2: ChebyshevInterpolant2D,
               3: ChebyshevInterpolant3D, 4: ChebyshevInterpolant4D}


def random_points(domains, n, seed=11):
    rng = np.random.default_rng(seed)
    return [tuple(rng.uniform(a, b) for a, b in domains) for _ in range(n)]


def build(dim, nodes="Chebyshev", **kwargs):
    f, domains, Ns = CASES[dim]
    axes, vals = sample_on_grid(f, domains, Ns, nodes=nodes)
    return f, domains, axes, vals, ChebyshevInterpolantND(vals, domains, nodes=nodes, **kwargs)


# ------------------------------------------------------------------ accuracy

@pytest.mark.parametrize("nodes", ["Chebyshev", "GaussLobatto"])
@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_matches_v01_classes(dim, nodes):
    f, domains, _, vals, nd = build(dim, nodes=nodes)
    bounds = [v for dom in domains for v in dom]
    old = V01_CLASSES[dim](vals, *bounds, nodes=nodes)
    for p in random_points(domains, 20):
        assert nd.evaluate(*p) == pytest.approx(old.evaluate(*p), rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("nodes", ["Chebyshev", "GaussLobatto"])
def test_reproduces_grid_values_exactly(nodes):
    _, _, axes, vals, nd = build(3, nodes=nodes)
    for i, j, k in [(0, 0, 0), (3, 7, 2), (21, 15, 11), (10, 4, 6)]:
        assert nd.evaluate(axes[0][i], axes[1][j], axes[2][k]) == pytest.approx(vals[i, j, k], abs=1e-13)


def test_5d_separable_function():
    f = lambda a, b, c, d, e: np.exp(0.3 * a) * np.cos(0.5 * b) * (1 + 0.2 * c) ** 2 * np.sin(0.4 * d + 0.1) * np.cos(e)
    domains = [(0.0, 1.0), (-1.0, 2.0), (0.5, 1.5), (-2.0, -1.0), (0.0, 0.5)]
    _, vals = sample_on_grid(f, domains, (12, 14, 6, 10, 11))
    nd = ChebyshevInterpolantND(vals, domains)
    for p in random_points(domains, 30):
        assert nd.evaluate(*p) == pytest.approx(f(*p), abs=1e-12)


def test_6d_separable_function():
    f = lambda a, b, c, d, e, g: np.cos(a) * np.exp(-b) * (c**2 + 1) * np.sin(d + 1) * np.cos(2 * e) * (1 + g)
    domains = [(0.0, 1.0), (0.0, 0.5), (-1.0, 1.0), (0.0, 1.0), (-0.5, 0.5), (1.0, 2.0)]
    _, vals = sample_on_grid(f, domains, (11, 10, 4, 12, 13, 3))
    nd = ChebyshevInterpolantND(vals, domains)
    for p in random_points(domains, 30):
        assert nd.evaluate(*p) == pytest.approx(f(*p), abs=1e-11)


def test_evaluate_returns_python_float():
    _, domains, _, _, nd = build(2)
    assert type(nd.evaluate(*random_points(domains, 1)[0])) is float


# ---------------------------------------------------------------- truncation

def test_optimize_bound_holds_pointwise():
    f, domains, _, _, nd = build(4)
    points = random_points(domains, 40)
    before = np.array([nd.evaluate(*p) for p in points])
    scale = np.abs(nd.get_coefficients(full=True)).max()

    rel_tol = 1e-6
    nd.optimize(rel_tol)
    assert nd.kmax < nd.grid_shape
    assert nd.get_coefficients().shape == nd.kmax
    after = np.array([nd.evaluate(*p) for p in points])
    # The discarded coefficient magnitudes bound the pointwise change
    assert np.max(np.abs(after - before)) <= rel_tol * scale


def test_optimize_restarts_from_full_tensor():
    _, domains, _, _, nd = build(3)
    p = random_points(domains, 1)[0]
    full = nd.evaluate(*p)
    nd.optimize(1e-3)
    loose = nd.kmax
    nd.optimize(1e-13)
    assert all(t >= l for t, l in zip(nd.kmax, loose))
    assert nd.evaluate(*p) == pytest.approx(full, abs=1e-12)


def test_optimize_without_full_tensor():
    _, _, _, _, nd = build(2)
    nd.optimize(1e-8, keep_full=False)
    with pytest.raises(RuntimeError):
        nd.optimize(1e-6)
    with pytest.raises(RuntimeError):
        nd.get_coefficients(full=True)


@pytest.mark.parametrize("rel_tol", [0.0, 1.0, -1e-3])
def test_optimize_rejects_bad_tolerance(rel_tol):
    _, _, _, _, nd = build(2)
    with pytest.raises(ValueError):
        nd.optimize(rel_tol)


def test_effective_grid_is_kmax():
    _, _, _, _, nd = build(3)
    assert nd.effective_grid() == nd.grid_shape
    nd.optimize(1e-6)
    assert nd.effective_grid() == nd.kmax


# ------------------------------------------------------------ error estimates

def test_error_estimate_matches_v01_and_bounds_actual_error():
    f, domains, _, vals, nd = build(3)
    old = ChebyshevInterpolant3D(vals, *[v for dom in domains for v in dom])
    assert nd.absolute_error() == pytest.approx(old.absolute_error(), rel=1e-12)
    assert nd.relative_error() == pytest.approx(old.relative_error(), rel=1e-12)
    actual = max(abs(nd.evaluate(*p) - f(*p)) for p in random_points(domains, 200))
    assert actual <= nd.absolute_error()


def test_absolute_error_scales_with_matching_axis():
    # White-box: a trailing coefficient along axis 1 is scaled by N_1, as in
    # the 2D-4D classes
    vals = np.zeros((4, 9, 5))
    nd = ChebyshevInterpolantND(vals, [(-1, 1)] * 3)
    c = np.zeros((4, 9, 5, 1))
    c[0, -1, 0, 0] = 5.0
    from chebyshevnd.interpnd import _trailing_error_estimate
    assert _trailing_error_estimate(np.abs(c))[0] == pytest.approx(5.0 * 9)


def test_absolute_error_includes_truncation():
    _, _, _, _, nd = build(3)
    before = nd.absolute_error()
    nd.optimize(1e-6)
    assert nd.absolute_error() > before


# ---------------------------------------------------------------- validation

def test_axis_names_default_and_custom():
    _, domains, _, vals, nd = build(3)
    assert nd.axis_names == ("x0", "x1", "x2")
    named = ChebyshevInterpolantND(vals, domains, axis_names=("nu", "u", "e"))
    assert named.axis_names == ("nu", "u", "e")
    assert "nu" in repr(named)


def test_properties():
    _, domains, _, _, nd = build(3)
    assert nd.ndim == 3
    assert nd.grid_shape == (22, 16, 12)
    assert nd.domains == tuple(domains)
    assert nd.nodes == "Chebyshev"


@pytest.mark.parametrize("names", [("a", "b"), ("a", "a", "b")])
def test_bad_axis_names(names):
    _, domains, _, vals, _ = build(3)
    with pytest.raises(ValueError):
        ChebyshevInterpolantND(vals, domains, axis_names=names)


def test_domain_count_must_match_vals():
    _, domains, _, vals, _ = build(3)
    with pytest.raises(ValueError, match="one \\(lower, upper\\) pair per axis"):
        ChebyshevInterpolantND(vals, domains[:2])


@pytest.mark.parametrize("bad", [(2.0, 0.0), (1.0, 1.0), (0.0, np.inf), (0.0,), "ab"])
def test_bad_domain_pairs(bad):
    with pytest.raises(ValueError):
        ChebyshevInterpolantND(np.ones((4, 4)), [(0.0, 1.0), bad])


def test_bad_node_family():
    with pytest.raises(ValueError):
        ChebyshevInterpolantND(np.ones((4, 4)), [(0, 1), (0, 1)], nodes="bogus")


def test_gauss_lobatto_needs_two_nodes_per_axis():
    with pytest.raises(ValueError):
        ChebyshevInterpolantND(np.ones((4, 1)), [(0, 1), (0, 1)], nodes="GaussLobatto")
    # A single root node per axis is a valid (constant) interpolant
    nd = ChebyshevInterpolantND(np.full((4, 1), 2.0), [(0, 1), (0, 1)])
    assert nd.evaluate(0.3, 0.7) == pytest.approx(2.0)


def test_evaluate_argument_checks():
    _, _, _, _, nd = build(3)
    with pytest.raises(TypeError):
        nd.evaluate(0.5, 0.5)
    with pytest.raises(TypeError):
        nd.evaluate(np.array([0.5, 0.6]), 0.5, 2.0)
