"""Tests for precontraction: fixing some axes of an interpolant."""

import itertools

import numpy as np
import pytest

from chebyshevnd import ChebyshevInterpolantND, OutOfDomainError
from helpers import sample_on_grid

DOMAINS = [(0.0, 0.25), (-3.0, -1.0), (0.0, 0.9), (-0.5, 0.5)]
NS = (8, 20, 14, 6)
NAMES = ("nu", "u", "e", "chi")
FUNCS = {
    "dp": lambda nu, u, e, chi: np.exp(-nu) * np.cos(u) * (1 + e**2) * (1 + 0.3 * chi),
    "de": lambda nu, u, e, chi: np.sin(u + nu) * e * np.exp(chi),
    "phase": lambda nu, u, e, chi: np.exp(1j * (u + e)) * (1 - nu) * (2 + chi),
}


def build(dtype=float, multi=True):
    axes = sample_on_grid(lambda *x: x[0], DOMAINS, NS)[0]
    grids = np.meshgrid(*axes, indexing="ij")
    vals = np.stack([f(*grids) for f in FUNCS.values()], axis=-1)
    vals = vals.real if dtype is float else vals
    if multi:
        return ChebyshevInterpolantND(vals, DOMAINS, axis_names=NAMES, output_names=list(FUNCS))
    return ChebyshevInterpolantND(vals[..., 0], DOMAINS, axis_names=NAMES)


def random_points(n, seed=2):
    rng = np.random.default_rng(seed)
    lo, hi = np.array(DOMAINS).T
    return rng.uniform(lo, hi, size=(n, len(DOMAINS)))


def check_matches_parent(parent, fixed_axes, n=10):
    for point in random_points(n):
        fixed = {NAMES[ax]: point[ax] for ax in fixed_axes}
        child = parent.precontract(fixed)
        live = [i for i in range(len(NAMES)) if i not in fixed_axes]
        np.testing.assert_allclose(child.evaluate(*point[live]), parent.evaluate(*point),
                                   rtol=1e-13, atol=1e-15)


SUBSETS = [s for r in (1, 2, 3) for s in itertools.combinations(range(4), r)]


@pytest.mark.parametrize("fixed_axes", SUBSETS, ids=lambda s: "+".join(NAMES[i] for i in s))
def test_every_subset_matches_full_evaluation(fixed_axes):
    check_matches_parent(build(), fixed_axes)


@pytest.mark.parametrize("dtype", [float, complex])
@pytest.mark.parametrize("multi", [False, True])
def test_complex_and_single_output(dtype, multi):
    check_matches_parent(build(dtype, multi), (0, 2))


def test_matches_parent_after_optimize():
    parent = build(complex)
    parent.optimize(1e-6)
    child = parent.precontract({"nu": 0.1, "chi": -0.2})
    assert child.kmax == (parent.kmax[1], parent.kmax[2])
    check_matches_parent(parent, (0, 3))


def test_child_metadata():
    parent = build()
    child = parent.precontract({"chi": 0.1, "nu": 0.2})
    assert child.axis_names == ("u", "e")
    assert child.domains == (DOMAINS[1], DOMAINS[2])
    assert child.grid_shape == (NS[1], NS[2])
    assert child.output_names == parent.output_names
    assert child.dtype == parent.dtype
    assert child.domain_tol == parent.domain_tol
    assert child.nodes == parent.nodes


def test_child_inherits_error_bound_and_can_optimize():
    parent = build()
    parent.optimize(1e-9)
    child = parent.precontract({"nu": 0.2})
    np.testing.assert_allclose(child.absolute_error(), parent.absolute_error())
    points = random_points(20)[:, 1:]
    before = np.array([child.evaluate(*p) for p in points])
    scales = np.abs(child.get_coefficients(full=True)).reshape(-1, 3).max(axis=0)
    child.optimize(1e-5)
    after = np.array([child.evaluate(*p) for p in points])
    assert np.all(np.abs(after - before).max(axis=0) <= 1e-5 * scales)
    assert np.all(child.absolute_error() >= parent.absolute_error())


def test_child_without_full_tensor():
    parent = build()
    parent.optimize(1e-8, keep_full=False)
    child = parent.precontract({"nu": 0.2})
    with pytest.raises(RuntimeError):
        child.optimize(1e-6)


def test_parent_is_unchanged():
    parent = build()
    coeffs = parent.get_coefficients().copy()
    point = random_points(1)[0]
    before = parent.evaluate(*point)
    parent.precontract({"u": -2.0})
    np.testing.assert_array_equal(parent.get_coefficients(), coeffs)
    np.testing.assert_array_equal(parent.evaluate(*point), before)


def test_child_supports_batches():
    parent = build()
    child = parent.precontract({"nu": 0.2, "chi": 0.0})
    pts = random_points(15)[:, 1:3]
    np.testing.assert_allclose(child.evaluate_batch(pts), [child.evaluate(*p) for p in pts],
                               rtol=1e-13, atol=1e-16)


def test_axes_by_index_and_negative_index():
    parent = build()
    by_name = parent.precontract({"nu": 0.1, "chi": 0.2})
    by_index = parent.precontract({0: 0.1, -1: 0.2})
    assert by_index.axis_names == by_name.axis_names
    np.testing.assert_array_equal(by_index.get_coefficients(), by_name.get_coefficients())


def test_chained_precontraction_equals_joint():
    parent = build()
    joint = parent.precontract({"nu": 0.1, "chi": 0.2})
    chained = parent.precontract({"nu": 0.1}).precontract({"chi": 0.2})
    np.testing.assert_allclose(chained.get_coefficients(), joint.get_coefficients(), rtol=1e-13, atol=1e-16)


@pytest.mark.parametrize("fixed, error", [
    ({"q": 0.1}, KeyError),
    ({7: 0.1}, KeyError),
    ({1.5: 0.1}, TypeError),
    ({"nu": 0.1, 0: 0.1}, ValueError),
    ({}, ValueError),
    ({"nu": 0.1, "u": -2.0, "e": 0.1, "chi": 0.0}, ValueError),
    ({"nu": np.array([0.1, 0.2])}, TypeError),
])
def test_bad_fixed_arguments(fixed, error):
    with pytest.raises(error):
        build().precontract(fixed)


def test_fixed_argument_must_be_a_mapping():
    with pytest.raises(TypeError, match="mapping"):
        build().precontract([("nu", 0.1)])


def test_fixed_value_outside_domain():
    with pytest.raises(OutOfDomainError, match=r"Fixed value .*nu = 0.3 not in \[0.0, 0.25\]"):
        build().precontract({"nu": 0.3, "chi": 0.0})
    with pytest.raises(OutOfDomainError, match="chi = nan"):
        build().precontract({"chi": np.nan})
    # Exact boundaries are inside
    build().precontract({"nu": 0.25, "chi": -0.5})
