"""Tests for precontraction: fixing some axes of an interpolant."""

import itertools

import numpy as np
import pytest

from chebyshevnd import ChebyshevBatch, ChebyshevInterpolantND, OutOfDomainError
from chebyshevnd.backend import array_namespace, to_numpy
from chebyshevnd.batched import precontract_tables
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


def test_cached_layout_follows_optimize():
    # precontract caches the table with the fixed axes first; optimize()
    # must invalidate that cache
    parent = build()
    before = parent.precontract({"nu": 0.1, "chi": 0.2})
    assert before.kmax == (NS[1], NS[2])
    parent.optimize(1e-5)
    after = parent.precontract({"nu": 0.1, "chi": 0.2})
    assert after.kmax == (parent.kmax[1], parent.kmax[2])
    check_matches_parent(parent, (0, 3))
    batch = parent.precontract_batch({"nu": np.array([0.1]), "chi": np.array([0.2])})
    assert batch.kmax == after.kmax


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


# ------------------------------------------------------------ batched (2b)


def member_values(n, fixed_axes, seed=9):
    lo, hi = np.array(DOMAINS).T
    rng = np.random.default_rng(seed)
    return rng.uniform(lo[list(fixed_axes)], hi[list(fixed_axes)], size=(n, len(fixed_axes)))


@pytest.mark.parametrize("fixed_axes", [(0,), (0, 3), (3, 0), (1, 2), (0, 2, 3)],
                         ids=lambda s: "+".join(NAMES[i] for i in s))
@pytest.mark.parametrize("dtype", [float, complex])
def test_batch_tables_match_single_precontraction(fixed_axes, dtype):
    parent = build(dtype)
    values = member_values(6, fixed_axes)
    batch = parent.precontract_batch({NAMES[ax]: values[:, i] for i, ax in enumerate(fixed_axes)})
    assert isinstance(batch, ChebyshevBatch)
    assert batch.n_members == 6
    for j in range(6):
        single = parent.precontract({NAMES[ax]: values[j, i] for i, ax in enumerate(fixed_axes)})
        np.testing.assert_allclose(batch.tables[j], single.get_coefficients(), rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("dtype", [float, complex])
@pytest.mark.parametrize("multi", [False, True])
def test_batch_evaluate_matches_full_evaluation(dtype, multi):
    parent = build(dtype, multi)
    points = random_points(8)
    batch = parent.precontract_batch({"nu": points[:, 0], "chi": points[:, 3]})
    got = batch.evaluate(points[:, 1:3])
    expected = np.array([parent.evaluate(*p) for p in points])
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-13, atol=1e-15)


def test_batch_after_optimize():
    parent = build()
    parent.optimize(1e-6)
    points = random_points(5)
    batch = parent.precontract_batch({"nu": points[:, 0]})
    assert batch.kmax == parent.kmax[1:]
    np.testing.assert_allclose(batch.evaluate(points[:, 1:]), [parent.evaluate(*p) for p in points],
                               rtol=1e-13, atol=1e-15)


def test_batch_metadata():
    parent = build()
    batch = parent.precontract_batch({"chi": np.zeros(3), "nu": np.full(3, 0.1)})
    assert batch.axis_names == ("u", "e")
    assert batch.domains == (DOMAINS[1], DOMAINS[2])
    assert batch.output_names == parent.output_names
    assert batch.output_index == parent.output_index
    assert batch.n_outputs == 3 and batch.ndim == 2
    assert batch.dtype == parent.dtype
    assert "members=3" in repr(batch)


def test_scalars_broadcast_to_every_member():
    parent = build()
    nus = np.array([0.05, 0.1, 0.2])
    mixed = parent.precontract_batch({"nu": nus, "chi": 0.3})
    explicit = parent.precontract_batch({"nu": nus, "chi": np.full(3, 0.3)})
    np.testing.assert_array_equal(mixed.tables, explicit.tables)


def test_batch_argument_checks():
    parent = build()
    with pytest.raises(ValueError, match="different lengths"):
        parent.precontract_batch({"nu": np.zeros(3), "chi": np.zeros(4)})
    with pytest.raises(ValueError, match="at least one array"):
        parent.precontract_batch({"nu": 0.1})
    with pytest.raises(ValueError, match="1D array"):
        parent.precontract_batch({"nu": np.zeros((2, 2))})
    with pytest.raises(ValueError, match="nothing to interpolate"):
        parent.precontract_batch({n: np.zeros(2) for n in NAMES})


def test_batch_fixed_values_outside_domain():
    parent = build()
    nus = np.array([0.1, 0.3, 0.2, np.nan])
    chis = np.array([0.0, 0.0, 0.9, 0.0])
    with pytest.raises(OutOfDomainError, match=r"3 of 4 members.*member 1: nu = 0.3 not in"):
        parent.precontract_batch({"nu": nus, "chi": chis})
    _, status = parent.precontract_batch({"nu": nus, "chi": chis}, return_status=True)
    # Bits use the parent's axis numbering: nu is bit 0, chi is bit 3
    np.testing.assert_array_equal(status, [0, 1, 8, 1])


def test_batch_evaluate_status_and_checks():
    parent = build()
    batch = parent.precontract_batch({"nu": np.full(3, 0.1), "chi": np.zeros(3)})
    points = np.array([[-2.0, 0.3], [-0.5, 0.3], [-2.0, 1.2]])
    _, status = batch.evaluate(points, return_status=True)
    np.testing.assert_array_equal(status, [0, 1, 2])
    with pytest.raises(OutOfDomainError, match="2 of 3 members.*member 1: u = -0.5"):
        batch.evaluate(points)
    with pytest.raises(ValueError, match=r"shape \(3, 2\)"):
        batch.evaluate(points[:2])


def test_empty_batch():
    batch = build().precontract_batch({"nu": np.zeros(0)})
    assert batch.n_members == 0
    values, status = batch.evaluate(np.zeros((0, 3)), return_status=True)
    assert values.shape == (0, 3) and status.shape == (0,)


def test_precontract_tables_validates_arguments():
    coeffs = build().get_coefficients()
    a, b = np.array(DOMAINS).T
    with pytest.raises(ValueError, match="shape"):
        precontract_tables(coeffs, a, b, [0], np.zeros((3, 2)))
    with pytest.raises(ValueError, match="distinct"):
        precontract_tables(coeffs, a, b, [0, 0], np.zeros((3, 2)))


# ----------------------------------------------------------------- GPU (CuPy)

@pytest.fixture
def cupy():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as err:  # CUDA driver or runtime missing
        pytest.skip(f"CUDA unavailable: {err}")
    return cp


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [float, complex])
def test_gpu_batch_matches_cpu(cupy, dtype):
    parent = build(dtype)
    points = random_points(50)
    cpu = parent.precontract_batch({"nu": points[:, 0], "chi": points[:, 3]})
    gpu = parent.precontract_batch({"nu": cupy.asarray(points[:, 0]), "chi": cupy.asarray(points[:, 3])})
    assert array_namespace(gpu.tables) is cupy
    np.testing.assert_allclose(to_numpy(gpu.tables), cpu.tables, rtol=1e-13, atol=1e-15)
    values = gpu.evaluate(cupy.asarray(points[:, 1:3]))
    np.testing.assert_allclose(to_numpy(values), cpu.evaluate(points[:, 1:3]), rtol=1e-13, atol=1e-15)


@pytest.mark.gpu
def test_gpu_batch_status(cupy):
    parent = build()
    nus = cupy.asarray([0.1, 0.3])
    _, status = parent.precontract_batch({"nu": nus}, return_status=True)
    np.testing.assert_array_equal(to_numpy(status), [0, 1])
