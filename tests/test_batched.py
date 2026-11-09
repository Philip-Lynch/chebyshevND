"""Tests for batched evaluation (NumPy here; CuPy under the `gpu` marker)."""

import numpy as np
import pytest

from chebyshevnd import ChebyshevInterpolantND, OutOfDomainError, evaluate_batch
from chebyshevnd.backend import array_namespace, to_numpy
from chebyshevnd.batched import chebyshev_basis_batch
from helpers import sample_on_grid

DOMAINS = [(0.0, 2.0), (-1.0, 3.0), (1.0, 4.0)]
NS = (22, 16, 12)
FUNCS = [
    lambda x, y, z: np.sin(x) * np.cos(y) * np.exp(-z),
    lambda x, y, z: np.exp(1j * x) * np.cos(0.5 * y) / z,
    lambda x, y, z: 1e-3 * (x * y + z**2),
]


def grid_values(funcs, dtype):
    axes = sample_on_grid(lambda *x: x[0], DOMAINS, NS)[0]
    grids = np.meshgrid(*axes, indexing="ij")
    vals = np.stack([f(*grids) for f in funcs], axis=-1)
    return vals.real if dtype is float else vals.astype(dtype)


def build(dtype=float, multi=True):
    vals = grid_values(FUNCS, dtype)
    if multi:
        return ChebyshevInterpolantND(vals, DOMAINS, output_names=["a", "b", "c"])
    return ChebyshevInterpolantND(vals[..., 0], DOMAINS)


def random_points(n, seed=5):
    rng = np.random.default_rng(seed)
    lo, hi = np.array(DOMAINS).T
    return rng.uniform(lo, hi, size=(n, len(DOMAINS)))


@pytest.mark.parametrize("dtype", [float, complex])
@pytest.mark.parametrize("multi", [False, True])
def test_batch_matches_pointwise(dtype, multi):
    nd = build(dtype, multi)
    pts = random_points(50)
    got = nd.evaluate_batch(pts)
    expected = np.array([nd.evaluate(*p) for p in pts])
    assert got.shape == expected.shape
    assert got.dtype == nd.dtype
    np.testing.assert_allclose(got, expected, rtol=1e-13, atol=1e-16)


def test_batch_after_optimize_uses_truncation():
    nd = build()
    nd.optimize(1e-6)
    pts = random_points(20)
    np.testing.assert_allclose(nd.evaluate_batch(pts), [nd.evaluate(*p) for p in pts], rtol=1e-13, atol=1e-16)


@pytest.mark.parametrize("dtype", [float, complex])
def test_per_member_tables(dtype):
    # Each batch member gets its own table, as after per-waveform
    # precontraction. Members here differ by an overall factor and a phase.
    base = build(dtype).get_coefficients()
    factors = np.array([1.0, -2.0, 0.5, 3.0])
    if dtype is complex:
        factors = factors * np.exp(1j * np.arange(4))
    tables = np.stack([f * base for f in factors])
    pts = random_points(4)
    nd = build(dtype)
    a, b = np.array(DOMAINS).T
    values, status = evaluate_batch(tables, a, b, pts)
    assert values.shape == (4, 3)
    assert np.all(status == 0)
    for i, p in enumerate(pts):
        np.testing.assert_allclose(values[i], factors[i] * nd.evaluate(*p), rtol=1e-13, atol=1e-16)


def test_per_member_count_must_match_points():
    tables = np.zeros((3,) + NS + (1,))
    with pytest.raises(ValueError, match="3 members but 2 points"):
        evaluate_batch(tables, [0, 0, 0], [1, 1, 1], np.zeros((2, 3)))


def test_status_bitmask_and_errors():
    nd = build()
    pts = random_points(5)
    pts[1, 0] = 2.5            # axis 0 out  -> 1
    pts[2, 2] = 0.0            # axis 2 out  -> 4
    pts[3, [0, 2]] = [-1, 9]   # axes 0 and 2 -> 5
    pts[4, 1] = np.nan         # axis 1 NaN  -> 2
    values, status = nd.evaluate_batch(pts, return_status=True)
    assert values.shape == (5, 3)
    np.testing.assert_array_equal(status, [0, 1, 4, 5, 2])
    with pytest.raises(OutOfDomainError, match="4 of 5 points.*first is point 1: x0 = 2.5 not in"):
        nd.evaluate_batch(pts)


def test_boundary_points_are_inside():
    nd = build()
    lo, hi = np.array(DOMAINS).T
    _, status = nd.evaluate_batch(np.array([lo, hi]), return_status=True)
    np.testing.assert_array_equal(status, [0, 0])


def test_one_dimensional_accepts_flat_points():
    x = sample_on_grid(np.exp, [(0.0, 1.0)], (20,))[0][0]
    nd = ChebyshevInterpolantND(np.exp(x), [(0.0, 1.0)])
    pts = np.linspace(0.0, 1.0, 7)
    np.testing.assert_allclose(nd.evaluate_batch(pts), np.exp(pts), rtol=1e-14)


def test_point_shape_is_checked():
    nd = build()
    with pytest.raises(ValueError, match=r"shape \(B, 3\)"):
        nd.evaluate_batch(np.zeros((4, 2)))
    with pytest.raises(ValueError):
        nd.evaluate_batch(np.zeros(3))


def test_empty_batch():
    nd = build()
    values, status = nd.evaluate_batch(np.zeros((0, 3)), return_status=True)
    assert values.shape == (0, 3) and status.shape == (0,)


def test_chunking_gives_same_result():
    nd = build(complex)
    pts = random_points(37)
    a, b = np.array(DOMAINS).T
    whole, _ = evaluate_batch(nd.get_coefficients(), a, b, pts)
    # Small enough to force a chunk of a single point
    chunked, _ = evaluate_batch(nd.get_coefficients(), a, b, pts, max_chunk_bytes=1)
    # BLAS may sum in a different order for different matrix sizes, so the
    # results agree to rounding rather than bit for bit
    np.testing.assert_allclose(whole, chunked, rtol=1e-13, atol=1e-16)


def test_basis_matches_cosine_formula_inside():
    ts = np.linspace(-0.99, 0.99, 11)
    T = chebyshev_basis_batch(ts, 9, np)
    np.testing.assert_allclose(T, np.cos(np.outer(np.arccos(ts), np.arange(9))), atol=1e-14)


def test_array_namespace_defaults_to_numpy():
    assert array_namespace(np.zeros(2), [1.0], 3.0) is np
    np.testing.assert_array_equal(to_numpy([1, 2]), np.array([1, 2]))


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
@pytest.mark.parametrize("multi", [False, True])
def test_gpu_matches_cpu(cupy, dtype, multi):
    nd = build(dtype, multi)
    pts = random_points(200)
    cpu = nd.evaluate_batch(pts)
    gpu = nd.evaluate_batch(cupy.asarray(pts))
    assert array_namespace(gpu) is cupy
    np.testing.assert_allclose(to_numpy(gpu), cpu, rtol=1e-13, atol=1e-16)


@pytest.mark.gpu
def test_gpu_table_cache_follows_optimize(cupy):
    nd = build()
    pts = cupy.asarray(random_points(30))
    nd.evaluate_batch(pts)
    nd.optimize(1e-6)
    np.testing.assert_allclose(to_numpy(nd.evaluate_batch(pts)), nd.evaluate_batch(to_numpy(pts)),
                               rtol=1e-13, atol=1e-16)


@pytest.mark.gpu
def test_gpu_status_and_per_member_tables(cupy):
    nd = build(complex)
    pts = random_points(4)
    pts[2, 1] = 10.0
    tables = cupy.asarray(np.stack([nd.get_coefficients()] * 4))
    a, b = np.array(DOMAINS).T
    values, status = evaluate_batch(tables, a, b, cupy.asarray(pts))
    np.testing.assert_array_equal(to_numpy(status), [0, 0, 2, 0])
    expected, _ = evaluate_batch(np.stack([nd.get_coefficients()] * 4), a, b, pts)
    np.testing.assert_allclose(to_numpy(values), expected, rtol=1e-13, atol=1e-16)
