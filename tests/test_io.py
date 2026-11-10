"""Tests for saving and loading tables (Step 6)."""

import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from chebyshevnd import (  # noqa: E402
    Axis,
    AxisMap,
    ChebyshevGrid,
    ChebyshevInterpolantND,
    MappedInterpolant,
    load,
    read_metadata,
    save,
)
from chebyshevnd.backend import array_namespace, to_numpy  # noqa: E402
from chebyshevnd.io import FORMAT_VERSION  # noqa: E402
from helpers import sample_on_grid  # noqa: E402
from test_mapped import DOMAINS, MAPS, SHAPE, F, G, fit_mapped, make_boundary, random_physical  # noqa: E402

BOX = [(0.0, 2.0), (-1.0, 3.0), (1.0, 4.0)]
NS = (18, 14, 10)


def build(dtype=float, multi=True, nodes="Chebyshev"):
    funcs = [lambda x, y, z: np.sin(x) * np.cos(y) * np.exp(-z),
             lambda x, y, z: np.exp(1j * x) * np.cos(0.5 * y) / z]
    axes = sample_on_grid(lambda *x: x[0], BOX, NS, nodes)[0]
    grids = np.meshgrid(*axes, indexing="ij")
    vals = np.stack([f(*grids) for f in funcs], axis=-1)
    vals = vals.real if dtype is float else vals
    if multi:
        return ChebyshevInterpolantND(vals, BOX, nodes=nodes, axis_names=["x", "y", "z"],
                                      output_names=["f", "g"], domain_tol=1e-10)
    return ChebyshevInterpolantND(vals[..., 0], BOX, nodes=nodes)


def points(n=25, seed=0, domains=BOX):
    lo, hi = np.array(domains).T
    return np.random.default_rng(seed).uniform(lo, hi, size=(n, len(domains)))


def assert_same_table(a, b):
    """Two ChebyshevInterpolantND are the same, bit for bit."""
    assert type(a) is type(b)
    for attr in ("axis_names", "output_names", "grid_shape", "kmax", "domains", "nodes",
                 "domain_tol", "dtype", "n_outputs"):
        assert getattr(a, attr) == getattr(b, attr), attr
    np.testing.assert_array_equal(a.get_coefficients(), b.get_coefficients())
    np.testing.assert_array_equal(a.absolute_error(), b.absolute_error())
    np.testing.assert_array_equal(a.relative_error(), b.relative_error())
    pts = points(domains=a.domains)
    np.testing.assert_array_equal(a.evaluate_batch(pts), b.evaluate_batch(pts))
    for p in pts[:5]:
        np.testing.assert_array_equal(a.evaluate(*p), b.evaluate(*p))


# ------------------------------------------------------------------ tables


@pytest.mark.parametrize("dtype", [float, complex])
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("optimized", [False, True])
def test_table_round_trip_is_bit_exact(tmp_path, dtype, multi, optimized):
    interp = build(dtype, multi)
    if optimized:
        interp.optimize(1e-6)
        assert interp.kmax != interp.grid_shape
    save(tmp_path / "t.h5", interp)
    loaded = load(tmp_path / "t.h5")
    assert_same_table(interp, loaded)
    # Without full=True only the evaluation tensor is stored, which for an
    # unoptimized table is the full tensor
    if optimized:
        with pytest.raises(RuntimeError, match="dropped"):
            loaded.get_coefficients(full=True)
    else:
        np.testing.assert_array_equal(loaded.get_coefficients(full=True), interp.get_coefficients(full=True))
        np.testing.assert_array_equal(loaded.values_at_nodes(), interp.values_at_nodes())


def test_full_tensor_round_trip(tmp_path):
    interp = build()
    interp.optimize(1e-6)
    save(tmp_path / "t.h5", interp, full=True)
    loaded = load(tmp_path / "t.h5")
    np.testing.assert_array_equal(loaded.get_coefficients(full=True), interp.get_coefficients(full=True))
    np.testing.assert_array_equal(loaded.values_at_nodes(), interp.values_at_nodes())
    loaded.optimize(1e-10)
    interp.optimize(1e-10)
    assert_same_table(interp, loaded)
    # An unoptimized table stores its tensor once
    save(tmp_path / "u.h5", build(), full=True)
    with h5py.File(tmp_path / "u.h5", "r") as f:
        assert "coeffs_full" not in f and bool(f.attrs["full_is_coeffs"])
    interp.optimize(1e-6, keep_full=False)
    with pytest.raises(ValueError, match="dropped"):
        save(tmp_path / "v.h5", interp, full=True)


def test_gauss_lobatto_and_derived_tables(tmp_path):
    interp = build(nodes="GaussLobatto")
    save(tmp_path / "gl.h5", interp)
    assert_same_table(interp, load(tmp_path / "gl.h5"))
    child = build().precontract({"y": 0.4})
    save(tmp_path / "child.h5", child)
    loaded = load(tmp_path / "child.h5")
    np.testing.assert_array_equal(loaded.get_coefficients(), child.get_coefficients())
    np.testing.assert_array_equal(loaded.absolute_error(), child.absolute_error())
    deriv = build(complex).derivative({"x": 1, "z": 1})
    save(tmp_path / "deriv.h5", deriv)
    np.testing.assert_array_equal(load(tmp_path / "deriv.h5").evaluate_batch(points()),
                                  deriv.evaluate_batch(points()))


@pytest.mark.parametrize("dtype", [float, complex])
def test_float32_storage(tmp_path, dtype):
    interp = build(dtype)
    save(tmp_path / "t.h5", interp, float32=True)
    with h5py.File(tmp_path / "t.h5", "r") as f:
        assert f["coeffs"].dtype == (np.complex64 if dtype is complex else np.float32)
    loaded = load(tmp_path / "t.h5")
    assert loaded.dtype == interp.dtype
    scale = np.abs(interp.get_coefficients()).max()
    np.testing.assert_allclose(loaded.get_coefficients(), interp.get_coefficients(), rtol=0,
                               atol=1e-7 * scale)
    pts = points()
    actual = np.abs(loaded.evaluate_batch(pts) - interp.evaluate_batch(pts)).max(axis=0)
    assert np.all(actual < 1e-6 * scale)
    # The stored error estimate grew by at least the rounding that was lost
    grown = loaded.absolute_error() - interp.absolute_error()
    assert np.all(grown >= actual) and np.all(grown < 1e-5 * scale)


# ------------------------------------------------------------ mapped, grids


@pytest.fixture(scope="module")
def mapped():
    return fit_mapped(["r", "p", "e", "q"], SHAPE, DOMAINS, MAPS, make_boundary(), [F, G],
                      output_names=["F", "G"])


def test_mapped_round_trip(tmp_path, mapped):
    save(tmp_path / "m.h5", mapped)
    loaded = load(tmp_path / "m.h5")
    assert isinstance(loaded, MappedInterpolant)
    assert loaded.maps == mapped.maps
    assert_same_table(mapped.table, loaded.table)
    np.testing.assert_array_equal(loaded.boundary.get_coefficients(), mapped.boundary.get_coefficients())
    x = random_physical(30, seed=3)
    for a, b in zip(mapped.value_and_grad_batch(x), loaded.value_and_grad_batch(x)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(mapped.value_and_grad(*x[0])[1], loaded.value_and_grad(*x[0])[1])
    for a, b in zip(mapped.raw(), loaded.raw()):
        np.testing.assert_array_equal(a, b)


def test_mapped_without_boundary(tmp_path):
    inv = AxisMap("power", power=-1)
    dom = inv.table_domain(2.0, 10.0)
    _, vals = sample_on_grid(lambda u, v: u + u**2 * v, [dom, (0.0, 1.0)], (5, 3))
    mapped = MappedInterpolant(ChebyshevInterpolantND(vals, [dom, (0.0, 1.0)], axis_names=["r", "v"]),
                               {"r": inv})
    save(tmp_path / "m.h5", mapped, float32=True)
    loaded = load(tmp_path / "m.h5")
    assert loaded.boundary is None and loaded.maps == mapped.maps
    assert loaded.evaluate(4.0, 0.5) == pytest.approx(0.25 + 0.5 / 16, rel=1e-7)


def test_grid_round_trip(tmp_path):
    _, ref = sample_on_grid(lambda r: np.sqrt(r), [(2.0, 10.0)], (12,))
    boundary = ChebyshevInterpolantND(ref[:, None], [(2.0, 10.0)], axis_names=["r"],
                                      output_names=["ref"])
    inv = AxisMap("power", power=-1)
    grid = ChebyshevGrid([Axis("r", 9, inv.table_domain(2.0, 10.0), inv),
                          Axis("p", 6, (-0.5, 0.5), AxisMap(lower="ref", width=0.2)),
                          Axis("s", 4, (0.0, 1.0))], nodes="GaussLobatto", boundary=boundary)
    save(tmp_path / "g.h5", grid, metadata={"purpose": "test"})
    loaded = load(tmp_path / "g.h5")
    assert isinstance(loaded, ChebyshevGrid)
    assert loaded.grid_hash() == grid.grid_hash()
    assert loaded.maps == grid.maps and loaded.nodes == "GaussLobatto"
    np.testing.assert_array_equal(loaded.physical_points(), grid.physical_points())
    # Chunks written with the original grid load with the loaded one
    x = grid.physical_points(0, 100)
    grid.save_chunk(tmp_path / "c0.npz", 0, 100, {"h": x[:, 1]})
    x = grid.physical_points(100, grid.size)
    grid.save_chunk(tmp_path / "c1.npz", 100, grid.size, {"h": x[:, 1]})
    assert set(loaded.load_chunks([tmp_path / "c0.npz", tmp_path / "c1.npz"])) == {"h"}
    with pytest.raises(ValueError, match="apply to tables"):
        save(tmp_path / "g2.h5", grid, float32=True)


def test_corrupted_grid_hash(tmp_path):
    grid = ChebyshevGrid([Axis("x", 4, (0.0, 1.0))])
    save(tmp_path / "g.h5", grid)
    with h5py.File(tmp_path / "g.h5", "a") as f:
        f["axes/0"].attrs["n"] = 5
    with pytest.raises(ValueError, match="grid_hash"):
        load(tmp_path / "g.h5")


# ------------------------------------------------------------- metadata etc


def test_metadata_round_trip(tmp_path):
    meta = {"model": "SEOBNRv6EHM", "kappa_max": 10, "q_range": [1.0, 20.0], "converged": True,
            "outputs": ["dudchi", "dedchi"], "spins": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "source": {"hash": "abc123", "lines": 4200}}
    save(tmp_path / "t.h5", build(), metadata=meta)
    read = read_metadata(tmp_path / "t.h5")
    assert read["model"] == "SEOBNRv6EHM" and read["kappa_max"] == 10 and read["converged"] is True
    np.testing.assert_array_equal(read["q_range"], [1.0, 20.0])
    assert read["outputs"] == ["dudchi", "dedchi"]
    np.testing.assert_array_equal(read["spins"], meta["spins"])
    assert read["source"] == {"hash": "abc123", "lines": 4200}
    assert read["kind"] == "ChebyshevInterpolantND" and read["format_version"] == FORMAT_VERSION
    assert "created" in read and "chebyshevnd_version" in read
    interp, read2 = load(tmp_path / "t.h5", return_metadata=True)
    assert read2["model"] == "SEOBNRv6EHM"
    with pytest.raises(ValueError, match="reserved"):
        save(tmp_path / "u.h5", build(), metadata={"kind": "x"})
    with pytest.raises(TypeError, match="metadata"):
        save(tmp_path / "v.h5", build(), metadata={"bad": None})


def test_several_tables_in_one_file(tmp_path, mapped):
    interp = build(complex)
    with h5py.File(tmp_path / "all.h5", "w") as f:
        save(f.create_group("rhs"), interp, metadata={"order": "0PA"})
        save(f.create_group("hamiltonian"), mapped)
        with pytest.raises(ValueError, match="not empty"):
            save(f["rhs"], interp)
        save(f["rhs"], build(), overwrite=True)
    with h5py.File(tmp_path / "all.h5", "r") as f:
        assert_same_table(load(f["rhs"]), build())
        assert isinstance(load(f["hamiltonian"]), MappedInterpolant)
        assert read_metadata(f["rhs"]).get("order") is None  # replaced by the overwrite


def test_errors(tmp_path):
    path = tmp_path / "t.h5"
    save(path, build())
    with pytest.raises(FileExistsError, match="overwrite"):
        save(path, build())
    save(path, build(multi=False), overwrite=True)
    assert load(path).output_names == ()
    with pytest.raises(TypeError, match="save"):
        save(tmp_path / "x.h5", np.zeros(3))
    with h5py.File(path, "a") as f:
        f.attrs["format_version"] = FORMAT_VERSION + 1
    with pytest.raises(ValueError, match=r"newer file format version 2.*Upgrade"):
        load(path)
    with h5py.File(tmp_path / "other.h5", "w") as f:
        f.create_dataset("x", data=np.zeros(3))
    with pytest.raises(ValueError, match="not a ChebyshevND table"):
        load(tmp_path / "other.h5")
    save(tmp_path / "k.h5", build())
    with h5py.File(tmp_path / "k.h5", "a") as f:
        f.attrs["kind"] = "Spline"
    with pytest.raises(ValueError, match="unknown kind"):
        load(tmp_path / "k.h5")
    with pytest.raises(ValueError, match="device"):
        build().to_device("tpu")


def test_missing_h5py_message(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(ImportError, match=r"chebyshevnd\[io\]"):
        save(tmp_path / "t.h5", build())


def test_cuda_without_cupy(tmp_path, monkeypatch):
    save(tmp_path / "t.h5", build())
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(ImportError, match=r"chebyshevnd\[gpu\]"):
        load(tmp_path / "t.h5", device="cuda")


# ------------------------------------------------------------------- GPU


@pytest.fixture
def cupy():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("no CUDA device")
    return cp


@pytest.mark.gpu
def test_gpu_load(cupy, tmp_path, mapped):
    interp = build(complex)
    save(tmp_path / "t.h5", interp)
    save(tmp_path / "m.h5", mapped)
    gpu = load(tmp_path / "t.h5", device="cuda")
    assert "cupy" in gpu._device_tables
    values = gpu.evaluate_batch(cupy.asarray(points()))
    assert array_namespace(values) is cupy
    np.testing.assert_allclose(to_numpy(values), interp.evaluate_batch(points()), rtol=1e-13, atol=1e-16)
    gpu_mapped = load(tmp_path / "m.h5", device="cuda")
    x = random_physical(20, seed=4)
    for c, g in zip(mapped.value_and_grad_batch(x), gpu_mapped.value_and_grad_batch(cupy.asarray(x))):
        np.testing.assert_allclose(to_numpy(g), c, rtol=1e-12, atol=1e-14)
