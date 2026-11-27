"""
Write the format version 1 fixture used by ``tests/test_io.py``.

The fixture was generated once, with chebyshevnd at commit 094ff24 (dev,
format version 1), before blocked tables and format version 2 existed. Its
reference values let later versions check that old files still load and
evaluate bit for bit. Do not regenerate it with a newer chebyshevnd: that
would only compare the new code with itself.

Usage: python tests/data/make_v1_fixture.py
"""

import os
import platform
import sys

import h5py
import numpy as np

import chebyshevnd as cnd

HERE = os.path.dirname(os.path.abspath(__file__))


def grid(shape, domains):
    nodes = [cnd.chebyshev_nodes(n, a, b) for n, (a, b) in zip(shape, domains)]
    return np.meshgrid(*nodes, indexing="ij")


def make_tables():
    # Real, three outputs, optimized: saved without the full tensor
    dom3 = [(0.0, 0.25), (-1.0, 2.0), (0.0, 0.8)]
    x, y, z = grid((9, 7, 6), dom3)
    vals = np.stack([np.exp(-x) * np.cos(y) + z**2, np.sin(x + y * z), 1e-3 * (1 + x * y * z)], axis=-1)
    plain = cnd.ChebyshevInterpolantND(vals, dom3, axis_names=["nu", "p", "e2"],
                                       output_names=["a", "b", "c"])
    plain.optimize(1e-6)

    # Complex, a single function, never optimized (full_is_coeffs)
    dom2 = [(-1.0, 1.0), (0.5, 3.0)]
    x, y = grid((8, 6), dom2)
    single = cnd.ChebyshevInterpolantND(np.exp(1j * x * y) / y, dom2, axis_names=["x", "y"])

    # Mapped, with a boundary table: u = 1/r, and q in [sep(r), 4] on a log scale
    rn = cnd.chebyshev_nodes(12, 6.0, 16.0)
    boundary = cnd.ChebyshevInterpolantND(np.stack([1.0 + 0.05 * rn], axis=-1), [(6.0, 16.0)],
                                          axis_names=["r"], output_names=["sep"])
    maps = {"r": cnd.AxisMap("power", power=-1),
            "q": cnd.AxisMap("log", lower="sep", upper=4.0, offset=0.1)}
    tdom = [maps["r"].table_domain(6.0, 16.0), (np.log(0.1), np.log(1.1))]
    ty = np.stack([m.ravel() for m in grid((10, 8), tdom)], axis=1)
    probe = cnd.MappedInterpolant(cnd.ChebyshevInterpolantND(np.zeros((10, 8)), tdom,
                                                             axis_names=["r", "q"]), maps, boundary)
    phys = probe.to_physical(ty)
    r, q = phys[:, 0].reshape(10, 8), phys[:, 1].reshape(10, 8)
    vals = np.stack([np.sin(2 / r) + 0.1 * q**2, r * q * 1e-2], axis=-1)
    table = cnd.ChebyshevInterpolantND(vals, tdom, axis_names=["r", "q"], output_names=["F", "G"])
    table.optimize(1e-8)
    mapped = cnd.MappedInterpolant(table, maps, boundary)
    return plain, single, mapped


def points(name):
    rng = np.random.default_rng(7)
    if name in ("plain", "plain32"):
        return np.stack([rng.uniform(0, 0.25, 5), rng.uniform(-1, 2, 5), rng.uniform(0, 0.8, 5)], axis=1)
    if name == "single":
        return np.stack([rng.uniform(-1, 1, 5), rng.uniform(0.5, 3, 5)], axis=1)
    r = rng.uniform(6, 16, 5)
    sep = 1.0 + 0.05 * r
    q = sep + (4.0 - sep) * rng.uniform(0.05, 0.95, 5)
    return np.stack([r, q], axis=1)


def reference(name, obj):
    table = obj.table if isinstance(obj, cnd.MappedInterpolant) else obj
    pts = points(name)
    ref = {"points": pts, "coeffs": table.get_coefficients(), "kmax": np.array(table.kmax),
           "abs_error": np.atleast_1d(table.absolute_error())}
    ref["evaluate"] = np.array([np.atleast_1d(obj.evaluate(*p)) for p in pts])
    grads = [obj.value_and_grad(*p) for p in pts]
    ref["vg_value"] = np.array([np.atleast_1d(v) for v, _ in grads])
    ref["vg_grad"] = np.array([g for _, g in grads])
    ref["batch"] = np.asarray(obj.evaluate_batch(pts))
    return {f"{name}/{k}": v for k, v in ref.items()}


def main():
    plain, single, mapped = make_tables()
    path = os.path.join(HERE, "v1_tables.h5")
    with h5py.File(path, "w") as f:
        cnd.save(f.create_group("plain"), plain, metadata={"note": "real, optimized"})
        cnd.save(f.create_group("plain32"), plain, float32=True)
        cnd.save(f.create_group("single"), single)
        cnd.save(f.create_group("mapped"), mapped, full=True)
    refs = {"platform": np.array(f"{sys.platform}-{platform.machine()}")}
    with h5py.File(path, "r") as f:
        for name in ("plain", "plain32", "single", "mapped"):
            # Reference values of the object as loaded, so float32 rounding is included
            refs.update(reference(name, cnd.load(f[name])))
    np.savez(os.path.join(HERE, "v1_reference.npz"), **refs)
    print(f"wrote {path} ({os.path.getsize(path)} bytes) and v1_reference.npz")


if __name__ == "__main__":
    main()
