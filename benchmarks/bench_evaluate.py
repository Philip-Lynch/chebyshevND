"""
Per-call evaluation timings for the 1D-4D interpolants and ChebyshevInterpolantND,
including precontraction.

Run before and after a change to see its effect on evaluation cost:

    python benchmarks/bench_evaluate.py
    python benchmarks/bench_evaluate.py --json results.json

Timings are the best of several repeats, in microseconds per call, and
include the Python-level call overhead a user sees. They depend on the
machine, so compare runs from the same machine only.
"""

import argparse
import datetime
import importlib.metadata
import json
import platform
import timeit

import numba
import numpy as np
import scipy

from chebyshevnd import (
    ChebyshevInterpolant1D,
    ChebyshevInterpolant2D,
    ChebyshevInterpolant3D,
    ChebyshevInterpolant4D,
    ChebyshevInterpolantND,
    chebyshev_nodes,
)

# (class, test function, per-axis domains, per-axis node counts). The 4D grid
# matches the baseline in docs/DESIGN_NIT_v6EPHM.md.
CASES = {
    1: (ChebyshevInterpolant1D, lambda x: np.exp(x) * np.sin(2 * x),
        [(0.0, 2.0)], (64,)),
    2: (ChebyshevInterpolant2D, lambda x, y: np.sin(x) * np.cos(y),
        [(-1.0, 2.0), (0.0, 3.0)], (32, 24)),
    3: (ChebyshevInterpolant3D, lambda x, y, z: np.sin(x) * np.cos(y) * np.exp(-z),
        [(0.0, 2.0), (-1.0, 3.0), (1.0, 4.0)], (20, 16, 12)),
    4: (ChebyshevInterpolant4D,
        lambda x, y, z, w: np.sin(x) * np.cos(y) * np.exp(-z) * (w**3 + w),
        [(0.0, 2.0), (-1.0, 3.0), (1.0, 4.0), (-2.0, 0.0)], (12, 12, 12, 12)),
}
REL_TOL = 1e-8

# ChebyshevInterpolantND runs the cases above plus this 5D one
ND_5D = (lambda a, b, c, d, e: np.exp(0.3 * a) * np.cos(0.5 * b) * (1 + 0.2 * c) ** 2
         * np.sin(0.4 * d + 0.1) * np.cos(e),
         [(0.0, 1.0), (-1.0, 2.0), (0.5, 1.5), (-2.0, -1.0), (0.0, 0.5)], (12, 10, 8, 8, 6))
BATCH = 1000

# Precontraction cases shaped like the NIT tables, with 4 outputs each:
# (name, axis names, domains, node counts, axes fixed per waveform)
PRECONTRACT_CASES = [
    ("nonspinning", ("nu", "u", "e"), [(0.0, 0.25), (-3.0, -1.0), (0.0, 0.9)],
     (10, 30, 20), ("nu",)),
    ("aligned-spin", ("nu", "u", "e", "chi1", "chi2"),
     [(0.0, 0.25), (-3.0, -1.0), (0.0, 0.9), (-1.0, 1.0), (-1.0, 1.0)],
     (10, 24, 16, 6, 6), ("nu", "chi1", "chi2")),
]


def _nit_like(*xs, k=0):
    """A smooth stand-in for an NIT forcing function; k varies the output."""
    nu, u, e = xs[:3]
    out = np.exp(-nu * (k + 1)) * np.cos(u + k) * (1 + e**2) ** (k + 1)
    for j, chi in enumerate(xs[3:]):
        out = out * (1 + 0.3 * (j + 1) * chi)
    return out


def time_us(fn, number=20000, repeat=5):
    """Best-of-`repeat` time per call in microseconds (after one warm-up call)."""
    fn()
    return min(timeit.repeat(fn, number=number, repeat=repeat)) / number * 1e6


def build(dim):
    cls, f, domains, Ns = CASES[dim]
    axes = [chebyshev_nodes(N, a, b) for N, (a, b) in zip(Ns, domains)]
    vals = f(*np.meshgrid(*axes, indexing="ij"))
    bounds = [v for dom in domains for v in dom]
    point = [0.5 * (a + b) + 0.1 * (b - a) for a, b in domains]
    return cls(vals, *bounds), point


def run():
    results = []
    for dim in sorted(CASES):
        interp, point = build(dim)
        row = {"dim": dim, "grid": list(CASES[dim][3])}
        row["evaluate_us"] = time_us(lambda: interp.evaluate(*point))
        interp.optimize(REL_TOL)
        row["kmax_after_optimize"] = [int(k) for k in np.atleast_1d(
            interp.k_max if dim == 1 else interp.effective_grid())]
        row["evaluate_optimized_us"] = time_us(lambda: interp.evaluate(*point))
        if dim in (3, 4):
            interp.precompute_x(point[0])
            rest = point[1:]
            fn = interp.evaluate_yz if dim == 3 else interp.evaluate_yzw
            row["evaluate_precomputed_us"] = time_us(lambda: fn(*rest))
        results.append(row)
    return results


def run_nd():
    cases = {dim: (f, domains, Ns) for dim, (_, f, domains, Ns) in CASES.items()}
    cases[5] = ND_5D
    results = []
    for dim, (f, domains, Ns) in sorted(cases.items()):
        axes = [chebyshev_nodes(N, a, b) for N, (a, b) in zip(Ns, domains)]
        interp = ChebyshevInterpolantND(f(*np.meshgrid(*axes, indexing="ij")), domains)
        point = [0.5 * (a + b) + 0.1 * (b - a) for a, b in domains]
        lo, hi = np.array(domains).T
        batch = np.random.default_rng(0).uniform(lo, hi, size=(BATCH, dim))

        row = {"dim": dim, "grid": list(Ns)}
        row["evaluate_us"] = time_us(lambda: interp.evaluate(*point))
        row["batch_us_per_point"] = time_us(lambda: interp.evaluate_batch(batch), number=20) / BATCH
        interp.optimize(REL_TOL)
        row["kmax_after_optimize"] = list(interp.kmax)
        row["evaluate_optimized_us"] = time_us(lambda: interp.evaluate(*point))
        row["batch_optimized_us_per_point"] = time_us(lambda: interp.evaluate_batch(batch), number=20) / BATCH
        results.append(row)
    return results


def run_precontract():
    results = []
    for name, axis_names, domains, Ns, fixed in PRECONTRACT_CASES:
        axes = [chebyshev_nodes(N, a, b) for N, (a, b) in zip(Ns, domains)]
        grids = np.meshgrid(*axes, indexing="ij")
        vals = np.stack([_nit_like(*grids, k=k) for k in range(4)], axis=-1)
        interp = ChebyshevInterpolantND(vals, domains, axis_names=axis_names, output_names=list("abcd"))
        lo, hi = np.array(domains).T
        points = np.random.default_rng(0).uniform(lo, hi, size=(BATCH, len(domains)))
        fixed_idx = [axis_names.index(a) for a in fixed]
        live_idx = [i for i in range(len(domains)) if i not in fixed_idx]
        one = {a: points[0, i] for a, i in zip(fixed, fixed_idx)}
        many = {a: points[:, i] for a, i in zip(fixed, fixed_idx)}

        child = interp.precontract(one)
        batch = interp.precontract_batch(many)
        row = {"case": name, "grid": list(Ns), "fixed": list(fixed)}
        row["evaluate_full_us"] = time_us(lambda: interp.evaluate(*points[0]), number=2000)
        row["precontract_us"] = time_us(lambda: interp.precontract(one), number=200)
        row["evaluate_precontracted_us"] = time_us(lambda: child.evaluate(*points[0, live_idx]))
        row["precontract_batch_us_per_member"] = time_us(lambda: interp.precontract_batch(many), number=10) / BATCH
        row["batch_evaluate_us_per_member"] = time_us(lambda: batch.evaluate(points[:, live_idx]), number=20) / BATCH
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--json", help="also write the results to this JSON file")
    args = parser.parse_args()

    results = run()
    results_nd = run_nd()
    results_pc = run_precontract()
    version = importlib.metadata.version("ChebyshevND")
    print(f"chebyshevnd {version} | numpy {np.__version__} | "
          f"scipy {scipy.__version__} | numba {numba.__version__} | "
          f"python {platform.python_version()} | {platform.machine()}")
    print("\nv0.1 classes (ChebyshevInterpolant1D ... 4D)")
    print(f"{'dim':>3}  {'grid':<18} {'evaluate':>10} {'optimized':>10} "
          f"{'precomputed':>12}  kmax after optimize({REL_TOL:g})")
    for r in results:
        pre = r.get("evaluate_precomputed_us")
        pre_s = f"{pre:10.2f}us" if pre is not None else f"{'-':>12}"
        print(f"{r['dim']:>3}  {str(tuple(r['grid'])):<18} {r['evaluate_us']:8.2f}us "
              f"{r['evaluate_optimized_us']:8.2f}us {pre_s}  {tuple(r['kmax_after_optimize'])}")

    print(f"\nChebyshevInterpolantND (batch columns: per point, B = {BATCH})")
    print(f"{'dim':>3}  {'grid':<18} {'evaluate':>10} {'batch':>10} {'optimized':>10} "
          f"{'opt. batch':>11}  kmax after optimize({REL_TOL:g})")
    for r in results_nd:
        print(f"{r['dim']:>3}  {str(tuple(r['grid'])):<18} {r['evaluate_us']:8.2f}us "
              f"{r['batch_us_per_point']:8.2f}us {r['evaluate_optimized_us']:8.2f}us "
              f"{r['batch_optimized_us_per_point']:9.2f}us  {tuple(r['kmax_after_optimize'])}")

    print(f"\nPrecontraction (4 outputs; batch columns: per member, B = {BATCH})")
    print(f"{'case':<13} {'grid':<20} {'fixed':<17} {'full eval':>10} {'precontract':>12} "
          f"{'then eval':>10} {'batch build':>12} {'batch eval':>11}")
    for r in results_pc:
        print(f"{r['case']:<13} {str(tuple(r['grid'])):<20} {','.join(r['fixed']):<17} "
              f"{r['evaluate_full_us']:8.2f}us {r['precontract_us']:10.2f}us "
              f"{r['evaluate_precontracted_us']:8.2f}us {r['precontract_batch_us_per_member']:10.2f}us "
              f"{r['batch_evaluate_us_per_member']:9.2f}us")

    if args.json:
        meta = {
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "chebyshevnd": version,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "numba": numba.__version__,
        }
        with open(args.json, "w") as fh:
            json.dump({"meta": meta, "results": results, "results_nd": results_nd,
                       "results_precontract": results_pc}, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
