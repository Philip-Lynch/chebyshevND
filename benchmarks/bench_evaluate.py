"""
Per-call evaluation timings for the 1D-4D interpolants.

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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--json", help="also write the results to this JSON file")
    args = parser.parse_args()

    results = run()
    version = importlib.metadata.version("ChebyshevND")
    print(f"chebyshevnd {version} | numpy {np.__version__} | "
          f"scipy {scipy.__version__} | numba {numba.__version__} | "
          f"python {platform.python_version()} | {platform.machine()}")
    print(f"{'dim':>3}  {'grid':<18} {'evaluate':>10} {'optimized':>10} "
          f"{'precomputed':>12}  kmax after optimize({REL_TOL:g})")
    for r in results:
        pre = r.get("evaluate_precomputed_us")
        pre_s = f"{pre:10.2f}us" if pre is not None else f"{'-':>12}"
        print(f"{r['dim']:>3}  {str(tuple(r['grid'])):<18} {r['evaluate_us']:8.2f}us "
              f"{r['evaluate_optimized_us']:8.2f}us {pre_s}  {tuple(r['kmax_after_optimize'])}")

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
            json.dump({"meta": meta, "results": results}, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
