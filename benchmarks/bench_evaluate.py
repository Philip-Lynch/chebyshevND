"""
Per-call evaluation timings for the 1D-4D interpolants and ChebyshevInterpolantND,
including precontraction, gradients and use inside an ODE right-hand side.

Run before and after a change to see its effect on evaluation cost:

    python benchmarks/bench_evaluate.py
    python benchmarks/bench_evaluate.py --json results.json
    python benchmarks/bench_evaluate.py --gpu   # adds a CuPy batch-size sweep

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
from numba import njit
from scipy.integrate import solve_ivp

from chebyshevnd import (
    ChebyshevInterpolant1D,
    ChebyshevInterpolant2D,
    ChebyshevInterpolant3D,
    ChebyshevInterpolant4D,
    ChebyshevInterpolantND,
    chebyshev_nodes,
    eval_grad_into,
    eval_into,
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


def _nd_case(dim):
    """The ChebyshevInterpolantND benchmark case of dimension `dim`: (interp, point, domains)."""
    f, domains, Ns = ND_5D if dim == 5 else (CASES[dim][1], CASES[dim][2], CASES[dim][3])
    axes = [chebyshev_nodes(N, a, b) for N, (a, b) in zip(Ns, domains)]
    interp = ChebyshevInterpolantND(f(*np.meshgrid(*axes, indexing="ij")), domains)
    point = [0.5 * (a + b) + 0.1 * (b - a) for a, b in domains]
    return interp, point, domains


@njit
def _loop_eval(points, coeffs, shape, bounds, tol, scratch, work, out):
    status = 0
    for i in range(points.shape[0]):
        status |= eval_into(coeffs, shape, bounds, tol, points[i], scratch, work, out)
    return status


@njit
def _loop_grad(points, coeffs, shape, bounds, tol, scratch, work, out):
    status = 0
    for i in range(points.shape[0]):
        status |= eval_grad_into(coeffs, shape, bounds, tol, points[i], scratch, work, out)
    return status


def run_gradient():
    """Value vs value + gradient: Python methods, raw kernels from Numba, and batches."""
    results = []
    for dim in range(1, 6):
        interp, point, domains = _nd_case(dim)
        lo, hi = np.array(domains).T
        batch = np.random.default_rng(0).uniform(lo, hi, size=(BATCH, dim))
        row = {"dim": dim, "kmax": list(interp.kmax)}
        row["evaluate_us"] = time_us(lambda: interp.evaluate(*point))
        row["value_and_grad_us"] = time_us(lambda: interp.value_and_grad(*point))

        table = interp.raw()
        ws = table.workspace()
        out = np.empty(table.n_outputs)
        out_grad = np.empty((dim + 1, table.n_outputs))
        row["eval_into_us"] = time_us(lambda: _loop_eval(batch, *table, *ws, out), number=20) / BATCH
        row["eval_grad_into_us"] = time_us(lambda: _loop_grad(batch, *table, *ws, out_grad), number=20) / BATCH
        row["batch_us_per_point"] = time_us(lambda: interp.evaluate_batch(batch), number=20) / BATCH
        row["batch_grad_us_per_point"] = time_us(lambda: interp.value_and_grad_batch(batch), number=20) / BATCH
        results.append(row)
    return results


# Toy averaged RHS: (nu, u, e) table with outputs (du/dchi, de/dchi), nu fixed per waveform
ODE_DOMAINS = [(0.0, 0.25), (-3.0, -1.0), (0.0, 0.9)]
ODE_NODES = (10, 30, 20)


def _ode_rhs_values(nu, u, e):
    du = 0.02 * (1 + nu) * (1 + 0.3 * e**2 * np.cos(u))
    de = -0.01 * e * (1 + 0.2 * np.sin(u) + nu)
    return np.stack([du, de], axis=-1)


@njit
def _rhs_trivial(y):
    return np.array([0.02 * (1 + 0.3 * y[1] ** 2), -0.01 * y[1]])


@njit
def _rhs_table(y, coeffs, shape, bounds, tol, scratch, work, out):
    eval_into(coeffs, shape, bounds, tol, y, scratch, work, out)
    return out.copy()


def run_ode():
    """A toy 2-variable averaged RHS integrated with scipy's DOP853."""
    axes = [chebyshev_nodes(N, a, b) for N, (a, b) in zip(ODE_NODES, ODE_DOMAINS)]
    vals = _ode_rhs_values(*np.meshgrid(*axes, indexing="ij"))
    interp = ChebyshevInterpolantND(vals, ODE_DOMAINS, axis_names=["nu", "u", "e"],
                                    output_names=["du", "de"])
    child = interp.precontract({"nu": 0.2})
    table = child.raw()
    ws = table.workspace()
    out = np.empty(2)
    y0 = np.array([-2.9, 0.5])
    span = (0.0, 80.0)

    rhs = {
        "trivial jitted RHS (integrator floor)": lambda t, y: _rhs_trivial(y),
        "Python RHS calling evaluate()": lambda t, y: child.evaluate(*y),
        "jitted RHS calling eval_into()": lambda t, y: _rhs_table(y, *table, *ws, out),
    }
    results = []
    for name, fn in rhs.items():
        sol = solve_ivp(fn, span, y0, method="DOP853", rtol=1e-10, atol=1e-12)
        total = time_us(lambda: solve_ivp(fn, span, y0, method="DOP853", rtol=1e-10, atol=1e-12),
                        number=5, repeat=5)
        results.append({"rhs": name, "nfev": int(sol.nfev), "total_ms": total / 1e3,
                        "us_per_call": total / sol.nfev})
    floor = results[0]["us_per_call"]
    for r in results:
        r["evaluation_share"] = max(0.0, 1.0 - floor / r["us_per_call"])
    return results


SWEEP_SIZES = (1, 10, 100, 1000, 10000)


def run_batch_sweep(xp):
    """
    Per-member cost of a batched value + gradient on per-member tables, vs batch size.

    Compares ``ChebyshevBatch.value_and_grad`` on module `xp` (numpy or cupy)
    with evaluating the members one by one with ``eval_grad_into`` in a Numba
    loop on the CPU (one precontracted table, the single-waveform path).
    """
    axis_names, domains, Ns = ("nu", "u", "e"), [(0.0, 0.25), (-3.0, -1.0), (0.0, 0.9)], (10, 30, 20)
    axes = [chebyshev_nodes(N, a, b) for N, (a, b) in zip(Ns, domains)]
    grids = np.meshgrid(*axes, indexing="ij")
    vals = np.stack([_nit_like(*grids, k=k) for k in range(4)], axis=-1)
    interp = ChebyshevInterpolantND(vals, domains, axis_names=axis_names, output_names=list("abcd"))
    sync = xp.cuda.Device().synchronize if xp.__name__ == "cupy" else (lambda: None)

    table = interp.precontract({"nu": 0.1}).raw()
    ws = table.workspace()
    out = np.empty((3, 4))
    results = []
    for B in SWEEP_SIZES:
        lo, hi = np.array(domains).T
        points = np.random.default_rng(1).uniform(lo, hi, size=(B, 3))
        batch = interp.precontract_batch({"nu": xp.asarray(points[:, 0])})
        live = xp.asarray(points[:, 1:])

        def batched():
            batch.value_and_grad(live)
            sync()

        number = max(5, 20000 // B)
        row = {"B": B, "backend": xp.__name__}
        row["batch_us_per_call"] = time_us(batched, number=number)
        row["batch_us_per_member"] = row["batch_us_per_call"] / B
        cpu_points = np.ascontiguousarray(points[:, 1:])
        row["numba_loop_us_per_member"] = time_us(lambda: _loop_grad(cpu_points, *table, *ws, out),
                                                  number=number) / B
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--json", help="also write the results to this JSON file")
    parser.add_argument("--gpu", action="store_true", help="also run the batch-size sweep with CuPy")
    args = parser.parse_args()

    results = run()
    results_nd = run_nd()
    results_pc = run_precontract()
    results_grad = run_gradient()
    results_ode = run_ode()
    results_sweep = run_batch_sweep(np)
    if args.gpu:
        import cupy

        results_sweep += run_batch_sweep(cupy)
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

    print(f"\nValue vs value + gradient (single function; raw and batch columns: per point, B = {BATCH})")
    print(f"{'dim':>3}  {'kmax':<18} {'evaluate':>10} {'val+grad':>10} {'ratio':>6} "
          f"{'eval_into':>10} {'eval_grad':>10} {'ratio':>6} {'batch':>9} {'batch grad':>11}")
    for r in results_grad:
        print(f"{r['dim']:>3}  {str(tuple(r['kmax'])):<18} {r['evaluate_us']:8.2f}us "
              f"{r['value_and_grad_us']:8.2f}us {r['value_and_grad_us'] / r['evaluate_us']:6.2f} "
              f"{r['eval_into_us']:8.2f}us {r['eval_grad_into_us']:8.2f}us "
              f"{r['eval_grad_into_us'] / r['eval_into_us']:6.2f} {r['batch_us_per_point']:7.2f}us "
              f"{r['batch_grad_us_per_point']:9.2f}us")

    print(f"\nODE right-hand side: toy 2-variable averaged RHS, (nu, u, e) = {ODE_NODES} table "
          f"precontracted at nu, scipy DOP853")
    print(f"{'right-hand side':<40} {'calls':>6} {'total':>10} {'per call':>10} {'evaluation share':>17}")
    for r in results_ode:
        print(f"{r['rhs']:<40} {r['nfev']:>6} {r['total_ms']:8.2f}ms {r['us_per_call']:8.2f}us "
              f"{100 * r['evaluation_share']:16.0f}%")

    print("\nBatch-size sweep: value + gradient on per-member (u, e) tables, 4 outputs, per member")
    print(f"{'backend':<8} {'B':>6} {'batch call':>12} {'batch/member':>13} {'numba loop/member':>18}")
    for r in results_sweep:
        print(f"{r['backend']:<8} {r['B']:>6} {r['batch_us_per_call']:10.2f}us "
              f"{r['batch_us_per_member']:11.3f}us {r['numba_loop_us_per_member']:16.3f}us")

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
                       "results_precontract": results_pc, "results_gradient": results_grad,
                       "results_ode": results_ode, "results_batch_sweep": results_sweep}, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
