"""
Per-output truncation on multi-output tables: stored size, precontraction and evaluation times, accuracy.

Made for the NIT tables of pyseobnr_nits (multi-output MappedInterpolants
over (nu, p, e2) whose outputs are the Fourier harmonics of a few
families), but runs on any file of saved tables with a ``nu`` axis:

    python benchmarks/bench_blocked.py tables/nit_pmin81_e09.h5
    python benchmarks/bench_blocked.py tables.h5 --groups fourier0 inv --json out.json

For each table group, it optimizes a fresh copy with ``norm="l2"``,
``per_output=True`` and ``groups`` from the output names (the harmonic
index and any ``_re``/``_im`` removed), for each ``rel_tol`` and
``max_blocks``, after a dense row for the table as saved. Each row reports:

- the stored coefficients, in MB, and the fraction of the dense tensor;
- the time of ``optimize`` itself;
- the time of ``precontract({"nu": 0.1})``, the per-waveform floor: warm
  (the best of many calls in a row, the table partly in cache) and cold
  (the median over calls made after flushing the caches, as for a
  precontraction once per waveform between other work);
- on the precontracted table, ``evaluate_batch`` at 200 points and one
  ``evaluate`` (best of many calls);
- the largest error against the table as saved, at 1000 random points,
  relative to each output's group scale (its family's largest coefficient).

The file is only read. Timings depend on the machine and on its load;
compare rows from one run.
"""

import argparse
import json
import re
import timeit

import h5py
import numpy as np

from chebyshevnd import MappedInterpolant, load

LABEL = re.compile(r"(_re|_im)?_-?\d+$")


#: Read between cold calls, to push the tables out of the caches (256 MB)
_FLUSH = np.ones(32 * 1024 * 1024)


def best_ms(fn, calls=100):
    """The best time of single calls, in ms (robust to the machine's other load)."""
    return min(timeit.repeat(fn, number=1, repeat=calls)) * 1e3


def cold_ms(fn, calls=25):
    """The median time of single calls made with cold caches, in ms."""
    times = []
    for _ in range(calls):
        _FLUSH.sum()  # reading leaves clean lines, so evicting them costs nothing later
        start = timeit.default_timer()
        fn()
        times.append(timeit.default_timer() - start)
    return float(np.median(times)) * 1e3


def table_of(obj):
    return obj.table if isinstance(obj, MappedInterpolant) else obj


def random_physical(obj, n, seed):
    """Random points in physical coordinates, uniform in table coordinates."""
    table = table_of(obj)
    lo, hi = np.array(table.domains).T
    y = np.random.default_rng(seed).uniform(lo, hi, size=(n, table.ndim))
    return obj.to_physical(y) if isinstance(obj, MappedInterpolant) else y


def stored_bytes(obj):
    return sum(c.nbytes for _, c in table_of(obj).blocks)


def measure(obj, reference, labels, scale, nu):
    """The timings and the error of one (optimized or dense) object."""
    table = table_of(obj)
    nu_axis = table.axis_names.index("nu")
    child = obj.precontract({"nu": nu})
    row = {"precontract_ms": best_ms(lambda: obj.precontract({"nu": nu})),
           "precontract_cold_ms": cold_ms(lambda: obj.precontract({"nu": nu}))}

    # Points of the precontracted table: random in the remaining axes (nu
    # is not mapped in the NIT tables, so its column does not move the others)
    live = [i for i in range(table.ndim) if i != nu_axis]
    child_pts = random_physical(obj, 200, seed=1)[:, live]
    child.evaluate_batch(child_pts)
    child.evaluate(*child_pts[0])
    row["evaluate_batch_200_ms"] = best_ms(lambda: child.evaluate_batch(child_pts))
    row["evaluate_us"] = best_ms(lambda: child.evaluate(*child_pts[0]), calls=2000) * 1e3

    x = random_physical(obj, 1000, seed=2)
    err = np.abs(obj.evaluate_batch(x) - reference.evaluate_batch(x)).max(axis=0)
    row["max_error_rel_group"] = float((err / np.where(scale > 0, scale, 1.0)).max())
    worst = int(np.argmax(err / np.where(scale > 0, scale, 1.0)))
    row["worst_output"] = table.output_names[worst] if table.output_names else 0
    return row


def run_group(group, rel_tols, max_blocks_list, nu):
    reference = load(group)
    table = table_of(reference)
    labels = [LABEL.sub("", name) for name in table.output_names]
    full = np.abs(table.get_coefficients(full=True)).reshape(-1, table.n_outputs).max(axis=0)
    group_max = {}
    for label, s in zip(labels, full):
        group_max[label] = max(group_max.get(label, 0.0), float(s))
    scale = np.array([group_max[label] for label in labels])
    dense_bytes = int(np.prod(table.grid_shape)) * table.n_outputs * table.dtype.itemsize

    families = {}
    for label in labels:
        families[label] = families.get(label, 0) + 1
    print(f"\n[{group.name.strip('/')}] {type(reference).__name__} over {table.axis_names}, "
          f"grid {table.grid_shape}, {table.n_outputs} outputs, {dense_bytes / 1e6:.1f} MB dense; "
          f"families {families}")
    header = (f"{'rel_tol':>8} {'blocks':>7} {'MB':>7} {'stored':>7} {'empty':>6} {'optimize':>9} "
              f"{'precontr.':>10} {'(cold)':>8} {'batch200':>9} {'evaluate':>9} {'max err':>9}")
    print(header)
    print(f"{'':>8} {'':>7} {'':>7} {'':>7} {'':>6} {'s':>9} {'ms':>10} {'ms':>8} {'ms':>9} {'us':>9} "
          f"{'/ S_g':>9}")

    rows = []
    settings = [(None, None)] + [(t, b) for t in rel_tols for b in max_blocks_list]
    for rel_tol, max_blocks in settings:
        obj = load(group)
        seconds = 0.0
        if rel_tol is not None:
            start = timeit.default_timer()
            obj.optimize(rel_tol, keep_full=False, groups=labels, norm="l2", per_output=True,
                         max_blocks=max_blocks)
            seconds = timeit.default_timer() - start
        t = table_of(obj)
        row = {"group": group.name.strip("/"), "rel_tol": rel_tol, "max_blocks": max_blocks,
               "n_blocks": len(t.blocks), "stored_mb": stored_bytes(obj) / 1e6,
               "stored_fraction": stored_bytes(obj) / dense_bytes,
               "empty_fraction": float(np.mean(t.output_kmax[:, 0] == 0)), "optimize_s": seconds}
        row.update(measure(obj, reference, labels, scale, nu))
        rows.append(row)
        blocks = "dense" if rel_tol is None else str(row["n_blocks"])
        tol = "-" if rel_tol is None else f"{rel_tol:g}"
        print(f"{tol:>8} {blocks:>7} {row['stored_mb']:7.1f} {row['stored_fraction']:7.1%} "
              f"{row['empty_fraction']:6.0%} {seconds:9.2f} {row['precontract_ms']:10.3f} "
              f"{row['precontract_cold_ms']:8.3f} "
              f"{row['evaluate_batch_200_ms']:9.3f} {row['evaluate_us']:9.1f} "
              f"{row['max_error_rel_group']:9.1e}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("path", help="HDF5 file of saved tables (opened read-only)")
    parser.add_argument("--groups", nargs="+", default=["fourier0", "inv"],
                        help="the table groups in the file to benchmark")
    parser.add_argument("--rel-tol", nargs="+", type=float, default=[1e-9, 1e-10])
    parser.add_argument("--max-blocks", nargs="+", default=["4", "8", "16", "None"],
                        help="max_blocks values; 'None' for one block per output")
    parser.add_argument("--nu", type=float, default=0.1, help="the value of nu to precontract at")
    parser.add_argument("--json", help="also write the results to this JSON file")
    args = parser.parse_args()
    max_blocks = [None if b == "None" else int(b) for b in args.max_blocks]

    rows = []
    with h5py.File(args.path, "r") as f:
        for name in args.groups:
            rows += run_group(f[name], args.rel_tol, max_blocks, args.nu)
    if args.json:
        with open(args.json, "w") as out:
            json.dump(rows, out, indent=2)


if __name__ == "__main__":
    main()
