# ChebyshevND

[![Tests](https://github.com/Philip-Lynch/chebyshevND/actions/workflows/tests.yml/badge.svg)](https://github.com/Philip-Lynch/chebyshevND/actions/workflows/tests.yml)
[![Documentation](https://readthedocs.org/projects/chebyshevnd/badge/?version=latest)](https://chebyshevnd.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/chebyshevnd)](https://pypi.org/project/ChebyshevND/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21825112.svg)](https://doi.org/10.5281/zenodo.21825112)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fast multidimensional Chebyshev interpolation in Python, using DCTs to compute
coefficients, Numba-accelerated Clenshaw recursions for evaluation, and
structured coefficient pruning for compression.

For smooth functions, Chebyshev interpolation converges *spectrally*: the
error decays exponentially with the number of nodes per axis. ChebyshevND
turns a tensor grid of function samples into an interpolant that can be
evaluated anywhere in the domain at close to machine precision — with built-in
error estimates that come free from the coefficient decay.

**Features**

- Interpolant classes for 1D–4D (`ChebyshevInterpolant1D` … `ChebyshevInterpolant4D`)
- Coefficients computed with a single fast DCT (`scipy.fft`), on either
  Chebyshev root nodes or Gauss–Lobatto (endpoint-including) nodes
- Numba-jitted Clenshaw evaluation kernels, also usable standalone
  (`clenshaw_1d` … `clenshaw_4d`, `clenshaw_nd`)
- A-posteriori error estimates from trailing coefficient magnitudes
- Structured pruning (`optimize`) that trims insignificant trailing
  coefficient slices under a global error budget
- Dimension precomputation (`precompute_x`) for cheap repeated evaluation
  when one variable is held fixed

## Installation

```bash
pip install chebyshevnd
```

Requires Python ≥ 3.10, with `numpy`, `scipy`, and `numba` installed
automatically. To install the development version from source:

```bash
git clone https://github.com/Philip-Lynch/chebyshevND.git
cd chebyshevND
pip install -e .
```

## Quickstart

Sample your function at Chebyshev nodes, build the interpolant from the
values, evaluate anywhere:

```python
import numpy as np
from chebyshevnd import ChebyshevInterpolant1D, chebyshev_nodes

f = lambda x: 1.0 / (1.0 + 25 * x**2)   # Runge function
a, b = -1.0, 1.0

x_nodes = chebyshev_nodes(100, a, b)
interp = ChebyshevInterpolant1D(f(x_nodes), a, b)

interp.evaluate(0.3)                     # a float
interp.evaluate(np.linspace(a, b, 50))   # an array
interp.relative_error()                  # a-posteriori error estimate
```

In higher dimensions, sample on the tensor grid and pass the value array with
one (lower, upper) pair of bounds per axis, in the order of the array's axes
(see [Axis order and domain bounds](#axis-order-and-domain-bounds)):

```python
from chebyshevnd import ChebyshevInterpolant3D, chebyshev_nodes

f = lambda x, y, z: np.sin(x) * np.cos(y) * np.exp(-z)

x = chebyshev_nodes(24, 0.0, 2.0)    # a different number of nodes per axis
y = chebyshev_nodes(16, -1.0, 3.0)   # makes ordering mistakes easy to spot
z = chebyshev_nodes(12, 1.0, 4.0)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")   # "ij", not the default "xy"
vals = f(X, Y, Z)                    # shape (24, 16, 12): axes are (x, y, z)

interp = ChebyshevInterpolant3D(vals,
                                ax=0.0, bx=2.0,    # bounds of axis 0 (x)
                                ay=-1.0, by=3.0,   # bounds of axis 1 (y)
                                az=1.0, bz=4.0)    # bounds of axis 2 (z)
interp.evaluate(1.3, 0.5, 2.5)       # coordinates in the same order: x, y, z

interp.optimize(rel_tol=1e-8)   # prune insignificant coefficients
interp.effective_grid()         # (10, 13, 11) — down from (24, 16, 12)

interp.precompute_x(1.3)        # collapse x once ...
interp.evaluate_yz(0.5, 2.5)    # ... then evaluate repeatedly in (y, z), ~3x faster
```

The [example notebooks](examples/) walk through each dimension, including
convergence, coefficient decay, pruning, and precomputation timings.

## Axis order and domain bounds

Most wrong results come from mismatched orderings. On a square grid (the same
number of nodes and the same domain on every axis) these mistakes raise no
error: you get an interpolant, just of the wrong function. Keep these four
orderings consistent:

1. **Array axes ↔ variables.** Axis *k* of `vals` holds variable *k*. Build
   the grid with `np.meshgrid(..., indexing="ij")`, so that
   `vals[i, j, k] == f(x[i], y[j], z[k])`. NumPy's default `indexing="xy"`
   swaps the first two axes.
2. **One (lower, upper) pair per axis, in axis order.** The constructors take
   `ax, bx` for axis 0, then `ay, by` for axis 1, and so on. Passing all the
   lower bounds first (`ax, ay, bx, by`) builds the interpolant on the wrong
   domain. Passing the bounds by keyword (`ax=..., bx=...`) avoids this. The
   standalone kernel `clenshaw_nd` is the exception: it takes an array `a` of
   lower bounds and an array `b` of upper bounds.
3. **The same bounds for nodes and interpolant.** Generate each axis's nodes
   with the same `(a, b)` pair, in the same order, that you pass to the
   constructor for that axis.
4. **Nodes in the order returned.** `chebyshev_nodes` and
   `chebyshev_gauss_lobatto_nodes` return the nodes in descending order (from
   `b` to `a`), and the coefficient computation relies on it. Sample at the
   nodes as returned; don't sort them.

`evaluate` takes its coordinates in the same axis order. It does not check
that they lie inside the domain: outside `[a, b]` the series extrapolates and
quickly diverges.

To catch mistakes early, use a different number of nodes on each axis, check
that `vals.shape` is `(Nx, Ny, ...)`, and compare the interpolant with your
function at a few random points. The
[2D example notebook](examples/example_2d_interp.ipynb) shows what each
mistake does.

## How it works

1. **Sampling.** The function is sampled on a tensor grid of Chebyshev nodes —
   either the roots of $T_N$ (`chebyshev_nodes`) or the Gauss–Lobatto extrema
   including the endpoints (`chebyshev_gauss_lobatto_nodes`).
2. **Coefficients.** The Chebyshev expansion coefficients are obtained with a
   single n-dimensional DCT (type II for root nodes, type I for Gauss–Lobatto),
   costing $O(M \log M)$ for $M$ grid points.
3. **Evaluation.** The tensor series is evaluated with nested Clenshaw
   recursions, compiled with Numba. Evaluation never reconstructs polynomial
   values explicitly, which keeps it numerically stable.
4. **Compression.** For smooth functions the coefficients decay geometrically
   along every axis, so most of the tensor is numerically irrelevant.
   `optimize` greedily discards trailing slices while keeping the summed
   discarded magnitude — an upper bound on the introduced uniform error —
   under a relative tolerance you choose. Trailing coefficient magnitudes also
   provide error estimates (`absolute_error`, `relative_error`) without any
   extra function samples.

## Roadmap

Planned for future releases:

- Interpolants beyond 4D (5D/6D) via a generic n-dimensional kernel
- Analytic partial derivatives of interpolants, evaluated efficiently through
  the relationship between Chebyshev polynomials of the first and second kind
- Precomputation over multiple leading dimensions for even cheaper repeated
  evaluation

## Development

Run the test suite with:

```bash
pip install -e .
pip install pytest
pytest
```

To compare per-call evaluation timings before and after a change (on the same
machine):

```bash
python benchmarks/bench_evaluate.py              # prints a table
python benchmarks/bench_evaluate.py --json out.json
```

## Citation

The techniques implemented in this package were derived and developed for:

> H. Khalvati, P. Lynch, O. Burke, L. Speri, M. van de Meent, and Z. Nasipak,
> *Systematic errors in fast relativistic waveforms for Extreme Mass Ratio
> Inspirals*, [Phys. Rev. D **113**, 084042 (2026)](https://doi.org/10.1103/4ly7-zn15),
> [arXiv:2509.08875](https://arxiv.org/abs/2509.08875).

If this package contributes to your research, please cite that paper — machine-readable
citation metadata is in [CITATION.cff](CITATION.cff).

```bibtex
@article{Khalvati2026systematic,
  author        = {Khalvati, Hassan and Lynch, Philip and Burke, Ollie and
                   Speri, Lorenzo and van de Meent, Maarten and Nasipak, Zachary},
  title         = {Systematic errors in fast relativistic waveforms for
                   Extreme Mass Ratio Inspirals},
  journal       = {Phys. Rev. D},
  volume        = {113},
  pages         = {084042},
  year          = {2026},
  doi           = {10.1103/4ly7-zn15},
  eprint        = {2509.08875},
  archivePrefix = {arXiv},
  primaryClass  = {gr-qc}
}
```

## License

MIT — see [LICENSE](LICENSE).
