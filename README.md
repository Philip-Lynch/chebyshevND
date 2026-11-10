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

- `ChebyshevInterpolantND`: any number of dimensions, several real or complex
  functions on one grid, named axes and outputs, and a domain check
- Simple fixed-dimension classes for 1D–4D (`ChebyshevInterpolant1D` …
  `ChebyshevInterpolant4D`)
- Coefficients computed with a single fast DCT (`scipy.fft`), on either
  Chebyshev root nodes or Gauss–Lobatto (endpoint-including) nodes
- A-posteriori error estimates from trailing coefficient magnitudes, per
  function and per axis
- Structured pruning (`optimize`) that trims insignificant trailing
  coefficient slices under a global error budget
- **Precontraction**: fix some variables (e.g. the parameters of one
  waveform) once, for much cheaper evaluations over the rest, also for a whole
  batch of parameter values at once
- **Derivatives**: exact partial derivatives as new interpolants, and the
  value together with the full gradient at ~1.0–1.4× the cost of the value
- **Compiled and GPU evaluation**: entry points callable from your own Numba
  code (e.g. an ODE right-hand side), and batched evaluation on NumPy or, with
  CuPy, on the GPU
- **Domain maps** for regions that are not boxes (e.g. `u = 1/r`, or a
  variable measured from a boundary that depends on other variables), with
  derivatives in the physical variables
- **Grids** for sampling expensive functions in parallel chunks, and **HDF5
  files** with a documented layout to keep and share tables

## Installation

```bash
pip install chebyshevnd
```

Requires Python ≥ 3.10, with `numpy`, `scipy`, and `numba` installed
automatically. Optional extras:

```bash
pip install "chebyshevnd[io]"    # saving and loading tables (h5py)
pip install "chebyshevnd[gpu]"   # batched evaluation on NVIDIA GPUs (CuPy, CUDA 12)
```

To install the development version from source:

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

In any number of dimensions, and for several functions at once, use
`ChebyshevInterpolantND`. It takes one `(lower, upper)` pair per axis, and
names for the axes and functions:

```python
from chebyshevnd import ChebyshevInterpolantND

nu = chebyshev_nodes(10, 0.0, 0.25)
u = chebyshev_nodes(24, -3.0, -1.0)
e = chebyshev_nodes(14, 0.0, 0.6)
NU, U, E = np.meshgrid(nu, u, e, indexing="ij")
vals = np.stack([np.exp(-NU) * np.cos(U), E * np.sin(U)], axis=-1)   # two functions

table = ChebyshevInterpolantND(vals, [(0.0, 0.25), (-3.0, -1.0), (0.0, 0.6)],
                               axis_names=["nu", "u", "e"], output_names=["f", "g"])
table.evaluate(0.1, -2.0, 0.3)                  # array([f, g])

waveform = table.precontract({"nu": 0.1})       # fix nu once ...
waveform.evaluate(-2.0, 0.3)                    # ... then evaluate over (u, e), much faster
value, grad = waveform.value_and_grad(-2.0, 0.3)   # with the gradient along u and e
```

The [example notebooks](examples/) walk through each dimension, including
convergence, coefficient decay, pruning, and precomputation timings. The
[nD workflow notebook](examples/example_nd_workflow.ipynb) covers
precontraction, derivatives, Numba right-hand sides, domain maps, grids and
files.

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

`evaluate` takes its coordinates in the same axis order. The 1D–4D classes do
not check that they lie inside the domain: outside `[a, b]` the series
extrapolates and quickly diverges. `ChebyshevInterpolantND` raises an
`OutOfDomainError` instead.

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
3. **Evaluation.** The 1D–4D classes evaluate the tensor series with nested
   Clenshaw recursions. `ChebyshevInterpolantND` contracts the coefficient
   tensor one axis at a time with the values of $T_k$ at the point, computed
   by the stable three-term recurrence. Its first contraction is a sequence
   of contiguous vector updates (and, for a batch of points, a single matrix
   product), so its cost is dominated by one pass over the coefficients.
   Both are compiled with Numba; batches also run on NumPy or CuPy.
4. **Compression.** For smooth functions the coefficients decay geometrically
   along every axis, so most of the tensor is numerically irrelevant.
   `optimize` greedily discards trailing slices while keeping the summed
   discarded magnitude — an upper bound on the introduced uniform error —
   under a relative tolerance you choose. Trailing coefficient magnitudes also
   provide error estimates (`absolute_error`, `relative_error`) without any
   extra function samples.

## Roadmap

- Validate and tune the GPU path on CUDA hardware (custom CUDA kernels where
  benchmarks show kernel-launch overhead dominates)
- Prefactors for tabulating functions with known singular factors, if a use
  case needs them

## Development

Run the test suite with:

```bash
pip install -e ".[io]"
pip install pytest
pytest
```

Tests of the GPU path are marked `gpu` and are skipped without CuPy and a
CUDA device; on a machine with both, run them with `pytest -m gpu`.

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
