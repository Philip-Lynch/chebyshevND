# ChebyshevND

[![Tests](https://github.com/Philip-Lynch/chebyshevND/actions/workflows/tests.yml/badge.svg)](https://github.com/Philip-Lynch/chebyshevND/actions/workflows/tests.yml)
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

In higher dimensions, sample on the tensor grid (`indexing='ij'`) and pass
the value array with per-axis domain bounds:

```python
from chebyshevnd import ChebyshevInterpolant3D, chebyshev_nodes

f = lambda x, y, z: np.sin(x) * np.cos(y) * np.exp(-z)

x = chebyshev_nodes(20, 0.0, 2.0)
y = chebyshev_nodes(20, -1.0, 3.0)
z = chebyshev_nodes(20, 1.0, 4.0)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

interp = ChebyshevInterpolant3D(f(X, Y, Z), 0.0, 2.0, -1.0, 3.0, 1.0, 4.0)
interp.evaluate(1.3, 0.5, 2.5)

interp.optimize(rel_tol=1e-8)   # prune insignificant coefficients
interp.effective_grid()         # e.g. (10, 13, 11) — down from (20, 20, 20)

interp.precompute_x(1.3)        # collapse x once ...
interp.evaluate_yz(0.5, 2.5)    # ... then evaluate repeatedly in (y, z), ~3x faster
```

The [example notebooks](examples/) walk through each dimension, including
convergence, coefficient decay, pruning, and precomputation timings.

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

## Citation

If this package contributes to your research, please cite it — citation
metadata is in [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
