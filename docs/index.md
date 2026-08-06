# ChebyshevND

Fast multidimensional Chebyshev interpolation in Python, using DCTs to compute
coefficients, Numba-accelerated Clenshaw recursions for evaluation, and
structured coefficient pruning for compression.

For smooth functions, Chebyshev interpolation converges *spectrally*: the
error decays exponentially with the number of nodes per axis. ChebyshevND
turns a tensor grid of function samples into an interpolant that can be
evaluated anywhere in the domain at close to machine precision — with built-in
error estimates that come free from the coefficient decay.

## Installation

```bash
pip install chebyshevnd
```

Requires Python ≥ 3.10; `numpy`, `scipy`, and `numba` are installed
automatically.

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

The example notebooks below walk through each dimension, including
convergence, coefficient decay, pruning, and precomputation timings.

```{toctree}
:maxdepth: 1
:caption: Examples

examples/example_1d_interp
examples/example_2d_interp
examples/example_3d_interp
examples/example_4d_interp
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```

## Citation

The techniques implemented in this package were derived and developed for:

> H. Khalvati, P. Lynch, O. Burke, L. Speri, M. van de Meent, and Z. Nasipak,
> *Systematic errors in fast relativistic waveforms for Extreme Mass Ratio
> Inspirals*, [Phys. Rev. D **113**, 084042 (2026)](https://doi.org/10.1103/4ly7-zn15),
> [arXiv:2509.08875](https://arxiv.org/abs/2509.08875).

If this package contributes to your research, please cite that paper —
machine-readable citation metadata is in the repository's `CITATION.cff`.
