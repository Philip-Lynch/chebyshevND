"""Shared helpers for the ChebyshevND test suite."""

import numpy as np

from chebyshevnd import chebyshev_nodes
from chebyshevnd.utils import chebyshev_gauss_lobatto_nodes


def scale_to_unit(x, a, b):
    """Map x from [a, b] to [-1, 1]."""
    return (2.0 * x - a - b) / (b - a)


def direct_eval(coeffs, ts):
    """
    Evaluate a tensor Chebyshev series at one point by direct summation,
    using T_k(t) = cos(k * arccos(t)). Each t must already lie in [-1, 1].
    Independent of both Clenshaw recursion and numpy's chebval, so it can
    serve as an oracle for either.
    """
    vecs = [np.cos(np.arange(n) * np.arccos(t)) for n, t in zip(coeffs.shape, ts)]
    letters = "ijkl"[: coeffs.ndim]
    subscripts = letters + "," + ",".join(letters) + "->"
    return float(np.einsum(subscripts, coeffs, *vecs))


def decaying_coeffs(rng, shape, base=0.5):
    """Random coefficients with geometric decay along every axis."""
    c = rng.standard_normal(shape)
    for axis, n in enumerate(shape):
        decay = base ** np.arange(n, dtype=float)
        c *= decay.reshape([n if ax == axis else 1 for ax in range(len(shape))])
    return c


def grid_axes(domains, Ns, nodes="Chebyshev"):
    """Per-axis node arrays for a tensor grid."""
    node_fn = chebyshev_nodes if nodes == "Chebyshev" else chebyshev_gauss_lobatto_nodes
    return [node_fn(N, a, b) for (a, b), N in zip(domains, Ns)]


def sample_on_grid(f, domains, Ns, nodes="Chebyshev"):
    """Sample f on a tensor grid; returns (per-axis nodes, values array)."""
    axes = grid_axes(domains, Ns, nodes)
    grids = np.meshgrid(*axes, indexing="ij")
    return axes, f(*grids)
