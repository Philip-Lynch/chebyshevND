"""
Array-backend helpers, so the batched evaluator runs unchanged on NumPy
(CPU) or CuPy (GPU) arrays. CuPy is optional: it is only imported when a
CuPy array is passed in.
"""

import numpy as np


def _is_cupy(x):
    return type(x).__module__.split(".")[0] == "cupy"


def array_namespace(*arrays):
    """
    Return the array module for the given arrays: ``cupy`` if any of them is
    a CuPy array, otherwise ``numpy``.
    """
    for x in arrays:
        if _is_cupy(x):
            import cupy

            return cupy
    return np


def to_numpy(x):
    """Return ``x`` as a NumPy array, copying it from the GPU if it is a CuPy array."""
    if _is_cupy(x):
        return x.get()
    return np.asarray(x)
