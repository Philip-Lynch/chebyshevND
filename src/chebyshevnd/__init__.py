
"""
ChebyshevND: Fast multidimensional Chebyshev interpolation in Python.
"""

# Core interpolant classes
from .interp1d import ChebyshevInterpolant1D
from .interp2d import ChebyshevInterpolant2D
from .interp3d import ChebyshevInterpolant3D
from .interp4d import ChebyshevInterpolant4D
from .errors import OutOfDomainError
from .interpnd import ChebyshevInterpolantND
from .batch import ChebyshevBatch
from .batched import evaluate_batch, value_and_grad_batch
from .derivatives import differentiate_coefficients
from .kernels import RawTable, Workspace, eval_grad_into, eval_into
from .maps import AxisMap, RawMappedTable, eval_grad_physical_into, eval_physical_into
from .mapped import MappedBatch, MappedInterpolant
from .grid import Axis, ChebyshevGrid
from .io import load, read_metadata, save

# Clenshaw evaluation kernels
from .clenshaw import(clenshaw_1d,
                       clenshaw_2d,
                       clenshaw_3d,
                       clenshaw_4d,
                       clenshaw_nd
)

# Utilities
from .utils import (
    find_significant_coeffs_1d,
    find_significant_coeffs_2d,
    renormalize_coefficients_GL,
    renormalize_coefficients_Chebyshev,
    compute_chebyshev_coefficients_nd,
    chebyshev_values_at_nodes_nd,
    chebyshev_nodes,
    chebyshev_gauss_lobatto_nodes,
    prune_trailing_slices_greedy_nd
)

__all__ = [
    "ChebyshevInterpolant1D",
    "ChebyshevInterpolant2D",
    "ChebyshevInterpolant3D",
    "ChebyshevInterpolant4D",
    "ChebyshevInterpolantND",
    "ChebyshevBatch",
    "OutOfDomainError",
    "evaluate_batch",
    "value_and_grad_batch",
    "RawTable",
    "Workspace",
    "eval_into",
    "eval_grad_into",
    "AxisMap",
    "MappedInterpolant",
    "MappedBatch",
    "RawMappedTable",
    "eval_physical_into",
    "eval_grad_physical_into",
    "Axis",
    "ChebyshevGrid",
    "save",
    "load",
    "read_metadata",
    "differentiate_coefficients",
    "clenshaw_1d",
    "clenshaw_2d",
    "clenshaw_3d",
    "clenshaw_4d",
    "clenshaw_nd",
    "find_significant_coeffs_1d",
    "find_significant_coeffs_2d",
    "renormalize_coefficients_GL",
    "renormalize_coefficients_Chebyshev",
    "compute_chebyshev_coefficients_nd",
    "chebyshev_values_at_nodes_nd",
    "chebyshev_nodes",
    "chebyshev_gauss_lobatto_nodes",
    "prune_trailing_slices_greedy_nd"
]
