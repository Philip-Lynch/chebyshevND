
"""
chebND: Fast multidimensional Chebyshev interpolation in Python.
Currently supports upto 4D interpolation. Higher dimensions coming soon!
"""

# Core interpolant class (1D)
from .interp1d import ChebyshevInterpolant1D
from .interp2d import ChebyshevInterpolant2D
from .interp3d import ChebyshevInterpolant3D
from .interp4d import ChebyshevInterpolant4D

# Core 1D Interpolation API
from .interp1d import compute_chebyshev_coefficients_1d
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
    chebyshev_nodes,
    prune_trailing_slices_greedy_nd
)

__all__ = [
    "ChebyshevInterpolant1D",
    "ChebyshevInterpolant2D",
    "ChebyshevInterpolant3D",
    "ChebyshevInterpolant4D",
    "compute_chebyshev_coefficients_1d",
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
    "chebyshev_nodes",
    "prune_trailing_slices_greedy_nd"
]