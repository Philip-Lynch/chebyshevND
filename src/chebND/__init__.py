
"""
chebND: Fast multidimensional Chebyshev interpolation in Python.
Currently supports 1D interpolation. Higher dimensions coming soon!
"""

# Core interpolant class (1D)
from .interp1d import ChebyshevInterpolant1D
from .interp2d import ChebyshevInterpolant2D

# Core 1D Interpolation API
from .interp1d import compute_chebyshev_coefficients_1d
from .clenshaw import clenshaw_1d, clenshaw_2d

# Utilities
from .utils import (
    find_significant_coeffs_1d,
    find_significant_coeffs_2d
)

__all__ = [
    "ChebyshevInterpolant1D",
    "ChebyshevInterpolant2D",
    "compute_chebyshev_coefficients_1d",
    "clenshaw_1d",
    "clenshaw_2d",
    "find_significant_coeffs_1d",
    "find_significant_coeffs_2d"
]