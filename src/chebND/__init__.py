
"""
chebND: Fast multidimensional Chebyshev interpolation in Python.
Currently supports 1D interpolation. Higher dimensions coming soon!
"""

# Core interpolant class (1D)
from .interp1d import ChebyshevInterpolant1D

# Core 1D Interpolation API
from .interp1d import compute_chebyshev_coefficients_1d
from .clenshaw import clenshaw_1d

# Utilities
from .utils import (
    sort_2d_by_first_column_desc,
    read_1d_data_from_file,
    find_significant_coeffs_1d,
)

__all__ = [
    "ChebyshevInterpolant1D",
    "compute_chebyshev_coefficients_1d",
    "clenshaw_1d",
    "sort_2d_by_first_column_desc",
    "read_1d_data_from_file",
    "find_significant_coeffs_1d",
]