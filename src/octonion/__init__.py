"""Octonionic algebra library for geometric ML research.

PyTorch-native implementation of octonion arithmetic with Fano plane
structure constants and Cayley-Dickson cross-check.
"""

__version__ = "0.1.0"

from octonion._cayley_dickson import cayley_dickson_mul
from octonion._fano import FANO_PLANE, FanoPlane
from octonion._multiplication import STRUCTURE_CONSTANTS, octonion_mul

__all__ = [
    "FANO_PLANE",
    "FanoPlane",
    "STRUCTURE_CONSTANTS",
    "cayley_dickson_mul",
    "octonion_mul",
]
