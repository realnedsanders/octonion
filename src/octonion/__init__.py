"""Octonion: PyTorch-native octonionic algebra for ML research."""

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
