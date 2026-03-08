"""Octonion: PyTorch-native octonionic algebra for ML research."""

from octonion._cayley_dickson import cayley_dickson_mul
from octonion._fano import FANO_PLANE, FanoPlane
from octonion._multiplication import STRUCTURE_CONSTANTS, octonion_mul
from octonion._octonion import Octonion, PureOctonion, UnitOctonion, associator
from octonion._random import random_octonion, random_pure_octonion, random_unit_octonion
from octonion._tower import Complex, Quaternion, Real
from octonion._types import NormedDivisionAlgebra

__all__ = [
    # Core types
    "Octonion",
    "UnitOctonion",
    "PureOctonion",
    # Tower types
    "Real",
    "Complex",
    "Quaternion",
    # Abstract base
    "NormedDivisionAlgebra",
    # Functions
    "associator",
    "octonion_mul",
    "cayley_dickson_mul",
    # Random generation
    "random_octonion",
    "random_unit_octonion",
    "random_pure_octonion",
    # Data structures
    "FANO_PLANE",
    "FanoPlane",
    "STRUCTURE_CONSTANTS",
]
