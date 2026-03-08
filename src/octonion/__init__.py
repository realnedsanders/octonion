"""Octonion: PyTorch-native octonionic algebra for ML research.

The ``octonion.calculus`` submodule provides GHR differentiation tools::

    from octonion import calculus
    from octonion.calculus import ghr_derivative, jacobian_mul, octonion_gradcheck
"""

from octonion._cayley_dickson import cayley_dickson_mul
from octonion._fano import FANO_PLANE, FanoPlane
from octonion._linear import OctonionLinear
from octonion._linear_algebra import left_mul_matrix, right_mul_matrix
from octonion._multiplication import STRUCTURE_CONSTANTS, octonion_mul
from octonion._octonion import Octonion, PureOctonion, UnitOctonion, associator
from octonion._operations import (
    commutator,
    cross_product,
    inner_product,
    octonion_exp,
    octonion_log,
)
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
    # Extended operations
    "octonion_exp",
    "octonion_log",
    "commutator",
    "inner_product",
    "cross_product",
    # Linear algebra
    "left_mul_matrix",
    "right_mul_matrix",
    # Neural network layers
    "OctonionLinear",
    # Random generation
    "random_octonion",
    "random_unit_octonion",
    "random_pure_octonion",
    # Data structures
    "FANO_PLANE",
    "FanoPlane",
    "STRUCTURE_CONSTANTS",
]
