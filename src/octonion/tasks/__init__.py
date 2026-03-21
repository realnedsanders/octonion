"""Synthetic task generators for optimization landscape experiments.

Five standardized data generation tasks with known optima for rigorous
loss ratio comparisons in the go/no-go evaluation gate.

Each generator produces deterministic (train, test) TensorDataset pairs.
"""

from octonion.tasks._algebra_native import (
    build_algebra_native_multi,
    build_algebra_native_single,
)
from octonion.tasks._classification import build_classification
from octonion.tasks._cross_product import build_cross_product_recovery
from octonion.tasks._sinusoidal import build_sinusoidal_regression

__all__ = [
    "build_algebra_native_single",
    "build_algebra_native_multi",
    "build_cross_product_recovery",
    "build_sinusoidal_regression",
    "build_classification",
]
