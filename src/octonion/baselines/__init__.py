"""Baseline comparison infrastructure for algebra-specific neural networks.

Provides per-algebra linear layers, parameter matching utilities,
and configuration dataclasses for fair R/C/H/O comparisons.
"""

from octonion.baselines._config import (
    AlgebraType,
    ComparisonConfig,
    NetworkConfig,
    TrainConfig,
)
from octonion.baselines._initialization import (
    complex_init,
    octonion_init,
    quaternion_init,
    real_init,
)

__all__ = [
    # Config
    "AlgebraType",
    "NetworkConfig",
    "TrainConfig",
    "ComparisonConfig",
    # Initialization
    "real_init",
    "complex_init",
    "quaternion_init",
    "octonion_init",
]
