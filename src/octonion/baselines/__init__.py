"""Baseline comparison infrastructure for algebra-specific neural networks.

Provides per-algebra linear layers, parameter matching utilities,
and configuration dataclasses for fair R/C/H/O comparisons.
"""

from octonion.baselines._algebra_linear import (
    ComplexLinear,
    OctonionDenseLinear,
    QuaternionLinear,
    RealLinear,
)
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
from octonion.baselines._param_matching import (
    find_matched_width,
    flop_report,
    param_report,
)

__all__ = [
    # Config
    "AlgebraType",
    "NetworkConfig",
    "TrainConfig",
    "ComparisonConfig",
    # Linear layers
    "RealLinear",
    "ComplexLinear",
    "QuaternionLinear",
    "OctonionDenseLinear",
    # Initialization
    "real_init",
    "complex_init",
    "quaternion_init",
    "octonion_init",
    # Parameter matching
    "find_matched_width",
    "param_report",
    "flop_report",
]
