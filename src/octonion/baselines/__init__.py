"""Baseline comparison infrastructure for algebra-specific neural networks.

Provides per-algebra linear layers, convolutional layers, normalization,
activation functions, parameter matching utilities, and configuration
dataclasses for fair R/C/H/O comparisons.
"""

from octonion.baselines._activation import (
    NormPreservingActivation,
    SplitActivation,
)
from octonion.baselines._algebra_conv import (
    ComplexConv1d,
    ComplexConv2d,
    OctonionConv1d,
    OctonionConv2d,
    QuaternionConv1d,
    QuaternionConv2d,
    RealConv1d,
    RealConv2d,
)
from octonion.baselines._algebra_linear import (
    ComplexLinear,
    OctonionDenseLinear,
    QuaternionLinear,
    RealLinear,
)
from octonion.baselines._algebra_rnn import (
    ComplexGRUCell,
    OctonionLSTMCell,
    QuaternionLSTMCell,
    RealLSTMCell,
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
from octonion.baselines._network import (
    AlgebraNetwork,
)
from octonion.baselines._normalization import (
    ComplexBatchNorm,
    OctonionBatchNorm,
    QuaternionBatchNorm,
    RealBatchNorm,
)
from octonion.baselines._param_matching import (
    find_matched_width,
    flop_report,
    param_report,
)
from octonion.baselines._plotting import (
    plot_comparison_bars,
    plot_convergence,
    plot_param_table,
)
from octonion.baselines._stats import (
    cohen_d,
    confidence_interval,
    holm_bonferroni,
    paired_comparison,
)
from octonion.baselines._trainer import (
    evaluate,
    load_checkpoint,
    run_optuna_study,
    save_checkpoint,
    seed_everything,
    train_model,
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
    # Recurrent cells
    "RealLSTMCell",
    "ComplexGRUCell",
    "QuaternionLSTMCell",
    "OctonionLSTMCell",
    # Network skeleton
    "AlgebraNetwork",
    # Convolutional layers
    "RealConv1d",
    "RealConv2d",
    "ComplexConv1d",
    "ComplexConv2d",
    "QuaternionConv1d",
    "QuaternionConv2d",
    "OctonionConv1d",
    "OctonionConv2d",
    # Normalization
    "RealBatchNorm",
    "ComplexBatchNorm",
    "QuaternionBatchNorm",
    "OctonionBatchNorm",
    # Activation
    "SplitActivation",
    "NormPreservingActivation",
    # Initialization
    "real_init",
    "complex_init",
    "quaternion_init",
    "octonion_init",
    # Parameter matching
    "find_matched_width",
    "param_report",
    "flop_report",
    # Training
    "seed_everything",
    "train_model",
    "evaluate",
    "save_checkpoint",
    "load_checkpoint",
    "run_optuna_study",
    # Statistics
    "paired_comparison",
    "cohen_d",
    "holm_bonferroni",
    "confidence_interval",
    # Plotting
    "plot_convergence",
    "plot_comparison_bars",
    "plot_param_table",
]
