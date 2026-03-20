"""Configuration dataclasses for baseline comparison experiments.

Provides type-safe, IDE-friendly configuration for:
- AlgebraType: enum mapping algebra to dimension and param multiplier
- NetworkConfig: architecture specification
- TrainConfig: training hyperparameters
- ComparisonConfig: multi-algebra experiment specification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AlgebraType(Enum):
    """Algebra type with dimension and parameter multiplier.

    The multiplier is how many times the hidden width must be scaled
    relative to the octonionic baseline to achieve the same number
    of real parameters. E.g., Real needs 8x width because each
    real unit has 1 real param vs octonion's 8.
    """

    REAL = ("R", 1, 8)
    COMPLEX = ("C", 2, 4)
    QUATERNION = ("H", 4, 2)
    OCTONION = ("O", 8, 1)

    @property
    def dim(self) -> int:
        """Number of real components per algebra element."""
        return self.value[1]

    @property
    def multiplier(self) -> int:
        """Width multiplier for parameter matching vs octonionic baseline."""
        return self.value[2]

    @property
    def short_name(self) -> str:
        """Short mathematical name (R, C, H, O)."""
        return self.value[0]


@dataclass
class NetworkConfig:
    """Configuration for an algebra-specific neural network.

    Attributes:
        algebra: Which algebra to use for hidden layers.
        topology: Network topology type.
        depth: Number of hidden layers.
        base_hidden: Hidden units for octonionic baseline (others auto-scaled).
        activation: Activation function strategy.
        output_projection: How to project algebra output to real-valued output.
        use_batchnorm: Whether to use algebra-aware batch normalization.
        input_dim: Input feature dimension (real-valued).
        output_dim: Output feature dimension (real-valued).
    """

    algebra: AlgebraType
    topology: str = "mlp"  # "mlp", "conv1d", "conv2d", "recurrent"
    depth: int = 3
    base_hidden: int = 64
    activation: str = "split_relu"  # "split_relu", "split_gelu", "norm_preserving"
    output_projection: str = "real"  # "real", "flatten", "norm", "learned"
    use_batchnorm: bool = True
    stabilize_every: int | None = None  # Insert StabilizingNorm every N layers (None = disabled)
    input_dim: int = 784
    output_dim: int = 10


@dataclass
class TrainConfig:
    """Training hyperparameters.

    Attributes:
        epochs: Maximum training epochs.
        lr: Learning rate.
        optimizer: Optimizer name.
        scheduler: LR scheduler type. Options:
            - "cosine": CosineAnnealingLR (default for general use)
            - "step": StepLR with step_size = epochs // 3
            - "plateau": ReduceLROnPlateau
            - "step_cifar": Step-decay matching Gaudet & Maida 2018 / Trabelsi
              2018 CIFAR protocol: LR=lr for 10 warmup epochs, LR=lr*10 for
              epochs 10-119, LR=lr at epoch 120, LR=lr/10 at epoch 150.
        weight_decay: L2 regularization weight.
        early_stopping_patience: Epochs to wait before early stopping.
        warmup_epochs: Number of warmup epochs.
        use_amp: Whether to use automatic mixed precision. BN whitening is
            protected with autocast(enabled=False) + explicit float32 casting,
            so AMP is safe for all four algebras.
        use_compile: Whether to apply torch.compile with inductor backend
            (opt-in, experimental on ROCm). Falls back to eager mode if
            compilation fails. Default: False.
        gradient_clip_norm: Max gradient norm for clipping. 0 = disabled.
            Gaudet & Maida 2018 and Trabelsi 2018 use gradient norm clipping
            at 1.0 for CIFAR experiments.
        nesterov: Use Nesterov momentum with SGD. Both Gaudet & Maida 2018
            and Trabelsi 2018 use Nesterov momentum.
        checkpoint_every: Save checkpoint every N epochs.
        seed: Random seed for reproducibility.
        batch_size: Training batch size.
        lock_optimizer: If True, all algebras use same optimizer (Adam)
            for controlled comparison.
    """

    epochs: int = 100
    lr: float = 1e-3
    optimizer: str = "adam"
    scheduler: str = "cosine"
    weight_decay: float = 0.0
    early_stopping_patience: int = 10
    warmup_epochs: int = 5
    use_amp: bool = False
    use_compile: bool = False  # Opt-in torch.compile (experimental on ROCm)
    gradient_clip_norm: float = 0.0  # 0 = disabled; papers use 1.0
    nesterov: bool = False  # Nesterov momentum for SGD
    checkpoint_every: int = 10
    seed: int = 42
    batch_size: int = 128
    lock_optimizer: bool = False


@dataclass
class ComparisonConfig:
    """Configuration for a multi-algebra comparison experiment.

    Attributes:
        task: Task/benchmark name (e.g., "cifar10", "cifar100").
        algebras: Which algebras to compare.
        seeds: Number of random seeds per algebra.
        train_config: Training hyperparameters.
        output_dir: Base directory for experiment outputs.
    """

    task: str = "cifar10"
    algebras: list[AlgebraType] = field(
        default_factory=lambda: list(AlgebraType)
    )
    seeds: int = 10
    train_config: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "experiments"
