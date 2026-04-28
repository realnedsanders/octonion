"""Parameter matching utilities and FLOP reporting.

Provides:
- find_matched_width: Binary search for hidden width achieving target param count
- param_report: Per-layer parameter breakdown
- flop_report: Per-layer FLOP counts via torchinfo (for transparency, not matching)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from octonion.baselines._algebra_linear import (
    ComplexLinear,
    OctonionDenseLinear,
    QuaternionLinear,
    RealLinear,
)
from octonion.baselines._config import AlgebraType
from octonion.baselines._dense_mixing import DenseMixingLinear
from octonion.baselines._phm_linear import PHM8Linear


class _SimpleAlgebraMLP(nn.Module):
    """Simple trainable MLP with algebra-specific hidden layers.

    Used for parameter counting AND training in comparison experiments.
    Handles the reshape between real-valued input/output projections
    and algebra-valued hidden layers.

    Architecture:
    - Input projection: nn.Linear(input_dim, hidden * algebra.dim) + reshape
    - Hidden layers: AlgebraLinear(hidden, hidden) x depth
    - Output projection: flatten algebra components + nn.Linear to output_dim
    """

    def __init__(
        self,
        algebra: AlgebraType,
        hidden: int,
        depth: int,
        input_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.algebra = algebra
        self.hidden = hidden
        self.dim = algebra.dim

        # Select the appropriate linear layer class
        LayerClass: type[nn.Module]
        if algebra == AlgebraType.REAL:
            LayerClass = RealLinear
        elif algebra == AlgebraType.COMPLEX:
            LayerClass = ComplexLinear
        elif algebra == AlgebraType.QUATERNION:
            LayerClass = QuaternionLinear
        elif algebra == AlgebraType.OCTONION:
            LayerClass = OctonionDenseLinear
        elif algebra == AlgebraType.PHM8:
            LayerClass = PHM8Linear
        elif algebra == AlgebraType.R8_DENSE:
            LayerClass = DenseMixingLinear
        else:
            raise ValueError(f"Unknown algebra: {algebra}")

        # Input projection: real -> algebra
        self.input_proj = nn.Linear(input_dim, hidden * algebra.dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(depth):
            self.hidden_layers.append(LayerClass(hidden, hidden))

        # Output projection: algebra -> real
        self.output_proj = nn.Linear(hidden * algebra.dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with proper reshape between projections and algebra layers.

        Args:
            x: [B, input_dim] real-valued input.

        Returns:
            [B, output_dim] real-valued output.
        """
        # Input projection: [B, input_dim] -> [B, hidden * dim]
        h = self.input_proj(x)

        # Reshape to algebra-valued: [B, hidden, dim] (skip reshape for real)
        if self.dim > 1:
            h = h.view(h.shape[0], self.hidden, self.dim)

        # Hidden layers
        for layer in self.hidden_layers:
            h = layer(h)

        # Flatten back to real: [B, hidden * dim] or [B, hidden]
        if self.dim > 1:
            h = h.reshape(h.shape[0], -1)

        # Output projection: [B, hidden * dim] -> [B, output_dim]
        return self.output_proj(h)  # type: ignore[no-any-return]


def _build_simple_mlp(
    algebra: AlgebraType,
    hidden: int,
    depth: int,
    input_dim: int,
    output_dim: int,
) -> nn.Module:
    """Build a simple trainable MLP with algebra-specific hidden layers.

    Architecture:
    - Input projection: nn.Linear(input_dim, hidden * algebra.dim) + reshape
    - Hidden layers: AlgebraLinear(hidden, hidden) x depth
    - Output projection: flatten algebra components + nn.Linear to output_dim

    This model is used both for parameter counting in find_matched_width
    and for actual training in run_comparison.

    Args:
        algebra: Which algebra to use for hidden layers.
        hidden: Number of algebra units per hidden layer.
        depth: Number of hidden algebra linear layers.
        input_dim: Real-valued input dimension.
        output_dim: Real-valued output dimension.

    Returns:
        Trainable _SimpleAlgebraMLP model.
    """
    return _SimpleAlgebraMLP(
        algebra=algebra,
        hidden=hidden,
        depth=depth,
        input_dim=input_dim,
        output_dim=output_dim,
    )


def _build_conv_model(
    algebra: AlgebraType,
    base_hidden: int,
    depth: int,
    input_dim: int,
    output_dim: int,
    **kwargs: object,
) -> nn.Module:
    """Build an AlgebraNetwork with conv2d topology for param counting.

    This is the conv2d analog of _build_simple_mlp. Used both for
    parameter counting in find_matched_width and for model building
    in run_comparison.

    Args:
        algebra: Which algebra to use.
        base_hidden: Base filter count (before algebra multiplier scaling).
        depth: Number of residual blocks (distributed across 3 stages).
        input_dim: Number of input channels (e.g. 3 for RGB).
        output_dim: Number of output classes.
        **kwargs: Optional overrides for activation, output_projection, use_batchnorm.

    Returns:
        AlgebraNetwork with conv2d topology.
    """
    from octonion.baselines._config import NetworkConfig
    from octonion.baselines._network import AlgebraNetwork

    config = NetworkConfig(
        algebra=algebra,
        topology="conv2d",
        depth=depth,
        base_hidden=base_hidden,
        input_dim=input_dim,
        output_dim=output_dim,
        activation=str(kwargs.get("activation", "split_relu")),
        output_projection=str(kwargs.get("output_projection", "flatten")),
        use_batchnorm=bool(kwargs.get("use_batchnorm", True)),
    )
    return AlgebraNetwork(config)


def find_matched_width(
    target_params: int,
    algebra: AlgebraType,
    topology: str,
    depth: int,
    tolerance: float = 0.01,
    input_dim: int = 784,
    output_dim: int = 10,
    **kwargs: object,
) -> int:
    """Binary search for hidden width achieving target param count within tolerance.

    Builds temporary models at candidate widths and counts all trainable
    parameters. Converges to a width where |actual - target| / target <= tolerance.

    For MLP topology, returns the algebra-unit hidden width (passed to _SimpleAlgebraMLP).
    For conv2d topology, returns the base_hidden value (base filter count for NetworkConfig).

    Args:
        target_params: Target number of trainable parameters.
        algebra: Which algebra to use.
        topology: Network topology ("mlp" or "conv2d").
        depth: Number of hidden layers.
        tolerance: Acceptable relative error (default 1%).
        input_dim: Input feature dimension.
        output_dim: Output feature dimension.
        **kwargs: Additional keyword arguments (e.g. activation, output_projection
            for conv2d topology).

    Returns:
        Hidden width (int) achieving target within tolerance.

    Raises:
        ValueError: If no width can match the target within tolerance.
        NotImplementedError: If topology is not "mlp" or "conv2d".
    """
    if topology == "mlp":
        lo, hi = 1, 4096  # generous upper bound
        def build_fn(w: int) -> nn.Module:
            return _build_simple_mlp(
                    algebra=algebra, hidden=w, depth=depth,
                    input_dim=input_dim, output_dim=output_dim,
                )
    elif topology == "conv2d":
        # Start with a tight upper bound. Conv2d params grow quadratically
        # with base_hidden * multiplier, so bh=64 gives 512 base filters
        # for real (multiplier=8), which is already huge. Start at 64 and
        # expand only if needed.
        lo, hi = 1, 64
        def build_fn(w: int) -> nn.Module:
            return _build_conv_model(
                    algebra=algebra, base_hidden=w, depth=depth,
                    input_dim=input_dim, output_dim=output_dim, **kwargs,
                )
        # Quick check: if hi is too small, expand
        hi_model = build_fn(hi)
        hi_params = sum(p.numel() for p in hi_model.parameters() if p.requires_grad)
        while hi_params < target_params and hi < 512:
            hi = min(hi * 2, 512)
            hi_model = build_fn(hi)
            hi_params = sum(p.numel() for p in hi_model.parameters() if p.requires_grad)
    else:
        raise NotImplementedError(
            f"Topology {topology!r} not supported for param matching. "
            f"Supported: 'mlp', 'conv2d'."
        )

    best_width, best_diff = lo, float("inf")

    while lo <= hi:
        mid = (lo + hi) // 2
        model = build_fn(mid)
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        diff = abs(count - target_params) / target_params

        if diff < best_diff:
            best_diff = diff
            best_width = mid

        if diff <= tolerance:
            return mid
        elif count < target_params:
            lo = mid + 1
        else:
            hi = mid - 1

    if best_diff > tolerance:
        raise ValueError(
            f"Cannot match {target_params} params within {tolerance * 100:.1f}% "
            f"for {algebra.short_name}. Best width: {best_width} "
            f"({best_diff * 100:.2f}% error)."
        )
    return best_width


def param_report(model: nn.Module) -> list[dict[str, Any]]:
    """Per-layer parameter breakdown.

    Args:
        model: PyTorch model to analyze.

    Returns:
        List of dicts with keys: name, shape, real_params, pct.
    """
    total = sum(p.numel() for p in model.parameters())
    if total == 0:
        return []

    entries = []
    for name, param in model.named_parameters():
        numel = param.numel()
        entries.append({
            "name": name,
            "shape": list(param.shape),
            "real_params": numel,
            "pct": numel / total * 100.0,
        })

    return entries


def flop_report(
    model: nn.Module,
    input_size: tuple[int, ...],
    device: str = "cpu",
) -> dict[str, Any]:
    """Per-layer FLOP counts via torchinfo.

    Reported for transparency per CONTEXT.md decision. FLOPs are NOT matched
    across algebras -- only parameters are matched.

    Args:
        model: PyTorch model to analyze.
        input_size: Input tensor size (including batch dimension).
        device: Device for computation. Default: "cpu".

    Returns:
        Dict with keys:
            - total_mult_adds: Total multiply-accumulate operations.
            - per_layer: List of dicts with name and mult_adds per layer.
    """
    import torchinfo

    summary = torchinfo.summary(
        model,
        input_size=input_size,
        device=device,
        verbose=0,
    )

    per_layer = []
    for layer_info in summary.summary_list:
        if layer_info.is_leaf_layer:
            per_layer.append({
                "name": layer_info.var_name or str(layer_info.class_name),
                "mult_adds": layer_info.macs or 0,
            })

    return {
        "total_mult_adds": summary.total_mult_adds or 0,
        "per_layer": per_layer,
    }
