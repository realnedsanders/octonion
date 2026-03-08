"""Parameter matching utilities and FLOP reporting.

Provides:
- find_matched_width: Binary search for hidden width achieving target param count
- param_report: Per-layer parameter breakdown
- flop_report: Per-layer FLOP counts via torchinfo (for transparency, not matching)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from octonion.baselines._algebra_linear import (
    ComplexLinear,
    OctonionDenseLinear,
    QuaternionLinear,
    RealLinear,
)
from octonion.baselines._config import AlgebraType


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
        return self.output_proj(h)


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

    Args:
        target_params: Target number of trainable parameters.
        algebra: Which algebra to use.
        topology: Network topology ("mlp" supported).
        depth: Number of hidden layers.
        tolerance: Acceptable relative error (default 1%).
        input_dim: Input feature dimension.
        output_dim: Output feature dimension.
        **kwargs: Additional keyword arguments (reserved for future topologies).

    Returns:
        Hidden width (int) achieving target within tolerance.

    Raises:
        ValueError: If no width can match the target within tolerance.
    """
    if topology != "mlp":
        raise NotImplementedError(
            f"Topology {topology!r} not yet supported for param matching. "
            f"Only 'mlp' is currently implemented."
        )

    lo, hi = 1, 4096  # generous upper bound
    best_width, best_diff = lo, float("inf")

    while lo <= hi:
        mid = (lo + hi) // 2
        model = _build_simple_mlp(
            algebra=algebra,
            hidden=mid,
            depth=depth,
            input_dim=input_dim,
            output_dim=output_dim,
        )
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


def param_report(model: nn.Module) -> list[dict]:
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
) -> dict:
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
