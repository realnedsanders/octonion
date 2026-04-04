"""Stabilizing normalization for algebra-valued activations.

Provides periodic unit-norm re-normalization to prevent unbounded growth
or collapse through deep operation chains. Designed to be inserted every
K layers via the ``stabilize_every`` config field in NetworkConfig.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StabilizingNorm(nn.Module):
    """Periodic unit-norm re-normalization for algebra-valued activations.

    Projects each algebra element to unit norm, preventing unbounded growth
    or collapse through deep operation chains. Inserted every K layers via
    the ``stabilize_every`` config field.

    Works for all four algebra types:
    - Real (algebra_dim=1): normalizes per-feature by absolute value
    - Complex (algebra_dim=2): normalizes per-feature by 2D Euclidean norm
    - Quaternion (algebra_dim=4): normalizes per-feature by 4D Euclidean norm
    - Octonion (algebra_dim=8): normalizes per-feature by 8D Euclidean norm

    Args:
        algebra_dim: Dimension of the algebra (1, 2, 4, or 8).
        eps: Small constant to avoid division by zero. Default: 1e-8.
    """

    def __init__(self, algebra_dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        if algebra_dim not in (1, 2, 4, 8):
            raise ValueError(
                f"algebra_dim must be 1, 2, 4, or 8, got {algebra_dim}"
            )
        self.algebra_dim = algebra_dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each algebra element to unit norm.

        Args:
            x: For algebra_dim==1: ``[..., features]``.
                For algebra_dim>1: ``[..., features, algebra_dim]``.

        Returns:
            Tensor with same shape, each algebra element having unit norm.
        """
        if self.algebra_dim == 1:
            # Real: normalize per-feature by absolute value
            norm = x.abs().clamp(min=self.eps)
            return x / norm
        else:
            # Hypercomplex: normalize along last (algebra) dimension
            norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
            return x / norm

    def extra_repr(self) -> str:
        return f"algebra_dim={self.algebra_dim}, eps={self.eps}"
