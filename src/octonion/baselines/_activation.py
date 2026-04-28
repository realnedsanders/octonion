"""Activation functions for algebra-valued neural networks.

Two strategies for applying nonlinearities to hypercomplex features:

- SplitActivation: applies activation independently to each algebra component
- NormPreservingActivation: applies activation to the algebra norm, then rescales
  to preserve direction (algebraic structure)
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# Map activation names to PyTorch functions
_ACTIVATION_FNS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": F.relu,
    "gelu": F.gelu,
    "leaky_relu": F.leaky_relu,
    "tanh": torch.tanh,
    "silu": F.silu,
}


class SplitActivation(nn.Module):
    """Apply activation function independently to each algebra component.

    This is the simplest activation strategy: treat each real component
    of the algebra element as an independent scalar and apply a pointwise
    nonlinearity. Works for any algebra dimension since PyTorch elementwise
    ops broadcast over any shape.

    Args:
        activation: Name of activation function. One of:
            "relu", "gelu", "leaky_relu", "tanh", "silu".
    """

    def __init__(self, activation: str = "relu") -> None:
        super().__init__()
        if activation not in _ACTIVATION_FNS:
            raise ValueError(
                f"Unknown activation: {activation!r}. "
                f"Supported: {list(_ACTIVATION_FNS.keys())}"
            )
        self.activation_name = activation
        self._fn = _ACTIVATION_FNS[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation elementwise.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor of same shape with activation applied to each element.
        """
        return self._fn(x)

    def extra_repr(self) -> str:
        return f"activation={self.activation_name!r}"


class NormPreservingActivation(nn.Module):
    """Apply activation to algebra norm, preserving direction.

    Computes the norm of the algebra element along the last dimension,
    applies the activation function to the norm (a scalar), then rescales
    the original element to have the activated norm. This preserves the
    algebraic direction while allowing gating of magnitude.

    For an input x with norm ||x||:
        output = x * f(||x||) / (||x|| + eps)

    where f is the activation function.

    Args:
        activation: Name of activation function. One of:
            "relu", "gelu", "leaky_relu", "tanh", "silu".
        eps: Small value for numerical stability in division.
    """

    def __init__(
        self, activation: str = "relu", eps: float = 1e-8
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATION_FNS:
            raise ValueError(
                f"Unknown activation: {activation!r}. "
                f"Supported: {list(_ACTIVATION_FNS.keys())}"
            )
        self.activation_name = activation
        self._fn = _ACTIVATION_FNS[activation]
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply norm-preserving activation.

        Args:
            x: Input tensor of shape [..., dim] where dim is the algebra dimension.

        Returns:
            Tensor of same shape with activated norm and preserved direction.
        """
        # Compute norm along last dimension
        norm = x.norm(dim=-1, keepdim=True)  # [..., 1]

        # Apply activation to norm
        activated_norm = self._fn(norm)  # [..., 1]

        # Compute scale factor: activated_norm / (norm + eps)
        scale = activated_norm / (norm + self.eps)  # [..., 1]

        return x * scale  # type: ignore[no-any-return]

    def extra_repr(self) -> str:
        return f"activation={self.activation_name!r}, eps={self.eps}"
