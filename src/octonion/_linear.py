"""OctonionLinear: a neural network layer computing (a * x) * b.

Implements a two-sided octonionic multiplication layer where both a and b
are learnable parameters. This is a natural linear map in the octonionic
algebra, producing 8-dimensional output from 8-dimensional input.

Parenthesization: (a * x) * b (left-to-right, fixed convention).
No bias term per user decision.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from octonion._multiplication import octonion_mul


class OctonionLinear(nn.Module):
    """Linear layer computing output = (a * x) * b.

    Both a and b are learnable nn.Parameter tensors of shape [8],
    initialized as unit-norm random octonions.

    Args:
        dtype: Tensor dtype for parameters (default float32, matching PyTorch convention).

    Example:
        >>> layer = OctonionLinear()
        >>> x = torch.randn(8)  # float32 by default
        >>> y = layer(x)  # shape [8]
    """

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        # Initialize as unit-norm octonions (preserves scale at init)
        a_init = torch.randn(8, dtype=dtype)
        a_init = a_init / torch.linalg.norm(a_init)
        b_init = torch.randn(8, dtype=dtype)
        b_init = b_init / torch.linalg.norm(b_init)

        self.a = nn.Parameter(a_init)
        self.b = nn.Parameter(b_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply (a * x) * b to input x.

        Args:
            x: Input tensor of shape [..., 8].

        Returns:
            Output tensor of shape [..., 8].
        """
        # Broadcast a to match x's batch dimensions
        a_expanded = self.a.expand_as(x)
        ax = octonion_mul(a_expanded, x)
        # Broadcast b to match ax's batch dimensions
        b_expanded = self.b.expand_as(ax)
        return octonion_mul(ax, b_expanded)
