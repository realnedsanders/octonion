"""R8 Dense Mixing linear layer -- the "no algebra structure" baseline.

Implements a full dense linear map from R^{in_f * 8} to R^{out_f * 8}
with no algebraic structure whatsoever. This is deliberately the simplest
possible baseline: it uses a single weight matrix that freely mixes all
component-feature pairs.

This baseline answers: "Does the octonionic multiplication structure
actually help, or would any mixing of 8-dimensional features do equally
well?" If OctonionDenseLinear outperforms DenseMixingLinear at matched
parameter counts, the structured mixing provides genuine value.

Parameter count (no bias): out_f * 8 * in_f * 8
Parameter count (with bias): out_f * 8 * in_f * 8 + out_f * 8
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseMixingLinear(nn.Module):
    """Dense mixing linear layer with no algebra structure.

    Equivalent to nn.Linear(in_features * 8, out_features * 8) with
    reshape wrappers to maintain the [..., features, 8] convention.

    Input shape: [..., in_features, 8]
    Output shape: [..., out_features, 8]

    Args:
        in_features: Number of 8D input features.
        out_features: Number of 8D output features.
        bias: If True, adds a learnable bias. Default: True.
        dtype: Tensor dtype. Default: float32.
    """

    DIM: int = 8

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Full dense weight: [out_f * 8, in_f * 8]
        self.weight = nn.Parameter(
            torch.empty(out_features * self.DIM, in_features * self.DIM, dtype=dtype)
        )
        nn.init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features * self.DIM, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: dense linear transformation.

        Args:
            x: Input tensor of shape [..., in_features, 8].

        Returns:
            Output tensor of shape [..., out_features, 8].
        """
        batch_shape = x.shape[:-2]

        # Flatten: [..., in_f, 8] -> [..., in_f * 8]
        x_flat = x.reshape(*batch_shape, self.in_features * self.DIM)

        # Linear: [..., in_f * 8] -> [..., out_f * 8]
        out_flat = F.linear(x_flat, self.weight, self.bias)

        # Reshape: [..., out_f * 8] -> [..., out_f, 8]
        return out_flat.reshape(*batch_shape, self.out_features, self.DIM)
