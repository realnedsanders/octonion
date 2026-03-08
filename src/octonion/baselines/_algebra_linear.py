"""Per-algebra linear layers for baseline comparison experiments.

Each layer implements a full linear map over its respective algebra:
- RealLinear: standard nn.Linear wrapper
- ComplexLinear: complex matrix-vector product (2 weight matrices)
- QuaternionLinear: Hamilton product (4 weight matrices)
- OctonionDenseLinear: full octonionic product via structure constants (8 weight matrices)

All layers follow the [..., in_features, dim] -> [..., out_features, dim] convention,
where dim is the algebra dimension (1, 2, 4, or 8). RealLinear operates on [..., in_features]
directly (dim=1 is implicit).

Parameter counts per layer (no bias):
- RealLinear(in, out): in * out
- ComplexLinear(in, out): 2 * in * out
- QuaternionLinear(in, out): 4 * in * out
- OctonionDenseLinear(in, out): 8 * in * out
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from octonion._multiplication import STRUCTURE_CONSTANTS
from octonion.baselines._initialization import (
    complex_init,
    octonion_init,
    quaternion_init,
    real_init,
)


class RealLinear(nn.Module):
    """Real-valued linear layer wrapping nn.Linear.

    Input shape: [..., in_features]
    Output shape: [..., out_features]

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If True, adds a learnable bias. Default: True.
        dtype: Tensor dtype. Default: float32.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
        real_init(self.linear.weight, criterion="he")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: standard linear transformation."""
        return self.linear(x)


class ComplexLinear(nn.Module):
    """Complex-valued linear layer using the complex product rule.

    Computes W*x where W = W_r + W_i*i and x = x_r + x_i*i:
      out_r = W_r @ x_r - W_i @ x_i
      out_i = W_i @ x_r + W_r @ x_i

    Input shape: [..., in_features, 2]
    Output shape: [..., out_features, 2]

    Args:
        in_features: Number of complex input features.
        out_features: Number of complex output features.
        bias: If True, adds a learnable bias. Default: True.
        dtype: Tensor dtype. Default: float32.
    """

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

        self.W_r = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.W_i = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))

        if bias:
            self.bias_r = nn.Parameter(torch.zeros(out_features, dtype=dtype))
            self.bias_i = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter("bias_r", None)
            self.register_parameter("bias_i", None)

        complex_init(self.W_r, self.W_i, criterion="glorot")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: complex matrix-vector product.

        Args:
            x: Input tensor of shape [..., in_features, 2].

        Returns:
            Output tensor of shape [..., out_features, 2].
        """
        x_r = x[..., 0]  # [..., in_features]
        x_i = x[..., 1]  # [..., in_features]

        # Complex multiplication: (W_r + W_i*i)(x_r + x_i*i)
        out_r = F.linear(x_r, self.W_r) - F.linear(x_i, self.W_i)
        out_i = F.linear(x_r, self.W_i) + F.linear(x_i, self.W_r)

        if self.bias_r is not None:
            out_r = out_r + self.bias_r
            out_i = out_i + self.bias_i

        return torch.stack([out_r, out_i], dim=-1)


class QuaternionLinear(nn.Module):
    """Quaternion-valued linear layer using the Hamilton product.

    Computes W*x where W = W_r + W_i*i + W_j*j + W_k*k and
    x = x_r + x_i*i + x_j*j + x_k*k using the full 16-term Hamilton product.

    Sign pattern cross-referenced with verified Quaternion.__mul__ in _tower.py:
      out_r = W_r*x_r - W_i*x_i - W_j*x_j - W_k*x_k
      out_i = W_r*x_i + W_i*x_r + W_j*x_k - W_k*x_j
      out_j = W_r*x_j - W_i*x_k + W_j*x_r + W_k*x_i
      out_k = W_r*x_k + W_i*x_j - W_j*x_i + W_k*x_r

    Input shape: [..., in_features, 4]
    Output shape: [..., out_features, 4]

    Args:
        in_features: Number of quaternion input features.
        out_features: Number of quaternion output features.
        bias: If True, adds a learnable bias. Default: True.
        dtype: Tensor dtype. Default: float32.
    """

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

        self.W_r = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.W_i = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.W_j = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.W_k = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, 4, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        quaternion_init(self.W_r, self.W_i, self.W_j, self.W_k, criterion="glorot")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: Hamilton product matrix-vector multiplication.

        Args:
            x: Input tensor of shape [..., in_features, 4].

        Returns:
            Output tensor of shape [..., out_features, 4].
        """
        x_r = x[..., 0]  # [..., in_features]
        x_i = x[..., 1]
        x_j = x[..., 2]
        x_k = x[..., 3]

        # Hamilton product -- 16 terms with signs from i^2=j^2=k^2=ijk=-1
        # Cross-referenced with Quaternion.__mul__ in _tower.py
        out_r = (
            F.linear(x_r, self.W_r)
            - F.linear(x_i, self.W_i)
            - F.linear(x_j, self.W_j)
            - F.linear(x_k, self.W_k)
        )
        out_i = (
            F.linear(x_i, self.W_r)
            + F.linear(x_r, self.W_i)
            + F.linear(x_k, self.W_j)
            - F.linear(x_j, self.W_k)
        )
        out_j = (
            F.linear(x_j, self.W_r)
            - F.linear(x_k, self.W_i)
            + F.linear(x_r, self.W_j)
            + F.linear(x_i, self.W_k)
        )
        out_k = (
            F.linear(x_k, self.W_r)
            + F.linear(x_j, self.W_i)
            - F.linear(x_i, self.W_j)
            + F.linear(x_r, self.W_k)
        )

        result = torch.stack([out_r, out_i, out_j, out_k], dim=-1)

        if self.bias is not None:
            result = result + self.bias

        return result


class OctonionDenseLinear(nn.Module):
    """Full octonionic linear layer using structure constants.

    Unlike OctonionLinear (which computes rank-1 bilinear (a*x)*b with 16 params),
    this implements a full octonionic matrix-vector product with 8 weight matrices,
    giving 8 * in_features * out_features real parameters. This is the correct
    layer for parameter-matched fair comparisons with R/C/H baselines.

    Forward: for output component k,
      out_k = sum_{i,j} C[i,j,k] * F.linear(x_j, W_i)
    where C is the [8,8,8] structure constants tensor. This computes W*x
    where W is an octonionic weight (components W_0..W_7) and x is the input.

    Input shape: [..., in_features, 8]
    Output shape: [..., out_features, 8]

    Args:
        in_features: Number of octonionic input features.
        out_features: Number of octonionic output features.
        bias: If True, adds a learnable bias. Default: True.
        dtype: Tensor dtype. Default: float32.
    """

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

        # 8 weight matrices, one per basis element
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            for _ in range(8)
        ])

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, 8, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        octonion_init(list(self.weights), criterion="glorot")

        # Cache structure constants for forward pass
        self.register_buffer(
            "_C", STRUCTURE_CONSTANTS.to(dtype=dtype), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: full octonionic matrix-vector product.

        Args:
            x: Input tensor of shape [..., in_features, 8].

        Returns:
            Output tensor of shape [..., out_features, 8].
        """
        # Octonionic product: (W * x)_k = sum_{i,j} C[i,j,k] * W_i * x_j
        # W_i are weight matrices [out_features, in_features], one per basis
        # x[..., j] are input components [..., in_features]
        # F.linear(x_j, W_i) computes W_i @ x_j -> [..., out_features]

        C = self._C  # [8, 8, 8]
        batch_shape = x.shape[:-2]
        out_components = []

        for k in range(8):
            # out_k = sum_{i,j} C[i,j,k] * F.linear(x[..., j], W_i)
            acc = torch.zeros(*batch_shape, self.out_features, dtype=x.dtype, device=x.device)
            for i in range(8):
                for j in range(8):
                    c_ijk = C[i, j, k].item()
                    if c_ijk != 0.0:
                        acc = acc + c_ijk * F.linear(x[..., j], self.weights[i])
            out_components.append(acc)

        result = torch.stack(out_components, dim=-1)  # [..., out_features, 8]

        if self.bias is not None:
            result = result + self.bias

        return result
