"""Per-algebra convolutional layers for baseline comparison experiments.

Each layer implements convolution over its respective algebra:
- RealConv1d/2d: standard nn.Conv1d/2d wrappers
- ComplexConv1d/2d: complex convolution (2 weight tensors)
- QuaternionConv1d/2d: Hamilton product convolution (4 weight tensors)
- OctonionConv1d/2d: full octonionic convolution via structure constants (8 weight tensors)

Tensor layout: [B, channels, algebra_dim, *spatial] for hypercomplex types.
The algebra dimension comes BEFORE spatial dimensions so that standard
F.conv1d/F.conv2d can operate on each component pair.

Parameter counts (no bias):
- RealConv: out_ch * in_ch * kernel_size (1x)
- ComplexConv: 2 * out_ch * in_ch * kernel_size (2x)
- QuaternionConv: 4 * out_ch * in_ch * kernel_size (4x)
- OctonionConv: 8 * out_ch * in_ch * kernel_size (8x)
"""

from __future__ import annotations

from typing import Union

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

_KernelSize = Union[int, tuple[int, ...]]


# ── Real Convolutions ─────────────────────────────────────────────


class RealConv1d(nn.Module):
    """Real-valued 1D convolution wrapping nn.Conv1d.

    Input shape: [B, in_channels, L]
    Output shape: [B, out_channels, L']

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Zero-padding added to input.
        bias: If True, adds a learnable bias.
        dtype: Tensor dtype.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias, dtype=dtype,
        )
        real_init(self.conv.weight.data.view(out_channels, -1), criterion="he")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class RealConv2d(nn.Module):
    """Real-valued 2D convolution wrapping nn.Conv2d.

    Input shape: [B, in_channels, H, W]
    Output shape: [B, out_channels, H', W']
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _KernelSize,
        stride: _KernelSize = 1,
        padding: _KernelSize = 0,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias, dtype=dtype,
        )
        real_init(self.conv.weight.data.view(out_channels, -1), criterion="he")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ── Complex Convolutions ──────────────────────────────────────────


class ComplexConv1d(nn.Module):
    """Complex-valued 1D convolution using the complex product rule.

    Input shape: [B, in_ch, 2, L]
    Output shape: [B, out_ch, 2, L']

    Computes:
      out_r = conv(x_r, W_r) - conv(x_i, W_i)
      out_i = conv(x_r, W_i) + conv(x_i, W_r)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        self.W_r = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, dtype=dtype)
        )
        self.W_i = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, dtype=dtype)
        )

        if bias:
            self.bias_r = nn.Parameter(torch.zeros(out_channels, dtype=dtype))
            self.bias_i = nn.Parameter(torch.zeros(out_channels, dtype=dtype))
        else:
            self.register_parameter("bias_r", None)
            self.register_parameter("bias_i", None)

        # Init: treat spatial dims as part of fan_in
        complex_init(
            self.W_r.data.view(out_channels, -1),
            self.W_i.data.view(out_channels, -1),
            criterion="glorot",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: complex 1D convolution.

        Args:
            x: [B, in_ch, 2, L]

        Returns:
            [B, out_ch, 2, L']
        """
        x_r = x[:, :, 0, :]  # [B, in_ch, L]
        x_i = x[:, :, 1, :]  # [B, in_ch, L]

        out_r = (
            F.conv1d(x_r, self.W_r, stride=self.stride, padding=self.padding)
            - F.conv1d(x_i, self.W_i, stride=self.stride, padding=self.padding)
        )
        out_i = (
            F.conv1d(x_r, self.W_i, stride=self.stride, padding=self.padding)
            + F.conv1d(x_i, self.W_r, stride=self.stride, padding=self.padding)
        )

        if self.bias_r is not None:
            out_r = out_r + self.bias_r.unsqueeze(0).unsqueeze(-1)
            out_i = out_i + self.bias_i.unsqueeze(0).unsqueeze(-1)

        return torch.stack([out_r, out_i], dim=2)  # [B, out_ch, 2, L']


class ComplexConv2d(nn.Module):
    """Complex-valued 2D convolution using the complex product rule.

    Input shape: [B, in_ch, 2, H, W]
    Output shape: [B, out_ch, 2, H', W']
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _KernelSize,
        stride: _KernelSize = 1,
        padding: _KernelSize = 0,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self._kernel_size = kernel_size

        self.W_r = nn.Parameter(
            torch.empty(out_channels, in_channels, *kernel_size, dtype=dtype)
        )
        self.W_i = nn.Parameter(
            torch.empty(out_channels, in_channels, *kernel_size, dtype=dtype)
        )

        if bias:
            self.bias_r = nn.Parameter(torch.zeros(out_channels, dtype=dtype))
            self.bias_i = nn.Parameter(torch.zeros(out_channels, dtype=dtype))
        else:
            self.register_parameter("bias_r", None)
            self.register_parameter("bias_i", None)

        complex_init(
            self.W_r.data.view(out_channels, -1),
            self.W_i.data.view(out_channels, -1),
            criterion="glorot",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: complex 2D convolution.

        Args:
            x: [B, in_ch, 2, H, W]

        Returns:
            [B, out_ch, 2, H', W']
        """
        x_r = x[:, :, 0, :, :]  # [B, in_ch, H, W]
        x_i = x[:, :, 1, :, :]  # [B, in_ch, H, W]

        out_r = (
            F.conv2d(x_r, self.W_r, stride=self.stride, padding=self.padding)
            - F.conv2d(x_i, self.W_i, stride=self.stride, padding=self.padding)
        )
        out_i = (
            F.conv2d(x_r, self.W_i, stride=self.stride, padding=self.padding)
            + F.conv2d(x_i, self.W_r, stride=self.stride, padding=self.padding)
        )

        if self.bias_r is not None:
            out_r = out_r + self.bias_r.view(1, -1, 1, 1)
            out_i = out_i + self.bias_i.view(1, -1, 1, 1)

        return torch.stack([out_r, out_i], dim=2)  # [B, out_ch, 2, H', W']


# ── Quaternion Convolutions ───────────────────────────────────────


class QuaternionConv1d(nn.Module):
    """Quaternion-valued 1D convolution using the Hamilton product.

    Input shape: [B, in_ch, 4, L]
    Output shape: [B, out_ch, 4, L']

    Uses 16-term Hamilton product with sign pattern matching QuaternionLinear.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        self.W_r = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, dtype=dtype)
        )
        self.W_i = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, dtype=dtype)
        )
        self.W_j = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, dtype=dtype)
        )
        self.W_k = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, dtype=dtype)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_channels, 4, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        quaternion_init(
            self.W_r.data.view(out_channels, -1),
            self.W_i.data.view(out_channels, -1),
            self.W_j.data.view(out_channels, -1),
            self.W_k.data.view(out_channels, -1),
            criterion="glorot",
        )

    def _conv(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, w, stride=self.stride, padding=self.padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: quaternion 1D convolution (fused single-kernel).

        Args:
            x: [B, in_ch, 4, L]

        Returns:
            [B, out_ch, 4, L']
        """
        B, in_ch, _, L = x.shape

        # Stack input: [B, in_ch, 4, L] -> [B, 4*in_ch, L]
        x_cat = x.permute(0, 2, 1, 3).reshape(B, 4 * in_ch, L)

        # Hamilton product weight matrix: [4*out_ch, 4*in_ch, K]
        W = torch.cat([
            torch.cat([self.W_r, -self.W_i, -self.W_j, -self.W_k], dim=1),
            torch.cat([self.W_i,  self.W_r, -self.W_k,  self.W_j], dim=1),
            torch.cat([self.W_j,  self.W_k,  self.W_r, -self.W_i], dim=1),
            torch.cat([self.W_k, -self.W_j,  self.W_i,  self.W_r], dim=1),
        ], dim=0)

        out_cat = F.conv1d(x_cat, W, stride=self.stride, padding=self.padding)

        out_ch = self.out_channels
        result = out_cat.reshape(B, 4, out_ch, -1).permute(0, 2, 1, 3)

        if self.bias is not None:
            result = result + self.bias.unsqueeze(0).unsqueeze(-1)

        return result


class QuaternionConv2d(nn.Module):
    """Quaternion-valued 2D convolution using the Hamilton product.

    Input shape: [B, in_ch, 4, H, W]
    Output shape: [B, out_ch, 4, H', W']
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _KernelSize,
        stride: _KernelSize = 1,
        padding: _KernelSize = 0,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self._kernel_size = kernel_size

        self.W_r = nn.Parameter(
            torch.empty(out_channels, in_channels, *kernel_size, dtype=dtype)
        )
        self.W_i = nn.Parameter(
            torch.empty(out_channels, in_channels, *kernel_size, dtype=dtype)
        )
        self.W_j = nn.Parameter(
            torch.empty(out_channels, in_channels, *kernel_size, dtype=dtype)
        )
        self.W_k = nn.Parameter(
            torch.empty(out_channels, in_channels, *kernel_size, dtype=dtype)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_channels, 4, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        quaternion_init(
            self.W_r.data.view(out_channels, -1),
            self.W_i.data.view(out_channels, -1),
            self.W_j.data.view(out_channels, -1),
            self.W_k.data.view(out_channels, -1),
            criterion="glorot",
        )

    def _conv(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, w, stride=self.stride, padding=self.padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: quaternion 2D convolution (fused single-kernel).

        Constructs the Hamilton product weight matrix and performs a single
        F.conv2d instead of 16 separate calls, reducing GPU kernel launch
        overhead by ~16x.

        Args:
            x: [B, in_ch, 4, H, W]

        Returns:
            [B, out_ch, 4, H', W']
        """
        B, in_ch, _, H, W = x.shape

        # Stack input components: [B, in_ch, 4, H, W] -> [B, 4*in_ch, H, W]
        x_cat = x.permute(0, 2, 1, 3, 4).reshape(B, 4 * in_ch, H, W)

        # Build Hamilton product weight matrix: [4*out_ch, 4*in_ch, kH, kW]
        # Rows correspond to output components (r,i,j,k)
        # Columns correspond to input components (r,i,j,k)
        W = torch.cat([
            torch.cat([self.W_r, -self.W_i, -self.W_j, -self.W_k], dim=1),
            torch.cat([self.W_i,  self.W_r, -self.W_k,  self.W_j], dim=1),
            torch.cat([self.W_j,  self.W_k,  self.W_r, -self.W_i], dim=1),
            torch.cat([self.W_k, -self.W_j,  self.W_i,  self.W_r], dim=1),
        ], dim=0)

        # Single fused convolution
        out_cat = F.conv2d(x_cat, W, stride=self.stride, padding=self.padding)

        # Reshape back: [B, 4*out_ch, H', W'] -> [B, 4, out_ch, H', W'] -> [B, out_ch, 4, H', W']
        out_ch = self.out_channels
        result = out_cat.reshape(B, 4, out_ch, *out_cat.shape[2:]).permute(0, 2, 1, 3, 4)

        if self.bias is not None:
            result = result + self.bias.view(1, -1, 4, 1, 1)

        return result


# ── Octonion Convolutions ─────────────────────────────────────────


class OctonionConv1d(nn.Module):
    """Octonion-valued 1D convolution via structure constants.

    Input shape: [B, in_ch, 8, L]
    Output shape: [B, out_ch, 8, L']

    Uses STRUCTURE_CONSTANTS for the full octonionic convolution product
    with precomputed nonzero entries for efficiency.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        self.weights = nn.ParameterList([
            nn.Parameter(
                torch.empty(out_channels, in_channels, kernel_size, dtype=dtype)
            )
            for _ in range(8)
        ])

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_channels, 8, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        octonion_init(
            [w.data.view(out_channels, -1) for w in self.weights],
            criterion="glorot",
        )

        # Precompute nonzero structure constant entries
        C = STRUCTURE_CONSTANTS
        self._nonzero_entries: list[tuple[int, int, int, float]] = []
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    c = C[i, j, k].item()
                    if c != 0.0:
                        self._nonzero_entries.append((i, j, k, c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: octonionic 1D convolution.

        Args:
            x: [B, in_ch, 8, L]

        Returns:
            [B, out_ch, 8, L']
        """
        B = x.shape[0]
        L = x.shape[-1]

        # Compute conv results and accumulate using structure constants
        conv_cache: dict[tuple[int, int], torch.Tensor] = {}

        # Determine output spatial size from a trial convolution
        trial = F.conv1d(
            x[:, :, 0, :], self.weights[0],
            stride=self.stride, padding=self.padding,
        )
        L_out = trial.shape[-1]

        out_components = [
            torch.zeros(B, self.out_channels, L_out, dtype=x.dtype, device=x.device)
            for _ in range(8)
        ]

        for i, j, k, c in self._nonzero_entries:
            key = (i, j)
            if key not in conv_cache:
                conv_cache[key] = F.conv1d(
                    x[:, :, j, :], self.weights[i],
                    stride=self.stride, padding=self.padding,
                )
            out_components[k] = out_components[k] + c * conv_cache[key]

        result = torch.stack(out_components, dim=2)  # [B, out_ch, 8, L']

        if self.bias is not None:
            result = result + self.bias.unsqueeze(0).unsqueeze(-1)

        return result


class OctonionConv2d(nn.Module):
    """Octonion-valued 2D convolution via structure constants.

    Input shape: [B, in_ch, 8, H, W]
    Output shape: [B, out_ch, 8, H', W']
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _KernelSize,
        stride: _KernelSize = 1,
        padding: _KernelSize = 0,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self._kernel_size = kernel_size

        self.weights = nn.ParameterList([
            nn.Parameter(
                torch.empty(
                    out_channels, in_channels, *kernel_size, dtype=dtype
                )
            )
            for _ in range(8)
        ])

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_channels, 8, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        octonion_init(
            [w.data.view(out_channels, -1) for w in self.weights],
            criterion="glorot",
        )

        C = STRUCTURE_CONSTANTS
        self._nonzero_entries: list[tuple[int, int, int, float]] = []
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    c = C[i, j, k].item()
                    if c != 0.0:
                        self._nonzero_entries.append((i, j, k, c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: octonionic 2D convolution.

        Args:
            x: [B, in_ch, 8, H, W]

        Returns:
            [B, out_ch, 8, H', W']
        """
        B = x.shape[0]

        conv_cache: dict[tuple[int, int], torch.Tensor] = {}

        trial = F.conv2d(
            x[:, :, 0, :, :], self.weights[0],
            stride=self.stride, padding=self.padding,
        )
        spatial_shape = trial.shape[2:]  # (H', W')

        out_components = [
            torch.zeros(
                B, self.out_channels, *spatial_shape,
                dtype=x.dtype, device=x.device,
            )
            for _ in range(8)
        ]

        for i, j, k, c in self._nonzero_entries:
            key = (i, j)
            if key not in conv_cache:
                conv_cache[key] = F.conv2d(
                    x[:, :, j, :, :], self.weights[i],
                    stride=self.stride, padding=self.padding,
                )
            out_components[k] = out_components[k] + c * conv_cache[key]

        result = torch.stack(out_components, dim=2)  # [B, out_ch, 8, H', W']

        if self.bias is not None:
            result = result + self.bias.view(1, -1, 8, 1, 1)

        return result
