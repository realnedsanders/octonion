"""Tests for R8 Dense Mixing linear layer (no algebra structure baseline).

Verifies forward/backward pass shapes, parameter counts, gradient flow,
and integration with parameter matching infrastructure.
"""

from __future__ import annotations

import torch

from octonion.baselines._config import AlgebraType
from octonion.baselines._dense_mixing import DenseMixingLinear
from octonion.baselines._param_matching import find_matched_width


class TestDenseMixingLinearShapes:
    """Forward pass produces correct output shapes."""

    def test_basic_shape(self) -> None:
        layer = DenseMixingLinear(16, 32, bias=False)
        x = torch.randn(4, 16, 8)
        out = layer(x)
        assert out.shape == (4, 32, 8)

    def test_shape_with_bias(self) -> None:
        layer = DenseMixingLinear(16, 32, bias=True)
        x = torch.randn(4, 16, 8)
        out = layer(x)
        assert out.shape == (4, 32, 8)

    def test_single_sample(self) -> None:
        layer = DenseMixingLinear(8, 4, bias=False)
        x = torch.randn(1, 8, 8)
        out = layer(x)
        assert out.shape == (1, 4, 8)

    def test_higher_dim_batch(self) -> None:
        layer = DenseMixingLinear(8, 4, bias=False)
        x = torch.randn(2, 3, 8, 8)
        out = layer(x)
        assert out.shape == (2, 3, 4, 8)


class TestDenseMixingLinearParamCounts:
    """Parameter counts match theoretical values.

    DenseMixingLinear: Full nn.Linear(in_f * 8, out_f * 8)
    - weight: (out_f * 8) * (in_f * 8) params
    - bias: out_f * 8 params (if enabled)
    """

    def test_param_count_no_bias(self) -> None:
        layer = DenseMixingLinear(16, 16, bias=False)
        total = sum(p.numel() for p in layer.parameters())
        expected = 128 * 128  # 16384
        assert total == expected, f"Expected {expected}, got {total}"

    def test_param_count_with_bias(self) -> None:
        layer = DenseMixingLinear(16, 16, bias=True)
        total = sum(p.numel() for p in layer.parameters())
        expected = 128 * 128 + 128  # 16512
        assert total == expected, f"Expected {expected}, got {total}"

    def test_param_count_asymmetric(self) -> None:
        layer = DenseMixingLinear(32, 64, bias=False)
        total = sum(p.numel() for p in layer.parameters())
        expected = (64 * 8) * (32 * 8)  # 512 * 256 = 131072
        assert total == expected, f"Expected {expected}, got {total}"


class TestDenseMixingLinearGradients:
    """Backward pass produces non-None gradients for all parameters."""

    def test_backward_no_bias(self) -> None:
        layer = DenseMixingLinear(16, 16, bias=False)
        x = torch.randn(4, 16, 8, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        for name, p in layer.named_parameters():
            assert p.grad is not None, f"Gradient is None for {name}"
            assert p.grad.shape == p.shape, f"Gradient shape mismatch for {name}"

    def test_backward_with_bias(self) -> None:
        layer = DenseMixingLinear(16, 16, bias=True)
        x = torch.randn(4, 16, 8, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        for name, p in layer.named_parameters():
            assert p.grad is not None, f"Gradient is None for {name}"

    def test_input_grad(self) -> None:
        layer = DenseMixingLinear(8, 4, bias=False)
        x = torch.randn(2, 8, 8, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestR8DenseAlgebraType:
    """AlgebraType.R8_DENSE has correct properties."""

    def test_dim(self) -> None:
        assert AlgebraType.R8_DENSE.dim == 8

    def test_short_name(self) -> None:
        assert AlgebraType.R8_DENSE.short_name == "R8D"

    def test_multiplier(self) -> None:
        assert AlgebraType.R8_DENSE.multiplier == 1


class TestR8DenseParamMatching:
    """R8_DENSE works with find_matched_width binary search."""

    def test_find_matched_width_r8_dense(self) -> None:
        from octonion.baselines._param_matching import _build_simple_mlp

        ref = _build_simple_mlp(
            algebra=AlgebraType.OCTONION,
            hidden=25,
            depth=3,
            input_dim=32,
            output_dim=10,
        )
        target = sum(p.numel() for p in ref.parameters())

        width = find_matched_width(
            target_params=target,
            algebra=AlgebraType.R8_DENSE,
            topology="mlp",
            depth=3,
            input_dim=32,
            output_dim=10,
        )
        assert isinstance(width, int)
        assert width > 0
