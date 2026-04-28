"""Tests for PHM-8 (Parameterized Hypercomplex Multiplication) linear layer.

Verifies forward/backward pass shapes, parameter counts, gradient flow,
and integration with parameter matching infrastructure.
"""

from __future__ import annotations

import torch

from octonion.baselines._config import AlgebraType
from octonion.baselines._param_matching import find_matched_width
from octonion.baselines._phm_linear import PHM8Linear


class TestPHM8LinearShapes:
    """Forward pass produces correct output shapes."""

    def test_basic_shape(self) -> None:
        layer = PHM8Linear(16, 32, bias=False)
        x = torch.randn(4, 16, 8)
        out = layer(x)
        assert out.shape == (4, 32, 8)

    def test_shape_with_bias(self) -> None:
        layer = PHM8Linear(16, 32, bias=True)
        x = torch.randn(4, 16, 8)
        out = layer(x)
        assert out.shape == (4, 32, 8)

    def test_single_sample(self) -> None:
        layer = PHM8Linear(8, 4, bias=False)
        x = torch.randn(1, 8, 8)
        out = layer(x)
        assert out.shape == (1, 4, 8)

    def test_higher_dim_batch(self) -> None:
        layer = PHM8Linear(8, 4, bias=False)
        x = torch.randn(2, 3, 8, 8)
        out = layer(x)
        assert out.shape == (2, 3, 4, 8)


class TestPHM8LinearParamCounts:
    """Parameter counts match theoretical values.

    PHM-8: H = sum_{i=0}^{n-1} kron(A_i, S_i)
    - A: n mixing matrices of shape (n, n) => n*n*n = 512 params for n=8
    - S: n sub-matrices of shape (out_f, in_f) => n * out_f * in_f params
    - bias: out_f * 8 params (if enabled)
    """

    def test_param_count_no_bias(self) -> None:
        layer = PHM8Linear(16, 16, bias=False)
        total = sum(p.numel() for p in layer.parameters())
        # A_i: 8*8*8 = 512, S_i: 8*16*16 = 2048 => total = 2560
        assert total == 512 + 8 * 16 * 16, f"Expected 2560, got {total}"

    def test_param_count_with_bias(self) -> None:
        layer = PHM8Linear(16, 16, bias=True)
        total = sum(p.numel() for p in layer.parameters())
        # 512 + 8*16*16 + 16*8 = 512 + 2048 + 128 = 2688
        expected = 512 + 8 * 16 * 16 + 16 * 8
        assert total == expected, f"Expected {expected}, got {total}"

    def test_param_count_asymmetric(self) -> None:
        layer = PHM8Linear(32, 64, bias=False)
        total = sum(p.numel() for p in layer.parameters())
        expected = 512 + 8 * 64 * 32
        assert total == expected, f"Expected {expected}, got {total}"


class TestPHM8LinearGradients:
    """Backward pass produces non-None gradients for all parameters."""

    def test_backward_no_bias(self) -> None:
        layer = PHM8Linear(16, 16, bias=False)
        x = torch.randn(4, 16, 8, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        for name, p in layer.named_parameters():
            assert p.grad is not None, f"Gradient is None for {name}"
            assert p.grad.shape == p.shape, f"Gradient shape mismatch for {name}"

    def test_backward_with_bias(self) -> None:
        layer = PHM8Linear(16, 16, bias=True)
        x = torch.randn(4, 16, 8, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        for name, p in layer.named_parameters():
            assert p.grad is not None, f"Gradient is None for {name}"

    def test_input_grad(self) -> None:
        layer = PHM8Linear(8, 4, bias=False)
        x = torch.randn(2, 8, 8, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestPHM8AlgebraType:
    """AlgebraType.PHM8 has correct properties."""

    def test_dim(self) -> None:
        assert AlgebraType.PHM8.dim == 8

    def test_short_name(self) -> None:
        assert AlgebraType.PHM8.short_name == "PHM8"

    def test_multiplier(self) -> None:
        assert AlgebraType.PHM8.multiplier == 1


class TestPHM8ParamMatching:
    """PHM8 works with find_matched_width binary search."""

    def test_find_matched_width_phm8(self) -> None:
        # Build a reference model to get target param count
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
            algebra=AlgebraType.PHM8,
            topology="mlp",
            depth=3,
            input_dim=32,
            output_dim=10,
        )
        assert isinstance(width, int)
        assert width > 0
