"""Tests for algebra-aware normalization, activation, and convolutional layers.

Tests follow the behavior specs from 03-02-PLAN.md:
- Parameter count tests for all 4 BN variants
- BN output near zero mean after forward pass (train mode)
- BN running stats update correctly
- SplitActivation zeros negative components, preserves positive
- NormPreservingActivation preserves direction (cosine similarity ~ 1)
- NormPreservingActivation with relu zeros inputs with zero norm
"""

from __future__ import annotations

import pytest
import torch


# ── Normalization parameter count tests ────────────────────────────


class TestRealBatchNormParams:
    """RealBatchNorm wraps nn.BatchNorm1d with 2 params/feature."""

    def test_param_count(self) -> None:
        """RealBatchNorm(32) has 64 learnable params (32 scale + 32 shift)."""
        from octonion.baselines._normalization import RealBatchNorm

        bn = RealBatchNorm(32)
        total = sum(p.numel() for p in bn.parameters())
        assert total == 64


class TestComplexBatchNormParams:
    """ComplexBatchNorm has 5 learnable params per feature."""

    def test_param_count(self) -> None:
        """ComplexBatchNorm(32) has 160 learnable params (32*5)."""
        from octonion.baselines._normalization import ComplexBatchNorm

        bn = ComplexBatchNorm(32)
        total = sum(p.numel() for p in bn.parameters())
        assert total == 160


class TestQuaternionBatchNormParams:
    """QuaternionBatchNorm has 14 learnable params per feature."""

    def test_param_count(self) -> None:
        """QuaternionBatchNorm(32) has 448 learnable params (32*14)."""
        from octonion.baselines._normalization import QuaternionBatchNorm

        bn = QuaternionBatchNorm(32)
        total = sum(p.numel() for p in bn.parameters())
        assert total == 448


class TestOctonionBatchNormParams:
    """OctonionBatchNorm has 44 learnable params per feature."""

    def test_param_count(self) -> None:
        """OctonionBatchNorm(32) has 1408 learnable params (32*44)."""
        from octonion.baselines._normalization import OctonionBatchNorm

        bn = OctonionBatchNorm(32)
        total = sum(p.numel() for p in bn.parameters())
        assert total == 1408


# ── BN output statistics tests ─────────────────────────────────────


class TestBatchNormStatistics:
    """BN layers should produce near-zero-mean output in train mode."""

    def test_real_bn_zero_mean(self) -> None:
        from octonion.baselines._normalization import RealBatchNorm

        bn = RealBatchNorm(16)
        bn.train()
        x = torch.randn(256, 16) * 5.0 + 3.0  # non-zero mean, non-unit var
        out = bn(x)
        assert out.mean().abs().item() < 0.1

    def test_complex_bn_zero_mean(self) -> None:
        from octonion.baselines._normalization import ComplexBatchNorm

        bn = ComplexBatchNorm(16)
        bn.train()
        x = torch.randn(256, 16, 2) * 5.0 + 3.0
        out = bn(x)
        assert out.mean().abs().item() < 0.1

    def test_quaternion_bn_zero_mean(self) -> None:
        from octonion.baselines._normalization import QuaternionBatchNorm

        bn = QuaternionBatchNorm(16)
        bn.train()
        x = torch.randn(256, 16, 4) * 5.0 + 3.0
        out = bn(x)
        assert out.mean().abs().item() < 0.1

    def test_octonion_bn_zero_mean(self) -> None:
        from octonion.baselines._normalization import OctonionBatchNorm

        bn = OctonionBatchNorm(16)
        bn.train()
        x = torch.randn(256, 16, 8) * 5.0 + 3.0
        out = bn(x)
        assert out.mean().abs().item() < 0.1


class TestBatchNormRunningStats:
    """BN running stats should update during training."""

    def test_real_bn_running_stats(self) -> None:
        from octonion.baselines._normalization import RealBatchNorm

        bn = RealBatchNorm(8)
        bn.train()
        x = torch.randn(64, 8) + 5.0
        bn(x)
        # running_mean should have moved toward 5.0
        assert bn.bn.running_mean.mean().item() > 0.1

    def test_complex_bn_running_stats(self) -> None:
        from octonion.baselines._normalization import ComplexBatchNorm

        bn = ComplexBatchNorm(8)
        bn.train()
        x = torch.randn(64, 8, 2) + 5.0
        bn(x)
        # running_mean should have moved toward 5.0
        assert bn.running_mean.abs().mean().item() > 0.1

    def test_quaternion_bn_running_stats(self) -> None:
        from octonion.baselines._normalization import QuaternionBatchNorm

        bn = QuaternionBatchNorm(8)
        bn.train()
        x = torch.randn(64, 8, 4) + 5.0
        bn(x)
        assert bn.running_mean.abs().mean().item() > 0.1

    def test_octonion_bn_running_stats(self) -> None:
        from octonion.baselines._normalization import OctonionBatchNorm

        bn = OctonionBatchNorm(8)
        bn.train()
        x = torch.randn(64, 8, 8) + 5.0
        bn(x)
        assert bn.running_mean.abs().mean().item() > 0.1


# ── Activation tests ──────────────────────────────────────────────


class TestSplitActivation:
    """SplitActivation applies function independently to each component."""

    def test_relu_zeros_negative(self) -> None:
        from octonion.baselines._activation import SplitActivation

        act = SplitActivation("relu")
        x = torch.tensor([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0])
        out = act(x)
        expected = torch.tensor([0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0])
        torch.testing.assert_close(out, expected)

    def test_relu_preserves_positive(self) -> None:
        from octonion.baselines._activation import SplitActivation

        act = SplitActivation("relu")
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = act(x)
        torch.testing.assert_close(out, x)

    def test_gelu_supported(self) -> None:
        from octonion.baselines._activation import SplitActivation

        act = SplitActivation("gelu")
        x = torch.randn(4, 8)
        out = act(x)
        assert out.shape == x.shape

    def test_batch_shape_preserved(self) -> None:
        from octonion.baselines._activation import SplitActivation

        act = SplitActivation("relu")
        x = torch.randn(3, 16, 8)
        out = act(x)
        assert out.shape == (3, 16, 8)


class TestNormPreservingActivation:
    """NormPreservingActivation applies function to norm, preserves direction."""

    def test_preserves_direction(self) -> None:
        """Output should have same direction as input (cosine similarity ~ 1)."""
        from octonion.baselines._activation import NormPreservingActivation

        act = NormPreservingActivation("relu")
        # Use positive-norm inputs (all components contribute positively to norm)
        x = torch.randn(100, 8).abs() + 0.1  # ensure positive
        out = act(x)
        # Cosine similarity along last dimension
        cos_sim = torch.nn.functional.cosine_similarity(x, out, dim=-1)
        assert cos_sim.mean().item() > 0.99

    def test_relu_zeros_zero_norm(self) -> None:
        """Zero input should produce zero output."""
        from octonion.baselines._activation import NormPreservingActivation

        act = NormPreservingActivation("relu")
        x = torch.zeros(4, 8)
        out = act(x)
        assert out.abs().max().item() < 1e-6

    def test_batch_shape_preserved(self) -> None:
        from octonion.baselines._activation import NormPreservingActivation

        act = NormPreservingActivation("relu")
        x = torch.randn(3, 16, 4)
        out = act(x)
        assert out.shape == (3, 16, 4)

    def test_norm_activation_applied(self) -> None:
        """ReLU on norm: positive norms pass through, magnitude preserved."""
        from octonion.baselines._activation import NormPreservingActivation

        act = NormPreservingActivation("relu")
        # All positive components -> positive norm -> ReLU preserves
        x = torch.ones(1, 4) * 2.0
        out = act(x)
        # Norm should be preserved since relu(positive_norm) = positive_norm
        in_norm = x.norm(dim=-1)
        out_norm = out.norm(dim=-1)
        torch.testing.assert_close(in_norm, out_norm, atol=1e-5, rtol=1e-5)
