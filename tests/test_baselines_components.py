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


# ── Convolutional layer output shape tests ─────────────────────────


class TestRealConvShapes:
    """RealConv1d/2d output shapes."""

    def test_real_conv1d_shape(self) -> None:
        from octonion.baselines._algebra_conv import RealConv1d

        layer = RealConv1d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 10)  # [B, C, L]
        out = layer(x)
        assert out.shape == (2, 16, 10)

    def test_real_conv2d_shape(self) -> None:
        from octonion.baselines._algebra_conv import RealConv2d

        layer = RealConv2d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 8, 8)  # [B, C, H, W]
        out = layer(x)
        assert out.shape == (2, 16, 8, 8)


class TestComplexConvShapes:
    """ComplexConv1d/2d output shapes."""

    def test_complex_conv1d_shape(self) -> None:
        from octonion.baselines._algebra_conv import ComplexConv1d

        layer = ComplexConv1d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 2, 10)  # [B, C, 2, L]
        out = layer(x)
        assert out.shape == (2, 16, 2, 10)

    def test_complex_conv2d_shape(self) -> None:
        from octonion.baselines._algebra_conv import ComplexConv2d

        layer = ComplexConv2d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 2, 8, 8)  # [B, C, 2, H, W]
        out = layer(x)
        assert out.shape == (2, 16, 2, 8, 8)


class TestQuaternionConvShapes:
    """QuaternionConv1d/2d output shapes."""

    def test_quaternion_conv1d_shape(self) -> None:
        from octonion.baselines._algebra_conv import QuaternionConv1d

        layer = QuaternionConv1d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 4, 10)  # [B, C, 4, L]
        out = layer(x)
        assert out.shape == (2, 16, 4, 10)

    def test_quaternion_conv2d_shape(self) -> None:
        from octonion.baselines._algebra_conv import QuaternionConv2d

        layer = QuaternionConv2d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 4, 8, 8)  # [B, C, 4, H, W]
        out = layer(x)
        assert out.shape == (2, 16, 4, 8, 8)


class TestOctonionConvShapes:
    """OctonionConv1d/2d output shapes."""

    def test_octonion_conv1d_shape(self) -> None:
        from octonion.baselines._algebra_conv import OctonionConv1d

        layer = OctonionConv1d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 8, 10)  # [B, C, 8, L]
        out = layer(x)
        assert out.shape == (2, 16, 8, 10)

    def test_octonion_conv2d_shape(self) -> None:
        from octonion.baselines._algebra_conv import OctonionConv2d

        layer = OctonionConv2d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 8, 8, 8)  # [B, C, 8, H, W]
        out = layer(x)
        assert out.shape == (2, 16, 8, 8, 8)


# ── Conv parameter count ratio tests ──────────────────────────────


class TestConvParameterRatios:
    """Conv layers should have param counts scaling by algebra dimension."""

    def test_conv2d_param_ratios(self) -> None:
        """C has 2x, H has 4x, O has 8x params vs R for same in/out/kernel."""
        from octonion.baselines._algebra_conv import (
            ComplexConv2d,
            OctonionConv2d,
            QuaternionConv2d,
            RealConv2d,
        )

        in_ch, out_ch, k = 3, 16, 3
        r = RealConv2d(in_ch, out_ch, kernel_size=k, bias=False)
        c = ComplexConv2d(in_ch, out_ch, kernel_size=k, bias=False)
        h = QuaternionConv2d(in_ch, out_ch, kernel_size=k, bias=False)
        o = OctonionConv2d(in_ch, out_ch, kernel_size=k, bias=False)

        r_params = sum(p.numel() for p in r.parameters())
        c_params = sum(p.numel() for p in c.parameters())
        h_params = sum(p.numel() for p in h.parameters())
        o_params = sum(p.numel() for p in o.parameters())

        assert c_params == pytest.approx(2 * r_params, rel=0.01)
        assert h_params == pytest.approx(4 * r_params, rel=0.01)
        assert o_params == pytest.approx(8 * r_params, rel=0.01)

    def test_conv1d_param_ratios(self) -> None:
        """Same ratio test for 1D variants."""
        from octonion.baselines._algebra_conv import (
            ComplexConv1d,
            OctonionConv1d,
            QuaternionConv1d,
            RealConv1d,
        )

        in_ch, out_ch, k = 3, 16, 3
        r = RealConv1d(in_ch, out_ch, kernel_size=k, bias=False)
        c = ComplexConv1d(in_ch, out_ch, kernel_size=k, bias=False)
        h = QuaternionConv1d(in_ch, out_ch, kernel_size=k, bias=False)
        o = OctonionConv1d(in_ch, out_ch, kernel_size=k, bias=False)

        r_params = sum(p.numel() for p in r.parameters())
        c_params = sum(p.numel() for p in c.parameters())
        h_params = sum(p.numel() for p in h.parameters())
        o_params = sum(p.numel() for p in o.parameters())

        assert c_params == pytest.approx(2 * r_params, rel=0.01)
        assert h_params == pytest.approx(4 * r_params, rel=0.01)
        assert o_params == pytest.approx(8 * r_params, rel=0.01)


# ── Conv non-zero output test ─────────────────────────────────────


class TestConvNonZeroOutput:
    """Forward pass should produce non-zero output with random input."""

    @pytest.mark.parametrize(
        "ConvClass,input_shape",
        [
            ("RealConv1d", (2, 3, 10)),
            ("RealConv2d", (2, 3, 8, 8)),
            ("ComplexConv1d", (2, 3, 2, 10)),
            ("ComplexConv2d", (2, 3, 2, 8, 8)),
            ("QuaternionConv1d", (2, 3, 4, 10)),
            ("QuaternionConv2d", (2, 3, 4, 8, 8)),
            ("OctonionConv1d", (2, 3, 8, 10)),
            ("OctonionConv2d", (2, 3, 8, 8, 8)),
        ],
    )
    def test_nonzero_output(
        self, ConvClass: str, input_shape: tuple[int, ...]
    ) -> None:
        import octonion.baselines._algebra_conv as conv_mod

        cls = getattr(conv_mod, ConvClass)
        layer = cls(3, 16, kernel_size=3, padding=1)
        torch.manual_seed(42)
        x = torch.randn(*input_shape)
        with torch.no_grad():
            out = layer(x)
        assert out.abs().max().item() > 0.0


class TestOctonionConvFusedEquivalence:
    """Verify fused OctonionConv produces identical output to unfused reference."""

    @staticmethod
    def _unfused_forward_2d(
        layer: torch.nn.Module, x: torch.Tensor,
    ) -> torch.Tensor:
        """Reference unfused implementation for numerical comparison."""
        from octonion._multiplication import STRUCTURE_CONSTANTS as C

        B = x.shape[0]
        C_local = C.to(device=x.device, dtype=x.dtype)

        nonzero_entries = []
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    c = C_local[i, j, k].item()
                    if c != 0.0:
                        nonzero_entries.append((i, j, k, c))

        conv_cache: dict[tuple[int, int], torch.Tensor] = {}
        trial = torch.nn.functional.conv2d(
            x[:, :, 0, :, :], layer.weights[0],
            stride=layer.stride, padding=layer.padding,
        )
        spatial = trial.shape[2:]

        out_comps = [
            torch.zeros(B, layer.out_channels, *spatial, dtype=x.dtype, device=x.device)
            for _ in range(8)
        ]

        for i, j, k, c in nonzero_entries:
            key = (i, j)
            if key not in conv_cache:
                conv_cache[key] = torch.nn.functional.conv2d(
                    x[:, :, j, :, :], layer.weights[i],
                    stride=layer.stride, padding=layer.padding,
                )
            out_comps[k] = out_comps[k] + c * conv_cache[key]

        result = torch.stack(out_comps, dim=2)
        if layer.bias is not None:
            result = result + layer.bias.view(1, -1, 8, 1, 1)
        return result

    @staticmethod
    def _unfused_forward_1d(
        layer: torch.nn.Module, x: torch.Tensor,
    ) -> torch.Tensor:
        """Reference unfused implementation for 1D."""
        from octonion._multiplication import STRUCTURE_CONSTANTS as C

        B = x.shape[0]
        C_local = C.to(device=x.device, dtype=x.dtype)

        nonzero_entries = []
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    c = C_local[i, j, k].item()
                    if c != 0.0:
                        nonzero_entries.append((i, j, k, c))

        conv_cache: dict[tuple[int, int], torch.Tensor] = {}
        trial = torch.nn.functional.conv1d(
            x[:, :, 0, :], layer.weights[0],
            stride=layer.stride, padding=layer.padding,
        )
        L_out = trial.shape[-1]

        out_comps = [
            torch.zeros(B, layer.out_channels, L_out, dtype=x.dtype, device=x.device)
            for _ in range(8)
        ]

        for i, j, k, c in nonzero_entries:
            key = (i, j)
            if key not in conv_cache:
                conv_cache[key] = torch.nn.functional.conv1d(
                    x[:, :, j, :], layer.weights[i],
                    stride=layer.stride, padding=layer.padding,
                )
            out_comps[k] = out_comps[k] + c * conv_cache[key]

        result = torch.stack(out_comps, dim=2)
        if layer.bias is not None:
            result = result + layer.bias.unsqueeze(0).unsqueeze(-1)
        return result

    def test_octonion_conv2d_fused_matches_unfused(self) -> None:
        """Fused OctonionConv2d output matches unfused reference."""
        from octonion.baselines._algebra_conv import OctonionConv2d

        torch.manual_seed(42)
        layer = OctonionConv2d(3, 8, kernel_size=3, padding=1, bias=True)
        layer.eval()

        x = torch.randn(2, 3, 8, 8, 8)

        with torch.no_grad():
            fused_out = layer(x)
            unfused_out = self._unfused_forward_2d(layer, x)

        torch.testing.assert_close(fused_out, unfused_out, atol=1e-5, rtol=1e-5)

    def test_octonion_conv1d_fused_matches_unfused(self) -> None:
        """Fused OctonionConv1d output matches unfused reference."""
        from octonion.baselines._algebra_conv import OctonionConv1d

        torch.manual_seed(42)
        layer = OctonionConv1d(3, 8, kernel_size=3, padding=1, bias=True)
        layer.eval()

        x = torch.randn(2, 3, 8, 10)

        with torch.no_grad():
            fused_out = layer(x)
            unfused_out = self._unfused_forward_1d(layer, x)

        torch.testing.assert_close(fused_out, unfused_out, atol=1e-5, rtol=1e-5)

    def test_octonion_conv2d_fused_no_bias(self) -> None:
        """Fused OctonionConv2d without bias also matches."""
        from octonion.baselines._algebra_conv import OctonionConv2d

        torch.manual_seed(123)
        layer = OctonionConv2d(4, 6, kernel_size=3, padding=1, bias=False)
        layer.eval()

        x = torch.randn(1, 4, 8, 6, 6)

        with torch.no_grad():
            fused_out = layer(x)
            unfused_out = self._unfused_forward_2d(layer, x)

        torch.testing.assert_close(fused_out, unfused_out, atol=1e-5, rtol=1e-5)
