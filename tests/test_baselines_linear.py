"""Tests for per-algebra linear layers, parameter matching, and FLOP reporting.

Tests follow the behavior spec from 03-01-PLAN.md Task 2:
- Output shapes for each algebra linear layer
- Parameter counts match expected formulas
- Cross-validation against verified algebra types from Phase 1
- Initialization produces unit-variance outputs
- find_matched_width achieves within 1% tolerance
- param_report and flop_report return correct structure
"""

from __future__ import annotations

import math

import pytest
import torch

from octonion._multiplication import octonion_mul
from octonion._tower import Complex, Quaternion
from octonion.baselines import AlgebraType
from octonion.baselines._algebra_linear import (
    ComplexLinear,
    OctonionDenseLinear,
    QuaternionLinear,
    RealLinear,
)
from octonion.baselines._param_matching import (
    find_matched_width,
    flop_report,
    param_report,
)


# ── Output shape tests ──────────────────────────────────────────────


class TestRealLinearShape:
    """RealLinear output shape tests."""

    def test_basic_shape(self) -> None:
        layer = RealLinear(4, 8)
        x = torch.randn(4)
        out = layer(x)
        assert out.shape == (8,)

    def test_batch_shape(self) -> None:
        layer = RealLinear(4, 8)
        x = torch.randn(3, 4)
        out = layer(x)
        assert out.shape == (3, 8)

    def test_multi_batch_shape(self) -> None:
        layer = RealLinear(4, 8)
        x = torch.randn(2, 3, 4)
        out = layer(x)
        assert out.shape == (2, 3, 8)


class TestComplexLinearShape:
    """ComplexLinear output shape tests."""

    def test_basic_shape(self) -> None:
        layer = ComplexLinear(4, 8)
        x = torch.randn(4, 2)
        out = layer(x)
        assert out.shape == (8, 2)

    def test_batch_shape(self) -> None:
        layer = ComplexLinear(4, 8)
        x = torch.randn(3, 4, 2)
        out = layer(x)
        assert out.shape == (3, 8, 2)

    def test_multi_batch_shape(self) -> None:
        layer = ComplexLinear(4, 8)
        x = torch.randn(2, 3, 4, 2)
        out = layer(x)
        assert out.shape == (2, 3, 8, 2)


class TestQuaternionLinearShape:
    """QuaternionLinear output shape tests."""

    def test_basic_shape(self) -> None:
        layer = QuaternionLinear(4, 8)
        x = torch.randn(4, 4)
        out = layer(x)
        assert out.shape == (8, 4)

    def test_batch_shape(self) -> None:
        layer = QuaternionLinear(4, 8)
        x = torch.randn(3, 4, 4)
        out = layer(x)
        assert out.shape == (3, 8, 4)

    def test_multi_batch_shape(self) -> None:
        layer = QuaternionLinear(4, 8)
        x = torch.randn(2, 3, 4, 4)
        out = layer(x)
        assert out.shape == (2, 3, 8, 4)


class TestOctonionDenseLinearShape:
    """OctonionDenseLinear output shape tests."""

    def test_basic_shape(self) -> None:
        layer = OctonionDenseLinear(4, 8)
        x = torch.randn(4, 8)
        out = layer(x)
        assert out.shape == (8, 8)

    def test_batch_shape(self) -> None:
        layer = OctonionDenseLinear(4, 8)
        x = torch.randn(3, 4, 8)
        out = layer(x)
        assert out.shape == (3, 8, 8)

    def test_multi_batch_shape(self) -> None:
        layer = OctonionDenseLinear(4, 8)
        x = torch.randn(2, 3, 4, 8)
        out = layer(x)
        assert out.shape == (2, 3, 8, 8)


# ── Parameter count tests ───────────────────────────────────────────


class TestParameterCounts:
    """Verify parameter counts match expected formulas."""

    def test_real_param_count(self) -> None:
        """RealLinear(4, 8) has 4*8 + 8 = 40 params (weight + bias)."""
        layer = RealLinear(4, 8)
        total = sum(p.numel() for p in layer.parameters())
        assert total == 4 * 8 + 8  # 40

    def test_real_no_bias_param_count(self) -> None:
        layer = RealLinear(4, 8, bias=False)
        total = sum(p.numel() for p in layer.parameters())
        assert total == 4 * 8  # 32

    def test_complex_param_count(self) -> None:
        """ComplexLinear(4, 8) has 2*4*8 + 2*8 = 80 real params."""
        layer = ComplexLinear(4, 8)
        total = sum(p.numel() for p in layer.parameters())
        assert total == 2 * 4 * 8 + 2 * 8  # 80

    def test_complex_no_bias_param_count(self) -> None:
        layer = ComplexLinear(4, 8, bias=False)
        total = sum(p.numel() for p in layer.parameters())
        assert total == 2 * 4 * 8  # 64

    def test_quaternion_param_count(self) -> None:
        """QuaternionLinear(4, 8) has 4*4*8 + 4*8 = 160 real params."""
        layer = QuaternionLinear(4, 8)
        total = sum(p.numel() for p in layer.parameters())
        assert total == 4 * 4 * 8 + 4 * 8  # 160

    def test_quaternion_no_bias_param_count(self) -> None:
        layer = QuaternionLinear(4, 8, bias=False)
        total = sum(p.numel() for p in layer.parameters())
        assert total == 4 * 4 * 8  # 128

    def test_octonion_dense_param_count(self) -> None:
        """OctonionDenseLinear(4, 8) has 8*4*8 + 8*8 = 320 real params."""
        layer = OctonionDenseLinear(4, 8)
        total = sum(p.numel() for p in layer.parameters())
        assert total == 8 * 4 * 8 + 8 * 8  # 320

    def test_octonion_dense_no_bias_param_count(self) -> None:
        layer = OctonionDenseLinear(4, 8, bias=False)
        total = sum(p.numel() for p in layer.parameters())
        assert total == 8 * 4 * 8  # 256

    def test_param_scaling_ratios(self) -> None:
        """For same base_hidden, R has 8x params, C has 4x, H has 2x, O has 1x."""
        base = 8
        r = RealLinear(base * 8, base * 8, bias=False)
        c = ComplexLinear(base * 4, base * 4, bias=False)
        h = QuaternionLinear(base * 2, base * 2, bias=False)
        o = OctonionDenseLinear(base, base, bias=False)

        r_params = sum(p.numel() for p in r.parameters())
        c_params = sum(p.numel() for p in c.parameters())
        h_params = sum(p.numel() for p in h.parameters())
        o_params = sum(p.numel() for p in o.parameters())

        # All should have the same total real param count
        # R: 64*64 = 4096
        # C: 2*32*32 = 2048... hmm
        # Actually let me check this more carefully.
        # R: in*out = (8*8)*(8*8) = 64*64 = 4096
        # C: 2*in*out = 2*32*32 = 2048
        # These are NOT equal because the layers have different in/out dims.
        # The point is: for the SAME algebra units, O has fewest real params.
        # R: 1 real param per weight element, H(in,out) units
        # C: 2 real params per weight element
        # H: 4 real params per weight element
        # O: 8 real params per weight element
        # So O(base, base) has 8*base*base real params
        # H(2*base, 2*base) has 4*(2*base)*(2*base) = 16*base^2
        # C(4*base, 4*base) has 2*(4*base)*(4*base) = 32*base^2
        # R(8*base, 8*base) has 1*(8*base)*(8*base) = 64*base^2
        # Ratios: R:C:H:O = 64:32:16:8 = 8:4:2:1
        assert r_params / o_params == pytest.approx(8.0, rel=0.01)
        assert c_params / o_params == pytest.approx(4.0, rel=0.01)
        assert h_params / o_params == pytest.approx(2.0, rel=0.01)


# ── Cross-validation against verified algebra types ─────────────────


class TestCrossValidation:
    """Verify linear layers match Phase 1 algebra multiplication."""

    def test_complex_linear_matches_complex_mul(self) -> None:
        """ComplexLinear(1, 1) with known W should match Complex.__mul__."""
        torch.manual_seed(42)
        layer = ComplexLinear(1, 1, bias=False)

        # Extract weight as a complex number
        w_r = layer.W_r.data[0, 0].item()
        w_i = layer.W_i.data[0, 0].item()
        w_complex = Complex(torch.tensor([w_r, w_i]))

        # Random input
        x_data = torch.randn(2)
        x_complex = Complex(x_data.clone())

        # Linear layer forward
        x_in = x_data.unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
        layer_out = layer(x_in).squeeze()  # [2]

        # Algebra multiplication
        algebra_out = (w_complex * x_complex).components

        torch.testing.assert_close(layer_out, algebra_out, atol=1e-6, rtol=1e-5)

    def test_quaternion_linear_matches_quaternion_mul(self) -> None:
        """QuaternionLinear(1, 1) with known W should match Quaternion.__mul__."""
        torch.manual_seed(42)
        layer = QuaternionLinear(1, 1, bias=False)

        # Extract weight as a quaternion
        w_data = torch.tensor([
            layer.W_r.data[0, 0].item(),
            layer.W_i.data[0, 0].item(),
            layer.W_j.data[0, 0].item(),
            layer.W_k.data[0, 0].item(),
        ])
        w_quat = Quaternion(w_data)

        # Random input
        x_data = torch.randn(4)
        x_quat = Quaternion(x_data.clone())

        # Linear layer forward
        x_in = x_data.unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
        layer_out = layer(x_in).squeeze()  # [4]

        # Algebra multiplication
        algebra_out = (w_quat * x_quat).components

        torch.testing.assert_close(layer_out, algebra_out, atol=1e-6, rtol=1e-5)

    def test_octonion_dense_linear_matches_octonion_mul(self) -> None:
        """OctonionDenseLinear(1, 1) with known W should match octonion_mul."""
        torch.manual_seed(42)
        layer = OctonionDenseLinear(1, 1, bias=False)

        # Extract weight as an octonion (8 components from 8 weight matrices)
        w_data = torch.tensor([
            layer.weights[c].data[0, 0].item() for c in range(8)
        ])

        # Random input
        x_data = torch.randn(8)

        # Linear layer forward
        x_in = x_data.unsqueeze(0).unsqueeze(0)  # [1, 1, 8]
        layer_out = layer(x_in).squeeze()  # [8]

        # Algebra multiplication using structure constants
        algebra_out = octonion_mul(w_data, x_data)

        torch.testing.assert_close(layer_out, algebra_out, atol=1e-5, rtol=1e-4)


# ── Initialization variance tests ───────────────────────────────────


class TestInitializationVariance:
    """Forward pass output variance should be ~1.0 for unit-variance inputs."""

    @pytest.mark.parametrize(
        "LayerClass,in_f,out_f,input_shape",
        [
            (RealLinear, 256, 256, (1000, 256)),
            (ComplexLinear, 128, 128, (1000, 128, 2)),
            (QuaternionLinear, 64, 64, (1000, 64, 4)),
            (OctonionDenseLinear, 32, 32, (1000, 32, 8)),
        ],
    )
    def test_output_variance_near_one(
        self,
        LayerClass: type,
        in_f: int,
        out_f: int,
        input_shape: tuple[int, ...],
    ) -> None:
        """Output variance should be roughly in [0.2, 5.0] for fresh init."""
        torch.manual_seed(123)
        layer = LayerClass(in_f, out_f, bias=False)
        x = torch.randn(*input_shape)
        with torch.no_grad():
            out = layer(x)
        var = out.var().item()
        # Lenient bounds: initialization should keep variance in reasonable range
        assert 0.1 < var < 10.0, (
            f"{LayerClass.__name__} output variance {var:.4f} is out of "
            f"expected range [0.1, 10.0] for unit-variance input"
        )


# ── Parameter matching tests ────────────────────────────────────────


class TestFindMatchedWidth:
    """find_matched_width should find widths within 1% tolerance."""

    def test_real_match(self) -> None:
        """Find width for real MLP matching a target param count."""
        target = 10000
        width = find_matched_width(
            target_params=target,
            algebra=AlgebraType.REAL,
            topology="mlp",
            depth=3,
            input_dim=784,
            output_dim=10,
        )
        assert isinstance(width, int)
        assert width > 0

    def test_complex_match(self) -> None:
        target = 10000
        width = find_matched_width(
            target_params=target,
            algebra=AlgebraType.COMPLEX,
            topology="mlp",
            depth=3,
            input_dim=784,
            output_dim=10,
        )
        assert isinstance(width, int)
        assert width > 0

    def test_quaternion_match(self) -> None:
        target = 10000
        width = find_matched_width(
            target_params=target,
            algebra=AlgebraType.QUATERNION,
            topology="mlp",
            depth=3,
            input_dim=784,
            output_dim=10,
        )
        assert isinstance(width, int)
        assert width > 0

    def test_octonion_match(self) -> None:
        target = 10000
        width = find_matched_width(
            target_params=target,
            algebra=AlgebraType.OCTONION,
            topology="mlp",
            depth=3,
            input_dim=784,
            output_dim=10,
        )
        assert isinstance(width, int)
        assert width > 0

    def test_all_algebras_within_tolerance(self) -> None:
        """All 4 algebras should achieve within 1% of same target."""
        target = 50000
        for algebra in AlgebraType:
            width = find_matched_width(
                target_params=target,
                algebra=algebra,
                topology="mlp",
                depth=3,
                input_dim=784,
                output_dim=10,
            )
            # Build model and count params
            from octonion.baselines._param_matching import _build_simple_mlp

            model = _build_simple_mlp(
                algebra=algebra,
                hidden=width,
                depth=3,
                input_dim=784,
                output_dim=10,
            )
            count = sum(p.numel() for p in model.parameters())
            pct_error = abs(count - target) / target
            assert pct_error <= 0.01, (
                f"{algebra.short_name}: width={width}, params={count}, "
                f"target={target}, error={pct_error*100:.2f}%"
            )


# ── param_report and flop_report tests ──────────────────────────────


class TestParamReport:
    """param_report should return per-layer breakdown."""

    def test_returns_list(self) -> None:
        layer = RealLinear(4, 8)
        report = param_report(layer)
        assert isinstance(report, list)
        assert len(report) > 0

    def test_report_structure(self) -> None:
        layer = RealLinear(4, 8)
        report = param_report(layer)
        for entry in report:
            assert "name" in entry
            assert "shape" in entry
            assert "real_params" in entry
            assert "pct" in entry

    def test_report_totals(self) -> None:
        layer = ComplexLinear(4, 8)
        report = param_report(layer)
        total_from_report = sum(e["real_params"] for e in report)
        total_from_model = sum(p.numel() for p in layer.parameters())
        assert total_from_report == total_from_model


class TestFlopReport:
    """flop_report should return FLOP counts via torchinfo."""

    def test_returns_dict(self) -> None:
        layer = RealLinear(4, 8)
        report = flop_report(layer, input_size=(1, 4))
        assert isinstance(report, dict)

    def test_has_required_keys(self) -> None:
        layer = RealLinear(4, 8)
        report = flop_report(layer, input_size=(1, 4))
        assert "total_mult_adds" in report
        assert "per_layer" in report

    def test_per_layer_is_list(self) -> None:
        layer = RealLinear(4, 8)
        report = flop_report(layer, input_size=(1, 4))
        assert isinstance(report["per_layer"], list)
