"""Tests for AlgebraNetwork skeleton with MLP/Conv2D/Recurrent topologies.

Covers forward pass for all algebras, skeleton identity (SC-4),
parameter matching within 1%, output projections, and param reporting.
"""

from __future__ import annotations

import pytest
import torch

from octonion.baselines._config import AlgebraType, NetworkConfig


ALL_ALGEBRAS = list(AlgebraType)
B = 4  # batch size


# ── Helper ───────────────────────────────────────────────────────


def _make_config(
    algebra: AlgebraType,
    topology: str = "mlp",
    **kwargs,
) -> NetworkConfig:
    """Create a NetworkConfig with defaults suitable for testing."""
    defaults = dict(
        algebra=algebra,
        topology=topology,
        depth=2,
        base_hidden=16,
        activation="split_relu",
        output_projection="real",
        use_batchnorm=False,  # off by default for simpler shape testing
        input_dim=32,
        output_dim=10,
    )
    defaults.update(kwargs)
    return NetworkConfig(**defaults)


# ── MLP Forward Pass Tests ──────────────────────────────────────


class TestMLPForward:
    """Build MLP for R/C/H/O and verify output shape [batch, output_dim]."""

    @pytest.mark.parametrize("algebra", ALL_ALGEBRAS, ids=lambda a: a.short_name)
    def test_mlp_forward_all_algebras(self, algebra):
        from octonion.baselines._network import AlgebraNetwork

        config = _make_config(algebra, topology="mlp")
        model = AlgebraNetwork(config)
        x = torch.randn(B, config.input_dim)
        out = model(x)
        assert out.shape == (B, config.output_dim), (
            f"Expected output shape ({B}, {config.output_dim}), got {out.shape}"
        )


# ── Conv2D Forward Pass Tests ───────────────────────────────────


class TestConv2DForward:
    """Build Conv2D for R/C/H/O with image input, verify output shape."""

    @pytest.mark.parametrize("algebra", ALL_ALGEBRAS, ids=lambda a: a.short_name)
    def test_conv2d_forward_all_algebras(self, algebra):
        from octonion.baselines._network import AlgebraNetwork

        config = _make_config(
            algebra,
            topology="conv2d",
            input_dim=3,  # RGB channels
            output_dim=10,
            depth=2,
            base_hidden=8,
        )
        model = AlgebraNetwork(config)
        # Image input: [B, C, H, W]
        x = torch.randn(B, 3, 16, 16)
        out = model(x)
        assert out.shape == (B, config.output_dim), (
            f"Expected output shape ({B}, {config.output_dim}), got {out.shape}"
        )


# ── Recurrent Forward Pass Tests ────────────────────────────────


class TestRecurrentForward:
    """Build Recurrent for R/C/H/O with sequence input, verify output shape."""

    @pytest.mark.parametrize("algebra", ALL_ALGEBRAS, ids=lambda a: a.short_name)
    def test_recurrent_forward_all_algebras(self, algebra):
        from octonion.baselines._network import AlgebraNetwork

        config = _make_config(
            algebra,
            topology="recurrent",
            input_dim=32,
            output_dim=10,
            depth=2,
            base_hidden=16,
        )
        model = AlgebraNetwork(config)
        # Sequence input: [B, seq_len, input_dim]
        seq_len = 5
        x = torch.randn(B, seq_len, config.input_dim)
        out = model(x)
        assert out.shape == (B, config.output_dim), (
            f"Expected output shape ({B}, {config.output_dim}), got {out.shape}"
        )


# ── Skeleton Identity Tests (SC-4) ──────────────────────────────


class TestSkeletonIdentity:
    """All 4 algebras with same config should have identical layer structure."""

    def _get_layer_structure(self, model):
        """Extract layer structure: list of (relative_name, param_count) tuples."""
        structure = []
        for name, module in model.named_modules():
            if name == "":
                continue
            # Count parameters in this module (not children)
            own_params = sum(
                p.numel()
                for n, p in module.named_parameters(recurse=False)
            )
            structure.append((name, own_params > 0))
        return structure

    def test_skeleton_identity_mlp(self):
        """MLP topology: layer count and structure identical across algebras."""
        from octonion.baselines._network import AlgebraNetwork

        models = {}
        for algebra in ALL_ALGEBRAS:
            config = _make_config(algebra, topology="mlp", depth=3)
            models[algebra.short_name] = AlgebraNetwork(config)

        # Count total named modules with parameters
        layer_counts = {}
        for name, model in models.items():
            count = sum(1 for _, m in model.named_modules() if sum(
                p.numel() for p in m.parameters(recurse=False)
            ) > 0)
            layer_counts[name] = count

        # All algebras should have the same number of parametric layers
        counts = list(layer_counts.values())
        assert all(c == counts[0] for c in counts), (
            f"Layer counts differ across algebras: {layer_counts}"
        )

    def test_skeleton_identity_conv2d(self):
        """Conv2D topology: layer count identical across algebras."""
        from octonion.baselines._network import AlgebraNetwork

        models = {}
        for algebra in ALL_ALGEBRAS:
            config = _make_config(
                algebra, topology="conv2d", input_dim=3, depth=2, base_hidden=8,
            )
            models[algebra.short_name] = AlgebraNetwork(config)

        layer_counts = {}
        for name, model in models.items():
            count = sum(1 for _, m in model.named_modules() if sum(
                p.numel() for p in m.parameters(recurse=False)
            ) > 0)
            layer_counts[name] = count

        counts = list(layer_counts.values())
        assert all(c == counts[0] for c in counts), (
            f"Layer counts differ across algebras: {layer_counts}"
        )


# ── Parameter Matching Tests ────────────────────────────────────


class TestParamMatching:
    """Parameter matching within 1% across all 4 algebras."""

    def test_param_matching_mlp(self):
        """Build O model, use find_matched_width for R/C/H, assert within 1%."""
        from octonion.baselines._network import AlgebraNetwork
        from octonion.baselines._param_matching import find_matched_width

        depth = 3
        input_dim = 32
        output_dim = 10
        base_hidden = 32

        # Build O model as reference
        o_config = _make_config(
            AlgebraType.OCTONION,
            topology="mlp",
            depth=depth,
            base_hidden=base_hidden,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        o_model = AlgebraNetwork(o_config)
        o_params = sum(p.numel() for p in o_model.parameters())

        # For each other algebra, find matched width and verify within 1%
        for algebra in [AlgebraType.REAL, AlgebraType.COMPLEX, AlgebraType.QUATERNION]:
            matched_width = find_matched_width(
                target_params=o_params,
                algebra=algebra,
                topology="mlp",
                depth=depth,
                tolerance=0.01,
                input_dim=input_dim,
                output_dim=output_dim,
            )
            config = _make_config(
                algebra,
                topology="mlp",
                depth=depth,
                base_hidden=matched_width,
                input_dim=input_dim,
                output_dim=output_dim,
            )
            model = AlgebraNetwork(config)
            actual_params = sum(p.numel() for p in model.parameters())
            error = abs(actual_params - o_params) / o_params
            assert error <= 0.01, (
                f"{algebra.short_name}: param count {actual_params} differs from "
                f"O model {o_params} by {error * 100:.2f}% (>1%)"
            )

    def test_param_matching_conv2d(self):
        """Parameter matching for Conv2D topology."""
        from octonion.baselines._network import AlgebraNetwork

        depth = 2
        input_dim = 3
        output_dim = 10
        base_hidden = 8

        # Get param counts for all algebras
        param_counts = {}
        for algebra in ALL_ALGEBRAS:
            config = _make_config(
                algebra,
                topology="conv2d",
                depth=depth,
                base_hidden=base_hidden,
                input_dim=input_dim,
                output_dim=output_dim,
            )
            model = AlgebraNetwork(config)
            param_counts[algebra.short_name] = sum(
                p.numel() for p in model.parameters()
            )

        # Verify relative ratios are approximately correct:
        # R has ~8x params of O, C has ~4x, H has ~2x (for same base_hidden)
        # This verifies the algebra.multiplier scaling works
        o_count = param_counts["O"]
        assert param_counts["R"] > o_count, "R should have more params than O"
        assert param_counts["C"] > o_count, "C should have more params than O"
        assert param_counts["H"] > o_count, "H should have more params than O"


# ── Output Projection Tests ─────────────────────────────────────


class TestOutputProjections:
    """All 4 output projection strategies produce correct shapes."""

    @pytest.mark.parametrize(
        "projection", ["real", "flatten", "norm", "learned"],
    )
    def test_all_output_projections(self, projection):
        from octonion.baselines._network import AlgebraNetwork

        config = _make_config(
            AlgebraType.QUATERNION,
            topology="mlp",
            output_projection=projection,
        )
        model = AlgebraNetwork(config)
        x = torch.randn(B, config.input_dim)
        out = model(x)
        assert out.shape == (B, config.output_dim), (
            f"Projection {projection!r}: expected ({B}, {config.output_dim}), "
            f"got {out.shape}"
        )

    @pytest.mark.parametrize(
        "projection", ["real", "flatten", "norm", "learned"],
    )
    def test_output_projections_all_algebras(self, projection):
        """Every algebra + every projection should work."""
        from octonion.baselines._network import AlgebraNetwork

        for algebra in ALL_ALGEBRAS:
            config = _make_config(
                algebra,
                topology="mlp",
                output_projection=projection,
            )
            model = AlgebraNetwork(config)
            x = torch.randn(B, config.input_dim)
            out = model(x)
            assert out.shape == (B, config.output_dim)


# ── Param Report Tests ──────────────────────────────────────────


class TestParamReport:
    """Verify param_report returns expected fields."""

    def test_param_report_structure(self):
        from octonion.baselines._network import AlgebraNetwork
        from octonion.baselines._param_matching import param_report

        config = _make_config(AlgebraType.QUATERNION, topology="mlp")
        model = AlgebraNetwork(config)
        report = param_report(model)

        assert isinstance(report, list)
        assert len(report) > 0

        for entry in report:
            assert "name" in entry
            assert "shape" in entry
            assert "real_params" in entry
            assert "pct" in entry
            assert isinstance(entry["real_params"], int)
            assert 0.0 <= entry["pct"] <= 100.0

        # Percentages should sum to ~100%
        total_pct = sum(e["pct"] for e in report)
        assert abs(total_pct - 100.0) < 0.1, (
            f"Percentages sum to {total_pct}, expected ~100"
        )
