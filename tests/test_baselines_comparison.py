"""Tests for the comparison runner and experiment management.

Uses tiny MLP models with synthetic data for fast testing. Tests verify
directory structure, report contents, statistical testing, manifest
management, and plot generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from octonion.baselines._config import (
    AlgebraType,
    ComparisonConfig,
    TrainConfig,
)


# ── Test fixtures ─────────────────────────────────────────────────

# Network overrides for tests: ref_hidden=25 with input_dim=32 ensures
# per-algebra width steps are <1% of total params for R+C matching.
_TEST_NET_OVERRIDES = {"ref_hidden": 25, "depth": 1}


def _build_synthetic_data(batch_size: int = 16):
    """Build tiny synthetic classification dataset.

    Uses input_dim=32 so that per-algebra width steps are <1% of total
    params, enabling parameter matching within tolerance.

    Returns (train_loader, val_loader, test_loader, input_dim, output_dim, input_channels).
    """
    torch.manual_seed(0)
    n_train, n_val, n_test = 64, 32, 32
    input_dim = 32
    output_dim = 2

    def _make_loader(n: int) -> DataLoader:
        x = torch.randn(n, input_dim)
        y = torch.randint(0, output_dim, (n,))
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)

    return _make_loader(n_train), _make_loader(n_val), _make_loader(n_test), input_dim, output_dim, 1


def _make_train_config() -> TrainConfig:
    """Build a tiny TrainConfig for fast tests."""
    return TrainConfig(
        epochs=2,
        lr=1e-2,
        optimizer="adam",
        scheduler="cosine",
        weight_decay=0.0,
        early_stopping_patience=100,
        warmup_epochs=0,
        use_amp=False,
        checkpoint_every=100,
        seed=42,
        batch_size=16,
    )


@pytest.fixture
def tiny_comparison_config(tmp_path: Path) -> ComparisonConfig:
    """Build a tiny comparison config for fast tests (2 seeds)."""
    return ComparisonConfig(
        task="test_task",
        algebras=[AlgebraType.REAL, AlgebraType.COMPLEX],
        seeds=2,
        train_config=_make_train_config(),
        output_dir=str(tmp_path / "experiments"),
    )


@pytest.fixture
def tiny_comparison_1seed(tmp_path: Path) -> ComparisonConfig:
    """Single seed comparison for faster tests."""
    return ComparisonConfig(
        task="test_task",
        algebras=[AlgebraType.REAL, AlgebraType.COMPLEX],
        seeds=1,
        train_config=_make_train_config(),
        output_dir=str(tmp_path / "experiments"),
    )


def _run(config: ComparisonConfig, task_name: str = "test_task"):
    """Helper: run_comparison with test overrides."""
    from octonion.baselines._comparison import run_comparison

    return run_comparison(
        task_name=task_name,
        build_data_fn=_build_synthetic_data,
        config=config,
        device="cpu",
        network_config_overrides=_TEST_NET_OVERRIDES,
    )


# ── Tests ─────────────────────────────────────────────────────────


class TestRunComparisonDirectories:
    """Verify experiment directory structure."""

    def test_run_comparison_creates_directories(self, tiny_comparison_1seed: ComparisonConfig) -> None:
        """Run with 2 algebras x 1 seed, verify directory structure."""
        config = tiny_comparison_1seed
        _run(config)

        task_dir = Path(config.output_dir) / "test_task"
        assert task_dir.exists(), f"Task directory {task_dir} not found"

        # Each algebra should have a directory
        for algebra in config.algebras:
            algebra_dir = task_dir / algebra.name
            assert algebra_dir.exists(), f"Algebra dir {algebra_dir} not found"

            # Each seed should have a directory
            for seed in range(config.seeds):
                seed_dir = algebra_dir / str(seed)
                assert seed_dir.exists(), f"Seed dir {seed_dir} not found"


class TestRunComparisonReport:
    """Verify ComparisonReport dataclass contents."""

    def test_run_comparison_returns_report(self, tiny_comparison_config: ComparisonConfig) -> None:
        """Verify ComparisonReport has all expected fields."""
        from octonion.baselines._comparison import ComparisonReport

        config = tiny_comparison_config
        report = _run(config)

        assert isinstance(report, ComparisonReport)
        assert report.task == "test_task"
        assert report.algebras == ["R", "C"]
        assert report.seeds == 2
        assert len(report.per_run) == 4  # 2 algebras x 2 seeds
        assert isinstance(report.param_counts, dict)
        assert "R" in report.param_counts
        assert "C" in report.param_counts
        assert isinstance(report.config_hash, str)
        assert len(report.config_hash) > 0
        assert isinstance(report.timestamp, str)

    def test_per_run_metrics_complete(self, tiny_comparison_1seed: ComparisonConfig) -> None:
        """Verify each per_run entry has required metric keys."""
        report = _run(tiny_comparison_1seed)

        for run in report.per_run:
            assert "algebra" in run
            assert "seed" in run
            assert "metrics" in run
            metrics = run["metrics"]
            assert "train_losses" in metrics
            assert "val_losses" in metrics
            assert "val_accuracies" in metrics
            assert "best_val_acc" in metrics
            assert "total_time_seconds" in metrics


class TestManifest:
    """Verify manifest.json creation and contents."""

    def test_manifest_created(self, tiny_comparison_1seed: ComparisonConfig) -> None:
        """Verify manifest.json exists and contains entry."""
        config = tiny_comparison_1seed
        _run(config)

        manifest_path = Path(config.output_dir) / "manifest.json"
        assert manifest_path.exists(), "manifest.json not found"

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert isinstance(manifest, list)
        assert len(manifest) >= 1
        entry = manifest[-1]
        assert entry["task_name"] == "test_task"
        assert entry["status"] == "complete"
        assert "config_hash" in entry
        assert "timestamp" in entry


class TestPairwiseStats:
    """Verify pairwise statistical comparisons."""

    def test_pairwise_stats_computed(self, tiny_comparison_config: ComparisonConfig) -> None:
        """Verify pairwise comparison dict has entries for each pair."""
        report = _run(tiny_comparison_config)

        # 2 algebras => 1 pair (R_vs_C)
        assert "R_vs_C" in report.pairwise
        pair = report.pairwise["R_vs_C"]
        assert "t_p_value" in pair
        assert "w_p_value" in pair
        assert "effect_size" in pair

    def test_holm_bonferroni_applied(self, tiny_comparison_config: ComparisonConfig) -> None:
        """Verify Holm-Bonferroni correction applied to pairwise p-values."""
        report = _run(tiny_comparison_config)

        assert isinstance(report.corrected_pairwise, list)
        assert len(report.corrected_pairwise) >= 1
        for entry in report.corrected_pairwise:
            assert "original_p" in entry
            assert "adjusted_p" in entry
            assert "rejected" in entry


class TestParamCounts:
    """Verify parameter count validation."""

    def test_param_counts_within_tolerance(self, tiny_comparison_1seed: ComparisonConfig) -> None:
        """Verify all algebras within 1% of each other."""
        report = _run(tiny_comparison_1seed)

        counts = list(report.param_counts.values())
        assert len(counts) >= 2
        # All within 1% of reference
        ref = counts[0]
        for c in counts[1:]:
            relative_diff = abs(c - ref) / ref
            assert relative_diff <= 0.01, (
                f"Param counts differ by {relative_diff * 100:.2f}%: {report.param_counts}"
            )


class TestPlots:
    """Verify plot generation."""

    def test_plots_generated(self, tiny_comparison_1seed: ComparisonConfig) -> None:
        """Verify PNG files exist in output directory."""
        config = tiny_comparison_1seed
        _run(config)

        task_dir = Path(config.output_dir) / "test_task"

        # Comparison bar plot
        assert (task_dir / "comparison_accuracy.png").exists(), "comparison_accuracy.png not found"

        # Param table plot
        assert (task_dir / "param_table.png").exists(), "param_table.png not found"

        # Per-run convergence plots
        for algebra in config.algebras:
            for seed in range(config.seeds):
                conv_path = task_dir / algebra.name / str(seed) / "convergence.png"
                assert conv_path.exists(), f"Convergence plot {conv_path} not found"


class TestConv2dParamMatching:
    """Verify find_matched_width supports conv2d topology.

    Conv2D matching uses base_hidden as the search variable. Because
    AlgebraNetwork internally scales base_hidden by algebra.multiplier,
    the parameter granularity per base_hidden step is coarser than MLP.
    Tests use 10% tolerance (realistic for conv2d) and verify per-algebra
    matching independently.
    """

    _CONV2D_TOLERANCE = 0.10  # 10% tolerance for conv2d granularity

    def test_conv2d_matching_returns_valid_width_all_algebras(self) -> None:
        """find_matched_width with topology='conv2d' returns valid base_hidden for all 4 algebras."""
        from octonion.baselines._config import AlgebraType, NetworkConfig
        from octonion.baselines._network import AlgebraNetwork
        from octonion.baselines._param_matching import find_matched_width

        # Build a reference model to get target param count
        # Use octonion (smallest multiplier) as reference for best granularity
        ref_config = NetworkConfig(
            algebra=AlgebraType.OCTONION,
            topology="conv2d",
            depth=6,
            base_hidden=8,
            input_dim=3,
            output_dim=10,
            activation="split_relu",
            output_projection="flatten",
            use_batchnorm=True,
        )
        ref_model = AlgebraNetwork(ref_config)
        target_params = sum(p.numel() for p in ref_model.parameters())

        # Only iterate original 4 algebras that AlgebraNetwork supports
        _NETWORK_ALGEBRAS = [AlgebraType.REAL, AlgebraType.COMPLEX, AlgebraType.QUATERNION, AlgebraType.OCTONION]
        for algebra in _NETWORK_ALGEBRAS:
            width = find_matched_width(
                target_params=target_params,
                algebra=algebra,
                topology="conv2d",
                depth=6,
                tolerance=self._CONV2D_TOLERANCE,
                input_dim=3,
                output_dim=10,
            )
            assert isinstance(width, int)
            assert width >= 1, f"{algebra.short_name}: width must be >= 1"

    def test_conv2d_matched_models_within_tolerance(self) -> None:
        """Models built with matched width have param counts within tolerance of target."""
        from octonion.baselines._config import AlgebraType, NetworkConfig
        from octonion.baselines._network import AlgebraNetwork
        from octonion.baselines._param_matching import find_matched_width

        ref_config = NetworkConfig(
            algebra=AlgebraType.OCTONION,
            topology="conv2d",
            depth=6,
            base_hidden=8,
            input_dim=3,
            output_dim=10,
            activation="split_relu",
            output_projection="flatten",
            use_batchnorm=True,
        )
        ref_model = AlgebraNetwork(ref_config)
        target_params = sum(p.numel() for p in ref_model.parameters())

        # Only iterate original 4 algebras that AlgebraNetwork supports
        _NETWORK_ALGEBRAS = [AlgebraType.REAL, AlgebraType.COMPLEX, AlgebraType.QUATERNION, AlgebraType.OCTONION]
        for algebra in _NETWORK_ALGEBRAS:
            width = find_matched_width(
                target_params=target_params,
                algebra=algebra,
                topology="conv2d",
                depth=6,
                tolerance=self._CONV2D_TOLERANCE,
                input_dim=3,
                output_dim=10,
            )
            # Build model with the returned width
            config = NetworkConfig(
                algebra=algebra,
                topology="conv2d",
                depth=6,
                base_hidden=width,
                input_dim=3,
                output_dim=10,
                activation="split_relu",
                output_projection="flatten",
                use_batchnorm=True,
            )
            model = AlgebraNetwork(config)
            actual = sum(p.numel() for p in model.parameters())
            diff = abs(actual - target_params) / target_params
            assert diff <= self._CONV2D_TOLERANCE, (
                f"{algebra.short_name}: param diff {diff * 100:.2f}% > "
                f"{self._CONV2D_TOLERANCE * 100:.0f}% "
                f"(target={target_params}, actual={actual}, width={width})"
            )

    def test_mlp_matching_still_works(self) -> None:
        """topology='mlp' backward compatibility -- still works as before."""
        from octonion.baselines._config import AlgebraType
        from octonion.baselines._param_matching import (
            _build_simple_mlp,
            find_matched_width,
        )

        # Use ref_hidden=25 with input_dim=32 to ensure per-unit steps <1%
        # (matching the existing test fixture _TEST_NET_OVERRIDES)
        ref = _build_simple_mlp(AlgebraType.REAL, hidden=25, depth=1, input_dim=32, output_dim=2)
        target = sum(p.numel() for p in ref.parameters())

        width = find_matched_width(
            target_params=target,
            algebra=AlgebraType.COMPLEX,
            topology="mlp",
            depth=1,
            tolerance=0.01,
            input_dim=32,
            output_dim=2,
        )
        assert isinstance(width, int)
        assert width >= 1

    def test_conv2d_matching_raises_on_impossible_target(self) -> None:
        """find_matched_width raises ValueError when no width can match target."""
        from octonion.baselines._config import AlgebraType
        from octonion.baselines._param_matching import find_matched_width

        with pytest.raises(ValueError, match="Cannot match"):
            find_matched_width(
                target_params=1,  # impossibly small
                algebra=AlgebraType.REAL,
                topology="conv2d",
                depth=6,
                tolerance=0.001,  # very tight
                input_dim=3,
                output_dim=10,
            )


class TestConv2dComparison:
    """Verify run_comparison with topology='conv2d' dispatches correctly."""

    def _build_conv2d_data(self, batch_size: int = 8):
        """Build tiny CIFAR-like synthetic dataset: [B, 3, 8, 8] images.

        Returns (train_loader, val_loader, test_loader, input_dim=3, output_dim=2, input_channels=3).
        """
        torch.manual_seed(0)
        n_train, n_val, n_test = 32, 16, 16

        def _make_loader(n: int) -> DataLoader:
            x = torch.randn(n, 3, 8, 8)
            y = torch.randint(0, 2, (n,))
            return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)

        return _make_loader(n_train), _make_loader(n_val), _make_loader(n_test), 3, 2, 3

    def test_conv2d_comparison_trains_successfully(self, tmp_path: Path) -> None:
        """run_comparison with topology='conv2d' builds AlgebraNetwork and trains."""
        from octonion.baselines._comparison import run_comparison

        config = ComparisonConfig(
            task="conv2d_test",
            algebras=[AlgebraType.REAL, AlgebraType.QUATERNION],
            seeds=1,
            train_config=_make_train_config(),
            output_dir=str(tmp_path / "experiments"),
        )

        report = run_comparison(
            task_name="conv2d_test",
            build_data_fn=self._build_conv2d_data,
            config=config,
            device="cpu",
            network_config_overrides={
                "topology": "conv2d",
                "depth": 6,
                "ref_hidden": 4,
            },
        )

        assert len(report.per_run) == 2  # 2 algebras x 1 seed
        for run in report.per_run:
            assert "best_val_acc" in run["metrics"]

    def test_mlp_comparison_still_works(self, tmp_path: Path) -> None:
        """run_comparison with topology='mlp' (default) still builds _SimpleAlgebraMLP."""
        config = ComparisonConfig(
            task="mlp_test",
            algebras=[AlgebraType.REAL, AlgebraType.COMPLEX],
            seeds=1,
            train_config=_make_train_config(),
            output_dir=str(tmp_path / "experiments"),
        )

        report = _run(config, task_name="mlp_test")
        assert len(report.per_run) == 2

    def test_conv2d_cifar_shape_no_crash(self) -> None:
        """CIFAR-shaped data [B, 3, 32, 32] flows through conv2d without shape mismatch."""
        from octonion.baselines._config import NetworkConfig
        from octonion.baselines._network import AlgebraNetwork
        from octonion.baselines._param_matching import _build_conv_model

        for algebra in [AlgebraType.REAL, AlgebraType.QUATERNION]:
            model = _build_conv_model(
                algebra=algebra,
                base_hidden=4,
                depth=6,
                input_dim=3,
                output_dim=10,
            )
            model.eval()
            x = torch.randn(2, 3, 32, 32)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (2, 10), (
                f"{algebra.short_name}: output shape {out.shape}, expected (2, 10)"
            )

    def test_conv2d_config_json_records_topology(self, tmp_path: Path) -> None:
        """Per-run config.json records correct topology='conv2d'."""
        from octonion.baselines._comparison import run_comparison

        config = ComparisonConfig(
            task="topo_test",
            algebras=[AlgebraType.REAL],
            seeds=1,
            train_config=_make_train_config(),
            output_dir=str(tmp_path / "experiments"),
        )

        run_comparison(
            task_name="topo_test",
            build_data_fn=self._build_conv2d_data,
            config=config,
            device="cpu",
            network_config_overrides={
                "topology": "conv2d",
                "depth": 6,
                "ref_hidden": 4,
            },
        )

        config_path = tmp_path / "experiments" / "topo_test" / "REAL" / "0" / "config.json"
        with open(config_path) as f:
            saved = json.load(f)
        assert saved["network_config"]["topology"] == "conv2d"

    def test_conv2d_forward_pass_produces_correct_output_shape(self) -> None:
        """Forward pass on [B, 3, 32, 32] through conv2d network produces [B, 10]."""
        from octonion.baselines._param_matching import _build_conv_model

        # Only iterate original 4 algebras that AlgebraNetwork/conv2d supports
        _NETWORK_ALGEBRAS = [AlgebraType.REAL, AlgebraType.COMPLEX, AlgebraType.QUATERNION, AlgebraType.OCTONION]
        for algebra in _NETWORK_ALGEBRAS:
            model = _build_conv_model(
                algebra=algebra,
                base_hidden=4,
                depth=6,
                input_dim=3,
                output_dim=10,
            )
            model.eval()
            x = torch.randn(2, 3, 32, 32)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (2, 10), (
                f"{algebra.short_name}: output {out.shape}, expected (2, 10)"
            )


class TestConfigAndMetrics:
    """Verify per-experiment config and metrics JSON files."""

    def test_config_json_saved(self, tiny_comparison_1seed: ComparisonConfig) -> None:
        """Verify config.json in each experiment dir."""
        config = tiny_comparison_1seed
        _run(config)

        task_dir = Path(config.output_dir) / "test_task"
        for algebra in config.algebras:
            for seed in range(config.seeds):
                config_path = task_dir / algebra.name / str(seed) / "config.json"
                assert config_path.exists(), f"config.json not found at {config_path}"

                with open(config_path) as f:
                    saved_config = json.load(f)
                assert "train_config" in saved_config
                assert "network_config" in saved_config

    def test_metrics_json_saved(self, tiny_comparison_1seed: ComparisonConfig) -> None:
        """Verify metrics.json in each experiment dir."""
        config = tiny_comparison_1seed
        _run(config)

        task_dir = Path(config.output_dir) / "test_task"
        for algebra in config.algebras:
            for seed in range(config.seeds):
                metrics_path = task_dir / algebra.name / str(seed) / "metrics.json"
                assert metrics_path.exists(), f"metrics.json not found at {metrics_path}"

                with open(metrics_path) as f:
                    metrics = json.load(f)
                assert "train_losses" in metrics
                assert "val_losses" in metrics
                assert "best_val_acc" in metrics


# ── Same-width mode tests ─────────────────────────────────────────


class TestSameWidthMode:
    """Verify same-width mode: same base_hidden, different base_filters."""

    def test_same_width_real_representational_width(self) -> None:
        """Original 4 algebras get the same real representational width."""
        ref_hidden = 8
        _NETWORK_ALGEBRAS = [AlgebraType.REAL, AlgebraType.COMPLEX, AlgebraType.QUATERNION, AlgebraType.OCTONION]
        for algebra in _NETWORK_ALGEBRAS:
            # base_filters = base_hidden * multiplier
            # real_width = base_filters * dim (for non-real) or base_filters (for real)
            base_filters = ref_hidden * algebra.multiplier
            real_width = base_filters * algebra.dim if algebra != AlgebraType.REAL else base_filters
            # All should have same real width = ref_hidden * multiplier * dim
            # R: 8*8*1=64, C: 8*4*2=64, H: 8*2*4=64, O: 8*1*8=64
            assert real_width == ref_hidden * 8, (
                f"{algebra.short_name}: real_width={real_width}, expected={ref_hidden * 8}"
            )

    def test_same_width_r_has_more_params_than_o(self) -> None:
        """R should have more params than O at same base_hidden (paper protocol)."""
        from octonion.baselines._config import NetworkConfig
        from octonion.baselines._network import AlgebraNetwork

        ref_hidden = 8
        param_counts = {}
        _NETWORK_ALGEBRAS = [AlgebraType.REAL, AlgebraType.COMPLEX, AlgebraType.QUATERNION, AlgebraType.OCTONION]
        for algebra in _NETWORK_ALGEBRAS:
            config = NetworkConfig(
                algebra=algebra,
                topology="conv2d",
                depth=6,
                base_hidden=ref_hidden,
                input_dim=3,
                output_dim=10,
            )
            model = AlgebraNetwork(config)
            param_counts[algebra.short_name] = sum(
                p.numel() for p in model.parameters()
            )

        # Real has most params, octonion fewest (algebraic weight sharing)
        assert param_counts["R"] > param_counts["H"]
        assert param_counts["H"] > param_counts["O"]
