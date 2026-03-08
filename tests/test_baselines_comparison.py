"""Tests for the comparison runner and experiment management.

Uses tiny MLP models with synthetic data (2D input, 2-class classification)
for fast testing. Tests verify directory structure, report contents,
statistical testing, manifest management, and plot generation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from octonion.baselines._config import (
    AlgebraType,
    ComparisonConfig,
    NetworkConfig,
    TrainConfig,
)


# ── Test fixtures ─────────────────────────────────────────────────


def _build_synthetic_data(batch_size: int = 16):
    """Build tiny synthetic 2D classification dataset.

    Returns (train_loader, val_loader, test_loader, input_dim, output_dim, input_channels).
    """
    torch.manual_seed(0)
    n_train, n_val, n_test = 64, 32, 32
    input_dim = 4
    output_dim = 2

    def _make_loader(n: int) -> DataLoader:
        x = torch.randn(n, input_dim)
        y = torch.randint(0, output_dim, (n,))
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)

    return _make_loader(n_train), _make_loader(n_val), _make_loader(n_test), input_dim, output_dim, 1


@pytest.fixture
def tiny_comparison_config(tmp_path: Path) -> ComparisonConfig:
    """Build a tiny comparison config for fast tests."""
    train_cfg = TrainConfig(
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
    return ComparisonConfig(
        task="test_task",
        algebras=[AlgebraType.REAL, AlgebraType.COMPLEX],
        seeds=2,
        train_config=train_cfg,
        output_dir=str(tmp_path / "experiments"),
    )


@pytest.fixture
def tiny_comparison_1seed(tmp_path: Path) -> ComparisonConfig:
    """Single seed comparison for faster tests."""
    train_cfg = TrainConfig(
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
    return ComparisonConfig(
        task="test_task",
        algebras=[AlgebraType.REAL, AlgebraType.COMPLEX],
        seeds=1,
        train_config=train_cfg,
        output_dir=str(tmp_path / "experiments"),
    )


# ── Tests ─────────────────────────────────────────────────────────


class TestRunComparisonDirectories:
    """Verify experiment directory structure."""

    def test_run_comparison_creates_directories(self, tiny_comparison_1seed: ComparisonConfig) -> None:
        """Run with 2 algebras x 1 seed, verify directory structure."""
        from octonion.baselines._comparison import run_comparison

        config = tiny_comparison_1seed
        report = run_comparison(
            task_name="test_task",
            build_data_fn=_build_synthetic_data,
            config=config,
            device="cpu",
        )

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
        from octonion.baselines._comparison import ComparisonReport, run_comparison

        config = tiny_comparison_config
        report = run_comparison(
            task_name="test_task",
            build_data_fn=_build_synthetic_data,
            config=config,
            device="cpu",
        )

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
        from octonion.baselines._comparison import run_comparison

        report = run_comparison(
            task_name="test_task",
            build_data_fn=_build_synthetic_data,
            config=tiny_comparison_1seed,
            device="cpu",
        )

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
        from octonion.baselines._comparison import run_comparison

        config = tiny_comparison_1seed
        run_comparison(
            task_name="test_task",
            build_data_fn=_build_synthetic_data,
            config=config,
            device="cpu",
        )

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
        from octonion.baselines._comparison import run_comparison

        config = tiny_comparison_config
        report = run_comparison(
            task_name="test_task",
            build_data_fn=_build_synthetic_data,
            config=config,
            device="cpu",
        )

        # 2 algebras => 1 pair (R_vs_C)
        assert "R_vs_C" in report.pairwise
        pair = report.pairwise["R_vs_C"]
        assert "t_p_value" in pair
        assert "w_p_value" in pair
        assert "effect_size" in pair

    def test_holm_bonferroni_applied(self, tiny_comparison_config: ComparisonConfig) -> None:
        """Verify Holm-Bonferroni correction applied to pairwise p-values."""
        from octonion.baselines._comparison import run_comparison

        config = tiny_comparison_config
        report = run_comparison(
            task_name="test_task",
            build_data_fn=_build_synthetic_data,
            config=config,
            device="cpu",
        )

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
        from octonion.baselines._comparison import run_comparison

        config = tiny_comparison_1seed
        report = run_comparison(
            task_name="test_task",
            build_data_fn=_build_synthetic_data,
            config=config,
            device="cpu",
        )

        counts = list(report.param_counts.values())
        assert len(counts) >= 2
        max_count = max(counts)
        min_count = min(counts)
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
        from octonion.baselines._comparison import run_comparison

        config = tiny_comparison_1seed
        run_comparison(
            task_name="test_task",
            build_data_fn=_build_synthetic_data,
            config=config,
            device="cpu",
        )

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


class TestConfigAndMetrics:
    """Verify per-experiment config and metrics JSON files."""

    def test_config_json_saved(self, tiny_comparison_1seed: ComparisonConfig) -> None:
        """Verify config.json in each experiment dir."""
        from octonion.baselines._comparison import run_comparison

        config = tiny_comparison_1seed
        run_comparison(
            task_name="test_task",
            build_data_fn=_build_synthetic_data,
            config=config,
            device="cpu",
        )

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
        from octonion.baselines._comparison import run_comparison

        config = tiny_comparison_1seed
        run_comparison(
            task_name="test_task",
            build_data_fn=_build_synthetic_data,
            config=config,
            device="cpu",
        )

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
