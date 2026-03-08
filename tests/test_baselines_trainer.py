"""Tests for training utility, Optuna HP search, statistical testing, and plotting.

Tests follow the behavior specs from 03-04-PLAN.md:
- Training loop with full observability (gradient stats, VRAM, TensorBoard)
- Seed determinism, early stopping, checkpointing, LR warmup
- Optuna hyperparameter search integration
- Statistical significance tests (paired t-test, Wilcoxon, Holm-Bonferroni)
- Plotting utilities for convergence curves and comparison bars
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from octonion.baselines._config import TrainConfig


# ── Tiny model and data fixtures ──────────────────────────────────


class TinyMLP(nn.Module):
    """Minimal MLP for testing training infrastructure."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, output_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@pytest.fixture
def tiny_data() -> tuple[DataLoader, DataLoader]:
    """Create tiny synthetic classification data for training tests."""
    torch.manual_seed(0)
    n_train, n_val = 64, 16
    input_dim, num_classes = 8, 4

    x_train = torch.randn(n_train, input_dim)
    y_train = torch.randint(0, num_classes, (n_train,))
    x_val = torch.randn(n_val, input_dim)
    y_val = torch.randint(0, num_classes, (n_val,))

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    return train_loader, val_loader


@pytest.fixture
def tiny_config() -> TrainConfig:
    """Minimal training config for fast tests."""
    return TrainConfig(
        epochs=5,
        lr=1e-2,
        optimizer="adam",
        scheduler="cosine",
        weight_decay=0.0,
        early_stopping_patience=10,
        warmup_epochs=2,
        use_amp=False,
        checkpoint_every=2,
        seed=42,
        batch_size=16,
    )


# ── Task 1: Training utility tests ───────────────────────────────


class TestSeedDeterminism:
    """seed_everything produces deterministic training."""

    def test_seed_determinism(self, tiny_data: tuple, tmp_path: Path) -> None:
        """Same seed produces same loss after 3 epochs."""
        from octonion.baselines._trainer import seed_everything, train_model

        train_loader, val_loader = tiny_data
        config = TrainConfig(
            epochs=3,
            lr=1e-2,
            optimizer="adam",
            scheduler="cosine",
            warmup_epochs=0,
            early_stopping_patience=10,
            checkpoint_every=100,
            seed=42,
            batch_size=16,
        )

        # Run 1
        seed_everything(42)
        model1 = TinyMLP()
        out_dir1 = str(tmp_path / "run1")
        result1 = train_model(model1, train_loader, val_loader, config, out_dir1, device="cpu")

        # Run 2
        seed_everything(42)
        model2 = TinyMLP()
        out_dir2 = str(tmp_path / "run2")
        result2 = train_model(model2, train_loader, val_loader, config, out_dir2, device="cpu")

        assert result1["train_losses"][-1] == pytest.approx(
            result2["train_losses"][-1], abs=1e-6
        )


class TestTrainReturnsMetrics:
    """train_model returns dict with all expected keys."""

    def test_train_returns_metrics(self, tiny_data: tuple, tiny_config: TrainConfig, tmp_path: Path) -> None:
        from octonion.baselines._trainer import seed_everything, train_model

        train_loader, val_loader = tiny_data
        seed_everything(tiny_config.seed)
        model = TinyMLP()
        result = train_model(model, train_loader, val_loader, tiny_config, str(tmp_path), device="cpu")

        expected_keys = {
            "train_losses",
            "val_losses",
            "val_accuracies",
            "best_val_acc",
            "best_val_loss",
            "total_time_seconds",
            "epochs_trained",
            "early_stopped",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - result.keys()}"
        )
        assert isinstance(result["train_losses"], list)
        assert len(result["train_losses"]) == tiny_config.epochs
        assert isinstance(result["total_time_seconds"], float)
        assert result["total_time_seconds"] > 0


class TestEarlyStopping:
    """Early stopping halts training when val loss stops improving."""

    def test_early_stopping(self, tiny_data: tuple, tmp_path: Path) -> None:
        from octonion.baselines._trainer import seed_everything, train_model

        train_loader, val_loader = tiny_data
        config = TrainConfig(
            epochs=100,  # High max, but early stopping should trigger
            lr=1e-2,
            optimizer="adam",
            scheduler="cosine",
            warmup_epochs=0,
            early_stopping_patience=2,
            checkpoint_every=100,
            seed=42,
            batch_size=16,
        )
        seed_everything(42)
        model = TinyMLP()
        result = train_model(model, train_loader, val_loader, config, str(tmp_path), device="cpu")

        # Should stop before 100 epochs (patience=2)
        assert result["epochs_trained"] < config.epochs
        assert result["early_stopped"] is True


class TestCheckpointing:
    """Checkpoint save and load restores exact training state."""

    def test_checkpoint_save_load(self, tiny_data: tuple, tiny_config: TrainConfig, tmp_path: Path) -> None:
        from octonion.baselines._trainer import (
            load_checkpoint,
            save_checkpoint,
            seed_everything,
        )

        seed_everything(42)
        model = TinyMLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        epoch = 5
        best_val_loss = 0.42
        metrics = {"train_losses": [1.0, 0.9, 0.8, 0.7, 0.6]}

        ckpt_path = str(tmp_path / "checkpoint.pt")
        save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, best_val_loss, metrics)

        # Load into fresh model
        model2 = TinyMLP()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=10)

        meta = load_checkpoint(ckpt_path, model2, optimizer2, scheduler2)

        assert meta["epoch"] == epoch
        assert meta["best_val_loss"] == pytest.approx(best_val_loss)

        # Model weights should match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            torch.testing.assert_close(p1.data, p2.data)


class TestGradientStatsLogged:
    """Gradient statistics logged to TensorBoard."""

    def test_gradient_stats_logged(self, tiny_data: tuple, tmp_path: Path) -> None:
        from octonion.baselines._trainer import seed_everything, train_model

        train_loader, val_loader = tiny_data
        config = TrainConfig(
            epochs=2,
            lr=1e-2,
            optimizer="adam",
            scheduler="cosine",
            warmup_epochs=0,
            early_stopping_patience=10,
            checkpoint_every=100,
            seed=42,
            batch_size=16,
        )
        seed_everything(42)
        model = TinyMLP()
        train_model(model, train_loader, val_loader, config, str(tmp_path), device="cpu")

        # TensorBoard events file should exist
        events_files = list(tmp_path.rglob("events.out.tfevents.*"))
        assert len(events_files) > 0, "No TensorBoard events files found"


class TestLRWarmup:
    """LR warmup linearly scales from 0 to target lr."""

    def test_lr_warmup(self, tiny_data: tuple, tmp_path: Path) -> None:
        from octonion.baselines._trainer import seed_everything, train_model

        train_loader, val_loader = tiny_data
        target_lr = 1e-2
        warmup_epochs = 3
        config = TrainConfig(
            epochs=5,
            lr=target_lr,
            optimizer="adam",
            scheduler="cosine",
            warmup_epochs=warmup_epochs,
            early_stopping_patience=10,
            checkpoint_every=100,
            seed=42,
            batch_size=16,
        )
        seed_everything(42)
        model = TinyMLP()
        result = train_model(model, train_loader, val_loader, config, str(tmp_path), device="cpu")

        # LR at epoch 0 should be less than LR at warmup_epochs
        lr_history = result.get("lr_history", [])
        assert len(lr_history) >= warmup_epochs, (
            f"lr_history too short: {len(lr_history)} < {warmup_epochs}"
        )
        assert lr_history[0] < lr_history[warmup_epochs - 1], (
            f"LR at epoch 0 ({lr_history[0]}) should be less than LR at epoch "
            f"{warmup_epochs - 1} ({lr_history[warmup_epochs - 1]})"
        )


class TestOptunaStudy:
    """Optuna hyperparameter search integration."""

    def test_optuna_study_completes(self, tiny_data: tuple, tmp_path: Path) -> None:
        from octonion.baselines._config import AlgebraType
        from octonion.baselines._trainer import run_optuna_study, seed_everything

        train_loader, val_loader = tiny_data
        seed_everything(42)

        def model_builder(algebra: AlgebraType) -> nn.Module:
            return TinyMLP()

        result = run_optuna_study(
            model_builder_fn=model_builder,
            train_loader=train_loader,
            val_loader=val_loader,
            algebra=AlgebraType.REAL,
            n_trials=2,
            study_name="test_study",
            output_dir=str(tmp_path / "optuna"),
            device="cpu",
        )

        assert isinstance(result, dict)
        assert "best_params" in result
        assert "best_value" in result
        assert "n_trials" in result
        assert "study_name" in result
        # Check that best_params has expected keys
        assert "lr" in result["best_params"]
        assert "weight_decay" in result["best_params"]
        assert "optimizer" in result["best_params"]

    def test_optuna_study_saves_results(self, tiny_data: tuple, tmp_path: Path) -> None:
        from octonion.baselines._config import AlgebraType
        from octonion.baselines._trainer import run_optuna_study, seed_everything

        train_loader, val_loader = tiny_data
        seed_everything(42)

        def model_builder(algebra: AlgebraType) -> nn.Module:
            return TinyMLP()

        output_dir = str(tmp_path / "optuna_results")
        run_optuna_study(
            model_builder_fn=model_builder,
            train_loader=train_loader,
            val_loader=val_loader,
            algebra=AlgebraType.REAL,
            n_trials=2,
            study_name="save_test",
            output_dir=output_dir,
            device="cpu",
        )

        results_file = Path(output_dir) / "save_test_results.json"
        assert results_file.exists(), f"Results file not found at {results_file}"
        with open(results_file) as f:
            data = json.load(f)
        assert "best_params" in data
