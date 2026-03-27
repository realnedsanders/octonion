"""Experiment orchestration for optimization landscape analysis.

Ties together tasks, trainer, Hessian analysis, curvature measurement,
and gradient variance collection into a single configurable pipeline
with incremental result saving and crash resilience.

Provides:
- LandscapeConfig: Experiment configuration dataclass
- run_landscape_experiment: Full experiment runner with incremental saves
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from octonion.baselines._config import AlgebraType, TrainConfig
from octonion.baselines._param_matching import _SimpleAlgebraMLP, find_matched_width
from octonion.baselines._trainer import seed_everything, train_model
from octonion.landscape._curvature import measure_curvature
from octonion.landscape._gradient_stats import collect_gradient_stats
from octonion.landscape._hessian import compute_hessian_spectrum
from octonion.tasks import (
    build_algebra_native_multi,
    build_algebra_native_single,
    build_classification,
    build_cross_product_recovery,
    build_sinusoidal_regression,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LandscapeConfig:
    """Configuration for the full landscape experiment.

    Specifies all experiment parameters: tasks, algebras, optimizers,
    seeds, model sizes, and analysis settings.
    """

    tasks: list[str] = field(default_factory=lambda: [
        "algebra_native_single",
        "algebra_native_multi",
        "cross_product_3d",             # Positive control: quaternions should win
        "cross_product_7d_noise0",       # Clean signal: octonionic 7D cross product
        "cross_product_7d_noise5",       # Low noise (5%)
        "cross_product_7d_noise15",      # Medium noise (15%)
        "cross_product_7d_noise30",      # High noise (30%)
        "sinusoidal",
        "classification",
    ])
    algebras: list[AlgebraType] = field(default_factory=lambda: [
        AlgebraType.REAL, AlgebraType.COMPLEX, AlgebraType.QUATERNION,
        AlgebraType.OCTONION, AlgebraType.PHM8, AlgebraType.R8_DENSE,
    ])
    optimizers: list[str] = field(default_factory=lambda: [
        "sgd", "adam", "riemannian_adam", "lbfgs", "shampoo",
    ])
    seeds: list[int] = field(default_factory=lambda: list(range(20)))
    base_hidden: int = 16  # Octonionic units (small for tractability)
    depth: int = 1
    epochs: int = 100
    batch_size: int = 128
    lbfgs_batch_size: int = 512  # Larger for LBFGS stability
    output_dir: str = "results/landscape"
    hessian_checkpoints: list[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    hessian_seeds: list[int] = field(
        default_factory=lambda: [0, 4, 9, 14, 19]  # 5 representative
    )
    n_curvature_directions: int = 50
    n_train: int = 50_000
    n_test: int = 10_000
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Task metadata and dimensions
# ---------------------------------------------------------------------------

# Task-specific I/O dimensions.  Defaults are input_dim=8, output_dim=8.
_TASK_DIMS: dict[str, dict[str, int]] = {
    "cross_product_3d":         {"input_dim": 3, "output_dim": 3},
    "cross_product_7d_noise0":  {"input_dim": 7, "output_dim": 7},
    "cross_product_7d_noise5":  {"input_dim": 7, "output_dim": 7},
    "cross_product_7d_noise15": {"input_dim": 7, "output_dim": 7},
    "cross_product_7d_noise30": {"input_dim": 7, "output_dim": 7},
    "sinusoidal":               {"input_dim": 8, "output_dim": 3},  # n_components=3
    "classification":           {"input_dim": 8, "output_dim": 5},  # n_classes=5
}


# ---------------------------------------------------------------------------
# Optimizer-specific TrainConfig overrides
# ---------------------------------------------------------------------------


def _optimizer_train_config(
    optimizer_name: str,
    config: LandscapeConfig,
    manifold_type: str = "sphere",
    is_hessian_seed: bool = False,
) -> TrainConfig:
    """Build a TrainConfig with optimizer-specific settings.

    Args:
        optimizer_name: Optimizer name (sgd, adam, lbfgs, riemannian_adam, shampoo).
        config: LandscapeConfig with shared settings.
        manifold_type: Manifold type for riemannian_adam.
        is_hessian_seed: If True, override checkpoint_every to save
            intermediate checkpoints for Hessian analysis.

    Returns:
        TrainConfig instance tuned for the given optimizer.
    """
    base = dict(
        epochs=config.epochs,
        weight_decay=0.0,
        early_stopping_patience=max(20, config.epochs),  # Effectively disabled
        warmup_epochs=0,
        use_amp=False,
        checkpoint_every=max(1, config.epochs),  # No intermediate checkpoints
        seed=42,
        batch_size=config.batch_size,
    )

    if optimizer_name == "sgd":
        tc = TrainConfig(
            **base,
            optimizer="sgd",
            lr=0.01,
            scheduler="cosine",
            nesterov=True,
        )
    elif optimizer_name == "adam":
        tc = TrainConfig(
            **base,
            optimizer="adam",
            lr=1e-3,
            scheduler="cosine",
        )
    elif optimizer_name == "lbfgs":
        tc = TrainConfig(
            **{**base, "batch_size": config.lbfgs_batch_size},
            optimizer="lbfgs",
            lr=1.0,
            scheduler="cosine",
        )
    elif optimizer_name == "riemannian_adam":
        tc = TrainConfig(
            **base,
            optimizer="riemannian_adam",
            lr=1e-3,
            scheduler="cosine",
            manifold_type=manifold_type,
        )
    elif optimizer_name == "shampoo":
        tc = TrainConfig(
            **base,
            optimizer="shampoo",
            lr=1e-2,
            scheduler="cosine",
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name!r}")

    # For Hessian seeds, save frequent checkpoints to capture intermediate fractions
    if is_hessian_seed and config.epochs >= 4:
        tc.checkpoint_every = max(1, config.epochs // 4)

    return tc


# ---------------------------------------------------------------------------
# Loss function dispatch
# ---------------------------------------------------------------------------


def _get_loss_fn(task_name: str) -> nn.Module:
    """Return the appropriate loss function for a task.

    Regression tasks (algebra_native, sinusoidal, cross product):
        nn.MSELoss()
    Classification: nn.CrossEntropyLoss()

    Args:
        task_name: One of the 9 supported task names.

    Returns:
        Instantiated loss function module.
    """
    if task_name == "classification":
        return nn.CrossEntropyLoss()
    else:
        return nn.MSELoss()


# ---------------------------------------------------------------------------
# Task data builder dispatch
# ---------------------------------------------------------------------------


def _build_task_data(
    task_name: str, config: LandscapeConfig, seed: int = 42
) -> tuple[TensorDataset, TensorDataset, dict[str, Any]]:
    """Build train/test datasets for a given task name.

    Returns (train_ds, test_ds, metadata) where metadata includes
    task-specific info like bayes_optimal_accuracy for classification.

    Args:
        task_name: One of the 9 supported task names.
        config: LandscapeConfig for n_train, n_test, depth settings.
        seed: Random seed for data generation.

    Returns:
        Tuple of (train_dataset, test_dataset, metadata_dict).

    Raises:
        ValueError: If task_name is not recognized.
    """
    metadata: dict[str, Any] = {}

    if task_name == "algebra_native_single":
        train_ds, test_ds = build_algebra_native_single(
            n_train=config.n_train, n_test=config.n_test, dim=8, seed=seed,
        )
    elif task_name == "algebra_native_multi":
        train_ds, test_ds = build_algebra_native_multi(
            n_train=config.n_train, n_test=config.n_test, dim=8,
            depth=config.depth, seed=seed,
        )
    elif task_name == "cross_product_3d":
        train_ds, test_ds = build_cross_product_recovery(
            n_train=config.n_train, n_test=config.n_test,
            cross_dim=3, noise_level=0.0, seed=seed,
        )
    elif task_name == "cross_product_7d_noise0":
        train_ds, test_ds = build_cross_product_recovery(
            n_train=config.n_train, n_test=config.n_test,
            cross_dim=7, noise_level=0.0, seed=seed,
        )
    elif task_name == "cross_product_7d_noise5":
        train_ds, test_ds = build_cross_product_recovery(
            n_train=config.n_train, n_test=config.n_test,
            cross_dim=7, noise_level=0.05, seed=seed,
        )
    elif task_name == "cross_product_7d_noise15":
        train_ds, test_ds = build_cross_product_recovery(
            n_train=config.n_train, n_test=config.n_test,
            cross_dim=7, noise_level=0.15, seed=seed,
        )
    elif task_name == "cross_product_7d_noise30":
        train_ds, test_ds = build_cross_product_recovery(
            n_train=config.n_train, n_test=config.n_test,
            cross_dim=7, noise_level=0.30, seed=seed,
        )
    elif task_name == "sinusoidal":
        train_ds, test_ds = build_sinusoidal_regression(
            n_train=config.n_train, n_test=config.n_test, dim=8, seed=seed,
        )
    elif task_name == "classification":
        train_ds, test_ds, cls_meta = build_classification(
            n_train=config.n_train, n_test=config.n_test, dim=8, seed=seed,
        )
        metadata["bayes_optimal_accuracy"] = cls_meta["bayes_optimal_accuracy"]
    else:
        raise ValueError(
            f"Unknown task: {task_name!r}. "
            f"Valid tasks: algebra_native_single, algebra_native_multi, "
            f"cross_product_3d, cross_product_7d_noise0, cross_product_7d_noise5, "
            f"cross_product_7d_noise15, cross_product_7d_noise30, "
            f"sinusoidal, classification"
        )

    return train_ds, test_ds, metadata


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _build_model(
    algebra: AlgebraType,
    task_name: str,
    config: LandscapeConfig,
) -> nn.Module:
    """Build parameter-matched model for a given algebra and task.

    Models are parameter-matched against the octonionic reference so that
    differences in optimization landscape stem from algebra structure,
    not model capacity.

    Args:
        algebra: Which algebra type to use.
        task_name: Task name (determines input/output dimensions).
        config: LandscapeConfig for base_hidden, depth.

    Returns:
        Instantiated _SimpleAlgebraMLP model.
    """
    dims = _TASK_DIMS.get(task_name, {"input_dim": 8, "output_dim": 8})
    input_dim = dims["input_dim"]
    output_dim = dims["output_dim"]

    # Reference model is octonionic
    if algebra == AlgebraType.OCTONION:
        return _SimpleAlgebraMLP(
            algebra=algebra,
            hidden=config.base_hidden,
            depth=config.depth,
            input_dim=input_dim,
            output_dim=output_dim,
        )

    # For other algebras, find matched width
    ref_model = _SimpleAlgebraMLP(
        algebra=AlgebraType.OCTONION,
        hidden=config.base_hidden,
        depth=config.depth,
        input_dim=input_dim,
        output_dim=output_dim,
    )
    target_params = sum(p.numel() for p in ref_model.parameters())

    try:
        width = find_matched_width(
            target_params=target_params,
            algebra=algebra,
            topology="mlp",
            depth=config.depth,
            tolerance=0.10,  # 10% for small models
            input_dim=input_dim,
            output_dim=output_dim,
        )
    except ValueError:
        # Fallback: use base_hidden * multiplier ratio
        width = max(1, config.base_hidden * AlgebraType.OCTONION.multiplier // algebra.multiplier)
        logger.warning(
            f"Could not match params for {algebra.short_name}; "
            f"falling back to width={width}"
        )

    return _SimpleAlgebraMLP(
        algebra=algebra,
        hidden=width,
        depth=config.depth,
        input_dim=input_dim,
        output_dim=output_dim,
    )


# ---------------------------------------------------------------------------
# Incremental save/resume
# ---------------------------------------------------------------------------


def _result_path(
    output_dir: str, task: str, optimizer: str, algebra: str, seed: int
) -> Path:
    """Construct the path for a single run's result JSON.

    Args:
        output_dir: Base output directory.
        task: Task name.
        optimizer: Optimizer name.
        algebra: Algebra short name.
        seed: Random seed.

    Returns:
        Path object for the result JSON file.
    """
    return Path(output_dir) / task / optimizer / algebra / f"seed_{seed}" / "result.json"


def _result_exists(
    output_dir: str, task: str, optimizer: str, algebra: str, seed: int
) -> bool:
    """Check if a result already exists (for resume support).

    Args:
        output_dir: Base output directory.
        task: Task name.
        optimizer: Optimizer name.
        algebra: Algebra short name.
        seed: Random seed.

    Returns:
        True if result.json exists at the expected path.
    """
    return _result_path(output_dir, task, optimizer, algebra, seed).exists()


def _save_result(
    output_dir: str,
    task: str,
    optimizer: str,
    algebra: str,
    seed: int,
    result: dict[str, Any],
) -> None:
    """Save a single run's result to JSON.

    Creates parent directories as needed. Serializes lists of floats.

    Args:
        output_dir: Base output directory.
        task: Task name.
        optimizer: Optimizer name.
        algebra: Algebra short name.
        seed: Random seed.
        result: Dict with training metrics.
    """
    path = _result_path(output_dir, task, optimizer, algebra, seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Make JSON-serializable
    serializable = {}
    for k, v in result.items():
        if isinstance(v, list):
            serializable[k] = [float(x) if isinstance(x, (int, float)) else x for x in v]
        elif isinstance(v, (int, float, str, bool)):
            serializable[k] = v
        elif isinstance(v, dict):
            serializable[k] = v
        else:
            serializable[k] = str(v)

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def _load_result(
    output_dir: str, task: str, optimizer: str, algebra: str, seed: int
) -> dict[str, Any]:
    """Load a previously saved result from JSON.

    Args:
        output_dir: Base output directory.
        task: Task name.
        optimizer: Optimizer name.
        algebra: Algebra short name.
        seed: Random seed.

    Returns:
        Loaded result dict.
    """
    path = _result_path(output_dir, task, optimizer, algebra, seed)
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Hessian checkpoint save/load
# ---------------------------------------------------------------------------


def _hessian_checkpoint_dir(
    output_dir: str, task: str, optimizer: str, algebra: str, seed: int
) -> Path:
    """Directory for Hessian model checkpoints."""
    return (
        Path(output_dir) / task / optimizer / algebra
        / f"seed_{seed}" / "hessian_checkpoints"
    )


def _save_hessian_checkpoint(
    output_dir: str,
    task: str,
    optimizer: str,
    algebra: str,
    seed: int,
    fraction: float,
    model: nn.Module,
) -> None:
    """Save model state_dict at a Hessian checkpoint fraction.

    Args:
        output_dir: Base output directory.
        task: Task name.
        optimizer: Optimizer name.
        algebra: Algebra short name.
        seed: Random seed.
        fraction: Training fraction (0.0, 0.25, 0.5, 1.0).
        model: Model to save.
    """
    ckpt_dir = _hessian_checkpoint_dir(output_dir, task, optimizer, algebra, seed)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"checkpoint_{fraction:.2f}.pt"
    torch.save(model.state_dict(), path)


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------


def run_landscape_experiment(config: LandscapeConfig) -> dict[str, Any]:
    """Run the full landscape experiment with incremental saving.

    Outer loop: task -> optimizer -> algebra -> seed
    At each (task, optimizer, algebra, seed):
      1. Check if result already exists in output_dir (skip if so)
      2. Build model, build data, train
      3. Save result JSON immediately after training
      4. If seed in hessian_seeds, save model checkpoints at hessian_checkpoints

    After all training:
      5. Run Hessian analysis on saved checkpoints (if applicable)
      6. Run curvature measurement on converged models (if applicable)
      7. Collect gradient variance statistics

    Args:
        config: LandscapeConfig with all experiment parameters.

    Returns:
        Aggregated results dict keyed by task -> optimizer -> algebra ->
        seed -> metrics.
    """
    results: dict[str, Any] = {}
    total_runs = (
        len(config.tasks)
        * len(config.optimizers)
        * len(config.algebras)
        * len(config.seeds)
    )
    completed = 0
    skipped = 0

    start_time = time.time()

    for task_name in config.tasks:
        results.setdefault(task_name, {})

        # Build task data once per task (same data for all algebras/seeds)
        train_ds, test_ds, task_metadata = _build_task_data(task_name, config)
        loss_fn = _get_loss_fn(task_name)

        # Determine riemannian manifold variants
        # For riemannian_adam, run both sphere and stiefel per 05-04 decision
        def _expand_optimizers(optimizer_name: str) -> list[tuple[str, str]]:
            """Return (display_name, optimizer_name) pairs, expanding riemannian variants."""
            if optimizer_name == "riemannian_adam":
                return [
                    ("riemannian_adam_sphere", "riemannian_adam"),
                    ("riemannian_adam_stiefel", "riemannian_adam"),
                ]
            return [(optimizer_name, optimizer_name)]

        for opt_name in config.optimizers:
            for display_name, actual_opt in _expand_optimizers(opt_name):
                results[task_name].setdefault(display_name, {})

                # Determine manifold type for riemannian variants
                manifold_type = "sphere"
                if display_name == "riemannian_adam_stiefel":
                    manifold_type = "stiefel"

                for algebra in config.algebras:
                    alg_name = algebra.short_name
                    results[task_name][display_name].setdefault(alg_name, {})

                    for seed in config.seeds:
                        completed += 1

                        # Check if already completed (incremental resume)
                        if _result_exists(
                            config.output_dir, task_name, display_name, alg_name, seed
                        ):
                            skipped += 1
                            # Load existing result
                            existing = _load_result(
                                config.output_dir, task_name, display_name,
                                alg_name, seed,
                            )
                            results[task_name][display_name][alg_name][seed] = existing
                            continue

                        logger.info(
                            f"[{completed}/{total_runs}] "
                            f"{task_name}/{display_name}/{alg_name}/seed_{seed}"
                        )

                        # Seed everything
                        seed_everything(seed)

                        # Build model
                        model = _build_model(algebra, task_name, config)

                        # Save initial model state for Hessian checkpoint at 0.0
                        is_hessian_seed = seed in config.hessian_seeds
                        if is_hessian_seed and 0.0 in config.hessian_checkpoints:
                            _save_hessian_checkpoint(
                                config.output_dir, task_name, display_name,
                                alg_name, seed, 0.0, model,
                            )

                        # Build data loaders
                        train_config = _optimizer_train_config(
                            actual_opt, config, manifold_type=manifold_type,
                            is_hessian_seed=is_hessian_seed,
                        )
                        bs = min(train_config.batch_size, len(train_ds))
                        train_loader = DataLoader(
                            train_ds, batch_size=bs, shuffle=True, drop_last=True,
                        )
                        test_loader = DataLoader(
                            test_ds, batch_size=min(bs, len(test_ds)), shuffle=False,
                        )

                        # Create output directory for this run
                        run_dir = (
                            Path(config.output_dir) / task_name / display_name
                            / alg_name / f"seed_{seed}"
                        )
                        run_dir.mkdir(parents=True, exist_ok=True)

                        # Train
                        try:
                            metrics = train_model(
                                model=model,
                                train_loader=train_loader,
                                val_loader=test_loader,
                                config=train_config,
                                output_dir=str(run_dir),
                                device=config.device,
                                loss_fn=loss_fn,
                            )
                        except Exception as e:
                            logger.error(
                                f"Training failed for {task_name}/{display_name}/"
                                f"{alg_name}/seed_{seed}: {e}"
                            )
                            metrics = {
                                "error": str(e),
                                "train_losses": [],
                                "val_losses": [],
                                "best_val_loss": float("inf"),
                                "total_time_seconds": 0.0,
                                "epochs_trained": 0,
                            }

                        # Save Hessian checkpoints at intermediate + final fractions
                        if is_hessian_seed:
                            for frac in config.hessian_checkpoints:
                                if frac == 0.0:
                                    continue  # Already saved before training
                                if frac == 1.0:
                                    # Save converged model directly
                                    _save_hessian_checkpoint(
                                        config.output_dir, task_name, display_name,
                                        alg_name, seed, 1.0, model,
                                    )
                                else:
                                    # Extract intermediate checkpoint from trainer's saved checkpoints
                                    target_epoch = max(1, int(frac * config.epochs))
                                    trainer_ckpt = run_dir / f"checkpoint_epoch{target_epoch}.pt"
                                    if trainer_ckpt.exists():
                                        full_ckpt = torch.load(str(trainer_ckpt), weights_only=False)
                                        ckpt_dir = _hessian_checkpoint_dir(
                                            config.output_dir, task_name, display_name, alg_name, seed
                                        )
                                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                                        hessian_path = ckpt_dir / f"checkpoint_{frac:.2f}.pt"
                                        torch.save(full_ckpt["model_state_dict"], hessian_path)
                                        logger.info(f"Saved Hessian checkpoint at frac={frac:.2f} from epoch {target_epoch}")
                                    else:
                                        logger.warning(
                                            f"Trainer checkpoint at epoch {target_epoch} not found for "
                                            f"frac={frac:.2f}; expected {trainer_ckpt}"
                                        )

                        # Save result immediately (crash resilience)
                        run_result = {
                            "train_losses": metrics.get("train_losses", []),
                            "val_losses": metrics.get("val_losses", []),
                            "best_val_loss": metrics.get("best_val_loss", float("inf")),
                            "final_val_loss": (
                                metrics["val_losses"][-1]
                                if metrics.get("val_losses")
                                else float("inf")
                            ),
                            "epochs_trained": metrics.get("epochs_trained", 0),
                            "total_time_seconds": metrics.get("total_time_seconds", 0.0),
                        }
                        if "error" in metrics:
                            run_result["error"] = metrics["error"]

                        _save_result(
                            config.output_dir, task_name, display_name,
                            alg_name, seed, run_result,
                        )
                        results[task_name][display_name][alg_name][seed] = run_result

    elapsed = time.time() - start_time
    logger.info(
        f"Training complete: {completed} runs ({skipped} skipped) "
        f"in {elapsed:.1f}s"
    )

    # Post-training analysis is deferred for the full run (plan 05-06).
    # Here we just provide the training results for gate evaluation.

    return results
