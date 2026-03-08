"""Comparison runner for multi-algebra multi-seed experiments.

Orchestrates parameter-matched training of all algebra networks across
multiple seeds, computes pairwise statistical significance, generates
structured experiment directories with auto-manifest, and produces
auto-generated plots and reports.

Provides:
- ComparisonReport: Serializable dataclass with all experiment results
- run_comparison: Central experiment harness for Phase 5/7 downstream use
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np

from octonion.baselines._config import (
    AlgebraType,
    ComparisonConfig,
    NetworkConfig,
    TrainConfig,
)
from octonion.baselines._param_matching import (
    _build_simple_mlp,
    find_matched_width,
    param_report,
)
from octonion.baselines._plotting import (
    plot_comparison_bars,
    plot_convergence,
    plot_param_table,
)
from octonion.baselines._stats import holm_bonferroni, paired_comparison
from octonion.baselines._trainer import seed_everything, train_model

logger = logging.getLogger(__name__)


@dataclass
class ComparisonReport:
    """Structured report from a multi-algebra comparison experiment.

    All fields are JSON-serializable for provenance tracking.

    Attributes:
        task: Task/experiment name.
        algebras: List of algebra short names compared.
        seeds: Number of random seeds per algebra.
        per_run: List of per-run results, each with algebra, seed, metrics.
        pairwise: Dict of pairwise comparison results keyed by "A_vs_B".
        corrected_pairwise: Holm-Bonferroni corrected pairwise p-values.
        param_counts: Per-algebra total trainable parameter count.
        config_hash: SHA256 hash of the full config for reproducibility.
        timestamp: ISO 8601 timestamp of experiment completion.
    """

    task: str
    algebras: list[str]
    seeds: int
    per_run: list[dict[str, Any]]
    pairwise: dict[str, dict[str, Any]]
    corrected_pairwise: list[dict[str, Any]]
    param_counts: dict[str, int]
    config_hash: str
    timestamp: str


def _config_hash(config: ComparisonConfig) -> str:
    """Compute SHA256 hash of ComparisonConfig for reproducibility tracking.

    Uses deterministic JSON serialization of the config dataclass fields.

    Args:
        config: Comparison config to hash.

    Returns:
        Hex string of SHA256 hash.
    """
    config_dict = {
        "task": config.task,
        "algebras": [a.short_name for a in config.algebras],
        "seeds": config.seeds,
        "train_config": {
            "epochs": config.train_config.epochs,
            "lr": config.train_config.lr,
            "optimizer": config.train_config.optimizer,
            "scheduler": config.train_config.scheduler,
            "weight_decay": config.train_config.weight_decay,
            "early_stopping_patience": config.train_config.early_stopping_patience,
            "warmup_epochs": config.train_config.warmup_epochs,
            "use_amp": config.train_config.use_amp,
            "checkpoint_every": config.train_config.checkpoint_every,
            "seed": config.train_config.seed,
            "batch_size": config.train_config.batch_size,
            "lock_optimizer": config.train_config.lock_optimizer,
        },
        "output_dir": config.output_dir,
    }
    config_json = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(config_json.encode()).hexdigest()


def _update_manifest(
    output_dir: str,
    task_name: str,
    report: ComparisonReport,
) -> None:
    """Update the experiment manifest with a new entry.

    Loads existing manifest.json or creates a new one, appends
    the entry for this experiment run, and saves back.

    Args:
        output_dir: Base experiments directory.
        task_name: Name of the task/experiment.
        report: Completed ComparisonReport.
    """
    manifest_path = Path(output_dir) / "manifest.json"

    # Load existing or create new
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = []

    # Build summary metrics
    per_algebra_accs: dict[str, list[float]] = {}
    for run in report.per_run:
        alg = run["algebra"]
        acc = run["metrics"]["best_val_acc"]
        per_algebra_accs.setdefault(alg, []).append(acc)

    metrics_summary = {}
    for alg, accs in per_algebra_accs.items():
        metrics_summary[alg] = {
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        }

    entry = {
        "task_name": task_name,
        "config_hash": report.config_hash,
        "timestamp": report.timestamp,
        "algebras": report.algebras,
        "seeds": report.seeds,
        "param_counts": report.param_counts,
        "metrics_summary": metrics_summary,
        "status": "complete",
    }
    manifest.append(entry)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def run_comparison(
    task_name: str,
    build_data_fn: Callable[..., tuple],
    config: ComparisonConfig,
    device: str = "cuda",
    network_config_overrides: dict[str, Any] | None = None,
) -> ComparisonReport:
    """Orchestrate a full multi-algebra multi-seed comparison experiment.

    Trains parameter-matched networks for each algebra across multiple
    seeds, computes pairwise statistical significance, generates plots,
    and manages experiment provenance.

    Args:
        task_name: Experiment name (e.g., "cifar10_conv2d").
        build_data_fn: Callable returning
            (train_loader, val_loader, test_loader, input_dim, output_dim, input_channels).
            Accepts batch_size as first argument.
        config: ComparisonConfig with algebras, seeds, train_config, output_dir.
        device: "cuda" or "cpu".
        network_config_overrides: Optional overrides for NetworkConfig fields
            (e.g., depth, topology, base_hidden, ref_hidden).
            ref_hidden: algebra-unit hidden width for the reference model
            (default 20, used to compute target param count).

    Returns:
        ComparisonReport with all experiment results.

    Raises:
        ValueError: If parameter counts differ by more than 1% across algebras.
    """
    task_dir = Path(config.output_dir) / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    # Compute config hash
    cfg_hash = _config_hash(config)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Network config overrides
    nc_overrides = network_config_overrides or {}
    depth = nc_overrides.get("depth", 1)
    topology = nc_overrides.get("topology", "mlp")
    # ref_hidden is the algebra-unit hidden width for the reference algebra
    # (first algebra in the list). This width is passed directly to
    # _build_simple_mlp, so it represents algebra units, not real units.
    ref_hidden = nc_overrides.get("ref_hidden", 20)

    # Build data to get dimensions
    train_loader, val_loader, test_loader, input_dim, output_dim, input_channels = (
        build_data_fn(config.train_config.batch_size)
    )

    # ── Step 1: Build reference model to get target param count ──
    # Use the first algebra in the list as reference
    ref_algebra = config.algebras[0]
    ref_model = _build_simple_mlp(
        algebra=ref_algebra,
        hidden=ref_hidden,
        depth=depth,
        input_dim=input_dim,
        output_dim=output_dim,
    )
    target_params = sum(p.numel() for p in ref_model.parameters())
    logger.info(
        f"Reference: {ref_algebra.short_name} hidden={ref_hidden} "
        f"target={target_params} params"
    )

    # ── Step 2: Find matched widths for each algebra ──
    matched_widths: dict[str, int] = {}
    param_counts: dict[str, int] = {}

    for algebra in config.algebras:
        width = find_matched_width(
            target_params=target_params,
            algebra=algebra,
            topology="mlp",
            depth=depth,
            tolerance=0.01,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        matched_widths[algebra.short_name] = width

        # Verify actual param count
        test_model = _build_simple_mlp(
            algebra=algebra,
            hidden=width,
            depth=depth,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        actual = sum(p.numel() for p in test_model.parameters())
        param_counts[algebra.short_name] = actual

    # ── Step 3: Verify param counts within 1% tolerance ──
    ref_count = param_counts[ref_algebra.short_name]
    for alg_name, count in param_counts.items():
        relative_diff = abs(count - ref_count) / ref_count
        if relative_diff > 0.01:
            raise ValueError(
                f"Parameter count mismatch: {alg_name} has {count} params "
                f"({relative_diff * 100:.2f}% from reference {ref_count}). "
                f"All algebras must be within 1%."
            )

    logger.info(f"Parameter counts: {param_counts}")

    # ── Step 4: Save model summaries ──
    for algebra in config.algebras:
        alg_name = algebra.short_name
        algebra_dir = task_dir / algebra.name
        algebra_dir.mkdir(parents=True, exist_ok=True)

        try:
            import torchinfo

            width = matched_widths[alg_name]
            summary_model = _build_simple_mlp(
                algebra=algebra,
                hidden=width,
                depth=depth,
                input_dim=input_dim,
                output_dim=output_dim,
            )
            summary_str = str(
                torchinfo.summary(
                    summary_model,
                    input_size=(1, input_dim),
                    device="cpu",
                    verbose=0,
                )
            )
            summary_path = algebra_dir / "model_summary.txt"
            with open(summary_path, "w") as f:
                f.write(summary_str)
        except Exception as e:
            logger.warning(f"Could not save model summary for {alg_name}: {e}")

    # ── Step 5: Train all algebra/seed combinations ──
    per_run: list[dict[str, Any]] = []

    for algebra in config.algebras:
        alg_name = algebra.short_name
        width = matched_widths[alg_name]

        for seed_idx in range(config.seeds):
            seed_everything(seed_idx)

            # Build fresh data loaders for each run (seeded)
            train_loader, val_loader, test_loader, _, _, _ = build_data_fn(
                config.train_config.batch_size
            )

            # Build model using _build_simple_mlp for consistent param matching
            model = _build_simple_mlp(
                algebra=algebra,
                hidden=width,
                depth=depth,
                input_dim=input_dim,
                output_dim=output_dim,
            )

            # Create experiment directory
            experiment_dir = task_dir / algebra.name / str(seed_idx)
            experiment_dir.mkdir(parents=True, exist_ok=True)

            # Save config.json
            config_dict = {
                "train_config": {
                    "epochs": config.train_config.epochs,
                    "lr": config.train_config.lr,
                    "optimizer": config.train_config.optimizer,
                    "scheduler": config.train_config.scheduler,
                    "weight_decay": config.train_config.weight_decay,
                    "batch_size": config.train_config.batch_size,
                },
                "network_config": {
                    "algebra": alg_name,
                    "topology": topology,
                    "depth": depth,
                    "hidden_width": width,
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                },
                "seed": seed_idx,
            }
            with open(experiment_dir / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)

            # Train
            metrics = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config.train_config,
                output_dir=str(experiment_dir),
                device=device,
            )

            # Save metrics.json
            metrics_serializable = {
                k: v for k, v in metrics.items() if k != "lr_history"
            }
            metrics_serializable["lr_history"] = [
                float(lr) for lr in metrics.get("lr_history", [])
            ]
            with open(experiment_dir / "metrics.json", "w") as f:
                json.dump(metrics_serializable, f, indent=2)

            # Generate convergence plot
            plot_convergence(
                metrics, str(experiment_dir / "convergence.png")
            )

            # Collect run result
            per_run.append({
                "algebra": alg_name,
                "seed": seed_idx,
                "metrics": metrics,
            })

            logger.info(
                f"  {alg_name} seed {seed_idx}: "
                f"val_acc={metrics['best_val_acc']:.4f}, "
                f"time={metrics['total_time_seconds']:.1f}s"
            )

    # ── Step 6: Compute pairwise statistical comparisons ──
    # Collect per-algebra accuracy lists
    algebra_accs: dict[str, list[float]] = {}
    for run in per_run:
        alg = run["algebra"]
        acc = run["metrics"]["best_val_acc"]
        algebra_accs.setdefault(alg, []).append(acc)

    # Compute pairwise comparisons for all algebra pairs
    pairwise: dict[str, dict[str, Any]] = {}
    algebra_names = [a.short_name for a in config.algebras]

    for a_name, b_name in combinations(algebra_names, 2):
        key = f"{a_name}_vs_{b_name}"
        pairwise[key] = paired_comparison(
            algebra_accs[a_name], algebra_accs[b_name]
        )

    # ── Step 7: Apply Holm-Bonferroni correction ──
    raw_p_values = [p["t_p_value"] for p in pairwise.values()]
    if raw_p_values:
        corrected = holm_bonferroni(raw_p_values)
    else:
        corrected = []

    # ── Step 8: Generate comparison plots ──
    plot_comparison_bars(
        algebra_accs,
        "Best Validation Accuracy",
        str(task_dir / "comparison_accuracy.png"),
    )

    # Generate param table plot
    param_reports: dict[str, list[dict[str, Any]]] = {}
    for algebra in config.algebras:
        alg_name = algebra.short_name
        width = matched_widths[alg_name]
        model = _build_simple_mlp(
            algebra=algebra,
            hidden=width,
            depth=depth,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        param_reports[alg_name] = param_report(model)

    plot_param_table(param_reports, str(task_dir / "param_table.png"))

    # ── Step 9: Build and return report ──
    report = ComparisonReport(
        task=task_name,
        algebras=algebra_names,
        seeds=config.seeds,
        per_run=per_run,
        pairwise=pairwise,
        corrected_pairwise=corrected,
        param_counts=param_counts,
        config_hash=cfg_hash,
        timestamp=timestamp,
    )

    # ── Step 10: Update manifest ──
    _update_manifest(config.output_dir, task_name, report)

    logger.info(f"Comparison complete: {task_name}")
    return report
