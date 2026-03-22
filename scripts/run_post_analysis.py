"""Post-training analysis script for optimization landscape experiments.

Loads saved model checkpoints from completed experiment runs, computes:
- Hessian eigenspectrum at each checkpoint fraction (0.0, 0.25, 0.5, 1.0)
- Bill & Cox curvature on converged models
- Gradient variance across seeds

Results are written back to each seed's result.json in the format that
analyze_landscape.py expects.

Usage:
    python scripts/run_post_analysis.py --results-dir results/landscape
    python scripts/run_post_analysis.py --results-dir results/landscape --force --skip-gradient
    python scripts/run_post_analysis.py --results-dir results/landscape --device cpu --epochs 100
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from octonion.baselines._config import AlgebraType
from octonion.landscape._curvature import measure_curvature
from octonion.landscape._experiment import (
    LandscapeConfig,
    _build_model,
    _build_task_data,
    _get_loss_fn,
    _hessian_checkpoint_dir,
)
from octonion.landscape._gradient_stats import collect_gradient_variance_across_seeds
from octonion.landscape._hessian import compute_hessian_spectrum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Algebra type reverse lookup
# ---------------------------------------------------------------------------

_SHORT_TO_ALGEBRA: dict[str, AlgebraType] = {a.short_name: a for a in AlgebraType}


def _algebra_from_shortname(name: str) -> AlgebraType:
    """Map algebra short name to AlgebraType enum value.

    Args:
        name: Short name (e.g. "R", "C", "H", "O", "PHM8", "R8D").

    Returns:
        Corresponding AlgebraType enum value.

    Raises:
        ValueError: If name is not a valid short name.
    """
    if name not in _SHORT_TO_ALGEBRA:
        raise ValueError(
            f"Unknown algebra short name: {name!r}. "
            f"Valid: {list(_SHORT_TO_ALGEBRA.keys())}"
        )
    return _SHORT_TO_ALGEBRA[name]


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------


def _discover_runs(results_dir: str) -> list[dict[str, Any]]:
    """Discover all completed experiment runs from directory structure.

    Walks results_dir/{task}/{optimizer}/{algebra}/seed_{N}/ to find
    result.json files from completed training runs.

    Args:
        results_dir: Path to the results directory.

    Returns:
        List of dicts with keys: task, optimizer, algebra, seed, seed_dir,
        result_file.
    """
    runs: list[dict[str, Any]] = []
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.warning(f"Results directory does not exist: {results_dir}")
        return runs

    for task_dir in sorted(results_path.iterdir()):
        if not task_dir.is_dir():
            continue
        for opt_dir in sorted(task_dir.iterdir()):
            if not opt_dir.is_dir():
                continue
            for alg_dir in sorted(opt_dir.iterdir()):
                if not alg_dir.is_dir():
                    continue
                for seed_dir in sorted(alg_dir.iterdir()):
                    if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                        continue
                    result_file = seed_dir / "result.json"
                    if result_file.exists():
                        runs.append({
                            "task": task_dir.name,
                            "optimizer": opt_dir.name,
                            "algebra": alg_dir.name,
                            "seed": int(seed_dir.name.split("_")[1]),
                            "seed_dir": seed_dir,
                            "result_file": result_file,
                        })
    return runs


# ---------------------------------------------------------------------------
# JSON update helper
# ---------------------------------------------------------------------------


def _update_result_json(result_file: Path, updates: dict[str, Any]) -> None:
    """Read existing result.json, add/update keys, write back.

    Args:
        result_file: Path to the result.json file.
        updates: Dict of keys to add or update.
    """
    with open(result_file) as f:
        result = json.load(f)
    result.update(updates)
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)


# ---------------------------------------------------------------------------
# Core post-analysis
# ---------------------------------------------------------------------------


def run_post_analysis(
    results_dir: str,
    config: LandscapeConfig,
    device: str = "cpu",
    force: bool = False,
    skip_hessian: bool = False,
    skip_curvature: bool = False,
    skip_gradient: bool = False,
) -> dict[str, int]:
    """Run post-training analysis on saved checkpoints.

    Loads model checkpoints saved during training, computes Hessian
    eigenspectrum, Bill & Cox curvature, and gradient variance, then
    writes results back to each seed's result.json.

    Args:
        results_dir: Path to the results directory.
        config: LandscapeConfig matching the training configuration.
        device: Device for computation ('cpu' or 'cuda').
        force: If True, overwrite existing analysis results.
        skip_hessian: If True, skip Hessian eigenspectrum computation.
        skip_curvature: If True, skip curvature measurement.
        skip_gradient: If True, skip gradient variance collection.

    Returns:
        Dict with counts: hessian_computed, curvature_computed,
        gradient_computed, skipped, errors.
    """
    counts = {
        "hessian_computed": 0,
        "curvature_computed": 0,
        "gradient_computed": 0,
        "skipped": 0,
        "errors": 0,
    }

    runs = _discover_runs(results_dir)
    if not runs:
        logger.warning(f"No completed runs found in {results_dir}")
        return counts

    logger.info(f"Discovered {len(runs)} completed runs in {results_dir}")

    # -----------------------------------------------------------------------
    # Hessian analysis
    # -----------------------------------------------------------------------
    if not skip_hessian:
        hessian_runs = [
            r for r in runs if r["seed"] in config.hessian_seeds
        ]
        logger.info(f"Hessian analysis: {len(hessian_runs)} runs with hessian seeds")

        for run in hessian_runs:
            try:
                # Check if already computed
                with open(run["result_file"]) as f:
                    existing = json.load(f)
                if "hessian_spectrum" in existing and not force:
                    counts["skipped"] += 1
                    continue

                # Find available checkpoint files
                ckpt_dir = _hessian_checkpoint_dir(
                    results_dir, run["task"], run["optimizer"],
                    run["algebra"], run["seed"],
                )
                if not ckpt_dir.exists():
                    logger.warning(
                        f"No hessian checkpoint dir for "
                        f"{run['task']}/{run['optimizer']}/{run['algebra']}/seed_{run['seed']}"
                    )
                    continue

                # Build model and data
                algebra_enum = _algebra_from_shortname(run["algebra"])
                model = _build_model(algebra_enum, run["task"], config)
                loss_fn = _get_loss_fn(run["task"])

                _, test_ds, _ = _build_task_data(run["task"], config, seed=42)
                n_hessian = min(200, len(test_ds))
                hessian_x = test_ds.tensors[0][:n_hessian]
                hessian_y = test_ds.tensors[1][:n_hessian]

                hessian_spectrum: dict[str, list[float]] = {}

                for frac in config.hessian_checkpoints:
                    ckpt_path = ckpt_dir / f"checkpoint_{frac:.2f}.pt"
                    if not ckpt_path.exists():
                        logger.warning(
                            f"Checkpoint not found: {ckpt_path}"
                        )
                        continue

                    # Load checkpoint (raw state_dict format)
                    state_dict = torch.load(str(ckpt_path), weights_only=True)
                    model.load_state_dict(state_dict)
                    model = model.to(device)

                    # Compute Hessian spectrum
                    spectrum = compute_hessian_spectrum(
                        model, loss_fn,
                        hessian_x.to(device), hessian_y.to(device),
                        device=device,
                    )

                    # Extract eigenvalues and convert to list
                    key = "eigenvalues" if "eigenvalues" in spectrum else "ritz_values"
                    eigenvalues = spectrum[key]
                    if isinstance(eigenvalues, np.ndarray):
                        eigenvalues = eigenvalues.tolist()

                    frac_key = f"{frac:.1f}"
                    hessian_spectrum[frac_key] = eigenvalues

                    logger.info(
                        f"Hessian at frac={frac:.1f}: {len(eigenvalues)} eigenvalues "
                        f"({run['task']}/{run['optimizer']}/{run['algebra']}/seed_{run['seed']})"
                    )

                if hessian_spectrum:
                    _update_result_json(run["result_file"], {
                        "hessian_spectrum": hessian_spectrum,
                    })
                    counts["hessian_computed"] += 1

            except Exception as e:
                logger.error(
                    f"Hessian analysis failed for "
                    f"{run['task']}/{run['optimizer']}/{run['algebra']}/seed_{run['seed']}: {e}"
                )
                counts["errors"] += 1

    # -----------------------------------------------------------------------
    # Curvature measurement
    # -----------------------------------------------------------------------
    if not skip_curvature:
        logger.info(f"Curvature analysis: processing {len(runs)} runs")

        for run in runs:
            try:
                # Check if already computed
                with open(run["result_file"]) as f:
                    existing = json.load(f)
                if "curvature" in existing and not force:
                    counts["skipped"] += 1
                    continue

                # Need converged checkpoint (frac=1.0)
                ckpt_dir = _hessian_checkpoint_dir(
                    results_dir, run["task"], run["optimizer"],
                    run["algebra"], run["seed"],
                )
                converged_ckpt = ckpt_dir / "checkpoint_1.00.pt"

                if not converged_ckpt.exists():
                    # Not a hessian seed or no checkpoint saved -- skip
                    continue

                # Build model and data
                algebra_enum = _algebra_from_shortname(run["algebra"])
                model = _build_model(algebra_enum, run["task"], config)
                loss_fn = _get_loss_fn(run["task"])

                _, test_ds, _ = _build_task_data(run["task"], config, seed=42)
                n_curv = min(200, len(test_ds))
                data_x = test_ds.tensors[0][:n_curv]
                data_y = test_ds.tensors[1][:n_curv]

                # Load converged model
                state_dict = torch.load(str(converged_ckpt), weights_only=True)
                model.load_state_dict(state_dict)
                model = model.to(device)

                # Measure curvature
                curv_result = measure_curvature(
                    model, loss_fn,
                    data_x.to(device), data_y.to(device),
                    n_directions=config.n_curvature_directions,
                )

                # Convert curvatures list items for JSON safety
                curvature_detail = {
                    k: v if not isinstance(v, np.ndarray) else v.tolist()
                    for k, v in curv_result.items()
                }

                _update_result_json(run["result_file"], {
                    "curvature": curv_result["mean_curvature"],
                    "curvature_detail": curvature_detail,
                })
                counts["curvature_computed"] += 1

                logger.info(
                    f"Curvature: mean={curv_result['mean_curvature']:.4f} "
                    f"({run['task']}/{run['optimizer']}/{run['algebra']}/seed_{run['seed']})"
                )

            except Exception as e:
                logger.error(
                    f"Curvature analysis failed for "
                    f"{run['task']}/{run['optimizer']}/{run['algebra']}/seed_{run['seed']}: {e}"
                )
                counts["errors"] += 1

    # -----------------------------------------------------------------------
    # Gradient variance collection
    # -----------------------------------------------------------------------
    if not skip_gradient:
        # Collect unique (task, optimizer, algebra) combinations
        task_alg_combos: set[tuple[str, str, str]] = set()
        for run in runs:
            task_alg_combos.add((run["task"], run["optimizer"], run["algebra"]))

        logger.info(
            f"Gradient variance: {len(task_alg_combos)} (task, optimizer, algebra) combinations"
        )

        for task_name, opt_name, alg_name in sorted(task_alg_combos):
            try:
                # Check if gradient_variance file exists
                grad_var_path = (
                    Path(results_dir) / task_name / opt_name
                    / f"gradient_variance_{alg_name}.json"
                )
                if grad_var_path.exists() and not force:
                    counts["skipped"] += 1
                    continue

                algebra_enum = _algebra_from_shortname(alg_name)
                loss_fn = _get_loss_fn(task_name)

                _, test_ds, _ = _build_task_data(task_name, config, seed=42)
                n_grad = min(200, len(test_ds))
                data_x = test_ds.tensors[0][:n_grad].to(device)
                data_y = test_ds.tensors[1][:n_grad].to(device)

                # Model factory
                def model_factory(
                    _alg=algebra_enum, _task=task_name, _cfg=config, _dev=device
                ):
                    return _build_model(_alg, _task, _cfg).to(_dev)

                # Collect gradient variance
                grad_result = collect_gradient_variance_across_seeds(
                    model_factory=model_factory,
                    loss_fn=loss_fn,
                    data_x=data_x,
                    data_y=data_y,
                    seeds=config.seeds[:5],
                    n_steps=10,
                    device=device,
                )

                cross_seed_variance = grad_result["cross_seed_variance"]
                grad_norm_std = math.sqrt(cross_seed_variance) if cross_seed_variance > 0 else 0.0

                # Save aggregate gradient variance file
                grad_var_path.parent.mkdir(parents=True, exist_ok=True)
                grad_var_serializable = {
                    "cross_seed_variance": cross_seed_variance,
                    "grad_norm_std": grad_norm_std,
                    "mean_grad_norm_trajectory": grad_result["mean_grad_norm_trajectory"],
                }
                with open(grad_var_path, "w") as f:
                    json.dump(grad_var_serializable, f, indent=2)

                # Update each seed's result.json with gradient_stats
                matching_runs = [
                    r for r in runs
                    if r["task"] == task_name and r["optimizer"] == opt_name
                    and r["algebra"] == alg_name
                ]
                for run in matching_runs:
                    try:
                        # Compute per-seed gradient norm mean from the seed's
                        # position in the gradient variance data
                        seed_idx = None
                        for i, s in enumerate(config.seeds[:5]):
                            if s == run["seed"]:
                                seed_idx = i
                                break

                        grad_norm_mean = 0.0
                        if seed_idx is not None and seed_idx < len(grad_result["per_seed_stats"]):
                            seed_stats = grad_result["per_seed_stats"][seed_idx]
                            if seed_stats:
                                grad_norm_mean = sum(
                                    s["grad_norm_mean"] for s in seed_stats
                                ) / len(seed_stats)

                        _update_result_json(run["result_file"], {
                            "gradient_stats": {
                                "grad_norm_std": grad_norm_std,
                                "grad_norm_mean": grad_norm_mean,
                                "cross_seed_variance": cross_seed_variance,
                            },
                        })
                    except Exception as e:
                        logger.warning(
                            f"Failed to update gradient_stats for "
                            f"{run['task']}/{run['optimizer']}/{run['algebra']}/seed_{run['seed']}: {e}"
                        )

                counts["gradient_computed"] += 1
                logger.info(
                    f"Gradient variance: std={grad_norm_std:.4f} "
                    f"({task_name}/{opt_name}/{alg_name})"
                )

            except Exception as e:
                logger.error(
                    f"Gradient variance failed for "
                    f"{task_name}/{opt_name}/{alg_name}: {e}"
                )
                counts["errors"] += 1

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for post-training analysis."""
    parser = argparse.ArgumentParser(
        description="Run post-training analysis (Hessian, curvature, gradient variance) "
        "on saved experiment checkpoints.",
    )
    parser.add_argument(
        "--results-dir", default="results/landscape",
        help="Path to the results directory (default: results/landscape)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for computation (default: cuda)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing analysis results",
    )
    parser.add_argument(
        "--skip-hessian", action="store_true",
        help="Skip Hessian eigenspectrum computation",
    )
    parser.add_argument(
        "--skip-curvature", action="store_true",
        help="Skip curvature measurement",
    )
    parser.add_argument(
        "--skip-gradient", action="store_true",
        help="Skip gradient variance collection",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (must match training config, default: 100)",
    )
    parser.add_argument(
        "--base-hidden", type=int, default=16,
        help="Base hidden size (must match training config, default: 16)",
    )
    parser.add_argument(
        "--n-train", type=int, default=50_000,
        help="Number of training samples (must match training config, default: 50000)",
    )
    parser.add_argument(
        "--n-test", type=int, default=10_000,
        help="Number of test samples (must match training config, default: 10000)",
    )
    parser.add_argument(
        "--hessian-seeds", type=str, default="0,4,9,14,19",
        help="Comma-separated list of Hessian seed indices (default: 0,4,9,14,19)",
    )
    parser.add_argument(
        "--n-curvature-directions", type=int, default=50,
        help="Number of random directions for curvature measurement (default: 50)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Build config from CLI args
    hessian_seeds = [int(s.strip()) for s in args.hessian_seeds.split(",")]
    config = LandscapeConfig(
        epochs=args.epochs,
        base_hidden=args.base_hidden,
        n_train=args.n_train,
        n_test=args.n_test,
        hessian_seeds=hessian_seeds,
        n_curvature_directions=args.n_curvature_directions,
    )

    # Run analysis
    counts = run_post_analysis(
        results_dir=args.results_dir,
        config=config,
        device=args.device,
        force=args.force,
        skip_hessian=args.skip_hessian,
        skip_curvature=args.skip_curvature,
        skip_gradient=args.skip_gradient,
    )

    # Print summary
    print("\n--- Post-Analysis Summary ---")
    print(f"Hessian spectrum computed: {counts['hessian_computed']}")
    print(f"Curvature computed:       {counts['curvature_computed']}")
    print(f"Gradient variance computed: {counts['gradient_computed']}")
    print(f"Skipped (already exists): {counts['skipped']}")
    print(f"Errors:                   {counts['errors']}")

    if counts["errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
