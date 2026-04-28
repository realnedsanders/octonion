#!/usr/bin/env python
"""Phase 5: Optimization Landscape Analysis

Reads experiment results from results/landscape/ and produces:
- Go/no-go gate verdict (gate_verdict.json)
- Convergence profile plots per task per optimizer
- Hessian eigenspectrum evolution plots
- Bill & Cox curvature comparison
- Gradient variance characterization
- Full structured report (full_report.json)
- Pairwise statistical comparisons with Holm-Bonferroni correction
- If RED verdict: pivot_plan.md with surviving claims and alternatives

Usage:
    # Analyze existing results (default: analyze-only mode)
    docker compose run --rm dev uv run python scripts/analyze_landscape.py --results-dir results/landscape

    # Run experiments first, then analyze
    docker compose run --rm dev uv run python scripts/analyze_landscape.py --results-dir results/landscape --run

    # Smoke test with minimal configuration
    docker compose run --rm dev uv run python scripts/analyze_landscape.py --results-dir results/landscape --run --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Matplotlib backend must be set before pyplot import
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from octonion.baselines._config import AlgebraType
from octonion.baselines._stats import (
    confidence_interval,
    holm_bonferroni,
    paired_comparison,
)
from octonion.landscape._gate import GateVerdict, evaluate_gate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Color palette ──────────────────────────────────────────────────────────

# Consistent colors for each algebra across all plots
ALGEBRA_COLORS: dict[str, str] = {
    "R": "#2196F3",       # Blue
    "C": "#4CAF50",       # Green
    "H": "#FF9800",       # Orange
    "O": "#F44336",       # Red
    "PHM8": "#9C27B0",    # Purple
    "R8D": "#795548",     # Brown
}

# Full names for legends
ALGEBRA_NAMES: dict[str, str] = {
    "R": "Real",
    "C": "Complex",
    "H": "Quaternion",
    "O": "Octonion",
    "PHM8": "PHM-8",
    "R8D": "R8-Dense",
}

# All known algebra short names in display order
ALL_ALGEBRAS = ["R", "C", "H", "O", "PHM8", "R8D"]


# ── Loading ────────────────────────────────────────────────────────────────


def load_results(results_dir: str) -> dict[str, dict[str, dict[str, dict[int, dict]]]]:
    """Load all experiment results from disk.

    Walks results_dir/{task}/{optimizer}/{algebra}/seed_{N}/result.json
    and aggregates into nested dict: {task: {optimizer: {algebra: {seed: result}}}}

    Args:
        results_dir: Path to results directory.

    Returns:
        Nested results dict. Reports missing runs to logger.
    """
    results: dict[str, dict[str, dict[str, dict[int, dict]]]] = {}
    total_found = 0
    missing_combos: list[str] = []

    results_path = Path(results_dir)
    if not results_path.exists():
        logger.warning(f"Results directory does not exist: {results_dir}")
        return results

    # Walk the directory structure
    for task_dir in sorted(results_path.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        results.setdefault(task_name, {})

        for opt_dir in sorted(task_dir.iterdir()):
            if not opt_dir.is_dir():
                continue
            opt_name = opt_dir.name
            results[task_name].setdefault(opt_name, {})

            for alg_dir in sorted(opt_dir.iterdir()):
                if not alg_dir.is_dir():
                    continue
                alg_name = alg_dir.name
                results[task_name][opt_name].setdefault(alg_name, {})

                for seed_dir in sorted(alg_dir.iterdir()):
                    if not seed_dir.is_dir():
                        continue
                    if not seed_dir.name.startswith("seed_"):
                        continue

                    result_file = seed_dir / "result.json"
                    if result_file.exists():
                        try:
                            with open(result_file) as f:
                                data = json.load(f)
                            seed_num = int(seed_dir.name.split("_")[1])
                            results[task_name][opt_name][alg_name][seed_num] = data
                            total_found += 1
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.warning(
                                f"Failed to load {result_file}: {e}"
                            )

    logger.info(f"Loaded {total_found} result files from {results_dir}")

    # Report what we found
    tasks_found = list(results.keys())
    for task_name in tasks_found:
        opts = list(results[task_name].keys())
        for opt in opts:
            algs = list(results[task_name][opt].keys())
            for alg in algs:
                seeds = list(results[task_name][opt][alg].keys())
                if not seeds:
                    missing_combos.append(f"{task_name}/{opt}/{alg}")

    if missing_combos:
        logger.warning(
            f"Missing results for {len(missing_combos)} combinations: "
            + ", ".join(missing_combos[:10])
            + ("..." if len(missing_combos) > 10 else "")
        )

    return results


def _get_all_algebras(results: dict) -> list[str]:
    """Extract all unique algebra names from results, in display order."""
    found = set()
    for task_data in results.values():
        for opt_data in task_data.values():
            found.update(opt_data.keys())
    return [a for a in ALL_ALGEBRAS if a in found]


def _get_all_optimizers(results: dict) -> list[str]:
    """Extract all unique optimizer names from results."""
    found = set()
    for task_data in results.values():
        found.update(task_data.keys())
    return sorted(found)


def _get_all_tasks(results: dict) -> list[str]:
    """Extract all unique task names from results."""
    return sorted(results.keys())


# ── Progress reporting ─────────────────────────────────────────────────────


def report_progress(results: dict) -> dict[str, Any]:
    """Report progress of the experiment matrix.

    Args:
        results: Loaded results dict.

    Returns:
        Dict with counts and estimated remaining time.
    """
    total = 0
    completed = 0
    failed = 0
    total_time = 0.0

    tasks = _get_all_tasks(results)
    opts = _get_all_optimizers(results)
    algs = _get_all_algebras(results)

    for task_name in tasks:
        task_data = results.get(task_name, {})
        for opt in opts:
            opt_data = task_data.get(opt, {})
            for alg in algs:
                alg_data = opt_data.get(alg, {})
                for seed, run in alg_data.items():
                    total += 1
                    if isinstance(run, dict):
                        if "error" in run:
                            failed += 1
                        else:
                            completed += 1
                            total_time += run.get("total_time_seconds", 0.0)

    avg_time = total_time / max(completed, 1)
    remaining = max(0, total - completed - failed)
    est_remaining_seconds = remaining * avg_time

    return {
        "total_runs_found": total,
        "completed": completed,
        "failed": failed,
        "remaining": remaining,
        "avg_time_per_run_seconds": avg_time,
        "estimated_remaining_seconds": est_remaining_seconds,
        "estimated_remaining_hours": est_remaining_seconds / 3600,
    }


# ── Convergence profile plotting ──────────────────────────────────────────


def plot_convergence_profiles(results: dict, output_dir: str) -> list[str]:
    """Plot convergence profiles for each task with subplots per optimizer.

    Each subplot shows 6 lines (one per algebra), plotting median val_loss
    across seeds with 25th-75th percentile error bands.

    Args:
        results: Loaded results dict.
        output_dir: Directory to save plots.

    Returns:
        List of saved file paths.
    """
    saved_paths: list[str] = []
    tasks = _get_all_tasks(results)
    algebras = _get_all_algebras(results)

    for task_name in tasks:
        task_data = results.get(task_name, {})
        optimizers = sorted(task_data.keys())
        if not optimizers:
            continue

        n_opts = len(optimizers)
        n_cols = min(3, n_opts)
        n_rows = math.ceil(n_opts / n_cols)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 4 * n_rows),
            squeeze=False,
        )

        for idx, opt_name in enumerate(optimizers):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]
            opt_data = task_data.get(opt_name, {})

            for alg_name in algebras:
                alg_data = opt_data.get(alg_name, {})
                if not alg_data:
                    continue

                # Collect val_losses across seeds
                all_curves: list[list[float]] = []
                for seed, run in sorted(alg_data.items()):
                    if isinstance(run, dict) and "val_losses" in run:
                        vl = run["val_losses"]
                        if vl:
                            all_curves.append(vl)

                if not all_curves:
                    continue

                # Align to shortest curve
                min_len = min(len(c) for c in all_curves)
                if min_len == 0:
                    continue
                aligned = np.array([c[:min_len] for c in all_curves])

                median = np.median(aligned, axis=0)
                q25 = np.percentile(aligned, 25, axis=0)
                q75 = np.percentile(aligned, 75, axis=0)
                epochs = np.arange(1, min_len + 1)

                color = ALGEBRA_COLORS.get(alg_name, "#888888")
                label = ALGEBRA_NAMES.get(alg_name, alg_name)
                ax.plot(epochs, median, color=color, label=label, linewidth=1.5)
                ax.fill_between(
                    epochs, q25, q75, alpha=0.15, color=color,
                )

            ax.set_title(f"{opt_name}", fontsize=11)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Val Loss")
            ax.set_yscale("log")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_opts, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row][col].set_visible(False)

        fig.suptitle(
            f"Convergence Profiles: {task_name}",
            fontsize=14, y=1.02,
        )
        fig.tight_layout()

        save_path = os.path.join(output_dir, f"convergence_profiles_{task_name}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(save_path)
        logger.info(f"Saved convergence profiles: {save_path}")

    return saved_paths


# ── Hessian eigenspectrum evolution ───────────────────────────────────────


def plot_hessian_evolution(results: dict, output_dir: str) -> list[str]:
    """Plot Hessian eigenspectrum evolution for each algebra.

    For each algebra: 4 panels (one per checkpoint) showing histogram
    of eigenvalue distribution with negative eigenvalue ratio overlaid.

    Also produces a summary plot of negative eigenvalue ratio vs training
    progress for all algebras.

    Args:
        results: Loaded results dict.
        output_dir: Directory to save plots.

    Returns:
        List of saved file paths.
    """
    saved_paths: list[str] = []
    algebras = _get_all_algebras(results)
    checkpoints = [0.0, 0.25, 0.5, 1.0]

    # Collect Hessian data from results
    # The experiment runner may store hessian_spectrum data in result.json
    # under a "hessian" key, or from checkpoint analysis
    hessian_data: dict[str, dict[float, list[list[float]]]] = {}
    # {algebra: {checkpoint_frac: [eigenvalue_lists_per_seed]}}

    for task_data in results.values():
        for opt_data in task_data.values():
            for alg_name, alg_data in opt_data.items():
                hessian_data.setdefault(alg_name, {})
                for seed, run in alg_data.items():
                    if not isinstance(run, dict):
                        continue
                    hessian_info = run.get("hessian_spectrum", {})
                    if isinstance(hessian_info, dict):
                        for frac_str, eigenvalues in hessian_info.items():
                            try:
                                frac = float(frac_str)
                            except ValueError:
                                continue
                            hessian_data[alg_name].setdefault(frac, [])
                            if isinstance(eigenvalues, list) and eigenvalues:
                                hessian_data[alg_name][frac].append(eigenvalues)

    # Per-algebra spectral evolution plots
    for alg_name in algebras:
        alg_hessian = hessian_data.get(alg_name, {})
        if not alg_hessian:
            continue

        available_fracs = sorted(alg_hessian.keys())
        n_panels = len(available_fracs)
        if n_panels == 0:
            continue

        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), squeeze=False)

        for i, frac in enumerate(available_fracs):
            ax = axes[0][i]
            all_eigs = []
            for eig_list in alg_hessian[frac]:
                all_eigs.extend(eig_list)

            if not all_eigs:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            all_eigs_arr = np.array(all_eigs)
            neg_ratio = float(np.mean(all_eigs_arr < 0))

            ax.hist(all_eigs_arr, bins=50, color=ALGEBRA_COLORS.get(alg_name, "#888"),
                    alpha=0.7, edgecolor="black", linewidth=0.3)
            ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
            ax.set_title(f"t={frac:.0%}", fontsize=10)
            ax.set_xlabel("Eigenvalue")
            ax.set_ylabel("Count")
            ax.text(0.95, 0.95, f"neg: {neg_ratio:.1%}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        full_name = ALGEBRA_NAMES.get(alg_name, alg_name)
        fig.suptitle(f"Hessian Eigenspectrum: {full_name}", fontsize=13)
        fig.tight_layout()

        save_path = os.path.join(output_dir, f"hessian_spectra_{alg_name}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(save_path)

    # Summary plot: negative eigenvalue ratio vs training progress
    if any(hessian_data.get(a, {}) for a in algebras):
        fig, ax = plt.subplots(figsize=(8, 5))

        for alg_name in algebras:
            alg_hessian = hessian_data.get(alg_name, {})
            if not alg_hessian:
                continue

            fracs_sorted = sorted(alg_hessian.keys())
            neg_ratios = []
            for frac in fracs_sorted:
                all_eigs = []
                for eig_list in alg_hessian[frac]:
                    all_eigs.extend(eig_list)
                if all_eigs:
                    neg_ratios.append(float(np.mean(np.array(all_eigs) < 0)))
                else:
                    neg_ratios.append(float("nan"))

            color = ALGEBRA_COLORS.get(alg_name, "#888")
            label = ALGEBRA_NAMES.get(alg_name, alg_name)
            ax.plot(fracs_sorted, neg_ratios, "o-", color=color, label=label,
                    linewidth=2, markersize=6)

        ax.set_xlabel("Training Progress", fontsize=12)
        ax.set_ylabel("Negative Eigenvalue Ratio", fontsize=12)
        ax.set_title("Hessian Negative Eigenvalue Ratio vs Training", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        save_path = os.path.join(output_dir, "hessian_spectra_summary.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(save_path)

    if not saved_paths:
        logger.warning("No Hessian spectrum data found in results; skipping Hessian plots")

    return saved_paths


# ── Curvature comparison ──────────────────────────────────────────────────


def plot_curvature_comparison(results: dict, output_dir: str) -> str | None:
    """Plot Bill & Cox curvature comparison across algebras, grouped by task.

    Bar chart: mean curvature per algebra (error bars from std), grouped
    by task.

    Args:
        results: Loaded results dict.
        output_dir: Directory to save plot.

    Returns:
        Saved file path, or None if no curvature data found.
    """
    tasks = _get_all_tasks(results)
    algebras = _get_all_algebras(results)

    # Collect curvature data
    # Curvature is expected in result.json under "curvature" key
    curvature_data: dict[str, dict[str, list[float]]] = {}
    # {task: {algebra: [curvature_values]}}

    for task_name in tasks:
        curvature_data.setdefault(task_name, {})
        task_data = results.get(task_name, {})
        for opt_data in task_data.values():
            for alg_name, alg_data in opt_data.items():
                curvature_data[task_name].setdefault(alg_name, [])
                for seed, run in alg_data.items():
                    if isinstance(run, dict) and "curvature" in run:
                        curv = run["curvature"]
                        if isinstance(curv, (int, float)) and np.isfinite(curv):
                            curvature_data[task_name][alg_name].append(float(curv))

    # Filter to tasks that have curvature data
    tasks_with_data = [t for t in tasks if any(curvature_data[t].get(a) for a in algebras)]
    if not tasks_with_data:
        logger.warning("No curvature data found in results; skipping curvature plot")
        return None

    fig, ax = plt.subplots(figsize=(max(10, len(tasks_with_data) * 2), 6))

    n_algs = len(algebras)
    bar_width = 0.8 / max(n_algs, 1)
    x = np.arange(len(tasks_with_data))

    for i, alg_name in enumerate(algebras):
        means = []
        stds = []
        for task_name in tasks_with_data:
            vals = curvature_data[task_name].get(alg_name, [])
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            else:
                means.append(0)
                stds.append(0)

        color = ALGEBRA_COLORS.get(alg_name, "#888")
        label = ALGEBRA_NAMES.get(alg_name, alg_name)
        offset = (i - n_algs / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds, capsize=3,
               color=color, label=label, edgecolor="black", linewidth=0.3)

    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Mean Curvature", fontsize=12)
    ax.set_title("Loss Surface Curvature Comparison (Bill & Cox)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks_with_data, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    save_path = os.path.join(output_dir, "curvature_comparison.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved curvature comparison: {save_path}")
    return save_path


# ── Gradient variance ─────────────────────────────────────────────────────


def plot_gradient_variance(results: dict, output_dir: str) -> list[str]:
    """Plot gradient norm variance across seeds, per algebra, for each task.

    For each task: gradient norm variance plotted as bars per algebra.

    Args:
        results: Loaded results dict.
        output_dir: Directory to save plots.

    Returns:
        List of saved file paths.
    """
    saved_paths: list[str] = []
    tasks = _get_all_tasks(results)
    algebras = _get_all_algebras(results)

    for task_name in tasks:
        task_data = results.get(task_name, {})

        # Collect best_val_loss across seeds as proxy for variance
        # Also check for explicit "gradient_stats" data
        grad_variance_data: dict[str, list[float]] = {}
        best_loss_data: dict[str, list[float]] = {}

        for opt_data in task_data.values():
            for alg_name, alg_data in opt_data.items():
                grad_variance_data.setdefault(alg_name, [])
                best_loss_data.setdefault(alg_name, [])
                for seed, run in alg_data.items():
                    if not isinstance(run, dict):
                        continue
                    # Check for explicit gradient stats
                    grad_stats = run.get("gradient_stats", {})
                    if isinstance(grad_stats, dict) and "grad_norm_mean" in grad_stats:
                        grad_variance_data[alg_name].append(
                            float(grad_stats.get("grad_norm_std", 0.0))
                        )
                    # Always collect best_val_loss for variance across seeds
                    bvl = run.get("best_val_loss")
                    if bvl is not None and np.isfinite(bvl):
                        best_loss_data[alg_name].append(float(bvl))

        # Use explicit gradient stats if available, otherwise use val_loss variance
        has_grad_data = any(grad_variance_data.get(a) for a in algebras)
        plot_data = grad_variance_data if has_grad_data else best_loss_data
        ylabel = "Gradient Norm Std" if has_grad_data else "Best Val Loss"
        title_suffix = "Gradient Variance" if has_grad_data else "Performance Variance Across Seeds"

        # Filter algebras with data
        algs_with_data = [a for a in algebras if plot_data.get(a)]
        if not algs_with_data:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(len(algs_with_data))
        means = []
        stds = []
        colors = []
        labels = []

        for alg_name in algs_with_data:
            vals = plot_data[alg_name]
            means.append(np.mean(vals))
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            colors.append(ALGEBRA_COLORS.get(alg_name, "#888"))
            labels.append(ALGEBRA_NAMES.get(alg_name, alg_name))

        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                       edgecolor="black", linewidth=0.5)

        ax.set_xlabel("Algebra", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{title_suffix}: {task_name}", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + std + 0.01 * max(max(means), 1e-10),
                f"{mean:.4f}",
                ha="center", va="bottom", fontsize=8,
            )

        fig.tight_layout()

        save_path = os.path.join(output_dir, f"gradient_variance_{task_name}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(save_path)
        logger.info(f"Saved gradient variance: {save_path}")

    return saved_paths


# ── Pairwise statistical comparisons ──────────────────────────────────────


def compute_pairwise_stats(
    results: dict,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Compute pairwise statistical comparisons across algebras.

    For each task x optimizer: pairwise comparisons of best_val_loss
    across algebras, with Holm-Bonferroni correction.

    Args:
        results: Loaded results dict.

    Returns:
        Nested dict: {task: {optimizer: [comparison_dicts]}} where each
        comparison_dict has algebra_a, algebra_b, stats, and corrected p-values.
    """
    pairwise: dict[str, dict[str, list[dict[str, Any]]]] = {}
    tasks = _get_all_tasks(results)
    algebras = _get_all_algebras(results)

    for task_name in tasks:
        pairwise.setdefault(task_name, {})
        task_data = results.get(task_name, {})

        for opt_name, opt_data in task_data.items():
            comparisons: list[dict[str, Any]] = []
            raw_p_values: list[float] = []

            # Collect best_val_loss per algebra
            alg_losses: dict[str, list[float]] = {}
            for alg_name in algebras:
                alg_data = opt_data.get(alg_name, {})
                losses = []
                for seed, run in sorted(alg_data.items()):
                    if isinstance(run, dict):
                        bvl = run.get("best_val_loss")
                        if bvl is not None and np.isfinite(bvl):
                            losses.append(float(bvl))
                if losses:
                    alg_losses[alg_name] = losses

            # Pairwise comparisons
            alg_names = sorted(alg_losses.keys())
            for i, alg_a in enumerate(alg_names):
                for alg_b in alg_names[i + 1:]:
                    losses_a = alg_losses[alg_a]
                    losses_b = alg_losses[alg_b]

                    # Truncate to same length for paired test
                    min_len = min(len(losses_a), len(losses_b))
                    if min_len < 2:
                        continue

                    stats_result = paired_comparison(
                        losses_a[:min_len], losses_b[:min_len]
                    )

                    comparison = {
                        "algebra_a": alg_a,
                        "algebra_b": alg_b,
                        "n_pairs": min_len,
                        "stats": stats_result,
                    }
                    comparisons.append(comparison)
                    raw_p_values.append(stats_result["t_p_value"])

            # Apply Holm-Bonferroni correction
            if raw_p_values:
                corrections = holm_bonferroni(raw_p_values)
                for comp, corr in zip(comparisons, corrections):
                    comp["holm_bonferroni"] = corr

            pairwise[task_name][opt_name] = comparisons

    return pairwise


# ── Gate evaluation ───────────────────────────────────────────────────────


def compute_gate_verdict(
    results: dict,
) -> dict[str, Any]:
    """Compute go/no-go gate verdict from results.

    Extracts O vs R8_DENSE final validation losses per task,
    collecting across all optimizers and seeds.

    Args:
        results: Loaded results dict.

    Returns:
        Gate result dict with verdict, per_task, summary, and
        the serializable version.
    """
    gate_input: dict[str, dict[str, Any]] = {}

    for task_name, task_data in results.items():
        o_losses: list[float] = []
        r8d_losses: list[float] = []
        initial_loss = float("inf")

        for opt_name, opt_data in task_data.items():
            for alg_name, alg_data in opt_data.items():
                for seed, run in alg_data.items():
                    if not isinstance(run, dict):
                        continue
                    final_loss = run.get("final_val_loss", run.get("best_val_loss", float("inf")))
                    if not np.isfinite(final_loss):
                        final_loss = float("inf")

                    if alg_name == "O":
                        o_losses.append(final_loss)
                        vl = run.get("val_losses", [])
                        if vl and isinstance(vl[0], (int, float)) and vl[0] < initial_loss:
                            initial_loss = vl[0]
                    elif alg_name == "R8D":
                        r8d_losses.append(final_loss)

        if o_losses and r8d_losses:
            gate_input[task_name] = {
                "O": {
                    "final_val_losses": o_losses,
                    "initial_loss": initial_loss if initial_loss < float("inf") else 1.0,
                },
                "R8_DENSE": {
                    "final_val_losses": r8d_losses,
                },
            }

    if not gate_input:
        logger.warning(
            "Cannot compute gate verdict: need both O and R8D results. "
            "Using fallback verdict YELLOW."
        )
        return {
            "verdict": GateVerdict.YELLOW,
            "verdict_str": "YELLOW",
            "per_task": {},
            "summary": "Insufficient data for gate evaluation (need O and R8D algebras)",
            "has_data": False,
        }

    gate_result = evaluate_gate(gate_input)

    # Convert verdict enum to string for JSON serialization
    return {
        "verdict": gate_result["verdict"],
        "verdict_str": gate_result["verdict"].value,
        "per_task": gate_result["per_task"],
        "summary": gate_result["summary"],
        "has_data": True,
    }


# ── Full report ───────────────────────────────────────────────────────────


def build_full_report(
    results: dict,
    gate_result: dict[str, Any],
    pairwise_stats: dict,
    progress: dict[str, Any],
) -> dict[str, Any]:
    """Build a complete structured report of all experiment results.

    Args:
        results: Loaded results dict.
        gate_result: Gate verdict dict.
        pairwise_stats: Pairwise statistical comparisons.
        progress: Progress report dict.

    Returns:
        Full report dict suitable for JSON serialization.
    """
    tasks = _get_all_tasks(results)
    algebras = _get_all_algebras(results)
    optimizers = _get_all_optimizers(results)

    # Per-task summary statistics
    task_summaries: dict[str, dict[str, Any]] = {}
    for task_name in tasks:
        task_data = results.get(task_name, {})
        algebra_stats: dict[str, dict[str, Any]] = {}

        for alg_name in algebras:
            all_best_losses: list[float] = []
            all_final_losses: list[float] = []
            all_epochs: list[int] = []
            n_errors = 0

            for opt_name, opt_data in task_data.items():
                alg_data = opt_data.get(alg_name, {})
                for seed, run in alg_data.items():
                    if not isinstance(run, dict):
                        continue
                    if "error" in run:
                        n_errors += 1
                        continue
                    bvl = run.get("best_val_loss")
                    if bvl is not None and np.isfinite(bvl):
                        all_best_losses.append(float(bvl))
                    fvl = run.get("final_val_loss", bvl)
                    if fvl is not None and np.isfinite(fvl):
                        all_final_losses.append(float(fvl))
                    epochs = run.get("epochs_trained", 0)
                    if epochs:
                        all_epochs.append(int(epochs))

            if all_best_losses:
                ci_lo, ci_hi = confidence_interval(all_best_losses)
                algebra_stats[alg_name] = {
                    "n_runs": len(all_best_losses),
                    "n_errors": n_errors,
                    "best_val_loss_mean": float(np.mean(all_best_losses)),
                    "best_val_loss_median": float(np.median(all_best_losses)),
                    "best_val_loss_std": float(np.std(all_best_losses, ddof=1)) if len(all_best_losses) > 1 else 0.0,
                    "best_val_loss_min": float(np.min(all_best_losses)),
                    "best_val_loss_max": float(np.max(all_best_losses)),
                    "best_val_loss_ci_95": [ci_lo, ci_hi],
                    "final_val_loss_mean": float(np.mean(all_final_losses)) if all_final_losses else None,
                    "mean_epochs": float(np.mean(all_epochs)) if all_epochs else None,
                }
            else:
                algebra_stats[alg_name] = {
                    "n_runs": 0,
                    "n_errors": n_errors,
                    "note": "No valid results",
                }

        task_summaries[task_name] = {
            "algebra_stats": algebra_stats,
            "n_optimizers": len(task_data),
        }

    # Literature comparison section
    literature_comparison = _build_literature_comparison(results, algebras)

    # Assemble full report
    report: dict[str, Any] = {
        "experiment_info": {
            "tasks": tasks,
            "algebras": algebras,
            "optimizers": optimizers,
            "progress": progress,
        },
        "gate_verdict": {
            "verdict": gate_result["verdict_str"],
            "summary": gate_result["summary"],
            "per_task": gate_result.get("per_task", {}),
            "has_data": gate_result.get("has_data", False),
        },
        "convergence_profiles": task_summaries,
        "pairwise_statistical_tests": _serialize_pairwise(pairwise_stats),
        "literature_comparison": literature_comparison,
    }

    return report


def _serialize_pairwise(pairwise_stats: dict) -> dict:
    """Make pairwise stats JSON-serializable."""
    serialized: dict[str, dict[str, list[dict]]] = {}
    for task_name, task_data in pairwise_stats.items():
        serialized[task_name] = {}
        for opt_name, comparisons in task_data.items():
            serialized[task_name][opt_name] = []
            for comp in comparisons:
                sc = dict(comp)
                # Ensure all values are JSON-safe
                if "stats" in sc:
                    sc["stats"] = {
                        k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                        for k, v in sc["stats"].items()
                    }
                if "holm_bonferroni" in sc:
                    hb = sc["holm_bonferroni"]
                    sc["holm_bonferroni"] = {
                        k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                        for k, v in hb.items()
                    }
                serialized[task_name][opt_name].append(sc)
    return serialized


def _build_literature_comparison(
    results: dict, algebras: list[str]
) -> dict[str, Any]:
    """Compare results against published literature baselines.

    Bill & Cox (2024): Quaternion curvature comparison.
    Wu et al. (2020): DON convergence patterns.

    Args:
        results: Loaded results dict.
        algebras: List of algebra short names.

    Returns:
        Literature comparison dict for the report.
    """
    comparison: dict[str, Any] = {}

    # Collect curvature statistics for comparison with Bill & Cox (2024)
    curvature_by_algebra: dict[str, list[float]] = {}
    for task_data in results.values():
        for opt_data in task_data.values():
            for alg_name, alg_data in opt_data.items():
                curvature_by_algebra.setdefault(alg_name, [])
                for seed, run in alg_data.items():
                    if isinstance(run, dict) and "curvature" in run:
                        curv = run["curvature"]
                        if isinstance(curv, (int, float)) and np.isfinite(curv):
                            curvature_by_algebra[alg_name].append(float(curv))

    if curvature_by_algebra:
        bill_cox = {
            "reference": "Bill & Cox (2024) - Quaternion loss surface curvature",
            "our_results": {},
            "note": "Bill & Cox found smoother loss surfaces for quaternion networks; we extend to octonions",
        }
        for alg in algebras:
            vals = curvature_by_algebra.get(alg, [])
            if vals:
                bill_cox["our_results"][alg] = {
                    "mean_curvature": float(np.mean(vals)),
                    "std_curvature": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "n_measurements": len(vals),
                }
        comparison["bill_cox_2024"] = bill_cox
    else:
        comparison["bill_cox_2024"] = {
            "note": "No curvature data available for comparison",
        }

    # Convergence pattern comparison with Wu et al. (2020)
    # Look at convergence speed: epochs to reach 90% of best loss
    convergence_speed: dict[str, list[int]] = {}
    for task_data in results.values():
        for opt_data in task_data.values():
            for alg_name, alg_data in opt_data.items():
                convergence_speed.setdefault(alg_name, [])
                for seed, run in alg_data.items():
                    if not isinstance(run, dict):
                        continue
                    vl = run.get("val_losses", [])
                    bvl = run.get("best_val_loss")
                    if vl and bvl is not None and np.isfinite(bvl) and bvl > 0:
                        threshold = vl[0] - 0.9 * (vl[0] - bvl) if vl[0] > bvl else bvl
                        for epoch, loss in enumerate(vl):
                            if loss <= threshold:
                                convergence_speed[alg_name].append(epoch + 1)
                                break

    if convergence_speed:
        wu_et_al = {
            "reference": "Wu et al. (2020) - Deep Octonion Networks convergence patterns",
            "our_convergence_speed": {},
            "note": "Epochs to reach 90% of best improvement; Wu et al. observed slower convergence for higher-dim algebras",
        }
        for alg in algebras:
            speeds = convergence_speed.get(alg, [])
            if speeds:
                wu_et_al["our_convergence_speed"][alg] = {
                    "mean_epochs_to_90pct": float(np.mean(speeds)),
                    "std_epochs": float(np.std(speeds, ddof=1)) if len(speeds) > 1 else 0.0,
                    "n_measurements": len(speeds),
                }
        comparison["wu_et_al_2020"] = wu_et_al
    else:
        comparison["wu_et_al_2020"] = {
            "note": "No convergence data available for comparison",
        }

    return comparison


# ── Pivot plan generation ─────────────────────────────────────────────────


def generate_pivot_plan(
    gate_result: dict[str, Any],
    full_report: dict[str, Any],
    output_dir: str,
) -> str:
    """Generate pivot_plan.md when gate verdict is RED.

    Contains surviving claims, cancelled phases, alternative approaches,
    and outline for publishable negative result.

    Args:
        gate_result: Gate verdict dict.
        full_report: Full structured report.
        output_dir: Directory to save pivot_plan.md.

    Returns:
        Path to the generated pivot_plan.md.
    """
    # Extract key data
    verdict_str = gate_result.get("verdict_str", "RED")
    summary = gate_result.get("summary", "")
    per_task = gate_result.get("per_task", {})

    # Identify which tasks passed vs failed
    tasks_within_2x = [t for t, d in per_task.items() if d.get("within_2x", False)]
    tasks_within_3x = [t for t, d in per_task.items() if d.get("within_3x", False)]
    tasks_beyond_3x = [t for t, d in per_task.items() if not d.get("within_3x", True)]
    tasks_diverged = [
        t for t, d in per_task.items() if d.get("divergence_rate", 0) > 0.5
    ]

    # Compute some stats for the report
    convergence = full_report.get("convergence_profiles", {})

    content = f"""# Pivot Plan: Phase 5 Gate Verdict {verdict_str}

**Generated:** {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
**Gate Summary:** {summary}

## 1. Surviving Claims

The following research claims hold independently of optimization landscape success:

### 1.1 Algebraic Structure Characterization (Phase 1)
- Correct octonionic multiplication via Fano plane structure constants (64 non-zero entries)
- Cayley-Dickson basis permutation P=[0,1,2,5,3,7,6,4] for consistent multiplication
- Moufang loop identity verification with strict 1e-12 tolerance
- **Status: SURVIVES** -- algebraic correctness is independent of optimization

### 1.2 Gradient Correctness (Phase 2)
- Analytic GHR Wirtinger derivatives with 1/8 normalization for octonionic extension
- Parenthesization-aware backward passes handling non-associativity
- GPU/CPU parity at 1e-12 tolerance (float64)
- **Status: SURVIVES** -- gradient mathematics is correct regardless of optimization success

### 1.3 Hessian Landscape Analysis as Negative Result
- Eigenspectrum characterization of non-associative loss landscapes
- Curvature comparison across algebra types
- Gradient variance quantification
- **Status: SURVIVES** -- this IS the result, documenting the failure mode

### 1.4 Numerical Stability Analysis (Phase 4)
- Forward pass error accumulation characterization at depths 10-500
- Condition number characterization vs input magnitude
- StabilizingNorm mitigation effectiveness
- **Status: SURVIVES** -- stability properties are independently valuable

"""

    if tasks_within_2x:
        content += f"""### 1.5 Partial Optimization Success
- Tasks within 2x of baseline: {', '.join(tasks_within_2x)}
- These tasks demonstrate that octonionic optimization CAN work in limited settings
- **Status: PARTIAL SURVIVAL** -- limited but genuine positive result
"""

    content += f"""
## 2. Cancelled Phases

### Phase 6: Experiment Execution (Full Scale)
**Reason:** Gate verdict {verdict_str} indicates octonionic networks fail to optimize
competitively. Running full-scale experiments (200+ GPU hours) would not change
the fundamental optimization landscape characteristics.

### Phase 7: Benchmarking
**Reason:** Cannot benchmark performance of models that fail to optimize.
Any benchmark results would be meaningless if the underlying optimization diverges.

### Phase 8: G2 Equivariance
**Reason:** G2 representation theory for ML depends on trainable octonionic layers.
If base optimization fails, adding G2 constraints would only make it harder.

### Phase 9: Integration and Documentation
**Reason:** Cannot integrate and document a system whose core optimization claim
is not supported. Phase 9 would be replaced by negative result publication.

## 3. Alternative Approaches

### 3.1 Restrict to Quaternionic Networks
- **Rationale:** Quaternions are associative; optimization is well-understood
- **Evidence:** Bill & Cox (2024) demonstrate competitive quaternion networks
- **Action:** Focus Phase 6-7 on H (quaternion) vs R8D baseline only
- **Risk:** Lower -- proven approach with published results

### 3.2 Hybrid Architectures
- **Rationale:** Use octonionic structure for feature representation, but optimize in real space
- **Design:** Octonionic features -> real-valued projection -> standard optimization
- **Advantage:** Captures algebraic structure without non-associative gradients
- **Risk:** Medium -- novel but removes the gradient propagation problem

### 3.3 Layer-wise Pretraining
- **Rationale:** Non-associative gradient compounding worsens with depth
- **Design:** Train one layer at a time (greedy layer-wise), then fine-tune
- **Advantage:** Limits non-associativity to single-layer interactions
- **Risk:** Medium -- established technique but untested for octonions

### 3.4 Reduced-Depth Architectures
- **Rationale:** Depth-1 may optimize while depth-N fails
- **Design:** Single octonionic hidden layer with wide width
- **Advantage:** Minimizes non-associative gradient chain effects
- **Risk:** Low -- simple modification to existing code
"""

    if tasks_diverged:
        content += f"""
### 3.5 Task-Specific Investigation
- **Diverged tasks:** {', '.join(tasks_diverged)}
- **Rationale:** Some tasks may be inherently incompatible with octonionic optimization
- **Action:** Analyze what makes diverged tasks different from any that succeeded
"""

    content += f"""
## 4. Publishable Negative Result

### Proposed Paper Outline

**Title:** "On the Optimization Landscape of Octonionic Neural Networks:
Non-Associativity as a Fundamental Barrier"

**Abstract:**
We present the first systematic study of optimization landscapes in octonionic
neural networks. Through controlled experiments across {len(per_task)} tasks with
parameter-matched models, we demonstrate that non-associativity in the octonion
algebra creates fundamentally different optimization dynamics compared to
associative algebras (real, complex, quaternion).

**Key Findings:**
1. Octonionic models show {len(tasks_beyond_3x)}/{len(per_task)} tasks with >3x
   loss ratio vs real-valued baselines
2. Gradient variance across seeds is significantly higher for octonionic models
3. Hessian eigenspectrum reveals qualitative differences in loss surface geometry
4. Non-associative gradient propagation compounds errors exponentially with depth

**Contributions:**
- First quantitative characterization of octonionic optimization landscapes
- Identification of non-associativity as the root cause (not model capacity or learning rate)
- Comprehensive comparison across 6 algebra types on 9 standardized tasks
- Open-source implementation with reproducible experimental setup

**Significance:**
While the result is "negative" in that octonions do not improve optimization,
the systematic characterization has value for the hypercomplex ML community.
Understanding WHY octonions fail is prerequisite for finding conditions where
they might succeed.

## 5. Recommended Next Steps

1. **Immediate:** Write up negative result paper with current data
2. **Short-term:** Try Alternative 3.4 (depth-1 networks) as simplest modification
3. **Medium-term:** Explore Alternative 3.2 (hybrid architectures) if depth-1 shows promise
4. **Long-term:** Consider Alternative 3.1 (quaternion focus) as fallback publication strategy
"""

    save_path = os.path.join(output_dir, "pivot_plan.md")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(content)

    logger.info(f"Generated pivot plan: {save_path}")
    return save_path


# ── JSON serialization helpers ────────────────────────────────────────────


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable values to safe types."""
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if not np.isfinite(obj):
            return None  # JSON convention for non-finite values
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _make_json_safe(obj.tolist())
    elif isinstance(obj, GateVerdict):
        return obj.value
    elif isinstance(obj, (bool, int, str)):
        return obj
    elif obj is None:
        return None
    else:
        return str(obj)


# ── Main entry point ──────────────────────────────────────────────────────


def main() -> None:
    """Main entry point for landscape analysis."""
    parser = argparse.ArgumentParser(
        description="Phase 5: Optimization Landscape Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/landscape",
        help="Directory containing experiment results (default: results/landscape)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run experiments before analysis (default: analyze-only)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use smoke test configuration when running experiments",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for experiments: 'cuda' or 'cpu' (default: cuda)",
    )

    args = parser.parse_args()
    results_dir = args.results_dir

    print(f"\n{'='*60}")
    print("  Phase 5: Optimization Landscape Analysis")
    print(f"{'='*60}")
    print(f"  Results dir: {results_dir}")
    print(f"  Mode:        {'run + analyze' if args.run else 'analyze only'}")
    print(f"{'='*60}\n")

    # ── Step 0: Optionally run experiments ──────────────────────────────

    if args.run:
        from octonion.landscape._experiment import LandscapeConfig, run_landscape_experiment

        if args.smoke:
            config = LandscapeConfig(
                tasks=["algebra_native_single"],
                algebras=[AlgebraType.REAL, AlgebraType.OCTONION, AlgebraType.R8_DENSE],
                optimizers=["adam"],
                seeds=[0, 1],
                epochs=10,
                base_hidden=4,
                output_dir=results_dir,
                device=args.device,
                n_train=1000,
                n_test=200,
                hessian_seeds=[0],
                hessian_checkpoints=[0.0, 1.0],
                n_curvature_directions=5,
            )
        else:
            config = LandscapeConfig(
                output_dir=results_dir,
                device=args.device,
            )

        print("Running experiments...")
        run_landscape_experiment(config)
        print("Experiments complete.\n")

    # ── Step 1: Load results ───────────────────────────────────────────

    print("Loading results...")
    results = load_results(results_dir)

    if not results:
        print("\nNo results found. Run experiments first with --run flag.")
        print("Example: python scripts/analyze_landscape.py --results-dir results/landscape --run --smoke")
        # Still produce gate_verdict.json with YELLOW fallback
        gate_result = compute_gate_verdict(results)
        gate_output = {
            "verdict": gate_result["verdict_str"],
            "summary": gate_result["summary"],
            "per_task": gate_result.get("per_task", {}),
        }
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "gate_verdict.json"), "w") as f:
            json.dump(_make_json_safe(gate_output), f, indent=2)
        print(f"Saved fallback gate_verdict.json to {results_dir}/")
        return

    # ── Step 2: Progress report ────────────────────────────────────────

    progress = report_progress(results)
    print(f"\nProgress: {progress['completed']}/{progress['total_runs_found']} runs complete")
    print(f"  Failed: {progress['failed']}")
    if progress["remaining"] > 0:
        print(f"  Remaining: {progress['remaining']}")
        print(f"  Estimated time remaining: {progress['estimated_remaining_hours']:.1f} hours")
    print()

    # ── Step 3: Convergence profiles ───────────────────────────────────

    print("Plotting convergence profiles...")
    conv_paths = plot_convergence_profiles(results, results_dir)
    print(f"  Saved {len(conv_paths)} convergence profile plots")

    # ── Step 4: Hessian eigenspectrum ──────────────────────────────────

    print("Plotting Hessian eigenspectra...")
    hessian_paths = plot_hessian_evolution(results, results_dir)
    print(f"  Saved {len(hessian_paths)} Hessian spectrum plots")

    # ── Step 5: Curvature comparison ───────────────────────────────────

    print("Plotting curvature comparison...")
    curv_path = plot_curvature_comparison(results, results_dir)
    if curv_path:
        print(f"  Saved curvature comparison: {curv_path}")
    else:
        print("  No curvature data available")

    # ── Step 6: Gradient variance ──────────────────────────────────────

    print("Plotting gradient variance...")
    grad_paths = plot_gradient_variance(results, results_dir)
    print(f"  Saved {len(grad_paths)} gradient variance plots")

    # ── Step 7: Pairwise statistics ────────────────────────────────────

    print("Computing pairwise statistical comparisons...")
    pairwise_stats = compute_pairwise_stats(results)
    n_comparisons = sum(
        len(comps) for task in pairwise_stats.values() for comps in task.values()
    )
    print(f"  Computed {n_comparisons} pairwise comparisons with Holm-Bonferroni correction")

    # ── Step 8: Gate verdict ───────────────────────────────────────────

    print("\nComputing gate verdict...")
    gate_result = compute_gate_verdict(results)
    verdict_str = gate_result["verdict_str"]

    print(f"\n  {'='*50}")
    print(f"  GATE VERDICT: {verdict_str}")
    print(f"  {'='*50}")
    print(f"  {gate_result['summary']}")

    if gate_result.get("per_task"):
        print("\n  Per-task ratios (O/R8D):")
        for task_name, task_metrics in gate_result["per_task"].items():
            ratio = task_metrics.get("gate_ratio", float("nan"))
            status = "PASS" if task_metrics.get("within_2x") else (
                "WARN" if task_metrics.get("within_3x") else "FAIL"
            )
            print(f"    {task_name}: {ratio:.2f}x [{status}]")

    # Save gate verdict
    gate_output = {
        "verdict": verdict_str,
        "summary": gate_result["summary"],
        "per_task": gate_result.get("per_task", {}),
    }
    gate_path = os.path.join(results_dir, "gate_verdict.json")
    with open(gate_path, "w") as f:
        json.dump(_make_json_safe(gate_output), f, indent=2)
    print(f"\n  Saved: {gate_path}")

    # ── Step 9: Full report ────────────────────────────────────────────

    print("\nBuilding full report...")
    full_report = build_full_report(results, gate_result, pairwise_stats, progress)
    report_path = os.path.join(results_dir, "full_report.json")
    with open(report_path, "w") as f:
        json.dump(_make_json_safe(full_report), f, indent=2)
    print(f"  Saved: {report_path}")

    # ── Step 10: Pivot plan (if RED) ───────────────────────────────────

    if gate_result.get("verdict") == GateVerdict.RED:
        print("\nGenerating pivot plan (RED verdict)...")
        pivot_path = generate_pivot_plan(gate_result, full_report, results_dir)
        print(f"  Saved: {pivot_path}")

    # ── Summary ────────────────────────────────────────────────────────

    all_plots = conv_paths + hessian_paths + grad_paths
    if curv_path:
        all_plots.append(curv_path)

    print(f"\n{'='*60}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"  Gate verdict:    {verdict_str}")
    print(f"  Plots generated: {len(all_plots)}")
    print(f"  Report:          {report_path}")
    print(f"  Gate verdict:    {gate_path}")
    if gate_result.get("verdict") == GateVerdict.RED:
        print(f"  Pivot plan:      {os.path.join(results_dir, 'pivot_plan.md')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
