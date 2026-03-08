"""Plotting utilities for training convergence and algebra comparison.

Provides:
- plot_convergence: Train/val loss curves with optional accuracy
- plot_comparison_bars: Bar chart with error bars for algebra comparison
- plot_param_table: Formatted parameter count table as PNG

All plots use matplotlib/seaborn and save to PNG files. Figures are
always closed after saving to avoid memory leaks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend for server/container environments
matplotlib.use("Agg")

# Try to import seaborn for styling; fall back gracefully
try:
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="deep")
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False


def plot_convergence(metrics: dict[str, Any], output_path: str) -> None:
    """Plot training and validation loss curves.

    Creates a dual-axis plot with loss on the left y-axis and optional
    accuracy on the right y-axis.

    Args:
        metrics: Dict with train_losses, val_losses, and optionally
            val_accuracies (all lists of floats).
        output_path: File path for PNG output.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    train_losses = metrics.get("train_losses", [])
    val_losses = metrics.get("val_losses", [])
    val_accuracies = metrics.get("val_accuracies", [])
    epochs = list(range(1, max(len(train_losses), len(val_losses)) + 1))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Loss curves
    if train_losses:
        ax1.plot(epochs[: len(train_losses)], train_losses, "b-", label="Train Loss", linewidth=2)
    if val_losses:
        ax1.plot(epochs[: len(val_losses)], val_losses, "r-", label="Val Loss", linewidth=2)

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12, color="black")
    ax1.tick_params(axis="y")

    # Optional accuracy on secondary y-axis
    if val_accuracies:
        ax2 = ax1.twinx()
        ax2.plot(
            epochs[: len(val_accuracies)],
            val_accuracies,
            "g--",
            label="Val Accuracy",
            linewidth=2,
            alpha=0.7,
        )
        ax2.set_ylabel("Accuracy", fontsize=12, color="green")
        ax2.tick_params(axis="y", labelcolor="green")
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    else:
        ax1.legend(loc="upper right")

    ax1.set_title("Training Convergence", fontsize=14)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_bars(
    results: dict[str, list[float]], metric_name: str, output_path: str
) -> None:
    """Create a bar chart comparing algebras with error bars.

    Each bar represents an algebra (R, C, H, O), with height = mean and
    error bars from standard deviation across seeds.

    Args:
        results: Dict mapping algebra short names to lists of metric values.
        metric_name: Label for the y-axis (e.g., "Accuracy", "Loss").
        output_path: File path for PNG output.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    algebras = list(results.keys())
    means = [float(np.mean(results[a])) for a in algebras]
    stds = [float(np.std(results[a], ddof=1)) if len(results[a]) > 1 else 0.0 for a in algebras]

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    bar_colors = colors[: len(algebras)]

    x = np.arange(len(algebras))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Algebra", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"{metric_name} by Algebra", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(algebras, fontsize=11)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + std + 0.01 * max(means),
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_param_table(
    param_reports: dict[str, list[dict[str, Any]]], output_path: str
) -> None:
    """Create a formatted table image of per-algebra parameter counts.

    Args:
        param_reports: Dict mapping algebra short names to lists of dicts,
            each with 'name' and 'real_params' keys (from param_report()).
        output_path: File path for PNG output.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    algebras = list(param_reports.keys())
    if not algebras:
        return

    # Collect all unique layer names across algebras
    all_layers: list[str] = []
    for algebra in algebras:
        for entry in param_reports[algebra]:
            if entry["name"] not in all_layers:
                all_layers.append(entry["name"])

    # Build table data
    header = ["Layer"] + [f"{a} params" for a in algebras]
    rows: list[list[str]] = []
    for layer_name in all_layers:
        row = [layer_name]
        for algebra in algebras:
            count = 0
            for entry in param_reports[algebra]:
                if entry["name"] == layer_name:
                    count = entry["real_params"]
                    break
            row.append(f"{count:,}")
        rows.append(row)

    # Add total row
    total_row = ["TOTAL"]
    for algebra in algebras:
        total = sum(e["real_params"] for e in param_reports[algebra])
        total_row.append(f"{total:,}")
    rows.append(total_row)

    # Render as matplotlib table
    fig, ax = plt.subplots(figsize=(max(8, len(algebras) * 3), max(4, len(rows) * 0.5 + 1)))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=header,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Style header
    for j in range(len(header)):
        cell = table[0, j]
        cell.set_facecolor("#E0E0E0")
        cell.set_text_props(fontweight="bold")

    # Style total row
    for j in range(len(header)):
        cell = table[len(rows), j]
        cell.set_facecolor("#F5F5F5")
        cell.set_text_props(fontweight="bold")

    ax.set_title("Parameter Counts by Algebra", fontsize=14, pad=20)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
