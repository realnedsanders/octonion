"""Visualization functions for threshold sweep results per D-21.

Reads from SQLite database and produces PNG visualizations:
  - 2D heatmaps (assoc x sim) for each benchmark
  - 1D line plots for single-parameter sweeps
  - Pareto frontier (accuracy vs node count)
  - Noise interaction effect plots
  - Epoch convergence curves

All plots use consistent styling: figsize, dpi=150, tight_layout.
matplotlib.use("Agg") for headless rendering.

Usage:
    python scripts/sweep/sweep_plots.py --db results/T2/sweep.db --output-dir results/T2/plots
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────

ALL_BENCHMARKS = ["mnist", "fashion_mnist", "cifar10", "text_4class", "text_20class"]

BENCHMARK_DISPLAY_NAMES = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "cifar10": "CIFAR-10",
    "text_4class": "Text 4-Class",
    "text_20class": "Text 20-Class",
}

# Consistent plot styling
HEATMAP_FIGSIZE = (10, 8)
LINE_FIGSIZE = (10, 6)
DPI = 150


# ── Plot functions ────────────────────────────────────────────────


def plot_heatmap(
    db_path: str,
    benchmark: str,
    x_param: str,
    y_param: str,
    metric: str,
    fixed_params: dict[str, float | int] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create a 2D heatmap of metric vs (x_param, y_param).

    Queries SQLite for the 2D grid of (x_param, y_param) values at the
    final epoch, with other parameters fixed.

    Args:
        db_path: Path to the SQLite database.
        benchmark: Benchmark name (e.g., "mnist").
        x_param: Column name for X axis (e.g., "assoc_threshold").
        y_param: Column name for Y axis (e.g., "sim_threshold").
        metric: Column name for the color value (e.g., "accuracy").
        fixed_params: Dict of param_name -> value to fix (e.g., {"noise": 0.0}).
        save_path: Path to save PNG. If None, figure is returned but not saved.

    Returns:
        The matplotlib Figure.
    """
    if fixed_params is None:
        fixed_params = {}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # Build query with fixed params
        conditions = ["benchmark = ?"]
        params: list = [benchmark]

        for col, val in fixed_params.items():
            conditions.append(f"{col} = ?")
            params.append(val)

        # Get max epoch for this benchmark
        max_epoch = conn.execute(
            "SELECT MAX(epoch) FROM sweep_results WHERE benchmark = ?",
            (benchmark,),
        ).fetchone()[0]

        if max_epoch is None:
            logger.warning(f"No data for benchmark '{benchmark}'")
            fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return fig

        conditions.append("epoch = ?")
        params.append(max_epoch)

        where = " AND ".join(conditions)
        query = f"""
            SELECT {x_param}, {y_param}, AVG({metric}) as value
            FROM sweep_results
            WHERE {where}
            GROUP BY {x_param}, {y_param}
            ORDER BY {x_param}, {y_param}
        """

        rows = conn.execute(query, params).fetchall()

        if not rows:
            logger.warning(f"No data for heatmap: {benchmark}, {x_param} x {y_param}")
            fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return fig

        # Build 2D grid
        x_vals = sorted(set(row[x_param] for row in rows))
        y_vals = sorted(set(row[y_param] for row in rows))

        grid = np.full((len(y_vals), len(x_vals)), np.nan)
        x_idx = {v: i for i, v in enumerate(x_vals)}
        y_idx = {v: i for i, v in enumerate(y_vals)}

        for row in rows:
            xi = x_idx[row[x_param]]
            yi = y_idx[row[y_param]]
            grid[yi, xi] = row["value"]

        # Plot
        display_name = BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark)
        fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE)

        im = ax.imshow(
            grid,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
        )

        # Axis labels
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([f"{v:.3f}" for v in x_vals], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels([f"{v:.3f}" for v in y_vals], fontsize=8)

        ax.set_xlabel(x_param, fontsize=12)
        ax.set_ylabel(y_param, fontsize=12)

        fixed_str = ", ".join(f"{k}={v}" for k, v in fixed_params.items()) if fixed_params else ""
        title = f"{display_name}: {metric} vs {x_param} x {y_param}"
        if fixed_str:
            title += f"\n({fixed_str})"
        ax.set_title(title, fontsize=13)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric, fontsize=11)

        # Annotate cells with values (if grid is small enough)
        if len(x_vals) * len(y_vals) <= 200:
            for yi_idx in range(len(y_vals)):
                for xi_idx in range(len(x_vals)):
                    val = grid[yi_idx, xi_idx]
                    if not np.isnan(val):
                        text_color = "white" if val < np.nanmedian(grid) else "black"
                        ax.text(
                            xi_idx, yi_idx, f"{val:.3f}",
                            ha="center", va="center",
                            fontsize=6, color=text_color,
                        )

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
            logger.info(f"  Saved heatmap: {save_path}")

        return fig

    finally:
        conn.close()


def plot_1d_sweep(
    db_path: str,
    benchmark: str,
    param: str,
    metric: str,
    fixed_params: dict[str, float | int] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create a 1D line plot of metric vs param.

    Shows the metric value at each param value with error bars from
    epoch-to-epoch variance (if multiple epochs exist).

    Args:
        db_path: Path to the SQLite database.
        benchmark: Benchmark name.
        param: Column name to sweep on X axis.
        metric: Column name for Y axis.
        fixed_params: Dict of param_name -> value to fix.
        save_path: Path to save PNG.

    Returns:
        The matplotlib Figure.
    """
    if fixed_params is None:
        fixed_params = {}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conditions = ["benchmark = ?"]
        params_list: list = [benchmark]

        for col, val in fixed_params.items():
            conditions.append(f"{col} = ?")
            params_list.append(val)

        where = " AND ".join(conditions)

        # Get mean and std of metric across epochs for each param value
        query = f"""
            SELECT {param},
                   AVG({metric}) as mean_val,
                   CASE
                       WHEN COUNT({metric}) > 1
                       THEN SQRT(SUM(({metric} - sub.overall_mean) * ({metric} - sub.overall_mean)) / (COUNT({metric}) - 1))
                       ELSE 0.0
                   END as std_val,
                   COUNT({metric}) as n
            FROM sweep_results
            LEFT JOIN (
                SELECT {param} as p, AVG({metric}) as overall_mean
                FROM sweep_results
                WHERE {where}
                GROUP BY {param}
            ) sub ON sub.p = sweep_results.{param}
            WHERE {where}
            GROUP BY {param}
            ORDER BY {param}
        """

        # Simpler query: get final-epoch values grouped by param
        max_epoch_q = f"SELECT MAX(epoch) FROM sweep_results WHERE {where}"
        max_epoch = conn.execute(max_epoch_q, params_list).fetchone()[0]

        if max_epoch is None:
            fig, ax = plt.subplots(figsize=LINE_FIGSIZE)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return fig

        conditions.append("epoch = ?")
        params_list.append(max_epoch)
        where = " AND ".join(conditions)

        query = f"""
            SELECT {param}, AVG({metric}) as mean_val
            FROM sweep_results
            WHERE {where}
            GROUP BY {param}
            ORDER BY {param}
        """

        rows = conn.execute(query, params_list).fetchall()

        if not rows:
            fig, ax = plt.subplots(figsize=LINE_FIGSIZE)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return fig

        x_vals = [row[param] for row in rows]
        y_vals = [row["mean_val"] for row in rows]

        display_name = BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark)
        fig, ax = plt.subplots(figsize=LINE_FIGSIZE)

        ax.plot(x_vals, y_vals, "o-", linewidth=2, markersize=6, color="tab:blue")

        ax.set_xlabel(param, fontsize=12)
        ax.set_ylabel(metric, fontsize=12)

        fixed_str = ", ".join(f"{k}={v}" for k, v in fixed_params.items()) if fixed_params else ""
        title = f"{display_name}: {metric} vs {param}"
        if fixed_str:
            title += f" ({fixed_str})"
        ax.set_title(title, fontsize=13)

        ax.grid(True, alpha=0.3)

        # Log scale for x if values span > 2 orders of magnitude
        if len(x_vals) >= 2 and max(x_vals) / max(min(x_vals), 1e-10) > 100:
            ax.set_xscale("log")

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
            logger.info(f"  Saved 1D plot: {save_path}")

        return fig

    finally:
        conn.close()


def plot_pareto_frontier(
    db_path: str,
    benchmark: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot Pareto frontier of accuracy vs node count.

    Scatters all configs and highlights the Pareto-optimal frontier.
    Pareto: sort by accuracy descending, filter by cumulative min node count.

    Args:
        db_path: Path to the SQLite database.
        benchmark: Benchmark name.
        save_path: Path to save PNG.

    Returns:
        The matplotlib Figure.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        max_epoch = conn.execute(
            "SELECT MAX(epoch) FROM sweep_results WHERE benchmark = ?",
            (benchmark,),
        ).fetchone()[0]

        if max_epoch is None:
            fig, ax = plt.subplots(figsize=LINE_FIGSIZE)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return fig

        rows = conn.execute(
            """
            SELECT config_id, accuracy, n_nodes, assoc_threshold, sim_threshold
            FROM sweep_results
            WHERE benchmark = ? AND epoch = ?
            ORDER BY accuracy DESC
            """,
            (benchmark, max_epoch),
        ).fetchall()

        if not rows:
            fig, ax = plt.subplots(figsize=LINE_FIGSIZE)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return fig

        accuracies = [row["accuracy"] for row in rows]
        node_counts = [row["n_nodes"] for row in rows]

        # Compute Pareto frontier
        # Sort by accuracy desc, then track cumulative min nodes
        sorted_indices = np.argsort(accuracies)[::-1]
        pareto_acc = []
        pareto_nodes = []
        min_nodes = float("inf")

        for idx in sorted_indices:
            if node_counts[idx] < min_nodes:
                pareto_acc.append(accuracies[idx])
                pareto_nodes.append(node_counts[idx])
                min_nodes = node_counts[idx]

        display_name = BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark)
        fig, ax = plt.subplots(figsize=LINE_FIGSIZE)

        # All points
        ax.scatter(
            node_counts, accuracies,
            alpha=0.3, s=15, color="tab:gray",
            label="All configs",
        )

        # Pareto frontier
        ax.scatter(
            pareto_nodes, pareto_acc,
            s=60, color="tab:red", zorder=5,
            label="Pareto frontier",
            edgecolors="black", linewidths=0.5,
        )

        # Connect Pareto points
        sorted_pareto = sorted(zip(pareto_nodes, pareto_acc))
        if sorted_pareto:
            px, py = zip(*sorted_pareto)
            ax.plot(px, py, "r--", linewidth=1.5, alpha=0.7)

        ax.set_xlabel("Node Count", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"{display_name}: Accuracy vs Node Count (Pareto Frontier)", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
            logger.info(f"  Saved Pareto plot: {save_path}")

        return fig

    finally:
        conn.close()


def plot_noise_interaction(
    db_path: str,
    benchmark: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot noise interaction effect per D-05.

    For each noise level, plots accuracy vs assoc_threshold with
    sim_threshold fixed at best value. Shows whether noise helps or
    hurts at different threshold values.

    Args:
        db_path: Path to the SQLite database.
        benchmark: Benchmark name.
        save_path: Path to save PNG.

    Returns:
        The matplotlib Figure.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # Find best sim_threshold (highest mean accuracy at noise=0)
        max_epoch = conn.execute(
            "SELECT MAX(epoch) FROM sweep_results WHERE benchmark = ?",
            (benchmark,),
        ).fetchone()[0]

        if max_epoch is None:
            fig, ax = plt.subplots(figsize=LINE_FIGSIZE)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return fig

        best_sim_row = conn.execute(
            """
            SELECT sim_threshold, AVG(accuracy) as mean_acc
            FROM sweep_results
            WHERE benchmark = ? AND epoch = ? AND noise = 0.0
            GROUP BY sim_threshold
            ORDER BY mean_acc DESC
            LIMIT 1
            """,
            (benchmark, max_epoch),
        ).fetchone()

        best_sim = best_sim_row["sim_threshold"] if best_sim_row else 0.1

        # Get noise values
        noise_values = conn.execute(
            "SELECT DISTINCT noise FROM sweep_results WHERE benchmark = ? ORDER BY noise",
            (benchmark,),
        ).fetchall()

        display_name = BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark)
        fig, ax = plt.subplots(figsize=LINE_FIGSIZE)

        colors = plt.cm.viridis(np.linspace(0, 0.9, len(noise_values)))

        for i, noise_row in enumerate(noise_values):
            noise = noise_row["noise"]
            rows = conn.execute(
                """
                SELECT assoc_threshold, AVG(accuracy) as mean_acc
                FROM sweep_results
                WHERE benchmark = ? AND epoch = ? AND noise = ? AND sim_threshold = ?
                GROUP BY assoc_threshold
                ORDER BY assoc_threshold
                """,
                (benchmark, max_epoch, noise, best_sim),
            ).fetchall()

            if rows:
                x_vals = [row["assoc_threshold"] for row in rows]
                y_vals = [row["mean_acc"] for row in rows]
                ax.plot(
                    x_vals, y_vals,
                    "o-", linewidth=2, markersize=5,
                    color=colors[i],
                    label=f"noise={noise:.2f}",
                )

        ax.set_xlabel("Associator Threshold", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(
            f"{display_name}: Noise Interaction Effect\n(sim_threshold={best_sim:.2f} fixed at best)",
            fontsize=13,
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
            logger.info(f"  Saved noise interaction plot: {save_path}")

        return fig

    finally:
        conn.close()


def plot_epoch_curves(
    db_path: str,
    benchmark: str,
    config_ids: list[int] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot accuracy vs epoch for selected configurations.

    Shows convergence behavior across training epochs for specific configs.

    Args:
        db_path: Path to the SQLite database.
        benchmark: Benchmark name.
        config_ids: List of config_ids to plot. If None, uses top 5 by accuracy.
        save_path: Path to save PNG.

    Returns:
        The matplotlib Figure.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # Auto-select top configs if none specified
        if config_ids is None:
            max_epoch = conn.execute(
                "SELECT MAX(epoch) FROM sweep_results WHERE benchmark = ?",
                (benchmark,),
            ).fetchone()[0]

            if max_epoch is None:
                fig, ax = plt.subplots(figsize=LINE_FIGSIZE)
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                return fig

            top_rows = conn.execute(
                """
                SELECT DISTINCT config_id
                FROM sweep_results
                WHERE benchmark = ? AND epoch = ?
                ORDER BY accuracy DESC
                LIMIT 5
                """,
                (benchmark, max_epoch),
            ).fetchall()
            config_ids = [row["config_id"] for row in top_rows]

        if not config_ids:
            fig, ax = plt.subplots(figsize=LINE_FIGSIZE)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return fig

        display_name = BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark)
        fig, ax = plt.subplots(figsize=LINE_FIGSIZE)

        colors = plt.cm.tab10(np.linspace(0, 1, len(config_ids)))

        for i, cid in enumerate(config_ids):
            rows = conn.execute(
                """
                SELECT epoch, accuracy, assoc_threshold, sim_threshold
                FROM sweep_results
                WHERE benchmark = ? AND config_id = ?
                ORDER BY epoch
                """,
                (benchmark, cid),
            ).fetchall()

            if rows:
                epochs = [row["epoch"] for row in rows]
                accs = [row["accuracy"] for row in rows]
                assoc = rows[0]["assoc_threshold"]
                sim = rows[0]["sim_threshold"]
                ax.plot(
                    epochs, accs,
                    "o-", linewidth=2, markersize=6,
                    color=colors[i],
                    label=f"id={cid} (a={assoc:.3f}, s={sim:.2f})",
                )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"{display_name}: Accuracy vs Epoch", fontsize=13)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
            logger.info(f"  Saved epoch curves: {save_path}")

        return fig

    finally:
        conn.close()


def generate_all_plots(
    db_path: str,
    output_dir: str | Path,
) -> None:
    """Generate all standard plots for all benchmarks per D-21.

    Creates organized PNG files in output_dir:
      heatmaps/       - 2D assoc x sim heatmaps
      line_plots/     - 1D parameter sweeps
      pareto/         - Accuracy vs node count
      noise/          - Noise interaction effects
      epoch_curves/   - Convergence plots

    Args:
        db_path: Path to the SQLite database.
        output_dir: Base directory for output plots.
    """
    output_dir = Path(output_dir)

    # Detect which benchmarks have data
    conn = sqlite3.connect(db_path)
    try:
        benchmark_rows = conn.execute(
            "SELECT DISTINCT benchmark FROM sweep_results"
        ).fetchall()
        benchmarks = [row[0] for row in benchmark_rows]
    finally:
        conn.close()

    if not benchmarks:
        logger.warning("No data in database -- skipping all plots")
        return

    logger.info(f"Generating plots for {len(benchmarks)} benchmarks: {benchmarks}")

    # 1. Heatmaps: assoc x sim for each benchmark at each noise level
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    logger.info("\n--- 2D Heatmaps ---")

    for bm in benchmarks:
        # Default: noise=0.0
        plot_heatmap(
            db_path, bm,
            x_param="assoc_threshold",
            y_param="sim_threshold",
            metric="accuracy",
            fixed_params={"noise": 0.0},
            save_path=heatmap_dir / f"{bm}_assoc_sim_noise0.png",
        )
        plt.close("all")

        # Node count heatmap
        plot_heatmap(
            db_path, bm,
            x_param="assoc_threshold",
            y_param="sim_threshold",
            metric="n_nodes",
            fixed_params={"noise": 0.0},
            save_path=heatmap_dir / f"{bm}_assoc_sim_nodes.png",
        )
        plt.close("all")

    # 2. 1D line plots
    line_dir = output_dir / "line_plots"
    line_dir.mkdir(parents=True, exist_ok=True)
    logger.info("\n--- 1D Line Plots ---")

    for bm in benchmarks:
        # Accuracy vs assoc_threshold
        plot_1d_sweep(
            db_path, bm,
            param="assoc_threshold",
            metric="accuracy",
            fixed_params={"noise": 0.0, "sim_threshold": 0.1},
            save_path=line_dir / f"{bm}_acc_vs_assoc.png",
        )
        plt.close("all")

        # Accuracy vs sim_threshold
        plot_1d_sweep(
            db_path, bm,
            param="sim_threshold",
            metric="accuracy",
            fixed_params={"noise": 0.0, "assoc_threshold": 0.3},
            save_path=line_dir / f"{bm}_acc_vs_sim.png",
        )
        plt.close("all")

    # 3. Pareto frontiers
    pareto_dir = output_dir / "pareto"
    pareto_dir.mkdir(parents=True, exist_ok=True)
    logger.info("\n--- Pareto Frontiers ---")

    for bm in benchmarks:
        plot_pareto_frontier(
            db_path, bm,
            save_path=pareto_dir / f"{bm}_pareto.png",
        )
        plt.close("all")

    # 4. Noise interaction
    noise_dir = output_dir / "noise"
    noise_dir.mkdir(parents=True, exist_ok=True)
    logger.info("\n--- Noise Interaction ---")

    for bm in benchmarks:
        plot_noise_interaction(
            db_path, bm,
            save_path=noise_dir / f"{bm}_noise_interaction.png",
        )
        plt.close("all")

    # 5. Epoch curves
    epoch_dir = output_dir / "epoch_curves"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    logger.info("\n--- Epoch Curves ---")

    for bm in benchmarks:
        plot_epoch_curves(
            db_path, bm,
            save_path=epoch_dir / f"{bm}_epoch_curves.png",
        )
        plt.close("all")

    logger.info(f"\nAll plots saved to: {output_dir}")


# ── CLI entry point ───────────────────────────────────────────────


def main() -> None:
    """Command-line entry point for generating sweep visualizations."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations from threshold sweep results (D-21)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="results/T2/sweep.db",
        help="SQLite database path (default: results/T2/sweep.db)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/T2/plots",
        help="Output directory for PNG files (default: results/T2/plots)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("SWEEP VISUALIZATION")
    logger.info("=" * 60)
    logger.info(f"  Database: {args.db}")
    logger.info(f"  Output: {args.output_dir}")

    generate_all_plots(args.db, args.output_dir)


if __name__ == "__main__":
    main()
