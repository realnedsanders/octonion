"""Diagnostic visualizations for octonionic trie threshold analysis per D-54.

Produces per-node associator norm distributions, depth profiles, routing
statistics, and category routing path heatmaps. Designed for thesis figures
and informing future phases.

Usage:
    python scripts/sweep/sweep_diagnostics.py \
        --features-dir results/T2/features \
        --db results/T2/sweep.db \
        --output-dir results/T2/diagnostics
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── Trie walking utilities ─────────────────────────────────────


def _walk_trie(root: Any) -> list[Any]:
    """Collect all nodes from a trie via depth-first traversal.

    Args:
        root: TrieNode root of the trie.

    Returns:
        List of all TrieNode instances.
    """
    nodes: list[Any] = []
    stack = [root]
    while stack:
        node = stack.pop()
        nodes.append(node)
        for child in node.children.values():
            stack.append(child)
    return nodes


def _collect_associator_norms_by_depth(
    root: Any,
) -> dict[int, list[float]]:
    """Collect associator norm histories from _policy_state, grouped by depth.

    Searches for EMA mean/var, Welford mean, or raw meta_assoc_norms in
    each node's _policy_state dict. Falls back to computing norms from
    buffer entries if no policy state exists.

    Args:
        root: TrieNode root.

    Returns:
        Dict mapping depth -> list of associator norms observed at that depth.
    """
    from octonion._octonion import Octonion, associator

    norms_by_depth: dict[int, list[float]] = defaultdict(list)
    nodes = _walk_trie(root)

    for node in nodes:
        depth = node.depth
        state = getattr(node, "_policy_state", {})

        # Try to get norms from policy state
        if "meta_assoc_norms" in state:
            norms_by_depth[depth].extend(state["meta_assoc_norms"])
        elif "ema_mean" in state:
            # Reconstruct approximate distribution from EMA stats
            count = state.get("ema_count", 0)
            mean = state["ema_mean"]
            var = state.get("ema_var", 0.0)
            if count > 0:
                norms_by_depth[depth].append(mean)
                # Add synthetic samples around the mean to represent the distribution
                std = math.sqrt(max(var, 0.0))
                if std > 0 and count > 1:
                    norms_by_depth[depth].extend(
                        [mean - std, mean + std]
                    )
        elif "welford_mean" in state:
            count = state.get("welford_count", 0)
            mean = state["welford_mean"]
            M2 = state.get("welford_M2", 0.0)
            if count > 0:
                norms_by_depth[depth].append(mean)
                std = math.sqrt(max(M2 / count, 0.0)) if count > 1 else 0.0
                if std > 0:
                    norms_by_depth[depth].extend(
                        [mean - std, mean + std]
                    )
        else:
            # Compute norms from buffer entries directly
            if hasattr(node, "buffer") and len(node.buffer) >= 2:
                node_oct = Octonion(node.routing_key)
                for buf_x, _ in node.buffer:
                    buf_oct = Octonion(buf_x)
                    assoc_val = associator(buf_oct, node_oct, node_oct)
                    norm = assoc_val.components.norm().item()
                    norms_by_depth[depth].append(norm)

    return dict(norms_by_depth)


# ── Visualization functions ────────────────────────────────────


def plot_per_node_assoc_distributions(
    trie: Any,
    save_path: str | Path,
    max_depths: int = 8,
) -> None:
    """Plot histograms of associator norms per depth level per D-54.

    Creates a faceted figure with one histogram per depth level showing
    how associator norm distributions vary by depth.

    Args:
        trie: OctonionTrie instance.
        save_path: Path to save the PNG figure.
        max_depths: Maximum number of depth levels to display.
    """
    norms_by_depth = _collect_associator_norms_by_depth(trie.root)

    if not norms_by_depth:
        logger.warning("No associator norm data found in trie nodes.")
        _save_empty_figure(save_path, "No associator norm data available")
        return

    depths = sorted(norms_by_depth.keys())[:max_depths]
    n_depths = len(depths)
    n_cols = min(4, n_depths)
    n_rows = math.ceil(n_depths / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_depths == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, depth in enumerate(depths):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        norms = norms_by_depth[depth]
        # Filter out non-finite and negative norms
        norms = [n for n in norms if np.isfinite(n) and n >= 0]
        if norms:
            ax.hist(norms, bins=min(30, max(5, len(norms) // 3)), color="steelblue", edgecolor="white", alpha=0.8)
            ax.axvline(np.mean(norms), color="red", linestyle="--", linewidth=1.2, label=f"mean={np.mean(norms):.3f}")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Depth {depth} (n={len(norms)})", fontsize=9)
        ax.set_xlabel("Associator Norm", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for idx in range(n_depths, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle("Per-Node Associator Norm Distributions by Depth", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved associator distributions to {save_path}")


def plot_depth_profile(
    trie: Any,
    save_path: str | Path,
) -> None:
    """Plot 3-panel depth profile per D-54.

    Panel 1: Node count vs depth
    Panel 2: Mean branching factor vs depth
    Panel 3: Mean associator norm vs depth

    Args:
        trie: OctonionTrie instance.
        save_path: Path to save the PNG figure.
    """
    nodes = _walk_trie(trie.root)

    # Aggregate by depth
    depth_stats: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"node_count": [], "branching": [], "assoc_norms": []}
    )

    for node in nodes:
        d = node.depth
        depth_stats[d]["node_count"].append(1)
        depth_stats[d]["branching"].append(float(len(node.children)))

    # Collect associator norms by depth
    norms_by_depth = _collect_associator_norms_by_depth(trie.root)

    depths = sorted(depth_stats.keys())
    if not depths:
        _save_empty_figure(save_path, "No nodes in trie")
        return

    node_counts = [sum(depth_stats[d]["node_count"]) for d in depths]
    mean_branching = [np.mean(depth_stats[d]["branching"]) for d in depths]
    mean_assoc = []
    for d in depths:
        d_norms = norms_by_depth.get(d, [])
        d_norms = [n for n in d_norms if np.isfinite(n) and n >= 0]
        mean_assoc.append(np.mean(d_norms) if d_norms else 0.0)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Node count vs depth
    ax1.bar(depths, node_counts, color="steelblue", edgecolor="white", alpha=0.8)
    ax1.set_xlabel("Depth")
    ax1.set_ylabel("Node Count")
    ax1.set_title("Node Count vs Depth")

    # Panel 2: Mean branching factor vs depth
    ax2.plot(depths, mean_branching, "o-", color="darkorange", linewidth=2, markersize=6)
    ax2.axhline(y=7.0, color="gray", linestyle=":", alpha=0.5, label="Max (7)")
    ax2.set_xlabel("Depth")
    ax2.set_ylabel("Mean Branching Factor")
    ax2.set_title("Branching Factor vs Depth")
    ax2.legend(fontsize=8)

    # Panel 3: Mean associator norm vs depth
    ax3.plot(depths, mean_assoc, "s-", color="crimson", linewidth=2, markersize=6)
    ax3.set_xlabel("Depth")
    ax3.set_ylabel("Mean Associator Norm")
    ax3.set_title("Associator Norm vs Depth")

    fig.suptitle("Trie Depth Profile", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved depth profile to {save_path}")


def plot_routing_statistics(
    trie: Any,
    save_path: str | Path,
) -> None:
    """Plot routing statistics per D-54.

    Panel 1: Subalgebra slot utilization (which of 7 slots are used most)
    Panel 2: Category purity per depth level
    Panel 3: Buffer occupancy distribution

    Args:
        trie: OctonionTrie instance.
        save_path: Path to save the PNG figure.
    """
    nodes = _walk_trie(trie.root)

    # Panel 1: Subalgebra slot utilization
    slot_counts = defaultdict(int)
    for node in nodes:
        if node.subalgebra_idx is not None:
            slot_counts[node.subalgebra_idx] += 1

    # Panel 2: Category purity by depth
    depth_purities: dict[int, list[float]] = defaultdict(list)
    for node in nodes:
        if node.category_counts:
            total = sum(node.category_counts.values())
            if total > 0:
                dominant_count = max(node.category_counts.values())
                purity = dominant_count / total
                depth_purities[node.depth].append(purity)

    # Panel 3: Buffer occupancy
    buffer_sizes = [len(node.buffer) for node in nodes if hasattr(node, "buffer")]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Subalgebra slots
    slots = list(range(7))
    slot_vals = [slot_counts.get(s, 0) for s in slots]
    ax1.bar(slots, slot_vals, color="mediumpurple", edgecolor="white", alpha=0.8)
    ax1.set_xlabel("Subalgebra Slot Index")
    ax1.set_ylabel("Node Count")
    ax1.set_title("Subalgebra Slot Utilization")
    ax1.set_xticks(slots)

    # Panel 2: Category purity by depth
    if depth_purities:
        depths = sorted(depth_purities.keys())
        mean_purities = [np.mean(depth_purities[d]) for d in depths]
        std_purities = [np.std(depth_purities[d]) for d in depths]
        ax2.errorbar(
            depths, mean_purities, yerr=std_purities,
            fmt="o-", color="forestgreen", linewidth=2, markersize=5,
            capsize=3, ecolor="gray"
        )
        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel("Depth")
        ax2.set_ylabel("Mean Category Purity")
        ax2.set_title("Category Purity vs Depth")
    else:
        ax2.text(0.5, 0.5, "No category data", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Category Purity vs Depth")

    # Panel 3: Buffer occupancy
    if buffer_sizes:
        ax3.hist(buffer_sizes, bins=min(30, max(5, len(buffer_sizes) // 3)),
                 color="coral", edgecolor="white", alpha=0.8)
        ax3.axvline(np.mean(buffer_sizes), color="red", linestyle="--",
                    linewidth=1.2, label=f"mean={np.mean(buffer_sizes):.1f}")
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No buffer data", ha="center", va="center", transform=ax3.transAxes)
    ax3.set_xlabel("Buffer Size")
    ax3.set_ylabel("Count")
    ax3.set_title("Buffer Occupancy Distribution")

    fig.suptitle("Trie Routing Statistics", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved routing statistics to {save_path}")


def plot_category_routing_paths(
    trie: Any,
    test_samples: torch.Tensor,
    test_labels: torch.Tensor,
    save_path: str | Path,
) -> None:
    """Plot category routing path heatmap per D-54.

    For each category, traces routing paths through the trie and builds
    a heatmap showing which subalgebra slots are favored by which categories.

    Args:
        trie: OctonionTrie instance.
        test_samples: Tensor of shape [N, 8] with test octonion features.
        test_labels: Tensor or array of shape [N] with integer category labels.
        save_path: Path to save the PNG figure.
    """
    categories = sorted(set(int(l) for l in test_labels))
    n_cats = len(categories)

    if n_cats == 0:
        _save_empty_figure(save_path, "No categories in test data")
        return

    # Build heatmap: category x subalgebra slot
    cat_slot_counts = np.zeros((n_cats, 7), dtype=np.float64)

    for sample, label in zip(test_samples, test_labels, strict=False):
        cat_idx = categories.index(int(label))
        # Trace routing path
        x = sample.to(trie.dtype)
        norm = x.norm()
        if norm > 0:
            x = x / norm

        node = trie.root
        for _ in range(trie.max_depth):
            if not node.children:
                break
            _, child, _ = trie._find_best_child(node, x)
            if child is None:
                break
            # Record which subalgebra slot was chosen
            if child.subalgebra_idx is not None:
                cat_slot_counts[cat_idx, child.subalgebra_idx] += 1
            node = child

    # Normalize rows to get proportions
    row_sums = cat_slot_counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    cat_slot_proportions = cat_slot_counts / row_sums

    fig, ax = plt.subplots(figsize=(10, max(4, n_cats * 0.4 + 2)))
    im = ax.imshow(cat_slot_proportions, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Subalgebra Slot Index")
    ax.set_ylabel("Category")
    ax.set_title("Category x Subalgebra Routing Heatmap")
    ax.set_xticks(range(7))
    ax.set_xticklabels([str(i) for i in range(7)])

    # Use category labels for y-axis
    cat_labels = [str(c) for c in categories]
    if n_cats <= 30:
        ax.set_yticks(range(n_cats))
        ax.set_yticklabels(cat_labels, fontsize=max(6, 10 - n_cats // 5))
    else:
        # Too many categories for individual labels
        ax.set_yticks(range(0, n_cats, max(1, n_cats // 10)))

    fig.colorbar(im, ax=ax, label="Proportion of routes")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved category routing paths to {save_path}")


# ── Main diagnostic generation ────────────────────────────────


def generate_diagnostics(
    features_dir: str | Path,
    db_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Generate all diagnostic plots for each benchmark per D-54.

    Loads best config from statistical_report.json (or database fallback),
    builds trie with that config on each benchmark, and generates
    all diagnostic visualizations.

    Args:
        features_dir: Directory with cached benchmark features (from T2-02).
        db_path: Path to sweep SQLite database.
        output_dir: Directory to save diagnostic PNGs.

    Returns:
        Summary dict with generated files and benchmark info.
    """
    from octonion.trie import OctonionTrie

    features_dir = Path(features_dir)
    db_path = Path(db_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load best config from analysis report or fall back to defaults
    best_config = _load_best_config(db_path, output_dir)
    logger.info(f"Best config for diagnostics: {best_config}")

    # All T1/T2 benchmarks
    benchmarks = ["mnist", "fashion_mnist", "cifar10", "text_4class", "text_20class"]
    summary: dict[str, Any] = {
        "best_config": best_config,
        "benchmarks": {},
    }

    for benchmark in benchmarks:
        bm_dir = output_dir / benchmark
        bm_dir.mkdir(parents=True, exist_ok=True)

        # Load cached features
        features_file = features_dir / f"{benchmark}_features.pt"
        if not features_file.exists():
            logger.warning(f"No cached features for {benchmark} at {features_file}, skipping.")
            summary["benchmarks"][benchmark] = {"status": "skipped", "reason": "no features"}
            continue

        data = torch.load(str(features_file), weights_only=True)
        train_x = data["train_x"]
        train_y = data["train_y"]
        test_x = data["test_x"]
        test_y = data["test_y"]

        logger.info(f"Processing {benchmark}: train={len(train_x)}, test={len(test_x)}")

        # Build trie with best config
        policy = _make_policy_from_config(best_config)
        trie = OctonionTrie(
            associator_threshold=best_config.get("assoc_threshold", 0.3),
            similarity_threshold=best_config.get("sim_threshold", 0.1),
            policy=policy,
            seed=42,
        )

        # Train: insert training samples
        n_epochs = best_config.get("n_epochs", 1)
        for _epoch in range(n_epochs):
            for i in range(len(train_x)):
                cat = int(train_y[i].item()) if hasattr(train_y[i], "item") else int(train_y[i])
                trie.insert(train_x[i], category=cat)

        # Evaluate accuracy
        correct = 0
        for i in range(len(test_x)):
            leaf = trie.query(test_x[i])
            pred = leaf.dominant_category
            actual = int(test_y[i].item()) if hasattr(test_y[i], "item") else int(test_y[i])
            if pred == actual:
                correct += 1
        accuracy = correct / len(test_x) if len(test_x) > 0 else 0.0
        stats = trie.stats()

        # Generate all diagnostic plots
        plot_per_node_assoc_distributions(trie, bm_dir / "assoc_distributions.png")
        plot_depth_profile(trie, bm_dir / "depth_profile.png")
        plot_routing_statistics(trie, bm_dir / "routing_statistics.png")
        plot_category_routing_paths(trie, test_x, test_y, bm_dir / "category_routing.png")

        summary["benchmarks"][benchmark] = {
            "status": "complete",
            "accuracy": accuracy,
            "stats": stats,
            "plots": [
                str(bm_dir / "assoc_distributions.png"),
                str(bm_dir / "depth_profile.png"),
                str(bm_dir / "routing_statistics.png"),
                str(bm_dir / "category_routing.png"),
            ],
        }

        logger.info(
            f"  {benchmark}: acc={accuracy:.4f}, nodes={stats['n_nodes']}, "
            f"depth={stats['max_depth']}"
        )

    # Save summary
    summary_path = output_dir / "diagnostics_summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    logger.info(f"Saved diagnostics summary to {summary_path}")

    return summary


# ── Helper functions ──────────────────────────────────────────


def _save_empty_figure(save_path: str | Path, message: str) -> None:
    """Save a placeholder figure with a text message."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="gray")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load_best_config(
    db_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Load best config from statistical_report.json or fall back to database/defaults.

    Args:
        db_path: Path to sweep database.
        output_dir: Output directory (search parent for analysis/).

    Returns:
        Config dict with policy_type, assoc_threshold, sim_threshold, etc.
    """
    # Try to find statistical_report.json
    analysis_dir = db_path.parent / "analysis"
    report_path = analysis_dir / "statistical_report.json"

    if report_path.exists():
        with open(str(report_path)) as f:
            report = json.load(f)
        rec = report.get("recommendation", {})
        if rec:
            params = rec.get("params", {})
            return {
                "policy_type": rec.get("policy_type", "global"),
                "assoc_threshold": params.get("assoc_threshold", 0.3),
                "sim_threshold": params.get("sim_threshold", 0.1),
                "min_share": params.get("min_share", 0.05),
                "min_count": params.get("min_count", 3),
                "policy_params": params.get("policy_params", "{}"),
                "n_epochs": 1,
            }
        logger.info("Report found but no recommendation; using database fallback.")

    # Fallback: query database for best global config
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute(
                """SELECT assoc_threshold, sim_threshold, min_share, min_count
                   FROM sweep_results
                   WHERE policy_type = 'global'
                   ORDER BY accuracy DESC
                   LIMIT 1"""
            )
            row = cursor.fetchone()
            conn.close()
            if row:
                return {
                    "policy_type": "global",
                    "assoc_threshold": row[0],
                    "sim_threshold": row[1],
                    "min_share": row[2],
                    "min_count": row[3],
                    "n_epochs": 1,
                }
        except sqlite3.Error as e:
            logger.warning(f"Database query failed: {e}")

    # Final fallback: current trie defaults
    logger.info("Using default config (GlobalPolicy, assoc_threshold=0.3)")
    return {
        "policy_type": "global",
        "assoc_threshold": 0.3,
        "sim_threshold": 0.1,
        "min_share": 0.05,
        "min_count": 3,
        "n_epochs": 1,
    }


def _make_policy_from_config(config: dict[str, Any]) -> Any:
    """Create a ThresholdPolicy from a config dict.

    Args:
        config: Dict with policy_type and threshold parameters.

    Returns:
        ThresholdPolicy instance.
    """
    from octonion.trie import (
        AlgebraicPurityPolicy,
        DepthPolicy,
        GlobalPolicy,
        HybridPolicy,
        MetaTriePolicy,
        PerNodeEMAPolicy,
        PerNodeMeanStdPolicy,
    )

    policy_type = config.get("policy_type", "global")
    params_str = config.get("policy_params", "{}")
    if isinstance(params_str, str):
        try:
            policy_params = json.loads(params_str)
        except json.JSONDecodeError:
            policy_params = {}
    else:
        policy_params = params_str

    assoc = config.get("assoc_threshold", 0.3)
    sim = config.get("sim_threshold", 0.1)
    min_share = config.get("min_share", 0.05)
    min_count = config.get("min_count", 3)

    if policy_type == "global":
        return GlobalPolicy(
            assoc_threshold=assoc,
            sim_threshold=sim,
            min_share=min_share,
            min_count=min_count,
        )
    elif policy_type == "ema":
        return PerNodeEMAPolicy(
            alpha=policy_params.get("alpha", 0.1),
            k=policy_params.get("k", 1.5),
            base_assoc=assoc,
            sim_threshold=sim,
            min_share=min_share,
            min_count=min_count,
        )
    elif policy_type == "mean_std":
        return PerNodeMeanStdPolicy(
            k=policy_params.get("k", 1.5),
            base_assoc=assoc,
            sim_threshold=sim,
            min_share=min_share,
            min_count=min_count,
        )
    elif policy_type == "depth":
        return DepthPolicy(
            base_assoc=assoc,
            decay_factor=policy_params.get("decay_factor", 1.0),
            sim_threshold=sim,
            min_share=min_share,
            min_count=min_count,
        )
    elif policy_type == "purity":
        return AlgebraicPurityPolicy(
            base_assoc=assoc,
            assoc_weight=policy_params.get("assoc_weight", 0.5),
            sim_weight=policy_params.get("sim_weight", 0.5),
            sensitivity=policy_params.get("sensitivity", 2.0),
            sim_threshold=sim,
            min_share=min_share,
            min_count=min_count,
        )
    elif policy_type == "meta_trie":
        return MetaTriePolicy(
            base_assoc=assoc,
            sim_threshold=sim,
            min_share=min_share,
            min_count=min_count,
            signal_encoding=policy_params.get("signal_encoding", "algebraic"),
            update_frequency=policy_params.get("update_frequency", 10),
            observation_window=policy_params.get("observation_window", 5),
            self_referential=policy_params.get("self_referential", False),
        )
    elif policy_type == "hybrid":
        # Hybrid needs sub-policies -- use defaults if not specified
        sub_a_type = policy_params.get("policy_a_type", "global")
        sub_b_type = policy_params.get("policy_b_type", "ema")
        sub_a = _make_policy_from_config({
            "policy_type": sub_a_type,
            "assoc_threshold": assoc,
            "sim_threshold": sim,
            "min_share": min_share,
            "min_count": min_count,
            "policy_params": policy_params.get("policy_a_params", "{}"),
        })
        sub_b = _make_policy_from_config({
            "policy_type": sub_b_type,
            "assoc_threshold": assoc,
            "sim_threshold": sim,
            "min_share": min_share,
            "min_count": min_count,
            "policy_params": policy_params.get("policy_b_params", "{}"),
        })
        return HybridPolicy(
            policy_a=sub_a,
            policy_b=sub_b,
            combination=policy_params.get("combination", "mean"),
            transition_inserts=policy_params.get("transition_inserts", 0),
        )
    else:
        logger.warning(f"Unknown policy type {policy_type!r}, falling back to GlobalPolicy")
        return GlobalPolicy(assoc_threshold=assoc, sim_threshold=sim)


def _json_default(obj: Any) -> Any:
    """JSON serializer for non-standard types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        val = float(obj)
        if not np.isfinite(val):
            return None
        return val
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ── CLI ───────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for diagnostic generation."""
    parser = argparse.ArgumentParser(
        description="Generate diagnostic visualizations for octonionic trie threshold analysis (D-54)."
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="results/T2/features",
        help="Directory with cached benchmark features from T2-02.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="results/T2/sweep.db",
        help="Path to sweep SQLite database.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/T2/diagnostics",
        help="Directory to save diagnostic PNGs.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    generate_diagnostics(
        features_dir=args.features_dir,
        db_path=args.db,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
