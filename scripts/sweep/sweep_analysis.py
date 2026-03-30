"""Automated statistical analysis of threshold sweep results per D-41.

Reads SQLite database from sweep runs, performs comprehensive statistical
analysis, and produces JSON report + PNG visualizations.

Statistical tests implemented:
  - Paired Wilcoxon signed-rank test (D-34)
  - Paired t-test (D-34)
  - Bootstrap 95% confidence intervals (D-34)
  - Cohen's d effect sizes (D-35)
  - Friedman test + rank analysis for cross-benchmark consistency (D-36)
  - Bonferroni correction (D-38)
  - Structural variance across seeds (D-37)
  - Generalization gap analysis (D-40)
  - Auto-recommendation with Pareto rank + consistency (D-30)
  - Regime characterization: global vs adaptive (D-45)
  - All results reported without practical significance cutoff (D-39)

Usage:
    python scripts/sweep/sweep_analysis.py \\
        --db results/T2/sweep.db \\
        --output-dir results/T2/analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

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

# All adaptive strategy types to compare against global baseline
ADAPTIVE_STRATEGIES = ["ema", "mean_std", "depth", "purity", "meta_trie", "hybrid"]

# 10 fixed seeds for multi-seed validation per D-33
VALIDATION_SEEDS = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]

# Number of pairwise comparisons for Bonferroni (6 strategies vs global)
N_PAIRWISE_COMPARISONS = 6

# Bonferroni-corrected alpha per D-38
ALPHA = 0.05
BONFERRONI_ALPHA = ALPHA / N_PAIRWISE_COMPARISONS

# Plot styling
FIGSIZE_BAR = (12, 7)
FIGSIZE_RANK = (10, 8)
DPI = 150


# ── Effect size ───────────────────────────────────────────────────


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size per D-35.

    Uses the pooled standard deviation formula:
        d = mean(x - y) / sqrt((var(x, ddof=1) + var(y, ddof=1)) / 2)

    This is robust to the paired case where all differences are identical
    (which makes the paired-diff SD degenerate to zero).

    Args:
        x: First sample array.
        y: Second sample array (same length as x).

    Returns:
        Cohen's d effect size. Positive means x > y.
    """
    diff = x - y
    pooled_sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    if pooled_sd == 0.0:
        return 0.0 if diff.mean() == 0.0 else float("inf") * np.sign(diff.mean())
    return float(diff.mean() / pooled_sd)


# ── Data loading ──────────────────────────────────────────────────


def _load_best_config_per_strategy(
    conn: sqlite3.Connection,
    benchmark: str,
) -> dict[str, dict[str, Any]]:
    """Find the best configuration for each strategy on a benchmark.

    Returns the config with highest final-epoch accuracy for each policy_type.

    Args:
        conn: SQLite connection.
        benchmark: Benchmark name.

    Returns:
        Dict mapping policy_type -> {config_id, assoc_threshold, sim_threshold,
        accuracy, n_nodes, policy_params, ...}
    """
    max_epoch = conn.execute(
        "SELECT MAX(epoch) FROM sweep_results WHERE benchmark = ?",
        (benchmark,),
    ).fetchone()[0]

    if max_epoch is None:
        return {}

    results: dict[str, dict[str, Any]] = {}
    for policy_type in ["global"] + ADAPTIVE_STRATEGIES:
        row = conn.execute(
            """
            SELECT config_id, policy_type, assoc_threshold, sim_threshold,
                   min_share, min_count, noise, accuracy, n_nodes, n_leaves,
                   max_depth, rumination_rejections, consolidation_merges,
                   branching_factor_mean, branching_factor_std, policy_params
            FROM sweep_results
            WHERE benchmark = ? AND epoch = ? AND policy_type = ?
            ORDER BY accuracy DESC
            LIMIT 1
            """,
            (benchmark, max_epoch, policy_type),
        ).fetchone()

        if row is not None:
            results[policy_type] = {
                "config_id": row[0],
                "policy_type": row[1],
                "assoc_threshold": row[2],
                "sim_threshold": row[3],
                "min_share": row[4],
                "min_count": row[5],
                "noise": row[6],
                "accuracy": row[7],
                "n_nodes": row[8],
                "n_leaves": row[9],
                "max_depth": row[10],
                "rumination_rejections": row[11],
                "consolidation_merges": row[12],
                "branching_factor_mean": row[13],
                "branching_factor_std": row[14],
                "policy_params": row[15],
            }

    return results


def _load_multiseed_accuracies(
    conn: sqlite3.Connection,
    benchmark: str,
    config_id: int,
) -> np.ndarray:
    """Load accuracy values across multiple seeds for a config.

    Args:
        conn: SQLite connection.
        benchmark: Benchmark name.
        config_id: Configuration ID.

    Returns:
        Array of accuracy values (one per seed).
    """
    max_epoch = conn.execute(
        "SELECT MAX(epoch) FROM sweep_results WHERE benchmark = ? AND config_id = ?",
        (benchmark, config_id),
    ).fetchone()[0]

    if max_epoch is None:
        return np.array([])

    rows = conn.execute(
        """
        SELECT accuracy FROM sweep_results
        WHERE benchmark = ? AND config_id = ? AND epoch = ?
        ORDER BY seed
        """,
        (benchmark, config_id, max_epoch),
    ).fetchall()

    return np.array([r[0] for r in rows])


def _load_multiseed_by_policy(
    conn: sqlite3.Connection,
    benchmark: str,
    policy_type: str,
) -> np.ndarray:
    """Load multi-seed accuracies for the best config of a given policy type.

    If multi-seed validation data exists (multiple seeds for same config),
    uses that. Otherwise falls back to the single best config result.

    Args:
        conn: SQLite connection.
        benchmark: Benchmark name.
        policy_type: Policy type name.

    Returns:
        Array of accuracy values across seeds.
    """
    max_epoch = conn.execute(
        "SELECT MAX(epoch) FROM sweep_results WHERE benchmark = ? AND policy_type = ?",
        (benchmark, policy_type),
    ).fetchone()[0]

    if max_epoch is None:
        return np.array([])

    # Look for configs with multiple seeds (multi-seed validation data)
    rows = conn.execute(
        """
        SELECT config_id, COUNT(DISTINCT seed) as n_seeds
        FROM sweep_results
        WHERE benchmark = ? AND policy_type = ? AND epoch = ?
        GROUP BY config_id
        HAVING n_seeds > 1
        ORDER BY n_seeds DESC
        LIMIT 1
        """,
        (benchmark, policy_type, max_epoch),
    ).fetchone()

    if rows is not None:
        config_id = rows[0]
        seed_rows = conn.execute(
            """
            SELECT accuracy FROM sweep_results
            WHERE benchmark = ? AND config_id = ? AND epoch = ?
            ORDER BY seed
            """,
            (benchmark, config_id, max_epoch),
        ).fetchall()
        return np.array([r[0] for r in seed_rows])

    # Fallback: get best single-seed config
    row = conn.execute(
        """
        SELECT accuracy FROM sweep_results
        WHERE benchmark = ? AND policy_type = ? AND epoch = ?
        ORDER BY accuracy DESC
        LIMIT 1
        """,
        (benchmark, policy_type, max_epoch),
    ).fetchone()

    if row is not None:
        return np.array([row[0]])

    return np.array([])


# ── Pairwise tests (D-34, D-35, D-38) ────────────────────────────


def _pairwise_comparison(
    global_acc: np.ndarray,
    adaptive_acc: np.ndarray,
    strategy_name: str,
) -> dict[str, Any]:
    """Run paired statistical tests between global and adaptive strategy.

    Performs per D-34:
      - Wilcoxon signed-rank test
      - Paired t-test
      - Bootstrap 95% CI
    Per D-35:
      - Cohen's d effect size
    Per D-38:
      - Bonferroni correction

    Args:
        global_acc: Accuracy array for global baseline (across seeds).
        adaptive_acc: Accuracy array for adaptive strategy (across seeds).
        strategy_name: Name of the adaptive strategy.

    Returns:
        Dict with all test results.
    """
    n = min(len(global_acc), len(adaptive_acc))
    if n < 2:
        return {
            "strategy": strategy_name,
            "n_samples": n,
            "error": "Insufficient samples for paired tests (need >= 2)",
        }

    g = global_acc[:n]
    a = adaptive_acc[:n]
    diff = a - g

    result: dict[str, Any] = {
        "strategy": strategy_name,
        "n_samples": n,
        "global_mean": float(np.mean(g)),
        "adaptive_mean": float(np.mean(a)),
        "mean_diff": float(np.mean(diff)),
    }

    # Wilcoxon signed-rank test (D-34)
    try:
        if np.all(diff == 0):
            result["wilcoxon_statistic"] = 0.0
            result["wilcoxon_p"] = 1.0
        else:
            stat, p = scipy_stats.wilcoxon(g, a)
            result["wilcoxon_statistic"] = float(stat)
            result["wilcoxon_p"] = float(p)
    except ValueError as e:
        result["wilcoxon_p"] = None
        result["wilcoxon_error"] = str(e)

    # Paired t-test (D-34)
    try:
        stat, p = scipy_stats.ttest_rel(g, a)
        result["ttest_statistic"] = float(stat)
        result["ttest_p"] = float(p)
    except Exception as e:
        result["ttest_p"] = None
        result["ttest_error"] = str(e)

    # Bootstrap 95% CI (D-34)
    try:
        boot_result = scipy_stats.bootstrap(
            (diff,),
            np.mean,
            confidence_level=0.95,
            n_resamples=9999,
            method="percentile",
        )
        result["bootstrap_ci_95"] = [
            float(boot_result.confidence_interval.low),
            float(boot_result.confidence_interval.high),
        ]
    except Exception as e:
        result["bootstrap_ci_95"] = None
        result["bootstrap_error"] = str(e)

    # Cohen's d effect size (D-35)
    result["cohens_d"] = cohens_d(a, g)

    # Bonferroni correction (D-38)
    if result.get("wilcoxon_p") is not None:
        result["bonferroni_wilcoxon_p"] = min(
            1.0, result["wilcoxon_p"] * N_PAIRWISE_COMPARISONS
        )
        result["significant_raw_wilcoxon"] = result["wilcoxon_p"] < ALPHA
        result["significant_corrected_wilcoxon"] = (
            result["bonferroni_wilcoxon_p"] < ALPHA
        )

    if result.get("ttest_p") is not None:
        result["bonferroni_ttest_p"] = min(
            1.0, result["ttest_p"] * N_PAIRWISE_COMPARISONS
        )
        result["significant_raw_ttest"] = result["ttest_p"] < ALPHA
        result["significant_corrected_ttest"] = (
            result["bonferroni_ttest_p"] < ALPHA
        )

    return result


def run_pairwise_tests(
    conn: sqlite3.Connection,
    benchmarks: list[str],
) -> dict[str, dict[str, Any]]:
    """Run all pairwise comparisons per D-34, D-35, D-38.

    Compares best global config against each adaptive strategy on each
    benchmark. Uses multi-seed data when available.

    Args:
        conn: SQLite connection.
        benchmarks: List of benchmark names.

    Returns:
        Nested dict: {benchmark: {strategy: test_results}}.
    """
    results: dict[str, dict[str, Any]] = {}

    for benchmark in benchmarks:
        bm_results: dict[str, Any] = {}
        global_acc = _load_multiseed_by_policy(conn, benchmark, "global")

        if len(global_acc) == 0:
            logger.warning(f"No global baseline data for {benchmark}")
            continue

        for strategy in ADAPTIVE_STRATEGIES:
            adaptive_acc = _load_multiseed_by_policy(conn, benchmark, strategy)
            if len(adaptive_acc) == 0:
                bm_results[f"global_vs_{strategy}"] = {
                    "strategy": strategy,
                    "error": "No data for this strategy",
                }
                continue

            bm_results[f"global_vs_{strategy}"] = _pairwise_comparison(
                global_acc, adaptive_acc, strategy
            )

        results[benchmark] = bm_results

    return results


# ── Cross-benchmark consistency (D-36) ───────────────────────────


def run_friedman_test(
    conn: sqlite3.Connection,
    benchmarks: list[str],
) -> dict[str, Any]:
    """Run Friedman test for cross-benchmark consistency per D-36.

    Ranks strategies within each benchmark, then tests whether rankings
    are significantly different using the Friedman chi-square test.
    If significant, reports post-hoc pairwise Wilcoxon with Bonferroni
    (per research pitfall 8: Friedman + Nemenyi preferred over bare Bonferroni).

    Args:
        conn: SQLite connection.
        benchmarks: List of benchmark names.

    Returns:
        Dict with Friedman statistic, p-value, mean/median ranks, post-hoc.
    """
    all_strategies = ["global"] + ADAPTIVE_STRATEGIES
    strategy_accuracies: dict[str, list[float]] = {s: [] for s in all_strategies}

    valid_benchmarks: list[str] = []

    for benchmark in benchmarks:
        best_per_strategy = _load_best_config_per_strategy(conn, benchmark)

        # Only include benchmarks where all strategies have data
        available = set(best_per_strategy.keys())
        if not available.issuperset({"global"}):
            continue

        present_strategies = [s for s in all_strategies if s in available]
        if len(present_strategies) < 3:
            continue

        valid_benchmarks.append(benchmark)
        for strategy in all_strategies:
            if strategy in best_per_strategy:
                strategy_accuracies[strategy].append(
                    best_per_strategy[strategy]["accuracy"]
                )
            else:
                strategy_accuracies[strategy].append(0.0)

    if len(valid_benchmarks) < 2:
        return {
            "error": "Need at least 2 benchmarks with strategy data for Friedman test",
            "valid_benchmarks": valid_benchmarks,
        }

    # Rank within each benchmark (1 = best accuracy)
    n_benchmarks = len(valid_benchmarks)
    n_strategies = len(all_strategies)
    accuracy_matrix = np.zeros((n_benchmarks, n_strategies))

    for j, strategy in enumerate(all_strategies):
        for i in range(n_benchmarks):
            accuracy_matrix[i, j] = strategy_accuracies[strategy][i]

    # Compute ranks per benchmark (scipy rankdata, higher accuracy = lower rank)
    rank_matrix = np.zeros_like(accuracy_matrix)
    for i in range(n_benchmarks):
        # Negate so higher accuracy gets rank 1
        rank_matrix[i, :] = scipy_stats.rankdata(-accuracy_matrix[i, :])

    mean_ranks = {
        s: float(rank_matrix[:, j].mean()) for j, s in enumerate(all_strategies)
    }
    median_ranks = {
        s: float(np.median(rank_matrix[:, j])) for j, s in enumerate(all_strategies)
    }

    # Friedman test
    try:
        # scipy.stats.friedmanchisquare expects groups as separate arrays
        groups = [rank_matrix[:, j] for j in range(n_strategies)]
        # Need raw accuracy values, not ranks, for friedmanchisquare
        acc_groups = [accuracy_matrix[:, j] for j in range(n_strategies)]

        if n_benchmarks >= 2 and n_strategies >= 2:
            statistic, p_value = scipy_stats.friedmanchisquare(*acc_groups)
        else:
            statistic = 0.0
            p_value = 1.0
    except Exception as e:
        return {
            "error": f"Friedman test failed: {e}",
            "mean_ranks": mean_ranks,
            "valid_benchmarks": valid_benchmarks,
        }

    result: dict[str, Any] = {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < ALPHA,
        "n_benchmarks": n_benchmarks,
        "n_strategies": n_strategies,
        "mean_ranks": mean_ranks,
        "median_ranks": median_ranks,
        "rank_matrix": {
            bm: {s: float(rank_matrix[i, j]) for j, s in enumerate(all_strategies)}
            for i, bm in enumerate(valid_benchmarks)
        },
        "accuracy_matrix": {
            bm: {
                s: float(accuracy_matrix[i, j])
                for j, s in enumerate(all_strategies)
            }
            for i, bm in enumerate(valid_benchmarks)
        },
        "valid_benchmarks": valid_benchmarks,
    }

    # Post-hoc pairwise Wilcoxon with Bonferroni if Friedman is significant
    if p_value < ALPHA and n_benchmarks >= 5:
        posthoc: dict[str, Any] = {}
        n_posthoc = n_strategies * (n_strategies - 1) // 2
        pair_idx = 0
        for i in range(n_strategies):
            for j in range(i + 1, n_strategies):
                s_i = all_strategies[i]
                s_j = all_strategies[j]
                try:
                    a_i = accuracy_matrix[:, i]
                    a_j = accuracy_matrix[:, j]
                    diff = a_i - a_j
                    if np.all(diff == 0):
                        stat_w, p_w = 0.0, 1.0
                    else:
                        stat_w, p_w = scipy_stats.wilcoxon(a_i, a_j)
                    posthoc[f"{s_i}_vs_{s_j}"] = {
                        "wilcoxon_p": float(p_w),
                        "bonferroni_p": min(1.0, float(p_w) * n_posthoc),
                        "significant_corrected": float(p_w) * n_posthoc < ALPHA,
                    }
                except Exception as e:
                    posthoc[f"{s_i}_vs_{s_j}"] = {"error": str(e)}
                pair_idx += 1
        result["posthoc_pairwise"] = posthoc

    return result


# ── Structural variance (D-37) ───────────────────────────────────


def compute_structural_variance(
    conn: sqlite3.Connection,
    benchmarks: list[str],
) -> dict[str, dict[str, Any]]:
    """Compute structural variance across seeds per D-37.

    For each top config's multi-seed runs, reports mean +/- std for:
      - node_count
      - max_depth
      - branching_factor
      - rumination_rejections
      - consolidation_merges

    Args:
        conn: SQLite connection.
        benchmarks: List of benchmark names.

    Returns:
        Nested dict: {benchmark: {policy_type: {metric: {mean, std}}}}.
    """
    results: dict[str, dict[str, Any]] = {}

    metrics = [
        ("n_nodes", "node_count"),
        ("max_depth", "max_depth"),
        ("branching_factor_mean", "branching_factor"),
        ("rumination_rejections", "rumination_rejections"),
        ("consolidation_merges", "consolidation_merges"),
    ]

    for benchmark in benchmarks:
        bm_results: dict[str, Any] = {}

        for policy_type in ["global"] + ADAPTIVE_STRATEGIES:
            max_epoch = conn.execute(
                """SELECT MAX(epoch) FROM sweep_results
                   WHERE benchmark = ? AND policy_type = ?""",
                (benchmark, policy_type),
            ).fetchone()[0]

            if max_epoch is None:
                continue

            # Find config with most seeds
            config_row = conn.execute(
                """
                SELECT config_id, COUNT(DISTINCT seed) as n_seeds
                FROM sweep_results
                WHERE benchmark = ? AND policy_type = ? AND epoch = ?
                GROUP BY config_id
                ORDER BY n_seeds DESC, MAX(accuracy) DESC
                LIMIT 1
                """,
                (benchmark, policy_type, max_epoch),
            ).fetchone()

            if config_row is None:
                continue

            config_id, n_seeds = config_row

            rows = conn.execute(
                """
                SELECT n_nodes, max_depth, branching_factor_mean,
                       rumination_rejections, consolidation_merges
                FROM sweep_results
                WHERE benchmark = ? AND config_id = ? AND epoch = ?
                ORDER BY seed
                """,
                (benchmark, config_id, max_epoch),
            ).fetchall()

            if not rows:
                continue

            metric_results: dict[str, Any] = {
                "config_id": config_id,
                "n_seeds": n_seeds,
            }
            for col_idx, (col_name, display_name) in enumerate(metrics):
                values = np.array(
                    [r[col_idx] for r in rows if r[col_idx] is not None],
                    dtype=np.float64,
                )
                if len(values) > 0:
                    metric_results[display_name] = {
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    }

            bm_results[policy_type] = metric_results

        if bm_results:
            results[benchmark] = bm_results

    return results


# ── Generalization gap (D-40) ────────────────────────────────────


def compute_generalization_gap(
    conn: sqlite3.Connection,
    benchmarks: list[str],
) -> dict[str, dict[str, Any]]:
    """Compare 10K subset vs full training set accuracy per D-40.

    Looks for matching configs between reduced (10K) and full-scale runs
    and computes the generalization gap.

    Args:
        conn: SQLite connection.
        benchmarks: List of benchmark names.

    Returns:
        Nested dict: {benchmark: {policy_type: {subset_acc, full_acc, gap, ...}}}.
    """
    results: dict[str, dict[str, Any]] = {}

    for benchmark in benchmarks:
        bm_results: dict[str, Any] = {}

        for policy_type in ["global"] + ADAPTIVE_STRATEGIES:
            max_epoch = conn.execute(
                """SELECT MAX(epoch) FROM sweep_results
                   WHERE benchmark = ? AND policy_type = ?""",
                (benchmark, policy_type),
            ).fetchone()[0]

            if max_epoch is None:
                continue

            # Get best accuracy from reduced configs (config_id < 1400000, 10K subset)
            subset_row = conn.execute(
                """
                SELECT accuracy, config_id, assoc_threshold, sim_threshold
                FROM sweep_results
                WHERE benchmark = ? AND policy_type = ? AND epoch = ?
                      AND config_id < 1400000
                ORDER BY accuracy DESC
                LIMIT 1
                """,
                (benchmark, policy_type, max_epoch),
            ).fetchone()

            # Get best accuracy from full-scale configs (config_id >= 1400000)
            full_row = conn.execute(
                """
                SELECT accuracy, config_id, assoc_threshold, sim_threshold
                FROM sweep_results
                WHERE benchmark = ? AND policy_type = ? AND epoch = ?
                      AND config_id >= 1400000
                ORDER BY accuracy DESC
                LIMIT 1
                """,
                (benchmark, policy_type, max_epoch),
            ).fetchone()

            if subset_row is not None and full_row is not None:
                subset_acc = subset_row[0]
                full_acc = full_row[0]
                abs_gap = full_acc - subset_acc
                rel_gap = abs_gap / max(subset_acc, 1e-10)

                bm_results[policy_type] = {
                    "subset_accuracy": float(subset_acc),
                    "full_accuracy": float(full_acc),
                    "absolute_gap": float(abs_gap),
                    "relative_gap": float(rel_gap),
                    "subset_config_id": subset_row[1],
                    "full_config_id": full_row[1],
                }
            elif subset_row is not None:
                bm_results[policy_type] = {
                    "subset_accuracy": float(subset_row[0]),
                    "full_accuracy": None,
                    "absolute_gap": None,
                    "relative_gap": None,
                    "note": "No full-scale data available",
                }

        if bm_results:
            results[benchmark] = bm_results

    return results


# ── Auto-recommendation (D-30) ───────────────────────────────────


def _compute_pareto_rank(
    accuracies: list[float],
    node_counts: list[float],
) -> list[int]:
    """Compute Pareto rank for each point (accuracy vs node count).

    Rank 1 = Pareto-optimal front (max accuracy for min nodes).
    Higher ranks are dominated by more points.

    Args:
        accuracies: List of accuracy values.
        node_counts: List of node counts.

    Returns:
        List of Pareto ranks (1 = best frontier).
    """
    n = len(accuracies)
    ranks = [1] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j has >= accuracy AND <= nodes (and strictly better in at least one)
            if (
                accuracies[j] >= accuracies[i]
                and node_counts[j] <= node_counts[i]
                and (accuracies[j] > accuracies[i] or node_counts[j] < node_counts[i])
            ):
                ranks[i] += 1

    return ranks


def compute_auto_recommendation(
    conn: sqlite3.Connection,
    benchmarks: list[str],
    friedman_result: dict[str, Any],
    gen_gap: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Auto-recommend best configuration per D-30.

    Identifies top configs by:
    1. Pareto rank (accuracy vs node count) across benchmarks
    2. Cross-benchmark consistency (Friedman rank)
    3. Generalization gap (prefer small gap)

    Args:
        conn: SQLite connection.
        benchmarks: List of benchmark names.
        friedman_result: Friedman test results.
        gen_gap: Generalization gap results.

    Returns:
        Recommendation dict with policy_type, params, justification.
    """
    all_strategies = ["global"] + ADAPTIVE_STRATEGIES

    # Collect per-strategy scores
    scores: dict[str, dict[str, float]] = {}

    for strategy in all_strategies:
        score: dict[str, float] = {
            "mean_accuracy": 0.0,
            "mean_pareto_rank": 0.0,
            "friedman_rank": 0.0,
            "mean_gen_gap": 0.0,
            "n_benchmarks": 0,
        }

        accuracies_list: list[float] = []
        nodes_list: list[float] = []

        for benchmark in benchmarks:
            best = _load_best_config_per_strategy(conn, benchmark)
            if strategy in best:
                accuracies_list.append(best[strategy]["accuracy"])
                nodes_list.append(best[strategy]["n_nodes"])

        if accuracies_list:
            score["mean_accuracy"] = float(np.mean(accuracies_list))
            score["n_benchmarks"] = len(accuracies_list)

            # Pareto rank across benchmarks
            pareto_ranks = _compute_pareto_rank(accuracies_list, nodes_list)
            score["mean_pareto_rank"] = float(np.mean(pareto_ranks))

        # Friedman mean rank
        mean_ranks = friedman_result.get("mean_ranks", {})
        if strategy in mean_ranks:
            score["friedman_rank"] = mean_ranks[strategy]

        # Generalization gap
        gaps = []
        for benchmark in benchmarks:
            bm_gap = gen_gap.get(benchmark, {}).get(strategy, {})
            if bm_gap.get("absolute_gap") is not None:
                gaps.append(abs(bm_gap["absolute_gap"]))
        if gaps:
            score["mean_gen_gap"] = float(np.mean(gaps))

        scores[strategy] = score

    # Composite score: weighted combination
    # Lower is better for all components
    composite: dict[str, float] = {}
    for strategy, score in scores.items():
        if score["n_benchmarks"] == 0:
            composite[strategy] = float("inf")
            continue

        # Normalize components to [0, 1] range
        # Accuracy: higher is better -> invert
        # Pareto rank: lower is better
        # Friedman rank: lower is better
        # Gen gap: lower is better
        composite[strategy] = (
            -score["mean_accuracy"] * 3.0  # Weight accuracy most heavily
            + score["mean_pareto_rank"] * 1.0
            + score["friedman_rank"] * 1.5
            + score["mean_gen_gap"] * 2.0
        )

    # Select best
    best_strategy = min(composite, key=composite.get)  # type: ignore[arg-type]
    best_score = scores[best_strategy]

    # Get representative config params
    best_params: dict[str, Any] = {}
    for benchmark in benchmarks:
        best = _load_best_config_per_strategy(conn, benchmark)
        if best_strategy in best:
            cfg = best[best_strategy]
            best_params = {
                "assoc_threshold": cfg["assoc_threshold"],
                "sim_threshold": cfg["sim_threshold"],
                "min_share": cfg["min_share"],
                "min_count": cfg["min_count"],
                "policy_params": cfg.get("policy_params", "{}"),
            }
            break

    # Build justification
    justification_parts = [
        f"Mean accuracy {best_score['mean_accuracy']:.4f} across {best_score['n_benchmarks']} benchmarks.",
    ]
    if best_score["friedman_rank"] > 0:
        justification_parts.append(
            f"Friedman mean rank {best_score['friedman_rank']:.1f}/{len(all_strategies)}."
        )
    if best_score["mean_pareto_rank"] > 0:
        justification_parts.append(
            f"Mean Pareto rank {best_score['mean_pareto_rank']:.1f} (accuracy vs node count)."
        )
    if best_score["mean_gen_gap"] > 0:
        justification_parts.append(
            f"Mean generalization gap {best_score['mean_gen_gap']:.4f}."
        )

    return {
        "policy_type": best_strategy,
        "params": best_params,
        "justification": " ".join(justification_parts),
        "composite_scores": {k: float(v) for k, v in composite.items()},
        "per_strategy_scores": scores,
    }


# ── Regime characterization (D-45) ───────────────────────────────


def characterize_regimes(
    conn: sqlite3.Connection,
    benchmarks: list[str],
) -> dict[str, Any]:
    """Characterize when global suffices vs when adaptive is needed per D-45.

    Per-benchmark analysis of which benchmarks benefit from adaptive strategies,
    with feature analysis (number of classes, difficulty proxy).

    Args:
        conn: SQLite connection.
        benchmarks: List of benchmark names.

    Returns:
        Regime analysis dict.
    """
    # Known class counts and difficulty characteristics
    benchmark_info = {
        "mnist": {"n_classes": 10, "difficulty": "easy", "modality": "image"},
        "fashion_mnist": {"n_classes": 10, "difficulty": "medium", "modality": "image"},
        "cifar10": {"n_classes": 10, "difficulty": "hard", "modality": "image"},
        "text_4class": {"n_classes": 4, "difficulty": "easy", "modality": "text"},
        "text_20class": {"n_classes": 20, "difficulty": "hard", "modality": "text"},
    }

    per_benchmark: dict[str, Any] = {}
    global_sufficient: list[str] = []
    adaptive_better: list[str] = []

    for benchmark in benchmarks:
        best = _load_best_config_per_strategy(conn, benchmark)
        if "global" not in best:
            continue

        global_acc = best["global"]["accuracy"]
        global_nodes = best["global"]["n_nodes"]

        adaptive_results: dict[str, Any] = {}
        best_adaptive_acc = 0.0
        best_adaptive_strategy = ""

        for strategy in ADAPTIVE_STRATEGIES:
            if strategy not in best:
                continue
            s_acc = best[strategy]["accuracy"]
            s_nodes = best[strategy]["n_nodes"]
            delta = s_acc - global_acc
            adaptive_results[strategy] = {
                "accuracy": float(s_acc),
                "accuracy_delta": float(delta),
                "node_count": int(s_nodes),
                "node_delta": int(s_nodes - global_nodes),
                "improves_accuracy": delta > 0,
            }
            if s_acc > best_adaptive_acc:
                best_adaptive_acc = s_acc
                best_adaptive_strategy = strategy

        max_delta = best_adaptive_acc - global_acc

        # Classify regime
        if max_delta <= 0.001:  # Less than 0.1% improvement
            regime = "global_sufficient"
            global_sufficient.append(benchmark)
        elif max_delta > 0.01:  # More than 1% improvement
            regime = "adaptive_recommended"
            adaptive_better.append(benchmark)
        else:
            regime = "marginal"
            # Could go either way
            if max_delta > 0:
                adaptive_better.append(benchmark)
            else:
                global_sufficient.append(benchmark)

        info = benchmark_info.get(benchmark, {})
        per_benchmark[benchmark] = {
            "regime": regime,
            "global_accuracy": float(global_acc),
            "best_adaptive_accuracy": float(best_adaptive_acc),
            "best_adaptive_strategy": best_adaptive_strategy,
            "max_improvement": float(max_delta),
            "n_classes": info.get("n_classes", "unknown"),
            "difficulty": info.get("difficulty", "unknown"),
            "modality": info.get("modality", "unknown"),
            "strategy_details": adaptive_results,
        }

    # Feature analysis: do harder/more-class benchmarks benefit more?
    result: dict[str, Any] = {
        "per_benchmark": per_benchmark,
        "global_sufficient_benchmarks": global_sufficient,
        "adaptive_recommended_benchmarks": adaptive_better,
        "summary": {
            "n_global_sufficient": len(global_sufficient),
            "n_adaptive_recommended": len(adaptive_better),
            "n_total": len(per_benchmark),
        },
    }

    # Correlation analysis: difficulty vs adaptive benefit
    if len(per_benchmark) >= 3:
        difficulties = {"easy": 1, "medium": 2, "hard": 3}
        diff_scores = []
        improvements = []
        for bm, data in per_benchmark.items():
            d = data.get("difficulty", "unknown")
            if d in difficulties:
                diff_scores.append(difficulties[d])
                improvements.append(data["max_improvement"])

        if len(diff_scores) >= 3:
            try:
                corr, corr_p = scipy_stats.spearmanr(diff_scores, improvements)
                result["difficulty_correlation"] = {
                    "spearman_r": float(corr),
                    "p_value": float(corr_p),
                    "interpretation": (
                        "Harder benchmarks benefit more from adaptive thresholds"
                        if corr > 0 and corr_p < 0.1
                        else "No clear relationship between difficulty and adaptive benefit"
                    ),
                }
            except Exception:
                pass

    return result


# ── Visualization ─────────────────────────────────────────────────


def plot_strategy_comparison(
    conn: sqlite3.Connection,
    benchmarks: list[str],
    output_path: str | Path,
) -> None:
    """Bar chart of mean accuracy per strategy across benchmarks per D-41.

    Args:
        conn: SQLite connection.
        benchmarks: Benchmark names.
        output_path: Path to save PNG.
    """
    all_strategies = ["global"] + ADAPTIVE_STRATEGIES
    strategy_means: dict[str, list[float]] = {s: [] for s in all_strategies}

    valid_benchmarks = []
    for benchmark in benchmarks:
        best = _load_best_config_per_strategy(conn, benchmark)
        if not best:
            continue
        valid_benchmarks.append(benchmark)
        for strategy in all_strategies:
            if strategy in best:
                strategy_means[strategy].append(best[strategy]["accuracy"])
            else:
                strategy_means[strategy].append(0.0)

    if not valid_benchmarks:
        logger.warning("No data for strategy comparison plot")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)

    means = []
    stds = []
    labels = []
    for strategy in all_strategies:
        vals = strategy_means[strategy]
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        else:
            means.append(0.0)
            stds.append(0.0)
        labels.append(strategy)

    colors = plt.cm.Set2(np.linspace(0, 1, len(all_strategies)))
    x = np.arange(len(labels))

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Mean Accuracy", fontsize=12)
    ax.set_title("Strategy Comparison: Mean Accuracy Across Benchmarks", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate with values
    for bar, mean in zip(bars, means):
        if mean > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{mean:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved strategy comparison: {output_path}")


def plot_per_benchmark_comparison(
    conn: sqlite3.Connection,
    benchmarks: list[str],
    output_path: str | Path,
) -> None:
    """Grouped bar chart per benchmark per D-41.

    Args:
        conn: SQLite connection.
        benchmarks: Benchmark names.
        output_path: Path to save PNG.
    """
    all_strategies = ["global"] + ADAPTIVE_STRATEGIES

    valid_benchmarks = []
    data: dict[str, dict[str, float]] = {}

    for benchmark in benchmarks:
        best = _load_best_config_per_strategy(conn, benchmark)
        if not best:
            continue
        valid_benchmarks.append(benchmark)
        data[benchmark] = {}
        for strategy in all_strategies:
            if strategy in best:
                data[benchmark][strategy] = best[strategy]["accuracy"]

    if not valid_benchmarks:
        logger.warning("No data for per-benchmark comparison plot")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    n_strategies = len(all_strategies)
    n_benchmarks = len(valid_benchmarks)
    bar_width = 0.8 / n_strategies
    x = np.arange(n_benchmarks)
    colors = plt.cm.Set2(np.linspace(0, 1, n_strategies))

    for j, strategy in enumerate(all_strategies):
        vals = [data[bm].get(strategy, 0.0) for bm in valid_benchmarks]
        offset = (j - n_strategies / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            vals,
            bar_width,
            label=strategy,
            color=colors[j],
            edgecolor="black",
            linewidth=0.3,
        )

    display_names = [
        BENCHMARK_DISPLAY_NAMES.get(bm, bm) for bm in valid_benchmarks
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Per-Benchmark Strategy Comparison", fontsize=14)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved per-benchmark comparison: {output_path}")


def plot_friedman_ranks(
    friedman_result: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Rank plot (CD diagram style) for Friedman analysis per D-41.

    Args:
        friedman_result: Output from run_friedman_test.
        output_path: Path to save PNG.
    """
    mean_ranks = friedman_result.get("mean_ranks", {})
    if not mean_ranks:
        logger.warning("No rank data for Friedman plot")
        return

    # Sort strategies by mean rank (best first)
    sorted_strategies = sorted(mean_ranks.items(), key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=FIGSIZE_RANK)

    strategies = [s[0] for s in sorted_strategies]
    ranks = [s[1] for s in sorted_strategies]
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(strategies)))

    bars = ax.barh(
        range(len(strategies)),
        ranks,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=12)
    ax.set_xlabel("Mean Rank (lower is better)", fontsize=12)

    p_val = friedman_result.get("p_value", None)
    stat = friedman_result.get("statistic", None)
    title = "Friedman Test: Cross-Benchmark Strategy Ranks"
    if p_val is not None and stat is not None:
        sig = "SIGNIFICANT" if p_val < ALPHA else "not significant"
        title += f"\n(chi2={stat:.2f}, p={p_val:.4f}, {sig})"
    ax.set_title(title, fontsize=13)

    # Annotate with rank values
    for bar, rank in zip(bars, ranks):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{rank:.2f}",
            ha="left",
            va="center",
            fontsize=10,
        )

    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved Friedman ranks plot: {output_path}")


# ── Summary tables ────────────────────────────────────────────────


def _print_pairwise_table(pairwise: dict[str, dict[str, Any]]) -> None:
    """Print pairwise comparison results as formatted table."""
    print("\n" + "=" * 100)
    print("PAIRWISE COMPARISONS (per D-34, D-35, D-38)")
    print("=" * 100)

    for benchmark, tests in pairwise.items():
        display = BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark)
        print(f"\n--- {display} ---")
        print(
            f"{'Comparison':<25} {'Wilcoxon p':>12} {'t-test p':>12} "
            f"{'Cohen d':>10} {'95% CI':>24} {'Sig(raw)':>10} {'Sig(corr)':>10}"
        )
        print("-" * 105)

        for key, result in tests.items():
            if "error" in result and "strategy" not in result:
                continue
            name = result.get("strategy", key)
            w_p = result.get("wilcoxon_p", None)
            t_p = result.get("ttest_p", None)
            d = result.get("cohens_d", None)
            ci = result.get("bootstrap_ci_95", None)
            sig_raw = result.get("significant_raw_wilcoxon", None)
            sig_corr = result.get("significant_corrected_wilcoxon", None)

            w_str = f"{w_p:.6f}" if w_p is not None else "N/A"
            t_str = f"{t_p:.6f}" if t_p is not None else "N/A"
            d_str = f"{d:.4f}" if d is not None else "N/A"
            ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci is not None else "N/A"
            sig_raw_str = "YES" if sig_raw else ("no" if sig_raw is not None else "N/A")
            sig_corr_str = (
                "YES" if sig_corr else ("no" if sig_corr is not None else "N/A")
            )

            print(
                f"global vs {name:<15} {w_str:>12} {t_str:>12} "
                f"{d_str:>10} {ci_str:>24} {sig_raw_str:>10} {sig_corr_str:>10}"
            )


def _print_friedman_table(friedman: dict[str, Any]) -> None:
    """Print Friedman test results as formatted table."""
    print("\n" + "=" * 80)
    print("FRIEDMAN TEST: CROSS-BENCHMARK CONSISTENCY (per D-36)")
    print("=" * 80)

    if "error" in friedman:
        print(f"  Error: {friedman['error']}")
        return

    stat = friedman.get("statistic", 0)
    p = friedman.get("p_value", 1)
    sig = "SIGNIFICANT" if friedman.get("significant", False) else "not significant"
    print(f"  Chi-square statistic: {stat:.4f}")
    print(f"  p-value: {p:.6f}")
    print(f"  Result: {sig} at alpha={ALPHA}")

    mean_ranks = friedman.get("mean_ranks", {})
    if mean_ranks:
        print(f"\n  {'Strategy':<15} {'Mean Rank':>12} {'Median Rank':>12}")
        print("  " + "-" * 40)
        median_ranks = friedman.get("median_ranks", {})
        for strategy in sorted(mean_ranks, key=mean_ranks.get):  # type: ignore[arg-type]
            mr = mean_ranks[strategy]
            mdr = median_ranks.get(strategy, 0)
            print(f"  {strategy:<15} {mr:>12.2f} {mdr:>12.2f}")


def _print_recommendation(recommendation: dict[str, Any]) -> None:
    """Print auto-recommendation."""
    print("\n" + "=" * 80)
    print("AUTO-RECOMMENDATION (per D-30)")
    print("=" * 80)

    print(f"  Recommended configuration: {recommendation['policy_type']}")
    print(f"  Parameters: {json.dumps(recommendation.get('params', {}), indent=4)}")
    print(f"  Justification: {recommendation['justification']}")


# ── Main analysis pipeline ────────────────────────────────────────


def run_full_analysis(
    db_path: str,
    output_dir: str,
) -> dict[str, Any]:
    """Run complete statistical analysis pipeline per D-41.

    Produces:
      - statistical_report.json with all test results
      - strategy_comparison.png
      - per_benchmark_comparison.png
      - friedman_ranks.png

    Args:
        db_path: Path to the SQLite database.
        output_dir: Directory for output files (JSON + PNG).

    Returns:
        Complete analysis results dict.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # Detect available benchmarks
        benchmark_rows = conn.execute(
            "SELECT DISTINCT benchmark FROM sweep_results"
        ).fetchall()
        benchmarks = [row[0] for row in benchmark_rows]

        if not benchmarks:
            logger.warning("No data in database")
            return {"error": "No data in database"}

        logger.info(f"Analyzing {len(benchmarks)} benchmarks: {benchmarks}")

        # Reset row_factory for dict-free queries
        conn.row_factory = None

        # 1. Pairwise tests (D-34, D-35, D-38)
        logger.info("\n1. Running pairwise comparisons...")
        pairwise = run_pairwise_tests(conn, benchmarks)

        # 2. Friedman test (D-36)
        logger.info("2. Running Friedman test...")
        friedman = run_friedman_test(conn, benchmarks)

        # 3. Structural variance (D-37)
        logger.info("3. Computing structural variance...")
        structural = compute_structural_variance(conn, benchmarks)

        # 4. Generalization gap (D-40)
        logger.info("4. Computing generalization gap...")
        gen_gap = compute_generalization_gap(conn, benchmarks)

        # 5. Auto-recommendation (D-30)
        logger.info("5. Computing auto-recommendation...")
        recommendation = compute_auto_recommendation(
            conn, benchmarks, friedman, gen_gap
        )

        # 6. Regime characterization (D-45)
        logger.info("6. Characterizing regimes...")
        regime = characterize_regimes(conn, benchmarks)

        # Assemble report
        report: dict[str, Any] = {
            "pairwise_tests": pairwise,
            "friedman": friedman,
            "structural_variance": structural,
            "generalization_gap": gen_gap,
            "recommendation": recommendation,
            "regime_analysis": regime,
            "metadata": {
                "benchmarks": benchmarks,
                "n_benchmarks": len(benchmarks),
                "alpha": ALPHA,
                "bonferroni_alpha": BONFERRONI_ALPHA,
                "n_pairwise_comparisons": N_PAIRWISE_COMPARISONS,
                "adaptive_strategies": ADAPTIVE_STRATEGIES,
                "validation_seeds": VALIDATION_SEEDS,
            },
        }

        # Write JSON report
        report_path = output_path / "statistical_report.json"

        def _json_default(obj: Any) -> Any:
            """Handle non-serializable types."""
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            raise TypeError(f"Not serializable: {type(obj)}")

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=_json_default)
        logger.info(f"\nJSON report written: {report_path}")

        # Print summary tables
        _print_pairwise_table(pairwise)
        _print_friedman_table(friedman)
        _print_recommendation(recommendation)

        # Generate plots
        logger.info("\nGenerating comparison plots...")

        plot_strategy_comparison(
            conn,
            benchmarks,
            output_path / "strategy_comparison.png",
        )

        plot_per_benchmark_comparison(
            conn,
            benchmarks,
            output_path / "per_benchmark_comparison.png",
        )

        plot_friedman_ranks(
            friedman,
            output_path / "friedman_ranks.png",
        )

        logger.info(f"\nAnalysis complete. Output: {output_path}")
        return report

    finally:
        conn.close()


# ── CLI entry point ───────────────────────────────────────────────


def main() -> None:
    """Command-line entry point for statistical analysis."""
    parser = argparse.ArgumentParser(
        description="Run statistical analysis on threshold sweep results (D-41)",
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
        default="results/T2/analysis",
        help="Output directory for JSON + PNG (default: results/T2/analysis)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("STATISTICAL ANALYSIS OF THRESHOLD SWEEP RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Database: {args.db}")
    logger.info(f"  Output: {args.output_dir}")

    run_full_analysis(args.db, args.output_dir)


if __name__ == "__main__":
    main()
