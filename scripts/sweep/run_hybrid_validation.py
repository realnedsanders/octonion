"""Hybrid strategy sweep and multi-seed validation for T2-08.

Phase 1 -- Hybrid sweep (per D-09):
    Query SQLite for top-2 performing strategy types (by mean accuracy).
    Create HybridPolicy instances combining the top-2 with all 4 combination modes.
    For each mode, sweep with top configs from each base strategy.
    ~4 modes * top-5 pairs * 5 benchmarks = 100 configs.

Phase 2 -- Multi-seed validation (per D-33):
    Identify top-10 configs OVERALL (across all strategies including hybrid).
    For each top config, run with 10 seeds.
    Total: 10 configs * 10 seeds * 5 benchmarks = 500 runs.
    Per D-37: record structural variance across seeds.

Phase 3 -- Full-scale validation (per D-22, D-40):
    Top-5 overall configs re-run on FULL datasets (not 10K subsets).
    Compare full-scale accuracy vs 10K subset accuracy (generalization gap per D-40).

Usage:
    python scripts/sweep/run_hybrid_validation.py --phase 1 --features-dir results/T2/features --db results/T2/sweep.db --workers 24
    python scripts/sweep/run_hybrid_validation.py --phase 2  # multi-seed
    python scripts/sweep/run_hybrid_validation.py --phase 3  # full-scale
    python scripts/sweep/run_hybrid_validation.py --phase all
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.sweep.sweep_runner import SweepConfig, SweepRunner  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────

BENCHMARKS = ["mnist", "fashion_mnist", "cifar10", "text_4class", "text_20class"]
COMBINATION_MODES = ["mean", "min", "max", "adaptive"]
VALIDATION_SEEDS = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]
HYBRID_CONFIG_ID_OFFSET = 1200000
MULTISEED_CONFIG_ID_OFFSET = 1300000
FULLSCALE_CONFIG_ID_OFFSET = 1400000

# Transition inserts for adaptive mode (dataset-size-dependent)
BENCHMARK_TRAIN_SIZES = {
    "mnist": 10000,
    "fashion_mnist": 10000,
    "cifar10": 10000,
    "text_4class": 10000,
    "text_20class": 10000,
}


# ── Phase 1: Hybrid sweep ────────────────────────────────────────────


def _query_top_strategies(db_path: str, top_n: int = 2) -> list[str]:
    """Query SQLite for top-N performing strategy types by mean accuracy.

    Groups results by policy_type, computes mean final-epoch accuracy
    across all benchmarks, returns top-N strategy names.

    Falls back to ["global", "ema"] if DB is empty or has insufficient data.
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT policy_type, AVG(accuracy) as mean_acc
            FROM sweep_results
            WHERE epoch = (
                SELECT MAX(epoch) FROM sweep_results s2
                WHERE s2.config_id = sweep_results.config_id
                  AND s2.benchmark = sweep_results.benchmark
                  AND s2.seed = sweep_results.seed
            )
            GROUP BY policy_type
            ORDER BY mean_acc DESC
            LIMIT ?
            """,
            (top_n,),
        )
        rows = cursor.fetchall()
        if len(rows) >= top_n:
            strategies = [row[0] for row in rows]
            logger.info(f"Top-{top_n} strategies from DB: {strategies}")
            return strategies
    except sqlite3.OperationalError:
        pass
    finally:
        conn.close()

    # Fallback
    fallback = ["global", "ema"][:top_n]
    logger.info(f"Insufficient DB data, falling back to: {fallback}")
    return fallback


def _query_top_configs(
    db_path: str, strategy: str, top_n: int = 5
) -> list[dict[str, Any]]:
    """Query top-N configs for a given strategy by mean accuracy across benchmarks.

    Returns list of dicts with assoc_threshold, sim_threshold, policy_params,
    min_share, min_count, noise, epochs.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            """
            SELECT
                assoc_threshold, sim_threshold, policy_params,
                min_share, min_count, noise,
                MAX(epoch) + 1 as epochs,
                AVG(accuracy) as mean_acc
            FROM sweep_results
            WHERE policy_type = ?
              AND epoch = (
                SELECT MAX(epoch) FROM sweep_results s2
                WHERE s2.config_id = sweep_results.config_id
                  AND s2.benchmark = sweep_results.benchmark
                  AND s2.seed = sweep_results.seed
              )
            GROUP BY assoc_threshold, sim_threshold, policy_params,
                     min_share, min_count, noise
            ORDER BY mean_acc DESC
            LIMIT ?
            """,
            (strategy, top_n),
        )
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _query_top_overall_configs(
    db_path: str, top_n: int = 10
) -> list[dict[str, Any]]:
    """Query top-N configs OVERALL across all strategies.

    Returns list of dicts with policy_type, assoc_threshold, sim_threshold,
    policy_params, min_share, min_count, noise, epochs, mean_acc.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            """
            SELECT
                policy_type, assoc_threshold, sim_threshold, policy_params,
                min_share, min_count, noise,
                MAX(epoch) + 1 as epochs,
                AVG(accuracy) as mean_acc
            FROM sweep_results
            WHERE epoch = (
                SELECT MAX(epoch) FROM sweep_results s2
                WHERE s2.config_id = sweep_results.config_id
                  AND s2.benchmark = sweep_results.benchmark
                  AND s2.seed = sweep_results.seed
              )
            GROUP BY policy_type, assoc_threshold, sim_threshold,
                     policy_params, min_share, min_count, noise
            ORDER BY mean_acc DESC
            LIMIT ?
            """,
            (top_n,),
        )
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def generate_hybrid_sweep_configs(
    db_path: str,
    benchmarks: list[str] | None = None,
) -> list[SweepConfig]:
    """Generate hybrid policy sweep configs per D-09.

    Queries DB for top-2 strategies, creates HybridPolicy configs
    combining top-5 configs from each strategy across 4 combination modes.

    Args:
        db_path: Path to SQLite sweep results database.
        benchmarks: Benchmark names (default: all 5).

    Returns:
        List of SweepConfig instances for hybrid sweep.
    """
    if benchmarks is None:
        benchmarks = BENCHMARKS

    # Get top-2 strategies
    top_strategies = _query_top_strategies(db_path, top_n=2)
    if len(top_strategies) < 2:
        logger.warning("Need at least 2 strategies for hybrid; using fallback")
        top_strategies = ["global", "ema"]

    strategy_a, strategy_b = top_strategies[0], top_strategies[1]
    logger.info(f"Hybrid combining: {strategy_a} x {strategy_b}")

    # Get top-5 configs from each strategy
    configs_a = _query_top_configs(db_path, strategy_a, top_n=5)
    configs_b = _query_top_configs(db_path, strategy_b, top_n=5)

    # Fallback to default configs if DB is empty
    if not configs_a:
        configs_a = [
            {"assoc_threshold": 0.3, "sim_threshold": 0.1,
             "policy_params": "{}", "min_share": 0.05, "min_count": 3,
             "noise": 0.0, "epochs": 3}
        ]
    if not configs_b:
        configs_b = [
            {"assoc_threshold": 0.5, "sim_threshold": 0.1,
             "policy_params": "{}", "min_share": 0.05, "min_count": 3,
             "noise": 0.0, "epochs": 3}
        ]

    sweep_configs: list[SweepConfig] = []
    config_id = HYBRID_CONFIG_ID_OFFSET

    for benchmark in benchmarks:
        for mode in COMBINATION_MODES:
            for ca in configs_a:
                for cb in configs_b:
                    # Build hybrid policy params
                    transition = BENCHMARK_TRAIN_SIZES.get(benchmark, 10000) // 2
                    hybrid_params = {
                        "policy_a_type": strategy_a,
                        "policy_a_params": json.loads(ca.get("policy_params", "{}")),
                        "policy_a_assoc": ca["assoc_threshold"],
                        "policy_a_sim": ca["sim_threshold"],
                        "policy_b_type": strategy_b,
                        "policy_b_params": json.loads(cb.get("policy_params", "{}")),
                        "policy_b_assoc": cb["assoc_threshold"],
                        "policy_b_sim": cb["sim_threshold"],
                        "combination": mode,
                        "transition_inserts": transition if mode == "adaptive" else 0,
                    }

                    # Use mean of the two base thresholds as the config thresholds
                    assoc_t = (ca["assoc_threshold"] + cb["assoc_threshold"]) / 2
                    sim_t = (ca["sim_threshold"] + cb["sim_threshold"]) / 2

                    sweep_configs.append(
                        SweepConfig(
                            config_id=config_id,
                            benchmark=benchmark,
                            policy_type="hybrid",
                            assoc_threshold=float(assoc_t),
                            sim_threshold=float(sim_t),
                            min_share=float(ca.get("min_share", 0.05)),
                            min_count=int(ca.get("min_count", 3)),
                            noise=float(ca.get("noise", 0.0)),
                            epochs=int(ca.get("epochs", 3)),
                            seed=42,
                            policy_params=json.dumps(hybrid_params),
                        )
                    )
                    config_id += 1

    logger.info(
        f"Generated {len(sweep_configs)} hybrid configs: "
        f"{len(COMBINATION_MODES)} modes x {len(configs_a)} x {len(configs_b)} "
        f"x {len(benchmarks)} benchmarks"
    )
    return sweep_configs


def run_hybrid_sweep(
    db_path: str,
    features_dir: str,
    workers: int = 24,
    benchmarks: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Execute Phase 1: Hybrid strategy sweep.

    Args:
        db_path: SQLite database path.
        features_dir: Directory with cached feature files.
        workers: Number of parallel workers.
        benchmarks: Benchmark names (default: all 5).

    Returns:
        List of result summaries.
    """
    configs = generate_hybrid_sweep_configs(db_path, benchmarks)
    if not configs:
        logger.warning("No hybrid configs generated; skipping Phase 1")
        return []

    runner = SweepRunner(db_path, n_workers=workers)
    results = runner.run(configs, features_dir)

    ok = sum(1 for r in results if r["status"] == "ok")
    logger.info(f"Phase 1 complete: {ok}/{len(results)} hybrid configs succeeded")

    # Print cross-mode comparison
    _print_hybrid_comparison(db_path)

    return results


def _print_hybrid_comparison(db_path: str) -> None:
    """Print comparison table of hybrid combination modes."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT
                json_extract(policy_params, '$.combination') as mode,
                benchmark,
                AVG(accuracy) as mean_acc,
                MAX(accuracy) as best_acc,
                COUNT(DISTINCT config_id) as n_configs
            FROM sweep_results
            WHERE policy_type = 'hybrid'
              AND epoch = (
                SELECT MAX(epoch) FROM sweep_results s2
                WHERE s2.config_id = sweep_results.config_id
                  AND s2.benchmark = sweep_results.benchmark
                  AND s2.seed = sweep_results.seed
              )
            GROUP BY mode, benchmark
            ORDER BY mode, benchmark
            """
        )
        rows = cursor.fetchall()
        if not rows:
            logger.info("No hybrid results found")
            return

        print("\n=== Hybrid Combination Mode Comparison ===")
        print(f"{'Mode':<12} {'Benchmark':<15} {'Mean Acc':<10} {'Best Acc':<10} {'N':<5}")
        print("-" * 52)
        for mode, bm, mean_acc, best_acc, n in rows:
            print(f"{mode:<12} {bm:<15} {mean_acc:.4f}    {best_acc:.4f}    {n}")
    except sqlite3.OperationalError as e:
        logger.warning(f"Could not print hybrid comparison: {e}")
    finally:
        conn.close()


# ── Phase 2: Multi-seed validation ────────────────────────────────────


def generate_multiseed_configs(
    db_path: str,
    top_n: int = 10,
    benchmarks: list[str] | None = None,
) -> list[SweepConfig]:
    """Generate multi-seed validation configs per D-33.

    Identifies top-N configs OVERALL, generates configs for 10 seeds each.

    Args:
        db_path: SQLite database path.
        top_n: Number of top configs to validate.
        benchmarks: Benchmark names.

    Returns:
        List of SweepConfig instances for multi-seed validation.
    """
    if benchmarks is None:
        benchmarks = BENCHMARKS

    top_configs = _query_top_overall_configs(db_path, top_n=top_n)

    if not top_configs:
        logger.warning("No configs found in DB for multi-seed validation")
        return []

    sweep_configs: list[SweepConfig] = []
    config_id = MULTISEED_CONFIG_ID_OFFSET

    for rank, cfg in enumerate(top_configs):
        for benchmark in benchmarks:
            for seed in VALIDATION_SEEDS:
                sweep_configs.append(
                    SweepConfig(
                        config_id=config_id,
                        benchmark=benchmark,
                        policy_type=cfg["policy_type"],
                        assoc_threshold=float(cfg["assoc_threshold"]),
                        sim_threshold=float(cfg["sim_threshold"]),
                        min_share=float(cfg.get("min_share", 0.05)),
                        min_count=int(cfg.get("min_count", 3)),
                        noise=float(cfg.get("noise", 0.0)),
                        epochs=int(cfg.get("epochs", 3)),
                        seed=seed,
                        policy_params=cfg.get("policy_params", "{}"),
                    )
                )
                config_id += 1

    logger.info(
        f"Generated {len(sweep_configs)} multi-seed configs: "
        f"{len(top_configs)} configs x {len(VALIDATION_SEEDS)} seeds "
        f"x {len(benchmarks)} benchmarks"
    )
    return sweep_configs


def run_multiseed_validation(
    db_path: str,
    features_dir: str,
    workers: int = 24,
    benchmarks: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Execute Phase 2: Multi-seed validation of top configs.

    Per D-33: 10 seeds per config.
    Per D-37: structural variance across seeds.

    Args:
        db_path: SQLite database path.
        features_dir: Directory with cached feature files.
        workers: Number of parallel workers.
        benchmarks: Benchmark names.

    Returns:
        List of result summaries.
    """
    configs = generate_multiseed_configs(db_path, top_n=10, benchmarks=benchmarks)
    if not configs:
        logger.warning("No multi-seed configs generated; skipping Phase 2")
        return []

    runner = SweepRunner(db_path, n_workers=workers)
    results = runner.run(configs, features_dir)

    ok = sum(1 for r in results if r["status"] == "ok")
    logger.info(f"Phase 2 complete: {ok}/{len(results)} multi-seed runs succeeded")

    # Print structural variance per D-37
    _print_structural_variance(db_path)

    return results


def _print_structural_variance(db_path: str) -> None:
    """Print structural variance across seeds per D-37.

    Reports mean +/- std for: accuracy, node count, max depth,
    branching factor, rumination rate.
    """
    conn = sqlite3.connect(db_path)
    try:
        # Get configs that have multiple seeds
        cursor = conn.execute(
            """
            SELECT
                policy_type, assoc_threshold, sim_threshold,
                benchmark,
                AVG(accuracy) as mean_acc,
                -- Use STDEV via group concat trick for SQLite
                COUNT(DISTINCT seed) as n_seeds,
                AVG(n_nodes) as mean_nodes,
                AVG(max_depth) as mean_depth,
                AVG(branching_factor_mean) as mean_bf,
                AVG(rumination_rejections) as mean_rum
            FROM sweep_results
            WHERE config_id >= ? AND config_id < ?
              AND epoch = (
                SELECT MAX(epoch) FROM sweep_results s2
                WHERE s2.config_id = sweep_results.config_id
                  AND s2.benchmark = sweep_results.benchmark
                  AND s2.seed = sweep_results.seed
              )
            GROUP BY policy_type, assoc_threshold, sim_threshold, benchmark
            HAVING n_seeds > 1
            ORDER BY mean_acc DESC
            LIMIT 20
            """,
            (MULTISEED_CONFIG_ID_OFFSET, MULTISEED_CONFIG_ID_OFFSET + 1000000),
        )
        rows = cursor.fetchall()
        if not rows:
            logger.info("No multi-seed results found for structural variance")
            return

        print("\n=== Structural Variance Across Seeds (D-37) ===")
        print(
            f"{'Policy':<12} {'Assoc':<8} {'Sim':<6} {'BM':<15} "
            f"{'Seeds':<6} {'Acc':<10} {'Nodes':<10} {'Depth':<8} "
            f"{'BF':<8} {'Rum':<8}"
        )
        print("-" * 90)
        for row in rows:
            policy, assoc, sim, bm, seeds, acc, nodes, depth, bf, rum = row
            print(
                f"{policy:<12} {assoc:<8.3f} {sim:<6.2f} {bm:<15} "
                f"{seeds:<6} {acc:<10.4f} {nodes:<10.1f} {depth:<8.1f} "
                f"{bf:<8.2f} {rum:<8.1f}"
            )
    except sqlite3.OperationalError as e:
        logger.warning(f"Could not print structural variance: {e}")
    finally:
        conn.close()


# ── Phase 3: Full-scale validation ────────────────────────────────────


def generate_fullscale_configs(
    db_path: str,
    top_n: int = 5,
    benchmarks: list[str] | None = None,
) -> list[SweepConfig]:
    """Generate full-scale validation configs per D-22, D-40.

    Top-N overall configs re-run on FULL datasets.

    Args:
        db_path: SQLite database path.
        top_n: Number of top configs to validate at full scale.
        benchmarks: Benchmark names.

    Returns:
        List of SweepConfig instances for full-scale validation.
    """
    if benchmarks is None:
        benchmarks = BENCHMARKS

    top_configs = _query_top_overall_configs(db_path, top_n=top_n)

    if not top_configs:
        logger.warning("No configs found in DB for full-scale validation")
        return []

    sweep_configs: list[SweepConfig] = []
    config_id = FULLSCALE_CONFIG_ID_OFFSET

    for cfg in top_configs:
        for benchmark in benchmarks:
            sweep_configs.append(
                SweepConfig(
                    config_id=config_id,
                    benchmark=benchmark,
                    policy_type=cfg["policy_type"],
                    assoc_threshold=float(cfg["assoc_threshold"]),
                    sim_threshold=float(cfg["sim_threshold"]),
                    min_share=float(cfg.get("min_share", 0.05)),
                    min_count=int(cfg.get("min_count", 3)),
                    noise=float(cfg.get("noise", 0.0)),
                    epochs=int(cfg.get("epochs", 3)),
                    seed=42,
                    policy_params=cfg.get("policy_params", "{}"),
                )
            )
            config_id += 1

    logger.info(
        f"Generated {len(sweep_configs)} full-scale configs: "
        f"{len(top_configs)} configs x {len(benchmarks)} benchmarks"
    )
    return sweep_configs


def run_fullscale_validation(
    db_path: str,
    features_dir: str,
    workers: int = 24,
    benchmarks: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Execute Phase 3: Full-scale validation on complete datasets.

    Per D-22: Re-run top configs on full datasets.
    Per D-40: Compare full-scale vs 10K subset accuracy (generalization gap).

    Note: This function expects the features_dir to contain FULL feature files
    (not 10K subsets). Use cache_features.py without --max-samples for full features.

    Args:
        db_path: SQLite database path.
        features_dir: Directory with full feature files.
        workers: Number of parallel workers.
        benchmarks: Benchmark names.

    Returns:
        List of result summaries.
    """
    configs = generate_fullscale_configs(db_path, top_n=5, benchmarks=benchmarks)
    if not configs:
        logger.warning("No full-scale configs generated; skipping Phase 3")
        return []

    runner = SweepRunner(db_path, n_workers=workers)
    results = runner.run(configs, features_dir)

    ok = sum(1 for r in results if r["status"] == "ok")
    logger.info(f"Phase 3 complete: {ok}/{len(results)} full-scale runs succeeded")

    # Print generalization gap per D-40
    _print_generalization_gap(db_path)

    return results


def _print_generalization_gap(db_path: str) -> None:
    """Print generalization gap between 10K subset and full training set per D-40.

    Compares full-scale results (config_id >= FULLSCALE_CONFIG_ID_OFFSET) against
    the same config's 10K subset results.
    """
    conn = sqlite3.connect(db_path)
    try:
        # Get full-scale results
        cursor = conn.execute(
            """
            SELECT
                fs.policy_type, fs.assoc_threshold, fs.sim_threshold,
                fs.benchmark, fs.accuracy as full_acc,
                sub.accuracy as subset_acc,
                (fs.accuracy - sub.accuracy) as gap
            FROM sweep_results fs
            LEFT JOIN sweep_results sub
                ON sub.policy_type = fs.policy_type
                AND sub.assoc_threshold = fs.assoc_threshold
                AND sub.sim_threshold = fs.sim_threshold
                AND sub.benchmark = fs.benchmark
                AND sub.seed = fs.seed
                AND sub.config_id < ?
                AND sub.epoch = (
                    SELECT MAX(epoch) FROM sweep_results s2
                    WHERE s2.config_id = sub.config_id
                      AND s2.benchmark = sub.benchmark
                      AND s2.seed = sub.seed
                )
            WHERE fs.config_id >= ? AND fs.config_id < ?
              AND fs.epoch = (
                SELECT MAX(epoch) FROM sweep_results s3
                WHERE s3.config_id = fs.config_id
                  AND s3.benchmark = fs.benchmark
                  AND s3.seed = fs.seed
              )
            ORDER BY fs.policy_type, fs.benchmark
            """,
            (FULLSCALE_CONFIG_ID_OFFSET, FULLSCALE_CONFIG_ID_OFFSET,
             FULLSCALE_CONFIG_ID_OFFSET + 1000000),
        )
        rows = cursor.fetchall()
        if not rows:
            logger.info("No full-scale results for generalization gap")
            return

        print("\n=== Generalization Gap: Full vs 10K Subset (D-40) ===")
        print(
            f"{'Policy':<12} {'Assoc':<8} {'Sim':<6} {'BM':<15} "
            f"{'Full':<10} {'10K':<10} {'Gap':<10}"
        )
        print("-" * 71)
        for policy, assoc, sim, bm, full_acc, subset_acc, gap in rows:
            sub_str = f"{subset_acc:.4f}" if subset_acc is not None else "N/A"
            gap_str = f"{gap:+.4f}" if gap is not None else "N/A"
            print(
                f"{policy:<12} {assoc:<8.3f} {sim:<6.2f} {bm:<15} "
                f"{full_acc:<10.4f} {sub_str:<10} {gap_str:<10}"
            )
    except sqlite3.OperationalError as e:
        logger.warning(f"Could not print generalization gap: {e}")
    finally:
        conn.close()


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    """Command-line entry point for hybrid validation."""
    parser = argparse.ArgumentParser(
        description="Hybrid strategy sweep and multi-seed validation (T2-08)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --phase 1 --features-dir results/T2/features --db results/T2/sweep.db
  %(prog)s --phase 2 --features-dir results/T2/features --db results/T2/sweep.db
  %(prog)s --phase 3 --features-dir results/T2/features_full --db results/T2/sweep.db
  %(prog)s --phase all --features-dir results/T2/features --db results/T2/sweep.db
        """,
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "2", "3", "all"],
        required=True,
        help="Execution phase: 1=hybrid sweep, 2=multi-seed, 3=full-scale, all=all three",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="results/T2/features",
        help="Directory with cached feature .pt files",
    )
    parser.add_argument(
        "--features-dir-full",
        type=str,
        default=None,
        help="Directory with FULL feature files (for phase 3; defaults to --features-dir)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="results/T2/sweep.db",
        help="SQLite database path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Number of parallel workers (default: 24, per D-24)",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=BENCHMARKS,
        help="Benchmarks to run (default: all 5)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    features_dir_full = args.features_dir_full or args.features_dir
    phases = ["1", "2", "3"] if args.phase == "all" else [args.phase]
    total_start = time.time()

    for phase in phases:
        logger.info(f"\n{'='*60}")
        logger.info(f"Phase {phase}")
        logger.info(f"{'='*60}")

        if phase == "1":
            run_hybrid_sweep(
                db_path=args.db,
                features_dir=args.features_dir,
                workers=args.workers,
                benchmarks=args.benchmarks,
            )
        elif phase == "2":
            run_multiseed_validation(
                db_path=args.db,
                features_dir=args.features_dir,
                workers=args.workers,
                benchmarks=args.benchmarks,
            )
        elif phase == "3":
            run_fullscale_validation(
                db_path=args.db,
                features_dir=features_dir_full,
                workers=args.workers,
                benchmarks=args.benchmarks,
            )

    total_time = time.time() - total_start
    logger.info(f"\nTotal time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
