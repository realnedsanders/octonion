"""Adaptive strategy sweep for strategies 1-3 (EMA, mean+std, depth-dependent).

Runs hyperparameter sweeps for adaptive ThresholdPolicy strategies, comparing
each against the best-tuned global baseline from the global sweep.

Per D-01: strategies tested in order -- EMA (1), mean+std (2), depth (3).
Per D-29: sweep adaptive hyperparameters for fair comparison.
Per D-03: co-adapt sim_threshold and consolidation with assoc_threshold.
Per D-31: fixed seed=42.
Per D-27: all configs run to completion.

Usage:
    python scripts/sweep/run_adaptive_sweep.py --strategy ema --features-dir results/T2/features --db results/T2/sweep.db --workers 24
    python scripts/sweep/run_adaptive_sweep.py --strategy mean_std --features-dir results/T2/features --db results/T2/sweep.db
    python scripts/sweep/run_adaptive_sweep.py --strategy depth --features-dir results/T2/features --db results/T2/sweep.db
    python scripts/sweep/run_adaptive_sweep.py --strategy all  # runs all 3
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

# Ensure scripts/ is on sys.path for sibling imports
_scripts_dir = str(Path(__file__).resolve().parent.parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from sweep.sweep_runner import (
    SweepConfig,
    SweepRunner,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────

# All 5 T1 benchmarks per D-07
ALL_BENCHMARKS = ["mnist", "fashion_mnist", "cifar10", "text_4class", "text_20class"]

# Fixed seed per D-31
SEED = 42

# Default epochs for initial sweep
DEFAULT_EPOCHS = 3

# ── Strategy hyperparameter grids ─────────────────────────────────

# Per D-01, D-29: EMA alpha and k values
EMA_ALPHA_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5]
EMA_K_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0]

# Per D-01: mean+std k values
MEAN_STD_K_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0]

# Per D-01: depth decay_factor values (both directions per D-01)
DEPTH_DECAY_VALUES = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]


# ── Query best global configs from DB ────────────────────────────


def get_top_global_assoc_thresholds(
    db_path: str, n: int = 5
) -> list[float]:
    """Query top-N assoc_threshold values from global sweep by mean accuracy.

    Uses final-epoch results from global sweep (policy_type='global').

    Args:
        db_path: Path to SQLite database.
        n: Number of top values to return.

    Returns:
        List of assoc_threshold values, or defaults if DB has no global results.
    """
    defaults = [0.05, 0.1, 0.2, 0.3, 0.5]
    conn = sqlite3.connect(db_path)
    try:
        # Check if global results exist
        count = conn.execute(
            "SELECT COUNT(*) FROM sweep_results WHERE policy_type = 'global'"
        ).fetchone()[0]
        if count == 0:
            logger.warning("No global sweep results found -- using default assoc thresholds")
            return defaults[:n]

        rows = conn.execute(
            """
            SELECT assoc_threshold, AVG(accuracy) as mean_acc
            FROM sweep_results
            WHERE policy_type = 'global'
              AND epoch = (
                  SELECT MAX(epoch) FROM sweep_results WHERE policy_type = 'global'
              )
            GROUP BY assoc_threshold
            ORDER BY mean_acc DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()

        if not rows:
            return defaults[:n]

        return [row[0] for row in rows]
    except sqlite3.OperationalError:
        logger.warning("Database not initialized -- using default assoc thresholds")
        return defaults[:n]
    finally:
        conn.close()


def get_top_global_sim_thresholds(
    db_path: str, n: int = 3
) -> list[float]:
    """Query top-N sim_threshold values from global sweep by mean accuracy.

    Per D-03: co-adapt sim_threshold with adaptive hyperparameters.

    Args:
        db_path: Path to SQLite database.
        n: Number of top values to return.

    Returns:
        List of sim_threshold values, or defaults if DB has no global results.
    """
    defaults = [0.0, 0.05, 0.1]
    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM sweep_results WHERE policy_type = 'global'"
        ).fetchone()[0]
        if count == 0:
            logger.warning("No global sweep results found -- using default sim thresholds")
            return defaults[:n]

        rows = conn.execute(
            """
            SELECT sim_threshold, AVG(accuracy) as mean_acc
            FROM sweep_results
            WHERE policy_type = 'global'
              AND epoch = (
                  SELECT MAX(epoch) FROM sweep_results WHERE policy_type = 'global'
              )
            GROUP BY sim_threshold
            ORDER BY mean_acc DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()

        if not rows:
            return defaults[:n]

        return [row[0] for row in rows]
    except sqlite3.OperationalError:
        logger.warning("Database not initialized -- using default sim thresholds")
        return defaults[:n]
    finally:
        conn.close()


def get_top_global_consolidation_configs(
    db_path: str, n: int = 2
) -> list[tuple[float, int]]:
    """Query top-N consolidation configs from global sweep.

    Per D-03: co-adapt consolidation alongside threshold.

    Args:
        db_path: Path to SQLite database.
        n: Number of top configs to return.

    Returns:
        List of (min_share, min_count) tuples, or defaults.
    """
    defaults = [(0.05, 3), (0.03, 2)]
    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM sweep_results WHERE policy_type = 'global'"
        ).fetchone()[0]
        if count == 0:
            return defaults[:n]

        rows = conn.execute(
            """
            SELECT min_share, min_count, AVG(accuracy) as mean_acc
            FROM sweep_results
            WHERE policy_type = 'global'
              AND epoch = (
                  SELECT MAX(epoch) FROM sweep_results WHERE policy_type = 'global'
              )
            GROUP BY min_share, min_count
            ORDER BY mean_acc DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()

        if not rows:
            return defaults[:n]

        return [(row[0], row[1]) for row in rows]
    except sqlite3.OperationalError:
        return defaults[:n]
    finally:
        conn.close()


def get_best_global_noise(db_path: str) -> float:
    """Query the best noise value from the global sweep.

    Per D-05: include noise revisit as interaction effect.

    Args:
        db_path: Path to SQLite database.

    Returns:
        Best noise value, or 0.01 as default.
    """
    default = 0.01
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT noise, AVG(accuracy) as mean_acc
            FROM sweep_results
            WHERE policy_type = 'global'
              AND noise > 0.0
              AND epoch = (
                  SELECT MAX(epoch) FROM sweep_results WHERE policy_type = 'global'
              )
            GROUP BY noise
            ORDER BY mean_acc DESC
            LIMIT 1
            """,
        ).fetchone()

        return row[0] if row else default
    except sqlite3.OperationalError:
        return default
    finally:
        conn.close()


def get_best_global_per_benchmark(db_path: str) -> dict[str, dict[str, Any]]:
    """Query best global accuracy per benchmark for comparison.

    Returns:
        Dict mapping benchmark name to best config details.
    """
    results: dict[str, dict[str, Any]] = {}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for bm in ALL_BENCHMARKS:
            row = conn.execute(
                """
                SELECT assoc_threshold, sim_threshold, min_share, min_count,
                       noise, accuracy, n_nodes, config_id
                FROM sweep_results
                WHERE benchmark = ? AND policy_type = 'global'
                  AND epoch = (
                      SELECT MAX(epoch) FROM sweep_results
                      WHERE benchmark = ? AND policy_type = 'global'
                  )
                ORDER BY accuracy DESC
                LIMIT 1
                """,
                (bm, bm),
            ).fetchone()

            if row:
                results[bm] = dict(row)
            else:
                results[bm] = {"accuracy": 0.0}

        return results
    except sqlite3.OperationalError:
        return {bm: {"accuracy": 0.0} for bm in ALL_BENCHMARKS}
    finally:
        conn.close()


# ── Config generation per strategy ────────────────────────────────


def generate_ema_sweep_configs(
    benchmarks: list[str],
    db_path: str,
    seed: int = SEED,
) -> list[SweepConfig]:
    """Generate sweep configs for PerNodeEMAPolicy (Strategy 1).

    Per D-29: sweep alpha x k x base_assoc(top-5) x sim(top-3).
    Initial pass: fix noise=0, consolidation=(0.05,3) for tractable sweep.
    Total initial: 5(alpha) * 5(k) * 5(base_assoc) * 3(sim) = 375 per benchmark.

    Args:
        benchmarks: List of benchmark names.
        db_path: Path to SQLite database for querying global best configs.
        seed: Random seed.

    Returns:
        List of SweepConfig instances.
    """
    top_assoc = get_top_global_assoc_thresholds(db_path, n=5)
    top_sim = get_top_global_sim_thresholds(db_path, n=3)

    configs: list[SweepConfig] = []
    config_id = 300000  # Offset for EMA strategy

    for benchmark in benchmarks:
        for alpha in EMA_ALPHA_VALUES:
            for k in EMA_K_VALUES:
                for base_assoc in top_assoc:
                    for sim in top_sim:
                        # Policy params match PerNodeEMAPolicy constructor names
                        policy_params = {
                            "alpha": alpha,
                            "k": k,
                            "base_assoc": float(base_assoc),
                            "sim_threshold": float(sim),
                            "min_share": 0.05,
                            "min_count": 3,
                        }
                        configs.append(
                            SweepConfig(
                                config_id=config_id,
                                benchmark=benchmark,
                                policy_type="ema",
                                assoc_threshold=float(base_assoc),
                                sim_threshold=float(sim),
                                min_share=0.05,
                                min_count=3,
                                noise=0.0,
                                epochs=DEFAULT_EPOCHS,
                                seed=seed,
                                policy_params=json.dumps(policy_params),
                            )
                        )
                        config_id += 1

    return configs


def generate_mean_std_sweep_configs(
    benchmarks: list[str],
    db_path: str,
    seed: int = SEED,
) -> list[SweepConfig]:
    """Generate sweep configs for PerNodeMeanStdPolicy (Strategy 2).

    Per D-01: sweep k x base_assoc(top-5) x sim(top-3).
    Total initial: 5(k) * 5(base_assoc) * 3(sim) = 75 per benchmark.

    Args:
        benchmarks: List of benchmark names.
        db_path: Path to SQLite database.
        seed: Random seed.

    Returns:
        List of SweepConfig instances.
    """
    top_assoc = get_top_global_assoc_thresholds(db_path, n=5)
    top_sim = get_top_global_sim_thresholds(db_path, n=3)

    configs: list[SweepConfig] = []
    config_id = 400000  # Offset for mean_std strategy

    for benchmark in benchmarks:
        for k in MEAN_STD_K_VALUES:
            for base_assoc in top_assoc:
                for sim in top_sim:
                    # Note: constructor uses 'k' not 'n_std'
                    policy_params = {
                        "k": k,
                        "base_assoc": float(base_assoc),
                        "sim_threshold": float(sim),
                        "min_share": 0.05,
                        "min_count": 3,
                    }
                    configs.append(
                        SweepConfig(
                            config_id=config_id,
                            benchmark=benchmark,
                            policy_type="mean_std",
                            assoc_threshold=float(base_assoc),
                            sim_threshold=float(sim),
                            min_share=0.05,
                            min_count=3,
                            noise=0.0,
                            epochs=DEFAULT_EPOCHS,
                            seed=seed,
                            policy_params=json.dumps(policy_params),
                        )
                    )
                    config_id += 1

    return configs


def generate_depth_sweep_configs(
    benchmarks: list[str],
    db_path: str,
    seed: int = SEED,
) -> list[SweepConfig]:
    """Generate sweep configs for DepthPolicy (Strategy 3).

    Per D-01: sweep decay_factor (both directions) x base_assoc(top-5) x sim(top-3).
    Total initial: 7(decay) * 5(base_assoc) * 3(sim) = 105 per benchmark.

    Args:
        benchmarks: List of benchmark names.
        db_path: Path to SQLite database.
        seed: Random seed.

    Returns:
        List of SweepConfig instances.
    """
    top_assoc = get_top_global_assoc_thresholds(db_path, n=5)
    top_sim = get_top_global_sim_thresholds(db_path, n=3)

    configs: list[SweepConfig] = []
    config_id = 500000  # Offset for depth strategy

    for benchmark in benchmarks:
        for decay in DEPTH_DECAY_VALUES:
            for base_assoc in top_assoc:
                for sim in top_sim:
                    policy_params = {
                        "base_assoc": float(base_assoc),
                        "decay_factor": decay,
                        "sim_threshold": float(sim),
                        "min_share": 0.05,
                        "min_count": 3,
                    }
                    configs.append(
                        SweepConfig(
                            config_id=config_id,
                            benchmark=benchmark,
                            policy_type="depth",
                            assoc_threshold=float(base_assoc),
                            sim_threshold=float(sim),
                            min_share=0.05,
                            min_count=3,
                            noise=0.0,
                            epochs=DEFAULT_EPOCHS,
                            seed=seed,
                            policy_params=json.dumps(policy_params),
                        )
                    )
                    config_id += 1

    return configs


# ── Expanded sweep for top configs ────────────────────────────────


def generate_expanded_sweep_configs(
    strategy: str,
    db_path: str,
    benchmarks: list[str],
    n_top: int = 10,
    seed: int = SEED,
) -> list[SweepConfig]:
    """Generate expanded sweep on top-N configs with full consolidation, noise, epoch grids.

    After initial sweep, run expanded sweep on top configs per strategy:
    - Full consolidation sweep (5 configs)
    - Full noise sweep (4 values)
    - Full epoch sweep (1, 3, 5)

    Args:
        strategy: One of "ema", "mean_std", "depth".
        db_path: Path to SQLite database.
        benchmarks: List of benchmark names.
        n_top: Number of top configs per strategy to expand.
        seed: Random seed.

    Returns:
        List of SweepConfig instances for expanded sweep.
    """
    # Query top-N configs for this strategy
    top_configs = _get_top_strategy_configs(db_path, strategy, n=n_top)

    if not top_configs:
        logger.warning(f"No initial results for {strategy} -- skipping expanded sweep")
        return []

    # Expanded grids
    consolidation_grid = [
        (0.01, 1),
        (0.03, 2),
        (0.05, 3),
        (0.10, 5),
        (0.00, 0),  # Disabled
    ]
    noise_grid = [0.0, 0.01, 0.05, 0.1]
    epoch_grid = [1, 3, 5]

    configs: list[SweepConfig] = []
    # Use high offset to avoid collision
    config_id_base = {
        "ema": 600000,
        "mean_std": 700000,
        "depth": 800000,
    }
    config_id = config_id_base.get(strategy, 900000)

    for top_cfg in top_configs:
        base_policy_params = json.loads(top_cfg["policy_params"])

        for benchmark in benchmarks:
            for min_share, min_count in consolidation_grid:
                for noise in noise_grid:
                    for epochs in epoch_grid:
                        # Update policy params with consolidation co-sweep
                        policy_params = dict(base_policy_params)
                        policy_params["min_share"] = min_share
                        policy_params["min_count"] = min_count

                        configs.append(
                            SweepConfig(
                                config_id=config_id,
                                benchmark=benchmark,
                                policy_type=strategy,
                                assoc_threshold=top_cfg["assoc_threshold"],
                                sim_threshold=top_cfg["sim_threshold"],
                                min_share=min_share,
                                min_count=min_count,
                                noise=noise,
                                epochs=epochs,
                                seed=seed,
                                policy_params=json.dumps(policy_params),
                            )
                        )
                        config_id += 1

    return configs


def _get_top_strategy_configs(
    db_path: str, strategy: str, n: int = 10
) -> list[dict[str, Any]]:
    """Query top-N configs for a strategy by mean accuracy across benchmarks.

    Args:
        db_path: Path to SQLite database.
        strategy: Policy type to query.
        n: Number of top configs.

    Returns:
        List of config dicts with assoc_threshold, sim_threshold, policy_params.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT assoc_threshold, sim_threshold, policy_params,
                   AVG(accuracy) as mean_acc,
                   COUNT(DISTINCT benchmark) as n_benchmarks
            FROM sweep_results
            WHERE policy_type = ?
              AND epoch = (
                  SELECT MAX(epoch) FROM sweep_results WHERE policy_type = ?
              )
            GROUP BY assoc_threshold, sim_threshold, policy_params
            HAVING n_benchmarks >= 1
            ORDER BY mean_acc DESC
            LIMIT ?
            """,
            (strategy, strategy, n),
        ).fetchall()

        return [dict(row) for row in rows]
    finally:
        conn.close()


# ── Comparison table output ───────────────────────────────────────


def print_comparison_table(db_path: str, strategy: str, benchmarks: list[str]) -> None:
    """Print comparison table: strategy best vs global best per benchmark.

    Per D-29: fair comparison requires tuned adaptive vs tuned global.

    Args:
        db_path: Path to SQLite database.
        strategy: Strategy name to compare.
        benchmarks: List of benchmark names.
    """
    global_best = get_best_global_per_benchmark(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"COMPARISON: {strategy.upper()} vs GLOBAL BEST")
        logger.info("=" * 80)
        logger.info(
            f"{'Benchmark':<15} {'Global Best':>12} {'Strategy Best':>14} "
            f"{'Delta':>8} {'Delta%':>8} {'Strategy Config'}"
        )
        logger.info("-" * 80)

        strategy_wins = 0
        total_delta = 0.0
        n_compared = 0

        for bm in benchmarks:
            # Get best for this strategy on this benchmark
            row = conn.execute(
                """
                SELECT assoc_threshold, sim_threshold, policy_params,
                       accuracy, n_nodes, config_id
                FROM sweep_results
                WHERE benchmark = ? AND policy_type = ?
                  AND epoch = (
                      SELECT MAX(epoch) FROM sweep_results
                      WHERE benchmark = ? AND policy_type = ?
                  )
                ORDER BY accuracy DESC
                LIMIT 1
                """,
                (bm, strategy, bm, strategy),
            ).fetchone()

            g_acc = global_best.get(bm, {}).get("accuracy", 0.0)

            if row:
                s_acc = row["accuracy"]
                delta = s_acc - g_acc
                delta_pct = (delta / g_acc * 100) if g_acc > 0 else 0.0

                # Parse policy params for display
                pp = json.loads(row["policy_params"])
                pp_str = ", ".join(f"{k}={v}" for k, v in sorted(pp.items())
                                  if k not in ("min_share", "min_count",
                                               "sim_threshold", "base_assoc"))

                logger.info(
                    f"{bm:<15} {g_acc:>12.4f} {s_acc:>14.4f} "
                    f"{delta:>+8.4f} {delta_pct:>+7.1f}% {pp_str}"
                )

                if delta > 0:
                    strategy_wins += 1
                total_delta += delta
                n_compared += 1
            else:
                logger.info(f"{bm:<15} {g_acc:>12.4f} {'N/A':>14}")

        logger.info("-" * 80)
        if n_compared > 0:
            logger.info(
                f"{'SUMMARY':<15} {'Wins':>12}: {strategy_wins}/{n_compared}  "
                f"{'Mean delta':>14}: {total_delta / n_compared:+.4f}"
            )

        # Cross-benchmark ranking (per D-36 Friedman approach)
        _print_cross_benchmark_ranking(conn, strategy, benchmarks)

    finally:
        conn.close()


def _print_cross_benchmark_ranking(
    conn: sqlite3.Connection,
    strategy: str,
    benchmarks: list[str],
) -> None:
    """Print best config by mean rank across benchmarks (D-36 Friedman approach).

    Args:
        conn: Open SQLite connection.
        strategy: Strategy name.
        benchmarks: List of benchmark names.
    """
    # Get top-5 configs by mean accuracy across benchmarks
    rows = conn.execute(
        """
        SELECT assoc_threshold, sim_threshold, policy_params,
               AVG(accuracy) as mean_acc,
               MIN(accuracy) as min_acc,
               MAX(accuracy) as max_acc,
               COUNT(DISTINCT benchmark) as n_benchmarks
        FROM sweep_results
        WHERE policy_type = ?
          AND epoch = (
              SELECT MAX(epoch) FROM sweep_results WHERE policy_type = ?
          )
        GROUP BY assoc_threshold, sim_threshold, policy_params
        HAVING n_benchmarks >= 1
        ORDER BY mean_acc DESC
        LIMIT 5
        """,
        (strategy, strategy),
    ).fetchall()

    if rows:
        logger.info("")
        logger.info(f"Top-5 {strategy} configs by mean accuracy across benchmarks:")
        for i, row in enumerate(rows, 1):
            pp = json.loads(row["policy_params"])
            pp_str = ", ".join(f"{k}={v}" for k, v in sorted(pp.items())
                              if k not in ("min_share", "min_count",
                                           "sim_threshold", "base_assoc"))
            logger.info(
                f"  {i}. mean={row['mean_acc']:.4f} "
                f"[{row['min_acc']:.4f}, {row['max_acc']:.4f}] "
                f"({row['n_benchmarks']} bm) -- {pp_str}"
            )


# ── Strategy runner ───────────────────────────────────────────────


def run_strategy_sweep(
    strategy: str,
    runner: SweepRunner,
    features_dir: str,
    benchmarks: list[str],
    db_path: str,
    expanded: bool = False,
) -> list[dict]:
    """Run initial (and optionally expanded) sweep for a single strategy.

    Args:
        strategy: One of "ema", "mean_std", "depth".
        runner: SweepRunner instance.
        features_dir: Directory containing cached feature .pt files.
        benchmarks: List of benchmark names.
        db_path: Path to SQLite database.
        expanded: Whether to run expanded sweep after initial.

    Returns:
        List of result summaries.
    """
    logger.info("=" * 70)
    logger.info(f"STRATEGY SWEEP: {strategy.upper()}")
    logger.info("=" * 70)

    # 1. Generate initial configs
    config_generators = {
        "ema": generate_ema_sweep_configs,
        "mean_std": generate_mean_std_sweep_configs,
        "depth": generate_depth_sweep_configs,
    }

    if strategy not in config_generators:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. "
            f"Expected one of: {list(config_generators.keys())}"
        )

    configs = config_generators[strategy](benchmarks, db_path, seed=SEED)

    n_per_bm = len(configs) // max(len(benchmarks), 1)
    logger.info(f"  Strategy: {strategy}")
    logger.info(f"  Configs per benchmark: {n_per_bm}")
    logger.info(f"  Total configs: {len(configs)}")
    logger.info(f"  Benchmarks: {', '.join(benchmarks)}")
    logger.info(f"  Epochs: {DEFAULT_EPOCHS}")

    # 2. Run initial sweep
    t0 = time.time()
    results = runner.run(configs, features_dir)
    elapsed = time.time() - t0

    ok_count = sum(1 for r in results if r["status"] == "ok")
    logger.info(f"\nInitial {strategy} sweep: {ok_count}/{len(results)} succeeded in {elapsed:.1f}s")

    # 3. Print comparison table
    print_comparison_table(db_path, strategy, benchmarks)

    # 4. Optionally run expanded sweep on top configs
    if expanded:
        logger.info(f"\n--- Expanded sweep for top-10 {strategy} configs ---")
        expanded_configs = generate_expanded_sweep_configs(
            strategy, db_path, benchmarks, n_top=10, seed=SEED
        )

        if expanded_configs:
            logger.info(f"  Expanded configs: {len(expanded_configs)}")
            t1 = time.time()
            expanded_results = runner.run(expanded_configs, features_dir)
            elapsed2 = time.time() - t1

            ok2 = sum(1 for r in expanded_results if r["status"] == "ok")
            logger.info(
                f"\nExpanded {strategy} sweep: {ok2}/{len(expanded_results)} "
                f"succeeded in {elapsed2:.1f}s"
            )
            results.extend(expanded_results)

            # Reprint comparison after expanded sweep
            print_comparison_table(db_path, strategy, benchmarks)

    return results


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    """Command-line entry point for adaptive strategy sweep."""
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive strategy sweep for strategies 1-3 "
            "(EMA, mean+std, depth-dependent) per D-01, D-29"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run EMA sweep only:
  python scripts/sweep/run_adaptive_sweep.py --strategy ema --features-dir results/T2/features

  # Run all 3 strategies sequentially:
  python scripts/sweep/run_adaptive_sweep.py --strategy all --features-dir results/T2/features

  # Run with expanded sweep on top configs:
  python scripts/sweep/run_adaptive_sweep.py --strategy ema --expanded --features-dir results/T2/features

  # Custom workers and database:
  python scripts/sweep/run_adaptive_sweep.py --strategy all --workers 8 --db custom.db --features-dir features/
        """,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        choices=["ema", "mean_std", "depth", "all"],
        help=(
            "Strategy to sweep: ema (Strategy 1), mean_std (Strategy 2), "
            "depth (Strategy 3), or all (default: all)"
        ),
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="results/T2/features",
        help="Directory containing cached feature .pt files (default: results/T2/features)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="results/T2/sweep.db",
        help="SQLite database path (default: results/T2/sweep.db)",
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
        default=ALL_BENCHMARKS,
        help="Benchmarks to sweep (default: all 5 T1 benchmarks)",
    )
    parser.add_argument(
        "--expanded",
        action="store_true",
        default=False,
        help="Run expanded sweep on top-10 configs after initial sweep",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Ensure output directory exists
    db_dir = Path(args.db).parent
    db_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("ADAPTIVE STRATEGY SWEEP (Strategies 1-3)")
    logger.info("=" * 70)
    logger.info(f"  Strategy: {args.strategy}")
    logger.info(f"  Features: {args.features_dir}")
    logger.info(f"  Database: {args.db}")
    logger.info(f"  Workers: {args.workers}")
    logger.info(f"  Benchmarks: {args.benchmarks}")
    logger.info(f"  Expanded: {args.expanded}")
    logger.info(f"  Seed: {SEED}")

    runner = SweepRunner(args.db, n_workers=args.workers)

    t_total = time.time()

    strategies_to_run = (
        ["ema", "mean_std", "depth"] if args.strategy == "all" else [args.strategy]
    )

    all_results: list[dict] = []
    for strategy in strategies_to_run:
        results = run_strategy_sweep(
            strategy=strategy,
            runner=runner,
            features_dir=args.features_dir,
            benchmarks=args.benchmarks,
            db_path=args.db,
            expanded=args.expanded,
        )
        all_results.extend(results)

    total_elapsed = time.time() - t_total
    logger.info(f"\n{'=' * 70}")
    logger.info(f"ADAPTIVE SWEEP COMPLETE in {total_elapsed:.1f}s")
    logger.info(f"{'=' * 70}")

    # Final summary
    ok_total = sum(1 for r in all_results if r["status"] == "ok")
    logger.info(f"  Total configs run: {len(all_results)}")
    logger.info(f"  Succeeded: {ok_total}")
    logger.info(f"  Failed: {len(all_results) - ok_total}")

    # Final cross-strategy comparison
    if len(strategies_to_run) > 1:
        _print_final_comparison(args.db, args.benchmarks)


def _print_final_comparison(db_path: str, benchmarks: list[str]) -> None:
    """Print final comparison across all strategies and global baseline.

    Args:
        db_path: Path to SQLite database.
        benchmarks: List of benchmark names.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        logger.info("")
        logger.info("=" * 80)
        logger.info("FINAL CROSS-STRATEGY COMPARISON")
        logger.info("=" * 80)

        strategies = ["global", "ema", "mean_std", "depth"]

        # Header
        header = f"{'Benchmark':<15}"
        for s in strategies:
            header += f" {s:>12}"
        logger.info(header)
        logger.info("-" * 80)

        # Per-benchmark best accuracy
        strategy_means: dict[str, list[float]] = {s: [] for s in strategies}

        for bm in benchmarks:
            line = f"{bm:<15}"
            for s in strategies:
                row = conn.execute(
                    """
                    SELECT accuracy
                    FROM sweep_results
                    WHERE benchmark = ? AND policy_type = ?
                      AND epoch = (
                          SELECT MAX(epoch) FROM sweep_results
                          WHERE benchmark = ? AND policy_type = ?
                      )
                    ORDER BY accuracy DESC
                    LIMIT 1
                    """,
                    (bm, s, bm, s),
                ).fetchone()

                if row:
                    acc = row["accuracy"]
                    line += f" {acc:>12.4f}"
                    strategy_means[s].append(acc)
                else:
                    line += f" {'N/A':>12}"

            logger.info(line)

        # Mean across benchmarks
        logger.info("-" * 80)
        line = f"{'Mean':<15}"
        for s in strategies:
            vals = strategy_means[s]
            if vals:
                line += f" {sum(vals) / len(vals):>12.4f}"
            else:
                line += f" {'N/A':>12}"
        logger.info(line)

        # Identify overall winner
        best_strategy = None
        best_mean = -1.0
        for s, vals in strategy_means.items():
            if vals:
                mean = sum(vals) / len(vals)
                if mean > best_mean:
                    best_mean = mean
                    best_strategy = s

        if best_strategy:
            logger.info(f"\nBest overall strategy: {best_strategy} (mean accuracy: {best_mean:.4f})")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
