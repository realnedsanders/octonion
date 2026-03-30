"""Global threshold sensitivity sweep across all 5 benchmarks.

Orchestrates a 3-phase progressive sweep design per D-20 and D-22:
  Phase 1: Core 3D sweep (assoc x sim x noise) with fixed consolidation/epochs
  Phase 2: Consolidation sweep on top-5 configs from Phase 1
  Phase 3: Epoch sweep on top-10 configs from Phases 1+2

All sweeps run on 10K subsets per D-22 reduced-first approach.
Fixed seed=42 per D-31.

Usage:
    python scripts/sweep/run_global_sweep.py --phase 1 --features-dir results/T2/features --db results/T2/sweep.db --workers 24
    python scripts/sweep/run_global_sweep.py --phase 2 --features-dir results/T2/features --db results/T2/sweep.db --workers 24
    python scripts/sweep/run_global_sweep.py --phase 3 --features-dir results/T2/features --db results/T2/sweep.db --workers 24
    python scripts/sweep/run_global_sweep.py --phase all  # runs 1, 2, 3 sequentially
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

# Ensure scripts/ is on sys.path for sibling imports
_scripts_dir = str(Path(__file__).resolve().parent.parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from sweep.sweep_runner import (
    SweepConfig,
    SweepRunner,
    generate_consolidation_sweep_configs,
    generate_epoch_sweep_configs,
    generate_global_sweep_configs,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────

# All 5 T1 benchmarks per D-07
ALL_BENCHMARKS = ["mnist", "fashion_mnist", "cifar10", "text_4class", "text_20class"]

# Fixed seed per D-31
SEED = 42

# Phase 1: Core 3D sweep dimensions per D-20
ASSOC_THRESHOLDS = np.unique(
    np.sort(
        np.concatenate(
            [
                np.geomspace(0.001, 2.0, 15),
                np.linspace(0.05, 1.0, 10),
            ]
        )
    )
)

SIM_THRESHOLDS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

NOISE_VALUES = [0.0, 0.01, 0.05, 0.1]

# Phase 2: Consolidation sweep configs per D-20
CONSOLIDATION_CONFIGS = [
    (0.01, 1),
    (0.03, 2),
    (0.05, 3),
    (0.10, 5),
    (0.00, 0),  # Disabled
]

# Phase 3: Epoch sweep values per D-06
EPOCH_VALUES = [1, 3, 5]


# ── Phase 1: Core 3D sweep ────────────────────────────────────────


def run_phase1(
    runner: SweepRunner,
    features_dir: str,
    benchmarks: list[str],
) -> list[dict]:
    """Phase 1: Core 3D sweep (assoc x sim x noise).

    Fixed consolidation=(0.05, 3), epochs=3 per D-22 reduced-first approach.

    Args:
        runner: SweepRunner instance with initialized DB.
        features_dir: Directory containing cached feature .pt files.
        benchmarks: List of benchmark names to sweep.

    Returns:
        List of result summaries from SweepRunner.run().
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: Core 3D Sweep (assoc x sim x noise)")
    logger.info("=" * 60)

    configs = generate_global_sweep_configs(benchmarks, seed=SEED)

    n_assoc = len(ASSOC_THRESHOLDS)
    n_sim = len(SIM_THRESHOLDS)
    n_noise = len(NOISE_VALUES)
    n_benchmarks = len(benchmarks)

    logger.info(f"  Assoc thresholds: {n_assoc} values [{ASSOC_THRESHOLDS[0]:.4f} ... {ASSOC_THRESHOLDS[-1]:.4f}]")
    logger.info(f"  Sim thresholds: {n_sim} values {SIM_THRESHOLDS}")
    logger.info(f"  Noise values: {n_noise} values {NOISE_VALUES}")
    logger.info(f"  Benchmarks: {n_benchmarks} ({', '.join(benchmarks)})")
    logger.info(f"  Total configs: {len(configs)} ({n_assoc} x {n_sim} x {n_noise} x {n_benchmarks})")
    logger.info(f"  Fixed: epochs=3, consolidation=(0.05, 3)")

    t0 = time.time()
    results = runner.run(configs, features_dir)
    elapsed = time.time() - t0

    ok_count = sum(1 for r in results if r["status"] == "ok")
    logger.info(f"\nPhase 1 complete: {ok_count}/{len(results)} succeeded in {elapsed:.1f}s")

    _print_phase1_summary(runner.db_path, benchmarks)

    return results


def _print_phase1_summary(db_path: str, benchmarks: list[str]) -> None:
    """Print Phase 1 summary: best accuracy per benchmark and top-5 global configs."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        logger.info("\n--- Phase 1 Summary ---")
        logger.info("\nBest accuracy per benchmark (final epoch):")

        for bm in benchmarks:
            row = conn.execute(
                """
                SELECT assoc_threshold, sim_threshold, noise, accuracy, n_nodes
                FROM sweep_results
                WHERE benchmark = ? AND epoch = (
                    SELECT MAX(epoch) FROM sweep_results WHERE benchmark = ?
                )
                ORDER BY accuracy DESC
                LIMIT 1
                """,
                (bm, bm),
            ).fetchone()

            if row:
                logger.info(
                    f"  {bm}: {row['accuracy']:.4f} "
                    f"(assoc={row['assoc_threshold']:.4f}, sim={row['sim_threshold']:.2f}, "
                    f"noise={row['noise']:.2f}, nodes={row['n_nodes']})"
                )

        logger.info("\nTop-5 (assoc, sim) pairs by mean accuracy across benchmarks:")
        rows = conn.execute(
            """
            SELECT assoc_threshold, sim_threshold,
                   AVG(accuracy) as mean_acc,
                   MIN(accuracy) as min_acc,
                   MAX(accuracy) as max_acc,
                   COUNT(DISTINCT benchmark) as n_benchmarks
            FROM sweep_results
            WHERE epoch = (SELECT MAX(epoch) FROM sweep_results)
              AND noise = 0.0
            GROUP BY assoc_threshold, sim_threshold
            HAVING n_benchmarks >= 3
            ORDER BY mean_acc DESC
            LIMIT 5
            """,
        ).fetchall()

        for i, row in enumerate(rows, 1):
            logger.info(
                f"  {i}. assoc={row['assoc_threshold']:.4f}, "
                f"sim={row['sim_threshold']:.2f}: "
                f"mean={row['mean_acc']:.4f} "
                f"[{row['min_acc']:.4f}, {row['max_acc']:.4f}] "
                f"({row['n_benchmarks']} benchmarks)"
            )

    finally:
        conn.close()


# ── Phase 2: Consolidation sweep ──────────────────────────────────


def run_phase2(
    runner: SweepRunner,
    features_dir: str,
    benchmarks: list[str],
) -> list[dict]:
    """Phase 2: Consolidation sweep on top-5 (assoc, sim) pairs from Phase 1.

    Queries Phase 1 results for top-5 (assoc, sim) pairs by mean accuracy,
    then sweeps 5 consolidation configs on those pairs.

    Args:
        runner: SweepRunner instance.
        features_dir: Directory containing cached feature .pt files.
        benchmarks: List of benchmark names.

    Returns:
        List of result summaries.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: Consolidation Sweep")
    logger.info("=" * 60)

    # Query top-5 (assoc, sim) pairs from Phase 1
    top_pairs = _get_top_pairs(runner.db_path, n=5)

    if not top_pairs:
        logger.warning("No Phase 1 results found -- using defaults")
        top_pairs = [(0.3, 0.1)]

    logger.info(f"  Top {len(top_pairs)} (assoc, sim) pairs from Phase 1:")
    for assoc, sim in top_pairs:
        logger.info(f"    assoc={assoc:.4f}, sim={sim:.2f}")

    # Generate consolidation sweep configs for each top pair
    configs: list[SweepConfig] = []
    config_id = 100000

    for assoc, sim in top_pairs:
        for benchmark in benchmarks:
            for min_share, min_count in CONSOLIDATION_CONFIGS:
                configs.append(
                    SweepConfig(
                        config_id=config_id,
                        benchmark=benchmark,
                        policy_type="global",
                        assoc_threshold=float(assoc),
                        sim_threshold=float(sim),
                        min_share=min_share,
                        min_count=min_count,
                        noise=0.0,
                        epochs=3,
                        seed=SEED,
                    )
                )
                config_id += 1

    logger.info(f"  Consolidation configs: {len(CONSOLIDATION_CONFIGS)}")
    logger.info(f"  Total configs: {len(configs)} ({len(top_pairs)} pairs x {len(CONSOLIDATION_CONFIGS)} consol x {len(benchmarks)} benchmarks)")

    t0 = time.time()
    results = runner.run(configs, features_dir)
    elapsed = time.time() - t0

    ok_count = sum(1 for r in results if r["status"] == "ok")
    logger.info(f"\nPhase 2 complete: {ok_count}/{len(results)} succeeded in {elapsed:.1f}s")

    _print_phase2_summary(runner.db_path, benchmarks)

    return results


def _get_top_pairs(
    db_path: str, n: int = 5
) -> list[tuple[float, float]]:
    """Query top-N (assoc_threshold, sim_threshold) pairs by mean accuracy.

    Uses Phase 1 results (noise=0.0 baseline) grouped by (assoc, sim).

    Args:
        db_path: Path to SQLite database.
        n: Number of top pairs to return.

    Returns:
        List of (assoc_threshold, sim_threshold) tuples.
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT assoc_threshold, sim_threshold,
                   AVG(accuracy) as mean_acc
            FROM sweep_results
            WHERE epoch = (SELECT MAX(epoch) FROM sweep_results WHERE config_id < 100000)
              AND config_id < 100000
              AND noise = 0.0
            GROUP BY assoc_threshold, sim_threshold
            ORDER BY mean_acc DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()

        return [(row[0], row[1]) for row in rows]
    finally:
        conn.close()


def _print_phase2_summary(db_path: str, benchmarks: list[str]) -> None:
    """Print Phase 2 summary: best consolidation config per benchmark."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        logger.info("\n--- Phase 2 Summary ---")
        logger.info("\nBest consolidation config per benchmark:")

        for bm in benchmarks:
            row = conn.execute(
                """
                SELECT assoc_threshold, sim_threshold, min_share, min_count,
                       accuracy, n_nodes
                FROM sweep_results
                WHERE benchmark = ? AND config_id >= 100000 AND config_id < 200000
                  AND epoch = (
                      SELECT MAX(epoch) FROM sweep_results
                      WHERE benchmark = ? AND config_id >= 100000 AND config_id < 200000
                  )
                ORDER BY accuracy DESC
                LIMIT 1
                """,
                (bm, bm),
            ).fetchone()

            if row:
                logger.info(
                    f"  {bm}: {row['accuracy']:.4f} "
                    f"(assoc={row['assoc_threshold']:.4f}, sim={row['sim_threshold']:.2f}, "
                    f"min_share={row['min_share']:.2f}, min_count={row['min_count']}, "
                    f"nodes={row['n_nodes']})"
                )

    finally:
        conn.close()


# ── Phase 3: Epoch sweep ──────────────────────────────────────────


def run_phase3(
    runner: SweepRunner,
    features_dir: str,
    benchmarks: list[str],
) -> list[dict]:
    """Phase 3: Epoch sweep on top-10 configs from Phases 1+2.

    Queries top-10 overall configs from all previous phases,
    then sweeps epochs [1, 3, 5] per D-06.

    Args:
        runner: SweepRunner instance.
        features_dir: Directory containing cached feature .pt files.
        benchmarks: List of benchmark names.

    Returns:
        List of result summaries.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: Epoch Sweep")
    logger.info("=" * 60)

    # Query top-10 overall configs from Phases 1+2
    top_configs = _get_top_configs(runner.db_path, n=10)

    if not top_configs:
        logger.warning("No prior results found -- using defaults")
        top_configs = [(0.3, 0.1, 0.05, 3)]

    logger.info(f"  Top {len(top_configs)} configs from Phases 1+2:")
    for assoc, sim, ms, mc in top_configs:
        logger.info(f"    assoc={assoc:.4f}, sim={sim:.2f}, min_share={ms:.2f}, min_count={mc}")

    # Generate epoch sweep configs
    configs: list[SweepConfig] = []
    config_id = 200000

    for assoc, sim, min_share, min_count in top_configs:
        for benchmark in benchmarks:
            for epochs in EPOCH_VALUES:
                configs.append(
                    SweepConfig(
                        config_id=config_id,
                        benchmark=benchmark,
                        policy_type="global",
                        assoc_threshold=float(assoc),
                        sim_threshold=float(sim),
                        min_share=float(min_share),
                        min_count=int(min_count),
                        noise=0.0,
                        epochs=epochs,
                        seed=SEED,
                    )
                )
                config_id += 1

    logger.info(f"  Epoch values: {EPOCH_VALUES}")
    logger.info(f"  Total configs: {len(configs)} ({len(top_configs)} configs x {len(EPOCH_VALUES)} epochs x {len(benchmarks)} benchmarks)")

    t0 = time.time()
    results = runner.run(configs, features_dir)
    elapsed = time.time() - t0

    ok_count = sum(1 for r in results if r["status"] == "ok")
    logger.info(f"\nPhase 3 complete: {ok_count}/{len(results)} succeeded in {elapsed:.1f}s")

    _print_phase3_summary(runner.db_path, benchmarks)

    return results


def _get_top_configs(
    db_path: str, n: int = 10
) -> list[tuple[float, float, float, int]]:
    """Query top-N overall configs by mean accuracy across benchmarks.

    Considers all phases (no config_id filter).

    Args:
        db_path: Path to SQLite database.
        n: Number of top configs to return.

    Returns:
        List of (assoc_threshold, sim_threshold, min_share, min_count) tuples.
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT assoc_threshold, sim_threshold, min_share, min_count,
                   AVG(accuracy) as mean_acc
            FROM sweep_results
            WHERE epoch = (SELECT MAX(epoch) FROM sweep_results)
            GROUP BY assoc_threshold, sim_threshold, min_share, min_count
            ORDER BY mean_acc DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()

        return [(row[0], row[1], row[2], row[3]) for row in rows]
    finally:
        conn.close()


def _print_phase3_summary(db_path: str, benchmarks: list[str]) -> None:
    """Print Phase 3 summary: best epoch config per benchmark and overall Pareto."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        logger.info("\n--- Phase 3 Summary ---")
        logger.info("\nBest config per benchmark across all phases:")

        for bm in benchmarks:
            row = conn.execute(
                """
                SELECT assoc_threshold, sim_threshold, min_share, min_count,
                       noise, accuracy, n_nodes, epoch
                FROM sweep_results
                WHERE benchmark = ?
                ORDER BY accuracy DESC
                LIMIT 1
                """,
                (bm,),
            ).fetchone()

            if row:
                logger.info(
                    f"  {bm}: {row['accuracy']:.4f} at epoch {row['epoch']} "
                    f"(assoc={row['assoc_threshold']:.4f}, sim={row['sim_threshold']:.2f}, "
                    f"min_share={row['min_share']:.2f}, min_count={row['min_count']}, "
                    f"noise={row['noise']:.2f}, nodes={row['n_nodes']})"
                )

        # Pareto frontier points (accuracy vs node count)
        logger.info("\nPareto frontier (accuracy vs node count):")
        rows = conn.execute(
            """
            SELECT DISTINCT assoc_threshold, sim_threshold,
                   accuracy, n_nodes, benchmark
            FROM sweep_results
            WHERE epoch = (SELECT MAX(epoch) FROM sweep_results)
            ORDER BY accuracy DESC, n_nodes ASC
            LIMIT 20
            """,
        ).fetchall()

        # Compute Pareto front
        pareto_points = []
        min_nodes = float("inf")
        for row in rows:
            if row["n_nodes"] < min_nodes:
                pareto_points.append(row)
                min_nodes = row["n_nodes"]

        for pt in pareto_points[:10]:
            logger.info(
                f"  acc={pt['accuracy']:.4f}, nodes={pt['n_nodes']}, "
                f"assoc={pt['assoc_threshold']:.4f}, sim={pt['sim_threshold']:.2f}, "
                f"bm={pt['benchmark']}"
            )

    finally:
        conn.close()


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    """Command-line entry point for the global threshold sensitivity sweep."""
    parser = argparse.ArgumentParser(
        description="Global threshold sensitivity sweep across all benchmarks (D-20, D-22)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Phase 1 only (core 3D sweep):
  python scripts/sweep/run_global_sweep.py --phase 1 --features-dir results/T2/features

  # Run all phases sequentially:
  python scripts/sweep/run_global_sweep.py --phase all --features-dir results/T2/features

  # Custom workers and database:
  python scripts/sweep/run_global_sweep.py --phase 1 --workers 8 --db custom.db --features-dir features/
        """,
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["1", "2", "3", "all"],
        help="Sweep phase to run: 1 (core 3D), 2 (consolidation), 3 (epochs), or all (default: all)",
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Ensure output directory exists
    db_dir = Path(args.db).parent
    db_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("GLOBAL THRESHOLD SENSITIVITY SWEEP")
    logger.info("=" * 60)
    logger.info(f"  Phase: {args.phase}")
    logger.info(f"  Features: {args.features_dir}")
    logger.info(f"  Database: {args.db}")
    logger.info(f"  Workers: {args.workers}")
    logger.info(f"  Benchmarks: {args.benchmarks}")
    logger.info(f"  Seed: {SEED}")

    runner = SweepRunner(args.db, n_workers=args.workers)

    t_total = time.time()

    phases_to_run = (
        ["1", "2", "3"] if args.phase == "all" else [args.phase]
    )

    for phase in phases_to_run:
        if phase == "1":
            run_phase1(runner, args.features_dir, args.benchmarks)
        elif phase == "2":
            run_phase2(runner, args.features_dir, args.benchmarks)
        elif phase == "3":
            run_phase3(runner, args.features_dir, args.benchmarks)

    total_elapsed = time.time() - t_total
    logger.info(f"\n{'=' * 60}")
    logger.info(f"SWEEP COMPLETE in {total_elapsed:.1f}s")
    logger.info(f"{'=' * 60}")

    # Final summary: count total results in DB
    conn = sqlite3.connect(args.db)
    try:
        total_rows = conn.execute(
            "SELECT COUNT(*) FROM sweep_results"
        ).fetchone()[0]
        unique_configs = conn.execute(
            "SELECT COUNT(DISTINCT config_id) FROM sweep_results"
        ).fetchone()[0]
        unique_benchmarks = conn.execute(
            "SELECT COUNT(DISTINCT benchmark) FROM sweep_results"
        ).fetchone()[0]

        logger.info(f"  Total rows: {total_rows}")
        logger.info(f"  Unique configs: {unique_configs}")
        logger.info(f"  Unique benchmarks: {unique_benchmarks}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
