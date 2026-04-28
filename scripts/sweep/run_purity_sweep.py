"""Algebraic purity strategy sweep (Strategy 4) with noise interaction analysis.

Runs hyperparameter sweep for AlgebraicPurityPolicy, testing associator norm
variance and routing key similarity variance as independent signals per D-01.
Also runs noise interaction sweep across all strategies (strategies 1-4 + global)
per D-05.

Per D-01 strategy 4: Uses two independent purity signals:
  (a) assoc_weight controls associator norm variance signal
  (b) sim_weight controls routing key similarity variance signal

Per D-29: Sweep sensitivity, assoc_weight, sim_weight for fair comparison.
Per D-05: Noise interaction is the 4th sweep dimension.
Per D-31: Fixed seed=42.
Per D-07: All 5 benchmarks.

Usage:
    # Run purity strategy sweep:
    python scripts/sweep/run_purity_sweep.py --features-dir results/T2/features --db results/T2/sweep.db --workers 24

    # Run noise interaction analysis (after purity sweep):
    python scripts/sweep/run_purity_sweep.py --noise-interaction --features-dir results/T2/features --db results/T2/sweep.db
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

# Purity-specific hyperparameters per D-01 strategy 4, D-29
ASSOC_WEIGHT_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]
SIM_WEIGHT_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]
SENSITIVITY_VALUES = [0.1, 0.3, 0.5, 1.0, 2.0]

# Noise levels for interaction sweep per D-05
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1]

# Config ID offsets for purity strategy (avoid collision with strategies 1-3)
PURITY_CONFIG_ID_BASE = 900000
NOISE_INTERACTION_CONFIG_ID_BASE = 950000


# ── Query best configs from DB ────────────────────────────────────


def get_top_global_assoc_thresholds(
    db_path: str, n: int = 3
) -> list[float]:
    """Query top-N assoc_threshold values from global sweep by mean accuracy.

    Args:
        db_path: Path to SQLite database.
        n: Number of top values to return.

    Returns:
        List of assoc_threshold values, or defaults if DB has no global results.
    """
    defaults = [0.05, 0.1, 0.3]
    conn = sqlite3.connect(db_path)
    try:
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
    db_path: str, n: int = 2
) -> list[float]:
    """Query top-N sim_threshold values from global sweep by mean accuracy.

    Per D-03: co-adapt sim_threshold with purity hyperparameters.

    Args:
        db_path: Path to SQLite database.
        n: Number of top values to return.

    Returns:
        List of sim_threshold values, or defaults if DB has no global results.
    """
    defaults = [0.0, 0.1]
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


def get_best_strategy_per_benchmark(
    db_path: str, strategy: str
) -> dict[str, dict[str, Any]]:
    """Query best accuracy per benchmark for a specific strategy.

    Args:
        db_path: Path to SQLite database.
        strategy: Policy type to query.

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

            if row:
                results[bm] = dict(row)
            else:
                results[bm] = {"accuracy": 0.0}

        return results
    except sqlite3.OperationalError:
        return {bm: {"accuracy": 0.0} for bm in ALL_BENCHMARKS}
    finally:
        conn.close()


def get_top_strategy_configs(
    db_path: str, strategy: str, n: int = 5
) -> list[dict[str, Any]]:
    """Query top-N configs for a strategy by mean accuracy across benchmarks.

    Args:
        db_path: Path to SQLite database.
        strategy: Policy type to query.
        n: Number of top configs.

    Returns:
        List of config dicts.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT assoc_threshold, sim_threshold, policy_params,
                   AVG(accuracy) as mean_acc, noise, config_id,
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
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


# ── Config generation for purity sweep ────────────────────────────


def generate_purity_sweep_configs(
    benchmarks: list[str],
    db_path: str,
    seed: int = SEED,
) -> list[SweepConfig]:
    """Generate sweep configs for AlgebraicPurityPolicy (Strategy 4).

    Per D-01 strategy 4: test assoc norm variance and sim variance as
    independent signals. Structured in 3 phases:
      Phase A: assoc_weight > 0, sim_weight = 0 (assoc variance only)
      Phase B: assoc_weight = 0, sim_weight > 0 (sim variance only)
      Phase C: both > 0 (combined)

    Sweeps: assoc_weight x sim_weight x sensitivity x base_assoc(top-3) x sim(top-2)
    Minus invalid (both weights=0).

    Per plan:
    - Phase A: 4 assoc * 5 sens * 3 base * 2 sim = 120 per benchmark
    - Phase B: 4 sim * 5 sens * 3 base * 2 sim = 120 per benchmark
    - Phase C: 4 assoc * 4 sim * 5 sens * 3 base * 2 sim = 480 per benchmark
      Reduced to top combinations for tractability: 3 assoc * 3 sim * 5 sens * 3 base * 2 sim = 270
    Total: ~510 per benchmark (reduced from 720)

    Args:
        benchmarks: List of benchmark names.
        db_path: Path to SQLite database for querying global best configs.
        seed: Random seed.

    Returns:
        List of SweepConfig instances.
    """
    top_assoc = get_top_global_assoc_thresholds(db_path, n=3)
    top_sim = get_top_global_sim_thresholds(db_path, n=2)

    configs: list[SweepConfig] = []
    config_id = PURITY_CONFIG_ID_BASE

    for benchmark in benchmarks:
        # ── Phase A: assoc variance only (sim_weight = 0) ──
        for assoc_weight in [0.3, 0.5, 0.7, 1.0]:  # Skip 0.0
            for sensitivity in SENSITIVITY_VALUES:
                for base_assoc in top_assoc:
                    for sim_th in top_sim:
                        policy_params = {
                            "base_assoc": float(base_assoc),
                            "assoc_weight": assoc_weight,
                            "sim_weight": 0.0,
                            "sensitivity": sensitivity,
                            "sim_threshold": float(sim_th),
                            "min_share": 0.05,
                            "min_count": 3,
                        }
                        configs.append(
                            SweepConfig(
                                config_id=config_id,
                                benchmark=benchmark,
                                policy_type="purity",
                                assoc_threshold=float(base_assoc),
                                sim_threshold=float(sim_th),
                                min_share=0.05,
                                min_count=3,
                                noise=0.0,
                                epochs=DEFAULT_EPOCHS,
                                seed=seed,
                                policy_params=json.dumps(policy_params),
                            )
                        )
                        config_id += 1

        # ── Phase B: sim variance only (assoc_weight = 0) ──
        for sim_weight in [0.3, 0.5, 0.7, 1.0]:  # Skip 0.0
            for sensitivity in SENSITIVITY_VALUES:
                for base_assoc in top_assoc:
                    for sim_th in top_sim:
                        policy_params = {
                            "base_assoc": float(base_assoc),
                            "assoc_weight": 0.0,
                            "sim_weight": sim_weight,
                            "sensitivity": sensitivity,
                            "sim_threshold": float(sim_th),
                            "min_share": 0.05,
                            "min_count": 3,
                        }
                        configs.append(
                            SweepConfig(
                                config_id=config_id,
                                benchmark=benchmark,
                                policy_type="purity",
                                assoc_threshold=float(base_assoc),
                                sim_threshold=float(sim_th),
                                min_share=0.05,
                                min_count=3,
                                noise=0.0,
                                epochs=DEFAULT_EPOCHS,
                                seed=seed,
                                policy_params=json.dumps(policy_params),
                            )
                        )
                        config_id += 1

        # ── Phase C: combined (both > 0) — reduced grid ──
        combined_assoc_weights = [0.3, 0.5, 0.7]
        combined_sim_weights = [0.3, 0.5, 0.7]
        for assoc_weight in combined_assoc_weights:
            for sim_weight in combined_sim_weights:
                for sensitivity in SENSITIVITY_VALUES:
                    for base_assoc in top_assoc:
                        for sim_th in top_sim:
                            policy_params = {
                                "base_assoc": float(base_assoc),
                                "assoc_weight": assoc_weight,
                                "sim_weight": sim_weight,
                                "sensitivity": sensitivity,
                                "sim_threshold": float(sim_th),
                                "min_share": 0.05,
                                "min_count": 3,
                            }
                            configs.append(
                                SweepConfig(
                                    config_id=config_id,
                                    benchmark=benchmark,
                                    policy_type="purity",
                                    assoc_threshold=float(base_assoc),
                                    sim_threshold=float(sim_th),
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


# ── Noise interaction sweep ──────────────────────────────────────


def generate_noise_interaction_configs(
    db_path: str,
    benchmarks: list[str],
    seed: int = SEED,
) -> list[SweepConfig]:
    """Generate noise interaction configs for top purity and strategies 1-3.

    Per D-05: run noise sweep on top configs from all strategies to characterize
    whether noise helps or hurts each adaptive strategy.

    - Top-10 purity configs with noise=[0.0, 0.01, 0.05, 0.1]
    - Top-5 from each of strategies 1-3 (ema, mean_std, depth) with noise
    - Top-5 global configs with noise

    Args:
        db_path: Path to SQLite database.
        benchmarks: List of benchmark names.
        seed: Random seed.

    Returns:
        List of SweepConfig instances.
    """
    configs: list[SweepConfig] = []
    config_id = NOISE_INTERACTION_CONFIG_ID_BASE

    # ── Top-10 purity configs ──
    top_purity = get_top_strategy_configs(db_path, "purity", n=10)
    logger.info(f"  Noise interaction: {len(top_purity)} purity configs")

    for top_cfg in top_purity:
        base_policy_params = json.loads(top_cfg["policy_params"])
        for benchmark in benchmarks:
            for noise in NOISE_LEVELS:
                policy_params = dict(base_policy_params)
                configs.append(
                    SweepConfig(
                        config_id=config_id,
                        benchmark=benchmark,
                        policy_type="purity",
                        assoc_threshold=top_cfg["assoc_threshold"],
                        sim_threshold=top_cfg["sim_threshold"],
                        min_share=policy_params.get("min_share", 0.05),
                        min_count=policy_params.get("min_count", 3),
                        noise=noise,
                        epochs=DEFAULT_EPOCHS,
                        seed=seed,
                        policy_params=json.dumps(policy_params),
                    )
                )
                config_id += 1

    # ── Top-5 from strategies 1-3 ──
    for strategy in ["ema", "mean_std", "depth"]:
        top_strat = get_top_strategy_configs(db_path, strategy, n=5)
        logger.info(f"  Noise interaction: {len(top_strat)} {strategy} configs")

        for top_cfg in top_strat:
            base_policy_params = json.loads(top_cfg["policy_params"])
            for benchmark in benchmarks:
                for noise in NOISE_LEVELS:
                    policy_params = dict(base_policy_params)
                    configs.append(
                        SweepConfig(
                            config_id=config_id,
                            benchmark=benchmark,
                            policy_type=strategy,
                            assoc_threshold=top_cfg["assoc_threshold"],
                            sim_threshold=top_cfg["sim_threshold"],
                            min_share=policy_params.get("min_share", 0.05),
                            min_count=policy_params.get("min_count", 3),
                            noise=noise,
                            epochs=DEFAULT_EPOCHS,
                            seed=seed,
                            policy_params=json.dumps(policy_params),
                        )
                    )
                    config_id += 1

    # ── Top-5 global configs ──
    top_global = _get_top_global_configs(db_path, n=5)
    logger.info(f"  Noise interaction: {len(top_global)} global configs")

    for top_cfg in top_global:
        for benchmark in benchmarks:
            for noise in NOISE_LEVELS:
                configs.append(
                    SweepConfig(
                        config_id=config_id,
                        benchmark=benchmark,
                        policy_type="global",
                        assoc_threshold=top_cfg["assoc_threshold"],
                        sim_threshold=top_cfg["sim_threshold"],
                        min_share=top_cfg.get("min_share", 0.05),
                        min_count=top_cfg.get("min_count", 3),
                        noise=noise,
                        epochs=DEFAULT_EPOCHS,
                        seed=seed,
                        policy_params=json.dumps({
                            "assoc_threshold": top_cfg["assoc_threshold"],
                            "sim_threshold": top_cfg["sim_threshold"],
                            "min_share": top_cfg.get("min_share", 0.05),
                            "min_count": top_cfg.get("min_count", 3),
                        }),
                    )
                )
                config_id += 1

    return configs


def _get_top_global_configs(
    db_path: str, n: int = 5
) -> list[dict[str, Any]]:
    """Query top-N global configs by mean accuracy across benchmarks.

    Args:
        db_path: Path to SQLite database.
        n: Number of top configs.

    Returns:
        List of config dicts.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT assoc_threshold, sim_threshold, min_share, min_count,
                   AVG(accuracy) as mean_acc,
                   COUNT(DISTINCT benchmark) as n_benchmarks
            FROM sweep_results
            WHERE policy_type = 'global'
              AND epoch = (
                  SELECT MAX(epoch) FROM sweep_results WHERE policy_type = 'global'
              )
            GROUP BY assoc_threshold, sim_threshold, min_share, min_count
            HAVING n_benchmarks >= 1
            ORDER BY mean_acc DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()

        return [dict(row) for row in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


# ── Independent signal analysis ──────────────────────────────────


def print_independent_signal_analysis(db_path: str, benchmarks: list[str]) -> None:
    """Print analysis comparing assoc-only, sim-only, and combined purity signals.

    Per D-01: test associator norm variance and routing key similarity variance
    as independent signals.

    Args:
        db_path: Path to SQLite database.
        benchmarks: List of benchmark names.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        logger.info("")
        logger.info("=" * 80)
        logger.info("INDEPENDENT SIGNAL ANALYSIS")
        logger.info("=" * 80)
        logger.info(
            "Phase A: assoc variance only (sim_weight=0)")
        logger.info(
            "Phase B: sim variance only (assoc_weight=0)")
        logger.info(
            "Phase C: combined (both > 0)")
        logger.info("")

        for bm in benchmarks:
            logger.info(f"--- {bm} ---")

            # Phase A: assoc_weight > 0, sim_weight = 0
            phase_a = conn.execute(
                """
                SELECT policy_params, accuracy
                FROM sweep_results
                WHERE benchmark = ? AND policy_type = 'purity'
                  AND epoch = (
                      SELECT MAX(epoch) FROM sweep_results
                      WHERE benchmark = ? AND policy_type = 'purity'
                  )
                ORDER BY accuracy DESC
                """,
                (bm, bm),
            ).fetchall()

            best_a = {"accuracy": 0.0, "params": ""}
            best_b = {"accuracy": 0.0, "params": ""}
            best_c = {"accuracy": 0.0, "params": ""}

            for row in phase_a:
                pp = json.loads(row["policy_params"])
                acc = row["accuracy"]
                aw = pp.get("assoc_weight", 0)
                sw = pp.get("sim_weight", 0)

                if aw > 0 and sw == 0:
                    if acc > best_a["accuracy"]:
                        best_a = {
                            "accuracy": acc,
                            "params": f"aw={aw}, sens={pp.get('sensitivity', '?')}",
                        }
                elif aw == 0 and sw > 0:
                    if acc > best_b["accuracy"]:
                        best_b = {
                            "accuracy": acc,
                            "params": f"sw={sw}, sens={pp.get('sensitivity', '?')}",
                        }
                elif aw > 0 and sw > 0:
                    if acc > best_c["accuracy"]:
                        best_c = {
                            "accuracy": acc,
                            "params": f"aw={aw}, sw={sw}, sens={pp.get('sensitivity', '?')}",
                        }

            logger.info(
                f"  Phase A (assoc only): {best_a['accuracy']:.4f} -- {best_a['params']}"
            )
            logger.info(
                f"  Phase B (sim only):   {best_b['accuracy']:.4f} -- {best_b['params']}"
            )
            logger.info(
                f"  Phase C (combined):   {best_c['accuracy']:.4f} -- {best_c['params']}"
            )

            # Determine winner
            phases = [("A (assoc)", best_a), ("B (sim)", best_b), ("C (combined)", best_c)]
            winner = max(phases, key=lambda p: p[1]["accuracy"])
            logger.info(f"  Winner: Phase {winner[0]} ({winner[1]['accuracy']:.4f})")
            logger.info("")

    finally:
        conn.close()


# ── Noise interaction analysis ───────────────────────────────────


def print_noise_interaction_analysis(db_path: str, benchmarks: list[str]) -> None:
    """Print noise interaction analysis across all strategies.

    Per D-05: characterize whether noise helps or hurts each adaptive strategy.

    Args:
        db_path: Path to SQLite database.
        benchmarks: List of benchmark names.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        logger.info("")
        logger.info("=" * 80)
        logger.info("NOISE INTERACTION ANALYSIS")
        logger.info("=" * 80)

        all_strategies = ["global", "ema", "mean_std", "depth", "purity"]

        # Per-strategy, per-noise best accuracy
        logger.info("")
        logger.info(
            f"{'Strategy':<12} {'Noise=0.0':>10} {'Noise=0.01':>11} "
            f"{'Noise=0.05':>11} {'Noise=0.1':>10} {'Best Noise':>11} {'Synergy':>8}"
        )
        logger.info("-" * 80)

        for strategy in all_strategies:
            noise_accs: dict[float, list[float]] = {n: [] for n in NOISE_LEVELS}

            for bm in benchmarks:
                for noise in NOISE_LEVELS:
                    row = conn.execute(
                        """
                        SELECT accuracy
                        FROM sweep_results
                        WHERE benchmark = ? AND policy_type = ? AND noise = ?
                          AND epoch = (
                              SELECT MAX(epoch) FROM sweep_results
                              WHERE benchmark = ? AND policy_type = ?
                          )
                        ORDER BY accuracy DESC
                        LIMIT 1
                        """,
                        (bm, strategy, noise, bm, strategy),
                    ).fetchone()

                    if row:
                        noise_accs[noise].append(row["accuracy"])

            # Compute mean accuracy per noise level
            mean_accs = {}
            for noise, accs in noise_accs.items():
                if accs:
                    mean_accs[noise] = sum(accs) / len(accs)
                else:
                    mean_accs[noise] = 0.0

            # Find best noise
            best_noise = max(mean_accs, key=lambda n: mean_accs[n]) if mean_accs else 0.0
            baseline_acc = mean_accs.get(0.0, 0.0)
            best_acc = mean_accs.get(best_noise, 0.0)

            # Synergy: does noise improve over no-noise?
            synergy = "YES" if best_noise > 0.0 and best_acc > baseline_acc else "NO"

            noise_strs = []
            for n in NOISE_LEVELS:
                acc = mean_accs.get(n, 0.0)
                if acc > 0:
                    noise_strs.append(f"{acc:>10.4f}")
                else:
                    noise_strs.append(f"{'N/A':>10}")

            logger.info(
                f"{strategy:<12} {' '.join(noise_strs)} "
                f"{best_noise:>11.3f} {synergy:>8}"
            )

        logger.info("")
        logger.info("Synergy = YES means noise > 0 produces higher accuracy than noise = 0")

    finally:
        conn.close()


# ── Comparison table ─────────────────────────────────────────────


def print_comparison_table(db_path: str, benchmarks: list[str]) -> None:
    """Print comparison: purity best vs global/ema/mean_std/depth best per benchmark.

    Args:
        db_path: Path to SQLite database.
        benchmarks: List of benchmark names.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        logger.info("")
        logger.info("=" * 90)
        logger.info("PURITY vs ALL STRATEGIES COMPARISON")
        logger.info("=" * 90)

        strategies = ["global", "ema", "mean_std", "depth", "purity"]

        # Header
        header = f"{'Benchmark':<15}"
        for s in strategies:
            header += f" {s:>12}"
        logger.info(header)
        logger.info("-" * 90)

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
        logger.info("-" * 90)
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
            logger.info(
                f"\nBest overall strategy: {best_strategy} "
                f"(mean accuracy: {best_mean:.4f})"
            )

        # Purity-specific comparison: delta vs each strategy
        logger.info("")
        logger.info("PURITY DELTA vs OTHER STRATEGIES:")
        purity_best = get_best_strategy_per_benchmark(db_path, "purity")

        for other in ["global", "ema", "mean_std", "depth"]:
            other_best = get_best_strategy_per_benchmark(db_path, other)
            wins = 0
            total_delta = 0.0
            n_compared = 0

            for bm in benchmarks:
                p_acc = purity_best.get(bm, {}).get("accuracy", 0.0)
                o_acc = other_best.get(bm, {}).get("accuracy", 0.0)
                if p_acc > 0 and o_acc > 0:
                    delta = p_acc - o_acc
                    total_delta += delta
                    n_compared += 1
                    if delta > 0:
                        wins += 1

            if n_compared > 0:
                logger.info(
                    f"  vs {other:<10}: wins={wins}/{n_compared}, "
                    f"mean delta={total_delta / n_compared:+.4f}"
                )

    finally:
        conn.close()


# ── Purity top configs ───────────────────────────────────────────


def print_top_purity_configs(db_path: str) -> None:
    """Print top-10 purity configs by mean accuracy.

    Args:
        db_path: Path to SQLite database.
    """
    top_configs = get_top_strategy_configs(db_path, "purity", n=10)

    if not top_configs:
        logger.info("No purity results found.")
        return

    logger.info("")
    logger.info("=" * 80)
    logger.info("TOP-10 PURITY CONFIGS (by mean accuracy across benchmarks)")
    logger.info("=" * 80)

    for i, cfg in enumerate(top_configs, 1):
        pp = json.loads(cfg["policy_params"])
        logger.info(
            f"  {i:2d}. mean_acc={cfg['mean_acc']:.4f} "
            f"assoc_w={pp.get('assoc_weight', '?')}, "
            f"sim_w={pp.get('sim_weight', '?')}, "
            f"sens={pp.get('sensitivity', '?')}, "
            f"base={pp.get('base_assoc', '?')}, "
            f"sim_th={pp.get('sim_threshold', '?')}"
        )


# ── Main sweep runner ────────────────────────────────────────────


def run_purity_sweep(
    runner: SweepRunner,
    features_dir: str,
    benchmarks: list[str],
    db_path: str,
) -> list[dict]:
    """Run the full purity strategy sweep.

    Args:
        runner: SweepRunner instance.
        features_dir: Directory containing cached feature .pt files.
        benchmarks: List of benchmark names.
        db_path: Path to SQLite database.

    Returns:
        List of result summaries.
    """
    logger.info("=" * 70)
    logger.info("STRATEGY 4 SWEEP: ALGEBRAIC PURITY (AlgebraicPurityPolicy)")
    logger.info("=" * 70)

    # 1. Generate configs
    configs = generate_purity_sweep_configs(benchmarks, db_path, seed=SEED)

    n_per_bm = len(configs) // max(len(benchmarks), 1)
    logger.info("  Policy: AlgebraicPurityPolicy")
    logger.info(f"  Configs per benchmark: {n_per_bm}")
    logger.info(f"  Total configs: {len(configs)}")
    logger.info(f"  Benchmarks: {', '.join(benchmarks)}")
    logger.info(f"  Epochs: {DEFAULT_EPOCHS}")
    logger.info("  Signal phases: A (assoc only), B (sim only), C (combined)")

    # 2. Run sweep
    t0 = time.time()
    results = runner.run(configs, features_dir)
    elapsed = time.time() - t0

    ok_count = sum(1 for r in results if r["status"] == "ok")
    logger.info(
        f"\nPurity sweep: {ok_count}/{len(results)} succeeded in {elapsed:.1f}s"
    )

    # 3. Print analysis
    print_top_purity_configs(db_path)
    print_independent_signal_analysis(db_path, benchmarks)
    print_comparison_table(db_path, benchmarks)

    return results


def run_noise_interaction(
    runner: SweepRunner,
    features_dir: str,
    benchmarks: list[str],
    db_path: str,
) -> list[dict]:
    """Run noise interaction sweep across all strategies.

    Per D-05: characterize noise interaction for all strategies + global.

    Args:
        runner: SweepRunner instance.
        features_dir: Directory containing cached feature .pt files.
        benchmarks: List of benchmark names.
        db_path: Path to SQLite database.

    Returns:
        List of result summaries.
    """
    logger.info("=" * 70)
    logger.info("NOISE INTERACTION SWEEP (All Strategies + Global)")
    logger.info("=" * 70)

    # Generate noise interaction configs
    configs = generate_noise_interaction_configs(db_path, benchmarks, seed=SEED)

    logger.info(f"  Total noise interaction configs: {len(configs)}")
    logger.info(f"  Noise levels: {NOISE_LEVELS}")
    logger.info("  Strategies included: global, ema, mean_std, depth, purity")
    logger.info(f"  Benchmarks: {', '.join(benchmarks)}")

    if not configs:
        logger.warning("No configs generated -- run purity sweep first")
        return []

    # Run sweep
    t0 = time.time()
    results = runner.run(configs, features_dir)
    elapsed = time.time() - t0

    ok_count = sum(1 for r in results if r["status"] == "ok")
    logger.info(
        f"\nNoise interaction sweep: {ok_count}/{len(results)} succeeded in {elapsed:.1f}s"
    )

    # Print analysis
    print_noise_interaction_analysis(db_path, benchmarks)

    return results


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    """Command-line entry point for purity strategy sweep."""
    parser = argparse.ArgumentParser(
        description=(
            "Algebraic purity strategy sweep (Strategy 4) and noise "
            "interaction analysis per D-01, D-05, D-29"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run purity sweep:
  python scripts/sweep/run_purity_sweep.py --features-dir results/T2/features

  # Run noise interaction analysis (after purity and other sweeps):
  python scripts/sweep/run_purity_sweep.py --noise-interaction --features-dir results/T2/features

  # Run both:
  python scripts/sweep/run_purity_sweep.py --noise-interaction --features-dir results/T2/features

  # Custom workers and database:
  python scripts/sweep/run_purity_sweep.py --workers 8 --db custom.db --features-dir features/
        """,
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
        "--noise-interaction",
        action="store_true",
        default=False,
        help=(
            "Run noise interaction sweep on top configs from all strategies. "
            "Runs purity sweep first if no purity results exist."
        ),
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
    logger.info("ALGEBRAIC PURITY SWEEP (Strategy 4) + NOISE INTERACTION")
    logger.info("=" * 70)
    logger.info(f"  Features: {args.features_dir}")
    logger.info(f"  Database: {args.db}")
    logger.info(f"  Workers: {args.workers}")
    logger.info(f"  Benchmarks: {args.benchmarks}")
    logger.info(f"  Noise interaction: {args.noise_interaction}")
    logger.info(f"  Seed: {SEED}")

    runner = SweepRunner(args.db, n_workers=args.workers)

    t_total = time.time()
    all_results: list[dict] = []

    # Always run purity sweep first (or check if results exist)
    purity_count = _count_purity_results(args.db)
    if purity_count == 0:
        logger.info("\nNo purity results found -- running purity sweep first")
        results = run_purity_sweep(runner, args.features_dir, args.benchmarks, args.db)
        all_results.extend(results)
    else:
        logger.info(f"\n{purity_count} existing purity results found in DB")
        # Still print analysis of existing results
        print_top_purity_configs(args.db)
        print_independent_signal_analysis(args.db, args.benchmarks)
        print_comparison_table(args.db, args.benchmarks)

    # Run noise interaction if requested
    if args.noise_interaction:
        results = run_noise_interaction(
            runner, args.features_dir, args.benchmarks, args.db
        )
        all_results.extend(results)

    total_elapsed = time.time() - t_total
    logger.info(f"\n{'=' * 70}")
    logger.info(f"PURITY SWEEP + NOISE INTERACTION COMPLETE in {total_elapsed:.1f}s")
    logger.info(f"{'=' * 70}")

    ok_total = sum(1 for r in all_results if r["status"] == "ok")
    logger.info(f"  Total configs run: {len(all_results)}")
    logger.info(f"  Succeeded: {ok_total}")
    logger.info(f"  Failed: {len(all_results) - ok_total}")


def _count_purity_results(db_path: str) -> int:
    """Count existing purity results in the database.

    Args:
        db_path: Path to SQLite database.

    Returns:
        Number of purity result rows, or 0 if DB not initialized.
    """
    try:
        conn = sqlite3.connect(db_path)
        count = conn.execute(
            "SELECT COUNT(*) FROM sweep_results WHERE policy_type = 'purity'"
        ).fetchone()[0]
        conn.close()
        return count
    except sqlite3.OperationalError:
        return 0


if __name__ == "__main__":
    main()
