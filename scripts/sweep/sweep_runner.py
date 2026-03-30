"""Reusable parallel sweep framework with SQLite storage.

Provides a generic SweepRunner that executes N configurations across M workers,
writing epoch-by-epoch results to a SQLite database in WAL mode. Designed for
reuse by T2 (adaptive thresholds), T4, and T6 phases.

Key design:
- Each worker creates its own SQLite connection (no shared connections)
- WAL journal mode for concurrent write tolerance
- ProcessPoolExecutor for CPU-bound trie workloads
- tqdm progress bar for live progress
- Config generation helpers for global and adaptive sweep grids

Usage:
    from scripts.sweep.sweep_runner import SweepRunner, generate_global_sweep_configs
    configs = generate_global_sweep_configs(["mnist", "fashion_mnist"])
    runner = SweepRunner("results.db", n_workers=24)
    runner.run(configs, features_dir="features/")
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from octonion.trie import OctonionTrie

logger = logging.getLogger(__name__)

# ── Schema ─────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sweep_results (
    config_id INTEGER NOT NULL,
    benchmark TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    policy_type TEXT NOT NULL,
    assoc_threshold REAL,
    sim_threshold REAL,
    min_share REAL,
    min_count INTEGER,
    noise REAL,
    accuracy REAL,
    n_nodes INTEGER,
    n_leaves INTEGER,
    max_depth INTEGER,
    rumination_rejections INTEGER,
    consolidation_merges INTEGER,
    branching_factor_mean REAL,
    branching_factor_std REAL,
    train_time REAL,
    test_time REAL,
    policy_params TEXT DEFAULT '{}',
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (config_id, benchmark, epoch, seed)
);
CREATE INDEX IF NOT EXISTS idx_benchmark ON sweep_results(benchmark);
CREATE INDEX IF NOT EXISTS idx_policy ON sweep_results(policy_type);
CREATE INDEX IF NOT EXISTS idx_accuracy ON sweep_results(accuracy DESC);
"""


# ── Config ─────────────────────────────────────────────────────────


@dataclass
class SweepConfig:
    """Configuration for a single sweep experiment."""

    config_id: int
    benchmark: str
    policy_type: str  # "global", "ema", "mean_std", "depth", "purity", "meta_trie", "hybrid"
    assoc_threshold: float  # base threshold
    sim_threshold: float
    min_share: float
    min_count: int
    noise: float
    epochs: int
    seed: int
    policy_params: str = "{}"  # JSON string for policy-specific hyperparams


# ── Branching factor computation ───────────────────────────────────


def _compute_branching_factor(trie: OctonionTrie) -> tuple[float, float]:
    """Compute mean and std of branching factor for non-leaf nodes.

    Walks the trie and computes len(node.children) for each non-leaf node.

    Returns:
        (mean_branching_factor, std_branching_factor)
    """
    branching_factors: list[int] = []

    def _walk(node: Any) -> None:
        if node.children:
            branching_factors.append(len(node.children))
            for child in node.children.values():
                _walk(child)

    _walk(trie.root)

    if not branching_factors:
        return 0.0, 0.0

    arr = np.array(branching_factors, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


# ── Policy construction ────────────────────────────────────────────


def _make_policy(policy_type: str, policy_params: dict[str, Any]) -> Any:
    """Construct a ThresholdPolicy from type name and parameters.

    Imports policy classes lazily to support both scenarios:
    - When ThresholdPolicy classes are available in octonion.trie (post T2-01)
    - When only GlobalPolicy/basic trie is available (pre T2-01 fallback)

    Returns:
        A ThresholdPolicy instance, or None for "global" (uses trie defaults).
    """
    if policy_type == "global":
        # For global policy, threshold values are set directly on the trie
        # constructor; no separate policy object needed for basic global sweep.
        try:
            from octonion.trie import GlobalPolicy

            return GlobalPolicy(
                assoc_threshold=policy_params.get("assoc_threshold", 0.3),
                sim_threshold=policy_params.get("sim_threshold", 0.1),
                min_share=policy_params.get("min_share", 0.05),
                min_count=policy_params.get("min_count", 3),
            )
        except ImportError:
            return None

    # Adaptive policy types (require T2-01 ThresholdPolicy classes)
    policy_map = {
        "ema": "PerNodeEMAPolicy",
        "mean_std": "PerNodeMeanStdPolicy",
        "depth": "DepthPolicy",
        "purity": "AlgebraicPurityPolicy",
        "meta_trie": "MetaTriePolicy",
        "hybrid": "HybridPolicy",
    }

    class_name = policy_map.get(policy_type)
    if class_name is None:
        raise ValueError(f"Unknown policy_type: {policy_type!r}")

    # Dynamic import from octonion.trie
    import importlib

    trie_module = importlib.import_module("octonion.trie")
    cls = getattr(trie_module, class_name)
    return cls(**policy_params)


# ── Worker ─────────────────────────────────────────────────────────


def _run_single_config(
    config: SweepConfig, features_dir: str, db_path: str
) -> dict[str, Any]:
    """Execute a single sweep configuration and write results to SQLite.

    This function runs in a child process. It creates its own SQLite
    connection (no shared connections across processes) with timeout=30s
    for write contention handling.

    Args:
        config: The sweep configuration to run.
        features_dir: Directory containing cached feature .pt files.
        db_path: Path to the SQLite database.

    Returns:
        Summary dict with config_id, final_accuracy, n_epochs, status.
    """
    try:
        # 1. Load cached features
        features_path = _find_features_file(features_dir, config.benchmark)
        data = torch.load(features_path, map_location="cpu", weights_only=True)

        train_x = data["train_x"]
        train_y = data["train_y"]
        test_x = data["test_x"]
        test_y = data["test_y"]

        # 2. Construct policy
        policy_params = json.loads(config.policy_params)
        # Inject base thresholds into policy params for convenience
        policy_params.setdefault("assoc_threshold", config.assoc_threshold)
        policy_params.setdefault("sim_threshold", config.sim_threshold)
        policy_params.setdefault("min_share", config.min_share)
        policy_params.setdefault("min_count", config.min_count)

        policy = _make_policy(config.policy_type, policy_params)

        # 3. Create trie
        trie_kwargs: dict[str, Any] = {
            "seed": config.seed,
        }

        if policy is not None:
            # Use policy-based construction (T2-01+)
            trie_kwargs["policy"] = policy
        else:
            # Fallback: direct threshold parameters (pre T2-01)
            trie_kwargs["associator_threshold"] = config.assoc_threshold
            trie_kwargs["similarity_threshold"] = config.sim_threshold

        trie = OctonionTrie(**trie_kwargs)

        # 4. Train and evaluate epoch-by-epoch (per D-25)
        epoch_results: list[dict[str, Any]] = []

        for epoch in range(config.epochs):
            # Training pass
            t_train = time.time()
            for i in range(len(train_x)):
                label = (
                    train_y[i].item()
                    if isinstance(train_y[i], torch.Tensor)
                    else int(train_y[i])
                )
                x_i = train_x[i]
                if config.noise > 0:
                    x_i = x_i + config.noise * torch.randn_like(x_i)
                trie.insert(x_i, category=label)
            train_time = time.time() - t_train

            # Consolidate every 2 epochs
            if epoch % 2 == 1:
                trie.consolidate()

            # Evaluate
            t_test = time.time()
            correct = 0
            for i in range(len(test_x)):
                label = (
                    test_y[i].item()
                    if isinstance(test_y[i], torch.Tensor)
                    else int(test_y[i])
                )
                leaf = trie.query(test_x[i])
                if leaf.dominant_category == label:
                    correct += 1
            test_time = time.time() - t_test

            accuracy = correct / len(test_y) if len(test_y) > 0 else 0.0
            stats = trie.stats()
            bf_mean, bf_std = _compute_branching_factor(trie)

            epoch_results.append(
                {
                    "epoch": epoch,
                    "accuracy": accuracy,
                    "n_nodes": stats["n_nodes"],
                    "n_leaves": stats["n_leaves"],
                    "max_depth": stats["max_depth"],
                    "rumination_rejections": stats["rumination_rejections"],
                    "consolidation_merges": stats["consolidation_merges"],
                    "branching_factor_mean": bf_mean,
                    "branching_factor_std": bf_std,
                    "train_time": train_time,
                    "test_time": test_time,
                }
            )

        # Final consolidation
        trie.consolidate()

        # 5. Batch-write all epoch results to SQLite (per research pitfall 1)
        _write_results_batch(db_path, config, epoch_results)

        final_acc = epoch_results[-1]["accuracy"] if epoch_results else 0.0
        return {
            "config_id": config.config_id,
            "benchmark": config.benchmark,
            "final_accuracy": final_acc,
            "n_epochs": config.epochs,
            "status": "ok",
        }

    except Exception as e:
        logger.error(
            f"Config {config.config_id} ({config.benchmark}) failed: {e}"
        )
        return {
            "config_id": config.config_id,
            "benchmark": config.benchmark,
            "final_accuracy": 0.0,
            "n_epochs": 0,
            "status": f"error: {e}",
        }


def _find_features_file(features_dir: str, benchmark: str) -> Path:
    """Locate the cached features file for a benchmark.

    Searches for patterns like:
    - {benchmark}_10k_features.pt
    - {benchmark}_features.pt
    - {benchmark}.pt

    Args:
        features_dir: Directory containing feature files.
        benchmark: Benchmark name (e.g., "mnist", "fashion_mnist").

    Returns:
        Path to the features file.

    Raises:
        FileNotFoundError: If no matching file found.
    """
    features_path = Path(features_dir)
    candidates = [
        features_path / f"{benchmark}_10k_features.pt",
        features_path / f"{benchmark}_features.pt",
        features_path / f"{benchmark}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No features file found for benchmark '{benchmark}' in {features_dir}. "
        f"Searched: {[str(c) for c in candidates]}"
    )


def _write_results_batch(
    db_path: str,
    config: SweepConfig,
    epoch_results: list[dict[str, Any]],
) -> None:
    """Write all epoch results for a config to SQLite in a single transaction.

    Each worker opens its own connection with timeout=30s (per research pitfall 1).

    Args:
        db_path: Path to the SQLite database.
        config: The sweep configuration.
        epoch_results: List of per-epoch metric dicts.
    """
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        rows = []
        for result in epoch_results:
            rows.append(
                (
                    config.config_id,
                    config.benchmark,
                    result["epoch"],
                    config.seed,
                    config.policy_type,
                    config.assoc_threshold,
                    config.sim_threshold,
                    config.min_share,
                    config.min_count,
                    config.noise,
                    result["accuracy"],
                    result["n_nodes"],
                    result["n_leaves"],
                    result["max_depth"],
                    result["rumination_rejections"],
                    result["consolidation_merges"],
                    result["branching_factor_mean"],
                    result["branching_factor_std"],
                    result["train_time"],
                    result["test_time"],
                    config.policy_params,
                )
            )

        conn.executemany(
            """
            INSERT OR REPLACE INTO sweep_results (
                config_id, benchmark, epoch, seed, policy_type,
                assoc_threshold, sim_threshold, min_share, min_count, noise,
                accuracy, n_nodes, n_leaves, max_depth,
                rumination_rejections, consolidation_merges,
                branching_factor_mean, branching_factor_std,
                train_time, test_time, policy_params
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


# ── Runner ─────────────────────────────────────────────────────────


class SweepRunner:
    """Parallel sweep framework with SQLite storage.

    Executes N configurations across M workers using ProcessPoolExecutor,
    writing epoch-by-epoch results to a SQLite database in WAL mode.

    Args:
        db_path: Path to the SQLite database file.
        n_workers: Number of parallel worker processes (default 24, per D-24).
    """

    def __init__(self, db_path: str, n_workers: int = 24) -> None:
        self.db_path = db_path
        self.n_workers = n_workers
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database with WAL mode and schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    def run(self, configs: list[SweepConfig], features_dir: str) -> list[dict[str, Any]]:
        """Run all sweep configurations in parallel.

        Per D-24: ProcessPoolExecutor with n_workers.
        Per D-27: Run all configs to completion (no early stopping).
        Per D-28: tqdm progress bar for live progress.

        Args:
            configs: List of SweepConfig instances to execute.
            features_dir: Directory containing cached feature .pt files.

        Returns:
            List of summary dicts from each config execution.
        """
        results: list[dict[str, Any]] = []

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(
                    _run_single_config, config, features_dir, self.db_path
                ): config
                for config in configs
            }

            with tqdm(total=len(configs), desc="Sweep progress", unit="config") as pbar:
                for future in as_completed(futures):
                    config = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        status = result.get("status", "unknown")
                        acc = result.get("final_accuracy", 0.0)
                        if status == "ok":
                            pbar.set_postfix(
                                last_acc=f"{acc:.3f}",
                                benchmark=config.benchmark,
                            )
                    except Exception as e:
                        logger.error(
                            f"Config {config.config_id} raised: {e}"
                        )
                        results.append(
                            {
                                "config_id": config.config_id,
                                "benchmark": config.benchmark,
                                "final_accuracy": 0.0,
                                "n_epochs": 0,
                                "status": f"exception: {e}",
                            }
                        )
                    pbar.update(1)

        return results

    def query_results(
        self,
        benchmark: str | None = None,
        policy_type: str | None = None,
        min_accuracy: float | None = None,
        order_by: str = "accuracy DESC",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query sweep results with optional filtering.

        Args:
            benchmark: Filter by benchmark name.
            policy_type: Filter by policy type.
            min_accuracy: Minimum accuracy threshold.
            order_by: SQL ORDER BY clause.
            limit: Maximum number of results.

        Returns:
            List of result dicts.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            conditions: list[str] = []
            params: list[Any] = []

            if benchmark is not None:
                conditions.append("benchmark = ?")
                params.append(benchmark)
            if policy_type is not None:
                conditions.append("policy_type = ?")
                params.append(policy_type)
            if min_accuracy is not None:
                conditions.append("accuracy >= ?")
                params.append(min_accuracy)

            where = ""
            if conditions:
                where = "WHERE " + " AND ".join(conditions)

            query = f"SELECT * FROM sweep_results {where} ORDER BY {order_by} LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()


# ── Config generation ──────────────────────────────────────────────


def generate_global_sweep_configs(
    benchmarks: list[str],
    seed: int = 42,
) -> list[SweepConfig]:
    """Generate 4D sweep grid for global threshold baseline per D-20.

    Per D-20: assoc threshold 0.001-2.0 log-spaced, combined with linspace
    for critical region coverage (research pitfall 7).

    For initial 3D sweep (per D-22 reduced-first):
    - Fix epochs=3, consolidation=(0.05, 3)
    - Sweep assoc x sim x noise
    - That's ~24 assoc * 8 sim * 4 noise = 768 per benchmark

    Then 1D sweeps for consolidation and epochs are separate.

    Args:
        benchmarks: List of benchmark names to sweep.
        seed: Random seed.

    Returns:
        List of SweepConfig instances.
    """
    # Assoc threshold: combine geomspace + linspace per D-20, research pitfall 7
    assoc_values = np.unique(
        np.sort(
            np.concatenate(
                [
                    np.geomspace(0.001, 2.0, 15),
                    np.linspace(0.05, 1.0, 10),
                ]
            )
        )
    )

    # Similarity threshold grid
    sim_values = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

    # Noise values per D-05
    noise_values = [0.0, 0.01, 0.05, 0.1]

    # Fixed consolidation and epochs for initial 3D sweep
    fixed_min_share = 0.05
    fixed_min_count = 3
    fixed_epochs = 3

    configs: list[SweepConfig] = []
    config_id = 0

    for benchmark in benchmarks:
        for assoc in assoc_values:
            for sim in sim_values:
                for noise in noise_values:
                    configs.append(
                        SweepConfig(
                            config_id=config_id,
                            benchmark=benchmark,
                            policy_type="global",
                            assoc_threshold=float(assoc),
                            sim_threshold=float(sim),
                            min_share=fixed_min_share,
                            min_count=fixed_min_count,
                            noise=float(noise),
                            epochs=fixed_epochs,
                            seed=seed,
                        )
                    )
                    config_id += 1

    return configs


def generate_consolidation_sweep_configs(
    benchmarks: list[str],
    base_assoc_threshold: float = 0.3,
    base_sim_threshold: float = 0.1,
    seed: int = 42,
) -> list[SweepConfig]:
    """Generate 1D consolidation sweep per D-20.

    Sweeps 5 consolidation configs while holding other params at baseline.

    Args:
        benchmarks: List of benchmark names.
        base_assoc_threshold: Fixed associator threshold.
        base_sim_threshold: Fixed similarity threshold.
        seed: Random seed.

    Returns:
        List of SweepConfig instances.
    """
    consolidation_configs = [
        (0.01, 1),
        (0.03, 2),
        (0.05, 3),
        (0.10, 5),
        (0.00, 0),  # Disabled
    ]

    configs: list[SweepConfig] = []
    config_id = 100000  # Offset to avoid ID collision with main sweep

    for benchmark in benchmarks:
        for min_share, min_count in consolidation_configs:
            configs.append(
                SweepConfig(
                    config_id=config_id,
                    benchmark=benchmark,
                    policy_type="global",
                    assoc_threshold=base_assoc_threshold,
                    sim_threshold=base_sim_threshold,
                    min_share=min_share,
                    min_count=min_count,
                    noise=0.0,
                    epochs=3,
                    seed=seed,
                )
            )
            config_id += 1

    return configs


def generate_epoch_sweep_configs(
    benchmarks: list[str],
    base_assoc_threshold: float = 0.3,
    base_sim_threshold: float = 0.1,
    seed: int = 42,
) -> list[SweepConfig]:
    """Generate 1D epoch sweep per D-06.

    Sweeps epochs (1, 3, 5) while holding other params at baseline.

    Args:
        benchmarks: List of benchmark names.
        base_assoc_threshold: Fixed associator threshold.
        base_sim_threshold: Fixed similarity threshold.
        seed: Random seed.

    Returns:
        List of SweepConfig instances.
    """
    epoch_values = [1, 3, 5]

    configs: list[SweepConfig] = []
    config_id = 200000  # Offset to avoid ID collision

    for benchmark in benchmarks:
        for epochs in epoch_values:
            configs.append(
                SweepConfig(
                    config_id=config_id,
                    benchmark=benchmark,
                    policy_type="global",
                    assoc_threshold=base_assoc_threshold,
                    sim_threshold=base_sim_threshold,
                    min_share=0.05,
                    min_count=3,
                    noise=0.0,
                    epochs=epochs,
                    seed=seed,
                )
            )
            config_id += 1

    return configs


def generate_adaptive_sweep_configs(
    policy_type: str,
    benchmarks: list[str],
    seed: int = 42,
) -> list[SweepConfig]:
    """Generate sweep configs for an adaptive strategy per D-29.

    Sweeps policy-specific hyperparameters alongside base threshold.
    Each policy type has its own hyperparameter grid.

    Args:
        policy_type: One of "ema", "mean_std", "depth", "purity".
        benchmarks: List of benchmark names.
        seed: Random seed.

    Returns:
        List of SweepConfig instances.
    """
    # Base associator thresholds (reduced grid for adaptive)
    base_assoc = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    base_sim = [0.0, 0.05, 0.1, 0.2]

    # Policy-specific hyperparameter grids
    policy_grids: dict[str, list[dict[str, Any]]] = {
        "ema": [
            {"alpha": a} for a in [0.01, 0.05, 0.1, 0.2, 0.5]
        ],
        "mean_std": [
            {"n_std": n} for n in [0.5, 1.0, 1.5, 2.0, 3.0]
        ],
        "depth": [
            {"decay_factor": d} for d in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        ],
        "purity": [
            {"sensitivity": s} for s in [0.1, 0.3, 0.5, 0.7, 1.0]
        ],
    }

    if policy_type not in policy_grids:
        raise ValueError(
            f"Unknown adaptive policy_type: {policy_type!r}. "
            f"Expected one of: {list(policy_grids.keys())}"
        )

    grid = policy_grids[policy_type]
    configs: list[SweepConfig] = []
    config_id = 300000 + list(policy_grids.keys()).index(policy_type) * 100000

    for benchmark in benchmarks:
        for assoc in base_assoc:
            for sim in base_sim:
                for params in grid:
                    configs.append(
                        SweepConfig(
                            config_id=config_id,
                            benchmark=benchmark,
                            policy_type=policy_type,
                            assoc_threshold=float(assoc),
                            sim_threshold=float(sim),
                            min_share=0.05,
                            min_count=3,
                            noise=0.0,
                            epochs=3,
                            seed=seed,
                            policy_params=json.dumps(params),
                        )
                    )
                    config_id += 1

    return configs


# ── CLI entry point ────────────────────────────────────────────────


def main() -> None:
    """Command-line entry point for running sweeps."""
    import argparse

    parser = argparse.ArgumentParser(description="Run threshold sweep")
    parser.add_argument(
        "--db", type=str, default="sweep_results.db", help="SQLite database path"
    )
    parser.add_argument(
        "--features-dir", type=str, required=True, help="Directory with cached features"
    )
    parser.add_argument(
        "--workers", type=int, default=24, help="Number of parallel workers"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["mnist", "fashion_mnist", "cifar10", "text_4class", "text_20class"],
        help="Benchmarks to sweep",
    )
    parser.add_argument(
        "--policy", type=str, default="global", help="Policy type to sweep"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.policy == "global":
        configs = generate_global_sweep_configs(args.benchmarks, seed=42)
    else:
        configs = generate_adaptive_sweep_configs(
            args.policy, args.benchmarks, seed=42
        )

    logger.info(f"Generated {len(configs)} configs for {args.policy} sweep")
    logger.info(f"Benchmarks: {args.benchmarks}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Database: {args.db}")

    runner = SweepRunner(args.db, n_workers=args.workers)
    results = runner.run(configs, args.features_dir)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    logger.info(f"\nCompleted: {ok_count}/{len(results)} configs succeeded")


if __name__ == "__main__":
    main()
