"""Unit tests for the parallel sweep framework.

Tests SQLite initialization, WAL mode, config generation, pickling,
worker function, concurrent writes, and result querying.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pytest
import torch

# Import sweep_runner from scripts directory
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from sweep.sweep_runner import (
    SweepConfig,
    SweepRunner,
    _run_single_config,
    _write_results_batch,
    generate_global_sweep_configs,
)


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Return a temporary database path."""
    return str(tmp_path / "test_sweep.db")


@pytest.fixture
def features_dir(tmp_path: Path) -> str:
    """Create a temporary features directory with tiny synthetic data.

    Creates features for a 'test_bench' benchmark with 7 categories,
    10 training samples each (70 total), and 14 test samples (2 per category).
    """
    features_path = tmp_path / "features"
    features_path.mkdir()

    n_categories = 7
    n_train_per_cat = 10
    n_test_per_cat = 2

    gen = torch.Generator().manual_seed(42)

    # Training data
    train_x_list = []
    train_y_list = []
    for cat in range(n_categories):
        # Create category-specific octonionic features (slightly clustered)
        center = torch.randn(8, generator=gen)
        center = center / center.norm()
        for _ in range(n_train_per_cat):
            sample = center + 0.1 * torch.randn(8, generator=gen)
            sample = sample / sample.norm()
            train_x_list.append(sample)
            train_y_list.append(cat)

    train_x = torch.stack(train_x_list)
    train_y = torch.tensor(train_y_list)

    # Test data
    test_x_list = []
    test_y_list = []
    for cat in range(n_categories):
        center = torch.randn(8, generator=gen)
        center = center / center.norm()
        for _ in range(n_test_per_cat):
            sample = center + 0.1 * torch.randn(8, generator=gen)
            sample = sample / sample.norm()
            test_x_list.append(sample)
            test_y_list.append(cat)

    test_x = torch.stack(test_x_list)
    test_y = torch.tensor(test_y_list)

    data = {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
    }

    torch.save(data, features_path / "test_bench_features.pt")
    return str(features_path)


def _make_config(
    config_id: int = 0,
    benchmark: str = "test_bench",
    epochs: int = 2,
    seed: int = 42,
    assoc_threshold: float = 0.3,
    sim_threshold: float = 0.1,
    noise: float = 0.0,
) -> SweepConfig:
    """Helper to create a SweepConfig with test defaults."""
    return SweepConfig(
        config_id=config_id,
        benchmark=benchmark,
        policy_type="global",
        assoc_threshold=assoc_threshold,
        sim_threshold=sim_threshold,
        min_share=0.05,
        min_count=3,
        noise=noise,
        epochs=epochs,
        seed=seed,
    )


# ── Test 1: _init_db creates sweep_results table ──────────────────


def test_sqlite_init(db_path: str) -> None:
    """_init_db creates SQLite database with sweep_results table and all expected columns."""
    runner = SweepRunner(db_path, n_workers=1)

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("PRAGMA table_info(sweep_results)")
        columns = {row[1] for row in cursor.fetchall()}

        expected_columns = {
            "config_id",
            "benchmark",
            "epoch",
            "seed",
            "policy_type",
            "assoc_threshold",
            "sim_threshold",
            "min_share",
            "min_count",
            "noise",
            "accuracy",
            "n_nodes",
            "n_leaves",
            "max_depth",
            "rumination_rejections",
            "consolidation_merges",
            "branching_factor_mean",
            "branching_factor_std",
            "train_time",
            "test_time",
            "policy_params",
            "timestamp",
        }

        assert expected_columns.issubset(
            columns
        ), f"Missing columns: {expected_columns - columns}"
    finally:
        conn.close()


# ── Test 2: WAL journal mode ──────────────────────────────────────


def test_wal_mode(db_path: str) -> None:
    """_init_db sets WAL journal mode on the database."""
    runner = SweepRunner(db_path, n_workers=1)

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode == "wal", f"Expected WAL mode, got {mode}"
    finally:
        conn.close()


# ── Test 3: Config generation produces expected benchmarks/ranges ─


def test_global_sweep_configs_benchmarks() -> None:
    """generate_global_sweep_configs produces configs with all benchmarks and expected param ranges."""
    benchmarks = ["mnist", "fashion_mnist", "cifar10", "text_4class", "text_20class"]
    configs = generate_global_sweep_configs(benchmarks)

    # All 5 benchmarks present
    benchmark_set = {c.benchmark for c in configs}
    assert benchmark_set == set(benchmarks), f"Missing benchmarks: {set(benchmarks) - benchmark_set}"

    # Each benchmark has the same number of configs
    counts = {}
    for c in configs:
        counts[c.benchmark] = counts.get(c.benchmark, 0) + 1
    count_vals = list(counts.values())
    assert len(set(count_vals)) == 1, f"Uneven config counts: {counts}"

    # Param ranges are reasonable
    assoc_vals = sorted(set(c.assoc_threshold for c in configs))
    assert len(assoc_vals) >= 15, f"Too few assoc values: {len(assoc_vals)}"
    assert min(assoc_vals) < 0.01, f"Min assoc too high: {min(assoc_vals)}"
    assert max(assoc_vals) >= 1.5, f"Max assoc too low: {max(assoc_vals)}"

    sim_vals = sorted(set(c.sim_threshold for c in configs))
    assert len(sim_vals) == 8, f"Expected 8 sim values, got {len(sim_vals)}"

    noise_vals = sorted(set(c.noise for c in configs))
    assert len(noise_vals) == 4, f"Expected 4 noise values, got {len(noise_vals)}"


# ── Test 4: Config generation covers critical region ──────────────


def test_global_sweep_configs_critical_region() -> None:
    """generate_global_sweep_configs produces configs where assoc_threshold values
    include both geomspace and linspace points in 0.05-1.0 range."""
    configs = generate_global_sweep_configs(["mnist"])

    assoc_vals = sorted(set(c.assoc_threshold for c in configs))

    # The critical region 0.05-1.0 should have good coverage
    critical_region = [v for v in assoc_vals if 0.05 <= v <= 1.0]
    assert len(critical_region) >= 10, (
        f"Critical region 0.05-1.0 only has {len(critical_region)} points, "
        f"expected >= 10 from combined geomspace + linspace"
    )

    # Should include values near both geomspace and linspace endpoints
    # linspace(0.05, 1.0, 10) includes 0.05 and 1.0
    assert any(
        abs(v - 0.05) < 0.01 for v in assoc_vals
    ), "Missing value near 0.05"
    assert any(
        abs(v - 1.0) < 0.01 for v in assoc_vals
    ), "Missing value near 1.0"


# ── Test 5: SweepConfig is picklable ──────────────────────────────


def test_config_picklable() -> None:
    """SweepConfig is picklable, as required for ProcessPoolExecutor."""
    config = _make_config(
        config_id=42,
        benchmark="mnist",
        epochs=3,
        seed=123,
        assoc_threshold=0.5,
        sim_threshold=0.2,
        noise=0.01,
    )

    pickled = pickle.dumps(config)
    restored = pickle.loads(pickled)

    assert restored.config_id == config.config_id
    assert restored.benchmark == config.benchmark
    assert restored.policy_type == config.policy_type
    assert restored.assoc_threshold == config.assoc_threshold
    assert restored.sim_threshold == config.sim_threshold
    assert restored.min_share == config.min_share
    assert restored.min_count == config.min_count
    assert restored.noise == config.noise
    assert restored.epochs == config.epochs
    assert restored.seed == config.seed
    assert restored.policy_params == config.policy_params


# ── Test 6: Worker writes epoch-by-epoch results ──────────────────


def test_worker_writes_epochs(db_path: str, features_dir: str) -> None:
    """Worker function writes epoch-by-epoch results to SQLite.

    Test with 1 config, 2 epochs on tiny synthetic data.
    """
    # Initialize DB first
    runner = SweepRunner(db_path, n_workers=1)

    config = _make_config(config_id=99, epochs=2)

    # Run worker directly (not via ProcessPoolExecutor)
    result = _run_single_config(config, features_dir, db_path)

    assert result["status"] == "ok", f"Worker failed: {result['status']}"
    assert result["config_id"] == 99
    assert result["n_epochs"] == 2
    assert 0.0 <= result["final_accuracy"] <= 1.0

    # Verify SQLite has 2 rows for this config_id (one per epoch)
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT epoch, accuracy, n_nodes FROM sweep_results WHERE config_id = ? ORDER BY epoch",
            (99,),
        )
        rows = cursor.fetchall()
        assert len(rows) == 2, f"Expected 2 epoch rows, got {len(rows)}"
        assert rows[0][0] == 0, f"First epoch should be 0, got {rows[0][0]}"
        assert rows[1][0] == 1, f"Second epoch should be 1, got {rows[1][0]}"

        # Both epochs should have valid metrics
        for row in rows:
            assert row[1] is not None, "accuracy should not be None"
            assert row[2] is not None, "n_nodes should not be None"
            assert row[2] >= 1, "n_nodes should be >= 1 (at least root)"
    finally:
        conn.close()


# ── Test 7: Concurrent writes don't deadlock ─────────────────────


def test_concurrent_writes(db_path: str, features_dir: str) -> None:
    """Multiple workers writing to same database don't deadlock.

    3 workers, 3 configs each, all complete without 'database is locked' errors.
    """
    runner = SweepRunner(db_path, n_workers=3)

    configs = [
        _make_config(config_id=i, epochs=1, seed=i)
        for i in range(3)
    ]

    # Run via ProcessPoolExecutor to verify concurrent writes
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_run_single_config, cfg, features_dir, db_path): cfg
            for cfg in configs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    # All 3 configs should complete successfully
    ok_results = [r for r in results if r["status"] == "ok"]
    assert len(ok_results) == 3, (
        f"Expected 3 successful configs, got {len(ok_results)}. "
        f"Statuses: {[r['status'] for r in results]}"
    )

    # Verify all 3 configs' results are in the database
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT DISTINCT config_id FROM sweep_results ORDER BY config_id"
        )
        config_ids = [row[0] for row in cursor.fetchall()]
        assert set(config_ids) == {0, 1, 2}, f"Expected config_ids {{0,1,2}}, got {config_ids}"
    finally:
        conn.close()


# ── Test 8: Results can be queried by benchmark and policy_type ───


def test_query_results(db_path: str, features_dir: str) -> None:
    """Results can be queried by benchmark and policy_type after sweep completes."""
    runner = SweepRunner(db_path, n_workers=1)

    config = _make_config(config_id=50, epochs=1, benchmark="test_bench")

    # Run one config
    result = _run_single_config(config, features_dir, db_path)
    assert result["status"] == "ok"

    # Query by benchmark
    results = runner.query_results(benchmark="test_bench")
    assert len(results) >= 1, "Should have results for test_bench"
    assert all(r["benchmark"] == "test_bench" for r in results)

    # Query by policy_type
    results = runner.query_results(policy_type="global")
    assert len(results) >= 1, "Should have results for global policy"
    assert all(r["policy_type"] == "global" for r in results)

    # Query by benchmark AND policy_type
    results = runner.query_results(benchmark="test_bench", policy_type="global")
    assert len(results) >= 1

    # Query with min_accuracy (should work even if accuracy is low on synthetic data)
    results = runner.query_results(min_accuracy=0.0)
    assert len(results) >= 1
