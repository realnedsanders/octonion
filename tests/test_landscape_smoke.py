"""End-to-end smoke test for landscape experiment pipeline.

Tests the complete pipeline: task generation -> model building ->
training -> results saved -> gate evaluation.

Uses minimal configuration (tiny models, 2 seeds, 5 epochs)
to validate the pipeline works end-to-end.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from octonion.baselines._config import AlgebraType
from octonion.landscape._experiment import (
    LandscapeConfig,
    _result_exists,
    _result_path,
    run_landscape_experiment,
)
from octonion.landscape._gate import GateVerdict, evaluate_gate


def _smoke_config(output_dir: str) -> LandscapeConfig:
    """Build a minimal smoke test configuration.

    Args:
        output_dir: Temporary directory for experiment output.

    Returns:
        LandscapeConfig with tiny models and minimal data.
    """
    return LandscapeConfig(
        tasks=["algebra_native_single"],
        algebras=[AlgebraType.REAL, AlgebraType.OCTONION],
        optimizers=["adam"],
        seeds=[0, 1],
        epochs=5,
        base_hidden=4,
        n_train=500,
        n_test=100,
        output_dir=output_dir,
        device="cpu",
        hessian_seeds=[0],
        hessian_checkpoints=[0.0, 1.0],
        n_curvature_directions=3,
    )


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_smoke_end_to_end(tmp_path: Path) -> None:
    """Run full pipeline: task gen -> training -> results saved.

    Verifies:
    - Results dict is non-empty
    - Each run has train_losses, val_losses, best_val_loss
    - Result files exist on disk (incremental save)
    - All 2 algebras x 2 seeds = 4 runs complete
    """
    output_dir = str(tmp_path / "landscape_smoke")
    config = _smoke_config(output_dir)

    results = run_landscape_experiment(config)

    # Results dict is non-empty
    assert results, "Results dict is empty"
    assert "algebra_native_single" in results

    # Check all algebra/seed combinations
    adam_results = results["algebra_native_single"]["adam"]
    assert "R" in adam_results, "Missing REAL algebra results"
    assert "O" in adam_results, "Missing OCTONION algebra results"

    n_runs = 0
    for alg_name in ["R", "O"]:
        alg_data = adam_results[alg_name]
        for seed in [0, 1]:
            assert seed in alg_data, f"Missing seed {seed} for {alg_name}"
            run = alg_data[seed]

            # Each run has required fields
            assert "train_losses" in run, f"Missing train_losses for {alg_name}/seed_{seed}"
            assert "val_losses" in run, f"Missing val_losses for {alg_name}/seed_{seed}"
            assert "best_val_loss" in run, f"Missing best_val_loss for {alg_name}/seed_{seed}"
            assert len(run["train_losses"]) == 5, (
                f"Expected 5 epochs, got {len(run['train_losses'])} for {alg_name}/seed_{seed}"
            )
            assert run["best_val_loss"] < float("inf"), (
                f"best_val_loss is inf for {alg_name}/seed_{seed}"
            )
            n_runs += 1

    assert n_runs == 4, f"Expected 4 runs, got {n_runs}"

    # Verify result files exist on disk
    for alg_name in ["R", "O"]:
        for seed in [0, 1]:
            assert _result_exists(
                output_dir, "algebra_native_single", "adam", alg_name, seed
            ), f"Result file missing for {alg_name}/seed_{seed}"

            # Verify JSON is valid
            path = _result_path(
                output_dir, "algebra_native_single", "adam", alg_name, seed
            )
            with open(path) as f:
                loaded = json.load(f)
            assert "train_losses" in loaded


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_smoke_resume_skips_existing(tmp_path: Path) -> None:
    """Run once, then run again. Second run should skip all existing results.

    Verifies resume/skip logic by checking that the second run is
    significantly faster (near-instant) compared to the first run.
    """
    output_dir = str(tmp_path / "landscape_resume")
    config = _smoke_config(output_dir)

    # First run: trains models
    t0 = time.time()
    results1 = run_landscape_experiment(config)
    t_first = time.time() - t0

    # Second run: should skip everything
    t0 = time.time()
    results2 = run_landscape_experiment(config)
    t_second = time.time() - t0

    # Second run should be much faster (< 50% of first)
    # Because all results are already saved
    assert t_second < t_first * 0.5 or t_second < 1.0, (
        f"Resume not skipping: first={t_first:.1f}s, second={t_second:.1f}s"
    )

    # Results should match
    for alg_name in ["R", "O"]:
        for seed in [0, 1]:
            r1 = results1["algebra_native_single"]["adam"][alg_name][seed]
            r2 = results2["algebra_native_single"]["adam"][alg_name][seed]
            assert r1["best_val_loss"] == r2["best_val_loss"], (
                f"Results differ for {alg_name}/seed_{seed}"
            )


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_smoke_gate_evaluation(tmp_path: Path) -> None:
    """After smoke run, call evaluate_gate and verify it returns a valid GateVerdict.

    Uses REAL and OCTONION results (not R8_DENSE which the gate expects),
    so we construct synthetic gate input from the smoke results.
    """
    output_dir = str(tmp_path / "landscape_gate")
    config = _smoke_config(output_dir)

    results = run_landscape_experiment(config)

    # Build gate input using R as stand-in for R8_DENSE
    # (smoke test doesn't run R8_DENSE, but we test the gate interface)
    adam_results = results["algebra_native_single"]["adam"]

    o_losses = []
    r_losses = []
    initial_loss = 1.0

    for seed in [0, 1]:
        o_run = adam_results["O"][seed]
        r_run = adam_results["R"][seed]

        o_losses.append(o_run["final_val_loss"])
        r_losses.append(r_run["final_val_loss"])

        vl = o_run.get("val_losses", [])
        if vl:
            initial_loss = max(initial_loss, vl[0])

    gate_input = {
        "algebra_native_single": {
            "O": {
                "final_val_losses": o_losses,
                "initial_loss": initial_loss,
            },
            "R8_DENSE": {
                "final_val_losses": r_losses,
            },
        },
    }

    gate_result = evaluate_gate(gate_input)

    # Verify gate result structure
    assert "verdict" in gate_result
    assert isinstance(gate_result["verdict"], GateVerdict)
    assert gate_result["verdict"] in (GateVerdict.GREEN, GateVerdict.YELLOW, GateVerdict.RED)
    assert "per_task" in gate_result
    assert "summary" in gate_result
    assert "algebra_native_single" in gate_result["per_task"]

    task_metrics = gate_result["per_task"]["algebra_native_single"]
    assert "gate_ratio" in task_metrics
    assert "divergence_rate" in task_metrics
    assert "within_2x" in task_metrics or "within_3x" in task_metrics
