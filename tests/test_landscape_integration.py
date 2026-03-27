"""Integration tests for post-training analysis pipeline.

Verifies that the full pipeline (train -> checkpoint -> post-analysis ->
result.json) produces Hessian spectrum, curvature, and gradient variance
data in the format that analyze_landscape.py expects.

These tests close the gaps identified in 05-RESEARCH-gaps.md:
- Gap 1: Hessian spectrum and curvature data missing from result.json
- Gap 2: Intermediate Hessian checkpoints (0.25, 0.50) not saved
- Gap 3: Gradient variance not collected across seeds
- Gap 4: Smoke tests don't verify post-training analysis outputs
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

from octonion.baselines._config import AlgebraType
from octonion.landscape._experiment import (
    LandscapeConfig,
    _hessian_checkpoint_dir,
    run_landscape_experiment,
)

# Import run_post_analysis from the scripts directory (not a package)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from run_post_analysis import run_post_analysis  # noqa: E402


def _integration_config(output_dir: str) -> LandscapeConfig:
    """Minimal config for integration testing.

    Uses:
    - 1 task (algebra_native_single)
    - 1 algebra (REAL -- smallest, fastest)
    - 1 optimizer (adam)
    - 1 seed (seed 0, which is also a hessian_seed)
    - 20 epochs (enough for 4 checkpoints at epochs 5, 10, 15, 20)
    - base_hidden=4 (tiny model, ~200 params, uses full Hessian not Lanczos)
    - CPU only
    - 3 curvature directions (minimal but enough to verify)

    Args:
        output_dir: Temporary directory for experiment output.

    Returns:
        LandscapeConfig with tiny models and minimal data for integration tests.
    """
    return LandscapeConfig(
        tasks=["algebra_native_single"],
        algebras=[AlgebraType.REAL],
        optimizers=["adam"],
        seeds=[0],
        epochs=20,
        base_hidden=4,
        n_train=100,
        n_test=50,
        output_dir=output_dir,
        device="cpu",
        hessian_seeds=[0],
        hessian_checkpoints=[0.0, 0.25, 0.5, 0.75, 1.0],
        n_curvature_directions=3,
    )


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_intermediate_hessian_checkpoints_saved(tmp_path: Path) -> None:
    """Verify Hessian checkpoints saved at all 5 fractions (0.00, 0.25, 0.50, 0.75, 1.00)."""
    config = _integration_config(str(tmp_path))
    run_landscape_experiment(config)

    ckpt_dir = _hessian_checkpoint_dir(
        str(tmp_path), "algebra_native_single", "adam", "R", 0
    )

    assert (ckpt_dir / "checkpoint_0.00.pt").exists(), "Missing checkpoint at fraction 0.00"
    assert (ckpt_dir / "checkpoint_0.25.pt").exists(), "Missing checkpoint at fraction 0.25"
    assert (ckpt_dir / "checkpoint_0.50.pt").exists(), "Missing checkpoint at fraction 0.50"
    assert (ckpt_dir / "checkpoint_0.75.pt").exists(), "Missing checkpoint at fraction 0.75"
    assert (ckpt_dir / "checkpoint_1.00.pt").exists(), "Missing checkpoint at fraction 1.00"

    # Verify checkpoints are loadable (valid torch format)
    import torch

    for frac in ["0.00", "0.25", "0.50", "0.75", "1.00"]:
        state = torch.load(ckpt_dir / f"checkpoint_{frac}.pt", weights_only=True)
        assert isinstance(state, dict), f"Checkpoint {frac} is not a state_dict"
        assert len(state) > 0, f"Checkpoint {frac} is empty"


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_post_analysis_produces_hessian_data(tmp_path: Path) -> None:
    """Verify post-analysis computes Hessian eigenspectrum from checkpoints."""
    config = _integration_config(str(tmp_path))
    run_landscape_experiment(config)

    # Run post-analysis
    counts = run_post_analysis(str(tmp_path), config, device="cpu")

    # Load result.json and verify hessian_spectrum key
    result_file = (
        Path(tmp_path) / "algebra_native_single" / "adam" / "R" / "seed_0" / "result.json"
    )
    with open(result_file) as f:
        result = json.load(f)

    assert "hessian_spectrum" in result, "hessian_spectrum key missing from result.json"
    hessian = result["hessian_spectrum"]
    assert isinstance(hessian, dict), "hessian_spectrum should be a dict"
    assert "1.0" in hessian or "1.00" in hessian, "Missing converged checkpoint spectrum"

    # Eigenvalues should be a non-empty list of floats
    key = "1.0" if "1.0" in hessian else "1.00"
    eigenvalues = hessian[key]
    assert isinstance(eigenvalues, list), "Eigenvalues should be a list"
    assert len(eigenvalues) > 0, "Eigenvalue list is empty"
    assert all(isinstance(v, (int, float)) for v in eigenvalues), "Eigenvalues should be numeric"


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_post_analysis_produces_curvature_data(tmp_path: Path) -> None:
    """Verify post-analysis computes Bill & Cox curvature on converged model."""
    config = _integration_config(str(tmp_path))
    run_landscape_experiment(config)

    run_post_analysis(str(tmp_path), config, device="cpu")

    result_file = (
        Path(tmp_path) / "algebra_native_single" / "adam" / "R" / "seed_0" / "result.json"
    )
    with open(result_file) as f:
        result = json.load(f)

    assert "curvature" in result, "curvature key missing from result.json"
    assert isinstance(result["curvature"], (int, float)), "curvature should be numeric"
    assert math.isfinite(result["curvature"]), "curvature should be finite"

    assert "curvature_detail" in result, "curvature_detail key missing"
    detail = result["curvature_detail"]
    assert "mean_curvature" in detail, "mean_curvature missing from detail"
    assert "median_curvature" in detail, "median_curvature missing from detail"
    assert "curvatures" in detail, "curvatures list missing from detail"
    assert len(detail["curvatures"]) == 3, (
        f"Expected 3 curvature directions, got {len(detail['curvatures'])}"
    )


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_post_analysis_produces_gradient_variance(tmp_path: Path) -> None:
    """Verify post-analysis collects gradient variance across seeds."""
    config = _integration_config(str(tmp_path))
    run_landscape_experiment(config)

    run_post_analysis(str(tmp_path), config, device="cpu")

    result_file = (
        Path(tmp_path) / "algebra_native_single" / "adam" / "R" / "seed_0" / "result.json"
    )
    with open(result_file) as f:
        result = json.load(f)

    assert "gradient_stats" in result, "gradient_stats key missing from result.json"
    grad = result["gradient_stats"]
    assert "grad_norm_std" in grad, "grad_norm_std missing from gradient_stats"
    assert isinstance(grad["grad_norm_std"], (int, float)), "grad_norm_std should be numeric"
    assert "cross_seed_variance" in grad, "cross_seed_variance missing from gradient_stats"
