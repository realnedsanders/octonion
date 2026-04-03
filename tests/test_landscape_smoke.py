"""Lightweight tests for landscape experiment resume/skip logic.

Full end-to-end pipeline runs and gate evaluation live in
scripts/run_landscape.py (use --smoke for minimal config).
"""

from __future__ import annotations

import json
from pathlib import Path

from octonion.landscape._experiment import (
    _result_exists,
    _result_path,
)


def test_resume_skips_existing(tmp_path: Path) -> None:
    """Verify _result_exists correctly detects saved results.

    Tests the resume/skip mechanism without running actual training:
    writes a synthetic result file and confirms _result_exists finds it,
    and that missing results are correctly reported as absent.
    """
    output_dir = str(tmp_path / "landscape_resume")
    task = "algebra_native_single"
    optimizer = "adam"
    algebra = "R"
    seed = 0

    # Before saving: result should not exist
    assert not _result_exists(output_dir, task, optimizer, algebra, seed), (
        "Result should not exist before saving"
    )

    # Write a synthetic result file
    result_path = _result_path(output_dir, task, optimizer, algebra, seed)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_data = {
        "train_losses": [1.0, 0.8, 0.6, 0.5, 0.4],
        "val_losses": [1.1, 0.9, 0.7, 0.6, 0.5],
        "best_val_loss": 0.5,
    }
    with open(result_path, "w") as f:
        json.dump(result_data, f)

    # After saving: result should exist
    assert _result_exists(output_dir, task, optimizer, algebra, seed), (
        "Result should exist after saving"
    )

    # Different seed should not exist
    assert not _result_exists(output_dir, task, optimizer, algebra, seed=1), (
        "Result for different seed should not exist"
    )

    # Different algebra should not exist
    assert not _result_exists(output_dir, task, optimizer, "O", seed), (
        "Result for different algebra should not exist"
    )

    # Verify the saved file is valid JSON with expected structure
    with open(result_path) as f:
        loaded = json.load(f)
    assert loaded["train_losses"] == result_data["train_losses"]
    assert loaded["best_val_loss"] == result_data["best_val_loss"]
