"""Go/no-go gate evaluation for octonion optimization landscape.

Provides tiered GREEN/YELLOW/RED verdict based on how octonionic models
compare to the R8_DENSE (no-structure) baseline across multiple tasks.

Verdict semantics:
- GREEN: Proceed. O within 2x of R8D on ALL tasks, no high divergence.
- YELLOW: Proceed with caution. Intermediate results.
- RED: Pivot. O worse than 3x on majority of tasks, or high divergence.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np


class GateVerdict(Enum):
    """Tiered go/no-go gate verdict."""

    GREEN = "GREEN"    # Proceed: O within 2x on ALL tasks
    YELLOW = "YELLOW"  # Proceed with caution
    RED = "RED"        # Pivot: worse than 3x on majority


def evaluate_gate(
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate go/no-go gate from per-task optimization results.

    Args:
        results: Dict keyed by task_name, each with:
            - 'O': {'final_val_losses': list[float], 'initial_loss': float}
            - 'R8_DENSE': {'final_val_losses': list[float]}

    Returns:
        Dict with:
        - verdict: GateVerdict (GREEN, YELLOW, or RED)
        - per_task: {task_name: task_metrics_dict}
        - summary: Human-readable summary string

    Gate logic per task:
        a. best_ratio = min(O_losses) / min(R8D_losses)
        b. median_ratio = median(O_losses) / median(R8D_losses)
        c. gate_ratio = min(best_ratio, median_ratio) -- most favorable for O
        d. divergence_rate = fraction of O seeds where loss > 10x initial_loss
        e. within_2x = gate_ratio <= 2.0
        f. within_3x = gate_ratio <= 3.0

    Aggregate verdict:
        - GREEN: n_within_2x == n_tasks AND no high divergence (>50%)
        - RED: any_high_divergence OR n_within_3x < majority
        - YELLOW: otherwise
    """
    n_tasks = len(results)
    per_task: dict[str, dict[str, Any]] = {}
    n_within_2x = 0
    n_within_3x = 0
    any_high_divergence = False

    for task_name, task_data in results.items():
        o_data = task_data["O"]
        r8d_data = task_data["R8_DENSE"]

        o_losses = np.array(o_data["final_val_losses"], dtype=np.float64)
        r8d_losses = np.array(r8d_data["final_val_losses"], dtype=np.float64)
        initial_loss = float(o_data["initial_loss"])

        # Compute ratios
        best_ratio = float(np.min(o_losses) / np.min(r8d_losses))
        median_ratio = float(np.median(o_losses) / np.median(r8d_losses))
        gate_ratio = min(best_ratio, median_ratio)

        # Divergence rate: fraction of O seeds with loss > 10x initial_loss
        divergence_rate = float(np.mean(o_losses > 10.0 * initial_loss))

        within_2x = gate_ratio <= 2.0
        within_3x = gate_ratio <= 3.0

        if within_2x:
            n_within_2x += 1
        if within_3x:
            n_within_3x += 1
        if divergence_rate > 0.5:
            any_high_divergence = True

        per_task[task_name] = {
            "best_ratio": best_ratio,
            "median_ratio": median_ratio,
            "gate_ratio": gate_ratio,
            "divergence_rate": divergence_rate,
            "within_2x": within_2x,
            "within_3x": within_3x,
        }

    # Aggregate verdict
    majority = n_tasks / 2.0
    if n_within_2x == n_tasks and not any_high_divergence:
        verdict = GateVerdict.GREEN
    elif any_high_divergence or n_within_3x < majority:
        verdict = GateVerdict.RED
    else:
        verdict = GateVerdict.YELLOW

    # Summary string
    summary_parts = [f"Gate verdict: {verdict.value}"]
    summary_parts.append(f"Tasks within 2x: {n_within_2x}/{n_tasks}")
    summary_parts.append(f"Tasks within 3x: {n_within_3x}/{n_tasks}")
    if any_high_divergence:
        summary_parts.append("WARNING: High divergence detected (>50% of seeds)")
    summary = "; ".join(summary_parts)

    return {
        "verdict": verdict,
        "per_task": per_task,
        "summary": summary,
    }
