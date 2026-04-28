"""Statistical testing utilities for pairwise algebra comparison.

Provides:
- paired_comparison: Paired t-test, Wilcoxon, effect size, and CI
- cohen_d: Standard Cohen's d effect size
- holm_bonferroni: Holm-Bonferroni step-down correction for multiple testing
- confidence_interval: CI using t-distribution (handles small N)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats  # type: ignore[import-untyped]


def cohen_d(a: list[float], b: list[float]) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation for the denominator.

    Args:
        a: First group of measurements.
        b: Second group of measurements.

    Returns:
        Cohen's d value. Positive means a > b.
    """
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)

    n_a, n_b = len(a_arr), len(b_arr)
    if n_a < 2 or n_b < 2:
        # Not enough samples for pooled std; fall back to sign of mean difference.
        mean_diff = float(np.mean(a_arr) - np.mean(b_arr))
        return float(np.sign(mean_diff)) if mean_diff != 0.0 else 0.0

    var_a = np.var(a_arr, ddof=1)
    var_b = np.var(b_arr, ddof=1)

    # Pooled standard deviation
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1e-15

    return float((np.mean(a_arr) - np.mean(b_arr)) / pooled_std)


def confidence_interval(
    data: list[float], confidence: float = 0.95
) -> tuple[float, float]:
    """Compute confidence interval using t-distribution.

    Handles small sample sizes via t-distribution instead of normal.

    Args:
        data: Sample data.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    arr = np.array(data, dtype=np.float64)
    n = len(arr)
    if n < 2:
        mean = float(np.mean(arr))
        return (mean, mean)

    mean = float(np.mean(arr))
    sem = float(stats.sem(arr))
    alpha = 1 - confidence
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    margin = t_crit * sem

    return (mean - margin, mean + margin)


def paired_comparison(
    results_a: list[float], results_b: list[float]
) -> dict[str, Any]:
    """Perform paired statistical comparison between two result sets.

    Computes:
    - Paired t-test (parametric)
    - Wilcoxon signed-rank test (non-parametric)
    - Cohen's d effect size
    - 95% confidence interval for mean difference

    Args:
        results_a: First set of results (e.g., accuracies across seeds).
        results_b: Second set of results (same length as results_a).

    Returns:
        Dict with t_stat, t_p_value, w_stat, w_p_value, effect_size,
        mean_diff, ci_lower, ci_upper.
    """
    a = np.array(results_a, dtype=np.float64)
    b = np.array(results_b, dtype=np.float64)
    diff = a - b

    # Paired t-test
    # Degenerate case 1: all differences are zero (identical lists)
    # Degenerate case 2: all differences are the same non-zero constant
    #   (zero variance in diff — scipy ttest_rel emits catastrophic cancellation
    #   warning because it cannot distinguish signal from rounding error)
    diff_std = float(np.std(diff, ddof=1)) if len(diff) >= 2 else 0.0
    if np.all(diff == 0):
        t_stat_val, t_p_val = 0.0, 1.0
        w_stat_val, w_p_val = 0.0, 1.0
    else:
        # t-test: guard against zero-variance differences (constant nonzero diff)
        # which causes catastrophic cancellation in scipy's moment calculation.
        if diff_std == 0.0:
            t_stat_val = float("inf") * float(np.sign(np.mean(diff)))
            t_p_val = 0.0
        else:
            t_result = stats.ttest_rel(a, b)
            t_stat_val = float(t_result.statistic)
            t_p_val = float(t_result.pvalue)
            if np.isnan(t_p_val):
                t_stat_val, t_p_val = 0.0, 1.0

        # Wilcoxon signed-rank test (handles constant nonzero diffs correctly)
        try:
            w_result = stats.wilcoxon(diff)
            w_stat_val = float(w_result.statistic)
            w_p_val = float(w_result.pvalue)
        except ValueError:
            # wilcoxon can fail if all differences have same sign and too few samples
            w_stat_val, w_p_val = 0.0, 1.0

    # Effect size
    d = cohen_d(results_a, results_b)

    # Mean difference and CI
    mean_diff = float(np.mean(diff))
    ci_lower, ci_upper = confidence_interval(diff.tolist(), confidence=0.95)

    return {
        "t_stat": t_stat_val,
        "t_p_value": t_p_val,
        "w_stat": w_stat_val,
        "w_p_value": w_p_val,
        "effect_size": d,
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def holm_bonferroni(
    p_values: list[float], alpha: float = 0.05
) -> list[dict[str, Any]]:
    """Apply Holm-Bonferroni step-down correction for multiple testing.

    Controls family-wise error rate by adjusting p-values using the
    Holm step-down procedure.

    Args:
        p_values: List of raw p-values from independent tests.
        alpha: Family-wise significance level (default 0.05).

    Returns:
        List of dicts (one per p-value, in original order) with:
        - original_p: The raw p-value
        - adjusted_p: Holm-adjusted p-value
        - rejected: Whether the null hypothesis is rejected
    """
    n = len(p_values)
    if n == 0:
        return []

    # Create index-sorted pairs
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    # Compute adjusted p-values (Holm step-down)
    results: list[dict[str, Any]] = [{}] * n
    cumulative_max = 0.0
    rejected_so_far = True

    for rank, (orig_idx, p) in enumerate(indexed):
        k = n - rank  # number of remaining hypotheses
        adjusted_p = min(p * k, 1.0)
        # Holm procedure: adjusted p-values must be monotonically non-decreasing
        adjusted_p = max(adjusted_p, cumulative_max)
        cumulative_max = adjusted_p

        # Once one is not rejected, all subsequent are also not rejected
        if not rejected_so_far or adjusted_p > alpha:
            rejected_so_far = False

        results[orig_idx] = {
            "original_p": p,
            "adjusted_p": adjusted_p,
            "rejected": rejected_so_far and adjusted_p <= alpha,
        }

    return results
