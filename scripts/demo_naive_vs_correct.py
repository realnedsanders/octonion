"""Standalone demo: naive vs correct octonionic chain rule.

Demonstrates that ignoring parenthesization when computing gradients
through chains of octonion multiplications produces incorrect results.
The "naive" approach treats multiplication as associative (always left-to-right),
while the "correct" approach respects the actual parenthesization.

Usage:
    docker compose run --rm dev uv run python scripts/demo_naive_vs_correct.py

Outputs:
    - Summary statistics to stdout
    - Detailed results to results/naive_vs_correct.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from octonion.calculus._chain_rule import compose_jacobians, naive_chain_rule_jacobian
from octonion.calculus._composition import (
    Leaf,
    Node,
    all_parenthesizations,
)
from octonion.calculus._inspector import tree_to_string


def compute_gradient_difference(
    operands: list[torch.Tensor],
    tree: Leaf | Node,
) -> dict:
    """Compute difference between correct (tree-aware) and naive gradients."""
    correct_jacs = compose_jacobians(tree, operands)
    naive_jacs = naive_chain_rule_jacobian(operands)

    n = len(operands)
    per_operand: list[dict] = []
    total_diff_norm = 0.0
    total_correct_norm = 0.0

    for i in range(n):
        diff = correct_jacs[i] - naive_jacs[i]
        diff_norm = torch.norm(diff).item()
        correct_norm = torch.norm(correct_jacs[i]).item()
        naive_norm = torch.norm(naive_jacs[i]).item()

        # Direction cosine similarity between correct and naive
        if correct_norm > 1e-15 and naive_norm > 1e-15:
            cos_sim = (
                torch.sum(correct_jacs[i] * naive_jacs[i]).item()
                / (correct_norm * naive_norm)
            )
        else:
            cos_sim = 1.0

        # Per-component divergence
        component_div = torch.abs(diff).max().item()

        per_operand.append(
            {
                "operand_idx": i,
                "diff_norm": diff_norm,
                "correct_norm": correct_norm,
                "naive_norm": naive_norm,
                "relative_error": diff_norm / (correct_norm + 1e-15),
                "cosine_similarity": cos_sim,
                "max_component_divergence": component_div,
            }
        )
        total_diff_norm += diff_norm**2
        total_correct_norm += correct_norm**2

    total_diff_norm = total_diff_norm**0.5
    total_correct_norm = total_correct_norm**0.5

    return {
        "tree": tree_to_string(tree),
        "per_operand": per_operand,
        "total_diff_norm": total_diff_norm,
        "total_relative_error": total_diff_norm / (total_correct_norm + 1e-15),
    }


def run_statistical_analysis(
    n_operands: int, n_trials: int = 1000, seed: int = 42
) -> dict:
    """Run n_trials random inputs and collect statistics."""
    trees = all_parenthesizations(n_operands)
    # The fully left-associated tree is trees[-1] (naive == correct for it).
    # Test all other trees which should show differences.
    left_tree_str = tree_to_string(trees[-1])
    non_left_trees = [t for t in trees if tree_to_string(t) != left_tree_str]

    results_per_tree: list[dict] = []

    for tree in non_left_trees:
        tree_str = tree_to_string(tree)
        diffs: list[float] = []
        rel_errors: list[float] = []
        cos_sims: list[float] = []

        for trial in range(n_trials):
            torch.manual_seed(seed + trial)
            operands = [
                torch.randn(8, dtype=torch.float64) * 0.5 for _ in range(n_operands)
            ]

            result = compute_gradient_difference(operands, tree)
            diffs.append(result["total_diff_norm"])
            rel_errors.append(result["total_relative_error"])

            # Average cosine similarity across operands
            avg_cos = sum(
                op["cosine_similarity"] for op in result["per_operand"]
            ) / len(result["per_operand"])
            cos_sims.append(avg_cos)

        # Compute statistics
        mean_diff = sum(diffs) / len(diffs)
        std_diff = (sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)) ** 0.5
        ci_95 = 1.96 * std_diff / (len(diffs) ** 0.5)

        mean_rel = sum(rel_errors) / len(rel_errors)
        mean_cos = sum(cos_sims) / len(cos_sims)

        results_per_tree.append(
            {
                "tree": tree_str,
                "n_trials": n_trials,
                "mean_diff_norm": mean_diff,
                "std_diff_norm": std_diff,
                "ci_95": ci_95,
                "ci_95_lower": mean_diff - ci_95,
                "ci_95_upper": mean_diff + ci_95,
                "mean_relative_error": mean_rel,
                "mean_cosine_similarity": mean_cos,
                "min_diff": min(diffs),
                "max_diff": max(diffs),
            }
        )

    return {
        "n_operands": n_operands,
        "n_parenthesizations_tested": len(non_left_trees),
        "n_trials": n_trials,
        "per_tree": results_per_tree,
    }


def run_depth_scaling(depths: list[int], n_trials: int = 100, seed: int = 42) -> dict:
    """Measure how naive-vs-correct error scales with chain depth."""
    depth_results: list[dict] = []

    for depth in depths:
        trees = all_parenthesizations(depth)
        if len(trees) < 2:
            continue

        # Use the fully right-associated tree (first one) for maximum difference
        # (all_parenthesizations generates right-to-left first)
        right_tree = trees[0]

        diffs: list[float] = []
        for trial in range(n_trials):
            torch.manual_seed(seed + trial)
            operands = [
                torch.randn(8, dtype=torch.float64) * 0.5 for _ in range(depth)
            ]
            result = compute_gradient_difference(operands, right_tree)
            diffs.append(result["total_diff_norm"])

        mean_diff = sum(diffs) / len(diffs)
        std_diff = (sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)) ** 0.5

        depth_results.append(
            {
                "depth": depth,
                "n_parenthesizations": len(trees),
                "tree_tested": tree_to_string(right_tree),
                "mean_diff_norm": mean_diff,
                "std_diff_norm": std_diff,
                "min_diff": min(diffs),
                "max_diff": max(diffs),
            }
        )

    return {
        "depths": depths,
        "scaling_results": depth_results,
    }


def main() -> None:
    """Run full naive-vs-correct analysis and save results."""
    start = time.time()

    print("=" * 60)
    print("Naive vs Correct Octonionic Chain Rule Analysis")
    print("=" * 60)

    # 1. Statistical analysis with 1000 random inputs for 3-operand chains
    print("\n--- Statistical Analysis (3 operands, 1000 trials) ---")
    stats_3 = run_statistical_analysis(3, n_trials=1000)
    for r in stats_3["per_tree"]:
        print(f"  Tree: {r['tree']}")
        print(
            f"    Mean diff: {r['mean_diff_norm']:.6f} "
            f"+/- {r['ci_95']:.6f} (95% CI)"
        )
        print(f"    Mean relative error: {r['mean_relative_error']:.6f}")
        print(f"    Mean cosine similarity: {r['mean_cosine_similarity']:.6f}")

    # 2. Depth scaling analysis
    print("\n--- Depth Scaling Analysis ---")
    depths = [2, 3, 5, 7]
    # depth 10 has C_9 = 4862 trees, too many to enumerate -- just use 2,3,5,7
    scaling = run_depth_scaling(depths, n_trials=100)
    for r in scaling["scaling_results"]:
        print(
            f"  Depth {r['depth']}: mean diff = {r['mean_diff_norm']:.6f} "
            f"(+/- {r['std_diff_norm']:.6f}), "
            f"{r['n_parenthesizations']} parenthesizations"
        )

    # 3. Single detailed example
    print("\n--- Detailed Example (3 operands) ---")
    torch.manual_seed(42)
    operands = [torch.randn(8, dtype=torch.float64) * 0.5 for _ in range(3)]
    right_tree = Node("mul", Leaf(0), Node("mul", Leaf(1), Leaf(2)))
    detail = compute_gradient_difference(operands, right_tree)
    for op in detail["per_operand"]:
        print(
            f"  Operand {op['operand_idx']}: "
            f"diff_norm={op['diff_norm']:.6f}, "
            f"rel_error={op['relative_error']:.6f}, "
            f"cos_sim={op['cosine_similarity']:.6f}"
        )

    elapsed = time.time() - start

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    output = {
        "stats_3_operands": stats_3,
        "depth_scaling": scaling,
        "detailed_example": detail,
        "elapsed_seconds": elapsed,
    }
    output_path = os.path.join(results_dir, "naive_vs_correct.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"Total time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
