"""Cross-benchmark summary for octonionic trie evaluation.

Loads results from all benchmark scripts (MNIST, Fashion-MNIST, CIFAR-10,
text) and produces:
  1. Cross-benchmark comparison table (trie vs all baselines)
  2. Trie vs kNN gap analysis
  3. Per-benchmark failure mode summary (best/worst classes)
  4. Trie structure comparison across benchmarks
  5. Aggregated summary.json

Usage:
    docker compose run --rm dev uv run python scripts/run_trie_benchmark_summary.py
    docker compose run --rm dev uv run python scripts/run_trie_benchmark_summary.py --results-dir results/trie_benchmarks
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Width constants for table formatting
BENCHMARK_COL = 22
ACC_COL = 8

# Gap threshold: trie must be within this many percentage points of kNN k=5
GAP_TARGET_PP = 5.0


def load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning None if missing."""
    if not path.exists():
        logger.warning(f"  [SKIP] {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def extract_mnist(data: dict[str, Any]) -> dict[str, Any] | None:
    """Extract standardized results from MNIST benchmark JSON."""
    if data is None:
        return None

    trie_acc = data.get("trie")
    if trie_acc is None:
        return None

    trie_stats = data.get("trie_stats", {})
    per_class = data.get("trie_per_digit", {})

    # Convert per_class to accuracy format
    per_class_acc: dict[str, float] = {}
    for digit, info in per_class.items():
        total = info.get("total", 0)
        correct = info.get("correct", 0)
        per_class_acc[str(digit)] = correct / total if total > 0 else 0.0

    return {
        "name": "MNIST (PCA-8D)",
        "trie": trie_acc,
        "knn_k1": data.get("knn_k1"),
        "knn_k5": data.get("knn_k5"),
        "rf": None,
        "svm_rbf": None,
        "logreg": None,
        "upper_bound": None,
        "upper_bound_label": None,
        "per_class": per_class_acc,
        "trie_stats": trie_stats,
        "config": data.get("config", {}),
    }


def extract_fashion_mnist(data: dict[str, Any]) -> dict[str, Any] | None:
    """Extract standardized results from Fashion-MNIST benchmark JSON."""
    if data is None:
        return None

    trie_data = data.get("trie", {})
    baselines = data.get("baselines", {})
    cnn_head = data.get("cnn_head", {})

    per_class_acc: dict[str, float] = {}
    for cls_name, info in trie_data.get("per_class", {}).items():
        per_class_acc[cls_name] = info.get("accuracy", 0.0)

    return {
        "name": "Fashion-MNIST",
        "trie": trie_data.get("accuracy"),
        "knn_k1": baselines.get("knn_k1", {}).get("accuracy"),
        "knn_k5": baselines.get("knn_k5", {}).get("accuracy"),
        "rf": baselines.get("rf", {}).get("accuracy"),
        "svm_rbf": baselines.get("svm_rbf", {}).get("accuracy"),
        "logreg": baselines.get("logreg", {}).get("accuracy"),
        "upper_bound": cnn_head.get("accuracy"),
        "upper_bound_label": "CNN Head",
        "per_class": per_class_acc,
        "trie_stats": trie_data.get("trie_stats", {}),
        "config": data.get("config", {}),
    }


def extract_cifar10(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract standardized results from CIFAR-10 benchmark JSON.

    Returns one entry per encoder. The 'best' encoder (highest trie accuracy)
    is used for the main comparison table.
    """
    if data is None:
        return []

    encoders = data.get("encoders", {})
    results: list[dict[str, Any]] = []

    for enc_name, enc_data in encoders.items():
        trie_data = enc_data.get("trie", {})
        baselines = enc_data.get("baselines", {})

        per_class_acc: dict[str, float] = {}
        for cls_name, info in trie_data.get("per_class", {}).items():
            per_class_acc[cls_name] = info.get("accuracy", 0.0)

        results.append({
            "name": f"CIFAR-10 ({enc_name})",
            "encoder": enc_name,
            "trie": trie_data.get("accuracy"),
            "knn_k1": baselines.get("knn_k1", {}).get("accuracy"),
            "knn_k5": baselines.get("knn_k5", {}).get("accuracy"),
            "rf": baselines.get("rf", {}).get("accuracy"),
            "svm_rbf": baselines.get("svm_rbf", {}).get("accuracy"),
            "logreg": baselines.get("logreg", {}).get("accuracy"),
            "upper_bound": enc_data.get("cnn_head_accuracy"),
            "upper_bound_label": "CNN Head",
            "per_class": per_class_acc,
            "trie_stats": trie_data.get("trie_stats", {}),
            "config": data.get("config", {}),
        })

    return results


def extract_text(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract standardized results from text benchmark JSON.

    Returns entries for subset_4class and full_20class if available.
    """
    if data is None:
        return []

    results: list[dict[str, Any]] = []

    for mode_key, label in [
        ("subset_4class", "Text (4 classes)"),
        ("full_20class", "Text (20 classes)"),
    ]:
        mode_data = data.get(mode_key)
        if mode_data is None:
            continue

        trie_data = mode_data.get("trie", {})
        baselines = mode_data.get("baselines", {})
        full_logreg = mode_data.get("full_tfidf_logreg", {})

        per_class_acc: dict[str, float] = {}
        for cls_name, info in trie_data.get("per_class", {}).items():
            per_class_acc[cls_name] = info.get("accuracy", 0.0)

        results.append({
            "name": label,
            "trie": trie_data.get("accuracy"),
            "knn_k1": baselines.get("knn_k1", {}).get("accuracy"),
            "knn_k5": baselines.get("knn_k5", {}).get("accuracy"),
            "rf": baselines.get("rf", {}).get("accuracy"),
            "svm_rbf": baselines.get("svm_rbf", {}).get("accuracy"),
            "logreg": baselines.get("logreg", {}).get("accuracy"),
            "upper_bound": full_logreg.get("accuracy"),
            "upper_bound_label": "Full TF-IDF LR",
            "per_class": per_class_acc,
            "trie_stats": trie_data.get("trie_stats", {}),
            "config": {
                "n_train": mode_data.get("n_train"),
                "n_test": mode_data.get("n_test"),
                "n_classes": mode_data.get("n_classes"),
            },
        })

    return results


def fmt_acc(acc: float | None) -> str:
    """Format accuracy as percentage string."""
    if acc is None:
        return "  --  "
    return f"{acc * 100:5.1f}%"


def print_comparison_table(benchmarks: list[dict[str, Any]]) -> None:
    """Print the cross-benchmark comparison table."""
    print()
    print("=" * 95)
    print("OCTONIONIC TRIE -- CROSS-BENCHMARK COMPARISON")
    print("=" * 95)

    # Header
    methods = ["Trie", "kNN-5", "kNN-1", "RF", "SVM", "LR", "Upper Bnd"]
    header = f"{'Benchmark':<{BENCHMARK_COL}}"
    for m in methods:
        header += f" | {m:>{ACC_COL}}"
    print(header)
    print("-" * len(header))

    # Rows
    for bm in benchmarks:
        row = f"{bm['name']:<{BENCHMARK_COL}}"
        row += f" | {fmt_acc(bm.get('trie')):>{ACC_COL}}"
        row += f" | {fmt_acc(bm.get('knn_k5')):>{ACC_COL}}"
        row += f" | {fmt_acc(bm.get('knn_k1')):>{ACC_COL}}"
        row += f" | {fmt_acc(bm.get('rf')):>{ACC_COL}}"
        row += f" | {fmt_acc(bm.get('svm_rbf')):>{ACC_COL}}"
        row += f" | {fmt_acc(bm.get('logreg')):>{ACC_COL}}"

        ub = bm.get("upper_bound")
        if ub is not None:
            ub_label = bm.get("upper_bound_label", "UB")
            row += f" | {fmt_acc(ub):>{ACC_COL}}"
        else:
            row += f" | {'  --  ':>{ACC_COL}}"

        print(row)

    print("-" * len(header))
    print(f"Target: Trie within {GAP_TARGET_PP:.0f}pp of kNN-5 on same features")
    print()


def print_gap_analysis(benchmarks: list[dict[str, Any]]) -> None:
    """Print trie vs kNN-5 gap analysis."""
    print("=" * 70)
    print("TRIE vs kNN-5 GAP ANALYSIS")
    print("=" * 70)

    header = f"{'Benchmark':<{BENCHMARK_COL}} | {'Trie':>7} | {'kNN-5':>7} | {'Gap':>7} | {'Status':>12}"
    print(header)
    print("-" * len(header))

    for bm in benchmarks:
        trie_acc = bm.get("trie")
        knn_acc = bm.get("knn_k5")

        if trie_acc is None or knn_acc is None:
            gap_str = "  N/A  "
            status = "INCOMPLETE"
        else:
            gap_pp = (trie_acc - knn_acc) * 100
            gap_str = f"{gap_pp:+5.1f}pp"
            if abs(gap_pp) <= GAP_TARGET_PP:
                status = "ON TARGET"
            elif gap_pp < -GAP_TARGET_PP:
                status = "BELOW TARGET"
            else:
                status = "ABOVE TARGET"

        print(
            f"{bm['name']:<{BENCHMARK_COL}} | {fmt_acc(trie_acc):>7} | "
            f"{fmt_acc(knn_acc):>7} | {gap_str:>7} | {status:>12}"
        )

    print()


def print_failure_mode_summary(benchmarks: list[dict[str, Any]]) -> None:
    """Print per-benchmark best/worst classes for the trie."""
    print("=" * 70)
    print("PER-BENCHMARK FAILURE MODE SUMMARY (Trie)")
    print("=" * 70)

    for bm in benchmarks:
        per_class = bm.get("per_class", {})
        if not per_class:
            continue

        # Sort by accuracy
        sorted_classes = sorted(per_class.items(), key=lambda x: x[1])
        n_show = min(3, len(sorted_classes))

        worst = sorted_classes[:n_show]
        best = sorted_classes[-n_show:]

        print(f"\n  {bm['name']}")
        print(f"  {'Worst classes:':<16}", end="")
        for cls_name, acc in worst:
            print(f"  {cls_name} ({acc * 100:.1f}%)", end="")
        print()
        print(f"  {'Best classes:':<16}", end="")
        for cls_name, acc in reversed(best):
            print(f"  {cls_name} ({acc * 100:.1f}%)", end="")
        print()

    print()


def print_trie_structure(benchmarks: list[dict[str, Any]]) -> None:
    """Print trie structure comparison across benchmarks."""
    print("=" * 80)
    print("TRIE STRUCTURE COMPARISON")
    print("=" * 80)

    header = (
        f"{'Benchmark':<{BENCHMARK_COL}} | {'Nodes':>7} | {'Leaves':>7} | "
        f"{'Max Depth':>9} | {'Rumin. Rej.':>11} | {'Consol.':>7}"
    )
    print(header)
    print("-" * len(header))

    for bm in benchmarks:
        stats = bm.get("trie_stats", {})
        if not stats:
            continue

        n_nodes = stats.get("n_nodes", "?")
        n_leaves = stats.get("n_leaves", "?")
        max_depth = stats.get("max_depth", "?")
        rum_rej = stats.get("rumination_rejections", "?")
        consol = stats.get("consolidation_merges", "?")

        print(
            f"{bm['name']:<{BENCHMARK_COL}} | {n_nodes:>7} | {n_leaves:>7} | "
            f"{max_depth:>9} | {rum_rej:>11} | {consol:>7}"
        )

    print()


def build_summary_json(benchmarks: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the aggregated summary JSON."""
    comparison_rows: list[dict[str, Any]] = []
    gap_analysis: list[dict[str, Any]] = []
    structure_comparison: list[dict[str, Any]] = []

    for bm in benchmarks:
        trie_acc = bm.get("trie")
        knn_acc = bm.get("knn_k5")

        row = {
            "benchmark": bm["name"],
            "trie": trie_acc,
            "knn_k5": knn_acc,
            "knn_k1": bm.get("knn_k1"),
            "rf": bm.get("rf"),
            "svm_rbf": bm.get("svm_rbf"),
            "logreg": bm.get("logreg"),
            "upper_bound": bm.get("upper_bound"),
            "upper_bound_label": bm.get("upper_bound_label"),
        }
        comparison_rows.append(row)

        # Gap analysis
        if trie_acc is not None and knn_acc is not None:
            gap_pp = (trie_acc - knn_acc) * 100
            gap_analysis.append({
                "benchmark": bm["name"],
                "trie_accuracy": trie_acc,
                "knn_k5_accuracy": knn_acc,
                "gap_pp": round(gap_pp, 2),
                "within_target": abs(gap_pp) <= GAP_TARGET_PP,
            })

        # Structure
        stats = bm.get("trie_stats", {})
        if stats:
            structure_comparison.append({
                "benchmark": bm["name"],
                **stats,
            })

    # Failure modes
    failure_modes: list[dict[str, Any]] = []
    for bm in benchmarks:
        per_class = bm.get("per_class", {})
        if not per_class:
            continue
        sorted_classes = sorted(per_class.items(), key=lambda x: x[1])
        n_show = min(3, len(sorted_classes))
        failure_modes.append({
            "benchmark": bm["name"],
            "worst_classes": [
                {"class": c, "accuracy": a} for c, a in sorted_classes[:n_show]
            ],
            "best_classes": [
                {"class": c, "accuracy": a}
                for c, a in reversed(sorted_classes[-n_show:])
            ],
        })

    # Overall verdict
    on_target = sum(1 for g in gap_analysis if g["within_target"])
    total = len(gap_analysis)

    return {
        "benchmarks": comparison_rows,
        "gap_analysis": gap_analysis,
        "failure_modes": failure_modes,
        "trie_structure": structure_comparison,
        "verdict": {
            "target": f"Trie within {GAP_TARGET_PP:.0f}pp of kNN-5",
            "on_target_count": on_target,
            "total_benchmarks": total,
            "summary": (
                f"{on_target}/{total} benchmarks meet target"
                if total > 0
                else "No benchmarks with complete data"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-benchmark summary for octonionic trie evaluation"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/trie_benchmarks",
        help="Directory containing benchmark result subdirectories",
    )
    parser.add_argument(
        "--mnist-results",
        type=str,
        default="results/trie_validation/mnist_benchmark.json",
        help="Path to MNIST baseline results JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/trie_benchmarks",
        help="Directory for summary output",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("OCTONIONIC TRIE -- BENCHMARK SUMMARY")
    print("=" * 70)
    print()
    print("Loading results from:", results_dir)
    print()

    # ---- Load all results ----

    # 1. MNIST baseline
    mnist_path = Path(args.mnist_results)
    print(f"  Loading MNIST:         {mnist_path}")
    mnist_raw = load_json(mnist_path)
    mnist = extract_mnist(mnist_raw)

    # 2. Fashion-MNIST
    fmnist_path = results_dir / "fashion_mnist" / "results.json"
    print(f"  Loading Fashion-MNIST: {fmnist_path}")
    fmnist_raw = load_json(fmnist_path)
    fmnist = extract_fashion_mnist(fmnist_raw)

    # 3. CIFAR-10
    cifar_path = results_dir / "cifar10" / "results.json"
    print(f"  Loading CIFAR-10:      {cifar_path}")
    cifar_raw = load_json(cifar_path)
    cifar_entries = extract_cifar10(cifar_raw)

    # 4. Text
    text_path = results_dir / "text" / "results.json"
    print(f"  Loading Text:          {text_path}")
    text_raw = load_json(text_path)
    text_entries = extract_text(text_raw)

    print()

    # ---- Assemble benchmark list ----
    benchmarks: list[dict[str, Any]] = []

    if mnist is not None:
        benchmarks.append(mnist)
    if fmnist is not None:
        benchmarks.append(fmnist)

    # For CIFAR-10, include all encoders in detail tables and best in main table
    if cifar_entries:
        # Pick the encoder with highest trie accuracy for main comparison
        best_cifar = max(cifar_entries, key=lambda x: x.get("trie") or 0)
        benchmarks.append(best_cifar)

    for entry in text_entries:
        benchmarks.append(entry)

    if not benchmarks:
        logger.error("No benchmark results found. Run individual benchmarks first.")
        sys.exit(1)

    n_loaded = len(benchmarks)
    print(f"Loaded {n_loaded} benchmark(s)")
    print()

    # ---- Print all output tables ----

    # Output 1: Cross-benchmark comparison table
    print_comparison_table(benchmarks)

    # If multiple CIFAR encoders, show extended encoder table
    if len(cifar_entries) > 1:
        print("=" * 70)
        print("CIFAR-10 ENCODER COMPARISON (all encoders)")
        print("=" * 70)
        print_comparison_table(cifar_entries)

    # Output 2: Gap analysis
    print_gap_analysis(benchmarks)

    # Output 3: Failure mode summary
    print_failure_mode_summary(benchmarks)

    # Output 4: Trie structure comparison
    print_trie_structure(benchmarks)

    # ---- Build aggregated results ----
    summary = build_summary_json(benchmarks)

    # ---- Final verdict ----
    verdict = summary["verdict"]
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Target:  {verdict['target']}")
    print(f"  Result:  {verdict['summary']}")
    print()

    # List benchmarks that miss target
    for gap in summary["gap_analysis"]:
        if not gap["within_target"]:
            print(
                f"  [BELOW TARGET] {gap['benchmark']}: "
                f"gap = {gap['gap_pp']:+.1f}pp"
            )

    print()

    # ---- Save aggregated results ----
    output_path = output_dir / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved aggregated results to {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
