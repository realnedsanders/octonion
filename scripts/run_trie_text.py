"""20 Newsgroups text classification benchmark for the octonionic trie.

Fully gradient-free pipeline: TF-IDF vectorization -> TruncatedSVD to 8D ->
normalize to unit octonions -> classify with trie and sklearn baselines.

No neural encoder is used anywhere. This is the strongest test of the
algebraic encoder thesis: zero gradient computation end-to-end.

Usage:
    docker compose run --rm dev uv run python scripts/run_trie_text.py --mode subset --epochs 1
    docker compose run --rm dev uv run python scripts/run_trie_text.py --mode both --epochs 3
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless rendering

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Import shared benchmark utilities
sys.path.insert(0, str(Path(__file__).parent))
from trie_benchmark_utils import (
    compute_per_class_accuracy,
    plot_confusion_matrix,
    plot_learning_curves,
    run_sklearn_baselines,
    run_trie_classifier,
    save_results,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 4-class subset: well-separated categories from different top-level groups
SUBSET_CATEGORIES = [
    "comp.graphics",
    "rec.sport.baseball",
    "sci.med",
    "talk.politics.guns",
]

# Shortened display names for the 20 full classes (for confusion matrix readability)
SHORT_NAMES_20 = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-win",
    "comp.sys.ibm",
    "comp.sys.mac",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.bball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.rel.christian",
    "talk.pol.guns",
    "talk.pol.mideast",
    "talk.pol.misc",
    "talk.rel.misc",
]


def load_20newsgroups(
    categories: list[str] | None = None,
    remove: tuple[str, ...] = ("headers", "footers", "quotes"),
) -> tuple[list[str], np.ndarray, list[str], np.ndarray, list[str]]:
    """Load 20 Newsgroups train/test splits.

    Args:
        categories: Subset of categories to load, or None for all 20.
        remove: Metadata to strip to prevent leakage.

    Returns:
        train_texts, train_labels, test_texts, test_labels, target_names
    """
    train = fetch_20newsgroups(
        subset="train", categories=categories, remove=remove, random_state=42
    )
    test = fetch_20newsgroups(
        subset="test", categories=categories, remove=remove, random_state=42
    )

    return (
        train.data,
        np.array(train.target),
        test.data,
        np.array(test.target),
        list(train.target_names),
    )


def tfidf_svd_pipeline(
    train_texts: list[str],
    test_texts: list[str],
    n_components: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, TfidfVectorizer, TruncatedSVD]:
    """TF-IDF vectorization + TruncatedSVD dimensionality reduction.

    Args:
        train_texts: Training documents.
        test_texts: Test documents.
        n_components: Number of SVD components.
        seed: Random seed for TruncatedSVD.

    Returns:
        train_reduced: [n_train, n_components] reduced features.
        test_reduced: [n_test, n_components] reduced features.
        explained_variance: Sum of explained variance ratios.
        vectorizer: Fitted TfidfVectorizer.
        svd: Fitted TruncatedSVD.
    """
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=10000,
        sublinear_tf=True,
        max_df=0.5,
        min_df=5,
        stop_words="english",
    )
    train_tfidf = vectorizer.fit_transform(train_texts)
    test_tfidf = vectorizer.transform(test_texts)

    # TruncatedSVD (NOT PCA -- works directly on sparse CSR matrices)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    train_reduced = svd.fit_transform(train_tfidf)
    test_reduced = svd.transform(test_tfidf)

    explained_variance = float(svd.explained_variance_ratio_.sum())

    return train_reduced, test_reduced, explained_variance, vectorizer, svd


def normalize_to_unit_octonions(features: np.ndarray) -> np.ndarray:
    """Normalize feature vectors to unit norm (unit octonions).

    Args:
        features: [n_samples, 8] feature vectors.

    Returns:
        [n_samples, 8] unit-norm vectors.
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    return features / norms


def compute_full_tfidf_upper_bound(
    train_texts: list[str],
    test_texts: list[str],
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, float]:
    """Compute LogReg on full TF-IDF features as upper bound.

    This shows how much information the 8D bottleneck loses vs. the
    full-dimensional TF-IDF representation. Replaces the CNN head upper
    bound used in image benchmarks (since there is no neural encoder here).

    Args:
        train_texts: Training documents.
        test_texts: Test documents.
        train_labels: Training labels.
        test_labels: Test labels.

    Returns:
        Dict with accuracy and timing info.
    """
    vectorizer = TfidfVectorizer(
        max_features=10000,
        sublinear_tf=True,
        max_df=0.5,
        min_df=5,
        stop_words="english",
    )
    train_tfidf = vectorizer.fit_transform(train_texts)
    test_tfidf = vectorizer.transform(test_texts)

    logger.info("  Training LogReg on full TF-IDF features...")
    t0 = time.time()
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_tfidf, train_labels)
    train_time = time.time() - t0

    t0 = time.time()
    preds = clf.predict(test_tfidf)
    test_time = time.time() - t0

    accuracy = float(np.mean(preds == test_labels))

    return {
        "accuracy": accuracy,
        "train_time": train_time,
        "test_time": test_time,
        "n_features": train_tfidf.shape[1],
    }


def run_learning_curves(
    train_texts: list[str],
    train_labels: np.ndarray,
    test_texts: list[str],
    test_labels: np.ndarray,
    n_components: int,
    epochs: int,
    seed: int,
    fractions: list[float] | None = None,
) -> dict[str, list[dict[str, int | float]]]:
    """Run accuracy vs. training-set size at specified fractions.

    Args:
        train_texts: Full training documents.
        train_labels: Full training labels.
        test_texts: Test documents.
        test_labels: Test labels.
        n_components: SVD components.
        epochs: Trie training epochs.
        seed: Random seed.
        fractions: List of training set fractions to evaluate.

    Returns:
        Dict mapping method name to list of {n_train, accuracy} dicts.
    """
    if fractions is None:
        fractions = [0.1, 0.25, 0.5, 1.0]

    curves: dict[str, list[dict[str, int | float]]] = {
        "trie": [],
        "knn_k5": [],
        "logreg": [],
    }

    n_total = len(train_texts)
    rng = np.random.RandomState(seed)

    for frac in fractions:
        n_sub = max(1, int(n_total * frac))
        idx = rng.permutation(n_total)[:n_sub]
        sub_texts = [train_texts[i] for i in idx]
        sub_labels = train_labels[idx]

        logger.info(f"  Learning curve: {frac:.0%} ({n_sub} samples)...")

        # Pipeline on subset
        sub_reduced, test_reduced, _, _, _ = tfidf_svd_pipeline(
            sub_texts, list(test_texts), n_components=n_components, seed=seed
        )

        sub_oct = normalize_to_unit_octonions(sub_reduced)
        test_oct = normalize_to_unit_octonions(test_reduced)

        # Trie
        trie_result = run_trie_classifier(
            torch.from_numpy(sub_oct),
            torch.from_numpy(sub_labels.astype(np.int64)),
            torch.from_numpy(test_oct),
            torch.from_numpy(test_labels.astype(np.int64)),
            epochs=epochs,
            seed=seed,
        )
        curves["trie"].append({"n_train": n_sub, "accuracy": trie_result["accuracy"]})

        # kNN k=5 on same features
        from sklearn.neighbors import KNeighborsClassifier

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(sub_reduced, sub_labels)
        knn_acc = float(np.mean(knn.predict(test_reduced) == test_labels))
        curves["knn_k5"].append({"n_train": n_sub, "accuracy": knn_acc})

        # LogReg on same features
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(sub_reduced, sub_labels)
        lr_acc = float(np.mean(lr.predict(test_reduced) == test_labels))
        curves["logreg"].append({"n_train": n_sub, "accuracy": lr_acc})

    return curves


def run_experiment(
    mode: str,
    n_components: int,
    epochs: int,
    seed: int,
    output_dir: Path,
) -> dict:
    """Run a single experiment (full 20-class or 4-class subset).

    Args:
        mode: "full" for all 20 classes, "subset" for 4-class subset.
        n_components: SVD components (should be 8).
        epochs: Trie training epochs.
        seed: Random seed.
        output_dir: Directory for output files.

    Returns:
        Results dict for this experiment.
    """
    categories = None if mode == "full" else SUBSET_CATEGORIES
    mode_label = "full_20class" if mode == "full" else "subset_4class"
    n_classes = 20 if mode == "full" else len(SUBSET_CATEGORIES)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  {mode_label}: {n_classes} classes")
    logger.info(f"{'=' * 60}")

    # Step 1: Load data
    logger.info("\n[1/10] Loading 20 Newsgroups data...")
    train_texts, train_labels, test_texts, test_labels, target_names = (
        load_20newsgroups(categories=categories)
    )
    logger.info(f"  Train: {len(train_texts)}, Test: {len(test_texts)}")
    logger.info(f"  Classes: {n_classes} ({', '.join(target_names[:5])}{'...' if len(target_names) > 5 else ''})")

    # Step 2: TF-IDF + TruncatedSVD
    logger.info("\n[2/10] TF-IDF vectorization + TruncatedSVD...")
    train_reduced, test_reduced, explained_var, _, _ = tfidf_svd_pipeline(
        train_texts, test_texts, n_components=n_components, seed=seed
    )
    logger.info(f"  Explained variance ({n_components}D): {explained_var:.4f} ({explained_var:.1%})")
    logger.info(f"  Train shape: {train_reduced.shape}, Test shape: {test_reduced.shape}")

    # Step 3: Full TF-IDF upper bound (LogReg on full features)
    logger.info("\n[3/10] Computing full TF-IDF LogReg upper bound...")
    full_tfidf_result = compute_full_tfidf_upper_bound(
        train_texts, test_texts, train_labels, test_labels
    )
    logger.info(f"  Full TF-IDF LogReg accuracy: {full_tfidf_result['accuracy']:.4f}")
    logger.info(f"  Full TF-IDF features: {full_tfidf_result['n_features']}")

    # Step 4: Normalize to unit octonions
    logger.info("\n[4/10] Normalizing to unit octonions...")
    train_oct = normalize_to_unit_octonions(train_reduced)
    test_oct = normalize_to_unit_octonions(test_reduced)
    train_norms = np.linalg.norm(train_oct, axis=1)
    n_zero = int((train_norms < 1e-6).sum())
    nonzero_mean = float(train_norms[train_norms >= 1e-6].mean()) if n_zero < len(train_norms) else 0.0
    logger.info(f"  Norm check (non-zero): mean={nonzero_mean:.6f}")
    if n_zero > 0:
        logger.info(f"  WARNING: {n_zero} empty documents (zero vectors after TF-IDF)")

    # Step 5: sklearn baselines on same 8D features
    logger.info("\n[5/10] Running sklearn baselines on 8D features...")
    baselines = run_sklearn_baselines(
        train_oct, train_labels, test_oct, test_labels
    )
    for name, res in baselines.items():
        logger.info(f"  {name}: {res['accuracy']:.4f} (train: {res['train_time']:.1f}s, test: {res['test_time']:.1f}s)")

    # Step 6: Trie classifier
    logger.info(f"\n[6/10] Running octonionic trie ({epochs} epochs)...")
    trie_result = run_trie_classifier(
        torch.from_numpy(train_oct),
        torch.from_numpy(train_labels.astype(np.int64)),
        torch.from_numpy(test_oct),
        torch.from_numpy(test_labels.astype(np.int64)),
        epochs=epochs,
        seed=seed,
    )
    logger.info(f"  Trie accuracy: {trie_result['accuracy']:.4f}")
    logger.info(f"  Train: {trie_result['train_time']:.1f}s, Test: {trie_result['test_time']:.1f}s")
    logger.info(f"  Nodes: {trie_result['trie_stats']['n_nodes']}, "
                f"Leaves: {trie_result['trie_stats']['n_leaves']}, "
                f"Max depth: {trie_result['trie_stats']['max_depth']}")

    # Step 7: Learning curves
    logger.info("\n[7/10] Computing learning curves...")
    curves = run_learning_curves(
        train_texts, train_labels,
        test_texts, test_labels,
        n_components=n_components,
        epochs=epochs,
        seed=seed,
    )

    # Step 8: Per-class accuracy
    logger.info("\n[8/10] Per-class accuracy (trie):")
    # Use short names for 20-class, target names for subset
    display_names = SHORT_NAMES_20 if mode == "full" else target_names
    trie_preds = np.array(trie_result["predictions"])
    per_class = compute_per_class_accuracy(
        test_labels, trie_preds, display_names
    )
    for name, stats in per_class.items():
        logger.info(f"  {name:25s}: {stats['correct']:4d}/{stats['total']:4d} = {stats['accuracy']:.3f}")

    # Step 9: Confusion matrix
    logger.info("\n[9/10] Generating confusion matrix...")
    figsize = (14, 12) if mode == "full" else (8, 6)
    cm_path = output_dir / f"confusion_matrix_{mode}.png"

    # Custom confusion matrix plot with appropriate figsize
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix, ConfusionMatrixDisplay

    cm = sk_confusion_matrix(test_labels, trie_preds)
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Octonionic Trie: {mode_label}")
    if mode == "full":
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(fontsize=7)
    plt.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {cm_path}")

    # Step 10: Print comparison table
    logger.info(f"\n[10/10] Comparison table ({mode_label}):")
    logger.info(f"  {'Method':25s} {'Accuracy':>10s} {'Train(s)':>10s} {'Test(s)':>10s}")
    logger.info(f"  {'-' * 55}")
    logger.info(f"  {'Full TF-IDF LogReg*':25s} {full_tfidf_result['accuracy']:10.4f} {full_tfidf_result['train_time']:10.1f} {full_tfidf_result['test_time']:10.1f}")
    for name, res in baselines.items():
        logger.info(f"  {name:25s} {res['accuracy']:10.4f} {res['train_time']:10.1f} {res['test_time']:10.1f}")
    logger.info(f"  {'trie':25s} {trie_result['accuracy']:10.4f} {trie_result['train_time']:10.1f} {trie_result['test_time']:10.1f}")
    logger.info(f"  * Full TF-IDF LogReg uses {full_tfidf_result['n_features']}D features (upper bound)")

    # Learning curve plot
    lc_path = output_dir / f"learning_curves_{mode}.png"
    plot_learning_curves(
        curves,
        title=f"Learning Curves: {mode_label}",
        save_path=lc_path,
    )
    logger.info(f"  Learning curves saved: {lc_path}")

    # Strip predictions from baselines for JSON serialization (they are numpy arrays)
    baselines_clean = {}
    for name, res in baselines.items():
        baselines_clean[name] = {
            k: v for k, v in res.items() if k != "predictions"
        }

    return {
        "n_train": len(train_texts),
        "n_test": len(test_texts),
        "n_classes": n_classes,
        "class_names": target_names,
        "explained_variance_8d": explained_var,
        "full_tfidf_logreg": full_tfidf_result,
        "trie": {
            k: v for k, v in trie_result.items() if k != "predictions"
        },
        "baselines": baselines_clean,
        "learning_curve": curves,
        "per_class_accuracy": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="20 Newsgroups Text Classification — Octonionic Trie Benchmark"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "subset", "both"],
        default="both",
        help="Run full 20-class, 4-class subset, or both (default: both)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of trie training epochs (default: 3)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=8,
        help="Number of TruncatedSVD components (default: 8)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/trie_benchmarks/text",
        help="Output directory for results (default: results/trie_benchmarks/text)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("20 Newsgroups — Octonionic Trie Text Benchmark")
    logger.info("Pipeline: TF-IDF -> TruncatedSVD -> Unit Octonions -> Trie")
    logger.info("  (Fully gradient-free: no neural encoder)")
    logger.info("=" * 60)
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  SVD components: {args.n_components}")
    logger.info(f"  Trie epochs: {args.epochs}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Output: {output_dir}")

    all_results: dict = {
        "pipeline": "TF-IDF + TruncatedSVD (fully gradient-free)",
        "n_components": args.n_components,
        "seed": args.seed,
        "epochs": args.epochs,
    }

    modes = []
    if args.mode in ("full", "both"):
        modes.append("full")
    if args.mode in ("subset", "both"):
        modes.append("subset")

    for mode in modes:
        result_key = "full_20class" if mode == "full" else "subset_4class"
        all_results[result_key] = run_experiment(
            mode=mode,
            n_components=args.n_components,
            epochs=args.epochs,
            seed=args.seed,
            output_dir=output_dir,
        )

    # Save combined results
    results_path = output_dir / "results.json"
    save_results(all_results, results_path)
    logger.info(f"\nResults saved to {results_path}")

    # Final summary
    logger.info(f"\n{'=' * 60}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'=' * 60}")
    for mode in modes:
        result_key = "full_20class" if mode == "full" else "subset_4class"
        r = all_results[result_key]
        logger.info(f"\n  {result_key} ({r['n_classes']} classes):")
        logger.info(f"    Explained variance ({args.n_components}D): {r['explained_variance_8d']:.4f}")
        logger.info(f"    Full TF-IDF LogReg (upper bound): {r['full_tfidf_logreg']['accuracy']:.4f}")
        logger.info(f"    Trie:     {r['trie']['accuracy']:.4f}")
        best_baseline_name = max(r["baselines"], key=lambda k: r["baselines"][k]["accuracy"])
        best_baseline_acc = r["baselines"][best_baseline_name]["accuracy"]
        logger.info(f"    Best baseline ({best_baseline_name}): {best_baseline_acc:.4f}")


if __name__ == "__main__":
    main()
