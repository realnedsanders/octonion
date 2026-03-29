"""Shared benchmark utilities for octonionic trie evaluation.

Provides sklearn baselines, trie classification, metrics computation,
plotting, and result I/O used by all benchmark scripts (Fashion-MNIST,
CIFAR-10, text classification).

Usage:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from trie_benchmark_utils import run_sklearn_baselines, run_trie_classifier, ...
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless rendering

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from octonion.trie import OctonionTrie


def run_sklearn_baselines(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Run standard sklearn classifiers on the given features.

    All classifiers operate on the SAME features (e.g., 8D PCA projections)
    for a fair comparison with the octonionic trie.

    Args:
        train_x: Training features, shape [n_train, n_features].
        train_y: Training labels, shape [n_train].
        test_x: Test features, shape [n_test, n_features].
        test_y: Test labels, shape [n_test].

    Returns:
        Dict mapping method name to result dict containing:
            - accuracy: float in [0, 1]
            - predictions: np.ndarray of predicted labels
            - confusion_matrix: list[list[int]] confusion matrix
            - classification_report: dict from sklearn classification_report
            - train_time: float seconds for fitting
            - test_time: float seconds for prediction
    """
    classifiers: dict[str, Any] = {
        "knn_k1": KNeighborsClassifier(n_neighbors=1),
        "knn_k5": KNeighborsClassifier(n_neighbors=5),
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
        "svm_rbf": SVC(kernel="rbf", random_state=42),
        "logreg": LogisticRegression(max_iter=1000, random_state=42),
    }

    results: dict[str, dict[str, Any]] = {}

    for name, clf in classifiers.items():
        t0 = time.time()
        clf.fit(train_x, train_y)
        train_time = time.time() - t0

        t0 = time.time()
        preds = clf.predict(test_x)
        test_time = time.time() - t0

        accuracy = float(np.mean(preds == test_y))
        cm = confusion_matrix(test_y, preds)
        report = classification_report(test_y, preds, output_dict=True, zero_division=0)

        results[name] = {
            "accuracy": accuracy,
            "predictions": preds,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "train_time": train_time,
            "test_time": test_time,
        }

    return results


def run_trie_classifier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    epochs: int = 3,
    assoc_threshold: float = 0.3,
    seed: int = 42,
) -> dict[str, Any]:
    """Classify data using the octonionic trie.

    Wraps OctonionTrie insert/query loop following the pattern from
    run_trie_mnist.py trie_classify function.

    Args:
        train_x: Training features as unit octonions, shape [n_train, 8].
        train_y: Training labels, shape [n_train].
        test_x: Test features as unit octonions, shape [n_test, 8].
        test_y: Test labels, shape [n_test].
        epochs: Number of training passes over the data.
        assoc_threshold: Associator threshold for trie branching.
        seed: Random seed for trie initialization.

    Returns:
        Dict containing:
            - accuracy: float in [0, 1]
            - predictions: list[int] of predicted labels
            - per_class: dict from compute_per_class_accuracy
            - trie_stats: dict with n_nodes, n_leaves, max_depth, etc.
            - train_time: float seconds
            - test_time: float seconds
    """
    trie = OctonionTrie(
        associator_threshold=assoc_threshold,
        similarity_threshold=0.1,
        max_depth=15,
        seed=seed,
    )

    # Train: insert all samples for each epoch, consolidate periodically
    t0 = time.time()
    for ep in range(epochs):
        for i in range(len(train_x)):
            label = train_y[i].item() if isinstance(train_y[i], torch.Tensor) else int(train_y[i])
            trie.insert(train_x[i], category=label)
        if ep % 2 == 1:
            trie.consolidate()
    trie.consolidate()
    train_time = time.time() - t0

    # Test: query each sample and collect predictions
    t0 = time.time()
    correct = 0
    predictions: list[int] = []

    for i in range(len(test_x)):
        label = test_y[i].item() if isinstance(test_y[i], torch.Tensor) else int(test_y[i])
        leaf = trie.query(test_x[i])
        pred = leaf.dominant_category
        predictions.append(pred if pred is not None else -1)

        if pred == label:
            correct += 1
    test_time = time.time() - t0

    accuracy = correct / len(test_y)
    trie_stats = trie.stats()

    # Build class names from unique labels
    unique_labels = sorted(set(int(y.item() if isinstance(y, torch.Tensor) else y) for y in test_y))
    class_names = [str(c) for c in unique_labels]

    y_true = np.array([int(y.item() if isinstance(y, torch.Tensor) else y) for y in test_y])
    y_pred = np.array(predictions)
    per_class = compute_per_class_accuracy(y_true, y_pred, class_names)

    return {
        "accuracy": accuracy,
        "predictions": predictions,
        "per_class": per_class,
        "trie_stats": trie_stats,
        "train_time": train_time,
        "test_time": test_time,
    }


def compute_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, dict[str, int | float]]:
    """Compute per-class accuracy statistics.

    Args:
        y_true: Ground truth labels, shape [n_samples].
        y_pred: Predicted labels, shape [n_samples].
        class_names: List of class name strings.

    Returns:
        Dict mapping class name to {correct: int, total: int, accuracy: float}.
    """
    result: dict[str, dict[str, int | float]] = {}

    # Map each class name to its label index (0, 1, 2, ...)
    # This follows the convention that class_names[i] corresponds to label i.
    for idx, name in enumerate(class_names):
        mask = y_true == idx
        total = int(mask.sum())
        correct = int(((y_true == y_pred) & mask).sum())
        accuracy = correct / total if total > 0 else 0.0

        result[name] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
        }

    return result


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: str | Path,
) -> None:
    """Plot and save a confusion matrix.

    Uses sklearn's ConfusionMatrixDisplay with a blue colormap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Display names for each class.
        title: Plot title.
        save_path: Path to save the PNG file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_learning_curves(
    curves: dict[str, list[dict[str, int | float]]],
    title: str,
    save_path: str | Path,
) -> None:
    """Plot accuracy vs training set size for multiple methods.

    Args:
        curves: Dict mapping method name to list of
            {"n_train": int, "accuracy": float} dicts.
        title: Plot title.
        save_path: Path to save the PNG file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for method_name, points in curves.items():
        n_train_vals = [p["n_train"] for p in points]
        acc_vals = [p["accuracy"] for p in points]
        ax.plot(n_train_vals, acc_vals, marker="o", label=method_name)

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


class _NumpyTorchEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and torch tensors."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return super().default(obj)


def save_results(results: dict[str, Any], output_path: Path) -> None:
    """Save results dict to JSON, handling numpy/torch types.

    Args:
        results: Results dictionary (may contain numpy arrays, torch tensors).
        output_path: Path to write the JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=_NumpyTorchEncoder)
