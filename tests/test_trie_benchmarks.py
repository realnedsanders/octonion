"""Tests for shared trie benchmark utilities.

Uses small synthetic data (50 train, 20 test, 8 features, 3 classes)
to validate output structure without relying on specific accuracy values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

# scripts/ is not a package, so insert it into sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from trie_benchmark_utils import (
    compute_per_class_accuracy,
    plot_confusion_matrix,
    run_sklearn_baselines,
    run_trie_classifier,
    save_results,
)


def _make_synthetic_data(
    n_train: int = 50,
    n_test: int = 20,
    n_features: int = 8,
    n_classes: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic unit-norm 8D vectors with random labels."""
    rng = np.random.default_rng(seed)

    train_x = rng.standard_normal((n_train, n_features))
    train_x /= np.linalg.norm(train_x, axis=1, keepdims=True)
    train_y = rng.integers(0, n_classes, size=n_train)

    test_x = rng.standard_normal((n_test, n_features))
    test_x /= np.linalg.norm(test_x, axis=1, keepdims=True)
    test_y = rng.integers(0, n_classes, size=n_test)

    return train_x, train_y, test_x, test_y


class TestRunSklearnBaselines:
    """Test 1: run_sklearn_baselines returns correct structure."""

    def test_returns_all_methods(self) -> None:
        train_x, train_y, test_x, test_y = _make_synthetic_data()
        results = run_sklearn_baselines(train_x, train_y, test_x, test_y)

        expected_keys = {"knn_k1", "knn_k5", "rf", "svm_rbf", "logreg"}
        assert set(results.keys()) == expected_keys

    def test_each_method_has_correct_fields(self) -> None:
        train_x, train_y, test_x, test_y = _make_synthetic_data()
        results = run_sklearn_baselines(train_x, train_y, test_x, test_y)

        for name, result in results.items():
            # accuracy is a float in [0, 1]
            assert isinstance(result["accuracy"], float), f"{name}: accuracy not float"
            assert 0.0 <= result["accuracy"] <= 1.0, f"{name}: accuracy out of range"

            # predictions is an ndarray with correct shape
            assert isinstance(result["predictions"], np.ndarray), f"{name}: predictions not array"
            assert result["predictions"].shape == (20,), f"{name}: predictions wrong shape"

            # confusion_matrix is a list of lists
            assert isinstance(result["confusion_matrix"], list), f"{name}: cm not list"
            assert all(isinstance(row, list) for row in result["confusion_matrix"]), (
                f"{name}: cm rows not lists"
            )

            # classification_report is a dict
            assert isinstance(result["classification_report"], dict), f"{name}: report not dict"

            # timing fields
            assert isinstance(result["train_time"], float), f"{name}: train_time not float"
            assert isinstance(result["test_time"], float), f"{name}: test_time not float"


class TestRunTrieClassifier:
    """Test 2: run_trie_classifier returns correct structure."""

    def test_returns_correct_fields(self) -> None:
        train_x, train_y, test_x, test_y = _make_synthetic_data()

        # Convert to torch tensors for trie
        train_xt = torch.from_numpy(train_x).to(torch.float64)
        train_yt = torch.from_numpy(train_y.astype(np.int64))
        test_xt = torch.from_numpy(test_x).to(torch.float64)
        test_yt = torch.from_numpy(test_y.astype(np.int64))

        result = run_trie_classifier(
            train_xt, train_yt, test_xt, test_yt, epochs=1, seed=42
        )

        # accuracy is a float in [0, 1]
        assert isinstance(result["accuracy"], float)
        assert 0.0 <= result["accuracy"] <= 1.0

        # predictions is a list of ints
        assert isinstance(result["predictions"], list)
        assert len(result["predictions"]) == 20

        # trie_stats is a dict with n_nodes key
        assert isinstance(result["trie_stats"], dict)
        assert "n_nodes" in result["trie_stats"]

        # per_class is a dict
        assert isinstance(result["per_class"], dict)

        # timing fields
        assert isinstance(result["train_time"], float)
        assert isinstance(result["test_time"], float)


class TestComputePerClassAccuracy:
    """Test 3: compute_per_class_accuracy returns correct counts."""

    def test_known_input(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 2])
        y_pred = np.array([0, 0, 1, 1, 1, 0])
        class_names = ["cat", "dog", "bird"]

        result = compute_per_class_accuracy(y_true, y_pred, class_names)

        # Class 0 ("cat"): 3 total, 2 correct
        assert result["cat"]["total"] == 3
        assert result["cat"]["correct"] == 2
        assert abs(result["cat"]["accuracy"] - 2 / 3) < 1e-10

        # Class 1 ("dog"): 2 total, 2 correct
        assert result["dog"]["total"] == 2
        assert result["dog"]["correct"] == 2
        assert result["dog"]["accuracy"] == 1.0

        # Class 2 ("bird"): 1 total, 0 correct
        assert result["bird"]["total"] == 1
        assert result["bird"]["correct"] == 0
        assert result["bird"]["accuracy"] == 0.0

    def test_empty_class(self) -> None:
        """A class with no samples should have total=0, accuracy=0.0."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        class_names = ["a", "b", "c"]

        result = compute_per_class_accuracy(y_true, y_pred, class_names)
        # Class "c" has no samples since label 2 never appears
        assert result["c"]["total"] == 0
        assert result["c"]["accuracy"] == 0.0


class TestSaveResults:
    """Test 4: save_results handles numpy arrays and torch tensors."""

    def test_numpy_and_torch_serialization(self, tmp_path: Path) -> None:
        results = {
            "accuracy": 0.95,
            "numpy_array": np.array([1, 2, 3]),
            "torch_tensor": torch.tensor([4.0, 5.0, 6.0]),
            "numpy_float": np.float64(0.123),
            "numpy_int": np.int64(42),
            "nested": {
                "array": np.array([[1, 2], [3, 4]]),
            },
        }

        output_path = tmp_path / "subdir" / "results.json"
        save_results(results, output_path)

        # File should exist
        assert output_path.exists()

        # Should be valid JSON
        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["accuracy"] == 0.95
        assert loaded["numpy_array"] == [1, 2, 3]
        assert loaded["torch_tensor"] == [4.0, 5.0, 6.0]
        assert abs(loaded["numpy_float"] - 0.123) < 1e-10
        assert loaded["numpy_int"] == 42
        assert loaded["nested"]["array"] == [[1, 2], [3, 4]]


class TestPlotConfusionMatrix:
    """Test 5: plot_confusion_matrix creates a .png file."""

    def test_creates_png_file(self, tmp_path: Path) -> None:
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        class_names = ["A", "B", "C"]

        save_path = tmp_path / "plots" / "confusion.png"
        plot_confusion_matrix(
            y_true, y_pred, class_names,
            title="Test Confusion Matrix",
            save_path=save_path,
        )

        assert save_path.exists()
        assert save_path.stat().st_size > 0
