"""Fashion-MNIST benchmark for the octonionic trie.

Trains a small CNN encoder (same architecture as MNIST encoder), extracts 8D
features, and evaluates the octonionic trie plus all sklearn baselines.

Fashion-MNIST is the closest analog to MNIST (same image format, same number
of classes) but harder. This tests whether the trie generalizes beyond digit
recognition.

Usage:
    docker compose run --rm dev uv run python scripts/run_trie_fashion_mnist.py
    docker compose run --rm dev uv run python scripts/run_trie_fashion_mnist.py --n-train 10000
    docker compose run --rm dev uv run python scripts/run_trie_fashion_mnist.py --cnn-epochs 10 --epochs 3
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless rendering

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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

CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


class SmallCNN(nn.Module):
    """Small CNN encoder matching the MNIST encoder architecture.

    Architecture:
        Conv2d(1,16,3,pad=1) -> ReLU -> MaxPool2d(2) ->
        Conv2d(16,32,3,pad=1) -> ReLU -> MaxPool2d(2) ->
        Flatten -> Linear(32*7*7, 128) -> ReLU ->
        Linear(128, 8) -> ReLU  (feature extraction endpoint)

    The classifier head is Linear(8, 10) for evaluating CNN head accuracy
    as an upper bound on feature quality.
    """

    def __init__(self, feature_dim: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(feature_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass through features + classifier."""
        feat = self.features(x)
        return self.classifier(feat)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature_dim-dimensional features."""
        return self.features(x)


def train_cnn_encoder(
    n_train: int,
    cnn_epochs: int,
    seed: int,
) -> tuple[SmallCNN, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Train CNN encoder on Fashion-MNIST, return model and subsampled data.

    Trains on the full training set (60K), evaluates CNN head on test set,
    returns subsampled train/test splits for downstream use.

    Args:
        n_train: Number of training samples to subsample for trie/baselines.
        cnn_epochs: Number of CNN training epochs.
        seed: Random seed for reproducibility.

    Returns:
        cnn: Trained SmallCNN model.
        train_images: Subsampled training images [n_train, 1, 28, 28] float32.
        train_labels: Subsampled training labels [n_train].
        test_images: Full test images [n_test, 1, 28, 28] float32.
        test_labels: Full test labels [n_test].
        cnn_head_accuracy: CNN classification accuracy on test set.
    """
    from torchvision.datasets import FashionMNIST

    data_dir = tempfile.mkdtemp()
    train_ds = FashionMNIST(data_dir, train=True, download=True)
    test_ds = FashionMNIST(data_dir, train=False, download=True)

    # Full training data for CNN
    full_train_images = train_ds.data.unsqueeze(1).float() / 255.0
    full_train_labels = train_ds.targets

    # Full test data
    full_test_images = test_ds.data.unsqueeze(1).float() / 255.0
    full_test_labels = test_ds.targets

    # Train CNN on full training set
    cnn = SmallCNN(feature_dim=8)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(full_train_images, full_train_labels),
        batch_size=256,
        shuffle=True,
    )

    logger.info(f"\nTraining CNN encoder ({cnn_epochs} epochs on 60K samples)...")
    cnn.train()
    for epoch in range(cnn_epochs):
        total_loss = 0.0
        n_batches = 0
        for bx, by in loader:
            optimizer.zero_grad()
            out = cnn(bx)
            loss = loss_fn(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        logger.info(f"  Epoch {epoch + 1}/{cnn_epochs}: loss={avg_loss:.4f}")

    # Evaluate CNN head accuracy on full test set
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i in range(0, len(full_test_images), 1000):
            batch = full_test_images[i : i + 1000]
            labels = full_test_labels[i : i + 1000]
            preds = cnn(batch).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
        cnn_head_accuracy = correct / total
    logger.info(f"  CNN head accuracy (upper bound): {cnn_head_accuracy:.4f}")

    # Subsample training data
    gen = torch.Generator().manual_seed(seed)
    train_idx = torch.randperm(len(full_train_images), generator=gen)[:n_train]

    train_images = full_train_images[train_idx]
    train_labels = full_train_labels[train_idx]

    return cnn, train_images, train_labels, full_test_images, full_test_labels, cnn_head_accuracy


def extract_features(
    cnn: SmallCNN,
    train_images: torch.Tensor,
    test_images: torch.Tensor,
    n_test: int = 2000,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract 8D features from CNN, normalize to unit octonions.

    Args:
        cnn: Trained SmallCNN model.
        train_images: Training images [n_train, 1, 28, 28].
        test_images: Test images [n_test_full, 1, 28, 28].
        n_test: Unused, kept for API compatibility.
        seed: Unused, kept for API compatibility.

    Returns:
        train_oct: Training features as unit octonions [n_train, 8] float64.
        test_oct: Test features as unit octonions [n_test_full, 8] float64.
    """
    cnn.eval()
    with torch.no_grad():
        # Batch extraction for training features
        train_feats = []
        for i in range(0, len(train_images), 1000):
            train_feats.append(cnn.extract(train_images[i : i + 1000]))
        train_feats = torch.cat(train_feats).to(torch.float64)

        # Batch extraction for all test features
        test_feats = []
        for i in range(0, len(test_images), 1000):
            test_feats.append(cnn.extract(test_images[i : i + 1000]))
        test_feats = torch.cat(test_feats).to(torch.float64)

    # Normalize to unit octonions
    train_norms = train_feats.norm(dim=1, keepdim=True).clamp(min=1e-10)
    test_norms = test_feats.norm(dim=1, keepdim=True).clamp(min=1e-10)
    train_oct = train_feats / train_norms
    test_oct = test_feats / test_norms

    return train_oct, test_oct


def run_learning_curves(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    fractions: list[float],
    epochs: int,
    seed: int,
) -> dict[str, list[dict[str, int | float]]]:
    """Run learning curves for trie and kNN k=5 at various training set sizes.

    Args:
        train_x: Full training features [n_train, 8].
        train_y: Full training labels [n_train].
        test_x: Test features [n_test, 8].
        test_y: Test labels [n_test].
        fractions: List of training set fractions to evaluate.
        epochs: Number of trie training epochs.
        seed: Random seed.

    Returns:
        Dict mapping method name to list of {n_train, accuracy} dicts.
    """
    from sklearn.neighbors import KNeighborsClassifier

    curves: dict[str, list[dict[str, int | float]]] = {
        "Octonionic Trie": [],
        "kNN (k=5)": [],
    }

    n_total = len(train_x)
    gen = torch.Generator().manual_seed(seed)

    for frac in fractions:
        n = max(int(n_total * frac), 10)  # At least 10 samples
        idx = torch.randperm(n_total, generator=gen)[:n]

        sub_train_x = train_x[idx]
        sub_train_y = train_y[idx]

        logger.info(f"  Learning curve: n_train={n} ({frac:.0%})...")

        # Trie
        trie_result = run_trie_classifier(
            sub_train_x, sub_train_y, test_x, test_y,
            epochs=epochs, seed=seed,
        )
        curves["Octonionic Trie"].append({"n_train": n, "accuracy": trie_result["accuracy"]})

        # kNN k=5
        knn = KNeighborsClassifier(n_neighbors=min(5, n))
        knn.fit(sub_train_x.numpy(), sub_train_y.numpy())
        knn_preds = knn.predict(test_x.numpy())
        knn_acc = float(np.mean(knn_preds == test_y.numpy()))
        curves["kNN (k=5)"].append({"n_train": n, "accuracy": knn_acc})

    return curves


def print_comparison_table(
    results: dict,
    cnn_head_accuracy: float,
) -> None:
    """Print a formatted comparison table of all methods."""
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON TABLE")
    logger.info("=" * 60)
    logger.info(f"  {'Method':<20} | {'Accuracy':>8} | {'Train (s)':>10} | {'Test (s)':>9}")
    logger.info(f"  {'':->20}-+-{'':->8}-+-{'':->10}-+-{'':->9}")

    # CNN head (upper bound)
    logger.info(f"  {'CNN Head (bound)':.<20} | {cnn_head_accuracy:>8.4f} | {'N/A':>10} | {'N/A':>9}")

    # Trie
    if "trie" in results:
        t = results["trie"]
        logger.info(
            f"  {'Octonionic Trie':.<20} | {t['accuracy']:>8.4f} | "
            f"{t['train_time']:>10.2f} | {t['test_time']:>9.2f}"
        )

    # Baselines
    baseline_names = {
        "knn_k1": "kNN (k=1)",
        "knn_k5": "kNN (k=5)",
        "rf": "Random Forest",
        "svm_rbf": "SVM (RBF)",
        "logreg": "Logistic Reg",
    }
    if "baselines" in results:
        for key, display_name in baseline_names.items():
            if key in results["baselines"]:
                b = results["baselines"][key]
                logger.info(
                    f"  {display_name:.<20} | {b['accuracy']:>8.4f} | "
                    f"{b['train_time']:>10.2f} | {b['test_time']:>9.2f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fashion-MNIST Octonionic Trie Benchmark")
    parser.add_argument("--n-train", type=int, default=10000,
                        help="Number of training samples for trie/baselines (default: 10000)")
    parser.add_argument("--n-test", type=int, default=2000,
                        help="Number of test samples (default: 2000)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Trie training epochs (default: 3)")
    parser.add_argument("--cnn-epochs", type=int, default=10,
                        help="CNN encoder training epochs (default: 10)")
    parser.add_argument("--output-dir", type=str, default="results/trie_benchmarks/fashion_mnist",
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Fashion-MNIST Octonionic Trie Benchmark")
    logger.info("=" * 60)
    logger.info(f"  Train: {args.n_train}, Test: {args.n_test}")
    logger.info(f"  Trie epochs: {args.epochs}, CNN epochs: {args.cnn_epochs}")
    logger.info(f"  Seed: {args.seed}")

    # Step 1: Train CNN encoder on full Fashion-MNIST
    cnn, train_images, train_labels, test_images, test_labels, cnn_head_accuracy = (
        train_cnn_encoder(
            n_train=args.n_train,
            cnn_epochs=args.cnn_epochs,
            seed=args.seed,
        )
    )

    # Step 2: Extract 8D features from CNN
    logger.info("\nExtracting 8D CNN features...")
    train_oct, test_oct = extract_features(
        cnn, train_images, test_images, n_test=args.n_test, seed=args.seed,
    )

    # Subsample test set (train already subsampled in train_cnn_encoder)
    gen = torch.Generator().manual_seed(args.seed + 1)
    test_idx = torch.randperm(len(test_oct), generator=gen)[: args.n_test]
    train_x = train_oct
    train_y = train_labels
    test_x = test_oct[test_idx]
    test_y = test_labels[test_idx]

    logger.info(f"  Train features: {train_x.shape}, Test features: {test_x.shape}")

    # Step 3: Run sklearn baselines on same 8D CNN features
    logger.info("\nRunning sklearn baselines on 8D CNN features...")
    baseline_results = run_sklearn_baselines(
        train_x.numpy(), train_y.numpy(),
        test_x.numpy(), test_y.numpy(),
    )
    for name, res in baseline_results.items():
        logger.info(f"  {name}: {res['accuracy']:.4f}")

    # Step 4: Run octonionic trie
    logger.info(f"\nRunning octonionic trie ({args.epochs} epochs)...")
    trie_result = run_trie_classifier(
        train_x, train_y,
        test_x, test_y,
        epochs=args.epochs,
        seed=args.seed,
    )
    logger.info(f"  Trie accuracy: {trie_result['accuracy']:.4f}")
    logger.info(f"  Nodes: {trie_result['trie_stats']['n_nodes']}, "
                f"Leaves: {trie_result['trie_stats']['n_leaves']}, "
                f"Depth: {trie_result['trie_stats']['max_depth']}")

    # Step 5: Per-class accuracy for trie
    y_true = test_y.numpy()
    y_pred = np.array(trie_result["predictions"])
    per_class = compute_per_class_accuracy(y_true, y_pred, CLASSES)
    logger.info("\n  Per-class accuracy (trie):")
    for cls_name, stats in per_class.items():
        logger.info(f"    {cls_name:>15}: {stats['correct']:>3}/{stats['total']:>3} = {stats['accuracy']:.3f}")

    # Step 6: Confusion matrix
    logger.info("\nGenerating confusion matrix...")
    plot_confusion_matrix(
        y_true, y_pred, CLASSES,
        title="Fashion-MNIST: Octonionic Trie Confusion Matrix",
        save_path=output_dir / "confusion_matrix.png",
    )
    logger.info(f"  Saved to {output_dir}/confusion_matrix.png")

    # Step 7: Learning curves
    logger.info("\nGenerating learning curves...")
    curves = run_learning_curves(
        train_x, train_y,
        test_x, test_y,
        fractions=[0.1, 0.25, 0.5, 1.0],
        epochs=args.epochs,
        seed=args.seed,
    )
    plot_learning_curves(
        curves,
        title="Fashion-MNIST: Accuracy vs Training Set Size",
        save_path=output_dir / "learning_curve.png",
    )
    logger.info(f"  Saved to {output_dir}/learning_curve.png")

    # Step 8: Print comparison table
    all_results = {
        "trie": {
            "accuracy": trie_result["accuracy"],
            "train_time": trie_result["train_time"],
            "test_time": trie_result["test_time"],
            "per_class": per_class,
            "trie_stats": trie_result["trie_stats"],
        },
        "baselines": {
            name: {
                "accuracy": res["accuracy"],
                "train_time": res["train_time"],
                "test_time": res["test_time"],
            }
            for name, res in baseline_results.items()
        },
        "cnn_head": {
            "accuracy": cnn_head_accuracy,
        },
        "learning_curves": curves,
        "config": {
            "n_train": args.n_train,
            "n_test": args.n_test,
            "trie_epochs": args.epochs,
            "cnn_epochs": args.cnn_epochs,
            "seed": args.seed,
            "classes": CLASSES,
        },
    }

    print_comparison_table(all_results, cnn_head_accuracy)

    # Step 9: Save results
    save_results(all_results, output_dir / "results.json")
    logger.info(f"\n  Results saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
