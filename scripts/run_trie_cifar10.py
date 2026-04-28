"""CIFAR-10 benchmark for the octonionic trie with multi-encoder comparison.

Projects CIFAR-10 images to octonionic space via CNN encoders of varying
capacity (2-layer, 4-layer, ResNet-8-scale), then classifies using the
self-organizing octonionic trie. Compares against sklearn baselines on the
same 8D features.

The key question: how does encoder capacity affect trie classification
accuracy on complex color images?

Usage:
    docker compose run --rm dev uv run python scripts/run_trie_cifar10.py
    docker compose run --rm dev uv run python scripts/run_trie_cifar10.py --encoder 2layer
    docker compose run --rm dev uv run python scripts/run_trie_cifar10.py --encoder all --n-train 10000
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

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
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# ─── Encoder Architectures ───────────────────────────────────────────


class CIFAR_CNN_2Layer(nn.Module):
    """Minimal 2-layer CNN encoder for CIFAR-10."""

    def __init__(self, feature_dim: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, feature_dim),
        )
        self.classifier = nn.Linear(feature_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        return self.classifier(feats)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class CIFAR_CNN_4Layer(nn.Module):
    """Medium 4-layer CNN encoder for CIFAR-10."""

    def __init__(self, feature_dim: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, feature_dim),
        )
        self.classifier = nn.Linear(feature_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        return self.classifier(feats)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class ResidualBlock(nn.Module):
    """Standard residual block with optional downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class CIFAR_CNN_ResNet8(nn.Module):
    """Simplified ResNet with ~8 conv layers for CIFAR-10."""

    def __init__(self, feature_dim: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.block1 = ResidualBlock(16, 16)
        self.block2 = ResidualBlock(16, 32, stride=2)
        self.block3 = ResidualBlock(32, 64, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc_features = nn.Linear(64, feature_dim)
        self.classifier = nn.Linear(feature_dim, 10)

    def features_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.fc_features(self.flatten(self.pool(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features_forward(x))

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        return self.features_forward(x)


# ─── Encoder Registry ────────────────────────────────────────────────

ENCODER_CONFIGS: dict[str, dict] = {
    "2layer": {"cls": CIFAR_CNN_2Layer, "epochs": 20, "lr": 1e-3, "scheduler": None},
    "4layer": {"cls": CIFAR_CNN_4Layer, "epochs": 30, "lr": 1e-3, "scheduler": None},
    "resnet8": {
        "cls": CIFAR_CNN_ResNet8,
        "epochs": 50,
        "lr": 1e-3,
        "scheduler": "cosine",
    },
}


# ─── Data Loading ─────────────────────────────────────────────────────

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_loaders(
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader, CIFAR10, CIFAR10]:
    """Load CIFAR-10 with standard augmentation for training.

    Returns:
        train_loader: DataLoader with augmentation for CNN training.
        test_loader: DataLoader without augmentation for evaluation.
        train_dataset: Raw training dataset (for feature extraction without augmentation).
        test_dataset: Raw test dataset.
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ]
    )

    train_dataset = CIFAR10(
        root="/tmp/cifar10_data", train=True, download=True, transform=train_transform
    )
    test_dataset = CIFAR10(
        root="/tmp/cifar10_data", train=False, download=True, transform=test_transform
    )

    # Non-augmented version for feature extraction
    train_dataset_eval = CIFAR10(
        root="/tmp/cifar10_data", train=True, download=False, transform=test_transform
    )
    test_dataset_eval = CIFAR10(
        root="/tmp/cifar10_data", train=False, download=False, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        multiprocessing_context="spawn",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        multiprocessing_context="spawn",
    )

    return train_loader, test_loader, train_dataset_eval, test_dataset_eval


# ─── CNN Training ─────────────────────────────────────────────────────


def train_cnn_encoder(
    encoder_name: str,
    device: torch.device,
    cnn_epochs: int | None = None,
) -> tuple[nn.Module, float]:
    """Train a CNN encoder on CIFAR-10 and return it with test accuracy.

    Args:
        encoder_name: One of '2layer', '4layer', 'resnet8'.
        device: Torch device to train on.
        cnn_epochs: Override default epoch count (for quick testing).

    Returns:
        Trained model (on device, in eval mode) and test accuracy.
    """
    config = ENCODER_CONFIGS[encoder_name]
    model = config["cls"](feature_dim=8).to(device)
    epochs = cnn_epochs if cnn_epochs is not None else config["epochs"]
    lr = config["lr"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if config["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader, test_loader, _, _ = get_cifar10_loaders()

    criterion = nn.CrossEntropyLoss()

    logger.info(f"  Training {encoder_name} CNN for {epochs} epochs...")
    model.train()
    for ep in range(epochs):
        running_loss = 0.0
        n_batches = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        if scheduler is not None:
            scheduler.step()
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == epochs - 1:
            avg_loss = running_loss / max(n_batches, 1)
            logger.info(f"    Epoch {ep + 1}/{epochs}: loss={avg_loss:.4f}")

    # Evaluate CNN head accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    cnn_accuracy = correct / total
    logger.info(f"  {encoder_name} CNN head accuracy: {cnn_accuracy:.4f}")

    return model, cnn_accuracy


# ─── Feature Extraction ──────────────────────────────────────────────


def extract_features(
    model: nn.Module,
    n_train: int,
    n_test: int,
    device: torch.device,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract 8D features from trained CNN and normalize to unit octonions.

    Uses non-augmented transforms for deterministic feature extraction.
    Subsamples from the full 50K/10K datasets.

    Returns:
        train_x, train_y, test_x, test_y as numpy arrays.
    """
    _, _, train_dataset_eval, test_dataset_eval = get_cifar10_loaders()

    # Build non-augmented loaders for extraction
    train_eval_loader = DataLoader(
        train_dataset_eval,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        multiprocessing_context="spawn",
    )
    test_eval_loader = DataLoader(
        test_dataset_eval,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        multiprocessing_context="spawn",
    )

    model.eval()

    # Extract all features
    all_train_feats = []
    all_train_labels = []
    all_test_feats = []
    all_test_labels = []

    with torch.no_grad():
        for images, labels in train_eval_loader:
            images = images.to(device)
            feats = model.extract(images).cpu()
            all_train_feats.append(feats)
            all_train_labels.append(labels)
        for images, labels in test_eval_loader:
            images = images.to(device)
            feats = model.extract(images).cpu()
            all_test_feats.append(feats)
            all_test_labels.append(labels)

    train_feats = torch.cat(all_train_feats, dim=0)
    train_labels = torch.cat(all_train_labels, dim=0)
    test_feats = torch.cat(all_test_feats, dim=0)
    test_labels = torch.cat(all_test_labels, dim=0)

    # Subsample with seeded randperm
    gen = torch.Generator().manual_seed(seed)
    train_idx = torch.randperm(len(train_feats), generator=gen)[:n_train]
    test_idx = torch.randperm(len(test_feats), generator=gen)[:n_test]

    train_x = train_feats[train_idx].to(torch.float64)
    train_y = train_labels[train_idx]
    test_x = test_feats[test_idx].to(torch.float64)
    test_y = test_labels[test_idx]

    # Normalize to unit octonions
    train_norms = train_x.norm(dim=1, keepdim=True).clamp(min=1e-10)
    test_norms = test_x.norm(dim=1, keepdim=True).clamp(min=1e-10)
    train_x = train_x / train_norms
    test_x = test_x / test_norms

    return (
        train_x.numpy(),
        train_y.numpy(),
        test_x.numpy(),
        test_y.numpy(),
    )


# ─── Encoder Evaluation ──────────────────────────────────────────────


def evaluate_encoder(
    encoder_name: str,
    model: nn.Module,
    cnn_accuracy: float,
    n_train: int,
    n_test: int,
    trie_epochs: int,
    output_dir: Path,
    device: torch.device,
    seed: int = 42,
) -> dict:
    """Run full evaluation for a single encoder: baselines, trie, learning curves.

    Returns:
        Results dict for this encoder.
    """
    logger.info(f"\n  Extracting features with {encoder_name}...")
    train_x, train_y, test_x, test_y = extract_features(
        model, n_train, n_test, device, seed
    )
    logger.info(f"  Features: train={train_x.shape}, test={test_x.shape}")

    # Run sklearn baselines
    logger.info("  Running sklearn baselines...")
    baseline_results = run_sklearn_baselines(train_x, train_y, test_x, test_y)

    # Print baseline accuracies
    for name, res in baseline_results.items():
        logger.info(f"    {name}: {res['accuracy']:.4f}")

    # Run trie classifier
    logger.info(f"  Running octonionic trie ({trie_epochs} epochs)...")
    trie_result = run_trie_classifier(
        torch.from_numpy(train_x),
        torch.from_numpy(train_y.astype(np.int64)),
        torch.from_numpy(test_x),
        torch.from_numpy(test_y.astype(np.int64)),
        epochs=trie_epochs,
        seed=seed,
    )
    logger.info(f"    Trie accuracy: {trie_result['accuracy']:.4f}")
    logger.info(
        f"    Trie stats: {trie_result['trie_stats']['n_nodes']} nodes, "
        f"{trie_result['trie_stats']['n_leaves']} leaves, "
        f"depth={trie_result['trie_stats']['max_depth']}"
    )

    # Confusion matrix for trie predictions
    y_pred = np.array(trie_result["predictions"])
    plot_confusion_matrix(
        test_y,
        y_pred,
        CLASSES,
        f"CIFAR-10 Trie Confusion Matrix ({encoder_name})",
        output_dir / f"{encoder_name}_confusion_matrix.png",
    )
    logger.info(f"  Confusion matrix saved to {output_dir}/{encoder_name}_confusion_matrix.png")

    # Per-class accuracy
    per_class = compute_per_class_accuracy(test_y, y_pred, CLASSES)
    logger.info("  Per-class trie accuracy:")
    for cls_name, stats in per_class.items():
        logger.info(
            f"    {cls_name:>12s}: {stats['correct']:3d}/{stats['total']:3d} = {stats['accuracy']:.3f}"
        )

    # Learning curves: trie accuracy at different training set sizes
    fractions = [0.1, 0.25, 0.5, 1.0]
    learning_curve_data: dict[str, list[dict]] = {"trie": [], "knn_k5": []}
    logger.info("  Computing learning curves...")
    for frac in fractions:
        n_sub = max(10, int(n_train * frac))
        sub_train_x = train_x[:n_sub]
        sub_train_y = train_y[:n_sub]

        # Trie at this fraction
        sub_trie = run_trie_classifier(
            torch.from_numpy(sub_train_x),
            torch.from_numpy(sub_train_y.astype(np.int64)),
            torch.from_numpy(test_x),
            torch.from_numpy(test_y.astype(np.int64)),
            epochs=trie_epochs,
            seed=seed,
        )
        learning_curve_data["trie"].append(
            {"n_train": n_sub, "accuracy": sub_trie["accuracy"]}
        )

        # kNN k=5 at this fraction
        from sklearn.neighbors import KNeighborsClassifier

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(sub_train_x, sub_train_y)
        knn_acc = float(np.mean(knn.predict(test_x) == test_y))
        learning_curve_data["knn_k5"].append(
            {"n_train": n_sub, "accuracy": knn_acc}
        )
        logger.info(
            f"    n={n_sub}: trie={sub_trie['accuracy']:.4f}, knn={knn_acc:.4f}"
        )

    plot_learning_curves(
        learning_curve_data,
        f"CIFAR-10 Learning Curves ({encoder_name})",
        output_dir / f"{encoder_name}_learning_curves.png",
    )

    # Print comparison table
    logger.info(f"\n  === {encoder_name} Results ===")
    logger.info(f"  {'Method':<20s} {'Accuracy':>10s}")
    logger.info(f"  {'-' * 30}")
    logger.info(f"  {'CNN head':<20s} {cnn_accuracy:>10.4f}")
    for name, res in baseline_results.items():
        logger.info(f"  {name:<20s} {res['accuracy']:>10.4f}")
    logger.info(f"  {'trie':<20s} {trie_result['accuracy']:>10.4f}")

    # Strip non-serializable predictions from baseline results
    baseline_serializable = {}
    for name, res in baseline_results.items():
        baseline_serializable[name] = {
            k: v for k, v in res.items() if k != "predictions"
        }

    return {
        "cnn_head_accuracy": cnn_accuracy,
        "trie": {
            "accuracy": trie_result["accuracy"],
            "per_class": trie_result["per_class"],
            "trie_stats": trie_result["trie_stats"],
            "train_time": trie_result["train_time"],
            "test_time": trie_result["test_time"],
        },
        "baselines": baseline_serializable,
        "learning_curve": learning_curve_data,
    }


# ─── Encoder Comparison Plot ─────────────────────────────────────────


def plot_encoder_comparison(
    results: dict[str, dict],
    output_dir: Path,
) -> None:
    """Plot grouped bar chart comparing trie and CNN head accuracy across encoders.

    Args:
        results: Dict mapping encoder name to result dict with
            'cnn_head_accuracy' and 'trie' -> 'accuracy'.
        output_dir: Directory to save the plot.
    """
    encoder_names = list(results.keys())
    cnn_accs = [results[e]["cnn_head_accuracy"] for e in encoder_names]
    trie_accs = [results[e]["trie"]["accuracy"] for e in encoder_names]
    knn_accs = [results[e]["baselines"]["knn_k5"]["accuracy"] for e in encoder_names]

    x = np.arange(len(encoder_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, cnn_accs, width, label="CNN Head", color="#2196F3")
    bars2 = ax.bar(x, knn_accs, width, label="kNN (k=5)", color="#4CAF50")
    bars3 = ax.bar(x + width, trie_accs, width, label="Trie", color="#FF9800")

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Encoder Architecture")
    ax.set_ylabel("Accuracy")
    ax.set_title("CIFAR-10: Encoder Capacity vs Classification Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(encoder_names)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    save_path = output_dir / "encoder_comparison.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"\n  Encoder comparison chart saved to {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CIFAR-10 Octonionic Trie Benchmark (Multi-Encoder)"
    )
    parser.add_argument(
        "--encoder",
        choices=["2layer", "4layer", "resnet8", "all"],
        default="all",
        help="Which encoder to train and evaluate (default: all)",
    )
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=3, help="Trie training epochs")
    parser.add_argument(
        "--cnn-epochs",
        type=int,
        default=None,
        help="Override CNN training epochs (for quick testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/trie_benchmarks/cifar10",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("CIFAR-10 Octonionic Trie Benchmark (Multi-Encoder)")
    logger.info("=" * 60)
    logger.info(f"  Device: {device}")
    logger.info(f"  Train: {args.n_train}, Test: {args.n_test}")
    logger.info(f"  Trie epochs: {args.epochs}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Output: {output_dir}")

    # Determine which encoders to run
    encoder_names = list(ENCODER_CONFIGS.keys()) if args.encoder == "all" else [args.encoder]
    logger.info(f"  Encoders: {encoder_names}")

    # Seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Run each encoder
    all_results: dict[str, dict] = {}
    total_start = time.time()

    for enc_name in encoder_names:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Encoder: {enc_name}")
        logger.info("=" * 60)

        model, cnn_acc = train_cnn_encoder(enc_name, device, cnn_epochs=args.cnn_epochs)

        enc_result = evaluate_encoder(
            encoder_name=enc_name,
            model=model,
            cnn_accuracy=cnn_acc,
            n_train=args.n_train,
            n_test=args.n_test,
            trie_epochs=args.epochs,
            output_dir=output_dir,
            device=device,
            seed=args.seed,
        )
        all_results[enc_name] = enc_result

    total_time = time.time() - total_start

    # Encoder comparison plot (when running all encoders)
    if len(all_results) > 1:
        plot_encoder_comparison(all_results, output_dir)

    # Final summary
    logger.info(f"\n{'=' * 60}")
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  {'Encoder':<12s} {'CNN Head':>10s} {'kNN(k=5)':>10s} {'Trie':>10s}")
    logger.info(f"  {'-' * 42}")
    for enc_name, res in all_results.items():
        cnn_acc = res["cnn_head_accuracy"]
        knn_acc = res["baselines"]["knn_k5"]["accuracy"]
        trie_acc = res["trie"]["accuracy"]
        logger.info(f"  {enc_name:<12s} {cnn_acc:>10.4f} {knn_acc:>10.4f} {trie_acc:>10.4f}")
    logger.info(f"\n  Total time: {total_time:.1f}s")

    # Save all results
    output_data = {
        "encoders": all_results,
        "config": {
            "n_train": args.n_train,
            "n_test": args.n_test,
            "trie_epochs": args.epochs,
            "seed": args.seed,
            "encoder_names": encoder_names,
        },
        "total_time_seconds": total_time,
    }
    save_results(output_data, output_dir / "results.json")
    logger.info(f"  Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
