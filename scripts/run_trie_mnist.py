"""MNIST benchmark for the octonionic trie.

Projects MNIST digits to octonionic space via PCA (784D -> 8D), then
classifies using the self-organizing octonionic trie. Compares against
kNN on the same 8D PCA features as a baseline.

Usage:
    docker compose run --rm dev uv run python scripts/run_trie_mnist.py
    docker compose run --rm dev uv run python scripts/run_trie_mnist.py --n-train 10000
    docker compose run --rm dev uv run python scripts/run_trie_mnist.py --epochs 10
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from pathlib import Path

import torch

from octonion.trie import OctonionTrie

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_mnist_pca8(
    n_train: int = 10000, n_test: int = 2000, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load MNIST, PCA to 8D, normalize to unit octonions.

    Returns:
        train_x: [n_train, 8] unit octonions
        train_y: [n_train] integer labels 0-9
        test_x: [n_test, 8] unit octonions
        test_y: [n_test] integer labels 0-9
    """
    from torchvision.datasets import MNIST

    data_dir = tempfile.mkdtemp()
    train_ds = MNIST(data_dir, train=True, download=True)
    test_ds = MNIST(data_dir, train=False, download=True)

    # Flatten and convert to float64
    train_all = train_ds.data.reshape(-1, 784).to(torch.float64) / 255.0
    test_all = test_ds.data.reshape(-1, 784).to(torch.float64) / 255.0
    train_labels = train_ds.targets
    test_labels = test_ds.targets

    # Subsample
    gen = torch.Generator().manual_seed(seed)
    train_idx = torch.randperm(len(train_all), generator=gen)[:n_train]
    test_idx = torch.randperm(len(test_all), generator=gen)[:n_test]

    train_x = train_all[train_idx]
    train_y = train_labels[train_idx]
    test_x = test_all[test_idx]
    test_y = test_labels[test_idx]

    # PCA to 8D (compute on training data, apply to both)
    mean = train_x.mean(dim=0)
    train_centered = train_x - mean
    test_centered = test_x - mean

    # SVD for PCA
    U, S, Vt = torch.linalg.svd(train_centered, full_matrices=False)
    components = Vt[:8]  # Top 8 principal components

    train_pca = train_centered @ components.T
    test_pca = test_centered @ components.T

    # Normalize to unit octonions
    train_norms = train_pca.norm(dim=1, keepdim=True).clamp(min=1e-10)
    test_norms = test_pca.norm(dim=1, keepdim=True).clamp(min=1e-10)
    train_oct = train_pca / train_norms
    test_oct = test_pca / test_norms

    # Report variance explained
    total_var = (S**2).sum()
    explained = (S[:8] ** 2).sum() / total_var
    logger.info(f"PCA: 8 components explain {explained:.1%} of variance")

    return train_oct, train_y, test_oct, test_y


def knn_baseline(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    k: int = 5,
) -> float:
    """k-nearest neighbors baseline on the same PCA features."""
    correct = 0
    # Compute all pairwise distances at once
    dists = torch.cdist(test_x, train_x)  # [n_test, n_train]
    _, topk_idx = dists.topk(k, dim=1, largest=False)
    topk_labels = train_y[topk_idx]  # [n_test, k]

    for i in range(len(test_y)):
        # Majority vote
        counts = torch.bincount(topk_labels[i], minlength=10)
        pred = counts.argmax().item()
        if pred == test_y[i].item():
            correct += 1

    return correct / len(test_y)


def trie_classify(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    epochs: int = 3,
    assoc_threshold: float = 0.3,
    seed: int = 42,
) -> dict:
    """Classify MNIST digits using the octonionic trie."""
    trie = OctonionTrie(
        associator_threshold=assoc_threshold,
        similarity_threshold=0.1,
        max_depth=15,
        seed=seed,
    )

    # Train
    t0 = time.time()
    for ep in range(epochs):
        for i in range(len(train_x)):
            trie.insert(train_x[i], category=train_y[i].item())
        if ep % 2 == 1:
            trie.consolidate()
    trie.consolidate()
    train_time = time.time() - t0

    # Test
    t0 = time.time()
    correct = 0
    per_digit = {d: {"correct": 0, "total": 0} for d in range(10)}
    for i in range(len(test_x)):
        label = test_y[i].item()
        leaf = trie.query(test_x[i])
        pred = leaf.dominant_category

        per_digit[label]["total"] += 1
        if pred == label:
            correct += 1
            per_digit[label]["correct"] += 1
    test_time = time.time() - t0

    acc = correct / len(test_y)
    stats = trie.stats()

    return {
        "accuracy": acc,
        "per_digit": per_digit,
        "train_time": train_time,
        "test_time": test_time,
        "stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST Octonionic Trie Benchmark")
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--assoc-threshold", type=float, default=0.3)
    parser.add_argument("--output-dir", type=str, default="results/trie_validation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MNIST Octonionic Trie Benchmark")
    logger.info("=" * 60)
    logger.info(f"  Train: {args.n_train}, Test: {args.n_test}")
    logger.info(f"  Epochs: {args.epochs}, Threshold: {args.assoc_threshold}")

    # Load data
    logger.info("\nLoading MNIST and projecting to octonionic space...")
    train_x, train_y, test_x, test_y = load_mnist_pca8(
        n_train=args.n_train, n_test=args.n_test
    )
    logger.info(f"  Train: {train_x.shape}, Test: {test_x.shape}")

    # kNN baseline
    logger.info("\nRunning kNN baseline (k=5)...")
    t0 = time.time()
    knn_acc = knn_baseline(train_x, train_y, test_x, test_y, k=5)
    knn_time = time.time() - t0
    logger.info(f"  kNN accuracy: {knn_acc:.3f} ({knn_time:.1f}s)")

    logger.info(f"\nRunning kNN baseline (k=1)...")
    knn1_acc = knn_baseline(train_x, train_y, test_x, test_y, k=1)
    logger.info(f"  1-NN accuracy: {knn1_acc:.3f}")

    # Trie
    logger.info(f"\nRunning octonionic trie ({args.epochs} epochs)...")
    result = trie_classify(
        train_x, train_y, test_x, test_y,
        epochs=args.epochs,
        assoc_threshold=args.assoc_threshold,
    )

    logger.info(f"  Trie accuracy: {result['accuracy']:.3f} ({result['train_time']:.1f}s train, {result['test_time']:.1f}s test)")
    logger.info(f"  Nodes: {result['stats']['n_nodes']}, Leaves: {result['stats']['n_leaves']}, Depth: {result['stats']['max_depth']}")
    logger.info(f"  Rumination rejections: {result['stats']['rumination_rejections']}")

    logger.info("\n  Per-digit accuracy:")
    for d in range(10):
        pd = result["per_digit"][d]
        digit_acc = pd["correct"] / max(pd["total"], 1)
        logger.info(f"    Digit {d}: {pd['correct']:3d}/{pd['total']:3d} = {digit_acc:.3f}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)
    logger.info(f"  kNN (k=5) on PCA-8D: {knn_acc:.3f}")
    logger.info(f"  1-NN on PCA-8D:       {knn1_acc:.3f}")
    logger.info(f"  Octonionic Trie:      {result['accuracy']:.3f}")

    # Save
    all_results = {
        "knn_k5": knn_acc,
        "knn_k1": knn1_acc,
        "trie": result["accuracy"],
        "trie_per_digit": {
            str(d): result["per_digit"][d] for d in range(10)
        },
        "trie_stats": result["stats"],
        "config": {
            "n_train": args.n_train,
            "n_test": args.n_test,
            "epochs": args.epochs,
            "assoc_threshold": args.assoc_threshold,
        },
    }
    with open(output_dir / "mnist_benchmark.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n  Results saved to {output_dir}/mnist_benchmark.json")


if __name__ == "__main__":
    main()
