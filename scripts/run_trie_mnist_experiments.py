"""MNIST trie experiments: O^n projection and encoder comparison.

Experiment A: Vary the number of octonionic dimensions (O^1 through O^8)
    using PCA projection, measuring how more dimensions improve accuracy.

Experiment B: Compare encoders (PCA vs pre-trained CNN features) at fixed
    octonionic dimensionality, measuring how encoding quality affects the trie.

Usage:
    docker compose run --rm dev uv run python scripts/run_trie_mnist_experiments.py --experiment A
    docker compose run --rm dev uv run python scripts/run_trie_mnist_experiments.py --experiment B
    docker compose run --rm dev uv run python scripts/run_trie_mnist_experiments.py  # both
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn

from octonion.trie import OctonionTrie

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ── Data loading ─────────────────────────────────────────────────────


def load_mnist_raw(
    n_train: int = 10000, n_test: int = 2000, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load raw MNIST as flat float64 tensors."""
    from torchvision.datasets import MNIST

    data_dir = tempfile.mkdtemp()
    train_ds = MNIST(data_dir, train=True, download=True)
    test_ds = MNIST(data_dir, train=False, download=True)

    train_all = train_ds.data.reshape(-1, 784).to(torch.float64) / 255.0
    test_all = test_ds.data.reshape(-1, 784).to(torch.float64) / 255.0

    gen = torch.Generator().manual_seed(seed)
    train_idx = torch.randperm(len(train_all), generator=gen)[:n_train]
    test_idx = torch.randperm(len(test_all), generator=gen)[:n_test]

    return (
        train_all[train_idx], train_ds.targets[train_idx],
        test_all[test_idx], test_ds.targets[test_idx],
    )


# ── Encoders ─────────────────────────────────────────────────────────


def encode_pca(
    train_x: torch.Tensor, test_x: torch.Tensor, n_dims: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """PCA projection to n_dims, normalized to unit octonions per block of 8."""
    mean = train_x.mean(dim=0)
    _, S, Vt = torch.linalg.svd(train_x - mean, full_matrices=False)

    components = Vt[:n_dims]
    train_proj = (train_x - mean) @ components.T
    test_proj = (test_x - mean) @ components.T

    # Normalize each block of 8 to unit norm (each octonion independently)
    n_octs = n_dims // 8
    for i in range(n_octs):
        sl = slice(i * 8, (i + 1) * 8)
        train_norms = train_proj[:, sl].norm(dim=1, keepdim=True).clamp(min=1e-10)
        test_norms = test_proj[:, sl].norm(dim=1, keepdim=True).clamp(min=1e-10)
        train_proj[:, sl] = train_proj[:, sl] / train_norms
        test_proj[:, sl] = test_proj[:, sl] / test_norms

    explained = (S[:n_dims] ** 2).sum() / (S**2).sum()
    logger.info(f"    PCA-{n_dims}: {explained:.1%} variance explained")

    return train_proj, test_proj


def encode_cnn_features(
    train_x: torch.Tensor, test_x: torch.Tensor, n_dims: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract features from a small CNN trained on MNIST, project to n_dims.

    Trains a minimal CNN for 3 epochs, extracts penultimate layer features,
    then PCA-projects those features to n_dims.
    """
    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader, TensorDataset

    # Train a small CNN
    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            )
            self.classifier = nn.Linear(128, 10)

        def forward(self, x):
            feat = self.features(x)
            return self.classifier(feat)

        def extract(self, x):
            return self.features(x)

    logger.info("    Training small CNN encoder (3 epochs)...")
    # Use full MNIST for CNN training
    data_dir = tempfile.mkdtemp()
    full_train = MNIST(data_dir, train=True, download=True)
    full_data = full_train.data.unsqueeze(1).float() / 255.0
    full_labels = full_train.targets

    cnn = SmallCNN()
    opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(full_data, full_labels), batch_size=256, shuffle=True
    )

    cnn.train()
    for epoch in range(3):
        total_loss = 0
        for bx, by in loader:
            opt.zero_grad()
            out = cnn(bx)
            loss = loss_fn(out, by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        logger.info(f"      Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

    # Extract features for our subset
    cnn.eval()
    with torch.no_grad():
        train_images = train_x.reshape(-1, 1, 28, 28).float()
        test_images = test_x.reshape(-1, 1, 28, 28).float()

        # Batch extraction to avoid OOM
        train_feats = []
        for i in range(0, len(train_images), 1000):
            train_feats.append(cnn.extract(train_images[i:i+1000]))
        train_feats = torch.cat(train_feats).to(torch.float64)

        test_feats = []
        for i in range(0, len(test_images), 1000):
            test_feats.append(cnn.extract(test_images[i:i+1000]))
        test_feats = torch.cat(test_feats).to(torch.float64)

    logger.info(f"    CNN features: {train_feats.shape[1]}D")

    # PCA to n_dims
    mean = train_feats.mean(dim=0)
    _, S, Vt = torch.linalg.svd(train_feats - mean, full_matrices=False)
    components = Vt[:n_dims]
    train_proj = (train_feats - mean) @ components.T
    test_proj = (test_feats - mean) @ components.T

    # Normalize each octonion block
    n_octs = n_dims // 8
    for i in range(n_octs):
        sl = slice(i * 8, (i + 1) * 8)
        train_proj[:, sl] /= train_proj[:, sl].norm(dim=1, keepdim=True).clamp(min=1e-10)
        test_proj[:, sl] /= test_proj[:, sl].norm(dim=1, keepdim=True).clamp(min=1e-10)

    explained = (S[:n_dims] ** 2).sum() / (S**2).sum()
    logger.info(f"    CNN-PCA-{n_dims}: {explained:.1%} of CNN feature variance")

    return train_proj, test_proj


# ── Trie + kNN evaluation ───────────────────────────────────────────


def evaluate_trie_on(
    train_proj: torch.Tensor, train_y: torch.Tensor,
    test_proj: torch.Tensor, test_y: torch.Tensor,
    n_octs: int, epochs: int = 3, assoc_threshold: float = 0.3,
) -> dict:
    """Run the octonionic trie on O^n encoded data.

    For O^n with n > 1, uses cascaded routing: route through octonion 1
    first, then at the leaf, route through octonion 2, etc.
    """
    if n_octs == 1:
        return _eval_single_oct(train_proj, train_y, test_proj, test_y, epochs, assoc_threshold)
    else:
        return _eval_multi_oct(train_proj, train_y, test_proj, test_y, n_octs, epochs, assoc_threshold)


def _eval_single_oct(train_x, train_y, test_x, test_y, epochs, threshold):
    trie = OctonionTrie(associator_threshold=threshold, max_depth=15, seed=42)
    t0 = time.time()
    for _ in range(epochs):
        for i in range(len(train_x)):
            trie.insert(train_x[i, :8], category=train_y[i].item())
    trie.consolidate()
    train_time = time.time() - t0

    correct = 0
    for i in range(len(test_x)):
        leaf = trie.query(test_x[i, :8])
        if leaf.dominant_category == test_y[i].item():
            correct += 1

    return {"accuracy": correct / len(test_y), "train_time": train_time, "stats": trie.stats()}


def _eval_multi_oct(train_x, train_y, test_x, test_y, n_octs, epochs, threshold):
    """Ensemble of n tries, one per octonion. Majority vote at test time."""
    tries = []
    t0 = time.time()
    for oct_idx in range(n_octs):
        sl = slice(oct_idx * 8, (oct_idx + 1) * 8)
        trie = OctonionTrie(associator_threshold=threshold, max_depth=15, seed=42 + oct_idx)
        for _ in range(epochs):
            for i in range(len(train_x)):
                trie.insert(train_x[i, sl], category=train_y[i].item())
        trie.consolidate()
        tries.append((trie, sl))
    train_time = time.time() - t0

    # Majority vote
    correct = 0
    for i in range(len(test_x)):
        votes = torch.zeros(10, dtype=torch.int64)
        for trie, sl in tries:
            leaf = trie.query(test_x[i, sl])
            pred = leaf.dominant_category
            if pred is not None:
                votes[pred] += 1
        if votes.argmax().item() == test_y[i].item():
            correct += 1

    total_nodes = sum(t.stats()["n_nodes"] for t, _ in tries)
    return {"accuracy": correct / len(test_y), "train_time": train_time,
            "stats": {"n_nodes": total_nodes, "n_tries": n_octs}}


def knn_accuracy(train_x, train_y, test_x, test_y, k=5):
    dists = torch.cdist(test_x.float(), train_x.float())
    _, topk = dists.topk(k, dim=1, largest=False)
    correct = 0
    for i in range(len(test_y)):
        counts = torch.bincount(train_y[topk[i]], minlength=10)
        if counts.argmax().item() == test_y[i].item():
            correct += 1
    return correct / len(test_y)


# ── Experiment A: O^n dimensionality ─────────────────────────────────


def experiment_A(train_raw, train_y, test_raw, test_y, output_dir):
    logger.info("=" * 60)
    logger.info("Experiment A: O^n Dimensionality Sweep")
    logger.info("=" * 60)

    results = {}
    for n_octs in [1, 2, 3, 4, 6, 8]:
        n_dims = n_octs * 8
        logger.info(f"\n  O^{n_octs} (PCA-{n_dims}D):")

        train_proj, test_proj = encode_pca(train_raw, test_raw, n_dims)

        # kNN baseline on same features
        knn = knn_accuracy(train_proj, train_y, test_proj, test_y, k=5)
        logger.info(f"    kNN(k=5): {knn:.3f}")

        # Trie
        trie_result = evaluate_trie_on(train_proj, train_y, test_proj, test_y, n_octs)
        logger.info(f"    Trie:     {trie_result['accuracy']:.3f} ({trie_result['train_time']:.1f}s, {trie_result['stats']['n_nodes']} nodes)")

        results[f"O{n_octs}"] = {
            "n_dims": n_dims, "knn_k5": knn,
            "trie": trie_result["accuracy"],
            "trie_nodes": trie_result["stats"]["n_nodes"],
            "train_time": trie_result["train_time"],
        }

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Experiment A Summary")
    logger.info("=" * 60)
    logger.info(f"  {'Repr':>6} | {'Dims':>4} | {'kNN':>6} | {'Trie':>6} | {'Nodes':>6} | {'Time':>6}")
    logger.info(f"  {'':->6}-+-{'':->4}-+-{'':->6}-+-{'':->6}-+-{'':->6}-+-{'':->6}")
    for label, r in results.items():
        logger.info(
            f"  {label:>6} | {r['n_dims']:>4} | {r['knn_k5']:>6.3f} | {r['trie']:>6.3f} | "
            f"{r['trie_nodes']:>6} | {r['train_time']:>5.0f}s"
        )

    with open(output_dir / "experiment_A_dimensions.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ── Experiment B: Encoder comparison ─────────────────────────────────


def experiment_B(train_raw, train_y, test_raw, test_y, output_dir):
    logger.info("\n" + "=" * 60)
    logger.info("Experiment B: Encoder Comparison (fixed O^2 = 16D)")
    logger.info("=" * 60)

    n_dims = 16  # O^2
    results = {}

    # PCA encoder
    logger.info("\n  Encoder: PCA")
    train_pca, test_pca = encode_pca(train_raw, test_raw, n_dims)
    knn_pca = knn_accuracy(train_pca, train_y, test_pca, test_y, k=5)
    trie_pca = evaluate_trie_on(train_pca, train_y, test_pca, test_y, n_octs=2)
    logger.info(f"    kNN: {knn_pca:.3f}, Trie: {trie_pca['accuracy']:.3f}")
    results["PCA"] = {"knn": knn_pca, "trie": trie_pca["accuracy"]}

    # CNN encoder
    logger.info("\n  Encoder: CNN (3-epoch trained)")
    train_cnn, test_cnn = encode_cnn_features(train_raw, test_raw, n_dims)
    knn_cnn = knn_accuracy(train_cnn, train_y, test_cnn, test_y, k=5)
    trie_cnn = evaluate_trie_on(train_cnn, train_y, test_cnn, test_y, n_octs=2)
    logger.info(f"    kNN: {knn_cnn:.3f}, Trie: {trie_cnn['accuracy']:.3f}")
    results["CNN"] = {"knn": knn_cnn, "trie": trie_cnn["accuracy"]}

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Experiment B Summary (O^2 = 16D)")
    logger.info("=" * 60)
    logger.info(f"  {'Encoder':>8} | {'kNN(k=5)':>8} | {'Trie':>6}")
    logger.info(f"  {'':->8}-+-{'':->8}-+-{'':->6}")
    for enc, r in results.items():
        logger.info(f"  {enc:>8} | {r['knn']:>8.3f} | {r['trie']:>6.3f}")

    with open(output_dir / "experiment_B_encoders.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["A", "B"], default=None, help="Run one experiment (default: both)")
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="results/trie_validation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading MNIST...")
    train_raw, train_y, test_raw, test_y = load_mnist_raw(args.n_train, args.n_test)
    logger.info(f"  Train: {len(train_y)}, Test: {len(test_y)}")

    if args.experiment is None or args.experiment == "A":
        experiment_A(train_raw, train_y, test_raw, test_y, output_dir)

    if args.experiment is None or args.experiment == "B":
        experiment_B(train_raw, train_y, test_raw, test_y, output_dir)


if __name__ == "__main__":
    main()
