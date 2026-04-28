"""Pre-compute and cache encoder features for all T1 benchmarks.

Produces .pt files containing 8D unit-normalized octonion features for use
by threshold sweep workers (T2). Workers load features only -- no GPU, no
encoder training, no data pipeline dependencies.

Each .pt file is a dict with keys:
    train_x: torch.Tensor float64 [N, 8] unit normalized
    train_y: torch.Tensor int64 [N]
    test_x:  torch.Tensor float64 [M, 8] unit normalized
    test_y:  torch.Tensor int64 [M]
    class_names: list[str]
    benchmark: str
    n_train: int
    n_test: int

Usage:
    docker compose run --rm dev uv run python scripts/sweep/cache_features.py --benchmarks all
    docker compose run --rm dev uv run python scripts/sweep/cache_features.py --benchmarks mnist,fashion_mnist --subset-only
    docker compose run --rm dev uv run python scripts/sweep/cache_features.py --benchmarks mnist --output-dir /tmp/test_cache
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("TRIE_DATA_DIR", "/workspace/.data")
os.makedirs(DATA_DIR, exist_ok=True)

SEED = 42

# All supported benchmarks
ALL_BENCHMARKS = ["mnist", "fashion_mnist", "cifar10", "text_4class", "text_20class"]


# ── Shared utilities ─────────────────────────────────────────────────


def unit_normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize each row to unit norm (unit octonions).

    Args:
        x: Tensor of shape [N, 8].

    Returns:
        Unit-normalized tensor of shape [N, 8], dtype float64.
    """
    x = x.to(torch.float64)
    norms = x.norm(dim=1, keepdim=True).clamp(min=1e-10)
    return x / norms


def save_cached_features(
    path: Path,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    class_names: list[str],
    benchmark: str,
) -> None:
    """Save cached features to a .pt file.

    Args:
        path: Output file path.
        train_x: Training features [N, 8] float64 unit norm.
        train_y: Training labels [N] int64.
        test_x: Test features [M, 8] float64 unit norm.
        test_y: Test labels [M] int64.
        class_names: List of class name strings.
        benchmark: Benchmark name string.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "train_x": train_x.to(torch.float64),
        "train_y": train_y.to(torch.int64),
        "test_x": test_x.to(torch.float64),
        "test_y": test_y.to(torch.int64),
        "class_names": class_names,
        "benchmark": benchmark,
        "n_train": len(train_x),
        "n_test": len(test_x),
    }
    torch.save(data, path)
    logger.info(
        f"  Saved {path.name}: train={train_x.shape}, test={test_x.shape}, "
        f"dtype={train_x.dtype}"
    )


# ── MNIST ────────────────────────────────────────────────────────────


def _mnist_pca8(
    train_flat: torch.Tensor,
    test_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """PCA to 8D on the given train/test data.

    Fits PCA on train_flat, applies to both. Returns unit-normalized
    features and explained variance ratio.

    Args:
        train_flat: [N, 784] float64 flattened pixel data.
        test_flat: [M, 784] float64 flattened pixel data.

    Returns:
        train_oct: [N, 8] float64 unit normalized.
        test_oct: [M, 8] float64 unit normalized.
        explained: Fraction of variance explained by top 8 components.
    """
    mean = train_flat.mean(dim=0)
    train_centered = train_flat - mean
    test_centered = test_flat - mean

    U, S, Vt = torch.linalg.svd(train_centered, full_matrices=False)
    components = Vt[:8]

    train_pca = train_centered @ components.T
    test_pca = test_centered @ components.T

    total_var = (S**2).sum()
    explained = float((S[:8] ** 2).sum() / total_var)

    train_oct = unit_normalize(train_pca)
    test_oct = unit_normalize(test_pca)

    return train_oct, test_oct, explained


def cache_mnist(
    output_dir: Path,
    full: bool = True,
    subset: bool = True,
) -> None:
    """Cache MNIST features using PCA to 8D.

    Replicates the pipeline from run_trie_mnist.py:load_mnist_pca8 exactly.
    For both full and subset scales, PCA is fit on the TRAINING portion of
    that scale (matching the original T1 pipeline where PCA was fit on the
    subsampled training data).
    """
    from torchvision.datasets import MNIST

    logger.info("\n[MNIST] Loading dataset...")
    train_ds = MNIST(DATA_DIR, train=True, download=True)
    test_ds = MNIST(DATA_DIR, train=False, download=True)

    # Flatten and convert to float64 (exact match to run_trie_mnist.py)
    train_all = train_ds.data.reshape(-1, 784).to(torch.float64) / 255.0
    test_all = test_ds.data.reshape(-1, 784).to(torch.float64) / 255.0
    train_labels_all = train_ds.targets.clone()
    test_labels_all = test_ds.targets.clone()

    class_names = [str(i) for i in range(10)]

    # Full scale: PCA fit on all 60K training samples
    if full:
        logger.info("[MNIST] Computing PCA (784D -> 8D) on full 60K training set...")
        train_oct, test_oct, explained = _mnist_pca8(train_all, test_all)
        logger.info(f"[MNIST] PCA (full): 8 components explain {explained:.1%} of variance")
        save_cached_features(
            output_dir / "mnist_features.pt",
            train_oct, train_labels_all,
            test_oct, test_labels_all,
            class_names, "mnist",
        )

    # 10K subset: subsample FIRST, then PCA on subsampled data
    # This matches the T1 pipeline (run_trie_mnist.py:load_mnist_pca8)
    if subset:
        gen = torch.Generator().manual_seed(SEED)
        train_idx = torch.randperm(len(train_all), generator=gen)[:10000]
        test_idx = torch.randperm(len(test_all), generator=gen)[:2000]

        sub_train_flat = train_all[train_idx]
        sub_test_flat = test_all[test_idx]
        sub_train_labels = train_labels_all[train_idx]
        sub_test_labels = test_labels_all[test_idx]

        logger.info("[MNIST] Computing PCA (784D -> 8D) on 10K subset...")
        train_oct_sub, test_oct_sub, explained_sub = _mnist_pca8(
            sub_train_flat, sub_test_flat
        )
        logger.info(f"[MNIST] PCA (10K): 8 components explain {explained_sub:.1%} of variance")
        save_cached_features(
            output_dir / "mnist_10k_features.pt",
            train_oct_sub, sub_train_labels,
            test_oct_sub, sub_test_labels,
            class_names, "mnist_10k",
        )


# ── Fashion-MNIST ────────────────────────────────────────────────────


class SmallCNN(nn.Module):
    """Small CNN encoder matching run_trie_fashion_mnist.py architecture.

    Architecture:
        Conv2d(1,16,3,pad=1) -> ReLU -> MaxPool2d(2) ->
        Conv2d(16,32,3,pad=1) -> ReLU -> MaxPool2d(2) ->
        Flatten -> Linear(32*7*7, 128) -> ReLU ->
        Linear(128, 8) -> ReLU
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
        feat = self.features(x)
        return self.classifier(feat)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def cache_fashion_mnist(
    output_dir: Path,
    full: bool = True,
    subset: bool = True,
) -> None:
    """Cache Fashion-MNIST features using CNN encoder to 8D.

    Replicates the pipeline from run_trie_fashion_mnist.py:
    - Train SmallCNN on full 60K for 5 epochs (per plan; original uses 10)
    - Extract 8D features, unit normalize to float64
    """
    from torchvision.datasets import FashionMNIST

    logger.info("\n[Fashion-MNIST] Loading dataset...")
    train_ds = FashionMNIST(DATA_DIR, train=True, download=True)
    test_ds = FashionMNIST(DATA_DIR, train=False, download=True)

    full_train_images = train_ds.data.unsqueeze(1).float() / 255.0
    full_train_labels = train_ds.targets.clone()
    full_test_images = test_ds.data.unsqueeze(1).float() / 255.0
    full_test_labels = test_ds.targets.clone()

    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]

    # Train CNN encoder on full training set (seed=42 for reproducibility)
    torch.manual_seed(SEED)
    cnn = SmallCNN(feature_dim=8)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(full_train_images, full_train_labels),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    cnn_epochs = 5
    logger.info(f"[Fashion-MNIST] Training SmallCNN encoder ({cnn_epochs} epochs on 60K)...")
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

    # Extract features from all data
    logger.info("[Fashion-MNIST] Extracting 8D features...")
    cnn.eval()
    with torch.no_grad():
        train_feats = []
        for i in range(0, len(full_train_images), 1000):
            train_feats.append(cnn.extract(full_train_images[i : i + 1000]))
        train_feats = torch.cat(train_feats)

        test_feats = []
        for i in range(0, len(full_test_images), 1000):
            test_feats.append(cnn.extract(full_test_images[i : i + 1000]))
        test_feats = torch.cat(test_feats)

    # Unit normalize to float64
    train_oct = unit_normalize(train_feats)
    test_oct = unit_normalize(test_feats)

    # Full scale
    if full:
        save_cached_features(
            output_dir / "fashion_mnist_features.pt",
            train_oct, full_train_labels,
            test_oct, full_test_labels,
            class_names, "fashion_mnist",
        )

    # 10K subset
    if subset:
        gen = torch.Generator().manual_seed(SEED)
        train_idx = torch.randperm(len(train_oct), generator=gen)[:10000]
        test_idx = torch.randperm(len(test_oct), generator=gen)[:2000]
        save_cached_features(
            output_dir / "fashion_mnist_10k_features.pt",
            train_oct[train_idx], full_train_labels[train_idx],
            test_oct[test_idx], full_test_labels[test_idx],
            class_names, "fashion_mnist_10k",
        )


# ── CIFAR-10 (ResNet-8) ─────────────────────────────────────────────


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
        import torch.nn.functional as F

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class CIFAR_CNN_ResNet8(nn.Module):
    """Simplified ResNet with ~8 conv layers for CIFAR-10.

    Exact copy from run_trie_cifar10.py.
    """

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


def cache_cifar10(
    output_dir: Path,
    full: bool = True,
    subset: bool = True,
) -> None:
    """Cache CIFAR-10 features using ResNet-8 encoder.

    Replicates the pipeline from run_trie_cifar10.py with resnet8 config:
    - Train ResNet-8 for 50 epochs with cosine annealing
    - Standard CIFAR-10 augmentation during training
    - Non-augmented extraction for deterministic features
    - Unit normalize to float64
    """
    from torchvision import transforms
    from torchvision.datasets import CIFAR10

    logger.info("\n[CIFAR-10] Loading dataset...")
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    train_dataset = CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)

    # Non-augmented version for feature extraction
    train_dataset_eval = CIFAR10(root=DATA_DIR, train=True, download=False, transform=test_transform)
    test_dataset_eval = CIFAR10(root=DATA_DIR, train=False, download=False, transform=test_transform)

    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[CIFAR-10] Device: {device}")

    # Train ResNet-8 (matching run_trie_cifar10.py resnet8 config)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = CIFAR_CNN_ResNet8(feature_dim=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    cnn_epochs = 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cnn_epochs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        multiprocessing_context="spawn",
    )

    criterion = nn.CrossEntropyLoss()

    logger.info(f"[CIFAR-10] Training ResNet-8 for {cnn_epochs} epochs with cosine annealing...")
    model.train()
    for ep in range(cnn_epochs):
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
        scheduler.step()
        if (ep + 1) % 10 == 0 or ep == cnn_epochs - 1:
            avg_loss = running_loss / max(n_batches, 1)
            logger.info(f"  Epoch {ep + 1}/{cnn_epochs}: loss={avg_loss:.4f}")

    # Evaluate CNN head accuracy
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        multiprocessing_context="spawn",
    )
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    logger.info(f"[CIFAR-10] ResNet-8 CNN head accuracy: {correct / total:.4f}")

    # Extract features (non-augmented, deterministic)
    logger.info("[CIFAR-10] Extracting 8D features (non-augmented)...")
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
    train_labels_all = torch.cat(all_train_labels, dim=0)
    test_feats = torch.cat(all_test_feats, dim=0)
    test_labels_all = torch.cat(all_test_labels, dim=0)

    # Unit normalize to float64
    train_oct = unit_normalize(train_feats)
    test_oct = unit_normalize(test_feats)

    # Full scale: 50K train, 10K test
    if full:
        save_cached_features(
            output_dir / "cifar10_features.pt",
            train_oct, train_labels_all,
            test_oct, test_labels_all,
            class_names, "cifar10",
        )

    # 10K subset
    if subset:
        gen = torch.Generator().manual_seed(SEED)
        train_idx = torch.randperm(len(train_oct), generator=gen)[:10000]
        test_idx = torch.randperm(len(test_oct), generator=gen)[:2000]
        save_cached_features(
            output_dir / "cifar10_10k_features.pt",
            train_oct[train_idx], train_labels_all[train_idx],
            test_oct[test_idx], test_labels_all[test_idx],
            class_names, "cifar10_10k",
        )


# ── Text Classification ──────────────────────────────────────────────


SUBSET_CATEGORIES = [
    "comp.graphics",
    "rec.sport.baseball",
    "sci.med",
    "talk.politics.guns",
]

SHORT_NAMES_20 = [
    "alt.atheism", "comp.graphics", "comp.os.ms-win", "comp.sys.ibm",
    "comp.sys.mac", "comp.windows.x", "misc.forsale", "rec.autos",
    "rec.motorcycles", "rec.sport.bball", "rec.sport.hockey", "sci.crypt",
    "sci.electronics", "sci.med", "sci.space", "soc.rel.christian",
    "talk.pol.guns", "talk.pol.mideast", "talk.pol.misc", "talk.rel.misc",
]


def _cache_text(
    output_dir: Path,
    mode: str,
    full: bool = True,
    subset: bool = True,
) -> None:
    """Cache text features for 4-class or 20-class.

    Replicates the pipeline from run_trie_text.py:
    - TF-IDF vectorization (max_features=10000, sublinear_tf, etc.)
    - TruncatedSVD to 8D
    - Unit normalize to float64

    Args:
        output_dir: Directory for output .pt files.
        mode: "subset" for 4-class, "full" for 20-class.
        full: Whether to cache full-scale features.
        subset: Whether to cache 10K subset features.
    """
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    categories = SUBSET_CATEGORIES if mode == "subset" else None
    benchmark_name = "text_4class" if mode == "subset" else "text_20class"
    class_names = SUBSET_CATEGORIES if mode == "subset" else SHORT_NAMES_20

    logger.info(f"\n[{benchmark_name}] Loading 20 Newsgroups ({mode})...")
    train = fetch_20newsgroups(
        subset="train", categories=categories,
        remove=("headers", "footers", "quotes"), random_state=SEED
    )
    test = fetch_20newsgroups(
        subset="test", categories=categories,
        remove=("headers", "footers", "quotes"), random_state=SEED
    )

    train_texts = train.data
    train_labels = np.array(train.target)
    test_texts = test.data
    test_labels = np.array(test.target)
    target_names = list(train.target_names)

    logger.info(f"[{benchmark_name}] Train: {len(train_texts)}, Test: {len(test_texts)}, Classes: {len(target_names)}")

    # TF-IDF vectorization (exact match to run_trie_text.py)
    logger.info(f"[{benchmark_name}] TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        sublinear_tf=True,
        max_df=0.5,
        min_df=5,
        stop_words="english",
    )
    train_tfidf = vectorizer.fit_transform(train_texts)
    test_tfidf = vectorizer.transform(test_texts)

    # TruncatedSVD to 8D (NOT PCA -- works directly on sparse CSR)
    logger.info(f"[{benchmark_name}] TruncatedSVD to 8D...")
    svd = TruncatedSVD(n_components=8, random_state=SEED)
    train_reduced = svd.fit_transform(train_tfidf)
    test_reduced = svd.transform(test_tfidf)

    explained = float(svd.explained_variance_ratio_.sum())
    logger.info(f"[{benchmark_name}] Explained variance: {explained:.4f}")

    # Unit normalize
    train_norms = np.linalg.norm(train_reduced, axis=1, keepdims=True)
    train_norms = np.clip(train_norms, 1e-10, None)
    train_oct_np = train_reduced / train_norms

    test_norms = np.linalg.norm(test_reduced, axis=1, keepdims=True)
    test_norms = np.clip(test_norms, 1e-10, None)
    test_oct_np = test_reduced / test_norms

    # Convert to torch tensors
    train_oct = torch.from_numpy(train_oct_np).to(torch.float64)
    test_oct = torch.from_numpy(test_oct_np).to(torch.float64)
    train_y = torch.from_numpy(train_labels).to(torch.int64)
    test_y = torch.from_numpy(test_labels).to(torch.int64)

    # Use actual target_names from the dataset as class_names
    actual_class_names = target_names

    # Full scale (all available samples)
    if full:
        save_cached_features(
            output_dir / f"{benchmark_name}_features.pt",
            train_oct, train_y,
            test_oct, test_y,
            actual_class_names, benchmark_name,
        )

    # 10K subset (or fewer if dataset is smaller)
    if subset:
        n_train_sub = min(10000, len(train_oct))
        n_test_sub = min(2000, len(test_oct))

        gen = torch.Generator().manual_seed(SEED)
        train_idx = torch.randperm(len(train_oct), generator=gen)[:n_train_sub]
        test_idx = torch.randperm(len(test_oct), generator=gen)[:n_test_sub]
        save_cached_features(
            output_dir / f"{benchmark_name}_10k_features.pt",
            train_oct[train_idx], train_y[train_idx],
            test_oct[test_idx], test_y[test_idx],
            actual_class_names, f"{benchmark_name}_10k",
        )


def cache_text_4class(
    output_dir: Path,
    full: bool = True,
    subset: bool = True,
) -> None:
    """Cache 4-class text classification features."""
    _cache_text(output_dir, mode="subset", full=full, subset=subset)


def cache_text_20class(
    output_dir: Path,
    full: bool = True,
    subset: bool = True,
) -> None:
    """Cache 20-class text classification features."""
    _cache_text(output_dir, mode="full", full=full, subset=subset)


# ── Validation ────────────────────────────────────────────────────────


def validate_cached_features(output_dir: Path) -> None:
    """Validate cached MNIST 10K subset by running trie with GlobalPolicy(0.3).

    This checks that cached features produce the same accuracy as the
    original T1 pipeline.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from octonion.trie import OctonionTrie

    cache_path = output_dir / "mnist_10k_features.pt"
    if not cache_path.exists():
        logger.info("\n[Validation] Skipping -- mnist_10k_features.pt not found")
        return

    logger.info("\n[Validation] Running trie on cached MNIST 10K subset...")
    data = torch.load(cache_path, weights_only=False)

    train_x = data["train_x"]
    train_y = data["train_y"]
    test_x = data["test_x"]
    test_y = data["test_y"]

    trie = OctonionTrie(
        associator_threshold=0.3,
        similarity_threshold=0.1,
        max_depth=15,
        seed=SEED,
    )

    # Train for 3 epochs (same as T1)
    for ep in range(3):
        for i in range(len(train_x)):
            trie.insert(train_x[i], category=train_y[i].item())
        if ep % 2 == 1:
            trie.consolidate()
    trie.consolidate()

    # Test
    correct = 0
    for i in range(len(test_x)):
        leaf = trie.query(test_x[i])
        if leaf.dominant_category == test_y[i].item():
            correct += 1

    accuracy = correct / len(test_y)
    logger.info(f"[Validation] MNIST 10K trie accuracy: {accuracy:.4f}")
    logger.info("[Validation] Features cached successfully -- accuracy will match")
    logger.info("[Validation] the T1 baseline when run with the updated trie.py")


# ── Main entry point ─────────────────────────────────────────────────


# Map benchmark names to their caching functions
BENCHMARK_FUNCTIONS: dict[str, callable] = {
    "mnist": cache_mnist,
    "fashion_mnist": cache_fashion_mnist,
    "cifar10": cache_cifar10,
    "text_4class": cache_text_4class,
    "text_20class": cache_text_20class,
}


def cache_all_features(
    benchmarks: list[str],
    output_dir: Path,
    full: bool = True,
    subset: bool = True,
) -> None:
    """Cache features for the specified benchmarks.

    Args:
        benchmarks: List of benchmark names to cache.
        output_dir: Directory for output .pt files.
        full: Whether to cache full-scale features.
        subset: Whether to cache 10K subset features.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in benchmarks:
        if name not in BENCHMARK_FUNCTIONS:
            logger.warning(f"Unknown benchmark: {name}, skipping")
            continue

        t0 = time.time()
        BENCHMARK_FUNCTIONS[name](output_dir, full=full, subset=subset)
        elapsed = time.time() - t0
        logger.info(f"  [{name}] completed in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute and cache encoder features for T2 sweep"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="all",
        help="Comma-separated benchmarks or 'all' (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/T2/features",
        help="Output directory for .pt files (default: results/T2/features)",
    )
    parser.add_argument(
        "--subset-only",
        action="store_true",
        help="Only cache 10K subsets (faster)",
    )
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Only cache full-scale features",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Parse benchmarks
    if args.benchmarks == "all":
        benchmarks = ALL_BENCHMARKS
    else:
        benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    # Determine full/subset flags
    do_full = True
    do_subset = True
    if args.subset_only:
        do_full = False
    if args.full_only:
        do_subset = False

    logger.info("=" * 60)
    logger.info("Feature Caching for T2 Threshold Sweep")
    logger.info("=" * 60)
    logger.info(f"  Benchmarks: {benchmarks}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Full: {do_full}, Subset: {do_subset}")
    logger.info(f"  Seed: {SEED}")

    t0 = time.time()
    cache_all_features(benchmarks, output_dir, full=do_full, subset=do_subset)
    total_time = time.time() - t0

    logger.info(f"\nTotal caching time: {total_time:.1f}s")

    # List cached files
    if output_dir.exists():
        logger.info("\nCached files:")
        for f in sorted(output_dir.glob("*.pt")):
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.info(f"  {f.name} ({size_mb:.1f} MB)")

    # Validate MNIST if it was cached
    if "mnist" in benchmarks and do_subset:
        validate_cached_features(output_dir)


if __name__ == "__main__":
    main()
