"""CIFAR benchmark data loading, network configuration, and reproduction reporting.

Provides benchmark-specific infrastructure for reproducing published CIFAR results
from Trabelsi et al. 2018 (complex) and Gaudet & Maida 2018 (quaternion), plus
training octonionic baselines alongside for early signal.

Published targets:
- Real:       CIFAR-10 6.37%, CIFAR-100 28.07%  (Gaudet & Maida 2018)
- Complex:    CIFAR-10 6.17%, CIFAR-100 26.36%  (Trabelsi et al. 2018)
- Quaternion: CIFAR-10 5.44%, CIFAR-100 26.01%  (Gaudet & Maida 2018)
- Octonion:   (no target -- first measurement)

Provides:
- build_cifar10_data: CIFAR-10 loaders with standard augmentation
- build_cifar100_data: CIFAR-100 loaders with standard augmentation
- cifar_network_config: Per-algebra Conv2D network config matching published architectures
- cifar_train_config: Training hyperparameters matching published papers
- reproduction_report: Structured comparison of our results vs published
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Subset

from octonion.baselines._config import AlgebraType, NetworkConfig, TrainConfig

# ── Published Results ──────────────────────────────────────────────

PUBLISHED_RESULTS: dict[str, dict[str, dict[str, Any]]] = {
    "cifar10": {
        "R": {
            "error_pct": 6.37,
            "std_pct": 0.17,
            "source": "Gaudet and Maida 2018",
            "notes": "Real-valued baseline, validates training infrastructure",
        },
        "C": {
            "error_pct": 6.17,
            "std_pct": 0.20,
            "source": "Trabelsi et al. 2018",
            "notes": "Deep complex network on CIFAR-10",
        },
        "H": {
            "error_pct": 5.44,
            "std_pct": 0.18,
            "source": "Gaudet and Maida 2018",
            "notes": "Quaternion network on CIFAR-10",
        },
        "O": {
            "error_pct": None,
            "std_pct": None,
            "source": "First measurement",
            "notes": "No published target -- early signal only",
        },
    },
    "cifar100": {
        "R": {
            "error_pct": 28.07,
            "std_pct": 0.30,
            "source": "Gaudet and Maida 2018",
            "notes": "Real-valued baseline on CIFAR-100",
        },
        "C": {
            "error_pct": 26.36,
            "std_pct": 0.25,
            "source": "Trabelsi et al. 2018",
            "notes": "Deep complex network on CIFAR-100",
        },
        "H": {
            "error_pct": 26.01,
            "std_pct": 0.22,
            "source": "Gaudet and Maida 2018",
            "notes": "Quaternion network on CIFAR-100",
        },
        "O": {
            "error_pct": None,
            "std_pct": None,
            "source": "First measurement",
            "notes": "No published target -- early signal only",
        },
    },
}


# ── Data Loading ───────────────────────────────────────────────────


def build_cifar10_data(
    batch_size: int,
    data_dir: str = "./data",
    num_workers: int = 4,
    val_fraction: float = 0.1,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], int, int, int]:
    """Load CIFAR-10 with standard augmentation for benchmark reproduction.

    Train augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip, Normalize.
    Val/test: Normalize only.
    10% of training data held out for validation.

    Input encoding note: Returns raw 3-channel images. Per-algebra input
    encoding (real->direct, complex->pair, quaternion->RGB, octonion->extended)
    is handled by the network's input layer, not the data loader.

    Args:
        batch_size: Batch size for all loaders.
        data_dir: Directory for dataset download/cache.
        num_workers: DataLoader worker threads.
        val_fraction: Fraction of training data for validation.

    Returns:
        Tuple of (train_loader, val_loader, test_loader,
                  input_dim, output_dim, input_channels).
        input_dim = 3 (CIFAR channels), output_dim = 10, input_channels = 3.
    """
    import torchvision
    import torchvision.transforms as T

    # CIFAR-10 normalization statistics
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # Load full training set and test set
    full_train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform,
    )

    # Split training into train/val
    n_total = len(full_train)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val

    # Use deterministic indices for reproducibility
    indices = list(range(n_total))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Create validation set with test transforms (no augmentation)
    val_set_no_aug = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=test_transform,
    )

    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(val_set_no_aug, val_indices)

    # Use "spawn" multiprocessing to avoid fork+CUDA deadlocks in containers
    mp_ctx = "spawn" if num_workers > 0 else None
    persist = num_workers > 0

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        multiprocessing_context=mp_ctx, persistent_workers=persist,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        multiprocessing_context=mp_ctx, persistent_workers=persist,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        multiprocessing_context=mp_ctx, persistent_workers=persist,
    )

    # input_dim=3 (RGB channels), output_dim=10 (classes), input_channels=3
    return train_loader, val_loader, test_loader, 3, 10, 3


def build_cifar100_data(
    batch_size: int,
    data_dir: str = "./data",
    num_workers: int = 4,
    val_fraction: float = 0.1,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], int, int, int]:
    """Load CIFAR-100 with standard augmentation for benchmark reproduction.

    Same augmentation strategy as CIFAR-10 but with CIFAR-100
    normalization statistics and 100 output classes.

    Args:
        batch_size: Batch size for all loaders.
        data_dir: Directory for dataset download/cache.
        num_workers: DataLoader worker threads.
        val_fraction: Fraction of training data for validation.

    Returns:
        Tuple of (train_loader, val_loader, test_loader,
                  input_dim, output_dim, input_channels).
        input_dim = 3 (CIFAR channels), output_dim = 100, input_channels = 3.
    """
    import torchvision
    import torchvision.transforms as T

    # CIFAR-100 normalization statistics
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    full_train = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform,
    )
    test_set = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform,
    )

    n_total = len(full_train)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val

    indices = list(range(n_total))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    val_set_no_aug = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=False, transform=test_transform,
    )

    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(val_set_no_aug, val_indices)

    # Use "spawn" multiprocessing to avoid fork+CUDA deadlocks in containers
    mp_ctx = "spawn" if num_workers > 0 else None
    persist = num_workers > 0

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        multiprocessing_context=mp_ctx, persistent_workers=persist,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        multiprocessing_context=mp_ctx, persistent_workers=persist,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        multiprocessing_context=mp_ctx, persistent_workers=persist,
    )

    return train_loader, val_loader, test_loader, 3, 100, 3


# ── Network Configuration ─────────────────────────────────────────


def cifar_network_config(
    algebra: AlgebraType,
    dataset: str = "cifar10",
) -> NetworkConfig:
    """Build Conv2D network config matching published CIFAR architectures.

    Uses a ResNet-style Conv2D architecture matching Gaudet & Maida 2018 and
    Trabelsi et al. 2018. AlgebraNetwork's conv2d topology provides residual
    blocks with skip connections across 3 stages, stride-2 downsampling at
    stage boundaries only:

    - depth=28 residual blocks distributed across 3 stages (9+9+10)
    - Stage filters: 16 -> 32 -> 64 (scaled by algebra multiplier)
    - Spatial: 32x32 -> 32x32 (stage1) -> 16x16 (stage2) -> 8x8 (stage3) -> GAP

    Input encoding per algebra:
    - Real:       standard 3-channel image, no encoding change
    - Complex:    encode RGB as complex pair. Following Trabelsi: zero-pad
                  3 channels to 4, split into 2 complex components (real+imag)
    - Quaternion: Following Gaudet: quaternion with real=0, imag=RGB.
                  q = 0 + R*i + G*j + B*k
    - Octonion:   Extend quaternion pattern: zero-pad remaining components.
                  o = 0 + R*e1 + G*e2 + B*e3 + 0*e4 + 0*e5 + 0*e6 + 0*e7

    Note: The input encoding is handled at the network level (input conv layer),
    not at the data loader level. The Conv2D input layer accepts raw RGB images
    and the algebra-specific conv layer handles the encoding.

    Args:
        algebra: Which algebra to use (R, C, H, O).
        dataset: "cifar10" or "cifar100".

    Returns:
        NetworkConfig for the specified algebra and dataset.

    Raises:
        ValueError: If dataset is not "cifar10" or "cifar100".
    """
    if dataset not in ("cifar10", "cifar100"):
        raise ValueError(f"Unknown dataset: {dataset!r}. Use 'cifar10' or 'cifar100'.")

    output_dim = 10 if dataset == "cifar10" else 100

    # ResNet-style architecture with 28 residual blocks across 3 stages.
    # Matches Gaudet & Maida 2018 (10+9+9 residual block architecture)
    # with stride-2 downsampling only at stage boundaries.
    return NetworkConfig(
        algebra=algebra,
        topology="conv2d",
        depth=28,
        base_hidden=16,
        activation="split_relu",
        output_projection="flatten",
        use_batchnorm=True,
        input_dim=3,  # CIFAR RGB channels
        output_dim=output_dim,
    )


# ── Training Configuration ────────────────────────────────────────


def cifar_train_config(dataset: str = "cifar10") -> TrainConfig:
    """Build training config matching published CIFAR hyperparameters.

    Matches Gaudet & Maida 2018 and Trabelsi et al. 2018 protocols:
    - SGD with Nesterov momentum 0.9
    - Step-decay LR: 0.01 warmup (10 epochs) -> 0.1 peak -> /10 at epoch 120 -> /10 at 150
    - 200 epochs
    - Weight decay 5e-4
    - Gradient norm clipping at 1.0
    - No AMP (float32 for reproduction fidelity)

    Args:
        dataset: "cifar10" or "cifar100" (same hyperparameters for both).

    Returns:
        TrainConfig matching published protocols.
    """
    return TrainConfig(
        epochs=200,
        lr=0.01,
        optimizer="sgd",
        scheduler="step_cifar",
        weight_decay=5e-4,
        early_stopping_patience=200 + 1,  # Effectively disabled
        warmup_epochs=10,
        use_amp=False,
        gradient_clip_norm=1.0,
        nesterov=True,
        checkpoint_every=25,
        seed=42,
        batch_size=128,
        lock_optimizer=False,
    )


# ── Reproduction Report ───────────────────────────────────────────


def reproduction_report(
    published: dict[str, dict[str, Any]],
    ours: dict[str, dict[str, Any]],
    output_path: str,
) -> dict[str, Any]:
    """Generate structured reproduction comparison document.

    Creates both JSON and Markdown reports comparing our results against
    published baselines. Each algebra gets a pass/fail verdict based on
    whether our error rate is within 1 standard deviation of the published result.

    Args:
        published: Dict keyed by algebra short name (R, C, H, O), each with:
            - error_pct: Published error rate (or None for no target)
            - std_pct: Published standard deviation (or None)
            - source: Citation string
            - notes: Additional notes
        ours: Dict keyed by algebra short name, each with:
            - error_pct: Our mean error rate
            - std_pct: Our standard deviation
            - param_count: Number of trainable parameters
            - seeds: Number of random seeds used
            - per_seed_errors: List of per-seed error rates

    Returns:
        Report dict with verdicts, comparisons, and metadata.

    Side effects:
        Saves {output_path}.json and {output_path}.md files.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    verdicts: dict[str, dict[str, Any]] = {}
    for alg in ours:
        our = ours[alg]
        pub = published.get(alg, {})

        pub_error = pub.get("error_pct")
        pub_std = pub.get("std_pct")
        our_error = our["error_pct"]
        our_std = our["std_pct"]

        if pub_error is not None and pub_std is not None:
            # Pass if our result is within 1 std of published
            # Use the larger of published std and our std for the comparison
            comparison_std = max(pub_std, our_std) if our_std is not None else pub_std
            diff = abs(our_error - pub_error)
            within_1std = diff <= comparison_std
            verdict = "PASS" if within_1std else "FAIL"
        else:
            # No target to compare against (e.g., octonion)
            verdict = "N/A (no published target)"
            within_1std = None
            diff = None
            comparison_std = None

        verdicts[alg] = {
            "algebra": alg,
            "published_error_pct": pub_error,
            "published_std_pct": pub_std,
            "our_error_pct": our_error,
            "our_std_pct": our_std,
            "param_count": our.get("param_count"),
            "diff_pct": diff,
            "comparison_std": comparison_std,
            "within_1std": within_1std,
            "verdict": verdict,
            "source": pub.get("source", "N/A"),
            "notes": pub.get("notes", ""),
        }

    report = {
        "verdicts": verdicts,
        "overall_pass": all(
            v["verdict"] in ("PASS", "N/A (no published target)")
            for v in verdicts.values()
        ),
        "n_algebras": len(ours),
        "algebras_tested": list(ours.keys()),
    }

    # Save JSON
    json_path = str(output) + ".json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save Markdown
    md_path = str(output) + ".md"
    md_lines = [
        "# Benchmark Reproduction Report",
        "",
        "## Summary",
        "",
        f"**Overall:** {'PASS' if report['overall_pass'] else 'FAIL'}",
        f"**Algebras tested:** {', '.join(report['algebras_tested'])}",
        "",
        "## Results",
        "",
        "| Algebra | Published Error% | Our Error% (mean +/- std) | Param Count | Verdict |",
        "|---------|-----------------|---------------------------|-------------|---------|",
    ]

    for alg in ours:
        v = verdicts[alg]
        pub_str = (
            f"{v['published_error_pct']:.2f}% +/- {v['published_std_pct']:.2f}%"
            if v["published_error_pct"] is not None
            else "N/A"
        )
        our_str = f"{v['our_error_pct']:.2f}%"
        if v["our_std_pct"] is not None:
            our_str += f" +/- {v['our_std_pct']:.2f}%"
        param_str = f"{v['param_count']:,}" if v["param_count"] is not None else "N/A"
        md_lines.append(
            f"| {alg} | {pub_str} | {our_str} | {param_str} | {v['verdict']} |"
        )

    md_lines.extend([
        "",
        "## Details",
        "",
    ])

    for alg in ours:
        v = verdicts[alg]
        md_lines.append(f"### {alg} ({v['source']})")
        md_lines.append("")
        if v["published_error_pct"] is not None:
            md_lines.append(f"- Published: {v['published_error_pct']:.2f}% +/- {v['published_std_pct']:.2f}%")
            md_lines.append(f"- Ours: {v['our_error_pct']:.2f}% +/- {v['our_std_pct']:.2f}%")
            md_lines.append(f"- Difference: {v['diff_pct']:.2f}%")
            md_lines.append(f"- Comparison std: {v['comparison_std']:.2f}%")
            md_lines.append(f"- Within 1 std: {'Yes' if v['within_1std'] else 'No'}")
        else:
            md_lines.append(f"- Ours: {v['our_error_pct']:.2f}%")
            md_lines.append("- No published target for comparison")
        md_lines.append(f"- {v['notes']}")
        if "per_seed_errors" in ours[alg]:
            md_lines.append(f"- Per-seed errors: {ours[alg]['per_seed_errors']}")
        md_lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    return report
