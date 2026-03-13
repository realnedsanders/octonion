#!/usr/bin/env python3
"""CIFAR-10 benchmark reproduction for all 4 algebras (R, C, H, O).

Orchestrates full CIFAR-10 training using the Phase 3 baseline infrastructure:
- AlgebraNetwork conv2d topology (ResNet-style, depth=28)
- Parameter-matched architectures across R, C, H, O
- SGD with cosine LR for 200 epochs
- 3 random seeds for mean +/- std

Generates structured reproduction reports comparing our results against
published targets from Trabelsi et al. 2018 (C) and Gaudet & Maida 2018 (R, H).

Usage:
    # Full reproduction (all 4 algebras, 3 seeds, 200 epochs)
    docker compose run --rm dev uv run python scripts/run_cifar_reproduction.py --seeds 3

    # Run specific algebras (e.g., just R and H first since they're fastest)
    docker compose run --rm dev uv run python scripts/run_cifar_reproduction.py --algebras R H --seeds 3

    # Quick validation (2 epochs)
    docker compose run --rm dev uv run python scripts/run_cifar_reproduction.py --epochs 2 --seeds 1
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from functools import partial
from pathlib import Path

import numpy as np

from octonion.baselines._benchmarks import (
    PUBLISHED_RESULTS,
    build_cifar10_data,
    cifar_train_config,
    reproduction_report,
)
from octonion.baselines._comparison import run_comparison
from octonion.baselines._config import AlgebraType, ComparisonConfig

logger = logging.getLogger(__name__)

ALGEBRA_MAP = {
    "R": AlgebraType.REAL,
    "C": AlgebraType.COMPLEX,
    "H": AlgebraType.QUATERNION,
    "O": AlgebraType.OCTONION,
}


def _sigint_handler(signum: int, frame: object) -> None:
    """Handle SIGINT gracefully with checkpoint info."""
    print(
        "\n\nSIGINT received. The trainer saves checkpoints on interrupt.\n"
        "To resume, rerun the same command -- training resumes from the last checkpoint.\n"
        "Exiting.",
        flush=True,
    )
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CIFAR-10 benchmark reproduction for R, C, H, O algebras",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of random seeds per algebra",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for training (cuda or cpu)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for CIFAR-10 dataset download/cache",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Base directory for experiment outputs",
    )
    parser.add_argument(
        "--ref-hidden",
        type=int,
        default=16,
        help="Base hidden width (filter count) for reference model",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--algebras",
        nargs="+",
        default=["O", "R", "C", "H"],
        choices=["R", "C", "H", "O"],
        help="Which algebras to train (default: all four)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs (default: 200 from cifar_train_config)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=28,
        help="ResNet depth (number of residual blocks)",
    )
    parser.add_argument(
        "--no-match-params",
        action="store_true",
        default=False,
        help="Use same-width mode (all algebras get same base_filters). "
             "Matches the protocol in Gaudet & Maida 2018 / Trabelsi et al. 2018 "
             "where H had fewer params but same architecture width as R.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=False,
        help="Enable automatic mixed precision (AMP) for faster training. "
             "BN whitening is protected with float32 casting so AMP is safe "
             "for all four algebras. Default: off (float32 for reproduction fidelity).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Enable torch.compile with inductor backend (experimental on ROCm). "
             "Falls back to eager mode if compilation fails. Default: off.",
    )
    return parser.parse_args()


def main() -> None:
    """Run CIFAR-10 benchmark reproduction."""
    args = parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set up SIGINT handler
    signal.signal(signal.SIGINT, _sigint_handler)

    # Check GPU availability
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "CUDA not available. Falling back to CPU. "
            "Training will be very slow but should work for validation."
        )
        args.device = "cpu"

    if args.device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Configure experiment
    # OCTONION first: it has multiplier=1, so ref_hidden=base_hidden directly.
    # This gives base_filters=16, matching Gaudet & Maida 2018 architecture.
    # The comparison runner uses the first algebra as reference for param matching.
    algebras = [ALGEBRA_MAP[a] for a in args.algebras]

    train_config = cifar_train_config("cifar10")
    if args.epochs is not None:
        train_config.epochs = args.epochs
    if args.use_amp:
        train_config.use_amp = True
    if args.compile:
        train_config.use_compile = True

    config = ComparisonConfig(
        task="cifar10",
        algebras=algebras,
        seeds=args.seeds,
        train_config=train_config,
        output_dir=args.output_dir,
    )

    # Build partial data loader with custom data_dir and num_workers
    data_fn = partial(
        build_cifar10_data,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
    )

    # Network config overrides for conv2d CIFAR reproduction
    network_overrides = {
        "topology": "conv2d",
        "depth": args.depth,
        "ref_hidden": args.ref_hidden,
        "match_params": not args.no_match_params,
    }

    logger.info("=" * 60)
    logger.info("CIFAR-10 Benchmark Reproduction")
    logger.info("=" * 60)
    logger.info(f"Algebras: {[a.short_name for a in algebras]}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Epochs: {train_config.epochs}")
    logger.info(f"Depth: {args.depth}")
    logger.info(f"LR: {train_config.lr}, Optimizer: {train_config.optimizer}")
    logger.info(f"Scheduler: {train_config.scheduler}")
    logger.info(f"Batch size: {train_config.batch_size}")
    logger.info(f"Ref hidden: {args.ref_hidden}")
    logger.info(f"Match params: {not args.no_match_params}")
    logger.info(f"AMP: {train_config.use_amp}")
    logger.info(f"torch.compile: {getattr(train_config, 'use_compile', False)}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info("=" * 60)

    # Run the comparison
    task_name = "cifar10_reproduction"
    report = run_comparison(
        task_name=task_name,
        build_data_fn=data_fn,
        config=config,
        device=args.device,
        network_config_overrides=network_overrides,
    )

    # Generate reproduction report
    logger.info("Generating reproduction report...")

    # Extract per-algebra results from report.per_run
    algebra_results: dict[str, list[dict]] = {}
    for run in report.per_run:
        alg = run["algebra"]
        algebra_results.setdefault(alg, []).append(run)

    ours: dict[str, dict] = {}
    for alg_name, runs in algebra_results.items():
        errors = [(1 - r["metrics"]["best_val_acc"]) * 100 for r in runs]
        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0

        # Get param count from report
        param_count = report.param_counts.get(alg_name)

        ours[alg_name] = {
            "error_pct": mean_error,
            "std_pct": std_error,
            "param_count": param_count,
            "seeds": len(runs),
            "per_seed_errors": [round(e, 2) for e in errors],
        }

    report_path = Path(args.output_dir) / task_name / "reproduction_report"
    repro_result = reproduction_report(
        published=PUBLISHED_RESULTS["cifar10"],
        ours=ours,
        output_path=str(report_path),
    )

    # Print summary
    print("\n" + "=" * 70)
    print("  CIFAR-10 BENCHMARK REPRODUCTION RESULTS")
    print("=" * 70)
    print(
        f"\n{'Algebra':<10} {'Published':<20} {'Ours':<25} {'Verdict':<10}"
    )
    print("-" * 70)

    for alg_name in ["R", "C", "H", "O"]:
        if alg_name not in repro_result["verdicts"]:
            continue
        v = repro_result["verdicts"][alg_name]
        pub_str = (
            f"{v['published_error_pct']:.2f}% +/- {v['published_std_pct']:.2f}%"
            if v["published_error_pct"] is not None
            else "N/A"
        )
        our_str = f"{v['our_error_pct']:.2f}%"
        if v["our_std_pct"] is not None and v["our_std_pct"] > 0:
            our_str += f" +/- {v['our_std_pct']:.2f}%"
        print(f"{alg_name:<10} {pub_str:<20} {our_str:<25} {v['verdict']:<10}")

    overall = "PASS" if repro_result["overall_pass"] else "FAIL"
    print("-" * 70)
    print(f"\nOverall: {overall}")
    print(f"\nReport saved to: {report_path}.md")
    print(f"JSON report: {report_path}.json")
    print("=" * 70)

    # Exit with appropriate code
    sys.exit(0 if repro_result["overall_pass"] else 1)


if __name__ == "__main__":
    main()
