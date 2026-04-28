#!/usr/bin/env python
"""Phase 5: Optimization Landscape experiment orchestration.

Main entry point for running the landscape experiment matrix.
Ties together task generation, training, Hessian analysis, curvature
measurement, and gate evaluation.

Usage:
    # Full experiment (WARNING: 200+ GPU hours)
    docker compose run --rm dev uv run python scripts/run_landscape.py

    # Smoke test (2 algebras, 1 task, 2 seeds, ~5 min)
    docker compose run --rm dev uv run python scripts/run_landscape.py --smoke

    # Single task
    docker compose run --rm dev uv run python scripts/run_landscape.py --tasks algebra_native_single

    # Single optimizer
    docker compose run --rm dev uv run python scripts/run_landscape.py --optimizers adam

    # Reduced seeds
    docker compose run --rm dev uv run python scripts/run_landscape.py --n-seeds 5

    # Resume interrupted run
    docker compose run --rm dev uv run python scripts/run_landscape.py --resume

    # CPU-only
    docker compose run --rm dev uv run python scripts/run_landscape.py --smoke --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from octonion.baselines._config import AlgebraType
from octonion.landscape._experiment import LandscapeConfig, run_landscape_experiment
from octonion.landscape._gate import evaluate_gate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Valid choices ──────────────────────────────────────────────────────────

ALL_TASKS = [
    "algebra_native_single",
    "algebra_native_multi",
    "cross_product_3d",
    "cross_product_7d_noise0",
    "cross_product_7d_noise5",
    "cross_product_7d_noise15",
    "cross_product_7d_noise30",
    "sinusoidal",
    "classification",
]

ALL_OPTIMIZERS = ["sgd", "adam", "riemannian_adam", "lbfgs", "shampoo"]

ALGEBRA_MAP = {
    "real": AlgebraType.REAL,
    "complex": AlgebraType.COMPLEX,
    "quaternion": AlgebraType.QUATERNION,
    "octonion": AlgebraType.OCTONION,
    "phm8": AlgebraType.PHM8,
    "r8_dense": AlgebraType.R8_DENSE,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 5: Optimization Landscape experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run minimal smoke test: 1 task, 2 algebras, 1 optimizer, 2 seeds, 10 epochs",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=ALL_TASKS,
        default=None,
        help="Tasks to run (default: all 9)",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=ALL_OPTIMIZERS,
        default=None,
        help="Optimizers to use (default: all 5)",
    )
    parser.add_argument(
        "--algebras",
        nargs="+",
        choices=list(ALGEBRA_MAP.keys()),
        default=None,
        help="Algebra types to compare (default: all 6)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help="Number of seeds (default: 20, or 2 for --smoke)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs (default: 100, or 10 for --smoke)",
    )
    parser.add_argument(
        "--base-hidden",
        type=int,
        default=None,
        help="Base hidden width in octonionic units (default: 16, or 4 for --smoke)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previously saved results (skip existing)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/landscape",
        help="Output directory (default: results/landscape)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: 'cuda' or 'cpu' (default: cuda)",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="Training samples per task (default: 50000, or 1000 for --smoke)",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=None,
        help="Test samples per task (default: 10000, or 200 for --smoke)",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> LandscapeConfig:
    """Build LandscapeConfig from parsed CLI arguments.

    Args:
        args: Parsed argparse namespace.

    Returns:
        Configured LandscapeConfig instance.
    """
    if args.smoke:
        # Minimal configuration for quick validation
        config = LandscapeConfig(
            tasks=args.tasks or ["algebra_native_single"],
            algebras=(
                [ALGEBRA_MAP[a] for a in args.algebras]
                if args.algebras
                else [AlgebraType.REAL, AlgebraType.OCTONION]
            ),
            optimizers=args.optimizers or ["adam"],
            seeds=list(range(args.n_seeds or 2)),
            epochs=args.epochs or 10,
            base_hidden=args.base_hidden or 4,
            output_dir=args.output_dir,
            device=args.device,
            n_train=args.n_train or 1000,
            n_test=args.n_test or 200,
            hessian_seeds=[0],
            hessian_checkpoints=[0.0, 1.0],
            n_curvature_directions=5,
        )
    else:
        # Full experiment configuration
        config = LandscapeConfig(
            tasks=args.tasks or ALL_TASKS,
            algebras=(
                [ALGEBRA_MAP[a] for a in args.algebras]
                if args.algebras
                else [
                    AlgebraType.REAL, AlgebraType.COMPLEX,
                    AlgebraType.QUATERNION, AlgebraType.OCTONION,
                    AlgebraType.PHM8, AlgebraType.R8_DENSE,
                ]
            ),
            optimizers=args.optimizers or ALL_OPTIMIZERS,
            seeds=list(range(args.n_seeds or 20)),
            epochs=args.epochs or 100,
            base_hidden=args.base_hidden or 16,
            output_dir=args.output_dir,
            device=args.device,
            n_train=args.n_train or 50_000,
            n_test=args.n_test or 10_000,
        )

    return config


def _build_gate_input(
    results: dict,
    config: LandscapeConfig,
) -> dict | None:
    """Build gate evaluation input from experiment results.

    Extracts O vs R8_DENSE final validation losses per task,
    collecting across all optimizers and seeds.

    Args:
        results: Nested results dict from run_landscape_experiment.
        config: LandscapeConfig used for the experiment.

    Returns:
        Gate input dict or None if required algebras not present.
    """
    has_o = AlgebraType.OCTONION in config.algebras
    has_r8d = AlgebraType.R8_DENSE in config.algebras

    if not (has_o and has_r8d):
        logger.info(
            "Gate evaluation requires both OCTONION and R8_DENSE algebras. "
            "Skipping gate evaluation."
        )
        return None

    gate_input: dict = {}

    for task_name, task_data in results.items():
        o_losses: list[float] = []
        r8d_losses: list[float] = []
        initial_loss = float("inf")

        for opt_name, opt_data in task_data.items():
            for alg_name, alg_data in opt_data.items():
                for seed, run_result in alg_data.items():
                    if isinstance(run_result, dict):
                        final_loss = run_result.get("final_val_loss", float("inf"))
                        if alg_name == "O":
                            o_losses.append(final_loss)
                            # Use first val_loss as initial_loss proxy
                            vl = run_result.get("val_losses", [])
                            if vl and vl[0] < initial_loss:
                                initial_loss = vl[0]
                        elif alg_name == "R8D":
                            r8d_losses.append(final_loss)

        if o_losses and r8d_losses:
            gate_input[task_name] = {
                "O": {
                    "final_val_losses": o_losses,
                    "initial_loss": initial_loss if initial_loss < float("inf") else 1.0,
                },
                "R8_DENSE": {
                    "final_val_losses": r8d_losses,
                },
            }

    return gate_input if gate_input else None


def main() -> None:
    """Main entry point for landscape experiments."""
    args = parse_args()
    config = build_config(args)

    # Print configuration
    total_runs = (
        len(config.tasks) * len(config.optimizers)
        * len(config.algebras) * len(config.seeds)
    )
    print(f"\n{'='*60}")
    print("  Phase 5: Optimization Landscape Experiment")
    print(f"{'='*60}")
    print(f"  Tasks:      {len(config.tasks)} ({', '.join(config.tasks)})")
    print(f"  Algebras:   {len(config.algebras)} ({', '.join(a.short_name for a in config.algebras)})")
    print(f"  Optimizers: {len(config.optimizers)} ({', '.join(config.optimizers)})")
    print(f"  Seeds:      {len(config.seeds)}")
    print(f"  Total runs: {total_runs}")
    print(f"  Epochs:     {config.epochs}")
    print(f"  Hidden:     {config.base_hidden}")
    print(f"  Device:     {config.device}")
    print(f"  Output:     {config.output_dir}")
    if args.smoke:
        print("  Mode:       SMOKE TEST")
    if args.resume:
        print("  Resume:     YES (skipping existing results)")
    print(f"{'='*60}\n")

    # Run experiment
    start = time.time()
    results = run_landscape_experiment(config)
    elapsed = time.time() - start

    # Count completed vs failed
    n_completed = 0
    n_failed = 0
    for task_data in results.values():
        for opt_data in task_data.values():
            for alg_data in opt_data.values():
                for seed, run_result in alg_data.items():
                    if isinstance(run_result, dict):
                        if "error" in run_result:
                            n_failed += 1
                        else:
                            n_completed += 1

    # Print summary
    print(f"\n{'='*60}")
    print("  EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time:    {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Runs completed: {n_completed}")
    print(f"  Runs failed:    {n_failed}")
    print(f"  Results saved:  {config.output_dir}")

    # Gate evaluation
    gate_input = _build_gate_input(results, config)
    if gate_input:
        gate_result = evaluate_gate(gate_input)
        print(f"\n  Gate verdict: {gate_result['verdict'].value}")
        print(f"  {gate_result['summary']}")

        # Save gate result
        gate_output_path = os.path.join(config.output_dir, "gate_verdict.json")
        os.makedirs(os.path.dirname(gate_output_path), exist_ok=True)
        gate_serializable = {
            "verdict": gate_result["verdict"].value,
            "summary": gate_result["summary"],
            "per_task": gate_result["per_task"],
        }
        with open(gate_output_path, "w") as f:
            json.dump(gate_serializable, f, indent=2)
        print(f"  Gate result:   {gate_output_path}")
    else:
        print("\n  Gate evaluation: SKIPPED (need O and R8D algebras)")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
