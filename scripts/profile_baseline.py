"""GPU profiling script for octonion baseline forward+backward pass bottleneck analysis.

Profiles all 4 algebras (Real, Complex, Quaternion, Octonion) using
torch.profiler on CIFAR-like inputs (batch=128, 3x32x32) with the
conv2d topology matching benchmark configurations.

Usage:
    docker compose run --rm dev uv run python scripts/profile_baseline.py
    docker compose run --rm dev uv run python scripts/profile_baseline.py --algebras octonion quaternion
    docker compose run --rm dev uv run python scripts/profile_baseline.py --device cpu --batch-size 32

Outputs:
    - Per-algebra profiler tables (top-20 ops by CUDA time)
    - Peak memory usage per algebra
    - Chrome trace files in profile_traces/
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile forward+backward pass for all 4 algebra types on CIFAR-like inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run profiling on ('cuda' or 'cpu'). Falls back to cpu if cuda unavailable.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for profiling inputs.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations before profiling.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=5,
        help="Number of profiled iterations per algebra.",
    )
    parser.add_argument(
        "--algebras",
        nargs="+",
        default=["real", "complex", "quaternion", "octonion"],
        choices=["real", "complex", "quaternion", "octonion"],
        help="Algebras to profile.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="profile_traces",
        help="Directory for Chrome trace output files.",
    )
    parser.add_argument(
        "--no-traces",
        action="store_true",
        help="Skip exporting Chrome traces (faster, less disk usage).",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    """Resolve device, falling back to CPU if CUDA is unavailable."""
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            warnings.warn(
                f"CUDA requested but not available. Falling back to CPU. "
                f"(Requested: {requested!r})",
                UserWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
    return torch.device(requested)


def get_profiler_activities(device: torch.device) -> list[ProfilerActivity]:
    """Get profiler activities based on device."""
    activities = [ProfilerActivity.CPU]
    if str(device).startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)
    return activities


def _algebra_name_to_type(name: str):
    """Convert string algebra name to AlgebraType enum."""
    from octonion.baselines._config import AlgebraType

    mapping = {
        "real": AlgebraType.REAL,
        "complex": AlgebraType.COMPLEX,
        "quaternion": AlgebraType.QUATERNION,
        "octonion": AlgebraType.OCTONION,
    }
    return mapping[name.lower()]


def build_profile_model(algebra_name: str, device: torch.device) -> nn.Module:
    """Build a CIFAR-scale model for the given algebra type.

    Uses cifar_network_config() to ensure profiling uses the exact same
    architecture as the actual benchmark runs.

    Args:
        algebra_name: One of 'real', 'complex', 'quaternion', 'octonion'.
        device: Target device.

    Returns:
        AlgebraNetwork model on the specified device.
    """
    from octonion.baselines._benchmarks import cifar_network_config
    from octonion.baselines._network import AlgebraNetwork

    algebra = _algebra_name_to_type(algebra_name)
    config = cifar_network_config(algebra, dataset="cifar10")
    model = AlgebraNetwork(config).to(device)
    return model


def run_warmup(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    n_warmup: int,
    loss_fn: nn.Module,
    device: torch.device,
) -> None:
    """Run warmup iterations to stabilize GPU state.

    Args:
        model: Model to warm up.
        inputs: Input tensor.
        targets: Target labels.
        n_warmup: Number of warmup iterations.
        loss_fn: Loss function.
        device: Device (used for cuda sync).
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for _ in range(n_warmup):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

    # Synchronize GPU to ensure all warmup ops complete
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def profile_algebra(
    algebra_name: str,
    device: torch.device,
    batch_size: int,
    n_warmup: int,
    n_iters: int,
    activities: list[ProfilerActivity],
    output_dir: Path,
    export_traces: bool,
) -> dict[str, float]:
    """Profile forward+backward pass for a single algebra.

    Args:
        algebra_name: Algebra identifier string.
        device: Target device.
        batch_size: Input batch size.
        n_warmup: Number of warmup iterations.
        n_iters: Number of profiled iterations.
        activities: Profiler activities to capture.
        output_dir: Directory for trace output.
        export_traces: Whether to export Chrome trace files.

    Returns:
        Dict with timing and memory stats for this algebra.
    """
    print(f"\n{'='*60}")
    print(f"Profiling: {algebra_name.upper()}")
    print(f"{'='*60}")

    # Build model
    model = build_profile_model(algebra_name, device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # CIFAR-like synthetic inputs: [B, 3, 32, 32]
    inputs = torch.randn(batch_size, 3, 32, 32, device=device)
    targets = torch.randint(0, 10, (batch_size,), device=device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Warmup
    print(f"Running {n_warmup} warmup iterations...")
    run_warmup(model, inputs, targets, n_warmup, loss_fn, device)

    # Reset CUDA memory stats before profiling
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    # Profiled iterations
    print(f"Running {n_iters} profiled iterations...")
    trace_path = str(output_dir / f"trace_{algebra_name}.json") if export_traces else None

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(output_dir / algebra_name)
        ) if export_traces else None,
    ) as prof:
        model.train()
        for step in range(n_iters):
            with record_function(f"step_{step}"):
                optimizer.zero_grad(set_to_none=True)

                with record_function("forward"):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)

                with record_function("backward"):
                    loss.backward()

                optimizer.step()

            prof.step()

    # Export Chrome trace
    if export_traces and trace_path:
        try:
            prof.export_chrome_trace(trace_path)
            print(f"Chrome trace exported: {trace_path}")
        except Exception as e:
            print(f"Warning: Chrome trace export failed: {e}")

    # Print top-20 ops by CUDA (or CPU) time
    sort_key = "cuda_time_total" if ProfilerActivity.CUDA in activities else "cpu_time_total"
    print(f"\nTop-20 operations (sorted by {sort_key}):")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))

    # Extract summary stats
    stats: dict[str, float] = {"n_params": float(n_params)}

    key_avgs = prof.key_averages()
    total_cuda_time_us = sum(
        item.cuda_time_total for item in key_avgs
        if hasattr(item, "cuda_time_total")
    )
    total_cpu_time_us = sum(
        item.cpu_time_total for item in key_avgs
        if hasattr(item, "cpu_time_total")
    )
    stats["total_cuda_time_ms"] = total_cuda_time_us / 1000.0
    stats["total_cpu_time_ms"] = total_cpu_time_us / 1000.0
    stats["per_iter_cuda_ms"] = stats["total_cuda_time_ms"] / max(n_iters, 1)
    stats["per_iter_cpu_ms"] = stats["total_cpu_time_ms"] / max(n_iters, 1)

    if str(device).startswith("cuda"):
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
        stats["peak_memory_mb"] = peak_mb
        print(f"Peak GPU memory: {peak_mb:.1f} MB")

    return stats


def print_summary_table(results: dict[str, dict[str, float]]) -> None:
    """Print a summary comparison table for all profiled algebras.

    Args:
        results: Dict mapping algebra name to stats dict.
    """
    print("\n" + "="*70)
    print("PROFILING SUMMARY")
    print("="*70)

    headers = ["Algebra", "Params", "CUDA ms/iter", "CPU ms/iter", "Peak MB"]
    col_widths = [12, 12, 14, 14, 10]
    header_row = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * len(header_row))

    for algebra, stats in results.items():
        n_params = stats.get("n_params", 0)
        cuda_ms = stats.get("per_iter_cuda_ms", 0.0)
        cpu_ms = stats.get("per_iter_cpu_ms", 0.0)
        peak_mb = stats.get("peak_memory_mb", 0.0)

        row = "  ".join([
            algebra.upper().ljust(col_widths[0]),
            f"{n_params:,.0f}".ljust(col_widths[1]),
            f"{cuda_ms:.2f}".ljust(col_widths[2]),
            f"{cpu_ms:.2f}".ljust(col_widths[3]),
            f"{peak_mb:.1f}".ljust(col_widths[4]),
        ])
        print(row)

    print("="*70)


def main() -> int:
    """Main entry point for the profiling script.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    args = parse_args()

    # Resolve device
    device = resolve_device(args.device)
    print(f"Profiling device: {device}")
    if str(device).startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    # Setup output directory for traces
    output_dir = Path(args.output_dir)
    if not args.no_traces:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Chrome traces will be saved to: {output_dir}/")

    # Profiler activities
    activities = get_profiler_activities(device)
    print(f"Profiler activities: {[a.name for a in activities]}")
    print(f"Algebras: {args.algebras}")
    print(f"Batch size: {args.batch_size}, Warmup: {args.warmup}, Iters: {args.iters}")

    # Profile each algebra
    results: dict[str, dict[str, float]] = {}
    for algebra_name in args.algebras:
        try:
            stats = profile_algebra(
                algebra_name=algebra_name,
                device=device,
                batch_size=args.batch_size,
                n_warmup=args.warmup,
                n_iters=args.iters,
                activities=activities,
                output_dir=output_dir,
                export_traces=not args.no_traces,
            )
            results[algebra_name] = stats
        except Exception as exc:
            print(f"ERROR profiling {algebra_name}: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            # Continue with remaining algebras

    if results:
        print_summary_table(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
