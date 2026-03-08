"""Performance benchmark for octonion multiplication throughput.

Standalone script measuring octonion_mul throughput at various batch sizes.
NOT a pytest test -- run with:
  uv run python tests/benchmarks/bench_multiplication.py

Reports ops/sec and per-element throughput on CPU.
"""

import time

import torch

from octonion._multiplication import octonion_mul

# Batch sizes to benchmark
BATCH_SIZES = [1, 100, 10_000, 1_000_000]
WARMUP_ITERS = 10
BENCH_ITERS = 100


def benchmark_batch(batch_size: int) -> dict:
    """Benchmark octonion multiplication for a given batch size.

    Returns dict with timing statistics.
    """
    a = torch.randn(batch_size, 8, dtype=torch.float64)
    b = torch.randn(batch_size, 8, dtype=torch.float64)

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = octonion_mul(a, b)

    # Benchmark
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(BENCH_ITERS):
            _ = octonion_mul(a, b)
        elapsed = time.perf_counter() - start

    total_ops = batch_size * BENCH_ITERS
    ops_per_sec = total_ops / elapsed
    time_per_iter = elapsed / BENCH_ITERS

    return {
        "batch_size": batch_size,
        "total_time_sec": elapsed,
        "time_per_iter_sec": time_per_iter,
        "ops_per_sec": ops_per_sec,
        "throughput_per_element_us": (elapsed / total_ops) * 1e6,
    }


def main() -> None:
    """Run benchmarks and print results."""
    print("=" * 78)
    print("Octonion Multiplication Benchmark (CPU, float64)")
    print(f"Warmup: {WARMUP_ITERS} iterations, Bench: {BENCH_ITERS} iterations")
    print("=" * 78)
    print()

    header = f"{'Batch Size':>12} | {'Total Time (s)':>14} | {'Ops/sec':>14} | {'us/element':>12}"
    print(header)
    print("-" * len(header))

    results = []
    for batch_size in BATCH_SIZES:
        result = benchmark_batch(batch_size)
        results.append(result)
        print(
            f"{result['batch_size']:>12,} | "
            f"{result['total_time_sec']:>14.4f} | "
            f"{result['ops_per_sec']:>14,.0f} | "
            f"{result['throughput_per_element_us']:>12.4f}"
        )

    print()
    print("Notes:")
    print("  - CPU only (GPU benchmarks are environment-dependent)")
    print("  - Uses torch.no_grad() for fair comparison")
    print("  - 'us/element' = microseconds per single octonion multiplication")
    print("  - Higher ops/sec and lower us/element is better")
    print()

    # Summary
    if len(results) >= 2:
        small = results[0]
        large = results[-1]
        speedup = (large["ops_per_sec"] / small["ops_per_sec"]) if small["ops_per_sec"] > 0 else 0
        print(f"Batch scaling: {small['batch_size']:,} -> {large['batch_size']:,} = {speedup:.1f}x throughput improvement")


if __name__ == "__main__":
    main()
