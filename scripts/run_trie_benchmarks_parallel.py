"""Run all trie benchmarks in parallel subprocesses.

Launches Fashion-MNIST, CIFAR-10, and text benchmarks as independent
processes, utilizing multiple CPU cores. Each benchmark runs in its own
process with its own Python interpreter, avoiding GIL contention.

Usage:
    docker compose run --rm dev uv run python scripts/run_trie_benchmarks_parallel.py
    docker compose run --rm dev uv run python scripts/run_trie_benchmarks_parallel.py --workers 6
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_benchmark(name: str, cmd: list[str]) -> tuple[str, float, int]:
    """Run a benchmark script as a subprocess.

    Returns:
        (name, elapsed_seconds, return_code)
    """
    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=3600,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error(f"  {name} FAILED (exit {result.returncode})")
        if result.stderr:
            logger.error(f"  stderr: {result.stderr[-500:]}")
    else:
        logger.info(f"  {name} completed in {elapsed:.0f}s")

    return name, elapsed, result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trie benchmarks in parallel")
    parser.add_argument("--workers", type=int, default=3, help="Max parallel benchmarks")
    parser.add_argument("--n-train", type=int, default=10000, help="Training samples per benchmark")
    parser.add_argument("--n-test", type=int, default=2000, help="Test samples per benchmark")
    parser.add_argument("--cnn-epochs", type=int, default=5, help="CNN encoder training epochs")
    parser.add_argument("--trie-epochs", type=int, default=3, help="Trie training epochs")
    parser.add_argument("--cifar-encoder", type=str, default="all", help="CIFAR encoder (2layer/4layer/resnet8/all)")
    args = parser.parse_args()

    python = sys.executable
    benchmarks = {
        "Fashion-MNIST": [
            python, "scripts/run_trie_fashion_mnist.py",
            "--n-train", str(args.n_train),
            "--n-test", str(args.n_test),
            "--cnn-epochs", str(args.cnn_epochs),
            "--epochs", str(args.trie_epochs),
        ],
        "CIFAR-10": [
            python, "scripts/run_trie_cifar10.py",
            "--encoder", args.cifar_encoder,
            "--n-train", str(args.n_train),
            "--n-test", str(args.n_test),
            "--cnn-epochs", str(args.cnn_epochs * 2),  # CIFAR needs more CNN epochs
            "--epochs", str(args.trie_epochs),
        ],
        "Text": [
            python, "scripts/run_trie_text.py",
            "--mode", "both",
            "--epochs", str(args.trie_epochs),
        ],
    }

    logger.info("=" * 60)
    logger.info("Parallel Trie Benchmarks")
    logger.info("=" * 60)
    logger.info(f"  Workers: {args.workers}")
    logger.info(f"  Benchmarks: {', '.join(benchmarks.keys())}")
    logger.info(f"  Train: {args.n_train}, Test: {args.n_test}")
    logger.info("")

    t0 = time.time()
    results = {}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_benchmark, name, cmd): name
            for name, cmd in benchmarks.items()
        }

        for future in as_completed(futures):
            name = futures[future]
            try:
                name, elapsed, rc = future.result()
                results[name] = {"elapsed": elapsed, "return_code": rc}
            except Exception as e:
                logger.error(f"  {name} raised exception: {e}")
                results[name] = {"elapsed": 0, "return_code": -1, "error": str(e)}

    total = time.time() - t0

    logger.info("")
    logger.info("=" * 60)
    logger.info("Parallel Execution Summary")
    logger.info("=" * 60)
    sequential_total = sum(r["elapsed"] for r in results.values())
    speedup = sequential_total / max(total, 1)
    logger.info(f"  Wall time:       {total:.0f}s")
    logger.info(f"  Sequential sum:  {sequential_total:.0f}s")
    logger.info(f"  Speedup:         {speedup:.1f}x")
    logger.info("")
    for name, r in results.items():
        status = "OK" if r["return_code"] == 0 else f"FAILED ({r['return_code']})"
        logger.info(f"  {name:20s}: {r['elapsed']:6.0f}s  {status}")

    # Run summary if all succeeded
    if all(r["return_code"] == 0 for r in results.values()):
        logger.info("")
        logger.info("Running cross-benchmark summary...")
        subprocess.run([python, "scripts/run_trie_benchmark_summary.py"], timeout=60)


if __name__ == "__main__":
    main()
