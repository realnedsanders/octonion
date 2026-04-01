---
phase: T1-benchmark-generalization
plan: 05
type: summary
status: complete
---

# T1-05 Summary: Cross-Benchmark Results

## Completed Tasks

### Task 1: Cross-benchmark summary script
- Created `scripts/run_trie_benchmark_summary.py` (563 lines)
- Loads results from all benchmarks, produces comparison tables and gap analysis

### Task 2: Human verification (APPROVED)
- Results reviewed and approved by human
- Full-scale runs completed for MNIST (60K), Fashion-MNIST (60K), CIFAR-10 (10K, all encoders), Text (full)

## Results

### Full-Scale Benchmark Results

| Benchmark | Scale | Trie | kNN(k=5) | Gap |
|-----------|-------|------|----------|-----|
| MNIST | 60K, CNN-8D | 95.3% | 98.2% | -2.9pp |
| Fashion-MNIST | 60K, CNN-8D | 80.3% | 87.1% | -6.8pp |
| Text 4-class | Full, TF-IDF-8D | 79.4% | 85.6% | -6.2pp |
| CIFAR-10 (ResNet-8) | 10K, CNN-8D | 64.0% | 73.0% | -9.0pp |
| CIFAR-10 (4-layer) | 10K, CNN-8D | 60.6% | 69.6% | -9.0pp |
| CIFAR-10 (2-layer) | 10K, CNN-8D | 45.1% | 50.4% | -5.3pp |
| Text 20-class | Full, TF-IDF-8D | 32.5% | ~42.0% | - |

### Capacity Scaling (structured data)
- 50 classes in 8D: 88.6% accuracy (44x chance)
- Graceful degradation from 98.5% at 8 classes to 88.6% at 50
- Depth handles N >> 8 classes without architectural changes

### Training Noise Investigation
- Global noise=0.05: helps MNIST (+0.3pp), hurts Fashion-MNIST (-1.2pp)
- Adaptive noise (margin-based): same as global (fires too often)
- Deferred to Phase T2 (associator threshold investigation)

### Process-Level Parallelism
- Vectorized `_find_best_child`: 1.8x speedup (reverted, identical accuracy)
- Process-level parallel benchmarks: `scripts/run_trie_benchmarks_parallel.py`
- Threading doesn't help (tensor ops too small for multi-thread overhead)

### Dataset Caching
- Docker volume `datasets` mounted at `/workspace/.data`
- All benchmark scripts updated to use persistent cache
- Second load: 32ms vs ~10s download

## Decisions
- Training noise default set to 0.0 (disabled), revisit in T2
- Dataset caching via Docker named volume
- Loop version of `_find_best_child` kept over vectorized (identical accuracy, simpler code)
