---
phase: 03-baseline-implementations
plan: 10
subsystem: performance
tags: [torch.profiler, tril_indices, cudnn.benchmark, set_to_none, non_blocking, batch-normalization]

# Dependency graph
requires:
  - phase: 03-01
    provides: OctonionDenseLinear and AlgebraNetwork skeleton
  - phase: 03-02
    provides: QuaternionBatchNorm and OctonionBatchNorm with _tril_to_symmetric
  - phase: 03-06
    provides: CIFAR benchmark infrastructure and cifar_network_config
provides:
  - GPU profiling script with baseline forward+backward timing for all 4 algebras
  - Vectorized _tril_to_symmetric (no Python for-loops, uses torch.tril_indices)
  - Training loop with set_to_none=True, non_blocking=True, cudnn.benchmark=True
  - Equivalence tests proving vectorized output matches Python-loop reference exactly
affects: [03-11, 03-12, 09-final-evaluation]

# Tech tracking
tech-stack:
  added: [torch.profiler, torch.tril_indices]
  patterns:
    - Vectorized index-scatter via tril_indices for symmetric matrix assembly
    - cudnn.benchmark enabled only for CUDA (safe for fixed-size CIFAR inputs)
    - non_blocking=True data transfers paired with pin_memory=True data loaders
    - set_to_none=True on zero_grad to avoid memset-per-parameter overhead

key-files:
  created:
    - scripts/profile_baseline.py
    - tests/test_perf_equivalence.py
  modified:
    - src/octonion/baselines/_normalization.py
    - src/octonion/baselines/_trainer.py

key-decisions:
  - "tril_indices overhead negligible vs eliminated Python loop: 36 elements for dim=8, C++ call"
  - "cudnn.benchmark guarded with str(device).startswith('cuda') check; safe for fixed CIFAR batch sizes"
  - "checkpoint_every audited: TrainConfig default=10 (short/Optuna runs), cifar_train_config=25 (8 checkpoints/200 epochs, appropriate)"
  - "Equivalence tests require torch.equal (exact match) not allclose: index shuffling has no floating-point error"

patterns-established:
  - "Profile with --no-traces flag for CI/quick runs; Chrome traces optional for deep dives"
  - "Capture reference implementation in test file before optimizing (oracle pattern)"

requirements-completed: [BASE-01, BASE-02, BASE-03]

# Metrics
duration: 7min
completed: 2026-03-13
---

# Phase 3 Plan 10: Profiling Baseline and Tier 1 Optimizations Summary

**GPU profiling script for all 4 algebras plus vectorized _tril_to_symmetric and training loop micro-optimizations (set_to_none, non_blocking, cudnn.benchmark) with exact-match equivalence tests**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-13T07:30:18Z
- **Completed:** 2026-03-13T07:37:42Z
- **Tasks:** 2 of 2
- **Files modified:** 4

## Accomplishments

- Created `scripts/profile_baseline.py` with full torch.profiler integration: per-algebra timing tables, Chrome trace export, argparse CLI (--device/--batch-size/--warmup/--iters/--algebras/--no-traces)
- Vectorized `_tril_to_symmetric` in `_normalization.py`: replaced Python `for i/for j` double loop with `torch.tril_indices` vectorized scatter (no Python loops remain)
- Added `set_to_none=True` to `optimizer.zero_grad()` and `non_blocking=True` to device `.to()` calls in training loop
- Enabled `torch.backends.cudnn.benchmark = True` for CUDA devices before training loop
- Created `tests/test_perf_equivalence.py` with 23 exact-equality tests across dim=2/4/8, batch shapes, float32/float64

## Task Commits

Each task was committed atomically:

1. **Task 1: Create profiling script and equivalence test scaffold** - `89e2fe2` (feat)
2. **Task 2: Apply Tier 1 zero-risk optimizations and audit checkpoint frequency** - `ab1fd8e` (feat)

**Plan metadata:** (final docs commit, see below)

## Files Created/Modified

- `scripts/profile_baseline.py` - Profiles forward+backward pass for all 4 algebras using torch.profiler; exports Chrome traces; CLI with device/batch-size/warmup/iters/algebras flags
- `tests/test_perf_equivalence.py` - 23 exact-equality equivalence tests capturing Python-loop reference vs optimized vectorized _tril_to_symmetric
- `src/octonion/baselines/_normalization.py` - Vectorized _tril_to_symmetric: torch.tril_indices + scatter indexing, zero Python for-loops
- `src/octonion/baselines/_trainer.py` - set_to_none=True on zero_grad, non_blocking=True on data transfers, cudnn.benchmark=True for CUDA

## Decisions Made

- `torch.tril_indices` overhead is negligible (36 elements for dim=8, C++ call returning small tensor) vs eliminated Python loop; no need to cache as buffers
- `cudnn.benchmark` guarded with `str(device).startswith("cuda")` to be safe on CPU-only runs
- Equivalence tests use `torch.equal` (exact match) not `torch.allclose`: symmetric index shuffling has no floating-point computation, so any deviation would be a bug, not rounding
- Checkpoint audit: `TrainConfig` default `checkpoint_every=10` appropriate for short runs (Optuna 20-epoch trials); `cifar_train_config()` uses `checkpoint_every=25` producing 8 checkpoints over 200 epochs, appropriate for multi-hour training

## Deviations from Plan

None - plan executed exactly as written. The TDD flow for Task 1 proceeded normally: tests were written first (capturing the Python-loop reference as oracle), confirmed they passed against the existing implementation, then the profiling script was created. Task 2 applied all 5 optimizations as specified.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Profiling infrastructure ready: run `docker compose run --rm dev uv run python scripts/profile_baseline.py --no-traces` to capture baseline numbers before further optimizations
- Tier 1 optimizations applied; plans 03-11 and 03-12 can build on this measured baseline
- All 571 tests pass; codebase is clean for next optimization tier

---
*Phase: 03-baseline-implementations*
*Completed: 2026-03-13*
