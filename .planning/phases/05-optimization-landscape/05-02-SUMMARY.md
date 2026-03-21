---
phase: 05-optimization-landscape
plan: 02
subsystem: tasks
tags: [synthetic-data, optimization-landscape, cross-product, classification, regression, octonion-mul]

# Dependency graph
requires:
  - phase: 01-foundations
    provides: "octonion_mul, cross_product, structure constants"
provides:
  - "5 synthetic task generators with known optima for landscape experiments"
  - "Deterministic data generation with dim=8 and dim=64 variants"
  - "Cross product recovery with noise levels and 3D positive control"
  - "Classification with Bayes-optimal accuracy metadata"
affects: [05-03, 05-04, 05-05, 05-06]

# Tech tracking
tech-stack:
  added: []
  patterns: ["deterministic generator seeding via torch.Generator", "blocked 8x8 octonion mul for dim=64", "union-bound Bayes-optimal accuracy estimate"]

key-files:
  created:
    - src/octonion/tasks/__init__.py
    - src/octonion/tasks/_algebra_native.py
    - src/octonion/tasks/_cross_product.py
    - src/octonion/tasks/_sinusoidal.py
    - src/octonion/tasks/_classification.py
    - tests/test_landscape_tasks.py
  modified: []

key-decisions:
  - "dim=64 algebra-native task uses 8 independent octonion blocks + orthogonal rotation mixing (not a single 64D algebra)"
  - "7D cross product computed via Im(Im(a)*Im(b)) directly using octonion structure constants einsum"
  - "Bayes-optimal accuracy estimated via union bound on minimum inter-center distance"
  - "All generators compute in float64 and cast to float32 for TensorDataset output"
  - "Cross product dim=64 embeds signal in first cross_dim dims, pads remaining with noise (input) / zeros (target)"

patterns-established:
  - "Task generator API: build_*(n_train, n_test, dim, seed) -> (TensorDataset, TensorDataset)"
  - "Classification returns 3-tuple with metadata dict containing bayes_optimal_accuracy and centers"

requirements-completed: [FOUND-04]

# Metrics
duration: 4min
completed: 2026-03-21
---

# Phase 05 Plan 02: Task Generators Summary

**5 deterministic synthetic task generators (algebra-native, cross product, sinusoidal, classification) with dim=8/64 support and known-optimum metadata for go/no-go landscape experiments**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-21T05:29:11Z
- **Completed:** 2026-03-21T05:33:12Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 6

## Accomplishments

- Implemented all 5 synthetic task generators with consistent API
- Algebra-native tasks support dim=1/2/4/8/64 with algebra-specific multiplication
- Cross product recovery supports 7D (octonion structure constants) and 3D (positive control) with configurable noise
- Classification includes analytically-computed Bayes-optimal accuracy via union bound
- All 39 tests pass covering shapes, determinism, noise behavior, and dimensional variants

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Failing tests** - `893c2d4` (test)
2. **Task 1 (GREEN): Implement all 5 generators** - `e277554` (feat)

## Files Created/Modified

- `src/octonion/tasks/__init__.py` - Public API exporting all 5 generators
- `src/octonion/tasks/_algebra_native.py` - Tasks 1+2: single/multi-layer algebra-native (y=a*x*b)
- `src/octonion/tasks/_cross_product.py` - Task 3: 7D/3D cross product recovery with noise
- `src/octonion/tasks/_sinusoidal.py` - Task 4: multi-output sinusoidal regression
- `src/octonion/tasks/_classification.py` - Task 5: Gaussian cluster classification with Bayes-optimal accuracy
- `tests/test_landscape_tasks.py` - 39 tests covering all generators

## Decisions Made

- **dim=64 blocked multiplication:** Uses 8 independent 8D octonion blocks with QR-orthogonal mixing matrix, rather than inventing a 64D algebra
- **float64 computation:** All generators compute ground truth in float64 for numerical accuracy, then cast output to float32 for training compatibility
- **Cross product dim=64:** Signal embedded in first cross_dim dimensions; remaining input dims filled with noise, remaining target dims filled with zeros
- **Bayes-optimal accuracy:** Uses union bound approximation P(error) <= (K-1) * Phi(-d_min/2) with floor at chance level

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed torch.randn_like generator incompatibility**
- **Found during:** Task 1 GREEN phase
- **Issue:** `torch.randn_like()` does not accept a `generator` keyword argument in PyTorch
- **Fix:** Replaced with `torch.randn(shape, generator=g, dtype=dtype)` using explicit shape
- **Files modified:** `src/octonion/tasks/_cross_product.py`
- **Verification:** All 39 tests pass
- **Committed in:** `e277554` (part of GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor API correction, no scope change.

## Issues Encountered

None beyond the auto-fixed deviation.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All 5 task generators ready for use by 05-03 (training loop) and 05-04/05/06 experiments
- Deterministic seeding ensures reproducible landscape comparisons
- Default sizes (50K train / 10K test) match experiment requirements

## Self-Check: PASSED

All 6 created files verified present. Both commits (893c2d4, e277554) confirmed in git log.

---
*Phase: 05-optimization-landscape*
*Completed: 2026-03-21*
