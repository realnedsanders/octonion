---
phase: 05-optimization-landscape
plan: 01
subsystem: baselines
tags: [phm, kronecker, dense-mixing, parameter-matching, geoopt, pytorch-optimizer]

# Dependency graph
requires:
  - phase: 03-baselines
    provides: AlgebraType enum, _SimpleAlgebraMLP, find_matched_width, AlgebraNetwork
provides:
  - PHM8Linear layer for Kronecker-factored baseline (isolates algebra from factorization)
  - DenseMixingLinear layer for no-structure baseline (go/no-go gate)
  - AlgebraType.PHM8 and AlgebraType.R8_DENSE enum members
  - geoopt and pytorch-optimizer dependencies for Riemannian/Shampoo optimizers
affects: [05-02, 05-03, 05-04, 05-05, 05-06]

# Tech tracking
tech-stack:
  added: [geoopt 0.5.1, pytorch-optimizer 3.10.0]
  patterns: [PHM Kronecker factorization, dense mixing baseline, coarse param matching with tolerance]

key-files:
  created:
    - src/octonion/baselines/_phm_linear.py
    - src/octonion/baselines/_dense_mixing.py
    - tests/test_phm_linear.py
    - tests/test_dense_mixing.py
  modified:
    - pyproject.toml
    - src/octonion/baselines/_config.py
    - src/octonion/baselines/_param_matching.py
    - src/octonion/baselines/__init__.py
    - tests/test_baselines_comparison.py
    - tests/test_baselines_network.py
    - tests/test_baselines_reproduction.py

key-decisions:
  - "PHM8Linear stores A as [n,n,n] and S as [n,out_f,in_f] single Parameters for clean param counting"
  - "DenseMixingLinear param matching uses 10% tolerance (same as conv2d) due to 64x params-per-feature-pair granularity"
  - "ComparisonConfig.algebras default restricted to original 4 algebras (R,C,H,O) to prevent AlgebraNetwork failures"
  - "PHM8/R8_DENSE multiplier=1 nominal; actual param counts handled by binary search in find_matched_width"

patterns-established:
  - "Coarse-grained layers (DenseMixingLinear) use 10% tolerance for param matching, consistent with conv2d precedent"
  - "New AlgebraType members not supported by AlgebraNetwork are guarded via explicit algebra lists in tests"

requirements-completed: [FOUND-04]

# Metrics
duration: 7min
completed: 2026-03-21
---

# Phase 05 Plan 01: New Baselines Summary

**PHM-8 Kronecker-factored and R8 dense-mixing linear layers with geoopt/pytorch-optimizer dependencies for Phase 5 optimization landscape experiments**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-21T05:29:07Z
- **Completed:** 2026-03-21T05:36:23Z
- **Tasks:** 2
- **Files modified:** 11

## Accomplishments
- PHM8Linear implementing Zhang et al. ICLR 2021 PHM formulation with n=8: 512 mixing params + 8*out*in sub-matrix params
- DenseMixingLinear implementing full dense R^{in*8} -> R^{out*8} with no algebraic structure (go/no-go gate baseline)
- AlgebraType extended with PHM8 and R8_DENSE variants, parameter matching integrated for both
- geoopt 0.5.1 and pytorch-optimizer 3.10.0 installed for downstream Riemannian/Shampoo optimizer experiments
- All 186 existing + new tests pass with zero regressions

## Task Commits

Each task was committed atomically:

1. **Task 1 (TDD RED): Failing tests** - `ecf48f0` (test)
2. **Task 1 (TDD GREEN): PHM8Linear + DenseMixingLinear implementation** - `cee2884` (feat)
3. **Task 2: Full test suite compatibility** - `b43ee50` (fix)

## Files Created/Modified
- `src/octonion/baselines/_phm_linear.py` - PHM-8 Kronecker-factored linear layer (H = sum kron(A_i, S_i))
- `src/octonion/baselines/_dense_mixing.py` - Dense mixing linear layer (no algebra structure baseline)
- `src/octonion/baselines/_config.py` - Extended AlgebraType with PHM8 and R8_DENSE; fixed ComparisonConfig default
- `src/octonion/baselines/_param_matching.py` - Added PHM8Linear/DenseMixingLinear dispatch to _SimpleAlgebraMLP
- `src/octonion/baselines/__init__.py` - Exported PHM8Linear and DenseMixingLinear
- `pyproject.toml` - Added geoopt and pytorch-optimizer dependencies
- `tests/test_phm_linear.py` - 14 tests: shapes, param counts, gradients, AlgebraType, param matching
- `tests/test_dense_mixing.py` - 14 tests: shapes, param counts, gradients, AlgebraType, param matching
- `tests/test_baselines_network.py` - Restricted ALL_ALGEBRAS to original 4 for AlgebraNetwork tests
- `tests/test_baselines_comparison.py` - Restricted AlgebraType iterations to original 4 for conv2d/same-width tests
- `tests/test_baselines_reproduction.py` - Restricted parametrize to original 4 for CIFAR tests

## Decisions Made
- PHM8Linear stores A as single Parameter [8,8,8] and S as [8,out_f,in_f] rather than ParameterList, for cleaner parameter counting and einsum-based Kronecker product computation
- DenseMixingLinear param matching uses 10% tolerance (matching conv2d precedent) because 64*hidden^2 params per hidden layer makes unit-step jumps too coarse for 1% tolerance at small widths
- ComparisonConfig.algebras default changed from `list(AlgebraType)` to explicit `[R, C, H, O]` to prevent AlgebraNetwork construction failures when new types are added
- PHM8/R8_DENSE use multiplier=1 as nominal value; actual param counts differ significantly from other dim=8 algebras and are handled correctly by binary search in find_matched_width

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] R8_DENSE param matching test tolerance**
- **Found during:** Task 1 (TDD GREEN)
- **Issue:** DenseMixingLinear has 64x params per feature pair, making per-unit hidden width jumps too large for 1% tolerance at small targets
- **Fix:** Used ref_hidden=64 (larger target) and tolerance=0.10 in test, consistent with conv2d matching precedent
- **Files modified:** tests/test_dense_mixing.py
- **Verification:** find_matched_width succeeds and returns valid width
- **Committed in:** cee2884

**2. [Rule 2 - Missing Critical] Guard existing tests against new AlgebraType members**
- **Found during:** Task 2
- **Issue:** Multiple test files used `list(AlgebraType)` to parametrize tests that build AlgebraNetwork, which doesn't support PHM8/R8_DENSE
- **Fix:** Restricted iterations to [R, C, H, O] in test_baselines_network.py, test_baselines_comparison.py, test_baselines_reproduction.py
- **Files modified:** tests/test_baselines_network.py, tests/test_baselines_comparison.py, tests/test_baselines_reproduction.py
- **Verification:** All 157 baseline tests + 29 reproduction tests pass
- **Committed in:** b43ee50

---

**Total deviations:** 2 auto-fixed (1 bug, 1 missing critical)
**Impact on plan:** Both auto-fixes necessary for correctness. No scope creep.

## Issues Encountered
None - plan executed smoothly.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- PHM8Linear and DenseMixingLinear are ready for use in Plan 05-02 (optimizer configs) and all subsequent Phase 5 experiments
- geoopt and pytorch-optimizer are installed and importable for Riemannian/Shampoo optimizer integration
- Parameter matching works for both new types via find_matched_width binary search

## Self-Check: PASSED

- All 4 created files verified on disk
- All 3 task commits verified in git log

---
*Phase: 05-optimization-landscape*
*Completed: 2026-03-21*
