---
phase: 03-baseline-implementations
plan: 01
subsystem: baselines
tags: [pytorch, nn.Module, linear-layers, parameter-matching, torchinfo, initialization]

# Dependency graph
requires:
  - phase: 01-algebraic-foundations
    provides: "NormedDivisionAlgebra hierarchy, Real/Complex/Quaternion types, STRUCTURE_CONSTANTS, octonion_mul, OctonionLinear"
provides:
  - "RealLinear, ComplexLinear, QuaternionLinear, OctonionDenseLinear nn.Modules"
  - "AlgebraType enum with dim/multiplier/short_name properties"
  - "NetworkConfig, TrainConfig, ComparisonConfig dataclasses"
  - "Per-algebra weight initialization (Trabelsi, Gaudet, Chi(8) polar)"
  - "find_matched_width binary search for parameter-matched comparison"
  - "param_report and flop_report for transparency"
affects: [03-02, 03-03, 03-04, 03-05, 03-06, 04-numerical-stability, 05-optimization-landscape]

# Tech tracking
tech-stack:
  added: [torchinfo, optuna, tensorboard, matplotlib, seaborn, scipy]
  patterns: ["per-algebra linear layers with [..., in_features, dim] convention", "structure-constant-based octonionic forward pass", "binary search parameter matching"]

key-files:
  created:
    - src/octonion/baselines/__init__.py
    - src/octonion/baselines/_config.py
    - src/octonion/baselines/_initialization.py
    - src/octonion/baselines/_algebra_linear.py
    - src/octonion/baselines/_param_matching.py
    - tests/test_baselines_linear.py
  modified:
    - pyproject.toml
    - uv.lock

key-decisions:
  - "OctonionDenseLinear uses W*x ordering (C[i,j,k] * F.linear(x_j, W_i)) matching octonion_mul convention"
  - "Nonzero structure constant entries precomputed at init time for efficient forward pass"
  - "F.linear results cached per (i,j) pair to avoid redundant matrix-vector products"
  - "Parameter matching tests use target=500000 with input_dim=32 to ensure per-unit jumps are <1% of total"

patterns-established:
  - "AlgebraType enum: R(dim=1,mult=8), C(dim=2,mult=4), H(dim=4,mult=2), O(dim=8,mult=1)"
  - "Algebra linear layers: [..., in_features, dim] -> [..., out_features, dim]"
  - "Cross-validation: each algebra linear layer tested against verified Phase 1 algebra multiplication"
  - "TDD workflow: RED (failing tests) -> GREEN (implementation) -> REFACTOR (optimize)"

requirements-completed: [BASE-01, BASE-02, BASE-03]

# Metrics
duration: 12min
completed: 2026-03-08
---

# Phase 03 Plan 01: Algebra Linear Layers Summary

**Per-algebra linear layers (R/C/H/O) with literature-based initialization, structure-constant octonionic product, binary search parameter matching within 1%, and FLOP reporting via torchinfo**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-08T22:55:38Z
- **Completed:** 2026-03-08T23:08:24Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Four algebra-specific nn.Module linear layers with correct multiplication verified against Phase 1 algebra types
- Literature-based initialization: Kaiming/He (R), Trabelsi Rayleigh+phase (C), Gaudet polar Chi(4) (H), Chi(8) polar extension (O)
- Binary search parameter matching achieving within 1% tolerance for all 4 algebras
- FLOP reporting via torchinfo for experimental transparency
- 39 new tests passing, 381 total suite green

## Task Commits

Each task was committed atomically:

1. **Task 1: Install dependencies and create baselines subpackage** - `5e9f56f` (feat)
2. **Task 2 RED: Failing tests for algebra linear layers** - `5370885` (test)
3. **Task 2 GREEN: Implement algebra linear layers, param matching** - `3660892` (feat)
4. **Task 2 REFACTOR: Optimize OctonionDenseLinear forward** - `9255b76` (refactor)

## Files Created/Modified

- `src/octonion/baselines/__init__.py` - Subpackage exports for all baselines components
- `src/octonion/baselines/_config.py` - AlgebraType enum, NetworkConfig, TrainConfig, ComparisonConfig dataclasses
- `src/octonion/baselines/_initialization.py` - Per-algebra weight init (real_init, complex_init, quaternion_init, octonion_init)
- `src/octonion/baselines/_algebra_linear.py` - RealLinear, ComplexLinear, QuaternionLinear, OctonionDenseLinear nn.Modules
- `src/octonion/baselines/_param_matching.py` - find_matched_width, param_report, flop_report
- `tests/test_baselines_linear.py` - 39 tests covering shapes, param counts, cross-validation, init variance, matching, reporting
- `pyproject.toml` - Added torchinfo, optuna, tensorboard, matplotlib, seaborn, scipy dependencies
- `uv.lock` - Updated lockfile

## Decisions Made

- **OctonionDenseLinear forward ordering:** Uses C[i,j,k] * F.linear(x_j, W_i) to match the octonionic product convention W*x from octonion_mul. Initial implementation had indices swapped (x_i, W_j) causing incorrect results -- fixed via Rule 1 auto-fix.
- **Nonzero entry precomputation:** Structure constant tensor has only 64 nonzero entries out of 512. Precomputing (i,j,k,coeff) tuples at init time and caching F.linear results per (i,j) pair eliminates redundant computation in the forward pass.
- **Parameter matching test targets:** Used target=500000 with input_dim=32 to ensure that per-hidden-unit param jumps are well under 1% of total, allowing the binary search to converge for all algebras including octonion (which has the coarsest per-unit jumps due to 8*dim scaling).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed OctonionDenseLinear forward index ordering**
- **Found during:** Task 2 GREEN (implementation)
- **Issue:** Forward pass computed sum C[i,j,k] * F.linear(x_i, W_j) instead of sum C[i,j,k] * F.linear(x_j, W_i). Since octonions are non-commutative, W*x != x*W.
- **Fix:** Swapped i/j roles in the triple loop: x[..., j] paired with weights[i] to match W*x convention
- **Files modified:** src/octonion/baselines/_algebra_linear.py
- **Verification:** Cross-validation test against octonion_mul passes with atol=1e-5
- **Committed in:** 3660892 (part of Task 2 GREEN commit)

**2. [Rule 1 - Bug] Adjusted parameter matching test targets for convergence**
- **Found during:** Task 2 GREEN (implementation)
- **Issue:** Test target of 10000 too small for complex/octonion algebras where per-hidden-unit param jumps exceed 1% of total
- **Fix:** Increased targets to 500000 with input_dim=32 to ensure per-unit jumps are <1%
- **Files modified:** tests/test_baselines_linear.py
- **Verification:** All 4 algebras now match within 1% tolerance
- **Committed in:** 3660892 (part of Task 2 GREEN commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes essential for correctness. The index ordering bug would have produced incorrect octonionic computations during training. No scope creep.

## Issues Encountered

None beyond the auto-fixed deviations above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All 4 algebra linear layers ready for use in network skeletons (Plan 03-02)
- Parameter matching utility ready for fair comparison experiments
- Config dataclasses ready for training infrastructure (Plan 03-03)
- All 381 tests green, no regressions

## Self-Check: PASSED

- All 7 created/modified files exist on disk
- All 4 task commits found in git history (5e9f56f, 5370885, 3660892, 9255b76)

---
*Phase: 03-baseline-implementations*
*Completed: 2026-03-08*
