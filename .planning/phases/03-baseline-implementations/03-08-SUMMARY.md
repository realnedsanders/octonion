---
phase: 03-baseline-implementations
plan: 08
subsystem: baselines
tags: [conv2d, param-matching, topology-dispatch, comparison-runner, cifar]

# Dependency graph
requires:
  - phase: 03-07
    provides: "ResNet-style residual blocks in AlgebraNetwork conv2d topology"
provides:
  - "Topology-aware run_comparison dispatching to AlgebraNetwork for conv2d"
  - "find_matched_width supporting conv2d topology via binary search over base_hidden"
  - "_build_conv_model helper for building AlgebraNetwork with conv2d topology"
affects: [03-06, 05-experiments, 07-full-scale]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Topology-aware model dispatch in comparison runner"]

key-files:
  created: []
  modified:
    - src/octonion/baselines/_param_matching.py
    - src/octonion/baselines/_comparison.py
    - tests/test_baselines_comparison.py

key-decisions:
  - "Conv2d param matching uses 10% tolerance (not 1%) due to coarse base_hidden granularity from multiplier scaling"
  - "_build_conv_model helper parallels _build_simple_mlp for conv2d topology model building"
  - "run_comparison uses topology-aware _build_model closure dispatching to correct builder"

patterns-established:
  - "Topology dispatch: check topology string, use appropriate model builder for reference, matching, training, and summary"
  - "Conv2d tolerance: 10% for param matching due to discrete base_hidden steps scaled by algebra multiplier"

requirements-completed: [BASE-01, BASE-02, BASE-03]

# Metrics
duration: 26min
completed: 2026-03-09
---

# Phase 3 Plan 8: Topology-Aware Comparison Runner Summary

**Topology-aware run_comparison dispatching to AlgebraNetwork for conv2d via find_matched_width binary search over base_hidden**

## Performance

- **Duration:** 26 min
- **Started:** 2026-03-09T00:51:29Z
- **Completed:** 2026-03-09T01:17:50Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Extended find_matched_width to support conv2d topology via AlgebraNetwork param counting with binary search over base_hidden
- Made run_comparison topology-aware: conv2d dispatches to AlgebraNetwork, MLP dispatches to _SimpleAlgebraMLP (backward compatible)
- CIFAR-shaped [B, 3, 32, 32] data flows through conv2d models without shape mismatch errors
- All 47 comparison and reproduction tests pass, plus 118 broader baseline tests with no regressions

## Task Commits

Each task was committed atomically (TDD: test -> feat):

1. **Task 1: Make find_matched_width support conv2d topology**
   - `5ed510d` (test) - failing tests for conv2d param matching
   - `23bf9e6` (feat) - extend find_matched_width with _build_conv_model helper
2. **Task 2: Make run_comparison topology-aware with model dispatch**
   - `88454ee` (test) - failing tests for topology-aware run_comparison
   - `72cde05` (feat) - topology-aware dispatch in run_comparison

## Files Created/Modified
- `src/octonion/baselines/_param_matching.py` - Added _build_conv_model helper, refactored find_matched_width to dispatch between MLP and conv2d binary search
- `src/octonion/baselines/_comparison.py` - Added _build_conv_model import, refactored run_comparison with topology-aware _build_model closure
- `tests/test_baselines_comparison.py` - Added TestConv2dParamMatching (4 tests) and TestConv2dComparison (5 tests) classes

## Decisions Made
- **Conv2d 10% tolerance:** AlgebraNetwork internally scales base_hidden by algebra.multiplier (1-8x), making parameter jumps per base_hidden step large. For REAL (8x multiplier), each +1 base_hidden adds 8 actual filters with corresponding quadratic param growth. 1% matching is infeasible; 10% is realistic for conv2d.
- **Octonion reference for tests:** Conv2d param matching tests use octonion (smallest multiplier=1) as reference for best granularity, allowing all algebras to match within 10%.
- **Closure-based dispatch:** run_comparison uses a local `_build_model` closure that captures topology/overrides, keeping the 10 steps of run_comparison clean without conditional branching at each step.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Adjusted conv2d param matching tolerance from 1% to 10%**
- **Found during:** Task 1 (conv2d param matching)
- **Issue:** The plan specified 1% tolerance for conv2d matching, but AlgebraNetwork's multiplier-based scaling creates discrete jumps in parameter count per base_hidden step that are larger than 1% for most algebra/target combinations. For example, Complex with multiplier=4 at depth=6: bh=11 gives 2.7M params, bh=12 gives 3.2M params (18% jump).
- **Fix:** Used 10% tolerance for conv2d topology (MLP keeps 1%). Updated tests to use realistic tolerance.
- **Files modified:** src/octonion/baselines/_param_matching.py, src/octonion/baselines/_comparison.py, tests/test_baselines_comparison.py
- **Verification:** All 4 algebras successfully matched within 10% tolerance
- **Committed in:** 23bf9e6, 72cde05

---

**Total deviations:** 1 auto-fixed (1 bug - tolerance mismatch with architecture granularity)
**Impact on plan:** Tolerance adjustment is a practical necessity given the multiplier-based architecture. The 10% tolerance is standard for conv2d experiments where exact param matching is limited by channel count granularity.

## Issues Encountered
None beyond the tolerance adjustment documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Topology-aware comparison runner is ready for CIFAR reproduction experiments (03-06 slow tests)
- Slow CIFAR reproduction tests now correctly dispatch to conv2d AlgebraNetworks (no longer build MLPs)
- Phase 3 baseline implementations are complete; all 8 plans executed
- Ready for Phase 4 (experiment design) and Phase 5 (go/no-go experiments)

## Self-Check: PASSED

All files exist, all commits verified:
- src/octonion/baselines/_param_matching.py: FOUND
- src/octonion/baselines/_comparison.py: FOUND
- tests/test_baselines_comparison.py: FOUND
- .planning/phases/03-baseline-implementations/03-08-SUMMARY.md: FOUND
- Commits: 5ed510d, 23bf9e6, 88454ee, 72cde05: ALL FOUND

---
*Phase: 03-baseline-implementations*
*Completed: 2026-03-09*
