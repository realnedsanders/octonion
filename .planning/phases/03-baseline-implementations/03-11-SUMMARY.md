---
phase: 03-baseline-implementations
plan: 11
subsystem: baselines
tags: [pytorch, octonion, quaternion, performance, einsum, caching, convolution, linear]

# Dependency graph
requires:
  - phase: 03-10
    provides: Tier 1 optimizations and profiling baseline for algebra layers

provides:
  - OctonionDenseLinear with fused einsum+F.linear forward (zero Python loop iterations)
  - OctonionConv2d with eval-mode fused weight caching and buffer-registered structure constants
  - OctonionConv1d with buffer-registered structure constants
  - QuaternionConv2d with eval-mode fused weight caching
  - STRUCTURE_CONSTANTS registered as non-persistent buffer _C in all three octonion layers
  - TestOctonionDenseLinearFusedForward: 7 equivalence tests
  - TestOctonionConv2dEvalCache: 6 eval-cache tests
  - TestQuaternionConv2dEvalCache: 4 eval-cache tests

affects: [03-12, 03-09, phase-5-experiments]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Register STRUCTURE_CONSTANTS as non-persistent buffer: auto device migration, not in state_dict"
    - "Eval-mode fused weight cache: override train() to invalidate, populate on first eval forward"
    - "Fused einsum linear: stack weights -> einsum with structure constants -> single F.linear"
    - "Input layout for fused linear: transpose [..., in_f, 8] to [..., 8, in_f] before flatten"

key-files:
  created: []
  modified:
    - src/octonion/baselines/_algebra_linear.py
    - src/octonion/baselines/_algebra_conv.py
    - tests/test_perf_equivalence.py

key-decisions:
  - "Fused linear input ordering: x.transpose(-2,-1).reshape(..., 8*in_f) because x has shape [..., in_f, 8] — algebra dim last. Without transpose, the column ordering of fused_flat (j*in_f+q) mismatches x_flat (i_f*8+j)"
  - "Output reshape: out_flat.reshape(..., 8, out_f).transpose(-2,-1) for [..., out_f, 8] output"
  - "Eval cache type annotation: _fused_cache: torch.Tensor | None — not a buffer or parameter, just instance attribute"
  - "Non-persistent buffer for _C: persistent=False so STRUCTURE_CONSTANTS not saved in checkpoints"
  - "Tier 1 pytestmark refactored from module-level to class-level skipif — prevents Tier 2 tests from being skipped"

patterns-established:
  - "Non-persistent buffer pattern: register_buffer('_C', STRUCTURE_CONSTANTS.to(dtype=dtype), persistent=False)"
  - "Eval cache pattern: _fused_cache=None at init, override train() to invalidate, if not training and cache is not None: use cache else: compute and store if not training"

requirements-completed: [BASE-01, BASE-02, BASE-03]

# Metrics
duration: 10min
completed: 2026-03-13
---

# Phase 03 Plan 11: Tier 2 Performance Optimizations Summary

**Fused einsum OctonionDenseLinear (64 Python loop iterations -> single F.linear) plus eval-mode weight caching in OctonionConv2d and QuaternionConv2d**

## Performance

- **Duration:** 10 min
- **Started:** 2026-03-13T07:41:31Z
- **Completed:** 2026-03-13T07:51:40Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Replaced OctonionDenseLinear's 64-iteration Python loop with a single `torch.einsum` + `F.linear` call — zero Python loop iterations at inference time
- Added eval-mode fused weight caching to `OctonionConv2d` and `QuaternionConv2d` — eliminates redundant `stack + einsum + reshape` per validation batch
- Registered `STRUCTURE_CONSTANTS` as non-persistent buffer `_C` in all three octonion layers (`OctonionDenseLinear`, `OctonionConv1d`, `OctonionConv2d`) — auto device migration with `.to()`, not saved in state_dict
- Full 588-test suite passes with zero regressions

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Add failing tests** - `e2f67d7` (test)
2. **Task 1 GREEN: Fuse OctonionDenseLinear forward and register _C buffers** - `7733b54` (feat)
3. **Task 2: Add eval-mode fused weight caching to OctonionConv2d and QuaternionConv2d** - `af756e8` (feat)

_Note: TDD task produced test commit (RED) + implementation commit (GREEN)_

## Files Created/Modified

- `src/octonion/baselines/_algebra_linear.py` - OctonionDenseLinear: fused einsum forward, _C buffer registration, removed _nonzero_entries
- `src/octonion/baselines/_algebra_conv.py` - OctonionConv2d/_Conv1d: _C buffer registration; OctonionConv2d/QuaternionConv2d: eval-mode fused weight cache + train() override
- `tests/test_perf_equivalence.py` - Extended with Tier 2 tests: fused linear equivalence (7 tests), OctonionConv2d eval cache (6 tests), QuaternionConv2d eval cache (4 tests); Tier 1 tests migrated from module-level pytestmark to class-level skipif

## Decisions Made

- **Input transpose for fused linear**: `x` has shape `[..., in_f, 8]` (algebra dim last). Flattening directly gives `[..., in_f*8]` with ordering `(i_f, j)` — but `fused_flat`'s column ordering is `(j, i_f)`. Fix: `x.transpose(-2,-1).reshape(..., 8*in_f)` to get consistent `(j, i_f)` ordering. Confirmed by running tests after each attempt.
- **Output reshape**: After `F.linear` gives `[..., 8*out_f]`, reshape to `[..., 8, out_f]` then `.transpose(-2,-1)` to restore `[..., out_f, 8]` layout.
- **_fused_cache as plain attribute**: Not registered as buffer or parameter — it's a transient compute cache, should not appear in `state_dict()` or affect device migration logic.
- **pytestmark refactor**: Moved Tier 1 test skip from module-level `pytestmark` to class-level `@pytest.mark.skipif` to allow Tier 2 tests to run even when `_normalization` is available.
- **Non-persistent buffer for _C**: `persistent=False` keeps STRUCTURE_CONSTANTS out of state_dicts while still getting automatic device/dtype migration via `.to()`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed einsum+reshape input ordering in fused OctonionDenseLinear**
- **Found during:** Task 1 GREEN phase (first test run)
- **Issue:** Initial implementation used `x.reshape(..., 8*in_f)` on `[..., in_f, 8]` input, giving column ordering `(i_f, j)` — but `fused_flat` columns were ordered `(j, i_f)` from the permute. Max output diff was 6.22 (catastrophic mismatch).
- **Fix:** Changed to `x.transpose(-2,-1).reshape(..., 8*in_f)` and `out_flat.reshape(..., 8, out_f).transpose(-2,-1)` for matching ordering throughout.
- **Files modified:** `src/octonion/baselines/_algebra_linear.py`
- **Verification:** `TestOctonionDenseLinearFusedForward` — all 7 tests pass, max_diff < 1e-5
- **Committed in:** `7733b54` (Task 1 feat commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - correctness bug in einsum index ordering)
**Impact on plan:** Required fix for correct output. Resolved during implementation before final commit.

## Issues Encountered

None beyond the auto-fixed einsum ordering issue above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All Tier 2 optimizations complete; Tier 3 (mixed-precision, torch.compile) can proceed per plan 03-12
- 588 tests passing — full baseline suite regression-free
- OctonionDenseLinear and OctonionConv2d eval-mode throughput improved; ready for CIFAR GPU benchmark validation in plan 03-09

---
*Phase: 03-baseline-implementations*
*Completed: 2026-03-13*
