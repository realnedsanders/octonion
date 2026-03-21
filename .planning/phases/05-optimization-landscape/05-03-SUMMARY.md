---
phase: 05-optimization-landscape
plan: 03
subsystem: landscape-analysis
tags: [hessian, lanczos, curvature, eigenspectrum, loss-surface]

# Dependency graph
requires:
  - phase: 05-01
    provides: "PHM8Linear and DenseMixingLinear baseline layers"
provides:
  - "Full Hessian eigenspectrum computation via torch.autograd.functional.hessian"
  - "Stochastic Lanczos eigenspectrum approximation with full reorthogonalization"
  - "Auto-dispatching compute_hessian_spectrum based on parameter count"
  - "Bill & Cox curvature measurement with Li et al. 2018 filter normalization"
affects: [05-05, 05-06]

# Tech tracking
tech-stack:
  added: [torch.func.functional_call, torch.autograd.functional.hessian, np.polyfit]
  patterns: [stochastic-lanczos-quadrature, filter-normalization, functional-parameter-threading]

key-files:
  created:
    - src/octonion/landscape/_hessian.py
    - src/octonion/landscape/_curvature.py
    - tests/test_landscape_hessian.py
  modified:
    - src/octonion/landscape/__init__.py

key-decisions:
  - "torch.func.functional_call for Hessian gradient threading (avoids p.data mutation breaking autograd)"
  - "Full reorthogonalization in Lanczos (Ghorbani et al. 2019) for numerical stability"
  - "Float64 required for finite-difference Hessian cross-check (float32 eps=1e-4 insufficient)"

patterns-established:
  - "Functional parameter threading: use torch.func.functional_call instead of p.data mutation for autograd"
  - "Lanczos iteration: full reorthogonalization with Gram-Schmidt after each step"
  - "Curvature measurement: filter-normalize -> 1D profile sample -> quadratic fit -> curvature = 2a"

requirements-completed: [FOUND-04]

# Metrics
duration: 7min
completed: 2026-03-21
---

# Phase 05 Plan 03: Hessian Eigenspectrum and Curvature Analysis Summary

**Full Hessian + stochastic Lanczos eigenspectrum analysis and Bill & Cox curvature measurement with Li et al. 2018 filter normalization**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-21T05:39:21Z
- **Completed:** 2026-03-21T05:46:11Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Full Hessian computation verified against analytical eigenvalues ([2.0, 6.0] on toy quadratic) and finite-difference cross-check within 1e-3
- Stochastic Lanczos with full reorthogonalization validated: top eigenvalue within 20% of full Hessian on small models
- Auto method selection: full Hessian for <2000 params, Lanczos for larger models
- Bill & Cox curvature measurement with Li et al. 2018 per-filter normalization, verified positive curvature at minimum and weight restoration

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement full Hessian and stochastic Lanczos eigenspectrum analysis**
   - `12cdab5` (test): add failing tests for Hessian eigenspectrum analysis
   - `c286619` (feat): implement full Hessian and stochastic Lanczos eigenspectrum analysis
2. **Task 2: Implement Bill & Cox loss surface curvature measurement**
   - `dd8ba73` (test): add failing tests for Bill & Cox curvature measurement
   - `1f2c8c2` (feat): implement Bill & Cox curvature measurement with filter normalization

_Note: TDD tasks have RED (test) and GREEN (feat) commits._

## Files Created/Modified
- `src/octonion/landscape/_hessian.py` - Full Hessian, stochastic Lanczos, HVP, auto-dispatch
- `src/octonion/landscape/_curvature.py` - Bill & Cox curvature with filter normalization
- `src/octonion/landscape/__init__.py` - Module exports for all landscape functions
- `tests/test_landscape_hessian.py` - 12 tests covering Hessian, Lanczos, auto-selection, curvature

## Decisions Made
- Used `torch.func.functional_call` instead of `p.data` mutation for Hessian computation -- p.data assignment breaks autograd graph when using `torch.autograd.functional.hessian`
- Full reorthogonalization in Lanczos (not partial) following Ghorbani et al. 2019 for numerical stability
- Float64 required for finite-difference Hessian validation tests -- float32 with eps=1e-4 has insufficient precision for 4-point central difference

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed autograd graph breakage in compute_full_hessian**
- **Found during:** Task 1 (Full Hessian implementation)
- **Issue:** Plan specified setting model parameters via `p.data = p_flat_slice` which breaks the autograd graph -- `torch.autograd.functional.hessian` requires the function output to depend on the input through the computation graph
- **Fix:** Used `torch.func.functional_call(model, param_dict, (data_x,))` to thread gradients through parameters without mutating `p.data`
- **Files modified:** `src/octonion/landscape/_hessian.py`
- **Verification:** Toy quadratic eigenvalues [2.0, 6.0] match analytical solution; all tests pass
- **Committed in:** `c286619`

**2. [Rule 1 - Bug] Fixed finite-difference test using float32**
- **Found during:** Task 1 (Test validation)
- **Issue:** Finite-difference Hessian test used float32 which has insufficient precision for 4-point central difference with eps=1e-4 (denominator 4*eps^2 = 4e-8 is near float32 epsilon)
- **Fix:** Switched test model and data to float64 with eps=1e-5
- **Files modified:** `tests/test_landscape_hessian.py`
- **Verification:** Eigenvalues match within atol=1e-3
- **Committed in:** `c286619`

---

**Total deviations:** 2 auto-fixed (2 bug fixes)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered
None beyond the deviations documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Landscape analysis toolkit complete: Hessian, Lanczos, and curvature measurement all operational
- Ready for plan 05-04 (gradient statistics) and 05-05 (experiment orchestration)
- All functions work with arbitrary nn.Module including PHM8Linear and DenseMixingLinear

## Self-Check: PASSED

All 4 source/test files exist. All 4 task commits verified.

---
*Phase: 05-optimization-landscape*
*Completed: 2026-03-21*
