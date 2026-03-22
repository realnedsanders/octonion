---
phase: 05-optimization-landscape
plan: 07
subsystem: landscape-analysis
tags: [hessian, curvature, gradient-variance, post-training, checkpoint]

# Dependency graph
requires:
  - phase: 05-optimization-landscape (plans 01-05)
    provides: "Experiment pipeline with training, model checkpoints, and result.json"
provides:
  - "Fixed intermediate Hessian checkpoint saving at 0.25/0.50 fractions for hessian seeds"
  - "Standalone post-training analysis script (scripts/run_post_analysis.py)"
  - "Result JSON files enriched with hessian_spectrum, curvature, and gradient_stats keys"
affects: [05-08, analyze_landscape.py]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Intermediate checkpoint extraction from trainer epoch checkpoints", "Idempotent JSON update pattern for post-analysis results"]

key-files:
  created:
    - scripts/run_post_analysis.py
  modified:
    - src/octonion/landscape/_experiment.py

key-decisions:
  - "checkpoint_every set to epochs//4 for hessian seeds to capture intermediate fractions"
  - "Intermediate checkpoints extracted from trainer epoch checkpoints rather than callback mechanism"
  - "Post-analysis is standalone script (not integrated into experiment runner) for flexibility and re-runnability"
  - "Gradient variance per-seed result.json includes grad_norm_mean for analyze_landscape.py compatibility"

patterns-established:
  - "Hessian seed detection: is_hessian_seed flag flows through _optimizer_train_config to set checkpoint frequency"
  - "Post-analysis idempotency: skip-if-exists with --force override pattern"

requirements-completed: [FOUND-04]

# Metrics
duration: 3min
completed: 2026-03-22
---

# Phase 05 Plan 07: Post-Training Analysis Gap Closure Summary

**Fixed intermediate Hessian checkpoint saving and created standalone post-analysis script for computing Hessian eigenspectrum, curvature, and gradient variance on saved checkpoints**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-22T13:05:12Z
- **Completed:** 2026-03-22T13:09:07Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Fixed the silent skipping of intermediate Hessian checkpoints (0.25, 0.50 fractions) by overriding checkpoint_every for hessian seeds and extracting model_state_dict from trainer epoch checkpoints
- Created scripts/run_post_analysis.py with full CLI that computes Hessian eigenspectrum, Bill & Cox curvature, and gradient variance from saved checkpoints
- Post-analysis script writes results to result.json in exact format that analyze_landscape.py expects (hessian_spectrum, curvature, curvature_detail, gradient_stats keys)

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix intermediate Hessian checkpoint saving in _experiment.py** - `3714e5b` (feat)
2. **Task 2: Create post-training analysis script** - `7055efa` (feat)

## Files Created/Modified
- `src/octonion/landscape/_experiment.py` - Added is_hessian_seed parameter to _optimizer_train_config; fixed intermediate checkpoint saving loop to handle fractions 0.25/0.50
- `scripts/run_post_analysis.py` - Standalone post-training analysis script with CLI; computes Hessian eigenspectrum, curvature, gradient variance and writes back to result.json

## Decisions Made
- checkpoint_every set to epochs//4 for hessian seeds (e.g., 25 for 100 epochs) to ensure trainer saves checkpoints at epochs matching intermediate fractions
- Intermediate checkpoints extracted from trainer's epoch checkpoint files (model_state_dict key) rather than adding a callback mechanism to train_model()
- Post-analysis kept as standalone script rather than integrating into run_landscape_experiment() for re-runnability and separation of concerns
- gradient_stats includes grad_norm_mean key for compatibility with analyze_landscape.py line 607 check

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all functionality is fully implemented and wired to the existing analysis functions.

## Next Phase Readiness
- Post-analysis script can now be run after training to populate hessian_spectrum, curvature, and gradient_stats in result.json
- analyze_landscape.py will find and use these keys without modification
- Ready for plan 05-08 (full experiment run and gate evaluation)

---
*Phase: 05-optimization-landscape*
*Completed: 2026-03-22*
