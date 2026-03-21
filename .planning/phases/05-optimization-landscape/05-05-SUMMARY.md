---
phase: 05-optimization-landscape
plan: 05
subsystem: experiment-orchestration
tags: [experiment-runner, incremental-save, landscape, gate-evaluation, cli]

# Dependency graph
requires:
  - phase: 05-02
    provides: Task generators (algebra_native, cross_product, sinusoidal, classification)
  - phase: 05-03
    provides: Hessian spectrum analysis, curvature measurement, gradient stats
  - phase: 05-04
    provides: Riemannian optimization, LBFGS, Shampoo, gate evaluation
provides:
  - LandscapeConfig dataclass with all 9 tasks and 6 algebras
  - run_landscape_experiment function with incremental save/resume
  - run_landscape.py CLI script with --smoke mode
  - End-to-end smoke tests for pipeline validation
affects: [05-06, 06-experiment-execution]

# Tech tracking
tech-stack:
  added: []
  patterns: [incremental-save-resume, optimizer-specific-config, task-dispatch-table]

key-files:
  created:
    - src/octonion/landscape/_experiment.py
    - scripts/run_landscape.py
    - tests/test_landscape_smoke.py
  modified:
    - src/octonion/landscape/__init__.py
    - src/octonion/baselines/_trainer.py

key-decisions:
  - "evaluate() checks isinstance(loss_fn, CrossEntropyLoss) to skip accuracy for regression tasks"
  - "amp_device_type moved outside batch loop to avoid UnboundLocalError with empty DataLoaders"
  - "Batch size clamped to min(config.batch_size, len(train_ds)) for small smoke test datasets"
  - "Sinusoidal task has output_dim=3 (n_components), not 8 -- tracked in _TASK_DIMS"
  - "Parameter matching uses 10% tolerance for small models (base_hidden=4-16)"

patterns-established:
  - "Incremental save pattern: result.json per (task, optimizer, algebra, seed) with _result_exists check"
  - "Optimizer config dispatch: _optimizer_train_config returns TrainConfig per optimizer name"
  - "Task dispatch table: explicit if/elif for all 9 tasks with ValueError on unknown"

requirements-completed: [FOUND-04]

# Metrics
duration: 7min
completed: 2026-03-21
---

# Phase 05 Plan 05: Experiment Orchestration Summary

**LandscapeConfig with all 9 tasks (5 cross product variants) and incremental save/resume pipeline, validated by 3 end-to-end smoke tests**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-21T05:49:28Z
- **Completed:** 2026-03-21T05:56:49Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- LandscapeConfig dataclass with configurable tasks (9), algebras (6), optimizers (5), seeds (20), and Hessian checkpoint settings
- Full experiment runner with incremental save per (task, optimizer, algebra, seed) for crash resilience and resume support
- CLI script (run_landscape.py) with --smoke mode for quick validation and full experiment orchestration
- 3 end-to-end smoke tests: pipeline correctness, resume skip, gate evaluation -- all passing in ~1s on CPU

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement experiment orchestration with incremental saves and Hessian checkpoints** - `b062225` (feat)
2. **Task 2: Create run_landscape.py script and end-to-end smoke test** - `eff5875` (feat)

## Files Created/Modified
- `src/octonion/landscape/_experiment.py` - LandscapeConfig, run_landscape_experiment, task dispatch, model builder, incremental save
- `src/octonion/landscape/__init__.py` - Added LandscapeConfig and run_landscape_experiment exports
- `src/octonion/baselines/_trainer.py` - Fixed amp_device_type scope and evaluate() regression support
- `scripts/run_landscape.py` - CLI entry point with argparse, --smoke mode, gate evaluation
- `tests/test_landscape_smoke.py` - 3 smoke tests: end-to-end, resume, gate evaluation

## Decisions Made
- evaluate() uses isinstance check on loss_fn to distinguish classification vs regression (no accuracy for MSE)
- amp_device_type defined before the batch loop to avoid UnboundLocalError with empty DataLoaders (when batch_size > dataset size with drop_last=True)
- Batch size clamped to dataset size for small smoke test configurations
- Sinusoidal task output_dim=3 (n_components default), not 8 -- stored in _TASK_DIMS lookup table
- 10% parameter matching tolerance for small models (base_hidden 4-16) to avoid ValueError from coarse granularity

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed amp_device_type UnboundLocalError with empty DataLoaders**
- **Found during:** Task 1 (experiment runner verification)
- **Issue:** When batch_size > len(train_ds) with drop_last=True, the batch loop body never executes, leaving amp_device_type undefined. The validation section then references it, causing UnboundLocalError.
- **Fix:** Moved amp_device_type assignment before the batch loop
- **Files modified:** src/octonion/baselines/_trainer.py
- **Verification:** Smoke test with 100 samples, batch_size=128 passes
- **Committed in:** b062225

**2. [Rule 1 - Bug] Fixed evaluate() shape mismatch for regression tasks**
- **Found during:** Task 1 (experiment runner verification)
- **Issue:** evaluate() always computed argmax accuracy via outputs.max(1) and predicted.eq(targets), which fails for regression (targets shape [B, dim] vs predicted shape [B])
- **Fix:** Added isinstance(loss_fn, CrossEntropyLoss) check; skip accuracy computation for non-classification tasks
- **Files modified:** src/octonion/baselines/_trainer.py
- **Verification:** Both regression (MSELoss) and classification (CrossEntropyLoss) tasks work correctly
- **Committed in:** b062225

**3. [Rule 1 - Bug] Fixed sinusoidal output dimension mismatch**
- **Found during:** Task 1 (all-9-tasks dispatch verification)
- **Issue:** _TASK_DIMS defaulted sinusoidal output_dim to 8, but sinusoidal generates y.shape=[n_components=3]
- **Fix:** Added sinusoidal to _TASK_DIMS with output_dim=3
- **Files modified:** src/octonion/landscape/_experiment.py
- **Verification:** All 9 tasks dispatch correctly with matching shapes
- **Committed in:** b062225

---

**Total deviations:** 3 auto-fixed (3 Rule 1 bugs)
**Impact on plan:** All fixes were necessary for correctness. The trainer bugs were pre-existing but only surfaced because the landscape experiment was the first use of train_model with MSELoss and small datasets. No scope creep.

## Issues Encountered
None beyond the auto-fixed issues above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Experiment orchestration pipeline fully operational
- Plan 05-06 can run the full experiment matrix using run_landscape.py
- Gate evaluation integrated into the pipeline for automatic verdict computation
- Incremental save/resume ensures crash tolerance for long-running experiments

## Self-Check: PASSED

All created files verified on disk. Both task commits (b062225, eff5875) found in git log.

---
*Phase: 05-optimization-landscape*
*Completed: 2026-03-21*
