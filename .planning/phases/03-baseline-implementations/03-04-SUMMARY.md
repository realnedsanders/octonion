---
phase: 03-baseline-implementations
plan: 04
subsystem: baselines
tags: [pytorch, training, optuna, tensorboard, statistics, matplotlib, seaborn]

# Dependency graph
requires:
  - phase: 03-baseline-implementations
    provides: "AlgebraType enum, TrainConfig/ComparisonConfig dataclasses, per-algebra linear layers"
provides:
  - "train_model with full observability (TensorBoard, gradient stats, VRAM, early stopping, LR warmup, AMP, checkpointing)"
  - "seed_everything for deterministic training"
  - "run_optuna_study for Bayesian hyperparameter search"
  - "save_checkpoint / load_checkpoint for full training state persistence"
  - "paired_comparison, cohen_d, holm_bonferroni, confidence_interval for statistical testing"
  - "plot_convergence, plot_comparison_bars, plot_param_table for visualization"
affects: [03-05, 03-06, 04-numerical-stability, 05-optimization-landscape, 06-benchmark-experiments, 07-density-geometric]

# Tech tracking
tech-stack:
  added: []
  patterns: ["train_model returns metrics dict with lr_history for warmup diagnostics", "Optuna MedianPruner for early trial stopping", "Holm-Bonferroni step-down for family-wise error control", "matplotlib Agg backend for container/headless environments"]

key-files:
  created:
    - src/octonion/baselines/_trainer.py
    - src/octonion/baselines/_stats.py
    - src/octonion/baselines/_plotting.py
    - tests/test_baselines_trainer.py
  modified:
    - src/octonion/baselines/__init__.py

key-decisions:
  - "Warmup implemented via direct optimizer param group LR manipulation (not LambdaLR composition) for simplicity and correctness with scheduler interaction"
  - "Gradient stats computed after backward pass, before optimizer.zero_grad(), to capture actual training gradients"
  - "Optuna study uses reduced epochs (20) per trial with MedianPruner for efficient search"
  - "paired_comparison returns NaN-safe results for identical inputs (common edge case in early experiments)"

patterns-established:
  - "train_model returns dict with train_losses, val_losses, val_accuracies, lr_history, best_val_acc, best_val_loss, total_time_seconds, epochs_trained, early_stopped"
  - "Optuna objective function builds TrainConfig from trial.suggest_* params"
  - "All plot functions use plt.savefig + plt.close pattern to avoid memory leaks"
  - "Statistical tests handle degenerate cases (identical inputs, zero variance) gracefully"

requirements-completed: [BASE-01, BASE-02, BASE-03]

# Metrics
duration: 11min
completed: 2026-03-08
---

# Phase 03 Plan 04: Training Utility and Statistical Testing Summary

**Complete training loop with TensorBoard observability, Optuna Bayesian HP search, and rigorous statistical comparison utilities (paired t-test, Wilcoxon, Holm-Bonferroni correction)**

## Performance

- **Duration:** 11 min
- **Started:** 2026-03-08T23:12:00Z
- **Completed:** 2026-03-08T23:23:22Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Full training loop with LR warmup, early stopping, AMP, gradient stats, VRAM monitoring, TensorBoard logging, checkpointing, and graceful SIGINT shutdown
- Optuna integration for Bayesian hyperparameter search with MedianPruner and configurable search space (lr, weight_decay, optimizer, scheduler, batch_size, gradient_clip)
- Statistical testing module with paired t-test, Wilcoxon signed-rank, Cohen's d, Holm-Bonferroni multiple testing correction, and t-distribution confidence intervals
- Plotting utilities for convergence curves (dual-axis loss/accuracy), comparison bar charts with error bars, and parameter count tables
- 15 new tests passing, 434 total suite green

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for training utility** - `97863ad` (test)
2. **Task 1 GREEN: Implement training utility with Optuna** - `f2ab177` (feat)
3. **Task 2 GREEN: Statistical testing, plotting, and exports** - `c88c388` (feat)

_Note: Task 2 RED was verified via import error (ModuleNotFoundError) before implementation._

## Files Created/Modified

- `src/octonion/baselines/_trainer.py` - Training utility: seed_everything, train_model, evaluate, save/load_checkpoint, run_optuna_study
- `src/octonion/baselines/_stats.py` - Statistical tests: paired_comparison, cohen_d, holm_bonferroni, confidence_interval
- `src/octonion/baselines/_plotting.py` - Visualization: plot_convergence, plot_comparison_bars, plot_param_table
- `src/octonion/baselines/__init__.py` - Updated exports for trainer, stats, and plotting modules
- `tests/test_baselines_trainer.py` - 15 tests covering training, Optuna, stats, and plotting

## Decisions Made

- **Warmup via direct LR manipulation:** Instead of composing LambdaLR with the main scheduler, warmup directly sets optimizer param group LR values. This avoids scheduler interaction issues where LambdaLR would multiply with CosineAnnealingLR in unexpected ways.
- **Gradient stats after backward:** Gradient norms are collected after scaler.step()/scaler.update() but before the next optimizer.zero_grad(), ensuring we capture the actual gradients used for the parameter update.
- **NaN-safe statistical tests:** When paired_comparison receives identical input lists, scipy.stats.ttest_rel returns NaN. The implementation detects this case and returns p-value=1.0 (no difference), which is the correct interpretation.
- **Optuna verbosity suppressed:** Set optuna.logging.set_verbosity(WARNING) to avoid noisy trial output during search, keeping logs focused on training metrics.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed NaN edge case in paired_comparison for identical inputs**
- **Found during:** Task 2 GREEN (statistical testing)
- **Issue:** scipy.stats.ttest_rel returns NaN for t-statistic and p-value when all paired differences are zero (identical input lists)
- **Fix:** Added explicit check for all-zero differences, returning t_stat=0.0 and t_p_value=1.0
- **Files modified:** src/octonion/baselines/_stats.py
- **Verification:** test_paired_comparison_known_values passes with identical lists
- **Committed in:** c88c388 (part of Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential for correctness when comparing identical results (common in early development). No scope creep.

## Issues Encountered

None beyond the auto-fixed deviation above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Training utility ready for all downstream experimental phases (4, 5, 6, 7)
- Optuna integration ready for per-algebra hyperparameter search
- Statistical testing ready for rigorous pairwise comparison with multiple testing correction
- Plotting utilities ready for publication-quality convergence curves and comparison charts
- All 434 tests green, no regressions

## Self-Check: PASSED

- All 5 created/modified files exist on disk
- All 3 task commits found in git history (97863ad, f2ab177, c88c388)

---
*Phase: 03-baseline-implementations*
*Completed: 2026-03-08*
