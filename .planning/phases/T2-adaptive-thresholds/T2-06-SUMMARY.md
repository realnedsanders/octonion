---
phase: T2-adaptive-thresholds
plan: 06
subsystem: sweep-infrastructure
tags: [sweep, adaptive-thresholds, algebraic-purity, noise-interaction, sqlite, hyperparameter-tuning]

# Dependency graph
requires:
  - phase: T2-adaptive-thresholds
    provides: "SweepRunner, SweepConfig from T2-03; run_adaptive_sweep.py from T2-05; AlgebraicPurityPolicy from T2-01"
provides:
  - "run_purity_sweep.py: AlgebraicPurityPolicy sweep with independent signal testing and noise interaction"
  - "Purity hyperparameter grid: assoc_weight x sim_weight x sensitivity x base_assoc x sim_threshold"
  - "Independent signal analysis: Phase A (assoc only), Phase B (sim only), Phase C (combined)"
  - "Noise interaction analysis across all strategies (global, ema, mean_std, depth, purity)"
  - "Cross-strategy comparison table: purity vs all other strategies"
affects: [T2-07, T2-08, T2-09, T2-10, T2-11]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Three-phase independent signal testing (assoc-only, sim-only, combined)", "Noise interaction as cross-strategy characterization dimension", "Config ID offsets (900K purity, 950K noise) for collision avoidance"]

key-files:
  created:
    - scripts/sweep/run_purity_sweep.py
  modified: []

key-decisions:
  - "Config ID offset 900K for purity, 950K for noise interaction to avoid collision with strategies 1-3 (300-800K)"
  - "Phase C combined grid reduced from 4x4 to 3x3 assoc/sim weight for tractability (~510 vs 720 configs/benchmark)"
  - "Noise interaction runs top-10 purity + top-5 per strategy + top-5 global for comprehensive cross-strategy analysis"
  - "Script auto-runs purity sweep before noise interaction if no purity results exist in DB"

patterns-established:
  - "Independent signal testing: sweep each signal alone, then combined, compare to identify dominant signal"
  - "Noise interaction as cross-strategy validation dimension using same NOISE_LEVELS grid"
  - "Graceful fallback to default values when DB has no prior strategy results"

requirements-completed: []

# Metrics
duration: 3min
completed: 2026-03-30
---

# Phase T2 Plan 06: Algebraic Purity Strategy Sweep Summary

**AlgebraicPurityPolicy sweep with three-phase independent signal testing (assoc variance, sim variance, combined) and cross-strategy noise interaction analysis**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-30T02:36:21Z
- **Completed:** 2026-03-30T02:40:04Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Created run_purity_sweep.py with full AlgebraicPurityPolicy hyperparameter grid sweep
- Three-phase independent signal testing: assoc_weight-only (Phase A), sim_weight-only (Phase B), combined (Phase C)
- Noise interaction sweep across all 5 strategies (global, ema, mean_std, depth, purity) with 4 noise levels
- Cross-strategy comparison table showing purity vs all prior strategies with win counts and mean delta
- Independent signal analysis identifies which purity signal (associator norm variance vs routing key similarity variance) matters most per benchmark
- CLI verified working in container with argparse --help

## Task Commits

Each task was committed atomically:

1. **Task 1: Algebraic purity strategy sweep with noise interaction** - `3570832` (feat)

## Files Created/Modified
- `scripts/sweep/run_purity_sweep.py` - Strategy 4 sweep with independent signal testing, noise interaction, cross-strategy comparison

## Decisions Made
- Config ID offset 900K for purity configs, 950K for noise interaction configs to avoid collision with strategies 1-3 (300K ema, 400K mean_std, 500K depth, 600-800K expanded)
- Phase C combined grid uses 3x3 assoc_weight x sim_weight subset (0.3, 0.5, 0.7 for each) instead of full 4x4 to reduce from 720 to ~510 configs/benchmark
- Noise interaction queries top-10 purity, top-5 per strategy 1-3, and top-5 global for comprehensive cross-strategy characterization
- Script auto-detects existing purity results in DB and skips re-running if already present

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Worktree does not contain sweep infrastructure files (sweep_runner.py, run_adaptive_sweep.py, etc.) since those were created by parallel agents. Copied from main repo to enable import verification. These files will merge when all T2 worktrees converge.

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all functionality is fully wired. The script is complete and verified importable.

## Next Phase Readiness
- Purity sweep ready to execute once cached features exist (from T2-02) and global sweep has run (from T2-04)
- Noise interaction mode available after purity sweep and adaptive sweeps (T2-05) have run
- Results stored in same SQLite database alongside all prior sweep data for cross-strategy comparison
- Infrastructure reusable by T2-07 (meta-trie) through T2-11

## Self-Check: PASSED

- [x] scripts/sweep/run_purity_sweep.py exists
- [x] T2-06-SUMMARY.md exists
- [x] Commit 3570832 found

---
*Phase: T2-adaptive-thresholds*
*Completed: 2026-03-30*
