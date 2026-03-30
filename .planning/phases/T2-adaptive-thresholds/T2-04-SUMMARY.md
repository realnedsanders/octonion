---
phase: T2-adaptive-thresholds
plan: 04
subsystem: sweep-infrastructure
tags: [sweep, sqlite, matplotlib, threshold-sensitivity, pareto, visualization, heatmap]

# Dependency graph
requires:
  - phase: T2-adaptive-thresholds
    provides: "SweepRunner, SweepConfig, generate_global_sweep_configs, cached features (.pt files)"
provides:
  - "run_global_sweep.py: 3-phase progressive threshold sweep orchestration script"
  - "sweep_plots.py: 6 visualization functions for sweep analysis (heatmap, 1D, Pareto, noise, epoch, batch)"
affects: [T2-05, T2-06, T2-07, T2-08, T2-09, T2-10, T2-11]

# Tech tracking
tech-stack:
  added: []
  patterns: ["3-phase progressive sweep (3D core -> consolidation -> epoch)", "SQLite-based top-N config selection between phases", "Pareto frontier computation for accuracy vs node count"]

key-files:
  created:
    - scripts/sweep/run_global_sweep.py
    - scripts/sweep/sweep_plots.py
  modified: []

key-decisions:
  - "3-phase progressive design: Phase 1 sweeps assoc x sim x noise (704/benchmark), Phase 2 sweeps consolidation on top-5, Phase 3 sweeps epochs on top-10"
  - "Phase inter-dependencies via SQLite top-N queries: top-5 pairs for Phase 2, top-10 configs for Phase 3"
  - "Pareto frontier computed as cumulative min node count over accuracy-descending sort"
  - "Import path uses sys.path.insert for scripts/ directory (scripts/ is not a Python package)"

patterns-established:
  - "Progressive sweep: run broad sweep first, refine top configs with additional dimensions"
  - "Consistent plot styling: HEATMAP_FIGSIZE=(10,8), LINE_FIGSIZE=(10,6), dpi=150, tight_layout"
  - "Organized plot output: heatmaps/, line_plots/, pareto/, noise/, epoch_curves/ subdirectories"

requirements-completed: []

# Metrics
duration: 4min
completed: 2026-03-30
---

# Phase T2 Plan 04: Global Sweep and Visualization Summary

**3-phase progressive threshold sweep script (704 configs/benchmark core sweep) with 6 matplotlib visualization functions for heatmaps, line plots, Pareto frontiers, and noise interaction analysis**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-30T02:15:39Z
- **Completed:** 2026-03-30T02:19:45Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Created run_global_sweep.py with 3-phase progressive sweep design: Phase 1 (assoc x sim x noise, ~3520 total configs), Phase 2 (consolidation on top-5), Phase 3 (epochs on top-10)
- Created sweep_plots.py with 6 visualization functions: plot_heatmap, plot_1d_sweep, plot_pareto_frontier, plot_noise_interaction, plot_epoch_curves, generate_all_plots
- Both scripts verified to import correctly in the container environment
- CLI interfaces for both scripts with argparse, consistent with existing sweep infrastructure

## Task Commits

Each task was committed atomically:

1. **Task 1: Global sweep execution script** - `a44f720` (feat)
2. **Task 2: Sweep visualization functions** - `0d3fbc5` (feat)

## Files Created/Modified
- `scripts/sweep/run_global_sweep.py` - 3-phase progressive sweep orchestration with CLI interface
- `scripts/sweep/sweep_plots.py` - 6 visualization functions reading from SQLite, producing organized PNGs
- `scripts/sweep/__init__.py` - Package init for sweep module
- `scripts/sweep/sweep_runner.py` - Dependency from T2-03 (included for worktree completeness)
- `scripts/sweep/cache_features.py` - Dependency from T2-02 (included for worktree completeness)

## Decisions Made
- 3-phase progressive sweep design avoids combinatorial explosion: Phase 1 sweeps the core 3D grid (assoc x sim x noise = 704 configs/benchmark), Phase 2 refines consolidation on top-5 pairs, Phase 3 refines epochs on top-10 configs
- Phase-to-phase config selection uses SQLite AVG(accuracy) queries grouped by threshold parameters, selecting top-N by mean accuracy across benchmarks
- Pareto frontier uses accuracy-descending sort with cumulative minimum node count tracking
- Import path: sys.path.insert(0, scripts_dir) since scripts/ lacks __init__.py and is not an installable package

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed import path for container environment**
- **Found during:** Task 1 (Global sweep execution script)
- **Issue:** Initial `from scripts.sweep.sweep_runner import ...` failed because `scripts/` directory has no `__init__.py` and is not on sys.path in the container
- **Fix:** Changed to `sys.path.insert(0, scripts_dir)` with `from sweep.sweep_runner import ...`
- **Files modified:** scripts/sweep/run_global_sweep.py
- **Verification:** `docker compose run --rm dev uv run python scripts/sweep/run_global_sweep.py --help` succeeds
- **Committed in:** a44f720 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Essential fix for container compatibility. No scope creep.

## Issues Encountered
- Worktree does not contain T2-02/T2-03 files (sweep_runner.py, cache_features.py) since those were created by parallel agents on different branches. Copied from main repo to enable this plan's work. These files will merge together when all T2 worktrees converge.

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all functionality is fully wired. Both scripts are complete and verified importable.

## Next Phase Readiness
- Global sweep ready to execute once cached features exist (from T2-02 cache_features.py)
- Visualization functions ready to process any populated sweep.db
- Infrastructure reusable by T2-05 (adaptive strategy sweeps) and beyond

## Self-Check: PASSED

- [x] scripts/sweep/run_global_sweep.py exists
- [x] scripts/sweep/sweep_plots.py exists
- [x] T2-04-SUMMARY.md exists
- [x] Commit a44f720 found
- [x] Commit 0d3fbc5 found

---
*Phase: T2-adaptive-thresholds*
*Completed: 2026-03-30*
