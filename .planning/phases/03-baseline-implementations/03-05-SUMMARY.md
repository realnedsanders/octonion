---
phase: 03-baseline-implementations
plan: 05
subsystem: baselines
tags: [pytorch, comparison-runner, experiment-management, statistical-testing, parameter-matching, manifest, plotting]

# Dependency graph
requires:
  - phase: 03-baseline-implementations
    plan: 03
    provides: "AlgebraNetwork, RNN cells, param_report, find_matched_width"
  - phase: 03-baseline-implementations
    plan: 04
    provides: "train_model, seed_everything, paired_comparison, holm_bonferroni, plot_convergence, plot_comparison_bars, plot_param_table"
provides:
  - "run_comparison: multi-algebra multi-seed experiment orchestrator with provenance"
  - "ComparisonReport: serializable dataclass with per_run metrics, pairwise stats, Holm-Bonferroni corrected p-values"
  - "Experiment directory structure: {task}/{algebra}/{seed}/ with config.json, metrics.json, convergence.png"
  - "Auto-manifest (manifest.json) with config hashes and status tracking"
  - "_SimpleAlgebraMLP: trainable param-matched model for comparison experiments"
  - "Complete baselines package API (40 exports) accessible via octonion.baselines"
affects: [03-06, 04-numerical-stability, 05-optimization-landscape, 06-benchmark-experiments, 07-density-geometric]

# Tech tracking
tech-stack:
  added: []
  patterns: ["_build_simple_mlp returns trainable _SimpleAlgebraMLP with proper reshape (not nn.Sequential)", "config hashing via SHA256 for experiment reproducibility tracking", "ref_hidden parameter controls reference model size for param matching"]

key-files:
  created:
    - src/octonion/baselines/_comparison.py
    - tests/test_baselines_comparison.py
  modified:
    - src/octonion/baselines/_param_matching.py
    - src/octonion/baselines/__init__.py
    - src/octonion/__init__.py

key-decisions:
  - "_build_simple_mlp upgraded from nn.Sequential to _SimpleAlgebraMLP class with proper reshape between input/output projections and algebra-valued hidden layers, making it both countable and trainable"
  - "Reference model for param matching uses first algebra in config list (not always octonion), with ref_hidden controlling algebra-unit width"
  - "Test fixtures use input_dim=32 and ref_hidden=25 to ensure per-algebra width steps are <1% of total params for reliable matching"
  - "Comparison runner uses _build_simple_mlp for both param counting and training (ensures exact param count match)"

patterns-established:
  - "run_comparison returns ComparisonReport with all results serializable for downstream analysis"
  - "Experiment provenance: config_hash + timestamp + manifest.json for reproducibility tracking"
  - "network_config_overrides dict pattern for passing ref_hidden, depth, topology to run_comparison"

requirements-completed: [BASE-01, BASE-02, BASE-03]

# Metrics
duration: 13min
completed: 2026-03-08
---

# Phase 03 Plan 05: Comparison Runner and Experiment Management Summary

**Multi-algebra comparison runner with parameter-matched training, pairwise statistical testing (Holm-Bonferroni corrected), structured experiment directories with auto-manifest, and complete baselines API (40 exports)**

## Performance

- **Duration:** 13 min
- **Started:** 2026-03-08T23:39:59Z
- **Completed:** 2026-03-08T23:53:29Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Comparison runner (`run_comparison`) that orchestrates parameter-matched training of multiple algebras across multiple seeds, producing statistically rigorous results
- ComparisonReport dataclass with per-run metrics, pairwise statistical comparisons (t-test, Wilcoxon, Cohen's d), and Holm-Bonferroni corrected p-values
- Experiment directory structure with per-run config.json, metrics.json, convergence plots, and auto-manifest tracking
- `_build_simple_mlp` upgraded to proper trainable `_SimpleAlgebraMLP` with correct reshape handling between real projections and algebra-valued hidden layers
- Complete baselines package API with 40 exports accessible from `octonion.baselines`, baselines subpackage exposed from `octonion` package
- 10 new tests passing (comparison runner), 490 total suite green

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing comparison tests** - `d6c72d0` (test)
2. **Task 1 GREEN: Implement comparison runner** - `158ad91` (feat)
3. **Task 2: Finalize baselines API** - `e669142` (feat)

## Files Created/Modified

- `src/octonion/baselines/_comparison.py` - ComparisonReport dataclass, run_comparison orchestrator, _config_hash, _update_manifest
- `src/octonion/baselines/_param_matching.py` - _SimpleAlgebraMLP class replacing nn.Sequential for trainable param-matched models
- `tests/test_baselines_comparison.py` - 10 tests: directory structure, report contents, manifest, pairwise stats, param tolerance, plots, config/metrics JSON
- `src/octonion/baselines/__init__.py` - Added run_comparison and ComparisonReport exports
- `src/octonion/__init__.py` - Added baselines subpackage reference and __all__ entry

## Decisions Made

- **Trainable param-matched models:** `_build_simple_mlp` was previously an `nn.Sequential` that could not train (missing reshape between input projection and algebra layers). Upgraded to `_SimpleAlgebraMLP` class with proper forward pass, making it usable for both param counting and actual training in the comparison runner.
- **Reference algebra from config:** The comparison runner uses the first algebra in `config.algebras` as the reference for param matching (not hardcoded octonion), making it usable with any algebra subset.
- **Test input dimensions:** Tests use `input_dim=32` and `ref_hidden=25` to ensure the discrete width-step between algebra units is <1% of total param count. Small `input_dim` values (e.g., 4) cause width steps that exceed the 1% tolerance for param matching.
- **Config hashing:** SHA256 hash of deterministic JSON serialization of ComparisonConfig for experiment reproducibility tracking.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed _build_simple_mlp to be trainable**
- **Found during:** Task 1 GREEN (comparison runner training)
- **Issue:** `_build_simple_mlp` used `nn.Sequential` which cannot handle the reshape between `nn.Linear(input_dim, hidden*dim)` and `AlgebraLinear(hidden, hidden)`. Forward pass crashes with shape mismatch for non-real algebras.
- **Fix:** Created `_SimpleAlgebraMLP(nn.Module)` class with proper `forward()` that reshapes `[B, hidden*dim]` to `[B, hidden, dim]` between input projection and algebra layers, and flattens back before output projection.
- **Files modified:** src/octonion/baselines/_param_matching.py
- **Verification:** All 4 algebras forward pass correctly, param counts unchanged from original nn.Sequential
- **Committed in:** 158ad91 (Task 1 GREEN commit)

**2. [Rule 1 - Bug] Fixed test input dimensions for param matching tolerance**
- **Found during:** Task 1 GREEN (param matching in comparison runner)
- **Issue:** With `input_dim=4`, the discrete jump between adjacent algebra widths is >1% of total params for Complex and other algebras, causing `find_matched_width` to raise ValueError even when the code is correct.
- **Fix:** Changed test data to `input_dim=32` and added `ref_hidden=25` override so that per-unit jumps are <1% of total for all tested algebra pairs.
- **Files modified:** tests/test_baselines_comparison.py
- **Verification:** R+C matching within 0.33% at input_dim=32
- **Committed in:** 158ad91 (Task 1 GREEN commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes essential for correct operation. The nn.Sequential model was fundamentally non-trainable for non-real algebras. The input dimension fix ensures tests exercise the full pipeline without false positives from param matching limits.

## Issues Encountered

None beyond the auto-fixed deviations above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Comparison runner ready for all downstream experimental phases (5, 6, 7)
- Statistical testing infrastructure ready for publication-quality pairwise comparisons
- Experiment provenance tracking ready via config hashing and manifest
- Complete baselines API accessible from single import point
- 490 total tests green, no regressions
- Phase 3 Plan 6 (benchmark reproduction) can now use run_comparison for R/C/H/O experiments

## Self-Check: PASSED

- All 5 created/modified files exist on disk
- All 3 task commits found in git history (d6c72d0, 158ad91, e669142)

---
*Phase: 03-baseline-implementations*
*Completed: 2026-03-08*
