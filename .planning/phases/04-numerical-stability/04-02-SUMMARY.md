---
phase: 04-numerical-stability
plan: 02
subsystem: numerical-stability
tags: [stability-analysis, condition-numbers, depth-sweep, mitigation, float-precision]

# Dependency graph
requires:
  - phase: 04-numerical-stability
    provides: "StabilizingNorm, NetworkConfig.stabilize_every, smoke test infrastructure"
  - phase: 03-baselines
    provides: "OctonionDenseLinear, AlgebraNetwork, NetworkConfig, per-algebra initialization"
  - phase: 02-calculus
    provides: "numeric_jacobian for condition number computation"
provides:
  - "Comprehensive stability analysis script covering all 4 FOUND-03 success criteria"
  - "JSON data files for depth sweep, condition numbers, and mitigation results"
  - "PNG plot outputs for visual analysis of error accumulation and condition numbers"
  - "Empirical evidence for architecture depth/precision decisions in downstream phases"
affects: [05-go-no-go, 06-experiments, 07-density]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Dual-dtype forward pass comparison (f32 vs f64)", "Numeric Jacobian + SVD for condition number characterization", "Checkpoint-depth error measurement for mitigation demonstration"]

key-files:
  created:
    - scripts/analyze_stability.py
  modified: []

key-decisions:
  - "N_SAMPLES=500 per measurement point (middle of 100-1000 range, sufficient for tight confidence intervals)"
  - "Composition condition numbers use unbatched single-sample Jacobian (tractable for SVD at 64x64)"
  - "Mitigation measures both f64 and f32 chains independently with StabilizingNorm applied to each"

patterns-established:
  - "Dual-dtype analysis: build at f64, deepcopy to f32, compare outputs for precision loss measurement"
  - "Checkpoint-depth measurement: record intermediate errors at specific layer depths within a single forward pass"
  - "Condition number sweep: scale numeric Jacobian eps proportional to input magnitude for large-input stability"

requirements-completed: [FOUND-03]

# Metrics
duration: 4min
completed: 2026-03-20
---

# Phase 04 Plan 02: Comprehensive Stability Analysis Script Summary

**1068-line analysis script measuring error accumulation, condition numbers, float32/64 convergence, and StabilizingNorm mitigation across all four algebras (R/C/H/O) at depths 10-500**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-20T02:36:48Z
- **Completed:** 2026-03-20T02:40:45Z
- **Tasks:** 1
- **Files modified:** 1 (1 created)

## Accomplishments
- Complete `scripts/analyze_stability.py` covering all four FOUND-03 success criteria in a single standalone script
- SC-1 depth sweep measures both stripped chain and full AlgebraNetwork at depths {10,50,100,500} across 3 magnitude regimes for all 4 algebras
- SC-2 condition number characterization covers primitive octonion ops (mul/inv/exp/log), N-layer compositions (depth 2/5/10), and full network forward passes
- SC-3 float32 vs float64 stable depth computed from depth sweep data with 1e-3 threshold
- SC-4 mitigation demonstration sweeps StabilizingNorm K values {5,10,20} with checkpoint-depth error tracking
- Script outputs JSON data to `results/stability/`, PNG plots via matplotlib, and summary table to stdout

## Task Commits

Each task was committed atomically:

1. **Task 1: Create comprehensive stability analysis script** - `0a821f2` (feat)

## Files Created/Modified
- `scripts/analyze_stability.py` - 1068-line comprehensive analysis script covering all FOUND-03 criteria, with JSON/PNG/stdout outputs

## Decisions Made
- Used 500 random samples per measurement point (middle of the 100-1000 range specified in CONTEXT.md)
- Composition condition numbers computed on single unbatched inputs (no batch dimension) to keep Jacobian size tractable for SVD
- Mitigation demonstration runs independent f64/f32 chains with StabilizingNorm applied to each, measuring relative error at checkpoint depths {10,50,100,200,300,400,500}

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Analysis script ready for full execution (`docker compose run --rm dev uv run python scripts/analyze_stability.py`)
- JSON output from the script feeds Phase 5 go/no-go depth/precision decisions
- StabilizingNorm mitigation results inform whether periodic re-normalization is needed in experimental architectures
- Phase 4 complete -- all infrastructure for numerical stability characterization is in place

## Self-Check: PASSED

- scripts/analyze_stability.py exists on disk
- Commit hash 0a821f2 found in git log
- 11/11 smoke tests pass

---
*Phase: 04-numerical-stability*
*Completed: 2026-03-20*
