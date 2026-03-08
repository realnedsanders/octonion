---
phase: 02-ghr-calculus
plan: 02
subsystem: calculus
tags: [autograd, gradcheck, wirtinger, backward, create_graph, octonion, pytorch]

# Dependency graph
requires:
  - phase: 02-ghr-calculus
    provides: "Analytic 8x8 Jacobians for all 7 primitives, numeric Jacobian utility, GHR Wirtinger pair"
  - phase: 01-octonionic-algebra
    provides: "STRUCTURE_CONSTANTS tensor, octonion_mul, octonion_exp/log, Octonion class, inner_product, cross_product, OctonionLinear"
provides:
  - "torch.autograd.Function subclasses for all 7 octonionic primitives with create_graph=True support"
  - "Custom octonion_gradcheck with per-component errors and Wirtinger pair validation"
  - "Custom octonion_gradgradcheck for second-order derivative verification"
  - "SC-1 verified: OctonionLinear single layer gradient check at float64"
affects: [02-03-chain-rule, 02-04-analyticity, 04-numerical-stability, 05-optimization-landscape]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Autograd Functions save only inputs in ctx, recompute Jacobians in backward for create_graph=True"
    - "Einsum-based backward operations for differentiable double backward"
    - "Custom gradcheck computes column-by-column autograd Jacobian via unit vector backpropagation"
    - "Cross product backward zeros grad_output real component before einsum to avoid C[i,j,0] contamination"

key-files:
  created:
    - "src/octonion/calculus/_autograd_functions.py"
    - "src/octonion/calculus/_gradcheck.py"
    - "tests/test_autograd.py"
  modified:
    - "src/octonion/calculus/__init__.py"

key-decisions:
  - "Autograd Functions recompute Jacobians in backward (not cached from forward) for create_graph=True compatibility"
  - "Cross product backward zeros grad_output[..., 0] to exclude C[i,j,0] terms from imaginary-only output"
  - "Inner product Function returns scalar (shape []) matching torch.sum convention, not [..., 1]"
  - "Exp/log backward use sqrt(r_sq + 1e-30) for sqrt gradient stability, separate near_zero threshold at 1e-12"

patterns-established:
  - "Autograd Functions operate on raw [..., 8] tensors, not Octonion instances"
  - "All backward ops are differentiable PyTorch ops (einsum, exp, cos, sin) for double backward"
  - "OctonionLinear already works with autograd via PyTorch native einsum; custom Functions add Wirtinger formalism"
  - "Custom gradcheck returns structured dict with passed/max_abs_error/max_rel_error/per_component_errors/wirtinger_passed"

requirements-completed: [FOUND-02]

# Metrics
duration: 8min
completed: 2026-03-08
---

# Phase 2 Plan 02: Autograd Functions and Custom Gradcheck Summary

**7 torch.autograd.Function subclasses with create_graph=True double backward, custom octonion gradcheck with Wirtinger validation, and SC-1 verified at float64**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-08T18:24:42Z
- **Completed:** 2026-03-08T18:33:33Z
- **Tasks:** 2 (both TDD: RED-GREEN)
- **Files created:** 3
- **Files modified:** 1

## Accomplishments

- Implemented torch.autograd.Function subclasses for all 7 octonionic primitives (mul, exp, log, conjugate, inverse, inner_product, cross_product) with full create_graph=True support for Hessian computation
- Created custom octonion_gradcheck utility that reports per-component octonionic errors and validates both Wirtinger derivatives (df/do and df/do*)
- Verified SC-1: single OctonionLinear layer gradient check passes at float64 with relative error well under 1e-5
- All 43 autograd tests pass (forward, backward, gradcheck, gradgradcheck, batched, custom gradcheck, SC-1)
- Full 283-test suite passes with no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for 7 autograd Functions** - `a33bb0c` (test)
2. **Task 1 GREEN: Implement 7 autograd Functions** - `4d1b3bd` (feat)
3. **Task 2 RED: Failing tests for custom gradcheck and SC-1** - `a972e27` (test)
4. **Task 2 GREEN: Implement custom gradcheck, verify SC-1** - `997c6fb` (feat)

## Files Created/Modified

- `src/octonion/calculus/_autograd_functions.py` - 7 torch.autograd.Function subclasses (OctonionMulFunction, OctonionExpFunction, OctonionLogFunction, OctonionConjugateFunction, OctonionInverseFunction, OctonionInnerProductFunction, OctonionCrossProductFunction)
- `src/octonion/calculus/_gradcheck.py` - Custom octonion_gradcheck and octonion_gradgradcheck utilities with per-component error reporting and Wirtinger pair validation
- `tests/test_autograd.py` - 43 tests: forward correctness, backward vs analytic Jacobian, torch.autograd.gradcheck, gradgradcheck, batched inputs, custom gradcheck on all 7 primitives, wrong-backward detection, SC-1
- `src/octonion/calculus/__init__.py` - Updated exports: all 7 autograd Functions, octonion_gradcheck, octonion_gradgradcheck

## Decisions Made

- **Recompute Jacobians in backward:** Rather than caching intermediate computations from forward (which runs in no-grad mode), all backward passes recompute from saved inputs using differentiable ops. This ensures create_graph=True works for double backward/Hessian computation.
- **Cross product backward zeros grad_output real:** The cross product Jacobian has all-zero row 0 (output real component is always 0). The einsum over all k would include spurious C[i,j,0] contributions, so we explicitly zero grad_output[..., 0] before the einsum.
- **Inner product returns scalar shape:** OctonionInnerProductFunction returns shape [] (not [..., 1]) matching torch.sum convention. This is consistent with how loss functions are typically scalar-valued.
- **Exp/log backward sqrt stability:** Use sqrt(r_sq + 1e-30) to avoid NaN gradients from sqrt at exactly zero, with a separate near_zero threshold at 1e-12 for the sinc/outer_coeff formulas.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed cross product backward grad_output contamination**
- **Found during:** Task 1 (autograd Functions implementation)
- **Issue:** Cross product backward used full grad_output in einsum, but C[i,j,0] terms (imaginary squaring) contributed spurious gradients when grad_output[0] was non-zero
- **Fix:** Zero out grad_output[..., 0] before einsum computation
- **Files modified:** src/octonion/calculus/_autograd_functions.py
- **Verification:** Backward matches analytic Jacobian, gradcheck passes
- **Committed in:** 4d1b3bd

**2. [Rule 1 - Bug] Fixed inner_product backward test shape mismatch**
- **Found during:** Task 1 (test correction)
- **Issue:** Test passed grad_out of shape [1] but inner product output is scalar shape []
- **Fix:** Simplified test to use backward() with no argument and verify grad_a == b
- **Files modified:** tests/test_autograd.py
- **Committed in:** 4d1b3bd

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both auto-fixes necessary for correctness. No scope creep.

## Issues Encountered

None beyond the two auto-fixed bugs above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All 7 autograd Functions ready for parenthesization-aware chain rule composition (Plan 02-03)
- Custom octonion_gradcheck available for exhaustive parenthesization gradient verification
- create_graph=True verified for all Functions, enabling Hessian computation for Phase 5
- SC-1 verified, providing confidence in gradient correctness for training workflows
- No blockers for Plans 02-03 (chain rule) or 02-04 (analyticity)

## Self-Check: PASSED

All 4 created/modified files verified on disk. All 4 task commits verified in git log.

---
*Phase: 02-ghr-calculus*
*Completed: 2026-03-08*
