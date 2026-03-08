---
phase: 02-ghr-calculus
plan: 03
subsystem: calculus
tags: [composition, parenthesization, catalan, chain-rule, non-associativity, binary-tree, octonion, pytorch]

# Dependency graph
requires:
  - phase: 02-ghr-calculus
    provides: "Analytic 8x8 Jacobians for all 7 primitives, torch.autograd.Function subclasses with create_graph=True"
  - phase: 01-octonionic-algebra
    provides: "STRUCTURE_CONSTANTS tensor, octonion_mul, octonion_exp/log/conjugate/inverse, OctonionLinear"
provides:
  - "CompositionBuilder API for parenthesized octonionic compositions"
  - "Binary tree types (Leaf, Node) for representing computation parenthesizations"
  - "all_parenthesizations(n) generating C_{n-1} Catalan trees"
  - "evaluate_tree with autograd-tracked computation graph preserving parenthesization"
  - "compose_jacobians: parenthesization-aware chain rule Jacobian computation"
  - "naive_chain_rule_jacobian: left-to-right baseline for comparison"
  - "inspect_tree and tree_to_string for parenthesization debugging"
  - "SC-2 verified: all 14 Catalan(4) parenthesizations pass gradient check at 1e-5"
  - "SC-3 verified: naive vs correct gradients differ by >100% relative error"
  - "Quantitative parenthesization report and depth scaling analysis"
affects: [02-04-analyticity, 04-numerical-stability, 05-optimization-landscape]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Binary tree representation with frozen dataclasses for parenthesization patterns"
    - "Recursive Catalan number tree generation for exhaustive parenthesization enumeration"
    - "evaluate_tree dispatches to autograd Functions for correct backward computation"
    - "compose_jacobians traverses tree bottom-up composing Jacobians via chain rule"
    - "Naive chain rule assumes left-to-right association as incorrectness baseline"

key-files:
  created:
    - "src/octonion/calculus/_composition.py"
    - "src/octonion/calculus/_chain_rule.py"
    - "src/octonion/calculus/_inspector.py"
    - "tests/test_composition.py"
    - "scripts/demo_naive_vs_correct.py"
    - "results/parenthesization_report.json"
    - "results/naive_vs_correct.json"
  modified:
    - "src/octonion/calculus/__init__.py"

key-decisions:
  - "all_parenthesizations generates right-to-left first (fully right-associated is trees[0], fully left-associated is trees[-1])"
  - "evaluate_tree dispatches to OctonionMulFunction.apply (and other autograd Functions) to preserve computation graph for parenthesization-aware backward passes"
  - "compose_jacobians uses analytic jacobian_mul for bottom-up tree Jacobian composition, avoiding numeric differentiation"
  - "Naive chain rule defined as always-left-to-right association, matching common incorrect assumption"

patterns-established:
  - "Binary tree types (Leaf/Node) as first-class representation of octonionic composition structure"
  - "Catalan enumeration for exhaustive parenthesization testing"
  - "Autograd Function dispatch table for mixed-operation tree evaluation"
  - "SC verification pattern: exhaustive gradient check + quantitative report + JSON artifact"

requirements-completed: [FOUND-02]

# Metrics
duration: 13min
completed: 2026-03-08
---

# Phase 2 Plan 03: Parenthesization-Aware Composition and Chain Rule Summary

**CompositionBuilder with Catalan tree enumeration, parenthesization-aware chain rule, SC-2 verified (all 14 patterns), SC-3 quantified (>100% naive-vs-correct error with depth scaling)**

## Performance

- **Duration:** 13 min
- **Started:** 2026-03-08T18:36:47Z
- **Completed:** 2026-03-08T18:49:47Z
- **Tasks:** 2 (both TDD: RED-GREEN)
- **Files created:** 7
- **Files modified:** 1

## Accomplishments

- Implemented CompositionBuilder API with binary tree types (Leaf, Node), all_parenthesizations generating C_{n-1} Catalan trees, and evaluate_tree with autograd-tracked dispatch to octonionic autograd Functions
- Created parenthesization-aware chain rule (compose_jacobians) that computes correct Jacobians by bottom-up tree traversal, and naive_chain_rule_jacobian baseline that assumes left-to-right association
- Verified SC-2: All 14 Catalan(4) parenthesizations of 5 operands pass gradient check at 1e-5 relative error, confirmed via both custom comparison and torch.autograd.gradcheck
- Verified SC-3: Naive chain rule produces >100% relative error compared to correct parenthesization-aware computation, with error growing from 11.1 (depth 3) to 73.2 (depth 7)
- Mixed-operation compositions (mul + exp + log + conjugate + inverse) pass gradient checks
- Quantitative reports saved as JSON artifacts with per-pattern errors and depth scaling analysis
- Full 334-test suite passes with no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: CompositionBuilder, binary trees, chain rule, inspector** - `177ba1c` (feat)
2. **Task 2: SC-2 exhaustive testing, SC-3 demo, mixed-op compositions** - `58458cc` (feat)

## Files Created/Modified

- `src/octonion/calculus/_composition.py` - Leaf/Node frozen dataclasses, all_parenthesizations (Catalan enumeration), evaluate_tree (autograd dispatch), CompositionBuilder class, build_mixed_tree
- `src/octonion/calculus/_chain_rule.py` - compose_jacobians (bottom-up tree Jacobian composition), naive_chain_rule_jacobian (left-to-right baseline)
- `src/octonion/calculus/_inspector.py` - tree_to_string (mathematical notation), inspect_tree (ASCII art tree display)
- `tests/test_composition.py` - 36 tests: binary tree structure, Catalan numbers, evaluation, non-associativity, chain rule, naive-vs-correct, inspector, SC-2 exhaustive, SC-3 demonstration, mixed operations
- `scripts/demo_naive_vs_correct.py` - Standalone demo with 1000-trial statistical analysis, depth scaling (3->5->7), confidence intervals, cosine similarity
- `results/parenthesization_report.json` - SC-2 quantitative report with per-pattern gradient errors
- `results/naive_vs_correct.json` - SC-3 depth scaling analysis and statistical results
- `src/octonion/calculus/__init__.py` - Updated exports: CompositionBuilder, Leaf, Node, all_parenthesizations, compose_jacobians, naive_chain_rule_jacobian, inspect_tree, tree_to_string

## Decisions Made

- **Tree ordering**: all_parenthesizations generates right-associated trees first (trees[0] = fully right-associated, trees[-1] = fully left-associated). This follows naturally from the recursive Catalan generation where split=1 produces left-leaf + right-subtree.
- **Autograd dispatch**: evaluate_tree uses OctonionMulFunction.apply (not raw octonion_mul) to ensure the computation graph correctly tracks parenthesization for backward passes.
- **Naive baseline definition**: The "naive" chain rule always assumes left-to-right association `((o0*o1)*o2)*...` regardless of actual tree structure, matching the common incorrect assumption that multiplication order doesn't matter.
- **Analytic chain rule**: compose_jacobians uses analytic jacobian_mul for each node's Jacobian (not numeric differentiation), making the composition exact and fast.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed non-leaf tensor gradient access in exhaustive tests**
- **Found during:** Task 2 (SC-2 exhaustive testing)
- **Issue:** `torch.randn(..., requires_grad=True) * 0.5` creates a non-leaf tensor, causing `.grad` to be None after backward
- **Fix:** Changed to `(torch.randn(...) * 0.5).requires_grad_(True)` to keep tensors as leaf nodes
- **Files modified:** tests/test_composition.py
- **Verification:** All 36 tests pass including exhaustive SC-2
- **Committed in:** 58458cc

**2. [Rule 1 - Bug] Fixed tree ordering assumption in demo script**
- **Found during:** Task 2 (demo script verification)
- **Issue:** Demo assumed trees[0] was left-associated and trees[-1] was right-associated, but the opposite is true
- **Fix:** Corrected to skip left-associated tree (trees[-1]) in statistical analysis and use trees[0] (right-associated) for depth scaling
- **Files modified:** scripts/demo_naive_vs_correct.py, tests/test_composition.py
- **Verification:** Demo output shows non-zero gradient differences with correct tree labels
- **Committed in:** 58458cc

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both auto-fixes necessary for correctness. No scope creep.

## Issues Encountered

None beyond the two auto-fixed bugs above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- CompositionBuilder and all_parenthesizations ready for use in numerical stability testing (Phase 4)
- compose_jacobians provides ground-truth Jacobians for optimizer landscape analysis (Phase 5)
- SC-2 and SC-3 verified, providing quantitative evidence that parenthesization matters for octonionic gradients
- Mixed-operation composition support enables testing complex computation patterns
- Inspector tools available for debugging parenthesization issues in future work
- No blockers for Plan 02-04 (analyticity) or downstream phases

## Self-Check: PASSED

All 7 created files and 1 modified file verified on disk. Both task commits (177ba1c, 58458cc) verified in git log.

---
*Phase: 02-ghr-calculus*
*Completed: 2026-03-08*
