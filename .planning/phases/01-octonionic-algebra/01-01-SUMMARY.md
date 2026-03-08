---
phase: 01-octonionic-algebra
plan: 01
subsystem: algebra
tags: [octonion, fano-plane, cayley-dickson, pytorch, hypothesis, multiplication, structure-constants]

# Dependency graph
requires:
  - phase: 01-00
    provides: "Development container with ROCm PyTorch, uv, GPU passthrough"
provides:
  - "Installable octonion package with verified multiplication engine"
  - "Structure constants tensor [8,8,8] for vectorized octonion multiplication"
  - "Fano plane data structure with triples, incidence matrix, subalgebras"
  - "Cayley-Dickson cross-check with derived basis permutation mapping"
  - "Hypothesis strategies for octonion testing (octonions, unit_octonions, nonzero_octonions)"
  - "Tolerance constants (RTOL_FLOAT64=1e-12, ATOL_FLOAT64=1e-12)"
  - "NormedDivisionAlgebra abstract base class"
affects: [01-02, 01-03, 02-GHR-calculus, 03-baselines, 08-G2-equivariance, 09-associator-analysis]

# Tech tracking
tech-stack:
  added: [torch, pytest, hypothesis, hypothesis-torch, numpy, ruff, mypy, hatchling]
  patterns: [structure-constants-einsum, cayley-dickson-quaternion-pair, fano-plane-dataclass, hypothesis-composite-strategies]

key-files:
  created:
    - pyproject.toml
    - src/octonion/__init__.py
    - src/octonion/_types.py
    - src/octonion/_fano.py
    - src/octonion/_multiplication.py
    - src/octonion/_cayley_dickson.py
    - src/octonion/py.typed
    - tests/conftest.py
    - tests/test_multiplication.py
    - tests/test_cayley_dickson.py
  modified: []

key-decisions:
  - "Structure constants tensor has 64 non-zero entries (not 50 as estimated in plan/research -- the plan undercounted right-identity and squaring entries)"
  - "Cayley-Dickson quaternion-pair split requires permutation P=[0,1,2,5,3,7,6,4] to match Fano plane convention (pure permutation, no sign flips needed)"
  - "Distributivity tests use [-1e3, 1e3] input range to avoid float64 precision artifacts at large magnitudes"

patterns-established:
  - "Structure constants multiplication: torch.einsum('...i, ijk, ...j -> ...k', a, C, b) for vectorized octonion product"
  - "Hypothesis composite strategies producing raw torch.Tensor [8] at float64 for property-based algebra tests"
  - "Fano plane as frozen dataclass singleton with lazy-computed properties"
  - "Cayley-Dickson cross-check with explicit basis permutation mapping"

requirements-completed: [FOUND-01]

# Metrics
duration: 9min
completed: 2026-03-08
---

# Phase 01 Plan 01: Core Multiplication Engine Summary

**Fano plane structure constants tensor and Cayley-Dickson cross-check with derived basis permutation P=[0,1,2,5,3,7,6,4] matching Baez 2002 convention**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-08T05:28:28Z
- **Completed:** 2026-03-08T05:37:39Z
- **Tasks:** 2
- **Files created:** 10

## Accomplishments
- Installable Python package `octonion` with hatchling build system, all deps managed via uv
- Fano plane multiplication via [8,8,8] structure constants tensor with vectorized einsum
- Cayley-Dickson recursive multiplication matching Fano plane convention on all 64 basis pairs and random inputs
- Resolved Open Question 1 from RESEARCH.md: the CD quaternion-pair split uses a different basis labeling than Baez 2002, reconciled by pure permutation P=[0,1,2,5,3,7,6,4]
- 37 tests passing including property-based distributivity and non-associativity checks

## Task Commits

Each task was committed atomically:

1. **Task 1: Project scaffolding and package setup** - `f3bcf16` (feat)
2. **Task 2 RED: Failing tests for multiplication engine** - `bd09192` (test)
3. **Task 2 GREEN: Implement Fano plane, Cayley-Dickson, and cross-check** - `1d6f8c9` (feat)

_TDD task had RED (test) and GREEN (feat) commits._

## Files Created/Modified
- `pyproject.toml` - Package config with hatchling, torch>=2.7, dev deps (pytest, hypothesis, ruff, mypy)
- `src/octonion/__init__.py` - Public API: FANO_PLANE, FanoPlane, STRUCTURE_CONSTANTS, octonion_mul, cayley_dickson_mul
- `src/octonion/_types.py` - NormedDivisionAlgebra ABC with conjugate, norm, inverse, mul, components, dim
- `src/octonion/_fano.py` - FanoPlane frozen dataclass with 7 triples, lines, incidence matrix, subalgebras
- `src/octonion/_multiplication.py` - Structure constants [8,8,8] tensor and octonion_mul via einsum
- `src/octonion/_cayley_dickson.py` - quaternion_mul, quaternion_conj, cayley_dickson_mul with basis permutation
- `src/octonion/py.typed` - PEP 561 marker
- `tests/conftest.py` - Hypothesis strategies (octonions, unit_octonions, nonzero_octonions), tolerance constants
- `tests/test_multiplication.py` - 28 tests: identity, squaring, Fano triples, full table, sparsity, distributivity, non-associativity
- `tests/test_cayley_dickson.py` - 9 tests: Hamilton product, conjugation, 64-basis cross-check, random cross-check

## Decisions Made
- **Structure constants count is 64, not 50:** The plan and RESEARCH.md estimated 50 non-zero entries (8 left-identity + 42 triple entries). The correct count is 64 = 1 (e0*e0) + 7 (left-identity) + 7 (right-identity) + 7 (squaring) + 42 (triples). The estimate missed right-identity and squaring entries.
- **CD-Fano permutation mapping:** The Cayley-Dickson quaternion-pair split `x[:4], x[4:]` produces a valid but differently-labeled octonion algebra. A pure permutation P=[0,1,2,5,3,7,6,4] (no sign flips) maps between them. This was derived by brute-force search over all 7! permutations.
- **Distributivity test range:** Reduced from [-1e6, 1e6] to [-1e3, 1e3] to avoid float64 precision artifacts when products approach 1e12.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed structure constants sparsity count from 50 to 64**
- **Found during:** Task 2 (GREEN phase, test_sparsity)
- **Issue:** Plan claimed "exactly 50 non-zero entries out of 512" but actual count is 64. The RESEARCH.md formula "7*6 + 8 = 50" only counted left-identity entries (C[0,i,i]) and triple entries, missing right-identity (C[i,0,i]) and squaring (C[i,i,0]) entries.
- **Fix:** Updated test expectation to 64 with detailed breakdown in docstring.
- **Files modified:** tests/test_multiplication.py
- **Verification:** test_sparsity passes with count=64
- **Committed in:** 1d6f8c9

**2. [Rule 1 - Bug] Fixed distributivity test tolerance for large inputs**
- **Found during:** Task 2 (GREEN phase, test_left_distributivity)
- **Issue:** Default octonions strategy generates values up to 1e6. Products of such values reach 1e12, where float64 absolute errors can exceed 1e-6 due to ~15 significant digits.
- **Fix:** Constrained input range to [-1e3, 1e3] for distributivity tests, keeping the same absolute tolerance.
- **Files modified:** tests/test_multiplication.py
- **Verification:** Both distributivity tests pass on 200 random examples
- **Committed in:** 1d6f8c9

**3. [Rule 3 - Blocking] Derived Cayley-Dickson to Fano plane basis permutation**
- **Found during:** Task 2 (GREEN phase, test_all_64_basis_products_match)
- **Issue:** The naive CD quaternion-pair split produces octonion multiplication with different basis labeling than Baez 2002 Fano plane convention, as anticipated in Open Question 1 of RESEARCH.md.
- **Fix:** Brute-force searched all 5040 permutations of {1..7} to find P=[0,1,2,5,3,7,6,4] such that fano_table[i,j,k] = cd_table[P[i],P[j],P[k]]. Applied P as input/output permutation in cayley_dickson_mul().
- **Files modified:** src/octonion/_cayley_dickson.py
- **Verification:** All 64 basis products match; random cross-check passes within 1e-6
- **Committed in:** 1d6f8c9

---

**Total deviations:** 3 auto-fixed (2 bugs, 1 blocking)
**Impact on plan:** All auto-fixes necessary for correctness. No scope creep. The basis permutation finding is a research result anticipated by the plan.

## Issues Encountered
None beyond the deviations documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Core multiplication engine verified and ready for Octonion class wrapper (Plan 02)
- Hypothesis strategies and tolerance constants ready for property-based testing in all downstream plans
- Fano plane data structure ready for subalgebra analysis (Phase 9)
- NormedDivisionAlgebra ABC ready for R/C/H/O type hierarchy (Plan 02)
- Cayley-Dickson basis permutation documented for from_quaternion_pair/to_quaternion_pair conversions

---
*Phase: 01-octonionic-algebra*
*Completed: 2026-03-08*
