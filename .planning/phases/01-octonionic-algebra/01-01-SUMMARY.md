---
phase: 01-octonionic-algebra
plan: 01
subsystem: algebra
tags: [pytorch, octonion, fano-plane, cayley-dickson, hypothesis, property-testing, float64]

# Dependency graph
requires:
  - phase: 01-00
    provides: "ROCm PyTorch dev container with GPU passthrough"
provides:
  - "Installable octonion package with Fano plane structure constants multiplication"
  - "Cayley-Dickson cross-check with empirically-derived basis permutation mapping"
  - "FanoPlane data structure with triples, lines, incidence matrix, subalgebras"
  - "NormedDivisionAlgebra abstract base class"
  - "Hypothesis strategies for octonion tensors (octonions, unit_octonions, nonzero_octonions)"
  - "22 passing tests covering all basis products, distributivity, non-associativity, CD cross-check"
affects: [01-02, 01-03, 02-01, 02-02, 03-01, 08-01, 09-01]

# Tech tracking
tech-stack:
  added: [torch>=2.7, pytest, hypothesis>=6.0, hypothesis-torch, numpy>=1.26, ruff, mypy, hatchling]
  patterns: [structure-constants-einsum, fano-triples-mod7, cayley-dickson-basis-mapping, src-layout-uv]

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
  - "Structure constants tensor has 64 non-zero entries (not 50 as initially estimated in research -- the identity element contributes 15 entries, not 8)"
  - "CD-to-Fano basis mapping requires signed permutation P={0:0, 1:1, 2:2, 3:4, 4:3, 5:7, 6:5, 7:6} with S={0:+1, 1:-1, 2:-1, 3:+1, 4:+1, 5:+1, 6:+1, 7:+1} (Open Question 1 resolved)"
  - "Distributivity tests use moderate-magnitude inputs (1e3) with rtol=1e-9 to avoid false failures from float64 rounding at extreme scales"

patterns-established:
  - "Structure constants multiplication via torch.einsum('...i, ijk, ...j -> ...k', a, C, b)"
  - "Fano plane triples as single source of truth for all multiplication rules"
  - "TDD flow: write failing tests first, implement to pass, commit separately"
  - "Hypothesis strategies produce raw torch.Tensor[8] at float64 (not class wrappers)"

requirements-completed: [FOUND-01]

# Metrics
duration: 9min
completed: 2026-03-08
---

# Phase 1 Plan 01: Core Multiplication Engine Summary

**Fano plane structure constants multiplication with Cayley-Dickson cross-check, resolving Open Question 1 via empirical signed permutation mapping between CD and mod-7 bases**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-08T04:58:04Z
- **Completed:** 2026-03-08T05:07:24Z
- **Tasks:** 2
- **Files modified:** 10

## Accomplishments
- Installable octonion package with all dependencies via uv (torch, pytest, hypothesis, ruff, mypy)
- Fano plane multiplication engine using [8,8,8] structure constants tensor with einsum -- all 64 basis element products verified against Baez 2002 convention
- Cayley-Dickson recursive multiplication as independent cross-check, with empirically derived basis permutation mapping that makes both implementations agree on all 64 basis products and 200+ random octonion pairs
- Test infrastructure with Hypothesis strategies, tolerance constants, and CI/dev profiles ready for downstream plans
- Open Question 1 from RESEARCH.md resolved: the standard quaternion-pair split does NOT directly correspond to mod-7 Fano plane indices; a signed permutation is required

## Task Commits

Each task was committed atomically:

1. **Task 1: Project scaffolding and package setup** - `6d6889d` (feat)
2. **Task 2 RED: Failing tests for multiplication engine** - `fb8feab` (test)
3. **Task 2 GREEN: Fano plane + Cayley-Dickson implementation** - `bfa9751` (feat)

_TDD task had separate RED and GREEN commits._

## Files Created/Modified
- `pyproject.toml` - Project config with hatchling build, torch>=2.7, dev deps
- `src/octonion/__init__.py` - Public API: octonion_mul, STRUCTURE_CONSTANTS, FANO_PLANE, FanoPlane, cayley_dickson_mul
- `src/octonion/_types.py` - NormedDivisionAlgebra ABC with conjugate, norm, inverse, mul, components, dim
- `src/octonion/_fano.py` - FanoPlane frozen dataclass: 7 oriented triples, lines, incidence matrix, subalgebras, automorphism generators
- `src/octonion/_multiplication.py` - Structure constants [8,8,8] tensor built from Fano triples, octonion_mul via einsum
- `src/octonion/_cayley_dickson.py` - Hamilton product, CD formula, basis mapping (Fano<->CD), cayley_dickson_mul
- `src/octonion/py.typed` - PEP 561 marker
- `tests/conftest.py` - Hypothesis strategies (octonions, unit_octonions, nonzero_octonions), tolerances, profiles
- `tests/test_multiplication.py` - 16 tests: identity, squaring, Fano triples, full table, sparsity, distributivity, non-associativity
- `tests/test_cayley_dickson.py` - 6 tests: 64 basis cross-check, random cross-check, Hamilton product correctness

## Decisions Made
- **Structure constants sparsity is 64, not 50:** The plan and research doc claimed 50 non-zero entries based on the formula "7*6 + 8 = 50". The correct count is 64: identity contributes 15 entries (C[0,i,i] for i=0..7 plus C[i,0,i] for i=1..7), self-product contributes 7 entries, and Fano triples contribute 42 entries (7*6).
- **CD basis permutation discovered empirically:** Exhaustive search over all 7! permutations with 2^7 sign patterns found 1344 valid isomorphisms. Selected the simplest: P={0:0, 1:1, 2:2, 3:4, 4:3, 5:7, 6:5, 7:6}, S={0:+1, 1:-1, 2:-1, 3:+1, 4:+1, 5:+1, 6:+1, 7:+1}.
- **Distributivity tolerance:** Property tests use rtol=1e-9, atol=1e-9 with input range [-1e3, 1e3] rather than default [-1e6, 1e6] to avoid false failures from float64 rounding on extreme-magnitude products.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed hatchling build backend path**
- **Found during:** Task 1
- **Issue:** pyproject.toml had `build-backend = "hatchling.backends"` instead of `"hatchling.build"`
- **Fix:** Corrected to `"hatchling.build"`
- **Files modified:** pyproject.toml
- **Verification:** uv sync succeeds
- **Committed in:** 6d6889d (Task 1 commit)

**2. [Rule 3 - Blocking] Removed missing README.md reference**
- **Found during:** Task 1
- **Issue:** pyproject.toml referenced readme = "README.md" which does not exist
- **Fix:** Removed the readme field
- **Files modified:** pyproject.toml
- **Verification:** uv sync succeeds
- **Committed in:** 6d6889d (Task 1 commit)

**3. [Rule 1 - Bug] Fixed test import of conftest**
- **Found during:** Task 2 (RED phase)
- **Issue:** `from conftest import ...` fails because conftest is not a regular module
- **Fix:** Added sys.path.insert to make conftest importable from test files
- **Files modified:** tests/test_multiplication.py, tests/test_cayley_dickson.py
- **Verification:** Tests collect and run
- **Committed in:** fb8feab (Task 2 RED commit)

**4. [Rule 1 - Bug] Fixed distributivity test tolerance for large-magnitude inputs**
- **Found during:** Task 2 (GREEN phase)
- **Issue:** atol=1e-8 is too strict when components up to 1e6 produce products in 1e11 range
- **Fix:** Used moderate input range (1e3) with rtol=1e-9, atol=1e-9
- **Files modified:** tests/test_multiplication.py
- **Verification:** Distributivity tests pass on 200 random triples
- **Committed in:** bfa9751 (Task 2 GREEN commit)

**5. [Rule 1 - Bug] Resolved CD-Fano basis mismatch (Open Question 1)**
- **Found during:** Task 2 (GREEN phase)
- **Issue:** Naive Cayley-Dickson quaternion-pair split produces different multiplication table than Fano plane mod-7 convention -- 42 of 64 basis products disagreed
- **Fix:** Empirically determined signed permutation mapping via exhaustive search, applied inside cayley_dickson_mul
- **Files modified:** src/octonion/_cayley_dickson.py
- **Verification:** All 64 basis products and 200+ random pairs match between Fano and CD implementations
- **Committed in:** bfa9751 (Task 2 GREEN commit)

**6. [Rule 1 - Bug] Corrected structure constants sparsity count from 50 to 64**
- **Found during:** Task 2 (RED phase, while writing tests)
- **Issue:** Plan and RESEARCH.md claimed 50 non-zero entries; careful analysis shows 64
- **Fix:** Test asserts 64, with detailed derivation in comments
- **Files modified:** tests/test_multiplication.py
- **Verification:** Structure constants test passes
- **Committed in:** fb8feab (Task 2 RED commit)

---

**Total deviations:** 6 auto-fixed (5 bugs, 1 blocking)
**Impact on plan:** All fixes necessary for correctness. The CD basis mapping (deviation 5) resolves Open Question 1 from RESEARCH.md -- this was anticipated as a research question, not an error.

## Issues Encountered
- libstdc++.so.6 not in LD_LIBRARY_PATH on host system (available in nix store). Tests must be run with LD_LIBRARY_PATH set. This is expected since the project is designed for the ROCm container.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Multiplication engine ready: octonion_mul, STRUCTURE_CONSTANTS, FANO_PLANE all verified
- Cayley-Dickson cross-check confirmed -- both implementations produce identical results
- Hypothesis strategies and tolerance constants ready for Plan 02 (Octonion class, property tests)
- NormedDivisionAlgebra ABC ready for Plan 02 (R/C/H/O types)
- Test infrastructure (conftest, profiles) ready for all downstream plans

## Self-Check: PASSED

All 11 files exist. All 3 commits found. All line counts exceed plan minimums. 22/22 tests pass.

---
*Phase: 01-octonionic-algebra*
*Completed: 2026-03-08*
