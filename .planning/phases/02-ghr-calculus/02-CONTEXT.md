# Phase 2: GHR Calculus - Context

**Gathered:** 2026-03-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Octonionic gradients are computed correctly so that gradient-based training can be trusted. Implements the full GHR (Generalized Hamilton-Real) calculus formalism for octonions: Wirtinger-like octonionic derivatives, custom autograd backward passes for all octonionic operations, parenthesization-aware chain rule, and verification against finite-difference approximation. Does NOT include: optimization experiments (Phase 5), numerical stability analysis across depths (Phase 4), or baseline comparisons (Phase 3).

</domain>

<decisions>
## Implementation Decisions

### GHR Formalism Depth
- Full GHR formalism — not just correct autograd, but the complete octonionic extension of HR calculus
- Native octonionic derivation — derive GHR derivatives directly in octonionic algebra, not by reducing to real components (Parcollet et al. extension pattern rejected in favor of native approach)
- User-facing API exposed in `octonion.calculus` submodule: `from octonion.calculus import ghr_derivative, conjugate_derivative, jacobian`
- All operations covered: mul, exp, log, conjugate, inverse, inner product, cross product — not just multiplication
- Both Wirtinger derivatives tracked: full (df/do, df/do*) pair, not just the conjugate gradient
- Higher-order derivative support: autograd.Function implementations support create_graph=True for Hessian computation (Phase 5 needs eigenspectrum analysis)
- Explicit Jacobian matrices: each operation provides analytic 8x8 real Jacobians AND numeric (finite-difference) Jacobians — triple-check: analytic vs numeric vs autograd
- Composite operation gradients built via chain rule from primitive operation gradients (not fused composite derivations)
- Formal derivations included in module docstrings showing the Wirtinger derivative derivation for each operation
- Custom octonion-aware gradcheck utility (not standard torch.gradcheck) — tests both Wirtinger derivatives, reports per-component errors, validates parenthesization correctness
- Octonionic analyticity tests: define CR-like conditions, test which operations satisfy them
- Learning rate scaling heuristic based on GHR gradient magnitude characteristics
- If GHR derivation reveals the correct number of derivative components differs from 2 (the complex Wirtinger pair), follow the math — accommodate whatever the algebra requires

### Parenthesization Handling
- Arbitrary parenthesization supported via computation graph — whatever order the user writes gets tracked and differentiated correctly
- Parenthesization inspector utility: text-based tree output (ASCII/Unicode) for debugging gradient flow in complex compositions
- No canonical parenthesization forced — users can write any association pattern

### Fallback Strategy
- No fallback — solve the GHR extension or prove it impossible. This is the research contribution
- No time limit on research exploration
- All-or-nothing for operation coverage — don't ship partial GHR
- If GHR proves mathematically impossible: re-scope to what's possible, but produce a formal barrier document explaining what was attempted, where it breaks, why, and implications for the thesis
- The 4 success criteria remain unchanged regardless of implementation approach — they test gradient correctness, not GHR specifically

### Composition Patterns
- Exhaustive parenthesization testing: all 14 Catalan(4) patterns at depth 5
- Both OctonionLinear layers AND raw octonion_mul chains tested
- Mixed operations in compositions: mul + exp + log + conjugate + inverse interleaved
- Public CompositionBuilder API using tree data structures (nested Python objects, not string parsing)
- Enumeration utility: `all_parenthesizations(n)` generates all C_{n-1} binary tree structures
- Configurable depth in test suite (5 is the success criterion default, but tests can run at any depth)
- Quantitative report: per-pattern gradient error (max/mean/std vs finite-difference) + maximum gradient difference BETWEEN different parenthesizations of same operands
- Results saved as structured file (JSON/CSV) AND printed to stdout; tree structures included in results file
- Claude's discretion: mixed-operation composition ordering strategy (different orderings vs fixed sequence)

### Naive vs Correct Demonstration
- Standalone script: `scripts/demo_naive_vs_correct.py` (not part of pytest suite, explicit invocation only)
- Full quantification: relative error magnitude, direction cosine similarity, per-component divergence analysis
- 1000 random inputs with confidence intervals for statistical rigor
- Depth scaling analysis: report naive-vs-correct error as function of depth (2, 3, 5, 7, 10) to show compounding non-associativity effects
- Raw data output only (JSON/CSV) — no LaTeX formatting, thesis presentation handled separately

### GPU Parity
- Manual GPU verification only — not part of CI pipeline
- GPU tolerance: Claude decides realistic tolerance for ROCm PyTorch float64

### Claude's Discretion
- Definition of "naive" chain rule for the demonstration (whatever most clearly shows non-associativity problem)
- Whether to include a training comparison (naive vs correct gradients on toy task) in the demo script
- GPU/CPU parity tolerance value
- Mixed-operation ordering strategy in composition tests
- Internal module organization within octonion.calculus

</decisions>

<specifics>
## Specific Ideas

- The thesis claims GHR calculus extends naturally from quaternions to octonions — this phase validates or refutes that claim
- Parcollet et al. 2019 is the quaternionic HR calculus reference, but we're NOT following their extension pattern — going native octonionic instead
- The "open research problem" blocker from STATE.md is acknowledged: GHR octonionic extension may be novel mathematical territory
- Composition API should support Phase 9's associator-aware architecture experiments (different association patterns = different computational meaning)
- The quantitative parenthesization report (showing gradient differences between parenthesizations) is a potential thesis contribution in itself

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `octonion_mul` (src/octonion/_multiplication.py): Structure constants tensor + einsum — already differentiable by PyTorch autograd, but GHR requires custom backward passes
- `left_mul_matrix` / `right_mul_matrix` (src/octonion/_linear_algebra.py): 8x8 real matrices for multiplication — directly usable for analytic Jacobian derivation
- `STRUCTURE_CONSTANTS` tensor (src/octonion/_multiplication.py): The [8,8,8] tensor C[i,j,k] — foundation for GHR derivative expressions
- `OctonionLinear` (src/octonion/_linear.py): (a*x)*b layer — first target for gradient verification
- `octonion_exp` / `octonion_log` (src/octonion/_operations.py): Component-wise implementations to be wrapped with GHR autograd
- `Octonion` class with conjugate(), norm(), inverse() — all need GHR backward passes

### Established Patterns
- Immutable Octonion class — all operations return new instances
- float64 precision for verification (1e-12 tolerance in Phase 1)
- Batch-first design: [..., 8] tensor shapes throughout
- Module-level constant tensors (STRUCTURE_CONSTANTS built at import time)
- Verbose error messages with math context

### Integration Points
- `octonion.calculus` will be a new submodule alongside existing `_multiplication.py`, `_operations.py`, etc.
- OctonionLinear.forward() currently calls octonion_mul directly — will need to use GHR-aware autograd functions
- Phase 4 (Numerical Stability) will use the explicit Jacobian matrices for condition number analysis
- Phase 5 (Optimization Landscape) will use higher-order derivatives for Hessian eigenspectrum analysis
- Phase 9 (Associator Analysis) will use the CompositionBuilder for different association patterns

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-ghr-calculus*
*Context gathered: 2026-03-08*
