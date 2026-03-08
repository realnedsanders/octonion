# Phase 1: Octonionic Algebra - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

A verified octonionic algebra library that downstream code can trust unconditionally. Implements multiplication, conjugation, norm, inverse, and associator with property-based tests against Moufang identities. Also includes the full R/C/H/O Cayley-Dickson tower, Fano plane geometric structure, and foundational operations (exp, log, inner product, cross product, commutator, multiplication matrices) needed by downstream phases. Does NOT include: trilinear form, Freudenthal cross product, full tower conversions, neural network layers beyond basic OctonionLinear, or Phase 2+ calculus operations.

</domain>

<decisions>
## Implementation Decisions

### Tensor Framework
- PyTorch-native tensors from day one — octonions as torch.Tensor with shape [..., 8]
- Thin class wrapper: `class Octonion` with operator overloading (__mul__, __add__, __sub__, __neg__, __eq__, __repr__)
- float64 as default precision (success criteria demand 1e-12 tolerance)
- Immutable — all operations return new Octonion instances, no in-place mutation
- Caller controls device placement (library is device-agnostic, standard PyTorch convention)
- No `/` operator — require explicit `a * b.inverse()` to avoid left/right division ambiguity in non-associative algebra
- No integer power operator (`**`) — explicit multiplication required because parenthesization matters
- Full scalar interop: scalar * octonion, octonion * scalar, octonion + scalar (adds to real part)

### Package Structure
- Package name: `octonion`
- src layout: `src/octonion/` with pyproject.toml
- Dependency management: `uv`
- Import style: `from octonion import Octonion`

### Multiplication Convention
- Component ordering: e0 (real/scalar) first, [e0, e1, e2, ..., e7]
- Implement multiplication both via Fano plane table AND Cayley-Dickson recursion; test that they produce identical results (success criterion 3)
- Verify Baez 2002 convention against another reference; if they disagree, Claude resolves based on downstream cleanliness
- Cayley-Dickson doubling: hardcoded tables for production speed + recursive construction as cross-check
- Alternativity verified in test suite only — no runtime assertions

### Type Hierarchy
- Full R/C/H/O Cayley-Dickson tower: Real, Complex, Quaternion, Octonion types
- Abstract base class `NormedDivisionAlgebra` defining conjugate(), norm(), inverse(), mul()
- Algebra-generic operations: conjugation, norm, inverse work on all types (R, C, H, O)
- Separate `UnitOctonion` type (guarantees norm=1)
- Separate `PureOctonion` type (real part = 0)

### Fano Plane & Subalgebras
- Fano plane exposed as inspectable public object with full geometric structure (7 lines/triples, cyclic orderings, incidence matrix, automorphism group generators)
- 7 quaternionic subalgebras accessible via named constructors
- Subalgebra projection (project octonion onto specific subalgebra) deferred to Phase 9

### Batch Computation
- Batch-first design: all operations accept [..., 8] shaped tensors
- Full PyTorch broadcasting rules for batch dimensions
- Truncated display for batched octonions (like PyTorch tensor printing)

### Component Access
- `o.real` (scalar part), `o.imag` (7-vector), `o[i]` for component i, `o.components` for raw tensor
- `__str__` shows symbolic form (`1.0 + 2.0*e1 + ...`), `__repr__` shows tensor form

### Operations Included
- Core: multiplication, conjugation, norm, inverse, associator (success criteria)
- Extended algebra: exp, log, commutator [a,b] = ab - ba
- Products: inner product <a,b> = Re(a* * b), 7D cross product
- Linear algebra: left_mul_matrix(a), right_mul_matrix(a) returning 8x8 real matrices
- Transform: OctonionLinear (a*x*b) with both a,b learnable, no bias term
- Conversion: from_quaternion_pair(q1, q2) and to_quaternion_pair() bidirectional

### Random Generation
- Seed-controlled utilities: random_octonion(), random_unit_octonion(), random_pure_octonion()
- Multiple distributions: Gaussian, unit, pure imaginary, near-zero, large magnitude
- Controllable batch size, dtype, device, generator

### Testing Strategy
- Hypothesis for property-based testing with custom octonion strategies
- Known-answer tests (hand-computed basis element products, specific identity checks, edge cases) alongside property tests
- Moufang identity checker as reusable test utility: check_moufang(a, b, c, tol)
- Precision tracking: tests report max/mean/std of relative errors (not just pass/fail)
- Basic performance benchmarks: operation throughput (ops/sec) on CPU and GPU

### Error Handling
- Verbose error messages with math context (e.g., "Cannot invert zero octonion: norm is 0.0. Octonion inverse requires non-zero norm.")
- Serialization via PyTorch's torch.save/torch.load (no custom serialization)

### Claude's Discretion
- Batch testing strategy (separate batch tests vs batched Hypothesis strategies)
- Resolving sign convention conflicts between Baez 2002 and other references
- Internal module organization within src/octonion/
- Exact Hypothesis strategy implementations
- Benchmark script design

</decisions>

<specifics>
## Specific Ideas

- Baez 2002 is the canonical algebra reference (thesis foundation)
- Parcollet et al. 2019 for quaternionic neural net conventions (informs OctonionLinear design)
- No division operator — user explicitly chose this to respect non-associative left/right division ambiguity
- No power operator — user explicitly chose this because parenthesization matters
- Subalgebra projection deferred to Phase 9 despite subalgebra data being available in Phase 1
- Full Cayley-Dickson tower conversions (R<->C<->H<->O embeddings) deferred to Phase 8+

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- None — greenfield project with no existing code

### Established Patterns
- None — this phase establishes the project's foundational patterns

### Integration Points
- This library is the foundation for ALL downstream phases
- Phase 2 (GHR Calculus) will build autodiff on top of these operations and multiplication matrices
- Phase 3 (Baselines) will use the R/C/H types from the Cayley-Dickson hierarchy
- Phase 8 (G2 Equivariance) will use Fano plane structure and subalgebra data
- Phase 9 (Associator Analysis) will use associator, commutator, and subalgebra constructors

</code_context>

<deferred>
## Deferred Ideas

- Trilinear form t(a,b,c) = Re((ab)c*) — Phase 8+ (exceptional geometry)
- Freudenthal cross product — Phase 8+ (G2 equivariance)
- Full Cayley-Dickson tower conversions (embed/project between all levels) — Phase 8+
- Subalgebra projection (project octonion onto specific quaternionic subalgebra) — Phase 9
- Integer power operator — not implemented due to non-associativity parenthesization concerns

</deferred>

---

*Phase: 01-octonionic-algebra*
*Context gathered: 2026-03-07*
