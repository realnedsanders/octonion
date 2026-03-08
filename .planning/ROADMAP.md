# Roadmap: Octonionic Computation Substrate

## Overview

This roadmap validates the octonionic ML thesis bottom-up: build verified algebra, extend calculus to octonions, characterize numerical stability, build fair baselines, then hit the go/no-go gate on optimization landscape viability. If the gate passes, validate the three core claims (reversibility, density advantage, geometric signal detection), then tackle the advanced differentiators (G2 equivariance, hyperbolic hybrid, associator architecture, subalgebra analysis). Every phase produces observable experimental outcomes with statistical rigor.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Octonionic Algebra** - Verified core algebra library with property-based tests against Moufang identities
- [ ] **Phase 2: GHR Calculus** - Octonionic backpropagation gradients validated by finite-difference checks
- [ ] **Phase 3: Baseline Implementations** - Fair R/C/H comparison networks with matched parameter counts
- [ ] **Phase 4: Numerical Stability** - Precision characterization across depths, float widths, and operation chains
- [ ] **Phase 5: Optimization Landscape (GO/NO-GO)** - Quantitative landscape characterization determining project viability
- [ ] **Phase 6: Reversibility Claim** - Algebraic inversion quality vs RevNet/INN baselines across depth and noise
- [ ] **Phase 7: Density & Geometric Claims** - Matched-parameter density advantage and geometric signal detection experiments
- [ ] **Phase 8: G2 Equivariance & Hyperbolic Hybrid** - Novel G2-equivariant layers and hyperboloid-octonionic model
- [ ] **Phase 9: Associator & Subalgebra Analysis** - Associator-aware architecture and Fano plane decomposition of learned representations

## Phase Details

### Phase 1: Octonionic Algebra
**Goal**: A verified octonionic algebra library exists that downstream code can trust unconditionally
**Depends on**: Nothing (first phase)
**Requirements**: FOUND-01
**Success Criteria** (what must be TRUE):
  1. Moufang identities pass on 10,000+ random octonion triples at float64 precision
  2. Norm preservation (|ab| = |a||b|) holds to within 1e-12 relative error on random inputs
  3. Cayley-Dickson construction produces results identical to Fano-plane multiplication table (Baez 2002)
  4. Inverse operation satisfies a * a_inv = 1 and a_inv * a = 1 to within numerical precision
  5. Associator [a,b,c] = (ab)c - a(bc) is non-zero for generic triples but zero when any two arguments are equal (alternativity)
**Plans**: 6 plans

Plans:
- [x] 01-00-PLAN.md — Development container setup (ROCm PyTorch, uv, GPU passthrough)
- [x] 01-01-PLAN.md — Project scaffolding, core multiplication engine (Fano plane + Cayley-Dickson), and test infrastructure
- [x] 01-02-PLAN.md — Octonion class, R/C/H tower types, random generators, and FOUND-01 property-based test suite
- [x] 01-03-PLAN.md — Extended operations, linear algebra, OctonionLinear, batch tests, edge cases, and benchmarks
- [ ] 01-04-PLAN.md — [GAP CLOSURE] Fix pyproject.toml dep groups, Octonion copy-constructor, and __str__ noise suppression
- [ ] 01-05-PLAN.md — [GAP CLOSURE] Fix dtype promotion in octonion_mul, OctonionLinear float32 default, and exp/log raw tensor coercion

### Phase 2: GHR Calculus
**Goal**: Octonionic gradients are computed correctly so that gradient-based training can be trusted
**Depends on**: Phase 1
**Requirements**: FOUND-02
**Success Criteria** (what must be TRUE):
  1. Custom autograd backward pass matches finite-difference gradient approximation to within 1e-5 relative error at float64 on single-layer octonionic linear transform
  2. Gradient check passes on 5-layer compositions with mixed parenthesization patterns
  3. Explicit test demonstrates that naive (associativity-assuming) chain rule produces different (wrong) gradients than parenthesization-aware implementation
  4. Backward pass runs on ROCm GPU and produces identical results to CPU computation
**Plans**: 4 plans

Plans:
- [x] 02-01-PLAN.md — GHR Wirtinger formalism, analytic Jacobians for all 7 primitives, numeric Jacobian utility
- [ ] 02-02-PLAN.md — Autograd Functions for all 7 primitives, custom octonion gradcheck, SC-1 single-layer verification
- [ ] 02-03-PLAN.md — CompositionBuilder, parenthesization-aware chain rule, exhaustive testing (SC-2), naive-vs-correct demo (SC-3)
- [ ] 02-04-PLAN.md — Analyticity conditions, LR scaling heuristic, GPU/CPU parity (SC-4), public API finalization

### Phase 3: Baseline Implementations
**Goal**: Fair comparison networks exist for real, complex, and quaternionic algebras so that every octonionic experiment has trustworthy baselines
**Depends on**: Phase 1 (algebra-agnostic network skeleton)
**Requirements**: BASE-01, BASE-02, BASE-03
**Success Criteria** (what must be TRUE):
  1. Real-valued baseline with 8x units matches octonionic network total parameter count to within 1%
  2. Complex-valued baseline with 4x units matches octonionic network total parameter count and reproduces a published benchmark result within reported variance
  3. Quaternionic baseline with 2x units matches octonionic network total parameter count and reproduces a published benchmark result within reported variance
  4. All four networks (R, C, H, O) share identical architecture skeleton differing only in algebra module
**Plans**: 6 plans

Plans:
- [x] 03-01-PLAN.md — Config, algebra linear layers (R/C/H/O), initialization, parameter matching
- [x] 03-02-PLAN.md — Normalization layers, activation functions, convolutional layers
- [ ] 03-03-PLAN.md — AlgebraNetwork skeleton (MLP/Conv/Recurrent), recurrent cells, skeleton identity tests
- [ ] 03-04-PLAN.md — Training utility with full observability, statistical testing, plotting
- [ ] 03-05-PLAN.md — Comparison runner with experiment management and API finalization
- [ ] 03-06-PLAN.md — CIFAR benchmark reproduction and verification (C/H published results)

### Phase 4: Numerical Stability
**Goal**: Precision characteristics of octonionic operations are quantified so that architecture decisions (depth, float width, mitigations) are evidence-based
**Depends on**: Phase 1, Phase 2
**Requirements**: FOUND-03
**Success Criteria** (what must be TRUE):
  1. Forward pass precision degradation is measured at depths 10, 50, 100, and 500 layers with quantified error accumulation curves
  2. Condition numbers of octonionic multiplication, inversion, and composed operations are characterized as a function of input magnitude
  3. float32 vs float64 convergence comparison identifies the minimum precision required for each operation class
  4. At least one mitigation strategy (re-normalization, mixed precision, or compensation) is demonstrated to extend stable depth by at least 2x
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD

### Phase 5: Optimization Landscape (GO/NO-GO)
**Goal**: Determine whether octonionic networks can be trained reliably -- this gates the entire remainder of the project
**Depends on**: Phase 2, Phase 3, Phase 4
**Requirements**: FOUND-04
**Success Criteria** (what must be TRUE):
  1. Gradient variance across 20+ random seeds is characterized and compared to R/C/H baselines on at least 2 synthetic tasks with known optima
  2. Hessian eigenspectrum at convergence shows ratio of negative eigenvalues (saddle points) for octonionic networks is within 3x of quaternionic baseline
  3. Training convergence profiles (loss vs step) across 3+ optimizers (SGD, Adam, a Riemannian optimizer) are documented for all 4 algebras
  4. GO decision: octonionic networks converge to solutions within 2x loss of R8-dense-mixing baseline on 3+ tasks with 10+ seeds each
  5. If NO-GO: landscape pathology is characterized quantitatively (publishable negative result) and project pivots
**Plans**: TBD

Plans:
- [ ] 05-01: TBD
- [ ] 05-02: TBD

### Phase 6: Reversibility Claim
**Goal**: Determine whether octonionic algebraic inversion provides meaningful backward inference that outperforms trained invertible networks
**Depends on**: Phase 1, Phase 2, Phase 5 (must pass go/no-go)
**Requirements**: CLAIM-01
**Success Criteria** (what must be TRUE):
  1. Synthetic task with known forward model and ground-truth inverse is defined and validated (forward model recoverable to < 1e-6 error)
  2. Octonionic algebraic inversion reconstruction fidelity is measured as a function of network depth (1, 5, 10, 20 layers)
  3. Reconstruction fidelity is measured as a function of input noise level (0%, 1%, 5%, 10%, 25%)
  4. Comparison against RevNet (trained invertible), INN (coupling-layer), and standard-net + optimization-based inversion is statistically significant (p < 0.05)
**Plans**: TBD

Plans:
- [ ] 06-01: TBD
- [ ] 06-02: TBD

### Phase 7: Density & Geometric Claims
**Goal**: Determine whether octonionic representations achieve better parameter efficiency and whether they have genuine geometric structure affinity
**Depends on**: Phase 3, Phase 5 (must pass go/no-go)
**Requirements**: BASE-04, CLAIM-02
**Success Criteria** (what must be TRUE):
  1. Matched-parameter density comparison across R/C/H/O runs on at least 3 tasks (synthetic pattern recognition, time series, classification) with accuracy, convergence speed, and sample efficiency measured
  2. Statistical significance testing (p < 0.05) applied to all pairwise comparisons with correction for multiple testing
  3. Geometric signal detection experiment on synthetic data with known planted geometric structure shows octonionic detection accuracy vs R8/C4/H2 baselines
  4. R8-dense-mixing baseline included in every comparison to guard against the "Why Not Just R8?" trap
**Plans**: TBD

Plans:
- [ ] 07-01: TBD
- [ ] 07-02: TBD

### Phase 8: G2 Equivariance & Hyperbolic Hybrid
**Goal**: Implement and validate the two most novel architectural contributions -- G2-equivariant layers and hyperboloid-octonionic hybrid model
**Depends on**: Phase 5 (must pass go/no-go)
**Requirements**: ADV-01, ADV-02
**Success Criteria** (what must be TRUE):
  1. G2-equivariant layer satisfies numerical equivariance test: applying random G2 transform before vs after layer produces deviation < 1e-5
  2. G2 layer improves performance on a task with known G2 symmetry compared to unconstrained octonionic layer
  3. Hyperboloid-octonionic hybrid model (Option B: Lorentzian inner product, H7) trains stably on hierarchical data (trees or DAGs)
  4. Algebraic integrity metric |project(a*b) - project(a) *_H project(b)| is tracked throughout training and remains below a pre-defined threshold
  5. Hybrid model matches or exceeds Poincare embeddings baseline on hierarchical structure preservation
**Plans**: TBD

Plans:
- [ ] 08-01: TBD
- [ ] 08-02: TBD
- [ ] 08-03: TBD

### Phase 9: Associator & Subalgebra Analysis
**Goal**: Determine whether non-associativity carries useful information and whether trained octonionic networks develop interpretable subalgebra specialization
**Depends on**: Phase 5, Phase 7 (needs trained models to analyze)
**Requirements**: ADV-03, ADV-04
**Success Criteria** (what must be TRUE):
  1. Associator-aware architecture (using [a,b,c] as attention, regularization, or gating signal) trains successfully and the associator signal correlates with task-relevant structure
  2. Comparison shows associator-aware architecture outperforms associator-ignoring variant on at least one task
  3. Fano plane subalgebra decomposition of trained weights shows non-uniform subalgebra activity distribution (Gini coefficient > 0.3)
  4. Subalgebra ablation test: removing the most-active subalgebra degrades performance more than removing the least-active (demonstrating specialization)
**Plans**: TBD

Plans:
- [ ] 09-01: TBD
- [ ] 09-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 (gate) -> 6 -> 7 -> 8 -> 9
Note: Phase 3 (Baselines) can execute in parallel with Phases 2 and 4.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Octonionic Algebra | 4/6 | Gap closure | - |
| 2. GHR Calculus | 1/4 | In progress | - |
| 3. Baseline Implementations | 5/6 | In progress | - |
| 4. Numerical Stability | 0/? | Not started | - |
| 5. Optimization Landscape (GO/NO-GO) | 0/? | Not started | - |
| 6. Reversibility Claim | 0/? | Not started | - |
| 7. Density & Geometric Claims | 0/? | Not started | - |
| 8. G2 Equivariance & Hyperbolic Hybrid | 0/? | Not started | - |
| 9. Associator & Subalgebra Analysis | 0/? | Not started | - |
