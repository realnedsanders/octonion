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
- [x] **Phase 3: Baseline Implementations** - Fair R/C/H comparison networks with matched parameter counts (completed 2026-03-13)
- [ ] **Phase 4: Numerical Stability** - Precision characterization across depths, float widths, and operation chains
- [ ] **Phase 5: Optimization Landscape** - Quantitative landscape characterization determining project viability
- [ ] **Phase 6: Reversibility Claim** - Algebraic inversion quality vs RevNet/INN baselines across depth and noise
- [ ] **Phase 7: Density & Geometric Claims** - Matched-parameter density advantage and geometric signal detection experiments
- [ ] **Phase 8: G2 Equivariance & Hyperbolic Hybrid** - Novel G2-equivariant layers and hyperboloid-octonionic model
- [ ] **Phase 9: Associator & Subalgebra Analysis** - Associator-aware architecture and Fano plane decomposition of learned representations
- [ ] **Phase 10: Predict-and-Fill Benchmarks** - Reversible conjecture at scale with missing-data identification and completion tasks
- [ ] **Phase 11: Applied Single-Stream Benchmarks** - Anomaly detection and time series prediction vs LSTM/Transformer baselines
- [ ] **Phase 12: Hyperboloid Projection Stability** - Empirical characterization of the Euclidean-Lorentzian projection distortion problem
- [ ] **Phase 13: Multi-Stream Data Fusion** - ORE proof-of-concept ingesting heterogeneous real-time data streams

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
**Plans**: 15 plans

Plans:
- [x] 03-01-PLAN.md — Config, algebra linear layers (R/C/H/O), initialization, parameter matching
- [x] 03-02-PLAN.md — Normalization layers, activation functions, convolutional layers
- [x] 03-03-PLAN.md — AlgebraNetwork skeleton (MLP/Conv/Recurrent), recurrent cells, skeleton identity tests
- [x] 03-04-PLAN.md — Training utility with full observability, statistical testing, plotting
- [x] 03-05-PLAN.md — Comparison runner with experiment management and API finalization
- [x] 03-06-PLAN.md — CIFAR benchmark reproduction infrastructure (training deferred to 03-09)
- [x] 03-07-PLAN.md — [GAP CLOSURE] ResNet-style residual blocks in AlgebraNetwork conv2d topology, pytest-timeout
- [x] 03-08-PLAN.md — [GAP CLOSURE] Topology-aware run_comparison with conv2d model dispatch
- [ ] 03-09-PLAN.md — [GAP CLOSURE] Execute CIFAR-10 reproduction training and validate against published results
- [ ] 03-10-PLAN.md — Profiling baseline and Tier 1 zero-risk optimizations (vectorized BN, training loop micro-opts)
- [ ] 03-11-PLAN.md — Tier 2 optimizations: fused OctonionDenseLinear einsum, eval-mode conv caching, buffer registration
- [ ] 03-12-PLAN.md — Tier 3 optimizations: AMP BN float32 protection, torch.compile config flag
- [ ] 03-13-PLAN.md — [UAT GAP CLOSURE] Commit working-tree test fix for _tril_to_symmetric 4-arg signature
- [ ] 03-14-PLAN.md — [UAT GAP CLOSURE] Add base_filters to NetworkConfig; fix profile_baseline.py param explosion
- [ ] 03-15-PLAN.md — [UAT GAP CLOSURE] Re-verify AMP-safe BN with correct BN.__name__ invocation

### Phase 4: Numerical Stability
**Goal**: Precision characteristics of octonionic operations are quantified so that architecture decisions (depth, float width, mitigations) are evidence-based
**Depends on**: Phase 1, Phase 2
**Requirements**: FOUND-03
**Success Criteria** (what must be TRUE):
  1. Forward pass precision degradation is measured at depths 10, 50, 100, and 500 layers with quantified error accumulation curves
  2. Condition numbers of octonionic multiplication, inversion, and composed operations are characterized as a function of input magnitude
  3. float32 vs float64 convergence comparison identifies the minimum precision required for each operation class
  4. At least one mitigation strategy (re-normalization, mixed precision, or compensation) is demonstrated to extend stable depth by at least 2x
**Plans**: 3 plans

Plans:
- [ ] 04-01-PLAN.md — StabilizingNorm module, config integration, and Wave 0 smoke test infrastructure
- [ ] 04-02-PLAN.md — Comprehensive stability analysis script (depth sweep, condition numbers, float32/64 comparison, mitigation)
- [ ] 04-03-PLAN.md — [UAT GAP CLOSURE] Fix float32 overflow handling and JSON serialization in analysis script

### Phase 5: Optimization Landscape
**Goal**: Determine whether octonionic networks can be trained reliably -- go/no-go gate for project viability
**Depends on**: Phase 2, Phase 3, Phase 4
**Requirements**: FOUND-04
**Success Criteria** (what must be TRUE):
  1. Gradient variance across 20+ random seeds is characterized and compared to R/C/H baselines on at least 2 synthetic tasks with known optima
  2. Hessian eigenspectrum at convergence shows ratio of negative eigenvalues (saddle points) for octonionic networks is within 3x of quaternionic baseline
  3. Training convergence profiles (loss vs step) across 3+ optimizers (SGD, Adam, a Riemannian optimizer) are documented for all 4 algebras
  4. Determine if octonionic networks converge to solutions within 2x loss of R8-dense-mixing baseline on 3+ tasks with 10+ seeds each
  5. Determine if landscape pathology is characterized quantitatively (publishable negative result) and project pivots
**Plans**: 8 plans

Plans:
- [x] 05-01-PLAN.md — Install deps (geoopt, pytorch-optimizer), extend AlgebraType, implement PHM8Linear + DenseMixingLinear
- [x] 05-02-PLAN.md — 5 synthetic task generators (algebra-native, cross product, sinusoidal, classification) with controls
- [x] 05-03-PLAN.md — Hessian eigenspectrum toolkit (full Hessian + stochastic Lanczos) and Bill & Cox curvature measurement
- [x] 05-04-PLAN.md — Trainer extension (LBFGS/Riemannian/Shampoo), AlgebraNetwork PHM8/R8D dispatch, gradient stats, gate logic
- [x] 05-05-PLAN.md — Experiment orchestration with incremental saves, Hessian checkpoints, smoke test
- [ ] 05-06-PLAN.md — Analysis script, full experiment run, gate verdict, human review checkpoint
- [x] 05-07-PLAN.md — [GAP CLOSURE] Fix intermediate checkpoint saving, create post-training analysis script (Hessian/curvature/gradient)
- [x] 05-08-PLAN.md — [GAP CLOSURE] Integration tests verifying post-training analysis produces expected data

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
**Goal**: Implement and validate the two most novel architectural contributions -- G2-equivariant layers and hyperboloid-octonionic model
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

### Phase 10: Predict-and-Fill Benchmarks
**Goal**: Validate the "geometry of absence" concept and test reversible conjecture on practical missing-data tasks
**Depends on**: Phase 6 (reversibility), Phase 8 (hyperbolic geometry for uncertainty manifold structure)
**Requirements**: CLAIM-01 (extended)
**Success Criteria** (what must be TRUE):
  1. Predict-and-fill task defined where the model must identify which dimensions of input are missing and generate plausible completions using inverse projection
  2. Octonionic inverse projection produces lower reconstruction error than optimization-based inversion baselines on structured missing data
  3. Uncertainty manifold geometry (dimension, curvature) correlates with the amount and type of missing information
  4. The model accurately distinguishes "what it knows" from "what it doesn't know" as measured by calibration of uncertainty estimates
**Plans**: TBD

Plans:
- [ ] 10-01: TBD
- [ ] 10-02: TBD

### Phase 11: Applied Single-Stream Benchmarks
**Goal**: Test octonionic representations on practical time series tasks beyond synthetic data, comparing against standard sequence model baselines
**Depends on**: Phase 5 (gate must pass), Phase 7 (density comparison infrastructure)
**Requirements**: BASE-04 (extended)
**Success Criteria** (what must be TRUE):
  1. Anomaly detection benchmark on noisy time series with planted anomalies, comparing octonionic network vs LSTM vs Transformer baselines with matched parameter counts
  2. Time series prediction benchmark (e.g., multi-step forecasting) with statistical significance testing across all algebra variants
  3. Geometric signal hypothesis (associator coherence distinguishes signal from noise) is tested quantitatively on real-valued time series data
  4. Results document whether the octonionic density advantage observed on synthetic tasks transfers to applied tasks
**Plans**: TBD

Plans:
- [ ] 11-01: TBD
- [ ] 11-02: TBD

### Phase 12: Hyperboloid Projection Stability
**Goal**: Empirically characterize the central open problem of the hyperboloid-octonionic synthesis: does re-projection after octonionic multiplication preserve useful algebraic properties?
**Depends on**: Phase 8 (hyperboloid-octonionic hybrid implementation)
**Requirements**: ADV-02 (extended)
**Success Criteria** (what must be TRUE):
  1. Projection distortion |project(a*b) - project(a) *_H project(b)| measured for random and structured octonionic inputs across varying hyperboloid radii
  2. Distortion characterized as a function of distance from the hyperboloid apex (abstraction level)
  3. Invertibility degradation after N re-projection cycles is quantified (does forward-inverse round-trip error accumulate?)
  4. At least one mitigation strategy (learned projection, tangent-space approximation, or frequency of re-projection) reduces distortion by a measurable factor
**Plans**: TBD

Plans:
- [ ] 12-01: TBD
- [ ] 12-02: TBD

### Phase 13: Multi-Stream Data Fusion
**Goal**: Build an ORE proof-of-concept that ingests heterogeneous real-time data streams and demonstrates cross-stream geometric signal detection
**Depends on**: Phases 6-12 (requires validated components)
**Requirements**: APP-01, APP-02 (promoted from v2)
**Success Criteria** (what must be TRUE):
  1. At least 3 heterogeneous data streams (e.g., financial market data, news text sentiment, temporal event sequences) encoded into octonionic representations via learned projections
  2. Cross-stream signal detection demonstrates that octonionic geometric coherence (subalgebra alignment, associator structure) identifies genuine cross-domain correlations
  3. Fano plane subalgebra decomposition reveals interpretable specialization across stream types
  4. System produces actionable predictions or alerts that outperform single-stream baselines on at least one cross-domain task
**Plans**: TBD

Plans:
- [ ] 13-01: TBD
- [ ] 13-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 (gate) -> 6 -> 7 -> 8 -> 9 -> 10 -> 11 -> 12 -> 13
Note: Phase 3 (Baselines) can execute in parallel with Phases 2 and 4.
Note: Phases 10 and 11 can execute in parallel after their dependencies are met.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Octonionic Algebra | 4/6 | Gap closure | - |
| 2. GHR Calculus | 1/4 | In progress | - |
| 3. Baseline Implementations | 12/15 | UAT gap closure | - |
| 4. Numerical Stability | 2/3 | UAT gap closure | - |
| 5. Optimization Landscape (GO/NO-GO) | 5/8 | Gap closure | - |
| 6. Reversibility Claim | 0/? | Not started | - |
| 7. Density & Geometric Claims | 0/? | Not started | - |
| 8. G2 Equivariance & Hyperbolic Hybrid | 0/? | Not started | - |
| 9. Associator & Subalgebra Analysis | 0/? | Not started | - |
| 10. Predict-and-Fill Benchmarks | 0/? | Not started | - |
| 11. Applied Single-Stream Benchmarks | 0/? | Not started | - |
| 12. Hyperboloid Projection Stability | 0/? | Not started | - |
| 13. Multi-Stream Data Fusion | 0/? | Not started | - |
