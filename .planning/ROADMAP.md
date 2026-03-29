# Roadmap: Octonionic Computation Substrate

## Overview

This roadmap validates the octonionic ML thesis bottom-up through two lines of inquiry. **Foundation phases (1-5)** build verified algebra, extend calculus to octonions, characterize numerical stability, build fair baselines, and hit the go/no-go gate on optimization landscape viability. **Trie phases (T1-T7)** develop the self-organizing octonionic trie, which achieved 95.2% on MNIST with zero gradient descent in the classifier and zero catastrophic forgetting. The trie direction is the primary research focus following the MNIST result.

Gradient-trained network experiments (originally Phases 6-13) are deprioritized. Several of their claims (reversibility, associator as signal, subalgebra specialization) are validated more directly by the trie.

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
- [ ] **Phase T1: Benchmark Generalization** - Fashion-MNIST, CIFAR-10, text classification with the octonionic trie
- [ ] **Phase T2: Adaptive Thresholds** - Data-driven associator thresholds (per-node, context-specific, or provably global)
- [ ] **Phase T3: Algebraic Encoder** - Unsupervised/algebraic encoding to eliminate CNN dependency
- [ ] **Phase T4: Scaling Analysis** - Accuracy, node count, query time vs data size and number of categories
- [ ] **Phase T5: Continual Learning Comparison** - Formal comparison against EWC, PackNet, Progressive Nets on sequential benchmarks
- [ ] **Phase T6: Cascaded O^n Routing** - Multi-resolution trie with cascaded octonionic views
- [ ] **Phase T7: Streaming Classification** - Online classification with rumination and consolidation on live data
- [ ] **Phase T8: Multi-Stream Fusion** - ORE proof-of-concept with heterogeneous data streams via trie

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

### Phase T1: Benchmark Generalization
**Goal**: Determine whether the trie's 95.2% MNIST result generalizes to other standard benchmarks
**Depends on**: Phase 5 (foundation validation)
**Requirements**: TRIE-01
**Success Criteria** (what must be TRUE):
  1. Fashion-MNIST accuracy measured with CNN encoder, compared against kNN and sklearn baselines on same 8D features
  2. CIFAR-10 accuracy measured with multiple encoder sizes (2-layer, 4-layer, ResNet-8) showing encoder capacity effect
  3. 20 Newsgroups text classification tested with fully gradient-free pipeline (TF-IDF + TruncatedSVD to 8D)
  4. Per-benchmark analysis of which categories the trie handles well vs poorly, with confusion matrices and failure mode characterization
**Plans**: 5 plans

Plans:
- [x] T1-01-PLAN.md — Install scikit-learn, shared benchmark utilities (sklearn baselines, metrics, plotting), unit tests
- [ ] T1-02-PLAN.md — Fashion-MNIST benchmark (CNN encoder + trie + all baselines + learning curves)
- [ ] T1-03-PLAN.md — CIFAR-10 benchmark (3 CNN encoder sizes + trie + all baselines + encoder comparison)
- [ ] T1-04-PLAN.md — 20 Newsgroups text benchmark (TF-IDF + TruncatedSVD, fully gradient-free, 20-class + 4-class subset)
- [ ] T1-05-PLAN.md — Cross-benchmark summary script + human verification checkpoint

### Phase T2: Adaptive Thresholds
**Goal**: Determine whether the associator threshold should be global, per-node, context-specific, or whether a global value can be theoretically justified
**Depends on**: Phase T1 (need multiple benchmarks to test generalization of threshold)
**Success Criteria** (what must be TRUE):
  1. Per-node adaptive threshold (e.g., based on running mean of associator norms at that node) tested and compared against global threshold
  2. Context-specific threshold (e.g., depth-dependent, or based on the node's category purity) tested
  3. If global threshold works: mathematical justification for why (e.g., associator norm distribution is invariant to data distribution for unit octonions)
  4. If adaptive is better: the adaptation rule is characterized and the improvement is statistically significant across benchmarks
  5. Sensitivity analysis: accuracy vs threshold across a continuous range for each benchmark
**Plans**: TBD

### Phase T3: Algebraic Encoder
**Goal**: Develop an encoding pipeline that does not require gradient-trained neural networks, completing the "zero gradients end-to-end" vision
**Depends on**: Phase T1 (need benchmarks to evaluate encoder quality)
**Success Criteria** (what must be TRUE):
  1. PCA-based hierarchical encoding tested (different PCA components at different trie depths)
  2. Random projection encoding tested (Johnson-Lindenstrauss style)
  3. At least one algebraic encoding method (e.g., octonionic Fourier features, structure-constant-based projection) tested
  4. Best algebraic encoder achieves within 10pp of CNN encoder accuracy on MNIST
  5. Full pipeline (algebraic encoder + trie) runs with zero gradient computation end-to-end
**Plans**: TBD

### Phase T4: Scaling Analysis
**Goal**: Characterize how the trie scales with data size, number of categories, and input dimensionality
**Depends on**: Phase T1 (need benchmarks at different scales)
**Success Criteria** (what must be TRUE):
  1. Accuracy vs training set size curve (1K to 60K) with power-law or logarithmic fit
  2. Node count and query time vs training set size characterized
  3. Accuracy vs number of categories (10, 50, 100, 1000) with fixed data per category
  4. Memory usage characterized: bytes per node, total memory vs dataset size
  5. Comparison of trie query time vs kNN query time at each scale
**Plans**: TBD

### Phase T5: Continual Learning Comparison
**Goal**: Formally compare the trie's zero-forgetting property against established continual learning methods
**Depends on**: Phase T1 (need standard benchmarks), Phase T4 (need scaling understanding)
**Success Criteria** (what must be TRUE):
  1. Split-MNIST and Permuted-MNIST benchmarks reproduced with EWC, PackNet, and Progressive Nets baselines
  2. Octonionic trie evaluated on the same protocols with identical train/test splits
  3. Forgetting measured using standard continual learning metrics (backward transfer, forward transfer)
  4. Statistical significance testing across 10+ random seeds
  5. Analysis of trie growth pattern during sequential tasks (does the trie structure reflect task boundaries?)
**Plans**: TBD

### Phase T6: Cascaded O^n Routing
**Goal**: Develop cascaded multi-octonion routing into a production-quality multi-resolution trie
**Depends on**: Phase T2 (need adaptive thresholds for multi-level routing)
**Success Criteria** (what must be TRUE):
  1. Cascaded routing achieves higher accuracy than single-octonion trie on at least 2 benchmarks
  2. Optimal number of octonions characterized per benchmark (diminishing returns analysis)
  3. Depth cycling strategy (which octonion at which depth) is analyzed and optimized
  4. Memory and compute overhead of cascaded routing quantified relative to single-octonion
**Plans**: TBD

### Phase T7: Streaming Classification
**Goal**: Test the trie as an online classifier on streaming data with concept drift
**Depends on**: Phase T5 (need continual learning validation), Phase T2 (need adaptive thresholds)
**Success Criteria** (what must be TRUE):
  1. Streaming classification benchmark with known concept drift points (e.g., rotating MNIST, gradual class distribution shift)
  2. Rumination mechanism implemented and tested: does consistency checking improve accuracy on drifting data?
  3. Consolidation mechanism tested: does pruning/merging improve memory efficiency without accuracy loss?
  4. Comparison against online learning baselines (online SGD, streaming random forests)
**Plans**: TBD

### Phase T8: Multi-Stream Fusion
**Goal**: ORE proof-of-concept: trie ingesting heterogeneous real-time data streams
**Depends on**: Phase T7 (streaming infrastructure), Phase T6 (multi-resolution routing)
**Success Criteria** (what must be TRUE):
  1. At least 3 heterogeneous data streams encoded into octonionic representations
  2. Cross-stream signal detection via subalgebra alignment and associator structure
  3. Fano plane subalgebra decomposition reveals specialization across stream types
  4. System produces predictions that outperform single-stream baselines on a cross-domain task
**Plans**: TBD

## Progress

**Execution Order:**

Foundation: 1 -> 2 -> 3 -> 4 -> 5 (gate, experiments running)
Trie: T1 -> T2 -> T3 -> T4 -> T5 -> T6 -> T7 -> T8
Note: T1-T4 can partially overlap. T5 requires T1 and T4. T7 requires T2 and T5.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Octonionic Algebra | 4/6 | Complete (gap closure cosmetic) | - |
| 2. GHR Calculus | 4/4 | Complete | - |
| 3. Baseline Implementations | 12/15 | Complete (gap closure cosmetic) | - |
| 4. Numerical Stability | 3/3 | Complete | - |
| 5. Optimization Landscape | 7/8 | Experiments running | - |
| T1. Benchmark Generalization | 1/5 | In Progress|  |
| T2. Adaptive Thresholds | 0/? | Not started | - |
| T3. Algebraic Encoder | 0/? | Not started | - |
| T4. Scaling Analysis | 0/? | Not started | - |
| T5. Continual Learning Comparison | 0/? | Not started | - |
| T6. Cascaded O^n Routing | 0/? | Not started | - |
| T7. Streaming Classification | 0/? | Not started | - |
| T8. Multi-Stream Fusion | 0/? | Not started | - |
