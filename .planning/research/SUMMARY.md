# Project Research Summary

**Project:** Octonionic Computation Substrate -- Research PoC
**Domain:** Hypercomplex ML research validation (octonionic algebra, non-associative optimization)
**Researched:** 2026-03-07
**Confidence:** MEDIUM (mature surrounding tooling; octonionic ML itself is nascent and under-documented)

---

## Executive Summary

This project is an empirical research PoC aimed at validating 9 specific claims from a thesis on octonionic neural networks. The domain is a genuine frontier: no production-grade octonionic ML library exists, the GHR calculus extension from quaternions to octonions is an open research problem, and no published work has rigorously characterized the octonionic optimization landscape. The surrounding infrastructure -- PyTorch on ROCm, geoopt for hyperbolic geometry, wandb for experiment tracking, hypothesis for property-based testing -- is mature and well-supported. The core octonionic algebra, autograd integration, and G2-equivariant layers must be built entirely from scratch, representing roughly 4,000-5,000 lines of research-grade code that has no existing reference implementation to validate against.

The recommended approach is strictly bottom-up: algebra correctness gates everything. Before any training experiment runs, the Fano-plane multiplication table must be verified via property-based tests (Moufang identities, norm preservation, alternativity) against a known-correct oracle, and the custom autograd backward pass must be validated by finite-difference gradient checks on multi-layer compositions. The architecture follows a 4-layer dependency hierarchy (Algebra -> Calculus -> Network -> Experiment) with no upward dependencies. Every experiment must run all algebra-type baselines (real, complex, quaternion, octonion) with matched parameter counts as a first-class requirement, not an afterthought.

The dominant project risk is the optimization landscape. If non-associativity creates pathological loss surfaces -- excess local minima, saddle-point proliferation, gradient pathologies -- then the training-dependent claims cannot be validated and the thesis loses most of its empirical support. This is an explicit go/no-go gate: after building the algebra and gradient layer, landscape characterization on simple toy tasks must be run before investing in G2 layers, hyperbolic hybrid models, or multi-stream fusion. A negative result here is still a publishable contribution (a rigorous characterization of why octonionic optimization fails), but it changes the project's trajectory entirely.

---

## Key Findings

### Recommended Stack

The hypercomplex ML ecosystem is immature -- no usable octonionic library exists. Four major components must be built from scratch: the octonionic tensor library (~2,000 lines), the GHR calculus gradient engine (~500-1,000 lines), the G2-equivariant layers (~1,500 lines), and the hyperboloid-octonionic projection (~500 lines). The surrounding ecosystem is solid. PyTorch on ROCm works via HIP translation with zero code changes for standard operations. Custom `torch.autograd.Function` subclasses work on ROCm without modification. The RX 7900 XTX (gfx1100) is officially supported by ROCm 7.2.0. Docker is the recommended environment to avoid dependency hell with pytorch-triton-rocm versioning.

**Core technologies:**
- PyTorch 2.10 + ROCm 7.2: GPU compute -- only framework with reliable AMD support; HIP translation layer makes `torch.cuda` API work unchanged
- Python 3.12: Language runtime -- best typing support, required by PyTorch
- geoopt (master): Riemannian optimization and hyperbolic manifolds -- pure PyTorch, full ROCm compat, required for D-2 hybrid model
- wandb >=0.19.11: Experiment tracking -- explicit AMD GPU metrics via rocm-smi
- hypothesis + hypothesis-torch: Property-based testing -- essential for algebraic invariant verification; PyTorch itself uses this pattern
- clifford 1.5.1: Geometric algebra reference (CPU-only, for validation against known-correct implementations)
- numpy-quaternion: CPU quaternion oracle for baseline validation
- geomstats 2.8.0: Differential geometry for G2 manifold research

**What NOT to use:** TensorFlow/JAX (poor ROCm support), HyperNets PHM layers (learns algebra rules, opposite of needed), hTorch/Pytorch-Quaternion-Neural-Networks as pip dependencies (abandoned), float32 for algebra validation (non-associative errors compound), custom HIP kernels before profiling.

### Expected Features

The feature set is structured around 9 thesis claims and sequenced by a strict dependency graph. The critical path is TS-1 -> TS-4 -> TS-5 -> TS-3, meaning: algebra verification, gradient implementation, numerical stability, and optimization landscape characterization must all complete before training-dependent experiments.

**Must have (table stakes -- gates all other work):**
- TS-1: Core algebra verification suite -- property-based tests for multiplication, conjugation, norm, associator, Moufang identities. Fatal if skipped.
- TS-4: GHR gradient implementation and numerical verification -- custom autograd backward validated by finite differences. Fatal if skipped.
- TS-5: Numerical stability analysis -- forward pass depth experiments (10/50/100/500 layers), gradient magnitude distribution, float32 vs float64 comparison. Fatal if skipped.
- TS-3: Optimization landscape characterization -- go/no-go gate for all training-dependent claims. If landscapes are pathological, pivot to studying why (publishable negative result).
- TS-7: Baseline implementations -- matched-parameter real, complex, quaternion networks. Without fair baselines no comparison is meaningful.

**Should have (core thesis validation):**
- TS-2: Matched-parameter density comparison (Claim 1) -- must use matched total real-valued scalars, not hypercomplex units; 3+ different tasks
- TS-6: Controlled reversibility experiment (Claim 2) -- the thesis's strongest claimed differentiator; algebraic inversion vs RevNet/INN baselines
- D-3: Geometric signal detection on synthetic data (Claim 6) -- clean test of whether octonionic representations have genuine affinity for geometric structure

**Differentiators (novel contributions, high novelty if successful):**
- D-1: G2-equivariant layers (Claim 4) -- no existing implementation; genuinely first
- D-4: Multi-stream data fusion (Claim 7)
- D-2: Hyperboloid-octonionic hybrid (Claim 5) -- has an open problem at its core (projection stability)
- D-5: Fano plane subalgebra decomposition analysis
- D-6: Associator-aware architecture design

**Defer indefinitely (anti-features):**
- Large-scale benchmarks (ImageNet etc.) -- not the thesis's contribution
- Per-dimension semantic interpretability -- explicitly out of scope in thesis
- Option C exotic hyperbolic plane (OH^2) -- multi-year research program in itself
- Go port before Python validation -- premature
- Custom GPU kernels before profiling -- premature optimization

### Architecture Approach

The architecture follows a strict 4-layer dependency hierarchy: Layer 1 (Algebra) -> Layer 2 (Autograd/Calculus) -> Layer 3 (Network) -> Layer 4 (Experiment). No upward dependencies exist. This is a deliberate departure from most hypercomplex implementations, which fold the calculus into the network layer -- octonions require a dedicated calculus layer because non-associativity means the gradient depends on parenthesization order, breaking standard PyTorch chain rule assumptions. The key data-flow distinction is that training backward (GHR calculus, gradient-based) and inference backward (algebraic inversion, reasoning-based) use entirely different code paths and must not be conflated.

**Major components:**
1. `src/algebra/` (Layer 1): Octonion type, Fano-plane multiplication, conjugation/norm/inverse, associator, 7 quaternionic subalgebra extraction, G2 automorphism group actions. All tensors in `[..., 8]` layout. No ML dependencies.
2. `src/calculus/` (Layer 2): GHR-based gradient derivations, custom `torch.autograd.Function` subclasses, full Jacobian computation, gradient diagnostics. Depends only on Layer 1.
3. `src/networks/` (Layer 3): OctonionLinear, OctonionConv1d/2d, OctonionBatchNorm (8D whitening, not component-wise), activations, G2-equivariant layers, pooling, ORE model assembly, baseline models. Algebra-agnostic skeleton (same structure for R/C/H/O, algebra swapped via parameter).
4. `experiments/` (Layer 4): Hydra configs, comparison trainer (runs all baselines as default), synthetic data generators, evaluators, analysis scripts. ComparisonTrainer is the primary interface -- individual model training is a special case.

**Key architectural patterns:**
- Algebra-agnostic network skeleton: all layers accept an `algebra` parameter; real/complex/quaternion/octonion differ only in the multiplication function
- Diagnostic hooks throughout: every octonionic layer emits associativity defect, gradient norms per component, subalgebra activation patterns (build in from day one, not bolted on later)
- Reproducibility as infrastructure: every experiment fully determined by config + git hash; `torch.use_deterministic_algorithms(True)` required
- Staged validation: algebra property tests -> gradient checks -> single-layer convergence -> small network convergence -> full ORE assembly

### Critical Pitfalls

1. **Fano plane sign/order errors (CP-1):** 480 valid multiplication conventions across 30 Fano plane orientations. Mixing sources creates an algebra that looks correct but violates alternativity. Prevention: pick Baez (2002) as the single canonical source; validate with 10,000+ random triples against Moufang identities and norm preservation. This must be the very first CI test.

2. **Non-associative autograd ordering corrupts gradients silently (CP-2):** Standard PyTorch autograd assumes associative chain rule. `d/dW[(Wx)b]` differs from `d/dW[W(xb)]`. Standard autograd silently computes wrong gradients -- training runs, converges to worse solutions, and you blame the architecture instead of the gradient. Prevention: custom `torch.autograd.Function` for every octonionic operation; validate every backward with `gradcheck` at float64 on multi-layer compositions.

3. **The "Why Not Just R^8?" experimental design trap (CP-4):** Octonion multiplication creates 8x8 dense cross-component mixing. An unfair R^8-diagonal baseline (8 parameters, no cross-component interaction) will always lose. Publishing that result as evidence for octonionic advantage is methodologically invalid. Prevention: include an R^8-dense-mixing baseline (64 parameters, fully connected) in every experiment. If octonions don't outperform R^8-dense, the algebraic structure is not helping beyond the mixing.

4. **Hyperboloid projection destroys algebraic structure (CP-3):** Projecting octonionic vectors into hyperbolic space via `exp_0` does not preserve octonionic multiplication. The hybrid model degenerates into a standard hyperbolic network with extra parameters. Prevention: define which algebraic properties must survive projection; measure algebraic integrity metrics throughout training; consider tangent-space operation (algebra in tangent space, projection only for distance computation).

5. **Non-associative optimization landscape pathologies (MP-6):** Loss landscapes for octonionic networks may have qualitatively more local minima and saddle points than real/quaternion networks because parenthesization ordering affects what the optimizer can reach. This is the project's highest-identified risk. Prevention: characterize the landscape on 1-2 layer toy tasks immediately after algebra/gradient validation, before building complex architectures; use as explicit go/no-go gate.

6. **GHR calculus does not extend trivially to octonions (MP-2):** The quaternion GHR product rule uses associativity of the rotation group; octonion rotations are not associative. Do not assume GHR generalizes cleanly. Prevention: derive gradients from first principles via component-wise 8x8 Jacobian if full GHR extension proves intractable; validate numerically regardless of theoretical framework.

---

## Implications for Roadmap

Based on combined research, 5 phases are recommended, with an explicit go/no-go gate after Phase 2.

### Phase 1: Algebra Foundation
**Rationale:** Everything gates on this. Multiplication errors silently corrupt all downstream work. Gradient errors produce plausible-looking but invalid training. Initialization errors bias all comparative experiments. This phase has no dependencies and cannot be parallelized with later phases.
**Delivers:** Verified octonionic algebra library (multiplication, conjugation, norm, inverse, associator, subalgebra extraction) with passing property-based test suite. Verified GHR backward passes. Correct 8-component weight initialization. Fano plane constants and G2 algebra structure as pure data modules.
**Addresses:** TS-1 (Core Algebra Verification), TS-4 (GHR Gradients), TS-5 partial (float64 vs float32 precision characterization), CP-6 (weight initialization)
**Avoids:** CP-1 (multiplication errors), CP-2 (autograd ordering), MP-2 (GHR non-triviality), MP-5 (FP vs algebraic non-associativity)
**Validation gate:** Moufang identities pass on 10,000+ random triples; `gradcheck` passes on 5-layer compositions at float64; activation norms constant with depth at correct initialization.
**Research flag:** NEEDS RESEARCH -- octonionic GHR calculus extension is an open research problem; may need mathematical derivation from first principles with fallback to real-component Jacobian approach.

### Phase 2: Viability Gate (Optimization Landscape)
**Rationale:** This is the project's explicit go/no-go gate. If the optimization landscape is pathological -- meaning octonionic networks cannot reliably find good solutions -- then Phases 3-5 (all training-dependent claims) are invalid. Running this phase before building complex architectures prevents sunk cost. It also forces the baseline implementations (TS-7) to be built correctly before comparative experiments.
**Delivers:** Quantitative characterization of octonionic loss landscapes (2D landscape visualizations, Hessian eigenspectrum at convergence, gradient norm distributions, convergence profiles across optimizers) on 1-3 layer toy tasks with known optima. Fair baseline implementations (R, C, H, O) validated on published benchmarks.
**Addresses:** TS-3 (Optimization Landscape Characterization), TS-7 (Baseline Implementation and Validation)
**Avoids:** CP-4 (R^8 comparison trap), MP-6 (landscape pathologies), MP-1 (octonionic batch normalization)
**Decision rule:** If octonion networks converge to solutions within 2x of the R^8-dense-mixing baseline across 3+ toy tasks with 10+ seeds, proceed. If they are systematically worse, pivot to studying why (publishable negative result on octonionic optimization landscape).
**Research flag:** STANDARD PATTERNS -- loss landscape visualization (Li et al. 2018 style), Hessian eigenspectrum analysis, and convergence profiling are well-documented. No specialized research needed beyond the octonionic implementation itself.

### Phase 3: Core Claims Validation
**Rationale:** With algebra verified and optimization viability confirmed, the primary thesis claims can be tested. Reversibility (TS-6) is the first claim the project context identifies for testing and is structurally simpler than parameter density comparisons (no training needed for algebraic inversion). Density comparison (TS-2) requires careful experimental design (matched parameters). Geometric signal detection (D-3) provides clean evidence on synthetic data with known structure.
**Delivers:** Validated results for Claims 1, 2, and 6. Reversibility: algebraic inversion quality vs RevNet/INN baselines as a function of depth and noise. Density: matched-parameter comparisons across 3+ tasks. Geometric detection: octonionic advantage (or not) on data with planted octonionic geometric structure.
**Addresses:** TS-6 (Reversibility), TS-2 (Density Comparison), D-3 (Geometric Signal Detection)
**Avoids:** CP-4 (experimental design: always include R^8-dense-mixing baseline), mP-3 (subalgebra selection bias monitoring)
**Research flag:** NEEDS RESEARCH for reversibility experiment design -- no published protocol exists for comparing algebraic inversion quality against INN baselines; need to define metrics and synthetic task design carefully.

### Phase 4: Advanced Differentiators (conditional on Phase 3 results)
**Rationale:** These phases are gated on Phase 3 producing positive results. G2-equivariant layers (D-1) are the highest-novelty contribution (no existing implementation for exceptional Lie groups). Hyperbolic-octonionic hybrid (D-2) addresses an open problem identified in the thesis. Multi-stream fusion (D-4) is the most applied contribution. Fano plane analysis (D-5) and associator-aware architecture (D-6) are additive analysis layers.
**Delivers:** G2-equivariant layer implementation (genuinely novel), hyperboloid-octonionic hybrid with algebraic integrity metrics, multi-stream fusion architecture, subalgebra decomposition analysis tooling, and associator-aware gating mechanisms.
**Addresses:** D-1, D-2, D-4, D-5, D-6 (Claims 4, 5, 7, and cross-cutting)
**Avoids:** CP-3 (hyperboloid projection destroying algebra -- use tangent-space approach), CP-5 (NaN cascades -- float64, Euclidean parametrization, gradient clipping), MP-3 (G2 exp map instability -- use `torch.linalg.matrix_exp` + re-projection), mP-4 (confusion between octonion-valued and octonion-structured architectures)
**Research flag:** NEEDS RESEARCH -- G2 representation theory implementation has no ML library reference; hyperbolic projection stability is an open problem (thesis section 9.7); both require mathematical derivation work, not just engineering.

### Phase 5: End-to-End ORE Integration
**Rationale:** With all components validated in isolation, the full Octonionic Reasoning Engine from thesis section 4 can be assembled. This is integration testing, not new feature development. Each component should arrive at this phase already validated.
**Delivers:** Full ORE pipeline (encoder -> octonionic processing -> decoder with G2-equivariant blocks and reversible reasoning), end-to-end training and inference on the project's target domains (synthetic + financial time series + NLP fusion), final comparative results across all thesis claims.
**Addresses:** Full ORE from thesis section 4
**Avoids:** Anti-Pattern 5 (monolithic ORE assembled before component validation), MP-4 (ROCm-specific issues -- CPU-GPU equivalence must be verified throughout)
**Research flag:** STANDARD PATTERNS -- integration of validated components is engineering, not research; however, emergent instabilities from combining octonionic + hyperbolic + G2 components may surface; keep diagnostic hooks active throughout.

### Phase Ordering Rationale

- **Algebra before everything:** CP-1 and CP-2 are "silent corruption" pitfalls -- they produce plausible-looking results that are wrong. Catching them requires dedicated testing infrastructure built first.
- **Landscape characterization as a gate:** MP-6 (landscape pathologies) is the highest-risk item per the project's own assessment. Running it early (Phase 2) before investing in complex architectures is the responsible research methodology. Phases 3-5 depend on a positive result here.
- **Reversibility before density comparison:** TS-6 is identified as the first claim to test in the project context. It is also structurally cleaner (algebraic inversion doesn't require landscape viability to be proven) and tests a fundamental property of the algebra.
- **G2 and hyperbolic as late Phase 4:** Both are technically demanding open problems that should not compete for attention during algebra validation and landscape characterization.
- **Synthetic data throughout:** Synthetic tasks with known ground truth must precede real data at every phase. Real data adds confounds; synthetic data provides ground truth for validating whether the algebra is working.

### Research Flags

Phases needing deeper research during planning:
- **Phase 1 (GHR Calculus):** The octonionic extension of GHR calculus is genuinely open. Need to decide: attempt full GHR extension, or use real-component Jacobian fallback? This decision should be made before Phase 1 begins.
- **Phase 3 (Reversibility Experiment Design):** No published protocol for comparing algebraic inversion quality against INN/RevNet baselines. Need to define synthetic task, ground-truth metric, and noise-level sweep before executing.
- **Phase 4 (G2 Layers):** G2 representation theory implementation has no ML library reference. The 14-dimensional Lie algebra basis and kernel constraint solving must be derived from mathematical sources (Baez 2002, G2 Wikipedia). Consider whether a mathematician collaborator is warranted.
- **Phase 4 (Hyperbolic Projection):** The projection stability problem (thesis section 9.7) is open. Need to establish what "acceptable" algebraic integrity degradation means quantitatively before this phase begins.

Phases with standard patterns (research-phase less critical):
- **Phase 2 (Landscape Characterization):** Li et al. 2018-style loss landscape visualization, Hessian eigenspectrum analysis, and convergence profiling are well-documented methods. Experimental design is standard; octonionic implementation is the only novel part.
- **Phase 5 (Integration):** Integration of validated components is engineering. Diagnostic hooks from earlier phases handle anomaly detection.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Core stack (PyTorch, ROCm, geoopt) verified against official docs and compatibility matrices. The "build from scratch" conclusion for the algebra is unambiguous -- all existing alternatives were checked and rejected with documented reasons. |
| Features | MEDIUM | Table stakes (TS-1 through TS-7) are well-grounded in hypercomplex ML literature. Differentiators (D-1 through D-6) are speculative by nature -- their value depends on experimental outcomes. Feature dependencies graph is sound. |
| Architecture | MEDIUM-HIGH | 4-layer architecture follows proven patterns from geoopt, SpeechBrain, HyperNets. Tensor-native design, algebra-agnostic skeleton, and staged validation are well-validated patterns. G2 component boundaries are theoretical (no reference implementation exists). |
| Pitfalls | HIGH for critical pitfalls | CP-1 through CP-5 are documented from primary literature with specific reproduction conditions. MP-6 (landscape pathology) is the key unknown -- it is theoretically grounded but empirically uncharted. Minor pitfalls are well-reasoned extrapolations. |

**Overall confidence:** MEDIUM

The surrounding tooling is HIGH confidence. The core domain (octonionic ML) is MEDIUM at best because the field lacks standardized experimental methodology, no reference implementation exists to validate against, and the two most important questions (optimization landscape viability, GHR calculus extension to octonions) are genuinely open. This is appropriate for a research PoC -- the uncertainty is the research contribution.

### Gaps to Address

- **GHR calculus octonionic extension:** No literature treats this directly. During Phase 1, must decide whether to attempt full GHR derivation (research contribution, high risk) or use real-component Jacobian fallback (lower risk, less elegant, still valid). Recommend attempting GHR first with a clear time-box, then falling back.
- **Matched-parameter comparison methodology:** The "matched total real-valued scalars" requirement is clear in principle but implementation details (how to handle the encoder/decoder boundary, what counts as a parameter) need to be pinned down before Phase 3 experiments run.
- **Go/no-go criteria for optimization landscape:** Phase 2 is a go/no-go gate, but the specific quantitative criterion needs to be defined before the experiments run (not after seeing results). Recommend establishing this criterion at the start of Phase 2 planning.
- **Octonionic batch normalization:** No published octonionic BN implementation exists. The quaternionic case (8x8 Cholesky whitening) is an extrapolation. Correctness and computational feasibility need to be validated experimentally during Phase 1.
- **G2 equivariant layer feasibility:** The 14-dimensional G2 group and its 7x7 fundamental representation have no ML implementation precedent. FEATURES.md flags this as "may need collaboration with a mathematician." This risk needs explicit acknowledgment in the roadmap and a fallback plan if G2 layers prove intractable within the project's scope.

---

## Sources

### Primary (HIGH confidence)
- [ROCm Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) -- ROCm 7.2.0, gfx1100 (RX 7900 XTX) support confirmed
- [PyTorch HIP Semantics](https://docs.pytorch.org/docs/stable/notes/hip.html) -- ROCm/HIP compatibility, custom autograd compatibility
- [PyTorch Get Started](https://pytorch.org/get-started/locally/) -- version matrix, ROCm wheel URLs
- [geoopt GitHub](https://github.com/geoopt/geoopt) -- manifold support, PyTorch compatibility
- [The Numerical Stability of Hyperbolic Representation Learning (Mishne et al., ICML 2023)](https://arxiv.org/abs/2211.00181) -- float64 requirement, NaN failure modes, Euclidean parametrization
- [The Octonions (Baez, 2002)](https://math.ucr.edu/home/baez/octonions/octonions.pdf) -- canonical multiplication table, G2 structure
- [Deep Quaternion Networks (Gaudet & Maida, 2018)](https://arxiv.org/pdf/1712.04604) -- quaternion initialization (Chi distribution); direct precedent for octonionic extension
- [Enabling quaternion derivatives: the GHR calculus (Xu et al., 2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4555860/) -- GHR calculus foundation
- [PyTorch Autograd Mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) -- custom Function subclass interface

### Secondary (MEDIUM confidence)
- [Deep Octonion Networks (Wu & Xu, 2019)](https://arxiv.org/abs/1903.08478) -- OctonionConv/BN/ReLU architecture patterns; no public code
- [Hypercomplex Neural Networks: Exploring Quaternion, Octonion, and Beyond (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12513225/) -- current state of field; open problems enumerated
- [PHNNs: Parameterized Hypercomplex Convolutions (Grassucci et al., 2022)](https://arxiv.org/abs/2110.04176) -- Kronecker product hypercomplex framework; valid as baseline
- [Lie Group Decompositions for Equivariant NNs (ICLR 2024)](https://arxiv.org/abs/2310.11366) -- handles reductive groups only; confirms G2 has no existing ML implementation
- [GHR Calculus in QNNs](https://www.nnw.cz/doi/2017/NNW.2017.27.014.pdf) -- backpropagation with GHR for quaternions
- [SpeechBrain Quaternion Networks](https://speechbrain.readthedocs.io/en/v1.0.2/tutorials/nn/complex-and-quaternion-neural-networks.html) -- mature quaternion implementation architecture patterns
- [geoopt Poincare Ball](https://deepwiki.com/geoopt/geoopt/4.4-poincare-ball) -- boundary stability, gradient clipping

### Tertiary (LOW confidence -- extrapolation or sparse sourcing)
- G2 equivariant layer design specifics -- no ML reference; extrapolated from G2 mathematics and general equivariant network literature
- Octonionic batch normalization (8D whitening) -- extrapolated from quaternionic case; no published octonionic BN implementation found
- Subalgebra selection bias in training -- theoretical concern; no published evidence of this occurring in practice
- Optimization landscape pathology for octonionic networks -- theoretically motivated; no empirical characterization exists in literature (this is what the project will produce)

---
*Research completed: 2026-03-07*
*Ready for roadmap: yes*
