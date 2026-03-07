# Feature Landscape

**Domain:** Hypercomplex ML Research Validation (Octonionic Computation Substrate)
**Researched:** 2026-03-07
**Confidence:** MEDIUM -- octonionic ML is a nascent field; quaternionic/complex baselines well-documented, octonionic experimental methodology is sparse and under-standardized.

---

## Table Stakes

Features and experiments users (i.e., reviewers, the research community) expect. Missing any of these makes the thesis unvalidated.

### TS-1: Core Algebra Verification Suite

| Aspect | Detail |
|---------|--------|
| **What** | Property-based tests confirming octonionic multiplication, conjugation, norm, inverse, and associator computations are correct |
| **Why Expected** | Every hypercomplex network paper (DON, DQN, DCN) assumes correct algebra. Bugs here silently corrupt all downstream results |
| **Complexity** | Low |
| **Thesis Claim** | Foundation for all claims |
| **Notes** | Must verify: norm preservation (||ab|| = ||a|| ||b||), non-associativity ((ab)c != a(bc) in general), Moufang identities, Fano plane multiplication table. Use known algebraic identities as oracle. Include Cayley-Dickson construction cross-check against explicit multiplication table |

### TS-2: Matched-Parameter Density Comparison (Real vs Complex vs Quaternion vs Octonion)

| Aspect | Detail |
|---------|--------|
| **What** | Train equivalent architectures across all four algebras on the same tasks with matched total parameter counts, measuring accuracy, convergence speed, and sample efficiency |
| **Why Expected** | This is the central empirical claim. The 2020 Deep Octonion Networks paper (Zhu et al.) did this for CIFAR-10/100 but only image classification. The thesis claims generality -- it must be shown on the thesis's target domains |
| **Complexity** | Medium |
| **Thesis Claim** | Claim 1 (Representational Density) |
| **Notes** | CRITICAL: "matched parameters" means matching the total number of trainable real-valued scalars, not the number of hypercomplex units. An octonion layer with N units has 8N real parameters per weight connection. Fair comparison requires real networks with 8x the units OR octonion networks with 1/8 the units. This is the most common methodological error in hypercomplex papers. Must use at least 3 different tasks (synthetic pattern recognition, time series, and a classification task) to demonstrate generality, not just CIFAR |

### TS-3: Optimization Landscape Characterization

| Aspect | Detail |
|---------|--------|
| **What** | Systematically study whether non-associativity creates pathological loss surfaces -- measure gradient variance, loss curvature (Hessian eigenspectrum), frequency of saddle points vs local minima, and training stability across random seeds |
| **Why Expected** | This is the project's self-identified highest risk. If octonion loss surfaces are pathological, nothing else matters. No existing paper has rigorously studied this for octonionic networks |
| **Complexity** | High |
| **Thesis Claim** | Claim 3 (Optimization Landscape Viability) |
| **Notes** | Methods: (a) Loss landscape visualization via random 2D projections (Li et al. 2018 style), (b) Hessian eigenspectrum analysis at convergence, (c) Gradient norm distribution over training, (d) Comparison of Adam vs SGD vs SGD+momentum convergence profiles, (e) Sensitivity to learning rate. Compare all four algebras. Synthetic tasks with known optima are essential here -- real data has too many confounds |

### TS-4: GHR Calculus Gradient Implementation and Verification

| Aspect | Detail |
|---------|--------|
| **What** | Implement octonionic backpropagation using generalized HR (GHR) calculus, verify gradients against numerical finite-difference approximation |
| **Why Expected** | GHR calculus is established for quaternions (Xu et al. 2015) but extending to octonions requires handling non-associativity in the chain rule. Without verified gradients, all training results are suspect |
| **Complexity** | High |
| **Thesis Claim** | Claim 8 (GHR Calculus Gradients) |
| **Notes** | The GHR calculus provides left- and right-hand quaternion derivatives using rotations in general orthogonal systems. For octonions, non-associativity means the product rule requires explicit parenthesization tracking. Must verify: (a) gradient of real-valued loss w.r.t. octonion parameters matches finite differences to within numerical precision, (b) gradient computation is order-aware (different parenthesizations give different gradients), (c) chain rule correctness through multiple layers. Fallback: if full GHR extension proves intractable, use the decomposition approach (split into 8 real channels, compute real gradients, reassemble) as DON does -- but document the tradeoff |

### TS-5: Numerical Stability Analysis

| Aspect | Detail |
|---------|--------|
| **What** | Characterize numerical precision degradation across forward pass depth, measure condition numbers of octonionic operations, identify and mitigate stability failure modes |
| **Why Expected** | Octonion multiplication involves 64 real multiplications and 56 additions per product -- error accumulation is non-trivial. The PMC 2025 survey explicitly flags this as an open problem |
| **Complexity** | Medium |
| **Thesis Claim** | Claim 9 (Numerical Stability) |
| **Notes** | Specific tests: (a) Forward pass of identity-initialized network at depths 10, 50, 100, 500 -- measure deviation from identity transform, (b) Gradient magnitude distribution at different depths, (c) Norm preservation error accumulation through layers, (d) Effect of float32 vs float64 on training convergence. Include octonion batch normalization (from DON paper) and measure whether it suffices or if additional stabilization is needed |

### TS-6: Controlled Reversibility Experiment

| Aspect | Detail |
|---------|--------|
| **What** | Demonstrate that algebraic invertibility of octonion transformations enables meaningful backward inference (given output, recover input or latent state), compared against standard invertible neural network approaches (RevNets, INNs) |
| **Why Expected** | The thesis identifies this as its strongest differentiating claim. The PROJECT.md marks it as the first claim to test. If it fails, the thesis "loses its strongest differentiator" |
| **Complexity** | High |
| **Thesis Claim** | Claim 2 (Reversibility / Backward Reasoning) |
| **Notes** | Design: (a) Synthetic task with known forward model and ground-truth inverse, (b) Train octonionic network on forward direction, (c) Use algebraic inverse to perform backward inference WITHOUT retraining, (d) Compare reconstruction quality against: RevNet (trained invertible), INN (coupling-layer invertible), standard net + optimization-based inversion. Key metric: reconstruction fidelity as a function of network depth and noise level. The octonion advantage should be that invertibility is algebraic (exact in exact arithmetic) rather than architectural (approximate by construction). Must also measure: does invertibility degrade gracefully with non-associative error accumulation? |

### TS-7: Baseline Implementation and Validation

| Aspect | Detail |
|---------|--------|
| **What** | Implement or adapt real, complex, and quaternion network baselines that are structurally identical to the octonionic architecture, differing only in the underlying algebra |
| **Why Expected** | Without fair baselines, no comparison is meaningful. Published implementations exist for quaternion (Parcollet et al. 2019) and complex (Trabelsi et al. 2018) networks but may not match the exact architecture used |
| **Complexity** | Medium |
| **Thesis Claim** | All claims (baseline for comparison) |
| **Notes** | Use the PHC (parameterized hypercomplex convolution) framework from Grassucci et al. (2022) where possible -- it generalizes across dimensions via Kronecker products. For each baseline: verify on a known benchmark (e.g., CIFAR-10) that it reproduces published results before using in novel experiments. This prevents "weak baseline" criticism |

---

## Differentiators

Features that would make novel research contributions. Not strictly required to validate the thesis, but would make the work publishable and impactful.

### D-1: G2-Equivariant Layer Design and Training

| Aspect | Detail |
|---------|--------|
| **What** | Implement neural network layers whose transformations are equivariant under the 14-dimensional G2 automorphism group of the octonions |
| **Value Proposition** | No existing work implements G2-equivariant neural layers. Existing equivariant network frameworks (e2cnn, EMLP, Lie Group Decompositions) handle classical and reductive Lie groups but not exceptional groups. This would be a genuine first |
| **Complexity** | Very High |
| **Thesis Claim** | Claim 4 (G2-Equivariant Layers) |
| **Notes** | G2 is the smallest exceptional Lie group, 14-dimensional, compact and connected. It acts on Im(O) (7-dimensional imaginary octonions) and preserves the octonion multiplication structure. Approach: (a) Parameterize G2 using its embedding in SO(7) with the 7 constraints that make it G2-specific, (b) Use the representation theory of G2 to construct weight-sharing patterns, (c) Verify equivariance numerically (apply random G2 transform before vs after layer, measure deviation). This is genuinely novel -- the 2023 paper on discovering G2 via ML (He et al.) discovered G2 symmetries in data but did not build G2-equivariant layers. RISK: G2 representation theory is technically demanding; may need collaboration with a mathematician |

### D-2: Hyperboloid-Octonionic Hybrid Model

| Aspect | Detail |
|---------|--------|
| **What** | Combine hyperbolic geometry (Poincare/hyperboloid model) with octonionic representations: embed data in hyperbolic space, process with octonionic layers, project back |
| **Value Proposition** | Hyperbolic embeddings excel at hierarchical data (Nickel & Kiela 2017 demonstrated 5-10x better embedding quality). Octonions provide dense multi-relational encoding. The combination has not been explored. The thesis specifically recommends Option B (hyperboloid model) |
| **Complexity** | Very High |
| **Thesis Claim** | Claim 5 (Hyperbolic-Octonionic Hybrid) |
| **Notes** | Must address the open problem from thesis Section 9.7: stability of hyperbolic projection. Hyperbolic spaces have exponentially growing distances, and octonionic operations could amplify numerical issues. Concrete plan: (a) Implement exponential/logarithmic maps for the hyperboloid model, (b) Define octonionic operations in tangent spaces, (c) Test on hierarchical synthetic data (trees, DAGs) before real data, (d) Compare against Poincare embeddings alone. The key question is whether the octonionic structure adds value beyond what hyperbolic geometry already provides |

### D-3: Geometric Signal Detection on Synthetic Data

| Aspect | Detail |
|---------|--------|
| **What** | Design synthetic datasets with known geometric structure (rotations, reflections, hierarchies embedded in 7D imaginary octonion space), train octonionic networks to detect these structures, measure detection accuracy vs baselines |
| **Value Proposition** | Provides clean evidence for whether octonionic representations have genuine affinity for geometric patterns, separate from parameter efficiency. No existing work tests this |
| **Complexity** | Medium |
| **Thesis Claim** | Claim 6 (Geometric Signal Detection) |
| **Notes** | Synthetic data design is critical: (a) Embed known rotation groups in octonion space (use Fano plane subalgebras), (b) Generate noisy observations, (c) Task: identify which rotation group generated the data. Control: same task in R^8, C^4, H^2 with matched parameters. If octonions show no advantage on data with genuine octonionic geometric structure, the thesis claim is falsified. If they show advantage only on octonionic-structured data but not generic data, that is still a meaningful (narrower) result |

### D-4: Multi-Stream Data Fusion Architecture

| Aspect | Detail |
|---------|--------|
| **What** | Use octonionic representations to fuse heterogeneous data streams (financial time series + NLP text sentiment) by mapping each stream to a subset of the 8 octonion dimensions, with cross-stream relationships encoded by the algebra |
| **Value Proposition** | Recent work (HPMRec 2025, H2 model 2024) shows hypercomplex fusion outperforms concatenation-based fusion. Octonionic fusion using the Fano plane's 7 quaternionic subalgebras to capture inter-stream relationships is novel |
| **Complexity** | High |
| **Thesis Claim** | Claim 7 (Multi-Stream Data Fusion) |
| **Notes** | The thesis envisions the 7 imaginary dimensions mapping to different data modalities, with the multiplication table encoding their interactions. This is compelling but speculative. Test plan: (a) Synthetic fusion task with known cross-modal dependencies, (b) Financial fusion (price time series + news sentiment), (c) Compare against: concatenation, attention-based fusion, quaternionic fusion (PHM), cross-attention transformer. Key question: does the algebraic structure of octonionic multiplication capture cross-modal relationships better than learned attention? |

### D-5: Fano Plane Subalgebra Decomposition Analysis

| Aspect | Detail |
|---------|--------|
| **What** | Analyze trained octonionic representations by decomposing them into the 7 quaternionic subalgebras defined by the Fano plane, studying whether the network learns to use these substructures in interpretable ways |
| **Value Proposition** | Bridges the interpretability gap that the thesis explicitly defers. Even partial results would be novel -- no existing work has analyzed learned octonionic representations through this mathematical lens |
| **Complexity** | Medium |
| **Thesis Claim** | Cross-cutting (structural understanding) |
| **Notes** | After training any octonionic network, project learned weights onto each of the 7 quaternionic subalgebras. Measure: (a) Are some subalgebras more active than others? (b) Do different subalgebras specialize on different sub-tasks? (c) Does ablating individual subalgebras degrade performance on specific capabilities? This is analysis, not architecture -- it adds value to any octonionic network experiment |

### D-6: Associator-Aware Architecture Design

| Aspect | Detail |
|---------|--------|
| **What** | Design network architectures that explicitly measure and utilize the associator [a,b,c] = (ab)c - a(bc) as an additional signal rather than treating non-associativity purely as a nuisance |
| **Value Proposition** | Every existing octonionic network paper treats non-associativity as a problem to mitigate. If the associator carries useful information (e.g., measuring "incompatibility" between three representations), this reframes non-associativity as a feature |
| **Complexity** | High |
| **Thesis Claim** | Cross-cutting (novel architectural contribution) |
| **Notes** | The associator is alternating and has magnitude bounded by the product of norms. Potential uses: (a) as an attention-like signal (high associator = three elements are "geometrically incompatible"), (b) as a regularization term (penalize high associator to encourage near-associative computation), (c) as input to a gating mechanism. Requires TS-3 (optimization landscape) to be resolved first -- if non-associativity is pathological, exploiting it is moot |

---

## Anti-Features

Experiments that seem important but are wasteful, misleading, or premature. Explicitly do NOT build these.

### AF-1: Large-Scale Benchmark Competition (ImageNet, etc.)

| Anti-Feature | Matching state-of-the-art on large-scale benchmarks like ImageNet |
|--------------|-----------|
| **Why Avoid** | (a) CIFAR-10/100 already covered by DON paper. Repeating on larger datasets adds computational cost without theoretical insight. (b) State-of-the-art image classification is dominated by architectures (transformers, ConvNeXt) orthogonal to the hypercomplex question. (c) The thesis's claims are about geometric reasoning and density, not about beating ImageNet records. (d) ROCm on RX 7900 XTX may have compatibility issues with large-scale training pipelines |
| **What to Do Instead** | Use small, controlled synthetic datasets where ground truth is known. Use CIFAR only to validate that the implementation reproduces DON paper results (a sanity check, not a contribution) |

### AF-2: Dimensional Interpretability Deep Dive

| Anti-Feature | Attempting to assign human-readable semantic meaning to each of the 8 octonion dimensions |
|--------------|-----------|
| **Why Avoid** | PROJECT.md explicitly marks this out of scope. Word2vec dimensions are not individually interpretable yet the representation is useful. Forcing interpretability onto octonionic representations risks: (a) wasting time on a problem that may have no solution, (b) biasing the architecture toward interpretable-but-suboptimal representations, (c) conflating "I can explain it" with "it works." The thesis itself does not claim dimensional interpretability |
| **What to Do Instead** | Use aggregate analysis methods (D-5, Fano plane decomposition) that study structural properties without requiring per-dimension semantics |

### AF-3: Option C Exotic Hyperbolic Plane (OH^2)

| Anti-Feature | Implementing the exotic octonionic hyperbolic plane from thesis Section 6 |
|--------------|-----------|
| **Why Avoid** | The thesis itself defers this to future work. OH^2 has no established computational framework, no existing implementations to reference, and no clear advantage over the hyperboloid model (Option B) that would justify the enormous mathematical complexity. This is a multi-year research program in its own right |
| **What to Do Instead** | Implement Option B (hyperboloid model) as D-2. Document Option C as future work with specific prerequisites |

### AF-4: Go Language Port During Validation Phase

| Anti-Feature | Porting any component to Go before Python validation is complete |
|--------------|-----------|
| **Why Avoid** | Go lacks autodiff, GPU training support, and ML ecosystem. Porting before validation means maintaining two codebases while the algorithmic questions are still open. If experiments show octonions don't provide advantages, the Go port is wasted effort |
| **What to Do Instead** | Complete all Python validation first. Only port if results are positive and a production system is warranted |

### AF-5: Custom CUDA/ROCm Kernel Optimization

| Anti-Feature | Writing custom GPU kernels for octonionic operations before validating the approach works |
|--------------|-----------|
| **Why Avoid** | Premature optimization. PyTorch's existing operations (expressed as 8-channel real arithmetic) will be fast enough for research-scale experiments. Custom kernels add ROCm-specific debugging burden, risk introducing subtle numerical bugs, and become maintenance debt. The DON paper worked without custom kernels |
| **What to Do Instead** | Use PyTorch's native operations. Profile first. Only optimize if profiling reveals octonionic operations as the actual bottleneck (unlikely at research scale) |

### AF-6: Ablation Studies on Hyperparameters Without Validated Core

| Anti-Feature | Extensive hyperparameter sweeps (learning rate, batch size, architecture depth/width) before the core algebraic operations and gradient computations are verified |
|--------------|-----------|
| **Why Avoid** | If the gradient computation has a subtle bug due to non-associativity, no hyperparameter sweep will fix it -- and "good" hyperparameters found with buggy gradients are meaningless. Hyperparameter sensitivity is important but only after TS-4 (GHR gradients) and TS-5 (numerical stability) are validated |
| **What to Do Instead** | Use standard hyperparameters from quaternionic network literature as starting points. Run ablations only after core correctness is established |

### AF-7: Production-Grade Data Pipeline Architecture

| Anti-Feature | Building robust, scalable data pipelines for financial/NLP data ingestion |
|--------------|-----------|
| **Why Avoid** | This is a research PoC, not a production system. Over-engineering the data pipeline diverts attention from the mathematical and algorithmic questions. Simple CSV/JSON loading with manual preprocessing is sufficient for research |
| **What to Do Instead** | Minimal data loading code. Use existing financial data APIs and pre-processed NLP datasets. Focus engineering effort on the algebra and network layers |

---

## Feature Dependencies

```
TS-1 (Core Algebra)
 |
 +---> TS-4 (GHR Gradients)
 |      |
 |      +---> TS-5 (Numerical Stability)
 |      |      |
 |      |      +---> TS-3 (Optimization Landscape)
 |      |      |      |
 |      |      |      +---> D-6 (Associator-Aware Architecture)
 |      |      |
 |      |      +---> D-2 (Hyperbolic-Octonionic Hybrid)
 |      |
 |      +---> TS-7 (Baselines)
 |             |
 |             +---> TS-2 (Density Comparison)
 |             |      |
 |             |      +---> D-4 (Multi-Stream Fusion)
 |             |
 |             +---> TS-6 (Reversibility)
 |             |
 |             +---> D-3 (Geometric Signal Detection)
 |
 +---> D-1 (G2-Equivariant Layers) [partially independent -- needs algebra,
 |      |                            not necessarily GHR gradients if using
 |      |                            decomposition approach]
 |      +---> D-5 (Fano Plane Analysis) [can also apply to any trained
 |                                        octonionic model]
 |
 +---> D-5 (Fano Plane Analysis) [applicable after any octonionic
                                   network is trained]
```

### Critical Path

The longest dependency chain is:

**TS-1 --> TS-4 --> TS-5 --> TS-3 --> (all remaining experiments)**

This means the core algebra, gradient computation, and stability analysis must be done first. The optimization landscape characterization gates almost everything because a negative result there invalidates training-dependent experiments.

### Parallel Opportunities

- D-1 (G2-Equivariant Layers) can proceed in parallel with TS-4/TS-5 if using the decomposition-to-reals approach for gradients
- D-5 (Fano Plane Analysis) can be developed as tooling in parallel, applied whenever any octonionic model is trained
- TS-7 (Baselines) can be implemented and validated on published benchmarks in parallel with octonionic core development

---

## MVP Recommendation

### Phase 1: Foundation (must complete before any training)
1. **TS-1** -- Core Algebra Verification Suite
2. **TS-4** -- GHR Gradient Implementation (or decomposition fallback)
3. **TS-5** -- Numerical Stability Analysis

### Phase 2: Viability Gate (determines whether to continue)
4. **TS-3** -- Optimization Landscape Characterization
5. **TS-7** -- Baseline Implementation and Validation

**GO/NO-GO decision point: If TS-3 shows pathological landscapes, pivot to studying WHY (publishable negative result) rather than pushing forward.**

### Phase 3: Core Claims Validation
6. **TS-2** -- Matched-Parameter Density Comparison
7. **TS-6** -- Controlled Reversibility Experiment
8. **D-3** -- Geometric Signal Detection (synthetic data)

### Phase 4: Differentiation (only if Phase 3 positive)
9. **D-1** -- G2-Equivariant Layers (highest novelty)
10. **D-4** -- Multi-Stream Data Fusion
11. **D-2** -- Hyperbolic-Octonionic Hybrid
12. **D-5** -- Fano Plane Analysis
13. **D-6** -- Associator-Aware Architecture

### Defer Indefinitely
- AF-1 through AF-7 (all anti-features)

---

## Feature Prioritization Matrix

| ID | Feature | Thesis Claim | Complexity | Risk if Skipped | Novelty | Priority |
|----|---------|-------------|------------|-----------------|---------|----------|
| TS-1 | Core Algebra | All | Low | Fatal | None (expected) | **P0** |
| TS-4 | GHR Gradients | Claim 8 | High | Fatal | Medium | **P0** |
| TS-5 | Numerical Stability | Claim 9 | Medium | Fatal | Low | **P0** |
| TS-3 | Optimization Landscape | Claim 3 | High | Fatal | High | **P0** |
| TS-7 | Baselines | All | Medium | Fatal (no comparison) | None | **P1** |
| TS-2 | Density Comparison | Claim 1 | Medium | Thesis unvalidated | Low | **P1** |
| TS-6 | Reversibility | Claim 2 | High | Strongest claim lost | High | **P1** |
| D-3 | Geometric Signal Detection | Claim 6 | Medium | Moderate | Medium | **P2** |
| D-1 | G2-Equivariant Layers | Claim 4 | Very High | Moderate | Very High | **P2** |
| D-4 | Multi-Stream Fusion | Claim 7 | High | Moderate | Medium | **P3** |
| D-2 | Hyperbolic-Octonionic | Claim 5 | Very High | Moderate | High | **P3** |
| D-5 | Fano Plane Analysis | Cross-cutting | Medium | Low | Medium | **P3** |
| D-6 | Associator Architecture | Cross-cutting | High | Low | High | **P4** |

Priority Key:
- **P0**: Must complete; gates everything else
- **P1**: Must complete for thesis validation
- **P2**: Should complete for strong contribution
- **P3**: Complete if P2 results are encouraging
- **P4**: Complete if time and results permit

---

## Thesis Claim Coverage Matrix

| Thesis Claim | Primary Experiment | Supporting Experiments | Status |
|-------------|-------------------|----------------------|--------|
| 1. Representational Density | TS-2 | TS-7, D-3 | Covered |
| 2. Reversibility | TS-6 | TS-5 (stability of inverse) | Covered |
| 3. Optimization Landscape | TS-3 | TS-5 | Covered |
| 4. G2-Equivariant Layers | D-1 | TS-1 (algebra verification) | Covered |
| 5. Hyperbolic-Octonionic | D-2 | TS-5 (projection stability) | Covered |
| 6. Geometric Signal Detection | D-3 | TS-2 (density context) | Covered |
| 7. Multi-Stream Fusion | D-4 | D-5 (subalgebra analysis) | Covered |
| 8. GHR Calculus Gradients | TS-4 | TS-5 | Covered |
| 9. Numerical Stability | TS-5 | TS-3, TS-6 (stability under inversion/optimization) | Covered |

---

## Sources

### Octonion Neural Networks
- [Deep Octonion Networks (Zhu et al., 2019)](https://arxiv.org/abs/1903.08478) -- CIFAR-10/100 experiments with DON vs DRN/DCN/DQN baselines
- [Hypercomplex Neural Networks: Exploring Quaternion, Octonion, and Beyond (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12513225/) -- Comprehensive survey identifying open problems in ONN training stability and standardization
- [Octonion-Valued Neural Networks (Springer, 2016)](https://link.springer.com/content/pdf/10.1007/978-3-319-44778-0_51.pdf) -- Early ONN formulation

### Quaternion Networks and GHR Calculus
- [GHR Calculus for Quaternion Derivatives (Xu et al., 2015)](https://arxiv.org/abs/1409.8168) -- Foundation for quaternionic backpropagation
- [Learning Algorithms in QNN using GHR Calculus](https://www.nnw.cz/doi/2017/NNW.2017.27.014.pdf) -- First deep GHR-based backpropagation
- [A Comparison of Quaternion Neural Network Backpropagation Algorithms (2023)](https://www.sciencedirect.com/science/article/pii/S0957417423009508) -- DoE methodology for QNN benchmarking

### Parameterized Hypercomplex Networks
- [PHNNs: Parameterized Hypercomplex Convolutions (Grassucci et al., 2022)](https://arxiv.org/abs/2110.04176) -- Generalizable framework via Kronecker products
- [HyperNets PyTorch library](https://github.com/eleGAN23/HyperNets) -- Existing implementation for PHC layers
- [Towards Explaining Hypercomplex Neural Networks (2024)](https://arxiv.org/abs/2403.17929) -- Interpretability through PHB-cos transforms

### Equivariant Networks and Lie Groups
- [Lie Group Decompositions for Equivariant Neural Networks (ICLR 2024)](https://arxiv.org/abs/2310.11366) -- Handles reductive groups but NOT G2
- [Accelerated Discovery of Exceptional Lie Groups G2, F4, E6 (He et al., 2023)](https://www.sciencedirect.com/science/article/pii/S0370269323006007) -- ML discovers G2 symmetry but does not build G2-equivariant layers
- [General Framework for Equivariant NNs on Reductive Lie Groups](https://arxiv.org/abs/2306.00091) -- Broad framework, exceptional groups remain unimplemented

### Hyperbolic Networks
- [Poincare Embeddings (Nickel & Kiela, 2017)](https://arxiv.org/abs/1705.08039) -- Foundational hyperbolic embedding work
- [Hyperbolic Graph Convolutional Networks (Chami et al., 2019)](https://cs.stanford.edu/people/jure/pubs/hgcn-neurips19.pdf) -- HGCN benchmarks
- [Hyperbolic Graph Wavelet Neural Network (2025)](https://www.sciopen.com/article/10.26599/TST.2024.9010032) -- Recent hyperbolic GNN

### Invertible Neural Networks
- [Analyzing Inverse Problems with INNs (Ardizzone et al., 2018)](https://arxiv.org/abs/1808.04730) -- INN framework for backward inference
- [Reversible Residual Network (Gomez et al., 2017)](http://papers.neurips.cc/paper/6816-the-reversible-residual-network-backpropagation-without-storing-activations.pdf) -- RevNet architecture

### Multimodal Fusion
- [Hierarchical Hypercomplex Network for Multimodal Emotion Recognition (2024)](https://arxiv.org/html/2409.09194) -- PHM-based fusion
- [Hypercomplex Prompt-aware Multimodal Recommendation (2025)](https://arxiv.org/html/2508.10753) -- HPMRec fusion strategy

### Mathematical References
- [G2 as Automorphism Group of Octonions (Baez, 2002)](https://math.ucr.edu/home/baez/octonions/node14.html) -- G2 structure and properties
