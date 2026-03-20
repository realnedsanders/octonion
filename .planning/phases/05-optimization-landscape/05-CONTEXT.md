# Phase 5: Optimization Landscape - Context

**Gathered:** 2026-03-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Determine whether octonionic networks can be trained reliably. This is the **go/no-go gate** for the entire project. Produces: (1) gradient variance characterization across 20+ seeds on 5 synthetic tasks, (2) Hessian eigenspectrum analysis at convergence and 3 training checkpoints, (3) training convergence profiles across 5 optimizers (SGD, Adam, Riemannian Adam, LBFGS, Shampoo) for all algebras, (4) convergence comparison against R8-dense-mixing baseline with tiered gate verdict, (5) Bill & Cox loss surface curvature measurement. Does NOT include: density comparison experiments (Phase 7), reversibility experiments (Phase 6), G2 equivariance (Phase 8), or production training pipelines.

</domain>

<decisions>
## Implementation Decisions

### Synthetic Task Suite
- **5 tasks total**, each with 50K train / 10K test:
  1. **Algebra-native regression (single-layer)**: Learn y = a*x*b (one OctonionLinear layer). Known optimal loss = 0. 8D input.
  2. **Algebra-native regression (multi-layer)**: Learn y = f(g(h(x))) with OctonionLinear layers at depths 3, 5, and 10. Tests whether non-associativity compounds optimization difficulty. 8D input.
  3. **7D cross product recovery**: Plant 7D cross product signal in noisy 7D data. Full controls:
     - Quaternion negative control (quaternions cannot exploit 7D cross product)
     - 3D cross product positive control (validates methodology — quaternions should win)
     - PHM-8 baseline (isolates octonionic multiplication rules from generic Kronecker structure)
     - Multiple noise levels: clean (0%), low (5%), medium (15%), high (30%)
  4. **Sinusoidal regression**: Learn sin(w*x) or sum-of-sines. Continuous, differentiable, known optimum. 8D input.
  5. **Multi-class classification**: Synthetic Gaussian clusters in R^8. Known Bayes-optimal accuracy. Tests generic classification landscape.
- **Input dimensionality**: Task-appropriate dimensions (8D for algebra-native, 7D for geometric/cross-product tasks, 8D for standard tasks) PLUS a 64D variant of each task to test whether octonionic advantage scales with input dimensionality
- **20 seeds** per task per algebra per optimizer (uniform across all experiments)
- **Literature comparison**: Results explicitly reference and compare against published numbers (Bill & Cox loss surface smoothness, Wu et al. DON convergence) where possible

### PHM-8 Baseline
- Build a minimal PHM-8 layer (sum of Kronecker products with learned mixing coefficients) integrated directly into the existing codebase
- Do NOT use published PHM code — build from scratch for full control and codebase consistency
- Integrate into AlgebraNetwork skeleton alongside R/C/H/O modules
- Purpose: isolate octonionic algebra advantage from generic Kronecker factorization structure advantage (Zhang et al., ICLR 2021)

### R8-Dense-Mixing Baseline
- Build in Phase 5 (not deferred to Phase 7) — required for SC-4 go/no-go gate
- Real-valued 8D with dense cross-component interaction (full 8x8 mixing matrix per layer, not block-diagonal)
- Parameter-matched against octonionic network

### Go/No-Go Gate
- **Tiered gate** with thresholds locked before experiments:
  - **GREEN** (proceed): Octonionic loss within 2x of R8-dense-mixing on ALL tasks
  - **YELLOW** (proceed with caution): Within 2x on 2+ tasks OR within 3x on ALL tasks
  - **RED** (pivot): Worse than 3x on majority of tasks
- **Loss metrics** (both reported, gate uses most favorable for O):
  1. Best validation loss ratio: best_val_loss(O) / best_val_loss(R8-dense-mixing)
  2. Final converged loss ratio: median final val loss across 20 seeds (final = best loss in last 10% of epochs)
  3. Divergence rate: fraction of seeds where loss > 10x initial. Additional RED flag if >50%
- **Gate decision based purely on loss quality** — wall-clock time is reported but does not affect verdict
- **YELLOW handling**: User will decide based on results after review (no pre-committed constraints on subsequent phases)
- **RED output**: Full quantitative characterization paper (publishable negative result documenting exactly why non-associativity breaks optimization) AND pivot plan (which claims survive, alternative approaches)

### Optimizer Suite (SC-3)
- **5 optimizers** tested on all algebras across all tasks:
  1. **SGD** (with momentum 0.9, Nesterov)
  2. **Adam** (standard)
  3. **Riemannian Adam** (via geoopt library) — manifold constraints:
     - S^7 (unit norm per octonion) AND Stiefel manifold — try both
     - Applied to ALL algebras on their respective natural manifolds: R (Euclidean/unconstrained), C (S^1), H (S^3), O (S^7)
  4. **LBFGS** (PyTorch built-in) — curvature-informed, may reveal octonionic landscape structure
  5. **Shampoo/K-FAC** (existing library, e.g., google/shampoo or KFAC-PyTorch) — Kronecker-factored curvature, interesting interaction with octonionic Kronecker structure
- Convergence profiles: loss vs step for all (algebra, optimizer, task) combinations

### Hessian Analysis (SC-2)
- **Dual-scale approach**:
  1. **Small networks, full Hessian**: Deliberately small networks via torch.autograd.functional.hessian(). Exact eigenspectrum. Claude determines max parameter budget based on 24GB VRAM and compute time.
  2. **Larger networks, stochastic Lanczos**: Hessian-vector products (via create_graph=True) with ~200 Lanczos iterations for full spectral density estimate (Ghorbani et al. 2019 style). Approximates eigenvalue density.
- Cross-validate: if both approaches agree on small networks, trust stochastic results on larger ones
- **4 checkpoints**: initialization, 25% training, 50% training, convergence — shows landscape evolution
- **5 representative seeds** for evolution analysis (best, worst, median, 2 random). Full 20-seed Hessian only at convergence (for SC-2 gate metric).
- **Full eigendecomposition only** — no separate Hutchinson trace or Frobenius norm computation (derivable from eigenvalues)
- SC-2 metric: ratio of negative eigenvalues for octonionic vs quaternionic at convergence

### Loss Surface Curvature (Bill & Cox Extension)
- Implement Bill & Cox (2024) methodology: sample random directions from converged solutions, measure average loss surface curvature
- Compare curvature across: R, C, H, O, PHM-8, R8-dense-mixing
- Directly comparable to published quaternion results
- Applied to all tasks (geometric and non-geometric) to test whether curvature advantage is task-specific or general

### Claude's Discretion
- Exact network architectures for each task (hidden width, batch norm, activation)
- Full Hessian max parameter budget (based on GPU memory constraints)
- PHM-8 implementation details (number of Kronecker terms, initialization)
- R8-dense-mixing layer design
- Data generation procedures for synthetic tasks (sampling distributions, ground truth functions)
- Bill & Cox curvature sampling methodology details
- Geoopt manifold definitions for C and H (S^1 and S^3)
- Shampoo/K-FAC library choice and integration details
- Training hyperparameters per optimizer (LR, weight decay, scheduler)
- Which published results to compare against and how to normalize for comparison

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` §FOUND-04 — Exact success criteria for this phase (5 criteria: gradient variance, Hessian eigenspectrum, convergence profiles, 2x convergence gate, pathology characterization)

### Prior phase infrastructure (reuse these, don't rebuild)
- `.planning/phases/02-ghr-calculus/02-CONTEXT.md` — create_graph=True in all autograd Functions enables Hessian computation; GHR calculus provides gradient correctness guarantees
- `.planning/phases/03-baseline-implementations/03-CONTEXT.md` — AlgebraNetwork skeleton, training utility, comparison runner, statistical testing, per-algebra tuning + Adam-lock option
- `.planning/phases/04-numerical-stability/04-CONTEXT.md` — StabilizingNorm (stabilize_every config), depth stability characterization, Phase 4 results inform depth choices

### Key literature (inform experiment design)
- Bill & Cox (2024) "Exploring Quaternion Neural Network Loss Surfaces" — loss surface curvature methodology to replicate
- Zhang et al. (ICLR 2021) "Beyond Fully-Connected Layers with Quaternions: PHM" — PHM-n architecture for the PHM-8 baseline
- Ghorbani et al. (2019) — stochastic Lanczos quadrature for Hessian spectral density estimation
- Wu & Xu et al. (2020) "Deep Octonion Networks" — only published octonionic network experiments (CIFAR), compare convergence
- Shen et al. (ECCV 2020) "3D-Rotation-Equivariant Quaternion Neural Networks" — methodology validation for geometric tasks
- Comminiello et al. (IEEE SPM 2024) "Demystifying the Hypercomplex" — inductive bias framework for interpreting results

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_trainer.py` (src/octonion/baselines/): Full training loop with gradient stats, VRAM monitoring, checkpointing, early stopping, AMP — direct infrastructure for all experiments
- `_comparison.py` (src/octonion/baselines/): Multi-algebra multi-seed comparison runner with statistical significance testing — extend for PHM-8 and R8-dense-mixing
- `_stats.py` (src/octonion/baselines/): paired_comparison, holm_bonferroni — reuse for pairwise statistical tests
- `_plotting.py` (src/octonion/baselines/): Convergence curves, comparison bars — extend for landscape-specific plots
- `_config.py` (src/octonion/baselines/): AlgebraType enum, NetworkConfig with stabilize_every — extend for new baselines
- `_autograd_functions.py` (src/octonion/calculus/): All operations support create_graph=True for Hessian computation
- `StabilizingNorm` (src/octonion/baselines/_stabilization.py): Available via stabilize_every config for depth experiments
- `_numeric.py` (src/octonion/calculus/): Finite-difference Jacobian — reuse for gradient verification

### Established Patterns
- float32 for training, float64 for verification
- Batch-first `[..., dim]` tensor shapes
- seed_everything() for reproducibility
- Standalone analysis scripts (scripts/analyze_stability.py pattern) with JSON output + matplotlib plots
- AlgebraType enum for multi-algebra dispatch
- Config-driven via dataclasses

### Integration Points
- PHM-8 and R8-dense-mixing layers integrate alongside existing RealLinear/ComplexLinear/QuaternionLinear/OctonionDenseLinear
- Geoopt Riemannian Adam integrates via torch.optim.Optimizer interface in existing trainer
- Hessian computation uses create_graph=True from Phase 2 autograd Functions
- Phase 4 StabilizingNorm available for all depth experiments
- Results feed the tiered go/no-go gate that determines whether Phases 6-9 proceed

</code_context>

<specifics>
## Specific Ideas

- The 7D cross product task is framed as an **inductive bias / data efficiency** test, not a circular "can octonions detect octonionic structure" test. The quaternion negative control (can't exploit 7D) and PHM-8 baseline (separates algebra from Kronecker structure) are critical controls per Zhang et al. (ICLR 2021)
- The 3D cross product positive control validates the methodology: if quaternions don't show advantage on 3D cross product recovery, the experimental setup is broken
- Bill & Cox (2024) showed quaternion MLPs have smoother loss surfaces even on non-geometric tasks — our curvature measurement extends this to octonions and tests whether the smoothness advantage scales with algebra dimension
- Comminiello et al. (2024) framework: results should be interpretable along their three axes (algebraic bias, geometric bias, regularization bias) to identify WHICH mechanism drives any observed octonionic advantage or disadvantage
- Multi-layer algebra-native regression at depths 3/5/10 directly characterizes whether non-associativity compounds optimization difficulty — Phase 2's parenthesization-aware chain rule ensures gradients are correct, so any degradation is landscape-related not gradient-error-related
- Riemannian Adam on all algebras with their natural manifolds (R=Euclidean, C=S^1, H=S^3, O=S^7 + Stiefel) ensures fair comparison — each algebra gets its geometrically appropriate constraint

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-optimization-landscape*
*Context gathered: 2026-03-20*
