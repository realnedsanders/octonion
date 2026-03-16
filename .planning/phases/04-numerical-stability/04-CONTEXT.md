# Phase 4: Numerical Stability - Context

**Gathered:** 2026-03-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Precision characteristics of octonionic operations are quantified so that architecture decisions (depth, float width, mitigations) are evidence-based. Produces: (1) forward pass error accumulation curves across depths 10/50/100/500, (2) condition number characterization as a function of input magnitude, (3) float32 vs float64 convergence comparison, and (4) demonstrated re-normalization mitigation extending stable depth by at least 2x. Does NOT include: optimization landscape experiments (Phase 5), density comparison experiments (Phase 7), or gradient stability during training (Phase 5).

</domain>

<decisions>
## Implementation Decisions

### Error Accumulation Experiment Design
- Run TWO separate experiments for "depth N": (1) minimal stripped-down OctonionLinear chain (no BN/activation) to isolate algebraic precision; (2) full AlgebraNetwork with BN and activations to characterize practical degradation — results reported separately
- Depths tested: 10, 50, 100, 500 layers across both setups
- All four algebras (R, C, H, O) included at each depth for comparison — shows whether error growth is specific to octonions or a general pattern
- Fresh random weights per layer (independent initialization per layer, not repeated application of the same transform)
- Weight initialization matches Phase 3 initialization conventions: Kaiming/He for R, Trabelsi et al. for C, Parcollet et al. for H, unit-norm for O

### Error Accumulation Measurement
- Measure BOTH metrics per depth:
  1. Relative error: `||float32_output - float64_output|| / ||float64_output||` (float64 as ground truth)
  2. Norm drift: `||actual_norm - expected_norm|| / expected_norm` testing algebraic invariant ||ab|| = ||a||·||b||
- 100–1000 random inputs per (depth, algebra) measurement; report mean ± std
- Three input magnitude regimes tested: near-unit (||x|| ≈ 1), small (||x|| ≈ 0.01), large (||x|| ≈ 100)

### Condition Number Characterization
- Scope: full network-level condition numbers — includes primitive operations (mul, inv, exp, log), N-layer OctonionLinear compositions (2, 5, 10 layers), and full forward pass including OctonionBN and OctonionConv
- Computation method: numeric Jacobian via finite differences (reusing `_numeric.py` from Phase 2)
- Input dimension: small fixed size (8 or 16 octonionic units) to keep numeric Jacobian tractable (64–128 real inputs → 64×64–128×128 Jacobian)
- All four algebras (R, C, H, O) compared side-by-side as condition number vs input magnitude curves

### Stability Threshold
- "Stable" = relative error (float32 vs float64) < 1e-3; "Unstable" = error ≥ 1e-3
- "Stable depth" = maximum number of layers before error crosses 1e-3 threshold
- SC-4's "2x" = mitigation doubles the stable depth (layer count)
- Threshold applied to BOTH stripped-down chain AND full AlgebraNetwork; results reported separately
- Precision scope: float32 vs float64 only (no bfloat16/float16)

### Mitigation Strategy
- Strategy: periodic re-normalization (projects layer outputs back toward unit norm every K layers)
- Implementation: new `StabilizingNorm` nn.Module in `src/octonion/baselines/`, reusable by downstream phases (not one-off analysis-only code)
- Parameterized: `normalize_every=K` (configurable, default=10)
- Algebra coverage: all four algebras (R, C, H, O) — each using its native norm (abs, complex modulus, quaternion norm, octonion norm)
- Analysis script sweeps K ∈ {5, 10, 20} to characterize stability vs overhead trade-off
- StabilizingNorm becomes a reusable layer that downstream phases (5, 6, 7) can enable/disable via config flag

### Output and Delivery
- Single comprehensive standalone script: `scripts/analyze_stability.py` covering all four SCs
  - Sections: (1) depth sweep / error accumulation, (2) condition number characterization, (3) float32 vs float64 comparison, (4) mitigation demonstration
  - Outputs: matplotlib plots (depth-vs-error curves, condition number curves, mitigation comparison), JSON data files, printed summary table
  - Follows same pattern as `scripts/demo_naive_vs_correct.py`
- pytest suite (`tests/test_numerical_stability.py`): smoke tests verifying measurement infrastructure runs correctly — NOT pass/fail assertions on SC values (2x achievement is an experimental outcome, not a correctness requirement)

### Claude's Discretion
- Exact number of random inputs (within 100–1000 range) per measurement
- JSON output schema and file naming
- Plot styling and layout
- K sweep values within condition number composition chains (2, 5, 10 or similar)
- Whether to include a printed summary table in addition to JSON/plots

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` §FOUND-03 — Exact success criteria for this phase (4 criteria: depth sweep, condition numbers, float32 vs float64, mitigation 2x)

### Prior phase infrastructure (reuse these, don't rebuild)
- `.planning/phases/02-ghr-calculus/02-CONTEXT.md` — Established that `_jacobians.py` and `_numeric.py` (finite-difference Jacobian) feed Phase 4 condition number analysis; test tolerance conventions
- `.planning/phases/03-baseline-implementations/03-CONTEXT.md` — AlgebraNetwork with configurable depth, per-algebra initialization, training utility identified as depth-sweep infrastructure

No external thesis sections or ADRs were referenced during discussion — all requirements are captured in FOUND-03 and the decisions above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `left_mul_matrix` / `right_mul_matrix` (`src/octonion/_linear_algebra.py`): 8x8 real matrices for analytic condition numbers of multiplication — direct input to `torch.linalg.svd` for σ_max/σ_min
- `_jacobians.py` (`src/octonion/calculus/`): Analytic 8×8 real Jacobians for all operations (mul, exp, log, conjugate, inverse, inner product, cross product)
- `_numeric.py` (`src/octonion/calculus/`): Finite-difference Jacobian utility — reuse directly for network-level condition number computation
- `AlgebraNetwork` (`src/octonion/baselines/_network.py`): Configurable depth, topology (MLP/Conv/Recurrent) — use for the full AlgebraNetwork depth sweep
- `_initialization.py` (`src/octonion/baselines/`): Kaiming/He (R), Trabelsi (C), Parcollet (H), unit-norm (O) — use in depth sweep weight initialization
- `scripts/demo_naive_vs_correct.py`: Pattern for standalone analysis scripts (JSON output, matplotlib plots, 1000 random inputs)
- `scripts/profile_baseline.py`: Pattern for measurement/profiling scripts

### Established Patterns
- float64 for verification (ground truth), float32 for neural layers — both already exist in the codebase
- Batch-first `[..., dim]` tensor shapes throughout
- Module-level constant tensors built at import time (STRUCTURE_CONSTANTS)
- Standalone analysis scripts with JSON output + matplotlib plots (Phase 2 demo_naive_vs_correct.py)
- Phase 3 training: AMP disabled for octonion (float16 Cholesky failure) — relevant to mixed-precision options

### Integration Points
- `StabilizingNorm` → new module in `src/octonion/baselines/` alongside `_normalization.py`; inserted into AlgebraNetwork via config flag
- Phase 5 (Optimization Landscape) will use stable/unstable depth characterization to inform training depth choices
- Phase 5 will also use the Hessian eigenspectrum analysis (needs `create_graph=True` from Phase 2 autograd functions)
- Analyze_stability.py output (JSON) feeds architectural decisions for all subsequent experimental phases

</code_context>

<specifics>
## Specific Ideas

- BN whitening may already act as implicit stabilization — reporting stripped-down chain vs full AlgebraNetwork separately will expose whether this is the case
- K sweep (normalize_every ∈ {5, 10, 20}) will characterize the stability vs computation overhead trade-off for StabilizingNorm
- OctonionBN known bottleneck (131ms/forward at depth=28, Phase 3) — condition number characterization will show whether this cost is justified by stability benefits

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-numerical-stability*
*Context gathered: 2026-03-16*
