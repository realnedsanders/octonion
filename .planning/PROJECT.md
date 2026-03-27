# Octonionic Computation Substrate — Research PoC

## What This Is

A research project that systematically validates the claims made in the "Octonionic Computation as a Substrate for Geometric Reasoning in Machine Learning" thesis, and builds a proof-of-concept Octonionic Reasoning Engine (ORE). The project proceeds bottom-up: implement core algebra, validate each theoretical claim with controlled experiments against real/complex/quaternion baselines, then assemble validated components into a working system that processes real-world data streams (financial markets, NLP). All experimental work uses Python with ROCm PyTorch on an AMD RX 7900 XTX (24GB VRAM). A Go port is considered only after full validation.

## Core Value

Determine empirically whether octonionic representations provide measurable advantages over quaternionic, complex, and real-valued alternatives for geometric reasoning in ML — and if so, build a working system that demonstrates it.

## Requirements

### Validated

- [x] Core octonionic algebra library (multiplication, conjugation, inversion, associator) with property-based tests — Validated in Phase 1: Octonionic Algebra
- [x] GHR calculus gradient implementation for octonionic backpropagation — Validated in Phase 2: GHR Calculus
- [x] Baseline implementations: fair R/C/H/O comparison networks with matched parameter counts — Validated in Phase 3: Baseline Implementations
- [x] Numerical stability analysis across all components — Validated in Phase 4: Numerical Stability

### Active

- [ ] Optimization landscape characterization: does non-associativity create pathological loss surfaces?
- [ ] Controlled density experiments: octonion vs quaternion vs complex vs real on matched-parameter tasks
- [ ] Reversible reasoning experiments: demonstrate backward inference through octonionic transformations
- [ ] G₂-equivariant layer implementation and training pipeline
- [ ] Hyperboloid-octonionic hybrid model (Option B from thesis §6.2)
- [ ] Hyperbolic projection stability analysis (the central open problem from thesis §9.7)
- [ ] Geometric signal detection on synthetic data with known ground truth
- [ ] Multi-stream data fusion architecture (financial + text streams)
- [ ] End-to-end ORE proof-of-concept on real-world data

### Out of Scope

- Go port — deferred until full Python validation is complete and results warrant it
- Option C (𝕆H² exotic octonionic hyperbolic plane) — deferred to future research per thesis recommendation
- Production deployment — this is a research PoC, not a production system
- Mobile/edge optimization — focus on correctness and validation, not deployment constraints
- Dimensional semantics interpretability — not expected to yield human-readable dimension meanings; focus on whether it works, not explaining individual dimensions

## Context

**Thesis foundation:** The project follows the theoretical framework from `docs/thesis/main.tex` which argues that octonions (𝕆), as the largest normed division algebra, constitute an optimal substrate for encoding knowledge and performing reasoning in ML. Key mathematical underpinnings include Hurwitz's theorem (hard ceiling at dim 8), G₂ automorphism group (14-dim structure-preserving transforms), and the Fano plane decomposition into 7 overlapping quaternionic subalgebras.

**Primary risk:** The optimization landscape question — if non-associativity creates pathological loss surfaces, the entire approach may be untrainable. This must be addressed early.

**Key claim to test first:** Reversibility — that algebraic invertibility enables meaningful backward reasoning about missing/uncertain information, something conventional nets cannot do.

**Hardware:** AMD RX 7900 XTX (24GB VRAM, RDNA3) via ROCm PyTorch docker container. ROCm support for custom CUDA kernels may require adaptation.

**Methodology:** Bottom-up construction with controlled experiments at each layer. Each claim from the thesis gets independent testing with real/complex/quaternion/octonion baselines using matched parameter counts. Synthetic data first (known ground truth), then financial markets and NLP text.

**Baselines:** Hybrid approach — use published quaternionic/complex network implementations where available, build custom where needed for fair apple-to-apple comparison.

**Literature:** Parcollet et al. (2019) for quaternionic backprop, Nickel & Kiela (2017) for Poincaré embeddings, Ganea et al. (2018) for hyperbolic NNs, Chen et al. (2018) for Neural ODEs, Baez (2002) for octonionic algebra reference.

## Constraints

- **Hardware**: AMD RX 7900 XTX with ROCm — no CUDA, must verify PyTorch ROCm compatibility for all custom operations
- **Language**: Python (PyTorch) for all experimental work; Go port only after full validation
- **Methodology**: Every architectural claim must be tested against baselines with matched parameter counts before integration
- **Research-grade**: Code must be reproducible with fixed seeds, documented hyperparameters, and statistical significance testing

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Python/PyTorch over Go for experimentation | Autodiff, GPU acceleration, existing ML ecosystem; Go lacks these for training | — Pending |
| Synthetic data before real data | Need known ground truth to validate claims; real data has too many confounds | — Pending |
| Test reversibility first | Most novel claim; if it fails, remaining claims still have value but the thesis loses its strongest differentiator | — Pending |
| Optimization landscape as highest risk | If non-associative loss surfaces are pathological, nothing else matters | — Pending |
| Skip dimensional interpretability | Expecting opaque learned semantics (like word2vec); focus on performance not explanation | — Pending |
| Full thesis scope | Validate all major claims rather than cherry-picking; systematic approach | — Pending |

---
*Last updated: 2026-03-20 after Phase 4 gap closure (04-03: NaN/overflow fixes verified)*
