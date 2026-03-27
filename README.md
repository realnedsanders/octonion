# Octonionic Computation Substrate

A research project that tests whether **octonions** — the largest normed division algebra — provide measurable advantages over real, complex, and quaternionic alternatives for geometric reasoning in machine learning.

## What are octonions?

There are exactly four number systems where you can add, subtract, multiply, divide, and preserve lengths. These are the **normed division algebras**, and each is built by doubling the previous one:

| Algebra | Dimension | What you lose |
|---------|-----------|---------------|
| Real numbers (R) | 1 | — |
| Complex numbers (C) | 2 | Ordering |
| Quaternions (H) | 4 | Commutativity |
| **Octonions (O)** | **8** | **Associativity** |

This is it. [Hurwitz's theorem (1898)](https://en.wikipedia.org/wiki/Hurwitz%27s_theorem_(composition_algebras)) proves there will never be a 16-dimensional version — beyond octonions, you get zero divisors, which destroy invertibility and make the algebra unsuitable for reversible computation.

Octonions are already used in theoretical physics (string theory, M-theory, exceptional Lie groups). This project asks: **can they do useful work in machine learning?**

## The thesis

The full argument is in [`octonionic-ml-thesis.md`](octonionic-ml-thesis.md). The core claims:

1. **Density**: Octonions pack more geometric structure per parameter than smaller algebras. Their automorphism group (G2, 14-dimensional) is larger than the algebra itself (8-dimensional) — unique among division algebras.

2. **Reversibility**: Every non-zero octonion has a unique inverse. Networks built from octonionic operations can run backward through their own computations, enabling reasoning about missing or uncertain information.

3. **Inductive bias for free**: Octonionic multiplication automatically preserves norms, respects Moufang identities, and decomposes into 7 overlapping quaternionic subalgebras via the Fano plane. These are geometric priors that real-valued networks would have to learn from data.

The project validates these claims empirically, bottom-up, with controlled experiments against fair baselines.

## What's been built

This is a PyTorch-native library implementing the full stack from algebra to experiment infrastructure:

```
src/octonion/
    Core algebra       Multiplication, conjugation, norm, inverse, associator
                       Fano plane + Cayley-Dickson construction, R/C/H/O tower types
                       OctonionLinear layer, batch operations, exp/log

    calculus/           GHR Wirtinger derivatives for octonionic backpropagation
                       Analytic 8x8 Jacobians for all 7 primitives
                       Parenthesization-aware chain rule (non-associativity matters)
                       torch.autograd.Function implementations with create_graph=True

    baselines/          Fair comparison networks: R (8x width), C (4x), H (2x), O (1x)
                       Parameter matching via binary search (<1% deviation)
                       Algebra-specific batch normalization (Cholesky whitening)
                       AlgebraNetwork skeleton (MLP, Conv2D, Recurrent topologies)
                       Training infrastructure with gradient stats, AMP, TensorBoard

    landscape/          Hessian eigenspectrum (full + stochastic Lanczos)
                       Bill & Cox loss surface curvature measurement
                       Gradient variance collection across seeds
                       Go/no-go gate evaluation (GREEN/YELLOW/RED)
                       Experiment orchestration with incremental saves

    tasks/              Synthetic task generators with known optima
                       algebra_native, cross_product, sinusoidal, classification
```

**739 tests** covering algebra, calculus, baselines, stability, and landscape (pytest + Hypothesis property-based testing).

## Current status

The project proceeds in phases, each validating a specific claim before building on it:

| Phase | What it validates | Status |
|-------|-------------------|--------|
| 1. Octonionic Algebra | Core operations, Moufang identities, norm preservation | Complete |
| 2. GHR Calculus | Octonionic gradients match finite-difference to <1e-5 | Complete |
| 3. Baselines | Fair R/C/H/O comparison networks, matched parameters | Complete |
| 4. Numerical Stability | Precision vs depth, condition numbers, mitigations | Complete |
| **5. Optimization Landscape** | **Go/no-go gate: can octonionic networks train?** | **Experiments running** |
| 6. Reversibility | Algebraic inversion vs RevNet/INN baselines | Blocked on Phase 5 |
| 7. Density & Geometry | Matched-parameter density advantage, geometric detection | Blocked on Phase 5 |
| 8. G2 & Hyperbolic | G2-equivariant layers, hyperboloid-octonionic hybrid | Blocked on Phase 5 |
| 9. Associator Analysis | Non-associativity as signal, Fano subalgebra specialization | Blocked on Phase 5 |

Phase 5 is the critical gate. If non-associativity creates pathological loss surfaces, the project pivots to documenting a quantitative negative result. If the gate passes, Phases 6-9 proceed.

The full experiment matrix (6 algebras x 9 tasks x 5 optimizers x 20 seeds) is currently running. Results will be published here once the gate verdict is finalized.

## Development setup

All computation runs in a Docker container with ROCm PyTorch (AMD GPU). The host machine is used only for git and file editing.

```bash
# Build the container (first time)
docker compose build

# Install dependencies
docker compose run --rm dev uv sync

# Run tests
docker compose run --rm dev uv run pytest

# Run a smoke experiment
docker compose run --rm dev uv run python scripts/run_landscape.py --smoke
```

**Requirements**: Docker, an AMD GPU with ROCm support (developed on RX 7900 XTX, 24GB VRAM). NVIDIA GPUs would require swapping the base container image.

## Key technical details

### Why non-associativity matters for gradients

In associative algebras (R, C, H), the chain rule is straightforward: `d/dx f(g(x)) = f'(g(x)) * g'(x)`. Parenthesization doesn't matter because `(AB)C = A(BC)`.

In octonions, **parenthesization changes the answer**. The chain rule must track how intermediate results are grouped. This project implements a parenthesization-aware chain rule via GHR (Generalized HR) calculus, extending the quaternionic Wirtinger formalism to octonions.

Empirically, the naive (associativity-assuming) chain rule produces >100% relative error on compositions deeper than 3 layers. The correct implementation matches finite-difference gradients to <1e-7.

### Fair baseline comparison

Every experiment compares 4+ algebras with matched total parameter counts:

- **Real**: 8x the hidden units (8 params per unit x 1 real component)
- **Complex**: 4x the hidden units (matching 8 real params per octonion)
- **Quaternion**: 2x the hidden units
- **Octonion**: 1x (baseline width)
- **PHM-8**: Kronecker-factored parameterization (Zhang et al. 2021)
- **R8-Dense**: Unconstrained 8x8 real mixing (the "why not just R8?" control)

Parameter counts are matched to within 1% via binary search over hidden widths. This ensures performance differences reflect algebraic structure, not model capacity.

### Numerical stability

Octonionic operations are well-conditioned individually (condition numbers ~1-10), but error accumulates through deep compositions. Key findings from Phase 4:

- **Stripped chains** (no normalization): All algebras stable to depth 100 at float64. Quaternions uniquely stable to depth 500.
- **Full networks** (with batch norm): Complex networks stable to depth 500. Octonion and quaternion networks face instability from high-dimensional Cholesky whitening in batch normalization.
- **Mitigation**: `StabilizingNorm` (periodic unit-norm re-normalization) extends stable depth by 2.5-5x depending on algebra.

### Experiment infrastructure

The landscape experiment system supports:

- **5 optimizers**: SGD, Adam, L-BFGS, Riemannian Adam (sphere/Stiefel manifolds), Shampoo
- **Hessian analysis**: Full eigenspectrum for small models, stochastic Lanczos for large ones
- **Curvature measurement**: Bill & Cox (2024) random-direction curvature with Li et al. (2018) filter normalization
- **Gradient variance**: Cross-seed variance characterization (20 seeds per configuration)
- **Go/no-go gate**: Automated verdict (GREEN/YELLOW/RED) based on loss ratios, divergence rates, and Hessian spectrum comparison
- **Crash resilience**: Incremental JSON saves after each seed; experiments resume from last checkpoint

## Project structure

```
src/octonion/              # Main package
    __init__.py            # Public API
    _multiplication.py     # Fano plane multiplication engine
    _octonion.py           # Octonion class with operator overloading
    _tower.py              # Real, Complex, Quaternion tower types
    _operations.py         # exp, log, commutator, cross product
    _linear.py             # OctonionLinear nn.Module
    calculus/              # GHR derivatives and autograd
    baselines/             # Comparison networks and training
    landscape/             # Optimization analysis toolkit
    tasks/                 # Synthetic task generators

tests/                     # 739 tests (pytest + hypothesis)
scripts/                   # Experiment runners and analysis
results/                   # Experiment outputs (JSON + plots)
.planning/                 # Research planning and phase tracking
```

## References

- Baez, J. (2002). *The Octonions*. Bulletin of the AMS.
- Parcollet et al. (2019). *Quaternion Neural Networks*. ICLR.
- Nickel & Kiela (2017). *Poincare Embeddings for Learning Hierarchical Representations*. NeurIPS.
- Ganea et al. (2018). *Hyperbolic Neural Networks*. NeurIPS.
- Chen et al. (2018). *Neural Ordinary Differential Equations*. NeurIPS.
- Zhang et al. (2021). *Beyond Fully-Connected Layers with Quaternions: Parameterization of Hypercomplex Multiplications with 1/n Parameters*. ICLR.
- Bill & Cox (2024). *Exploring Quaternion Neural Network Loss Surfaces*. Advances in Applied Clifford Algebras, 34.
- Wu et al. (2020). *Deep Octonion Networks*. Neurocomputing.

## License

This project is licensed under the [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.en.html) (AGPL-3.0).

This means you can use, modify, and distribute this code, but any derivative work — including network services that use it — must also be released under the AGPL-3.0 with full source code. See [LICENSE](LICENSE) for the full text.
