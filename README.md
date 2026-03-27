# Octonionic Neural Networks

A research project investigating whether octonions, the largest normed division algebra, provide measurable advantages over real, complex, and quaternionic alternatives for geometric reasoning in machine learning.

## Background

There are exactly four number systems that support addition, subtraction, multiplication, division, and norm preservation. These are the **normed division algebras**, each constructed by doubling the previous one:

| Algebra             | Dimension | Property lost     |
| ------------------- | --------- | ----------------- |
| Real numbers (R)    | 1         | None              |
| Complex numbers (C) | 2         | Ordering          |
| Quaternions (H)     | 4         | Commutativity     |
| **Octonions (O)**   | **8**     | **Associativity** |

[Hurwitz's theorem (1898)](<https://en.wikipedia.org/wiki/Hurwitz%27s_theorem_(composition_algebras)>) proves that no further normed division algebra exists. Beyond octonions, the Cayley-Dickson construction produces algebras with zero divisors, which destroy invertibility and render the algebra unsuitable for reversible computation.

Octonions appear throughout theoretical physics (string theory, M-theory, exceptional Lie groups). This project investigates their potential as a computational substrate for machine learning.

## Thesis

The full argument is in [`docs/thesis/main.tex`](docs/thesis/main.pdf). The core claims:

1. **Density**: Octonions encode more geometric structure per parameter than smaller algebras. Their automorphism group (G_2, 14-dimensional) is larger than the algebra itself (8-dimensional), a property unique among division algebras.

2. **Reversibility**: Every non-zero octonion has a unique inverse. Networks built from octonionic operations can run backward through their own computations, enabling reasoning about missing or uncertain information.

3. **Intrinsic inductive bias**: Octonionic multiplication automatically preserves norms, respects Moufang identities, and decomposes into 7 overlapping quaternionic subalgebras via the Fano plane. These geometric priors would otherwise need to be learned from data in a structureless representation.

The project validates these claims empirically, bottom-up, with controlled experiments against fair baselines.

## Implementation

A PyTorch-native library implementing the full stack from algebra to experiment infrastructure:

| Module           | Description                                                                                                                                                                                                                                                                                                                             |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Core algebra** | Multiplication via Fano plane structure constants tensor, conjugation, norm, inverse, associator. Cayley-Dickson construction cross-check. R/C/H/O tower types implementing a shared `NormedDivisionAlgebra` interface. `OctonionLinear` layer, batch operations, exp/log.                                                              |
| **`calculus/`**  | GHR Wirtinger derivatives for octonionic backpropagation. Analytic 8x8 Jacobians for all 7 primitives. Parenthesization-aware chain rule handling non-associativity. `torch.autograd.Function` implementations with `create_graph=True` support.                                                                                        |
| **`baselines/`** | Fair comparison networks: R (8x width), C (4x), H (2x), O (1x). Parameter matching via binary search (<1% deviation). Algebra-specific batch normalization with Cholesky whitening. `AlgebraNetwork` skeleton supporting MLP, Conv2D, and Recurrent topologies. Training infrastructure with gradient statistics, AMP, and TensorBoard. |
| **`landscape/`** | Hessian eigenspectrum computation (full + stochastic Lanczos). Bill & Cox (2024) loss surface curvature measurement. Gradient variance collection across seeds. Go/no-go gate evaluation (GREEN/YELLOW/RED). Experiment orchestration with incremental saves.                                                                           |
| **`tasks/`**     | Synthetic task generators with known optima: algebra-native, cross product, sinusoidal, classification.                                                                                                                                                                                                                                 |

**739 tests** covering algebra, calculus, baselines, stability, and landscape (pytest + Hypothesis property-based testing).

## Current status

The project proceeds in phases, each validating a specific claim before building on it:

| Phase                         | What it validates                                           | Status                       |
| ----------------------------- | ----------------------------------------------------------- | ---------------------------- |
| 1. Octonionic Algebra         | Core operations, Moufang identities, norm preservation      | Complete                     |
| 2. GHR Calculus               | Octonionic gradients match finite-difference to <1e-5       | Complete                     |
| 3. Baselines                  | Fair R/C/H/O comparison networks, matched parameters        | Complete                     |
| 4. Numerical Stability        | Precision vs depth, condition numbers, mitigations          | Complete                     |
| **5. Optimization Landscape** | **Go/no-go gate: can octonionic networks train?**           | **Experiments running**      |
| 6. Reversibility              | Algebraic inversion vs RevNet/INN baselines                 | Pending (blocked on Phase 5) |
| 7. Density & Geometry         | Matched-parameter density advantage, geometric detection    | Pending (blocked on Phase 5) |
| 8. G_2 & Hyperbolic           | G_2-equivariant layers, hyperboloid-octonionic hybrid       | Pending (blocked on Phase 5) |
| 9. Associator Analysis        | Non-associativity as signal, Fano subalgebra specialization | Pending (blocked on Phase 5) |
| 10. Predict-and-Fill          | Reversible conjecture, geometry of absence validation       | Pending                      |
| 11. Applied Benchmarks        | Anomaly detection, time series vs LSTM/Transformer          | Pending                      |
| 12. Hyperboloid Projection    | Euclidean-Lorentzian projection distortion characterization | Pending                      |
| 13. Multi-Stream Fusion       | ORE proof-of-concept with heterogeneous data streams        | Pending                      |

Phase 5 is the critical gate. If non-associativity creates pathological loss surfaces, the project pivots to documenting a quantitative negative result. If the gate passes, Phases 6-13 proceed.

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

### Non-associativity and gradients

In associative algebras (R, C, H), the chain rule is straightforward: `d/dx f(g(x)) = f'(g(x)) * g'(x)`. Parenthesization does not affect the result because `(AB)C = A(BC)`.

In octonions, parenthesization changes the answer. The chain rule must track how intermediate results are grouped. This project implements a parenthesization-aware chain rule via GHR (Generalized Hamilton-Real) calculus, extending the quaternionic Wirtinger formalism to octonions.

Empirically, the naive (associativity-assuming) chain rule produces >100% relative error on compositions deeper than 3 layers. The correct implementation matches finite-difference gradients to <1e-7.

### Fair baseline comparison

Every experiment compares 4+ algebras with matched total parameter counts:

| Algebra        | Width multiplier | Role                                             |
| -------------- | ---------------- | ------------------------------------------------ |
| Real (R)       | 8x               | Standard baseline                                |
| Complex (C)    | 4x               | Published quaternionic/complex ML comparison     |
| Quaternion (H) | 2x               | Closest associative relative                     |
| Octonion (O)   | 1x               | Primary subject                                  |
| PHM-8          | Matched          | Kronecker-factored structure (Zhang et al. 2021) |
| R8-Dense       | Matched          | Unconstrained 8x8 real mixing control            |

Parameter counts are matched to within 1% via binary search over hidden widths. This ensures performance differences reflect algebraic structure rather than model capacity. The R8-Dense baseline specifically guards against the possibility that an unconstrained real-valued model with the same parameter budget could achieve equivalent results.

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

- Baez, J. C. (2002). "The Octonions." _Bulletin of the AMS_, 39(2), 145-205.
- Bill, J. & Cox, B. (2024). "Exploring Quaternion Neural Network Loss Surfaces." _Advances in Applied Clifford Algebras_, 34.
- Chen, R. T. Q., et al. (2018). "Neural Ordinary Differential Equations." _NeurIPS_.
- Ganea, O., Becigneul, G., & Hofmann, T. (2018). "Hyperbolic Neural Networks." _NeurIPS_.
- Nickel, M. & Kiela, D. (2017). "Poincare Embeddings for Learning Hierarchical Representations." _NeurIPS_.
- Parcollet, T., et al. (2019). "Quaternion Recurrent Neural Networks." _ICLR_.
- Wu, J., et al. (2020). "Deep Octonion Networks." _Neurocomputing_, 397, 179-191.
- Zhang, A., et al. (2021). "Beyond Fully-Connected Layers with Quaternions: Parameterization of Hypercomplex Multiplications with 1/n Parameters." _ICLR_.

## License

This project is licensed under the [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.en.html) (AGPL-3.0).

This means you can use, modify, and distribute this code, but any derivative work — including network services that use it — must also be released under the AGPL-3.0 with full source code. See [LICENSE](LICENSE) for the full text.
