# Octonion

A research project investigating octonions as a computational substrate for machine learning. The primary result is a **self-organizing octonionic trie** that classifies data without gradient descent, using only algebraic operations.

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

The project produced two companion theses:

- [`docs/thesis/oct-neural-nets.pdf`](docs/thesis/oct-neural-nets.pdf) -- Gradient-trained octonionic networks: algebra, calculus, fair baselines, and optimization landscape analysis.
- [`docs/thesis/oct-trie.pdf`](docs/thesis/oct-trie.pdf) -- Self-organizing octonionic tries: a zero-gradient classification architecture using associator-based novelty detection and Fano plane subalgebra routing.

### Core claims

1. **Density**: Octonions encode more geometric structure per parameter than smaller algebras. Their automorphism group (G_2, 14-dimensional) is larger than the algebra itself (8-dimensional), a property unique among division algebras.

2. **Reversibility**: Every non-zero octonion has a unique inverse. The trie exploits this for consistency verification via algebraic inversion, exact to machine precision at depth 200+.

3. **Algebraic economy**: A single 8-dimensional non-associative algebra provides five functionally independent components -- routing, novelty detection, content update, consistency check, and health monitoring -- through two properties of its multiplication structure (alternativity and Fano plane geometry).

## The Octonionic Trie

The primary research contribution. A self-organizing hierarchical memory where **octonionic algebra replaces gradient-based learning entirely**:

| Component         | Conventional systems       | Octonionic trie                       |
| ----------------- | -------------------------- | ------------------------------------- |
| Routing           | Learned attention / gating | Subalgebra decomposition (Fano plane) |
| Novelty detection | Engineered thresholds      | Associator norm                       |
| Content update    | Learned write / Hebbian    | Octonionic composition                |
| Consistency check | None                       | Algebraic inversion                   |
| Health monitoring | Separate metrics           | Associator norms + composition error  |

### Key results

| Experiment                      | Result                                            | What it shows                    |
| ------------------------------- | ------------------------------------------------- | -------------------------------- |
| Subalgebra routing              | 90%+ within-class consistency                     | Routing is discriminative        |
| Associator as novelty           | 5x spike ratio, zero false negatives              | Novelty detection works          |
| Composition depth               | Exact inversion to machine precision at depth 200 | Local inversion is reliable      |
| 7-category stability-plasticity | 97.7% accuracy, 0.0% forgetting                   | Trie learns without forgetting   |
| **MNIST (CNN encoder)**         | **95.2%** (vs. 98.2% kNN on same features)        | Competitive without gradients    |
| MNIST (PCA-only)                | 76.5% (vs. 88.8% kNN)                             | Works without any neural network |

### Theoretical results

- **Egan's theorem**: Mean associator norm for uniform unit octonion triples is 147456/(42875pi) ~ 1.095, establishing the baseline against which all threshold choices are measured. Validated by Monte Carlo with 95% CI.
- **G_2 invariance**: Associator norms are invariant under the automorphism group of the octonions, constraining the functional form of adaptive threshold policies.
- **Element proximity bound**: Elements clustered within epsilon of a common point have associator norms O(epsilon^2), from alternativity alone -- no subalgebra alignment needed. This provides quadratic within-class suppression.
- **Subalgebra proximity bound**: Elements spread across a quaternionic subalgebra have associator norms O(epsilon). The Fano plane geometry provides between-class separation bounded away from zero.

Two mechanisms -- quadratic floor suppression from alternativity, geometric ceiling elevation from the Fano plane -- emerge from the same algebraic object.

### Limitations

- **CNN encoder dependency**: The 95.2% result uses a separately-trained CNN encoder. The PCA-only result (76.5%) shows the trie works without it, but at lower accuracy.
- **Quaternionic trie ablation not done**: The strongest ablation -- a quaternionic trie lacking the associator signal -- has not been run.
- **Scalability open**: 26,042 nodes for 60K MNIST samples. Whether consolidation bounds growth in practice is an open question.

## Foundation: Octonionic Algebra and Calculus

The trie is built on a validated algebraic foundation (Phases 1-5, complete):

| Phase                     | What it validates                                                           | Status                 |
| ------------------------- | --------------------------------------------------------------------------- | ---------------------- |
| 1. Octonionic Algebra     | Moufang identities, norm preservation, alternativity, associator properties | Complete               |
| 2. GHR Calculus           | Octonionic gradients match finite-difference to <1e-5                       | Complete               |
| 3. Baselines              | Fair R/C/H/O comparison networks with matched parameters (<1% deviation)    | Complete               |
| 4. Numerical Stability    | Precision vs depth, condition numbers, StabilizingNorm mitigation           | Complete               |
| 5. Optimization Landscape | Go/no-go gate for gradient-trained octonionic networks                      | Complete (gate passed) |

### Fair baseline comparison

Every gradient-trained experiment compares 4+ algebras with matched total parameter counts:

| Algebra        | Width multiplier | Role                                             |
| -------------- | ---------------- | ------------------------------------------------ |
| Real (R)       | 8x               | Standard baseline                                |
| Complex (C)    | 4x               | Published quaternionic/complex ML comparison     |
| Quaternion (H) | 2x               | Closest associative relative                     |
| Octonion (O)   | 1x               | Primary subject                                  |
| PHM-8          | Matched          | Kronecker-factored structure (Zhang et al. 2021) |
| R8-Dense       | Matched          | Unconstrained 8x8 real mixing control            |

## Implementation

A PyTorch-native library implementing the full stack from algebra to experiment infrastructure:

| Module           | Description                                                                                                                                                                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Core algebra** | Multiplication via Fano plane structure constants, conjugation, norm, inverse, associator. Cayley-Dickson cross-check. R/C/H/O tower types. Batch operations, exp/log.                                                                     |
| **`trie.py`**    | Self-organizing octonionic trie with 7 pluggable threshold policies (Global, EMA, MeanStd, Depth, AlgebraicPurity, MetaTrie, Hybrid). Subalgebra routing, associator-based novelty detection, rumination consistency check, consolidation. |
| **`calculus/`**  | GHR Wirtinger derivatives for octonionic backpropagation. Analytic 8x8 Jacobians. Parenthesization-aware chain rule. `torch.autograd.Function` implementations.                                                                            |
| **`baselines/`** | Fair comparison networks: parameter-matched R/C/H/O. Algebra-specific batch norm with Cholesky whitening. MLP, Conv2D, Recurrent topologies. Training with gradient statistics, AMP, TensorBoard.                                          |
| **`landscape/`** | Hessian eigenspectrum, Bill & Cox curvature, gradient variance, go/no-go gate.                                                                                                                                                             |
| **`tasks/`**     | Synthetic task generators with known optima.                                                                                                                                                                                               |

**839 tests** covering algebra, calculus, baselines, stability, trie, threshold policies, proximity bounds, and thesis theorem validation (pytest + Hypothesis property-based testing).

## Development setup

All computation runs in a Docker container with ROCm PyTorch (AMD GPU). The host machine is used only for git and file editing.

```bash
# Build the container (first time)
docker compose build

# Install dependencies
docker compose run --rm dev uv sync

# Run tests
make test

# List available experiment scripts
make list-scripts

# Run a specific experiment (example)
make run-run_trie_mnist
make run-run_landscape ARGS="--smoke"
```

**Requirements**: Docker, an AMD GPU with ROCm support (developed on RX 7900 XTX, 24GB VRAM). NVIDIA GPUs would require swapping the base container image.

## Project structure

```
src/octonion/              # Main package
    _octonion.py           # Octonion class, associator
    _multiplication.py     # Fano plane multiplication engine
    _fano.py               # Fano plane structure and automorphisms
    _tower.py              # Real, Complex, Quaternion tower types
    _operations.py         # exp, log, commutator, cross product
    _linear.py             # OctonionLinear nn.Module
    _random.py             # Random octonion generation (uniform on S^7)
    trie.py                # Octonionic trie + threshold policies
    calculus/              # GHR derivatives and autograd
    baselines/             # Comparison networks and training
    landscape/             # Optimization analysis toolkit
    tasks/                 # Synthetic task generators

tests/                     # 839 tests (pytest + Hypothesis)
scripts/                   # Experiment runners and analysis
docs/thesis/               # Two thesis documents + build system
.planning/                 # Research planning and phase tracking
```

## References

- Baez, J. C. (2002). "The Octonions." _Bulletin of the AMS_, 39(2), 145-205.
- Bill, J. & Cox, B. (2024). "Exploring Quaternion Neural Network Loss Surfaces." _Advances in Applied Clifford Algebras_, 34.
- Egan, G. (2024). "Mean associator norm on S^7." Personal communication.
- Gaudet, C. & Maida, A. (2018). "Deep Quaternion Networks." _IJCNN_.
- Parcollet, T., et al. (2019). "Quaternion Recurrent Neural Networks." _ICLR_.
- Trabelsi, C., et al. (2018). "Deep Complex Networks." _ICLR_.
- Wu, J., et al. (2020). "Deep Octonion Networks." _Neurocomputing_, 397, 179-191.
- Zhang, A., et al. (2021). "Beyond Fully-Connected Layers with Quaternions: Parameterization of Hypercomplex Multiplications with 1/n Parameters." _ICLR_.

## License

This project is licensed under the [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.en.html) (AGPL-3.0).

This means you can use, modify, and distribute this code, but any derivative work -- including network services that use it -- must also be released under the AGPL-3.0 with full source code. See [LICENSE](LICENSE) for the full text.
