# Architecture Patterns

**Domain:** Hypercomplex ML research system (octonionic neural networks)
**Researched:** 2026-03-07

## Recommended Architecture

A four-layer architecture with strict upward dependency flow. Each layer depends only on layers below it, never sideways or upward. This mirrors proven patterns from geoopt (manifold algebra -> tensor types -> optimizers -> layers), SpeechBrain's quaternion implementation (q_ops -> q_linear/q_CNN/q_RNN -> q_normalization -> models), and the HyperNets library (layers -> models -> experiments).

```
+-------------------------------------------------------------------+
|                    LAYER 4: EXPERIMENT LAYER                       |
|   Configs, training scripts, evaluation, baselines, analysis      |
+-------------------------------------------------------------------+
|                    LAYER 3: NETWORK LAYER                          |
|   OctonionLinear, OctonionConv, OctonionBatchNorm, G2Equiv,       |
|   ORE architecture, loss functions, optimizers                    |
+-------------------------------------------------------------------+
|                    LAYER 2: AUTOGRAD / CALCULUS LAYER              |
|   GHR-based gradients, custom autograd.Functions,                 |
|   non-associativity-aware backprop, Jacobian computation          |
+-------------------------------------------------------------------+
|                    LAYER 1: ALGEBRA LAYER                          |
|   Octonion type, multiplication (Fano plane), conjugation,        |
|   inversion, norm, associator, G2 actions, subalgebra extraction  |
+-------------------------------------------------------------------+
```

### Why Four Layers (Not Three)

Most hypercomplex implementations (quaternion, complex) fold the calculus into the network layer because associativity makes gradients straightforward -- PyTorch autograd handles the chain rule, and the Hamilton product has a clean matrix representation. Octonions break this. Non-associativity means `(a * b) * c != a * (b * c)`, so the gradient depends on parenthesization order, and the standard chain rule must be extended via GHR calculus. This warrants a dedicated layer.

## Component Boundaries

### Layer 1: Algebra Layer (`src/algebra/`)

| Component | Responsibility | Interface |
|-----------|---------------|-----------|
| `octonion.py` | Core Octonion class wrapping 8-component tensors. Multiplication via Fano plane structure constants, conjugation, norm, inverse. | `Octonion(components: Tensor[..., 8])` with `__mul__`, `conj()`, `norm()`, `inv()` |
| `multiplication.py` | Octonion product implementation using the 7 quaternionic triad lookup from the Fano plane. Handles batched operations. | `octonion_mul(a: Tensor, b: Tensor) -> Tensor` operating on `[..., 8]` tensors |
| `associator.py` | Computes the associator `[a,b,c] = (ab)c - a(bc)` and measures non-associativity magnitude. Essential for diagnostics. | `associator(a, b, c) -> Tensor`, `associativity_defect(a, b, c) -> scalar` |
| `subalgebras.py` | Extracts the 7 quaternionic subalgebras from octonionic representations. Maps between full octonion space and quaternionic subspaces via Fano plane triads. | `extract_subalgebra(x: Tensor, triad_idx: int) -> Tensor[..., 4]` |
| `g2.py` | G2 automorphism group actions on octonions. 14-parameter Lie group that preserves octonion multiplication. Used for equivariant layers. | `g2_action(params: Tensor[14], x: Tensor[..., 8]) -> Tensor[..., 8]` |
| `utils.py` | Fano plane constants, structure constant tensor, index lookups. Pure data, no logic. | `STRUCTURE_CONSTANTS: Tensor[7, 7, 7]`, `FANO_TRIADS: List[Tuple[int,int,int]]` |

**Key design decisions:**

- **Tensor-native, not OOP-heavy.** Store octonions as `[..., 8]` tensors, not as a Python class with 8 scalar fields. This is what every successful hypercomplex library does (SpeechBrain, HyperNets, complexPyTorch). The class is a thin wrapper providing named access and operator overloads, but all operations work on raw tensors internally.
- **Fano plane lookup, not naive expansion.** Octonion multiplication via the 7 quaternionic triads (each a 3-element subset of {1,...,7} from the Fano plane) is cleaner and less error-prone than writing out all 64 sign terms. The structure constants tensor `c[i][j][k]` where `e_i * e_j = c[i][j][k] * e_k` encodes the multiplication table compactly.
- **Batch-first throughout.** All operations accept arbitrary leading batch dimensions. This is non-negotiable for GPU efficiency.

### Layer 2: Autograd / Calculus Layer (`src/calculus/`)

| Component | Responsibility | Interface |
|-----------|---------------|-----------|
| `ghr_derivatives.py` | GHR (Generalized HR) calculus derivatives for octonion-valued functions. Extends the quaternionic GHR calculus to the octonionic setting. Provides left/right derivatives and the novel product rule for non-commutative, non-associative algebras. | `ghr_derivative(f, x, mu) -> Tensor` where `mu` is the direction of differentiation |
| `octonion_autograd.py` | Custom `torch.autograd.Function` subclasses for octonion operations. Forward pass calls algebra layer; backward pass uses GHR calculus. | `OctonionMulFunction.apply(a, b)`, `OctonionLinearFunction.apply(x, W, b)` |
| `jacobian.py` | Full Jacobian computation for octonion-valued functions (8x8 real Jacobian for each octonionic mapping). Needed for analyzing loss landscape properties. | `octonion_jacobian(f, x) -> Tensor[..., 8, 8]` |
| `gradient_utils.py` | Gradient clipping, scaling, and monitoring utilities specific to octonionic parameters. Tracks associativity defect in gradients. | `clip_octonion_grad(params, max_norm)`, `gradient_diagnostics(params) -> dict` |

**Key design decisions:**

- **Custom autograd.Function, not full custom engine.** PyTorch's autograd handles the computational graph; we override only the octonionic operations. SpeechBrain's quaternion implementation proves this works -- they provide both autograd-based (faster development) and custom backprop (lower memory) paths. Start with autograd, optimize later.
- **GHR calculus as the mathematical foundation.** The GHR (Generalized HR) calculus, originally developed for quaternions by Xu et al. (2015) and extended to octonions conceptually in the thesis, provides the correct derivative framework. The key insight: for non-commutative algebras, you need left and right derivatives, and the product rule requires a novel formulation. For non-associative algebras (octonions), you additionally need to track parenthesization order.
- **Parenthesization tracking.** The backward pass must record which association was used in the forward pass. If forward computes `(a*b)*c`, backward cannot assume `a*(b*c)`. This is the fundamental difference from quaternionic backprop and the reason for a dedicated calculus layer.

### Layer 3: Network Layer (`src/networks/`)

| Component | Responsibility | Interface |
|-----------|---------------|-----------|
| `layers/linear.py` | Octonionic linear layer. Weight is an octonion matrix; forward performs octonionic matrix-vector product. 8x parameter efficiency vs real (one octonion weight links 8-dim features). | `OctonionLinear(in_features, out_features)` |
| `layers/conv.py` | Octonionic convolution. Extends the DON (Deep Octonion Network) approach: 8 real-valued channels processed through octonionic filters. | `OctonionConv1d/2d(in_channels, out_channels, kernel_size)` |
| `layers/normalization.py` | Octonionic batch normalization. Whitening in the 8D octonion space (not component-wise), following the DON approach. | `OctonionBatchNorm(num_features)` |
| `layers/activation.py` | Split activations (apply real activation per component) and norm-based activations. Split is simpler and proven; norm-based preserves algebraic structure better. | `OctonionReLU()`, `OctonionModReLU()` |
| `layers/g2_equivariant.py` | G2-equivariant layer. Operations that commute with G2 automorphism actions on octonions. Analogous to e3nn's SO(3)-equivariant layers but for G2. | `G2EquivariantLinear(in_features, out_features)` |
| `layers/pooling.py` | Octonionic pooling. Norm-based pooling (pool by octonion magnitude) and component-wise pooling. | `OctonionAvgPool()`, `OctonionNormPool()` |
| `init.py` | Weight initialization for octonionic parameters. Extends Glorot/He initialization to 8D, following DON's approach. | `octonion_init_(tensor, mode='glorot')` |
| `models/ore.py` | The Octonionic Reasoning Engine from thesis section 4. Assembles layers into the full ORE architecture: encoder -> octonionic processing -> decoder. | `OctonionicReasoningEngine(config)` |
| `models/baselines.py` | Real, complex, and quaternionic baseline models with matched parameter counts. Essential for fair comparison. | `RealBaseline(config)`, `ComplexBaseline(config)`, `QuaternionBaseline(config)` |
| `losses.py` | Loss functions that may operate in octonionic space or project to real-valued losses. | `OctonionMSELoss()`, `ReversibilityLoss()` |
| `optimizers.py` | Octonion-aware optimizer wrappers. May need special handling for parameter updates in non-associative space (similar to geoopt's Riemannian optimizers). | `OctonionSGD(params, lr)`, `OctonionAdam(params, lr)` |

**Key design decisions:**

- **DON architecture as baseline pattern.** Deep Octonion Networks (Wu & Xu, 2020) established the pattern: OctonionConv -> OctonionBN -> ReLU stacked in residual blocks. This is proven to work and should be the starting point before adding thesis-specific innovations.
- **Split activations first, algebraic activations later.** SpeechBrain and every quaternionic implementation uses split activations (apply ReLU to each component independently). It is simple and works. Norm-based activations (ModReLU, etc.) are more principled but harder to get right. Start simple.
- **Baselines as first-class citizens.** The entire project's value depends on fair comparison. Baseline models must share the same architecture skeleton with only the algebra layer swapped. This means abstracting the algebra behind an interface so Real/Complex/Quaternion/Octonion are interchangeable.

### Layer 4: Experiment Layer (`experiments/`)

| Component | Responsibility | Interface |
|-----------|---------------|-----------|
| `configs/` | Hydra/YAML configurations for all experiments. Separate configs for model, data, training, hardware. | `configs/experiment/density_test.yaml` etc. |
| `trainers/base_trainer.py` | Training loop with checkpointing, logging, seed management, and reproducibility controls. | `Trainer(config).fit(model, data)` |
| `trainers/comparison_trainer.py` | Runs matched-parameter experiments across all algebra types (R, C, H, O) and collects comparative metrics. | `ComparisonTrainer(config).run_all_baselines()` |
| `evaluators/` | Evaluation scripts: accuracy, convergence curves, parameter efficiency, associativity defect tracking, reversibility metrics. | `evaluate(model, test_data) -> ResultsDict` |
| `data/synthetic/` | Synthetic dataset generators with known ground truth (geometric structure, hierarchical relations). | `SyntheticGeometricDataset(n_samples, geometry_type)` |
| `data/financial/` | Financial time series data loading and preprocessing for multi-stream fusion experiments. | `FinancialStreamDataset(tickers, window)` |
| `data/text/` | NLP text data loading for the text stream in multi-stream fusion. | `TextStreamDataset(corpus, tokenizer)` |
| `analysis/` | Post-experiment analysis: statistical significance testing, visualization, loss landscape characterization. | `analyze_results(experiment_dir) -> Report` |
| `scripts/` | Entry point scripts: `train.py`, `evaluate.py`, `analyze.py`, `run_baselines.py`. | CLI scripts |

**Key design decisions:**

- **Hydra for configuration management.** The lightning-hydra-template pattern is battle-tested for ML research. Experiment configs as composable YAML files enable reproducibility (every experiment is fully specified by its config) and rapid iteration (override from command line).
- **Comparison as the default mode.** Every experiment should automatically run all baselines. The `ComparisonTrainer` is not optional -- it is the primary training interface. Individual model training is a special case.
- **Synthetic data first, always.** Synthetic datasets with known ground truth must be the first thing built. Real data adds confounds. If octonionic networks cannot demonstrate advantages on synthetic data with planted geometric structure, they won't work on real data.

## Project Directory Structure

```
octonion-computation-substrate/
|
+-- src/
|   +-- algebra/                    # Layer 1: Pure algebra, no ML
|   |   +-- __init__.py
|   |   +-- octonion.py             # Octonion class and core operations
|   |   +-- multiplication.py       # Fano-plane-based multiplication
|   |   +-- associator.py           # Associator and non-associativity metrics
|   |   +-- subalgebras.py          # Quaternionic subalgebra extraction
|   |   +-- g2.py                   # G2 automorphism group actions
|   |   +-- utils.py                # Structure constants, Fano plane data
|   |
|   +-- calculus/                   # Layer 2: Differentiation for non-assoc algebras
|   |   +-- __init__.py
|   |   +-- ghr_derivatives.py      # GHR calculus for octonions
|   |   +-- octonion_autograd.py    # Custom autograd.Function classes
|   |   +-- jacobian.py             # Full Jacobian computation
|   |   +-- gradient_utils.py       # Gradient monitoring and clipping
|   |
|   +-- networks/                   # Layer 3: Neural network components
|   |   +-- __init__.py
|   |   +-- layers/
|   |   |   +-- __init__.py
|   |   |   +-- linear.py           # OctonionLinear
|   |   |   +-- conv.py             # OctonionConv1d/2d
|   |   |   +-- normalization.py    # OctonionBatchNorm
|   |   |   +-- activation.py       # Split and norm-based activations
|   |   |   +-- g2_equivariant.py   # G2-equivariant layers
|   |   |   +-- pooling.py          # Octonionic pooling
|   |   +-- models/
|   |   |   +-- __init__.py
|   |   |   +-- ore.py              # Octonionic Reasoning Engine
|   |   |   +-- baselines.py        # R/C/H baseline models
|   |   +-- init.py                 # Weight initialization
|   |   +-- losses.py               # Loss functions
|   |   +-- optimizers.py           # Octonion-aware optimizers
|   |
|   +-- utils/                      # Cross-cutting utilities
|       +-- __init__.py
|       +-- reproducibility.py      # Seed management, deterministic mode
|       +-- logging.py              # Experiment logging setup
|       +-- visualization.py        # Plotting helpers
|       +-- numerical.py            # Numerical stability checks
|
+-- experiments/                    # Layer 4: Experiment infrastructure
|   +-- configs/
|   |   +-- model/                  # Model configs (octonion, quaternion, etc.)
|   |   +-- data/                   # Dataset configs
|   |   +-- trainer/                # Training configs
|   |   +-- experiment/             # Full experiment configs (compose above)
|   +-- trainers/
|   |   +-- base_trainer.py
|   |   +-- comparison_trainer.py
|   +-- evaluators/
|   |   +-- metrics.py
|   |   +-- statistical_tests.py
|   +-- data/
|   |   +-- synthetic/
|   |   +-- financial/
|   |   +-- text/
|   +-- analysis/
|   |   +-- loss_landscape.py
|   |   +-- convergence.py
|   |   +-- reversibility.py
|   +-- scripts/
|       +-- train.py
|       +-- evaluate.py
|       +-- run_baselines.py
|
+-- tests/                          # Mirrors src/ structure
|   +-- algebra/
|   |   +-- test_octonion.py        # Property-based tests (hypothesis)
|   |   +-- test_multiplication.py
|   |   +-- test_associator.py
|   |   +-- test_subalgebras.py
|   |   +-- test_g2.py
|   +-- calculus/
|   |   +-- test_ghr_derivatives.py
|   |   +-- test_autograd.py        # Gradient checking (finite differences)
|   +-- networks/
|   |   +-- test_layers.py
|   |   +-- test_models.py
|   +-- integration/
|       +-- test_end_to_end.py
|
+-- notebooks/                      # Exploration and visualization
|   +-- 01_algebra_exploration.ipynb
|   +-- 02_gradient_verification.ipynb
|   +-- 03_loss_landscape.ipynb
|
+-- results/                        # Experiment outputs (gitignored except summaries)
|   +-- .gitkeep
|
+-- pyproject.toml
+-- Makefile
+-- Dockerfile                      # ROCm PyTorch container
```

## Thesis ORE Architecture Mapping

The thesis (section 4) describes the Octonionic Reasoning Engine as a pipeline:

```
Input Stream(s) --> Encoder --> Octonionic Processing --> Decoder --> Output
                       |              |                      |
                       v              v                      v
                  Real -> O       G2-equiv layers       O -> Real
                  embedding      Reversible blocks      projection
                                 Subalgebra routing
```

This maps to implementation components as follows:

| Thesis Concept | Implementation Component | Layer |
|---------------|-------------------------|-------|
| Input encoding (real to octonionic) | `networks/models/ore.py` encoder section | 3 |
| Octonionic linear transform | `networks/layers/linear.py` | 3 |
| G2-equivariant processing | `networks/layers/g2_equivariant.py` | 3 |
| Subalgebra routing / Fano plane decomposition | `algebra/subalgebras.py` + routing logic in model | 1 + 3 |
| Reversible octonionic blocks | `networks/models/ore.py` using `algebra/octonion.py` inverse | 1 + 3 |
| Backward reasoning (invertibility) | `algebra/octonion.py` inverse + `calculus/` backward pass | 1 + 2 |
| Output decoding (octonionic to real) | `networks/models/ore.py` decoder section | 3 |
| GHR-based training | `calculus/ghr_derivatives.py` + `calculus/octonion_autograd.py` | 2 |
| G2 automorphism group | `algebra/g2.py` | 1 |
| Associator as regularizer | `networks/losses.py` using `algebra/associator.py` | 1 + 3 |

## Data Flow: Training vs Inference

### Training Data Flow

```
Raw Data (real-valued)
    |
    v
[Preprocessing] -- experiments/data/
    |
    v
Tensor [batch, features] (real-valued)
    |
    v
[Encoder] -- Learned embedding: R^n -> O^m
    |         Reshape features into groups of 8, apply learned octonionic embedding
    v
Octonion Tensor [batch, m, 8] -- internal representation
    |
    v
[OctonionLinear / OctonionConv] -- Forward pass through octonionic layers
    |   Uses algebra/multiplication.py for octonion products
    |   Records parenthesization order in autograd graph
    v
[OctonionBatchNorm] -- 8D whitening
    |
    v
[Activation] -- Split ReLU (per-component) or ModReLU (norm-based)
    |
    v
... (repeat for depth)
    |
    v
[Decoder] -- O^m -> R^k projection
    |         Extract scalar/norm or learned real projection
    v
Real-valued output [batch, k]
    |
    v
[Loss Function] -- Real-valued loss (cross-entropy, MSE)
    |               + optional associativity regularization
    v
[GHR Backward Pass] -- calculus/octonion_autograd.py
    |   Custom backward through octonionic operations
    |   Left/right derivatives, parenthesization-aware chain rule
    |   Produces real-valued gradients for each of 8 components
    v
[Optimizer Step] -- Update octonionic parameters
    |   Standard optimizer on the 8 real components
    |   Optional: manifold-aware updates (octonion-norm preserving)
    v
Updated Parameters
```

### Inference Data Flow

```
Raw Data
    |
    v
[Same preprocessing and encoding as training]
    |
    v
[Forward pass only -- no autograd graph needed]
    |
    v
Real-valued output
    |
    v
[OPTIONAL: Backward Reasoning]
    |   This is the thesis's key differentiator:
    |   Given an output, invert octonionic transformations
    |   to reason about what inputs could have produced it.
    |   Uses algebra/octonion.py inverse operations.
    |   This is algebraic inversion, NOT gradient-based.
    v
Inferred input / explanation
```

**Critical distinction:** Training backward (GHR calculus, gradient-based) is fundamentally different from inference backward (algebraic inversion, reasoning-based). These use different code paths:

- Training backward: `calculus/ghr_derivatives.py` -- computes parameter gradients
- Inference backward: `algebra/octonion.py` inverse operations -- inverts the learned transformation

This distinction must be architecturally clear because conflating them is a major source of confusion.

## Architectural Patterns to Follow

### Pattern 1: Algebra-Agnostic Network Skeleton

**What:** All network layers accept an `algebra` parameter that determines whether they perform real, complex, quaternion, or octonion operations. The layer structure (input/output dims, residual connections, normalization placement) is identical across all algebras.

**When:** Always. This is the foundation of fair comparison.

**Example:**
```python
class HypercomplexLinear(nn.Module):
    """Linear layer parameterized by algebra type."""

    def __init__(self, in_features: int, out_features: int, algebra: str = "octonion"):
        super().__init__()
        self.division_dim = {"real": 1, "complex": 2, "quaternion": 4, "octonion": 8}[algebra]
        assert in_features % self.division_dim == 0
        assert out_features % self.division_dim == 0

        # Weight has (out // div) x (in // div) entries, each a div-dimensional number
        self.weight = nn.Parameter(
            torch.empty(out_features // self.division_dim,
                        in_features // self.division_dim,
                        self.division_dim)
        )
        self.mul_fn = get_multiplication_fn(algebra)  # Returns algebra-specific product
        self.init_fn = get_init_fn(algebra)
        self.init_fn(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features] reshaped to [..., in_features // div, div]
        # Matrix-vector product using algebra-specific multiplication
        ...
```

**Why:** Parcollet's SpeechBrain achieves this for quaternions. The DON paper does this for octonions. Formalizing it as a pattern ensures every experiment is a true apples-to-apples comparison.

### Pattern 2: Diagnostic Hooks Throughout

**What:** Every octonionic layer emits diagnostic metrics during training: associativity defect of activations, gradient norms per component, subalgebra activation patterns, numerical condition numbers.

**When:** During development and characterization experiments. Can be disabled for speed in production runs.

**Example:**
```python
class OctonionLinear(HypercomplexLinear):
    def forward(self, x):
        out = super().forward(x)
        if self.diagnostics_enabled:
            self._log_associativity_defect(x, self.weight, out)
            self._log_component_magnitudes(out)
            self._log_subalgebra_activations(out)
        return out
```

**Why:** The thesis's central risk is that non-associativity creates pathological optimization landscapes. You cannot diagnose what you do not measure. Diagnostics must be built in from day one, not bolted on later.

### Pattern 3: Reproducibility as Infrastructure

**What:** Every experiment is fully determined by a config file + a git commit hash. Seeds, hardware config, library versions, and data preprocessing are all captured.

**When:** Always.

**Example:**
```python
# In every experiment config:
seed: 42
deterministic: true
benchmark: false  # Disable cuDNN benchmarking for determinism

# At experiment start:
def setup_reproducibility(config):
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.use_deterministic_algorithms(True)
    # Log environment
    log_environment(torch.__version__, rocm_version(), git_hash())
```

**Why:** ML research reproducibility crisis is well-documented. For a thesis validation project, every result must be reproducible. The PyTorch reproducibility guide notes that ROCm may have different nondeterminism sources than CUDA, so explicit deterministic mode is essential.

### Pattern 4: Staged Validation

**What:** Each component is validated in isolation before integration. The algebra layer has property-based tests. The calculus layer has finite-difference gradient checks. The network layer has convergence tests on toy problems. Only validated components are composed.

**When:** At every layer boundary.

**Example:**
```python
# Property-based test for algebra layer
@given(octonions(), octonions())
def test_multiplication_norm_preserving(a, b):
    """||a * b|| = ||a|| * ||b|| (norm is multiplicative for octonions)."""
    result = octonion_mul(a, b)
    assert torch.allclose(norm(result), norm(a) * norm(b), atol=1e-6)

# Gradient check for calculus layer
def test_octonion_linear_gradient():
    """Compare GHR gradient against finite differences."""
    layer = OctonionLinear(8, 8)
    x = torch.randn(1, 8, requires_grad=True)
    assert torch.autograd.gradcheck(layer, x, eps=1e-6)
```

**Why:** Bottom-up validation is the project methodology. An integrated system that fails gives no information about which component is broken. Isolated validation catches bugs early and builds confidence before composition.

### Pattern 5: Tensor Shape Convention

**What:** Octonionic tensors always have shape `[batch, ..., 8]` where the last dimension holds the 8 octonion components (1 real + 7 imaginary). All operations preserve this convention.

**When:** Everywhere in the algebra and calculus layers. Network layers may reshape for convolution but must restore the convention.

**Why:** SpeechBrain uses `[..., 4]` for quaternions with the 4 components concatenated. ComplexPyTorch uses separate real/imaginary tensors. The single-tensor `[..., 8]` approach is cleaner for octonions because:
- 8 components make separate tensors unwieldy
- A contiguous last dimension enables efficient GPU operations
- It naturally supports batched operations
- Structure constants can be applied with einsum

**Ordering convention:** `[e0, e1, e2, e3, e4, e5, e6, e7]` where `e0` is the real unit and `e1..e7` are the imaginary units following Baez (2002) / Fano plane standard ordering.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Naive Real-Valued Expansion

**What:** Treating octonions as "just 8 real numbers" and using standard real-valued layers with 8x channels.

**Why bad:** Destroys all algebraic structure. The cross-component relationships encoded by octonion multiplication are exactly what the thesis claims provides advantages. Without them, you have a standard 8-channel real network with no octonionic properties.

**Instead:** Use proper octonionic multiplication in every layer. The weight matrix performs octonionic matrix-vector products, not real-valued ones.

### Anti-Pattern 2: Component-Wise Batch Normalization

**What:** Applying standard batch normalization independently to each of the 8 octonion components.

**Why bad:** Breaks the algebraic relationships between components. The 8 components of an octonion are not independent -- they are coupled by the multiplication structure. Component-wise normalization decorrelates them incorrectly.

**Instead:** Use octonionic batch normalization (8D whitening) as described in the DON paper. This normalizes the octonion as a whole, preserving inter-component relationships.

### Anti-Pattern 3: Ignoring Parenthesization in Backward Pass

**What:** Using standard PyTorch autograd without custom backward functions for octonionic operations, assuming the chain rule "just works."

**Why bad:** For associative algebras (R, C, H), the chain rule is parenthesization-independent. For octonions, `d/dx[(a*b)*c]` differs from `d/dx[a*(b*c)]`. Standard autograd does not know which parenthesization was used and will silently compute incorrect gradients.

**Instead:** Implement custom `torch.autograd.Function` classes that record the parenthesization used in the forward pass and apply the matching GHR derivative rule in the backward pass.

### Anti-Pattern 4: Premature GPU Kernel Optimization

**What:** Writing custom HIP/ROCm kernels for octonionic operations before the algebra is validated.

**Why bad:** Custom GPU kernels are hard to debug, hard to modify, and brittle to changes in the algebra implementation. During research, you need flexibility and correctness, not raw speed. The 7900 XTX has 24GB VRAM -- pure PyTorch tensor operations on `[..., 8]` tensors will be fast enough for research-scale experiments.

**Instead:** Use pure PyTorch (einsum, matmul, tensor indexing) for all algebra operations. Profile first. Only write custom kernels if profiling shows a specific bottleneck that limits experiment throughput. The Hugging Face ROCm kernel tools make this easier when the time comes.

### Anti-Pattern 5: Monolithic ORE Model

**What:** Building the full Octonionic Reasoning Engine as a single module before validating individual components.

**Why bad:** If the integrated system fails (does not train, does not outperform baselines), you cannot determine whether the failure is in the algebra, the gradients, the layer design, the training procedure, or the data. Debugging a monolithic system is exponentially harder than debugging individual components.

**Instead:** Validate algebra -> validate gradients -> validate single layer convergence -> validate small network convergence -> assemble ORE. Each step must pass before proceeding.

## Build Order (Dependency-Driven)

The build order follows strict dependency chains. Each item depends only on items above it.

### Phase 1: Algebra Foundation
```
1a. utils.py (Fano plane constants, structure constants)
     |
1b. multiplication.py (depends on utils.py)
     |
1c. octonion.py (depends on multiplication.py -- wraps it with class interface)
     |
1d. associator.py (depends on multiplication.py)
     |
1e. subalgebras.py (depends on utils.py, multiplication.py)
```
**Validation gate:** Property-based tests pass for all algebraic identities:
- Norm multiplicativity: `||a*b|| = ||a||*||b||`
- Alternative law: `a*(a*b) = (a*a)*b` and `(a*b)*b = a*(b*b)`
- Moufang identities
- Inverse: `a * a^{-1} = 1`
- Conjugation: `conj(a*b) = conj(b) * conj(a)`

### Phase 2: Calculus Layer
```
2a. ghr_derivatives.py (depends on algebra layer)
     |
2b. octonion_autograd.py (depends on ghr_derivatives.py)
     |
2c. jacobian.py (depends on octonion_autograd.py)
     |
2d. gradient_utils.py (depends on octonion_autograd.py)
```
**Validation gate:** Finite-difference gradient checks pass for all custom autograd functions. GHR derivatives agree with numerical derivatives to within floating-point tolerance.

### Phase 3: Basic Network Components
```
3a. init.py (depends on algebra layer)
     |
3b. layers/linear.py (depends on octonion_autograd.py, init.py)
     |
3c. layers/activation.py (no algebra dependency -- split activations)
     |
3d. layers/normalization.py (depends on algebra layer for 8D stats)
     |
3e. layers/conv.py (depends on octonion_autograd.py, init.py)
     |
3f. layers/pooling.py (depends on algebra layer for norm)
```
**Validation gate:** Single-layer convergence tests. An OctonionLinear layer can fit a random mapping. OctonionBatchNorm produces zero-mean unit-variance outputs in octonion space.

### Phase 4: G2 and Advanced Components
```
4a. algebra/g2.py (depends on algebra layer -- complex, needs Lie algebra exponentiation)
     |
4b. layers/g2_equivariant.py (depends on g2.py, linear.py)
     |
4c. losses.py (depends on algebra layer, layers)
     |
4d. optimizers.py (depends on gradient_utils.py)
```
**Validation gate:** G2 actions preserve octonion multiplication. G2-equivariant layers produce G2-equivariant outputs. Associativity regularization reduces associator magnitude.

### Phase 5: Models and Baselines
```
5a. models/baselines.py (depends on layers -- uses algebra-agnostic skeleton)
     |
5b. models/ore.py (depends on all layers, g2_equivariant, losses)
```
**Validation gate:** All baselines train and converge on a toy task. ORE model trains without NaN/Inf.

### Phase 6: Experiment Infrastructure
```
6a. configs/ (no code dependency)
     |
6b. trainers/ (depends on models, losses, optimizers)
     |
6c. data/ (independent)
     |
6d. evaluators/ (depends on models)
     |
6e. analysis/ (depends on evaluators)
```
**Validation gate:** End-to-end experiment runs: config -> training -> evaluation -> analysis produces reproducible results.

## ROCm/GPU Considerations

### Current Approach: Pure PyTorch on ROCm

PyTorch's ROCm support reuses the CUDA interface (`torch.cuda.*` calls work as-is). For research-scale work on a single 7900 XTX:

- **No custom kernels needed initially.** Octonionic operations decompose into standard tensor operations (matmul, einsum, indexing). PyTorch's existing kernels handle these.
- **Memory management.** 24GB VRAM is generous for research-scale experiments. An octonionic network with 8x fewer semantic parameters than real needs 8x the components per parameter, so memory usage is roughly equivalent to a real network of the same representational capacity.
- **Deterministic mode.** Set `torch.use_deterministic_algorithms(True)`. ROCm may lack deterministic implementations for some operations -- test this early and document which operations are nondeterministic.

### When to Write Custom Kernels

Consider custom HIP kernels only if profiling reveals:

1. **Octonionic multiplication is the bottleneck.** The Fano-plane-based multiplication involves 7 quaternionic products, each a structured combination of component multiplications and additions. If this is >50% of training time, a fused kernel helps.
2. **Memory bandwidth limits batched operations.** If launching many small kernels for the 8 components creates overhead, a single fused kernel processing all 8 components is better.
3. **G2 equivariant layers have complex index patterns.** The 14-parameter G2 action involves non-trivial index computations that may not vectorize well in pure PyTorch.

### Custom Kernel Strategy (When Needed)

Use the Hugging Face ROCm kernel tooling for JIT compilation of HIP kernels packaged as PyTorch extensions. Write HIP (which is syntactically near-identical to CUDA) and let HIPIFY handle any remaining CUDA-isms. Target the gfx1100 architecture (RDNA3, 7900 XTX).

Key kernels to consider (in priority order):
1. Fused octonionic multiplication: one kernel for the full 8-component product
2. Fused octonionic batch normalization: 8D whitening in a single pass
3. Fused G2 action: apply 14-parameter transformation without intermediate tensors

### TF32 Note

TF32 (TensorFloat-32) is not supported on ROCm. This is irrelevant for this project -- research correctness requires FP32 or FP64 precision. Do not use reduced precision for validation experiments.

## Scalability Considerations

| Concern | Research Scale (now) | Publication Scale | Production Scale |
|---------|---------------------|-------------------|------------------|
| Batch size | 32-128, single GPU | 128-512, single GPU | Multi-GPU (out of scope) |
| Model size | <10M params | 10-100M params | Out of scope |
| Dataset size | Synthetic (thousands) | Real data (millions) | Out of scope |
| Training time | Minutes to hours | Hours to days | Out of scope |
| Precision | FP32/FP64 | FP32 | FP16/BF16 (out of scope) |
| Custom kernels | Not needed | Maybe for bottlenecks | Required |

This project targets research and publication scale only. Production optimization is explicitly out of scope per PROJECT.md.

## Sources

- [Orkis Pytorch-Quaternion-Neural-Networks](https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks) -- Quaternion network code organization pattern (core_qnn + experiments)
- [HyperNets (eleGAN23)](https://github.com/eleGAN23/HyperNets) -- Hypercomplex network library structure (layers + tutorials + experiments)
- [SpeechBrain Quaternion Networks](https://speechbrain.readthedocs.io/en/v1.0.2/tutorials/nn/complex-and-quaternion-neural-networks.html) -- Mature quaternion implementation architecture (ops -> layers -> models, dual backprop modes)
- [geoopt](https://github.com/geoopt/geoopt) -- Manifold-aware PyTorch library structure (manifolds -> tensors -> optimizers -> layers)
- [e3nn](https://e3nn.org/) -- Equivariant neural network framework (irreps -> tensor products -> layers -> models)
- [Lightning-Hydra Template](https://github.com/ashleve/lightning-hydra-template) -- ML research project organization (configs + src + experiments)
- [Deep Octonion Networks (Wu & Xu, 2020)](https://arxiv.org/abs/1903.08478) -- DON architecture: OctonionConv -> OctonionBN -> ReLU in residual blocks
- [GHR Calculus (Xu et al., 2015)](https://arxiv.org/abs/1409.8168) -- Gradient framework for non-commutative algebras
- [PyTorch ROCm/HIP docs](https://docs.pytorch.org/docs/stable/notes/hip.html) -- ROCm integration details and limitations
- [Hugging Face ROCm kernel tools](https://huggingface.co/blog/build-rocm-kernels) -- JIT/AOT compilation for custom AMD GPU kernels
- [Hypercomplex neural networks survey (PMC, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12513225/) -- Comprehensive survey of quaternion, octonion, and beyond
- [Comprehensive quaternion DNN analysis (Springer, 2024)](https://link.springer.com/article/10.1007/s11831-024-10216-1) -- Architecture patterns across quaternion network types
- [Facebook Poincare Embeddings](https://github.com/facebookresearch/poincare-embeddings) -- Hyperbolic embedding implementation reference
- [UvA DL Notebooks: Research Projects](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide2/Research_Projects.html) -- PyTorch research project structure best practices
- [Octonion multiplication / Fano plane](https://en.wikipedia.org/wiki/Octonion) -- Standard reference for octonionic algebra structure
- [Baez (2002) Octonions](https://math.ucr.edu/home/baez/octonions/node4.html) -- Canonical octonion algebra reference
