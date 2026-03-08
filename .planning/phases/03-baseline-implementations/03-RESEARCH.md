# Phase 3: Baseline Implementations - Research

**Researched:** 2026-03-08
**Domain:** Hypercomplex neural network baselines (R, C, H, O), training infrastructure, benchmark reproduction
**Confidence:** MEDIUM-HIGH

## Summary

Phase 3 builds the experimental infrastructure that every downstream phase depends on: algebra-agnostic network skeletons, per-algebra linear/conv/recurrent layers, a full training utility with Optuna-based hyperparameter search, and a comparison runner with statistical testing. The core technical challenge is parameter-matched fair comparison across four algebras (R, C, H, O) with structurally identical architectures differing only in the algebra module.

The hypercomplex ML ecosystem provides strong reference implementations but no reusable libraries. SpeechBrain's quaternion layers (q_ops.py) and Parcollet's Pytorch-Quaternion-Neural-Networks provide authoritative quaternion initialization (polar form with Glorot/He variance matching) and batch normalization (Cholesky-based whitening). Trabelsi et al.'s Deep Complex Networks provides the canonical complex initialization (Rayleigh magnitude + uniform phase) and complex batch normalization (2x2 covariance whitening). These are well-documented algorithms that should be reimplemented from scratch following the paper descriptions, not vendored as dependencies, consistent with project constraints.

**Primary recommendation:** Build algebra-agnostic `AlgebraLinear` base class following the existing `OctonionLinear` pattern (raw tensor [..., dim] operations via `NormedDivisionAlgebra` interface). Reproduce Gaudet & Maida's CIFAR-10/100 results for quaternion verification (5.44%/26.01% error) and Trabelsi et al.'s CIFAR-100 result for complex verification (26.36% error). Use CIFAR-10/100 as the benchmark for all four algebras since both C and H papers report on it with parameter-matched comparisons.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Configurable skeleton via `AlgebraNetwork(algebra, topology, depth=N)` pattern
- Three topology types supported at launch: MLP, Conv (1D + 2D), Recurrent
- Depth is a config parameter (variable layer count for downstream phases to sweep)
- Algebra-aware normalization layers (e.g., quaternionic batch norm from Gaudet & Maida 2018)
- Activation functions: both split activations (ReLU/GELU per component) AND algebra-aware activations (apply to norm, keep direction) supported as configurable option per experiment
- Recurrent layers: algebra-specific cells (QuaternionLSTM, ComplexGRU, etc.) following published designs
- Per-algebra weight initialization following literature: RealLinear = Kaiming/He, ComplexLinear = Trabelsi et al., QuaternionLinear = Parcollet et al., OctonionLinear = unit-norm (existing)
- Forward interface: output only. Use PyTorch `register_forward_hook()` for intermediate activation access
- Input projection: learned real-to-algebra linear embedding (nn.Linear(input_dim, hidden*algebra_dim) then reshape)
- Output projection: four strategies supported as configurable option
- Built-in parameter counting: `param_report()` returns per-layer breakdown
- FLOP counting: reported for transparency but not matched across algebras
- Config driven by Python dataclasses
- Published benchmark reproduction for C and H baselines
- Cross-reference authors' open-source code for implementation correctness
- Reproduction criterion: within 1 standard deviation of published mean
- Parameter matching: all trainable params, 1% tolerance, binary search for hidden width
- Automated pytest test verifying all 4 algebra models match within 1%
- Full training utility: `train_model(model, data, config)` with logging, checkpointing, metric tracking
- Per-algebra full tuning with Adam-lock option for controlled comparison
- Hyperparameter search: Optuna integration for Bayesian optimization
- Seed protocol: seed-controlled but not CUDA deterministic
- Gradient statistics: always logged every epoch (norm, variance, max)
- Wall-clock timing: per-epoch and total
- Early stopping: configurable patience
- LR warmup: configurable warmup steps/epochs
- LR schedulers: configurable per-algebra
- Mixed precision: optional AMP via config flag
- Multi-GPU: DDP-ready from the start
- Checkpointing: full state for training resumption
- Graceful shutdown: SIGINT handler saves checkpoint
- VRAM monitoring: peak usage tracked and logged
- Logging: TensorBoard for experiment visualization
- Auto-generated plots: matplotlib/seaborn convergence curves, accuracy bars with error bars, param count tables
- Comparison runner: `run_comparison(task, algebras=['R','C','H','O'], seeds=10)`
- Sequential execution (one model at a time, single GPU)
- Built-in statistical significance testing: paired t-test, Wilcoxon, correction for multiple testing, confidence intervals, effect sizes
- Structured directory: `experiments/{benchmark}/{algebra}/{seed}/`
- Auto-manifest: `experiments/manifest.json`
- Model summary via torchinfo package

### Claude's Discretion
- Package organization (octonion.baselines subpackage vs separate baselines package)
- Algebra module plug-in pattern (base class vs functional dispatch)
- Normalization layer param handling for matching
- Conv layer param matching strategy
- Specific published benchmarks for C and H reproduction
- Internal module organization
- Output projection default strategy
- Exact Optuna search space configuration
- Checkpoint frequency (every N epochs)

### Deferred Ideas (OUT OF SCOPE)
- R8-dense-mixing baseline -- Phase 7
- K-fold cross-validation -- add if needed for specific experiments
- Parallel comparison runs -- keep sequential for now, single GPU
- Custom CUDA/ROCm kernels -- per project constraints, premature optimization
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| BASE-01 | Real-valued baseline neural network with structurally identical architecture to octonionic network, using 8x units to match total real parameter count | Architecture skeleton with algebra-agnostic design; binary search width matching; Kaiming/He initialization; parameter counting verified by automated pytest |
| BASE-02 | Complex-valued baseline with 4x units matching total real parameter count, verified to reproduce published results on known benchmark | Complex initialization (Trabelsi Rayleigh+phase), complex batch norm (2x2 covariance whitening); CIFAR-100 benchmark target: 26.36% error rate (Trabelsi ICLR 2018) |
| BASE-03 | Quaternionic baseline with 2x units matching total real parameter count, verified to reproduce published results on known benchmark | Quaternion initialization (polar form Glorot/He), quaternion batch norm (Cholesky whitening); CIFAR-10/100 benchmark target: 5.44%/26.01% error rate (Gaudet & Maida 2018) |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.9.1 (container) | Neural network framework | Already in project container; autograd, nn.Module, DDP all needed |
| torchinfo | latest | Model summary & param counting | CONTEXT.md specifies it; provides clean per-layer param breakdown |
| optuna | >=4.0 | Bayesian hyperparameter optimization | CONTEXT.md specifies Optuna; TPE sampler + MedianPruner is standard |
| tensorboard | latest | Training visualization | CONTEXT.md specifies TensorBoard logging |
| matplotlib | latest | Static plot generation | Convergence curves, accuracy bars, parameter tables |
| seaborn | latest | Statistical visualization | Error bars, violin plots for comparison reports |
| scipy | >=1.12 | Statistical testing | paired t-test, Wilcoxon, Bonferroni/Holm correction |
| torchvision | (container) | CIFAR-10/100 dataset loading | Standard dataset access for benchmark reproduction |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | >=1.26 | Array operations | Test oracles, statistical computations |
| json | stdlib | Result serialization | Manifest, config, metrics files |
| dataclasses | stdlib | Config objects | Type-safe experiment configuration |
| signal | stdlib | SIGINT handling | Graceful shutdown with checkpoint save |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| TensorBoard | Weights & Biases | wandb has AMD GPU metrics but TensorBoard is lighter and locked in CONTEXT.md |
| Optuna | Ray Tune | Ray Tune more powerful for distributed HPO but Optuna simpler for single-GPU sequential runs |
| Dataclasses | Hydra/OmegaConf | Hydra is more powerful but CONTEXT.md specifies dataclasses for type safety |
| matplotlib/seaborn | plotly | plotly gives interactivity but static plots are more reproducible for papers |

**Installation (inside dev container):**
```bash
docker compose run --rm dev uv add torchinfo optuna tensorboard matplotlib seaborn scipy
```

## Architecture Patterns

### Recommended Project Structure
```
src/octonion/
  baselines/                  # New subpackage for all baseline infrastructure
    __init__.py
    _algebra_linear.py        # AlgebraLinear base class + R/C/H/O implementations
    _algebra_conv.py           # AlgebraConv1d/2d implementations
    _algebra_rnn.py            # AlgebraRNN/LSTM/GRU implementations
    _normalization.py          # Algebra-aware batch norm (real, complex, quaternion, octonion)
    _activation.py             # Split activations + norm-preserving activations
    _initialization.py         # Per-algebra weight init (He, Trabelsi, Parcollet, unit-norm)
    _network.py                # AlgebraNetwork skeleton (configurable topology + depth)
    _param_matching.py         # Binary search for width matching within 1% tolerance
    _config.py                 # Dataclass configs for networks, training, experiments
    _trainer.py                # train_model() with logging, checkpointing, metrics
    _comparison.py             # run_comparison() with statistical testing + plotting
    _stats.py                  # Statistical tests (t-test, Wilcoxon, effect sizes)
    _plotting.py               # Auto-generated convergence curves, param tables
```

### Pattern 1: Algebra-Agnostic Network Skeleton
**What:** A single `AlgebraNetwork` class that accepts an algebra type and builds the network using algebra-specific layers.
**When to use:** All experiments comparing R/C/H/O.
**Example:**
```python
# Based on existing NormedDivisionAlgebra hierarchy and OctonionLinear pattern
from dataclasses import dataclass
from enum import Enum

class AlgebraType(Enum):
    REAL = ("R", 1, 8)      # (name, dim, multiplier for param matching)
    COMPLEX = ("C", 2, 4)
    QUATERNION = ("H", 4, 2)
    OCTONION = ("O", 8, 1)

    @property
    def dim(self) -> int:
        return self.value[1]

    @property
    def multiplier(self) -> int:
        return self.value[2]

@dataclass
class NetworkConfig:
    algebra: AlgebraType
    topology: str  # "mlp", "conv1d", "conv2d", "recurrent"
    depth: int
    base_hidden: int          # hidden units for octonion (others auto-scaled)
    activation: str = "split_relu"  # or "norm_preserving"
    output_projection: str = "flatten"  # or "real", "norm", "learned"
    use_batchnorm: bool = True

class AlgebraNetwork(nn.Module):
    def __init__(self, config: NetworkConfig):
        super().__init__()
        # Auto-scale hidden size for param matching
        self.hidden = config.base_hidden * config.algebra.multiplier
        # Build layers using algebra-specific factories
        self.layers = self._build_layers(config)

    def _build_layers(self, config):
        # Dispatch to MLP, Conv, or Recurrent builder
        ...
```

### Pattern 2: Per-Algebra Linear Layer (following OctonionLinear pattern)
**What:** Each algebra gets its own nn.Module linear layer operating on [..., dim] tensors.
**When to use:** Building algebra-specific computation within the network skeleton.
**Example:**
```python
# RealLinear: standard nn.Linear wrapper operating on [..., 1] (or equivalently [..., hidden])
class RealLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, dtype=torch.float32):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, dtype=dtype)
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')

    def forward(self, x):
        return self.linear(x)

# ComplexLinear: operates on [..., 2] tensors
class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, dtype=torch.float32):
        super().__init__()
        # Weight is a complex matrix: W = W_r + W_i * i
        # For in_features complex inputs -> out_features complex outputs:
        # Total real params = 2 * in_features * out_features (W_r, W_i)
        self.W_r = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.W_i = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        # Initialize via Trabelsi (Rayleigh magnitude + uniform phase)
        _complex_init(self.W_r, self.W_i, criterion='glorot')

    def forward(self, x):
        # x shape: [..., in_features, 2]
        x_r, x_i = x[..., 0], x[..., 1]
        # Complex matmul: (W_r + W_i*i)(x_r + x_i*i) = (W_r*x_r - W_i*x_i) + (W_r*x_i + W_i*x_r)*i
        out_r = F.linear(x_r, self.W_r) - F.linear(x_i, self.W_i)
        out_i = F.linear(x_r, self.W_i) + F.linear(x_i, self.W_r)
        return torch.stack([out_r, out_i], dim=-1)

# QuaternionLinear: operates on [..., 4] tensors
class QuaternionLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, dtype=torch.float32):
        super().__init__()
        # 4 weight matrices for Hamilton product
        self.W_r = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.W_i = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.W_j = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.W_k = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        # Initialize via Parcollet (polar form with Glorot/He variance)
        _quaternion_init(self.W_r, self.W_i, self.W_j, self.W_k, criterion='glorot')

    def forward(self, x):
        # x shape: [..., in_features, 4]
        # Hamilton product of W*x (16 multiplications, 12 additions)
        ...
```

### Pattern 3: Parameter-Matched Width Search
**What:** Binary search to find the hidden width that achieves target parameter count within 1% tolerance.
**When to use:** Before every experiment to ensure fair comparison.
**Example:**
```python
def find_matched_width(
    target_params: int,
    algebra: AlgebraType,
    topology: str,
    depth: int,
    tolerance: float = 0.01,
) -> int:
    """Binary search for hidden width that matches target param count within tolerance."""
    lo, hi = 1, target_params  # upper bound is generous
    best_width, best_diff = lo, float('inf')

    while lo <= hi:
        mid = (lo + hi) // 2
        config = NetworkConfig(algebra=algebra, topology=topology,
                               depth=depth, base_hidden=mid)
        model = AlgebraNetwork(config)
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        diff = abs(count - target_params) / target_params

        if diff < best_diff:
            best_diff = diff
            best_width = mid

        if diff <= tolerance:
            return mid
        elif count < target_params:
            lo = mid + 1
        else:
            hi = mid - 1

    if best_diff > tolerance:
        raise ValueError(f"Cannot match {target_params} params within {tolerance*100}% "
                         f"for {algebra.name}. Best: {best_width} ({best_diff*100:.2f}%)")
    return best_width
```

### Pattern 4: Training Utility with Full Observability
**What:** Complete training loop with checkpointing, gradient stats, TensorBoard, and VRAM monitoring.
**When to use:** All training runs.
**Example:**
```python
@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    optimizer: str = "adam"  # or "sgd", "adamw"
    scheduler: str = "cosine"  # or "step", "plateau"
    weight_decay: float = 0.0
    early_stopping_patience: int = 10
    warmup_epochs: int = 5
    use_amp: bool = False
    checkpoint_every: int = 10  # epochs
    seed: int = 42

def train_model(model, train_loader, val_loader, config, writer, device):
    """Full training loop with logging, checkpointing, and gradient monitoring."""
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler(enabled=config.use_amp)

    for epoch in range(config.epochs):
        # Train
        model.train()
        for batch in train_loader:
            with torch.amp.autocast('cuda', enabled=config.use_amp):
                loss = compute_loss(model, batch, device)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Log gradient statistics
        grad_norms = [p.grad.norm().item() for p in model.parameters()
                      if p.grad is not None]
        writer.add_scalar('Grad/norm_mean', np.mean(grad_norms), epoch)
        writer.add_scalar('Grad/norm_max', max(grad_norms), epoch)
        writer.add_scalar('Grad/norm_var', np.var(grad_norms), epoch)

        # Log VRAM
        if torch.cuda.is_available():
            writer.add_scalar('VRAM/peak_MB',
                              torch.cuda.max_memory_allocated() / 1e6, epoch)

        # Validate and early stopping
        val_loss, val_acc = evaluate(model, val_loader, device)
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        ...
```

### Anti-Patterns to Avoid
- **Sharing weights across algebras:** Each algebra has its own weight structure. Never share real-valued weights and then "interpret" them as complex/quaternion -- this defeats the purpose of hypercomplex representations.
- **Component-wise normalization for H/O:** Do NOT apply standard BatchNorm per-component for quaternion/octonion. Use the proper whitening approach (Gaudet & Maida 2018 for H, extend to O). Component-wise norm ignores cross-component correlations.
- **Hard-coding hidden sizes:** Always compute hidden sizes via the parameter matching binary search. Manual calculation is error-prone and breaks when architecture changes.
- **Ignoring normalization layer params:** Normalization layers have learnable parameters (scale, shift). Include them in param count for fair comparison.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Hyperparameter search | Grid/random search loop | Optuna with TPE sampler | Bayesian optimization converges faster; pruning saves GPU time |
| Statistical significance | Manual p-value computation | scipy.stats (ttest_rel, wilcoxon, mannwhitneyu) | Edge cases in small samples, correction factors |
| Multiple testing correction | Manual Bonferroni | scipy.stats or statsmodels (multipletests) | Holm-Bonferroni is strictly more powerful than Bonferroni |
| Model parameter summary | Manual counting loop | torchinfo.summary() | Handles edge cases (shared params, buffers, nested modules) |
| TensorBoard logging | Custom file writing | torch.utils.tensorboard.SummaryWriter | Handles serialization, file management, async writes |
| CIFAR data loading | Custom download/transform | torchvision.datasets.CIFAR10/CIFAR100 | Standard splits, transforms, verified checksums |
| LR scheduling | Manual LR decay | torch.optim.lr_scheduler (CosineAnnealingLR, etc.) | Warm restarts, plateau detection built in |
| Mixed precision | Manual casting | torch.amp.autocast + GradScaler | Handles loss scaling, gradient overflow automatically |

**Key insight:** The training infrastructure is 60% of this phase's code volume but is entirely standard PyTorch patterns. Use the ecosystem; focus implementation effort on the algebra-specific layers and parameter matching logic.

## Common Pitfalls

### Pitfall 1: Parameter Count Mismatch Due to Normalization Layers
**What goes wrong:** Algebra-aware batch normalization has different parameter counts across algebras. Real BN has 2*features params (scale+shift). Quaternion BN has 10+4=14 params per feature (4D scale matrix has 10 unique entries + 4D shift). Complex BN has 3+2=5 params per feature (2x2 symmetric scale + 2D shift). Forgetting these in param matching breaks the 1% guarantee.
**Why it happens:** Focus on linear layer params while ignoring normalization layers.
**How to avoid:** Include ALL trainable parameters in the count. The binary search approach handles this automatically if the full model (including norm layers) is constructed before counting.
**Warning signs:** Param counts differ by more than 1% despite correct linear layer matching.

### Pitfall 2: Quaternion Linear Forward Pass Sign Errors
**What goes wrong:** The Hamilton product has 16 terms with specific signs. Getting even one sign wrong produces a layer that trains (gradients still flow) but produces incorrect algebra-specific structure.
**Why it happens:** The Hamilton product is easy to write incorrectly. There are 6 cross-terms with signs that depend on the cyclic ordering ijk.
**How to avoid:** Cross-reference with existing Quaternion class (already verified in Phase 1). Write a unit test: `QuaternionLinear(W, x)` should match `W_quat * x_quat` using the verified `Quaternion.__mul__` method. Also cross-reference with Parcollet/SpeechBrain implementations.
**Warning signs:** Quaternion linear layer forward pass disagrees with verified `Quaternion.__mul__` on random inputs.

### Pitfall 3: Complex/Quaternion Initialization Scale Mismatch
**What goes wrong:** Using standard He/Glorot initialization for complex/quaternion weights produces gradients with wrong magnitude, causing training instability or slow convergence.
**Why it happens:** Complex params have 2 DOF, quaternion params have 4 DOF. The variance must be adjusted: for Glorot, sigma = 1/sqrt(2 * (fan_in + fan_out)) for complex (2 DOF), sigma = 1/sqrt(2 * fan_in) for He. Quaternion uses Rayleigh distribution with 4-DOF chi distribution for magnitude.
**How to avoid:** Follow the exact formulas from Trabelsi et al. (complex) and Parcollet et al./Gaudet & Maida (quaternion). Test: forward pass output variance should be approximately 1.0 for random unit-variance inputs.
**Warning signs:** Forward pass output variance >> 1 or << 1 with freshly initialized weights.

### Pitfall 4: Incorrect OctonionLinear Parameter Counting
**What goes wrong:** The existing `OctonionLinear` has only 2 parameters (a and b, each [8]), giving 16 real params total. This is very different from `ComplexLinear` or `QuaternionLinear` which are parameterized as weight matrices. Parameter matching needs to compare apples-to-apples.
**Why it happens:** `OctonionLinear` computes (a*x)*b which is a rank-1 bilinear operation, not a full linear map. A full octonionic linear layer would need weight matrices like the other algebras.
**How to avoid:** Build a proper `OctonionDenseLinear` that implements full octonionic matrix-vector product (analogous to QuaternionLinear with 8 weight matrices), and use the existing `OctonionLinear` only for specific rank-1 experiments. Or: stack multiple `OctonionLinear` layers to increase expressiveness. Decision needed during planning.
**Warning signs:** Octonionic network has drastically fewer params than R/C/H counterparts.

### Pitfall 5: CIFAR Color Channel Encoding
**What goes wrong:** CIFAR images have 3 color channels. For quaternion networks, Gaudet & Maida encode RGB as the imaginary part of a quaternion with the real part set to a function of R, G, B (e.g., grayscale or zero). For complex networks, different encodings exist. Using wrong encoding invalidates benchmark reproduction.
**Why it happens:** Assuming standard real-valued input preprocessing works for hypercomplex networks.
**How to avoid:** Follow the exact input encoding from the reference papers. For quaternion CIFAR: real part = 0 or grayscale, imaginary parts = R, G, B. For complex CIFAR: use the encoding from Trabelsi et al. Document the encoding in config.
**Warning signs:** Cannot reproduce published accuracy despite correct architecture.

### Pitfall 6: DDP + Custom Algebra Ops Compatibility
**What goes wrong:** PyTorch DDP assumes all parameters participate in every forward pass. If algebra-specific parameters are conditionally used, DDP may deadlock or produce incorrect gradients.
**Why it happens:** DDP uses gradient bucketing and AllReduce; unused parameters cause synchronization issues.
**How to avoid:** Set `find_unused_parameters=True` in DDP wrapper if any parameters are conditionally used. Better: ensure all parameters are used in every forward pass (which should be the case for well-designed algebra layers).
**Warning signs:** Training hangs on multi-GPU or gradients are NaN on second GPU.

## Code Examples

### Complex Weight Initialization (Trabelsi et al. 2018)
```python
# Source: Trabelsi et al., "Deep Complex Networks", ICLR 2018
# arxiv.org/abs/1705.09792
def complex_init(W_r: torch.Tensor, W_i: torch.Tensor, criterion: str = 'glorot'):
    """Initialize complex weight using Rayleigh magnitude + uniform phase.

    For a complex weight W = W_r + i*W_i:
    1. Sample magnitude |W| from Rayleigh(sigma) where sigma depends on fan_in/fan_out
    2. Sample phase theta from Uniform(-pi, pi)
    3. W_r = |W| * cos(theta), W_i = |W| * sin(theta)
    """
    fan_in, fan_out = W_r.shape[1], W_r.shape[0]
    if criterion == 'glorot':
        sigma = 1.0 / math.sqrt(2 * (fan_in + fan_out))
    elif criterion == 'he':
        sigma = 1.0 / math.sqrt(2 * fan_in)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    # Rayleigh distribution: magnitude
    magnitude = torch.distributions.Chi2(df=2).sample(W_r.shape).sqrt() * sigma
    # Uniform phase
    phase = torch.empty_like(W_r).uniform_(-math.pi, math.pi)

    with torch.no_grad():
        W_r.copy_(magnitude * torch.cos(phase))
        W_i.copy_(magnitude * torch.sin(phase))
```

### Quaternion Weight Initialization (Parcollet et al. / Gaudet & Maida)
```python
# Source: Gaudet & Maida, "Deep Quaternion Networks", IJCNN 2018
# arxiv.org/abs/1712.04604
def quaternion_init(W_r, W_i, W_j, W_k, criterion='glorot'):
    """Initialize quaternion weight using polar form with chi(4) magnitude.

    Quaternion weight W = W_r + W_i*i + W_j*j + W_k*k in polar form:
    1. Sample magnitude from chi distribution with 4 DOF, scaled by sigma
    2. Sample 3 phase angles uniformly
    3. Decompose into components using quaternion polar form
    """
    fan_in, fan_out = W_r.shape[1], W_r.shape[0]
    if criterion == 'glorot':
        sigma = 1.0 / math.sqrt(2 * (fan_in + fan_out))
    elif criterion == 'he':
        sigma = 1.0 / math.sqrt(2 * fan_in)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    # Chi distribution with 4 DOF for magnitude
    magnitude = torch.distributions.Chi2(df=4).sample(W_r.shape).sqrt() * sigma
    # Three uniform phase angles
    phi1 = torch.empty_like(W_r).uniform_(-math.pi, math.pi)
    phi2 = torch.empty_like(W_r).uniform_(-math.pi, math.pi)
    phi3 = torch.empty_like(W_r).uniform_(-math.pi, math.pi)

    # Quaternion polar decomposition
    with torch.no_grad():
        W_r.copy_(magnitude * torch.cos(phi1))
        # Imaginary parts share remaining magnitude
        imag_mag = magnitude * torch.sin(phi1)
        W_i.copy_(imag_mag * torch.cos(phi2))
        remaining = imag_mag * torch.sin(phi2)
        W_j.copy_(remaining * torch.cos(phi3))
        W_k.copy_(remaining * torch.sin(phi3))
```

### Quaternion Batch Normalization (Gaudet & Maida 2018)
```python
# Source: Gaudet & Maida, "Deep Quaternion Networks", IJCNN 2018
# Uses Cholesky decomposition on 4x4 covariance matrix for whitening
class QuaternionBatchNorm(nn.Module):
    """Quaternion batch normalization via 4D whitening.

    Instead of normalizing each component independently (which destroys
    quaternionic structure), whiten the 4D quaternion vectors using
    the Cholesky decomposition of their covariance matrix.

    Learnable parameters:
    - gamma: 4x4 symmetric scaling matrix (10 unique entries)
    - beta: 4D quaternion shift
    Total: 14 learnable params per feature (vs 2 for real BN)
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 4x4 symmetric scaling matrix (10 params) + 4D shift (4 params) = 14 per feature
        self.gamma = nn.Parameter(torch.zeros(num_features, 4, 4))
        self.beta = nn.Parameter(torch.zeros(num_features, 4))
        # Initialize gamma to identity
        with torch.no_grad():
            for i in range(4):
                self.gamma[:, i, i] = 1.0

        # Running stats
        self.register_buffer('running_mean', torch.zeros(num_features, 4))
        self.register_buffer('running_cov', torch.eye(4).unsqueeze(0).repeat(num_features, 1, 1))
```

### Complex Batch Normalization (Trabelsi et al. 2018)
```python
# Source: Trabelsi et al., "Deep Complex Networks", ICLR 2018
class ComplexBatchNorm(nn.Module):
    """Complex batch normalization via 2D whitening.

    Whiten complex features using the inverse square root of the 2x2
    covariance matrix V = [[Vrr, Vri], [Vri, Vii]].

    Learnable parameters:
    - gamma: 2x2 symmetric scaling (3 unique entries: gamma_rr, gamma_ri, gamma_ii)
    - beta: 2D complex shift (beta_r, beta_i)
    Total: 5 learnable params per feature (vs 2 for real BN)
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 2x2 symmetric scale (3 params) + 2D shift (2 params) = 5 per feature
        self.gamma_rr = nn.Parameter(torch.ones(num_features))
        self.gamma_ri = nn.Parameter(torch.zeros(num_features))
        self.gamma_ii = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features, 2))

        self.register_buffer('running_mean', torch.zeros(num_features, 2))
        self.register_buffer('running_var_rr', torch.ones(num_features))
        self.register_buffer('running_var_ri', torch.zeros(num_features))
        self.register_buffer('running_var_ii', torch.ones(num_features))
```

### Comparison Runner with Statistical Testing
```python
# Pattern for run_comparison
from scipy import stats

def run_comparison(task, algebras, seeds, config):
    results = {alg: [] for alg in algebras}

    for algebra in algebras:
        for seed in range(seeds):
            # Build param-matched model
            model = build_matched_model(task, algebra, config)
            metric = train_and_evaluate(model, task, seed, config)
            results[algebra].append(metric)

    # Statistical testing
    report = {}
    pairs = [(a, b) for i, a in enumerate(algebras) for b in algebras[i+1:]]
    p_values = []

    for a, b in pairs:
        t_stat, p_val = stats.ttest_rel(results[a], results[b])
        w_stat, w_p = stats.wilcoxon(results[a], results[b])
        effect = cohen_d(results[a], results[b])
        p_values.append(p_val)
        report[f"{a}_vs_{b}"] = {
            "t_test_p": p_val, "wilcoxon_p": w_p,
            "effect_size": effect,
            "mean_diff": np.mean(results[a]) - np.mean(results[b]),
        }

    # Holm-Bonferroni correction for multiple comparisons
    corrected = holm_bonferroni(p_values)
    ...
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Component-wise BN for quaternion | Whitening-based BN (Cholesky/covariance) | Gaudet & Maida 2018 | Preserves algebraic structure; significantly better convergence |
| Random normal init for complex | Rayleigh magnitude + uniform phase | Trabelsi et al. 2018 | Proper variance matching for 2-DOF parameters |
| Random normal init for quaternion | Chi(4) magnitude + polar form | Parcollet et al. 2019 | Proper variance matching for 4-DOF parameters |
| Manual HP tuning per algebra | Optuna Bayesian optimization per algebra | Standard 2024+ | Fair comparison: each algebra gets optimal hyperparameters |
| wandb for tracking | TensorBoard (project decision) | N/A | Lighter weight, no account needed, sufficient for research |
| torch.cuda.amp | torch.amp (device-agnostic) | PyTorch 2.4+ | Works on ROCm without modification |

**Deprecated/outdated:**
- `torch.cuda.amp.autocast` and `torch.cuda.amp.GradScaler`: Use `torch.amp.autocast('cuda')` and `torch.amp.GradScaler()` instead (device-agnostic API as of PyTorch 2.4+)
- `torch.nn.SyncBatchNorm`: Still current, use `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)` for DDP

## Benchmark Selection (Claude's Discretion)

### Recommended: CIFAR-10/100 for Both C and H Reproduction

**Rationale:** Both Gaudet & Maida (H) and Trabelsi et al. (C) report results on CIFAR-10/100 with parameter-matched comparisons. Using the same benchmark for both allows direct cross-algebra comparison.

**Published results to reproduce:**

| Algebra | Dataset | Error Rate | Parameters | Source |
|---------|---------|-----------|------------|--------|
| Real (baseline) | CIFAR-10 | 6.37% | 3,619,844 | Gaudet & Maida 2018, Table I |
| Complex | CIFAR-10 | 6.17% | ~1.7M | Trabelsi et al. 2018, Table 1 |
| Quaternion | CIFAR-10 | 5.44% | 932,792 | Gaudet & Maida 2018, Table I |
| Real (baseline) | CIFAR-100 | 28.07% | 3,619,844 | Gaudet & Maida 2018, Table I |
| Complex | CIFAR-100 | 26.36% | ~1.7M | Trabelsi et al. 2018, Table 1 |
| Quaternion | CIFAR-100 | 26.01% | 932,792 | Gaudet & Maida 2018, Table I |

**Architecture for reproduction:** ResNet-style with residual blocks. Gaudet & Maida used "deep" configuration with 10, 9, and 9 residual blocks per stage. Trabelsi used Wide/Shallow, Deep/Narrow, and In-Between variants. The Conv2D topology in the skeleton should support both.

**TIMIT as secondary benchmark (optional):** Parcollet et al. report 15.1% PER with QLSTM vs 15.3% for standard LSTM on TIMIT (3.3x fewer params). This validates the recurrent topology but requires audio data preprocessing. Consider as stretch goal.

### Real baseline verification
The real-valued baseline should also match published results (Gaudet reports 6.37%/28.07% for deep real ResNet on CIFAR-10/100). This validates the training infrastructure itself.

## Octonionic Linear Layer Design Decision

The existing `OctonionLinear` computes `(a * x) * b` with only 16 parameters (two 8-component vectors). This is a rank-1 bilinear map, not a full linear map. For fair comparison with `ComplexLinear` (which has `2 * in_features * out_features` real params) and `QuaternionLinear` (which has `4 * in_features * out_features` real params), an octonionic linear layer needs to be parameterized as a full octonionic matrix-vector product.

**Recommendation:** Build `OctonionDenseLinear` with 8 weight matrices (W_0 through W_7), analogous to how `QuaternionLinear` has 4 weight matrices (W_r, W_i, W_j, W_k). The forward pass computes the full octonionic matrix-vector product using the structure constants tensor, equivalent to `sum_j C[i,j,k] * W_j @ x_i` for output component k. This gives `8 * in_features * out_features` real parameters, matching the 8x multiplier for the real baseline.

The existing `OctonionLinear` remains available for specialized experiments but is not suitable for fair parameter-matched comparisons.

## Open Questions

1. **Octonionic batch normalization dimensions**
   - What we know: Real BN has 2 params/feature (scale, shift). Complex BN has 5. Quaternion BN has 14. Following the pattern, octonionic BN would need a full 8x8 symmetric scaling matrix (36 unique entries) + 8D shift = 44 params per feature.
   - What's unclear: Whether 44 params/feature is tractable, and whether the Cholesky approach scales to 8x8. No published octonionic BN exists.
   - Recommendation: Implement the full 8x8 whitening approach analogous to quaternionic. If numerically unstable, fall back to block-diagonal whitening (using quaternionic subalgebra structure: two 4x4 blocks). Test early.

2. **OctonionDenseLinear gradient correctness**
   - What we know: Phase 2 built autograd for octonion multiplication primitives. A full octonionic matrix-vector product combines multiple multiplications.
   - What's unclear: Whether PyTorch autograd handles the composed octonionic operations correctly without custom backward (since it uses the autograd Functions from Phase 2).
   - Recommendation: Extensive gradient checking (numeric Jacobian vs autograd) on the new OctonionDenseLinear layer before any training experiments.

3. **Recurrent layer benchmark**
   - What we know: Parcollet reports QLSTM results on TIMIT (15.1% PER). No published octonionic RNN/LSTM exists.
   - What's unclear: Whether TIMIT data preprocessing is worth the effort for Phase 3. It requires phoneme label alignment, feature extraction (MFCC/filterbank), etc.
   - Recommendation: Build recurrent layers but defer TIMIT reproduction to a stretch goal. Focus Conv2D reproduction (CIFAR) for C and H validation. The recurrent topology is still needed for Phase 5+ experiments.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest + hypothesis (already configured) |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `docker compose run --rm dev uv run pytest tests/test_baselines.py -x` |
| Full suite command | `docker compose run --rm dev uv run pytest -x --tb=short` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BASE-01 | Real baseline with 8x units matches octonionic param count within 1% | unit | `docker compose run --rm dev uv run pytest tests/test_param_matching.py::test_real_param_match -x` | No -- Wave 0 |
| BASE-01 | Real baseline architecture skeleton identical to octonionic | unit | `docker compose run --rm dev uv run pytest tests/test_param_matching.py::test_skeleton_identity -x` | No -- Wave 0 |
| BASE-02 | Complex baseline 4x units matches param count within 1% | unit | `docker compose run --rm dev uv run pytest tests/test_param_matching.py::test_complex_param_match -x` | No -- Wave 0 |
| BASE-02 | Complex baseline reproduces published CIFAR result within 1 std | integration | `docker compose run --rm dev uv run pytest tests/test_reproduction.py::test_complex_cifar -x` | No -- Wave 0 |
| BASE-03 | Quaternion baseline 2x units matches param count within 1% | unit | `docker compose run --rm dev uv run pytest tests/test_param_matching.py::test_quaternion_param_match -x` | No -- Wave 0 |
| BASE-03 | Quaternion baseline reproduces published CIFAR result within 1 std | integration | `docker compose run --rm dev uv run pytest tests/test_reproduction.py::test_quaternion_cifar -x` | No -- Wave 0 |
| SC-4 | All four networks share identical architecture skeleton | unit | `docker compose run --rm dev uv run pytest tests/test_param_matching.py::test_all_four_share_skeleton -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `docker compose run --rm dev uv run pytest tests/test_baselines.py tests/test_param_matching.py -x`
- **Per wave merge:** `docker compose run --rm dev uv run pytest -x --tb=short`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_baselines.py` -- covers algebra-specific linear layers, normalization, initialization
- [ ] `tests/test_param_matching.py` -- covers BASE-01, BASE-02, BASE-03 param matching + skeleton identity
- [ ] `tests/test_reproduction.py` -- covers CIFAR benchmark reproduction (integration tests, long-running)
- [ ] `tests/test_trainer.py` -- covers training utility, checkpointing, gradient logging
- [ ] `tests/test_comparison.py` -- covers comparison runner, statistical tests, manifest
- [ ] Dependencies install: `docker compose run --rm dev uv add torchinfo optuna tensorboard matplotlib seaborn`

## Sources

### Primary (HIGH confidence)
- Gaudet & Maida, "Deep Quaternion Networks", IJCNN 2018 (ar5iv.labs.arxiv.org/html/1712.04604) -- quaternion BN, init, CIFAR-10/100 results
- Trabelsi et al., "Deep Complex Networks", ICLR 2018 (arxiv.org/abs/1705.09792) -- complex BN, init, CIFAR-10/100 results
- Parcollet et al., "Quaternion Recurrent Neural Networks", ICLR 2019 (arxiv.org/abs/1806.04418) -- QLSTM, TIMIT results
- SpeechBrain quaternion documentation (speechbrain.readthedocs.io) -- quaternion layer implementations
- Existing project codebase: `OctonionLinear`, `NormedDivisionAlgebra`, `_tower.py` types
- PyTorch 2.9.1 documentation -- DDP, AMP, TensorBoard integration

### Secondary (MEDIUM confidence)
- Optuna documentation (optuna.org) -- TPE sampler, MedianPruner, PyTorch integration
- torchinfo PyPI (pypi.org/project/torchinfo) -- model summary package
- AMD ROCm DDP blog (rocm.blogs.amd.com) -- DDP on AMD GPUs

### Tertiary (LOW confidence)
- Octonionic batch normalization scaling (no published reference exists; extrapolated from quaternionic pattern)
- OctonionDenseLinear design (no published reference for full octonionic matrix-vector product in neural networks)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- mature PyTorch ecosystem, well-documented libraries
- Architecture patterns: MEDIUM-HIGH -- patterns proven for C/H, novel for O
- Initialization/normalization: HIGH -- directly from published papers with code references
- Benchmark reproduction targets: HIGH -- exact numbers from published papers
- Octonionic-specific layers: MEDIUM -- no published reference, extrapolated from lower algebras
- Pitfalls: HIGH -- documented from multiple sources and project experience

**Research date:** 2026-03-08
**Valid until:** 2026-04-08 (stable domain; hypercomplex ML papers move slowly)
