# Phase 5: Optimization Landscape - Research

**Researched:** 2026-03-20
**Domain:** Optimization landscape characterization, Hessian analysis, Riemannian optimization, loss surface curvature, hypercomplex network training
**Confidence:** MEDIUM-HIGH

## Summary

Phase 5 is the project's go/no-go gate -- the most consequential phase in the entire research. It requires building two new baselines (PHM-8 and R8-dense-mixing), integrating two new optimizer families (Riemannian via geoopt and Kronecker-factored via pytorch_optimizer's Shampoo), implementing Hessian eigenspectrum analysis at dual scales (full and stochastic Lanczos), designing 5 synthetic tasks with known optima, and running a massive experiment matrix (6 algebras x 5 optimizers x 5 tasks x 20 seeds = 3,000 training runs). The existing infrastructure (trainer, comparison runner, statistical testing, AlgebraNetwork skeleton) provides strong foundations, but substantial new code is needed for the PHM-8 layer, R8-dense-mixing layer, Hessian analysis toolkit, loss surface curvature measurement, synthetic task generators, and the tiered gate evaluation.

The primary technical risks are: (1) GPU memory/time constraints -- 3,000 training runs plus Hessian analysis is compute-heavy for a single 24GB GPU, requiring careful batching and scheduling; (2) geoopt integration with custom algebra layers (ManifoldParameter must wrap existing weight tensors); (3) Stochastic Lanczos implementation quality -- no single PyTorch library is perfectly maintained for this, so a lean custom implementation using Hessian-vector products may be most reliable.

**Primary recommendation:** Structure the phase as: infrastructure first (PHM-8 layer, R8-dense-mixing layer, synthetic tasks, Hessian toolkit, geoopt/Shampoo integration), then systematic experiments (convergence profiles, gradient variance, Hessian analysis, curvature measurement), then gate evaluation. Keep networks deliberately small (hidden=16-32 octonionic units, ~10K-100K params) to make 3,000 runs feasible on one GPU.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **5 synthetic tasks**: (1) algebra-native regression single-layer, (2) algebra-native regression multi-layer at depths 3/5/10, (3) 7D cross product recovery with quaternion negative control + 3D positive control + PHM-8 baseline + 4 noise levels, (4) sinusoidal regression, (5) multi-class classification with Gaussian clusters
- **Input dimensionality**: task-appropriate (8D algebra-native, 7D geometric, 8D standard) PLUS 64D variants
- **20 seeds** per task per algebra per optimizer
- **PHM-8 baseline**: Build from scratch (no published PHM code), integrated into AlgebraNetwork skeleton
- **R8-dense-mixing baseline**: Build in Phase 5 (not deferred), parameter-matched against octonionic network
- **Tiered go/no-go gate**: GREEN (within 2x on ALL tasks), YELLOW (within 2x on 2+ OR within 3x on ALL), RED (worse than 3x on majority)
- **Loss metrics**: best_val_loss ratio AND median final val loss ratio across 20 seeds; divergence rate as additional RED flag if >50%
- **Gate decision based purely on loss quality** -- wall-clock time reported but does not affect verdict
- **YELLOW handling**: User decides after review
- **RED output**: Full quantitative characterization paper + pivot plan
- **5 optimizers**: SGD (momentum 0.9, Nesterov), Adam, Riemannian Adam (geoopt, S^7 + Stiefel for O), LBFGS, Shampoo/K-FAC
- **Hessian dual-scale**: small networks with full Hessian via torch.autograd.functional.hessian, larger via stochastic Lanczos (~200 iterations)
- **4 Hessian checkpoints**: initialization, 25%, 50%, convergence
- **5 representative seeds** for evolution analysis; full 20-seed Hessian only at convergence
- **Bill & Cox curvature methodology**: random directions from converged solutions, measure average loss surface curvature, compare across all 6 algebra variants
- **Literature comparison**: explicitly reference Bill & Cox quaternion results and Wu et al. DON convergence

### Claude's Discretion
- Exact network architectures for each task (hidden width, batch norm, activation)
- Full Hessian max parameter budget (based on 24GB VRAM)
- PHM-8 implementation details (number of Kronecker terms, initialization)
- R8-dense-mixing layer design
- Data generation procedures for synthetic tasks
- Bill & Cox curvature sampling methodology details
- Geoopt manifold definitions for C and H (S^1 and S^3)
- Shampoo/K-FAC library choice and integration details
- Training hyperparameters per optimizer
- Which published results to compare against and how to normalize

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FOUND-04 | Optimization landscape characterization: gradient variance, Hessian eigenspectrum, saddle point vs local minima frequency, training stability across random seeds for octonionic networks on synthetic tasks with known optima, compared against R/C/H baselines -- serves as explicit go/no-go gate | All research sections below provide implementation guidance for each component: gradient variance via existing trainer grad_stats, Hessian via dual-scale approach, training stability via 20-seed comparison runner, go/no-go gate via tiered loss ratio evaluation |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.9.1 | Training, autograd, Hessian-vector products | Already in container; create_graph=True verified in Phase 2 |
| geoopt | 0.5.1 | Riemannian Adam optimizer with Sphere/Stiefel manifolds | Only mature PyTorch Riemannian optimization library; ManifoldParameter integrates with nn.Module |
| pytorch_optimizer | 3.10.0 | Shampoo optimizer | Actively maintained (Mar 2026), 100+ optimizers including Shampoo; torch-optimizer (0.3.0) is unmaintained since 2021 |
| scipy | >=1.17.1 | Eigendecomposition (eigvalsh), statistical tests | Already a dependency; provides LAPACK eigensolver for full Hessian |
| numpy | >=1.26 | Array operations, random sampling | Already a dependency |
| matplotlib | >=3.10.8 | Visualization of landscapes, spectra, convergence | Already a dependency |
| seaborn | >=0.13.2 | Statistical plot styling | Already a dependency |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.autograd.functional.hessian | PyTorch built-in | Full Hessian for small networks | Networks under ~2000 params (Hessian matrix fits in 24GB VRAM) |
| torch.autograd.grad (create_graph=True) | PyTorch built-in | Hessian-vector products for Lanczos | Networks too large for full Hessian |
| torch.linalg.eigvalsh | PyTorch built-in | GPU-accelerated eigenvalue computation | Full Hessian eigenspectrum on GPU |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pytorch_optimizer Shampoo | torch-optimizer 0.3.0 | torch-optimizer unmaintained since 2021; pytorch_optimizer actively updated |
| Custom Lanczos | pytorch-hessian-eigenthings | Not on PyPI (v0.0.2, git install only); thin wrapper around scipy ARPACK; custom is more transparent and controllable |
| Custom Lanczos | PyHessian | Last commit 2019; stale API; pyhessian on PyPI but likely incompatible with PyTorch 2.9 |
| geoopt ManifoldParameter | Manual retraction | geoopt handles retraction/transport automatically; manual would be error-prone |

**Installation:**
```bash
docker compose run --rm dev uv add geoopt pytorch-optimizer
```

**Version verification:** geoopt 0.5.1 confirmed on PyPI (released 2025-06-19, requires PyTorch >=2.0.1). pytorch_optimizer 3.10.0 confirmed on PyPI (released 2026-03-01, requires PyTorch >=1.10).

## Architecture Patterns

### Recommended Project Structure
```
src/octonion/
├── baselines/
│   ├── _phm_linear.py           # PHM-8 layer (new)
│   ├── _dense_mixing.py         # R8-dense-mixing layer (new)
│   ├── _config.py               # Extended AlgebraType enum (PHM8, R8_DENSE)
│   ├── _trainer.py              # Extended for LBFGS closure pattern
│   ├── _comparison.py           # Extended for 6 algebras + 5 optimizers
│   └── ...                      # Existing infrastructure unchanged
├── landscape/                    # New module for Phase 5
│   ├── __init__.py
│   ├── _hessian.py              # Full Hessian + stochastic Lanczos
│   ├── _curvature.py            # Bill & Cox loss surface curvature
│   ├── _gradient_stats.py       # Per-seed gradient variance collection
│   └── _gate.py                 # Tiered go/no-go evaluation
├── tasks/                        # New module for synthetic tasks
│   ├── __init__.py
│   ├── _algebra_native.py       # Task 1 (single-layer) and Task 2 (multi-layer)
│   ├── _cross_product.py        # Task 3 (7D cross product + controls)
│   ├── _sinusoidal.py           # Task 4 (sinusoidal regression)
│   └── _classification.py       # Task 5 (Gaussian cluster classification)
scripts/
├── run_landscape.py              # Main experiment orchestration script
├── analyze_landscape.py          # Post-hoc analysis and gate evaluation
results/
└── landscape/                    # Experiment outputs (JSON + plots)
tests/
├── test_landscape_hessian.py     # Hessian computation tests
├── test_landscape_tasks.py       # Synthetic task tests
├── test_phm_linear.py            # PHM-8 layer tests
└── test_dense_mixing.py          # R8-dense-mixing tests
```

### Pattern 1: PHM-8 Layer (Kronecker Product Sum)
**What:** PHM-n implements a linear layer as a sum of n Kronecker products: H = sum_i(A_i kron S_i), where A_i in R^(n x n) are learned mixing matrices and S_i in R^(k/n x d/n) are sub-matrices. For PHM-8: n=8, so A_i in R^(8x8), S_i in R^(out_f/8 x in_f/8).
**When to use:** As a baseline that isolates octonionic algebra structure from generic Kronecker factorization advantage.
**Example:**
```python
# Based on Zhang et al. ICLR 2021
class PHM8Linear(nn.Module):
    """Parameterized Hypercomplex Multiplication layer with n=8.

    H = sum_{i=0}^{n-1} A_i (x) S_i
    where A_i in R^(8x8) learns mixing rules,
    S_i in R^(out_f x in_f) are weight sub-matrices.

    Total params: n * (n*n + (out_f/n)*(in_f/n)) for learned A + S
    For fair comparison: use n=8, same in_f/out_f as OctonionDenseLinear.
    """
    def __init__(self, in_features: int, out_features: int, n: int = 8):
        super().__init__()
        self.n = n
        # Learned mixing matrices (replaces fixed structure constants)
        self.A = nn.ParameterList([
            nn.Parameter(torch.randn(n, n) * 0.1) for _ in range(n)
        ])
        # Sub-weight matrices
        self.S = nn.ParameterList([
            nn.Parameter(torch.empty(out_features, in_features))
            for _ in range(n)
        ])
        # Initialize S with appropriate scale
        for s in self.S:
            nn.init.xavier_normal_(s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features, 8]  (matching algebra tensor format)
        # Compute H = sum_i kron(A_i, S_i), apply to flattened x
        batch_shape = x.shape[:-2]
        x_flat = x.reshape(*batch_shape, -1)  # [..., in_f * 8]

        # Build full weight via Kronecker sum
        H = sum(
            torch.kron(self.A[i], self.S[i]) for i in range(self.n)
        )  # [out_f*8, in_f*8]

        out_flat = F.linear(x_flat, H)
        return out_flat.reshape(*batch_shape, -1, self.n)  # [..., out_f, 8]
```

### Pattern 2: R8-Dense-Mixing Layer
**What:** A real-valued 8D layer with full 8x8 cross-component mixing matrix per feature pair. Parameter-matched to OctonionDenseLinear but without any algebraic structure.
**When to use:** As the primary baseline for the go/no-go gate -- isolates whether octonionic algebra provides value beyond dense cross-component interaction.
**Example:**
```python
class DenseMixingLinear(nn.Module):
    """Real 8D linear with dense cross-component interaction.

    Each output component is a full linear combination of ALL input
    components across ALL features. Equivalent to nn.Linear(in_f*8, out_f*8)
    but maintains the [..., features, 8] tensor format for fair comparison.

    Params: out_f * 8 * in_f * 8 = 64 * in_f * out_f
    vs OctonionDenseLinear: 8 * in_f * out_f (8x fewer due to structure constants)

    For parameter matching: use smaller in_f/out_f to compensate.
    """
    def __init__(self, in_features: int, out_features: int, dim: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dim = dim
        # Full mixing: [out_f*dim, in_f*dim]
        self.weight = nn.Parameter(torch.empty(out_features * dim, in_features * dim))
        self.bias = nn.Parameter(torch.zeros(out_features * dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-2]
        x_flat = x.reshape(*batch_shape, -1)
        out_flat = F.linear(x_flat, self.weight, self.bias)
        return out_flat.reshape(*batch_shape, self.out_features, self.dim)
```

### Pattern 3: LBFGS Integration with Existing Trainer
**What:** LBFGS requires a closure pattern instead of the standard forward-backward-step flow. The trainer needs a conditional path.
**When to use:** For the LBFGS optimizer in the 5-optimizer suite.
**Example:**
```python
# Inside the training loop, when optimizer is LBFGS:
if isinstance(optimizer, torch.optim.LBFGS):
    def closure():
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        return loss
    loss = optimizer.step(closure)
else:
    # Standard forward-backward-step
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
```

### Pattern 4: Geoopt ManifoldParameter Integration
**What:** Wrapping algebra weight tensors as ManifoldParameters so RiemannianAdam constrains them to appropriate manifolds during optimization.
**When to use:** For the Riemannian Adam optimizer experiments.
**Example:**
```python
import geoopt

def wrap_manifold_params(model, algebra_type):
    """Replace weight parameters with ManifoldParameters for Riemannian optimization.

    Each weight tensor's last dimension is the algebra dimension.
    For S^n constraint: each algebra element (row of weights) lives on S^(dim-1).
    """
    sphere = geoopt.Sphere()
    stiefel = geoopt.Stiefel()  # For Stiefel manifold experiments

    for name, module in model.named_modules():
        if hasattr(module, 'weights'):  # OctonionDenseLinear
            for i, w in enumerate(module.weights):
                # Normalize each weight to unit norm per-row
                manifold_w = geoopt.ManifoldParameter(w.data, manifold=sphere)
                manifold_w.proj_()  # Project onto manifold
                module.weights[i] = manifold_w

    return model
```

### Pattern 5: Stochastic Lanczos Quadrature for Hessian Spectral Density
**What:** Approximate the eigenvalue density of the Hessian using Lanczos iteration with Hessian-vector products. Avoids O(n^2) memory of full Hessian.
**When to use:** For networks larger than the full-Hessian parameter budget (~2000 params).
**Example:**
```python
def hessian_vector_product(loss_grad, params, v):
    """Compute Hv via second-order autograd (requires create_graph=True in loss_grad)."""
    # loss_grad: gradient of loss w.r.t. params (computed with create_graph=True)
    # v: vector to multiply with Hessian
    Hv = torch.autograd.grad(
        loss_grad, params, grad_outputs=v,
        retain_graph=True, allow_unused=True
    )
    return torch.cat([h.reshape(-1) for h in Hv if h is not None])

def stochastic_lanczos(model, loss_fn, data, n_iterations=200, n_samples=5):
    """Estimate Hessian spectral density via stochastic Lanczos quadrature.

    Algorithm (Ghorbani et al. 2019):
    1. For each random starting vector v:
       a. Run Lanczos iteration for n_iterations steps using HVP
       b. Obtain tridiagonal matrix T
       c. Compute eigenvalues of T (Ritz values)
    2. Aggregate Ritz values across samples for spectral density estimate
    """
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

    all_ritz_values = []
    for _ in range(n_samples):
        # Random starting vector on unit sphere
        v = torch.randn(n_params, device=params[0].device)
        v = v / v.norm()

        # Lanczos iteration
        alpha = []  # Diagonal of tridiagonal matrix
        beta = []   # Off-diagonal
        V = [v]     # Lanczos vectors

        # Compute gradient with create_graph=True
        model.zero_grad()
        output = model(data[0])
        loss = loss_fn(output, data[1])
        grad = torch.autograd.grad(loss, params, create_graph=True)
        grad_flat = torch.cat([g.reshape(-1) for g in grad])

        w = hessian_vector_product(grad_flat, params, _unflatten(v, params))
        alpha_k = w.dot(v).item()
        alpha.append(alpha_k)
        w = w - alpha_k * v

        for k in range(1, n_iterations):
            beta_k = w.norm().item()
            if beta_k < 1e-10:
                break
            beta.append(beta_k)
            v_prev = v
            v = w / beta_k
            V.append(v)

            w = hessian_vector_product(grad_flat, params, _unflatten(v, params))
            w = w - beta_k * v_prev
            alpha_k = w.dot(v).item()
            alpha.append(alpha_k)
            w = w - alpha_k * v

        # Build tridiagonal matrix and compute eigenvalues
        T = _build_tridiagonal(alpha, beta)
        ritz_values = torch.linalg.eigvalsh(T)
        all_ritz_values.append(ritz_values)

    return all_ritz_values
```

### Pattern 6: Bill & Cox Loss Surface Curvature Measurement
**What:** Sample random filter-normalized directions from a converged model, compute loss along each direction, and estimate average surface curvature.
**When to use:** For comparing loss landscape smoothness across algebra types (SC extends Bill & Cox 2024 quaternion results to octonions).
**Example:**
```python
def measure_curvature(model, loss_fn, dataloader, n_directions=50, n_steps=51, step_range=1.0):
    """Measure average loss surface curvature around converged model.

    Bill & Cox (2024) methodology:
    1. Save converged weights theta*
    2. For each random direction d (filter-normalized per Li et al. 2018):
       a. Sample loss along theta* + alpha * d for alpha in [-step_range, step_range]
       b. Fit quadratic to get curvature estimate
    3. Report mean/median curvature across directions
    """
    theta_star = {n: p.clone() for n, p in model.named_parameters()}

    curvatures = []
    for _ in range(n_directions):
        # Generate random direction, filter-normalize (Li et al. 2018)
        direction = {}
        for name, param in model.named_parameters():
            d = torch.randn_like(param)
            # Filter normalization: scale each filter to match param's norm
            if param.dim() >= 2:
                for j in range(d.shape[0]):
                    d[j] *= param[j].norm() / (d[j].norm() + 1e-10)
            direction[name] = d

        # Sample loss along direction
        alphas = torch.linspace(-step_range, step_range, n_steps)
        losses = []
        for alpha in alphas:
            # Set weights to theta* + alpha * d
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(theta_star[name] + alpha * direction[name])

            loss = evaluate_loss(model, loss_fn, dataloader)
            losses.append(loss)

        # Fit quadratic: loss(alpha) ~ a*alpha^2 + b*alpha + c
        # Curvature = 2*a
        coeffs = np.polyfit(alphas.numpy(), losses, 2)
        curvatures.append(2 * coeffs[0])

    # Restore original weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(theta_star[name])

    return {
        'mean_curvature': np.mean(curvatures),
        'median_curvature': np.median(curvatures),
        'std_curvature': np.std(curvatures),
        'curvatures': curvatures,
    }
```

### Anti-Patterns to Avoid
- **Training all 3,000 runs sequentially without checkpointing:** Would take weeks; batch by task, save intermediate results after each (task, algebra, optimizer) combination
- **Full Hessian on the "larger" networks:** Memory is O(n^2) for n params; 10K params = 10K x 10K float32 = 400MB, but 100K params = 100K x 100K = 40GB (OOM). Use stochastic Lanczos beyond ~2000 params.
- **Using geoopt ManifoldParameter on all parameters:** Only algebra weight tensors should be manifold-constrained; bias, batch norm parameters, projection layers remain Euclidean
- **LBFGS with minibatches:** LBFGS is designed for full-batch optimization or large-batch approximation; using small minibatches violates its assumptions and causes instability. Use full dataset or large batches.
- **Comparing raw loss values across different tasks:** Gate evaluation should use loss RATIOS (O/R8-dense) per task, not absolute loss differences
- **Parameter-matching PHM-8 by width only:** PHM-8's parameter count includes the A_i mixing matrices (n * n * n = 512 params for PHM-8) which OctonionDenseLinear doesn't have. Account for this in matching.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Riemannian optimization | Custom retraction/transport | geoopt RiemannianAdam + ManifoldParameter | Retraction on Stiefel manifold involves QR decomposition; getting numerical stability right is hard |
| Shampoo optimizer | Custom K-FAC/Shampoo | pytorch_optimizer.Shampoo | Shampoo requires matrix root computation; numerically delicate |
| Eigenvalue computation | Custom eigensolver | torch.linalg.eigvalsh / scipy.linalg.eigvalsh | LAPACK is battle-tested; custom implementations have convergence bugs |
| Statistical testing | Custom p-value correction | Existing _stats.py (paired_comparison, holm_bonferroni) | Already implemented and tested in Phase 3 |
| Seeded reproducibility | Custom seed management | Existing seed_everything() | Already handles all PyTorch/NumPy/Python RNG sources |
| Filter normalization for curvature | Custom normalization | Follow Li et al. 2018 recipe exactly | Filter normalization has non-obvious edge cases (BN layers, bias terms) |

**Key insight:** The existing Phase 3 infrastructure (trainer, comparison runner, statistical testing, plotting) handles 60% of what Phase 5 needs. The new code is primarily: (1) two new layer types, (2) synthetic task generators, (3) Hessian analysis toolkit, (4) curvature measurement, (5) gate evaluation logic.

## Common Pitfalls

### Pitfall 1: Full Hessian Memory Explosion
**What goes wrong:** Attempting torch.autograd.functional.hessian() on a network with >5000 params causes OOM on 24GB GPU.
**Why it happens:** Hessian is n x n where n = number of params. 5000 params = 25M entries = 100MB in float32, but autograd intermediate tensors multiply this by 10-50x.
**How to avoid:** Set maximum parameter budget for full Hessian at ~2000 params. Use 1-hidden-layer networks with hidden=16 octonionic units (16 * 8 * 2 + biases ~ 2000 params). For anything larger, use stochastic Lanczos.
**Warning signs:** torch.cuda.OutOfMemoryError during Hessian computation; extremely slow Hessian computation (>1 min for a single Hessian).

### Pitfall 2: LBFGS Closure Pattern Mismatch
**What goes wrong:** LBFGS step() requires a closure that recomputes the loss; standard training loop doesn't provide this.
**Why it happens:** LBFGS performs multiple function evaluations per step internally (line search). The existing trainer's loop is designed for first-order optimizers.
**How to avoid:** Create a separate LBFGS training path in the trainer that wraps forward+backward in a closure. Use max_iter=20 for the line search. Use full-batch or large-batch evaluation.
**Warning signs:** RuntimeError about closure being None; loss not decreasing (indicates line search failure).

### Pitfall 3: Geoopt ManifoldParameter + ParameterList Interaction
**What goes wrong:** OctonionDenseLinear uses nn.ParameterList for 8 weight matrices. Replacing these with ManifoldParameters may break ParameterList iteration.
**Why it happens:** ManifoldParameter is a subclass of nn.Parameter, so it should work, but custom manifold constraints may interfere with gradient computation for structure-constant-based operations.
**How to avoid:** Test ManifoldParameter wrapping on a single OctonionDenseLinear layer first, verify gradients match. Consider wrapping at the model level rather than layer level -- apply manifold constraint as a post-step projection rather than via ManifoldParameter.
**Warning signs:** NaN gradients after ManifoldParameter wrapping; model diverging where it previously converged.

### Pitfall 4: Divergent Seeds Contaminating Gate Metrics
**What goes wrong:** Some seeds diverge (loss -> inf or NaN), making mean/median comparisons meaningless.
**Why it happens:** Random initialization can land in bad basins, especially for octonionic networks with non-associative structure.
**How to avoid:** Track divergence rate separately (>10x initial loss = diverged). Use median (not mean) for the final val loss ratio. Report divergence rate as an independent metric. The gate already includes "divergence rate >50% = RED flag" rule.
**Warning signs:** NaN in loss values; loss increasing monotonically after warmup; large variance across seeds.

### Pitfall 5: PHM-8 Parameter Count Mismatch
**What goes wrong:** PHM-8 has additional parameters from learned mixing matrices A_i that OctonionDenseLinear doesn't have (structure constants are fixed). Naive parameter matching undercounts PHM-8 params.
**Why it happens:** OctonionDenseLinear: 8 * in_f * out_f params. PHM-8 with n=8: 8 * (8*8) + 8 * in_f * out_f = 512 + 8 * in_f * out_f. For small networks (in_f=out_f=16), the 512 mixing params are 25% of the total.
**How to avoid:** Use find_matched_width to binary-search the correct width for PHM-8 against the OctonionDenseLinear reference. The A_i matrices should be included in the parameter count.
**Warning signs:** PHM-8 consistently outperforming both O and R -- may indicate it has more parameters.

### Pitfall 6: Stochastic Lanczos Numerical Instability
**What goes wrong:** Lanczos iteration loses orthogonality after many steps, producing spurious eigenvalues.
**Why it happens:** Finite precision arithmetic causes Lanczos vectors to gradually lose mutual orthogonality (classic "ghost eigenvalue" problem).
**How to avoid:** Use full reorthogonalization (Gram-Schmidt against all previous Lanczos vectors at each step). For 200 iterations with ~2000 params, this is O(200^2 * 2000) = 80M FLOPs, which is negligible vs the HVP cost. Alternatively, run multiple independent Lanczos with fewer iterations and merge spectra.
**Warning signs:** Duplicate eigenvalues in spectrum; eigenvalues outside the known spectral range [lambda_min, lambda_max].

## Code Examples

### Synthetic Task 1: Algebra-Native Regression (Single Layer)
```python
def build_algebra_native_single(n_train=50000, n_test=10000, dim=8, seed=42):
    """Generate data for y = a * x * b where a, b are fixed algebra elements.

    Known optimal loss = 0 (single OctonionLinear can represent this exactly).
    Uses octonion multiplication for O; quaternion for H; complex for C; real for R.
    """
    rng = torch.Generator().manual_seed(seed)

    # Fixed ground truth weights (unit norm for numerical stability)
    a = torch.randn(dim, generator=rng)
    a = a / a.norm()
    b = torch.randn(dim, generator=rng)
    b = b / b.norm()

    # Generate inputs
    x_train = torch.randn(n_train, dim, generator=rng) * 0.5
    x_test = torch.randn(n_test, dim, generator=rng) * 0.5

    # Compute targets using algebra multiplication
    # For octonions: y = octonion_mul(octonion_mul(a, x), b)
    y_train = algebra_transform(a, x_train, b, dim)
    y_test = algebra_transform(a, x_test, b, dim)

    return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)
```

### Synthetic Task 3: 7D Cross Product Recovery
```python
def build_cross_product_recovery(n_train=50000, n_test=10000, noise_level=0.0,
                                  cross_dim=7, seed=42):
    """Generate data with 7D cross product signal embedded in noise.

    Controls:
    - cross_dim=7: Tests octonionic 7D cross product detection
    - cross_dim=3: Positive control (quaternions should excel here)

    The signal: y = cross_product(x[:d], fixed_vector[:d]) + noise
    where d = cross_dim and fixed_vector is a known unit vector.
    """
    rng = torch.Generator().manual_seed(seed)

    fixed_vec = torch.randn(cross_dim, generator=rng)
    fixed_vec = fixed_vec / fixed_vec.norm()

    x_train = torch.randn(n_train, cross_dim, generator=rng)
    y_clean = seven_dim_cross_product(x_train, fixed_vec.unsqueeze(0).expand_as(x_train))
    noise = torch.randn_like(y_clean) * noise_level * y_clean.std()
    y_train = y_clean + noise

    x_test = torch.randn(n_test, cross_dim, generator=rng)
    y_test = seven_dim_cross_product(x_test, fixed_vec.unsqueeze(0).expand_as(x_test))

    return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)
```

### Hessian Eigenspectrum at Checkpoint
```python
def compute_hessian_spectrum(model, loss_fn, dataloader, device, method='auto'):
    """Compute Hessian eigenspectrum using appropriate method based on model size.

    Auto-selects full Hessian (small models) or stochastic Lanczos (larger models).
    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if method == 'auto':
        method = 'full' if n_params <= 2000 else 'lanczos'

    if method == 'full':
        # Collect all data into single batch for full Hessian
        all_x, all_y = [], []
        for batch in dataloader:
            all_x.append(batch[0])
            all_y.append(batch[1])
        x = torch.cat(all_x).to(device)
        y = torch.cat(all_y).to(device)

        # Define loss as function of flattened parameters
        def loss_func(flat_params):
            # Unflatten and set model parameters
            _set_flat_params(model, flat_params)
            output = model(x)
            return loss_fn(output, y)

        flat_params = _get_flat_params(model)
        H = torch.autograd.functional.hessian(loss_func, flat_params)
        eigenvalues = torch.linalg.eigvalsh(H)

        return {
            'eigenvalues': eigenvalues.cpu().numpy(),
            'n_negative': int((eigenvalues < 0).sum()),
            'n_positive': int((eigenvalues > 0).sum()),
            'n_zero': int(torch.isclose(eigenvalues, torch.zeros(1, device=device)).sum()),
            'trace': float(eigenvalues.sum()),
            'spectral_norm': float(eigenvalues.abs().max()),
            'negative_ratio': float((eigenvalues < 0).sum()) / len(eigenvalues),
            'method': 'full',
        }
    else:
        # Stochastic Lanczos
        ritz_values = stochastic_lanczos(model, loss_fn, dataloader,
                                          n_iterations=200, n_samples=5, device=device)
        all_ritz = np.concatenate([rv.cpu().numpy() for rv in ritz_values])

        return {
            'ritz_values': all_ritz,
            'n_negative_approx': int((all_ritz < 0).sum()),
            'negative_ratio_approx': float((all_ritz < 0).sum()) / len(all_ritz),
            'trace_approx': float(all_ritz.sum()) / len(ritz_values),  # Average per sample
            'method': 'lanczos',
            'n_iterations': 200,
            'n_samples': 5,
        }
```

### Go/No-Go Gate Evaluation
```python
def evaluate_gate(results: dict) -> dict:
    """Evaluate tiered go/no-go gate from converged experiment results.

    Gate uses loss ratios: octonionic / R8-dense-mixing baseline.

    GREEN: O within 2x on ALL tasks
    YELLOW: O within 2x on 2+ tasks OR within 3x on ALL tasks
    RED: O worse than 3x on majority of tasks
    """
    gate_results = {}
    for task_name, task_data in results.items():
        o_losses = task_data['O']['final_val_losses']  # 20 seeds
        r8_losses = task_data['R8_DENSE']['final_val_losses']

        # Metric 1: Best validation loss ratio
        best_ratio = min(o_losses) / min(r8_losses)

        # Metric 2: Median final val loss ratio
        median_ratio = np.median(o_losses) / np.median(r8_losses)

        # Use most favorable for O
        gate_ratio = min(best_ratio, median_ratio)

        # Divergence rate
        initial_loss = task_data['O']['initial_loss']
        diverged = sum(1 for l in o_losses if l > 10 * initial_loss)
        divergence_rate = diverged / len(o_losses)

        gate_results[task_name] = {
            'best_ratio': best_ratio,
            'median_ratio': median_ratio,
            'gate_ratio': gate_ratio,
            'within_2x': gate_ratio <= 2.0,
            'within_3x': gate_ratio <= 3.0,
            'divergence_rate': divergence_rate,
        }

    # Aggregate gate decision
    n_tasks = len(gate_results)
    n_within_2x = sum(1 for r in gate_results.values() if r['within_2x'])
    n_within_3x = sum(1 for r in gate_results.values() if r['within_3x'])
    any_high_divergence = any(r['divergence_rate'] > 0.5 for r in gate_results.values())

    if n_within_2x == n_tasks and not any_high_divergence:
        verdict = 'GREEN'
    elif any_high_divergence:
        verdict = 'RED'  # Divergence flag overrides
    elif n_within_2x >= 2 or n_within_3x == n_tasks:
        verdict = 'YELLOW'
    elif n_within_3x < n_tasks // 2 + 1:
        verdict = 'RED'
    else:
        verdict = 'YELLOW'

    return {'verdict': verdict, 'per_task': gate_results}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| torch.autograd.functional.hessian | torch.func.hessian (functorch) | PyTorch 2.0+ | More performant with vmap vectorization; but torch.autograd.functional.hessian still works and is more straightforward for our use case |
| Manual Riemannian SGD | geoopt RiemannianAdam | 2020 | Automatic retraction, parallel transport, manifold projection |
| Hand-written Shampoo | pytorch_optimizer.Shampoo | 2024+ | Reliable matrix root computation, tested on modern PyTorch |
| Manual seed management | seed_everything (project utility) | Phase 3 | Already handles all RNG sources |
| Per-run statistical tests | holm_bonferroni correction (project utility) | Phase 3 | Already implements family-wise error control |

**Deprecated/outdated:**
- torch-optimizer package (v0.3.0, last updated 2021): Use pytorch_optimizer (v3.10.0) instead
- PyHessian (last commit 2019): Likely incompatible with PyTorch 2.9; implement Lanczos manually
- pytorch-hessian-eigenthings (v0.0.2, not on PyPI): Thin wrapper; manual implementation is more transparent

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest + hypothesis |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `docker compose run --rm dev uv run pytest tests/test_landscape_tasks.py tests/test_phm_linear.py tests/test_dense_mixing.py tests/test_landscape_hessian.py -x` |
| Full suite command | `docker compose run --rm dev uv run pytest -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FOUND-04a | Gradient variance across 20+ seeds characterized | integration | `docker compose run --rm dev uv run pytest tests/test_landscape_tasks.py::test_gradient_variance_collection -x` | No - Wave 0 |
| FOUND-04b | Hessian eigenspectrum at convergence, negative eigenvalue ratio within 3x of quaternionic | integration | `docker compose run --rm dev uv run pytest tests/test_landscape_hessian.py::test_hessian_spectrum_small -x` | No - Wave 0 |
| FOUND-04c | Convergence profiles across 3+ optimizers documented | integration | `docker compose run --rm dev uv run pytest tests/test_landscape_tasks.py::test_convergence_profiles -x` | No - Wave 0 |
| FOUND-04d | Octonionic loss within 2x of R8-dense-mixing on 3+ tasks | integration | `docker compose run --rm dev uv run pytest tests/test_landscape_tasks.py::test_gate_evaluation -x` | No - Wave 0 |
| FOUND-04e | Landscape pathology characterized quantitatively | integration | `docker compose run --rm dev uv run pytest tests/test_landscape_hessian.py::test_curvature_measurement -x` | No - Wave 0 |
| PHM-8 | PHM-8 layer forward/backward correct | unit | `docker compose run --rm dev uv run pytest tests/test_phm_linear.py -x` | No - Wave 0 |
| R8-dense | R8-dense-mixing layer correct | unit | `docker compose run --rm dev uv run pytest tests/test_dense_mixing.py -x` | No - Wave 0 |
| Hessian | Full Hessian matches finite-diff for toy model | unit | `docker compose run --rm dev uv run pytest tests/test_landscape_hessian.py::test_full_hessian_toy -x` | No - Wave 0 |
| Lanczos | Stochastic Lanczos eigenvalues match full on small model | unit | `docker compose run --rm dev uv run pytest tests/test_landscape_hessian.py::test_lanczos_vs_full -x` | No - Wave 0 |

### Sampling Rate
- **Per task commit:** `docker compose run --rm dev uv run pytest tests/test_phm_linear.py tests/test_dense_mixing.py tests/test_landscape_hessian.py tests/test_landscape_tasks.py -x --timeout=120`
- **Per wave merge:** `docker compose run --rm dev uv run pytest -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_phm_linear.py` -- PHM-8 layer unit tests (forward shape, backward gradients, parameter count)
- [ ] `tests/test_dense_mixing.py` -- R8-dense-mixing layer unit tests
- [ ] `tests/test_landscape_hessian.py` -- Hessian computation tests (full, Lanczos, curvature)
- [ ] `tests/test_landscape_tasks.py` -- Synthetic task generator tests + smoke integration tests
- [ ] `geoopt` and `pytorch_optimizer` packages installed: `docker compose run --rm dev uv add geoopt pytorch-optimizer`

## Open Questions

1. **Full Hessian parameter budget**
   - What we know: Hessian is n x n float32; 2000 params = 16MB Hessian matrix, but autograd intermediates can be 10-50x larger
   - What's unclear: Exact GPU memory overhead of torch.autograd.functional.hessian with create_graph=True intermediates
   - Recommendation: Start with n=500 params, benchmark memory, scale up to find the practical limit on 24GB VRAM. Likely safe up to ~2000 params. Test empirically in Wave 0.

2. **Geoopt ManifoldParameter + OctonionDenseLinear compatibility**
   - What we know: ManifoldParameter subclasses nn.Parameter; geoopt Sphere supports arbitrary dimensions
   - What's unclear: Whether wrapping weights inside ParameterList with ManifoldParameter works correctly with the fused einsum forward pass. The S^7 constraint means each weight "row" (8-element octonion) should have unit norm, but weights are stored as [out_features, in_features] per component.
   - Recommendation: Test first. Fallback: post-step projection (manually project weights onto manifold after each optimizer step) rather than ManifoldParameter.

3. **PHM-8 number of Kronecker terms**
   - What we know: Original Zhang et al. use n terms where n = algebra dimension. For PHM-4 (quaternion), n=4.
   - What's unclear: Whether PHM-8 should use exactly 8 Kronecker terms (matching the paper's formula) or fewer (parameter efficiency)
   - Recommendation: Use n=8 (matching the formula) for scientific rigor. The 512 extra params from A_i matrices are accounted for in parameter matching.

4. **Shampoo preconditioner interaction with octonionic Kronecker structure**
   - What we know: Shampoo uses Kronecker-factored preconditioners; OctonionDenseLinear fuses weights via Kronecker-like structure constants
   - What's unclear: Whether Shampoo's Kronecker approximation interacts favorably or adversely with the algebraic Kronecker structure
   - Recommendation: This is exactly what the experiment tests. Just run it and report. No special handling needed.

5. **Network size for practical 3,000 runs**
   - What we know: 20 seeds x 5 tasks x 6 algebras x 5 optimizers = 3,000 runs. Each must complete training.
   - What's unclear: Exact time per run. At 100 epochs with batch_size=128 on 50K samples (391 batches), each epoch takes ~1-5 seconds for small MLPs. So 100 epochs = 2-8 minutes per run. 3,000 runs = 100-400 hours.
   - Recommendation: Use small networks (hidden=16, depth=1 or 3, ~10K params) and moderate epochs (50-100). Budget 200-400 GPU-hours total. Run tasks sequentially, save intermediate results. Consider reducing to 10 seeds for the expensive Hessian analysis checkpoint runs.

## Sources

### Primary (HIGH confidence)
- [geoopt 0.5.1 on PyPI](https://pypi.org/project/geoopt/) -- Latest version, install requirements, PyTorch compatibility
- [geoopt manifolds documentation](https://geoopt.readthedocs.io/en/latest/manifolds.html) -- Sphere, Stiefel manifold implementations
- [geoopt Sphere source](https://github.com/geoopt/geoopt/blob/master/geoopt/manifolds/sphere.py) -- Constructor parameters, random_uniform, projx
- [geoopt optimizers documentation](https://geoopt.readthedocs.io/en/latest/optimizers.html) -- RiemannianAdam, RiemannianSGD API
- [pytorch_optimizer 3.10.0 on PyPI](https://pypi.org/project/pytorch_optimizer/) -- Latest version, 100+ optimizers including Shampoo
- [PyTorch torch.autograd.functional.hessian](https://docs.pytorch.org/docs/stable/generated/torch.autograd.functional.hessian.html) -- Full Hessian computation API
- Existing codebase: _trainer.py, _comparison.py, _config.py, _algebra_linear.py, _stats.py, _stabilization.py -- Phase 3/4 infrastructure

### Secondary (MEDIUM confidence)
- [Zhang et al. ICLR 2021 - PHM paper](https://arxiv.org/pdf/2102.08597) -- PHM-n architecture definition, Kronecker product formulation
- [Bill & Cox 2024 - Quaternion loss surfaces](https://link.springer.com/article/10.1007/s00006-024-01313-2) -- Loss surface curvature methodology (could not access full paper; methodology inferred from abstract and related Li et al. 2018 work)
- [Li et al. 2018 - Visualizing Loss Landscapes](https://arxiv.org/abs/1712.09913) -- Filter normalization methodology for random direction sampling
- [Ghorbani et al. 2019 - Hessian Eigenvalue Density](https://arxiv.org/pdf/1901.10159) -- Stochastic Lanczos quadrature algorithm specification
- [pytorch-hessian-eigenthings](https://github.com/noahgolmant/pytorch-hessian-eigenthings) -- Lanczos implementation reference (not recommended for direct use)
- [Wu et al. 2020 - Deep Octonion Networks](https://arxiv.org/abs/1903.08478) -- DON convergence results on CIFAR (reported better convergence and accuracy vs R/C/H baselines)

### Tertiary (LOW confidence)
- Bill & Cox curvature methodology details: Could not access full paper text; surface curvature measurement procedure reconstructed from abstract + Li et al. 2018 + standard practices. Should verify against full paper before implementation.
- Exact DON CIFAR error rates: Could not extract specific numbers from Wu et al. PDF. Need to reference during implementation for comparison.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries verified on PyPI with current versions and compatibility
- Architecture (PHM-8, R8-dense): MEDIUM -- PHM formulation from ICLR 2021 paper is clear, but implementation details for integration with existing codebase need careful testing
- Architecture (Hessian analysis): MEDIUM -- algorithm is well-known, but practical parameter budget on 24GB VRAM needs empirical validation
- Architecture (Bill & Cox curvature): MEDIUM-LOW -- methodology inferred from abstract and Li et al. 2018; full paper inaccessible
- Pitfalls: HIGH -- based on direct codebase analysis and known numerical issues

**Research date:** 2026-03-20
**Valid until:** 2026-04-20 (30 days -- stable domain, no fast-moving dependencies)
