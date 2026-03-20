# Phase 4: Numerical Stability - Research

**Researched:** 2026-03-19
**Domain:** Floating-point precision analysis, condition number characterization, mitigation strategies for hypercomplex neural networks
**Confidence:** HIGH

## Summary

Phase 4 is a measurement and characterization phase, not a modeling phase. The core task is to build instrumentation that quantifies how floating-point errors accumulate through chains of octonionic operations at various depths (10, 50, 100, 500), characterize condition numbers of primitive and composed operations, compare float32 vs float64 precision, and demonstrate that periodic re-normalization extends stable depth by at least 2x. All four algebras (R, C, H, O) are compared side-by-side at every measurement point.

The existing codebase provides nearly all infrastructure needed: `_numeric.py` for finite-difference Jacobians, `_jacobians.py` for analytic Jacobians, `_linear_algebra.py` for left/right multiplication matrices, `AlgebraNetwork` for full-network depth sweeps, and `_initialization.py` for per-algebra weight initialization. The primary new code artifacts are: (1) `scripts/analyze_stability.py` -- a comprehensive analysis script, (2) `StabilizingNorm` -- a reusable nn.Module in `src/octonion/baselines/`, and (3) `tests/test_numerical_stability.py` -- smoke tests for the measurement infrastructure.

**Primary recommendation:** Build the analysis script in four clearly separated sections matching the four success criteria, reuse existing infrastructure maximally, and implement `StabilizingNorm` as a proper nn.Module that downstream phases can toggle via config flag.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Error Accumulation Experiment Design:**
- TWO separate experiments per depth: (1) stripped-down OctonionLinear chain (no BN/activation), (2) full AlgebraNetwork with BN and activations
- Depths: 10, 50, 100, 500 layers across both setups
- All four algebras (R, C, H, O) at each depth
- Fresh random weights per layer using Phase 3 initialization conventions
- Weight initialization: Kaiming/He (R), Trabelsi (C), Parcollet (H), unit-norm (O)

**Error Accumulation Measurement:**
- TWO metrics per depth: (1) relative error float32 vs float64, (2) norm drift ||ab|| = ||a||*||b|| invariant
- 100-1000 random inputs per (depth, algebra); report mean +/- std
- THREE input magnitude regimes: near-unit (||x|| ~ 1), small (||x|| ~ 0.01), large (||x|| ~ 100)

**Condition Number Characterization:**
- Scope: primitive operations (mul, inv, exp, log), N-layer compositions (2, 5, 10), full forward pass including BN and Conv
- Method: numeric Jacobian via finite differences (reuse `_numeric.py`)
- Input dimension: small fixed (8 or 16 octonionic units = 64-128 real inputs)
- All four algebras compared side-by-side as condition number vs input magnitude curves

**Stability Threshold:**
- "Stable" = relative error (float32 vs float64) < 1e-3
- "Stable depth" = max layers before error crosses 1e-3
- SC-4 "2x" = mitigation doubles the stable depth (layer count)
- Threshold applied to BOTH stripped-down and full AlgebraNetwork; results separate
- Precision scope: float32 vs float64 only (no bfloat16/float16)

**Mitigation Strategy:**
- Strategy: periodic re-normalization projecting layer outputs toward unit norm every K layers
- Implementation: `StabilizingNorm` nn.Module in `src/octonion/baselines/`
- Parameterized: `normalize_every=K` (default=10)
- Algebra coverage: all four, each using native norm
- K sweep: {5, 10, 20}
- Reusable by downstream phases via config flag

**Output and Delivery:**
- Single script: `scripts/analyze_stability.py` covering all four SCs
- Outputs: matplotlib plots, JSON data files, printed summary table
- Pattern: follows `scripts/demo_naive_vs_correct.py`
- pytest suite: `tests/test_numerical_stability.py` -- smoke tests, NOT pass/fail on SC values

### Claude's Discretion

- Exact number of random inputs within 100-1000 range per measurement
- JSON output schema and file naming
- Plot styling and layout
- K sweep values within condition number composition chains (2, 5, 10 or similar)
- Whether to include a printed summary table in addition to JSON/plots

### Deferred Ideas (OUT OF SCOPE)

None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FOUND-03 | Numerical stability analysis characterizes precision degradation across forward pass depths (10, 50, 100, 500 layers), measures condition numbers of octonionic operations, compares float32 vs float64 convergence, and identifies mitigation strategies | SC-1: depth sweep with error curves; SC-2: condition number characterization via numeric Jacobian + SVD; SC-3: float32/float64 comparison is the primary metric in SC-1; SC-4: StabilizingNorm periodic re-normalization demonstrated to extend stable depth 2x |
</phase_requirements>

## Standard Stack

### Core (Already in project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.9.1 | All tensor operations, linalg, SVD, condition numbers | Project's ML framework; already in container |
| matplotlib | >=3.10.8 | Plot generation (depth-vs-error curves, condition number plots) | Already a project dependency |
| numpy | >=1.26 | Statistical computations (mean, std, percentiles) | Already in dev dependencies |

### Supporting (Already in project)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy | >=1.17.1 | Potential curve fitting for error growth models | Already a project dependency; use if fitting exponential/polynomial error growth curves |
| json (stdlib) | N/A | Structured data output | Always -- results saved as JSON for downstream phases |
| seaborn | >=0.13.2 | Enhanced plot styling | Already a dependency; use for publication-quality figures |

### No New Dependencies Needed

All required functionality is available through existing project dependencies. No `npm install` or `uv add` commands needed.

**Key PyTorch APIs for this phase:**
- `torch.linalg.svd()` / `torch.linalg.svdvals()` -- singular values for condition number computation
- `torch.linalg.cond()` -- direct condition number (supports 2-norm, Frobenius, inf, etc.)
- `torch.linalg.norm()` -- matrix and vector norms
- `torch.isfinite()` -- detecting NaN/Inf before they propagate
- `torch.no_grad()` -- all measurements are inference-only

## Architecture Patterns

### Recommended Project Structure

New files (3 total):
```
src/octonion/baselines/
  _stabilization.py          # StabilizingNorm nn.Module
scripts/
  analyze_stability.py       # Comprehensive analysis script (all 4 SCs)
tests/
  test_numerical_stability.py # Smoke tests for measurement infrastructure
```

Modified files (1):
```
src/octonion/baselines/_config.py  # Add stabilize_every: int | None to NetworkConfig
```

### Pattern 1: Dual-Dtype Forward Pass Comparison (SC-1 / SC-3)

**What:** Run the same network architecture at both float32 and float64, compare outputs.
**When to use:** Error accumulation measurement at each depth.
**Example:**
```python
# Build identical networks at both dtypes, share initialization
def measure_error_at_depth(
    algebra: AlgebraType,
    depth: int,
    input_magnitude: float,
    n_samples: int = 500,
    seed: int = 42,
) -> dict:
    """Measure float32-vs-float64 relative error for a stripped chain."""
    torch.manual_seed(seed)

    # Build layers (each independently initialized)
    layers_f64 = []
    for _ in range(depth):
        layer = build_algebra_linear(algebra, hidden, hidden, dtype=torch.float64)
        layers_f64.append(layer)

    # Clone to float32
    layers_f32 = [copy_layer_to_dtype(l, torch.float32) for l in layers_f64]

    errors = []
    for _ in range(n_samples):
        x = torch.randn(batch, hidden, algebra.dim, dtype=torch.float64) * input_magnitude
        x32 = x.float()

        # Forward pass
        out64 = forward_chain(layers_f64, x)
        out32 = forward_chain(layers_f32, x32)

        # Relative error
        rel_err = (out32.double() - out64).norm() / (out64.norm() + 1e-30)
        errors.append(rel_err.item())

    return {"mean": np.mean(errors), "std": np.std(errors)}
```

### Pattern 2: Condition Number via Numeric Jacobian + SVD (SC-2)

**What:** Compute the condition number of a function by building its full Jacobian via finite differences, then taking sigma_max / sigma_min from SVD.
**When to use:** Condition number characterization of primitive ops and composed networks.
**Example:**
```python
from octonion.calculus._numeric import numeric_jacobian

def condition_number(fn, x, eps=1e-7):
    """Compute condition number of fn at x via numeric Jacobian."""
    J = numeric_jacobian(fn, x, eps=eps)  # [..., m, n]
    # Use svdvals for numerical stability (no gradient needed)
    sv = torch.linalg.svdvals(J)  # [..., min(m,n)]
    sigma_max = sv[..., 0]
    sigma_min = sv[..., -1]
    # Clamp sigma_min to avoid division by zero
    return sigma_max / sigma_min.clamp(min=1e-30)
```

**Important:** For network-level condition numbers, the Jacobian is computed over the full flattened input/output. With 8 or 16 octonionic units, the Jacobian is 64x64 or 128x128 -- tractable for SVD. Keep input dimension small as specified in CONTEXT.md.

### Pattern 3: StabilizingNorm Module

**What:** Periodic re-normalization that projects algebra-valued activations toward unit norm.
**When to use:** Inserted every K layers in a chain to prevent error accumulation.
**Example:**
```python
class StabilizingNorm(nn.Module):
    """Periodic unit-norm re-normalization for algebra-valued activations.

    Projects activations so each algebra element has unit norm,
    preventing unbounded growth or collapse through deep chains.
    Inserted every `normalize_every` layers via AlgebraNetwork config.
    """

    def __init__(self, algebra_dim: int, eps: float = 1e-8):
        super().__init__()
        self.algebra_dim = algebra_dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each algebra element to unit norm.

        Args:
            x: [..., features, algebra_dim] for hypercomplex,
               [..., features] for real.
        """
        if self.algebra_dim == 1:
            # Real: normalize per-feature
            norm = x.abs().clamp(min=self.eps)
            return x / norm
        else:
            # Hypercomplex: normalize along algebra dimension
            norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
            return x / norm
```

### Pattern 4: Norm Drift Measurement

**What:** Test the algebraic invariant ||ab|| = ||a||*||b|| after many chained operations.
**When to use:** Measuring how well the norm multiplicative property is preserved through deep chains.
**Example:**
```python
def measure_norm_drift(layers, x):
    """Measure deviation from ||f(x)|| = product of per-layer norms."""
    expected_norm = x.norm(dim=-1)  # initial norm

    h = x
    for layer in layers:
        # Track what norm multiplication property would predict
        w_norm = compute_weight_norm(layer)
        expected_norm = expected_norm * w_norm
        h = layer(h)

    actual_norm = h.norm(dim=-1)
    drift = (actual_norm - expected_norm).abs() / (expected_norm + 1e-30)
    return drift
```

**Note:** For the stripped-down chain (no BN, no activation), the multiplicative norm property ||Wx|| should relate to ||W|| * ||x|| via the operator norm. For full AlgebraNetwork with BN + activations, the relationship is more complex and the measurement reveals how these components affect norm dynamics.

### Anti-Patterns to Avoid

- **Computing Jacobian on full-size networks:** Even with 64 hidden units, the full network Jacobian would be 512x512 = 262K entries at float64. Keep to 8-16 octonionic units (64-128 real dims) as specified.
- **Using autograd for Jacobian:** The numeric Jacobian from `_numeric.py` is the right tool here -- it works in inference mode without requiring the autograd graph, and is the established ground-truth method in this project.
- **Running measurements in training mode:** All measurements should use `model.eval()` and `torch.no_grad()`. BN running stats must be pre-populated (a few warmup passes in train mode) before eval-mode measurements.
- **Testing SC thresholds in pytest:** The CONTEXT.md explicitly says tests are smoke tests only -- do NOT assert that stable depth doubles. That is an experimental outcome.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Condition number computation | Custom SVD + ratio | `torch.linalg.svdvals()` then ratio, or `torch.linalg.cond()` directly | Handles edge cases (zero singular values, batching) correctly |
| Finite-difference Jacobian | New implementation | `octonion.calculus._numeric.numeric_jacobian()` | Already validated in Phase 2, known-good eps=1e-7 |
| Per-algebra initialization | Custom init code | `octonion.baselines._initialization` functions | Phase 3 established these; consistency required |
| Network construction | Manual layer stacking | `AlgebraNetwork(config)` for full network; manual for stripped chain | Full network already handles all algebra types correctly |
| Algebra-specific norm | Per-type if/elif | `x.norm(dim=-1)` for tensors (works for all algebras since norms are Euclidean over components) | PyTorch's norm handles all dims uniformly |
| JSON serialization of numpy | Custom encoder | `json.dump(data, f, indent=2, default=float)` | Handles numpy floats cleanly |

**Key insight:** This phase is primarily an analysis/measurement phase. The only truly new code is `StabilizingNorm` and the analysis script. Everything else reuses Phase 1-3 infrastructure.

## Common Pitfalls

### Pitfall 1: Float32 Weight Initialization Diverging from Float64

**What goes wrong:** Building float64 layers, then converting weights to float32 introduces initial rounding that compounds through the chain. This is NOT the same as initializing in float32 directly.
**Why it happens:** Random weight distributions at float64 precision have more entropy than float32 can represent; casting truncates.
**How to avoid:** Initialize at float64, then explicitly cast to float32 via `.to(torch.float32)`. The float32 network receives the float64 weights rounded to float32 -- this IS the correct methodology for measuring "how much precision does float32 lose relative to float64". Both networks see "the same" weights (within float32 precision).
**Warning signs:** If relative error at depth=1 is already > 1e-7, the casting is introducing spurious error.

### Pitfall 2: BN Running Stats Not Populated for Eval-Mode Measurement

**What goes wrong:** Measuring full AlgebraNetwork in eval mode without first populating BN running statistics gives identity-like normalization (running_mean=0, running_var=I), which is unrealistic.
**Why it happens:** BN layers initialize running stats to mean=0, cov=I. Without training-mode passes, eval mode uses these defaults.
**How to avoid:** Do a warmup phase: run 10-50 batches through the network in training mode (with `torch.no_grad()` for speed) to populate running statistics, then switch to eval mode for measurements.
**Warning signs:** BN condition numbers are all exactly 1.0; BN has no effect on outputs.

### Pitfall 3: Numeric Jacobian Overflow at Large Input Magnitudes

**What goes wrong:** When inputs have ||x|| ~ 100, the Jacobian elements can be very large, causing the numeric finite-difference to lose relative precision.
**Why it happens:** The finite-difference step eps=1e-7 is tuned for O(1) inputs. For ||x||=100, the relative perturbation is 1e-9, which may be below float32 rounding threshold.
**How to avoid:** Scale eps proportionally to input magnitude. For large inputs, use eps = 1e-7 * max(1.0, ||x||). For the Jacobian computation, always use float64 (the established pattern in `_numeric.py`).
**Warning signs:** Condition numbers at large magnitude showing nonsensical spikes or drops.

### Pitfall 4: Cholesky Failure in OctonionBN at Extreme Depths

**What goes wrong:** At depth 500, the covariance matrix computed by OctonionBN may become degenerate (all activations collapsed to near-zero or all-same values), causing Cholesky decomposition to fail.
**Why it happens:** Without mitigation, deep chains either explode or vanish, leading to rank-deficient covariance matrices.
**How to avoid:** The existing `_whiten()` method has a three-level fallback (primary -> regularized -> identity). For the full AlgebraNetwork experiments, the BN fallback should handle this. For the stripped chain, BN is not used, so this is not an issue.
**Warning signs:** `last_cond` buffer reporting very large values (>1e6); error curves showing sudden jumps.

### Pitfall 5: Norm Drift Metric Assumes Specific Layer Structure

**What goes wrong:** The norm multiplicative property ||ab|| = ||a||*||b|| holds for individual octonion multiplications but not for the full linear layer operation (which involves sums of products via structure constants).
**Why it happens:** OctonionDenseLinear computes output_k = sum_{i,j} C[i,j,k] * W_i * x_j, which is a sum of many octonionic multiplications. The norm of a sum is not the sum of norms.
**How to avoid:** For the stripped chain, measure norm drift as: how does ||output|| compare to ||input|| * (some expected scaling factor based on weight norms). For the full network, just track ||output||/||input|| ratio across depths as a growth/decay indicator.
**Warning signs:** Norm drift metric shows > 100% deviation even at depth=1.

### Pitfall 6: Docker Container Required for All Python Execution

**What goes wrong:** Running analysis scripts or tests on the host machine fails because PyTorch/ROCm is only in the container.
**Why it happens:** Project rule: all Python runs inside `docker compose run --rm dev`.
**How to avoid:** Always prefix: `docker compose run --rm dev uv run python scripts/analyze_stability.py`.
**Warning signs:** ImportError for torch, or CUDA not available.

## Code Examples

### Example 1: Building a Stripped-Down Chain (No BN, No Activation)

```python
# Source: Derived from existing _algebra_linear.py and _initialization.py patterns
from octonion.baselines._algebra_linear import (
    RealLinear, ComplexLinear, QuaternionLinear, OctonionDenseLinear,
)
from octonion.baselines._config import AlgebraType

ALGEBRA_LINEAR = {
    AlgebraType.REAL: RealLinear,
    AlgebraType.COMPLEX: ComplexLinear,
    AlgebraType.QUATERNION: QuaternionLinear,
    AlgebraType.OCTONION: OctonionDenseLinear,
}

def build_stripped_chain(
    algebra: AlgebraType,
    depth: int,
    hidden: int,
    dtype: torch.dtype = torch.float64,
) -> nn.ModuleList:
    """Build a chain of algebra-specific linear layers without BN or activations."""
    LinearClass = ALGEBRA_LINEAR[algebra]
    layers = nn.ModuleList()
    for _ in range(depth):
        layer = LinearClass(hidden, hidden, bias=False, dtype=dtype)
        layers.append(layer)
    return layers

def forward_stripped_chain(
    layers: nn.ModuleList,
    x: torch.Tensor,
) -> torch.Tensor:
    """Forward pass through stripped chain."""
    h = x
    for layer in layers:
        h = layer(h)
    return h
```

### Example 2: Condition Number Sweep Over Input Magnitudes

```python
# Source: Combines _numeric.py with torch.linalg.svdvals
from octonion.calculus._numeric import numeric_jacobian
import torch

def condition_number_sweep(
    fn,
    magnitudes: list[float],
    n_samples: int = 100,
    input_dim: int = 8,
    seed: int = 42,
) -> dict[float, dict[str, float]]:
    """Sweep condition numbers across input magnitudes."""
    results = {}
    for mag in magnitudes:
        conds = []
        for i in range(n_samples):
            torch.manual_seed(seed + i)
            x = torch.randn(input_dim, dtype=torch.float64)
            x = x / x.norm() * mag  # Normalize to target magnitude

            J = numeric_jacobian(fn, x, eps=1e-7 * max(1.0, mag))
            sv = torch.linalg.svdvals(J)
            cond = (sv[0] / sv[-1].clamp(min=1e-30)).item()
            conds.append(cond)

        results[mag] = {
            "mean": float(np.mean(conds)),
            "std": float(np.std(conds)),
            "median": float(np.median(conds)),
            "max": float(np.max(conds)),
        }
    return results
```

### Example 3: StabilizingNorm Integration with Stripped Chain

```python
def forward_stabilized_chain(
    layers: nn.ModuleList,
    x: torch.Tensor,
    stabilizer: StabilizingNorm,
    normalize_every: int = 10,
) -> torch.Tensor:
    """Forward pass with periodic re-normalization."""
    h = x
    for i, layer in enumerate(layers):
        h = layer(h)
        if (i + 1) % normalize_every == 0:
            h = stabilizer(h)
    return h
```

### Example 4: Full AlgebraNetwork Depth Sweep

```python
# Source: Reusing AlgebraNetwork from _network.py
from octonion.baselines._config import AlgebraType, NetworkConfig
from octonion.baselines._network import AlgebraNetwork

def build_full_network(
    algebra: AlgebraType,
    depth: int,
    hidden: int = 8,
    dtype: torch.dtype = torch.float64,
) -> AlgebraNetwork:
    """Build full AlgebraNetwork for depth sweep measurement."""
    config = NetworkConfig(
        algebra=algebra,
        topology="mlp",
        depth=depth,
        base_hidden=hidden,
        activation="split_relu",
        output_projection="flatten",
        use_batchnorm=True,
        input_dim=hidden * algebra.dim,  # match algebra-valued input
        output_dim=hidden * algebra.dim,  # preserve full output
    )
    model = AlgebraNetwork(config).to(dtype)
    return model
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-component BN | Covariance-whitening BN (Cholesky) | Gaudet & Maida 2018 | This project already uses whitening BN; stability analysis must account for Cholesky overhead and potential failures |
| Ignore non-associativity | Explicit parenthesization in chain rule | Phase 2 (project-specific) | Condition number analysis of composed operations must use correct evaluation order |
| float16/bfloat16 for speed | float32 minimum for hypercomplex BN | Phase 3 discovery (AMP fails) | Float32 vs float64 is the right comparison scope; float16 already ruled out |
| torch.svd (deprecated) | torch.linalg.svdvals / torch.linalg.svd | PyTorch 1.9+ | Use new API; svdvals preferred when only singular values needed (more stable gradients, though we don't need gradients here) |
| torch.cond (nonexistent) | torch.linalg.cond() | PyTorch 1.9+ | Direct condition number computation available |

**Deprecated/outdated:**
- `torch.svd()`: Deprecated in favor of `torch.linalg.svd()`. Use `torch.linalg.svdvals()` for condition numbers (more numerically stable).
- AMP (float16) for hypercomplex: Phase 3 established that float16 Cholesky fails for OctonionBN. This phase's scope is float32 vs float64 only.

## Error Growth Theory

### Expected Error Accumulation Patterns (Hypothesis to Validate)

**Linear error growth** is the standard IEEE 754 result for a chain of N floating-point multiplications: accumulated relative error ~ N * machine_epsilon. For float32 (epsilon ~ 1.2e-7), a chain of 500 operations would predict relative error ~ 6e-5, well below the 1e-3 stability threshold.

**However, for matrix operations (linear layers):** Each layer involves multiple multiplications and additions. For an 8x8 matrix-vector product (octonionic linear), each output element involves 8 multiplications and 7 additions = 15 operations. A 500-layer chain thus involves ~7500 elementary operations. Expected error ~ 7500 * 1.2e-7 ~ 9e-4, approaching the 1e-3 threshold.

**Octonion-specific concerns:**
1. **Structure constant multiplication:** OctonionDenseLinear involves 64 nonzero structure constant entries. Each output involves many fused multiply-add operations through C[i,j,k] * W_i * x_j sums.
2. **Non-commutativity amplification:** Since ab != ba, the error in computing a product depends on the specific operand ordering, potentially creating correlated rather than independent rounding errors.
3. **Norm property interaction:** The norm multiplicative property ||ab|| = ||a||*||b|| may help self-correct errors (if error pushes norm away from expected, subsequent operations don't compound it) or may amplify them (if the structure constants create systematic drift in one direction).

These are hypotheses -- the analysis script will produce empirical data to validate or refute them.

### Condition Number Expectations

For octonion multiplication by a unit-norm element, the left multiplication matrix L_a has all singular values equal to ||a|| (since ||L_a x|| = ||ax|| = ||a||*||x||). This means **condition number = 1 for unit-norm multiplication** -- ideal conditioning. As ||a|| deviates from 1, the condition number should still be 1 (since all singular values scale uniformly).

However, for a **chain of multiplications with different elements**, the composed matrix L_{a_1} * L_{a_2} * ... * L_{a_n} can have condition number > 1 because the individual multiplication matrices, while individually well-conditioned, may not commute, and their composition can amplify certain directions.

For **inversion** (`conj(x)/||x||^2`), the condition number scales as ||x||^2 / ||x||^4 * ||x||^2 = 1 for unit norm, but grows for small ||x|| (amplifies small inputs) and large ||x|| (compresses large inputs).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (via Docker: `docker compose run --rm dev uv run pytest`) |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `docker compose run --rm dev uv run pytest tests/test_numerical_stability.py -v` |
| Full suite command | `docker compose run --rm dev uv run pytest tests/ -v --timeout=300` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FOUND-03 SC-1 | Depth sweep measurement infrastructure runs without error | smoke | `docker compose run --rm dev uv run pytest tests/test_numerical_stability.py::test_depth_sweep_smoke -x` | Wave 0 |
| FOUND-03 SC-2 | Condition number computation produces finite values for all ops | smoke | `docker compose run --rm dev uv run pytest tests/test_numerical_stability.py::test_condition_number_smoke -x` | Wave 0 |
| FOUND-03 SC-3 | Float32 vs float64 comparison produces meaningful differences | smoke | `docker compose run --rm dev uv run pytest tests/test_numerical_stability.py::test_dtype_comparison_smoke -x` | Wave 0 |
| FOUND-03 SC-4 | StabilizingNorm module forward pass runs for all algebras | unit | `docker compose run --rm dev uv run pytest tests/test_numerical_stability.py::test_stabilizing_norm -x` | Wave 0 |
| FOUND-03 SC-4 | StabilizingNorm output has unit norm | unit | `docker compose run --rm dev uv run pytest tests/test_numerical_stability.py::test_stabilizing_norm_output_norm -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `docker compose run --rm dev uv run pytest tests/test_numerical_stability.py -x`
- **Per wave merge:** `docker compose run --rm dev uv run pytest tests/ -v --timeout=300`
- **Phase gate:** Full suite green + `scripts/analyze_stability.py` runs to completion

### Wave 0 Gaps
- [ ] `tests/test_numerical_stability.py` -- smoke tests for all measurement infrastructure (covers FOUND-03 SCs 1-4)
- [ ] `src/octonion/baselines/_stabilization.py` -- StabilizingNorm module (must exist before tests)

No new test framework or config changes needed -- existing pytest infrastructure is sufficient.

## Open Questions

1. **Optimal number of random samples per measurement**
   - What we know: CONTEXT.md specifies 100-1000 range. Phase 2 demo used 1000 trials.
   - What's unclear: Whether 500 is sufficient for tight confidence intervals at depth=500, or whether deeper networks need more samples due to higher variance.
   - Recommendation: Start with 500 samples. If std/mean > 0.5 for any measurement, increase to 1000. Report the actual count used in JSON output.

2. **Whether BN whitening acts as implicit stabilization**
   - What we know: CONTEXT.md notes this as a hypothesis. Phase 3 showed OctonionBN's 8D Cholesky is expensive (131ms/forward at depth=28).
   - What's unclear: Quantitative comparison of error at depth=100 with vs without BN.
   - Recommendation: The dual-experiment design (stripped vs full) directly answers this. The results will show whether BN provides stability benefit worth the computational cost.

3. **StabilizingNorm interaction with BN**
   - What we know: StabilizingNorm projects to unit norm; BN whitens to zero-mean, unit-covariance. These are complementary.
   - What's unclear: Whether applying both creates redundancy or interference.
   - Recommendation: In the mitigation demonstration (SC-4), test StabilizingNorm on the stripped chain (primary demonstration) and optionally on the full network (secondary). The stripped chain is the clean test; BN interaction is an observation.

4. **eps scaling for large-magnitude numeric Jacobian**
   - What we know: Default eps=1e-7 is optimal for O(1) inputs at float64.
   - What's unclear: Exact scaling formula for ||x||=100 inputs.
   - Recommendation: Use `eps = 1e-7 * max(1.0, x.norm().item())`. This keeps the relative perturbation constant. Validate by checking that condition numbers are smooth across magnitude transitions.

## Sources

### Primary (HIGH confidence)
- PyTorch numerical accuracy documentation: https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html -- float32/64 precision characteristics, non-associativity of floating-point ops
- `torch.linalg.svdvals` documentation: https://docs.pytorch.org/docs/stable/generated/torch.linalg.svdvals.html -- condition number computation API
- `torch.linalg.cond` documentation: https://docs.pytorch.org/docs/stable/generated/torch.linalg.cond.html -- direct condition number, supports multiple norm types
- Project codebase: `src/octonion/calculus/_numeric.py` -- validated finite-difference Jacobian (Phase 2)
- Project codebase: `src/octonion/baselines/_network.py` -- AlgebraNetwork with configurable depth
- Project codebase: `src/octonion/baselines/_normalization.py` -- BN whitening with Cholesky fallback

### Secondary (MEDIUM confidence)
- IEEE 754 floating-point error propagation: linear error growth in multiplication chains ~ N * epsilon (standard numerical analysis result, verified via https://floating-point-gui.de/errors/propagation/)
- Norm-preservation in residual networks (https://arxiv.org/pdf/1805.07477): skip connections preserve norm, relevant to understanding why stripped chains (no residuals) may be less stable
- Phase 3 decisions (CONTEXT.md): AMP disabled for octonion (float16 Cholesky failure), OctonionBN 131ms/forward at depth=28

### Tertiary (LOW confidence)
- Hypercomplex neural networks survey (PMC 2025): mentions stability as primary concern for octonion networks, but no quantitative benchmarks for error accumulation -- this phase will produce the first such benchmarks for this project

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all tools already in project, APIs verified via official PyTorch docs
- Architecture: HIGH -- follows established project patterns (demo scripts, AlgebraNetwork, _numeric.py)
- Pitfalls: HIGH -- based on concrete project experience (Phase 3 Cholesky failures, AMP issues) and standard numerical analysis
- Error theory: MEDIUM -- growth rate predictions are standard but hypercomplex-specific behavior is untested (that's the point of this phase)
- StabilizingNorm design: HIGH -- simple module following NormPreservingActivation pattern already in codebase

**Research date:** 2026-03-19
**Valid until:** 2026-04-19 (stable domain -- numerical analysis fundamentals don't change; PyTorch APIs stable)
