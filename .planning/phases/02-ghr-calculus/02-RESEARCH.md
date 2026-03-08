# Phase 2: GHR Calculus - Research

**Researched:** 2026-03-08
**Domain:** Octonionic differentiation, custom autograd, non-associative chain rule
**Confidence:** MEDIUM (novel mathematical territory; implementation patterns are well-understood but the GHR octonionic extension is genuinely new)

## Summary

Phase 2 requires implementing correct gradient computation for octonionic operations. The core challenge is that GHR (Generalized Hamilton-Real) calculus was developed for quaternions (associative, non-commutative) and extending it to octonions (non-associative, non-commutative) is genuinely novel. The quaternionic GHR framework relies on quaternion rotations q_mu = mu * q * mu^{-1} at its core -- this rotation is well-defined because quaternion multiplication is associative. For octonions, the expression mu * q * mu^{-1} requires specifying parenthesization: (mu * q) * mu^{-1} vs mu * (q * mu^{-1}), and these may differ.

The user has decided on a **native octonionic derivation** (not reducing to real components a la Parcollet), with **no fallback** -- either solve the GHR extension or produce a formal barrier document. The practical implementation path is clear: use PyTorch's `torch.autograd.Function` with custom backward passes for each octonionic primitive, verified against finite-difference Jacobians. The mathematical challenge is deriving the correct derivative formulas that account for non-associativity in the chain rule.

**Primary recommendation:** Implement gradients via explicit 8x8 real Jacobian matrices for each primitive operation (using the existing `left_mul_matrix`/`right_mul_matrix` infrastructure), wrapped in `torch.autograd.Function` subclasses. Derive the GHR octonionic extension by treating each operation's Jacobian as a linear map on R^8, which sidesteps the non-associativity problem at the single-operation level. The non-associativity challenge manifests in the chain rule for compositions, which must be parenthesization-aware -- the Jacobian of a composition depends on which binary tree structure was used.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- Full GHR formalism -- not just correct autograd, but the complete octonionic extension of HR calculus
- Native octonionic derivation -- derive GHR derivatives directly in octonionic algebra, not by reducing to real components (Parcollet et al. extension pattern rejected in favor of native approach)
- User-facing API exposed in `octonion.calculus` submodule: `from octonion.calculus import ghr_derivative, conjugate_derivative, jacobian`
- All operations covered: mul, exp, log, conjugate, inverse, inner product, cross product -- not just multiplication
- Both Wirtinger derivatives tracked: full (df/do, df/do*) pair, not just the conjugate gradient
- Higher-order derivative support: autograd.Function implementations support create_graph=True for Hessian computation (Phase 5 needs eigenspectrum analysis)
- Explicit Jacobian matrices: each operation provides analytic 8x8 real Jacobians AND numeric (finite-difference) Jacobians -- triple-check: analytic vs numeric vs autograd
- Composite operation gradients built via chain rule from primitive operation gradients (not fused composite derivations)
- Formal derivations included in module docstrings showing the Wirtinger derivative derivation for each operation
- Custom octonion-aware gradcheck utility (not standard torch.gradcheck) -- tests both Wirtinger derivatives, reports per-component errors, validates parenthesization correctness
- Octonionic analyticity tests: define CR-like conditions, test which operations satisfy them
- Learning rate scaling heuristic based on GHR gradient magnitude characteristics
- If GHR derivation reveals the correct number of derivative components differs from 2 (the complex Wirtinger pair), follow the math -- accommodate whatever the algebra requires
- Arbitrary parenthesization supported via computation graph -- whatever order the user writes gets tracked and differentiated correctly
- Parenthesization inspector utility: text-based tree output (ASCII/Unicode) for debugging gradient flow in complex compositions
- No canonical parenthesization forced -- users can write any association pattern
- No fallback -- solve the GHR extension or prove it impossible. This is the research contribution
- No time limit on research exploration
- All-or-nothing for operation coverage -- don't ship partial GHR
- If GHR proves mathematically impossible: re-scope to what's possible, but produce a formal barrier document explaining what was attempted, where it breaks, why, and implications for the thesis
- The 4 success criteria remain unchanged regardless of implementation approach -- they test gradient correctness, not GHR specifically
- Exhaustive parenthesization testing: all 14 Catalan(4) patterns at depth 5
- Both OctonionLinear layers AND raw octonion_mul chains tested
- Mixed operations in compositions: mul + exp + log + conjugate + inverse interleaved
- Public CompositionBuilder API using tree data structures (nested Python objects, not string parsing)
- Enumeration utility: `all_parenthesizations(n)` generates all C_{n-1} binary tree structures
- Configurable depth in test suite (5 is the success criterion default, but tests can run at any depth)
- Quantitative report: per-pattern gradient error (max/mean/std vs finite-difference) + maximum gradient difference BETWEEN different parenthesizations of same operands
- Results saved as structured file (JSON/CSV) AND printed to stdout; tree structures included in results file
- Standalone script: `scripts/demo_naive_vs_correct.py` (not part of pytest suite, explicit invocation only)
- Full quantification: relative error magnitude, direction cosine similarity, per-component divergence analysis
- 1000 random inputs with confidence intervals for statistical rigor
- Depth scaling analysis: report naive-vs-correct error as function of depth (2, 3, 5, 7, 10) to show compounding non-associativity effects
- Raw data output only (JSON/CSV) -- no LaTeX formatting, thesis presentation handled separately
- Manual GPU verification only -- not part of CI pipeline

### Claude's Discretion
- Definition of "naive" chain rule for the demonstration (whatever most clearly shows non-associativity problem)
- Whether to include a training comparison (naive vs correct gradients on toy task) in the demo script
- GPU/CPU parity tolerance value
- Mixed-operation ordering strategy in composition tests
- Internal module organization within octonion.calculus

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FOUND-02 | GHR calculus gradient implementation computes octonionic backpropagation gradients that match finite-difference approximation to within numerical precision, with explicit parenthesization-aware chain rule handling non-associativity | All research findings below: GHR extension theory, autograd.Function patterns, Jacobian-based gradient computation, parenthesization tracking via binary trees, finite-difference verification |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch (torch.autograd) | 2.9.1 | Custom backward pass via autograd.Function | Already in project; native gradient infrastructure |
| PyTorch (torch.autograd.gradcheck) | 2.9.1 | Reference for finite-difference gradient verification | Built-in, well-tested central difference implementation |
| pytest | latest | Test framework | Already in project |
| hypothesis | >=6.0 | Property-based testing for gradient properties | Already in project |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | >=1.26 | Already a dev dependency; useful for Catalan number generation, tree enumeration | Binary tree generation, report data processing |
| json (stdlib) | N/A | Structured output for quantitative reports | Saving parenthesization reports and demo results |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom autograd.Function | PyTorch native autograd (einsum is differentiable) | Native autograd gives correct gradients but treats octonion as R^8 vector; doesn't track parenthesization or provide Wirtinger decomposition. Custom Functions needed for GHR formalism |
| Custom gradcheck | torch.autograd.gradcheck | Standard gradcheck works on R^8 tensors but doesn't understand octonionic Wirtinger pairs, per-component octonionic errors, or parenthesization validation |
| Native GHR derivation | Pseudo-real-matrix representation (Octonion Phase Retrieval approach) | User explicitly rejected reduction to real components; native approach required |

**Installation:**
No new dependencies needed. All libraries already in project.

## Architecture Patterns

### Recommended Project Structure
```
src/octonion/
  calculus/
    __init__.py              # Public API: ghr_derivative, conjugate_derivative, jacobian
    _ghr.py                  # GHR derivative definitions and Wirtinger pair
    _jacobians.py            # Analytic 8x8 Jacobians for each primitive
    _autograd_functions.py   # torch.autograd.Function subclasses
    _chain_rule.py           # Parenthesization-aware chain rule
    _composition.py          # CompositionBuilder, BinaryTree, all_parenthesizations
    _gradcheck.py            # Custom octonion-aware gradient checking
    _analyticity.py          # CR-like analyticity conditions
    _inspector.py            # Parenthesization tree inspector (ASCII output)
```

### Pattern 1: Autograd Function for Each Primitive Operation
**What:** Each octonionic operation (mul, exp, log, conjugate, inverse, inner_product, cross_product) gets a `torch.autograd.Function` subclass that computes the forward pass and provides the analytic backward pass via its 8x8 Jacobian.
**When to use:** Every primitive octonionic operation that participates in gradient computation.
**Example:**
```python
# Source: PyTorch docs + project structure constants
class OctonionMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(a, b)
        return octonion_mul(a, b)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = ctx.saved_tensors
        # For f(a, b) = a * b:
        # df/da: Left-multiplying grad by R_b^T (right multiplication matrix of b, transposed)
        # df/db: Left-multiplying grad by L_a^T (left multiplication matrix of a, transposed)
        C = STRUCTURE_CONSTANTS.to(device=a.device, dtype=a.dtype)

        # Jacobian w.r.t. a: J_a[k, i] = sum_j C[i, j, k] * b[j] = R_b[k, i]
        # grad_a = grad_output @ R_b = grad_output @ J_a^T... careful with convention
        # Actually: grad_a[i] = sum_k grad_output[k] * J_a[k, i]
        #                     = sum_k grad_output[k] * sum_j C[i, j, k] * b[j]
        grad_a = torch.einsum("...k, ijk, ...j -> ...i", grad_output, C, b)

        # Jacobian w.r.t. b: J_b[k, j] = sum_i C[i, j, k] * a[i] = L_a[k, j]
        # grad_b[j] = sum_k grad_output[k] * J_b[k, j]
        grad_b = torch.einsum("...k, ijk, ...i -> ...j", grad_output, C, a)

        return grad_a, grad_b
```

### Pattern 2: Jacobian Triple-Check Architecture
**What:** Every operation provides three independent gradient computations: (1) analytic Jacobian from mathematical derivation, (2) numeric Jacobian via finite differences, (3) autograd backward pass. All three must agree.
**When to use:** Verification of every primitive operation.
**Example:**
```python
def analytic_jacobian_mul(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic 8x8 Jacobians for octonion_mul(a, b).

    Returns (J_a, J_b) where J_a[k, i] = d(a*b)_k / da_i, etc.
    """
    C = STRUCTURE_CONSTANTS.to(device=a.device, dtype=a.dtype)
    # J_a[k, i] = sum_j C[i, j, k] * b[j]  (this is R_b transposed indices)
    J_a = torch.einsum("ijk, ...j -> ...ki", C, b)
    # J_b[k, j] = sum_i C[i, j, k] * a[i]  (this is L_a transposed indices)
    J_b = torch.einsum("ijk, ...i -> ...kj", C, a)
    return J_a, J_b

def numeric_jacobian(fn, x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Finite-difference Jacobian J[k, i] = d f(x)_k / d x_i."""
    n = x.shape[-1]
    f0 = fn(x)
    m = f0.shape[-1]
    J = torch.zeros(*x.shape[:-1], m, n, dtype=x.dtype, device=x.device)
    for i in range(n):
        e = torch.zeros_like(x)
        e[..., i] = eps
        J[..., :, i] = (fn(x + e) - fn(x - e)) / (2 * eps)
    return J
```

### Pattern 3: Parenthesization-Aware Composition via Binary Trees
**What:** Represent multiplication chains as explicit binary trees. Each leaf is an operand, each internal node is an operation. The chain rule Jacobian is computed by traversing the tree bottom-up, multiplying Jacobians according to the tree structure.
**When to use:** Any composition of 3+ operations where associativity matters.
**Example:**
```python
from dataclasses import dataclass
from typing import Union

@dataclass(frozen=True)
class Leaf:
    """Terminal node: holds an operand index."""
    index: int

@dataclass(frozen=True)
class Node:
    """Internal node: binary operation applied to left and right subtrees."""
    op: str  # "mul", "exp", "log", etc.
    left: Union["Node", "Leaf"]
    right: Union["Node", "Leaf"]

def all_parenthesizations(n: int) -> list[Node]:
    """Generate all C_{n-1} binary tree structures for n operands.

    For n=4: C_3 = 5 trees (but with 4 operands in a product chain,
    we get 5 distinct parenthesizations).
    For n=5: C_4 = 14 trees.
    """
    if n == 1:
        return [Leaf(0)]
    results = []
    for split in range(1, n):
        for left in all_parenthesizations(split):
            for right in all_parenthesizations(n - split):
                # Reindex right subtree
                right_shifted = _shift_indices(right, split)
                results.append(Node("mul", left, right_shifted))
    return results
```

### Pattern 4: GHR Wirtinger Derivative Pair
**What:** For octonionic functions, track both df/do (the "full" derivative) and df/do* (the "conjugate" derivative), analogous to complex Wirtinger calculus. For a real-valued loss L and octonionic variable o, the gradient direction is dL/do* (conjugate derivative).
**When to use:** All GHR derivative computations; this is the formalism that makes the work a "GHR extension" rather than just "correct autograd."

**Mathematical Foundation:**
For quaternions, GHR defines:
```
df/dq_mu = (1/4)(df/dq_a - df/dq_b * i_mu - df/dq_c * j_mu - df/dq_d * k_mu)
df/dq_mu* = (1/4)(df/dq_a + df/dq_b * i_mu + df/dq_c * j_mu + df/dq_d * k_mu)
```

For octonions, the natural extension uses 8 partial derivatives (1 real + 7 imaginary):
```
df/do_mu = (1/8)(df/do_0 - sum_{k=1}^{7} df/do_k * e_k_mu)
df/do_mu* = (1/8)(df/do_0 + sum_{k=1}^{7} df/do_k * e_k_mu)
```
where e_k_mu denotes the basis element e_k in the mu-rotated frame.

**Key research question:** The quaternionic GHR uses the quaternion rotation q_mu = mu * q * mu^{-1} which requires associativity. For octonions, we need to define what "rotation" means. The octonionic automorphism group G2 (14-dimensional) is smaller than the naive rotation group SO(7) (21-dimensional), so the "general orthogonal system" for octonions may have different structure than for quaternions.

**Practical resolution:** For gradient computation of real-valued loss functions, Proposition 4.8 from the GHR paper shows that left and right derivatives coincide. This means for the optimization use case (real loss), the Wirtinger conjugate gradient is simply the R^8 gradient vector reinterpreted in octonionic notation. The GHR formalism adds value by:
1. Providing the derivative in octonionic form (8 components, not 8 separate reals)
2. Tracking both derivatives for non-real-valued intermediate computations
3. Enabling product and chain rules that respect octonionic structure

### Anti-Patterns to Avoid
- **Assuming associativity in chain rule:** The Jacobian of f(g(h(x))) depends on whether you compute (f . g) . h or f . (g . h). For standard R^n functions this doesn't matter (matrix multiplication IS associative), but when the operations themselves are non-associative (g is octonion_mul), the computation graph encodes different mathematical functions.
- **Using torch.autograd.gradcheck directly:** It works on R^8 tensors but doesn't validate octonionic structure. Custom gradcheck must verify both Wirtinger derivatives and parenthesization correctness.
- **Fusing composite operation Jacobians:** Each primitive must have its own Jacobian; composites use chain rule from primitives. This is explicitly required by user decision.
- **Computing gradients in no-grad mode for double backward:** If create_graph=True, the backward pass itself must be differentiable. Save only inputs/outputs (which have grad_fn), never intermediate computations from the forward pass (which runs in no-grad mode).
- **Breaking the immutable Octonion pattern:** All existing operations return new instances. Autograd Functions should operate on raw [..., 8] tensors internally, with Octonion wrapper methods calling the Function.apply().

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Finite-difference Jacobian | Custom central-difference loop | Adapt torch.autograd.gradcheck's approach (central difference, eps=1e-6 for float64) | Numerical stability, epsilon selection, complex function handling already solved |
| Binary tree enumeration | Ad-hoc recursive parenthesization | Standard Catalan number recursive generation | Well-known combinatorial algorithm; C_{n-1} for n operands |
| Einsum-based Jacobian computation | Manual loops over components | `torch.einsum` with structure constants tensor | Already used in project; vectorized, GPU-compatible |
| Parameter broadcasting | Manual expand/reshape | `expand_as` pattern from existing OctonionLinear | Established project pattern, handles arbitrary batch dims |

**Key insight:** The project already has the mathematical infrastructure (structure constants tensor, left/right multiplication matrices) needed for analytic Jacobians. The challenge is organizing the GHR formalism around this infrastructure, not building new tensor computation primitives.

## Common Pitfalls

### Pitfall 1: Confusing "parenthesization of multiplication" with "order of chain rule application"
**What goes wrong:** The Jacobian chain rule (J_total = J_outer @ J_inner) uses matrix multiplication, which IS associative. So the chain rule application order doesn't matter. But the operations being differentiated (octonion_mul) are NOT associative, so (a*b)*c and a*(b*c) are genuinely different functions with different Jacobians.
**Why it happens:** In standard neural networks, every operation is a function R^n -> R^m, and the chain rule is just matrix multiplication. With octonions, the same operands with different parenthesization produce different functions.
**How to avoid:** The computation graph must record which binary tree structure was used. The Jacobians themselves are 8x8 real matrices and compose associatively via standard matrix multiplication.
**Warning signs:** Getting the same gradient for all parenthesizations of the same operands.

### Pitfall 2: Non-associativity of rotation operation for GHR basis
**What goes wrong:** GHR calculus for quaternions uses the rotation q_mu = mu * q * mu^{-1}. For octonions, this expression is ambiguous: (mu * q) * mu^{-1} may differ from mu * (q * mu^{-1}).
**Why it happens:** Quaternion multiplication is associative; octonion multiplication is not.
**How to avoid:** Either: (a) fix a canonical parenthesization for the rotation (e.g., left-to-right: (mu * q) * mu^{-1}), or (b) use the Moufang identity R_mu(q) = (mu * q) * mu^{-1} = mu * (q * mu^{-1}) which DOES hold for octonions (alternativity means this particular expression IS well-defined when two of the three elements are the same -- and here mu appears twice). Actually, the Moufang identity states (xyx) is well-defined (independent of parenthesization) for alternative algebras. So R_mu(q) = mu * q * mu^{-1} is well-defined as a Moufang loop operation.
**Warning signs:** Getting different GHR derivatives depending on how you parenthesize the rotation.

### Pitfall 3: Epsilon selection for finite-difference Jacobians at float64
**What goes wrong:** Too large eps gives truncation error; too small eps gives roundoff error. The optimal eps for central differences at float64 is approximately eps = (machine_epsilon)^{1/3} approx 6e-6.
**Why it happens:** Central difference error is O(eps^2) + O(u/eps) where u is machine epsilon (~2.2e-16 for float64).
**How to avoid:** Use eps in range 1e-7 to 1e-5 for float64. The success criterion requires 1e-5 relative error, so eps=1e-7 with central differences gives O(1e-14) truncation error, well within budget.
**Warning signs:** Gradient errors that improve then worsen as eps decreases.

### Pitfall 4: create_graph=True and no-grad forward pass interaction
**What goes wrong:** PyTorch runs autograd.Function.forward() in no-grad mode. If you compute intermediate values in forward() and save them for backward(), those intermediates have no grad_fn. When backward() uses them with create_graph=True, the second-order graph is incomplete.
**Why it happens:** PyTorch design: forward is for computation, backward is for gradients.
**How to avoid:** Only save inputs and outputs in ctx.save_for_backward() -- these have grad_fn. If you need intermediates, either (a) recompute them in backward(), or (b) return them as additional outputs of forward().
**Warning signs:** gradgradcheck fails but gradcheck passes; second-order gradients are zero or incorrect.

### Pitfall 5: Batch dimension handling in Jacobian computation
**What goes wrong:** Jacobians are shape [..., 8, 8] for batched inputs [..., 8]. Einsum index collisions or incorrect contractions produce wrong batch broadcasting.
**Why it happens:** The structure constants tensor is [8, 8, 8] (no batch dims), while inputs are [..., 8].
**How to avoid:** Use the established project pattern: `C.to(device=a.device, dtype=a.dtype)` and `torch.einsum("...i, ijk, ...j -> ...k", ...)` with explicit batch ellipsis.
**Warning signs:** Jacobians that are correct for single inputs but wrong for batched inputs.

### Pitfall 6: GPU/CPU numerical parity at float64
**What goes wrong:** ROCm GPU float64 operations may use different instruction ordering than CPU, producing slightly different results due to floating-point non-associativity of addition.
**Why it happens:** GPU parallelism means reductions (sums) happen in different order than CPU sequential computation.
**How to avoid:** Use a reasonable tolerance (recommend 1e-12 relative error for GPU/CPU parity at float64). The structure constants are exact integers ({-1, 0, 1}) so the only source of difference is the order of accumulation in einsum.
**Warning signs:** Tests that pass on CPU but fail on GPU with errors around 1e-14 to 1e-15.

## Code Examples

### Example 1: Analytic Jacobian for octonion_mul using existing infrastructure
```python
# Source: Project's _linear_algebra.py + _multiplication.py
def jacobian_mul_wrt_a(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """8x8 Jacobian of octonion_mul(a, b) w.r.t. a.

    J[k, i] = d(a*b)_k / da_i = sum_j C[i, j, k] * b[j]

    This is the transpose of the right multiplication matrix R_b.
    """
    C = STRUCTURE_CONSTANTS.to(device=a.device, dtype=a.dtype)
    return torch.einsum("ijk, ...j -> ...ki", C, b)


def jacobian_mul_wrt_b(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """8x8 Jacobian of octonion_mul(a, b) w.r.t. b.

    J[k, j] = d(a*b)_k / db_j = sum_i C[i, j, k] * a[i]

    This is the transpose of the left multiplication matrix L_a.
    """
    C = STRUCTURE_CONSTANTS.to(device=a.device, dtype=a.dtype)
    return torch.einsum("ijk, ...i -> ...kj", C, a)
```

### Example 2: Jacobian for octonion_exp
```python
# Source: Derived from _operations.py octonion_exp formula
# exp(o) = exp(a) * (cos(||v||) + sin(||v||) * v_hat)
# where a = o[0], v = o[1:7], v_hat = v / ||v||
#
# The Jacobian is 8x8:
# d(exp(o))_k / do_i
# This has structure because the formula decomposes into real part a and
# imaginary vector v. The full 8x8 Jacobian can be derived analytically.
def jacobian_exp(o: torch.Tensor) -> torch.Tensor:
    """Analytic 8x8 Jacobian of octonion_exp at point o.

    Uses the decomposition exp(a + v) = exp(a)(cos||v|| + sin||v||/||v|| * v)
    """
    a = o[..., 0:1]           # [..., 1]
    v = o[..., 1:]             # [..., 7]
    v_norm = torch.sqrt(torch.sum(v**2, dim=-1, keepdim=True))  # [..., 1]
    exp_a = torch.exp(a)       # [..., 1]

    eps = 1e-15
    safe_norm = torch.where(v_norm > eps, v_norm, torch.ones_like(v_norm))
    v_hat = v / safe_norm      # [..., 7]

    cos_v = torch.cos(v_norm)  # [..., 1]
    sin_v = torch.sin(v_norm)  # [..., 1]

    # Output: result[0] = exp(a)*cos(||v||), result[1:] = exp(a)*sin(||v||)*v_hat
    # Jacobian blocks:
    # d result[0] / d a = exp(a) * cos(||v||)
    # d result[0] / d v_i = -exp(a) * sin(||v||) * v_i / ||v||
    # d result[k] / d a = exp(a) * sin(||v||) * v_hat[k-1]  (k=1..7)
    # d result[k] / d v_i = exp(a) * [ cos(||v||)*v_i*v_hat[k-1]/||v||
    #                                   + sin(||v||)*(delta_{k-1,i}/||v|| - v_i*v_hat[k-1]/||v||^2) ]

    # Build 8x8 Jacobian
    J = torch.zeros(*o.shape[:-1], 8, 8, dtype=o.dtype, device=o.device)
    # ... (full derivation in implementation)
    return J
```

### Example 3: Catalan number binary tree enumeration
```python
# Source: Standard combinatorics algorithm
def catalan(n: int) -> int:
    """Compute the nth Catalan number C_n."""
    if n <= 1:
        return 1
    c = 1
    for i in range(n):
        c = c * 2 * (2 * i + 1) // (i + 2)
    return c

# C_0=1, C_1=1, C_2=2, C_3=5, C_4=14, C_5=42
# For n operands in a product chain: C_{n-1} parenthesizations
# n=4 operands: C_3 = 5
# n=5 operands (depth 5): C_4 = 14 (matches success criterion)
```

### Example 4: autograd.Function with create_graph support
```python
# Source: PyTorch Double Backward tutorial
class OctonionMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Only save inputs (which have grad_fn) for double backward support
        ctx.save_for_backward(a, b)
        return octonion_mul(a, b)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = ctx.saved_tensors
        C = STRUCTURE_CONSTANTS.to(device=a.device, dtype=a.dtype)

        # These einsum operations are differentiable by PyTorch autograd,
        # so create_graph=True will work for second-order gradients.
        grad_a = torch.einsum("...k, ijk, ...j -> ...i", grad_output, C, b)
        grad_b = torch.einsum("...k, ijk, ...i -> ...j", grad_output, C, a)

        return grad_a, grad_b
```

### Example 5: Custom octonion gradcheck
```python
# Source: Adapted from torch.autograd.gradcheck mechanics
def octonion_gradcheck(
    fn,
    inputs: tuple[torch.Tensor, ...],
    eps: float = 1e-7,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> dict:
    """Octonion-aware gradient check.

    Unlike torch.autograd.gradcheck:
    - Reports per-component octonionic errors (e0..e7)
    - Validates both Wirtinger derivatives (df/do and df/do*)
    - Tests parenthesization correctness for compositions
    """
    results = {}
    for idx, inp in enumerate(inputs):
        if not inp.requires_grad:
            continue
        # Numeric Jacobian via central differences
        J_numeric = _numeric_jacobian(fn, inputs, idx, eps)
        # Analytic Jacobian via autograd
        J_analytic = _autograd_jacobian(fn, inputs, idx)
        # Per-component error
        abs_err = torch.abs(J_numeric - J_analytic)
        rel_err = abs_err / (torch.abs(J_numeric) + 1e-15)
        results[f"input_{idx}"] = {
            "max_abs_error": abs_err.max().item(),
            "max_rel_error": rel_err.max().item(),
            "per_component_max": abs_err.max(dim=-2).values.tolist(),
            "passed": bool(torch.allclose(J_numeric, J_analytic, atol=atol, rtol=rtol)),
        }
    return results
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| HR calculus (Xu 2014) | GHR calculus (Xu & Mandic 2015) | 2015 | Solved product and chain rules for quaternions via rotated bases |
| Real-component backprop | GHR-based backprop for QNNs | 2017-2023 | More elegant, compact derivations; same numerical results |
| No octonionic derivative theory | Pseudo-real-matrix representation (OPR 2023) | 2023 | First working octonion gradient approach, but reduces to R^8 |
| torch.autograd.Function (old API) | setup_context + forward separation | PyTorch 2.0+ | Cleaner separation of concerns; better composability |
| Single backward only | Double backward via create_graph=True | Long-standing | Enables Hessian computation needed for Phase 5 |

**Deprecated/outdated:**
- Original HR calculus (before GHR): Lacked product and chain rules; cumbersome for complex expressions
- Component-wise real backpropagation for quaternions: Works but misses algebraic structure

## Open Questions

1. **Does the Moufang identity fully resolve the GHR rotation ambiguity?**
   - What we know: For alternative algebras, the expression xyx (with x appearing twice) IS well-defined regardless of parenthesization (Moufang identity). So mu * q * mu^{-1} should be unambiguous when mu is fixed.
   - What's unclear: The GHR chain rule involves rotations by intermediate function values (the gμ term in the product rule). When g is itself an octonion-valued function, the rotation g(q) * something * g(q)^{-1} involves g appearing twice, so Moufang should apply. But this needs formal verification.
   - Recommendation: Start implementation assuming Moufang resolves it. If numerical tests show discrepancies, this is where the barrier document begins.

2. **How many independent Wirtinger-like derivatives exist for octonions?**
   - What we know: Complex numbers have 2 (df/dz, df/dz*). Quaternions have 4 in GHR (df/dq, df/dq*, df/dq_i, df/dq_i* -- but the latter two are redundant for real-valued loss). The CONTEXT.md says "if GHR derivation reveals the correct number differs from 2, follow the math."
   - What's unclear: For octonions, a naive extension gives 8 partial derivatives (1 real + 7 imaginary directions). The Wirtinger decomposition into df/do and df/do* gives a pair, but the GHR framework for quaternions actually has a pair for EACH rotation direction mu. For octonions, this could mean the full derivative information is captured by a pair (df/do_mu, df/do_mu*) for some chosen mu, analogous to quaternions.
   - Recommendation: Implement the pair (df/do, df/do*) as primary, corresponding to mu=1 (identity rotation). The 8 real partial derivatives are always available as the Jacobian columns. Document whether additional rotation-dependent derivatives provide extra information.

3. **Learning rate scaling heuristic**
   - What we know: The GHR gradient magnitude for octonions involves 8 components vs 4 for quaternions vs 2 for complex. Empirically, gradient norms may scale differently.
   - What's unclear: The exact scaling factor and whether it's input-dependent.
   - Recommendation: Defer to empirical measurement during implementation. Compute gradient norm statistics across random inputs and derive a heuristic scaling factor.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest + hypothesis (already configured) |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `docker compose run --rm dev uv run pytest tests/test_calculus.py -x` |
| Full suite command | `docker compose run --rm dev uv run pytest -x` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SC-1 | Custom autograd backward matches finite-difference to 1e-5 rel error on single OctonionLinear at float64 | unit | `docker compose run --rm dev uv run pytest tests/test_calculus.py::test_gradcheck_octonion_linear -x` | No -- Wave 0 |
| SC-2 | Gradient check passes on 5-layer compositions with all 14 Catalan(4) parenthesization patterns | integration | `docker compose run --rm dev uv run pytest tests/test_calculus.py::test_parenthesization_exhaustive -x` | No -- Wave 0 |
| SC-3 | Naive (associativity-assuming) chain rule produces different/wrong gradients vs parenthesization-aware | unit | `docker compose run --rm dev uv run pytest tests/test_calculus.py::test_naive_vs_correct_differs -x` | No -- Wave 0 |
| SC-4 | Backward pass on ROCm GPU produces identical results to CPU | manual | `docker compose run --rm dev uv run pytest tests/test_calculus.py::test_gpu_cpu_parity -x` (manual run) | No -- Wave 0 |
| TRIP-1 | Analytic Jacobian matches numeric Jacobian for each primitive | unit | `docker compose run --rm dev uv run pytest tests/test_calculus.py::TestJacobians -x` | No -- Wave 0 |
| TRIP-2 | Autograd backward matches analytic Jacobian for each primitive | unit | `docker compose run --rm dev uv run pytest tests/test_calculus.py::TestAutogradVsAnalytic -x` | No -- Wave 0 |
| TRIP-3 | gradgradcheck passes (create_graph=True support) | unit | `docker compose run --rm dev uv run pytest tests/test_calculus.py::test_gradgradcheck -x` | No -- Wave 0 |
| GHR-1 | Wirtinger derivative pair (df/do, df/do*) computed correctly | unit | `docker compose run --rm dev uv run pytest tests/test_calculus.py::TestWirtingerPair -x` | No -- Wave 0 |
| GHR-2 | CR-like analyticity conditions defined and tested | unit | `docker compose run --rm dev uv run pytest tests/test_calculus.py::TestAnalyticity -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `docker compose run --rm dev uv run pytest tests/test_calculus.py -x`
- **Per wave merge:** `docker compose run --rm dev uv run pytest -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_calculus.py` -- covers SC-1, SC-2, SC-3, SC-4, TRIP-1, TRIP-2, TRIP-3, GHR-1, GHR-2
- [ ] `tests/conftest.py` additions -- strategies for composition trees, gradcheck-friendly octonion tensors (small norm, requires_grad=True)
- [ ] `src/octonion/calculus/__init__.py` -- new submodule package init
- [ ] Framework install: none needed (pytest + hypothesis already present)

## Sources

### Primary (HIGH confidence)
- [PyTorch autograd.Function docs](https://docs.pytorch.org/docs/stable/notes/extending.html) - Custom Function pattern, save_for_backward, backward return rules
- [PyTorch Double Backward tutorial](https://docs.pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html) - create_graph=True patterns, gotchas about no-grad forward mode
- [PyTorch gradcheck docs](https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html) - Finite difference parameters, tolerance defaults (eps=1e-6, atol=1e-5, rtol=1e-3)
- [PyTorch gradcheck mechanics](https://docs.pytorch.org/docs/stable/notes/gradcheck.html) - Central difference formula, complex Wirtinger handling
- Project source code: `_multiplication.py`, `_linear_algebra.py`, `_operations.py`, `_linear.py` - Existing octonionic infrastructure

### Secondary (MEDIUM confidence)
- [GHR Calculus paper (Xu & Mandic 2015)](https://royalsocietypublishing.org/doi/10.1098/rsos.150255) - Quaternionic GHR derivative definitions, product rule, chain rule via quaternion rotations
- [New proof of GHR calculus (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5043305/) - Detailed formulas: df/dq_mu = (1/4)(df/dq_a - df/dq_b * i_mu - df/dq_c * j_mu - df/dq_d * k_mu), product rule df(fg)/dq_mu = f * dg/dq_mu + df/dq_{g*mu} * g
- [Quaternion Derivatives: The GHR Calculus (arxiv 1409.8168)](https://arxiv.org/abs/1409.8168) - Original GHR paper: quaternion rotation at core, product/chain/mean-value/Taylor theorems
- [Octonion Phase Retrieval (arxiv 2308.15784)](https://arxiv.org/abs/2308.15784) - Pseudo-real-matrix approach for octonion derivatives (approach we're NOT using, but validates that the derivative problem exists and is solvable)
- [GHR Calculus Learning Algorithms (NNW 2017)](https://www.nnw.cz/doi/2017/NNW.2017.27.014.pdf) - First deep implementation of GHR-based backprop for QNNs
- [Comparison of QNN Backprop Algorithms](https://www.sciencedirect.com/science/article/pii/S0957417423009508) - Evaluates 4 QNN backprop methods including GHR-calculus approach

### Tertiary (LOW confidence)
- [Deep Octonion Networks (arxiv 1903.08478)](https://arxiv.org/abs/1903.08478) - Claims octonion network building blocks but uses pseudo-real-matrix approach internally; non-associativity handling details unclear
- [Hypercomplex Neural Networks survey (2025)](https://www.sciencedirect.com/science/article/pii/S2215016125004881) - Confirms autograd engines don't natively support octonion non-associativity; backprop must be user-handled
- Moufang identity resolution for octonionic rotation ambiguity - Based on algebraic theory (alternativity implies xyx is well-defined) but not verified in GHR context specifically

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - PyTorch autograd.Function is well-documented, project infrastructure is established
- Architecture: MEDIUM - Module organization and Jacobian patterns are clear; GHR extension formulas need verification during implementation
- Pitfalls: HIGH - Well-understood from PyTorch docs and hypercomplex algebra literature
- GHR octonionic extension: MEDIUM-LOW - Novel territory; quaternionic GHR is well-established, but octonionic extension has not been published. The Moufang identity argument is theoretically sound but unverified in this context
- Parenthesization handling: HIGH - Catalan numbers and binary tree enumeration are standard combinatorics

**Research date:** 2026-03-08
**Valid until:** No expiry (mathematical foundations are stable; PyTorch API is stable at 2.9.x)
