# Phase 1: Octonionic Algebra - Research

**Researched:** 2026-03-07
**Domain:** Octonionic algebra implementation, property-based testing, PyTorch tensor operations
**Confidence:** HIGH

## Summary

Phase 1 builds a verified octonionic algebra library from scratch as a PyTorch-native package. No usable existing library exists -- pyoctonion is CPU-only pure Python, the `hypercomplex` package lacks GPU/autograd support, and all quaternionic ML libraries (hTorch, Orkis-Research) are dormant and quaternion-only. The implementation must encode the exact Fano plane multiplication table (not learn it from data), support batched `[..., 8]` tensor operations at float64 precision, and pass exhaustive property-based tests verifying Moufang identities, norm preservation, alternativity, and Cayley-Dickson cross-check.

The most critical risk is **sign convention inconsistency** (CP-1 from pitfalls research): there are 480 valid octonion multiplication tables across 30 Fano plane orientations. The implementation must commit to ONE canonical convention (Baez 2002, using the `e1*e2=e4` cycle-mod-7 triples) and verify every operation against that single source. A secondary risk is the Cayley-Dickson formula convention: Baez uses `(a,b)(c,d) = (ac - db*, a*d + cb)` while other sources use `(a,b)(c,d) = (ac - d*b, da + bc*)`. These produce isomorphic but distinct multiplication tables; the cross-check test (success criterion 3) must account for this by testing structural equivalence, not bitwise equality.

**Primary recommendation:** Implement multiplication via hardcoded Fano plane structure constants tensor for production, with Cayley-Dickson recursive construction as a separate cross-check function. Use the `e1*e2=e4` mod-7 convention (Baez 2002). Validate with 10,000+ random triples via Hypothesis property-based testing from day one. Use `uv` for dependency management with src layout.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- PyTorch-native tensors from day one -- octonions as torch.Tensor with shape [..., 8]
- Thin class wrapper: `class Octonion` with operator overloading (__mul__, __add__, __sub__, __neg__, __eq__, __repr__)
- float64 as default precision (success criteria demand 1e-12 tolerance)
- Immutable -- all operations return new Octonion instances, no in-place mutation
- Caller controls device placement (library is device-agnostic, standard PyTorch convention)
- No `/` operator -- require explicit `a * b.inverse()` to avoid left/right division ambiguity
- No integer power operator (`**`) -- explicit multiplication required because parenthesization matters
- Full scalar interop: scalar * octonion, octonion * scalar, octonion + scalar (adds to real part)
- Package name: `octonion`, src layout: `src/octonion/` with pyproject.toml
- Dependency management: `uv`
- Import style: `from octonion import Octonion`
- Component ordering: e0 (real/scalar) first, [e0, e1, e2, ..., e7]
- Implement multiplication both via Fano plane table AND Cayley-Dickson recursion; test identical results
- Verify Baez 2002 convention against another reference
- Cayley-Dickson doubling: hardcoded tables for production speed + recursive construction as cross-check
- Alternativity verified in test suite only -- no runtime assertions
- Full R/C/H/O Cayley-Dickson tower: Real, Complex, Quaternion, Octonion types
- Abstract base class `NormedDivisionAlgebra` defining conjugate(), norm(), inverse(), mul()
- Separate `UnitOctonion` type (guarantees norm=1) and `PureOctonion` type (real part = 0)
- Fano plane exposed as inspectable public object with full geometric structure
- 7 quaternionic subalgebras accessible via named constructors
- Batch-first design: all operations accept [..., 8] shaped tensors
- Full PyTorch broadcasting rules for batch dimensions
- `o.real`, `o.imag`, `o[i]`, `o.components` for component access
- Core ops: multiplication, conjugation, norm, inverse, associator
- Extended ops: exp, log, commutator, inner product, 7D cross product
- Linear algebra: left_mul_matrix(a), right_mul_matrix(a) returning 8x8 real matrices
- Transform: OctonionLinear (a*x*b) with both a,b learnable, no bias term
- Conversion: from_quaternion_pair(q1, q2) and to_quaternion_pair() bidirectional
- Random generation: random_octonion(), random_unit_octonion(), random_pure_octonion() with seed control
- Hypothesis for property-based testing with custom octonion strategies
- Known-answer tests alongside property tests
- Moufang identity checker as reusable test utility: check_moufang(a, b, c, tol)
- Precision tracking: tests report max/mean/std of relative errors
- Basic performance benchmarks: operation throughput (ops/sec) on CPU and GPU
- Verbose error messages with math context
- Serialization via torch.save/torch.load

### Claude's Discretion
- Batch testing strategy (separate batch tests vs batched Hypothesis strategies)
- Resolving sign convention conflicts between Baez 2002 and other references
- Internal module organization within src/octonion/
- Exact Hypothesis strategy implementations
- Benchmark script design

### Deferred Ideas (OUT OF SCOPE)
- Trilinear form t(a,b,c) = Re((ab)c*) -- Phase 8+
- Freudenthal cross product -- Phase 8+
- Full Cayley-Dickson tower conversions (embed/project between all levels) -- Phase 8+
- Subalgebra projection (project octonion onto specific quaternionic subalgebra) -- Phase 9
- Integer power operator -- not implemented due to non-associativity parenthesization concerns
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FOUND-01 | Core octonionic algebra library implements multiplication, conjugation, norm, inverse, and associator with property-based tests verifying Moufang identities, norm preservation (\|ab\| = \|a\|\|b\|), alternativity, Fano plane multiplication table correctness, and Cayley-Dickson construction cross-check | Fano plane triples (mod-7 convention), Cayley-Dickson formula `(a,b)(c,d) = (ac - db*, a*d + cb)`, Hypothesis + hypothesis-torch for property-based testing, structure constants tensor for vectorized multiplication, SageMath as cross-validation oracle |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | >=2.7,<2.11 | Tensor operations, autograd, GPU compute | Industry standard ML framework; ROCm-compatible via HIP; custom autograd.Function support |
| Python | 3.12 | Language runtime | Latest stable supported by PyTorch; good typing support |
| uv | latest | Dependency management, virtual env | Fast, modern Python package manager; handles src layout well |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | latest | Test framework | All tests |
| hypothesis | >=6.0 | Property-based testing | Algebraic property verification (Moufang, norm preservation, alternativity) |
| hypothesis-torch | latest | PyTorch tensor strategies | Generating random tensors for hypothesis tests |
| numpy | >=1.26 | CPU reference computations | Test oracles, float64 reference implementations |
| ruff | latest | Linting and formatting | Code quality |
| mypy | latest | Type checking | Static analysis |

### Validation / Cross-Check
| Library | Purpose | When to Use |
|---------|---------|-------------|
| SageMath (external) | Cross-validate multiplication table | One-time verification of Fano plane convention correctness |
| clifford | CPU geometric algebra reference | Validate octonionic algebra against known Clifford algebra identities |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| hypothesis-torch | Manual torch.randn strategies | hypothesis-torch provides dtype/shape/device control out of the box |
| uv | pip + venv | uv is faster and handles lockfiles; pip works but slower |
| pyproject.toml (hatchling) | setuptools | hatchling is simpler for src layout; setuptools works but more boilerplate |

**Installation:**
```bash
# Initialize project with uv
uv init --package octonion
# Or if project dir already exists:
uv add torch --extra-index-url https://download.pytorch.org/whl/rocm7.1
uv add --dev pytest hypothesis hypothesis-torch ruff mypy numpy
```

## Architecture Patterns

### Recommended Project Structure
```
octonion-computation-substrate/
  src/
    octonion/
      __init__.py              # Public API: Octonion, UnitOctonion, PureOctonion, etc.
      _types.py                # NormedDivisionAlgebra ABC, type classes
      _octonion.py             # Core Octonion class implementation
      _multiplication.py       # Fano plane multiplication + structure constants
      _cayley_dickson.py       # Cayley-Dickson recursive construction (cross-check)
      _fano.py                 # Fano plane data structure (public inspectable object)
      _tower.py                # R/C/H types (Real, Complex, Quaternion)
      _operations.py           # exp, log, commutator, inner product, cross product
      _linear_algebra.py       # left_mul_matrix, right_mul_matrix
      _random.py               # Random generation utilities
      _linear.py               # OctonionLinear layer
      py.typed                 # PEP 561 marker
  tests/
    conftest.py                # Shared fixtures, hypothesis strategies, tolerance helpers
    test_multiplication.py     # Fano plane multiplication correctness
    test_cayley_dickson.py     # CD construction + cross-check vs Fano plane
    test_algebraic_properties.py  # Moufang, norm preservation, alternativity, inverse
    test_octonion_class.py     # Octonion class API, operator overloading, component access
    test_types.py              # NormedDivisionAlgebra, UnitOctonion, PureOctonion
    test_tower.py              # Real, Complex, Quaternion types
    test_operations.py         # exp, log, commutator, inner product, cross product
    test_linear_algebra.py     # left_mul_matrix, right_mul_matrix
    test_random.py             # Random generation
    test_linear.py             # OctonionLinear layer
    test_batch.py              # Batched operations, broadcasting
    test_edge_cases.py         # Zero octonion, identity, near-zero, large magnitude
    benchmarks/
      bench_multiplication.py  # Throughput benchmarks
  pyproject.toml
  uv.lock
```

### Pattern 1: Structure Constants Tensor for Vectorized Multiplication
**What:** Encode the Fano plane multiplication as a `[8, 8, 8]` structure constants tensor `C` where `(e_i * e_j) = sum_k C[i,j,k] * e_k`. Octonion multiplication becomes a single `torch.einsum('...i, ijk, ...j -> ...k', a, C, b)` call.
**When to use:** All production multiplication calls.
**Example:**
```python
# Source: Derived from Baez 2002 Fano plane, mod-7 convention
# The 7 oriented triples: (1,2,4), (2,3,5), (3,4,6), (4,5,7), (5,6,1), (6,7,2), (7,1,3)
# For each triple (i,j,k): e_i * e_j = e_k, e_j * e_i = -e_k (+ cyclic)
# Plus: e_0 is identity, e_i * e_i = -e_0

def _build_structure_constants() -> torch.Tensor:
    """Build the [8, 8, 8] structure constants tensor for octonion multiplication.

    C[i,j,k] gives the coefficient of e_k in the product e_i * e_j.
    Uses the Baez 2002 / mod-7 Fano plane convention.
    """
    C = torch.zeros(8, 8, 8, dtype=torch.float64)

    # e_0 is the identity
    for i in range(8):
        C[0, i, i] = 1.0  # e_0 * e_i = e_i
        C[i, 0, i] = 1.0  # e_i * e_0 = e_i

    # e_i * e_i = -e_0 for i > 0
    for i in range(1, 8):
        C[i, i, 0] = -1.0

    # The 7 Fano plane triples (1-indexed imaginary units)
    # Convention: e_i * e_j = e_k for each oriented triple (i, j, k)
    triples = [
        (1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 7),
        (5, 6, 1), (6, 7, 2), (7, 1, 3),
    ]

    for i, j, k in triples:
        # Forward cyclic: e_i * e_j = e_k
        C[i, j, k] = 1.0
        C[j, k, i] = 1.0
        C[k, i, j] = 1.0
        # Reverse: e_j * e_i = -e_k
        C[j, i, k] = -1.0
        C[k, j, i] = -1.0
        C[i, k, j] = -1.0

    return C

# Register as buffer (not parameter) so it moves with device
STRUCTURE_CONSTANTS = _build_structure_constants()

def octonion_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply two octonions represented as [..., 8] tensors.

    Uses structure constants tensor for fully vectorized computation.
    Supports arbitrary batch dimensions via broadcasting.
    """
    C = STRUCTURE_CONSTANTS.to(device=a.device, dtype=a.dtype)
    return torch.einsum('...i, ijk, ...j -> ...k', a, C, b)
```

### Pattern 2: Cayley-Dickson Cross-Check
**What:** Implement multiplication via Cayley-Dickson recursion (octonions as quaternion pairs) as a separate function, then verify it produces identical results to the Fano plane implementation.
**When to use:** Test suite only (success criterion 3).
**Example:**
```python
# Source: Baez 2002, Section 2.2
# (a,b)(c,d) = (ac - db*, a*d + cb)
# where a,b,c,d are quaternions, * is quaternion conjugation

def quaternion_mul(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Multiply quaternions as [..., 4] tensors. Hamilton product."""
    # p = p0 + p1*i + p2*j + p3*k
    # q = q0 + q1*i + q2*j + q3*k
    p0, p1, p2, p3 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    return torch.stack([
        p0*q0 - p1*q1 - p2*q2 - p3*q3,
        p0*q1 + p1*q0 + p2*q3 - p3*q2,
        p0*q2 - p1*q3 + p2*q0 + p3*q1,
        p0*q3 + p1*q2 - p2*q1 + p3*q0,
    ], dim=-1)

def quaternion_conj(q: torch.Tensor) -> torch.Tensor:
    """Conjugate quaternion: negate imaginary parts."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

def cayley_dickson_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Multiply octonions via Cayley-Dickson construction.

    Represents octonion as pair of quaternions (a, b) where
    x = a + b*l, with a,b quaternions and l the 5th basis element.

    Formula (Baez 2002): (a,b)(c,d) = (ac - db*, a*d + cb)
    """
    a, b = x[..., :4], x[..., 4:]  # Split into quaternion pairs
    c, d = y[..., :4], y[..., 4:]

    # (a,b)(c,d) = (ac - db*, a*d + cb)
    real_part = quaternion_mul(a, c) - quaternion_mul(d, quaternion_conj(b))
    imag_part = quaternion_mul(quaternion_conj(a), d) + quaternion_mul(c, b)

    return torch.cat([real_part, imag_part], dim=-1)
```

### Pattern 3: Immutable Octonion Wrapper
**What:** Thin class wrapping a `torch.Tensor` of shape `[..., 8]`, providing operator overloading and named access without copying data unnecessarily.
**When to use:** User-facing API.
**Example:**
```python
class Octonion:
    """Immutable octonion backed by a PyTorch tensor.

    All operations return new Octonion instances.
    The underlying tensor is accessible via .components.
    """
    __slots__ = ('_data',)

    def __init__(self, data: torch.Tensor):
        if data.shape[-1] != 8:
            raise ValueError(
                f"Octonion requires last dimension to be 8, got shape {data.shape}. "
                f"An octonion has 8 components: 1 real (e0) + 7 imaginary (e1..e7)."
            )
        self._data = data

    @property
    def components(self) -> torch.Tensor:
        return self._data

    @property
    def real(self) -> torch.Tensor:
        return self._data[..., 0]

    @property
    def imag(self) -> torch.Tensor:
        return self._data[..., 1:]

    def __getitem__(self, i: int) -> torch.Tensor:
        return self._data[..., i]

    def __mul__(self, other):
        if isinstance(other, Octonion):
            return Octonion(octonion_mul(self._data, other._data))
        if isinstance(other, (int, float, torch.Tensor)):
            return Octonion(self._data * other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return Octonion(other * self._data)
        return NotImplemented

    def conjugate(self) -> 'Octonion':
        return Octonion(torch.cat([self._data[..., :1], -self._data[..., 1:]], dim=-1))

    def norm(self) -> torch.Tensor:
        return torch.sqrt(torch.sum(self._data ** 2, dim=-1))

    def inverse(self) -> 'Octonion':
        n_sq = torch.sum(self._data ** 2, dim=-1, keepdim=True)
        if torch.any(n_sq == 0):
            raise ValueError(
                "Cannot invert zero octonion: norm is 0.0. "
                "Octonion inverse requires non-zero norm."
            )
        conj = self.conjugate()
        return Octonion(conj._data / n_sq)
```

### Pattern 4: Hypothesis Octonion Strategies
**What:** Custom Hypothesis strategies for generating random octonions with controlled properties.
**When to use:** All property-based tests.
**Example:**
```python
# tests/conftest.py
import hypothesis.strategies as st
from hypothesis import given, settings
import torch

@st.composite
def octonions(draw, *, batch_shape=(), dtype=torch.float64,
              min_value=-1e6, max_value=1e6):
    """Strategy generating random Octonion instances."""
    shape = batch_shape + (8,)
    elements = st.floats(min_value=min_value, max_value=max_value,
                         allow_nan=False, allow_infinity=False)
    components = [draw(elements) for _ in range(8)]
    data = torch.tensor(components, dtype=dtype)
    # For batched: would need to generate full batch
    return Octonion(data)

@st.composite
def unit_octonions(draw, *, dtype=torch.float64):
    """Strategy generating unit-norm octonions (on S^7)."""
    o = draw(octonions(dtype=dtype, min_value=-10, max_value=10))
    n = o.norm()
    while n < 1e-10:
        o = draw(octonions(dtype=dtype, min_value=-10, max_value=10))
        n = o.norm()
    return Octonion(o.components / n.unsqueeze(-1))

@st.composite
def nonzero_octonions(draw, *, dtype=torch.float64):
    """Strategy generating non-zero octonions (for inverse testing)."""
    o = draw(octonions(dtype=dtype, min_value=-1e3, max_value=1e3))
    # Ensure non-zero by adding a small real part if needed
    if o.norm() < 1e-10:
        data = o.components.clone()
        data[..., 0] = 1.0
        o = Octonion(data)
    return o
```

### Pattern 5: Precision-Tracking Test Utilities
**What:** Test helpers that report detailed error statistics, not just pass/fail.
**When to use:** All numerical property tests (Moufang, norm preservation, etc.).
**Example:**
```python
def check_moufang(a, b, c, tol=1e-12):
    """Verify all four Moufang identities and return detailed error statistics.

    Returns dict with max_error, mean_error, std_error for each identity.
    """
    results = {}

    # Identity 1: z(x(zy)) = ((zx)z)y
    lhs = c * (a * (c * b))
    rhs = ((c * a) * c) * b
    err = (lhs.components - rhs.components).abs()
    results['moufang_1'] = {
        'max_error': err.max().item(),
        'mean_error': err.mean().item(),
        'identity': 'z(x(zy)) = ((zx)z)y'
    }

    # Identity 2: x(z(yz)) = ((xz)y)z
    lhs = a * (c * (b * c))
    rhs = ((a * c) * b) * c
    err = (lhs.components - rhs.components).abs()
    results['moufang_2'] = {
        'max_error': err.max().item(),
        'mean_error': err.mean().item(),
        'identity': 'x(z(yz)) = ((xz)y)z'
    }

    # Identity 3: (zx)(yz) = (z(xy))z
    lhs = (c * a) * (b * c)
    rhs = (c * (a * b)) * c
    err = (lhs.components - rhs.components).abs()
    results['moufang_3'] = {
        'max_error': err.max().item(),
        'mean_error': err.mean().item(),
        'identity': '(zx)(yz) = (z(xy))z'
    }

    # Identity 4 (flexibility): (xy)x = x(yx)
    lhs = (a * b) * a
    rhs = a * (b * a)
    err = (lhs.components - rhs.components).abs()
    results['flexibility'] = {
        'max_error': err.max().item(),
        'mean_error': err.mean().item(),
        'identity': '(xy)x = x(yx)'
    }

    # Overall pass/fail
    max_err = max(r['max_error'] for r in results.values())
    results['passed'] = max_err < tol
    results['max_error_overall'] = max_err

    return results
```

### Anti-Patterns to Avoid
- **Mixing multiplication conventions:** Never combine a Fano plane table from one source with identities from another. Use Baez 2002 exclusively for the primary implementation.
- **Component-wise operations where algebraic operations are needed:** Do not implement `__add__` as component-wise addition for non-scalar operands without understanding the algebra. Octonion addition IS component-wise, but multiplication is NOT.
- **In-place tensor mutation:** All operations must return new tensors. The Octonion class is immutable per user decision. Never use in-place ops like `data.mul_()`.
- **Using `torch.autograd` naively for multiplication:** In Phase 1, multiplication is a pure tensor operation (einsum with structure constants) so autograd works correctly. But do NOT assume this generalizes -- Phase 2 will need custom backward for composed non-associative operations.
- **Testing only with small/unit octonions:** Test with a range of magnitudes including near-zero (1e-15), unit (1.0), moderate (1e3), and large (1e10) to catch numerical issues.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Property-based test generation | Manual random loop with seed | Hypothesis + hypothesis-torch | Hypothesis provides shrinking (finds minimal failing example), reproducibility, and configurable test count |
| Package management | requirements.txt + pip | uv + pyproject.toml | uv handles lockfiles, virtual envs, and is 10-100x faster than pip |
| Quaternion multiplication reference | Custom quaternion implementation for validation | numpy-quaternion or SageMath | Well-tested reference implementations for cross-validation |
| Tensor broadcasting | Manual shape expansion | PyTorch broadcasting rules | PyTorch handles all broadcasting edge cases correctly |
| Code formatting/linting | Manual style enforcement | ruff | Fast, comprehensive, replaces black + isort + flake8 |

**Key insight:** The algebra itself must be built from scratch (no existing library is suitable), but everything around it (testing, packaging, validation) should use mature tools.

## Common Pitfalls

### Pitfall 1: Fano Plane Sign Convention Errors (CP-1)
**What goes wrong:** The Fano plane has 480 valid sign conventions across 30 orientations. Implementing the wrong one, or mixing conventions between components, produces an algebra that looks correct but violates alternativity or Moufang identities.
**Why it happens:** Copy-pasting a multiplication table from one source and combining with identities from another source that uses a different convention.
**How to avoid:** Use ONE source (Baez 2002, mod-7 convention). The 7 triples are: (1,2,4), (2,3,5), (3,4,6), (4,5,7), (5,6,1), (6,7,2), (7,1,3). Verify with exhaustive Moufang identity tests on 10,000+ random triples.
**Warning signs:** Moufang identity fails on >0.1% of random triples. Any single failure means the table is wrong.

### Pitfall 2: Cayley-Dickson Formula Convention Mismatch
**What goes wrong:** The Cayley-Dickson formula has multiple valid forms. Baez 2002 uses `(a,b)(c,d) = (ac - db*, a*d + cb)`. Wikipedia and John D. Cook's code use `(a,b)(c,d) = (ac - d*b, da + bc*)`. These produce different (but isomorphic) multiplication tables. Testing Fano plane multiplication against the wrong Cayley-Dickson formula will fail.
**Why it happens:** The placement of conjugation in the formula matters; different sources use different conventions.
**How to avoid:** When implementing the Cayley-Dickson cross-check, derive the formula that corresponds to the chosen Fano plane convention. Specifically, if using the Baez 2002 Fano plane triples, use the Baez 2002 Cayley-Dickson formula. Verify by testing basis element products: `e1 * e2` should yield `e4` in both implementations.
**Warning signs:** Cross-check test fails on basis element products but passes on some random inputs (sign flips cancel out statistically).

### Pitfall 3: Floating-Point vs Algebraic Non-Associativity (MP-5)
**What goes wrong:** IEEE 754 float arithmetic is itself non-associative. Tests checking `(xy)z != x(yz)` may pass due to floating-point rounding, not because the algebra was correctly implemented.
**Why it happens:** Octonion multiplication involves 64 real multiplications and many additions; GPU parallel reduction introduces non-deterministic rounding.
**How to avoid:** Test associator magnitude: for random unit octonions, `||[x,y,z]|| / (||x|| * ||y|| * ||z||)` should be O(1), not O(machine_epsilon). If the measured associator is the same order as machine epsilon, you are measuring noise. Use `torch.use_deterministic_algorithms(True)` during validation.
**Warning signs:** Associator magnitude changes significantly between float32 and float64 runs.

### Pitfall 4: Inverse Numerical Instability Near Zero
**What goes wrong:** `a.inverse()` computes `conjugate(a) / norm(a)^2`. For near-zero octonions, `norm(a)^2` is tiny, causing the inverse to have enormous magnitude and poor relative precision.
**Why it happens:** Division by small numbers amplifies floating-point errors.
**How to avoid:** Test inverse at multiple magnitude scales. The success criterion (a * a_inv = 1 to numerical precision) should be tested with octonions of norm ~1, not near-zero. For edge cases, raise a clear error when norm is below a threshold.
**Warning signs:** `a * a.inverse()` deviates from identity by more than 1e-12 for norm(a) < 1e-6.

### Pitfall 5: einsum Performance on Large Batches
**What goes wrong:** The naive `torch.einsum('...i, ijk, ...j -> ...k', a, C, b)` may not be the fastest approach for large batches because it creates intermediate tensors of size `[batch, 8, 8, 8]`.
**Why it happens:** The structure constants tensor is sparse (only 7*6 + 8 = 50 non-zero entries out of 512), but einsum does not exploit sparsity.
**How to avoid:** Profile first. If einsum is a bottleneck, implement multiplication as explicit component-wise operations (7 quaternionic triple products). The hardcoded approach has 48 multiply-add operations vs einsum's full contraction. In Phase 1, correctness matters more than speed -- optimize only if benchmarks show it is needed.
**Warning signs:** Multiplication throughput on GPU is less than 10x CPU throughput for batch sizes > 10,000.

## Code Examples

Verified patterns from official sources and mathematical references:

### Complete Fano Plane Data Structure
```python
# Source: Baez 2002, mod-7 convention verified against nLab and John D. Cook
from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class FanoPlane:
    """The Fano plane PG(2,2) encoding octonionic multiplication structure.

    The 7 lines correspond to the 7 oriented triples (i, j, k) where
    e_i * e_j = e_k (with cyclic permutations positive, anti-cyclic negative).

    Convention: Baez 2002, e_i * e_{i+1 mod 7} = e_{i+3 mod 7} (with 1-indexed units)
    Equivalently: triples (1,2,4), (2,3,5), (3,4,6), (4,5,7), (5,6,1), (6,7,2), (7,1,3)
    """
    # The 7 oriented triples (using 1-indexed imaginary unit labels)
    triples: Tuple[Tuple[int, int, int], ...] = (
        (1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 7),
        (5, 6, 1), (6, 7, 2), (7, 1, 3),
    )

    @property
    def lines(self) -> List[frozenset]:
        """The 7 unoriented lines (sets of 3 points)."""
        return [frozenset(t) for t in self.triples]

    @property
    def incidence_matrix(self) -> 'torch.Tensor':
        """7x7 incidence matrix: M[point][line] = 1 if point is on line."""
        import torch
        M = torch.zeros(7, 7, dtype=torch.int)
        for line_idx, (i, j, k) in enumerate(self.triples):
            M[i-1, line_idx] = 1
            M[j-1, line_idx] = 1
            M[k-1, line_idx] = 1
        return M

    def quaternionic_subalgebra(self, line_index: int) -> Tuple[int, int, int]:
        """Return the triple of imaginary unit indices forming the line_index-th subalgebra.

        Each line of the Fano plane defines a quaternionic subalgebra
        {e_0, e_i, e_j, e_k} isomorphic to H.
        """
        return self.triples[line_index]

    @property
    def automorphism_generators(self) -> List:
        """Generators of GL(3, F_2), the symmetry group of the Fano plane (order 168).

        These are permutations of {1,...,7} that preserve the incidence structure.
        """
        # The cyclic permutation i -> i+1 mod 7 (order 7)
        cycle_7 = {i: (i % 7) + 1 for i in range(1, 8)}
        # The quadratic residue map i -> 2i mod 7 (order 3)
        quad_res = {i: (2 * i - 1) % 7 + 1 for i in range(1, 8)}
        return [cycle_7, quad_res]

FANO_PLANE = FanoPlane()
```

### NormedDivisionAlgebra Abstract Base Class
```python
from abc import ABC, abstractmethod
import torch

class NormedDivisionAlgebra(ABC):
    """Abstract base class for the Cayley-Dickson tower: R, C, H, O.

    All normed division algebras share: conjugation, norm, inverse, multiplication.
    """

    @abstractmethod
    def conjugate(self) -> 'NormedDivisionAlgebra':
        """Return the conjugate (negate all imaginary components)."""
        ...

    @abstractmethod
    def norm(self) -> torch.Tensor:
        """Return the norm (sqrt of sum of squared components)."""
        ...

    def norm_squared(self) -> torch.Tensor:
        """Return the squared norm (avoids sqrt for precision)."""
        return torch.sum(self.components ** 2, dim=-1)

    @abstractmethod
    def inverse(self) -> 'NormedDivisionAlgebra':
        """Return the multiplicative inverse: x^{-1} = conj(x) / |x|^2."""
        ...

    @abstractmethod
    def __mul__(self, other) -> 'NormedDivisionAlgebra':
        """Algebra-specific multiplication."""
        ...

    @property
    @abstractmethod
    def components(self) -> torch.Tensor:
        """Raw tensor of shape [..., dim] where dim is 1/2/4/8."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Algebraic dimension: 1 for R, 2 for C, 4 for H, 8 for O."""
        ...
```

### Associator Computation
```python
# Source: Mathematical definition from Baez 2002
def associator(a: 'Octonion', b: 'Octonion', c: 'Octonion') -> 'Octonion':
    """Compute the associator [a, b, c] = (a*b)*c - a*(b*c).

    Properties (verified in tests):
    - Totally antisymmetric: [a,b,c] = -[b,a,c] = -[a,c,b] = -[c,b,a]
    - Zero when any two arguments are equal (alternativity)
    - Non-zero for generic triples (octonions are NOT associative)
    """
    left = (a * b) * c   # Left-associated product
    right = a * (b * c)  # Right-associated product
    return Octonion(left.components - right.components)

def associator_magnitude(a: 'Octonion', b: 'Octonion', c: 'Octonion') -> torch.Tensor:
    """Normalized associator magnitude: ||[a,b,c]|| / (||a|| * ||b|| * ||c||).

    For non-zero generic triples, this should be O(1).
    If it is O(machine_epsilon), you are measuring floating-point noise.
    """
    assoc = associator(a, b, c)
    return assoc.norm() / (a.norm() * b.norm() * c.norm())
```

### Left/Right Multiplication Matrices
```python
def left_mul_matrix(a: 'Octonion') -> torch.Tensor:
    """Return the 8x8 real matrix L_a such that a*x = L_a @ x for all x.

    L_a[i,j] = sum_k C[i_comp, j_comp, k_comp] * a[i_comp]
    where C is the structure constants tensor.

    Returns: Tensor of shape [..., 8, 8]
    """
    C = STRUCTURE_CONSTANTS.to(device=a.components.device, dtype=a.components.dtype)
    # L_a[out_k, in_j] = sum_i a[i] * C[i, j, k]
    return torch.einsum('...i, ijk -> ...kj', a.components, C)

def right_mul_matrix(b: 'Octonion') -> torch.Tensor:
    """Return the 8x8 real matrix R_b such that x*b = R_b @ x for all x.

    Returns: Tensor of shape [..., 8, 8]
    """
    C = STRUCTURE_CONSTANTS.to(device=b.components.device, dtype=b.components.dtype)
    # R_b[out_k, in_i] = sum_j b[j] * C[i, j, k]
    return torch.einsum('...j, ijk -> ...ki', b.components, C)
```

### OctonionLinear Layer
```python
import torch.nn as nn

class OctonionLinear(nn.Module):
    """Linear layer: output = a * x * b where a, b are learnable octonions.

    No bias term (per user decision).
    Both a and b are learnable parameters.
    This is a two-sided multiplication, not a matrix multiplication.
    """

    def __init__(self, dtype=torch.float64):
        super().__init__()
        # Initialize as unit-norm octonions (preserves scale at init)
        a_init = torch.randn(8, dtype=dtype)
        a_init = a_init / a_init.norm()
        b_init = torch.randn(8, dtype=dtype)
        b_init = b_init / b_init.norm()

        self.a = nn.Parameter(a_init)
        self.b = nn.Parameter(b_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a * x * b. x has shape [..., 8]."""
        # Two sequential multiplications: first a*x, then result*b
        # Parenthesization is (a*x)*b (left-to-right)
        ax = octonion_mul(self.a.unsqueeze(0).expand_as(x), x)
        return octonion_mul(ax, self.b.unsqueeze(0).expand_as(ax))
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| CPU-only octonion libraries (pyoctonion) | PyTorch GPU-native tensor operations | N/A (no GPU library exists) | Must build from scratch; opportunity to set the standard |
| Learned hypercomplex multiplication (PHM/HyperNets) | Hardcoded algebraic multiplication tables | PHM: 2021 | PHM learns rules from data; this project encodes exact algebra. Different goals. |
| Quaternionic backprop via GHR calculus | Open research for octonions | GHR for Q: 2015 | Octonion extension is Phase 2's research contribution |
| Independent init per component (Glorot/He) | Chi-distribution octonionic init | Gaudet & Maida 2018 (for Q) | Must extend to 8-DOF Chi distribution for octonions |

**Deprecated/outdated:**
- hTorch (last commit 2021): Quaternion-only, experimental. Do not use as dependency.
- Pytorch-Quaternion-Neural-Networks (last commit 2019): Functional math but abandoned. Vendor relevant code for quaternion baselines only.
- pyoctonion: CPU-only, pure Python. Not suitable for GPU workloads.

## Open Questions

1. **Exact mapping between Baez 2002 Fano plane and Cayley-Dickson formula**
   - What we know: Baez 2002 uses triples (1,2,4), (2,3,5), ..., (7,1,3) for the Fano plane AND `(a,b)(c,d) = (ac - db*, a*d + cb)` for Cayley-Dickson.
   - What's unclear: Whether the Cayley-Dickson formula with the standard quaternion-pair split `x[:4], x[4:]` exactly reproduces the mod-7 Fano plane triples, or requires a permutation of basis elements.
   - Recommendation: Implement both, test basis element products `e_i * e_j` for all 49 pairs in both systems. If they disagree, determine the permutation mapping. This IS the cross-check test (success criterion 3).

2. **einsum vs explicit component multiplication performance**
   - What we know: einsum is elegant but may not exploit the sparsity of structure constants (50/512 non-zero).
   - What's unclear: Actual performance difference on GPU for batch sizes 100-100,000.
   - Recommendation: Start with einsum for clarity. Add an explicit component-wise implementation as an alternative. Benchmark both in the performance benchmarks task. Choose the faster one as default.

3. **Hypothesis strategy for batched octonions**
   - What we know: hypothesis-torch provides tensor strategies. Batched octonion testing needs `[batch, 8]` tensors.
   - What's unclear: Whether to test batch properties via separate batch test functions or by having the Hypothesis strategy produce batched tensors.
   - Recommendation (Claude's discretion): Use both approaches. Have strategies that produce single octonions for algebraic property tests, and separate strategies that produce batched octonions `[N, 8]` for verifying batch broadcasting correctness. The algebraic properties only need to hold element-wise, so single-octonion strategies with @given(n=10000) are cleaner for Moufang tests.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest + hypothesis |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/ -x --tb=short -q` |
| Full suite command | `uv run pytest tests/ -v --hypothesis-seed=0` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FOUND-01a | Moufang identities pass on 10,000+ random triples | property | `uv run pytest tests/test_algebraic_properties.py::test_moufang_identities -x` | Wave 0 |
| FOUND-01b | Norm preservation \|ab\| = \|a\|\|b\| within 1e-12 | property | `uv run pytest tests/test_algebraic_properties.py::test_norm_preservation -x` | Wave 0 |
| FOUND-01c | Cayley-Dickson matches Fano plane table | unit | `uv run pytest tests/test_cayley_dickson.py::test_fano_cd_crosscheck -x` | Wave 0 |
| FOUND-01d | Inverse: a * a_inv = 1 and a_inv * a = 1 | property | `uv run pytest tests/test_algebraic_properties.py::test_inverse -x` | Wave 0 |
| FOUND-01e | Associator non-zero for generic triples, zero for equal args | property | `uv run pytest tests/test_algebraic_properties.py::test_alternativity -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/ -x --tb=short -q`
- **Per wave merge:** `uv run pytest tests/ -v --hypothesis-seed=0 --hypothesis-settings=ci`
- **Phase gate:** Full suite green before verification

### Wave 0 Gaps
- [ ] `pyproject.toml` -- project configuration with uv, pytest, hypothesis settings
- [ ] `src/octonion/__init__.py` -- package entry point
- [ ] `tests/conftest.py` -- shared fixtures, hypothesis strategies, tolerance constants
- [ ] Framework install: `uv add --dev pytest hypothesis hypothesis-torch`

## Sources

### Primary (HIGH confidence)
- [Baez 2002 - The Octonions (Fano plane section)](https://math.ucr.edu/home/baez/octonions/node4.html) - Fano plane structure, multiplication convention
- [Baez 2002 - Cayley-Dickson Construction](https://math.ucr.edu/home/baez/octonions/node5.html) - CD formula: (a,b)(c,d) = (ac - db*, a*d + cb)
- [nLab - Octonion](https://ncatlab.org/nlab/show/octonion) - Fano plane triples verification, multiplication rules
- [John D. Cook - How to multiply octonions (2021)](https://www.johndcook.com/blog/2021/12/19/multiply-octonions/) - mod-7 convention verification, e1*e2=e4 confirmed
- [John D. Cook - Cayley-Dickson (2018)](https://www.johndcook.com/blog/2018/07/10/cayley-dickson/) - Python Cayley-Dickson implementation reference
- [PyTorch autograd documentation](https://docs.pytorch.org/docs/stable/autograd.html) - Custom autograd.Function patterns
- [Hypothesis documentation](https://hypothesis.readthedocs.io/) - Property-based testing framework
- [hypothesis-torch](https://github.com/qthequartermasterman/hypothesis-torch) - PyTorch tensor strategies

### Secondary (MEDIUM confidence)
- [SageMath Octonion Algebra](https://doc.sagemath.org/html/en/reference/algebras/sage/algebras/octonion_algebra.html) - Uses Schafer 1996 convention; basis (1, i, j, k, l, li, lj, lk)
- [Wikipedia - Cayley-Dickson construction](https://en.wikipedia.org/wiki/Cayley%E2%80%93Dickson_construction) - Alternative CD formula: (a,b)(c,d) = (ac - d*b, da + bc*)
- [Hypercomplex neural networks survey (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12513225/) - Field overview
- [uv documentation - Projects](https://docs.astral.sh/uv/guides/projects/) - Package management with src layout
- [pytest good practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html) - src layout test configuration
- [Python Packaging Guide - pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) - Package configuration

### Tertiary (LOW confidence)
- einsum performance vs explicit component multiplication: untested assumption, needs benchmarking
- Chi-distribution initialization for 8-DOF: extrapolated from Gaudet & Maida 2018 quaternion case, no published octonion validation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - PyTorch, pytest, hypothesis are mature and well-documented
- Architecture: HIGH - Patterns derived from Baez 2002 mathematics and proven hypercomplex ML library structures
- Pitfalls: HIGH - Sign convention errors and floating-point non-associativity are well-documented in mathematical computing literature
- Multiplication convention: HIGH - Baez 2002 mod-7 convention verified across multiple independent sources (nLab, John D. Cook, Wikipedia)
- Cayley-Dickson cross-check: MEDIUM - Multiple valid CD formulas exist; exact mapping to chosen Fano plane triples needs implementation-time verification

**Research date:** 2026-03-07
**Valid until:** 2026-04-07 (stable mathematical domain; PyTorch API stable across minor versions)
