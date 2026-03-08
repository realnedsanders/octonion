---
status: resolved
phase: 01-octonionic-algebra
source: 01-00-SUMMARY.md, 01-01-SUMMARY.md, 01-02-SUMMARY.md, 01-03-SUMMARY.md
started: 2026-03-08T06:30:00Z
updated: 2026-03-08T07:30:00Z
---

## Current Test
<!-- OVERWRITE each test - shows where we are -->

[testing complete]

## Tests

### 1. Cold Start Smoke Test
expected: Kill any running containers. Run `docker compose build` then `docker compose run --rm dev uv sync` then `docker compose run --rm dev uv run pytest`. Container builds without errors, dependencies install, and all 209 tests pass.
result: issue
reported: "fail, python error on docker compose run --rm dev uv run pytest, no module named octonion found. Previous two commands appear successful."
severity: blocker

### 2. Package Import and API Surface
expected: Run `docker compose run --rm dev uv run python -c "from octonion import Octonion, FANO_PLANE, STRUCTURE_CONSTANTS, octonion_mul, octonion_exp, octonion_log, OctonionLinear, Real, Complex, Quaternion, random_octonion; print('All imports OK')"`. All symbols import without error and prints "All imports OK".
result: pass

### 3. Octonion Multiplication Non-Commutativity
expected: Run `docker compose run --rm dev uv run python -c "from octonion import random_octonion, Octonion; a,b = Octonion(random_octonion()),Octonion(random_octonion()); print('a*b:', a*b); print('b*a:', b*a); print('Equal?', a*b == b*a)"`. Should print two different octonion products and "Equal? False", demonstrating non-commutativity.
result: issue
reported: "fail, Octonion(random_octonion()) raises AttributeError: 'Octonion' object has no attribute 'shape'. Octonion.__init__ checks data.shape[-1] but random_octonion() returns an Octonion not a tensor. Also numpy missing warning."
severity: blocker

### 4. Octonion Class Operators
expected: Run `docker compose run --rm dev uv run python -c "from octonion import Octonion; import torch; a = Octonion(torch.tensor([1.,2.,3.,4.,5.,6.,7.,8.])); print('a:', a); print('conj:', a.conjugate()); print('norm:', a.norm()); print('inv:', a.inverse()); print('a*inv:', a * a.inverse())"`. Shows octonion repr, conjugate, norm (scalar), inverse, and a*a^-1 close to identity [1,0,0,0,0,0,0,0].
result: issue
reported: "identity looks wrong at minimum. a*inv shows 1.0 with ~1.5e-08 residuals on imaginary components instead of clean identity. float32 precision noise visible in output."
severity: minor

### 5. OctonionLinear Forward and Backward Pass
expected: Run `docker compose run --rm dev uv run python -c "from octonion import OctonionLinear; import torch; layer = OctonionLinear(); x = torch.randn(4, 8); y = layer(x); loss = y.sum(); loss.backward(); print('Input shape:', x.shape); print('Output shape:', y.shape); print('Grad exists:', layer.left.grad is not None)"`. Output shape is [4,8], gradients exist (True).
result: issue
reported: "RuntimeError: expected scalar type Double but found Float. OctonionLinear forward crashes because STRUCTURE_CONSTANTS is float64 but torch.randn defaults to float32. einsum refuses mixed dtypes. Also numpy missing warning."
severity: blocker

### 6. Exp/Log Roundtrip
expected: Run `docker compose run --rm dev uv run python -c "import torch; from octonion import octonion_exp, octonion_log; x = torch.zeros(8); x[0]=0.5; x[1]=0.3; x[2]=0.1; y = octonion_exp(x); z = octonion_log(y); print('Original:', x); print('Exp:', y); print('Log(Exp):', z); print('Close?', torch.allclose(x, z, atol=1e-12))"`. Should print Close? True, showing exp/log are inverses.
result: issue
reported: "fail. RuntimeError: imag is not implemented for tensors with non-complex dtypes. octonion_exp uses o.imag on a real-valued [8] tensor — .imag only works on complex dtypes."
severity: blocker

## Summary

total: 6
passed: 1
issues: 5
pending: 0
skipped: 0

## Gaps

- truth: "Container builds, dependencies install, and all 209 tests pass on cold start"
  status: resolved
  reason: "User reported: fail, python error on docker compose run --rm dev uv run pytest, no module named octonion found. Previous two commands appear successful."
  severity: blocker
  test: 1
  root_cause: "Dev dependencies declared under [project.optional-dependencies] dev (PEP 621 extras) but uv sync only auto-includes [dependency-groups] dev (PEP 735). uv sync installs only torch, leaving pytest uninstalled. uv run pytest uses ephemeral resolution without .pth file access."
  artifacts:
    - path: "pyproject.toml"
      issue: "Dev deps in [project.optional-dependencies] instead of [dependency-groups]"
  missing:
    - "Migrate dev dependencies from [project.optional-dependencies] to [dependency-groups] dev"
  debug_session: ".planning/debug/no-module-octonion-pytest.md"
- truth: "Two random octonions can be multiplied and show non-commutativity"
  status: resolved
  reason: "User reported: Octonion(random_octonion()) raises AttributeError: 'Octonion' object has no attribute 'shape'. random_octonion() returns an Octonion not a tensor, so wrapping it in Octonion() fails. Also numpy missing warning."
  severity: blocker
  test: 3
  root_cause: "Octonion.__init__ expects torch.Tensor but random_octonion() returns Octonion. Constructor has no type guard to unwrap Octonion input via .components."
  artifacts:
    - path: "src/octonion/_octonion.py"
      issue: "Octonion.__init__ lacks copy-constructor support (no Octonion-unwrap guard)"
  missing:
    - "Add isinstance(data, Octonion) guard in __init__ to extract .components"
  debug_session: ".planning/debug/uat3-octonion-wrapping-octonion.md"
- truth: "a*a^-1 shows clean identity [1,0,0,0,0,0,0,0]"
  status: resolved
  reason: "User reported: identity looks wrong at minimum. a*inv shows 1.0 with ~1.5e-08 residuals on imaginary components instead of clean identity. float32 precision noise visible in output."
  severity: minor
  test: 4
  root_cause: "Octonion.__str__ uses val != 0.0 with no near-zero suppression threshold. float32 ~1.5e-08 residuals are mathematically correct but display as ugly noise."
  artifacts:
    - path: "src/octonion/_octonion.py"
      issue: "__str__ line 234 uses val != 0.0 with no tolerance-based display threshold"
  missing:
    - "Add dtype-aware near-zero suppression in __str__ (e.g., abs(val) > atol)"
  debug_session: ".planning/debug/float32-precision-noise-identity.md"
- truth: "OctonionLinear forward pass works with default float32 input"
  status: resolved
  reason: "User reported: RuntimeError: expected scalar type Double but found Float. STRUCTURE_CONSTANTS is float64, torch.randn defaults to float32, einsum refuses mixed dtypes in octonion_mul."
  severity: blocker
  test: 5
  root_cause: "OctonionLinear defaults to dtype=torch.float64 (not PyTorch convention float32). octonion_mul casts C to match a.dtype but does not harmonize a and b dtypes, so einsum gets mixed float64/float32."
  artifacts:
    - path: "src/octonion/_linear.py"
      issue: "Default dtype is torch.float64 instead of torch.float32 (line 34)"
    - path: "src/octonion/_multiplication.py"
      issue: "octonion_mul does not promote a/b to common dtype before einsum (lines 77-78)"
  missing:
    - "Change OctonionLinear default dtype to torch.float32"
    - "Add dtype promotion in octonion_mul via torch.promote_types"
  debug_session: ".planning/debug/dtype-mismatch-einsum.md"
- truth: "octonion_exp and octonion_log are inverse operations on raw tensors"
  status: resolved
  reason: "User reported: RuntimeError: imag is not implemented for tensors with non-complex dtypes. octonion_exp uses o.imag on a real-valued [8] tensor — .imag only works on complex dtypes."
  severity: blocker
  test: 6
  root_cause: "octonion_exp/log call o.real and o.imag assuming Octonion instance. On raw tensors, .imag resolves to PyTorch's Tensor.imag which requires complex dtype. No type guard or auto-coercion to Octonion."
  artifacts:
    - path: "src/octonion/_operations.py"
      issue: "octonion_exp (line 32) and octonion_log (line 78) use o.imag/o.real with no type guard for raw tensors"
  missing:
    - "Add auto-coercion guard: if isinstance(o, torch.Tensor): o = Octonion(o)"
  debug_session: ".planning/debug/exp-log-imag-on-real-tensor.md"
