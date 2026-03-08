---
status: resolved
trigger: "RuntimeError: imag is not implemented for tensors with non-complex dtypes in octonion_exp/octonion_log"
created: 2026-03-08T07:00:00Z
updated: 2026-03-08T07:30:00Z
---

## Current Focus

hypothesis: octonion_exp and octonion_log call `.real` and `.imag` on a raw torch.Tensor instead of an Octonion, hitting PyTorch's complex-only `.imag`
test: trace the attribute resolution path when a raw tensor is passed
expecting: PyTorch's Tensor.imag raises RuntimeError on real-dtype tensors
next_action: return diagnosis

## Symptoms

expected: `octonion_exp(torch.zeros(8))` computes the octonion exponential
actual: RuntimeError at _operations.py line 32 (`v = o.imag`)
errors: "RuntimeError: imag is not implemented for tensors with non-complex dtypes"
reproduction: `from octonion import octonion_exp; import torch; octonion_exp(torch.zeros(8))`
started: discovered during UAT test 6

## Eliminated

(none needed -- root cause found on first hypothesis)

## Evidence

- timestamp: 2026-03-08T07:00:00Z
  checked: _operations.py lines 17-57 (octonion_exp) and 60-103 (octonion_log)
  found: Both functions accept `o` typed as `Octonion` but call `o.real` and `o.imag` with no type guard. When a raw `torch.Tensor` is passed, these resolve to PyTorch's `.real` and `.imag` properties, not the Octonion class properties.
  implication: The type annotation says `Octonion` but nothing enforces it; raw tensors pass through and hit PyTorch's complex-only `.imag`.

- timestamp: 2026-03-08T07:00:00Z
  checked: _octonion.py lines 52-59 (Octonion.real and Octonion.imag properties)
  found: Octonion.real returns `self._data[..., 0]` (scalar part). Octonion.imag returns `self._data[..., 1:]` (7 imaginary components). These are custom properties that work on real-dtype tensors.
  implication: The operations work correctly when given an actual Octonion instance. The bug only manifests when a raw tensor is passed.

- timestamp: 2026-03-08T07:00:00Z
  checked: UAT test 6 reproduction command
  found: The test passes `torch.zeros(8)` (a plain float32 tensor) directly to `octonion_exp`. No Octonion wrapping.
  implication: The API has a dual problem -- (1) the functions don't wrap/coerce raw tensors into Octonion instances, and (2) the function names suggest they operate on raw tensors (like `octonion_mul` does) but they actually require Octonion objects.

- timestamp: 2026-03-08T07:00:00Z
  checked: octonion_mul function signature vs octonion_exp/octonion_log
  found: `octonion_mul(a, b)` operates on raw `torch.Tensor` shapes [..., 8]. But `octonion_exp(o)` and `octonion_log(o)` require `Octonion` instances. This is an inconsistent API contract -- functions with the `octonion_` prefix have mixed input type expectations.
  implication: Users naturally expect all `octonion_*` functions to accept the same input type.

## Resolution

root_cause: |
  `octonion_exp` and `octonion_log` in `_operations.py` call `o.real` and `o.imag` assuming `o` is an `Octonion` instance (which defines custom `.real` and `.imag` properties that slice the 8-component tensor). However, these functions have no type guard or auto-coercion. When a raw `torch.Tensor` (real dtype like float32) is passed, `.real` resolves to PyTorch's `Tensor.real` (no-op for real tensors) and `.imag` resolves to PyTorch's `Tensor.imag` which raises `RuntimeError: imag is not implemented for tensors with non-complex dtypes`.

  The secondary issue is API inconsistency: `octonion_mul` accepts raw tensors directly, but `octonion_exp`/`octonion_log` require Octonion instances, despite all sharing the `octonion_` prefix.

  Affected lines:
  - _operations.py:31 -- `a = o.real` (works on raw tensor but gives wrong semantics)
  - _operations.py:32 -- `v = o.imag` (crashes on raw tensor)
  - _operations.py:77 -- `a = o.real` (same pattern in octonion_log)
  - _operations.py:78 -- `v = o.imag` (same crash in octonion_log)
  - _operations.py:80 -- `q_norm = o.norm()` (would also fail on raw tensor -- Tensor.norm() has different semantics)

fix: (not applied -- diagnosis only)
verification: (not applied -- diagnosis only)
files_changed: []
