---
status: resolved
trigger: "RuntimeError: expected scalar type Double but found Float in OctonionLinear forward with float32 input"
created: 2026-03-08T07:00:00Z
updated: 2026-03-08T07:30:00Z
---

## Current Focus

hypothesis: CONFIRMED - OctonionLinear defaults its parameters to float64 while typical user input is float32, and torch.einsum requires matching dtypes
test: Traced code path from OctonionLinear.__init__ through forward to octonion_mul
expecting: dtype mismatch between parameters (float64) and input (float32)
next_action: Return diagnosis

## Symptoms

expected: OctonionLinear()(torch.randn(4, 8)) produces output shape [4,8] with gradients
actual: RuntimeError: expected scalar type Double but found Float
errors: "RuntimeError: expected scalar type Double but found Float" in octonion_mul einsum
reproduction: `from octonion import OctonionLinear; import torch; layer = OctonionLinear(); x = torch.randn(4, 8); y = layer(x)`
started: Discovered during UAT Test 5

## Eliminated

(none needed - root cause identified on first hypothesis)

## Evidence

- timestamp: 2026-03-08T07:01:00Z
  checked: _multiplication.py line 35
  found: STRUCTURE_CONSTANTS built with dtype=torch.float64. However, octonion_mul (line 77) casts C to match input dtype via `.to(device=a.device, dtype=a.dtype)` - so STRUCTURE_CONSTANTS itself is NOT the direct cause.
  implication: The module-level constant is float64 but octonion_mul handles this correctly. The bug must be elsewhere.

- timestamp: 2026-03-08T07:02:00Z
  checked: _linear.py line 34
  found: OctonionLinear.__init__ defaults dtype=torch.float64. Parameters self.a and self.b are created as float64 tensors.
  implication: When user passes float32 input, layer.a and layer.b are float64 Parameters.

- timestamp: 2026-03-08T07:03:00Z
  checked: _linear.py lines 55-56
  found: In forward(), a_expanded = self.a.expand_as(x) then octonion_mul(a_expanded, x). a_expanded is float64 (from Parameter), x is float32 (from user).
  implication: octonion_mul receives a=float64 and b=float32 as its two arguments.

- timestamp: 2026-03-08T07:04:00Z
  checked: _multiplication.py line 77
  found: C = STRUCTURE_CONSTANTS.to(device=a.device, dtype=a.dtype) casts C to match `a`'s dtype (float64). Then einsum("...i, ijk, ...j -> ...k", a, C, b) has a=float64, C=float64, b=float32. Three operands with mismatched dtypes causes the RuntimeError.
  implication: ROOT CAUSE CONFIRMED. The einsum has three tensor operands and two are float64 while b (the user input x) is float32. torch.einsum does not auto-promote.

## Resolution

root_cause: |
  OctonionLinear.__init__ defaults dtype to torch.float64 (line 34 of _linear.py).
  When a user passes float32 input (the PyTorch default), the forward pass calls
  octonion_mul(a_expanded, x) where a_expanded is float64 (from the Parameter) and
  x is float32 (from the user). Inside octonion_mul, STRUCTURE_CONSTANTS is cast to
  match `a`'s dtype (float64), so the einsum receives two float64 tensors and one
  float32 tensor. torch.einsum does not auto-promote mixed dtypes, causing the
  RuntimeError.

  The issue has TWO contributing factors:
  1. OctonionLinear defaults to float64 parameters instead of float32 (the PyTorch convention).
  2. octonion_mul does not ensure both input tensors share the same dtype before the einsum.

  Factor 1 is the primary cause (wrong default), Factor 2 is a robustness gap (no dtype
  harmonization between the two input tensors).

fix: (diagnosis only)
verification: (diagnosis only)
files_changed: []
