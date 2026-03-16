---
status: diagnosed
trigger: "AttributeError: type object 'QuaternionBatchNorm' has no attribute 'name'"
created: 2026-03-16T00:00:00Z
updated: 2026-03-16T00:00:00Z
goal: find_root_cause_only
---

## Current Focus

hypothesis: The test script (or a variation of it) uses `BN.name` instead of `BN.__name__` at line 13; QuaternionBatchNorm is a plain nn.Module subclass with no `name` class attribute, so the access fails.
test: Static code analysis — all call sites accessing `.name` on BN class objects
expecting: The usage of `.name` on a BN *class* (not instance) is the exact cause
next_action: DONE — root cause confirmed by elimination

## Symptoms

expected: QuaternionBatchNorm and OctonionBatchNorm forward passes succeed under AMP autocast with no NaN and no errors
actual: AttributeError: type object 'QuaternionBatchNorm' has no attribute 'name'
errors: "Traceback (most recent call last): File \"<string>\", line 13, in <module> AttributeError: type object 'QuaternionBatchNorm' has no attribute 'name'"
reproduction: Run the AMP-safe BN test from UAT test 9 (or a variant of it with `BN.name` instead of `BN.__name__`)
started: reported in 03-UAT.md as test 9 failure

## Eliminated

- hypothesis: The error comes from inside _normalization.py forward() or __init__()
  evidence: Read all 543 lines of _normalization.py — no `.name` attribute access on any class object; only `self.*` instance attribute accesses
  timestamp: 2026-03-16

- hypothesis: PyTorch internals (torch.amp.autocast) call `.name` on the BN class
  evidence: torch.amp.autocast receives device_type (a string from x.device.type) and enabled (a bool); it has no code path that would call `.name` on the caller's class. The error traceback shows File "<string>", line 13, in <module> — meaning the error is directly in the -c script itself, not inside a library call (no additional frames shown)
  timestamp: 2026-03-16

- hypothesis: _comparison.py algebra.name call is the source
  evidence: _comparison.py:325 and :388 call algebra.name on AlgebraType enum members, not on BN classes. These are only called during run_comparison(), not during a simple BN instantiation test.
  timestamp: 2026-03-16

- hypothesis: register_buffer or nn.Module internals access .name on the class
  evidence: PyTorch's register_buffer takes a string name parameter (not a class); nn.Module.__setattr__ and friends operate on string names and tensor values, not on class objects
  timestamp: 2026-03-16

## Evidence

- timestamp: 2026-03-16
  checked: src/octonion/baselines/_normalization.py (all 543 lines)
  found: No `.name` attribute access on any class object. QuaternionBatchNorm and OctonionBatchNorm are standard nn.Module subclasses. Only `.device.type` (on x tensor) and self.* instance attributes are accessed.
  implication: The bug is NOT inside _normalization.py

- timestamp: 2026-03-16
  checked: grep for `.name` across all .py files in src/ and scripts/
  found: Only two usages of .name on a non-trivial object: _comparison.py:325 (`algebra.name`, AlgebraType enum member) and _comparison.py:388 (same). profile_baseline.py:355 uses `a.name` for ProfilerActivity enum members.
  implication: No code in the library calls .name on a BN class. The call must originate from the test script itself.

- timestamp: 2026-03-16
  checked: UAT test 9 expected command string (03-UAT.md line 52)
  found: The expected command uses `BN.__name__` (correct dunder attribute). However, the reported traceback is "line 13, in <module>" — the UAT command only produces 9 lines when expanded. The user ran a DIFFERENT (13-line) version of the script.
  implication: The user's actual test script used `BN.name` (no double underscores) instead of `BN.__name__` at line 13. This is a typo in the test invocation, not a bug in the library.

- timestamp: 2026-03-16
  checked: QuaternionBatchNorm and OctonionBatchNorm class definitions for any `.name` class attribute
  found: Neither class defines a `name` class attribute. Both inherit from nn.Module. nn.Module does not define a `.name` class attribute either (it uses `_get_name()` instance method for string representation).
  implication: Accessing `.name` on either class will always raise AttributeError. A `name` class attribute must be added if the test requires it, OR the test must use `.__name__` instead.

## Resolution

root_cause: The test script calls `BN.name` (accessing a non-existent class attribute) instead of `BN.__name__` (the correct Python dunder for a class's name); neither QuaternionBatchNorm nor OctonionBatchNorm define a `name` class attribute, and nn.Module does not provide one either.

fix: (not applied — diagnose-only mode) Either (a) change `BN.name` to `BN.__name__` in the test script, or (b) add a `name: ClassVar[str]` class attribute to QuaternionBatchNorm and OctonionBatchNorm if a short programmatic name is needed for these classes.

verification: not performed
files_changed: []
