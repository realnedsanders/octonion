---
status: resolved
trigger: "Octonion(random_octonion()) raises AttributeError: 'Octonion' object has no attribute 'shape'"
created: 2026-03-08T07:00:00Z
updated: 2026-03-08T07:30:00Z
---

## Current Focus

hypothesis: CONFIRMED - Two-sided API mismatch: random_octonion() returns Octonion, not tensor; Octonion.__init__ only accepts tensors
test: Read source code of both functions
expecting: Confirm return type of random_octonion and input expectations of Octonion.__init__
next_action: Return diagnosis

## Symptoms

expected: Octonion(random_octonion()) creates an Octonion and multiplication works to show non-commutativity
actual: AttributeError 'Octonion' object has no attribute 'shape' on line 36 of _octonion.py
errors: "AttributeError: 'Octonion' object has no attribute 'shape'"
reproduction: `from octonion import random_octonion, Octonion; Octonion(random_octonion())`
started: Discovered during UAT test 3

## Eliminated

(none needed - root cause found on first hypothesis)

## Evidence

- timestamp: 2026-03-08T07:00:00Z
  checked: src/octonion/_random.py random_octonion() return type
  found: random_octonion() returns Octonion(data) on line 44 - it returns an Octonion instance, not a raw tensor
  implication: The UAT test wraps Octonion(random_octonion()) which becomes Octonion(Octonion(data)) - double wrapping

- timestamp: 2026-03-08T07:00:00Z
  checked: src/octonion/_octonion.py Octonion.__init__ parameter handling
  found: __init__(self, data: torch.Tensor) accesses data.shape[-1] on line 36 with no guard for non-Tensor input. Octonion class uses __slots__ = ("_data",) which means it has no .shape attribute.
  implication: When an Octonion is passed as `data`, Python calls data.shape which fails because Octonion has no .shape attribute

- timestamp: 2026-03-08T07:00:00Z
  checked: Whether this is a UAT test design error or an API design error
  found: This is a two-sided issue. (1) The UAT test is arguably wrong because random_octonion() already returns an Octonion - wrapping it again is redundant. (2) However, the Octonion constructor is also fragile - it has no copy-constructor capability and provides a confusing error when given an Octonion instead of a clear type error.
  implication: The primary fix should be in the Octonion constructor (accept Octonion input as copy constructor), with the UAT test also being correctable to just use random_octonion() directly

## Resolution

root_cause: |
  TWO-SIDED API MISMATCH between random_octonion() and Octonion.__init__:

  1. random_octonion() (line 44 of _random.py) returns `Octonion(data)` -- an Octonion instance, not a raw tensor.
  2. Octonion.__init__ (line 36 of _octonion.py) expects a `torch.Tensor` and immediately accesses `data.shape[-1]`.
  3. The Octonion class uses `__slots__ = ("_data",)` and has no `.shape` attribute.
  4. When the UAT test calls `Octonion(random_octonion())`, it passes an Octonion as `data`.
  5. Line 36 does `data.shape[-1]` on an Octonion object -> AttributeError: 'Octonion' object has no attribute 'shape'.

  The constructor lacks a copy-constructor pattern (accepting an existing Octonion and extracting its .components tensor).
  The type hint says `data: torch.Tensor` but there is no runtime check enforcing this.
fix: (not applying - diagnosis only)
verification: (not applying - diagnosis only)
files_changed: []
