---
status: diagnosed
trigger: "TypeError: _tril_to_symmetric() missing 2 required positional arguments: 'rows' and 'cols' at tests/test_perf_equivalence.py:171"
created: 2026-03-16T00:00:00Z
updated: 2026-03-16T00:00:00Z
---

## Current Focus

hypothesis: The function signature in _normalization.py was updated to require `rows` and `cols` as mandatory parameters, but the committed version of the test file still calls it with only `(tril_flat, dim)`. The test file has unstaged local changes that fix the calls, but those changes have not been committed.
test: confirmed via git diff HEAD -- tests/test_perf_equivalence.py
expecting: n/a — diagnosis complete
next_action: return ROOT CAUSE FOUND

## Symptoms

expected: _tril_to_symmetric(flat, dim=N) passes with no TypeError
actual: TypeError: _tril_to_symmetric() missing 2 required positional arguments: 'rows' and 'cols'
errors: tests/test_perf_equivalence.py:171: TypeError: _tril_to_symmetric() missing 2 required positional arguments: 'rows' and 'cols'
reproduction: run committed test file against current _normalization.py (the test file on disk has the fix but it is unstaged)
started: after _tril_to_symmetric signature was changed to require rows/cols (commit after 8c88651)

## Eliminated

- hypothesis: test file is calling function with wrong argument order
  evidence: all test call sites in the working-tree version pass rows and cols correctly as keyword args
  timestamp: 2026-03-16

- hypothesis: function was broken inside _normalization.py internal callers
  evidence: internal callers at lines 380 and 538 pass self._tril_rows, self._tril_cols — they are correct
  timestamp: 2026-03-16

## Evidence

- timestamp: 2026-03-16
  checked: src/octonion/baselines/_normalization.py lines 23-25
  found: function signature is `def _tril_to_symmetric(tril_flat, dim, rows, cols)` — rows and cols are required positional args
  implication: any caller that omits rows and cols will get TypeError

- timestamp: 2026-03-16
  checked: git show 8c88651 -- tests/test_perf_equivalence.py
  found: original committed test calls are `_tril_to_symmetric(flat, dim=N)` — no rows or cols
  implication: the committed test is stale relative to the current function signature

- timestamp: 2026-03-16
  checked: git diff HEAD -- tests/test_perf_equivalence.py
  found: working-tree test file has all calls updated to `_tril_to_symmetric(flat, dim=N, rows=rows, cols=cols)` but these changes are NOT staged/committed
  implication: the test file on disk works, but if tests are run against the committed version (e.g. in CI or after a clean checkout), they fail with the TypeError

- timestamp: 2026-03-16
  checked: git log --oneline -- src/octonion/baselines/_normalization.py
  found: the `rows` and `cols` parameters were introduced in the Tier 1 optimization commit (ab1fd8e) or a subsequent commit; the test file update was done in local working tree only
  implication: committed test and committed implementation are out of sync

## Resolution

root_cause: The committed version of `tests/test_perf_equivalence.py` calls `_tril_to_symmetric(flat, dim=N)` with only two arguments, but the current implementation in `src/octonion/baselines/_normalization.py` (committed) requires four: `tril_flat`, `dim`, `rows`, `cols`. The fix exists only in the unstaged working-tree changes to the test file and was never committed.
fix: (not applied — diagnose only)
verification: (not applied — diagnose only)
files_changed: []
