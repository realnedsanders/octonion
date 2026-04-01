# Phase T2: After Action Report — Adaptive Threshold Sweep Failures

**Date:** 2026-04-01
**Scope:** Investigation into why all adaptive threshold strategies produced identical or near-identical results to the global baseline, with the meta-trie showing zero sensitivity to hyperparameters.

## Observation

All adaptive strategies (EMA, mean+std, depth, purity, meta-trie, hybrid) produced nearly identical accuracy across all benchmarks, regardless of hyperparameters. Meta-trie and purity produced exactly ONE distinct accuracy per benchmark. EMA varied only by `base_assoc`, with alpha/k having zero effect. Global was the only policy type that showed genuine parameter sensitivity.

| Policy Type | Distinct accuracies (MNIST) | Sensitivity to hyperparams |
|-------------|---------------------------|---------------------------|
| global      | 10                        | Yes — assoc_threshold, sim_threshold, noise all matter |
| ema         | 2                         | Only base_assoc matters; alpha, k have zero effect |
| mean_std    | 2                         | Only base_assoc matters; k has zero effect |
| depth       | 4                         | decay_factor has small effect |
| purity      | 1                         | Zero — all configs identical |
| meta_trie   | 1                         | Zero — all configs identical |
| hybrid      | 3                         | Minimal — driven by sub-policy base values |

## Root Causes

### RC-1: `float("inf")` poisoning of per-node state (CRITICAL)

**Location:** `src/octonion/trie.py` lines 762, 769, 792, 809, 817

When a new child node is created, `on_insert` is called with `assoc_norm=float("inf")`:
```python
self.policy.on_insert(child, x, float("inf"))  # line 762
```

This is the very first `on_insert` call for that node (count==0), so the EMA policy sets:
```python
node._policy_state["ema_mean"] = assoc_norm  # = inf
```

All subsequent EMA updates compute `(1-alpha) * inf + alpha * real_value = inf`. The infinity poisons the running average permanently. The same issue affects PerNodeMeanStdPolicy, AlgebraicPurityPolicy, and MetaTriePolicy.

**Impact:** 199 of 200 nodes in a test trie had `ema_mean=inf`, meaning they ALWAYS fall back to `base_assoc` (since `inf + k*std = inf > any threshold`). The adaptive policies degenerate to a global policy with threshold = `base_assoc`.

**Why `float("inf")` was used:** The original intent was to signal "this is a new child, there is no meaningful associator norm" — when a new child is created because no compatible child exists, the "associator norm" that triggered the split is conceptually infinite (maximally incompatible). But this value was meant to be a sentinel, not an actual statistic to track.

### RC-2: Sparse `on_insert` coverage (SIGNIFICANT)

**Location:** `src/octonion/trie.py` insert() method

`on_insert` is only called on the **final destination node** of each insertion — the node where the sample actually lands. It is NOT called on intermediate nodes that the sample routes through. In a trie of depth 5-7, a sample passes through ~5 nodes but only the leaf gets `on_insert`.

**Impact:** Only 22 of 200 nodes received `on_insert` more than once. Most internal nodes have `ema_count=1` (from creation), which is below `min_obs=3`, so they return `base_assoc`. The adaptive policies never accumulate enough observations on internal nodes to actually adapt.

### RC-3: Root node never receives `on_insert` (MODERATE)

**Location:** `src/octonion/trie.py` insert() method

The root node is the entry point for all routing. It is counted (`_count(node, category)`) but never receives `on_insert`. Since the root's threshold determines whether to route to any child at all, having it stuck at `base_assoc` means the first routing decision is always identical to GlobalPolicy.

## Why This Wasn't Caught

1. **Unit tests tested policy mechanics, not trie integration.** Tests verified that `PerNodeEMAPolicy.on_insert` updates `_policy_state` correctly in isolation, and that `get_assoc_threshold` returns adapted values when state exists. But no test verified that the TRIE passes meaningful values to `on_insert` during actual classification.

2. **The `float("inf")` sentinel was semantically reasonable** — new child creation happens because no compatible child exists, so the "incompatibility" is maximal. But it breaks the statistical tracking that `on_insert` was designed to support.

3. **Sweep results looked plausible at first glance.** The adaptive strategies still achieved ~77% on MNIST — competitive but lower than global's 80%. This looked like "adaptive doesn't help" rather than "adaptive is broken." Only the observation that ALL meta-trie hyperparameters produce EXACTLY the same result revealed the problem.

4. **The plan checker verified structural properties** (task has read_first, acceptance_criteria, etc.) but not cross-plan data contracts at the semantic level. It couldn't catch that "pass `assoc_norm` to `on_insert`" means something different at line 778 (real associator norm) vs lines 762/769 (sentinel `inf`).

## Fixes Required

### Fix 1: Replace `float("inf")` with a meaningful value or sentinel handling

Options:
- **A. Use 0.0 for new children:** New children are compatible (they were just created to match the input). `assoc_norm=0.0` signals "perfect compatibility" which is correct — the child's routing key IS the input.
- **B. Skip `on_insert` for new children entirely:** Don't call `on_insert` at creation time. Let the first real routing-through-this-node call establish the baseline.
- **C. Add sentinel handling in policies:** Each policy's `on_insert` checks for `inf` and ignores it. Brittle — requires every policy to handle this.

**Recommendation: Option A.** A newly created child's routing key IS the input, so associator norm with itself is genuinely 0. This is the most semantically correct value.

### Fix 2: Call `on_insert` on intermediate nodes during routing

When a sample routes through an internal node (the `continue` path), `on_insert` should be called on that node with the computed `assoc_norm`. This gives internal nodes enough observations to adapt.

### Fix 3: Call `on_insert` on root

The root node should receive `on_insert` with the associator norm computed between the input and the root's routing key.

### Fix 4: Re-run affected sweeps

After fixing the trie, all adaptive strategy sweeps (EMA, mean+std, purity, meta-trie, hybrid) need to be re-run. The global sweep results are valid and don't need re-running.

## Sweep Infrastructure Bugs (Fixed)

During execution, three additional bugs were found and fixed in the sweep infrastructure:

### Bug 1: `assoc_threshold` → `base_assoc` parameter name mismatch

**Location:** `scripts/sweep/sweep_runner.py` `_run_single_config` worker function

The worker injected SweepConfig field names (`assoc_threshold`) as constructor kwargs for ALL policy types, but only `GlobalPolicy` uses `assoc_threshold` — all adaptive policies use `base_assoc`.

**Fix:** Scoped injection to `global` type with empty `policy_params` only. Adaptive policies already have correctly-named params in their `policy_params` JSON.

### Bug 2: HybridPolicy serialization/deserialization

**Location:** `scripts/sweep/sweep_runner.py` `_make_policy` function

`HybridPolicy` takes constructed policy objects (`policy_a`, `policy_b`), but the sweep runner tried to pass serialized type names as kwargs. Added a `hybrid` branch to `_make_policy` that constructs sub-policies recursively.

### Bug 3: Analysis script epoch mismatch + config_id assumption

**Location:** `scripts/sweep/sweep_analysis.py` multiple functions

- Used `MAX(epoch)` across all configs for a policy type, but multi-seed validation used different epoch counts than the original sweep (3 vs 5). Fixed to use each config's own final epoch.
- Assumed same `config_id` for multi-seed data, but `run_hybrid_validation.py` assigns unique config_ids per seed. Fixed to group by parameter values instead of config_id.
- Missing `numpy.bool_` handler in JSON serialization.

## Lessons Learned

1. **Integration tests > unit tests for policy effects.** A test that inserts 100 samples through a trie with PerNodeEMAPolicy and then checks that `ema_mean != inf` on at least 50% of nodes would have caught RC-1 immediately.

2. **Sentinel values in statistical trackers are dangerous.** Using `inf` as "no meaningful value" in a system that computes running averages is a category error. The sentinel propagates through arithmetic and corrupts all downstream computation.

3. **Sweep results that show zero hyperparameter sensitivity are a diagnostic signal.** When 96 meta-trie configurations produce exactly one accuracy value, the mechanism is broken, not irrelevant.

4. **Cross-plan data contracts need semantic validation.** The plan checker verified that `on_insert` was called (structural check) but not that the values passed to it were usable (semantic check).
