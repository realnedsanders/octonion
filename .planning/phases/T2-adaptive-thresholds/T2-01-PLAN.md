---
phase: T2-adaptive-thresholds
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/octonion/trie.py
  - tests/test_threshold_policy.py
  - tests/test_trie.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "OctonionTrie accepts a ThresholdPolicy object at construction"
    - "GlobalPolicy reproduces identical behavior to current hardcoded thresholds"
    - "All 18 existing trie tests pass without modification"
    - "ThresholdPolicy.on_insert is called after each insertion for policy updates"
    - "Consolidation uses policy-provided min_share and min_count"
  artifacts:
    - path: "src/octonion/trie.py"
      provides: "ThresholdPolicy ABC, GlobalPolicy, PerNodeEMAPolicy, PerNodeMeanStdPolicy, DepthPolicy, AlgebraicPurityPolicy stubs, OctonionTrie refactored to use policy"
      contains: "class ThresholdPolicy"
    - path: "tests/test_threshold_policy.py"
      provides: "Unit tests for all policy classes and backward compatibility"
      contains: "def test_global_policy_backward_compat"
  key_links:
    - from: "src/octonion/trie.py"
      to: "OctonionTrie._find_best_child"
      via: "self.policy.get_assoc_threshold(child, node.depth)"
      pattern: "self\\.policy\\.get_assoc_threshold"
    - from: "src/octonion/trie.py"
      to: "OctonionTrie._ruminate"
      via: "self.policy.get_sim_threshold(node, node.depth)"
      pattern: "self\\.policy\\.get_sim_threshold"
    - from: "src/octonion/trie.py"
      to: "OctonionTrie._consolidate_node"
      via: "self.policy.get_consolidation_params(node, node.depth)"
      pattern: "self\\.policy\\.get_consolidation_params"
---

<objective>
Add pluggable ThresholdPolicy abstraction to OctonionTrie and implement all 6 strategy classes.

Purpose: Per D-10, the trie must not know how thresholds are set. This is the foundation for all adaptive threshold experiments. Per D-04, adaptive thresholds apply during both insert and query.

Output: Refactored trie.py with ThresholdPolicy ABC, GlobalPolicy (baseline), PerNodeEMAPolicy, PerNodeMeanStdPolicy, DepthPolicy, AlgebraicPurityPolicy, plus stub MetaTriePolicy and HybridPolicy. Full backward compatibility with existing API.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/T2-adaptive-thresholds/T2-CONTEXT.md
@.planning/phases/T2-adaptive-thresholds/T2-RESEARCH.md

@src/octonion/trie.py
@tests/test_trie.py

<interfaces>
<!-- Current trie.py threshold usage points that must be refactored -->

Line 98: self.assoc_threshold = associator_threshold (float)
Line 111: self.sim_threshold = similarity_threshold (float)
Line 145: if assoc_norm < self.assoc_threshold:
Line 176: if key_sim < self.sim_threshold * 0.5:
Line 179: return sum(sims) / len(sims) > self.sim_threshold * 0.3
Line 262: if assoc_norm < self.assoc_threshold and self._ruminate(child, x):
Line 269: if assoc_norm >= self.assoc_threshold:
Lines 400-404: child.insert_count / max(total, 1) < 0.05 and child.insert_count < 3
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: ThresholdPolicy ABC and all strategy implementations</name>
  <files>src/octonion/trie.py</files>
  <read_first>
    - src/octonion/trie.py (current threshold usage at lines 98, 111, 145, 176, 179, 262, 269, 400-404)
    - src/octonion/_octonion.py (associator function, needed for AlgebraicPurityPolicy)
    - src/octonion/_fano.py (Fano plane triples, needed for subalgebra routing context)
  </read_first>
  <action>
Add ThresholdPolicy abstraction and all 6 strategy classes to trie.py. Place them ABOVE the OctonionTrie class definition.

**ThresholdPolicy ABC** (per D-10):
```python
from abc import ABC, abstractmethod

class ThresholdPolicy(ABC):
    @abstractmethod
    def get_assoc_threshold(self, node: TrieNode, depth: int) -> float: ...
    @abstractmethod
    def get_sim_threshold(self, node: TrieNode, depth: int) -> float: ...
    @abstractmethod
    def get_consolidation_params(self, node: TrieNode, depth: int) -> tuple[float, int]: ...
    def on_insert(self, node: TrieNode, x: torch.Tensor, assoc_norm: float) -> None:
        pass  # Optional hook for policy updates (EMA, etc.)
```

**GlobalPolicy** (baseline, preserves current behavior):
```python
class GlobalPolicy(ThresholdPolicy):
    def __init__(self, assoc_threshold: float = 0.3, sim_threshold: float = 0.1,
                 min_share: float = 0.05, min_count: int = 3):
        self.assoc_threshold = assoc_threshold
        self.sim_threshold = sim_threshold
        self.min_share = min_share
        self.min_count = min_count
    def get_assoc_threshold(self, node, depth): return self.assoc_threshold
    def get_sim_threshold(self, node, depth): return self.sim_threshold
    def get_consolidation_params(self, node, depth): return self.min_share, self.min_count
```

**PerNodeEMAPolicy** (D-01 strategy 1): Uses `_policy_state` dict on TrieNode keyed by `id(self)` policy. Each node stores EMA mean and variance of observed associator norms. Threshold = mean + k * std. Falls back to `base` threshold until node has >= 3 observations. Hyperparameters: `alpha` (EMA decay, default 0.1), `k` (std multiplier, default 1.5), `base_assoc` (fallback, default 0.3), `sim_threshold`, `min_share`, `min_count`.

**PerNodeMeanStdPolicy** (D-01 strategy 2): Like EMA but uses running mean and variance (Welford's online algorithm). No decay -- all observations weighted equally. Same interface as EMA. Hyperparameters: `k` (std multiplier), `base_assoc`, `sim_threshold`, `min_share`, `min_count`.

**DepthPolicy** (D-01 strategy 3): `threshold = base_assoc * decay_factor ^ depth`. Decay < 1 = tighter with depth, decay > 1 = looser with depth. Hyperparameters: `base_assoc`, `decay_factor` (default 1.0), `sim_threshold`, `min_share`, `min_count`.

**AlgebraicPurityPolicy** (D-01 strategy 4): Uses two independent signals: (a) variance of associator norms in node's buffer, (b) variance of inner products between buffer entries and routing key. Low variance = high purity = can tighten threshold. Threshold = base * (1 + sensitivity * combined_signal). Hyperparameters: `base_assoc`, `assoc_weight` (weight for assoc norm variance signal), `sim_weight` (weight for similarity variance signal), `sensitivity`, `sim_threshold`, `min_share`, `min_count`.

**MetaTriePolicy** (D-19, stub): Full implementation deferred to Plan 07. Stub class that raises NotImplementedError with message "MetaTriePolicy requires setup via configure_meta_trie()". Store placeholder attributes: `meta_trie: OctonionTrie | None = None`, `signal_encoding`, `feedback_signal`, `update_frequency`.

**HybridPolicy** (D-09, stub): Combines two ThresholdPolicy instances. Stub that raises NotImplementedError with message "HybridPolicy requires configure() with two base policies". Store: `policy_a: ThresholdPolicy | None = None`, `policy_b: ThresholdPolicy | None = None`, `combination: str = "mean"`.

**OctonionTrie refactoring** (per D-04 -- both insert AND query):
1. Add `policy: ThresholdPolicy | None = None` parameter to `__init__`. If None, create `GlobalPolicy(associator_threshold, similarity_threshold)`. Store as `self.policy`.
2. Keep `self.assoc_threshold` and `self.sim_threshold` as properties delegating to `self.policy.get_assoc_threshold(self.root, 0)` and `self.policy.get_sim_threshold(self.root, 0)` for backward compatibility.
3. Replace `self.assoc_threshold` in `_find_best_child` (line 145) with `self.policy.get_assoc_threshold(child, node.depth)`.
4. Replace `self.sim_threshold` in `_ruminate` (lines 176, 179) with `sim_thresh = self.policy.get_sim_threshold(node, node.depth)`, then use `sim_thresh * 0.5` and `sim_thresh * 0.3`.
5. Replace hardcoded `0.05` and `3` in `_consolidate_node` (lines 400-404) with `min_share, min_count = self.policy.get_consolidation_params(node, node.depth)`.
6. In `insert`, after line 266 (`node.buffer.append(...)`), call `self.policy.on_insert(node, x, assoc_norm)`.
7. In `insert`, also call `self.policy.on_insert` after `_create_child` returns (pass assoc_norm=float('inf') for new child creation).
8. In `insert` line 262, use `self.policy.get_assoc_threshold(child, node.depth)` instead of `self.assoc_threshold`.
9. In `insert` line 269, use `self.policy.get_assoc_threshold(child, node.depth)` instead of `self.assoc_threshold`.
10. Ensure `query` also uses policy (line 314 calls `_find_best_child` which already uses policy -- confirm no direct `self.assoc_threshold` usage in query path).

**Per-node state storage**: Add optional `_policy_state: dict` field to TrieNode dataclass (default_factory=dict). Policies store their per-node state here keyed by policy type name string (e.g., `"ema_mean"`, `"ema_var"`, `"ema_count"`, `"welford_mean"`, `"welford_M2"`, `"welford_count"`). This avoids id()-based lookups that break with pickling.

**Exports**: Add to module `__all__` or ensure all policy classes are importable from `octonion.trie`: ThresholdPolicy, GlobalPolicy, PerNodeEMAPolicy, PerNodeMeanStdPolicy, DepthPolicy, AlgebraicPurityPolicy, MetaTriePolicy, HybridPolicy.
  </action>
  <verify>
    <automated>docker compose run --rm dev uv run pytest tests/test_trie.py -x -v</automated>
  </verify>
  <acceptance_criteria>
    - trie.py contains `class ThresholdPolicy(ABC):`
    - trie.py contains `class GlobalPolicy(ThresholdPolicy):`
    - trie.py contains `class PerNodeEMAPolicy(ThresholdPolicy):`
    - trie.py contains `class PerNodeMeanStdPolicy(ThresholdPolicy):`
    - trie.py contains `class DepthPolicy(ThresholdPolicy):`
    - trie.py contains `class AlgebraicPurityPolicy(ThresholdPolicy):`
    - trie.py contains `class MetaTriePolicy(ThresholdPolicy):`
    - trie.py contains `class HybridPolicy(ThresholdPolicy):`
    - trie.py contains `self.policy.get_assoc_threshold(`
    - trie.py contains `self.policy.get_sim_threshold(`
    - trie.py contains `self.policy.get_consolidation_params(`
    - trie.py contains `self.policy.on_insert(`
    - OctonionTrie.__init__ accepts `policy: ThresholdPolicy | None = None`
    - All 18 existing test_trie.py tests pass (backward compatibility)
    - TrieNode dataclass has `_policy_state: dict` field
  </acceptance_criteria>
  <done>All 8 ThresholdPolicy classes defined, OctonionTrie refactored to use policy object at all threshold comparison points, all existing tests pass unchanged</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: ThresholdPolicy unit tests</name>
  <files>tests/test_threshold_policy.py</files>
  <read_first>
    - src/octonion/trie.py (after Task 1 modifications -- read the ThresholdPolicy classes)
    - tests/test_trie.py (existing test patterns, helpers like _make_aligned_centers, _generate_samples)
  </read_first>
  <behavior>
    - Test 1: GlobalPolicy with default args returns assoc=0.3, sim=0.1, consolidation=(0.05, 3) for any node/depth
    - Test 2: GlobalPolicy backward compat -- OctonionTrie(associator_threshold=0.5) produces identical results to OctonionTrie(policy=GlobalPolicy(assoc_threshold=0.5)) on 7-category classification (same accuracy, same n_nodes, same stats)
    - Test 3: PerNodeEMAPolicy returns base threshold for node with < 3 observations, adapts after 10+ insertions
    - Test 4: PerNodeEMAPolicy.on_insert updates node._policy_state with ema_mean and ema_var keys
    - Test 5: PerNodeMeanStdPolicy converges to sample mean + k*std after many insertions
    - Test 6: DepthPolicy with decay=0.8 returns base*0.8^depth (e.g., depth=3 -> base*0.512)
    - Test 7: DepthPolicy with decay=1.2 returns base*1.2^depth (increases with depth)
    - Test 8: AlgebraicPurityPolicy returns base threshold when buffer is empty, adjusts after buffer fills
    - Test 9: MetaTriePolicy stub raises NotImplementedError on get_assoc_threshold
    - Test 10: HybridPolicy stub raises NotImplementedError on get_assoc_threshold
    - Test 11: OctonionTrie with PerNodeEMAPolicy produces different tree structure than GlobalPolicy on same data (adaptive changes behavior)
    - Test 12: All policies honor D-02 (no category labels used in threshold computation)
  </behavior>
  <action>
Create tests/test_threshold_policy.py with the 12 tests above.

Import from octonion.trie: OctonionTrie, TrieNode, ThresholdPolicy, GlobalPolicy, PerNodeEMAPolicy, PerNodeMeanStdPolicy, DepthPolicy, AlgebraicPurityPolicy, MetaTriePolicy, HybridPolicy.

Reuse test helpers from test_trie.py: copy `_make_aligned_centers`, `_generate_samples`, `_accuracy` into this file (or import if test_trie exports them).

Use the same test data generation pattern: 7 categories aligned with Fano subalgebras, 20 samples per category, noise=0.05, seed=99.

For Test 2 (backward compat): Create two tries with same seed, insert same data in same order, compare `.stats()` dicts and accuracy. They MUST be identical.

For Test 11 (adaptive changes behavior): Use PerNodeEMAPolicy with aggressive k=0.5 (very tight adaptive threshold). Compare n_nodes between GlobalPolicy and PerNodeEMAPolicy tries -- they should differ.

For Test 12 (D-02 unsupervised): Verify that no ThresholdPolicy method signature accepts `category` parameter. Inspect all concrete policy classes' get_assoc_threshold, get_sim_threshold, get_consolidation_params, on_insert method signatures.
  </action>
  <verify>
    <automated>docker compose run --rm dev uv run pytest tests/test_threshold_policy.py -x -v</automated>
  </verify>
  <acceptance_criteria>
    - tests/test_threshold_policy.py exists
    - tests/test_threshold_policy.py contains `def test_global_policy_backward_compat`
    - tests/test_threshold_policy.py contains `def test_per_node_ema_adapts`
    - tests/test_threshold_policy.py contains `def test_depth_policy_decay`
    - tests/test_threshold_policy.py contains `def test_meta_trie_stub_raises`
    - tests/test_threshold_policy.py contains `def test_hybrid_stub_raises`
    - tests/test_threshold_policy.py contains `def test_unsupervised_constraint`
    - All 12 tests pass
    - docker compose run --rm dev uv run pytest tests/test_trie.py tests/test_threshold_policy.py -x exits 0
  </acceptance_criteria>
  <done>12 ThresholdPolicy unit tests pass, backward compatibility verified, all policies validated as unsupervised per D-02</done>
</task>

</tasks>

<verification>
docker compose run --rm dev uv run pytest tests/test_trie.py tests/test_threshold_policy.py -x -v
</verification>

<success_criteria>
- ThresholdPolicy ABC and 8 concrete classes exist in trie.py
- OctonionTrie uses policy object at all 6 threshold comparison points
- All 18 existing trie tests pass (zero regression)
- 12 new policy tests pass
- Backward compatibility verified: same data + same params = identical results with/without explicit GlobalPolicy
</success_criteria>

<output>
After completion, create `.planning/phases/T2-adaptive-thresholds/T2-01-SUMMARY.md`
</output>
