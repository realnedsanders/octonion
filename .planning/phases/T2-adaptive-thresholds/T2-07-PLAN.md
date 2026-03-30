---
phase: T2-adaptive-thresholds
plan: 07
type: execute
wave: 6
depends_on: ["T2-05", "T2-06"]
files_modified:
  - src/octonion/trie.py
  - scripts/sweep/run_meta_trie_sweep.py
  - tests/test_threshold_policy.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "MetaTriePolicy fully implemented using a second OctonionTrie instance per D-12"
    - "Meta-trie categories are discretized threshold actions per D-13"
    - "Two input encodings tested independently per D-14"
    - "Two feedback signals tested independently per D-15"
    - "Multiple update frequencies tested per D-16"
    - "Both fixed and self-referential meta-trie variants tested per D-17"
    - "Convergence criterion tracked per D-18"
  artifacts:
    - path: "src/octonion/trie.py"
      provides: "Full MetaTriePolicy implementation"
      contains: "class MetaTriePolicy"
    - path: "scripts/sweep/run_meta_trie_sweep.py"
      provides: "Meta-trie sweep execution"
      contains: "def run_meta_trie_sweep"
    - path: "tests/test_threshold_policy.py"
      provides: "MetaTriePolicy unit tests (extended)"
      contains: "def test_meta_trie_convergence"
  key_links:
    - from: "src/octonion/trie.py MetaTriePolicy"
      to: "src/octonion/trie.py OctonionTrie"
      via: "MetaTriePolicy.meta_trie is an OctonionTrie instance per D-12"
      pattern: "self\\.meta_trie.*OctonionTrie"
---

<objective>
Implement full MetaTriePolicy and run meta-trie optimizer sweep.

Purpose: Per D-12, a second OctonionTrie acts as optimizer for the classifier trie -- same class, same algebra. Per D-08, this runs after simpler strategies to benefit from that understanding. This is the strongest thesis statement: the trie optimizes itself.

Output: Full MetaTriePolicy implementation, meta-trie sweep across input encodings, feedback signals, update frequencies, and self-referential variants.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/phases/T2-adaptive-thresholds/T2-CONTEXT.md
@.planning/phases/T2-adaptive-thresholds/T2-RESEARCH.md
@.planning/phases/T2-adaptive-thresholds/T2-01-SUMMARY.md
@.planning/phases/T2-adaptive-thresholds/T2-05-SUMMARY.md
@.planning/phases/T2-adaptive-thresholds/T2-06-SUMMARY.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: MetaTriePolicy full implementation</name>
  <files>src/octonion/trie.py, tests/test_threshold_policy.py</files>
  <read_first>
    - src/octonion/trie.py (MetaTriePolicy stub, OctonionTrie API, ThresholdPolicy ABC, TrieNode fields)
    - tests/test_threshold_policy.py (existing tests, test patterns)
    - .planning/phases/T2-adaptive-thresholds/T2-CONTEXT.md (D-12 through D-19)
  </read_first>
  <action>
Replace the MetaTriePolicy stub in trie.py with full implementation.

**MetaTriePolicy class** (per D-12, D-19):

```python
class MetaTriePolicy(ThresholdPolicy):
    """Meta-trie optimizer: a second OctonionTrie adapts classifier thresholds.

    Per D-12: Uses the same OctonionTrie class (not a subclass).
    Per D-13: Categories are discretized threshold actions.
    Per D-14: Two input encoding modes.
    Per D-15: Two feedback signal modes.
    """

    # Threshold actions per D-13
    ACTIONS = {
        0: -0.20,  # "tighten 20%"
        1: -0.10,  # "tighten 10%"
        2:  0.00,  # "keep"
        3:  0.10,  # "loosen 10%"
        4:  0.20,  # "loosen 20%"
    }

    def __init__(
        self,
        base_assoc: float = 0.3,
        sim_threshold: float = 0.1,
        min_share: float = 0.05,
        min_count: int = 3,
        signal_encoding: str = "signal_vector",  # or "algebraic" per D-14
        feedback_signal: str = "stability",       # or "accuracy" per D-15
        update_frequency: int = 100,              # per D-16: per-N-inserts
        self_referential: bool = False,            # per D-17
        meta_seed: int = 7919,
    ):
        self.base_assoc = base_assoc
        self.sim_threshold = sim_threshold
        self.min_share = min_share
        self.min_count = min_count
        self.signal_encoding = signal_encoding
        self.feedback_signal = feedback_signal
        self.update_frequency = update_frequency
        self.self_referential = self_referential

        # Create the meta-trie per D-12
        self.meta_trie = OctonionTrie(
            associator_threshold=base_assoc,
            similarity_threshold=sim_threshold,
            seed=meta_seed,
        )

        # Per-node threshold adjustments (accumulated from meta-trie decisions)
        self._node_adjustments: dict[int, float] = {}  # id(node) -> adjustment factor
        self._insert_counter = 0
        self._convergence_history: list[float] = []  # per D-18: track threshold change rate
        self._prev_adjustments: dict[int, float] = {}

    def get_assoc_threshold(self, node: TrieNode, depth: int) -> float:
        adjustment = self._node_adjustments.get(id(node), 0.0)
        return max(0.001, self.base_assoc * (1.0 + adjustment))

    def get_sim_threshold(self, node: TrieNode, depth: int) -> float:
        return self.sim_threshold

    def get_consolidation_params(self, node: TrieNode, depth: int) -> tuple[float, int]:
        return self.min_share, self.min_count

    def on_insert(self, node: TrieNode, x: torch.Tensor, assoc_norm: float) -> None:
        # Track stats in node._policy_state
        state = node._policy_state
        state.setdefault("meta_assoc_norms", []).append(assoc_norm)
        state["meta_insert_count"] = state.get("meta_insert_count", 0) + 1
        # Keep only last 100 norms to avoid unbounded memory
        if len(state["meta_assoc_norms"]) > 100:
            state["meta_assoc_norms"] = state["meta_assoc_norms"][-100:]

        self._insert_counter += 1
        if self._insert_counter % self.update_frequency == 0:
            self._update_thresholds(node)

    def _encode_signal_vector(self, node: TrieNode) -> torch.Tensor:
        """Encode node state as 8D signal vector per D-14 option 1."""
        state = node._policy_state
        norms = state.get("meta_assoc_norms", [0.0])
        norms_t = torch.tensor(norms, dtype=torch.float64)
        return torch.tensor([
            norms_t.mean().item(),           # assoc_norm_mean
            norms_t.std().item() if len(norms) > 1 else 0.0,  # assoc_norm_std
            len(node.children) / 7.0,        # branching_factor / 7
            node.insert_count / max(self._insert_counter, 1),  # insert_rate
            0.0,  # rumination_rate (computed from parent trie stats if available)
            node.depth / 15.0,               # depth / max_depth
            0.0,  # buffer_consistency (computed from buffer similarity)
            0.0,  # consolidation_rate
        ], dtype=torch.float64)

    def _encode_algebraic(self, node: TrieNode) -> torch.Tensor:
        """Use node's routing key as meta-trie input per D-14 option 2."""
        return node.routing_key.clone()

    def _compute_stability_signal(self, node: TrieNode) -> int:
        """Compute unsupervised stability signal per D-15 option 1.

        Returns action category (0-4) based on node stability indicators.
        Low rumination + balanced branching + consistent norms -> "keep" (2)
        """
        state = node._policy_state
        norms = state.get("meta_assoc_norms", [])
        if len(norms) < 3:
            return 2  # "keep" -- not enough data

        norms_t = torch.tensor(norms[-30:], dtype=torch.float64)
        cv = (norms_t.std() / norms_t.mean()).item() if norms_t.mean() > 1e-10 else 0.0

        # High CV = inconsistent = should tighten; Low CV = stable = can loosen
        if cv > 1.0:
            return 0  # tighten 20%
        elif cv > 0.5:
            return 1  # tighten 10%
        elif cv < 0.1:
            return 4  # loosen 20%
        elif cv < 0.2:
            return 3  # loosen 10%
        else:
            return 2  # keep

    def _update_thresholds(self, trigger_node: TrieNode) -> None:
        """Update threshold adjustments via meta-trie.

        Encodes trigger_node state, inserts into meta-trie,
        queries meta-trie for recommended action, applies adjustment.
        """
        # Encode input based on D-14
        if self.signal_encoding == "signal_vector":
            meta_input = self._encode_signal_vector(trigger_node)
        else:
            meta_input = self._encode_algebraic(trigger_node)

        # Determine category based on D-15
        if self.feedback_signal == "stability":
            action_cat = self._compute_stability_signal(trigger_node)
        else:
            action_cat = 2  # "keep" for accuracy mode (set externally)

        # Insert into meta-trie
        self.meta_trie.insert(meta_input, category=action_cat)

        # Query meta-trie for recommendation
        leaf = self.meta_trie.query(meta_input)
        recommended = leaf.dominant_category
        if recommended is not None and recommended in self.ACTIONS:
            adjustment = self.ACTIONS[recommended]
            self._node_adjustments[id(trigger_node)] = adjustment

        # Per D-17: self-referential -- meta-trie adapts its own thresholds
        if self.self_referential:
            meta_signal = self._encode_signal_vector(trigger_node)
            meta_action = self._compute_stability_signal(trigger_node)
            meta_leaf = self.meta_trie.query(meta_signal)
            if meta_leaf.dominant_category is not None:
                meta_adj = self.ACTIONS.get(meta_leaf.dominant_category, 0.0)
                self.meta_trie.assoc_threshold = max(
                    0.001, self.base_assoc * (1.0 + meta_adj)
                )

        # Per D-18: convergence tracking
        curr_adj = dict(self._node_adjustments)
        if self._prev_adjustments:
            changes = []
            for k in set(curr_adj) | set(self._prev_adjustments):
                old = self._prev_adjustments.get(k, 0.0)
                new = curr_adj.get(k, 0.0)
                changes.append(abs(new - old))
            change_rate = sum(changes) / max(len(changes), 1)
            self._convergence_history.append(change_rate)
        self._prev_adjustments = curr_adj

    @property
    def converged(self) -> bool:
        """Per D-18: converged if threshold change rate < 1%."""
        if len(self._convergence_history) < 3:
            return False
        return self._convergence_history[-1] < 0.01
```

Also add tests to tests/test_threshold_policy.py:
- test_meta_trie_uses_same_class: `isinstance(policy.meta_trie, OctonionTrie)` is True
- test_meta_trie_signal_encoding: signal_vector produces 8D tensor, algebraic uses routing_key
- test_meta_trie_actions: ACTIONS dict has 5 entries, values sum to 0 (symmetric)
- test_meta_trie_convergence_tracking: convergence_history grows after updates
- test_meta_trie_self_referential: self-referential mode updates meta_trie threshold
- Replace the stub test (test_meta_trie_stub_raises) with these functional tests
  </action>
  <verify>
    <automated>docker compose run --rm dev uv run pytest tests/test_threshold_policy.py -x -v -k "meta_trie"</automated>
  </verify>
  <acceptance_criteria>
    - trie.py MetaTriePolicy contains `self.meta_trie = OctonionTrie(` per D-12
    - trie.py MetaTriePolicy contains `ACTIONS = {` with 5 threshold actions per D-13
    - trie.py MetaTriePolicy contains `_encode_signal_vector` per D-14 option 1
    - trie.py MetaTriePolicy contains `_encode_algebraic` per D-14 option 2
    - trie.py MetaTriePolicy contains `_compute_stability_signal` per D-15 option 1
    - trie.py MetaTriePolicy contains `self.update_frequency` per D-16
    - trie.py MetaTriePolicy contains `self.self_referential` per D-17
    - trie.py MetaTriePolicy contains `self._convergence_history` per D-18
    - test_threshold_policy.py contains `def test_meta_trie_uses_same_class`
    - All meta_trie tests pass
  </acceptance_criteria>
  <done>MetaTriePolicy fully implemented with all D-12 through D-18 features, 5+ new unit tests pass</done>
</task>

<task type="auto">
  <name>Task 2: Meta-trie sweep execution</name>
  <files>scripts/sweep/run_meta_trie_sweep.py</files>
  <read_first>
    - src/octonion/trie.py (MetaTriePolicy constructor, hyperparameters from Task 1)
    - scripts/sweep/run_adaptive_sweep.py (adaptive sweep pattern)
    - scripts/sweep/sweep_runner.py (SweepRunner API)
    - .planning/phases/T2-adaptive-thresholds/T2-CONTEXT.md (D-14, D-15, D-16, D-17, D-32)
  </read_first>
  <action>
Create scripts/sweep/run_meta_trie_sweep.py per D-32 (uses same sweep infrastructure).

**Meta-trie sweep dimensions** (per D-14, D-15, D-16, D-17):

Dimension 1 -- Input encoding (per D-14, test independently):
- "signal_vector": 8D signal vector
- "algebraic": routing key / content octonion

Dimension 2 -- Feedback signal (per D-15, test independently):
- "stability": unsupervised (low rumination, balanced branching)
- "accuracy": supervised (held-out accuracy -- for comparison only)

Dimension 3 -- Update frequency (per D-16):
- per-100-inserts, per-1000-inserts, per-epoch (computed as per-N where N=train_size)

Dimension 4 -- Self-referential (per D-17):
- False: fixed meta-trie thresholds
- True: meta-trie adapts its own thresholds

Dimension 5 -- base_assoc: top-3 from global sweep
Dimension 6 -- sim_threshold: top-2 from global sweep

**Total configs**: 2 * 2 * 3 * 2 * 3 * 2 = 144 per benchmark, 720 total. Feasible.

**After initial sweep**, run expanded configs on top-10 with:
- Additional base_assoc values
- Full noise sweep per D-05
- Epoch sweep per D-06

**Convergence tracking** per D-18: For each config, record convergence_history from MetaTriePolicy. Store in a separate SQLite table `meta_convergence` with columns: config_id, benchmark, update_idx, change_rate, n_adjustments.

**CLI interface**:
```
python scripts/sweep/run_meta_trie_sweep.py --features-dir results/T2/features --db results/T2/sweep.db --workers 24
```

**Comparison output**: After sweep, print:
1. Best meta-trie config vs best global, best adaptive (strategies 1-4)
2. Input encoding comparison (signal_vector vs algebraic) per D-14
3. Feedback signal comparison (stability vs accuracy) per D-15
4. Update frequency comparison per D-16
5. Self-referential vs fixed comparison per D-17
6. Convergence statistics per D-18

Per D-32, MetaTriePolicy plugs into SweepRunner like any other policy.
  </action>
  <verify>
    <automated>docker compose run --rm dev uv run python scripts/sweep/run_meta_trie_sweep.py --help</automated>
  </verify>
  <acceptance_criteria>
    - scripts/sweep/run_meta_trie_sweep.py exists
    - run_meta_trie_sweep.py contains `MetaTriePolicy`
    - run_meta_trie_sweep.py contains signal_encoding options ["signal_vector", "algebraic"]
    - run_meta_trie_sweep.py contains feedback_signal options ["stability", "accuracy"]
    - run_meta_trie_sweep.py contains update_frequency values
    - run_meta_trie_sweep.py contains self_referential True/False configs
    - run_meta_trie_sweep.py creates meta_convergence table in SQLite
    - run_meta_trie_sweep.py imports SweepRunner
  </acceptance_criteria>
  <done>Meta-trie sweep covers all D-14 through D-17 dimensions, convergence tracked per D-18, results in SQLite, comparison against all prior strategies printed</done>
</task>

</tasks>

<verification>
docker compose run --rm dev uv run pytest tests/test_threshold_policy.py -x -v -k "meta_trie"
docker compose run --rm dev uv run python scripts/sweep/run_meta_trie_sweep.py --help
</verification>

<success_criteria>
- MetaTriePolicy uses OctonionTrie (same class, per D-12)
- All 4 meta-trie dimensions independently tested
- Convergence tracking shows threshold change rate history
- Self-referential variant runs without divergence
- Meta-trie sweep uses same SweepRunner infrastructure per D-32
</success_criteria>

<output>
After completion, create `.planning/phases/T2-adaptive-thresholds/T2-07-SUMMARY.md`
</output>
