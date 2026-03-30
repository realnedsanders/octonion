---
phase: T2-adaptive-thresholds
plan: 08
type: execute
wave: 7
depends_on: ["T2-05", "T2-06", "T2-07"]
files_modified:
  - src/octonion/trie.py
  - scripts/sweep/run_hybrid_validation.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "HybridPolicy fully implemented combining top 2 strategy performers per D-09"
    - "Top-10 configs from each strategy validated with 10 seeds per D-33"
    - "Full-scale validation on complete datasets for top configs per D-22/D-40"
    - "Generalization gap measured between 10K subset and full training set per D-40"
  artifacts:
    - path: "src/octonion/trie.py"
      provides: "Full HybridPolicy implementation"
      contains: "class HybridPolicy"
    - path: "scripts/sweep/run_hybrid_validation.py"
      provides: "Hybrid sweep and multi-seed validation"
      contains: "def run_hybrid_sweep"
  key_links:
    - from: "src/octonion/trie.py HybridPolicy"
      to: "src/octonion/trie.py ThresholdPolicy"
      via: "Combines two ThresholdPolicy instances"
      pattern: "self\\.policy_a|self\\.policy_b"
---

<objective>
Implement HybridPolicy and run multi-seed validation on top configs from all strategies.

Purpose: Per D-09, hybrid strategy tested regardless of individual results. Per D-33, top configs validated with 10 seeds. Per D-22/D-40, full-scale validation on complete datasets for top configs. This plan produces the final experimental data needed for statistical analysis.

Output: Full HybridPolicy implementation, multi-seed validation results for all strategy top configs, generalization gap analysis.
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
@.planning/phases/T2-adaptive-thresholds/T2-05-SUMMARY.md
@.planning/phases/T2-adaptive-thresholds/T2-06-SUMMARY.md
@.planning/phases/T2-adaptive-thresholds/T2-07-SUMMARY.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: HybridPolicy implementation and hybrid sweep</name>
  <files>src/octonion/trie.py, scripts/sweep/run_hybrid_validation.py</files>
  <read_first>
    - src/octonion/trie.py (HybridPolicy stub, ThresholdPolicy ABC, all concrete policies)
    - scripts/sweep/sweep_runner.py (SweepRunner API)
    - .planning/phases/T2-adaptive-thresholds/T2-CONTEXT.md (D-09)
    - .planning/phases/T2-adaptive-thresholds/T2-06-SUMMARY.md (top performers from strategies 1-4)
    - .planning/phases/T2-adaptive-thresholds/T2-07-SUMMARY.md (top meta-trie configs)
  </read_first>
  <action>
**Part A: Replace HybridPolicy stub** in trie.py with full implementation per D-09.

```python
class HybridPolicy(ThresholdPolicy):
    """Combines two ThresholdPolicy instances per D-09.

    Combination modes:
    - "mean": average of both policies' thresholds
    - "min": minimum (more conservative / tighter)
    - "max": maximum (more permissive / looser)
    - "adaptive": use policy_a in early epochs, transition to policy_b
    """

    def __init__(
        self,
        policy_a: ThresholdPolicy,
        policy_b: ThresholdPolicy,
        combination: str = "mean",
        transition_inserts: int = 0,  # for "adaptive" mode: switch after N inserts
    ):
        self.policy_a = policy_a
        self.policy_b = policy_b
        self.combination = combination
        self.transition_inserts = transition_inserts
        self._total_inserts = 0

    def _combine(self, val_a: float, val_b: float) -> float:
        if self.combination == "mean":
            return (val_a + val_b) / 2.0
        elif self.combination == "min":
            return min(val_a, val_b)
        elif self.combination == "max":
            return max(val_a, val_b)
        elif self.combination == "adaptive":
            # Smooth transition from policy_a to policy_b
            if self.transition_inserts <= 0:
                return val_b
            alpha = min(1.0, self._total_inserts / self.transition_inserts)
            return (1 - alpha) * val_a + alpha * val_b
        return (val_a + val_b) / 2.0

    def get_assoc_threshold(self, node, depth):
        return self._combine(
            self.policy_a.get_assoc_threshold(node, depth),
            self.policy_b.get_assoc_threshold(node, depth),
        )

    def get_sim_threshold(self, node, depth):
        return self._combine(
            self.policy_a.get_sim_threshold(node, depth),
            self.policy_b.get_sim_threshold(node, depth),
        )

    def get_consolidation_params(self, node, depth):
        ms_a, mc_a = self.policy_a.get_consolidation_params(node, depth)
        ms_b, mc_b = self.policy_b.get_consolidation_params(node, depth)
        return self._combine(ms_a, ms_b), int(self._combine(mc_a, mc_b))

    def on_insert(self, node, x, assoc_norm):
        self._total_inserts += 1
        self.policy_a.on_insert(node, x, assoc_norm)
        self.policy_b.on_insert(node, x, assoc_norm)
```

Update tests/test_threshold_policy.py: Replace test_hybrid_stub_raises with functional tests:
- test_hybrid_mean_combines: mean of two GlobalPolicies with different thresholds
- test_hybrid_min_conservative: min combination is always tighter
- test_hybrid_adaptive_transition: starts with policy_a, transitions to policy_b

**Part B: Create scripts/sweep/run_hybrid_validation.py**

This script does two things:

**Phase 1 -- Hybrid sweep** (per D-09):
1. Query SQLite for top-2 performing strategy types (by mean accuracy across benchmarks)
2. Create HybridPolicy instances combining the top-2 with all 4 combination modes
3. For each combination mode, sweep with top configs from each base strategy
4. ~4 modes * top-5 pairs * 5 benchmarks = 100 configs

**Phase 2 -- Multi-seed validation** (per D-33):
1. Identify top-10 configs OVERALL (across all strategies including hybrid)
2. For each top config, run with 10 seeds: [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]
3. Total: 10 configs * 10 seeds * 5 benchmarks = 500 runs
4. Store all results in SQLite with different seed values

**Phase 3 -- Full-scale validation** (per D-22, D-40):
1. Top-5 overall configs re-run on FULL datasets (not 10K subsets)
2. Uses full feature files from cache_features.py
3. Compare full-scale accuracy vs 10K subset accuracy (generalization gap per D-40)

Per D-37, record structural variance across seeds: node count, max depth, branching factor, rumination rate as mean +/- std.

**CLI interface**:
```
python scripts/sweep/run_hybrid_validation.py --phase 1 --features-dir results/T2/features --db results/T2/sweep.db --workers 24
python scripts/sweep/run_hybrid_validation.py --phase 2  # multi-seed
python scripts/sweep/run_hybrid_validation.py --phase 3  # full-scale
python scripts/sweep/run_hybrid_validation.py --phase all
```
  </action>
  <verify>
    <automated>docker compose run --rm dev uv run pytest tests/test_threshold_policy.py -x -v -k "hybrid"</automated>
  </verify>
  <acceptance_criteria>
    - trie.py HybridPolicy contains `self.policy_a` and `self.policy_b`
    - trie.py HybridPolicy contains combination modes: mean, min, max, adaptive
    - trie.py HybridPolicy.on_insert delegates to both policies
    - scripts/sweep/run_hybrid_validation.py exists
    - run_hybrid_validation.py contains `--phase` with choices 1, 2, 3, all
    - run_hybrid_validation.py uses 10 seeds per D-33
    - run_hybrid_validation.py queries SQLite for top strategies from prior sweeps
    - run_hybrid_validation.py computes generalization gap per D-40
    - test_threshold_policy.py contains `def test_hybrid_mean_combines`
    - Hybrid tests pass
  </acceptance_criteria>
  <done>HybridPolicy fully implemented with 4 combination modes, hybrid sweep completed, multi-seed validation on top-10 configs with 10 seeds each, full-scale generalization gap measured</done>
</task>

</tasks>

<verification>
docker compose run --rm dev uv run pytest tests/test_threshold_policy.py -x -v
docker compose run --rm dev uv run python scripts/sweep/run_hybrid_validation.py --help
</verification>

<success_criteria>
- HybridPolicy combines two arbitrary ThresholdPolicy instances
- Top-10 configs validated with 10 seeds (500 runs)
- Full-scale validation shows generalization gap for top-5 configs
- Structural variance (node count, depth, branching) reported across seeds
</success_criteria>

<output>
After completion, create `.planning/phases/T2-adaptive-thresholds/T2-08-SUMMARY.md`
</output>
