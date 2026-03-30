---
phase: T2-adaptive-thresholds
plan: 11
type: execute
wave: 9
depends_on: ["T2-09", "T2-10"]
files_modified:
  - scripts/sweep/sweep_diagnostics.py
  - src/octonion/trie.py
autonomous: false
requirements: []
must_haves:
  truths:
    - "Per-node associator norm distributions visualized as histograms per D-54"
    - "Depth profiles and routing statistics produced per D-54"
    - "trie.py defaults updated if results justify per D-11"
    - "Human verifies final results, threshold recommendation, and theory"
  artifacts:
    - path: "scripts/sweep/sweep_diagnostics.py"
      provides: "Diagnostic visualization script"
      contains: "def generate_diagnostics"
    - path: "results/T2/diagnostics/"
      provides: "Diagnostic PNG visualizations"
  key_links:
    - from: "scripts/sweep/sweep_diagnostics.py"
      to: "src/octonion/trie.py"
      via: "OctonionTrie with best policy for diagnostic data collection"
      pattern: "OctonionTrie"
---

<objective>
Produce diagnostic visualizations and update trie defaults based on results.

Purpose: Per D-54, produce full diagnostic output. Per D-11, update trie.py defaults if results justify. Final human verification of all T2 results.

Output: Diagnostic PNGs, updated trie.py defaults (if warranted), human approval of phase results.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/phases/T2-adaptive-thresholds/T2-CONTEXT.md
@.planning/phases/T2-adaptive-thresholds/T2-09-SUMMARY.md
@.planning/phases/T2-adaptive-thresholds/T2-10-SUMMARY.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Diagnostic visualizations and trie.py defaults update</name>
  <files>scripts/sweep/sweep_diagnostics.py, src/octonion/trie.py</files>
  <read_first>
    - src/octonion/trie.py (current defaults: assoc_threshold=0.3, sim_threshold=0.1, GlobalPolicy)
    - scripts/sweep/sweep_analysis.py (statistical_report.json location, recommendation)
    - .planning/phases/T2-adaptive-thresholds/T2-CONTEXT.md (D-11, D-54)
    - .planning/phases/T2-adaptive-thresholds/T2-09-SUMMARY.md (statistical analysis results, recommendation)
  </read_first>
  <action>
**Part A: Create scripts/sweep/sweep_diagnostics.py** (per D-54).

Diagnostic visualizations for thesis and future phases:

1. `plot_per_node_assoc_distributions(trie, save_path)`:
   - Walk trie, collect associator norm history from node._policy_state
   - Plot histogram per depth level (stacked or faceted)
   - Show how associator norm distributions vary by depth

2. `plot_depth_profile(trie, save_path)`:
   - Node count vs depth
   - Mean branching factor vs depth
   - Mean associator norm vs depth
   - 3-panel figure

3. `plot_routing_statistics(trie, save_path)`:
   - Subalgebra slot utilization (which of 7 slots are used most)
   - Category purity per depth level
   - Buffer occupancy distribution

4. `plot_category_routing_paths(trie, test_samples, test_labels, save_path)`:
   - For each category, show the distribution of routing paths
   - Which subalgebras are favored by which categories
   - Heatmap: category x subalgebra

5. `generate_diagnostics(features_dir, db_path, output_dir)`:
   - Load best config from statistical_report.json
   - Build trie with that config on each benchmark
   - Generate all diagnostic plots
   - Save to output_dir/diagnostics/

CLI:
```
python scripts/sweep/sweep_diagnostics.py --features-dir results/T2/features --db results/T2/sweep.db --output-dir results/T2/diagnostics
```

**Part B: Update trie.py defaults** (per D-11).

Read the recommendation from results/T2/analysis/statistical_report.json. If the recommended config differs from current defaults:

1. If a better global threshold is found: Update `OctonionTrie.__init__` default `associator_threshold` from 0.3 to the recommended value
2. If an adaptive policy is significantly better: Update `OctonionTrie.__init__` default `policy` parameter to instantiate the recommended policy type with its best hyperparameters
3. Add a comment in trie.py documenting: "Default threshold/policy updated in Phase T2 based on cross-benchmark analysis. See results/T2/analysis/statistical_report.json"
4. If no change is justified (global 0.3 remains best): Document that in a comment

**IMPORTANT**: This update is conditional on T2-09's recommendation. If the recommendation is "keep GlobalPolicy(0.3)", make no code change, only add a validating comment.

Update existing tests if defaults change -- any test that constructs OctonionTrie() without explicit parameters should still pass.
  </action>
  <verify>
    <automated>docker compose run --rm dev uv run pytest tests/test_trie.py tests/test_threshold_policy.py -x</automated>
  </verify>
  <acceptance_criteria>
    - scripts/sweep/sweep_diagnostics.py exists
    - sweep_diagnostics.py contains `def plot_per_node_assoc_distributions`
    - sweep_diagnostics.py contains `def plot_depth_profile`
    - sweep_diagnostics.py contains `def plot_routing_statistics`
    - sweep_diagnostics.py contains `def generate_diagnostics`
    - sweep_diagnostics.py contains `matplotlib.use("Agg")`
    - trie.py contains comment about Phase T2 threshold analysis
    - All existing trie tests still pass after any defaults update
  </acceptance_criteria>
  <done>Diagnostic visualizations generated for thesis, trie.py defaults reviewed and updated (or confirmed) based on T2 analysis, all tests pass</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 2: Human verification of T2 results</name>
  <files>results/T2/analysis/statistical_report.json</files>
  <action>
Present Phase T2 results for human review. The following artifacts were produced:
1. ThresholdPolicy abstraction with 8 strategy implementations in trie.py
2. Parallel sweep framework with SQLite storage in scripts/sweep/
3. Global sensitivity sweep across 5 benchmarks
4. Adaptive strategies 1-4 with hyperparameter tuning
5. Meta-trie optimizer with self-referential variant
6. Hybrid strategy combining top 2 performers
7. Multi-seed validation on top configs
8. Statistical analysis with all required tests
9. Auto-recommendation in statistical_report.json
10. Theory section in oct-trie.tex
11. Diagnostic visualizations in results/T2/diagnostics/
  </action>
  <verify>
    <automated>docker compose run --rm dev uv run pytest tests/test_trie.py tests/test_threshold_policy.py tests/test_sweep_runner.py -x</automated>
  </verify>
  <acceptance_criteria>
    - results/T2/analysis/statistical_report.json exists and contains recommendation
    - results/T2/diagnostics/ directory contains PNG files
    - docs/thesis/oct-trie.tex contains adaptive thresholds section
    - All test suites pass
  </acceptance_criteria>
  <what-built>
Complete Phase T2 adaptive threshold investigation:
1. ThresholdPolicy abstraction with 8 strategy implementations
2. Parallel sweep framework with SQLite storage
3. Global sensitivity sweep (4D grid, all 5 benchmarks)
4. Adaptive strategies 1-4 (EMA, mean+std, depth, purity) with hyperparameter tuning
5. Meta-trie optimizer (self-referential trie optimization)
6. Hybrid strategy combining top 2 performers
7. Multi-seed validation (10 seeds) on top configs
8. Full statistical analysis (Wilcoxon, t-test, bootstrap CI, Friedman, Cohen's d)
9. Auto-recommendation with Pareto ranking
10. Theory section in oct-trie.tex (associator distribution, Fano geometry, G2, proofs)
11. Diagnostic visualizations
  </what-built>
  <how-to-verify>
1. Review results/T2/analysis/statistical_report.json -- check recommendation and statistical significance
2. Review results/T2/plots/ -- examine heatmaps and Pareto frontiers
3. Review results/T2/diagnostics/ -- examine per-node associator distributions and routing statistics
4. Review results/T2/theory/monte_carlo_results.json -- check Egan's result validation
5. Review docs/thesis/oct-trie.tex new section -- evaluate proof quality and narrative
6. Run: docker compose run --rm dev uv run pytest tests/test_trie.py tests/test_threshold_policy.py tests/test_sweep_runner.py -x
7. Verify that trie.py defaults are appropriate based on the recommendation
8. Check that the self-organization narrative (D-53) is compelling
  </how-to-verify>
  <resume-signal>Type "approved" to complete T2, or describe any issues to address</resume-signal>
  <done>Human has reviewed and approved all T2 results including statistical analysis, theory, and recommendation</done>
</task>

</tasks>

<verification>
docker compose run --rm dev uv run pytest tests/test_trie.py tests/test_threshold_policy.py -x
ls results/T2/diagnostics/ results/T2/analysis/ results/T2/theory/
</verification>

<success_criteria>
- Diagnostic visualizations produced for all 5 benchmarks
- trie.py defaults reviewed/updated based on evidence
- Human approves final results, statistical analysis, theory, and recommendation
- All tests pass
</success_criteria>

<output>
After completion, create `.planning/phases/T2-adaptive-thresholds/T2-11-SUMMARY.md`
</output>
