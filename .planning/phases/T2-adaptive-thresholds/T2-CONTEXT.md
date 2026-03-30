# Phase T2: Adaptive Thresholds - Context

**Gathered:** 2026-03-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Determine whether the associator threshold should be global, per-node, context-specific, meta-trie-optimized, or theoretically justified as global. Co-adapt rumination and consolidation thresholds alongside the associator threshold. Build a reusable parallel sweep framework. Produce theoretical analysis connecting thresholds to octonionic algebra. Introduce a meta-trie optimizer that uses a second OctonionTrie to adapt classifier thresholds via environment feedback.

</domain>

<decisions>
## Implementation Decisions

### Adaptation Strategies

- **D-01:** Test 6 strategies against global baseline, in order: (1) per-node EMA of associator norms, (2) per-node running mean+std, (3) depth-dependent with both directions (tighter and looser with depth, sweep decay factors 0.7-1.3), (4) algebraic purity — test associator norm variance and routing key similarity variance as independent signals, (5) meta-trie optimizer, (6) hybrid combining top 2 performers
- **D-02:** All strategies must be unsupervised — threshold adaptation uses only algebraic signals (associator norms, routing structure), never category labels. Labels used only for accuracy evaluation.
- **D-03:** Co-adapt rumination threshold (similarity_threshold) and consolidation thresholds (min_share, min_count) alongside associator threshold. Not isolated experiments.
- **D-04:** Adaptive thresholds apply during both training (insert) and inference (query). Learned per-node thresholds become part of the trie's structure.
- **D-05:** Include noise revisit from T1 deferral. Test noise as interaction effect with adaptive thresholds. Noise becomes 4th sweep dimension.
- **D-06:** Sweep training epochs (1, 3, 5) for each strategy to test epoch-threshold interaction.
- **D-07:** Run all experiments on all T1 benchmarks: MNIST, Fashion-MNIST, CIFAR-10 (ResNet-8), Text 4-class, Text 20-class.
- **D-08:** Test simple strategies first, meta-trie last. Build understanding progressively. Meta-trie design benefits from knowing what simpler strategies reveal.
- **D-09:** Hybrid strategy tested regardless of individual strategy results (always test, not gated on individual performance).

### ThresholdPolicy Abstraction

- **D-10:** Add pluggable ThresholdPolicy to OctonionTrie in trie.py. All strategies (GlobalPolicy, PerNodeEMAPolicy, DepthPolicy, AlgebraicPurityPolicy, MetaTriePolicy, HybridPolicy) are implementations. Classifier trie doesn't know how thresholds are set.
- **D-11:** Update trie.py defaults if results justify a different global threshold or default ThresholdPolicy.

### Meta-Trie Optimizer

- **D-12:** A second OctonionTrie acts as optimizer for the classifier trie. Uses the same OctonionTrie class (not a subclass). Strongest thesis statement: the trie optimizes itself using the same algebra.
- **D-13:** Meta-trie categories are discretized threshold actions (e.g., "tighten 10%", "keep", "loosen 10%", "loosen 20%").
- **D-14:** Test two meta-trie input encodings independently: (1) direct 8D signal vector (assoc_norm_mean, assoc_norm_std, branching_factor/7, insert_rate, rumination_rate, depth/max_depth, buffer_consistency, consolidation_rate), (2) classifier node's routing key or content octonion as algebraic input. User suspects the algebraic encoding will be important.
- **D-15:** Test two meta-trie feedback signals independently: (1) classifier trie stability (low rumination rejections, balanced branching, consistent associator norms — fully unsupervised), (2) classifier accuracy on held-out set (supervised, for comparison).
- **D-16:** Test multiple update frequencies: per-100-inserts, per-1000-inserts, per-epoch. Sweep to find right granularity.
- **D-17:** Test both fixed meta-trie thresholds (simple bootstrap) and self-referential (meta-trie adapts its own thresholds) independently.
- **D-18:** Meta-trie convergence criterion: threshold change rate < 1% between updates. Track convergence curve.
- **D-19:** MetaTriePolicy is an implementation of ThresholdPolicy — integrates into the same abstraction as simple strategies.

### Sensitivity Sweep Design

- **D-20:** 4D sweep grid: associator threshold (0.001-2.0, wide log-spaced) x similarity threshold (range TBD) x consolidation (5 configs) x noise (0.0, 0.01, 0.05, 0.1). ~2000 configs per benchmark.
- **D-21:** 2D heatmaps for assoc x similarity grid, line plots for 1D sweeps, Pareto frontier (accuracy vs node count). Matplotlib, saved as PNG.
- **D-22:** Reduced-first approach: 10K subset for initial 4D sweep, full-scale on top 10 configurations.
- **D-23:** Pre-compute and cache encoder features. Workers load 8D features and only run the trie. No GPU needed for sweep.
- **D-24:** 24-worker ProcessPoolExecutor for parallel sweep. Each config is independent.
- **D-25:** SQLite database for all sweep results. Epoch-by-epoch tracking (accuracy + structure after each epoch).
- **D-26:** Reusable parallel sweep framework — generic runner with param grid, worker function, SQLite output. Reusable by T4, T6.
- **D-27:** Run all configs to completion (no early stopping). Even degenerate configs run all epochs.
- **D-28:** tqdm progress bar for live progress.
- **D-29:** Sweep adaptive strategy hyperparameters too (EMA alpha, depth decay factor, purity sensitivity, etc.) using same infrastructure. Fair comparison requires tuned adaptive vs tuned global.
- **D-30:** Auto-recommend best configuration with justification. Script identifies top configs by Pareto rank and cross-benchmark consistency.
- **D-31:** Fixed seed (42) for sweep exploration. Multi-seed (10 seeds) for top-config validation.
- **D-32:** Meta-trie experiments use same sweep infrastructure (MetaTriePolicy plugs in like any other policy).

### Statistical Rigor

- **D-33:** 10 random seeds for top-config validation runs.
- **D-34:** Both paired Wilcoxon signed-rank test AND paired t-test, plus bootstrap 95% confidence intervals for accuracy differences.
- **D-35:** Cohen's d effect sizes alongside p-values.
- **D-36:** Friedman test + rank analysis for cross-benchmark consistency.
- **D-37:** Full structural variance across seeds: mean +/- std for node count, max depth, branching factor, rumination rate.
- **D-38:** Significance threshold: Claude's discretion (likely p < 0.05 with Bonferroni correction, but field-appropriate).
- **D-39:** No fixed practical significance threshold — report all results. Context determines meaningfulness.
- **D-40:** Generalization gap: compare 10K subset vs full training set accuracy for top configs.
- **D-41:** Automated analysis script (not notebook). Reads SQLite, runs all statistical tests, produces tables and plots. JSON + PNG output.

### Theory Approach

- **D-42:** Pursue both associator norm distribution analysis (analytic characterization for unit octonions) AND Fano plane geometry argument. Verify predictions empirically against measured distributions from sweep diagnostics.
- **D-43:** Full proof attempt. If proof succeeds, strong thesis contribution. If not, document where it breaks and state as conjecture with evidence.
- **D-44:** Develop novel G2 results if existing literature doesn't directly provide needed bounds. Potentially significant math contribution.
- **D-45:** Characterize both: when global threshold suffices AND when/why adaptive is needed. The boundary between regimes is interesting regardless of which wins.
- **D-46:** Monte Carlo sampling on unit 7-sphere to validate analytical bounds. Sample random unit octonion triples, compute associator norms, fit distribution.
- **D-47:** Investigate closed-form relationship between optimal threshold and octonion algebra constants (structure constants, Fano plane geometry).
- **D-48:** Formal convergence analysis for meta-trie feedback loop (fixed point analysis). Characterize conditions for guaranteed convergence.
- **D-49:** Formal stability-plasticity connection: frame threshold as stability-plasticity tradeoff (tight = stable, loose = plastic), derive conditions for balance.
- **D-50:** Formal complexity analysis: time/space per insert and query for each ThresholdPolicy.
- **D-51:** Cayley-Dickson doubling connection: Claude's discretion on whether this adds insight.
- **D-52:** Theory goes in oct-trie.tex as a new section, not a separate document.
- **D-53:** Self-organization narrative: frame as "the trie discovers its own operating parameters," not purely empirical characterization.

### Diagnostics

- **D-54:** Full diagnostic output: per-node associator norm distributions (histograms), depth profiles, routing statistics. Produces visualizations supporting thesis and informing future phases.

### Claude's Discretion

- Significance level choice (likely Bonferroni-corrected p < 0.05)
- Cayley-Dickson connection depth
- Specific similarity_threshold sweep range for 2D grid
- 5 representative consolidation configs (min_share x min_count combinations)
- Mathematical references beyond Baez (2002) and Harvey (1990)
- Specific EMA decay rates, depth decay factors, and purity sensitivity values to sweep

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Trie Implementation
- `src/octonion/trie.py` — OctonionTrie class, TrieNode, subalgebra_activation. Current threshold at line 98 (default 0.3), used at lines 145, 262, 269. Rumination at lines 171-179. Consolidation at lines 390-418.
- `tests/test_trie.py` — Existing trie tests (18 tests), including threshold variation tests at lines 189, 205

### Benchmark Scripts (for feature caching and evaluation)
- `scripts/trie_benchmark_utils.py` — Shared benchmark utilities with run_trie_classifier, run_sklearn_baselines
- `scripts/run_trie_mnist.py` — MNIST benchmark reference
- `scripts/run_trie_fashion_mnist.py` — Fashion-MNIST benchmark
- `scripts/run_trie_cifar10.py` — CIFAR-10 benchmark
- `scripts/run_trie_benchmarks_parallel.py` — Existing parallel benchmark runner (process-level parallelism reference)

### Algebra Foundations
- `src/octonion/_fano.py` — Fano plane triples (subalgebra structure)
- `src/octonion/_multiplication.py` — octonion_mul with structure constants
- `src/octonion/_octonion.py` — Octonion class, associator function

### Thesis
- `docs/thesis/oct-trie.tex` — Trie thesis, target for new theory section

### Prior Phase Context
- `.planning/phases/T1-benchmark-generalization/T1-CONTEXT.md` — T1 decisions (benchmarks, encoder strategy, analysis depth)
- `.planning/phases/T1-benchmark-generalization/T1-05-SUMMARY.md` — T1 results (accuracy table, noise investigation deferral)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `OctonionTrie` class: current global threshold, all node state (depth, insert_count, category_counts, buffer) available for adaptive logic
- `subalgebra_activation()`: computes 7 subalgebra activation norms, used in routing
- `trie_benchmark_utils.py`: run_trie_classifier, run_sklearn_baselines, metrics computation, plotting — reuse for sweep evaluation
- `run_trie_benchmarks_parallel.py`: existing process-level parallelism pattern for reference
- `associator()` function in `_octonion.py`: computes [a,b,c] = (ab)c - a(bc), core algebraic signal

### Established Patterns
- Trie uses float64 by default for algebraic precision
- Unit normalization on all inputs (line 237-238)
- Buffer stores last 30 samples per node (deque maxlen=30)
- Consolidation uses fixed 5%/3-count rule (lines 400-404)
- Benchmark scripts use Agg matplotlib backend for headless rendering
- Dataset caching via Docker volume at `/workspace/.data`

### Integration Points
- ThresholdPolicy replaces `self.assoc_threshold` (single float) with policy object
- `_find_best_child` line 145 and `insert` lines 262, 269 are the comparison points
- Rumination `_ruminate` line 176-179 uses `self.sim_threshold` — needs policy integration
- Consolidation `_consolidate_node` lines 400-404 uses hardcoded 0.05 and 3 — needs policy integration
- Meta-trie receives signals from classifier nodes — needs signal extraction hooks
- SQLite replaces JSON output pattern from T1 benchmarks

</code_context>

<specifics>
## Specific Ideas

- **Meta-trie as self-organizing optimizer**: User specifically wants the trie to optimize itself using a second instance of the same OctonionTrie class. Not a separate architecture — same algebra, same self-organization mechanism. "Have the trie for the classifier, and a separate trie that optimizes the classifier based on signals from the classifier."
- **Unsupervised emphasis**: "The primary goal of the trie architecture is to resolve the stability-plasticity dilemma unsupervised. Environment feedback is probably the right unsupervised signal."
- **Algebraic purity over label purity**: Reframe "purity" from category-label-based to associator-norm-distribution-based. The associator IS the environment feedback.
- **Self-referential variant**: User specifically wants to test the meta-trie adapting its own thresholds. Separate from the fixed-meta-trie variant.
- **Node routing key / content as meta-trie input**: User suspects this algebraic encoding "is going to be important" for the meta-trie, though uncertain. Test independently against the 8D signal vector encoding.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: T2-adaptive-thresholds*
*Context gathered: 2026-03-29*
