# Phase T2: Adaptive Thresholds - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-29
**Phase:** T2-adaptive-thresholds
**Areas discussed:** Adaptation strategies, Sensitivity sweep design, Statistical rigor, Theory approach

---

## Adaptation Strategies

### Adaptive threshold rules to test

| Option | Description | Selected |
|--------|-------------|----------|
| Per-node running stats | Each node tracks running mean/std of associator norms | ✓ |
| Depth-dependent | Threshold varies with depth (tighter or looser) | ✓ |
| Category purity-based | Mixed-category nodes get tighter thresholds | ✓ (reframed) |
| Hybrid (top 2 combined) | Combine best-performing strategies | ✓ |

**User's choice:** All four, but category purity was reframed to "algebraic purity" — use associator norm variance and routing key similarity variance instead of category labels.
**Notes:** User emphasized unsupervised signals: "The primary goal of the trie architecture is to resolve the stability-plasticity dilemma unsupervised. Environment feedback is probably the right unsupervised signal."

### Category purity handling

| Option | Description | Selected |
|--------|-------------|----------|
| Test as adaptive rule | Use purity to drive threshold adaptation | |
| Diagnostic only | Measure purity to explain WHY thresholds work | |
| Skip entirely | Focus on label-free strategies | |

**User's choice:** "Test as an adaptive rule, but how do we enable self-supervision or no supervision?" — led to reframing purity as algebraic (unsupervised) rather than label-based.

### Supervised baseline for purity

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, as comparison baseline | Test both label-based and algebraic purity | |
| No, algebraic only | Stay fully unsupervised for threshold adaptation | ✓ |

**User's choice:** Algebraic only. Labels only for accuracy measurement.

### Noise revisit from T1

| Option | Description | Selected |
|--------|-------------|----------|
| Include noise revisit | Test noise + adaptive thresholds as interaction effect | ✓ |
| Keep separate | Focus purely on threshold adaptation | |

### Per-node tracking approach

| Option | Description | Selected |
|--------|-------------|----------|
| EMA of associator norms | Exponential moving average, one float per node | ✓ (test independently) |
| Running mean + std | Track both mean and standard deviation | ✓ (test independently) |
| You decide | Claude's discretion | |

**User's choice:** Test each approach independently.

### Rumination co-adaptation

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed rumination | Keep rumination logic unchanged | |
| Co-adapt rumination | Also make rumination threshold adaptive | ✓ |

### Consolidation co-adaptation

| Option | Description | Selected |
|--------|-------------|----------|
| Keep fixed | Current 5%/3-count rule unchanged | |
| Co-adapt consolidation | Make consolidation thresholds adaptive | ✓ |

### Depth direction

| Option | Description | Selected |
|--------|-------------|----------|
| Tighter with depth | threshold = base * decay^depth | |
| Looser with depth | threshold = base * growth^depth | |
| Test both directions | Sweep decay factors 0.7-1.3 | ✓ |

### Epoch sweep

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, sweep epochs | Test 1, 3, 5 epochs for each strategy | ✓ |
| No, fix at 3 epochs | Keep epochs constant | |

### Benchmarks

| Option | Description | Selected |
|--------|-------------|----------|
| All T1 benchmarks | MNIST, Fashion-MNIST, CIFAR-10, Text 4-class, Text 20-class | ✓ |
| Three benchmarks | MNIST + Fashion-MNIST + Text 4-class | |
| Two benchmarks | MNIST + Fashion-MNIST only | |

### Code design

| Option | Description | Selected |
|--------|-------------|----------|
| ThresholdPolicy abstraction | Pluggable threshold policy in trie.py | ✓ |
| Scripts only | Keep trie.py simple, experiment in scripts | |

### Algebraic purity signal

| Option | Description | Selected |
|--------|-------------|----------|
| Associator norm variance only | Track variance of associator norms | ✓ (test independently) |
| Both associator + routing key similarity | Track both signals | ✓ (test independently) |

**User's choice:** "Try each independently."

### Diagnostics

| Option | Description | Selected |
|--------|-------------|----------|
| Full diagnostic output | Per-node associator norm distributions, depth profiles, routing stats | ✓ |
| Minimal diagnostics | Just accuracy numbers and trie stats | |

### Meta-trie as optimizer (user-initiated)

**User proposed:** "Have the trie for the classifier, and a separate trie that optimizes the classifier based on signals from the classifier." Folded into T2 scope because it's necessary for co-adapting multiple thresholds.

### Meta-trie self-observation strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Same OctonionTrie class | Reuse exact same trie implementation | ✓ |
| Specialized MetaTrie subclass | Subclass with additional hooks | |

### Meta-trie feedback signal

| Option | Description | Selected |
|--------|-------------|----------|
| Classifier trie stability | Low rumination rejections, balanced branching (unsupervised) | ✓ (test independently) |
| Classifier accuracy on held-out set | Periodically evaluate accuracy (supervised) | ✓ (test independently) |

### Meta-trie update frequency

| Option | Description | Selected |
|--------|-------------|----------|
| Test multiple frequencies | per-100-inserts, per-1000-inserts, per-epoch | ✓ |

### Meta-trie self-reference

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed meta-trie thresholds | Simple bootstrap, one level | ✓ (test independently) |
| Self-referential | Meta-trie adapts its own thresholds | ✓ (test independently) |

### Meta-trie categories

| Option | Description | Selected |
|--------|-------------|----------|
| Discretized threshold actions | "tighten 10%", "keep", "loosen 10%", "loosen 20%" | ✓ |
| Classifier node types | "saturated leaf", "balanced internal", etc. | |

### Meta-trie input encoding

| Option | Description | Selected |
|--------|-------------|----------|
| Direct 8D signal vector | Pack 8 classifier signals into octonion | ✓ (test independently) |
| Algebraic encoding | Use classifier node's routing key or content | ✓ (test independently) |

**Notes:** User: "I think the node's routing key or content octonion is going to be important to the meta-trie, but I'm not certain."

### Test order

| Option | Description | Selected |
|--------|-------------|----------|
| Simple first, meta-trie last | Build understanding progressively | ✓ |
| Meta-trie first | Jump to most promising approach | |
| All in parallel | Fastest wall-clock time | |

### ThresholdPolicy scope

| Option | Description | Selected |
|--------|-------------|----------|
| ThresholdPolicy covers all | MetaTriePolicy is a ThresholdPolicy implementation | ✓ |
| Meta-trie as separate orchestration | Different architectural layer | |

---

## Sensitivity Sweep Design

### Sweep range

| Option | Description | Selected |
|--------|-------------|----------|
| 0.01 to 1.0, log-spaced | ~20 points in operational range | |
| 0.05 to 0.5, linear-spaced | ~10 points around default | |
| 0.001 to 2.0, wide log-spaced | ~25 points including extremes | ✓ |

### Multi-threshold sweep

| Option | Description | Selected |
|--------|-------------|----------|
| 2D sweep: assoc + similarity | Grid sweep over both thresholds | ✓ (base) |
| Sequential 1D sweeps | One at a time | |
| Assoc threshold only | Single dimension | |

### Sweep metrics

| Option | Description | Selected |
|--------|-------------|----------|
| Trie structure | nodes, depth, branching | ✓ |
| Training time | Wall-clock per config | ✓ |
| Per-class accuracy breakdown | Which classes benefit/suffer | ✓ |
| Rumination/consolidation rates | Reflects threshold behavior directly | ✓ |

### Visualization

| Option | Description | Selected |
|--------|-------------|----------|
| Heatmaps + line plots | 2D heatmaps, 1D line plots, matplotlib PNG | ✓ |
| Interactive plots | Plotly for exploration | |

### Sweep scale

| Option | Description | Selected |
|--------|-------------|----------|
| Reduced first, full on best | 10K subset for grid, full on top 10 | ✓ |
| Full scale for all | Every config on full training set | |
| Reduced only | 10K subset for all | |

### Consolidation sweep

| Option | Description | Selected |
|--------|-------------|----------|
| 3D grid (add consolidation) | Add consolidation as separate dimension | |
| Sequential after 2D | Sweep consolidation on best 2D configs | |
| Full simultaneous 4D | Add noise too — all parallel with 24 workers | ✓ |

**Notes:** User: "Should be able to do these all in parallel since they're independent experiments (perhaps worker pool approach with 24 workers), right?"

### Feature caching

| Option | Description | Selected |
|--------|-------------|----------|
| Pre-compute and cache | Run encoder once, save 8D features | ✓ |
| Each worker runs full pipeline | Worker runs encoder + trie | |

### Output format

| Option | Description | Selected |
|--------|-------------|----------|
| JSON per config + summary CSV | Consistent with T1 | |
| SQLite database | All results in one DB, easy SQL queries | ✓ |

### Reusability

| Option | Description | Selected |
|--------|-------------|----------|
| Reusable sweep framework | Generic runner for T4/T6 reuse | ✓ |
| Purpose-built for T2 | Simpler, faster to implement | |

### Consolidation grid size

| Option | Description | Selected |
|--------|-------------|----------|
| 5 consolidation configs | 10x10x5 = 500 per benchmark (before noise) | ✓ |
| 3 consolidation configs | Coarser but faster | |

### Subset size

| Option | Description | Selected |
|--------|-------------|----------|
| 10K samples | Large enough for threshold effects | ✓ |
| 5K samples | Faster iteration | |

### Epoch tracking

| Option | Description | Selected |
|--------|-------------|----------|
| Final metrics only | After all epochs | |
| Epoch-by-epoch tracking | Accuracy + structure after each epoch | ✓ |

### Noise dimension

| Option | Description | Selected |
|--------|-------------|----------|
| Separate follow-up | Noise on top 10 configs only | |
| Add noise as 4th dimension | 10x10x5x4 = 2000 per benchmark | ✓ |

### Seed policy

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed seed for sweep, multi-seed for top | Seed 42 for grid, 10 seeds for validation | ✓ |
| Multi-seed in the grid | 3 seeds per config | |

### Adaptive hyperparameter sweep

| Option | Description | Selected |
|--------|-------------|----------|
| Sweep adaptive hyperparams | Fair comparison: tuned adaptive vs tuned global | ✓ |
| Use reasonable defaults | Less fair but faster | |

### Early stopping

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, early stopping | Kill clearly bad configs | |
| Run all to completion | No risk of missing late-blooming configs | ✓ |

### Progress reporting

| Option | Description | Selected |
|--------|-------------|----------|
| tqdm progress bar | Simple, low overhead | ✓ |
| Rich dashboard | Real-time table with best-so-far | |

### Pareto analysis

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, Pareto frontier | Accuracy vs node count trade-off | ✓ |
| Rank by accuracy only | Simple ranking | |

### Meta-trie sweep infrastructure

| Option | Description | Selected |
|--------|-------------|----------|
| Same infrastructure | MetaTriePolicy plugs into sweep framework | ✓ |
| Separate setup | Dedicated meta-trie runner | |

### Auto-recommendation

| Option | Description | Selected |
|--------|-------------|----------|
| Auto-recommend with justification | Script identifies top configs, produces recommendation | ✓ |
| Data only, human decides | Tables and plots only | |

---

## Statistical Rigor

### Seed count

| Option | Description | Selected |
|--------|-------------|----------|
| 10 seeds | Standard for paired tests with reasonable power | ✓ |
| 5 seeds | Lower power minimum | |
| 20 seeds | High power, detect 1pp differences | |

### Statistical tests

| Option | Description | Selected |
|--------|-------------|----------|
| Paired Wilcoxon signed-rank | Non-parametric, paired by seed | ✓ (included) |
| Paired t-test | Parametric | ✓ (included) |
| Both + bootstrap CI | Belt-and-suspenders | ✓ |

### Significance level

| Option | Description | Selected |
|--------|-------------|----------|
| p < 0.05 with Bonferroni | Conservative, defensible | |
| p < 0.01 uncorrected | Stricter per-test | |
| You decide | Claude's discretion | ✓ |

### Effect sizes

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, Cohen's d | Contextualizes p-value | ✓ |
| P-values + CIs sufficient | Bootstrap CIs show magnitude | |

### Cross-benchmark consistency

| Option | Description | Selected |
|--------|-------------|----------|
| Friedman test + rank analysis | Formal cross-benchmark test | ✓ |
| Per-benchmark only | Informal patterns | |

### Analysis infrastructure

| Option | Description | Selected |
|--------|-------------|----------|
| Automated analysis script | Reads SQLite, runs tests, produces output | ✓ |
| Jupyter notebook | Interactive exploration | |
| Both | Standard + exploratory | |

### Practical significance threshold

| Option | Description | Selected |
|--------|-------------|----------|
| 1pp minimum | Must beat global by 1pp | |
| 0.5pp minimum | Lower bar for novel mechanism | |
| No fixed threshold | Report all, context determines meaning | ✓ |

### Meta-trie convergence

| Option | Description | Selected |
|--------|-------------|----------|
| Threshold change rate < epsilon | Converged when outputs change < 1% | ✓ |
| Classifier accuracy plateau | Simpler but conflated | |

### Structural variance

| Option | Description | Selected |
|--------|-------------|----------|
| Full structural variance | Mean +/- std for all structural metrics across seeds | ✓ |
| Accuracy variance only | Focus on accuracy | |

### Generalization gap

| Option | Description | Selected |
|--------|-------------|----------|
| Compare reduced vs full | 10K vs full training set for top configs | ✓ |
| Separate concern | Belongs in T4 | |

---

## Theory Approach

### Theoretical direction

| Option | Description | Selected |
|--------|-------------|----------|
| Associator norm distribution | Analytic characterization for unit octonions | ✓ (included) |
| Fano plane geometry | Combinatorial/algebraic argument | ✓ (included) |
| Both + empirical verification | Verify predictions against measured distributions | ✓ |
| Empirical only | No formal theory attempt | |

### Theory rigor

| Option | Description | Selected |
|--------|-------------|----------|
| Conjecture + strong evidence | Precise conjecture with support | |
| Full proof attempt | Complete proof, document breakdowns if needed | ✓ |

### Theory location

| Option | Description | Selected |
|--------|-------------|----------|
| In oct-trie.tex | New "Theoretical Analysis" section | ✓ |
| Separate document | Standalone note | |
| Both | Key results in thesis, details separate | |

### G2 depth

| Option | Description | Selected |
|--------|-------------|----------|
| Use known results, cite references | Apply existing G2 structure theorems | |
| Develop novel G2 results if needed | New math if literature insufficient | ✓ |

### Both regimes

| Option | Description | Selected |
|--------|-------------|----------|
| Characterize both (when global suffices AND when adaptive needed) | Boundary between regimes is interesting | ✓ |
| Focus on winner only | Only develop theory for best approach | |

### Monte Carlo

| Option | Description | Selected |
|--------|-------------|----------|
| MC + analytical | Sample + fit + compare to bounds | ✓ |
| Analytical only | Pure math | |

### Closed-form investigation

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, investigate closed-form | Connection to algebra constants (structure constants, Fano geometry) | ✓ |
| Empirical optimal sufficient | Report number only | |

### Meta-trie convergence theory

| Option | Description | Selected |
|--------|-------------|----------|
| Formal convergence analysis | Fixed point analysis, characterize conditions | ✓ |
| Empirical convergence only | Report that it does/doesn't converge | |

### Stability-plasticity formalization

| Option | Description | Selected |
|--------|-------------|----------|
| Formal connection | Derive conditions for balance | ✓ |
| Informal motivation only | Mention but don't formalize | |

### Cayley-Dickson connection

| Option | Description | Selected |
|--------|-------------|----------|
| Octonion-specific + CD if helpful | Claude's discretion | ✓ |

### Complexity analysis

| Option | Description | Selected |
|--------|-------------|----------|
| Formal complexity comparison | Big-O for each ThresholdPolicy | ✓ |
| Empirical timing sufficient | Wall-clock from sweep | |

### Narrative framing

| Option | Description | Selected |
|--------|-------------|----------|
| Self-organization narrative | "Trie discovers its own operating parameters" | ✓ |
| Empirical science narrative | Systematic characterization | |
| Both, layered | Methods + interpretation | |

### Mathematical references

| Option | Description | Selected |
|--------|-------------|----------|
| Let researcher discover | Survey literature, no specific starting points | ✓ |

### Update defaults

| Option | Description | Selected |
|--------|-------------|----------|
| Update default if justified | Change trie.py defaults based on results | ✓ |
| Leave 0.3, document only | Keep current default | |

---

## Claude's Discretion

- Significance level (likely Bonferroni-corrected p < 0.05)
- Cayley-Dickson connection depth
- Specific similarity_threshold sweep range
- 5 representative consolidation configs
- Additional mathematical references
- Specific adaptive hyperparameter ranges (EMA alpha, decay factors)
