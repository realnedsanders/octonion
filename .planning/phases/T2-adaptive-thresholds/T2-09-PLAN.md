---
phase: T2-adaptive-thresholds
plan: 09
type: execute
wave: 8
depends_on: ["T2-08"]
files_modified:
  - scripts/sweep/sweep_analysis.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "Paired Wilcoxon signed-rank test and paired t-test run on multi-seed results per D-34"
    - "Bootstrap 95% confidence intervals computed per D-34"
    - "Cohen's d effect sizes reported alongside p-values per D-35"
    - "Friedman test + rank analysis for cross-benchmark consistency per D-36"
    - "Structural variance reported as mean +/- std per D-37"
    - "Bonferroni correction applied per D-38"
    - "Auto-recommendation identifies best config per D-30"
    - "All results output as JSON + tables per D-41"
  artifacts:
    - path: "scripts/sweep/sweep_analysis.py"
      provides: "Automated statistical analysis script"
      contains: "def run_full_analysis"
    - path: "results/T2/analysis/statistical_report.json"
      provides: "Complete statistical analysis output"
  key_links:
    - from: "scripts/sweep/sweep_analysis.py"
      to: "results/T2/sweep.db"
      via: "sqlite3 queries for multi-seed results"
      pattern: "sqlite3\\.connect"
    - from: "scripts/sweep/sweep_analysis.py"
      to: "scipy.stats"
      via: "wilcoxon, ttest_rel, friedmanchisquare, bootstrap"
      pattern: "scipy\\.stats"
---

<objective>
Run comprehensive statistical analysis on all sweep results and produce auto-recommendation.

Purpose: Per D-33-D-41, this plan produces the full statistical analysis of all threshold strategies. Per D-30, auto-recommend best configuration. Per D-41, automated script (not notebook). Per D-45, characterize both when global suffices and when adaptive is needed.

Output: statistical_report.json with all tests, tables, and auto-recommendation. PNG plots for key comparisons.
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
@.planning/phases/T2-adaptive-thresholds/T2-08-SUMMARY.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Automated statistical analysis script</name>
  <files>scripts/sweep/sweep_analysis.py</files>
  <read_first>
    - scripts/sweep/sweep_runner.py (SQLite schema, column names)
    - scripts/sweep/sweep_plots.py (existing visualization patterns)
    - .planning/phases/T2-adaptive-thresholds/T2-CONTEXT.md (D-30, D-33 through D-41, D-45)
  </read_first>
  <action>
Create scripts/sweep/sweep_analysis.py -- an automated analysis script per D-41.

**Statistical tests** (all use scipy.stats):

1. **Paired comparisons** (per D-34):
   For each pair (best_adaptive vs best_global, best_hybrid vs best_global, etc.):
   - `scipy.stats.wilcoxon(x, y)` -- Wilcoxon signed-rank test
   - `scipy.stats.ttest_rel(x, y)` -- paired t-test
   - `scipy.stats.bootstrap((x - y,), np.mean, confidence_level=0.95)` -- bootstrap 95% CI
   - Input: arrays of accuracy values across 10 seeds per D-33
   - Pairs compared: best_global vs each of (ema, mean_std, depth, purity, meta_trie, hybrid)

2. **Effect sizes** (per D-35):
   ```python
   def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
       diff = x - y
       return diff.mean() / diff.std(ddof=1)
   ```
   Report Cohen's d alongside each p-value.

3. **Cross-benchmark consistency** (per D-36):
   - `scipy.stats.friedmanchisquare(*ranks_per_benchmark)` -- Friedman test
   - Rank each strategy within each benchmark (1 = best, N = worst)
   - Report mean rank, median rank per strategy
   - If Friedman significant, post-hoc Nemenyi test or pairwise Wilcoxon with Bonferroni

4. **Bonferroni correction** (per D-38):
   - Number of comparisons: 6 strategies vs global = 6 pairwise tests
   - Corrected alpha = 0.05 / 6 = 0.00833
   - Report both raw and corrected p-values
   - Per research pitfall 8: for cross-benchmark, use Friedman + Nemenyi instead of Bonferroni

5. **Structural variance** (per D-37):
   For each top config's 10-seed runs:
   - node_count: mean +/- std
   - max_depth: mean +/- std
   - branching_factor: mean +/- std
   - rumination_rejections: mean +/- std
   - consolidation_merges: mean +/- std

6. **Generalization gap** (per D-40):
   Compare 10K subset accuracy vs full training set accuracy for top configs.
   Report absolute difference and relative difference.

7. **Auto-recommendation** (per D-30):
   Identify top configs by:
   - Pareto rank (accuracy vs node count) across benchmarks
   - Cross-benchmark consistency (Friedman rank)
   - Generalization gap (prefer configs where gap is small)
   - Output: "Recommended configuration: {policy_type} with params {params}. Justification: {reasoning}"

8. **Regime characterization** (per D-45):
   Characterize WHEN global suffices vs WHEN adaptive is needed:
   - Per-benchmark analysis: which benchmarks benefit from adaptive?
   - Feature analysis: do benchmarks with more classes or harder separation benefit more?
   - Threshold: what accuracy delta qualifies as "adaptive is better"?

**Output format** (per D-41):

`results/T2/analysis/statistical_report.json`:
```json
{
    "pairwise_tests": {
        "global_vs_ema": {
            "wilcoxon_p": 0.031,
            "ttest_p": 0.028,
            "cohens_d": 0.42,
            "bootstrap_ci_95": [0.002, 0.018],
            "bonferroni_wilcoxon_p": 0.186,
            "bonferroni_ttest_p": 0.168,
            "significant_raw": true,
            "significant_corrected": false
        }
    },
    "friedman": {
        "statistic": 12.4,
        "p_value": 0.029,
        "mean_ranks": {"global": 3.2, "ema": 2.1, ...}
    },
    "structural_variance": {...},
    "generalization_gap": {...},
    "recommendation": {
        "policy_type": "...",
        "params": {...},
        "justification": "..."
    },
    "regime_analysis": {...}
}
```

Also generate summary tables to stdout and comparison plots to results/T2/analysis/:
- strategy_comparison.png: bar chart of mean accuracy per strategy across benchmarks
- per_benchmark_comparison.png: grouped bar chart per benchmark
- friedman_ranks.png: CD diagram or rank plot

**CLI interface**:
```
python scripts/sweep/sweep_analysis.py --db results/T2/sweep.db --output-dir results/T2/analysis
```

Per D-39, report ALL results (no fixed practical significance threshold).
  </action>
  <verify>
    <automated>docker compose run --rm dev uv run python -c "from scripts.sweep.sweep_analysis import run_full_analysis, cohens_d; print('Import OK')"</automated>
  </verify>
  <acceptance_criteria>
    - scripts/sweep/sweep_analysis.py exists
    - sweep_analysis.py contains `scipy.stats.wilcoxon` per D-34
    - sweep_analysis.py contains `scipy.stats.ttest_rel` per D-34
    - sweep_analysis.py contains `scipy.stats.bootstrap` per D-34
    - sweep_analysis.py contains `def cohens_d` per D-35
    - sweep_analysis.py contains `scipy.stats.friedmanchisquare` per D-36
    - sweep_analysis.py contains Bonferroni correction (p * n_comparisons) per D-38
    - sweep_analysis.py contains `def run_full_analysis`
    - sweep_analysis.py contains auto-recommendation logic per D-30
    - sweep_analysis.py writes statistical_report.json per D-41
    - sweep_analysis.py contains `--output-dir` argparse argument
  </acceptance_criteria>
  <done>Full statistical analysis script produces JSON report with all required tests, auto-recommendation, regime characterization, comparison plots, covering D-30 and D-33 through D-41</done>
</task>

</tasks>

<verification>
docker compose run --rm dev uv run python -c "
from scripts.sweep.sweep_analysis import run_full_analysis, cohens_d
import numpy as np
# Smoke test cohens_d
x = np.array([0.95, 0.94, 0.96])
y = np.array([0.93, 0.92, 0.94])
d = cohens_d(x, y)
assert abs(d - 2.0) < 0.5, f'Unexpected Cohen d: {d}'
print('Analysis import and smoke test OK')
"
</verification>

<success_criteria>
- All D-33 through D-41 statistical requirements implemented
- Auto-recommendation identifies best configuration with justification
- Regime analysis characterizes when global vs adaptive is better
- JSON report and PNG plots generated
- All results reported without practical significance cutoff per D-39
</success_criteria>

<output>
After completion, create `.planning/phases/T2-adaptive-thresholds/T2-09-SUMMARY.md`
</output>
