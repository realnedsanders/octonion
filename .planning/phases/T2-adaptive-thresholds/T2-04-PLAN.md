---
phase: T2-adaptive-thresholds
plan: 04
type: execute
wave: 3
depends_on: ["T2-02", "T2-03"]
files_modified:
  - scripts/sweep/run_global_sweep.py
  - scripts/sweep/sweep_plots.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "Global threshold sensitivity sweep completed across all 5 benchmarks on 10K subsets"
    - "2D heatmaps (assoc x sim) show accuracy landscape for each benchmark"
    - "1D line plots show accuracy vs each parameter with others fixed at best"
    - "Pareto frontier (accuracy vs node count) identifies efficient configurations"
    - "Noise interaction effect measured across all benchmarks per D-05"
  artifacts:
    - path: "scripts/sweep/run_global_sweep.py"
      provides: "Global threshold sweep execution script"
      contains: "def main"
    - path: "scripts/sweep/sweep_plots.py"
      provides: "Visualization functions for sweep results"
      contains: "def plot_heatmap"
    - path: "results/T2/sweep.db"
      provides: "SQLite database with all sweep results"
  key_links:
    - from: "scripts/sweep/run_global_sweep.py"
      to: "scripts/sweep/sweep_runner.py"
      via: "SweepRunner.run(configs, features_dir)"
      pattern: "SweepRunner"
    - from: "scripts/sweep/sweep_plots.py"
      to: "results/T2/sweep.db"
      via: "sqlite3 queries for plot data"
      pattern: "sqlite3\\.connect"
---

<objective>
Run the global threshold sensitivity sweep and produce visualizations.

Purpose: Per D-20, sweep 4D grid (assoc x sim x consolidation x noise) on 10K subsets. Per D-22, reduced-first approach. Per D-21, produce 2D heatmaps, 1D line plots, Pareto frontiers. Per D-05, noise becomes the 4th sweep dimension. Per D-06, sweep epochs (1, 3, 5).

Output: Populated SQLite database with ~2000+ configs per benchmark, visualization PNGs, identification of top global configurations.
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
@.planning/phases/T2-adaptive-thresholds/T2-02-SUMMARY.md
@.planning/phases/T2-adaptive-thresholds/T2-03-SUMMARY.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Global sweep execution script</name>
  <files>scripts/sweep/run_global_sweep.py</files>
  <read_first>
    - scripts/sweep/sweep_runner.py (SweepRunner, SweepConfig, generate_global_sweep_configs)
    - scripts/sweep/cache_features.py (feature file naming convention)
    - .planning/phases/T2-adaptive-thresholds/T2-CONTEXT.md (D-20 through D-28)
  </read_first>
  <action>
Create scripts/sweep/run_global_sweep.py that orchestrates the global threshold sensitivity sweep.

**Sweep design** (per D-20, D-22):
Phase 1 -- Core 3D sweep (assoc x sim x noise) with fixed consolidation=(0.05,3), epochs=3:
- assoc_threshold: np.unique(np.sort(np.concatenate([np.geomspace(0.001, 2.0, 15), np.linspace(0.05, 1.0, 10)]))) -- ~22 values
- sim_threshold: [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5] -- 8 values
- noise: [0.0, 0.01, 0.05, 0.1] -- 4 values per D-05
- Total: ~22 * 8 * 4 = 704 configs per benchmark
- 5 benchmarks per D-07: 3520 total configs

Phase 2 -- Consolidation sweep: Fix assoc and sim to top-5 from Phase 1, sweep consolidation:
- consolidation: [(0.01,1), (0.03,2), (0.05,3), (0.10,5), (0.00,0)] -- 5 configs per D-20
- ~5 * 5 * 5 benchmarks = 125 configs

Phase 3 -- Epoch sweep: Fix to top-10 from Phase 1+2, sweep epochs per D-06:
- epochs: [1, 3, 5] -- 3 values
- ~10 * 3 * 5 benchmarks = 150 configs

All on 10K subset per D-22. Per D-31, seed=42.

**CLI interface**:
```
python scripts/sweep/run_global_sweep.py --phase 1 --features-dir results/T2/features --db results/T2/sweep.db --workers 24
python scripts/sweep/run_global_sweep.py --phase 2 --features-dir results/T2/features --db results/T2/sweep.db --workers 24
python scripts/sweep/run_global_sweep.py --phase 3 --features-dir results/T2/features --db results/T2/sweep.db --workers 24
python scripts/sweep/run_global_sweep.py --phase all  # runs 1, 2, 3 sequentially
```

Per D-27, all configs run to completion (no early stopping).
Per D-28, tqdm progress bar via SweepRunner.
Per D-31, fixed seed=42 for sweep exploration.

After Phase 1 completes, query SQLite for top-5 (assoc, sim) pairs by mean accuracy across benchmarks and use those for Phase 2. After Phase 2, query top-10 overall configs for Phase 3.

Print summary after each phase: best accuracy per benchmark, top-5 configs, Pareto frontier points.
  </action>
  <verify>
    <automated>docker compose run --rm dev uv run python scripts/sweep/run_global_sweep.py --help</automated>
  </verify>
  <acceptance_criteria>
    - scripts/sweep/run_global_sweep.py exists
    - run_global_sweep.py contains `--phase` argparse argument
    - run_global_sweep.py contains `--workers` argparse argument with default 24
    - run_global_sweep.py imports SweepRunner from sweep.sweep_runner
    - run_global_sweep.py contains `geomspace(0.001, 2.0, 15)`
    - run_global_sweep.py contains `linspace(0.05, 1.0, 10)`
    - run_global_sweep.py contains noise values [0.0, 0.01, 0.05, 0.1]
    - run_global_sweep.py contains epoch values [1, 3, 5]
    - run_global_sweep.py references all 5 benchmarks
  </acceptance_criteria>
  <done>Global sweep script handles 3-phase progressive sweep design, integrates with SweepRunner framework, produces populated SQLite database</done>
</task>

<task type="auto">
  <name>Task 2: Sweep visualization functions</name>
  <files>scripts/sweep/sweep_plots.py</files>
  <read_first>
    - scripts/sweep/sweep_runner.py (SQLite schema, column names)
    - scripts/trie_benchmark_utils.py (existing matplotlib patterns -- Agg backend, plot_confusion_matrix)
    - .planning/phases/T2-adaptive-thresholds/T2-CONTEXT.md (D-21 visualization requirements)
  </read_first>
  <action>
Create scripts/sweep/sweep_plots.py with visualization functions per D-21.

**matplotlib.use("Agg")** at top for headless rendering.

**Functions**:

1. `plot_heatmap(db_path, benchmark, x_param, y_param, metric, fixed_params, save_path)`:
   - Query SQLite for the 2D grid of (x_param, y_param) values at final epoch
   - fixed_params: dict of param_name -> value to fix (e.g., {"noise": 0.0, "epochs": 3})
   - Use matplotlib.pyplot.imshow with annotation and colorbar
   - Title: f"{benchmark}: {metric} vs {x_param} x {y_param}"
   - Save as PNG at save_path

2. `plot_1d_sweep(db_path, benchmark, param, metric, fixed_params, save_path)`:
   - Query SQLite for 1D sweep of param values
   - Line plot with markers, error bars from epoch variance
   - X-axis: param values, Y-axis: metric (accuracy)
   - Save as PNG

3. `plot_pareto_frontier(db_path, benchmark, save_path)`:
   - Query all configs for this benchmark at final epoch
   - X-axis: n_nodes, Y-axis: accuracy
   - Scatter all points, highlight Pareto frontier
   - Pareto: sort by accuracy desc, filter by cumulative min node count
   - Save as PNG

4. `plot_noise_interaction(db_path, benchmark, save_path)`:
   - For each noise level, plot accuracy vs assoc_threshold (with sim_threshold fixed at best)
   - Multiple lines (one per noise level) on same plot
   - Shows whether noise helps/hurts at different threshold values

5. `plot_epoch_curves(db_path, benchmark, config_ids, save_path)`:
   - For selected config_ids, plot accuracy vs epoch
   - Shows convergence behavior

6. `generate_all_plots(db_path, output_dir)`:
   - Calls all plot functions for all benchmarks
   - Creates output_dir/plots/ with organized PNG files
   - Per D-21: 2D heatmaps, 1D line plots, Pareto frontiers

**CLI interface**:
```
python scripts/sweep/sweep_plots.py --db results/T2/sweep.db --output-dir results/T2/plots
```

All plots use consistent styling: figsize=(10, 8) for heatmaps, (10, 6) for line plots, dpi=150, tight_layout.
  </action>
  <verify>
    <automated>docker compose run --rm dev uv run python -c "from scripts.sweep.sweep_plots import plot_heatmap, plot_1d_sweep, plot_pareto_frontier, generate_all_plots; print('Import OK')"</automated>
  </verify>
  <acceptance_criteria>
    - scripts/sweep/sweep_plots.py exists
    - sweep_plots.py contains `matplotlib.use("Agg")`
    - sweep_plots.py contains `def plot_heatmap(`
    - sweep_plots.py contains `def plot_1d_sweep(`
    - sweep_plots.py contains `def plot_pareto_frontier(`
    - sweep_plots.py contains `def plot_noise_interaction(`
    - sweep_plots.py contains `def plot_epoch_curves(`
    - sweep_plots.py contains `def generate_all_plots(`
    - sweep_plots.py contains `sqlite3.connect`
    - sweep_plots.py contains `dpi=150`
  </acceptance_criteria>
  <done>6 visualization functions implemented, all reading from SQLite and producing PNG files, consistent styling, CLI interface for batch generation</done>
</task>

</tasks>

<verification>
docker compose run --rm dev uv run python -c "
from scripts.sweep.sweep_plots import plot_heatmap, plot_1d_sweep, plot_pareto_frontier
from scripts.sweep.sweep_runner import SweepRunner, generate_global_sweep_configs
print('All imports OK')
"
</verification>

<success_criteria>
- Global sweep script produces ~3500+ configs across 5 benchmarks
- SQLite database populated with epoch-by-epoch results
- 2D heatmaps, 1D line plots, Pareto frontiers generated as PNG
- Noise interaction effect visible across thresholds
- Top global configurations identified for comparison with adaptive strategies
</success_criteria>

<output>
After completion, create `.planning/phases/T2-adaptive-thresholds/T2-04-SUMMARY.md`
</output>
