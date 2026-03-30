---
phase: T2-adaptive-thresholds
plan: 05
type: execute
wave: 4
depends_on: ["T2-03", "T2-04"]
files_modified:
  - scripts/sweep/run_adaptive_sweep.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "PerNodeEMAPolicy sweep tests alpha=[0.01,0.05,0.1,0.2,0.5] and k=[0.5,1.0,1.5,2.0,3.0]"
    - "PerNodeMeanStdPolicy sweep tests k=[0.5,1.0,1.5,2.0,3.0]"
    - "DepthPolicy sweep tests decay_factor=[0.7,0.8,0.9,1.0,1.1,1.2,1.3]"
    - "All 3 strategies compared against best-tuned global on all 5 benchmarks"
    - "Results stored in same SQLite database alongside global sweep results"
  artifacts:
    - path: "scripts/sweep/run_adaptive_sweep.py"
      provides: "Adaptive strategy sweep execution for strategies 1-3"
      contains: "def run_ema_sweep"
  key_links:
    - from: "scripts/sweep/run_adaptive_sweep.py"
      to: "scripts/sweep/sweep_runner.py"
      via: "SweepRunner.run() with adaptive configs"
      pattern: "SweepRunner"
    - from: "scripts/sweep/run_adaptive_sweep.py"
      to: "src/octonion/trie.py"
      via: "PerNodeEMAPolicy, PerNodeMeanStdPolicy, DepthPolicy construction"
      pattern: "PerNodeEMAPolicy|PerNodeMeanStdPolicy|DepthPolicy"
---

<objective>
Run adaptive strategy sweeps for strategies 1-3 (EMA, mean+std, depth-dependent) with hyperparameter tuning.

Purpose: Per D-01, test strategies in order. Per D-29, sweep adaptive strategy hyperparameters for fair comparison. Per D-08, build understanding progressively before meta-trie. Per D-03, co-adapt sim_threshold and consolidation alongside assoc_threshold.

Output: Sweep results for 3 adaptive strategies in SQLite, comparison against best-tuned global, identification of top performers.
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
@.planning/phases/T2-adaptive-thresholds/T2-03-SUMMARY.md
@.planning/phases/T2-adaptive-thresholds/T2-04-SUMMARY.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Adaptive strategies 1-3 sweep script</name>
  <files>scripts/sweep/run_adaptive_sweep.py</files>
  <read_first>
    - scripts/sweep/sweep_runner.py (SweepRunner, SweepConfig, generate_adaptive_sweep_configs)
    - scripts/sweep/run_global_sweep.py (sweep execution pattern)
    - src/octonion/trie.py (PerNodeEMAPolicy, PerNodeMeanStdPolicy, DepthPolicy constructors and hyperparameters)
    - .planning/phases/T2-adaptive-thresholds/T2-04-SUMMARY.md (best global configs from Phase 1 sweep)
  </read_first>
  <action>
Create scripts/sweep/run_adaptive_sweep.py that runs hyperparameter sweeps for strategies 1-3.

**Strategy 1 -- PerNodeEMAPolicy** (per D-01, D-29):
Sweep hyperparameters:
- alpha (EMA decay): [0.01, 0.05, 0.1, 0.2, 0.5]
- k (std multiplier): [0.5, 1.0, 1.5, 2.0, 3.0]
- base_assoc: top-5 assoc_threshold values from global sweep
- sim_threshold: top-3 sim_threshold values from global sweep (per D-03 co-adaptation)
- consolidation: top-2 configs from global sweep (per D-03)
- noise: [0.0, best_noise_from_global] (per D-05)
- epochs: 3 (default)
- Total: ~5 * 5 * 5 * 3 * 2 * 2 = 1500 per benchmark, but reduce by fixing noise=0 and consolidation=(0.05,3) for initial pass: 5*5*5*3 = 375 per benchmark

**Strategy 2 -- PerNodeMeanStdPolicy** (per D-01):
Sweep hyperparameters:
- k: [0.5, 1.0, 1.5, 2.0, 3.0]
- base_assoc: top-5 from global sweep
- sim_threshold: top-3 from global sweep
- Total initial: 5*5*3 = 75 per benchmark

**Strategy 3 -- DepthPolicy** (per D-01):
Sweep hyperparameters (per D-01: both directions):
- decay_factor: [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
- base_assoc: top-5 from global sweep
- sim_threshold: top-3 from global sweep
- Total initial: 7*5*3 = 105 per benchmark

**Total initial sweep**: ~555 configs per benchmark, 2775 total. Feasible with 24 workers.

**After initial sweep**, run expanded sweep on top-10 configs per strategy with:
- Full consolidation sweep (5 configs)
- Full noise sweep (4 values)
- Full epoch sweep (1, 3, 5)

**CLI interface**:
```
python scripts/sweep/run_adaptive_sweep.py --strategy ema --features-dir results/T2/features --db results/T2/sweep.db --workers 24
python scripts/sweep/run_adaptive_sweep.py --strategy mean_std --features-dir results/T2/features --db results/T2/sweep.db
python scripts/sweep/run_adaptive_sweep.py --strategy depth --features-dir results/T2/features --db results/T2/sweep.db
python scripts/sweep/run_adaptive_sweep.py --strategy all  # runs all 3
```

**Comparison output**: After each strategy sweep, query SQLite for:
1. Best config per benchmark (max accuracy)
2. Best config across benchmarks (best mean rank per D-36 Friedman approach)
3. Delta vs best global config per benchmark
4. Print comparison table to stdout

Per D-31, seed=42. Per D-27, all configs run to completion.

The script should read the top global configs from SQLite (query sweep.db for policy_type="global" ORDER BY accuracy DESC).

Per D-03, sim_threshold and consolidation params are co-swept (not fixed while sweeping assoc). The config generation must cross these with the adaptive hyperparameters.
  </action>
  <verify>
    <automated>docker compose run --rm dev uv run python scripts/sweep/run_adaptive_sweep.py --help</automated>
  </verify>
  <acceptance_criteria>
    - scripts/sweep/run_adaptive_sweep.py exists
    - run_adaptive_sweep.py contains `--strategy` argparse with choices including ema, mean_std, depth, all
    - run_adaptive_sweep.py contains `PerNodeEMAPolicy`
    - run_adaptive_sweep.py contains `PerNodeMeanStdPolicy`
    - run_adaptive_sweep.py contains `DepthPolicy`
    - run_adaptive_sweep.py contains EMA alpha values [0.01, 0.05, 0.1, 0.2, 0.5]
    - run_adaptive_sweep.py contains k values [0.5, 1.0, 1.5, 2.0, 3.0]
    - run_adaptive_sweep.py contains decay_factor values [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    - run_adaptive_sweep.py queries SQLite for best global configs as comparison baseline
    - run_adaptive_sweep.py imports SweepRunner
  </acceptance_criteria>
  <done>Strategies 1-3 swept with full hyperparameter grids, results in SQLite alongside global sweep, comparison table printed showing delta vs best-tuned global</done>
</task>

</tasks>

<verification>
docker compose run --rm dev uv run python scripts/sweep/run_adaptive_sweep.py --help
</verification>

<success_criteria>
- All 3 strategies have hyperparameter sweeps completed on all 5 benchmarks
- Results stored in same SQLite database as global sweep
- Comparison table shows each strategy's best accuracy vs best global accuracy per benchmark
- Co-adaptation of sim_threshold and consolidation verified (not swept in isolation)
</success_criteria>

<output>
After completion, create `.planning/phases/T2-adaptive-thresholds/T2-05-SUMMARY.md`
</output>
