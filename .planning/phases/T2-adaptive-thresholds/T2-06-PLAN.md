---
phase: T2-adaptive-thresholds
plan: 06
type: execute
wave: 5
depends_on: ["T2-05"]
files_modified:
  - scripts/sweep/run_purity_sweep.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "AlgebraicPurityPolicy swept with assoc_weight, sim_weight, sensitivity hyperparameters"
    - "Two independent purity signals tested: associator norm variance and routing key similarity variance"
    - "Noise interaction specifically tested with AlgebraicPurityPolicy"
    - "Results compared against best EMA/MeanStd/Depth strategies"
  artifacts:
    - path: "scripts/sweep/run_purity_sweep.py"
      provides: "Algebraic purity strategy sweep with noise interaction"
      contains: "def run_purity_sweep"
  key_links:
    - from: "scripts/sweep/run_purity_sweep.py"
      to: "src/octonion/trie.py"
      via: "AlgebraicPurityPolicy construction"
      pattern: "AlgebraicPurityPolicy"
---

<objective>
Run algebraic purity strategy sweep (D-01 strategy 4) and characterize noise interaction across strategies.

Purpose: Per D-01, strategy 4 uses associator norm variance and routing key similarity variance as independent signals. Per D-05, noise interaction is the 4th sweep dimension. This plan specifically investigates whether noise and algebraic purity interact synergistically (noise helps purity-based adaptation discover better thresholds).

Output: Purity strategy sweep results, noise interaction analysis, comparison against strategies 1-3.
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
</context>

<tasks>

<task type="auto">
  <name>Task 1: Algebraic purity strategy sweep with noise interaction</name>
  <files>scripts/sweep/run_purity_sweep.py</files>
  <read_first>
    - src/octonion/trie.py (AlgebraicPurityPolicy constructor, hyperparameters)
    - scripts/sweep/run_adaptive_sweep.py (adaptive sweep pattern)
    - scripts/sweep/sweep_runner.py (SweepRunner API)
    - .planning/phases/T2-adaptive-thresholds/T2-05-SUMMARY.md (best strategies 1-3 results)
  </read_first>
  <action>
Create scripts/sweep/run_purity_sweep.py for strategy 4.

**AlgebraicPurityPolicy hyperparameter sweep** (per D-01 strategy 4, D-29):
- assoc_weight: [0.0, 0.3, 0.5, 0.7, 1.0] -- weight for associator norm variance signal
  - assoc_weight=0.0 tests similarity variance signal alone
  - assoc_weight=1.0 tests associator norm variance signal alone
- sim_weight: [0.0, 0.3, 0.5, 0.7, 1.0] -- weight for routing key similarity variance
  - At least one of assoc_weight/sim_weight must be > 0
- sensitivity: [0.1, 0.3, 0.5, 1.0, 2.0] -- how strongly purity affects threshold
- base_assoc: top-3 values from global sweep
- sim_threshold: top-2 from global sweep (per D-03)

**Independent signal testing** (per D-01 "test associator norm variance and routing key similarity variance as independent signals"):
- Phase A: assoc_weight > 0, sim_weight = 0 (assoc variance only)
- Phase B: assoc_weight = 0, sim_weight > 0 (sim variance only)
- Phase C: both > 0 (combined)

Initial sweep: 5*5*5*3*2 = 750 per benchmark, minus invalid (both weights=0) = 720.
But many are redundant (weight=0 cases). Reduce: Phase A (5 assoc * 5 sens * 3 base * 2 sim = 150), Phase B (5 sim * 5 sens * 3 base * 2 sim = 150), Phase C (4*4*5*3*2 = 480 but reduce to top pairs) = ~400 per benchmark.

**Noise interaction sweep** (per D-05):
For top-10 purity configs, run full noise sweep: [0.0, 0.01, 0.05, 0.1].
Also run noise sweep on top-5 configs from each of strategies 1-3 (from SQLite).
This characterizes whether noise helps or hurts each adaptive strategy.

**CLI interface**:
```
python scripts/sweep/run_purity_sweep.py --features-dir results/T2/features --db results/T2/sweep.db --workers 24
python scripts/sweep/run_purity_sweep.py --noise-interaction  # runs noise interaction after purity sweep
```

**Comparison output**: After sweep, print:
1. Best purity config vs best global, EMA, mean_std, depth per benchmark
2. Independent signal analysis: which signal (assoc variance, sim variance, or combined) performs best
3. Noise interaction: which strategy-noise combinations are synergistic

Per D-31, seed=42. Per D-07, all 5 benchmarks.
  </action>
  <verify>
    <automated>docker compose run --rm dev uv run python scripts/sweep/run_purity_sweep.py --help</automated>
  </verify>
  <acceptance_criteria>
    - scripts/sweep/run_purity_sweep.py exists
    - run_purity_sweep.py contains `AlgebraicPurityPolicy`
    - run_purity_sweep.py contains assoc_weight values
    - run_purity_sweep.py contains sim_weight values
    - run_purity_sweep.py contains sensitivity values [0.1, 0.3, 0.5, 1.0, 2.0]
    - run_purity_sweep.py contains `--noise-interaction` flag
    - run_purity_sweep.py queries SQLite for best configs from strategies 1-3
    - run_purity_sweep.py tests each signal independently (assoc-only, sim-only, combined)
  </acceptance_criteria>
  <done>Strategy 4 swept with all hyperparameters, independent signals tested, noise interaction characterized across all strategies, comparison table vs strategies 1-3 and global printed</done>
</task>

</tasks>

<verification>
docker compose run --rm dev uv run python scripts/sweep/run_purity_sweep.py --help
</verification>

<success_criteria>
- AlgebraicPurityPolicy tested with full hyperparameter grid
- Independent signal analysis shows which purity signal matters most
- Noise interaction characterized for all 4 adaptive strategies + global
- Results in SQLite alongside all prior sweep data
</success_criteria>

<output>
After completion, create `.planning/phases/T2-adaptive-thresholds/T2-06-SUMMARY.md`
</output>
