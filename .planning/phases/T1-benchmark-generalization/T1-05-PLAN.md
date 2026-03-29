---
phase: T1-benchmark-generalization
plan: 05
type: execute
wave: 3
depends_on: [T1-02, T1-03, T1-04]
files_modified:
  - scripts/run_trie_benchmark_summary.py
autonomous: false
requirements: [TRIE-01]

must_haves:
  truths:
    - "Cross-benchmark comparison table shows trie vs all baselines across all benchmarks"
    - "Per-benchmark analysis identifies which categories the trie handles well vs poorly"
    - "Trie structure analysis (node count, depth, branching factor) compared across benchmarks"
    - "User has reviewed all benchmark results and confirmed findings"
  artifacts:
    - path: "scripts/run_trie_benchmark_summary.py"
      provides: "Cross-benchmark aggregation and summary script"
      min_lines: 100
    - path: "results/trie_benchmarks/summary.json"
      provides: "Aggregated cross-benchmark results"
      contains: "benchmarks"
  key_links:
    - from: "scripts/run_trie_benchmark_summary.py"
      to: "results/trie_benchmarks/fashion_mnist/results.json"
      via: "JSON loading"
      pattern: "json.load"
    - from: "scripts/run_trie_benchmark_summary.py"
      to: "results/trie_benchmarks/cifar10/results.json"
      via: "JSON loading"
      pattern: "json.load"
    - from: "scripts/run_trie_benchmark_summary.py"
      to: "results/trie_benchmarks/text/results.json"
      via: "JSON loading"
      pattern: "json.load"
---

<objective>
Create a cross-benchmark summary script that aggregates results from all three benchmarks (Fashion-MNIST, CIFAR-10, text) and produces a unified comparison table. Then verify all results with the user.

Purpose: The individual benchmark scripts each produce their own results. This script combines them into a single comparison view that answers the phase question: does the trie generalize? Per user decision, analyze which categories the trie handles well vs poorly, and compare trie structure across benchmarks.

Output: `scripts/run_trie_benchmark_summary.py` + human verification of all results.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/phases/T1-benchmark-generalization/T1-CONTEXT.md
@.planning/phases/T1-benchmark-generalization/T1-02-SUMMARY.md
@.planning/phases/T1-benchmark-generalization/T1-03-SUMMARY.md
@.planning/phases/T1-benchmark-generalization/T1-04-SUMMARY.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create cross-benchmark summary script</name>
  <files>scripts/run_trie_benchmark_summary.py</files>
  <action>
Create `scripts/run_trie_benchmark_summary.py` that loads results from all three benchmark scripts and produces a unified analysis.

**Input:** Load JSON results from:
- results/trie_benchmarks/fashion_mnist/results.json
- results/trie_benchmarks/cifar10/results.json (use best encoder for comparison table)
- results/trie_benchmarks/text/results.json (use both full and subset)
- results/trie_validation/mnist_benchmark.json (existing MNIST baseline for reference)

If any result file is missing, skip that benchmark with a warning (allow partial runs).

**Output 1: Cross-Benchmark Comparison Table (printed to stdout):**
```
=====================================================================
OCTONIONIC TRIE -- CROSS-BENCHMARK COMPARISON
=====================================================================
Benchmark         | Trie  | kNN-5 | kNN-1 | RF    | SVM   | LR    | CNN Head
------------------|-------|-------|-------|-------|-------|-------|--------
MNIST (PCA-8D)    | 95.2% | 93.4% | ...
Fashion-MNIST     | ...
CIFAR-10 (best)   | ...
Text (20 classes)  | ...
Text (4 classes)   | ...

Target: Trie within 5pp of kNN-5 on same features
```

**Output 2: Trie vs kNN Gap Analysis:**
For each benchmark, compute trie_accuracy - knn_k5_accuracy. Flag benchmarks where gap > 5pp as "below target" per user decision.

**Output 3: Per-Benchmark Failure Mode Summary:**
For each benchmark, identify the 3 classes with lowest trie accuracy and the 3 with highest. Print a table showing best/worst classes.

**Output 4: Trie Structure Comparison:**
Compare trie stats across benchmarks:
```
Benchmark         | Nodes | Leaves | Max Depth | Avg Depth
MNIST             | ...
Fashion-MNIST     | ...
CIFAR-10          | ...
Text (20 classes) | ...
```

**Output 5: Save aggregated results:**
Save to results/trie_benchmarks/summary.json with all the above data structured.

**CLI args:**
- --results-dir: default results/trie_benchmarks
- --mnist-results: default results/trie_validation/mnist_benchmark.json
- --output-dir: default results/trie_benchmarks

The script should be runnable with `docker compose run --rm dev uv run python scripts/run_trie_benchmark_summary.py` without any arguments (all defaults).
  </action>
  <verify>
    <automated>docker compose run --rm dev uv run python -c "import ast; ast.parse(open('scripts/run_trie_benchmark_summary.py').read()); print('Syntax OK')"</automated>
  </verify>
  <done>Summary script exists and is syntactically valid. When run after all benchmarks complete, produces cross-benchmark comparison table, gap analysis, failure mode summary, trie structure comparison, and aggregated results JSON.</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 2: Verify all benchmark results</name>
  <files>results/trie_benchmarks/</files>
  <action>
Present all benchmark results to the user for review. The user runs the benchmark scripts (development-scale or full-scale) and verifies:

1. Run all benchmarks with development-scale data (fast):
   ```
   docker compose run --rm dev uv run python scripts/run_trie_fashion_mnist.py --n-train 1000 --n-test 200 --cnn-epochs 3 --epochs 2
   docker compose run --rm dev uv run python scripts/run_trie_cifar10.py --encoder 2layer --n-train 1000 --n-test 200 --epochs 1
   docker compose run --rm dev uv run python scripts/run_trie_text.py --mode both --epochs 1
   ```
2. Run summary: `docker compose run --rm dev uv run python scripts/run_trie_benchmark_summary.py`
3. Check that the comparison table prints correctly with accuracy values for all methods
4. Verify confusion matrix PNGs exist in results/trie_benchmarks/{fashion_mnist,cifar10,text}/
5. Review the gap analysis: is the trie within 5pp of kNN on each benchmark?

For final publishable results, run with full data (will take longer):
   ```
   docker compose run --rm dev uv run python scripts/run_trie_fashion_mnist.py --n-train 10000 --n-test 2000
   docker compose run --rm dev uv run python scripts/run_trie_cifar10.py --encoder all --n-train 10000 --n-test 2000
   docker compose run --rm dev uv run python scripts/run_trie_text.py --mode both
   docker compose run --rm dev uv run python scripts/run_trie_benchmark_summary.py
   ```
  </action>
  <verify>User confirms results are reasonable and the phase question is answered</verify>
  <done>User has reviewed cross-benchmark comparison and confirmed findings. Type "approved" or describe issues to fix.</done>
</task>

</tasks>

<verification>
- Summary script runs without error when result files exist
- Cross-benchmark comparison table is printed with accuracy values for all methods
- Gap analysis identifies which benchmarks meet the 5pp target
- Aggregated results saved to results/trie_benchmarks/summary.json
</verification>

<success_criteria>
All benchmark results are aggregated into a single comparison view. User has reviewed the cross-benchmark comparison and confirmed the findings. The phase question ("does the trie generalize?") is answered with data.
</success_criteria>

<output>
After completion, create `.planning/phases/T1-benchmark-generalization/T1-05-SUMMARY.md`
</output>
