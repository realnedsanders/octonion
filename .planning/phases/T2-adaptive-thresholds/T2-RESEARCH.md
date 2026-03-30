# Phase T2: Adaptive Thresholds - Research

**Researched:** 2026-03-29
**Domain:** Adaptive threshold strategies for octonionic trie routing, parallel sweep infrastructure, statistical analysis, octonionic algebra theory
**Confidence:** HIGH

## Summary

Phase T2 is a large, multi-faceted research phase that investigates whether the octonionic trie's associator threshold should be global, per-node adaptive, depth-dependent, algebraic-purity-based, meta-trie-optimized, or a hybrid. The phase also co-adapts rumination and consolidation thresholds, builds reusable parallel sweep infrastructure with SQLite storage, produces theoretical analysis connecting thresholds to octonionic algebra, and introduces a meta-trie optimizer.

The core codebase is well-suited for this work. The `OctonionTrie` class (418 lines) uses a single `self.assoc_threshold` float that's compared at exactly two points (line 145 in `_find_best_child` and lines 262, 269 in `insert`). The `self.sim_threshold` is used in `_ruminate` (lines 176, 179), and consolidation uses hardcoded 0.05/3 thresholds (lines 400-404). All of these are clean insertion points for a `ThresholdPolicy` abstraction. The existing `ProcessPoolExecutor`-based parallel runner (`run_trie_benchmarks_parallel.py`) and the benchmark utilities provide a solid foundation for the sweep infrastructure.

A critical analytical result exists: Greg Egan computed the exact mean associator norm for random unit octonions as 147456/(42875*pi) approximately 1.0947, which provides a theoretical anchor for understanding why a global threshold might work (or not) and what the "natural" scale of the associator is on the unit 7-sphere.

**Primary recommendation:** Build the ThresholdPolicy abstraction first (Wave 0/1), then the reusable parallel sweep framework with SQLite (Wave 2), then run sensitivity sweeps and strategies in order of complexity (Wave 3-5), then meta-trie (Wave 6), then statistical analysis and theory (Wave 7-8). Feature caching is a prerequisite that should precede all sweeps.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Test 6 strategies against global baseline, in order: (1) per-node EMA of associator norms, (2) per-node running mean+std, (3) depth-dependent with both directions (tighter and looser with depth, sweep decay factors 0.7-1.3), (4) algebraic purity -- test associator norm variance and routing key similarity variance as independent signals, (5) meta-trie optimizer, (6) hybrid combining top 2 performers
- **D-02:** All strategies must be unsupervised -- threshold adaptation uses only algebraic signals (associator norms, routing structure), never category labels. Labels used only for accuracy evaluation.
- **D-03:** Co-adapt rumination threshold (similarity_threshold) and consolidation thresholds (min_share, min_count) alongside associator threshold. Not isolated experiments.
- **D-04:** Adaptive thresholds apply during both training (insert) and inference (query). Learned per-node thresholds become part of the trie's structure.
- **D-05:** Include noise revisit from T1 deferral. Test noise as interaction effect with adaptive thresholds. Noise becomes 4th sweep dimension.
- **D-06:** Sweep training epochs (1, 3, 5) for each strategy to test epoch-threshold interaction.
- **D-07:** Run all experiments on all T1 benchmarks: MNIST, Fashion-MNIST, CIFAR-10 (ResNet-8), Text 4-class, Text 20-class.
- **D-08:** Test simple strategies first, meta-trie last. Build understanding progressively. Meta-trie design benefits from knowing what simpler strategies reveal.
- **D-09:** Hybrid strategy tested regardless of individual strategy results (always test, not gated on individual performance).
- **D-10:** Add pluggable ThresholdPolicy to OctonionTrie in trie.py. All strategies (GlobalPolicy, PerNodeEMAPolicy, DepthPolicy, AlgebraicPurityPolicy, MetaTriePolicy, HybridPolicy) are implementations. Classifier trie doesn't know how thresholds are set.
- **D-11:** Update trie.py defaults if results justify a different global threshold or default ThresholdPolicy.
- **D-12:** A second OctonionTrie acts as optimizer for the classifier trie. Uses the same OctonionTrie class (not a subclass). Strongest thesis statement: the trie optimizes itself using the same algebra.
- **D-13:** Meta-trie categories are discretized threshold actions (e.g., "tighten 10%", "keep", "loosen 10%", "loosen 20%").
- **D-14:** Test two meta-trie input encodings independently: (1) direct 8D signal vector (assoc_norm_mean, assoc_norm_std, branching_factor/7, insert_rate, rumination_rate, depth/max_depth, buffer_consistency, consolidation_rate), (2) classifier node's routing key or content octonion as algebraic input.
- **D-15:** Test two meta-trie feedback signals independently: (1) classifier trie stability (low rumination rejections, balanced branching, consistent associator norms -- fully unsupervised), (2) classifier accuracy on held-out set (supervised, for comparison).
- **D-16:** Test multiple update frequencies: per-100-inserts, per-1000-inserts, per-epoch. Sweep to find right granularity.
- **D-17:** Test both fixed meta-trie thresholds (simple bootstrap) and self-referential (meta-trie adapts its own thresholds) independently.
- **D-18:** Meta-trie convergence criterion: threshold change rate < 1% between updates. Track convergence curve.
- **D-19:** MetaTriePolicy is an implementation of ThresholdPolicy -- integrates into the same abstraction as simple strategies.
- **D-20:** 4D sweep grid: associator threshold (0.001-2.0, wide log-spaced) x similarity threshold (range TBD) x consolidation (5 configs) x noise (0.0, 0.01, 0.05, 0.1). ~2000 configs per benchmark.
- **D-21:** 2D heatmaps for assoc x similarity grid, line plots for 1D sweeps, Pareto frontier (accuracy vs node count). Matplotlib, saved as PNG.
- **D-22:** Reduced-first approach: 10K subset for initial 4D sweep, full-scale on top 10 configurations.
- **D-23:** Pre-compute and cache encoder features. Workers load 8D features and only run the trie. No GPU needed for sweep.
- **D-24:** 24-worker ProcessPoolExecutor for parallel sweep. Each config is independent.
- **D-25:** SQLite database for all sweep results. Epoch-by-epoch tracking (accuracy + structure after each epoch).
- **D-26:** Reusable parallel sweep framework -- generic runner with param grid, worker function, SQLite output. Reusable by T4, T6.
- **D-27:** Run all configs to completion (no early stopping). Even degenerate configs run all epochs.
- **D-28:** tqdm progress bar for live progress.
- **D-29:** Sweep adaptive strategy hyperparameters too (EMA alpha, depth decay factor, purity sensitivity, etc.) using same infrastructure. Fair comparison requires tuned adaptive vs tuned global.
- **D-30:** Auto-recommend best configuration with justification. Script identifies top configs by Pareto rank and cross-benchmark consistency.
- **D-31:** Fixed seed (42) for sweep exploration. Multi-seed (10 seeds) for top-config validation.
- **D-32:** Meta-trie experiments use same sweep infrastructure (MetaTriePolicy plugs in like any other policy).
- **D-33:** 10 random seeds for top-config validation runs.
- **D-34:** Both paired Wilcoxon signed-rank test AND paired t-test, plus bootstrap 95% confidence intervals for accuracy differences.
- **D-35:** Cohen's d effect sizes alongside p-values.
- **D-36:** Friedman test + rank analysis for cross-benchmark consistency.
- **D-37:** Full structural variance across seeds: mean +/- std for node count, max depth, branching factor, rumination rate.
- **D-38:** Significance threshold: Claude's discretion (likely p < 0.05 with Bonferroni correction, but field-appropriate).
- **D-39:** No fixed practical significance threshold -- report all results. Context determines meaningfulness.
- **D-40:** Generalization gap: compare 10K subset vs full training set accuracy for top configs.
- **D-41:** Automated analysis script (not notebook). Reads SQLite, runs all statistical tests, produces tables and plots. JSON + PNG output.
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
- **D-54:** Full diagnostic output: per-node associator norm distributions (histograms), depth profiles, routing statistics. Produces visualizations supporting thesis and informing future phases.

### Claude's Discretion

- Significance level choice (likely Bonferroni-corrected p < 0.05)
- Cayley-Dickson connection depth
- Specific similarity_threshold sweep range for 2D grid
- 5 representative consolidation configs (min_share x min_count combinations)
- Mathematical references beyond Baez (2002) and Harvey (1990)
- Specific EMA decay rates, depth decay factors, and purity sensitivity values to sweep

### Deferred Ideas (OUT OF SCOPE)

None -- discussion stayed within phase scope.

</user_constraints>

## Project Constraints (from CLAUDE.md)

- **All Python commands MUST run inside the dev container** via `docker compose run --rm dev`
- **Never run `uv`, `python`, or `pytest` directly on the host**
- Python 3.12+, type hints on all public APIs
- pytest + hypothesis for property-based testing
- Minimize dependencies beyond PyTorch
- Container has: ROCm 7.2, PyTorch 2.9.1, Python 3.12, uv
- File edits happen on the host (mounted at `/workspace` in container)
- Dataset caching via Docker volume at `/workspace/.data`
- Agg matplotlib backend for headless rendering
- float64 by default for algebraic precision

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.9.1 (container) | Tensor ops, octonion algebra | Already in use, ROCm-enabled |
| scipy | 1.17.0 (verified in container) | Statistical tests (Wilcoxon, Friedman, t-test, bootstrap) | Already installed; D-34, D-35, D-36 |
| matplotlib | 3.10.8 (in pyproject.toml) | 2D heatmaps, line plots, histograms, Pareto frontiers | Already in use; D-21 |
| scikit-learn | 1.8.0 (in pyproject.toml) | Baselines (kNN, RF, SVM, LR) for sweep evaluation | Already in use from T1 |
| sqlite3 | 3.45.1 (stdlib, verified) | Sweep results storage with WAL mode | D-25; stdlib, no install needed |
| tqdm | 4.67.1 (verified in container) | Progress bars for parallel sweeps | D-28; already installed |
| numpy | >=1.26 (dev dependency) | Array operations, statistical computations | Already in use |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| concurrent.futures | stdlib | ProcessPoolExecutor for parallel sweeps | D-24; 24-worker parallelism |
| json | stdlib | Config serialization, intermediate output | Schema for sweep configs |
| dataclasses | stdlib | ThresholdPolicy, SweepConfig data structures | Clean API design |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| SQLite | JSON files per config | SQLite enables cross-config queries, epoch-by-epoch tracking; JSON would require post-hoc aggregation |
| ProcessPoolExecutor | Ray/Dask | Unnecessary complexity; each trie run is pure CPU, no distributed state needed |
| Manual bootstrap | scipy.stats.bootstrap | scipy.stats.bootstrap available since scipy 1.7; use it for D-34 confidence intervals |
| pingouin for stats | scipy.stats only | pingouin adds effect sizes; but Cohen's d is trivial to compute manually from paired differences -- avoid extra dependency |

**Installation:**
No new packages needed. All dependencies are already installed in the container.

**Version verification:**
```
scipy: 1.17.0 (verified in container)
sqlite3: 3.45.1 (verified in container)
tqdm: 4.67.1 (verified in container)
matplotlib: 3.10.8 (in pyproject.toml)
scikit-learn: 1.8.0 (in pyproject.toml)
```

## Architecture Patterns

### Recommended Project Structure
```
src/octonion/
├── trie.py                    # OctonionTrie + ThresholdPolicy abstraction
├── _fano.py                   # Fano plane (unchanged)
├── _octonion.py               # Octonion class, associator (unchanged)
├── _multiplication.py         # octonion_mul (unchanged)
scripts/
├── sweep/
│   ├── sweep_runner.py        # Reusable parallel sweep framework (D-26)
│   ├── sweep_analysis.py      # Automated statistical analysis (D-41)
│   ├── sweep_plots.py         # Heatmaps, Pareto frontiers, diagnostics (D-21)
│   ├── cache_features.py      # Pre-compute encoder features for all benchmarks (D-23)
│   └── meta_trie_sweep.py     # Meta-trie specific sweep configs (D-32)
├── trie_benchmark_utils.py    # Existing utils (extended for policy support)
├── run_trie_*.py              # Existing benchmark scripts (unchanged)
├── theory/
│   └── monte_carlo_assoc.py   # Monte Carlo validation of analytical bounds (D-46)
tests/
├── test_trie.py               # Extended with ThresholdPolicy tests
├── test_threshold_policy.py   # Dedicated policy tests
├── test_sweep_runner.py       # Sweep infrastructure tests
results/
└── T2/
    ├── sweep.db               # SQLite database for all sweep results (D-25)
    ├── plots/                 # Generated PNG visualizations
    └── analysis/              # JSON statistical outputs
docs/thesis/
└── oct-trie.tex               # New section on threshold theory (D-52)
```

### Pattern 1: ThresholdPolicy Abstraction (D-10)
**What:** Abstract base class that replaces `self.assoc_threshold` (float) with a policy object providing per-node, per-context threshold values.
**When to use:** All threshold computations in `_find_best_child`, `insert`, `_ruminate`, `_consolidate_node`.
**Example:**
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

class ThresholdPolicy(ABC):
    """Base class for threshold adaptation strategies."""

    @abstractmethod
    def get_assoc_threshold(self, node: TrieNode, depth: int) -> float:
        """Return associator threshold for the given node context."""
        ...

    @abstractmethod
    def get_sim_threshold(self, node: TrieNode, depth: int) -> float:
        """Return similarity threshold for rumination at this node."""
        ...

    @abstractmethod
    def get_consolidation_params(
        self, node: TrieNode, depth: int
    ) -> tuple[float, int]:
        """Return (min_share, min_count) for consolidation at this node."""
        ...

    def on_insert(
        self, node: TrieNode, x: torch.Tensor, assoc_norm: float
    ) -> None:
        """Hook called after each insertion for policy updates (e.g., EMA)."""
        pass


class GlobalPolicy(ThresholdPolicy):
    """Fixed global thresholds (current behavior, baseline)."""

    def __init__(
        self,
        assoc_threshold: float = 0.3,
        sim_threshold: float = 0.1,
        min_share: float = 0.05,
        min_count: int = 3,
    ):
        self.assoc_threshold = assoc_threshold
        self.sim_threshold = sim_threshold
        self.min_share = min_share
        self.min_count = min_count

    def get_assoc_threshold(self, node: TrieNode, depth: int) -> float:
        return self.assoc_threshold

    def get_sim_threshold(self, node: TrieNode, depth: int) -> float:
        return self.sim_threshold

    def get_consolidation_params(
        self, node: TrieNode, depth: int
    ) -> tuple[float, int]:
        return self.min_share, self.min_count
```

### Pattern 2: Parallel Sweep Framework (D-26)
**What:** Generic parallel sweep runner that takes a parameter grid, a worker function, and writes results to SQLite.
**When to use:** All sweep experiments across all phases (T2, T4, T6).
**Example:**
```python
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from tqdm import tqdm

@dataclass
class SweepConfig:
    """Single sweep configuration."""
    config_id: int
    assoc_threshold: float
    sim_threshold: float
    min_share: float
    min_count: int
    noise: float
    epochs: int
    seed: int
    benchmark: str
    # Policy-specific params stored as JSON string
    policy_params: str = "{}"

def run_sweep(
    configs: list[SweepConfig],
    worker_fn: callable,
    db_path: str,
    n_workers: int = 24,
) -> None:
    """Run parallel sweep, writing results to SQLite."""
    _init_db(db_path)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(worker_fn, cfg): cfg
            for cfg in configs
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Sweep"
        ):
            cfg = futures[future]
            result = future.result()
            _write_result(db_path, cfg, result)

def _init_db(db_path: str) -> None:
    """Create SQLite tables with WAL mode."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sweep_results (
            config_id INTEGER,
            benchmark TEXT,
            epoch INTEGER,
            accuracy REAL,
            n_nodes INTEGER,
            n_leaves INTEGER,
            max_depth INTEGER,
            rumination_rejections INTEGER,
            consolidation_merges INTEGER,
            assoc_threshold REAL,
            sim_threshold REAL,
            min_share REAL,
            min_count INTEGER,
            noise REAL,
            epochs INTEGER,
            seed INTEGER,
            policy_type TEXT,
            policy_params TEXT,
            train_time REAL,
            test_time REAL,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
```

### Pattern 3: Feature Caching (D-23)
**What:** Pre-compute all encoder features (8D unit octonions) for all benchmarks, save to disk. Sweep workers load cached features only.
**When to use:** Before any sweep experiments. Eliminates GPU/encoder dependency from sweep workers.
**Example:**
```python
def cache_features(benchmark: str, output_dir: str) -> None:
    """Pre-compute and cache encoder features for a benchmark."""
    if benchmark == "mnist":
        train_x, train_y, test_x, test_y = load_mnist_pca8(n_train=60000, n_test=10000)
    elif benchmark == "fashion_mnist":
        # Load and encode via CNN
        ...
    # Save as .pt files
    torch.save({
        "train_x": train_x, "train_y": train_y,
        "test_x": test_x, "test_y": test_y,
    }, f"{output_dir}/{benchmark}_features.pt")
```

### Pattern 4: SQLite Write Pattern for Multi-Process
**What:** Each worker opens its own SQLite connection, writes results with WAL mode. Single writer at a time (SQLite limitation), but WAL mode allows concurrent reads.
**When to use:** All sweep result storage.
**Critical:** Workers must NOT share a connection. Each process creates its own connection with a timeout to handle write contention.
```python
def _write_result(db_path: str, config: SweepConfig, result: dict) -> None:
    """Write a single result from a worker process."""
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "INSERT INTO sweep_results (...) VALUES (...)",
        (...)
    )
    conn.commit()
    conn.close()
```

### Pattern 5: PerNodeEMAPolicy (D-01, strategy 1)
**What:** Each node maintains an exponential moving average of observed associator norms. The threshold is set as `mean + k * std` of the EMA.
**When to use:** Strategy 1 in the progression.
```python
class PerNodeEMAPolicy(ThresholdPolicy):
    """Per-node EMA of associator norms."""

    def __init__(self, alpha: float = 0.1, k: float = 1.5, base: float = 0.3):
        self.alpha = alpha
        self.k = k
        self.base = base
        # Node state stored in node itself via on_insert hook
        # Uses node.buffer or a separate dict keyed by node id

    def get_assoc_threshold(self, node: TrieNode, depth: int) -> float:
        state = self._get_state(node)
        if state.count < 3:
            return self.base  # Fallback until enough data
        return state.mean + self.k * state.std

    def on_insert(self, node: TrieNode, x: torch.Tensor, assoc_norm: float) -> None:
        state = self._get_state(node)
        state.update(assoc_norm, self.alpha)
```

### Anti-Patterns to Avoid
- **Sharing SQLite connections across processes:** SQLite connections are NOT fork-safe. Each process must create its own connection. Violating this causes `OperationalError: database is locked` or silent corruption.
- **Using labels in threshold adaptation:** D-02 explicitly forbids this. All threshold updates must use algebraic signals only. Labels are only for evaluation metrics stored in the results database.
- **Passing OctonionTrie objects between processes:** Trie objects are not picklable (contain deques of tensors). Pass configuration parameters and let each worker construct its own trie.
- **GPU in sweep workers:** D-23 mandates feature pre-caching. Sweep workers must not import torch.cuda or require GPU. Features are pre-computed 8D float64 tensors.
- **Modifying trie.py's public API during insert/query:** The ThresholdPolicy must be injected at construction time. Insert and query signatures should not change -- only internal behavior changes based on the policy.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Statistical tests | Custom Wilcoxon/Friedman | `scipy.stats.wilcoxon`, `scipy.stats.friedmanchisquare`, `scipy.stats.ttest_rel` | Edge cases in tie-handling, continuity corrections; scipy handles all |
| Bootstrap CIs | Manual resampling loop | `scipy.stats.bootstrap` (available since scipy 1.7) | Handles BCa method, confidence level, random state properly |
| Parallel execution | Manual multiprocessing with Queue | `concurrent.futures.ProcessPoolExecutor` with `as_completed` | Clean future-based API, exception propagation, context manager cleanup |
| Progress bars | Print statements with counters | `tqdm` with `total` parameter around `as_completed` | Handles terminal width, rate estimation, ETA, nested bars |
| Bonferroni correction | Manual p-value division | Multiply p-values by number of comparisons, cap at 1.0 | Simple enough to do manually, but document the number of comparisons |
| Effect sizes | Custom formula | `cohens_d = mean_diff / pooled_std` | Cohen's d is simple, but use the formula consistently: `d = (M1 - M2) / sqrt((s1^2 + s2^2) / 2)` |
| 2D heatmaps | Manual grid plotting | `matplotlib.pyplot.imshow` or `seaborn.heatmap` | Annotation, colorbar, axis labels handled correctly |
| Pareto frontier | Custom dominance check | Sort by accuracy desc, filter by cumulative min node count | Standard algorithm, but easy to get wrong with ties |

**Key insight:** The statistical analysis (D-33 through D-41) involves well-established methods with known edge cases. Using scipy's implementations avoids subtle bugs in tie handling, small-sample corrections, and distributional assumptions.

## Common Pitfalls

### Pitfall 1: SQLite Write Contention in Parallel Sweeps
**What goes wrong:** 24 workers writing to the same SQLite database simultaneously causes `OperationalError: database is locked` or very slow throughput.
**Why it happens:** SQLite allows only one writer at a time. WAL mode helps by not blocking readers, but writes still serialize.
**How to avoid:** Use WAL mode (`PRAGMA journal_mode=WAL`), set timeout to 30 seconds (`sqlite3.connect(db_path, timeout=30.0)`), and batch writes. Each worker should write one row per completed config, not per epoch. Alternatively, workers accumulate results in memory and write all epochs for one config in a single transaction.
**Warning signs:** Sweep runtime dominated by I/O waits rather than trie computation; "database is locked" errors in logs.

### Pitfall 2: Non-Picklable Trie Objects in ProcessPoolExecutor
**What goes wrong:** Passing OctonionTrie instances or TrieNodes as arguments to worker functions fails with pickle errors (deque of tensors, nested dataclasses).
**Why it happens:** ProcessPoolExecutor serializes arguments via pickle. Complex nested objects with tensors and deques don't serialize cleanly.
**How to avoid:** Pass only primitive configuration values (floats, ints, strings) and file paths to workers. Each worker constructs its own OctonionTrie internally.
**Warning signs:** `pickle.PicklingError` or `AttributeError: Can't pickle` at sweep start.

### Pitfall 3: Biased Adaptive Threshold Comparison
**What goes wrong:** Comparing an optimally-tuned global threshold against an untuned adaptive strategy, or vice versa, produces misleading results.
**Why it happens:** D-29 explicitly requires sweeping adaptive hyperparameters too. If EMA alpha is fixed at 0.1 while global threshold is swept over 20 values, the comparison is unfair.
**How to avoid:** For each adaptive strategy, sweep its hyperparameters (EMA alpha, depth decay, purity sensitivity) using the same sweep infrastructure. Compare best-tuned global against best-tuned adaptive.
**Warning signs:** Adaptive strategies consistently losing to global by small margins.

### Pitfall 4: Feature Caching Inconsistency
**What goes wrong:** Pre-cached features were computed with different seeds, encoder epochs, or normalization than the original benchmarks, causing accuracy discrepancies.
**Why it happens:** Feature caching script doesn't exactly replicate the original benchmark's data pipeline.
**How to avoid:** Feature caching script must use identical data loading, encoding, and normalization as the original benchmark scripts. Validate by running one config with cached features and comparing against original benchmark results.
**Warning signs:** Sweep baseline (global 0.3 threshold, 3 epochs) doesn't match T1 reported accuracy.

### Pitfall 5: Meta-Trie State Leaking Between Configs
**What goes wrong:** Meta-trie optimizer accumulates state from previous sweep configs if not properly reset.
**Why it happens:** If the meta-trie is shared across configs or not freshly initialized per sweep worker.
**How to avoid:** Each sweep worker creates a fresh classifier trie AND a fresh meta-trie. No state sharing between configs.
**Warning signs:** Meta-trie results depend on execution order.

### Pitfall 6: Epoch-by-Epoch Tracking Overhead
**What goes wrong:** Evaluating on the full test set after every epoch during a 4D sweep (2000 configs x 5 benchmarks x 5 epochs) creates 50,000 evaluation passes, dominating runtime.
**Why it happens:** D-25 requires epoch-by-epoch tracking.
**How to avoid:** Use the 10K subset (D-22) for the initial 4D sweep. Full-scale evaluation only on the top 10 configs. For the 10K sweep, evaluation on a 2K test set per epoch is manageable.
**Warning signs:** Sweep taking days instead of hours; >80% time spent in evaluation, not training.

### Pitfall 7: Log-Spaced Threshold Values Missing Critical Region
**What goes wrong:** Log-spacing from 0.001 to 2.0 clusters points near 0.001 and 2.0 but may miss the 0.1-0.5 region where the default 0.3 lives and where most interesting behavior occurs.
**Why it happens:** Pure log-spacing covers many decades but is sparse within any single decade.
**How to avoid:** Use np.geomspace for broad coverage plus np.linspace for fine resolution in the 0.05-1.0 range. Example: `np.unique(np.sort(np.concatenate([np.geomspace(0.001, 2.0, 15), np.linspace(0.05, 1.0, 10)])))`.
**Warning signs:** Accuracy cliff between two adjacent sweep points suggesting important behavior is between them.

### Pitfall 8: Bonferroni Over-Correction
**What goes wrong:** With 6 strategies x 5 benchmarks x multiple comparisons, Bonferroni correction with k=30+ comparisons makes everything non-significant.
**Why it happens:** Bonferroni is very conservative for large numbers of comparisons.
**How to avoid:** D-38 says "field-appropriate." For ML benchmarks, Bonferroni over 6 strategies (k=6) is reasonable. For within-strategy cross-benchmark tests, Friedman + post-hoc Nemenyi is more appropriate than pairwise Bonferroni. Report both corrected and uncorrected p-values.
**Warning signs:** All p-values > 0.05 after correction despite visible accuracy differences.

## Code Examples

### Associator Norm Distribution Sampling (D-46)
```python
# Source: Greg Egan analytical result + project's _octonion.py
import torch
from octonion._octonion import Octonion, associator

def sample_associator_norms(n_samples: int = 100000, seed: int = 42) -> torch.Tensor:
    """Monte Carlo sampling of associator norms on the unit 7-sphere."""
    gen = torch.Generator().manual_seed(seed)
    norms = torch.zeros(n_samples, dtype=torch.float64)
    for i in range(n_samples):
        # Sample uniformly from S^7 (Gaussian normalization)
        a = torch.randn(8, dtype=torch.float64, generator=gen)
        b = torch.randn(8, dtype=torch.float64, generator=gen)
        c = torch.randn(8, dtype=torch.float64, generator=gen)
        a, b, c = a / a.norm(), b / b.norm(), c / c.norm()
        assoc = associator(Octonion(a), Octonion(b), Octonion(c))
        norms[i] = assoc.components.norm()
    return norms

# Expected mean: 147456 / (42875 * pi) ~ 1.0947
# This establishes the "natural scale" of associator norms
```

### Cohen's d Computation (D-35)
```python
import numpy as np

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Paired Cohen's d for two matched arrays of accuracy values."""
    diff = x - y
    return diff.mean() / diff.std(ddof=1)
```

### SQLite Schema for Sweep Results (D-25)
```python
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sweep_results (
    config_id INTEGER NOT NULL,
    benchmark TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    -- Threshold parameters
    policy_type TEXT NOT NULL,
    assoc_threshold REAL,
    sim_threshold REAL,
    min_share REAL,
    min_count INTEGER,
    noise REAL,
    -- Policy-specific hyperparameters (JSON)
    policy_params TEXT DEFAULT '{}',
    -- Accuracy metrics
    accuracy REAL,
    -- Trie structure metrics
    n_nodes INTEGER,
    n_leaves INTEGER,
    max_depth INTEGER,
    rumination_rejections INTEGER,
    consolidation_merges INTEGER,
    branching_factor_mean REAL,
    branching_factor_std REAL,
    -- Timing
    train_time REAL,
    test_time REAL,
    -- Metadata
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (config_id, benchmark, epoch, seed)
)
"""

CREATE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_benchmark ON sweep_results(benchmark)",
    "CREATE INDEX IF NOT EXISTS idx_policy ON sweep_results(policy_type)",
    "CREATE INDEX IF NOT EXISTS idx_accuracy ON sweep_results(accuracy DESC)",
]
```

### ThresholdPolicy Integration Points in trie.py
```python
# Current code (line 145):
#   if assoc_norm < self.assoc_threshold:
# Becomes:
#   if assoc_norm < self.policy.get_assoc_threshold(child, node.depth):

# Current code (lines 176, 179):
#   if key_sim < self.sim_threshold * 0.5:
#   return sum(sims) / len(sims) > self.sim_threshold * 0.3
# Becomes:
#   sim_thresh = self.policy.get_sim_threshold(node, node.depth)
#   if key_sim < sim_thresh * 0.5:
#   return sum(sims) / len(sims) > sim_thresh * 0.3

# Current code (lines 400-404):
#   if child.insert_count / max(total, 1) < 0.05 and child.insert_count < 3
# Becomes:
#   min_share, min_count = self.policy.get_consolidation_params(node, node.depth)
#   if child.insert_count / max(total, 1) < min_share and child.insert_count < min_count
```

### Similarity Threshold Sweep Range (Claude's Discretion)
```python
# Recommendation: sim_threshold range based on T1 observations
# Current default is 0.1. Inner products of unit octonions range [-1, 1].
# For well-separated classes, similarities are typically 0.3-0.8.
# For overlapping classes, similarities drop to 0.0-0.3.
# Sweep range: [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
SIM_THRESHOLD_VALUES = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
```

### Consolidation Config Sweep (Claude's Discretion)
```python
# 5 representative (min_share, min_count) combinations:
CONSOLIDATION_CONFIGS = [
    (0.01, 1),   # Very aggressive: prune anything below 1%
    (0.03, 2),   # Moderate-aggressive
    (0.05, 3),   # Current default
    (0.10, 5),   # Conservative: only prune very underused
    (0.00, 0),   # No consolidation (disabled)
]
```

### EMA/Depth/Purity Hyperparameter Sweep (Claude's Discretion)
```python
# EMA decay rates (alpha):
EMA_ALPHAS = [0.01, 0.05, 0.1, 0.2, 0.5]
# k multiplier for mean + k*std threshold:
EMA_K_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0]

# Depth decay factors (threshold *= decay^depth):
DEPTH_DECAY_FACTORS = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

# Purity sensitivity (how strongly purity signal affects threshold):
PURITY_SENSITIVITY = [0.1, 0.3, 0.5, 1.0, 2.0]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Fixed global threshold 0.3 | TBD (this phase determines) | T2 | Core contribution of the phase |
| Hardcoded consolidation 0.05/3 | Co-adapted with assoc_threshold | T2 | D-03 requires co-adaptation |
| No epoch tracking | Epoch-by-epoch metrics in SQLite | T2 | Enables convergence analysis |
| JSON results per experiment | SQLite database | T2 | Enables cross-config queries, D-25 |

**Mathematical Foundation:**
- Mean associator norm for random unit octonions: 147456/(42875*pi) ~ 1.0947 (Egan, analytical)
- The associator vanishes precisely on associative (quaternionic) three-planes
- G2 (14-dimensional exceptional Lie group) is the automorphism group of the octonions
- The associator is totally antisymmetric and has no real part
- Alternativity: the associator vanishes when any two arguments are equal

**Key References:**
- Baez (2002): "The Octonions" -- comprehensive survey, Fano plane conventions
- Harvey (1990): "Spinors and Calibrations" -- G2 geometry, calibration forms
- Egan: "Peeling the Octonions" -- analytical mean associator norm
- Salamon & Walpuski (ETH): "Notes on the Octonions" -- modern treatment, G2 cross-sections

## Mathematical Theory Context (D-42 through D-51)

### Associator Norm Distribution Analysis

The mean associator norm for random unit octonions is exactly 147456/(42875*pi) ~ 1.0947. This is computed by integrating over three copies of S^7 (a 21-dimensional integral). The key observation for T2: this mean applies to RANDOM unit octonions. In the trie, unit octonions are NOT random -- they are class-structured. The question is whether this structure causes the associator norm distribution to deviate significantly from the random baseline.

**Research approach for theory:**
1. Sample associator norms from random unit octonion triples (Monte Carlo validation of Egan's result)
2. Sample associator norms from WITHIN-class triples (should be smaller if classes are coherent)
3. Sample associator norms from BETWEEN-class triples (should differ from random)
4. If within-class norms are consistently smaller than the random mean, this provides a theoretical justification for a global threshold: set it between the within-class and between-class distributions

### Fano Plane Geometry Argument

Each routing decision projects onto one of 7 quaternionic subalgebras. Within a quaternionic subalgebra, the algebra IS associative (associator is identically zero). The threshold controls the "radius of associativity" around each subalgebra. If data naturally clusters near subalgebra axes (as the subalgebra_activation routing suggests), then the associator norm for within-cluster triples may have a universal upper bound independent of the data distribution.

**Potential proof structure:**
1. Show that for unit octonions within angular distance epsilon of a quaternionic subalgebra, the associator norm is bounded by O(epsilon^2)
2. The routing mechanism (subalgebra_activation) routes inputs to their nearest subalgebra
3. Therefore, within a well-routed node, the associator norm is bounded
4. This bound depends on the angular separation between subalgebras (a Fano plane constant), not on the data

### G2 Connection

G2 is the automorphism group of the octonions (preserves the multiplication table). G2 acts transitively on the unit 6-sphere in the imaginary octonions (S^6). The isotropy subgroup of G2 at any point of S^6 is SU(3). This means the associator norm distribution on S^7 has G2 symmetry. Any threshold derived from the associator must be G2-invariant.

**Confidence:** MEDIUM -- The proof outline is plausible but the specific bounds need to be computed. The key question is whether the angular separation between subalgebras (Fano plane geometry) gives a tight enough bound for practical threshold setting.

### Stability-Plasticity Framework (D-49)

- **Tight threshold (small):** Only very compatible inputs pass through -- high stability (existing routing preserved), low plasticity (hard to add new categories)
- **Loose threshold (large):** Most inputs are compatible -- low stability (routing becomes ambiguous), high plasticity (new categories easily accommodated)
- The optimal threshold balances these pressures. In neural CL literature, this is the stability-plasticity dilemma.
- **Theoretical prediction:** The optimal threshold should be near the within-class associator norm distribution's upper tail, rejecting between-class inputs while accepting within-class ones. This is analogous to a Neyman-Pearson threshold in hypothesis testing.

## Open Questions

1. **What is the actual within-class vs between-class associator norm distribution on the trie's data?**
   - What we know: Random unit octonion mean is ~1.095. Trie data is NOT random.
   - What's unclear: How much smaller are within-class associator norms? Is there a clean separation?
   - Recommendation: Monte Carlo sampling from actual T1 benchmark features (first experiment to run).

2. **Does the Fano plane angular separation give a useful bound?**
   - What we know: The 7 subalgebras are at specific geometric positions. Angular separation is computable.
   - What's unclear: Whether the bound is tight enough to be practically useful vs. the empirical distribution.
   - Recommendation: Compute the bound analytically, then compare against measured distributions.

3. **How will the meta-trie's convergence behave?**
   - What we know: Fixed-point analysis is the right framework (D-48). The meta-trie is a discrete dynamical system.
   - What's unclear: Whether convergence is monotone, oscillatory, or chaotic. Whether the self-referential variant (D-17) converges at all.
   - Recommendation: Track convergence curves for all meta-trie configurations. Plot threshold change rate vs. update count.

4. **Sweep scale: is ~2000 configs per benchmark feasible in reasonable time?**
   - What we know: Each trie run on 10K samples takes ~5-15 seconds (from T1 benchmarks). 2000 configs x 5 benchmarks = 10000 runs. At 10s each with 24 workers: ~70 minutes.
   - What's unclear: Whether epoch-by-epoch evaluation adds significant overhead.
   - Recommendation: Time a single config with epoch-by-epoch evaluation. If >30s, reduce sweep resolution for initial exploration.

5. **Similarity threshold range: what values are meaningful?**
   - What we know: Current default is 0.1. Inner products of random unit octonions in R^8 concentrate near 0.
   - What's unclear: The typical range of inner products for within-class trie data.
   - Recommendation: Use range [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5] initially. Refine based on diagnostics.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Docker + docker compose | Container execution | Yes | N/A | -- |
| Python 3.12 | All code | Yes (container) | 3.12 | -- |
| PyTorch | Tensor ops | Yes (container) | 2.9.1 | -- |
| scipy | Statistical tests | Yes (container) | 1.17.0 | -- |
| tqdm | Progress bars | Yes (container) | 4.67.1 | -- |
| sqlite3 | Sweep storage | Yes (stdlib) | 3.45.1 | -- |
| matplotlib | Plotting | Yes (container) | 3.10.8 | -- |
| scikit-learn | Baselines | Yes (container) | 1.8.0 | -- |
| 32 CPU cores | Parallel sweep | Yes | -- | Reduce worker count |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:** None.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (with hypothesis) |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `docker compose run --rm dev uv run pytest tests/test_trie.py -x` |
| Full suite command | `docker compose run --rm dev uv run pytest -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SC-1 | Per-node adaptive threshold tested vs global | integration | `docker compose run --rm dev uv run pytest tests/test_threshold_policy.py -x -k "per_node"` | Wave 0 |
| SC-2 | Context-specific threshold (depth, purity) tested | integration | `docker compose run --rm dev uv run pytest tests/test_threshold_policy.py -x -k "depth or purity"` | Wave 0 |
| SC-3 | Global threshold mathematical justification | manual-only | N/A (theory/proof work) | N/A |
| SC-4 | Adaptive improvement statistically significant | integration | `docker compose run --rm dev uv run pytest tests/test_sweep_runner.py -x -k "statistical"` | Wave 0 |
| SC-5 | Sensitivity analysis: accuracy vs threshold | integration | `docker compose run --rm dev uv run pytest tests/test_sweep_runner.py -x -k "sensitivity"` | Wave 0 |
| D-10 | ThresholdPolicy abstraction | unit | `docker compose run --rm dev uv run pytest tests/test_threshold_policy.py -x -k "policy"` | Wave 0 |
| D-26 | Reusable sweep framework | unit | `docker compose run --rm dev uv run pytest tests/test_sweep_runner.py -x` | Wave 0 |
| D-25 | SQLite storage works correctly | unit | `docker compose run --rm dev uv run pytest tests/test_sweep_runner.py -x -k "sqlite"` | Wave 0 |

### Sampling Rate
- **Per task commit:** `docker compose run --rm dev uv run pytest tests/test_trie.py tests/test_threshold_policy.py -x`
- **Per wave merge:** `docker compose run --rm dev uv run pytest -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_threshold_policy.py` -- ThresholdPolicy unit tests (GlobalPolicy, backward compat, all 6 strategies)
- [ ] `tests/test_sweep_runner.py` -- Sweep framework tests (SQLite init, config generation, parallel execution, result retrieval)

## Sources

### Primary (HIGH confidence)
- `src/octonion/trie.py` -- Direct code inspection of current threshold usage (lines 98, 145, 176, 179, 262, 269, 400-404)
- `src/octonion/_octonion.py` -- Associator function implementation
- `src/octonion/_fano.py` -- Fano plane triples (7 quaternionic subalgebras)
- `scripts/trie_benchmark_utils.py` -- Existing benchmark utilities for feature loading and evaluation
- `scripts/run_trie_benchmarks_parallel.py` -- Existing ProcessPoolExecutor pattern
- Greg Egan, ["Peeling the Octonions"](https://www.gregegan.net/SCIENCE/Octonions/Octonions.html) -- Analytical mean associator norm: 147456/(42875*pi)
- [scipy.stats.wilcoxon documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html) -- Wilcoxon signed-rank test API
- [scipy.stats.friedmanchisquare documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html) -- Friedman test API
- [SQLite WAL mode documentation](https://www.sqlite.org/wal.html) -- WAL mode for concurrent access

### Secondary (MEDIUM confidence)
- John D. Cook, ["How close is octonion multiplication to being associative?"](https://www.johndcook.com/blog/2018/07/09/octonion-associator/) -- Numerical verification of Egan's result
- [ProcessPoolExecutor best practices](https://superfastpython.com/processpoolexecutor-best-practices/) -- Pickling, worker count, exception handling
- [tqdm with concurrent.futures](https://rednafi.com/python/tqdm-progressbar-with-concurrent-futures/) -- Progress bar integration pattern
- [SQLite concurrent writes](https://blog.skypilot.co/abusing-sqlite-to-handle-concurrency/) -- Timeout strategy for write contention
- Baez (2002), "The Octonions" -- G2 as automorphism group, Fano plane structure
- Salamon & Walpuski, ["Notes on the Octonions"](https://people.math.ethz.ch/~salamon/PREPRINTS/Octonions.pdf) -- G2 cross-sections, modern algebraic treatment

### Tertiary (LOW confidence)
- G2 associator bounds for practical threshold setting -- no direct literature found for this specific application; will need novel derivation per D-44
- Cayley-Dickson connection to threshold theory -- unclear if this adds insight; Claude's discretion per D-51

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All libraries verified installed in container, versions confirmed
- Architecture: HIGH -- ThresholdPolicy pattern is a clean refactor of existing code with clear insertion points
- Sweep infrastructure: HIGH -- ProcessPoolExecutor + SQLite + tqdm all verified; pattern well-established
- Statistical analysis: HIGH -- scipy provides all needed tests; Cohen's d is straightforward
- Theory/proofs: MEDIUM -- Egan's result is solid; Fano plane bound argument is plausible but unproven; G2 connection needs novel work
- Meta-trie convergence: LOW -- No precedent for self-referential trie optimization; convergence behavior unknown

**Research date:** 2026-03-29
**Valid until:** 2026-04-28 (stable dependencies, mathematical results are permanent)
