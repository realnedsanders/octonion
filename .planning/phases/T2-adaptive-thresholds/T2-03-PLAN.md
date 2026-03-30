---
phase: T2-adaptive-thresholds
plan: 03
type: execute
wave: 2
depends_on: ["T2-01"]
files_modified:
  - scripts/sweep/sweep_runner.py
  - tests/test_sweep_runner.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "Parallel sweep framework runs N configs across M workers writing results to SQLite"
    - "SQLite database uses WAL mode and handles concurrent writes from 24 workers"
    - "Sweep configs are generated from parameter grids including policy type and hyperparameters"
    - "Results include epoch-by-epoch accuracy and trie structure metrics"
    - "Framework is reusable by T4 and T6 phases"
  artifacts:
    - path: "scripts/sweep/sweep_runner.py"
      provides: "Reusable parallel sweep framework with SQLite storage"
      contains: "class SweepRunner"
    - path: "tests/test_sweep_runner.py"
      provides: "Unit tests for sweep infrastructure"
      contains: "def test_sqlite_init"
  key_links:
    - from: "scripts/sweep/sweep_runner.py"
      to: "sqlite3"
      via: "WAL mode connection with 30s timeout"
      pattern: "PRAGMA journal_mode=WAL"
    - from: "scripts/sweep/sweep_runner.py"
      to: "concurrent.futures.ProcessPoolExecutor"
      via: "Parallel worker execution"
      pattern: "ProcessPoolExecutor"
---

<objective>
Build the reusable parallel sweep framework with SQLite storage and tqdm progress.

Purpose: Per D-26, this framework must be generic (param grid + worker function + SQLite output) and reusable by T4 and T6. Per D-24, 24-worker ProcessPoolExecutor. Per D-25, SQLite with epoch-by-epoch tracking. Per D-28, tqdm progress bar.

Output: sweep_runner.py with SweepRunner class, SQLite schema, config generation, parallel execution. Tested with unit tests.
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

@scripts/run_trie_benchmarks_parallel.py
@scripts/trie_benchmark_utils.py
@src/octonion/trie.py

<interfaces>
<!-- From T2-01: ThresholdPolicy classes available in octonion.trie -->
from octonion.trie import (
    OctonionTrie, ThresholdPolicy, GlobalPolicy,
    PerNodeEMAPolicy, PerNodeMeanStdPolicy, DepthPolicy,
    AlgebraicPurityPolicy, MetaTriePolicy, HybridPolicy,
)
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Parallel sweep framework with SQLite storage</name>
  <files>scripts/sweep/sweep_runner.py</files>
  <read_first>
    - scripts/run_trie_benchmarks_parallel.py (existing ProcessPoolExecutor pattern)
    - src/octonion/trie.py (ThresholdPolicy classes from T2-01, OctonionTrie API)
    - scripts/trie_benchmark_utils.py (run_trie_classifier pattern)
  </read_first>
  <action>
Create scripts/sweep/sweep_runner.py with the following components.

**SweepConfig dataclass** (per D-20):
```python
@dataclass
class SweepConfig:
    config_id: int
    benchmark: str
    policy_type: str  # "global", "ema", "mean_std", "depth", "purity", "meta_trie", "hybrid"
    assoc_threshold: float  # base threshold
    sim_threshold: float
    min_share: float
    min_count: int
    noise: float
    epochs: int
    seed: int
    policy_params: str = "{}"  # JSON string for policy-specific hyperparams
```

**SQLite schema** (per D-25):
```sql
CREATE TABLE IF NOT EXISTS sweep_results (
    config_id INTEGER NOT NULL,
    benchmark TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    policy_type TEXT NOT NULL,
    assoc_threshold REAL,
    sim_threshold REAL,
    min_share REAL,
    min_count INTEGER,
    noise REAL,
    accuracy REAL,
    n_nodes INTEGER,
    n_leaves INTEGER,
    max_depth INTEGER,
    rumination_rejections INTEGER,
    consolidation_merges INTEGER,
    branching_factor_mean REAL,
    branching_factor_std REAL,
    train_time REAL,
    test_time REAL,
    policy_params TEXT DEFAULT '{}',
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (config_id, benchmark, epoch, seed)
);
CREATE INDEX IF NOT EXISTS idx_benchmark ON sweep_results(benchmark);
CREATE INDEX IF NOT EXISTS idx_policy ON sweep_results(policy_type);
CREATE INDEX IF NOT EXISTS idx_accuracy ON sweep_results(accuracy DESC);
```

**SweepRunner class** (per D-26):
```python
class SweepRunner:
    def __init__(self, db_path: str, n_workers: int = 24):
        self.db_path = db_path
        self.n_workers = n_workers
        self._init_db()

    def _init_db(self) -> None:
        # Create tables with WAL mode

    def run(self, configs: list[SweepConfig], features_dir: str) -> None:
        # Per D-24: ProcessPoolExecutor with n_workers
        # Per D-28: tqdm progress bar
        # Per D-27: Run all configs to completion (no early stopping)

    @staticmethod
    def _worker(config: SweepConfig, features_dir: str, db_path: str) -> dict:
        # Load cached features from features_dir/{benchmark}_features.pt
        # Create ThresholdPolicy from config.policy_type and config.policy_params
        # Create OctonionTrie with policy
        # Train for config.epochs, evaluating after each epoch (D-25 epoch-by-epoch)
        # Write results to SQLite (each worker opens its own connection, timeout=30s)
        # Return summary dict

    def _write_result(self, config: SweepConfig, epoch: int, metrics: dict) -> None:
        # Write single epoch result to SQLite
```

**Config generation helpers**:
```python
def generate_global_sweep_configs(
    benchmarks: list[str],
    seed: int = 42,
) -> list[SweepConfig]:
    """Generate 4D sweep grid per D-20."""
    # Per D-20: assoc threshold 0.001-2.0 log-spaced
    # Use np.unique(np.sort(np.concatenate([np.geomspace(0.001, 2.0, 15), np.linspace(0.05, 1.0, 10)])))
    # Per research pitfall 7: combine geomspace + linspace for critical region coverage
    assoc_values = np.unique(np.sort(np.concatenate([
        np.geomspace(0.001, 2.0, 15),
        np.linspace(0.05, 1.0, 10),
    ])))
    # Similarity threshold: [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    sim_values = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    # Consolidation: 5 configs per D-20
    consolidation_configs = [
        (0.01, 1), (0.03, 2), (0.05, 3), (0.10, 5), (0.00, 0),
    ]
    # Noise: per D-05
    noise_values = [0.0, 0.01, 0.05, 0.1]
    # Epochs: per D-06
    epoch_values = [1, 3, 5]

    # For initial 4D sweep per D-22: reduced grid
    # Full grid would be ~24*8*5*4*3 = 11520 -- too large
    # Reduced: fix epochs=3, fix consolidation=(0.05,3), sweep assoc x sim x noise
    # That's ~24*8*4 = 768 per benchmark for the core 3D sweep
    # Then 1D sweeps for consolidation and epochs

def generate_adaptive_sweep_configs(
    policy_type: str,
    benchmarks: list[str],
    seed: int = 42,
) -> list[SweepConfig]:
    """Generate sweep configs for an adaptive strategy per D-29."""
    # Sweep policy-specific hyperparameters alongside base threshold
```

**Worker function details**:
The worker function must:
1. Load features from `{features_dir}/{benchmark}_10k_features.pt` or `{benchmark}_features.pt`
2. Construct the appropriate ThresholdPolicy from config.policy_type using json.loads(config.policy_params)
3. Create OctonionTrie(policy=policy, training_noise=config.noise, seed=config.seed)
4. For each epoch (per D-25 epoch-by-epoch tracking):
   - Insert all training samples
   - Evaluate on test set
   - Record accuracy, trie.stats(), train_time, test_time
   - Consolidate every 2 epochs
5. Write ALL epoch results to SQLite in a single transaction (batch write to reduce contention per research pitfall 1)
6. Each worker creates its own sqlite3.connect(db_path, timeout=30.0) per research pitfall 1

**Branching factor computation**: Add branching_factor_mean and branching_factor_std to stats by walking the trie and computing mean/std of len(node.children) for non-leaf nodes.

**Critical**: Workers must NOT share SQLite connections. Each process creates its own. Use timeout=30.0 for write contention handling.

**Critical**: Trie objects are NOT passed between processes. Only primitive config values are serialized. Each worker constructs its own trie.
  </action>
  <verify>
    <automated>docker compose run --rm dev uv run python -c "from scripts.sweep.sweep_runner import SweepRunner, SweepConfig, generate_global_sweep_configs; print('Import OK')"</automated>
  </verify>
  <acceptance_criteria>
    - scripts/sweep/sweep_runner.py exists
    - sweep_runner.py contains `class SweepRunner`
    - sweep_runner.py contains `class SweepConfig`
    - sweep_runner.py contains `PRAGMA journal_mode=WAL`
    - sweep_runner.py contains `ProcessPoolExecutor`
    - sweep_runner.py contains `tqdm`
    - sweep_runner.py contains `timeout=30`
    - sweep_runner.py contains `def generate_global_sweep_configs`
    - sweep_runner.py contains `def generate_adaptive_sweep_configs`
    - sweep_runner.py contains `CREATE TABLE IF NOT EXISTS sweep_results`
    - sweep_runner.py contains `PRIMARY KEY (config_id, benchmark, epoch, seed)`
  </acceptance_criteria>
  <done>Parallel sweep framework with SQLite storage, config generation, worker function, and tqdm progress bar fully implemented</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Sweep framework unit tests</name>
  <files>tests/test_sweep_runner.py</files>
  <read_first>
    - scripts/sweep/sweep_runner.py (SweepRunner, SweepConfig APIs from Task 1)
    - src/octonion/trie.py (ThresholdPolicy classes)
  </read_first>
  <behavior>
    - Test 1: _init_db creates SQLite database with sweep_results table and all expected columns
    - Test 2: _init_db sets WAL journal mode
    - Test 3: generate_global_sweep_configs produces configs with all 5 benchmarks and expected param ranges
    - Test 4: generate_global_sweep_configs produces configs where assoc_threshold values include both geomspace and linspace points in 0.05-1.0 range
    - Test 5: SweepConfig is picklable (required for ProcessPoolExecutor)
    - Test 6: Worker function writes epoch-by-epoch results to SQLite (test with 1 config, 2 epochs on tiny synthetic data)
    - Test 7: Multiple workers writing to same database don't deadlock (3 workers, 3 configs each, all complete)
    - Test 8: Results can be queried by benchmark and policy_type after sweep completes
  </behavior>
  <action>
Create tests/test_sweep_runner.py.

For tests requiring trie execution, create tiny synthetic data (7 categories, 10 samples each) and save as a temporary .pt file to simulate cached features. Use pytest tmp_path fixture for temporary SQLite databases.

Test 6 (worker writes epochs): Create 1 SweepConfig with epochs=2, run worker directly (not via ProcessPoolExecutor), verify SQLite has 2 rows for that config_id.

Test 7 (concurrent writes): Use ProcessPoolExecutor(max_workers=3) with 3 configs, verify all 3 complete without "database is locked" errors.

Import path: `sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))` then `from sweep.sweep_runner import SweepRunner, SweepConfig, generate_global_sweep_configs`.
  </action>
  <verify>
    <automated>docker compose run --rm dev uv run pytest tests/test_sweep_runner.py -x -v</automated>
  </verify>
  <acceptance_criteria>
    - tests/test_sweep_runner.py exists
    - tests/test_sweep_runner.py contains `def test_sqlite_init`
    - tests/test_sweep_runner.py contains `def test_wal_mode`
    - tests/test_sweep_runner.py contains `def test_config_picklable`
    - tests/test_sweep_runner.py contains `def test_concurrent_writes`
    - All 8 tests pass
  </acceptance_criteria>
  <done>8 sweep framework tests pass, verifying SQLite schema, WAL mode, config generation, pickling, concurrent writes, and epoch-by-epoch storage</done>
</task>

</tasks>

<verification>
docker compose run --rm dev uv run pytest tests/test_sweep_runner.py -x -v
</verification>

<success_criteria>
- SweepRunner class handles parallel execution with SQLite storage
- Config generation produces proper parameter grids
- Concurrent SQLite writes work without deadlocks
- Framework is generic enough for T4/T6 reuse (parameterized worker function)
</success_criteria>

<output>
After completion, create `.planning/phases/T2-adaptive-thresholds/T2-03-SUMMARY.md`
</output>
