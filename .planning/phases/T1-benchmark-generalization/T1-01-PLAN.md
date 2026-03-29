---
phase: T1-benchmark-generalization
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - pyproject.toml
  - scripts/trie_benchmark_utils.py
  - tests/test_trie_benchmarks.py
autonomous: true
requirements: [TRIE-01]

must_haves:
  truths:
    - "scikit-learn is installed and importable in the container"
    - "A shared baseline runner function exists that runs kNN, RF, SVM, LR on any 8D features"
    - "A shared result schema and analysis utility exists for confusion matrices and per-class metrics"
    - "Tests validate the shared utilities produce correct output shapes and fields"
  artifacts:
    - path: "scripts/trie_benchmark_utils.py"
      provides: "Shared sklearn baselines, metrics, plotting, result I/O for all benchmark scripts"
      exports: ["run_sklearn_baselines", "run_trie_classifier", "plot_confusion_matrix", "plot_learning_curves", "save_results", "compute_per_class_accuracy"]
    - path: "tests/test_trie_benchmarks.py"
      provides: "Unit tests for the shared benchmark utilities"
      min_lines: 50
  key_links:
    - from: "scripts/trie_benchmark_utils.py"
      to: "sklearn.neighbors, sklearn.ensemble, sklearn.svm, sklearn.linear_model, sklearn.metrics"
      via: "import"
      pattern: "from sklearn"
    - from: "scripts/trie_benchmark_utils.py"
      to: "src/octonion/trie.py"
      via: "import OctonionTrie"
      pattern: "from octonion.trie import OctonionTrie"
---

<objective>
Install scikit-learn dependency and create shared benchmark utilities used by all three benchmark scripts (Fashion-MNIST, CIFAR-10, text).

Purpose: Avoid code duplication across benchmark scripts. All baselines (kNN k=1, k=5, RF, SVM, LR) and analysis (confusion matrix, per-class accuracy, learning curves, result JSON) use the same utility module.

Output: `scripts/trie_benchmark_utils.py` with tested shared functions, scikit-learn available in container.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/T1-benchmark-generalization/T1-CONTEXT.md
@.planning/phases/T1-benchmark-generalization/T1-RESEARCH.md
@scripts/run_trie_mnist.py

<interfaces>
<!-- Key types and contracts from trie module that benchmark utilities need -->

From src/octonion/trie.py:
```python
class TrieNode:
    routing_key: torch.Tensor      # [8] unit octonion
    content: torch.Tensor           # [8] accumulated content
    category_counts: dict[int, int] # {category: count}
    children: list[TrieNode]

    @property
    def dominant_category(self) -> int | None: ...
    @property
    def is_leaf(self) -> bool: ...

class OctonionTrie:
    def __init__(self, associator_threshold=0.3, similarity_threshold=0.1,
                 max_depth=15, seed=42): ...
    def insert(self, x: torch.Tensor, category: int | None = None) -> TrieNode: ...
    def query(self, x: torch.Tensor) -> TrieNode: ...
    def consolidate(self) -> None: ...
    def stats(self) -> dict: ...  # returns {n_nodes, n_leaves, max_depth, avg_depth, ...}
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Install scikit-learn and create shared benchmark utilities</name>
  <files>pyproject.toml, scripts/trie_benchmark_utils.py</files>
  <action>
1. Install scikit-learn:
   `docker compose run --rm dev uv add scikit-learn`

2. Create `scripts/trie_benchmark_utils.py` with the following functions:

**run_sklearn_baselines(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> dict:**
- Run kNN (k=1), kNN (k=5), RandomForest (n_estimators=100, random_state=42), SVM (RBF kernel, random_state=42), LogisticRegression (max_iter=1000, random_state=42) on the given features.
- Return dict mapping method name to {"accuracy": float, "predictions": np.ndarray, "confusion_matrix": list[list[int]], "classification_report": dict}.
- Per user decision: all baselines use the SAME 8D features as the trie.

**run_trie_classifier(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor, epochs: int = 3, assoc_threshold: float = 0.3, seed: int = 42) -> dict:**
- Wraps OctonionTrie insert/query loop (follow pattern from run_trie_mnist.py trie_classify function).
- Returns {"accuracy": float, "predictions": list[int], "per_class": dict, "trie_stats": dict, "train_time": float, "test_time": float}.

**compute_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:**
- Returns {class_name: {"correct": int, "total": int, "accuracy": float}} for each class.

**plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):**
- Use sklearn.metrics.ConfusionMatrixDisplay with cmap="Blues", values_format="d".
- Save to save_path at dpi=150. Close figure after save.

**plot_learning_curves(curves: dict, title: str, save_path: str):**
- Plot accuracy vs training set size for multiple methods.
- curves format: {method_name: [{"n_train": int, "accuracy": float}, ...]}.

**save_results(results: dict, output_path: Path):**
- JSON-serialize results with indent=2.
- Handle numpy arrays and torch tensors by converting to lists/floats.

All functions must have type hints. Use Python 3.12+ syntax.
  </action>
  <verify>
    <automated>docker compose run --rm dev uv run python -c "from sklearn.neighbors import KNeighborsClassifier; from sklearn.metrics import confusion_matrix; print('sklearn OK')"</automated>
  </verify>
  <done>scikit-learn importable, scripts/trie_benchmark_utils.py exists with all 6 functions exported</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Test shared benchmark utilities</name>
  <files>tests/test_trie_benchmarks.py</files>
  <behavior>
    - Test 1: run_sklearn_baselines returns dict with keys knn_k1, knn_k5, rf, svm_rbf, logreg, each containing accuracy (float 0-1), predictions (array), confusion_matrix (list)
    - Test 2: run_trie_classifier returns dict with accuracy (float 0-1), predictions (list), trie_stats (dict with n_nodes key)
    - Test 3: compute_per_class_accuracy returns correct counts for known input
    - Test 4: save_results handles numpy arrays and torch tensors without error
    - Test 5: plot_confusion_matrix creates a .png file at the given path
  </behavior>
  <action>
Create `tests/test_trie_benchmarks.py` with pytest tests for the shared utilities. Use small synthetic data (50 train, 20 test, 8 features, 3 classes) to keep tests fast. Generate random unit-norm 8D vectors and random labels for tests. Use tmp_path fixture for file output tests.

Import from scripts path using sys.path insertion pattern (scripts/ is not a package):
```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from trie_benchmark_utils import run_sklearn_baselines, run_trie_classifier, ...
```

Tests validate output STRUCTURE, not accuracy values (random data produces random accuracy).
  </action>
  <verify>
    <automated>docker compose run --rm dev uv run pytest tests/test_trie_benchmarks.py -x -v</automated>
  </verify>
  <done>All 5 tests pass, validating the shared benchmark utilities produce correct output schemas</done>
</task>

</tasks>

<verification>
- `docker compose run --rm dev uv run python -c "import sklearn; print(sklearn.__version__)"` prints a version
- `docker compose run --rm dev uv run pytest tests/test_trie_benchmarks.py -x -v` all pass
- `scripts/trie_benchmark_utils.py` contains all 6 exported functions with type hints
</verification>

<success_criteria>
scikit-learn installed, shared benchmark utilities exist and are tested, ready for all three benchmark scripts to import.
</success_criteria>

<output>
After completion, create `.planning/phases/T1-benchmark-generalization/T1-01-SUMMARY.md`
</output>
