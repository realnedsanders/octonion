---
phase: T1-benchmark-generalization
plan: 04
type: execute
wave: 2
depends_on: [T1-01]
files_modified:
  - scripts/run_trie_text.py
autonomous: true
requirements: [TRIE-01]

must_haves:
  truths:
    - "20 Newsgroups text classification runs with TF-IDF + TruncatedSVD to 8D (fully gradient-free)"
    - "Both full 20-class and 4-5 class subset are evaluated"
    - "Explained variance ratio from TruncatedSVD is reported"
    - "All baselines run on the same 8D features"
    - "Confusion matrix and per-class accuracy produced for both class configurations"
  artifacts:
    - path: "scripts/run_trie_text.py"
      provides: "Text classification benchmark script (fully gradient-free)"
      min_lines: 150
    - path: "results/trie_benchmarks/text/results.json"
      provides: "Text benchmark results (generated at runtime)"
      contains: "accuracy"
  key_links:
    - from: "scripts/run_trie_text.py"
      to: "scripts/trie_benchmark_utils.py"
      via: "import shared utilities"
      pattern: "from trie_benchmark_utils import"
    - from: "scripts/run_trie_text.py"
      to: "sklearn.datasets.fetch_20newsgroups"
      via: "dataset loading"
      pattern: "fetch_20newsgroups"
    - from: "scripts/run_trie_text.py"
      to: "sklearn.feature_extraction.text.TfidfVectorizer"
      via: "text vectorization"
      pattern: "TfidfVectorizer"
    - from: "scripts/run_trie_text.py"
      to: "sklearn.decomposition.TruncatedSVD"
      via: "dimensionality reduction (NOT PCA)"
      pattern: "TruncatedSVD"
---

<objective>
Create the text classification benchmark using 20 Newsgroups with a fully gradient-free pipeline: TF-IDF vectorization, TruncatedSVD to 8D, normalize to unit octonions, classify with trie and baselines.

Purpose: This is the strongest test of the algebraic encoder thesis -- zero gradient computation end-to-end. Per user decision, no neural text encoder. The text benchmark also tests the trie on a 20-class problem (vs 10 for image benchmarks) and on fundamentally different data (sparse high-dimensional text vs dense image features).

Output: `scripts/run_trie_text.py` script with both 20-class and 4-5 class subset results.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/phases/T1-benchmark-generalization/T1-CONTEXT.md
@.planning/phases/T1-benchmark-generalization/T1-RESEARCH.md
@.planning/phases/T1-benchmark-generalization/T1-01-SUMMARY.md

<interfaces>
<!-- From T1-01: shared benchmark utilities -->
From scripts/trie_benchmark_utils.py:
```python
def run_sklearn_baselines(train_x: np.ndarray, train_y: np.ndarray,
                          test_x: np.ndarray, test_y: np.ndarray) -> dict: ...
def run_trie_classifier(train_x: torch.Tensor, train_y: torch.Tensor,
                        test_x: torch.Tensor, test_y: torch.Tensor,
                        epochs: int = 3, assoc_threshold: float = 0.3,
                        seed: int = 42) -> dict: ...
def compute_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                               class_names: list[str]) -> dict: ...
def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path): ...
def plot_learning_curves(curves: dict, title: str, save_path: str): ...
def save_results(results: dict, output_path: Path): ...
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create 20 Newsgroups text classification benchmark script</name>
  <files>scripts/run_trie_text.py</files>
  <action>
Create `scripts/run_trie_text.py` implementing the fully gradient-free text classification pipeline.

**Data Loading:**
- Use `sklearn.datasets.fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))` for training and `subset="test"` for test. The `remove` parameter prevents metadata leakage (per research Pitfall 5).
- For the 4-class subset, use `categories=["comp.graphics", "rec.sport.baseball", "sci.med", "talk.politics.guns"]` parameter in fetch_20newsgroups. These are well-separated categories.
- No subsampling needed for 20 Newsgroups -- it's only ~11K train / ~7.5K test, manageable.

**TF-IDF Vectorization (per research, Claude's discretion):**
```python
TfidfVectorizer(
    max_features=10000,
    sublinear_tf=True,
    max_df=0.5,
    min_df=5,
    stop_words="english",
)
```

**Dimensionality Reduction:**
- Use `sklearn.decomposition.TruncatedSVD(n_components=8, random_state=42)` -- NOT PCA. PCA would densify the sparse TF-IDF matrix, potentially causing OOM. TruncatedSVD works directly on sparse CSR matrices.
- Fit on training data, transform both train and test.
- Report explained variance ratio: `svd.explained_variance_ratio_.sum()`

**Normalization to unit octonions:**
```python
norms = np.linalg.norm(reduced, axis=1, keepdims=True)
norms = np.clip(norms, 1e-10, None)
octonionic = reduced / norms
```
Then convert to torch.float64 tensors for trie.

**Pipeline (run twice: full 20-class and 4-class subset):**
1. Load data (full or subset categories)
2. TF-IDF vectorize (fit on train, transform both)
3. TruncatedSVD to 8D (fit on train, transform both)
4. Report explained variance
5. Normalize to unit octonions
6. Run all sklearn baselines on same 8D features
7. Run trie classifier
8. Generate learning curves at fractions [0.1, 0.25, 0.5, 1.0]
9. Per-class accuracy and confusion matrix
10. Print comparison table

**CLI args:**
- --mode: choices=[full, subset, both], default=both
- --epochs: trie epochs, default 3
- --n-components: SVD components, default 8
- --output-dir: default results/trie_benchmarks/text
- --seed: default 42

**Output structure:**
```json
{
  "full_20class": {
    "n_train": 11314, "n_test": 7532,
    "n_classes": 20,
    "explained_variance_8d": 0.15,
    "trie": {"accuracy": ..., ...},
    "baselines": {"knn_k1": ..., "knn_k5": ..., "rf": ..., "svm_rbf": ..., "logreg": ...},
    "learning_curve": [...],
    "class_names": [...]
  },
  "subset_4class": {
    "n_train": ..., "n_test": ...,
    "n_classes": 4,
    "explained_variance_8d": ...,
    ...same structure...
  }
}
```

**Note on SVM timing (Pitfall 4):** Full 20 Newsgroups has ~11K training samples. With 8D features, RBF SVM should be tractable (O(n^2) on 11K with 8D is fine). No subsampling needed. But add timing print so we know.

**Note on this being a no-neural-encoder pipeline (per user decision):**
- There is NO CNN head upper bound for text (unlike image benchmarks)
- The upper bound is instead the best sklearn classifier on full TF-IDF features (pre-SVD). Compute this: run LogisticRegression on the full TF-IDF matrix (before SVD) and report as "full_tfidf_logreg" -- this shows how much information the 8D bottleneck loses.

**Confusion matrix plots:**
- Full 20-class: figsize=(14, 12) to fit 20 class names. Use shortened names if needed.
- 4-class subset: figsize=(8, 6).
- Save as output_dir/confusion_matrix_full.png and output_dir/confusion_matrix_subset.png
  </action>
  <verify>
    <automated>docker compose run --rm dev uv run python scripts/run_trie_text.py --mode subset --epochs 1</automated>
  </verify>
  <done>Script runs end-to-end on 4-class subset. When run with --mode both, produces results for both 20-class and 4-class configurations. Reports explained variance, comparison tables, confusion matrices, and learning curves. Pipeline is fully gradient-free (no neural encoder).</done>
</task>

</tasks>

<verification>
- `docker compose run --rm dev uv run python scripts/run_trie_text.py --mode subset --epochs 1` completes in < 5 minutes
- results.json contains explained_variance_8d, trie accuracy, and all baseline accuracies
- Confusion matrix PNG exists
- No neural network is used anywhere in the pipeline (verify no `nn.Module` imports)
</verification>

<success_criteria>
Text benchmark runs with fully gradient-free pipeline (TF-IDF + TruncatedSVD, no neural encoder) per user decision. Both 20-class and 4-class subset evaluated. Explained variance reported. All baselines on same 8D features. Full TF-IDF LogReg upper bound included.
</success_criteria>

<output>
After completion, create `.planning/phases/T1-benchmark-generalization/T1-04-SUMMARY.md`
</output>
