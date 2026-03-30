---
phase: T2-adaptive-thresholds
plan: 02
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/sweep/cache_features.py
  - tests/test_cache_features.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "Pre-computed 8D unit octonion features exist on disk for all 5 T1 benchmarks"
    - "Cached features exactly reproduce T1 benchmark accuracies when used with GlobalPolicy(0.3)"
    - "Sweep workers can load cached features without GPU or encoder dependencies"
  artifacts:
    - path: "scripts/sweep/cache_features.py"
      provides: "Feature caching for MNIST, Fashion-MNIST, CIFAR-10, Text 4-class, Text 20-class"
      contains: "def cache_all_features"
    - path: "tests/test_cache_features.py"
      provides: "Validation that cached features match original pipeline"
      contains: "def test_cached_features_match_original"
  key_links:
    - from: "scripts/sweep/cache_features.py"
      to: "scripts/run_trie_mnist.py"
      via: "Same PCA pipeline for MNIST features"
      pattern: "PCA.*n_components.*8"
    - from: "scripts/sweep/cache_features.py"
      to: "results/T2/features/"
      via: "torch.save .pt files"
      pattern: "torch\\.save"
---

<objective>
Pre-compute and cache encoder features for all 5 T1 benchmarks as .pt files.

Purpose: Per D-23, sweep workers must load 8D features only -- no GPU, no encoder training. Feature caching is a prerequisite for all sweep experiments. Per D-07, all experiments run on all T1 benchmarks: MNIST, Fashion-MNIST, CIFAR-10 (ResNet-8), Text 4-class, Text 20-class.

Output: cache_features.py script that produces {benchmark}_features.pt files containing train_x, train_y, test_x, test_y as torch tensors.
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

@scripts/run_trie_mnist.py
@scripts/run_trie_fashion_mnist.py
@scripts/run_trie_cifar10.py
@scripts/trie_benchmark_utils.py

<interfaces>
<!-- From run_trie_mnist.py: PCA to 8D, unit normalize -->
<!-- From run_trie_fashion_mnist.py: CNN encoder to 8D, unit normalize -->
<!-- From run_trie_cifar10.py: 3 CNN encoders (2-layer, 4-layer, resnet8) to 8D, unit normalize -->
<!-- From run_trie_text.py: TF-IDF + TruncatedSVD to 8D, unit normalize -->
<!-- All benchmarks produce float64 tensors of shape [N, 8] with unit norm -->
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Feature caching script for all benchmarks</name>
  <files>scripts/sweep/cache_features.py</files>
  <read_first>
    - scripts/run_trie_mnist.py (PCA pipeline: flatten -> PCA(8) -> unit normalize)
    - scripts/run_trie_fashion_mnist.py (CNN encoder pipeline)
    - scripts/run_trie_cifar10.py (3 encoder sizes pipeline)
    - scripts/run_trie_text.py (TF-IDF + TruncatedSVD pipeline, both 4-class and 20-class)
    - scripts/trie_benchmark_utils.py (DATA_DIR constant, run_trie_classifier for validation)
  </read_first>
  <action>
Create scripts/sweep/__init__.py (empty) and scripts/sweep/cache_features.py.

The script must EXACTLY replicate each benchmark's data pipeline to produce identical features. Per D-23, workers load 8D features and only run the trie. No GPU needed for sweep.

**MNIST** (per D-07):
- Load MNIST from torchvision (data_dir=DATA_DIR)
- Flatten 28x28 to 784
- PCA to 8 dimensions (sklearn PCA, fit on train)
- Unit normalize each sample
- Full scale: n_train=60000, n_test=10000
- Also cache a 10K subset (n_train=10000, n_test=2000) per D-22

**Fashion-MNIST** (per D-07):
- Load Fashion-MNIST from torchvision
- Train SmallCNN encoder (same architecture as run_trie_fashion_mnist.py) for 5 epochs
- Extract 8D features, unit normalize
- Full scale: n_train=60000, n_test=10000
- Also cache 10K subset
- CNN training uses seed=42 for reproducibility

**CIFAR-10** (per D-07, ResNet-8 only):
- Load CIFAR-10 from torchvision
- Train ResNet-8 encoder for 50 epochs with cosine annealing (same as run_trie_cifar10.py)
- Extract 8D features, unit normalize
- Full scale: n_train=50000, n_test=10000
- Also cache 10K subset (n_train=10000, n_test=2000)

**Text 4-class** (per D-07):
- Load 20 Newsgroups with 4-class grouping (same as run_trie_text.py)
- TF-IDF vectorization -> TruncatedSVD to 8D -> unit normalize
- Full scale: all available samples
- Also cache subset

**Text 20-class** (per D-07):
- Load 20 Newsgroups full 20-class
- TF-IDF vectorization -> TruncatedSVD to 8D -> unit normalize
- Full scale: all available samples
- Also cache subset

**Output format**: Each benchmark saved as `results/T2/features/{benchmark}_features.pt` and `results/T2/features/{benchmark}_10k_features.pt` containing dict with keys:
- `train_x`: torch.Tensor float64 shape [N, 8] unit normalized
- `train_y`: torch.Tensor int64 shape [N]
- `test_x`: torch.Tensor float64 shape [M, 8] unit normalized
- `test_y`: torch.Tensor int64 shape [M]
- `class_names`: list[str]
- `benchmark`: str
- `n_train`: int
- `n_test`: int

**CLI interface**:
```
python scripts/sweep/cache_features.py --benchmarks all --output-dir results/T2/features
python scripts/sweep/cache_features.py --benchmarks mnist,fashion_mnist --subset-only
```

Options: `--benchmarks` (comma-separated or "all"), `--output-dir` (default results/T2/features), `--subset-only` (only 10K subsets, faster), `--full-only` (only full scale).

**Validation step**: After caching, run the trie with GlobalPolicy(0.3) on MNIST 10K subset and print accuracy. This must match T1 results (95.2-95.3% for full MNIST).

Per D-31, use seed=42 for all random operations.
  </action>
  <verify>
    <automated>docker compose run --rm dev uv run python scripts/sweep/cache_features.py --benchmarks mnist --subset-only --output-dir /tmp/test_cache && docker compose run --rm dev uv run python -c "import torch; d=torch.load('/tmp/test_cache/mnist_10k_features.pt', weights_only=False); print(f'shape={d[\"train_x\"].shape}, norm={d[\"train_x\"][0].norm():.4f}')"</automated>
  </verify>
  <acceptance_criteria>
    - scripts/sweep/cache_features.py exists
    - scripts/sweep/__init__.py exists
    - cache_features.py contains `def cache_all_features`
    - cache_features.py handles all 5 benchmarks: mnist, fashion_mnist, cifar10, text_4class, text_20class
    - cache_features.py contains `torch.save(`
    - cache_features.py contains `--benchmarks` argparse argument
    - Running with --benchmarks mnist --subset-only produces a .pt file with train_x shape [10000, 8]
    - train_x tensor has unit norm (norm of each row within 1e-6 of 1.0)
    - train_x dtype is torch.float64
  </acceptance_criteria>
  <done>Feature caching script produces .pt files for all 5 benchmarks at both full and 10K subset scales, features are unit-normalized float64 8D tensors, MNIST subset accuracy matches T1 baseline</done>
</task>

</tasks>

<verification>
docker compose run --rm dev uv run python scripts/sweep/cache_features.py --benchmarks mnist --subset-only --output-dir /tmp/test_cache
docker compose run --rm dev uv run python -c "
import torch
d = torch.load('/tmp/test_cache/mnist_10k_features.pt', weights_only=False)
assert d['train_x'].shape == (10000, 8), f'Wrong shape: {d[\"train_x\"].shape}'
assert d['train_x'].dtype == torch.float64, f'Wrong dtype: {d[\"train_x\"].dtype}'
norms = d['train_x'].norm(dim=1)
assert (norms - 1.0).abs().max() < 1e-6, f'Not unit normalized: max deviation {(norms-1.0).abs().max()}'
print('Feature cache validation PASSED')
"
</verification>

<success_criteria>
- cache_features.py script handles all 5 benchmarks
- Each .pt file contains train_x, train_y, test_x, test_y as torch tensors
- Features are float64, shape [N, 8], unit normalized
- MNIST cached features produce same trie accuracy as T1 (within 0.5pp)
</success_criteria>

<output>
After completion, create `.planning/phases/T2-adaptive-thresholds/T2-02-SUMMARY.md`
</output>
