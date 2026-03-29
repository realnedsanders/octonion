---
phase: T1-benchmark-generalization
plan: 02
type: execute
wave: 2
depends_on: [T1-01]
files_modified:
  - scripts/run_trie_fashion_mnist.py
autonomous: true
requirements: [TRIE-01]

must_haves:
  truths:
    - "Fashion-MNIST accuracy is measured with CNN encoder for the trie and all baselines"
    - "Confusion matrix reveals which Fashion-MNIST classes the trie confuses"
    - "Learning curve shows trie accuracy vs training set size at 4+ points"
    - "CNN encoder head accuracy provides an upper bound on feature quality"
    - "All baselines (kNN k=1, k=5, RF, SVM, LR) run on the same 8D features"
  artifacts:
    - path: "scripts/run_trie_fashion_mnist.py"
      provides: "Fashion-MNIST benchmark script"
      min_lines: 150
    - path: "results/trie_benchmarks/fashion_mnist/results.json"
      provides: "Fashion-MNIST benchmark results (generated at runtime)"
      contains: "accuracy"
  key_links:
    - from: "scripts/run_trie_fashion_mnist.py"
      to: "scripts/trie_benchmark_utils.py"
      via: "import shared utilities"
      pattern: "from trie_benchmark_utils import"
    - from: "scripts/run_trie_fashion_mnist.py"
      to: "torchvision.datasets.FashionMNIST"
      via: "dataset loading"
      pattern: "FashionMNIST"
---

<objective>
Create the Fashion-MNIST benchmark script that trains a small CNN encoder (same architecture as MNIST), extracts 8D features, and evaluates the octonionic trie plus all sklearn baselines.

Purpose: Fashion-MNIST is the closest analog to MNIST (same image format, same number of classes) but harder. This tests whether the trie generalizes beyond digit recognition.

Output: `scripts/run_trie_fashion_mnist.py` script that produces accuracy comparison, confusion matrix, learning curves.
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
@scripts/run_trie_mnist.py
@scripts/run_trie_mnist_experiments.py

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

<!-- Existing MNIST CNN encoder pattern (reuse for Fashion-MNIST per user decision) -->
From scripts/run_trie_mnist_experiments.py:
```python
class SmallCNN(nn.Module):
    def __init__(self):
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
        )
        self.classifier = nn.Linear(128, 10)
    def forward(self, x): return self.classifier(self.features(x))
    def extract(self, x): return self.features(x)
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create Fashion-MNIST benchmark script</name>
  <files>scripts/run_trie_fashion_mnist.py</files>
  <action>
Create `scripts/run_trie_fashion_mnist.py` following the structure of `run_trie_mnist.py` but extended with sklearn baselines and CNN encoder. Per user decision: use same CNN architecture as MNIST encoder.

**CNN Encoder (per CONTEXT.md: same as MNIST):**
- SmallCNN: Conv2d(1,16,3,pad=1)->ReLU->MaxPool2d(2)->Conv2d(16,32,3,pad=1)->ReLU->MaxPool2d(2)->Flatten->Linear(32*7*7, 128)->ReLU->Linear(128, feature_dim)
- Train with Adam lr=1e-3 for 10 epochs (research recommends 10 for Fashion-MNIST)
- Feature dim = 8, final layer is `self.classifier = nn.Linear(feature_dim, 10)` for classification head evaluation
- Use `self.features` up to Linear(128, feature_dim) + ReLU for extraction, then add a final nn.Linear(feature_dim, 8) without ReLU to get 8D features. Actually simpler: features ends at an 8D output already (feature_dim=8). Then `self.classifier = nn.Linear(8, 10)`.

**Pipeline:**
1. CLI args: --n-train (default 10000), --n-test (default 2000), --epochs (trie epochs, default 3), --cnn-epochs (default 10), --output-dir (default results/trie_benchmarks/fashion_mnist), --seed (default 42)
2. Load FashionMNIST from torchvision (standard train/test split)
3. Subsample train/test per CLI args using seeded randperm
4. Train CNN encoder on full training set (60K) using DataLoader(batch_size=256, shuffle=True). Report CNN head test accuracy as upper bound.
5. Extract 8D features from trained CNN. Convert to float64. Normalize to unit octonions.
6. Run all baselines via `run_sklearn_baselines(train_x.numpy(), train_y.numpy(), test_x.numpy(), test_y.numpy())`
7. Run trie via `run_trie_classifier(train_x, train_y, test_x, test_y, epochs=args.epochs)`
8. Generate learning curves at fractions [0.1, 0.25, 0.5, 1.0] for trie and kNN k=5 (run trie + knn on subsets of training data, always test on full test set)
9. Plot confusion matrix for trie via `plot_confusion_matrix(...)`, save to output_dir/confusion_matrix.png
10. Plot learning curves via `plot_learning_curves(...)`, save to output_dir/learning_curve.png
11. Print comparison table to stdout
12. Save all results to output_dir/results.json via `save_results(...)`

**Fashion-MNIST class names:**
```python
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
```

**Import pattern for trie_benchmark_utils:**
```python
import sys
sys.path.insert(0, str(Path(__file__).parent))
from trie_benchmark_utils import (run_sklearn_baselines, run_trie_classifier,
    compute_per_class_accuracy, plot_confusion_matrix, plot_learning_curves, save_results)
```

**Key details:**
- All classifiers (including sklearn) operate on the SAME 8D CNN features (per user decision)
- CNN head evaluated on raw images (not 8D features) as upper bound
- Features converted to float64 before trie (matching MNIST script pattern)
- Use torchvision.datasets.FashionMNIST with download=True, data_dir=tempfile.mkdtemp()
  </action>
  <verify>
    <automated>docker compose run --rm dev uv run python scripts/run_trie_fashion_mnist.py --n-train 500 --n-test 100 --cnn-epochs 2 --epochs 1</automated>
  </verify>
  <done>Script runs end-to-end with small data, produces results.json with accuracy for trie and all baselines, and generates confusion_matrix.png and learning_curve.png</done>
</task>

</tasks>

<verification>
- `docker compose run --rm dev uv run python scripts/run_trie_fashion_mnist.py --n-train 500 --n-test 100 --cnn-epochs 2 --epochs 1` completes without error
- `results/trie_benchmarks/fashion_mnist/results.json` contains accuracy values for trie, knn_k1, knn_k5, rf, svm_rbf, logreg, cnn_head
- `results/trie_benchmarks/fashion_mnist/confusion_matrix.png` exists
- `results/trie_benchmarks/fashion_mnist/learning_curve.png` exists
</verification>

<success_criteria>
Fashion-MNIST benchmark script produces accuracy comparison, confusion matrix, and learning curves. CNN encoder uses same architecture as MNIST encoder per user decision. All baselines evaluated on same 8D features.
</success_criteria>

<output>
After completion, create `.planning/phases/T1-benchmark-generalization/T1-02-SUMMARY.md`
</output>
