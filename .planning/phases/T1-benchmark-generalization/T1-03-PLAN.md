---
phase: T1-benchmark-generalization
plan: 03
type: execute
wave: 2
depends_on: [T1-01]
files_modified:
  - scripts/run_trie_cifar10.py
autonomous: true
requirements: [TRIE-01]

must_haves:
  truths:
    - "CIFAR-10 accuracy is measured with three CNN encoder sizes (2-layer, 4-layer, ResNet-8)"
    - "Encoder capacity vs trie accuracy is reported to show how encoder quality affects the trie"
    - "All baselines run on the same 8D features for each encoder"
    - "Confusion matrix and per-class accuracy reveal which CIFAR-10 classes are hardest"
    - "Learning curve shows trie accuracy vs training set size"
  artifacts:
    - path: "scripts/run_trie_cifar10.py"
      provides: "CIFAR-10 multi-encoder benchmark script"
      min_lines: 250
    - path: "results/trie_benchmarks/cifar10/results.json"
      provides: "CIFAR-10 benchmark results (generated at runtime)"
      contains: "accuracy"
  key_links:
    - from: "scripts/run_trie_cifar10.py"
      to: "scripts/trie_benchmark_utils.py"
      via: "import shared utilities"
      pattern: "from trie_benchmark_utils import"
    - from: "scripts/run_trie_cifar10.py"
      to: "torchvision.datasets.CIFAR10"
      via: "dataset loading"
      pattern: "CIFAR10"
---

<objective>
Create the CIFAR-10 benchmark script with three CNN encoder architectures (2-layer, 4-layer, ResNet-8-scale) to test how encoder capacity affects trie classification on color images.

Purpose: CIFAR-10 is significantly harder than MNIST/Fashion-MNIST (color, complex backgrounds). Per user decision, try multiple encoder sizes and report how encoder capacity affects trie accuracy. This is the key test of whether the trie can work with more complex visual data.

Output: `scripts/run_trie_cifar10.py` script comparing three encoder sizes, with accuracy tables, confusion matrices, and encoder comparison chart.
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
  <name>Task 1: Implement three CIFAR-10 CNN encoder architectures</name>
  <files>scripts/run_trie_cifar10.py</files>
  <action>
Create `scripts/run_trie_cifar10.py` with three CNN encoder classes and the full benchmark pipeline.

**Encoder 1: CIFAR_CNN_2Layer (minimal baseline)**
```python
class CIFAR_CNN_2Layer(nn.Module):
    def __init__(self, feature_dim=8):
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, feature_dim),
        )
        self.classifier = nn.Linear(feature_dim, 10)
```

**Encoder 2: CIFAR_CNN_4Layer**
```python
class CIFAR_CNN_4Layer(nn.Module):
    def __init__(self, feature_dim=8):
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, feature_dim),
        )
        self.classifier = nn.Linear(feature_dim, 10)
```

**Encoder 3: CIFAR_CNN_ResNet8 (simplified ResNet with ~8 conv layers)**
```python
class ResidualBlock(nn.Module):
    """Standard residual block with optional downsampling."""
    def __init__(self, in_ch, out_ch, stride=1):
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)
        ) if stride != 1 or in_ch != out_ch else nn.Identity()
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))

class CIFAR_CNN_ResNet8(nn.Module):
    def __init__(self, feature_dim=8):
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.block1 = ResidualBlock(16, 16)
        self.block2 = ResidualBlock(16, 32, stride=2)
        self.block3 = ResidualBlock(32, 64, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc_features = nn.Linear(64, feature_dim)
        self.classifier = nn.Linear(feature_dim, 10)
    def features_forward(self, x):
        x = self.conv1(x); x = self.block1(x); x = self.block2(x); x = self.block3(x)
        return self.fc_features(self.flatten(self.pool(x)))
    def forward(self, x): return self.classifier(self.features_forward(x))
    def extract(self, x): return self.features_forward(x)
```

All three encoders must have both `forward(x)` (returns logits) and `extract(x)` (returns feature_dim-D features) methods.

**Training details per encoder:**
- 2-layer: 20 epochs, Adam lr=1e-3
- 4-layer: 30 epochs, Adam lr=1e-3
- ResNet-8: 50 epochs, Adam lr=1e-3, cosine annealing LR schedule (`torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)`)
- All: batch_size=128, use standard CIFAR-10 augmentation for training (RandomHorizontalFlip, RandomCrop(32, padding=4)), no augmentation for test/feature extraction
- Train on FULL training set (50K), subsample only for trie/baseline evaluation

**Data pipeline:**
1. Load CIFAR-10 via torchvision.datasets.CIFAR10 (download=True)
2. Training transforms: Compose([RandomHorizontalFlip(), RandomCrop(32, padding=4), ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
3. Test transforms: Compose([ToTensor(), Normalize(...same means/stds...)])
4. Subsample n_train/n_test from test/eval sets using seeded randperm (but train CNN on full 50K)

**CLI args:**
- --encoder: choices=[2layer, 4layer, resnet8, all], default=all
- --n-train: default 10000
- --n-test: default 2000
- --epochs: trie epochs, default 3
- --output-dir: default results/trie_benchmarks/cifar10
- --seed: default 42

**Per-encoder evaluation:**
For each selected encoder:
1. Train CNN, report CNN head test accuracy (upper bound)
2. Extract 8D features, convert to float64, normalize to unit octonions
3. Run all sklearn baselines on same 8D features
4. Run trie classifier
5. Generate learning curves at fractions [0.1, 0.25, 0.5, 1.0]
6. Plot confusion matrix for trie: output_dir/{encoder}_confusion_matrix.png
7. Print comparison table for this encoder

**Encoder comparison (when --encoder=all):**
- Plot bar chart: encoder name on x-axis, trie accuracy and CNN head accuracy on y-axis, grouped bars. Save to output_dir/encoder_comparison.png
- This directly addresses the user decision: "Report how encoder capacity affects trie accuracy"

Save all results to output_dir/results.json with structure:
```json
{
  "encoders": {
    "2layer": {"cnn_head_accuracy": ..., "trie": {...}, "baselines": {...}, "learning_curve": [...]},
    "4layer": {...},
    "resnet8": {...}
  },
  "config": {"n_train": ..., "n_test": ..., ...}
}
```

**CIFAR-10 class names:**
```python
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
```
  </action>
  <verify>
    <automated>docker compose run --rm dev uv run python scripts/run_trie_cifar10.py --encoder 2layer --n-train 500 --n-test 100 --epochs 1</automated>
  </verify>
  <done>Script runs end-to-end with 2layer encoder on small data. When run with --encoder all, produces encoder comparison showing accuracy across all three architectures. results.json, confusion matrix PNGs, and encoder_comparison.png generated.</done>
</task>

</tasks>

<verification>
- `docker compose run --rm dev uv run python scripts/run_trie_cifar10.py --encoder 2layer --n-train 500 --n-test 100 --epochs 1` completes in < 5 minutes
- results.json contains cnn_head_accuracy and trie accuracy for the 2layer encoder
- Confusion matrix PNG is generated
- Script has all three encoder classes defined
</verification>

<success_criteria>
CIFAR-10 benchmark script implements three CNN encoder sizes per user decision, runs trie + all baselines on same 8D features, and produces encoder comparison chart showing how encoder capacity affects trie accuracy.
</success_criteria>

<output>
After completion, create `.planning/phases/T1-benchmark-generalization/T1-03-SUMMARY.md`
</output>
