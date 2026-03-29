# Phase T1: Benchmark Generalization - Research

**Researched:** 2026-03-29
**Domain:** Non-parametric classification benchmarking (Fashion-MNIST, CIFAR-10, text) with octonionic trie
**Confidence:** MEDIUM-HIGH

## Summary

This phase extends the octonionic trie's validated 95.2% MNIST result to three additional benchmarks: Fashion-MNIST (harder image classification), CIFAR-10 (color images requiring stronger encoders), and text classification via TF-IDF+PCA (fully gradient-free pipeline). The core architecture is settled: train a task-specific encoder, project features to 8D unit octonions, run the trie, compare against classical methods on the same features.

The critical insight from MNIST experiments is that **encoder quality is the highest-leverage variable**. PCA-8D (43.6% variance) yielded trie accuracy of 76.5% vs CNN-8D (87.3% variance) yielding 95.2%. This means the phase's success hinges on getting good 8D features for each benchmark, not on trie modifications.

**Primary recommendation:** Use the existing `run_trie_mnist.py` script architecture as a template. Add scikit-learn for baseline classifiers. Use 20 Newsgroups (bundled with sklearn, 20 classes, standard benchmark) for text. Use TruncatedSVD (not PCA) for text feature reduction since TF-IDF matrices are sparse. Expect text classification to be the weakest benchmark due to extreme dimensionality reduction from ~50K TF-IDF features to 8D.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Fashion-MNIST**: 10 classes, 28x28 grayscale. Same image pipeline as MNIST.
- **CIFAR-10**: 10 classes, 32x32 color. Requires larger CNN encoder.
- **Text classification**: At least one benchmark (e.g., 20 Newsgroups or AG News). Use TF-IDF / bag-of-words features projected to O via PCA. No neural text encoder. This produces a fully gradient-free pipeline and is a stronger test of the algebraic encoder thesis.
- **No other modalities** (audio, tabular) in this phase.
- **Fashion-MNIST encoder**: Small task-specific CNN (same architecture as MNIST encoder). Train from scratch.
- **CIFAR-10 encoder**: Try multiple CNN sizes (2-layer, 4-layer, ResNet-8-scale). Report how encoder capacity affects trie accuracy. Train all from scratch, do not use pre-trained models.
- **Text encoder**: TF-IDF or bag-of-words features, PCA to 8D, normalize to unit octonions. No neural encoder. Explore option deeply before considering fallback to pre-trained models.
- **General principle**: Deeply investigate small task-specific encoders first. If they fail, understand why and iterate before falling back to off-the-shelf models.
- **Analysis**: Per-benchmark top-line accuracy, per-class accuracy, confusion matrix, failure mode characterization, trie structure analysis, learning curves (at least 3 points)
- **Success criteria**: Within 5pp of kNN on same features. If missed, characterize why.
- **Comparison methods**: kNN (k=1, k=5), Random Forest, SVM (RBF kernel), Logistic Regression, CNN encoder's own classification head, Octonionic Trie
- **Dependencies**: scikit-learn needed for RF, SVM, LR (add to deps); torchvision already available; text dataset bundled with sklearn or downloaded separately

### Claude's Discretion
- Specific CNN architectures for each benchmark
- Which text classification dataset to use (20 Newsgroups, AG News, or similar)
- TF-IDF vectorization parameters (max features, n-gram range)
- Number of training epochs for CNN encoders
- Exact train/test splits (use standard splits where available)

### Deferred Ideas (OUT OF SCOPE)
- Class-subalgebra routing characterization -- later investigation phase, not T1
- Audio and tabular modality benchmarks -- future phase
- Pre-trained encoder comparison -- only if task-specific CNNs fail and we understand why
</user_constraints>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-learn | 1.8.0 | RF, SVM, LR classifiers + TF-IDF + TruncatedSVD + metrics | Industry standard for classical ML; bundles 20 Newsgroups dataset |
| torch | 2.9.1+rocm7.2 | CNN encoder training, tensor operations | Already installed in container |
| torchvision | 0.24.0+rocm7.2 | FashionMNIST, CIFAR10 datasets | Already installed in container |
| matplotlib | >=3.10.8 | Confusion matrix plots, learning curves | Already in dependencies |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| seaborn | >=0.13.2 | Heatmap confusion matrices | Already installed; use for prettier confusion matrices |
| numpy | >=1.26 | Array operations for sklearn interop | Already in dev dependencies |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| 20 Newsgroups | AG News (4 classes) | AG News requires torchtext (deprecated) or HuggingFace. 20 Newsgroups bundled with sklearn, 20 classes = harder test, better. Use 20 Newsgroups. |
| TruncatedSVD | PCA | PCA requires dense matrix (centers the data), destroying sparsity of TF-IDF. TruncatedSVD works directly on sparse matrices. Use TruncatedSVD for text. |
| sklearn kNN | Custom PyTorch kNN (existing) | Existing kNN in `run_trie_mnist.py` uses torch.cdist. Keep it for consistency with prior MNIST results, but also run sklearn KNeighborsClassifier for the comparison table. |

**Installation:**
```bash
docker compose run --rm dev uv add scikit-learn
```

**Version verification:** scikit-learn 1.8.0 is current on PyPI (released Dec 2025). Supports Python 3.12.

## Architecture Patterns

### Recommended Project Structure
```
scripts/
  run_trie_mnist.py              # Existing reference (DO NOT modify)
  run_trie_mnist_experiments.py   # Existing reference (DO NOT modify)
  run_trie_fashion_mnist.py       # New: Fashion-MNIST benchmark
  run_trie_cifar10.py             # New: CIFAR-10 benchmark (multi-encoder)
  run_trie_text.py                # New: Text classification benchmark
  run_trie_benchmark_summary.py   # New: Aggregate results across all benchmarks
results/
  trie_benchmarks/
    fashion_mnist/
      results.json
      confusion_matrix.png
      learning_curve.png
    cifar10/
      results.json
      confusion_matrix.png
      encoder_comparison.png
    text/
      results.json
      confusion_matrix.png
```

### Pattern 1: Benchmark Script Structure (follow run_trie_mnist.py)
**What:** Each benchmark script follows the same structure: load data, encode to 8D, run trie + baselines, compute metrics, save results.
**When to use:** Every benchmark script.
**Example:**
```python
def main():
    # 1. Load dataset (standard train/test split)
    train_x, train_y, test_x, test_y = load_dataset()

    # 2. Train encoder (CNN or TF-IDF+SVD)
    encoder = train_encoder(train_x, train_y)
    train_features = encode(encoder, train_x)  # -> [N, 8]
    test_features = encode(encoder, test_x)    # -> [N, 8]

    # 3. Normalize to unit octonions
    train_oct = train_features / train_features.norm(dim=1, keepdim=True).clamp(min=1e-10)
    test_oct = test_features / test_features.norm(dim=1, keepdim=True).clamp(min=1e-10)

    # 4. Run all classifiers on same features
    results = {}
    results["trie"] = run_trie(train_oct, train_y, test_oct, test_y)
    results["knn_k1"] = run_knn(train_oct, train_y, test_oct, test_y, k=1)
    results["knn_k5"] = run_knn(train_oct, train_y, test_oct, test_y, k=5)
    results["rf"] = run_random_forest(train_oct, train_y, test_oct, test_y)
    results["svm_rbf"] = run_svm(train_oct, train_y, test_oct, test_y)
    results["logreg"] = run_logistic_regression(train_oct, train_y, test_oct, test_y)
    results["cnn_head"] = evaluate_cnn_head(encoder, test_x, test_y)  # upper bound

    # 5. Per-class analysis, confusion matrices, save
    analyze_results(results, test_y)
```

### Pattern 2: Sklearn Baseline Runner
**What:** Common function for running sklearn classifiers on the same 8D features.
**When to use:** Every benchmark.
**Example:**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def run_sklearn_baselines(
    train_x: np.ndarray, train_y: np.ndarray,
    test_x: np.ndarray, test_y: np.ndarray,
) -> dict:
    classifiers = {
        "knn_k1": KNeighborsClassifier(n_neighbors=1),
        "knn_k5": KNeighborsClassifier(n_neighbors=5),
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
        "svm_rbf": SVC(kernel="rbf", random_state=42),
        "logreg": LogisticRegression(max_iter=1000, random_state=42),
    }
    results = {}
    for name, clf in classifiers.items():
        clf.fit(train_x, train_y)
        preds = clf.predict(test_x)
        results[name] = {
            "accuracy": accuracy_score(test_y, preds),
            "confusion_matrix": confusion_matrix(test_y, preds).tolist(),
            "classification_report": classification_report(test_y, preds, output_dict=True),
        }
    return results
```

### Pattern 3: CNN Encoder with Feature Extraction
**What:** Train a CNN for classification, then use penultimate layer as feature extractor.
**When to use:** Fashion-MNIST and CIFAR-10.
**Example:**
```python
class CNNEncoder(nn.Module):
    def __init__(self, feature_dim=8, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, feature_dim), nn.ReLU(),
        )
        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        feat = self.features(x)
        return self.classifier(feat)

    def extract(self, x):
        return self.features(x)
```

### Pattern 4: TF-IDF + TruncatedSVD for Text
**What:** Build gradient-free text pipeline using sklearn.
**When to use:** Text classification benchmark.
**Example:**
```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def load_text_features(n_components=8):
    train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))

    vectorizer = TfidfVectorizer(max_features=10000, sublinear_tf=True, max_df=0.5, stop_words="english")
    train_tfidf = vectorizer.fit_transform(train.data)
    test_tfidf = vectorizer.transform(test.data)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    train_reduced = svd.fit_transform(train_tfidf)
    test_reduced = svd.transform(test_tfidf)

    # Report variance explained
    print(f"TruncatedSVD {n_components}D: {svd.explained_variance_ratio_.sum():.1%} variance")

    # Normalize to unit octonions
    train_norms = np.linalg.norm(train_reduced, axis=1, keepdims=True)
    train_norms = np.clip(train_norms, 1e-10, None)
    train_oct = train_reduced / train_norms

    test_norms = np.linalg.norm(test_reduced, axis=1, keepdims=True)
    test_norms = np.clip(test_norms, 1e-10, None)
    test_oct = test_reduced / test_norms

    return (
        torch.tensor(train_oct, dtype=torch.float64),
        torch.tensor(train.target),
        torch.tensor(test_oct, dtype=torch.float64),
        torch.tensor(test.target),
        train.target_names,
    )
```

### Pattern 5: Learning Curve Generation
**What:** Run accuracy at multiple training set sizes.
**When to use:** Required for all benchmarks (CONTEXT.md says "at least 3 points").
**Example:**
```python
def learning_curve(train_x, train_y, test_x, test_y, fractions=[0.1, 0.25, 0.5, 1.0]):
    results = []
    for frac in fractions:
        n = int(len(train_x) * frac)
        subset_x, subset_y = train_x[:n], train_y[:n]
        acc = run_trie_and_measure(subset_x, subset_y, test_x, test_y)
        results.append({"fraction": frac, "n_train": n, "accuracy": acc})
    return results
```

### Anti-Patterns to Avoid
- **Training baselines on different features than the trie:** All methods MUST use the same 8D features. The CNN head is the ONLY method that sees original-dimension features (as an upper bound).
- **Using PCA instead of TruncatedSVD for text:** PCA centers the data, which densifies the sparse TF-IDF matrix, causing potential OOM. TruncatedSVD (=LSA) works directly on sparse matrices.
- **Comparing against published full-feature baselines:** A kNN on raw 784D MNIST pixels gets ~97%. That is NOT a fair comparison. All baselines must use the same 8D projection.
- **Modifying the trie for this phase:** The trie code is frozen for T1. Understanding comes from varying encoders and data, not trie parameters.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| TF-IDF vectorization | Custom tokenizer + TF-IDF | `sklearn.feature_extraction.text.TfidfVectorizer` | Handles tokenization, stopwords, sublinear scaling, IDF |
| SVD on sparse matrices | Dense PCA on sparse TF-IDF | `sklearn.decomposition.TruncatedSVD` | Works directly on sparse CSR matrices without densification |
| SVM classifier | - | `sklearn.svm.SVC(kernel='rbf')` | Optimized C implementation, handles kernel computation |
| Random Forest | - | `sklearn.ensemble.RandomForestClassifier` | Proven implementation with 100+ estimators |
| Logistic Regression | - | `sklearn.linear_model.LogisticRegression` | Multi-class with LBFGS solver, max_iter=1000 |
| Confusion matrix | Manual counting loop | `sklearn.metrics.confusion_matrix` | Handles edge cases, integrates with `ConfusionMatrixDisplay` |
| Classification report | Manual per-class stats | `sklearn.metrics.classification_report(output_dict=True)` | Precision, recall, F1, support per class |
| Dataset loading | Manual download/parsing | `torchvision.datasets.FashionMNIST`, `CIFAR10`, `sklearn.datasets.fetch_20newsgroups` | Standard splits, caching, error handling |
| Data augmentation for CIFAR-10 CNN training | Custom transforms | `torchvision.transforms.Compose([RandomHorizontalFlip, RandomCrop, ...])` | Standard augmentation pipeline |
| kNN with sklearn | - | `sklearn.neighbors.KNeighborsClassifier` | For consistency in the comparison table; existing PyTorch kNN can be kept for trie comparison |

**Key insight:** The trie is the research artifact. Everything else (encoders, baselines, metrics) should use battle-tested libraries. The research value is in the trie's accuracy relative to baselines, not in building infrastructure.

## Common Pitfalls

### Pitfall 1: TF-IDF to 8D Loses Too Much Information
**What goes wrong:** TF-IDF produces ~10K-50K sparse features. Reducing to 8D via TruncatedSVD preserves maybe 5-15% of variance. Classification accuracy drops dramatically for ALL methods including baselines.
**Why it happens:** Text features are inherently high-dimensional. 8D is an extreme bottleneck for 20-class text classification.
**How to avoid:** (1) Report the explained variance ratio from TruncatedSVD. (2) Run baselines on the same 8D features so comparison is fair. (3) Consider reducing to fewer classes (subset of 20 Newsgroups -- e.g., 4-5 well-separated categories) as a secondary experiment. (4) Document this limitation honestly as a finding about the octonionic approach's text capability.
**Warning signs:** All methods below 30% accuracy on 20-class text with 8D features.

### Pitfall 2: CIFAR-10 2-Layer CNN Encoder is Too Weak
**What goes wrong:** A simple 2-layer CNN on CIFAR-10 barely exceeds 50% accuracy. Its 8D features will be poor, making the trie comparison meaningless.
**Why it happens:** CIFAR-10 is significantly harder than MNIST. Color images with complex backgrounds require deeper models.
**How to avoid:** Try multiple encoder sizes as specified in CONTEXT.md. Start with 4-layer, then try ResNet-8 scale. The CNN head accuracy is the upper bound -- if it's <70%, features are too weak.
**Warning signs:** CNN head accuracy < 70%, large gap between CNN head and all baselines.

### Pitfall 3: Unfair Comparison via Feature Dimension Mismatch
**What goes wrong:** Comparing trie accuracy on 8D features against published SOTA on raw features makes the trie look bad.
**Why it happens:** 8D is a heavy bottleneck. SOTA Fashion-MNIST methods use full 784D or 28x28 images.
**How to avoid:** ALL methods in the comparison table must use the SAME 8D features. The CNN head on original features is clearly marked as an upper bound.
**Warning signs:** Results section comparing against published numbers without noting the feature dimension.

### Pitfall 4: SVM Takes Forever on Large Datasets
**What goes wrong:** SVC with RBF kernel has O(n^2) to O(n^3) complexity. On 60K training samples, it can take hours.
**Why it happens:** Kernel SVM must compute kernel matrix between all pairs.
**How to avoid:** For large datasets, subsample to 10K-20K training points for SVM, OR use `LinearSVC` as a faster alternative. Since features are only 8D, RBF SVM should be tractable on 10K-20K samples. Document if subsampling was used.
**Warning signs:** SVM fitting taking >10 minutes.

### Pitfall 5: 20 Newsgroups Metadata Leakage
**What goes wrong:** The 20 Newsgroups dataset includes email headers, footers, and quoted text that reveal the newsgroup (class). Classifiers overfit to metadata, not content.
**Why it happens:** Standard dataset loading includes all metadata by default.
**How to avoid:** Use `fetch_20newsgroups(remove=("headers", "footers", "quotes"))`. This is the standard benchmark convention.
**Warning signs:** Suspiciously high accuracy (>90%) on all-20-classes text classification.

### Pitfall 6: Trie Training Time Scales Linearly with Epochs x Samples
**What goes wrong:** The trie inserts one sample at a time (no batching). With 60K samples and 3 epochs, that's 180K sequential insertions.
**Why it happens:** The trie is inherently sequential -- each insert modifies the tree.
**How to avoid:** Use the existing MNIST patterns: subsample to 10K-20K for initial experiments, full 60K for final runs. Report timing alongside accuracy.
**Warning signs:** Scripts taking >30 minutes per run.

### Pitfall 7: Float64 vs Float32 Feature Mismatch
**What goes wrong:** The trie uses float64 by default. CNN features are typically float32. sklearn returns float64. Inconsistent dtypes cause subtle precision differences.
**Why it happens:** Mixed ecosystems (PyTorch defaults to float32, trie to float64, sklearn to float64).
**How to avoid:** Convert all features to float64 before feeding to the trie, matching the existing MNIST script pattern. Use `.to(torch.float64)` after CNN feature extraction.
**Warning signs:** Trie accuracy differs between runs with same data.

## Code Examples

### Confusion Matrix Visualization
```python
# Source: sklearn.metrics.ConfusionMatrixDisplay docs
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
```

### CIFAR-10 CNN Encoders (Multiple Sizes)
```python
# 2-layer CNN for CIFAR-10 (minimal baseline)
class CIFAR_CNN_2Layer(nn.Module):
    def __init__(self, feature_dim=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, feature_dim),
        )
        self.classifier = nn.Linear(feature_dim, 10)

# 4-layer CNN for CIFAR-10
class CIFAR_CNN_4Layer(nn.Module):
    def __init__(self, feature_dim=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, feature_dim),
        )
        self.classifier = nn.Linear(feature_dim, 10)

# ResNet-8-scale for CIFAR-10
class CIFAR_CNN_ResNet8(nn.Module):
    """Simplified ResNet with 8 conv layers for CIFAR-10."""
    def __init__(self, feature_dim=8):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU()
        )
        # 3 residual blocks (6 conv layers + 1 initial + 1 final = 8)
        self.block1 = self._make_block(16, 16)
        self.block2 = self._make_block(16, 32, stride=2)
        self.block3 = self._make_block(32, 64, stride=2)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, feature_dim),
        )
        self.classifier = nn.Linear(feature_dim, 10)

    def _make_block(self, in_ch, out_ch, stride=1):
        # standard residual block
        ...
```

### Fashion-MNIST CNN Encoder (Reuse MNIST Architecture)
```python
# Same as existing SmallCNN in run_trie_mnist_experiments.py
class FashionMNIST_CNN(nn.Module):
    def __init__(self, feature_dim=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, feature_dim), nn.ReLU(),
        )
        self.classifier = nn.Linear(feature_dim, 10)
```

### Learning Curve with Multiple Methods
```python
def generate_learning_curves(train_oct, train_y, test_oct, test_y, fractions):
    curves = {"trie": [], "knn_k5": [], "svm_rbf": []}
    for frac in fractions:
        n = int(len(train_oct) * frac)
        sub_x, sub_y = train_oct[:n], train_y[:n]

        # Trie
        trie_acc = run_trie(sub_x, sub_y, test_oct, test_y)
        curves["trie"].append({"frac": frac, "n": n, "acc": trie_acc})

        # kNN
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(sub_x.numpy(), sub_y.numpy())
        knn_acc = knn.score(test_oct.numpy(), test_y.numpy())
        curves["knn_k5"].append({"frac": frac, "n": n, "acc": knn_acc})

        # SVM (subsample if needed)
        svm = SVC(kernel="rbf", random_state=42)
        svm.fit(sub_x.numpy(), sub_y.numpy())
        svm_acc = svm.score(test_oct.numpy(), test_y.numpy())
        curves["svm_rbf"].append({"frac": frac, "n": n, "acc": svm_acc})

    return curves
```

## State of the Art

### Expected Accuracy Ranges on 8D Features

These are estimates based on full-feature published results, adjusted for the extreme dimensionality reduction to 8D. Actual numbers will depend on encoder quality.

| Benchmark | Full-Feature SOTA | kNN on 8D (expected) | Trie Target (within 5pp of kNN) |
|-----------|-------------------|-------------------|---------------------------------|
| Fashion-MNIST | ~88.8% (SVM/RBF), ~96% (CNN) | ~80-86% (with good CNN encoder) | ~75-81% |
| CIFAR-10 | ~28% (kNN raw pixels), ~94% (CNN) | ~55-75% (depends heavily on encoder) | ~50-70% |
| Text (20NG, 20 classes) | ~77-89% (various, full TF-IDF) | ~15-35% (8D is extreme bottleneck) | ~10-30% |
| Text (20NG, 4-5 classes) | ~90%+ | ~40-60% | ~35-55% |

### Key Benchmark Dataset Details

| Dataset | Classes | Train | Test | Input Dim | Format |
|---------|---------|-------|------|-----------|--------|
| Fashion-MNIST | 10 | 60,000 | 10,000 | 28x28 grayscale | torchvision.datasets.FashionMNIST |
| CIFAR-10 | 10 | 50,000 | 10,000 | 32x32x3 color | torchvision.datasets.CIFAR10 |
| 20 Newsgroups | 20 | 11,314 | 7,532 | ~50K TF-IDF sparse | sklearn.datasets.fetch_20newsgroups |

### Fashion-MNIST Class Names
```python
FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
```

### CIFAR-10 Class Names
```python
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
```

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| torchtext for AG News | Use sklearn fetch_20newsgroups or huggingface datasets | 2024 (torchtext deprecated) | torchtext is unmaintained; 20 Newsgroups is built into sklearn |
| PCA on sparse matrices | TruncatedSVD for sparse text features | Long-standing | PCA densifies; TruncatedSVD preserves sparsity, is called LSA in text domain |

## Discretion Recommendations

### Text Dataset: Use 20 Newsgroups
**Rationale:** Bundled with sklearn (no extra dependency or download complexity). 20 classes makes it a harder test than AG News (4 classes). Standard benchmark with well-understood baselines. Available via `fetch_20newsgroups()`.

**Secondary experiment:** Also run on a 4-5 class subset (e.g., `['comp.graphics', 'rec.sport.baseball', 'sci.med', 'talk.politics.guns']`) to see if fewer classes with more separation helps the 8D bottleneck.

### TF-IDF Parameters
```python
TfidfVectorizer(
    max_features=10000,    # Limit vocabulary size for tractable SVD
    sublinear_tf=True,     # Apply log(1+tf) -- standard for text classification
    max_df=0.5,            # Ignore terms in >50% of docs (near-stopwords)
    min_df=5,              # Ignore rare terms (< 5 docs)
    stop_words="english",  # Remove standard English stopwords
)
```

### CNN Training Epochs
- **Fashion-MNIST**: 10 epochs (same dataset complexity as MNIST, converges fast with Adam)
- **CIFAR-10 2-layer**: 20 epochs
- **CIFAR-10 4-layer**: 30 epochs
- **CIFAR-10 ResNet-8**: 50 epochs (with cosine annealing LR schedule)

### Train/Test Splits
Use standard splits for all datasets. No custom splitting. For learning curves, subsample from the training set only.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| torch | CNN training, trie | Yes | 2.9.1+rocm7.2 | -- |
| torchvision | FashionMNIST, CIFAR10 datasets | Yes | 0.24.0+rocm7.2 | -- |
| matplotlib | Plots | Yes | >=3.10.8 | -- |
| seaborn | Heatmaps | Yes | >=0.13.2 | -- |
| scikit-learn | Baselines + TF-IDF + metrics | **No** | -- | Must be added via `uv add scikit-learn` |
| 20 Newsgroups data | Text benchmark | Available via sklearn download | -- | Downloads automatically on first use |

**Missing dependencies with no fallback:**
- scikit-learn: MUST be added before any baseline code runs. Required for RF, SVM, LR, TF-IDF, TruncatedSVD, confusion_matrix, classification_report.

**Missing dependencies with fallback:**
- None. Everything else is installed or bundled.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest + hypothesis |
| Config file | `pyproject.toml [tool.pytest.ini_options]` |
| Quick run command | `docker compose run --rm dev uv run pytest tests/test_trie.py -x` |
| Full suite command | `docker compose run --rm dev uv run pytest tests/ -x --timeout=120` |

### Phase Requirements to Test Map

Since T1 is primarily an experimental/benchmarking phase (not a library phase), the "tests" are the benchmark scripts themselves producing results. However, we should add integration tests for the new utilities.

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| T1-01 | Fashion-MNIST benchmark runs and produces accuracy | integration | `docker compose run --rm dev uv run python scripts/run_trie_fashion_mnist.py --n-train 1000 --n-test 200` | No (Wave 0) |
| T1-02 | CIFAR-10 benchmark runs with multiple encoders | integration | `docker compose run --rm dev uv run python scripts/run_trie_cifar10.py --n-train 1000 --n-test 200 --encoder 2layer` | No (Wave 0) |
| T1-03 | Text benchmark runs with TF-IDF+SVD | integration | `docker compose run --rm dev uv run python scripts/run_trie_text.py --n-train 500 --n-test 200` | No (Wave 0) |
| T1-04 | sklearn baselines produce valid results | unit | `docker compose run --rm dev uv run pytest tests/test_trie_benchmarks.py -x` | No (Wave 0) |
| T1-05 | Results JSON files contain required fields | unit | `docker compose run --rm dev uv run pytest tests/test_trie_benchmarks.py::test_result_schema -x` | No (Wave 0) |

### Sampling Rate
- **Per task commit:** `docker compose run --rm dev uv run pytest tests/test_trie.py -x`
- **Per wave merge:** `docker compose run --rm dev uv run pytest tests/ -x --timeout=120`
- **Phase gate:** Full suite green + all benchmark scripts run successfully with `--n-train 1000 --n-test 200`

### Wave 0 Gaps
- [ ] `tests/test_trie_benchmarks.py` -- validates sklearn baseline runner, result schema, feature normalization
- [ ] scikit-learn installation: `docker compose run --rm dev uv add scikit-learn`

## Open Questions

1. **How much variance does TruncatedSVD-8D explain on 20 Newsgroups TF-IDF?**
   - What we know: 2 components explain ~6% on 20 Newsgroups. 8D will be more but still very low (likely 10-20%).
   - What's unclear: Whether the trie can do anything useful with so little information.
   - Recommendation: Run the experiment. If all methods are below 30% on full 20 classes, report that as a finding and also run on 4-5 class subset.

2. **What CIFAR-10 CNN encoder size is needed for meaningful 8D features?**
   - What we know: 2-layer CNN on CIFAR-10 gets ~50% as a classifier. 8D bottleneck features will be worse.
   - What's unclear: Whether ResNet-8 scale is sufficient or if we need more capacity.
   - Recommendation: Try all three sizes (2-layer, 4-layer, ResNet-8). Report encoder head accuracy alongside trie accuracy. The encoder that gives the highest CNN-head accuracy is the one to use for final trie comparison.

3. **Should we use full 60K training samples or subsample?**
   - What we know: MNIST experiments used 10K train / 2K test. Full MNIST is 60K/10K. More data helps trie more than kNN.
   - What's unclear: Whether the time cost of full datasets justifies the accuracy gain.
   - Recommendation: Run initial experiments with 10K/2K (matching MNIST experiments). Run full-dataset for the final result table. Learning curves will show the impact of data size.

## Sources

### Primary (HIGH confidence)
- [scikit-learn 1.8.0 docs] - TfidfVectorizer, TruncatedSVD, fetch_20newsgroups, classifier APIs
- [torchvision 0.24.0 docs] - FashionMNIST and CIFAR10 dataset APIs
- [Fashion-MNIST GitHub README](https://github.com/zalandoresearch/fashion-mnist) - Dataset details, standard splits
- Existing project code: `scripts/run_trie_mnist.py`, `scripts/run_trie_mnist_experiments.py` - Validated patterns

### Secondary (MEDIUM confidence)
- [sklearn 20 Newsgroups text classification example](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html) - Baseline accuracy numbers (77-78% with metadata removed)
- Fashion-MNIST classical ML benchmarks: SVM ~88.8%, RF ~87.6%, kNN ~85.9% (on full features, not 8D)
- CIFAR-10 kNN on raw pixels: ~28-40% (demonstrates need for feature extraction)

### Tertiary (LOW confidence)
- Expected accuracy on 8D features: extrapolated from full-feature results, not directly measured. Actual numbers may differ significantly.
- TruncatedSVD-8D variance on 20 Newsgroups: estimated ~10-20%, needs empirical verification.

## Project Constraints (from CLAUDE.md)

- **All Python commands MUST run inside the dev container**: `docker compose run --rm dev uv run ...`
- **Never run `uv`, `python`, or `pytest` directly on the host**
- **File edits happen on the host** (mounted at /workspace in container)
- **First run requires `docker compose run --rm dev uv sync`** to install project deps
- **Code style**: Python 3.12+, type hints on all public APIs
- **Testing**: pytest + hypothesis for property-based testing
- **Dependencies**: Minimize dependencies beyond PyTorch (but scikit-learn is explicitly required for this phase)
- **Ruff**: line-length=100, target-version=py312

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- scikit-learn, torchvision are well-documented, stable libraries
- Architecture: HIGH -- following established patterns from existing MNIST scripts
- Pitfalls: MEDIUM-HIGH -- TF-IDF-to-8D information loss is well-understood; encoder capacity uncertainty is the main risk
- Expected accuracy ranges: MEDIUM -- extrapolated from full-feature benchmarks, not empirically measured at 8D

**Research date:** 2026-03-29
**Valid until:** 2026-04-28 (30 days -- stable domain, no expected library changes)
