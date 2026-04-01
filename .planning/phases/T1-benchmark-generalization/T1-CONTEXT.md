---
phase: T1-benchmark-generalization
type: context
created: 2026-03-29
---

# Phase T1: Benchmark Generalization — Context

## Domain Boundary

Determine whether the octonionic trie's 95.2% MNIST result generalizes to other standard benchmarks. Produce per-benchmark accuracy comparisons against kNN and other standard methods, per-class analysis, confusion matrices, and failure mode characterization. Understand how and why the trie succeeds or fails on each benchmark.

## Canonical References

- `docs/thesis/oct-trie.tex` — Trie thesis with MNIST results (Section 7)
- `src/octonion/trie.py` — Trie implementation (338 lines)
- `tests/test_trie.py` — Existing unit tests (18 tests)
- `scripts/run_trie_mnist.py` — MNIST benchmark script (reference for new benchmarks)
- `scripts/run_trie_mnist_experiments.py` — Encoder/dimensionality experiments

## Prior Decisions (Locked)

From trie development (validated experimentally):
- Fixed routing keys (adaptive keys cause forgetting)
- Content/routing separation in each node
- Similarity-based child selection (inner product with routing key)
- Encoder decoupled from trie
- Associator threshold 0.3 as default (T2 will investigate adaptive)
- kNN (k=5) on same features as primary baseline

## Decisions

### Benchmarks

- **Fashion-MNIST**: 10 classes, 28x28 grayscale. Same image pipeline as MNIST.
- **CIFAR-10**: 10 classes, 32x32 color. Requires larger CNN encoder.
- **Text classification**: At least one benchmark (e.g., 20 Newsgroups or AG News). Use TF-IDF / bag-of-words features projected to O via PCA. No neural text encoder. This produces a fully gradient-free pipeline and is a stronger test of the algebraic encoder thesis.
- **No other modalities** (audio, tabular) in this phase.

### Encoder Strategy

- **Fashion-MNIST**: Small task-specific CNN (same architecture as MNIST encoder). Train from scratch.
- **CIFAR-10**: Try multiple CNN sizes (2-layer, 4-layer, ResNet-8-scale). Report how encoder capacity affects trie accuracy. Train all from scratch, do not use pre-trained models.
- **Text**: TF-IDF or bag-of-words features, PCA to 8D, normalize to unit octonions. No neural encoder. Explore option deeply before considering fallback to pre-trained models.
- **General principle**: Deeply investigate small task-specific encoders first. If they fail, understand why and iterate before falling back to off-the-shelf models.

### Analysis Depth

- Per-benchmark: top-line accuracy, per-class accuracy, confusion matrix
- Failure mode characterization: which classes does the trie confuse and why
- Trie structure analysis: node count, depth, branching factor per benchmark
- Learning curve: accuracy vs training set size (at least 3 points)

### Success Criteria

- Primary goal is understanding how the trie works and why, not hitting a specific number
- Target: within 5pp of kNN on same features for each benchmark
- If target is missed: characterize why (encoder quality? class confusion? insufficient depth?)

### Comparison Methods

All methods evaluated on the same PCA-8D features (or equivalent projection) for fair comparison:
- kNN (k=1, k=5)
- Random Forest
- SVM (RBF kernel)
- Logistic Regression
- CNN encoder's own classification head (upper bound on feature quality)
- Octonionic Trie

### Dependencies

- scikit-learn needed for RF, SVM, LR baselines (add to dev dependencies)
- torchvision for Fashion-MNIST, CIFAR-10 (already available)
- Text dataset: either bundled with sklearn or downloaded separately

## Deferred Ideas

- Class-subalgebra routing characterization (which subalgebras each class activates) — potentially interesting for understanding the algebra's organizational role, but belongs in a later investigation phase, not T1
- Audio and tabular modality benchmarks — future phase
- Pre-trained encoder comparison — only if task-specific CNNs fail and we understand why

## Claude's Discretion

- Specific CNN architectures for each benchmark (researcher/planner decide based on dataset characteristics)
- Which text classification dataset to use (20 Newsgroups, AG News, or similar — researcher recommends based on availability and benchmark conventions)
- TF-IDF vectorization parameters (max features, n-gram range)
- Number of training epochs for CNN encoders
- Exact train/test splits (use standard splits where available)
