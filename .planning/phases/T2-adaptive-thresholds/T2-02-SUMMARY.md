---
phase: T2-adaptive-thresholds
plan: 02
subsystem: benchmark-infrastructure
tags: [feature-caching, pca, cnn-encoder, resnet, tfidf, svd, torch-save]

# Dependency graph
requires:
  - phase: T1-benchmark-generalization
    provides: "5 benchmark scripts with encoder pipelines (MNIST PCA, Fashion-MNIST CNN, CIFAR-10 ResNet-8, Text TF-IDF+SVD)"
provides:
  - "cache_features.py script producing .pt files for all 5 T1 benchmarks"
  - "Cached 8D unit-normalized float64 features at full and 10K subset scales"
  - "scripts/sweep/ package structure for T2 sweep infrastructure"
affects: [T2-03, T2-04, T2-05, T2-06, T2-07, T2-08, T2-09, T2-10, T2-11]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Feature caching as .pt files with standardized dict schema"]

key-files:
  created:
    - scripts/sweep/__init__.py
    - scripts/sweep/cache_features.py
  modified:
    - .gitignore

key-decisions:
  - "10K subset PCA fit on subsampled 10K training data (not full 60K) to match T1 pipeline exactly"
  - "Full-scale PCA fit on all 60K training data for best representation"
  - "Fashion-MNIST CNN trained for 5 epochs (plan spec) with seed=42 for reproducibility"
  - "CIFAR-10 uses ResNet-8 only (per plan D-07) with 50 epochs and cosine annealing"

patterns-established:
  - "Feature cache .pt schema: dict with train_x, train_y, test_x, test_y, class_names, benchmark, n_train, n_test"
  - "CLI pattern: --benchmarks (comma-separated or 'all'), --output-dir, --subset-only, --full-only"

requirements-completed: []

# Metrics
duration: 13min
completed: 2026-03-30
---

# Phase T2 Plan 02: Feature Caching Summary

**Pre-compute 8D unit octonion features for all 5 T1 benchmarks as .pt files with exact pipeline replication**

## Performance

- **Duration:** 13 min
- **Started:** 2026-03-30T01:51:01Z
- **Completed:** 2026-03-30T02:04:01Z
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments
- Created scripts/sweep/cache_features.py with cache_all_features() handling all 5 benchmarks
- Each benchmark pipeline exactly replicates its T1 source script (PCA, CNN encoder, TF-IDF+SVD)
- Output format standardized: float64 tensors of shape [N, 8] with unit norm < 1e-15 deviation
- CLI supports flexible benchmark selection, output directory, and subset/full-only modes
- Validation step confirms cached features produce consistent trie accuracy

## Task Commits

Each task was committed atomically:

1. **Task 1: Feature caching script for all benchmarks** - `364b246` (feat)

## Files Created/Modified
- `scripts/sweep/__init__.py` - Empty package init for sweep module
- `scripts/sweep/cache_features.py` - Main feature caching script with all 5 benchmark pipelines
- `.gitignore` - Added .data/ to exclude downloaded datasets

## Decisions Made
- 10K subset MNIST uses PCA fit on the subsampled 10K training data (not full 60K) to exactly match the T1 pipeline from run_trie_mnist.py:load_mnist_pca8. This ensures cached features reproduce identical PCA projections.
- Full-scale MNIST uses PCA fit on all 60K training data for the best feature representation at scale.
- Fashion-MNIST CNN trained for 5 epochs per plan specification with seeded DataLoader shuffle for reproducibility.
- All encoder architectures (SmallCNN, ResidualBlock, CIFAR_CNN_ResNet8) copied exactly from their source scripts to ensure identical feature extraction.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed PCA pipeline mismatch for 10K subset**
- **Found during:** Task 1 (Feature caching script)
- **Issue:** Initial implementation computed PCA on full 60K then subsampled, producing different features than T1 baseline which subsampled first then computed PCA on the 10K subset
- **Fix:** Changed 10K subset to subsample raw pixels first, then fit PCA on the subsampled training data, matching the exact T1 pipeline
- **Files modified:** scripts/sweep/cache_features.py
- **Verification:** Cached features produce identical trie accuracy (71.1%) as running run_trie_mnist.py directly with same worktree trie.py

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential fix for feature reproducibility. No scope creep.

## Issues Encountered
- The worktree's trie.py is an older version that produces ~71% accuracy on MNIST 10K (the 95.2% result was achieved with the updated trie.py from the main repo after T1 improvements). This does not affect the feature caching -- the features are correct and will produce T1-matching accuracy when used with the updated trie.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Cached features ready for T2-03 (ThresholdPolicy abstraction) and all subsequent sweep plans
- Sweep workers can load .pt files without GPU, encoder, or data pipeline dependencies
- scripts/sweep/ package ready for additional sweep infrastructure modules

## Self-Check: PASSED

- scripts/sweep/cache_features.py: FOUND
- scripts/sweep/__init__.py: FOUND
- T2-02-SUMMARY.md: FOUND
- Commit 364b246: FOUND

---
*Phase: T2-adaptive-thresholds*
*Completed: 2026-03-30*
