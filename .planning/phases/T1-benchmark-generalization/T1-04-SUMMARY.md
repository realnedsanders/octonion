---
phase: T1-benchmark-generalization
plan: 04
subsystem: benchmarking
tags: [20-newsgroups, text-classification, tfidf, truncatedsvd, gradient-free, trie, sklearn]

# Dependency graph
requires:
  - phase: T1-benchmark-generalization
    provides: Shared benchmark utilities (trie_benchmark_utils.py) and scikit-learn
provides:
  - 20 Newsgroups text classification benchmark (fully gradient-free)
  - TF-IDF + TruncatedSVD -> unit octonions pipeline
  - Full TF-IDF LogReg upper bound (pre-SVD accuracy reference)
  - 4-class subset and 20-class full experiments
affects: [T1-05-cross-benchmark]

# Tech tracking
tech-stack:
  added: []
  patterns: [TF-IDF + TruncatedSVD for sparse-to-dense 8D projection, full TF-IDF LogReg as text upper bound]

key-files:
  created:
    - scripts/run_trie_text.py
  modified: []

key-decisions:
  - "TruncatedSVD (not PCA) for sparse TF-IDF matrices to avoid densification OOM"
  - "Full TF-IDF LogReg as upper bound replaces CNN head (no neural encoder for text)"
  - "78 empty documents from header/footer/quote removal kept as zero vectors (not filtered)"

patterns-established:
  - "Gradient-free classification pipeline: TF-IDF -> TruncatedSVD -> unit norm -> trie"
  - "Full-feature LogReg as upper bound for non-neural benchmarks"

requirements-completed: [TRIE-01]

# Metrics
duration: 5min
completed: 2026-03-29
---

# Phase T1 Plan 04: Text Classification Benchmark Summary

**Fully gradient-free 20 Newsgroups benchmark via TF-IDF + TruncatedSVD to 8D unit octonions, achieving 79% trie accuracy on 4-class subset (vs 87% SVM-RBF on same features)**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-29T18:41:53Z
- **Completed:** 2026-03-29T18:47:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Created `scripts/run_trie_text.py` (563 lines) implementing fully gradient-free text classification: TF-IDF vectorization, TruncatedSVD to 8D, unit-norm normalization, trie + 5 sklearn baselines
- Both 20-class full and 4-class subset experiments supported via `--mode` CLI argument
- Full TF-IDF LogReg upper bound shows 8D bottleneck information loss (88.7% full vs 87.1% on 8D for 4-class subset)
- Zero neural network components anywhere in the pipeline (verified: no `nn.Module` imports)

## Results (4-class subset, 1 epoch)

| Method | Accuracy |
|--------|----------|
| Full TF-IDF LogReg (upper bound) | 88.7% |
| SVM-RBF (8D) | 87.4% |
| LogReg (8D) | 87.1% |
| Random Forest (8D) | 86.0% |
| kNN k=5 (8D) | 85.6% |
| kNN k=1 (8D) | 82.1% |
| Octonionic Trie (8D) | 79.0% |

Explained variance at 8D: 4.2% (very low -- text is high-dimensional)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create 20 Newsgroups text classification benchmark script** - `9b2eaf1` (feat)

## Files Created/Modified
- `scripts/run_trie_text.py` - Fully gradient-free text classification benchmark (563 lines)

## Decisions Made
- Used TruncatedSVD (not PCA) for dimensionality reduction because PCA would densify the sparse TF-IDF matrix, potentially causing OOM
- Full TF-IDF LogReg serves as the upper bound (no CNN head available since this is a zero-neural-encoder pipeline)
- 78 empty documents (zero vectors after TF-IDF with header/footer/quote removal) are kept as-is; the normalization reports them but doesn't filter them since they still have valid labels
- Used SHORT_NAMES_20 abbreviated class names for 20-class confusion matrix readability

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed norm check misleading output for zero-vector documents**
- **Found during:** Task 1
- **Issue:** Norm check reported mean=0.966 because 78 empty documents (zero vectors after TF-IDF removal of headers/footers/quotes) dragged down the average
- **Fix:** Changed norm check to report only non-zero vector norms and added explicit warning about empty documents
- **Files modified:** scripts/run_trie_text.py
- **Verification:** Non-zero norm mean is now 1.000000, warning printed for 78 empty documents
- **Committed in:** 9b2eaf1

**2. [Rule 1 - Bug] Fixed step numbering inconsistency**
- **Found during:** Task 1
- **Issue:** Step 2 was labeled `[2/3]` instead of `[2/10]`
- **Fix:** Corrected to `[2/10]`
- **Files modified:** scripts/run_trie_text.py
- **Committed in:** 9b2eaf1

**3. [Rule 3 - Blocking] Created output directory with correct permissions**
- **Found during:** Task 1
- **Issue:** `results/` directory owned by root from prior container runs; container runs as uid 1000 and couldn't create subdirectories
- **Fix:** Used `docker compose run --rm --user root dev` to create and chmod the output directory
- **Verification:** Script runs successfully, writes results.json and PNG files
- **Not committed** (runtime directory, not tracked)

---

**Total deviations:** 3 auto-fixed (2 bugs, 1 blocking)
**Impact on plan:** All fixes necessary for correctness and functionality. No scope creep.

## Issues Encountered
- `results/` directory owned by root from prior container runs -- resolved by running mkdir as root in container
- 20 Newsgroups dataset downloaded on first run (~14 MB) -- cached for subsequent runs

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Text benchmark script ready for T1-05 cross-benchmark analysis
- Results JSON output compatible with shared benchmark utilities
- No blockers

## Known Stubs
None - all pipeline stages fully implemented and functional.

## Self-Check: PASSED

- [x] scripts/run_trie_text.py exists (563 lines)
- [x] Commit 9b2eaf1 exists
- [x] No stubs found
- [x] No nn.Module imports (fully gradient-free verified)

---
*Phase: T1-benchmark-generalization*
*Completed: 2026-03-29*
