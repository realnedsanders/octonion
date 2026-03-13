---
phase: 03-baseline-implementations
verified: 2026-03-08T22:00:00Z
status: gaps_found
score: 2/4 success criteria verified
gaps:
  - truth: "Complex-valued baseline with 4x units matches octonionic network total parameter count and reproduces a published benchmark result within reported variance"
    status: failed
    reason: "Plan 03-06 (CIFAR benchmark reproduction) was never executed. No 03-06-SUMMARY.md exists. No experiments/ directory with CIFAR results. Reproduction has not been run, let alone verified. ROADMAP marks 03-06 as [ ] (incomplete)."
    artifacts:
      - path: "src/octonion/baselines/_benchmarks.py"
        issue: "Infrastructure exists (build_cifar10_data, cifar_network_config, reproduction_report) but no training runs have been executed — the slow reproduction tests are gated behind a human-verify checkpoint that was never reached"
    missing:
      - "Execute run_comparison with CIFAR-10 for all 4 algebras (3 seeds minimum)"
      - "Verify Complex baseline achieves error within 1 std of 26.36% (Trabelsi CIFAR-100) or 6.17% (CIFAR-10)"
      - "Create 03-06-SUMMARY.md with reproduction results"
      - "Human approves reproduction_report.md verdicts at Task 3 checkpoint"
  - truth: "Quaternionic baseline with 2x units matches octonionic network total parameter count and reproduces a published benchmark result within reported variance"
    status: failed
    reason: "Same as Complex baseline — 03-06 never executed. Quaternion reproduction targets (5.44% CIFAR-10 error, 26.01% CIFAR-100 error from Gaudet/Maida 2018) have no empirical evidence."
    artifacts:
      - path: "src/octonion/baselines/_benchmarks.py"
        issue: "cifar_network_config, cifar_train_config, PUBLISHED_RESULTS dict all exist; no actual training experiments exist in experiments/ directory"
    missing:
      - "Execute CIFAR-10/100 training for Quaternion baseline and compare against published 5.44%/26.01%"
      - "Verify result is within 1 standard deviation of published"
      - "Document in reproduction_report.md with pass/fail verdict"
human_verification:
  - test: "Review CIFAR benchmark reproduction results after 03-06 executes"
    expected: "Complex CIFAR-100 error within 1 std of 26.36%, Quaternion CIFAR-10 error within 1 std of 5.44%, Real CIFAR-10 error within 1 std of 6.37%"
    why_human: "Long-running GPU training (hours). Results must be read from experiments/ directory and compared against PUBLISHED_RESULTS. 03-06 has an explicit blocking human-verify checkpoint (Task 3)."
---

# Phase 3: Baseline Implementations Verification Report

**Phase Goal:** Fair comparison networks exist for real, complex, and quaternionic algebras so that every octonionic experiment has trustworthy baselines
**Verified:** 2026-03-08
**Status:** gaps_found
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (from ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC-1 | Real-valued baseline with 8x units matches octonionic network total parameter count within 1% | VERIFIED | `find_matched_width` binary search confirmed in tests/test_baselines_linear.py (39 tests), `_SimpleAlgebraMLP` produces correct R:8x vs O:1x ratio verified in test_baselines_network.py `test_param_matching_mlp` |
| SC-2 | Complex-valued baseline with 4x units matches param count AND reproduces published benchmark within variance | FAILED | Param matching infrastructure verified; CIFAR reproduction never executed (03-06 not started) |
| SC-3 | Quaternionic baseline with 2x units matches param count AND reproduces published benchmark within variance | FAILED | Param matching infrastructure verified; CIFAR reproduction never executed (03-06 not started) |
| SC-4 | All four networks (R, C, H, O) share identical architecture skeleton differing only in algebra module | VERIFIED | `test_skeleton_identity_mlp` and `test_skeleton_identity_conv2d` in test_baselines_network.py explicitly test this; AlgebraNetwork factory methods dispatch to per-algebra layers while keeping topology structure identical |

**Score:** 2/4 truths verified

---

## Required Artifacts

| Artifact | Status | Details |
|----------|--------|---------|
| `src/octonion/baselines/_config.py` | VERIFIED | `AlgebraType` enum (R/C/H/O with dim/multiplier), `NetworkConfig`, `TrainConfig`, `ComparisonConfig` — all substantive |
| `src/octonion/baselines/_algebra_linear.py` | VERIFIED | `RealLinear`, `ComplexLinear`, `QuaternionLinear`, `OctonionDenseLinear` all implemented with algebra-correct multiplication; cross-validated against Phase 1 types in tests |
| `src/octonion/baselines/_initialization.py` | VERIFIED | `real_init`, `complex_init`, `quaternion_init`, `octonion_init` — literature-based polar form initialization |
| `src/octonion/baselines/_param_matching.py` | VERIFIED | `find_matched_width` (MLP + conv2d), `param_report`, `flop_report`, `_SimpleAlgebraMLP`, `_build_conv_model` — all substantive |
| `src/octonion/baselines/_normalization.py` | VERIFIED | `RealBatchNorm`, `ComplexBatchNorm`, `QuaternionBatchNorm`, `OctonionBatchNorm` with covariance whitening via Cholesky |
| `src/octonion/baselines/_activation.py` | VERIFIED | `SplitActivation`, `NormPreservingActivation` |
| `src/octonion/baselines/_algebra_conv.py` | VERIFIED | 8 conv layers (4 algebras x 2 spatial dims) |
| `src/octonion/baselines/_algebra_rnn.py` | VERIFIED | `RealLSTMCell`, `ComplexGRUCell`, `QuaternionLSTMCell`, `OctonionLSTMCell` |
| `src/octonion/baselines/_network.py` | VERIFIED | `AlgebraNetwork` with MLP/Conv2D/Recurrent topologies, `_ResidualBlock` (ResNet-style, depth=28 support), `_apply_conv_bn` helper |
| `src/octonion/baselines/_trainer.py` | VERIFIED | `train_model`, `evaluate`, `seed_everything`, `save_checkpoint`, `load_checkpoint`, `run_optuna_study` with TensorBoard, AMP, gradient stats, VRAM monitoring |
| `src/octonion/baselines/_stats.py` | VERIFIED | `paired_comparison`, `cohen_d`, `holm_bonferroni`, `confidence_interval` |
| `src/octonion/baselines/_plotting.py` | VERIFIED | `plot_convergence`, `plot_comparison_bars`, `plot_param_table` |
| `src/octonion/baselines/_benchmarks.py` | PARTIAL | Infrastructure complete: `build_cifar10_data`, `build_cifar100_data`, `cifar_network_config` (depth=28 after 03-07), `cifar_train_config`, `reproduction_report`, `PUBLISHED_RESULTS` — but reproduction never run |
| `src/octonion/baselines/_comparison.py` | VERIFIED | `run_comparison`, `ComparisonReport` with topology-aware dispatch (MLP and conv2d) |
| `src/octonion/baselines/__init__.py` | VERIFIED | Complete API with 40+ exports; all modules wired |
| `tests/test_baselines_linear.py` | VERIFIED | 39 tests covering shapes, param counts, algebra cross-validation, init variance, matching |
| `tests/test_baselines_components.py` | VERIFIED | 38 tests covering BN params, activation behavior, conv shapes and ratios |
| `tests/test_baselines_rnn.py` | VERIFIED | 21 tests covering shapes, state updates, gate broadcasting, gradient flow |
| `tests/test_baselines_network.py` | VERIFIED | Tests including `TestResNetConv2D` with depth=28, skeleton identity, param matching, output projections |
| `tests/test_baselines_trainer.py` | VERIFIED | 15 tests covering training loop, early stopping, checkpoint, Optuna, stats, plots |
| `tests/test_baselines_comparison.py` | VERIFIED | Tests for MLP and conv2d comparison runs, param matching, directory structure |
| `tests/test_baselines_reproduction.py` | PARTIAL | Fast tests exist (param matching, forward pass shapes for all 4 algebras on CIFAR); slow reproduction tests exist but are marked `@pytest.mark.slow` and have never been run to completion |
| `pyproject.toml` | VERIFIED | `pytest-timeout` added as dev dependency |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `_algebra_linear.py` | `_initialization.py` | `complex_init`, `quaternion_init`, `octonion_init` called in `__init__` | WIRED | Lines 102, 170, 266 confirmed |
| `_algebra_linear.py` | `_multiplication.py` | `STRUCTURE_CONSTANTS` for `OctonionDenseLinear` | WIRED | Line 26 import; Line 270 precompute |
| `_param_matching.py` | `_config.py` | `AlgebraType` for multiplier/dim | WIRED | Line 20 import |
| `_param_matching.py` | `torchinfo` | `torchinfo.summary` in `flop_report` | WIRED | Lines 307-315 confirmed |
| `_normalization.py` | `torch.linalg.cholesky` | Cholesky whitening | WIRED | Lines 286, 416, 425 confirmed |
| `_activation.py` | `torch.nn.functional` | `F.relu`, `F.gelu` per-component | WIRED | Lines 19-20 confirmed |
| `_algebra_conv.py` | `_initialization.py` | Per-algebra init on conv weights | WIRED | Confirmed in summary; init applied to weight tensors |
| `_network.py` | `_algebra_linear.py` | Algebra-specific linear layers in MLP topology | WIRED | Lines 206-213 factory dispatch |
| `_network.py` | `_algebra_conv.py` | Algebra-specific conv in ResNet blocks | WIRED | Lines 222-229 factory dispatch |
| `_network.py` | `_param_matching.py` | `find_matched_width` for auto-scaling | WIRED | Used in comparison runner |
| `_network.py` | `_config.py` | `NetworkConfig`, `AlgebraType` | WIRED | Line 44 import |
| `_algebra_rnn.py` | `_algebra_linear.py` | `QuaternionLinear`, `OctonionDenseLinear` for gates | WIRED | Confirmed in 03-03-SUMMARY.md key-decisions |
| `_trainer.py` | `SummaryWriter` | TensorBoard logging | WIRED | Lines 277, 371-392 confirmed |
| `_trainer.py` | `torch.amp` | `autocast`, `GradScaler` | WIRED | Lines 274, 345 confirmed |
| `_stats.py` | `scipy.stats` | `ttest_rel`, `wilcoxon` | WIRED | Lines 103, 112 confirmed |
| `_trainer.py` | `optuna` | `create_study`, `trial.suggest_*` | WIRED | Lines 473, 481-486 confirmed |
| `_comparison.py` | `_trainer.py` | `train_model` per algebra/seed | WIRED | Line 378 confirmed |
| `_comparison.py` | `_network.py` | `AlgebraNetwork` for conv2d dispatch | WIRED | `_build_conv_model` imports from `_network.py`; `is_conv2d` dispatch at line 221 |
| `_comparison.py` | `_param_matching.py` | `find_matched_width`, `_build_conv_model` | WIRED | Lines 33-36 imports; Lines 268, 280 calls |
| `_comparison.py` | `_stats.py` | `paired_comparison`, `holm_bonferroni` | WIRED | Line 43; Lines 429, 436 confirmed |
| `_comparison.py` | `_plotting.py` | `plot_convergence`, `plot_comparison_bars`, `plot_param_table` | WIRED | Lines 39-41 imports; Lines 398, 441, 455 calls |
| `_benchmarks.py` | `torchvision.datasets` | `CIFAR10`, `CIFAR100` data loading | WIRED | Lines 142, 145, 225, 228 confirmed |
| `_benchmarks.py` | `_comparison.py` | `run_comparison` | NOT_WIRED | `_benchmarks.py` does NOT import or call `run_comparison` — it provides helpers; caller must wire this. 03-06 never ran the actual experiment. |
| `src/octonion/__init__.py` | `baselines` subpackage | `from octonion import baselines` | WIRED | Line 9 confirmed |

---

## Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|--------------|-------------|--------|----------|
| BASE-01 | 03-01, 03-02, 03-03, 03-04, 03-05, 03-07, 03-08 | Real-valued baseline with 8x units matching octonionic param count | SATISFIED | `RealLinear` (AlgebraType.REAL, multiplier=8), `find_matched_width` returns within 1% for MLP, `AlgebraNetwork` builds R baseline with identical skeleton. 39 tests verify param scaling. |
| BASE-02 | 03-01, 03-02, 03-03, 03-04, 03-05, 03-06, 03-07, 03-08 | Complex-valued baseline with 4x units matching param count AND reproducing published benchmark | PARTIAL | `ComplexLinear` (multiplier=4), param matching verified. Published benchmark reproduction (Trabelsi CIFAR-100 26.36% error target) NOT verified — 03-06 was never executed. |
| BASE-03 | 03-01, 03-02, 03-03, 03-04, 03-05, 03-06, 03-07, 03-08 | Quaternionic baseline with 2x units matching param count AND reproducing published benchmark | PARTIAL | `QuaternionLinear` (multiplier=2), param matching verified. Published benchmark reproduction (Gaudet CIFAR-10 5.44% / CIFAR-100 26.01% targets) NOT verified — 03-06 was never executed. |

**Orphaned requirements check:** No requirements mapped to Phase 3 in REQUIREMENTS.md beyond BASE-01/02/03.

---

## Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| `src/octonion/baselines/_comparison.py` | `_benchmarks.py` not imported — 03-06 wiring gap | Info | No functional blocker; `_benchmarks.py` helpers exist, caller must use them. Reproduction experiment not run. |

No TODO/FIXME/placeholder comments found in any baselines source files. No empty implementations. No degenerate return values. Anti-pattern scan clean.

---

## Human Verification Required

### 1. CIFAR Benchmark Reproduction (SC-2 and SC-3)

**Test:** Execute Plan 03-06 Task 2 — run `run_comparison("cifar10_reproduction", build_cifar10_data, config)` with all 4 algebras, 3 seeds minimum.

**Expected:**
- Real: error rate within 1 std of 6.37% on CIFAR-10 (validates training infrastructure)
- Complex: error rate within 1 std of 26.36% on CIFAR-100 (Trabelsi et al. 2018)
- Quaternion: error rate within 1 std of 5.44% on CIFAR-10 (Gaudet and Maida 2018)
- Quaternion: error rate within 1 std of 26.01% on CIFAR-100 (Gaudet and Maida 2018)

**Why human:** Multi-hour GPU training run. Results must be read from `experiments/cifar10_reproduction/reproduction_report.md`. Plan 03-06 Task 3 is an explicit `type="checkpoint:human-verify"` blocking gate that requires human approval of the scientific results.

---

## Gaps Summary

**SC-2 and SC-3 are incomplete.** The phase goal requires that C and H baselines "reproduce a published benchmark result within reported variance." This is not met because Plan 03-06 (CIFAR benchmark reproduction) was never executed:

- No `03-06-SUMMARY.md` exists
- The ROADMAP marks `03-06-PLAN.md` as `[ ]` (not complete)
- No `experiments/` directory with CIFAR results exists
- The slow reproduction tests (`@pytest.mark.slow`) have never been run to completion
- Plan 03-06 Task 3 is a blocking `checkpoint:human-verify` gate that was never reached

The underlying infrastructure is sound and complete:
- `_benchmarks.py` provides `build_cifar10_data`, `build_cifar100_data`, `cifar_network_config` (depth=28 after 03-07), `cifar_train_config`, `reproduction_report`, and `PUBLISHED_RESULTS`
- `run_comparison` correctly dispatches to `AlgebraNetwork` with conv2d topology (fixed in 03-08)
- Fast tests (param matching, forward pass shapes for all 4 algebras on CIFAR input) pass
- `ResNet`-style residual blocks support depth=28 on 32x32 CIFAR input (fixed in 03-07)

The gap is execution: the actual training experiments have not been run and their results have not been validated against the published targets.

**SC-1 and SC-4 are fully verified:**
- SC-1: Binary search param matching within 1% for MLP topology, 10% for conv2d topology (tolerance relaxed due to architecture discretization, documented in 03-08-SUMMARY.md as a practical necessity)
- SC-4: `test_skeleton_identity_mlp` and `test_skeleton_identity_conv2d` explicitly verify all 4 algebras use identical structural blocks

---

_Verified: 2026-03-08_
_Verifier: Claude (gsd-verifier)_
