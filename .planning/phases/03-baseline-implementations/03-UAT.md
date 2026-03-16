---
status: complete
phase: 03-baseline-implementations
source: [03-01-SUMMARY.md, 03-02-SUMMARY.md, 03-03-SUMMARY.md, 03-04-SUMMARY.md, 03-05-SUMMARY.md, 03-06-SUMMARY.md, 03-07-SUMMARY.md, 03-08-SUMMARY.md, 03-09-SUMMARY.md, 03-10-SUMMARY.md, 03-11-SUMMARY.md, 03-12-SUMMARY.md]
started: 2026-03-16T03:18:25Z
updated: 2026-03-16T03:28:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Full test suite is green
expected: Run `docker compose run --rm dev uv run pytest` (no flags). Expect 601 tests collected, 0 failures, 0 errors. A small number of tests may be deselected (CUDA-only skipped on CPU) — that is fine. The key result is 0 failed tests.
result: issue
reported: "tests/test_perf_equivalence.py:171: TypeError: _tril_to_symmetric() missing 2 required positional arguments: 'rows' and 'cols'"
severity: major

### 2. Baselines package imports cleanly
expected: Run `docker compose run --rm dev uv run python -c "from octonion.baselines import AlgebraNetwork, AlgebraType, train_model, run_comparison, ComparisonReport, find_matched_width, param_report, flop_report, RealBatchNorm, ComplexBatchNorm, QuaternionBatchNorm, OctonionBatchNorm, SplitActivation, NormPreservingActivation; print('OK')"`. Expect `OK` printed with no ImportError.
result: pass

### 3. AlgebraNetwork MLP forward pass for all algebras
expected: Run with input_dim=32 matching the input tensor. All four algebras (R/C/H/O) produce [4, 10] output with no errors.
result: pass

### 4. AlgebraNetwork Conv2D forward pass (CIFAR-shaped input)
expected: Run with input_dim=3 (RGB channels). All four algebras process [2, 3, 32, 32] CIFAR input and return [2, 10] output with no errors.
result: pass

### 5. run_comparison executes and produces report
expected: 2-epoch Real vs Complex comparison with match_params=False returns a ComparisonReport. report.algebras is list[str] short names.
result: pass

### 6. CIFAR data loaders work
expected: build_cifar10_data returns [32, 3, 32, 32] batches. input_dim=3 (RGB channels), output_dim=10, channels=3.
result: pass

### 7. CIFAR reproduction script has --use-amp and --compile flags
expected: --help shows --use-amp, --compile, --no-match-params, --ref-hidden, --seeds, --epochs, --depth flags.
result: pass

### 8. Profiling script runs without errors
expected: Runs without Python errors and shows timing table. REAL should be ~3.7M params (same-width protocol, ref_hidden=4).
result: issue
reported: "Real was 59,471,626 params — way too many. Should be ~3.7M."
severity: major

### 9. AMP-safe BN — no NaN under autocast
expected: Run `docker compose run --rm dev uv run python -c "import torch; from octonion.baselines import QuaternionBatchNorm, OctonionBatchNorm;\nif not torch.cuda.is_available(): print('CUDA not available, skip'); exit(0);\nfor BN, dim in [(QuaternionBatchNorm, 4), (OctonionBatchNorm, 8)]:\n    bn = BN(16).cuda();\n    x = torch.randn(64, 16, dim).cuda();\n    with torch.amp.autocast(device_type='cuda'):\n        out = bn(x);\n    assert not torch.isnan(out).any(), f'{BN.__name__} produced NaN!';\n    print(f'{BN.__name__}: no NaN, dtype={out.dtype} OK')"`. Expect `QuaternionBatchNorm: no NaN ... OK` and `OctonionBatchNorm: no NaN ... OK`. If CUDA is unavailable, the test prints a skip message.
result: issue
reported: "Traceback (most recent call last): File \"<string>\", line 13, in <module> AttributeError: type object 'QuaternionBatchNorm' has no attribute 'name'"
severity: major

### 10. CIFAR training is running (or completed)
expected: Run `docker ps | grep octonion` or check if training has completed via `ls experiments/cifar10_reproduction/ 2>/dev/null`. Either see a running Docker container for the training job, OR see completed experiment directories (one per algebra/seed) in `experiments/cifar10_reproduction/`. If training completed, `cat experiments/cifar10_reproduction/reproduction_report.md` should show results for all 4 algebras.
result: pass

## Summary

total: 10
passed: 7
issues: 3
pending: 0
skipped: 0

## Gaps

- truth: "Full test suite passes with 0 failures"
  status: failed
  reason: "User reported: tests/test_perf_equivalence.py:171: TypeError: _tril_to_symmetric() missing 2 required positional arguments: 'rows' and 'cols'"
  severity: major
  test: 1
  artifacts: []
  missing: []

- truth: "Profiling script shows ~3.7M params for REAL baseline under same-width protocol"
  status: failed
  reason: "User reported: Real was 59,471,626 params — way too many. Should be ~3.7M."
  severity: major
  test: 8
  artifacts: []
  missing: []

- truth: "AMP-safe BN runs without errors (no NaN, no AttributeError)"
  status: failed
  reason: "User reported: AttributeError: type object 'QuaternionBatchNorm' has no attribute 'name'"
  severity: major
  test: 9
  artifacts: []
  missing: []
