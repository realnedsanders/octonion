---
phase: 3
slug: baseline-implementations
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-03-08
updated: 2026-03-08
---

# Phase 3 -- Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest + hypothesis (already configured) |
| **Config file** | `pyproject.toml` [tool.pytest.ini_options] |
| **Quick run command** | `docker compose run --rm dev uv run pytest tests/test_baselines_linear.py tests/test_baselines_components.py tests/test_baselines_rnn.py tests/test_baselines_network.py -x` |
| **Full suite command** | `docker compose run --rm dev uv run pytest -x --tb=short` |
| **Estimated runtime** | ~30 seconds (unit), ~hours (reproduction) |

---

## Sampling Rate

- **After every task commit:** Run `docker compose run --rm dev uv run pytest tests/test_baselines_linear.py tests/test_baselines_components.py tests/test_baselines_rnn.py tests/test_baselines_network.py -x`
- **After every plan wave:** Run `docker compose run --rm dev uv run pytest -x --tb=short`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds (unit tests only; reproduction tests excluded from per-commit sampling)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File | Status |
|---------|------|------|-------------|-----------|-------------------|------|--------|
| 03-01-01 | 01 | 1 | BASE-01 | unit | `docker compose run --rm dev uv run python -c "from octonion.baselines import AlgebraType, NetworkConfig, TrainConfig, ComparisonConfig; print('OK')"` | N/A (import check) | pending |
| 03-01-02 | 01 | 1 | BASE-01/02/03 | unit | `docker compose run --rm dev uv run pytest tests/test_baselines_linear.py -x -v` | tests/test_baselines_linear.py | pending |
| 03-02-01 | 02 | 2 | BASE-01/02/03 | unit | `docker compose run --rm dev uv run pytest tests/test_baselines_components.py -x -v -k "norm or activation"` | tests/test_baselines_components.py | pending |
| 03-02-02 | 02 | 2 | BASE-01/02/03 | unit | `docker compose run --rm dev uv run pytest tests/test_baselines_components.py -x -v` | tests/test_baselines_components.py | pending |
| 03-03-01 | 03 | 3 | BASE-01/02/03 | unit | `docker compose run --rm dev uv run pytest tests/test_baselines_rnn.py -x -v` | tests/test_baselines_rnn.py | pending |
| 03-03-02 | 03 | 3 | BASE-01/02/03 | unit | `docker compose run --rm dev uv run pytest tests/test_baselines_network.py -x -v` | tests/test_baselines_network.py | pending |
| 03-04-01 | 04 | 2 | BASE-01/02/03 | unit | `docker compose run --rm dev uv run pytest tests/test_baselines_trainer.py -x -v` | tests/test_baselines_trainer.py | pending |
| 03-04-02 | 04 | 2 | BASE-01/02/03 | unit | `docker compose run --rm dev uv run pytest tests/test_baselines_trainer.py -x -v` | tests/test_baselines_trainer.py | pending |
| 03-05-01 | 05 | 4 | BASE-01/02/03 | unit | `docker compose run --rm dev uv run pytest tests/test_baselines_comparison.py -x -v` | tests/test_baselines_comparison.py | pending |
| 03-05-02 | 05 | 4 | BASE-01/02/03 | unit | `docker compose run --rm dev uv run python -c "from octonion.baselines import AlgebraNetwork, run_comparison, AlgebraType, flop_report, run_optuna_study; print('OK')"` | N/A (import check) | pending |
| 03-06-01 | 06 | 5 | BASE-02/03 | unit | `docker compose run --rm dev uv run python -c "from octonion.baselines._benchmarks import build_cifar10_data, cifar_network_config; print('OK')"` | N/A (import check) | pending |
| 03-06-02 | 06 | 5 | BASE-02/03 | integration | `docker compose run --rm dev uv run pytest tests/test_baselines_reproduction.py -x -v -k "not slow"` | tests/test_baselines_reproduction.py | pending |
| 03-06-03 | 06 | 5 | BASE-02/03 | checkpoint | Task 3 human-verify gates slow reproduction results | N/A (human verify) | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_baselines_linear.py` -- stubs for algebra-specific linear layers, param matching, flop_report
- [ ] `tests/test_baselines_components.py` -- stubs for normalization, activation, conv layers
- [ ] `tests/test_baselines_rnn.py` -- stubs for RNN cell output shapes, state management, gate correctness
- [ ] `tests/test_baselines_network.py` -- stubs for AlgebraNetwork topologies, skeleton identity, param matching
- [ ] `tests/test_baselines_trainer.py` -- stubs for training utility, checkpointing, gradient logging, Optuna study, stats, plotting
- [ ] `tests/test_baselines_comparison.py` -- stubs for comparison runner, statistical tests, manifest
- [ ] `tests/test_baselines_reproduction.py` -- stubs for CIFAR benchmark reproduction (integration, long-running)
- [ ] Dependencies: `docker compose run --rm dev uv add torchinfo optuna tensorboard matplotlib seaborn scipy`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Published result reproduction within 1 std | BASE-02, BASE-03 | Long-running GPU training (~hours), flaky if run per-commit | Run full reproduction after implementation complete; compare final metrics to published tables; gated by Plan 03-06 Task 3 checkpoint |

---

## Validation Sign-Off

- [x] All tasks have automated verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter
- [x] Test file names match actual plan-created files

**Approval:** approved
