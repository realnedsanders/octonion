---
phase: 3
slug: baseline-implementations
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-08
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest + hypothesis (already configured) |
| **Config file** | `pyproject.toml` [tool.pytest.ini_options] |
| **Quick run command** | `docker compose run --rm dev uv run pytest tests/test_baselines.py tests/test_param_matching.py -x` |
| **Full suite command** | `docker compose run --rm dev uv run pytest -x --tb=short` |
| **Estimated runtime** | ~30 seconds (unit), ~hours (reproduction) |

---

## Sampling Rate

- **After every task commit:** Run `docker compose run --rm dev uv run pytest tests/test_baselines.py tests/test_param_matching.py -x`
- **After every plan wave:** Run `docker compose run --rm dev uv run pytest -x --tb=short`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds (unit tests only; reproduction tests excluded from per-commit sampling)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | BASE-01 | unit | `docker compose run --rm dev uv run pytest tests/test_param_matching.py::test_real_param_match -x` | No -- Wave 0 | pending |
| 03-01-02 | 01 | 1 | BASE-01 | unit | `docker compose run --rm dev uv run pytest tests/test_param_matching.py::test_skeleton_identity -x` | No -- Wave 0 | pending |
| 03-01-03 | 01 | 1 | BASE-02 | unit | `docker compose run --rm dev uv run pytest tests/test_param_matching.py::test_complex_param_match -x` | No -- Wave 0 | pending |
| 03-01-04 | 01 | 1 | BASE-02 | integration | `docker compose run --rm dev uv run pytest tests/test_reproduction.py::test_complex_cifar -x` | No -- Wave 0 | pending |
| 03-01-05 | 01 | 1 | BASE-03 | unit | `docker compose run --rm dev uv run pytest tests/test_param_matching.py::test_quaternion_param_match -x` | No -- Wave 0 | pending |
| 03-01-06 | 01 | 1 | BASE-03 | integration | `docker compose run --rm dev uv run pytest tests/test_reproduction.py::test_quaternion_cifar -x` | No -- Wave 0 | pending |
| 03-01-07 | 01 | 1 | SC-4 | unit | `docker compose run --rm dev uv run pytest tests/test_param_matching.py::test_all_four_share_skeleton -x` | No -- Wave 0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_baselines.py` -- stubs for algebra-specific linear layers, normalization, initialization
- [ ] `tests/test_param_matching.py` -- stubs for BASE-01, BASE-02, BASE-03 param matching + skeleton identity
- [ ] `tests/test_reproduction.py` -- stubs for CIFAR benchmark reproduction (integration, long-running)
- [ ] `tests/test_trainer.py` -- stubs for training utility, checkpointing, gradient logging
- [ ] `tests/test_comparison.py` -- stubs for comparison runner, statistical tests, manifest
- [ ] Dependencies: `docker compose run --rm dev uv add torchinfo optuna tensorboard matplotlib seaborn`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Published result reproduction within 1 std | BASE-02, BASE-03 | Long-running GPU training (~hours), flaky if run per-commit | Run full reproduction after implementation complete; compare final metrics to published tables |

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
