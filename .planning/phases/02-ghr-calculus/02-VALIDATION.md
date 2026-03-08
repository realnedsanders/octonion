---
phase: 2
slug: ghr-calculus
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-08
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest + hypothesis (already configured) |
| **Config file** | pyproject.toml [tool.pytest.ini_options] |
| **Quick run command** | `docker compose run --rm dev uv run pytest tests/test_calculus.py -x` |
| **Full suite command** | `docker compose run --rm dev uv run pytest -x` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `docker compose run --rm dev uv run pytest tests/test_calculus.py -x`
- **After every plan wave:** Run `docker compose run --rm dev uv run pytest -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| SC-1 | 01 | 1 | FOUND-02 | unit | `docker compose run --rm dev uv run pytest tests/test_calculus.py::test_gradcheck_octonion_linear -x` | ❌ W0 | ⬜ pending |
| SC-2 | 02 | 2 | FOUND-02 | integration | `docker compose run --rm dev uv run pytest tests/test_calculus.py::test_parenthesization_exhaustive -x` | ❌ W0 | ⬜ pending |
| SC-3 | 02 | 2 | FOUND-02 | unit | `docker compose run --rm dev uv run pytest tests/test_calculus.py::test_naive_vs_correct_differs -x` | ❌ W0 | ⬜ pending |
| SC-4 | 02 | 2 | FOUND-02 | manual | `docker compose run --rm dev uv run pytest tests/test_calculus.py::test_gpu_cpu_parity -x` | ❌ W0 | ⬜ pending |
| TRIP-1 | 01 | 1 | FOUND-02 | unit | `docker compose run --rm dev uv run pytest tests/test_calculus.py::TestJacobians -x` | ❌ W0 | ⬜ pending |
| TRIP-2 | 01 | 1 | FOUND-02 | unit | `docker compose run --rm dev uv run pytest tests/test_calculus.py::TestAutogradVsAnalytic -x` | ❌ W0 | ⬜ pending |
| TRIP-3 | 01 | 1 | FOUND-02 | unit | `docker compose run --rm dev uv run pytest tests/test_calculus.py::test_gradgradcheck -x` | ❌ W0 | ⬜ pending |
| GHR-1 | 01 | 1 | FOUND-02 | unit | `docker compose run --rm dev uv run pytest tests/test_calculus.py::TestWirtingerPair -x` | ❌ W0 | ⬜ pending |
| GHR-2 | 01 | 1 | FOUND-02 | unit | `docker compose run --rm dev uv run pytest tests/test_calculus.py::TestAnalyticity -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_calculus.py` — stubs for SC-1, SC-2, SC-3, SC-4, TRIP-1, TRIP-2, TRIP-3, GHR-1, GHR-2
- [ ] `tests/conftest.py` additions — strategies for composition trees, gradcheck-friendly octonion tensors (small norm, requires_grad=True)
- [ ] `src/octonion/calculus/__init__.py` — new submodule package init

*Framework install: none needed (pytest + hypothesis already present)*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| GPU/CPU parity (SC-4) | FOUND-02 | Requires ROCm GPU hardware | Run `docker compose run --rm dev uv run pytest tests/test_calculus.py::test_gpu_cpu_parity -x` on GPU-equipped machine |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
