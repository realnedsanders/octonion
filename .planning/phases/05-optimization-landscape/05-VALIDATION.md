---
phase: 5
slug: optimization-landscape
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-20
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x + hypothesis |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `docker compose run --rm dev uv run pytest tests/test_optimization_landscape.py -x` |
| **Full suite command** | `docker compose run --rm dev uv run pytest tests/ -v` |
| **Estimated runtime** | ~120 seconds |

---

## Sampling Rate

- **After every task commit:** Run `docker compose run --rm dev uv run pytest tests/test_optimization_landscape.py -x`
- **After every plan wave:** Run `docker compose run --rm dev uv run pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | FOUND-04 | unit | `docker compose run --rm dev uv run pytest tests/test_optimization_landscape.py -k "gradient"` | ❌ W0 | ⬜ pending |
| 05-01-02 | 01 | 1 | FOUND-04 | unit | `docker compose run --rm dev uv run pytest tests/test_optimization_landscape.py -k "hessian"` | ❌ W0 | ⬜ pending |
| 05-02-01 | 02 | 2 | FOUND-04 | integration | `docker compose run --rm dev uv run pytest tests/test_optimization_landscape.py -k "convergence"` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_optimization_landscape.py` — stubs for gradient variance, Hessian analysis, convergence experiments
- [ ] `tests/conftest.py` — shared fixtures (already exists, may need extension)
- [ ] `geoopt` and `pytorch_optimizer` — new dependencies

*Existing test infrastructure (pytest, hypothesis, conftest.py) covers base requirements. Wave 0 adds phase-specific test stubs and new dependencies.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Loss landscape visualization quality | FOUND-04 | Visual inspection of plots | Review generated PNGs in results/optimization/ for readability and correct labeling |
| Publishable negative result quality | FOUND-04 | Qualitative assessment | Review generated analysis report for scientific rigor and completeness |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
