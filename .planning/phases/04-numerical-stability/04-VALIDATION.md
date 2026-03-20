---
phase: 4
slug: numerical-stability
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-19
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (via Docker: `docker compose run --rm dev uv run pytest`) |
| **Config file** | `pyproject.toml` [tool.pytest.ini_options] |
| **Quick run command** | `docker compose run --rm dev uv run pytest tests/test_numerical_stability.py -v` |
| **Full suite command** | `docker compose run --rm dev uv run pytest tests/ -v --timeout=300` |
| **Estimated runtime** | ~30 seconds (smoke tests only; full analysis script is separate) |

---

## Sampling Rate

- **After every task commit:** Run `docker compose run --rm dev uv run pytest tests/test_numerical_stability.py -x`
- **After every plan wave:** Run `docker compose run --rm dev uv run pytest tests/ -v --timeout=300`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | FOUND-03 SC-4 | unit | `docker compose run --rm dev uv run pytest tests/test_numerical_stability.py::test_stabilizing_norm -x` | ❌ W0 | ⬜ pending |
| 04-01-02 | 01 | 1 | FOUND-03 SC-4 | unit | `docker compose run --rm dev uv run pytest tests/test_numerical_stability.py::test_stabilizing_norm_output_norm -x` | ❌ W0 | ⬜ pending |
| 04-02-01 | 02 | 1 | FOUND-03 SC-1 | smoke | `docker compose run --rm dev uv run pytest tests/test_numerical_stability.py::test_depth_sweep_smoke -x` | ❌ W0 | ⬜ pending |
| 04-02-02 | 02 | 1 | FOUND-03 SC-2 | smoke | `docker compose run --rm dev uv run pytest tests/test_numerical_stability.py::test_condition_number_smoke -x` | ❌ W0 | ⬜ pending |
| 04-02-03 | 02 | 1 | FOUND-03 SC-3 | smoke | `docker compose run --rm dev uv run pytest tests/test_numerical_stability.py::test_dtype_comparison_smoke -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_numerical_stability.py` — smoke tests for all measurement infrastructure (covers FOUND-03 SCs 1-4)
- [ ] `src/octonion/baselines/_stabilization.py` — StabilizingNorm module (must exist before tests)

*Existing infrastructure covers test framework and config — no new framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Analysis plots are readable | FOUND-03 | Visual inspection of matplotlib output | Run `scripts/analyze_stability.py`, check PNG files for correct axes, labels, and data trends |
| SC-4 2x stable depth achieved | FOUND-03 SC-4 | Experimental outcome, not correctness | Run analysis script, check JSON output: mitigation stable_depth >= 2 * baseline stable_depth |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
