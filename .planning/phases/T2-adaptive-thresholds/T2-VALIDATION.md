---
phase: T2
slug: adaptive-thresholds
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-29
---

# Phase T2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `docker compose run --rm dev uv run pytest tests/test_trie.py -v --timeout=60` |
| **Full suite command** | `docker compose run --rm dev uv run pytest tests/ -v --timeout=300` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `docker compose run --rm dev uv run pytest tests/test_trie.py -v --timeout=60`
- **After every plan wave:** Run `docker compose run --rm dev uv run pytest tests/ -v --timeout=300`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| T2-01-01 | 01 | 1 | SC-1 | unit | `pytest tests/test_threshold_policy.py -v` | ❌ W0 | ⬜ pending |
| T2-01-02 | 01 | 1 | SC-1 | unit | `pytest tests/test_trie.py -v` | ✅ | ⬜ pending |
| T2-02-01 | 02 | 1 | SC-5 | integration | `pytest tests/test_sweep.py -v` | ❌ W0 | ⬜ pending |
| T2-03-01 | 03 | 2 | SC-1,SC-2 | integration | `python scripts/run_trie_threshold_sweep.py --quick` | ❌ W0 | ⬜ pending |
| T2-04-01 | 04 | 3 | SC-3 | unit | `pytest tests/test_associator_theory.py -v` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_threshold_policy.py` — stubs for ThresholdPolicy abstraction tests
- [ ] `tests/test_sweep.py` — stubs for parallel sweep framework tests
- [ ] Existing `tests/test_trie.py` covers basic trie operations

*Existing infrastructure covers basic trie requirements. New test files needed for threshold policies and sweep framework.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Theory proof correctness | SC-3 | Mathematical proof review requires human judgment | Review oct-trie.tex theory section |
| Meta-trie convergence visual inspection | SC-4 | Convergence curves require visual assessment of stability | Review convergence plots in results/ |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
