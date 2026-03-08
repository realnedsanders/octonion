---
status: resolved
trigger: "pytest can't find octonion module after docker compose build and uv sync"
created: 2026-03-08T07:00:00Z
updated: 2026-03-08T07:30:00Z
---

## Current Focus

hypothesis: Dev dependencies declared in wrong pyproject.toml section for uv
test: Compare uv sync vs uv sync --extra dev
expecting: --extra dev installs pytest/hypothesis into venv, fixing import
next_action: Return diagnosis (find_root_cause_only mode)

## Symptoms

expected: `docker compose run --rm dev uv run pytest` runs all 209 tests
actual: ModuleNotFoundError: No module named 'octonion' during conftest.py load
errors: "ModuleNotFoundError: No module named 'octonion'" at tests/conftest.py:96
reproduction: docker compose build && docker compose run --rm dev uv sync && docker compose run --rm dev uv run pytest
started: Discovered during UAT

## Eliminated

- hypothesis: Package not installed in editable mode
  evidence: uv pip list shows "octonion 0.1.0 /workspace" with editable project location. _octonion.pth exists in site-packages pointing to /workspace/src. uv run python -c 'import octonion' succeeds.
  timestamp: 2026-03-08T07:00:00Z

- hypothesis: src layout misconfigured in pyproject.toml
  evidence: [tool.hatch.build.targets.wheel] packages = ["src/octonion"] is correct. The .pth file contains /workspace/src which properly puts src/octonion on the import path.
  timestamp: 2026-03-08T07:00:00Z

## Evidence

- timestamp: 2026-03-08T07:00:00Z
  checked: uv pip list in container after uv sync
  found: pytest, hypothesis, hypothesis-torch, numpy, ruff, mypy are all MISSING from installed packages. Only torch and its transitive deps installed.
  implication: uv sync is not installing dev dependencies

- timestamp: 2026-03-08T07:00:00Z
  checked: pyproject.toml dependency declarations
  found: Dev tools are under [project.optional-dependencies] dev (PEP 621 extras), NOT under [dependency-groups] dev (PEP 735 dependency groups)
  implication: uv sync includes [dependency-groups] dev by default but ignores [project.optional-dependencies] extras unless --extra flag is used

- timestamp: 2026-03-08T07:00:00Z
  checked: uv run python -m pytest vs uv run pytest
  found: "uv run python -m pytest" says "No module named pytest" (not installed in venv). "uv run pytest" runs but fails on octonion import.
  implication: uv run pytest falls back to ephemeral tool resolution (finds pytest outside venv), which has no access to the project's .pth editable install

- timestamp: 2026-03-08T07:00:00Z
  checked: sys.path from uv run python vs the ephemeral pytest
  found: uv run python includes /workspace/src via .pth file. Ephemeral pytest environment does NOT include /workspace/src.
  implication: The ephemeral pytest can't import octonion because it's not in its sys.path

- timestamp: 2026-03-08T07:00:00Z
  checked: uv sync --extra dev --dry-run
  found: Would install 14 additional packages including pytest==9.0.2, hypothesis==6.151.9, numpy==2.4.2
  implication: Using --extra dev (or migrating to [dependency-groups]) would fix the issue

## Resolution

root_cause: Dev dependencies (pytest, hypothesis, numpy, etc.) are declared under [project.optional-dependencies] dev in pyproject.toml instead of [dependency-groups] dev. uv sync includes dependency groups by default but NOT optional extras. This means `uv sync` installs only torch, leaving pytest uninstalled. When `uv run pytest` is invoked, uv resolves pytest ephemerally as a tool (outside the project venv), and that ephemeral environment lacks the .pth file that maps /workspace/src for the editable octonion install, causing ModuleNotFoundError.
fix:
verification:
files_changed: []
