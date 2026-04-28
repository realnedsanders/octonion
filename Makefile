DOCKER := docker compose run --rm dev
UV := $(DOCKER) uv run
PYTHON := $(UV) python

# ── Tests ────────────────────────────────────────────────────────────

.PHONY: test
test: ## Run all tests
	$(UV) pytest

.PHONY: lint
lint: ## Run ruff linter
	$(UV) ruff check --fix .
	$(UV) ruff format .

typecheck:
	$(UV) mypy .

.PHONY: docs
docs: ## Build docs locally (catches docstring issues before CI)
	$(UV) --with mkdocs-material --with "mkdocstrings[python]" --with mkdocs-gen-files --with mkdocs-literate-nav mkdocs build --strict

# ── Scripts ──────────────────────────────────────────────────────────
# Auto-generates a `run-<stem>` target for every scripts/*.py file.
# Example: `make run-run_cifar_reproduction` runs scripts/run_cifar_reproduction.py

SCRIPTS := $(wildcard scripts/*.py)
SCRIPT_TARGETS := $(patsubst scripts/%.py,run-%,$(filter-out scripts/__init__.py,$(SCRIPTS)))

.PHONY: $(SCRIPT_TARGETS)
$(SCRIPT_TARGETS): run-%: scripts/%.py
	$(UV) python $< $(ARGS)

.PHONY: list-scripts
list-scripts: ## List available run-* targets
	@echo "Available script targets (pass args with ARGS=\"...\"):"
	@for t in $(sort $(SCRIPT_TARGETS)); do echo "  make $$t"; done

# ── Help ─────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-24s %s\n", $$1, $$2}'
	@echo ""
	@echo "Script targets: run 'make list-scripts' to see all."
	@echo "Pass arguments:  make run-run_landscape ARGS=\"--smoke\""
