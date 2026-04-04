# CLAUDE.md

## Development Environment

**All Python commands MUST run inside the dev container.** The host machine does not have PyTorch/ROCm installed — those live in the container.

### Container Commands

```bash
# Build (first time or after Containerfile changes)
docker compose build

# Run any command inside the container
docker compose run --rm dev <command>

# Examples
docker compose run --rm dev uv sync
docker compose run --rm dev uv run pytest
docker compose run --rm dev uv run pytest tests/test_multiplication.py -v
docker compose run --rm dev uv run python -c "import torch; print(torch.cuda.is_available())"
```

### What runs on the host vs container

| Host (direct bash) | Container (`docker compose run --rm dev`) |
|---|---|
| git commands | `uv sync`, `uv add`, `uv run` |
| file editing | `pytest`, `python` |
| docker compose | Any pip/package operations |
| planning docs | Anything importing `torch` or project code |

### Key Rules

1. **Never run `uv`, `python`, or `pytest` directly on the host** — always prefix with `docker compose run --rm dev`
2. File edits happen on the host (mounted at `/workspace` in container)
3. The container has: ROCm 7.2, PyTorch 2.9.1, Python 3.12, uv
4. First run in a fresh container requires `docker compose run --rm dev uv sync --all-extras --all-groups` to install project deps (core + torch + full research stack + dev tools)

## Makefile

A project-level Makefile wraps common container commands:

```bash
make test              # Run all tests
make list-scripts      # Show available experiment scripts
make run-<script>      # Run scripts/<script>.py in the container
make run-<script> ARGS="--flag"  # Pass arguments to a script
```

Examples:
```bash
make run-run_trie_mnist
make run-run_landscape ARGS="--smoke"
make run-run_cifar_reproduction
```

## Project Structure

```
src/octonion/          # Main package — octonionic algebra for ML
    trie.py            # Self-organizing trie + threshold policies (primary research)
    calculus/          # GHR derivatives, autograd functions, Jacobians
    baselines/         # Fair R/C/H/O comparison networks + training
    landscape/         # Optimization landscape analysis + go/no-go gate
    tasks/             # Synthetic task generators
tests/                 # Pytest test suite (839 tests, Hypothesis property-based)
scripts/               # Experiment runners, benchmarks, analysis, sweeps
docs/thesis/           # Two thesis documents (oct-neural-nets, oct-trie)
.planning/             # Research planning, roadmap, phase tracking
```

## Code Style

- Python 3.12+, type hints on all public APIs
- pytest + hypothesis for property-based testing
- Minimize dependencies beyond PyTorch
