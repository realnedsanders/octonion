# Installation

## Step 1: Install PyTorch

The `octonion` library requires PyTorch but does not install it automatically, because the correct PyTorch version depends on your hardware (CPU, CUDA, ROCm).

Install PyTorch first following the [official instructions](https://pytorch.org/get-started/locally/):

```bash
# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.x
pip install torch --index-url https://download.pytorch.org/whl/cu124

# ROCm 6.x (AMD GPUs)
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
```

## Step 2: Install octonion

```bash
# Core library (algebra + trie)
pip install octonion

# Full research stack (includes visualization, optimization, benchmarks)
pip install octonion[full]
```

## For development

```bash
git clone https://github.com/realnedsanders/octonion-computation-substrate.git
cd octonion-computation-substrate
pip install -e ".[dev]"
```

Or using the Docker dev container (recommended for ROCm):

```bash
docker compose build
docker compose run --rm dev uv sync
```
