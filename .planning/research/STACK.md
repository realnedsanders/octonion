# Technology Stack

**Project:** Octonionic Computation Substrate -- Research PoC
**Researched:** 2026-03-07
**Overall Confidence:** MEDIUM (mature base stack, nascent hypercomplex ecosystem)

---

## Executive Assessment

The hypercomplex ML ecosystem is immature. No production-grade octonionic ML library exists. Quaternionic libraries exist but are dormant (hTorch: last commit 2021; Pytorch-Quaternion-Neural-Networks: last commit 2019). The parameterized hypercomplex approach (HyperNets/PHM) is the most active but learns algebra rules from data rather than encoding them -- the opposite of what this project needs. The practical consequence: **core octonionic algebra, autograd integration, and G2-equivariant layers must be built from scratch.** The surrounding ecosystem (PyTorch ROCm, geoopt for hyperbolic geometry, geomstats for differential geometry, experiment tracking) is mature and well-supported.

---

## Recommended Stack

### GPU Compute Platform

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| ROCm | 7.2.0 (stable) | GPU compute platform | Latest stable release (2026-02-18). RX 7900 XTX (gfx1100) officially supported. Supports PyTorch 2.9.1, 2.8.0, 2.7.1. | HIGH |
| Ubuntu | 24.04.3 LTS | Host OS | Officially supported by ROCm 7.2.0. Most community examples and Docker images target Ubuntu. | HIGH |

**ROCm installation approach:** Use the `rocm/pytorch` Docker image from Docker Hub as the base environment. This avoids the "dependency hell" of matching pytorch-triton-rocm versions with the underlying ROCm driver. For RDNA3 (RX 7900 XTX), set `HSA_OVERRIDE_GFX_VERSION=11.0.0` if needed.

**Install command (stable):**
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1
```

**Install command (nightly, for latest ROCm 7.2 support):**
```bash
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.2
```

### Core Framework

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| PyTorch | 2.10.0 stable (or latest nightly with ROCm 7.2) | Deep learning framework | Industry standard for ML research. ROCm support via `torch.cuda` API (HIP reuses CUDA interfaces -- minimal code changes). Custom autograd functions work identically on ROCm. `torch.compile` with Triton backend functional on AMD. | HIGH |
| Python | 3.12 | Language runtime | Latest stable Python supported by PyTorch. Good typing support, f-strings, match statements. | HIGH |

**ROCm compatibility notes for PyTorch:**
- `torch.cuda` API works identically on ROCm -- no code changes needed for standard operations
- Custom `torch.autograd.Function` subclasses work on ROCm without modification
- TensorFloat-32 (TF32) is NOT supported on ROCm -- irrelevant for this project (we need float64 precision for numerical stability of octonionic operations)
- `torch.compile` with Triton backend works but may need manual tuning for AMD wavefront size (64 threads vs NVIDIA's 32)
- hipFFT/rocFFT plan caching is not supported (minor limitation)

### Octonionic Algebra (BUILD FROM SCRATCH)

| Component | Approach | Why | Confidence |
|-----------|----------|-----|------------|
| Octonion tensor type | Custom class wrapping 8 real-valued PyTorch tensors | No usable library exists. pyoctonion is CPU-only pure Python. `hypercomplex` package is Cayley-Dickson but not GPU/autograd aware. Must build GPU-accelerated version with PyTorch autograd integration. | HIGH |
| Octonion multiplication | Custom `torch.autograd.Function` with explicit Fano plane multiplication table | Non-associativity means we cannot rely on generic hypercomplex multiplication. Must encode the specific 480 valid multiplication tables (or canonical choice) and handle parenthesization-dependent gradients. | HIGH |
| GHR calculus gradients | Custom backward pass implementation | GHR calculus for quaternions exists in literature (Xu et al., 2015) but no implementation for octonions. Extension to octonions is a research contribution of this project. Must handle non-associative chain rule carefully. | MEDIUM |
| G2-equivariant layers | Custom implementation | No library supports G2 equivariance. escnn supports arbitrary compact groups in theory but focuses on E(2)/E(3) subgroups. Must implement G2 representation theory and constraint solving from scratch. | MEDIUM |
| Fano plane decomposition | Custom implementation | 7 quaternionic subalgebras for structured computation. No existing library. Pure math implementation. | HIGH |

**Critical implementation note:** PyTorch autograd assumes associative chain rule. For octonionic backpropagation, you must implement custom `torch.autograd.Function` classes that explicitly handle the parenthesization ordering in the backward pass. The `ctx` object should store the specific multiplication ordering used in forward to reconstruct correct gradients.

### Hyperbolic Geometry

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| geoopt | 0.5.1 (install from master) | Riemannian optimization, hyperbolic manifolds | Provides PoincareBall, Lorentz (hyperboloid), and Stereographic models. Drop-in replacement optimizers (RiemannianSGD, RiemannianAdam). Pure PyTorch -- works on ROCm without modification. Required for hyperboloid-octonionic hybrid model (thesis Option B). | HIGH |
| geomstats | 2.8.0 | Differential geometry computations | Supports PyTorch backend. Provides SO(n), SE(n) Lie groups, hyperbolic spaces, Riemannian metrics with exponential/log maps. Useful for G2 manifold geometry research and validation. 77 contributors, actively maintained. | MEDIUM |

**Install:**
```bash
# geoopt -- install from master for latest fixes
pip install git+https://github.com/geoopt/geoopt.git

# geomstats
pip install geomstats
```

**ROCm compatibility:** Both are pure PyTorch/NumPy -- no custom CUDA kernels. Full ROCm compatibility via PyTorch's HIP translation layer.

### Geometric Algebra (Reference / Validation)

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| clifford | 1.5.1 | Geometric algebra reference implementation | CPU-only but mathematically rigorous. Use for validating octonionic algebra implementations against known Clifford algebra identities. Not for training -- for testing and verification. | HIGH |
| kingdon | 1.3.0 | GA with PyTorch tensor support | Input-type-agnostic GA library. Supports PyTorch tensors. Symbolically optimized. Useful for cross-validating octonionic operations. Published 2025 (arxiv 2503.10451). | MEDIUM |

**Install:**
```bash
pip install clifford kingdon
```

**Note:** These are validation/reference tools, not training infrastructure. Use them to verify your custom octonion implementation produces correct results, then discard for actual experiments.

### Quaternion Baselines (Reference Implementations)

| Technology | Source | Purpose | Why | Confidence |
|------------|--------|---------|-----|------------|
| Pytorch-Quaternion-Neural-Networks | github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks | Quaternion baseline layers | Parcollet et al. (2019) -- the reference implementation cited in the thesis. Provides QuaternionLinear, QuaternionConv. Dormant (last commit 2019) but code is functional and well-understood. Extract core ops, don't pip install. | MEDIUM |
| SpeechBrain quaternion layers | speechbrain.readthedocs.io | Additional quaternion reference | Production-quality quaternion Linear, Conv, RNN, and Normalization layers. More maintained than standalone repos. | MEDIUM |

**Approach:** Do NOT depend on these as pip packages. Copy the relevant quaternion operation code (quaternion_ops.py from Orkis-Research) into your project, adapt to your tensor layout, and use as baseline comparisons. These repos are abandonware but the math is correct.

### Experiment Infrastructure

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Weights & Biases (wandb) | >=0.19.11 | Experiment tracking | Tracks AMD GPU metrics via rocm-smi (utilization, memory, temperature, power). Critical for comparing octonion vs quaternion vs complex vs real baselines across matched experiments. | HIGH |
| pytest | latest | Testing framework | Standard Python testing. Combine with hypothesis for property-based tests of algebraic properties. | HIGH |
| hypothesis | latest | Property-based testing | Test algebraic properties (norm preservation, Moufang identities, alternativity) by generating random octonions and verifying invariants. PyTorch itself uses hypothesis. | HIGH |
| hypothesis-torch | latest | PyTorch tensor strategies for hypothesis | Generates random tensors with controlled shape, dtype, device for testing octonionic operations on GPU. | HIGH |

**Install:**
```bash
pip install wandb pytest hypothesis hypothesis-torch
```

### Numerical Computation

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| NumPy | latest | CPU array operations, test oracles | Reference implementations, data preprocessing. Use float64 as ground truth for numerical stability comparisons. | HIGH |
| SciPy | latest | Optimization, linear algebra, rotations | scipy.spatial.transform.Rotation for SO(3) quaternion operations. scipy.linalg for eigendecomposition needed in G2 representation theory. scipy.optimize for non-gradient optimization baselines. | HIGH |
| numpy-quaternion | 2023.0.0 | CPU quaternion operations | High-quality quaternion arithmetic for NumPy. Use as CPU reference oracle for quaternion baseline validation. | HIGH |

**Install:**
```bash
pip install numpy scipy numpy-quaternion
```

### Visualization & Analysis

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| matplotlib | latest | Plotting | Loss landscapes, optimization trajectories, embedding visualizations. | HIGH |
| seaborn | latest | Statistical visualization | Box plots, violin plots for baseline comparisons with statistical significance. | HIGH |
| plotly | latest | Interactive 3D visualization | Visualize hyperbolic embeddings, octonionic manifold projections in 3D. | MEDIUM |

**Install:**
```bash
pip install matplotlib seaborn plotly
```

### Reproducibility

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| hydra-core | latest | Configuration management | Structured experiment configs with overrides. Enables reproducible hyperparameter sweeps. | MEDIUM |
| tensorboard | latest | Training monitoring (backup to wandb) | Lightweight alternative if wandb is overkill for quick experiments. Works with PyTorch out of box. | MEDIUM |

---

## What Must Be Built From Scratch

This is the most critical section. The following components have NO usable existing implementation:

### 1. Octonionic Tensor Library
**What:** A GPU-accelerated octonion tensor type with full autograd support.
**Why from scratch:** pyoctonion is CPU-only Python. `hypercomplex` uses Cayley-Dickson construction but is not GPU-aware or autograd-aware. KHNN claims arbitrary algebra support but has 3 commits and experimental PyTorch support (Dense + Conv2D only).
**Scope:** ~2000 lines. Multiplication, conjugation, norm, inverse, exp, log. Custom autograd for each.
**Key challenge:** Non-associative multiplication requires explicit parenthesization tracking in backward pass.

### 2. GHR Calculus Gradient Engine
**What:** Generalized HR calculus extended from quaternions to octonions for backpropagation.
**Why from scratch:** GHR calculus exists only for quaternions in literature. Octonion extension is an open research problem.
**Scope:** Research contribution. ~500-1000 lines once the math is worked out.
**Key challenge:** Non-associativity means the quaternion GHR formulas don't directly generalize. May need to work within quaternionic subalgebras and compose.

### 3. G2-Equivariant Neural Network Layers
**What:** Layers whose transformations commute with the G2 automorphism group of octonions.
**Why from scratch:** escnn handles E(2)/E(3) equivariance but not exceptional Lie groups. No library supports G2. The 14-dimensional G2 structure requires custom representation theory implementation.
**Scope:** ~1500 lines. Requires implementing G2 generators, Lie algebra basis, and kernel constraint solving.
**Key challenge:** G2 has no "nice" matrix representation like SO(3). Must work with the 7x7 fundamental representation.

### 4. Hyperboloid-Octonionic Projection Layer
**What:** Mapping between octonionic representations and the hyperboloid model of hyperbolic space.
**Why from scratch:** geoopt provides hyperboloid operations. The novel part is the octonionic-to-hyperbolic projection and its gradient.
**Scope:** ~500 lines. Leverages geoopt for the hyperbolic side.
**Key challenge:** Numerical stability of the projection (thesis section 9.7 central open problem).

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| DL Framework | PyTorch | TensorFlow/JAX | PyTorch has best ROCm support via HIP translation. JAX ROCm support is experimental. TF ROCm is poorly maintained. Custom autograd functions are cleaner in PyTorch. |
| Hypercomplex lib | Build from scratch | HyperNets (PHM layers) | PHM learns algebra rules from data via Kronecker products. This project needs to ENCODE specific octonionic algebra, not learn it. Wrong abstraction. |
| Hypercomplex lib | Build from scratch | KHNN | 3 commits, experimental PyTorch support, copy-paste installation. Not production-ready. No octonion-specific handling of non-associativity. |
| Hypercomplex lib | Build from scratch | hTorch | Quaternion-only. Last commit Nov 2021. No octonion support. Experimental warning in README. |
| Hyperbolic geometry | geoopt | Custom implementation | geoopt is well-tested, provides Riemannian optimizers, and works on ROCm. No reason to rebuild. |
| Geometric algebra | clifford (reference only) | torch_ga | torch_ga is less maintained. clifford is the gold standard for correctness verification. |
| Equivariant NN | Build G2 from scratch | escnn | escnn is designed for E(2)/E(3). Adapting it to G2 would be more work than building G2 layers directly, since G2 doesn't decompose into standard rotation group representations. |
| Experiment tracking | wandb | MLflow / neptune | wandb has explicit AMD GPU metrics via rocm-smi. MLflow doesn't. Neptune is paid. |
| Testing | hypothesis + pytest | unittest | Property-based testing is essential for algebraic invariants. unittest lacks generation strategies. |
| Config management | hydra-core | argparse / sacred | hydra handles complex experiment configs with overrides, sweeps. argparse is too manual. sacred is unmaintained. |

---

## What NOT to Use

### Do NOT use: TensorFlow or JAX
**Why:** ROCm support is a second-class citizen. TensorFlow's ROCm docker setup on RX 7900 XTX is documented as painful (cprimozic.net notes). JAX ROCm is experimental. PyTorch's HIP translation layer makes `torch.cuda` code work on AMD GPUs with zero changes.

### Do NOT use: HyperNets PHM layers as your algebra implementation
**Why:** PHM layers LEARN multiplication rules via parameterized Kronecker products. This project needs to ENCODE the specific octonionic multiplication table. Using PHM would defeat the purpose -- you'd be learning an approximation of octonion multiplication rather than using the exact algebra. However, PHM is a valid baseline for comparison (does learned hypercomplex structure outperform hard-coded octonionic algebra?).

### Do NOT use: hTorch or Pytorch-Quaternion-Neural-Networks as pip dependencies
**Why:** Both are abandoned (2021 and 2019 respectively). Pin to specific commits and vendor the relevant source files if needed for quaternion baselines. Do not add them to requirements.txt.

### Do NOT use: Numba for GPU kernels
**Why:** numba-hip exists but is less mature than PyTorch's native custom autograd functions. PyTorch's `torch.autograd.Function` with standard tensor operations runs on ROCm automatically. Only reach for custom HIP kernels if profiling reveals bottlenecks in octonionic multiplication.

### Do NOT use: float32 for octonionic algebra validation
**Why:** Non-associative operations compound floating-point errors. Use float64 for all correctness validation and property-based tests. Switch to float32/bfloat16 only for training speed experiments after validating numerical stability.

### Do NOT use: Deep Octonion Networks (Zhu 2019) as a codebase
**Why:** No public code release exists (confirmed via Papers with Code). The paper describes octonion convolution, batch norm, and weight init, but you'd be reimplementing from the paper anyway. Better to build cleanly with your own autograd-aware design.

---

## Installation

```bash
# ============================================================
# Full environment setup for Octonionic Computation Substrate
# ============================================================

# Option A: Docker (RECOMMENDED for ROCm stability)
# Use official ROCm PyTorch image as base
docker pull rocm/pytorch:latest
# Then install additional packages inside container

# Option B: Native install (requires ROCm 7.1+ on Ubuntu 24.04)
# Core ML framework
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1

# Hyperbolic geometry & Riemannian optimization
pip install git+https://github.com/geoopt/geoopt.git
pip install geomstats

# Geometric algebra (reference/validation)
pip install clifford kingdon

# Numerical computation
pip install numpy scipy numpy-quaternion

# Experiment infrastructure
pip install wandb pytest hypothesis hypothesis-torch

# Visualization
pip install matplotlib seaborn plotly

# Configuration & reproducibility
pip install hydra-core tensorboard

# Dev tools
pip install black ruff mypy ipython jupyter
```

```bash
# Verify ROCm + PyTorch setup
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
# Quick test: tensor operations on GPU
x = torch.randn(8, 8, device='cuda', dtype=torch.float64)
y = torch.randn(8, 8, device='cuda', dtype=torch.float64)
z = x @ y
print(f'GPU matmul OK: {z.shape}')
"
```

---

## Version Pinning Strategy

For reproducibility, pin exact versions in `requirements.txt` but track compatibility ranges:

| Package | Pin Strategy | Rationale |
|---------|-------------|-----------|
| torch | `>=2.7,<2.11` | Must support ROCm 7.x. Autograd API stable across these versions. |
| geoopt | Install from git master | PyPI version (0.5.1) is from 2023. Master has fixes. |
| geomstats | `>=2.8.0` | Latest release, PyTorch backend. |
| clifford | `>=1.5.0` | Stable, infrequent releases. |
| wandb | `>=0.19.11` | AMD GPU metrics support confirmed at this version. |
| hypothesis | `>=6.0` | PyTorch tensor strategies. |
| numpy | `>=1.26` | Float64 precision, quaternion package compat. |
| scipy | `>=1.12` | Rotation class improvements. |

---

## ROCm Compatibility Summary

| Component | ROCm Compatible? | Notes |
|-----------|-------------------|-------|
| PyTorch core (tensors, autograd, nn) | YES | HIP translation layer. `torch.cuda` API works unchanged. |
| Custom `torch.autograd.Function` | YES | Forward/backward with standard tensor ops. No CUDA-specific code needed. |
| `torch.compile` + Triton | YES (with caveats) | Works but may need AMD wavefront tuning. Not critical for research PoC. |
| geoopt | YES | Pure PyTorch. No custom CUDA. |
| geomstats | YES | PyTorch backend uses standard ops only. |
| clifford | YES (CPU only) | NumPy-based. No GPU involvement. |
| kingdon | YES | Agnostic to tensor backend. |
| wandb | YES | Reads AMD GPU metrics via rocm-smi. |
| hypothesis-torch | YES | Generates tensors; device-agnostic. |
| Custom HIP/Triton kernels | POSSIBLE | Only if profiling demands it. PyTorch's HIPIFY tool converts CUDA to HIP. |

---

## Architecture of Custom Components

```
octonion_substrate/
    algebra/
        octonion.py          # OctonionTensor class (8 real components)
        multiplication.py    # Fano plane multiplication table + autograd
        conjugation.py       # Conjugation, norm, inverse + autograd
        subalgebras.py       # 7 quaternionic subalgebra decomposition
        associator.py        # [x,y,z] = (xy)z - x(yz) computation

    calculus/
        ghr.py               # GHR calculus gradient formulas
        jacobian.py          # Octonionic Jacobian computation

    layers/
        linear.py            # OctonionLinear (analogue of QuaternionLinear)
        conv.py              # OctonionConv1d, OctonionConv2d
        normalization.py     # OctonionBatchNorm, OctonionLayerNorm
        g2_equivariant.py    # G2-equivariant layer
        projection.py        # Hyperboloid-octonionic projection

    baselines/
        real_layers.py       # Standard real-valued baselines
        complex_layers.py    # Complex-valued layers (use torch.complex64)
        quaternion_layers.py # Adapted from Orkis-Research / SpeechBrain

    utils/
        fano_plane.py        # Fano plane structure constants
        g2_algebra.py        # G2 Lie algebra generators (14-dim)
        numerical.py         # Stability checks, condition numbers
        reproducibility.py   # Seed management, deterministic mode
```

---

## Sources

### Verified (HIGH confidence)
- [PyTorch Get Started](https://pytorch.org/get-started/locally/) -- PyTorch 2.10.0 stable, ROCm 7.1 stable wheels
- [ROCm Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) -- ROCm 7.2.0, gfx1100 support confirmed
- [PyTorch HIP Semantics](https://docs.pytorch.org/docs/stable/notes/hip.html) -- ROCm/HIP compatibility details
- [geoopt GitHub](https://github.com/geoopt/geoopt) -- v0.5.1, PyTorch >= 2.0.1, manifold support list
- [geomstats GitHub](https://github.com/geomstats/geomstats) -- v2.8.0, PyTorch backend, Lie groups
- [clifford PyPI](https://pypi.org/project/clifford/) -- v1.5.1, geometric algebra
- [Hypothesis GitHub](https://github.com/HypothesisWorks/hypothesis) -- property-based testing
- [hypothesis-torch PyPI](https://pypi.org/project/hypothesis-torch/) -- PyTorch tensor strategies
- [numpy-quaternion PyPI](https://pypi.org/project/numpy-quaternion/) -- v2023.0.0

### Verified (MEDIUM confidence)
- [HyperNets GitHub](https://github.com/eleGAN23/HyperNets) -- PHM/PHC layers, n-dimensional hypercomplex
- [KHNN GitHub](https://github.com/rkycia/KHNN) -- 3 commits, experimental PyTorch
- [hTorch GitHub](https://github.com/ispamm/hTorch) -- Last commit Nov 2021, quaternion only
- [Pytorch-Quaternion-Neural-Networks](https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks) -- Last commit 2019
- [kingdon GitHub](https://github.com/tBuLi/kingdon) -- v1.3.0, PyTorch tensor support
- [SpeechBrain quaternion layers](https://speechbrain.readthedocs.io/en/v1.0.2/tutorials/nn/complex-and-quaternion-neural-networks.html)
- [escnn GitHub](https://github.com/QUVA-Lab/escnn) -- Generalized Wigner-Eckart, E(2)/E(3) focus
- [Kingdon paper](https://arxiv.org/abs/2503.10451) -- arXiv 2025
- [ROCm Triton kernel development](https://rocm.blogs.amd.com/artificial-intelligence/triton/README.html)
- [W&B AMD GPU metrics](https://docs.wandb.ai/ref/python/experiments/system-metrics/)
- [numba-hip GitHub](https://github.com/ROCm/numba-hip)

### Research references (for implementation guidance)
- [GHR Calculus paper](https://www.researchgate.net/publication/266261808_Quaternion_Derivatives_The_GHR_Calculus) -- Quaternion derivatives foundation
- [GHR Calculus in QNNs](https://www.nnw.cz/doi/2017/NNW.2017.27.014.pdf) -- Backpropagation with GHR
- [Deep Octonion Networks (arXiv 1903.08478)](https://arxiv.org/abs/1903.08478) -- Architecture reference (no code available)
- [Hypercomplex neural networks survey (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12513225/) -- Current state of field
- [Papers with Code: Deep Octonion Networks](https://paperswithcode.com/paper/deep-octonion-networks) -- Confirms no public implementation
- [G2 mathematics (Wikipedia)](https://en.wikipedia.org/wiki/G2_(mathematics)) -- G2 as octonion automorphism group
