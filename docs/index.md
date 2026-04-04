# octonion

**PyTorch-native octonionic algebra for ML research.**

The `octonion` library provides a complete implementation of the octonion algebra (the largest normed division algebra) built on PyTorch, plus a self-organizing octonionic trie that classifies data without gradient descent.

## Features

- **Full octonionic algebra**: multiplication, conjugation, norm, inverse, associator, exp/log
- **Fano plane structure**: 7 quaternionic subalgebras, automorphism generators
- **Self-organizing trie**: zero-gradient classification via associator-based novelty detection
- **Pluggable threshold policies**: Global, EMA, MeanStd, Depth, AlgebraicPurity, MetaTrie, Hybrid
- **GHR calculus**: Wirtinger derivatives, analytic Jacobians, parenthesization-aware chain rule
- **Fair baselines**: parameter-matched R/C/H/O comparison networks
- **839 tests**: property-based testing with Hypothesis

## Quick install

```bash
# Install PyTorch first for your hardware:
# https://pytorch.org/get-started/locally/

pip install octonion
```

## Minimal example

```python
import torch
from octonion import Octonion, associator

# Create two octonions
a = Octonion(torch.randn(8, dtype=torch.float64))
b = Octonion(torch.randn(8, dtype=torch.float64))

# Multiply (non-commutative, non-associative)
c = a * b

# Norm is preserved: |a*b| = |a|*|b|
assert abs((a * b).norm() - a.norm() * b.norm()) < 1e-12

# The associator measures non-associativity
assoc = associator(a, b, c)
print(f"Associator norm: {assoc.components.norm():.4f}")
```
