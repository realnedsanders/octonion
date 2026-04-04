# Quick Start

## Basic algebra

```python
import torch
from octonion import Octonion, associator

# Octonions are 8-dimensional: [real, e1, e2, e3, e4, e5, e6, e7]
a = Octonion(torch.tensor([1.0, 2, 0, 0, 0, 0, 0, 0], dtype=torch.float64))
b = Octonion(torch.tensor([0.0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float64))

# Multiplication is non-commutative
print(a * b)  # different from b * a

# Norm is always preserved
product = a * b
assert abs(product.norm() - a.norm() * b.norm()) < 1e-12

# Every non-zero octonion has an inverse
a_inv = a.inverse()
identity = a * a_inv  # close to [1, 0, 0, 0, 0, 0, 0, 0]

# The associator measures failure of associativity
c = Octonion(torch.randn(8, dtype=torch.float64))
assoc = associator(a, b, c)  # (a*b)*c - a*(b*c)
print(f"||[a,b,c]|| = {assoc.components.norm():.4f}")
```

## The octonionic trie

```python
import torch
from octonion.trie import OctonionTrie

# Create a trie
trie = OctonionTrie(associator_threshold=0.3, seed=42)

# Insert labeled octonionic data
for i in range(100):
    x = torch.randn(8, dtype=torch.float64)
    x = x / x.norm()
    trie.insert(x, category=i % 5)

# Query: which category does a new input belong to?
query = torch.randn(8, dtype=torch.float64)
query = query / query.norm()
leaf = trie.query(query)
print(f"Predicted category: {leaf.dominant_category}")
print(f"Trie stats: {trie.stats()}")
```

## Fano plane structure

The 7 quaternionic subalgebras of the octonions are indexed by the lines of the Fano plane:

```python
from octonion._fano import FANO_PLANE

# 7 triples, each defining a quaternionic subalgebra
for i, triple in enumerate(FANO_PLANE.triples):
    print(f"S_{i}: span{{1, e_{triple[0]}, e_{triple[1]}, e_{triple[2]}}}")
```
