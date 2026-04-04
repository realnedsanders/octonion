# The Octonionic Trie

A self-organizing hierarchical memory where octonionic algebra replaces gradient-based learning entirely.

## How it works

The trie routes inputs through a tree structure using the algebraic properties of octonions:

1. **Routing**: At each node, the product $n \cdot x$ is decomposed along the 7 quaternionic subalgebras. The child whose subalgebra has the largest projection is selected.
2. **Compatibility check**: The associator $[x, c, n]$ is computed. If its norm is below the threshold, the input descends into child $c$.
3. **Novelty detection**: If no child is compatible (all associator norms exceed the threshold), a new branch is created.
4. **Content update**: At the leaf, the node's content is updated by octonionic composition: $o' = o \cdot x$.

## Key results

| Experiment | Result |
|------------|--------|
| MNIST (CNN encoder, 60K training) | 95.2% accuracy (no gradients in classifier) |
| 7-category stability-plasticity | 97.7% accuracy, 0% forgetting |
| Novelty detection | 5x spike ratio at category transitions, zero false negatives |

## Threshold policies

The associator compatibility threshold is the single most consequential parameter. The library provides 7 pluggable policies:

| Policy | Strategy |
|--------|----------|
| `GlobalPolicy` | Fixed threshold at all nodes |
| `PerNodeEMAPolicy` | Exponential moving average of observed associator norms |
| `PerNodeMeanStdPolicy` | Welford's online mean + k*std |
| `DepthPolicy` | Threshold decays (or grows) exponentially with depth |
| `AlgebraicPurityPolicy` | Variance of associator norms and similarities in node buffer |
| `MetaTriePolicy` | A second trie optimizes thresholds via ratio-feedback signal |
| `HybridPolicy` | Blends two policies (mean, min, max, or adaptive transition) |

## API

::: octonion.trie.OctonionTrie

::: octonion.trie.TrieNode
