# Self-Organizing Octonionic Tries: Summary for Discussion

**Antonio Escalera -- Working Draft, March 2026**

---

## What this is

A self-organizing hierarchical memory structure where **octonionic algebra replaces gradient-based learning entirely**. The structure is an *octonionic trie*: a dynamically growing tree whose nodes are individual octonions. The system routes, grows, updates, verifies, and self-monitors using only the algebraic properties of the octonions -- no weight matrices, loss functions, or backward passes.

## Why octonions (and not quaternions or Clifford algebras)

Quaternions are associative -- the associator [a,b,c] = (ab)c - a(bc) is identically zero for all triples. This eliminates the novelty detection signal that drives trie growth. Clifford algebras of the right dimension can be non-associative, but octonions are the **unique** 8-dimensional normed division algebra (Hurwitz's theorem). The norm-preserving property |ab| = |a||b| guarantees that composition neither amplifies nor attenuates representations -- a property Clifford algebras do not guarantee in general. The Fano plane provides exactly 7 overlapping quaternionic subalgebras, giving a rich fixed routing structure that requires no learning.

**Open question**: A direct comparison with a "quaternionic trie" (which would lack the associator signal) would be the strongest ablation. This has not been done.

## Five components from one algebra

| Component | Conventional systems | Octonionic trie |
|-----------|---------------------|-----------------|
| Routing | Learned attention / gating | Subalgebra decomposition (Fano plane) |
| Novelty detection | Engineered thresholds | Associator norm |
| Content update | Learned write / Hebbian | Octonionic composition |
| Consistency check | None | Algebraic inversion |
| Health monitoring | Separate metrics | Associator norms + composition error |

These are four distinct derived operations, not one -- the economy is that they are **algebraically determined** by a single multiplication table rather than independently engineered.

## Key experimental results

| Experiment | Result | What it shows |
|------------|--------|---------------|
| Subalgebra routing | 90%+ within-class consistency, 95.2% cross-class separation | Routing is discriminative |
| Associator as novelty | 5x spike ratio at category transitions, zero false negatives | Novelty detection works |
| Composition depth | Last-input recovery exact to machine precision at depth 200 | Local inversion is reliable |
| 7-category stability-plasticity | 97.7% accuracy, 0.0% forgetting | Trie learns without forgetting |
| MNIST (CNN encoder) | **95.2%** (vs. 98.2% kNN on same features) | Competitive without gradients |
| MNIST (PCA-only) | 76.5% (vs. 88.8% kNN) | Works without any neural network |

## Honest limitations

- **CNN encoder dependency**: The 95.2% result uses a separately-trained CNN encoder. The PCA-only result (76.5%) shows the trie works without it, but at significantly lower accuracy.
- **Tested only on MNIST and synthetic data**: No results on CIFAR-10, larger class counts, or non-image modalities.
- **No decision tree or random forest baselines**: The comparison to kNN isolates the trie's contribution, but tree-based baselines on the same features are needed.
- **Scalability unknown**: 26,042 nodes for 60K MNIST samples. Whether consolidation bounds growth in practice is an open question.
- **Single-tree, not ensemble**: The trie does not benefit from the variance reduction that makes random forests and gradient boosting dominant on tabular data.

## What would falsify this approach

1. If the associator provides no better novelty signal than cosine similarity on the same octonionic features -- the non-associativity is not doing useful work
2. If a Hoeffding tree on the same features matches or beats the trie -- the algebraic routing adds no value over standard streaming tree methods
3. If a quaternionic trie (with inner-product routing replacing associator-based routing) achieves comparable performance -- the specific properties of the octonions are unnecessary
4. If consolidation cannot bound node growth -- the architecture is impractical for real-world deployment

## Threshold sensitivity

The associator compatibility threshold tau is the single most consequential parameter. Section 7 develops the theory:
- **Egan's result**: Mean associator norm for random unit octonions is 147456/(42875*pi) ~ 1.095
- **G2 invariance**: Optimal threshold functions must be G2-invariant, constraining the policy space
- **Neyman-Pearson framing**: tau controls the stability-plasticity tradeoff as a Type I/II error balance
- **Per-node EMA adaptation**: Empirically effective; convergence analyzed as a contraction mapping

## Contribution framing

This is a **proof of concept** that non-associative algebra can serve as a computational substrate for self-organizing memory. The 95.2% MNIST result demonstrates viability; the theoretical framework (associator as novelty signal, G2 invariance of thresholds, composition error bounds) is the deeper contribution. The paper proposes a qualitatively different computational paradigm -- self-organization from algebraic structure -- complementing the companion thesis on gradient-trained octonionic networks.

## Where this could be submitted

- **NeurIPS / ICML**: Novel ML paradigm with theoretical + experimental validation (would need to cut Related Work, add tree baselines)
- **JMLR**: Full-length journal paper accommodating the theoretical depth
- **Neural Computation**: Emphasis on the algebraic/computational theory
- The current draft is thesis-length (~25 pages) and thorough; a conference submission would require significant condensation

## Questions I'd like feedback on

1. Is the "algebraic economy" framing compelling, or does it need a stronger formal characterization?
2. Which additional baselines would be most valuable: decision trees, random forests, ART, or something else?
3. Is the connection to physics (Furey, G2, Standard Model) a strength or a distraction for an ML audience?
4. Should the adaptive threshold theory (Section 7) be its own paper?
