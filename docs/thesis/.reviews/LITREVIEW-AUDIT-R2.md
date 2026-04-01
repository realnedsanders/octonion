# Literature Review Audit — Round 2

**Date**: 2026-04-01
**Scope**: Logic, rhetoric, structural gaps, and deep metadata verification
**Method**: Hostile peer review agent + deep metadata verification agent
**Context**: All Round 1 issues have been fixed. This round focuses on NEW issues only.

---

## SEVERITY LEGEND

- **CRITICAL**: A knowledgeable reviewer would reject the paper on this basis.
- **MODERATE**: Weakens the paper significantly; fix before submission.
- **MINOR**: Polish issue; fix if time permits.

---

## CRITICAL (4 issues)

### R2-C1. Missing entire literature thread: Mixture-of-Experts and hierarchical routing

The octonionic trie routes inputs down a tree using algebraic signals as a gating mechanism. This is functionally analogous to **Hierarchical Mixture-of-Experts** (Jordan & Jacobs 1994), where a tree of gating networks routes inputs to specialist experts at the leaves. The connection to Sparse MoE routing (Shazeer et al. 2017), expert-choice routing (Zhou et al. 2022), and the modern MoE-in-Transformers literature is completely absent.

**Why this is critical**: MoE routing is arguably the closest functional analog in mainstream ML to what the trie does. A reviewer familiar with MoE will immediately notice this gap. The omission is particularly damaging because the comparison would force the paper to articulate what algebraic routing provides that learned gating does not — a comparison the paper currently avoids.

**Fix**: Add 2-3 sentences in Section 2.3 or 2.9 explicitly contrasting the trie's algebraic routing with learned gating in MoE architectures. Cite Jordan & Jacobs (1994) at minimum.

---

### R2-C2. Missing comparison: Hierarchical Temporal Memory (HTM)

HTM (Hawkins & Ahmad 2016, already cited elsewhere in the paper for the comparison table) is a biologically-inspired, non-gradient-descent, self-organizing, hierarchical memory system that uses sparse distributed representations and grows representations based on novelty/anomaly signals. It hits nearly every claimed distinguishing property of the trie: hierarchical, self-organizing, non-gradient, novelty-driven growth.

**Why this is critical**: HTM is already in the paper's comparison table (Section 6) but receives zero discussion in the Related Work. A reviewer who notices this will question the survey's integrity. The paper needs to explain WHY the trie differs from HTM, not just ignore it.

**Fix**: HTM already gets a row in Table 1 (Section 6). Add 2-3 sentences in Section 2.4 discussing HTM explicitly, noting its hierarchical non-gradient approach and explaining the distinction (HTM uses engineered spatial pooling + temporal memory; the trie uses a single algebraic substrate).

---

### R2-C3. The "algebraic economy" thesis conflates derived quantities with unified mechanism

The central claim in Section 2.9:
> "The octonionic trie replaces engineered diversity of mechanism with algebraic unity of structure."

The trie uses four DIFFERENT derived quantities from the octonionic algebra:
1. Direct product for routing (subalgebra decomposition)
2. Associator norm for novelty detection
3. Algebraic inversion for consistency verification
4. Running norm statistics for health monitoring

Calling these all "the same mechanism" because they derive from the same algebra is like saying ART uses "a single mechanism" because matching, reset, and vigilance all derive from inner products and norms in R^n. The paper conflates "derived from the same algebra" with "the same mechanism."

**Why this is critical**: This is the paper's central contribution claim. A careful reviewer will dismantle it by noting that four distinct operations ≠ one mechanism, regardless of shared algebraic origin.

**Fix**: Reframe the claim more precisely. The economy is not that there is "one mechanism" but that there is **one algebraic structure from which all mechanisms are derived**, with no additional engineering. The distinction from ART/GHSOM is that those systems require choosing and tuning separate metrics/mechanisms, while the trie's mechanisms are algebraically determined. This is still a meaningful claim but needs to be stated honestly.

---

### R2-C4. Sevennec (2013) citation has no argued connection to trie routing

> "Sevennec (2013) recovers the complete multiplication table from a regular tessellation of the equilateral torus, providing a topological perspective on the combinatorial structure that governs trie routing."

The phrase "that governs trie routing" asserts a connection that is never argued. How does the torus tessellation perspective inform or illuminate how the trie routes? The citation appears to be included for mathematical interest rather than genuine relevance to the contribution.

**Why this is critical at a venue level**: Citation-padding with tangentially related mathematical results is a common reviewer complaint. The sentence should either argue the connection or be honest about it being background rather than motivation.

**Fix**: Either (a) remove "that governs trie routing" and replace with neutral language like "underlying the octonionic multiplication", or (b) add a sentence explaining the specific connection (e.g., "This perspective suggests that the 7 subalgebra routing channels correspond to the 7 hexagonal regions of Heawood's map, providing a topological invariant for the trie's branching structure.").

---

## MODERATE (7 issues)

### R2-M1. ART comparison applies asymmetric standard

> "In ART, these functions require separate mechanisms (matching, reset, and vigilance)"

ART's matching, reset, and vigilance all derive from the same theoretical framework (adaptive resonance theory has a unified energy-function formulation). Describing them as "separate mechanisms" while describing the trie's four distinct operations as a "single algebraic structure" applies a double standard. An ART expert would object.

**Fix**: Rephrase to acknowledge ART's theoretical unity while noting the trie's advantage: "In ART, routing, novelty detection, and stability are governed by a coherent theoretical framework but require separate design choices for the matching function, vigilance criterion, and reset mechanism. In the octonionic trie, these choices are algebraically determined: the Fano plane fixes routing, the associator provides novelty detection, and inversion enables consistency verification, with no free design parameters beyond the compatibility threshold."

---

### R2-M2. Landauer-Bennett framework adds no falsifiable content

> "In the Landauer-Bennett framework, the trie's compositional updates erase no information."

This is true but trivially so — any mathematically invertible function "erases no information" in the Landauer sense. The framework adds nothing specific to the trie that couldn't be said about standard matrix multiplication (which is also invertible for non-singular matrices). Invoking thermodynamic computation theory for a software system running on a standard irreversible computer is physics-envy framing.

**Fix**: Either (a) remove the Landauer-Bennett framing and simply state that octonionic composition is algebraically invertible, or (b) acknowledge that this is a property of the mathematical abstraction, not the physical implementation, and explain what insight it provides beyond stating invertibility.

---

### R2-M3. Furey extrapolation from physics to ML is unjustified

> "suggesting that octonionic algebra has intrinsic organizational capacity for structured data"

The leap from "organizes particle physics representations" to "has intrinsic organizational capacity for structured data [in ML]" is a massive extrapolation. Particle physics representations are constrained by gauge symmetry; ML data has no such constraint. The word "suggesting" does not adequately hedge the inferential gap.

**Fix**: Replace "suggesting that octonionic algebra has intrinsic organizational capacity for structured data" with "suggesting that the octonionic algebraic structure has a natural affinity for organizing representations with rich internal symmetry."

---

### R2-M4. Missing scalability counterargument for architecture-based continual learning

The paper acknowledges open questions about zero-forgetting under distribution shift but does NOT address the well-known scalability objection: a system that never modifies existing representations will have unbounded growth. Progressive networks have this exact problem, and the continual learning literature is well aware that "just add new parameters" is not scalable. The trie creates new branches for new data — this IS a form of capacity growth.

**Fix**: Add one sentence acknowledging this: "Like progressive networks, the trie trades forgetting for growth: its node count increases with data complexity, and whether consolidation (sibling absorption, child merging) can bound this growth in practice remains an open question."

---

### R2-M5. "None exploits the non-associativity as a computational signal" — unverified universal negative

> "All three approaches use the octonions as a representation space within a gradient-trained architecture; none exploits the non-associativity as a computational signal."

This is a strong negative claim about three other papers (Popa 2016, Wu 2020, Saoud 2020). Saoud & Ghorbani's metacognitive architecture uses magnitude and phase information from octonion components for learning decisions. Whether this is sensitive to associator-like quantities is not obvious from abstracts alone. If any of these papers does something non-trivial with non-associativity, this claim collapses.

**Fix**: Soften to "none explicitly exploits the non-associativity — specifically the associator — as a computational signal for routing or novelty detection."

---

### R2-M6. Missing thread: Neurosymbolic AI

The trie computes with algebraic (symbolic-like) operations over continuous representations, performing routing and structural growth without gradient descent. This is within the scope of neurosymbolic AI, which is entirely absent from the Related Work. At minimum, the geometric deep learning framing (Bronstein et al. 2021) partially covers this, but the neurosymbolic angle deserves acknowledgment.

**Fix**: Add a sentence in Section 2.9 noting the neurosymbolic connection: "The trie's use of algebraic operations for both representation and reasoning also connects to the neurosymbolic program, where discrete structural operations are applied to continuous representations."

---

### R2-M7. "sole source" claim in Section 2.4 is false

> "none leverages algebraic structure as the sole source of routing, growth, and consolidation signals"

ART's vigilance parameter IS derived from algebraic structure (inner products, norms). The distinction the paper wants is "non-associative algebraic structure" or "octonionic algebraic structure," not "algebraic structure" full stop. As written, the claim is false.

**Fix**: Change to "none derives routing, growth, and consolidation signals from a single non-associative algebraic structure."

---

## MINOR (7 issues)

### R2-m1. Structural redundancy: Ruhe2023 cited in both Section 2.2 and 2.8

Ruhe et al. (2023) appears in both the hypercomplex networks section AND the equivariant architectures section. The split between these sections is not clean. Consider adding a forward reference in 2.2 ("the equivariance properties of Clifford networks are discussed further in Section 2.8") to signal the intentional split.

---

### R2-m2. "To our knowledge" hedge missing

> "no existing work has implemented and evaluated equivariance for G2 specifically"

Universal negatives should be hedged with "To our knowledge" to avoid asserting survey completeness.

---

### R2-m3. Section 2.6 misses an analytical opportunity

The section surveys hyperbolic embeddings but never explains whether the trie IS or ISN'T a hyperbolic structure. If trees embed naturally in hyperbolic space (Sarkar 2011), and the trie is a tree, what is the relationship? The section's final paragraph waves at this ("Whether augmenting the trie...") but the question is more fundamental: does the octonionic representation already have hyperbolic-like distance properties?

---

### R2-m4. Krotov & Hopfield "rival" is still imprecise

"Rival backpropagation on benchmarks such as MNIST" — the actual result is ~1.4% test error vs. ~0.2% SOTA. "Rival" is generous. "Competitive" or "approach" would be more accurate.

---

### R2-m5. Internal tension: Sections 2.5 and 2.9 on parameter growth

Section 2.5 carefully notes open questions about scalability. Section 2.9 then presents the trie as categorically distinct from progressive networks. But the trie creates new branches for new data, which IS a form of parameter growth. The self-awareness in 2.5 partially mitigates this, but 2.9 re-inflates the distinction.

---

### R2-m6. "log-norm conservation" is physics-envy framing

> "the norm-preserving property |ab| = |a||b| conserves an additive quantity (log-norm)"

This holds for any multiplicative function, including real multiplication. Framing it as "conservation" in the context of Landauer-Bennett thermodynamics implies physical significance that does not exist for a software implementation.

---

### R2-m7. Dinh2016 bib key inconsistency

Bib key is `Dinh2016` but the year field correctly says 2017 (ICLR 2017 publication). The rendered citation "(Dinh et al., 2017)" is correct, but the key name is misleading for maintainability.

---

## METADATA VERIFICATION (15 references deep-checked)

All 15 references that received only light verification in Round 1 were deep-checked against publisher pages, DOIs, and authoritative databases.

| Reference | Verdict | Notes |
|-----------|---------|-------|
| Harvey1990 | CONFIRMED | G2 covered via normed algebras chapter; Baez cites Harvey for key G2 results |
| Conway2003 | CONFIRMED | "A K Peters" correct for original 2003 edition |
| Springer2000 | CONFIRMED | Author surname = publisher name is genuine coincidence |
| Hopfield1982 | CONFIRMED | Vol 79, No 8, pp 2554-2558 exact |
| Hebb1949 | CONFIRMED | Wiley, New York, 1949 exact |
| Oja1982 | CONFIRMED | Vol 15, No 3, pp 267-273 exact |
| Kohonen1990 | CONFIRMED | "Proceedings of the IEEE" is the journal name |
| MartinetzSchulten1991 | CONFIRMED | ICANN 1991, pp 397-402, North-Holland |
| Fritzke1994 | CONFIRMED | Vol 7, No 9, pp 1441-1460 exact |
| Alahakoon2000 | CONFIRMED | Vol 11, No 3, pp 601-614 exact |
| Law2019 | CONFIRMED | ICML 2019, PMLR 97, pp 3672-3681 |
| Chami2019 | CONFIRMED | NeurIPS 2019 correct |
| Dinh2016 | DISCREPANCY | Key says 2016, year field correctly says 2017 (cosmetic only) |
| Boyle2020 | CONFIRMED | Remains arXiv preprint; no journal publication found |
| Hinton2022 | CONFIRMED | Remains arXiv preprint; NeurIPS 2022 talk only |

---

## SUMMARY

| Severity | Round 1 | Round 2 | Combined |
|----------|---------|---------|----------|
| CRITICAL | 7 (all fixed) | 4 | 4 remaining |
| MODERATE | 11 (all fixed) | 7 | 7 remaining |
| MINOR | 0 | 7 | 7 remaining |
| Hallucinated refs | 0 | 0 | 0 total |
| Metadata errors | 3 (all fixed) | 1 cosmetic | 1 cosmetic |

### Round 2 Priority Fix List

1. **R2-C1** — Add MoE/hierarchical routing comparison (biggest gap)
2. **R2-C2** — Add explicit HTM discussion (already in comparison table, missing from lit review)
3. **R2-C3** — Reframe "algebraic economy" more precisely (central thesis claim)
4. **R2-C4** — Fix Sevennec connection or remove routing claim
5. **R2-M1** — Fix asymmetric ART comparison
6. **R2-M4** — Add scalability acknowledgment
7. **R2-M5** — Soften non-associativity universal negative
8. **R2-M7** — Fix "sole source" to "single non-associative algebraic structure"
