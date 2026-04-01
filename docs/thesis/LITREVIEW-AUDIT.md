# Literature Review Audit Log

**Date**: 2026-04-01
**Scope**: All 68 new references added to Related Work section of oct-trie.tex
**Method**: 4 parallel verification agents using WebSearch + WebFetch against arXiv, Semantic Scholar, publisher pages, PubMed, DBLP
**Standard**: Every reference checked for (1) existence, (2) metadata accuracy, (3) content claim accuracy, (4) logical coherence

---

## SEVERITY LEGEND

- **PROBLEM**: Must fix before submission. Factual error, hallucinated reference, or material mischaracterization.
- **CONCERN**: Should fix. Overstated claim, imprecise attribution, or weak logic.
- **CONFIRMED**: No issues found. Reference exists and claims are accurate.

---

## PROBLEMS (7 issues — must fix)

### P1. Baez2012 — Material mischaracterization
- **Reference**: Baez, J.C. "Division Algebras and Quantum Theory." Foundations of Physics, 42:819-855, 2012.
- **Our claim**: "formalizes the connection between division algebras and quantum theory, showing that octonions uniquely obstruct standard Hilbert space formulations while suggesting alternative computational frameworks"
- **Reality**: Paper is primarily about the THREE ASSOCIATIVE division algebras (R, C, H) and Dyson's "three-fold way." Octonions are mentioned only briefly as the algebra that FAILS to support quantum theory. The paper does NOT suggest alternative computational frameworks based on octonions.
- **Fix**: Rewrite the sentence. Either (a) replace with Baez2002 which discusses octonionic quantum mechanics more extensively, or (b) accurately describe Baez2012 as establishing the role of R/C/H via the three-fold way, with octonions as the notable exclusion.

### P2. Furey2018 — Wrong author first name
- **Reference**: Furey, N. "Three generations, two unbroken gauge symmetries, and one eight-dimensional algebra." Physics Letters B, 785:84-89, 2018.
- **Bib entry says**: author = "Nicola" Furey
- **Actual**: Author publishes as "C. Furey" or "N. Furey." Full name is Cohl Furey / Nichol Furey. "Nicola" is not a documented name variant.
- **Fix**: Change to `Furey, Nichol` or `Furey, C.` in references.bib.

### P3. Masi2021 — Wrong author first name
- **Reference**: Masi, N. "An Exceptional G2 Extension of the Standard Model..." Scientific Reports, 11:22528, 2021.
- **Bib entry says**: author = "Nichol" Masi
- **Actual**: Author is Nicolo Masi (with accent on o). Published as "N. Masi."
- **Fix**: Change to `Masi, Nicol\`{o}` in references.bib.

### P4. Saoud2020 — Both author names wrong
- **Reference**: Saoud & Ghorbani. "Metacognitive Octonion-Valued Neural Networks..." IEEE TNNLS, 31(2):539-548, 2020.
- **Bib entry says**: `Saoud, Lyes S. and Ghorbani, Rafik`
- **Actual**: First author is "Lyes Saad Saoud" (Saad is part of the name, not an initial). Second author is "Reza Ghorbani" (not Rafik).
- **Fix**: Change to `Saoud, Lyes Saad and Ghorbani, Reza` in references.bib.

### P5. "Every hypercomplex network uses gradient descent" — FALSIFIED
- **Our claim** (tex ~line 147): "every existing hypercomplex network, without exception, uses gradient descent for training"
- **Counterexamples found**:
  - Quaternion-valued Echo State Networks (Xia, Jahanchahi, Mandic, IEEE TNNLS 2015) — reservoir weights are random/fixed, not gradient-trained
  - Quaternion Extreme Learning Machines (multiple papers, e.g., Augmented Quaternion ELM, IEEE 2019) — output weights solved analytically via pseudoinverse, zero gradient descent
  - General Framework for Hypercomplex-Valued ELMs (Vieira et al., J. Computational Mathematics and Data Science, 2022) — explicit generalization to arbitrary hypercomplex algebras
- **Fix**: Soften to "the vast majority of hypercomplex deep networks" or "nearly all hypercomplex architectures" and add a footnote acknowledging ELM/ESN exceptions.

### P6. Wang2024 — Taxonomy misrepresented
- **Our claim** (tex ~line 169): "Solutions fall into three families (Wang 2024): regularization-based, replay-based, architecture-based"
- **Reality**: Wang2024 uses a FIVE-category taxonomy: (1) regularization-based, (2) replay-based, (3) optimization-based, (4) representation-based, (5) architecture-based. We omit two categories and attribute the reduced taxonomy to Wang2024.
- **Fix**: Either (a) cite Wang2024 with all 5 categories, or (b) attribute the classic 3-family taxonomy to an earlier source (e.g., Parisi et al. 2019 or De Lange et al. 2021) and cite Wang2024 separately.

### P7. Sala2018 — Misrepresents comparison as hyperbolic-vs-Euclidean
- **Our claim** (tex ~line 175): "hyperbolic embeddings achieve 0.989 MAP in 2 dimensions where Euclidean embeddings require 200 dimensions"
- **Reality**: BOTH the 0.989 (2D) and 0.87 (200D) results are HYPERBOLIC embeddings. The 0.989 is Sala et al.'s combinatorial hyperbolic embedding; the 0.87 is Nickel & Kiela's learned Poincare embedding. The comparison is hyperbolic-vs-hyperbolic, NOT hyperbolic-vs-Euclidean. Additionally, the 2D result requires ~500 bits of numerical precision per coordinate.
- **Fix**: Correct to "a combinatorial hyperbolic embedding achieves 0.989 MAP in 2 dimensions, compared to 0.87 MAP with learned hyperbolic embeddings in 200 dimensions" or remove the specific numbers.

---

## CONCERNS (11 issues — should fix)

### C1. Zero-forgetting framing — circular reasoning
- **Our claim** (tex ~line 171): "The experimental results confirm 0.0% catastrophic forgetting"
- **Issue**: The 0.0% forgetting is a DESIGN GUARANTEE (routing keys are fixed, new data creates new branches), not an empirical discovery. Using "confirm" implies the experiment tested an uncertain hypothesis. The experiment verifies implementation correctness, not a hypothesis.
- **Fix**: Reframe as "The experimental results verify that the implementation preserves these structural invariants, yielding the expected 0.0% catastrophic forgetting." Acknowledge that the harder test — forgetting under distribution shift or evolving class boundaries — has not been performed.

### C2. Grossberg1976a — anachronistic attribution
- **Our claim**: Cites Grossberg1976a alongside CarpenterGrossberg1987a as if both equally represent ART.
- **Issue**: The 1976 paper is a PRECURSOR to ART, not ART itself. The formal ART framework (with the stability-plasticity dilemma framing and vigilance parameter) was published in 1987.
- **Fix**: Clarify that Grossberg (1976) laid the theoretical groundwork, and Carpenter & Grossberg (1987) formalized it as ART. E.g., "building on the theoretical foundations of \citet{Grossberg1976a}, Adaptive Resonance Theory \citep{CarpenterGrossberg1987a} addresses the stability-plasticity dilemma directly."

### C3. KrotovHopfield2019 — "match" overstates
- **Our claim**: "biologically plausible competition-based learning rules can match backpropagation on standard benchmarks"
- **Reality**: Matches backprop on MNIST but is "slightly poorer" on CIFAR-10 per the paper itself.
- **Fix**: Change "match" to "approach" or "rival" and optionally note the benchmark dependence.

### C4. Jacobsen2018 (i-RevNet) — "match" hides parameter cost
- **Our claim**: "fully invertible networks match the classification performance of non-invertible architectures"
- **Reality**: The injective i-RevNet matches ResNet-50 error (24.7%) but with 181M parameters vs. 26M (7x). The parameter-matched bijective version is 2 percentage points worse (26.7%).
- **Fix**: Qualify with "at increased parameter cost" or focus on the paper's thesis claim about information preservation rather than the performance-matching claim.

### C5. Comminiello2024 — editorializing + wrong term
- **Our claim**: "the first theoretical framework"; "inter-channel coupling"
- **Issues**: (a) Paper describes itself as "a foundational framework" not "the first." (b) Paper uses "inter-channel correlation" not "inter-channel coupling."
- **Fix**: Change "the first" to "a comprehensive" and "coupling" to "correlation."

### C6. 4x parameter reduction attribution
- **Our claim** (tex ~line 139): Attributes 4x reduction to Gaudet2018 and Zhu2018 jointly.
- **Issue**: Neither paper prominently quantifies the reduction as "4x." The explicit 1/4 claim comes from Parcollet2019/2020.
- **Fix**: Attribute the 4x figure to the quaternion network literature generally (citing Parcollet2020 survey) rather than specifically to Gaudet2018/Zhu2018.

### C7. Arjovsky2016 — "directly analogous" slightly strong
- **Our claim**: Unitary RNN weight matrices are "directly analogous" to octonionic norm preservation.
- **Issue**: Both preserve norms, but the mechanisms differ (matrix-vector multiplication in C^n vs. multiplicative property |ab|=|a||b| in a normed division algebra). This is an analogy, not a formal correspondence.
- **Fix**: Change "directly analogous" to "analogous."

### C8. Invertibility distinction — philosophically weak
- **Our claim**: The trie's invertibility is "qualitatively different" from normalizing flows because theirs requires "architectural constraints" while ours is "intrinsic to the algebra."
- **Issue**: Choosing to use octonions IS an architectural decision. The real distinction is that normalizing flows must engineer invertibility at each layer, whereas octonionic invertibility holds for ANY non-zero element by algebraic necessity.
- **Fix**: Reframe as "normalizing flows must engineer invertibility layer by layer via coupling architectures, whereas octonionic invertibility holds for any non-zero element as an algebraic identity."

### C9. G2 equivariance claim — defensible but fragile
- **Our claim**: "no existing work instantiates equivariance for G2 specifically"
- **Issue**: General frameworks for equivariant networks on reductive Lie groups (e.g., NeurIPS 2023) could in principle handle G2. No one has IMPLEMENTED and EVALUATED G2-equivariance, but the capability exists.
- **Fix**: Qualify as "no existing work has implemented and evaluated equivariance for G2 specifically, though general frameworks for reductive Lie groups could in principle be applied."

### C10. Trabelsi2018 — "phase-sensitive" not explicit
- **Our claim**: Complex-valued networks gain "phase-sensitive representations"
- **Issue**: The phrase "phase-sensitive" doesn't appear in Trabelsi2018. The paper discusses phase manipulation via CReLU but frames contributions in terms of building blocks (complex batch norm, complex weight init).
- **Fix**: Minor — either soften to "representations that capture both amplitude and phase" or attribute the phase-sensitivity framing to the broader CVNN literature (Hirose2012).

### C11. Bronstein2021 — outdated citation form
- **Our claim**: Cites as arXiv:2104.13478, 2021.
- **Issue**: Published as a book by MIT Press in 2023. The arXiv preprint is acceptable but the published version is preferable.
- **Fix**: Optional — update to cite the MIT Press book if desired.

---

## ALL CONFIRMED (no issues)

The following references were verified to exist with accurate metadata and accurate content claims:

### Octonionic Algebra
- Schafer1966 (book, Academic Press) -- foundational nonassociative algebra text
- Dixon1994 (book, Kluwer) -- division algebras and physics
- DrayManogue2015 (book, World Scientific) -- octonion geometry
- Okubo1995 (book, Cambridge) -- octonions in physics
- GunaydinGursey1973 (J. Math. Phys.) -- quarks and octonions
- Furey2016 (PhD thesis, Waterloo) -- Standard Model from algebra
- Todorov2018 (Adv. Appl. Clifford Algebras) -- F4 in particle physics
- Boyle2020 (arXiv) -- Jordan algebra and triality
- Sevennec2013 (Confluentes Mathematici) -- Fano plane and Heawood's map
- Agricola2008 (Notices AMS) -- G2 survey
- Tian2000 (Adv. Appl. Clifford Algebras) -- matrix representations of octonions

### Hypercomplex Neural Networks
- Hirose2012 (book, Springer) -- CVNN monograph
- Parcollet2020 (AI Review) -- quaternion NN survey
- Popa2016 (ICANN, LNCS 9886) -- first octonion NN paper
- Ruhe2023 (NeurIPS, oral) -- Clifford group equivariant NNs
- Brandstetter2023 (ICLR) -- Clifford layers for PDEs
- Brehmer2023 (NeurIPS) -- geometric algebra transformer (16D confirmed)
- Grassucci2022 (IEEE TNNLS) -- parameterized hypercomplex (1/n confirmed)

### Memory-Augmented Architectures
- Sukhbaatar2015 (NeurIPS) -- end-to-end memory networks
- Miller2016 (EMNLP) -- key-value memory networks
- Santoro2016 (ICML) -- MANN meta-learning
- Munkhdalai2017 (ICML) -- meta networks
- Wu2022 (ICLR) -- memorizing transformers (non-differentiable kNN confirmed)
- Ramsauer2021 (ICLR) -- modern Hopfield = transformer attention (confirmed)
- Plate1995 (IEEE TNN) -- holographic reduced representations
- Kanerva2009 (Cognitive Computation) -- hyperdimensional computing
- Kleyko2023a/b (ACM CSUR) -- VSA survey parts I and II
- Hopfield1982 (PNAS) -- associative memory

### Catastrophic Forgetting
- McCloskey1989 (Psych. Learn. Motiv.) -- original catastrophic interference
- French1999 (Trends Cogn. Sci.) -- review
- Kirkpatrick2017 (PNAS) -- EWC (accurately described)
- Zenke2017 (ICML) -- synaptic intelligence
- Rusu2016 (arXiv) -- progressive neural networks ("columns" confirmed)

### Self-Organizing Systems
- Kohonen1982 (Biol. Cybernetics) -- original SOM paper
- CarpenterGrossberg1987a (CVGIP) -- ART1 (vigilance parameter confirmed)
- CarpenterGrossbergReynolds1991 (Neural Networks) -- ARTMAP (no backprop confirmed)
- Rauber2002 (IEEE TNN) -- GHSOM (tree of SOMs confirmed)
- Hinton2022 (arXiv) -- forward-forward algorithm ("goodness scores" confirmed)

### Reversible Computation
- Landauer1961 (IBM J. R&D) -- kT ln 2 bound
- Bennett1973 (IBM J. R&D) -- logical reversibility of computation
- Kingma2018 (NeurIPS) -- Glow
- Papamakarios2021 (JMLR) -- normalizing flows review

### Hyperbolic / Tree Structures
- NickelKiela2018 (ICML) -- Lorentz model (numerical stability confirmed)
- Tanno2019 (ICML) -- adaptive neural trees
- Morin2005 (AISTATS) -- hierarchical softmax
- Cohen2016 (ICML) -- group equivariant CNNs

---

## SUMMARY STATISTICS

| Category | Count |
|----------|-------|
| Total references audited | 68 |
| CONFIRMED (no issues) | 50 |
| CONCERN (should fix) | 11 |
| PROBLEM (must fix) | 7 |
| Hallucinated references | 0 |
| Wrong author names | 3 (Furey2018, Masi2021, Saoud2020) |
| Material mischaracterizations | 2 (Baez2012, Sala2018) |
| Falsified universal claims | 1 ("every hypercomplex network uses gradient descent") |
| Misrepresented taxonomies | 1 (Wang2024 five-family → three-family) |

---

## PRIORITIZED FIX LIST

1. **P5** — Soften "every hypercomplex network" universal claim (strongest falsification)
2. **P1** — Rewrite Baez2012 characterization (material mischaracterization)
3. **P7** — Correct Sala2018 hyperbolic-vs-hyperbolic comparison (factual error)
4. **P6** — Fix Wang2024 taxonomy (5 categories not 3)
5. **P2/P3/P4** — Fix author names in bib (Furey, Masi, Saoud)
6. **C1** — Reframe zero-forgetting as design verification
7. **C2** — Clarify Grossberg1976 as ART precursor
8. **C3/C4** — Qualify "match" claims (Krotov, Jacobsen)
9. **C5** — Fix Comminiello "first" and "coupling"
10. **C8** — Reframe invertibility distinction
