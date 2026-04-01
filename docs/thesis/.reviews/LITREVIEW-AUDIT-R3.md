# Literature Review Audit — Round 3 (Decision Trees Section)

**Date**: 2026-04-01
**Scope**: Section 2.7 "Decision Trees and Tree-Based Classification" — all 13 references + 5 logical claims
**Method**: Hostile verification agent with WebSearch + WebFetch against publisher pages, arXiv, ACM DL, JMLR, Project Euclid

---

## PROBLEMS (4 issues — must fix)

### P1. Quinlan1986 cited as "C4.5" but describes ID3
- **Text**: `C4.5~\citep{Quinlan1986,Quinlan1993}`
- **Reality**: Quinlan1986 describes ID3, not C4.5. C4.5 was introduced in the 1993 book.
- **Fix**: Change to `ID3~\citep{Quinlan1986} and its successor C4.5~\citep{Quinlan1993}`

### P2. "Stateless internal nodes" claim is too strong for neural trees
- **Text**: "classical and neural trees have stateless internal nodes --- each node stores a split criterion but no representation of the data that has passed through it"
- **Reality**: In soft decision trees (Frosst & Hinton 2017), nodes have learned filter weights and biases. In Deep Neural Decision Forests (Kontschieder 2015), routing functions are parameterized by learned CNN features. These ARE learned representations at nodes.
- **Fix**: Restrict to classical trees, or redefine "stateless" precisely: "do not maintain a running representation updated by each passing example"

### P3. (Logical L1) "Classical decision trees are built top-down" — overstated
- **Text**: States as absolute fact; bottom-up tree algorithms exist.
- **Fix**: Add "typically" — "Classical decision trees are typically built top-down"

### P4. (Logical L2) "Hoeffding trees select axis-aligned splits" — overstated
- **Text**: States as inherent property; variants exist with multivariate splits.
- **Fix**: Add "in their original formulation"

---

## CONCERNS (3 issues — should fix)

### C1. Fredkin1960 priority
- **Text**: "The trie data structure itself dates to Fredkin (1960)"
- **Reality**: René de la Briandais described the same structure in 1959. Fredkin coined the name.
- **Fix**: Change "dates to" to "named by" or add "(building on de la Briandais, 1959)"

### C2. Frosst2017 entry type
- **Bib**: Listed as `@article` with only arXiv note; actually a CEX workshop paper at AI*IA 2017.
- **Fix**: Minor — either change to `@misc` or leave as-is (arXiv citation is standard)

### C3. (Logical L4) "Prefix-tree topology" claim
- **Text**: "borrows the name and the prefix-tree topology"
- **Reality**: Standard tries have the prefix property (all descendants share a common prefix). Whether the octonionic trie truly has this property needs verification against the implementation.
- **Fix**: The text already hedges with "borrows the name" — consider softening "prefix-tree topology" to "tree topology" if the prefix property doesn't strictly hold.

---

## CONFIRMED (9 references — no issues)

| Reference | Verdict |
|-----------|---------|
| Quinlan1993 | CONFIRMED — Morgan Kaufmann 1993, correct |
| Murthy1994 | CONFIRMED — JAIR vol 2, standard oblique tree citation |
| Friedman2001 | CONFIRMED — Annals of Statistics vol 29, correct originator |
| ChenGuestrin2016 | CONFIRMED — KDD 2016, pp 785-794 |
| Domingos2000 | CONFIRMED — KDD 2000, Hoeffding bound description accurate |
| Lakshminarayanan2014 | CONFIRMED — NeurIPS 2014, "match the distribution" verified |
| Kontschieder2015 | CONFIRMED — ICCV 2015, "stochastic" correct |
| Lee2013 | CONFIRMED — EJS vol 7, projection pursuit description accurate |
| Tomita2020 | CONFIRMED — JMLR vol 21, "state-of-the-art" verified |
| Grinsztajn2022 | CONFIRMED — NeurIPS 2022 D&B track |

---

## SUMMARY

| Category | Count |
|----------|-------|
| References checked | 13 |
| CONFIRMED | 9 |
| CONCERN | 3 |
| PROBLEM | 4 (2 reference, 2 logical) |
| Hallucinated | 0 |
