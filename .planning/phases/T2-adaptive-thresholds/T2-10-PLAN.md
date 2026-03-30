---
phase: T2-adaptive-thresholds
plan: 10
type: execute
wave: 4
depends_on: ["T2-01"]
files_modified:
  - scripts/theory/monte_carlo_assoc.py
  - docs/thesis/oct-trie.tex
autonomous: true
requirements: []
must_haves:
  truths:
    - "Monte Carlo sampling validates Egan's mean associator norm result (~1.0947) per D-46"
    - "Within-class vs between-class associator norm distributions characterized per D-42"
    - "Fano plane angular separation bound computed per D-42"
    - "Formal proof attempt for global threshold justification per D-43"
    - "G2 invariance implications for thresholds analyzed per D-44"
    - "Stability-plasticity tradeoff formalized per D-49"
    - "Complexity analysis for each ThresholdPolicy per D-50"
    - "Theory section written in oct-trie.tex per D-52"
  artifacts:
    - path: "scripts/theory/monte_carlo_assoc.py"
      provides: "Monte Carlo validation and distribution analysis"
      contains: "def sample_associator_norms"
    - path: "docs/thesis/oct-trie.tex"
      provides: "New threshold theory section"
      contains: "\\section{Adaptive Thresholds"
  key_links:
    - from: "scripts/theory/monte_carlo_assoc.py"
      to: "src/octonion/_octonion.py"
      via: "associator() function for norm computation"
      pattern: "from octonion._octonion import.*associator"
---

<objective>
Develop theoretical analysis of associator thresholds and write thesis section.

Purpose: Per D-42, pursue both associator norm distribution analysis and Fano plane geometry argument. Per D-43, full proof attempt. Per D-52, theory goes in oct-trie.tex. Per D-53, frame as self-organization narrative. This plan runs in parallel with experimental sweeps (Wave 4, independent of sweep infrastructure).

Output: Monte Carlo validation script, distribution analysis, formal proof attempt in oct-trie.tex.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/phases/T2-adaptive-thresholds/T2-CONTEXT.md
@.planning/phases/T2-adaptive-thresholds/T2-RESEARCH.md

@src/octonion/_octonion.py
@src/octonion/_fano.py
@src/octonion/_multiplication.py
@docs/thesis/oct-trie.tex
</context>

<tasks>

<task type="auto">
  <name>Task 1: Monte Carlo associator distribution analysis</name>
  <files>scripts/theory/monte_carlo_assoc.py</files>
  <read_first>
    - src/octonion/_octonion.py (Octonion class, associator function)
    - src/octonion/_fano.py (FANO_PLANE, triples -- 7 quaternionic subalgebras)
    - src/octonion/_multiplication.py (octonion_mul, structure constants)
    - .planning/phases/T2-adaptive-thresholds/T2-CONTEXT.md (D-42, D-46, D-47)
  </read_first>
  <action>
Create scripts/theory/__init__.py (empty) and scripts/theory/monte_carlo_assoc.py.

**Sampling functions** (per D-46):

1. `sample_random_associator_norms(n_samples=100000, seed=42)`:
   - Sample uniformly from S^7 via Gaussian normalization (standard method)
   - Compute ||[a,b,c]|| for each triple
   - Return array of norms
   - Validate: mean should be ~147456/(42875*pi) ~ 1.0947 per Egan

2. `sample_within_class_norms(features_path, n_samples=50000, seed=42)`:
   - Load cached 8D features from .pt file
   - Sample triples where all 3 come from SAME class
   - Compute ||[a,b,c]|| for each
   - Return array of norms per class

3. `sample_between_class_norms(features_path, n_samples=50000, seed=42)`:
   - Load cached features
   - Sample triples where all 3 come from DIFFERENT classes
   - Compute ||[a,b,c]|| for each
   - Return array of norms

4. `sample_subalgebra_proximity_norms(n_samples=50000, seed=42)`:
   - For each of 7 quaternionic subalgebras, sample unit octonions within angular distance epsilon of the subalgebra
   - Compute associator norms
   - Verify bound: ||[a,b,c]|| should be O(epsilon^2) for small epsilon
   - Sweep epsilon = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

5. `compute_fano_angular_separations()`:
   - For each pair of Fano plane subalgebras, compute the angular separation
   - This is the minimum angle between the quaternionic 3-planes in R^7
   - Report as a 7x7 matrix of pairwise angles
   - This gives the "radius of associativity" around each subalgebra

6. `fit_distribution(norms)`:
   - Fit candidate distributions: Rayleigh, half-normal, gamma, beta
   - Report best fit by KS test p-value
   - Useful for characterizing what the "natural" distribution looks like

**Output**: JSON file at results/T2/theory/monte_carlo_results.json with:
- random_mean, random_std, random_median
- egan_theoretical_mean (147456/(42875*pi))
- within_class_mean_per_class, within_class_std_per_class
- between_class_mean, between_class_std
- subalgebra_proximity_bounds (epsilon -> max_norm mapping)
- fano_angular_separations (7x7 matrix)
- distribution_fit_params

Also save plots to results/T2/theory/:
- random_distribution.png: histogram of random associator norms with Egan's mean marked
- within_vs_between.png: overlapping histograms of within-class and between-class norms
- subalgebra_bound.png: log-log plot of epsilon vs max associator norm (should show O(epsilon^2) scaling)
- fano_separations.png: heatmap of pairwise subalgebra angular separations

**CLI interface**:
```
python scripts/theory/monte_carlo_assoc.py --features-dir results/T2/features --output-dir results/T2/theory --n-samples 100000
```

Per D-46, use unit 7-sphere sampling. Per D-47, investigate closed-form relationship between optimal threshold and algebra constants.
  </action>
  <verify>
    <automated>docker compose run --rm dev uv run python -c "
from scripts.theory.monte_carlo_assoc import sample_random_associator_norms, compute_fano_angular_separations
import torch
norms = sample_random_associator_norms(n_samples=1000, seed=42)
mean = norms.mean().item()
egan = 147456 / (42875 * 3.14159265358979)
print(f'MC mean={mean:.4f}, Egan={egan:.4f}, diff={abs(mean-egan):.4f}')
assert abs(mean - egan) < 0.1, f'MC mean {mean} too far from Egan {egan}'
seps = compute_fano_angular_separations()
print(f'Fano separations shape: {seps.shape}')
print('Monte Carlo smoke test PASSED')
"</automated>
  </verify>
  <acceptance_criteria>
    - scripts/theory/monte_carlo_assoc.py exists
    - scripts/theory/__init__.py exists
    - monte_carlo_assoc.py contains `def sample_random_associator_norms`
    - monte_carlo_assoc.py contains `def sample_within_class_norms`
    - monte_carlo_assoc.py contains `def sample_between_class_norms`
    - monte_carlo_assoc.py contains `def sample_subalgebra_proximity_norms`
    - monte_carlo_assoc.py contains `def compute_fano_angular_separations`
    - monte_carlo_assoc.py contains `147456 / (42875 * ` (Egan's constant)
    - Monte Carlo mean of random norms within 0.1 of Egan's theoretical value (1000 samples)
  </acceptance_criteria>
  <done>Monte Carlo validation confirms Egan's result, within-class vs between-class distributions characterized, subalgebra proximity bounds computed, Fano angular separations measured</done>
</task>

<task type="auto">
  <name>Task 2: Threshold theory section in oct-trie.tex</name>
  <files>docs/thesis/oct-trie.tex</files>
  <read_first>
    - docs/thesis/oct-trie.tex (existing structure, LaTeX conventions, where to add new section)
    - .planning/phases/T2-adaptive-thresholds/T2-CONTEXT.md (D-42 through D-53)
    - .planning/phases/T2-adaptive-thresholds/T2-RESEARCH.md (theory context: Egan result, Fano geometry, G2 connection)
    - src/octonion/_fano.py (Fano plane triples for formal definition)
  </read_first>
  <action>
Add a new section to docs/thesis/oct-trie.tex titled "Adaptive Thresholds and Self-Organization" (per D-52). Place it after the existing trie description section.

**Section structure** (per D-42 through D-53):

1. **Associator Norm Distribution on S^7** (per D-42, D-46):
   - Definition: ||[a,b,c]|| for unit octonions a, b, c in S^7
   - Egan's analytical result: E[||[a,b,c]||] = 147456/(42875*pi) ~ 1.0947
   - Cite Egan's "Peeling the Octonions" and Cook's verification
   - Key insight: random unit octonions have a well-defined mean associator norm. The question is whether class-structured data deviates from this.

2. **Fano Plane Geometry and Subalgebra Routing** (per D-42, D-47):
   - The 7 quaternionic subalgebras correspond to Fano plane triples
   - Within a quaternionic subalgebra, the algebra is associative (||[a,b,c]|| = 0)
   - Define angular distance from a subalgebra: theta = arccos(|proj_S(x)| / |x|) where S is the subalgebra's R^3
   - Proposition: For unit octonions within angular distance epsilon of a quaternionic subalgebra, ||[a,b,c]|| = O(epsilon^2)
   - Proof sketch: Taylor expansion of the associator around the associative limit
   - The routing mechanism (subalgebra_activation) routes inputs to their nearest subalgebra, bounding the angular distance

3. **Global Threshold Justification** (per D-43, D-45):
   - Theorem attempt: If the input distribution is such that class-conditional means align with distinct Fano subalgebras, then a global threshold tau* exists such that within-class associator norms are below tau* and between-class norms are above tau*
   - If provable: this is a strong result connecting algebraic structure to classification
   - If not provable: state as conjecture with Monte Carlo evidence (from Task 1)
   - Characterize the boundary: when global suffices (well-separated classes in subalgebra space) vs when adaptive is needed (classes overlapping across subalgebras)

4. **G2 Symmetry and Threshold Invariance** (per D-44):
   - G2 is the automorphism group of the octonions (preserves multiplication table)
   - G2 acts transitively on S^6 (unit imaginary octonions)
   - The associator is G2-invariant: ||[ga, gb, gc]|| = ||[a,b,c]|| for g in G2
   - Implication: any optimal threshold must be G2-invariant
   - This constrains the functional form of adaptive thresholds
   - Cite Baez (2002) for G2 structure, Harvey (1990) for G2 geometry

5. **Stability-Plasticity Tradeoff** (per D-49):
   - Define: tight threshold (small tau) = high stability, low plasticity
   - Define: loose threshold (large tau) = low stability, high plasticity
   - Formal analogy to Neyman-Pearson hypothesis testing:
     - H0: input belongs to current node's class (accept if associator < tau)
     - H1: input belongs to different class (reject if associator >= tau)
     - tau controls Type I (false acceptance) vs Type II (false rejection) error
   - Optimal tau minimizes classification error = argmin(FPR * w1 + FNR * w2) where weights depend on class priors

6. **Self-Organization Narrative** (per D-53):
   - Frame as: "the trie discovers its own operating parameters through algebraic feedback"
   - The associator IS the environment feedback signal (per D-02: unsupervised)
   - Connect to biological self-organizing systems: the threshold adapts to the data distribution's algebraic structure
   - If meta-trie works: "the same algebraic mechanism that routes data also optimizes its own routing parameters"

7. **Complexity Analysis** (per D-50):
   - GlobalPolicy: O(1) per threshold query
   - PerNodeEMAPolicy: O(1) per query (stored EMA), O(1) per update
   - PerNodeMeanStdPolicy: O(1) per query (Welford), O(1) per update
   - DepthPolicy: O(1) per query (exponentiation)
   - AlgebraicPurityPolicy: O(B) per query where B = buffer size (compute variance)
   - MetaTriePolicy: O(D_meta * 7) per update where D_meta = meta-trie depth
   - HybridPolicy: sum of both sub-policies

8. **Convergence of Meta-Trie Feedback** (per D-48):
   - Model the meta-trie as a discrete dynamical system: tau_{t+1} = f(tau_t, data_t)
   - Fixed point analysis: when does tau converge?
   - For the self-referential variant: is the fixed point stable?
   - State formal conditions for guaranteed convergence (contraction mapping, monotonicity)

Use LaTeX conventions matching the existing document: \theorem, \proof, \proposition, \definition environments as used in prior sections. Use \ref for cross-referencing. Add BibTeX entries for Egan and Cook if not already present.

Per D-43: If the proof attempt succeeds, include full proof. If not, state as conjecture with supporting evidence and document where it breaks.
  </action>
  <verify>
    <automated>docker compose run --rm dev bash -c "cd /workspace && grep -c 'section{Adaptive Thresholds' docs/thesis/oct-trie.tex"</automated>
  </verify>
  <acceptance_criteria>
    - docs/thesis/oct-trie.tex contains `\section{Adaptive Thresholds` (or similar section heading)
    - docs/thesis/oct-trie.tex contains `147456` (Egan's constant numerator)
    - docs/thesis/oct-trie.tex contains `Fano` (Fano plane references)
    - docs/thesis/oct-trie.tex contains `G_2` or `G2` (G2 automorphism group)
    - docs/thesis/oct-trie.tex contains `stability` and `plasticity` (stability-plasticity tradeoff)
    - docs/thesis/oct-trie.tex contains complexity analysis for each policy
    - docs/thesis/oct-trie.tex mentions convergence of meta-trie
    - Section contains at least one formal proposition/theorem/conjecture
  </acceptance_criteria>
  <done>Theory section in oct-trie.tex covers all D-42 through D-53 topics: associator distribution, Fano geometry, global threshold justification, G2 invariance, stability-plasticity, self-organization, complexity, and meta-trie convergence</done>
</task>

</tasks>

<verification>
docker compose run --rm dev uv run python -c "
from scripts.theory.monte_carlo_assoc import sample_random_associator_norms
norms = sample_random_associator_norms(1000, seed=42)
print(f'Random associator norm mean: {norms.mean():.4f}')
print('Theory scripts OK')
"
docker compose run --rm dev bash -c "cd /workspace && grep 'section{Adaptive' docs/thesis/oct-trie.tex"
</verification>

<success_criteria>
- Monte Carlo validates Egan's result (mean within 2% of theoretical)
- Within-class vs between-class associator norms show measurable separation
- Subalgebra proximity bounds show O(epsilon^2) scaling
- Thesis section covers all 8 theoretical topics
- Formal proof attempt either succeeds or is documented as conjecture with evidence
</success_criteria>

<output>
After completion, create `.planning/phases/T2-adaptive-thresholds/T2-10-SUMMARY.md`
</output>
