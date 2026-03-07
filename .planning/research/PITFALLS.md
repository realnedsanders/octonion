# Domain Pitfalls: Octonionic Computation Substrate

**Domain:** Hypercomplex ML research -- octonionic algebra, non-associative optimization, hyperbolic geometry, GPU computation on ROCm
**Researched:** 2026-03-07

---

## Critical Pitfalls

Mistakes that cause rewrites, produce silently wrong results, or invalidate experimental findings.

---

### CP-1: Octonion Multiplication Sign/Order Errors (The Fano Plane Trap)

**What goes wrong:** The octonion multiplication table encodes 480 distinct valid sign conventions across 30 Fano plane orientations (30 planes x 16 sign masks). Implementing the wrong convention, or mixing conventions between components, produces an algebra that *looks* like octonions but violates alternativity or norm-preservation. Tests pass superficially (multiplication produces 8-component outputs) but the algebra is broken.

**Why it happens:** Unlike quaternion multiplication (which has essentially one convention up to handedness), octonion multiplication has multiple valid representations. Copy-pasting a multiplication table from one source and combining it with identities from another source that uses a different convention creates silent inconsistencies. The Fano plane mnemonic requires consistent cyclic ordering on all 7 lines -- reversing any single arrow negates that triple's product.

**Consequences:** Every downstream computation (associator, G2 action, norm preservation, invertibility) will produce wrong results. Property-based tests may pass individually but cross-property tests fail. The entire algebra layer must be rewritten.

**Prevention:**
1. Choose ONE canonical reference (Baez 2002 is the standard) and implement the complete multiplication table from that single source.
2. Implement exhaustive property-based tests from day one:
   - Alternativity: `(xx)y = x(xy)` and `(xy)y = x(yy)` for random x, y
   - Norm preservation: `|xy| = |x||y|` for random x, y
   - Moufang identities: `z(x(zy)) = ((zx)z)y` for random x, y, z
   - All 7 quaternionic subalgebra embeddings are valid quaternion algebras
3. Test that the associator `[x,y,z] = (xy)z - x(yz)` is non-zero for generic triples but vanishes when any two arguments are equal.
4. Verify against a known-correct implementation (e.g., SageMath's `Octonions()` or the `octonion` package).

**Detection:** Run the full Moufang identity test suite on 10,000+ random triples. If any identity fails, the multiplication table is wrong. This should be the very first test in CI.

**Phase:** Must be addressed in Phase 1 (Core Algebra). Gate all subsequent work on this.

**Confidence:** HIGH -- this is well-documented in mathematical computing literature.

---

### CP-2: Non-Associative Autograd Ordering (The Silent Gradient Corruption Trap)

**What goes wrong:** PyTorch's autograd engine assumes associative operations. When computing octonionic expressions like `(W * x) * b` vs `W * (x * b)`, the forward pass respects parenthesization because you explicitly code it, but the backward pass may silently reorder operations if you rely on standard autograd. The gradients computed are mathematically wrong, but training still runs -- the network just converges to a worse solution or fails to converge, and you blame the architecture instead of the gradient computation.

**Why it happens:** Standard autograd records a DAG of operations and replays them in reverse. For associative operations, any parenthesization gives the same gradient. For non-associative operations, `d/dW [(Wx)b]` is different from `d/dW [W(xb)]`. If you implement octonion multiplication as a custom `torch.autograd.Function`, you must derive and hardcode the correct backward for each specific parenthesization used in the forward pass.

**Consequences:** Training proceeds but produces suboptimal or random results. Ablation studies comparing octonions to quaternions/reals will show octonions performing worse, leading you to incorrectly conclude the architecture is flawed rather than the gradient computation.

**Prevention:**
1. Implement octonionic multiplication as a `torch.autograd.Function` with explicit `forward()` and `backward()` methods. Never rely on autograd composing sub-operations.
2. Validate gradients numerically using finite differences: for every operation `f(x)`, verify `backward()` matches `(f(x + eps) - f(x - eps)) / (2*eps)` to within `1e-5` relative error.
3. Write the backward pass by hand using the GHR calculus framework (Xu & Mandic, 2015), extended to octonions. The key insight: for octonions, you need separate left-gradient and right-gradient because `d(ax)/da` differs from `d(xa)/da` due to non-commutativity, AND the chain rule must account for non-associativity.
4. Test gradient correctness on known-answer problems: e.g., minimizing `|Wx - y|^2` where `W`, `x`, `y` are octonions with known optimal `W = y * x^{-1}`.

**Detection:** `torch.autograd.gradcheck()` will catch this IF you use float64 and sufficiently small epsilon. Always run gradcheck on every new octonionic operation before integrating it.

**Phase:** Must be addressed in Phase 1 (Core Algebra) alongside multiplication. Cannot proceed to training without verified gradients.

**Confidence:** HIGH -- this is a fundamental consequence of non-associativity interacting with autograd.

**Sources:**
- [Enabling quaternion derivatives: the generalized HR calculus](https://pmc.ncbi.nlm.nih.gov/articles/PMC4555860/)
- [Quaternion Derivatives: The GHR Calculus](https://arxiv.org/abs/1409.8168)
- [PyTorch Autograd Mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)

---

### CP-3: Hyperboloid Projection Destroying Algebraic Properties (The Geometry-Algebra Conflict)

**What goes wrong:** When combining octonionic representations with hyperbolic geometry (the thesis's Option B hybrid model), projecting octonionic vectors onto the hyperboloid or Poincare ball can destroy the algebraic structure that makes octonions useful. Specifically, the exponential map `exp_0: T_0 H^n -> H^n` does not preserve the octonionic multiplication structure. After projection, `exp_0(x) * exp_0(y) != exp_0(x * y)` -- the projected representations no longer form a valid algebra.

**Why it happens:** Hyperbolic space is a Riemannian manifold; octonion multiplication is an algebraic operation on a vector space. These are fundamentally different mathematical structures. Naively applying one on top of the other conflates geometric curvature with algebraic closure. The thesis acknowledges this as an open problem (section 9.7).

**Consequences:** The model "works" in the sense that it trains and produces outputs, but the octonionic algebraic properties (invertibility, norm-preservation, G2 symmetry) are not actually present in the hyperbolic representations. You cannot do backward reasoning through projected representations. The model degenerates into a standard hyperbolic neural network with extra parameters.

**Prevention:**
1. Define precisely WHICH algebraic properties must be preserved after projection, and test for them explicitly.
2. Consider operating in tangent space (where octonionic algebra is valid) and only projecting for distance computations, keeping algebra and geometry in separate lanes.
3. Implement quantitative "algebraic integrity" metrics: after projection, measure `|proj(x * y) - proj(x) * proj(y)| / |proj(x * y)|` on test data. If this ratio exceeds a threshold, the combination is not working.
4. Consider the Euclidean parametrization approach from Mishne et al. (2023): parameterize hyperbolic points via `exp_0(z)` where `z` lives in tangent space, and perform octonionic operations on `z` rather than on the projected point.

**Detection:** Track algebraic integrity metrics during training. If they degrade, the model is losing octonionic structure.

**Phase:** This is the central open problem. Should be investigated in a dedicated research phase (likely Phase 3 or 4) AFTER both the algebra and hyperbolic components are independently validated.

**Confidence:** HIGH for the problem existing; LOW for specific solutions (this is genuinely open research).

**Sources:**
- [The Numerical Stability of Hyperbolic Representation Learning (Mishne et al., 2023)](https://arxiv.org/abs/2211.00181)

---

### CP-4: The "Why Not Just R^8?" Experimental Design Trap

**What goes wrong:** Experiments comparing octonionic networks to baselines show octonionic networks outperforming real-valued R^8 networks, and the researcher concludes the octonionic structure is responsible. But the comparison is unfair: the octonionic multiplication implicitly creates cross-component interactions that the R^8 baseline lacks, giving it a hidden capacity advantage equivalent to a fully-connected mixing layer. The observed improvement comes from the additional mixing, not from octonionic algebraic structure.

**Why it happens:** An octonion multiplication `W * x` where W and x are octonions involves 64 real multiplications coupling all 8 components. An equivalent R^8 operation `W_diag * x` with matched parameter count (8 parameters) only does component-wise scaling with no cross-component interaction. The fair comparison is against an R^8 network with an 8x8 mixing matrix (64 parameters), not against 8 independent scalar multiplications.

**Consequences:** Published results are invalid. Claims about octonionic structure providing advantages are actually claims about dense cross-component mixing providing advantages. The entire research contribution collapses to "dense layers are better than diagonal layers," which is trivially true and already known.

**Prevention:**
1. **Matched-structure baselines:** For every octonionic layer, create a real-valued baseline that has the SAME connectivity pattern but uses arbitrary learned mixing weights instead of octonion-constrained weights. This isolates whether the specific algebraic constraints (alternativity, norm preservation, Moufang identities) help or hurt compared to unconstrained mixing.
2. **Matched-parameter baselines:** ALSO compare against real-valued networks with the same total parameter count, but use standard architectures (fully connected, properly initialized).
3. **Ablation ladder:** Test: (a) R^8 diagonal, (b) R^8 dense mixing, (c) R^8 with octonion connectivity pattern but random signs, (d) full octonion multiplication. If (d) does not outperform (b) and (c), the algebraic structure is not helping.
4. **Algebraic necessity tests:** Design tasks where the octonionic structure is THEORETICALLY necessary (e.g., tasks requiring norm-preserving transformations, tasks with G2 symmetry in the data). If octonions don't outperform on these tasks, the structure is not being exploited.
5. **Statistical rigor:** Run each comparison with 10+ seeds, report confidence intervals, and use appropriate statistical tests (paired t-tests or Wilcoxon signed-rank, not just comparing means).

**Detection:** If octonionic networks outperform R^8-diagonal but NOT R^8-dense-mixing baselines, the algebraic structure is not providing value beyond cross-component interaction.

**Phase:** Experimental design must be established in Phase 2 (first experiments). Every experiment from that point forward must include the matched-structure baseline.

**Confidence:** HIGH -- this is the most common methodological error in hypercomplex ML papers. The PMC survey (2025) explicitly notes "a side-by-side comparison of various hypercomplex systems in practical machine learning tasks remains absent."

**Sources:**
- [Hypercomplex neural networks: Exploring quaternion, octonion, and beyond in deep learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC12513225/)
- [Deep Octonion Networks](https://arxiv.org/abs/1903.08478)

---

### CP-5: Numerical Instability in Hyperbolic Coordinates (The NaN Cascade)

**What goes wrong:** Hyperbolic embeddings produce NaN values during training, crashing the optimization or silently corrupting model parameters. This happens because both the Poincare ball and Lorentz hyperboloid models have finite numerical capacity in float64, and standard training can push points beyond representable limits.

**Why it happens:** Three distinct failure modes:
1. **Poincare boundary collapse:** Points with `||x|| -> 1` are rounded to the boundary, where the metric tensor `g = 4/(1-||x||^2)^2 * I` becomes infinite. The conformal factor `(1-c||x||^2)^2` in the denominator of gradient computations causes gradient explosion near the boundary.
2. **Lorentz coordinate overflow:** In the Lorentz model, the time coordinate `x_0 = cosh(r)` grows exponentially with distance `r`. At `r ~ 19` in float64, `cosh(r)` overflows representable values. The constraint check `x_0^2 - ||x_spatial||^2 = 1` fails because `x_0^2 - (x_0^2 - 1)` is rounded to `0` rather than `1`.
3. **Exponential/logarithmic map singularities:** `exp_0` and `log_0` involve `tanh`, `arccosh`, `sinh` which saturate or overflow for large arguments.

**Consequences:** Training crashes with NaN loss. Or worse: some parameters become NaN while others remain valid, producing corrupted model states that are difficult to debug. In multi-component systems (octonionic + hyperbolic), NaN in the hyperbolic component silently propagates through the octonionic component.

**Prevention:**
1. **Use float64 for ALL hyperbolic operations.** This is non-negotiable per Mishne et al. (2023). Float32 has a usable radius of only ~7 in the Poincare model.
2. **Implement Euclidean parametrization:** Store learnable parameters as unconstrained Euclidean vectors `z`, compute hyperbolic features via `exp_0(z)`. This eliminates boundary issues entirely. Both Poincare and Lorentz features can be derived from `z`.
3. **Gradient clipping with norm awareness:** Clip Riemannian gradients by norm, not component-wise. Use `geoopt.optim.RiemannianAdam` or implement equivalent.
4. **Feature clipping (norm bounding):** Apply `CLIP(x; r) = min(1, r/||x||) * x` with `r < 1 - eps` in the Poincare ball before every operation. This prevents boundary approach.
5. **NaN detection hooks:** Register PyTorch hooks that check for NaN/Inf after every forward pass. Halt training immediately on detection rather than allowing corruption to propagate.
6. **Mixed precision strategy:** Use float64 for hyperbolic geometry, float32 for octonionic algebra and standard layers. Be extremely careful at the boundary between precision domains.

**Detection:** Monitor `max(||x||)` for Poincare embeddings and `max(x_0)` for Lorentz embeddings during training. If either approaches the representability limit (0.9999... for Poincare, ~10^8 for Lorentz), intervene immediately.

**Phase:** Must be addressed when implementing the hyperbolic component (likely Phase 3). The monitoring infrastructure should be built into the training loop from the start.

**Confidence:** HIGH -- Mishne et al. (2023) provide rigorous analysis with reproducible experiments.

**Sources:**
- [The Numerical Stability of Hyperbolic Representation Learning (Mishne et al., ICML 2023)](https://arxiv.org/abs/2211.00181)
- [Poincare Ball in geoopt](https://deepwiki.com/geoopt/geoopt/4.4-poincare-ball)
- [Representing Hyperbolic Space Accurately using Multi-Component Floats (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/file/832353270aacb6e3322f493a66aaf5b9-Paper.pdf)

---

### CP-6: Weight Initialization Destroying Hypercomplex Structure

**What goes wrong:** Initializing octonionic weight components independently using standard Xavier/Glorot or He initialization breaks the algebraic structure from the very first forward pass. The initialized weights do not form "well-behaved" octonions, causing the network to spend significant training time just learning to form valid octonionic structure rather than learning the task.

**Why it happens:** Glorot/He initialization is derived for real-valued weights assuming a specific variance relationship between fan-in and fan-out. For quaternions, Gaudet & Maida (2018) showed the weight magnitude follows a Chi distribution with 4 DOF, requiring different variance scaling. For octonions with 8 DOF, the Chi distribution has 8 degrees of freedom, and the variance relationship is `Var(|W|) = (8 - (Gamma(4.5)/Gamma(4))^2) * sigma^2`. Using real-valued initialization formulas yields weights with wrong variance, leading to signal explosion or collapse in early layers.

**Consequences:** Slow convergence, training instability in the first epochs, and potentially convergence to suboptimal solutions. In comparative experiments, the octonion network starts with a handicap that quaternion/complex/real baselines don't have, biasing results against octonions.

**Prevention:**
1. Derive the correct octonionic initialization by extending the Gaudet & Maida (2018) quaternion approach to 8 components. The key: sample the weight magnitude from a Chi distribution with 8 DOF, then sample a random unit octonion (uniform on S^7) for the direction.
2. Verify empirically: initialize a deep (10+ layer) octonionic network, pass random data through it, and check that the output variance matches the input variance (the hallmark of correct initialization).
3. Consider polar initialization: initialize weights as unit-norm octonions scaled by the correct variance, ensuring algebraic structure from the start.
4. Compare convergence curves between initialization methods as a sanity check.

**Detection:** Plot the norm of activations at each layer for the first batch. If norms grow or shrink exponentially with depth, initialization is wrong.

**Phase:** Must be addressed in Phase 1 alongside core algebra, before any training experiments.

**Confidence:** MEDIUM -- the quaternion case is well-established (Gaudet & Maida 2018); the extension to octonions is straightforward mathematically but has not been empirically validated in literature.

**Sources:**
- [Deep Quaternion Networks (Gaudet & Maida, 2018)](https://arxiv.org/pdf/1712.04604)
- [Quaternion Convolutional Neural Networks (Joris-JW implementation)](https://github.com/Joris-JW/Quaternion-Convolutional-Neural-Networks)

---

## Moderate Pitfalls

Mistakes that cause significant delays or incorrect intermediate results but are recoverable.

---

### MP-1: Component-wise Batch Normalization Breaks Algebraic Structure

**What goes wrong:** Applying standard batch normalization independently to each of the 8 octonionic components destroys the inter-component relationships that define the octonionic structure. The normalized outputs no longer satisfy octonionic algebraic identities.

**Why it happens:** Standard BN normalizes each component to zero mean and unit variance independently. But octonionic components are coupled -- their relationships encode algebraic structure. Independent normalization decorrelates them, treating the octonion as 8 unrelated real numbers.

**Prevention:**
1. Implement octonionic batch normalization using whitening (extending the quaternion approach from Gaudet & Maida 2018). This involves computing the 8x8 covariance matrix of octonionic components and applying its inverse square root (Cholesky decomposition) to whiten the data while preserving inter-component relationships.
2. Alternative: use group normalization or layer normalization, which may interact better with octonionic structure than batch normalization.
3. Test: after normalization, verify that the Moufang identities still approximately hold for normalized activations.

**Detection:** Compare training curves between component-wise BN and octonionic-aware BN. If component-wise BN performs as well, the network is not exploiting octonionic structure (which itself is a finding worth reporting).

**Phase:** Phase 2 (training pipeline).

**Confidence:** HIGH for the problem; MEDIUM for the specific whitening solution at 8 dimensions (computational cost of 8x8 Cholesky may become relevant in tight loops).

---

### MP-2: GHR Calculus Extension to Octonions Is Non-trivial

**What goes wrong:** The GHR (Generalized HR) calculus provides a framework for derivatives of quaternion functions, including product and chain rules. Researchers assume this extends straightforwardly to octonions, but the extension is complicated by non-associativity. The quaternion GHR calculus relies on quaternion rotations (`q * x * q^{-1}`) being associative (which they are, because quaternions are associative). Octonion rotations are NOT associative, so the rotation-based framework breaks down.

**Why it happens:** The GHR calculus uses the identity `q * f(x) = f(q * x * q^{-1}) * q` (where `q` is a unit quaternion) to define derivatives in a rotation-invariant way. For octonions, the analogous identity requires alternativity (which octonions have) but the chain rule derivation uses associativity of the rotation group, which fails for the full octonionic automorphism group.

**Prevention:**
1. Do NOT assume GHR extends trivially. Derive octonionic gradients from first principles using the component-wise Jacobian approach: treat the octonionic function `f: R^8 -> R^8` and compute the real 8x8 Jacobian matrix, then verify it has the correct structure.
2. For loss functions that are real-valued (as most ML losses are), the gradient simplifies: `df/dW = sum_i (df/dW_i) * e_i` where `e_i` are octonion basis elements and `W_i` are real components. This bypasses the need for a full octonionic calculus.
3. Validate numerically (finite differences) for every gradient derivation.
4. Consider using the Wirtinger-like approach: for real-valued loss `L(W, W*)`, the gradient with respect to octonionic `W` can be expressed in terms of partial derivatives with respect to the 8 real components.

**Detection:** If gradient-based optimization converges to known optima on toy problems (e.g., octonionic least squares), the gradient implementation is correct regardless of the theoretical framework used.

**Phase:** Phase 1 (Core Algebra) -- must be resolved before any training.

**Confidence:** MEDIUM -- the specific mathematical challenges are clear but the literature on octonionic calculus for ML is extremely sparse.

**Sources:**
- [Quaternion Derivatives: The GHR Calculus (Xu et al., 2015)](https://arxiv.org/abs/1409.8168)

---

### MP-3: G2 Exponential Map Numerical Instability

**What goes wrong:** The G2 automorphism group of the octonions is a 14-dimensional compact Lie group. Computing the exponential map `exp: g2 -> G2` (mapping from the Lie algebra to the group) using the matrix exponential of 7x7 or 14x14 matrices is numerically unstable for large algebra elements, and the naive power series truncation introduces systematic errors.

**Why it happens:** The exponential map involves summing a power series `exp(A) = I + A + A^2/2! + ...` which requires many terms for `||A|| >> 1`. For G2 specifically, the group is compact so `exp` is surjective, but the map is highly non-injective (multiple algebra elements map to the same group element), creating numerical sensitivity. Additionally, the 7x7 representation of G2 within SO(7) requires maintaining exact orthogonality, and floating-point errors in the exponential map produce matrices that are only approximately orthogonal, violating the octonionic automorphism property.

**Prevention:**
1. Use the scaling-and-squaring method: compute `exp(A) = (exp(A/2^k))^{2^k}` where `k` is chosen so `||A/2^k|| < 1`, requiring fewer series terms.
2. After every exponential map computation, re-project onto the G2 manifold using Gram-Schmidt-like orthogonalization that preserves G2 structure (not just SO(7) orthogonality).
3. Use `torch.linalg.matrix_exp()` which implements Pade approximants with scaling-and-squaring, rather than rolling your own series.
4. Verify G2 membership after exponentiation: for a matrix `M = exp(A)`, check that `M` preserves octonionic multiplication: `M(x*y) = M(x) * M(y)` for random x, y.

**Detection:** Check that `exp(A) @ exp(A).T = I` (orthogonality) and that the determinant is exactly 1. Monitor the G2 membership test `M(x*y) = M(x)*M(y)` residual throughout training.

**Phase:** Phase 2 or 3 (G2-equivariant layers).

**Confidence:** MEDIUM -- matrix exponential numerics are well-understood; the G2-specific projection step is less standard.

**Sources:**
- [Highly accurate differentiation of the exponential map and its tangent operator (Sonneville et al., 2023)](https://www.sciencedirect.com/science/article/abs/pii/S0094114X23002227)

---

### MP-4: ROCm/HIP-Specific Custom Kernel Failures

**What goes wrong:** Custom PyTorch extensions written for CUDA do not compile or run correctly on ROCm/HIP without modification. The `hipify` tool handles most syntax translation, but subtle differences in warp sizes (64 on AMD vs 32 on NVIDIA), shared memory behavior, and atomic operations cause silent numerical differences or outright crashes on the RX 7900 XTX (gfx1100).

**Why it happens:** RDNA3 (gfx1100) is a consumer GPU architecture with different characteristics from the MI-series data center GPUs that ROCm primarily targets. While officially supported for PyTorch, custom extensions may hit edge cases. Specific issues:
1. Warp size is 64 (vs 32 on NVIDIA) -- warp-level primitives like `__shfl_sync` have different semantics.
2. Some CUDA intrinsics (`__syncwarp`, certain atomics) have no direct HIP equivalent.
3. Flash Attention was only upstreamed for ROCm in PyTorch 2.5+, indicating ongoing compatibility work.
4. Environment variable `HSA_OVERRIDE_GFX_VERSION` may be needed for gfx1100 compatibility.

**Prevention:**
1. **Avoid custom CUDA/HIP kernels entirely if possible.** Implement octonionic operations using standard PyTorch tensor operations (matmul, element-wise ops). PyTorch's built-in operations are already HIP-compatible.
2. If custom kernels are necessary, use the `hipify` tool and test exhaustively on gfx1100 specifically.
3. Use the ROCm Docker container (`rocm/pytorch:latest`) for a reproducible environment.
4. Test numerical equivalence between CPU (reference) and GPU implementations with tight tolerances.
5. Monitor the [ROCm GitHub issues](https://github.com/ROCm/ROCm/issues) for gfx1100-specific problems.

**Detection:** Run all unit tests on both CPU and GPU, comparing results. Any difference beyond float32 rounding (`> 1e-6`) indicates a HIP compatibility issue.

**Phase:** Cross-cutting concern from Phase 1 onwards. Establish CPU-GPU equivalence testing early.

**Confidence:** MEDIUM -- ROCm support for RX 7900 XTX is functional but less battle-tested than CUDA on NVIDIA.

**Sources:**
- [HIP (ROCm) semantics -- PyTorch documentation](https://docs.pytorch.org/docs/stable/notes/hip.html)
- [ROCm/ROCm Issue #5555: gfx1102 kernel initialization failures](https://github.com/ROCm/ROCm/issues/5555)

---

### MP-5: Floating-Point Non-Associativity Masking Algebraic Non-Associativity

**What goes wrong:** IEEE 754 floating-point arithmetic is itself non-associative: `(a + b) + c != a + (b + c)` in general for floats. When implementing octonionic operations (which are algebraically non-associative), you cannot distinguish between algebraic non-associativity (a feature of octonions) and numerical non-associativity (a bug in floating point). Property-based tests that check `(xy)z != x(yz)` may pass for the wrong reason -- the test passes because of floating-point rounding, not because octonion multiplication was implemented correctly.

**Why it happens:** Octonion multiplication involves 64 real multiplications and many additions. The order in which these are accumulated on a GPU (which uses parallel reduction with non-deterministic ordering) introduces floating-point rounding that varies between runs. This variation is on the same scale as the associator for near-unit octonions.

**Prevention:**
1. Test for non-associativity using HIGH-precision arithmetic (float64 or even `mpmath` with 50+ digits) as the reference.
2. Quantify the associator magnitude: for random unit octonions, `||[x,y,z]|| / (||x|| * ||y|| * ||z||)` should be O(1), not O(epsilon). If the measured associator is the same order as machine epsilon, you're measuring floating-point noise.
3. Use deterministic GPU execution (`torch.use_deterministic_algorithms(True)`) for reproducibility, at least during validation.
4. Implement the associator computation in a numerically stable way: compute `(xy)z` and `x(yz)` independently (not as a difference), then subtract.

**Detection:** Compare associator magnitudes at float32, float64, and arbitrary precision. If the magnitude changes with precision, you're measuring numerical noise rather than algebraic non-associativity.

**Phase:** Phase 1 (Core Algebra).

**Confidence:** HIGH -- this is a direct consequence of the intersection of two well-known non-associativity sources.

**Sources:**
- [Impacts of floating-point non-associativity on reproducibility (Bak et al., 2024)](https://arxiv.org/html/2408.05148v3)
- [Effects of Floating-Point non-Associativity on Numerical Computation](https://www.sci.utah.edu/~beiwang/teaching/cs6210-fall-2016/nonassociativity.pdf)

---

### MP-6: Non-Associative Optimization Landscape Pathologies

**What goes wrong:** The loss landscape for octonionic networks may have qualitatively different properties than for real/complex/quaternion networks. Non-associativity means that `L(W1 * (W2 * x))` and `L((W1 * W2) * x)` define different functions of the same parameters, creating a proliferation of local minima, saddle points, or plateaus that standard optimizers cannot navigate.

**Why it happens:** In associative algebras, composing layers (`W1 * W2 * ... * Wn * x`) defines a unique function regardless of parenthesization. This means gradient descent only needs to optimize over weight values. In non-associative algebras, the parenthesization itself becomes a discrete architectural choice that affects the loss landscape. Fixed (left-to-right) parenthesization may place the optimal solution in a region unreachable by gradient descent from typical initializations.

**Prevention:**
1. **Characterize the landscape empirically FIRST**, before building complex architectures. Train simple 1-2 layer octonionic networks on toy tasks. Plot loss landscapes (2D slices) and compare to quaternionic and real baselines. Measure: (a) number of local minima, (b) condition number of Hessian at solutions, (c) gradient norm distribution during training.
2. Try multiple parenthesization strategies (left-to-right, right-to-left, balanced tree) and compare convergence.
3. Use warm restarts or cyclical learning rates to escape potential non-associativity-induced traps.
4. Consider alternative association: the thesis mentions using the alternativity property to constrain parenthesizations to those where the Moufang identities provide some associativity guarantees.

**Detection:** If octonionic networks converge to significantly worse loss values than matched-capacity real networks on the same task across multiple initializations, the landscape may be pathological.

**Phase:** This is identified as the highest-risk item in PROJECT.md. Must be Phase 2 (first experiments), immediately after core algebra.

**Confidence:** MEDIUM -- the theoretical concern is clear; whether it manifests as a practical problem is unknown (this is what the project aims to determine).

---

## Minor Pitfalls

Mistakes that cause small delays or minor quality issues.

---

### mP-1: Octonion Norm Drift in Deep Networks

**What goes wrong:** Repeated octonionic multiplications accumulate floating-point errors that cause unit-norm octonions to gradually lose their unit-norm property. After 10+ multiplications, `||x|| = 1.0 + O(1e-7)` in float32, which compounds through the network.

**Prevention:** Re-normalize after every 2-3 octonionic multiplications. Use float64 for norm-critical operations. Monitor norm statistics during training.

**Phase:** Phase 2 (training pipeline).

---

### mP-2: Moufang Identity Violation as a Training Signal

**What goes wrong:** Researchers try to enforce Moufang identities as a regularization loss. But the Moufang identities are properties of exact octonionic multiplication -- they are always satisfied by correctly-implemented multiplication and never violated by data. Adding them as a loss term is mathematically vacuous and wastes compute.

**Prevention:** Use Moufang identities as a VALIDATION check on your implementation, not as a training objective. If they're violated, your multiplication is buggy, not your data.

**Phase:** Phase 2.

---

### mP-3: Quaternionic Subalgebra Selection Bias

**What goes wrong:** The octonions contain 7 distinct quaternionic subalgebras (corresponding to the 7 lines of the Fano plane). If the network learns to use only one or two subalgebras, it effectively degenerates into a quaternionic network and you lose the benefits of the full octonionic structure. This is hard to detect because the network still trains and produces reasonable results.

**Prevention:** Monitor which subalgebra dimensions are active (have significant gradient flow). Add diversity regularization if needed. Compare against an explicit quaternion baseline to detect degeneration.

**Phase:** Phase 2-3 (training and analysis).

---

### mP-4: Confusing Octonion-Valued Networks with Octonion-Structured Networks

**What goes wrong:** There's a conceptual trap in confusing two distinct architectures: (a) networks whose activations ARE octonions (8-dimensional with multiplication), and (b) networks that use octonionic structure to constrain weight sharing (8 independent channels with octonion-structured connectivity). The thesis describes (a) but existing literature (Deep Octonion Networks, Wu et al. 2019) primarily implements (b). Mixing insights from (b) papers into architecture (a) leads to incorrect design choices.

**Prevention:** Be explicit about which paradigm each component follows. When citing literature, note whether the paper implements octonion-valued or octonion-structured networks.

**Phase:** Phase 1 (architecture design).

---

## "Looks Done But Isn't" Checklist

Things that appear to work in testing but will fail in practice.

| Symptom | Actual Problem | How to Catch |
|---------|---------------|--------------|
| Octonion multiplication passes unit tests | Tests only check specific triples, not all 480 sign convention combinations | Run Moufang + alternativity tests on 10,000+ random triples |
| Gradients pass `gradcheck` at small scale | Order-dependent gradient errors only manifest in deep networks with many compositions | Test gradient correctness on 5+ layer networks, not just single operations |
| Training converges on toy tasks | Non-associative landscape issues only appear at scale or with complex loss surfaces | Test on progressively harder tasks before declaring success |
| Octonion network beats R^8 baseline | Unfair comparison due to hidden cross-component mixing | Include the R^8 dense-mixing baseline (see CP-4) |
| Hyperbolic projections work in float32 | Float32 only supports radius ~7; real data needs radius 20+ | Switch to float64 and re-test on data with deep hierarchies |
| Model runs on GPU | Custom operations may silently produce different results on ROCm vs CPU | Compare CPU vs GPU outputs on identical inputs, assert tolerance |
| G2-equivariant layer preserves structure | Only tested at initialization; structure may degrade during training | Monitor G2 membership residual and octonionic automorphism error per-epoch |
| Batch normalization works | Component-wise BN breaks octonionic structure | Compare algebraic property preservation before/after normalization |
| Loss decreases steadily | Network may be exploiting only a quaternionic subspace | Monitor per-dimension gradient flow and activation statistics |
| Associator is non-zero | May be measuring floating-point noise, not algebraic non-associativity | Verify with multi-precision arithmetic (float64 + mpmath) |

---

## Pitfall-to-Phase Mapping

| Phase Topic | Likely Pitfalls | Severity | Mitigation Priority |
|---|---|---|---|
| **Phase 1: Core Algebra** | CP-1 (sign errors), CP-2 (autograd ordering), CP-6 (initialization), MP-2 (GHR extension), MP-5 (FP vs algebraic non-associativity) | Critical | Must be fully resolved before proceeding |
| **Phase 2: First Experiments** | CP-4 (R^8 comparison), MP-6 (landscape pathology), mP-1 (norm drift), MP-1 (batch normalization) | Critical/Moderate | Experimental design must be established; landscape characterization is the go/no-go gate |
| **Phase 3: Hyperbolic Integration** | CP-3 (projection destroying algebra), CP-5 (NaN cascades) | Critical | Requires float64, Euclidean parametrization, and algebraic integrity metrics |
| **Phase 4: G2-Equivariant Layers** | MP-3 (exp map instability), mP-3 (subalgebra selection bias) | Moderate | Use library matrix_exp, monitor G2 membership |
| **Phase 5: Multi-stream Fusion** | mP-4 (architecture confusion), MP-4 (ROCm issues) | Moderate/Minor | Clear architectural documentation, CPU-GPU equivalence tests |
| **Phase 6: End-to-End PoC** | All pitfalls compound | Varies | Each earlier phase should have resolved its pitfalls; this phase is integration testing |

---

## Phase-Specific Warnings

| Phase Topic | Warning Signs | Immediate Action |
|---|---|---|
| Core Algebra | Moufang identity test fails on >0.1% of random triples | Stop. Multiplication table is wrong. Fix before anything else. |
| Core Algebra | `gradcheck` fails with relative error >1e-4 | Stop. Backward pass has ordering error. Re-derive from component Jacobian. |
| First Experiments | Octonion network loss is >2x worse than R^8-dense-mixing baseline | Landscape may be pathological. Try alternative parenthesizations before concluding. |
| First Experiments | Training diverges in first 100 steps | Initialization or learning rate problem, not fundamental architecture issue. Fix initialization first. |
| Hyperbolic Integration | NaN in loss within first epoch | Missing float64 or missing feature clipping. Never ignore NaN -- diagnose immediately. |
| Hyperbolic Integration | `||proj(x*y) - proj(x)*proj(y)|| / ||proj(x*y)||` > 0.1 consistently | Projection is destroying algebraic structure. Tangent-space approach needed. |
| G2 Layers | `det(exp(A))` deviates from 1.0 by >1e-10 | Matrix exponential precision issue. Use scaling-and-squaring with more refinement. |
| G2 Layers | Network performance identical to non-equivariant baseline | G2 symmetry may not be present in the data, or the equivariance is not being exploited. |
| ROCm/GPU | CPU and GPU outputs differ by >1e-5 (float32) | HIP kernel issue. Fall back to pure PyTorch operations. |

---

## Sources

### Primary (HIGH confidence)
- [The Numerical Stability of Hyperbolic Representation Learning (Mishne et al., ICML 2023)](https://arxiv.org/abs/2211.00181)
- [The Octonions (Baez, 2002)](https://math.ucr.edu/home/baez/octonions/octonions.pdf)
- [Deep Quaternion Networks (Gaudet & Maida, 2018)](https://arxiv.org/pdf/1712.04604)
- [Enabling quaternion derivatives: the GHR calculus (Xu et al., 2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4555860/)
- [PyTorch Numerical Accuracy Notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [PyTorch HIP (ROCm) Semantics](https://docs.pytorch.org/docs/stable/notes/hip.html)

### Secondary (MEDIUM confidence)
- [Hypercomplex neural networks: Exploring quaternion, octonion, and beyond (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12513225/)
- [Deep Octonion Networks (Wu et al., 2019)](https://arxiv.org/abs/1903.08478)
- [Impacts of floating-point non-associativity on reproducibility (Bak et al., 2024)](https://arxiv.org/html/2408.05148v3)
- [Representing Hyperbolic Space Accurately using Multi-Component Floats (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/file/832353270aacb6e3322f493a66aaf5b9-Paper.pdf)
- [Lie Group Decompositions for Equivariant Neural Networks](https://arxiv.org/abs/2310.11366)
- [Quaternion Convolutional Neural Networks: Current Advances](https://link.springer.com/article/10.1007/s00006-024-01350-x)
- [Octonion Associators (2015)](https://arxiv.org/abs/1509.07718)

### Tertiary (LOW confidence -- training data / synthesis)
- G2 exponential map specifics (no octonionic-ML-specific sources found; general Lie group numerics applied)
- Octonionic batch normalization (extrapolated from quaternionic case; no published octonionic BN implementation found)
- Subalgebra selection bias (theoretical concern, no published evidence of this occurring)
