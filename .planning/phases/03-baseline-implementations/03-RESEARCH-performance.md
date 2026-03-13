# Phase 3: Baseline Implementations - Performance Optimization Research

**Researched:** 2026-03-13
**Domain:** PyTorch performance optimization for hypercomplex (R/C/H/O) neural network training
**Confidence:** HIGH (code audit) / MEDIUM (ROCm-specific behaviors)

## Summary

This research investigates safe performance improvements for the Phase 3 baseline infrastructure -- the CIFAR-10 reproduction script, training pipeline, algebra-specific layers, and comparison runner. The goal is to identify bottlenecks and safe optimizations that preserve mathematical correctness and reproducibility across all four algebras (R, C, H, O).

The codebase is well-structured but has several addressable performance bottlenecks: (1) the `_tril_to_symmetric` function in batch normalization uses Python loops instead of vectorized indexing, (2) the `OctonionDenseLinear` forward pass uses a Python dict cache with per-entry iteration over 64 nonzero structure constant entries, (3) the fused octonion convolution recomputes the fused weight matrix every forward call, (4) the training loop uses `optimizer.zero_grad()` without `set_to_none=True`, (5) `cudnn.benchmark` is not enabled despite fixed input sizes, and (6) data transfers do not use `non_blocking=True`. Additionally, AMP (mixed precision) is supported but disabled by default, and `torch.compile` is a viable but risky option on ROCm 7.2.

**Primary recommendation:** Apply safe, evidence-based optimizations in three tiers: (1) zero-risk Python-level fixes (vectorized indexing, set_to_none, cudnn.benchmark, non_blocking), (2) moderate-risk caching/precomputation (fused OctonionDenseLinear via einsum pattern already proven in conv layers, structure constant buffer registration), (3) opt-in experimental features (AMP with float16, torch.compile) gated behind config flags. Profile before and after each change to prove improvement. All existing tests must pass unchanged.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Configurable skeleton via `AlgebraNetwork(algebra, topology, depth=N)` pattern
- Three topology types: MLP, Conv (1D + 2D), Recurrent
- Per-algebra full tuning as default with Adam-lock comparison option
- Seed protocol: seed-controlled but not CUDA deterministic (performance over bit-exactness)
- Mixed precision: optional AMP via config flag (ROCm supports AMP)
- Sequential comparison execution (one model at a time, single GPU)
- No time cap on training runs
- Real baseline also verified against published result (validates training infrastructure)
- Octonionic baseline included alongside R/C/H on same benchmarks
- CIFAR training: SGD momentum=0.9, cosine LR, 200 epochs, weight_decay=5e-4, batch_size=128

### Claude's Discretion
- Internal module organization
- Specific optimization strategies (as long as math is preserved)
- Checkpoint frequency
- Data loader configuration details

### Deferred Ideas (OUT OF SCOPE)
- Custom CUDA/ROCm kernels -- per project constraints, premature optimization
- Parallel comparison runs -- keep sequential for now, single GPU
- R8-dense-mixing baseline -- Phase 7
- K-fold cross-validation
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| BASE-01 | Real-valued baseline with 8x units matches octonionic param count within 1% | Performance improvements must not change param counts or model architecture |
| BASE-02 | Complex baseline reproduces published CIFAR-10 result within 1 std | AMP and compile optimizations must preserve convergence behavior |
| BASE-03 | Quaternion baseline reproduces published CIFAR-10 result within 1 std | All optimizations must be algebra-agnostic (apply equally to all 4) |
</phase_requirements>

## Bottleneck Analysis (Code Audit)

### Bottleneck 1: `_tril_to_symmetric` Python Loop -- HIGH impact for Octonion BN

**File:** `src/octonion/baselines/_normalization.py` lines 23-45
**What:** Converts flat lower-triangular entries to symmetric matrix using nested Python `for` loops iterating `dim*(dim+1)/2` times. For octonions (dim=8), this is 36 iterations of scalar Python indexing per feature. Called every forward pass for every BN layer.

**Impact:** For OctonionBatchNorm with 64 features (stage3 of ResNet), this function runs 36 scalar Python assignment pairs per call. With ~56+ BN layers in a depth-28 ResNet (each residual block has 2 BN + optional shortcut BN), plus BN in the conv-to-algebra reshape path, this accumulates to thousands of scalar Python operations per forward pass. For QuaternionBN (dim=4, 10 entries) the impact is smaller but still unnecessary.

**Fix:** Use `torch.tril_indices` to build index tensors once at init time, then use vectorized scatter:
```python
# At init (once):
rows, cols = torch.tril_indices(dim, dim)
# At forward (every call):
mat = torch.zeros(*batch_shape, dim, dim, device=..., dtype=...)
mat[..., rows, cols] = tril_flat
mat[..., cols, rows] = tril_flat  # symmetric copy
```
**Confidence:** HIGH -- this is pure vectorization, mathematically identical. The index tensors are small (36 entries for dim=8) and can be precomputed.

### Bottleneck 2: `OctonionDenseLinear` Python Dict Cache -- MEDIUM-HIGH impact

**File:** `src/octonion/baselines/_algebra_linear.py` lines 279-313
**What:** The forward pass iterates over `self._nonzero_entries` (64 entries of the structure constants tensor) with a Python dict (`linear_cache`) for caching `F.linear` results. Each iteration performs a dict lookup, a conditional `F.linear` call, and an in-place tensor addition with scalar multiplication. The 8 output components are accumulated in a Python list of tensors.

**Impact:** 64 Python loop iterations with dict lookups and conditional branches per forward pass per `OctonionDenseLinear` layer. For the ResNet architecture, the `fc_hidden` layer at the end uses this. For MLP topology with depth=N, this is 64*N iterations per forward pass. Each iteration involves Python interpreter overhead that is significant relative to the actual GPU kernel time for small feature dimensions.

**Fix:** Apply the fused einsum pattern already proven in `OctonionConv2d`:
```python
W_stack = torch.stack(list(self.weights))  # [8, out_f, in_f]
C = self._structure_constants  # registered buffer
fused = torch.einsum("ijk, iof -> kjof", C, W_stack)  # [8, 8, out_f, in_f]
fused_flat = fused.permute(0, 2, 1, 3).reshape(8 * out_f, 8 * in_f)
x_flat = x.reshape(*batch_shape, 8 * in_f)
out_flat = F.linear(x_flat, fused_flat)
result = out_flat.reshape(*batch_shape, out_f, 8)
```
This replaces 64 Python iterations + 22 unique `F.linear` calls + 64 accumulations with: 1 `torch.stack` + 1 `torch.einsum` + 1 `F.linear` + 2 reshapes. The einsum fuses the structure constants into a single weight matrix, identical to what `OctonionConv2d` already does.

**Confidence:** HIGH -- the conv layers (`_algebra_conv.py` lines 593-631) already implement this exact pattern and it is tested. The linear version is a simpler case (no spatial dimensions).

### Bottleneck 3: Fused Conv/Linear Weight Recomputation Every Forward -- MEDIUM impact

**File:** `src/octonion/baselines/_algebra_conv.py` lines 593-631 (OctonionConv2d), lines 411-449 (QuaternionConv2d)
**What:** Both `OctonionConv2d.forward()` and `QuaternionConv2d.forward()` recompute the fused weight matrix from scratch every forward call. For OctonionConv2d: `torch.stack(list(self.weights))` + `torch.einsum("ijk, iochw -> kjochw", C, W_stack)` + reshape. For QuaternionConv2d: `torch.cat` of 16 blocks.

**Impact:** During training, weights change after every optimizer step, so the fused weight MUST be recomputed for each forward pass. However, during evaluation (validation loop), weights don't change between batches, so the fused weight could be cached. With 56+ conv layers in the ResNet, this is 56 einsum + stack operations per forward pass -- significant overhead.

Additionally, `torch.stack(list(self.weights))` iterates a `ParameterList` in Python and creates a new tensor every call. If weights were stored as a single stacked `nn.Parameter` of shape `[8, out_ch, in_ch, kH, kW]`, the stack operation would be eliminated.

**Fix (conservative):** Cache fused weight during `eval()` mode. Invalidate on `train()`:
```python
def train(self, mode=True):
    super().train(mode)
    self._fused_cache = None
    return self

def forward(self, x):
    if self._fused_cache is not None and not self.training:
        fused = self._fused_cache
    else:
        fused = self._compute_fused_weight()
        if not self.training:
            self._fused_cache = fused
    ...
```

**Fix (aggressive):** Replace `ParameterList` with a single stacked `Parameter`. This changes state_dict keys (`weights.0`, `weights.1`, ... -> `weight_stack`) but eliminates the `torch.stack(list(...))` overhead. Requires checkpoint migration or backward-compatible loading.

**Confidence:** MEDIUM -- eval caching is safe and simple. Parameter restructuring is higher risk due to checkpoint compatibility but provides better training speedup.

### Bottleneck 4: Training Loop Micro-Optimizations -- LOW-MEDIUM impact

**File:** `src/octonion/baselines/_trainer.py`

Several individually small but collectively meaningful optimizations:

1. **`optimizer.zero_grad()` without `set_to_none=True`** (line 341): Default fills gradients with zeros via memset. `set_to_none=True` sets them to None, saving the memset per parameter per step. For a model with ~600K parameters across hundreds of tensors, this is a non-trivial memory bandwidth saving.
   - **Confidence:** HIGH -- since PyTorch 2.0, `set_to_none=True` is the recommended default. No mathematical impact. The only behavioral difference: accessing `.grad` after `zero_grad(set_to_none=True)` returns None instead of a zero tensor, but the training loop never does this between zero_grad and backward.

2. **`torch.backends.cudnn.benchmark` not enabled**: The CIFAR-10 training uses fixed input sizes (batch=128, channels=3, H=32, W=32). With `cudnn.benchmark = True`, the first forward pass benchmarks multiple convolution algorithms and selects the fastest. Subsequent calls reuse the selection. Since all training batches have the same shape (data loader uses `drop_last=True`), this is safe and beneficial. For ROCm, this maps to MIOpen's algorithm selection.
   - **Confidence:** HIGH -- standard practice for fixed-size inputs. Introduces non-determinism in algorithm selection, but the project already chose "performance over bit-exactness."

3. **No `non_blocking=True` on `.to(device)`** (line 341): `inputs = batch[0].to(device)` blocks until the transfer completes. With `pin_memory=True` already enabled in data loaders, adding `non_blocking=True` allows the CPU to continue (e.g., preprocessing the next batch) while the GPU transfer happens asynchronously.
   - **Confidence:** HIGH -- standard PyTorch best practice. The data is not accessed on CPU after the `.to()` call, so async transfer is safe.

4. **Gradient statistics computed via Python loop** (lines 360-373): Iterates `model.named_parameters()` in Python to collect grad norms into a list, then computes mean/max/var in Python. Could use vectorized approach.
   - **Confidence:** LOW priority -- this runs once per epoch (not per batch), and the loop is over ~100 parameters, not 10000+. Unlikely to be a meaningful bottleneck.

### Bottleneck 5: `_apply_conv_bn` Permute/Reshape per Call -- LOW-MEDIUM impact

**File:** `src/octonion/baselines/_network.py` lines 53-82
**What:** Every BN application in conv topology requires permute + reshape before BN, then reshape + permute after. For hypercomplex: 5D tensor `[B, ch, dim, H, W]` -> `[-1, ch, dim]` -> BN -> `[B, H, W, ch, dim]` -> `[B, ch, dim, H, W]`. This is 2 permutations + 2 reshapes per BN call.

**Impact:** With ~60 BN calls per forward pass (28 blocks * 2 main BN + shortcut BN), this is ~120 permute+reshape pairs. In PyTorch, `reshape` on contiguous tensors is a metadata-only operation (no copy). `permute` creates a view (no copy) but the result is non-contiguous, so the subsequent `reshape` after `permute` will force a contiguous copy.

**Fix:** Use `torch.movedim` which can sometimes avoid copies. Alternatively, redesign BN to accept conv-layout tensors directly (reshape internally). Or simply ensure the intermediate tensors are made contiguous explicitly with `.contiguous()` before the reshape to avoid hidden copies in unexpected places.

**Confidence:** MEDIUM -- need profiling to confirm this is actually a bottleneck vs just metadata overhead. The copy cost scales with tensor size, so for large spatial dimensions it matters more.

### Bottleneck 6: Structure Constants Device Transfer -- LOW impact

**File:** `src/octonion/_multiplication.py` line 82, and `_algebra_conv.py` lines 521, 614
**What:** `octonion_mul` calls `STRUCTURE_CONSTANTS.to(device=a.device, dtype=common_dtype)` every call. The conv layers do similar: `C = STRUCTURE_CONSTANTS.to(device=W_stack.device, dtype=W_stack.dtype)`.

**Impact:** PyTorch's `.to()` is a no-op when device and dtype already match (returns the same tensor). The overhead is the device/dtype comparison check, which is negligible per call. However, this creates a pattern where the module-level constant is not on the correct device, requiring the check.

**Fix:** Register structure constants as a `nn.Module` buffer in each layer at `__init__` time. The buffer automatically moves with the module when `.to(device)` is called:
```python
self.register_buffer("_C", STRUCTURE_CONSTANTS.to(dtype=dtype), persistent=False)
```
Using `persistent=False` excludes it from `state_dict()`, avoiding checkpoint bloat.

**Confidence:** HIGH for correctness, LOW for performance impact (already fast due to PyTorch's internal caching).

## Safe Optimizations (Three Tiers)

### Tier 1: Zero-Risk (Pure Infrastructure, No Math Change)

These optimizations have no effect on mathematical results. They affect only execution speed and memory usage.

| # | Optimization | File | Change | Expected Impact | Risk |
|---|---|---|---|---|---|
| T1-1 | `optimizer.zero_grad(set_to_none=True)` | `_trainer.py:341` | Add kwarg | 5-10% memory bandwidth savings per step | None |
| T1-2 | `torch.backends.cudnn.benchmark = True` | `_trainer.py` (before training loop) | Add 1 line | 10-30% conv operation speedup after warmup | Non-determinism (already accepted per seed policy) |
| T1-3 | `non_blocking=True` on data `.to(device)` | `_trainer.py:341` | Add kwarg | Overlaps CPU-GPU transfer with computation | None |
| T1-4 | Vectorize `_tril_to_symmetric` | `_normalization.py:23-45` | Replace loop with `tril_indices` indexing | Eliminates 36 Python iterations per BN call per feature | None (mathematically identical) |
| T1-5 | Pre-compute `tril_indices` at init | `_normalization.py` (QuaternionBN, OctonionBN `__init__`) | Store indices as buffer | Avoids recomputing indices per call | None |

### Tier 2: Moderate Risk (Caching / Algorithmic Improvement)

These optimizations change the computational approach but produce mathematically identical results. They require equivalence tests.

| # | Optimization | File | Change | Expected Impact | Risk |
|---|---|---|---|---|---|
| T2-1 | Fuse `OctonionDenseLinear` forward (einsum pattern) | `_algebra_linear.py:279-313` | Replace Python loop with einsum+single F.linear | Eliminates 64-iteration Python loop; single GPU kernel | State dict compatible (same ParameterList structure) |
| T2-2 | Cache fused weight in eval mode (conv layers) | `_algebra_conv.py` (OctonionConv2d, QuaternionConv2d) | Cache fused weight during eval, invalidate on train | Faster validation/inference (no weight recompute) | Must invalidate cache correctly |
| T2-3 | Register structure constants as buffer | All algebra modules | `register_buffer("_C", ...)` at init | Avoids per-forward `.to()` call; auto device migration | Adds key to module (use `persistent=False`) |
| T2-4 | Add `torch.compile` config option | `_config.py`, `_trainer.py` | New `use_compile: bool = False` in TrainConfig | 2-3x potential speedup (opt-in) | ROCm Triton backend may have issues |

### Tier 3: Experimental (Config-Gated, Requires Validation)

These optimizations may produce slightly different numerical results due to precision changes. They must be opt-in and validated.

| # | Optimization | File | Change | Expected Impact | Risk |
|---|---|---|---|---|---|
| T3-1 | AMP float16 with BN protection | `_normalization.py`, `_trainer.py` | Wrap BN whitening in `autocast(enabled=False)` | 30-50% training speedup, 50% memory reduction | May affect convergence; BN needs fp32 protection |
| T3-2 | `torch.compile(model)` | `_trainer.py` or training script | Compile model before training | 2-3x potential on standard ops | Graph breaks from ParameterList, try/except in BN |
| T3-3 | Larger batch size with AMP | Training script | batch_size=256 enabled by AMP memory savings | Better GPU utilization, fewer gradient steps | Changes training dynamics |

## AMP Safety Analysis for Hypercomplex Operations

### What AMP Does
`torch.amp.autocast` automatically casts operations to float16 where safe. Operations like linear layers and convolutions run in float16 (faster), while reductions, loss computation, and some operations stay in float32. `GradScaler` prevents gradient underflow in float16.

### Safety Verdict Per Operation

| Operation | AMP Safe? | Rationale |
|---|---|---|
| `F.linear(x, W)` in all algebra layers | YES | Standard linear algebra, AMP's primary target |
| `F.conv2d(x, W)` in all conv layers | YES | Standard convolution, AMP's primary target |
| Hamilton product weight cat in QuaternionConv2d | YES | torch.cat/stack are dtype-preserving |
| Einsum for structure constants fusion | YES | C has values {-1,0,+1}, exact in float16 |
| `torch.linalg.cholesky` in BN whitening | **NO** | Requires float32 for numerical stability |
| `torch.linalg.solve_triangular` in BN | **NO** | Requires float32 for numerical stability |
| Covariance computation `einsum("bfi,bfj->fij",...)` in BN | CAUTION | Accumulation over batch may lose precision |
| `_tril_to_symmetric` gamma reconstruction | YES | Simple indexing, no precision concern |
| SplitActivation (ReLU/GELU) | YES | Pointwise, AMP handles natively |
| NormPreservingActivation (norm + scale) | CAUTION | Division by norm could amplify float16 errors |
| Cross-entropy loss | YES | AMP keeps loss in float32 by default |

### BN Protection Pattern
```python
# In QuaternionBatchNorm._whiten and OctonionBatchNorm._whiten:
def _whiten(self, x_centered, cov):
    with torch.amp.autocast("cuda", enabled=False):
        # Force float32 for numerically sensitive operations
        cov_f32 = cov.float()
        x_f32 = x_centered.float()
        # ... Cholesky, solve_triangular, matmul ...
    return result.to(x_centered.dtype)
```

### AMP Recommendation
Enable AMP for the conv/linear forward paths (the dominant compute). Protect BN whitening with `autocast(enabled=False)` to keep Cholesky in float32. The `use_amp` config flag already exists in `TrainConfig` -- the work is adding the BN protection and testing convergence.

### float16 vs bfloat16 on ROCm
ROCm hardware supports both. bfloat16 has the same exponent range as float32 (no gradient underflow, GradScaler unnecessary) but some ROCm operations may not be fully optimized for bfloat16. float16 is more mature on AMD GPUs. **Recommendation:** Use float16 with GradScaler (the existing AMP path).

## torch.compile Analysis

### Viability on ROCm 7.2 + PyTorch 2.9.1

**Status:** Functional but with caveats. AMD's ROCm blog reports 3.5x speedup on ResNet-152 and 2.3x on ViT using `backend="inductor"` on MI210. However, the Triton backend on ROCm has known issues with:
- Dependency version mismatches (pytorch-triton-rocm versions)
- Autotuning maturity (heuristics optimized for NVIDIA warp size 32, not AMD wavefront size 64)
- Some vision workloads in 16-bit mode

### Potential Graph Breaks in This Codebase

| Source | Location | Severity | Fix Available? |
|---|---|---|---|
| Python loop in OctonionDenseLinear | `_algebra_linear.py:302-306` | HIGH | Yes -- fuse to einsum (Tier 2) |
| `try/except` Cholesky fallback in BN | `_normalization.py:297-308` | MEDIUM | Yes -- use `cholesky_ex` instead |
| `torch.stack(list(self.weights))` | `_algebra_conv.py:613` | LOW | Yes -- pre-stack at init |
| Python dict `linear_cache` | `_algebra_linear.py:296` | HIGH | Yes -- eliminated by fused pattern |
| `isinstance` checks in AlgebraNetwork | `_network.py:569-577` | LOW | Tracing captures this |

### Recommendation
1. Complete Tier 2 optimizations first (eliminate Python loops, fuse OctonionDenseLinear)
2. Then try `torch.compile(model, backend="inductor", mode="default")`
3. Use `TORCH_LOGS=graph_breaks` to identify remaining breaks
4. If graph breaks dominate, try `torch.compile` on individual submodules instead of full model
5. Add as opt-in config flag: `use_compile: bool = False` in TrainConfig

## ROCm-Specific Considerations

### ROCm 7.2 + PyTorch 2.9.1 Environment

1. **MIOpen algorithm selection**: `torch.backends.cudnn.benchmark = True` maps to MIOpen's equivalent. Safe for fixed input sizes.

2. **AMP device type**: The codebase correctly uses `torch.amp.autocast("cuda", ...)` which works for ROCm -- PyTorch maps "cuda" to the HIP/ROCm backend.

3. **Kernel launch overhead**: AMD GPUs (wavefront size 64) benefit greatly from the fused convolution approach. Each GPU kernel launch has fixed overhead (~5-15us on AMD); reducing 64 launches to 1 (as OctonionConv2d does) saves ~300-900us per conv layer.

4. **Memory bandwidth**: ROCm GPUs typically have high memory bandwidth. The `torch.stack` in conv layers creates temporary copies that stress bandwidth. Eliminating these via pre-stacked parameters would help.

5. **batch_size tuning**: Current batch_size=128 is standard. With AMP enabled (50% memory reduction), batch_size=256 becomes feasible. However, this changes training dynamics (fewer gradient updates per epoch), so it should be a separate experiment, not a default change.

## Data Pipeline Assessment

### Current State (Already Well-Optimized)
- `num_workers=4` -- appropriate for CIFAR-10 (small images, fast loading)
- `pin_memory=True` -- enables async CPU->GPU transfer
- `persistent_workers=True` -- avoids worker restart per epoch
- `multiprocessing_context="spawn"` -- avoids fork+CUDA deadlocks
- `drop_last=True` on training loader -- ensures fixed batch size
- Standard torchvision augmentation (RandomCrop, RandomHorizontalFlip)

### Minimal Improvements Available
- Add `non_blocking=True` to `.to(device)` calls (covered in Tier 1)
- Consider `prefetch_factor=4` if GPU is starved (unlikely for CIFAR-10)
- GPU-based augmentation via `torchvision.transforms.v2` -- not worth it for 32x32 images

**Verdict:** Data pipeline is not the bottleneck. Focus optimization on forward/backward pass.

## Expected Training Time Profile

### CIFAR-10, depth=28 ResNet, batch_size=128, 200 epochs

| Algebra | base_filters (after multiplier) | Total Params (approx) | Fused Conv Kernel Size | Relative Forward Time |
|---------|------|------|------|------|
| R | 128 (16*8) | ~600K | [128, 128, 3, 3] native | 1.0x (baseline) |
| C | 64 (16*4) | ~600K | [128, 128, 3, 3] (2 weight mats) | ~1.2-1.5x |
| H | 32 (16*2) | ~600K | [128, 128, 3, 3] (fused Hamilton) | ~1.5-2x |
| O | 16 (16*1) | ~600K | [128, 128, 3, 3] (fused struct const) | ~2-4x |

**Key insight:** The real baseline has 128 base_filters but uses standard `nn.Conv2d` which is maximally optimized by cuDNN/MIOpen. The octonion baseline has only 16 base_filters but the fused weight matrix is `[8*16, 8*16, 3, 3] = [128, 128, 3, 3]` -- the SAME effective kernel size. The difference is that the fused kernel must be recomputed each forward pass (einsum overhead) and the non-standard channel layout may not benefit from the same MIOpen algorithm tuning.

### Estimated Wall-Clock Times (Need Profiling to Confirm)

These are rough estimates pending actual GPU identification and profiling:

| Scenario | R (3 seeds) | C (3 seeds) | H (3 seeds) | O (3 seeds) | Total |
|---|---|---|---|---|---|
| No optimizations | 6-15h | 8-20h | 10-25h | 15-40h | 40-100h |
| Tier 1 only | 5-12h | 7-16h | 8-20h | 12-32h | 32-80h |
| Tier 1+2 | 4-10h | 6-14h | 7-18h | 10-28h | 27-70h |
| Tier 1+2+AMP | 3-7h | 4-10h | 5-12h | 7-19h | 19-48h |

**Recommendation:** Profile 5 epochs with each algebra before committing to the full 200-epoch run. This gives a reliable per-epoch time estimate with only ~2.5% of the total compute budget.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Fused kernel optimization | Custom HIP/CUDA kernels | `torch.compile(backend="inductor")` | Out of scope per project constraints; inductor handles kernel fusion |
| Mixed precision management | Manual dtype casting per operation | `torch.amp.autocast` + `GradScaler` | Already implemented correctly in trainer |
| Convolution algorithm selection | Manual algorithm benchmarking | `torch.backends.cudnn.benchmark = True` | MIOpen handles this transparently |
| Gradient accumulation optimization | Manual gradient zeroing tricks | `optimizer.zero_grad(set_to_none=True)` | Built-in PyTorch, well-tested |
| Memory profiling | Custom VRAM tracking code | `torch.cuda.max_memory_allocated()` + `torch.profiler` | Already have basic VRAM tracking; torch.profiler gives detailed breakdown |
| Profiling for bottleneck identification | Manual timing with `time.time()` | `torch.profiler.profile` with `tensorboard_trace_handler` | Gives per-op breakdown, kernel launch times, memory allocation |

## Common Pitfalls

### Pitfall 1: AMP Breaking Cholesky in Batch Normalization
**What goes wrong:** Enabling AMP globally autocasts Cholesky decomposition inputs to float16. The 8x8 covariance matrix for OctonionBatchNorm becomes ill-conditioned in float16, causing Cholesky failures and frequent fallback to heavy regularization.
**Why it happens:** AMP's autocast applies to all operations in the forward pass, including linalg operations that need float32 precision.
**How to avoid:** Wrap BN whitening computation in `torch.amp.autocast("cuda", enabled=False)`. This forces float32 for Cholesky while allowing the rest of the forward pass to benefit from float16.
**Warning signs:** Frequent "Cholesky decomposition failed" warnings in logs; training loss spikes; BN condition number warnings increasing.

### Pitfall 2: torch.compile Graph Breaks Eliminating Speedup
**What goes wrong:** torch.compile fragments the model into many small subgraphs due to Python control flow, making compilation overhead exceed speedup.
**Why it happens:** Python dict lookups in OctonionDenseLinear, try/except blocks in BN Cholesky fallback, and ParameterList iteration all cause graph breaks.
**How to avoid:** Complete Tier 2 optimizations first (eliminate Python loops in forward path). Use `cholesky_ex` instead of try/except for Cholesky failure detection. Then apply torch.compile.
**Warning signs:** `TORCH_LOGS=graph_breaks` shows many breaks; compiled model is slower than eager mode.

### Pitfall 3: cudnn.benchmark Re-benchmarking on Variable Batch Sizes
**What goes wrong:** If the last validation batch has a different size (no `drop_last`), cudnn.benchmark re-runs algorithm selection for that size, causing a periodic slowdown every validation epoch.
**Why it happens:** The training loader uses `drop_last=True` but validation loader does not.
**How to avoid:** Accept the one-time re-benchmark cost per validation epoch (it is small). Or add `drop_last=True` to validation loader (loses a few validation samples, negligible impact on accuracy measurement).
**Warning signs:** Periodic timing spikes at the start of validation within each epoch.

### Pitfall 4: Optimization Changes Invalidating Benchmark Reproduction
**What goes wrong:** Applying optimizations that change numerical behavior (even slightly) invalidates comparison with published results.
**Why it happens:** AMP changes precision; torch.compile may reorder operations; cudnn.benchmark selects different algorithms.
**How to avoid:** Keep a "reproduction mode" with `use_amp=False`, `use_compile=False` for benchmark reproduction runs. Apply optimizations only for performance experiments after reproduction is validated. cudnn.benchmark is safe because it only changes algorithm selection, not mathematical results (within floating-point tolerance).
**Warning signs:** Reproduced results diverge from published targets after applying optimizations.

### Pitfall 5: State Dict Incompatibility After Parameter Restructuring
**What goes wrong:** Changing ParameterList to stacked Parameter changes state_dict keys, breaking loading of existing checkpoints.
**Why it happens:** Checkpoint files store exact key names from `model.state_dict()`.
**How to avoid:** Keep the ParameterList structure but optimize the forward pass (fused einsum). If restructuring is needed, provide backward-compatible checkpoint loading (key name mapping). Better yet: add optimizations as new code in the forward method without changing parameter storage.
**Warning signs:** `KeyError` or tensor size mismatch when loading pre-optimization checkpoints.

## Code Examples

### Example 1: Vectorized `_tril_to_symmetric`

```python
# Replaces the Python-loop version in _normalization.py
# Source: PyTorch torch.tril_indices documentation

class OctonionBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.dim = 8
        self._tril_size = self.dim * (self.dim + 1) // 2  # 36

        # Precompute tril indices (registered as non-persistent buffer)
        rows, cols = torch.tril_indices(self.dim, self.dim)
        self.register_buffer("_tril_rows", rows, persistent=False)
        self.register_buffer("_tril_cols", cols, persistent=False)

        # ... rest of init ...

    def _tril_to_symmetric(self, tril_flat):
        """Vectorized: flat lower-triangular -> symmetric matrix."""
        batch_shape = tril_flat.shape[:-1]
        mat = torch.zeros(
            *batch_shape, self.dim, self.dim,
            device=tril_flat.device, dtype=tril_flat.dtype,
        )
        mat[..., self._tril_rows, self._tril_cols] = tril_flat
        mat[..., self._tril_cols, self._tril_rows] = tril_flat
        return mat
```

### Example 2: Fused OctonionDenseLinear Forward

```python
# Replaces the Python-loop version in _algebra_linear.py
# Pattern from existing OctonionConv2d in _algebra_conv.py

class OctonionDenseLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Keep ParameterList for backward compatibility
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            for _ in range(8)
        ])

        # Register structure constants as non-persistent buffer
        self.register_buffer(
            "_C", STRUCTURE_CONSTANTS.to(dtype=dtype), persistent=False
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, 8, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        octonion_init(list(self.weights), criterion="glorot")

    def forward(self, x):
        """Fused forward: single F.linear via structure-constant weight fusion."""
        batch_shape = x.shape[:-2]

        # Stack weights: [8, out_f, in_f]
        W_stack = torch.stack(list(self.weights))

        # Fuse with structure constants:
        # fused[k, j] = sum_i C[i,j,k] * W_i  ->  [8, 8, out_f, in_f]
        fused = torch.einsum("ijk, iof -> kjof", self._C, W_stack)

        # Reshape to [8*out_f, 8*in_f] for single F.linear
        fused_flat = fused.permute(0, 2, 1, 3).reshape(
            8 * self.out_features, 8 * self.in_features
        )

        # Flatten input: [..., in_f, 8] -> [..., 8*in_f]
        x_flat = x.reshape(*batch_shape, 8 * self.in_features)

        # Single F.linear call
        out_flat = F.linear(x_flat, fused_flat)

        # Reshape: [..., 8*out_f] -> [..., out_f, 8]
        result = out_flat.reshape(*batch_shape, self.out_features, 8)

        if self.bias is not None:
            result = result + self.bias

        return result
```

### Example 3: AMP-Safe BN Whitening

```python
# Source: PyTorch AMP documentation (torch.amp.autocast)

def _whiten(self, x_centered, cov):
    """Whiten with AMP-safe Cholesky (forced float32)."""
    # Disable autocast for numerically sensitive Cholesky
    with torch.amp.autocast("cuda", enabled=False):
        cov_f32 = cov.float()
        x_f32 = x_centered.float()

        eye = torch.eye(
            self.dim, device=cov.device, dtype=torch.float32
        ).unsqueeze(0)
        cov_reg = cov_f32 + self.eps * eye

        L = torch.linalg.cholesky(cov_reg)

        # Condition number monitoring
        if self.training:
            with torch.no_grad():
                diag = L.diagonal(dim1=-2, dim2=-1)
                diag_abs = diag.abs().clamp(min=1e-12)
                per_feature_cond = (
                    diag_abs.max(dim=-1).values / diag_abs.min(dim=-1).values
                )
                self.last_cond.fill_(per_feature_cond.max().item())

        identity = eye.expand_as(L)
        L_inv = torch.linalg.solve_triangular(L, identity, upper=False)

        x_col = x_f32.unsqueeze(-1)
        w = (L_inv.unsqueeze(0) @ x_col).squeeze(-1)

    # Return in original dtype (autocast handles downstream)
    return w.to(x_centered.dtype)
```

### Example 4: Training Loop Optimizations

```python
# Applied to _trainer.py train_model()
# Sources: PyTorch Performance Tuning Guide, cudnn.benchmark documentation

def train_model(model, train_loader, val_loader, config, output_dir, device="cuda", ...):
    # Enable algorithm benchmarking for fixed-size inputs
    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    # Optional torch.compile
    if getattr(config, 'use_compile', False):
        model = torch.compile(model, backend="inductor", mode="default")

    # ... setup ...

    for epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            # Non-blocking async transfer (works with pin_memory=True)
            inputs = batch[0].to(device, non_blocking=True)
            targets = batch[1].to(device, non_blocking=True)

            # set_to_none saves memset per parameter
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        # ... rest of epoch ...
```

### Example 5: Profiling Baseline (Run Before Optimizing)

```python
# scripts/profile_baseline.py
# Run inside container: docker compose run --rm dev uv run python scripts/profile_baseline.py

import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Build model and data
model = build_cifar_model(algebra="O", depth=28)
model = model.to("cuda")
batch = torch.randn(128, 3, 32, 32, device="cuda")

# Warmup
for _ in range(10):
    model(batch)

# Profile
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with record_function("forward"):
        output = model(batch)
    with record_function("backward"):
        output.sum().backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("profile_trace.json")
```

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest + hypothesis (already configured) |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `docker compose run --rm dev uv run pytest tests/ -x --timeout=60 -k "not slow"` |
| Full suite command | `docker compose run --rm dev uv run pytest tests/ -x` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BASE-01 | Param counts unchanged after optimization | unit | `docker compose run --rm dev uv run pytest tests/test_baselines_linear.py tests/test_baselines_reproduction.py -x -k "param"` | Yes |
| BASE-02 | Complex forward pass unchanged | unit | `docker compose run --rm dev uv run pytest tests/test_baselines_linear.py -x -k "complex"` | Yes |
| BASE-03 | Quaternion forward pass unchanged | unit | `docker compose run --rm dev uv run pytest tests/test_baselines_linear.py -x -k "quaternion"` | Yes |
| PERF-01 | Vectorized `_tril_to_symmetric` matches original | unit | New test: compare old vs new outputs | Wave 0 |
| PERF-02 | Fused OctonionDenseLinear matches original | unit | New test: compare fused vs loop forward | Wave 0 |
| PERF-03 | AMP with BN protection converges | integration | New test: short training with use_amp=True | Wave 0 |
| PERF-04 | All existing tests pass with optimizations | regression | `docker compose run --rm dev uv run pytest tests/ -x --timeout=60 -k "not slow"` | Yes |

### Sampling Rate
- **Per task commit:** `docker compose run --rm dev uv run pytest tests/ -x --timeout=60 -k "not slow"`
- **Per wave merge:** `docker compose run --rm dev uv run pytest tests/ -x`
- **Phase gate:** Full suite green before verification

### Wave 0 Gaps
- [ ] `tests/test_perf_equivalence.py` -- verifies optimized code paths produce identical outputs to original implementations (within floating-point tolerance)
- [ ] `scripts/profile_baseline.py` -- profiling script to measure bottlenecks before optimization
- [ ] Benchmark timing harness (measure per-epoch time for each algebra before/after optimization)

*(These gaps are small -- the existing test suite already covers all mathematical correctness. Wave 0 only needs equivalence tests for new code paths and a profiling baseline.)*

## Open Questions

1. **What is the actual GPU model in the dev container?**
   - What we know: ROCm 7.2, PyTorch 2.9.1, likely AMD Instinct or Radeon Pro
   - What's unclear: Specific GPU model determines AMP speedup magnitude, memory limits, and torch.compile effectiveness
   - Recommendation: Run `docker compose run --rm dev uv run python -c "import torch; print(torch.cuda.get_device_name(0)); print(f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')"` before planning specific batch sizes or memory budgets

2. **Does torch.compile work reliably with the current ParameterList + einsum pattern?**
   - What we know: torch.compile handles standard ModuleList/ParameterList but may create graph breaks
   - What's unclear: Whether the specific einsum + ParameterList pattern causes breaks on ROCm
   - Recommendation: Test with `TORCH_LOGS=graph_breaks` on a small model before committing. This is a 10-minute experiment.

3. **What is the actual per-epoch time for each algebra on the target hardware?**
   - What we know: Rough estimates based on FLOP ratios (R ~1x, C ~1.5x, H ~2x, O ~3-4x)
   - What's unclear: Actual kernel launch overhead, memory bandwidth utilization, MIOpen algorithm selection
   - Recommendation: Run 5-epoch warmup with each algebra and measure wall-clock time. Budget ~30 minutes for this profiling step.

4. **Does AMP with float16 degrade convergence for octonion networks?**
   - What we know: AMP is safe for conv/linear; BN whitening needs float32 protection
   - What's unclear: Whether octonion-specific numerical patterns (8D accumulation) are more sensitive to float16
   - Recommendation: Run 20-epoch A/B comparison (AMP vs float32) for each algebra. Compare validation accuracy curves. Budget ~2 hours.

## Sources

### Primary (HIGH confidence)
- Direct code audit of `src/octonion/baselines/` -- all bottlenecks identified from reading the source
- [PyTorch torch.amp documentation](https://docs.pytorch.org/docs/stable/amp.html) -- AMP usage, autocast behavior, GradScaler
- [PyTorch torch.compile documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) -- compilation modes, backends, troubleshooting
- [PyTorch optimizer.zero_grad documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html) -- set_to_none parameter
- [PyTorch cudnn.benchmark documentation](https://docs.pytorch.org/docs/stable/backends.html) -- convolution algorithm selection

### Secondary (MEDIUM confidence)
- [AMD ROCm blog: torch.compile on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/torch_compile/README.html) -- 3.5x ResNet-152 speedup on MI210, inductor backend
- [AMD ROCm blog: AMP on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/automatic-mixed-precision/README.html) -- 46% training speedup, 50% memory reduction
- [State of PyTorch Hardware Acceleration 2025](https://tunguz.github.io/PyTorch_Hardware_2025/) -- ROCm maturity assessment, Triton limitations
- [PyTorch einsum + opt_einsum](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html) -- einsum optimization with contraction order
- [ROCm precision support](https://rocm.docs.amd.com/en/latest/reference/precision-support.html) -- float16/bfloat16 hardware support
- [PyTorch batched Cholesky performance](https://www.pugetsystems.com/labs/hpc/pytorch-for-scientific-computing-quantum-mechanics-example-part-3-code-optimizations-batched-matrix-operations-cholesky-decomposition-and-inverse-1225/) -- GPU batched linalg performance characteristics

### Tertiary (LOW confidence)
- Training time estimates are rough projections based on FLOP ratios, not measured on actual hardware
- torch.compile graph break behavior with the specific ParameterList+einsum pattern is inferred from general documentation, not tested
- bfloat16 vs float16 performance comparison on ROCm is based on general documentation, not benchmarked

## Metadata

**Confidence breakdown:**
- Bottleneck identification: HIGH -- based on direct source code reading and established PyTorch performance patterns
- Tier 1 optimizations: HIGH -- well-documented PyTorch best practices, zero mathematical impact
- Tier 2 optimizations: HIGH -- fused einsum pattern already proven in the codebase (conv layers); applying same pattern to linear
- Tier 3 optimizations (AMP, compile): MEDIUM -- ROCm support is functional but less mature than CUDA; convergence impact needs validation
- Training time estimates: LOW -- need actual profiling on target hardware
- ROCm-specific behaviors: MEDIUM -- documented by AMD but not verified on this specific workload

**Research date:** 2026-03-13
**Valid until:** 2026-04-13 (stable domain; PyTorch 2.9.1 and ROCm 7.2 are fixed versions)
