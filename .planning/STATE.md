---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Phase 4 context gathered
last_updated: "2026-03-16T07:25:54.170Z"
last_activity: 2026-03-13 -- Launched CIFAR-10 reproduction training (Docker container running)
progress:
  total_phases: 9
  completed_phases: 2
  total_plans: 25
  completed_plans: 22
  percent: 95
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-07)

**Core value:** Determine empirically whether octonionic representations provide measurable advantages over quaternionic, complex, and real-valued alternatives for geometric reasoning in ML
**Current focus:** Phase 3: Baseline Implementations

## Current Position

Phase: 3 of 9 (Baseline Implementations) -- In Progress
Plan: 03-09 -- Training in progress (4 algebras x 3 seeds x 200 epochs, ~40-100h remaining)
Status: Executing
Last activity: 2026-03-13 -- Launched CIFAR-10 reproduction training (Docker container running)

Progress: [██████████] 95%

## Performance Metrics

**Velocity:**
- Total plans completed: 14
- Average duration: 9min
- Total execution time: 2.07 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-00 | 1 | 1min | 1min |
| 01-01 | 1 | 9min | 9min |
| 01-02 | 1 | 16min | 16min |
| 01-03 | 1 | 13min | 13min |
| 01-04 | 1 | 4min | 4min |
| 01-05 | 1 | 5min | 5min |
| 02-01 | 1 | 12min | 12min |
| 02-02 | 1 | 8min | 8min |
| 02-04 | 1 | 7min | 7min |

| 03-01 | 1 | 12min | 12min |
| 03-02 | 1 | 9min | 9min |
| 03-04 | 1 | 11min | 11min |

| 03-05 | 1 | 13min | 13min |

**Recent Trend:**
- Last 5 plans: 02-03 (13min), 03-01 (12min), 03-02 (9min), 03-04 (11min), 03-05 (13min)
- Trend: Comparison runner plan (13min) consistent with Phase 3 complexity

*Updated after each plan completion*
| Phase 03 P03 | 10min | 2 tasks | 5 files |
| Phase 03 P07 | 5min | 2 tasks | 4 files |
| Phase 03 P08 | 26min | 2 tasks | 3 files |
| Phase 03 P06 | 5min | 3 tasks | 7 files |
| Phase 03 P10 | 7min | 2 tasks | 4 files |
| Phase 03 P11 | 10min | 2 tasks | 3 files |
| Phase 03 P12 | 9min | 2 tasks | 5 files |
| Phase 03 P09 | 90min | 1 tasks | 5 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Roadmap: 9 phases derived from 14 requirements with go/no-go gate at Phase 5
- Roadmap: Phase 3 (Baselines) can run in parallel with Phases 2/4
- [Phase 01-00]: Used rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1 as base image (Python 3.12, PyTorch 2.9.1, ROCm 7.2)
- [Phase 01-00]: Project source mounted as volume with uv in copy mode for container compatibility
- [Phase 01-01]: Structure constants tensor has 64 non-zero entries (not 50 as estimated in research)
- [Phase 01-01]: CD-Fano basis permutation P=[0,1,2,5,3,7,6,4] (pure permutation, no sign flips) resolves Open Question 1
- [Phase 01-01]: Distributivity tests use [-1e3, 1e3] range to avoid float64 precision artifacts
- [Phase 01-02]: Moufang tests use [-1, 1] input range to keep triple products O(1) for strict 1e-12 absolute tolerance
- [Phase 01-02]: conftest strategies renamed: raw tensor strategies are octonion_tensors etc., Octonion-wrapping strategies are octonions etc.
- [Phase 01-02]: from_quaternion_pair/to_quaternion_pair use raw CD basis (simple concatenation), not Fano-permuted
- [Phase 01-03]: Cross product uses Im(Im(a)*Im(b)) for antisymmetry with non-pure inputs
- [Phase 01-03]: Exp/log roundtrip valid only within principal branch (||v|| < pi)
- [Phase 01-03]: Mul matrix tests use rtol=1e-10 for einsum vs matmul path rounding differences
- [Phase 01-03]: OctonionLinear uses expand_as for parameter broadcasting to batch dimensions
- [Phase 01-04]: float32 display threshold 1e-7, float64 threshold 1e-14 (matched to dtype epsilon)
- [Phase 01-04]: Copy constructor unwraps via .components, no deep copy (shared tensor)
- [Phase 01-05]: OctonionLinear default dtype changed from float64 to float32 (PyTorch convention)
- [Phase 01-05]: octonion_mul uses torch.promote_types for mixed-dtype safety
- [Phase 01-05]: exp/log use isinstance guard for raw tensor auto-coercion to Octonion
- [Phase 01]: OctonionLinear default dtype changed from float64 to float32 (PyTorch convention)
- [Phase 01]: octonion_mul uses torch.promote_types for mixed-dtype safety
- [Phase 01]: exp/log use isinstance guard for raw tensor auto-coercion to Octonion
- [Phase 02-01]: Numeric Jacobian eps=1e-5 for tests (reduces roundoff on adversarial Hypothesis inputs)
- [Phase 02-01]: Test tolerances: atol=1e-7 standard, 1e-6 transcendental (100x tighter than 1e-5 criterion)
- [Phase 02-01]: GHR Wirtinger pair uses 1/8 normalization (octonionic extension of quaternionic 1/4)
- [Phase 02-01]: Cross product Jacobian via imaginary-block extraction from mul Jacobian
- [Phase 02-02]: Autograd Functions recompute Jacobians in backward (not cached from forward) for create_graph=True
- [Phase 02-02]: Cross product backward zeros grad_output[..., 0] to exclude C[i,j,0] terms
- [Phase 02-02]: Inner product Function returns scalar shape [] matching torch.sum convention
- [Phase 02-02]: Exp/log backward use sqrt(r_sq + 1e-30) for sqrt gradient stability
- [Phase 02-04]: GPU/CPU parity tolerance 1e-12 at float64 (~4500x machine epsilon for ROCm)
- [Phase 02-04]: CR analyticity extracts putative c from J[:, 0], verifies J == L_c via Frobenius norm
- [Phase 02-04]: LR scaling uses simple 1/K inverse heuristic (K = octonionic/real gradient norm ratio)
- [Phase 02-03]: all_parenthesizations generates right-associated trees first; fully left-associated is trees[-1]
- [Phase 02-03]: evaluate_tree dispatches to autograd Functions (not raw octonion_mul) for correct parenthesization-aware backward
- [Phase 02-03]: Naive chain rule defined as always-left-to-right association baseline
- [Phase 02-03]: compose_jacobians uses analytic jacobian_mul (not numeric) for bottom-up tree Jacobian composition
- [Phase 03-01]: OctonionDenseLinear uses W*x ordering: C[i,j,k] * F.linear(x_j, W_i) matching octonion_mul convention
- [Phase 03-01]: Nonzero structure constant entries precomputed at init time; F.linear results cached per (i,j) pair
- [Phase 03-01]: Parameter matching tests use target=500000 with input_dim=32 to ensure per-unit jumps <1% of total
- [Phase 03-02]: QuaternionBN/OctonionBN gamma stored as flat lower-triangular entries (10/36) for exact param counts
- [Phase 03-02]: ComplexBN uses analytic Cayley-Hamilton V^{-1/2}; QuaternionBN/OctonionBN use Cholesky whitening
- [Phase 03-02]: Conv tensor layout [B, C, algebra_dim, *spatial] for F.conv compatibility
- [Phase 03-02]: OctonionConv precomputes nonzero structure constant entries; caches F.convNd per (i,j) pair
- [Phase 03-04]: Warmup via direct optimizer param group LR manipulation (not LambdaLR composition) for simplicity
- [Phase 03-04]: Gradient stats computed after backward pass, before zero_grad, to capture actual training gradients
- [Phase 03-04]: Optuna study uses reduced epochs (20) per trial with MedianPruner for efficient search
- [Phase 03-04]: paired_comparison returns NaN-safe results for identical inputs (common edge case)
- [Phase 03]: QuaternionLSTMCell/OctonionLSTMCell gates derived from real component of algebra computation, broadcasting scalar gates across dims
- [Phase 03]: AlgebraNetwork uses base_hidden * multiplier for hidden width, with layer factories dispatching to algebra-specific modules
- [Phase 03]: Conv2D input handling: real channels replicated across algebra dims for non-real algebras
- [Phase 03-05]: _build_simple_mlp upgraded to _SimpleAlgebraMLP class for trainable param-matched models
- [Phase 03-05]: Comparison runner uses first algebra in config as reference for param matching (not hardcoded octonion)
- [Phase 03-05]: Tests use input_dim=32 and ref_hidden=25 to ensure width steps <1% of total params for matching
- [Phase 03]: ResidualBlock uses parent network factory methods (_get_conv2d, _get_bn, _get_activation) for algebra-specific layers
- [Phase 03]: BN reshape logic factored into _apply_conv_bn helper for reuse across ResidualBlock and forward pass
- [Phase 03]: 3 stages with 1x/2x/4x base_filters and stride-2 at stages 2 and 3 matching standard CIFAR ResNet
- [Phase 03-08]: Conv2d param matching uses 10% tolerance (not 1%) due to coarse base_hidden granularity from multiplier scaling
- [Phase 03-08]: _build_conv_model helper parallels _build_simple_mlp for conv2d topology model building
- [Phase 03-08]: run_comparison uses topology-aware _build_model closure dispatching to correct builder
- [Phase 03-06]: Fused Hamilton product convolution (single kernel vs N^2 separate convs) for 16x fewer GPU kernel launches
- [Phase 03-06]: BN whitening fix: precompute L_inv instead of expanding to [features, batch, dim, dim] (avoids OOM for conv spatial dims)
- [Phase 03-06]: ref_hidden=4 for CIFAR reproduction tests to match published ~600K param scale
- [Phase 03-06]: Full GPU training runs deferred to plan 03-09; this plan validates infrastructure correctness
- [Phase 03-06]: spawn multiprocessing in CIFAR data loaders to avoid fork+CUDA deadlocks
- [Phase 03-10]: tril_indices overhead negligible vs eliminated Python loop (36 elements for dim=8, C++ call)
- [Phase 03-10]: checkpoint_every audited: TrainConfig default=10 (short runs), cifar_train_config=25 (8 checkpoints/200 epochs)
- [Phase 03-11]: Fused linear input transpose: x.transpose(-2,-1) needed before reshape to match fused_flat column ordering (j*in_f+q)
- [Phase 03-11]: Non-persistent buffer _C: register_buffer('_C', STRUCTURE_CONSTANTS, persistent=False) for auto device migration without state_dict pollution
- [Phase 03-11]: Eval cache pattern: _fused_cache as plain None attribute, invalidated by train() override, populated on first eval forward
- [Phase 03-12]: AMP tolerance set to atol=5e-3 (float16 gamma/beta arithmetic introduces ~2e-3 rounding vs fp32 path)
- [Phase 03-12]: cholesky_ex per-feature fallback: only degenerate features get extra regularization (not all-or-nothing try/except)
- [Phase 03-12]: use_compile try/except retained at trainer level (not in _whiten): acceptable there, graph-break-free in compiled model
- [Phase 03]: Use --no-match-params --ref-hidden 4 (same-width protocol) due to OOM with match_params=True ref_hidden=16 in OCTONION BN whitening
- [Phase 03]: AMP disabled for CIFAR reproduction: float16 Cholesky fails at initialization producing NaN loss
- [Phase 03]: CIFAR training time: 40-100 hours for all 12 runs due to OctonionBN 8D Cholesky overhead (131ms/forward at depth=28)

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 2: GHR calculus octonionic extension is an open research problem; may need fallback to real-component Jacobian approach
- Phase 5: Go/no-go quantitative criterion must be defined before experiments run (not after seeing results)
- Phase 8: G2 representation theory has no ML library reference; may prove intractable within project scope

## Session Continuity

Last session: 2026-03-16T07:25:54.168Z
Stopped at: Phase 4 context gathered
Resume file: .planning/phases/04-numerical-stability/04-CONTEXT.md
