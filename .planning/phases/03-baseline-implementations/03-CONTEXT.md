# Phase 3: Baseline Implementations - Context

**Gathered:** 2026-03-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Fair comparison networks exist for real, complex, quaternionic, and octonionic algebras so that every octonionic experiment has trustworthy baselines. Implements a configurable algebra-agnostic network skeleton (MLP, Conv 1D/2D, Recurrent), per-algebra linear layers with proper initialization and normalization, a comparison runner with statistical testing and plotting, and a full training utility with checkpointing, logging, and hyperparameter search. Reproduces published benchmark results for C and H baselines. Does NOT include: R8-dense-mixing baseline (Phase 7), optimization landscape experiments (Phase 5), density comparison experiments (Phase 7), or numerical stability sweeps (Phase 4).

</domain>

<decisions>
## Implementation Decisions

### Architecture Skeleton
- Configurable skeleton via `AlgebraNetwork(algebra, topology, depth=N)` pattern
- Three topology types supported at launch: MLP, Conv (1D + 2D), Recurrent
- Depth is a config parameter (variable layer count for downstream phases to sweep)
- Algebra module plug-in design: Claude's discretion on pattern (AlgebraLinear base class vs functional dispatch) given existing NormedDivisionAlgebra hierarchy and OctonionLinear
- Algebra-aware normalization layers (e.g., quaternionic batch norm from Gaudet & Maida 2018) — not just real-component norm
- Activation functions: both split activations (ReLU/GELU per component) AND algebra-aware activations (apply to norm, keep direction) supported as configurable option per experiment
- Recurrent layers: algebra-specific cells (QuaternionLSTM, ComplexGRU, etc.) following published designs, not standard structure with algebra linear
- Per-algebra weight initialization following literature: RealLinear = Kaiming/He, ComplexLinear = Trabelsi et al., QuaternionLinear = Parcollet et al., OctonionLinear = unit-norm (existing)
- Forward interface: output only. Use PyTorch `register_forward_hook()` for intermediate activation access
- Input projection: learned real-to-algebra linear embedding (nn.Linear(input_dim, hidden*algebra_dim) then reshape)
- Output projection: four strategies supported as configurable option:
  1. Real component extraction (component 0)
  2. Flatten all components to real
  3. Norm-based (algebra norm as feature)
  4. Learned embedding (algebra-to-real linear)
  - Phased training protocol: start with simple projection, switch to learned embedding after reaching acceptability threshold
- Built-in parameter counting: `param_report()` returns per-layer breakdown (layer name, algebra type, algebra units, real params, % of total)
- FLOP counting: reported for transparency but not matched across algebras
- Config driven by Python dataclasses (type-safe, IDE autocomplete, no parsing)

### Published Benchmarks
- Claude chooses which published benchmarks to reproduce for complex and quaternionic baselines (most relevant to downstream octonionic comparison)
- Cross-reference authors' open-source code for implementation correctness (Trabelsi, Parcollet, Gaudet repos)
- Reproduction criterion: within 1 standard deviation of published mean
- If reproduction fails: debug until reproduced (invest time to match results)
- Hyperparameter protocol: match published hyperparams first, then tune if reproduction fails — document both results
- Real baseline also verified against a standard published result (validates training infrastructure)
- Octonionic baseline included alongside R/C/H on same benchmarks (early signal on O performance)
- Structured reproduction report: our result vs published, confidence intervals, param count verification, pass/fail verdict
- No time cap on training runs

### Parameter Matching
- Count: all trainable params (weights, biases, normalization, embeddings — everything with requires_grad=True)
- Width computation: auto-compute via binary search for hidden width that achieves target param count within tolerance
- Tolerance: fixed at 1% (per success criteria, not configurable)
- Normalization layer params: Claude's discretion on include/exclude for scientifically defensible matching
- Convolutional layer params: Claude's discretion on matching strategy (total real params or filter count)
- Automated pytest test verifying all 4 algebra models match within 1% (directly validates SC-1/2/3)

### Training Utility
- Full training utility: `train_model(model, data, config)` with logging, checkpointing, metric tracking
- Per-algebra full tuning as default: each algebra gets its own optimizer, LR, and schedule
- Config flag to lock all algebras to same optimizer (Adam) for controlled comparison runs
- Hyperparameter search: Optuna integration for Bayesian optimization
- Seed protocol: seed-controlled (all RNGs seeded) but not CUDA deterministic (performance over bit-exactness)
- Data split: standard train/val/test (use published split if available, otherwise 80/10/10)
- LR scaling heuristic from Phase 2 NOT auto-applied — let Optuna search find optimal LR for each algebra
- No gradient clipping by default (Optuna can include clipping as search parameter)
- Gradient statistics: always logged every epoch (norm, variance, max) — essential for Phase 5
- Wall-clock timing: per-epoch and total, logged automatically
- Early stopping: configurable patience
- LR warmup: configurable number of warmup steps/epochs
- LR schedulers: configurable per-algebra (CosineAnnealingLR, StepLR, ReduceLROnPlateau, etc.)
- Weight decay: per-algebra configurable
- Data augmentation: dataset-specific, not built into trainer
- Mixed precision: optional AMP via config flag (ROCm supports AMP)
- Multi-GPU: DDP-ready from the start
- Checkpointing: full state (model + optimizer + scheduler + epoch + metrics) for training resumption
- Graceful shutdown: SIGINT handler saves checkpoint before exiting
- VRAM monitoring: peak usage tracked and logged per run
- Logging: TensorBoard for experiment visualization
- Auto-generated plots: matplotlib/seaborn convergence curves, accuracy bars with error bars, param count tables

### Comparison Runner
- `run_comparison(task, algebras=['R','C','H','O'], seeds=10)` produces structured comparison report
- Random seeds: configurable, default 10
- Sequential execution (one model at a time, single GPU)
- Built-in statistical significance testing: paired t-test, Wilcoxon, correction for multiple testing, confidence intervals, effect sizes
- Auto plots saved alongside JSON results

### Experiment Organization
- Structured directory: `experiments/{benchmark}/{algebra}/{seed}/` with config.json, metrics.json, checkpoints/, plots/
- Auto-manifest: `experiments/manifest.json` updated after each run with config hash, final metrics, status
- Model summary via torchinfo package (add as dependency)

### Claude's Discretion
- Package organization (octonion.baselines subpackage vs separate baselines package)
- Algebra module plug-in pattern (base class vs functional dispatch)
- Normalization layer param handling for matching
- Conv layer param matching strategy
- Specific published benchmarks for C and H reproduction
- Internal module organization
- Output projection default strategy
- Exact Optuna search space configuration
- Checkpoint frequency (every N epochs)

</decisions>

<specifics>
## Specific Ideas

- Output projection phased training: start with simple approach (real component, flatten, or norm), train until acceptable, then switch to learned embedding. User wants to experiment with learned output embeddings as a research direction
- Per-algebra full tuning with Adam-lock option: default allows each algebra its own optimizer/schedule, but comparison flag locks all to Adam for controlled experiments — captures both flexibility and fairness
- The 1/K LR scaling heuristic from Phase 2 is available but deliberately NOT auto-applied — Optuna search should independently discover optimal LR, which may validate or refute the heuristic
- R8-dense-mixing baseline explicitly deferred to Phase 7 — not built here
- Including octonionic baseline on reproduction benchmarks provides early training signal before Phase 5 go/no-go gate

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `NormedDivisionAlgebra` (src/octonion/_types.py): Abstract base with conjugate(), norm(), inverse(), __mul__(), components, dim — algebra-agnostic interface
- `Real`, `Complex`, `Quaternion` (src/octonion/_tower.py): Full implementations with [..., dim] tensor shapes and proper multiplication rules
- `Octonion` (src/octonion/_octonion.py): Complete octonionic type with all operations
- `OctonionLinear` (src/octonion/_linear.py): Neural layer computing (a * x) * b — pattern to follow for other algebra linear layers
- `octonion_mul` + `STRUCTURE_CONSTANTS` (src/octonion/_multiplication.py): Differentiable multiplication via einsum
- GHR autograd Functions (src/octonion/calculus/_autograd_functions.py): Custom backward passes for all octonionic ops with create_graph=True
- LR scaling utility (src/octonion/calculus/_lr_scaling.py): 1/K inverse heuristic available but not auto-applied
- Random generators (src/octonion/_random.py): Seed-controlled random_octonion(), random_unit_octonion(), random_pure_octonion()

### Established Patterns
- Immutable algebra objects — all operations return new instances
- Batch-first design: [..., dim] tensor shapes throughout
- float32 for neural layers, float64 for verification
- Verbose error messages with math context
- Module-level constant tensors built at import time

### Integration Points
- New algebra-specific linear layers (RealLinear, ComplexLinear, QuaternionLinear) will parallel OctonionLinear pattern
- Phase 4 (Numerical Stability) will use these networks for depth/precision sweeps
- Phase 5 (Optimization Landscape) will use the comparison runner and training utility for convergence profiling
- Phase 7 (Density & Geometric Claims) will extend the comparison runner with R8-dense-mixing baseline and density experiments
- Training utility will be the standard training infrastructure for all downstream experimental phases

</code_context>

<deferred>
## Deferred Ideas

- R8-dense-mixing baseline — Phase 7 (guards against "Why Not Just R8?" trap)
- K-fold cross-validation — add if needed for specific experiments
- Parallel comparison runs — keep sequential for now, single GPU
- Custom CUDA/ROCm kernels — per project constraints, premature optimization

</deferred>

---

*Phase: 03-baseline-implementations*
*Context gathered: 2026-03-08*
