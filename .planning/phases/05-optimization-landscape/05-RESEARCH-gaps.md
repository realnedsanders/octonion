# Phase 5: Optimization Landscape - Gap Research

**Researched:** 2026-03-21
**Domain:** Post-training analysis integration gaps in experiment pipeline
**Confidence:** HIGH (based on direct codebase inspection, not external sources)

## Summary

Phase 5 was executed across 6 plans, producing a working experiment pipeline that trains models across the full (task x algebra x optimizer x seed) matrix, saves training metrics and model checkpoints, and produces a gate verdict. However, critical post-training analysis was never integrated into the pipeline. The experiment runner (`_experiment.py` line 690) explicitly defers Hessian analysis, curvature measurement, and gradient variance collection with the comment "Post-training analysis is deferred for the full run (plan 05-06)." Plan 05-06 then focused only on the analysis *script* (`analyze_landscape.py`) which reads result.json files and looks for `hessian_spectrum` and `curvature` keys that are never populated.

The result is a pipeline with three disconnected pieces: (1) training + checkpoint saving works, (2) Hessian/curvature/gradient analysis functions work in isolation (unit tested), (3) the analysis script expects results that are never computed. Additionally, intermediate Hessian checkpoints (0.25, 0.50 fractions) are never saved because the saving logic only handles fractions 0.0 and 1.0.

**Primary recommendation:** Build a post-training analysis pass that loads saved checkpoints, runs `compute_hessian_spectrum()` and `measure_curvature()` on each, and writes results back to result.json. This can be a standalone script or integrated into the experiment runner. Fix the intermediate checkpoint saving to capture 0.25 and 0.50 fractions during training.

## Gap Analysis

### Gap 1: Post-Training Analysis Never Runs (CRITICAL)

**What is broken:**
`run_landscape_experiment()` in `_experiment.py` imports `compute_hessian_spectrum`, `measure_curvature`, and `collect_gradient_stats` (lines 28-30) but never calls them. The function ends at line 693 with only training metrics in the results dict.

**Evidence:**
- Line 690: `# Post-training analysis is deferred for the full run (plan 05-06).`
- The result.json files contain only: `train_losses`, `val_losses`, `best_val_loss`, `final_val_loss`, `epochs_trained`, `total_time_seconds`
- No `hessian_spectrum`, `curvature`, or `gradient_stats` keys exist in any result.json
- `analyze_landscape.py` warns "No Hessian spectrum data found" and "No curvature data found" when run

**Root cause:** Plan 05-05 was scoped as "build and smoke-test the pipeline" with post-training analysis planned for plan 05-06. But plan 05-06 was scoped as "analysis script + gate evaluation" -- it reads results but doesn't compute them. The Hessian/curvature computation step fell between the two plans.

**Impact on success criteria:**
- SC-1 (gradient variance): NOT MET -- no gradient variance data collected across seeds
- SC-2 (Hessian eigenspectrum): NOT MET -- no Hessian analysis run on any checkpoint
- SC-5 (Bill & Cox curvature): NOT MET -- no curvature measurement performed
- SC-3 (convergence profiles): MET -- training loss curves are saved
- SC-4 (2x convergence gate): PARTIALLY MET -- gate evaluation runs but only on smoke results (1 task, 2 optimizers, 4 algebras), not full matrix

### Gap 2: Intermediate Hessian Checkpoints Not Saved (CRITICAL)

**What is broken:**
`LandscapeConfig.hessian_checkpoints` defaults to `[0.0, 0.25, 0.5, 1.0]`, but the saving logic in lines 651-660 of `_experiment.py` only handles `frac == 0.0` (pre-training) and `frac == 1.0` (post-training). Fractions 0.25 and 0.50 are silently skipped.

**Evidence:**
```python
# Lines 651-660 of _experiment.py
if is_hessian_seed:
    for frac in config.hessian_checkpoints:
        if frac == 0.0:
            continue  # Already saved above
        if frac == 1.0:
            # Save converged model
            _save_hessian_checkpoint(...)
# NOTE: no else clause -- fractions 0.25 and 0.50 are silently ignored
```

- `find results/landscape -name "checkpoint_0.25*" -o -name "checkpoint_0.50*"` returns zero files
- Only `checkpoint_0.00.pt` and `checkpoint_1.00.pt` exist (38 total Hessian checkpoint files = 19 seeds x 2 fractions)

**Root cause:** Saving intermediate checkpoints requires either: (a) a callback mechanism inside `train_model()` to save at 25%/50% of epochs, or (b) training in stages (0-25%, save, 25-50%, save, 50-100%, save). Neither was implemented. The `train_model()` function has no callback/hook mechanism.

**Impact:** SC-2 requires "Hessian eigenspectrum at convergence: ratio of negative eigenvalues within 3x of quaternionic baseline" and the CONTEXT.md specifies "4 checkpoints: initialization, 25%, 50%, convergence." Without intermediate checkpoints, we cannot show landscape evolution during training.

### Gap 3: Experiment Coverage Incomplete

**What is broken:**
The existing results only cover a fraction of the full experiment matrix.

**Evidence from `full_report.json`:**
- Tasks completed: 1 of 9 (`algebra_native_single` only)
- Optimizers completed: 2 of 5 (`adam`, `sgd` -- missing `riemannian_adam`, `lbfgs`, `shampoo`)
- Algebras completed:
  - adam: R, O, R8D (3 of 6, missing C, H, PHM8)
  - sgd: R, C, H, O (4 of 6, missing PHM8, R8D)
- Seeds completed: mixed (adam has 2 seeds for O/R8D, sgd has 20 for R/C/H but only 1-2 for O)
- Total result files: 69 out of ~5,400 expected (9 tasks x 5 optimizers x 6 algebras x 20 seeds)

**Impact:** The gate verdict ("GREEN") is based on only 1 task with 2 seeds for O and R8D, making it statistically meaningless. The full experiment must run to produce a valid gate evaluation.

### Gap 4: Smoke Tests Don't Verify Post-Training Analysis

**What is broken:**
`test_landscape_smoke.py` validates that training completes and results are saved, but does not verify that Hessian, curvature, or gradient data is computed.

**Evidence:**
- `test_smoke_end_to_end` checks for: `train_losses`, `val_losses`, `best_val_loss` (lines 87-96)
- No assertion for `hessian_spectrum`, `curvature`, or `gradient_stats` keys
- No test calls `compute_hessian_spectrum()` or `measure_curvature()` on experiment outputs

**Impact:** The smoke test "passes" while the most scientifically important data (Hessian analysis, curvature measurement) is completely absent.

### Gap 5: Gradient Stats Not Collected During Training

**What is broken:**
`collect_gradient_stats()` and `collect_gradient_variance_across_seeds()` in `_gradient_stats.py` are standalone functions that compute gradient statistics at a single point. They are not called during or after training.

**Evidence:**
- `_experiment.py` imports `collect_gradient_stats` but never calls it
- `analyze_landscape.py` falls back to using `best_val_loss` variance across seeds as a proxy for gradient variance (line 618: `plot_data = grad_variance_data if has_grad_data else best_loss_data`)
- The gradient variance plots show "Performance Variance Across Seeds" (loss variance) not actual gradient norm variance

**Impact:** SC-1 requires "Gradient variance across 20+ random seeds characterized and compared to R/C/H baselines." Using loss variance as proxy is informative but not what the success criteria specify.

## Interface Inventory

### compute_hessian_spectrum() -- from _hessian.py

**Signature:**
```python
def compute_hessian_spectrum(
    model: nn.Module,           # PyTorch model with parameters
    loss_fn: Callable,          # loss_fn(output, target) -> scalar
    data_x: torch.Tensor,      # Input data batch
    data_y: torch.Tensor,      # Target data batch
    device: str = "cpu",        # Device for computation
    method: str = "auto",       # 'full', 'lanczos', or 'auto'
    max_full_params: int = 2000,# Auto threshold
    **lanczos_kwargs,           # n_iterations, n_samples for lanczos
) -> dict[str, Any]
```

**Returns (full method):**
```python
{
    "eigenvalues": np.ndarray,    # Sorted ascending
    "n_negative": int,            # Count of eigenvalues < -1e-10
    "n_positive": int,            # Count > 1e-10
    "n_zero": int,                # Count in [-1e-10, 1e-10]
    "trace": float,               # Sum of eigenvalues
    "spectral_norm": float,       # Max absolute eigenvalue
    "negative_ratio": float,      # n_negative / total
    "method": "full",
}
```

**Returns (lanczos method):**
```python
{
    "ritz_values": np.ndarray,          # All Ritz values across samples
    "n_negative_approx": int,           # Count of negative Ritz values
    "negative_ratio_approx": float,     # Fraction negative
    "trace_approx": float,              # Mean trace estimate
    "method": "lanczos",
    "n_iterations": int,
    "n_samples": int,
}
```

**Data requirements:**
- `data_x` and `data_y` must be the same data used for training (or a representative subset)
- For full Hessian: entire dataset fits in memory (model params < 2000)
- For Lanczos: needs `create_graph=True` in autograd (verified in Phase 2)
- Computation requires gradient graph -- model must be in eval mode but with gradients enabled

**JSON serialization note:** `eigenvalues` and `ritz_values` are numpy arrays -- must call `.tolist()` before saving to JSON.

### measure_curvature() -- from _curvature.py

**Signature:**
```python
def measure_curvature(
    model: nn.Module,           # Model at converged parameters
    loss_fn: Callable,          # loss_fn(output, target) -> scalar
    data_x: torch.Tensor,      # Input data batch
    data_y: torch.Tensor,      # Target data batch
    n_directions: int = 50,     # Random directions to sample
    n_steps: int = 51,          # Steps per direction
    step_range: float = 1.0,    # Max step size
    seed: int = 42,             # Reproducibility seed
) -> dict[str, Any]
```

**Returns:**
```python
{
    "mean_curvature": float,
    "median_curvature": float,
    "std_curvature": float,
    "curvatures": list[float],   # One per direction
    "n_directions": int,
}
```

**Data requirements:**
- Model must be at converged parameters (saves and restores weights internally)
- Uses `torch.no_grad()` for loss evaluation -- no autograd needed
- Modifies model weights temporarily (for each direction+step), then restores
- Data should be a representative batch, not necessarily the full dataset

**Performance characteristics:**
- Cost: `n_directions * n_steps` forward passes
- With defaults (50 directions x 51 steps): 2,550 forward passes per model
- For small models (base_hidden=4-16): < 10 seconds on GPU

### collect_gradient_stats() -- from _gradient_stats.py

**Signature:**
```python
def collect_gradient_stats(
    model: nn.Module,
    loss_fn: Callable,
    data_x: torch.Tensor,
    data_y: torch.Tensor,
    device: str = "cpu",
) -> dict[str, Any]
```

**Returns:**
```python
{
    "grad_norm_mean": float,
    "grad_norm_std": float,
    "grad_norm_max": float,
    "grad_norm_min": float,
    "per_layer_stats": list[dict],  # Each: {name, norm, mean, std, max, min}
}
```

### collect_gradient_variance_across_seeds() -- from _gradient_stats.py

**Signature:**
```python
def collect_gradient_variance_across_seeds(
    model_factory: Callable[[], nn.Module],
    loss_fn: Callable,
    data_x: torch.Tensor,
    data_y: torch.Tensor,
    seeds: list[int],
    n_steps: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict[str, Any]
```

**Returns:**
```python
{
    "per_seed_stats": list[list[dict]],     # [seed][step] -> gradient stats
    "cross_seed_variance": float,           # Variance of mean norms across seeds
    "mean_grad_norm_trajectory": list[float],  # Mean norm at each step (avg across seeds)
}
```

**Note:** This function retrains from scratch for each seed. It is designed for pre-experiment gradient characterization, not for extracting gradient data from already-trained models.

### _build_model() -- from _experiment.py

**Signature:**
```python
def _build_model(
    algebra: AlgebraType,
    task_name: str,
    config: LandscapeConfig,
) -> nn.Module
```

This is the key function for reconstructing models for checkpoint loading. It creates `_SimpleAlgebraMLP` instances with the correct architecture for a given (algebra, task) combination.

### Model Checkpoint Format

Checkpoints saved via `_save_hessian_checkpoint()` contain `model.state_dict()` only (raw `torch.save`). To reload:
```python
model = _build_model(algebra, task_name, config)
state_dict = torch.load(checkpoint_path, weights_only=True)
model.load_state_dict(state_dict)
```

The trainer also saves full checkpoints with optimizer/scheduler state at `checkpoint_epoch{N}.pt`, but the Hessian checkpoints are lighter (model state only).

### _build_task_data() -- from _experiment.py

**Signature:**
```python
def _build_task_data(
    task_name: str,
    config: LandscapeConfig,
    seed: int = 42,
) -> tuple[TensorDataset, TensorDataset, dict[str, Any]]
```

Returns `(train_dataset, test_dataset, metadata)`. The data is deterministic for a given seed.

### _get_loss_fn() -- from _experiment.py

```python
def _get_loss_fn(task_name: str) -> nn.Module:
    if task_name == "classification":
        return nn.CrossEntropyLoss()
    else:
        return nn.MSELoss()
```

## Integration Design

### Option A: Post-Training Analysis Script (RECOMMENDED)

Create a standalone script `scripts/run_post_analysis.py` that:

1. Walks `results/landscape/{task}/{optimizer}/{algebra}/seed_{N}/`
2. For each seed in `hessian_seeds`, loads `hessian_checkpoints/checkpoint_{frac}.pt`
3. Reconstructs the model via `_build_model(algebra, task_name, config)`
4. Loads the checkpoint state_dict
5. Builds task data via `_build_task_data(task_name, config, seed=42)` (same data seed as training)
6. Runs `compute_hessian_spectrum(model, loss_fn, data_x, data_y)` on a data subset
7. Runs `measure_curvature(model, loss_fn, data_x, data_y)` on converged models (frac=1.0)
8. Writes results back to `result.json` under `hessian_spectrum` and `curvature` keys

**Why this is better than inline analysis:**
- Does not slow down training (decoupled)
- Can re-run analysis without re-training
- Can run on partial results while training continues
- Can be parallelized across checkpoints
- Matches the existing `analyze_landscape.py` expectation (it already looks for these keys)

**Implementation sketch:**
```python
def run_post_analysis(
    results_dir: str,
    config: LandscapeConfig,
    device: str = "cpu",
) -> None:
    for task_name in config.tasks:
        loss_fn = _get_loss_fn(task_name)
        train_ds, test_ds, _ = _build_task_data(task_name, config, seed=42)
        # Use subset of test data for Hessian (memory-bounded)
        n_hessian = min(200, len(test_ds))
        hessian_x = test_ds.tensors[0][:n_hessian]
        hessian_y = test_ds.tensors[1][:n_hessian]

        for opt_name in _discovered_optimizers(results_dir, task_name):
            for alg_name in _discovered_algebras(results_dir, task_name, opt_name):
                algebra = _algebra_from_shortname(alg_name)

                for seed in config.hessian_seeds:
                    ckpt_dir = Path(results_dir) / task_name / opt_name / alg_name / f"seed_{seed}" / "hessian_checkpoints"
                    if not ckpt_dir.exists():
                        continue

                    hessian_results = {}
                    for ckpt_file in sorted(ckpt_dir.glob("checkpoint_*.pt")):
                        frac = float(ckpt_file.stem.split("_")[1])
                        model = _build_model(algebra, task_name, config)
                        model.load_state_dict(torch.load(ckpt_file, weights_only=True))
                        model = model.to(device)

                        spectrum = compute_hessian_spectrum(
                            model, loss_fn, hessian_x.to(device), hessian_y.to(device),
                            device=device,
                        )
                        # Convert numpy arrays to lists for JSON
                        key = "eigenvalues" if "eigenvalues" in spectrum else "ritz_values"
                        hessian_results[str(frac)] = spectrum[key].tolist()

                    # Curvature on converged model (frac=1.0)
                    converged_ckpt = ckpt_dir / "checkpoint_1.00.pt"
                    curvature_result = None
                    if converged_ckpt.exists():
                        model = _build_model(algebra, task_name, config)
                        model.load_state_dict(torch.load(converged_ckpt, weights_only=True))
                        curvature_result = measure_curvature(
                            model, loss_fn, hessian_x, hessian_y,
                            n_directions=config.n_curvature_directions,
                        )

                    # Update result.json
                    result_path = Path(results_dir) / task_name / opt_name / alg_name / f"seed_{seed}" / "result.json"
                    if result_path.exists():
                        with open(result_path) as f:
                            result = json.load(f)
                        result["hessian_spectrum"] = hessian_results
                        if curvature_result:
                            result["curvature"] = curvature_result["mean_curvature"]
                            result["curvature_detail"] = curvature_result
                        with open(result_path, "w") as f:
                            json.dump(result, f, indent=2)
```

### Option B: Inline Post-Training Analysis (simpler, slower)

Add analysis directly after training in `run_landscape_experiment()`, replacing the comment at line 690. This is simpler but couples analysis to training and prevents re-running analysis separately.

### Option C: Extend analyze_landscape.py (hybrid)

Make `analyze_landscape.py` compute Hessian/curvature when it finds checkpoints but no analysis results. This is convenient but mixes "analysis" (plotting/statistics) with "computation" (Hessian/curvature).

**Recommendation: Option A** -- cleanest separation of concerns. The script can run independently, supports partial results, and the analysis script already expects the output format.

### Intermediate Checkpoint Fix

To save checkpoints at 0.25 and 0.50 fractions, modify `run_landscape_experiment()` to:

1. Before training, compute checkpoint epochs: `ckpt_epochs = {int(frac * config.epochs) for frac in config.hessian_checkpoints if 0 < frac < 1}`
2. Use `train_model()`'s existing `checkpoint_every` mechanism -- but this saves full checkpoints, not the lean Hessian checkpoints.

**Better approach:** After training completes, the full trainer checkpoint at `checkpoint_epoch{N}.pt` contains the model state. For epochs at 25%/50%:
- Set `checkpoint_every = max(1, config.epochs // 4)` in TrainConfig for Hessian seeds
- After training, extract model state from `checkpoint_epoch{25}.pt` and `checkpoint_epoch{50}.pt`
- Copy to `hessian_checkpoints/checkpoint_0.25.pt` and `checkpoint_0.50.pt`

**Simplest approach:** Modify `_optimizer_train_config()` to set `checkpoint_every = max(1, config.epochs // 4)` when the seed is a Hessian seed. Then add post-training extraction of intermediate checkpoints from the trainer's checkpoint files.

**Important:** The trainer saves `save_checkpoint()` format with `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, etc. The Hessian checkpoint format is just `model.state_dict()`. A conversion step is needed:
```python
full_ckpt = torch.load(trainer_ckpt_path, weights_only=False)
torch.save(full_ckpt["model_state_dict"], hessian_ckpt_path)
```

### Gradient Variance Collection

Two options for collecting gradient variance data:

**Option 1: Post-hoc collection** (recommended for consistency)
For each (task, algebra) combination, call `collect_gradient_variance_across_seeds()` with the model factory and data. This retrains small models from scratch for a few steps to measure gradient variance at initialization and during early training.

**Option 2: Extract from trainer logs**
The trainer already logs gradient stats to TensorBoard (`events.out.tfevents.*`). Parse these to extract gradient norms across seeds.

**Recommendation:** Option 1 is simpler and produces the exact format expected by the analysis script. The `collect_gradient_variance_across_seeds()` function already exists and works -- it just needs to be called.

## Test Strategy

### Tests That Would Have Caught This

**Test 1: Integration test -- post-training analysis produces data**
```python
def test_experiment_produces_hessian_data(tmp_path):
    """Verify that running the experiment produces Hessian spectrum data."""
    config = LandscapeConfig(
        tasks=["algebra_native_single"],
        algebras=[AlgebraType.REAL],
        optimizers=["adam"],
        seeds=[0],
        epochs=5,
        base_hidden=4,
        n_train=100, n_test=50,
        output_dir=str(tmp_path),
        device="cpu",
        hessian_seeds=[0],
        hessian_checkpoints=[0.0, 1.0],
    )
    results = run_landscape_experiment(config)
    result = results["algebra_native_single"]["adam"]["R"][0]

    # THIS ASSERTION WOULD HAVE CAUGHT THE GAP
    assert "hessian_spectrum" in result, "Hessian analysis never ran"
    assert result["hessian_spectrum"], "Hessian data is empty"
```

**Test 2: Integration test -- curvature data exists**
```python
def test_experiment_produces_curvature_data(tmp_path):
    """Verify curvature measurement runs on converged models."""
    config = _smoke_config(str(tmp_path))
    results = run_landscape_experiment(config)
    result = results["algebra_native_single"]["adam"]["R"][0]

    assert "curvature" in result, "Curvature measurement never ran"
    assert isinstance(result["curvature"], (int, float)), "Curvature should be numeric"
```

**Test 3: Intermediate checkpoint test**
```python
def test_intermediate_hessian_checkpoints_saved(tmp_path):
    """Verify 0.25 and 0.50 checkpoints are saved for Hessian seeds."""
    config = LandscapeConfig(
        tasks=["algebra_native_single"],
        algebras=[AlgebraType.REAL],
        optimizers=["adam"],
        seeds=[0],
        epochs=20,
        base_hidden=4,
        n_train=100, n_test=50,
        output_dir=str(tmp_path),
        device="cpu",
        hessian_seeds=[0],
        hessian_checkpoints=[0.0, 0.25, 0.5, 1.0],
    )
    run_landscape_experiment(config)

    ckpt_dir = Path(tmp_path) / "algebra_native_single" / "adam" / "R" / "seed_0" / "hessian_checkpoints"
    assert (ckpt_dir / "checkpoint_0.00.pt").exists()
    assert (ckpt_dir / "checkpoint_0.25.pt").exists()
    assert (ckpt_dir / "checkpoint_0.50.pt").exists()
    assert (ckpt_dir / "checkpoint_1.00.pt").exists()
```

**Test 4: End-to-end analysis test**
```python
def test_analyze_landscape_finds_hessian_data(tmp_path):
    """Verify analysis script processes Hessian data when present."""
    # Run experiment + post-analysis
    config = _smoke_config(str(tmp_path))
    run_landscape_experiment(config)
    run_post_analysis(str(tmp_path), config, device="cpu")

    # Load and verify
    result_path = Path(tmp_path) / "algebra_native_single" / "adam" / "R" / "seed_0" / "result.json"
    with open(result_path) as f:
        result = json.load(f)
    assert "hessian_spectrum" in result
    assert "1.0" in result["hessian_spectrum"]
```

**Test 5: Gradient variance test**
```python
def test_gradient_variance_collected(tmp_path):
    """Verify gradient variance is characterized across seeds."""
    # Should use collect_gradient_variance_across_seeds()
    # and produce cross_seed_variance metric
```

### Existing Test Coverage Assessment

| Component | Unit Tests | Integration Tests | Gap |
|-----------|-----------|-------------------|-----|
| `compute_hessian_spectrum()` | `test_landscape_hessian.py` (6 tests) | NONE | Never called from pipeline |
| `measure_curvature()` | `test_landscape_hessian.py` (4 tests) | NONE | Never called from pipeline |
| `collect_gradient_stats()` | NONE | NONE | No tests at all |
| `evaluate_gate()` | `test_landscape_smoke.py::test_smoke_gate_evaluation` | Synthetic input only | Works but never gets real Hessian/curvature data |
| `run_landscape_experiment()` | `test_landscape_smoke.py` (3 tests) | Validates training only | Does not check post-training analysis |
| `analyze_landscape.py` | Plan 05-06 verification only | Fake data only | Never tested with real Hessian/curvature data |

## GPU Utilization

### Current State

From `full_report.json`:
- Average time per run: 31.7 seconds
- Most time spent on R/C with sgd (100 epochs, 20 seeds) and adam (10 epochs, 2 seeds)
- Total 69 result files completed

### Observations

1. **Small models are fast**: base_hidden=4 (smoke) completes in <0.1s per run. base_hidden=16 (full) should take 1-5s per run for 100 epochs. The 31.7s average includes overhead (data generation, model building, checkpoint saving).

2. **Data regeneration overhead**: `_build_task_data()` is called once per task (line 544: outside the optimizer/algebra/seed loops), so this is efficient.

3. **Hessian computation will be expensive**: For a model with ~1000 parameters (base_hidden=16, octonion), full Hessian requires O(n^2) autograd calls. With n=1000, this is ~10^6 operations. Lanczos with 200 iterations and 5 samples is more tractable but still ~1000 HVP operations per checkpoint.

4. **Curvature measurement is moderate**: 50 directions x 51 steps = 2,550 forward passes. For small models this is seconds.

### Recommendations for GPU Efficiency

1. **Batch checkpoint analysis**: Run Hessian analysis on all checkpoints for a given task+algebra before moving to the next. This keeps the model architecture and data in memory.

2. **Use Lanczos for all models**: Even small models (1000 params) can use Lanczos. Full Hessian is only useful for very small models (<200 params) where exact eigenvalues matter. The smoke config (base_hidden=4) produces ~200 params, so auto-dispatch to full is fine. The real config (base_hidden=16) produces ~5000+ params, which needs Lanczos.

3. **Subset data for Hessian**: Use 200-500 data points from the test set for Hessian computation. The Hessian depends on the loss landscape geometry, not the full dataset statistics.

4. **Run post-analysis on GPU**: Both Hessian and curvature benefit from GPU acceleration. The post-analysis script should default to GPU when available.

5. **Parallelize across tasks**: Different tasks use different data and models. Post-analysis for task A can run while training for task B continues.

## Literature Context

### How Published Hessian Studies Structure Their Pipeline

**Ghorbani et al. 2019 ("An Investigation of Why Overparameterization Exacerbates Spurious Correlations"):**
- Train models to convergence first, THEN run stochastic Lanczos
- Use 200 Lanczos iterations with full reorthogonalization (matches our implementation)
- Compute spectral density on a fixed batch of data (not the full training set)
- Report: spectral density plots, negative eigenvalue fraction, trace

**Li et al. 2018 ("Visualizing the Loss Landscape of Neural Nets"):**
- Train model, THEN sample random directions with filter normalization
- Measure 1D loss profiles along normalized directions (matches our `_filter_normalize`)
- Report: 1D loss profile curvature, 2D contour plots

**Sagun et al. 2017 ("Eigenvalues of the Hessian in Deep Learning"):**
- Compute Hessian at multiple training snapshots (matches our 4-checkpoint design)
- Use full eigendecomposition for small networks, power method for large ones
- Report: eigenvalue distribution evolution, bulk vs outlier eigenvalues

**Common pattern across all studies:** Training and analysis are decoupled. Models are trained first with checkpoints saved at key points. Hessian/landscape analysis runs separately, often on different hardware or in a different script. This validates our Option A (post-training analysis script) as the standard approach.

### Bill & Cox 2024 ("Exploring Quaternion Neural Network Loss Surfaces")

- Train quaternion and real models on standard benchmarks
- After convergence, sample random directions with Li et al. filter normalization
- Measure 1D curvature via quadratic fit (exact match to our `measure_curvature()`)
- Key finding: quaternion networks have smoother loss surfaces than real-valued on CIFAR-10/100
- Our extension: same methodology applied to octonionic, PHM-8, and R8-dense-mixing

## Recommendations (Ordered Fix Actions)

### Priority 1: Fix Intermediate Checkpoint Saving

**What:** Modify `run_landscape_experiment()` to save model checkpoints at 0.25 and 0.50 training fractions.

**How:**
1. For Hessian seeds, set `checkpoint_every = max(1, config.epochs // 4)` in the TrainConfig
2. After `train_model()` completes, iterate over trainer checkpoints to find epochs closest to 25% and 50%
3. Extract `model_state_dict` from trainer checkpoint and save as Hessian checkpoint
4. Alternatively, implement a training callback -- but modifying `train_model()` is more invasive

**Scope:** Modify `_experiment.py` only. Existing trained results with only 0.0/1.0 checkpoints can still be analyzed (just fewer checkpoints).

### Priority 2: Build Post-Training Analysis Script

**What:** Create `scripts/run_post_analysis.py` that loads checkpoints and computes Hessian spectrum + curvature.

**How:**
1. Walk `results/landscape/` directory tree
2. For each seed with `hessian_checkpoints/`, reconstruct model and load checkpoint
3. Call `compute_hessian_spectrum()` on each checkpoint
4. Call `measure_curvature()` on converged checkpoints (frac=1.0)
5. Write results back to `result.json` under `hessian_spectrum` and `curvature` keys
6. Format must match what `analyze_landscape.py` expects:
   - `hessian_spectrum`: `{frac_str: eigenvalue_list}` (lines 386-396 of analyze_landscape.py)
   - `curvature`: float (line 521 of analyze_landscape.py)

**Scope:** New script. Also add helper function for `AlgebraType` lookup from short name (e.g., "O" -> AlgebraType.OCTONION).

### Priority 3: Add Gradient Variance Collection

**What:** Compute gradient variance across seeds for each (task, algebra) combination.

**How:**
1. Call `collect_gradient_variance_across_seeds()` with appropriate model factory and data
2. The function already exists and works -- just needs to be called
3. Save results to a `gradient_variance.json` file or append to result.json files
4. Format must match what `analyze_landscape.py` expects (line 607: `gradient_stats.grad_norm_std`)

**Scope:** Add to post-analysis script or create separate gradient analysis step.

### Priority 4: Add Integration Tests

**What:** Tests that verify the full pipeline produces Hessian, curvature, and gradient data.

**How:**
1. `test_post_analysis_produces_hessian_data` -- runs experiment + post-analysis, checks result.json keys
2. `test_intermediate_checkpoints_saved` -- verifies 0.25/0.50 checkpoints exist
3. `test_curvature_data_in_results` -- verifies curvature is a finite float in results
4. `test_gradient_variance_collection` -- verifies gradient stats are collected

**Scope:** New test file `tests/test_landscape_integration.py`.

### Priority 5: Complete Experiment Matrix

**What:** Run the full experiment across all tasks, optimizers, and algebras.

**How:**
1. Use `scripts/run_landscape.py` with full configuration (no --smoke)
2. Resume capability means existing 69 results are preserved
3. Estimated total: 9 tasks x 7 optimizers (5 + 2 riemannian variants) x 6 algebras x 20 seeds = 7,560 runs
4. At ~5s per run: ~10.5 hours
5. Run in stages: one task at a time, verify results, then proceed

**Scope:** Operational, not code changes. But depends on Priorities 1-2 being complete.

### Priority 6: Run Post-Analysis on Completed Results

**What:** After experiment completes, run post-analysis to compute Hessian/curvature/gradient data.

**How:**
1. Run `scripts/run_post_analysis.py --results-dir results/landscape`
2. Run `scripts/analyze_landscape.py --results-dir results/landscape` to generate final plots and gate verdict
3. Review results for go/no-go gate decision

**Scope:** Operational. Depends on Priorities 2-5.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest with hypothesis |
| Config file | pyproject.toml (pytest section) |
| Quick run command | `docker compose run --rm dev uv run pytest tests/test_landscape_hessian.py -x` |
| Full suite command | `docker compose run --rm dev uv run pytest tests/ -x --timeout=300` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FOUND-04a | Hessian spectrum computed from checkpoints | integration | `docker compose run --rm dev uv run pytest tests/test_landscape_integration.py::test_post_analysis_produces_hessian -x` | Wave 0 |
| FOUND-04b | Curvature measured at convergence | integration | `docker compose run --rm dev uv run pytest tests/test_landscape_integration.py::test_curvature_in_results -x` | Wave 0 |
| FOUND-04c | Gradient variance across seeds | integration | `docker compose run --rm dev uv run pytest tests/test_landscape_integration.py::test_gradient_variance -x` | Wave 0 |
| FOUND-04d | Intermediate checkpoints saved | unit | `docker compose run --rm dev uv run pytest tests/test_landscape_integration.py::test_intermediate_checkpoints -x` | Wave 0 |
| FOUND-04e | Post-analysis result format matches analyze_landscape.py expectations | integration | `docker compose run --rm dev uv run pytest tests/test_landscape_integration.py::test_analysis_reads_post_analysis -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `docker compose run --rm dev uv run pytest tests/test_landscape_hessian.py tests/test_landscape_integration.py -x --timeout=300`
- **Per wave merge:** `docker compose run --rm dev uv run pytest tests/ -x --timeout=600`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_landscape_integration.py` -- covers FOUND-04 a/b/c/d/e (new file)
- [ ] `tests/test_gradient_stats.py` -- unit tests for `collect_gradient_stats` and `collect_gradient_variance_across_seeds` (new file, currently untested)

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection: `_experiment.py`, `_hessian.py`, `_curvature.py`, `_gradient_stats.py`, `_gate.py`, `analyze_landscape.py`, `run_landscape.py`, test files
- Existing result files: `results/landscape/` (69 result.json files, 38 Hessian checkpoint .pt files, 0 intermediate checkpoints)
- Plan documents: `05-05-PLAN.md`, `05-06-PLAN.md` (confirm deferred analysis scope)
- `full_report.json` (confirms missing Hessian/curvature data)

### Secondary (MEDIUM confidence)
- Published pipeline patterns from Ghorbani et al. 2019, Li et al. 2018, Sagun et al. 2017, Bill & Cox 2024 (based on training knowledge of these papers)

## Metadata

**Confidence breakdown:**
- Gap identification: HIGH -- based on direct code reading and file system inspection
- Interface inventory: HIGH -- function signatures verified from source code
- Integration design: HIGH -- follows existing patterns in codebase
- Test strategy: HIGH -- specific assertions that would catch each gap
- GPU utilization: MEDIUM -- estimates based on model sizes, not profiling
- Literature context: MEDIUM -- from training knowledge, not verified with current sources

**Research date:** 2026-03-21
**Valid until:** Until gaps are fixed (indefinite -- this is codebase analysis, not library versioning)
