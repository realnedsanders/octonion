#!/usr/bin/env python
"""Comprehensive numerical stability analysis for octonionic operations.

Covers all four FOUND-03 success criteria:
  SC-1: Forward pass error accumulation at depths 10, 50, 100, 500
  SC-2: Condition number characterization vs input magnitude
  SC-3: Float32 vs float64 convergence comparison (integrated into SC-1)
  SC-4: StabilizingNorm mitigation demonstration

Usage:
    docker compose run --rm dev uv run python scripts/analyze_stability.py

Outputs:
    results/stability/depth_sweep.json
    results/stability/condition_numbers.json
    results/stability/mitigation.json
    results/stability/depth_sweep_stripped.png
    results/stability/depth_sweep_full.png
    results/stability/condition_numbers.png
    results/stability/mitigation.png
    Summary table to stdout
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from octonion._multiplication import octonion_mul
from octonion._operations import octonion_exp, octonion_log
from octonion.baselines._algebra_linear import (
    ComplexLinear,
    OctonionDenseLinear,
    QuaternionLinear,
    RealLinear,
)
from octonion.baselines._config import AlgebraType, NetworkConfig
from octonion.baselines._network import AlgebraNetwork
from octonion.baselines._stabilization import StabilizingNorm
from octonion.calculus._numeric import numeric_jacobian

# ── Constants ───────────────────────────────────────────────────────────

DEPTHS = [10, 50, 100, 500]
MAGNITUDES = [0.01, 0.1, 1.0, 10.0, 100.0]
DEPTH_SWEEP_MAGNITUDES = [0.01, 1.0, 100.0]  # 3 regimes for depth sweep
ALGEBRAS = [
    AlgebraType.REAL,
    AlgebraType.COMPLEX,
    AlgebraType.QUATERNION,
    AlgebraType.OCTONION,
]
N_SAMPLES = 500  # per (depth, algebra, magnitude) measurement
HIDDEN = 8  # algebra units (= 8*8=64 real dims for octonion, tractable for Jacobian)
STABILITY_THRESHOLD = 1e-3  # relative error threshold for "stable"
K_VALUES = [5, 10, 20]  # StabilizingNorm normalize_every sweep
SEED = 42
BN_WARMUP_PASSES = 20  # passes to populate BN running statistics

ALGEBRA_LINEAR = {
    AlgebraType.REAL: RealLinear,
    AlgebraType.COMPLEX: ComplexLinear,
    AlgebraType.QUATERNION: QuaternionLinear,
    AlgebraType.OCTONION: OctonionDenseLinear,
}

# Consistent colors per algebra across all plots
ALGEBRA_COLORS = {
    AlgebraType.REAL: "#1f77b4",       # blue
    AlgebraType.COMPLEX: "#2ca02c",    # green
    AlgebraType.QUATERNION: "#ff7f0e", # orange
    AlgebraType.OCTONION: "#d62728",   # red
}

MAGNITUDE_LABELS = {0.01: "small (0.01)", 1.0: "near-unit (1.0)", 100.0: "large (100.0)"}


# ── Helpers ─────────────────────────────────────────────────────────────

def make_input(algebra: AlgebraType, batch: int, hidden: int,
               magnitude: float, dtype: torch.dtype) -> torch.Tensor:
    """Generate random input tensor for an algebra at the target magnitude."""
    if algebra == AlgebraType.REAL:
        x = torch.randn(batch, hidden, dtype=dtype) * magnitude
    else:
        x = torch.randn(batch, hidden, algebra.dim, dtype=dtype) * magnitude
    return x


def make_flat_input(algebra: AlgebraType, batch: int, hidden: int,
                    magnitude: float, dtype: torch.dtype) -> torch.Tensor:
    """Generate flattened real-valued input for AlgebraNetwork."""
    x = torch.randn(batch, hidden * algebra.dim, dtype=dtype) * magnitude
    return x


def build_stripped_chain(algebra: AlgebraType, depth: int, hidden: int,
                         dtype: torch.dtype = torch.float64) -> nn.ModuleList:
    """Build a chain of algebra-specific linear layers without BN or activations."""
    LinearClass = ALGEBRA_LINEAR[algebra]
    layers = nn.ModuleList()
    for _ in range(depth):
        layer = LinearClass(hidden, hidden, bias=False, dtype=dtype)
        layers.append(layer)
    return layers


def forward_stripped_chain(layers: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through stripped chain."""
    h = x
    for layer in layers:
        h = layer(h)
    return h


def forward_with_checkpoints(
    layers: nn.ModuleList,
    x_f64: torch.Tensor,
    x_f32: torch.Tensor,
    checkpoint_depths: list[int],
    stabilizer: StabilizingNorm | None = None,
    normalize_every: int | None = None,
) -> dict[int, dict[str, float]]:
    """Forward pass recording relative error at checkpoint depths.

    Runs both f64 and f32 chains in parallel, computing relative error
    at each checkpoint depth.
    """
    errors = {}
    h64 = x_f64
    h32 = x_f32
    for i, layer in enumerate(layers):
        h64 = layer(h64)
        # Find corresponding f32 layer via separate chain
        depth = i + 1
        if stabilizer is not None and normalize_every is not None and depth % normalize_every == 0:
            h64 = stabilizer(h64)
        if depth in checkpoint_depths:
            # Compute relative error
            rel_err = (h32.double() - h64).norm() / (h64.norm() + 1e-30)
            errors[depth] = rel_err.item()
    return errors


def build_full_network(algebra: AlgebraType, depth: int, hidden: int = HIDDEN,
                       dtype: torch.dtype = torch.float64) -> AlgebraNetwork:
    """Build full AlgebraNetwork for depth sweep measurement."""
    config = NetworkConfig(
        algebra=algebra,
        topology="mlp",
        depth=depth,
        base_hidden=hidden,
        activation="split_relu",
        output_projection="flatten",
        use_batchnorm=True,
        input_dim=hidden * algebra.dim,
        output_dim=hidden * algebra.dim,
    )
    model = AlgebraNetwork(config).to(dtype)
    return model


def warmup_bn(model: nn.Module, algebra: AlgebraType, hidden: int = HIDDEN,
              n_passes: int = BN_WARMUP_PASSES, dtype: torch.dtype = torch.float64) -> None:
    """Run warmup passes in train mode to populate BN running statistics."""
    model.train()
    with torch.no_grad():
        for _ in range(n_passes):
            x = make_flat_input(algebra, batch=32, hidden=hidden, magnitude=1.0, dtype=dtype)
            model(x)
    model.eval()


def compute_condition_number(fn, x: torch.Tensor, eps: float = 1e-7) -> float:
    """Compute condition number of fn at x via numeric Jacobian + SVD."""
    J = numeric_jacobian(fn, x, eps=eps)
    sv = torch.linalg.svdvals(J)
    if sv.numel() == 0 or sv[-1].item() < 1e-30:
        return float("inf")
    return (sv[0] / sv[-1].clamp(min=1e-30)).item()


def find_stable_depth(errors_by_depth: dict[int, float], threshold: float = STABILITY_THRESHOLD) -> int:
    """Find max depth where error stays below threshold."""
    sorted_depths = sorted(errors_by_depth.keys())
    stable = 0
    for d in sorted_depths:
        if errors_by_depth[d] < threshold:
            stable = d
        else:
            break
    return stable


def _sanitize_for_json(obj):
    """Recursively replace NaN/inf with null for valid JSON output."""
    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def save_json(data: dict, path: str) -> None:
    """Save data to JSON, converting NaN/inf to null for valid JSON."""
    sanitized = _sanitize_for_json(data)
    with open(path, "w") as f:
        json.dump(sanitized, f, indent=2, default=float)


# ── Section 1: Depth Sweep / Error Accumulation (SC-1 + SC-3) ──────────

def run_depth_sweep() -> dict:
    """Run depth sweep for both stripped chains and full AlgebraNetworks.

    For each algebra, depth, and input magnitude regime:
    - Build layers at float64, clone weights to float32
    - Forward pass both dtype chains with N_SAMPLES random inputs
    - Compute relative error (float32 vs float64)
    - Track norm drift (output norm / input norm ratio)

    Returns structured results dict.
    """
    results = {"stripped": {}, "full": {}, "stable_depths": {"stripped": {}, "full": {}}}

    # ── Stripped chain experiments ──
    print("  [1a] Stripped chain depth sweep...")
    for algebra in ALGEBRAS:
        alg_name = algebra.short_name
        results["stripped"][alg_name] = {}

        for depth in DEPTHS:
            results["stripped"][alg_name][str(depth)] = {}

            for magnitude in DEPTH_SWEEP_MAGNITUDES:
                torch.manual_seed(SEED)
                mag_label = str(magnitude)

                # Build layers at float64
                layers_f64 = build_stripped_chain(algebra, depth, HIDDEN, dtype=torch.float64)
                layers_f64.eval()

                # Clone to float32 (shared initialization, different precision)
                layers_f32 = copy.deepcopy(layers_f64).float()
                layers_f32.eval()

                errors = []
                norm_ratios = []

                with torch.no_grad():
                    for sample_idx in range(N_SAMPLES):
                        torch.manual_seed(SEED + sample_idx + 1)
                        x64 = make_input(algebra, batch=1, hidden=HIDDEN,
                                         magnitude=magnitude, dtype=torch.float64)
                        x32 = x64.float()

                        out64 = forward_stripped_chain(layers_f64, x64)
                        out32 = forward_stripped_chain(layers_f32, x32)

                        # Relative error
                        out64_norm = out64.norm()
                        if out64_norm.item() > 1e-30:
                            if not torch.isfinite(out32).all():
                                errors.append(float("inf"))
                            else:
                                rel_err = (out32.double() - out64).norm() / out64_norm
                                errors.append(rel_err.item())

                        # Norm drift: output norm / input norm
                        in_norm = x64.norm().item()
                        out_norm = out64.norm().item()
                        if in_norm > 1e-30:
                            if not np.isfinite(out_norm):
                                norm_ratios.append(float("inf"))
                            else:
                                norm_ratios.append(out_norm / in_norm)

                if errors:
                    results["stripped"][alg_name][str(depth)][mag_label] = {
                        "mean_rel_error": float(np.mean(errors)),
                        "std_rel_error": float(np.std(errors)),
                        "mean_norm_ratio": float(np.mean(norm_ratios)) if norm_ratios else 0.0,
                        "std_norm_ratio": float(np.std(norm_ratios)) if norm_ratios else 0.0,
                        "n_samples": len(errors),
                    }
                else:
                    results["stripped"][alg_name][str(depth)][mag_label] = {
                        "mean_rel_error": float("nan"),
                        "std_rel_error": float("nan"),
                        "mean_norm_ratio": float("nan"),
                        "std_norm_ratio": float("nan"),
                        "n_samples": 0,
                    }

                print(f"    {alg_name} depth={depth:3d} mag={magnitude:6.2f}: "
                      f"rel_err={np.mean(errors) if errors else float('nan'):.2e}")

    # ── Compute stripped stable depths (SC-3) ──
    for algebra in ALGEBRAS:
        alg_name = algebra.short_name
        results["stable_depths"]["stripped"][alg_name] = {}
        for magnitude in DEPTH_SWEEP_MAGNITUDES:
            mag_label = str(magnitude)
            depth_errors = {}
            for depth in DEPTHS:
                entry = results["stripped"][alg_name][str(depth)].get(mag_label, {})
                err = entry.get("mean_rel_error", float("nan"))
                if not np.isnan(err):
                    depth_errors[depth] = err
            stable = find_stable_depth(depth_errors)
            results["stable_depths"]["stripped"][alg_name][mag_label] = stable

    # ── Full AlgebraNetwork experiments ──
    print("  [1b] Full AlgebraNetwork depth sweep...")
    for algebra in ALGEBRAS:
        alg_name = algebra.short_name
        results["full"][alg_name] = {}

        for depth in DEPTHS:
            results["full"][alg_name][str(depth)] = {}

            for magnitude in DEPTH_SWEEP_MAGNITUDES:
                torch.manual_seed(SEED)
                mag_label = str(magnitude)

                try:
                    # Build at float64
                    model_f64 = build_full_network(algebra, depth, HIDDEN, dtype=torch.float64)
                    warmup_bn(model_f64, algebra, HIDDEN, dtype=torch.float64)
                    model_f64.eval()

                    # Clone to float32
                    model_f32 = copy.deepcopy(model_f64).float()
                    model_f32.eval()

                    errors = []
                    with torch.no_grad():
                        for sample_idx in range(N_SAMPLES):
                            torch.manual_seed(SEED + sample_idx + 1)
                            x64 = make_flat_input(algebra, batch=1, hidden=HIDDEN,
                                                  magnitude=magnitude, dtype=torch.float64)
                            x32 = x64.float()

                            out64 = model_f64(x64)
                            out32 = model_f32(x32)

                            out64_norm = out64.norm()
                            if out64_norm.item() > 1e-30:
                                if not torch.isfinite(out32).all() or not torch.isfinite(out64).all():
                                    errors.append(float("inf"))
                                else:
                                    rel_err = (out32.double() - out64).norm() / out64_norm
                                    errors.append(rel_err.item())

                    if errors:
                        results["full"][alg_name][str(depth)][mag_label] = {
                            "mean_rel_error": float(np.mean(errors)),
                            "std_rel_error": float(np.std(errors)),
                            "n_samples": len(errors),
                        }
                    else:
                        results["full"][alg_name][str(depth)][mag_label] = {
                            "mean_rel_error": float("nan"),
                            "std_rel_error": float("nan"),
                            "n_samples": 0,
                        }

                    print(f"    {alg_name} depth={depth:3d} mag={magnitude:6.2f}: "
                          f"rel_err={np.mean(errors) if errors else float('nan'):.2e}")

                except Exception as e:
                    print(f"    {alg_name} depth={depth:3d} mag={magnitude:6.2f}: ERROR - {e}")
                    results["full"][alg_name][str(depth)][mag_label] = {
                        "mean_rel_error": float("nan"),
                        "std_rel_error": float("nan"),
                        "n_samples": 0,
                        "error": str(e),
                    }

    # ── Compute full network stable depths (SC-3) ──
    for algebra in ALGEBRAS:
        alg_name = algebra.short_name
        results["stable_depths"]["full"][alg_name] = {}
        for magnitude in DEPTH_SWEEP_MAGNITUDES:
            mag_label = str(magnitude)
            depth_errors = {}
            for depth in DEPTHS:
                entry = results["full"][alg_name].get(str(depth), {}).get(mag_label, {})
                err = entry.get("mean_rel_error", float("nan"))
                if not np.isnan(err):
                    depth_errors[depth] = err
            stable = find_stable_depth(depth_errors)
            results["stable_depths"]["full"][alg_name][mag_label] = stable

    return results


# ── Section 2: Condition Number Characterization (SC-2) ─────────────────

def run_condition_numbers() -> dict:
    """Characterize condition numbers of primitive ops, compositions, and networks.

    Primitives (octonion-only): mul, inverse, exp, log
    Compositions (all 4 algebras): stripped chains at depths 2, 5, 10
    Networks (all 4 algebras): full AlgebraNetwork at depth 3

    Returns structured condition number statistics.
    """
    results = {"primitives": {}, "compositions": {}, "networks": {}}

    # ── Primitive operations (octonion-only) ──
    print("  [2a] Primitive operation condition numbers...")

    def octonion_inverse(x: torch.Tensor) -> torch.Tensor:
        """Compute octonion inverse: conj(x) / ||x||^2."""
        conj = x.clone()
        conj[..., 1:] = -conj[..., 1:]
        norm_sq = (x * x).sum(dim=-1, keepdim=True)
        return conj / norm_sq.clamp(min=1e-30)

    primitives = {
        "mul": lambda x: octonion_mul(torch.randn(8, dtype=torch.float64), x),
        "inv": octonion_inverse,
        "exp": lambda x: octonion_exp(x),
        "log": lambda x: octonion_log(x),
    }

    for op_name, op_fn in primitives.items():
        results["primitives"][op_name] = {}
        for magnitude in MAGNITUDES:
            conds = []
            n_cond_samples = 100
            for i in range(n_cond_samples):
                torch.manual_seed(SEED + i)
                x = torch.randn(8, dtype=torch.float64)
                x = x / x.norm() * magnitude

                # For log, restrict to principal branch: ||Im(x)|| < pi
                if op_name == "log":
                    x_norm = x.norm()
                    if x_norm.item() < 1e-10:
                        continue
                    # Ensure positive real part for principal branch
                    x[0] = x[0].abs() + 0.1

                eps = 1e-7 * max(1.0, magnitude)

                # For mul, fix the reference operand for this magnitude
                if op_name == "mul":
                    torch.manual_seed(SEED + 1000)
                    a_fixed = torch.randn(8, dtype=torch.float64)
                    a_fixed = a_fixed / a_fixed.norm() * magnitude
                    fn = lambda x, a=a_fixed: octonion_mul(a, x)
                else:
                    fn = op_fn

                try:
                    cond = compute_condition_number(fn, x, eps=eps)
                    if np.isfinite(cond):
                        conds.append(cond)
                except Exception:
                    continue

            if conds:
                results["primitives"][op_name][str(magnitude)] = {
                    "mean": float(np.mean(conds)),
                    "std": float(np.std(conds)),
                    "median": float(np.median(conds)),
                    "max": float(np.max(conds)),
                    "n_samples": len(conds),
                }
            else:
                results["primitives"][op_name][str(magnitude)] = {
                    "mean": float("inf"),
                    "std": float("inf"),
                    "median": float("inf"),
                    "max": float("inf"),
                    "n_samples": 0,
                }
            print(f"    {op_name} mag={magnitude:6.2f}: "
                  f"mean_cond={np.mean(conds) if conds else float('nan'):.2e}")

    # ── N-layer compositions (all 4 algebras) ──
    print("  [2b] N-layer composition condition numbers...")
    composition_depths = [2, 5, 10]
    n_comp_samples = 50  # fewer samples for compositions (more expensive)

    for algebra in ALGEBRAS:
        alg_name = algebra.short_name
        results["compositions"][alg_name] = {}

        for comp_depth in composition_depths:
            torch.manual_seed(SEED)
            layers = build_stripped_chain(algebra, comp_depth, HIDDEN, dtype=torch.float64)
            layers.eval()

            # Build forward function for the full chain
            if algebra == AlgebraType.REAL:
                in_dim = HIDDEN
                def chain_fn(x, _layers=layers):
                    h = x
                    for layer in _layers:
                        h = layer(h)
                    return h
            else:
                in_dim = HIDDEN * algebra.dim
                def chain_fn(x, _layers=layers, _alg=algebra):
                    h = x.view(HIDDEN, _alg.dim)
                    for layer in _layers:
                        h = layer(h)
                    return h.view(-1)

            conds = []
            for i in range(n_comp_samples):
                torch.manual_seed(SEED + i)
                x = torch.randn(in_dim, dtype=torch.float64)
                # Normalize to unit magnitude
                x = x / x.norm()

                try:
                    cond = compute_condition_number(chain_fn, x, eps=1e-7)
                    if np.isfinite(cond):
                        conds.append(cond)
                except Exception:
                    continue

            if conds:
                results["compositions"][alg_name][str(comp_depth)] = {
                    "mean": float(np.mean(conds)),
                    "std": float(np.std(conds)),
                    "median": float(np.median(conds)),
                    "max": float(np.max(conds)),
                    "n_samples": len(conds),
                }
            else:
                results["compositions"][alg_name][str(comp_depth)] = {
                    "mean": float("inf"),
                    "std": float("inf"),
                    "median": float("inf"),
                    "max": float("inf"),
                    "n_samples": 0,
                }
            print(f"    {alg_name} chain depth={comp_depth}: "
                  f"mean_cond={np.mean(conds) if conds else float('nan'):.2e}")

    # ── Full network condition numbers (all 4 algebras) ──
    print("  [2c] Full network condition numbers...")
    network_magnitudes = [0.01, 1.0, 100.0]

    for algebra in ALGEBRAS:
        alg_name = algebra.short_name
        results["networks"][alg_name] = {}

        torch.manual_seed(SEED)
        try:
            model = build_full_network(algebra, depth=3, hidden=HIDDEN, dtype=torch.float64)
            warmup_bn(model, algebra, HIDDEN, dtype=torch.float64)
            model.eval()

            in_dim = HIDDEN * algebra.dim

            def net_fn(x, _model=model):
                with torch.no_grad():
                    return _model(x.unsqueeze(0)).squeeze(0)

            for magnitude in network_magnitudes:
                conds = []
                n_net_samples = 30  # network Jacobians are expensive
                for i in range(n_net_samples):
                    torch.manual_seed(SEED + i)
                    x = torch.randn(in_dim, dtype=torch.float64)
                    x = x / x.norm() * magnitude
                    eps = 1e-7 * max(1.0, magnitude)

                    try:
                        cond = compute_condition_number(net_fn, x, eps=eps)
                        if np.isfinite(cond):
                            conds.append(cond)
                    except Exception:
                        continue

                if conds:
                    results["networks"][alg_name][str(magnitude)] = {
                        "mean": float(np.mean(conds)),
                        "std": float(np.std(conds)),
                        "median": float(np.median(conds)),
                        "max": float(np.max(conds)),
                        "n_samples": len(conds),
                    }
                else:
                    results["networks"][alg_name][str(magnitude)] = {
                        "mean": float("inf"),
                        "std": float("inf"),
                        "median": float("inf"),
                        "max": float("inf"),
                        "n_samples": 0,
                    }
                print(f"    {alg_name} network mag={magnitude:6.2f}: "
                      f"mean_cond={np.mean(conds) if conds else float('nan'):.2e}")

        except Exception as e:
            print(f"    {alg_name} network: ERROR - {e}")
            for magnitude in network_magnitudes:
                results["networks"][alg_name][str(magnitude)] = {
                    "mean": float("inf"),
                    "std": float("inf"),
                    "median": float("inf"),
                    "max": float("inf"),
                    "n_samples": 0,
                    "error": str(e),
                }

    return results


# ── Section 4: Mitigation Demonstration (SC-4) ─────────────────────────

def run_mitigation() -> dict:
    """Demonstrate StabilizingNorm extends stable depth.

    For each algebra and K in {5, 10, 20}:
    - Build depth=500 stripped chain at float64, clone to float32
    - Measure relative error at checkpoint depths with and without StabilizingNorm
    - Compute stable depth for baseline and mitigated
    - Report improvement ratio
    """
    results = {}
    checkpoint_depths = [10, 50, 100, 200, 300, 400, 500]

    for algebra in ALGEBRAS:
        alg_name = algebra.short_name
        results[alg_name] = {"baseline": {}, "mitigated": {}}

        torch.manual_seed(SEED)
        max_depth = 500

        # Build layers at float64
        layers_f64 = build_stripped_chain(algebra, max_depth, HIDDEN, dtype=torch.float64)
        layers_f64.eval()

        # Clone to float32
        layers_f32 = copy.deepcopy(layers_f64).float()
        layers_f32.eval()

        stabilizer_f64 = StabilizingNorm(algebra.dim)
        stabilizer_f32 = StabilizingNorm(algebra.dim)

        # ── Baseline (no StabilizingNorm) ──
        baseline_errors = {d: [] for d in checkpoint_depths}
        with torch.no_grad():
            for sample_idx in range(N_SAMPLES):
                torch.manual_seed(SEED + sample_idx + 1)
                x64 = make_input(algebra, batch=1, hidden=HIDDEN,
                                 magnitude=1.0, dtype=torch.float64)
                x32 = x64.float()

                h64 = x64
                h32 = x32
                for i, (layer_f64, layer_f32) in enumerate(zip(layers_f64, layers_f32)):
                    h64 = layer_f64(h64)
                    h32 = layer_f32(h32)
                    depth = i + 1
                    if depth in checkpoint_depths:
                        if not torch.isfinite(h64).all():
                            # Both chains diverged, skip sample
                            continue
                        out64_norm = h64.norm()
                        if out64_norm.item() > 1e-30:
                            if not torch.isfinite(h32).all():
                                baseline_errors[depth].append(float("inf"))
                            else:
                                rel_err = (h32.double() - h64).norm() / out64_norm
                                baseline_errors[depth].append(rel_err.item())

        for d in checkpoint_depths:
            if baseline_errors[d]:
                results[alg_name]["baseline"][str(d)] = {
                    "mean_rel_error": float(np.mean(baseline_errors[d])),
                    "std_rel_error": float(np.std(baseline_errors[d])),
                }
            else:
                results[alg_name]["baseline"][str(d)] = {
                    "mean_rel_error": float("nan"),
                    "std_rel_error": float("nan"),
                }

        # Baseline stable depth
        baseline_depth_errors = {
            d: results[alg_name]["baseline"][str(d)]["mean_rel_error"]
            for d in checkpoint_depths
            if not np.isnan(results[alg_name]["baseline"][str(d)]["mean_rel_error"])
        }
        baseline_stable = find_stable_depth(baseline_depth_errors)
        results[alg_name]["baseline_stable_depth"] = baseline_stable

        print(f"    {alg_name} baseline stable depth: {baseline_stable}")

        # ── Mitigated (with StabilizingNorm) ──
        for K in K_VALUES:
            mitigated_errors = {d: [] for d in checkpoint_depths}
            with torch.no_grad():
                for sample_idx in range(N_SAMPLES):
                    torch.manual_seed(SEED + sample_idx + 1)
                    x64 = make_input(algebra, batch=1, hidden=HIDDEN,
                                     magnitude=1.0, dtype=torch.float64)
                    x32 = x64.float()

                    h64 = x64
                    h32 = x32
                    for i, (layer_f64, layer_f32) in enumerate(zip(layers_f64, layers_f32)):
                        h64 = layer_f64(h64)
                        h32 = layer_f32(h32)
                        depth = i + 1
                        if depth % K == 0:
                            h64 = stabilizer_f64(h64)
                            h32 = stabilizer_f32(h32)
                        if depth in checkpoint_depths:
                            if not torch.isfinite(h64).all():
                                # Both chains diverged, skip sample
                                continue
                            out64_norm = h64.norm()
                            if out64_norm.item() > 1e-30:
                                if not torch.isfinite(h32).all():
                                    mitigated_errors[depth].append(float("inf"))
                                else:
                                    rel_err = (h32.double() - h64).norm() / out64_norm
                                    mitigated_errors[depth].append(rel_err.item())

            k_key = f"K={K}"
            results[alg_name]["mitigated"][k_key] = {}
            for d in checkpoint_depths:
                if mitigated_errors[d]:
                    results[alg_name]["mitigated"][k_key][str(d)] = {
                        "mean_rel_error": float(np.mean(mitigated_errors[d])),
                        "std_rel_error": float(np.std(mitigated_errors[d])),
                    }
                else:
                    results[alg_name]["mitigated"][k_key][str(d)] = {
                        "mean_rel_error": float("nan"),
                        "std_rel_error": float("nan"),
                    }

            # Mitigated stable depth
            mit_depth_errors = {
                d: results[alg_name]["mitigated"][k_key][str(d)]["mean_rel_error"]
                for d in checkpoint_depths
                if not np.isnan(results[alg_name]["mitigated"][k_key][str(d)]["mean_rel_error"])
            }
            mit_stable = find_stable_depth(mit_depth_errors)
            results[alg_name]["mitigated"][k_key]["stable_depth"] = mit_stable

            # Improvement ratio
            if baseline_stable > 0:
                ratio = mit_stable / baseline_stable
            elif mit_stable > 0:
                ratio = float("inf")
            else:
                ratio = 1.0
            results[alg_name]["mitigated"][k_key]["improvement_ratio"] = ratio

            print(f"    {alg_name} K={K}: stable depth={mit_stable}, ratio={ratio:.1f}x")

    return results


# ── Plotting ─────────────────────────────────────────────────────────────

def setup_plotting():
    """Import and configure matplotlib with seaborn styling if available."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_theme()
    except ImportError:
        pass
    return plt


def plot_depth_sweep(depth_results: dict, output_dir: str = "results/stability") -> None:
    """Plot depth sweep results for stripped chains and full networks."""
    plt = setup_plotting()

    for exp_type in ["stripped", "full"]:
        fig, axes = plt.subplots(1, len(DEPTH_SWEEP_MAGNITUDES) + 1, figsize=(20, 5))
        fig.suptitle(f"Depth Sweep: {exp_type.title()} Chain - Relative Error (float32 vs float64)",
                     fontsize=14)

        data = depth_results.get(exp_type, {})
        if not data:
            plt.close(fig)
            continue

        # Per-magnitude subplots
        for mag_idx, magnitude in enumerate(DEPTH_SWEEP_MAGNITUDES):
            ax = axes[mag_idx]
            mag_label = str(magnitude)

            for algebra in ALGEBRAS:
                alg_name = algebra.short_name
                depths_list = []
                errors_list = []
                for d in DEPTHS:
                    entry = data.get(alg_name, {}).get(str(d), {}).get(mag_label, {})
                    err = entry.get("mean_rel_error", float("nan"))
                    if np.isfinite(err) and err > 0:
                        depths_list.append(d)
                        errors_list.append(err)

                if depths_list:
                    ax.semilogy(depths_list, errors_list, "o-",
                                color=ALGEBRA_COLORS[algebra], label=alg_name,
                                markersize=5, linewidth=1.5)

            ax.axhline(y=STABILITY_THRESHOLD, color="gray", linestyle="--",
                       alpha=0.7, label="threshold (1e-3)")
            ax.set_xlabel("Depth (layers)")
            ax.set_ylabel("Relative Error")
            ax.set_title(MAGNITUDE_LABELS.get(magnitude, f"mag={magnitude}"))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Combined subplot (near-unit only)
        ax = axes[-1]
        mag_label = "1.0"
        for algebra in ALGEBRAS:
            alg_name = algebra.short_name
            depths_list = []
            errors_list = []
            for d in DEPTHS:
                entry = data.get(alg_name, {}).get(str(d), {}).get(mag_label, {})
                err = entry.get("mean_rel_error", float("nan"))
                if np.isfinite(err) and err > 0:
                    depths_list.append(d)
                    errors_list.append(err)
            if depths_list:
                ax.semilogy(depths_list, errors_list, "o-",
                            color=ALGEBRA_COLORS[algebra], label=alg_name,
                            markersize=5, linewidth=1.5)

        ax.axhline(y=STABILITY_THRESHOLD, color="gray", linestyle="--",
                   alpha=0.7, label="threshold (1e-3)")
        ax.set_xlabel("Depth (layers)")
        ax.set_ylabel("Relative Error")
        ax.set_title("Combined (near-unit)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, f"depth_sweep_{exp_type}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_condition_numbers(cond_results: dict, output_dir: str = "results/stability") -> None:
    """Plot condition number results."""
    plt = setup_plotting()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Condition Number Characterization", fontsize=14)

    # Left panel: primitive operations vs magnitude
    for op_name in ["mul", "inv", "exp", "log"]:
        op_data = cond_results.get("primitives", {}).get(op_name, {})
        mags = []
        means = []
        for mag in MAGNITUDES:
            entry = op_data.get(str(mag), {})
            mean_cond = entry.get("mean", float("nan"))
            if np.isfinite(mean_cond):
                mags.append(mag)
                means.append(mean_cond)
        if mags:
            ax1.loglog(mags, means, "o-", label=op_name, markersize=5, linewidth=1.5)

    ax1.set_xlabel("Input Magnitude")
    ax1.set_ylabel("Condition Number")
    ax1.set_title("Primitive Operations (Octonion)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right panel: N-layer composition per algebra
    composition_depths = [2, 5, 10]
    bar_width = 0.18
    x_positions = np.arange(len(composition_depths))

    for i, algebra in enumerate(ALGEBRAS):
        alg_name = algebra.short_name
        comp_data = cond_results.get("compositions", {}).get(alg_name, {})
        means = []
        for cd in composition_depths:
            entry = comp_data.get(str(cd), {})
            means.append(entry.get("mean", 0.0))
        ax2.bar(x_positions + i * bar_width, means, bar_width,
                color=ALGEBRA_COLORS[algebra], label=alg_name)

    ax2.set_xlabel("Composition Depth")
    ax2.set_ylabel("Mean Condition Number")
    ax2.set_title("N-Layer Composition")
    ax2.set_xticks(x_positions + bar_width * 1.5)
    ax2.set_xticklabels([str(d) for d in composition_depths])
    ax2.legend()
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "condition_numbers.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_mitigation(mit_results: dict, output_dir: str = "results/stability") -> None:
    """Plot mitigation demonstration results."""
    plt = setup_plotting()

    n_alg = len(ALGEBRAS)
    fig, axes = plt.subplots(1, n_alg, figsize=(5 * n_alg, 5))
    if n_alg == 1:
        axes = [axes]
    fig.suptitle("StabilizingNorm Mitigation: Relative Error vs Depth", fontsize=14)

    checkpoint_depths = [10, 50, 100, 200, 300, 400, 500]

    for ax, algebra in zip(axes, ALGEBRAS):
        alg_name = algebra.short_name
        alg_data = mit_results.get(alg_name, {})

        # Baseline
        baseline = alg_data.get("baseline", {})
        base_depths = []
        base_errors = []
        for d in checkpoint_depths:
            entry = baseline.get(str(d), {})
            err = entry.get("mean_rel_error", float("nan"))
            if np.isfinite(err) and err > 0:
                base_depths.append(d)
                base_errors.append(err)
        if base_depths:
            ax.semilogy(base_depths, base_errors, "k-o", label="baseline",
                        markersize=4, linewidth=2)

        # Mitigated curves
        colors_k = {5: "#e377c2", 10: "#7f7f7f", 20: "#bcbd22"}
        for K in K_VALUES:
            k_key = f"K={K}"
            k_data = alg_data.get("mitigated", {}).get(k_key, {})
            k_depths = []
            k_errors = []
            for d in checkpoint_depths:
                entry = k_data.get(str(d), {})
                err = entry.get("mean_rel_error", float("nan"))
                if np.isfinite(err) and err > 0:
                    k_depths.append(d)
                    k_errors.append(err)
            if k_depths:
                ax.semilogy(k_depths, k_errors, "s--", color=colors_k[K],
                            label=f"K={K}", markersize=4, linewidth=1.5)

        ax.axhline(y=STABILITY_THRESHOLD, color="gray", linestyle=":",
                   alpha=0.7, label="threshold (1e-3)")
        ax.set_xlabel("Depth (layers)")
        ax.set_ylabel("Relative Error")
        ax.set_title(f"{alg_name}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "mitigation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Summary Table ────────────────────────────────────────────────────────

def print_summary(depth_results: dict, cond_results: dict, mit_results: dict) -> None:
    """Print formatted summary table to stdout."""
    print("\n" + "=" * 70)
    print("=== Numerical Stability Analysis Summary ===")
    print("=" * 70)

    # SC-1: Error Accumulation (stripped chain, near-unit magnitude)
    print("\nSC-1: Error Accumulation (stripped chain, near-unit magnitude)")
    print(f"  {'Algebra':<8} | {'Depth 10':>10} | {'Depth 50':>10} | {'Depth 100':>10} | {'Depth 500':>10}")
    print("  " + "-" * 60)
    for algebra in ALGEBRAS:
        alg_name = algebra.short_name
        row = f"  {alg_name:<8} |"
        for d in DEPTHS:
            entry = depth_results.get("stripped", {}).get(alg_name, {}).get(str(d), {}).get("1.0", {})
            err = entry.get("mean_rel_error", float("nan"))
            row += f" {err:>10.2e} |"
        print(row)

    # SC-2: Condition Numbers (primitives at unit magnitude)
    print("\nSC-2: Condition Numbers (primitives at unit magnitude)")
    print(f"  {'Operation':<10} | {'Mean Cond':>12} | {'Max Cond':>12}")
    print("  " + "-" * 42)
    for op_name in ["mul", "inv", "exp", "log"]:
        entry = cond_results.get("primitives", {}).get(op_name, {}).get("1.0", {})
        mean_c = entry.get("mean", float("nan"))
        max_c = entry.get("max", float("nan"))
        print(f"  {op_name:<10} | {mean_c:>12.2e} | {max_c:>12.2e}")

    # SC-3: Stable Depth
    print("\nSC-3: Stable Depth (max depth < 1e-3 relative error, near-unit)")
    print(f"  {'Algebra':<8} | {'Stripped Chain':>15} | {'Full Network':>15}")
    print("  " + "-" * 44)
    for algebra in ALGEBRAS:
        alg_name = algebra.short_name
        stripped_sd = depth_results.get("stable_depths", {}).get("stripped", {}).get(alg_name, {}).get("1.0", 0)
        full_sd = depth_results.get("stable_depths", {}).get("full", {}).get(alg_name, {}).get("1.0", 0)
        stripped_str = f">{stripped_sd}" if stripped_sd >= max(DEPTHS) else str(stripped_sd)
        full_str = f">{full_sd}" if full_sd >= max(DEPTHS) else str(full_sd)
        print(f"  {alg_name:<8} | {stripped_str:>15} | {full_str:>15}")

    # SC-4: Mitigation
    print("\nSC-4: Mitigation (StabilizingNorm, stripped chain, near-unit)")
    header = f"  {'Algebra':<8} | {'Baseline':>10}"
    for K in K_VALUES:
        header += f" | {'K='+str(K):>8}"
    header += f" | {'Best Ratio':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for algebra in ALGEBRAS:
        alg_name = algebra.short_name
        alg_data = mit_results.get(alg_name, {})
        baseline_sd = alg_data.get("baseline_stable_depth", 0)
        row = f"  {alg_name:<8} | {baseline_sd:>10}"
        best_ratio = 0.0
        for K in K_VALUES:
            k_key = f"K={K}"
            k_data = alg_data.get("mitigated", {}).get(k_key, {})
            mit_sd = k_data.get("stable_depth", 0)
            ratio = k_data.get("improvement_ratio", 0.0)
            row += f" | {mit_sd:>8}"
            best_ratio = max(best_ratio, ratio)
        row += f" | {best_ratio:>9.1f}x"
        print(row)

    print("=" * 70)


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    """Run comprehensive numerical stability analysis."""
    os.makedirs("results/stability", exist_ok=True)
    torch.manual_seed(SEED)

    start_time = time.time()
    print("=== Phase 4: Numerical Stability Analysis ===\n")

    print("[1/4] Running depth sweep (SC-1 + SC-3)...")
    depth_results = run_depth_sweep()
    save_json(depth_results, "results/stability/depth_sweep.json")
    plot_depth_sweep(depth_results)

    print("\n[2/4] Running condition number characterization (SC-2)...")
    cond_results = run_condition_numbers()
    save_json(cond_results, "results/stability/condition_numbers.json")
    plot_condition_numbers(cond_results)

    print("\n[3/4] Computing stable depths (SC-3)...")
    # Already extracted in depth_results, just report
    for exp_type in ["stripped", "full"]:
        for algebra in ALGEBRAS:
            alg_name = algebra.short_name
            sd = depth_results.get("stable_depths", {}).get(exp_type, {}).get(alg_name, {}).get("1.0", 0)
            label = ">500" if sd >= max(DEPTHS) else str(sd)
            print(f"  {exp_type:>8} {alg_name}: stable depth = {label}")

    print("\n[4/4] Running mitigation demonstration (SC-4)...")
    mit_results = run_mitigation()
    save_json(mit_results, "results/stability/mitigation.json")
    plot_mitigation(mit_results)

    print_summary(depth_results, cond_results, mit_results)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results saved to: results/stability/")


if __name__ == "__main__":
    main()
