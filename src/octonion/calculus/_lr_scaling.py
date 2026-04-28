"""Learning rate scaling heuristic based on GHR gradient magnitude statistics.

Octonionic layers produce gradients that may differ in magnitude from standard
real-valued (R^8) linear layers. This module provides tools to measure these
differences and recommend learning rate scaling factors.

The key insight: if octonionic gradient norms are on average K times larger than
equivalent real-valued gradient norms, then the learning rate should be scaled
by 1/K to maintain comparable update magnitudes. This prevents training
instabilities when mixing octonionic and real-valued layers.

Usage::

    from octonion.calculus import gradient_magnitude_stats, lr_scaling_heuristic, suggest_lr
    from octonion import OctonionLinear

    layer = OctonionLinear()
    stats = gradient_magnitude_stats(layer, n_samples=500)
    factor = lr_scaling_heuristic(stats)
    adjusted_lr = suggest_lr(0.01, layer)

Convention: Baez 2002, mod-7 Fano plane basis.
"""

from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn as nn


class GradientStats(TypedDict):
    """Per-key types for gradient_magnitude_stats output."""

    grad_norm_mean: float
    grad_norm_std: float
    grad_norm_max: float
    grad_norm_min: float
    grad_per_component: list[float]
    ratio_to_real: float


def gradient_magnitude_stats(
    layer: nn.Module,
    n_samples: int = 1000,
    input_range: tuple[float, float] = (-1.0, 1.0),
    dtype: torch.dtype = torch.float64,
) -> GradientStats:
    """Compute gradient magnitude statistics for an octonionic layer.

    Generates n_samples random inputs, runs forward + backward through the layer,
    and collects gradient statistics. Also computes gradients through a reference
    R^8 linear layer for comparison.

    Args:
        layer: An nn.Module (typically OctonionLinear) to analyze.
        n_samples: Number of random inputs to sample.
        input_range: (low, high) range for uniform random inputs.
        dtype: Tensor dtype for computations.

    Returns:
        Dictionary with keys:
        - grad_norm_mean: Mean of ||grad|| across samples
        - grad_norm_std: Std of ||grad|| across samples
        - grad_norm_max: Max ||grad|| across samples
        - grad_norm_min: Min ||grad|| across samples
        - grad_per_component: Mean |grad_i| for each of 8 components
        - ratio_to_real: ||O_grad|| / ||R8_grad|| comparing octonionic to R^8 gradients
    """
    low, high = input_range

    # Collect octonionic gradients
    oct_grad_norms: list[float] = []
    oct_grad_components = torch.zeros(8, dtype=dtype)

    for _ in range(n_samples):
        x = (high - low) * torch.rand(8, dtype=dtype) + low
        x.requires_grad_(True)

        out = layer(x)
        loss = out.sum()
        loss.backward()

        if x.grad is not None:
            g = x.grad.detach()
            oct_grad_norms.append(torch.linalg.norm(g).item())
            oct_grad_components += g.abs()
            x.grad = None

        layer.zero_grad()

    oct_norms = torch.tensor(oct_grad_norms, dtype=dtype)
    oct_grad_components /= n_samples

    # Collect R^8 reference gradients (simple linear layer W @ x)
    ref_layer = nn.Linear(8, 8, bias=False, dtype=dtype)
    # Initialize with similar scale to octonionic layer
    nn.init.orthogonal_(ref_layer.weight)
    ref_grad_norms: list[float] = []

    for _ in range(n_samples):
        x = (high - low) * torch.rand(8, dtype=dtype) + low
        x.requires_grad_(True)

        out = ref_layer(x)
        loss = out.sum()
        loss.backward()

        if x.grad is not None:
            ref_grad_norms.append(torch.linalg.norm(x.grad.detach()).item())
            x.grad = None

        ref_layer.zero_grad()

    ref_norms = torch.tensor(ref_grad_norms, dtype=dtype)

    # Compute ratio
    oct_mean = oct_norms.mean().item()
    ref_mean = ref_norms.mean().item()
    ratio = oct_mean / ref_mean if ref_mean > 0 else 1.0

    return {
        "grad_norm_mean": oct_mean,
        "grad_norm_std": oct_norms.std().item(),
        "grad_norm_max": oct_norms.max().item(),
        "grad_norm_min": oct_norms.min().item(),
        "grad_per_component": oct_grad_components.tolist(),
        "ratio_to_real": ratio,
    }


def lr_scaling_heuristic(stats: GradientStats) -> float:
    """Recommend a learning rate scaling factor from gradient statistics.

    The heuristic: if octonionic gradient norms are K times larger than
    real-valued gradient norms on average, scale the learning rate by 1/K
    to maintain comparable update magnitudes.

    Args:
        stats: Dictionary from gradient_magnitude_stats() containing at
            minimum the key "ratio_to_real".

    Returns:
        Positive float scaling factor. Multiply base learning rate by this
        value to get the recommended octonionic learning rate.
    """
    ratio = stats["ratio_to_real"]
    if ratio <= 0:
        return 1.0
    return 1.0 / ratio


def suggest_lr(
    base_lr: float,
    layer: nn.Module,
    n_samples: int = 500,
    input_range: tuple[float, float] = (-1.0, 1.0),
    dtype: torch.dtype = torch.float64,
) -> float:
    """Compute a suggested learning rate for an octonionic layer.

    Convenience function: computes gradient statistics, applies the scaling
    heuristic, and returns the adjusted learning rate.

    Args:
        base_lr: Base learning rate (e.g., for real-valued layers).
        layer: Octonionic layer to analyze.
        n_samples: Number of random inputs to sample.
        input_range: (low, high) range for uniform random inputs.
        dtype: Tensor dtype for computations.

    Returns:
        Adjusted learning rate (base_lr * scaling_factor).
    """
    stats = gradient_magnitude_stats(
        layer, n_samples=n_samples, input_range=input_range, dtype=dtype
    )
    factor = lr_scaling_heuristic(stats)
    return base_lr * factor
