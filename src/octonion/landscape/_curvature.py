"""Bill & Cox loss surface curvature measurement.

Implements the curvature measurement methodology from Bill & Cox (2024)
for characterizing loss surface geometry. Extends their quaternion results
to octonionic models.

Key concepts:
- Random directions with Li et al. (2018) filter normalization
- 1D loss profile sampling along each direction
- Quadratic fit to estimate curvature (2 * leading coefficient)

This measures how "sharp" or "flat" the loss surface is around the
converged model parameters, providing insights into generalization
properties across different algebraic structures.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn


def _filter_normalize(
    direction: dict[str, torch.Tensor], model: nn.Module
) -> None:
    """Apply Li et al. 2018 filter normalization in-place.

    For each parameter with dim >= 2, scale each filter (row) of the direction
    to match the corresponding filter's norm in the model parameters.
    For 1D params (bias), normalize to match parameter norm.

    Args:
        direction: Dict mapping param names to random direction tensors.
        model: Model whose parameter norms define the target scaling.
    """
    for name, param in model.named_parameters():
        if name not in direction:
            continue
        d = direction[name]
        if param.dim() >= 2:
            # Per-filter normalization: scale each row independently
            for j in range(d.shape[0]):
                param_norm = param[j].norm().item()
                d_norm = d[j].norm().item()
                if d_norm > 1e-10:
                    d[j] *= param_norm / d_norm
        else:
            # Scalar/bias: match overall norm
            param_norm = param.norm().item()
            d_norm = d.norm().item()
            if d_norm > 1e-10:
                d *= param_norm / d_norm


def _evaluate_loss(
    model: nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    data_x: torch.Tensor,
    data_y: torch.Tensor,
) -> float:
    """Evaluate model loss on data batch (no grad).

    Args:
        model: PyTorch model.
        loss_fn: Loss function(output, target) -> scalar.
        data_x: Input data tensor.
        data_y: Target data tensor.

    Returns:
        Scalar loss value.
    """
    model.eval()
    with torch.no_grad():
        output = model(data_x)
        return loss_fn(output, data_y).item()


def measure_curvature(
    model: nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    data_x: torch.Tensor,
    data_y: torch.Tensor,
    n_directions: int = 50,
    n_steps: int = 51,
    step_range: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Measure loss surface curvature via 1D profile sampling.

    Implements the Bill & Cox (2024) curvature measurement methodology:
    1. For each random direction, apply Li et al. (2018) filter normalization
    2. Sample loss along the direction at evenly-spaced step sizes
    3. Fit a quadratic to the 1D profile; curvature = 2 * leading coefficient

    The model's original weights are saved and restored after measurement.

    Args:
        model: PyTorch model (at converged parameters).
        loss_fn: Loss function(output, target) -> scalar.
        data_x: Input data tensor.
        data_y: Target data tensor.
        n_directions: Number of random directions to sample.
        n_steps: Number of step sizes per direction.
        step_range: Maximum step size (symmetric around 0).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
          - mean_curvature: Mean curvature across directions
          - median_curvature: Median curvature across directions
          - std_curvature: Standard deviation of curvatures
          - curvatures: List of curvature values (one per direction)
          - n_directions: Number of directions sampled
    """
    # Save converged weights
    theta_star: dict[str, torch.Tensor] = {
        name: p.data.clone() for name, p in model.named_parameters()
    }

    alphas = np.linspace(-step_range, step_range, n_steps)
    curvatures: list[float] = []

    for i in range(n_directions):
        # Generate random direction (seeded for reproducibility)
        gen = torch.Generator()
        gen.manual_seed(seed + i)
        direction: dict[str, torch.Tensor] = {
            name: torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype)
            for name, p in model.named_parameters()
        }

        # Apply filter normalization
        _filter_normalize(direction, model)

        # Sample loss at each step
        losses: list[float] = []
        for alpha in alphas:
            # Set params to theta_star + alpha * direction
            for name, p in model.named_parameters():
                p.data.copy_(theta_star[name] + alpha * direction[name])
            loss_val = _evaluate_loss(model, loss_fn, data_x, data_y)
            losses.append(loss_val)

        # Fit quadratic: loss(alpha) ≈ a * alpha^2 + b * alpha + c
        coeffs = np.polyfit(alphas, losses, 2)
        # Curvature = second derivative of quadratic = 2 * a
        curvature = 2.0 * coeffs[0]
        curvatures.append(float(curvature))

    # Restore original weights
    for name, p in model.named_parameters():
        p.data.copy_(theta_star[name])

    curvatures_arr = np.array(curvatures)
    return {
        "mean_curvature": float(np.mean(curvatures_arr)),
        "median_curvature": float(np.median(curvatures_arr)),
        "std_curvature": float(np.std(curvatures_arr)),
        "curvatures": curvatures,
        "n_directions": n_directions,
    }
