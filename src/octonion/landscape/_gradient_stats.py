"""Gradient variance collection for optimization landscape characterization.

Provides:
- collect_gradient_stats: Single-point gradient statistics snapshot
- collect_gradient_variance_across_seeds: Per-seed gradient variance over training
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn


def collect_gradient_stats(
    model: nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    data_x: torch.Tensor,
    data_y: torch.Tensor,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Collect gradient statistics at a single point.

    Performs forward pass, computes loss, backward pass, then collects
    gradient norms both overall and per-layer.

    Args:
        model: Neural network model.
        loss_fn: Loss function callable.
        data_x: Input data tensor.
        data_y: Target data tensor.
        device: Device to run on.

    Returns:
        Dict with keys:
        - grad_norm_mean: Mean gradient norm across all parameters
        - grad_norm_std: Std of gradient norms across parameters
        - grad_norm_max: Maximum gradient norm
        - grad_norm_min: Minimum gradient norm
        - per_layer_stats: List of dicts with per-layer gradient info
    """
    model = model.to(device)
    model.train()

    data_x = data_x.to(device)
    data_y = data_y.to(device)

    # Zero gradients
    model.zero_grad()

    # Forward + backward
    outputs = model(data_x)
    loss = loss_fn(outputs, data_y)
    loss.backward()

    # Collect per-layer stats
    per_layer_stats: list[dict[str, Any]] = []
    all_norms: list[float] = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            norm_val = float(grad.norm().item())
            mean_val = float(grad.mean().item())
            std_val = float(grad.std().item()) if grad.numel() > 1 else 0.0
            max_val = float(grad.max().item())
            min_val = float(grad.min().item())

            per_layer_stats.append({
                "name": name,
                "norm": norm_val,
                "mean": mean_val,
                "std": std_val,
                "max": max_val,
                "min": min_val,
            })
            all_norms.append(norm_val)

    if all_norms:
        norms_tensor = torch.tensor(all_norms)
        grad_norm_mean = float(norms_tensor.mean().item())
        grad_norm_std = float(norms_tensor.std().item()) if len(all_norms) > 1 else 0.0
        grad_norm_max = float(norms_tensor.max().item())
        grad_norm_min = float(norms_tensor.min().item())
    else:
        grad_norm_mean = 0.0
        grad_norm_std = 0.0
        grad_norm_max = 0.0
        grad_norm_min = 0.0

    return {
        "grad_norm_mean": grad_norm_mean,
        "grad_norm_std": grad_norm_std,
        "grad_norm_max": grad_norm_max,
        "grad_norm_min": grad_norm_min,
        "per_layer_stats": per_layer_stats,
    }


def collect_gradient_variance_across_seeds(
    model_factory: Callable[[], nn.Module],
    loss_fn: Callable[..., torch.Tensor],
    data_x: torch.Tensor,
    data_y: torch.Tensor,
    seeds: list[int],
    n_steps: int = 10,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Collect gradient variance across multiple random seeds.

    For each seed: initialize model, run n_steps of training, collect
    gradient statistics at each step. Then aggregate across seeds.

    Args:
        model_factory: Callable that returns a fresh nn.Module.
        loss_fn: Loss function callable.
        data_x: Input data tensor.
        data_y: Target data tensor.
        seeds: List of random seeds to use.
        n_steps: Number of training steps per seed.
        lr: Learning rate for optimizer.
        device: Device to run on.

    Returns:
        Dict with keys:
        - per_seed_stats: List of per-seed gradient stat histories
        - cross_seed_variance: Variance of mean gradient norms across seeds
        - mean_grad_norm_trajectory: List of mean norms at each step (averaged across seeds)
    """
    data_x = data_x.to(device)
    data_y = data_y.to(device)

    per_seed_stats: list[list[dict[str, Any]]] = []
    # per_seed_mean_norms[seed_idx][step] = mean_grad_norm at that step
    per_seed_mean_norms: list[list[float]] = []

    for seed in seeds:
        torch.manual_seed(seed)
        model = model_factory().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        seed_history: list[dict[str, Any]] = []
        seed_norms: list[float] = []

        for _step in range(n_steps):
            model.train()
            optimizer.zero_grad()
            outputs = model(data_x)
            loss = loss_fn(outputs, data_y)
            loss.backward()

            # Collect stats before stepping
            stats = collect_gradient_stats(model, loss_fn, data_x, data_y, device)
            seed_history.append(stats)
            seed_norms.append(stats["grad_norm_mean"])

            optimizer.step()

        per_seed_stats.append(seed_history)
        per_seed_mean_norms.append(seed_norms)

    # Aggregate across seeds
    # mean_grad_norm_trajectory: at each step, average the mean norm across seeds
    mean_grad_norm_trajectory: list[float] = []
    for step in range(n_steps):
        step_norms = [per_seed_mean_norms[s][step] for s in range(len(seeds))]
        mean_grad_norm_trajectory.append(
            sum(step_norms) / len(step_norms) if step_norms else 0.0
        )

    # Cross-seed variance: variance of the per-seed overall mean gradient norm
    per_seed_overall_means = [
        sum(norms) / len(norms) if norms else 0.0
        for norms in per_seed_mean_norms
    ]
    if len(per_seed_overall_means) > 1:
        overall_mean = sum(per_seed_overall_means) / len(per_seed_overall_means)
        cross_seed_variance = sum(
            (m - overall_mean) ** 2 for m in per_seed_overall_means
        ) / (len(per_seed_overall_means) - 1)
    else:
        cross_seed_variance = 0.0

    return {
        "per_seed_stats": per_seed_stats,
        "cross_seed_variance": cross_seed_variance,
        "mean_grad_norm_trajectory": mean_grad_norm_trajectory,
    }
