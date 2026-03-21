"""Sinusoidal regression task for optimization landscape experiments.

Task 4: Multi-output regression on sum-of-sines functions.
  y_k = alpha_k * sin(w_k @ x + phi_k)
  where w_k are random frequency vectors and phi_k are phase offsets.

Known global optimum: exact representation of the sum-of-sines function.
"""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset


def build_sinusoidal_regression(
    n_train: int = 50_000,
    n_test: int = 10_000,
    dim: int = 8,
    n_components: int = 3,
    seed: int = 42,
) -> tuple[TensorDataset, TensorDataset]:
    """Build a sinusoidal regression task with known optimal solution.

    Generates multi-output regression data where each output component
    is a sinusoidal function of a linear projection of the input.

    Args:
        n_train: Number of training samples.
        n_test: Number of test samples.
        dim: Input dimension.
        n_components: Number of output sinusoidal components.
        seed: Random seed for deterministic generation.

    Returns:
        (train_dataset, test_dataset) tuple of TensorDatasets.
        x shape: [n, dim], y shape: [n, n_components].
    """
    g = torch.Generator().manual_seed(seed)
    dtype = torch.float64

    # Generate random frequency vectors, phases, and amplitudes
    w = torch.randn(n_components, dim, generator=g, dtype=dtype)  # [n_components, dim]
    phi = torch.rand(n_components, generator=g, dtype=dtype) * 2 * torch.pi  # [n_components]
    alpha = torch.randn(n_components, generator=g, dtype=dtype)  # [n_components]

    # Generate input data
    x_train = torch.randn(n_train, dim, generator=g, dtype=dtype) * 0.5
    x_test = torch.randn(n_test, dim, generator=g, dtype=dtype) * 0.5

    def compute_targets(x: torch.Tensor) -> torch.Tensor:
        # x: [N, dim], w: [n_components, dim]
        # projections: [N, n_components] = x @ w.T
        projections = x @ w.T  # [N, n_components]
        # y_k = alpha_k * sin(w_k @ x + phi_k)
        y = alpha.unsqueeze(0) * torch.sin(projections + phi.unsqueeze(0))
        return y

    y_train = compute_targets(x_train)
    y_test = compute_targets(x_test)

    return (
        TensorDataset(x_train.float(), y_train.float()),
        TensorDataset(x_test.float(), y_test.float()),
    )
