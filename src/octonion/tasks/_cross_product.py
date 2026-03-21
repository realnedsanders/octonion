"""Cross product recovery task for optimization landscape experiments.

Task 3: Recover y = cross(x, v) from noisy observations.
  - 7D cross product uses octonion structure constants: Im(Im(a)*Im(b)).
  - 3D cross product uses standard torch.linalg.cross (positive control).
  - Noise added to training targets only; test targets are always clean.
  - Supports dim=64 variant (embed signal in first cross_dim dimensions).
"""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from octonion._multiplication import STRUCTURE_CONSTANTS


def _seven_dim_cross_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute the 7D cross product using octonion structure constants.

    The 7D cross product is defined as Im(Im(a) * Im(b)) where Im embeds
    7D vectors as pure imaginary octonions (prepend 0 real component).

    Args:
        a: Tensor of shape [..., 7].
        b: Tensor of shape [..., 7].

    Returns:
        Tensor of shape [..., 7] representing the 7D cross product.
    """
    # Embed as pure imaginary octonions: prepend zero real part
    zero_real = torch.zeros_like(a[..., :1])
    a_oct = torch.cat([zero_real, a], dim=-1)  # [..., 8]
    b_oct = torch.cat([zero_real, b], dim=-1)  # [..., 8]

    C = STRUCTURE_CONSTANTS.to(device=a.device, dtype=a.dtype)
    product = torch.einsum("...i, ijk, ...j -> ...k", a_oct, C, b_oct)

    # Extract imaginary part (indices 1:8)
    return product[..., 1:]


def build_cross_product_recovery(
    n_train: int = 50_000,
    n_test: int = 10_000,
    cross_dim: int = 7,
    noise_level: float = 0.0,
    seed: int = 42,
    dim: int | None = None,
) -> tuple[TensorDataset, TensorDataset]:
    """Build a cross product recovery task.

    Generates data where y = cross(x, v) for a fixed unit vector v,
    optionally with additive Gaussian noise on training targets.

    Args:
        n_train: Number of training samples.
        n_test: Number of test samples.
        cross_dim: Dimension of cross product (3 or 7).
        noise_level: Noise standard deviation as fraction of signal std.
        seed: Random seed for deterministic generation.
        dim: If set (e.g., 64), embed cross product in higher-dimensional space.

    Returns:
        (train_dataset, test_dataset) tuple of TensorDatasets.
    """
    if cross_dim not in (3, 7):
        raise ValueError(f"cross_dim must be 3 or 7, got {cross_dim}")

    g = torch.Generator().manual_seed(seed)
    dtype = torch.float64

    # Fixed unit vector v
    v = torch.randn(cross_dim, generator=g, dtype=dtype)
    v = v / v.norm()

    # Generate input data
    x_train_raw = torch.randn(n_train, cross_dim, generator=g, dtype=dtype)
    x_test_raw = torch.randn(n_test, cross_dim, generator=g, dtype=dtype)

    # Compute clean cross products
    v_expanded_train = v.unsqueeze(0).expand(n_train, -1)
    v_expanded_test = v.unsqueeze(0).expand(n_test, -1)

    if cross_dim == 3:
        y_train_clean = torch.linalg.cross(x_train_raw, v_expanded_train)
        y_test_clean = torch.linalg.cross(x_test_raw, v_expanded_test)
    else:  # cross_dim == 7
        y_train_clean = _seven_dim_cross_product(x_train_raw, v_expanded_train)
        y_test_clean = _seven_dim_cross_product(x_test_raw, v_expanded_test)

    # Add noise to training targets only
    if noise_level > 0.0:
        noise_std = noise_level * y_train_clean.std()
        noise = torch.randn(
            y_train_clean.shape, generator=g, dtype=y_train_clean.dtype
        ) * noise_std
        y_train = y_train_clean + noise
    else:
        y_train = y_train_clean

    # Test targets are always clean
    y_test = y_test_clean

    # Handle dim=64 embedding
    if dim is not None and dim > cross_dim:
        # Embed x in higher-dimensional space (first cross_dim dims are signal)
        x_train_full = torch.randn(n_train, dim, generator=g, dtype=dtype)
        x_train_full[:, :cross_dim] = x_train_raw
        x_test_full = torch.randn(n_test, dim, generator=g, dtype=dtype)
        x_test_full[:, :cross_dim] = x_test_raw

        # Embed y similarly (pad with zeros)
        y_train_full = torch.zeros(n_train, dim, dtype=dtype)
        y_train_full[:, :cross_dim] = y_train
        y_test_full = torch.zeros(n_test, dim, dtype=dtype)
        y_test_full[:, :cross_dim] = y_test

        return (
            TensorDataset(x_train_full.float(), y_train_full.float()),
            TensorDataset(x_test_full.float(), y_test_full.float()),
        )

    return (
        TensorDataset(x_train_raw.float(), y_train.float()),
        TensorDataset(x_test_raw.float(), y_test.float()),
    )
