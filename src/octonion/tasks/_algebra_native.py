"""Algebra-native synthetic tasks for optimization landscape experiments.

Task 1 (single-layer): y = a * x * b where a, b are fixed unit-norm parameters.
  - Known optimal loss = 0 (single linear layer can exactly represent this).
  - Supports dim=1 (real), 2 (complex), 4 (quaternion), 8 (octonion), 64 (blocked).

Task 2 (multi-layer): y = f_d(f_{d-1}(...f_1(x)...)) where f_i(x) = a_i * x * b_i.
  - Tests whether non-associativity compounds optimization difficulty at depth.
"""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from octonion._multiplication import octonion_mul


def _quaternion_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product for [..., 4] tensors."""
    p0, p1, p2, p3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    q0, q1, q2, q3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack(
        [
            p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3,
            p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2,
            p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1,
            p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0,
        ],
        dim=-1,
    )


def _complex_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Complex multiplication for [..., 2] tensors."""
    ar, ai = a[..., 0], a[..., 1]
    br, bi = b[..., 0], b[..., 1]
    return torch.stack([ar * br - ai * bi, ar * bi + ai * br], dim=-1)


def _algebra_mul(a: torch.Tensor, b: torch.Tensor, dim: int) -> torch.Tensor:
    """Dispatch to the correct algebra multiplication based on dimension."""
    if dim == 1:
        return a * b
    elif dim == 2:
        return _complex_mul(a, b)
    elif dim == 4:
        return _quaternion_mul(a, b)
    elif dim == 8:
        return octonion_mul(a, b)
    else:
        raise ValueError(f"Unsupported algebra dimension: {dim}")


def _algebra_native_transform(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Compute y = a * x * b for the appropriate algebra dimension.

    For dim=64: applies 8 blocks of 8D octonion multiplication independently,
    then mixes via a random rotation (the rotation is not passed here; it's
    applied externally).
    """
    a_exp = a.expand_as(x)
    b_exp = b.expand_as(x)
    return _algebra_mul(_algebra_mul(a_exp, x, dim), b_exp, dim)


def _make_unit_vector(g: torch.Generator, dim: int, dtype: torch.dtype) -> torch.Tensor:
    """Generate a unit-norm random vector of given dimension."""
    v = torch.randn(dim, generator=g, dtype=dtype)
    return v / v.norm()


def build_algebra_native_single(
    n_train: int = 50_000,
    n_test: int = 10_000,
    dim: int = 8,
    seed: int = 42,
) -> tuple[TensorDataset, TensorDataset]:
    """Build a single-layer algebra-native task: y = a * x * b.

    Known optimal loss = 0. The transformation is a single bilinear map
    in the algebra, exactly representable by one OctonionLinear layer.

    Args:
        n_train: Number of training samples.
        n_test: Number of test samples.
        dim: Algebra dimension (1, 2, 4, 8, or 64).
        seed: Random seed for deterministic generation.

    Returns:
        (train_dataset, test_dataset) tuple of TensorDatasets.
        Each dataset contains (x, y) tensors.
    """
    g = torch.Generator().manual_seed(seed)
    dtype = torch.float64

    if dim == 64:
        # 8 blocks of 8D octonion multiplication + random rotation
        a_blocks = torch.stack([_make_unit_vector(g, 8, dtype) for _ in range(8)])  # [8, 8]
        b_blocks = torch.stack([_make_unit_vector(g, 8, dtype) for _ in range(8)])  # [8, 8]
        # Random orthogonal mixing matrix
        rand_mat = torch.randn(64, 64, generator=g, dtype=dtype)
        Q, _ = torch.linalg.qr(rand_mat)

        def transform_64(x: torch.Tensor) -> torch.Tensor:
            # Split x into 8 blocks of 8
            blocks = x.reshape(-1, 8, 8)  # [N, 8, 8]
            out_blocks = []
            for i in range(8):
                block_x = blocks[:, i, :]  # [N, 8]
                y_block = _algebra_native_transform(block_x, a_blocks[i], b_blocks[i], 8)
                out_blocks.append(y_block)
            y_blocked = torch.stack(out_blocks, dim=1).reshape(-1, 64)  # [N, 64]
            return y_blocked @ Q.T

        x_train = torch.randn(n_train, 64, generator=g, dtype=dtype) * 0.5
        x_test = torch.randn(n_test, 64, generator=g, dtype=dtype) * 0.5
        y_train = transform_64(x_train)
        y_test = transform_64(x_test)
    else:
        a = _make_unit_vector(g, dim, dtype)
        b = _make_unit_vector(g, dim, dtype)
        x_train = torch.randn(n_train, dim, generator=g, dtype=dtype) * 0.5
        x_test = torch.randn(n_test, dim, generator=g, dtype=dtype) * 0.5
        y_train = _algebra_native_transform(x_train, a, b, dim)
        y_test = _algebra_native_transform(x_test, a, b, dim)

    return (
        TensorDataset(x_train.float(), y_train.float()),
        TensorDataset(x_test.float(), y_test.float()),
    )


def build_algebra_native_multi(
    n_train: int = 50_000,
    n_test: int = 10_000,
    dim: int = 8,
    depth: int = 3,
    seed: int = 42,
) -> tuple[TensorDataset, TensorDataset]:
    """Build a multi-layer algebra-native task: y = f_d(f_{d-1}(...f_1(x)...)).

    Each f_i(x) = a_i * x * b_i with unit-norm a_i, b_i.
    Tests whether non-associativity compounds optimization difficulty.

    Args:
        n_train: Number of training samples.
        n_test: Number of test samples.
        dim: Algebra dimension (1, 2, 4, 8, or 64).
        depth: Number of chained transformations.
        seed: Random seed for deterministic generation.

    Returns:
        (train_dataset, test_dataset) tuple of TensorDatasets.
    """
    g = torch.Generator().manual_seed(seed)
    dtype = torch.float64

    effective_dim = dim if dim != 64 else 8

    # Generate depth pairs of unit-norm vectors
    params = []
    for _ in range(depth):
        if dim == 64:
            a_blocks = torch.stack([_make_unit_vector(g, 8, dtype) for _ in range(8)])
            b_blocks = torch.stack([_make_unit_vector(g, 8, dtype) for _ in range(8)])
            params.append((a_blocks, b_blocks))
        else:
            a = _make_unit_vector(g, effective_dim, dtype)
            b = _make_unit_vector(g, effective_dim, dtype)
            params.append((a, b))

    def chain_transform(x: torch.Tensor) -> torch.Tensor:
        y = x
        for a_param, b_param in params:
            if dim == 64:
                blocks = y.reshape(-1, 8, 8)
                out_blocks = []
                for i in range(8):
                    block_x = blocks[:, i, :]
                    y_block = _algebra_native_transform(
                        block_x, a_param[i], b_param[i], 8
                    )
                    out_blocks.append(y_block)
                y = torch.stack(out_blocks, dim=1).reshape(-1, 64)
            else:
                y = _algebra_native_transform(y, a_param, b_param, effective_dim)
        return y

    x_train = torch.randn(n_train, dim, generator=g, dtype=dtype) * 0.5
    x_test = torch.randn(n_test, dim, generator=g, dtype=dtype) * 0.5
    y_train = chain_transform(x_train)
    y_test = chain_transform(x_test)

    return (
        TensorDataset(x_train.float(), y_train.float()),
        TensorDataset(x_test.float(), y_test.float()),
    )
