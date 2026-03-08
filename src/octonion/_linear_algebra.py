"""Left and right multiplication matrices for octonions.

Given an octonion a, the left multiplication matrix L_a is the 8x8 real matrix
such that L_a @ x = a * x for all octonions x. Similarly, R_b @ x = x * b.

These matrices are constructed from the structure constants tensor.

Convention: Baez 2002, mod-7 Fano plane.
"""

from __future__ import annotations

import torch

from octonion._multiplication import STRUCTURE_CONSTANTS
from octonion._octonion import Octonion


def left_mul_matrix(a: Octonion) -> torch.Tensor:
    """Return the 8x8 real matrix L_a such that a*x = L_a @ x for all x.

    The matrix is constructed via:
      L_a[k, j] = sum_i a[i] * C[i, j, k]

    where C is the structure constants tensor: (e_i * e_j)_k = C[i, j, k].

    Args:
        a: Octonion instance with shape [..., 8].

    Returns:
        Tensor of shape [..., 8, 8] where result[..., k, j] = sum_i a_i * C[i, j, k].
    """
    C = STRUCTURE_CONSTANTS.to(device=a.components.device, dtype=a.components.dtype)
    # L[k, j] = sum_i a[i] * C[i, j, k]
    # So (L @ x)[k] = sum_j L[k,j] * x[j] = sum_{i,j} a[i] * C[i,j,k] * x[j] = (a*x)[k]
    return torch.einsum("...i, ijk -> ...kj", a.components, C)


def right_mul_matrix(b: Octonion) -> torch.Tensor:
    """Return the 8x8 real matrix R_b such that x*b = R_b @ x for all x.

    The matrix is constructed via:
      R_b[k, i] = sum_j b[j] * C[i, j, k]

    where C is the structure constants tensor.

    Args:
        b: Octonion instance with shape [..., 8].

    Returns:
        Tensor of shape [..., 8, 8] where result[..., k, i] = sum_j b_j * C[i, j, k].
    """
    C = STRUCTURE_CONSTANTS.to(device=b.components.device, dtype=b.components.dtype)
    # R[k, i] = sum_j b[j] * C[i, j, k]
    # So (R @ x)[k] = sum_i R[k,i] * x[i] = sum_{i,j} b[j] * C[i,j,k] * x[i] = (x*b)[k]
    return torch.einsum("...j, ijk -> ...ki", b.components, C)
