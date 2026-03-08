"""Structure constants tensor and octonion multiplication via Fano plane.

Implements vectorized octonion multiplication using a [8, 8, 8] structure
constants tensor built from the Fano plane triples. This is the production
multiplication path -- all downstream operations use this.

Convention: Baez 2002, mod-7 Fano plane.
"""

from __future__ import annotations

import torch

from octonion._fano import FANO_PLANE


def _build_structure_constants() -> torch.Tensor:
    """Build the [8, 8, 8] structure constants tensor for octonion multiplication.

    C[i,j,k] gives the coefficient of e_k in the product e_i * e_j.
    Uses the Baez 2002 / mod-7 Fano plane convention.

    The tensor has exactly 64 non-zero entries:
      - 15 from identity element (C[0,i,i] and C[i,0,i] for all i, with overlap at C[0,0,0])
      - 7 from imaginary unit squaring (C[i,i,0] = -1 for i=1..7)
      - 42 from Fano plane triples (7 triples * 6 cyclic/anti-cyclic orientations)
    """
    C = torch.zeros(8, 8, 8, dtype=torch.float64)

    # e_0 is the identity
    for i in range(8):
        C[0, i, i] = 1.0  # e_0 * e_i = e_i
        C[i, 0, i] = 1.0  # e_i * e_0 = e_i

    # e_i * e_i = -e_0 for i > 0
    for i in range(1, 8):
        C[i, i, 0] = -1.0

    # The 7 Fano plane triples with cyclic and anti-cyclic products
    for i, j, k in FANO_PLANE.triples:
        # Forward cyclic: e_i * e_j = e_k, e_j * e_k = e_i, e_k * e_i = e_j
        C[i, j, k] = 1.0
        C[j, k, i] = 1.0
        C[k, i, j] = 1.0
        # Reverse (anti-cyclic): e_j * e_i = -e_k, e_k * e_j = -e_i, e_i * e_k = -e_j
        C[j, i, k] = -1.0
        C[k, j, i] = -1.0
        C[i, k, j] = -1.0

    return C


# Built once at import time -- module-level constant
STRUCTURE_CONSTANTS = _build_structure_constants()


def octonion_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply two octonions represented as [..., 8] tensors.

    Uses structure constants tensor for fully vectorized computation.
    Supports arbitrary batch dimensions via broadcasting.

    Args:
        a: Tensor of shape [..., 8] (left operand).
        b: Tensor of shape [..., 8] (right operand).

    Returns:
        Tensor of shape [..., 8] representing the octonion product a * b.
    """
    C = STRUCTURE_CONSTANTS.to(device=a.device, dtype=a.dtype)
    return torch.einsum("...i, ijk, ...j -> ...k", a, C, b)
