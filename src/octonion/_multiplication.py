"""Structure constants tensor and octonion multiplication via Fano plane.

The multiplication of two octonions a, b with components a_i, b_j is:
  (a * b)_k = sum_{i,j} C[i,j,k] * a_i * b_j

where C is the [8,8,8] structure constants tensor built from the Fano plane.
This is computed as a single torch.einsum call for full vectorization.

Convention: Baez 2002, mod-7 Fano plane triples.
"""

from __future__ import annotations

import torch

from octonion._fano import FANO_PLANE


def _build_structure_constants() -> torch.Tensor:
    """Build the [8, 8, 8] structure constants tensor for octonion multiplication.

    C[i,j,k] gives the coefficient of e_k in the product e_i * e_j.
    Uses the Baez 2002 / mod-7 Fano plane convention.

    Non-zero entries:
    - e_0 is the identity: C[0,i,i] = 1 and C[i,0,i] = 1 for all i
    - Imaginary squaring: C[i,i,0] = -1 for i > 0
    - Fano triples: For each triple (i,j,k):
        Cyclic: C[i,j,k] = C[j,k,i] = C[k,i,j] = +1
        Anti-cyclic: C[j,i,k] = C[k,j,i] = C[i,k,j] = -1

    Returns:
        Tensor of shape [8, 8, 8] with float64 entries in {-1, 0, +1}.
    """
    C = torch.zeros(8, 8, 8, dtype=torch.float64)

    # e_0 is the identity
    for i in range(8):
        C[0, i, i] = 1.0  # e_0 * e_i = e_i
        C[i, 0, i] = 1.0  # e_i * e_0 = e_i

    # e_i * e_i = -e_0 for i > 0
    for i in range(1, 8):
        C[i, i, 0] = -1.0

    # The 7 Fano plane triples
    for i, j, k in FANO_PLANE.triples:
        # Forward cyclic: e_i * e_j = e_k, e_j * e_k = e_i, e_k * e_i = e_j
        C[i, j, k] = 1.0
        C[j, k, i] = 1.0
        C[k, i, j] = 1.0
        # Reverse anti-cyclic: e_j * e_i = -e_k, etc.
        C[j, i, k] = -1.0
        C[k, j, i] = -1.0
        C[i, k, j] = -1.0

    return C


# Built once at import time (module-level constant)
STRUCTURE_CONSTANTS = _build_structure_constants()


def octonion_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply two octonions represented as [..., 8] tensors.

    Uses structure constants tensor for fully vectorized computation.
    Supports arbitrary batch dimensions via PyTorch broadcasting.
    Handles mixed dtypes by promoting to a common type.

    Args:
        a: Octonion tensor of shape [..., 8].
        b: Octonion tensor of shape [..., 8].

    Returns:
        Product tensor of shape [..., 8] where ... is the broadcast batch shape.
    """
    # Promote a and b to common dtype (handles float32/float64 mismatch)
    common_dtype = torch.promote_types(a.dtype, b.dtype)
    a = a.to(dtype=common_dtype)
    b = b.to(dtype=common_dtype)
    C = STRUCTURE_CONSTANTS.to(device=a.device, dtype=common_dtype)
    return torch.einsum("...i, ijk, ...j -> ...k", a, C, b)
