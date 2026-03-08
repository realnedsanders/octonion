"""Cayley-Dickson recursive multiplication as cross-check for Fano plane implementation.

Represents octonions as pairs of quaternions (a, b) where x = (a, b),
and implements multiplication via the Cayley-Dickson formula:

    (a, b)(c, d) = (ac - db*, a*d + cb)

where a, b, c, d are quaternions and * denotes quaternion conjugation.

Convention: Baez 2002, Section 2.2.

BASIS MAPPING (derived empirically):
The naive Cayley-Dickson quaternion-pair split x[:4], x[4:] produces a valid
octonion algebra but with a DIFFERENT basis labeling than the Baez 2002 mod-7
Fano plane convention. The relationship is:

    fano_table[i, j, k] = cd_table[P[i], P[j], P[k]]

where P = [0, 1, 2, 5, 3, 7, 6, 4] maps Fano indices to CD indices.

This means Fano basis element e_i^fano corresponds to CD basis element e_{P[i]}^cd.
To convert a Fano-convention vector to CD convention, we permute components via Pinv.
To convert a CD-convention result back to Fano convention, we index via P.

This mapping is applied internally by cayley_dickson_mul() so that it accepts
and returns octonions in the Fano plane (Baez 2002) convention, making the
cross-check a direct comparison.
"""

from __future__ import annotations

import torch

# P: maps Fano indices to CD indices.
# fano_table[i, j, k] = cd_table[P[i], P[j], P[k]]
# Verified by brute-force search over all 7! permutations.
_P = [0, 1, 2, 5, 3, 7, 6, 4]

# Pinv: maps CD indices to Fano indices (inverse permutation of P).
# Pinv[P[i]] = i for all i.
_PINV = [0, 1, 2, 4, 7, 3, 6, 5]


def quaternion_mul(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Multiply quaternions as [..., 4] tensors using the Hamilton product.

    Convention: q = q0 + q1*i + q2*j + q3*k
    Product rules: i^2 = j^2 = k^2 = ijk = -1

    Args:
        p: Quaternion tensor of shape [..., 4].
        q: Quaternion tensor of shape [..., 4].

    Returns:
        Product tensor of shape [..., 4].
    """
    p0, p1, p2, p3 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    return torch.stack(
        [
            p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3,
            p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2,
            p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1,
            p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0,
        ],
        dim=-1,
    )


def quaternion_conj(q: torch.Tensor) -> torch.Tensor:
    """Conjugate a quaternion: negate imaginary parts, preserve real part.

    Args:
        q: Quaternion tensor of shape [..., 4].

    Returns:
        Conjugated quaternion tensor of shape [..., 4].
    """
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _raw_cayley_dickson_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Raw Cayley-Dickson multiplication WITHOUT basis permutation.

    This uses the naive quaternion-pair split and produces results in the
    CD-native basis labeling. Used internally; most callers should use
    cayley_dickson_mul() which handles the Fano plane convention mapping.

    Formula (Baez 2002): (a,b)(c,d) = (ac - db*, a*d + cb)
    """
    a, b = x[..., :4], x[..., 4:]
    c, d = y[..., :4], y[..., 4:]

    real_part = quaternion_mul(a, c) - quaternion_mul(d, quaternion_conj(b))
    imag_part = quaternion_mul(quaternion_conj(a), d) + quaternion_mul(c, b)

    return torch.cat([real_part, imag_part], dim=-1)


def cayley_dickson_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Multiply octonions via Cayley-Dickson construction (Fano convention).

    Accepts octonions in the Baez 2002 Fano plane basis convention,
    internally permutes to the Cayley-Dickson natural basis, performs
    the multiplication, then permutes back.

    This ensures cayley_dickson_mul(a, b) produces results identical to
    octonion_mul(a, b) from _multiplication.py, enabling the cross-check
    (FOUND-01 success criterion 3).

    Formula (Baez 2002): (a,b)(c,d) = (ac - db*, a*d + cb)

    Args:
        x: Octonion tensor of shape [..., 8] in Fano convention.
        y: Octonion tensor of shape [..., 8] in Fano convention.

    Returns:
        Product tensor of shape [..., 8] in Fano convention.
    """
    # Permute inputs from Fano basis to CD basis.
    # Fano e_i has component i=1. In CD, this should be e_{P[i]} with component P[i]=1.
    # x[..., Pinv] gives x_cd where x_cd[k] = x[Pinv[k]], so x_cd[P[i]] = x[Pinv[P[i]]] = x[i] = 1.
    x_cd = x[..., _PINV]
    y_cd = y[..., _PINV]

    # Multiply in CD basis
    result_cd = _raw_cayley_dickson_mul(x_cd, y_cd)

    # Permute result from CD basis back to Fano basis.
    # We need fano_result[k] = result_cd[P[k]].
    # result_cd[..., P] gives a tensor where position k has result_cd[P[k]].
    return result_cd[..., _P]
