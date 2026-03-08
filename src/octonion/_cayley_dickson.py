"""Cayley-Dickson recursive multiplication as cross-check.

Implements octonion multiplication via the Cayley-Dickson construction,
treating octonions as pairs of quaternions. This is a separate implementation
used to cross-validate the Fano plane structure constants approach.

Formula (Baez 2002): (a,b)(c,d) = (ac - db*, a*d + cb)
where a,b,c,d are quaternions and * denotes quaternion conjugation.

IMPORTANT: The naive quaternion-pair split x[:4], x[4:] uses the "standard"
Cayley-Dickson basis (1, i, j, k, l, li, lj, lk) which does NOT directly
correspond to the Baez 2002 mod-7 Fano plane indexing (e1..e7). A basis
permutation is required to make the two conventions agree. This module
applies the permutation internally so that cayley_dickson_mul() accepts
and returns octonions in the Fano plane basis convention.

The permutation mapping (discovered empirically by exhaustive search):
  Fano -> CD: [f0, -f1, -f2, f4, f3, f7, f5, f6]
  CD -> Fano: [c0, -c1, -c2, c4, c3, c6, c7, c5]

This corresponds to a signed index permutation P with:
  P = {0:0, 1:1, 2:2, 3:4, 4:3, 5:7, 6:5, 7:6}
  S = {0:+1, 1:-1, 2:-1, 3:+1, 4:+1, 5:+1, 6:+1, 7:+1}
"""

from __future__ import annotations

import torch


def quaternion_mul(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Multiply quaternions as [..., 4] tensors using the Hamilton product.

    Convention: q = q0 + q1*i + q2*j + q3*k
    with i^2 = j^2 = k^2 = ijk = -1.

    Args:
        p: Tensor of shape [..., 4] (left quaternion).
        q: Tensor of shape [..., 4] (right quaternion).

    Returns:
        Tensor of shape [..., 4] representing the quaternion product p * q.
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
    """Conjugate quaternion: negate imaginary parts.

    Args:
        q: Tensor of shape [..., 4].

    Returns:
        Tensor of shape [..., 4] with q0 unchanged and q1,q2,q3 negated.
    """
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _fano_to_cd(x: torch.Tensor) -> torch.Tensor:
    """Convert octonion from Fano plane basis to Cayley-Dickson basis.

    Applies the signed permutation: [f0, -f1, -f2, f4, f3, f7, f5, f6]
    """
    cd = torch.zeros_like(x)
    cd[..., 0] = x[..., 0]
    cd[..., 1] = -x[..., 1]
    cd[..., 2] = -x[..., 2]
    cd[..., 3] = x[..., 4]
    cd[..., 4] = x[..., 3]
    cd[..., 5] = x[..., 7]
    cd[..., 6] = x[..., 5]
    cd[..., 7] = x[..., 6]
    return cd


def _cd_to_fano(cd: torch.Tensor) -> torch.Tensor:
    """Convert octonion from Cayley-Dickson basis to Fano plane basis.

    Applies the inverse signed permutation: [c0, -c1, -c2, c4, c3, c6, c7, c5]
    """
    fano = torch.zeros_like(cd)
    fano[..., 0] = cd[..., 0]
    fano[..., 1] = -cd[..., 1]
    fano[..., 2] = -cd[..., 2]
    fano[..., 3] = cd[..., 4]
    fano[..., 4] = cd[..., 3]
    fano[..., 5] = cd[..., 6]
    fano[..., 6] = cd[..., 7]
    fano[..., 7] = cd[..., 5]
    return fano


def _cayley_dickson_mul_raw(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Multiply octonions in the raw Cayley-Dickson basis (1, i, j, k, l, li, lj, lk).

    Formula (Baez 2002): (a,b)(c,d) = (ac - db*, a*d + cb)
    where a,b,c,d are quaternions and * denotes quaternion conjugation.

    This is the pure CD formula without basis conversion. Used internally.
    """
    a, b = x[..., :4], x[..., 4:]
    c, d = y[..., :4], y[..., 4:]

    # (a,b)(c,d) = (ac - db*, a*d + cb)
    first = quaternion_mul(a, c) - quaternion_mul(d, quaternion_conj(b))
    second = quaternion_mul(quaternion_conj(a), d) + quaternion_mul(c, b)

    return torch.cat([first, second], dim=-1)


def cayley_dickson_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Multiply octonions via Cayley-Dickson construction (Fano plane basis).

    Accepts and returns octonions in the Fano plane basis convention
    (Baez 2002 mod-7: e1*e2=e4). Internally converts to the CD natural basis,
    applies the Cayley-Dickson formula, and converts back.

    This function exists as a cross-check against the Fano plane structure
    constants implementation (octonion_mul). Both should produce identical
    results on all inputs.

    Args:
        x: Tensor of shape [..., 8] in Fano plane basis (left octonion).
        y: Tensor of shape [..., 8] in Fano plane basis (right octonion).

    Returns:
        Tensor of shape [..., 8] in Fano plane basis representing x * y.
    """
    # Convert Fano basis -> CD basis
    cx = _fano_to_cd(x)
    cy = _fano_to_cd(y)

    # Multiply in CD basis
    cr = _cayley_dickson_mul_raw(cx, cy)

    # Convert CD basis -> Fano basis
    return _cd_to_fano(cr)
