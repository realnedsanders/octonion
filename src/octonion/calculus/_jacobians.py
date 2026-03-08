"""Analytic 8x8 Jacobian matrices for all 7 octonionic primitives.

Each function computes the exact Jacobian matrix of an octonionic operation
at a given point. These are used for:

1. **Autograd backward passes**: The Jacobian encodes the derivative that
   backward() uses to propagate gradients.
2. **Triple-check verification**: Compared against finite-difference
   numeric Jacobians and autograd-computed Jacobians.
3. **Chain rule composition**: Composite operation Jacobians are products
   of individual primitive Jacobians (matrix multiplication IS associative
   even though octonion multiplication is not).

All Jacobians operate on raw [..., 8] tensors, not Octonion instances.
Output shape is [..., 8, 8] for octonionic-valued operations and
[..., 1, 8] for scalar-valued operations (inner_product).

Convention: Baez 2002, mod-7 Fano plane basis.
"""

from __future__ import annotations

from typing import Tuple

import torch

from octonion._multiplication import STRUCTURE_CONSTANTS


def jacobian_mul(
    a: torch.Tensor, b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Analytic 8x8 Jacobians for octonion_mul(a, b).

    For f(a, b) = a * b with structure constants C[i,j,k]:
      (a*b)_k = sum_{i,j} C[i,j,k] * a_i * b_j

    Jacobian w.r.t. a:
      J_a[k, i] = d(a*b)_k / da_i = sum_j C[i,j,k] * b_j

    Jacobian w.r.t. b:
      J_b[k, j] = d(a*b)_k / db_j = sum_i C[i,j,k] * a_i

    Args:
        a: Octonion tensor [..., 8].
        b: Octonion tensor [..., 8].

    Returns:
        (J_a, J_b) each of shape [..., 8, 8].
    """
    C = STRUCTURE_CONSTANTS.to(device=a.device, dtype=a.dtype)
    # J_a[k, i] = sum_j C[i,j,k] * b[j]
    J_a = torch.einsum("ijk, ...j -> ...ki", C, b)
    # J_b[k, j] = sum_i C[i,j,k] * a[i]
    J_b = torch.einsum("ijk, ...i -> ...kj", C, a)
    return J_a, J_b


def jacobian_exp(o: torch.Tensor) -> torch.Tensor:
    """Analytic 8x8 Jacobian for octonion_exp at point o.

    For exp(o) where o = a + v (a = o[0] scalar, v = o[1:7] imaginary vector):
      exp(o) = exp(a) * (cos(||v||) + sin(||v||)/||v|| * v)

    Let r = ||v||, ea = exp(a). Then:
      result[0]   = ea * cos(r)
      result[k]   = ea * sin(r)/r * v[k-1]  for k=1..7

    Jacobian structure (partials in 0-indexed output vs 0-indexed input):

    Row 0 (real output):
      d/da:   ea * cos(r)
      d/dv_i: -ea * sin(r) * v_i / r

    Rows k=1..7 (imag output component k):
      d/da:   ea * sin(r)/r * v[k-1]
      d/dv_i: ea * [ cos(r)/r * v[k-1]*v_i/r
                      + sin(r) * (delta_{k-1,i}/r - v[k-1]*v_i/r^2) / r ]
            = ea * [ (cos(r)*r - sin(r)) * v[k-1]*v_i / r^3
                      + sin(r)/r * delta_{k-1,i} ]

    Near-zero ||v|| handling via L'Hopital:
      sin(r)/r -> 1
      cos(r)/r -> 1/r (but multiplied by terms that vanish, so overall 0)
      (cos(r)*r - sin(r))/r^3 -> -1/3 as r -> 0

    Args:
        o: Octonion tensor [..., 8].

    Returns:
        Jacobian tensor [..., 8, 8].
    """
    a = o[..., 0:1]   # [..., 1]
    v = o[..., 1:]     # [..., 7]

    r_sq = torch.sum(v ** 2, dim=-1, keepdim=True)  # [..., 1]
    r = torch.sqrt(r_sq)  # [..., 1]

    ea = torch.exp(a)  # [..., 1]

    # Threshold for near-zero handling
    eps = 1e-12
    near_zero = (r < eps)

    # Safe denominators
    safe_r = torch.where(near_zero, torch.ones_like(r), r)
    safe_r_sq = safe_r ** 2
    safe_r_cu = safe_r ** 3

    cos_r = torch.cos(r)   # [..., 1]
    sin_r = torch.sin(r)   # [..., 1]

    # sinc = sin(r)/r, handling r -> 0 via sinc(0) = 1
    sinc = torch.where(near_zero, torch.ones_like(r), sin_r / safe_r)

    # coeff for the outer-product term: (cos(r)*r - sin(r)) / r^3
    # At r=0: limit is -1/3
    outer_coeff = torch.where(
        near_zero,
        torch.full_like(r, -1.0 / 3.0),
        (cos_r * safe_r - sin_r) / safe_r_cu,
    )

    # Build 8x8 Jacobian
    batch_shape = o.shape[:-1]
    J = torch.zeros(*batch_shape, 8, 8, dtype=o.dtype, device=o.device)

    # --- Row 0: real output ---
    # d(ea*cos(r)) / da = ea * cos(r)
    J[..., 0, 0] = (ea * cos_r).squeeze(-1)

    # d(ea*cos(r)) / dv_i = -ea * sin(r) * v_i / r = -ea * sinc * v_i
    # Shapes: ea: [..., 1], sinc: [..., 1], v: [..., 7]
    # Product broadcasts to [..., 7], which matches J[..., 0, 1:] target.
    row0_imag = (-ea * sinc * v)  # [..., 7]
    J[..., 0, 1:] = row0_imag

    # --- Rows 1..7: imaginary output components ---
    # d(ea * sinc * v_{k-1}) / da = ea * sinc * v_{k-1}
    # Shapes: ea*sinc: [..., 1], v: [..., 7] -> [..., 7]
    col0_imag = ea * sinc * v  # [..., 7]
    J[..., 1:, 0] = col0_imag

    # d(ea * sinc * v_{k-1}) / dv_i:
    #   = ea * [ outer_coeff * v_{k-1} * v_i + sinc * delta_{k-1,i} ]
    # The sinc * delta term is a diagonal matrix
    # The outer_coeff * v_{k-1} * v_i term is an outer product of v with itself

    # v outer product: v[..., k] * v[..., i] -> [..., 7, 7]
    v_outer = v.unsqueeze(-2) * v.unsqueeze(-1)  # [..., 7, 7]

    # Diagonal: sinc * I_7
    diag = sinc.unsqueeze(-1) * torch.eye(7, dtype=o.dtype, device=o.device)
    # diag shape: [..., 1, 7] * [7, 7] -> [..., 7, 7] (broadcast)

    # Full 7x7 block:
    block_77 = ea.unsqueeze(-1) * (outer_coeff.unsqueeze(-1) * v_outer + diag)
    # ea: [..., 1, 1] (after unsqueeze), outer_coeff: [..., 1, 1] (after unsqueeze)
    # v_outer: [..., 7, 7], diag: [..., 7, 7]
    # block_77: [..., 7, 7]

    J[..., 1:, 1:] = block_77

    return J


def jacobian_log(o: torch.Tensor) -> torch.Tensor:
    """Analytic 8x8 Jacobian for octonion_log at point o.

    For log(o) where o = a + v (a = o[0] scalar, v = o[1:7] imaginary vector):
      log(o) = log(||o||) + arccos(a/||o||) / ||v|| * v

    Let q = ||o|| = sqrt(a^2 + ||v||^2), r = ||v||, theta = arccos(a/q).
    Then:
      result[0]   = log(q)
      result[k]   = theta/r * v[k-1]  for k=1..7

    Jacobian derivation:
      q^2 = a^2 + r^2, so dq/da = a/q, dq/dv_i = v_i/q
      theta = arccos(a/q), so dtheta/da = -1/sqrt(1-(a/q)^2) * (1/q - a^2/q^3)
                                         = -1/(r/q) * (r^2/q^3) = -r/q * r/q^2... etc.

    We derive carefully using q^2 = a^2 + r^2 and r = ||v||:
      dtheta/da  = -1/sin(theta) * d(a/q)/da = -1/sin(theta) * (q - a*a/q)/q^2
                 = -(1/sin(theta)) * r^2/(q^3)    [since q^2 - a^2 = r^2]
                 but sin(theta) = r/q, so = -(q/r) * r^2/q^3 = -r/q^2

      dtheta/dv_i = -1/sin(theta) * d(a/q)/dv_i = -1/sin(theta) * (-a*v_i/q^2)/q... etc.
      Wait, d(a/q)/dv_i = -a/(q^2) * dq/dv_i = -a/(q^2) * (v_i/q) = -a*v_i/q^3
      So dtheta/dv_i = -(q/r) * (-a*v_i/q^3) = a*v_i / (r*q^2)

    Row 0 (real output = log(q)):
      d(log q)/da   = a/q^2
      d(log q)/dv_i = v_i/q^2

    Rows k=1..7 (output = theta/r * v[k-1]):
      d/da:   (dtheta/da)/r * v[k-1] = (-r/q^2)/r * v[k-1] = -v[k-1]/q^2
      d/dv_i: d(theta/r * v[k-1])/dv_i
            = (dtheta/dv_i * r - theta * v_i/r) / r^2 * v[k-1]
              + theta/r * delta_{k-1,i}
            = v[k-1] * (a*v_i/(r*q^2)*r - theta*v_i/r) / r^2
              + theta/r * delta_{k-1,i}
            = v[k-1] * v_i * (a/q^2 - theta/r) / r^2
              + theta/r * delta_{k-1,i}

    Near-zero ||v||: limit of theta/r -> 1/q (since theta ~ r/q for small r).
    The outer coefficient (a/q^2 - theta/r)/r^2 needs careful limit analysis.

    Args:
        o: Octonion tensor [..., 8]. Must have positive norm.

    Returns:
        Jacobian tensor [..., 8, 8].
    """
    a = o[..., 0:1]   # [..., 1]
    v = o[..., 1:]     # [..., 7]

    r_sq = torch.sum(v ** 2, dim=-1, keepdim=True)  # [..., 1]
    r = torch.sqrt(r_sq)  # [..., 1]
    q_sq = a ** 2 + r_sq   # [..., 1]
    q = torch.sqrt(q_sq)   # [..., 1]

    eps = 1e-12
    near_zero_r = (r < eps)
    near_zero_q = (q < eps)

    safe_r = torch.where(near_zero_r, torch.ones_like(r), r)
    safe_q = torch.where(near_zero_q, torch.ones_like(q), q)
    safe_q_sq = safe_q ** 2
    safe_r_sq = safe_r ** 2

    # theta = arccos(a/q), clamped for numerical safety
    ratio = torch.clamp(a / safe_q, min=-1.0, max=1.0)
    theta = torch.acos(ratio)  # [..., 1]

    # theta_over_r = theta / r, with limit 1/q as r -> 0
    theta_over_r = torch.where(
        near_zero_r,
        1.0 / safe_q,
        theta / safe_r,
    )

    batch_shape = o.shape[:-1]
    J = torch.zeros(*batch_shape, 8, 8, dtype=o.dtype, device=o.device)

    # --- Row 0: real output = log(q) ---
    # d(log q)/da = a/q^2
    J[..., 0, 0] = (a / safe_q_sq).squeeze(-1)
    # d(log q)/dv_i = v_i/q^2
    J[..., 0, 1:] = v / safe_q_sq

    # --- Rows 1..7: imag output = theta/r * v ---
    # d(theta/r * v_{k-1}) / da = -v_{k-1}/q^2
    J[..., 1:, 0] = -v / safe_q_sq

    # d(theta/r * v_{k-1})/dv_i:
    #   = v_{k-1} * v_i * (a/q^2 - theta/r) / r^2  + theta/r * delta_{k-1,i}
    #
    # The outer coefficient: (a/q^2 - theta/r) / r^2
    # At r -> 0: theta ~ r/q - r^3/(3q^3) + ...
    #   theta/r ~ 1/q - r^2/(3q^3) + ...
    #   a/q^2 - theta/r ~ a/q^2 - 1/q + r^2/(3q^3) + ...
    #                   = (a - q)/q^2 + r^2/(3q^3) + ...
    #                   = (a - q)/q^2 + r^2/(3q^3) + ...
    # Since q = sqrt(a^2 + r^2) ~ a + r^2/(2a) for small r (assuming a > 0):
    #   a - q ~ -r^2/(2a)
    #   (a - q)/q^2 ~ -r^2/(2a * a^2) = -r^2/(2a^3)
    # So (a/q^2 - theta/r) / r^2 ~ (-1/(2a^3) + 1/(3a^3)) + ...
    #                               = -1/(6a^3) + ...
    # This limit depends on a. More generally, using Taylor expansion of theta:
    #   theta = arccos(a/q), where a/q = cos(theta)
    #   For small r: a/q ~ 1 - r^2/(2q^2), theta ~ r * sqrt(1 - (a/q)^2) / ... Hmm.
    #
    # A cleaner approach: at r=0, the function theta/r * v = 0 (since v=0).
    # The Jacobian rows 1..7 at r=0: the imaginary output IS zero, and
    # the derivative d(theta/r * v_k)/dv_i at v=0 = (theta/r)|_{r=0} * delta_{k,i}
    # = (1/q) * delta_{k,i}. The outer product term vanishes since v=0.
    #
    # For the outer_coeff at general r:
    outer_coeff = torch.where(
        near_zero_r,
        # limit: -1/(3*q^3) is the leading correction; but at v=0 the v_outer
        # product is 0 anyway, so the exact value here doesn't matter for correctness.
        # We use a numerically stable expression.
        -torch.ones_like(r) / (3.0 * safe_q * safe_q_sq),
        (a / safe_q_sq - theta_over_r) / safe_r_sq,
    )

    # v outer product
    v_outer = v.unsqueeze(-2) * v.unsqueeze(-1)  # [..., 7, 7]

    # diagonal term
    diag = theta_over_r.unsqueeze(-1) * torch.eye(7, dtype=o.dtype, device=o.device)

    # Full 7x7 block
    block_77 = outer_coeff.unsqueeze(-1) * v_outer + diag

    J[..., 1:, 1:] = block_77

    return J


def jacobian_conjugate(o: torch.Tensor) -> torch.Tensor:
    """Analytic 8x8 Jacobian for octonion conjugation.

    conj(o) = [o_0, -o_1, -o_2, ..., -o_7]

    The Jacobian is the constant diagonal matrix:
      diag([1, -1, -1, -1, -1, -1, -1, -1])

    Args:
        o: Octonion tensor [..., 8]. Used only for shape/dtype/device.

    Returns:
        Jacobian tensor [..., 8, 8].
    """
    batch_shape = o.shape[:-1]
    diag_vals = torch.tensor(
        [1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        dtype=o.dtype,
        device=o.device,
    )
    # Expand to batch shape + [8]
    diag_vals = diag_vals.expand(*batch_shape, 8)
    return torch.diag_embed(diag_vals)


def jacobian_inverse(o: torch.Tensor) -> torch.Tensor:
    """Analytic 8x8 Jacobian for octonion inverse.

    For f(o) = conj(o) / ||o||^2:
      f_k(o) = conj(o)_k / ||o||^2

    where conj(o)_0 = o_0, conj(o)_k = -o_k for k>0.

    Using the quotient rule:
      d f_k / d o_i = (d(conj(o)_k)/d o_i * ||o||^2 - conj(o)_k * d(||o||^2)/d o_i) / ||o||^4

    with d(||o||^2)/d o_i = 2*o_i and d(conj(o)_k)/d o_i = diag_conj[k,i].

    Simplifies to:
      J[k, i] = (diag_conj[k,i] * n2 - conj_k * 2 * o_i) / n2^2
              = diag_conj[k,i] / n2 - 2 * conj_k * o_i / n2^2

    Args:
        o: Octonion tensor [..., 8]. Must have non-zero norm.

    Returns:
        Jacobian tensor [..., 8, 8].
    """
    n2 = torch.sum(o ** 2, dim=-1, keepdim=True)  # [..., 1]
    n4 = n2 ** 2  # [..., 1]

    # conj(o): negate imaginary, keep real
    conj = o.clone()
    conj[..., 1:] = -conj[..., 1:]

    # diag_conj: diagonal matrix [1, -1, -1, ..., -1]
    batch_shape = o.shape[:-1]
    diag_vals = torch.tensor(
        [1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        dtype=o.dtype,
        device=o.device,
    )
    diag_conj = torch.diag_embed(diag_vals.expand(*batch_shape, 8))  # [..., 8, 8]

    # J[k, i] = diag_conj[k,i] / n2 - 2 * conj_k * o_i / n2^2
    # conj: [..., 8] -> [..., 8, 1] for outer product
    # o: [..., 8] -> [..., 1, 8] for outer product
    outer = conj.unsqueeze(-1) * o.unsqueeze(-2)  # [..., 8, 8] = conj_k * o_i

    J = diag_conj / n2.unsqueeze(-1) - 2.0 * outer / n4.unsqueeze(-1)

    return J


def jacobian_inner_product(
    a: torch.Tensor, b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Analytic Jacobians for inner_product(a, b) = sum_i a_i * b_i.

    This is a scalar-valued function of two octonion arguments.
    The Jacobian is simply the other argument (as a row vector):
      J_a[0, i] = b_i    (d<a,b>/da_i = b_i)
      J_b[0, j] = a_j    (d<a,b>/db_j = a_j)

    Args:
        a: Octonion tensor [..., 8].
        b: Octonion tensor [..., 8].

    Returns:
        (J_a, J_b) each of shape [..., 1, 8].
    """
    J_a = b.unsqueeze(-2)  # [..., 1, 8]
    J_b = a.unsqueeze(-2)  # [..., 1, 8]
    return J_a, J_b


def jacobian_cross_product(
    a: torch.Tensor, b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Analytic 8x8 Jacobians for cross_product(a, b).

    The cross product is defined as:
      cross(a, b) = Im( Im(a) * Im(b) )

    where Im(x) zeros out the real part. Let a_pure = [0, a_1, ..., a_7]
    and b_pure = [0, b_1, ..., b_7].

    Then cross(a, b)_k = (a_pure * b_pure)_k for k=1..7, and cross(a,b)_0 = 0.

    Since a_pure_i depends on a_i for i>0 (and is 0 for i=0), and the
    multiplication is bilinear in (a_pure, b_pure):

    d cross(a,b)_k / d a_i:
      - For i=0: = 0 (a_pure doesn't depend on a_0)
      - For i>0: = d(a_pure * b_pure)_k / d a_pure_i = sum_j C[i,j,k] * b_pure_j
                 = sum_{j=1..7} C[i,j,k] * b_j  (since b_pure_0 = 0, but C[i,0,k] * 0 = 0)

    Similarly for d cross(a,b)_k / d b_j.

    Also cross(a,b)_0 = 0 always, so row 0 of the Jacobian is zero.

    Args:
        a: Octonion tensor [..., 8].
        b: Octonion tensor [..., 8].

    Returns:
        (J_a, J_b) each of shape [..., 8, 8].
    """
    C = STRUCTURE_CONSTANTS.to(device=a.device, dtype=a.dtype)

    # Construct pure imaginary versions: zero real part
    a_pure = torch.zeros_like(a)
    a_pure[..., 1:] = a[..., 1:]
    b_pure = torch.zeros_like(b)
    b_pure[..., 1:] = b[..., 1:]

    # The cross product output_k = sum_{i,j} C[i,j,k] * a_pure_i * b_pure_j
    # for k=1..7, and output_0 = 0.
    #
    # J_a[k, i] = d output_k / d a_i
    #   If k=0 or i=0: J_a[k,i] = 0
    #   Otherwise: = sum_j C[i,j,k] * b_pure_j
    #
    # J_b[k, j] = d output_k / d b_j
    #   If k=0 or j=0: J_b[k,j] = 0
    #   Otherwise: = sum_i C[i,j,k] * a_pure_i

    # Full Jacobian of multiplication at (a_pure, b_pure)
    # J_mul_a[k, i] = sum_j C[i,j,k] * b_pure_j
    J_mul_a = torch.einsum("ijk, ...j -> ...ki", C, b_pure)
    # J_mul_b[k, j] = sum_i C[i,j,k] * a_pure_i
    J_mul_b = torch.einsum("ijk, ...i -> ...kj", C, a_pure)

    # Zero out: row 0 (output real = 0), col 0 (input real doesn't contribute)
    batch_shape = a.shape[:-1]
    J_a = torch.zeros(*batch_shape, 8, 8, dtype=a.dtype, device=a.device)
    J_b = torch.zeros(*batch_shape, 8, 8, dtype=a.dtype, device=a.device)

    # Copy the 7x7 imaginary-to-imaginary block
    # But actually, J_mul_a already has the right structure because b_pure has
    # zero real part, so column 0 contribution (from C[0,j,k]*b_pure_j) is
    # C[0,j,k]*0 = C[0,j,k]*b_j for j>0 only... wait, b_pure[0]=0 but
    # the sum over j includes j=0 with b_pure[0]=0.
    # And row 0 (k=0): the product of two pure imaginary octonions has
    # a real part = -<a_pure, b_pure>, which we need to zero out.

    # The cleanest approach: just copy the relevant block
    # Output rows 1..7, input cols 0..7 from J_mul_a, then zero col 0
    J_a[..., 1:, 1:] = J_mul_a[..., 1:, 1:]
    J_b[..., 1:, 1:] = J_mul_b[..., 1:, 1:]

    # Column 0 stays zero (real input doesn't affect cross product)
    # Row 0 stays zero (cross product has zero real output)

    return J_a, J_b
