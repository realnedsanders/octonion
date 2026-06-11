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

import torch

from octonion._multiplication import structure_constants
from octonion._operations import _series_threshold


def jacobian_mul(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    C = structure_constants(a.device, a.dtype)
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

    Near-zero ||v|| handling via Taylor series (below sqrt(machine eps),
    where the exact expressions lose digits to cancellation):
      sin(r)/r = 1 - r^2/6 + O(r^4)
      (cos(r)*r - sin(r))/r^3 = -1/3 + r^2/30 + O(r^4)

    Args:
        o: Octonion tensor [..., 8].

    Returns:
        Jacobian tensor [..., 8, 8].
    """
    a = o[..., 0:1]  # [..., 1]
    v = o[..., 1:]  # [..., 7]

    r_sq = torch.sum(v**2, dim=-1, keepdim=True)  # [..., 1]
    r = torch.sqrt(r_sq)  # [..., 1]

    ea = torch.exp(a)  # [..., 1]

    # Below sqrt(machine eps) the exact expressions cancel catastrophically;
    # the O(r^2) series are accurate to one ulp there.
    near_zero = r < _series_threshold(o.dtype)

    # Safe denominators
    safe_r = torch.where(near_zero, torch.ones_like(r), r)
    safe_r_cu = safe_r**3

    cos_r = torch.cos(r)  # [..., 1]
    sin_r = torch.sin(r)  # [..., 1]

    # sinc = sin(r)/r = 1 - r^2/6 + O(r^4)
    sinc = torch.where(near_zero, 1.0 - r_sq / 6.0, sin_r / safe_r)

    # coeff for the outer-product term:
    # (cos(r)*r - sin(r)) / r^3 = -1/3 + r^2/30 + O(r^4)
    outer_coeff = torch.where(
        near_zero,
        -1.0 / 3.0 + r_sq / 30.0,
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
    row0_imag = -ea * sinc * v  # [..., 7]
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

    Near-zero ||v|| handling (below r < sqrt(machine eps) * q, where the
    exact expressions cancel catastrophically), via Taylor series valid
    for a > 0:
      theta/r = 1/a - r^2/(3 a^3) + O(r^4)
      (a/q^2 - theta/r)/r^2 = -2/(3 a^3) + O(r^2)

    On the negative real axis (r = 0, a < 0) the logarithm itself is
    undefined (no preferred imaginary direction) and theta/r diverges, so
    the Jacobian rows/columns touching the imaginary block are NaN there.
    Arbitrarily close to the axis (tiny non-zero r) the exact expressions
    are accurate: theta = atan2(r, a) ~ pi is O(1) and no cancellation
    occurs for a < 0.

    Args:
        o: Octonion tensor [..., 8]. Must have positive norm.

    Returns:
        Jacobian tensor [..., 8, 8].
    """
    a = o[..., 0:1]  # [..., 1]
    v = o[..., 1:]  # [..., 7]

    r_sq = torch.sum(v**2, dim=-1, keepdim=True)  # [..., 1]
    r = torch.sqrt(r_sq)  # [..., 1]
    q_sq = a**2 + r_sq  # [..., 1]
    q = torch.sqrt(q_sq)  # [..., 1]

    near_zero_q = q < torch.finfo(o.dtype).tiny ** 0.5
    safe_q = torch.where(near_zero_q, torch.ones_like(q), q)
    safe_q_sq = safe_q**2

    # Series branch: only valid (and only needed) for a > 0. For a < 0 the
    # exact formulas are accurate at any r > 0; at exactly r = 0 the
    # derivative genuinely diverges -> NaN.
    series = (r < _series_threshold(o.dtype) * safe_q) & (a > 0)
    undefined = (r == 0) & (a < 0)

    safe_r = torch.where(r == 0, torch.ones_like(r), r)
    safe_r_sq = safe_r**2
    safe_a = torch.where(a > 0, a, torch.ones_like(a))

    # theta = atan2(r, a): accurate in all quadrants. (arccos(a/q) loses
    # half the significant digits for small theta.)
    theta = torch.atan2(r, a)  # [..., 1]

    # theta_over_r = theta / r, series 1/a - r^2/(3a^3) near the positive
    # real axis, NaN on the negative real axis.
    theta_over_r = torch.where(
        series,
        1.0 / safe_a - r_sq / (3.0 * safe_a**3),
        theta / safe_r,
    )
    nan = torch.full_like(theta_over_r, float("nan"))
    theta_over_r = torch.where(undefined, nan, theta_over_r)

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
    # The outer coefficient (a/q^2 - theta/r) / r^2. Near the positive real
    # axis (a > 0, r -> 0), with theta/r = 1/a - r^2/(3a^3) + O(r^4) and
    # a/q^2 = 1/a - r^2/a^3 + O(r^4):
    #   a/q^2 - theta/r = -r^2/a^3 + r^2/(3a^3) + O(r^4)
    #                   = -(2/3) r^2/a^3 + O(r^4)
    # so the limit is -2/(3 a^3). (The outer product v v^T vanishes as r^2
    # there, but the correct limit keeps the Jacobian continuous.) On the
    # negative real axis the coefficient diverges -> NaN, matching
    # theta_over_r above.
    outer_coeff = torch.where(
        series,
        -2.0 / (3.0 * safe_a**3),
        (a / safe_q_sq - theta_over_r) / safe_r_sq,
    )
    outer_coeff = torch.where(undefined, nan, outer_coeff)

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
    n2 = torch.sum(o**2, dim=-1, keepdim=True)  # [..., 1]
    n4 = n2**2  # [..., 1]

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


def jacobian_inner_product(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def jacobian_cross_product(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    C = structure_constants(a.device, a.dtype)

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
