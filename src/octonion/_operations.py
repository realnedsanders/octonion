"""Extended octonion operations: exp, log, commutator, inner product, cross product.

All operations accept Octonion instances and support batched [..., 8] tensors
via the underlying octonion_mul vectorized computation.

Convention: Baez 2002, mod-7 Fano plane.
"""

from __future__ import annotations

import torch

from octonion._multiplication import octonion_mul
from octonion._octonion import Octonion


def octonion_exp(o: Octonion | torch.Tensor) -> Octonion | torch.Tensor:
    """Compute the exponential of an octonion.

    For octonion q = a + v where a is the real part and v is the imaginary vector:
      exp(q) = exp(a) * (cos(||v||) * e_0 + sin(||v||) * v / ||v||)

    When ||v|| is near zero, reduces to exp(a) * e_0 (pure real exponential).

    Args:
        o: Octonion instance or raw [..., 8] tensor.

    Returns:
        Same type as input: Octonion if given Octonion, raw tensor if given tensor.
    """
    raw_tensor = isinstance(o, torch.Tensor) and not isinstance(o, Octonion)
    if raw_tensor:
        o = Octonion(o)
    a = o.real  # [...] scalar part
    v = o.imag  # [..., 7] imaginary vector
    v_norm = torch.sqrt(torch.sum(v**2, dim=-1))  # [...]

    # exp(a) factor
    exp_a = torch.exp(a)  # [...]

    # Handle near-zero imaginary norm: pure real exponential
    eps = 1e-15
    safe_v_norm = torch.where(v_norm > eps, v_norm, torch.ones_like(v_norm))

    cos_v = torch.cos(v_norm)  # [...]
    sin_v = torch.sin(v_norm)  # [...]

    # Normalized imaginary direction: v / ||v||
    v_hat = v / safe_v_norm.unsqueeze(-1)  # [..., 7]

    # For near-zero ||v||, sin(||v||)/||v|| -> 1, so the imaginary part -> v itself
    # But we use the full formula and zero out when near-zero
    result_real = (exp_a * cos_v).unsqueeze(-1)  # [..., 1]
    result_imag = (exp_a * sin_v).unsqueeze(-1) * v_hat  # [..., 7]

    # When ||v|| is near zero, imaginary part should be zero (not v_hat which is arbitrary)
    near_zero_mask = (v_norm < eps).unsqueeze(-1)  # [..., 1]
    result_imag = torch.where(near_zero_mask, torch.zeros_like(result_imag), result_imag)

    result = torch.cat([result_real, result_imag], dim=-1)
    return result if raw_tensor else Octonion(result)


def octonion_log(o: Octonion | torch.Tensor) -> Octonion | torch.Tensor:
    """Compute the logarithm of an octonion.

    For octonion q = a + v where a is the real part and v is the imaginary vector:
      log(q) = log(||q||) * e_0 + (arccos(a / ||q||) / ||v||) * v

    When ||v|| is near zero, reduces to log(||q||) * e_0 (pure real log).

    This is the principal logarithm. The function is an approximate inverse
    of exp for pure octonions and near-identity octonions.

    Args:
        o: Octonion instance or raw [..., 8] tensor. Must have non-zero norm.

    Returns:
        Same type as input: Octonion if given Octonion, raw tensor if given tensor.
    """
    raw_tensor = isinstance(o, torch.Tensor) and not isinstance(o, Octonion)
    if raw_tensor:
        o = Octonion(o)
    a = o.real  # [...] scalar part
    v = o.imag  # [..., 7] imaginary vector
    v_norm = torch.sqrt(torch.sum(v**2, dim=-1))  # [...]
    q_norm = o.norm()  # [...]

    eps = 1e-15

    # log(||q||) for real part
    safe_q_norm = torch.clamp(q_norm, min=eps)
    log_q_norm = torch.log(safe_q_norm)  # [...]

    # arccos(a / ||q||) / ||v|| for imaginary coefficient
    # Clamp a/||q|| to [-1, 1] for numerical safety with arccos
    ratio = torch.clamp(a / safe_q_norm, min=-1.0, max=1.0)
    theta = torch.acos(ratio)  # [...]

    safe_v_norm = torch.where(v_norm > eps, v_norm, torch.ones_like(v_norm))
    imag_coeff = theta / safe_v_norm  # [...]

    result_real = log_q_norm.unsqueeze(-1)  # [..., 1]
    result_imag = imag_coeff.unsqueeze(-1) * v  # [..., 7]

    # When ||v|| is near zero, imaginary part should be zero
    near_zero_mask = (v_norm < eps).unsqueeze(-1)
    result_imag = torch.where(near_zero_mask, torch.zeros_like(result_imag), result_imag)

    result = torch.cat([result_real, result_imag], dim=-1)
    return result if raw_tensor else Octonion(result)


def commutator(a: Octonion, b: Octonion) -> Octonion:
    """Compute the commutator [a, b] = a*b - b*a.

    The commutator measures the failure of commutativity. Properties:
    - [a, a] = 0 for all a
    - [a, b] = -[b, a] (antisymmetry)
    - For real scalars, [s, a] = 0

    Args:
        a, b: Octonion instances.

    Returns:
        Octonion representing a*b - b*a.
    """
    return (a * b) - (b * a)


def inner_product(a: Octonion, b: Octonion) -> torch.Tensor:
    """Compute the real inner product <a, b> = Re(a* * b).

    This is the standard Euclidean inner product on R^8, which equals
    the sum of component-wise products: sum_i a_i * b_i.

    Properties:
    - Symmetric: <a, b> = <b, a>
    - Positive definite: <a, a> >= 0, with equality iff a = 0
    - <a, a> = |a|^2 (norm squared)
    - Bilinear: <sa + tb, c> = s<a, c> + t<b, c>

    Args:
        a, b: Octonion instances.

    Returns:
        Scalar tensor (batch dims preserved) with the inner product value.
    """
    return torch.sum(a.components * b.components, dim=-1)


def cross_product(a: Octonion, b: Octonion) -> Octonion:
    """Compute the 7D cross product of two octonions.

    The cross product operates on the imaginary parts of the input octonions.
    For pure imaginary octonions u, v in Im(O):
      u x v = Im(u * v)

    This generalizes the 3D cross product to 7 dimensions using the Fano
    plane multiplication structure. For general octonions, the real parts
    are ignored and only the imaginary parts participate.

    Properties:
    - Antisymmetric: cross(a, b) = -cross(b, a)
    - Output is pure imaginary (real part = 0)
    - For pure imaginary a, b: antisymmetry follows from a*b + b*a = -2<a,b> (real)

    Args:
        a, b: Octonion instances.

    Returns:
        Octonion with zero real part (pure imaginary) representing the cross product.
    """
    # Extract imaginary parts as pure octonions
    a_pure_data = torch.cat(
        [torch.zeros_like(a.components[..., :1]), a.components[..., 1:]], dim=-1
    )
    b_pure_data = torch.cat(
        [torch.zeros_like(b.components[..., :1]), b.components[..., 1:]], dim=-1
    )
    # Compute product of pure imaginary parts
    product = octonion_mul(a_pure_data, b_pure_data)
    # Extract imaginary part of the product (real part of pure*pure is -<a,b>, drop it)
    result_data = torch.cat(
        [torch.zeros_like(product[..., :1]), product[..., 1:]], dim=-1
    )
    return Octonion(result_data)
