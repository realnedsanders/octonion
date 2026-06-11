"""Extended octonion operations: exp, log, commutator, inner product, cross product.

All operations accept Octonion instances and support batched [..., 8] tensors
via the underlying octonion_mul vectorized computation.

Convention: Baez 2002, mod-7 Fano plane.
"""

from __future__ import annotations

import math

import torch

from octonion._multiplication import octonion_mul
from octonion._octonion import Octonion


def _series_threshold(dtype: torch.dtype) -> float:
    """Branch point between exact formulas and small-r Taylor series.

    sqrt(machine eps): below it the O(r^2) series truncation error is under
    one ulp, while the exact expressions start losing digits to cancellation.
    Dtype-aware so float32 (~3.4e-4) gets a usable guard band, not the
    float64 one.
    """
    eps: float = torch.finfo(dtype).eps
    return math.sqrt(eps)


def _exp_forward(o: torch.Tensor) -> torch.Tensor:
    """Tensor-level forward for octonion_exp (shared with OctonionExpFunction)."""
    a = o[..., 0:1]  # [..., 1]
    v = o[..., 1:]  # [..., 7]
    r = torch.sqrt(torch.sum(v**2, dim=-1, keepdim=True))  # [..., 1]

    exp_a = torch.exp(a)

    # sin(r)/r with the r -> 0 limit handled by series (1 - r^2/6)
    thresh = _series_threshold(o.dtype)
    near_zero = r < thresh
    safe_r = torch.where(near_zero, torch.ones_like(r), r)
    sinc = torch.where(near_zero, 1.0 - r**2 / 6.0, torch.sin(r) / safe_r)

    result_real = exp_a * torch.cos(r)
    result_imag = exp_a * sinc * v
    return torch.cat([result_real, result_imag], dim=-1)


def _log_forward(o: torch.Tensor) -> torch.Tensor:
    """Tensor-level forward for octonion_log (shared with OctonionLogFunction)."""
    a = o[..., 0:1]  # [..., 1]
    v = o[..., 1:]  # [..., 7]
    r = torch.sqrt(torch.sum(v**2, dim=-1, keepdim=True))  # [..., 1]
    q = torch.sqrt(a**2 + r**2)  # [..., 1]

    if bool(torch.any(q == 0)):
        raise ValueError("octonion_log requires non-zero norm (log of zero octonion)")

    log_q = torch.log(q)

    # theta = angle between o and the positive real axis. atan2 is accurate
    # in all quadrants (acos(a/q) loses half the significant digits for
    # small theta, which corrupted theta/r for small ||v||).
    theta = torch.atan2(r, a)

    # imag coefficient theta/r:
    #   - exact for r away from 0 (accurate in all quadrants via atan2)
    #   - series 1/a - r^2/(3 a^3) for r/q below threshold with a > 0
    #   - NaN at r == 0 with a < 0: log(-|a|) has no preferred imaginary
    #     direction in O (any unit imaginary works); the principal branch
    #     is genuinely undefined there.
    thresh = _series_threshold(o.dtype)
    series_mask = (r < thresh * q) & (a > 0)
    safe_r = torch.where(r == 0, torch.ones_like(r), r)
    safe_a = torch.where(a > 0, a, torch.ones_like(a))
    coeff = torch.where(
        series_mask,
        1.0 / safe_a - r**2 / (3.0 * safe_a**3),
        theta / safe_r,
    )
    undefined = (r == 0) & (a < 0)
    coeff = torch.where(undefined, torch.full_like(coeff, float("nan")), coeff)

    return torch.cat([log_q, coeff * v], dim=-1)


def octonion_exp(o: Octonion | torch.Tensor) -> Octonion | torch.Tensor:
    """Compute the exponential of an octonion.

    For octonion q = a + v where a is the real part and v is the imaginary vector:
      exp(q) = exp(a) * (cos(||v||) * e_0 + sin(||v||) * v / ||v||)

    with the smooth limit exp(a) * e_0 as ||v|| -> 0.

    Gradients flow through the analytic Jacobian (OctonionExpFunction), so
    differentiation is exact and NaN-free even at pure-real inputs where the
    naive sqrt(||v||^2) formulation has an undefined derivative.

    Args:
        o: Octonion instance or raw [..., 8] tensor.

    Returns:
        Same type as input: Octonion if given Octonion, raw tensor if given tensor.
    """
    from octonion.calculus._autograd_functions import OctonionExpFunction

    data = o.components if isinstance(o, Octonion) else o
    result: torch.Tensor = OctonionExpFunction.apply(data)  # type: ignore[no-untyped-call]
    return Octonion(result) if isinstance(o, Octonion) else result


def octonion_log(o: Octonion | torch.Tensor) -> Octonion | torch.Tensor:
    """Compute the principal logarithm of an octonion.

    For octonion q = a + v where a is the real part and v is the imaginary vector:
      log(q) = log(||q||) * e_0 + (theta / ||v||) * v,   theta = atan2(||v||, a)

    with the smooth limit log(a) * e_0 as ||v|| -> 0 for a > 0.

    Domain notes:
      - Raises ValueError if any element has zero norm.
      - On the negative real axis (a < 0, v = 0) the principal logarithm is
        undefined in O (every unit imaginary direction is equally valid, the
        analogue of log(-1) = i*pi having no preferred i); the imaginary
        components are NaN there. Arbitrarily close to the axis (tiny but
        non-zero v) the value is well-defined and computed accurately.

    Gradients flow through the analytic Jacobian (OctonionLogFunction), so
    differentiation is exact and NaN-free away from the singular set.

    Args:
        o: Octonion instance or raw [..., 8] tensor. Must have non-zero norm.

    Returns:
        Same type as input: Octonion if given Octonion, raw tensor if given tensor.
    """
    from octonion.calculus._autograd_functions import OctonionLogFunction

    data = o.components if isinstance(o, Octonion) else o
    result: torch.Tensor = OctonionLogFunction.apply(data)  # type: ignore[no-untyped-call]
    return Octonion(result) if isinstance(o, Octonion) else result


def commutator(a: Octonion, b: Octonion) -> Octonion:
    """Compute the commutator [a, b] = a*b - b*a.

    The commutator measures the failure of commutativity. Properties:
    - [a, a] = 0 for all a
    - [a, b] = -[b, a] (antisymmetry)
    - For real scalars, [s, a] = 0

    Args:
        a: Octonion instance.
        b: Octonion instance.

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
        a: Octonion instance.
        b: Octonion instance.

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
        a: Octonion instance.
        b: Octonion instance.

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
    result_data = torch.cat([torch.zeros_like(product[..., :1]), product[..., 1:]], dim=-1)
    return Octonion(result_data)
