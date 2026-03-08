"""GHR (Generalized Hamilton-Real) Wirtinger derivative formalism for octonions.

This module extends the quaternionic GHR calculus (Xu & Mandic 2015) to the
octonionic domain, providing the Wirtinger-like derivative pair (df/do, df/do*)
that forms the mathematical foundation for correct octonionic backpropagation.

Mathematical Foundation
=======================

**Quaternionic GHR (reference):**
For a quaternion q = q_a + q_b*i + q_c*j + q_d*k and rotation frame mu,
the GHR derivatives are defined as:

    df/dq_mu  = (1/4)(df/dq_a - df/dq_b * i_mu - df/dq_c * j_mu - df/dq_d * k_mu)
    df/dq_mu* = (1/4)(df/dq_a + df/dq_b * i_mu + df/dq_c * j_mu + df/dq_d * k_mu)

where (i_mu, j_mu, k_mu) is a rotated imaginary basis. The key property is that
for real-valued loss L, the optimization gradient is dL/dq* (conjugate derivative),
and left and right derivatives coincide (Proposition 4.8 of Xu & Mandic 2015).

**Octonionic Extension:**
For an octonion o = o_0 + sum_{k=1}^{7} o_k * e_k, the natural extension uses
8 partial derivatives (1 real + 7 imaginary):

    df/do  = (1/8)(df/do_0 - sum_{k=1}^{7} df/do_k * e_k)
    df/do* = (1/8)(df/do_0 + sum_{k=1}^{7} df/do_k * e_k)

The factor 1/8 ensures the fundamental identity:
    df = (df/do) * do + (df/do*) * do*

holds under the octonionic inner product convention.

**Rotation well-definedness:**
The quaternionic GHR framework relies on the rotation q_mu = mu * q * mu^{-1},
which requires associativity. For octonions, the Moufang identity guarantees
that (mu * q) * mu^{-1} = mu * (q * mu^{-1}) when mu appears twice, because
alternative algebras satisfy the Moufang loop property. This makes the octonionic
rotation R_mu(q) = mu * q * mu^{-1} well-defined despite non-associativity.

**Practical consequence for optimization:**
For a real-valued loss function L and octonionic variable o, the gradient used
for optimization is:

    grad_o = dL/do* = (1/8)(dL/do_0 + sum_{k=1}^{7} dL/do_k * e_k)

Since dL/do_k are real scalars (L is real), this simplifies to the R^8 gradient
vector reinterpreted as an octonion (up to 1/8 scaling). The GHR formalism adds
value by providing the derivative in octonionic form and enabling product and
chain rules that respect octonionic structure for non-real-valued intermediates.

Convention: Baez 2002, mod-7 Fano plane basis.
"""

from __future__ import annotations

from typing import Tuple

import torch


def ghr_derivative(f_real_partials: torch.Tensor) -> torch.Tensor:
    """Compute the GHR "full" Wirtinger derivative df/do.

    Given the 8 real partial derivatives (df/do_0, df/do_1, ..., df/do_7),
    computes the GHR derivative:

        df/do = (1/8)(df/do_0 - sum_{k=1}^{7} df/do_k * e_k)

    This produces an octonionic-valued derivative where the real part is
    (1/8)*df/do_0 and the k-th imaginary part is -(1/8)*df/do_k.

    Args:
        f_real_partials: Tensor of shape [..., 8] containing the 8 real
            partial derivatives (df/do_0, df/do_1, ..., df/do_7).

    Returns:
        Tensor of shape [..., 8] representing the GHR derivative as an
        octonion: components [df/do_0, -df/do_1, ..., -df/do_7] / 8.
    """
    result = f_real_partials.clone()
    result[..., 1:] = -result[..., 1:]
    return result / 8.0


def conjugate_derivative(f_real_partials: torch.Tensor) -> torch.Tensor:
    """Compute the GHR conjugate Wirtinger derivative df/do*.

    Given the 8 real partial derivatives (df/do_0, df/do_1, ..., df/do_7),
    computes the conjugate GHR derivative:

        df/do* = (1/8)(df/do_0 + sum_{k=1}^{7} df/do_k * e_k)

    For a real-valued loss function L, the gradient direction for optimization
    is dL/do* = (1/8) * [dL/do_0, dL/do_1, ..., dL/do_7], which is simply
    the R^8 gradient scaled by 1/8.

    Args:
        f_real_partials: Tensor of shape [..., 8] containing the 8 real
            partial derivatives (df/do_0, df/do_1, ..., df/do_7).

    Returns:
        Tensor of shape [..., 8] representing the conjugate GHR derivative
        as an octonion: f_real_partials / 8.
    """
    return f_real_partials / 8.0


def wirtinger_from_jacobian(
    J: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert an 8x8 real Jacobian to the (df/do, df/do*) Wirtinger pair.

    Given the real Jacobian J[k, i] = d f(o)_k / d o_i, this function
    computes the Wirtinger derivative pair for each output component.

    The Wirtinger pair encodes how each output component changes with respect
    to the input octonion and its conjugate:

        (df/do)_k  = (1/8)(J[k, 0] - sum_{m=1}^{7} J[k, m] * e_m)
        (df/do*)_k = (1/8)(J[k, 0] + sum_{m=1}^{7} J[k, m] * e_m)

    Each output row of the Jacobian is treated as the real partial derivatives
    of the k-th output component, and converted to Wirtinger form.

    Args:
        J: Real Jacobian tensor of shape [..., 8, 8] where J[..., k, i] is
            the partial derivative of the k-th output w.r.t. the i-th input.

    Returns:
        Tuple (df_do, df_do_conj) where each has shape [..., 8, 8]:
        - df_do[..., k, :] is the GHR derivative of output component k
        - df_do_conj[..., k, :] is the conjugate derivative of output component k

    Example:
        >>> J = jacobian_mul(a, b)[0]  # 8x8 Jacobian wrt a
        >>> df_do, df_do_star = wirtinger_from_jacobian(J)
        >>> # df_do_star gives the gradient contribution from each output component
    """
    # df/do: negate imaginary columns, scale by 1/8
    df_do = J.clone()
    df_do[..., :, 1:] = -df_do[..., :, 1:]
    df_do = df_do / 8.0

    # df/do*: keep all columns as-is, scale by 1/8
    df_do_conj = J / 8.0

    return df_do, df_do_conj
