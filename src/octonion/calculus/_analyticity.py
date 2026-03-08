"""Octonionic CR-like (Cauchy-Riemann) analyticity conditions.

In complex analysis, a function f(z) = u(x,y) + i*v(x,y) is holomorphic (analytic)
iff its 2x2 real Jacobian has the form [[a, -b], [b, a]], i.e., satisfies the
Cauchy-Riemann equations du/dx = dv/dy and du/dy = -dv/dx. Equivalently, the
Jacobian must be a left multiplication matrix for some complex number a + bi.

The octonionic analogue extends this to 8x8 real Jacobians. An octonionic function
f is "octonionic-analytic" (or "regular" in the sense of Fueter) if its Jacobian
at each point can be represented as left multiplication by some octonion c:

    J = L_c  where  L_c[k, j] = sum_i c[i] * C[i, j, k]

**Very few octonionic functions are analytic in this sense.** The octonionic
Cauchy-Riemann conditions are extremely restrictive:

- Left multiplication by a fixed octonion c: f(x) = c * x  -->  ANALYTIC (J = L_c)
- Identity: f(x) = x  -->  ANALYTIC (J = I = L_{e_0})
- Scalar multiplication: f(x) = alpha * x  -->  ANALYTIC (J = alpha*I = L_{alpha*e_0})
- Right multiplication by c: f(x) = x * c  -->  NOT analytic (J = R_c != L_c in general)
- Conjugation: f(x) = x*  -->  NOT analytic (J = diag(1,-1,...,-1) != any L_c)
- exp, log:  -->  NOT analytic (complex Jacobian structure)

This is a known feature of octonionic analysis and contrasts with complex analysis
where many common functions (polynomials, exp, trig) are analytic. The non-associativity
of octonions means that the notion of "regular function" is far more constrained.

References:
    - Baez (2002), "The Octonions"
    - Gentili & Struppa (2007), "A new theory of regular functions of a quaternionic variable"
    - This implementation: Checks whether J = L_c for the putative c = J[:, 0].

Convention: Baez 2002, mod-7 Fano plane basis.
"""

from __future__ import annotations

from typing import Callable

import torch

from octonion._multiplication import STRUCTURE_CONSTANTS
from octonion.calculus._numeric import numeric_jacobian


def cauchy_riemann_octonion(J: torch.Tensor) -> torch.Tensor:
    """Compute the CR residual for an 8x8 real Jacobian.

    An octonionic function f is "analytic" if its Jacobian J equals L_c for
    some octonion c. The algorithm:
    1. Extract putative c from the first column: c = J[:, 0] (since L_c[:, 0] = c * e_0 = c)
    2. Reconstruct L_c from c using the structure constants
    3. Return ||J - L_c||_F (Frobenius norm of the residual)

    Args:
        J: Real Jacobian tensor of shape [..., 8, 8].

    Returns:
        Scalar (or batched) tensor: the Frobenius norm of J - L_c, where c = J[:, 0].
        A value near zero indicates the function is octonionic-analytic at the
        evaluated point.
    """
    C = STRUCTURE_CONSTANTS.to(device=J.device, dtype=J.dtype)

    # Extract putative c from first column: c[k] = J[..., k, 0]
    c = J[..., :, 0]  # [..., 8]

    # Reconstruct L_c: L_c[k, j] = sum_i c[i] * C[i, j, k]
    L_c = torch.einsum("...i, ijk -> ...kj", c, C)

    # Residual: Frobenius norm of J - L_c
    diff = J - L_c
    # Frobenius norm: sqrt(sum of squared elements) over last two dims
    residual = torch.sqrt(torch.sum(diff**2, dim=(-2, -1)))

    return residual


def analyticity_residual(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute the CR residual for function fn at point x.

    Computes the numeric Jacobian of fn at x and checks whether it satisfies
    the octonionic Cauchy-Riemann condition (i.e., whether J = L_c for some c).

    Args:
        fn: A function mapping tensors of shape [..., 8] to [..., 8].
        x: Input tensor of shape [..., 8] at which to evaluate.
        eps: Finite-difference step size for numeric Jacobian.

    Returns:
        Scalar tensor: the CR residual (Frobenius norm of J - L_c).
    """
    J = numeric_jacobian(fn, x, eps=eps)
    return cauchy_riemann_octonion(J)


def is_octonionic_analytic(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    tol: float = 1e-6,
    eps: float = 1e-7,
) -> bool:
    """Test whether fn is octonionic-analytic at point x.

    Returns True if the CR residual (||J - L_c||_F) is below the tolerance.

    Args:
        fn: A function mapping tensors of shape [..., 8] to [..., 8].
        x: Input tensor of shape [..., 8] at which to evaluate.
        tol: Tolerance for the CR residual.
        eps: Finite-difference step size for numeric Jacobian.

    Returns:
        True if the function appears octonionic-analytic at x, False otherwise.
    """
    residual = analyticity_residual(fn, x, eps=eps)
    return bool(residual.item() < tol)
