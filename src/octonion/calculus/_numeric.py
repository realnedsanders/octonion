"""Finite-difference numeric Jacobian utilities.

Provides central-difference approximations of Jacobian matrices for
octonionic functions. Used as the ground-truth reference for verifying
analytic Jacobian derivations and autograd backward passes.

Epsilon selection rationale:
  Central difference error is O(eps^2) + O(u/eps) where u is machine
  epsilon (~2.2e-16 for float64). The optimal eps is (u)^{1/3} ~ 6e-6.
  We default to eps=1e-7 which gives O(1e-14) truncation error, well
  within the 1e-10 tolerance used in tests and far inside the 1e-5
  success criterion budget.

Convention: Baez 2002, mod-7 Fano plane basis.
"""

from __future__ import annotations

from typing import Callable

import torch


def numeric_jacobian(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute the Jacobian of fn at x via central finite differences.

    Computes J[..., k, i] = d f(x)_k / d x_i using the central-difference
    formula: (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps) for each input
    component i.

    All inputs are detached from the autograd graph (operates on pure
    values, not graph nodes).

    Args:
        fn: A function mapping tensors of shape [..., n] to [..., m].
        x: Input tensor of shape [..., n]. Typically n=8 for octonionic
            operations.
        eps: Finite-difference step size. Default 1e-7 is optimal for
            float64 central differences.

    Returns:
        Jacobian tensor of shape [..., m, n] where m is the output
        dimension and n is the input dimension.
    """
    x = x.detach()
    n = x.shape[-1]
    f0 = fn(x)
    m = f0.shape[-1]

    J = torch.zeros(*x.shape[:-1], m, n, dtype=x.dtype, device=x.device)

    for i in range(n):
        e = torch.zeros_like(x)
        e[..., i] = eps
        f_plus = fn(x + e)
        f_minus = fn(x - e)
        J[..., :, i] = (f_plus - f_minus) / (2.0 * eps)

    return J


def numeric_jacobian_2arg(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
    wrt: str,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute the Jacobian of a two-argument function via central differences.

    For a function f(a, b), computes the Jacobian with respect to the
    argument specified by `wrt`:
      - wrt="a": J[..., k, i] = d f(a, b)_k / d a_i
      - wrt="b": J[..., k, j] = d f(a, b)_k / d b_j

    All inputs are detached from the autograd graph.

    Args:
        fn: A function mapping (tensor [..., n1], tensor [..., n2]) to [..., m].
        a: First input tensor of shape [..., n1].
        b: Second input tensor of shape [..., n2].
        wrt: Which argument to differentiate with respect to. Must be "a" or "b".
        eps: Finite-difference step size.

    Returns:
        Jacobian tensor of shape [..., m, n] where n is the dimension of the
        differentiation variable.

    Raises:
        ValueError: If `wrt` is not "a" or "b".
    """
    if wrt not in ("a", "b"):
        raise ValueError(f"wrt must be 'a' or 'b', got '{wrt}'")

    a = a.detach()
    b = b.detach()

    if wrt == "a":
        return numeric_jacobian(lambda x: fn(x, b), a, eps=eps)
    else:
        return numeric_jacobian(lambda x: fn(a, x), b, eps=eps)
