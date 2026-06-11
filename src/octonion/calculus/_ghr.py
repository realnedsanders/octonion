"""GHR (Generalized Hamilton-Real) calculus for octonions.

This module extends the quaternionic GHR calculus (Xu & Mandic 2015) to the
octonionic domain. Quaternionic GHR expresses the differential of a function
f: H -> H exactly as a sum over the four involutions q^mu = mu q mu^{-1},
mu in {1, i, j, k}:

    df = sum_mu (df/dq^mu) dq^mu

The octonionic analogue developed here uses the eight conjugation maps

    sigma_a(x) = e_a x e_a^{-1},   a = 0..7   (sigma_0 = identity)

which are well-defined despite non-associativity (any two elements generate
an associative subalgebra, so e_a x e_a is unambiguous by Artin's theorem).
On components, sigma_a fixes x_0 and x_a and negates the other six imaginary
components, so each sigma_a is the diagonal sign matrix

    D_a = diag(s_a),  s_a[t] = +1 if t in {0, a} else -1,  D_0 = I.

**The fundamental identity.** For any R-differentiable f: O -> O with real
8x8 Jacobian J (so df = J do as R^8 vectors), there exist UNIQUE octonions
A_0, ..., A_7 — the GHR derivatives df/do^{sigma_a} — such that

    df = sum_{a=0}^{7} (df/do^{sigma_a}) * do^{sigma_a}        (exact)

where * is octonion multiplication and do^{sigma_a} = sigma_a(do).

*Proof sketch.* Writing left multiplication by A as the matrix L_A, the
identity reads J = sum_a L_{A_a} D_a. Applying both sides to basis vector
e_t and using right-alternativity to cancel e_t^{-1}:

    sum_a S[a, t] A_a = (J e_t) * e_t^{-1} =: B_t,
    S[a, t] = s_a[t]  (the 8x8 sign matrix above).

S is symmetric and invertible with

    S^{-1} = [[5/12, (1/12) 1^T], [(1/12) 1, (1/2) I - (1/12) J_7]]

(J_7 the all-ones 7x7 block), so A = S^{-1} B exists and is unique. Sanity
checks: f(o) = o gives A = (1, 0, ..., 0); f(o) = o* gives A_0 = -1/6,
A_a = 1/6, reproducing the classical identity
o* = (-o + sum_a sigma_a(o)) / 6.

**Differences from the quaternionic case (important):**

- In H the sign matrix is a Hadamard matrix (S^{-1} = S/4), which makes all
  GHR coefficients a uniform 1/4. In O no choice of conjugation maps yields
  a Hadamard family, so the coefficients are intrinsically asymmetric
  (5/12 vs 1/12). A two-term Wirtinger-style decomposition
  df = (df/do) do + (df/do*) do* does NOT exist for octonions.
- In H the maps q -> mu q mu^{-1} are algebra automorphisms; in O the
  sigma_a are NOT multiplicative (sigma_a(xy) != sigma_a(x) sigma_a(y) in
  general). They are real-linear involutions commuting with conjugation,
  which is all the identity requires.
- For a real-valued loss L the quaternionic conjugate derivative dL/dq* is
  parallel to the R^4 gradient; the octonionic A*_0 is NOT parallel to the
  R^8 gradient (it weights the real component 5x). Steepest descent should
  use the plain R^8 gradient; the GHR derivatives provide the structured
  first-order model, not a rescaled gradient.

Convention: Baez 2002, mod-7 Fano plane basis.
"""

from __future__ import annotations

import torch

from octonion._multiplication import structure_constants

# Sign table S[a, t] of the involution sigma_a acting on component t:
# sigma_a fixes components 0 and a, negates the rest. Row 0 is the identity.
_S = -torch.ones(8, 8, dtype=torch.float64)
_S[0, :] = 1.0
_S[:, 0] = 1.0
_S.fill_diagonal_(1.0)
INVOLUTION_SIGNS = _S

# Closed-form inverse of S: [[5/12, 1/12 ...], [1/12 ..., 1/2 I - 1/12 J_7]].
# Entrywise: 5/12 on the diagonal, 1/12 in row/column 0, -1/12 elsewhere.
_S_INV = -torch.full((8, 8), 1.0 / 12.0, dtype=torch.float64)
_S_INV[0, :] = 1.0 / 12.0
_S_INV[:, 0] = 1.0 / 12.0
_S_INV.fill_diagonal_(5.0 / 12.0)


def involute(o: torch.Tensor, a: int) -> torch.Tensor:
    """Apply the involution sigma_a(o) = e_a o e_a^{-1}.

    sigma_0 is the identity. For a >= 1, sigma_a fixes components 0 and a
    and negates the remaining six imaginary components.

    Args:
        o: Octonion tensor [..., 8].
        a: Involution index in 0..7.

    Returns:
        Tensor [..., 8] with sigma_a applied along the last dimension.
    """
    if not 0 <= a <= 7:
        raise ValueError(f"Involution index must be in 0..7, got {a}")
    signs = INVOLUTION_SIGNS[a].to(device=o.device, dtype=o.dtype)
    return o * signs


def ghr_derivatives_from_jacobian(J: torch.Tensor) -> torch.Tensor:
    """Compute the 8 octonionic GHR derivatives from a real 8x8 Jacobian.

    Returns A with A[..., a, :] = df/do^{sigma_a}, the unique octonions
    satisfying the exact differential identity

        df = sum_{a=0}^{7} A_a * sigma_a(do)

    equivalently J = sum_a L_{A_a} D_a (left-multiplication matrices times
    involution sign matrices). Use :func:`reconstruct_jacobian` to map back.

    Args:
        J: Real Jacobian tensor [..., 8, 8] with J[..., k, i] = d f_k / d o_i.

    Returns:
        Tensor [..., 8, 8]: stack of the 8 GHR derivatives along dim -2.
    """
    C = structure_constants(J.device, J.dtype)
    # B_t = (column t of J, read as an octonion over the output index) * e_t^{-1}.
    # (c * e_t)_k = sum_i C[i, t, k] c_i; then scale by the sign of e_t^{-1}
    # (e_0^{-1} = e_0; e_t^{-1} = -e_t for t >= 1).
    B = torch.einsum("itk, ...it -> ...tk", C, J)
    inv_signs = torch.tensor([1.0] + [-1.0] * 7, dtype=J.dtype, device=J.device)
    B = B * inv_signs.unsqueeze(-1)
    # A_a = sum_t S^{-1}[a, t] B_t
    S_inv = _S_INV.to(device=J.device, dtype=J.dtype)
    return torch.einsum("at, ...tk -> ...ak", S_inv, B)


def ghr_conjugate_derivatives_from_jacobian(J: torch.Tensor) -> torch.Tensor:
    """Compute the 8 conjugate GHR derivatives from a real 8x8 Jacobian.

    Returns A* with A*[..., a, :] = df/d(o^{sigma_a})*, the unique octonions
    satisfying

        df = sum_{a=0}^{7} A*_a * sigma_a(do*)

    (conjugation commutes with every sigma_a, so sigma_a(do*) = sigma_a(do)*).

    Args:
        J: Real Jacobian tensor [..., 8, 8].

    Returns:
        Tensor [..., 8, 8]: stack of the 8 conjugate GHR derivatives.
    """
    # Composing with conjugation flips the sign of every imaginary column
    # equation: B'_0 = B_0, B'_t = -B_t for t >= 1.
    C = structure_constants(J.device, J.dtype)
    B = torch.einsum("itk, ...it -> ...tk", C, J)
    # inv_signs (for e_t^{-1}) and the conjugation signs are both
    # [1, -1, ..., -1] over t, so they cancel: B' = B with no t-dependent sign.
    S_inv = _S_INV.to(device=J.device, dtype=J.dtype)
    return torch.einsum("at, ...tk -> ...ak", S_inv, B)


def reconstruct_jacobian(A: torch.Tensor, conjugate: bool = False) -> torch.Tensor:
    """Reconstruct the real 8x8 Jacobian from GHR derivatives.

    Inverse of :func:`ghr_derivatives_from_jacobian` (or of
    :func:`ghr_conjugate_derivatives_from_jacobian` with ``conjugate=True``):

        J[..., k, i] = sum_a S[a, i'] * L_{A_a}[k, i]

    where L_A[k, i] = sum_j C[j, i, k] A_j is the left-multiplication matrix.

    Args:
        A: GHR derivative stack [..., 8, 8] (involution index on dim -2).
        conjugate: If True, A holds conjugate GHR derivatives.

    Returns:
        Real Jacobian tensor [..., 8, 8].
    """
    C = structure_constants(A.device, A.dtype)
    S = INVOLUTION_SIGNS.to(device=A.device, dtype=A.dtype)
    if conjugate:
        conj_signs = torch.tensor([1.0] + [-1.0] * 7, dtype=A.dtype, device=A.device)
        S = S * conj_signs.unsqueeze(0)
    # J[k, i] = sum_a S[a, i] * sum_j C[j, i, k] * A[a, j]
    return torch.einsum("ai, jik, ...aj -> ...ki", S, C, A)


def ghr_derivative(f_real_partials: torch.Tensor) -> torch.Tensor:
    """GHR derivative df/do of a REAL-VALUED function from its R^8 gradient.

    For real-valued f with gradient g = (df/do_0, ..., df/do_7), the leading
    GHR derivative (the a = 0 term of the decomposition) is

        df/do = (1/12) * (5 g_0 - sum_{k=1}^{7} g_k e_k)

    Note the coefficients (5/12, -1/12) — not the uniform 1/4 of the
    quaternionic case (see module docstring). The full differential also
    involves the other seven GHR derivatives; df/do alone does not determine
    df. For steepest descent, use the plain R^8 gradient g.

    Args:
        f_real_partials: Tensor [..., 8] of real partial derivatives.

    Returns:
        Tensor [..., 8]: df/do as an octonion.
    """
    result = f_real_partials.clone()
    result[..., 0] = 5.0 * result[..., 0]
    result[..., 1:] = -result[..., 1:]
    return result / 12.0


def conjugate_derivative(f_real_partials: torch.Tensor) -> torch.Tensor:
    """Conjugate GHR derivative df/do* of a REAL-VALUED function.

    For real-valued f with gradient g:

        df/do* = (1/12) * (5 g_0 + sum_{k=1}^{7} g_k e_k)

    Unlike the quaternionic case, this is NOT parallel to the R^8 gradient
    (the real component is weighted 5x), so it is not a steepest-descent
    direction. Use the plain R^8 gradient for optimization.

    Args:
        f_real_partials: Tensor [..., 8] of real partial derivatives.

    Returns:
        Tensor [..., 8]: df/do* as an octonion.
    """
    result = f_real_partials.clone()
    result[..., 0] = 5.0 * result[..., 0]
    return result / 12.0
