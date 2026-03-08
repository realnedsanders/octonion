"""torch.autograd.Function subclasses for all 7 octonionic primitives.

Each Function provides a custom backward pass using the analytic Jacobian
formulas derived in _jacobians.py. All backward passes use differentiable
PyTorch operations (einsum, exp, cos, sin, etc.) so that create_graph=True
works automatically for double backward / Hessian computation.

Critical design patterns (per RESEARCH.md anti-patterns):
  - Only inputs are saved in ctx.save_for_backward() -- never intermediate
    computations from forward (which runs in no-grad mode and would break
    create_graph=True).
  - All backward operations use differentiable PyTorch ops for double backward.
  - STRUCTURE_CONSTANTS is fetched fresh in backward() with correct device/dtype
    (not cached from forward).

Convention: Baez 2002, mod-7 Fano plane basis.
"""

from __future__ import annotations

import torch

from octonion._multiplication import STRUCTURE_CONSTANTS, octonion_mul
from octonion._octonion import Octonion
from octonion._operations import (
    cross_product,
    inner_product,
    octonion_exp,
    octonion_log,
)


class OctonionMulFunction(torch.autograd.Function):
    """Differentiable octonion multiplication f(a, b) = a * b.

    For the product (a*b)_k = sum_{i,j} C[i,j,k] * a_i * b_j, the Jacobians are:
      J_a[k, i] = sum_j C[i,j,k] * b_j
      J_b[k, j] = sum_i C[i,j,k] * a_i

    Backward computes:
      grad_a[i] = sum_k grad_out[k] * J_a[k,i] = sum_k grad_out[k] * sum_j C[i,j,k] * b[j]
      grad_b[j] = sum_k grad_out[k] * J_b[k,j] = sum_k grad_out[k] * sum_i C[i,j,k] * a[i]

    The einsum operations are differentiable, so create_graph=True works.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(a, b)
        return octonion_mul(a, b)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = ctx.saved_tensors
        C = STRUCTURE_CONSTANTS.to(device=a.device, dtype=a.dtype)
        grad_a = torch.einsum("...k, ijk, ...j -> ...i", grad_output, C, b)
        grad_b = torch.einsum("...k, ijk, ...i -> ...j", grad_output, C, a)
        return grad_a, grad_b


class OctonionExpFunction(torch.autograd.Function):
    """Differentiable octonion exponential.

    exp(o) = exp(a) * (cos(||v||) + sin(||v||)/||v|| * v)

    where a = o[..., 0] is the scalar part and v = o[..., 1:] is the imaginary vector.

    Backward recomputes the Jacobian from saved input using differentiable ops.
    The Jacobian has block structure:
      Row 0:    [ea*cos(r),  -ea*sinc*v_1, ..., -ea*sinc*v_7]
      Rows 1-7: [ea*sinc*v_{k-1},  ea*(outer_coeff*v_{k-1}*v_i + sinc*delta_{k-1,i})]
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        o: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(o)
        return octonion_exp(o)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        (o,) = ctx.saved_tensors

        # Recompute Jacobian using differentiable operations
        a = o[..., 0:1]  # [..., 1]
        v = o[..., 1:]  # [..., 7]

        r_sq = torch.sum(v**2, dim=-1, keepdim=True)  # [..., 1]
        r = torch.sqrt(r_sq + 1e-30)  # [..., 1] (add tiny eps for sqrt grad stability)

        ea = torch.exp(a)  # [..., 1]

        eps = 1e-12
        near_zero = r < eps

        safe_r = torch.where(near_zero, torch.ones_like(r), r)
        safe_r_cu = safe_r**3

        cos_r = torch.cos(r)
        sin_r = torch.sin(r)

        sinc = torch.where(near_zero, torch.ones_like(r), sin_r / safe_r)
        outer_coeff = torch.where(
            near_zero,
            torch.full_like(r, -1.0 / 3.0),
            (cos_r * safe_r - sin_r) / safe_r_cu,
        )

        # Build J^T @ grad_output efficiently (without materializing full 8x8 Jacobian)
        # Split grad_output into real and imaginary parts
        g0 = grad_output[..., 0:1]  # [..., 1]
        gv = grad_output[..., 1:]  # [..., 7]

        # grad_input[0] = ea*cos(r)*g0 + ea*sinc * (v . gv)
        # where (v . gv) = sum_k v_k * gv_k
        v_dot_gv = torch.sum(v * gv, dim=-1, keepdim=True)  # [..., 1]

        grad_a = ea * cos_r * g0 + ea * sinc * v_dot_gv  # [..., 1]

        # grad_input[i] for i=1..7:
        # = -ea*sinc*v_{i-1}*g0 + ea*(outer_coeff * v_{i-1} * (v . gv) + sinc * gv_{i-1})
        # = ea * (-sinc * v * g0 + outer_coeff * v * v_dot_gv + sinc * gv)
        grad_v = ea * (-sinc * v * g0 + outer_coeff * v * v_dot_gv + sinc * gv)

        grad_input = torch.cat([grad_a, grad_v], dim=-1)
        return (grad_input,)


class OctonionLogFunction(torch.autograd.Function):
    """Differentiable octonion logarithm.

    log(o) = log(||o||) + arccos(a/||o||) / ||v|| * v

    Backward recomputes the Jacobian from saved input using differentiable ops.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        o: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(o)
        return octonion_log(o)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        (o,) = ctx.saved_tensors

        a = o[..., 0:1]
        v = o[..., 1:]

        r_sq = torch.sum(v**2, dim=-1, keepdim=True)
        r = torch.sqrt(r_sq + 1e-30)
        q_sq = a**2 + r_sq
        q = torch.sqrt(q_sq + 1e-30)

        eps = 1e-12
        near_zero_r = r < eps

        safe_r = torch.where(near_zero_r, torch.ones_like(r), r)
        safe_r_sq = safe_r**2
        safe_q_sq = q**2

        ratio = torch.clamp(a / q, min=-1.0, max=1.0)
        theta = torch.acos(ratio)

        theta_over_r = torch.where(
            near_zero_r,
            1.0 / q,
            theta / safe_r,
        )

        outer_coeff = torch.where(
            near_zero_r,
            -torch.ones_like(r) / (3.0 * q * safe_q_sq),
            (a / safe_q_sq - theta_over_r) / safe_r_sq,
        )

        # Compute J^T @ grad_output efficiently
        g0 = grad_output[..., 0:1]
        gv = grad_output[..., 1:]

        v_dot_gv = torch.sum(v * gv, dim=-1, keepdim=True)

        # grad_a = (a/q^2)*g0 + (-v/q^2) . gv  -- but gv is [..., 7] and v is [..., 7]
        # Actually: grad_a = J[0,0]*g0 + sum_{k=1..7} J[k,0]*gv_{k-1}
        # J[0,0] = a/q^2, J[k,0] = -v_{k-1}/q^2
        grad_a = (a / safe_q_sq) * g0 + (-1.0 / safe_q_sq) * v_dot_gv

        # grad_v_i = J[0, i+1]*g0 + sum_{k=1..7} J[k, i+1]*gv_{k-1}
        # J[0, i+1] = v_i/q^2
        # J[k, i+1] = v_{k-1}*v_i*outer_coeff + theta_over_r * delta_{k-1,i}
        # So: sum_k J[k,i+1]*gv_{k-1} = outer_coeff * v_i * (v.gv) + theta_over_r * gv_i
        grad_v = (v / safe_q_sq) * g0 + outer_coeff * v * v_dot_gv + theta_over_r * gv

        grad_input = torch.cat([grad_a, grad_v], dim=-1)
        return (grad_input,)


class OctonionConjugateFunction(torch.autograd.Function):
    """Differentiable octonion conjugation.

    conj(o) = [o_0, -o_1, ..., -o_7]

    Jacobian is the constant diagonal: diag([1, -1, -1, -1, -1, -1, -1, -1]).
    Backward is trivially: grad_input = grad_output * [1, -1, ..., -1].
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        o: torch.Tensor,
    ) -> torch.Tensor:
        result = o.clone()
        result[..., 1:] = -result[..., 1:]
        return result

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        # The Jacobian is diag([1, -1, -1, ..., -1]), so grad_input = grad_output * signs
        grad_input = grad_output.clone()
        grad_input[..., 1:] = -grad_input[..., 1:]
        return (grad_input,)


class OctonionInverseFunction(torch.autograd.Function):
    """Differentiable octonion inverse: f(o) = conj(o) / ||o||^2.

    Jacobian: J[k,i] = diag_conj[k,i] / n2 - 2 * conj_k * o_i / n2^2
    where n2 = ||o||^2 and diag_conj = diag([1, -1, ..., -1]).

    Backward recomputes from saved input using differentiable operations.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        o: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(o)
        return Octonion(o).inverse().components

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        (o,) = ctx.saved_tensors

        n2 = torch.sum(o**2, dim=-1, keepdim=True)  # [..., 1]

        # conj(o)
        conj = o.clone()
        conj[..., 1:] = -conj[..., 1:]

        # J[k,i] = diag_conj[k,i] / n2 - 2 * conj_k * o_i / n2^2
        # grad_input[i] = sum_k grad_output[k] * J[k,i]
        #               = sum_k grad_output[k] * diag_conj[k,i] / n2
        #                 - 2/n2^2 * sum_k grad_output[k] * conj_k * o_i

        # First term: sum_k grad_output[k] * diag_conj[k,i] / n2
        # diag_conj is diagonal, so sum_k grad_output[k] * diag_conj[k,i]
        # = grad_output[i] * diag_conj[i,i] (only k=i survives)
        signs = torch.ones(
            *o.shape[:-1], 8, dtype=o.dtype, device=o.device
        )
        signs[..., 1:] = -1.0
        term1 = signs * grad_output / n2

        # Second term: -2/n2^2 * (sum_k grad_output[k] * conj_k) * o_i
        go_dot_conj = torch.sum(grad_output * conj, dim=-1, keepdim=True)
        term2 = -2.0 * go_dot_conj * o / (n2**2)

        grad_input = term1 + term2
        return (grad_input,)


class OctonionInnerProductFunction(torch.autograd.Function):
    """Differentiable octonion inner product: <a, b> = sum_i a_i * b_i.

    This is a scalar-valued function. The gradient is trivial:
      grad_a = grad_output * b
      grad_b = grad_output * a
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(a, b)
        return torch.sum(a * b, dim=-1)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = ctx.saved_tensors
        # grad_output shape [...], need to broadcast to [..., 8]
        go = grad_output.unsqueeze(-1)
        grad_a = go * b
        grad_b = go * a
        return grad_a, grad_b


class OctonionCrossProductFunction(torch.autograd.Function):
    """Differentiable octonion cross product: cross(a, b) = Im(Im(a) * Im(b)).

    The cross product zeros out the real parts of both inputs, multiplies,
    then takes the imaginary part. The Jacobian is derived from the
    multiplication Jacobian restricted to the imaginary-to-imaginary block.

    Backward uses einsum with structure constants, which is differentiable
    for create_graph=True.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(a, b)
        # Compute cross product on raw tensors
        a_pure = torch.zeros_like(a)
        a_pure[..., 1:] = a[..., 1:]
        b_pure = torch.zeros_like(b)
        b_pure[..., 1:] = b[..., 1:]
        product = octonion_mul(a_pure, b_pure)
        result = torch.zeros_like(product)
        result[..., 1:] = product[..., 1:]
        return result

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = ctx.saved_tensors
        C = STRUCTURE_CONSTANTS.to(device=a.device, dtype=a.dtype)

        # grad_output has zero real part (output is pure imaginary), but
        # the grad_output from upstream may have arbitrary values.
        # The cross product Jacobian:
        # J_a[k, i] = 0 if k=0 or i=0
        # J_a[k, i] = sum_j C[i,j,k] * b_pure_j for k>0, i>0
        # (where b_pure has zero real part)

        # Construct pure imaginary versions
        b_pure = torch.zeros_like(b)
        b_pure[..., 1:] = b[..., 1:]
        a_pure = torch.zeros_like(a)
        a_pure[..., 1:] = a[..., 1:]

        # Zero out real component of grad_output since cross product output
        # has J[0, :] = 0 (real output is always 0). Without this, the einsum
        # over k would include k=0 contributions from C[i,j,0] which are non-zero.
        go = grad_output.clone()
        go[..., 0] = 0.0

        # grad_a[i] = sum_{k=1..7} go[k] * sum_{j} C[i,j,k] * b_pure[j]
        # For i=0: gives 0 because b_pure[0]=0 and C[0,j,k] with j>0 gives
        # contributions only for k matching the Fano triple, but we need to
        # ensure grad_a[0] = 0 since cross product doesn't depend on a[0].
        grad_a = torch.einsum("...k, ijk, ...j -> ...i", go, C, b_pure)
        grad_a = grad_a.clone()
        grad_a[..., 0] = 0.0

        # For grad_b: same pattern
        grad_b = torch.einsum("...k, ijk, ...i -> ...j", go, C, a_pure)
        grad_b = grad_b.clone()
        grad_b[..., 0] = 0.0

        return grad_a, grad_b
