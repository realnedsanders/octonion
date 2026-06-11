"""Custom octonion-aware gradient checking utilities.

Unlike torch.autograd.gradcheck, these utilities:
1. Report per-component octonionic errors (e0, e1, ..., e7) for each input
2. Validate the GHR decomposition (the 8 involution-basis derivatives from
   ghr_derivatives_from_jacobian in _ghr.py) and its round-trip
3. Return structured dicts with detailed error information

The checking approach:
  - Compute numeric 8x8 Jacobian via central differences
  - Compute autograd 8x8 Jacobian by backpropagating unit vectors
  - Compare element-wise for per-component reporting
  - Decompose both into GHR derivatives, compare, and verify the
    reconstruction round-trips to the original Jacobian

Convention: Baez 2002, mod-7 Fano plane basis.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.autograd.gradcheck import GradcheckError

from octonion.calculus._ghr import ghr_derivatives_from_jacobian, reconstruct_jacobian
from octonion.calculus._numeric import numeric_jacobian


def _autograd_jacobian(
    fn: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    input_idx: int,
) -> torch.Tensor:
    """Compute the Jacobian of fn w.r.t. inputs[input_idx] via autograd.

    For each output component k, backpropagates a unit vector e_k to get
    the k-th row of the Jacobian: J[k, :] = backward(e_k).

    Args:
        fn: Function mapping input tensors to output tensor.
        inputs: Tuple of input tensors.
        input_idx: Index of input to differentiate w.r.t.

    Returns:
        Jacobian tensor of shape [m, n] where m is output dim and n is
        input dim of the specified input.
    """
    inp = inputs[input_idx]
    n = inp.shape[-1]

    # Run forward to get output shape
    output = fn(*inputs)
    # Scalar output counts as a single component
    m = 1 if output.dim() == 0 else output.shape[-1]

    J = torch.zeros(m, n, dtype=inp.dtype, device=inp.device)

    for k in range(m):
        # Zero gradients
        for t in inputs:
            if t.requires_grad and t.grad is not None:
                t.grad.zero_()

        # Forward pass
        output = fn(*inputs)

        if output.dim() == 0:
            # Scalar output: backward directly
            output.backward(retain_graph=True)  # type: ignore[no-untyped-call]
        else:
            # Create unit vector for output component k
            grad_output = torch.zeros_like(output)
            grad_output[..., k] = 1.0
            output.backward(grad_output, retain_graph=True)  # type: ignore[no-untyped-call]

        if inp.grad is not None:
            J[k, :] = inp.grad.clone()

    return J


def _numeric_jacobian_for_gradcheck(
    fn: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    input_idx: int,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute numeric Jacobian of fn w.r.t. inputs[input_idx].

    This wraps the function to work with the numeric_jacobian utility,
    handling multi-argument functions by fixing all other arguments.

    Args:
        fn: Function mapping input tensors to output tensor.
        inputs: Tuple of input tensors.
        input_idx: Index of input to differentiate w.r.t.
        eps: Finite-difference step size.

    Returns:
        Jacobian tensor of shape [m, n].
    """
    inp = inputs[input_idx].detach()

    def single_arg_fn(x: torch.Tensor) -> torch.Tensor:
        modified_inputs = list(inputs)
        modified_inputs[input_idx] = x
        result = fn(*modified_inputs)
        if result.dim() == 0:
            return result.unsqueeze(-1)  # [1] for scalar output
        return result

    return numeric_jacobian(single_arg_fn, inp, eps=eps)


def octonion_gradcheck(
    fn: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...] | torch.Tensor,
    eps: float = 1e-7,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> dict[str, Any]:
    """Octonion-aware gradient check with per-component error reporting.

    Unlike torch.autograd.gradcheck, this:
    1. Reports per-component octonionic errors (e0..e7) for each input
    2. Validates the GHR involution-basis decomposition of the Jacobian
       (autograd vs numeric, plus reconstruction round-trip)
    3. Returns a structured dict with detailed error information

    Args:
        fn: Function to check. Takes tensors, returns tensor.
        inputs: Single tensor or tuple of input tensors. Those with
            requires_grad=True will be checked.
        eps: Finite-difference step size for numeric Jacobian.
        atol: Absolute tolerance for comparison.
        rtol: Relative tolerance for comparison.

    Returns:
        dict with keys:
            - "passed" (bool): Whether all checks passed
            - "max_abs_error" (float): Maximum absolute error across all inputs
            - "max_rel_error" (float): Maximum relative error across all inputs
            - "per_component_errors" (list[list[float]]): Per-input, per-component
              max absolute errors
            - "ghr_passed" (bool): Whether GHR decomposition validation passed
            - "ghr_error" (float): Max error in GHR decomposition validation
    """
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)

    all_passed = True
    max_abs_error = 0.0
    max_rel_error = 0.0
    per_component_errors: list[list[float]] = []
    ghr_passed = True
    ghr_error = 0.0

    for idx, inp in enumerate(inputs):
        if not inp.requires_grad:
            continue

        # Compute autograd Jacobian
        J_autograd = _autograd_jacobian(fn, inputs, idx)

        # Compute numeric Jacobian
        J_numeric = _numeric_jacobian_for_gradcheck(fn, inputs, idx, eps=eps)

        # Element-wise comparison
        abs_err = torch.abs(J_autograd - J_numeric)
        # Relative error: |a - n| / (|n| + eps_denom) to avoid division by zero
        eps_denom = 1e-15
        rel_err = abs_err / (torch.abs(J_numeric) + eps_denom)

        max_abs = abs_err.max().item()
        max_rel = rel_err.max().item()

        max_abs_error = max(max_abs_error, max_abs)
        max_rel_error = max(max_rel_error, max_rel)

        # Per-component errors (max over output rows for each input column)
        n = inp.shape[-1]
        comp_errors = []
        for i in range(n):
            comp_errors.append(abs_err[:, i].max().item())
        per_component_errors.append(comp_errors)

        # Check pass/fail using allclose logic
        close = torch.abs(J_autograd - J_numeric) <= atol + rtol * torch.abs(J_numeric)
        if not close.all():
            all_passed = False

        # GHR derivative validation (only for 8x8 Jacobians): the autograd
        # and numeric Jacobians must yield matching GHR decompositions, and
        # the decomposition must round-trip back to the Jacobian exactly
        # (df = sum_a (df/do^{sigma_a}) do^{sigma_a} — see _ghr.py).
        m = J_autograd.shape[0]
        if m == 8 and n == 8:
            ghr_auto = ghr_derivatives_from_jacobian(J_autograd)
            ghr_num = ghr_derivatives_from_jacobian(J_numeric)

            w_err = torch.abs(ghr_auto - ghr_num).max().item()
            roundtrip_err = torch.abs(reconstruct_jacobian(ghr_auto) - J_autograd).max().item()
            ghr_error = max(ghr_error, w_err, roundtrip_err)

            ghr_close = torch.allclose(ghr_auto, ghr_num, atol=atol, rtol=rtol)
            roundtrip_close = torch.allclose(
                reconstruct_jacobian(ghr_auto), J_autograd, atol=atol, rtol=rtol
            )
            if not (ghr_close and roundtrip_close):
                ghr_passed = False

    return {
        "passed": all_passed,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "per_component_errors": per_component_errors,
        "ghr_passed": ghr_passed,
        "ghr_error": ghr_error,
    }


def octonion_gradgradcheck(
    fn: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...] | torch.Tensor,
    eps: float = 1e-7,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> dict[str, Any]:
    """Verify second-order derivatives via torch.autograd.gradgradcheck.

    Wraps torch.autograd.gradgradcheck with octonionic error reporting.
    Verifies that create_graph=True produces correct second-order gradients
    by comparing first backward gradients against finite differences of
    the backward pass.

    Args:
        fn: Function to check.
        inputs: Single tensor or tuple of input tensors.
        eps: Finite-difference step size.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        dict with keys:
        - "passed" (bool): Whether gradgradcheck passed
    """
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)

    try:
        passed = torch.autograd.gradgradcheck(fn, inputs, eps=eps, atol=atol, rtol=rtol)
        return {"passed": bool(passed)}
    except GradcheckError:
        # Genuine gradient mismatch -> failed check. Anything else (device
        # mismatch, shape bugs, ...) propagates instead of masquerading as
        # a gradient failure.
        return {"passed": False}
