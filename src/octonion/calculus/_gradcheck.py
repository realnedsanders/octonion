"""Custom octonion-aware gradient checking utilities.

Unlike torch.autograd.gradcheck, these utilities:
1. Report per-component octonionic errors (e0, e1, ..., e7) for each input
2. Validate both Wirtinger derivatives (df/do and df/do*) using
   wirtinger_from_jacobian from _ghr.py
3. Return structured dicts with detailed error information

The checking approach:
  - Compute numeric 8x8 Jacobian via central differences
  - Compute autograd 8x8 Jacobian by backpropagating unit vectors
  - Compare element-wise for per-component reporting
  - Convert both to Wirtinger pairs and compare

Convention: Baez 2002, mod-7 Fano plane basis.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from octonion.calculus._ghr import wirtinger_from_jacobian
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
    if output.dim() == 0:
        # Scalar output
        m = 1
    else:
        m = output.shape[-1]

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
    2. Validates both Wirtinger derivatives (df/do and df/do*) by converting
       the Jacobian to Wirtinger form and comparing
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
            - "wirtinger_passed" (bool): Whether Wirtinger pair comparison passed
            - "wirtinger_error" (float): Max error in Wirtinger pair comparison
    """
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)

    all_passed = True
    max_abs_error = 0.0
    max_rel_error = 0.0
    per_component_errors: list[list[float]] = []
    wirtinger_passed = True
    wirtinger_error = 0.0

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
        close = torch.abs(J_autograd - J_numeric) <= atol + rtol * torch.abs(
            J_numeric
        )
        if not close.all():
            all_passed = False

        # Wirtinger derivative validation (only for 8x8 Jacobians)
        m = J_autograd.shape[0]
        if m == 8 and n == 8:
            w_auto_do, w_auto_dostar = wirtinger_from_jacobian(
                J_autograd.unsqueeze(0)
            )
            w_num_do, w_num_dostar = wirtinger_from_jacobian(
                J_numeric.unsqueeze(0)
            )

            w_err_do = torch.abs(w_auto_do - w_num_do).max().item()
            w_err_dostar = torch.abs(w_auto_dostar - w_num_dostar).max().item()
            w_err = max(w_err_do, w_err_dostar)
            wirtinger_error = max(wirtinger_error, w_err)

            w_close_do = torch.allclose(
                w_auto_do, w_num_do, atol=atol, rtol=rtol
            )
            w_close_dostar = torch.allclose(
                w_auto_dostar, w_num_dostar, atol=atol, rtol=rtol
            )
            if not (w_close_do and w_close_dostar):
                wirtinger_passed = False

    return {
        "passed": all_passed,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "per_component_errors": per_component_errors,
        "wirtinger_passed": wirtinger_passed,
        "wirtinger_error": wirtinger_error,
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
        passed = torch.autograd.gradgradcheck(
            fn, inputs, eps=eps, atol=atol, rtol=rtol
        )
        return {"passed": bool(passed)}
    except Exception:
        return {"passed": False}
