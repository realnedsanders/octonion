"""Hessian eigenspectrum analysis: full Hessian and stochastic Lanczos.

Provides tools for characterizing the loss landscape geometry around
a model's current parameters:
- compute_full_hessian: Exact Hessian via torch.autograd.functional.hessian
- stochastic_lanczos: Scalable approximate eigenspectrum via Lanczos iteration
  with full reorthogonalization (Ghorbani et al. 2019)
- compute_hessian_spectrum: Auto-dispatches based on parameter count
- hessian_vector_product: Core HVP primitive for Lanczos iteration

These tools support Hessian analysis (SC-2) for saddle point characterization
and landscape geometry measurement on octonionic, quaternionic, complex, and
real-valued models.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Flat-parameter helpers
# ---------------------------------------------------------------------------


def _get_flat_params(model: nn.Module) -> torch.Tensor:
    """Flatten all trainable parameters into a single 1D tensor."""
    return torch.cat([p.reshape(-1) for p in model.parameters() if p.requires_grad])


def _set_flat_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    """Set model parameters from a flat 1D tensor."""
    offset = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.numel()
            p.data.copy_(flat_params[offset : offset + numel].reshape(p.shape))
            offset += numel


def _unflatten_like(
    flat_vector: torch.Tensor, params: list[nn.Parameter]
) -> list[torch.Tensor]:
    """Reshape flat vector into list of tensors matching param shapes."""
    views: list[torch.Tensor] = []
    offset = 0
    for p in params:
        numel = p.numel()
        views.append(flat_vector[offset : offset + numel].reshape(p.shape))
        offset += numel
    return views


# ---------------------------------------------------------------------------
# Full Hessian computation
# ---------------------------------------------------------------------------


def compute_full_hessian(
    model: nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    data_x: torch.Tensor,
    data_y: torch.Tensor,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Compute the full Hessian matrix and its eigenspectrum.

    Uses torch.autograd.functional.hessian for exact second derivatives.
    Only feasible for small models (< ~2000 parameters).

    Args:
        model: PyTorch model.
        loss_fn: Loss function(output, target) -> scalar.
        data_x: Input data tensor.
        data_y: Target data tensor.
        device: Device for computation.

    Returns:
        Dict with keys:
          - eigenvalues: numpy array of eigenvalues (sorted ascending)
          - n_negative: count of eigenvalues < -1e-10
          - n_positive: count of eigenvalues > 1e-10
          - n_zero: count of eigenvalues in [-1e-10, 1e-10]
          - trace: sum of eigenvalues
          - spectral_norm: max absolute eigenvalue
          - negative_ratio: fraction of negative eigenvalues
          - method: 'full'
    """
    model = model.to(device)
    data_x = data_x.to(device)
    data_y = data_y.to(device)

    # Collect trainable params and build name -> param mapping
    param_names: list[str] = []
    param_shapes: list[torch.Size] = []
    param_numels: list[int] = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            param_names.append(name)
            param_shapes.append(p.shape)
            param_numels.append(p.numel())

    flat_params = _get_flat_params(model).detach().clone()
    flat_params.requires_grad_(True)
    n_params = flat_params.numel()

    def loss_func(flat: torch.Tensor) -> torch.Tensor:
        """Scalar loss as a function of flat parameter vector."""
        # Build parameter dict from flat vector for functional_call
        param_dict: dict[str, torch.Tensor] = {}
        offset = 0
        for name, shape, numel in zip(param_names, param_shapes, param_numels):
            param_dict[name] = flat[offset : offset + numel].reshape(shape)
            offset += numel
        # Use torch.func.functional_call to thread gradient through params
        output = torch.func.functional_call(model, param_dict, (data_x,))
        return loss_fn(output, data_y)

    # Compute full Hessian via autograd
    H = torch.autograd.functional.hessian(loss_func, flat_params)
    H = H.detach().reshape(n_params, n_params)

    # Symmetrize (numerical noise)
    H = (H + H.T) / 2.0

    # Eigendecomposition
    eigenvalues = torch.linalg.eigvalsh(H).cpu().numpy()

    # Classify eigenvalues
    n_negative = int(np.sum(eigenvalues < -1e-10))
    n_positive = int(np.sum(eigenvalues > 1e-10))
    n_zero = int(np.sum(np.abs(eigenvalues) <= 1e-10))
    trace = float(np.sum(eigenvalues))
    spectral_norm = float(np.max(np.abs(eigenvalues)))
    total = len(eigenvalues)
    negative_ratio = n_negative / total if total > 0 else 0.0

    return {
        "eigenvalues": eigenvalues,
        "n_negative": n_negative,
        "n_positive": n_positive,
        "n_zero": n_zero,
        "trace": trace,
        "spectral_norm": spectral_norm,
        "negative_ratio": negative_ratio,
        "method": "full",
    }


# ---------------------------------------------------------------------------
# Hessian-vector product
# ---------------------------------------------------------------------------


def hessian_vector_product(
    grad_flat: torch.Tensor,
    params: list[nn.Parameter],
    v_list: list[torch.Tensor],
) -> torch.Tensor:
    """Compute Hessian-vector product Hv via backward on gradient.

    Args:
        grad_flat: Flat gradient vector (with grad graph retained).
        params: List of model parameters (must match grad_flat ordering).
        v_list: List of tensors matching param shapes (the vector v).

    Returns:
        Flat Hv tensor.
    """
    # Reshape grad_flat into param-shaped pieces for grad computation
    grad_pieces = _unflatten_like(grad_flat, params)

    # Compute Hv = d(grad . v) / d(params)
    hvp = torch.autograd.grad(
        grad_pieces,
        params,
        grad_outputs=v_list,
        retain_graph=True,
    )

    return torch.cat([h.reshape(-1) for h in hvp])


# ---------------------------------------------------------------------------
# Stochastic Lanczos quadrature
# ---------------------------------------------------------------------------


def stochastic_lanczos(
    model: nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    data_x: torch.Tensor,
    data_y: torch.Tensor,
    n_iterations: int = 200,
    n_samples: int = 5,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Stochastic Lanczos eigenspectrum approximation.

    Implements the Lanczos algorithm with full reorthogonalization for
    approximate Hessian eigenspectrum estimation (Ghorbani et al. 2019).

    For each random starting vector, runs Lanczos iteration to build a
    tridiagonal matrix whose eigenvalues (Ritz values) approximate
    the Hessian eigenspectrum.

    Args:
        model: PyTorch model.
        loss_fn: Loss function(output, target) -> scalar.
        data_x: Input data tensor.
        data_y: Target data tensor.
        n_iterations: Number of Lanczos iterations per sample.
        n_samples: Number of random starting vectors.
        device: Device for computation.

    Returns:
        Dict with keys:
          - ritz_values: numpy array of all Ritz values across samples
          - n_negative_approx: count of negative Ritz values
          - negative_ratio_approx: fraction of negative Ritz values
          - trace_approx: mean trace estimate across samples
          - method: 'lanczos'
          - n_iterations: actual iterations used
          - n_samples: number of samples used
    """
    model = model.to(device)
    data_x = data_x.to(device)
    data_y = data_y.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

    # Clamp iterations to at most n_params (can't exceed matrix dimension)
    actual_iters = min(n_iterations, n_params)

    all_ritz_values: list[np.ndarray] = []
    trace_estimates: list[float] = []

    for sample_idx in range(n_samples):
        # Compute gradient once with create_graph=True for HVP
        model.zero_grad()
        output = model(data_x)
        loss = loss_fn(output, data_y)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        grad_flat = torch.cat([g.reshape(-1) for g in grads])

        # Random starting vector
        rng = torch.Generator(device="cpu")
        rng.manual_seed(42 + sample_idx)
        v = torch.randn(n_params, generator=rng, device=device)
        v = v / v.norm()

        # Storage for Lanczos vectors (for full reorthogonalization)
        V = torch.zeros(actual_iters + 1, n_params, device=device)
        V[0] = v

        # Tridiagonal matrix entries
        alphas = torch.zeros(actual_iters, device=device)
        betas = torch.zeros(actual_iters, device=device)

        v_prev = torch.zeros_like(v)
        beta_prev = 0.0

        for k in range(actual_iters):
            # Hessian-vector product
            v_list = _unflatten_like(V[k], params)
            Hv = hessian_vector_product(grad_flat, params, v_list)

            # alpha_k = v_k^T H v_k
            alpha_k = V[k].dot(Hv).item()
            alphas[k] = alpha_k

            # w = Hv - alpha_k * v_k - beta_{k-1} * v_{k-1}
            w = Hv - alpha_k * V[k]
            if k > 0:
                w = w - betas[k - 1] * V[k - 1]

            # Full reorthogonalization: remove all components along previous vectors
            for j in range(k + 1):
                coeff = w.dot(V[j])
                w = w - coeff * V[j]

            beta_k = w.norm().item()
            betas[k] = beta_k

            if k < actual_iters - 1:
                if beta_k < 1e-12:
                    # Lanczos breakdown -- invariant subspace found
                    actual_iters_used = k + 1
                    alphas = alphas[:actual_iters_used]
                    betas = betas[:actual_iters_used]
                    break
                V[k + 1] = w / beta_k

        # Build tridiagonal matrix T
        m = len(alphas)
        T = torch.zeros(m, m, device=device)
        for i in range(m):
            T[i, i] = alphas[i]
            if i < m - 1:
                T[i, i + 1] = betas[i]
                T[i + 1, i] = betas[i]

        # Ritz values = eigenvalues of T
        ritz = torch.linalg.eigvalsh(T).cpu().numpy()
        all_ritz_values.append(ritz)
        trace_estimates.append(float(np.sum(ritz)))

    # Aggregate across samples
    ritz_combined = np.concatenate(all_ritz_values)
    n_neg = int(np.sum(ritz_combined < -1e-10))
    total = len(ritz_combined)
    neg_ratio = n_neg / total if total > 0 else 0.0
    trace_approx = float(np.mean(trace_estimates))

    return {
        "ritz_values": ritz_combined,
        "n_negative_approx": n_neg,
        "negative_ratio_approx": neg_ratio,
        "trace_approx": trace_approx,
        "method": "lanczos",
        "n_iterations": n_iterations,
        "n_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# Auto-dispatching spectrum analysis
# ---------------------------------------------------------------------------


def compute_hessian_spectrum(
    model: nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    data_x: torch.Tensor,
    data_y: torch.Tensor,
    device: str | torch.device = "cpu",
    method: str = "auto",
    max_full_params: int = 2000,
    **lanczos_kwargs: Any,
) -> dict[str, Any]:
    """Compute Hessian eigenspectrum, auto-selecting method by model size.

    Args:
        model: PyTorch model.
        loss_fn: Loss function(output, target) -> scalar.
        data_x: Input data tensor.
        data_y: Target data tensor.
        device: Device for computation.
        method: 'full', 'lanczos', or 'auto' (default).
        max_full_params: Threshold for auto selection. Models with fewer
            trainable params use full Hessian; larger models use Lanczos.
        **lanczos_kwargs: Extra kwargs forwarded to stochastic_lanczos.

    Returns:
        Dict with eigenspectrum results (see compute_full_hessian or
        stochastic_lanczos for field details).
    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if method == "auto":
        method = "full" if n_params <= max_full_params else "lanczos"

    if method == "full":
        return compute_full_hessian(model, loss_fn, data_x, data_y, device=device)
    elif method == "lanczos":
        return stochastic_lanczos(
            model, loss_fn, data_x, data_y, device=device, **lanczos_kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'full', 'lanczos', or 'auto'.")
