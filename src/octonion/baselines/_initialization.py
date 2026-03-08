"""Per-algebra weight initialization following published literature.

- Real: Kaiming/He initialization (standard PyTorch)
- Complex: Trabelsi et al. (ICLR 2018) -- Rayleigh magnitude + uniform phase
- Quaternion: Parcollet/Gaudet (IJCNN 2018) -- Chi(4) magnitude + polar form
- Octonion: Extension to 8-DOF with Chi(8) magnitude + 7 phase angles

All functions modify tensors in-place using .copy_() under torch.no_grad().
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def real_init(weight: torch.Tensor, criterion: str = "he") -> None:
    """Initialize real-valued weight using Kaiming/He initialization.

    Args:
        weight: Weight tensor to initialize in-place.
        criterion: "he" for Kaiming normal, "glorot" for Xavier normal.
    """
    with torch.no_grad():
        if criterion == "he":
            nn.init.kaiming_normal_(weight, nonlinearity="relu")
        elif criterion == "glorot":
            nn.init.xavier_normal_(weight)
        else:
            raise ValueError(
                f"Unknown criterion: {criterion!r}. Use 'he' or 'glorot'."
            )


def _compute_sigma(
    fan_in: int, fan_out: int, criterion: str
) -> float:
    """Compute standard deviation for hypercomplex initialization.

    Args:
        fan_in: Number of input features.
        fan_out: Number of output features.
        criterion: "glorot" or "he".

    Returns:
        Standard deviation for magnitude sampling.
    """
    if criterion == "glorot":
        return 1.0 / math.sqrt(2 * (fan_in + fan_out))
    elif criterion == "he":
        return 1.0 / math.sqrt(2 * fan_in)
    else:
        raise ValueError(
            f"Unknown criterion: {criterion!r}. Use 'glorot' or 'he'."
        )


def complex_init(
    W_r: torch.Tensor,
    W_i: torch.Tensor,
    criterion: str = "glorot",
) -> None:
    """Initialize complex weight using Rayleigh magnitude + uniform phase.

    Following Trabelsi et al., "Deep Complex Networks" (ICLR 2018):
    1. Sample magnitude |W| from Chi2(df=2).sqrt() * sigma (= Rayleigh(sigma))
    2. Sample phase theta from Uniform(-pi, pi)
    3. W_r = |W| * cos(theta), W_i = |W| * sin(theta)

    Args:
        W_r: Real part weight tensor [out_features, in_features].
        W_i: Imaginary part weight tensor [out_features, in_features].
        criterion: "glorot" or "he".
    """
    fan_in, fan_out = W_r.shape[1], W_r.shape[0]
    sigma = _compute_sigma(fan_in, fan_out, criterion)

    # Rayleigh distribution via Chi2(df=2)
    magnitude = torch.distributions.Chi2(df=2).sample(W_r.shape).sqrt() * sigma
    phase = torch.empty_like(W_r).uniform_(-math.pi, math.pi)

    with torch.no_grad():
        W_r.copy_(magnitude.to(W_r.device, W_r.dtype) * torch.cos(phase))
        W_i.copy_(magnitude.to(W_i.device, W_i.dtype) * torch.sin(phase))


def quaternion_init(
    W_r: torch.Tensor,
    W_i: torch.Tensor,
    W_j: torch.Tensor,
    W_k: torch.Tensor,
    criterion: str = "glorot",
) -> None:
    """Initialize quaternion weight using polar form with Chi(4) magnitude.

    Following Gaudet & Maida, "Deep Quaternion Networks" (IJCNN 2018):
    1. Sample magnitude from Chi2(df=4).sqrt() * sigma
    2. Sample 3 phase angles from Uniform(-pi, pi)
    3. Decompose into 4 components via polar form

    Args:
        W_r: Real part weight tensor [out_features, in_features].
        W_i: i-component weight tensor [out_features, in_features].
        W_j: j-component weight tensor [out_features, in_features].
        W_k: k-component weight tensor [out_features, in_features].
        criterion: "glorot" or "he".
    """
    fan_in, fan_out = W_r.shape[1], W_r.shape[0]
    sigma = _compute_sigma(fan_in, fan_out, criterion)

    # Chi distribution with 4 DOF for magnitude
    magnitude = torch.distributions.Chi2(df=4).sample(W_r.shape).sqrt() * sigma
    phi1 = torch.empty_like(W_r).uniform_(-math.pi, math.pi)
    phi2 = torch.empty_like(W_r).uniform_(-math.pi, math.pi)
    phi3 = torch.empty_like(W_r).uniform_(-math.pi, math.pi)

    with torch.no_grad():
        W_r.copy_((magnitude * torch.cos(phi1)).to(W_r.device, W_r.dtype))
        imag_mag = magnitude * torch.sin(phi1)
        W_i.copy_((imag_mag * torch.cos(phi2)).to(W_i.device, W_i.dtype))
        remaining = imag_mag * torch.sin(phi2)
        W_j.copy_((remaining * torch.cos(phi3)).to(W_j.device, W_j.dtype))
        W_k.copy_((remaining * torch.sin(phi3)).to(W_k.device, W_k.dtype))


def octonion_init(
    weights: list[torch.Tensor],
    criterion: str = "glorot",
) -> None:
    """Initialize octonion weight using polar form with Chi(8) magnitude.

    Extension of the quaternion initialization pattern to 8-DOF:
    1. Sample magnitude from Chi2(df=8).sqrt() * sigma
    2. Sample 7 phase angles from Uniform(-pi, pi)
    3. Decompose into 8 components via nested polar form

    The decomposition recursively splits the magnitude using cos/sin:
    w0 = mag * cos(phi1)
    rem1 = mag * sin(phi1)
    w1 = rem1 * cos(phi2)
    rem2 = rem1 * sin(phi2)
    ...continuing for all 8 components using 7 phase angles.

    Args:
        weights: List of 8 weight tensors, each [out_features, in_features].
        criterion: "glorot" or "he".
    """
    if len(weights) != 8:
        raise ValueError(
            f"Expected 8 weight tensors for octonion init, got {len(weights)}."
        )

    W0 = weights[0]
    fan_in, fan_out = W0.shape[1], W0.shape[0]
    sigma = _compute_sigma(fan_in, fan_out, criterion)

    # Chi distribution with 8 DOF for magnitude
    magnitude = torch.distributions.Chi2(df=8).sample(W0.shape).sqrt() * sigma
    phases = [
        torch.empty_like(W0).uniform_(-math.pi, math.pi) for _ in range(7)
    ]

    with torch.no_grad():
        # Nested polar decomposition: 7 phases -> 8 components
        remaining = magnitude
        for idx in range(7):
            weights[idx].copy_(
                (remaining * torch.cos(phases[idx])).to(
                    weights[idx].device, weights[idx].dtype
                )
            )
            remaining = remaining * torch.sin(phases[idx])
        # Last component gets whatever remains
        weights[7].copy_(remaining.to(weights[7].device, weights[7].dtype))
