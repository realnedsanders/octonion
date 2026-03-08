"""Random octonion generation utilities.

Provides functions for generating random octonions with controlled properties:
- random_octonion: general random octonion (Gaussian components)
- random_unit_octonion: unit-norm octonion on S^7
- random_pure_octonion: pure imaginary (real part = 0)

All support batch generation, dtype control, and reproducible seeding via
torch.Generator.
"""

from __future__ import annotations

from typing import Optional

import torch

from octonion._octonion import Octonion


def random_octonion(
    *,
    batch_size: Optional[int] = None,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> Octonion:
    """Generate a random octonion with Gaussian-distributed components.

    Components are drawn independently from N(0, 1).

    Args:
        batch_size: If None, returns a single octonion (shape [8]).
            If int, returns a batch (shape [batch_size, 8]).
        dtype: Tensor dtype (default float64 for precision).
        device: Tensor device (default None = CPU).
        generator: PyTorch Generator for reproducible randomness.

    Returns:
        Octonion wrapping a tensor of shape [8] or [batch_size, 8].
    """
    shape = (8,) if batch_size is None else (batch_size, 8)
    data = torch.randn(*shape, dtype=dtype, device=device, generator=generator)
    return Octonion(data)


def random_unit_octonion(
    *,
    batch_size: Optional[int] = None,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> Octonion:
    """Generate a random unit octonion (uniformly distributed on S^7).

    Draws Gaussian components and normalizes to unit norm.
    This produces a uniform distribution on the 7-sphere by the
    rotational symmetry of the Gaussian distribution.

    Args:
        batch_size: If None, returns a single octonion (shape [8]).
            If int, returns a batch (shape [batch_size, 8]).
        dtype: Tensor dtype (default float64 for precision).
        device: Tensor device (default None = CPU).
        generator: PyTorch Generator for reproducible randomness.

    Returns:
        Octonion with norm 1 (to numerical precision).
    """
    shape = (8,) if batch_size is None else (batch_size, 8)
    data = torch.randn(*shape, dtype=dtype, device=device, generator=generator)
    n = torch.sqrt(torch.sum(data**2, dim=-1, keepdim=True))
    # Extremely unlikely to get zero norm from Gaussian, but guard anyway
    n = torch.clamp(n, min=1e-30)
    return Octonion(data / n)


def random_pure_octonion(
    *,
    batch_size: Optional[int] = None,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> Octonion:
    """Generate a random pure (imaginary) octonion with real part = 0.

    Draws 7 Gaussian components for e1..e7 and sets e0 = 0.

    Args:
        batch_size: If None, returns a single octonion (shape [8]).
            If int, returns a batch (shape [batch_size, 8]).
        dtype: Tensor dtype (default float64 for precision).
        device: Tensor device (default None = CPU).
        generator: PyTorch Generator for reproducible randomness.

    Returns:
        Octonion with real component exactly 0.
    """
    imag_shape = (7,) if batch_size is None else (batch_size, 7)
    imag = torch.randn(*imag_shape, dtype=dtype, device=device, generator=generator)

    if batch_size is None:
        real = torch.zeros(1, dtype=dtype, device=device)
    else:
        real = torch.zeros(batch_size, 1, dtype=dtype, device=device)

    data = torch.cat([real, imag], dim=-1)
    return Octonion(data)
