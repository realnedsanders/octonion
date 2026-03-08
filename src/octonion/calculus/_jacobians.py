"""Analytic 8x8 Jacobian matrices for all 7 octonionic primitives.

Stub module -- full implementation in Task 2.

Convention: Baez 2002, mod-7 Fano plane basis.
"""

from __future__ import annotations

from typing import Tuple

import torch


def jacobian_mul(
    a: torch.Tensor, b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Analytic Jacobian of octonion multiplication. Stub -- see Task 2."""
    raise NotImplementedError("jacobian_mul: implemented in Task 2")


def jacobian_exp(o: torch.Tensor) -> torch.Tensor:
    """Analytic Jacobian of octonion exponential. Stub -- see Task 2."""
    raise NotImplementedError("jacobian_exp: implemented in Task 2")


def jacobian_log(o: torch.Tensor) -> torch.Tensor:
    """Analytic Jacobian of octonion logarithm. Stub -- see Task 2."""
    raise NotImplementedError("jacobian_log: implemented in Task 2")


def jacobian_conjugate(o: torch.Tensor) -> torch.Tensor:
    """Analytic Jacobian of octonion conjugation. Stub -- see Task 2."""
    raise NotImplementedError("jacobian_conjugate: implemented in Task 2")


def jacobian_inverse(o: torch.Tensor) -> torch.Tensor:
    """Analytic Jacobian of octonion inverse. Stub -- see Task 2."""
    raise NotImplementedError("jacobian_inverse: implemented in Task 2")


def jacobian_inner_product(
    a: torch.Tensor, b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Analytic Jacobian of inner product. Stub -- see Task 2."""
    raise NotImplementedError("jacobian_inner_product: implemented in Task 2")


def jacobian_cross_product(
    a: torch.Tensor, b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Analytic Jacobian of cross product. Stub -- see Task 2."""
    raise NotImplementedError("jacobian_cross_product: implemented in Task 2")
