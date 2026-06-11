"""Robustness tests for octonion exp/log: singular set, gradients, float32.

Covers the fixes for:
- NaN gradients at pure-real inputs through the public octonion_exp/log
  (they now route through the analytic-Jacobian autograd Functions)
- the negative real axis, where the principal log is genuinely undefined
  (NaN imaginary parts) but everything arbitrarily close to it is accurate
- the acos cancellation band (theta now computed via atan2)
- dtype-aware series thresholds so float32 gets correct branches
- broadcasting through the binary autograd Functions
"""

from __future__ import annotations

import math

import pytest
import torch

from octonion._operations import octonion_exp, octonion_log
from octonion.calculus import (
    OctonionCrossProductFunction,
    OctonionInnerProductFunction,
    OctonionMulFunction,
    jacobian_exp,
    jacobian_log,
)
from octonion.calculus._numeric import numeric_jacobian


def _real(value: float, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    t = torch.zeros(8, dtype=dtype)
    t[0] = value
    return t


class TestGradientsAtPureReal:
    def test_exp_gradient_finite_at_pure_real(self) -> None:
        """d exp/do at v = 0 is finite (was NaN via sqrt(0) backward)."""
        o = _real(0.5).requires_grad_()
        octonion_exp(o).sum().backward()
        assert o.grad is not None
        assert bool(torch.isfinite(o.grad).all())
        # At v=0: d(exp)_k/dv_k = e^a, d(exp)_0/da = e^a
        ea = math.exp(0.5)
        assert torch.allclose(o.grad, torch.full((8,), ea, dtype=torch.float64))

    def test_log_gradient_finite_at_positive_real(self) -> None:
        o = _real(2.0).requires_grad_()
        octonion_log(o).sum().backward()
        assert o.grad is not None
        assert bool(torch.isfinite(o.grad).all())

    def test_public_exp_log_use_analytic_jacobian(self) -> None:
        """Autograd through the public API equals the analytic Jacobian."""
        torch.manual_seed(0)
        o = torch.randn(8, dtype=torch.float64)
        o[0] = o[0].abs() + 1.0
        J_auto = torch.autograd.functional.jacobian(octonion_log, o)
        assert torch.allclose(J_auto, jacobian_log(o), atol=1e-10)
        J_auto_exp = torch.autograd.functional.jacobian(octonion_exp, o)
        assert torch.allclose(J_auto_exp, jacobian_exp(o), atol=1e-10)


class TestNegativeRealAxis:
    def test_log_of_minus_one_imag_is_nan(self) -> None:
        """log(-1) has no preferred imaginary direction: imag is NaN,
        real part log|q| = 0 (was silently 0, i.e. log(-1) == log(1))."""
        result = octonion_log(_real(-1.0))
        assert result[0].item() == 0.0
        assert bool(torch.isnan(result[1:]).all())

    def test_log_jacobian_nan_on_negative_axis(self) -> None:
        """The directional derivative genuinely diverges there."""
        J = jacobian_log(_real(-1.0))
        # Real row/column (log q part) stays finite ...
        assert bool(torch.isfinite(J[0, 0]))
        # ... the imaginary diagonal (theta/r) is NaN
        assert bool(torch.isnan(torch.diagonal(J)[1:]).all())

    def test_log_accurate_arbitrarily_close_to_axis(self) -> None:
        """log(-1 + eps*e1) ~ pi*e1: defined, finite, accurate."""
        near = torch.tensor([-1.0, 1e-10, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = octonion_log(near)
        assert bool(torch.isfinite(result).all())
        assert abs(result[1].item() - math.pi) < 1e-9
        # exp(log(o)) == o roundtrip survives next to the singular set
        assert torch.allclose(octonion_exp(result), near, atol=1e-12)

    def test_log_zero_norm_raises(self) -> None:
        with pytest.raises(ValueError, match="non-zero norm"):
            octonion_log(torch.zeros(8, dtype=torch.float64))


class TestCancellationBand:
    """theta = acos(a/q) lost half its digits for small ||v||/a; atan2 does not."""

    @pytest.mark.parametrize("r_scale", [1e-9, 1e-7, 1e-5, 1e-3])
    def test_log_jacobian_matches_fd_in_band(self, r_scale: float) -> None:
        o = torch.tensor([2.0] + [r_scale] * 7, dtype=torch.float64)
        J = jacobian_log(o)
        J_fd = numeric_jacobian(octonion_log, o, eps=1e-8)
        assert (J - J_fd).abs().max().item() < 1e-6

    @pytest.mark.parametrize("r_scale", [1e-9, 1e-7, 1e-5, 1e-3])
    def test_exp_jacobian_matches_fd_in_band(self, r_scale: float) -> None:
        o = torch.tensor([0.3] + [r_scale] * 7, dtype=torch.float64)
        J = jacobian_exp(o)
        J_fd = numeric_jacobian(octonion_exp, o, eps=1e-8)
        assert (J - J_fd).abs().max().item() < 1e-6

    def test_log_exp_roundtrip_tiny_imaginary(self) -> None:
        """Roundtrip through the series branch (used to flush imag to 0)."""
        o = torch.tensor([1.5] + [1e-12] * 7, dtype=torch.float64)
        rt = octonion_log(octonion_exp(o))
        assert torch.allclose(rt, o, atol=1e-15)


class TestFloat32:
    def test_exp_jacobian_float32_matches_float64(self) -> None:
        """The float32 branch thresholds give float32-accurate Jacobians
        across the band where float64 thresholds would mis-branch."""
        for r_scale in (0.0, 1e-6, 1e-4, 1e-2, 1.0):
            o = torch.tensor([0.3] + [r_scale] * 7, dtype=torch.float32)
            J32 = jacobian_exp(o)
            J64 = jacobian_exp(o.double())
            assert (J32.double() - J64).abs().max().item() < 1e-5, r_scale

    def test_log_jacobian_float32_matches_float64(self) -> None:
        for r_scale in (0.0, 1e-6, 1e-4, 1e-2, 1.0):
            o = torch.tensor([2.0] + [r_scale] * 7, dtype=torch.float32)
            J32 = jacobian_log(o)
            J64 = jacobian_log(o.double())
            assert (J32.double() - J64).abs().max().item() < 1e-5, r_scale

    def test_exp_log_roundtrip_float32(self) -> None:
        torch.manual_seed(1)
        o = torch.randn(64, 8, dtype=torch.float32) * 0.5
        o[:, 0] = o[:, 0].abs() + 0.5  # stay near the principal branch
        rt = octonion_log(octonion_exp(o))
        assert torch.allclose(rt, o, atol=1e-5)

    def test_exp_gradient_finite_at_pure_real_float32(self) -> None:
        o = _real(0.5, dtype=torch.float32).requires_grad_()
        octonion_exp(o).sum().backward()
        assert o.grad is not None
        assert bool(torch.isfinite(o.grad).all())


class TestBroadcasting:
    """Binary autograd Functions accept broadcast inputs and reduce grads."""

    def test_mul_broadcast_gradients(self) -> None:
        torch.manual_seed(2)
        a = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(16, 8, dtype=torch.float64, requires_grad=True)
        OctonionMulFunction.apply(a, b).sum().backward()
        assert a.grad is not None and a.grad.shape == (8,)
        assert b.grad is not None and b.grad.shape == (16, 8)

    def test_mul_broadcast_matches_unbroadcast(self) -> None:
        """Grad of broadcast a equals the sum of grads of expanded a."""
        torch.manual_seed(3)
        a = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(16, 8, dtype=torch.float64)
        OctonionMulFunction.apply(a, b).sum().backward()
        a2 = a.detach().expand(16, 8).clone().requires_grad_()
        OctonionMulFunction.apply(a2, b).sum().backward()
        assert a.grad is not None and a2.grad is not None
        assert torch.allclose(a.grad, a2.grad.sum(dim=0), atol=1e-12)

    def test_inner_product_broadcast_gradients(self) -> None:
        a = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
        OctonionInnerProductFunction.apply(a, b).sum().backward()
        assert a.grad is not None and a.grad.shape == (8,)
        assert b.grad is not None and b.grad.shape == (5, 8)

    def test_cross_product_broadcast_gradients(self) -> None:
        a = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
        OctonionCrossProductFunction.apply(a, b).sum().backward()
        assert a.grad is not None and a.grad.shape == (8,)
        assert b.grad is not None and b.grad.shape == (5, 8)
