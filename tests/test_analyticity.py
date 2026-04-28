"""Tests for CR-like analyticity conditions and learning rate scaling heuristic.

Verifies:
- Octonionic CR-like (Cauchy-Riemann) conditions correctly identify analytic functions
- Left multiplication by a fixed octonion satisfies CR condition
- Right multiplication, conjugation, exp do NOT satisfy CR condition
- Analyticity residual API returns correct types
- Gradient magnitude statistics and LR scaling heuristic

Convention: Baez 2002, mod-7 Fano plane basis.
"""

from __future__ import annotations

import torch

from octonion._multiplication import octonion_mul
from octonion.calculus._analyticity import (
    analyticity_residual,
    cauchy_riemann_octonion,
    is_octonionic_analytic,
)
from octonion.calculus._lr_scaling import (
    gradient_magnitude_stats,
    lr_scaling_heuristic,
    suggest_lr,
)


class TestAnalyticityConditions:
    """CR-like conditions for octonionic functions."""

    def test_left_mul_is_analytic(self) -> None:
        """Left multiplication by a fixed octonion c: f(x) = c * x is analytic.

        Its Jacobian is exactly L_c (the left multiplication matrix), which
        by definition satisfies the CR condition with residual 0.
        """
        torch.manual_seed(42)
        c = torch.randn(8, dtype=torch.float64)
        x = torch.randn(8, dtype=torch.float64)

        def left_mul(x: torch.Tensor) -> torch.Tensor:
            return octonion_mul(c.expand_as(x), x)

        assert is_octonionic_analytic(left_mul, x, tol=1e-4)

    def test_identity_is_analytic(self) -> None:
        """Identity function f(x) = x has Jacobian I = L_{e_0}, so it is analytic."""
        torch.manual_seed(42)
        x = torch.randn(8, dtype=torch.float64)

        def identity(x: torch.Tensor) -> torch.Tensor:
            return x

        assert is_octonionic_analytic(identity, x, tol=1e-4)

    def test_right_mul_not_analytic(self) -> None:
        """Right multiplication f(x) = x * c is NOT analytic (Jacobian is R_c, not L_c).

        Unless c is purely real, in which case R_c = c_0 * I = L_{c_0 * e_0}.
        """
        torch.manual_seed(42)
        # Use a non-real c to ensure R_c != L_c
        c = torch.randn(8, dtype=torch.float64)
        c[0] = 0.0  # Make c purely imaginary for maximum non-analyticity
        x = torch.randn(8, dtype=torch.float64)

        def right_mul(x: torch.Tensor) -> torch.Tensor:
            return octonion_mul(x, c.expand_as(x))

        assert not is_octonionic_analytic(right_mul, x, tol=1e-4)

    def test_conjugate_not_analytic(self) -> None:
        """Conjugation f(x) = x* is NOT analytic.

        Its Jacobian is diag([1, -1, ..., -1]) which is not a left multiplication
        matrix (L_c has L[0,0] = c_0 and L[k,0] = c_k, so first column = c;
        for diag([1,-1,...,-1]), first column is [1,0,...,0] = e_0, and
        L_{e_0} = I, not diag([1,-1,...,-1])).
        """
        torch.manual_seed(42)
        x = torch.randn(8, dtype=torch.float64)

        def conjugate(x: torch.Tensor) -> torch.Tensor:
            result = x.clone()
            result[..., 1:] = -result[..., 1:]
            return result

        assert not is_octonionic_analytic(conjugate, x, tol=1e-4)

    def test_exp_not_analytic(self) -> None:
        """exp is NOT analytic at generic points.

        The octonionic exponential's Jacobian has a complex structure involving
        sinc and outer products that generally does not correspond to any L_c.
        """
        torch.manual_seed(42)
        from octonion._operations import octonion_exp

        x = torch.randn(8, dtype=torch.float64)

        def exp_fn(x: torch.Tensor) -> torch.Tensor:
            return octonion_exp(x)

        assert not is_octonionic_analytic(exp_fn, x, tol=1e-4)

    def test_analyticity_residual_api(self) -> None:
        """analyticity_residual returns a scalar tensor with correct properties."""
        torch.manual_seed(42)
        x = torch.randn(8, dtype=torch.float64)

        def identity(x: torch.Tensor) -> torch.Tensor:
            return x

        residual = analyticity_residual(identity, x)
        assert isinstance(residual, torch.Tensor)
        assert residual.dim() == 0  # scalar
        assert residual.item() >= 0.0  # non-negative (Frobenius norm)

    def test_cauchy_riemann_octonion_shape(self) -> None:
        """cauchy_riemann_octonion accepts 8x8 Jacobian and returns scalar residual."""
        J = torch.eye(8, dtype=torch.float64)
        residual = cauchy_riemann_octonion(J)
        assert isinstance(residual, torch.Tensor)
        assert residual.dim() == 0  # scalar
        assert residual.item() >= 0.0

    def test_cauchy_riemann_identity_zero_residual(self) -> None:
        """Identity matrix (J = I = L_{e_0}) should have zero CR residual."""
        J = torch.eye(8, dtype=torch.float64)
        residual = cauchy_riemann_octonion(J)
        assert residual.item() < 1e-12

    def test_cauchy_riemann_batched(self) -> None:
        """cauchy_riemann_octonion supports batched Jacobians."""
        batch = torch.eye(8, dtype=torch.float64).unsqueeze(0).expand(5, -1, -1)
        residual = cauchy_riemann_octonion(batch)
        assert residual.shape == (5,)
        assert (residual < 1e-12).all()


class TestLRScaling:
    """Learning rate scaling heuristic from gradient magnitude statistics."""

    def test_gradient_magnitude_stats(self) -> None:
        """gradient_magnitude_stats returns dict with expected keys and positive values."""
        from octonion._linear import OctonionLinear

        torch.manual_seed(42)
        layer = OctonionLinear(dtype=torch.float64)

        stats = gradient_magnitude_stats(layer, n_samples=50)

        expected_keys = {
            "grad_norm_mean",
            "grad_norm_std",
            "grad_norm_max",
            "grad_norm_min",
            "grad_per_component",
            "ratio_to_real",
        }
        assert expected_keys.issubset(stats.keys())
        assert stats["grad_norm_mean"] > 0.0
        assert stats["grad_norm_std"] >= 0.0
        assert stats["grad_norm_max"] > 0.0
        assert stats["grad_norm_min"] >= 0.0
        assert stats["ratio_to_real"] > 0.0
        assert len(stats["grad_per_component"]) == 8

    def test_lr_scaling_heuristic(self) -> None:
        """lr_scaling_heuristic returns a positive scaling factor."""
        stats = {
            "grad_norm_mean": 2.5,
            "grad_norm_std": 0.5,
            "grad_norm_max": 4.0,
            "grad_norm_min": 1.0,
            "grad_per_component": [0.3] * 8,
            "ratio_to_real": 2.0,
        }
        factor = lr_scaling_heuristic(stats)
        assert isinstance(factor, float)
        assert factor > 0.0

    def test_lr_scaling_inverse_relationship(self) -> None:
        """If octonionic gradients are K times real, scaling factor should be ~1/K."""
        stats = {
            "grad_norm_mean": 5.0,
            "grad_norm_std": 1.0,
            "grad_norm_max": 8.0,
            "grad_norm_min": 2.0,
            "grad_per_component": [0.6] * 8,
            "ratio_to_real": 3.0,
        }
        factor = lr_scaling_heuristic(stats)
        # Factor should be approximately 1/3.0
        assert abs(factor - 1.0 / 3.0) < 0.01

    def test_suggest_lr(self) -> None:
        """suggest_lr returns a positive learning rate."""
        from octonion._linear import OctonionLinear

        torch.manual_seed(42)
        layer = OctonionLinear(dtype=torch.float64)

        adjusted_lr = suggest_lr(0.01, layer, n_samples=50)
        assert isinstance(adjusted_lr, float)
        assert adjusted_lr > 0.0


class TestPublicAPI:
    """Verify the public API exports."""

    def test_calculus_imports(self) -> None:
        """All core calculus exports are importable."""

    def test_calculus_submodule_importable(self) -> None:
        """octonion.calculus is importable as a submodule."""
        from octonion import calculus

        assert hasattr(calculus, "ghr_derivative")
        assert hasattr(calculus, "jacobian_mul")
        assert hasattr(calculus, "octonion_gradcheck")
