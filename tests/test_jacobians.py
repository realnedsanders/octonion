"""Triple-check tests: analytic Jacobians vs numeric Jacobians for all 7 primitives.

Tests verify that each analytic Jacobian matches the finite-difference numeric
Jacobian to within atol=1e-10 on random float64 inputs. This is well within
the 1e-5 success criterion budget.

Primitives tested:
  1. mul (wrt a and wrt b)
  2. exp
  3. log
  4. conjugate
  5. inverse
  6. inner_product (wrt a and wrt b)
  7. cross_product (wrt a and wrt b)
"""

import pytest
import torch
from hypothesis import given, settings, assume

from conftest import octonion_tensors, nonzero_octonion_tensors

from octonion._multiplication import octonion_mul
from octonion._octonion import Octonion
from octonion._operations import (
    octonion_exp,
    octonion_log,
    inner_product,
    cross_product,
)
from octonion.calculus import (
    jacobian_mul,
    jacobian_exp,
    jacobian_log,
    jacobian_conjugate,
    jacobian_inverse,
    jacobian_inner_product,
    jacobian_cross_product,
    numeric_jacobian,
    numeric_jacobian_2arg,
)


ATOL = 1e-10


# =============================================================================
# Multiplication Jacobians
# =============================================================================


class TestJacobianMul:
    """Analytic mul Jacobian vs numeric for both arguments."""

    @given(a=octonion_tensors(min_value=-10, max_value=10),
           b=octonion_tensors(min_value=-10, max_value=10))
    @settings(max_examples=50, deadline=None)
    def test_mul_wrt_a(self, a: torch.Tensor, b: torch.Tensor) -> None:
        J_a_analytic, _ = jacobian_mul(a, b)
        J_a_numeric = numeric_jacobian_2arg(octonion_mul, a, b, wrt="a")
        assert torch.allclose(J_a_analytic, J_a_numeric, atol=ATOL), (
            f"mul Jacobian wrt a: max error {(J_a_analytic - J_a_numeric).abs().max().item():.2e}"
        )

    @given(a=octonion_tensors(min_value=-10, max_value=10),
           b=octonion_tensors(min_value=-10, max_value=10))
    @settings(max_examples=50, deadline=None)
    def test_mul_wrt_b(self, a: torch.Tensor, b: torch.Tensor) -> None:
        _, J_b_analytic = jacobian_mul(a, b)
        J_b_numeric = numeric_jacobian_2arg(octonion_mul, a, b, wrt="b")
        assert torch.allclose(J_b_analytic, J_b_numeric, atol=ATOL), (
            f"mul Jacobian wrt b: max error {(J_b_analytic - J_b_numeric).abs().max().item():.2e}"
        )


# =============================================================================
# Exponential Jacobian
# =============================================================================


class TestJacobianExp:
    """Analytic exp Jacobian vs numeric."""

    @given(o=octonion_tensors(min_value=-2, max_value=2))
    @settings(max_examples=50, deadline=None)
    def test_exp_general(self, o: torch.Tensor) -> None:
        # Ensure non-trivial imaginary norm for general test
        v_norm = torch.sqrt(torch.sum(o[1:] ** 2))
        assume(v_norm.item() > 0.1)

        def exp_raw(x: torch.Tensor) -> torch.Tensor:
            return octonion_exp(x)

        J_analytic = jacobian_exp(o)
        J_numeric = numeric_jacobian(exp_raw, o)
        assert torch.allclose(J_analytic, J_numeric, atol=ATOL), (
            f"exp Jacobian: max error {(J_analytic - J_numeric).abs().max().item():.2e}"
        )

    def test_exp_near_zero_imag(self) -> None:
        """Near-zero ||v|| should not produce NaN."""
        o = torch.tensor([1.0, 1e-15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         dtype=torch.float64)
        J = jacobian_exp(o)
        assert not torch.any(torch.isnan(J)), "exp Jacobian has NaN for near-zero imag"
        assert not torch.any(torch.isinf(J)), "exp Jacobian has Inf for near-zero imag"


# =============================================================================
# Logarithm Jacobian
# =============================================================================


class TestJacobianLog:
    """Analytic log Jacobian vs numeric."""

    @given(o=octonion_tensors(min_value=-3, max_value=3))
    @settings(max_examples=50, deadline=None)
    def test_log_general(self, o: torch.Tensor) -> None:
        # Need positive norm and non-trivial imaginary part
        q_norm = torch.sqrt(torch.sum(o ** 2))
        v_norm = torch.sqrt(torch.sum(o[1:] ** 2))
        assume(q_norm.item() > 0.1)
        assume(v_norm.item() > 0.1)

        def log_raw(x: torch.Tensor) -> torch.Tensor:
            return octonion_log(x)

        J_analytic = jacobian_log(o)
        J_numeric = numeric_jacobian(log_raw, o)
        assert torch.allclose(J_analytic, J_numeric, atol=ATOL), (
            f"log Jacobian: max error {(J_analytic - J_numeric).abs().max().item():.2e}"
        )


# =============================================================================
# Conjugate Jacobian
# =============================================================================


class TestJacobianConjugate:
    """Conjugate Jacobian is trivially diag([1, -1, -1, ..., -1])."""

    @given(o=octonion_tensors(min_value=-10, max_value=10))
    @settings(max_examples=50, deadline=None)
    def test_conjugate(self, o: torch.Tensor) -> None:
        def conj_raw(x: torch.Tensor) -> torch.Tensor:
            return Octonion(x).conjugate().components

        J_analytic = jacobian_conjugate(o)
        J_numeric = numeric_jacobian(conj_raw, o)
        assert torch.allclose(J_analytic, J_numeric, atol=ATOL), (
            f"conjugate Jacobian: max error {(J_analytic - J_numeric).abs().max().item():.2e}"
        )


# =============================================================================
# Inverse Jacobian
# =============================================================================


class TestJacobianInverse:
    """Analytic inverse Jacobian vs numeric on nonzero inputs."""

    @given(o=nonzero_octonion_tensors())
    @settings(max_examples=50, deadline=None)
    def test_inverse(self, o: torch.Tensor) -> None:
        # Ensure norm is not too small or too large for numerical stability
        n = torch.sqrt(torch.sum(o ** 2))
        assume(0.1 < n.item() < 100.0)

        def inv_raw(x: torch.Tensor) -> torch.Tensor:
            return Octonion(x).inverse().components

        J_analytic = jacobian_inverse(o)
        J_numeric = numeric_jacobian(inv_raw, o)
        assert torch.allclose(J_analytic, J_numeric, atol=ATOL), (
            f"inverse Jacobian: max error {(J_analytic - J_numeric).abs().max().item():.2e}"
        )


# =============================================================================
# Inner Product Jacobians
# =============================================================================


class TestJacobianInnerProduct:
    """Inner product Jacobians are trivially the other argument."""

    @given(a=octonion_tensors(min_value=-10, max_value=10),
           b=octonion_tensors(min_value=-10, max_value=10))
    @settings(max_examples=50, deadline=None)
    def test_inner_product_wrt_a(self, a: torch.Tensor, b: torch.Tensor) -> None:
        def ip_raw(x: torch.Tensor) -> torch.Tensor:
            return inner_product(Octonion(x), Octonion(b)).unsqueeze(-1)

        J_a_analytic, _ = jacobian_inner_product(a, b)
        J_a_numeric = numeric_jacobian(ip_raw, a)
        assert torch.allclose(J_a_analytic, J_a_numeric, atol=ATOL), (
            f"inner_product Jacobian wrt a: max error "
            f"{(J_a_analytic - J_a_numeric).abs().max().item():.2e}"
        )

    @given(a=octonion_tensors(min_value=-10, max_value=10),
           b=octonion_tensors(min_value=-10, max_value=10))
    @settings(max_examples=50, deadline=None)
    def test_inner_product_wrt_b(self, a: torch.Tensor, b: torch.Tensor) -> None:
        def ip_raw(x: torch.Tensor) -> torch.Tensor:
            return inner_product(Octonion(a), Octonion(x)).unsqueeze(-1)

        _, J_b_analytic = jacobian_inner_product(a, b)
        J_b_numeric = numeric_jacobian(ip_raw, b)
        assert torch.allclose(J_b_analytic, J_b_numeric, atol=ATOL), (
            f"inner_product Jacobian wrt b: max error "
            f"{(J_b_analytic - J_b_numeric).abs().max().item():.2e}"
        )


# =============================================================================
# Cross Product Jacobians
# =============================================================================


class TestJacobianCrossProduct:
    """Cross product Jacobians via structure constants."""

    @given(a=octonion_tensors(min_value=-10, max_value=10),
           b=octonion_tensors(min_value=-10, max_value=10))
    @settings(max_examples=50, deadline=None)
    def test_cross_product_wrt_a(self, a: torch.Tensor, b: torch.Tensor) -> None:
        def cp_raw(x: torch.Tensor) -> torch.Tensor:
            return cross_product(Octonion(x), Octonion(b)).components

        J_a_analytic, _ = jacobian_cross_product(a, b)
        J_a_numeric = numeric_jacobian(cp_raw, a)
        assert torch.allclose(J_a_analytic, J_a_numeric, atol=ATOL), (
            f"cross_product Jacobian wrt a: max error "
            f"{(J_a_analytic - J_a_numeric).abs().max().item():.2e}"
        )

    @given(a=octonion_tensors(min_value=-10, max_value=10),
           b=octonion_tensors(min_value=-10, max_value=10))
    @settings(max_examples=50, deadline=None)
    def test_cross_product_wrt_b(self, a: torch.Tensor, b: torch.Tensor) -> None:
        def cp_raw(x: torch.Tensor) -> torch.Tensor:
            return cross_product(Octonion(a), Octonion(x)).components

        _, J_b_analytic = jacobian_cross_product(a, b)
        J_b_numeric = numeric_jacobian(cp_raw, b)
        assert torch.allclose(J_b_analytic, J_b_numeric, atol=ATOL), (
            f"cross_product Jacobian wrt b: max error "
            f"{(J_b_analytic - J_b_numeric).abs().max().item():.2e}"
        )


# =============================================================================
# Batched Jacobian tests
# =============================================================================


class TestBatchedJacobians:
    """Verify Jacobians on batched [B, 8] inputs match per-element computation."""

    def test_mul_batched(self) -> None:
        B = 4
        a = torch.randn(B, 8, dtype=torch.float64)
        b = torch.randn(B, 8, dtype=torch.float64)

        J_a_batch, J_b_batch = jacobian_mul(a, b)
        assert J_a_batch.shape == (B, 8, 8)
        assert J_b_batch.shape == (B, 8, 8)

        for i in range(B):
            J_a_single, J_b_single = jacobian_mul(a[i], b[i])
            assert torch.allclose(J_a_batch[i], J_a_single, atol=1e-14)
            assert torch.allclose(J_b_batch[i], J_b_single, atol=1e-14)

    def test_exp_batched(self) -> None:
        B = 4
        o = torch.randn(B, 8, dtype=torch.float64)

        J_batch = jacobian_exp(o)
        assert J_batch.shape == (B, 8, 8)

        for i in range(B):
            J_single = jacobian_exp(o[i])
            assert torch.allclose(J_batch[i], J_single, atol=1e-14)

    def test_log_batched(self) -> None:
        B = 4
        # Use inputs with good norm and imaginary norm
        o = torch.randn(B, 8, dtype=torch.float64) + 1.0

        J_batch = jacobian_log(o)
        assert J_batch.shape == (B, 8, 8)

        for i in range(B):
            J_single = jacobian_log(o[i])
            assert torch.allclose(J_batch[i], J_single, atol=1e-14)

    def test_inverse_batched(self) -> None:
        B = 4
        o = torch.randn(B, 8, dtype=torch.float64) + 0.5

        J_batch = jacobian_inverse(o)
        assert J_batch.shape == (B, 8, 8)

        for i in range(B):
            J_single = jacobian_inverse(o[i])
            assert torch.allclose(J_batch[i], J_single, atol=1e-14)

    def test_inner_product_batched(self) -> None:
        B = 4
        a = torch.randn(B, 8, dtype=torch.float64)
        b = torch.randn(B, 8, dtype=torch.float64)

        J_a_batch, J_b_batch = jacobian_inner_product(a, b)
        assert J_a_batch.shape == (B, 1, 8)
        assert J_b_batch.shape == (B, 1, 8)

        for i in range(B):
            J_a_single, J_b_single = jacobian_inner_product(a[i], b[i])
            assert torch.allclose(J_a_batch[i], J_a_single, atol=1e-14)
            assert torch.allclose(J_b_batch[i], J_b_single, atol=1e-14)

    def test_cross_product_batched(self) -> None:
        B = 4
        a = torch.randn(B, 8, dtype=torch.float64)
        b = torch.randn(B, 8, dtype=torch.float64)

        J_a_batch, J_b_batch = jacobian_cross_product(a, b)
        assert J_a_batch.shape == (B, 8, 8)
        assert J_b_batch.shape == (B, 8, 8)

        for i in range(B):
            J_a_single, J_b_single = jacobian_cross_product(a[i], b[i])
            assert torch.allclose(J_a_batch[i], J_a_single, atol=1e-14)
            assert torch.allclose(J_b_batch[i], J_b_single, atol=1e-14)
