"""Tests for octonionic autograd Functions, custom gradient checking, and SC-1.

Verifies that torch.autograd.Function subclasses for all 7 octonionic
primitives produce correct forward output, correct backward gradients,
pass torch.autograd.gradcheck, and support double backward (create_graph=True)
via gradgradcheck.

Also tests the custom octonion_gradcheck utility that reports per-component
errors and validates both Wirtinger derivatives.

SC-1: Single OctonionLinear layer gradient check at float64 with rel error < 1e-5.

All gradient checks use float64 for finite-difference precision.
Inputs are kept in [-2, 2] range for numerical stability of transcendental functions.

Primitives tested:
  1. OctonionMulFunction
  2. OctonionExpFunction
  3. OctonionLogFunction
  4. OctonionConjugateFunction
  5. OctonionInverseFunction
  6. OctonionInnerProductFunction
  7. OctonionCrossProductFunction
"""

import torch
import torch.autograd
from conftest import octonion_tensors
from hypothesis import assume, given, settings

from octonion._linear import OctonionLinear
from octonion._multiplication import octonion_mul
from octonion._octonion import Octonion
from octonion._operations import (
    cross_product,
    inner_product,
    octonion_exp,
    octonion_log,
)
from octonion.calculus import (
    jacobian_conjugate,
    jacobian_cross_product,
    jacobian_exp,
    jacobian_inverse,
    jacobian_log,
    jacobian_mul,
)
from octonion.calculus._autograd_functions import (
    OctonionConjugateFunction,
    OctonionCrossProductFunction,
    OctonionExpFunction,
    OctonionInnerProductFunction,
    OctonionInverseFunction,
    OctonionLogFunction,
    OctonionMulFunction,
)
from octonion.calculus._gradcheck import (
    octonion_gradcheck,
    octonion_gradgradcheck,
)

# =============================================================================
# Forward correctness tests
# =============================================================================


class TestAutogradForward:
    """Each autograd Function produces the same output as the original function."""

    @given(
        a=octonion_tensors(min_value=-2, max_value=2),
        b=octonion_tensors(min_value=-2, max_value=2),
    )
    @settings(max_examples=20, deadline=None)
    def test_mul_forward(self, a: torch.Tensor, b: torch.Tensor) -> None:
        expected = octonion_mul(a, b)
        result = OctonionMulFunction.apply(a, b)
        assert torch.allclose(result, expected, atol=1e-14)

    @given(o=octonion_tensors(min_value=-2, max_value=2))
    @settings(max_examples=20, deadline=None)
    def test_exp_forward(self, o: torch.Tensor) -> None:
        expected = octonion_exp(o)
        result = OctonionExpFunction.apply(o)
        assert torch.allclose(result, expected, atol=1e-14)

    @given(o=octonion_tensors(min_value=-2, max_value=2))
    @settings(max_examples=20, deadline=None)
    def test_log_forward(self, o: torch.Tensor) -> None:
        q_norm = torch.sqrt(torch.sum(o**2))
        v_norm = torch.sqrt(torch.sum(o[1:] ** 2))
        assume(q_norm.item() > 0.1)
        assume(v_norm.item() > 0.1)
        expected = octonion_log(o)
        result = OctonionLogFunction.apply(o)
        assert torch.allclose(result, expected, atol=1e-14)

    @given(o=octonion_tensors(min_value=-2, max_value=2))
    @settings(max_examples=20, deadline=None)
    def test_conjugate_forward(self, o: torch.Tensor) -> None:
        expected = Octonion(o).conjugate().components
        result = OctonionConjugateFunction.apply(o)
        assert torch.allclose(result, expected, atol=1e-14)

    @given(o=octonion_tensors(min_value=-2, max_value=2))
    @settings(max_examples=20, deadline=None)
    def test_inverse_forward(self, o: torch.Tensor) -> None:
        n = torch.sqrt(torch.sum(o**2))
        assume(n.item() > 0.5)
        expected = Octonion(o).inverse().components
        result = OctonionInverseFunction.apply(o)
        assert torch.allclose(result, expected, atol=1e-14)

    @given(
        a=octonion_tensors(min_value=-2, max_value=2),
        b=octonion_tensors(min_value=-2, max_value=2),
    )
    @settings(max_examples=20, deadline=None)
    def test_inner_product_forward(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> None:
        expected = inner_product(Octonion(a), Octonion(b))
        result = OctonionInnerProductFunction.apply(a, b)
        assert torch.allclose(result, expected, atol=1e-14)

    @given(
        a=octonion_tensors(min_value=-2, max_value=2),
        b=octonion_tensors(min_value=-2, max_value=2),
    )
    @settings(max_examples=20, deadline=None)
    def test_cross_product_forward(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> None:
        expected = cross_product(Octonion(a), Octonion(b)).components
        result = OctonionCrossProductFunction.apply(a, b)
        assert torch.allclose(result, expected, atol=1e-14)


# =============================================================================
# Backward correctness: autograd backward matches analytic Jacobian
# =============================================================================


class TestAutogradBackward:
    """Each Function's backward matches the analytic Jacobian: grad = J^T @ grad_output."""

    def test_mul_backward(self) -> None:
        a = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(8, dtype=torch.float64, requires_grad=True)
        grad_out = torch.randn(8, dtype=torch.float64)

        result = OctonionMulFunction.apply(a, b)
        result.backward(grad_out)

        J_a, J_b = jacobian_mul(a.detach(), b.detach())
        expected_grad_a = torch.einsum("ki,k->i", J_a, grad_out)
        expected_grad_b = torch.einsum("kj,k->j", J_b, grad_out)

        assert torch.allclose(a.grad, expected_grad_a, atol=1e-12)
        assert torch.allclose(b.grad, expected_grad_b, atol=1e-12)

    def test_exp_backward(self) -> None:
        o = torch.randn(8, dtype=torch.float64, requires_grad=True)
        grad_out = torch.randn(8, dtype=torch.float64)

        result = OctonionExpFunction.apply(o)
        result.backward(grad_out)

        J = jacobian_exp(o.detach())
        expected_grad = torch.einsum("ki,k->i", J, grad_out)
        assert torch.allclose(o.grad, expected_grad, atol=1e-10)

    def test_log_backward(self) -> None:
        # Use input with good norm and imaginary norm
        o = torch.randn(8, dtype=torch.float64)
        o = o / torch.linalg.norm(o) * 2.0  # scale to norm 2
        o = o.clone().requires_grad_(True)
        grad_out = torch.randn(8, dtype=torch.float64)

        result = OctonionLogFunction.apply(o)
        result.backward(grad_out)

        J = jacobian_log(o.detach())
        expected_grad = torch.einsum("ki,k->i", J, grad_out)
        assert torch.allclose(o.grad, expected_grad, atol=1e-10)

    def test_conjugate_backward(self) -> None:
        o = torch.randn(8, dtype=torch.float64, requires_grad=True)
        grad_out = torch.randn(8, dtype=torch.float64)

        result = OctonionConjugateFunction.apply(o)
        result.backward(grad_out)

        J = jacobian_conjugate(o.detach())
        expected_grad = torch.einsum("ki,k->i", J, grad_out)
        assert torch.allclose(o.grad, expected_grad, atol=1e-14)

    def test_inverse_backward(self) -> None:
        o = torch.randn(8, dtype=torch.float64)
        o = o / torch.linalg.norm(o) * 2.0
        o = o.clone().requires_grad_(True)
        grad_out = torch.randn(8, dtype=torch.float64)

        result = OctonionInverseFunction.apply(o)
        result.backward(grad_out)

        J = jacobian_inverse(o.detach())
        expected_grad = torch.einsum("ki,k->i", J, grad_out)
        assert torch.allclose(o.grad, expected_grad, atol=1e-10)

    def test_inner_product_backward(self) -> None:
        a = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(8, dtype=torch.float64, requires_grad=True)

        result = OctonionInnerProductFunction.apply(a, b)
        # result is scalar (shape []), so backward with no grad_out argument
        result.backward()

        # For inner_product, grad_a = b, grad_b = a (grad_output = 1 scalar)
        assert torch.allclose(a.grad, b.detach(), atol=1e-14)
        assert torch.allclose(b.grad, a.detach(), atol=1e-14)

    def test_cross_product_backward(self) -> None:
        a = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(8, dtype=torch.float64, requires_grad=True)
        grad_out = torch.randn(8, dtype=torch.float64)

        result = OctonionCrossProductFunction.apply(a, b)
        result.backward(grad_out)

        J_a, J_b = jacobian_cross_product(a.detach(), b.detach())
        expected_grad_a = torch.einsum("ki,k->i", J_a, grad_out)
        expected_grad_b = torch.einsum("kj,k->j", J_b, grad_out)

        assert torch.allclose(a.grad, expected_grad_a, atol=1e-12)
        assert torch.allclose(b.grad, expected_grad_b, atol=1e-12)


# =============================================================================
# torch.autograd.gradcheck (finite-difference verification)
# =============================================================================


class TestGradcheck:
    """torch.autograd.gradcheck passes for each Function at eps=1e-6, atol=1e-5."""

    def test_gradcheck_mul(self) -> None:
        a = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            OctonionMulFunction.apply, (a, b), eps=1e-6, atol=1e-5, rtol=1e-3
        )

    def test_gradcheck_exp(self) -> None:
        o = torch.randn(8, dtype=torch.float64, requires_grad=True) * 0.5
        assert torch.autograd.gradcheck(
            OctonionExpFunction.apply, (o,), eps=1e-6, atol=1e-5, rtol=1e-3
        )

    def test_gradcheck_log(self) -> None:
        o = torch.randn(8, dtype=torch.float64) + 1.0
        o = o.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(
            OctonionLogFunction.apply, (o,), eps=1e-6, atol=1e-5, rtol=1e-3
        )

    def test_gradcheck_conjugate(self) -> None:
        o = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            OctonionConjugateFunction.apply,
            (o,),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-3,
        )

    def test_gradcheck_inverse(self) -> None:
        o = torch.randn(8, dtype=torch.float64) + 1.0
        o = o.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(
            OctonionInverseFunction.apply,
            (o,),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-3,
        )

    def test_gradcheck_inner_product(self) -> None:
        a = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            OctonionInnerProductFunction.apply,
            (a, b),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-3,
        )

    def test_gradcheck_cross_product(self) -> None:
        a = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            OctonionCrossProductFunction.apply,
            (a, b),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-3,
        )


# =============================================================================
# torch.autograd.gradgradcheck (create_graph=True support)
# =============================================================================


class TestGradgradcheck:
    """torch.autograd.gradgradcheck passes for each Function."""

    def test_gradgradcheck_mul(self) -> None:
        a = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradgradcheck(
            OctonionMulFunction.apply, (a, b), eps=1e-6, atol=1e-5, rtol=1e-3
        )

    def test_gradgradcheck_exp(self) -> None:
        o = torch.randn(8, dtype=torch.float64, requires_grad=True) * 0.5
        assert torch.autograd.gradgradcheck(
            OctonionExpFunction.apply, (o,), eps=1e-6, atol=1e-5, rtol=1e-3
        )

    def test_gradgradcheck_log(self) -> None:
        o = torch.randn(8, dtype=torch.float64) + 1.0
        o = o.clone().requires_grad_(True)
        assert torch.autograd.gradgradcheck(
            OctonionLogFunction.apply, (o,), eps=1e-6, atol=1e-5, rtol=1e-3
        )

    def test_gradgradcheck_conjugate(self) -> None:
        o = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradgradcheck(
            OctonionConjugateFunction.apply,
            (o,),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-3,
        )

    def test_gradgradcheck_inverse(self) -> None:
        o = torch.randn(8, dtype=torch.float64) + 1.0
        o = o.clone().requires_grad_(True)
        assert torch.autograd.gradgradcheck(
            OctonionInverseFunction.apply,
            (o,),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-3,
        )

    def test_gradgradcheck_inner_product(self) -> None:
        a = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradgradcheck(
            OctonionInnerProductFunction.apply,
            (a, b),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-3,
        )

    def test_gradgradcheck_cross_product(self) -> None:
        a = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradgradcheck(
            OctonionCrossProductFunction.apply,
            (a, b),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-3,
        )


# =============================================================================
# Batched input handling
# =============================================================================


class TestBatchedAutograd:
    """All autograd Functions handle batched [..., 8] inputs correctly."""

    def test_mul_batched(self) -> None:
        a = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
        result = OctonionMulFunction.apply(a, b)
        assert result.shape == (3, 8)
        loss = result.sum()
        loss.backward()
        assert a.grad is not None and a.grad.shape == (3, 8)
        assert b.grad is not None and b.grad.shape == (3, 8)

    def test_exp_batched(self) -> None:
        o = (torch.randn(3, 8, dtype=torch.float64) * 0.5).requires_grad_(True)
        result = OctonionExpFunction.apply(o)
        assert result.shape == (3, 8)
        loss = result.sum()
        loss.backward()
        assert o.grad is not None and o.grad.shape == (3, 8)

    def test_inner_product_batched(self) -> None:
        a = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
        result = OctonionInnerProductFunction.apply(a, b)
        assert result.shape == (3,)
        loss = result.sum()
        loss.backward()
        assert a.grad is not None and a.grad.shape == (3, 8)
        assert b.grad is not None and b.grad.shape == (3, 8)


# =============================================================================
# Custom octonion gradcheck tests
# =============================================================================


class TestOctonionGradcheck:
    """Test the custom octonion gradcheck utility on all 7 primitives."""

    def test_gradcheck_mul(self) -> None:
        a = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(8, dtype=torch.float64, requires_grad=True)
        result = octonion_gradcheck(OctonionMulFunction.apply, (a, b))
        assert result["passed"] is True
        assert "per_component_errors" in result
        assert "wirtinger_passed" in result
        assert result["wirtinger_passed"] is True

    def test_gradcheck_exp(self) -> None:
        o = (torch.randn(8, dtype=torch.float64) * 0.5).requires_grad_(True)
        result = octonion_gradcheck(OctonionExpFunction.apply, (o,))
        assert result["passed"] is True
        assert result["wirtinger_passed"] is True

    def test_gradcheck_log(self) -> None:
        o = (torch.randn(8, dtype=torch.float64) + 1.0).requires_grad_(True)
        result = octonion_gradcheck(OctonionLogFunction.apply, (o,))
        assert result["passed"] is True
        assert result["wirtinger_passed"] is True

    def test_gradcheck_conjugate(self) -> None:
        o = torch.randn(8, dtype=torch.float64, requires_grad=True)
        result = octonion_gradcheck(OctonionConjugateFunction.apply, (o,))
        assert result["passed"] is True
        assert result["wirtinger_passed"] is True

    def test_gradcheck_inverse(self) -> None:
        o = (torch.randn(8, dtype=torch.float64) + 1.0).requires_grad_(True)
        result = octonion_gradcheck(OctonionInverseFunction.apply, (o,))
        assert result["passed"] is True
        assert result["wirtinger_passed"] is True

    def test_gradcheck_inner_product(self) -> None:
        a = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(8, dtype=torch.float64, requires_grad=True)
        result = octonion_gradcheck(OctonionInnerProductFunction.apply, (a, b))
        assert result["passed"] is True
        # Inner product returns scalar, Wirtinger check not applicable for 1D output
        # but the function should still report on what it can

    def test_gradcheck_cross_product(self) -> None:
        a = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(8, dtype=torch.float64, requires_grad=True)
        result = octonion_gradcheck(OctonionCrossProductFunction.apply, (a, b))
        assert result["passed"] is True
        assert result["wirtinger_passed"] is True

    def test_gradcheck_reports_per_component_errors(self) -> None:
        """Verify that per-component errors are reported for each input."""
        a = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(8, dtype=torch.float64, requires_grad=True)
        result = octonion_gradcheck(OctonionMulFunction.apply, (a, b))
        # Should have per-component error info
        assert len(result["per_component_errors"]) == 2  # two inputs
        for comp_errors in result["per_component_errors"]:
            assert len(comp_errors) == 8  # 8 components per octonion input


class TestOctonionGradcheckDetectsWrong:
    """Verify gradcheck detects wrong backward implementations."""

    def test_gradcheck_detects_wrong_backward(self) -> None:
        """Create a deliberately wrong backward and verify detection."""

        class WrongMulFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, a, b):
                ctx.save_for_backward(a, b)
                return octonion_mul(a, b)

            @staticmethod
            def backward(ctx, grad_output):
                a, b = ctx.saved_tensors
                # Deliberately wrong: swap a and b in gradient computation
                return grad_output * 2.0, grad_output * 3.0

        a = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(8, dtype=torch.float64, requires_grad=True)
        result = octonion_gradcheck(WrongMulFunction.apply, (a, b))
        assert result["passed"] is False


class TestOctonionGradgradcheck:
    """octonion_gradgradcheck passes for mul and exp (create_graph=True)."""

    def test_gradgradcheck_mul(self) -> None:
        a = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(8, dtype=torch.float64, requires_grad=True)
        result = octonion_gradgradcheck(OctonionMulFunction.apply, (a, b))
        assert result["passed"] is True

    def test_gradgradcheck_exp(self) -> None:
        o = (torch.randn(8, dtype=torch.float64) * 0.5).requires_grad_(True)
        result = octonion_gradgradcheck(OctonionExpFunction.apply, (o,))
        assert result["passed"] is True


# =============================================================================
# SC-1: OctonionLinear single layer gradient check
# =============================================================================


class TestGradcheckOctonionLinear:
    """SC-1: Single OctonionLinear layer gradient check at float64, rel error < 1e-5."""

    def test_gradcheck_octonion_linear(self) -> None:
        """Create an OctonionLinear(dtype=float64) layer, verify gradient check passes.

        This is Success Criterion 1 from the ROADMAP.
        """
        layer = OctonionLinear(dtype=torch.float64)

        x = torch.randn(8, dtype=torch.float64, requires_grad=True)

        result = octonion_gradcheck(layer.forward, (x,))

        assert result["passed"] is True, (
            f"SC-1 FAILED: OctonionLinear gradcheck failed.\n"
            f"max_abs_error: {result['max_abs_error']:.2e}\n"
            f"max_rel_error: {result['max_rel_error']:.2e}"
        )
        assert result["max_rel_error"] < 1e-5, (
            f"SC-1 FAILED: rel error {result['max_rel_error']:.2e} >= 1e-5"
        )
