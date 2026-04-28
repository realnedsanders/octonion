"""Tests for octonion linear algebra utilities: left and right multiplication matrices.

Covers:
- left_mul_matrix(a) @ x = (a * x) for random a, x
- right_mul_matrix(b) @ x = (x * b) for random b, x
- Matrix shapes are [..., 8, 8]
"""

import torch
from hypothesis import given, settings

from octonion import Octonion
from octonion._linear_algebra import left_mul_matrix, right_mul_matrix
from tests.conftest import ATOL_FLOAT64, octonions


class TestLeftMulMatrix:
    """Tests for the left multiplication matrix L_a: a*x = L_a @ x."""

    @given(a=octonions(min_value=-1e3, max_value=1e3), x=octonions(min_value=-1e3, max_value=1e3))
    @settings(max_examples=200)
    def test_left_mul_matrix_property(self, a: Octonion, x: Octonion) -> None:
        """L_a @ x.components = (a * x).components for all a, x.

        Uses [-1e3, 1e3] range: the matrix multiply and einsum take
        slightly different computation paths, producing ~1e-12 rounding
        for O(1e3) inputs. The 1e-12 atol is suitable for this range.
        """
        L = left_mul_matrix(a)
        result = L @ x.components
        expected = (a * x).components
        # Use both rtol and atol: matrix multiply and einsum contract
        # indices in different orders, producing ~ULP differences for
        # intermediate products of magnitude ~1e6
        assert torch.allclose(result, expected, rtol=1e-10, atol=1e-9), (
            f"L_a @ x != a*x: max diff = {(result - expected).abs().max().item()}"
        )

    def test_left_mul_matrix_shape_single(self) -> None:
        """Left mul matrix has shape [8, 8] for single octonion."""
        a = Octonion(torch.randn(8, dtype=torch.float64))
        L = left_mul_matrix(a)
        assert L.shape == (8, 8), f"Expected shape (8, 8), got {L.shape}"

    def test_left_mul_matrix_shape_batched(self) -> None:
        """Left mul matrix has shape [N, 8, 8] for batched octonion."""
        a = Octonion(torch.randn(5, 8, dtype=torch.float64))
        L = left_mul_matrix(a)
        assert L.shape == (5, 8, 8), f"Expected shape (5, 8, 8), got {L.shape}"

    def test_left_mul_identity(self) -> None:
        """L_{e_0} is the 8x8 identity matrix."""
        e0 = Octonion(torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64))
        L = left_mul_matrix(e0)
        expected = torch.eye(8, dtype=torch.float64)
        assert torch.allclose(L, expected, atol=ATOL_FLOAT64)


class TestRightMulMatrix:
    """Tests for the right multiplication matrix R_b: x*b = R_b @ x."""

    @given(b=octonions(min_value=-1e3, max_value=1e3), x=octonions(min_value=-1e3, max_value=1e3))
    @settings(max_examples=200)
    def test_right_mul_matrix_property(self, b: Octonion, x: Octonion) -> None:
        """R_b @ x.components = (x * b).components for all b, x.

        Uses [-1e3, 1e3] range for numerical stability (see left_mul test).
        """
        R = right_mul_matrix(b)
        result = R @ x.components
        expected = (x * b).components
        # Use both rtol and atol: matrix multiply and einsum contract
        # indices in different orders, producing ~ULP differences
        assert torch.allclose(result, expected, rtol=1e-10, atol=1e-9), (
            f"R_b @ x != x*b: max diff = {(result - expected).abs().max().item()}"
        )

    def test_right_mul_matrix_shape_single(self) -> None:
        """Right mul matrix has shape [8, 8] for single octonion."""
        b = Octonion(torch.randn(8, dtype=torch.float64))
        R = right_mul_matrix(b)
        assert R.shape == (8, 8), f"Expected shape (8, 8), got {R.shape}"

    def test_right_mul_matrix_shape_batched(self) -> None:
        """Right mul matrix has shape [N, 8, 8] for batched octonion."""
        b = Octonion(torch.randn(5, 8, dtype=torch.float64))
        R = right_mul_matrix(b)
        assert R.shape == (5, 8, 8), f"Expected shape (5, 8, 8), got {R.shape}"

    def test_right_mul_identity(self) -> None:
        """R_{e_0} is the 8x8 identity matrix."""
        e0 = Octonion(torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64))
        R = right_mul_matrix(e0)
        expected = torch.eye(8, dtype=torch.float64)
        assert torch.allclose(R, expected, atol=ATOL_FLOAT64)
