"""Tests for the octonionic GHR calculus (involution-basis decomposition).

The central claim of octonion.calculus._ghr is the exact differential
identity

    df = sum_{a=0}^{7} (df/do^{sigma_a}) * sigma_a(do)

for the eight involutions sigma_a(x) = e_a x e_a^{-1}. These tests verify
the identity itself (not just internal consistency), the uniqueness
structure, and the special cases that pin down the convention.
"""

from __future__ import annotations

import hypothesis.strategies as st
import torch
from hypothesis import given, settings

from octonion._multiplication import octonion_mul
from octonion.calculus._ghr import (
    _S_INV,
    INVOLUTION_SIGNS,
    conjugate_derivative,
    ghr_conjugate_derivatives_from_jacobian,
    ghr_derivative,
    ghr_derivatives_from_jacobian,
    involute,
    reconstruct_jacobian,
)


def _basis(a: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    e = torch.zeros(8, dtype=dtype)
    e[a] = 1.0
    return e


@st.composite
def jacobians(draw: st.DrawFn) -> torch.Tensor:
    """Random real 8x8 Jacobians with bounded entries."""
    elements = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False)
    vals = [draw(elements) for _ in range(64)]
    return torch.tensor(vals, dtype=torch.float64).reshape(8, 8)


@st.composite
def octonion_vectors(draw: st.DrawFn) -> torch.Tensor:
    elements = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False)
    return torch.tensor([draw(elements) for _ in range(8)], dtype=torch.float64)


class TestInvolutions:
    def test_sign_matrix_inverse(self) -> None:
        """S @ S^{-1} == I for the closed-form inverse used in the solver."""
        eye = torch.eye(8, dtype=torch.float64)
        assert torch.allclose(INVOLUTION_SIGNS @ _S_INV, eye, atol=1e-14)

    def test_involute_matches_conjugation_by_basis_unit(self) -> None:
        """involute(x, a) == e_a x e_a^{-1}, well-defined by alternativity."""
        torch.manual_seed(0)
        x = torch.randn(8, dtype=torch.float64)
        for a in range(8):
            e = _basis(a)
            e_inv = e.clone()
            if a >= 1:
                e_inv[a] = -1.0
            left_first = octonion_mul(octonion_mul(e, x), e_inv)
            right_first = octonion_mul(e, octonion_mul(x, e_inv))
            # Both parenthesizations agree (Artin), and match the sign table
            assert torch.allclose(left_first, right_first, atol=1e-14)
            assert torch.allclose(involute(x, a), left_first, atol=1e-14)

    def test_involutions_are_involutions(self) -> None:
        """sigma_a(sigma_a(x)) == x."""
        torch.manual_seed(1)
        x = torch.randn(3, 8, dtype=torch.float64)
        for a in range(8):
            assert torch.allclose(involute(involute(x, a), a), x)

    def test_involutions_commute_with_conjugation(self) -> None:
        """sigma_a(x*) == sigma_a(x)* — required by the conjugate variant."""
        torch.manual_seed(2)
        x = torch.randn(8, dtype=torch.float64)
        conj = torch.tensor([1.0] + [-1.0] * 7, dtype=torch.float64)
        for a in range(8):
            assert torch.allclose(involute(x * conj, a), involute(x, a) * conj)


class TestFundamentalIdentity:
    @given(J=jacobians(), do=octonion_vectors())
    @settings(max_examples=500, deadline=None)
    def test_differential_identity(self, J: torch.Tensor, do: torch.Tensor) -> None:
        """df = sum_a A_a * sigma_a(do) reproduces J @ do for ANY Jacobian."""
        A = ghr_derivatives_from_jacobian(J)
        df_true = J @ do
        df_ghr = torch.zeros(8, dtype=torch.float64)
        for a in range(8):
            df_ghr = df_ghr + octonion_mul(A[a], involute(do, a))
        torch.testing.assert_close(df_ghr, df_true, rtol=1e-9, atol=1e-7)

    @given(J=jacobians(), do=octonion_vectors())
    @settings(max_examples=500, deadline=None)
    def test_conjugate_differential_identity(self, J: torch.Tensor, do: torch.Tensor) -> None:
        """df = sum_a A*_a * sigma_a(do*) — the conjugate-basis variant."""
        A = ghr_conjugate_derivatives_from_jacobian(J)
        conj = torch.tensor([1.0] + [-1.0] * 7, dtype=torch.float64)
        df_true = J @ do
        df_ghr = torch.zeros(8, dtype=torch.float64)
        for a in range(8):
            df_ghr = df_ghr + octonion_mul(A[a], involute(do * conj, a))
        torch.testing.assert_close(df_ghr, df_true, rtol=1e-9, atol=1e-7)

    @given(J=jacobians())
    @settings(max_examples=500, deadline=None)
    def test_roundtrip(self, J: torch.Tensor) -> None:
        """Decompose-then-reconstruct is the identity on Jacobians."""
        A = ghr_derivatives_from_jacobian(J)
        torch.testing.assert_close(reconstruct_jacobian(A), J, rtol=1e-10, atol=1e-9)
        Ac = ghr_conjugate_derivatives_from_jacobian(J)
        torch.testing.assert_close(
            reconstruct_jacobian(Ac, conjugate=True), J, rtol=1e-10, atol=1e-9
        )

    def test_batched(self) -> None:
        """Batch dimensions are preserved through decompose/reconstruct."""
        torch.manual_seed(3)
        J = torch.randn(4, 5, 8, 8, dtype=torch.float64)
        A = ghr_derivatives_from_jacobian(J)
        assert A.shape == (4, 5, 8, 8)
        assert torch.allclose(reconstruct_jacobian(A), J, atol=1e-12)


class TestSpecialCases:
    def test_identity_map(self) -> None:
        """f(o) = o has A_0 = 1 and all other GHR derivatives zero."""
        A = ghr_derivatives_from_jacobian(torch.eye(8, dtype=torch.float64))
        expected = torch.zeros(8, 8, dtype=torch.float64)
        expected[0, 0] = 1.0
        assert torch.allclose(A, expected, atol=1e-14)

    def test_conjugation_map(self) -> None:
        """f(o) = o* gives A_0 = -1/6, A_a = 1/6: the classical identity
        o* = (-o + sum_a sigma_a(o)) / 6."""
        D_conj = torch.diag(torch.tensor([1.0] + [-1.0] * 7, dtype=torch.float64))
        A = ghr_derivatives_from_jacobian(D_conj)
        expected = torch.zeros(8, 8, dtype=torch.float64)
        expected[0, 0] = -1.0 / 6.0
        expected[1:, 0] = 1.0 / 6.0
        assert torch.allclose(A, expected, atol=1e-14)

    def test_left_multiplication_is_pure_a0(self) -> None:
        """f(o) = c * o has df = c do exactly, so A_0 = c and the rest 0.

        This is the uniqueness property doing real work: left multiplication
        needs no involution terms.
        """
        torch.manual_seed(4)
        c = torch.randn(8, dtype=torch.float64)
        from octonion.calculus import jacobian_mul

        # f(o) = c * o: Jacobian w.r.t. o is the second return of jacobian_mul
        _, J_o = jacobian_mul(c, torch.randn(8, dtype=torch.float64))
        A = ghr_derivatives_from_jacobian(J_o)
        assert torch.allclose(A[0], c, atol=1e-12)
        assert torch.allclose(A[1:], torch.zeros(7, 8, dtype=torch.float64), atol=1e-12)

    @given(g=octonion_vectors())
    @settings(max_examples=200, deadline=None)
    def test_real_valued_shortcuts_match_full_machinery(self, g: torch.Tensor) -> None:
        """ghr_derivative / conjugate_derivative equal the a=0 rows of the
        full decomposition of the row-Jacobian of a real-valued function."""
        J = torch.zeros(8, 8, dtype=torch.float64)
        J[0, :] = g
        A = ghr_derivatives_from_jacobian(J)
        torch.testing.assert_close(ghr_derivative(g), A[0], rtol=1e-12, atol=1e-12)
        Ac = ghr_conjugate_derivatives_from_jacobian(J)
        torch.testing.assert_close(conjugate_derivative(g), Ac[0], rtol=1e-12, atol=1e-12)

    def test_float32_roundtrip(self) -> None:
        """The decomposition works in float32 with appropriate tolerance."""
        torch.manual_seed(5)
        J = torch.randn(8, 8, dtype=torch.float32)
        A = ghr_derivatives_from_jacobian(J)
        assert A.dtype == torch.float32
        assert torch.allclose(reconstruct_jacobian(A), J, atol=1e-5)
