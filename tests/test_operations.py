"""Tests for extended octonion operations: exp, log, commutator, inner product, cross product.

Covers:
- exp of zero and pure imaginary octonions
- log/exp round-trip for pure octonions
- Commutator definition, antisymmetry, and self-commutator
- Inner product symmetry, positive definiteness, and norm-squared relation
- Cross product antisymmetry and Fano plane consistency
"""

import pytest
import torch
from hypothesis import given, settings

from conftest import ATOL_FLOAT64, octonions, unit_octonions

from octonion import Octonion, PureOctonion
from octonion._operations import (
    commutator,
    cross_product,
    inner_product,
    octonion_exp,
    octonion_log,
)


class TestOctonionExp:
    """Tests for the octonion exponential map."""

    def test_exp_zero_is_identity(self) -> None:
        """exp(0) = [1, 0, 0, 0, 0, 0, 0, 0] (multiplicative identity)."""
        zero = Octonion(torch.zeros(8, dtype=torch.float64))
        result = octonion_exp(zero)
        expected = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
        assert torch.allclose(result.components, expected, atol=ATOL_FLOAT64)

    def test_exp_pure_imaginary_unit(self) -> None:
        """exp(e_i) = cos(1)*e_0 + sin(1)*e_i for each imaginary unit."""
        import math

        for i in range(1, 8):
            data = torch.zeros(8, dtype=torch.float64)
            data[i] = 1.0
            o = Octonion(data)
            result = octonion_exp(o)
            expected = torch.zeros(8, dtype=torch.float64)
            expected[0] = math.cos(1.0)
            expected[i] = math.sin(1.0)
            assert torch.allclose(result.components, expected, atol=1e-10), (
                f"exp(e_{i}) failed: got {result.components}, expected {expected}"
            )

    def test_log_exp_roundtrip_pure_octonions(self) -> None:
        """log(exp(a)) ~ a for pure octonions with ||v|| < pi within 1e-10.

        The log/exp roundtrip works only within the principal branch:
        ||imaginary part|| < pi (arccos range constraint).
        """
        import math

        torch.manual_seed(42)
        for _ in range(20):
            # Pure octonion: real part = 0, imaginary norm < pi
            data = torch.zeros(8, dtype=torch.float64)
            v = torch.randn(7, dtype=torch.float64)
            # Scale so ||v|| is in (0.1, pi - 0.1) for safety margin
            v_norm = torch.linalg.norm(v)
            if v_norm < 1e-10:
                v[0] = 1.0
                v_norm = torch.linalg.norm(v)
            target_norm = 0.1 + torch.rand(1).item() * (math.pi - 0.2)
            v = v / v_norm * target_norm
            data[1:] = v
            a = Octonion(data)
            result = octonion_log(octonion_exp(a))
            assert torch.allclose(result.components, a.components, atol=1e-10), (
                f"log(exp(a)) roundtrip failed: max diff = "
                f"{(result.components - a.components).abs().max().item()}"
            )

    def test_exp_log_roundtrip_near_identity(self) -> None:
        """exp(log(a)) ~ a for octonions near identity within 1e-10."""
        torch.manual_seed(42)
        for _ in range(20):
            # Near identity: [1 + eps, small, small, ...]
            data = torch.zeros(8, dtype=torch.float64)
            data[0] = 1.0 + torch.randn(1, dtype=torch.float64).item() * 0.1
            data[1:] = torch.randn(7, dtype=torch.float64) * 0.1
            a = Octonion(data)
            result = octonion_exp(octonion_log(a))
            assert torch.allclose(result.components, a.components, atol=1e-10), (
                f"exp(log(a)) roundtrip failed: max diff = "
                f"{(result.components - a.components).abs().max().item()}"
            )


class TestRawTensorCoercion:
    """Tests for auto-coercion of raw tensors to Octonion in exp/log."""

    def test_exp_raw_tensor(self) -> None:
        """octonion_exp(raw_tensor) returns raw tensor (same type as input)."""
        result = octonion_exp(torch.zeros(8, dtype=torch.float64))
        assert isinstance(result, torch.Tensor) and not isinstance(result, Octonion), (
            f"Expected raw Tensor, got {type(result)}"
        )
        expected = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
        assert torch.allclose(result, expected, atol=ATOL_FLOAT64)

    def test_log_raw_tensor(self) -> None:
        """octonion_log(raw_tensor) returns raw tensor (same type as input)."""
        identity = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = octonion_log(identity)
        assert isinstance(result, torch.Tensor) and not isinstance(result, Octonion), (
            f"Expected raw Tensor, got {type(result)}"
        )
        expected = torch.zeros(8, dtype=torch.float64)
        assert torch.allclose(result, expected, atol=ATOL_FLOAT64)

    def test_exp_log_roundtrip_raw_tensor(self) -> None:
        """exp(log(x)) roundtrip works with raw tensor input within principal branch."""
        torch.manual_seed(99)
        for _ in range(10):
            # Near identity with positive real part
            data = torch.zeros(8, dtype=torch.float64)
            data[0] = 1.0 + torch.randn(1, dtype=torch.float64).item() * 0.1
            data[1:] = torch.randn(7, dtype=torch.float64) * 0.1
            result = octonion_exp(octonion_log(data))
            assert isinstance(result, torch.Tensor) and not isinstance(result, Octonion)
            assert torch.allclose(result, data, atol=1e-10), (
                f"exp(log(x)) roundtrip failed on raw tensor: max diff = "
                f"{(result - data).abs().max().item()}"
            )

    def test_exp_still_works_with_octonion(self) -> None:
        """octonion_exp(Octonion(tensor)) still works (existing behavior preserved)."""
        o = Octonion(torch.zeros(8, dtype=torch.float64))
        result = octonion_exp(o)
        assert isinstance(result, Octonion)
        expected = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
        assert torch.allclose(result.components, expected, atol=ATOL_FLOAT64)

    def test_log_still_works_with_octonion(self) -> None:
        """octonion_log(Octonion(tensor)) still works (existing behavior preserved)."""
        identity = Octonion(torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64))
        result = octonion_log(identity)
        assert isinstance(result, Octonion)
        expected = torch.zeros(8, dtype=torch.float64)
        assert torch.allclose(result.components, expected, atol=ATOL_FLOAT64)


class TestCommutator:
    """Tests for the commutator [a, b] = ab - ba."""

    @given(a=octonions(), b=octonions())
    @settings(max_examples=200)
    def test_commutator_definition(self, a: Octonion, b: Octonion) -> None:
        """commutator(a, b) = a*b - b*a by definition."""
        result = commutator(a, b)
        expected = (a * b) - (b * a)
        assert torch.allclose(result.components, expected.components, atol=ATOL_FLOAT64)

    @given(a=octonions())
    @settings(max_examples=200)
    def test_commutator_self_is_zero(self, a: Octonion) -> None:
        """commutator(a, a) = 0 for all a."""
        result = commutator(a, a)
        assert torch.allclose(
            result.components, torch.zeros(8, dtype=torch.float64), atol=ATOL_FLOAT64
        )

    @given(a=octonions(), b=octonions())
    @settings(max_examples=200)
    def test_commutator_antisymmetry(self, a: Octonion, b: Octonion) -> None:
        """commutator(a, b) = -commutator(b, a)."""
        ab = commutator(a, b)
        ba = commutator(b, a)
        assert torch.allclose(ab.components, -ba.components, atol=ATOL_FLOAT64)


class TestInnerProduct:
    """Tests for the octonion inner product <a, b> = Re(a* * b)."""

    @given(a=octonions(), b=octonions())
    @settings(max_examples=200)
    def test_inner_product_definition(self, a: Octonion, b: Octonion) -> None:
        """inner_product(a, b) = Re(a.conjugate() * b)."""
        result = inner_product(a, b)
        expected = (a.conjugate() * b).real
        assert torch.allclose(result, expected, atol=ATOL_FLOAT64)

    @given(a=octonions(), b=octonions())
    @settings(max_examples=200)
    def test_inner_product_symmetric(self, a: Octonion, b: Octonion) -> None:
        """<a, b> = <b, a> (symmetry)."""
        ab = inner_product(a, b)
        ba = inner_product(b, a)
        assert torch.allclose(ab, ba, atol=ATOL_FLOAT64)

    @given(a=octonions())
    @settings(max_examples=200)
    def test_inner_product_positive_definite(self, a: Octonion) -> None:
        """<a, a> >= 0, with equality only for zero octonion."""
        result = inner_product(a, a)
        assert result >= -ATOL_FLOAT64  # >= 0 within tolerance
        if a.norm() > 1e-10:
            assert result > 0

    @given(a=octonions())
    @settings(max_examples=200)
    def test_inner_product_matches_norm_squared(self, a: Octonion) -> None:
        """<a, a> = |a|^2."""
        ip = inner_product(a, a)
        norm_sq = a.norm() ** 2
        assert torch.allclose(ip, norm_sq, atol=ATOL_FLOAT64)


class TestCrossProduct:
    """Tests for the 7D cross product on pure imaginary octonions."""

    @given(a=octonions(min_value=-1e3, max_value=1e3), b=octonions(min_value=-1e3, max_value=1e3))
    @settings(max_examples=200)
    def test_cross_product_antisymmetry(self, a: Octonion, b: Octonion) -> None:
        """cross_product(a, b) = -cross_product(b, a) (antisymmetry).

        Uses looser tolerance (1e-9) because cross product involves
        multiplication of components that can be O(1e3), producing
        floating-point rounding at the ~1e-12 level.
        """
        ab = cross_product(a, b)
        ba = cross_product(b, a)
        assert torch.allclose(ab.components, -ba.components, atol=1e-9)

    def test_cross_product_basis_elements_fano(self) -> None:
        """Cross product of basis elements matches Fano plane structure.

        For pure imaginary basis elements e_i, e_j:
        cross(e_i, e_j) = e_k if (i, j, k) is a Fano triple
        """
        from octonion._fano import FANO_PLANE

        for i, j, k in FANO_PLANE.triples:
            ei = torch.zeros(8, dtype=torch.float64)
            ei[i] = 1.0
            ej = torch.zeros(8, dtype=torch.float64)
            ej[j] = 1.0
            ek = torch.zeros(8, dtype=torch.float64)
            ek[k] = 1.0

            result = cross_product(Octonion(ei), Octonion(ej))
            # cross(e_i, e_j) = Im(e_i * e_j) = e_k (the Fano product)
            assert torch.allclose(result.components, ek, atol=ATOL_FLOAT64), (
                f"cross(e_{i}, e_{j}) should be e_{k}, got {result.components}"
            )

    def test_cross_product_result_is_pure(self) -> None:
        """Cross product output has zero real part."""
        torch.manual_seed(42)
        for _ in range(10):
            a = Octonion(torch.randn(8, dtype=torch.float64))
            b = Octonion(torch.randn(8, dtype=torch.float64))
            result = cross_product(a, b)
            assert abs(result.real.item()) < ATOL_FLOAT64, (
                f"Cross product real part should be 0, got {result.real.item()}"
            )
