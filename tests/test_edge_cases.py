"""Edge case tests: zero, identity, near-zero, large magnitude, pure imaginary, basis elements.

Verifies graceful handling of extreme inputs without NaN, Inf, or silent errors.
"""

import pytest
import torch

from octonion import Octonion, PureOctonion
from octonion._multiplication import octonion_mul


class TestZeroOctonion:
    """Tests for the zero octonion [0, 0, ..., 0]."""

    def test_zero_norm(self) -> None:
        """Zero octonion has norm 0."""
        zero = Octonion(torch.zeros(8, dtype=torch.float64))
        assert zero.norm().item() == 0.0

    def test_zero_conjugate(self) -> None:
        """Conjugate of zero is zero."""
        zero = Octonion(torch.zeros(8, dtype=torch.float64))
        conj = zero.conjugate()
        assert torch.equal(conj.components, torch.zeros(8, dtype=torch.float64))

    def test_mul_by_zero(self) -> None:
        """a * zero = zero for any a."""
        zero = Octonion(torch.zeros(8, dtype=torch.float64))
        a = Octonion(torch.randn(8, dtype=torch.float64))
        result = a * zero
        assert torch.allclose(
            result.components, torch.zeros(8, dtype=torch.float64), atol=1e-12
        )

    def test_zero_mul_a(self) -> None:
        """zero * a = zero for any a."""
        zero = Octonion(torch.zeros(8, dtype=torch.float64))
        a = Octonion(torch.randn(8, dtype=torch.float64))
        result = zero * a
        assert torch.allclose(
            result.components, torch.zeros(8, dtype=torch.float64), atol=1e-12
        )

    def test_inverse_raises(self) -> None:
        """Inverse of zero raises ValueError."""
        zero = Octonion(torch.zeros(8, dtype=torch.float64))
        with pytest.raises(ValueError, match="Cannot invert zero octonion"):
            zero.inverse()


class TestIdentityOctonion:
    """Tests for the identity [1, 0, 0, ..., 0]."""

    def test_identity_norm(self) -> None:
        """Identity octonion has norm 1."""
        e0 = Octonion(torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64))
        assert abs(e0.norm().item() - 1.0) < 1e-15

    def test_left_identity(self) -> None:
        """e0 * a = a for any a."""
        e0 = Octonion(torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64))
        torch.manual_seed(42)
        a = Octonion(torch.randn(8, dtype=torch.float64))
        result = e0 * a
        assert torch.allclose(result.components, a.components, atol=1e-12)

    def test_right_identity(self) -> None:
        """a * e0 = a for any a."""
        e0 = Octonion(torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64))
        torch.manual_seed(42)
        a = Octonion(torch.randn(8, dtype=torch.float64))
        result = a * e0
        assert torch.allclose(result.components, a.components, atol=1e-12)

    def test_inverse_is_self(self) -> None:
        """Inverse of identity is identity."""
        e0 = Octonion(torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64))
        inv = e0.inverse()
        assert torch.allclose(inv.components, e0.components, atol=1e-12)


class TestNearZeroStability:
    """Tests for near-zero octonions (1e-15 magnitude)."""

    def test_near_zero_norm(self) -> None:
        """Near-zero octonion norm is non-NaN and non-Inf."""
        data = torch.randn(8, dtype=torch.float64) * 1e-15
        o = Octonion(data)
        n = o.norm()
        assert not torch.isnan(n)
        assert not torch.isinf(n)
        assert n.item() > 0

    def test_near_zero_conjugate(self) -> None:
        """Conjugate of near-zero octonion produces no NaN or Inf."""
        data = torch.randn(8, dtype=torch.float64) * 1e-15
        o = Octonion(data)
        conj = o.conjugate()
        assert not torch.any(torch.isnan(conj.components))
        assert not torch.any(torch.isinf(conj.components))

    def test_near_zero_mul(self) -> None:
        """Multiplication with near-zero octonion produces no NaN or Inf."""
        data = torch.randn(8, dtype=torch.float64) * 1e-15
        o = Octonion(data)
        a = Octonion(torch.randn(8, dtype=torch.float64))
        result = a * o
        assert not torch.any(torch.isnan(result.components))
        assert not torch.any(torch.isinf(result.components))


class TestLargeMagnitudePrecision:
    """Tests for large magnitude octonions (1e10)."""

    def test_large_mul_no_overflow(self) -> None:
        """Multiplication of moderate-large octonions doesn't overflow."""
        # Use 1e10 magnitude (product will be ~1e20 which fits in float64)
        data_a = torch.randn(8, dtype=torch.float64) * 1e10
        data_b = torch.randn(8, dtype=torch.float64)
        result = octonion_mul(data_a, data_b)
        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))

    def test_large_inverse_relative_precision(self) -> None:
        """Inverse of large octonion maintains machine-epsilon precision.

        The inverse x^{-1} = conj(x)/|x|^2 is well-conditioned at all scales
        because the product x * x^{-1} normalizes out the magnitude. Empirically,
        a * a^{-1} achieves ~1e-16 absolute error (machine epsilon) for ||a|| up
        to 1e15. We use 1e-12 tolerance to match the project standard (ATOL_FLOAT64)
        with ample headroom.
        """
        torch.manual_seed(42)
        for scale in [1e5, 1e10, 1e15]:
            data = torch.randn(8, dtype=torch.float64) * scale
            a = Octonion(data)
            inv = a.inverse()
            product = (a * inv).components
            identity = torch.zeros(8, dtype=torch.float64)
            identity[0] = 1.0
            diff = (product - identity).abs().max().item()
            assert diff < 1e-12, (
                f"Large octonion inverse (||a||~{scale:.0e}): "
                f"max |a*a^{{-1}} - 1| = {diff:.2e}, expected < 1e-12"
            )

    def test_large_norm_correct(self) -> None:
        """Norm of large octonion is correct order of magnitude."""
        data = torch.ones(8, dtype=torch.float64) * 1e10
        o = Octonion(data)
        expected_norm = (8.0**0.5) * 1e10
        actual_norm = o.norm().item()
        assert abs(actual_norm - expected_norm) / expected_norm < 1e-12


class TestPureImaginaryProperties:
    """Tests for pure imaginary octonions."""

    def test_pure_imaginary_real_zero(self) -> None:
        """Pure imaginary octonion has real part zero."""
        data = torch.randn(8, dtype=torch.float64)
        pure = PureOctonion(data)
        assert pure.real.item() == 0.0

    def test_pure_imaginary_conjugation(self) -> None:
        """Conjugation of pure imaginary flips sign (since real=0, conj = -a)."""
        data = torch.zeros(8, dtype=torch.float64)
        data[1:] = torch.randn(7, dtype=torch.float64)
        pure = PureOctonion(data)
        conj = pure.conjugate()
        assert conj.real.item() == 0.0
        assert torch.allclose(conj.imag, -pure.imag)

    def test_pure_imaginary_norm(self) -> None:
        """Norm of pure imaginary is sqrt of sum of squared imaginary components."""
        data = torch.zeros(8, dtype=torch.float64)
        data[1:] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=torch.float64)
        pure = PureOctonion(data)
        expected = (1 + 4 + 9 + 16 + 25 + 36 + 49) ** 0.5
        assert abs(pure.norm().item() - expected) < 1e-12


class TestBasisElementSquares:
    """Tests for basis element squaring: e_i * e_i = -e_0 for i in 1..7."""

    def test_basis_element_squares(self) -> None:
        """e_i * e_i = -[1, 0, ..., 0] for all imaginary basis elements."""
        neg_e0 = torch.zeros(8, dtype=torch.float64)
        neg_e0[0] = -1.0
        for i in range(1, 8):
            data = torch.zeros(8, dtype=torch.float64)
            data[i] = 1.0
            ei = Octonion(data)
            result = ei * ei
            assert torch.allclose(result.components, neg_e0, atol=1e-12), (
                f"e_{i} * e_{i} should be -e_0, got {result.components}"
            )


class TestErrorMessages:
    """Tests for verbose error messages with math context."""

    def test_inverse_near_zero_error_message(self) -> None:
        """ValueError message for zero inverse contains helpful context."""
        zero = Octonion(torch.zeros(8, dtype=torch.float64))
        with pytest.raises(ValueError, match="zero octonion"):
            zero.inverse()

    def test_wrong_dimension_error(self) -> None:
        """Octonion(tensor_shape_7) raises ValueError with helpful message."""
        bad_data = torch.randn(7, dtype=torch.float64)
        with pytest.raises(ValueError, match="8"):
            Octonion(bad_data)
