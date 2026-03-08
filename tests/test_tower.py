"""Tests for R/C/H Cayley-Dickson tower types."""

import pytest
import torch

from octonion import Real, Complex, Quaternion


class TestReal:
    """Real number type in the Cayley-Dickson tower."""

    def test_construction(self) -> None:
        """Real wraps a [..., 1] tensor."""
        r = Real(torch.tensor([3.0], dtype=torch.float64))
        assert torch.equal(r.components, torch.tensor([3.0], dtype=torch.float64))

    def test_conjugate(self) -> None:
        """Real conjugate is identity (reals have no imaginary part)."""
        r = Real(torch.tensor([5.0], dtype=torch.float64))
        assert torch.equal(r.conjugate().components, r.components)

    def test_norm(self) -> None:
        """Real norm is absolute value."""
        r = Real(torch.tensor([-3.0], dtype=torch.float64))
        assert torch.isclose(r.norm(), torch.tensor(3.0, dtype=torch.float64))

    def test_inverse(self) -> None:
        """Real inverse is 1/x."""
        r = Real(torch.tensor([4.0], dtype=torch.float64))
        inv = r.inverse()
        assert torch.isclose(inv.components[0], torch.tensor(0.25, dtype=torch.float64))

    def test_inverse_zero_raises(self) -> None:
        """Real(0) inverse raises ValueError."""
        r = Real(torch.tensor([0.0], dtype=torch.float64))
        with pytest.raises(ValueError):
            r.inverse()

    def test_mul(self) -> None:
        """Real * Real is scalar multiplication."""
        a = Real(torch.tensor([3.0], dtype=torch.float64))
        b = Real(torch.tensor([4.0], dtype=torch.float64))
        result = a * b
        assert isinstance(result, Real)
        assert torch.isclose(result.components[0], torch.tensor(12.0, dtype=torch.float64))

    def test_add(self) -> None:
        """Real + Real is addition."""
        a = Real(torch.tensor([3.0], dtype=torch.float64))
        b = Real(torch.tensor([4.0], dtype=torch.float64))
        result = a + b
        assert isinstance(result, Real)
        assert torch.isclose(result.components[0], torch.tensor(7.0, dtype=torch.float64))

    def test_sub(self) -> None:
        """Real - Real is subtraction."""
        a = Real(torch.tensor([5.0], dtype=torch.float64))
        b = Real(torch.tensor([3.0], dtype=torch.float64))
        result = a - b
        assert isinstance(result, Real)
        assert torch.isclose(result.components[0], torch.tensor(2.0, dtype=torch.float64))

    def test_neg(self) -> None:
        """-Real negates."""
        r = Real(torch.tensor([3.0], dtype=torch.float64))
        result = -r
        assert isinstance(result, Real)
        assert torch.isclose(result.components[0], torch.tensor(-3.0, dtype=torch.float64))

    def test_eq(self) -> None:
        """Real == Real."""
        a = Real(torch.tensor([3.0], dtype=torch.float64))
        b = Real(torch.tensor([3.0], dtype=torch.float64))
        assert a == b

    def test_scalar_mul(self) -> None:
        """Real * scalar and scalar * Real work."""
        r = Real(torch.tensor([3.0], dtype=torch.float64))
        assert (r * 2.0).components[0].item() == 6.0
        assert (2.0 * r).components[0].item() == 6.0


class TestComplex:
    """Complex number type in the Cayley-Dickson tower."""

    def test_construction(self) -> None:
        """Complex wraps a [..., 2] tensor."""
        c = Complex(torch.tensor([3.0, 4.0], dtype=torch.float64))
        assert torch.equal(c.components, torch.tensor([3.0, 4.0], dtype=torch.float64))

    def test_conjugate(self) -> None:
        """Complex conjugate negates imaginary part."""
        c = Complex(torch.tensor([3.0, 4.0], dtype=torch.float64))
        conj = c.conjugate()
        expected = torch.tensor([3.0, -4.0], dtype=torch.float64)
        assert torch.equal(conj.components, expected)

    def test_norm(self) -> None:
        """Complex norm is modulus."""
        c = Complex(torch.tensor([3.0, 4.0], dtype=torch.float64))
        assert torch.isclose(c.norm(), torch.tensor(5.0, dtype=torch.float64))

    def test_inverse(self) -> None:
        """Complex inverse is conj/|z|^2."""
        c = Complex(torch.tensor([3.0, 4.0], dtype=torch.float64))
        inv = c.inverse()
        # (3-4i)/25 = (0.12, -0.16)
        expected = torch.tensor([3.0 / 25.0, -4.0 / 25.0], dtype=torch.float64)
        assert torch.allclose(inv.components, expected)

    def test_mul_matches_standard(self) -> None:
        """Complex multiplication matches standard complex arithmetic.

        (3+4i)(1+2i) = 3+6i+4i+8i^2 = 3+10i-8 = -5+10i
        """
        a = Complex(torch.tensor([3.0, 4.0], dtype=torch.float64))
        b = Complex(torch.tensor([1.0, 2.0], dtype=torch.float64))
        result = a * b
        expected = torch.tensor([-5.0, 10.0], dtype=torch.float64)
        assert torch.allclose(result.components, expected)

    def test_mul_commutativity(self) -> None:
        """Complex multiplication is commutative."""
        a = Complex(torch.tensor([3.0, 4.0], dtype=torch.float64))
        b = Complex(torch.tensor([1.0, 2.0], dtype=torch.float64))
        assert torch.allclose((a * b).components, (b * a).components)

    def test_add(self) -> None:
        """Complex + Complex."""
        a = Complex(torch.tensor([1.0, 2.0], dtype=torch.float64))
        b = Complex(torch.tensor([3.0, 4.0], dtype=torch.float64))
        result = a + b
        expected = torch.tensor([4.0, 6.0], dtype=torch.float64)
        assert torch.equal(result.components, expected)

    def test_sub(self) -> None:
        """Complex - Complex."""
        a = Complex(torch.tensor([5.0, 6.0], dtype=torch.float64))
        b = Complex(torch.tensor([1.0, 2.0], dtype=torch.float64))
        result = a - b
        expected = torch.tensor([4.0, 4.0], dtype=torch.float64)
        assert torch.equal(result.components, expected)

    def test_neg(self) -> None:
        """-Complex."""
        c = Complex(torch.tensor([3.0, 4.0], dtype=torch.float64))
        result = -c
        expected = torch.tensor([-3.0, -4.0], dtype=torch.float64)
        assert torch.equal(result.components, expected)

    def test_eq(self) -> None:
        """Complex == Complex."""
        a = Complex(torch.tensor([3.0, 4.0], dtype=torch.float64))
        b = Complex(torch.tensor([3.0, 4.0], dtype=torch.float64))
        assert a == b

    def test_scalar_mul(self) -> None:
        """Complex * scalar and scalar * Complex."""
        c = Complex(torch.tensor([3.0, 4.0], dtype=torch.float64))
        result = c * 2.0
        expected = torch.tensor([6.0, 8.0], dtype=torch.float64)
        assert torch.equal(result.components, expected)


class TestQuaternion:
    """Quaternion type in the Cayley-Dickson tower."""

    def test_construction(self) -> None:
        """Quaternion wraps a [..., 4] tensor."""
        q = Quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64))
        assert q.components.shape[-1] == 4

    def test_conjugate(self) -> None:
        """Quaternion conjugate negates imaginary parts."""
        q = Quaternion(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
        conj = q.conjugate()
        expected = torch.tensor([1.0, -2.0, -3.0, -4.0], dtype=torch.float64)
        assert torch.equal(conj.components, expected)

    def test_norm(self) -> None:
        """Quaternion norm."""
        q = Quaternion(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
        expected = torch.sqrt(torch.tensor(30.0, dtype=torch.float64))
        assert torch.isclose(q.norm(), expected)

    def test_inverse(self) -> None:
        """Quaternion inverse: conj/|q|^2."""
        q = Quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64))
        inv = q.inverse()
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(inv.components, expected)

    def test_mul_hamilton_product(self) -> None:
        """Quaternion multiplication matches Hamilton product for known values.

        i * j = k  =>  (0,1,0,0) * (0,0,1,0) = (0,0,0,1)
        """
        i = Quaternion(torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64))
        j = Quaternion(torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64))
        result = i * j
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
        assert torch.allclose(result.components, expected)

    def test_mul_hamilton_ji(self) -> None:
        """j * i = -k (anti-commutativity)."""
        i = Quaternion(torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64))
        j = Quaternion(torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64))
        result = j * i
        expected = torch.tensor([0.0, 0.0, 0.0, -1.0], dtype=torch.float64)
        assert torch.allclose(result.components, expected)

    def test_mul_squaring(self) -> None:
        """i^2 = j^2 = k^2 = -1."""
        i = Quaternion(torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64))
        j = Quaternion(torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64))
        k = Quaternion(torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64))
        minus_one = torch.tensor([-1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose((i * i).components, minus_one)
        assert torch.allclose((j * j).components, minus_one)
        assert torch.allclose((k * k).components, minus_one)

    def test_add(self) -> None:
        """Quaternion + Quaternion."""
        a = Quaternion(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
        b = Quaternion(torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64))
        result = a + b
        expected = torch.tensor([6.0, 8.0, 10.0, 12.0], dtype=torch.float64)
        assert torch.equal(result.components, expected)

    def test_sub(self) -> None:
        """Quaternion - Quaternion."""
        a = Quaternion(torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64))
        b = Quaternion(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
        result = a - b
        expected = torch.tensor([4.0, 4.0, 4.0, 4.0], dtype=torch.float64)
        assert torch.equal(result.components, expected)

    def test_neg(self) -> None:
        """-Quaternion."""
        q = Quaternion(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
        result = -q
        expected = torch.tensor([-1.0, -2.0, -3.0, -4.0], dtype=torch.float64)
        assert torch.equal(result.components, expected)

    def test_eq(self) -> None:
        """Quaternion == Quaternion."""
        a = Quaternion(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
        b = Quaternion(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
        assert a == b

    def test_scalar_mul(self) -> None:
        """Quaternion * scalar and scalar * Quaternion."""
        q = Quaternion(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
        result = q * 3.0
        expected = torch.tensor([3.0, 6.0, 9.0, 12.0], dtype=torch.float64)
        assert torch.equal(result.components, expected)

    def test_inverse_product_identity(self) -> None:
        """q * q.inverse() should be identity."""
        q = Quaternion(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = q * q.inverse()
        assert torch.allclose(result.components, identity, atol=1e-12)
