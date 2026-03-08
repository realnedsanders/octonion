"""Tests for the Octonion class wrapper: construction, operators, methods, subtypes."""

import pytest
import torch

from octonion import Octonion, UnitOctonion, PureOctonion, associator
from octonion._multiplication import octonion_mul
from octonion._random import random_octonion


class TestOctonionConstruction:
    """Octonion construction and validation."""

    def test_construct_from_tensor(self) -> None:
        """Octonion(tensor) stores data and .components returns it."""
        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        o = Octonion(data)
        assert torch.equal(o.components, data)

    def test_real_returns_e0(self) -> None:
        """Octonion.real returns the e0 component."""
        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        o = Octonion(data)
        assert o.real.item() == 1.0

    def test_imag_returns_e1_to_e7(self) -> None:
        """Octonion.imag returns components e1..e7."""
        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        o = Octonion(data)
        expected = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        assert torch.equal(o.imag, expected)

    def test_getitem_returns_component(self) -> None:
        """o[i] returns component i for i in 0..7."""
        data = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], dtype=torch.float64)
        o = Octonion(data)
        for i in range(8):
            assert o[i].item() == data[i].item()

    def test_dim_returns_8(self) -> None:
        """Octonion.dim returns 8."""
        o = Octonion(torch.zeros(8, dtype=torch.float64))
        assert o.dim == 8

    def test_reject_wrong_last_dim(self) -> None:
        """Octonion rejects tensor with last dim != 8."""
        with pytest.raises(ValueError, match="8"):
            Octonion(torch.zeros(7, dtype=torch.float64))
        with pytest.raises(ValueError, match="8"):
            Octonion(torch.zeros(9, dtype=torch.float64))
        with pytest.raises(ValueError, match="8"):
            Octonion(torch.zeros(3, dtype=torch.float64))


class TestOctonionMultiplication:
    """Multiplication operator delegates to octonion_mul."""

    def test_mul_delegates_to_octonion_mul(self) -> None:
        """Octonion * Octonion delegates to octonion_mul and returns Octonion."""
        a_data = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        b_data = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        a = Octonion(a_data)
        b = Octonion(b_data)
        result = a * b
        expected = Octonion(octonion_mul(a_data, b_data))
        assert isinstance(result, Octonion)
        assert torch.allclose(result.components, expected.components)

    def test_mul_random_pair(self) -> None:
        """Multiplication of random octonions matches direct octonion_mul."""
        torch.manual_seed(42)
        a_data = torch.randn(8, dtype=torch.float64)
        b_data = torch.randn(8, dtype=torch.float64)
        a = Octonion(a_data)
        b = Octonion(b_data)
        result = a * b
        expected = octonion_mul(a_data, b_data)
        assert torch.allclose(result.components, expected, atol=1e-14)


class TestOctonionAddition:
    """Addition is component-wise."""

    def test_add_octonions(self) -> None:
        """Octonion + Octonion is component-wise addition."""
        a = Octonion(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64))
        b = Octonion(torch.tensor([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=torch.float64))
        result = a + b
        assert isinstance(result, Octonion)
        expected = torch.tensor([9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0], dtype=torch.float64)
        assert torch.equal(result.components, expected)

    def test_add_scalar_adds_to_real(self) -> None:
        """Octonion + scalar adds to real part only."""
        a = Octonion(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64))
        result = a + 10.0
        expected = torch.tensor([11.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        assert torch.equal(result.components, expected)

    def test_radd_scalar(self) -> None:
        """scalar + Octonion adds to real part only."""
        a = Octonion(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64))
        result = 10.0 + a
        expected = torch.tensor([11.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        assert torch.equal(result.components, expected)


class TestOctonionSubtraction:
    """Subtraction is component-wise."""

    def test_sub_octonions(self) -> None:
        """Octonion - Octonion is component-wise subtraction."""
        a = Octonion(torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], dtype=torch.float64))
        b = Octonion(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64))
        result = a - b
        assert isinstance(result, Octonion)
        expected = torch.tensor([9.0, 18.0, 27.0, 36.0, 45.0, 54.0, 63.0, 72.0], dtype=torch.float64)
        assert torch.equal(result.components, expected)

    def test_rsub_scalar(self) -> None:
        """scalar - Octonion subtracts from real part and negates imaginary."""
        a = Octonion(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64))
        result = 10.0 - a
        expected = torch.tensor([9.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0], dtype=torch.float64)
        assert torch.equal(result.components, expected)


class TestOctonionNeg:
    """Unary negation."""

    def test_neg(self) -> None:
        """-Octonion negates all components."""
        a = Octonion(torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], dtype=torch.float64))
        result = -a
        assert isinstance(result, Octonion)
        expected = torch.tensor([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0], dtype=torch.float64)
        assert torch.equal(result.components, expected)


class TestOctonionScalarInterop:
    """Full scalar interop: scalar * octonion, octonion * scalar."""

    def test_mul_scalar_right(self) -> None:
        """Octonion * scalar scales all components."""
        a = Octonion(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64))
        result = a * 2.0
        expected = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0], dtype=torch.float64)
        assert torch.equal(result.components, expected)

    def test_mul_scalar_left(self) -> None:
        """scalar * Octonion scales all components."""
        a = Octonion(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64))
        result = 3.0 * a
        expected = torch.tensor([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0], dtype=torch.float64)
        assert torch.equal(result.components, expected)

    def test_mul_int_scalar(self) -> None:
        """Octonion * int works."""
        a = Octonion(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64))
        result = a * 2
        expected = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0], dtype=torch.float64)
        assert torch.equal(result.components, expected)


class TestOctonionEquality:
    """Equality comparison."""

    def test_eq_same(self) -> None:
        """Octonion == Octonion compares components element-wise."""
        a = Octonion(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64))
        b = Octonion(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64))
        assert a == b

    def test_eq_different(self) -> None:
        """Octonion != Octonion when components differ."""
        a = Octonion(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64))
        b = Octonion(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0], dtype=torch.float64))
        assert not (a == b)


class TestOctonionConjugate:
    """Conjugation negates imaginary, preserves real."""

    def test_conjugate(self) -> None:
        """.conjugate() negates imaginary, preserves real."""
        a = Octonion(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64))
        conj = a.conjugate()
        assert isinstance(conj, Octonion)
        expected = torch.tensor([1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0], dtype=torch.float64)
        assert torch.equal(conj.components, expected)


class TestOctonionNorm:
    """Norm computation."""

    def test_norm(self) -> None:
        """.norm() returns sqrt of sum of squared components."""
        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        o = Octonion(data)
        expected = torch.sqrt(torch.sum(data ** 2))
        assert torch.isclose(o.norm(), expected)

    def test_norm_unit(self) -> None:
        """Unit octonion (identity) has norm 1."""
        identity = Octonion(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64))
        assert torch.isclose(identity.norm(), torch.tensor(1.0, dtype=torch.float64))


class TestOctonionInverse:
    """Inverse computation."""

    def test_inverse_basic(self) -> None:
        """.inverse() returns conjugate/norm_squared."""
        data = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        o = Octonion(data)
        inv = o.inverse()
        assert isinstance(inv, Octonion)
        # Inverse of real 1.0 is 1.0
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(inv.components, expected)

    def test_inverse_zero_raises(self) -> None:
        """Zero octonion raises ValueError with math context."""
        zero = Octonion(torch.zeros(8, dtype=torch.float64))
        with pytest.raises(ValueError, match="(?i)zero|norm"):
            zero.inverse()

    def test_inverse_product_identity(self) -> None:
        """a * a.inverse() is close to identity."""
        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        o = Octonion(data)
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = o * o.inverse()
        assert torch.allclose(result.components, identity, atol=1e-12)


class TestOctonionRepr:
    """String representations."""

    def test_repr_shows_tensor_form(self) -> None:
        """__repr__ shows tensor form."""
        data = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        o = Octonion(data)
        r = repr(o)
        assert "Octonion" in r

    def test_str_shows_symbolic_form(self) -> None:
        """__str__ shows symbolic form (e.g., 1.0 + 2.0*e1 + ...)."""
        data = torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        o = Octonion(data)
        s = str(o)
        assert "e1" in s


class TestOctonionImmutability:
    """Octonion is immutable -- no in-place mutation methods."""

    def test_no_inplace_mutation(self) -> None:
        """Octonion has no in-place mutation methods."""
        o = Octonion(torch.zeros(8, dtype=torch.float64))
        # Verify no __iadd__, __imul__, __isub__ etc.
        assert not hasattr(o, '__iadd__')
        assert not hasattr(o, '__imul__')
        assert not hasattr(o, '__isub__')

    def test_operations_return_new_instances(self) -> None:
        """All operations return new Octonion instances, not the same object."""
        a = Octonion(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64))
        b = Octonion(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64))
        assert a + b is not a
        assert a - b is not a
        assert a * b is not a
        assert -a is not a
        assert a.conjugate() is not a


class TestOctonionNoDiv:
    """No __truediv__ or __pow__ operators exist (user decision)."""

    def test_no_truediv(self) -> None:
        """Octonion has no __truediv__ operator."""
        a = Octonion(torch.ones(8, dtype=torch.float64))
        b = Octonion(torch.ones(8, dtype=torch.float64))
        with pytest.raises(TypeError):
            a / b  # type: ignore[operator]

    def test_no_pow(self) -> None:
        """Octonion has no __pow__ operator."""
        a = Octonion(torch.ones(8, dtype=torch.float64))
        with pytest.raises(TypeError):
            a ** 2  # type: ignore[operator]


class TestAssociator:
    """Module-level associator function."""

    def test_associator_basic(self) -> None:
        """associator(a, b, c) = (a*b)*c - a*(b*c)."""
        a = Octonion(torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64))
        b = Octonion(torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64))
        c = Octonion(torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64))
        result = associator(a, b, c)
        assert isinstance(result, Octonion)
        # For basis elements where non-associativity manifests, the result should be non-zero
        # (a*b)*c vs a*(b*c) can differ for non-coplanar basis elements


class TestQuaternionPairConversion:
    """from_quaternion_pair and to_quaternion_pair."""

    def test_from_quaternion_pair(self) -> None:
        """from_quaternion_pair(q1, q2) creates correct octonion."""
        q1 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        q2 = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        o = Octonion.from_quaternion_pair(q1, q2)
        assert isinstance(o, Octonion)
        assert o.components.shape[-1] == 8

    def test_roundtrip_quaternion_pair(self) -> None:
        """to_quaternion_pair() and from_quaternion_pair() are inverses of each other."""
        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        o = Octonion(data)
        q1, q2 = o.to_quaternion_pair()
        reconstructed = Octonion.from_quaternion_pair(q1, q2)
        assert torch.allclose(reconstructed.components, o.components, atol=1e-14)


class TestUnitOctonion:
    """UnitOctonion subtype."""

    def test_unit_norm(self) -> None:
        """UnitOctonion has norm == 1 (within tolerance)."""
        data = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        u = UnitOctonion(data)
        assert torch.isclose(u.norm(), torch.tensor(1.0, dtype=torch.float64), atol=1e-12)

    def test_normalizes_input(self) -> None:
        """UnitOctonion normalizes its input to unit norm."""
        data = torch.tensor([3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        u = UnitOctonion(data)
        assert torch.isclose(u.norm(), torch.tensor(1.0, dtype=torch.float64), atol=1e-12)


class TestPureOctonion:
    """PureOctonion subtype."""

    def test_pure_real_zero(self) -> None:
        """PureOctonion has real part == 0."""
        data = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=torch.float64)
        p = PureOctonion(data)
        assert p.real.item() == 0.0

    def test_enforces_real_zero(self) -> None:
        """PureOctonion enforces real=0 at construction."""
        data = torch.tensor([5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=torch.float64)
        p = PureOctonion(data)
        assert p.real.item() == 0.0


class TestOctonionCopyConstructor:
    """Octonion(Octonion(...)) copy constructor and __str__ noise suppression."""

    def test_copy_constructor(self) -> None:
        """Octonion(Octonion(t)) returns Octonion with identical components."""
        t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        inner = Octonion(t)
        outer = Octonion(inner)
        assert isinstance(outer, Octonion)
        assert torch.equal(outer.components, t)

    def test_copy_constructor_from_random(self) -> None:
        """Octonion(random_octonion()) succeeds without error."""
        a = random_octonion()
        wrapped = Octonion(a)
        assert isinstance(wrapped, Octonion)
        assert torch.equal(wrapped.components, a.components)

    def test_str_suppresses_float32_noise(self) -> None:
        """str() of near-identity (a*a.inverse()) suppresses float32 noise."""
        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float32)
        a = Octonion(data)
        identity_approx = a * a.inverse()
        s = str(identity_approx)
        # Should show "1.0" with no imaginary noise terms
        assert "e" not in s, f"Float32 noise leaked into display: {s}"

    def test_str_preserves_real_values(self) -> None:
        """str() of exact non-zero values still displays correctly."""
        data = torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0], dtype=torch.float64)
        o = Octonion(data)
        s = str(o)
        assert "e1" in s, f"Expected e1 in display: {s}"
        assert "e7" in s, f"Expected e7 in display: {s}"
        assert "2.0" in s
        assert "3.0" in s
