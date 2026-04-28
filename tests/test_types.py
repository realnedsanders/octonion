"""Tests for NormedDivisionAlgebra type hierarchy and subtype contracts."""

import torch

from octonion import Complex, NormedDivisionAlgebra, Octonion, Quaternion, Real


class TestNormedDivisionAlgebraContract:
    """All concrete types implement NormedDivisionAlgebra."""

    def test_octonion_is_normed_division_algebra(self) -> None:
        """Octonion extends NormedDivisionAlgebra."""
        o = Octonion(torch.zeros(8, dtype=torch.float64))
        assert isinstance(o, NormedDivisionAlgebra)

    def test_real_is_normed_division_algebra(self) -> None:
        """Real extends NormedDivisionAlgebra."""
        r = Real(torch.tensor([1.0], dtype=torch.float64))
        assert isinstance(r, NormedDivisionAlgebra)

    def test_complex_is_normed_division_algebra(self) -> None:
        """Complex extends NormedDivisionAlgebra."""
        c = Complex(torch.tensor([1.0, 0.0], dtype=torch.float64))
        assert isinstance(c, NormedDivisionAlgebra)

    def test_quaternion_is_normed_division_algebra(self) -> None:
        """Quaternion extends NormedDivisionAlgebra."""
        q = Quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64))
        assert isinstance(q, NormedDivisionAlgebra)


class TestDimensions:
    """Each type reports correct dimension."""

    def test_real_dim_1(self) -> None:
        r = Real(torch.tensor([1.0], dtype=torch.float64))
        assert r.dim == 1

    def test_complex_dim_2(self) -> None:
        c = Complex(torch.tensor([1.0, 0.0], dtype=torch.float64))
        assert c.dim == 2

    def test_quaternion_dim_4(self) -> None:
        q = Quaternion(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64))
        assert q.dim == 4

    def test_octonion_dim_8(self) -> None:
        o = Octonion(torch.zeros(8, dtype=torch.float64))
        assert o.dim == 8


class TestNormSquared:
    """norm_squared() is consistent across all types."""

    def test_real_norm_squared(self) -> None:
        r = Real(torch.tensor([3.0], dtype=torch.float64))
        assert torch.isclose(r.norm_squared(), torch.tensor(9.0, dtype=torch.float64))

    def test_complex_norm_squared(self) -> None:
        c = Complex(torch.tensor([3.0, 4.0], dtype=torch.float64))
        assert torch.isclose(c.norm_squared(), torch.tensor(25.0, dtype=torch.float64))

    def test_quaternion_norm_squared(self) -> None:
        q = Quaternion(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
        assert torch.isclose(q.norm_squared(), torch.tensor(30.0, dtype=torch.float64))
