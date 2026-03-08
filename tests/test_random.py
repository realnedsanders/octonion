"""Tests for random octonion generation utilities."""

import torch

from octonion import Octonion, random_octonion, random_unit_octonion, random_pure_octonion


class TestRandomOctonion:
    """random_octonion() generates general octonions."""

    def test_shape(self) -> None:
        """random_octonion() produces [8] tensor wrapped in Octonion."""
        o = random_octonion()
        assert isinstance(o, Octonion)
        assert o.components.shape == (8,)

    def test_dtype(self) -> None:
        """random_octonion() respects dtype parameter."""
        o = random_octonion(dtype=torch.float32)
        assert o.components.dtype == torch.float32
        o64 = random_octonion(dtype=torch.float64)
        assert o64.components.dtype == torch.float64

    def test_seed_reproducibility(self) -> None:
        """random_octonion() produces reproducible results with generator."""
        gen1 = torch.Generator().manual_seed(42)
        o1 = random_octonion(generator=gen1)
        gen2 = torch.Generator().manual_seed(42)
        o2 = random_octonion(generator=gen2)
        assert torch.equal(o1.components, o2.components)

    def test_batch_size(self) -> None:
        """random_octonion() supports batch_size parameter."""
        o = random_octonion(batch_size=10)
        assert isinstance(o, Octonion)
        assert o.components.shape == (10, 8)

    def test_batch_none_is_single(self) -> None:
        """random_octonion(batch_size=None) returns single octonion."""
        o = random_octonion(batch_size=None)
        assert o.components.shape == (8,)


class TestRandomUnitOctonion:
    """random_unit_octonion() generates unit-norm octonions."""

    def test_unit_norm(self) -> None:
        """random_unit_octonion() produces norm-1 octonion."""
        o = random_unit_octonion()
        assert isinstance(o, Octonion)
        assert torch.isclose(o.norm(), torch.tensor(1.0, dtype=torch.float64), atol=1e-12)

    def test_batch_unit_norm(self) -> None:
        """Batched random_unit_octonion all have norm 1."""
        o = random_unit_octonion(batch_size=100)
        norms = o.norm()
        ones = torch.ones(100, dtype=torch.float64)
        assert torch.allclose(norms, ones, atol=1e-12)

    def test_seed_reproducibility(self) -> None:
        """random_unit_octonion() is reproducible with seed."""
        gen1 = torch.Generator().manual_seed(123)
        o1 = random_unit_octonion(generator=gen1)
        gen2 = torch.Generator().manual_seed(123)
        o2 = random_unit_octonion(generator=gen2)
        assert torch.equal(o1.components, o2.components)


class TestRandomPureOctonion:
    """random_pure_octonion() generates pure octonions (real part = 0)."""

    def test_pure_real_zero(self) -> None:
        """random_pure_octonion() produces real-part-0 octonion."""
        o = random_pure_octonion()
        assert isinstance(o, Octonion)
        assert o.real.item() == 0.0

    def test_batch_pure(self) -> None:
        """Batched random_pure_octonion all have real part 0."""
        o = random_pure_octonion(batch_size=50)
        assert torch.all(o.real == 0.0)

    def test_seed_reproducibility(self) -> None:
        """random_pure_octonion() is reproducible with seed."""
        gen1 = torch.Generator().manual_seed(99)
        o1 = random_pure_octonion(generator=gen1)
        gen2 = torch.Generator().manual_seed(99)
        o2 = random_pure_octonion(generator=gen2)
        assert torch.equal(o1.components, o2.components)
