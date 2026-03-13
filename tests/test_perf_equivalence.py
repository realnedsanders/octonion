"""Equivalence tests for performance-optimized code paths.

Verifies that Tier 1 optimizations (vectorized _tril_to_symmetric)
produce identical outputs to the original Python-loop reference
implementation. These tests exist to prove zero-risk correctness:
index shuffling has no floating-point error, so we require EXACT
equality (not approximate).
"""

from __future__ import annotations

import pytest
import torch


# ── Reference implementation (original Python-loop version) ────────
# Captured here BEFORE vectorization so the test is self-contained
# and independent of the optimized implementation in _normalization.py.

def _tril_to_symmetric_reference(tril_flat: torch.Tensor, dim: int) -> torch.Tensor:
    """Reference: convert flat lower-triangular entries to a symmetric matrix.

    This is the ORIGINAL Python-loop implementation from _normalization.py
    lines 34-45, captured verbatim as a correctness oracle.

    Args:
        tril_flat: [..., dim*(dim+1)/2] flat lower-triangular entries.
        dim: Matrix dimension.

    Returns:
        [..., dim, dim] symmetric matrix.
    """
    batch_shape = tril_flat.shape[:-1]
    mat = torch.zeros(
        *batch_shape, dim, dim,
        device=tril_flat.device, dtype=tril_flat.dtype,
    )
    idx = 0
    for i in range(dim):
        for j in range(i + 1):
            mat[..., i, j] = tril_flat[..., idx]
            mat[..., j, i] = tril_flat[..., idx]
            idx += 1
    return mat


# ── Import the module under test ────────────────────────────────────

try:
    from octonion.baselines._normalization import _tril_to_symmetric
    _NORMALIZATION_AVAILABLE = True
except ImportError:
    _tril_to_symmetric = None  # type: ignore[assignment]
    _NORMALIZATION_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not _NORMALIZATION_AVAILABLE,
    reason="_tril_to_symmetric not importable from octonion.baselines._normalization"
)


# ── Helper ──────────────────────────────────────────────────────────

def _random_tril_flat(batch_shape: tuple[int, ...], dim: int, dtype: torch.dtype) -> torch.Tensor:
    """Generate random flat lower-triangular entries."""
    tril_size = dim * (dim + 1) // 2
    return torch.randn(*batch_shape, tril_size, dtype=dtype)


# ── Test: dim=2 (complex, 3 entries) ───────────────────────────────

class TestDim2:
    """Equivalence tests for dim=2 (complex algebra)."""

    @pytest.mark.parametrize("batch_shape", [(10,), (3, 5), (1,)])
    def test_float32_equivalence(self, batch_shape: tuple[int, ...]) -> None:
        """Vectorized output matches reference for float32 inputs."""
        flat = _random_tril_flat(batch_shape, dim=2, dtype=torch.float32)
        ref = _tril_to_symmetric_reference(flat, dim=2)
        out = _tril_to_symmetric(flat, dim=2)
        assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
        assert torch.equal(out, ref), "Outputs differ (float32, dim=2)"

    @pytest.mark.parametrize("batch_shape", [(10,), (3, 5), (1,)])
    def test_float64_equivalence(self, batch_shape: tuple[int, ...]) -> None:
        """Vectorized output matches reference for float64 inputs."""
        flat = _random_tril_flat(batch_shape, dim=2, dtype=torch.float64)
        ref = _tril_to_symmetric_reference(flat, dim=2)
        out = _tril_to_symmetric(flat, dim=2)
        assert out.shape == ref.shape
        assert torch.equal(out, ref), "Outputs differ (float64, dim=2)"


# ── Test: dim=4 (quaternion, 10 entries) ───────────────────────────

class TestDim4:
    """Equivalence tests for dim=4 (quaternion algebra)."""

    @pytest.mark.parametrize("batch_shape", [(10,), (3, 5), (1,)])
    def test_float32_equivalence(self, batch_shape: tuple[int, ...]) -> None:
        """Vectorized output matches reference for float32 inputs."""
        flat = _random_tril_flat(batch_shape, dim=4, dtype=torch.float32)
        ref = _tril_to_symmetric_reference(flat, dim=4)
        out = _tril_to_symmetric(flat, dim=4)
        assert out.shape == ref.shape
        assert torch.equal(out, ref), "Outputs differ (float32, dim=4)"

    @pytest.mark.parametrize("batch_shape", [(10,), (3, 5), (1,)])
    def test_float64_equivalence(self, batch_shape: tuple[int, ...]) -> None:
        """Vectorized output matches reference for float64 inputs."""
        flat = _random_tril_flat(batch_shape, dim=4, dtype=torch.float64)
        ref = _tril_to_symmetric_reference(flat, dim=4)
        out = _tril_to_symmetric(flat, dim=4)
        assert out.shape == ref.shape
        assert torch.equal(out, ref), "Outputs differ (float64, dim=4)"

    def test_symmetric_property(self) -> None:
        """Output is always symmetric: M[i,j] == M[j,i]."""
        flat = _random_tril_flat((5,), dim=4, dtype=torch.float32)
        out = _tril_to_symmetric(flat, dim=4)
        # out: [5, 4, 4]
        assert torch.equal(out, out.transpose(-2, -1)), "Output is not symmetric"

    def test_lower_triangular_entries_used(self) -> None:
        """Each lower-triangular entry appears in correct (i,j) and (j,i) positions."""
        # Construct input where each entry is uniquely identifiable
        dim = 4
        tril_size = dim * (dim + 1) // 2
        flat = torch.arange(tril_size, dtype=torch.float32).unsqueeze(0)  # [1, 10]
        ref = _tril_to_symmetric_reference(flat, dim=4)
        out = _tril_to_symmetric(flat, dim=4)
        assert torch.equal(out, ref), "Entry placement differs from reference"


# ── Test: dim=8 (octonion, 36 entries) ─────────────────────────────

class TestDim8:
    """Equivalence tests for dim=8 (octonion algebra)."""

    @pytest.mark.parametrize("batch_shape", [(10,), (3, 5), (1,)])
    def test_float32_equivalence(self, batch_shape: tuple[int, ...]) -> None:
        """Vectorized output matches reference for float32 inputs."""
        flat = _random_tril_flat(batch_shape, dim=8, dtype=torch.float32)
        ref = _tril_to_symmetric_reference(flat, dim=8)
        out = _tril_to_symmetric(flat, dim=8)
        assert out.shape == ref.shape
        assert torch.equal(out, ref), "Outputs differ (float32, dim=8)"

    @pytest.mark.parametrize("batch_shape", [(10,), (3, 5), (1,)])
    def test_float64_equivalence(self, batch_shape: tuple[int, ...]) -> None:
        """Vectorized output matches reference for float64 inputs."""
        flat = _random_tril_flat(batch_shape, dim=8, dtype=torch.float64)
        ref = _tril_to_symmetric_reference(flat, dim=8)
        out = _tril_to_symmetric(flat, dim=8)
        assert out.shape == ref.shape
        assert torch.equal(out, ref), "Outputs differ (float64, dim=8)"

    def test_symmetric_property(self) -> None:
        """Output is always symmetric: M[i,j] == M[j,i]."""
        flat = _random_tril_flat((5,), dim=8, dtype=torch.float32)
        out = _tril_to_symmetric(flat, dim=8)
        # out: [5, 8, 8]
        assert torch.equal(out, out.transpose(-2, -1)), "Output is not symmetric"

    def test_lower_triangular_entries_used(self) -> None:
        """Each lower-triangular entry is placed correctly in (i,j) and (j,i)."""
        dim = 8
        tril_size = dim * (dim + 1) // 2
        flat = torch.arange(tril_size, dtype=torch.float32).unsqueeze(0)  # [1, 36]
        ref = _tril_to_symmetric_reference(flat, dim=8)
        out = _tril_to_symmetric(flat, dim=8)
        assert torch.equal(out, ref), "Entry placement differs from reference (dim=8)"

    def test_shape_is_correct(self) -> None:
        """Output shape is [..., dim, dim]."""
        for batch_shape in [(10,), (3, 5), (1,)]:
            flat = _random_tril_flat(batch_shape, dim=8, dtype=torch.float32)
            out = _tril_to_symmetric(flat, dim=8)
            expected_shape = torch.Size([*batch_shape, 8, 8])
            assert out.shape == expected_shape, f"Shape {out.shape} != {expected_shape}"
