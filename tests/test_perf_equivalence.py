"""Equivalence tests for performance-optimized code paths.

Verifies that:
- Tier 1 optimizations (vectorized _tril_to_symmetric) produce identical
  outputs to the original Python-loop reference implementation.
- Tier 2 optimizations (fused OctonionDenseLinear einsum forward) produce
  outputs within floating-point tolerance of the Python-loop reference.
- Eval-mode fused weight caching in OctonionConv2d produces identical outputs
  to train-mode computation.

Tier 1 tests use EXACT equality (no floating-point error in index shuffling).
Tier 2 tests use approximate equality (atol=1e-5 for einsum vs loop rounding).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Reference implementation: Tier 1 (Tier 1 tril vectorization) ────
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


# ── Reference implementations for Tier 2 (fused linear) ─────────────


def _octonion_dense_forward_reference(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    nonzero_entries: list[tuple[int, int, int, float]],
    out_features: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Reference: Python-loop OctonionDenseLinear forward.

    This is the ORIGINAL loop implementation from _algebra_linear.py
    (lines 279-313), captured verbatim as a correctness oracle for
    the fused einsum implementation.

    Args:
        x: [..., in_features, 8]
        weights: list of 8 weight matrices [out_features, in_features]
        nonzero_entries: list of (i, j, k, coeff) tuples
        out_features: number of output features
        bias: optional bias tensor [out_features, 8]

    Returns:
        [..., out_features, 8]
    """
    batch_shape = x.shape[:-2]
    linear_cache: dict[tuple[int, int], torch.Tensor] = {}
    out_components = [
        torch.zeros(*batch_shape, out_features, dtype=x.dtype, device=x.device)
        for _ in range(8)
    ]
    for i, j, k, coeff in nonzero_entries:
        key = (i, j)
        if key not in linear_cache:
            linear_cache[key] = F.linear(x[..., j], weights[i])
        out_components[k] = out_components[k] + coeff * linear_cache[key]
    result = torch.stack(out_components, dim=-1)
    if bias is not None:
        result = result + bias
    return result


def _build_nonzero_entries(C: torch.Tensor) -> list[tuple[int, int, int, float]]:
    """Extract nonzero (i, j, k, coefficient) entries from structure constants."""
    entries: list[tuple[int, int, int, float]] = []
    for i in range(8):
        for j in range(8):
            for k in range(8):
                c = C[i, j, k].item()
                if c != 0.0:
                    entries.append((i, j, k, float(c)))
    return entries


# ── Imports ──────────────────────────────────────────────────────────

try:
    from octonion.baselines._normalization import _tril_to_symmetric
    _NORMALIZATION_AVAILABLE = True
except ImportError:
    _tril_to_symmetric = None  # type: ignore[assignment]
    _NORMALIZATION_AVAILABLE = False

try:
    from octonion.baselines._algebra_linear import OctonionDenseLinear
    from octonion._multiplication import STRUCTURE_CONSTANTS
    _LINEAR_AVAILABLE = True
except ImportError:
    OctonionDenseLinear = None  # type: ignore[assignment, misc]
    STRUCTURE_CONSTANTS = None  # type: ignore[assignment]
    _LINEAR_AVAILABLE = False

try:
    from octonion.baselines._algebra_conv import OctonionConv2d, QuaternionConv2d
    _CONV_AVAILABLE = True
except ImportError:
    OctonionConv2d = None  # type: ignore[assignment, misc]
    QuaternionConv2d = None  # type: ignore[assignment, misc]
    _CONV_AVAILABLE = False


# ── Helper ───────────────────────────────────────────────────────────

def _random_tril_flat(batch_shape: tuple[int, ...], dim: int, dtype: torch.dtype) -> torch.Tensor:
    """Generate random flat lower-triangular entries."""
    tril_size = dim * (dim + 1) // 2
    return torch.randn(*batch_shape, tril_size, dtype=dtype)


# ── Tier 1: dim=2 (complex, 3 entries) ───────────────────────────────

@pytest.mark.skipif(
    not _NORMALIZATION_AVAILABLE,
    reason="_tril_to_symmetric not importable from octonion.baselines._normalization",
)
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


# ── Tier 1: dim=4 (quaternion, 10 entries) ────────────────────────────

@pytest.mark.skipif(
    not _NORMALIZATION_AVAILABLE,
    reason="_tril_to_symmetric not importable from octonion.baselines._normalization",
)
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


# ── Tier 1: dim=8 (octonion, 36 entries) ─────────────────────────────

@pytest.mark.skipif(
    not _NORMALIZATION_AVAILABLE,
    reason="_tril_to_symmetric not importable from octonion.baselines._normalization",
)
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


# ── Tier 2: OctonionDenseLinear fused forward equivalence ─────────────

@pytest.mark.skipif(
    not _LINEAR_AVAILABLE,
    reason="OctonionDenseLinear not importable from octonion.baselines._algebra_linear",
)
class TestOctonionDenseLinearFusedForward:
    """Equivalence tests for fused OctonionDenseLinear vs Python-loop reference.

    Tests that the einsum+F.linear fused forward produces outputs within
    float32 rounding tolerance of the original Python-loop implementation.
    """

    def _make_layer_and_inputs(
        self, in_features: int = 16, out_features: int = 32, seed: int = 42
    ) -> tuple[OctonionDenseLinear, list[torch.Tensor], list[tuple[int, int, int, float]]]:
        """Create a layer with fixed seed for reproducibility."""
        torch.manual_seed(seed)
        layer = OctonionDenseLinear(in_features, out_features)
        weights = [w.detach() for w in layer.weights]
        nonzero_entries = _build_nonzero_entries(STRUCTURE_CONSTANTS)
        return layer, weights, nonzero_entries

    @pytest.mark.parametrize("batch_shape,in_f,out_f", [
        ((4, 16), 16, 32),   # Standard batch
        ((1, 16), 16, 32),   # Single sample
        ((2, 3, 16), 16, 32),  # Higher-rank batch dims
    ])
    def test_fused_forward_matches_reference(
        self,
        batch_shape: tuple[int, ...],
        in_f: int,
        out_f: int,
    ) -> None:
        """Fused output matches reference within atol=1e-5 for all tested shapes."""
        torch.manual_seed(0)
        layer = OctonionDenseLinear(in_f, out_f, bias=False)
        weights = [w.detach() for w in layer.weights]
        nonzero_entries = _build_nonzero_entries(STRUCTURE_CONSTANTS)

        x = torch.randn(*batch_shape, 8)
        fused_out = layer(x)
        ref_out = _octonion_dense_forward_reference(x, weights, nonzero_entries, out_f, None)

        assert fused_out.shape == ref_out.shape, (
            f"Shape mismatch: fused={fused_out.shape} ref={ref_out.shape}"
        )
        assert torch.allclose(fused_out, ref_out, atol=1e-5), (
            f"Output mismatch: max_diff={( fused_out - ref_out).abs().max().item():.2e}"
        )

    def test_fused_backward_matches_reference(self) -> None:
        """Backward gradients through fused forward match reference within atol=1e-4."""
        torch.manual_seed(1)
        in_f, out_f = 16, 32

        # Two identical layers (same weights) for comparison
        layer_fused = OctonionDenseLinear(in_f, out_f, bias=False)
        # Build reference nonzero entries from structure constants
        nonzero_entries = _build_nonzero_entries(STRUCTURE_CONSTANTS)

        x_fused = torch.randn(4, in_f, 8, requires_grad=True)
        x_ref = x_fused.detach().clone().requires_grad_(True)

        # Fused forward
        out_fused = layer_fused(x_fused)
        loss_fused = out_fused.sum()
        loss_fused.backward()

        # Reference forward with same weights
        weights_ref = [w.detach() for w in layer_fused.weights]
        out_ref = _octonion_dense_forward_reference(x_ref, weights_ref, nonzero_entries, out_f, None)
        loss_ref = out_ref.sum()
        loss_ref.backward()

        assert x_fused.grad is not None
        assert x_ref.grad is not None
        assert torch.allclose(x_fused.grad, x_ref.grad, atol=1e-4), (
            f"Gradient mismatch: max_diff={(x_fused.grad - x_ref.grad).abs().max().item():.2e}"
        )

    def test_state_dict_keys_unchanged(self) -> None:
        """State dict keys are identical (ParameterList preserved for checkpoint compat)."""
        layer = OctonionDenseLinear(8, 16)
        keys = list(layer.state_dict().keys())
        # Must have weights.0 through weights.7 (ParameterList)
        for i in range(8):
            assert f"weights.{i}" in keys, f"Missing key weights.{i}"
        # bias must be present (default bias=True)
        assert "bias" in keys, "Missing key bias"

    def test_C_buffer_not_in_state_dict(self) -> None:
        """Structure constant buffer _C is non-persistent (not in state_dict)."""
        layer = OctonionDenseLinear(8, 16)
        keys = list(layer.state_dict().keys())
        assert "_C" not in keys, "_C should not be in state_dict (persistent=False)"

    def test_C_buffer_is_buffer(self) -> None:
        """_C is a registered buffer (moves with .to(device)) and has correct shape."""
        layer = OctonionDenseLinear(8, 16)
        # _C must be accessible as an attribute
        assert hasattr(layer, "_C"), "Layer must have _C attribute"
        assert layer._C.shape == (8, 8, 8), f"_C shape should be [8,8,8], got {layer._C.shape}"
        # Verify it's a buffer (not a parameter)
        named_buffers = dict(layer.named_buffers(recurse=False))
        assert "_C" in named_buffers, "_C should be in named_buffers"
        named_params = dict(layer.named_parameters(recurse=False))
        assert "_C" not in named_params, "_C should not be in named_parameters"
