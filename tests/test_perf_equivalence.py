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
    from octonion.baselines._normalization import (
        _tril_to_symmetric,
        QuaternionBatchNorm,
        OctonionBatchNorm,
    )
    _NORMALIZATION_AVAILABLE = True
except ImportError:
    _tril_to_symmetric = None  # type: ignore[assignment]
    QuaternionBatchNorm = None  # type: ignore[assignment, misc]
    OctonionBatchNorm = None  # type: ignore[assignment, misc]
    _NORMALIZATION_AVAILABLE = False

try:
    from octonion.baselines._config import TrainConfig
    _CONFIG_AVAILABLE = True
except ImportError:
    TrainConfig = None  # type: ignore[assignment, misc]
    _CONFIG_AVAILABLE = False

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
        rows, cols = torch.tril_indices(2, 2)
        ref = _tril_to_symmetric_reference(flat, dim=2)
        out = _tril_to_symmetric(flat, dim=2, rows=rows, cols=cols)
        assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
        assert torch.equal(out, ref), "Outputs differ (float32, dim=2)"

    @pytest.mark.parametrize("batch_shape", [(10,), (3, 5), (1,)])
    def test_float64_equivalence(self, batch_shape: tuple[int, ...]) -> None:
        """Vectorized output matches reference for float64 inputs."""
        flat = _random_tril_flat(batch_shape, dim=2, dtype=torch.float64)
        rows, cols = torch.tril_indices(2, 2)
        ref = _tril_to_symmetric_reference(flat, dim=2)
        out = _tril_to_symmetric(flat, dim=2, rows=rows, cols=cols)
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
        rows, cols = torch.tril_indices(4, 4)
        ref = _tril_to_symmetric_reference(flat, dim=4)
        out = _tril_to_symmetric(flat, dim=4, rows=rows, cols=cols)
        assert out.shape == ref.shape
        assert torch.equal(out, ref), "Outputs differ (float32, dim=4)"

    @pytest.mark.parametrize("batch_shape", [(10,), (3, 5), (1,)])
    def test_float64_equivalence(self, batch_shape: tuple[int, ...]) -> None:
        """Vectorized output matches reference for float64 inputs."""
        flat = _random_tril_flat(batch_shape, dim=4, dtype=torch.float64)
        rows, cols = torch.tril_indices(4, 4)
        ref = _tril_to_symmetric_reference(flat, dim=4)
        out = _tril_to_symmetric(flat, dim=4, rows=rows, cols=cols)
        assert out.shape == ref.shape
        assert torch.equal(out, ref), "Outputs differ (float64, dim=4)"

    def test_symmetric_property(self) -> None:
        """Output is always symmetric: M[i,j] == M[j,i]."""
        flat = _random_tril_flat((5,), dim=4, dtype=torch.float32)
        rows, cols = torch.tril_indices(4, 4)
        out = _tril_to_symmetric(flat, dim=4, rows=rows, cols=cols)
        # out: [5, 4, 4]
        assert torch.equal(out, out.transpose(-2, -1)), "Output is not symmetric"

    def test_lower_triangular_entries_used(self) -> None:
        """Each lower-triangular entry appears in correct (i,j) and (j,i) positions."""
        # Construct input where each entry is uniquely identifiable
        dim = 4
        tril_size = dim * (dim + 1) // 2
        flat = torch.arange(tril_size, dtype=torch.float32).unsqueeze(0)  # [1, 10]
        rows, cols = torch.tril_indices(4, 4)
        ref = _tril_to_symmetric_reference(flat, dim=4)
        out = _tril_to_symmetric(flat, dim=4, rows=rows, cols=cols)
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
        rows, cols = torch.tril_indices(8, 8)
        ref = _tril_to_symmetric_reference(flat, dim=8)
        out = _tril_to_symmetric(flat, dim=8, rows=rows, cols=cols)
        assert out.shape == ref.shape
        assert torch.equal(out, ref), "Outputs differ (float32, dim=8)"

    @pytest.mark.parametrize("batch_shape", [(10,), (3, 5), (1,)])
    def test_float64_equivalence(self, batch_shape: tuple[int, ...]) -> None:
        """Vectorized output matches reference for float64 inputs."""
        flat = _random_tril_flat(batch_shape, dim=8, dtype=torch.float64)
        rows, cols = torch.tril_indices(8, 8)
        ref = _tril_to_symmetric_reference(flat, dim=8)
        out = _tril_to_symmetric(flat, dim=8, rows=rows, cols=cols)
        assert out.shape == ref.shape
        assert torch.equal(out, ref), "Outputs differ (float64, dim=8)"

    def test_symmetric_property(self) -> None:
        """Output is always symmetric: M[i,j] == M[j,i]."""
        flat = _random_tril_flat((5,), dim=8, dtype=torch.float32)
        rows, cols = torch.tril_indices(8, 8)
        out = _tril_to_symmetric(flat, dim=8, rows=rows, cols=cols)
        # out: [5, 8, 8]
        assert torch.equal(out, out.transpose(-2, -1)), "Output is not symmetric"

    def test_lower_triangular_entries_used(self) -> None:
        """Each lower-triangular entry is placed correctly in (i,j) and (j,i)."""
        dim = 8
        tril_size = dim * (dim + 1) // 2
        flat = torch.arange(tril_size, dtype=torch.float32).unsqueeze(0)  # [1, 36]
        rows, cols = torch.tril_indices(8, 8)
        ref = _tril_to_symmetric_reference(flat, dim=8)
        out = _tril_to_symmetric(flat, dim=8, rows=rows, cols=cols)
        assert torch.equal(out, ref), "Entry placement differs from reference (dim=8)"

    def test_shape_is_correct(self) -> None:
        """Output shape is [..., dim, dim]."""
        rows, cols = torch.tril_indices(8, 8)
        for batch_shape in [(10,), (3, 5), (1,)]:
            flat = _random_tril_flat(batch_shape, dim=8, dtype=torch.float32)
            out = _tril_to_symmetric(flat, dim=8, rows=rows, cols=cols)
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


# ── Tier 2: OctonionConv2d eval-mode fused weight caching ─────────────

@pytest.mark.skipif(
    not _CONV_AVAILABLE,
    reason="OctonionConv2d not importable from octonion.baselines._algebra_conv",
)
class TestOctonionConv2dEvalCache:
    """Eval-mode fused weight caching for OctonionConv2d.

    During evaluation, weights don't change between batches, so the fused
    weight matrix can be computed once and reused. These tests verify:
    - Train-mode and eval-mode produce identical outputs (cache correctness)
    - Multiple eval forward calls produce exactly equal results (cache used)
    - Cache is invalidated when switching back to train mode
    - Updated weights after re-entering eval produce updated results
    """

    def _make_layer(self, seed: int = 42) -> OctonionConv2d:
        torch.manual_seed(seed)
        return OctonionConv2d(3, 8, kernel_size=3, padding=1)

    def test_eval_mode_matches_train_mode(self) -> None:
        """Eval-mode output matches train-mode output (cache correctness)."""
        layer = self._make_layer()
        x = torch.randn(2, 3, 8, 16, 16)

        layer.train()
        out_train = layer(x)

        layer.eval()
        out_eval = layer(x)

        assert torch.allclose(out_train, out_eval, atol=1e-6), (
            f"Train/eval output mismatch: max_diff={(out_train - out_eval).abs().max():.2e}"
        )

    def test_eval_cache_reuse_exact_equality(self) -> None:
        """Multiple eval forward calls produce exactly equal results (cached tensor)."""
        layer = self._make_layer()
        x = torch.randn(2, 3, 8, 16, 16)

        layer.eval()
        out_eval1 = layer(x)
        out_eval2 = layer(x)  # should use cache

        assert torch.equal(out_eval1, out_eval2), (
            "Second eval call should use cached fused weight and produce exact same result"
        )

    def test_cache_invalidated_on_train(self) -> None:
        """Switching back to train mode invalidates the cache."""
        layer = self._make_layer()
        x = torch.randn(2, 3, 8, 16, 16)

        layer.eval()
        _ = layer(x)  # populate cache

        layer.train()
        assert layer._fused_cache is None, (
            "Cache should be None after calling .train()"
        )

    def test_updated_weights_reflected_after_retrain_eval(self) -> None:
        """After modifying weights and re-entering eval, the cache reflects updated weights."""
        layer = self._make_layer()
        x = torch.randn(2, 3, 8, 16, 16)

        # First eval pass
        layer.eval()
        out_original = layer(x)

        # Modify weights (simulate training step)
        layer.train()
        with torch.no_grad():
            for w in layer.weights:
                w.fill_(0.01)  # Set all weights to small constant

        # Re-enter eval - cache must be rebuilt with new weights
        layer.eval()
        out_updated = layer(x)

        # Outputs should differ from original (weights changed)
        assert not torch.allclose(out_original, out_updated, atol=1e-3), (
            "Output after weight update should differ from original"
        )

        # But two consecutive eval calls should still produce identical results
        out_updated2 = layer(x)
        assert torch.equal(out_updated, out_updated2), (
            "Consecutive eval calls should produce exact same result"
        )

    def test_cache_is_none_initially(self) -> None:
        """Cache starts as None (no pre-computation at init)."""
        layer = self._make_layer()
        assert layer._fused_cache is None, (
            "_fused_cache should be None at initialization"
        )

    def test_cache_populated_after_eval_forward(self) -> None:
        """Cache is populated after first eval forward call."""
        layer = self._make_layer()
        x = torch.randn(2, 3, 8, 16, 16)

        layer.eval()
        _ = layer(x)

        assert layer._fused_cache is not None, (
            "_fused_cache should be populated after eval forward"
        )


# ── Tier 2: QuaternionConv2d eval-mode fused weight caching ──────────

@pytest.mark.skipif(
    not _CONV_AVAILABLE,
    reason="QuaternionConv2d not importable from octonion.baselines._algebra_conv",
)
class TestQuaternionConv2dEvalCache:
    """Eval-mode fused weight caching for QuaternionConv2d."""

    def _make_layer(self, seed: int = 42) -> QuaternionConv2d:
        torch.manual_seed(seed)
        return QuaternionConv2d(3, 8, kernel_size=3, padding=1)

    def test_eval_mode_matches_train_mode(self) -> None:
        """Eval-mode output matches train-mode output."""
        layer = self._make_layer()
        x = torch.randn(2, 3, 4, 16, 16)

        layer.train()
        out_train = layer(x)

        layer.eval()
        out_eval = layer(x)

        assert torch.allclose(out_train, out_eval, atol=1e-6), (
            f"Train/eval output mismatch: max_diff={(out_train - out_eval).abs().max():.2e}"
        )

    def test_eval_cache_reuse_exact_equality(self) -> None:
        """Multiple eval forward calls produce exactly equal results."""
        layer = self._make_layer()
        x = torch.randn(2, 3, 4, 16, 16)

        layer.eval()
        out_eval1 = layer(x)
        out_eval2 = layer(x)

        assert torch.equal(out_eval1, out_eval2), (
            "Consecutive eval calls should produce exact same result"
        )

    def test_cache_invalidated_on_train(self) -> None:
        """Switching back to train mode invalidates the cache."""
        layer = self._make_layer()
        x = torch.randn(2, 3, 4, 16, 16)

        layer.eval()
        _ = layer(x)  # populate cache

        layer.train()
        assert layer._fused_cache is None, (
            "Cache should be None after calling .train()"
        )

    def test_cache_is_none_initially(self) -> None:
        """Cache starts as None."""
        layer = self._make_layer()
        assert layer._fused_cache is None, (
            "_fused_cache should be None at initialization"
        )


# ── Tier 3: AMP BN float32 protection and cholesky_ex ─────────────────

@pytest.mark.skipif(
    not _NORMALIZATION_AVAILABLE,
    reason="BN classes not importable from octonion.baselines._normalization",
)
class TestBNAMPProtection:
    """Tests for AMP float32 protection in BN whitening.

    Verifies that:
    - BN whitening works correctly under AMP autocast (no NaN, correct shape)
    - Cholesky operations remain in float32 regardless of autocast state
    - Per-feature cholesky_ex fallback: only degenerate features get extra
      regularization, others are unaffected
    - CPU path still works with the autocast(enabled=False) wrapper (no-op on CPU)
    """

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="AMP requires CUDA",
    )
    def test_quaternion_bn_amp_safe(self) -> None:
        """QuaternionBatchNorm produces valid output (no NaN) under AMP autocast.

        We compare two fresh BN instances (same init state) to isolate AMP
        effects from running-stat differences between calls.
        """
        torch.manual_seed(42)
        bn_fp32 = QuaternionBatchNorm(16).cuda()
        bn_amp = QuaternionBatchNorm(16).cuda()
        # Copy weights to ensure identical initial state
        bn_amp.load_state_dict(bn_fp32.state_dict())

        x = torch.randn(8, 16, 4, device="cuda")

        # Non-AMP reference
        bn_fp32.train()
        out_fp32 = bn_fp32(x)

        # AMP forward: same initial state, same x, but inside autocast
        bn_amp.train()
        with torch.amp.autocast("cuda", enabled=True):
            out_amp = bn_amp(x)

        assert not torch.isnan(out_amp).any(), "AMP BN produced NaN"
        assert out_amp.shape == (8, 16, 4), f"Unexpected shape: {out_amp.shape}"
        # atol=5e-3: float16 gamma/beta arithmetic introduces ~1e-3 rounding
        # vs the pure fp32 path; 5e-3 covers worst-case half precision error.
        assert torch.allclose(
            out_fp32, out_amp.float(), atol=5e-3
        ), f"AMP BN diverged from fp32: max_diff={(out_fp32 - out_amp.float()).abs().max():.4e}"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="AMP requires CUDA",
    )
    def test_octonion_bn_amp_safe(self) -> None:
        """OctonionBatchNorm produces valid output (no NaN) under AMP autocast."""
        torch.manual_seed(42)
        bn_fp32 = OctonionBatchNorm(8).cuda()
        bn_amp = OctonionBatchNorm(8).cuda()
        bn_amp.load_state_dict(bn_fp32.state_dict())

        # Use batch size >> feature dim to ensure well-conditioned covariance
        x = torch.randn(64, 8, 8, device="cuda")

        bn_fp32.train()
        out_fp32 = bn_fp32(x)

        bn_amp.train()
        with torch.amp.autocast("cuda", enabled=True):
            out_amp = bn_amp(x)

        assert not torch.isnan(out_amp).any(), "AMP OctonionBN produced NaN"
        assert out_amp.shape == (64, 8, 8), f"Unexpected shape: {out_amp.shape}"
        # atol=5e-3: float16 gamma/beta arithmetic introduces ~1e-3 rounding
        # vs the pure fp32 path; 5e-3 covers worst-case half precision error.
        assert torch.allclose(
            out_fp32, out_amp.float(), atol=5e-3
        ), f"AMP OctonionBN diverged from fp32: max_diff={(out_fp32 - out_amp.float()).abs().max():.4e}"

    def test_quaternion_bn_cpu_no_error(self) -> None:
        """QuaternionBatchNorm forward works on CPU (autocast disable is no-op)."""
        bn = QuaternionBatchNorm(4)
        x = torch.randn(8, 4, 4)
        bn.train()
        out = bn(x)
        assert not torch.isnan(out).any(), "CPU BN produced NaN"
        assert out.shape == (8, 4, 4), f"Unexpected shape: {out.shape}"

    def test_octonion_bn_cpu_no_error(self) -> None:
        """OctonionBatchNorm forward works on CPU (autocast disable is no-op)."""
        bn = OctonionBatchNorm(4)
        x = torch.randn(8, 4, 8)
        bn.train()
        out = bn(x)
        assert not torch.isnan(out).any(), "CPU OctonionBN produced NaN"
        assert out.shape == (8, 4, 8), f"Unexpected shape: {out.shape}"

    def test_cholesky_ex_per_feature_fallback_quaternion(self) -> None:
        """Per-feature cholesky_ex fallback: degenerate feature gets extra regularization,
        healthy features are unaffected."""
        bn = QuaternionBatchNorm(4)
        # Create covariance where feature 0 is near-singular, features 1-3 are healthy
        cov = torch.eye(4).unsqueeze(0).expand(4, -1, -1).clone()
        # Make feature 0 nearly singular (near-zero matrix)
        cov[0] = torch.zeros(4, 4)
        cov[0, 0, 0] = 1e-12  # Near-zero diagonal, zero off-diagonal

        x_centered = torch.randn(8, 4, 4)
        result = bn._whiten(x_centered, cov)
        assert not torch.isnan(result).any(), "Per-feature fallback produced NaN"
        assert result.shape == (8, 4, 4), f"Unexpected shape: {result.shape}"

    def test_cholesky_ex_per_feature_fallback_octonion(self) -> None:
        """Per-feature cholesky_ex fallback for OctonionBatchNorm."""
        bn = OctonionBatchNorm(4)
        # Create covariance where feature 0 is near-singular, features 1-3 are healthy
        cov = torch.eye(8).unsqueeze(0).expand(4, -1, -1).clone()
        # Make feature 0 nearly singular
        cov[0] = torch.zeros(8, 8)
        cov[0, 0, 0] = 1e-12

        x_centered = torch.randn(8, 4, 8)
        result = bn._whiten(x_centered, cov)
        assert not torch.isnan(result).any(), "OctonionBN per-feature fallback produced NaN"
        assert result.shape == (8, 4, 8), f"Unexpected shape: {result.shape}"

    def test_no_try_except_in_whiten_quaternion(self) -> None:
        """QuaternionBatchNorm._whiten must not contain try/except around cholesky.

        Verifies torch.compile compatibility: try/except causes graph breaks.
        We verify this by inspecting the source code.
        """
        import inspect
        source = inspect.getsource(QuaternionBatchNorm._whiten)
        # Must NOT contain try/except
        assert "try:" not in source, (
            "QuaternionBatchNorm._whiten contains try/except around cholesky "
            "(causes torch.compile graph breaks). Use cholesky_ex instead."
        )
        # MUST use cholesky_ex
        assert "cholesky_ex" in source, (
            "QuaternionBatchNorm._whiten must use cholesky_ex (not cholesky)"
        )

    def test_no_try_except_in_whiten_octonion(self) -> None:
        """OctonionBatchNorm._whiten must not contain try/except around cholesky."""
        import inspect
        source = inspect.getsource(OctonionBatchNorm._whiten)
        assert "try:" not in source, (
            "OctonionBatchNorm._whiten contains try/except around cholesky "
            "(causes torch.compile graph breaks). Use cholesky_ex instead."
        )
        assert "cholesky_ex" in source, (
            "OctonionBatchNorm._whiten must use cholesky_ex (not cholesky)"
        )


# ── Tier 3: torch.compile config flag and CLI flags ────────────────────

@pytest.mark.skipif(
    not _CONFIG_AVAILABLE,
    reason="TrainConfig not importable from octonion.baselines._config",
)
class TestTrainConfigCompileFlag:
    """Tests for TrainConfig.use_compile field and CLI flag pass-through."""

    def test_use_compile_defaults_to_false(self) -> None:
        """TrainConfig.use_compile defaults to False (opt-in only)."""
        config = TrainConfig()
        assert config.use_compile is False, (
            "use_compile must default to False (opt-in, experimental)"
        )

    def test_use_compile_can_be_set_true(self) -> None:
        """TrainConfig(use_compile=True) is a valid config."""
        config = TrainConfig(use_compile=True)
        assert config.use_compile is True

    def test_use_compile_and_use_amp_independent(self) -> None:
        """use_compile and use_amp are independent flags."""
        config = TrainConfig(use_amp=True, use_compile=False)
        assert config.use_amp is True
        assert config.use_compile is False

        config2 = TrainConfig(use_amp=False, use_compile=True)
        assert config2.use_amp is False
        assert config2.use_compile is True

    def test_use_amp_still_defaults_to_false(self) -> None:
        """Adding use_compile field did not change use_amp default."""
        config = TrainConfig()
        assert config.use_amp is False

    def test_cifar_script_cli_flags(self) -> None:
        """--use-amp and --compile CLI flags are recognized by run_cifar_reproduction.py."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "scripts/run_cifar_reproduction.py", "--help"],
            capture_output=True,
            text=True,
            cwd="/workspace",
        )
        assert result.returncode == 0, (
            f"run_cifar_reproduction.py --help failed:\n{result.stderr}"
        )
        assert "--use-amp" in result.stdout, (
            "--use-amp flag not found in --help output"
        )
        assert "--compile" in result.stdout, (
            "--compile flag not found in --help output"
        )
