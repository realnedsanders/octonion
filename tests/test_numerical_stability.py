"""Smoke tests for Phase 4 numerical stability measurement infrastructure.

Verifies that all Phase 4 measurement code runs without error and produces
sane results. These are smoke tests only -- they do NOT assert on SC
threshold values. All tests run on CPU only (no GPU required).

Tests cover:
- SC-1 proxy: StabilizingNorm forward pass and unit-norm output
- SC-2 proxy: Depth sweep float32 vs float64 relative error measurement
- SC-3 proxy: Condition number of octonion multiplication Jacobian
- SC-4 proxy: Single-layer dtype comparison producing nonzero error
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# ── SC-1: StabilizingNorm ─────────────────────────────────────────


@pytest.mark.parametrize("algebra_dim", [1, 2, 4, 8])
def test_stabilizing_norm(algebra_dim: int) -> None:
    """StabilizingNorm forward pass runs and preserves shape for all algebra dims."""
    torch.manual_seed(42)
    from octonion.baselines._stabilization import StabilizingNorm

    m = StabilizingNorm(algebra_dim)
    if algebra_dim == 1:
        x = torch.randn(4, 16)  # [batch, features]
    else:
        x = torch.randn(4, 16, algebra_dim)  # [batch, features, dim]
    out = m(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("algebra_dim", [1, 2, 4, 8])
def test_stabilizing_norm_output_norm(algebra_dim: int) -> None:
    """StabilizingNorm output has unit norm for all algebra dims."""
    torch.manual_seed(42)
    from octonion.baselines._stabilization import StabilizingNorm

    m = StabilizingNorm(algebra_dim)
    if algebra_dim == 1:
        x = torch.randn(4, 16) * 5.0  # non-unit magnitude
        out = m(x)
        # Real: each element should have |out| = 1.0
        assert torch.allclose(out.abs(), torch.ones_like(out), atol=1e-6)
    else:
        x = torch.randn(4, 16, algebra_dim) * 5.0
        out = m(x)
        norms = out.norm(dim=-1)  # [batch, features]
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


# ── SC-2: Depth Sweep Measurement ─────────────────────────────────


def test_depth_sweep_smoke() -> None:
    """Stripped chain of 3 OctonionDenseLinear layers: float32 vs float64."""
    torch.manual_seed(42)
    from octonion.baselines._algebra_linear import OctonionDenseLinear

    depth = 3
    hidden = 4
    # Build at float64
    layers_f64 = nn.ModuleList(
        [
            OctonionDenseLinear(hidden, hidden, bias=False, dtype=torch.float64)
            for _ in range(depth)
        ]
    )
    # Clone to float32
    layers_f32 = nn.ModuleList(
        [
            OctonionDenseLinear(hidden, hidden, bias=False, dtype=torch.float32)
            for _ in range(depth)
        ]
    )
    with torch.no_grad():
        for l64, l32 in zip(layers_f64, layers_f32):
            for p32, p64 in zip(l32.parameters(), l64.parameters()):
                p32.copy_(p64.float())

    x = torch.randn(2, hidden, 8, dtype=torch.float64)
    with torch.no_grad():
        h64 = x
        for layer in layers_f64:
            h64 = layer(h64)
        h32 = x.float()
        for layer in layers_f32:
            h32 = layer(h32)

    rel_err = (h32.double() - h64).norm() / (h64.norm() + 1e-30)
    assert torch.isfinite(rel_err)
    assert rel_err.item() > 0  # not exactly zero
    assert rel_err.item() < 1.0  # not diverged


# ── SC-3: Condition Number ─────────────────────────────────────────


def test_condition_number_smoke() -> None:
    """Condition number of octonion multiplication at a random point."""
    torch.manual_seed(42)
    from octonion._multiplication import octonion_mul
    from octonion.calculus._numeric import numeric_jacobian

    a = torch.randn(8, dtype=torch.float64)
    fn = lambda x: octonion_mul(a, x)
    J = numeric_jacobian(fn, torch.randn(8, dtype=torch.float64))
    sv = torch.linalg.svdvals(J)
    cond = (sv[0] / sv[-1].clamp(min=1e-30)).item()
    assert cond >= 1.0
    assert torch.isfinite(torch.tensor(cond))


# ── SC-4: Dtype Comparison ─────────────────────────────────────────


def test_dtype_comparison_smoke() -> None:
    """Single OctonionDenseLinear: float32 output differs from float64."""
    torch.manual_seed(42)
    from octonion.baselines._algebra_linear import OctonionDenseLinear

    layer64 = OctonionDenseLinear(4, 4, bias=False, dtype=torch.float64)
    layer32 = OctonionDenseLinear(4, 4, bias=False, dtype=torch.float32)
    with torch.no_grad():
        for p32, p64 in zip(layer32.parameters(), layer64.parameters()):
            p32.copy_(p64.float())
    x = torch.randn(2, 4, 8, dtype=torch.float64)
    x = x / x.norm(dim=-1, keepdim=True)  # unit magnitude
    with torch.no_grad():
        out64 = layer64(x)
        out32 = layer32(x.float())
    rel_err = (out32.double() - out64).norm() / (out64.norm() + 1e-30)
    assert rel_err.item() > 0
    assert rel_err.item() < 1.0
