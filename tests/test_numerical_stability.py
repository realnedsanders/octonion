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

import copy
import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from octonion.baselines._algebra_linear import (
    ComplexLinear,
    OctonionDenseLinear,
    QuaternionLinear,
    RealLinear,
)
from octonion.baselines._config import AlgebraType

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


# ── Measurement Integrity Tests ──────────────────────────────────


@pytest.mark.parametrize(
    "algebra,layer_cls",
    [
        (AlgebraType.REAL, RealLinear),
        (AlgebraType.COMPLEX, ComplexLinear),
        (AlgebraType.QUATERNION, QuaternionLinear),
        (AlgebraType.OCTONION, OctonionDenseLinear),
    ],
)
def test_full_network_float64_all_algebras(algebra, layer_cls) -> None:
    """AlgebraNetwork at float64 with BN: forward pass works for R/C/H/O."""
    torch.manual_seed(42)
    from octonion.baselines._config import NetworkConfig
    from octonion.baselines._network import AlgebraNetwork

    hidden = 4
    config = NetworkConfig(
        algebra=algebra,
        topology="mlp",
        depth=2,
        base_hidden=hidden,
        activation="split_relu",
        output_projection="flatten",
        use_batchnorm=True,
        input_dim=hidden * algebra.dim,
        output_dim=hidden * algebra.dim,
    )
    model = AlgebraNetwork(config).to(torch.float64)

    # Warmup BN running stats in train mode
    model.train()
    with torch.no_grad():
        for _ in range(5):
            x = torch.randn(8, hidden * algebra.dim, dtype=torch.float64)
            model(x)

    # Eval mode forward pass
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, hidden * algebra.dim, dtype=torch.float64)
        out = model(x)

    assert out.shape[0] == 2
    assert torch.isfinite(out).all(), f"{algebra.short_name} float64 output contains non-finite values"


def test_stripped_chain_depth500_no_nan() -> None:
    """Depth 500 stripped chain: all errors are finite or inf, never NaN."""
    torch.manual_seed(42)
    depth = 500
    hidden = 4
    n_samples = 10

    layers_f64 = nn.ModuleList(
        [OctonionDenseLinear(hidden, hidden, bias=False, dtype=torch.float64)
         for _ in range(depth)]
    )
    layers_f32 = copy.deepcopy(layers_f64).float()
    layers_f64.eval()
    layers_f32.eval()

    errors = []
    with torch.no_grad():
        for i in range(n_samples):
            torch.manual_seed(42 + i + 1)
            x64 = torch.randn(1, hidden, 8, dtype=torch.float64)
            x32 = x64.float()

            h64, h32 = x64, x32
            for l64, l32 in zip(layers_f64, layers_f32):
                h64 = l64(h64)
                h32 = l32(h32)

            # Apply the same guard logic as the analysis script
            if not torch.isfinite(h32).all() or not torch.isfinite(h64).all() or h64.norm().item() <= 1e-30:
                errors.append(float("inf"))
            else:
                rel_err = (h32.double() - h64).norm() / h64.norm()
                errors.append(rel_err.item())

    assert len(errors) == n_samples, "Every sample must produce an error entry"
    for e in errors:
        assert not np.isnan(e), "Got NaN error — guard logic is broken"
        assert e >= 0 or np.isinf(e), f"Error must be non-negative or inf, got {e}"


def test_json_serialization_no_nan_infinity() -> None:
    """_sanitize_for_json produces valid JSON with no bare NaN/Infinity."""
    # Import from the script by adding its directory to sys.path
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from analyze_stability import _sanitize_for_json

    data = {
        "a": float("nan"),
        "b": float("inf"),
        "c": float("-inf"),
        "d": 1.5,
        "nested": {"x": float("nan"), "y": [float("inf"), 2.0]},
    }
    sanitized = _sanitize_for_json(data)
    json_str = json.dumps(sanitized)

    # Must be valid JSON (no bare NaN/Infinity tokens)
    parsed = json.loads(json_str)
    assert parsed["a"] is None
    assert parsed["b"] is None
    assert parsed["c"] is None
    assert parsed["d"] == 1.5
    assert parsed["nested"]["x"] is None
    assert parsed["nested"]["y"][0] is None
    assert parsed["nested"]["y"][1] == 2.0


def test_condition_number_composition_octonion() -> None:
    """2-layer O chain: condition number is finite or inf, never NaN."""
    torch.manual_seed(42)
    from octonion.calculus._numeric import numeric_jacobian

    hidden = 4
    layers = nn.ModuleList(
        [OctonionDenseLinear(hidden, hidden, bias=False, dtype=torch.float64)
         for _ in range(2)]
    )
    layers.eval()

    in_dim = hidden * 8

    def chain_fn(x, _layers=layers):
        h = x.reshape(hidden, 8)
        for layer in _layers:
            h = layer(h)
        return h.reshape(-1)

    x = torch.randn(in_dim, dtype=torch.float64)
    x = x / x.norm()

    J = numeric_jacobian(chain_fn, x)
    sv = torch.linalg.svdvals(J)
    cond = (sv[0] / sv[-1].clamp(min=1e-30)).item()

    assert not np.isnan(cond), "Condition number must not be NaN"
    assert cond >= 1.0, "Condition number must be >= 1"
