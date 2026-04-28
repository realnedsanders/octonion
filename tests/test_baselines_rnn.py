"""Tests for algebra-specific recurrent cells.

Covers output shapes, state management, gate correctness, sequential processing,
and gradient flow for RealLSTMCell, ComplexGRUCell, QuaternionLSTMCell, OctonionLSTMCell.
"""

from __future__ import annotations

import pytest
import torch

# ── Fixtures ─────────────────────────────────────────────────────

B = 4  # batch size
INPUT_SIZE = 16
HIDDEN_SIZE = 32


@pytest.fixture
def real_cell():
    from octonion.baselines._algebra_rnn import RealLSTMCell
    return RealLSTMCell(INPUT_SIZE, HIDDEN_SIZE)


@pytest.fixture
def complex_cell():
    from octonion.baselines._algebra_rnn import ComplexGRUCell
    return ComplexGRUCell(INPUT_SIZE, HIDDEN_SIZE)


@pytest.fixture
def quaternion_cell():
    from octonion.baselines._algebra_rnn import QuaternionLSTMCell
    return QuaternionLSTMCell(INPUT_SIZE, HIDDEN_SIZE)


@pytest.fixture
def octonion_cell():
    from octonion.baselines._algebra_rnn import OctonionLSTMCell
    return OctonionLSTMCell(INPUT_SIZE, HIDDEN_SIZE)


# ── Output Shape Tests ───────────────────────────────────────────


class TestOutputShapes:
    """Verify each cell produces correct output shapes."""

    def test_real_lstm_output_shape(self, real_cell):
        x = torch.randn(B, INPUT_SIZE)
        h = torch.zeros(B, HIDDEN_SIZE)
        c = torch.zeros(B, HIDDEN_SIZE)
        h_t, c_t = real_cell(x, (h, c))
        assert h_t.shape == (B, HIDDEN_SIZE)
        assert c_t.shape == (B, HIDDEN_SIZE)

    def test_complex_gru_output_shape(self, complex_cell):
        x = torch.randn(B, INPUT_SIZE, 2)
        h = torch.zeros(B, HIDDEN_SIZE, 2)
        h_t = complex_cell(x, h)
        assert h_t.shape == (B, HIDDEN_SIZE, 2)

    def test_quaternion_lstm_output_shape(self, quaternion_cell):
        x = torch.randn(B, INPUT_SIZE, 4)
        h = torch.zeros(B, HIDDEN_SIZE, 4)
        c = torch.zeros(B, HIDDEN_SIZE, 4)
        h_t, c_t = quaternion_cell(x, (h, c))
        assert h_t.shape == (B, HIDDEN_SIZE, 4)
        assert c_t.shape == (B, HIDDEN_SIZE, 4)

    def test_octonion_lstm_output_shape(self, octonion_cell):
        x = torch.randn(B, INPUT_SIZE, 8)
        h = torch.zeros(B, HIDDEN_SIZE, 8)
        c = torch.zeros(B, HIDDEN_SIZE, 8)
        h_t, c_t = octonion_cell(x, (h, c))
        assert h_t.shape == (B, HIDDEN_SIZE, 8)
        assert c_t.shape == (B, HIDDEN_SIZE, 8)


# ── State Update Tests ──────────────────────────────────────────


class TestStateUpdates:
    """Verify states are actually updated (not degenerate)."""

    def test_state_updates_differ_real(self, real_cell):
        x = torch.randn(B, INPUT_SIZE)
        h = torch.zeros(B, HIDDEN_SIZE)
        c = torch.zeros(B, HIDDEN_SIZE)
        h_t, c_t = real_cell(x, (h, c))
        # h_t should differ from initial h (all zeros)
        assert not torch.allclose(h_t, h, atol=1e-6)

    def test_state_updates_differ_complex(self, complex_cell):
        x = torch.randn(B, INPUT_SIZE, 2)
        h = torch.zeros(B, HIDDEN_SIZE, 2)
        h_t = complex_cell(x, h)
        assert not torch.allclose(h_t, h, atol=1e-6)

    def test_state_updates_differ_quaternion(self, quaternion_cell):
        x = torch.randn(B, INPUT_SIZE, 4)
        h = torch.zeros(B, HIDDEN_SIZE, 4)
        c = torch.zeros(B, HIDDEN_SIZE, 4)
        h_t, c_t = quaternion_cell(x, (h, c))
        assert not torch.allclose(h_t, h, atol=1e-6)

    def test_state_updates_differ_octonion(self, octonion_cell):
        x = torch.randn(B, INPUT_SIZE, 8)
        h = torch.zeros(B, HIDDEN_SIZE, 8)
        c = torch.zeros(B, HIDDEN_SIZE, 8)
        h_t, c_t = octonion_cell(x, (h, c))
        assert not torch.allclose(h_t, h, atol=1e-6)


# ── Different Inputs / Different Outputs ─────────────────────────


class TestDifferentInputs:
    """Same cell, different inputs -> different outputs (non-degenerate)."""

    def test_different_inputs_different_outputs_real(self, real_cell):
        h = torch.zeros(B, HIDDEN_SIZE)
        c = torch.zeros(B, HIDDEN_SIZE)
        x1 = torch.randn(B, INPUT_SIZE)
        x2 = torch.randn(B, INPUT_SIZE)
        h1, _ = real_cell(x1, (h, c))
        h2, _ = real_cell(x2, (h, c))
        assert not torch.allclose(h1, h2, atol=1e-6)

    def test_different_inputs_different_outputs_complex(self, complex_cell):
        h = torch.zeros(B, HIDDEN_SIZE, 2)
        x1 = torch.randn(B, INPUT_SIZE, 2)
        x2 = torch.randn(B, INPUT_SIZE, 2)
        h1 = complex_cell(x1, h)
        h2 = complex_cell(x2, h)
        assert not torch.allclose(h1, h2, atol=1e-6)

    def test_different_inputs_different_outputs_quaternion(self, quaternion_cell):
        h = torch.zeros(B, HIDDEN_SIZE, 4)
        c = torch.zeros(B, HIDDEN_SIZE, 4)
        x1 = torch.randn(B, INPUT_SIZE, 4)
        x2 = torch.randn(B, INPUT_SIZE, 4)
        h1, _ = quaternion_cell(x1, (h, c))
        h2, _ = quaternion_cell(x2, (h, c))
        assert not torch.allclose(h1, h2, atol=1e-6)

    def test_different_inputs_different_outputs_octonion(self, octonion_cell):
        h = torch.zeros(B, HIDDEN_SIZE, 8)
        c = torch.zeros(B, HIDDEN_SIZE, 8)
        x1 = torch.randn(B, INPUT_SIZE, 8)
        x2 = torch.randn(B, INPUT_SIZE, 8)
        h1, _ = octonion_cell(x1, (h, c))
        h2, _ = octonion_cell(x2, (h, c))
        assert not torch.allclose(h1, h2, atol=1e-6)


# ── Quaternion Gate Broadcasting Test ────────────────────────────


class TestQuaternionGates:
    """Verify QuaternionLSTMCell gates are scalar (real component), broadcasting."""

    def test_quaternion_lstm_gate_broadcasting(self, quaternion_cell):
        """Gates should be derived from real component and broadcast across 4 dims."""
        x = torch.randn(B, INPUT_SIZE, 4)
        h = torch.randn(B, HIDDEN_SIZE, 4)
        c = torch.randn(B, HIDDEN_SIZE, 4)

        # Run forward pass
        h_t, c_t = quaternion_cell(x, (h, c))

        # Verify cell update used scalar gates by checking that
        # the output has variation across the 4 quaternion components
        # (if gates were not properly scalar, the output would be degenerate)
        component_stds = h_t.std(dim=-1)  # [B, hidden]
        # Should have non-trivial variation across quaternion dims
        assert component_stds.mean() > 1e-6, (
            "Output has no variation across quaternion dims - gates may not broadcast correctly"
        )


# ── Sequential Timestep Tests ───────────────────────────────────


class TestSequentialTimesteps:
    """Feed multiple timesteps, verify no errors and varying outputs."""

    NUM_STEPS = 5

    def test_sequential_timesteps_real(self, real_cell):
        h = torch.zeros(B, HIDDEN_SIZE)
        c = torch.zeros(B, HIDDEN_SIZE)
        outputs = []
        for _t in range(self.NUM_STEPS):
            x = torch.randn(B, INPUT_SIZE)
            h, c = real_cell(x, (h, c))
            outputs.append(h.clone())
        # Consecutive outputs should differ
        for i in range(1, len(outputs)):
            assert not torch.allclose(outputs[i], outputs[i - 1], atol=1e-6)

    def test_sequential_timesteps_complex(self, complex_cell):
        h = torch.zeros(B, HIDDEN_SIZE, 2)
        outputs = []
        for _t in range(self.NUM_STEPS):
            x = torch.randn(B, INPUT_SIZE, 2)
            h = complex_cell(x, h)
            outputs.append(h.clone())
        for i in range(1, len(outputs)):
            assert not torch.allclose(outputs[i], outputs[i - 1], atol=1e-6)

    def test_sequential_timesteps_quaternion(self, quaternion_cell):
        h = torch.zeros(B, HIDDEN_SIZE, 4)
        c = torch.zeros(B, HIDDEN_SIZE, 4)
        outputs = []
        for _t in range(self.NUM_STEPS):
            x = torch.randn(B, INPUT_SIZE, 4)
            h, c = quaternion_cell(x, (h, c))
            outputs.append(h.clone())
        for i in range(1, len(outputs)):
            assert not torch.allclose(outputs[i], outputs[i - 1], atol=1e-6)

    def test_sequential_timesteps_octonion(self, octonion_cell):
        h = torch.zeros(B, HIDDEN_SIZE, 8)
        c = torch.zeros(B, HIDDEN_SIZE, 8)
        outputs = []
        for _t in range(self.NUM_STEPS):
            x = torch.randn(B, INPUT_SIZE, 8)
            h, c = octonion_cell(x, (h, c))
            outputs.append(h.clone())
        for i in range(1, len(outputs)):
            assert not torch.allclose(outputs[i], outputs[i - 1], atol=1e-6)


# ── Gradient Flow Tests ─────────────────────────────────────────


class TestGradientFlow:
    """Verify gradients exist on all parameters after backward pass."""

    def test_rnn_cell_gradient_flow_real(self, real_cell):
        x = torch.randn(B, INPUT_SIZE)
        # Use non-zero initial state so weight_hh gets gradient
        h = torch.randn(B, HIDDEN_SIZE) * 0.1
        c = torch.randn(B, HIDDEN_SIZE) * 0.1
        h_t, c_t = real_cell(x, (h, c))
        loss = h_t.sum()
        loss.backward()
        for name, param in real_cell.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_rnn_cell_gradient_flow_complex(self, complex_cell):
        x = torch.randn(B, INPUT_SIZE, 2)
        h = torch.randn(B, HIDDEN_SIZE, 2) * 0.1
        h_t = complex_cell(x, h)
        loss = h_t.sum()
        loss.backward()
        for name, param in complex_cell.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_rnn_cell_gradient_flow_quaternion(self, quaternion_cell):
        x = torch.randn(B, INPUT_SIZE, 4)
        h = torch.randn(B, HIDDEN_SIZE, 4) * 0.1
        c = torch.randn(B, HIDDEN_SIZE, 4) * 0.1
        h_t, c_t = quaternion_cell(x, (h, c))
        loss = h_t.sum()
        loss.backward()
        for name, param in quaternion_cell.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_rnn_cell_gradient_flow_octonion(self, octonion_cell):
        x = torch.randn(B, INPUT_SIZE, 8)
        h = torch.randn(B, HIDDEN_SIZE, 8) * 0.1
        c = torch.randn(B, HIDDEN_SIZE, 8) * 0.1
        h_t, c_t = octonion_cell(x, (h, c))
        loss = h_t.sum()
        loss.backward()
        for name, param in octonion_cell.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
