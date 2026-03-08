"""Tests for OctonionLinear nn.Module layer.

Covers:
- Output shape and validity
- Learnable parameters a, b of shape [8]
- Forward pass is differentiable (gradients flow)
- Optimizer step changes output
"""

import torch
import torch.nn as nn

from octonion._linear import OctonionLinear


class TestOctonionLinear:
    """Tests for the OctonionLinear neural network layer."""

    def test_output_shape_single(self) -> None:
        """OctonionLinear produces valid [8] output for [8] input."""
        layer = OctonionLinear(dtype=torch.float64)
        x = torch.randn(8, dtype=torch.float64)
        y = layer(x)
        assert y.shape == (8,), f"Expected shape (8,), got {y.shape}"

    def test_output_shape_batched(self) -> None:
        """OctonionLinear produces valid [N, 8] output for [N, 8] input."""
        layer = OctonionLinear(dtype=torch.float64)
        x = torch.randn(5, 8, dtype=torch.float64)
        y = layer(x)
        assert y.shape == (5, 8), f"Expected shape (5, 8), got {y.shape}"

    def test_learnable_parameters(self) -> None:
        """OctonionLinear has learnable parameters a and b of shape [8]."""
        layer = OctonionLinear(dtype=torch.float64)
        assert hasattr(layer, "a")
        assert hasattr(layer, "b")
        assert isinstance(layer.a, nn.Parameter)
        assert isinstance(layer.b, nn.Parameter)
        assert layer.a.shape == (8,), f"Expected a shape (8,), got {layer.a.shape}"
        assert layer.b.shape == (8,), f"Expected b shape (8,), got {layer.b.shape}"

    def test_forward_differentiable(self) -> None:
        """Gradients flow through OctonionLinear.forward."""
        layer = OctonionLinear(dtype=torch.float64)
        x = torch.randn(8, dtype=torch.float64, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        # Gradients should exist for parameters and input
        assert layer.a.grad is not None, "Gradient for parameter a should exist"
        assert layer.b.grad is not None, "Gradient for parameter b should exist"
        assert x.grad is not None, "Gradient for input x should exist"

    def test_optimizer_step_changes_output(self) -> None:
        """Output changes after an optimizer step (learning signal propagates)."""
        torch.manual_seed(42)
        layer = OctonionLinear(dtype=torch.float64)
        x = torch.randn(8, dtype=torch.float64)

        # Forward pass before optimizer step
        y_before = layer(x).detach().clone()

        # Compute loss and backward
        y = layer(x)
        loss = y.sum()
        loss.backward()

        # Optimizer step
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
        optimizer.step()

        # Forward pass after optimizer step
        y_after = layer(x).detach()

        assert not torch.allclose(y_before, y_after), (
            "Output should change after optimizer step"
        )

    def test_parameters_initialized_unit_norm(self) -> None:
        """Parameters a and b are initialized with approximately unit norm."""
        layer = OctonionLinear(dtype=torch.float64)
        a_norm = torch.linalg.norm(layer.a.data)
        b_norm = torch.linalg.norm(layer.b.data)
        assert torch.allclose(a_norm, torch.tensor(1.0, dtype=torch.float64), atol=1e-6)
        assert torch.allclose(b_norm, torch.tensor(1.0, dtype=torch.float64), atol=1e-6)
