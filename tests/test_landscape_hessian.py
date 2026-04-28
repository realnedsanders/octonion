"""Tests for Hessian eigenspectrum analysis and curvature measurement.

Covers:
- Full Hessian computation on toy quadratic with known eigenvalues
- Finite-difference cross-check on nn.Linear model
- Stochastic Lanczos validation against full Hessian
- Auto method selection based on parameter count
- Result dict field verification
- Edge case: single-param model
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from octonion.landscape._curvature import measure_curvature
from octonion.landscape._hessian import (
    compute_full_hessian,
    compute_hessian_spectrum,
    stochastic_lanczos,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ToyQuadratic(nn.Module):
    """f(x, y) = x^2 + 3*y^2.  Hessian eigenvalues are [2, 6]."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([1.0, 1.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ignore input, just use params as the function variables
        return self.w[0] ** 2 + 3.0 * self.w[1] ** 2


class SmallLinear(nn.Module):
    """Tiny nn.Linear wrapper for Hessian tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SingleParamModel(nn.Module):
    """Model with a single trainable parameter: f = p * x."""

    def __init__(self) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.p * x


def _mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((output - target) ** 2).mean()


def _dummy_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Ignores target, just returns output (used for quadratic model)."""
    return output


# ---------------------------------------------------------------------------
# Full Hessian tests
# ---------------------------------------------------------------------------


class TestFullHessian:
    """Tests for compute_full_hessian."""

    def test_full_hessian_toy(self) -> None:
        """Toy quadratic f(x,y) = x^2 + 3y^2 should have eigenvalues [2, 6]."""
        model = ToyQuadratic()
        # Dummy data (not used by model, but required by interface)
        x = torch.zeros(1, 1)
        y = torch.zeros(1, 1)

        result = compute_full_hessian(model, _dummy_loss, x, y)

        eigenvalues = np.sort(result["eigenvalues"])
        np.testing.assert_allclose(eigenvalues, [2.0, 6.0], atol=1e-4)
        assert result["method"] == "full"

    def test_full_hessian_nn_linear_vs_finite_diff(self) -> None:
        """Full Hessian on nn.Linear matches finite-difference Hessian within 1e-3."""
        torch.manual_seed(42)
        # Use float64 for finite-difference precision
        model = SmallLinear()
        model.double()
        x = torch.randn(10, 2, dtype=torch.float64)
        y = torch.randn(10, 1, dtype=torch.float64)

        # Save original params BEFORE any Hessian computation
        orig_params = torch.cat(
            [p.reshape(-1).detach().clone() for p in model.parameters()]
        )

        # Finite-difference Hessian (do this first so params are untouched)
        n = orig_params.numel()
        H_fd = torch.zeros(n, n, dtype=torch.float64)
        eps = 1e-5

        def _loss_at(flat: torch.Tensor) -> float:
            offset = 0
            for p in model.parameters():
                numel = p.numel()
                p.data.copy_(flat[offset:offset + numel].reshape(p.shape))
                offset += numel
            with torch.no_grad():
                out = model(x)
                return _mse_loss(out, y).item()

        for i in range(n):
            e_i = torch.zeros(n, dtype=torch.float64)
            e_i[i] = eps
            for j in range(i, n):  # Symmetric, only upper triangle
                e_j = torch.zeros(n, dtype=torch.float64)
                e_j[j] = eps
                fpp = _loss_at(orig_params + e_i + e_j)
                fpm = _loss_at(orig_params + e_i - e_j)
                fmp = _loss_at(orig_params - e_i + e_j)
                fmm = _loss_at(orig_params - e_i - e_j)
                val = (fpp - fpm - fmp + fmm) / (4 * eps * eps)
                H_fd[i, j] = val
                H_fd[j, i] = val

        # Restore original params
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            p.data.copy_(orig_params[offset:offset + numel].reshape(p.shape))
            offset += numel

        # Now compute full Hessian via autograd
        result = compute_full_hessian(model, _mse_loss, x, y)

        eigs_fd = np.sort(np.linalg.eigvalsh(H_fd.numpy()))
        eigs_full = np.sort(result["eigenvalues"])
        np.testing.assert_allclose(eigs_full, eigs_fd, atol=1e-3)

    def test_full_hessian_result_fields(self) -> None:
        """Result contains n_negative, n_positive, negative_ratio, trace, spectral_norm."""
        model = ToyQuadratic()
        x = torch.zeros(1, 1)
        y = torch.zeros(1, 1)

        result = compute_full_hessian(model, _dummy_loss, x, y)

        assert "eigenvalues" in result
        assert "n_negative" in result
        assert "n_positive" in result
        assert "n_zero" in result
        assert "trace" in result
        assert "spectral_norm" in result
        assert "negative_ratio" in result
        assert result["method"] == "full"

        # For a PD quadratic: no negative eigenvalues
        assert result["n_negative"] == 0
        assert result["n_positive"] == 2

    def test_full_hessian_single_param(self) -> None:
        """Single-parameter model works correctly."""
        model = SingleParamModel()
        x = torch.tensor([[1.0]])
        y = torch.tensor([[0.0]])

        result = compute_full_hessian(model, _mse_loss, x, y)

        assert result["eigenvalues"].shape == (1,)
        assert result["n_positive"] + result["n_negative"] + result["n_zero"] == 1


# ---------------------------------------------------------------------------
# Stochastic Lanczos tests
# ---------------------------------------------------------------------------


class TestStochasticLanczos:
    """Tests for stochastic_lanczos."""

    def test_lanczos_vs_full(self) -> None:
        """Lanczos top eigenvalue within 20% of full Hessian on small model."""
        torch.manual_seed(42)
        model = SmallLinear()
        x = torch.randn(10, 2)
        y = torch.randn(10, 1)

        full = compute_full_hessian(model, _mse_loss, x, y)
        lanczos = stochastic_lanczos(
            model, _mse_loss, x, y,
            n_iterations=50,
            n_samples=3,
        )

        top_full = np.max(np.abs(full["eigenvalues"]))
        top_lanczos = np.max(np.abs(lanczos["ritz_values"]))
        # Within 20% of full
        assert abs(top_lanczos - top_full) / (top_full + 1e-10) < 0.20, (
            f"Lanczos top eigenvalue {top_lanczos:.4f} not within 20% of "
            f"full {top_full:.4f}"
        )

    def test_lanczos_result_fields(self) -> None:
        """Lanczos result contains ritz_values, negative_ratio_approx."""
        torch.manual_seed(42)
        model = SmallLinear()
        x = torch.randn(10, 2)
        y = torch.randn(10, 1)

        result = stochastic_lanczos(
            model, _mse_loss, x, y,
            n_iterations=20,
            n_samples=2,
        )

        assert "ritz_values" in result
        assert "negative_ratio_approx" in result
        assert "trace_approx" in result
        assert result["method"] == "lanczos"
        assert result["n_iterations"] == 20
        assert result["n_samples"] == 2


# ---------------------------------------------------------------------------
# Auto method selection tests
# ---------------------------------------------------------------------------


class TestAutoSelection:
    """Tests for compute_hessian_spectrum method selection."""

    def test_auto_selects_full_small_model(self) -> None:
        """Auto selects 'full' for model with < 2000 params."""
        model = SmallLinear()
        x = torch.randn(5, 2)
        y = torch.randn(5, 1)

        result = compute_hessian_spectrum(model, _mse_loss, x, y)
        assert result["method"] == "full"

    def test_auto_selects_lanczos_large_model(self) -> None:
        """Auto selects 'lanczos' for model with > 2000 params."""

        class BigModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(100, 100)  # 10100 params

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        model = BigModel()
        x = torch.randn(5, 100)
        y = torch.randn(5, 100)

        result = compute_hessian_spectrum(
            model, _mse_loss, x, y,
            method="auto",
            max_full_params=2000,
        )
        assert result["method"] == "lanczos"


# ---------------------------------------------------------------------------
# Curvature measurement tests
# ---------------------------------------------------------------------------


class QuadraticModel(nn.Module):
    """Simple model whose loss surface has known positive curvature.

    Forward: output = linear(x), loss = MSE.
    The model is "trained" to a fixed point so curvature is well-defined.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestCurvature:
    """Tests for measure_curvature."""

    def test_curvature_quadratic(self) -> None:
        """Curvature on a simple trained model should be positive."""
        torch.manual_seed(42)
        model = QuadraticModel()

        # Create simple data that model "converges" on
        x = torch.randn(20, 4)
        y = x[:, 0:1] + 0.5 * x[:, 1:2]  # Deterministic target

        # "Train" model briefly to a decent point
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(100):
            optimizer.zero_grad()
            loss = _mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        result = measure_curvature(
            model, _mse_loss, x, y,
            n_directions=10,
            n_steps=21,
            seed=42,
        )

        # Curvature should be positive around a minimum
        assert result["mean_curvature"] > 0, (
            f"Expected positive mean curvature, got {result['mean_curvature']}"
        )

    def test_curvature_restores_weights(self) -> None:
        """Model parameters should be unchanged after curvature measurement."""
        torch.manual_seed(42)
        model = QuadraticModel()
        x = torch.randn(10, 4)
        y = torch.randn(10, 1)

        # Save original weights
        orig_state = {
            name: p.clone() for name, p in model.named_parameters()
        }

        measure_curvature(model, _mse_loss, x, y, n_directions=5, n_steps=11)

        # Verify weights restored
        for name, p in model.named_parameters():
            torch.testing.assert_close(
                p.data, orig_state[name],
                msg=f"Parameter {name} was not restored after curvature measurement"
            )

    def test_curvature_result_structure(self) -> None:
        """Result dict has all expected keys."""
        torch.manual_seed(42)
        model = QuadraticModel()
        x = torch.randn(10, 4)
        y = torch.randn(10, 1)

        result = measure_curvature(model, _mse_loss, x, y, n_directions=3, n_steps=11)

        assert "mean_curvature" in result
        assert "median_curvature" in result
        assert "std_curvature" in result
        assert "curvatures" in result
        assert "n_directions" in result

    def test_curvature_n_directions(self) -> None:
        """Curvatures list length matches n_directions."""
        torch.manual_seed(42)
        model = QuadraticModel()
        x = torch.randn(10, 4)
        y = torch.randn(10, 1)

        n_dir = 7
        result = measure_curvature(model, _mse_loss, x, y, n_directions=n_dir, n_steps=11)

        assert len(result["curvatures"]) == n_dir
        assert result["n_directions"] == n_dir
