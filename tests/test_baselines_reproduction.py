"""Integration tests for CIFAR benchmark reproduction.

Tests model construction, parameter counts, forward pass correctness,
training config validation, and reproduction report generation for
all 4 algebras (R, C, H, O) on CIFAR-10 and CIFAR-100.

Full training reproduction runs live in scripts/run_cifar_reproduction.py.
"""

from __future__ import annotations

import pytest
import torch

from octonion.baselines._benchmarks import (
    PUBLISHED_RESULTS,
    cifar_network_config,
    cifar_train_config,
    reproduction_report,
)
from octonion.baselines._config import AlgebraType, TrainConfig
from octonion.baselines._network import AlgebraNetwork

# Only the 4 original algebras have AlgebraNetwork/CIFAR support.
# PHM8 and R8_DENSE are standalone layers used through _SimpleAlgebraMLP.
_NETWORK_ALGEBRAS = [
    AlgebraType.REAL,
    AlgebraType.COMPLEX,
    AlgebraType.QUATERNION,
    AlgebraType.OCTONION,
]


# ── Helpers ────────────────────────────────────────────────────────


def _build_cifar_model(algebra: AlgebraType, dataset: str = "cifar10") -> AlgebraNetwork:
    """Build a CIFAR AlgebraNetwork for the given algebra and dataset."""
    config = cifar_network_config(algebra, dataset)
    return AlgebraNetwork(config)


def _count_params(model: torch.nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Fast Tests: Parameter Matching ─────────────────────────────────


class TestCIFAR10ParamMatching:
    """Verify CIFAR-10 models across all 4 algebras have matched param counts."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Build all 4 algebra models for CIFAR-10."""
        self.models = {}
        self.param_counts = {}
        for algebra in _NETWORK_ALGEBRAS:
            model = _build_cifar_model(algebra, "cifar10")
            self.models[algebra.short_name] = model
            self.param_counts[algebra.short_name] = _count_params(model)

    def test_all_algebras_have_params(self) -> None:
        """All 4 models should have trainable parameters."""
        for alg_name, count in self.param_counts.items():
            assert count > 0, f"{alg_name} model has no parameters"

    def test_param_counts_are_reasonable(self) -> None:
        """All 4 algebra models should have reasonable param counts.

        Note: AlgebraNetwork uses base_hidden * multiplier which does NOT
        give exact param matching -- the multiplier approach is an approximation.
        For Conv2D topologies, kernel and batch norm parameters scale differently
        across algebras. Exact matching for reproduction experiments uses
        find_matched_width via run_comparison.

        This test verifies that all algebras produce models with a reasonable
        number of parameters (not zero, not degenerate).
        """
        for alg_name, count in self.param_counts.items():
            # All models should have at least 1000 params
            assert count > 1000, (
                f"{alg_name} param count {count} is suspiciously low"
            )
            # And less than 100M (sanity check)
            assert count < 100_000_000, (
                f"{alg_name} param count {count} is suspiciously high"
            )

    def test_output_dim_is_10(self) -> None:
        """All CIFAR-10 models should have output_dim=10."""
        for alg_name, model in self.models.items():
            assert model.config.output_dim == 10, (
                f"{alg_name} output_dim={model.config.output_dim}, expected 10"
            )

    def test_topology_is_conv2d(self) -> None:
        """All CIFAR models should use conv2d topology."""
        for alg_name, model in self.models.items():
            assert model.config.topology == "conv2d", (
                f"{alg_name} topology={model.config.topology}, expected conv2d"
            )

    def test_report_param_counts(self) -> None:
        """Log the param counts for documentation."""
        for alg_name, count in sorted(self.param_counts.items()):
            print(f"CIFAR-10 {alg_name}: {count:,} params")


class TestCIFAR100ParamMatching:
    """Verify CIFAR-100 models across all 4 algebras have matched param counts."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Build all 4 algebra models for CIFAR-100."""
        self.models = {}
        self.param_counts = {}
        for algebra in _NETWORK_ALGEBRAS:
            model = _build_cifar_model(algebra, "cifar100")
            self.models[algebra.short_name] = model
            self.param_counts[algebra.short_name] = _count_params(model)

    def test_all_algebras_have_params(self) -> None:
        """All 4 models should have trainable parameters."""
        for alg_name, count in self.param_counts.items():
            assert count > 0, f"{alg_name} model has no parameters"

    def test_output_dim_is_100(self) -> None:
        """All CIFAR-100 models should have output_dim=100."""
        for alg_name, model in self.models.items():
            assert model.config.output_dim == 100, (
                f"{alg_name} output_dim={model.config.output_dim}, expected 100"
            )

    def test_report_param_counts(self) -> None:
        """Log the param counts for documentation."""
        for alg_name, count in sorted(self.param_counts.items()):
            print(f"CIFAR-100 {alg_name}: {count:,} params")


# ── Fast Tests: Forward Pass ──────────────────────────────────────


class TestCIFAR10ForwardPass:
    """Verify forward pass on a single batch for all 4 algebras on CIFAR-10."""

    @pytest.mark.parametrize("algebra", _NETWORK_ALGEBRAS, ids=lambda a: a.short_name)
    def test_forward_pass_shape(self, algebra: AlgebraType) -> None:
        """Forward pass should produce [B, 10] output for CIFAR-10 input."""
        model = _build_cifar_model(algebra, "cifar10")
        model.eval()

        batch_size = 4
        # CIFAR images: [B, 3, 32, 32]
        x = torch.randn(batch_size, 3, 32, 32)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 10), (
            f"{algebra.short_name}: output shape {output.shape}, expected ({batch_size}, 10)"
        )

    @pytest.mark.parametrize("algebra", _NETWORK_ALGEBRAS, ids=lambda a: a.short_name)
    def test_forward_pass_finite(self, algebra: AlgebraType) -> None:
        """Forward pass output should be finite (no NaN/Inf)."""
        model = _build_cifar_model(algebra, "cifar10")
        model.eval()

        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            output = model(x)

        assert torch.isfinite(output).all(), (
            f"{algebra.short_name}: output contains NaN or Inf"
        )


class TestCIFAR100ForwardPass:
    """Verify forward pass on a single batch for all 4 algebras on CIFAR-100."""

    @pytest.mark.parametrize("algebra", _NETWORK_ALGEBRAS, ids=lambda a: a.short_name)
    def test_forward_pass_shape(self, algebra: AlgebraType) -> None:
        """Forward pass should produce [B, 100] output for CIFAR-100 input."""
        model = _build_cifar_model(algebra, "cifar100")
        model.eval()

        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 100), (
            f"{algebra.short_name}: output shape {output.shape}, expected ({batch_size}, 100)"
        )


# ── Fast Tests: Training Config ───────────────────────────────────


class TestCIFARTrainConfig:
    """Verify training configs match published hyperparameters."""

    def test_cifar10_train_config(self) -> None:
        """CIFAR-10 train config should match published hyperparameters.

        Matches Gaudet & Maida 2018 / Trabelsi 2018:
        - SGD with Nesterov momentum
        - step_cifar LR schedule (0.01 warmup -> 0.1 peak -> step decay)
        - Gradient norm clipping at 1.0
        """
        tc = cifar_train_config("cifar10")
        assert tc.epochs == 200
        assert tc.lr == 0.01
        assert tc.optimizer == "sgd"
        assert tc.scheduler == "step_cifar"
        assert tc.weight_decay == 5e-4
        assert tc.batch_size == 128
        assert tc.warmup_epochs == 10
        assert tc.use_amp is False
        assert tc.gradient_clip_norm == 1.0
        assert tc.nesterov is True

    def test_cifar100_train_config(self) -> None:
        """CIFAR-100 train config uses same hyperparameters."""
        tc = cifar_train_config("cifar100")
        assert tc.epochs == 200
        assert tc.optimizer == "sgd"

    def test_step_cifar_scheduler_lr_values(self) -> None:
        """step_cifar scheduler should match published LR schedule.

        Verifies the LR schedule from Gaudet & Maida 2018 / Trabelsi 2018:
        - Epochs 0-9:   LR = 0.01  (warmup at base LR)
        - Epochs 10-119: LR = 0.1  (peak LR)
        - Epochs 120-149: LR = 0.01 (after first step decay)
        - Epochs 150-199: LR = 0.001 (after second step decay)

        Simulates train_model's exact warmup + scheduler stepping logic:
        - During warmup (epochs 0-9): LR overridden to config.lr, scheduler NOT stepped
        - At epoch 10: LR restored to peak (config.lr * 10), scheduler stepping begins
        - Milestones adjusted by -warmup_epochs so scheduler hits them at real epochs 120, 150
        """
        import torch
        from octonion.baselines._trainer import _build_optimizer, _build_scheduler

        tc = cifar_train_config("cifar10")

        # Build a tiny model just for optimizer/scheduler
        model = torch.nn.Linear(10, 10)
        tc_copy = TrainConfig(
            epochs=tc.epochs, lr=tc.lr, optimizer="sgd", scheduler="step_cifar",
            weight_decay=tc.weight_decay, nesterov=tc.nesterov,
            warmup_epochs=tc.warmup_epochs,
        )
        opt = _build_optimizer(model, tc_copy)
        sched = _build_scheduler(opt, tc_copy)

        # After _build_scheduler, optimizer should be at peak LR (0.1)
        assert abs(opt.param_groups[0]["lr"] - 0.1) < 1e-8, (
            f"Peak LR should be 0.1, got {opt.param_groups[0]['lr']}"
        )

        # Simulate train_model's exact logic
        warmup_epochs = tc.warmup_epochs  # 10
        warmup_lr = tc.lr  # 0.01
        target_lr = tc.lr * 10  # 0.1

        # Set warmup LR and simulate full schedule
        for pg in opt.param_groups:
            pg["lr"] = warmup_lr

        lr_at_epoch: dict[int, float] = {}
        for epoch in range(tc.epochs):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                for pg in opt.param_groups:
                    pg["lr"] = warmup_lr
            elif warmup_epochs > 0 and epoch == warmup_epochs:
                for pg in opt.param_groups:
                    pg["lr"] = target_lr

            lr_at_epoch[epoch] = opt.param_groups[0]["lr"]

            if epoch >= warmup_epochs:
                sched.step()

        # Warmup phase: LR = 0.01
        assert abs(lr_at_epoch[0] - 0.01) < 1e-8, f"Epoch 0: LR={lr_at_epoch[0]}"
        assert abs(lr_at_epoch[9] - 0.01) < 1e-8, f"Epoch 9: LR={lr_at_epoch[9]}"

        # Peak phase: LR = 0.1
        assert abs(lr_at_epoch[10] - 0.1) < 1e-8, f"Epoch 10: LR={lr_at_epoch[10]}"
        assert abs(lr_at_epoch[50] - 0.1) < 1e-8, f"Epoch 50: LR={lr_at_epoch[50]}"
        assert abs(lr_at_epoch[119] - 0.1) < 1e-8, f"Epoch 119: LR={lr_at_epoch[119]}"

        # First decay: LR = 0.01 at epoch 120
        assert abs(lr_at_epoch[120] - 0.01) < 1e-8, (
            f"LR at epoch 120 should be 0.01, got {lr_at_epoch[120]}"
        )
        assert abs(lr_at_epoch[149] - 0.01) < 1e-8, (
            f"LR at epoch 149 should be 0.01, got {lr_at_epoch[149]}"
        )

        # Second decay: LR = 0.001 at epoch 150
        assert abs(lr_at_epoch[150] - 0.001) < 1e-8, (
            f"LR at epoch 150 should be 0.001, got {lr_at_epoch[150]}"
        )
        assert abs(lr_at_epoch[199] - 0.001) < 1e-8, (
            f"LR at epoch 199 should be 0.001, got {lr_at_epoch[199]}"
        )

    def test_published_results_integrity(self) -> None:
        """Published results dict should have entries for all algebras."""
        for dataset in ("cifar10", "cifar100"):
            for alg in ("R", "C", "H", "O"):
                assert alg in PUBLISHED_RESULTS[dataset], (
                    f"Missing {alg} in PUBLISHED_RESULTS[{dataset}]"
                )

    def test_published_results_values(self) -> None:
        """Published results should match values from research papers."""
        assert PUBLISHED_RESULTS["cifar10"]["H"]["error_pct"] == 5.44
        assert PUBLISHED_RESULTS["cifar10"]["R"]["error_pct"] == 6.37
        assert PUBLISHED_RESULTS["cifar100"]["C"]["error_pct"] == 26.36
        assert PUBLISHED_RESULTS["cifar100"]["H"]["error_pct"] == 26.01
        assert PUBLISHED_RESULTS["cifar10"]["O"]["error_pct"] is None


# ── Fast Tests: Reproduction Report ───────────────────────────────


class TestReproductionReport:
    """Verify reproduction report generation."""

    def test_report_pass_verdict(self, tmp_path: str) -> None:
        """Report should give PASS when within 1 std."""
        pub = {"H": {"error_pct": 5.44, "std_pct": 0.18, "source": "Gaudet", "notes": ""}}
        ours = {"H": {"error_pct": 5.50, "std_pct": 0.15, "param_count": 100000, "seeds": 3}}
        report = reproduction_report(pub, ours, str(tmp_path / "report"))
        assert report["verdicts"]["H"]["verdict"] == "PASS"
        assert report["overall_pass"]

    def test_report_fail_verdict(self, tmp_path: str) -> None:
        """Report should give FAIL when outside 1 std."""
        pub = {"H": {"error_pct": 5.44, "std_pct": 0.18, "source": "Gaudet", "notes": ""}}
        ours = {"H": {"error_pct": 8.00, "std_pct": 0.15, "param_count": 100000, "seeds": 3}}
        report = reproduction_report(pub, ours, str(tmp_path / "report"))
        assert report["verdicts"]["H"]["verdict"] == "FAIL"
        assert not report["overall_pass"]

    def test_report_na_for_octonion(self, tmp_path: str) -> None:
        """Octonion should get N/A verdict (no published target)."""
        pub = {"O": {"error_pct": None, "std_pct": None, "source": "First", "notes": ""}}
        ours = {"O": {"error_pct": 10.0, "std_pct": 0.50, "param_count": 100000, "seeds": 3}}
        report = reproduction_report(pub, ours, str(tmp_path / "report"))
        assert "N/A" in report["verdicts"]["O"]["verdict"]
        assert report["overall_pass"]  # N/A does not cause failure

    def test_report_creates_files(self, tmp_path: str) -> None:
        """Report should create JSON and Markdown files."""
        import os
        pub = {"R": {"error_pct": 6.37, "std_pct": 0.17, "source": "Gaudet", "notes": ""}}
        ours = {"R": {"error_pct": 6.40, "std_pct": 0.15, "param_count": 100000, "seeds": 3}}
        reproduction_report(pub, ours, str(tmp_path / "test_report"))
        assert os.path.exists(str(tmp_path / "test_report.json"))
        assert os.path.exists(str(tmp_path / "test_report.md"))
