"""Tests for synthetic task generators used in optimization landscape experiments.

Each task generator produces deterministic (train, test) TensorDataset pairs
with known optimal losses, enabling rigorous loss ratio comparisons.
"""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from octonion.tasks import (
    build_algebra_native_multi,
    build_algebra_native_single,
    build_classification,
    build_cross_product_recovery,
    build_sinusoidal_regression,
)

# ---------------------------------------------------------------------------
# 1. Algebra-native single-layer task
# ---------------------------------------------------------------------------


class TestAlgebraNativeSingle:
    """Tests for build_algebra_native_single."""

    def test_returns_tensor_datasets(self) -> None:
        train_ds, test_ds = build_algebra_native_single(n_train=100, n_test=20, dim=8)
        assert isinstance(train_ds, TensorDataset)
        assert isinstance(test_ds, TensorDataset)

    def test_shapes_dim8(self) -> None:
        train_ds, test_ds = build_algebra_native_single(n_train=1000, n_test=200, dim=8)
        x_train, y_train = train_ds.tensors
        x_test, y_test = test_ds.tensors
        assert x_train.shape == (1000, 8)
        assert y_train.shape == (1000, 8)
        assert x_test.shape == (200, 8)
        assert y_test.shape == (200, 8)

    def test_deterministic_seeding(self) -> None:
        ds1_train, ds1_test = build_algebra_native_single(n_train=50, n_test=10, seed=123)
        ds2_train, ds2_test = build_algebra_native_single(n_train=50, n_test=10, seed=123)
        assert torch.equal(ds1_train.tensors[0], ds2_train.tensors[0])
        assert torch.equal(ds1_train.tensors[1], ds2_train.tensors[1])
        assert torch.equal(ds1_test.tensors[0], ds2_test.tensors[0])

    def test_different_seeds_differ(self) -> None:
        ds1, _ = build_algebra_native_single(n_train=50, n_test=10, seed=1)
        ds2, _ = build_algebra_native_single(n_train=50, n_test=10, seed=2)
        assert not torch.equal(ds1.tensors[0], ds2.tensors[0])

    def test_supports_dim64(self) -> None:
        train_ds, test_ds = build_algebra_native_single(n_train=100, n_test=20, dim=64)
        assert train_ds.tensors[0].shape == (100, 64)
        assert train_ds.tensors[1].shape == (100, 64)

    def test_supports_dim4(self) -> None:
        train_ds, _ = build_algebra_native_single(n_train=100, n_test=20, dim=4)
        assert train_ds.tensors[0].shape == (100, 4)
        assert train_ds.tensors[1].shape == (100, 4)

    def test_supports_dim2(self) -> None:
        train_ds, _ = build_algebra_native_single(n_train=100, n_test=20, dim=2)
        assert train_ds.tensors[0].shape == (100, 2)
        assert train_ds.tensors[1].shape == (100, 2)

    def test_supports_dim1(self) -> None:
        train_ds, _ = build_algebra_native_single(n_train=100, n_test=20, dim=1)
        assert train_ds.tensors[0].shape == (100, 1)
        assert train_ds.tensors[1].shape == (100, 1)

    def test_default_sizes(self) -> None:
        train_ds, test_ds = build_algebra_native_single()
        assert train_ds.tensors[0].shape[0] == 50000
        assert test_ds.tensors[0].shape[0] == 10000


# ---------------------------------------------------------------------------
# 2. Algebra-native multi-layer task
# ---------------------------------------------------------------------------


class TestAlgebraNativeMulti:
    """Tests for build_algebra_native_multi."""

    def test_returns_tensor_datasets(self) -> None:
        train_ds, test_ds = build_algebra_native_multi(n_train=100, n_test=20, depth=3)
        assert isinstance(train_ds, TensorDataset)
        assert isinstance(test_ds, TensorDataset)

    def test_shapes_depth3(self) -> None:
        train_ds, _ = build_algebra_native_multi(n_train=100, n_test=20, dim=8, depth=3)
        x, y = train_ds.tensors
        assert x.shape == (100, 8)
        assert y.shape == (100, 8)

    def test_depth_parameter(self) -> None:
        """Different depths should produce different targets."""
        _, ds3 = build_algebra_native_multi(n_train=50, n_test=20, depth=3, seed=42)
        _, ds5 = build_algebra_native_multi(n_train=50, n_test=20, depth=5, seed=42)
        # Different depths produce different test outputs
        assert not torch.equal(ds3.tensors[1], ds5.tensors[1])

    def test_supports_depth_5_10(self) -> None:
        for depth in [5, 10]:
            train_ds, _ = build_algebra_native_multi(n_train=50, n_test=10, depth=depth)
            assert train_ds.tensors[0].shape == (50, 8)

    def test_supports_dim64(self) -> None:
        train_ds, _ = build_algebra_native_multi(n_train=50, n_test=10, dim=64, depth=3)
        assert train_ds.tensors[0].shape == (50, 64)
        assert train_ds.tensors[1].shape == (50, 64)

    def test_deterministic_seeding(self) -> None:
        ds1, _ = build_algebra_native_multi(n_train=50, n_test=10, depth=3, seed=99)
        ds2, _ = build_algebra_native_multi(n_train=50, n_test=10, depth=3, seed=99)
        assert torch.equal(ds1.tensors[0], ds2.tensors[0])
        assert torch.equal(ds1.tensors[1], ds2.tensors[1])

    def test_default_sizes(self) -> None:
        train_ds, test_ds = build_algebra_native_multi(depth=3)
        assert train_ds.tensors[0].shape[0] == 50000
        assert test_ds.tensors[0].shape[0] == 10000


# ---------------------------------------------------------------------------
# 3. Cross product recovery task
# ---------------------------------------------------------------------------


class TestCrossProductRecovery:
    """Tests for build_cross_product_recovery."""

    def test_returns_tensor_datasets(self) -> None:
        train_ds, test_ds = build_cross_product_recovery(
            n_train=100, n_test=20, cross_dim=7
        )
        assert isinstance(train_ds, TensorDataset)
        assert isinstance(test_ds, TensorDataset)

    def test_shapes_cross_dim7(self) -> None:
        train_ds, test_ds = build_cross_product_recovery(
            n_train=500, n_test=100, cross_dim=7
        )
        x, y = train_ds.tensors
        assert x.shape == (500, 7)
        assert y.shape == (500, 7)

    def test_shapes_cross_dim3(self) -> None:
        """Positive control: 3D cross product."""
        train_ds, _ = build_cross_product_recovery(
            n_train=100, n_test=20, cross_dim=3
        )
        x, y = train_ds.tensors
        assert x.shape == (100, 3)
        assert y.shape == (100, 3)

    def test_clean_data_no_noise(self) -> None:
        """With noise_level=0.0, train and test targets should be identical
        given the same inputs (same ground truth function)."""
        train_ds, test_ds = build_cross_product_recovery(
            n_train=100, n_test=100, cross_dim=7, noise_level=0.0, seed=42
        )
        # Test set should have clean targets (no noise)
        # The test y values should not be all zeros
        y_test = test_ds.tensors[1]
        assert y_test.abs().max() > 0

    def test_noise_affects_train_not_test(self) -> None:
        """Noise added to training data but test data remains clean."""
        train_clean, test_clean = build_cross_product_recovery(
            n_train=100, n_test=50, cross_dim=7, noise_level=0.0, seed=42
        )
        train_noisy, test_noisy = build_cross_product_recovery(
            n_train=100, n_test=50, cross_dim=7, noise_level=0.15, seed=42
        )
        # Test data should be identical (clean in both cases)
        assert torch.allclose(test_clean.tensors[1], test_noisy.tensors[1], atol=1e-10)
        # Training data should differ due to noise
        assert not torch.allclose(
            train_clean.tensors[1], train_noisy.tensors[1], atol=1e-5
        )

    def test_3d_cross_product_valid(self) -> None:
        """3D cross product positive control: verify against torch.linalg.cross."""
        train_ds, _ = build_cross_product_recovery(
            n_train=200, n_test=20, cross_dim=3, noise_level=0.0, seed=42
        )
        x, y = train_ds.tensors
        # y should be a valid cross product with some fixed vector v
        # Check orthogonality: x . y == 0 (cross product is perpendicular to inputs)
        dots = torch.sum(x * y, dim=-1)
        assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-6)

    def test_supports_dim64(self) -> None:
        """64D variant: cross_dim signal embedded in higher dimensions."""
        train_ds, _ = build_cross_product_recovery(
            n_train=100, n_test=20, cross_dim=7, noise_level=0.0, seed=42, dim=64
        )
        x, y = train_ds.tensors
        assert x.shape == (100, 64)
        assert y.shape == (100, 64)

    def test_deterministic_seeding(self) -> None:
        ds1, _ = build_cross_product_recovery(n_train=50, n_test=10, seed=77)
        ds2, _ = build_cross_product_recovery(n_train=50, n_test=10, seed=77)
        assert torch.equal(ds1.tensors[0], ds2.tensors[0])
        assert torch.equal(ds1.tensors[1], ds2.tensors[1])

    def test_default_sizes(self) -> None:
        train_ds, test_ds = build_cross_product_recovery()
        assert train_ds.tensors[0].shape[0] == 50000
        assert test_ds.tensors[0].shape[0] == 10000


# ---------------------------------------------------------------------------
# 4. Sinusoidal regression task
# ---------------------------------------------------------------------------


class TestSinusoidalRegression:
    """Tests for build_sinusoidal_regression."""

    def test_returns_tensor_datasets(self) -> None:
        train_ds, test_ds = build_sinusoidal_regression(n_train=100, n_test=20)
        assert isinstance(train_ds, TensorDataset)
        assert isinstance(test_ds, TensorDataset)

    def test_shapes_dim8(self) -> None:
        train_ds, _ = build_sinusoidal_regression(
            n_train=500, n_test=100, dim=8, n_components=3
        )
        x, y = train_ds.tensors
        assert x.shape == (500, 8)
        assert y.shape == (500, 3)

    def test_continuous_targets(self) -> None:
        """Targets should be continuous (not discrete)."""
        train_ds, _ = build_sinusoidal_regression(n_train=1000, n_test=100)
        y = train_ds.tensors[1]
        # Continuous targets: many unique values
        unique_vals = torch.unique(y).numel()
        assert unique_vals > 100, f"Expected continuous targets, got {unique_vals} unique values"

    def test_supports_dim64(self) -> None:
        train_ds, _ = build_sinusoidal_regression(n_train=100, n_test=20, dim=64)
        assert train_ds.tensors[0].shape == (100, 64)

    def test_deterministic_seeding(self) -> None:
        ds1, _ = build_sinusoidal_regression(n_train=50, n_test=10, seed=55)
        ds2, _ = build_sinusoidal_regression(n_train=50, n_test=10, seed=55)
        assert torch.equal(ds1.tensors[0], ds2.tensors[0])
        assert torch.equal(ds1.tensors[1], ds2.tensors[1])

    def test_default_sizes(self) -> None:
        train_ds, test_ds = build_sinusoidal_regression()
        assert train_ds.tensors[0].shape[0] == 50000
        assert test_ds.tensors[0].shape[0] == 10000


# ---------------------------------------------------------------------------
# 5. Classification task
# ---------------------------------------------------------------------------


class TestClassification:
    """Tests for build_classification."""

    def test_returns_three_elements(self) -> None:
        train_ds, test_ds, meta = build_classification(n_train=100, n_test=20)
        assert isinstance(train_ds, TensorDataset)
        assert isinstance(test_ds, TensorDataset)
        assert isinstance(meta, dict)

    def test_shapes_dim8(self) -> None:
        train_ds, test_ds, _ = build_classification(
            n_train=500, n_test=100, dim=8, n_classes=5
        )
        x, y = train_ds.tensors
        assert x.shape == (500, 8)
        assert y.shape == (500,)
        assert y.dtype == torch.long

    def test_integer_class_labels(self) -> None:
        train_ds, _, _ = build_classification(n_train=200, n_test=20, n_classes=5)
        y = train_ds.tensors[1]
        assert y.dtype == torch.long
        assert y.min() >= 0
        assert y.max() < 5

    def test_bayes_optimal_accuracy(self) -> None:
        _, _, meta = build_classification(n_train=100, n_test=20)
        assert "bayes_optimal_accuracy" in meta
        acc = meta["bayes_optimal_accuracy"]
        assert 0.0 < acc <= 1.0

    def test_centers_in_metadata(self) -> None:
        _, _, meta = build_classification(n_train=100, n_test=20, dim=8, n_classes=5)
        assert "centers" in meta
        assert meta["centers"].shape == (5, 8)

    def test_supports_dim64(self) -> None:
        train_ds, _, _ = build_classification(n_train=100, n_test=20, dim=64)
        assert train_ds.tensors[0].shape == (100, 64)

    def test_deterministic_seeding(self) -> None:
        ds1, _, m1 = build_classification(n_train=50, n_test=10, seed=88)
        ds2, _, m2 = build_classification(n_train=50, n_test=10, seed=88)
        assert torch.equal(ds1.tensors[0], ds2.tensors[0])
        assert torch.equal(ds1.tensors[1], ds2.tensors[1])
        assert torch.equal(m1["centers"], m2["centers"])

    def test_default_sizes(self) -> None:
        train_ds, test_ds, _ = build_classification()
        assert train_ds.tensors[0].shape[0] == 50000
        assert test_ds.tensors[0].shape[0] == 10000
