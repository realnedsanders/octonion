"""Classification task for optimization landscape experiments.

Task 5: Gaussian cluster classification with known Bayes-optimal accuracy.
  - n_classes clusters with centers on a sphere of radius `separation`.
  - Unit Gaussian noise added to each sample.
  - Bayes-optimal accuracy computed analytically from geometry.
"""

from __future__ import annotations

import math

import torch
from torch.utils.data import TensorDataset


def _estimate_bayes_optimal_accuracy(
    centers: torch.Tensor,
    n_classes: int,
    dim: int,
) -> float:
    """Estimate Bayes-optimal accuracy for Gaussian clusters.

    For equal-prior Gaussian clusters with identity covariance,
    the Bayes-optimal classifier assigns each point to the nearest center.
    The error probability depends on the pairwise distances between centers
    and the dimensionality.

    Uses the union bound approximation:
      P(error) <= (n_classes - 1) * Phi(-d_min / 2)
    where d_min is the minimum inter-center distance and Phi is the
    standard normal CDF.

    Args:
        centers: Cluster centers of shape [n_classes, dim].
        n_classes: Number of classes.
        dim: Input dimension.

    Returns:
        Estimated Bayes-optimal accuracy in [0, 1].
    """
    # Compute pairwise distances between centers
    dists = torch.cdist(centers, centers)
    # Mask diagonal (self-distances = 0)
    mask = ~torch.eye(n_classes, dtype=torch.bool)
    min_dist = dists[mask].min().item()

    # Union bound: P(error for one pair) = Phi(-d/2)
    # where d is the distance between two centers and noise is N(0, I)
    # The decision boundary is the perpendicular bisector
    # P(x falls on wrong side) = Phi(-d/2) for unit variance Gaussian
    z = min_dist / 2.0
    # Standard normal CDF approximation: Phi(z) = 0.5 * erfc(-z / sqrt(2))
    p_error_pair = 0.5 * math.erfc(z / math.sqrt(2.0))

    # Union bound over all other classes
    p_error = min((n_classes - 1) * p_error_pair, 1.0)
    accuracy = max(1.0 - p_error, 1.0 / n_classes)  # At least chance level
    return accuracy


def build_classification(
    n_train: int = 50_000,
    n_test: int = 10_000,
    dim: int = 8,
    n_classes: int = 5,
    separation: float = 3.0,
    seed: int = 42,
) -> tuple[TensorDataset, TensorDataset, dict]:
    """Build a Gaussian cluster classification task.

    Generates n_classes clusters with centers uniformly distributed on a
    sphere of radius `separation`. Samples are drawn from unit-variance
    Gaussian distributions centered at each cluster center.

    Args:
        n_train: Number of training samples.
        n_test: Number of test samples.
        dim: Input dimension.
        n_classes: Number of classes.
        separation: Radius of the sphere on which cluster centers lie.
        seed: Random seed for deterministic generation.

    Returns:
        (train_dataset, test_dataset, metadata) tuple.
        metadata contains:
            - "bayes_optimal_accuracy": float
            - "centers": Tensor of shape [n_classes, dim]
    """
    g = torch.Generator().manual_seed(seed)
    dtype = torch.float64

    # Generate cluster centers uniformly on sphere of radius `separation`
    raw_centers = torch.randn(n_classes, dim, generator=g, dtype=dtype)
    centers = raw_centers / raw_centers.norm(dim=-1, keepdim=True) * separation

    # Assign samples to classes uniformly
    labels_train = torch.randint(0, n_classes, (n_train,), generator=g)
    labels_test = torch.randint(0, n_classes, (n_test,), generator=g)

    # Generate samples: center + unit Gaussian noise
    x_train = centers[labels_train] + torch.randn(n_train, dim, generator=g, dtype=dtype)
    x_test = centers[labels_test] + torch.randn(n_test, dim, generator=g, dtype=dtype)

    # Compute Bayes-optimal accuracy
    bayes_acc = _estimate_bayes_optimal_accuracy(centers, n_classes, dim)

    metadata = {
        "bayes_optimal_accuracy": bayes_acc,
        "centers": centers.float(),
    }

    return (
        TensorDataset(x_train.float(), labels_train.long()),
        TensorDataset(x_test.float(), labels_test.long()),
        metadata,
    )
