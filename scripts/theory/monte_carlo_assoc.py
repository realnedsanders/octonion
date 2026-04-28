"""Monte Carlo validation of associator norm distributions on the unit 7-sphere.

Validates Egan's analytical result for the mean associator norm of random unit
octonions, characterizes within-class vs between-class associator norm distributions,
computes subalgebra proximity bounds, and measures Fano plane angular separations.

Reference: Greg Egan, "Peeling the Octonions" -- E[||[a,b,c]||] = 147456/(42875*pi)

Usage:
    python scripts/theory/monte_carlo_assoc.py \
        --features-dir results/T2/features \
        --output-dir results/T2/theory \
        --n-samples 100000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Ensure project root is on path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from octonion._fano import FANO_PLANE
from octonion._multiplication import octonion_mul

# Egan's exact analytical result for E[||[a,b,c]||] on S^7
EGAN_THEORETICAL_MEAN = 147456 / (42875 * math.pi)


def _sample_unit_octonions(n: int, seed: int = 42) -> torch.Tensor:
    """Sample n uniform random unit octonions on S^7.

    Uses the standard method: sample from isotropic Gaussian in R^8,
    then normalize to unit norm. This produces uniform distribution on S^7.

    Args:
        n: Number of samples.
        seed: Random seed for reproducibility.

    Returns:
        Tensor of shape [n, 8] with unit norm rows.
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 8, dtype=torch.float64, generator=gen)
    norms = torch.norm(x, dim=-1, keepdim=True)
    return x / norms


def sample_random_associator_norms(
    n_samples: int = 100000, seed: int = 42
) -> torch.Tensor:
    """Sample associator norms ||[a,b,c]|| for random unit octonion triples.

    Validates against Egan's analytical result:
        E[||[a,b,c]||] = 147456/(42875*pi) ~ 1.0947

    Args:
        n_samples: Number of triples to sample.
        seed: Random seed.

    Returns:
        Tensor of shape [n_samples] with associator norms.
    """
    # Sample 3 independent sets of unit octonions
    a = _sample_unit_octonions(n_samples, seed=seed)
    b = _sample_unit_octonions(n_samples, seed=seed + 1)
    c = _sample_unit_octonions(n_samples, seed=seed + 2)

    # Compute associators: [a,b,c] = (a*b)*c - a*(b*c)
    ab = octonion_mul(a, b)
    ab_c = octonion_mul(ab, c)
    bc = octonion_mul(b, c)
    a_bc = octonion_mul(a, bc)

    assoc = ab_c - a_bc  # shape [n_samples, 8]
    norms = torch.norm(assoc, dim=-1)  # shape [n_samples]
    return norms


def sample_within_class_norms(
    features_path: str, n_samples: int = 50000, seed: int = 42
) -> dict[int, torch.Tensor]:
    """Sample associator norms for triples within the same class.

    Args:
        features_path: Path to directory containing cached .pt feature files.
            Expected format: {features_path}/features.pt and {features_path}/labels.pt
        n_samples: Total number of triples to sample (distributed across classes).
        seed: Random seed.

    Returns:
        Dict mapping class_id -> Tensor of associator norms for that class.
    """
    features = torch.load(
        os.path.join(features_path, "features.pt"), weights_only=True
    ).to(torch.float64)
    labels = torch.load(
        os.path.join(features_path, "labels.pt"), weights_only=True
    )

    # Normalize to unit octonions
    norms = torch.norm(features, dim=-1, keepdim=True)
    features = features / norms.clamp(min=1e-15)

    classes = torch.unique(labels)
    n_per_class = max(1, n_samples // len(classes))
    gen = torch.Generator().manual_seed(seed)

    result: dict[int, torch.Tensor] = {}
    for cls in classes:
        cls_id = cls.item()
        mask = labels == cls
        cls_features = features[mask]
        n_cls = cls_features.shape[0]

        if n_cls < 3:
            result[cls_id] = torch.tensor([], dtype=torch.float64)
            continue

        # Sample random triples within this class
        idx = torch.randint(0, n_cls, (n_per_class, 3), generator=gen)
        a = cls_features[idx[:, 0]]
        b = cls_features[idx[:, 1]]
        c = cls_features[idx[:, 2]]

        ab = octonion_mul(a, b)
        ab_c = octonion_mul(ab, c)
        bc = octonion_mul(b, c)
        a_bc = octonion_mul(a, bc)

        assoc = ab_c - a_bc
        assoc_norms = torch.norm(assoc, dim=-1)
        result[cls_id] = assoc_norms

    return result


def sample_between_class_norms(
    features_path: str, n_samples: int = 50000, seed: int = 42
) -> torch.Tensor:
    """Sample associator norms for triples from different classes.

    Each triple has elements from 3 different classes.

    Args:
        features_path: Path to directory containing cached feature files.
        n_samples: Number of triples to sample.
        seed: Random seed.

    Returns:
        Tensor of associator norms for between-class triples.
    """
    features = torch.load(
        os.path.join(features_path, "features.pt"), weights_only=True
    ).to(torch.float64)
    labels = torch.load(
        os.path.join(features_path, "labels.pt"), weights_only=True
    )

    # Normalize to unit octonions
    norms = torch.norm(features, dim=-1, keepdim=True)
    features = features / norms.clamp(min=1e-15)

    classes = torch.unique(labels).tolist()
    n_classes = len(classes)

    if n_classes < 3:
        raise ValueError(
            f"Need at least 3 classes for between-class sampling, got {n_classes}"
        )

    gen = torch.Generator().manual_seed(seed)

    # Group features by class
    cls_features = {
        cls: features[labels == cls] for cls in classes
    }

    all_norms = []
    for _ in range(n_samples):
        # Pick 3 different classes
        cls_idx = torch.randperm(n_classes, generator=gen)[:3]
        c1, c2, c3 = classes[cls_idx[0].item()], classes[cls_idx[1].item()], classes[cls_idx[2].item()]

        # Pick one random sample from each class
        f1 = cls_features[c1]
        f2 = cls_features[c2]
        f3 = cls_features[c3]

        i1 = torch.randint(0, len(f1), (1,), generator=gen).item()
        i2 = torch.randint(0, len(f2), (1,), generator=gen).item()
        i3 = torch.randint(0, len(f3), (1,), generator=gen).item()

        a = f1[i1].unsqueeze(0)
        b = f2[i2].unsqueeze(0)
        c = f3[i3].unsqueeze(0)

        ab = octonion_mul(a, b)
        ab_c = octonion_mul(ab, c)
        bc = octonion_mul(b, c)
        a_bc = octonion_mul(a, bc)

        assoc = ab_c - a_bc
        norm = torch.norm(assoc, dim=-1)
        all_norms.append(norm.item())

    return torch.tensor(all_norms, dtype=torch.float64)


def sample_subalgebra_proximity_norms(
    n_samples: int = 50000, seed: int = 42
) -> dict[str, dict[float, float]]:
    """Sample associator norms near quaternionic subalgebras.

    For each of the 7 Fano plane subalgebras, sample unit octonions within
    angular distance epsilon of the subalgebra, compute associator norms,
    and verify the O(epsilon^2) bound.

    Args:
        n_samples: Number of triples per epsilon value per subalgebra.
        seed: Random seed.

    Returns:
        Dict with:
            - "per_subalgebra": {subalgebra_idx: {epsilon: max_norm}}
            - "aggregated": {epsilon: max_norm} (max over all subalgebras)
    """
    epsilons = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    gen = torch.Generator().manual_seed(seed)

    per_subalgebra: dict[int, dict[float, float]] = {}
    aggregated: dict[float, float] = {eps: 0.0 for eps in epsilons}

    for sub_idx, (i, j, k) in enumerate(FANO_PLANE.triples):
        per_subalgebra[sub_idx] = {}

        # The quaternionic subalgebra spans {e0, e_i, e_j, e_k}
        # in R^8 basis: indices [0, i, j, k]
        subalg_dims = [0, i, j, k]
        other_dims = [d for d in range(8) if d not in subalg_dims]

        for eps in epsilons:
            # Sample unit octonions near this subalgebra:
            # Large component in subalgebra dims, small (O(eps)) in others
            n_per = max(100, n_samples // (len(epsilons) * 7))

            # Generate base quaternionic part (on the subalgebra)
            base = torch.randn(n_per * 3, 4, dtype=torch.float64, generator=gen)
            base = base / torch.norm(base, dim=-1, keepdim=True)

            # Generate perturbation in orthogonal directions
            perturb = torch.randn(
                n_per * 3, 4, dtype=torch.float64, generator=gen
            ) * eps

            # Construct full 8D octonion
            full = torch.zeros(n_per * 3, 8, dtype=torch.float64)
            for col_idx, dim in enumerate(subalg_dims):
                full[:, dim] = base[:, col_idx]
            for col_idx, dim in enumerate(other_dims):
                full[:, dim] = perturb[:, col_idx]

            # Normalize to S^7
            full = full / torch.norm(full, dim=-1, keepdim=True)

            # Split into triples
            a = full[:n_per]
            b = full[n_per:2*n_per]
            c = full[2*n_per:3*n_per]

            # Compute associator norms
            ab = octonion_mul(a, b)
            ab_c = octonion_mul(ab, c)
            bc = octonion_mul(b, c)
            a_bc = octonion_mul(a, bc)

            assoc = ab_c - a_bc
            norms = torch.norm(assoc, dim=-1)

            max_norm = norms.max().item()
            norms.mean().item()

            per_subalgebra[sub_idx][eps] = max_norm
            aggregated[eps] = max(aggregated[eps], max_norm)

    return {
        "per_subalgebra": {str(k): v for k, v in per_subalgebra.items()},
        "aggregated": aggregated,
    }


def compute_fano_angular_separations() -> torch.Tensor:
    """Compute pairwise angular separations between Fano plane subalgebras.

    Each quaternionic subalgebra occupies a 3D subspace of the 7D imaginary
    octonion space (R^7, basis {e1,...,e7}). The angular separation between
    two subalgebras is the minimum principal angle between their 3-planes.

    Returns:
        Tensor of shape [7, 7] with pairwise angular separations in radians.
    """
    # Build projection matrices for each subalgebra in R^7 (imaginary part)
    # Each subalgebra (i, j, k) spans e_i, e_j, e_k in R^7
    # In 0-indexed R^7: e_i is at index i-1
    projections = []
    for i, j, k in FANO_PLANE.triples:
        # Basis vectors for this subalgebra in R^7
        basis = torch.zeros(3, 7, dtype=torch.float64)
        basis[0, i - 1] = 1.0
        basis[1, j - 1] = 1.0
        basis[2, k - 1] = 1.0
        projections.append(basis)

    # Compute pairwise principal angles
    separations = torch.zeros(7, 7, dtype=torch.float64)

    for s1 in range(7):
        for s2 in range(7):
            if s1 == s2:
                separations[s1, s2] = 0.0
                continue

            # Compute principal angles between subspaces via SVD
            # M = P1^T @ P2 where P1, P2 are orthonormal bases
            P1 = projections[s1]  # [3, 7]
            P2 = projections[s2]  # [3, 7]

            M = P1 @ P2.T  # [3, 3]
            svd_vals = torch.linalg.svdvals(M)

            # Clamp to [-1, 1] for numerical safety
            svd_vals = torch.clamp(svd_vals, -1.0, 1.0)

            # Principal angles are arccos of singular values
            angles = torch.acos(svd_vals)

            # Minimum principal angle = smallest angle = arccos(largest sv)
            min_angle = angles.min().item()
            separations[s1, s2] = min_angle

    return separations


def fit_distribution(norms: torch.Tensor) -> dict[str, Any]:
    """Fit candidate distributions to associator norm samples.

    Tests: Rayleigh, half-normal, gamma, beta distributions.
    Reports best fit by Kolmogorov-Smirnov test p-value.

    Args:
        norms: 1D tensor of associator norm samples.

    Returns:
        Dict with fit results for each distribution and the best fit.
    """
    from scipy import stats

    data = norms.numpy()

    # Remove any zeros or negatives (shouldn't happen but be safe)
    data = data[data > 0]

    results: dict[str, Any] = {}

    # Rayleigh distribution
    try:
        ray_params = stats.rayleigh.fit(data)
        ks_stat, ks_pval = stats.kstest(data, "rayleigh", args=ray_params)
        results["rayleigh"] = {
            "params": {"loc": float(ray_params[0]), "scale": float(ray_params[1])},
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
        }
    except Exception as e:
        results["rayleigh"] = {"error": str(e)}

    # Half-normal distribution
    try:
        hn_params = stats.halfnorm.fit(data)
        ks_stat, ks_pval = stats.kstest(data, "halfnorm", args=hn_params)
        results["halfnorm"] = {
            "params": {"loc": float(hn_params[0]), "scale": float(hn_params[1])},
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
        }
    except Exception as e:
        results["halfnorm"] = {"error": str(e)}

    # Gamma distribution
    try:
        gamma_params = stats.gamma.fit(data)
        ks_stat, ks_pval = stats.kstest(data, "gamma", args=gamma_params)
        results["gamma"] = {
            "params": {
                "a": float(gamma_params[0]),
                "loc": float(gamma_params[1]),
                "scale": float(gamma_params[2]),
            },
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
        }
    except Exception as e:
        results["gamma"] = {"error": str(e)}

    # Beta distribution (need to scale data to [0,1] using max=2 bound)
    try:
        scaled = data / 2.0  # Map [0, 2] -> [0, 1]
        scaled = np.clip(scaled, 1e-10, 1.0 - 1e-10)
        beta_params = stats.beta.fit(scaled)
        ks_stat, ks_pval = stats.kstest(scaled, "beta", args=beta_params)
        results["beta"] = {
            "params": {
                "a": float(beta_params[0]),
                "b": float(beta_params[1]),
                "loc": float(beta_params[2]),
                "scale": float(beta_params[3]),
            },
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "note": "Fitted on data/2 (scaled to [0,1])",
        }
    except Exception as e:
        results["beta"] = {"error": str(e)}

    # Find best fit by highest p-value
    best_name = None
    best_pval = -1.0
    for name, res in results.items():
        if "ks_pvalue" in res and res["ks_pvalue"] > best_pval:
            best_pval = res["ks_pvalue"]
            best_name = name

    results["best_fit"] = {
        "distribution": best_name,
        "ks_pvalue": best_pval,
    }

    return results


def _plot_random_distribution(
    norms: torch.Tensor, output_dir: str
) -> None:
    """Plot histogram of random associator norms with Egan's mean marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = norms.numpy()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=100, density=True, alpha=0.7, color="steelblue",
            label="Monte Carlo samples")
    ax.axvline(EGAN_THEORETICAL_MEAN, color="red", linestyle="--", linewidth=2,
               label=f"Egan theoretical mean = {EGAN_THEORETICAL_MEAN:.4f}")
    ax.axvline(data.mean(), color="orange", linestyle="-", linewidth=2,
               label=f"MC sample mean = {data.mean():.4f}")
    ax.set_xlabel("Associator norm ||[a,b,c]||", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Distribution of Associator Norms for Random Unit Octonions on S^7",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "random_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_within_vs_between(
    within: dict[int, torch.Tensor],
    between: torch.Tensor,
    output_dir: str,
) -> None:
    """Plot overlapping histograms of within-class and between-class norms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Aggregate all within-class norms
    all_within = torch.cat([v for v in within.values() if len(v) > 0])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_within.numpy(), bins=80, density=True, alpha=0.6,
            color="green", label=f"Within-class (n={len(all_within)})")
    ax.hist(between.numpy(), bins=80, density=True, alpha=0.6,
            color="red", label=f"Between-class (n={len(between)})")

    ax.axvline(all_within.mean().item(), color="darkgreen", linestyle="--",
               linewidth=2, label=f"Within mean = {all_within.mean():.4f}")
    ax.axvline(between.mean().item(), color="darkred", linestyle="--",
               linewidth=2, label=f"Between mean = {between.mean():.4f}")

    ax.set_xlabel("Associator norm ||[a,b,c]||", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Within-Class vs Between-Class Associator Norms", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "within_vs_between.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_subalgebra_bound(
    aggregated: dict[float, float], output_dir: str
) -> None:
    """Log-log plot of epsilon vs max associator norm (O(eps^2) scaling)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epsilons = sorted(aggregated.keys())
    max_norms = [aggregated[eps] for eps in epsilons]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(epsilons, max_norms, "o-", color="steelblue", linewidth=2,
              markersize=8, label="Max ||[a,b,c]|| (measured)")

    # Reference O(eps^2) line
    eps_arr = np.array(epsilons)
    # Scale reference line to match at smallest epsilon
    scale = max_norms[0] / (epsilons[0] ** 2) if epsilons[0] > 0 else 1.0
    ref_line = scale * eps_arr ** 2
    ax.loglog(epsilons, ref_line, "--", color="red", linewidth=1.5,
              label=r"$O(\epsilon^2)$ reference")

    ax.set_xlabel(r"Angular distance $\epsilon$ from subalgebra", fontsize=12)
    ax.set_ylabel("Max associator norm", fontsize=12)
    ax.set_title("Subalgebra Proximity Bound: Associator Norm vs Angular Distance",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    path = os.path.join(output_dir, "subalgebra_bound.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_fano_separations(
    separations: torch.Tensor, output_dir: str
) -> None:
    """Heatmap of pairwise subalgebra angular separations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    data = separations.numpy()

    # Convert to degrees for readability
    data_deg = np.degrees(data)

    im = ax.imshow(data_deg, cmap="YlOrRd", aspect="equal")
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))

    # Label with Fano plane triples
    labels = [f"S{i}: ({t[0]},{t[1]},{t[2]})"
              for i, t in enumerate(FANO_PLANE.triples)]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Add value annotations
    for i in range(7):
        for j in range(7):
            val = data_deg[i, j]
            color = "white" if val > data_deg.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    color=color, fontsize=8)

    plt.colorbar(im, ax=ax, label="Angular separation (degrees)")
    ax.set_title("Pairwise Angular Separations Between\nFano Plane Quaternionic Subalgebras",
                 fontsize=13)

    path = os.path.join(output_dir, "fano_separations.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def run_analysis(
    features_dir: str | None = None,
    output_dir: str = "results/T2/theory",
    n_samples: int = 100000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run full Monte Carlo analysis and save results.

    Args:
        features_dir: Path to cached features (for within/between class analysis).
            If None, only random and subalgebra analyses are performed.
        output_dir: Output directory for JSON and plots.
        n_samples: Number of samples for Monte Carlo estimation.
        seed: Random seed.

    Returns:
        Dict with all analysis results.
    """
    os.makedirs(output_dir, exist_ok=True)

    results: dict[str, Any] = {}

    # --- 1. Random associator norms ---
    print(f"Sampling {n_samples} random associator norms...")
    random_norms = sample_random_associator_norms(n_samples=n_samples, seed=seed)
    results["random_mean"] = float(random_norms.mean().item())
    results["random_std"] = float(random_norms.std().item())
    results["random_median"] = float(random_norms.median().item())
    results["random_min"] = float(random_norms.min().item())
    results["random_max"] = float(random_norms.max().item())
    results["egan_theoretical_mean"] = float(EGAN_THEORETICAL_MEAN)
    results["egan_deviation_pct"] = float(
        abs(results["random_mean"] - EGAN_THEORETICAL_MEAN)
        / EGAN_THEORETICAL_MEAN * 100
    )

    print(f"  MC mean = {results['random_mean']:.6f}")
    print(f"  Egan    = {results['egan_theoretical_mean']:.6f}")
    print(f"  Deviation = {results['egan_deviation_pct']:.3f}%")

    _plot_random_distribution(random_norms, output_dir)

    # --- 2. Distribution fitting ---
    print("Fitting distributions...")
    dist_fit = fit_distribution(random_norms)
    results["distribution_fit"] = dist_fit
    print(f"  Best fit: {dist_fit['best_fit']['distribution']} "
          f"(KS p-value = {dist_fit['best_fit']['ks_pvalue']:.6f})")

    # --- 3. Within-class vs between-class (if features available) ---
    if features_dir and os.path.isdir(features_dir):
        print("Analyzing within-class vs between-class norms...")
        within = sample_within_class_norms(features_dir, n_samples=n_samples // 2, seed=seed)
        between = sample_between_class_norms(features_dir, n_samples=n_samples // 2, seed=seed)

        results["within_class_mean_per_class"] = {
            str(k): float(v.mean().item()) for k, v in within.items() if len(v) > 0
        }
        results["within_class_std_per_class"] = {
            str(k): float(v.std().item()) for k, v in within.items() if len(v) > 0
        }
        results["between_class_mean"] = float(between.mean().item())
        results["between_class_std"] = float(between.std().item())

        _plot_within_vs_between(within, between, output_dir)
    else:
        print("  No features directory provided; skipping within/between analysis.")
        results["within_class_mean_per_class"] = None
        results["between_class_mean"] = None
        results["between_class_std"] = None

    # --- 4. Subalgebra proximity bounds ---
    print("Computing subalgebra proximity bounds...")
    proximity = sample_subalgebra_proximity_norms(n_samples=n_samples, seed=seed)
    results["subalgebra_proximity_bounds"] = proximity["aggregated"]
    results["subalgebra_proximity_per_sub"] = proximity["per_subalgebra"]

    _plot_subalgebra_bound(proximity["aggregated"], output_dir)

    # --- 5. Fano angular separations ---
    print("Computing Fano plane angular separations...")
    separations = compute_fano_angular_separations()
    results["fano_angular_separations"] = separations.tolist()
    results["fano_angular_separations_degrees"] = torch.rad2deg(separations).tolist()

    _plot_fano_separations(separations, output_dir)

    # --- Save results ---
    json_path = os.path.join(output_dir, "monte_carlo_results.json")

    # Convert any remaining non-serializable values
    def _make_serializable(obj: Any) -> Any:
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serializable(x) for x in obj]
        if obj is None or isinstance(obj, (int, float, str, bool)):
            return obj
        return str(obj)

    serializable = _make_serializable(results)
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved results: {json_path}")

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo validation of associator norm distributions"
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default=None,
        help="Directory with cached .pt features (features.pt + labels.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/T2/theory",
        help="Output directory for JSON and plots",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100000,
        help="Number of Monte Carlo samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    run_analysis(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
