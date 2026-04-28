"""Tests validating mathematical claims from the thesis (oct-trie.tex).

Covers:
- Egan's mean associator norm theorem (Section 9, lines 714-721)
- Associator norm range [0, 2] for unit octonions (line 276/711)
- G2 automorphism invariance of associator norms (lines 842-861)
"""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings

from octonion import Octonion
from octonion._fano import FANO_PLANE
from octonion._octonion import associator
from tests.conftest import ATOL_FLOAT64, unit_octonions

# ── Egan's Mean Associator Norm ──────────────────────────────────────


class TestEganMeanAssociatorNorm:
    """Validate E[||[a,b,c]||] = 147456/(42875*pi) for uniform S^7."""

    THEORETICAL_VALUE = 147456.0 / (42875.0 * math.pi)  # ≈ 1.0947

    def test_monte_carlo(self) -> None:
        """Monte Carlo estimate of mean associator norm on S^7.

        Samples 100K uniform unit octonion triples (Gaussian normalization
        for uniformity on S^7) and checks that the theoretical value falls
        within the 95% confidence interval of the sample mean.
        """
        gen = torch.Generator().manual_seed(314159)
        n_samples = 100_000
        norms = []

        for _ in range(n_samples):
            # Uniform on S^7 via Gaussian normalization
            a = torch.randn(8, dtype=torch.float64, generator=gen)
            a = a / a.norm()
            b = torch.randn(8, dtype=torch.float64, generator=gen)
            b = b / b.norm()
            c = torch.randn(8, dtype=torch.float64, generator=gen)
            c = c / c.norm()

            assoc = associator(Octonion(a), Octonion(b), Octonion(c))
            norms.append(assoc._data.norm().item())

        norms_t = torch.tensor(norms, dtype=torch.float64)
        assert torch.isfinite(norms_t).all(), (
            f"Non-finite associator norms detected in {(~torch.isfinite(norms_t)).sum()} "
            f"of {n_samples} samples — possible NaN/Inf in octonion multiplication"
        )
        mean_norm = norms_t.mean().item()
        std_norm = norms_t.std().item()
        se = std_norm / math.sqrt(n_samples)

        # 95% CI: mean ± 1.96 * SE
        ci_lower = mean_norm - 1.96 * se
        ci_upper = mean_norm + 1.96 * se

        assert ci_lower <= self.THEORETICAL_VALUE <= ci_upper, (
            f"Theoretical value {self.THEORETICAL_VALUE:.6f} outside 95% CI "
            f"[{ci_lower:.6f}, {ci_upper:.6f}] "
            f"(sample mean={mean_norm:.6f}, SE={se:.6f})"
        )


# ── Associator Norm Range ────────────────────────────────────────────


class TestAssociatorNormRange:
    """Validate ||[a,b,c]|| in [0, 2] for unit octonions."""

    @given(a=unit_octonions(), b=unit_octonions(), c=unit_octonions())
    @settings(max_examples=10000, deadline=None)
    def test_unit_octonion_bound(self, a: Octonion, b: Octonion, c: Octonion) -> None:
        """Associator norm must be in [0, 2] for unit octonions."""
        assoc = associator(a, b, c)
        norm = assoc._data.norm().item()
        assert 0.0 <= norm <= 2.0 + ATOL_FLOAT64, (
            f"Associator norm {norm} outside [0, 2]"
        )

    def test_upper_bound_approachable(self) -> None:
        """The upper bound of 2 should be approachable (max observed > 1.5).

        Uses basis elements where the associator achieves its maximum:
        [e1, e2, e3] = -2*e6, which has norm 2.
        """
        e1 = Octonion(torch.tensor([0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float64))
        e2 = Octonion(torch.tensor([0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float64))
        e3 = Octonion(torch.tensor([0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float64))

        assoc = associator(e1, e2, e3)
        norm = assoc._data.norm().item()
        assert norm > 1.5, f"Expected norm > 1.5, got {norm}"
        assert abs(norm - 2.0) < ATOL_FLOAT64, f"Expected norm = 2.0, got {norm}"


# ── G2 Automorphism Invariance ───────────────────────────────────────


def _apply_fano_automorphism(x: Octonion, perm: dict[int, int]) -> Octonion:
    """Apply a Fano plane automorphism to an octonion.

    The automorphism permutes imaginary basis elements according to
    the permutation dict {old_index -> new_index} (1-indexed imaginary units).
    The real part (index 0) is unchanged.
    """
    data = x._data.clone()
    result = torch.zeros_like(data)
    result[0] = data[0]  # real part unchanged
    for old_idx, new_idx in perm.items():
        result[new_idx] = data[old_idx]
    return Octonion(result)


def _compose_perm(p1: dict[int, int], p2: dict[int, int]) -> dict[int, int]:
    """Compose two permutations: (p1 ∘ p2)(x) = p1(p2(x))."""
    return {k: p1[v] for k, v in p2.items()}


class TestG2Invariance:
    """Validate ||[g(a),g(b),g(c)]|| = ||[a,b,c]|| for Fano automorphisms.

    Fano plane automorphisms form a finite subgroup (order 168) of the
    exceptional Lie group G2. Testing both generators plus their composition
    covers a non-trivially generated group element.
    """

    @pytest.mark.parametrize(
        "gen_idx", [0, 1], ids=["cycle_7", "quad_res"]
    )
    @given(a=unit_octonions(), b=unit_octonions(), c=unit_octonions())
    @settings(max_examples=2000, deadline=None)
    def test_fano_automorphism_preserves_associator_norm(
        self, gen_idx: int, a: Octonion, b: Octonion, c: Octonion
    ) -> None:
        """Associator norm is invariant under Fano plane automorphism generators."""
        perm = FANO_PLANE.automorphism_generators[gen_idx]

        original = associator(a, b, c)._data.norm().item()
        transformed = associator(
            _apply_fano_automorphism(a, perm),
            _apply_fano_automorphism(b, perm),
            _apply_fano_automorphism(c, perm),
        )._data.norm().item()

        assert abs(original - transformed) < ATOL_FLOAT64, (
            f"Associator norm changed under automorphism: "
            f"{original:.12f} -> {transformed:.12f}"
        )

    @given(a=unit_octonions(), b=unit_octonions(), c=unit_octonions())
    @settings(max_examples=2000, deadline=None)
    def test_composed_automorphism_preserves_associator_norm(
        self, a: Octonion, b: Octonion, c: Octonion
    ) -> None:
        """Associator norm is invariant under cycle_7 ∘ quad_res.

        This tests a non-trivially generated group element beyond the two
        individual generators, covering a product element of the order-168 group.
        """
        gen1, gen2 = FANO_PLANE.automorphism_generators
        composed = _compose_perm(gen1, gen2)

        original = associator(a, b, c)._data.norm().item()
        transformed = associator(
            _apply_fano_automorphism(a, composed),
            _apply_fano_automorphism(b, composed),
            _apply_fano_automorphism(c, composed),
        )._data.norm().item()

        assert abs(original - transformed) < ATOL_FLOAT64, (
            f"Associator norm changed under composed automorphism: "
            f"{original:.12f} -> {transformed:.12f}"
        )
