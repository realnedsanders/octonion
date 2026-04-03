"""Tests validating proximity bounds from the thesis (oct-trie.tex).

Covers:
- Proposition 9.5: Subalgebra proximity bound O(epsilon) (lines 750-770)
- Corollary 9.6: Element proximity bound O(epsilon^2) (lines 774-785)
"""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings, assume
import hypothesis.strategies as st

from octonion import Octonion
from octonion._octonion import associator


# ── Helpers ──────────────────────────────────────────────────────────


def _make_near_subalgebra(
    parallel: torch.Tensor,
    perp: torch.Tensor,
    epsilon: float,
) -> Octonion:
    """Create a unit octonion within epsilon of subalgebra span{1, e1, e2, e4}.

    Args:
        parallel: 4 coefficients for {1, e1, e2, e4} components.
        perp: 4 coefficients for {e3, e5, e6, e7} components (scaled by epsilon).
        epsilon: Distance scale for perpendicular component.
    """
    data = torch.zeros(8, dtype=torch.float64)
    par_indices = [0, 1, 2, 4]
    perp_indices = [3, 5, 6, 7]
    for i, idx in enumerate(par_indices):
        data[idx] = parallel[i]
    for i, idx in enumerate(perp_indices):
        data[idx] = epsilon * perp[i]
    data = data / data.norm()
    return Octonion(data)


def _make_near_element(
    q: torch.Tensor,
    delta: torch.Tensor,
    epsilon: float,
) -> Octonion:
    """Create a unit octonion within epsilon of element q.

    Args:
        q: Base unit octonion (8D, should be unit norm).
        delta: Perturbation direction (8D, should be unit norm).
        epsilon: Perturbation magnitude.
    """
    data = q + epsilon * delta
    data = data / data.norm()
    return Octonion(data)


# ── Custom Strategies ────────────────────────────────────────────────


@st.composite
def _unit_4d(draw: st.DrawFn) -> torch.Tensor:
    """Generate a random unit 4-vector."""
    elements = st.floats(min_value=-1.0, max_value=1.0,
                         allow_nan=False, allow_infinity=False)
    components = [draw(elements) for _ in range(4)]
    t = torch.tensor(components, dtype=torch.float64)
    assume(t.norm().item() > 0.1)
    return t / t.norm()


@st.composite
def _unit_8d(draw: st.DrawFn) -> torch.Tensor:
    """Generate a random unit 8-vector."""
    elements = st.floats(min_value=-1.0, max_value=1.0,
                         allow_nan=False, allow_infinity=False)
    components = [draw(elements) for _ in range(8)]
    t = torch.tensor(components, dtype=torch.float64)
    assume(t.norm().item() > 0.1)
    return t / t.norm()


# ── Proposition 9.5: Subalgebra Proximity O(epsilon) ────────────────


class TestSubalgebraProximity:
    """Elements within epsilon of a quaternionic subalgebra have ||[a,b,c]|| = O(epsilon)."""

    @pytest.mark.parametrize("epsilon", [0.1, 0.01, 0.001])
    @given(
        a_par=_unit_4d(), a_perp=_unit_4d(),
        b_par=_unit_4d(), b_perp=_unit_4d(),
        c_par=_unit_4d(), c_perp=_unit_4d(),
    )
    @settings(max_examples=500, deadline=None)
    def test_linear_scaling(
        self,
        epsilon: float,
        a_par: torch.Tensor, a_perp: torch.Tensor,
        b_par: torch.Tensor, b_perp: torch.Tensor,
        c_par: torch.Tensor, c_perp: torch.Tensor,
    ) -> None:
        """||[a,b,c]|| / epsilon should be bounded by a constant K.

        We use K = 20 which is generous but validates the O(epsilon) claim.
        """
        a = _make_near_subalgebra(a_par, a_perp, epsilon)
        b = _make_near_subalgebra(b_par, b_perp, epsilon)
        c = _make_near_subalgebra(c_par, c_perp, epsilon)

        assoc = associator(a, b, c)
        norm = assoc._data.norm().item()

        # O(epsilon) means norm/epsilon should be bounded
        ratio = norm / epsilon
        assert ratio < 20.0, (
            f"||[a,b,c]||/epsilon = {ratio:.4f} > 20 at epsilon={epsilon}, "
            f"violating O(epsilon) bound"
        )


# ── Corollary 9.6: Element Proximity O(epsilon^2) ───────────────────


class TestElementProximity:
    """Elements within epsilon of the same element q have ||[a,b,c]|| = O(epsilon^2)."""

    @pytest.mark.parametrize("epsilon", [0.1, 0.01, 0.001])
    @given(q=_unit_8d(), da=_unit_8d(), db=_unit_8d(), dc=_unit_8d())
    @settings(max_examples=500, deadline=None)
    def test_quadratic_scaling(
        self,
        epsilon: float,
        q: torch.Tensor, da: torch.Tensor, db: torch.Tensor, dc: torch.Tensor,
    ) -> None:
        """||[a,b,c]|| / epsilon^2 should be bounded by a constant K.

        We use K = 50 which is generous but validates the O(epsilon^2) claim.
        """
        a = _make_near_element(q, da, epsilon)
        b = _make_near_element(q, db, epsilon)
        c = _make_near_element(q, dc, epsilon)

        assoc = associator(a, b, c)
        norm = assoc._data.norm().item()

        ratio = norm / (epsilon ** 2)
        assert ratio < 50.0, (
            f"||[a,b,c]||/epsilon^2 = {ratio:.4f} > 50 at epsilon={epsilon}, "
            f"violating O(epsilon^2) bound"
        )

    def test_scaling_exponent(self) -> None:
        """The scaling exponent should be approximately 2 (not 1).

        Computes associator norms at two epsilon values and derives the
        exponent alpha = log(norm_ratio) / log(eps_ratio). For O(epsilon^2),
        alpha should be close to 2.
        """
        torch.manual_seed(42)

        exponents = []
        for seed in range(10):
            gen = torch.Generator().manual_seed(seed)
            q = torch.randn(8, dtype=torch.float64, generator=gen)
            q = q / q.norm()
            da = torch.randn(8, dtype=torch.float64, generator=gen)
            da = da / da.norm()
            db = torch.randn(8, dtype=torch.float64, generator=gen)
            db = db / db.norm()
            dc = torch.randn(8, dtype=torch.float64, generator=gen)
            dc = dc / dc.norm()

            norms = {}
            for eps in [0.1, 0.01]:
                a = _make_near_element(q, da, eps)
                b = _make_near_element(q, db, eps)
                c = _make_near_element(q, dc, eps)
                assoc = associator(a, b, c)
                norms[eps] = assoc._data.norm().item()

            if norms[0.1] > 1e-15 and norms[0.01] > 1e-15:
                alpha = math.log(norms[0.1] / norms[0.01]) / math.log(0.1 / 0.01)
                exponents.append(alpha)

        assert len(exponents) >= 5, "Too few valid exponent measurements"
        mean_alpha = sum(exponents) / len(exponents)
        assert 1.8 <= mean_alpha <= 2.2, (
            f"Mean scaling exponent {mean_alpha:.3f} not in [1.8, 2.2]; "
            f"expected ~2.0 for O(epsilon^2)"
        )
