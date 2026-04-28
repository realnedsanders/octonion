"""Tests validating proximity bounds from the thesis (oct-trie.tex).

Covers:
- Proposition 9.5: Subalgebra proximity bound O(epsilon) (lines 750-770)
- Corollary 9.6: Element proximity bound O(epsilon^2) (lines 774-785)
"""

from __future__ import annotations

import math

import hypothesis.strategies as st
import pytest
import torch
from hypothesis import assume, given, settings

from octonion import Octonion
from octonion._fano import FANO_PLANE
from octonion._octonion import associator

# ── Helpers ──────────────────────────────────────────────────────────


def _subalgebra_indices(sub_idx: int) -> tuple[list[int], list[int]]:
    """Return (parallel_indices, perpendicular_indices) for a subalgebra.

    parallel_indices: [0] + list(triple) — the real part plus 3 imaginary units.
    perpendicular_indices: the remaining 4 imaginary indices.
    """
    triple = FANO_PLANE.triples[sub_idx]
    par = [0] + list(triple)
    perp = [i for i in range(1, 8) if i not in triple]
    return par, perp


def _make_near_subalgebra(
    parallel: torch.Tensor,
    perp: torch.Tensor,
    epsilon: float,
    sub_idx: int = 0,
) -> Octonion:
    """Create a unit octonion within epsilon of a quaternionic subalgebra.

    Args:
        parallel: 4 coefficients for the subalgebra components.
        perp: 4 coefficients for the perpendicular components (scaled by epsilon).
        epsilon: Distance scale for perpendicular component.
        sub_idx: Index into FANO_PLANE.triples (0-6).
    """
    par_indices, perp_indices = _subalgebra_indices(sub_idx)
    data = torch.zeros(8, dtype=torch.float64)
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

    @pytest.mark.parametrize("sub_idx", range(7))
    @pytest.mark.parametrize("epsilon", [0.1, 0.01, 0.001])
    @given(
        a_par=_unit_4d(), a_perp=_unit_4d(),
        b_par=_unit_4d(), b_perp=_unit_4d(),
        c_par=_unit_4d(), c_perp=_unit_4d(),
    )
    @settings(max_examples=200, deadline=None)
    def test_linear_scaling(
        self,
        sub_idx: int,
        epsilon: float,
        a_par: torch.Tensor, a_perp: torch.Tensor,
        b_par: torch.Tensor, b_perp: torch.Tensor,
        c_par: torch.Tensor, c_perp: torch.Tensor,
    ) -> None:
        """||[a,b,c]|| / epsilon should be bounded by a constant K.

        Calibrated bound: observed max ratio across all 7 subalgebras is ~6,
        so K=12 gives 2x headroom. Tested across all 7 quaternionic subalgebras.
        """
        a = _make_near_subalgebra(a_par, a_perp, epsilon, sub_idx)
        b = _make_near_subalgebra(b_par, b_perp, epsilon, sub_idx)
        c = _make_near_subalgebra(c_par, c_perp, epsilon, sub_idx)

        assoc = associator(a, b, c)
        norm = assoc._data.norm().item()

        # O(epsilon) means norm/epsilon should be bounded
        ratio = norm / epsilon
        assert ratio < 12.0, (
            f"||[a,b,c]||/epsilon = {ratio:.4f} > 12 at epsilon={epsilon}, "
            f"sub_idx={sub_idx}, violating O(epsilon) bound"
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

        Calibrated bound: observed max ratio across 5K random trials is ~4.3,
        so K=10 gives ~2.3x headroom.
        """
        a = _make_near_element(q, da, epsilon)
        b = _make_near_element(q, db, epsilon)
        c = _make_near_element(q, dc, epsilon)

        assoc = associator(a, b, c)
        norm = assoc._data.norm().item()

        ratio = norm / (epsilon ** 2)
        assert ratio < 10.0, (
            f"||[a,b,c]||/epsilon^2 = {ratio:.4f} > 10 at epsilon={epsilon}, "
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


# ── Bound Calibration ────────────────────────────────────────────────


class TestBoundCalibration:
    """Empirical calibration of the K constants used in the scaling tests.

    These tests document the observed maximum ratios that justify the K bounds
    in TestSubalgebraProximity (K=12) and TestElementProximity (K=10). If the
    algebra implementation changes and the true constants shift, these tests
    will detect it — either by the observed max exceeding K/2 (suggesting K
    is too tight) or by staying well below K/4 (suggesting K could be tightened).
    """

    N_CALIBRATION_SAMPLES = 5000

    def test_subalgebra_bound_calibration(self) -> None:
        """Measure the empirical max of ||[a,b,c]||/epsilon across all 7 subalgebras.

        The K=12 bound in TestSubalgebraProximity is set to 2x the observed
        maximum (~5). This test verifies the observed max and documents it.
        """
        gen = torch.Generator().manual_seed(42)
        max_ratio = 0.0
        eps = 0.1

        for sub_idx in range(7):
            par_indices, perp_indices = _subalgebra_indices(sub_idx)
            for _ in range(self.N_CALIBRATION_SAMPLES // 7):
                data_list = []
                for _ in range(3):  # a, b, c
                    par = torch.randn(4, dtype=torch.float64, generator=gen)
                    par = par / par.norm()
                    perp = torch.randn(4, dtype=torch.float64, generator=gen)
                    perp = perp / perp.norm()
                    data = torch.zeros(8, dtype=torch.float64)
                    for i, idx in enumerate(par_indices):
                        data[idx] = par[i]
                    for i, idx in enumerate(perp_indices):
                        data[idx] = eps * perp[i]
                    data = data / data.norm()
                    data_list.append(Octonion(data))

                norm = associator(*data_list)._data.norm().item()
                max_ratio = max(max_ratio, norm / eps)

        # Document the observed maximum
        K_BOUND = 12.0
        assert max_ratio < K_BOUND, (
            f"Observed max ratio {max_ratio:.4f} exceeds K={K_BOUND}; "
            f"bound needs updating"
        )
        assert max_ratio > K_BOUND / 4, (
            f"Observed max ratio {max_ratio:.4f} is far below K/4={K_BOUND/4:.1f}; "
            f"bound could be tightened (current 2x headroom is excessive)"
        )

    def test_element_bound_calibration(self) -> None:
        """Measure the empirical max of ||[a,b,c]||/epsilon^2 for element proximity.

        The K=10 bound in TestElementProximity is set to ~2.3x the observed
        maximum (~4.3). This test verifies the observed max and documents it.
        """
        gen = torch.Generator().manual_seed(42)
        max_ratio = 0.0
        eps = 0.1

        for _ in range(self.N_CALIBRATION_SAMPLES):
            q = torch.randn(8, dtype=torch.float64, generator=gen)
            q = q / q.norm()

            elems = []
            for _ in range(3):
                delta = torch.randn(8, dtype=torch.float64, generator=gen)
                delta = delta / delta.norm()
                data = (q + eps * delta)
                data = data / data.norm()
                elems.append(Octonion(data))

            norm = associator(*elems)._data.norm().item()
            max_ratio = max(max_ratio, norm / (eps ** 2))

        K_BOUND = 10.0
        assert max_ratio < K_BOUND, (
            f"Observed max ratio {max_ratio:.4f} exceeds K={K_BOUND}; "
            f"bound needs updating"
        )
        assert max_ratio > K_BOUND / 4, (
            f"Observed max ratio {max_ratio:.4f} is far below K/4={K_BOUND/4:.1f}; "
            f"bound could be tightened"
        )
