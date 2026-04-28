"""FOUND-01 property-based test suite: Moufang identities, norm preservation,
inverse, alternativity, and associator properties.

This file implements the success criteria for Phase 1 of the ROADMAP:
1. Moufang identities pass on 10,000+ random octonion triples at float64 precision
2. Norm preservation |ab| = |a||b| holds to within 1e-12 relative error
4. Inverse satisfies a * a_inv = 1 and a_inv * a = 1 to numerical precision
5. Associator is non-zero for generic triples but zero when any two args are equal

(Criterion 3, Cayley-Dickson cross-check, was tested in Plan 01.)

All tests use Hypothesis property-based testing with detailed precision statistics
reporting (max/mean/std of relative errors) per user decision.
"""


import hypothesis.strategies as st
import pytest
import torch
from conftest import (
    ATOL_FLOAT64,
    nonzero_octonions,
    octonions,
    subalgebra_octonions,
)
from hypothesis import example, given, settings, target

from octonion import Octonion, associator

# =============================================================================
# Precision tracking utilities
# =============================================================================


def report_precision(errors: torch.Tensor, test_name: str) -> dict[str, float]:
    """Compute and print precision statistics for a set of errors.

    Args:
        errors: Tensor of absolute error values.
        test_name: Name of the test for reporting.

    Returns:
        Dict with max_error, mean_error, std_error.
    """
    stats = {
        "max_error": errors.max().item(),
        "mean_error": errors.mean().item(),
        "std_error": errors.std().item() if errors.numel() > 1 else 0.0,
    }
    print(
        f"\n  [{test_name}] max={stats['max_error']:.2e}  "
        f"mean={stats['mean_error']:.2e}  std={stats['std_error']:.2e}"
    )
    return stats


def check_moufang(
    a: Octonion, b: Octonion, c: Octonion, tol: float = ATOL_FLOAT64
) -> dict[str, dict]:
    """Verify all four Moufang identities and return detailed error statistics.

    The four identities (using x=a, y=b, z=c):
    1. z(x(zy)) = ((zx)z)y
    2. x(z(yz)) = ((xz)y)z
    3. (zx)(yz) = (z(xy))z
    4. (xy)x = x(yx)  [flexibility]

    Args:
        a, b, c: Octonion instances.
        tol: Tolerance for pass/fail determination.

    Returns:
        Dict with per-identity error statistics and overall pass/fail.
    """
    results: dict[str, dict] = {}

    # Identity 1: z(x(zy)) = ((zx)z)y
    lhs1 = c * (a * (c * b))
    rhs1 = ((c * a) * c) * b
    err1 = (lhs1.components - rhs1.components).abs()
    results["moufang_1"] = {
        "max_error": err1.max().item(),
        "mean_error": err1.mean().item(),
        "identity": "z(x(zy)) = ((zx)z)y",
    }

    # Identity 2: x(z(yz)) = ((xz)y)z
    lhs2 = a * (c * (b * c))
    rhs2 = ((a * c) * b) * c
    err2 = (lhs2.components - rhs2.components).abs()
    results["moufang_2"] = {
        "max_error": err2.max().item(),
        "mean_error": err2.mean().item(),
        "identity": "x(z(yz)) = ((xz)y)z",
    }

    # Identity 3: (zx)(yz) = (z(xy))z
    lhs3 = (c * a) * (b * c)
    rhs3 = (c * (a * b)) * c
    err3 = (lhs3.components - rhs3.components).abs()
    results["moufang_3"] = {
        "max_error": err3.max().item(),
        "mean_error": err3.mean().item(),
        "identity": "(zx)(yz) = (z(xy))z",
    }

    # Identity 4 (flexibility): (xy)x = x(yx)
    lhs4 = (a * b) * a
    rhs4 = a * (b * a)
    err4 = (lhs4.components - rhs4.components).abs()
    results["flexibility"] = {
        "max_error": err4.max().item(),
        "mean_error": err4.mean().item(),
        "identity": "(xy)x = x(yx)",
    }

    # Overall pass/fail
    max_err = max(r["max_error"] for r in results.values())
    results["passed"] = {"value": max_err < tol}  # type: ignore[assignment]
    results["max_error_overall"] = {"value": max_err}  # type: ignore[assignment]

    return results


# =============================================================================
# Moufang identities (ROADMAP success criterion 1)
# =============================================================================


class TestMoufangIdentities:
    """Moufang identities on 10,000+ random triples at float64 precision.

    Uses @settings(max_examples=10000) to meet ROADMAP criterion 1.

    Tests use octonions with components in [-1, 1] range to ensure that
    products (which involve 3-4 multiplications of 8D vectors) stay in a
    regime where absolute errors are well below 1e-12. For larger inputs,
    the triple products can reach magnitudes ~N^4 and float64 rounding
    errors accumulate to ~N^4 * eps ≈ 1e-12 at N=10.
    """

    @given(
        a=octonions(min_value=-1.0, max_value=1.0),
        b=octonions(min_value=-1.0, max_value=1.0),
        c=octonions(min_value=-1.0, max_value=1.0),
    )
    @settings(max_examples=10000, deadline=None)
    def test_moufang_identity_1(self, a: Octonion, b: Octonion, c: Octonion) -> None:
        """z(x(zy)) = ((zx)z)y for random triples within 1e-12."""
        lhs = c * (a * (c * b))
        rhs = ((c * a) * c) * b
        err = (lhs.components - rhs.components).abs()
        max_err = err.max().item()
        assert max_err < ATOL_FLOAT64, (
            f"Moufang identity 1 failed: z(x(zy)) = ((zx)z)y, max_error={max_err:.2e}"
        )

    @given(
        a=octonions(min_value=-1.0, max_value=1.0),
        b=octonions(min_value=-1.0, max_value=1.0),
        c=octonions(min_value=-1.0, max_value=1.0),
    )
    @settings(max_examples=10000, deadline=None)
    def test_moufang_identity_2(self, a: Octonion, b: Octonion, c: Octonion) -> None:
        """x(z(yz)) = ((xz)y)z for random triples within 1e-12."""
        lhs = a * (c * (b * c))
        rhs = ((a * c) * b) * c
        err = (lhs.components - rhs.components).abs()
        max_err = err.max().item()
        assert max_err < ATOL_FLOAT64, (
            f"Moufang identity 2 failed: x(z(yz)) = ((xz)y)z, max_error={max_err:.2e}"
        )

    @given(
        a=octonions(min_value=-1.0, max_value=1.0),
        b=octonions(min_value=-1.0, max_value=1.0),
        c=octonions(min_value=-1.0, max_value=1.0),
    )
    @settings(max_examples=10000, deadline=None)
    def test_moufang_identity_3(self, a: Octonion, b: Octonion, c: Octonion) -> None:
        """(zx)(yz) = (z(xy))z for random triples within 1e-12."""
        lhs = (c * a) * (b * c)
        rhs = (c * (a * b)) * c
        err = (lhs.components - rhs.components).abs()
        max_err = err.max().item()
        assert max_err < ATOL_FLOAT64, (
            f"Moufang identity 3 failed: (zx)(yz) = (z(xy))z, max_error={max_err:.2e}"
        )

    @given(
        a=octonions(min_value=-1.0, max_value=1.0),
        b=octonions(min_value=-1.0, max_value=1.0),
    )
    @settings(max_examples=10000, deadline=None)
    def test_flexibility(self, a: Octonion, b: Octonion) -> None:
        """(xy)x = x(yx) for random pairs within 1e-12."""
        lhs = (a * b) * a
        rhs = a * (b * a)
        err = (lhs.components - rhs.components).abs()
        max_err = err.max().item()
        assert max_err < ATOL_FLOAT64, (
            f"Flexibility failed: (xy)x = x(yx), max_error={max_err:.2e}"
        )


# =============================================================================
# Norm preservation (ROADMAP success criterion 2)
# =============================================================================


class TestNormPreservation:
    """Norm preservation |a*b| = |a|*|b| within 1e-12 relative error."""

    # Edge cases: identity, basis elements, pure-real, near-zero
    @example(
        a=Octonion(torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)),
        b=Octonion(torch.tensor([0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float64)),
    )
    @example(
        a=Octonion(torch.tensor([1e-15, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)),
        b=Octonion(torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)),
    )
    @given(
        a=octonions(min_value=-10, max_value=10),
        b=octonions(min_value=-10, max_value=10),
    )
    @settings(max_examples=10000, deadline=None)
    def test_norm_preservation(self, a: Octonion, b: Octonion) -> None:
        """|a*b| = |a|*|b| within 1e-12 relative error for random inputs."""
        product_norm = (a * b).norm()
        expected_norm = a.norm() * b.norm()

        # Relative error when expected > 0; absolute when near zero
        if expected_norm.item() > 1e-15:
            rel_error = torch.abs(product_norm - expected_norm) / expected_norm
            # Direct Hypothesis toward worst-case inputs
            target(float(rel_error.item()), label="norm_preservation_rel_error")
            assert rel_error.item() < ATOL_FLOAT64, (
                f"Norm preservation failed: |a*b|={product_norm.item():.6e}, "
                f"|a|*|b|={expected_norm.item():.6e}, "
                f"rel_error={rel_error.item():.2e}"
            )
        else:
            abs_error = torch.abs(product_norm - expected_norm)
            assert abs_error.item() < ATOL_FLOAT64, (
                f"Norm preservation failed near zero: abs_error={abs_error.item():.2e}"
            )


# =============================================================================
# Inverse (ROADMAP success criterion 4)
# =============================================================================


class TestInverse:
    """Inverse satisfies a * a_inv = 1 and a_inv * a = 1 to numerical precision."""

    @given(a=nonzero_octonions())
    @settings(max_examples=10000, deadline=None)
    def test_inverse_left(self, a: Octonion) -> None:
        """a * a.inverse() has all components within 1e-12 of identity [1,0,...,0]."""
        identity = torch.zeros(8, dtype=torch.float64)
        identity[0] = 1.0

        result = a * a.inverse()
        err = (result.components - identity).abs()
        max_err = err.max().item()
        assert max_err < ATOL_FLOAT64, (
            f"Left inverse failed: a * a^-1 should be identity, "
            f"max_error={max_err:.2e}, norm(a)={a.norm().item():.2e}"
        )

    @given(a=nonzero_octonions())
    @settings(max_examples=10000, deadline=None)
    def test_inverse_right(self, a: Octonion) -> None:
        """a.inverse() * a has all components within 1e-12 of identity [1,0,...,0]."""
        identity = torch.zeros(8, dtype=torch.float64)
        identity[0] = 1.0

        result = a.inverse() * a
        err = (result.components - identity).abs()
        max_err = err.max().item()
        assert max_err < ATOL_FLOAT64, (
            f"Right inverse failed: a^-1 * a should be identity, "
            f"max_error={max_err:.2e}, norm(a)={a.norm().item():.2e}"
        )


# =============================================================================
# Alternativity (ROADMAP success criterion 5)
# =============================================================================


class TestAlternativity:
    """Alternativity: associator is zero when any two args are equal."""

    @given(
        a=octonions(min_value=-1.0, max_value=1.0),
        b=octonions(min_value=-1.0, max_value=1.0),
    )
    @settings(max_examples=10000, deadline=None)
    def test_left_alternativity(self, a: Octonion, b: Octonion) -> None:
        """a*(a*b) = (a*a)*b within 1e-12 (left alternativity)."""
        lhs = a * (a * b)
        rhs = (a * a) * b
        err = (lhs.components - rhs.components).abs()
        max_err = err.max().item()
        assert max_err < ATOL_FLOAT64, (
            f"Left alternativity failed: a*(a*b) = (a*a)*b, max_error={max_err:.2e}"
        )

    @given(
        a=octonions(min_value=-1.0, max_value=1.0),
        b=octonions(min_value=-1.0, max_value=1.0),
    )
    @settings(max_examples=10000, deadline=None)
    def test_right_alternativity(self, a: Octonion, b: Octonion) -> None:
        """(b*a)*a = b*(a*a) within 1e-12 (right alternativity)."""
        lhs = (b * a) * a
        rhs = b * (a * a)
        err = (lhs.components - rhs.components).abs()
        max_err = err.max().item()
        assert max_err < ATOL_FLOAT64, (
            f"Right alternativity failed: (b*a)*a = b*(a*a), max_error={max_err:.2e}"
        )

    @given(
        a=octonions(min_value=-1.0, max_value=1.0),
        b=octonions(min_value=-1.0, max_value=1.0),
    )
    @settings(max_examples=10000, deadline=None)
    def test_associator_zero_equal_args(self, a: Octonion, b: Octonion) -> None:
        """Associator is zero when any two arguments are equal.

        [a,a,b] = 0, [a,b,a] = 0, [a,b,b] = 0
        """
        # [a, a, b] = 0
        assoc_aab = associator(a, a, b)
        err_aab = assoc_aab.components.abs().max().item()
        assert err_aab < ATOL_FLOAT64, (
            f"[a,a,b] should be zero, max_error={err_aab:.2e}"
        )

        # [a, b, a] = 0
        assoc_aba = associator(a, b, a)
        err_aba = assoc_aba.components.abs().max().item()
        assert err_aba < ATOL_FLOAT64, (
            f"[a,b,a] should be zero, max_error={err_aba:.2e}"
        )

        # [a, b, b] = 0
        assoc_abb = associator(a, b, b)
        err_abb = assoc_abb.components.abs().max().item()
        assert err_abb < ATOL_FLOAT64, (
            f"[a,b,b] should be zero, max_error={err_abb:.2e}"
        )


class TestAssociatorNonzero:
    """Associator is non-zero for generic random triples (genuine non-associativity)."""

    def test_associator_nonzero_generic(self) -> None:
        """For random non-degenerate triples, ||[a,b,c]||/(||a||*||b||*||c||) is O(1).

        Tests that the algebra is genuinely non-associative (not just floating-point noise).
        Uses 100 random triples and checks that a significant fraction have large associators.
        """
        torch.manual_seed(42)
        n_trials = 100
        n_nonzero = 0
        magnitudes = []

        for _ in range(n_trials):
            a = Octonion(torch.randn(8, dtype=torch.float64))
            b = Octonion(torch.randn(8, dtype=torch.float64))
            c = Octonion(torch.randn(8, dtype=torch.float64))

            assoc = associator(a, b, c)
            assoc_norm = assoc.norm().item()
            denom = a.norm().item() * b.norm().item() * c.norm().item()

            if denom > 1e-10:
                relative_magnitude = assoc_norm / denom
                magnitudes.append(relative_magnitude)
                if relative_magnitude > 0.01:
                    n_nonzero += 1

        # At least 50% of random triples should have large associators
        assert n_nonzero > n_trials * 0.5, (
            f"Only {n_nonzero}/{n_trials} triples had ||[a,b,c]||/(||a||*||b||*||c||) > 0.01. "
            f"Expected genuine non-associativity."
        )

        # Report statistics
        t_mags = torch.tensor(magnitudes, dtype=torch.float64)
        report_precision(t_mags, "associator_magnitude")


class TestAssociatorAntisymmetry:
    """Associator is totally antisymmetric: [a,b,c] = -[b,a,c] = -[a,c,b]."""

    @given(
        a=octonions(min_value=-1.0, max_value=1.0),
        b=octonions(min_value=-1.0, max_value=1.0),
        c=octonions(min_value=-1.0, max_value=1.0),
    )
    @settings(max_examples=5000, deadline=None)
    def test_associator_antisymmetry(
        self, a: Octonion, b: Octonion, c: Octonion
    ) -> None:
        """[a,b,c] = -[b,a,c] and [a,b,c] = -[a,c,b]."""
        abc = associator(a, b, c)
        bac = associator(b, a, c)
        acb = associator(a, c, b)

        # [a,b,c] = -[b,a,c]  =>  [a,b,c] + [b,a,c] = 0
        sum1 = abc.components + bac.components
        err1 = sum1.abs().max().item()
        assert err1 < ATOL_FLOAT64, (
            f"Antisymmetry failed: [a,b,c] + [b,a,c] should be 0, max_error={err1:.2e}"
        )

        # [a,b,c] = -[a,c,b]  =>  [a,b,c] + [a,c,b] = 0
        sum2 = abc.components + acb.components
        err2 = sum2.abs().max().item()
        assert err2 < ATOL_FLOAT64, (
            f"Antisymmetry failed: [a,b,c] + [a,c,b] should be 0, max_error={err2:.2e}"
        )


# =============================================================================
# Precision statistics reporting (aggregated Moufang check)
# =============================================================================


class TestMoufangPrecisionReport:
    """Run check_moufang on a batch and report detailed precision statistics."""

    def test_moufang_precision_statistics(self) -> None:
        """Run Moufang check on 1000 random triples and report max/mean/std errors."""
        torch.manual_seed(0)
        all_errors = {"moufang_1": [], "moufang_2": [], "moufang_3": [], "flexibility": []}

        for _ in range(1000):
            # Use unit-scale inputs so triple products stay O(1) and
            # absolute errors stay well below 1e-12
            a = Octonion(torch.randn(8, dtype=torch.float64))
            b = Octonion(torch.randn(8, dtype=torch.float64))
            c = Octonion(torch.randn(8, dtype=torch.float64))

            results = check_moufang(a, b, c)
            for key in all_errors:
                all_errors[key].append(results[key]["max_error"])

        print("\n=== Moufang Precision Statistics (1000 random triples) ===")
        for key in all_errors:
            errors_t = torch.tensor(all_errors[key], dtype=torch.float64)
            stats = report_precision(errors_t, key)
            assert stats["max_error"] < ATOL_FLOAT64, (
                f"{key}: max_error={stats['max_error']:.2e} exceeds tolerance {ATOL_FLOAT64}"
            )


# =============================================================================
# Conjugation rule
# =============================================================================


class TestConjugationRule:
    """conj(x*y) = conj(y) * conj(x) — the anti-automorphism property."""

    # Edge cases: identity, pure-imaginary, basis elements
    @example(
        a=Octonion(torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)),
        b=Octonion(torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float64)),
    )
    @example(
        a=Octonion(torch.tensor([0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float64)),
        b=Octonion(torch.tensor([0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float64)),
    )
    @given(
        a=octonions(min_value=-1.0, max_value=1.0),
        b=octonions(min_value=-1.0, max_value=1.0),
    )
    @settings(max_examples=5000, deadline=None)
    def test_product_reversal(self, a: Octonion, b: Octonion) -> None:
        """conj(a*b) should equal conj(b)*conj(a)."""
        lhs = (a * b).conjugate()
        rhs = b.conjugate() * a.conjugate()
        err = (lhs.components - rhs.components).abs().max().item()
        assert err < ATOL_FLOAT64, (
            f"Conjugation reversal failed: max_error={err:.2e}"
        )


# =============================================================================
# Associator trilinearity
# =============================================================================


class TestAssociatorTrilinearity:
    """Associator is trilinear: [s*a, b, c] = s*[a, b, c] in each slot."""

    @given(
        a=octonions(min_value=-1.0, max_value=1.0),
        b=octonions(min_value=-1.0, max_value=1.0),
        c=octonions(min_value=-1.0, max_value=1.0),
        s=st.floats(min_value=-2.0, max_value=2.0,
                     allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=3000, deadline=None)
    def test_scalar_first_slot(
        self, a: Octonion, b: Octonion, c: Octonion, s: float
    ) -> None:
        """[s*a, b, c] = s * [a, b, c]."""
        lhs = associator(a * s, b, c)
        rhs = associator(a, b, c) * s
        err = (lhs.components - rhs.components).abs().max().item()
        target(err, label="trilinearity_slot1_error")
        assert err < ATOL_FLOAT64 * 10, (
            f"Trilinearity (slot 1) failed: max_error={err:.2e}"
        )

    @given(
        a=octonions(min_value=-1.0, max_value=1.0),
        b=octonions(min_value=-1.0, max_value=1.0),
        c=octonions(min_value=-1.0, max_value=1.0),
        s=st.floats(min_value=-2.0, max_value=2.0,
                     allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=3000, deadline=None)
    def test_scalar_second_slot(
        self, a: Octonion, b: Octonion, c: Octonion, s: float
    ) -> None:
        """[a, s*b, c] = s * [a, b, c]."""
        lhs = associator(a, b * s, c)
        rhs = associator(a, b, c) * s
        err = (lhs.components - rhs.components).abs().max().item()
        assert err < ATOL_FLOAT64 * 10, (
            f"Trilinearity (slot 2) failed: max_error={err:.2e}"
        )

    @given(
        a=octonions(min_value=-1.0, max_value=1.0),
        b=octonions(min_value=-1.0, max_value=1.0),
        c=octonions(min_value=-1.0, max_value=1.0),
        s=st.floats(min_value=-2.0, max_value=2.0,
                     allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=3000, deadline=None)
    def test_scalar_third_slot(
        self, a: Octonion, b: Octonion, c: Octonion, s: float
    ) -> None:
        """[a, b, s*c] = s * [a, b, c]."""
        lhs = associator(a, b, c * s)
        rhs = associator(a, b, c) * s
        err = (lhs.components - rhs.components).abs().max().item()
        assert err < ATOL_FLOAT64 * 10, (
            f"Trilinearity (slot 3) failed: max_error={err:.2e}"
        )


# =============================================================================
# Full associator antisymmetry (all 6 permutations)
# =============================================================================


class TestAssociatorFullAntisymmetry:
    """Associator is totally antisymmetric across all 6 permutations."""

    @given(
        a=octonions(min_value=-1.0, max_value=1.0),
        b=octonions(min_value=-1.0, max_value=1.0),
        c=octonions(min_value=-1.0, max_value=1.0),
    )
    @settings(max_examples=5000, deadline=None)
    def test_all_six_permutations(
        self, a: Octonion, b: Octonion, c: Octonion
    ) -> None:
        """[a,b,c] = -[b,a,c] = -[a,c,b] = -[c,b,a] = [b,c,a] = [c,a,b]."""
        abc = associator(a, b, c).components

        # Odd permutations: negate
        bac = associator(b, a, c).components
        acb = associator(a, c, b).components
        cba = associator(c, b, a).components

        # Even permutations: same sign
        bca = associator(b, c, a).components
        cab = associator(c, a, b).components

        for name, val in [("-[b,a,c]", -bac), ("-[a,c,b]", -acb),
                          ("-[c,b,a]", -cba), ("[b,c,a]", bca),
                          ("[c,a,b]", cab)]:
            err = (abc - val).abs().max().item()
            assert err < ATOL_FLOAT64, (
                f"Full antisymmetry failed for {name}: max_error={err:.2e}"
            )


# =============================================================================
# Subalgebra associativity
# =============================================================================


class TestSubalgebraAssociativity:
    """Elements in the same quaternionic subalgebra must associate."""

    @pytest.mark.parametrize("sub_idx", range(7), ids=[
        f"S{i}({t})" for i, t in enumerate([
            "1,2,4", "2,3,5", "3,4,6", "4,5,7", "5,6,1", "6,7,2", "7,1,3"
        ])
    ])
    @given(data=st.data())
    @settings(max_examples=1000, deadline=None)
    def test_quaternionic_subalgebra_zero_associator(
        self, sub_idx: int, data: st.DataObject
    ) -> None:
        """[a,b,c] = 0 when a,b,c lie in the same quaternionic subalgebra."""
        a = data.draw(subalgebra_octonions(sub_idx, min_value=-1.0, max_value=1.0))
        b = data.draw(subalgebra_octonions(sub_idx, min_value=-1.0, max_value=1.0))
        c = data.draw(subalgebra_octonions(sub_idx, min_value=-1.0, max_value=1.0))

        assoc = associator(a, b, c)
        norm = assoc.components.norm().item()
        assert norm < ATOL_FLOAT64, (
            f"Subalgebra {sub_idx}: associator norm {norm:.2e} > 0 "
            f"(should be 0 for same-subalgebra elements)"
        )


# =============================================================================
# Artin's theorem
# =============================================================================


class TestArtinsTheorem:
    """Any two octonions generate an associative subalgebra (Artin's theorem)."""

    @given(
        a=octonions(min_value=-1.0, max_value=1.0),
        b=octonions(min_value=-1.0, max_value=1.0),
    )
    @settings(max_examples=5000, deadline=None)
    def test_two_element_products_associate(
        self, a: Octonion, b: Octonion
    ) -> None:
        """Products of any two elements form an associative set.

        Tests representative triples from the subalgebra generated by a,b:
        - [a, b, a*b] = 0
        - [a*b, a, b] = 0
        - [a, b, a] = 0  (flexibility, included for completeness)
        """
        ab = a * b

        for name, x, y, z in [
            ("[a,b,a*b]", a, b, ab),
            ("[a*b,a,b]", ab, a, b),
            ("[a,b,a]", a, b, a),
        ]:
            assoc = associator(x, y, z)
            norm = assoc.components.norm().item()
            assert norm < ATOL_FLOAT64 * 10, (
                f"Artin's theorem failed for {name}: norm={norm:.2e}"
            )


# =============================================================================
# General associator norm bound
# =============================================================================


class TestAssociatorNormBound:
    """||[a,b,c]|| <= 2*||a||*||b||*||c|| for arbitrary octonions."""

    @given(
        a=octonions(min_value=-1.0, max_value=1.0),
        b=octonions(min_value=-1.0, max_value=1.0),
        c=octonions(min_value=-1.0, max_value=1.0),
    )
    @settings(max_examples=5000, deadline=None)
    def test_general_bound(
        self, a: Octonion, b: Octonion, c: Octonion
    ) -> None:
        """Associator norm is bounded by 2 * product of input norms."""
        assoc_norm = associator(a, b, c).components.norm().item()
        bound = 2.0 * a.components.norm().item() * b.components.norm().item() * c.components.norm().item()
        assert assoc_norm <= bound + ATOL_FLOAT64, (
            f"Norm bound violated: ||[a,b,c]||={assoc_norm:.6f} > "
            f"2*||a||*||b||*||c||={bound:.6f}"
        )
