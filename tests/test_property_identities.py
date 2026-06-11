"""Property-based tests for additional octonion identities.

Covers identities not exercised by test_algebraic_properties.py:

1. Inverse anti-homomorphism: (x*y)^-1 = y^-1 * x^-1
2. Left/right multiplication matrices: transpose/conjugate duality,
   orthogonality scaling L_a^T L_a = |a|^2 I, and consistency with
   the analytic Jacobians from octonion.calculus.jacobian_mul
3. 7D cross product geometry: orthogonality and the Lagrange identity
4. Trace-form identities: Re(x*y) = Re(y*x) and trace associativity
5. exp/log algebraic identities: norm, conjugation equivariance,
   one-parameter subgroup property, and principal-branch roundtrips

All tests use float64 with components bounded to sane ranges, and norms
bounded away from zero wherever inverses or logs are involved.
"""

import math

import hypothesis.strategies as st
import torch
from hypothesis import assume, given, settings

from octonion import (
    Octonion,
    cross_product,
    inner_product,
    left_mul_matrix,
    octonion_exp,
    octonion_log,
    right_mul_matrix,
)
from octonion.calculus import jacobian_mul
from tests.conftest import octonion_tensors, octonions, unit_octonion_tensors

# Tolerances for identities involving transcendental functions (exp/log/acos)
# or division by squared norms, which amplify float64 rounding beyond the
# 1e-12 budget used for pure multiplicative identities in conftest.
RTOL = 1e-9
ATOL = 1e-9
# Looser absolute tolerance for identities with catastrophic cancellation of
# O(|u|^2 |v|^2) terms (Lagrange identity) or O(|x||y||z|) triple products.
ATOL_CANCEL = 1e-7


# =============================================================================
# Local strategies (complement the ones in tests/conftest.py)
# =============================================================================


@st.composite
def bounded_invertible_octonions(
    draw: st.DrawFn,
    *,
    min_norm: float = 0.1,
    max_value: float = 10.0,
) -> Octonion:
    """Octonions with components in [-max_value, max_value] and norm >= min_norm."""
    t = draw(octonion_tensors(min_value=-max_value, max_value=max_value))
    assume(torch.linalg.norm(t).item() >= min_norm)
    return Octonion(t)


@st.composite
def pure_imaginary_octonions(
    draw: st.DrawFn,
    *,
    min_value: float = -10.0,
    max_value: float = 10.0,
) -> Octonion:
    """Pure imaginary octonions (real part exactly zero)."""
    t = draw(octonion_tensors(min_value=min_value, max_value=max_value))
    t = t.clone()
    t[0] = 0.0
    return Octonion(t)


@st.composite
def unit_pure_imaginary_octonions(draw: st.DrawFn) -> Octonion:
    """Unit-norm pure imaginary octonions (directions in Im(O) ~ S^6)."""
    t = draw(unit_octonion_tensors())
    t = t.clone()
    t[0] = 0.0
    n = torch.linalg.norm(t)
    assume(n.item() > 1e-6)
    return Octonion(t / n)


@st.composite
def log_domain_octonions(draw: st.DrawFn) -> Octonion:
    """Octonions o = r + theta*u with imaginary norm theta in (0.05, pi - 0.1).

    This keeps exp(o) inside the principal branch of octonion_log, so the
    roundtrip log(exp(o)) = o is exact (up to float64 rounding).
    """
    r = draw(st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False))
    theta = draw(
        st.floats(
            min_value=0.05,
            max_value=math.pi - 0.1,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    u = draw(unit_pure_imaginary_octonions())
    t = u.components * theta
    t = t.clone()
    t[0] = r
    return Octonion(t)


@st.composite
def exp_domain_octonions(draw: st.DrawFn) -> Octonion:
    """Octonions with norm in (0.1, 10) and imaginary-part norm >= 0.05.

    The imaginary-norm floor keeps inputs away from the (negative) real axis,
    where octonion_log's principal branch is discontinuous.
    """
    direction = draw(unit_octonion_tensors())
    scale = draw(st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False))
    t = direction * scale
    assume(torch.linalg.norm(t[1:]).item() >= 0.05)
    return Octonion(t)


# =============================================================================
# 1. Inverse anti-homomorphism
# =============================================================================


class TestInverseAntiHomomorphism:
    """(x*y)^-1 = y^-1 * x^-1 — inversion reverses products."""

    @given(x=bounded_invertible_octonions(), y=bounded_invertible_octonions())
    @settings(max_examples=1000, deadline=None)
    def test_inverse_anti_homomorphism(self, x: Octonion, y: Octonion) -> None:
        """(x*y)^-1 = y^-1 * x^-1 for octonions with norm >= 0.1."""
        lhs = (x * y).inverse()
        rhs = y.inverse() * x.inverse()
        torch.testing.assert_close(lhs.components, rhs.components, rtol=RTOL, atol=ATOL)


# =============================================================================
# 2. Left/right multiplication matrices
# =============================================================================


class TestMultiplicationMatrices:
    """Structural identities of L_a (a*x = L_a x) and R_a (x*a = R_a x)."""

    @given(a=octonions(min_value=-10.0, max_value=10.0))
    @settings(max_examples=1000, deadline=None)
    def test_transpose_is_conjugate_left(self, a: Octonion) -> None:
        """L_a^T = L_{a*} — transposing the left matrix conjugates its argument."""
        lhs = left_mul_matrix(a).mT
        rhs = left_mul_matrix(a.conjugate())
        torch.testing.assert_close(lhs, rhs, rtol=RTOL, atol=ATOL)

    @given(a=octonions(min_value=-10.0, max_value=10.0))
    @settings(max_examples=1000, deadline=None)
    def test_transpose_is_conjugate_right(self, a: Octonion) -> None:
        """R_a^T = R_{a*} — transposing the right matrix conjugates its argument."""
        lhs = right_mul_matrix(a).mT
        rhs = right_mul_matrix(a.conjugate())
        torch.testing.assert_close(lhs, rhs, rtol=RTOL, atol=ATOL)

    @given(a=octonions(min_value=-10.0, max_value=10.0))
    @settings(max_examples=1000, deadline=None)
    def test_left_matrix_orthogonality_scaling(self, a: Octonion) -> None:
        """L_a^T @ L_a = |a|^2 I — L_a is a scaled orthogonal matrix."""
        L = left_mul_matrix(a)
        expected = a.norm() ** 2 * torch.eye(8, dtype=torch.float64)
        torch.testing.assert_close(L.mT @ L, expected, rtol=RTOL, atol=ATOL_CANCEL)

    @given(a=octonions(min_value=-10.0, max_value=10.0))
    @settings(max_examples=1000, deadline=None)
    def test_right_matrix_orthogonality_scaling(self, a: Octonion) -> None:
        """R_a^T @ R_a = |a|^2 I — R_a is a scaled orthogonal matrix."""
        R = right_mul_matrix(a)
        expected = a.norm() ** 2 * torch.eye(8, dtype=torch.float64)
        torch.testing.assert_close(R.mT @ R, expected, rtol=RTOL, atol=ATOL_CANCEL)

    @given(
        a=octonions(min_value=-10.0, max_value=10.0),
        b=octonions(min_value=-10.0, max_value=10.0),
    )
    @settings(max_examples=1000, deadline=None)
    def test_jacobian_mul_consistency(self, a: Octonion, b: Octonion) -> None:
        """jacobian_mul(a, b) = (R_b, L_a).

        Orientation verified empirically on basis elements (a = e1, b = e2)
        before writing this test: d(a*b)/da is multiplication-by-b on the
        right, so J_a = right_mul_matrix(b); d(a*b)/db is multiplication-by-a
        on the left, so J_b = left_mul_matrix(a). The documented mapping holds
        in the current implementation (no swap).
        """
        J_a, J_b = jacobian_mul(a.components, b.components)
        torch.testing.assert_close(J_a, right_mul_matrix(b), rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(J_b, left_mul_matrix(a), rtol=RTOL, atol=ATOL)


# =============================================================================
# 3. 7D cross product geometry
# =============================================================================


class TestCrossProductGeometry:
    """Geometric identities of the 7D cross product on pure imaginary octonions."""

    @given(u=pure_imaginary_octonions(), v=pure_imaginary_octonions())
    @settings(max_examples=1000, deadline=None)
    def test_orthogonality(self, u: Octonion, v: Octonion) -> None:
        """<u x v, u> = 0 and <u x v, v> = 0 — the cross product is orthogonal to both factors."""
        w = cross_product(u, v)
        zero = torch.zeros((), dtype=torch.float64)
        torch.testing.assert_close(inner_product(w, u), zero, rtol=RTOL, atol=ATOL_CANCEL)
        torch.testing.assert_close(inner_product(w, v), zero, rtol=RTOL, atol=ATOL_CANCEL)

    @given(u=pure_imaginary_octonions(), v=pure_imaginary_octonions())
    @settings(max_examples=1000, deadline=None)
    def test_lagrange_identity(self, u: Octonion, v: Octonion) -> None:
        """|u x v|^2 = |u|^2 |v|^2 - <u, v>^2 — the Lagrange identity in 7D."""
        lhs = cross_product(u, v).norm() ** 2
        rhs = u.norm() ** 2 * v.norm() ** 2 - inner_product(u, v) ** 2
        torch.testing.assert_close(lhs, rhs, rtol=RTOL, atol=ATOL_CANCEL)


# =============================================================================
# 4. Trace-form identities
# =============================================================================


class TestTraceForm:
    """The trace form Re(x*y) is symmetric and associative (the algebra is not)."""

    @given(
        x=octonions(min_value=-10.0, max_value=10.0),
        y=octonions(min_value=-10.0, max_value=10.0),
    )
    @settings(max_examples=1000, deadline=None)
    def test_trace_symmetry(self, x: Octonion, y: Octonion) -> None:
        """Re(x*y) = Re(y*x) — the trace form is symmetric."""
        torch.testing.assert_close((x * y).real, (y * x).real, rtol=RTOL, atol=ATOL_CANCEL)

    @given(
        x=octonions(min_value=-10.0, max_value=10.0),
        y=octonions(min_value=-10.0, max_value=10.0),
        z=octonions(min_value=-10.0, max_value=10.0),
    )
    @settings(max_examples=1000, deadline=None)
    def test_trace_associativity(self, x: Octonion, y: Octonion, z: Octonion) -> None:
        """Re((x*y)*z) = Re(x*(y*z)) — the trace form is associative."""
        torch.testing.assert_close(
            ((x * y) * z).real, (x * (y * z)).real, rtol=RTOL, atol=ATOL_CANCEL
        )


# =============================================================================
# 5. exp/log algebraic identities
# =============================================================================


class TestExpLogIdentities:
    """Algebraic identities of the octonionic exponential and logarithm."""

    @given(o=octonions(min_value=-3.0, max_value=3.0))
    @settings(max_examples=1000, deadline=None)
    def test_exp_norm(self, o: Octonion) -> None:
        """|exp(o)| = e^{Re(o)} — the norm of the exponential is the real exponential."""
        exp_o = octonion_exp(o)
        assert isinstance(exp_o, Octonion)
        torch.testing.assert_close(exp_o.norm(), torch.exp(o.real), rtol=RTOL, atol=ATOL)

    @given(o=octonions(min_value=-3.0, max_value=3.0))
    @settings(max_examples=1000, deadline=None)
    def test_exp_conjugation_equivariance(self, o: Octonion) -> None:
        """exp(o*) = exp(o)* — exp commutes with conjugation."""
        lhs = octonion_exp(o.conjugate())
        rhs_oct = octonion_exp(o)
        assert isinstance(lhs, Octonion)
        assert isinstance(rhs_oct, Octonion)
        rhs = rhs_oct.conjugate()
        torch.testing.assert_close(lhs.components, rhs.components, rtol=RTOL, atol=ATOL)

    @given(
        u=unit_pure_imaginary_octonions(),
        s=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        t=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=1000, deadline=None)
    def test_one_parameter_subgroup(self, u: Octonion, s: float, t: float) -> None:
        """exp((s+t)*u) = exp(s*u) * exp(t*u) for unit pure imaginary u.

        t -> exp(t*u) is a one-parameter subgroup (a circle in the plane
        spanned by 1 and u).
        """
        lhs = octonion_exp(u * (s + t))
        exp_s = octonion_exp(u * s)
        exp_t = octonion_exp(u * t)
        assert isinstance(lhs, Octonion)
        assert isinstance(exp_s, Octonion)
        assert isinstance(exp_t, Octonion)
        rhs = exp_s * exp_t
        torch.testing.assert_close(lhs.components, rhs.components, rtol=RTOL, atol=ATOL)

    @given(o=log_domain_octonions())
    @settings(max_examples=1000, deadline=None)
    def test_log_exp_roundtrip(self, o: Octonion) -> None:
        """log(exp(o)) = o for imaginary norm in (0.05, pi - 0.1) (principal branch)."""
        exp_o = octonion_exp(o)
        assert isinstance(exp_o, Octonion)
        back = octonion_log(exp_o)
        assert isinstance(back, Octonion)
        torch.testing.assert_close(back.components, o.components, rtol=RTOL, atol=ATOL)

    @given(o=exp_domain_octonions())
    @settings(max_examples=1000, deadline=None)
    def test_exp_log_roundtrip(self, o: Octonion) -> None:
        """exp(log(o)) = o for norm in (0.1, 10) and imaginary norm >= 0.05.

        The imaginary-norm floor keeps the input away from the negative real
        axis, where the principal branch of octonion_log is discontinuous.
        """
        log_o = octonion_log(o)
        assert isinstance(log_o, Octonion)
        back = octonion_exp(log_o)
        assert isinstance(back, Octonion)
        torch.testing.assert_close(back.components, o.components, rtol=RTOL, atol=ATOL)
