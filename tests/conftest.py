"""Shared test fixtures, Hypothesis strategies, and tolerance constants."""

import hypothesis.strategies as st
import torch
from hypothesis import assume, settings

# --- Tolerance constants ---
# float64 tolerance for numerical comparison in property-based tests
RTOL_FLOAT64 = 1e-12
ATOL_FLOAT64 = 1e-12

# --- Hypothesis profiles ---
settings.register_profile(
    "ci",
    max_examples=10000,
    deadline=None,
)
settings.register_profile(
    "dev",
    max_examples=200,
    deadline=5000,
)
settings.load_profile("dev")


# =============================================================================
# Raw tensor strategies (from Plan 01, kept for backward compatibility)
# =============================================================================


@st.composite
def octonion_tensors(
    draw: st.DrawFn,
    *,
    dtype: torch.dtype = torch.float64,
    min_value: float = -1e6,
    max_value: float = 1e6,
) -> torch.Tensor:
    """Strategy generating random octonion tensors of shape [8]."""
    elements = st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )
    components = [draw(elements) for _ in range(8)]
    return torch.tensor(components, dtype=dtype)


@st.composite
def unit_octonion_tensors(
    draw: st.DrawFn,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Strategy generating unit-norm octonion tensors on S^7."""
    elements = st.floats(
        min_value=-10.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
    )
    components = [draw(elements) for _ in range(8)]
    t = torch.tensor(components, dtype=dtype)
    n = torch.linalg.norm(t)
    assume(n > 1e-10)
    return t / n


@st.composite
def nonzero_octonion_tensors(
    draw: st.DrawFn,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Strategy generating non-zero octonion tensors (for inverse testing)."""
    elements = st.floats(
        min_value=-1e3,
        max_value=1e3,
        allow_nan=False,
        allow_infinity=False,
    )
    components = [draw(elements) for _ in range(8)]
    t = torch.tensor(components, dtype=dtype)
    n = torch.linalg.norm(t)
    if n < 1e-10:
        t = t.clone()
        t[0] = 1.0
    return t


# =============================================================================
# Octonion class strategies (Plan 02+)
# =============================================================================

from octonion import Octonion  # noqa: E402
from octonion._fano import FANO_PLANE  # noqa: E402


@st.composite
def octonions(
    draw: st.DrawFn,
    *,
    dtype: torch.dtype = torch.float64,
    min_value: float = -1e6,
    max_value: float = 1e6,
) -> Octonion:
    """Strategy generating random Octonion instances."""
    t = draw(octonion_tensors(dtype=dtype, min_value=min_value, max_value=max_value))
    return Octonion(t)


@st.composite
def unit_octonions(
    draw: st.DrawFn,
    *,
    dtype: torch.dtype = torch.float64,
) -> Octonion:
    """Strategy generating unit-norm Octonion instances on S^7."""
    t = draw(unit_octonion_tensors(dtype=dtype))
    return Octonion(t)


@st.composite
def nonzero_octonions(
    draw: st.DrawFn,
    *,
    dtype: torch.dtype = torch.float64,
) -> Octonion:
    """Strategy generating non-zero Octonion instances (for inverse testing)."""
    t = draw(nonzero_octonion_tensors(dtype=dtype))
    return Octonion(t)


@st.composite
def subalgebra_octonions(
    draw: st.DrawFn,
    subalgebra_idx: int = 0,
    *,
    dtype: torch.dtype = torch.float64,
    min_value: float = -1.0,
    max_value: float = 1.0,
) -> Octonion:
    """Strategy generating Octonions restricted to a quaternionic subalgebra.

    The subalgebra is span{1, e_i, e_j, e_k} where (i,j,k) is the
    Fano triple at the given index. Components outside these 4 slots are zero.
    """
    triple = FANO_PLANE.triples[subalgebra_idx]
    elements = st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )
    data = torch.zeros(8, dtype=dtype)
    data[0] = draw(elements)  # real part
    for idx in triple:
        data[idx] = draw(elements)  # imaginary parts in subalgebra
    return Octonion(data)
