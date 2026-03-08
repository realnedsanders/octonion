"""Shared test fixtures, Hypothesis strategies, and tolerance constants.

Strategies produce raw torch.Tensor of shape [8] at float64 precision.
The Octonion class wrapper is introduced in Plan 02.
"""

import hypothesis.strategies as st
from hypothesis import settings
import torch

# --- Tolerance constants ---

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


# --- Hypothesis strategies ---


@st.composite
def octonions(
    draw: st.DrawFn,
    *,
    dtype: torch.dtype = torch.float64,
    min_value: float = -1e6,
    max_value: float = 1e6,
) -> torch.Tensor:
    """Strategy generating random octonion tensors of shape [8].

    Produces raw torch.Tensor (not Octonion class) at the specified dtype.
    Components are drawn independently from the given range.
    """
    elements = st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )
    components = [draw(elements) for _ in range(8)]
    return torch.tensor(components, dtype=dtype)


@st.composite
def unit_octonions(
    draw: st.DrawFn,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Strategy generating unit-norm octonion tensors on S^7.

    Produces raw torch.Tensor of shape [8] with norm 1.
    """
    elements = st.floats(
        min_value=-10.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
    )
    components = [draw(elements) for _ in range(8)]
    t = torch.tensor(components, dtype=dtype)
    n = torch.sqrt(torch.sum(t**2))

    # Ensure non-zero before normalizing
    while n < 1e-10:
        components = [draw(elements) for _ in range(8)]
        t = torch.tensor(components, dtype=dtype)
        n = torch.sqrt(torch.sum(t**2))

    return t / n


@st.composite
def nonzero_octonions(
    draw: st.DrawFn,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Strategy generating non-zero octonion tensors (for inverse testing).

    Produces raw torch.Tensor of shape [8] guaranteed to have non-zero norm.
    """
    elements = st.floats(
        min_value=-1e3,
        max_value=1e3,
        allow_nan=False,
        allow_infinity=False,
    )
    components = [draw(elements) for _ in range(8)]
    t = torch.tensor(components, dtype=dtype)
    n = torch.sqrt(torch.sum(t**2))

    # Ensure non-zero by adding a real part if needed
    if n < 1e-10:
        t[0] = 1.0

    return t
