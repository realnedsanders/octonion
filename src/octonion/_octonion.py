"""Core Octonion class with operator overloading, subtypes, and associator.

Provides an immutable wrapper around a PyTorch tensor of shape [..., 8],
with full operator overloading for the octonionic algebra.

Convention: Baez 2002 mod-7 Fano plane. Components ordered [e0, e1, ..., e7]
where e0 is the real/scalar part.
"""

from __future__ import annotations

from typing import Tuple

import torch

from octonion._multiplication import octonion_mul
from octonion._types import NormedDivisionAlgebra

# Basis labels for symbolic display
_BASIS_LABELS = ["", "e1", "e2", "e3", "e4", "e5", "e6", "e7"]


class Octonion(NormedDivisionAlgebra):
    """Immutable octonion backed by a PyTorch tensor.

    All operations return new Octonion instances -- no in-place mutation.
    The underlying tensor is accessible via .components.

    Shape: The last dimension must be 8. Batch dimensions are fully supported:
    a tensor of shape [B, 8] represents a batch of B octonions.
    """

    __slots__ = ("_data",)

    def __init__(self, data: torch.Tensor) -> None:
        if isinstance(data, Octonion):
            data = data.components
        if data.shape[-1] != 8:
            raise ValueError(
                f"Octonion requires last dimension to be 8, got shape {data.shape}. "
                f"An octonion has 8 components: 1 real (e0) + 7 imaginary (e1..e7)."
            )
        # Store without copying -- caller is responsible for not mutating
        object.__setattr__(self, "_data", data)

    # --- Properties ---

    @property
    def components(self) -> torch.Tensor:
        """Raw tensor of shape [..., 8]."""
        return self._data

    @property
    def real(self) -> torch.Tensor:
        """Scalar (e0) component."""
        return self._data[..., 0]

    @property
    def imag(self) -> torch.Tensor:
        """Imaginary components e1..e7 as tensor of shape [..., 7]."""
        return self._data[..., 1:]

    @property
    def dim(self) -> int:
        """Algebraic dimension: 8 for octonions."""
        return 8

    # --- Indexing ---

    def __getitem__(self, i: int) -> torch.Tensor:
        """Return component e_i (i in 0..7)."""
        return self._data[..., i]

    # --- Arithmetic operators ---

    def __mul__(self, other: object) -> Octonion:
        """Multiply: Octonion * Octonion uses octonionic product; Octonion * scalar scales."""
        if isinstance(other, Octonion):
            return Octonion(octonion_mul(self._data, other._data))
        if isinstance(other, (int, float)):
            return Octonion(self._data * other)
        if isinstance(other, torch.Tensor) and other.dim() == 0:
            return Octonion(self._data * other)
        return NotImplemented

    def __rmul__(self, other: object) -> Octonion:
        """scalar * Octonion scales all components."""
        if isinstance(other, (int, float)):
            return Octonion(other * self._data)
        if isinstance(other, torch.Tensor) and other.dim() == 0:
            return Octonion(other * self._data)
        return NotImplemented

    def __add__(self, other: object) -> Octonion:
        """Add: Octonion + Octonion component-wise; Octonion + scalar adds to real part."""
        if isinstance(other, Octonion):
            return Octonion(self._data + other._data)
        if isinstance(other, (int, float)):
            result = self._data.clone()
            result[..., 0] = result[..., 0] + other
            return Octonion(result)
        return NotImplemented

    def __radd__(self, other: object) -> Octonion:
        """scalar + Octonion adds to real part."""
        return self.__add__(other)

    def __sub__(self, other: object) -> Octonion:
        """Subtract: Octonion - Octonion component-wise; Octonion - scalar subtracts from real."""
        if isinstance(other, Octonion):
            return Octonion(self._data - other._data)
        if isinstance(other, (int, float)):
            result = self._data.clone()
            result[..., 0] = result[..., 0] - other
            return Octonion(result)
        return NotImplemented

    def __rsub__(self, other: object) -> Octonion:
        """scalar - Octonion."""
        if isinstance(other, (int, float)):
            result = -self._data.clone()
            result[..., 0] = result[..., 0] + other
            return Octonion(result)
        return NotImplemented

    def __neg__(self) -> Octonion:
        """Negate all components."""
        return Octonion(-self._data)

    def __eq__(self, other: object) -> bool:
        """Element-wise equality of components."""
        if not isinstance(other, Octonion):
            return NotImplemented
        return bool(torch.equal(self._data, other._data))

    # --- NO __truediv__ or __pow__ (user decision: division ambiguity, parenthesization matters) ---

    # --- Algebraic methods ---

    def conjugate(self) -> Octonion:
        """Return the conjugate: negate imaginary parts, preserve real.

        For octonion x = x0 + x1*e1 + ... + x7*e7:
        conj(x) = x0 - x1*e1 - ... - x7*e7

        Property: x * conj(x) = |x|^2 * e0 (pure real)
        """
        return Octonion(
            torch.cat([self._data[..., :1], -self._data[..., 1:]], dim=-1)
        )

    def norm(self) -> torch.Tensor:
        """Return the norm: sqrt(sum of squared components).

        For octonion x: |x| = sqrt(x0^2 + x1^2 + ... + x7^2)
        Equivalently: |x| = sqrt(x * conj(x))

        Returns scalar tensor (batch dims preserved, last dim contracted).
        """
        return torch.sqrt(torch.sum(self._data**2, dim=-1))

    def inverse(self) -> Octonion:
        """Return the multiplicative inverse: conj(x) / |x|^2.

        For non-zero octonion x:
          x^{-1} = conj(x) / |x|^2

        satisfying x * x^{-1} = x^{-1} * x = 1 (the identity octonion).

        Raises:
            ValueError: If the octonion has zero norm (division by zero in
                the algebra). The zero octonion [0,0,0,0,0,0,0,0] has no
                multiplicative inverse, analogous to division by zero in R.
        """
        n_sq = torch.sum(self._data**2, dim=-1, keepdim=True)
        if torch.any(n_sq == 0):
            raise ValueError(
                "Cannot invert zero octonion: norm is 0.0. "
                "The zero octonion has no multiplicative inverse, "
                "analogous to division by zero in the real numbers. "
                "Use a non-zero octonion or check norm before inverting."
            )
        conj = self.conjugate()
        return Octonion(conj._data / n_sq)

    # --- Quaternion pair conversion ---

    @classmethod
    def from_quaternion_pair(cls, q1: torch.Tensor, q2: torch.Tensor) -> Octonion:
        """Create an Octonion from two quaternion tensors.

        The octonion is formed as x = (q1, q2) where q1 is the "real quaternion"
        and q2 is the "imaginary quaternion" in the Cayley-Dickson sense.

        Note: This uses the RAW Cayley-Dickson basis ordering, not the Fano plane
        convention. The components are simply concatenated as [q1, q2].
        Use cayley_dickson_mul for multiplication that accounts for basis permutation.

        Args:
            q1: Quaternion tensor of shape [..., 4].
            q2: Quaternion tensor of shape [..., 4].

        Returns:
            Octonion with components [q1[0], q1[1], q1[2], q1[3], q2[0], q2[1], q2[2], q2[3]].
        """
        data = torch.cat([q1, q2], dim=-1)
        return cls(data)

    def to_quaternion_pair(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split this octonion into two quaternion tensors.

        Returns (q1, q2) where q1 = self.components[..., :4] and
        q2 = self.components[..., 4:].

        This is the inverse of from_quaternion_pair.
        """
        return self._data[..., :4], self._data[..., 4:]

    # --- String representations ---

    def __repr__(self) -> str:
        """Tensor-form representation for debugging."""
        return f"Octonion({self._data})"

    def __str__(self) -> str:
        """Symbolic form: e.g. '1.0 + 2.0*e1 + 3.0*e2 + ...'."""
        if self._data.dim() > 1:
            # Batched: show shape summary like PyTorch tensors
            return f"Octonion(shape={list(self._data.shape[:-1])}, dtype={self._data.dtype})"

        # Dtype-aware display threshold: suppress near-zero noise
        # float32 eps ~1.2e-7, float64 eps ~2.2e-16
        atol = 1e-7 if self._data.dtype == torch.float32 else 1e-14

        parts = []
        for i in range(8):
            val = self._data[i].item()
            if i == 0:
                parts.append(f"{val}")
            elif abs(val) > atol:
                sign = " + " if val > 0 else " - "
                parts.append(f"{sign}{abs(val)}*{_BASIS_LABELS[i]}")
        return "".join(parts) if parts else "0.0"

    # --- Immutability enforcement ---
    # __slots__ prevents adding new attributes.
    # No __iadd__, __imul__, __isub__ etc. defined.
    # All operators return new instances.

    def __hash__(self) -> int:
        """Not hashable since equality is value-based on mutable tensors."""
        raise TypeError("unhashable type: 'Octonion'")


class UnitOctonion(Octonion):
    """Octonion constrained to have unit norm (lives on S^7).

    Normalizes input data at construction time.
    """

    def __init__(self, data: torch.Tensor) -> None:
        if data.shape[-1] != 8:
            raise ValueError(
                f"UnitOctonion requires last dimension to be 8, got shape {data.shape}."
            )
        # Normalize to unit norm
        n = torch.sqrt(torch.sum(data**2, dim=-1, keepdim=True))
        if torch.any(n == 0):
            raise ValueError(
                "Cannot create UnitOctonion from zero vector: "
                "normalization requires non-zero norm."
            )
        normalized = data / n
        super().__init__(normalized)


class PureOctonion(Octonion):
    """Octonion with zero real part (pure imaginary).

    Enforces real component = 0 at construction time.
    """

    def __init__(self, data: torch.Tensor) -> None:
        if data.shape[-1] != 8:
            raise ValueError(
                f"PureOctonion requires last dimension to be 8, got shape {data.shape}."
            )
        # Force real part to zero
        pure_data = data.clone()
        pure_data[..., 0] = 0.0
        super().__init__(pure_data)


def associator(a: Octonion, b: Octonion, c: Octonion) -> Octonion:
    """Compute the associator [a, b, c] = (a*b)*c - a*(b*c).

    The associator measures the failure of associativity. For octonions:
    - Totally antisymmetric: [a,b,c] = -[b,a,c] = -[a,c,b] = -[c,b,a]
    - Zero when any two arguments are equal (alternativity)
    - Non-zero for generic triples (octonions are NOT associative)

    Args:
        a: Octonion instance.
        b: Octonion instance.
        c: Octonion instance.

    Returns:
        Octonion representing (a*b)*c - a*(b*c).
    """
    left = (a * b) * c  # Left-associated product
    right = a * (b * c)  # Right-associated product
    return Octonion(left.components - right.components)
