"""Real, Complex, and Quaternion types in the Cayley-Dickson tower.

Each type implements NormedDivisionAlgebra with algebra-specific multiplication:
- Real: dim=1, scalar multiplication
- Complex: dim=2, standard complex multiplication
- Quaternion: dim=4, Hamilton product

These form the tower R -> C -> H -> O, where each level is constructed
from pairs of the previous level via the Cayley-Dickson construction.
"""

from __future__ import annotations

import torch

from octonion._types import NormedDivisionAlgebra


class Real(NormedDivisionAlgebra):
    """Real number as a normed division algebra element.

    Wraps a tensor of shape [..., 1]. Multiplication is scalar multiplication.
    The only normed division algebra that is both commutative and associative.
    """

    __slots__ = ("_data",)

    _data: torch.Tensor

    def __init__(self, data: torch.Tensor) -> None:
        if data.shape[-1] != 1:
            raise ValueError(
                f"Real requires last dimension to be 1, got shape {data.shape}. "
                f"A real number has 1 component."
            )
        object.__setattr__(self, "_data", data)

    @property
    def components(self) -> torch.Tensor:
        return self._data

    @property
    def dim(self) -> int:
        return 1

    def conjugate(self) -> Real:
        """Real conjugation is identity (no imaginary part)."""
        return Real(self._data.clone())

    def norm(self) -> torch.Tensor:
        """Absolute value."""
        return torch.abs(self._data[..., 0])

    def inverse(self) -> Real:
        """Multiplicative inverse: 1/x."""
        if torch.any(self._data == 0):
            raise ValueError(
                "Cannot invert zero real number: division by zero. "
                "The real number 0 has no multiplicative inverse."
            )
        return Real(1.0 / self._data)

    def __mul__(self, other: object) -> Real:
        if isinstance(other, Real):
            return Real(self._data * other._data)
        if isinstance(other, (int, float)):
            return Real(self._data * other)
        return NotImplemented

    def __rmul__(self, other: object) -> Real:
        if isinstance(other, (int, float)):
            return Real(other * self._data)
        return NotImplemented

    def __add__(self, other: object) -> Real:
        if isinstance(other, Real):
            return Real(self._data + other._data)
        if isinstance(other, (int, float)):
            return Real(self._data + other)
        return NotImplemented

    def __radd__(self, other: object) -> Real:
        return self.__add__(other)

    def __sub__(self, other: object) -> Real:
        if isinstance(other, Real):
            return Real(self._data - other._data)
        if isinstance(other, (int, float)):
            return Real(self._data - other)
        return NotImplemented

    def __rsub__(self, other: object) -> Real:
        if isinstance(other, (int, float)):
            return Real(other - self._data)
        return NotImplemented

    def __neg__(self) -> Real:
        return Real(-self._data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Real):
            return NotImplemented
        return bool(torch.equal(self._data, other._data))

    def __repr__(self) -> str:
        return f"Real({self._data})"


class Complex(NormedDivisionAlgebra):
    """Complex number as a normed division algebra element.

    Wraps a tensor of shape [..., 2] where component 0 is real, component 1 is imaginary.
    Multiplication is standard complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i.
    Commutative and associative, but not ordered.
    """

    __slots__ = ("_data",)

    _data: torch.Tensor

    def __init__(self, data: torch.Tensor) -> None:
        if data.shape[-1] != 2:
            raise ValueError(
                f"Complex requires last dimension to be 2, got shape {data.shape}. "
                f"A complex number has 2 components: real + imaginary."
            )
        object.__setattr__(self, "_data", data)

    @property
    def components(self) -> torch.Tensor:
        return self._data

    @property
    def dim(self) -> int:
        return 2

    def conjugate(self) -> Complex:
        """Negate imaginary part: conj(a+bi) = a-bi."""
        return Complex(
            torch.cat([self._data[..., :1], -self._data[..., 1:]], dim=-1)
        )

    def norm(self) -> torch.Tensor:
        """Modulus: |a+bi| = sqrt(a^2 + b^2)."""
        return torch.sqrt(torch.sum(self._data**2, dim=-1))

    def inverse(self) -> Complex:
        """Inverse: conj(z) / |z|^2."""
        n_sq = torch.sum(self._data**2, dim=-1, keepdim=True)
        if torch.any(n_sq == 0):
            raise ValueError(
                "Cannot invert zero complex number: norm is 0.0. "
                "The complex number 0+0i has no multiplicative inverse."
            )
        conj = self.conjugate()
        return Complex(conj._data / n_sq)

    def __mul__(self, other: object) -> Complex:
        """Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i."""
        if isinstance(other, Complex):
            a, b = self._data[..., 0], self._data[..., 1]
            c, d = other._data[..., 0], other._data[..., 1]
            return Complex(
                torch.stack([a * c - b * d, a * d + b * c], dim=-1)
            )
        if isinstance(other, (int, float)):
            return Complex(self._data * other)
        return NotImplemented

    def __rmul__(self, other: object) -> Complex:
        if isinstance(other, (int, float)):
            return Complex(other * self._data)
        return NotImplemented

    def __add__(self, other: object) -> Complex:
        if isinstance(other, Complex):
            return Complex(self._data + other._data)
        if isinstance(other, (int, float)):
            result = self._data.clone()
            result[..., 0] = result[..., 0] + other
            return Complex(result)
        return NotImplemented

    def __radd__(self, other: object) -> Complex:
        return self.__add__(other)

    def __sub__(self, other: object) -> Complex:
        if isinstance(other, Complex):
            return Complex(self._data - other._data)
        if isinstance(other, (int, float)):
            result = self._data.clone()
            result[..., 0] = result[..., 0] - other
            return Complex(result)
        return NotImplemented

    def __rsub__(self, other: object) -> Complex:
        if isinstance(other, (int, float)):
            result = -self._data.clone()
            result[..., 0] = result[..., 0] + other
            return Complex(result)
        return NotImplemented

    def __neg__(self) -> Complex:
        return Complex(-self._data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Complex):
            return NotImplemented
        return bool(torch.equal(self._data, other._data))

    def __repr__(self) -> str:
        return f"Complex({self._data})"


class Quaternion(NormedDivisionAlgebra):
    """Quaternion as a normed division algebra element.

    Wraps a tensor of shape [..., 4] where components are [q0, q1, q2, q3]
    representing q0 + q1*i + q2*j + q3*k.

    Multiplication is the Hamilton product. Non-commutative but associative.
    """

    __slots__ = ("_data",)

    _data: torch.Tensor

    def __init__(self, data: torch.Tensor) -> None:
        if data.shape[-1] != 4:
            raise ValueError(
                f"Quaternion requires last dimension to be 4, got shape {data.shape}. "
                f"A quaternion has 4 components: q0 + q1*i + q2*j + q3*k."
            )
        object.__setattr__(self, "_data", data)

    @property
    def components(self) -> torch.Tensor:
        return self._data

    @property
    def dim(self) -> int:
        return 4

    def conjugate(self) -> Quaternion:
        """Negate imaginary parts: conj(q0+q1i+q2j+q3k) = q0-q1i-q2j-q3k."""
        return Quaternion(
            torch.cat([self._data[..., :1], -self._data[..., 1:]], dim=-1)
        )

    def norm(self) -> torch.Tensor:
        """Quaternion norm: sqrt(q0^2 + q1^2 + q2^2 + q3^2)."""
        return torch.sqrt(torch.sum(self._data**2, dim=-1))

    def inverse(self) -> Quaternion:
        """Inverse: conj(q) / |q|^2."""
        n_sq = torch.sum(self._data**2, dim=-1, keepdim=True)
        if torch.any(n_sq == 0):
            raise ValueError(
                "Cannot invert zero quaternion: norm is 0.0. "
                "The zero quaternion has no multiplicative inverse."
            )
        conj = self.conjugate()
        return Quaternion(conj._data / n_sq)

    def __mul__(self, other: object) -> Quaternion:
        """Hamilton product: non-commutative, associative quaternion multiplication.

        Product rules: i^2 = j^2 = k^2 = ijk = -1
        ij = k, jk = i, ki = j
        ji = -k, kj = -i, ik = -j
        """
        if isinstance(other, Quaternion):
            p0, p1, p2, p3 = (
                self._data[..., 0],
                self._data[..., 1],
                self._data[..., 2],
                self._data[..., 3],
            )
            q0, q1, q2, q3 = (
                other._data[..., 0],
                other._data[..., 1],
                other._data[..., 2],
                other._data[..., 3],
            )
            return Quaternion(
                torch.stack(
                    [
                        p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3,
                        p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2,
                        p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1,
                        p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0,
                    ],
                    dim=-1,
                )
            )
        if isinstance(other, (int, float)):
            return Quaternion(self._data * other)
        return NotImplemented

    def __rmul__(self, other: object) -> Quaternion:
        if isinstance(other, (int, float)):
            return Quaternion(other * self._data)
        return NotImplemented

    def __add__(self, other: object) -> Quaternion:
        if isinstance(other, Quaternion):
            return Quaternion(self._data + other._data)
        if isinstance(other, (int, float)):
            result = self._data.clone()
            result[..., 0] = result[..., 0] + other
            return Quaternion(result)
        return NotImplemented

    def __radd__(self, other: object) -> Quaternion:
        return self.__add__(other)

    def __sub__(self, other: object) -> Quaternion:
        if isinstance(other, Quaternion):
            return Quaternion(self._data - other._data)
        if isinstance(other, (int, float)):
            result = self._data.clone()
            result[..., 0] = result[..., 0] - other
            return Quaternion(result)
        return NotImplemented

    def __rsub__(self, other: object) -> Quaternion:
        if isinstance(other, (int, float)):
            result = -self._data.clone()
            result[..., 0] = result[..., 0] + other
            return Quaternion(result)
        return NotImplemented

    def __neg__(self) -> Quaternion:
        return Quaternion(-self._data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quaternion):
            return NotImplemented
        return bool(torch.equal(self._data, other._data))

    def __repr__(self) -> str:
        return f"Quaternion({self._data})"
