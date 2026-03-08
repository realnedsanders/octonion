"""Abstract base class for normed division algebras (R, C, H, O)."""

from abc import ABC, abstractmethod

import torch


class NormedDivisionAlgebra(ABC):
    """Abstract base class for the Cayley-Dickson tower: R, C, H, O.

    All normed division algebras share: conjugation, norm, inverse, multiplication.
    Concrete implementations provide algebra-specific multiplication rules.
    """

    @abstractmethod
    def conjugate(self) -> "NormedDivisionAlgebra":
        """Return the conjugate (negate all imaginary components)."""
        ...

    @abstractmethod
    def norm(self) -> torch.Tensor:
        """Return the norm (sqrt of sum of squared components)."""
        ...

    def norm_squared(self) -> torch.Tensor:
        """Return the squared norm (avoids sqrt for precision).

        For all normed division algebras: |x|^2 = sum of squared components.
        """
        return torch.sum(self.components ** 2, dim=-1)

    @abstractmethod
    def inverse(self) -> "NormedDivisionAlgebra":
        """Return the multiplicative inverse: x^{-1} = conj(x) / |x|^2.

        Raises ValueError if the element has zero norm.
        """
        ...

    @abstractmethod
    def __mul__(self, other: object) -> "NormedDivisionAlgebra":
        """Algebra-specific multiplication."""
        ...

    @property
    @abstractmethod
    def components(self) -> torch.Tensor:
        """Raw tensor of shape [..., dim] where dim is 1/2/4/8."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Algebraic dimension: 1 for R, 2 for C, 4 for H, 8 for O."""
        ...
