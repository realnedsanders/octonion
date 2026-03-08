"""Abstract base class for the Cayley-Dickson tower: R, C, H, O.

All normed division algebras share: conjugation, norm, inverse, multiplication.
"""

from abc import ABC, abstractmethod

import torch


class NormedDivisionAlgebra(ABC):
    """Abstract base class for the Cayley-Dickson tower: R, C, H, O.

    All normed division algebras share: conjugation, norm, inverse, multiplication.
    The underlying representation is always a PyTorch tensor of shape [..., dim]
    where dim is 1 (R), 2 (C), 4 (H), or 8 (O).
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
        """Return the squared norm (avoids sqrt for precision)."""
        return torch.sum(self.components ** 2, dim=-1)

    @abstractmethod
    def inverse(self) -> "NormedDivisionAlgebra":
        """Return the multiplicative inverse: x^{-1} = conj(x) / |x|^2."""
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
