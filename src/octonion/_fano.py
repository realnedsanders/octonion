"""Fano plane data structure encoding octonionic multiplication structure.

The Fano plane PG(2,2) has 7 points and 7 lines, where each line passes
through exactly 3 points. When oriented, it encodes the multiplication
rules for the 7 imaginary octonion units.

Convention: Baez 2002, mod-7. The 7 oriented triples are:
  (1,2,4), (2,3,5), (3,4,6), (4,5,7), (5,6,1), (6,7,2), (7,1,3)

For each triple (i,j,k): e_i * e_j = e_k, with cyclic permutations
positive and anti-cyclic permutations negative.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FanoPlane:
    """The Fano plane PG(2,2) encoding octonionic multiplication structure.

    The 7 lines correspond to the 7 oriented triples (i, j, k) where
    e_i * e_j = e_k (with cyclic permutations positive, anti-cyclic negative).

    Convention: Baez 2002, e_i * e_{i+1 mod 7} = e_{i+3 mod 7} (with 1-indexed units)
    Equivalently: triples (1,2,4), (2,3,5), (3,4,6), (4,5,7), (5,6,1), (6,7,2), (7,1,3)
    """

    # The 7 oriented triples (using 1-indexed imaginary unit labels)
    triples: tuple[tuple[int, int, int], ...] = (
        (1, 2, 4),
        (2, 3, 5),
        (3, 4, 6),
        (4, 5, 7),
        (5, 6, 1),
        (6, 7, 2),
        (7, 1, 3),
    )

    @property
    def lines(self) -> list[frozenset[int]]:
        """The 7 unoriented lines (sets of 3 points)."""
        return [frozenset(t) for t in self.triples]

    @property
    def incidence_matrix(self) -> torch.Tensor:
        """7x7 incidence matrix: M[point-1][line] = 1 if point is on line.

        Rows are points (1-indexed, so row 0 = point 1), columns are lines.
        """
        M = torch.zeros(7, 7, dtype=torch.int64)
        for line_idx, (i, j, k) in enumerate(self.triples):
            M[i - 1, line_idx] = 1
            M[j - 1, line_idx] = 1
            M[k - 1, line_idx] = 1
        return M

    def quaternionic_subalgebra(self, line_index: int) -> tuple[int, int, int]:
        """Return the triple of imaginary unit indices forming the line_index-th subalgebra.

        Each line of the Fano plane defines a quaternionic subalgebra
        {e_0, e_i, e_j, e_k} isomorphic to H.

        Args:
            line_index: Index into self.triples (0-6).

        Returns:
            Oriented triple (i, j, k) where e_i * e_j = e_k.
        """
        return self.triples[line_index]

    @property
    def automorphism_generators(self) -> list[dict[int, int]]:
        """Generators of GL(3, F_2), the symmetry group of the Fano plane (order 168).

        These are permutations of {1,...,7} that preserve the incidence structure.

        Returns:
            List of two permutation dicts:
            - cycle_7: i -> (i mod 7) + 1 (order 7 cyclic permutation)
            - quad_res: i -> (2*i - 1) % 7 + 1 (order 3 quadratic residue map)
        """
        # The cyclic permutation i -> i+1 mod 7 (order 7)
        cycle_7 = {i: (i % 7) + 1 for i in range(1, 8)}
        # The quadratic residue map i -> 2i mod 7 (order 3)
        quad_res = {i: (2 * i - 1) % 7 + 1 for i in range(1, 8)}
        return [cycle_7, quad_res]


# Module-level singleton
FANO_PLANE = FanoPlane()
