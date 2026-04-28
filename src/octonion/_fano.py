"""Fano plane data structure encoding octonionic multiplication structure.

The Fano plane PG(2,2) is the simplest finite projective plane, with 7 points
and 7 lines (each containing 3 points, each point on 3 lines). Its oriented
lines encode the multiplication table of the octonion imaginary units.

Convention: Baez 2002, mod-7. The triples are (1,2,4), (2,3,5), (3,4,6),
(4,5,7), (5,6,1), (6,7,2), (7,1,3). Each triple (i,j,k) means:
  e_i * e_j = e_k  (and cyclic permutations positive, anti-cyclic negative)
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
        """7x7 incidence matrix where entry (point, line_idx) is 1 if point is on that line.

        Points are 0-indexed (point p corresponds to imaginary unit e_{p+1}).
        Lines are indexed 0..6 matching the triple order.
        """
        M = torch.zeros(7, 7, dtype=torch.int32)
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
            line_index: Index 0..6 of the Fano plane line.

        Returns:
            Tuple of 3 imaginary unit indices (1-indexed) forming the subalgebra.
        """
        return self.triples[line_index]

    @property
    def automorphism_generators(self) -> list[dict[int, int]]:
        """Generators of GL(3, F_2), the symmetry group of the Fano plane (order 168).

        These are permutations of {1,...,7} that preserve the incidence structure.
        Returns two generators whose products generate the full group:
          - cycle_7: i -> i+1 mod 7 (with 1-indexing), order 7
          - quad_res: i -> 2i mod 7, order 3
        """
        # The cyclic permutation i -> (i mod 7) + 1 (order 7)
        cycle_7 = {i: (i % 7) + 1 for i in range(1, 8)}
        # The quadratic residue map i -> 2i mod 7 (order 3)
        quad_res = {i: (2 * i - 1) % 7 + 1 for i in range(1, 8)}
        return [cycle_7, quad_res]


# Module-level singleton
FANO_PLANE = FanoPlane()
