"""Cross-check tests: Fano plane multiplication vs Cayley-Dickson construction.

This implements success criterion 3 from the ROADMAP:
"Cayley-Dickson construction produces results identical to Fano-plane multiplication table"
"""

import torch
from tests.conftest import ATOL_FLOAT64, octonion_tensors
from hypothesis import given, settings

from octonion._cayley_dickson import cayley_dickson_mul, quaternion_conj, quaternion_mul
from octonion._multiplication import octonion_mul


def _basis(i: int) -> torch.Tensor:
    """Return the i-th basis octonion e_i as a tensor of shape [8]."""
    e = torch.zeros(8, dtype=torch.float64)
    e[i] = 1.0
    return e


class TestQuaternionMul:
    """Verify the internal quaternion multiplication matches Hamilton product."""

    def test_ij_equals_k(self) -> None:
        """i * j = k in Hamilton's quaternions."""
        i = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
        j = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
        k = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
        result = quaternion_mul(i, j)
        assert torch.allclose(result, k, atol=ATOL_FLOAT64)

    def test_ji_equals_minus_k(self) -> None:
        """j * i = -k in Hamilton's quaternions."""
        i = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
        j = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
        k = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
        result = quaternion_mul(j, i)
        assert torch.allclose(result, -k, atol=ATOL_FLOAT64)

    def test_ii_equals_minus_one(self) -> None:
        """i * i = -1 in Hamilton's quaternions."""
        i = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
        one = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = quaternion_mul(i, i)
        assert torch.allclose(result, -one, atol=ATOL_FLOAT64)

    def test_identity(self) -> None:
        """1 * q = q for any quaternion."""
        one = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        q = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        result = quaternion_mul(one, q)
        assert torch.allclose(result, q, atol=ATOL_FLOAT64)

    def test_jk_equals_i(self) -> None:
        """j * k = i in Hamilton's quaternions."""
        i = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
        j = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
        k = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
        result = quaternion_mul(j, k)
        assert torch.allclose(result, i, atol=ATOL_FLOAT64)

    def test_ki_equals_j(self) -> None:
        """k * i = j in Hamilton's quaternions."""
        i = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
        j = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
        k = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
        result = quaternion_mul(k, i)
        assert torch.allclose(result, j, atol=ATOL_FLOAT64)


class TestQuaternionConj:
    """Verify quaternion conjugation."""

    def test_conj_negates_imaginary(self) -> None:
        """Conjugation negates imaginary parts, preserves real part."""
        q = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        result = quaternion_conj(q)
        expected = torch.tensor([1.0, -2.0, -3.0, -4.0], dtype=torch.float64)
        assert torch.allclose(result, expected, atol=ATOL_FLOAT64)


class TestFanoCDCrosscheckBasis:
    """All 64 basis element products match between Fano and CD implementations.

    CRITICAL: This is the cross-check (FOUND-01 success criterion 3).
    If this test fails, the Cayley-Dickson formula convention does not match
    the Fano plane convention, and a permutation mapping must be determined.
    """

    def test_all_64_basis_products_match(self) -> None:
        """Fano plane mul and Cayley-Dickson mul produce identical results on all basis pairs."""
        mismatches = []
        for i in range(8):
            for j in range(8):
                ei = _basis(i)
                ej = _basis(j)
                fano_result = octonion_mul(ei, ej)
                cd_result = cayley_dickson_mul(ei, ej)
                if not torch.allclose(fano_result, cd_result, atol=ATOL_FLOAT64):
                    mismatches.append(
                        f"e_{i} * e_{j}: fano={fano_result.tolist()}, cd={cd_result.tolist()}"
                    )
        assert len(mismatches) == 0, (
            f"{len(mismatches)} basis element mismatches between Fano and CD:\n"
            + "\n".join(mismatches)
        )


class TestFanoCDCrosscheckRandom:
    """Random octonion pairs produce matching results (within tolerance)."""

    @given(a=octonion_tensors(min_value=-100, max_value=100),
           b=octonion_tensors(min_value=-100, max_value=100))
    @settings(max_examples=200)
    def test_fano_cd_random_match(self, a: torch.Tensor, b: torch.Tensor) -> None:
        """For random octonions, Fano and CD multiplication give the same result."""
        fano_result = octonion_mul(a, b)
        cd_result = cayley_dickson_mul(a, b)
        assert torch.allclose(fano_result, cd_result, atol=1e-6, rtol=1e-6), (
            f"Fano-CD mismatch: max error {(fano_result - cd_result).abs().max().item()}"
        )
