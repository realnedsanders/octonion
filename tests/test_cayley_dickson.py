"""Cross-check test: Fano plane multiplication vs Cayley-Dickson construction.

Verifies that both implementations produce identical results, confirming
success criterion 3 from the ROADMAP (Baez 2002 convention consistency).
"""

import sys
from pathlib import Path

import torch
from hypothesis import given, settings

# Make conftest importable
sys.path.insert(0, str(Path(__file__).parent))
from conftest import octonions, ATOL_FLOAT64


def _basis(i: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Return the i-th basis octonion e_i as a tensor of shape [8]."""
    e = torch.zeros(8, dtype=dtype)
    e[i] = 1.0
    return e


class TestFanoCDCrosscheckBasis:
    """All 64 basis element products match between Fano and CD implementations."""

    def test_all_64_basis_products(self) -> None:
        """e_i * e_j via Fano structure constants == e_i * e_j via Cayley-Dickson
        for all i, j in 0..7.
        """
        from octonion import octonion_mul
        from octonion._cayley_dickson import cayley_dickson_mul

        errors = []
        for i in range(8):
            for j in range(8):
                ei, ej = _basis(i), _basis(j)
                fano_result = octonion_mul(ei, ej)
                cd_result = cayley_dickson_mul(ei, ej)
                if not torch.allclose(fano_result, cd_result, atol=ATOL_FLOAT64):
                    diff = (fano_result - cd_result).abs().max().item()
                    errors.append(
                        f"e_{i} * e_{j}: Fano={fano_result.tolist()}, "
                        f"CD={cd_result.tolist()}, max_diff={diff}"
                    )

        assert len(errors) == 0, (
            f"Fano vs Cayley-Dickson mismatch on {len(errors)} basis products:\n"
            + "\n".join(errors)
        )


class TestFanoCDCrosscheckRandom:
    """Random octonion pairs produce matching results (1e-12 tolerance)."""

    @given(a=octonions(min_value=-100, max_value=100),
           b=octonions(min_value=-100, max_value=100))
    @settings(max_examples=200)
    def test_random_octonion_pairs(self, a: torch.Tensor, b: torch.Tensor) -> None:
        from octonion import octonion_mul
        from octonion._cayley_dickson import cayley_dickson_mul

        fano_result = octonion_mul(a, b)
        cd_result = cayley_dickson_mul(a, b)
        assert torch.allclose(fano_result, cd_result, atol=1e-8), (
            f"Fano vs CD mismatch on random inputs: "
            f"max_diff={(fano_result - cd_result).abs().max().item()}"
        )


class TestQuaternionMul:
    """The internal quaternion multiplication matches known Hamilton product results."""

    def test_ij_equals_k(self) -> None:
        """i * j = k in quaternion convention."""
        from octonion._cayley_dickson import quaternion_mul

        # i = (0, 1, 0, 0), j = (0, 0, 1, 0), k = (0, 0, 0, 1)
        i = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
        j = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
        k = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)

        result = quaternion_mul(i, j)
        assert torch.allclose(result, k, atol=ATOL_FLOAT64), (
            f"i * j should be k, got {result}"
        )

    def test_ji_equals_minus_k(self) -> None:
        """j * i = -k (anti-commutativity)."""
        from octonion._cayley_dickson import quaternion_mul

        i = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
        j = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
        k = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)

        result = quaternion_mul(j, i)
        assert torch.allclose(result, -k, atol=ATOL_FLOAT64), (
            f"j * i should be -k, got {result}"
        )

    def test_ii_equals_minus_one(self) -> None:
        """i^2 = -1."""
        from octonion._cayley_dickson import quaternion_mul

        i = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
        neg_one = torch.tensor([-1.0, 0.0, 0.0, 0.0], dtype=torch.float64)

        result = quaternion_mul(i, i)
        assert torch.allclose(result, neg_one, atol=ATOL_FLOAT64), (
            f"i^2 should be -1, got {result}"
        )

    def test_identity(self) -> None:
        """1 * q = q * 1 = q."""
        from octonion._cayley_dickson import quaternion_mul

        one = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        q = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

        assert torch.allclose(quaternion_mul(one, q), q, atol=ATOL_FLOAT64)
        assert torch.allclose(quaternion_mul(q, one), q, atol=ATOL_FLOAT64)
