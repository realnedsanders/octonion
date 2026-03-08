"""Tests for batched operations and broadcasting with [..., 8] shaped tensors.

Verifies that all octonion operations support arbitrary batch dimensions
and follow standard PyTorch broadcasting rules.
"""

import torch

from octonion import Octonion
from octonion._linear import OctonionLinear
from octonion._linear_algebra import left_mul_matrix
from octonion._multiplication import octonion_mul
from octonion._operations import inner_product, octonion_exp, octonion_log


class TestBatchMultiplication:
    """Tests for batched octonion multiplication."""

    def test_mul_batch_1d(self) -> None:
        """Multiplication works with [N, 8] batches, producing [N, 8]."""
        N = 10
        a = torch.randn(N, 8, dtype=torch.float64)
        b = torch.randn(N, 8, dtype=torch.float64)
        result = octonion_mul(a, b)
        assert result.shape == (N, 8), f"Expected ({N}, 8), got {result.shape}"

    def test_mul_batch_2d(self) -> None:
        """Multiplication works with [N, M, 8] batches, producing [N, M, 8]."""
        N, M = 5, 7
        a = torch.randn(N, M, 8, dtype=torch.float64)
        b = torch.randn(N, M, 8, dtype=torch.float64)
        result = octonion_mul(a, b)
        assert result.shape == (N, M, 8), f"Expected ({N}, {M}, 8), got {result.shape}"

    def test_mul_broadcast_expand(self) -> None:
        """Broadcasting: [N, 1, 8] * [1, M, 8] produces [N, M, 8]."""
        N, M = 4, 6
        a = torch.randn(N, 1, 8, dtype=torch.float64)
        b = torch.randn(1, M, 8, dtype=torch.float64)
        result = octonion_mul(a, b)
        assert result.shape == (N, M, 8), f"Expected ({N}, {M}, 8), got {result.shape}"

    def test_mul_broadcast_scalar_batch(self) -> None:
        """Broadcasting: scalar [8] * batch [N, 8] produces [N, 8]."""
        N = 10
        a = torch.randn(8, dtype=torch.float64)
        b = torch.randn(N, 8, dtype=torch.float64)
        result = octonion_mul(a, b)
        assert result.shape == (N, 8), f"Expected ({N}, 8), got {result.shape}"


class TestBatchOctonionOperations:
    """Tests for batched Octonion class operations."""

    def test_conjugate_batch(self) -> None:
        """Conjugate works on [N, 8] batches."""
        N = 10
        o = Octonion(torch.randn(N, 8, dtype=torch.float64))
        conj = o.conjugate()
        assert conj.components.shape == (N, 8)
        # Real part preserved, imaginary parts negated
        assert torch.allclose(conj.real, o.real)
        assert torch.allclose(conj.imag, -o.imag)

    def test_norm_batch(self) -> None:
        """Norm returns [N] for [N, 8] input."""
        N = 10
        o = Octonion(torch.randn(N, 8, dtype=torch.float64))
        n = o.norm()
        assert n.shape == (N,), f"Expected ({N},), got {n.shape}"
        assert torch.all(n >= 0)

    def test_inverse_batch(self) -> None:
        """Inverse works on [N, 8] batches."""
        N = 10
        # Use non-zero octonions
        data = torch.randn(N, 8, dtype=torch.float64) + 0.5
        o = Octonion(data)
        inv = o.inverse()
        assert inv.components.shape == (N, 8)
        # Verify a * a_inv ~ identity for each element
        identity = torch.zeros(N, 8, dtype=torch.float64)
        identity[:, 0] = 1.0
        product = (o * inv).components
        assert torch.allclose(product, identity, atol=1e-10)

    def test_exp_batch(self) -> None:
        """Exp works on [N, 8] batches."""
        N = 10
        data = torch.randn(N, 8, dtype=torch.float64) * 0.5
        o = Octonion(data)
        result = octonion_exp(o)
        assert result.components.shape == (N, 8)
        # Check no NaN or Inf
        assert not torch.any(torch.isnan(result.components))
        assert not torch.any(torch.isinf(result.components))

    def test_log_batch(self) -> None:
        """Log works on [N, 8] batches."""
        N = 10
        # Use octonions with positive norm (avoid zero)
        data = torch.randn(N, 8, dtype=torch.float64) + 0.5
        o = Octonion(data)
        result = octonion_log(o)
        assert result.components.shape == (N, 8)
        assert not torch.any(torch.isnan(result.components))
        assert not torch.any(torch.isinf(result.components))

    def test_inner_product_batch(self) -> None:
        """Inner product works with [N, 8] batched inputs."""
        N = 10
        a = Octonion(torch.randn(N, 8, dtype=torch.float64))
        b = Octonion(torch.randn(N, 8, dtype=torch.float64))
        result = inner_product(a, b)
        assert result.shape == (N,), f"Expected ({N},), got {result.shape}"

    def test_left_mul_matrix_batch(self) -> None:
        """Left mul matrix has shape [N, 8, 8] for [N, 8] input."""
        N = 10
        a = Octonion(torch.randn(N, 8, dtype=torch.float64))
        L = left_mul_matrix(a)
        assert L.shape == (N, 8, 8), f"Expected ({N}, 8, 8), got {L.shape}"

    def test_octonion_linear_batch(self) -> None:
        """OctonionLinear produces [N, 8] output for [N, 8] input."""
        N = 10
        layer = OctonionLinear(dtype=torch.float64)
        x = torch.randn(N, 8, dtype=torch.float64)
        y = layer(x)
        assert y.shape == (N, 8), f"Expected ({N}, 8), got {y.shape}"


class TestBatchConsistency:
    """Verify batch results match element-wise application."""

    def test_batch_consistency_mul(self) -> None:
        """Batch multiplication matches loop over individual elements."""
        N = 10
        torch.manual_seed(42)
        a = torch.randn(N, 8, dtype=torch.float64)
        b = torch.randn(N, 8, dtype=torch.float64)

        # Batch computation
        batch_result = octonion_mul(a, b)

        # Element-wise computation
        for i in range(N):
            single_result = octonion_mul(a[i], b[i])
            assert torch.allclose(batch_result[i], single_result, atol=1e-12), (
                f"Batch result[{i}] differs from single: "
                f"max diff = {(batch_result[i] - single_result).abs().max().item()}"
            )

    def test_batch_consistency_exp(self) -> None:
        """Batch exp matches loop over individual elements."""
        N = 10
        torch.manual_seed(42)
        data = torch.randn(N, 8, dtype=torch.float64) * 0.5
        batch_octonion = Octonion(data)

        # Batch computation
        batch_result = octonion_exp(batch_octonion)

        # Element-wise computation
        for i in range(N):
            single_result = octonion_exp(Octonion(data[i]))
            assert torch.allclose(
                batch_result.components[i], single_result.components, atol=1e-12
            )

    def test_batch_consistency_inner_product(self) -> None:
        """Batch inner product matches loop over individual elements."""
        N = 10
        torch.manual_seed(42)
        a_data = torch.randn(N, 8, dtype=torch.float64)
        b_data = torch.randn(N, 8, dtype=torch.float64)
        a = Octonion(a_data)
        b = Octonion(b_data)

        # Batch computation
        batch_result = inner_product(a, b)

        # Element-wise computation
        for i in range(N):
            single_result = inner_product(Octonion(a_data[i]), Octonion(b_data[i]))
            assert torch.allclose(batch_result[i], single_result, atol=1e-12)
