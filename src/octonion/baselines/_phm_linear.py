"""PHM-8 (Parameterized Hypercomplex Multiplication) linear layer.

Implements the n=8 case of the PHM formulation from Zhang et al. (ICLR 2021):
  H = sum_{i=0}^{n-1} kron(A_i, S_i)

where A_i are learned (n x n) mixing matrices and S_i are learned
(out_features x in_features) sub-matrices.

This baseline isolates octonionic algebra structure from generic Kronecker
factorization: PHM-8 has the SAME Kronecker decomposition rank (n=8) as
the octonionic layer but learns its mixing matrices freely instead of
fixing them to octonionic structure constants.

Parameter count (no bias): n*n*n + n * out_f * in_f = 512 + 8 * out_f * in_f
Parameter count (with bias): 512 + 8 * out_f * in_f + out_f * 8
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PHM8Linear(nn.Module):
    """Parameterized Hypercomplex Multiplication layer with n=8.

    Forward: flatten x from [..., in_f, 8] to [..., in_f*8],
    compute H = sum_i kron(A_i, S_i), apply F.linear(x_flat, H),
    reshape to [..., out_f, 8].

    Input shape: [..., in_features, 8]
    Output shape: [..., out_features, 8]

    Args:
        in_features: Number of hypercomplex input features.
        out_features: Number of hypercomplex output features.
        bias: If True, adds a learnable bias. Default: True.
        dtype: Tensor dtype. Default: float32.
    """

    N: int = 8  # PHM rank (matches octonionic dimension)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        n = self.N

        # A_i: n learned mixing matrices of shape (n, n) each
        # Stored as single tensor [n, n, n] for clean parameter counting
        self.A = nn.Parameter(torch.randn(n, n, n, dtype=dtype) * 0.1)

        # S_i: n learned sub-matrices of shape (out_f, in_f) each
        # Stored as single tensor [n, out_f, in_f]
        self.S = nn.Parameter(torch.empty(n, out_features, in_features, dtype=dtype))
        nn.init.xavier_normal_(self.S.view(n * out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features * n, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: PHM-8 linear transformation.

        Args:
            x: Input tensor of shape [..., in_features, 8].

        Returns:
            Output tensor of shape [..., out_features, 8].
        """
        n = self.N
        batch_shape = x.shape[:-2]

        # Flatten input: [..., in_f, 8] -> [..., in_f*8]
        x_flat = x.reshape(*batch_shape, self.in_features * n)

        # Compute H = sum_i kron(A_i, S_i)
        # A[i]: (n, n), S[i]: (out_f, in_f)
        # kron(A[i], S[i]): (n*out_f, n*in_f)
        # Sum over i: (n*out_f, n*in_f)
        #
        # Using einsum for efficient batched Kronecker:
        # H[a, b, c, d] = sum_i A[i, a, c] * S[i, b, d]
        # where a,c index n-dims and b,d index feature-dims
        # Reshape to (n*out_f, n*in_f): row = a*out_f + b, col = c*in_f + d
        H = torch.einsum("iac, ibd -> abcd", self.A, self.S)
        H = H.reshape(n * self.out_features, n * self.in_features)

        # Apply linear: [..., in_f*8] x [out_f*8, in_f*8]^T -> [..., out_f*8]
        out_flat = F.linear(x_flat, H, self.bias)

        # Reshape: [..., out_f*8] -> [..., out_f, 8]
        return out_flat.reshape(*batch_shape, self.out_features, n)
