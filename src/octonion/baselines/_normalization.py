"""Algebra-aware batch normalization layers using covariance whitening.

Each layer normalizes algebra-valued features using the full covariance
structure of the algebra, NOT per-component normalization (which would
destroy algebraic structure).

- RealBatchNorm: wraps nn.BatchNorm1d (2 params/feature)
- ComplexBatchNorm: 2x2 covariance whitening (5 params/feature), Trabelsi et al.
- QuaternionBatchNorm: 4x4 Cholesky whitening (14 params/feature), Gaudet & Maida
- OctonionBatchNorm: 8x8 Cholesky whitening (44 params/feature)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _tril_to_symmetric(
    tril_flat: torch.Tensor, dim: int, rows: torch.Tensor, cols: torch.Tensor
) -> torch.Tensor:
    """Convert flat lower-triangular entries to a symmetric matrix.

    Accepts pre-computed rows/cols from torch.tril_indices to avoid
    calling tril_indices inside compiled regions (dynamic=True would
    try to symbolically trace the quadratic index expression, causing
    recursion in the shape solver).

    Args:
        tril_flat: [..., dim*(dim+1)/2] flat lower-triangular entries.
        dim: Matrix dimension.
        rows: Row indices from torch.tril_indices(dim, dim).
        cols: Col indices from torch.tril_indices(dim, dim).

    Returns:
        [..., dim, dim] symmetric matrix.
    """
    batch_shape = tril_flat.shape[:-1]
    mat = torch.zeros(
        *batch_shape, dim, dim,
        device=tril_flat.device, dtype=tril_flat.dtype,
    )
    mat[..., rows, cols] = tril_flat
    mat[..., cols, rows] = tril_flat
    return mat


class RealBatchNorm(nn.Module):
    """Real-valued batch normalization wrapping nn.BatchNorm1d.

    Input shape: [batch, features]
    Output shape: [batch, features]

    Args:
        num_features: Number of features.
        eps: Value added for numerical stability.
        momentum: Running stats momentum.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: standard real batch normalization."""
        return self.bn(x)


class ComplexBatchNorm(nn.Module):
    """Complex batch normalization using 2x2 covariance whitening.

    Following Trabelsi et al., "Deep Complex Networks" (ICLR 2018).
    Uses analytic 2x2 inverse square root for the covariance matrix.

    Input shape: [batch, features, 2]
    Output shape: [batch, features, 2]

    Learnable parameters per feature:
    - gamma_rr, gamma_ri, gamma_ii (3 scale params, symmetric 2x2)
    - beta_r, beta_i (2 shift params)
    Total: 5 params/feature

    Args:
        num_features: Number of complex features.
        eps: Value added for numerical stability.
        momentum: Running stats momentum.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 3 scale params per feature (symmetric 2x2 matrix entries)
        self.gamma_rr = nn.Parameter(torch.ones(num_features))
        self.gamma_ri = nn.Parameter(torch.zeros(num_features))
        self.gamma_ii = nn.Parameter(torch.ones(num_features))

        # 2 shift params per feature
        self.beta = nn.Parameter(torch.zeros(num_features, 2))

        # Running stats (not learnable)
        self.register_buffer("running_mean", torch.zeros(num_features, 2))
        self.register_buffer("running_var_rr", torch.ones(num_features))
        self.register_buffer("running_var_ri", torch.zeros(num_features))
        self.register_buffer("running_var_ii", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def _compute_inverse_sqrt_2x2(
        self,
        vrr: torch.Tensor,
        vri: torch.Tensor,
        vii: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute V^{-1/2} for symmetric positive definite 2x2 matrix.

        Uses Cayley-Hamilton theorem for 2x2 matrices:
        V^{1/2} = (V + s*I) / t, where s = sqrt(det(V)), t = sqrt(tr(V) + 2s)
        V^{-1/2} = (V^{1/2})^{-1} = t * (V + s*I)^{-1}

        Returns:
            (inv_rr, inv_ri, inv_ii) entries of V^{-1/2}.
        """
        det = vrr * vii - vri * vri
        det = torch.clamp(det, min=self.eps)
        trace = vrr + vii

        s = torch.sqrt(det)
        t = torch.sqrt(torch.clamp(trace + 2 * s, min=self.eps))

        # det(V + sI) = (vrr+s)(vii+s) - vri^2 = det + s*trace + s^2 = 2*det + s*trace
        det_VsI = 2 * det + s * trace
        det_VsI = torch.clamp(det_VsI, min=self.eps)

        # V^{-1/2} = t * (V + sI)^{-1}
        inv_rr = t * (vii + s) / det_VsI
        inv_ri = t * (-vri) / det_VsI
        inv_ii = t * (vrr + s) / det_VsI

        return inv_rr, inv_ri, inv_ii

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: complex batch normalization.

        Always computes in float32 and casts back to input dtype on exit.
        This ensures AMP safety: mean, covariance, inverse square root, and
        the gamma/beta affine transform all run in float32 regardless of
        autocast state (matching QuaternionBN/OctonionBN behavior).

        Args:
            x: Input tensor of shape [batch, features, 2].

        Returns:
            Normalized tensor of shape [batch, features, 2].
        """
        input_dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = x.float()

            if self.training:
                mean = x.mean(dim=0)  # [features, 2]
                x_centered = x - mean
                x_r = x_centered[..., 0]
                x_i = x_centered[..., 1]

                vrr = (x_r * x_r).mean(dim=0) + self.eps
                vri = (x_r * x_i).mean(dim=0)
                vii = (x_i * x_i).mean(dim=0) + self.eps

                with torch.no_grad():
                    self.running_mean.mul_(1 - self.momentum).add_(
                        self.momentum * mean
                    )
                    self.running_var_rr.mul_(1 - self.momentum).add_(
                        self.momentum * vrr
                    )
                    self.running_var_ri.mul_(1 - self.momentum).add_(
                        self.momentum * vri
                    )
                    self.running_var_ii.mul_(1 - self.momentum).add_(
                        self.momentum * vii
                    )
                    self.num_batches_tracked += 1
            else:
                mean = self.running_mean
                vrr = self.running_var_rr + self.eps
                vri = self.running_var_ri
                vii = self.running_var_ii + self.eps
                x_centered = x - mean

            inv_rr, inv_ri, inv_ii = self._compute_inverse_sqrt_2x2(vrr, vri, vii)

            x_r = x_centered[..., 0]
            x_i = x_centered[..., 1]
            w_r = inv_rr * x_r + inv_ri * x_i
            w_i = inv_ri * x_r + inv_ii * x_i

            out_r = self.gamma_rr * w_r + self.gamma_ri * w_i
            out_i = self.gamma_ri * w_r + self.gamma_ii * w_i

            result = torch.stack([out_r, out_i], dim=-1)
            result = result + self.beta

        return result.to(input_dtype)


class QuaternionBatchNorm(nn.Module):
    """Quaternion batch normalization using 4x4 Cholesky whitening.

    Following Gaudet & Maida, "Deep Quaternion Networks" (IJCNN 2018).

    Input shape: [batch, features, 4]
    Output shape: [batch, features, 4]

    Learnable parameters per feature:
    - gamma: 4x4 symmetric matrix stored as 10 lower-triangular entries
    - beta: 4 shift values
    Total: 14 params/feature

    Args:
        num_features: Number of quaternion features.
        eps: Value added for numerical stability.
        momentum: Running stats momentum.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.dim = 4
        # Number of unique entries in a 4x4 symmetric matrix
        self._tril_size = self.dim * (self.dim + 1) // 2  # 10

        # Learnable scale: lower-triangular entries of symmetric matrix
        # Initialize to identity: diagonal entries = 1, off-diagonal = 0
        gamma_init = torch.zeros(num_features, self._tril_size)
        idx = 0
        for i in range(self.dim):
            for j in range(i + 1):
                if i == j:
                    gamma_init[:, idx] = 1.0
                idx += 1
        self.gamma = nn.Parameter(gamma_init)

        # Learnable shift
        self.beta = nn.Parameter(torch.zeros(num_features, self.dim))

        # Running stats
        self.register_buffer(
            "running_mean", torch.zeros(num_features, self.dim)
        )
        self.register_buffer(
            "running_cov",
            torch.eye(self.dim)
            .unsqueeze(0)
            .expand(num_features, -1, -1)
            .clone(),
        )
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        # Condition number monitoring (updated each forward pass during training)
        self.register_buffer("last_cond", torch.tensor(1.0))
        # Pre-computed tril indices as buffers so tril_indices is never called
        # inside a compiled region (dynamic=True can't trace the quadratic expr).
        _rows, _cols = torch.tril_indices(self.dim, self.dim)
        self.register_buffer("_tril_rows", _rows, persistent=False)
        self.register_buffer("_tril_cols", _cols, persistent=False)

    def _whiten(
        self, x_centered: torch.Tensor, cov: torch.Tensor
    ) -> torch.Tensor:
        """Whiten using Cholesky decomposition.

        Fully torch.compile-compatible: no Python conditionals on tensors,
        no .item() calls, no autocast context managers. forward() guarantees
        float32 inputs so no explicit dtype casts are needed here either.

        Three-level fallback via unconditional torch.where (no graph breaks):
          1. Primary: cholesky_ex(cov + eps*I)
          2. Fallback: cholesky_ex(cov + 0.1*I) for any feature where info > 0
          3. Last resort: scaled identity for features where fallback also fails

        if self.training is a Python bool — Dynamo specialises on it.

        Args:
            x_centered: [batch, features, dim] float32.
            cov: [features, dim, dim] float32.

        Returns:
            Whitened tensor [batch, features, dim] float32.
        """
        eye = torch.eye(self.dim, device=cov.device, dtype=cov.dtype).unsqueeze(0)
        cov_reg = cov + self.eps * eye

        L, info = torch.linalg.cholesky_ex(cov_reg)
        failed = info > 0  # bool tensor — never used as Python bool

        # Fallback: always computed, selected via where
        cov_fallback = torch.where(
            failed.unsqueeze(-1).unsqueeze(-1),
            cov + 1e-1 * eye.squeeze(0),
            cov_reg,
        )
        L_fallback, info_fallback = torch.linalg.cholesky_ex(cov_fallback)
        L = torch.where(failed.unsqueeze(-1).unsqueeze(-1), L_fallback, L)

        # Last resort: scaled identity, always computed, selected via where
        still_failed = info_fallback > 0
        scale = (
            cov.diagonal(dim1=-2, dim2=-1)
            .abs()
            .mean(dim=-1)
            .clamp(min=1.0)
            .sqrt()
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        L = torch.where(
            still_failed.unsqueeze(-1).unsqueeze(-1), eye.squeeze(0) * scale, L
        )

        if self.training:
            with torch.no_grad():
                diag = L.diagonal(dim1=-2, dim2=-1)
                diag_abs = diag.abs().clamp(min=1e-12)
                per_feature_cond = (
                    diag_abs.max(dim=-1).values / diag_abs.min(dim=-1).values
                )
                self.last_cond.copy_(per_feature_cond.max())

        identity = eye.expand_as(L)
        L_inv = torch.linalg.solve_triangular(L, identity, upper=False)
        return (L_inv.unsqueeze(0) @ x_centered.unsqueeze(-1)).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: quaternion batch normalization.

        Computes in the parameter dtype (float32 normally, float64 if model
        was cast via .to(float64)). Disables autocast for AMP safety.

        Args:
            x: Input tensor of shape [batch, features, 4].

        Returns:
            Normalized tensor of shape [batch, features, 4].
        """
        input_dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = x.to(self.gamma.dtype)

            if self.training:
                mean = x.mean(dim=0)
                x_centered = x - mean
                cov = torch.einsum("bfi, bfj -> fij", x_centered, x_centered) / x.shape[0]

                with torch.no_grad():
                    self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                    self.running_cov.mul_(1 - self.momentum).add_(self.momentum * cov)
                    self.num_batches_tracked += 1
            else:
                mean = self.running_mean
                x_centered = x - mean
                cov = self.running_cov

            x_whitened = self._whiten(x_centered, cov)

            # Reconstruct symmetric gamma from lower-triangular entries
            gamma_sym = _tril_to_symmetric(self.gamma, self.dim, self._tril_rows, self._tril_cols)
            out = torch.einsum("fij, bfj -> bfi", gamma_sym, x_whitened)
            out = out + self.beta

        return out.to(input_dtype)


class OctonionBatchNorm(nn.Module):
    """Octonion batch normalization using 8x8 Cholesky whitening.

    Extension of quaternion BN to octonionic algebra (8 components).

    Input shape: [batch, features, 8]
    Output shape: [batch, features, 8]

    Learnable parameters per feature:
    - gamma: 8x8 symmetric matrix stored as 36 lower-triangular entries
    - beta: 8 shift values
    Total: 44 params/feature

    Args:
        num_features: Number of octonion features.
        eps: Value added for numerical stability.
        momentum: Running stats momentum.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.dim = 8
        self._tril_size = self.dim * (self.dim + 1) // 2  # 36

        # Learnable scale: lower-triangular entries of symmetric matrix
        gamma_init = torch.zeros(num_features, self._tril_size)
        idx = 0
        for i in range(self.dim):
            for j in range(i + 1):
                if i == j:
                    gamma_init[:, idx] = 1.0
                idx += 1
        self.gamma = nn.Parameter(gamma_init)

        # Learnable shift
        self.beta = nn.Parameter(torch.zeros(num_features, self.dim))

        # Running stats
        self.register_buffer(
            "running_mean", torch.zeros(num_features, self.dim)
        )
        self.register_buffer(
            "running_cov",
            torch.eye(self.dim)
            .unsqueeze(0)
            .expand(num_features, -1, -1)
            .clone(),
        )
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_cond", torch.tensor(1.0))
        _rows, _cols = torch.tril_indices(self.dim, self.dim)
        self.register_buffer("_tril_rows", _rows, persistent=False)
        self.register_buffer("_tril_cols", _cols, persistent=False)

    def _whiten(
        self, x_centered: torch.Tensor, cov: torch.Tensor
    ) -> torch.Tensor:
        """Whiten using Cholesky decomposition.

        Fully torch.compile-compatible. See QuaternionBatchNorm._whiten
        for full rationale. Identical strategy, dim=8 instead of dim=4.

        Args:
            x_centered: [batch, features, 8] float32.
            cov: [features, 8, 8] float32.

        Returns:
            Whitened tensor [batch, features, 8] float32.
        """
        eye = torch.eye(self.dim, device=cov.device, dtype=cov.dtype).unsqueeze(0)
        cov_reg = cov + self.eps * eye

        L, info = torch.linalg.cholesky_ex(cov_reg)
        failed = info > 0

        cov_fallback = torch.where(
            failed.unsqueeze(-1).unsqueeze(-1),
            cov + 1e-1 * eye.squeeze(0),
            cov_reg,
        )
        L_fallback, info_fallback = torch.linalg.cholesky_ex(cov_fallback)
        L = torch.where(failed.unsqueeze(-1).unsqueeze(-1), L_fallback, L)

        still_failed = info_fallback > 0
        scale = (
            cov.diagonal(dim1=-2, dim2=-1)
            .abs()
            .mean(dim=-1)
            .clamp(min=1.0)
            .sqrt()
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        L = torch.where(
            still_failed.unsqueeze(-1).unsqueeze(-1), eye.squeeze(0) * scale, L
        )

        if self.training:
            with torch.no_grad():
                diag = L.diagonal(dim1=-2, dim2=-1)
                diag_abs = diag.abs().clamp(min=1e-12)
                per_feature_cond = (
                    diag_abs.max(dim=-1).values / diag_abs.min(dim=-1).values
                )
                self.last_cond.copy_(per_feature_cond.max())

        identity = eye.expand_as(L)
        L_inv = torch.linalg.solve_triangular(L, identity, upper=False)
        return (L_inv.unsqueeze(0) @ x_centered.unsqueeze(-1)).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: octonion batch normalization.

        Computes in the parameter dtype (float32 normally, float64 if model
        was cast via .to(float64)). Disables autocast for AMP safety.

        Args:
            x: Input tensor of shape [batch, features, 8].

        Returns:
            Normalized tensor of shape [batch, features, 8].
        """
        input_dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = x.to(self.gamma.dtype)

            if self.training:
                mean = x.mean(dim=0)
                x_centered = x - mean
                cov = torch.einsum("bfi, bfj -> fij", x_centered, x_centered) / x.shape[0]

                with torch.no_grad():
                    self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                    self.running_cov.mul_(1 - self.momentum).add_(self.momentum * cov)
                    self.num_batches_tracked += 1
            else:
                mean = self.running_mean
                x_centered = x - mean
                cov = self.running_cov

            x_whitened = self._whiten(x_centered, cov)

            gamma_sym = _tril_to_symmetric(self.gamma, self.dim, self._tril_rows, self._tril_cols)
            out = torch.einsum("fij, bfj -> bfi", gamma_sym, x_whitened)
            out = out + self.beta

        return out.to(input_dtype)
