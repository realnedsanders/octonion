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


def _tril_to_symmetric(tril_flat: torch.Tensor, dim: int) -> torch.Tensor:
    """Convert flat lower-triangular entries to a symmetric matrix.

    Vectorized via torch.tril_indices: no Python-level loops.
    The index computation is a fast C++ call returning a small tensor
    (dim*(dim+1)/2 elements), so overhead is negligible versus the
    eliminated Python loop.

    Args:
        tril_flat: [..., dim*(dim+1)/2] flat lower-triangular entries.
        dim: Matrix dimension.

    Returns:
        [..., dim, dim] symmetric matrix.
    """
    batch_shape = tril_flat.shape[:-1]
    rows, cols = torch.tril_indices(dim, dim, device=tril_flat.device)
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

        Args:
            x: Input tensor of shape [batch, features, 2].

        Returns:
            Normalized tensor of shape [batch, features, 2].
        """
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

        return result


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

    @torch.compiler.disable
    def _whiten(
        self, x_centered: torch.Tensor, cov: torch.Tensor
    ) -> torch.Tensor:
        """Whiten using Cholesky decomposition with AMP float32 protection.

        Computes L = cholesky_ex(cov), then L_inv once per feature, then
        applies via broadcast matmul. This matches the reference approach
        from Gaudet & Maida 2018 (precompute inverse whitening matrix,
        multiply) rather than expanding L to [features, batch, dim, dim]
        for solve_triangular — which is mathematically equivalent but
        creates pathological memory/compute when batch includes spatial
        positions from conv layers (batch = B * H * W).

        AMP protection: Cholesky and solve_triangular always run in float32
        regardless of autocast state. When AMP is disabled, the
        autocast(enabled=False) context is a no-op.

        cholesky_ex: Uses cholesky_ex instead of try/except cholesky for
        torch.compile compatibility (no graph breaks) and per-feature
        failure detection via the info tensor.

        @torch.compiler.disable: This function contains inescapable Python
        control flow (per-feature fallback logic, logging) that cannot be
        traced by torch.compile. The decorator tells Dynamo to execute this
        function eagerly, eliminating graph breaks and recompilation storms.
        The outer network (conv, linear, residual) still gets compiled.

        Args:
            x_centered: [batch, features, dim] centered input (float32).
            cov: [features, dim, dim] covariance matrix (float32).

        Returns:
            Whitened tensor [batch, features, dim] in float32.
        """
        # Disable autocast for numerically sensitive Cholesky/solve_triangular.
        # When AMP is enabled, inputs may arrive as float16 but Cholesky needs
        # float32. When AMP is disabled, this context manager is a no-op.
        with torch.amp.autocast(device_type=x_centered.device.type, enabled=False):
            # Force float32 for all linalg operations
            cov_f32 = cov.float()
            x_f32 = x_centered.float()

            eye = torch.eye(
                self.dim, device=cov_f32.device, dtype=torch.float32
            ).unsqueeze(0)
            cov_reg = cov_f32 + self.eps * eye

            # Use cholesky_ex instead of try/except cholesky:
            # - No graph breaks (torch.compile compatible)
            # - Per-feature failure detection via info tensor
            # - info[i] == 0 means feature i succeeded, info[i] > 0 means failure
            L, info = torch.linalg.cholesky_ex(cov_reg)

            # Per-feature fallback: only re-regularize features that failed
            failed = info > 0  # [features] boolean mask
            if failed.any():
                num_failed = int(failed.sum().item())
                logger.warning(
                    "Cholesky failed for %d/%d features in QuaternionBatchNorm, "
                    "applying per-feature regularization.",
                    num_failed,
                    cov_reg.shape[0],
                )
                cov_fixed = cov_reg.clone()
                cov_fixed[failed] = cov_f32[failed] + 1e-1 * eye.squeeze(0)
                L_fixed, info_fixed = torch.linalg.cholesky_ex(cov_fixed)
                # Replace only failed features' L factors
                L = torch.where(failed.unsqueeze(-1).unsqueeze(-1), L_fixed, L)

                still_failed = info_fixed > 0
                if still_failed.any():
                    # Last resort: use scaled identity for features that still fail
                    logger.warning(
                        "Cholesky still failing for %d features after strong "
                        "regularization, using scaled identity.",
                        int(still_failed.sum().item()),
                    )
                    scale = (
                        cov_f32[still_failed]
                        .diagonal(dim1=-2, dim2=-1)
                        .abs()
                        .mean(dim=-1, keepdim=True)
                        .unsqueeze(-1)
                        .clamp(min=1.0)
                        .sqrt()
                    )
                    L[still_failed] = eye.squeeze(0) * scale

            # Track condition number (max across features, no grad needed)
            if self.training:
                with torch.no_grad():
                    diag = L.diagonal(dim1=-2, dim2=-1)  # [features, dim]
                    diag_abs = diag.abs().clamp(min=1e-12)
                    per_feature_cond = (
                        diag_abs.max(dim=-1).values / diag_abs.min(dim=-1).values
                    )
                    self.last_cond.copy_(per_feature_cond.max())

            # Compute L_inv once: [features, dim, dim]
            # Only `features` number of dim×dim inversions (not features × batch).
            identity = eye.expand_as(L)
            L_inv = torch.linalg.solve_triangular(L, identity, upper=False)

            # Apply whitening via broadcast matmul:
            # L_inv: [features, dim, dim] -> [1, features, dim, dim]
            # x:     [batch, features, dim] -> [batch, features, dim, 1]
            # result: [batch, features, dim, 1] -> [batch, features, dim]
            x_col = x_f32.unsqueeze(-1)
            w = (L_inv.unsqueeze(0) @ x_col).squeeze(-1)

        # Return in original dtype (autocast handles downstream casting)
        return w.to(x_centered.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: quaternion batch normalization.

        Always computes in float32 and casts back to input dtype on exit.
        This ensures AMP safety: mean, cov, whitening, and the gamma/beta
        affine transform all run in float32 regardless of autocast state.

        Args:
            x: Input tensor of shape [batch, features, 4].

        Returns:
            Normalized tensor of shape [batch, features, 4].
        """
        input_dtype = x.dtype
        x = x.float()  # entire BN forward in float32 for numerical stability

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
        gamma_sym = _tril_to_symmetric(self.gamma, self.dim)
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

    @torch.compiler.disable
    def _whiten(
        self, x_centered: torch.Tensor, cov: torch.Tensor
    ) -> torch.Tensor:
        """Whiten using Cholesky decomposition with AMP float32 protection.

        Same precomputed-inverse strategy as QuaternionBatchNorm._whiten.
        See that method's docstring for rationale on AMP protection,
        cholesky_ex usage, and @torch.compiler.disable.

        Args:
            x_centered: [batch, features, dim] centered input (float32).
            cov: [features, dim, dim] covariance matrix (float32).

        Returns:
            Whitened tensor [batch, features, dim] in float32.
        """
        # Disable autocast for numerically sensitive Cholesky/solve_triangular.
        # When AMP is enabled, inputs may arrive as float16 but Cholesky needs
        # float32. When AMP is disabled, this context manager is a no-op.
        with torch.amp.autocast(device_type=x_centered.device.type, enabled=False):
            # Force float32 for all linalg operations
            cov_f32 = cov.float()
            x_f32 = x_centered.float()

            eye = torch.eye(
                self.dim, device=cov_f32.device, dtype=torch.float32
            ).unsqueeze(0)
            cov_reg = cov_f32 + self.eps * eye

            # Use cholesky_ex instead of try/except cholesky:
            # - No graph breaks (torch.compile compatible)
            # - Per-feature failure detection via info tensor
            # - info[i] == 0 means feature i succeeded, info[i] > 0 means failure
            L, info = torch.linalg.cholesky_ex(cov_reg)

            # Per-feature fallback: only re-regularize features that failed
            failed = info > 0  # [features] boolean mask
            if failed.any():
                num_failed = int(failed.sum().item())
                logger.warning(
                    "Cholesky failed for %d/%d features in OctonionBatchNorm, "
                    "applying per-feature regularization.",
                    num_failed,
                    cov_reg.shape[0],
                )
                cov_fixed = cov_reg.clone()
                cov_fixed[failed] = cov_f32[failed] + 1e-1 * eye.squeeze(0)
                L_fixed, info_fixed = torch.linalg.cholesky_ex(cov_fixed)
                # Replace only failed features' L factors
                L = torch.where(failed.unsqueeze(-1).unsqueeze(-1), L_fixed, L)

                still_failed = info_fixed > 0
                if still_failed.any():
                    # Last resort: use scaled identity for features that still fail
                    logger.warning(
                        "Cholesky still failing for %d features after strong "
                        "regularization, using scaled identity.",
                        int(still_failed.sum().item()),
                    )
                    scale = (
                        cov_f32[still_failed]
                        .diagonal(dim1=-2, dim2=-1)
                        .abs()
                        .mean(dim=-1, keepdim=True)
                        .unsqueeze(-1)
                        .clamp(min=1.0)
                        .sqrt()
                    )
                    L[still_failed] = eye.squeeze(0) * scale

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

            x_col = x_f32.unsqueeze(-1)
            w = (L_inv.unsqueeze(0) @ x_col).squeeze(-1)

        # Return in original dtype (autocast handles downstream casting)
        return w.to(x_centered.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: octonion batch normalization.

        Always computes in float32 and casts back to input dtype on exit.
        This ensures AMP safety: mean, cov, whitening, and the gamma/beta
        affine transform all run in float32 regardless of autocast state.

        Args:
            x: Input tensor of shape [batch, features, 8].

        Returns:
            Normalized tensor of shape [batch, features, 8].
        """
        input_dtype = x.dtype
        x = x.float()  # entire BN forward in float32 for numerical stability

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

        gamma_sym = _tril_to_symmetric(self.gamma, self.dim)
        out = torch.einsum("fij, bfj -> bfi", gamma_sym, x_whitened)
        out = out + self.beta

        return out.to(input_dtype)
