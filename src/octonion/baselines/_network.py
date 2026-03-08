"""Algebra-agnostic network skeleton for baseline comparison experiments.

AlgebraNetwork is the central abstraction enabling fair comparison: one
configurable nn.Module that builds R/C/H/O variants with matched parameters
and identical topology. SC-4 (shared skeleton) is directly testable.

Three topology types:
- MLP: input_proj -> [hidden + BN + activation] * depth -> output_proj
- Conv2D: input_conv -> [conv_block + BN + activation + pool] * depth -> GAP -> fc
- Recurrent: input_proj -> RNN cells * depth -> output_proj

Output projection strategies:
1. "real": extract component 0, then linear to output_dim
2. "flatten": reshape all components to real, then linear
3. "norm": compute algebra norm, then linear
4. "learned": linear from flattened (named differently for phased training)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from octonion.baselines._activation import NormPreservingActivation, SplitActivation
from octonion.baselines._algebra_conv import (
    ComplexConv2d,
    OctonionConv2d,
    QuaternionConv2d,
    RealConv2d,
)
from octonion.baselines._algebra_linear import (
    ComplexLinear,
    OctonionDenseLinear,
    QuaternionLinear,
    RealLinear,
)
from octonion.baselines._algebra_rnn import (
    ComplexGRUCell,
    OctonionLSTMCell,
    QuaternionLSTMCell,
    RealLSTMCell,
)
from octonion.baselines._config import AlgebraType, NetworkConfig
from octonion.baselines._normalization import (
    ComplexBatchNorm,
    OctonionBatchNorm,
    QuaternionBatchNorm,
    RealBatchNorm,
)


class AlgebraNetwork(nn.Module):
    """Algebra-agnostic neural network with configurable topology.

    Builds identical topology skeletons for R/C/H/O algebras,
    differing only in the algebra-specific layer modules. This
    directly enables SC-4 (shared skeleton) testing.

    Args:
        config: NetworkConfig specifying algebra, topology, depth, etc.
    """

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config
        # hidden is the number of algebra units (not real dims)
        # For parameter matching, the multiplier scales the width
        self.hidden = config.base_hidden * config.algebra.multiplier
        self.algebra_dim = config.algebra.dim

        if config.topology == "mlp":
            self._build_mlp(config)
        elif config.topology == "conv2d":
            self._build_conv(config)
        elif config.topology == "recurrent":
            self._build_recurrent(config)
        else:
            raise ValueError(f"Unknown topology: {config.topology}")

        # Build output projection
        self._build_output_projection(config)

    # ── Layer Factories ─────────────────────────────────────────

    def _get_linear(self, in_f: int, out_f: int) -> nn.Module:
        """Return algebra-specific linear layer."""
        algebra = self.config.algebra
        if algebra == AlgebraType.REAL:
            return RealLinear(in_f, out_f)
        elif algebra == AlgebraType.COMPLEX:
            return ComplexLinear(in_f, out_f)
        elif algebra == AlgebraType.QUATERNION:
            return QuaternionLinear(in_f, out_f)
        elif algebra == AlgebraType.OCTONION:
            return OctonionDenseLinear(in_f, out_f)
        raise ValueError(f"Unknown algebra: {algebra}")

    def _get_conv2d(
        self, in_ch: int, out_ch: int, kernel: int, padding: int = 0,
    ) -> nn.Module:
        """Return algebra-specific 2D conv layer."""
        algebra = self.config.algebra
        if algebra == AlgebraType.REAL:
            return RealConv2d(in_ch, out_ch, kernel, padding=padding)
        elif algebra == AlgebraType.COMPLEX:
            return ComplexConv2d(in_ch, out_ch, kernel, padding=padding)
        elif algebra == AlgebraType.QUATERNION:
            return QuaternionConv2d(in_ch, out_ch, kernel, padding=padding)
        elif algebra == AlgebraType.OCTONION:
            return OctonionConv2d(in_ch, out_ch, kernel, padding=padding)
        raise ValueError(f"Unknown algebra: {algebra}")

    def _get_bn(self, features: int) -> nn.Module:
        """Return algebra-specific batch normalization layer."""
        algebra = self.config.algebra
        if algebra == AlgebraType.REAL:
            return RealBatchNorm(features)
        elif algebra == AlgebraType.COMPLEX:
            return ComplexBatchNorm(features)
        elif algebra == AlgebraType.QUATERNION:
            return QuaternionBatchNorm(features)
        elif algebra == AlgebraType.OCTONION:
            return OctonionBatchNorm(features)
        raise ValueError(f"Unknown algebra: {algebra}")

    def _get_activation(self) -> nn.Module:
        """Return activation module based on config."""
        act = self.config.activation
        if act.startswith("split_"):
            fn_name = act[len("split_"):]
            return SplitActivation(fn_name)
        elif act == "norm_preserving":
            return NormPreservingActivation("relu")
        raise ValueError(f"Unknown activation: {act}")

    def _get_rnn_cell(self, in_size: int, hidden_size: int) -> nn.Module:
        """Return algebra-specific RNN cell."""
        algebra = self.config.algebra
        if algebra == AlgebraType.REAL:
            return RealLSTMCell(in_size, hidden_size)
        elif algebra == AlgebraType.COMPLEX:
            return ComplexGRUCell(in_size, hidden_size)
        elif algebra == AlgebraType.QUATERNION:
            return QuaternionLSTMCell(in_size, hidden_size)
        elif algebra == AlgebraType.OCTONION:
            return OctonionLSTMCell(in_size, hidden_size)
        raise ValueError(f"Unknown algebra: {algebra}")

    # ── Topology Builders ───────────────────────────────────────

    def _build_mlp(self, config: NetworkConfig) -> None:
        """Build MLP topology: input_proj -> hidden blocks -> (output via projection).

        Input: [B, input_dim] (real-valued)
        Hidden: [B, hidden, algebra_dim] (algebra-valued)
        """
        hidden = self.hidden
        dim = self.algebra_dim

        # Input projection: real -> algebra
        self.input_proj = nn.Linear(config.input_dim, hidden * dim)

        # Hidden blocks
        self.hidden_blocks = nn.ModuleList()
        for _ in range(config.depth):
            block = nn.ModuleDict()
            block["linear"] = self._get_linear(hidden, hidden)
            if config.use_batchnorm:
                block["bn"] = self._get_bn(hidden)
            block["activation"] = self._get_activation()
            self.hidden_blocks.append(block)

    def _build_conv(self, config: NetworkConfig) -> None:
        """Build Conv2D topology: input_conv -> conv blocks -> GAP -> fc.

        Input: [B, C, H, W] (real-valued image)
        Hidden: algebra-specific conv layout
        """
        dim = self.algebra_dim
        base_filters = config.base_hidden * config.algebra.multiplier

        # Initial conv block: map input channels to algebra-valued features
        # For non-real algebras, we project input channels into algebra-valued space
        self.input_conv = self._get_conv2d(
            config.input_dim, base_filters, kernel=3, padding=1,
        )
        self.input_bn = self._get_bn(base_filters) if config.use_batchnorm else None
        self.input_act = self._get_activation()

        # Conv blocks with stage boundaries
        self.conv_blocks = nn.ModuleList()
        in_filters = base_filters
        out_filters = base_filters
        for i in range(config.depth):
            block = nn.ModuleDict()
            block["conv"] = self._get_conv2d(
                in_filters, out_filters, kernel=3, padding=1,
            )
            if config.use_batchnorm:
                block["bn"] = self._get_bn(out_filters)
            block["activation"] = self._get_activation()
            # Pool at each stage to reduce spatial dims
            block["pool"] = nn.MaxPool2d(2, 2)
            self.conv_blocks.append(block)
            in_filters = out_filters

        # Final fc layers
        self._conv_final_filters = out_filters
        self.fc_hidden = self._get_linear(out_filters, out_filters)
        self.fc_act = self._get_activation()

    def _build_recurrent(self, config: NetworkConfig) -> None:
        """Build Recurrent topology: input_proj -> RNN cells -> (output via projection).

        Input: [B, seq_len, input_dim] (real-valued sequence)
        Hidden: algebra-valued RNN states
        """
        hidden = self.hidden
        dim = self.algebra_dim

        # Input projection: per-timestep, real -> algebra
        self.input_proj = nn.Linear(config.input_dim, hidden * dim)

        # Stack of RNN cells
        self.rnn_cells = nn.ModuleList()
        for layer_idx in range(config.depth):
            cell = self._get_rnn_cell(hidden, hidden)
            self.rnn_cells.append(cell)

    # ── Output Projection ───────────────────────────────────────

    def _build_output_projection(self, config: NetworkConfig) -> None:
        """Build output projection from algebra-valued to real-valued.

        Four strategies:
        1. "real": component 0 -> nn.Linear
        2. "flatten": all components -> nn.Linear
        3. "norm": algebra norm -> nn.Linear
        4. "learned": flattened -> nn.Linear (same as flatten, named for phased training)
        """
        hidden = self.hidden
        dim = self.algebra_dim
        strategy = config.output_projection

        if strategy == "real":
            # Extract component 0, then project
            self.output_proj = nn.Linear(hidden, config.output_dim)
        elif strategy in ("flatten", "learned"):
            self.output_proj = nn.Linear(hidden * dim, config.output_dim)
        elif strategy == "norm":
            self.output_proj = nn.Linear(hidden, config.output_dim)
        else:
            raise ValueError(f"Unknown output projection: {strategy}")

    def _apply_output_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Apply output projection to algebra-valued features.

        Args:
            x: Algebra-valued tensor.
                For REAL: [B, hidden]
                For others: [B, hidden, dim]

        Returns:
            [B, output_dim] real-valued output.
        """
        strategy = self.config.output_projection
        dim = self.algebra_dim

        if dim == 1:
            # Real algebra: already real-valued [B, hidden]
            return self.output_proj(x)

        if strategy == "real":
            # Extract component 0
            return self.output_proj(x[..., 0])
        elif strategy in ("flatten", "learned"):
            # Flatten all components
            flat = x.reshape(x.shape[0], -1)
            return self.output_proj(flat)
        elif strategy == "norm":
            # Algebra norm
            norm = x.norm(dim=-1)  # [B, hidden]
            return self.output_proj(norm)
        else:
            raise ValueError(f"Unknown output projection: {strategy}")

    # ── Forward Pass ────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor. Shape depends on topology:
                - MLP: [B, input_dim]
                - Conv2D: [B, C, H, W]
                - Recurrent: [B, seq_len, input_dim]

        Returns:
            [B, output_dim] output tensor.
        """
        topology = self.config.topology
        if topology == "mlp":
            return self._forward_mlp(x)
        elif topology == "conv2d":
            return self._forward_conv(x)
        elif topology == "recurrent":
            return self._forward_recurrent(x)
        raise ValueError(f"Unknown topology: {topology}")

    def _forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        """MLP forward: input_proj -> hidden blocks -> output projection.

        Args:
            x: [B, input_dim]
        """
        dim = self.algebra_dim

        # Input projection: [B, input_dim] -> [B, hidden * dim]
        h = self.input_proj(x)

        # Reshape to algebra-valued: [B, hidden, dim] (skip for real)
        if dim > 1:
            h = h.view(h.shape[0], self.hidden, dim)

        # Hidden blocks
        for block in self.hidden_blocks:
            h = block["linear"](h)
            if "bn" in block:
                h = block["bn"](h)
            h = block["activation"](h)

        # Output projection
        return self._apply_output_projection(h)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Conv2D forward: input_conv -> conv blocks -> GAP -> fc -> output.

        Args:
            x: [B, C, H, W]
        """
        dim = self.algebra_dim

        # For non-real algebras, we need to expand input channels into algebra dims
        # Conv layout: Real=[B, ch, H, W], others=[B, ch, dim, H, W]
        if dim > 1:
            # Replicate input across algebra dimensions
            # [B, C, H, W] -> [B, C, dim, H, W]
            x = x.unsqueeze(2).expand(-1, -1, dim, -1, -1)

        # Initial conv
        h = self.input_conv(x)
        if self.input_bn is not None:
            if dim == 1:
                # RealBatchNorm expects [B, features] but conv gives [B, ch, H, W]
                # Use a pass-through for conv shapes
                b, ch, hh, ww = h.shape
                h = h.permute(0, 2, 3, 1).reshape(-1, ch)
                h = self.input_bn(h)
                h = h.reshape(b, hh, ww, ch).permute(0, 3, 1, 2)
            else:
                # Hypercomplex BN expects [B, features, dim]
                # Conv output: [B, ch, dim, H, W]
                b, ch, d, hh, ww = h.shape
                h = h.permute(0, 3, 4, 1, 2).reshape(-1, ch, d)
                h = self.input_bn(h)
                h = h.reshape(b, hh, ww, ch, d).permute(0, 3, 4, 1, 2)
        h = self.input_act(h)

        # Conv blocks
        for block in self.conv_blocks:
            h = block["conv"](h)
            if "bn" in block:
                if dim == 1:
                    b, ch, hh, ww = h.shape
                    h = h.permute(0, 2, 3, 1).reshape(-1, ch)
                    h = block["bn"](h)
                    h = h.reshape(b, hh, ww, ch).permute(0, 3, 1, 2)
                else:
                    b, ch, d, hh, ww = h.shape
                    h = h.permute(0, 3, 4, 1, 2).reshape(-1, ch, d)
                    h = block["bn"](h)
                    h = h.reshape(b, hh, ww, ch, d).permute(0, 3, 4, 1, 2)
            h = block["activation"](h)
            # Pool: need to handle algebra dim correctly
            if dim == 1:
                h = block["pool"](h)
            else:
                # Pool over spatial dims only, keeping algebra dim
                # [B, ch, dim, H, W] -> pool over H, W
                b, ch, d, hh, ww = h.shape
                h = h.reshape(b * ch * d, 1, hh, ww)
                h = block["pool"](h)
                hh2, ww2 = h.shape[2], h.shape[3]
                h = h.reshape(b, ch, d, hh2, ww2)

        # Global average pooling over spatial dims
        if dim == 1:
            # [B, ch, H, W] -> [B, ch]
            h = h.mean(dim=[2, 3])
        else:
            # [B, ch, dim, H, W] -> [B, ch, dim]
            h = h.mean(dim=[3, 4])

        # FC hidden layer
        h = self.fc_hidden(h)
        h = self.fc_act(h)

        # Output projection
        return self._apply_output_projection(h)

    def _forward_recurrent(self, x: torch.Tensor) -> torch.Tensor:
        """Recurrent forward: input_proj -> RNN cells over time -> output.

        Args:
            x: [B, seq_len, input_dim]
        """
        dim = self.algebra_dim
        B, seq_len, _ = x.shape

        # Initialize states for all layers
        states = []
        for cell in self.rnn_cells:
            if isinstance(cell, (RealLSTMCell,)):
                h0 = torch.zeros(B, self.hidden, device=x.device, dtype=x.dtype)
                c0 = torch.zeros(B, self.hidden, device=x.device, dtype=x.dtype)
                states.append((h0, c0))
            elif isinstance(cell, ComplexGRUCell):
                states.append(
                    torch.zeros(B, self.hidden, dim, device=x.device, dtype=x.dtype)
                )
            elif isinstance(cell, (QuaternionLSTMCell, OctonionLSTMCell)):
                h0 = torch.zeros(
                    B, self.hidden, dim, device=x.device, dtype=x.dtype,
                )
                c0 = torch.zeros(
                    B, self.hidden, dim, device=x.device, dtype=x.dtype,
                )
                states.append((h0, c0))

        # Process each timestep
        for t in range(seq_len):
            # Input projection for this timestep
            xt = self.input_proj(x[:, t, :])  # [B, hidden * dim]
            if dim > 1:
                xt = xt.view(B, self.hidden, dim)

            # Pass through stacked RNN cells
            h_in = xt
            for layer_idx, cell in enumerate(self.rnn_cells):
                if isinstance(cell, RealLSTMCell):
                    h_out, c_out = cell(h_in, states[layer_idx])
                    states[layer_idx] = (h_out, c_out)
                    h_in = h_out
                elif isinstance(cell, ComplexGRUCell):
                    h_out = cell(h_in, states[layer_idx])
                    states[layer_idx] = h_out
                    h_in = h_out
                elif isinstance(cell, (QuaternionLSTMCell, OctonionLSTMCell)):
                    h_out, c_out = cell(h_in, states[layer_idx])
                    states[layer_idx] = (h_out, c_out)
                    h_in = h_out

        # Take final hidden state from last layer
        final_state = states[-1]
        if isinstance(final_state, tuple):
            final_h = final_state[0]  # h from (h, c)
        else:
            final_h = final_state  # h from GRU

        # Output projection
        return self._apply_output_projection(final_h)

    def param_report(self) -> list[dict]:
        """Per-layer parameter breakdown.

        Returns:
            List of dicts with keys: name, shape, real_params, pct.
        """
        total = sum(p.numel() for p in self.parameters())
        if total == 0:
            return []

        entries = []
        for name, param in self.named_parameters():
            numel = param.numel()
            entries.append({
                "name": name,
                "shape": list(param.shape),
                "real_params": numel,
                "pct": numel / total * 100.0,
            })
        return entries
