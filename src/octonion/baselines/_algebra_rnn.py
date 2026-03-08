"""Algebra-specific recurrent cells for baseline comparison experiments.

Implements RNN cells for each algebra following published designs:
- RealLSTMCell: wraps nn.LSTMCell (standard LSTM)
- ComplexGRUCell: complex-valued GRU following Trabelsi et al. pattern
- QuaternionLSTMCell: quaternion LSTM following Parcollet et al. 2019 (ICLR)
- OctonionLSTMCell: octonionic LSTM extending Parcollet's quaternion pattern to dim 8

All cells follow the interface:
    forward(x, state) -> new_state
where state is h (for GRU) or (h, c) (for LSTM).

Gate design for QuaternionLSTMCell and OctonionLSTMCell:
    Gates (i, f, o) are real-valued scalars derived from the real component
    of the algebra-valued gate computation: gate = sigmoid(real_part(W*x + U*h + b)).
    These scalar gates broadcast across all algebra components.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from octonion.baselines._algebra_linear import (
    ComplexLinear,
    OctonionDenseLinear,
    QuaternionLinear,
)


class RealLSTMCell(nn.Module):
    """Real-valued LSTM cell wrapping nn.LSTMCell.

    Input shape: [batch, input_size]
    State: (h, c) each [batch, hidden_size]
    Output: (h_t, c_t) tuple

    Args:
        input_size: Number of input features.
        hidden_size: Number of hidden features.
        dtype: Tensor dtype. Default: float32.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(input_size, hidden_size, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: standard LSTM cell.

        Args:
            x: Input tensor [batch, input_size].
            state: Tuple of (h, c) each [batch, hidden_size].

        Returns:
            Tuple of (h_t, c_t) each [batch, hidden_size].
        """
        h, c = state
        h_t, c_t = self.cell(x, (h, c))
        return h_t, c_t


class ComplexGRUCell(nn.Module):
    """Complex-valued GRU cell following Trabelsi et al. pattern.

    Gates (reset r, update z) computed via complex linear + sigmoid on magnitude.
    Candidate h_tilde uses complex tanh (tanh per component).

    Input shape: [batch, input_size, 2]
    State: h [batch, hidden_size, 2]
    Output: h_t [batch, hidden_size, 2]

    Args:
        input_size: Number of complex input features.
        hidden_size: Number of complex hidden features.
        dtype: Tensor dtype. Default: float32.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Gates: r (reset), z (update), h_tilde (candidate)
        # Each has input and hidden projections
        self.W_r = ComplexLinear(input_size, hidden_size, bias=False, dtype=dtype)
        self.U_r = ComplexLinear(hidden_size, hidden_size, bias=True, dtype=dtype)

        self.W_z = ComplexLinear(input_size, hidden_size, bias=False, dtype=dtype)
        self.U_z = ComplexLinear(hidden_size, hidden_size, bias=True, dtype=dtype)

        self.W_h = ComplexLinear(input_size, hidden_size, bias=False, dtype=dtype)
        self.U_h = ComplexLinear(hidden_size, hidden_size, bias=True, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: complex GRU cell.

        Args:
            x: Input tensor [batch, input_size, 2].
            h: Hidden state [batch, hidden_size, 2].

        Returns:
            h_t: Updated hidden state [batch, hidden_size, 2].
        """
        # Reset gate: sigmoid on magnitude of complex gate
        r_pre = self.W_r(x) + self.U_r(h)  # [B, hidden, 2]
        r_mag = torch.sqrt(r_pre[..., 0] ** 2 + r_pre[..., 1] ** 2 + 1e-8)
        r = torch.sigmoid(r_mag).unsqueeze(-1)  # [B, hidden, 1]

        # Update gate: sigmoid on magnitude
        z_pre = self.W_z(x) + self.U_z(h)
        z_mag = torch.sqrt(z_pre[..., 0] ** 2 + z_pre[..., 1] ** 2 + 1e-8)
        z = torch.sigmoid(z_mag).unsqueeze(-1)  # [B, hidden, 1]

        # Candidate: complex tanh (tanh per component)
        h_tilde_pre = self.W_h(x) + self.U_h(r * h)
        h_tilde = torch.tanh(h_tilde_pre)  # [B, hidden, 2]

        # Update: h_t = (1 - z) * h + z * h_tilde
        h_t = (1 - z) * h + z * h_tilde

        return h_t


class QuaternionLSTMCell(nn.Module):
    """Quaternion LSTM cell following Parcollet et al. 2019 (ICLR).

    Gates (i, f, o) are real-valued scalars derived from the real component
    of quaternion gate computations. These scalar gates broadcast across
    all 4 quaternion components.

    Input shape: [batch, input_size, 4]
    State: (h, c) each [batch, hidden_size, 4]
    Output: (h_t, c_t) tuple

    Args:
        input_size: Number of quaternion input features.
        hidden_size: Number of quaternion hidden features.
        dtype: Tensor dtype. Default: float32.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input gates: 4 quaternion linear layers (i, f, o, c)
        self.W_i = QuaternionLinear(input_size, hidden_size, bias=False, dtype=dtype)
        self.U_i = QuaternionLinear(hidden_size, hidden_size, bias=True, dtype=dtype)

        self.W_f = QuaternionLinear(input_size, hidden_size, bias=False, dtype=dtype)
        self.U_f = QuaternionLinear(hidden_size, hidden_size, bias=True, dtype=dtype)

        self.W_o = QuaternionLinear(input_size, hidden_size, bias=False, dtype=dtype)
        self.U_o = QuaternionLinear(hidden_size, hidden_size, bias=True, dtype=dtype)

        self.W_c = QuaternionLinear(input_size, hidden_size, bias=False, dtype=dtype)
        self.U_c = QuaternionLinear(hidden_size, hidden_size, bias=True, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: quaternion LSTM cell.

        Args:
            x: Input tensor [batch, input_size, 4].
            state: Tuple of (h, c) each [batch, hidden_size, 4].

        Returns:
            Tuple of (h_t, c_t) each [batch, hidden_size, 4].
        """
        h, c = state

        # Gates derived from real component of quaternion computation
        # i = sigmoid(real_part(W_i * x + U_i * h + b_i))
        i_pre = self.W_i(x) + self.U_i(h)  # [B, hidden, 4]
        i = torch.sigmoid(i_pre[..., 0]).unsqueeze(-1)  # [B, hidden, 1]

        f_pre = self.W_f(x) + self.U_f(h)
        f = torch.sigmoid(f_pre[..., 0]).unsqueeze(-1)  # [B, hidden, 1]

        o_pre = self.W_o(x) + self.U_o(h)
        o = torch.sigmoid(o_pre[..., 0]).unsqueeze(-1)  # [B, hidden, 1]

        # Cell candidate: tanh applied per component
        c_tilde = torch.tanh(self.W_c(x) + self.U_c(h))  # [B, hidden, 4]

        # Cell update: scalar gates broadcast across quaternion components
        c_t = f * c + i * c_tilde  # [B, hidden, 4]
        h_t = o * torch.tanh(c_t)  # [B, hidden, 4]

        return h_t, c_t


class OctonionLSTMCell(nn.Module):
    """Octonion LSTM cell extending Parcollet's quaternion pattern to dim 8.

    Same gate structure as QuaternionLSTMCell: gates derived from real component
    of octonionic computation, broadcasting across all 8 components.
    Uses OctonionDenseLinear for algebra-specific transforms.

    Input shape: [batch, input_size, 8]
    State: (h, c) each [batch, hidden_size, 8]
    Output: (h_t, c_t) tuple

    Args:
        input_size: Number of octonion input features.
        hidden_size: Number of octonion hidden features.
        dtype: Tensor dtype. Default: float32.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input gates: 4 octonion linear layers (i, f, o, c)
        self.W_i = OctonionDenseLinear(input_size, hidden_size, bias=False, dtype=dtype)
        self.U_i = OctonionDenseLinear(hidden_size, hidden_size, bias=True, dtype=dtype)

        self.W_f = OctonionDenseLinear(input_size, hidden_size, bias=False, dtype=dtype)
        self.U_f = OctonionDenseLinear(hidden_size, hidden_size, bias=True, dtype=dtype)

        self.W_o = OctonionDenseLinear(input_size, hidden_size, bias=False, dtype=dtype)
        self.U_o = OctonionDenseLinear(hidden_size, hidden_size, bias=True, dtype=dtype)

        self.W_c = OctonionDenseLinear(input_size, hidden_size, bias=False, dtype=dtype)
        self.U_c = OctonionDenseLinear(hidden_size, hidden_size, bias=True, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: octonion LSTM cell.

        Args:
            x: Input tensor [batch, input_size, 8].
            state: Tuple of (h, c) each [batch, hidden_size, 8].

        Returns:
            Tuple of (h_t, c_t) each [batch, hidden_size, 8].
        """
        h, c = state

        # Gates derived from real component of octonionic computation
        i_pre = self.W_i(x) + self.U_i(h)  # [B, hidden, 8]
        i = torch.sigmoid(i_pre[..., 0]).unsqueeze(-1)  # [B, hidden, 1]

        f_pre = self.W_f(x) + self.U_f(h)
        f = torch.sigmoid(f_pre[..., 0]).unsqueeze(-1)

        o_pre = self.W_o(x) + self.U_o(h)
        o = torch.sigmoid(o_pre[..., 0]).unsqueeze(-1)

        # Cell candidate
        c_tilde = torch.tanh(self.W_c(x) + self.U_c(h))  # [B, hidden, 8]

        # Cell update: scalar gates broadcast across octonion components
        c_t = f * c + i * c_tilde
        h_t = o * torch.tanh(c_t)

        return h_t, c_t
