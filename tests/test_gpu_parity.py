"""GPU/CPU parity tests for octonionic backward passes (SC-4).

Verifies that backward passes on ROCm GPU produce identical results to CPU
computation within tolerance at float64 precision.

Per CONTEXT.md: "Manual GPU verification only -- not part of CI pipeline."
All tests are marked with @pytest.mark.gpu and skipped when no GPU is available.

Tolerance rationale:
  At float64, GPU and CPU should agree to near machine-epsilon precision.
  We use 1e-12 absolute tolerance, which is ~4500x the float64 machine
  epsilon (2.2e-16). This accommodates minor differences in reduction order
  between ROCm and CPU backends. If ROCm accumulation order produces
  differences in the 1e-13 to 1e-14 range, this tolerance absorbs them
  while still being strict enough to catch real bugs.

Convention: Baez 2002, mod-7 Fano plane basis.
"""

from __future__ import annotations

import pytest
import torch

from octonion.calculus._autograd_functions import (
    OctonionConjugateFunction,
    OctonionExpFunction,
    OctonionInverseFunction,
    OctonionLogFunction,
    OctonionMulFunction,
)

GPU_AVAILABLE = torch.cuda.is_available()
gpu = pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU available")

# Tolerance for GPU/CPU parity: 1e-12 at float64
# Rationale: ~4500x machine epsilon, accommodates reduction order differences
GPU_TOL = 1e-12


def _make_pair(
    *shapes: tuple[int, ...],
    dtype: torch.dtype = torch.float64,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Create paired CPU and GPU tensors with requires_grad=True.

    Returns (cpu_inputs, gpu_inputs) where gpu_inputs are .cuda() clones.
    """
    cpu_inputs = []
    gpu_inputs = []
    for shape in shapes:
        t = torch.randn(shape, dtype=dtype)
        t_cpu = t.clone().requires_grad_(True)
        t_gpu = t.clone().cuda().requires_grad_(True)
        cpu_inputs.append(t_cpu)
        gpu_inputs.append(t_gpu)
    return cpu_inputs, gpu_inputs


def _compare_gradients(
    fn_cpu,
    fn_gpu,
    inputs_cpu: list[torch.Tensor],
    inputs_gpu: list[torch.Tensor],
    tol: float = GPU_TOL,
) -> float:
    """Run forward+backward on CPU and GPU, compare gradients.

    Returns the maximum absolute difference across all input gradients.
    Raises AssertionError if any gradient exceeds tolerance.
    """
    # CPU forward + backward
    out_cpu = fn_cpu(*inputs_cpu)
    loss_cpu = out_cpu.sum()
    loss_cpu.backward()
    grads_cpu = [inp.grad.clone() for inp in inputs_cpu if inp.grad is not None]

    # GPU forward + backward
    out_gpu = fn_gpu(*inputs_gpu)
    loss_gpu = out_gpu.sum()
    loss_gpu.backward()
    grads_gpu = [inp.grad.cpu().clone() for inp in inputs_gpu if inp.grad is not None]

    assert len(grads_cpu) == len(grads_gpu), (
        f"Different number of gradient tensors: CPU={len(grads_cpu)}, GPU={len(grads_gpu)}"
    )

    max_diff = 0.0
    for i, (gc, gg) in enumerate(zip(grads_cpu, grads_gpu, strict=False)):
        diff = torch.abs(gc - gg)
        d = diff.max().item()
        max_diff = max(max_diff, d)
        assert d < tol, (
            f"Input {i}: max GPU/CPU gradient diff = {d:.2e} > tol {tol:.2e}\n"
            f"CPU grad: {gc}\nGPU grad: {gg}\nDiff: {diff}"
        )

    return max_diff


@gpu
class TestGPUCPUParity:
    """Verify backward passes produce identical results on GPU and CPU (SC-4)."""

    def test_mul_parity(self) -> None:
        """OctonionMulFunction backward: GPU vs CPU at float64."""
        torch.manual_seed(42)
        inputs_cpu, inputs_gpu = _make_pair((8,), (8,))

        _compare_gradients(
            OctonionMulFunction.apply,
            OctonionMulFunction.apply,
            inputs_cpu,
            inputs_gpu,
        )

    def test_exp_parity(self) -> None:
        """OctonionExpFunction backward: GPU vs CPU at float64."""
        torch.manual_seed(42)
        inputs_cpu, inputs_gpu = _make_pair((8,))

        _compare_gradients(
            OctonionExpFunction.apply,
            OctonionExpFunction.apply,
            inputs_cpu,
            inputs_gpu,
        )

    def test_log_parity(self) -> None:
        """OctonionLogFunction backward: GPU vs CPU at float64.

        Uses input with positive norm and non-zero imaginary part to avoid
        singularities in the log Jacobian.
        """
        torch.manual_seed(42)
        # Construct a safe input: shift scalar part positive, ensure non-zero imaginary
        t = torch.randn(8, dtype=torch.float64)
        t[0] = abs(t[0]) + 1.0  # Ensure positive scalar part


        t_cpu = t.clone().requires_grad_(True)
        t_gpu = t.clone().cuda().requires_grad_(True)

        _compare_gradients(
            OctonionLogFunction.apply,
            OctonionLogFunction.apply,
            [t_cpu],
            [t_gpu],
        )

    def test_conjugate_parity(self) -> None:
        """OctonionConjugateFunction backward: GPU vs CPU at float64."""
        torch.manual_seed(42)
        inputs_cpu, inputs_gpu = _make_pair((8,))

        _compare_gradients(
            OctonionConjugateFunction.apply,
            OctonionConjugateFunction.apply,
            inputs_cpu,
            inputs_gpu,
        )

    def test_inverse_parity(self) -> None:
        """OctonionInverseFunction backward: GPU vs CPU at float64.

        Uses input with non-zero norm to avoid singularity.
        """
        torch.manual_seed(42)
        t = torch.randn(8, dtype=torch.float64)
        t = t + 0.1 * torch.sign(t)  # Push away from zero

        t_cpu = t.clone().requires_grad_(True)
        t_gpu = t.clone().cuda().requires_grad_(True)

        _compare_gradients(
            OctonionInverseFunction.apply,
            OctonionInverseFunction.apply,
            [t_cpu],
            [t_gpu],
        )

    def test_octonion_linear_parity(self) -> None:
        """Full OctonionLinear layer forward+backward: GPU vs CPU at float64."""
        torch.manual_seed(42)
        from octonion._linear import OctonionLinear

        layer_cpu = OctonionLinear(dtype=torch.float64)
        layer_gpu = OctonionLinear(dtype=torch.float64).cuda()

        # Copy weights from CPU to GPU for identical initialization
        with torch.no_grad():
            layer_gpu.a.copy_(layer_cpu.a)
            layer_gpu.b.copy_(layer_cpu.b)

        x = torch.randn(8, dtype=torch.float64)
        x_cpu = x.clone().requires_grad_(True)
        x_gpu = x.clone().cuda().requires_grad_(True)

        _compare_gradients(
            layer_cpu,
            layer_gpu,
            [x_cpu],
            [x_gpu],
        )

    def test_batched_mul_parity(self) -> None:
        """Batched OctonionMulFunction backward: GPU vs CPU at float64."""
        torch.manual_seed(42)
        inputs_cpu, inputs_gpu = _make_pair((4, 8), (4, 8))

        _compare_gradients(
            OctonionMulFunction.apply,
            OctonionMulFunction.apply,
            inputs_cpu,
            inputs_gpu,
        )

    def test_composition_parity(self) -> None:
        """3-operand composition backward: GPU vs CPU at float64.

        Tests (a * b) * c with all three gradients computed.
        """
        torch.manual_seed(42)
        inputs_cpu, inputs_gpu = _make_pair((8,), (8,), (8,))

        def compose_3(a, b, c):
            ab = OctonionMulFunction.apply(a, b)
            return OctonionMulFunction.apply(ab, c)

        _compare_gradients(
            compose_3,
            compose_3,
            inputs_cpu,
            inputs_gpu,
        )
