"""Forward-only fused RMSNorm implemented with Triton."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_forward_kernel(
    x_ptr,
    weight_ptr,
    y_ptr,
    row_stride,
    hidden_size: tl.constexpr,
    epsilon: tl.constexpr,
    block_size: tl.constexpr,
):
    """One Triton program processes one logical RMSNorm row."""
    row_index = tl.program_id(axis=0)
    column_offsets = tl.arange(0, block_size)
    mask = column_offsets < hidden_size

    x = tl.load(
        x_ptr + row_index * row_stride + column_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(weight_ptr + column_offsets, mask=mask, other=0.0).to(
        tl.float32
    )

    mean_square = tl.sum(x * x, axis=0) / hidden_size
    inverse_rms = tl.rsqrt(mean_square + epsilon)
    y = x * inverse_rms * weight

    tl.store(
        y_ptr + row_index * row_stride + column_offsets,
        y,
        mask=mask,
    )


def rmsnorm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    epsilon: float = 1e-6,
    num_warps: int | None = None,
) -> torch.Tensor:
    """Run fused forward RMSNorm.

    The final dimension is normalized. This first implementation is deliberately
    forward-only: it is for inference-style benchmarking, not autograd training.
    """
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor.")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("x must use float16 or bfloat16.")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension.")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous.")
    if weight.ndim != 1:
        raise ValueError("weight must be one-dimensional.")
    if not weight.is_cuda or weight.device != x.device:
        raise ValueError("weight must be on the same CUDA device as x.")
    if weight.dtype != x.dtype:
        raise ValueError("weight dtype must match x dtype.")

    hidden_size = x.shape[-1]
    if hidden_size != weight.numel():
        raise ValueError("weight size must match x.shape[-1].")
    if hidden_size <= 0:
        raise ValueError("hidden size must be positive.")
    if hidden_size > 65536:
        raise ValueError("this educational kernel supports hidden_size <= 65536.")
    if x.requires_grad or weight.requires_grad:
        raise ValueError("rmsnorm_triton is forward-only and has no autograd support.")

    rows = x.numel() // hidden_size
    block_size = triton.next_power_of_2(hidden_size)

    if num_warps is None:
        num_warps = 4 if block_size <= 4096 else 8

    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be one of 1, 2, 4, 8")

    y = torch.empty_like(x)
    row_stride = x.stride(-2) if x.ndim >= 2 else hidden_size

    grid = (rows,)

    _rmsnorm_forward_kernel[grid](
        x,
        weight,
        y,
        x.stride(-2) if x.ndim >= 2 else hidden_size,
        hidden_size=hidden_size,
        epsilon=epsilon,
        block_size=block_size,
        num_warps=num_warps,
    )
    return y