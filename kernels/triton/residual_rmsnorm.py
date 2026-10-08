from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _residual_rmsnorm_forward_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    y_ptr,
    row_stride,
    hidden_size: tl.constexpr,
    epsilon: tl.constexpr,
    block_size: tl.constexpr,
):
    row_id = tl.program_id(axis=0)
    offsets = tl.arange(0, block_size)
    mask = offsets < hidden_size

    row_start = row_id * row_stride + offsets

    x = tl.load(x_ptr + row_start, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + row_start, mask=mask, other=0.0).to(
        tl.float32
    )
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # z 不写回全局显存，只保留在该 program 的寄存器中。
    z = x + residual

    mean_square = tl.sum(z * z, axis=0) / hidden_size
    inverse_rms = tl.rsqrt(mean_square + epsilon)

    y = z * inverse_rms * weight
    tl.store(y_ptr + row_start, y, mask=mask)


def residual_rmsnorm_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    epsilon: float = 1e-6,
    num_warps: int | None = None,
) -> torch.Tensor:
    """Forward-only fused Residual + RMSNorm.

    Computes:
        z = x + residual
        y = z * rsqrt(mean(z^2) + epsilon) * weight

    x and residual must be contiguous CUDA tensors with matching shape.
    weight is a contiguous 1D tensor whose length is x.shape[-1].
    """

    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("x must use float16 or bfloat16")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.requires_grad:
        raise ValueError("forward-only kernel does not support requires_grad=True")

    if residual.device != x.device:
        raise ValueError("residual must be on the same device as x")
    if residual.dtype != x.dtype:
        raise ValueError("residual must have the same dtype as x")
    if residual.shape != x.shape:
        raise ValueError("residual must have the same shape as x")
    if not residual.is_contiguous():
        raise ValueError("residual must be contiguous")
    if residual.requires_grad:
        raise ValueError(
            "forward-only kernel does not support residual.requires_grad=True"
        )

    hidden_size = x.shape[-1]
    if weight.ndim != 1 or weight.numel() != hidden_size:
        raise ValueError("weight must be 1D with length x.shape[-1]")
    if weight.device != x.device or weight.dtype != x.dtype:
        raise ValueError("weight must match x device and dtype")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous")
    if weight.requires_grad:
        raise ValueError(
            "forward-only kernel does not support weight.requires_grad=True"
        )

    if hidden_size > 65536:
        raise ValueError("hidden_size > 65536 is not supported by this kernel")

    rows = x.numel() // hidden_size
    block_size = triton.next_power_of_2(hidden_size)

    if num_warps is None:
        num_warps = 4 if block_size <= 4096 else 8
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be one of 1, 2, 4, 8")

    y = torch.empty_like(x)
    row_stride = x.stride(-2) if x.ndim >= 2 else hidden_size

    grid = (rows,)
    _residual_rmsnorm_forward_kernel[grid](
        x,
        residual,
        weight,
        y,
        row_stride,
        hidden_size=hidden_size,
        epsilon=epsilon,
        block_size=block_size,
        num_warps=num_warps,
    )
    return y