from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_forward_kernel(
    x_ptr,
    y_ptr,
    row_stride,
    columns: tl.constexpr,
    block_size: tl.constexpr,
):
    row_id = tl.program_id(axis=0)
    offsets = tl.arange(0, block_size)
    mask = offsets < columns

    row_offsets = row_id * row_stride + offsets
    x = tl.load(x_ptr + row_offsets, mask=mask, other=-float("inf")).to(tl.float32)

    row_max = tl.max(x, axis=0)
    numerator = tl.exp(x - row_max)
    denominator = tl.sum(numerator, axis=0)

    y = numerator / denominator
    tl.store(y_ptr + row_offsets, y, mask=mask)


def softmax_triton(
    x: torch.Tensor,
    *,
    num_warps: int | None = None,
) -> torch.Tensor:
    """Forward-only numerically stable row-wise softmax over the last dimension."""

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

    columns = x.shape[-1]
    if columns > 65536:
        raise ValueError("columns > 65536 is not supported by this kernel")

    rows = x.numel() // columns
    block_size = triton.next_power_of_2(columns)

    if num_warps is None:
        num_warps = 4 if block_size <= 4096 else 8
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be one of 1, 2, 4, 8")

    y = torch.empty_like(x)
    row_stride = x.stride(-2) if x.ndim >= 2 else columns

    _softmax_forward_kernel[(rows,)](
        x,
        y,
        row_stride,
        columns=columns,
        block_size=block_size,
        num_warps=num_warps,
    )
    return y