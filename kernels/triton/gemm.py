from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_naive_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 一个 program 只计算一个输出元素 C[m, n]。
    m = tl.program_id(axis=0)
    n = tl.program_id(axis=1)

    acc = 0.0

    for k_start in range(0, K, BLOCK_K):
        offsets_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offsets_k < K

        a = tl.load(
            a_ptr + m * stride_am + offsets_k * stride_ak,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            b_ptr + offsets_k * stride_bk + n * stride_bn,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)

        acc += tl.sum(a * b, axis=0)

    tl.store(c_ptr + m * stride_cm + n * stride_cn, acc)


@triton.jit
def _gemm_tiled_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 一个 program 计算一个 [BLOCK_M, BLOCK_N] 输出 tile。
    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offsets_k = k_start + tl.arange(0, BLOCK_K)

        a = tl.load(
            a_ptr
            + offsets_m[:, None] * stride_am
            + offsets_k[None, :] * stride_ak,
            mask=(offsets_m[:, None] < M) & (offsets_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr
            + offsets_k[:, None] * stride_bk
            + offsets_n[None, :] * stride_bn,
            mask=(offsets_k[:, None] < K) & (offsets_n[None, :] < N),
            other=0.0,
        )

        accumulator += tl.dot(a, b)

    tl.store(
        c_ptr
        + offsets_m[:, None] * stride_cm
        + offsets_n[None, :] * stride_cn,
        accumulator,
        mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
    )


def _validate_inputs(a: torch.Tensor, b: torch.Tensor) -> None:
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("a and b must be CUDA tensors")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D matrices")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("a and b must use float16 or bfloat16")
    if a.dtype != b.dtype:
        raise ValueError("a and b must have the same dtype")
    if a.device != b.device:
        raise ValueError("a and b must be on the same device")
    if a.shape[1] != b.shape[0]:
        raise ValueError("a.shape[1] must equal b.shape[0]")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("a and b must be contiguous")
    if a.requires_grad or b.requires_grad:
        raise ValueError("forward-only kernels do not support requires_grad=True")


def gemm_naive_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Naive GEMM: one Triton program computes one output element."""

    _validate_inputs(a, b)

    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)

    _gemm_naive_kernel[(m, n)](
        a,
        b,
        c,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        M=m,
        N=n,
        K=k,
        BLOCK_K=32,
        num_warps=1,
    )
    return c


def gemm_tiled_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Tiled GEMM: one Triton program computes a 32x32 output tile."""

    _validate_inputs(a, b)

    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)

    block_m = 32
    block_n = 32
    block_k = 32
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

    _gemm_tiled_kernel[grid](
        a,
        b,
        c,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        M=m,
        N=n,
        K=k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return c