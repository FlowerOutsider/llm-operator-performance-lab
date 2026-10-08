"""PyTorch eager RMSNorm baseline for later CUDA/Triton comparisons."""

from __future__ import annotations

import argparse

import torch

try:
    # 被其他模块导入时，例如 profiles/profile_rmsnorm.py
    from .benchmark_utils import benchmark_cuda, print_result
except ImportError:
    # 直接执行：python benchmarks/benchmark_rmsnorm.py
    from benchmark_utils import benchmark_cuda, print_result


def rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Reference RMSNorm with FP32 reduction for numerical stability."""
    inverse_rms = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + epsilon)
    return (x * inverse_rms.to(dtype=x.dtype)) * weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the eager PyTorch RMSNorm baseline."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hidden_size <= 0:
        raise ValueError("hidden_size must be positive.")

    device = torch.device("cuda")
    dtype = torch.float16
    torch.manual_seed(2026)

    x = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    weight = torch.ones(args.hidden_size, device=device, dtype=dtype)

    result = benchmark_cuda(
        "torch_eager_rmsnorm",
        lambda: rmsnorm_reference(x, weight),
        warmup=args.warmup,
        iterations=args.iterations,
    )

    print(
        "device="
        f"{torch.cuda.get_device_name(device)} | "
        f"shape={tuple(x.shape)} | dtype={dtype}"
    )
    print_result(result)


if __name__ == "__main__":
    main()