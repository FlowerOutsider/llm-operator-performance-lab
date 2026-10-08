"""Sweep Triton RMSNorm num_warps for one fixed benchmark contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.benchmark_rmsnorm import rmsnorm_reference
from benchmarks.benchmark_utils import benchmark_cuda
from kernels.triton.rmsnorm import rmsnorm_triton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(2026)

    x = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.hidden_size,
        device="cuda",
        dtype=torch.float16,
    )
    weight = torch.ones(args.hidden_size, device="cuda", dtype=torch.float16)
    reference = rmsnorm_reference(x, weight)

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"shape={tuple(x.shape)} | dtype={x.dtype}")
    print()
    print("| num_warps | median ms | p95 ms | speedup vs eager |")
    print("|---:|---:|---:|---:|")

    eager_result = benchmark_cuda(
        "torch_eager_rmsnorm",
        lambda: rmsnorm_reference(x, weight),
        warmup=args.warmup,
        iterations=args.iterations,
    )

    for num_warps in (1, 2, 4, 8):
        output = rmsnorm_triton(x, weight, num_warps=num_warps)
        torch.testing.assert_close(output, reference, rtol=2e-3, atol=2e-3)

        result = benchmark_cuda(
            f"triton_rmsnorm_warps_{num_warps}",
            lambda: rmsnorm_triton(x, weight, num_warps=num_warps),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        speedup = eager_result.median_ms / result.median_ms

        print(
            f"| {num_warps} | {result.median_ms:.4f} | "
            f"{result.p95_ms:.4f} | {speedup:.2f}x |"
        )


if __name__ == "__main__":
    main()