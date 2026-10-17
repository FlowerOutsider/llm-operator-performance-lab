from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.benchmark_utils import benchmark_cuda_interleaved
from kernels.triton.gemm import gemm_naive_triton, gemm_tiled_triton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="cuBLAS、Triton naive GEMM、Triton tiled GEMM 对比"
    )
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--warmup-rounds", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)

    a = torch.randn(
        args.m,
        args.k,
        device="cuda",
        dtype=torch.float16,
    )
    b = torch.randn(
        args.k,
        args.n,
        device="cuda",
        dtype=torch.float16,
    )

    def torch_matmul() -> torch.Tensor:
        return torch.matmul(a, b)

    def triton_naive() -> torch.Tensor:
        return gemm_naive_triton(a, b)

    def triton_tiled() -> torch.Tensor:
        return gemm_tiled_triton(a, b)

    expected = torch_matmul()
    for name, function in {
        "triton_naive": triton_naive,
        "triton_tiled": triton_tiled,
    }.items():
        actual = function()
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        print(f"correctness={name}: passed")

    results = benchmark_cuda_interleaved(
        {
            "torch.matmul (cuBLAS)": torch_matmul,
            "Triton naive GEMM": triton_naive,
            "Triton tiled GEMM": triton_tiled,
        },
        warmup_rounds=args.warmup_rounds,
        rounds=args.rounds,
        seed=args.seed,
    )

    cublas_median = results["torch.matmul (cuBLAS)"].median_ms
    flop_count = 2 * args.m * args.n * args.k

    print()
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"A=[{args.m}, {args.k}] | B=[{args.k}, {args.n}] | dtype=torch.float16")
    print()
    print("| Implementation | Median ms | P95 ms | TFLOPS | Relative to cuBLAS |")
    print("|---|---:|---:|---:|---:|")

    for name, result in results.items():
        tflops = flop_count / (result.median_ms * 1e-3) / 1e12
        relative = cublas_median / result.median_ms
        print(
            f"| {name} | {result.median_ms:.4f} | "
            f"{result.p95_ms:.4f} | {tflops:.2f} | {relative:.3f}x |"
        )


if __name__ == "__main__":
    main()