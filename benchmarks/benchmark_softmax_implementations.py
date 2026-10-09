from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.benchmark_utils import benchmark_cuda_interleaved
from kernels.triton.softmax import softmax_triton


def softmax_reference(x: torch.Tensor) -> torch.Tensor:
    x_fp32 = x.float()
    row_max = x_fp32.max(dim=-1, keepdim=True).values
    numerator = torch.exp(x_fp32 - row_max)
    return (numerator / numerator.sum(dim=-1, keepdim=True)).to(dtype=x.dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="公平比较 row-wise Softmax 的不同实现"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--columns", type=int, default=4096)
    parser.add_argument("--warmup-rounds", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)

    shape = (args.batch_size, args.sequence_length, args.columns)
    x = torch.randn(shape, device="cuda", dtype=torch.float16)

    def eager_reference() -> torch.Tensor:
        return softmax_reference(x)

    def torch_softmax() -> torch.Tensor:
        return torch.softmax(x, dim=-1)

    compiled_reference = torch.compile(
        softmax_reference,
        fullgraph=True,
        dynamic=False,
    )

    def compiled() -> torch.Tensor:
        return compiled_reference(x)

    def triton_fused() -> torch.Tensor:
        return softmax_triton(x)

    expected = eager_reference()
    candidates = {
        "torch_softmax": torch_softmax,
        "torch_compile_reference": compiled,
        "triton_fused_softmax": triton_fused,
    }

    for name, function in candidates.items():
        actual = function()
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
        print(f"correctness={name}: passed")

    functions = {
        "PyTorch eager reference": eager_reference,
        "torch.softmax": torch_softmax,
        "torch.compile reference": compiled,
        "Triton fused Softmax": triton_fused,
    }
    results = benchmark_cuda_interleaved(
        functions,
        warmup_rounds=args.warmup_rounds,
        rounds=args.rounds,
        seed=args.seed,
    )

    eager_median = results["PyTorch eager reference"].median_ms

    print()
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"shape={shape} | dtype=torch.float16")
    print(
        f"rounds={args.rounds} | randomized order per round | "
        f"warmup_rounds={args.warmup_rounds}"
    )
    print()
    print("| Implementation | Median ms | P95 ms | Speedup vs eager |")
    print("|---|---:|---:|---:|")

    for name, result in results.items():
        speedup = eager_median / result.median_ms
        print(
            f"| {name} | {result.median_ms:.4f} | "
            f"{result.p95_ms:.4f} | {speedup:.2f}x |"
        )


if __name__ == "__main__":
    main()