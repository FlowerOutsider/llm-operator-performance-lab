"""Fair, interleaved comparison of torch.compile and fused Triton RMSNorm."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.benchmark_rmsnorm import rmsnorm_reference
from benchmarks.benchmark_utils import benchmark_cuda_interleaved
from kernels.triton.rmsnorm import rmsnorm_triton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmup-rounds", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
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

    compiled_reference = torch.compile(
        lambda inputs, scale: rmsnorm_reference(inputs, scale),
        fullgraph=True,
        dynamic=False,
    )

    # 在正式计时前完成编译。
    compiled_reference(x, weight)
    torch.cuda.synchronize()

    implementations = {
        "torch_compile": lambda: compiled_reference(x, weight),
        "triton_fused": lambda: rmsnorm_triton(x, weight),
    }

    for name, fn in implementations.items():
        torch.testing.assert_close(fn(), reference, rtol=2e-3, atol=2e-3)
        print(f"correctness={name}: passed")

    results = benchmark_cuda_interleaved(
        implementations,
        warmup_rounds=args.warmup_rounds,
        rounds=args.rounds,
        seed=args.seed,
    )

    compile_result = results["torch_compile"]
    triton_result = results["triton_fused"]
    speedup = compile_result.median_ms / triton_result.median_ms

    print()
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"shape={tuple(x.shape)} | dtype={x.dtype}")
    print(f"rounds={args.rounds} | randomized order per round")
    print()
    print("| Implementation | Median ms | P95 ms | Mean ms |")
    print("|---|---:|---:|---:|")
    print(
        f"| torch.compile | {compile_result.median_ms:.4f} | "
        f"{compile_result.p95_ms:.4f} | {compile_result.mean_ms:.4f} |"
    )
    print(
        f"| Triton fused | {triton_result.median_ms:.4f} | "
        f"{triton_result.p95_ms:.4f} | {triton_result.mean_ms:.4f} |"
    )
    print()
    print(f"triton_speedup_vs_compile={speedup:.3f}x")


if __name__ == "__main__":
    main()