from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.benchmark_utils import benchmark_cuda_interleaved
from kernels.triton.residual_rmsnorm import residual_rmsnorm_triton


def residual_rmsnorm_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    z = x.float() + residual.float()
    inverse_rms = torch.rsqrt(z.square().mean(dim=-1, keepdim=True) + epsilon)
    return (z * inverse_rms * weight.float()).to(dtype=x.dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="公平比较 Residual + RMSNorm 的不同实现"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmup-rounds", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.float16

    shape = (
        args.batch_size,
        args.sequence_length,
        args.hidden_size,
    )
    x = torch.randn(shape, device=device, dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(args.hidden_size, device=device, dtype=dtype)

    def eager_reference() -> torch.Tensor:
        return residual_rmsnorm_reference(x, residual, weight)

    def functional_rms_norm() -> torch.Tensor:
        # 这里的 residual add 仍是独立的 Tensor 表达式；
        # F.rms_norm 本身通常是原生 fused RMSNorm 实现。
        return F.rms_norm(
            x + residual,
            normalized_shape=(args.hidden_size,),
            weight=weight,
            eps=1e-6,
        )

    compiled_reference = torch.compile(
        residual_rmsnorm_reference,
        fullgraph=True,
        dynamic=False,
    )

    def compiled() -> torch.Tensor:
        return compiled_reference(x, residual, weight)

    def triton_fused() -> torch.Tensor:
        return residual_rmsnorm_triton(x, residual, weight)

    expected = eager_reference()
    candidates = {
        "torch_functional_rms_norm": functional_rms_norm,
        "torch_compile_reference": compiled,
        "triton_fused_residual_rmsnorm": triton_fused,
    }

    for name, function in candidates.items():
        actual = function()
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
        print(f"correctness={name}: passed")

    functions = {
        "PyTorch eager reference": eager_reference,
        "PyTorch F.rms_norm after add": functional_rms_norm,
        "torch.compile reference": compiled,
        "Triton fused Residual + RMSNorm": triton_fused,
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
    print(f"shape={shape} | dtype={dtype}")
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