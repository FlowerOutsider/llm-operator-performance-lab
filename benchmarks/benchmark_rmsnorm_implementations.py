"""Compare RMSNorm implementations under one benchmark contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as functional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.benchmark_rmsnorm import rmsnorm_reference
from benchmarks.benchmark_utils import BenchmarkResult, benchmark_cuda
from kernels.triton.rmsnorm import rmsnorm_triton


def rmsnorm_torch_functional(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Use PyTorch's official RMSNorm functional API."""
    return functional.rms_norm(
        x,
        normalized_shape=(x.shape[-1],),
        weight=weight,
        eps=epsilon,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    return parser.parse_args()


def assert_matches_reference(
    name: str,
    fn: Callable[[], torch.Tensor],
    reference: torch.Tensor,
) -> None:
    output = fn()
    torch.testing.assert_close(output, reference, rtol=2e-3, atol=2e-3)
    print(f"correctness={name}: passed")


def print_row(
    name: str,
    result: BenchmarkResult,
    eager_median_ms: float,
) -> None:
    speedup = eager_median_ms / result.median_ms
    print(
        f"| {name} | {result.median_ms:.4f} | {result.p95_ms:.4f} | "
        f"{speedup:.2f}x |"
    )


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

    eager_fn = lambda: rmsnorm_reference(x, weight)
    functional_fn = lambda: rmsnorm_torch_functional(x, weight)
    triton_fn = lambda: rmsnorm_triton(x, weight)

    # Compile before benchmark timing. Compilation itself is not a kernel-latency result.
    compiled_fn = torch.compile(
        lambda inputs, scale: rmsnorm_reference(inputs, scale),
        fullgraph=True,
        dynamic=False,
    )
    compiled_call = lambda: compiled_fn(x, weight)
    compiled_call()
    torch.cuda.synchronize()

    assert_matches_reference("torch_eager_reference", eager_fn, reference)
    assert_matches_reference("torch_functional_rms_norm", functional_fn, reference)
    assert_matches_reference("torch_compile_reference", compiled_call, reference)
    assert_matches_reference("triton_fused", triton_fn, reference)

    results = {
        "PyTorch eager reference": benchmark_cuda(
            "torch_eager_reference",
            eager_fn,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "PyTorch F.rms_norm": benchmark_cuda(
            "torch_functional_rms_norm",
            functional_fn,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "torch.compile reference": benchmark_cuda(
            "torch_compile_reference",
            compiled_call,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "Triton fused RMSNorm": benchmark_cuda(
            "triton_fused_rmsnorm",
            triton_fn,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
    }

    eager_median_ms = results["PyTorch eager reference"].median_ms

    print()
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"shape={tuple(x.shape)} | dtype={x.dtype}")
    print()
    print("| Implementation | Median ms | P95 ms | Speedup vs eager |")
    print("|---|---:|---:|---:|")

    for name, result in results.items():
        print_row(name, result, eager_median_ms)


if __name__ == "__main__":
    main()