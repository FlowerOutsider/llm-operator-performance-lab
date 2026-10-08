"""Profile eager, PyTorch-native, torch.compile, and Triton RMSNorm variants."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.profiler import ProfilerActivity, profile, record_function

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.benchmark_rmsnorm import rmsnorm_reference
from kernels.triton.rmsnorm import rmsnorm_triton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--active-steps", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("profiles/raw/rmsnorm_implementations"),
    )
    return parser.parse_args()


def profile_implementation(
    name: str,
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    active_steps: int,
    output_path: Path,
) -> None:
    """Warm up one implementation, save its trace, then print CUDA hot spots."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        for _ in range(active_steps):
            with record_function(f"rmsnorm/{name}"):
                fn()
        torch.cuda.synchronize()

    profiler.export_chrome_trace(str(output_path))

    print()
    print("=" * 88)
    print(f"implementation={name}")
    print(f"trace saved: {output_path}")
    print(
        profiler.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=15,
        )
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    torch.manual_seed(2026)
    x = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.hidden_size,
        device="cuda",
        dtype=torch.float16,
    )
    weight = torch.ones(args.hidden_size, device="cuda", dtype=torch.float16)

    compiled_reference = torch.compile(
        lambda inputs, scale: rmsnorm_reference(inputs, scale),
        fullgraph=True,
        dynamic=False,
    )

    # 编译发生在 profiler 之外，避免编译耗时污染算子分析。
    compiled_reference(x, weight)
    torch.cuda.synchronize()

    implementations: dict[str, Callable[[], torch.Tensor]] = {
        "torch_eager_reference": lambda: rmsnorm_reference(x, weight),
        "torch_functional_rms_norm": lambda: functional.rms_norm(
            x,
            normalized_shape=(x.shape[-1],),
            weight=weight,
            eps=1e-6,
        ),
        "torch_compile_reference": lambda: compiled_reference(x, weight),
        "triton_fused_rmsnorm": lambda: rmsnorm_triton(x, weight),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"shape={tuple(x.shape)} | dtype={x.dtype}")

    for name, fn in implementations.items():
        profile_implementation(
            name,
            fn,
            warmup=args.warmup,
            active_steps=args.active_steps,
            output_path=args.output_dir / f"{name}.json",
        )


if __name__ == "__main__":
    main()