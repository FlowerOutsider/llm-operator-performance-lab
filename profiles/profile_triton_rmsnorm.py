"""Capture a PyTorch-profiler trace for fused Triton RMSNorm."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kernels.triton.rmsnorm import rmsnorm_triton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--active-steps", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profiles/raw/rmsnorm_triton_trace.json"),
    )
    return parser.parse_args()


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

    for _ in range(args.warmup):
        rmsnorm_triton(x, weight)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        for _ in range(args.active_steps):
            with record_function("rmsnorm_triton_fused"):
                rmsnorm_triton(x, weight)
        torch.cuda.synchronize()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(args.output))

    print(f"trace saved: {args.output}")
    print(
        profiler.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=15,
        )
    )


if __name__ == "__main__":
    main()