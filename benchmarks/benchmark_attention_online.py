from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attention.online_softmax import (
    attention_matrix_bytes,
    full_attention_reference,
    online_score_tile_bytes,
    online_softmax_attention_reference,
)
from benchmarks.benchmark_utils import benchmark_cuda_interleaved


def format_bytes(value: int) -> str:
    if value >= 1024**3:
        return f"{value / 1024**3:.2f} GiB"
    if value >= 1024**2:
        return f"{value / 1024**2:.2f} MiB"
    if value >= 1024:
        return f"{value / 1024:.2f} KiB"
    return f"{value} B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialized Attention 与 online-softmax Attention 对比"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--q-block-size", type=int, default=64)
    parser.add_argument("--kv-block-size", type=int, default=128)
    parser.add_argument("--warmup-rounds", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)

    shape = (
        args.batch_size,
        args.heads,
        args.sequence_length,
        args.head_dim,
    )
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def materialized_attention() -> torch.Tensor:
        return full_attention_reference(q, k, v)

    def online_attention() -> torch.Tensor:
        return online_softmax_attention_reference(
            q,
            k,
            v,
            q_block_size=args.q_block_size,
            kv_block_size=args.kv_block_size,
        )

    expected = materialized_attention()
    actual = online_attention()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    print("correctness=online_softmax_attention: passed")

    results = benchmark_cuda_interleaved(
        {
            "Materialized attention reference": materialized_attention,
            "Online-softmax block reference": online_attention,
        },
        warmup_rounds=args.warmup_rounds,
        rounds=args.rounds,
        seed=args.seed,
    )

    full_score_bytes = attention_matrix_bytes(
        batch_size=args.batch_size,
        heads=args.heads,
        query_length=args.sequence_length,
        key_length=args.sequence_length,
    )
    score_tile_bytes = online_score_tile_bytes(
        batch_size=args.batch_size,
        heads=args.heads,
        q_block_size=args.q_block_size,
        kv_block_size=args.kv_block_size,
    )

    print()
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"Q/K/V shape={shape} | dtype=torch.float16")
    print()
    print("Theoretical FP32 attention-matrix storage:")
    print(f"  one full S or P matrix: {format_bytes(full_score_bytes)}")
    print(f"  full S + P matrices:    {format_bytes(2 * full_score_bytes)}")
    print(f"  one online score tile:  {format_bytes(score_tile_bytes)}")
    print()
    print("| Implementation | Median ms | P95 ms |")
    print("|---|---:|---:|")

    for name, result in results.items():
        print(
            f"| {name} | {result.median_ms:.4f} | "
            f"{result.p95_ms:.4f} |"
        )


if __name__ == "__main__":
    main()