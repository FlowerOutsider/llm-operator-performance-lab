from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attention.online_softmax import full_attention_reference
from benchmarks.benchmark_utils import benchmark_cuda_interleaved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialized Attention 与 PyTorch SDPA FlashAttention 对比"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup-rounds", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not torch.backends.cuda.is_flash_attention_available():
        raise RuntimeError("This PyTorch build does not include FlashAttention")

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

    def sdpa_default() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        )

    def sdpa_flash_forced() -> torch.Tensor:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=False,
            )

    expected = materialized_attention()

    for name, function in {
        "sdpa_default": sdpa_default,
        "sdpa_flash_forced": sdpa_flash_forced,
    }.items():
        actual = function()
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
        print(f"correctness={name}: passed")

    results = benchmark_cuda_interleaved(
        {
            "Materialized attention reference": materialized_attention,
            "SDPA default backend": sdpa_default,
            "SDPA forced FlashAttention": sdpa_flash_forced,
        },
        warmup_rounds=args.warmup_rounds,
        rounds=args.rounds,
        seed=args.seed,
    )

    materialized_median = results[
        "Materialized attention reference"
    ].median_ms

    print()
    print(f"torch={torch.__version__}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"Q/K/V shape={shape} | dtype=torch.float16")
    print("FlashAttention compiled:", torch.backends.cuda.is_flash_attention_available())
    print(
        f"rounds={args.rounds} | randomized order per round | "
        f"warmup_rounds={args.warmup_rounds}"
    )
    print()
    print("| Implementation | Median ms | P95 ms | Speedup vs materialized |")
    print("|---|---:|---:|---:|")

    for name, result in results.items():
        speedup = materialized_median / result.median_ms
        print(
            f"| {name} | {result.median_ms:.4f} | "
            f"{result.p95_ms:.4f} | {speedup:.2f}x |"
        )


if __name__ == "__main__":
    main()