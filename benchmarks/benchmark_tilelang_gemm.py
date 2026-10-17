import argparse
import random
import statistics
from collections import defaultdict

import torch

from kernels.tilelang.gemm import gemm_tilelang


def summarize(latencies_ms: list[float]) -> tuple[float, float, float]:
    values = sorted(latencies_ms)
    p95_index = int(0.95 * (len(values) - 1))
    return (
        statistics.median(values),
        values[p95_index],
        statistics.mean(values),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--warmup-rounds", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    a = torch.randn((args.size, args.size), device="cuda", dtype=torch.float16)
    b = torch.randn((args.size, args.size), device="cuda", dtype=torch.float16)

    implementations = {
        "torch.matmul (cuBLAS)": lambda: torch.matmul(a, b),
        "TileLang tiled GEMM": lambda: gemm_tilelang(a, b),
    }

    # 先触发 TileLang JIT；该编译时间不计入性能数据。
    tilelang_output = implementations["TileLang tiled GEMM"]()
    torch.testing.assert_close(tilelang_output, a @ b, rtol=1e-2, atol=1e-2)
    print("correctness=tilelang_gemm: passed")

    # 分别预热。
    for implementation in implementations.values():
        for _ in range(args.warmup_rounds):
            implementation()
    torch.cuda.synchronize()

    # 每轮随机化顺序，避免始终先跑某一个实现带来系统性偏差。
    random_generator = random.Random(args.seed)
    latencies_ms = defaultdict(list)

    for _ in range(args.rounds):
        names = list(implementations)
        random_generator.shuffle(names)

        for name in names:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            implementations[name]()
            end.record()
            end.synchronize()

            latencies_ms[name].append(start.elapsed_time(end))

    print()
    print(f"device={torch.cuda.get_device_name(0)}")
    print(
        f"A=[{args.size}, {args.size}] | "
        f"B=[{args.size}, {args.size}] | dtype=torch.float16"
    )
    print(
        f"rounds={args.rounds} | randomized order per round | "
        f"warmup_rounds={args.warmup_rounds}"
    )
    print()
    print("| Implementation | Median ms | P95 ms | Mean ms | TFLOPS |")
    print("|---|---:|---:|---:|---:|")

    operations = 2 * args.size**3
    for name in implementations:
        median_ms, p95_ms, mean_ms = summarize(latencies_ms[name])
        tflops = operations / (median_ms / 1_000) / 1e12
        print(
            f"| {name} | {median_ms:.4f} | {p95_ms:.4f} | "
            f"{mean_ms:.4f} | {tflops:.2f} |"
        )


if __name__ == "__main__":
    main()