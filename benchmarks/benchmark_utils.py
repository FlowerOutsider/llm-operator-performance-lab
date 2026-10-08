"""Minimal, reproducible CUDA benchmark helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from random import Random
from statistics import mean, median

import torch


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    iterations: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def summarize_latencies(name: str, latencies_ms: Sequence[float]) -> BenchmarkResult:
    """Summarize per-iteration latency measurements."""
    if not latencies_ms:
        raise ValueError("latencies_ms must not be empty.")

    ordered = sorted(latencies_ms)
    p95_index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))

    return BenchmarkResult(
        name=name,
        iterations=len(ordered),
        mean_ms=mean(ordered),
        median_ms=median(ordered),
        p95_ms=ordered[p95_index],
        min_ms=ordered[0],
    )


def benchmark_cuda(
    name: str,
    fn: Callable[[], object],
    *,
    warmup: int = 25,
    iterations: int = 100,
) -> BenchmarkResult:
    """Benchmark one CUDA callable using CUDA events."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark_cuda().")
    if warmup < 0:
        raise ValueError("warmup must be non-negative.")
    if iterations <= 0:
        raise ValueError("iterations must be positive.")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    latencies_ms: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn()
        end.record()

        end.synchronize()
        latencies_ms.append(start.elapsed_time(end))

    return summarize_latencies(name, latencies_ms)


def benchmark_cuda_interleaved(
    functions: Mapping[str, Callable[[], object]],
    *,
    warmup_rounds: int = 25,
    rounds: int = 100,
    seed: int = 2026,
) -> dict[str, BenchmarkResult]:
    """Benchmark implementations in random order every round.

    Interleaving avoids always measuring one implementation after another,
    which can bias results through GPU clock, temperature, and cache state.
    """
    if not functions:
        raise ValueError("functions must not be empty.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark_cuda_interleaved().")
    if warmup_rounds < 0:
        raise ValueError("warmup_rounds must be non-negative.")
    if rounds <= 0:
        raise ValueError("rounds must be positive.")

    names = list(functions)
    random = Random(seed)

    for _ in range(warmup_rounds):
        order = names.copy()
        random.shuffle(order)
        for name in order:
            functions[name]()

    torch.cuda.synchronize()

    latencies_ms: dict[str, list[float]] = {name: [] for name in names}

    for _ in range(rounds):
        order = names.copy()
        random.shuffle(order)

        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            functions[name]()
            end.record()

            end.synchronize()
            latencies_ms[name].append(start.elapsed_time(end))

    return {
        name: summarize_latencies(name, measurements)
        for name, measurements in latencies_ms.items()
    }


def print_result(result: BenchmarkResult) -> None:
    """Print one stable, human-readable benchmark result."""
    print(
        f"{result.name}: "
        f"median={result.median_ms:.4f} ms | "
        f"p95={result.p95_ms:.4f} ms | "
        f"mean={result.mean_ms:.4f} ms | "
        f"min={result.min_ms:.4f} ms | "
        f"iterations={result.iterations}"
    )