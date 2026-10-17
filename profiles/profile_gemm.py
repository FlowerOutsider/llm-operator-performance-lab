from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels.triton.gemm import gemm_naive_triton, gemm_tiled_triton


def profile_case(
    name: str,
    function,
    trace_path: Path,
    iterations: int = 5,
) -> None:
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        for _ in range(iterations):
            with record_function(name):
                function()

        torch.cuda.synchronize()

    profiler.export_chrome_trace(str(trace_path))

    print()
    print(f"===== {name} =====")
    print(f"trace saved: {trace_path}")
    print(
        profiler.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=20,
        )
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(2026)

    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.float16)

    def torch_matmul() -> torch.Tensor:
        return torch.matmul(a, b)

    def triton_naive() -> torch.Tensor:
        return gemm_naive_triton(a, b)

    def triton_tiled() -> torch.Tensor:
        return gemm_tiled_triton(a, b)

    for function in (torch_matmul, triton_naive, triton_tiled):
        for _ in range(10):
            function()
    torch.cuda.synchronize()

    expected = torch_matmul()
    for name, function in {
        "triton_naive": triton_naive,
        "triton_tiled": triton_tiled,
    }.items():
        torch.testing.assert_close(
            function(),
            expected,
            rtol=2e-2,
            atol=2e-2,
        )
        print(f"correctness={name}: passed")

    trace_dir = PROJECT_ROOT / "profiles" / "raw"
    trace_dir.mkdir(parents=True, exist_ok=True)

    for name, function in {
        "gemm_torch_matmul": torch_matmul,
        "gemm_triton_naive": triton_naive,
        "gemm_triton_tiled": triton_tiled,
    }.items():
        profile_case(
            name,
            function,
            trace_dir / f"{name}_trace.json",
        )


if __name__ == "__main__":
    main()