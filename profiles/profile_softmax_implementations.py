from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels.triton.softmax import softmax_triton


def softmax_reference(x: torch.Tensor) -> torch.Tensor:
    x_fp32 = x.float()
    row_max = x_fp32.max(dim=-1, keepdim=True).values
    numerator = torch.exp(x_fp32 - row_max)
    return (numerator / numerator.sum(dim=-1, keepdim=True)).to(dtype=x.dtype)


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

    shape = (8, 512, 4096)
    x = torch.randn(shape, device="cuda", dtype=torch.float16)

    def eager_reference() -> torch.Tensor:
        return softmax_reference(x)

    def torch_softmax() -> torch.Tensor:
        return torch.softmax(x, dim=-1)

    compiled_reference = torch.compile(
        softmax_reference,
        fullgraph=True,
        dynamic=False,
    )

    def compiled() -> torch.Tensor:
        return compiled_reference(x)

    def triton_fused() -> torch.Tensor:
        return softmax_triton(x)

    # 编译和缓存预热不计入 profiler。
    for function in (
        eager_reference,
        torch_softmax,
        compiled,
        triton_fused,
    ):
        for _ in range(10):
            function()
    torch.cuda.synchronize()

    expected = eager_reference()
    for name, function in {
        "torch_softmax": torch_softmax,
        "torch_compile": compiled,
        "triton_fused": triton_fused,
    }.items():
        torch.testing.assert_close(
            function(),
            expected,
            rtol=2e-3,
            atol=2e-3,
        )
        print(f"correctness={name}: passed")

    trace_dir = PROJECT_ROOT / "profiles" / "raw"
    trace_dir.mkdir(parents=True, exist_ok=True)

    cases = {
        "softmax_eager": eager_reference,
        "softmax_torch": torch_softmax,
        "softmax_compile": compiled,
        "softmax_triton": triton_fused,
    }

    for name, function in cases.items():
        profile_case(
            name,
            function,
            trace_dir / f"{name}_trace.json",
        )


if __name__ == "__main__":
    main()