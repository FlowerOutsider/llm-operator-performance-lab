from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    residual = torch.randn_like(x)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.float16)

    def eager_reference() -> torch.Tensor:
        return residual_rmsnorm_reference(x, residual, weight)

    def functional_rms_norm() -> torch.Tensor:
        return F.rms_norm(
            x + residual,
            normalized_shape=(shape[-1],),
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

    # 预热并触发 torch.compile 编译；不要把编译时间放进 profiler。
    for function in (
        eager_reference,
        functional_rms_norm,
        compiled,
        triton_fused,
    ):
        for _ in range(10):
            function()
    torch.cuda.synchronize()

    expected = eager_reference()
    for name, function in {
        "functional_rms_norm": functional_rms_norm,
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
        "residual_rmsnorm_eager": eager_reference,
        "residual_rmsnorm_functional": functional_rms_norm,
        "residual_rmsnorm_compile": compiled,
        "residual_rmsnorm_triton": triton_fused,
    }

    for name, function in cases.items():
        profile_case(
            name,
            function,
            trace_dir / f"{name}_trace.json",
        )


if __name__ == "__main__":
    main()