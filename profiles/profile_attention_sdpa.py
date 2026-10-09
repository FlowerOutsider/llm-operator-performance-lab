from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile, record_function

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attention.online_softmax import full_attention_reference


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
            row_limit=30,
        )
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not torch.backends.cuda.is_flash_attention_available():
        raise RuntimeError("This PyTorch build does not include FlashAttention")

    torch.manual_seed(2026)

    shape = (1, 8, 1024, 64)
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

    # 预热与 Flash backend 初始化不计入 profiler。
    for function in (
        materialized_attention,
        sdpa_default,
        sdpa_flash_forced,
    ):
        for _ in range(10):
            function()
    torch.cuda.synchronize()

    expected = materialized_attention()
    for name, function in {
        "sdpa_default": sdpa_default,
        "sdpa_flash_forced": sdpa_flash_forced,
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

    for name, function in {
        "attention_materialized": materialized_attention,
        "attention_sdpa_default": sdpa_default,
        "attention_sdpa_flash_forced": sdpa_flash_forced,
    }.items():
        profile_case(
            name,
            function,
            trace_dir / f"{name}_trace.json",
        )


if __name__ == "__main__":
    main()