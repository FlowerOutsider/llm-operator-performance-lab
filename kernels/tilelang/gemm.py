import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def gemm_tilelang(
    A,
    B,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 32,
):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), T.float16)
    B: T.Tensor((K, N), T.float16)
    C = T.empty((M, N), T.float16)

    with T.Kernel(
        T.ceildiv(N, block_n),
        T.ceildiv(M, block_m),
        threads=128,
    ) as (bx, by):
        # 每个 block 的输入 tile：位于 shared memory。
        A_shared = T.alloc_shared((block_m, block_k), T.float16)
        B_shared = T.alloc_shared((block_k, block_n), T.float16)

        # 每个 block 的输出累加 tile：位于 fragment / 寄存器层级，FP32 累加。
        C_local = T.alloc_fragment((block_m, block_n), T.float32)

        T.clear(C_local)

        # K 维按 block_k 分块；num_stages=3 表示复制与计算流水重叠。
        for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=3):
            T.copy(A[by * block_m, k * block_k], A_shared)
            T.copy(B[k * block_k, bx * block_n], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        # 仅最终结果从 fragment 写回 HBM。
        T.copy(C_local, C[by * block_m, bx * block_n])

    return C


def main() -> None:
    m = n = k = 512
    torch.manual_seed(2026)

    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)

    output = gemm_tilelang(a, b)
    reference = a @ b

    torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)

    max_abs_error = (output.float() - reference.float()).abs().max().item()
    print("TileLang FP16 GEMM correctness: passed")
    print(f"shape: A=[{m}, {k}], B=[{k}, {n}]")
    print(f"max_abs_error: {max_abs_error:.6f}")


if __name__ == "__main__":
    main()