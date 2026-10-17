# TileLang GEMM 学习实验

## 环境

- GPU：NVIDIA GeForce RTX 3060 Laptop GPU（SM 8.6，6 GiB）
- PyTorch：2.14.1 + CUDA 12.6
- TileLang：0.1.15
- 矩阵：FP16 `A[512, 512] @ B[512, 512]`
- 累加：FP32

## 实现结构

TileLang kernel 显式描述了 GEMM 的 GPU 存储层级：

```python
A_shared = T.alloc_shared((128, 32), T.float16)
B_shared = T.alloc_shared((32, 128), T.float16)
C_local = T.alloc_fragment((128, 128), T.float32)

for k in T.Pipelined(T.ceildiv(K, 32), num_stages=3):
    T.copy(..., A_shared)
    T.copy(..., B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

数据路径为：

```
HBM → A/B shared-memory tiles → FP32 fragment accumulators → HBM 输出 C
```

`T.Pipelined(..., num_stages=3)` 表示 global-to-shared 数据搬运与 Tensor Core 计算可以流水重叠。

正确性
---

```
TileLang FP16 GEMM correctness: passed
max_abs_error: 0.000000
```

性能
--

使用预热后的 CUDA Event 计时；每轮随机化实现执行顺序，排除首次 JIT 编译时间。

| 实现 | Median ms | P95 ms | Mean ms | TFLOPS |
| --- |  --- |  --- |  --- |  --- |
| `torch.matmul`（cuBLAS） | 0.0604 | 0.0942 | 0.0622 | 4.44 |
| --- |  --- |  --- |  --- |  --- |
| TileLang tiled GEMM | 0.0803 | 0.1514 | 0.0949 | 3.34 |

TileLang 达到 cuBLAS 约 75% 的中位性能。未进行 tile 尺寸、warp 配置和架构专属策略的 autotuning，因此不以超过 cuBLAS 为目标。

与 Triton / CUTLASS 的关系
----------------------

-   Triton：Python DSL，适合快速编写自定义 GPU kernel。
-   CUTLASS：C++ 模板库，适合在 CUDA 工程中精细配置高性能 GEMM。
-   TileLang：以 tile、shared memory、fragment、pipeline 为核心抽象的 GPU DSL，适合表达矩阵计算与复杂融合算子。

三者共同面对的核心问题都是：如何减少 HBM 数据搬运、增加片上复用，并有效使用 Tensor Core。
