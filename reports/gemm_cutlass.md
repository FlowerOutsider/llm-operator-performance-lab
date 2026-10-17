# GEMM：从 Triton Tile 到 CUTLASS Tensor Core

## 目标

实现并对比三类 FP16 GEMM：

1. Triton naive：一个 program 计算一个输出元素。
2. Triton tiled：一个 program 计算一个输出 tile，并通过 `tl.dot` 使用 Tensor Core。
3. CUTLASS：使用 C++ 模板配置 Ampere Tensor Core GEMM。

测试设备：NVIDIA GeForce RTX 3060 Laptop GPU（SM 8.6，6 GiB）
测试形状：`A[512, 512] @ B[512, 512]`，FP16 输入，FP32 累加。

## Triton 基准结果

随机交错执行的 CUDA Event 基准显示：

| 实现 | Median ms | TFLOPS | 相对 cuBLAS |
|---|---:|---:|---:|
| `torch.matmul`（cuBLAS） | 0.0509--0.0530 | 5.07--5.27 | 1.00x |
| Triton naive GEMM | 2.9208--2.9368 | 0.09 | 0.017--0.018x |
| Triton tiled GEMM | 0.0561--0.0696 | 3.86--4.78 | 0.761--0.907x |

结论：单 kernel 不代表高性能。naive 与 tiled 都是单 kernel，但 tiled 通过 block tiling 复用 A/B 数据，并以矩阵 tile 的方式执行 Tensor Core MMA，因此快约数十倍。

## CUTLASS 实现

文件：`kernels/cutlass/cutlass_gemm_sm86.cu`

CUTLASS 配置：

- `cutlass::half_t`：A/B/C 使用 FP16。
- `float`：累加器使用 FP32。
- `cutlass::arch::OpClassTensorOp`：使用 Tensor Core。
- `cutlass::arch::Sm80`：Ampere Tensor Core 指令族；RTX 3060 的 `sm_86` 可执行。
- Threadblock tile：`128 x 128 x 32`。
- Warp tile：`64 x 64 x 32`。
- MMA instruction tile：`16 x 8 x 16`。

初始默认 `device::Gemm` 配置在 RTX 3060 上启动失败，错误为 `CUDA error: invalid argument`。原因是默认 tile 的动态 shared-memory 需求不适合该消费级 Ampere GPU。显式指定较小 tile 后成功运行。

## CUTLASS 结果

单次测得：

```text
max_abs_error=0.0037117
average_ms=0.0260902
tflops=10.2887
```

FP16 输出下约 `3.7e-3` 的最大绝对误差合理。CUTLASS 程序使用 CUDA Event 计算平均延迟；它与 Python 基准的随机交错中位数方法不同，因此结果不应直接做严格横向排名。

核心收获
----

GEMM 优化的关键不是"少写几行代码"，而是让硬件层级持续有工作可做：

1.  从 HBM 连续、合并地加载 tile；
2.  在 shared memory / 寄存器中复用 tile；
3.  用 Tensor Core 完成矩阵乘加；
4.  选择适合具体 GPU shared-memory 容量、寄存器压力和 occupancy 的 tile 配置。

Triton 适合快速编写、研究自定义 kernel；CUTLASS 适合在 C++/CUDA 工程中组合和调优成熟的高性能 GEMM 构件。
