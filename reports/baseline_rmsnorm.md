# LLM Operator Performance Lab

面向 LLM 热点算子的 GPU 性能分析与优化实验仓库。

当前完成：FP16 RMSNorm 的 PyTorch eager、PyTorch native、`torch.compile` 与 Triton fused 实现对比。

## 已验证结果

硬件与输入：

- GPU：NVIDIA GeForce RTX 3060 Laptop GPU，6 GiB
- PyTorch：2.14.1+cu126
- Triton：3.8.0
- 输入：FP16，`(8, 512, 4096)`

最终使用随机交错 CUDA Event benchmark 对比 `torch.compile` 与手写 Triton：

| 随机种子 | torch.compile | Triton fused | Triton 加速比 |
|---:|---:|---:|---:|
| 2026 | 0.2970 ms | 0.2601 ms | 1.142× |
| 2027 | 0.2959 ms | 0.2601 ms | 1.138× |
| 2028 | 0.2928 ms | 0.2588 ms | 1.132× |

结论：手写 Triton fused RMSNorm 相比 `torch.compile` 自动融合版本，稳定获得约 13%～14% 的中位延迟优化。

## 仓库结构

```text
benchmarks/       CUDA Event benchmark 与实现对比
kernels/triton/   手写 Triton kernels
profiles/         PyTorch Profiler 脚本
reports/          实验报告
tests/            正确性和 benchmark 工具测试
```

快速开始
source .venv/bin/activate

python -m unittest discover -s tests -v

python benchmarks/benchmark_rmsnorm_implementations.py

python benchmarks/benchmark_rmsnorm_compile_vs_triton.py \
  --warmup-rounds 100 \
  --rounds 1000
性能实验原则
- 相同 GPU、shape、dtype 与正确性标准下比较；
- benchmark 前 warmup；
- 使用 CUDA Event 计时，而非 CPU wall-clock；
- 对微秒级差异使用随机交错顺序，降低 GPU 时钟、温度、cache 状态带来的顺序偏差；
- 将 Profiler 用于解释 kernel 结构，不直接用它发布最终微秒级性能结论。
当前限制
- Triton RMSNorm 仅支持 forward；
- 输入必须是 contiguous CUDA FP16/BF16 Tensor；
- 不支持 autograd backward；
- 当前结果只对已测试的 GPU、软件版本和输入 shape 有效。

文件 2：`~/real-work/llm-operator-performance-lab/reports/baseline_rmsnorm.md`

# Triton Fused RMSNorm 优化报告

## 目标

实现并验证一个 Triton fused RMSNorm forward kernel，并将它与 PyTorch eager、PyTorch native RMSNorm、`torch.compile` 自动融合版本进行比较。

优化目标不是改变 RMSNorm 数学定义，而是减少 kernel launch 和中间 Tensor 的 global memory 往返。

## 环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| 显存 | 6 GiB |
| PyTorch | 2.14.1+cu126 |
| CUDA Runtime | 12.6 |
| Triton | 3.8.0 |
| dtype | FP16 |
| 输入 shape | `(8, 512, 4096)` |

## RMSNorm 定义

```text
inverse_rms = rsqrt(mean(x²) + ε)
y = x * inverse_rms * weight
参考实现使用 FP32 reduction，最终输出保留 FP16。
实现
PyTorch eager reference
inverse_rms = torch.rsqrt(
    x.float().square().mean(dim=-1, keepdim=True) + epsilon
)
y = (x * inverse_rms.to(dtype=x.dtype)) * weight
该写法会形成多个 eager Tensor 操作，典型执行路径为：
cast → square → mean → add → rsqrt → cast → mul → mul
不同 kernel 之间不能共享 register/shared memory，因此完整中间 Tensor 需要通过 global memory 交接。
PyTorch F.rms_norm
使用官方 functional API：
torch.nn.functional.rms_norm(...)
Profiler 表明该环境下调用了一个 native fused RMSNorm kernel：
aten::_fused_rms_norm
torch.compile
使用：
torch.compile(..., fullgraph=True, dynamic=False)
Profiler 表明 PyTorch 编译器将参考实现融合为一个自动生成的 Triton kernel：
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0
手写 Triton kernel
手写 kernel 的数据流：
global memory 读取 x / weight
→ program 内完成 FP32 square 和 reduction
→ 计算 inverse_rms
→ 计算 x * inverse_rms * weight
→ global memory 写最终 y
Profiler 中每次调用对应一个核心 kernel：
_rmsnorm_forward_kernel
正确性
全部实现均与 PyTorch reference 比较：
rtol = 2e-3
atol = 2e-3
已验证：
- Triton 输出匹配 reference；
- torch.compile 输出匹配 reference；
- non-contiguous 输入被拒绝；
- benchmark 工具的空输入统计被拒绝。
Profiler 观察
5 次调用的平均 CUDA 时间：
实现	核心 CUDA 结构	平均时间
PyTorch eager	多个 mul/pow/mean/to/copy	1.448 ms
F.rms_norm	一个 native fused kernel	0.310 ms
torch.compile	一个自动生成 Triton kernel	0.219 ms
手写 Triton	一个自定义 Triton kernel	0.212 ms


Profiler 用于确认 fusion 结构；Profiler 会改变微秒级 kernel 的运行状态，因此不直接作为最终性能结论。
最终公平 Benchmark
测量契约
- CUDA Event 计时；
- warmup 100 轮；
- 每个实现 1000 次测量；
- 每一轮随机交换 torch.compile 与 Triton 的执行顺序；
- 三个不同随机种子；
- 编译发生在正式计时前；
- 每次测量前均通过数值正确性校验。
结果
随机种子	torch.compile	Triton fused	Triton 加速比
2026	0.2970 ms	0.2601 ms	1.142×
2027	0.2959 ms	0.2601 ms	1.138×
2028	0.2928 ms	0.2588 ms	1.132×


稳定中位结果：
torch.compile = 0.2959 ms
Triton fused = 0.2601 ms
speedup       = 1.138×
结论
1. 相对手写 PyTorch eager reference，Triton fused RMSNorm 有大幅加速；主要原因是将多个 eager 操作合为一个 kernel，避免完整中间 Tensor 的 global memory 往返。
2. PyTorch F.rms_norm 与 torch.compile 已经进行了强力 fusion；因此手写 Triton 的优势不是“从多个 kernel 变成一个 kernel”。
3. 相对 torch.compile，手写 Triton kernel 在当前 GPU、软件版本与输入 shape 下，通过随机交错 benchmark 稳定获得约 13%～14% 中位延迟优化。
4. 当前 num_warps=1/2/4/8 的性能差异不显著，因此保留简单默认值 num_warps=4，不引入缺乏收益依据的 shape-specific 启发式。
限制与后续工作
- 当前仅支持 forward，不支持 autograd backward；
- 结果不应泛化到所有 GPU、shape、dtype 或 PyTorch 版本；
- 后续可扩展：backward kernel、不同 shape sweep、Nsight Compute 的 DRAM throughput / occupancy / warp stall 分析，以及其他 Transformer 算子。