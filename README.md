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


benchmarks/       CUDA Event benchmark 与实现对比
kernels/triton/   手写 Triton kernels
profiles/         PyTorch Profiler 脚本
reports/          实验报告
tests/            正确性和 benchmark 工具测试


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