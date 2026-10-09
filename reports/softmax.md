# Row-wise Softmax 性能研究

## 目标

实现数值稳定的 row-wise Softmax：


m = max(x, dim=-1)
p = exp(x - m)
y = p / sum(p, dim=-1)
减去每行最大值可避免 exp 溢出，例如 exp(1000)。


环境
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB
- PyTorch: 2.14.1+cu126
- Triton: 3.8.0
- dtype: FP16
- shape: [8, 512, 4096]
- Python: 3.12


实现对比
1. PyTorch eager reference
   显式执行 FP32 转换、max、subtract、exp、sum、divide、输出类型转换。
2. torch.softmax
   PyTorch 原生 CUDA Softmax 实现。
3. torch.compile reference
   对 eager reference 执行图捕获、编译与自动融合。
4. 手写 Triton fused Softmax
   每个 Triton program 处理一行，在同一个 kernel 内完成 max reduction、exp、sum reduction 和归一化。


正确性
- 与 torch.softmax 和 eager reference 对齐；
- 覆盖随机输入、大 logits 数值稳定性、非连续输入拒绝；
- 当前完整单元测试：10 项通过；
- 对比容差：rtol=2e-3、atol=2e-3。


基准方法
- CUDA Events 计时；
- 每轮随机打乱实现顺序；
- warmup 100 轮，测量 1000 轮；
- profiler 仅用于确认 kernel 结构，不作为最终微秒级性能结论。


性能结果
Seed	PyTorch eager	torch.softmax	torch.compile	Triton fused
2026	2.3487 ms	0.3656 ms	0.3327 ms	0.2743 ms
2027	2.3532 ms	0.3690 ms	0.3133 ms	0.2642 ms
2028	2.3540 ms	0.3697 ms	0.3082 ms	0.2621 ms


手写 Triton 相比 torch.compile 的中位延迟降低：
Seed	延迟降低
2026	17.6%
2027	15.7%
2028	15.0%


结论：在当前 GPU、软件版本、FP16 和输入 shape 下，手写 Triton Softmax 相比 torch.compile 自动融合实现，稳定实现约 15.0%–17.6% 的中位延迟降低。
Profiler 结构证据
eager reference 在 5 次调用中出现 35 次 cudaLaunchKernel，约为每次 7 个 CUDA kernel，对应类型转换、max、subtract、exp、sum、divide 和输出转换等步骤。
eager 中的 aten::sub、aten::exp、aten::div 均显示 320 MB / 5 次 CUDA 内存分配，即每次约 64 MiB 的完整 FP32 中间张量。
8 × 512 × 4096 × 4 bytes = 64 MiB
torch.softmax 显示一个 PyTorch 原生 fused Softmax kernel。
torch.compile 显示一个自动融合 kernel：
triton_red_fused__to_copy_div_exp_max_sub_sum_0
手写 Triton 显示一个 kernel：
_softmax_forward_kernel
Triton 路径中只观察到最终 FP16 输出的约 32 MiB 分配；完整中间 Tensor 未被物化到全局显存。

局限
- 结论仅适用于当前硬件、输入 shape、软件版本和 FP16；
- 当前实现只支持 forward；
- 生产代码应优先评估框架原生实现与 torch.compile；
- profiler 改变执行行为，最终性能比较以随机交错 CUDA Event benchmark 为准。