# Fused Residual + RMSNorm 性能研究

## 目标

实现并评估 Transformer 常见的前向计算：

```python
z = x + residual
y = z * rsqrt(mean(z ** 2) + epsilon) * weight
目标是避免中间张量 z 被写入全局显存后，再被 RMSNorm 读取。
环境
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB
- PyTorch: 2.14.1+cu126
- Triton: 3.8.0
- dtype: FP16
- shape: [8, 512, 4096]
- Python: 3.12
每个完整 FP16 张量的大小：
8 × 512 × 4096 × 2 bytes = 32 MiB
实现对比
1. PyTorch eager reference
   手写多个 Tensor 表达式，实现 FP32 中间计算。
2. F.rms_norm(x + residual)
   x + residual 为独立 add 操作；RMSNorm 使用 PyTorch 原生 fused 实现。
3. torch.compile reference
   使用 torch.compile(fullgraph=True, dynamic=False) 自动编译并尝试融合。
4. 手写 Triton fused kernel
   在一个 Triton kernel 内完成 residual add、平方和 reduction、rsqrt、缩放和输出。
正确性
- PyTorch functional、torch.compile、手写 Triton 输出均与 eager reference 通过 torch.testing.assert_close。
- 容差：rtol=2e-3、atol=2e-3。
- 单元测试：7 项全部通过。
基准方法
- 使用 CUDA Events 计时；
- 每轮随机打乱各实现执行顺序，避免固定顺序导致 GPU 时钟、温度或缓存状态偏差；
- 每轮 100 次 warmup，1000 次测量；
- profiler 仅用于确认 kernel 结构，不用于最终微秒级性能结论。
性能结果
Seed	PyTorch eager	F.rms_norm after add	torch.compile	Triton fused
2026	3.0976 ms	0.6357 ms	0.4024 ms	0.3471 ms
2027	3.0985 ms	0.6369 ms	0.4342 ms	0.3736 ms
2028	3.1089 ms	0.6450 ms	0.4465 ms	0.3814 ms


手写 Triton 相比 torch.compile 的中位延迟降低：
Seed	延迟降低
2026	13.7%
2027	14.0%
2028	14.6%


结论：在该 GPU、PyTorch 和 shape 下，手写 Triton fused Residual + RMSNorm 相比 torch.compile 自动融合版本，稳定实现约 13.7%–14.6% 的中位延迟降低。
Profiler 结构证据
F.rms_norm(x + residual) 的 profiler 显示两个主要 CUDA kernel：
aten::add                        315.858 us / call
aten::_fused_rms_norm            307.200 us / call
其中：
- aten::add 为完整中间张量 z 分配约 32 MiB / call；
- aten::_fused_rms_norm 是 RMSNorm 内部融合 kernel；
- 因此原生 functional 写法仍存在 z 的全局显存写入与重读。
而 torch.compile 显示一个自动生成的融合 kernel：
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0
手写实现也显示一个 kernel：
_residual_rmsnorm_forward_kernel
二者都避免了将中间张量 z 作为跨 kernel Tensor 物化到全局显存。
局限
- 结论仅适用于当前硬件、软件版本、输入 shape 和 FP16；
- 当前 Triton 实现仅支持 forward，不支持 autograd backward；
- profiler 会改变运行行为，因此最终比较以随机交错 CUDA Event 基准为准；
- 生产环境仍应优先评估框架原生实现与 torch.compile，仅在热点路径且有稳定收益时维护自定义 kernel。