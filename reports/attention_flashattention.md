
# Attention 与 FlashAttention 后端研究

## 目标

理解并对比三类 Attention 实现：

1. 显式物化 score/probability 矩阵的普通 Attention；
2. PyTorch 分块 online-softmax reference；
3. PyTorch SDPA 的 FlashAttention fused backend。

## 环境

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB
- PyTorch: 2.14.1+cu126
- dtype: FP16
- Q/K/V shape: `[1, 8, 1024, 64]`

## 普通 Attention

```python
S = Q @ K.transpose(-2, -1) / sqrt(D)
P = softmax(S)
O = P @ V
```

其中：

```
S / P shape: [B, H, Lq, Lk]
```

本实验中一个 FP32 `S` 或 `P` 为：

```
1 × 8 × 1024 × 1024 × 4 bytes = 32 MiB
```

完整 `S + P` 为约 64 MiB。

Online Softmax
--------------

对每个 Q/K/V block，维护：

```
m   = 当前最大 score
l   = 当前 exp(score - m) 的和
acc = 当前 exp(score - m) × V 的加权和
```

分块更新：

```
m_new   = max(m_old, max(S_block))
alpha   = exp(m_old - m_new)
l_new   = alpha × l_old + sum(exp(S_block - m_new))
acc_new = alpha × acc_old + exp(S_block - m_new) @ V_block
O       = acc_final / l_final
```

该算法与完整 Softmax 数学等价，但不需要构造完整 `[B, H, Lq, Lk]` 矩阵。

Online reference 结果
-------------------

配置：

```
Q block = 64
KV block = 128
```

完整 attention matrix 中间量：

```
S + P: 64 MiB
```

单个 online score tile：

```
1 × 8 × 64 × 128 × 4 bytes = 256 KiB
```

性能：

| Implementation | Median latency |
| --- |  --- |
| Materialized attention reference | 1.1160 ms |
| --- |  --- |
| PyTorch online-softmax block reference | 35.0368 ms |

online reference 更慢约 31×，原因是：

-   `16 × 8 = 128` 个 Q/KV block 组合；
-   每个 block 都触发小 GEMM、reduction、exp、状态更新等多个 kernel；
-   PyTorch block Tensor 仍会在 kernel 边界物化到全局显存；
-   K/V block 的类型转换和读取会重复发生。

结论：online softmax 算法本身降低峰值中间存储，但未融合的 PyTorch 原型不等于高性能 FlashAttention。

FlashAttention backend
----------------------

使用：

```
F.scaled_dot_product_attention(q, k, v)
```

并用：

```
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    ...
```

强制指定 FlashAttention backend。

三次随机交错 CUDA Event 测量：

| Seed | Materialized | SDPA default | Forced FlashAttention |
| --- |  --- |  --- |  --- |
| 2026 | 1.1131 ms | 0.1452 ms | 0.1489 ms |
| --- |  --- |  --- |  --- |
| 2027 | 1.0711 ms | 0.1390 ms | 0.1509 ms |
| 2028 | 1.0301 ms | 0.1423 ms | 0.1505 ms |

强制 FlashAttention 相比 materialized reference：

```
6.84× ～ 7.48× 加速
```

Profiler 证据
-----------

默认 SDPA 与强制 FlashAttention 均显示：

```
aten::_flash_attention_forward
pytorch_flash::flash_fwd_kernel<...>
```

因此当前环境中默认 SDPA 实际选择了 FlashAttention backend。

平均 kernel CUDA 时间：

| 路径 | Flash kernel CUDA time |
| --- |  --- |
| SDPA default | 143.539 us |
| --- |  --- |
| Forced FlashAttention | 143.776 us |

materialized reference 在 5 次调用中有 40 次 kernel launch，约每次 8 个 CUDA kernel。

FlashAttention 路径在 5 次调用中有 5 次 kernel launch，约每次一个主要 fused kernel。

FlashAttention 的 `CUDA Mem` 约为 5.16 MiB / 5 次，主要对应输出 O 和辅助统计；没有观察到完整 S/P Tensor 的分配。

结论
--

FlashAttention 的收益来自两个条件同时成立：

1.  使用 online softmax，避免完整 S/P 的全局显存物化；
2.  将块循环、状态更新和矩阵计算融合到高效 GPU kernel 中。

仅做 Python/PyTorch 分块循环可以降低理论峰值中间存储，但由于大量小 kernel 与 HBM 往返，可能比普通 Attention 更慢。

注意
--

PyTorch profiler 的 CUDA 时间包含 kernel 的整体 GPU 执行时间，包括计算、HBM 读写影响和硬件等待；`CUDA Mem` 表示 Tensor 分配归因，不是 HBM 读写流量。精确 DRAM 带宽与读写字节需要 Nsight Compute 等硬件 profiler。