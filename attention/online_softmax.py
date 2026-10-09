from __future__ import annotations

import math

import torch


def _validate_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [B, H, L, D]")

    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if not q.dtype.is_floating_point:
        raise ValueError("q, k, and v must use a floating-point dtype")

    batch_size, heads, _, query_dim = q.shape
    if k.shape[0] != batch_size or k.shape[1] != heads:
        raise ValueError("q and k must have matching batch/head dimensions")
    if v.shape[0] != batch_size or v.shape[1] != heads:
        raise ValueError("q and v must have matching batch/head dimensions")
    if k.shape[-1] != query_dim:
        raise ValueError("q and k must have the same head dimension")
    if v.shape[-2] != k.shape[-2]:
        raise ValueError("k and v must have the same sequence length")


def full_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Materialized scaled dot-product attention reference.

    This explicitly creates the full score matrix S and probability matrix P.
    """

    _validate_attention_inputs(q, k, v)

    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v.float()).to(dtype=q.dtype)


def online_softmax_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_block_size: int = 64,
    kv_block_size: int = 128,
) -> torch.Tensor:
    """Block-wise online-softmax attention reference.

    This is mathematically equivalent to materialized attention, but it never
    constructs a full [B, H, Lq, Lk] score/probability matrix.

    It is intentionally a PyTorch reference implementation, not a fused
    FlashAttention kernel. Its intermediate block tensors can still materialize
    in global memory between PyTorch operations.
    """

    _validate_attention_inputs(q, k, v)

    if q_block_size <= 0 or kv_block_size <= 0:
        raise ValueError("q_block_size and kv_block_size must be positive")

    _, _, query_length, query_dim = q.shape
    key_length = k.shape[-2]
    value_dim = v.shape[-1]
    scale = 1.0 / math.sqrt(query_dim)

    output_blocks: list[torch.Tensor] = []

    for q_start in range(0, query_length, q_block_size):
        q_end = min(q_start + q_block_size, query_length)
        q_block = q[..., q_start:q_end, :].float()

        state_shape = q_block.shape[:-1] + (1,)
        m = torch.full(
            state_shape,
            -float("inf"),
            device=q.device,
            dtype=torch.float32,
        )
        l = torch.zeros(state_shape, device=q.device, dtype=torch.float32)
        acc = torch.zeros(
            q_block.shape[:-1] + (value_dim,),
            device=q.device,
            dtype=torch.float32,
        )

        for kv_start in range(0, key_length, kv_block_size):
            kv_end = min(kv_start + kv_block_size, key_length)

            k_block = k[..., kv_start:kv_end, :].float()
            v_block = v[..., kv_start:kv_end, :].float()

            scores = torch.matmul(
                q_block,
                k_block.transpose(-2, -1),
            ) * scale

            block_max = scores.max(dim=-1, keepdim=True).values
            new_m = torch.maximum(m, block_max)

            rescale_old_state = torch.exp(m - new_m)
            probabilities_unnormalized = torch.exp(scores - new_m)

            l = (
                rescale_old_state * l
                + probabilities_unnormalized.sum(dim=-1, keepdim=True)
            )
            acc = (
                rescale_old_state * acc
                + torch.matmul(probabilities_unnormalized, v_block)
            )
            m = new_m

        output_blocks.append(acc / l)

    return torch.cat(output_blocks, dim=-2).to(dtype=q.dtype)


def attention_matrix_bytes(
    *,
    batch_size: int,
    heads: int,
    query_length: int,
    key_length: int,
    dtype: torch.dtype = torch.float32,
) -> int:
    """Bytes occupied by one full attention-shaped [B, H, Lq, Lk] tensor."""

    element_size = torch.empty((), dtype=dtype).element_size()
    return batch_size * heads * query_length * key_length * element_size


def online_score_tile_bytes(
    *,
    batch_size: int,
    heads: int,
    q_block_size: int,
    kv_block_size: int,
    dtype: torch.dtype = torch.float32,
) -> int:
    """Bytes occupied by one score-shaped [B, H, BLOCK_M, BLOCK_N] tile."""

    element_size = torch.empty((), dtype=dtype).element_size()
    return batch_size * heads * q_block_size * kv_block_size * element_size