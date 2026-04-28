from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fp16_block_scores_kernel(
    keys_ptr,
    q_ptr,
    block_idx_ptr,
    block_slot_ptr,
    scores_ptr,
    logmass_ptr,
    num_heads: tl.constexpr,
    k_count: tl.constexpr,
    num_scoring_tokens: tl.constexpr,
    key_scratch_tokens: tl.constexpr,
    gqa_group: tl.constexpr,
    q_scale: tl.constexpr,
    keys_s0: tl.constexpr,
    keys_s1: tl.constexpr,
    keys_s2: tl.constexpr,
    q_s0: tl.constexpr,
    q_s1: tl.constexpr,
    idx_s0: tl.constexpr,
    idx_s1: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    h = tl.program_id(0)
    kk = tl.program_id(1)
    tok = tl.arange(0, BLOCK_SIZE)
    d = tl.arange(0, BLOCK_D)
    block_id = tl.load(block_idx_ptr + h * idx_s0 + kk * idx_s1).to(tl.int64)
    valid_block_id = (block_id >= 0) & ((block_id * BLOCK_SIZE) < num_scoring_tokens)
    slot = tl.load(block_slot_ptr + block_id, mask=valid_block_id, other=-1).to(tl.int64)
    token_idx = slot * BLOCK_SIZE + tok
    valid_tok = (
        valid_block_id
        & ((block_id * BLOCK_SIZE + tok) < num_scoring_tokens)
        & (slot >= 0)
        & (token_idx < key_scratch_tokens)
    )
    kv = h // gqa_group
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for d0 in range(0, HEAD_DIM, BLOCK_D):
        d_idx = d0 + d
        d_mask = d_idx < HEAD_DIM
        q = tl.load(q_ptr + h * q_s0 + d_idx * q_s1, mask=d_mask, other=0.0).to(tl.float32)
        k = tl.load(
            keys_ptr
            + kv * keys_s0
            + token_idx[:, None] * keys_s1
            + d_idx[None, :] * keys_s2,
            mask=valid_tok[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(k * q[None, :], axis=1)
    logits = acc * q_scale
    logits = tl.where(valid_tok, logits, -float("inf"))
    m = tl.max(logits, axis=0)
    l = tl.sum(tl.exp(logits - m), axis=0)
    logmass = m + tl.log(tl.maximum(l, 1.0e-30))
    tl.store(scores_ptr + h * out_s0 + kk * out_s1, m)
    tl.store(logmass_ptr + h * out_s0 + kk * out_s1, logmass)


def fp16_block_scores_triton(
    keys: torch.Tensor,
    q_all: torch.Tensor,
    block_indices: torch.Tensor,
    *,
    num_scoring_blocks: int,
    gqa_group: int,
    block_size: int,
    q_scale: float,
    key_block_slots: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP16 selected-block max score and log-mass for ranking checks."""
    if keys.device.type != "cuda" or q_all.device.type != "cuda":
        raise ValueError("fp16_block_scores_triton requires CUDA tensors")
    if block_indices.device.type != "cuda":
        raise ValueError("block_indices must be CUDA")
    if keys.ndim != 3 or q_all.ndim != 2 or block_indices.ndim != 2:
        raise ValueError("invalid tensor ranks for fp16_block_scores_triton")
    num_heads, head_dim = q_all.shape
    if int(block_indices.shape[0]) != int(num_heads):
        raise ValueError("block_indices head dimension must match q_all")
    k_count = int(block_indices.shape[1])
    scores = torch.empty((num_heads, k_count), dtype=torch.float32, device=q_all.device)
    logmass = torch.empty_like(scores)
    if k_count == 0 or num_scoring_blocks <= 0:
        scores.fill_(float("-inf"))
        logmass.fill_(float("-inf"))
        return scores, logmass
    block_d = triton.next_power_of_2(int(head_dim))
    if block_d > 256:
        raise ValueError("fp16_block_scores_triton supports head_dim <= 256")
    keys_c = keys.contiguous()
    q_c = q_all.contiguous()
    idx_c = block_indices.to(dtype=torch.int64).contiguous()
    if key_block_slots is None:
        key_block_slots = torch.arange(
            int(num_scoring_blocks), dtype=torch.int32, device=q_all.device,
        )
    elif int(key_block_slots.numel()) < int(num_scoring_blocks):
        padded_slots = torch.full(
            (int(num_scoring_blocks),), -1, dtype=torch.int32, device=q_all.device,
        )
        padded_slots[: int(key_block_slots.numel())] = key_block_slots.to(
            dtype=torch.int32, device=q_all.device,
        )
        key_block_slots = padded_slots
    slots_c = key_block_slots.to(dtype=torch.int64).contiguous()
    _fp16_block_scores_kernel[(num_heads, k_count)](
        keys_c,
        q_c,
        idx_c,
        slots_c,
        scores,
        logmass,
        num_heads,
        k_count,
        int(num_scoring_blocks) * int(block_size),
        int(keys_c.shape[1]),
        int(gqa_group),
        float(q_scale),
        keys_c.stride(0),
        keys_c.stride(1),
        keys_c.stride(2),
        q_c.stride(0),
        q_c.stride(1),
        idx_c.stride(0),
        idx_c.stride(1),
        scores.stride(0),
        scores.stride(1),
        BLOCK_SIZE=int(block_size),
        HEAD_DIM=int(head_dim),
        BLOCK_D=int(block_d),
    )
    return scores, logmass
