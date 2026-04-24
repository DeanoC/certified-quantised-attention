"""Phase 1: Fused INT8 block scoring + certification via Triton.

For each block of 16 keys:
  1. Load K_int8 (block_size × head_dim bytes)
  2. Dequantise: K_fp32 = K_int8 * scale
  3. Score: logits = K_fp32 @ q  (per-token dot products)
  4. Compute: m_b = max(logits), S_b = sum(exp(logits - m_b))
  5. Write m_b, S_b to output arrays

All in one fused kernel — one pass over K_int8, no intermediate writes.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _score_blocks_int8_kernel(
    # Pointers
    K_int8_ptr,       # [num_blocks * block_size, head_dim] int8
    K_scale_ptr,      # [num_blocks] float32 (per-block symmetric scale)
    Q_ptr,            # [head_dim] float32
    M_b_ptr,          # [num_blocks] float32 output: per-block max score
    S_b_ptr,          # [num_blocks] float32 output: per-block softmax sum
    # Dimensions
    num_blocks: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    q_scale: tl.constexpr,
    # Block parameters
    BLOCK_D: tl.constexpr,  # tile size for head_dim (must be >= head_dim)
):
    """One program instance per block."""
    block_idx = tl.program_id(0)

    # Load query vector (shared across all blocks, but loaded per-block for simplicity)
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim
    q = tl.load(Q_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

    # Load per-block quantisation scale
    k_scale = tl.load(K_scale_ptr + block_idx).to(tl.float32)

    # Base offset for this block's keys
    base_offset = block_idx * block_size * head_dim

    # Compute q·k for each token in the block
    m_b = tl.full((), float("-inf"), dtype=tl.float32)
    s_b = tl.full((), 0.0, dtype=tl.float32)

    for t in range(block_size):
        # Load K_int8[t, :] for this block
        k_offsets = base_offset + t * head_dim + d_offsets
        k_int8 = tl.load(K_int8_ptr + k_offsets, mask=d_mask, other=0).to(tl.float32)
        # Dequantise
        k_fp = k_int8 * k_scale
        # Dot product: q · k
        score = tl.sum(q * k_fp) * q_scale

        # Online max and softmax sum
        new_m = tl.maximum(m_b, score)
        s_b = s_b * tl.exp(m_b - new_m) + tl.exp(score - new_m)
        m_b = new_m

    # Write outputs
    tl.store(M_b_ptr + block_idx, m_b)
    tl.store(S_b_ptr + block_idx, s_b)


def score_blocks_int8(
    K_int8: torch.Tensor,       # [N, head_dim] int8
    K_scale: torch.Tensor,      # [num_blocks] float32
    q: torch.Tensor,            # [head_dim] float32
    block_size: int = 16,
    q_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score all blocks using INT8 keys.

    Returns (m_b, S_b) each [num_blocks] float32.
    """
    assert K_int8.dtype == torch.int8
    assert K_int8.ndim == 2
    N, head_dim = K_int8.shape
    num_blocks = N // block_size
    assert num_blocks * block_size == N, f"N={N} not divisible by block_size={block_size}"

    device = K_int8.device
    m_b = torch.empty(num_blocks, dtype=torch.float32, device=device)
    S_b = torch.empty(num_blocks, dtype=torch.float32, device=device)

    # Triton block dim must be power of 2 >= head_dim
    BLOCK_D = triton.next_power_of_2(head_dim)

    _score_blocks_int8_kernel[(num_blocks,)](
        K_int8, K_scale, q, m_b, S_b,
        num_blocks=num_blocks,
        block_size=block_size,
        head_dim=head_dim,
        q_scale=q_scale,
        BLOCK_D=BLOCK_D,
    )
    return m_b, S_b


def certify_skip_mask(
    m_b: torch.Tensor,          # [num_blocks] float32
    S_b: torch.Tensor,          # [num_blocks] float32
    correction: torch.Tensor,   # [num_blocks] float32: exp(3 * delta_per_block)
    block_epsilon: float = 0.001,
) -> tuple[torch.Tensor, float]:
    """Compute skip mask from INT8 scores.

    Returns (skip_mask [num_blocks] bool, total_mass float).
    """
    m_global = m_b.max()
    residual = S_b * correction * torch.exp(m_b - m_global)
    total_mass = residual.sum()
    skip_mask = (residual / total_mass) < block_epsilon
    return skip_mask, total_mass.item()


def selective_attend(
    keys_fp: torch.Tensor,      # [N, head_dim] float32 or float16
    values_fp: torch.Tensor,    # [N, d_v] float32 or float16
    q: torch.Tensor,            # [head_dim] float32
    skip_mask: torch.Tensor,    # [num_blocks] bool
    block_size: int = 16,
    q_scale: float = 1.0,
) -> torch.Tensor:
    """Attend only to non-skipped blocks. Returns [d_v] output vector."""
    num_blocks = skip_mask.shape[0]
    attend_mask = ~skip_mask
    attend_indices = attend_mask.nonzero(as_tuple=True)[0]

    if attend_indices.numel() == 0:
        return torch.zeros(values_fp.shape[-1], dtype=torch.float32, device=q.device)

    # Gather attended blocks
    keys_bl = keys_fp.reshape(num_blocks, block_size, -1)
    vals_bl = values_fp.reshape(num_blocks, block_size, -1)
    att_keys = keys_bl[attend_indices].reshape(-1, keys_fp.shape[-1]).to(torch.float32)
    att_vals = vals_bl[attend_indices].reshape(-1, values_fp.shape[-1]).to(torch.float32)

    # Standard attention on selected subset
    scores = torch.matmul(att_keys, q.to(torch.float32)) * q_scale
    weights = torch.softmax(scores, dim=0)
    return weights @ att_vals
