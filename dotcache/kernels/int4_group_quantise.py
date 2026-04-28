"""Per-group INT4 quantisation for value vectors.

Paper §7 specifies group_size=16; this is the default below. The kernel
parameter is overridable so the appendix-B ablation can sweep g ∈ {16, 32}.

Storage layout (for one block of 16 tokens × 128 dims, group_size=16):
  - Data: 16 × 128 × 4 bits = 1024 bytes (vs 2048 for INT8)
  - Scales: 16 tokens × 8 groups × float16 = 256 bytes
  - Zeros: 16 tokens × 8 groups × float16 = 256 bytes
  - Total: ~1536 bytes per block, i.e. 96 bytes/token (paper §8.5)
    (vs 2048 INT8, 4096 FP16)

Packing: two INT4 values per byte, low nibble first.
"""
from __future__ import annotations

import torch

GROUP_SIZE = 16  # paper §7: g=16 → 128/16 = 8 groups per token


def quantise_int4_grouped(
    values: torch.Tensor,
    group_size: int = GROUP_SIZE,
) -> dict:
    """Quantise values to per-group INT4.

    Args:
        values: [num_tokens, head_dim] float16/float32

    Returns:
        dict with:
            'data_packed': [num_tokens, head_dim // 2] uint8
            'scales': [num_tokens, num_groups] float16
            'zeros': [num_tokens, num_groups] float16
            'group_size': int
            'per_token_error': [num_tokens] float32 tensor, relative ℓ₂ error
            'per_token_abs_error': [num_tokens] float32 tensor, absolute ℓ₂ error
            'error_bound': 0-dim float32 tensor (max per-token relative ℓ₂ error)
            'abs_error_bound': 0-dim float32 tensor (max per-token absolute ℓ₂ error)
    """
    num_tokens, head_dim = values.shape
    num_groups = head_dim // group_size

    # Reshape to [num_tokens, num_groups, group_size]
    grouped = values.float().reshape(num_tokens, num_groups, group_size)

    # Per-group min/max
    g_min = grouped.amin(dim=-1)  # [num_tokens, num_groups]
    g_max = grouped.amax(dim=-1)

    # Scale and zero point for INT4 (0..15 range)
    g_range = (g_max - g_min).clamp(min=1e-8)
    scales = g_range / 15.0
    zeros = g_min

    # Quantise to 0..15
    grouped_shifted = grouped - zeros.unsqueeze(-1)
    grouped_scaled = grouped_shifted / scales.unsqueeze(-1)
    quantised = grouped_scaled.round().clamp(0, 15).to(torch.uint8)

    # Pack two int4 values per byte (low nibble first)
    quantised_flat = quantised.reshape(num_tokens, head_dim)
    packed = (quantised_flat[:, 0::2] & 0x0F) | ((quantised_flat[:, 1::2] & 0x0F) << 4)

    # Compute actual error annotations. The paper operating point reports
    # INT4 value distortion as relative reconstruction error (~5% at g=16),
    # so η_b is dimensionless: ||V - Vhat||_2 / ||V||_2. Keeping absolute
    # error separately avoids losing diagnostics.
    dequant = quantised.float() * scales.unsqueeze(-1) + zeros.unsqueeze(-1)
    flat_source = grouped.reshape(num_tokens, head_dim)
    flat_dequant = dequant.reshape(num_tokens, head_dim)
    per_token_abs_error = (flat_source - flat_dequant).norm(dim=-1)
    value_norm = flat_source.norm(dim=-1).clamp(min=1e-6)
    per_token_error = per_token_abs_error / value_norm
    error_bound = per_token_error.max()
    abs_error_bound = per_token_abs_error.max()

    return {
        "data_packed": packed,
        "scales": scales.half(),
        "zeros": zeros.half(),
        "group_size": group_size,
        "per_token_error": per_token_error,
        "per_token_abs_error": per_token_abs_error,
        "error_bound": error_bound,
        "abs_error_bound": abs_error_bound,
    }


def dequantise_int4_grouped(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Dequantise per-group INT4 back to float16.

    Args:
        packed: [num_tokens, head_dim // 2] uint8
        scales: [num_tokens, num_groups] float16
        zeros: [num_tokens, num_groups] float16

    Returns:
        [num_tokens, head_dim] float16
    """
    num_tokens = packed.shape[0]
    head_dim = packed.shape[1] * 2
    num_groups = head_dim // group_size

    # Unpack: low nibble = even indices, high nibble = odd indices
    low = (packed & 0x0F).to(torch.uint8)
    high = ((packed >> 4) & 0x0F).to(torch.uint8)
    unpacked = torch.stack([low, high], dim=-1).reshape(num_tokens, head_dim)

    # Reshape to groups for dequant
    grouped = unpacked.reshape(num_tokens, num_groups, group_size).float()
    dequant = grouped * scales.float().unsqueeze(-1) + zeros.float().unsqueeze(-1)

    return dequant.reshape(num_tokens, head_dim).half()


def quantise_int4_grouped_block(
    values: torch.Tensor,
    block_size: int = 16,
    group_size: int = GROUP_SIZE,
) -> dict:
    """Quantise a full [kv_heads, N, head_dim] value tensor block-by-block.

    Returns per-block INT4 data + per-block error bounds.
    """
    kv_heads, N, head_dim = values.shape
    num_blocks = N // block_size

    flat_values = values.reshape(kv_heads * N, head_dim)
    result = quantise_int4_grouped(flat_values, group_size)
    per_token_error = result["per_token_error"].reshape(kv_heads, N)
    per_token_abs_error = result["per_token_abs_error"].reshape(kv_heads, N)
    block_errors = per_token_error.reshape(kv_heads, num_blocks, block_size)
    error_bounds = block_errors.amax(dim=2)
    error_sums = block_errors.sum(dim=2)
    abs_error_bounds = per_token_abs_error.reshape(kv_heads, num_blocks, block_size).amax(dim=2)
    return {
        "data_packed": result["data_packed"].reshape(kv_heads, N, head_dim // 2),
        "scales": result["scales"].reshape(kv_heads, N, head_dim // group_size),
        "zeros": result["zeros"].reshape(kv_heads, N, head_dim // group_size),
        "error_bounds": error_bounds,
        "error_sums": error_sums,
        "abs_error_bounds": abs_error_bounds,
        "group_size": group_size,
    }
