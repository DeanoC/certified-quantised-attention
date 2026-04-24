"""Per-group INT4 quantisation for value vectors.

Storage layout (for one block of 16 tokens × 128 dims, group_size=32):
  - Data: 16 × 128 × 4 bits = 1024 bytes (vs 2048 for INT8)
  - Scales: 16 tokens × 4 groups × float16 = 128 bytes
  - Zeros: 16 tokens × 4 groups × float16 = 128 bytes
  - Total: ~1280 bytes per block (vs 2048 INT8, 4096 FP16)

Packing: two INT4 values per byte, low nibble first.
"""
from __future__ import annotations

import torch

GROUP_SIZE = 32  # 128 / 32 = 4 groups per token


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
            'error_bound': float  (max per-token ℓ₂ error)
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

    # Compute actual error bound
    dequant = quantised.float() * scales.unsqueeze(-1) + zeros.unsqueeze(-1)
    per_token_error = (grouped - dequant).reshape(num_tokens, head_dim).norm(dim=-1)
    error_bound = per_token_error.max().item()

    return {
        "data_packed": packed,
        "scales": scales.half(),
        "zeros": zeros.half(),
        "group_size": group_size,
        "error_bound": error_bound,
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

    all_packed = torch.empty(kv_heads, N, head_dim // 2, dtype=torch.uint8, device=values.device)
    all_scales = torch.empty(kv_heads, N, head_dim // group_size, dtype=torch.float16, device=values.device)
    all_zeros = torch.empty(kv_heads, N, head_dim // group_size, dtype=torch.float16, device=values.device)
    all_errors = torch.empty(kv_heads, num_blocks, dtype=torch.float32, device=values.device)

    for h in range(kv_heads):
        for b in range(num_blocks):
            start = b * block_size
            end = start + block_size
            block_vals = values[h, start:end, :]
            result = quantise_int4_grouped(block_vals, group_size)
            all_packed[h, start:end] = result["data_packed"]
            all_scales[h, start:end] = result["scales"]
            all_zeros[h, start:end] = result["zeros"]
            all_errors[h, b] = result["error_bound"]

    return {
        "data_packed": all_packed,
        "scales": all_scales,
        "zeros": all_zeros,
        "error_bounds": all_errors,
        "group_size": group_size,
    }
