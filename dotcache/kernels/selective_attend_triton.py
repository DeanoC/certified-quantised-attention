"""Phase 2: Triton selective attention — single-head and multi-head versions.

Multi-head version processes all Q heads in one kernel launch, with each
program handling one Q head. Skip mask is per-head.

Two variants:
  - Float32 keys (original): takes pre-dequantised K
  - INT8 keys (new): takes K_int8 + per-block K_scale, dequantises in-register
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


def _workspace_tensor(
    workspace: dict[str, torch.Tensor] | None,
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if workspace is None:
        return torch.empty(shape, dtype=dtype, device=device)
    current = workspace.get(name)
    needs_alloc = (
        current is None
        or current.dtype != dtype
        or current.device != device
        or tuple(int(x) for x in current.shape) != tuple(int(x) for x in shape)
    )
    if needs_alloc:
        current = torch.empty(shape, dtype=dtype, device=device)
        workspace[name] = current
    return current


def _splitk_target_blocks_per_split(num_blocks: int, *, mixed_values: bool) -> int:
    env_name = (
        "DOTCACHE_MIXEDV_SPLITK_BLOCKS_PER_SPLIT"
        if mixed_values
        else "DOTCACHE_SPLITK_BLOCKS_PER_SPLIT"
    )
    raw = os.environ.get(env_name)
    if raw:
        return max(1, int(raw))
    raw = os.environ.get("DOTCACHE_SPLITK_BLOCKS_PER_SPLIT")
    if raw:
        return max(1, int(raw))
    if num_blocks <= 512:
        return 128
    if num_blocks <= 2048:
        return 256
    return 512


def _splitk_num_splits(num_blocks: int, *, mixed_values: bool) -> int:
    target_blocks_per_split = _splitk_target_blocks_per_split(
        num_blocks, mixed_values=mixed_values,
    )
    ns = max(1, (int(num_blocks) + target_blocks_per_split - 1) // target_blocks_per_split)
    num_splits = 1
    while num_splits < ns:
        num_splits *= 2
    return min(num_splits, max(int(num_blocks), 1))


@triton.jit
def _multihead_selective_attend_kernel(
    # Data — KV heads packed
    K_ptr,           # [num_kv_heads * N, head_dim] float32 contiguous
    V_ptr,           # [num_kv_heads * N, d_v] float32 contiguous
    Q_ptr,           # [num_q_heads, head_dim] float32
    Skip_ptr,        # [num_q_heads, num_blocks] int32
    Out_ptr,         # [num_q_heads, d_v] float32
    # Layout
    N: tl.constexpr,
    stride_k: tl.constexpr,     # head_dim
    stride_v: tl.constexpr,     # d_v
    num_blocks: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    d_v: tl.constexpr,
    q_scale: tl.constexpr,
    num_q_heads: tl.constexpr,
    gqa_group: tl.constexpr,
    # Tiles
    TILE_D: tl.constexpr,
    TILE_V: tl.constexpr,
):
    """One program per Q head. Iterates blocks, skips flagged ones."""
    qh = tl.program_id(0)
    valid = qh < num_q_heads
    if valid:
        kvh = qh // gqa_group
        kv_base = kvh * N

        t_offs = tl.arange(0, block_size)
        d_offs = tl.arange(0, TILE_D)
        v_offs = tl.arange(0, TILE_V)
        v_mask = v_offs < d_v

        # Online softmax state — FP64 per paper §9.9 (avoids catastrophic
        # cancellation when scale_old is very small). Intermediate compute
        # (scores, exp, weights) stays FP32; only the running state is FP64.
        m = tl.full((), float("-inf"), dtype=tl.float64)
        l = tl.full((), 0.0, dtype=tl.float64)
        acc = tl.zeros((TILE_V,), dtype=tl.float64)

        for bid in range(num_blocks):
            skip_val = tl.load(Skip_ptr + qh * num_blocks + bid)
            if skip_val == 0:  # attend
                base_tok = kv_base + bid * block_size

                # Score: q · K for block_size tokens
                scores = tl.zeros((block_size,), dtype=tl.float32)
                row_ptrs = K_ptr + (base_tok + t_offs) * stride_k
                for d_start in range(0, head_dim, TILE_D):
                    d_off = d_start + d_offs
                    dm = d_off < head_dim
                    q_tile = tl.load(Q_ptr + qh * head_dim + d_off, mask=dm, other=0.0).to(tl.float32)
                    k_ptrs = row_ptrs[:, None] + d_off[None, :]
                    k_tile = tl.load(k_ptrs, mask=dm[None, :], other=0).to(tl.float32)
                    scores += tl.sum(k_tile * q_tile[None, :], axis=1)
                scores = scores * q_scale

                # Online softmax update (FP64 state, FP32 intermediates)
                block_max = tl.max(scores)                          # fp32
                new_m = tl.maximum(m, block_max.to(tl.float64))     # fp64
                alpha = tl.exp(m - new_m)                           # fp64
                acc = acc * alpha
                l = l * alpha
                weights = tl.exp(scores - new_m.to(tl.float32))     # fp32 intermediate
                l += tl.sum(weights).to(tl.float64)

                # V accumulation
                v_row_ptrs = V_ptr + (base_tok + t_offs) * stride_v
                v_off = v_offs  # assumes TILE_V >= d_v
                vm = v_off < d_v
                v_ptrs = v_row_ptrs[:, None] + v_off[None, :]
                v_tile = tl.load(v_ptrs, mask=vm[None, :], other=0).to(tl.float32)
                w_v = tl.sum(weights[:, None] * v_tile, axis=0)     # fp32
                acc += w_v.to(tl.float64)
                m = new_m

        # Normalise in fp64; cast down at the store.
        safe_l = tl.where(l > 0.0, l, 1.0)
        output = (acc / safe_l).to(tl.float32)
        tl.store(Out_ptr + qh * d_v + v_offs, output, mask=v_mask)


# ─── INT8 variant: dequantise keys in-register ────────────────────────

@triton.jit
def _multihead_selective_attend_int8_kernel(
    # Data — INT8 keys + scale + zero point (paper §2.3 asymmetric)
    K_int8_ptr,      # [num_kv_heads * N, head_dim] int8 contiguous
    K_scale_ptr,     # [num_kv_heads, num_blocks, head_dim] float32
    K_zp_ptr,        # [num_kv_heads, num_blocks, head_dim] float32
    V_ptr,           # [num_kv_heads * N, d_v] float16 contiguous
    Q_ptr,           # [num_q_heads, head_dim] float32
    Skip_ptr,        # [num_q_heads, num_blocks] int32
    Out_ptr,         # [num_q_heads, d_v] float32
    # Layout
    N: tl.constexpr,
    stride_k: tl.constexpr,     # head_dim
    stride_v: tl.constexpr,     # d_v
    num_blocks: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    d_v: tl.constexpr,
    q_scale: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    gqa_group: tl.constexpr,
    # Tiles
    TILE_D: tl.constexpr,
    TILE_V: tl.constexpr,
):
    """One program per Q head. Loads INT8 keys, dequantises in-register."""
    qh = tl.program_id(0)
    valid = qh < num_q_heads
    if valid:
        kvh = qh // gqa_group
        kv_base = kvh * N

        t_offs = tl.arange(0, block_size)
        d_offs = tl.arange(0, TILE_D)
        v_offs = tl.arange(0, TILE_V)
        v_mask = v_offs < d_v

        # Online softmax state — FP64 per paper §9.9.
        m = tl.full((), float("-inf"), dtype=tl.float64)
        l = tl.full((), 0.0, dtype=tl.float64)
        acc = tl.zeros((TILE_V,), dtype=tl.float64)

        for bid in range(num_blocks):
            skip_val = tl.load(Skip_ptr + qh * num_blocks + bid)
            if skip_val == 0:  # attend this block
                base_tok = kv_base + bid * block_size
                scale_base = (kvh * num_blocks + bid) * head_dim

                # Score: q · (K_int8 * scale + z) for block_size tokens.
                # Paper §2.3 Eq. 1 asymmetric dequant: k̂ = q · s + z.
                # The +z term yields the §5.2 L575 Σ q_c z_c correction.
                scores = tl.zeros((block_size,), dtype=tl.float32)
                row_ptrs = K_int8_ptr + (base_tok + t_offs) * stride_k
                for d_start in range(0, head_dim, TILE_D):
                    d_off = d_start + d_offs
                    dm = d_off < head_dim
                    q_tile = tl.load(Q_ptr + qh * head_dim + d_off, mask=dm, other=0.0).to(tl.float32)
                    ch_scale = tl.load(K_scale_ptr + scale_base + d_off, mask=dm, other=0.0).to(tl.float32)
                    ch_zp = tl.load(K_zp_ptr + scale_base + d_off, mask=dm, other=0.0).to(tl.float32)
                    k_ptrs = row_ptrs[:, None] + d_off[None, :]
                    # Load INT8, cast to float32, dequant in-register: q*s + z
                    k_int8_tile = tl.load(k_ptrs, mask=dm[None, :], other=0)
                    k_tile = k_int8_tile.to(tl.float32) * ch_scale[None, :] + ch_zp[None, :]
                    scores += tl.sum(k_tile * q_tile[None, :], axis=1)
                scores = scores * q_scale

                # Online softmax update (FP64 state, FP32 intermediates)
                block_max = tl.max(scores)
                new_m = tl.maximum(m, block_max.to(tl.float64))
                alpha = tl.exp(m - new_m)
                acc = acc * alpha
                l = l * alpha
                weights = tl.exp(scores - new_m.to(tl.float32))
                l += tl.sum(weights).to(tl.float64)

                # V accumulation (FP16 values → float32 → float64)
                v_row_ptrs = V_ptr + (base_tok + t_offs) * stride_v
                v_off = v_offs
                vm = v_off < d_v
                v_ptrs = v_row_ptrs[:, None] + v_off[None, :]
                v_tile = tl.load(v_ptrs, mask=vm[None, :], other=0).to(tl.float32)
                w_v = tl.sum(weights[:, None] * v_tile, axis=0)
                acc += w_v.to(tl.float64)
                m = new_m

        # Normalise in fp64; cast down at the store.
        safe_l = tl.where(l > 0.0, l, 1.0)
        output = (acc / safe_l).to(tl.float32)
        tl.store(Out_ptr + qh * d_v + v_offs, output, mask=v_mask)


def selective_attend_multihead_int8(
    keys_int8: torch.Tensor,      # [num_kv_heads, N, head_dim] int8
    keys_scale: torch.Tensor,     # [num_kv_heads, num_blocks, head_dim] float32
    keys_zero_points: torch.Tensor,  # [num_kv_heads, num_blocks, head_dim] float32 (paper §2.3)
    values_fp16: torch.Tensor,    # [num_kv_heads, N, d_v] float16
    q_all: torch.Tensor,          # [num_q_heads, head_dim] float32
    skip_mask_i32: torch.Tensor,  # [num_q_heads, num_blocks] int32
    gqa_group: int,
    block_size: int = 16,
    q_scale: float = 1.0,
) -> torch.Tensor:
    """Multi-head selective attention with INT8 keys, dequantised in-register.

    Eliminates the need to materialise a float32 key tensor in VRAM.
    Keys are loaded as INT8 and dequantised per-block inside the kernel.
    Values are read as FP16 and cast to float32 in-register.

    Returns [num_q_heads, d_v] float32.
    """
    num_kv_heads, N, head_dim = keys_int8.shape
    d_v = values_fp16.shape[2]
    num_q_heads = q_all.shape[0]
    num_blocks = N // block_size
    device = keys_int8.device

    K_flat = keys_int8.reshape(num_kv_heads * N, head_dim).contiguous()
    V_flat = values_fp16.reshape(num_kv_heads * N, d_v).contiguous()
    output = torch.empty(num_q_heads, d_v, dtype=torch.float32, device=device)

    TILE_D = triton.next_power_of_2(head_dim)
    TILE_V = triton.next_power_of_2(d_v)

    _multihead_selective_attend_int8_kernel[(num_q_heads,)](
        K_flat, keys_scale.contiguous(), keys_zero_points.contiguous(), V_flat,
        q_all.contiguous(), skip_mask_i32.contiguous(), output,
        N=N,
        stride_k=head_dim,
        stride_v=d_v,
        num_blocks=num_blocks,
        block_size=block_size,
        head_dim=head_dim,
        d_v=d_v,
        q_scale=q_scale,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        gqa_group=gqa_group,
        TILE_D=TILE_D,
        TILE_V=TILE_V,
    )
    return output


# ─── INT8 keys + INT4 per-group values variant ───────────────────────

@triton.jit
def _multihead_selective_attend_int8k_int4v_kernel(
    # INT8 keys (paper §2.3 asymmetric)
    K_int8_ptr,      # [num_kv_heads * N, head_dim] int8 contiguous
    K_scale_ptr,     # [num_kv_heads, num_blocks, head_dim] float32
    K_zp_ptr,        # [num_kv_heads, num_blocks, head_dim] float32
    # INT4 packed values + per-group scales/zeros
    V_packed_ptr,    # [num_kv_heads * N, d_v // 2] uint8 contiguous
    V_scales_ptr,    # [num_kv_heads * N, num_groups] float16 contiguous
    V_zeros_ptr,     # [num_kv_heads * N, num_groups] float16 contiguous
    # Query + skip
    Q_ptr,           # [num_q_heads, head_dim] float32
    Skip_ptr,        # [num_q_heads, num_blocks] int32
    Out_ptr,         # [num_q_heads, d_v] float32
    # Layout
    N: tl.constexpr,
    stride_k: tl.constexpr,
    d_v: tl.constexpr,
    d_v_half: tl.constexpr,     # d_v // 2
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    num_blocks: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    q_scale: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    gqa_group: tl.constexpr,
    TILE_D: tl.constexpr,
    TILE_V: tl.constexpr,
):
    """INT8 keys + INT4 per-group values, both dequantised in-register."""
    qh = tl.program_id(0)
    valid = qh < num_q_heads
    if valid:
        kvh = qh // gqa_group
        kv_base = kvh * N

        t_offs = tl.arange(0, block_size)
        d_offs = tl.arange(0, TILE_D)
        v_offs = tl.arange(0, TILE_V)
        v_mask = v_offs < d_v

        # Online softmax state — FP64 per paper §9.9.
        m = tl.full((), float("-inf"), dtype=tl.float64)
        l = tl.full((), 0.0, dtype=tl.float64)
        acc = tl.zeros((TILE_V,), dtype=tl.float64)

        for bid in range(num_blocks):
            skip_val = tl.load(Skip_ptr + qh * num_blocks + bid)
            if skip_val == 0:
                base_tok = kv_base + bid * block_size

                # ── K scoring (asymmetric INT8 in-register dequant: q*s + z) ──
                scale_base = (kvh * num_blocks + bid) * head_dim
                scores = tl.zeros((block_size,), dtype=tl.float32)
                row_ptrs = K_int8_ptr + (base_tok + t_offs) * stride_k
                for d_start in range(0, head_dim, TILE_D):
                    d_off = d_start + d_offs
                    dm = d_off < head_dim
                    q_tile = tl.load(Q_ptr + qh * head_dim + d_off, mask=dm, other=0.0).to(tl.float32)
                    ch_scale = tl.load(K_scale_ptr + scale_base + d_off, mask=dm, other=0.0).to(tl.float32)
                    ch_zp = tl.load(K_zp_ptr + scale_base + d_off, mask=dm, other=0.0).to(tl.float32)
                    k_ptrs = row_ptrs[:, None] + d_off[None, :]
                    k_int8_tile = tl.load(k_ptrs, mask=dm[None, :], other=0)
                    k_tile = k_int8_tile.to(tl.float32) * ch_scale[None, :] + ch_zp[None, :]
                    scores += tl.sum(k_tile * q_tile[None, :], axis=1)
                scores = scores * q_scale

                # ── Online softmax (FP64 state, FP32 intermediates) ──
                block_max = tl.max(scores)
                new_m = tl.maximum(m, block_max.to(tl.float64))
                alpha = tl.exp(m - new_m)
                acc = acc * alpha
                l = l * alpha
                weights = tl.exp(scores - new_m.to(tl.float32))
                l += tl.sum(weights).to(tl.float64)

                # ── V accumulation (INT4 per-group in-register dequant) ──
                # Load packed uint8 [block_size, d_v//2], unpack to [block_size, d_v]
                # Then apply per-group scale+zero
                # Process in TILE_V chunks over the d_v dimension
                for v_start in range(0, d_v, TILE_V):
                    v_off = v_start + v_offs
                    vm_local = v_off < d_v

                    # Packed index: each byte holds 2 values
                    # Even indices = low nibble, odd indices = high nibble
                    packed_idx = v_off // 2  # [TILE_V]
                    is_high = (v_off % 2)    # 0 for low nibble, 1 for high

                    # Load packed bytes for all tokens in block
                    v_packed_ptrs = V_packed_ptr + (base_tok + t_offs[:, None]) * d_v_half + packed_idx[None, :]
                    packed_bytes = tl.load(v_packed_ptrs, mask=vm_local[None, :], other=0)

                    # Unpack: select low or high nibble
                    low_nibble = packed_bytes & 0x0F
                    high_nibble = (packed_bytes >> 4) & 0x0F
                    unpacked = tl.where(is_high[None, :] == 1, high_nibble, low_nibble)
                    v_int4 = unpacked.to(tl.float32)

                    # Per-group dequant: which group does each v_off belong to?
                    group_idx = v_off // group_size  # [TILE_V]
                    # Load scales and zeros for all tokens × relevant groups
                    scale_ptrs = V_scales_ptr + (base_tok + t_offs[:, None]) * num_groups + group_idx[None, :]
                    zero_ptrs = V_zeros_ptr + (base_tok + t_offs[:, None]) * num_groups + group_idx[None, :]
                    v_scale = tl.load(scale_ptrs, mask=vm_local[None, :], other=0.0).to(tl.float32)
                    v_zero = tl.load(zero_ptrs, mask=vm_local[None, :], other=0.0).to(tl.float32)

                    # Dequantise: val = int4 * scale + zero
                    v_tile = v_int4 * v_scale + v_zero  # [block_size, TILE_V]

                    # Weighted accumulation (cast up to fp64 before add-in)
                    w_v = tl.sum(weights[:, None] * v_tile, axis=0)  # fp32 [TILE_V]
                    w_v64 = w_v.to(tl.float64)

                    # Add to accumulator at the right offset. TILE_V >= d_v in
                    # practice so this is the only iteration; the guard exists
                    # for future tiling.
                    if v_start > 0:
                        acc = tl.where(vm_local, acc + w_v64, acc)
                    else:
                        zeros64 = tl.zeros_like(w_v64)
                        acc = acc + tl.where(vm_local, w_v64, zeros64)

                m = new_m

        safe_l = tl.where(l > 0.0, l, 1.0)
        output = (acc / safe_l).to(tl.float32)
        tl.store(Out_ptr + qh * d_v + v_offs, output, mask=v_mask)


def selective_attend_multihead_int8k_int4v(
    keys_int8: torch.Tensor,          # [num_kv_heads, N, head_dim] int8
    keys_scale: torch.Tensor,         # [num_kv_heads, num_blocks, head_dim] float32
    keys_zero_points: torch.Tensor,   # [num_kv_heads, num_blocks, head_dim] float32 (paper §2.3)
    values_int4_packed: torch.Tensor,  # [num_kv_heads, N, d_v//2] uint8
    values_int4_scales: torch.Tensor,  # [num_kv_heads, N, num_groups] float16
    values_int4_zeros: torch.Tensor,   # [num_kv_heads, N, num_groups] float16
    q_all: torch.Tensor,              # [num_q_heads, head_dim] float32
    skip_mask_i32: torch.Tensor,      # [num_q_heads, num_blocks] int32
    gqa_group: int,
    block_size: int = 16,
    group_size: int = 16,  # paper §7
    q_scale: float = 1.0,
) -> torch.Tensor:
    """INT8 keys + INT4 per-group values, both dequantised in-register.

    Keys: INT8 per-block ASYMMETRIC (paper §2.3), dequant = q · k_scale + k_zp
    Values: INT4 per-group asymmetric, dequant = int4 * v_scale + v_zero

    Returns [num_q_heads, d_v] float32.
    """
    num_kv_heads, N, head_dim = keys_int8.shape
    d_v = values_int4_packed.shape[2] * 2  # packed: d_v // 2
    num_groups = d_v // group_size
    num_q_heads = q_all.shape[0]
    num_blocks = N // block_size
    device = keys_int8.device

    K_flat = keys_int8.reshape(num_kv_heads * N, head_dim).contiguous()
    V_packed_flat = values_int4_packed.reshape(num_kv_heads * N, d_v // 2).contiguous()
    V_scales_flat = values_int4_scales.reshape(num_kv_heads * N, num_groups).contiguous()
    V_zeros_flat = values_int4_zeros.reshape(num_kv_heads * N, num_groups).contiguous()

    output = torch.empty(num_q_heads, d_v, dtype=torch.float32, device=device)

    TILE_D = triton.next_power_of_2(head_dim)
    TILE_V = triton.next_power_of_2(d_v)

    _multihead_selective_attend_int8k_int4v_kernel[(num_q_heads,)](
        K_flat, keys_scale.contiguous(), keys_zero_points.contiguous(),
        V_packed_flat, V_scales_flat, V_zeros_flat,
        q_all.contiguous(), skip_mask_i32.contiguous(), output,
        N=N,
        stride_k=head_dim,
        d_v=d_v,
        d_v_half=d_v // 2,
        num_groups=num_groups,
        group_size=group_size,
        num_blocks=num_blocks,
        block_size=block_size,
        head_dim=head_dim,
        q_scale=q_scale,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        gqa_group=gqa_group,
        TILE_D=TILE_D,
        TILE_V=TILE_V,
    )
    return output


@triton.jit
def _multihead_selective_attend_hybrid_int4v_kernel(
    # INT8 keys (paper §2.3 asymmetric)
    K_int8_ptr,      # [num_kv_heads * N, head_dim] int8 contiguous
    K_scale_ptr,     # [num_kv_heads, num_blocks, head_dim] float32
    K_zp_ptr,        # [num_kv_heads, num_blocks, head_dim] float32
    # FP16 keys for promoted blocks
    K_fp16_ptr,      # [num_kv_heads * K_scratch_tokens, head_dim] float16
    K_block_slots_ptr, # [num_blocks] int32, block id -> FP16 scratch slot
    TopK_mask_ptr,   # [num_q_heads, num_blocks] int32, 1 = use FP16 key
    # INT4 packed values + per-group scales/zeros
    V_packed_ptr,    # [num_kv_heads * N, d_v // 2] uint8 contiguous
    V_scales_ptr,    # [num_kv_heads * N, num_groups] float16 contiguous
    V_zeros_ptr,     # [num_kv_heads * N, num_groups] float16 contiguous
    # Query + skip
    Q_ptr,           # [num_q_heads, head_dim] float32
    Skip_ptr,        # [num_q_heads, num_blocks] int32
    Out_ptr,         # [num_q_heads, d_v] float32
    # Layout
    N: tl.constexpr,
    K_scratch_tokens: tl.constexpr,
    stride_k: tl.constexpr,
    d_v: tl.constexpr,
    d_v_half: tl.constexpr,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    num_blocks: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    q_scale: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    gqa_group: tl.constexpr,
    last_block_valid: tl.constexpr,
    TILE_D: tl.constexpr,
    TILE_V: tl.constexpr,
):
    """Hybrid FP16/INT8 keys + INT4 values.

    This is the paper Phase-2 path: promoted blocks read FP16 keys, tail
    blocks read INT8 keys, and INT4 values are consumed directly.
    """
    qh = tl.program_id(0)
    valid = qh < num_q_heads
    if valid:
        kvh = qh // gqa_group
        kv_base = kvh * N
        kfp16_base = kvh * K_scratch_tokens

        t_offs = tl.arange(0, block_size)
        d_offs = tl.arange(0, TILE_D)
        v_offs = tl.arange(0, TILE_V)
        v_mask = v_offs < d_v

        m = tl.full((), float("-inf"), dtype=tl.float64)
        l = tl.full((), 0.0, dtype=tl.float64)
        # Paper Algorithm 1: FP64 online-softmax scalars, FP32 output
        # accumulator. Do not use FP64 for the value vector.
        acc = tl.zeros((TILE_V,), dtype=tl.float32)

        for bid in range(num_blocks):
            skip_val = tl.load(Skip_ptr + qh * num_blocks + bid)
            if skip_val == 0:
                base_tok = kv_base + bid * block_size
                use_fp16 = tl.load(TopK_mask_ptr + qh * num_blocks + bid)
                k_slot = tl.maximum(tl.load(K_block_slots_ptr + bid), 0)
                fp16_base_tok = kfp16_base + k_slot * block_size

                scale_base = (kvh * num_blocks + bid) * head_dim
                int8_row_ptrs = K_int8_ptr + (base_tok + t_offs) * stride_k
                fp16_row_ptrs = K_fp16_ptr + (fp16_base_tok + t_offs) * stride_k
                scores = tl.zeros((block_size,), dtype=tl.float32)

                for d_start in range(0, head_dim, TILE_D):
                    d_off = d_start + d_offs
                    dm = d_off < head_dim
                    q_tile = tl.load(Q_ptr + qh * head_dim + d_off, mask=dm, other=0.0).to(tl.float32)
                    ch_scale = tl.load(K_scale_ptr + scale_base + d_off, mask=dm, other=0.0).to(tl.float32)
                    ch_zp = tl.load(K_zp_ptr + scale_base + d_off, mask=dm, other=0.0).to(tl.float32)

                    k_int8 = tl.load(int8_row_ptrs[:, None] + d_off[None, :], mask=dm[None, :], other=0)
                    k_tile = k_int8.to(tl.float32) * ch_scale[None, :] + ch_zp[None, :]
                    k_fp16 = tl.load(
                        fp16_row_ptrs[:, None] + d_off[None, :],
                        mask=(dm[None, :] & (use_fp16 == 1)),
                        other=0,
                    ).to(tl.float32)
                    k_tile = tl.where(use_fp16 == 1, k_fp16, k_tile)
                    scores += tl.sum(k_tile * q_tile[None, :], axis=1)
                scores = scores * q_scale

                if bid == num_blocks - 1:
                    valid_tok = t_offs < last_block_valid
                    scores = tl.where(valid_tok, scores, float("-inf"))

                block_max = tl.max(scores)
                new_m = tl.maximum(m, block_max.to(tl.float64))
                alpha = tl.exp(m - new_m)
                acc = acc * alpha.to(tl.float32)
                l = l * alpha
                weights = tl.exp(scores - new_m.to(tl.float32))
                l += tl.sum(weights).to(tl.float64)

                for v_start in range(0, d_v, TILE_V):
                    v_off = v_start + v_offs
                    vm_local = v_off < d_v
                    packed_idx = v_off // 2
                    is_high = v_off % 2

                    v_packed_ptrs = V_packed_ptr + (base_tok + t_offs[:, None]) * d_v_half + packed_idx[None, :]
                    packed_bytes = tl.load(v_packed_ptrs, mask=vm_local[None, :], other=0)
                    low_nibble = packed_bytes & 0x0F
                    high_nibble = (packed_bytes >> 4) & 0x0F
                    unpacked = tl.where(is_high[None, :] == 1, high_nibble, low_nibble)
                    v_int4 = unpacked.to(tl.float32)

                    group_idx = v_off // group_size
                    scale_ptrs = V_scales_ptr + (base_tok + t_offs[:, None]) * num_groups + group_idx[None, :]
                    zero_ptrs = V_zeros_ptr + (base_tok + t_offs[:, None]) * num_groups + group_idx[None, :]
                    v_scale = tl.load(scale_ptrs, mask=vm_local[None, :], other=0.0).to(tl.float32)
                    v_zero = tl.load(zero_ptrs, mask=vm_local[None, :], other=0.0).to(tl.float32)
                    v_tile = v_int4 * v_scale + v_zero

                    w_v = tl.sum(weights[:, None] * v_tile, axis=0).to(tl.float32)
                    if v_start > 0:
                        acc = tl.where(vm_local, acc + w_v, acc)
                    else:
                        acc = acc + tl.where(vm_local, w_v, tl.zeros_like(w_v))

                m = new_m

        safe_l = tl.where(l > 0.0, l, 1.0)
        output = (acc / safe_l.to(tl.float32)).to(tl.float32)
        tl.store(Out_ptr + qh * d_v + v_offs, output, mask=v_mask)


def selective_attend_multihead_hybrid_int4v(
    keys_int8: torch.Tensor,
    keys_scale: torch.Tensor,
    keys_zero_points: torch.Tensor,
    keys_fp16: torch.Tensor,
    topk_mask: torch.Tensor,
    values_int4_packed: torch.Tensor,
    values_int4_scales: torch.Tensor,
    values_int4_zeros: torch.Tensor,
    q_all: torch.Tensor,
    skip_mask_i32: torch.Tensor,
    gqa_group: int,
    block_size: int = 16,
    group_size: int = 16,
    q_scale: float = 1.0,
    last_block_valid: int | None = None,
    key_block_slots: torch.Tensor | None = None,
    workspace: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Hybrid FP16/INT8 keys with INT4 values, matching paper Algorithm 1."""
    num_kv_heads, N, head_dim = keys_int8.shape
    d_v = values_int4_packed.shape[2] * 2
    num_groups = d_v // group_size
    num_q_heads = q_all.shape[0]
    num_blocks = N // block_size
    device = keys_int8.device

    K_int8_flat = keys_int8.reshape(num_kv_heads * N, head_dim).contiguous()
    k_scratch_tokens = keys_fp16.shape[1]
    K_fp16_flat = keys_fp16.reshape(num_kv_heads * k_scratch_tokens, head_dim).contiguous()
    if key_block_slots is None:
        key_block_slots = torch.arange(num_blocks, dtype=torch.int32, device=device)
    V_packed_flat = values_int4_packed.reshape(num_kv_heads * N, d_v // 2).contiguous()
    V_scales_flat = values_int4_scales.reshape(num_kv_heads * N, num_groups).contiguous()
    V_zeros_flat = values_int4_zeros.reshape(num_kv_heads * N, num_groups).contiguous()
    output = _workspace_tensor(
        workspace, "mixedv_output", (num_q_heads, d_v),
        dtype=torch.float32, device=device,
    )

    TILE_D = triton.next_power_of_2(head_dim)
    TILE_V = triton.next_power_of_2(d_v)
    lbv = block_size if last_block_valid is None else int(last_block_valid)

    _multihead_selective_attend_hybrid_int4v_kernel[(num_q_heads,)](
        K_int8_flat,
        keys_scale.contiguous(),
        keys_zero_points.contiguous(),
        K_fp16_flat,
        key_block_slots.contiguous(),
        topk_mask.contiguous(),
        V_packed_flat,
        V_scales_flat,
        V_zeros_flat,
        q_all.contiguous(),
        skip_mask_i32.contiguous(),
        output,
        N=N,
        K_scratch_tokens=k_scratch_tokens,
        stride_k=head_dim,
        d_v=d_v,
        d_v_half=d_v // 2,
        num_groups=num_groups,
        group_size=group_size,
        num_blocks=num_blocks,
        block_size=block_size,
        head_dim=head_dim,
        q_scale=q_scale,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        gqa_group=gqa_group,
        last_block_valid=lbv,
        TILE_D=TILE_D,
        TILE_V=TILE_V,
    )
    return output


@triton.jit
def _multihead_selective_attend_hybrid_mixedv_kernel(
    # INT8 keys (paper §2.3 asymmetric)
    K_int8_ptr,
    K_scale_ptr,
    K_zp_ptr,
    # FP16 keys for promoted blocks
    K_fp16_ptr,
    K_block_slots_ptr,
    TopK_mask_ptr,
    # INT4 packed values + per-group scales/zeros
    V_packed_ptr,
    V_scales_ptr,
    V_zeros_ptr,
    # Compact FP16 value scratch for Rung-2 promoted blocks
    V_fp16_ptr,
    V_fp16_mask_ptr,   # [num_q_heads, num_blocks] int32, 1 = use FP16 value
    V_block_slots_ptr, # [num_blocks] int32, maps block id -> scratch slot
    # Query + skip
    Q_ptr,
    Skip_ptr,
    Out_ptr,
    # Layout
    N: tl.constexpr,
    K_scratch_tokens: tl.constexpr,
    V_scratch_tokens,
    stride_k: tl.constexpr,
    d_v: tl.constexpr,
    d_v_half: tl.constexpr,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    num_blocks: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    q_scale: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    gqa_group: tl.constexpr,
    last_block_valid: tl.constexpr,
    TILE_D: tl.constexpr,
    TILE_V: tl.constexpr,
):
    """Hybrid FP16/INT8 keys with per-block mixed FP16/INT4 values.

    Rung-2 promotes only value blocks whose per-head rho_b * eta_b exceeds
    the paper threshold; all other value blocks remain INT4.
    """
    qh = tl.program_id(0)
    valid = qh < num_q_heads
    if valid:
        kvh = qh // gqa_group
        kv_base = kvh * N
        kfp16_base = kvh * K_scratch_tokens
        v_scratch_base = kvh * V_scratch_tokens

        t_offs = tl.arange(0, block_size)
        d_offs = tl.arange(0, TILE_D)
        v_offs = tl.arange(0, TILE_V)
        v_mask = v_offs < d_v

        m = tl.full((), float("-inf"), dtype=tl.float64)
        l = tl.full((), 0.0, dtype=tl.float64)
        acc = tl.zeros((TILE_V,), dtype=tl.float32)

        for bid in range(num_blocks):
            skip_val = tl.load(Skip_ptr + qh * num_blocks + bid)
            if skip_val == 0:
                base_tok = kv_base + bid * block_size
                use_fp16_key = tl.load(TopK_mask_ptr + qh * num_blocks + bid)
                use_fp16_value = tl.load(V_fp16_mask_ptr + qh * num_blocks + bid)
                k_slot = tl.maximum(tl.load(K_block_slots_ptr + bid), 0)
                fp16_base_tok = kfp16_base + k_slot * block_size

                scale_base = (kvh * num_blocks + bid) * head_dim
                int8_row_ptrs = K_int8_ptr + (base_tok + t_offs) * stride_k
                fp16_row_ptrs = K_fp16_ptr + (fp16_base_tok + t_offs) * stride_k
                scores = tl.zeros((block_size,), dtype=tl.float32)

                for d_start in range(0, head_dim, TILE_D):
                    d_off = d_start + d_offs
                    dm = d_off < head_dim
                    q_tile = tl.load(Q_ptr + qh * head_dim + d_off, mask=dm, other=0.0).to(tl.float32)
                    if use_fp16_key == 1:
                        k_tile = tl.load(
                            fp16_row_ptrs[:, None] + d_off[None, :],
                            mask=dm[None, :],
                            other=0,
                        ).to(tl.float32)
                    else:
                        ch_scale = tl.load(K_scale_ptr + scale_base + d_off, mask=dm, other=0.0).to(tl.float32)
                        ch_zp = tl.load(K_zp_ptr + scale_base + d_off, mask=dm, other=0.0).to(tl.float32)
                        k_int8 = tl.load(int8_row_ptrs[:, None] + d_off[None, :], mask=dm[None, :], other=0)
                        k_tile = k_int8.to(tl.float32) * ch_scale[None, :] + ch_zp[None, :]
                    scores += tl.sum(k_tile * q_tile[None, :], axis=1)
                scores = scores * q_scale

                if bid == num_blocks - 1:
                    valid_tok = t_offs < last_block_valid
                    scores = tl.where(valid_tok, scores, float("-inf"))

                block_max = tl.max(scores)
                new_m = tl.maximum(m, block_max.to(tl.float64))
                alpha = tl.exp(m - new_m)
                acc = acc * alpha.to(tl.float32)
                l = l * alpha
                weights = tl.exp(scores - new_m.to(tl.float32))
                l += tl.sum(weights).to(tl.float64)

                slot = tl.load(V_block_slots_ptr + bid)
                slot = tl.maximum(slot, 0)
                v_fp16_tok_base = v_scratch_base + slot * block_size

                for v_start in range(0, d_v, TILE_V):
                    v_off = v_start + v_offs
                    vm_local = v_off < d_v
                    packed_idx = v_off // 2
                    is_high = v_off % 2

                    v_packed_ptrs = V_packed_ptr + (base_tok + t_offs[:, None]) * d_v_half + packed_idx[None, :]
                    packed_bytes = tl.load(
                        v_packed_ptrs,
                        mask=(vm_local[None, :] & (use_fp16_value == 0)),
                        other=0,
                    )
                    low_nibble = packed_bytes & 0x0F
                    high_nibble = (packed_bytes >> 4) & 0x0F
                    unpacked = tl.where(is_high[None, :] == 1, high_nibble, low_nibble)
                    v_int4 = unpacked.to(tl.float32)

                    group_idx = v_off // group_size
                    scale_ptrs = V_scales_ptr + (base_tok + t_offs[:, None]) * num_groups + group_idx[None, :]
                    zero_ptrs = V_zeros_ptr + (base_tok + t_offs[:, None]) * num_groups + group_idx[None, :]
                    v_scale = tl.load(
                        scale_ptrs,
                        mask=(vm_local[None, :] & (use_fp16_value == 0)),
                        other=0.0,
                    ).to(tl.float32)
                    v_zero = tl.load(
                        zero_ptrs,
                        mask=(vm_local[None, :] & (use_fp16_value == 0)),
                        other=0.0,
                    ).to(tl.float32)
                    v_int4_tile = v_int4 * v_scale + v_zero

                    v_fp16_ptrs = V_fp16_ptr + (v_fp16_tok_base + t_offs[:, None]) * d_v + v_off[None, :]
                    v_fp16_tile = tl.load(
                        v_fp16_ptrs,
                        mask=(vm_local[None, :] & (use_fp16_value == 1)),
                        other=0.0,
                    ).to(tl.float32)
                    v_tile = tl.where(use_fp16_value == 1, v_fp16_tile, v_int4_tile)

                    w_v = tl.sum(weights[:, None] * v_tile, axis=0).to(tl.float32)
                    if v_start > 0:
                        acc = tl.where(vm_local, acc + w_v, acc)
                    else:
                        acc = acc + tl.where(vm_local, w_v, tl.zeros_like(w_v))

                m = new_m

        safe_l = tl.where(l > 0.0, l, 1.0)
        output = (acc / safe_l.to(tl.float32)).to(tl.float32)
        tl.store(Out_ptr + qh * d_v + v_offs, output, mask=v_mask)


def selective_attend_multihead_hybrid_mixedv(
    keys_int8: torch.Tensor,
    keys_scale: torch.Tensor,
    keys_zero_points: torch.Tensor,
    keys_fp16: torch.Tensor,
    topk_mask: torch.Tensor,
    values_int4_packed: torch.Tensor,
    values_int4_scales: torch.Tensor,
    values_int4_zeros: torch.Tensor,
    values_fp16_scratch: torch.Tensor,
    value_fp16_mask: torch.Tensor,
    value_block_slots: torch.Tensor,
    q_all: torch.Tensor,
    skip_mask_i32: torch.Tensor,
    gqa_group: int,
    block_size: int = 16,
    group_size: int = 16,
    q_scale: float = 1.0,
    last_block_valid: int | None = None,
    key_block_slots: torch.Tensor | None = None,
    workspace: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Hybrid keys with INT4 values plus Rung-2 per-block FP16 value fallback."""
    num_kv_heads, N, head_dim = keys_int8.shape
    d_v = values_int4_packed.shape[2] * 2
    num_groups = d_v // group_size
    num_q_heads = q_all.shape[0]
    num_blocks = N // block_size
    device = keys_int8.device
    k_scratch_tokens = keys_fp16.shape[1]
    v_scratch_tokens = values_fp16_scratch.shape[1]

    K_int8_flat = keys_int8.reshape(num_kv_heads * N, head_dim).contiguous()
    K_fp16_flat = keys_fp16.reshape(num_kv_heads * k_scratch_tokens, head_dim).contiguous()
    if key_block_slots is None:
        key_block_slots = torch.arange(num_blocks, dtype=torch.int32, device=device)
    V_packed_flat = values_int4_packed.reshape(num_kv_heads * N, d_v // 2).contiguous()
    V_scales_flat = values_int4_scales.reshape(num_kv_heads * N, num_groups).contiguous()
    V_zeros_flat = values_int4_zeros.reshape(num_kv_heads * N, num_groups).contiguous()
    V_fp16_flat = values_fp16_scratch.reshape(num_kv_heads * v_scratch_tokens, d_v).contiguous()
    output = _workspace_tensor(
        workspace, "mixedv_output", (num_q_heads, d_v),
        dtype=torch.float32, device=device,
    )

    TILE_D = triton.next_power_of_2(head_dim)
    TILE_V = triton.next_power_of_2(d_v)
    lbv = block_size if last_block_valid is None else int(last_block_valid)

    _multihead_selective_attend_hybrid_mixedv_kernel[(num_q_heads,)](
        K_int8_flat,
        keys_scale.contiguous(),
        keys_zero_points.contiguous(),
        K_fp16_flat,
        key_block_slots.contiguous(),
        topk_mask.contiguous(),
        V_packed_flat,
        V_scales_flat,
        V_zeros_flat,
        V_fp16_flat,
        value_fp16_mask.contiguous(),
        value_block_slots.contiguous(),
        q_all.contiguous(),
        skip_mask_i32.contiguous(),
        output,
        N=N,
        K_scratch_tokens=k_scratch_tokens,
        V_scratch_tokens=v_scratch_tokens,
        stride_k=head_dim,
        d_v=d_v,
        d_v_half=d_v // 2,
        num_groups=num_groups,
        group_size=group_size,
        num_blocks=num_blocks,
        block_size=block_size,
        head_dim=head_dim,
        q_scale=q_scale,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        gqa_group=gqa_group,
        last_block_valid=lbv,
        TILE_D=TILE_D,
        TILE_V=TILE_V,
    )
    return output

# ─── Hybrid INT8/FP16 keys variant (top-K FP16 fallback) ─────────────

@triton.jit
def _multihead_selective_attend_hybrid_kernel(
    # INT8 keys (paper §2.3 asymmetric)
    K_int8_ptr,      # [num_kv_heads * N, head_dim] int8
    K_scale_ptr,     # [num_kv_heads, num_blocks, head_dim] float32
    K_zp_ptr,        # [num_kv_heads, num_blocks, head_dim] float32
    # FP16 keys (same layout, only top-K blocks read from here)
    K_fp16_ptr,      # [num_kv_heads * N, head_dim] float16
    # Per-head mask: 1 = use FP16 keys, 0 = use INT8 keys
    TopK_mask_ptr,   # [num_q_heads, num_blocks] int32
    # Values + query + skip
    V_ptr,           # [num_kv_heads * N, d_v] float16
    Q_ptr,           # [num_q_heads, head_dim] float32
    Skip_ptr,        # [num_q_heads, num_blocks] int32
    Out_ptr,         # [num_q_heads, d_v] float32
    # Layout
    N: tl.constexpr,
    stride_k: tl.constexpr,
    stride_v: tl.constexpr,
    num_blocks: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    d_v: tl.constexpr,
    q_scale: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    gqa_group: tl.constexpr,
    # Valid token count within the LAST block (trailing partial block).
    # Positions t_offs >= last_block_valid inside the last block are masked
    # to -inf before the softmax so they contribute zero mass. Pass
    # block_size when the last block is full (no masking).
    last_block_valid: tl.constexpr,
    TILE_D: tl.constexpr,
    TILE_V: tl.constexpr,
):
    """Hybrid: INT8 keys for most blocks, FP16 keys for top-K blocks."""
    qh = tl.program_id(0)
    valid = qh < num_q_heads
    if valid:
        kvh = qh // gqa_group
        kv_base = kvh * N

        t_offs = tl.arange(0, block_size)
        d_offs = tl.arange(0, TILE_D)
        v_offs = tl.arange(0, TILE_V)
        v_mask = v_offs < d_v

        # Online softmax state — FP64 per paper §9.9.
        m = tl.full((), float("-inf"), dtype=tl.float64)
        l = tl.full((), 0.0, dtype=tl.float64)
        acc = tl.zeros((TILE_V,), dtype=tl.float64)

        for bid in range(num_blocks):
            skip_val = tl.load(Skip_ptr + qh * num_blocks + bid)
            if skip_val == 0:
                base_tok = kv_base + bid * block_size
                use_fp16 = tl.load(TopK_mask_ptr + qh * num_blocks + bid)

                # Score: always load INT8 + dequant. For top-K, also load FP16
                # and overwrite. INT8 load is 1 byte/elem (cheap); FP16 load
                # only happens for ~4 blocks out of hundreds (amortised zero cost).
                scores = tl.zeros((block_size,), dtype=tl.float32)
                scale_base = (kvh * num_blocks + bid) * head_dim
                int8_row_ptrs = K_int8_ptr + (base_tok + t_offs) * stride_k
                fp16_row_ptrs = K_fp16_ptr + (base_tok + t_offs) * stride_k

                for d_start in range(0, head_dim, TILE_D):
                    d_off = d_start + d_offs
                    dm = d_off < head_dim
                    q_tile = tl.load(Q_ptr + qh * head_dim + d_off, mask=dm, other=0.0).to(tl.float32)
                    ch_scale = tl.load(K_scale_ptr + scale_base + d_off, mask=dm, other=0.0).to(tl.float32)
                    ch_zp = tl.load(K_zp_ptr + scale_base + d_off, mask=dm, other=0.0).to(tl.float32)

                    # Always load INT8 (cheap — 1 byte per element). Asymmetric
                    # dequant (paper §2.3 Eq. 1): k̂ = q · s + z.
                    k_int8 = tl.load(int8_row_ptrs[:, None] + d_off[None, :], mask=dm[None, :], other=0)
                    k_tile = k_int8.to(tl.float32) * ch_scale[None, :] + ch_zp[None, :]

                    # Conditionally load FP16 for top-K blocks (mask-gated)
                    # For non-top-K blocks, the mask zeros out the FP16 contribution.
                    # The load still happens but data is discarded — the cost is the
                    # memory transaction, which hits cache for the zeros buffer.
                    k_fp16 = tl.load(fp16_row_ptrs[:, None] + d_off[None, :], mask=(dm[None, :] & (use_fp16 == 1)), other=0).to(tl.float32)
                    # If use_fp16, replace int8 result with fp16 result
                    k_tile = tl.where(use_fp16 == 1, k_fp16, k_tile)

                    scores += tl.sum(k_tile * q_tile[None, :], axis=1)
                scores = scores * q_scale

                # Mask out-of-range tokens in the trailing partial block. For
                # non-last blocks or a fully-valid last block (last_block_valid
                # == block_size) this is a no-op. The mask sets invalid
                # positions to -inf *before* block_max so they cannot win the
                # softmax argmax and cannot contribute to l or acc.
                if bid == num_blocks - 1:
                    valid_tok = t_offs < last_block_valid
                    scores = tl.where(valid_tok, scores, float("-inf"))

                block_max = tl.max(scores)
                new_m = tl.maximum(m, block_max.to(tl.float64))
                alpha = tl.exp(m - new_m)
                acc = acc * alpha
                l = l * alpha
                weights = tl.exp(scores - new_m.to(tl.float32))
                l += tl.sum(weights).to(tl.float64)

                v_row_ptrs = V_ptr + (base_tok + t_offs) * stride_v
                v_off = v_offs
                vm = v_off < d_v
                v_ptrs = v_row_ptrs[:, None] + v_off[None, :]
                v_tile = tl.load(v_ptrs, mask=vm[None, :], other=0).to(tl.float32)
                w_v = tl.sum(weights[:, None] * v_tile, axis=0)
                acc += w_v.to(tl.float64)
                m = new_m

        safe_l = tl.where(l > 0.0, l, 1.0)
        output = (acc / safe_l).to(tl.float32)
        tl.store(Out_ptr + qh * d_v + v_offs, output, mask=v_mask)


# ─── Split-K (FlashDecoding-style) hybrid kernel ──────────────────────
#
# The single-program-per-Q-head hybrid kernel above launches only
# `num_q_heads` programs (32 for Llama-3.1-8B), leaving most SMs idle on a
# wide GPU like Blackwell (188 SMs on RTX Pro 6000). Each program walks
# `num_blocks` blocks serially — at 64K context / block_size=16 that's
# 4096 blocks per program.
#
# Split-K partitions the block axis across `num_splits` programs per Q
# head:
#   grid = (num_q_heads * num_splits,)
#   each program handles `blocks_per_split` contiguous blocks
#   each emits a partial (m, l, acc) for its chunk
# Then a tiny reduction kernel merges the partials per Q head using
# online-softmax rules.
#
# State is FP32 (not FP64) — FlashAttention uses FP32 state without
# numerical issues; on Blackwell consumer parts FP64 is ~1/32 FP32 rate.


@triton.jit
def _hybrid_split_k_partial_kernel(
    K_int8_ptr, K_scale_ptr, K_zp_ptr, K_fp16_ptr,
    K_block_slots_ptr, TopK_mask_ptr, V_ptr, Q_ptr, Skip_ptr,
    # Per-split outputs: [num_q_heads, num_splits, ...]
    M_part_ptr,   # [num_q_heads, num_splits] float32
    L_part_ptr,   # [num_q_heads, num_splits] float32
    Acc_part_ptr, # [num_q_heads, num_splits, d_v] float32
    # Layout — per-KV-head token strides (in elements) let the kernel walk
    # a non-contiguous slice of the cache tensor without needing a prior
    # .contiguous() copy. For a contig-packed [kv, N, D] tensor with dim-0
    # stride = N*D, stride_kv_k = N*D. For a slice that keeps the same
    # dim-0 stride as the parent allocation, stride_kv_k = N_alloc*D > N*D.
    # Per-KV strides are passed as runtime args (not constexpr) because
    # their values (allocation size × head_dim) vary across models/caches;
    # making them constexpr would force a Triton recompile per unique stride,
    # without any observed codegen benefit.
    stride_kv_k,     # K_int8: elements between consecutive kv heads
    stride_kv_kfp16, # K_fp16: elements between consecutive kv heads
    stride_kv_v,     # V: elements between consecutive kv heads
    stride_kv_scale, # K_scale: elements between consecutive kv heads
    stride_k: tl.constexpr,        # elements between consecutive tokens in K (= head_dim)
    stride_v: tl.constexpr,        # elements between consecutive tokens in V (= d_v)
    num_blocks: tl.constexpr,
    num_splits: tl.constexpr,
    blocks_per_split: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    d_v: tl.constexpr,
    q_scale: tl.constexpr,
    num_q_heads: tl.constexpr,
    gqa_group: tl.constexpr,
    last_block_valid: tl.constexpr,
    TILE_D: tl.constexpr,
    TILE_V: tl.constexpr,
):
    """One program per (Q head, split). Computes partial online-softmax
    state over [split_start, split_end) blocks; reduction kernel merges."""
    prog = tl.program_id(0)
    qh = prog // num_splits
    sp = prog % num_splits
    valid_q = qh < num_q_heads
    if valid_q:
        kvh = qh // gqa_group
        k_base_elem = kvh * stride_kv_k
        kfp16_base_elem = kvh * stride_kv_kfp16
        v_base_elem = kvh * stride_kv_v

        block_start = sp * blocks_per_split
        block_end = tl.minimum(block_start + blocks_per_split, num_blocks)

        t_offs = tl.arange(0, block_size)
        d_offs = tl.arange(0, TILE_D)
        v_offs = tl.arange(0, TILE_V)
        v_mask = v_offs < d_v

        # FP32 online-softmax state (partial for this split).
        m = tl.full((), float("-inf"), dtype=tl.float32)
        l = tl.full((), 0.0, dtype=tl.float32)
        acc = tl.zeros((TILE_V,), dtype=tl.float32)

        for bid in range(block_start, block_end):
            skip_val = tl.load(Skip_ptr + qh * num_blocks + bid)
            if skip_val == 0:
                base_tok = bid * block_size
                use_fp16 = tl.load(TopK_mask_ptr + qh * num_blocks + bid)
                k_slot = tl.maximum(tl.load(K_block_slots_ptr + bid), 0)
                fp16_base_tok = k_slot * block_size

                scores = tl.zeros((block_size,), dtype=tl.float32)
                scale_base = kvh * stride_kv_scale + bid * head_dim
                int8_row_ptrs = K_int8_ptr + k_base_elem + (base_tok + t_offs) * stride_k
                fp16_row_ptrs = K_fp16_ptr + kfp16_base_elem + (fp16_base_tok + t_offs) * stride_k

                for d_start in range(0, head_dim, TILE_D):
                    d_off = d_start + d_offs
                    dm = d_off < head_dim
                    q_tile = tl.load(Q_ptr + qh * head_dim + d_off, mask=dm, other=0.0).to(tl.float32)
                    ch_scale = tl.load(K_scale_ptr + scale_base + d_off, mask=dm, other=0.0).to(tl.float32)
                    ch_zp = tl.load(K_zp_ptr + scale_base + d_off, mask=dm, other=0.0).to(tl.float32)

                    # Asymmetric INT8 dequant (paper §2.3 Eq. 1): k̂ = q · s + z.
                    k_int8 = tl.load(int8_row_ptrs[:, None] + d_off[None, :], mask=dm[None, :], other=0)
                    k_tile = k_int8.to(tl.float32) * ch_scale[None, :] + ch_zp[None, :]

                    k_fp16 = tl.load(
                        fp16_row_ptrs[:, None] + d_off[None, :],
                        mask=(dm[None, :] & (use_fp16 == 1)),
                        other=0,
                    ).to(tl.float32)
                    k_tile = tl.where(use_fp16 == 1, k_fp16, k_tile)

                    scores += tl.sum(k_tile * q_tile[None, :], axis=1)
                scores = scores * q_scale

                if bid == num_blocks - 1:
                    valid_tok = t_offs < last_block_valid
                    scores = tl.where(valid_tok, scores, float("-inf"))

                block_max = tl.max(scores)
                new_m = tl.maximum(m, block_max)
                alpha = tl.exp(m - new_m)
                acc = acc * alpha
                l = l * alpha
                weights = tl.exp(scores - new_m)
                l += tl.sum(weights)

                v_row_ptrs = V_ptr + v_base_elem + (base_tok + t_offs) * stride_v
                v_off = v_offs
                vm = v_off < d_v
                v_ptrs = v_row_ptrs[:, None] + v_off[None, :]
                v_tile = tl.load(v_ptrs, mask=vm[None, :], other=0).to(tl.float32)
                acc += tl.sum(weights[:, None] * v_tile, axis=0)
                m = new_m

        # Store partials. Empty-split guard: if no attended block in this
        # split, m stays -inf and l stays 0 — reduction handles this (alpha=0
        # for -inf m_i vs finite m_global; l contribution is 0).
        part_idx = qh * num_splits + sp
        tl.store(M_part_ptr + part_idx, m)
        tl.store(L_part_ptr + part_idx, l)
        tl.store(Acc_part_ptr + part_idx * d_v + v_offs, acc, mask=v_mask)


@triton.jit
def _hybrid_split_k_reduce_kernel(
    M_part_ptr, L_part_ptr, Acc_part_ptr,
    Out_ptr,
    num_q_heads: tl.constexpr,
    num_splits: tl.constexpr,
    d_v: tl.constexpr,
    TILE_V: tl.constexpr,
):
    """One program per Q head. Merges `num_splits` partials via
    online-softmax recombination:
      m* = max_i m_i
      scale_i = exp(m_i - m*)
      acc* = sum_i scale_i * acc_i
      l*   = sum_i scale_i * l_i
      out  = acc* / l*
    """
    qh = tl.program_id(0)
    valid = qh < num_q_heads
    if valid:
        v_offs = tl.arange(0, TILE_V)
        v_mask = v_offs < d_v

        # Pass 1: find m*.
        m_global = tl.full((), float("-inf"), dtype=tl.float32)
        for sp in range(num_splits):
            m_i = tl.load(M_part_ptr + qh * num_splits + sp)
            m_global = tl.maximum(m_global, m_i)

        # Pass 2: accumulate acc and l with the global max.
        acc_total = tl.zeros((TILE_V,), dtype=tl.float32)
        l_total = tl.full((), 0.0, dtype=tl.float32)
        for sp in range(num_splits):
            part_idx = qh * num_splits + sp
            m_i = tl.load(M_part_ptr + part_idx)
            l_i = tl.load(L_part_ptr + part_idx)
            # tl.exp on -inf gives 0 (correct — empty split contributes nothing).
            scale = tl.exp(m_i - m_global)
            acc_i = tl.load(Acc_part_ptr + part_idx * d_v + v_offs, mask=v_mask, other=0.0)
            acc_total += acc_i * scale
            l_total += l_i * scale

        safe_l = tl.where(l_total > 0.0, l_total, 1.0)
        out = acc_total / safe_l
        tl.store(Out_ptr + qh * d_v + v_offs, out, mask=v_mask)


@triton.jit
def _hybrid_mixedv_split_k_partial_kernel(
    K_int8_ptr, K_scale_ptr, K_zp_ptr, K_fp16_ptr, K_block_slots_ptr,
    TopK_mask_ptr,
    V_packed_ptr, V_scales_ptr, V_zeros_ptr,
    V_fp16_ptr, V_fp16_mask_ptr, V_block_slots_ptr,
    Q_ptr, Skip_ptr,
    M_part_ptr, L_part_ptr, Acc_part_ptr,
    stride_kv_k,
    stride_kv_kfp16,
    stride_kv_vpack,
    stride_kv_vscale,
    stride_kv_vscratch,
    stride_kv_scale,
    stride_k: tl.constexpr,
    stride_vpack: tl.constexpr,
    stride_vscale: tl.constexpr,
    stride_vscratch: tl.constexpr,
    d_v: tl.constexpr,
    d_v_half: tl.constexpr,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    num_blocks: tl.constexpr,
    num_splits: tl.constexpr,
    blocks_per_split: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    q_scale: tl.constexpr,
    num_q_heads: tl.constexpr,
    gqa_group: tl.constexpr,
    last_block_valid: tl.constexpr,
    TILE_D: tl.constexpr,
    TILE_V: tl.constexpr,
):
    """Split-K variant of the mixed INT4/FP16 value fallback kernel."""
    prog = tl.program_id(0)
    qh = prog // num_splits
    sp = prog % num_splits
    valid_q = qh < num_q_heads
    if valid_q:
        kvh = qh // gqa_group
        k_base_elem = kvh * stride_kv_k
        kfp16_base_elem = kvh * stride_kv_kfp16
        vpack_base_elem = kvh * stride_kv_vpack
        vscale_base_elem = kvh * stride_kv_vscale
        vscratch_base_elem = kvh * stride_kv_vscratch

        block_start = sp * blocks_per_split
        block_end = tl.minimum(block_start + blocks_per_split, num_blocks)

        t_offs = tl.arange(0, block_size)
        d_offs = tl.arange(0, TILE_D)
        v_offs = tl.arange(0, TILE_V)
        v_mask = v_offs < d_v

        m = tl.full((), float("-inf"), dtype=tl.float32)
        l = tl.full((), 0.0, dtype=tl.float32)
        acc = tl.zeros((TILE_V,), dtype=tl.float32)

        for bid in range(block_start, block_end):
            skip_val = tl.load(Skip_ptr + qh * num_blocks + bid)
            if skip_val == 0:
                base_tok = bid * block_size
                use_fp16_key = tl.load(TopK_mask_ptr + qh * num_blocks + bid)
                use_fp16_value = tl.load(V_fp16_mask_ptr + qh * num_blocks + bid)
                k_slot = tl.maximum(tl.load(K_block_slots_ptr + bid), 0)

                scale_base = kvh * stride_kv_scale + bid * head_dim
                int8_row_ptrs = K_int8_ptr + k_base_elem + (base_tok + t_offs) * stride_k
                fp16_row_ptrs = K_fp16_ptr + kfp16_base_elem + (k_slot * block_size + t_offs) * stride_k
                scores = tl.zeros((block_size,), dtype=tl.float32)

                for d_start in range(0, head_dim, TILE_D):
                    d_off = d_start + d_offs
                    dm = d_off < head_dim
                    q_tile = tl.load(Q_ptr + qh * head_dim + d_off, mask=dm, other=0.0).to(tl.float32)
                    ch_scale = tl.load(K_scale_ptr + scale_base + d_off, mask=dm, other=0.0).to(tl.float32)
                    ch_zp = tl.load(K_zp_ptr + scale_base + d_off, mask=dm, other=0.0).to(tl.float32)

                    k_int8 = tl.load(int8_row_ptrs[:, None] + d_off[None, :], mask=dm[None, :], other=0)
                    k_tile = k_int8.to(tl.float32) * ch_scale[None, :] + ch_zp[None, :]
                    k_fp16 = tl.load(
                        fp16_row_ptrs[:, None] + d_off[None, :],
                        mask=(dm[None, :] & (use_fp16_key == 1)),
                        other=0,
                    ).to(tl.float32)
                    k_tile = tl.where(use_fp16_key == 1, k_fp16, k_tile)
                    scores += tl.sum(k_tile * q_tile[None, :], axis=1)
                scores = scores * q_scale

                if bid == num_blocks - 1:
                    valid_tok = t_offs < last_block_valid
                    scores = tl.where(valid_tok, scores, float("-inf"))

                block_max = tl.max(scores)
                new_m = tl.maximum(m, block_max)
                alpha = tl.exp(m - new_m)
                acc = acc * alpha
                l = l * alpha
                weights = tl.exp(scores - new_m)
                l += tl.sum(weights)

                slot = tl.load(V_block_slots_ptr + bid)
                slot = tl.maximum(slot, 0)
                v_fp16_tok_base = slot * block_size

                for v_start in range(0, d_v, TILE_V):
                    v_off = v_start + v_offs
                    vm_local = v_off < d_v
                    packed_idx = v_off // 2
                    is_high = v_off % 2
                    v_packed_ptrs = (
                        V_packed_ptr
                        + vpack_base_elem
                        + (base_tok + t_offs[:, None]) * stride_vpack
                        + packed_idx[None, :]
                    )
                    packed_bytes = tl.load(
                        v_packed_ptrs,
                        mask=(vm_local[None, :] & (use_fp16_value == 0)),
                        other=0,
                    )
                    low_nibble = packed_bytes & 0x0F
                    high_nibble = (packed_bytes >> 4) & 0x0F
                    unpacked = tl.where(is_high[None, :] == 1, high_nibble, low_nibble)
                    v_int4 = unpacked.to(tl.float32)
                    group_idx = v_off // group_size
                    scale_ptrs = (
                        V_scales_ptr
                        + vscale_base_elem
                        + (base_tok + t_offs[:, None]) * stride_vscale
                        + group_idx[None, :]
                    )
                    zero_ptrs = (
                        V_zeros_ptr
                        + vscale_base_elem
                        + (base_tok + t_offs[:, None]) * stride_vscale
                        + group_idx[None, :]
                    )
                    v_scale = tl.load(
                        scale_ptrs,
                        mask=(vm_local[None, :] & (use_fp16_value == 0)),
                        other=0.0,
                    ).to(tl.float32)
                    v_zero = tl.load(
                        zero_ptrs,
                        mask=(vm_local[None, :] & (use_fp16_value == 0)),
                        other=0.0,
                    ).to(tl.float32)
                    v_int4_tile = v_int4 * v_scale + v_zero

                    v_fp16_ptrs = (
                        V_fp16_ptr
                        + vscratch_base_elem
                        + (v_fp16_tok_base + t_offs[:, None]) * stride_vscratch
                        + v_off[None, :]
                    )
                    v_fp16_tile = tl.load(
                        v_fp16_ptrs,
                        mask=(vm_local[None, :] & (use_fp16_value == 1)),
                        other=0.0,
                    ).to(tl.float32)
                    v_tile = tl.where(use_fp16_value == 1, v_fp16_tile, v_int4_tile)
                    acc += tl.sum(weights[:, None] * v_tile, axis=0)

                m = new_m

        part_idx = qh * num_splits + sp
        tl.store(M_part_ptr + part_idx, m)
        tl.store(L_part_ptr + part_idx, l)
        tl.store(Acc_part_ptr + part_idx * d_v + v_offs, acc, mask=v_mask)


def selective_attend_multihead_hybrid_mixedv_split_k(
    keys_int8: torch.Tensor,
    keys_scale: torch.Tensor,
    keys_zero_points: torch.Tensor,
    keys_fp16: torch.Tensor,
    topk_mask: torch.Tensor,
    values_int4_packed: torch.Tensor,
    values_int4_scales: torch.Tensor,
    values_int4_zeros: torch.Tensor,
    values_fp16_scratch: torch.Tensor,
    value_fp16_mask: torch.Tensor,
    value_block_slots: torch.Tensor,
    q_all: torch.Tensor,
    skip_mask_i32: torch.Tensor,
    gqa_group: int,
    block_size: int = 16,
    group_size: int = 16,
    q_scale: float = 1.0,
    last_block_valid: int | None = None,
    num_splits: int | None = None,
    key_block_slots: torch.Tensor | None = None,
    int8_token_scores: torch.Tensor | None = None,
    workspace: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Split-K mixed INT4/FP16 value fallback for large contexts."""
    num_kv_heads, N, head_dim = keys_int8.shape
    d_v = values_int4_packed.shape[2] * 2
    num_groups = d_v // group_size
    num_q_heads = q_all.shape[0]
    num_blocks = N // block_size
    device = keys_int8.device
    if key_block_slots is None:
        key_block_slots = torch.arange(num_blocks, dtype=torch.int32, device=device)
    v_scratch_tokens = values_fp16_scratch.shape[1]

    if num_splits is None:
        num_splits = _splitk_num_splits(num_blocks, mixed_values=True)
    num_splits = max(1, int(num_splits))
    blocks_per_split = (num_blocks + num_splits - 1) // num_splits

    assert keys_int8.stride(2) == 1 and keys_int8.stride(1) == head_dim
    assert keys_fp16.stride(2) == 1 and keys_fp16.stride(1) == head_dim
    assert keys_scale.stride(2) == 1 and keys_scale.stride(1) == head_dim
    assert keys_zero_points.stride(2) == 1 and keys_zero_points.stride(1) == head_dim
    assert values_int4_packed.stride(2) == 1 and values_int4_packed.stride(1) == d_v // 2
    assert values_int4_scales.stride(2) == 1 and values_int4_scales.stride(1) == num_groups
    assert values_int4_zeros.stride(2) == 1 and values_int4_zeros.stride(1) == num_groups
    assert values_fp16_scratch.stride(2) == 1 and values_fp16_scratch.stride(1) == d_v

    stride_kv_k = keys_int8.stride(0)
    stride_kv_kfp16 = keys_fp16.stride(0)
    stride_kv_vpack = values_int4_packed.stride(0)
    stride_kv_vscale = values_int4_scales.stride(0)
    stride_kv_vscratch = values_fp16_scratch.stride(0)
    stride_kv_scale = keys_scale.stride(0)

    m_part = _workspace_tensor(
        workspace, "mixedv_m_part", (num_q_heads, num_splits),
        dtype=torch.float32, device=device,
    )
    l_part = _workspace_tensor(
        workspace, "mixedv_l_part", (num_q_heads, num_splits),
        dtype=torch.float32, device=device,
    )
    acc_part = _workspace_tensor(
        workspace, "mixedv_acc_part", (num_q_heads, num_splits, d_v),
        dtype=torch.float32, device=device,
    )
    output = _workspace_tensor(
        workspace, "mixedv_output", (num_q_heads, d_v),
        dtype=torch.float32, device=device,
    )

    TILE_D = triton.next_power_of_2(head_dim)
    TILE_V = triton.next_power_of_2(d_v)
    lbv = block_size if last_block_valid is None else int(last_block_valid)

    _hybrid_mixedv_split_k_partial_kernel[(num_q_heads * num_splits,)](
        keys_int8,
        keys_scale,
        keys_zero_points,
        keys_fp16,
        key_block_slots.contiguous(),
        topk_mask.contiguous(),
        values_int4_packed,
        values_int4_scales,
        values_int4_zeros,
        values_fp16_scratch,
        value_fp16_mask.contiguous(),
        value_block_slots.contiguous(),
        q_all.contiguous(),
        skip_mask_i32.contiguous(),
        m_part,
        l_part,
        acc_part,
        stride_kv_k=stride_kv_k,
        stride_kv_kfp16=stride_kv_kfp16,
        stride_kv_vpack=stride_kv_vpack,
        stride_kv_vscale=stride_kv_vscale,
        stride_kv_vscratch=stride_kv_vscratch,
        stride_kv_scale=stride_kv_scale,
        stride_k=head_dim,
        stride_vpack=d_v // 2,
        stride_vscale=num_groups,
        stride_vscratch=d_v,
        d_v=d_v,
        d_v_half=d_v // 2,
        num_groups=num_groups,
        group_size=group_size,
        num_blocks=num_blocks,
        num_splits=num_splits,
        blocks_per_split=blocks_per_split,
        block_size=block_size,
        head_dim=head_dim,
        q_scale=q_scale,
        num_q_heads=num_q_heads,
        gqa_group=gqa_group,
        last_block_valid=lbv,
        TILE_D=TILE_D,
        TILE_V=TILE_V,
    )
    _hybrid_split_k_reduce_kernel[(num_q_heads,)](
        m_part, l_part, acc_part, output,
        num_q_heads=num_q_heads,
        num_splits=num_splits,
        d_v=d_v,
        TILE_V=TILE_V,
    )
    return output


def selective_attend_multihead_hybrid_split_k(
    keys_int8: torch.Tensor,
    keys_scale: torch.Tensor,
    keys_zero_points: torch.Tensor,  # [num_kv_heads, num_blocks, head_dim] float32 (paper §2.3)
    keys_fp16: torch.Tensor,
    topk_mask: torch.Tensor,
    values_fp16: torch.Tensor,
    q_all: torch.Tensor,
    skip_mask_i32: torch.Tensor,
    gqa_group: int,
    block_size: int = 16,
    q_scale: float = 1.0,
    last_block_valid: int | None = None,
    num_splits: int | None = None,
    key_block_slots: torch.Tensor | None = None,
) -> torch.Tensor:
    """Split-K hybrid attend — same semantics as `selective_attend_multihead_hybrid`,
    partitioned across the block axis for GPU occupancy.

    num_splits is chosen to target ~≥ (#SMs / num_q_heads) programs-per-head
    when not specified. On Blackwell RTX Pro 6000 (188 SMs) with 32 Q heads,
    num_splits=8-16 yields 256-512 programs, filling the machine.
    """
    num_kv_heads, N, head_dim = keys_int8.shape
    d_v = values_fp16.shape[2]
    num_q_heads = q_all.shape[0]
    num_blocks = N // block_size
    device = keys_int8.device
    if key_block_slots is None:
        key_block_slots = torch.arange(num_blocks, dtype=torch.int32, device=device)

    if num_splits is None:
        num_splits = _splitk_num_splits(num_blocks, mixed_values=False)
    num_splits = max(1, int(num_splits))
    blocks_per_split = (num_blocks + num_splits - 1) // num_splits

    # Stride between consecutive KV heads (in elements). For a contiguous
    # packed tensor this equals N * head_dim (or N * d_v for V); for a
    # slice that keeps the parent allocation's dim-0 stride (e.g.
    # `cache.keys_int8[:, :nt, :]` against an oversized cache) it is the
    # full dim-0 stride of the parent. Either way we read it from
    # tensor.stride(0), which is the number of elements to skip to
    # advance dim 0 by one. This lets us avoid the ~60+ MB .contiguous()
    # copy the old wrapper did per call.
    assert keys_int8.stride(2) == 1 and keys_int8.stride(1) == head_dim
    assert keys_fp16.stride(2) == 1 and keys_fp16.stride(1) == head_dim
    assert values_fp16.stride(2) == 1 and values_fp16.stride(1) == d_v
    assert keys_scale.stride(2) == 1 and keys_scale.stride(1) == head_dim
    # Asymmetric zero points share the same layout as keys_scale
    # (paper §2.3 — both are per-channel-per-block float32).
    assert keys_zero_points.stride(2) == 1 and keys_zero_points.stride(1) == head_dim
    assert keys_zero_points.stride(0) == keys_scale.stride(0), (
        "keys_zero_points must have the same kv-head stride as keys_scale "
        "so the kernel's scale_base offset reaches the matching zp slot."
    )
    stride_kv_k = keys_int8.stride(0)
    stride_kv_kfp16 = keys_fp16.stride(0)
    stride_kv_v = values_fp16.stride(0)
    stride_kv_scale = keys_scale.stride(0)

    m_part = torch.empty(num_q_heads, num_splits, dtype=torch.float32, device=device)
    l_part = torch.empty(num_q_heads, num_splits, dtype=torch.float32, device=device)
    acc_part = torch.empty(num_q_heads, num_splits, d_v, dtype=torch.float32, device=device)
    output = torch.empty(num_q_heads, d_v, dtype=torch.float32, device=device)

    TILE_D = triton.next_power_of_2(head_dim)
    TILE_V = triton.next_power_of_2(d_v)

    lbv = block_size if last_block_valid is None else int(last_block_valid)

    _hybrid_split_k_partial_kernel[(num_q_heads * num_splits,)](
        keys_int8, keys_scale, keys_zero_points, keys_fp16,
        key_block_slots.contiguous(),
        topk_mask, values_fp16, q_all, skip_mask_i32,
        m_part, l_part, acc_part,
        stride_kv_k=stride_kv_k,
        stride_kv_kfp16=stride_kv_kfp16,
        stride_kv_v=stride_kv_v,
        stride_kv_scale=stride_kv_scale,
        stride_k=head_dim,
        stride_v=d_v,
        num_blocks=num_blocks,
        num_splits=num_splits,
        blocks_per_split=blocks_per_split,
        block_size=block_size,
        head_dim=head_dim,
        d_v=d_v,
        q_scale=q_scale,
        num_q_heads=num_q_heads,
        gqa_group=gqa_group,
        last_block_valid=lbv,
        TILE_D=TILE_D,
        TILE_V=TILE_V,
    )

    _hybrid_split_k_reduce_kernel[(num_q_heads,)](
        m_part, l_part, acc_part, output,
        num_q_heads=num_q_heads,
        num_splits=num_splits,
        d_v=d_v,
        TILE_V=TILE_V,
    )
    return output


def selective_attend_multihead_hybrid(
    keys_int8: torch.Tensor,       # [num_kv_heads, N, head_dim] int8
    keys_scale: torch.Tensor,      # [num_kv_heads, num_blocks, head_dim] float32
    keys_zero_points: torch.Tensor,  # [num_kv_heads, num_blocks, head_dim] float32 (paper §2.3)
    keys_fp16: torch.Tensor,       # [num_kv_heads, N, head_dim] float16
    topk_mask: torch.Tensor,       # [num_q_heads, num_blocks] int32 (1=fp16, 0=int8)
    values_fp16: torch.Tensor,     # [num_kv_heads, N, d_v] float16
    q_all: torch.Tensor,           # [num_q_heads, head_dim] float32
    skip_mask_i32: torch.Tensor,   # [num_q_heads, num_blocks] int32
    gqa_group: int,
    block_size: int = 16,
    q_scale: float = 1.0,
    last_block_valid: int | None = None,
) -> torch.Tensor:
    """Hybrid INT8/FP16 keys: reads INT8 for most blocks, FP16 for top-K.

    The top-K mask is per-head: different Q heads can use FP16 for different blocks.
    Only the top-K blocks are read from keys_fp16 — the rest from keys_int8.
    Zero Python-level materialisation of dequant tensors.
    """
    num_kv_heads, N, head_dim = keys_int8.shape
    d_v = values_fp16.shape[2]
    num_q_heads = q_all.shape[0]
    num_blocks = N // block_size
    device = keys_int8.device

    K_int8_flat = keys_int8.reshape(num_kv_heads * N, head_dim).contiguous()
    K_fp16_flat = keys_fp16.reshape(num_kv_heads * N, head_dim).contiguous()
    V_flat = values_fp16.reshape(num_kv_heads * N, d_v).contiguous()
    output = torch.empty(num_q_heads, d_v, dtype=torch.float32, device=device)

    TILE_D = triton.next_power_of_2(head_dim)
    TILE_V = triton.next_power_of_2(d_v)

    lbv = block_size if last_block_valid is None else int(last_block_valid)
    _multihead_selective_attend_hybrid_kernel[(num_q_heads,)](
        K_int8_flat, keys_scale.contiguous(), keys_zero_points.contiguous(),
        K_fp16_flat,
        topk_mask.contiguous(),
        V_flat,
        q_all.contiguous(), skip_mask_i32.contiguous(), output,
        N=N,
        stride_k=head_dim,
        stride_v=d_v,
        num_blocks=num_blocks,
        block_size=block_size,
        head_dim=head_dim,
        d_v=d_v,
        q_scale=q_scale,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        gqa_group=gqa_group,
        last_block_valid=lbv,
        TILE_D=TILE_D,
        TILE_V=TILE_V,
    )
    return output


def selective_attend_multihead(
    keys_packed: torch.Tensor,    # [num_kv_heads, N, head_dim] float32
    values_packed: torch.Tensor,  # [num_kv_heads, N, d_v] float32
    q_all: torch.Tensor,          # [num_q_heads, head_dim] float32
    skip_mask_i32: torch.Tensor,  # [num_q_heads, num_blocks] int32
    gqa_group: int,
    block_size: int = 16,
    q_scale: float = 1.0,
) -> torch.Tensor:
    """Multi-head selective attention in one kernel launch.

    Returns [num_q_heads, d_v] output.
    """
    num_kv_heads, N, head_dim = keys_packed.shape
    d_v = values_packed.shape[2]
    num_q_heads = q_all.shape[0]
    num_blocks = N // block_size
    device = keys_packed.device

    K_flat = keys_packed.reshape(num_kv_heads * N, head_dim).contiguous()
    V_flat = values_packed.reshape(num_kv_heads * N, d_v).contiguous()
    output = torch.empty(num_q_heads, d_v, dtype=torch.float32, device=device)

    TILE_D = triton.next_power_of_2(head_dim)
    TILE_V = triton.next_power_of_2(d_v)

    _multihead_selective_attend_kernel[(num_q_heads,)](
        K_flat, V_flat, q_all.contiguous(), skip_mask_i32.contiguous(), output,
        N=N,
        stride_k=head_dim,
        stride_v=d_v,
        num_blocks=num_blocks,
        block_size=block_size,
        head_dim=head_dim,
        d_v=d_v,
        q_scale=q_scale,
        num_q_heads=num_q_heads,
        gqa_group=gqa_group,
        TILE_D=TILE_D,
        TILE_V=TILE_V,
    )
    return output


# Keep single-head version for compatibility
def selective_attend_triton(
    keys: torch.Tensor,
    values: torch.Tensor,
    q: torch.Tensor,
    skip_mask_i32: torch.Tensor,
    block_size: int = 16,
    q_scale: float = 1.0,
) -> torch.Tensor:
    """Single-head selective attention."""
    N, head_dim = keys.shape
    d_v = values.shape[1]
    # Wrap as multi-head with 1 head
    out = selective_attend_multihead(
        keys.unsqueeze(0), values.unsqueeze(0),
        q.unsqueeze(0), skip_mask_i32.unsqueeze(0),
        gqa_group=1, block_size=block_size, q_scale=q_scale,
    )
    return out.squeeze(0)
