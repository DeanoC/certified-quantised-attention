"""Fused INT8 score + block-reduce + certify: single kernel launch.

One Triton kernel that:
  1. Loads K_int8 tiles from global memory
  2. Dequantises in registers
  3. Computes q·K dot products via vectorised multiply-accumulate
  4. Reduces per-block: m_b = max(scores), S_b = sum(exp(scores - m_b))
  5. Writes m_b, S_b to global memory

Then a tiny certification kernel (single launch, O(num_blocks) scalar ops)
computes global max, total mass, and skip mask.

Total: 2 kernel launches for the entire Phase 1 pipeline.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: Fused matmul + block reduce (one launch, one program per block)
# ---------------------------------------------------------------------------
@triton.jit
def _fused_score_certify_single_launch_kernel(
    # Data
    K_int8_ptr,      # [N, head_dim] int8 contiguous
    K_scale_ptr,     # [num_blocks, head_dim] float32
    Q_ptr,           # [head_dim] float32
    Corr_ptr,        # [num_blocks] float32
    # Output
    M_b_ptr,         # [num_blocks] float32
    S_b_ptr,         # [num_blocks] float32
    Skip_ptr,        # [num_blocks] int32
    # Sync
    Counter_ptr,     # [1] int32 — atomic counter for last-program detection
    # Layout
    stride_k_n: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    q_scale: tl.constexpr,
    num_blocks: tl.constexpr,
    n_programs: tl.constexpr,
    block_epsilon: tl.constexpr,
    # Tuning
    TILE_D: tl.constexpr,
    BPP: tl.constexpr,
    TILE_N: tl.constexpr,         # tile for certify pass over num_blocks
):
    """Score + certify in one kernel launch.

    Each program scores BPP blocks, then atomically increments a counter.
    The last program to finish runs the certify pass (global max, mass, skip mask).
    """
    pid = tl.program_id(0)
    t_offs = tl.arange(0, block_size)
    d_offs = tl.arange(0, TILE_D)
    num_tiles = (head_dim + TILE_D - 1) // TILE_D

    # Phase 1: score this program's blocks
    for local_b in range(BPP):
        bid = pid * BPP + local_b
        still_valid = bid < num_blocks
        if still_valid:
            base = bid * block_size
            scale_base = bid * head_dim
            scores = tl.zeros((block_size,), dtype=tl.float32)
            row_ptrs = K_int8_ptr + (base + t_offs) * stride_k_n

            for tile_idx in range(num_tiles):
                d_start = tile_idx * TILE_D
                d_off = d_start + d_offs
                d_mask = d_off < head_dim
                q_tile = tl.load(Q_ptr + d_off, mask=d_mask, other=0.0).to(tl.float32)
                ch_scale = tl.load(K_scale_ptr + scale_base + d_off, mask=d_mask, other=0.0).to(tl.float32)
                k_ptrs = row_ptrs[:, None] + d_off[None, :]
                k_tile = tl.load(k_ptrs, mask=d_mask[None, :], other=0).to(tl.float32)
                k_fp = k_tile * ch_scale[None, :]
                scores += tl.sum(k_fp * q_tile[None, :], axis=1)

            scores = scores * q_scale
            m_b = tl.max(scores)
            s_b = tl.sum(tl.exp(scores - m_b))
            tl.store(M_b_ptr + bid, m_b)
            tl.store(S_b_ptr + bid, s_b)

    # Barrier: atomic increment, last program does certify
    old_count = tl.atomic_add(Counter_ptr, 1)
    is_last = old_count == (n_programs - 1)

    if is_last:
        # Certify pass: global max → total mass → skip mask
        # This runs on one SM after all scoring is done
        m_global = tl.full((), float("-inf"), dtype=tl.float32)
        for s in range(0, num_blocks, TILE_N):
            o = s + tl.arange(0, TILE_N)
            mk = o < num_blocks
            m_global = tl.maximum(m_global, tl.max(tl.load(M_b_ptr + o, mask=mk, other=float("-inf"))))

        total_mass = tl.full((), 0.0, dtype=tl.float32)
        for s in range(0, num_blocks, TILE_N):
            o = s + tl.arange(0, TILE_N)
            mk = o < num_blocks
            sb = tl.load(S_b_ptr + o, mask=mk, other=0.0)
            mb = tl.load(M_b_ptr + o, mask=mk, other=float("-inf"))
            cr = tl.load(Corr_ptr + o, mask=mk, other=1.0)
            total_mass += tl.sum(tl.where(mk, sb * cr * tl.exp(mb - m_global), 0.0))

        for s in range(0, num_blocks, TILE_N):
            o = s + tl.arange(0, TILE_N)
            mk = o < num_blocks
            sb = tl.load(S_b_ptr + o, mask=mk, other=0.0)
            mb = tl.load(M_b_ptr + o, mask=mk, other=float("-inf"))
            cr = tl.load(Corr_ptr + o, mask=mk, other=1.0)
            res = sb * cr * tl.exp(mb - m_global)
            skip = (res / total_mass) < block_epsilon
            tl.store(Skip_ptr + o, tl.where(mk, skip.to(tl.int32), 0), mask=mk)


# Keep the old two-kernel versions for comparison
@triton.jit
def _fused_matmul_reduce_kernel(
    K_int8_ptr, K_scale_ptr, Q_ptr, M_b_ptr, S_b_ptr,
    stride_k_n: tl.constexpr, head_dim: tl.constexpr,
    block_size: tl.constexpr, q_scale: tl.constexpr,
    num_blocks: tl.constexpr, TILE_D: tl.constexpr, BPP: tl.constexpr,
):
    pid = tl.program_id(0)
    t_offs = tl.arange(0, block_size)
    d_offs = tl.arange(0, TILE_D)
    num_tiles = (head_dim + TILE_D - 1) // TILE_D
    for local_b in range(BPP):
        bid = pid * BPP + local_b
        still_valid = bid < num_blocks
        if still_valid:
            base = bid * block_size
            scale_base = bid * head_dim
            scores = tl.zeros((block_size,), dtype=tl.float32)
            row_ptrs = K_int8_ptr + (base + t_offs) * stride_k_n
            for tile_idx in range(num_tiles):
                d_start = tile_idx * TILE_D
                d_off = d_start + d_offs
                d_mask = d_off < head_dim
                q_tile = tl.load(Q_ptr + d_off, mask=d_mask, other=0.0).to(tl.float32)
                ch_scale = tl.load(K_scale_ptr + scale_base + d_off, mask=d_mask, other=0.0).to(tl.float32)
                k_ptrs = row_ptrs[:, None] + d_off[None, :]
                k_tile = tl.load(k_ptrs, mask=d_mask[None, :], other=0).to(tl.float32)
                k_fp = k_tile * ch_scale[None, :]
                scores += tl.sum(k_fp * q_tile[None, :], axis=1)
            scores = scores * q_scale
            m_b = tl.max(scores)
            s_b = tl.sum(tl.exp(scores - m_b))
            tl.store(M_b_ptr + bid, m_b)
            tl.store(S_b_ptr + bid, s_b)


# ---------------------------------------------------------------------------
# Kernel 2: Certify (single program, O(num_blocks) work, ~1-2μs)
# ---------------------------------------------------------------------------
@triton.jit
def _certify_kernel(
    M_b_ptr,          # [num_blocks] float32
    S_b_ptr,          # [num_blocks] float32
    Corr_ptr,         # [num_blocks] float32
    Skip_ptr,         # [num_blocks] int32 output
    num_blocks: tl.constexpr,
    block_epsilon: tl.constexpr,
    TILE_N: tl.constexpr,
):
    # Pass 1: global max
    m_g = tl.full((), float("-inf"), dtype=tl.float32)
    for s in range(0, num_blocks, TILE_N):
        o = s + tl.arange(0, TILE_N)
        mk = o < num_blocks
        m_g = tl.maximum(m_g, tl.max(tl.load(M_b_ptr + o, mask=mk, other=float("-inf"))))

    # Pass 2: total mass
    total = tl.full((), 0.0, dtype=tl.float32)
    for s in range(0, num_blocks, TILE_N):
        o = s + tl.arange(0, TILE_N)
        mk = o < num_blocks
        sb = tl.load(S_b_ptr + o, mask=mk, other=0.0)
        mb = tl.load(M_b_ptr + o, mask=mk, other=float("-inf"))
        cr = tl.load(Corr_ptr + o, mask=mk, other=1.0)
        total += tl.sum(tl.where(mk, sb * cr * tl.exp(mb - m_g), 0.0))

    # Pass 3: skip mask
    for s in range(0, num_blocks, TILE_N):
        o = s + tl.arange(0, TILE_N)
        mk = o < num_blocks
        sb = tl.load(S_b_ptr + o, mask=mk, other=0.0)
        mb = tl.load(M_b_ptr + o, mask=mk, other=float("-inf"))
        cr = tl.load(Corr_ptr + o, mask=mk, other=1.0)
        res = sb * cr * tl.exp(mb - m_g)
        skip = (res / total) < block_epsilon
        tl.store(Skip_ptr + o, tl.where(mk, skip.to(tl.int32), 0), mask=mk)


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------
@triton.jit
def _multihead_score_certify_kernel(
    # Data — all KV heads packed: K_int8[kv_head, N, head_dim]
    K_int8_ptr,      # [num_kv_heads * N, head_dim] int8 contiguous (heads packed)
    K_scale_ptr,     # [num_kv_heads, num_blocks, head_dim] float32
    K_zp_ptr,        # [num_kv_heads, num_blocks, head_dim] float32 (paper §2.3 asymmetric)
    Q_ptr,           # [num_q_heads, head_dim] float32
    Corr_ptr,        # [num_kv_heads, num_blocks] float32
    # Output
    Skip_ptr,        # [num_q_heads, num_blocks] int32
    M_b_ptr,         # [num_q_heads, num_blocks] float32
    S_b_ptr,         # [num_q_heads, num_blocks] float32
    Scores_ptr,      # [num_q_heads, num_blocks, block_size] float32 scratch
    # Sync
    Counter_ptr,     # [1] int32
    # Layout
    N: tl.constexpr,             # tokens per KV head
    stride_k_n: tl.constexpr,   # head_dim
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    q_scale: tl.constexpr,
    num_blocks: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    gqa_group: tl.constexpr,     # num_q_heads // num_kv_heads
    n_programs: tl.constexpr,
    block_epsilon: tl.constexpr,
    TILE_D: tl.constexpr,
    TILE_N: tl.constexpr,
    STORE_SCORES: tl.constexpr,
):
    """Score + certify ALL heads in one kernel launch.

    Grid: (num_q_heads * ceil(num_blocks / BPP),)
    Each program handles one q_head × one chunk of blocks.
    The last program to finish (across ALL heads) runs per-head certify.
    """
    pid = tl.program_id(0)
    # Map pid → (q_head_idx, block_chunk_idx)
    blocks_per_chunk = (num_blocks + 31) // 32  # BPP
    chunks_per_head = (num_blocks + blocks_per_chunk - 1) // blocks_per_chunk
    q_head_idx = pid // chunks_per_head
    chunk_idx = pid % chunks_per_head

    valid_head = q_head_idx < num_q_heads
    if valid_head:
        kv_head_idx = q_head_idx // gqa_group
        t_offs = tl.arange(0, block_size)
        d_offs = tl.arange(0, TILE_D)
        num_tiles = (head_dim + TILE_D - 1) // TILE_D

        # Base pointer for this KV head's keys
        kv_base = kv_head_idx * N * stride_k_n
        scale_head_base = kv_head_idx * num_blocks * head_dim

        # Score blocks in this chunk
        for local_b in range(blocks_per_chunk):
            bid = chunk_idx * blocks_per_chunk + local_b
            still_valid = bid < num_blocks
            if still_valid:
                base_tok = bid * block_size
                scale_block_base = scale_head_base + bid * head_dim
                scores = tl.zeros((block_size,), dtype=tl.float32)
                row_ptrs = K_int8_ptr + kv_base + (base_tok + t_offs) * stride_k_n

                for tile_idx in range(num_tiles):
                    d_start = tile_idx * TILE_D
                    d_off = d_start + d_offs
                    d_mask = d_off < head_dim
                    q_tile = tl.load(Q_ptr + q_head_idx * head_dim + d_off, mask=d_mask, other=0.0).to(tl.float32)
                    ch_scale = tl.load(K_scale_ptr + scale_block_base + d_off, mask=d_mask, other=0.0).to(tl.float32)
                    ch_zp = tl.load(K_zp_ptr + scale_block_base + d_off, mask=d_mask, other=0.0).to(tl.float32)
                    k_ptrs = row_ptrs[:, None] + d_off[None, :]
                    k_tile = tl.load(k_ptrs, mask=d_mask[None, :], other=0).to(tl.float32)
                    # Asymmetric dequant (paper §2.3 Eq. 1): k̂ = q · s + z.
                    # Expansion in dot product: q · k̂ = Σ q_c (Q_c s_c + z_c)
                    #   = Σ q_c Q_c s_c  +  Σ q_c z_c
                    # The +z_c term is the paper §5.2 L575 correction.
                    k_fp = k_tile * ch_scale[None, :] + ch_zp[None, :]
                    scores += tl.sum(k_fp * q_tile[None, :], axis=1)

                scores = scores * q_scale
                m_b = tl.max(scores)
                s_b = tl.sum(tl.exp(scores - m_b))
                out_idx = q_head_idx * num_blocks + bid
                tl.store(M_b_ptr + out_idx, m_b)
                tl.store(S_b_ptr + out_idx, s_b)
                if STORE_SCORES:
                    score_base = out_idx * block_size
                    tl.store(Scores_ptr + score_base + t_offs, scores)

    # Barrier: last program does certify for ALL heads
    old_count = tl.atomic_add(Counter_ptr, 1)
    is_last = old_count == (n_programs - 1)

    if is_last:
        for qh in range(num_q_heads):
            kvh = qh // gqa_group
            m_base = qh * num_blocks
            corr_base = kvh * num_blocks

            # Global max for this head
            m_global = tl.full((), float("-inf"), dtype=tl.float32)
            for s in range(0, num_blocks, TILE_N):
                o = s + tl.arange(0, TILE_N)
                mk = o < num_blocks
                m_global = tl.maximum(m_global, tl.max(tl.load(M_b_ptr + m_base + o, mask=mk, other=float("-inf"))))

            # Total mass
            total = tl.full((), 0.0, dtype=tl.float32)
            for s in range(0, num_blocks, TILE_N):
                o = s + tl.arange(0, TILE_N)
                mk = o < num_blocks
                sb = tl.load(S_b_ptr + m_base + o, mask=mk, other=0.0)
                mb = tl.load(M_b_ptr + m_base + o, mask=mk, other=float("-inf"))
                cr = tl.load(Corr_ptr + corr_base + o, mask=mk, other=1.0)
                total += tl.sum(tl.where(mk, sb * cr * tl.exp(mb - m_global), 0.0))

            # Skip mask
            for s in range(0, num_blocks, TILE_N):
                o = s + tl.arange(0, TILE_N)
                mk = o < num_blocks
                sb = tl.load(S_b_ptr + m_base + o, mask=mk, other=0.0)
                mb = tl.load(M_b_ptr + m_base + o, mask=mk, other=float("-inf"))
                cr = tl.load(Corr_ptr + corr_base + o, mask=mk, other=1.0)
                res = sb * cr * tl.exp(mb - m_global)
                skip = (res / total) < block_epsilon
                tl.store(Skip_ptr + m_base + o, tl.where(mk, skip.to(tl.int32), 0), mask=mk)


def _fused_score_certify_multihead_triton(
    K_int8_packed: torch.Tensor,   # [num_kv_heads, N, head_dim] int8
    K_scale: torch.Tensor,         # [num_kv_heads, num_blocks, head_dim] float32
    K_zero_points: torch.Tensor,   # [num_kv_heads, num_blocks, head_dim] float32 (paper §2.3)
    q_all: torch.Tensor,           # [num_q_heads, head_dim] float32
    correction: torch.Tensor,      # [num_kv_heads, num_blocks] float32
    gqa_group: int,
    block_size: int = 16,
    q_scale: float = 1.0,
    block_epsilon: float = 0.001,
    return_token_scores: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Score + certify ALL heads in one kernel launch.

    Returns (m_b, S_b, skip_mask), and optionally token_scores
    [num_q_heads, num_blocks, block_size].
    """
    num_kv_heads, N, head_dim = K_int8_packed.shape
    num_q_heads = q_all.shape[0]
    num_blocks = N // block_size
    device = K_int8_packed.device

    # Flatten KV heads for contiguous access
    K_flat = K_int8_packed.reshape(num_kv_heads * N, head_dim).contiguous()

    m_b = torch.empty(num_q_heads, num_blocks, dtype=torch.float32, device=device)
    S_b = torch.empty(num_q_heads, num_blocks, dtype=torch.float32, device=device)
    skip_i32 = torch.empty(num_q_heads, num_blocks, dtype=torch.int32, device=device)
    token_scores = (
        torch.empty(num_q_heads, num_blocks, block_size, dtype=torch.float32, device=device)
        if return_token_scores
        else torch.empty(0, dtype=torch.float32, device=device)
    )
    counter = torch.zeros(1, dtype=torch.int32, device=device)

    TILE_D = triton.next_power_of_2(head_dim)
    TILE_N = min(triton.next_power_of_2(num_blocks), 1024)
    blocks_per_chunk = max(1, (num_blocks + 31) // 32)
    chunks_per_head = (num_blocks + blocks_per_chunk - 1) // blocks_per_chunk
    n_programs = num_q_heads * chunks_per_head

    _multihead_score_certify_kernel[(n_programs,)](
        K_flat, K_scale.contiguous(), K_zero_points.contiguous(),
        q_all.contiguous(), correction.contiguous(),
        skip_i32, m_b, S_b, token_scores, counter,
        N=N,
        stride_k_n=head_dim,
        head_dim=head_dim,
        block_size=block_size,
        q_scale=q_scale,
        num_blocks=num_blocks,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        gqa_group=gqa_group,
        n_programs=n_programs,
        block_epsilon=block_epsilon,
        TILE_D=TILE_D,
        TILE_N=TILE_N,
        STORE_SCORES=return_token_scores,
    )
    if return_token_scores:
        return m_b, S_b, skip_i32.bool(), token_scores
    return m_b, S_b, skip_i32.bool()


def fused_score_certify_multihead(
    K_int8_packed: torch.Tensor,   # [num_kv_heads, N, head_dim] int8
    K_scale: torch.Tensor,         # [num_kv_heads, num_blocks, head_dim] float32
    K_zero_points: torch.Tensor,   # [num_kv_heads, num_blocks, head_dim] float32 (paper §2.3)
    q_all: torch.Tensor,           # [num_q_heads, head_dim] float32
    correction: torch.Tensor,      # [num_kv_heads, num_blocks] float32
    gqa_group: int,
    block_size: int = 16,
    q_scale: float = 1.0,
    block_epsilon: float = 0.001,
    return_token_scores: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Score + certify ALL heads.

    `DOTCACHE_SCORE_BACKEND=cutlass_sm120` probes the future tensor-core
    backend boundary, but the Triton implementation remains the exact fallback
    until `DOTCACHE_CUTLASS_SM120_ENABLE_SCORE=1` and the CUTLASS kernels pass
    the phase-1 correctness/performance gates.
    """
    import os as _os

    backend = _os.environ.get("DOTCACHE_SCORE_BACKEND", "triton").strip().lower()
    if backend == "cutlass_sm120" and not return_token_scores:
        try:
            from dotcache.backends.cutlass_sm120 import (
                cutlass_sm120_available,
                score_certify_cutlass,
            )

            if (
                _os.environ.get("DOTCACHE_CUTLASS_SM120_ENABLE_SCORE", "0") == "1"
                and cutlass_sm120_available()
            ):
                return score_certify_cutlass(
                    K_int8_packed=K_int8_packed,
                    K_scale=K_scale,
                    K_zero_points=K_zero_points,
                    q_all=q_all,
                    correction=correction,
                    gqa_group=gqa_group,
                    block_size=block_size,
                    q_scale=q_scale,
                    block_epsilon=block_epsilon,
                )
        except Exception:
            pass

    return _fused_score_certify_multihead_triton(
        K_int8_packed=K_int8_packed,
        K_scale=K_scale,
        K_zero_points=K_zero_points,
        q_all=q_all,
        correction=correction,
        gqa_group=gqa_group,
        block_size=block_size,
        q_scale=q_scale,
        block_epsilon=block_epsilon,
        return_token_scores=return_token_scores,
    )


def fused_score_certify(
    K_int8: torch.Tensor,         # [N, head_dim] int8 contiguous
    K_scale: torch.Tensor,        # [num_blocks, head_dim] float32
    q: torch.Tensor,              # [head_dim] float32
    correction: torch.Tensor,     # [num_blocks] float32
    block_size: int = 16,
    q_scale: float = 1.0,
    block_epsilon: float = 0.001,
    single_launch: bool = True,   # True = one kernel launch, False = two
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused INT8 scoring + certification.

    Returns (m_b, S_b, skip_mask).
    """
    N, head_dim = K_int8.shape
    num_blocks = N // block_size
    device = K_int8.device

    m_b = torch.empty(num_blocks, dtype=torch.float32, device=device)
    S_b = torch.empty(num_blocks, dtype=torch.float32, device=device)

    TILE_D = triton.next_power_of_2(head_dim)
    BPP = max(1, (num_blocks + 31) // 32)
    n_progs = (num_blocks + BPP - 1) // BPP
    TILE_N = min(triton.next_power_of_2(num_blocks), 1024)

    if single_launch:
        # Single kernel: score + certify via atomic last-program pattern
        skip_i32 = torch.empty(num_blocks, dtype=torch.int32, device=device)
        counter = torch.zeros(1, dtype=torch.int32, device=device)

        _fused_score_certify_single_launch_kernel[(n_progs,)](
            K_int8, K_scale, q, correction,
            m_b, S_b, skip_i32, counter,
            stride_k_n=head_dim,
            head_dim=head_dim,
            block_size=block_size,
            q_scale=q_scale,
            num_blocks=num_blocks,
            n_programs=n_progs,
            block_epsilon=block_epsilon,
            TILE_D=TILE_D,
            BPP=BPP,
            TILE_N=TILE_N,
        )
        return m_b, S_b, skip_i32.bool()
    else:
        # Two kernels: score then certify
        _fused_matmul_reduce_kernel[(n_progs,)](
            K_int8, K_scale, q, m_b, S_b,
            stride_k_n=head_dim,
            head_dim=head_dim,
            block_size=block_size,
            q_scale=q_scale,
            num_blocks=num_blocks,
            TILE_D=TILE_D,
            BPP=BPP,
        )
        skip_i32 = torch.empty(num_blocks, dtype=torch.int32, device=device)
        _certify_kernel[(1,)](
            m_b, S_b, correction, skip_i32,
            num_blocks=num_blocks,
            block_epsilon=block_epsilon,
            TILE_N=TILE_N,
        )
        return m_b, S_b, skip_i32.bool()


def selective_attend(
    keys_fp: torch.Tensor,        # [N, head_dim]
    values_fp: torch.Tensor,      # [N, d_v]
    q: torch.Tensor,              # [head_dim]
    skip_mask: torch.Tensor,      # [num_blocks] bool
    block_size: int = 16,
    q_scale: float = 1.0,
) -> torch.Tensor:
    """Phase 2: attend non-skipped blocks."""
    num_blocks = skip_mask.shape[0]
    idx = (~skip_mask).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return torch.zeros(values_fp.shape[-1], dtype=torch.float32, device=q.device)
    k_bl = keys_fp.reshape(num_blocks, block_size, -1)
    v_bl = values_fp.reshape(num_blocks, block_size, -1)
    ak = k_bl[idx].reshape(-1, keys_fp.shape[-1]).to(torch.float32)
    av = v_bl[idx].reshape(-1, values_fp.shape[-1]).to(torch.float32)
    s = torch.matmul(ak, q.to(torch.float32)) * q_scale
    w = torch.softmax(s, dim=0)
    return w @ av
