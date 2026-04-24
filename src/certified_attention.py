"""Certified attention: complete pipeline for one layer.

Orchestrates:
  1. Multi-head INT8 scoring + certification (fused Triton kernel)
  2. Runtime V-format decision based on mass partition (ρ check)
  3. Multi-head selective attention (fused Triton kernel)

The V-format decision uses Phase 1 outputs at zero additional cost:
  - Compute tier-2 residual mass ρ from m_b, S_b, skip_mask
  - If η₄ · ρ < tolerance → INT4 values (45% less VRAM)
  - Else → page in FP16 values from CPU
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from typing import Any

from tiered_cache import TieredKeyCacheLayer
from fused_score_certify import fused_score_certify_multihead
from selective_attend_triton import (
    selective_attend_multihead,
    selective_attend_multihead_int8,
    selective_attend_multihead_int8k_int4v,
    selective_attend_multihead_hybrid,
    selective_attend_multihead_hybrid_split_k,
)


# Default tolerance for INT4 value error (η₄ · ρ must be below this)
DEFAULT_V_TOLERANCE = 0.5

# Number of top-K blocks whose mass counts toward α_K (not tier-2)
TOP_K_BLOCKS = 4

# Adaptive top-K* defaults (paper §3.3).
DEFAULT_TAU_COV = 0.995
DEFAULT_K_MIN = 2
# None = no upper clamp; the selector lets tau_cov fully dictate K* per head.
DEFAULT_K_MAX: int | None = None

# Rung-1 fallback defaults (paper §3.4). When the adaptive selector's tail
# mass exceeds DEFAULT_RUNG1_THRESHOLD (k_max hit, τ_cov not reached), expand
# the top-K set by multiplying K* by DEFAULT_RUNG1_MULTIPLIER.
DEFAULT_RUNG1_THRESHOLD = 0.02
DEFAULT_RUNG1_MULTIPLIER = 2.0


class _PhaseTimer:
    """Context manager that times a code region via torch.cuda.Event when a
    phase_timings dict is supplied; zero overhead otherwise. Accumulates
    microseconds under `{name}_us` inside the dict so multiple entries sum
    cleanly across layers within a decode step."""
    __slots__ = ("_timings", "_name", "_start", "_end")

    def __init__(self, timings: dict | None, name: str):
        self._timings = timings
        self._name = name
        self._start = None
        self._end = None

    def __enter__(self):
        if self._timings is not None:
            self._start = torch.cuda.Event(enable_timing=True)
            self._start.record()
        return self

    def __exit__(self, *_):
        if self._timings is not None:
            self._end = torch.cuda.Event(enable_timing=True)
            self._end.record()
            torch.cuda.synchronize()
            us = self._start.elapsed_time(self._end) * 1000.0
            key = f"{self._name}_us"
            self._timings[key] = self._timings.get(key, 0.0) + us


def compute_tier2_residual_mass(
    m_b: torch.Tensor,       # [num_q_heads, num_blocks] block maxima
    S_b: torch.Tensor,       # [num_q_heads, num_blocks] block sums
    skip_mask: torch.Tensor,  # [num_q_heads, num_blocks] bool (True=skip)
    top_k: int = TOP_K_BLOCKS,
    return_details: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-head tier-2 residual mass ρ from Phase 1 outputs.

    ρ = 1 - α_K - β, where:
      α_K = fraction of mass on top-K blocks (by m_b)
      β = fraction of mass on skipped blocks

    Returns ρ: [num_q_heads] float32, worst-case per head.

    When return_details=True, also returns (mass_frac, topk_idx) so callers
    can compute tighter per-block bounds (e.g. Σ_b ρ_b · η_b) without
    recomputing the softmax mass partition.
    """
    num_q_heads, num_blocks = m_b.shape

    # Unnormalised mass per block: S_b * exp(m_b - m_global)
    m_global = m_b.amax(dim=1, keepdim=True)  # [num_q_heads, 1]
    log_mass = torch.log(S_b.clamp(min=1e-30)) + m_b - m_global
    mass = torch.exp(log_mass)  # [num_q_heads, num_blocks]
    total_mass = mass.sum(dim=1, keepdim=True).clamp(min=1e-30)

    # Normalised mass fractions
    mass_frac = mass / total_mass  # [num_q_heads, num_blocks]

    # α_K: mass on top-K blocks (by m_b score, per head)
    k = min(top_k, num_blocks)
    _, topk_idx = m_b.topk(k, dim=1)  # [num_q_heads, k]
    alpha_K = mass_frac.gather(1, topk_idx).sum(dim=1)  # [num_q_heads]

    # β: mass on skipped blocks
    beta = (mass_frac * skip_mask.float()).sum(dim=1)  # [num_q_heads]

    # ρ = 1 - α_K - β (clamped to [0, 1])
    rho = (1.0 - alpha_K - beta).clamp(min=0.0, max=1.0)
    if return_details:
        return rho, mass_frac, topk_idx
    return rho


def compute_value_error_bound(
    mass_frac: torch.Tensor,       # [num_q_heads, num_blocks] from compute_tier2_residual_mass
    topk_idx: torch.Tensor,        # [num_q_heads, top_k] indices of top-K blocks
    skip_mask: torch.Tensor,       # [num_q_heads, num_blocks] bool (True = skipped)
    eta_per_block: torch.Tensor,   # [num_kv_heads, num_blocks] per-block INT4 error bound η_b
    gqa_group: int,
) -> torch.Tensor:
    """Tight per-head value-error bound Σ_{b ∉ top-K ∪ skipped} mass_frac[h,b] · η[kv(h), b].

    Upper-bounds the INT4 output error for that head's weighted V sum on
    the residual (non-top-K, non-skipped) blocks. Strictly ≤ ρ_total · η_max
    (the conservative bound used by the legacy `decide_v_format` scalar path)
    because η_max ≥ η_b ∀b and ρ_total = Σ_b mass_frac over the residual set.

    Cheap: the mass_frac and topk_idx inputs are already computed inside
    `compute_tier2_residual_mass(..., return_details=True)`, so the only
    extra work is a scatter + two elementwise ops + one reduction.

    Returns: [num_q_heads] float32, the per-head tight bound. Callers
    typically take .max() across heads for a per-layer step decision.
    """
    num_q_heads, num_blocks_mass = mass_frac.shape
    num_kv_heads, num_blocks_eta = eta_per_block.shape

    # Trailing partial blocks appear in mass_frac (the residual-mass path
    # cats a pad row for them) but have no INT4 data in eta_per_block —
    # they're always FP16-attended, so they contribute zero to E_val and
    # trimming to num_blocks_eta is equivalent. Any blocks past
    # num_blocks_eta are either (a) the trailing partial, always in
    # top-K → residual False anyway, or (b) skipped → residual False.
    num_blocks = min(num_blocks_mass, num_blocks_eta)
    mass_frac_q = mass_frac[:, :num_blocks]
    skip_mask_q = skip_mask[:, :num_blocks]
    eta_q = eta_per_block[:, :num_blocks]

    # Residual mask: not top-K AND not skipped (these are the blocks we
    # attend to with INT4 values — i.e. where η_b contributes to the bound).
    residual = torch.ones_like(mass_frac_q, dtype=torch.bool)
    # topk_idx may index into the (possibly padded) num_blocks_mass range;
    # clamp it so we don't scatter past the trimmed extent.
    topk_trim = topk_idx.clamp(max=num_blocks - 1)
    residual.scatter_(1, topk_trim, False)
    residual &= ~skip_mask_q

    # Broadcast η from [num_kv_heads, num_blocks] → [num_q_heads, num_blocks]
    # via the GQA mapping kv_of(qh) = qh // gqa_group.
    kv_idx = torch.arange(num_q_heads, device=mass_frac.device) // gqa_group
    eta_per_qhead = eta_q[kv_idx]  # [num_q_heads, num_blocks]

    e_val = (mass_frac_q * eta_per_qhead * residual.to(mass_frac.dtype)).sum(dim=1)
    return e_val


def decide_v_format(
    rho: torch.Tensor,       # [num_q_heads] tier-2 residual mass
    eta_int4: float,          # worst-case INT4 error bound for this layer
    tolerance: float = DEFAULT_V_TOLERANCE,
) -> str:
    """Decide INT4 vs FP16 values based on the legacy loose bound
    ρ_worst · η_worst < tolerance.

    This upper-bounds Σ_b ρ_b · η_b via ρ_total · η_max. For a tighter
    per-block bound, see `decide_v_format_tight` below.

    Returns 'int4' or 'fp16'.
    """
    rho_worst = rho.max().item()
    int4_error = eta_int4 * rho_worst
    return "int4" if int4_error < tolerance else "fp16"


def decide_v_format_tight(
    e_val_head: torch.Tensor,   # [num_q_heads] from compute_value_error_bound
    tolerance: float = DEFAULT_V_TOLERANCE,
) -> str:
    """Decide INT4 vs FP16 values using the tight per-block bound
    max_h Σ_{b ∉ top-K ∪ skipped} mass_frac[h, b] · η_b.

    Strictly less conservative than `decide_v_format` (returns 'int4'
    at least as often). Use when the paper's certified bound should
    track achieved per-step error, not its ρ_total·η_max upper bound.

    Returns 'int4' or 'fp16'.
    """
    return "int4" if e_val_head.max().item() < tolerance else "fp16"


def compute_adaptive_topk_mask(
    m_b: torch.Tensor,       # [num_q_heads, num_blocks] Phase-1 block max (INT8 estimate)
    S_b: torch.Tensor,       # [num_q_heads, num_blocks] Phase-1 block sum
    tau_cov: float = DEFAULT_TAU_COV,
    k_min: int = DEFAULT_K_MIN,
    k_max: int | None = DEFAULT_K_MAX,
    per_kv_group_topk: bool = False,
    gqa_group: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paper §3.3 adaptive top-K* selector (cumulative-mass threshold).

    Per head: sort blocks by estimated mass, find smallest K such that
    cumulative mass ≥ `tau_cov`, clamp to [k_min, k_max]. Returns:

    - topk_mask [H, B] bool: True = block is in the top-K* for that head.
    - k_star    [H] int32: actual K* selected per head (post-clamp).
    - tail_mass [H] float32: 1 − Σ mass on top-K* (INT8-estimated).
    - tau_cov_actual [H] float32: actual cumulative mass captured at K*.

    All computation stays on device; this function has zero CPU syncs.
    """
    num_q_heads, num_blocks = m_b.shape
    device = m_b.device
    if num_blocks == 0:
        empty_bool = torch.zeros(num_q_heads, 0, dtype=torch.bool, device=device)
        zeros_int = torch.zeros(num_q_heads, dtype=torch.int32, device=device)
        zeros_f32 = torch.zeros(num_q_heads, dtype=torch.float32, device=device)
        return empty_bool, zeros_int, zeros_f32, zeros_f32

    # Per-head normalised mass, stable via log-sum-exp.
    m_global = m_b.amax(dim=1, keepdim=True)
    log_mass = torch.log(S_b.clamp(min=1e-30)) + m_b - m_global
    mass = torch.exp(log_mass)
    total = mass.sum(dim=1, keepdim=True).clamp(min=1e-30)
    mass_frac = mass / total                                       # [H, B]

    # Per-KV-group selection (experimental, paper follow-up). With GQA,
    # 4 Q heads share a KV head but independently pick their own top-K;
    # the union across the group (from the cache's point of view) is
    # ~350 blocks at k_max=128 even before rung-1. This option collapses
    # the 4 Q heads into a single top-K decision: sum mass_frac across
    # the group, pick top-K on the aggregated distribution, then broadcast
    # the mask back to all 4 Q heads. Trades the per-head bound for a
    # per-group bound; the paper would need to re-derive §3.3 for this
    # variant. See cache_sweep_tau/SUMMARY.md for the measurement.
    if per_kv_group_topk and gqa_group > 1 and num_q_heads % gqa_group == 0:
        num_kv = num_q_heads // gqa_group
        group_mass = mass_frac.view(num_kv, gqa_group, num_blocks).sum(dim=1)  # [num_kv, B]
        group_total = group_mass.sum(dim=1, keepdim=True).clamp(min=1e-30)
        mass_frac_eff = (group_mass / group_total)  # [num_kv, B]
        # Run the selection on mass_frac_eff, then broadcast back to [num_q_heads, B].
        sorted_mass, sorted_idx = mass_frac_eff.sort(dim=1, descending=True)
        cumsum = sorted_mass.cumsum(dim=1)
        tau_vec = torch.full((num_kv, 1), float(tau_cov), device=device, dtype=cumsum.dtype)
        k_star_group = torch.searchsorted(cumsum, tau_vec).squeeze(1) + 1
        hi = num_blocks if k_max is None else min(int(k_max), num_blocks)
        lo = min(int(k_min), hi)
        k_star_group = k_star_group.clamp(min=lo, max=hi).to(torch.int32)
        pos = torch.arange(num_blocks, device=device).unsqueeze(0)
        keep_sorted = pos < k_star_group.unsqueeze(1).to(pos.dtype)
        topk_mask_group = torch.zeros_like(mass_frac_eff, dtype=torch.bool)
        topk_mask_group.scatter_(1, sorted_idx, keep_sorted)
        # Tail / coverage per group.
        k_idx = (k_star_group.long() - 1).clamp(min=0, max=num_blocks - 1).unsqueeze(1)
        tau_actual_group = cumsum.gather(1, k_idx).squeeze(1).float()
        tail_mass_group = (1.0 - tau_actual_group).clamp(min=0.0)
        # Broadcast back to [num_q_heads, num_blocks].
        topk_mask = topk_mask_group.unsqueeze(1).expand(-1, gqa_group, -1).reshape(num_q_heads, num_blocks).contiguous()
        k_star = k_star_group.unsqueeze(1).expand(-1, gqa_group).reshape(num_q_heads).contiguous()
        tail_mass = tail_mass_group.unsqueeze(1).expand(-1, gqa_group).reshape(num_q_heads).contiguous()
        tau_actual = tau_actual_group.unsqueeze(1).expand(-1, gqa_group).reshape(num_q_heads).contiguous()
        return topk_mask, k_star, tail_mass, tau_actual

    # Sort descending per head; cumulative mass in sorted order.
    sorted_mass, sorted_idx = mass_frac.sort(dim=1, descending=True)
    cumsum = sorted_mass.cumsum(dim=1)                             # [H, B]

    # K*[h] = smallest k such that cumsum[h, k-1] ≥ tau_cov.
    # searchsorted on each row returns the insertion index of tau_cov;
    # since cumsum is non-decreasing in [0, 1], that index = (K* - 1).
    tau_vec = torch.full((num_q_heads, 1), float(tau_cov), device=device, dtype=cumsum.dtype)
    k_star = torch.searchsorted(cumsum, tau_vec).squeeze(1) + 1    # [H]
    # Clamp to [k_min, min(k_max, num_blocks)]. k_max=None means no cap
    # beyond num_blocks — let tau_cov alone dictate K* per head.
    hi = num_blocks if k_max is None else min(int(k_max), num_blocks)
    lo = min(int(k_min), hi)
    k_star = k_star.clamp(min=lo, max=hi).to(torch.int32)

    # Build [H, B] top-K mask: position < k_star[h] in the sorted order.
    pos = torch.arange(num_blocks, device=device).unsqueeze(0)     # [1, B]
    keep_sorted = pos < k_star.unsqueeze(1).to(pos.dtype)           # [H, B] bool
    topk_mask = torch.zeros_like(mass_frac, dtype=torch.bool)
    topk_mask.scatter_(1, sorted_idx, keep_sorted)                 # [H, B]

    # Tail mass + actual coverage using cumsum at (K*-1).
    k_idx = (k_star.long() - 1).clamp(min=0, max=num_blocks - 1).unsqueeze(1)
    tau_actual = cumsum.gather(1, k_idx).squeeze(1).float()
    tail_mass = (1.0 - tau_actual).clamp(min=0.0)
    return topk_mask, k_star, tail_mass, tau_actual


def compute_fp16_block_scores(
    cache: TieredKeyCacheLayer,
    q_all: torch.Tensor,           # [num_q_heads, head_dim]
    block_indices: torch.Tensor,   # [num_q_heads, K] int64 block ids to score
    num_scoring_blocks: int,       # upper bound on valid block id (fully-quantized blocks)
    gqa_group: int,
    q_scale: float,
) -> torch.Tensor:
    """Compute per-head per-block FP16 max-logit for the given block set.

    Mirrors Phase-1's m_b (the per-block max pre-softmax logit) but uses the
    FP16 keys from the tiered cache's GPU mirror (or CPU if no mirror). Only
    blocks in [0, num_scoring_blocks) are valid; others receive -inf.

    Returns: [num_q_heads, K] float32 block scores suitable for ranking.
    """
    num_q_heads, head_dim = q_all.shape
    _, K = block_indices.shape
    bs = cache.block_size
    device = q_all.device

    # Total tokens covered by the fully-quantized block range.
    nt = num_scoring_blocks * bs

    neg_inf = torch.full((num_q_heads, K), float("-inf"), dtype=torch.float32, device=device)
    if nt == 0 or K == 0:
        return neg_inf

    if cache.keys_fp16_gpu is not None:
        keys = cache.keys_fp16_gpu[:, :nt, :]
    else:
        keys = cache.keys_fp16_cpu[:, :nt, :].to(device=device, non_blocking=True)
    if keys.dtype != q_all.dtype:
        keys = keys.to(dtype=q_all.dtype)

    # [num_q_heads, K, bs, head_dim] gather: for each (h, k) pick tokens
    # [block*bs, block*bs + bs) from keys[kv_h].
    kv_per_h = torch.arange(num_q_heads, device=device) // gqa_group          # [H]
    kv_per_hk = kv_per_h.unsqueeze(1).expand(-1, K)                            # [H, K]
    starts = block_indices.to(torch.long) * bs                                 # [H, K]
    token_offsets = torch.arange(bs, device=device)                            # [bs]
    token_idx = starts.unsqueeze(-1) + token_offsets                           # [H, K, bs]
    valid = (token_idx < nt) & (starts.unsqueeze(-1) >= 0)                     # [H, K, bs]
    # Clamp out-of-range indices so the gather is always valid; masked later.
    token_idx_clamped = token_idx.clamp(min=0, max=max(nt - 1, 0))

    # keys[kv, t]: fancy indexing with [H, K, bs] index tensors.
    kv_idx = kv_per_hk.unsqueeze(-1).expand(-1, -1, bs)                        # [H, K, bs]
    k_gathered = keys[kv_idx, token_idx_clamped]                               # [H, K, bs, head_dim]

    # Dot with q_h: q_all [H, head_dim] → [H, 1, 1, head_dim]
    q_expanded = q_all.unsqueeze(1).unsqueeze(1)
    logits = (k_gathered.float() * q_expanded.float()).sum(dim=-1) * q_scale   # [H, K, bs]
    neg_inf_tok = torch.full_like(logits, float("-inf"))
    logits = torch.where(valid, logits, neg_inf_tok)
    scores = logits.amax(dim=-1)                                               # [H, K]
    return scores


def recompute_heads_dense_fp16(
    cache: TieredKeyCacheLayer,
    q_all: torch.Tensor,               # [num_q_heads, head_dim]
    output: torch.Tensor,              # [num_q_heads, d_v] to be patched in-place
    head_indices: torch.Tensor,        # [num_to_recompute] int64 q-head ids
    gqa_group: int,
    q_scale: float,
) -> torch.Tensor:
    """Rung-3 recompute: for each listed head, replace output[h] with a full
    FP16 dense attention using the cache's FP16 keys + FP16 values (dequantised
    from INT4 if that's the value tier).

    The recompute only touches listed heads — non-disagreeing heads keep their
    Phase-2 output unchanged. This is intentional: the spec calls for per-head
    granularity so only the heads that paid the detection are corrected.
    """
    if head_indices.numel() == 0:
        return output
    nt = cache.num_tokens
    device = q_all.device
    if cache.keys_fp16_gpu is not None:
        keys = cache.keys_fp16_gpu[:, :nt, :]
    else:
        keys = cache.keys_fp16_cpu[:, :nt, :].to(device=device, non_blocking=True)
    values_f32 = cache.get_values_f32()[:, :nt, :]  # FP32 from VRAM (either tier)
    keys_f32 = keys.to(device=device, dtype=torch.float32)

    # Loop-free per-head recompute: pull the rows we need and vectorise the
    # dot-products. head_indices is typically small (≤ num_q_heads).
    heads = head_indices.to(device=device, dtype=torch.long)
    kv_ids = heads // gqa_group                                        # [M]
    q_sel = q_all.index_select(0, heads).float()                        # [M, head_dim]
    k_sel = keys_f32.index_select(0, kv_ids)                            # [M, nt, head_dim]
    v_sel = values_f32.index_select(0, kv_ids)                          # [M, nt, d_v]
    logits = torch.einsum("mnd,md->mn", k_sel, q_sel) * q_scale        # [M, nt]
    weights = torch.softmax(logits, dim=1)                              # [M, nt]
    head_out = torch.einsum("mn,mnd->md", weights, v_sel)              # [M, d_v]
    output.index_copy_(0, heads, head_out.to(output.dtype))
    return output


def augment_mask_with_exploration(
    topk_mask: torch.Tensor,         # [H, B] bool — top-K* mask from adaptive selector
    exploration_rate: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Paper §6 exploration budget: randomly promote `exploration_rate` of
    the non-promoted blocks per head to FP16 for monitoring purposes.

    Returns (augmented_mask, exploration_mask, total_explored).

    Uses a rejection-free per-element Bernoulli so the number of exploration
    picks is a random variable around `rate · non_promoted`. Fully on device,
    no sync. Pass a `generator` to keep the exploration reproducible.
    """
    if exploration_rate <= 0.0 or topk_mask.numel() == 0:
        empty = torch.zeros_like(topk_mask)
        return topk_mask, empty, 0
    # Per-element Bernoulli(exploration_rate) on the non-promoted blocks only.
    non_promoted = ~topk_mask
    # Draw one uniform per block per head; no CPU sync.
    rand = torch.rand(topk_mask.shape, device=topk_mask.device, generator=generator)
    draw = rand < float(exploration_rate)
    exploration_mask = non_promoted & draw
    augmented = topk_mask | exploration_mask
    # Running total is on device until the caller decides to item() it.
    return augmented, exploration_mask, int(exploration_mask.sum().item())


def compute_delta_bound(
    q_all: torch.Tensor,        # [num_q_heads, head_dim]
    key_scales: torch.Tensor,    # [num_kv_heads, num_blocks, head_dim] float32
    gqa_group: int,
    q_scale: float,
) -> torch.Tensor:
    """Per-head tight Δ bound (paper Eq. 4, runtime form).

    Per-channel INT8 quantisation: |K_fp16[c] − K_int8[c]·s_c| ≤ s_c/2.
    Score error per token per block is ≤ Σ_c |q_c|·(s_c/2) = (1/2)·Σ|q_c|·s_c
    in the pre-q_scale space. The attention kernels multiply the logit by
    q_scale = 1/√d before taking the per-block max, so the post-q_scale
    bound — the one we compare against `m_b` and `fp16_block_scores`, both
    of which are already post-q_scale — is

        Δ[h] = (1 / (2·√d)) · Σ_c |q[h,c]| · s_c.

    The 1/√d factor IS the q_scale; there is no separate q_scale factor
    applied afterwards. A prior version of this function multiplied by
    q_scale a second time, making Δ √d× too small and causing the
    score-consistency monitor to fire ≈36% per-head on real 8K/h_d=128
    workloads. Fixed to match the derivation above.
    """
    num_q_heads, head_dim = q_all.shape
    if key_scales.numel() == 0:
        return torch.zeros(num_q_heads, dtype=torch.float32, device=q_all.device)
    per_channel_scale = key_scales.amax(dim=1)                                   # [kv_h, d]
    kv_per_h = torch.arange(num_q_heads, device=q_all.device) // gqa_group
    s_per_h = per_channel_scale.index_select(0, kv_per_h)                        # [H, d]
    delta = (q_all.abs().float() * s_per_h.float()).sum(dim=1) / (2.0 * math.sqrt(head_dim))
    # `q_scale` is already folded into the 1/(2·√d) factor above — do not
    # multiply again. Accepted in the signature for call-site compatibility.
    _ = q_scale
    return delta


def score_consistency_violations(
    int8_scores: torch.Tensor,     # [H, K] INT8 block scores on the re-ranked set
    fp16_scores: torch.Tensor,     # [H, K] FP16 block scores on the same set
    delta_per_head: torch.Tensor,  # [H] Δ bound (paper Eq. 4)
    eps_guard: float = 0.01,
) -> torch.Tensor:
    """Per-head score-consistency (paper §6).

    Returns a [H] bool tensor: True when any block's |FP16 - INT8| score
    exceeds Δ + eps_guard. A non-zero count here indicates the Theorem-2
    bound is empirically broken on this step — a correctness red flag
    (stale quant metadata, cache corruption, etc.), not a quality knob.
    """
    if int8_scores.numel() == 0:
        return torch.zeros(int8_scores.shape[0], dtype=torch.bool, device=int8_scores.device)
    diff = (fp16_scores - int8_scores).abs().float()
    threshold = (delta_per_head + float(eps_guard)).unsqueeze(1)  # [H, 1]
    return (diff > threshold).any(dim=1)


def detect_ranking_disagreement(
    int8_scores: torch.Tensor,     # [num_q_heads, K]
    fp16_scores: torch.Tensor,     # [num_q_heads, K]
    r: int,
) -> torch.Tensor:
    """Per-head: does the top-r INT8 ranking match the top-r FP16 ranking?

    Returns a [num_q_heads] bool tensor; True = rankings disagree on at least
    one of the top-r positions. Uses argsort over the scoring set rather than
    global block ids so the two rankings share a vocabulary.
    """
    if int8_scores.numel() == 0 or r <= 0:
        return torch.zeros(int8_scores.shape[0], dtype=torch.bool, device=int8_scores.device)
    k = int8_scores.shape[1]
    r_eff = min(r, k)
    rank_int8 = int8_scores.argsort(dim=1, descending=True)[:, :r_eff]
    rank_fp16 = fp16_scores.argsort(dim=1, descending=True)[:, :r_eff]
    # Ordered top-r must match position-by-position (rank_int8[i] == rank_fp16[i])
    return (rank_int8 != rank_fp16).any(dim=1)


def sdpa_attend_with_skip(
    cache: TieredKeyCacheLayer,
    q_all: torch.Tensor,           # [num_q_heads, head_dim] (model dtype, e.g. BF16)
    skip_mask: torch.Tensor,       # [num_q_heads, num_active_blocks] bool (True=skip)
    gqa_group: int,
    q_scale: float,
) -> torch.Tensor:
    """Phase 2 attend using PyTorch SDPA — matches dense attention precision exactly.

    Uses the FP16 CPU keys and VRAM values from the tiered cache, expanding
    the block-level skip_mask into a per-token attention mask for SDPA.
    """
    num_q_heads, head_dim = q_all.shape
    nt = cache.num_tokens
    device = q_all.device
    dtype = q_all.dtype  # keep computation in model's native dtype (BF16)

    # Keys from GPU mirror (falls back to CPU copy only if mirror absent).
    # Values already live on GPU.
    if cache.keys_fp16_gpu is not None:
        keys = cache.keys_fp16_gpu[:, :nt, :]
        if keys.dtype != dtype:
            keys = keys.to(dtype=dtype)
    else:
        keys = cache.keys_fp16_cpu[:, :nt, :].to(device=device, dtype=dtype)
    values = cache.values_fp16[:, :nt, :]
    if values.dtype != dtype:
        values = values.to(dtype=dtype)
    num_kv_heads = keys.shape[0]

    # Build per-token attention mask from block-level skip_mask.
    # CRITICAL: pass attn_mask=None when nothing is skipped, otherwise
    # PyTorch SDPA falls back from the FlashAttention kernel to MATH/MEM_EFFICIENT,
    # which has slightly different accumulator precision.  That drift flips
    # near-tied argmax tokens and cascades into repetition loops on
    # enumeration outputs (RULER vt/fwe).  The .any().item() is one GPU sync
    # per layer per step (~3μs × 32 layers ≈ 0.1 ms overhead) — trivial vs
    # the decode floor.
    bs = cache.block_size
    num_active_blocks = skip_mask.shape[1]
    if skip_mask.any().item():
        token_skip = skip_mask.unsqueeze(-1).expand(-1, -1, bs).reshape(num_q_heads, -1)[:, :nt]
        attn_mask = torch.where(token_skip, float("-inf"), 0.0).to(dtype=dtype)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(2)  # [1, num_q_heads, 1, nt]
    else:
        attn_mask = None

    # Expand keys/values for GQA: [kv_heads, nt, hd] → [num_q_heads, nt, hd]
    # Use expand (not repeat_interleave) to match HF's GQA handling — same
    # memory layout means SDPA takes the same FlashAttention code path.
    keys_exp = keys.unsqueeze(1).expand(-1, gqa_group, -1, -1).reshape(
        num_q_heads, nt, head_dim).contiguous()
    values_exp = values.unsqueeze(1).expand(-1, gqa_group, -1, -1).reshape(
        num_q_heads, nt, values.shape[2]).contiguous()

    # SDPA: [batch=1, heads, seq, dim]
    q_sdpa = q_all.unsqueeze(0).unsqueeze(2)   # [1, num_q_heads, 1, hd]
    k_sdpa = keys_exp.unsqueeze(0)              # [1, num_q_heads, nt, hd]
    v_sdpa = values_exp.unsqueeze(0)            # [1, num_q_heads, nt, dv]

    output = F.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa,
        attn_mask=attn_mask,
        scale=q_scale,
    )  # [1, num_q_heads, 1, dv]

    return output[0, :, 0, :].float()  # [num_q_heads, dv] float32


def certified_attention_layer(
    cache: TieredKeyCacheLayer,
    q_all: torch.Tensor,           # [num_q_heads, head_dim] model dtype (BF16) or float32
    gqa_group: int,
    q_scale: float = None,
    block_epsilon: float = 0.001,
    collect_stats: bool = True,
    v_tolerance: float = DEFAULT_V_TOLERANCE,
    top_k_fp16_keys: int = 0,
    concentration_threshold: float = 0.0,
    ranking_fallback: bool = False,
    ranking_r: int = 1,
    ranking_fallback_mode: str = "full",
    tau_cov: float | None = None,
    k_min: int = DEFAULT_K_MIN,
    k_max: int | None = DEFAULT_K_MAX,
    rung1_threshold: float = DEFAULT_RUNG1_THRESHOLD,
    rung1_multiplier: float = DEFAULT_RUNG1_MULTIPLIER,
    score_consistency_check: bool = False,
    eps_guard: float = 0.01,
    exploration_rate: float = 0.0,
    exploration_generator: torch.Generator | None = None,
    phase_timings: dict | None = None,
    per_kv_group_topk: bool = False,
    # Value-error bound mode (paper §3.4 follow-up). "loose" keeps the
    # legacy ρ_worst · η_worst check used by decide_v_format — an upper
    # bound on Σ_b ρ_b · η_b but wastes slack. "tight" computes the actual
    # Σ_b ρ_b · η_b per head and uses its max for decide_v_format. Tight
    # will flip INT4 more often than loose (strictly less conservative).
    # Either way, telemetry emits e_val_max/e_val_mean from the tight bound.
    # Default "tight" (paper-correct); see CertifiedAttentionState for the
    # sweep that confirmed the flip is safe at v_tolerance=0.5.
    value_error_mode: str = "tight",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Full certified attention for one layer, all heads.

    When the cache has INT4 values, Phase 1 scoring outputs are used to
    decide at runtime whether INT4 is safe (η₄ · ρ < tolerance) or
    FP16 values should be paged in from CPU.

    Rung-3 ranking-consistency fallback: when enabled, after Phase-1 INT8
    scoring picks the top-K blocks, re-rank those K blocks with FP16 keys
    and compare top-`ranking_r` positions. Heads whose rankings disagree
    are recomputed with full FP16 keys + values for this step (mode="full"),
    or just recorded in telemetry without action (mode="measure").

    Returns:
        output: [num_q_heads, d_v] float32
        stats: dict with skip counts, v_format decision, ranking metrics
    """
    if q_scale is None:
        q_scale = 1.0 / (cache.head_dim ** 0.5)

    num_q_heads = q_all.shape[0]
    n_qblocks = cache.num_quantized_blocks
    bs = cache.block_size

    # Page-in telemetry (paper §3.4 runtime cost accounting). Keys-side H2D
    # is zero when the VRAM FP16 mirror (keys_fp16_gpu) is allocated — the
    # default for the arXiv sweep. Values-side H2D fires only on Rung-2
    # escalation when ρ·η exceeds v_tolerance. All four rung booleans are
    # emitted in the per-layer stats for uniform aggregation; the harness
    # ORs them across layers to get a step-level rung-fired decision.
    h2d_key_bytes = 0
    h2d_key_blocks = 0
    h2d_value_bytes = 0
    h2d_value_blocks = 0
    rung2_fired = False
    # FP16 key cache telemetry (populated in the hybrid path when the cache
    # is bounded). Zero when the cache is in full-mirror mode or not used.
    fp16_cache_hits_step = 0
    fp16_cache_misses_step = 0
    fp16_cache_evictions_step = 0
    fp16_cache_needed_blocks = 0

    # Phase 1: INT8 scoring only on fully quantized blocks
    if n_qblocks > 0:
        with _PhaseTimer(phase_timings, "phase1_int8_scoring"):
            m_b, S_b, skip_mask = fused_score_certify_multihead(
                K_int8_packed=cache.keys_int8[:, :n_qblocks * bs, :],
                K_scale=cache.keys_scale[:, :n_qblocks, :],
                q_all=q_all,
                correction=cache.correction[:, :n_qblocks],
                gqa_group=gqa_group,
                block_size=bs,
                q_scale=q_scale,
                block_epsilon=block_epsilon,
            )
    else:
        device = q_all.device
        m_b = torch.empty(num_q_heads, 0, dtype=torch.float32, device=device)
        S_b = torch.empty(num_q_heads, 0, dtype=torch.float32, device=device)
        skip_mask = torch.empty(num_q_heads, 0, dtype=torch.bool, device=device)

    # If there's a trailing partial block, force-attend it via hybrid FP16 path
    num_active_blocks = cache.active_blocks
    if cache.has_trailing_partial_block:
        trailing_bid = cache.trailing_block_idx
        # Extend scoring arrays to include trailing block
        pad_m = torch.zeros(num_q_heads, 1, dtype=torch.float32, device=q_all.device)
        pad_S = torch.ones(num_q_heads, 1, dtype=torch.float32, device=q_all.device)
        pad_skip = torch.zeros(num_q_heads, 1, dtype=torch.bool, device=q_all.device)
        m_b = torch.cat([m_b, pad_m], dim=1)
        S_b = torch.cat([S_b, pad_S], dim=1)
        skip_mask = torch.cat([skip_mask, pad_skip], dim=1)

    # Top-K safety: clear skip bit for highest-scoring blocks so they're
    # always attended — the certification correction may underestimate their mass.
    num_active_blocks = cache.active_blocks
    top_k_fp16 = top_k_fp16_keys
    if top_k_fp16 > 0 and num_active_blocks > 0:
        k = min(top_k_fp16, num_active_blocks)
        topk_idx = m_b.topk(k, dim=1).indices  # [num_q_heads, k]
        skip_mask.scatter_(1, topk_idx, False)

    # Paper §3.3 adaptive K*: when tau_cov is supplied, replace the skip mask
    # with the per-head top-K* selection so that blocks whose cumulative
    # INT8-estimated mass reaches tau_cov are attended and the rest are
    # skipped. Block-epsilon certification still runs above this point; the
    # adaptive selection supersedes it when enabled (the paper's bound
    # E_key ≤ 2·V_max·(1-tau_cov) is tighter than the epsilon-only bound at
    # the default tau_cov=0.995).
    adaptive_topk_mask = None
    k_star: torch.Tensor | None = None
    tail_mass_est: torch.Tensor | None = None
    tau_cov_actual: torch.Tensor | None = None
    rung1_triggered_heads = 0
    explored_blocks_count = 0
    if tau_cov is not None and tau_cov > 0 and n_qblocks > 0:
        # Restrict adaptive selection to the fully-quantised block range —
        # the trailing partial block has no INT8 score and is force-attended
        # below. Build the [H, n_blocks] mask by padding the fully-quantised
        # selection with the trailing block forced in.
        m_b_cert = m_b[:, :n_qblocks]
        S_b_cert = S_b[:, :n_qblocks]
        with _PhaseTimer(phase_timings, "adaptive_selection"):
            topk_mask_cert, k_star, tail_mass_est, tau_cov_actual = compute_adaptive_topk_mask(
                m_b_cert, S_b_cert, tau_cov=tau_cov, k_min=k_min, k_max=k_max,
                per_kv_group_topk=per_kv_group_topk, gqa_group=gqa_group,
            )

        # Rung-1 (paper §3.4): if any head's tail mass exceeded the configured
        # threshold — typically because k_max capped the selection before
        # tau_cov was reached on a diffuse head — expand the budget and re-pick.
        # The expansion uses a larger k_max = min(k_max * multiplier, n_qblocks);
        # heads whose selection was already good at the original k_max will just
        # land on the same (or a smaller) K* because the tau_cov threshold is
        # unchanged. Accounting: count heads that triggered the expansion.
        # k_max=None means no upper cap so the adaptive selector already hit
        # tau_cov fully → no expansion to do, skip the whole check (also the
        # cheap path sync-wise).
        if (
            rung1_threshold is not None and rung1_threshold >= 0
            and k_max is not None
        ):
            rung1_trigger_mask = tail_mass_est > rung1_threshold  # [H] bool
            # One .sum().item() sync covers both "any?" and "how many?" — no
            # need for a separate .any().item() gate.
            rung1_triggered_heads = int(rung1_trigger_mask.sum().item())
            if rung1_triggered_heads > 0:
                expanded_k_max = min(int(math.ceil(k_max * float(rung1_multiplier))), n_qblocks)
                topk_mask_cert2, k_star2, tail_mass_est2, tau_cov_actual2 = compute_adaptive_topk_mask(
                    m_b_cert, S_b_cert, tau_cov=tau_cov, k_min=k_min, k_max=expanded_k_max,
                    per_kv_group_topk=per_kv_group_topk, gqa_group=gqa_group,
                )
                # Only apply the expanded selection to triggered heads so
                # non-triggered heads keep their original K* (avoiding
                # unnecessary bandwidth). The selector is deterministic on the
                # same m_b/S_b so the original top-K entries are a subset of
                # the expanded top-K entries for triggered heads.
                trig = rung1_trigger_mask.unsqueeze(1)
                topk_mask_cert = torch.where(trig, topk_mask_cert2, topk_mask_cert)
                k_star = torch.where(rung1_trigger_mask, k_star2, k_star)
                tail_mass_est = torch.where(rung1_trigger_mask, tail_mass_est2, tail_mass_est)
                tau_cov_actual = torch.where(rung1_trigger_mask, tau_cov_actual2, tau_cov_actual)

        # Paper §6 exploration budget: randomly promote a small fraction of
        # the non-top-K* blocks so their FP16 scores can be cross-checked
        # against the INT8 estimates. Defence-in-depth only — does not
        # affect the paper's certified bounds because the explored blocks
        # are *added* to the attended set (never demote a top-K* block).
        exploration_mask_cert: torch.Tensor | None = None
        explored_blocks_count = 0
        if exploration_rate > 0.0:
            topk_mask_cert, exploration_mask_cert, explored_blocks_count = (
                augment_mask_with_exploration(
                    topk_mask_cert, exploration_rate, exploration_generator,
                )
            )

        # Skip = NOT top-K*; false for trailing partial block (force-attended).
        skip_cert = ~topk_mask_cert
        if cache.has_trailing_partial_block:
            trailing = torch.zeros(num_q_heads, 1, dtype=torch.bool, device=q_all.device)
            skip_mask = torch.cat([skip_cert, trailing], dim=1)
            adaptive_topk_mask = torch.cat(
                [topk_mask_cert, torch.ones(num_q_heads, 1, dtype=torch.bool, device=q_all.device)],
                dim=1,
            )
        else:
            skip_mask = skip_cert
            adaptive_topk_mask = topk_mask_cert

    # Force-attend trailing partial block (it has no INT8 data for scoring)
    if cache.has_trailing_partial_block:
        skip_mask[:, cache.trailing_block_idx] = 0

    # Rung-3 ranking-consistency detection. Runs over the fully-quantized
    # block range (excludes the trailing partial block, which has no INT8
    # score). Populates the telemetry counters below; commit 3 will add the
    # per-head fallback action that sets skip_mask[h, :] = False on disagree.
    ranking_disagree_r1_heads = 0
    ranking_disagree_r3_heads = 0
    ranking_disagree_mask: torch.Tensor | None = None
    fp16_block_scores: torch.Tensor | None = None
    top_block_indices: torch.Tensor | None = None
    ranking_k = 0
    score_consistency_violation_heads = 0
    score_consistency_violation_mask: torch.Tensor | None = None
    delta_bound_mean = 0.0
    # The FP16 block re-scoring is needed for either Rung-3 ranking check or
    # the score-consistency check. Compute it once when either is enabled.
    need_fp16_scores = (ranking_fallback or score_consistency_check) and n_qblocks > 0
    if need_fp16_scores:
        with _PhaseTimer(phase_timings, "ranking_check"):
            ranking_k = min(max(ranking_r, top_k_fp16_keys, 4), n_qblocks)
            int8_scores = m_b[:, :n_qblocks]
            top_block_indices = int8_scores.topk(ranking_k, dim=1).indices  # [H, K]
            top_int8_scores = int8_scores.gather(1, top_block_indices)       # [H, K]
            # In bounded-cache mode, compute_fp16_block_scores reads the VRAM
            # scratch (cache.keys_fp16_gpu). The blocks selected by INT8
            # top-K must be resident before this call, otherwise the scratch
            # returns zeros and the FP16 rescore is garbage — which would
            # trip the score-consistency monitor and fire Rung-4 spuriously.
            if cache.fp16_key_cache_capacity is not None:
                _ranking_needed = top_block_indices.unique().tolist()
                with _PhaseTimer(phase_timings, "h2d_pagein"):
                    _rh, _rm, _rb, _re = cache.ensure_fp16_keys_resident(_ranking_needed)
                h2d_key_bytes += _rb
                h2d_key_blocks += _rm
                fp16_cache_hits_step += _rh
                fp16_cache_misses_step += _rm
                fp16_cache_evictions_step += _re
                fp16_cache_needed_blocks += len(_ranking_needed)
            fp16_block_scores = compute_fp16_block_scores(
                cache, q_all, top_block_indices, n_qblocks, gqa_group, q_scale,
            )
        # Empirical per-block score residual |FP16 − INT8| paired with the
        # analytical per-head delta bound — exposed for the certificate
        # figure (reviewer Item 3). Measured over the top-K blocks where
        # attention mass concentrates. delta_per_head is also used by
        # score_consistency_check below; hoisted here so both ratio and
        # violation count can reuse it (one compute_delta_bound call).
        score_residual = (fp16_block_scores - top_int8_scores).abs()  # [H, K]
        score_residual_max_value = float(score_residual.max().item())
        score_residual_mean_value = float(score_residual.mean().item())
        delta_per_head = compute_delta_bound(
            q_all, cache.keys_scale[:, :n_qblocks, :], gqa_group, q_scale,
        )
        delta_bound_mean = float(delta_per_head.mean().item())
        delta_bound_max_value = float(delta_per_head.max().item())
        # Tightness ratio: observed / bound, per block. 0.0 = bound far from
        # tight; 1.0 = bound saturated; >1.0 would be a Theorem-2 violation
        # (separately caught by the score_consistency_check canary).
        delta_per_block = delta_per_head.unsqueeze(1).clamp(min=1e-12)  # [H, 1]
        score_residual_ratio = score_residual / delta_per_block
        score_residual_ratio_max_value = float(score_residual_ratio.max().item())
        score_residual_ratio_mean_value = float(score_residual_ratio.mean().item())
        if ranking_fallback:
            # Single pair of argsorts covers r=1, r=3, and r=ranking_r — no
            # need to call detect_ranking_disagreement three times (each call
            # was redoing the same sort).
            k_for_rank = top_int8_scores.shape[1]
            if k_for_rank > 0 and ranking_r > 0:
                rank_int8 = top_int8_scores.argsort(dim=1, descending=True)
                rank_fp16 = fp16_block_scores.argsort(dim=1, descending=True)
                rank_diff = rank_int8 != rank_fp16  # [H, K]
                r_main = min(int(ranking_r), k_for_rank)
                r1 = min(1, k_for_rank)
                r3 = min(3, k_for_rank)
                ranking_disagree_mask = rank_diff[:, :r_main].any(dim=1)
                ranking_disagree_r1_heads = int(rank_diff[:, :r1].any(dim=1).sum().item())
                ranking_disagree_r3_heads = int(rank_diff[:, :r3].any(dim=1).sum().item())
            else:
                ranking_disagree_mask = torch.zeros(
                    num_q_heads, dtype=torch.bool, device=q_all.device,
                )
        if score_consistency_check:
            # Paper §6 instability-detection: |FP16 - INT8| per block bounded
            # by Δ + eps_guard. Any violation means Theorem 2 was empirically
            # broken on this step — a canary for stale metadata / cache
            # corruption, expected 0-count on well-behaved runs.
            # delta_per_head is already computed above (hoisted alongside
            # the residual telemetry); reuse it here instead of recomputing.
            score_consistency_violation_mask = score_consistency_violations(
                top_int8_scores, fp16_block_scores, delta_per_head, eps_guard,
            )
            score_consistency_violation_heads = int(
                score_consistency_violation_mask.sum().item()
            )

    # Rung-4 (paper §3.4) full FP16 recomputation. When the score-consistency
    # monitor detects any head on this step where |FP16 − INT8| block scores
    # exceed Δ + eps_guard, Theorem 2's bound was empirically broken, so we
    # page in all FP16 keys+values from Tier 2 and recompute dense attention
    # for every head via SDPA. This guarantees dense-equivalent output on the
    # compromised step and subsumes the per-head Rung-3 action. Expected
    # zero-fire on well-behaved runs; observed zero across every cell of the
    # arXiv v1 sweep (4K/8K/16K/32K).
    rung4_fired = (
        score_consistency_check
        and score_consistency_violation_heads > 0
    )
    if rung4_fired:
        # Rung-4 pages in all FP16 keys+values; bookkeep H2D for telemetry.
        if cache.keys_fp16_gpu is None and cache.keys_fp16_cpu is not None:
            nt_r4 = cache.num_tokens
            kv_k, _, hd_k = cache.keys_fp16_cpu.shape
            h2d_key_bytes += kv_k * nt_r4 * hd_k * cache.keys_fp16_cpu.element_size()
            h2d_key_blocks += (nt_r4 + bs - 1) // bs
        zero_skip = torch.zeros_like(skip_mask)
        output = sdpa_attend_with_skip(
            cache, q_all, zero_skip, gqa_group, q_scale,
        )
        if collect_stats:
            total_blocks = num_q_heads * cache.num_blocks
            vram_fp16_key_cache_bytes = 0
            if cache.keys_fp16_gpu is not None:
                vram_fp16_key_cache_bytes = (
                    cache.keys_fp16_gpu[:, :cache.num_tokens, :].numel()
                    * cache.keys_fp16_gpu.element_size()
                )
            vram_fp16_value_cache_bytes = 0
            if cache.values_fp16 is not None:
                vram_fp16_value_cache_bytes = (
                    cache.values_fp16[:, :cache.num_tokens, :].numel()
                    * cache.values_fp16.element_size()
                )
            stats = {
                "total_blocks": total_blocks,
                # Paper-1 vocabulary: every block is attended; the split is
                # FP16-key (top-K*) vs INT8-key (tail). The legacy
                # skipped_blocks/skip_rate/attended_blocks keys are kept as
                # aliases so older readers don't break.
                "fp16_topk_blocks": total_blocks,
                "int8_tail_blocks": 0,
                "int8_tail_rate": 0.0,
                "skipped_blocks": 0,
                "skip_rate": 0.0,
                "attended_blocks": total_blocks,
                "v_format": "fp16",
                "score_consistency_violation_heads": score_consistency_violation_heads,
                "delta_bound_mean": float(delta_bound_mean),
                "delta_bound_max": delta_bound_max_value,
                # Residual telemetry also populated on Rung-4 steps (where
                # the bound was empirically violated, ratio_max ≥ 1.0).
                "score_residual_max": score_residual_max_value,
                "score_residual_mean": score_residual_mean_value,
                "score_residual_ratio_max": score_residual_ratio_max_value,
                "score_residual_ratio_mean": score_residual_ratio_mean_value,
                "score_residual_top_k": int(ranking_k),
                "eps_guard": float(eps_guard),
                "h2d_key_bytes": int(h2d_key_bytes),
                "h2d_key_blocks": int(h2d_key_blocks),
                "h2d_value_bytes": int(h2d_value_bytes),
                "h2d_value_blocks": int(h2d_value_blocks),
                "h2d_total_bytes": int(h2d_key_bytes + h2d_value_bytes),
                "vram_fp16_key_cache_bytes": int(vram_fp16_key_cache_bytes),
                "vram_fp16_value_cache_bytes": int(vram_fp16_value_cache_bytes),
                "rung1_fired": bool(rung1_triggered_heads > 0),
                "rung2_fired": bool(rung2_fired),
                "rung3_fired": False,  # Rung-4 subsumes Rung-3; no separate ranking recompute runs
                "rung4_fired": True,
                "rung4_violating_heads": score_consistency_violation_heads,
            }
        else:
            stats = {}
        return output, stats

    # Entropy gating: if attention is diffuse (no block dominates),
    # disable skipping for that head — small-mass blocks may carry critical
    # information (e.g., needle retrieval with weak signal).
    # Uses Phase 1 outputs so it's essentially free.
    if num_active_blocks > 0 and concentration_threshold > 0:
        # Per-block mass fraction per head
        m_global = m_b.amax(dim=1, keepdim=True)  # [num_q_heads, 1]
        log_mass = torch.log(S_b.clamp(min=1e-30)) + m_b - m_global
        mass = torch.exp(log_mass)  # [num_q_heads, num_active_blocks]
        total_mass = mass.sum(dim=1, keepdim=True).clamp(min=1e-30)
        mass_frac = mass / total_mass
        mass_max_per_head = mass_frac.max(dim=1).values  # [num_q_heads]
        # Diffuse heads: no single block has enough mass → don't skip anything
        diffuse_heads = mass_max_per_head < concentration_threshold
        if diffuse_heads.any():
            skip_mask[diffuse_heads, :] = False

    # Phase 2: Attend using SDPA for exact precision matching with dense path.
    # The Triton kernels compute in F32 which diverges from the BF16 SDPA used
    # in dense mode.  Using SDPA here ensures identical numerical behaviour.
    v_format = "fp16"

    # Paper §3.3 hybrid attend — Algorithm 1 Phase 2. When adaptive K* is
    # active and the cache has FP16 values in VRAM, route to the mask-gated
    # INT8/FP16 kernel so **every** block contributes to the output (top-K*
    # with FP16 keys, the rest with INT8 keys; no blocks are dropped). The
    # prior default SDPA-with-skip path drops tail blocks to mask=-inf,
    # which is a different algorithm (block skipping) and breaks the
    # paper's error bound.
    use_paper_hybrid = (
        adaptive_topk_mask is not None
        and cache.values_fp16 is not None
    )
    if use_paper_hybrid:
        # Iterate only fully-quantised blocks; the trailing partial block
        # (if any) has no INT8 data. We zero-out its scale before the call
        # so any speculative INT8 load is safe, then force topk_mask[trail]=1
        # so the kernel selects FP16 there. n_active_blocks_hybrid counts
        # the blocks we pass to the kernel.
        n_active_blocks_hybrid = n_qblocks
        keys_scale_active = cache.keys_scale_active()
        # Valid tokens in the LAST block passed to the kernel. When no
        # trailing partial block exists, the last block is full → block_size.
        last_block_valid = bs
        if cache.has_trailing_partial_block:
            n_active_blocks_hybrid = n_qblocks + 1
            # keys_scale is [kv_heads, num_blocks, head_dim]; the trailing
            # slot may be uninit from torch.empty allocation. Zero it so the
            # INT8 tile × scale path produces zeros regardless of int8 data,
            # and tl.where selects FP16 cleanly.
            cache.keys_scale[:, cache.trailing_block_idx, :].zero_()
            keys_scale_active = cache.keys_scale[:, :n_active_blocks_hybrid, :]
            last_block_valid = cache.num_tokens - n_qblocks * bs
            assert 1 <= last_block_valid < bs, last_block_valid

        hybrid_topk = adaptive_topk_mask[:, :n_active_blocks_hybrid].to(torch.int32).contiguous()
        # Force-attend every block — Paper 1: no skipping.
        no_skip = torch.zeros(
            num_q_heads, n_active_blocks_hybrid, dtype=torch.int32, device=q_all.device,
        )
        nt_hybrid = n_active_blocks_hybrid * bs
        keys_fp16_gpu = cache.keys_fp16_gpu
        if keys_fp16_gpu is None:
            with _PhaseTimer(phase_timings, "h2d_pagein"):
                keys_fp16_gpu = cache.keys_fp16_cpu.to(device=q_all.device, non_blocking=True)
            # Full-tensor H2D fallback (no VRAM buffer at all — rare).
            kv_k, _, hd_k = cache.keys_fp16_cpu.shape
            h2d_key_bytes += kv_k * nt_hybrid * hd_k * cache.keys_fp16_cpu.element_size()
            h2d_key_blocks += n_active_blocks_hybrid
        elif cache.fp16_key_cache_capacity is not None:
            # Paper-matching bounded FP16 cache: top-K blocks (union across
            # heads) must be resident in the VRAM scratch before the kernel
            # reads them. Miss → H2D from keys_fp16_cpu, evict LRU if full.
            # Trailing partial block is kept current by append_token writes
            # and doesn't need cache tracking.
            #
            # Priority-ordered iteration (paper §3.2, follow-up): the cache
            # is insert-MRU-last, so whatever we iterate LAST becomes the
            # hardest-to-evict. Sort ASCENDING by max m_b across heads so
            # high-scoring blocks (more likely needed next step) end up at
            # MRU-tail and survive longer; low-scoring blocks land near the
            # LRU-front and are evicted first on the next miss. This
            # replaces the prior block-ID-sorted iteration, which made
            # low-ID blocks systematically the LRU victims regardless of
            # their actual mass.
            top_union = adaptive_topk_mask[:, :n_qblocks].any(dim=0)
            union_block_ids = top_union.nonzero().flatten()
            if union_block_ids.numel() > 0:
                # Max score across heads — union-mass proxy.
                block_priority = m_b[:, :n_qblocks].amax(dim=0)[union_block_ids]
                sort_order = torch.argsort(block_priority, descending=False)
                needed_blocks = union_block_ids[sort_order].tolist()
            else:
                needed_blocks = []
            with _PhaseTimer(phase_timings, "h2d_pagein"):
                c_hits, c_misses, c_bytes, c_evict = cache.ensure_fp16_keys_resident(needed_blocks)
            h2d_key_bytes += c_bytes
            h2d_key_blocks += c_misses
            fp16_cache_hits_step += c_hits
            fp16_cache_misses_step += c_misses
            fp16_cache_evictions_step += c_evict
            fp16_cache_needed_blocks += len(needed_blocks)
        # DOTCACHE_FAST_ATTEND=0 reverts to the single-program-per-head
        # kernel for A/B comparison. Default is split-K (FlashDecoding-style),
        # which is 15-20× faster at 64K context on Blackwell (grid expands
        # from num_q_heads to num_q_heads × num_splits, filling the SMs).
        import os as _os
        _fast = _os.environ.get("DOTCACHE_FAST_ATTEND", "1") != "0"
        if _fast:
            # Stride-aware split-K kernel reads non-contig slices directly
            # via per-KV stride args — no pre-copy needed. torch.profiler
            # shows aten::copy_ at ~25 ms/step of self-CUDA time at 64K;
            # dropping the four per-layer .contiguous() calls eliminates
            # most of that.
            with _PhaseTimer(phase_timings, "phase2_fused_attend"):
                output = selective_attend_multihead_hybrid_split_k(
                    keys_int8=cache.keys_int8[:, :nt_hybrid, :],
                    keys_scale=keys_scale_active,
                    keys_fp16=keys_fp16_gpu[:, :nt_hybrid, :],
                    topk_mask=hybrid_topk,
                    values_fp16=cache.values_fp16[:, :nt_hybrid, :],
                    q_all=q_all,
                    skip_mask_i32=no_skip,
                    gqa_group=gqa_group,
                    block_size=bs,
                    q_scale=q_scale,
                    last_block_valid=last_block_valid,
                )
        else:
            with _PhaseTimer(phase_timings, "phase2_fused_attend"):
                output = selective_attend_multihead_hybrid(
                    keys_int8=cache.keys_int8[:, :nt_hybrid, :].contiguous(),
                    keys_scale=keys_scale_active.contiguous(),
                    keys_fp16=keys_fp16_gpu[:, :nt_hybrid, :].contiguous(),
                    topk_mask=hybrid_topk,
                    values_fp16=cache.values_fp16[:, :nt_hybrid, :].contiguous(),
                    q_all=q_all,
                    skip_mask_i32=no_skip,
                    gqa_group=gqa_group,
                    block_size=bs,
                    q_scale=q_scale,
                    last_block_valid=last_block_valid,
                )
    elif cache.values_fp16 is not None:
        # Legacy path: SDPA-with-skip. Tail blocks are masked to -inf (block
        # skipping — Paper 2 semantics). Used when adaptive K* is disabled.
        if cache.keys_fp16_gpu is None and cache.keys_fp16_cpu is not None:
            nt_sdpa = cache.num_tokens
            kv_k, _, hd_k = cache.keys_fp16_cpu.shape
            h2d_key_bytes += kv_k * nt_sdpa * hd_k * cache.keys_fp16_cpu.element_size()
            h2d_key_blocks += (nt_sdpa + bs - 1) // bs
        with _PhaseTimer(phase_timings, "phase2_fused_attend"):
            output = sdpa_attend_with_skip(
                cache, q_all, skip_mask, gqa_group, q_scale,
            )
    elif cache.values_int4_packed is not None:
        # INT4 values: must use Triton kernel (SDPA can't handle INT4)
        e_val_head: torch.Tensor | None = None
        # Fail fast on typos — silently loosening a certification check
        # because someone wrote "tigt" would invalidate the experiment
        # without a signal.
        if value_error_mode not in ("loose", "tight"):
            raise ValueError(
                f"value_error_mode must be 'loose' or 'tight', got {value_error_mode!r}"
            )
        if collect_stats:
            with _PhaseTimer(phase_timings, "value_check"):
                # Keep mass_frac + topk_idx so the tight bound can reuse
                # them without recomputing the softmax mass partition.
                # top_k here must match top_k_fp16_keys — the set of blocks
                # this layer actually force-attends. Using the module
                # default would make E_val's residual set diverge from the
                # attended set, invalidating the bound.
                rho, mass_frac, topk_idx = compute_tier2_residual_mass(
                    m_b, S_b, skip_mask,
                    top_k=top_k_fp16_keys,
                    return_details=True,
                )
                eta_int4 = cache.values_int4_errors.max().item()
                # Tight per-head bound Σ_b ρ_b · η_b. Always computed so
                # telemetry can report it, regardless of which bound the
                # v_format decision uses.
                e_val_head = compute_value_error_bound(
                    mass_frac=mass_frac,
                    topk_idx=topk_idx,
                    skip_mask=skip_mask,
                    eta_per_block=cache.values_int4_errors,
                    gqa_group=gqa_group,
                )
                if value_error_mode == "tight":
                    v_format = decide_v_format_tight(e_val_head, v_tolerance)
                else:
                    v_format = decide_v_format(rho, eta_int4, v_tolerance)
        else:
            v_format = "int4"

        if v_format == "int4":
            with _PhaseTimer(phase_timings, "phase2_fused_attend"):
                output = selective_attend_multihead_int8k_int4v(
                    keys_int8=cache.keys_int8_active(),
                    keys_scale=cache.keys_scale_active(),
                    values_int4_packed=cache.values_int4_packed,
                    values_int4_scales=cache.values_int4_scales,
                    values_int4_zeros=cache.values_int4_zeros,
                    q_all=q_all,
                    skip_mask_i32=skip_mask.to(torch.int32),
                    gqa_group=gqa_group,
                    block_size=cache.block_size,
                    group_size=cache.values_int4_group_size,
                    q_scale=q_scale,
                )
        else:
            # Rung-2 (paper §3.4): INT4 values unsafe (ρ·η > v_tolerance) —
            # page in FP16 values from the Tier-2 CPU pinned mirror.
            rung2_fired = True
            if cache.values_fp16_cpu is not None:
                with _PhaseTimer(phase_timings, "h2d_pagein"):
                    values_fp16 = cache.values_fp16_cpu.to(
                        device=cache.keys_int8.device, non_blocking=True,
                    )
                nt_v = cache.num_tokens
                kv_v, _, dv_v = cache.values_fp16_cpu.shape
                h2d_value_bytes += kv_v * nt_v * dv_v * cache.values_fp16_cpu.element_size()
                h2d_value_blocks += (nt_v + bs - 1) // bs
            elif cache.values_fp16 is not None:
                values_fp16 = cache.values_fp16
            else:
                raise ValueError("INT4 unsafe and no FP16 fallback available")
            with _PhaseTimer(phase_timings, "phase2_fused_attend"):
                output = selective_attend_multihead_int8(
                    keys_int8=cache.keys_int8_active(),
                    keys_scale=cache.keys_scale_active(),
                    values_fp16=values_fp16,
                    q_all=q_all,
                    skip_mask_i32=skip_mask.to(torch.int32),
                    gqa_group=gqa_group,
                    block_size=cache.block_size,
                    q_scale=q_scale,
                )
    else:
        raise ValueError("No values available in cache")

    # Rung-3 action: for every head whose INT8/FP16 rankings disagree on the
    # top-r positions, replace its output with a full FP16 dense attention.
    # Non-disagreeing heads keep their Phase-2 output exactly. torch.nonzero
    # already does the device→host transfer needed to shape the index
    # tensor, so a separate .any().item() guard would be a redundant sync.
    ranking_fallback_heads = 0
    if (
        ranking_fallback
        and ranking_fallback_mode == "full"
        and ranking_disagree_mask is not None
    ):
        disagree_heads = torch.nonzero(ranking_disagree_mask, as_tuple=True)[0]
        if disagree_heads.numel() > 0:
            output = recompute_heads_dense_fp16(
                cache=cache,
                q_all=q_all,
                output=output,
                head_indices=disagree_heads,
                gqa_group=gqa_group,
                q_scale=q_scale,
            )
            ranking_fallback_heads = int(disagree_heads.numel())

    # Stats
    if collect_stats:
        total_blocks = num_q_heads * cache.num_blocks
        # In the Paper-1 hybrid path (use_paper_hybrid=True above) every block
        # is attended. skip_mask here marks the blocks NOT in the adaptive
        # top-K* set, i.e. the ones attended with INT8 keys (cheap path)
        # rather than FP16 keys. We expose that under the int8_tail_* names
        # and keep skipped_blocks/skip_rate/attended_blocks as legacy aliases
        # so older readers (calibration, archived bench outputs) don't break.
        int8_tail = skip_mask.sum().item()
        stats = {
            "total_blocks": total_blocks,
            "int8_tail_blocks": int(int8_tail),
            "int8_tail_rate": float(int8_tail) / float(total_blocks),
            "fp16_topk_blocks": total_blocks - int(int8_tail),
            "skipped_blocks": int(int8_tail),
            "skip_rate": float(int8_tail) / float(total_blocks),
            "attended_blocks": total_blocks - int(int8_tail),
            "v_format": v_format,
        }
        # Page-in telemetry (paper §3.4 runtime cost accounting).
        stats["h2d_key_bytes"] = int(h2d_key_bytes)
        stats["h2d_key_blocks"] = int(h2d_key_blocks)
        stats["h2d_value_bytes"] = int(h2d_value_bytes)
        stats["h2d_value_blocks"] = int(h2d_value_blocks)
        stats["h2d_total_bytes"] = int(h2d_key_bytes + h2d_value_bytes)
        # FP16 key cache behavior (paper §3.2 tiered memory model). Populated
        # only when the cache is in bounded-capacity mode; in full-mirror mode
        # the attention path bypasses ensure_fp16_keys_resident entirely.
        if cache.fp16_key_cache_capacity is not None:
            stats["fp16_cache_capacity_blocks"] = int(cache.fp16_key_cache_capacity)
            stats["fp16_cache_resident_blocks"] = int(len(cache._fp16_key_resident))
            stats["fp16_cache_hits_step"] = int(fp16_cache_hits_step)
            stats["fp16_cache_misses_step"] = int(fp16_cache_misses_step)
            stats["fp16_cache_evictions_step"] = int(fp16_cache_evictions_step)
            stats["fp16_cache_needed_blocks_step"] = int(fp16_cache_needed_blocks)
            total_access = fp16_cache_hits_step + fp16_cache_misses_step
            stats["fp16_cache_hit_rate_step"] = (
                float(fp16_cache_hits_step) / total_access if total_access else 0.0
            )
        # VRAM-resident FP16 cache sizes (semantic bytes — kv_heads × nt × dim × 2).
        vram_fp16_key_cache_bytes = 0
        if cache.keys_fp16_gpu is not None:
            vram_fp16_key_cache_bytes = (
                cache.keys_fp16_gpu[:, :cache.num_tokens, :].numel()
                * cache.keys_fp16_gpu.element_size()
            )
        stats["vram_fp16_key_cache_bytes"] = int(vram_fp16_key_cache_bytes)
        vram_fp16_value_cache_bytes = 0
        if cache.values_fp16 is not None:
            vram_fp16_value_cache_bytes = (
                cache.values_fp16[:, :cache.num_tokens, :].numel()
                * cache.values_fp16.element_size()
            )
        stats["vram_fp16_value_cache_bytes"] = int(vram_fp16_value_cache_bytes)
        # Per-layer per-step rung-fired flags. Harness ORs across layers to
        # get a step-level rung-fired decision; aggregator sums to get rate.
        stats["rung1_fired"] = bool(rung1_triggered_heads > 0)
        stats["rung2_fired"] = bool(rung2_fired)
        stats["rung3_fired"] = bool(ranking_fallback_heads > 0)
        # rung4_fired here is always False — the Rung-4 path returns early
        # with its own stats dict; reaching this block means Rung-4 didn't
        # fire for this layer.
        stats["rung4_fired"] = False
        if cache.values_int4_packed is not None:
            stats["rho_max"] = rho.max().item()
            stats["rho_mean"] = rho.mean().item()
            stats["eta_int4"] = eta_int4
            # Loose bound used by decide_v_format's legacy scalar path.
            stats["int4_error_bound_loose"] = eta_int4 * rho.max().item()
            # Back-compat alias; removed once downstream harnesses migrate
            # to the explicit _loose / _tight naming.
            stats["int4_error_bound"] = stats["int4_error_bound_loose"]
            # Tight per-block bound Σ_b ρ_b · η_b, max across heads. This
            # is the "achieved" value-error upper bound per step the paper
            # should report alongside ε_val — always ≤ int4_error_bound_loose.
            if e_val_head is not None:
                stats["e_val_max"] = float(e_val_head.max().item())
                stats["e_val_mean"] = float(e_val_head.mean().item())
                stats["value_error_mode"] = value_error_mode
        # Adaptive K* telemetry (paper §3.3). Present only when enabled.
        if k_star is not None:
            stats["k_star_mean"] = float(k_star.float().mean().item())
            stats["k_star_min"] = int(k_star.min().item())
            stats["k_star_max"] = int(k_star.max().item())
            stats["tau_cov"] = float(tau_cov) if tau_cov is not None else 0.0
            stats["tau_cov_actual_mean"] = float(tau_cov_actual.mean().item())
            stats["tail_mass_int8_est_mean"] = float(tail_mass_est.mean().item())
            stats["tail_mass_int8_est_max"] = float(tail_mass_est.max().item())
            # Rung-1 fallback (expand K*) counters. Only relevant when adaptive
            # K* is active; zero on steps where no head hit the tail-mass gate.
            stats["rung1_triggered_heads"] = int(rung1_triggered_heads)
            stats["rung1_threshold"] = float(rung1_threshold)
            stats["rung1_multiplier"] = float(rung1_multiplier)
            # Exploration-budget telemetry (paper §6): blocks randomly added
            # to the attended set beyond adaptive K*. Does not affect the
            # certified bound; purely for monitoring.
            stats["exploration_rate"] = float(exploration_rate)
            stats["exploration_blocks"] = int(explored_blocks_count)
        # Paper §6 observed-vs-bound telemetry (reviewer Item 3). Emitted
        # whenever fp16_block_scores was materialised (i.e. ranking_fallback
        # or score_consistency_check); the residual quantities measure how
        # tight the per-block key-score delta bound is in practice.
        if need_fp16_scores:
            stats["score_residual_max"] = score_residual_max_value
            stats["score_residual_mean"] = score_residual_mean_value
            stats["score_residual_ratio_max"] = score_residual_ratio_max_value
            stats["score_residual_ratio_mean"] = score_residual_ratio_mean_value
            stats["delta_bound_mean"] = float(delta_bound_mean)
            stats["delta_bound_max"] = delta_bound_max_value
            stats["score_residual_top_k"] = int(ranking_k)
        # Score-consistency violation counters (paper §6). Always emitted
        # when the feature is enabled so runs can confirm the 0-count baseline.
        if score_consistency_check:
            stats["score_consistency_violation_heads"] = int(score_consistency_violation_heads)
            stats["eps_guard"] = float(eps_guard)
            # Rung-4 escalation telemetry. Reaching this stats block means
            # Rung-4 did NOT fire on this step (otherwise we returned early);
            # emitting 0/False here keeps the non-fired baseline visible
            # alongside the violation counter.
            stats["rung4_fired"] = False
            stats["rung4_violating_heads"] = 0
        # Ranking-consistency fallback telemetry (Rung 3).
        # Populated by the detection block above when enabled; the trigger
        # count is still zero here because commit 2 is detection-only — the
        # per-head fallback action arrives in the next commit.
        if ranking_fallback:
            stats["ranking_heads_total"] = num_q_heads
            stats["ranking_disagree_r1"] = int(ranking_disagree_r1_heads)
            stats["ranking_disagree_r3"] = int(ranking_disagree_r3_heads)
            stats["ranking_fallback_triggered"] = int(ranking_fallback_heads)
            stats["ranking_r"] = int(ranking_r)
            stats["ranking_k"] = int(ranking_k)
            stats["ranking_fallback_mode"] = ranking_fallback_mode
            # Score-gap diagnostics (spec §5) — only emitted when we actually
            # computed FP16 scores for at least one block per head.
            if fp16_block_scores is not None and fp16_block_scores.shape[1] > 0:
                # Top-1/top-2 gap on the FP16 re-rank: measures ranking fragility.
                # Larger gap → more stable ranking → disagreement less likely.
                if fp16_block_scores.shape[1] >= 2:
                    sorted_fp16 = fp16_block_scores.sort(dim=1, descending=True).values
                    gap_top12 = (sorted_fp16[:, 0] - sorted_fp16[:, 1]).float()
                    stats["score_gap_top12_mean"] = float(gap_top12.mean().item())
                    stats["score_gap_top12_min"] = float(gap_top12.min().item())
                int8_top1 = m_b[:, :n_qblocks].gather(
                    1, top_block_indices[:, :1]
                ).squeeze(1).float()
                fp16_top1 = fp16_block_scores.gather(
                    1, fp16_block_scores.argsort(dim=1, descending=True)[:, :1]
                ).squeeze(1).float()
                stats["s_int8_top1_mean"] = float(int8_top1.mean().item())
                stats["s_fp16_top1_mean"] = float(fp16_top1.mean().item())
    else:
        stats = {}

    return output, stats


def benchmark_certified_vs_full(
    cache: TieredKeyCacheLayer,
    q_all: torch.Tensor,
    gqa_group: int,
    iters: int = 1000,
    q_scale: float = None,
    block_epsilon: float = 0.001,
) -> dict[str, float]:
    """Benchmark certified attention vs full attention on one layer."""
    import time

    if q_scale is None:
        q_scale = 1.0 / (cache.head_dim ** 0.5)

    num_q_heads = q_all.shape[0]
    keys_fp32 = cache.keys_fp16_cpu.to(dtype=torch.float32, device=q_all.device)
    vals_fp32 = cache.values_fp16.to(torch.float32)

    # Warmup
    for _ in range(10):
        certified_attention_layer(cache, q_all, gqa_group, q_scale, block_epsilon)
        for qh in range(num_q_heads):
            kv = qh // gqa_group
            s = torch.matmul(keys_fp32[kv], q_all[qh]) * q_scale
            w = torch.softmax(s, dim=0)
            o = w @ vals_fp32[kv]
    torch.cuda.synchronize()

    # Full attention
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        for qh in range(num_q_heads):
            kv = qh // gqa_group
            s = torch.matmul(keys_fp32[kv], q_all[qh]) * q_scale
            w = torch.softmax(s, dim=0)
            o = w @ vals_fp32[kv]
    torch.cuda.synchronize()
    t_full = (time.perf_counter() - t0) / iters * 1e6

    # Certified attention
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        output, stats = certified_attention_layer(
            cache, q_all, gqa_group, q_scale, block_epsilon,
        )
    torch.cuda.synchronize()
    t_cert = (time.perf_counter() - t0) / iters * 1e6

    # Correctness
    output_cert, stats = certified_attention_layer(
        cache, q_all, gqa_group, q_scale, block_epsilon,
    )
    output_full = torch.empty_like(output_cert)
    for qh in range(num_q_heads):
        kv = qh // gqa_group
        s = torch.matmul(keys_fp32[kv], q_all[qh]) * q_scale
        w = torch.softmax(s, dim=0)
        output_full[qh] = w @ vals_fp32[kv]

    cos = torch.nn.functional.cosine_similarity(output_cert, output_full, dim=1)

    return {
        "full_attention_us": t_full,
        "certified_attention_us": t_cert,
        "speedup": (t_full - t_cert) / t_full,
        "int8_tail_rate": stats["int8_tail_rate"],
        # Legacy alias (Paper-2 vocabulary) — see decode_step stats comment.
        "skip_rate": stats["int8_tail_rate"],
        "cosine_min": cos.min().item(),
        "cosine_mean": cos.mean().item(),
        "vram_mb": cache.vram_bytes() / 1e6,
        "cpu_mb": cache.cpu_bytes() / 1e6,
    }
