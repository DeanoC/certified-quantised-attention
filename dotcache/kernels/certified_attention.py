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
import os as _os
import torch
import torch.nn.functional as F
from typing import Any

from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer
from dotcache.kernels.fused_score_certify import fused_score_certify_multihead
from dotcache.kernels.selective_attend_triton import (
    selective_attend_multihead,
    selective_attend_multihead_int8,
    selective_attend_multihead_int8k_int4v,
    selective_attend_multihead_hybrid,
    selective_attend_multihead_hybrid_int4v,
    selective_attend_multihead_hybrid_mixedv,
    selective_attend_multihead_hybrid_mixedv_split_k,
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


def compute_block_mass_fraction(
    m_b: torch.Tensor,
    S_b: torch.Tensor,
) -> torch.Tensor:
    """Normalised per-block attention mass from Phase-1 block stats."""
    m_global = m_b.amax(dim=1, keepdim=True)
    log_mass = torch.log(S_b.clamp(min=1e-30)) + m_b - m_global
    mass = torch.exp(log_mass)
    return mass / mass.sum(dim=1, keepdim=True).clamp(min=1e-30)


def compute_value_error_bound_for_mask(
    mass_frac: torch.Tensor,
    int4_value_mask: torch.Tensor,
    eta_per_block: torch.Tensor,
    gqa_group: int,
) -> torch.Tensor:
    """Σ_b mass[h,b]·η[kv(h),b] for blocks whose values use INT4."""
    num_q_heads, num_blocks_mass = mass_frac.shape
    num_kv_heads, num_blocks_eta = eta_per_block.shape
    num_blocks = min(num_blocks_mass, num_blocks_eta)
    mass_q = mass_frac[:, :num_blocks]
    mask_q = int4_value_mask[:, :num_blocks]
    eta_q = eta_per_block[:, :num_blocks]
    kv_idx = torch.arange(num_q_heads, device=mass_frac.device) // gqa_group
    kv_idx = kv_idx.clamp(max=num_kv_heads - 1)
    eta_for_q = eta_q[kv_idx]
    return (mass_q * mask_q.float() * eta_for_q).sum(dim=1)


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
    return_mass_frac: bool = False,
    return_selection_indices: bool = False,
    defer_mask: bool = False,
    return_sorted_cumsum: bool = False,
) -> tuple:
    """Paper §3.3 adaptive top-K* selector (cumulative-mass threshold).

    Per head: sort blocks by estimated mass, find smallest K such that
    cumulative mass ≥ `tau_cov`, clamp to [k_min, k_max]. Returns:

    - topk_mask [H, B] bool: True = block is in the top-K* for that head.
    - k_star    [H] int32: actual K* selected per head (post-clamp).
    - tail_mass [H] float32: 1 − Σ mass on top-K* (INT8-estimated).
    - tau_cov_actual [H] float32: actual cumulative mass captured at K*.
    - mass_frac [H, B] float32 when return_mass_frac=True, for downstream
      paper bounds that use the same normalised block mass distribution.
    - selected_idx_source [H|KV, W] and selected_k_star [H|KV] when
      return_selection_indices=True. These describe the same selected blocks
      without requiring a dense-mask nonzero scan downstream.
    - defer_mask=True skips constructing topk_mask when the caller will derive
      the final mask from selected_idx_source/k_star itself.
    - sorted_cumsum [H, W] when return_sorted_cumsum=True, for callers that
      need to derive a smaller base-k decision from an already expanded prefix.

    All computation stays on device; this function has zero CPU syncs.
    """
    num_q_heads, num_blocks = m_b.shape
    device = m_b.device
    if num_blocks == 0:
        empty_bool = torch.zeros(num_q_heads, 0, dtype=torch.bool, device=device)
        zeros_int = torch.zeros(num_q_heads, dtype=torch.int32, device=device)
        zeros_f32 = torch.zeros(num_q_heads, dtype=torch.float32, device=device)
        if return_mass_frac:
            empty_f32 = torch.zeros(num_q_heads, 0, dtype=torch.float32, device=device)
            result = (empty_bool, zeros_int, zeros_f32, zeros_f32, empty_f32)
        else:
            result = (empty_bool, zeros_int, zeros_f32, zeros_f32)
        if return_selection_indices:
            result = (*result, empty_bool.long(), zeros_int)
        if return_sorted_cumsum:
            return (*result, torch.zeros(num_q_heads, 0, dtype=torch.float32, device=device))
        return result

    if (
        m_b.is_cuda
        and S_b.is_cuda
        and m_b.dtype == torch.float32
        and S_b.dtype == torch.float32
        and not per_kv_group_topk
        and k_max is not None
        and int(k_max) > 0
        and _os.environ.get("DOTCACHE_NATIVE_ADAPTIVE_SELECTOR", "0") != "0"
    ):
        try:
            if torch.cuda.get_device_capability(m_b.device)[0] >= 12:
                hi_native = min(int(k_max), num_blocks)
                native_result = None
                if hi_native > 0:
                    from dotcache.backends.certified_blackwell import adaptive_topk_cuda

                    native_result = adaptive_topk_cuda(
                        m_b=m_b,
                        s_b=S_b,
                        tau_cov=tau_cov,
                        k_min=k_min,
                        k_max=hi_native,
                    )
                if native_result is not None:
                    (
                        topk_mask,
                        k_star,
                        tail_mass,
                        tau_actual,
                        mass_frac,
                        selected_idx,
                        selected_k_star,
                        sorted_cumsum,
                    ) = native_result
                    result = (
                        (topk_mask, k_star, tail_mass, tau_actual, mass_frac)
                        if return_mass_frac
                        else (topk_mask, k_star, tail_mass, tau_actual)
                    )
                    if return_selection_indices:
                        result = (*result, selected_idx, selected_k_star)
                    if return_sorted_cumsum:
                        result = (*result, sorted_cumsum)
                    return result
        except Exception:
            if _os.environ.get("DOTCACHE_NATIVE_ADAPTIVE_SELECTOR_STRICT", "0") != "0":
                raise

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
        result = (
            (topk_mask, k_star, tail_mass, tau_actual, mass_frac)
            if return_mass_frac
            else (topk_mask, k_star, tail_mass, tau_actual)
        )
        if return_selection_indices:
            result = (*result, sorted_idx, k_star_group)
        if return_sorted_cumsum:
            return (*result, cumsum)
        return result

    # Sort only the usable prefix when k_max caps K*. Full sorting all blocks
    # at 32K/64K is unnecessary for the paper config (k_max=128): if tau_cov
    # is not reached within the cap, K* clamps to k_max and only the top-k_max
    # set contributes to tau_actual/tail_mass. If tau_cov is reached earlier,
    # the prefix is exactly the same as a full descending sort.
    hi = num_blocks if k_max is None else min(int(k_max), num_blocks)
    if hi < num_blocks:
        sorted_mass, sorted_idx = mass_frac.topk(hi, dim=1, largest=True, sorted=True)
        selection_width = hi
    else:
        sorted_mass, sorted_idx = mass_frac.sort(dim=1, descending=True)
        selection_width = num_blocks
    cumsum = sorted_mass.cumsum(dim=1)                             # [H, B]

    # K*[h] = smallest k such that cumsum[h, k-1] ≥ tau_cov.
    # searchsorted on each row returns the insertion index of tau_cov;
    # since cumsum is non-decreasing in [0, 1], that index = (K* - 1).
    tau_vec = torch.full((num_q_heads, 1), float(tau_cov), device=device, dtype=cumsum.dtype)
    k_star = torch.searchsorted(cumsum, tau_vec).squeeze(1) + 1    # [H]
    # Clamp to [k_min, min(k_max, num_blocks)]. k_max=None means no cap
    # beyond num_blocks — let tau_cov alone dictate K* per head.
    lo = min(int(k_min), hi)
    k_star = k_star.clamp(min=lo, max=hi).to(torch.int32)

    # Build [H, B] top-K mask: position < k_star[h] in the sorted order.
    if defer_mask:
        topk_mask = torch.empty(num_q_heads, 0, dtype=torch.bool, device=device)
    else:
        pos = torch.arange(selection_width, device=device).unsqueeze(0)  # [1, selection_width]
        keep_sorted = pos < k_star.unsqueeze(1).to(pos.dtype)           # [H, B] bool
        topk_mask = torch.zeros(num_q_heads, num_blocks, dtype=torch.bool, device=device)
        topk_mask.scatter_(1, sorted_idx, keep_sorted)                 # [H, B]

    # Tail mass + actual coverage using cumsum at (K*-1).
    k_idx = (k_star.long() - 1).clamp(min=0, max=selection_width - 1).unsqueeze(1)
    tau_actual = cumsum.gather(1, k_idx).squeeze(1).float()
    tail_mass = (1.0 - tau_actual).clamp(min=0.0)
    result = (
        (topk_mask, k_star, tail_mass, tau_actual, mass_frac)
        if return_mass_frac
        else (topk_mask, k_star, tail_mass, tau_actual)
    )
    if return_selection_indices:
        result = (*result, sorted_idx, k_star)
    if return_sorted_cumsum:
        return (*result, cumsum)
    return result


def compute_fp16_block_scores(
    cache: TieredKeyCacheLayer,
    q_all: torch.Tensor,           # [num_q_heads, head_dim]
    block_indices: torch.Tensor,   # [num_q_heads, K] int64 block ids to score
    num_scoring_blocks: int,       # upper bound on valid block id (fully-quantized blocks)
    gqa_group: int,
    q_scale: float,
    return_log_mass: bool = False,
    keys_fp16_override: torch.Tensor | None = None,
    key_block_slots: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute per-head per-block FP16 scores for the given block set.

    Mirrors Phase-1's m_b (the per-block max pre-softmax logit) but uses the
    FP16 keys from the tiered cache's GPU mirror (or CPU if no mirror). Only
    blocks in [0, num_scoring_blocks) are valid; others receive -inf.

    Args:
        return_log_mass: when True, also returns per-block log-mass
            (paper §6.1 ℓ_b^fp16 = m_b^fp16 + log Σ_t exp(s_b,t - m_b^fp16))
            for use by the Eq. 30 boundary verification.

    Returns: [num_q_heads, K] float32 block max-logits, suitable for ranking.
        If return_log_mass: (max_logits, log_masses) — both [num_q_heads, K].
    """
    num_q_heads, head_dim = q_all.shape
    _, K = block_indices.shape
    bs = cache.block_size
    device = q_all.device

    # Total tokens covered by the fully-quantized block range.
    nt = num_scoring_blocks * bs

    neg_inf = torch.full((num_q_heads, K), float("-inf"), dtype=torch.float32, device=device)
    if nt == 0 or K == 0:
        if return_log_mass:
            return neg_inf, neg_inf.clone()
        return neg_inf

    if keys_fp16_override is not None:
        keys = keys_fp16_override
    elif cache.keys_fp16_gpu is not None:
        keys = cache.keys_fp16_gpu[:, :nt, :]
    else:
        keys = cache.keys_fp16_cpu[:, :nt, :].to(device=device, non_blocking=True)
    if keys.dtype != q_all.dtype:
        keys = keys.to(dtype=q_all.dtype)

    if keys.device.type == "cuda" and q_all.device.type == "cuda" and head_dim <= 256:
        try:
            if _os.environ.get("DOTCACHE_FP16_BLOCK_SCORE_TRITON", "1") != "0":
                from dotcache.kernels.fp16_block_scores_triton import fp16_block_scores_triton
                scores, log_masses = fp16_block_scores_triton(
                    keys,
                    q_all,
                    block_indices,
                    num_scoring_blocks=num_scoring_blocks,
                    gqa_group=gqa_group,
                    block_size=bs,
                    q_scale=q_scale,
                    key_block_slots=key_block_slots,
                )
                if return_log_mass:
                    return scores, log_masses
                return scores
        except Exception:
            pass

    # [num_q_heads, K, bs, head_dim] gather: for each (h, k) pick tokens
    # [block*bs, block*bs + bs) from keys[kv_h].
    kv_per_h = torch.arange(num_q_heads, device=device) // gqa_group          # [H]
    kv_per_hk = kv_per_h.unsqueeze(1).expand(-1, K)                            # [H, K]
    if key_block_slots is None:
        starts = block_indices.to(torch.long) * bs                             # [H, K]
    else:
        if int(key_block_slots.numel()) < int(num_scoring_blocks):
            padded_slots = torch.full(
                (int(num_scoring_blocks),), -1,
                dtype=key_block_slots.dtype, device=device,
            )
            padded_slots[: int(key_block_slots.numel())] = key_block_slots.to(device=device)
            key_block_slots = padded_slots
        slot_idx = block_indices.to(torch.long).clamp(
            min=0, max=max(int(num_scoring_blocks) - 1, 0),
        )
        starts = key_block_slots[slot_idx] * bs
    token_offsets = torch.arange(bs, device=device)                            # [bs]
    token_idx = starts.unsqueeze(-1) + token_offsets                           # [H, K, bs]
    valid_block = (block_indices >= 0) & (block_indices < num_scoring_blocks)
    valid = (
        (token_idx < keys.shape[1])
        & (starts.unsqueeze(-1) >= 0)
        & valid_block.unsqueeze(-1)
    )                                                                          # [H, K, bs]
    # Clamp out-of-range indices so the gather is always valid; masked later.
    token_idx_clamped = token_idx.clamp(min=0, max=max(int(keys.shape[1]) - 1, 0))

    # keys[kv, t]: fancy indexing with [H, K, bs] index tensors.
    kv_idx = kv_per_hk.unsqueeze(-1).expand(-1, -1, bs)                        # [H, K, bs]
    k_gathered = keys[kv_idx, token_idx_clamped]                               # [H, K, bs, head_dim]

    # Dot with q_h: q_all [H, head_dim] → [H, 1, 1, head_dim]
    q_expanded = q_all.unsqueeze(1).unsqueeze(1)
    logits = (k_gathered.float() * q_expanded.float()).sum(dim=-1) * q_scale   # [H, K, bs]
    neg_inf_tok = torch.full_like(logits, float("-inf"))
    logits = torch.where(valid, logits, neg_inf_tok)
    scores = logits.amax(dim=-1)                                               # [H, K]
    if return_log_mass:
        # Per-block log-mass via numerically-stable log-sum-exp. Blocks with
        # no valid tokens (all -inf logits) get -inf log-mass.
        log_masses = torch.logsumexp(logits, dim=-1)                            # [H, K]
        return scores, log_masses
    return scores


def _priority_order_blocks(
    block_ids: torch.Tensor,
    m_b: torch.Tensor,
    n_qblocks: int,
) -> list[int]:
    if block_ids.numel() == 0:
        return []
    block_priority = m_b[:, :n_qblocks].amax(dim=0)[block_ids]
    sort_order = torch.argsort(block_priority, descending=False)
    return [int(b) for b in block_ids[sort_order].tolist()]


def _priority_order_selected_blocks(
    sorted_idx: torch.Tensor | None,
    selected_k_star: torch.Tensor | None,
    m_b: torch.Tensor,
    n_qblocks: int,
) -> list[int]:
    """Build the same page-in union from adaptive top-K selector outputs.

    This avoids `topk_mask.nonzero()` on CUDA. We still need one CPU transfer
    because the bounded page-in/LRU layer is CPU-orchestrated.
    """
    if sorted_idx is None or selected_k_star is None or sorted_idx.numel() == 0:
        return []
    width = sorted_idx.shape[1]
    pos = torch.arange(width, device=sorted_idx.device).unsqueeze(0)
    valid = pos < selected_k_star.to(device=sorted_idx.device, dtype=pos.dtype).unsqueeze(1)
    selected = torch.where(valid, sorted_idx, torch.full_like(sorted_idx, -1)).reshape(-1)

    seen: set[int] = set()
    unique_ids: list[int] = []
    for bid in selected.detach().cpu().tolist():
        ib = int(bid)
        if 0 <= ib < n_qblocks and ib not in seen:
            seen.add(ib)
            unique_ids.append(ib)
    if not unique_ids:
        return []

    ids = torch.tensor(unique_ids, dtype=torch.long, device=m_b.device)
    priorities = m_b[:, :n_qblocks].amax(dim=0).index_select(0, ids).detach().cpu().tolist()
    return [
        bid for bid, _priority in sorted(
            zip(unique_ids, priorities, strict=True),
            key=lambda item: item[1],
        )
    ]


def _prepare_bounded_fp16_key_scratch(
    cache: TieredKeyCacheLayer,
    needed_blocks: list[int],
    n_active_blocks: int,
    device: torch.device,
    phase_timings: dict | None,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int, int]:
    """Ensure a compact FP16 key scratch covers needed block ids.

    Returns (keys_scratch, block_slots, hits, misses, bytes, evictions,
    needed_count). ``block_slots[bid]`` maps original block id to scratch slot.
    This keeps the paper path bounded: scratch shape is capacity*B or the
    one-step working set, never a full-context FP16 mirror unless the caller
    explicitly configured full-mirror mode elsewhere.
    """
    if not needed_blocks:
        scratch = cache.keys_fp16_gpu
        if scratch is None:
            scratch = torch.empty(
                cache.kv_heads, 0, cache.head_dim,
                dtype=cache.keys_fp16_cpu.dtype, device=device,
            )
            slots = torch.empty((0,), dtype=torch.int32, device=device)
        else:
            slots = cache.fp16_key_block_slots_gpu(n_active_blocks)
        return scratch, slots, 0, 0, 0, 0, 0

    # Deduplicate while preserving the caller's priority order.
    seen: set[int] = set()
    ordered: list[int] = []
    for bid in needed_blocks:
        ib = int(bid)
        if ib < 0 or ib >= n_active_blocks or ib in seen:
            continue
        seen.add(ib)
        ordered.append(ib)

    cap = int(cache.fp16_key_cache_capacity or 0)
    slot_items = []
    if cache.keys_fp16_gpu is not None and cap > 0 and len(ordered) <= cap:
        with _PhaseTimer(phase_timings, "h2d_pagein"):
            _block_to_slot, hits, misses, h2d_bytes, evictions = (
                cache.ensure_fp16_keys_resident_batched(ordered)
            )
        scratch = cache.keys_fp16_gpu
        slots = cache.fp16_key_block_slots_gpu(n_active_blocks)
    else:
        slots = torch.full((n_active_blocks,), -1, dtype=torch.int32, device=device)
        hits = 0
        misses = len(ordered)
        evictions = 0
        h2d_bytes = 0
        scratch = torch.empty(
            cache.kv_heads,
            max(len(ordered), 1) * cache.block_size,
            cache.head_dim,
            dtype=cache.keys_fp16_cpu.dtype,
            device=device,
        )
        with _PhaseTimer(phase_timings, "h2d_pagein"):
            for slot, bid in enumerate(ordered):
                start = bid * cache.block_size
                end = min(start + cache.block_size, cache.num_tokens)
                dst_start = slot * cache.block_size
                dst_end = dst_start + (end - start)
                if end > start:
                    src = cache.keys_fp16_cpu[:, start:end, :]
                    scratch[:, dst_start:dst_end, :].copy_(src, non_blocking=True)
                    h2d_bytes += (
                        cache.kv_heads * (end - start) * cache.head_dim
                        * cache.keys_fp16_cpu.element_size()
                    )
                slot_items.append((bid, slot))

    if slot_items:
        ids = torch.tensor([bid for bid, _ in slot_items], dtype=torch.long, device=device)
        slot_vals = torch.tensor([slot for _, slot in slot_items], dtype=torch.int32, device=device)
        slots[ids] = slot_vals
    return scratch, slots, hits, misses, h2d_bytes, evictions, len(ordered)


def _cert_workspace(cache: TieredKeyCacheLayer) -> dict[str, torch.Tensor]:
    workspace = getattr(cache, "_certified_workspace", None)
    if workspace is None:
        workspace = {}
        cache._certified_workspace = workspace
    return workspace


def _workspace_tensor(
    cache: TieredKeyCacheLayer,
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    fill_value: int | float | None = None,
) -> torch.Tensor:
    workspace = _cert_workspace(cache)
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
    out = current
    if fill_value is not None:
        out.fill_(fill_value)
    return out


def _workspace_arange(
    cache: TieredKeyCacheLayer,
    name: str,
    length: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    workspace = _cert_workspace(cache)
    current = workspace.get(name)
    if (
        current is None
        or current.dtype != dtype
        or current.device != device
        or current.ndim != 1
        or int(current.shape[0]) < int(length)
    ):
        current = torch.arange(int(length), dtype=dtype, device=device)
        workspace[name] = current
    return current[: int(length)]


def _cpu_pinned_workspace(
    cache: TieredKeyCacheLayer,
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    workspace = _cert_workspace(cache)
    current = workspace.get(name)
    needs_alloc = (
        current is None
        or current.dtype != dtype
        or current.device.type != "cpu"
        or not current.is_pinned()
        or tuple(int(x) for x in current.shape) != tuple(int(x) for x in shape)
    )
    if needs_alloc:
        current = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        workspace[name] = current
    return current


def _identity_resident_blocks(cache: TieredKeyCacheLayer, n_active_blocks: int, *, kind: str) -> bool:
    if kind == "key":
        capacity = cache.fp16_key_cache_capacity
        resident = cache._fp16_key_resident
        gpu = cache.keys_fp16_gpu
    elif kind == "value":
        capacity = cache.fp16_value_cache_capacity
        resident = cache._fp16_value_resident
        gpu = cache.values_fp16_gpu
    else:
        raise ValueError(f"unknown resident-block kind {kind!r}")
    if gpu is None or capacity is None or int(capacity) < int(n_active_blocks):
        return False
    if len(resident) < int(n_active_blocks):
        return False
    return all(resident.get(i) == i for i in range(int(n_active_blocks)))


def recompute_heads_dense_fp16(
    cache: TieredKeyCacheLayer,
    q_all: torch.Tensor,               # [num_q_heads, head_dim]
    output: torch.Tensor,              # [num_q_heads, d_v] to be patched in-place
    head_indices: torch.Tensor,        # [num_to_recompute] int64 q-head ids
    gqa_group: int,
    q_scale: float,
) -> tuple[torch.Tensor, int, int, int, int]:
    """Rung-3 recompute: for each listed head, replace output[h] with a full
    FP16 dense attention using the cache's FP16 keys + FP16 values (dequantised
    from INT4 if that's the value tier).

    The recompute only touches listed heads — non-disagreeing heads keep their
    Phase-2 output unchanged. This is intentional: the spec calls for per-head
    granularity so only the heads that paid the detection are corrected.
    """
    if head_indices.numel() == 0:
        return output, 0, 0, 0, 0
    nt = cache.num_tokens
    device = q_all.device
    bs = cache.block_size
    heads_cpu = [int(h) for h in head_indices.detach().cpu().tolist()]
    kv_ids_cpu = sorted({h // int(gqa_group) for h in heads_cpu})
    kv_to_local = {kv: i for i, kv in enumerate(kv_ids_cpu)}
    local_kv_ids = torch.tensor(
        [kv_to_local[h // int(gqa_group)] for h in heads_cpu],
        dtype=torch.long,
        device=device,
    )
    heads = torch.tensor(heads_cpu, dtype=torch.long, device=device)
    kv_ids = torch.tensor(kv_ids_cpu, dtype=torch.long, device=device)
    h2d_key_bytes = 0
    h2d_key_blocks = 0
    h2d_value_bytes = 0
    h2d_value_blocks = 0

    n_active_blocks = (nt + bs - 1) // bs
    if (
        cache.keys_fp16_gpu is not None
        and (
            cache.fp16_key_cache_capacity is None
            or _identity_resident_blocks(cache, n_active_blocks, kind="key")
        )
    ):
        keys = cache.keys_fp16_gpu[:, :nt, :].index_select(0, kv_ids)
    else:
        keys = torch.empty(
            len(kv_ids_cpu), nt, cache.head_dim,
            dtype=cache.keys_fp16_cpu.dtype,
            device=device,
        )
        for local, kvh in enumerate(kv_ids_cpu):
            keys[local].copy_(cache.keys_fp16_cpu[kvh, :nt, :], non_blocking=True)
        h2d_key_bytes = len(kv_ids_cpu) * nt * cache.head_dim * cache.keys_fp16_cpu.element_size()
        h2d_key_blocks = len(kv_ids_cpu) * ((nt + bs - 1) // bs)

    if cache.values_fp16 is not None:
        values = cache.values_fp16[:, :nt, :].index_select(0, kv_ids)
    elif (
        cache.values_fp16_gpu is not None
        and (
            cache.fp16_value_cache_capacity is None
            or _identity_resident_blocks(cache, n_active_blocks, kind="value")
        )
    ):
        values = cache.values_fp16_gpu[:, :nt, :].index_select(0, kv_ids)
    elif cache.values_fp16_cpu is not None:
        values = torch.empty(
            len(kv_ids_cpu), nt, cache.d_v,
            dtype=cache.values_fp16_cpu.dtype,
            device=device,
        )
        for local, kvh in enumerate(kv_ids_cpu):
            values[local].copy_(cache.values_fp16_cpu[kvh, :nt, :], non_blocking=True)
        h2d_value_bytes = len(kv_ids_cpu) * nt * cache.d_v * cache.values_fp16_cpu.element_size()
        h2d_value_blocks = len(kv_ids_cpu) * ((nt + bs - 1) // bs)
    else:
        values = cache.get_values_f32()[:, :nt, :].index_select(0, kv_ids)
    values_f32 = values.to(device=device, dtype=torch.float32)
    keys_f32 = keys.to(device=device, dtype=torch.float32)

    # Loop-free per-head recompute: pull the rows we need and vectorise the
    # dot-products. head_indices is typically small (≤ num_q_heads).
    q_sel = q_all.index_select(0, heads).float()                        # [M, head_dim]
    k_sel = keys_f32.index_select(0, local_kv_ids)                      # [M, nt, head_dim]
    v_sel = values_f32.index_select(0, local_kv_ids)                    # [M, nt, d_v]
    logits = torch.einsum("mnd,md->mn", k_sel, q_sel) * q_scale        # [M, nt]
    weights = torch.softmax(logits, dim=1)                              # [M, nt]
    head_out = torch.einsum("mn,mnd->md", weights, v_sel)              # [M, d_v]
    output.index_copy_(0, heads, head_out.to(output.dtype))
    return output, h2d_key_bytes, h2d_key_blocks, h2d_value_bytes, h2d_value_blocks


def augment_mask_with_exploration(
    topk_mask: torch.Tensor,         # [H, B] bool — top-K* mask from adaptive selector
    exploration_rate: float,
    generator: torch.Generator | None = None,
    count: bool = True,
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
    explored = int(exploration_mask.sum().item()) if count else 0
    return augmented, exploration_mask, explored


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

    # Full mirrors are only valid when capacity is None. Bounded cache tensors
    # are compact scratch, not full-context layouts.
    if cache.keys_fp16_gpu is not None and cache.fp16_key_cache_capacity is None:
        keys = cache.keys_fp16_gpu[:, :nt, :]
        if keys.dtype != dtype:
            keys = keys.to(dtype=dtype)
    else:
        keys = cache.keys_fp16_cpu[:, :nt, :].to(device=device, dtype=dtype)
    if cache.values_fp16 is not None:
        values = cache.values_fp16[:, :nt, :]
    elif cache.values_fp16_gpu is not None and cache.fp16_value_cache_capacity is None:
        values = cache.values_fp16_gpu[:, :nt, :]
    elif cache.values_fp16_cpu is not None:
        values = cache.values_fp16_cpu[:, :nt, :].to(device=device, dtype=dtype)
    else:
        values = cache.get_values_f32()[:, :nt, :]
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
    collect_stats: bool = True,
    *,
    v_tolerance: float,
    top_k_fp16_keys: int = 0,
    ranking_fallback: bool = False,
    ranking_r: int = 1,
    ranking_fallback_mode: str = "full",
    tau_cov: float | None = None,
    k_min: int = DEFAULT_K_MIN,
    k_max: int | None = DEFAULT_K_MAX,
    rung1_threshold: float = DEFAULT_RUNG1_THRESHOLD,
    rung1_multiplier: float = DEFAULT_RUNG1_MULTIPLIER,
    score_consistency_check: bool = True,
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
    fp16_value_cache_hits_step = 0
    fp16_value_cache_misses_step = 0
    fp16_value_cache_evictions_step = 0
    fp16_value_cache_needed_blocks = 0
    fp16_value_cache_overflow_step = 0
    mixedv_splitk_fallback_step = 0
    prefetched_keys_fp16_gpu: torch.Tensor | None = None
    prefetched_key_block_slots: torch.Tensor | None = None
    int8_token_scores: torch.Tensor | None = None

    # Phase 1: INT8 scoring only on fully quantized blocks
    if n_qblocks > 0:
        with _PhaseTimer(phase_timings, "phase1_int8_scoring"):
            # block_epsilon=0.0 forces an all-False skip_mask: the paper
            # flow attends every block (top-K via adaptive K*, tail via
            # paper §4.5 E_key bound). Legacy block-mass skipping is
            # permanently disabled here — do NOT re-expose block_epsilon
            # as a parameter.
            reuse_int8_token_scores = (
                cache.values_int4_packed is not None
                and _os.environ.get("DOTCACHE_CERTIFIED_BACKEND", "triton").strip().lower() == "native_blackwell"
                and _os.environ.get("DOTCACHE_REUSE_INT8_TOKEN_SCORES", "0").strip().lower()
                not in {"0", "false", "no", "off"}
            )
            use_native_phase1_score = (
                cache.values_int4_packed is not None
                and not reuse_int8_token_scores
                and _os.environ.get("DOTCACHE_CERTIFIED_BACKEND", "triton").strip().lower() == "native_blackwell"
                and _os.environ.get("DOTCACHE_NATIVE_PHASE1_SCORE", "1").strip().lower()
                not in {"0", "false", "no", "off"}
            )
            if use_native_phase1_score:
                try:
                    from dotcache.backends.certified_blackwell import (
                        certified_blackwell_available,
                        score_blocks_cuda,
                    )
                    if certified_blackwell_available():
                        m_b, S_b = score_blocks_cuda(
                            keys_int8=cache.keys_int8[:, :n_qblocks * bs, :],
                            keys_scale=cache.keys_scale[:, :n_qblocks, :],
                            keys_zero_points=cache.keys_zero_points[:, :n_qblocks, :],
                            q_all=q_all,
                            gqa_group=gqa_group,
                            block_size=bs,
                            q_scale=q_scale,
                        )
                        skip_mask = _workspace_tensor(
                            cache,
                            "phase1_skip_mask_false_bool",
                            (num_q_heads, n_qblocks),
                            dtype=torch.bool,
                            device=q_all.device,
                            fill_value=False,
                        )
                    else:
                        use_native_phase1_score = False
                except Exception:
                    use_native_phase1_score = False
            if not use_native_phase1_score:
                _score_result = fused_score_certify_multihead(
                    K_int8_packed=cache.keys_int8[:, :n_qblocks * bs, :],
                    K_scale=cache.keys_scale[:, :n_qblocks, :],
                    K_zero_points=cache.keys_zero_points[:, :n_qblocks, :],
                    q_all=q_all,
                    correction=cache.correction[:, :n_qblocks],
                    gqa_group=gqa_group,
                    block_size=bs,
                    q_scale=q_scale,
                    block_epsilon=0.0,
                    return_token_scores=reuse_int8_token_scores,
                )
                if len(_score_result) == 4:
                    m_b, S_b, skip_mask, int8_token_scores = _score_result
                else:
                    m_b, S_b, skip_mask = _score_result
    else:
        device = q_all.device
        m_b = torch.empty(num_q_heads, 0, dtype=torch.float32, device=device)
        S_b = torch.empty(num_q_heads, 0, dtype=torch.float32, device=device)
        skip_mask = torch.empty(num_q_heads, 0, dtype=torch.bool, device=device)

    adaptive_enabled = tau_cov is not None and tau_cov > 0 and n_qblocks > 0

    # If there's a trailing partial block, force-attend it via hybrid FP16 path.
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
    if top_k_fp16 > 0 and num_active_blocks > 0 and not adaptive_enabled:
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
    adaptive_topk_mask_i32: torch.Tensor | None = None
    k_star: torch.Tensor | None = None
    tail_mass_est: torch.Tensor | None = None
    tau_cov_actual: torch.Tensor | None = None
    mass_frac_cert: torch.Tensor | None = None
    adaptive_selected_idx: torch.Tensor | None = None
    adaptive_selected_k_star: torch.Tensor | None = None
    adaptive_sorted_cumsum: torch.Tensor | None = None
    rung1_triggered_heads = 0
    explored_blocks_count = 0
    if adaptive_enabled:
        topk_mask_cert_i32: torch.Tensor | None = None
        build_topk_i32_mask = (
            _os.environ.get("DOTCACHE_BUILD_TOPK_I32_MASK", "0").strip().lower()
            not in {"0", "false", "no", "off"}
        )
        # Restrict adaptive selection to the fully-quantised block range —
        # the trailing partial block has no INT8 score and is force-attended
        # below. Build the [H, n_blocks] mask by padding the fully-quantised
        # selection with the trailing block forced in.
        m_b_cert = m_b[:, :n_qblocks]
        S_b_cert = S_b[:, :n_qblocks]
        # If Rung-1 can expand k_max, compute the expanded sorted prefix once
        # and derive both the base-k and expanded-k decisions from it below.
        # This preserves the paper rule while avoiding two top-k passes per
        # layer at long context.
        rung1_can_expand = (
            rung1_threshold is not None and rung1_threshold >= 0
            and k_max is not None
            and not per_kv_group_topk
        )
        expanded_k_max = None
        selection_k_max = k_max
        if rung1_can_expand:
            expanded_k_max = min(int(math.ceil(int(k_max) * float(rung1_multiplier))), n_qblocks)
            if expanded_k_max > int(k_max):
                selection_k_max = expanded_k_max

        with _PhaseTimer(phase_timings, "adaptive_selection"):
            _topk_result = compute_adaptive_topk_mask(
                m_b_cert, S_b_cert, tau_cov=tau_cov, k_min=k_min, k_max=selection_k_max,
                per_kv_group_topk=per_kv_group_topk, gqa_group=gqa_group,
                return_mass_frac=(cache.values_int4_packed is not None or rung1_can_expand),
                return_selection_indices=True,
                defer_mask=(
                    rung1_can_expand
                    and expanded_k_max is not None
                    and expanded_k_max > int(k_max)
                ),
                return_sorted_cumsum=(
                    rung1_can_expand
                    and expanded_k_max is not None
                    and expanded_k_max > int(k_max)
                ),
            )
            if len(_topk_result) == 8:
                (
                    topk_mask_cert,
                    k_star,
                    tail_mass_est,
                    tau_cov_actual,
                    mass_frac_cert,
                    adaptive_selected_idx,
                    adaptive_selected_k_star,
                    adaptive_sorted_cumsum,
                ) = _topk_result
            elif len(_topk_result) == 7:
                (
                    topk_mask_cert,
                    k_star,
                    tail_mass_est,
                    tau_cov_actual,
                    mass_frac_cert,
                    adaptive_selected_idx,
                    adaptive_selected_k_star,
                ) = _topk_result
            else:
                (
                    topk_mask_cert,
                    k_star,
                    tail_mass_est,
                    tau_cov_actual,
                    adaptive_selected_idx,
                    adaptive_selected_k_star,
                ) = _topk_result

        # Rung-1 (paper §3.4): if any head's base-k tail mass exceeded the
        # configured threshold, use the expanded-k decision for that head.
        # For the normal per-head paper selector this is derived from the same
        # expanded sorted prefix computed above; the experimental per-KV-group
        # selector keeps the old two-pass path because its index shape differs.
        if rung1_can_expand and expanded_k_max is not None and expanded_k_max > int(k_max):
            with _PhaseTimer(phase_timings, "adaptive_selection"):
                assert adaptive_selected_idx is not None
                assert mass_frac_cert is not None
                assert adaptive_sorted_cumsum is not None
                k_star_expanded = k_star
                tail_mass_expanded = tail_mass_est
                tau_cov_actual_expanded = tau_cov_actual

                base_width = min(int(k_max), adaptive_sorted_cumsum.shape[1])
                cumsum_base = adaptive_sorted_cumsum[:, :base_width].contiguous()
                tau_vec = torch.full(
                    (num_q_heads, 1),
                    float(tau_cov),
                    device=q_all.device,
                    dtype=cumsum_base.dtype,
                )
                k_star_base = torch.searchsorted(cumsum_base, tau_vec).squeeze(1) + 1
                lo = min(int(k_min), base_width)
                k_star_base = k_star_base.clamp(min=lo, max=base_width).to(torch.int32)
                k_idx = (k_star_base.long() - 1).clamp(min=0, max=base_width - 1).unsqueeze(1)
                tau_cov_actual_base = cumsum_base.gather(1, k_idx).squeeze(1).float()
                tail_mass_base = (1.0 - tau_cov_actual_base).clamp(min=0.0)
                rung1_trigger_mask = tail_mass_base > rung1_threshold
                if collect_stats:
                    rung1_triggered_heads = int(rung1_trigger_mask.sum().item())

                k_star = torch.where(rung1_trigger_mask, k_star_expanded, k_star_base)
                tail_mass_est = torch.where(
                    rung1_trigger_mask, tail_mass_expanded, tail_mass_base,
                )
                tau_cov_actual = torch.where(
                    rung1_trigger_mask, tau_cov_actual_expanded, tau_cov_actual_base,
                )

                pos = torch.arange(
                    adaptive_selected_idx.shape[1],
                    device=q_all.device,
                ).unsqueeze(0)
                keep_sorted = pos < k_star.unsqueeze(1).to(pos.dtype)
                topk_mask_cert = _workspace_tensor(
                    cache,
                    "adaptive_topk_mask_cert_bool",
                    (num_q_heads, n_qblocks),
                    dtype=torch.bool,
                    device=q_all.device,
                    fill_value=False,
                )
                topk_mask_cert.scatter_(1, adaptive_selected_idx, keep_sorted)
                if build_topk_i32_mask:
                    topk_mask_cert_i32 = _workspace_tensor(
                        cache,
                        "adaptive_topk_mask_cert_i32",
                        (num_q_heads, n_qblocks),
                        dtype=torch.int32,
                        device=q_all.device,
                        fill_value=0,
                    )
                    topk_mask_cert_i32.scatter_(
                        1, adaptive_selected_idx, keep_sorted.to(torch.int32),
                    )
                adaptive_selected_k_star = k_star
        elif (
            rung1_threshold is not None and rung1_threshold >= 0
            and k_max is not None
        ):
            rung1_trigger_mask = tail_mass_est > rung1_threshold  # [H] bool
            if collect_stats:
                rung1_triggered_heads = int(rung1_trigger_mask.sum().item())
            expanded_k_max_fallback = min(int(math.ceil(k_max * float(rung1_multiplier))), n_qblocks)
            if expanded_k_max_fallback > int(k_max):
                topk_result2 = compute_adaptive_topk_mask(
                    m_b_cert, S_b_cert, tau_cov=tau_cov, k_min=k_min, k_max=expanded_k_max_fallback,
                    per_kv_group_topk=per_kv_group_topk, gqa_group=gqa_group,
                    return_selection_indices=True,
                )
                (
                    topk_mask_cert2,
                    k_star2,
                    tail_mass_est2,
                    tau_cov_actual2,
                    adaptive_selected_idx2,
                    adaptive_selected_k_star2,
                ) = topk_result2
                trig = rung1_trigger_mask.unsqueeze(1)
                topk_mask_cert = torch.where(trig, topk_mask_cert2, topk_mask_cert)
                k_star = torch.where(rung1_trigger_mask, k_star2, k_star)
                tail_mass_est = torch.where(rung1_trigger_mask, tail_mass_est2, tail_mass_est)
                tau_cov_actual = torch.where(rung1_trigger_mask, tau_cov_actual2, tau_cov_actual)
                adaptive_selected_idx = None
                adaptive_selected_k_star = None

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
                    count=collect_stats,
                )
            )
            topk_mask_cert_i32 = None

        # Skip = NOT top-K*; false for trailing partial block (force-attended).
        if cache.has_trailing_partial_block:
            adaptive_topk_mask = _workspace_tensor(
                cache,
                "adaptive_topk_mask_active_bool",
                (num_q_heads, num_active_blocks),
                dtype=torch.bool,
                device=q_all.device,
            )
            adaptive_topk_mask[:, :n_qblocks].copy_(topk_mask_cert)
            adaptive_topk_mask[:, n_qblocks:num_active_blocks].fill_(True)
            if topk_mask_cert_i32 is not None:
                adaptive_topk_mask_i32 = _workspace_tensor(
                    cache,
                    "adaptive_topk_mask_active_i32",
                    (num_q_heads, num_active_blocks),
                    dtype=torch.int32,
                    device=q_all.device,
                    fill_value=0,
                )
                adaptive_topk_mask_i32[:, :n_qblocks].copy_(topk_mask_cert_i32)
                adaptive_topk_mask_i32[:, n_qblocks:num_active_blocks].fill_(1)
            skip_mask = _workspace_tensor(
                cache,
                "adaptive_skip_mask_active_bool",
                (num_q_heads, num_active_blocks),
                dtype=torch.bool,
                device=q_all.device,
            )
            torch.logical_not(
                adaptive_topk_mask,
                out=skip_mask,
            )
        else:
            adaptive_topk_mask = topk_mask_cert
            adaptive_topk_mask_i32 = topk_mask_cert_i32
            skip_mask = _workspace_tensor(
                cache,
                "adaptive_skip_mask_cert_bool",
                (num_q_heads, n_qblocks),
                dtype=torch.bool,
                device=q_all.device,
            )
            torch.logical_not(topk_mask_cert, out=skip_mask)

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
    fp16_block_log_masses: torch.Tensor | None = None
    top_block_indices: torch.Tensor | None = None
    ranking_k = 0
    score_consistency_violation_heads = 0
    score_consistency_violation_mask: torch.Tensor | None = None
    delta_bound_mean = 0.0
    # Per-head Δ bound (paper Eq. 4). Shared across boundary check (§6.1),
    # score-consistency check (§6), and E_key telemetry (§4.5). Computed
    # once per step in whichever consumer fires first.
    delta_per_head: torch.Tensor | None = None
    # Eq. 30 boundary verification (paper §6.1): for each tail block b not in
    # the promoted top-K set, check ℓ_b^int8 + Δ > ℓ^fp16_(r). When any
    # tail block fails, the ranking certificate cannot be issued for that
    # head — escalate to Rung-3 alongside ranking-disagreement.
    boundary_triggered_heads = 0
    boundary_triggered_mask: torch.Tensor | None = None
    # The FP16 block re-scoring is needed for either Rung-3 ranking check or
    # the score-consistency check. Compute it once when either is enabled.
    need_fp16_scores = (ranking_fallback or score_consistency_check) and n_qblocks > 0
    if need_fp16_scores:
        with _PhaseTimer(phase_timings, "ranking_check"):
            ranking_k = min(max(ranking_r, top_k_fp16_keys, 4), n_qblocks)
            int8_scores = m_b[:, :n_qblocks]
            top_block_indices = int8_scores.topk(
                ranking_k, dim=1, sorted=False,
            ).indices  # [H, K]
            top_int8_scores = int8_scores.gather(1, top_block_indices)       # [H, K]
            if cache.fp16_key_cache_capacity is not None:
                # Bounded-cache mode uses one compact FP16-key working set for
                # both the certification rescore and the subsequent Phase-2
                # top-K attention. This avoids paging/scattering the same
                # blocks twice while keeping the paper path bounded.
                if _identity_resident_blocks(cache, cache.active_blocks, kind="key"):
                    prefetched_keys_fp16_gpu = cache.keys_fp16_gpu
                    prefetched_key_block_slots = _workspace_arange(
                        cache,
                        "identity_key_block_slots",
                        cache.active_blocks,
                        dtype=torch.int32,
                        device=q_all.device,
                    )
                else:
                    ranking_needed = [int(x) for x in top_block_indices.unique().tolist()]
                    phase2_needed: list[int] = []
                    if adaptive_topk_mask is not None:
                        phase2_needed = _priority_order_selected_blocks(
                            adaptive_selected_idx,
                            adaptive_selected_k_star,
                            m_b,
                            n_qblocks,
                        )
                        if not phase2_needed:
                            top_union = adaptive_topk_mask[:, :n_qblocks].any(dim=0)
                            phase2_needed = _priority_order_blocks(
                                top_union.nonzero().flatten(), m_b, n_qblocks,
                            )
                        if cache.has_trailing_partial_block:
                            phase2_needed.append(int(cache.trailing_block_idx))
                    combined_needed: list[int] = []
                    seen_needed: set[int] = set()
                    for bid in phase2_needed + ranking_needed:
                        ib = int(bid)
                        if ib not in seen_needed:
                            seen_needed.add(ib)
                            combined_needed.append(ib)
                    (
                        prefetched_keys_fp16_gpu,
                        prefetched_key_block_slots,
                        _rh,
                        _rm,
                        _rb,
                        _re,
                        _needed,
                    ) = _prepare_bounded_fp16_key_scratch(
                        cache,
                        combined_needed,
                        cache.active_blocks,
                        q_all.device,
                        phase_timings,
                    )
                    h2d_key_bytes += _rb
                    h2d_key_blocks += _rm
                    fp16_cache_hits_step += _rh
                    fp16_cache_misses_step += _rm
                    fp16_cache_evictions_step += _re
                    fp16_cache_needed_blocks += _needed
            # Compute both max-logit (for score-consistency / ranking) and
            # log-mass (for Eq. 30 boundary check) — one fused FP16 rescore.
            fp16_block_scores, fp16_block_log_masses = compute_fp16_block_scores(
                cache, q_all, top_block_indices, n_qblocks, gqa_group, q_scale,
                return_log_mass=True,
                keys_fp16_override=prefetched_keys_fp16_gpu,
                key_block_slots=prefetched_key_block_slots,
            )
        if ranking_fallback:
            # Single pair of argsorts covers r=1, r=3, and r=ranking_r — no
            # need to call detect_ranking_disagreement three times (each call
            # was redoing the same sort).
            k_for_rank = top_int8_scores.shape[1]
            if k_for_rank > 0 and ranking_r > 0:
                if int(ranking_r) == 1 and not collect_stats:
                    ranking_disagree_mask = (
                        top_int8_scores.argmax(dim=1) != fp16_block_scores.argmax(dim=1)
                    )
                else:
                    rank_int8 = top_int8_scores.argsort(dim=1, descending=True)
                    rank_fp16 = fp16_block_scores.argsort(dim=1, descending=True)
                    rank_diff = rank_int8 != rank_fp16  # [H, K]
                    r_main = min(int(ranking_r), k_for_rank)
                    ranking_disagree_mask = rank_diff[:, :r_main].any(dim=1)
                    if collect_stats:
                        r1 = min(1, k_for_rank)
                        r3 = min(3, k_for_rank)
                        ranking_disagree_r1_heads = int(rank_diff[:, :r1].any(dim=1).sum().item())
                        ranking_disagree_r3_heads = int(rank_diff[:, :r3].any(dim=1).sum().item())
            else:
                ranking_disagree_mask = torch.zeros(
                    num_q_heads, dtype=torch.bool, device=q_all.device,
                )

            # Eq. 30 boundary verification (paper §6.1, eq:boundary_check).
            # For each head, flag if any tail block's INT8 log-mass + Δ
            # exceeds the r-th highest FP16 log-mass among promoted blocks.
            # The check closes the residual blind spot of the ranking-only
            # check: a tail block could outrank promoted blocks under FP16
            # without us seeing it. Triggers should be 0 in practice
            # (paper §8.6) when the adaptive selector's τ_cov coverage
            # margin is healthy. The check is a hard merge gate — not a
            # silent telemetry-only assertion.
            if k_for_rank > 0 and n_qblocks > 0:
                # Δ per head from the same per-channel scale used in scoring.
                # Shared with score-consistency and E_key — compute once.
                if delta_per_head is None:
                    delta_per_head = compute_delta_bound(
                        q_all, cache.keys_scale[:, :n_qblocks, :], gqa_group, q_scale,
                    )                                                   # [H]
                _delta_for_boundary = delta_per_head
                # ℓ_b^int8 for ALL fully-quantized blocks: m_b + log(S_b).
                ell_int8_all = (
                    m_b[:, :n_qblocks].float()
                    + torch.log(S_b[:, :n_qblocks].float().clamp(min=1e-30))
                )                                                       # [H, n_qblocks]
                # Per head: r-th highest FP16 log-mass in the promoted set.
                k_for_boundary = min(int(ranking_r), k_for_rank)
                ell_fp16_top = fp16_block_log_masses.topk(
                    k_for_boundary, dim=1,
                ).values                                                # [H, r]
                ell_fp16_r_per_head = ell_fp16_top[:, -1]               # [H]
                # LHS: ℓ_b^int8 + Δ; RHS: ℓ^fp16_(r). Only consider tail
                # blocks by masking the promoted set in-place on a scratch
                # tensor instead of materialising promoted/tail/trigger masks.
                ub = ell_int8_all + _delta_for_boundary.unsqueeze(1)    # [H, n_qblocks]
                ub_tail = ub.clone()
                ub_tail.scatter_(1, top_block_indices.long(), float("-inf"))
                boundary_triggered_mask = (
                    ub_tail.amax(dim=1) > ell_fp16_r_per_head
                )                                                       # [H]
                if collect_stats:
                    boundary_triggered_heads = int(
                        boundary_triggered_mask.sum().item()
                    )
                # Merge with ranking disagreement so existing Rung-3
                # escalation handles boundary triggers identically.
                ranking_disagree_mask = (
                    ranking_disagree_mask | boundary_triggered_mask
                )
        if score_consistency_check:
            # Paper §6 instability-detection: |FP16 - INT8| per block bounded
            # by Δ + eps_guard. Any violation means Theorem 2 was empirically
            # broken on this step — a canary for stale metadata / cache
            # corruption, expected 0-count on well-behaved runs.
            if delta_per_head is None:
                delta_per_head = compute_delta_bound(
                    q_all, cache.keys_scale[:, :n_qblocks, :], gqa_group, q_scale,
                )
            if collect_stats:
                delta_bound_mean = float(delta_per_head.mean().item())
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
    # compromised step and subsumes the per-head Rung-3 action.
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
            elif cache.values_fp16_gpu is not None:
                vram_fp16_value_cache_bytes = (
                    cache.values_fp16_gpu.numel()
                    * cache.values_fp16_gpu.element_size()
                )
            stats = {
                "total_blocks": total_blocks,
                "skipped_blocks": 0,
                "skip_rate": 0.0,
                "attended_blocks": total_blocks,
                "v_format": "fp16",
                "score_consistency_violation_heads": score_consistency_violation_heads,
                "delta_bound_mean": float(delta_bound_mean),
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
                # Eq. 30 boundary: even on Rung-4, expose the boundary check
                # detection for telemetry continuity (paper §8.6 expects 0).
                "boundary_check_triggered_heads": int(boundary_triggered_heads),
                "boundary_check_fired": bool(boundary_triggered_heads > 0),
            }
        else:
            stats = {}
        return output, stats

    # (Removed: legacy entropy-gating / concentration_threshold heuristic.
    # With block_epsilon=0.0 the skip_mask is already all-False, so every
    # block contributes to attention via top-K* + tail E_key. The paper
    # §7 knob set has no concentration_threshold — do not re-introduce.)

    # Phase 2: Attend using SDPA for exact precision matching with dense path.
    # The Triton kernels compute in F32 which diverges from the BF16 SDPA used
    # in dense mode.  Using SDPA here ensures identical numerical behaviour.
    v_format = "fp16"
    e_val_head_before_rung2: torch.Tensor | None = None
    value_fallback_block_count = 0
    value_fallback_head_block_count = 0

    # Paper §3.3 hybrid attend — Algorithm 1 Phase 2. When adaptive K* is
    # active and the cache has FP16 values in VRAM, route to the mask-gated
    # INT8/FP16 kernel so **every** block contributes to the output (top-K*
    # with FP16 keys, the rest with INT8 keys; no blocks are dropped). The
    # prior default SDPA-with-skip path drops tail blocks to mask=-inf,
    # which is a different algorithm (block skipping) and breaks the
    # paper's error bound.
    use_paper_hybrid_int4 = (
        adaptive_topk_mask is not None
        and cache.values_int4_packed is not None
    )
    use_paper_hybrid = (
        adaptive_topk_mask is not None
        and cache.values_fp16 is not None
    )
    if use_paper_hybrid_int4:
        e_val_head: torch.Tensor | None = None
        e_val_head_before_rung2: torch.Tensor | None = None
        value_unsafe_mask: torch.Tensor | None = None
        value_fallback_block_count = 0
        value_fallback_head_block_count = 0
        if value_error_mode not in ("loose", "tight"):
            raise ValueError(
                f"value_error_mode must be 'loose' or 'tight', got {value_error_mode!r}"
            )

        n_active_blocks_hybrid = n_qblocks
        keys_scale_active = cache.keys_scale_active()
        last_block_valid = bs
        if cache.has_trailing_partial_block:
            n_active_blocks_hybrid = n_qblocks + 1
            cache.keys_scale[:, cache.trailing_block_idx, :].zero_()
            cache.keys_zero_points[:, cache.trailing_block_idx, :].zero_()
            keys_scale_active = cache.keys_scale[:, :n_active_blocks_hybrid, :]
            last_block_valid = cache.num_tokens - n_qblocks * bs
            assert 1 <= last_block_valid < bs, last_block_valid
        # The trailing partial block is not represented in
        # values_int4_errors[:, :n_qblocks], so it is outside the Rung-2
        # E_val certificate.  Serve it from the exact FP16 fallback whenever
        # it exists; otherwise bounded INT4 runs can silently use uncertified
        # INT4 values for the newest tokens, which dominate PG-19 decode.
        force_trailing_value_fp16 = (
            cache.has_trailing_partial_block
            and cache.values_fp16_gpu is not None
            and (cache.values_fp16_cpu is not None or cache.fp16_value_cache_capacity is None)
        )

        topk_for_attend = (
            adaptive_topk_mask_i32
            if adaptive_topk_mask_i32 is not None
            else adaptive_topk_mask
        )
        hybrid_topk = topk_for_attend[:, :n_active_blocks_hybrid].contiguous()
        no_skip = _workspace_tensor(
            cache,
            "no_skip_i32",
            (num_q_heads, n_active_blocks_hybrid),
            dtype=torch.int32,
            device=q_all.device,
            fill_value=0,
        )
        nt_hybrid = n_active_blocks_hybrid * bs

        keys_fp16_gpu = cache.keys_fp16_gpu
        key_block_slots = _workspace_arange(
            cache,
            "identity_key_block_slots",
            n_active_blocks_hybrid,
            dtype=torch.int32,
            device=q_all.device,
        )
        if prefetched_keys_fp16_gpu is not None and prefetched_key_block_slots is not None:
            keys_fp16_gpu = prefetched_keys_fp16_gpu
            key_block_slots = prefetched_key_block_slots[:n_active_blocks_hybrid]
        elif cache.fp16_key_cache_capacity is not None:
            if _identity_resident_blocks(cache, n_active_blocks_hybrid, kind="key"):
                keys_fp16_gpu = cache.keys_fp16_gpu
                key_block_slots = _workspace_arange(
                    cache,
                    "identity_key_block_slots",
                    n_active_blocks_hybrid,
                    dtype=torch.int32,
                    device=q_all.device,
                )
            else:
                needed_blocks = _priority_order_selected_blocks(
                    adaptive_selected_idx,
                    adaptive_selected_k_star,
                    m_b,
                    n_qblocks,
                )
                if not needed_blocks:
                    top_union = adaptive_topk_mask[:, :n_qblocks].any(dim=0)
                    needed_blocks = _priority_order_blocks(
                        top_union.nonzero().flatten(), m_b, n_qblocks,
                    )
                if cache.has_trailing_partial_block:
                    needed_blocks.append(int(cache.trailing_block_idx))
                (
                    keys_fp16_gpu,
                    key_block_slots,
                    c_hits,
                    c_misses,
                    c_bytes,
                    c_evict,
                    c_needed,
                ) = _prepare_bounded_fp16_key_scratch(
                    cache,
                    needed_blocks,
                    n_active_blocks_hybrid,
                    q_all.device,
                    phase_timings,
                )
                h2d_key_bytes += c_bytes
                h2d_key_blocks += c_misses
                fp16_cache_hits_step += c_hits
                fp16_cache_misses_step += c_misses
                fp16_cache_evictions_step += c_evict
                fp16_cache_needed_blocks += c_needed
        elif keys_fp16_gpu is None:
            with _PhaseTimer(phase_timings, "h2d_pagein"):
                keys_fp16_gpu = cache.keys_fp16_cpu.to(device=q_all.device, non_blocking=True)
            kv_k, _, hd_k = cache.keys_fp16_cpu.shape
            h2d_key_bytes += kv_k * nt_hybrid * hd_k * cache.keys_fp16_cpu.element_size()
            h2d_key_blocks += n_active_blocks_hybrid

        with _PhaseTimer(phase_timings, "value_check"):
            if mass_frac_cert is not None and mass_frac_cert.shape[1] == n_qblocks:
                mass_frac = mass_frac_cert
            else:
                mass_frac = compute_block_mass_fraction(
                    m_b[:, :n_qblocks], S_b[:, :n_qblocks],
                )
            # η_b is stored as relative value reconstruction error, matching
            # the paper §7 v_tol=0.05 operating point.
            eta_blocks = cache.values_int4_errors[:, :n_qblocks]
            kv_idx = torch.arange(num_q_heads, device=mass_frac.device) // gqa_group
            kv_idx = kv_idx.clamp(max=eta_blocks.shape[0] - 1)
            eta_per_qhead = eta_blocks[kv_idx]
            per_block_e_val = mass_frac * eta_per_qhead
            e_pre = per_block_e_val.sum(dim=1)
            if value_error_mode == "tight":
                # Promote the largest value-error contributors until the
                # achieved per-head E_val is within the paper budget. A
                # simple per-block threshold can fire while leaving the
                # reported total bound above v_tol; the certificate needs the
                # post-promotion total, not just individual blocks, under
                # control.
                conditional_value_sort = (
                    _os.environ.get("DOTCACHE_CONDITIONAL_VALUE_SORT", "0").strip().lower()
                    not in {"0", "false", "no", "off"}
                )
                has_value_excess = (
                    bool((e_pre > v_tolerance).any().item())
                    if conditional_value_sort
                    else True
                )
                if has_value_excess:
                    excess = (e_pre - v_tolerance).clamp(min=0.0)
                    value_promotion_topk = int(
                        _os.environ.get("DOTCACHE_VALUE_PROMOTION_TOPK", "0")
                    )
                    if 0 < value_promotion_topk < per_block_e_val.shape[1]:
                        sorted_vals, sorted_idx = torch.topk(
                            per_block_e_val,
                            value_promotion_topk,
                            dim=1,
                            largest=True,
                            sorted=True,
                        )
                    else:
                        sorted_vals, sorted_idx = torch.sort(
                            per_block_e_val, dim=1, descending=True,
                        )
                    cumsum_before = torch.cumsum(sorted_vals, dim=1) - sorted_vals
                    promote_sorted = (excess[:, None] > 0.0) & (cumsum_before < excess[:, None])
                    promoted_e_val = (sorted_vals * promote_sorted.to(sorted_vals.dtype)).sum(dim=1)
                    value_unsafe_mask = _workspace_tensor(
                        cache,
                        "value_unsafe_mask_bool",
                        per_block_e_val.shape,
                        dtype=torch.bool,
                        device=promote_sorted.device,
                        fill_value=False,
                    )
                    value_unsafe_mask.scatter_(1, sorted_idx, promote_sorted)
                    if sorted_vals.shape[1] < per_block_e_val.shape[1]:
                        # Conservative bounded promotion: if the capped top-k
                        # contributors cannot cover the required excess for a
                        # head, promote all values for that head. This keeps
                        # the paper certificate exact (E_val becomes zero)
                        # without paying a full sort on every normal step.
                        capped_sum = sorted_vals.sum(dim=1)
                        full_value_fallback_heads = excess > capped_sum
                        value_unsafe_mask |= full_value_fallback_heads.unsqueeze(1)
                        promoted_e_val = torch.where(
                            full_value_fallback_heads,
                            e_pre,
                            promoted_e_val,
                        )
                else:
                    promoted_e_val = torch.zeros_like(e_pre)
                    value_unsafe_mask = _workspace_tensor(
                        cache,
                        "value_unsafe_mask_bool",
                        per_block_e_val.shape,
                        dtype=torch.bool,
                        device=per_block_e_val.device,
                        fill_value=False,
                    )
            else:
                value_unsafe_mask = eta_per_qhead > v_tolerance
                promoted_e_val = (per_block_e_val * value_unsafe_mask.to(per_block_e_val.dtype)).sum(dim=1)
            value_fallback_any = (
                True
                if force_trailing_value_fp16
                else bool(value_unsafe_mask.any().item())
            )
            int4_value_mask = ~value_unsafe_mask
            e_val_head_before_rung2 = e_pre
            e_val_head = (e_pre - promoted_e_val).clamp(min=0.0)
            if collect_stats:
                rho = mass_frac.sum(dim=1)
                eta_int4 = cache.values_int4_errors.max().item()
                value_fallback_head_block_count = int(value_unsafe_mask.sum().item())
                value_fallback_block_count = int(value_unsafe_mask.any(dim=0).sum().item())
                v_format = "mixed" if (value_fallback_head_block_count or force_trailing_value_fp16) else "int4"
            elif (
                cache.values_fp16_gpu is not None
                and cache.fp16_value_cache_capacity is None
            ):
                # Explicit legacy full-mirror path. Kept for debug/sweep
                # comparisons only; paper quality runs use bounded scratch so
                # the VRAM savings remain meaningful.
                rho = None
                eta_int4 = 0.0
                v_format = "mixed"
            else:
                # Bounded/CPU fallback needs a compact block list only if
                # Rung 2 actually fires. With stats disabled, use a single
                # any() sync to stay on the pure INT4 path for the common case
                # instead of also syncing fallback counts.
                rho = None
                eta_int4 = 0.0
                v_format = "mixed" if value_fallback_any else "int4"

        if v_format == "int4":
            with _PhaseTimer(phase_timings, "phase2_fused_attend"):
                output = selective_attend_multihead_hybrid_int4v(
                    keys_int8=cache.keys_int8[:, :nt_hybrid, :],
                    keys_scale=keys_scale_active,
                    keys_zero_points=cache.keys_zero_points[:, :n_active_blocks_hybrid, :],
                    keys_fp16=keys_fp16_gpu[:, :nt_hybrid, :],
                    key_block_slots=key_block_slots,
                    topk_mask=hybrid_topk,
                    values_int4_packed=cache.values_int4_packed[:, :nt_hybrid, :],
                    values_int4_scales=cache.values_int4_scales[:, :nt_hybrid, :],
                    values_int4_zeros=cache.values_int4_zeros[:, :nt_hybrid, :],
                    q_all=q_all,
                    skip_mask_i32=no_skip,
                    gqa_group=gqa_group,
                    block_size=bs,
                    group_size=cache.values_int4_group_size,
                    q_scale=q_scale,
                    last_block_valid=last_block_valid,
                )
        else:
            rung2_fired = True
            assert value_unsafe_mask is not None
            if (
                cache.values_fp16_gpu is None
                and cache.values_fp16_cpu is None
                and cache.values_fp16 is None
            ):
                raise ValueError("INT4 unsafe and no FP16 fallback available")
            value_fp16_mask = _workspace_tensor(
                cache,
                "value_fp16_mask_i32",
                (num_q_heads, n_active_blocks_hybrid),
                dtype=torch.int32,
                device=q_all.device,
                fill_value=0,
            )
            value_fp16_mask[:, :n_qblocks].copy_(value_unsafe_mask)
            if force_trailing_value_fp16:
                value_fp16_mask[:, n_qblocks:n_active_blocks_hybrid] = 1
            use_one_step_value_pagein = False
            dynamic_one_step_value_scratch = False
            if (
                cache.values_fp16_gpu is not None
                and cache.fp16_value_cache_capacity is None
            ):
                value_block_slots = _workspace_arange(
                    cache,
                    "identity_value_block_slots",
                    n_active_blocks_hybrid,
                    dtype=torch.int32,
                    device=q_all.device,
                )
                values_fp16_scratch = cache.values_fp16_gpu[:, :nt_hybrid, :]
            else:
                value_block_slots = _workspace_tensor(
                    cache,
                    "value_block_slots_i32",
                    (n_active_blocks_hybrid,),
                    dtype=torch.int32,
                    device=q_all.device,
                    fill_value=-1,
                )
                if _identity_resident_blocks(cache, n_active_blocks_hybrid, kind="value"):
                    value_block_slots = _workspace_arange(
                        cache,
                        "identity_value_block_slots",
                        n_active_blocks_hybrid,
                        dtype=torch.int32,
                        device=q_all.device,
                    )
                    values_fp16_scratch = cache.values_fp16_gpu
                else:
                    unsafe_block_mask = value_unsafe_mask.any(dim=0)
                    n_value_slots = int(unsafe_block_mask.sum().item())
                    if force_trailing_value_fp16:
                        n_value_slots += 1
                    gpu_mask_pagein = (
                        _os.environ.get("DOTCACHE_VALUE_GPU_MASK_PAGEIN", "1").strip().lower()
                        not in {"0", "false", "no", "off"}
                        and cache.values_fp16_cpu is not None
                        and cache.values_fp16_gpu is not None
                        and cache.fp16_value_cache_capacity is not None
                        and int(cache.fp16_value_cache_capacity) > 0
                        and n_value_slots <= int(cache.fp16_value_cache_capacity)
                    )
                    if gpu_mask_pagein:
                        # One-step bounded scratch page-in driven by a GPU
                        # block->slot table. This avoids materialising the
                        # unsafe block list on CPU while preserving the paper
                        # exact mixed INT4/FP16 value semantics.
                        slots = torch.cumsum(
                            unsafe_block_mask.to(torch.int32), dim=0,
                        ) - 1
                        value_block_slots[:n_qblocks].copy_(
                            torch.where(
                                unsafe_block_mask,
                                slots,
                                torch.full_like(slots, -1),
                            )
                        )
                        if force_trailing_value_fp16:
                            value_block_slots[n_qblocks] = n_value_slots - 1
                        values_fp16_scratch = cache.values_fp16_gpu
                        cache._fp16_value_resident.clear()
                        cache._fp16_value_free_slots = list(range(int(cache.fp16_value_cache_capacity)))
                        with _PhaseTimer(phase_timings, "h2d_pagein"):
                            from dotcache.backends.cuda_pagein import page_in_fp16_blocks_by_slots_cuda

                            page_in_fp16_blocks_by_slots_cuda(
                                src_cpu=cache.values_fp16_cpu,
                                dst_gpu=values_fp16_scratch,
                                block_slots_gpu=value_block_slots,
                                block_size=bs,
                                active_tokens=cache.num_tokens,
                                n_blocks=n_active_blocks_hybrid,
                            )
                        h2d_value_bytes += (
                            cache.kv_heads
                            * n_value_slots
                            * bs
                            * cache.d_v
                            * cache.values_fp16_cpu.element_size()
                        )
                        h2d_value_blocks += n_value_slots
                        fp16_value_cache_needed_blocks += n_value_slots
                    else:
                        unsafe_block_ids = unsafe_block_mask.nonzero().flatten()
                        unsafe_block_list = [int(b) for b in unsafe_block_ids.tolist()]
                        if force_trailing_value_fp16:
                            unsafe_block_list.append(int(n_qblocks))
                            unsafe_block_ids = torch.cat([
                                unsafe_block_ids,
                                torch.tensor([int(n_qblocks)], dtype=unsafe_block_ids.dtype, device=unsafe_block_ids.device),
                            ])
                    large_value_pagein = (
                        n_value_slots >= int(_os.environ.get("DOTCACHE_VALUE_ONESTEP_PAGEIN_MIN_BLOCKS", "256"))
                    )
                    if (not gpu_mask_pagein and
                        cache.values_fp16_gpu is not None
                        and cache.fp16_value_cache_capacity is not None
                        and n_value_slots <= int(cache.fp16_value_cache_capacity)
                        and int(cache.fp16_value_cache_capacity) > 0
                        and not large_value_pagein
                    ):
                        with _PhaseTimer(phase_timings, "h2d_pagein"):
                            block_to_slot, v_hits, v_misses, v_bytes, v_evict = (
                                cache.ensure_fp16_values_resident_batched(unsafe_block_list)
                            )
                        if block_to_slot is None:
                            fp16_value_cache_overflow_step = 1
                            use_one_step_value_pagein = True
                        else:
                            values_fp16_scratch = cache.values_fp16_gpu
                            value_block_slots = cache.fp16_value_block_slots_gpu(n_active_blocks_hybrid)
                            h2d_value_bytes += v_bytes
                            h2d_value_blocks += v_misses
                            fp16_value_cache_hits_step += v_hits
                            fp16_value_cache_misses_step += v_misses
                            fp16_value_cache_evictions_step += v_evict
                            fp16_value_cache_needed_blocks += len(unsafe_block_list)
                    elif not gpu_mask_pagein:
                        fp16_value_cache_overflow_step = int(
                            cache.fp16_value_cache_capacity is not None
                            and n_value_slots > int(cache.fp16_value_cache_capacity)
                        )
                        use_one_step_value_pagein = True

                    if (not gpu_mask_pagein) and use_one_step_value_pagein:
                        dynamic_one_step_value_scratch = True
                        mixedv_splitk_fallback_step = 1
                        value_block_slots[unsafe_block_ids] = torch.arange(
                            n_value_slots,
                            dtype=torch.int32,
                            device=q_all.device,
                        )
                        value_dtype = (
                            cache.values_fp16_cpu.dtype
                            if cache.values_fp16_cpu is not None
                            else cache.values_fp16.dtype
                        )
                        values_fp16_scratch = torch.empty(
                            cache.kv_heads,
                            max(n_value_slots, 1) * bs,
                            cache.d_v,
                            dtype=value_dtype,
                            device=q_all.device,
                        )
                        with _PhaseTimer(phase_timings, "h2d_pagein"):
                            if n_value_slots > 0:
                                if cache.values_fp16_cpu is not None:
                                    h2d_value_bytes += cache._page_in_fp16_blocks(
                                        src_cpu=cache.values_fp16_cpu,
                                        dst_gpu=values_fp16_scratch,
                                        loaded_blocks=unsafe_block_list,
                                        loaded_slots=list(range(n_value_slots)),
                                        evicted_blocks=[],
                                        slot_table=value_block_slots,
                                        feature_dim=cache.d_v,
                                    )
                                else:
                                    src_blocks = cache.values_fp16[:, :n_qblocks * bs, :].reshape(
                                        cache.kv_heads, n_qblocks, bs, cache.d_v,
                                    ).index_select(1, unsafe_block_ids.to(dtype=torch.long))
                                    values_fp16_scratch[:, :n_value_slots * bs, :] = src_blocks.reshape(
                                        cache.kv_heads, n_value_slots * bs, cache.d_v,
                                    )
                                h2d_value_blocks += n_value_slots
            _split_threshold = int(_os.environ.get("DOTCACHE_MIXEDV_SPLITK_MIN_BLOCKS", "512"))
            _use_split_mixed = _os.environ.get("DOTCACHE_MIXEDV_SPLITK", "1") != "0" and n_active_blocks_hybrid >= _split_threshold
            _backend = _os.environ.get("DOTCACHE_CERTIFIED_BACKEND", "triton").strip().lower()
            mixed_attend = selective_attend_multihead_hybrid_mixedv
            mixed_attend_is_native = False
            if _use_split_mixed:
                mixed_attend = selective_attend_multihead_hybrid_mixedv_split_k
            if _backend == "cutlass_sm120" and _use_split_mixed:
                try:
                    from dotcache.backends.cutlass_sm120 import (
                        cutlass_sm120_available,
                        hybrid_mixedv_split_k_cutlass,
                    )
                    # The CUTLASS backend is wired behind an explicit kernel
                    # enable until its tensor-core implementation passes the
                    # correctness/performance gates. Availability still probes
                    # the vendored CUTLASS toolchain for CI/perf-gate scripts.
                    if (
                        _os.environ.get("DOTCACHE_CUTLASS_SM120_ENABLE_KERNELS", "0") == "1"
                        and cutlass_sm120_available()
                    ):
                        mixed_attend = hybrid_mixedv_split_k_cutlass
                except Exception:
                    mixed_attend = selective_attend_multihead_hybrid_mixedv_split_k
            if (
                _backend == "native_blackwell"
                and _use_split_mixed
            ):
                try:
                    from dotcache.backends.certified_blackwell import (
                        certified_blackwell_available,
                        hybrid_mixedv_split_k_cuda,
                    )
                    # The overflow path builds a compact per-step FP16 value
                    # scratch. Keep that on the reference split-K backend until
                    # the native kernel's compact value-slot path is proven at
                    # paper scale; cache capacity must not affect quality.
                    if certified_blackwell_available() and not dynamic_one_step_value_scratch:
                        mixed_attend = hybrid_mixedv_split_k_cuda
                        mixed_attend_is_native = True
                except Exception:
                    mixed_attend = selective_attend_multihead_hybrid_mixedv_split_k
                    mixed_attend_is_native = False
            with _PhaseTimer(phase_timings, "phase2_fused_attend"):
                mixed_kwargs = dict(
                    keys_int8=cache.keys_int8[:, :nt_hybrid, :],
                    keys_scale=keys_scale_active,
                    keys_zero_points=cache.keys_zero_points[:, :n_active_blocks_hybrid, :],
                    keys_fp16=keys_fp16_gpu[:, :nt_hybrid, :],
                    key_block_slots=key_block_slots,
                    topk_mask=hybrid_topk,
                    values_int4_packed=cache.values_int4_packed[:, :nt_hybrid, :],
                    values_int4_scales=cache.values_int4_scales[:, :nt_hybrid, :],
                    values_int4_zeros=cache.values_int4_zeros[:, :nt_hybrid, :],
                    values_fp16_scratch=values_fp16_scratch,
                    value_fp16_mask=value_fp16_mask,
                    value_block_slots=value_block_slots,
                    q_all=q_all,
                    gqa_group=gqa_group,
                    block_size=bs,
                    group_size=cache.values_int4_group_size,
                    q_scale=q_scale,
                    last_block_valid=last_block_valid,
                    workspace=_cert_workspace(cache),
                )
                if not mixed_attend_is_native:
                    mixed_kwargs["skip_mask_i32"] = no_skip
                if mixed_attend_is_native or mixed_attend is selective_attend_multihead_hybrid_mixedv_split_k:
                    mixed_kwargs["int8_token_scores"] = int8_token_scores
                output = mixed_attend(**mixed_kwargs)
    elif use_paper_hybrid:
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
            # Same for asymmetric zero points — kernel reads (q - z)*s; if z is
            # uninit and s is zero, the product is zero regardless. Belt-and-
            # braces zeroing of z keeps the dequant numerically clean.
            cache.keys_zero_points[:, cache.trailing_block_idx, :].zero_()
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
        key_block_slots = _workspace_arange(
            cache,
            "identity_key_block_slots",
            n_active_blocks_hybrid,
            dtype=torch.int32,
            device=q_all.device,
        )
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
            needed_blocks = _priority_order_selected_blocks(
                adaptive_selected_idx,
                adaptive_selected_k_star,
                m_b,
                n_qblocks,
            )
            if not needed_blocks:
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
            key_block_slots = cache.fp16_key_block_slots_gpu(n_active_blocks_hybrid)
        # DOTCACHE_FAST_ATTEND=0 reverts to the single-program-per-head
        # kernel for A/B comparison. Default is split-K (FlashDecoding-style),
        # which is 15-20× faster at 64K context on Blackwell (grid expands
        # from num_q_heads to num_q_heads × num_splits, filling the SMs).
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
                    keys_zero_points=cache.keys_zero_points[:, :n_active_blocks_hybrid, :],
                    keys_fp16=keys_fp16_gpu[:, :nt_hybrid, :],
                    key_block_slots=key_block_slots,
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
                    keys_zero_points=cache.keys_zero_points[:, :n_active_blocks_hybrid, :].contiguous(),
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
                # Slice INT4 buffers to the same active-tokens window as
                # keys_int8_active() so the kernel's reshape sees consistent
                # dims after decode-time append_token has grown the cache.
                _nt_active = cache.aligned_tokens
                output = selective_attend_multihead_int8k_int4v(
                    keys_int8=cache.keys_int8_active(),
                    keys_scale=cache.keys_scale_active(),
                    keys_zero_points=cache.keys_zero_points_active(),
                    values_int4_packed=cache.values_int4_packed[:, :_nt_active, :],
                    values_int4_scales=cache.values_int4_scales[:, :_nt_active, :],
                    values_int4_zeros=cache.values_int4_zeros[:, :_nt_active, :],
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
            _nt_active = cache.aligned_tokens
            if cache.values_fp16_cpu is not None:
                with _PhaseTimer(phase_timings, "h2d_pagein"):
                    # Slice to active tokens — the CPU mirror is sized to
                    # capacity (= prefill + max_new_tokens) so the kernel
                    # would otherwise see far more tokens than the keys it
                    # has, breaking the reshape.
                    values_fp16 = cache.values_fp16_cpu[:, :_nt_active, :].to(
                        device=cache.keys_int8.device, non_blocking=True,
                    )
                nt_v = cache.num_tokens
                kv_v, _, dv_v = cache.values_fp16_cpu.shape
                h2d_value_bytes += kv_v * nt_v * dv_v * cache.values_fp16_cpu.element_size()
                h2d_value_blocks += (nt_v + bs - 1) // bs
            elif cache.values_fp16 is not None:
                values_fp16 = cache.values_fp16[:, :_nt_active, :]
            else:
                raise ValueError("INT4 unsafe and no FP16 fallback available")
            with _PhaseTimer(phase_timings, "phase2_fused_attend"):
                output = selective_attend_multihead_int8(
                    keys_int8=cache.keys_int8_active(),
                    keys_scale=cache.keys_scale_active(),
                    keys_zero_points=cache.keys_zero_points_active(),
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
            with _PhaseTimer(phase_timings, "rung3_dense_recompute"):
                output, r3_key_bytes, r3_key_blocks, r3_value_bytes, r3_value_blocks = recompute_heads_dense_fp16(
                    cache=cache,
                    q_all=q_all,
                    output=output,
                    head_indices=disagree_heads,
                    gqa_group=gqa_group,
                    q_scale=q_scale,
                )
            h2d_key_bytes += r3_key_bytes
            h2d_key_blocks += r3_key_blocks
            h2d_value_bytes += r3_value_bytes
            h2d_value_blocks += r3_value_blocks
            ranking_fallback_heads = int(disagree_heads.numel())

    # Stats
    if collect_stats:
        total_blocks = num_q_heads * cache.num_blocks
        skipped = skip_mask.sum().item()
        stats = {
            "total_blocks": total_blocks,
            "skipped_blocks": int(skipped),
            "skip_rate": float(skipped) / float(total_blocks),
            "attended_blocks": total_blocks - int(skipped),
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
        if cache.fp16_value_cache_capacity is not None:
            stats["fp16_value_cache_capacity_blocks"] = int(cache.fp16_value_cache_capacity)
            stats["fp16_value_cache_resident_blocks"] = int(len(cache._fp16_value_resident))
            stats["fp16_value_cache_hits_step"] = int(fp16_value_cache_hits_step)
            stats["fp16_value_cache_misses_step"] = int(fp16_value_cache_misses_step)
            stats["fp16_value_cache_evictions_step"] = int(fp16_value_cache_evictions_step)
            stats["fp16_value_cache_needed_blocks_step"] = int(fp16_value_cache_needed_blocks)
            stats["fp16_value_cache_overflow_step"] = int(fp16_value_cache_overflow_step)
            stats["mixedv_splitk_fallback_step"] = int(mixedv_splitk_fallback_step)
            total_value_access = fp16_value_cache_hits_step + fp16_value_cache_misses_step
            stats["fp16_value_cache_hit_rate_step"] = (
                float(fp16_value_cache_hits_step) / total_value_access
                if total_value_access else 0.0
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
        elif cache.values_fp16_gpu is not None:
            vram_fp16_value_cache_bytes = (
                cache.values_fp16_gpu.numel()
                * cache.values_fp16_gpu.element_size()
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
                if e_val_head_before_rung2 is not None:
                    stats["e_val_pre_rung2_max"] = float(e_val_head_before_rung2.max().item())
                    stats["e_val_pre_rung2_mean"] = float(e_val_head_before_rung2.mean().item())
            if value_fallback_head_block_count:
                stats["value_fallback_blocks"] = int(value_fallback_block_count)
                stats["value_fallback_head_blocks"] = int(value_fallback_head_block_count)
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
            # Paper §4.5 E_key contract: E_key = 2·V_max·e^{2Δ}·ᾱ_T·(e^{2Δ}−1).
            # Per-head computation, then mean/max for telemetry. Only assembled
            # when Δ is available (ranking_fallback or score_consistency_check
            # enabled). V_max comes from the cache's per-block ν_b annotation
            # (paper §2.3), tail mass ᾱ_T from the adaptive selector after
            # Rung-1 expansion. The audit's Mismatch 4 ("E_key not computed,
            # V_max not tracked") is closed by this block + the
            # values_norm_max_per_block field on the cache.
            v_max_layer = cache.v_max_global()
            stats["v_max_layer"] = float(v_max_layer)
            if delta_per_head is not None and v_max_layer > 0.0:
                exp_2delta = torch.exp(2.0 * delta_per_head.float())
                e_key_per_head = (
                    2.0 * v_max_layer * exp_2delta
                    * tail_mass_est.float() * (exp_2delta - 1.0)
                )                                                       # [H]
                stats["e_key_step_mean"] = float(e_key_per_head.mean().item())
                stats["e_key_step_max"] = float(e_key_per_head.max().item())
        # Score-consistency violation counters (paper §6). Always emitted
        # when the feature is enabled so runs can confirm the 0-count baseline.
        if score_consistency_check:
            stats["score_consistency_violation_heads"] = int(score_consistency_violation_heads)
            stats["delta_bound_mean"] = float(delta_bound_mean)
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
            # Eq. 30 boundary verification telemetry (paper §6.1, §8.6).
            # The §8.6 "0 boundary triggers" claim is now empirically
            # verifiable, not aspirational.
            stats["boundary_check_triggered_heads"] = int(boundary_triggered_heads)
            stats["boundary_check_fired"] = bool(boundary_triggered_heads > 0)
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
        certified_attention_layer(
            cache, q_all, gqa_group, q_scale,
            v_tolerance=DEFAULT_V_TOLERANCE,
        )
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
            cache, q_all, gqa_group, q_scale,
            v_tolerance=DEFAULT_V_TOLERANCE,
        )
    torch.cuda.synchronize()
    t_cert = (time.perf_counter() - t0) / iters * 1e6

    # Correctness
    output_cert, stats = certified_attention_layer(
        cache, q_all, gqa_group, q_scale,
        v_tolerance=DEFAULT_V_TOLERANCE,
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
        "skip_rate": stats["skip_rate"],
        "cosine_min": cos.min().item(),
        "cosine_mean": cos.mean().item(),
        "vram_mb": cache.vram_bytes() / 1e6,
        "cpu_mb": cache.cpu_bytes() / 1e6,
    }
