"""PG-19 perplexity benchmark: dense vs certified attention.

Measures teacher-forced perplexity on PG-19 test books at various
context lengths. For each document chunk:
  1. Dense: standard HF forward pass → per-token NLL
  2. Certified: dense prefill → tiered KV cache → certified decode
     (teacher-forced, one token at a time) → per-token NLL

Reports: perplexity (dense & certified), ratio, skip rate.
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.paper._provenance import (  # noqa: E402
    PAPER_EPS_GUARD,
    PAPER_EXPLORATION_RATE,
    PAPER_K_MAX,
    PAPER_K_MIN,
    PAPER_RANKING_R,
    PAPER_RUNG1_MULTIPLIER,
    PAPER_RUNG1_THRESHOLD,
    PAPER_TAU_COV,
)


# Natural log of 2 — converts NLL (nats) to bits-per-token.
_LN2 = math.log(2.0)


def per_chunk_bpt_stats(
    per_chunk: list[dict],
    field: str = "nll",
    tokens_field: str = "tokens",
    bootstrap_iters: int = 10_000,
    seed: int = 0,
) -> dict:
    """Per-chunk bits-per-token mean and 95% bootstrap CI.

    Each chunk contributes (nll / tokens) / ln(2). The CI is computed by
    resampling chunks with replacement — treats each chunk as the
    independent unit of randomness, which matches how PG-19 sweeps are
    reported. Also emits Gaussian mean ± 1.96·SE for a quick sanity check.

    Returns a dict with bpt_mean, bpt_std, bpt_se, bpt_ci_lo / bpt_ci_hi
    (bootstrap), bpt_gaussian_lo / bpt_gaussian_hi (Wald), n_chunks.

    `field` lets callers pick "nll" (full chunk: prefix + suffix) or
    "suffix_nll" (certified region only — the reviewer-meaningful slice
    since prefix is always dense).
    """
    if not per_chunk:
        return {"n_chunks": 0}
    bpt = []
    for c in per_chunk:
        toks = c.get(tokens_field)
        nll = c.get(field)
        if toks is None or nll is None or toks <= 0:
            continue
        bpt.append((nll / toks) / _LN2)
    n = len(bpt)
    if n == 0:
        return {"n_chunks": 0}
    arr = np.asarray(bpt, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    # Bootstrap 95% CI (percentile).
    if n > 1 and bootstrap_iters > 0:
        rng = np.random.default_rng(seed)
        boot_means = arr[rng.integers(0, n, size=(bootstrap_iters, n))].mean(axis=1)
        lo = float(np.quantile(boot_means, 0.025))
        hi = float(np.quantile(boot_means, 0.975))
    else:
        lo = hi = mean
    return {
        "n_chunks": n,
        "bpt_mean": mean,
        "bpt_std": std,
        "bpt_se": se,
        "bpt_ci_lo": lo,        # bootstrap percentile
        "bpt_ci_hi": hi,
        "bpt_gaussian_lo": mean - 1.96 * se,
        "bpt_gaussian_hi": mean + 1.96 * se,
    }


def per_chunk_ppl_delta_stats(
    dense_per_chunk: list[dict],
    cert_per_chunk: list[dict],
    *,
    bootstrap_iters: int = 10_000,
    seed: int = 20260425,
) -> dict:
    """Paired per-chunk perplexity deltas for paper table generation.

    Each chunk contributes dense ppl, certified ppl, Δppl, and ratio. The
    reported CI is a bootstrap percentile interval over chunk-level Δppl.
    """
    dense_by_idx = {int(c["chunk_idx"]): c for c in dense_per_chunk if "chunk_idx" in c}
    rows: list[dict] = []
    for cert in cert_per_chunk:
        if "chunk_idx" not in cert:
            continue
        idx = int(cert["chunk_idx"])
        dense = dense_by_idx.get(idx)
        if dense is None:
            continue
        dense_tokens = int(dense.get("tokens") or 0)
        cert_tokens = int(cert.get("tokens") or 0)
        if dense_tokens <= 0 or cert_tokens <= 0:
            continue
        dense_ppl = math.exp(float(dense["nll"]) / dense_tokens)
        cert_ppl = math.exp(float(cert["nll"]) / cert_tokens)
        rows.append({
            "chunk_idx": idx,
            "book_idx": cert.get("book_idx", dense.get("book_idx")),
            "dense_ppl": dense_ppl,
            "certified_ppl": cert_ppl,
            "delta_ppl": cert_ppl - dense_ppl,
            "ratio": cert_ppl / dense_ppl if dense_ppl else None,
            "dense_tokens": dense_tokens,
            "certified_tokens": cert_tokens,
        })

    n = len(rows)
    if n == 0:
        return {"n_chunks": 0, "per_chunk": []}
    deltas = np.asarray([r["delta_ppl"] for r in rows], dtype=np.float64)
    ratios = np.asarray([r["ratio"] for r in rows if r["ratio"] is not None], dtype=np.float64)
    mean_delta = float(deltas.mean())
    if n > 1 and bootstrap_iters > 0:
        rng = np.random.default_rng(seed)
        boot = deltas[rng.integers(0, n, size=(int(bootstrap_iters), n))].mean(axis=1)
        ci_lo = float(np.quantile(boot, 0.025))
        ci_hi = float(np.quantile(boot, 0.975))
    else:
        ci_lo = ci_hi = mean_delta
    return {
        "n_chunks": n,
        "mean_delta_ppl": mean_delta,
        "delta_ppl_ci_lo": ci_lo,
        "delta_ppl_ci_hi": ci_hi,
        "mean_ratio": float(ratios.mean()) if ratios.size else None,
        "bootstrap_iters": int(bootstrap_iters),
        "bootstrap_seed": int(seed),
        "per_chunk": rows,
    }


def load_pg19_chunks(tokenizer, context_length: int, num_chunks: int,
                     stride: int = None, start_chunk: int = 0) -> tuple[list[torch.Tensor], list[int]]:
    """Load PG-19 test set and chunk into fixed-length token sequences.

    Uses strided windowing: each chunk starts `stride` tokens after the
    previous one. Default stride = context_length (no overlap).
    """
    from datasets import load_dataset

    if stride is None:
        stride = context_length

    chunks = []
    book_indices: list[int] = []  # parallel to chunks; used for per-book CI grouping
    seen_chunks = 0
    ds = load_dataset("emozilla/pg19", split="test", streaming=True)

    for book_idx, book in enumerate(ds):
        text = book["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        # Slide window across the book
        for start in range(0, len(tokens) - context_length, stride):
            if seen_chunks < int(start_chunk):
                seen_chunks += 1
                continue
            chunk = tokens[start : start + context_length]
            chunks.append(torch.tensor(chunk, dtype=torch.long))
            book_indices.append(book_idx)
            seen_chunks += 1
            if len(chunks) >= num_chunks:
                return chunks, book_indices

    print(f"Warning: only found {len(chunks)} chunks (requested {num_chunks})")
    return chunks, book_indices


def compute_dense_perplexity(
    model,
    chunks: list[torch.Tensor],
    device: str = "cuda",
    book_indices: list[int] | None = None,
    chunk_offset: int = 0,
) -> dict:
    """Compute perplexity using standard dense forward pass.

    Retains per-chunk NLL and token counts so the reporting layer can
    compute bootstrap CIs over per-chunk bits-per-token without needing
    to rerun the model. `book_indices` (optional, parallel to `chunks`)
    is pass-through metadata so downstream code can also group by book.
    """
    total_nll = 0.0
    total_tokens = 0
    per_chunk: list[dict] = []

    for i, chunk in enumerate(chunks):
        input_ids = chunk.unsqueeze(0).to(device)
        with torch.inference_mode():
            outputs = model(input_ids=input_ids)
        # Compute loss in chunks to avoid materialising full FP32 logit tensor
        # (8K × 128K × 4 bytes = 4 GB — too large when sharing GPU)
        logits = outputs.logits[:, :-1, :]  # keep native dtype
        targets = input_ids[:, 1:]
        chunk_nll_tensor = torch.zeros((), dtype=torch.float32, device=device)
        chunk_size = 512  # process 512 positions at a time
        for start in range(0, logits.shape[1], chunk_size):
            end = min(start + chunk_size, logits.shape[1])
            chunk_logits = logits[:, start:end, :].float()
            chunk_targets = targets[:, start:end]
            nll = F.cross_entropy(chunk_logits.reshape(-1, chunk_logits.size(-1)),
                                  chunk_targets.reshape(-1), reduction="sum")
            chunk_nll_tensor = chunk_nll_tensor + nll
            del chunk_logits, nll
        chunk_nll = float(chunk_nll_tensor.item())
        total_nll += chunk_nll
        total_tokens += targets.numel()

        per_chunk.append({
            "chunk_idx": int(chunk_offset) + i,
            "book_idx": (book_indices[i] if book_indices is not None else None),
            "nll": chunk_nll,
            "tokens": int(targets.numel()),
        })

        if (i + 1) % 5 == 0 or i == len(chunks) - 1:
            ppl_so_far = math.exp(total_nll / total_tokens)
            print(f"  Dense [{i+1}/{len(chunks)}]: ppl={ppl_so_far:.2f}")

        del outputs, logits
        gc.collect()
        torch.cuda.empty_cache()

    ppl = math.exp(total_nll / total_tokens)
    return {
        "perplexity": ppl,
        "total_nll": total_nll,
        "total_tokens": total_tokens,
        "per_chunk": per_chunk,
    }


def _reduce_step_aggs(step_aggs: list[dict]) -> dict:
    """Reduce a list of per-step aggregated telemetry dicts to a single
    summary covering the hard-STOP triggers and paper §8.6 fields that
    were previously dropped by compute_certified_perplexity. Everything is
    a scalar — safe to embed in the output JSON.

    Reductions:
      - cumulative counts (sum): *_fired_layers, *_triggered_heads,
        *_violation_heads, ranking_disagree_r*
      - boolean-fired counts (sum of True): rung*_fired_steps,
        boundary_check_fired_steps
      - means (avg across steps): e_key_step_mean, delta_bound_step_mean,
        tail_mass_int8_est_step_mean, k_star_mean
      - maxes (max across steps): e_key_step_max, v_max_global,
        tail_mass_int8_est_step_max, k_star_max
    """
    if not step_aggs:
        return {"n_steps_aggregated": 0}

    def _sum(key: str) -> int | float:
        return sum(s.get(key, 0) for s in step_aggs)

    def _count_true(key: str) -> int:
        return sum(1 for s in step_aggs if bool(s.get(key)))

    def _mean(key: str) -> float | None:
        vals = [s.get(key) for s in step_aggs if s.get(key) is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def _max(key: str) -> float | None:
        vals = [s.get(key) for s in step_aggs if s.get(key) is not None]
        if not vals:
            return None
        return float(max(vals))

    summary: dict = {"n_steps_aggregated": len(step_aggs)}
    # Hard-STOP triggers — these MUST be in the output JSON for §8.6 checks.
    summary["score_consistency_violation_heads_total"] = _sum(
        "score_consistency_violation_heads_total"
    )
    summary["rung4_fired_steps"] = _count_true("rung4_fired")
    summary["rung1_fired_steps"] = _count_true("rung1_fired")
    summary["rung2_fired_steps"] = _count_true("rung2_fired")
    summary["rung3_fired_steps"] = _count_true("rung3_fired")
    summary["rung1_fired_layers_total"] = _sum("rung1_fired_layers")
    summary["rung2_fired_layers_total"] = _sum("rung2_fired_layers")
    summary["rung3_fired_layers_total"] = _sum("rung3_fired_layers")
    summary["rung4_fired_layers_total"] = _sum("rung4_fired_layers")
    # Eq. 30 boundary verification (paper §8.6 expects 0 triggers).
    summary["boundary_check_fired_steps"] = _count_true("boundary_check_fired")
    summary["boundary_check_fired_layers_total"] = _sum("boundary_check_fired_layers")
    summary["boundary_check_triggered_heads_total"] = _sum(
        "boundary_check_triggered_heads_total"
    )
    # Ranking disagreement (rare on paper-§7 config).
    summary["ranking_disagree_r1_total"] = _sum("ranking_disagree_r1")
    summary["ranking_disagree_r3_total"] = _sum("ranking_disagree_r3")
    summary["ranking_fallback_triggered_total"] = _sum("ranking_fallback_triggered")
    summary["ranking_heads_total_all_steps"] = _sum("ranking_heads_total")
    # Bound / telemetry scalars (paper §4.5 E_key, Δ, V_max, ᾱ_T).
    for key, reducer_name in [
        ("e_key_step_mean", "mean"),
        ("e_key_step_max", "max"),
        ("v_max_global", "max"),
        ("delta_bound_step_mean", "mean"),
        ("tail_mass_int8_est_step_mean", "mean"),
        ("tail_mass_int8_est_step_max", "max"),
        ("k_star_mean", "mean"),
        ("k_star_max", "max"),
        ("e_val_pre_rung2_step_max", "max"),
        ("e_val_step_max", "max"),
    ]:
        val = _mean(key) if reducer_name == "mean" else _max(key)
        if val is not None:
            summary[key] = val
    # H2D bandwidth totals (paper §3.2 tiered-memory diagnostics).
    summary["h2d_key_bytes_total"] = _sum("h2d_key_bytes")
    summary["h2d_value_bytes_total"] = _sum("h2d_value_bytes")
    summary["h2d_total_bytes_total"] = _sum("h2d_total_bytes")
    summary["value_fallback_blocks_total"] = _sum("value_fallback_blocks")
    summary["value_fallback_head_blocks_total"] = _sum("value_fallback_head_blocks")
    summary["vram_fp16_key_cache_bytes_max"] = _max("vram_fp16_key_cache_bytes") or 0
    summary["vram_fp16_value_cache_bytes_max"] = _max("vram_fp16_value_cache_bytes") or 0
    summary["fp16_value_cache_hits_total"] = _sum("fp16_value_cache_hits_step")
    summary["fp16_value_cache_misses_total"] = _sum("fp16_value_cache_misses_step")
    summary["fp16_value_cache_evictions_total"] = _sum("fp16_value_cache_evictions_step")
    summary["fp16_value_cache_overflow_steps"] = _sum("fp16_value_cache_overflow_step")
    value_accesses = (
        summary["fp16_value_cache_hits_total"]
        + summary["fp16_value_cache_misses_total"]
    )
    summary["fp16_value_cache_hit_rate"] = (
        summary["fp16_value_cache_hits_total"] / value_accesses
        if value_accesses else 0.0
    )
    return summary


def _layer_entries_to_step_aggs(entries: list[dict], num_layers: int) -> list[dict]:
    """Aggregate layer-level stat entries into per-token step summaries.

    This preserves the output telemetry shape without calling
    aggregate_step_stats() in the decode hot loop.
    """
    if not entries or num_layers <= 0:
        return []
    step_aggs: list[dict] = []
    for start in range(0, len(entries), num_layers):
        group = entries[start:start + num_layers]
        if not group:
            continue
        total = sum(s.get("total_blocks", 0) for s in group)
        skipped = sum(s.get("skipped_blocks", 0) for s in group)
        agg: dict = {
            "skip_rate": skipped / total if total else 0.0,
            "total_blocks": total,
            "skipped_blocks": skipped,
        }
        for key in ("ranking_heads_total", "ranking_disagree_r1", "ranking_disagree_r3",
                    "ranking_fallback_triggered", "score_consistency_violation_heads",
                    "h2d_key_bytes", "h2d_value_bytes", "h2d_total_bytes",
                    "h2d_key_blocks", "h2d_value_blocks",
                    "boundary_check_triggered_heads",
                    "value_fallback_blocks", "value_fallback_head_blocks",
                    "fp16_value_cache_hits_step", "fp16_value_cache_misses_step",
                    "fp16_value_cache_evictions_step",
                    "fp16_value_cache_needed_blocks_step",
                    "fp16_value_cache_overflow_step",
                    "vram_fp16_key_cache_bytes",
                    "vram_fp16_value_cache_bytes"):
            if any(key in s for s in group):
                out_key = (
                    "score_consistency_violation_heads_total"
                    if key == "score_consistency_violation_heads"
                    else "boundary_check_triggered_heads_total"
                    if key == "boundary_check_triggered_heads"
                    else key
                )
                agg[out_key] = sum(s.get(key, 0) for s in group)
        for rung_k in ("rung1_fired", "rung2_fired", "rung3_fired", "rung4_fired"):
            if any(rung_k in s for s in group):
                agg[rung_k] = any(bool(s.get(rung_k)) for s in group)
                agg[rung_k.replace("fired", "fired_layers")] = sum(
                    1 for s in group if bool(s.get(rung_k))
                )
        if any("boundary_check_fired" in s for s in group):
            agg["boundary_check_fired"] = any(bool(s.get("boundary_check_fired")) for s in group)
            agg["boundary_check_fired_layers"] = sum(
                1 for s in group if bool(s.get("boundary_check_fired"))
            )
        for key in ("k_star_mean", "tail_mass_int8_est_mean", "delta_bound_mean",
                    "e_key_step_mean"):
            vals = [s.get(key) for s in group if s.get(key) is not None]
            if vals:
                out_key = (
                    "tail_mass_int8_est_step_mean"
                    if key == "tail_mass_int8_est_mean"
                    else "delta_bound_step_mean"
                    if key == "delta_bound_mean"
                    else key
                )
                agg[out_key] = float(sum(vals) / len(vals))
        for key in ("k_star_max", "tail_mass_int8_est_max", "e_key_step_max",
                    "v_max_layer", "e_val_pre_rung2_max", "e_val_max"):
            vals = [s.get(key) for s in group if s.get(key) is not None]
            if vals:
                out_key = (
                    "tail_mass_int8_est_step_max"
                    if key == "tail_mass_int8_est_max"
                    else "v_max_global"
                    if key == "v_max_layer"
                    else "e_val_pre_rung2_step_max"
                    if key == "e_val_pre_rung2_max"
                    else "e_val_step_max"
                    if key == "e_val_max"
                    else key
                )
                agg[out_key] = float(max(vals))
        step_aggs.append(agg)
    return step_aggs


def compute_certified_perplexity(
    model, adapter, chunks: list[torch.Tensor],
    *,
    v_tolerance: float,
    eval_start_frac: float = 0.5,
    top_k_override: int = None,
    device: str = "cuda",
    use_int4_values: bool = False,
    group_size: int = 16,
    fp16_key_cache_blocks: int | str | None = None,
    fp16_value_cache_blocks: int | str | None = None,
    # Paper-alignment features (T4/T7/Rung1/T9/T10).
    tau_cov: float | None = PAPER_TAU_COV,
    k_min: int = PAPER_K_MIN,
    k_max: int | None = PAPER_K_MAX,
    ranking_fallback: bool = True,
    ranking_r: int = PAPER_RANKING_R,
    ranking_fallback_mode: str = "full",
    score_consistency_check: bool = False,
    score_consistency_interval: int = 1,
    eps_guard: float = PAPER_EPS_GUARD,
    exploration_rate: float = PAPER_EXPLORATION_RATE,
    rung1_threshold: float = PAPER_RUNG1_THRESHOLD,
    rung1_multiplier: float = PAPER_RUNG1_MULTIPLIER,
    telemetry_collector=None,
    book_indices: list[int] | None = None,
    telemetry_mode: str = "summary",
    collect_step_stats: bool = False,
    max_certified_steps: int | None = None,
    certified_warmup_steps: int = 128,
    chunk_offset: int = 0,
) -> dict:
    """Compute perplexity using certified attention decode.

    For each chunk:
      1. Dense prefill on the first `prefix_len` tokens
      2. Build tiered KV cache from dense KV pairs
      3. Teacher-forced certified decode for remaining tokens
      4. Compute NLL from certified logits

    Args:
        eval_start_frac: fraction of context to use as dense prefix.
            Tokens after this point are evaluated with certified attention.
    """
    from dotcache.integrations.llama import _ensure_certified_imports, CertifiedAttentionState
    from dotcache.kernels.tiered_kv_cache import (
        create_tiered_cache_from_model,
        create_tiered_cache_int4v_from_model,
    )

    total_nll = 0.0
    total_tokens = 0
    total_skipped = 0
    total_blocks = 0
    total_steps = 0
    per_chunk: list[dict] = []
    # Accumulator across all chunks — reduced into `overall_telemetry` at
    # the end. Captures the full aggregate dict per step so we can emit
    # the §8.6 hard-STOP triggers that were previously dropped.
    all_step_aggs_accumulator: list[dict] = []
    if telemetry_mode not in {"debug", "summary", "off"}:
        raise ValueError("telemetry_mode must be one of: debug, summary, off")
    collect_step_stats = bool(collect_step_stats or telemetry_mode == "debug")

    def _cache_runtime_summary(tiered_caches: dict) -> dict:
        caches = list(tiered_caches.values())
        if not caches:
            return {}
        return {
            "static_resident_key_cache": bool(all(
                bool(getattr(c, "static_resident_key_cache", False)) for c in caches
            )),
            "static_resident_value_cache": bool(all(
                bool(getattr(c, "static_resident_value_cache", False)) for c in caches
            )),
            "static_resident_key_prepare_bytes": int(sum(
                int(getattr(c, "static_resident_key_prepare_bytes", 0)) for c in caches
            )),
            "static_resident_value_prepare_bytes": int(sum(
                int(getattr(c, "static_resident_value_prepare_bytes", 0)) for c in caches
            )),
            "vram_fp16_key_cache_bytes": int(sum(
                c.keys_fp16_gpu.nelement() * c.keys_fp16_gpu.element_size()
                for c in caches if getattr(c, "keys_fp16_gpu", None) is not None
            )),
            "vram_fp16_value_cache_bytes": int(sum(
                c.values_fp16_gpu.nelement() * c.values_fp16_gpu.element_size()
                for c in caches if getattr(c, "values_fp16_gpu", None) is not None
            )),
        }

    for i, chunk in enumerate(chunks):
        seq_len = chunk.shape[0]
        prefix_len = int(seq_len * eval_start_frac)
        eval_len = seq_len - prefix_len
        decode_limit = eval_len - 1
        if max_certified_steps is not None:
            decode_limit = min(decode_limit, max(0, int(max_certified_steps)))

        input_ids = chunk.unsqueeze(0).to(device)

        # Phase 1: Dense prefill for prefix
        setup_t0 = time.perf_counter()
        adapter.set_mode("dense")
        with torch.inference_mode():
            prefix_out = model(input_ids=input_ids[:, :prefix_len], use_cache=True)
        past_kv = prefix_out.past_key_values
        print(
            f"  Certified [{i+1}/{len(chunks)}] prefill_ms={(time.perf_counter() - setup_t0) * 1000.0:.1f}",
            flush=True,
        )

        # Compute prefix NLL in chunks (same technique as dense path)
        nll_t0 = time.perf_counter()
        prefix_logits = prefix_out.logits[:, :-1, :]
        prefix_targets = input_ids[:, 1:prefix_len]
        prefix_nll_tensor = torch.zeros((), dtype=torch.float32, device=device)
        pchunk = 512
        for pstart in range(0, prefix_logits.shape[1], pchunk):
            pend = min(pstart + pchunk, prefix_logits.shape[1])
            pl = prefix_logits[:, pstart:pend, :].float()
            pt = prefix_targets[:, pstart:pend]
            prefix_nll_tensor = prefix_nll_tensor + F.cross_entropy(
                pl.reshape(-1, pl.size(-1)), pt.reshape(-1), reduction="sum"
            )
            del pl
        prefix_nll = float(prefix_nll_tensor.item())
        del prefix_out, prefix_logits
        print(
            f"  Certified [{i+1}/{len(chunks)}] prefix_nll_ms={(time.perf_counter() - nll_t0) * 1000.0:.1f}",
            flush=True,
        )

        # Phase 2: Build tiered cache with enough room for eval tokens
        _ensure_certified_imports()
        layer_ids = list(range(model.config.num_hidden_layers))
        _env_key_cap = os.environ.get("DOTCACHE_FP16_CACHE_BLOCKS")
        _env_value_cap = os.environ.get("DOTCACHE_FP16_VALUE_CACHE_BLOCKS")
        from _provenance import resolve_fp16_key_cache_blocks, resolve_fp16_value_cache_blocks
        _key_cap = resolve_fp16_key_cache_blocks(fp16_key_cache_blocks, _env_key_cap)
        _value_cap = resolve_fp16_value_cache_blocks(fp16_value_cache_blocks, _env_value_cap)

        def _build_tiered_caches():
            if use_int4_values:
                return create_tiered_cache_int4v_from_model(
                    past_kv, layer_ids, group_size=group_size,
                    max_new_tokens=decode_limit + 16,
                    fp16_key_cache_capacity=_key_cap,
                    fp16_value_cache_capacity=_value_cap,
                )
            return create_tiered_cache_from_model(
                past_kv, layer_ids, max_new_tokens=decode_limit + 16,
                fp16_key_cache_capacity=None,
            )

        top_k = top_k_override if top_k_override is not None else 4

        def _make_certified_state(tiered_caches, *, collect_stats: bool):
            return CertifiedAttentionState(
                tiered_caches=tiered_caches,
                collect_stats=collect_stats,
                append_kv=True,
                top_k_fp16_keys=top_k,
                v_tolerance=v_tolerance,
                tau_cov=tau_cov,
                k_min=k_min,
                k_max=k_max,
                ranking_fallback=ranking_fallback,
                ranking_r=ranking_r,
                ranking_fallback_mode=ranking_fallback_mode,
                score_consistency_check=score_consistency_check,
                score_consistency_interval=score_consistency_interval,
                eps_guard=eps_guard,
                exploration_rate=exploration_rate,
                rung1_threshold=rung1_threshold,
                rung1_multiplier=rung1_multiplier,
            )

        warm_steps = min(max(0, int(certified_warmup_steps)), max(0, decode_limit))
        if i == 0 and warm_steps > 0:
            warm_t0 = time.perf_counter()
            warm_caches = _build_tiered_caches()
            adapter.certified_state = _make_certified_state(
                warm_caches, collect_stats=False,
            )
            adapter.set_mode("certified")
            adapter.reset_runtime_metrics()
            warm_cache_position = torch.tensor([prefix_len], dtype=torch.long, device=device)
            with torch.inference_mode():
                for wt in range(warm_steps):
                    warm_token_id = input_ids[:, prefix_len + wt:prefix_len + wt + 1]
                    warm_out = model(
                        input_ids=warm_token_id,
                        use_cache=False,
                        cache_position=warm_cache_position,
                        position_ids=warm_cache_position.unsqueeze(0),
                    )
                    warm_cache_position.add_(1)
                    del warm_out
            torch.cuda.synchronize()
            adapter.certified_state = None
            adapter.set_mode("dense")
            del warm_caches
            gc.collect()
            torch.cuda.empty_cache()
            print(
                f"  Certified [{i+1}/{len(chunks)}] warmup_steps={warm_steps} "
                f"warmup_ms={(time.perf_counter() - warm_t0) * 1000.0:.1f}",
                flush=True,
            )

        cache_t0 = time.perf_counter()
        tiered_caches = _build_tiered_caches()
        cache_runtime = _cache_runtime_summary(tiered_caches)
        print(
            f"  Certified [{i+1}/{len(chunks)}] cache_build_ms={(time.perf_counter() - cache_t0) * 1000.0:.1f}",
            flush=True,
        )
        del past_kv
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"  Certified [{i+1}/{len(chunks)}] cache_cleanup_ms={(time.perf_counter() - cache_t0) * 1000.0:.1f}",
            flush=True,
        )

        adapter.certified_state = _make_certified_state(
            tiered_caches, collect_stats=collect_step_stats,
        )
        if os.environ.get("DOTCACHE_PHASE_TIMING") == "1":
            adapter.certified_state.phase_timings = {}
        adapter.set_mode("certified")
        adapter.reset_runtime_metrics()

        # Phase 3: Teacher-forced certified decode
        cache_position = torch.tensor([prefix_len], dtype=torch.long, device=device)
        suffix_nll_tensor = torch.zeros((), dtype=torch.float32, device=device)
        chunk_skipped = 0
        chunk_blocks = 0
        # Per-step aggregated telemetry — previously thrown away, now
        # reduced into per-chunk + overall summaries so the hard-STOP
        # triggers from docs/paper_v1_run_handoff.md §5
        # (score_consistency_violation_heads_total, rung4_fired,
        # e_key_step_mean, boundary_check_fired_layers, …) are actually
        # visible in the output JSON for diagnosis.
        chunk_step_aggs: list[dict] = []
        chunk_cert_t0 = time.perf_counter()
        progress_stats_cursor = 0

        for t in range(decode_limit):
            token_id = input_ids[:, prefix_len + t:prefix_len + t + 1]
            with torch.inference_mode():
                out = model(
                    input_ids=token_id,
                    use_cache=False,
                    cache_position=cache_position,
                    position_ids=cache_position.unsqueeze(0),
                )
            if telemetry_collector is not None:
                telemetry_collector.record_step()
            # Loss: predict token at prefix_len + t + 1
            logits = out.logits[:, -1, :].float()
            target = input_ids[:, prefix_len + t + 1]
            nll = F.cross_entropy(logits, target, reduction="sum")
            suffix_nll_tensor = suffix_nll_tensor + nll
            cache_position.add_(1)

            if telemetry_mode == "debug":
                # Drain per-step aggregated stats — keep the FULL dict, not just
                # two fields. _reduce_step_aggs() below rolls them up.
                step = adapter.certified_state.aggregate_step_stats()
                chunk_blocks += step.get("total_blocks", 0)
                chunk_skipped += step.get("skipped_blocks", 0)
                chunk_step_aggs.append(step)
                adapter.certified_state.clear_step_stats()
            total_steps += 1
            if (t + 1) % 128 == 0:
                elapsed = time.perf_counter() - chunk_cert_t0
                prof = adapter.runtime_profile_summary(
                    model_forward_ms_total=elapsed * 1000.0,
                )
                calls = max(
                    sum(layer.get("call_count", 0) for layer in prof.get("per_layer", [])),
                    1,
                )
                print(
                    f"    certified progress chunk={i+1}/{len(chunks)} "
                    f"tokens={t+1}/{decode_limit} "
                    f"tok/s={(t+1)/max(elapsed, 1e-9):.2f} "
                    f"qkv_ms/layer={prof['qkv_projection_ms_total']/calls:.3f} "
                    f"append_ms/layer={prof['append_runtime_ms_total']/calls:.3f} "
                    f"attn_ms/layer={prof['decode_runtime_ms_total']/calls:.3f} "
                    f"out_ms/layer={prof['output_projection_ms_total']/calls:.3f}",
                    flush=True,
                )
                phase_timings = getattr(adapter.certified_state, "phase_timings", None)
                if phase_timings:
                    denom = max((t + 1) * len(layer_ids), 1)
                    phase_msg = " ".join(
                        f"{k[:-3]}_ms/layer={v / denom / 1000.0:.3f}"
                        for k, v in sorted(phase_timings.items())
                    )
                    print(f"    certified phases {phase_msg}", flush=True)
                progress_entries = []
                if collect_step_stats:
                    progress_entries = adapter.certified_state.step_stats[progress_stats_cursor:]
                    progress_stats_cursor += len(progress_entries)
                if progress_entries:
                    rung2_layers = sum(bool(s.get("rung2_fired")) for s in progress_entries)
                    int4_layers = sum(s.get("v_format") == "int4" for s in progress_entries)
                    mixed_layers = sum(s.get("v_format") == "mixed" for s in progress_entries)
                    fp16_layers = sum(s.get("v_format") == "fp16" for s in progress_entries)
                    h2d_value_mb = sum(s.get("h2d_value_bytes", 0) for s in progress_entries) / 1e6
                    fallback_blocks = sum(s.get("value_fallback_blocks", 0) for s in progress_entries)
                    fallback_head_blocks = sum(s.get("value_fallback_head_blocks", 0) for s in progress_entries)
                    e_pre_vals = [
                        s.get("e_val_pre_rung2_max")
                        for s in progress_entries
                        if s.get("e_val_pre_rung2_max") is not None
                    ]
                    e_post_vals = [
                        s.get("e_val_max")
                        for s in progress_entries
                        if s.get("e_val_max") is not None
                    ]
                    print(
                        f"    certified value_fallback "
                        f"layers(r2/int4/mixed/fp16)="
                        f"{rung2_layers}/{int4_layers}/{mixed_layers}/{fp16_layers} "
                        f"blocks={fallback_blocks} head_blocks={fallback_head_blocks} "
                        f"h2d_value_mb={h2d_value_mb:.2f} "
                        f"e_val_pre_max={(max(e_pre_vals) if e_pre_vals else 0.0):.4f} "
                        f"e_val_post_max={(max(e_post_vals) if e_post_vals else 0.0):.4f}",
                        flush=True,
                    )

        suffix_nll = float(suffix_nll_tensor.item())
        chunk_runtime_profile = adapter.runtime_profile_summary(
            model_forward_ms_total=(time.perf_counter() - chunk_cert_t0) * 1000.0,
        )
        if telemetry_mode == "summary" and collect_step_stats:
            layer_entries = adapter.certified_state.clear_step_stats()
            chunk_step_aggs = _layer_entries_to_step_aggs(layer_entries, len(layer_ids))
            chunk_blocks = sum(s.get("total_blocks", 0) for s in chunk_step_aggs)
            chunk_skipped = sum(s.get("skipped_blocks", 0) for s in chunk_step_aggs)
        elif telemetry_mode in {"summary", "off"}:
            chunk_step_aggs = []

        adapter.certified_state = None
        adapter.set_mode("dense")

        # Total NLL = prefix (dense) + suffix (certified)
        chunk_nll = prefix_nll + suffix_nll
        chunk_tokens = (prefix_len - 1) + decode_limit
        total_nll += chunk_nll
        total_tokens += chunk_tokens

        total_skipped += chunk_skipped
        total_blocks += chunk_blocks

        # Keep prefix_nll / suffix_nll separately — reviewers may want to
        # report CI only over the suffix (the actually-certified portion)
        # rather than the full chunk. prefix_tokens and suffix_tokens let
        # them weight consistently.
        chunk_telemetry = _reduce_step_aggs(chunk_step_aggs)
        per_chunk.append({
            "chunk_idx": int(chunk_offset) + i,
            "book_idx": (book_indices[i] if book_indices is not None else None),
            "prefix_nll": float(prefix_nll),
            "prefix_tokens": int(prefix_len - 1),
            "suffix_nll": float(suffix_nll),
            "suffix_tokens": int(decode_limit),
            "nll": float(chunk_nll),
            "tokens": int(chunk_tokens),
            "skipped_blocks": int(chunk_skipped),
            "total_blocks": int(chunk_blocks),
            "telemetry": chunk_telemetry,
            "cache_runtime": cache_runtime,
            "runtime_profile": chunk_runtime_profile,
        })
        # Accumulate for overall summary (all chunks reduced together).
        all_step_aggs_accumulator.extend(chunk_step_aggs)

        chunk_ppl = math.exp(chunk_nll / chunk_tokens)
        suffix_ppl = math.exp(suffix_nll / max(decode_limit, 1))
        chunk_skip_rate = chunk_skipped / chunk_blocks if chunk_blocks else 0.0
        if (i + 1) % 5 == 0 or i == len(chunks) - 1:
            ppl_so_far = math.exp(total_nll / total_tokens)
            overall_skip = total_skipped / total_blocks if total_blocks else 0.0
            print(f"  Certified [{i+1}/{len(chunks)}]: chunk_ppl={chunk_ppl:.2f}, "
                  f"suffix_ppl={suffix_ppl:.2f}, running_ppl={ppl_so_far:.2f}, "
                  f"skip={chunk_skip_rate:.3f} (overall {overall_skip:.3f})")

        del tiered_caches
        gc.collect()
        torch.cuda.empty_cache()

    ppl = math.exp(total_nll / total_tokens)
    skip_rate = total_skipped / total_blocks if total_blocks else 0.0
    return {
        "perplexity": ppl,
        "total_nll": total_nll,
        "total_tokens": total_tokens,
        # `skip_rate` / `skipped_blocks` are legacy from Paper-2's
        # block-skipping semantics. Paper-1 (hybrid attend-all) does NOT
        # skip blocks — the hybrid kernel attends EVERY block, some with
        # FP16 keys (top-K*) and some with INT8 keys (tail). What this
        # counter actually measures is the tail-block fraction from the
        # adaptive top-K* selector, i.e. `1 - (K* / num_blocks)`. Do NOT
        # use it as a quality/correctness signal. Kept here for legacy
        # consumers; the meaningful fields are in `telemetry` below.
        "skip_rate": skip_rate,
        "skipped_blocks": total_skipped,
        "total_blocks": total_blocks,
        "decode_steps": total_steps,
        "max_certified_steps": max_certified_steps,
        "truncated_certified_decode": max_certified_steps is not None,
        "collect_step_stats": collect_step_stats,
        "score_consistency_interval": int(score_consistency_interval),
        # Reduced telemetry across all certified decode steps — THIS is
        # where the §8.6 hard-STOP signals and paper §4.5 bound scalars
        # actually live. See docs/paper_v1_run_handoff.md §5.
        "telemetry": _reduce_step_aggs(all_step_aggs_accumulator),
        "per_chunk": per_chunk,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PG-19 perplexity: dense vs certified")
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
    parser.add_argument("--context", type=int, default=8192,
                        help="Context length per chunk")
    parser.add_argument("--num-chunks", type=int, default=20,
                        help="Number of document chunks to evaluate")
    parser.add_argument("--chunk-start", type=int, default=0,
                        help="Global PG-19 chunk index to start from (for distributed shards).")
    parser.add_argument("--chunk-index", type=int, default=None,
                        help="Alias for --chunk-start with --num-chunks 1.")
    parser.add_argument("--eval-start", type=float, default=0.5,
                        help="Fraction of context for dense prefix (rest is certified)")
    parser.add_argument("--output", default="benchmarks/results/pg19_perplexity.json")
    parser.add_argument("--dense-only", action="store_true",
                        help="Only run dense baseline (skip certified)")
    parser.add_argument("--top-k-override", type=int, default=None,
                        help="Override top_k_fp16_keys (default: 4)")
    # Paper-alignment flags (T4/T7/Rung1/T9/T10).
    parser.add_argument("--tau-cov", type=float, default=PAPER_TAU_COV,
                        help=f"Adaptive K* cumulative-mass threshold (paper default {PAPER_TAU_COV}; set 0 to disable)")
    parser.add_argument("--k-min", type=int, default=PAPER_K_MIN)
    parser.add_argument("--k-max", type=int, default=PAPER_K_MAX,
                        help=f"Adaptive K* upper clamp (paper default {PAPER_K_MAX})")
    parser.add_argument("--ranking-fallback", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Rung-3 ranking-consistency fallback (paper default: enabled)")
    parser.add_argument("--ranking-r", type=int, default=PAPER_RANKING_R)
    parser.add_argument("--ranking-fallback-mode", default="full", choices=["full", "measure"])
    parser.add_argument("--score-consistency-check", action="store_true")
    parser.add_argument("--score-consistency-interval", type=int, default=1,
                        help="Run score-consistency canary every N decode steps (1 = exact every step).")
    parser.add_argument("--eps-guard", type=float, default=PAPER_EPS_GUARD)
    parser.add_argument("--exploration-rate", type=float, default=PAPER_EXPLORATION_RATE)
    parser.add_argument("--rung1-threshold", type=float, default=PAPER_RUNG1_THRESHOLD)
    parser.add_argument("--rung1-multiplier", type=float, default=PAPER_RUNG1_MULTIPLIER)
    parser.add_argument("--pagein-telemetry", action="store_true",
                        help="Collect per-step page-in / rung / VRAM-cache telemetry (Test 3)")
    parser.add_argument("--telemetry-output", default=None,
                        help="Path to write per-step telemetry JSON (default: <output>.pagein.json)")
    parser.add_argument("--telemetry-mode", default="summary",
                        choices=["debug", "summary", "off"],
                        help="debug drains telemetry every token; summary keeps hot-path stats off unless --collect-step-stats is set; off disables stats")
    parser.add_argument("--collect-step-stats", action="store_true",
                        help="Collect full per-layer certified telemetry in summary mode. Slow; use for diagnostic sweeps, not throughput timing.")
    parser.add_argument("--max-certified-steps", type=int, default=None,
                        help="Calibration only: limit certified suffix decode steps. Do not use for paper tables.")
    parser.add_argument("--certified-warmup-steps", type=int, default=128,
                        help="Run untimed certified decode steps on a throwaway cache before timing (0 disables).")
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from _provenance import (
        add_paper_cache_args,
        cache_config_dict,
        configure_paper_runtime_defaults,
    )
    add_paper_cache_args(parser)
    args = parser.parse_args()
    configure_paper_runtime_defaults()

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from dotcache.integrations.llama import LlamaDotCacheModelAdapter
    from dotcache.config import DotCacheConfig

    token = os.environ.get("HF_TOKEN") or None
    warnings.filterwarnings(
        "ignore",
        message=r"MatMul8bitLt: inputs will be cast from .* during quantization",
        category=UserWarning,
    )
    print(f"Loading {args.model} (INT8)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant_config, device_map="auto",
        dtype=torch.float16, token=token,
    )
    model.eval()
    print(f"Model: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    config = DotCacheConfig(head_dim=head_dim)
    adapter = LlamaDotCacheModelAdapter(model, config)

    if args.chunk_index is not None:
        args.chunk_start = int(args.chunk_index)
        args.num_chunks = 1

    # Load PG-19 chunks
    print(
        f"\nLoading PG-19 test set: {args.num_chunks} chunks × {args.context} tokens "
        f"(start={args.chunk_start})..."
    )
    t0 = time.perf_counter()
    chunks, book_indices = load_pg19_chunks(
        tokenizer, args.context, args.num_chunks, start_chunk=args.chunk_start,
    )
    print(f"Loaded {len(chunks)} chunks in {time.perf_counter()-t0:.1f}s")

    # Dense perplexity
    print(f"\n{'='*50}")
    print("Dense perplexity")
    print(f"{'='*50}")
    t0 = time.perf_counter()
    dense_result = compute_dense_perplexity(
        model, chunks, book_indices=book_indices, chunk_offset=args.chunk_start,
    )
    t_dense = time.perf_counter() - t0
    print(f"Dense: ppl={dense_result['perplexity']:.2f} "
          f"({dense_result['total_tokens']} tokens, {t_dense:.1f}s)")
    # Per-chunk bits-per-token CI — bootstrap over chunks.
    dense_bpt = per_chunk_bpt_stats(dense_result["per_chunk"], field="nll", tokens_field="tokens")
    dense_result["bpt_stats"] = dense_bpt
    if dense_bpt.get("n_chunks", 0) > 0:
        print(f"  bits/token: {dense_bpt['bpt_mean']:.4f} "
              f"(95% bootstrap CI [{dense_bpt['bpt_ci_lo']:.4f}, {dense_bpt['bpt_ci_hi']:.4f}], "
              f"n={dense_bpt['n_chunks']} chunks)")

    # Certified perplexity
    cert_result = None
    telemetry_collector = None
    if args.pagein_telemetry and not args.dense_only:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from _pagein_telemetry import PageinTelemetry
        telemetry_collector = PageinTelemetry(adapter, enabled=True)
        telemetry_collector.start()
    if not args.dense_only:
        print(f"\n{'='*50}")
        print("Certified perplexity")
        print(f"{'='*50}")
        t0 = time.perf_counter()
        cert_result = compute_certified_perplexity(
            model, adapter, chunks,
            eval_start_frac=args.eval_start,
            top_k_override=args.top_k_override,
            v_tolerance=args.v_tolerance,
            use_int4_values=args.use_int4_values,
            group_size=args.group_size,
            fp16_key_cache_blocks=args.fp16_key_cache_blocks,
            fp16_value_cache_blocks=args.fp16_value_cache_blocks,
            tau_cov=(args.tau_cov if args.tau_cov and args.tau_cov > 0 else None),
            k_min=args.k_min,
            k_max=args.k_max,
            ranking_fallback=args.ranking_fallback,
            ranking_r=args.ranking_r,
            ranking_fallback_mode=args.ranking_fallback_mode,
            score_consistency_check=args.score_consistency_check,
            score_consistency_interval=args.score_consistency_interval,
            eps_guard=args.eps_guard,
            exploration_rate=args.exploration_rate,
            rung1_threshold=args.rung1_threshold,
            rung1_multiplier=args.rung1_multiplier,
            telemetry_collector=telemetry_collector,
            book_indices=book_indices,
            telemetry_mode=args.telemetry_mode,
            collect_step_stats=args.collect_step_stats,
            max_certified_steps=args.max_certified_steps,
            certified_warmup_steps=args.certified_warmup_steps,
            chunk_offset=args.chunk_start,
        )
        t_cert = time.perf_counter() - t0
        print(f"Certified: ppl={cert_result['perplexity']:.2f} "
              f"({cert_result['total_tokens']} tokens, {t_cert:.1f}s)")
        # Per-chunk bits-per-token CI, full chunk + suffix-only (the
        # certified portion). The suffix slice is what the reviewer
        # usually wants since the prefix is always dense.
        cert_bpt_full = per_chunk_bpt_stats(
            cert_result["per_chunk"], field="nll", tokens_field="tokens",
        )
        cert_bpt_suffix = per_chunk_bpt_stats(
            cert_result["per_chunk"], field="suffix_nll", tokens_field="suffix_tokens",
        )
        cert_result["bpt_stats"] = cert_bpt_full
        cert_result["bpt_stats_suffix"] = cert_bpt_suffix
        if cert_bpt_full.get("n_chunks", 0) > 0:
            print(f"  bits/token (full): {cert_bpt_full['bpt_mean']:.4f} "
                  f"(95% CI [{cert_bpt_full['bpt_ci_lo']:.4f}, {cert_bpt_full['bpt_ci_hi']:.4f}])")
            print(f"  bits/token (certified suffix only): {cert_bpt_suffix['bpt_mean']:.4f} "
                  f"(95% CI [{cert_bpt_suffix['bpt_ci_lo']:.4f}, {cert_bpt_suffix['bpt_ci_hi']:.4f}])")
        if telemetry_collector is not None:
            telemetry_collector.finish()
            tele_path = args.telemetry_output or (str(args.output).replace(".json", ".pagein.json"))
            telemetry_collector.write_json(tele_path)
            s = telemetry_collector.summary()
            print(f"Page-in telemetry: n_steps={s.get('n_steps',0)} "
                  f"h2d_mean={s.get('h2d_total_bytes_mean',0)/1024:.1f} KB/step, "
                  f"pct_zero_pagein={s.get('pct_steps_zero_pagein',0):.1%}, "
                  f"rung1_rate={s.get('rung1_rate',0):.2%}, rung2_rate={s.get('rung2_rate',0):.2%}, "
                  f"rung3_rate={s.get('rung3_rate',0):.2%}, rung4_rate={s.get('rung4_rate',0):.2%}")

    # Summary
    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    print(f"Context: {args.context} tokens, {len(chunks)} chunks")
    print(f"Dense perplexity:     {dense_result['perplexity']:.4f}")
    if cert_result:
        ratio = cert_result['perplexity'] / dense_result['perplexity']
        delta = cert_result['perplexity'] - dense_result['perplexity']
        print(f"Certified perplexity: {cert_result['perplexity']:.4f}")
        print(f"Ratio (cert/dense):   {ratio:.6f}")
        print(f"Delta:                {delta:+.4f}")
        paired_delta = per_chunk_ppl_delta_stats(
            dense_result["per_chunk"], cert_result["per_chunk"],
        )
        cert_result["paired_delta_stats"] = paired_delta
        if paired_delta.get("n_chunks", 0):
            print(
                "Per-chunk Δppl:       "
                f"{paired_delta['mean_delta_ppl']:+.4f} "
                f"(95% CI [{paired_delta['delta_ppl_ci_lo']:+.4f}, "
                f"{paired_delta['delta_ppl_ci_hi']:+.4f}], "
                f"n={paired_delta['n_chunks']})"
            )

    # Save results
    output = {
        "benchmark": "pg19_perplexity",
        "model": args.model,
        "context_length": args.context,
        "num_chunks": len(chunks),
        "eval_start_frac": args.eval_start,
        "top_k_override": args.top_k_override,
        "max_certified_steps": args.max_certified_steps,
        "truncated_certified_decode": args.max_certified_steps is not None,
        "telemetry_mode": args.telemetry_mode,
        "collect_step_stats": args.collect_step_stats,
        "score_consistency_interval": args.score_consistency_interval,
        "cache_config": cache_config_dict(args),
        "dense": dense_result,
        "certified": cert_result,
    }
    if cert_result:
        output["ratio"] = cert_result['perplexity'] / dense_result['perplexity']
        output["delta"] = cert_result['perplexity'] - dense_result['perplexity']

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON -> {out_path}")


if __name__ == "__main__":
    main()
