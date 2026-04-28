"""Dense-vs-certified one-token decode speed comparison.

This is the useful speed comparison for paper planning: both paths run the
same PG-19 chunk, same dense prefix, same warmup token positions, and same
teacher-forced one-token decode window. Setup is reported separately so decode
throughput is not confused with model load, prefill, or cache construction.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pg19_perplexity import load_pg19_chunks
from _provenance import (
    PAPER_EPS_GUARD,
    PAPER_EXPLORATION_RATE,
    PAPER_K_MAX,
    PAPER_K_MIN,
    PAPER_RANKING_R,
    PAPER_RUNG1_MULTIPLIER,
    PAPER_RUNG1_THRESHOLD,
    PAPER_TAU_COV,
    add_paper_cache_args,
    cache_config_dict,
    configure_paper_runtime_defaults,
    resolve_fp16_key_cache_blocks,
    resolve_fp16_value_cache_blocks,
)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _event_pair() -> tuple[torch.cuda.Event | None, torch.cuda.Event | None]:
    if not torch.cuda.is_available():
        return None, None
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _ms_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    mean = float(sum(values) / len(values))
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {"mean": mean, "std": float(var ** 0.5), "n": len(values)}


def _runtime_cache_summary(tiered_caches: dict[int, Any]) -> dict[str, Any]:
    caches = list(tiered_caches.values())
    return {
        "static_resident_key_cache": bool(caches) and all(
            bool(getattr(c, "static_resident_key_cache", False)) for c in caches
        ),
        "static_resident_value_cache": bool(caches) and all(
            bool(getattr(c, "static_resident_value_cache", False)) for c in caches
        ),
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


def _ppl(nll: float, tokens: int) -> float:
    return float(math.exp(nll / max(tokens, 1)))


def measure_dense_decode(
    model,
    input_ids: torch.Tensor,
    *,
    prefix_len: int,
    warmup_steps: int,
    measure_steps: int,
) -> dict[str, Any]:
    print(f"[dense] prefill start prefix_len={prefix_len}", flush=True)
    _sync()
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model(input_ids=input_ids[:, :prefix_len], use_cache=True)
    _sync()
    prefill_s = time.perf_counter() - t0
    past = outputs.past_key_values
    del outputs

    cache_position = torch.tensor([prefix_len], dtype=torch.long, device=input_ids.device)
    print(f"[dense] warmup start steps={warmup_steps}", flush=True)
    with torch.inference_mode():
        for t in range(warmup_steps):
            token_id = input_ids[:, prefix_len + t:prefix_len + t + 1]
            out = model(
                input_ids=token_id,
                use_cache=True,
                past_key_values=past,
                cache_position=cache_position,
                position_ids=cache_position.unsqueeze(0),
            )
            past = out.past_key_values
            cache_position.add_(1)
            del out
    _sync()

    nll = torch.zeros((), dtype=torch.float32, device=input_ids.device)
    print(f"[dense] measure start steps={measure_steps}", flush=True)
    _sync()
    t0 = time.perf_counter()
    step_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    with torch.inference_mode():
        for t in range(warmup_steps, warmup_steps + measure_steps):
            ev0, ev1 = _event_pair()
            if ev0 is not None and ev1 is not None:
                ev0.record()
            token_id = input_ids[:, prefix_len + t:prefix_len + t + 1]
            out = model(
                input_ids=token_id,
                use_cache=True,
                past_key_values=past,
                cache_position=cache_position,
                position_ids=cache_position.unsqueeze(0),
            )
            past = out.past_key_values
            target = input_ids[:, prefix_len + t + 1]
            nll = nll + F.cross_entropy(out.logits[:, -1, :].float(), target, reduction="sum")
            cache_position.add_(1)
            if ev0 is not None and ev1 is not None:
                ev1.record()
                step_events.append((ev0, ev1))
            del out
    _sync()
    decode_s = time.perf_counter() - t0
    step_ms = [float(a.elapsed_time(b)) for a, b in step_events]
    nll_f = float(nll.item())
    del past
    print(f"[dense] measure done tok_s={measure_steps / max(decode_s, 1e-9):.2f}", flush=True)
    return {
        "prefill_s": float(prefill_s),
        "decode_s": float(decode_s),
        "decode_steps": int(measure_steps),
        "decode_tok_s": float(measure_steps / max(decode_s, 1e-9)),
        "decode_ms_per_token": float(1000.0 * decode_s / max(measure_steps, 1)),
        "decode_step_ms": step_ms,
        "decode_step_ms_stats": _ms_stats(step_ms),
        "nll": nll_f,
        "perplexity": _ppl(nll_f, measure_steps),
    }


def measure_certified_decode(
    model,
    adapter,
    input_ids: torch.Tensor,
    args: argparse.Namespace,
    *,
    prefix_len: int,
    warmup_steps: int,
    measure_steps: int,
) -> dict[str, Any]:
    from dotcache.integrations.llama import _ensure_certified_imports, CertifiedAttentionState
    from dotcache.kernels.tiered_kv_cache import (
        create_tiered_cache_from_model,
        create_tiered_cache_int4v_from_model,
    )

    _ensure_certified_imports()
    adapter.set_mode("dense")
    print(f"[certified] prefill start prefix_len={prefix_len}", flush=True)
    _sync()
    t0 = time.perf_counter()
    with torch.inference_mode():
        prefix_out = model(input_ids=input_ids[:, :prefix_len], use_cache=True)
    _sync()
    prefill_s = time.perf_counter() - t0
    past_kv = prefix_out.past_key_values
    del prefix_out

    layer_ids = list(range(model.config.num_hidden_layers))
    key_cap = resolve_fp16_key_cache_blocks(
        args.fp16_key_cache_blocks,
        os.environ.get("DOTCACHE_FP16_CACHE_BLOCKS"),
    )
    value_cap = resolve_fp16_value_cache_blocks(
        args.fp16_value_cache_blocks,
        os.environ.get("DOTCACHE_FP16_VALUE_CACHE_BLOCKS"),
    )
    max_new = int(args.cache_decode_budget) if args.cache_decode_budget is not None else warmup_steps + measure_steps + 16
    print(
        f"[certified] cache build start key_cap={key_cap} value_cap={value_cap} "
        f"max_new_tokens={max_new}",
        flush=True,
    )
    _sync()
    t0 = time.perf_counter()
    if args.use_int4_values:
        tiered_caches = create_tiered_cache_int4v_from_model(
            past_kv,
            layer_ids,
            group_size=args.group_size,
            max_new_tokens=max_new,
            fp16_key_cache_capacity=key_cap,
            fp16_value_cache_capacity=value_cap,
        )
    else:
        tiered_caches = create_tiered_cache_from_model(
            past_kv,
            layer_ids,
            max_new_tokens=max_new,
            fp16_key_cache_capacity=key_cap,
        )
    _sync()
    cache_build_s = time.perf_counter() - t0
    cache_runtime = _runtime_cache_summary(tiered_caches)
    print(
        f"[certified] cache build done seconds={cache_build_s:.2f} "
        f"key_vram_mb={cache_runtime['vram_fp16_key_cache_bytes'] / 1e6:.1f} "
        f"value_vram_mb={cache_runtime['vram_fp16_value_cache_bytes'] / 1e6:.1f}",
        flush=True,
    )
    del past_kv
    gc.collect()
    torch.cuda.empty_cache()
    print("[certified] cache cleanup done", flush=True)

    adapter.certified_state = CertifiedAttentionState(
        tiered_caches=tiered_caches,
        collect_stats=bool(args.collect_step_stats),
        append_kv=True,
        top_k_fp16_keys=args.top_k_fp16,
        v_tolerance=args.v_tolerance,
        tau_cov=(args.tau_cov if args.tau_cov and args.tau_cov > 0 else None),
        k_min=args.k_min,
        k_max=args.k_max,
        ranking_fallback=args.ranking_fallback,
        ranking_r=args.ranking_r,
        ranking_fallback_mode=args.ranking_fallback_mode,
        score_consistency_check=False,
        eps_guard=args.eps_guard,
        exploration_rate=args.exploration_rate,
        rung1_threshold=args.rung1_threshold,
        rung1_multiplier=args.rung1_multiplier,
    )
    if args.phase_profile:
        adapter.certified_state.phase_timings = {}
    adapter.set_mode("certified")
    adapter.reset_runtime_metrics()

    cache_position = torch.tensor([prefix_len], dtype=torch.long, device=input_ids.device)
    print(f"[certified] warmup start steps={warmup_steps}", flush=True)
    with torch.inference_mode():
        for t in range(warmup_steps):
            token_id = input_ids[:, prefix_len + t:prefix_len + t + 1]
            out = model(
                input_ids=token_id,
                use_cache=False,
                cache_position=cache_position,
                position_ids=cache_position.unsqueeze(0),
            )
            cache_position.add_(1)
            del out
    _sync()
    adapter.reset_runtime_metrics()
    if args.phase_profile:
        adapter.certified_state.phase_timings = {}
    if args.native_profile:
        from dotcache.backends.certified_blackwell import reset_native_profile

        reset_native_profile()

    nll = torch.zeros((), dtype=torch.float32, device=input_ids.device)
    step_aggs: list[dict[str, Any]] = []
    stats_cursor = 0
    _sync()
    t0 = time.perf_counter()
    step_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    print(f"[certified] measure start steps={measure_steps}", flush=True)
    with torch.inference_mode():
        for t in range(warmup_steps, warmup_steps + measure_steps):
            ev0, ev1 = _event_pair()
            if ev0 is not None and ev1 is not None:
                ev0.record()
            token_id = input_ids[:, prefix_len + t:prefix_len + t + 1]
            out = model(
                input_ids=token_id,
                use_cache=False,
                cache_position=cache_position,
                position_ids=cache_position.unsqueeze(0),
            )
            target = input_ids[:, prefix_len + t + 1]
            nll = nll + F.cross_entropy(out.logits[:, -1, :].float(), target, reduction="sum")
            cache_position.add_(1)
            if args.collect_step_stats:
                step_agg = adapter.certified_state.aggregate_step_stats(since=stats_cursor)
                if args.fail_on_value_cache_overflow and int(step_agg.get("fp16_value_cache_overflow_step", 0) or 0) > 0:
                    raise RuntimeError(
                        "FP16 value cache overflowed in at least one layer during measured decode; "
                        "increase --fp16-value-cache-blocks or disable --fail-on-value-cache-overflow"
                    )
                step_aggs.append(step_agg)
                stats_cursor = len(adapter.certified_state.step_stats)
            if ev0 is not None and ev1 is not None:
                ev1.record()
                step_events.append((ev0, ev1))
            del out
    _sync()
    decode_s = time.perf_counter() - t0
    step_ms = [float(a.elapsed_time(b)) for a, b in step_events]
    runtime_profile = adapter.runtime_profile_summary(model_forward_ms_total=decode_s * 1000.0)
    phase_timings_us = dict(adapter.certified_state.phase_timings or {}) if args.phase_profile else {}
    phase_timings_ms = {k.removesuffix("_us") + "_ms": float(v) / 1000.0 for k, v in phase_timings_us.items()}
    native_profile = None
    if args.native_profile:
        from dotcache.backends.certified_blackwell import native_profile_summary

        native_profile = native_profile_summary()
    nll_f = float(nll.item())
    telemetry = _summarize_step_aggs(step_aggs)

    adapter.certified_state = None
    adapter.set_mode("dense")
    print(f"[certified] measure done tok_s={measure_steps / max(decode_s, 1e-9):.2f}", flush=True)
    return {
        "prefill_s": float(prefill_s),
        "cache_build_s": float(cache_build_s),
        "cache_decode_budget": int(max_new),
        "cache_runtime": cache_runtime,
        "decode_s": float(decode_s),
        "decode_steps": int(measure_steps),
        "decode_tok_s": float(measure_steps / max(decode_s, 1e-9)),
        "decode_ms_per_token": float(1000.0 * decode_s / max(measure_steps, 1)),
        "decode_step_ms": step_ms,
        "decode_step_ms_stats": _ms_stats(step_ms),
        "nll": nll_f,
        "perplexity": _ppl(nll_f, measure_steps),
        "runtime_profile": runtime_profile,
        "phase_timings_ms": phase_timings_ms,
        "native_profile": native_profile,
        "telemetry": telemetry,
    }


def _mean(vals: list[float]) -> float | None:
    return float(sum(vals) / len(vals)) if vals else None


def _summarize_step_aggs(step_aggs: list[dict[str, Any]]) -> dict[str, Any]:
    if not step_aggs:
        return {"n_steps": 0}
    n = len(step_aggs)

    def rate(key: str) -> float:
        return float(sum(1 for s in step_aggs if s.get(key)) / n)

    def total(key: str) -> int:
        return int(sum(int(s.get(key, 0) or 0) for s in step_aggs))

    fp16_hits = total("fp16_cache_hits")
    fp16_misses = total("fp16_cache_misses")
    fp16_access = fp16_hits + fp16_misses
    value_hits = total("fp16_value_cache_hits_step")
    value_misses = total("fp16_value_cache_misses_step")
    value_access = value_hits + value_misses
    total_blocks = total("total_blocks")
    skipped_blocks = total("skipped_blocks")
    h2d_total = total("h2d_total_bytes")
    h2d_key_total = total("h2d_key_bytes")
    h2d_value_total = total("h2d_value_bytes")
    k_star_vals = [float(s["k_star_mean"]) for s in step_aggs if s.get("k_star_mean") is not None]
    e_key_means = [float(s["e_key_step_mean"]) for s in step_aggs if s.get("e_key_step_mean") is not None]
    e_key_maxes = [float(s["e_key_step_max"]) for s in step_aggs if s.get("e_key_step_max") is not None]
    e_val_means = [float(s["e_val_mean"]) for s in step_aggs if s.get("e_val_mean") is not None]
    e_val_maxes = [float(s["e_val_max"]) for s in step_aggs if s.get("e_val_max") is not None]
    ranking_heads = total("ranking_heads_total")
    ranking_fallback = total("ranking_fallback_triggered")
    value_fallback_blocks = total("value_fallback_blocks")
    value_overflow_layer_steps = total("fp16_value_cache_overflow_step")
    value_overflow_decode_steps = sum(
        1 for s in step_aggs if int(s.get("fp16_value_cache_overflow_step", 0) or 0) > 0
    )
    mixedv_splitk_fallback_layer_steps = total("mixedv_splitk_fallback_step")
    mixedv_splitk_fallback_decode_steps = sum(
        1 for s in step_aggs if int(s.get("mixedv_splitk_fallback_step", 0) or 0) > 0
    )
    value_needed_vals = [
        int(s["fp16_value_cache_needed_blocks_step"])
        for s in step_aggs
        if s.get("fp16_value_cache_needed_blocks_step") is not None
    ]
    return {
        "n_steps": n,
        "int8_tail_fraction": float(skipped_blocks / total_blocks) if total_blocks else None,
        "k_star_mean": _mean(k_star_vals),
        "k_star_max": int(max((s.get("k_star_max", 0) or 0) for s in step_aggs)),
        "rung1_trigger_rate": rate("rung1_fired"),
        "rung2_trigger_rate": rate("rung2_fired"),
        "rung3_trigger_rate": rate("rung3_fired"),
        "rung4_trigger_rate": rate("rung4_fired"),
        "rung1_expansion_rate": rate("rung1_fired"),
        "boundary_check_triggers": total("boundary_check_triggered_heads_total"),
        "score_consistency_violations": total("score_consistency_violation_heads_total"),
        "ranking_consistency_fire_rate": float(ranking_fallback / ranking_heads) if ranking_heads else 0.0,
        "e_key_mean": _mean(e_key_means),
        "e_key_max": float(max(e_key_maxes)) if e_key_maxes else None,
        "e_val_mean": _mean(e_val_means),
        "e_val_max": float(max(e_val_maxes)) if e_val_maxes else None,
        "value_fallback_blocks": value_fallback_blocks,
        "h2d_bytes_per_step": float(h2d_total / n),
        "h2d_key_bytes_per_step": float(h2d_key_total / n),
        "h2d_value_bytes_per_step": float(h2d_value_total / n),
        "h2d_total_bytes": h2d_total,
        "h2d_key_total_bytes": h2d_key_total,
        "h2d_value_total_bytes": h2d_value_total,
        "fp16_cache_hit_rate": float(fp16_hits / fp16_access) if fp16_access else None,
        "fp16_cache_total_hits": fp16_hits,
        "fp16_cache_total_misses": fp16_misses,
        "fp16_value_cache_hit_rate": float(value_hits / value_access) if value_access else None,
        "fp16_value_cache_total_hits": value_hits,
        "fp16_value_cache_total_misses": value_misses,
        "fp16_value_cache_overflow_steps": value_overflow_decode_steps,
        "fp16_value_cache_overflow_rate": float(value_overflow_decode_steps / n),
        "fp16_value_cache_overflow_layer_steps": value_overflow_layer_steps,
        "fp16_value_cache_overflow_layers_per_step": float(value_overflow_layer_steps / n),
        "fp16_value_cache_needed_blocks_mean": _mean([float(v) for v in value_needed_vals]),
        "fp16_value_cache_needed_blocks_max": int(max(value_needed_vals)) if value_needed_vals else 0,
        "mixedv_splitk_fallback_steps": mixedv_splitk_fallback_decode_steps,
        "mixedv_splitk_fallback_rate": float(mixedv_splitk_fallback_decode_steps / n),
        "mixedv_splitk_fallback_layer_steps": mixedv_splitk_fallback_layer_steps,
        "mixedv_splitk_fallback_layers_per_step": float(mixedv_splitk_fallback_layer_steps / n),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
    parser.add_argument("--mode", choices=["both", "dense", "certified"], default="both",
                        help="Which decode path(s) to run. Certified-only is useful for cache sweeps.")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--eval-start", type=float, default=0.5)
    parser.add_argument("--prefix-len", type=int, default=None,
                        help="Exact prefilled context length before measured decode. Overrides --eval-start.")
    parser.add_argument("--chunk-tokens", type=int, default=None,
                        help="Number of PG-19 tokens to load. Defaults to --context.")
    parser.add_argument("--cache-decode-budget", type=int, default=None,
                        help=(
                            "max_new_tokens used when constructing the certified cache. "
                            "Defaults to warmup+measure+16; set to the full quality suffix "
                            "budget to match pg19_perplexity.py allocation."
                        ))
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--measure-steps", type=int, default=128)
    parser.add_argument("--top-k-fp16", type=int, default=4)
    parser.add_argument("--tau-cov", type=float, default=PAPER_TAU_COV)
    parser.add_argument("--k-min", type=int, default=PAPER_K_MIN)
    parser.add_argument("--k-max", type=int, default=PAPER_K_MAX)
    parser.add_argument("--ranking-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ranking-r", type=int, default=PAPER_RANKING_R)
    parser.add_argument("--ranking-fallback-mode", default="full", choices=["full", "measure"])
    parser.add_argument("--eps-guard", type=float, default=PAPER_EPS_GUARD)
    parser.add_argument("--exploration-rate", type=float, default=PAPER_EXPLORATION_RATE)
    parser.add_argument("--rung1-threshold", type=float, default=PAPER_RUNG1_THRESHOLD)
    parser.add_argument("--rung1-multiplier", type=float, default=PAPER_RUNG1_MULTIPLIER)
    parser.add_argument("--phase-profile", action="store_true",
                        help="Collect synchronized CUDA phase timings inside certified_attention_layer. Slow; profiling only.")
    parser.add_argument("--native-profile", action="store_true",
                        help="Collect native Blackwell partial-vs-reduce timings. Slow; profiling only.")
    parser.add_argument("--collect-step-stats", action="store_true",
                        help="Collect certified per-step telemetry for cache/selector/rung summaries. Slow; avoid for pure throughput.")
    parser.add_argument("--fail-on-value-cache-overflow", action="store_true",
                        help="Abort measured certified decode if bounded FP16 value scratch overflows.")
    parser.add_argument("--output", default="runs/decode_speed_compare.json")
    add_paper_cache_args(parser)
    args = parser.parse_args()
    configure_paper_runtime_defaults()
    if args.native_profile:
        os.environ["DOTCACHE_NATIVE_PROFILE"] = "1"

    if args.prefix_len is None and (args.eval_start <= 0.0 or args.eval_start >= 1.0):
        raise SystemExit("--eval-start must be in (0, 1)")
    prefix_len = int(args.prefix_len) if args.prefix_len is not None else int(args.context * args.eval_start)
    chunk_tokens = int(args.chunk_tokens) if args.chunk_tokens is not None else int(args.context)
    required = prefix_len + args.warmup_steps + args.measure_steps + 1
    if required > chunk_tokens:
        raise SystemExit(
            f"chunk too short for prefix+warmup+measure+target: need {required}, have {chunk_tokens}"
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    token = os.environ.get("HF_TOKEN") or None
    warnings.filterwarnings(
        "ignore",
        message=r"MatMul8bitLt: inputs will be cast from .* during quantization",
        category=UserWarning,
    )
    print(f"Loading {args.model} (INT8)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        dtype=torch.float16,
        token=token,
    )
    model.eval()

    from dotcache.config import DotCacheConfig
    from dotcache.integrations.llama import LlamaDotCacheModelAdapter

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    adapter = LlamaDotCacheModelAdapter(model, DotCacheConfig(head_dim=head_dim))

    chunks, book_indices = load_pg19_chunks(
        tokenizer,
        chunk_tokens,
        args.chunk_index + 1,
    )
    input_ids = chunks[args.chunk_index].unsqueeze(0).to("cuda")
    print(
        f"Compare context={args.context} prefix={prefix_len} "
        f"warmup={args.warmup_steps} measured={args.measure_steps} "
        f"backend={os.environ.get('DOTCACHE_CERTIFIED_BACKEND')}",
        flush=True,
    )

    dense = None
    certified = None
    if args.mode in ("both", "dense"):
        dense = measure_dense_decode(
            model,
            input_ids,
            prefix_len=prefix_len,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
        )
        gc.collect()
        torch.cuda.empty_cache()
    if args.mode in ("both", "certified"):
        certified = measure_certified_decode(
            model,
            adapter,
            input_ids,
            args,
            prefix_len=prefix_len,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
        )

    output = {
        "benchmark": "decode_speed_compare",
        "model": args.model,
        "mode": args.mode,
        "context_length": args.context,
        "prefix_len": prefix_len,
        "chunk_tokens": chunk_tokens,
        "chunk_index": args.chunk_index,
        "book_idx": book_indices[args.chunk_index] if book_indices else None,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "measured_token_start": prefix_len + args.warmup_steps,
        "measured_token_end_exclusive": prefix_len + args.warmup_steps + args.measure_steps,
        "cache_config": cache_config_dict(args),
        "dense": dense,
        "certified": certified,
        "certified_vs_dense_decode_speed": None,
        "quality_window": None,
    }
    if dense is not None and certified is not None:
        output["certified_vs_dense_decode_speed"] = {
            "tok_s_ratio": float(certified["decode_tok_s"] / max(dense["decode_tok_s"], 1e-9)),
            "slowdown": float(dense["decode_tok_s"] / max(certified["decode_tok_s"], 1e-9)),
            "dense_tok_s": dense["decode_tok_s"],
            "certified_tok_s": certified["decode_tok_s"],
        }
        output["quality_window"] = {
            "dense_ppl": dense["perplexity"],
            "certified_ppl": certified["perplexity"],
            "ppl_ratio": float(certified["perplexity"] / max(dense["perplexity"], 1e-9)),
            "delta_ppl": float(certified["perplexity"] - dense["perplexity"]),
        }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps({
        "dense_tok_s": dense["decode_tok_s"] if dense is not None else None,
        "certified_tok_s": certified["decode_tok_s"] if certified is not None else None,
        "certified_vs_dense_ratio": (
            output["certified_vs_dense_decode_speed"]["tok_s_ratio"]
            if output["certified_vs_dense_decode_speed"] is not None else None
        ),
        "slowdown": (
            output["certified_vs_dense_decode_speed"]["slowdown"]
            if output["certified_vs_dense_decode_speed"] is not None else None
        ),
        "dense_ppl": dense["perplexity"] if dense is not None else None,
        "certified_ppl": certified["perplexity"] if certified is not None else None,
        "ppl_ratio": output["quality_window"]["ppl_ratio"] if output["quality_window"] is not None else None,
        "phase_timings_ms": certified.get("phase_timings_ms") if certified is not None else None,
        "native_profile": certified.get("native_profile") if certified is not None else None,
        "json": str(out_path),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
