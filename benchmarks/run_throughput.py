"""Test 1: decode throughput (tok/s) at 8K context, dense vs certified variants.

Measures steady-state decode throughput on NousResearch/Meta-Llama-3.1-8B
(INT8 bitsandbytes). For each config:

  1. Build an 8K-token prefill once.
  2. Prefill → first token.
  3. Decode `decode_tokens` more tokens one at a time.
  4. Discard the first `warmup_tokens` as warmup; time the remainder with
     torch.cuda.Event timers.

Repeats the prefill+decode block `repeats` times. Reports mean ± std tok/s
plus P50/P95/P99 per-token latency, prefill time, GPU mem peak.

Configs:

  dense                  — adapter.set_mode("dense"): HF's FlashAttention path,
                           reference baseline.
  certified              — full certified path (tau_cov=0.995, k_min=2,
                           k_max=128, ranking_fallback, score_consistency_check,
                           Rung-1/2/3/4 all active, exploration_rate=0.02).
  certified-no-fallback  — certified with ranking_fallback=False,
                           score_consistency_check=False, exploration_rate=0
                           (Rung-3/4 silenced; Rung-1/2 still active).
  quantised-only         — tau_cov=None (no adaptive K*), no fallbacks;
                           Phase-1 scoring still runs for block ε certification
                           but the full rung machinery is off.
  triton-fp16            — *not implemented*. A true Phase-1 bypass would
                           require a new adapter path. Skipped with a note.

Per-repeat output JSON fields match the spec in the paper's Test 1 appendix.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def build_prefill(tokenizer, context_tokens: int) -> str:
    FILLER = (
        "The history of mathematics spans thousands of years and encompasses many "
        "different cultures and civilizations. "
    )
    question = "\nContinue:"
    ft = len(tokenizer.encode(FILLER, add_special_tokens=False))
    qt = len(tokenizer.encode(question, add_special_tokens=False))
    avail = context_tokens - qt - 50
    nb = max(avail // ft, 2)
    return FILLER * nb + question


def load_pg19_prefill_and_ref(tokenizer, context_tokens: int, ref_tokens: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the first (context_tokens + ref_tokens) tokens of the first
    long enough PG-19 test-split book. Returns (prefill, ref) where
    prefill is [1, context_tokens] and ref is [ref_tokens] (the ground-
    truth continuation used for teacher-forced decode).
    """
    from datasets import load_dataset
    ds = load_dataset("emozilla/pg19", split="test", streaming=True)
    need = context_tokens + ref_tokens
    for book in ds:
        tokens = tokenizer.encode(book["text"], add_special_tokens=False)
        if len(tokens) >= need:
            prefill = torch.tensor(tokens[:context_tokens], dtype=torch.long).unsqueeze(0)
            ref = torch.tensor(tokens[context_tokens:context_tokens + ref_tokens], dtype=torch.long)
            return prefill, ref
    raise RuntimeError(f"no PG-19 book with >= {need} tokens found")


def load_pg19_prefill(tokenizer, context_tokens: int) -> torch.Tensor:
    prefill, _ = load_pg19_prefill_and_ref(tokenizer, context_tokens, ref_tokens=0)
    return prefill


def _legacy_load_pg19_prefill(tokenizer, context_tokens: int) -> torch.Tensor:
    """Load the first context_tokens-token chunk of PG-19's test split.

    Returns a pre-tokenised [1, N] tensor ready to feed into the model.
    Matches the prompt distribution used by benchmarks/paper/pg19_perplexity.py
    — real book text, as opposed to the repetitive filler used by the
    default prompt builder.
    """
    from datasets import load_dataset
    ds = load_dataset("emozilla/pg19", split="test", streaming=True)
    for book in ds:
        text = book["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) >= context_tokens:
            return torch.tensor(tokens[:context_tokens], dtype=torch.long).unsqueeze(0)
    raise RuntimeError("no PG-19 book with >= context_tokens tokens found")


def _cert_kwargs(config: str) -> dict[str, Any]:
    """Parameters for CertifiedAttentionState by config.

    Honors env vars for the tau_cov / per-KV-group sweep:
      DOTCACHE_TAU_COV            float, override default 0.995
      DOTCACHE_PER_KV_GROUP_TOPK  '1' to enable per-KV-head group top-K
    """
    if config == "dense":
        return {}
    _tau = float(os.environ.get("DOTCACHE_TAU_COV", "0.995"))
    _kvg = os.environ.get("DOTCACHE_PER_KV_GROUP_TOPK", "0") == "1"
    base = dict(
        default_epsilon=1e-4,
        top_k_fp16_keys=4,
        tau_cov=_tau,
        k_min=2,
        k_max=128,
        rung1_threshold=0.02,
        rung1_multiplier=2.0,
        per_kv_group_topk=_kvg,
    )
    if config == "certified":
        base.update(dict(
            ranking_fallback=True,
            ranking_r=1,
            ranking_fallback_mode="full",
            score_consistency_check=True,
            eps_guard=0.01,
            exploration_rate=0.02,
        ))
    elif config == "certified-no-fallback":
        base.update(dict(
            ranking_fallback=False,
            score_consistency_check=False,
            exploration_rate=0.0,
        ))
    elif config == "quantised-only":
        base.update(dict(
            tau_cov=None,
            rung1_threshold=1.0,
            ranking_fallback=False,
            score_consistency_check=False,
            exploration_rate=0.0,
        ))
    elif config == "triton-fp16":
        raise NotImplementedError(
            "triton-fp16 requires a Phase-1-bypass adapter path that does not "
            "exist in this codebase; measure separately once implemented."
        )
    else:
        raise ValueError(f"unknown config: {config}")
    return base


def run_one_repeat(
    model, tokenizer, adapter, config: str,
    prefill_input_ids: torch.Tensor, decode_tokens: int, warmup_tokens: int,
    device: str,
    ref_tokens: torch.Tensor | None = None,
    time_only_first: int | None = None,
) -> dict[str, Any]:
    """Run one prefill + decode pass, return timing dict."""
    from llama_integration import CertifiedAttentionState, _ensure_certified_imports
    from tiered_cache import create_tiered_cache_from_model

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    seq_len = prefill_input_ids.shape[1]

    # Prefill timing: one model pass + optional tiered-cache build.
    prefill_t0 = time.perf_counter()
    adapter.set_mode("dense")
    with torch.inference_mode():
        out = model(input_ids=prefill_input_ids, use_cache=True)
    past_kv = out.past_key_values
    first_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del out
    prefill_dense_s = time.perf_counter() - prefill_t0

    if config == "dense":
        # Continue HF's dense KV-cache decode.
        cache = past_kv
        cert_state = None
    else:
        _ensure_certified_imports()
        layer_ids = list(range(model.config.num_hidden_layers))
        _env_cap = os.environ.get("DOTCACHE_FP16_CACHE_BLOCKS")
        _cap = None if _env_cap is None or _env_cap == "" else int(_env_cap)
        tiered_caches = create_tiered_cache_from_model(
            past_kv, layer_ids, fp16_key_cache_capacity=_cap,
        )
        del past_kv
        gc.collect()
        torch.cuda.empty_cache()
        cert_state = CertifiedAttentionState(
            tiered_caches=tiered_caches,
            layer_epsilons={},
            collect_stats=False,   # timed run — avoid the per-layer stat sync
            append_kv=True,
            **_cert_kwargs(config),
        )
        adapter.certified_state = cert_state
        adapter.set_mode("certified")
    prefill_total_s = time.perf_counter() - prefill_t0

    # Decode loop: per-token CUDA Event timing.
    per_tok_ms: list[float] = []
    cache_position = torch.tensor([seq_len], dtype=torch.long, device=device)
    current_input = first_token
    gen_count = 0

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    for t in range(decode_tokens):
        start_evt.record()
        with torch.inference_mode():
            if config == "dense":
                out = model(
                    input_ids=current_input, past_key_values=cache, use_cache=True,
                )
                cache = out.past_key_values
            else:
                out = model(
                    input_ids=current_input, use_cache=False,
                    cache_position=cache_position,
                    position_ids=cache_position.unsqueeze(0),
                )
        end_evt.record()
        torch.cuda.synchronize()
        per_tok_ms.append(start_evt.elapsed_time(end_evt))

        # Teacher-forced mode: feed the ref sequence's next token instead of
        # argmax. Keeps the certified attention path exercising concentrated
        # pg19 patterns the way pg19_perplexity.py's scoring loop does.
        if ref_tokens is not None and t < ref_tokens.shape[0]:
            tid = ref_tokens[t].to(device).view(1)
        else:
            tid = out.logits[:, -1, :].argmax(dim=-1)
        current_input = tid.view(1, 1)
        cache_position = cache_position + 1
        gen_count += 1

    cache_stats = {}
    if cert_state is not None:
        # Sum the FP16 cache counters across all layer caches before we
        # tear the state down. Counters accumulate independently of
        # `collect_stats`, so they're valid even in a timed (no-sync) run.
        hits = misses = bytes_ = evicts = 0
        capacity = None
        for c in cert_state.tiered_caches.values():
            hits += int(getattr(c, "_fp16_key_cache_hits", 0))
            misses += int(getattr(c, "_fp16_key_cache_misses", 0))
            bytes_ += int(getattr(c, "_fp16_key_cache_h2d_bytes", 0))
            evicts += int(getattr(c, "_fp16_key_cache_evictions", 0))
            if capacity is None:
                capacity = c.fp16_key_cache_capacity
        total_acc = hits + misses
        cache_stats = {
            "fp16_cache_capacity_blocks": capacity,
            "fp16_cache_hits": hits,
            "fp16_cache_misses": misses,
            "fp16_cache_h2d_bytes": bytes_,
            "fp16_cache_evictions": evicts,
            "fp16_cache_hit_rate": (hits / total_acc) if total_acc else 0.0,
            "fp16_cache_h2d_mb_per_decode_step": (
                (bytes_ / gen_count / (1024 ** 2)) if gen_count else 0.0
            ),
        }
        adapter.certified_state = None
        adapter.set_mode("dense")
        del cert_state

    gpu_mem_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Warmup-discarded timings. When time_only_first is set (pre-drift
    # window mode), further restrict to tokens [warmup, time_only_first).
    if time_only_first is not None:
        timed = per_tok_ms[warmup_tokens:time_only_first]
    else:
        timed = per_tok_ms[warmup_tokens:]
    total_ms = sum(timed)
    n = len(timed)
    tok_per_sec = 1000.0 * n / total_ms if total_ms > 0 else 0.0
    timed_sorted = sorted(timed)

    def pct(p):
        if not timed_sorted:
            return 0.0
        return timed_sorted[min(n - 1, int(p * (n - 1)))]

    return {
        "config": config,
        "tok_per_sec": tok_per_sec,
        "ms_per_token_mean": total_ms / n if n else 0.0,
        "ms_per_token_median": pct(0.5),
        "ms_per_token_p50": pct(0.5),
        "ms_per_token_p95": pct(0.95),
        "ms_per_token_p99": pct(0.99),
        "ms_per_token_min": min(timed) if timed else 0.0,
        "ms_per_token_max": max(timed) if timed else 0.0,
        "prefill_time_ms": prefill_total_s * 1000.0,
        "prefill_time_dense_ms": prefill_dense_s * 1000.0,
        "decode_tokens_measured": n,
        "decode_tokens_warmup": warmup_tokens,
        "gpu_mem_peak_mb": gpu_mem_peak_mb,
        "seq_len": seq_len,
        **cache_stats,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
    ap.add_argument("--context-length", type=int, default=8192)
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--warmup-tokens", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--configs", nargs="+",
                    default=["dense", "certified", "certified-no-fallback", "quantised-only"])
    ap.add_argument("--output", default="benchmarks/results/perf_tests_20260422/test1_throughput.json")
    ap.add_argument("--prompt-source", choices=["filler", "pg19"], default="filler",
                    help="'filler' = repetitive history-of-mathematics text (scattered attention), "
                         "'pg19' = first PG-19 test-split book (concentrated attention).")
    ap.add_argument("--teacher-forced", action="store_true",
                    help="Feed ground-truth pg19 tokens as the next input each step "
                         "(requires --prompt-source pg19). Exercises the same "
                         "concentrated-attention pattern as pg19_perplexity.py.")
    ap.add_argument("--time-only-first", type=int, default=None,
                    help="Only time the first N decode tokens (after warmup). Used "
                         "for the pre-drift window sweep after the drift knee is "
                         "identified by the per-token cache trace.")
    args = ap.parse_args()

    os.environ.setdefault("DOTCACHE_V_TOL", "0.05")

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from llama_integration import LlamaDotCacheModelAdapter
    from config import DotCacheConfig

    token = os.environ.get("HF_TOKEN") or None
    print(f"Loading {args.model} (INT8)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    quant = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, device_map="auto", token=token,
    )
    model.eval()
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    cfg = DotCacheConfig(head_dim=head_dim)
    adapter = LlamaDotCacheModelAdapter(model, cfg)
    device = next(model.parameters()).device

    ref_tokens = None
    if args.teacher_forced:
        if args.prompt_source != "pg19":
            raise SystemExit("--teacher-forced requires --prompt-source pg19")
        print("Loading PG-19 prefill + reference continuation for teacher-forced decode…")
        pg19_ids, ref_tokens = load_pg19_prefill_and_ref(
            tokenizer, args.context_length, ref_tokens=args.decode_tokens + 4,
        )
        pg19_ids = pg19_ids.to(device)
        ref_tokens = ref_tokens.to(device)
        ids = {"input_ids": pg19_ids}
    elif args.prompt_source == "pg19":
        print("Loading PG-19 prefill…")
        pg19_ids = load_pg19_prefill(tokenizer, args.context_length).to(device)
        ids = {"input_ids": pg19_ids}
    else:
        prompt = build_prefill(tokenizer, args.context_length)
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=args.context_length).to(device)
    seq_len = ids["input_ids"].shape[1]
    mode_tag = "pg19+teacher-forced" if args.teacher_forced else args.prompt_source
    print(f"Prefill seq_len = {seq_len}  ({mode_tag})")
    print(f"Configs: {args.configs}  repeats={args.repeats}  decode={args.decode_tokens}  warmup={args.warmup_tokens}")

    # Warm the loader with one throwaway prefill.
    adapter.set_mode("dense")
    with torch.inference_mode():
        _ = model(**ids, use_cache=True)
    torch.cuda.empty_cache()
    gc.collect()

    per_config: dict[str, list[dict]] = {}
    for config in args.configs:
        per_config[config] = []
        print(f"\n=== config={config} ===")
        for r in range(args.repeats):
            try:
                rep = run_one_repeat(
                    model, tokenizer, adapter, config,
                    prefill_input_ids=ids["input_ids"],
                    decode_tokens=args.decode_tokens,
                    warmup_tokens=args.warmup_tokens,
                    device=str(device),
                    ref_tokens=ref_tokens,
                    time_only_first=args.time_only_first,
                )
                per_config[config].append(rep)
                print(f"  [{r+1}/{args.repeats}] tok/s={rep['tok_per_sec']:.2f}  "
                      f"p50={rep['ms_per_token_p50']:.2f}ms  p95={rep['ms_per_token_p95']:.2f}ms  "
                      f"prefill={rep['prefill_time_ms']:.1f}ms  gpu_peak={rep['gpu_mem_peak_mb']:.0f}MB")
            except NotImplementedError as e:
                print(f"  skip: {e}")
                per_config[config].append({"skipped": True, "reason": str(e)})
                break
            gc.collect()
            torch.cuda.empty_cache()

    # Aggregate per config.
    summary: dict[str, dict] = {}
    for config, reps in per_config.items():
        good = [r for r in reps if "skipped" not in r]
        if not good:
            summary[config] = {"skipped": True}
            continue
        tps = [r["tok_per_sec"] for r in good]
        p50s = [r["ms_per_token_p50"] for r in good]
        p95s = [r["ms_per_token_p95"] for r in good]
        p99s = [r["ms_per_token_p99"] for r in good]
        pre = [r["prefill_time_ms"] for r in good]
        gpu = [r["gpu_mem_peak_mb"] for r in good]
        summary[config] = {
            "n_repeats": len(good),
            "tok_per_sec_mean": statistics.mean(tps),
            "tok_per_sec_std": statistics.stdev(tps) if len(tps) > 1 else 0.0,
            "tok_per_sec_min": min(tps),
            "tok_per_sec_max": max(tps),
            "ms_per_token_p50_median": statistics.median(p50s),
            "ms_per_token_p95_median": statistics.median(p95s),
            "ms_per_token_p99_median": statistics.median(p99s),
            "prefill_time_ms_median": statistics.median(pre),
            "gpu_mem_peak_mb_max": max(gpu),
        }

    # Derived decomposition vs dense.
    if "dense" in summary and not summary["dense"].get("skipped"):
        d = summary["dense"]["tok_per_sec_mean"]
        for k, v in summary.items():
            if k == "dense" or v.get("skipped"): continue
            v["overhead_vs_dense_pct"] = (d / v["tok_per_sec_mean"] - 1.0) * 100.0

    payload = {
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
        "model": args.model,
        "context_length": args.context_length,
        "decode_tokens": args.decode_tokens,
        "warmup_tokens": args.warmup_tokens,
        "repeats": args.repeats,
        "seq_len": seq_len,
        "per_config": per_config,
        "summary": summary,
    }
    p = Path(args.output); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {p}")

    print("\n=== Summary ===")
    print(f"{'config':<24} {'tok/s mean':>12} {'± std':>8} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8} {'vs dense':>10}")
    for k in args.configs:
        s = summary.get(k, {})
        if s.get("skipped"):
            print(f"{k:<24} SKIPPED")
            continue
        oh = s.get("overhead_vs_dense_pct", float("nan"))
        oh_str = f"{oh:+.1f}%" if oh == oh else "—"
        print(f"{k:<24} {s['tok_per_sec_mean']:>12.2f} {s.get('tok_per_sec_std',0):>8.2f} "
              f"{s['ms_per_token_p50_median']:>8.2f} {s['ms_per_token_p95_median']:>8.2f} "
              f"{s['ms_per_token_p99_median']:>8.2f} {oh_str:>10}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
