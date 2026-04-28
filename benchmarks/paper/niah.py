"""Needle-in-a-Haystack (NIAH) benchmark.

Plants a fact at varying (depth, context_length) positions, generates text,
checks retrieval accuracy. Produces the classic depth×length heatmap.

Run for both dense (FP16 attention, INT8 weights) and certified (full pipeline).
Any cell where dense succeeds but certified fails = critical bug.
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

# Needles: distinct facts with unique answers
NEEDLES = [
    ("The secret project codename is Crimson Falcon.", "Crimson Falcon"),
    ("The emergency access code is 7429-DELTA.", "7429-DELTA"),
    ("The primary backup server is located in Zurich.", "Zurich"),
    ("The annual budget for the initiative is $4.7 million.", "4.7 million"),
    ("The lead researcher's employee ID is RX-9031.", "RX-9031"),
    ("The field team leader's callsign is Viper-Seven.", "Viper-Seven"),
    ("The backup data vault is labelled Hotel-Echo-4.", "Hotel-Echo-4"),
    ("The encryption key rotation interval is 47 minutes.", "47 minutes"),
    ("The classified document's tracking ID is KGB-9982.", "KGB-9982"),
    ("The compliance audit contact is Dr. Nakamura.", "Dr. Nakamura"),
]

# Filler: repeated innocuous text (avoids any content that could match needles)
FILLER_BLOCK = (
    "The history of mathematics spans thousands of years and encompasses many "
    "different cultures and civilizations. From the earliest counting systems "
    "developed by ancient peoples to the sophisticated abstract algebras of the "
    "modern era, mathematical knowledge has grown through a process of discovery, "
    "invention, and refinement. The Babylonians developed a base-60 number system "
    "that still influences how we measure time and angles today. The ancient Greeks "
    "made foundational contributions to geometry, number theory, and logic. During "
    "the Islamic Golden Age, scholars preserved and extended Greek mathematics while "
    "making original advances in algebra and trigonometry. The Renaissance saw a "
    "flowering of mathematical activity in Europe, leading eventually to the "
    "development of calculus by Newton and Leibniz in the seventeenth century. "
    "In the nineteenth and twentieth centuries, mathematics became increasingly "
    "abstract and specialized, with new fields emerging at a rapid pace. Today, "
    "mathematics is essential to science, engineering, economics, and many other "
    "domains of human activity.\n\n"
)


def build_niah_prompt(needle_text: str, context_tokens: int, depth_fraction: float,
                      tokenizer) -> str:
    """Build a NIAH prompt with needle at specified depth.

    Args:
        needle_text: the fact to plant
        context_tokens: total context length in tokens
        depth_fraction: 0.0=start, 0.5=middle, 0.9=near end
        tokenizer: HF tokenizer for length estimation

    Returns:
        prompt string with needle embedded at depth
    """
    # Build filler to approximate target length
    # Estimate tokens per filler block
    filler_tokens = len(tokenizer.encode(FILLER_BLOCK, add_special_tokens=False))
    needle_tokens = len(tokenizer.encode(needle_text, add_special_tokens=False))

    # Reserve tokens for needle + question + some margin
    question = "\n\nBased on the above text, what is the answer to: What is the secret, code, location, budget, or ID mentioned in the special statement?\nAnswer:"
    question_tokens = len(tokenizer.encode(question, add_special_tokens=False))
    available = context_tokens - needle_tokens - question_tokens - 50  # margin

    num_blocks = max(available // filler_tokens, 2)
    needle_position = max(0, int(num_blocks * depth_fraction))

    parts = []
    for i in range(num_blocks):
        if i == needle_position:
            parts.append(f"\n[IMPORTANT] {needle_text}\n\n")
        parts.append(FILLER_BLOCK)

    prompt = "".join(parts) + question
    return prompt


def check_retrieval(generated_text: str, expected_answer: str) -> bool:
    """Check if the generated text contains the expected answer."""
    return expected_answer.lower() in generated_text.lower()


def _binomial_two_sided_p_value(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value for McNemar's discordant pairs.

    For McNemar, only discordant pairs matter. Under the null, dense-only and
    certified-only wins are equally likely, so this is Binomial(n=b+c, p=0.5).
    """
    if n <= 0:
        return 1.0
    k = max(0, min(int(k), int(n)))
    observed = math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))
    prob = 0.0
    for i in range(n + 1):
        pi = math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
        if pi <= observed + 1e-15:
            prob += pi
    return float(min(1.0, prob))


def paired_niah_stats(
    dense_results: list[dict],
    certified_results: list[dict],
    *,
    bootstrap_iters: int = 10_000,
    seed: int = 20260425,
) -> dict:
    """Paired NIAH delta, bootstrap CI, and exact McNemar test.

    Returns statistics for Certified - Dense accuracy. Pairing key is the
    benchmark unit: (target_context, depth, needle_idx).
    """
    dense_map = {
        (r["target_context"], r["depth"], r["needle_idx"]): bool(r["correct"])
        for r in dense_results
    }
    pairs: list[tuple[bool, bool]] = []
    for r in certified_results:
        key = (r["target_context"], r["depth"], r["needle_idx"])
        if key in dense_map:
            pairs.append((dense_map[key], bool(r["correct"])))

    n = len(pairs)
    if n == 0:
        return {
            "n": 0,
            "dense_accuracy": None,
            "certified_accuracy": None,
            "delta_accuracy": None,
            "delta_pp": None,
            "bootstrap_ci_lo": None,
            "bootstrap_ci_hi": None,
            "bootstrap_ci_pp_lo": None,
            "bootstrap_ci_pp_hi": None,
            "bootstrap_iters": int(bootstrap_iters),
            "mcnemar_p": None,
            "paired_table": {"both_correct": 0, "dense_only": 0, "certified_only": 0, "both_wrong": 0},
        }

    both_correct = sum(1 for d, c in pairs if d and c)
    dense_only = sum(1 for d, c in pairs if d and not c)
    certified_only = sum(1 for d, c in pairs if (not d) and c)
    both_wrong = sum(1 for d, c in pairs if (not d) and (not c))
    dense_correct = both_correct + dense_only
    cert_correct = both_correct + certified_only
    delta = (certified_only - dense_only) / n

    diffs = np.asarray([(1 if c else 0) - (1 if d else 0) for d, c in pairs], dtype=np.float64)
    if n > 1 and bootstrap_iters > 0:
        rng = np.random.default_rng(seed)
        boot = diffs[rng.integers(0, n, size=(int(bootstrap_iters), n))].mean(axis=1)
        ci_lo = float(np.quantile(boot, 0.025))
        ci_hi = float(np.quantile(boot, 0.975))
    else:
        ci_lo = ci_hi = float(delta)

    discordant = dense_only + certified_only
    p_value = _binomial_two_sided_p_value(min(dense_only, certified_only), discordant)
    return {
        "n": int(n),
        "dense_correct": int(dense_correct),
        "certified_correct": int(cert_correct),
        "dense_accuracy": dense_correct / n,
        "certified_accuracy": cert_correct / n,
        "delta_accuracy": float(delta),
        "delta_pp": float(delta * 100.0),
        "bootstrap_ci_lo": ci_lo,
        "bootstrap_ci_hi": ci_hi,
        "bootstrap_ci_pp_lo": float(ci_lo * 100.0),
        "bootstrap_ci_pp_hi": float(ci_hi * 100.0),
        "bootstrap_iters": int(bootstrap_iters),
        "bootstrap_seed": int(seed),
        "mcnemar_p": p_value,
        "mcnemar": {
            "test": "exact_two_sided_binomial",
            "p_value": p_value,
            "discordant": int(discordant),
            "dense_only": int(dense_only),
            "certified_only": int(certified_only),
        },
        "paired_table": {
            "both_correct": int(both_correct),
            "dense_only": int(dense_only),
            "certified_only": int(certified_only),
            "both_wrong": int(both_wrong),
        },
    }


def paired_niah_stats_by_context(
    dense_results: list[dict],
    certified_results: list[dict],
    *,
    bootstrap_iters: int = 10_000,
    seed: int = 20260425,
) -> dict[str, dict]:
    contexts = sorted({int(r["target_context"]) for r in dense_results + certified_results})
    out: dict[str, dict] = {}
    for idx, ctx in enumerate(contexts):
        dense_ctx = [r for r in dense_results if int(r["target_context"]) == ctx]
        cert_ctx = [r for r in certified_results if int(r["target_context"]) == ctx]
        out[f"{ctx // 1024}K"] = paired_niah_stats(
            dense_ctx,
            cert_ctx,
            bootstrap_iters=bootstrap_iters,
            seed=seed + idx,
        )
    return out


def paired_niah_stats_by_needle_group(
    dense_results: list[dict],
    certified_results: list[dict],
    *,
    bootstrap_iters: int = 10_000,
    seed: int = 20260425,
) -> dict[str, dict]:
    """Stats for the paper's original-vs-harder 8K NIAH follow-up.

    Needles 0-4 are the original five; needles 5+ are the harder follow-up
    set. Groups with no rows are omitted so smaller 30-trial cells stay clean.
    """
    groups = {
        "original": lambda r: int(r["needle_idx"]) < 5,
        "harder": lambda r: int(r["needle_idx"]) >= 5,
    }
    out: dict[str, dict] = {}
    for idx, (name, pred) in enumerate(groups.items()):
        dense_group = [r for r in dense_results if pred(r)]
        cert_group = [r for r in certified_results if pred(r)]
        if dense_group or cert_group:
            out[name] = paired_niah_stats(
                dense_group,
                cert_group,
                bootstrap_iters=bootstrap_iters,
                seed=seed + idx,
            )
    return out


def run_niah_cell(  # noqa: C901  # large signature is the consequence of paper-§7 plumbing
    model, tokenizer, adapter, mode: str,
    context_tokens: int, depth: float, needle_idx: int,
    *,
    v_tolerance: float,
    max_new_tokens: int = 50,
    device: str = "cuda",
    top_k_fp16_keys: int = 4,
    use_int4_values: bool = False,
    group_size: int = 16,
    fp16_key_cache_blocks: int | str | None = None,
    fp16_value_cache_blocks: int | str | None = None,
    ranking_fallback: bool = True,
    ranking_r: int = PAPER_RANKING_R,
    ranking_fallback_mode: str = "full",
    tau_cov: float | None = PAPER_TAU_COV,
    k_min: int = PAPER_K_MIN,
    k_max: int | None = PAPER_K_MAX,
    rung1_threshold: float = PAPER_RUNG1_THRESHOLD,
    rung1_multiplier: float = PAPER_RUNG1_MULTIPLIER,
    score_consistency_check: bool = False,
    score_consistency_interval: int = 1,
    eps_guard: float = PAPER_EPS_GUARD,
    exploration_rate: float = PAPER_EXPLORATION_RATE,
    telemetry_collector=None,
) -> dict:
    """Run one NIAH cell: plant needle, generate, check retrieval.

    Args:
        mode: 'dense' or 'certified'
    """
    needle_text, expected_answer = NEEDLES[needle_idx]
    prompt = build_niah_prompt(needle_text, context_tokens, depth, tokenizer)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=context_tokens).to(device)
    seq_len = inputs["input_ids"].shape[1]

    cell_agg: dict = {}

    if mode == "dense":
        # Dense: standard HF generation
        adapter.set_mode("dense")
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        generated_ids = outputs[0, seq_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    elif mode == "certified":
        # Certified: prefill dense, then decode certified
        from dotcache.integrations.llama import _ensure_certified_imports, CertifiedAttentionState
        from dotcache.kernels.tiered_kv_cache import (
            create_tiered_cache_from_model,
            create_tiered_cache_int4v_from_model,
        )

        adapter.set_mode("dense")
        with torch.inference_mode():
            outputs = model(**inputs, use_cache=True)
        past_kv = outputs.past_key_values
        first_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        del outputs

        # Build tiered cache. Value fallback defaults to bounded scratch so
        # paper runs do not allocate a persistent full FP16 value mirror.
        _ensure_certified_imports()
        layer_ids = list(range(model.config.num_hidden_layers))
        _env_key_cap = os.environ.get("DOTCACHE_FP16_CACHE_BLOCKS")
        _env_value_cap = os.environ.get("DOTCACHE_FP16_VALUE_CACHE_BLOCKS")
        from _provenance import resolve_fp16_key_cache_blocks, resolve_fp16_value_cache_blocks
        _key_cap = resolve_fp16_key_cache_blocks(fp16_key_cache_blocks, _env_key_cap)
        _value_cap = resolve_fp16_value_cache_blocks(fp16_value_cache_blocks, _env_value_cap)
        if use_int4_values:
            tiered_caches = create_tiered_cache_int4v_from_model(
                past_kv, layer_ids, group_size=group_size,
                max_new_tokens=max_new_tokens + 8,
                fp16_key_cache_capacity=_key_cap,
                fp16_value_cache_capacity=_value_cap,
            )
        else:
            tiered_caches = create_tiered_cache_from_model(
                past_kv, layer_ids, max_new_tokens=max_new_tokens + 8,
                fp16_key_cache_capacity=None,
            )
        del past_kv
        gc.collect()
        torch.cuda.empty_cache()

        # Enable stats whenever a diagnostic feature is on (ranking fallback,
        # adaptive K*, score-consistency, or exploration) so the aggregator
        # has data to report; otherwise keep stats off to match the previous
        # timed-run behaviour exactly.
        collect_stats = (
            bool(ranking_fallback)
            or (tau_cov is not None and tau_cov > 0)
            or bool(score_consistency_check)
            or (exploration_rate and exploration_rate > 0)
        )
        adapter.certified_state = CertifiedAttentionState(
            tiered_caches=tiered_caches,
            collect_stats=collect_stats,
            append_kv=True,  # Append new K/V tokens during decode
            top_k_fp16_keys=top_k_fp16_keys,
            v_tolerance=v_tolerance,
            ranking_fallback=ranking_fallback,
            ranking_r=ranking_r,
            ranking_fallback_mode=ranking_fallback_mode,
            tau_cov=tau_cov,
            k_min=k_min,
            k_max=k_max,
            rung1_threshold=rung1_threshold,
            rung1_multiplier=rung1_multiplier,
            score_consistency_check=score_consistency_check,
            score_consistency_interval=score_consistency_interval,
            eps_guard=eps_guard,
            exploration_rate=exploration_rate,
        )
        adapter.set_mode("certified")

        # Generate tokens
        cache_position = torch.tensor([seq_len], dtype=torch.long, device=device)
        current_input = first_token
        gen_token_tensors = [first_token]

        for _ in range(max(0, max_new_tokens - 1)):
            with torch.inference_mode():
                out = model(
                    input_ids=current_input, use_cache=False,
                    cache_position=cache_position,
                    position_ids=cache_position.unsqueeze(0),
                )
            if telemetry_collector is not None:
                telemetry_collector.record_step()
            tid = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_token_tensors.append(tid)
            current_input = tid
            cache_position.add_(1)

        gen_ids = torch.cat(gen_token_tensors, dim=1)[0].tolist()
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in gen_ids:
            gen_ids = gen_ids[:gen_ids.index(tokenizer.eos_token_id) + 1]
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        cell_agg = adapter.certified_state.aggregate_step_stats() if collect_stats else {}
        adapter.certified_state = None
        adapter.set_mode("dense")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    correct = check_retrieval(generated_text, expected_answer)

    gc.collect()
    torch.cuda.empty_cache()

    result = {
        "context_tokens": seq_len,
        "target_context": context_tokens,
        "depth": depth,
        "needle_idx": needle_idx,
        "mode": mode,
        "correct": correct,
        "expected": expected_answer,
        "generated": generated_text[:200],
    }
    # Surface Rung-3 ranking-fallback counters (if collected) so the sweep
    # can sum them across cells. Keys are absent on dense runs and on certified
    # runs with ranking_fallback disabled.
    for key in (
        "ranking_heads_total", "ranking_disagree_r1", "ranking_disagree_r3",
        "ranking_fallback_triggered",
    ):
        if key in cell_agg:
            result[key] = int(cell_agg[key])
    # Paper §8.6 hard-STOP triggers + §4.5 bound scalars. Previously
    # dropped (only ranking_* fields were persisted). See
    # docs/paper_v1_run_handoff.md §5.
    for key in (
        "score_consistency_violation_heads_total",
        "rung1_fired", "rung2_fired", "rung3_fired", "rung4_fired",
        "rung1_fired_layers", "rung2_fired_layers",
        "rung3_fired_layers", "rung4_fired_layers",
        "boundary_check_fired", "boundary_check_fired_layers",
        "boundary_check_triggered_heads_total",
        "e_key_step_mean", "e_key_step_max", "v_max_global",
        "e_val_max", "e_val_mean", "e_val_pre_rung2_max",
        "e_val_pre_rung2_mean", "value_error_mode",
        "value_fallback_blocks", "value_fallback_head_blocks",
        "delta_bound_step_mean",
        "tail_mass_int8_est_step_mean", "tail_mass_int8_est_step_max",
        "k_star_mean", "k_star_max",
        "h2d_key_bytes", "h2d_value_bytes", "h2d_total_bytes",
        "vram_fp16_key_cache_bytes", "vram_fp16_value_cache_bytes",
        "fp16_value_cache_hits_step", "fp16_value_cache_misses_step",
        "fp16_value_cache_evictions_step", "fp16_value_cache_needed_blocks_step",
        "fp16_value_cache_overflow_step",
    ):
        if key in cell_agg:
            result[key] = cell_agg[key]
    return result


def run_niah_sweep(
    model, tokenizer, adapter,
    context_lengths: list[int],
    *,
    v_tolerance: float,
    depths: list[float] = None,
    num_needles: int = 5,
    device: str = "cuda",
    top_k_fp16_keys: int = 4,
    use_int4_values: bool = False,
    group_size: int = 16,
    fp16_key_cache_blocks: int | str | None = None,
    fp16_value_cache_blocks: int | str | None = None,
    ranking_fallback: bool = True,
    ranking_r: int = PAPER_RANKING_R,
    ranking_fallback_mode: str = "full",
    tau_cov: float | None = PAPER_TAU_COV,
    k_min: int = PAPER_K_MIN,
    k_max: int | None = PAPER_K_MAX,
    rung1_threshold: float = PAPER_RUNG1_THRESHOLD,
    rung1_multiplier: float = PAPER_RUNG1_MULTIPLIER,
    score_consistency_check: bool = False,
    score_consistency_interval: int = 1,
    eps_guard: float = PAPER_EPS_GUARD,
    exploration_rate: float = PAPER_EXPLORATION_RATE,
    telemetry_collector=None,
    trial_start: int = 0,
    trial_count: int | None = None,
) -> dict:
    """Run full NIAH sweep across depths and context lengths."""
    if depths is None:
        depths = [i / 10 for i in range(10)]  # 0.0, 0.1, ..., 0.9

    results = {"dense": [], "certified": []}
    all_trials = [
        (depth, needle_idx)
        for depth in depths
        for needle_idx in range(num_needles)
    ]
    start = max(0, int(trial_start))
    end = len(all_trials) if trial_count is None else min(len(all_trials), start + max(0, int(trial_count)))
    trials = all_trials[start:end]
    total = len(context_lengths) * len(trials) * 2
    done = 0

    for ctx_len in context_lengths:
        for local_trial_idx, (depth, needle_idx) in enumerate(trials):
            trial_idx = start + local_trial_idx
            for mode in ["dense", "certified"]:
                done += 1
                r = run_niah_cell(
                    model, tokenizer, adapter, mode,
                    ctx_len, depth, needle_idx,
                    device=device,
                    top_k_fp16_keys=top_k_fp16_keys,
                    v_tolerance=v_tolerance,
                    use_int4_values=use_int4_values,
                    group_size=group_size,
                    fp16_key_cache_blocks=fp16_key_cache_blocks,
                    fp16_value_cache_blocks=fp16_value_cache_blocks,
                    ranking_fallback=ranking_fallback,
                    ranking_r=ranking_r,
                    ranking_fallback_mode=ranking_fallback_mode,
                    tau_cov=tau_cov,
                    k_min=k_min,
                    k_max=k_max,
                    rung1_threshold=rung1_threshold,
                    rung1_multiplier=rung1_multiplier,
                    score_consistency_check=score_consistency_check,
                    score_consistency_interval=score_consistency_interval,
                    eps_guard=eps_guard,
                    exploration_rate=exploration_rate,
                    telemetry_collector=telemetry_collector if mode == "certified" else None,
                )
                r["trial_idx"] = trial_idx
                results[mode].append(r)

                status = "OK" if r["correct"] else "FAIL"
                print(f"  [{done}/{total}] {mode:>10} {ctx_len//1024}K d={depth:.1f} "
                      f"n={needle_idx} -> {status}")

                if not r["correct"] and mode == "certified":
                    # Check if dense also failed
                    dense_r = results["dense"][-1] if results["dense"] else None
                    if dense_r and dense_r["correct"]:
                        print(f"    *** CRITICAL: dense OK but certified FAILED ***")

    # Compute accuracy heatmaps
    heatmaps = {}
    for mode in ["dense", "certified"]:
        heatmap = {}
        for r in results[mode]:
            key = (r["target_context"], r["depth"])
            if key not in heatmap:
                heatmap[key] = {"correct": 0, "total": 0}
            heatmap[key]["total"] += 1
            if r["correct"]:
                heatmap[key]["correct"] += 1
        heatmaps[mode] = {
            f"{k[0]//1024}K_d{k[1]:.1f}": v["correct"] / v["total"]
            for k, v in heatmap.items()
        }

    # Summary
    for mode in ["dense", "certified"]:
        total_correct = sum(1 for r in results[mode] if r["correct"])
        total_count = len(results[mode])
        print(f"\n{mode}: {total_correct}/{total_count} correct ({total_correct/total_count:.1%})")

    # Critical failures: certified fails where dense succeeds
    critical = []
    dense_map = {(r["target_context"], r["depth"], r["needle_idx"]): r
                 for r in results["dense"]}
    for r in results["certified"]:
        key = (r["target_context"], r["depth"], r["needle_idx"])
        if key in dense_map and dense_map[key]["correct"] and not r["correct"]:
            critical.append(r)

    if critical:
        print(f"\n*** {len(critical)} CRITICAL FAILURES: dense OK but certified FAILED ***")
        for c in critical[:5]:
            print(f"  {c['target_context']//1024}K d={c['depth']:.1f} n={c['needle_idx']}: "
                  f"expected '{c['expected']}', got '{c['generated'][:50]}'")

    sweep_result = {
        "results": results,
        "heatmaps": heatmaps,
        "critical_failures": len(critical),
        "dense_accuracy": sum(1 for r in results["dense"] if r["correct"]) / max(len(results["dense"]), 1),
        "certified_accuracy": sum(1 for r in results["certified"] if r["correct"]) / max(len(results["certified"]), 1),
    }
    sweep_result["paired_stats"] = paired_niah_stats(results["dense"], results["certified"])
    sweep_result["paired_stats_by_context"] = paired_niah_stats_by_context(
        results["dense"], results["certified"]
    )
    sweep_result["paired_stats_by_needle_group"] = paired_niah_stats_by_needle_group(
        results["dense"], results["certified"]
    )
    if ranking_fallback:
        heads_total = sum(r.get("ranking_heads_total", 0) for r in results["certified"])
        disagree_r1 = sum(r.get("ranking_disagree_r1", 0) for r in results["certified"])
        disagree_r3 = sum(r.get("ranking_disagree_r3", 0) for r in results["certified"])
        triggered = sum(r.get("ranking_fallback_triggered", 0) for r in results["certified"])
        sweep_result["ranking_fallback_summary"] = {
            "mode": ranking_fallback_mode,
            "r": int(ranking_r),
            "heads_total": int(heads_total),
            "disagree_r1": int(disagree_r1),
            "disagree_r3": int(disagree_r3),
            "triggered": int(triggered),
            "disagree_rate_r1": (disagree_r1 / heads_total) if heads_total else 0.0,
            "disagree_rate_r3": (disagree_r3 / heads_total) if heads_total else 0.0,
            "fallback_rate": (triggered / heads_total) if heads_total else 0.0,
        }
    return sweep_result


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
    parser.add_argument("--contexts", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--depths", type=float, nargs="+", default=None,
                        help="Needle depths to run. Default is the paper sweep 0.0,0.1,...,0.9.")
    parser.add_argument("--needles", type=int, default=3)
    parser.add_argument("--trial-start", type=int, default=0,
                        help="Start paired trial index after flattening depth-major depths x needles.")
    parser.add_argument("--trial-count", type=int, default=None,
                        help="Number of paired trials to run from --trial-start.")
    parser.add_argument("--trial-index", type=int, default=None,
                        help="Alias for --trial-start with --trial-count 1.")
    parser.add_argument("--output", default="benchmarks/results/niah.json")
    parser.add_argument("--top-k-fp16", type=int, default=4,
                        help="Top-K blocks use FP16 keys (999=all FP16, 0=all INT8)")
    parser.add_argument("--ranking-fallback", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Rung-3 ranking-consistency fallback (paper default: enabled)")
    parser.add_argument("--ranking-r", type=int, default=PAPER_RANKING_R,
                        help=f"Top-r positions that must agree between INT8 and FP16 rankings (default: {PAPER_RANKING_R})")
    parser.add_argument("--ranking-fallback-mode", default="full", choices=["full", "measure"],
                        help="'full' = per-head dense FP16 recompute on disagreement (Option A); 'measure' = detect only, no action")
    # Paper §3.3 adaptive K* selector. Defaults match the paper's experimental setup.
    parser.add_argument("--tau-cov", type=float, default=PAPER_TAU_COV,
                        help="Adaptive K*: minimum cumulative INT8-estimated mass per head (0=disable, use fixed top-k-fp16 floor)")
    parser.add_argument("--k-min", type=int, default=PAPER_K_MIN,
                        help=f"Adaptive K* lower clamp (default {PAPER_K_MIN})")
    parser.add_argument("--k-max", type=int, default=PAPER_K_MAX,
                        help=f"Adaptive K* upper clamp (paper default {PAPER_K_MAX})")
    parser.add_argument("--rung1-threshold", type=float, default=PAPER_RUNG1_THRESHOLD,
                        help=f"Rung 1 (expand K*): tail-mass above which K* is expanded for that head (default {PAPER_RUNG1_THRESHOLD}). Set 1.0 to disable.")
    parser.add_argument("--rung1-multiplier", type=float, default=PAPER_RUNG1_MULTIPLIER,
                        help=f"Rung 1 (expand K*): k_max multiplier on trigger (default {PAPER_RUNG1_MULTIPLIER})")
    parser.add_argument("--score-consistency-check", action="store_true",
                        help="Paper §6 defence-in-depth: compare FP16 vs INT8 block scores on promoted blocks; expected 0 violations")
    parser.add_argument("--score-consistency-interval", type=int, default=1,
                        help="Run score-consistency canary every N decode steps (1 = exact every step).")
    parser.add_argument("--eps-guard", type=float, default=PAPER_EPS_GUARD,
                        help=f"Score-consistency tolerance above the theoretical Delta bound (default {PAPER_EPS_GUARD})")
    parser.add_argument("--exploration-rate", type=float, default=PAPER_EXPLORATION_RATE,
                        help=f"Paper exploration budget: per-step fraction of non-top-K* blocks promoted to FP16 for monitoring (default {PAPER_EXPLORATION_RATE})")
    parser.add_argument("--pagein-telemetry", action="store_true",
                        help="Collect per-step page-in / rung / VRAM-cache telemetry during certified decode (Test 3)")
    parser.add_argument("--telemetry-output", default=None,
                        help="Path to write per-step telemetry JSON (default: <output>.pagein.json)")
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from _provenance import (
        add_paper_cache_args,
        cache_config_dict,
        configure_paper_runtime_defaults,
    )
    add_paper_cache_args(parser)
    args = parser.parse_args()
    if args.trial_index is not None:
        args.trial_start = int(args.trial_index)
        args.trial_count = 1
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

    rf_tag = "off"
    if args.ranking_fallback:
        rf_tag = f"{args.ranking_fallback_mode}(r={args.ranking_r})"
    tau_cov = args.tau_cov if args.tau_cov and args.tau_cov > 0 else None
    adaptive_tag = f"tau_cov={tau_cov} k=[{args.k_min},{args.k_max}]" if tau_cov else "fixed"
    print(f"\nNIAH: contexts={[c//1024 for c in args.contexts]}K, needles={args.needles}, top_k_fp16={args.top_k_fp16}, adaptive={adaptive_tag}, ranking_fallback={rf_tag}")

    telemetry_collector = None
    if args.pagein_telemetry:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from _pagein_telemetry import PageinTelemetry
        telemetry_collector = PageinTelemetry(adapter, enabled=True)
        telemetry_collector.start()

    result = run_niah_sweep(
        model, tokenizer, adapter,
        context_lengths=args.contexts,
        depths=args.depths,
        num_needles=args.needles,
        top_k_fp16_keys=args.top_k_fp16,
        v_tolerance=args.v_tolerance,
        use_int4_values=args.use_int4_values,
        group_size=args.group_size,
        fp16_key_cache_blocks=args.fp16_key_cache_blocks,
        fp16_value_cache_blocks=args.fp16_value_cache_blocks,
        ranking_fallback=args.ranking_fallback,
        ranking_r=args.ranking_r,
        ranking_fallback_mode=args.ranking_fallback_mode,
        tau_cov=tau_cov,
        k_min=args.k_min,
        k_max=args.k_max,
        rung1_threshold=args.rung1_threshold,
        rung1_multiplier=args.rung1_multiplier,
        score_consistency_check=args.score_consistency_check,
        score_consistency_interval=args.score_consistency_interval,
        eps_guard=args.eps_guard,
        exploration_rate=args.exploration_rate,
        telemetry_collector=telemetry_collector,
        trial_start=args.trial_start,
        trial_count=args.trial_count,
    )

    if telemetry_collector is not None:
        telemetry_collector.finish()
        tele_path = args.telemetry_output or (str(args.output).replace(".json", ".pagein.json"))
        telemetry_collector.write_json(tele_path)
        s = telemetry_collector.summary()
        print(f"\nPage-in telemetry: n_steps={s.get('n_steps',0)} "
              f"h2d_mean={s.get('h2d_total_bytes_mean',0)/1024:.1f} KB/step, "
              f"pct_zero_pagein={s.get('pct_steps_zero_pagein',0):.1%}, "
              f"rung1_rate={s.get('rung1_rate',0):.2%}, rung2_rate={s.get('rung2_rate',0):.2%}, "
              f"rung3_rate={s.get('rung3_rate',0):.2%}, rung4_rate={s.get('rung4_rate',0):.2%}")

    print(f"\n{'='*50}")
    print(f"Dense accuracy:     {result['dense_accuracy']:.1%}")
    print(f"Certified accuracy: {result['certified_accuracy']:.1%}")
    print(f"Critical failures:  {result['critical_failures']}")
    stats = result.get("paired_stats", {})
    if stats.get("n"):
        print(
            "Paired Δ: "
            f"{stats['delta_pp']:+.1f} pp "
            f"(95% bootstrap CI "
            f"[{stats['bootstrap_ci_pp_lo']:+.1f}, {stats['bootstrap_ci_pp_hi']:+.1f}] pp, "
            f"McNemar p={stats['mcnemar_p']:.4g}, n={stats['n']})"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": result["results"],
        "heatmaps": result["heatmaps"],
        "dense_accuracy": result["dense_accuracy"],
        "certified_accuracy": result["certified_accuracy"],
        "critical_failures": result["critical_failures"],
        "paired_stats": result["paired_stats"],
        "paired_stats_by_context": result["paired_stats_by_context"],
        "paired_stats_by_needle_group": result["paired_stats_by_needle_group"],
        "cache_config": cache_config_dict(args),
    }
    if "ranking_fallback_summary" in result:
        payload["ranking_fallback_summary"] = result["ranking_fallback_summary"]
        s = result["ranking_fallback_summary"]
        print(f"Ranking fallback: mode={s['mode']} r={s['r']} "
              f"disagree_r1={s['disagree_rate_r1']:.1%} "
              f"disagree_r3={s['disagree_rate_r3']:.1%} "
              f"triggered={s['fallback_rate']:.1%} "
              f"({s['triggered']}/{s['heads_total']})")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"JSON -> {out_path}")


if __name__ == "__main__":
    main()
