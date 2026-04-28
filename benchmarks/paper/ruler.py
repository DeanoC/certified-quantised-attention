"""RULER benchmark (subset): dense vs certified attention.

Implements a self-contained subset of NVIDIA RULER (arXiv:2404.06654) using
only the synthetic subtasks that need no external dataset:

    niah_single      - 1 key, 1 value planted in noise haystack
    niah_multikey    - N distractor keys, 1 target; retrieve target value
    niah_multivalue  - 1 key with N values; retrieve ALL values
    niah_multiquery  - 1 key/value planted; query lists N candidate keys
    vt               - variable tracking: X1 = 12345; X2 = X1; ... ask Xk
    cwe              - common-words extraction (top-10 by frequency)
    fwe              - frequent-words extraction (top-3, zipfian)

Templates follow the upstream RULER wording. Haystack filler is the same block
used in niah.py, seeded RNG per-sample for reproducibility.

Scoring follows upstream:
    niah_*, vt, cwe, fwe: string_match_all (all refs must appear as substring,
                          case-insensitive; per-sample score = found/total).

Run: dense and certified generate on identical prompts; paired comparison.
Critical failure = certified FAIL where dense was OK (same as NIAH).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import string
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

# Shared filler (same block NIAH uses)
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

# RULER-style "magic number" needle templates (word-form keys, number values)
NIAH_PREAMBLE = (
    "Some special magic numbers are hidden within the following text. "
    "Make sure to memorize them. I will quiz you about the numbers afterwards.\n\n"
)
NIAH_SINGLE_PREAMBLE = (
    "A special magic number is hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the number afterwards.\n\n"
)
NIAH_KEY_WORDS = [
    "pelican", "quarry", "moonlit", "cipher", "lantern", "nebula", "prism",
    "velvet", "orbit", "rustic", "ember", "zenith", "hollow", "falcon",
    "meadow", "silver", "coral", "breeze", "ivory", "twilight",
]


def _random_number_value(rng: random.Random, digits: int = 7) -> str:
    return "".join(rng.choice(string.digits) for _ in range(digits))


def _pick_keys(rng: random.Random, k: int) -> list[str]:
    return rng.sample(NIAH_KEY_WORDS, k)


def _fill_haystack(target_tokens: int, tokenizer, payload_blocks: list[str],
                   rng: random.Random) -> str:
    """Interleave payload blocks among filler blocks to hit target_tokens.

    payload_blocks are inserted at random positions in the filler stream.
    Returns haystack string (without any preamble/question).
    """
    filler_tpb = len(tokenizer.encode(FILLER_BLOCK, add_special_tokens=False))
    payload_total = sum(len(tokenizer.encode(p, add_special_tokens=False))
                        for p in payload_blocks)
    # Leave headroom for preamble + question (~100 tokens)
    available = target_tokens - payload_total - 100
    num_filler = max(available // filler_tpb, len(payload_blocks) + 1)

    # Pick insertion positions uniformly across filler stream
    positions = sorted(rng.sample(range(num_filler), len(payload_blocks)))

    parts = []
    pi = 0
    for i in range(num_filler):
        if pi < len(positions) and i == positions[pi]:
            parts.append(payload_blocks[pi])
            pi += 1
        parts.append(FILLER_BLOCK)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Subtask prompt builders — each returns (prompt, list[reference_answers])
# ---------------------------------------------------------------------------

def make_niah_single(rng: random.Random, ctx_tokens: int, tokenizer):
    key = _pick_keys(rng, 1)[0]
    value = _random_number_value(rng)
    payload = f"\nOne of the special magic numbers for {key} is: {value}.\n\n"
    haystack = _fill_haystack(ctx_tokens, tokenizer, [payload], rng)
    question = (
        f"\nWhat is the special magic number for {key} mentioned in the "
        f"provided text?\nThe special magic number for {key} mentioned in "
        f"the provided text is"
    )
    prompt = NIAH_SINGLE_PREAMBLE + haystack + question
    return prompt, [value]


def make_niah_multikey(rng: random.Random, ctx_tokens: int, tokenizer,
                       num_keys: int = 4):
    keys = _pick_keys(rng, num_keys)
    values = [_random_number_value(rng) for _ in keys]
    target_idx = rng.randrange(num_keys)
    target_key = keys[target_idx]
    target_value = values[target_idx]
    payload_blocks = [
        f"\nOne of the special magic numbers for {k} is: {v}.\n\n"
        for k, v in zip(keys, values)
    ]
    haystack = _fill_haystack(ctx_tokens, tokenizer, payload_blocks, rng)
    question = (
        f"\nWhat is the special magic number for {target_key} mentioned in "
        f"the provided text?\nThe special magic number for {target_key} "
        f"mentioned in the provided text is"
    )
    prompt = NIAH_PREAMBLE + haystack + question
    return prompt, [target_value]


def make_niah_multivalue(rng: random.Random, ctx_tokens: int, tokenizer,
                         num_values: int = 4):
    key = _pick_keys(rng, 1)[0]
    values = [_random_number_value(rng) for _ in range(num_values)]
    payload_blocks = [
        f"\nOne of the special magic numbers for {key} is: {v}.\n\n"
        for v in values
    ]
    haystack = _fill_haystack(ctx_tokens, tokenizer, payload_blocks, rng)
    question = (
        f"\nWhat are all the special magic numbers for {key} mentioned in "
        f"the provided text?\nThe special magic numbers for {key} mentioned "
        f"in the provided text are"
    )
    prompt = NIAH_PREAMBLE + haystack + question
    return prompt, values  # all four must be recalled


def make_niah_multiquery(rng: random.Random, ctx_tokens: int, tokenizer,
                         num_queries: int = 4):
    all_keys = _pick_keys(rng, num_queries)
    planted_idx = rng.randrange(num_queries)
    planted_key = all_keys[planted_idx]
    planted_value = _random_number_value(rng)
    payload = (
        f"\nOne of the special magic numbers for {planted_key} is: "
        f"{planted_value}.\n\n"
    )
    haystack = _fill_haystack(ctx_tokens, tokenizer, [payload], rng)
    query_list = ", ".join(all_keys)
    question = (
        f"\nFor which of the following keys was a special magic number "
        f"mentioned in the provided text: {query_list}? Provide the key "
        f"and its magic number.\nThe key with the special magic number is"
    )
    prompt = NIAH_PREAMBLE + haystack + question
    return prompt, [planted_key, planted_value]


def make_vt(rng: random.Random, ctx_tokens: int, tokenizer,
            chain_len: int = 4, noise_chains: int = 3):
    """Variable tracking: emit several chains of X = value; Y = X; Z = Y.
    Ask for all variables in the target chain. RULER metric: string_match_all.
    """
    # Target chain
    target_value = _random_number_value(rng, digits=5)
    target_vars = [f"VAR_{rng.randint(10, 99)}_{i}" for i in range(chain_len)]
    # de-dup if collision
    while len(set(target_vars)) < chain_len:
        target_vars = [f"VAR_{rng.randint(10, 99)}_{i}" for i in range(chain_len)]

    chain_lines = [f"{target_vars[0]} = {target_value}"]
    for i in range(1, chain_len):
        chain_lines.append(f"{target_vars[i]} = {target_vars[i-1]}")

    # Noise chains with different values, different var names
    noise_lines = []
    for c in range(noise_chains):
        noise_value = _random_number_value(rng, digits=5)
        noise_vars = [f"VAR_{rng.randint(100, 999)}_{i}" for i in range(chain_len)]
        noise_lines.append(f"{noise_vars[0]} = {noise_value}")
        for i in range(1, chain_len):
            noise_lines.append(f"{noise_vars[i]} = {noise_vars[i-1]}")

    all_lines = chain_lines + noise_lines
    rng.shuffle(all_lines)
    payload = "\n\n" + "\n".join(all_lines) + "\n\n"

    haystack = _fill_haystack(ctx_tokens, tokenizer, [payload], rng)
    preamble = (
        "Memorize and track the chains of variable assignment hidden within "
        "the following text.\n\n"
    )
    question = (
        f"\nFind all variables that are assigned the value {target_value} "
        f"in the text above.\nAnswer: The variables are"
    )
    prompt = preamble + haystack + question
    return prompt, target_vars  # all four must appear


_CWE_POOL = [
    "apple", "river", "stone", "cloud", "forest", "mountain", "valley",
    "ocean", "desert", "island", "meadow", "garden", "bridge", "castle",
    "tower", "harbor", "market", "temple", "palace", "village", "country",
    "highway", "railway", "canyon", "glacier", "volcano", "prairie",
    "jungle", "marsh", "lagoon",
]


def make_cwe(rng: random.Random, ctx_tokens: int, tokenizer,
             top_k: int = 10, freq_high: int = 30, freq_low: int = 3):
    """Common-Words Extraction: mix of high-freq (target) and low-freq (noise)
    words. Ask for the top_k most common.
    """
    pool = list(_CWE_POOL)
    rng.shuffle(pool)
    targets = pool[:top_k]
    noise = pool[top_k:top_k + 15]

    tokens = []
    for w in targets:
        tokens.extend([w] * freq_high)
    for w in noise:
        tokens.extend([w] * freq_low)
    rng.shuffle(tokens)

    # Number the list, one word per line (RULER style)
    numbered = "\n".join(f"{i+1}. {w}" for i, w in enumerate(tokens))
    payload = "\n" + numbered + "\n"
    # For CWE we pad filler only if target_tokens significantly exceeds payload
    filler_tpb = len(tokenizer.encode(FILLER_BLOCK, add_special_tokens=False))
    payload_tpb = len(tokenizer.encode(payload, add_special_tokens=False))
    num_filler = max((ctx_tokens - payload_tpb - 100) // filler_tpb, 0)
    haystack = FILLER_BLOCK * num_filler + payload

    preamble = (
        "Below is a numbered list of words. In these words, some appear more "
        "often than others. Memorize the ones that appear most often.\n\n"
    )
    question = (
        f"\nQuestion: What are the {top_k} most common words in the above "
        f"list?\nAnswer: The {top_k} most common words are"
    )
    prompt = preamble + haystack + question
    return prompt, targets  # all top_k must appear


def make_fwe(rng: random.Random, ctx_tokens: int, tokenizer,
             top_k: int = 3, alpha: float = 2.0, pool_size: int = 20):
    """Frequent-Words Extraction: zipfian distribution, ask for top-3."""
    pool = list(_CWE_POOL)
    rng.shuffle(pool)
    pool = pool[:pool_size]
    # zipfian weights
    weights = [1.0 / ((i + 1) ** alpha) for i in range(len(pool))]
    # Pick total tokens so generated list fills but stays under ctx
    total_word_tokens = max(200, ctx_tokens // 4)
    tokens = rng.choices(pool, weights=weights, k=total_word_tokens)

    # Track actual frequencies and pick top-k observed
    from collections import Counter
    counts = Counter(tokens)
    top = [w for w, _ in counts.most_common(top_k)]

    numbered = " ".join(f".... {t}" for t in tokens)
    payload = "\n" + numbered + "\n"
    filler_tpb = len(tokenizer.encode(FILLER_BLOCK, add_special_tokens=False))
    payload_tpb = len(tokenizer.encode(payload, add_special_tokens=False))
    num_filler = max((ctx_tokens - payload_tpb - 100) // filler_tpb, 0)
    haystack = FILLER_BLOCK * num_filler + payload

    preamble = (
        "Read the following coded text and track the frequency of each coded "
        "word. Find the three most frequently appeared coded words.\n\n"
    )
    question = (
        "\nQuestion: Do not provide any explanation. Please ignore the dots "
        "'....'. What are the three most frequently appeared words in the "
        "above coded text?\nAnswer: The three most frequently appeared words "
        "are"
    )
    prompt = preamble + haystack + question
    return prompt, top


SUBTASK_BUILDERS = {
    "niah_single": make_niah_single,
    "niah_multikey": make_niah_multikey,
    "niah_multivalue": make_niah_multivalue,
    "niah_multiquery": make_niah_multiquery,
    "vt": make_vt,
    "cwe": make_cwe,
    "fwe": make_fwe,
}

# Max new tokens per subtask (just enough for the expected answer)
SUBTASK_MAX_NEW = {
    "niah_single": 32,
    "niah_multikey": 32,
    "niah_multivalue": 96,      # needs 4 numbers
    "niah_multiquery": 48,
    "vt": 96,                    # 4 variable names
    "cwe": 96,                   # 10 words
    "fwe": 32,                   # 3 words
}


def score_string_match_all(generated: str, references: list[str]) -> float:
    """Fraction of references found as case-insensitive substring."""
    if not references:
        return 1.0
    g = generated.lower()
    hits = sum(1 for r in references if r.lower() in g)
    return hits / len(references)


def paired_ruler_stats(
    results: list[dict],
    *,
    bootstrap_iters: int = 10_000,
    seed: int = 20260425,
) -> dict:
    """Paired RULER score delta summary for paper table generation.

    Scores are continuous in [0, 1]. The CI is a bootstrap percentile interval
    over per-sample (cert_score - dense_score), preserving the paired design.
    """
    n = len(results)
    if n == 0:
        return {
            "n": 0,
            "dense_mean": None,
            "certified_mean": None,
            "delta": None,
            "delta_ci_lo": None,
            "delta_ci_hi": None,
        }
    dense = np.asarray([float(r["dense_score"]) for r in results], dtype=np.float64)
    cert = np.asarray([float(r["cert_score"]) for r in results], dtype=np.float64)
    diffs = cert - dense
    delta = float(diffs.mean())
    if n > 1 and bootstrap_iters > 0:
        rng = np.random.default_rng(seed)
        boot = diffs[rng.integers(0, n, size=(int(bootstrap_iters), n))].mean(axis=1)
        ci_lo = float(np.quantile(boot, 0.025))
        ci_hi = float(np.quantile(boot, 0.975))
    else:
        ci_lo = ci_hi = delta
    return {
        "n": int(n),
        "dense_mean": float(dense.mean()),
        "certified_mean": float(cert.mean()),
        "delta": delta,
        "delta_ci_lo": ci_lo,
        "delta_ci_hi": ci_hi,
        "bootstrap_iters": int(bootstrap_iters),
        "bootstrap_seed": int(seed),
    }


def paired_ruler_stats_by_context(results: list[dict]) -> dict[str, dict]:
    contexts = sorted({int(r["ctx_tokens"]) for r in results})
    return {
        f"{ctx // 1024}K": paired_ruler_stats(
            [r for r in results if int(r["ctx_tokens"]) == ctx],
            seed=20260425 + idx,
        )
        for idx, ctx in enumerate(contexts)
    }


def paired_ruler_stats_by_task_context(results: list[dict]) -> dict[str, dict]:
    keys = sorted({(str(r["subtask"]), int(r["ctx_tokens"])) for r in results})
    return {
        f"{task}_{ctx // 1024}K": paired_ruler_stats(
            [r for r in results if str(r["subtask"]) == task and int(r["ctx_tokens"]) == ctx],
            seed=20260425 + idx,
        )
        for idx, (task, ctx) in enumerate(keys)
    }


# ---------------------------------------------------------------------------
# Generation paths — mirror niah.py run_niah_cell
# ---------------------------------------------------------------------------

def generate_dense(model, tokenizer, adapter, prompt: str, max_new: int,
                   device: str = "cuda") -> tuple[str, int]:
    adapter.set_mode("dense")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(device)
    seq_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False, temperature=1.0,
        )
    gen_ids = outputs[0, seq_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, seq_len


def generate_certified(model, tokenizer, adapter, prompt: str, max_new: int,
                       *,
                       v_tolerance: float,
                       top_k_fp16_keys: int = 4,
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
                       telemetry_collector=None) -> tuple[str, int, dict]:
    from dotcache.integrations.llama import (
        _ensure_certified_imports, CertifiedAttentionState,
    )
    from dotcache.kernels.tiered_kv_cache import (
        create_tiered_cache_from_model,
        create_tiered_cache_int4v_from_model,
    )

    adapter.set_mode("dense")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(device)
    seq_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
    past_kv = outputs.past_key_values
    first_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del outputs

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
            max_new_tokens=max_new + 8,
            fp16_key_cache_capacity=_key_cap,
            fp16_value_cache_capacity=_value_cap,
        )
    else:
        tiered_caches = create_tiered_cache_from_model(
            past_kv, layer_ids, max_new_tokens=max_new + 8,
            fp16_key_cache_capacity=None,
        )
    del past_kv
    gc.collect()
    torch.cuda.empty_cache()

    collect_stats = (
        bool(ranking_fallback)
        or (tau_cov is not None and tau_cov > 0)
        or bool(score_consistency_check)
        or (exploration_rate and exploration_rate > 0)
    )
    adapter.certified_state = CertifiedAttentionState(
        tiered_caches=tiered_caches,
        collect_stats=collect_stats,
        append_kv=True,
        top_k_fp16_keys=top_k_fp16_keys,
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
    adapter.set_mode("certified")

    cache_position = torch.tensor([seq_len], dtype=torch.long, device=device)
    current_input = first_token
    gen_token_tensors = [first_token]
    for _ in range(max_new - 1):
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
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    # Drain aggregate stats (ranking / adaptive / exploration / score-consistency)
    # before clearing state; the orchestrator embeds these in the paper JSON.
    cell_agg = adapter.certified_state.aggregate_step_stats() if collect_stats else {}
    adapter.certified_state = None
    adapter.set_mode("dense")
    gc.collect()
    torch.cuda.empty_cache()
    return text, seq_len, cell_agg


# ---------------------------------------------------------------------------
# Top-level sweep
# ---------------------------------------------------------------------------

def run_ruler(
    model, tokenizer, adapter,
    subtasks: list[str], contexts: list[int], num_samples: int,
    *,
    v_tolerance: float,
    top_k_fp16_keys: int = 4,
    seed_base: int = 20260416, device: str = "cuda",
    sample_start: int = 0,
    use_int4_values: bool = False,
    group_size: int = 16,
    fp16_key_cache_blocks: int | str | None = None,
    fp16_value_cache_blocks: int | str | None = None,
    # Paper-alignment features.
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
) -> dict:
    results = []
    sample_start = max(0, int(sample_start))
    total = len(subtasks) * len(contexts) * num_samples * 2
    done = 0

    for ctx_len in contexts:
        for subtask in subtasks:
            builder = SUBTASK_BUILDERS[subtask]
            max_new = SUBTASK_MAX_NEW[subtask]
            for sidx in range(sample_start, sample_start + num_samples):
                # Deterministic per (subtask, ctx, sample) — avoid Python's
                # salted hash so seeds are stable across runs.
                key = f"{subtask}|{ctx_len}|{sidx}".encode()
                seed = seed_base + int(hashlib.md5(key).hexdigest()[:8], 16)
                rng = random.Random(seed)
                prompt, refs = builder(rng, ctx_len, tokenizer)

                dense_text, seq_dense = generate_dense(
                    model, tokenizer, adapter, prompt, max_new, device=device,
                )
                dense_score = score_string_match_all(dense_text, refs)
                done += 1
                status_d = f"{dense_score:.2f}"
                print(f"  [{done}/{total}] dense     {subtask:<18} "
                      f"{ctx_len//1024}K s={sidx} -> {status_d}")

                cert_text, seq_cert, cert_stats = generate_certified(
                    model, tokenizer, adapter, prompt, max_new,
                    v_tolerance=v_tolerance,
                    use_int4_values=use_int4_values,
                    group_size=group_size,
                    fp16_key_cache_blocks=fp16_key_cache_blocks,
                    fp16_value_cache_blocks=fp16_value_cache_blocks,
                    top_k_fp16_keys=top_k_fp16_keys,
                    device=device,
                    tau_cov=tau_cov, k_min=k_min, k_max=k_max,
                    ranking_fallback=ranking_fallback, ranking_r=ranking_r,
                    ranking_fallback_mode=ranking_fallback_mode,
                    score_consistency_check=score_consistency_check,
                    score_consistency_interval=score_consistency_interval,
                    eps_guard=eps_guard,
                    exploration_rate=exploration_rate,
                    rung1_threshold=rung1_threshold,
                    rung1_multiplier=rung1_multiplier,
                    telemetry_collector=telemetry_collector,
                )
                cert_score = score_string_match_all(cert_text, refs)
                done += 1
                status_c = f"{cert_score:.2f}"
                crit = (dense_score == 1.0 and cert_score < 1.0)
                flag = "  *** CRITICAL ***" if crit else ""
                print(f"  [{done}/{total}] certified {subtask:<18} "
                      f"{ctx_len//1024}K s={sidx} -> {status_c}{flag}")

                results.append({
                    "subtask": subtask, "ctx_tokens": ctx_len,
                    "sample_idx": sidx, "seq_len": seq_dense,
                    "refs": refs,
                    "dense_score": dense_score, "dense_gen": dense_text[:200],
                    "cert_score": cert_score, "cert_gen": cert_text[:200],
                    "critical": bool(crit),
                    "cert_stats": cert_stats,
                })

    # Aggregate
    by_task_ctx = {}
    for r in results:
        key = (r["subtask"], r["ctx_tokens"])
        agg = by_task_ctx.setdefault(key, {"dense": [], "cert": [], "crit": 0})
        agg["dense"].append(r["dense_score"])
        agg["cert"].append(r["cert_score"])
        if r["critical"]:
            agg["crit"] += 1

    summary = {}
    for (subtask, ctx), agg in by_task_ctx.items():
        summary[f"{subtask}_{ctx//1024}K"] = {
            "dense_mean": sum(agg["dense"]) / len(agg["dense"]),
            "cert_mean": sum(agg["cert"]) / len(agg["cert"]),
            "critical": agg["crit"],
            "n": len(agg["dense"]),
        }
    return {
        "results": results,
        "summary": summary,
        "paired_stats": paired_ruler_stats(results),
        "paired_stats_by_context": paired_ruler_stats_by_context(results),
        "paired_stats_by_task_context": paired_ruler_stats_by_task_context(results),
    }


def main():
    parser = argparse.ArgumentParser(description="RULER (subset): dense vs certified")
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
    parser.add_argument("--contexts", type=int, nargs="+", default=[4096])
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--sample-start", type=int, default=0,
                        help="First deterministic sample index to run (for distributed shards).")
    parser.add_argument("--sample-index", type=int, default=None,
                        help="Alias for --sample-start with --num-samples 1.")
    parser.add_argument("--subtasks", nargs="+", default=list(SUBTASK_BUILDERS.keys()))
    parser.add_argument("--output", default="benchmarks/results/ruler.json")
    parser.add_argument("--top-k-fp16", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260416)
    # Paper-alignment flags (T4/T7/Rung1/T9/T10).
    parser.add_argument("--tau-cov", type=float, default=PAPER_TAU_COV,
                        help=f"Adaptive K* cumulative-mass threshold (paper default {PAPER_TAU_COV}; set 0 to disable)")
    parser.add_argument("--k-min", type=int, default=PAPER_K_MIN)
    parser.add_argument("--k-max", type=int, default=PAPER_K_MAX)
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
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from _provenance import (
        add_paper_cache_args,
        cache_config_dict,
        configure_paper_runtime_defaults,
    )
    add_paper_cache_args(parser)
    args = parser.parse_args()
    if args.sample_index is not None:
        args.sample_start = int(args.sample_index)
        args.num_samples = 1
    configure_paper_runtime_defaults()

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from dotcache.integrations.llama import LlamaDotCacheModelAdapter
    from dotcache.config import DotCacheConfig

    for st in args.subtasks:
        if st not in SUBTASK_BUILDERS:
            raise SystemExit(f"Unknown subtask: {st} (valid: {list(SUBTASK_BUILDERS.keys())})")

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

    print(f"\nRULER: subtasks={args.subtasks}, "
          f"contexts={[c//1024 for c in args.contexts]}K, "
          f"n={args.num_samples}, "
          f"top_k_fp16={args.top_k_fp16}")

    telemetry_collector = None
    if args.pagein_telemetry:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from _pagein_telemetry import PageinTelemetry
        telemetry_collector = PageinTelemetry(adapter, enabled=True)
        telemetry_collector.start()

    t0 = time.perf_counter()
    out = run_ruler(
        model, tokenizer, adapter,
        subtasks=args.subtasks, contexts=args.contexts,
        num_samples=args.num_samples,
        v_tolerance=args.v_tolerance,
        use_int4_values=args.use_int4_values,
        group_size=args.group_size,
        fp16_key_cache_blocks=args.fp16_key_cache_blocks,
        fp16_value_cache_blocks=args.fp16_value_cache_blocks,
        top_k_fp16_keys=args.top_k_fp16,
        seed_base=args.seed,
        sample_start=args.sample_start,
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
    )
    wall = time.perf_counter() - t0

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

    print(f"\n{'='*60}")
    print(f"RULER subset results ({wall/60:.1f} min)")
    print(f"{'='*60}")
    print(f"{'task_ctx':<22} {'dense':>8} {'cert':>8} {'Δ':>8} {'crit':>6}")
    for key, s in sorted(out["summary"].items()):
        delta = s["cert_mean"] - s["dense_mean"]
        print(f"{key:<22} {s['dense_mean']:>8.3f} {s['cert_mean']:>8.3f} "
              f"{delta:>+8.3f} {s['critical']:>6}")

    # Overall
    dmean = [r["dense_score"] for r in out["results"]]
    cmean = [r["cert_score"] for r in out["results"]]
    critical = sum(1 for r in out["results"] if r["critical"])
    overall_d = sum(dmean) / len(dmean)
    overall_c = sum(cmean) / len(cmean)
    print(f"\nOverall: dense={overall_d:.3f} cert={overall_c:.3f} "
          f"Δ={overall_c-overall_d:+.3f} critical={critical}/{len(dmean)}")
    paired = out.get("paired_stats", {})
    if paired.get("n"):
        print(
            "Paired Δ CI: "
            f"{paired['delta']:+.3f} "
            f"[{paired['delta_ci_lo']:+.3f}, {paired['delta_ci_hi']:+.3f}] "
            f"n={paired['n']}"
        )

    payload = {
        "benchmark": "ruler_subset",
        "model": args.model,
        "subtasks": args.subtasks,
        "contexts": args.contexts,
        "num_samples": args.num_samples,
        "sample_start": args.sample_start,
        "top_k_fp16": args.top_k_fp16,
        "seed": args.seed,
        "wall_minutes": wall / 60.0,
        "overall_dense": overall_d,
        "overall_cert": overall_c,
        "critical_failures": critical,
        "paired_stats": out["paired_stats"],
        "paired_stats_by_context": out["paired_stats_by_context"],
        "paired_stats_by_task_context": out["paired_stats_by_task_context"],
        "summary": out["summary"],
        "results": out["results"],
        "cache_config": cache_config_dict(args),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON -> {out_path}")


if __name__ == "__main__":
    main()
