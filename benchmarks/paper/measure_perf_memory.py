"""Measure cache memory for the single-machine performance section."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pg19_perplexity import load_pg19_chunks
from _provenance import (
    add_paper_cache_args,
    add_paper_section7_args,
    configure_paper_runtime_defaults,
    resolve_fp16_key_cache_blocks,
    resolve_fp16_value_cache_blocks,
)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _tensor_bytes(x: Any) -> int:
    return int(x.nelement() * x.element_size()) if x is not None else 0


def _past_kv_bytes(past_kv: Any) -> int:
    total = 0
    for layer in past_kv:
        if isinstance(layer, (tuple, list)):
            for item in layer:
                if torch.is_tensor(item):
                    total += _tensor_bytes(item)
    return int(total)


def _cache_memory(tiered_caches: dict[int, Any]) -> dict[str, int]:
    tier1 = 0
    scratch = 0
    system = 0
    key_scratch = 0
    value_scratch = 0
    pagein_scratch = 0
    for cache in tiered_caches.values():
        k_scratch = _tensor_bytes(getattr(cache, "keys_fp16_gpu", None))
        v_scratch = _tensor_bytes(getattr(cache, "values_fp16_gpu", None))
        pagein = _tensor_bytes(getattr(cache, "_pagein_buffer", None))
        key_scratch += k_scratch
        value_scratch += v_scratch
        pagein_scratch += pagein
        scratch += k_scratch + v_scratch + pagein
        tier1 += _tensor_bytes(getattr(cache, "keys_int8", None))
        tier1 += _tensor_bytes(getattr(cache, "keys_scale", None))
        tier1 += _tensor_bytes(getattr(cache, "keys_zero_points", None))
        tier1 += _tensor_bytes(getattr(cache, "correction", None))
        tier1 += _tensor_bytes(getattr(cache, "values_norm_max_per_block", None))
        tier1 += _tensor_bytes(getattr(cache, "values_int4_packed", None))
        tier1 += _tensor_bytes(getattr(cache, "values_int4_scales", None))
        tier1 += _tensor_bytes(getattr(cache, "values_int4_zeros", None))
        tier1 += _tensor_bytes(getattr(cache, "values_int4_errors", None))
        tier1 += _tensor_bytes(getattr(cache, "values_int4_error_sums", None))
        tier1 += _tensor_bytes(getattr(cache, "values_int4_error_counts", None))
        system += int(cache.cpu_bytes())
    return {
        "cert_tier1_vram_bytes": int(tier1),
        "cert_scratch_vram_bytes": int(scratch),
        "cert_key_scratch_vram_bytes": int(key_scratch),
        "cert_value_scratch_vram_bytes": int(value_scratch),
        "cert_pagein_scratch_vram_bytes": int(pagein_scratch),
        "cert_total_vram_bytes": int(tier1 + scratch),
        "cert_system_ram_bytes": int(system),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
    parser.add_argument("--contexts", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    add_paper_cache_args(parser)
    add_paper_section7_args(parser)
    args = parser.parse_args()
    configure_paper_runtime_defaults()

    from dotcache.kernels.tiered_kv_cache import create_tiered_cache_int4v_from_model

    token = os.environ.get("HF_TOKEN") or None
    warnings.filterwarnings(
        "ignore",
        message=r"MatMul8bitLt: inputs will be cast from .* during quantization",
        category=UserWarning,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        dtype=torch.float16,
        token=token,
    )
    model.eval()

    rows: dict[str, Any] = {}
    for context in args.contexts:
        chunks, _ = load_pg19_chunks(tokenizer, context, 1)
        input_ids = chunks[0].unsqueeze(0).to("cuda")
        with torch.inference_mode():
            dense_out = model(input_ids=input_ids[:, :context], use_cache=True)
        _sync()
        past_kv = dense_out.past_key_values
        dense_bytes = _past_kv_bytes(past_kv)
        layer_ids = list(range(model.config.num_hidden_layers))
        key_cap = resolve_fp16_key_cache_blocks(
            args.fp16_key_cache_blocks,
            os.environ.get("DOTCACHE_FP16_CACHE_BLOCKS"),
        )
        value_cap = resolve_fp16_value_cache_blocks(
            args.fp16_value_cache_blocks,
            os.environ.get("DOTCACHE_FP16_VALUE_CACHE_BLOCKS"),
        )
        max_new = 16
        tiered_caches = create_tiered_cache_int4v_from_model(
            past_kv,
            layer_ids,
            group_size=args.group_size,
            max_new_tokens=max_new,
            fp16_key_cache_capacity=key_cap,
            fp16_value_cache_capacity=value_cap,
        )
        _sync()
        mem = _cache_memory(tiered_caches)
        rows[str(context)] = {
            "dense_vram_mb": dense_bytes / 1e6,
            "cert_tier1_vram_mb": mem["cert_tier1_vram_bytes"] / 1e6,
            "cert_scratch_vram_mb": mem["cert_scratch_vram_bytes"] / 1e6,
            "cert_key_scratch_vram_mb": mem["cert_key_scratch_vram_bytes"] / 1e6,
            "cert_value_scratch_vram_mb": mem["cert_value_scratch_vram_bytes"] / 1e6,
            "cert_pagein_scratch_vram_mb": mem["cert_pagein_scratch_vram_bytes"] / 1e6,
            "cert_total_vram_mb": mem["cert_total_vram_bytes"] / 1e6,
            "cert_system_ram_mb": mem["cert_system_ram_bytes"] / 1e6,
            "vram_ratio": mem["cert_total_vram_bytes"] / max(dense_bytes, 1),
            "tier1_ratio": mem["cert_tier1_vram_bytes"] / max(dense_bytes, 1),
            "fp16_key_cache_blocks": key_cap if key_cap is not None else "full",
            "fp16_value_cache_blocks": value_cap if value_cap is not None else "full",
            "cache_mode": "full" if key_cap is None or value_cap is None else f"capped-{key_cap}",
        }
        del tiered_caches, past_kv, dense_out, input_ids
        gc.collect()
        torch.cuda.empty_cache()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
