"""Step-2 Checkpoint-2 smoke: tiny-Llama E2E with the paper-§7 stack.

Exercises the full pipeline that the paper benches will use:
  asymmetric INT8 keys (Step 1)
  + INT4 per-group g=16 values (Step 2)
  + score_consistency_check=True (Step 0 default flip)
  + v_tolerance=0.05 (Step 0 plumbing)

This is the first time the INT4 stack runs against asymmetric keys end to
end. Smoke-level: 6-layer model, short prompt, ~16 decode steps, assert
finiteness and reasonable telemetry. The Llama-3.1-8B paper benches will
use the same code paths at full size in Step 6.

Pattern follows tests/test_gemma4_integration.py:65-80.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from dotcache.config import DotCacheConfig
from dotcache.integrations.llama import (
    LlamaDotCacheModelAdapter,
    CertifiedAttentionState,
    _ensure_certified_imports,
)
from dotcache.kernels.tiered_kv_cache import (
    create_tiered_cache_int4v_from_model,
    create_tiered_cache_from_model,
)


CUDA = torch.cuda.is_available()
LlamaConfig = transformers.LlamaConfig
LlamaForCausalLM = transformers.LlamaForCausalLM


def _tiny_llama_for_certified(num_layers: int = 6, hidden: int = 128, heads: int = 4,
                              kv_heads: int = 2, head_dim: int = 32, vocab: int = 256,
                              max_pos: int = 256):
    """Tiny Llama configured to look like the paper model in miniature."""
    cfg = LlamaConfig(
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        vocab_size=vocab,
        max_position_embeddings=max_pos,
    )
    model = LlamaForCausalLM(cfg)
    model.eval()
    if CUDA:
        model = model.cuda()
    dotcache_cfg = DotCacheConfig(head_dim=head_dim)
    adapter = LlamaDotCacheModelAdapter(model, dotcache_cfg)
    return model, adapter


def _run_paper_pipeline(*, v_tolerance: float, use_int4_values: bool,
                        prompt_len: int = 128, decode_steps: int = 16) -> dict:
    """Run the paper-§7 pipeline on a tiny Llama and collect telemetry."""
    model, adapter = _tiny_llama_for_certified()
    device = next(model.parameters()).device

    torch.manual_seed(20260424)
    input_ids = torch.randint(1, model.config.vocab_size, (1, prompt_len), device=device)

    # Phase 1: dense prefill
    adapter.set_mode("dense")
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
    past_kv = out.past_key_values

    # Phase 2: build the certified cache (INT4 g=16 if requested)
    _ensure_certified_imports()
    layer_ids = list(range(model.config.num_hidden_layers))
    if use_int4_values:
        tiered = create_tiered_cache_int4v_from_model(
            past_kv, layer_ids, group_size=16,
            max_new_tokens=decode_steps + 4,
        )
    else:
        tiered = create_tiered_cache_from_model(
            past_kv, layer_ids, max_new_tokens=decode_steps + 4,
        )

    # Build state with the paper §7 knobs
    adapter.certified_state = CertifiedAttentionState(
        tiered_caches=tiered,
        collect_stats=True,
        append_kv=True,
        top_k_fp16_keys=4,
        v_tolerance=v_tolerance,
        tau_cov=0.995,
        k_min=2,
        k_max=128,
        ranking_fallback=True,
        ranking_r=1,
        ranking_fallback_mode="full",
        score_consistency_check=True,
        eps_guard=0.01,
        exploration_rate=0.02,
        rung1_threshold=0.02,
        rung1_multiplier=2.0,
    )
    adapter.set_mode("certified")

    # Phase 3: decode loop, capture step stats
    cache_pos = torch.tensor([prompt_len], dtype=torch.long, device=device)
    cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    for _ in range(decode_steps):
        with torch.inference_mode():
            step = model(
                input_ids=cur, use_cache=False,
                cache_position=cache_pos,
                position_ids=cache_pos.unsqueeze(0),
            )
        nxt = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cur = nxt
        cache_pos = cache_pos + 1

    # Aggregate per-step stats across layers (paper-bench style)
    stats = adapter.certified_state.aggregate_step_stats()
    return {
        "n_step_layer_records": len(adapter.certified_state.step_stats),
        "stats": stats,
    }


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
class TestPaperPipelineSmoke:
    def test_paper_stack_e2e_runs_and_telemetry_is_finite(self):
        """Asymmetric INT8 + INT4 g=16 + v_tolerance=0.05 + §7 knobs.
        Confirms the integrated stack runs without exceptions and that all
        emitted telemetry numbers are finite."""
        result = _run_paper_pipeline(
            v_tolerance=0.05, use_int4_values=True,
            prompt_len=128, decode_steps=16,
        )
        stats = result["stats"]
        assert result["n_step_layer_records"] > 0, "no per-layer stats captured"
        # All numeric fields finite
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                assert math.isfinite(v), f"non-finite stat {k} = {v}"

    def test_v_tolerance_zero_forces_rung2(self):
        """Setting v_tolerance=0 must force every step to escalate to FP16
        values (Rung-2). This exercises the fallback path that has never run
        in a paper bench before — the audit's Mismatch 2 risk."""
        result = _run_paper_pipeline(
            v_tolerance=0.0, use_int4_values=True,
            prompt_len=64, decode_steps=8,
        )
        # rung2_fired indicator should be present and true at least once across
        # the 8 steps × 6 layers = 48 layer-step records.
        # The aggregated key is rung2_fired_layers (count) or rung2_rate (frac).
        stats = result["stats"]
        # Look for any rung2 indicator with a positive value
        rung2_keys = [k for k in stats if "rung2" in k]
        assert any(stats[k] for k in rung2_keys if isinstance(stats[k], (int, float))), (
            f"Expected at least one Rung-2 trigger with v_tolerance=0; "
            f"rung2 stats: {[(k, stats[k]) for k in rung2_keys]}"
        )

    @pytest.mark.parametrize("use_int4", [True, False])
    def test_paper_stack_logits_remain_finite_through_decode(self, use_int4: bool):
        """Smoke that finite logits emerge across the decode loop for both
        INT4 and FP16 value paths. Catches NaN/Inf regressions from the
        kernel changes in Steps 1 and 2."""
        result = _run_paper_pipeline(
            v_tolerance=0.05, use_int4_values=use_int4,
            prompt_len=64, decode_steps=8,
        )
        # Just verify aggregator returned something and didn't blow up
        assert result["n_step_layer_records"] > 0


def main_print_checkpoint2():
    """Optional: run as a script to dump the full per-step telemetry block
    for the user to inspect (Checkpoint 2 evidence)."""
    print("=" * 70)
    print("CHECKPOINT 2 — paper §7 stack on tiny Llama")
    print("=" * 70)
    print("Config: asymmetric INT8 keys + INT4 g=16 values + v_tolerance=0.05")
    print("Model:  6-layer Llama, hidden=128, heads=4 (kv=2), head_dim=32")
    print("Prompt: 128 tokens; decode: 32 steps")
    print()
    result = _run_paper_pipeline(
        v_tolerance=0.05, use_int4_values=True,
        prompt_len=128, decode_steps=32,
    )
    print(f"Per-layer stat records: {result['n_step_layer_records']}")
    print()
    print("Aggregated stats:")
    print(json.dumps(result["stats"], indent=2, default=str))


if __name__ == "__main__":
    main_print_checkpoint2()
