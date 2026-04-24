"""Minimal LLaMA adapter for the certified-attention benchmark scripts.

This file intentionally contains only the paper benchmark path extracted from
DotCache's broader experimental adapter:

- dense mode delegates to the model's original attention module;
- certified mode replaces each LLaMA self-attention module with a wrapper that
  projects Q/K/V, appends decode tokens to the tiered cache, and calls the
  local certified attention kernel.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal


def transformers_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


if transformers_available():
    import torch
    import torch.nn as nn
    import transformers.models.llama.modeling_llama as llama_mod
else:  # pragma: no cover
    torch = None

    class _FallbackNN:
        class Module:
            pass

    nn = _FallbackNN()  # type: ignore[assignment]
    llama_mod = None


AttentionMode = Literal["dense", "certified"]


def _ensure_certified_imports() -> None:
    import certified_attention  # noqa: F401
    import tiered_cache  # noqa: F401


@dataclass
class CertifiedAttentionState:
    tiered_caches: dict
    layer_epsilons: dict[int, float]
    default_epsilon: float = 1e-4
    block_size: int = 16
    collect_stats: bool = True
    append_kv: bool = False
    top_k_fp16_keys: int = 4
    concentration_threshold: float = 0.0
    ranking_fallback: bool = False
    ranking_r: int = 1
    ranking_fallback_mode: str = "full"
    tau_cov: float | None = None
    k_min: int = 2
    k_max: int | None = None
    rung1_threshold: float = 0.02
    rung1_multiplier: float = 2.0
    score_consistency_check: bool = False
    eps_guard: float = 0.01
    exploration_rate: float = 0.0
    per_kv_group_topk: bool = False
    value_error_mode: str = "tight"
    step_stats: list | None = None
    phase_timings: dict | None = None
    _clear_seq: int = 0

    def __post_init__(self) -> None:
        if self.step_stats is None:
            self.step_stats = []

    def clear_step_stats(self) -> list[dict]:
        stats = self.step_stats or []
        self.step_stats = []
        self._clear_seq += 1
        return stats

    def aggregate_step_stats(self, since: int = 0) -> dict:
        entries = (self.step_stats or []) if since <= 0 else (self.step_stats or [])[since:]
        if not entries:
            return {
                "int8_tail_rate": 0.0,
                "total_blocks": 0,
                "int8_tail_blocks": 0,
                "skip_rate": 0.0,
                "skipped_blocks": 0,
            }

        total = sum(s.get("total_blocks", 0) for s in entries)
        int8_tail = sum(s.get("int8_tail_blocks", s.get("skipped_blocks", 0)) for s in entries)
        per_layer = {
            s.get("layer", i): s.get("int8_tail_rate", s.get("skip_rate", 0.0))
            for i, s in enumerate(entries)
        }
        agg: dict[str, Any] = {
            "int8_tail_rate": int8_tail / total if total else 0.0,
            "total_blocks": total,
            "int8_tail_blocks": int8_tail,
            "per_layer_int8_tail_rate": per_layer,
            "skip_rate": int8_tail / total if total else 0.0,
            "skipped_blocks": int8_tail,
            "per_layer_skip_rate": per_layer,
        }

        if any("ranking_heads_total" in s for s in entries):
            heads_total = sum(s.get("ranking_heads_total", 0) for s in entries)
            triggered = sum(s.get("ranking_fallback_triggered", 0) for s in entries)
            disagree_r1 = sum(s.get("ranking_disagree_r1", 0) for s in entries)
            disagree_r3 = sum(s.get("ranking_disagree_r3", 0) for s in entries)
            agg.update({
                "ranking_heads_total": heads_total,
                "ranking_fallback_triggered": triggered,
                "ranking_disagree_r1": disagree_r1,
                "ranking_disagree_r3": disagree_r3,
                "ranking_fallback_rate": triggered / heads_total if heads_total else 0.0,
                "ranking_disagree_rate_r1": disagree_r1 / heads_total if heads_total else 0.0,
                "ranking_disagree_rate_r3": disagree_r3 / heads_total if heads_total else 0.0,
            })

        for key in ("rung1_fired", "rung2_fired", "rung3_fired", "rung4_fired"):
            if any(key in s for s in entries):
                agg[key] = any(bool(s.get(key)) for s in entries)
                agg[key.replace("fired", "fired_layers")] = sum(bool(s.get(key)) for s in entries)

        scalar_rollups = {
            "tail_mass_step_mean": "tail_mass_int8_est_mean",
            "tail_mass_step_max": "tail_mass_int8_est_max",
            "tau_cov_actual_step_mean": "tau_cov_actual_mean",
            "e_val_step_max": "e_val_max",
            "e_val_step_mean": "e_val_mean",
            "score_residual_step_max": "score_residual_max",
            "score_residual_step_mean": "score_residual_mean",
            "score_residual_ratio_step_max": "score_residual_ratio_max",
            "score_residual_ratio_step_mean": "score_residual_ratio_mean",
            "delta_bound_step_max": "delta_bound_max",
            "delta_bound_step_mean": "delta_bound_mean",
        }
        for out_key, in_key in scalar_rollups.items():
            vals = [float(s[in_key]) for s in entries if s.get(in_key) is not None]
            if vals:
                agg[out_key] = max(vals) if out_key.endswith("_max") else sum(vals) / len(vals)

        k_star = [float(s["k_star_mean"]) for s in entries if s.get("k_star_mean") is not None]
        if k_star:
            agg["k_star_mean"] = sum(k_star) / len(k_star)
            agg["k_star_max"] = int(max(s.get("k_star_max", 0) for s in entries))

        if any("h2d_total_bytes" in s for s in entries):
            agg["h2d_key_bytes"] = int(sum(s.get("h2d_key_bytes", 0) for s in entries))
            agg["h2d_value_bytes"] = int(sum(s.get("h2d_value_bytes", 0) for s in entries))
            agg["h2d_total_bytes"] = agg["h2d_key_bytes"] + agg["h2d_value_bytes"]

        if any("fp16_cache_capacity_blocks" in s for s in entries):
            hits = int(sum(s.get("fp16_cache_hits_step", 0) for s in entries))
            misses = int(sum(s.get("fp16_cache_misses_step", 0) for s in entries))
            agg["fp16_cache_capacity_blocks"] = int(entries[0].get("fp16_cache_capacity_blocks", 0))
            agg["fp16_cache_hits"] = hits
            agg["fp16_cache_misses"] = misses
            agg["fp16_cache_hit_rate"] = hits / (hits + misses) if hits + misses else 0.0

        return agg


class LlamaCertifiedAttention(nn.Module):
    def __init__(self, base_attention: nn.Module, adapter: "LlamaDotCacheModelAdapter") -> None:
        super().__init__()
        self.base_attention = base_attention
        self.adapter = adapter
        self.layer_idx = int(base_attention.layer_idx)
        self.config = base_attention.config

    def forward(
        self,
        hidden_states: "torch.Tensor",
        position_embeddings: tuple["torch.Tensor", "torch.Tensor"] | None = None,
        attention_mask: "torch.Tensor" | None = None,
        past_key_values=None,
        cache_position: "torch.LongTensor" | None = None,
        **kwargs: Any,
    ):
        if self.adapter.mode == "dense":
            return self.base_attention(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        return self._forward_certified(hidden_states, position_embeddings=position_embeddings)

    def _project_qkv(self, hidden_states, position_embeddings):
        if position_embeddings is None:
            raise ValueError("position_embeddings are required for LLaMA attention")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.base_attention.head_dim)
        query_states = self.base_attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.base_attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.base_attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = llama_mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        return query_states, key_states, value_states

    def _forward_certified(self, hidden_states, position_embeddings=None):
        if tuple(hidden_states.shape[:2]) != (1, 1):
            raise ValueError("Certified decode mode only supports batch=1 and query_len=1")

        from certified_attention import certified_attention_layer

        cert_state = self.adapter.certified_state
        if cert_state is None:
            raise ValueError("No CertifiedAttentionState is attached to the adapter")
        cache = cert_state.tiered_caches.get(self.layer_idx)
        if cache is None:
            raise ValueError(f"No tiered cache for layer {self.layer_idx}")

        query_states, key_states, value_states = self._project_qkv(hidden_states, position_embeddings)
        q_all = query_states[0, :, 0, :]

        if cert_state.append_kv:
            cache.append_token(key_states[0], value_states[0])

        gqa_group = self.config.num_attention_heads // self.config.num_key_value_heads
        context_states, stats = certified_attention_layer(
            cache,
            q_all,
            gqa_group,
            float(self.base_attention.scaling),
            block_epsilon=cert_state.layer_epsilons.get(self.layer_idx, cert_state.default_epsilon),
            collect_stats=cert_state.collect_stats,
            top_k_fp16_keys=cert_state.top_k_fp16_keys,
            concentration_threshold=cert_state.concentration_threshold,
            ranking_fallback=cert_state.ranking_fallback,
            ranking_r=cert_state.ranking_r,
            ranking_fallback_mode=cert_state.ranking_fallback_mode,
            tau_cov=cert_state.tau_cov,
            k_min=cert_state.k_min,
            k_max=cert_state.k_max,
            rung1_threshold=cert_state.rung1_threshold,
            rung1_multiplier=cert_state.rung1_multiplier,
            score_consistency_check=cert_state.score_consistency_check,
            eps_guard=cert_state.eps_guard,
            exploration_rate=cert_state.exploration_rate,
            phase_timings=cert_state.phase_timings,
            per_kv_group_topk=cert_state.per_kv_group_topk,
            value_error_mode=cert_state.value_error_mode,
        )
        if cert_state.collect_stats and stats:
            cert_state.step_stats.append({"layer": self.layer_idx, **stats})

        context_tensor = context_states.to(dtype=hidden_states.dtype, device=hidden_states.device).unsqueeze(0)
        output = self.base_attention.o_proj(context_tensor.reshape(1, 1, -1).contiguous())
        return output, None


class LlamaDotCacheModelAdapter:
    def __init__(self, model, dotcache_config, *, backend: str = "auto", cache: Any = None) -> None:
        if not transformers_available():
            raise RuntimeError("transformers and torch are required for LLaMA benchmarks")
        self.model = model
        self.dotcache_config = dotcache_config
        self.backend = backend
        self.mode: AttentionMode = "dense"
        self.certified_state: CertifiedAttentionState | None = None
        self._wrappers: list[LlamaCertifiedAttention] = []
        self._install_wrappers()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _install_wrappers(self) -> None:
        for layer in self.model.model.layers[: self.model.config.num_hidden_layers]:
            wrapper = LlamaCertifiedAttention(layer.self_attn, self)
            layer.self_attn = wrapper
            self._wrappers.append(wrapper)

    def set_mode(self, mode: AttentionMode) -> None:
        if mode == "certified" and self.certified_state is None:
            raise ValueError("Must attach certified_state before setting mode='certified'")
        self.mode = mode

    def load_certified_cache(self, past_key_values, **kwargs: Any) -> None:
        from tiered_cache import create_tiered_cache_from_model, create_tiered_cache_int4v_from_model

        layer_ids = list(range(self.model.config.num_hidden_layers))
        block_size = int(kwargs.get("block_size", 16))
        use_int4_values = bool(kwargs.get("use_int4_values", False))
        factory = create_tiered_cache_int4v_from_model if use_int4_values else create_tiered_cache_from_model
        tiered_caches = factory(past_key_values, layer_ids, block_size=block_size)

        profile = kwargs.get("epsilon_profile")
        context_length = kwargs.get("context_length")
        if profile is not None:
            layer_epsilons = profile.get_layer_epsilons_min(context_length or 8192)
        else:
            layer_epsilons = kwargs.get("layer_epsilons") or {}

        forwarded = {
            key: kwargs[key]
            for key in (
                "top_k_fp16_keys",
                "tau_cov",
                "k_min",
                "k_max",
                "ranking_fallback",
                "ranking_r",
                "ranking_fallback_mode",
                "score_consistency_check",
                "eps_guard",
                "exploration_rate",
                "rung1_threshold",
                "rung1_multiplier",
                "per_kv_group_topk",
            )
            if key in kwargs and kwargs[key] is not None
        }
        self.certified_state = CertifiedAttentionState(
            tiered_caches=tiered_caches,
            layer_epsilons=layer_epsilons,
            default_epsilon=float(kwargs.get("default_epsilon", 1e-4)),
            block_size=block_size,
            **forwarded,
        )


def resolve_hf_auth_kwargs() -> dict[str, str]:
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(env_name, "").strip()
        if token:
            return {"token": token}
    return {}

