from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import math

from .modes.m4_key_project import valid_m4_basis_families
from .planner import LayerPolicy, PageModeSpec, make_explicit_policy, make_tier_candidates, parse_page_mode_token


_VALID_KEY_MODES = ("M0", "M1", "M2", "M3", "M4", "T3")
_VALID_VALUE_MODES = ("M0", "M1", "M3", "T3")
_VALID_M4_BASIS_FAMILIES = valid_m4_basis_families()


def _parse_mode_override_spec(spec: str, *, allowed_modes: tuple[str, ...], field_name: str) -> tuple[int, int | None, str]:
    if "=" not in spec:
        raise ValueError(f"{field_name} entries must use layer:<id>=<mode> or layer:<id>:kv:<id>=<mode>")
    target, mode = spec.split("=", 1)
    mode = mode.strip()
    if mode not in allowed_modes:
        allowed = ", ".join(allowed_modes)
        raise ValueError(f"{field_name} mode must be one of {allowed}")
    parts = target.strip().split(":")
    if len(parts) == 2 and parts[0] == "layer":
        return int(parts[1]), None, mode
    if len(parts) == 4 and parts[0] == "layer" and parts[2] == "kv":
        return int(parts[1]), int(parts[3]), mode
    raise ValueError(f"{field_name} entries must use layer:<id>=<mode> or layer:<id>:kv:<id>=<mode>")


def _parse_layer_value_spec(
    spec: str,
    *,
    field_name: str,
    allowed_values: tuple[str, ...],
) -> tuple[int, str]:
    if "=" not in spec:
        raise ValueError(f"{field_name} entries must use layer:<id>=<value>")
    target, value = spec.split("=", 1)
    parts = target.strip().split(":")
    if len(parts) != 2 or parts[0] != "layer":
        raise ValueError(f"{field_name} entries must use layer:<id>=<value>")
    value = value.strip()
    if value not in allowed_values:
        allowed = ", ".join(allowed_values)
        raise ValueError(f"{field_name} values must be one of {allowed}")
    return int(parts[1]), value


def _parse_layer_candidate_spec(spec: str, *, field_name: str) -> tuple[int, tuple[PageModeSpec, ...]]:
    if "=" not in spec:
        raise ValueError(f"{field_name} entries must use layer:<id>=MODE/SCHEME/BITS[,MODE/SCHEME/BITS...]")
    target, raw_candidates = spec.split("=", 1)
    parts = target.strip().split(":")
    if len(parts) != 2 or parts[0] != "layer":
        raise ValueError(f"{field_name} entries must use layer:<id>=MODE/SCHEME/BITS[,MODE/SCHEME/BITS...]")
    candidates = tuple(
        parse_page_mode_token(token.strip())
        for token in raw_candidates.split(",")
        if token.strip()
    )
    if not candidates:
        raise ValueError(f"{field_name} entries must include at least one candidate")
    return int(parts[1]), candidates


def _parse_layer_positive_int_spec(spec: str, *, field_name: str) -> tuple[int, int]:
    if "=" not in spec:
        raise ValueError(f"{field_name} entries must use layer:<id>=<positive_int>")
    target, value = spec.split("=", 1)
    parts = target.strip().split(":")
    if len(parts) != 2 or parts[0] != "layer":
        raise ValueError(f"{field_name} entries must use layer:<id>=<positive_int>")
    parsed_value = int(value.strip())
    if parsed_value <= 0:
        raise ValueError(f"{field_name} values must be positive integers")
    return int(parts[1]), parsed_value


def _parse_layer_context_positive_int_spec(spec: str, *, field_name: str) -> tuple[int, int, int]:
    if "=" not in spec:
        raise ValueError(f"{field_name} entries must use layer:<id>:min_ctx:<non_negative_int>=<positive_int>")
    target, value = spec.split("=", 1)
    parts = target.strip().split(":")
    if len(parts) != 4 or parts[0] != "layer" or parts[2] != "min_ctx":
        raise ValueError(f"{field_name} entries must use layer:<id>:min_ctx:<non_negative_int>=<positive_int>")
    min_context = int(parts[3])
    if min_context < 0:
        raise ValueError(f"{field_name} min_ctx must be non-negative")
    parsed_value = int(value.strip())
    if parsed_value <= 0:
        raise ValueError(f"{field_name} values must be positive integers")
    return int(parts[1]), min_context, parsed_value


@dataclass(frozen=True, slots=True)
class DotCacheConfig:
    head_dim: int
    group_size: int = 32
    bits_k: int = 4
    bits_v: int = 4
    tokens_per_page: int = 64
    recent_window: int = 128
    sink_window: int = 0
    execution_recent_window: int = 0
    execution_sink_window: int = 0
    execution_recent_window_overrides: tuple[str, ...] = ()
    execution_recent_window_context_overrides: tuple[str, ...] = ()
    execution_relevance_top_k: int = 0
    execution_relevance_mode: str = "envelope"
    execution_relevance_top_k_overrides: tuple[str, ...] = ()
    execution_relevance_top_k_context_overrides: tuple[str, ...] = ()
    execution_full_context_layers: tuple[int, ...] = ()
    execution_disable_grouped_batching_layers: tuple[int, ...] = ()
    execution_recent_old_bonus_window: int = 0
    execution_recent_old_bonus_strength: float = 0.0
    execution_recent_old_bonus_layers: tuple[int, ...] = ()
    execution_secondary_relevance_mode: str = ""
    execution_secondary_relevance_top_k: int = 0
    execution_secondary_relevance_min_overlap: float = 0.0
    execution_secondary_relevance_layers: tuple[int, ...] = ()
    execution_recent_neighbor_rescue_top_k: int = 0
    execution_recent_neighbor_rescue_anchor_window: int = 0
    execution_recent_neighbor_rescue_min_anchor_pages: int = 0
    execution_recent_neighbor_rescue_layers: tuple[int, ...] = ()
    execution_exact_promote_top_k: int = 0
    execution_exact_promote_min_margin_threshold: float = 0.0
    execution_exact_promote_max_context: int = 0
    execution_exact_promote_margin_threshold: float = 0.0
    execution_exact_promote_layers: tuple[int, ...] = ()
    execution_exact_promote_union_rescue_top_k: int = 0
    execution_grouped_decode_compact: bool = False
    execution_grouped_mix_compact: bool = False
    execution_grouped_mix_disable_packed_cuda: bool = False
    execution_freeze_chunk_budget_during_decode: bool = False
    execution_builtin_selector_cache: bool = False
    execution_builtin_selector_score_all_pages: bool = False
    execution_builtin_selector_candidate_only: bool = False
    execution_builtin_selector_score_all_pages_min_candidate_fraction: float = 0.0
    execution_value_escape_layers: tuple[int, ...] = ()
    execution_value_escape_mode: str = "M3"
    execution_value_escape_old_only: bool = False
    execution_value_escape_top_k: int = 0
    execution_value_escape_prewarm: bool = False
    execution_value_escape_prewarm_min_context: int = 0
    execution_exact_refine_top_k: int = 0
    execution_exact_refine_layers: tuple[int, ...] = ()
    store_scales_dtype: str = "float16"
    store_bias_dtype: str = "float16"
    payload_layout_k: str = "group_major"
    payload_layout_v: str = "group_major"
    default_mode_k: str = "M0"
    default_mode_v: str = "M0"
    quant_scheme_k: str = "affine"
    quant_scheme_v: str = "affine"
    escape_dtype: str = "float16"
    recent_page_escape_dtype: str = "float16"
    m2_sketch_dim_k: int = 8
    m4_project_basis_k: str = "hadamard"
    m4_project_basis_k_overrides: tuple[str, ...] = ()
    m4_project_dim_k_overrides: tuple[str, ...] = ()
    m2_center_k: bool = False
    m2_segment_count_k: int = 1
    m2_adaptive_segments_k: bool = False
    m2_adaptive_min_improvement_k: float = 0.1
    m2_prefilter_top_k: int = 0
    m2_prefilter_min_pages: int = 8
    prefer_m4_project_k: bool = False
    lut_refine_steps: int = 6
    preconditioner: str = "none"
    precondition_strength: float = 2.0
    m1_segment_count_k: int = 1
    m1_segment_count_v: int = 1
    m1_fallback_to_m0: bool = True
    m1_error_threshold: float = 0.35
    m1_token_p95_error_threshold: float = 1000000.0
    prepared_chunk_cache_budget_ratio: float = 0.5
    prepared_chunk_cache_min_bytes: int = 1 * 1024 * 1024
    prepared_chunk_cache_max_bytes: int = 64 * 1024 * 1024
    key_mode_overrides: tuple[str, ...] = ()
    value_mode_overrides: tuple[str, ...] = ()
    key_policy_tier: str = "exact"
    value_policy_tier: str = "exact"
    key_layer_sensitivity: tuple[str, ...] = ()
    value_layer_sensitivity: tuple[str, ...] = ()
    key_policy_overrides: tuple[str, ...] = ()
    value_policy_overrides: tuple[str, ...] = ()
    learned_page_selector_path: str | None = None
    learned_page_selector_prompt_family: str | None = None
    learned_page_selector_prompt_variant: str | None = None
    learned_page_selector_profile: str = "quality"
    learned_page_selector_scope: str = "KV"
    learned_page_selector_target_candidate: str = "M3/affine/4/float16"
    learned_page_selector_logit_offset: float = 0.0

    def __post_init__(self) -> None:
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.bits_k not in (2, 3, 4, 8):
            raise ValueError("bits_k must be 2, 3, 4, or 8 for the current runtime")
        if self.bits_v not in (2, 3, 4, 8):
            raise ValueError("bits_v must be 2, 3, 4, or 8 for the current runtime")
        if self.tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be positive")
        if self.execution_recent_window < 0:
            raise ValueError("execution_recent_window must be non-negative")
        if self.execution_sink_window < 0:
            raise ValueError("execution_sink_window must be non-negative")
        for spec in self.execution_recent_window_overrides:
            _parse_layer_positive_int_spec(spec, field_name="execution_recent_window_overrides")
        for spec in self.execution_recent_window_context_overrides:
            _parse_layer_context_positive_int_spec(spec, field_name="execution_recent_window_context_overrides")
        if self.execution_relevance_top_k < 0:
            raise ValueError("execution_relevance_top_k must be non-negative")
        if self.execution_relevance_mode not in ("sketch", "envelope"):
            raise ValueError("execution_relevance_mode must be sketch or envelope")
        for spec in self.execution_relevance_top_k_overrides:
            _parse_layer_positive_int_spec(spec, field_name="execution_relevance_top_k_overrides")
        for spec in self.execution_relevance_top_k_context_overrides:
            _parse_layer_context_positive_int_spec(spec, field_name="execution_relevance_top_k_context_overrides")
        for layer_id in self.execution_full_context_layers:
            if int(layer_id) < 0:
                raise ValueError("execution_full_context_layers must be non-negative")
        for layer_id in self.execution_disable_grouped_batching_layers:
            if int(layer_id) < 0:
                raise ValueError("execution_disable_grouped_batching_layers must be non-negative")
        if self.execution_recent_old_bonus_window < 0:
            raise ValueError("execution_recent_old_bonus_window must be non-negative")
        if self.execution_recent_old_bonus_strength < 0:
            raise ValueError("execution_recent_old_bonus_strength must be non-negative")
        for layer_id in self.execution_recent_old_bonus_layers:
            if int(layer_id) < 0:
                raise ValueError("execution_recent_old_bonus_layers must be non-negative")
        if self.execution_secondary_relevance_mode not in ("", "sketch", "envelope"):
            raise ValueError("execution_secondary_relevance_mode must be empty, sketch, or envelope")
        if self.execution_secondary_relevance_top_k < 0:
            raise ValueError("execution_secondary_relevance_top_k must be non-negative")
        if not 0.0 <= float(self.execution_secondary_relevance_min_overlap) <= 1.0:
            raise ValueError("execution_secondary_relevance_min_overlap must be between 0 and 1")
        for layer_id in self.execution_secondary_relevance_layers:
            if int(layer_id) < 0:
                raise ValueError("execution_secondary_relevance_layers must be non-negative")
        if self.execution_recent_neighbor_rescue_top_k < 0:
            raise ValueError("execution_recent_neighbor_rescue_top_k must be non-negative")
        if self.execution_recent_neighbor_rescue_anchor_window < 0:
            raise ValueError("execution_recent_neighbor_rescue_anchor_window must be non-negative")
        if self.execution_recent_neighbor_rescue_min_anchor_pages < 0:
            raise ValueError("execution_recent_neighbor_rescue_min_anchor_pages must be non-negative")
        for layer_id in self.execution_recent_neighbor_rescue_layers:
            if int(layer_id) < 0:
                raise ValueError("execution_recent_neighbor_rescue_layers must be non-negative")
        if self.execution_exact_promote_top_k < 0:
            raise ValueError("execution_exact_promote_top_k must be non-negative")
        if self.execution_exact_promote_min_margin_threshold < 0:
            raise ValueError("execution_exact_promote_min_margin_threshold must be non-negative")
        if self.execution_exact_promote_max_context < 0:
            raise ValueError("execution_exact_promote_max_context must be non-negative")
        if self.execution_exact_promote_margin_threshold < 0:
            raise ValueError("execution_exact_promote_margin_threshold must be non-negative")
        for layer_id in self.execution_exact_promote_layers:
            if int(layer_id) < 0:
                raise ValueError("execution_exact_promote_layers must be non-negative")
        if self.execution_exact_promote_union_rescue_top_k < 0:
            raise ValueError("execution_exact_promote_union_rescue_top_k must be non-negative")
        for layer_id in self.execution_value_escape_layers:
            if int(layer_id) < 0:
                raise ValueError("execution_value_escape_layers must be non-negative")
        if self.execution_value_escape_mode not in _VALID_VALUE_MODES:
            allowed = ", ".join(_VALID_VALUE_MODES)
            raise ValueError(f"execution_value_escape_mode must be one of {allowed}")
        if self.execution_exact_refine_top_k < 0:
            raise ValueError("execution_exact_refine_top_k must be non-negative")
        for layer_id in self.execution_exact_refine_layers:
            if int(layer_id) < 0:
                raise ValueError("execution_exact_refine_layers must be non-negative")
        if self.payload_layout_k not in ("group_major", "token_major"):
            raise ValueError("payload_layout_k must be group_major or token_major")
        if self.payload_layout_v not in ("group_major", "token_major"):
            raise ValueError("payload_layout_v must be group_major or token_major")
        if self.default_mode_k not in _VALID_KEY_MODES:
            raise ValueError("default_mode_k must be M0, M1, M2, M3, M4, or T3")
        if self.default_mode_v not in _VALID_VALUE_MODES:
            raise ValueError("default_mode_v must be M0, M1, M3, or T3")
        if self.quant_scheme_k not in ("affine", "symmetric", "lut", "sketch", "project", "turbo3"):
            raise ValueError("quant_scheme_k must be affine, symmetric, lut, sketch, project, or turbo3")
        if self.quant_scheme_v not in ("affine", "symmetric", "lut", "turbo3"):
            raise ValueError("quant_scheme_v must be affine, symmetric, lut, or turbo3")
        if self.escape_dtype not in ("float16", "float32", "int8"):
            raise ValueError("escape_dtype must be float16, float32, or int8")
        if self.recent_page_escape_dtype not in ("float16", "float32", "int8"):
            raise ValueError("recent_page_escape_dtype must be float16, float32, or int8")
        if self.m2_sketch_dim_k <= 0:
            raise ValueError("m2_sketch_dim_k must be positive")
        if self.m4_project_basis_k not in _VALID_M4_BASIS_FAMILIES:
            allowed = ", ".join(_VALID_M4_BASIS_FAMILIES)
            raise ValueError(f"m4_project_basis_k must be one of {allowed}")
        for spec in self.m4_project_basis_k_overrides:
            _parse_layer_value_spec(
                spec,
                field_name="m4_project_basis_k_overrides",
                allowed_values=_VALID_M4_BASIS_FAMILIES,
            )
        for spec in self.m4_project_dim_k_overrides:
            _parse_layer_positive_int_spec(spec, field_name="m4_project_dim_k_overrides")
        if not isinstance(self.m2_center_k, bool):
            raise ValueError("m2_center_k must be a bool")
        if self.m2_segment_count_k <= 0:
            raise ValueError("m2_segment_count_k must be positive")
        if not isinstance(self.m2_adaptive_segments_k, bool):
            raise ValueError("m2_adaptive_segments_k must be a bool")
        if self.m2_adaptive_min_improvement_k < 0:
            raise ValueError("m2_adaptive_min_improvement_k must be non-negative")
        if self.m2_prefilter_top_k < 0:
            raise ValueError("m2_prefilter_top_k must be non-negative")
        if self.m2_prefilter_min_pages < 0:
            raise ValueError("m2_prefilter_min_pages must be non-negative")
        if self.lut_refine_steps < 0:
            raise ValueError("lut_refine_steps must be non-negative")
        if self.preconditioner not in ("none", "tanh"):
            raise ValueError("preconditioner must be none or tanh")
        if self.precondition_strength <= 0:
            raise ValueError("precondition_strength must be positive")
        if self.m1_segment_count_k <= 0:
            raise ValueError("m1_segment_count_k must be positive")
        if self.m1_segment_count_v <= 0:
            raise ValueError("m1_segment_count_v must be positive")
        if self.m1_error_threshold <= 0:
            raise ValueError("m1_error_threshold must be positive")
        if self.m1_token_p95_error_threshold <= 0:
            raise ValueError("m1_token_p95_error_threshold must be positive")
        if self.prepared_chunk_cache_budget_ratio < 0:
            raise ValueError("prepared_chunk_cache_budget_ratio must be non-negative")
        if self.prepared_chunk_cache_min_bytes < 0:
            raise ValueError("prepared_chunk_cache_min_bytes must be non-negative")
        if self.prepared_chunk_cache_max_bytes < 0:
            raise ValueError("prepared_chunk_cache_max_bytes must be non-negative")
        if (
            self.prepared_chunk_cache_max_bytes > 0
            and self.prepared_chunk_cache_min_bytes > self.prepared_chunk_cache_max_bytes
        ):
            raise ValueError("prepared_chunk_cache_min_bytes must not exceed prepared_chunk_cache_max_bytes")
        for spec in self.key_mode_overrides:
            _parse_mode_override_spec(spec, allowed_modes=_VALID_KEY_MODES, field_name="key_mode_overrides")
        for spec in self.value_mode_overrides:
            _parse_mode_override_spec(spec, allowed_modes=_VALID_VALUE_MODES, field_name="value_mode_overrides")
        for field_name, tier in (("key_policy_tier", self.key_policy_tier), ("value_policy_tier", self.value_policy_tier)):
            if tier not in ("exact", "strict", "balanced", "aggressive"):
                raise ValueError(f"{field_name} must be exact, strict, balanced, or aggressive")
        for spec in self.key_layer_sensitivity:
            _parse_layer_value_spec(
                spec,
                field_name="key_layer_sensitivity",
                allowed_values=("strict", "balanced", "aggressive"),
            )
        for spec in self.value_layer_sensitivity:
            _parse_layer_value_spec(
                spec,
                field_name="value_layer_sensitivity",
                allowed_values=("strict", "balanced", "aggressive"),
            )
        for spec in self.key_policy_overrides:
            _parse_layer_candidate_spec(spec, field_name="key_policy_overrides")
        for spec in self.value_policy_overrides:
            _parse_layer_candidate_spec(spec, field_name="value_policy_overrides")
        if self.learned_page_selector_path is not None and not str(self.learned_page_selector_path).strip():
            raise ValueError("learned_page_selector_path must be a non-empty string when provided")
        if self.learned_page_selector_prompt_family is not None and not str(self.learned_page_selector_prompt_family).strip():
            raise ValueError("learned_page_selector_prompt_family must be a non-empty string when provided")
        if self.learned_page_selector_prompt_variant is not None and not str(self.learned_page_selector_prompt_variant).strip():
            raise ValueError("learned_page_selector_prompt_variant must be a non-empty string when provided")
        if str(self.learned_page_selector_profile) not in {"quality", "systems", "manual"}:
            raise ValueError("learned_page_selector_profile must be quality, systems, or manual")
        if str(self.learned_page_selector_scope) not in {"KV", "K", "V"}:
            raise ValueError("learned_page_selector_scope must be KV, K, or V")
        if not str(self.learned_page_selector_target_candidate).strip():
            raise ValueError("learned_page_selector_target_candidate must be a non-empty string")
        if not math.isfinite(float(self.learned_page_selector_logit_offset)):
            raise ValueError("learned_page_selector_logit_offset must be finite")

    @property
    def num_groups(self) -> int:
        return ceil(self.head_dim / self.group_size)

    @property
    def padded_head_dim(self) -> int:
        return self.num_groups * self.group_size

    def has_mode_overrides(self, *, kind: str | None = None) -> bool:
        if kind == "K":
            return bool(self.key_mode_overrides)
        if kind == "V":
            return bool(self.value_mode_overrides)
        return bool(self.key_mode_overrides or self.value_mode_overrides)

    def has_policy_overrides(self, *, kind: str | None = None) -> bool:
        if kind == "K":
            return bool(self.key_layer_sensitivity or self.key_policy_overrides or self.key_policy_tier != "exact")
        if kind == "V":
            return bool(self.value_layer_sensitivity or self.value_policy_overrides or self.value_policy_tier != "exact")
        return bool(
            self.key_layer_sensitivity
            or self.value_layer_sensitivity
            or self.key_policy_overrides
            or self.value_policy_overrides
            or self.key_policy_tier != "exact"
            or self.value_policy_tier != "exact"
        )

    def learned_page_selector_enabled(self) -> bool:
        return self.learned_page_selector_path is not None and bool(str(self.learned_page_selector_path).strip())

    def learned_page_selector_applies_to_kind(self, *, kind: str) -> bool:
        scope = str(self.learned_page_selector_scope)
        if scope == "KV":
            return kind in {"K", "V"}
        return str(kind) == scope

    def resolve_page_mode(self, *, kind: str, layer_id: int, kv_head_id: int) -> str:
        if kind == "K":
            resolved = self.default_mode_k
            specs = self.key_mode_overrides
            allowed_modes = _VALID_KEY_MODES
            field_name = "key_mode_overrides"
        elif kind == "V":
            resolved = self.default_mode_v
            specs = self.value_mode_overrides
            allowed_modes = _VALID_VALUE_MODES
            field_name = "value_mode_overrides"
        else:
            raise ValueError("kind must be K or V")
        for spec in specs:
            override_layer_id, override_kv_head_id, override_mode = _parse_mode_override_spec(
                spec,
                allowed_modes=allowed_modes,
                field_name=field_name,
            )
            if override_layer_id != int(layer_id):
                continue
            if override_kv_head_id is not None and override_kv_head_id != int(kv_head_id):
                continue
            resolved = override_mode
        return resolved

    def resolve_m4_project_dim_k(self, *, layer_id: int) -> int:
        resolved = int(self.m2_sketch_dim_k)
        for spec in self.m4_project_dim_k_overrides:
            override_layer_id, override_dim = _parse_layer_positive_int_spec(
                spec,
                field_name="m4_project_dim_k_overrides",
            )
            if override_layer_id == int(layer_id):
                resolved = int(override_dim)
        return resolved

    def resolve_execution_relevance_top_k(self, *, layer_id: int) -> int:
        resolved = int(self.execution_relevance_top_k)
        for spec in self.execution_relevance_top_k_overrides:
            override_layer_id, override_value = _parse_layer_positive_int_spec(
                spec,
                field_name="execution_relevance_top_k_overrides",
            )
            if override_layer_id == int(layer_id):
                resolved = int(override_value)
        return resolved

    def resolve_execution_recent_window(self, *, layer_id: int) -> int:
        resolved = int(self.execution_recent_window)
        for spec in self.execution_recent_window_overrides:
            override_layer_id, override_value = _parse_layer_positive_int_spec(
                spec,
                field_name="execution_recent_window_overrides",
            )
            if override_layer_id == int(layer_id):
                resolved = int(override_value)
        return resolved

    def execution_shortlist_enabled(self) -> bool:
        return (
            self.execution_recent_window > 0
            or self.execution_sink_window > 0
            or bool(self.execution_recent_window_overrides)
            or bool(self.execution_recent_window_context_overrides)
            or self.execution_relevance_top_k > 0
            or bool(self.execution_relevance_top_k_overrides)
            or bool(self.execution_relevance_top_k_context_overrides)
        )

    def execution_shortlist_disabled_for_layer(self, *, layer_id: int) -> bool:
        return int(layer_id) in {int(value) for value in self.execution_full_context_layers}

    def execution_grouped_batching_disabled_for_layer(self, *, layer_id: int) -> bool:
        return int(layer_id) in {int(value) for value in self.execution_disable_grouped_batching_layers}

    def execution_value_escape_enabled_for_layer(self, *, layer_id: int) -> bool:
        if not self.execution_value_escape_layers:
            return False
        return int(layer_id) in {int(value) for value in self.execution_value_escape_layers}

    def execution_recent_old_bonus_enabled_for_layer(self, *, layer_id: int) -> bool:
        if self.execution_recent_old_bonus_window <= 0 or self.execution_recent_old_bonus_strength <= 0:
            return False
        if not self.execution_recent_old_bonus_layers:
            return False
        return int(layer_id) in {int(value) for value in self.execution_recent_old_bonus_layers}

    def execution_secondary_relevance_enabled_for_layer(self, *, layer_id: int) -> bool:
        if self.execution_secondary_relevance_mode not in ("sketch", "envelope"):
            return False
        if self.execution_secondary_relevance_top_k <= 0:
            return False
        if not self.execution_secondary_relevance_layers:
            return False
        return int(layer_id) in {int(value) for value in self.execution_secondary_relevance_layers}

    def execution_recent_neighbor_rescue_enabled_for_layer(self, *, layer_id: int) -> bool:
        if self.execution_recent_neighbor_rescue_top_k <= 0:
            return False
        if self.execution_recent_neighbor_rescue_anchor_window <= 0:
            return False
        if self.execution_recent_neighbor_rescue_min_anchor_pages <= 0:
            return False
        if not self.execution_recent_neighbor_rescue_layers:
            return False
        return int(layer_id) in {int(value) for value in self.execution_recent_neighbor_rescue_layers}

    def resolve_execution_relevance_top_k_for_context(self, *, layer_id: int, context_length: int | None = None) -> int:
        resolved = self.resolve_execution_relevance_top_k(layer_id=layer_id)
        if context_length is None:
            return resolved
        best_min_context = -1
        for spec in self.execution_relevance_top_k_context_overrides:
            override_layer_id, min_context, override_value = _parse_layer_context_positive_int_spec(
                spec,
                field_name="execution_relevance_top_k_context_overrides",
            )
            if override_layer_id != int(layer_id):
                continue
            if int(context_length) < int(min_context) or int(min_context) < best_min_context:
                continue
            resolved = int(override_value)
            best_min_context = int(min_context)
        return resolved

    def resolve_execution_recent_window_for_context(self, *, layer_id: int, context_length: int | None = None) -> int:
        resolved = self.resolve_execution_recent_window(layer_id=layer_id)
        if context_length is None:
            return resolved
        best_min_context = -1
        for spec in self.execution_recent_window_context_overrides:
            override_layer_id, min_context, override_value = _parse_layer_context_positive_int_spec(
                spec,
                field_name="execution_recent_window_context_overrides",
            )
            if override_layer_id != int(layer_id):
                continue
            if int(context_length) < int(min_context) or int(min_context) < best_min_context:
                continue
            resolved = int(override_value)
            best_min_context = int(min_context)
        return resolved

    def resolve_m4_project_basis_k(self, *, layer_id: int) -> str:
        resolved = self.m4_project_basis_k
        for spec in self.m4_project_basis_k_overrides:
            override_layer_id, override_basis = _parse_layer_value_spec(
                spec,
                field_name="m4_project_basis_k_overrides",
                allowed_values=_VALID_M4_BASIS_FAMILIES,
            )
            if override_layer_id == int(layer_id):
                resolved = override_basis
        return resolved

    def resolve_layer_policy(self, *, kind: str, layer_id: int, kv_head_id: int) -> LayerPolicy:
        if kind == "K":
            default_mode = self.default_mode_k
            default_bits = self.bits_k
            default_quant_scheme = self.quant_scheme_k
            default_tier = self.key_policy_tier
            sensitivity_specs = self.key_layer_sensitivity
            explicit_specs = self.key_policy_overrides
            mode_overrides = self.key_mode_overrides
        elif kind == "V":
            default_mode = self.default_mode_v
            default_bits = self.bits_v
            default_quant_scheme = self.quant_scheme_v
            default_tier = self.value_policy_tier
            sensitivity_specs = self.value_layer_sensitivity
            explicit_specs = self.value_policy_overrides
            mode_overrides = self.value_mode_overrides
        else:
            raise ValueError("kind must be K or V")

        resolved_mode = self.resolve_page_mode(kind=kind, layer_id=layer_id, kv_head_id=kv_head_id)
        if resolved_mode != default_mode:
            override_scheme = (
                "lut" if resolved_mode == "M1"
                else "sketch" if resolved_mode == "M2"
                else "project" if resolved_mode == "M4"
                else "turbo3" if resolved_mode == "T3"
                else default_quant_scheme
            )
            return make_explicit_policy(
                kind=kind,
                policy_id=f"{kind.lower()}_mode_override_layer_{int(layer_id)}",
                sensitivity_tier="exact",
                candidates=(PageModeSpec(mode=resolved_mode, bits=default_bits, quant_scheme=override_scheme),),
                recent_escape_dtype=self.recent_page_escape_dtype,
                recent_window=0,
            )

        for spec in explicit_specs:
            override_layer_id, candidates = _parse_layer_candidate_spec(spec, field_name="key_policy_overrides" if kind == "K" else "value_policy_overrides")
            if override_layer_id == int(layer_id):
                return make_explicit_policy(
                    kind=kind,
                    policy_id=f"{kind.lower()}_policy_override_layer_{int(layer_id)}",
                    sensitivity_tier="balanced",
                    candidates=candidates,
                    recent_escape_dtype=self.recent_page_escape_dtype,
                    recent_window=self.recent_window,
                )

        tier = default_tier
        for spec in sensitivity_specs:
            override_layer_id, override_tier = _parse_layer_value_spec(
                spec,
                field_name="key_layer_sensitivity" if kind == "K" else "value_layer_sensitivity",
                allowed_values=("strict", "balanced", "aggressive"),
            )
            if override_layer_id == int(layer_id):
                tier = override_tier
        return make_tier_candidates(
            kind=kind,
            sensitivity_tier=tier,
            default_bits=default_bits,
            default_quant_scheme=default_quant_scheme,
            default_mode=default_mode,
            recent_escape_dtype=self.recent_page_escape_dtype,
            recent_window=self.recent_window,
            prefer_project_key_mode=self.prefer_m4_project_k if kind == "K" else False,
        )
