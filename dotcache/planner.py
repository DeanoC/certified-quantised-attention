from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

ModeName = Literal["M0", "M1", "M2", "M3", "M4", "T3"]
QuantSchemeName = Literal["affine", "symmetric", "lut", "sketch", "project", "turbo3"]
SensitivityTier = Literal["exact", "strict", "balanced", "aggressive"]


@dataclass(frozen=True, slots=True)
class PageStats:
    token_count: int
    rms: float
    abs_max: float
    outlier_fraction: float
    channel_range_mean: float


@dataclass(frozen=True, slots=True)
class PageModeSpec:
    mode: ModeName
    bits: int
    quant_scheme: QuantSchemeName
    escape_dtype: str | None = None
    policy_id: str = "exact_baseline"
    sensitivity_tier: SensitivityTier = "exact"
    fallback_reason: str = ""
    age_bucket: str = "aged"


@dataclass(frozen=True, slots=True)
class LayerPolicy:
    policy_id: str
    sensitivity_tier: SensitivityTier
    kind: str
    candidates: tuple[PageModeSpec, ...]
    recent_candidate: PageModeSpec | None = None
    recent_window: int = 128
    outlier_fraction_threshold: float = 0.05
    abs_max_threshold: float = 6.0
    channel_range_threshold: float = 4.0


def observe_page(tensor_slice: np.ndarray) -> PageStats:
    values = np.asarray(tensor_slice, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("tensor_slice must have shape [token_count, head_dim]")
    if values.shape[0] == 0:
        return PageStats(
            token_count=0,
            rms=0.0,
            abs_max=0.0,
            outlier_fraction=0.0,
            channel_range_mean=0.0,
        )

    abs_values = np.abs(values, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float32), dtype=np.float64)))
    abs_max = float(np.max(abs_values))
    outlier_cutoff = max(3.0 * max(rms, 1e-6), 6.0)
    outlier_fraction = float(np.mean(abs_values >= outlier_cutoff, dtype=np.float64))
    channel_range_mean = float(np.mean(np.max(values, axis=0) - np.min(values, axis=0), dtype=np.float64))
    return PageStats(
        token_count=int(values.shape[0]),
        rms=rms,
        abs_max=abs_max,
        outlier_fraction=outlier_fraction,
        channel_range_mean=channel_range_mean,
    )


def choose_page_mode(
    layer: int,
    kind: str,
    token_age: int,
    page_stats: PageStats | None,
    *,
    layer_policy: LayerPolicy,
) -> PageModeSpec:
    del layer
    if token_age < int(layer_policy.recent_window):
        recent_candidate = layer_policy.recent_candidate
        if recent_candidate is not None:
            return PageModeSpec(
                mode=recent_candidate.mode,
                bits=recent_candidate.bits,
                quant_scheme=recent_candidate.quant_scheme,
                escape_dtype=recent_candidate.escape_dtype,
                policy_id=layer_policy.policy_id,
                sensitivity_tier=layer_policy.sensitivity_tier,
                fallback_reason="recent_window",
                age_bucket="recent",
            )
        return PageModeSpec(
            mode="M3",
            bits=layer_policy.candidates[-1].bits if layer_policy.candidates else 4,
            quant_scheme=layer_policy.candidates[-1].quant_scheme if layer_policy.candidates else "affine",
            escape_dtype=None,
            policy_id=layer_policy.policy_id,
            sensitivity_tier=layer_policy.sensitivity_tier,
            fallback_reason="recent_window",
            age_bucket="recent",
        )

    if not layer_policy.candidates:
        fallback_mode: ModeName = "M0" if kind == "K" else "M0"
        return PageModeSpec(
            mode=fallback_mode,
            bits=4,
            quant_scheme="affine",
            policy_id=layer_policy.policy_id,
            sensitivity_tier=layer_policy.sensitivity_tier,
            fallback_reason="no_candidates",
            age_bucket="aged",
        )

    stats = page_stats
    failure_reasons: list[str] = []
    for index, candidate in enumerate(layer_policy.candidates):
        if _candidate_is_allowed(candidate, kind=kind, stats=stats, policy=layer_policy):
            fallback_reason = "" if index == 0 else "+".join(failure_reasons) or "fallback"
            return PageModeSpec(
                mode=candidate.mode,
                bits=candidate.bits,
                quant_scheme=candidate.quant_scheme,
                escape_dtype=candidate.escape_dtype,
                policy_id=layer_policy.policy_id,
                sensitivity_tier=layer_policy.sensitivity_tier,
                fallback_reason=fallback_reason,
                age_bucket="aged",
            )
        failure_reasons.append(f"{candidate.mode.lower()}_stats")

    safest = layer_policy.candidates[-1]
    return PageModeSpec(
        mode=safest.mode,
        bits=safest.bits,
        quant_scheme=safest.quant_scheme,
        escape_dtype=safest.escape_dtype,
        policy_id=layer_policy.policy_id,
        sensitivity_tier=layer_policy.sensitivity_tier,
        fallback_reason="+".join(failure_reasons) if failure_reasons else "threshold_fallback",
        age_bucket="aged",
    )


def choose_mode(
    layer: int,
    head: int,
    token_age: int,
    stats: dict[str, float | bool] | None = None,
    *,
    recent_window: int = 128,
    error_threshold: float | None = None,
) -> str:
    del layer
    del head

    if token_age < recent_window:
        return "M3"

    if stats is None:
        return "M0"

    if bool(stats.get("force_escape", False)):
        return "M3"

    quant_error = float(stats.get("quant_error", 0.0))
    if error_threshold is not None and quant_error > error_threshold:
        return "M3"

    return "M0"


def parse_page_mode_token(token: str) -> PageModeSpec:
    parts = [part.strip() for part in token.split("/") if part.strip()]
    if len(parts) not in (3, 4):
        raise ValueError("page mode tokens must use MODE/SCHEME/BITS[/ESCAPE_DTYPE], for example M0/affine/4 or M3/affine/4/int8")
    mode_text, scheme_text, bits_text = parts[:3]
    mode = mode_text.upper()
    if mode not in {"M0", "M1", "M2", "M3", "M4", "T3"}:
        raise ValueError(f"unsupported page mode: {mode_text}")
    quant_scheme = scheme_text.lower()
    if quant_scheme not in {"affine", "symmetric", "lut", "sketch", "project", "turbo3"}:
        raise ValueError(f"unsupported quant scheme: {scheme_text}")
    bits = int(bits_text)
    escape_dtype = None
    if len(parts) == 4:
        escape_dtype = parts[3].lower()
        if escape_dtype not in {"float16", "float32", "int8"}:
            raise ValueError(f"unsupported escape dtype: {parts[3]}")
        if mode != "M3":
            raise ValueError("escape dtype qualifiers are only supported for M3 page modes")
    return PageModeSpec(mode=mode, bits=bits, quant_scheme=quant_scheme, escape_dtype=escape_dtype)


def make_tier_candidates(
    *,
    kind: str,
    sensitivity_tier: SensitivityTier,
    default_bits: int,
    default_quant_scheme: str,
    default_mode: str,
    recent_window: int,
    recent_escape_dtype: str = "float16",
    prefer_project_key_mode: bool = False,
) -> LayerPolicy:
    def candidate(mode: ModeName, scheme: QuantSchemeName, bits: int) -> PageModeSpec:
        return PageModeSpec(mode=mode, quant_scheme=scheme, bits=bits, sensitivity_tier=sensitivity_tier)

    exact_mode = candidate(
        default_mode if default_mode in {"M0", "M1", "M2", "M3", "M4", "T3"} else "M0",
        default_quant_scheme if default_quant_scheme in {"affine", "symmetric", "lut", "sketch", "project", "turbo3"} else "affine",
        default_bits,
    )
    if sensitivity_tier == "exact":
        candidates = (exact_mode,)
        thresholds = (0.0, float("inf"), float("inf"))
    elif kind == "K":
        if sensitivity_tier == "strict":
            candidates = (candidate("M0", "affine", 4),)
            thresholds = (0.02, 4.5, 3.0)
        elif sensitivity_tier == "aggressive":
            candidates = (
                candidate("M0", "affine", 2),
                candidate("M4", "project", 4) if prefer_project_key_mode else candidate("M2", "sketch", 4),
                candidate("M0", "affine", 4),
            )
            thresholds = (0.10, 8.0, 5.5)
        else:
            candidates = (
                candidate("M0", "affine", 2),
                candidate("M4", "project", 4) if prefer_project_key_mode else candidate("M2", "sketch", 4),
                candidate("M0", "affine", 4),
            )
            thresholds = (0.05, 6.0, 4.0)
    else:
        if sensitivity_tier == "strict":
            candidates = (
                candidate("M1", "lut", 4),
                candidate("M0", "affine", 4),
            )
            thresholds = (0.02, 4.5, 3.0)
        elif sensitivity_tier == "aggressive":
            candidates = (
                candidate("M0", "affine", 2),
                candidate("M0", "affine", 3),
                candidate("M1", "lut", 4),
                candidate("M0", "affine", 4),
            )
            thresholds = (0.10, 8.0, 5.5)
        else:
            candidates = (
                candidate("M0", "affine", 3),
                candidate("M1", "lut", 4),
                candidate("M0", "affine", 4),
            )
            thresholds = (0.05, 6.0, 4.0)
    return LayerPolicy(
        policy_id=f"{kind.lower()}_{sensitivity_tier}",
        sensitivity_tier=sensitivity_tier,
        kind=kind,
        candidates=candidates,
        recent_candidate=PageModeSpec(
            mode="M3",
            bits=default_bits,
            quant_scheme="affine",
            escape_dtype=recent_escape_dtype,
            sensitivity_tier=sensitivity_tier,
        ),
        recent_window=0 if sensitivity_tier == "exact" else recent_window,
        outlier_fraction_threshold=float(thresholds[0]),
        abs_max_threshold=float(thresholds[1]),
        channel_range_threshold=float(thresholds[2]),
    )


def make_explicit_policy(
    *,
    kind: str,
    policy_id: str,
    sensitivity_tier: SensitivityTier,
    candidates: Sequence[PageModeSpec],
    recent_window: int,
    recent_escape_dtype: str = "float16",
) -> LayerPolicy:
    if not candidates:
        raise ValueError("explicit policies must provide at least one candidate")
    recent_candidate = next((candidate for candidate in candidates if candidate.mode == "M3"), None)
    if recent_candidate is None:
        recent_candidate = PageModeSpec(
            mode="M3",
            bits=4,
            quant_scheme="affine",
            escape_dtype=recent_escape_dtype,
            sensitivity_tier=sensitivity_tier,
        )
    return LayerPolicy(
        policy_id=policy_id,
        sensitivity_tier=sensitivity_tier,
        kind=kind,
        candidates=tuple(candidates),
        recent_candidate=recent_candidate,
        recent_window=recent_window,
    )


def _candidate_is_allowed(
    candidate: PageModeSpec,
    *,
    kind: str,
    stats: PageStats | None,
    policy: LayerPolicy,
) -> bool:
    if stats is None or candidate.mode == "M0" and candidate.bits >= 4:
        return True
    if candidate.mode == "M3":
        return True
    if candidate.mode == "M2":
        if kind != "K":
            return False
        return (
            stats.outlier_fraction <= policy.outlier_fraction_threshold
            and stats.channel_range_mean <= policy.channel_range_threshold
        )
    if candidate.mode == "M4":
        if kind != "K":
            return False
        return (
            stats.outlier_fraction <= policy.outlier_fraction_threshold
            and stats.channel_range_mean <= policy.channel_range_threshold
        )
    if candidate.mode == "M1":
        if kind != "V":
            return False
        return (
            stats.outlier_fraction <= policy.outlier_fraction_threshold
            and stats.channel_range_mean <= policy.channel_range_threshold
        )
    if candidate.mode == "T3":
        return False
    if candidate.mode == "M0" and candidate.bits <= 2:
        return (
            stats.outlier_fraction <= policy.outlier_fraction_threshold
            and stats.abs_max <= policy.abs_max_threshold
        )
    if candidate.mode == "M0" and candidate.bits == 3:
        return (
            stats.outlier_fraction <= (policy.outlier_fraction_threshold * 1.25)
            and stats.abs_max <= (policy.abs_max_threshold * 1.2)
        )
    return True
