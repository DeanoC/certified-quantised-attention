from __future__ import annotations

import fcntl
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import DotCacheConfig
from .decode_reference import decode_page
from .encode import encode_page
from .planner import PageModeSpec, observe_page, parse_page_mode_token
from .types import Kind


@dataclass(slots=True)
class PageTraceRecord:
    source: str
    kind: Kind
    layer_id: int
    kv_head_id: int
    token_start: int
    token_age: int
    values: np.ndarray
    query: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("values must have shape [token_count, head_dim]")
        self.values = values
        if self.query is not None:
            query = np.asarray(self.query, dtype=np.float32)
            if query.ndim != 1:
                raise ValueError("query must have shape [head_dim]")
            if int(query.shape[0]) != int(values.shape[1]):
                raise ValueError("query head_dim must match values head_dim")
            self.query = query
        self.layer_id = int(self.layer_id)
        self.kv_head_id = int(self.kv_head_id)
        self.token_start = int(self.token_start)
        self.token_age = int(self.token_age)
        if self.token_age < 0:
            raise ValueError("token_age must be non-negative")

    @property
    def token_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def head_dim(self) -> int:
        return int(self.values.shape[1])

    @property
    def stats(self):  # pragma: no cover - thin wrapper
        return observe_page(self.values)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "source": self.source,
            "kind": self.kind,
            "layer_id": self.layer_id,
            "kv_head_id": self.kv_head_id,
            "token_start": self.token_start,
            "token_age": self.token_age,
            "token_count": self.token_count,
            "head_dim": self.head_dim,
            "notes": list(self.notes),
            "stats": asdict(self.stats),
        }
        if self.query is not None:
            payload["query_present"] = True
        return payload


@dataclass(frozen=True, slots=True)
class OracleThresholds:
    max_mean_abs_error_ratio: float = 0.10
    max_max_abs_error_ratio: float = 1.00
    max_token_p95_error_ratio: float = 0.25
    max_score_max_abs_error: float | None = None
    min_score_topk_agreement: float | None = None


@dataclass(slots=True)
class OracleCandidateResult:
    candidate: str
    mode: str
    bits: int
    quant_scheme: str
    payload_bytes: int
    metadata_bytes: int
    total_bytes: int
    mean_abs_error: float
    max_abs_error: float
    token_p95_error: float
    score_max_abs_error: float | None
    score_topk_agreement: float | None
    safe: bool
    failure_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OracleReplayResult:
    trace: dict[str, Any]
    thresholds: dict[str, Any]
    candidates: list[OracleCandidateResult]
    cheapest_safe_candidate: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace": dict(self.trace),
            "thresholds": dict(self.thresholds),
            "cheapest_safe_candidate": self.cheapest_safe_candidate,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(slots=True)
class OracleBatchTraceResult:
    trace_path: str
    stage: str
    trace: dict[str, Any]
    cheapest_safe_candidate: str | None
    safe_candidate_count: int
    best_safe_total_bytes: int | None
    candidates: list[OracleCandidateResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_path": self.trace_path,
            "stage": self.stage,
            "trace": dict(self.trace),
            "cheapest_safe_candidate": self.cheapest_safe_candidate,
            "safe_candidate_count": self.safe_candidate_count,
            "best_safe_total_bytes": self.best_safe_total_bytes,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(slots=True)
class OracleBatchReplayResult:
    manifest: dict[str, Any]
    sampling: dict[str, Any]
    thresholds: dict[str, Any]
    group_size: int
    tokens_per_page: int | None
    selected_trace_count: int
    selected_trace_paths: list[str]
    selected_trace_counts_by_stage: dict[str, int]
    selected_trace_counts_by_kind: dict[str, int]
    cheapest_safe_candidate_histogram: dict[str, int]
    failure_reason_histogram: dict[str, int]
    candidate_stats: dict[str, dict[str, Any]]
    summary_table_markdown: str
    traces: list[OracleBatchTraceResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": dict(self.manifest),
            "sampling": dict(self.sampling),
            "thresholds": dict(self.thresholds),
            "group_size": self.group_size,
            "tokens_per_page": self.tokens_per_page,
            "selected_trace_count": self.selected_trace_count,
            "selected_trace_paths": list(self.selected_trace_paths),
            "selected_trace_counts_by_stage": dict(self.selected_trace_counts_by_stage),
            "selected_trace_counts_by_kind": dict(self.selected_trace_counts_by_kind),
            "cheapest_safe_candidate_histogram": dict(self.cheapest_safe_candidate_histogram),
            "failure_reason_histogram": dict(self.failure_reason_histogram),
            "candidate_stats": dict(self.candidate_stats),
            "summary_table_markdown": self.summary_table_markdown,
            "traces": [trace.to_dict() for trace in self.traces],
        }


@dataclass(slots=True)
class OracleLabelRecord:
    trace_path: str
    stage: str
    prompt_family: str | None
    prompt_variant: str | None
    source: str
    kind: str
    layer_id: int
    kv_head_id: int
    token_start: int
    token_age: int
    token_count: int
    head_dim: int
    query_present: bool
    cheapest_safe_candidate: str | None
    safe_candidates: list[str]
    best_safe_total_bytes: int | None
    candidate_labels: list[dict[str, Any]]
    trace_stats: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_path": self.trace_path,
            "stage": self.stage,
            "prompt_family": self.prompt_family,
            "prompt_variant": self.prompt_variant,
            "source": self.source,
            "kind": self.kind,
            "layer_id": self.layer_id,
            "kv_head_id": self.kv_head_id,
            "token_start": self.token_start,
            "token_age": self.token_age,
            "token_count": self.token_count,
            "head_dim": self.head_dim,
            "query_present": self.query_present,
            "cheapest_safe_candidate": self.cheapest_safe_candidate,
            "safe_candidates": list(self.safe_candidates),
            "best_safe_total_bytes": self.best_safe_total_bytes,
            "candidate_labels": list(self.candidate_labels),
            "trace_stats": dict(self.trace_stats),
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class OracleLabelingResult:
    labels: list[OracleLabelRecord]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": dict(self.summary),
            "labels": [label.to_dict() for label in self.labels],
        }


@dataclass(slots=True)
class OracleSelectorTrainingRow:
    trace_path: str
    source: str
    stage: str
    prompt_family: str | None
    prompt_variant: str | None
    kind: str
    layer_id: int
    layer_fraction: float
    kv_head_id: int
    kv_head_fraction: float
    token_start: int
    token_age: int
    token_count: int
    head_dim: int
    query_present: bool
    safe_candidate_count: int
    best_safe_total_bytes: int | None
    target_candidate: str | None
    target_present: bool
    trace_rms: float
    trace_abs_max: float
    trace_channel_range_mean: float
    trace_outlier_fraction: float
    age_per_token: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OracleSelectorCandidateTrainingRow:
    trace_path: str
    source: str
    stage: str
    prompt_family: str | None
    prompt_variant: str | None
    kind: str
    layer_id: int
    layer_fraction: float
    kv_head_id: int
    kv_head_fraction: float
    token_start: int
    token_age: int
    token_count: int
    head_dim: int
    query_present: bool
    safe_candidate_count: int
    best_safe_total_bytes: int | None
    target_candidate: str | None
    target_present: bool
    trace_rms: float
    trace_abs_max: float
    trace_channel_range_mean: float
    trace_outlier_fraction: float
    age_per_token: float
    candidate: str
    candidate_mode: str
    candidate_bits: int
    candidate_quant_scheme: str
    candidate_total_bytes: int
    candidate_payload_bytes: int
    candidate_metadata_bytes: int
    candidate_has_escape_dtype: bool
    candidate_safe: bool
    candidate_is_target: bool
    candidate_bytes_over_best_safe: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OracleDatasetSplitSummary:
    split_name: str
    holdout_prompt_families: list[str]
    holdout_prompt_variants: list[str]
    holdout_layers: list[int]
    train_trace_paths: list[str]
    test_trace_paths: list[str]
    train_label_count: int
    test_label_count: int
    train_selector_row_count: int
    test_selector_row_count: int
    train_selector_candidate_row_count: int
    test_selector_candidate_row_count: int
    train_prompt_family_histogram: dict[str, int]
    test_prompt_family_histogram: dict[str, int]
    train_prompt_variant_histogram: dict[str, int]
    test_prompt_variant_histogram: dict[str, int]
    train_layer_histogram: dict[str, int]
    test_layer_histogram: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OracleDatasetSplitManifestEntry:
    split_name: str
    split_dir: str
    split_summary_path: str
    holdout_prompt_families: list[str]
    holdout_prompt_variants: list[str]
    holdout_layers: list[int]
    train_label_count: int
    test_label_count: int
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OracleDatasetSplitSuiteSpec:
    split_name: str
    output_subdir: str | None = None
    holdout_prompt_families: list[str] = field(default_factory=list)
    holdout_prompt_variants: list[str] = field(default_factory=list)
    holdout_layers: list[int] = field(default_factory=list)
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OracleDatasetSplitSuiteResult:
    suite_name: str
    output_root: str
    manifest_path: str | None
    split_count: int
    split_names: list[str]
    split_dirs: list[str]
    splits: list[OracleDatasetSplitSummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "output_root": self.output_root,
            "manifest_path": self.manifest_path,
            "split_count": self.split_count,
            "split_names": list(self.split_names),
            "split_dirs": list(self.split_dirs),
            "splits": [split.to_dict() for split in self.splits],
        }


def default_candidate_specs(kind: Kind) -> tuple[PageModeSpec, ...]:
    if kind == "K":
        tokens = (
            "M0/affine/2",
            "M0/affine/3",
            "M0/affine/4",
            "M2/sketch/4",
            "M4/project/4",
            "M3/affine/4/float16",
            "T3/turbo3/3",
        )
    else:
        tokens = (
            "M0/affine/2",
            "M0/affine/3",
            "M0/affine/4",
            "M1/lut/4",
            "M3/affine/4/float16",
            "T3/turbo3/3",
        )
    return tuple(parse_page_mode_token(token) for token in tokens)


def save_page_trace(record: PageTraceRecord, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "values": np.asarray(record.values, dtype=np.float32),
        "metadata_json": np.array(json.dumps(record.to_dict()), dtype=np.str_),
    }
    if record.query is not None:
        arrays["query"] = np.asarray(record.query, dtype=np.float32)
    np.savez_compressed(target, **arrays)


def load_page_trace(path: str | Path) -> PageTraceRecord:
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
        query = payload["query"] if "query" in payload else None
        return PageTraceRecord(
            source=str(metadata.get("source", "")),
            kind=str(metadata["kind"]),
            layer_id=int(metadata["layer_id"]),
            kv_head_id=int(metadata["kv_head_id"]),
            token_start=int(metadata["token_start"]),
            token_age=int(metadata["token_age"]),
            values=np.asarray(payload["values"], dtype=np.float32),
            query=None if query is None else np.asarray(query, dtype=np.float32),
            notes=list(metadata.get("notes", [])),
        )


def _config_for_candidate(
    base_config: DotCacheConfig,
    *,
    kind: Kind,
    candidate: PageModeSpec,
) -> DotCacheConfig:
    if kind == "K":
        return replace(
            base_config,
            bits_k=int(candidate.bits),
            default_mode_k=str(candidate.mode),
            quant_scheme_k=str(candidate.quant_scheme),
            escape_dtype=str(candidate.escape_dtype or base_config.escape_dtype),
        )
    return replace(
        base_config,
        bits_v=int(candidate.bits),
        default_mode_v=str(candidate.mode),
        quant_scheme_v=str(candidate.quant_scheme),
        escape_dtype=str(candidate.escape_dtype or base_config.escape_dtype),
    )


def _score_metrics(
    values: np.ndarray,
    reconstructed: np.ndarray,
    query: np.ndarray | None,
) -> tuple[float | None, float | None]:
    if query is None:
        return None, None
    dense_scores = values @ query
    approx_scores = reconstructed @ query
    score_max_abs_error = float(np.max(np.abs(dense_scores - approx_scores)))
    top_k = min(4, int(values.shape[0]))
    dense_top = set(np.argsort(dense_scores)[-top_k:].tolist())
    approx_top = set(np.argsort(approx_scores)[-top_k:].tolist())
    topk_agreement = float(len(dense_top & approx_top) / max(top_k, 1))
    return score_max_abs_error, topk_agreement


def evaluate_page_candidate(
    record: PageTraceRecord,
    *,
    base_config: DotCacheConfig,
    candidate: PageModeSpec,
    thresholds: OracleThresholds,
) -> OracleCandidateResult:
    candidate_config = _config_for_candidate(base_config, kind=record.kind, candidate=candidate)
    encoded = encode_page(
        record.values,
        candidate_config,
        kind=record.kind,
        layer_id=record.layer_id,
        kv_head_id=record.kv_head_id,
        token_start=record.token_start,
        page_mode=candidate,
    )
    reconstructed = decode_page(encoded)
    abs_error = np.abs(record.values - reconstructed)
    token_max_error = np.max(abs_error, axis=1)
    page_rms = max(float(record.stats.rms), 1e-6)
    mean_abs_error = float(np.mean(abs_error))
    max_abs_error = float(np.max(abs_error))
    token_p95_error = float(np.percentile(token_max_error, 95))
    score_max_abs_error, score_topk_agreement = _score_metrics(record.values, reconstructed, record.query)

    failure_reasons: list[str] = []
    if mean_abs_error > thresholds.max_mean_abs_error_ratio * page_rms:
        failure_reasons.append("mean_abs_error")
    if max_abs_error > thresholds.max_max_abs_error_ratio * page_rms:
        failure_reasons.append("max_abs_error")
    if token_p95_error > thresholds.max_token_p95_error_ratio * page_rms:
        failure_reasons.append("token_p95_error")
    if thresholds.max_score_max_abs_error is not None and score_max_abs_error is not None:
        if score_max_abs_error > float(thresholds.max_score_max_abs_error):
            failure_reasons.append("score_max_abs_error")
    if thresholds.min_score_topk_agreement is not None and score_topk_agreement is not None:
        if score_topk_agreement < float(thresholds.min_score_topk_agreement):
            failure_reasons.append("score_topk_agreement")

    return OracleCandidateResult(
        candidate=_format_candidate_token(candidate),
        mode=str(candidate.mode),
        bits=int(candidate.bits),
        quant_scheme=str(candidate.quant_scheme),
        payload_bytes=int(encoded.payload_nbytes),
        metadata_bytes=int(encoded.metadata_nbytes),
        total_bytes=int(encoded.total_nbytes),
        mean_abs_error=mean_abs_error,
        max_abs_error=max_abs_error,
        token_p95_error=token_p95_error,
        score_max_abs_error=score_max_abs_error,
        score_topk_agreement=score_topk_agreement,
        safe=not failure_reasons,
        failure_reasons=failure_reasons,
    )


def run_oracle_replay(
    record: PageTraceRecord,
    *,
    base_config: DotCacheConfig,
    candidates: Sequence[PageModeSpec] | None = None,
    thresholds: OracleThresholds | None = None,
) -> OracleReplayResult:
    resolved_thresholds = thresholds or OracleThresholds()
    resolved_candidates = tuple(candidates or default_candidate_specs(record.kind))
    candidate_results = [
        evaluate_page_candidate(
            record,
            base_config=base_config,
            candidate=candidate,
            thresholds=resolved_thresholds,
        )
        for candidate in resolved_candidates
    ]
    safe_candidates = [candidate for candidate in candidate_results if candidate.safe]
    safe_candidates.sort(key=lambda item: (item.total_bytes, item.mean_abs_error, item.max_abs_error))
    return OracleReplayResult(
        trace=record.to_dict(),
        thresholds=asdict(resolved_thresholds),
        candidates=candidate_results,
        cheapest_safe_candidate=safe_candidates[0].candidate if safe_candidates else None,
    )


def load_page_trace_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_page_trace_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def merge_page_trace_manifests(
    manifests: Sequence[dict[str, Any] | str | Path],
    *,
    output_dir: str | Path | None = None,
    source: str = "merged_page_trace_manifest",
) -> dict[str, Any]:
    resolved_manifests = [
        load_page_trace_manifest(manifest) if isinstance(manifest, (str, Path)) else dict(manifest)
        for manifest in manifests
    ]
    page_trace_paths: list[str] = []
    counts_by_kind: dict[str, int] = {}
    counts_by_stage: dict[str, int] = {}
    counts_by_layer: dict[str, int] = {}
    tokens_per_page_values: set[int] = set()
    kinds_union: set[str] = set()
    member_output_dirs: list[str] = []
    member_sources: list[str] = []
    for manifest in resolved_manifests:
        page_trace_paths.extend(str(path) for path in manifest.get("page_trace_paths", []))
        for key, value in dict(manifest.get("page_trace_counts_by_kind", {})).items():
            counts_by_kind[str(key)] = counts_by_kind.get(str(key), 0) + int(value)
        for key, value in dict(manifest.get("page_trace_counts_by_stage", {})).items():
            counts_by_stage[str(key)] = counts_by_stage.get(str(key), 0) + int(value)
        for key, value in dict(manifest.get("page_trace_counts_by_layer", {})).items():
            counts_by_layer[str(key)] = counts_by_layer.get(str(key), 0) + int(value)
        if "tokens_per_page" in manifest:
            tokens_per_page_values.add(int(manifest["tokens_per_page"]))
        for kind in manifest.get("kinds", []):
            kinds_union.add(str(kind))
        if "output_dir" in manifest:
            member_output_dirs.append(str(manifest["output_dir"]))
        if "source" in manifest:
            member_sources.append(str(manifest["source"]))

    merged_manifest = {
        "output_dir": None if output_dir is None else str(output_dir),
        "page_trace_count": len(page_trace_paths),
        "page_trace_paths": page_trace_paths,
        "page_trace_counts_by_kind": dict(sorted(counts_by_kind.items())),
        "page_trace_counts_by_stage": dict(sorted(counts_by_stage.items())),
        "page_trace_counts_by_layer": dict(sorted(counts_by_layer.items())),
        "tokens_per_page": None if len(tokens_per_page_values) != 1 else next(iter(tokens_per_page_values)),
        "tokens_per_page_values": sorted(tokens_per_page_values),
        "kinds": sorted(kinds_union),
        "source": source,
        "member_output_dirs": member_output_dirs,
        "member_sources": sorted(set(member_sources)),
        "member_manifest_count": len(resolved_manifests),
    }
    return merged_manifest


def select_page_trace_paths(
    manifest: dict[str, Any] | str | Path,
    *,
    max_traces: int | None = None,
    max_per_stage_kind: int | None = None,
    seed: int = 0,
    kinds: Sequence[str] | None = None,
    stages: Sequence[str] | None = None,
    layer_ids: Sequence[int] | None = None,
) -> list[str]:
    manifest_payload = load_page_trace_manifest(manifest) if isinstance(manifest, (str, Path)) else dict(manifest)
    trace_paths = [str(path) for path in manifest_payload.get("page_trace_paths", [])]
    normalized_kinds = None if kinds is None else {str(kind).upper() for kind in kinds}
    normalized_stages = None if stages is None else {str(stage).lower() for stage in stages}
    normalized_layers = None if layer_ids is None else {int(layer_id) for layer_id in layer_ids}

    rng = np.random.default_rng(int(seed))
    grouped_matches: dict[tuple[str, str], list[str]] = {}
    for trace_path in trace_paths:
        record = load_page_trace(trace_path)
        stage = _trace_stage(record)
        kind = str(record.kind).upper()
        if normalized_kinds is not None and kind not in normalized_kinds:
            continue
        if normalized_stages is not None and stage not in normalized_stages:
            continue
        if normalized_layers is not None and int(record.layer_id) not in normalized_layers:
            continue
        grouped_matches.setdefault((stage, kind), []).append(str(trace_path))

    selected_paths: list[str] = []
    for group_key in sorted(grouped_matches):
        group_paths = list(grouped_matches[group_key])
        if len(group_paths) > 1:
            order = rng.permutation(len(group_paths)).tolist()
            group_paths = [group_paths[index] for index in order]
        if max_per_stage_kind is not None:
            group_paths = group_paths[: max(int(max_per_stage_kind), 0)]
        selected_paths.extend(group_paths)

    if len(selected_paths) > 1:
        order = rng.permutation(len(selected_paths)).tolist()
        selected_paths = [selected_paths[index] for index in order]
    if max_traces is not None:
        selected_paths = selected_paths[: max(int(max_traces), 0)]
    return selected_paths


def run_oracle_batch_replay(
    manifest: dict[str, Any] | str | Path,
    *,
    group_size: int,
    tokens_per_page: int | None = None,
    candidates: Sequence[PageModeSpec] | None = None,
    thresholds: OracleThresholds | None = None,
    max_traces: int | None = None,
    max_per_stage_kind: int | None = None,
    seed: int = 0,
    kinds: Sequence[str] | None = None,
    stages: Sequence[str] | None = None,
    layer_ids: Sequence[int] | None = None,
) -> OracleBatchReplayResult:
    manifest_payload = load_page_trace_manifest(manifest) if isinstance(manifest, (str, Path)) else dict(manifest)
    resolved_thresholds = thresholds or OracleThresholds()
    selected_paths = select_page_trace_paths(
        manifest_payload,
        max_traces=max_traces,
        max_per_stage_kind=max_per_stage_kind,
        seed=seed,
        kinds=kinds,
        stages=stages,
        layer_ids=layer_ids,
    )

    traces: list[OracleBatchTraceResult] = []
    counts_by_stage: dict[str, int] = {}
    counts_by_kind: dict[str, int] = {}
    selected_histogram: dict[str, int] = {}
    failure_reason_histogram: dict[str, int] = {}
    candidate_stats_raw: dict[str, dict[str, float]] = {}

    for trace_path in selected_paths:
        record = load_page_trace(trace_path)
        stage = _trace_stage(record)
        counts_by_stage[stage] = counts_by_stage.get(stage, 0) + 1
        counts_by_kind[record.kind] = counts_by_kind.get(record.kind, 0) + 1
        replay = run_oracle_replay(
            record,
            base_config=DotCacheConfig(
                head_dim=record.head_dim,
                group_size=int(group_size),
                tokens_per_page=int(tokens_per_page or record.token_count),
            ),
            candidates=candidates,
            thresholds=resolved_thresholds,
        )
        safe_candidates = [candidate for candidate in replay.candidates if candidate.safe]
        if replay.cheapest_safe_candidate is not None:
            selected_histogram[replay.cheapest_safe_candidate] = selected_histogram.get(replay.cheapest_safe_candidate, 0) + 1
        for candidate in replay.candidates:
            stats = candidate_stats_raw.setdefault(
                candidate.candidate,
                {
                    "count": 0.0,
                    "safe_count": 0.0,
                    "sum_total_bytes": 0.0,
                    "sum_mean_abs_error": 0.0,
                    "sum_token_p95_error": 0.0,
                },
            )
            stats["count"] += 1.0
            stats["safe_count"] += 1.0 if candidate.safe else 0.0
            stats["sum_total_bytes"] += float(candidate.total_bytes)
            stats["sum_mean_abs_error"] += float(candidate.mean_abs_error)
            stats["sum_token_p95_error"] += float(candidate.token_p95_error)
            for reason in candidate.failure_reasons:
                failure_reason_histogram[reason] = failure_reason_histogram.get(reason, 0) + 1
        traces.append(
            OracleBatchTraceResult(
                trace_path=str(trace_path),
                stage=stage,
                trace=replay.trace,
                cheapest_safe_candidate=replay.cheapest_safe_candidate,
                safe_candidate_count=len(safe_candidates),
                best_safe_total_bytes=min((candidate.total_bytes for candidate in safe_candidates), default=None),
                candidates=replay.candidates,
            )
        )

    candidate_stats: dict[str, dict[str, Any]] = {}
    for candidate, stats in sorted(candidate_stats_raw.items()):
        count = max(int(stats["count"]), 1)
        candidate_stats[candidate] = {
            "count": int(stats["count"]),
            "safe_count": int(stats["safe_count"]),
            "safe_rate": float(stats["safe_count"] / count),
            "mean_total_bytes": float(stats["sum_total_bytes"] / count),
            "mean_abs_error": float(stats["sum_mean_abs_error"] / count),
            "mean_token_p95_error": float(stats["sum_token_p95_error"] / count),
            "selected_count": int(selected_histogram.get(candidate, 0)),
        }

    return OracleBatchReplayResult(
        manifest=manifest_payload,
        sampling={
            "seed": int(seed),
            "max_traces": None if max_traces is None else int(max_traces),
            "max_per_stage_kind": None if max_per_stage_kind is None else int(max_per_stage_kind),
            "kinds": None if kinds is None else [str(kind).upper() for kind in kinds],
            "stages": None if stages is None else [str(stage).lower() for stage in stages],
            "layer_ids": None if layer_ids is None else [int(layer_id) for layer_id in layer_ids],
        },
        thresholds=asdict(resolved_thresholds),
        group_size=int(group_size),
        tokens_per_page=None if tokens_per_page is None else int(tokens_per_page),
        selected_trace_count=len(traces),
        selected_trace_paths=list(selected_paths),
        selected_trace_counts_by_stage=dict(sorted(counts_by_stage.items())),
        selected_trace_counts_by_kind=dict(sorted(counts_by_kind.items())),
        cheapest_safe_candidate_histogram=dict(sorted(selected_histogram.items())),
        failure_reason_histogram=dict(sorted(failure_reason_histogram.items())),
        candidate_stats=candidate_stats,
        summary_table_markdown=_render_batch_summary_table(candidate_stats),
        traces=traces,
    )


def build_oracle_label_records(batch_result: OracleBatchReplayResult) -> list[OracleLabelRecord]:
    labels: list[OracleLabelRecord] = []
    for trace_result in batch_result.traces:
        safe_candidates = sorted(
            (candidate for candidate in trace_result.candidates if candidate.safe),
            key=lambda item: (item.total_bytes, item.mean_abs_error, item.max_abs_error, item.candidate),
        )
        trace = dict(trace_result.trace)
        labels.append(
            OracleLabelRecord(
                trace_path=str(trace_result.trace_path),
                stage=str(trace_result.stage),
                prompt_family=_infer_prompt_family_from_trace_path(trace_result.trace_path),
                prompt_variant=_infer_prompt_variant_from_trace_path(trace_result.trace_path),
                source=str(trace.get("source", "")),
                kind=str(trace["kind"]),
                layer_id=int(trace["layer_id"]),
                kv_head_id=int(trace["kv_head_id"]),
                token_start=int(trace["token_start"]),
                token_age=int(trace["token_age"]),
                token_count=int(trace["token_count"]),
                head_dim=int(trace["head_dim"]),
                query_present=bool(trace.get("query_present", False)),
                cheapest_safe_candidate=trace_result.cheapest_safe_candidate,
                safe_candidates=[candidate.candidate for candidate in safe_candidates],
                best_safe_total_bytes=trace_result.best_safe_total_bytes,
                candidate_labels=[candidate.to_dict() for candidate in trace_result.candidates],
                trace_stats=dict(trace.get("stats", {})),
                notes=list(trace.get("notes", [])),
            )
        )
    return labels


def run_oracle_labeling(
    manifest: dict[str, Any] | str | Path,
    *,
    group_size: int,
    tokens_per_page: int | None = None,
    candidates: Sequence[PageModeSpec] | None = None,
    thresholds: OracleThresholds | None = None,
    max_traces: int | None = None,
    max_per_stage_kind: int | None = None,
    seed: int = 0,
    kinds: Sequence[str] | None = None,
    stages: Sequence[str] | None = None,
    layer_ids: Sequence[int] | None = None,
) -> OracleLabelingResult:
    batch_result = run_oracle_batch_replay(
        manifest,
        group_size=group_size,
        tokens_per_page=tokens_per_page,
        candidates=candidates,
        thresholds=thresholds,
        max_traces=max_traces,
        max_per_stage_kind=max_per_stage_kind,
        seed=seed,
        kinds=kinds,
        stages=stages,
        layer_ids=layer_ids,
    )
    labels = build_oracle_label_records(batch_result)
    summary = {
        "manifest": dict(batch_result.manifest),
        "sampling": dict(batch_result.sampling),
        "thresholds": dict(batch_result.thresholds),
        "group_size": int(batch_result.group_size),
        "tokens_per_page": batch_result.tokens_per_page,
        "label_count": len(labels),
        "selected_trace_counts_by_stage": dict(batch_result.selected_trace_counts_by_stage),
        "selected_trace_counts_by_kind": dict(batch_result.selected_trace_counts_by_kind),
        "cheapest_safe_candidate_histogram": dict(batch_result.cheapest_safe_candidate_histogram),
        "failure_reason_histogram": dict(batch_result.failure_reason_histogram),
        "candidate_stats": dict(batch_result.candidate_stats),
        "summary_table_markdown": batch_result.summary_table_markdown,
    }
    return OracleLabelingResult(labels=labels, summary=summary)


def save_oracle_labels(
    labeling_result: OracleLabelingResult,
    *,
    labels_path: str | Path,
    summary_path: str | Path | None = None,
) -> None:
    labels_target = Path(labels_path)
    labels_target.parent.mkdir(parents=True, exist_ok=True)
    with labels_target.open("w", encoding="utf-8") as handle:
        for label in labeling_result.labels:
            handle.write(json.dumps(label.to_dict(), sort_keys=True) + "\n")
    if summary_path is not None:
        summary_target = Path(summary_path)
        summary_target.parent.mkdir(parents=True, exist_ok=True)
        summary_target.write_text(json.dumps(labeling_result.summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def build_selector_training_rows(labels: Sequence[OracleLabelRecord]) -> list[OracleSelectorTrainingRow]:
    if not labels:
        return []
    max_layer_id = max(label.layer_id for label in labels)
    max_kv_head_id = max(label.kv_head_id for label in labels)
    rows: list[OracleSelectorTrainingRow] = []
    for label in labels:
        stats = dict(label.trace_stats)
        rows.append(
            OracleSelectorTrainingRow(
                trace_path=str(label.trace_path),
                source=str(label.source),
                stage=str(label.stage),
                prompt_family=label.prompt_family,
                prompt_variant=label.prompt_variant,
                kind=str(label.kind),
                layer_id=int(label.layer_id),
                layer_fraction=float(label.layer_id / max(max_layer_id, 1)),
                kv_head_id=int(label.kv_head_id),
                kv_head_fraction=float(label.kv_head_id / max(max_kv_head_id, 1)),
                token_start=int(label.token_start),
                token_age=int(label.token_age),
                token_count=int(label.token_count),
                head_dim=int(label.head_dim),
                query_present=bool(label.query_present),
                safe_candidate_count=len(label.safe_candidates),
                best_safe_total_bytes=None if label.best_safe_total_bytes is None else int(label.best_safe_total_bytes),
                target_candidate=label.cheapest_safe_candidate,
                target_present=label.cheapest_safe_candidate is not None,
                trace_rms=float(stats.get("rms", 0.0)),
                trace_abs_max=float(stats.get("abs_max", 0.0)),
                trace_channel_range_mean=float(stats.get("channel_range_mean", 0.0)),
                trace_outlier_fraction=float(stats.get("outlier_fraction", 0.0)),
                age_per_token=float(label.token_age / max(label.token_count, 1)),
            )
        )
    return rows


def save_selector_training_rows(
    rows: Sequence[OracleSelectorTrainingRow],
    path: str | Path,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")


def build_selector_candidate_training_rows(labels: Sequence[OracleLabelRecord]) -> list[OracleSelectorCandidateTrainingRow]:
    if not labels:
        return []
    max_layer_id = max(label.layer_id for label in labels)
    max_kv_head_id = max(label.kv_head_id for label in labels)
    rows: list[OracleSelectorCandidateTrainingRow] = []
    for label in labels:
        stats = dict(label.trace_stats)
        for candidate_payload in label.candidate_labels:
            candidate = str(candidate_payload["candidate"])
            rows.append(
                OracleSelectorCandidateTrainingRow(
                    trace_path=str(label.trace_path),
                    source=str(label.source),
                    stage=str(label.stage),
                    prompt_family=label.prompt_family,
                    prompt_variant=label.prompt_variant,
                    kind=str(label.kind),
                    layer_id=int(label.layer_id),
                    layer_fraction=float(label.layer_id / max(max_layer_id, 1)),
                    kv_head_id=int(label.kv_head_id),
                    kv_head_fraction=float(label.kv_head_id / max(max_kv_head_id, 1)),
                    token_start=int(label.token_start),
                    token_age=int(label.token_age),
                    token_count=int(label.token_count),
                    head_dim=int(label.head_dim),
                    query_present=bool(label.query_present),
                    safe_candidate_count=len(label.safe_candidates),
                    best_safe_total_bytes=None if label.best_safe_total_bytes is None else int(label.best_safe_total_bytes),
                    target_candidate=label.cheapest_safe_candidate,
                    target_present=label.cheapest_safe_candidate is not None,
                    trace_rms=float(stats.get("rms", 0.0)),
                    trace_abs_max=float(stats.get("abs_max", 0.0)),
                    trace_channel_range_mean=float(stats.get("channel_range_mean", 0.0)),
                    trace_outlier_fraction=float(stats.get("outlier_fraction", 0.0)),
                    age_per_token=float(label.token_age / max(label.token_count, 1)),
                    candidate=candidate,
                    candidate_mode=str(candidate_payload["mode"]),
                    candidate_bits=int(candidate_payload["bits"]),
                    candidate_quant_scheme=str(candidate_payload["quant_scheme"]),
                    candidate_total_bytes=int(candidate_payload["total_bytes"]),
                    candidate_payload_bytes=int(candidate_payload["payload_bytes"]),
                    candidate_metadata_bytes=int(candidate_payload["metadata_bytes"]),
                    candidate_has_escape_dtype=len(candidate.split("/")) >= 4,
                    candidate_safe=bool(candidate_payload.get("safe", False)),
                    candidate_is_target=bool(label.cheapest_safe_candidate is not None and candidate == label.cheapest_safe_candidate),
                    candidate_bytes_over_best_safe=(
                        None
                        if label.best_safe_total_bytes is None
                        else int(candidate_payload["total_bytes"]) - int(label.best_safe_total_bytes)
                    ),
                )
            )
    return rows


def save_selector_candidate_training_rows(
    rows: Sequence[OracleSelectorCandidateTrainingRow],
    path: str | Path,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")


def load_oracle_label_records(path: str | Path) -> list[OracleLabelRecord]:
    records: list[OracleLabelRecord] = []
    for payload in _load_jsonl_records(path):
        records.append(
            OracleLabelRecord(
                trace_path=str(payload["trace_path"]),
                stage=str(payload["stage"]),
                prompt_family=None if payload.get("prompt_family") in (None, "") else str(payload.get("prompt_family")),
                prompt_variant=None if payload.get("prompt_variant") in (None, "") else str(payload.get("prompt_variant")),
                source=str(payload["source"]),
                kind=str(payload["kind"]),
                layer_id=int(payload["layer_id"]),
                kv_head_id=int(payload["kv_head_id"]),
                token_start=int(payload["token_start"]),
                token_age=int(payload["token_age"]),
                token_count=int(payload["token_count"]),
                head_dim=int(payload["head_dim"]),
                query_present=bool(payload["query_present"]),
                cheapest_safe_candidate=None if payload.get("cheapest_safe_candidate") in (None, "") else str(payload.get("cheapest_safe_candidate")),
                safe_candidates=[str(candidate) for candidate in payload.get("safe_candidates", [])],
                best_safe_total_bytes=None if payload.get("best_safe_total_bytes") is None else int(payload["best_safe_total_bytes"]),
                candidate_labels=[dict(candidate) for candidate in payload.get("candidate_labels", [])],
                trace_stats=dict(payload.get("trace_stats", {})),
                notes=[str(note) for note in payload.get("notes", [])],
            )
        )
    return records


def materialize_oracle_dataset_split(
    *,
    labels_path: str | Path,
    selector_dataset_path: str | Path,
    output_dir: str | Path,
    selector_candidate_dataset_path: str | Path | None = None,
    holdout_prompt_families: Sequence[str] | None = None,
    holdout_prompt_variants: Sequence[str] | None = None,
    holdout_layers: Sequence[int] | None = None,
    split_name: str = "heldout_split",
    manifest_path: str | Path | None = None,
    annotations: dict[str, Any] | None = None,
) -> OracleDatasetSplitSummary:
    normalized_holdout_families = {
        token
        for token in (_normalize_split_token(value) for value in (holdout_prompt_families or ()))
        if token is not None
    }
    normalized_holdout_variants = {
        token
        for token in (_normalize_split_token(value) for value in (holdout_prompt_variants or ()))
        if token is not None
    }
    normalized_holdout_layers = {int(layer) for layer in (holdout_layers or ())}
    if not normalized_holdout_families and not normalized_holdout_variants and not normalized_holdout_layers:
        raise ValueError("at least one holdout filter must be provided")

    labels = load_oracle_label_records(labels_path)
    selector_rows = _load_jsonl_records(selector_dataset_path)
    selector_candidate_rows = (
        []
        if selector_candidate_dataset_path is None
        else _load_jsonl_records(selector_candidate_dataset_path)
    )

    label_by_trace_path = {label.trace_path: label for label in labels}
    selector_trace_paths = {str(row["trace_path"]) for row in selector_rows}
    if selector_candidate_rows:
        selector_trace_paths |= {str(row["trace_path"]) for row in selector_candidate_rows}
    missing_trace_paths = sorted(selector_trace_paths - set(label_by_trace_path))
    if missing_trace_paths:
        raise ValueError(f"dataset rows are missing matching labels for trace paths: {missing_trace_paths[:3]}")

    test_trace_paths = sorted(
        trace_path
        for trace_path, label in label_by_trace_path.items()
        if _matches_holdout_filters(
            prompt_family=label.prompt_family,
            prompt_variant=label.prompt_variant,
            layer_id=label.layer_id,
            holdout_prompt_families=normalized_holdout_families,
            holdout_prompt_variants=normalized_holdout_variants,
            holdout_layers=normalized_holdout_layers,
        )
    )
    train_trace_paths = sorted(trace_path for trace_path in label_by_trace_path if trace_path not in set(test_trace_paths))
    if not train_trace_paths or not test_trace_paths:
        raise ValueError("split would produce an empty train or test partition")

    test_trace_path_set = set(test_trace_paths)
    train_trace_path_set = set(train_trace_paths)
    train_labels = [label for label in labels if label.trace_path in train_trace_path_set]
    test_labels = [label for label in labels if label.trace_path in test_trace_path_set]
    train_selector_rows = [row for row in selector_rows if str(row["trace_path"]) in train_trace_path_set]
    test_selector_rows = [row for row in selector_rows if str(row["trace_path"]) in test_trace_path_set]
    train_selector_candidate_rows = [row for row in selector_candidate_rows if str(row["trace_path"]) in train_trace_path_set]
    test_selector_candidate_rows = [row for row in selector_candidate_rows if str(row["trace_path"]) in test_trace_path_set]

    output_root = Path(output_dir)
    train_dir = output_root / "train"
    test_dir = output_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    _save_jsonl_records(train_dir / "labels.jsonl", [label.to_dict() for label in train_labels])
    _save_jsonl_records(test_dir / "labels.jsonl", [label.to_dict() for label in test_labels])
    _save_jsonl_records(train_dir / "selector_dataset.jsonl", train_selector_rows)
    _save_jsonl_records(test_dir / "selector_dataset.jsonl", test_selector_rows)
    _save_schema(train_dir / "selector_schema.json", train_selector_rows)
    _save_schema(test_dir / "selector_schema.json", test_selector_rows)
    if selector_candidate_dataset_path is not None:
        _save_jsonl_records(train_dir / "selector_candidate_dataset.jsonl", train_selector_candidate_rows)
        _save_jsonl_records(test_dir / "selector_candidate_dataset.jsonl", test_selector_candidate_rows)
        _save_schema(train_dir / "selector_candidate_schema.json", train_selector_candidate_rows)
        _save_schema(test_dir / "selector_candidate_schema.json", test_selector_candidate_rows)

    summary = OracleDatasetSplitSummary(
        split_name=str(split_name),
        holdout_prompt_families=sorted(normalized_holdout_families),
        holdout_prompt_variants=sorted(normalized_holdout_variants),
        holdout_layers=sorted(normalized_holdout_layers),
        train_trace_paths=train_trace_paths,
        test_trace_paths=test_trace_paths,
        train_label_count=len(train_labels),
        test_label_count=len(test_labels),
        train_selector_row_count=len(train_selector_rows),
        test_selector_row_count=len(test_selector_rows),
        train_selector_candidate_row_count=len(train_selector_candidate_rows),
        test_selector_candidate_row_count=len(test_selector_candidate_rows),
        train_prompt_family_histogram=_label_histogram(train_labels, field_name="prompt_family"),
        test_prompt_family_histogram=_label_histogram(test_labels, field_name="prompt_family"),
        train_prompt_variant_histogram=_label_histogram(train_labels, field_name="prompt_variant"),
        test_prompt_variant_histogram=_label_histogram(test_labels, field_name="prompt_variant"),
        train_layer_histogram=_label_histogram(train_labels, field_name="layer_id"),
        test_layer_histogram=_label_histogram(test_labels, field_name="layer_id"),
    )
    (output_root / "split_summary.json").write_text(json.dumps(summary.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if manifest_path is not None:
        upsert_oracle_dataset_split_manifest_entry(
            manifest_path,
            split_dir=output_root,
            summary=summary,
            annotations=annotations,
        )
    return summary


def materialize_oracle_dataset_split_suite(
    *,
    labels_path: str | Path,
    selector_dataset_path: str | Path,
    output_root: str | Path,
    suite_specs: Sequence[OracleDatasetSplitSuiteSpec | dict[str, Any]],
    selector_candidate_dataset_path: str | Path | None = None,
    suite_name: str = "selector_split_suite",
    manifest_path: str | Path | None = None,
    overwrite_manifest: bool = True,
) -> OracleDatasetSplitSuiteResult:
    resolved_specs = [_coerce_split_suite_spec(spec) for spec in suite_specs]
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    if manifest_path is not None and overwrite_manifest:
        save_oracle_dataset_split_manifest(
            {
                "manifest_version": 1,
                "suite_name": str(suite_name),
                "split_count": 0,
                "splits": [],
            },
            manifest_path,
        )

    split_summaries: list[OracleDatasetSplitSummary] = []
    split_dirs: list[str] = []
    for spec in resolved_specs:
        split_dir = output_root_path / (spec.output_subdir or spec.split_name)
        summary = materialize_oracle_dataset_split(
            labels_path=labels_path,
            selector_dataset_path=selector_dataset_path,
            selector_candidate_dataset_path=selector_candidate_dataset_path,
            output_dir=split_dir,
            holdout_prompt_families=tuple(spec.holdout_prompt_families) or None,
            holdout_prompt_variants=tuple(spec.holdout_prompt_variants) or None,
            holdout_layers=tuple(spec.holdout_layers) or None,
            split_name=spec.split_name,
            manifest_path=manifest_path,
            annotations=spec.annotations,
        )
        split_summaries.append(summary)
        split_dirs.append(str(split_dir))

    result = OracleDatasetSplitSuiteResult(
        suite_name=str(suite_name),
        output_root=str(output_root_path),
        manifest_path=None if manifest_path is None else str(manifest_path),
        split_count=len(split_summaries),
        split_names=[summary.split_name for summary in split_summaries],
        split_dirs=split_dirs,
        splits=split_summaries,
    )
    (output_root_path / "split_suite_summary.json").write_text(
        json.dumps(result.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _format_candidate_token(candidate: PageModeSpec) -> str:
    token = f"{candidate.mode}/{candidate.quant_scheme}/{candidate.bits}"
    if candidate.escape_dtype is not None:
        token = f"{token}/{candidate.escape_dtype}"
    return token


def _trace_stage(record: PageTraceRecord) -> str:
    for note in record.notes:
        if note.startswith("stage="):
            return note.split("=", 1)[1].lower()
    return "unknown"


def _infer_prompt_family_from_trace_path(trace_path: str | Path) -> str | None:
    for part in Path(trace_path).parts:
        if part.startswith("family-"):
            family = part.removeprefix("family-")
            if "_variant-" in family:
                family, _ = family.split("_variant-", 1)
            elif "_prompt" in family:
                family, _ = family.split("_prompt", 1)
            return family or None
    return None


def _infer_prompt_variant_from_trace_path(trace_path: str | Path) -> str | None:
    for part in Path(trace_path).parts:
        if "_variant-" in part:
            _, rest = part.split("_variant-", 1)
            variant, *_ = rest.split("_prompt", 1)
            return variant
    return None


def _load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def _save_jsonl_records(path: str | Path, records: Sequence[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _save_schema(path: str | Path, records: Sequence[dict[str, Any]]) -> None:
    schema = {
        "row_count": len(records),
        "fields": sorted(records[0].keys()) if records else [],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(schema, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _normalize_split_token(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value).strip().lower() or None


def _coerce_split_suite_spec(spec: OracleDatasetSplitSuiteSpec | dict[str, Any]) -> OracleDatasetSplitSuiteSpec:
    if isinstance(spec, OracleDatasetSplitSuiteSpec):
        return spec
    payload = dict(spec)
    split_name = str(payload["split_name"])
    output_subdir = payload.get("output_subdir")
    return OracleDatasetSplitSuiteSpec(
        split_name=split_name,
        output_subdir=None if output_subdir in (None, "") else str(output_subdir),
        holdout_prompt_families=[str(item) for item in payload.get("holdout_prompt_families", [])],
        holdout_prompt_variants=[str(item) for item in payload.get("holdout_prompt_variants", [])],
        holdout_layers=[int(item) for item in payload.get("holdout_layers", [])],
        annotations=dict(payload.get("annotations", {})),
    )


def _matches_holdout_filters(
    *,
    prompt_family: str | None,
    prompt_variant: str | None,
    layer_id: int,
    holdout_prompt_families: set[str],
    holdout_prompt_variants: set[str],
    holdout_layers: set[int],
) -> bool:
    if holdout_prompt_families and _normalize_split_token(prompt_family) not in holdout_prompt_families:
        return False
    if holdout_prompt_variants and _normalize_split_token(prompt_variant) not in holdout_prompt_variants:
        return False
    if holdout_layers and int(layer_id) not in holdout_layers:
        return False
    return True


def _label_histogram(labels: Sequence[OracleLabelRecord], *, field_name: str) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for label in labels:
        raw_value = getattr(label, field_name)
        key = "null" if raw_value is None else str(raw_value)
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items()))


def load_oracle_dataset_split_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {
            "manifest_version": 1,
            "split_count": 0,
            "splits": [],
        }
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_oracle_dataset_split_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def upsert_oracle_dataset_split_manifest_entry(
    manifest_path: str | Path,
    *,
    split_dir: str | Path,
    summary: OracleDatasetSplitSummary,
    annotations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    split_root = Path(split_dir)
    entry = OracleDatasetSplitManifestEntry(
        split_name=str(summary.split_name),
        split_dir=str(split_root),
        split_summary_path=str(split_root / "split_summary.json"),
        holdout_prompt_families=list(summary.holdout_prompt_families),
        holdout_prompt_variants=list(summary.holdout_prompt_variants),
        holdout_layers=list(summary.holdout_layers),
        train_label_count=int(summary.train_label_count),
        test_label_count=int(summary.test_label_count),
        annotations={} if annotations is None else dict(annotations),
    )
    manifest_target = Path(manifest_path)
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    with manifest_target.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        payload_text = handle.read()
        locked_manifest = (
            {
                "manifest_version": 1,
                "split_count": 0,
                "splits": [],
            }
            if not payload_text.strip()
            else json.loads(payload_text)
        )

        splits = [dict(item) for item in locked_manifest.get("splits", [])]
        updated = False
        for index, payload in enumerate(splits):
            if str(payload.get("split_name")) == entry.split_name or str(payload.get("split_dir")) == entry.split_dir:
                splits[index] = entry.to_dict()
                updated = True
                break
        if not updated:
            splits.append(entry.to_dict())

        next_manifest = {
            **locked_manifest,
            "manifest_version": int(locked_manifest.get("manifest_version", 1)),
            "split_count": len(splits),
            "splits": splits,
        }
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(next_manifest, sort_keys=True, indent=2) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return next_manifest


def _render_batch_summary_table(candidate_stats: dict[str, dict[str, Any]]) -> str:
    header = "| candidate | selected | safe_count | safe_rate | mean_total_bytes | mean_abs_error | mean_token_p95_error |"
    separator = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    rows = [header, separator]
    for candidate, stats in sorted(
        candidate_stats.items(),
        key=lambda item: (-int(item[1]["selected_count"]), -float(item[1]["safe_rate"]), float(item[1]["mean_total_bytes"]), item[0]),
    ):
        rows.append(
            "| "
            + " | ".join(
                [
                    candidate,
                    str(int(stats["selected_count"])),
                    str(int(stats["safe_count"])),
                    f"{float(stats['safe_rate']):.3f}",
                    f"{float(stats['mean_total_bytes']):.1f}",
                    f"{float(stats['mean_abs_error']):.6f}",
                    f"{float(stats['mean_token_p95_error']):.6f}",
                ]
            )
            + " |"
        )
    return "\n".join(rows)
