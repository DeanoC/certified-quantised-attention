from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from math import ceil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .packing import words_per_group
from .planner import parse_page_mode_token
from .types import PageHeader

_BASE_SELECTOR_FEATURE_NAMES = (
    "stage_decode",
    "kind_key",
    "query_present",
    "layer_fraction",
    "kv_head_fraction",
    "log_sequence_length",
    "log_token_start",
    "log_token_age",
    "token_count",
    "head_dim",
    "safe_candidate_count",
    "trace_rms",
    "log_trace_abs_max",
    "trace_channel_range_mean",
    "trace_outlier_fraction",
    "age_per_token",
    "page_distance",
    "log_page_distance",
    "page_distance_ge_2",
    "page_distance_ge_4",
    "page_distance_ge_8",
    "token_end_fraction",
    "token_age_fraction",
    "age_bucket_ge_64",
    "age_bucket_ge_256",
    "age_bucket_ge_1024",
    "sequence_length_ge_512",
    "sequence_length_ge_1024",
    "sequence_length_ge_2048",
    "decode_old_page_indicator",
    "decode_long_context_indicator",
    "decode_key_indicator",
)

RUNTIME_SELECTOR_FEATURE_NAMES = (
    "stage_decode",
    "kind_key",
    "query_present",
    "layer_fraction",
    "kv_head_fraction",
    "log_sequence_length",
    "log_token_start",
    "log_token_age",
    "token_count",
    "head_dim",
    "trace_rms",
    "log_trace_abs_max",
    "trace_channel_range_mean",
    "trace_outlier_fraction",
    "age_per_token",
    "page_distance",
    "log_page_distance",
    "page_distance_ge_2",
    "page_distance_ge_4",
    "page_distance_ge_8",
    "token_end_fraction",
    "token_age_fraction",
    "age_bucket_ge_64",
    "age_bucket_ge_256",
    "age_bucket_ge_1024",
    "sequence_length_ge_512",
    "sequence_length_ge_1024",
    "sequence_length_ge_2048",
    "decode_old_page_indicator",
    "decode_long_context_indicator",
    "decode_key_indicator",
)

_BASE_CANDIDATE_FEATURE_NAMES = (
    *_BASE_SELECTOR_FEATURE_NAMES,
    "candidate_mode_m0",
    "candidate_mode_m1",
    "candidate_mode_m2",
    "candidate_mode_m3",
    "candidate_mode_m4",
    "candidate_mode_t3",
    "decode_candidate_mode_m0",
    "decode_candidate_mode_m1",
    "decode_candidate_mode_m2",
    "decode_candidate_mode_m3",
    "decode_candidate_mode_m4",
    "decode_candidate_mode_t3",
    "candidate_bits",
    "candidate_scheme_affine",
    "candidate_scheme_lut",
    "candidate_scheme_sketch",
    "candidate_scheme_project",
    "candidate_scheme_turbo3",
    "log_candidate_total_bytes",
    "log_candidate_payload_bytes",
    "log_candidate_metadata_bytes",
    "candidate_has_escape_dtype",
)

_RUNTIME_CANDIDATE_FEATURE_NAMES = (
    *RUNTIME_SELECTOR_FEATURE_NAMES,
    "candidate_mode_m0",
    "candidate_mode_m1",
    "candidate_mode_m2",
    "candidate_mode_m3",
    "candidate_mode_m4",
    "candidate_mode_t3",
    "decode_candidate_mode_m0",
    "decode_candidate_mode_m1",
    "decode_candidate_mode_m2",
    "decode_candidate_mode_m3",
    "decode_candidate_mode_m4",
    "decode_candidate_mode_t3",
    "candidate_bits",
    "candidate_scheme_affine",
    "candidate_scheme_lut",
    "candidate_scheme_sketch",
    "candidate_scheme_project",
    "candidate_scheme_turbo3",
    "log_candidate_total_bytes",
    "log_candidate_payload_bytes",
    "log_candidate_metadata_bytes",
    "candidate_has_escape_dtype",
)

_RESEARCH_SELECTOR_EXTRA_FEATURE_NAMES: tuple[str, ...] = ()

_RESEARCH_CANDIDATE_EXTRA_FEATURE_NAMES: tuple[str, ...] = ()

_PROMPT_LENGTH_RE = re.compile(r"_prompt(?P<prompt_length>\d{3,5})(?:_|$)")


@dataclass(slots=True)
class SelectorExample:
    trace_path: str
    row: dict[str, Any]
    label: dict[str, Any]
    candidate_map: dict[str, dict[str, Any]]

    @property
    def stage(self) -> str:
        return str(self.row["stage"])

    @property
    def kind(self) -> str:
        return str(self.row["kind"])

    @property
    def layer_id(self) -> int:
        return int(self.row["layer_id"])

    @property
    def token_age(self) -> int:
        return int(self.row["token_age"])

    @property
    def token_count(self) -> int:
        return int(self.row["token_count"])

    @property
    def query_present(self) -> bool:
        return bool(self.row["query_present"])

    @property
    def prompt_family(self) -> str | None:
        value = self.row.get("prompt_family")
        return None if value in (None, "") else str(value)

    @property
    def prompt_variant(self) -> str | None:
        value = self.row.get("prompt_variant")
        return None if value in (None, "") else str(value)

    @property
    def prompt_length(self) -> int | None:
        return selector_prompt_length_from_row(self.row, trace_path=self.trace_path)

    @property
    def target_candidate(self) -> str | None:
        target = self.row.get("target_candidate")
        return None if target in (None, "") else str(target)

    @property
    def best_safe_total_bytes(self) -> int | None:
        value = self.row.get("best_safe_total_bytes")
        return None if value is None else int(value)

    @property
    def safe_candidates(self) -> tuple[str, ...]:
        return tuple(str(candidate) for candidate in self.label.get("safe_candidates", []))

    @property
    def target_present(self) -> bool:
        return bool(self.row.get("target_present", self.target_candidate is not None))


@dataclass(slots=True)
class SelectorCandidateExample:
    trace_path: str
    row: dict[str, Any]

    @property
    def stage(self) -> str:
        return str(self.row["stage"])

    @property
    def kind(self) -> str:
        return str(self.row["kind"])

    @property
    def layer_id(self) -> int:
        return int(self.row["layer_id"])

    @property
    def prompt_family(self) -> str | None:
        value = self.row.get("prompt_family")
        return None if value in (None, "") else str(value)

    @property
    def candidate(self) -> str:
        return str(self.row["candidate"])

    @property
    def prompt_length(self) -> int | None:
        return selector_prompt_length_from_row(self.row, trace_path=self.trace_path)

    @property
    def candidate_safe(self) -> bool:
        return bool(self.row["candidate_safe"])

    @property
    def candidate_total_bytes(self) -> int:
        return int(self.row["candidate_total_bytes"])

    @property
    def oracle_target_candidate(self) -> str | None:
        target = self.row.get("target_candidate")
        return None if target in (None, "") else str(target)

    @property
    def best_safe_total_bytes(self) -> int | None:
        value = self.row.get("best_safe_total_bytes")
        return None if value is None else int(value)


@dataclass(frozen=True, slots=True)
class SelectorSplit:
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


@dataclass(slots=True)
class SelectorPrediction:
    trace_path: str
    predicted_candidate: str | None
    oracle_target_candidate: str | None
    correct_target: bool
    predicted_safe: bool
    predicted_total_bytes: int | None
    best_safe_total_bytes: int | None
    safe_bytes_regret: int | None
    stage: str
    kind: str
    layer_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SelectorEvaluationSummary:
    example_count: int
    targetable_count: int
    target_accuracy: float
    safe_prediction_rate: float
    unsafe_prediction_rate: float
    mean_safe_bytes_regret: float | None
    p95_safe_bytes_regret: float | None
    max_safe_bytes_regret: int | None
    mean_predicted_total_bytes: float | None
    predicted_candidate_histogram: dict[str, int]
    oracle_target_histogram: dict[str, int]
    per_stage_accuracy: dict[str, float]
    per_kind_accuracy: dict[str, float]
    predictions: list[SelectorPrediction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_count": self.example_count,
            "targetable_count": self.targetable_count,
            "target_accuracy": self.target_accuracy,
            "safe_prediction_rate": self.safe_prediction_rate,
            "unsafe_prediction_rate": self.unsafe_prediction_rate,
            "mean_safe_bytes_regret": self.mean_safe_bytes_regret,
            "p95_safe_bytes_regret": self.p95_safe_bytes_regret,
            "max_safe_bytes_regret": self.max_safe_bytes_regret,
            "mean_predicted_total_bytes": self.mean_predicted_total_bytes,
            "predicted_candidate_histogram": dict(self.predicted_candidate_histogram),
            "oracle_target_histogram": dict(self.oracle_target_histogram),
            "per_stage_accuracy": dict(self.per_stage_accuracy),
            "per_kind_accuracy": dict(self.per_kind_accuracy),
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }


@dataclass(slots=True)
class StaticRuleSelectorModel:
    global_candidate: str | None
    key_with_age: dict[tuple[str, str, int, int, bool], str]
    key_without_age: dict[tuple[str, str, int, bool], str]
    key_stage_kind: dict[tuple[str, str, bool], str]

    def predict(self, example: SelectorExample) -> str | None:
        age_bucket = _age_bucket(example.token_age)
        key_with_age = (example.stage, example.kind, example.layer_id, age_bucket, example.query_present)
        if key_with_age in self.key_with_age:
            return self.key_with_age[key_with_age]
        key_without_age = (example.stage, example.kind, example.layer_id, example.query_present)
        if key_without_age in self.key_without_age:
            return self.key_without_age[key_without_age]
        key_stage_kind = (example.stage, example.kind, example.query_present)
        if key_stage_kind in self.key_stage_kind:
            return self.key_stage_kind[key_stage_kind]
        return self.global_candidate


@dataclass(slots=True)
class LinearSelectorModel:
    classes: tuple[str, ...]
    weight: np.ndarray
    bias: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_names: tuple[str, ...]

    def predict(self, example: SelectorExample) -> str | None:
        if not self.classes:
            return None
        logits = self.predict_logits_for_row(example.row)
        return self.classes[int(np.argmax(logits))]

    def predict_logits_for_row(self, row: dict[str, Any]) -> np.ndarray:
        features = selector_feature_vector_from_row(row, feature_names=self.feature_names)
        standardized = (features - self.feature_mean) / self.feature_std
        return standardized @ self.weight + self.bias

    def predict_row(self, row: dict[str, Any]) -> str | None:
        if not self.classes:
            return None
        logits = self.predict_logits_for_row(row)
        return self.classes[int(np.argmax(logits))]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "linear_selector_model",
            "classes": list(self.classes),
            "weight": self.weight.tolist(),
            "bias": self.bias.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LinearSelectorModel":
        return cls(
            classes=tuple(str(value) for value in payload.get("classes", [])),
            weight=np.asarray(payload.get("weight", []), dtype=np.float32),
            bias=np.asarray(payload.get("bias", []), dtype=np.float32),
            feature_mean=np.asarray(payload.get("feature_mean", []), dtype=np.float32),
            feature_std=np.asarray(payload.get("feature_std", []), dtype=np.float32),
            feature_names=tuple(str(value) for value in payload.get("feature_names", [])),
        )


@dataclass(slots=True)
class CandidateSafeLinearSelectorModel:
    weight: np.ndarray
    bias: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_names: tuple[str, ...]
    decision_threshold: float = 0.5

    def predict_probability(self, example: SelectorCandidateExample) -> float:
        features = _candidate_feature_vector(example, feature_names=self.feature_names)
        standardized = (features - self.feature_mean) / self.feature_std
        logit = float(standardized @ self.weight + self.bias)
        return float(1.0 / (1.0 + np.exp(-logit)))

    def predict_probability_for_row(self, row: dict[str, Any]) -> float:
        features = selector_candidate_feature_vector_from_row(row, feature_names=self.feature_names)
        standardized = (features - self.feature_mean) / self.feature_std
        logit = float(standardized @ self.weight + self.bias)
        return float(1.0 / (1.0 + np.exp(-logit)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "candidate_safe_linear_selector_model",
            "weight": self.weight.tolist(),
            "bias": float(self.bias),
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "feature_names": list(self.feature_names),
            "decision_threshold": float(self.decision_threshold),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateSafeLinearSelectorModel":
        return cls(
            weight=np.asarray(payload.get("weight", []), dtype=np.float32),
            bias=float(payload.get("bias", 0.0)),
            feature_mean=np.asarray(payload.get("feature_mean", []), dtype=np.float32),
            feature_std=np.asarray(payload.get("feature_std", []), dtype=np.float32),
            feature_names=tuple(str(value) for value in payload.get("feature_names", [])),
            decision_threshold=float(payload.get("decision_threshold", 0.5)),
        )


@dataclass(slots=True)
class CandidateTargetLinearSelectorModel:
    weight: np.ndarray
    bias: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_names: tuple[str, ...]
    decision_threshold: float = 0.5

    def predict_probability(self, example: SelectorCandidateExample) -> float:
        features = _candidate_feature_vector(example, feature_names=self.feature_names)
        standardized = (features - self.feature_mean) / self.feature_std
        logit = float(standardized @ self.weight + self.bias)
        return float(1.0 / (1.0 + np.exp(-logit)))

    def predict_probability_for_row(self, row: dict[str, Any]) -> float:
        features = selector_candidate_feature_vector_from_row(row, feature_names=self.feature_names)
        standardized = (features - self.feature_mean) / self.feature_std
        logit = float(standardized @ self.weight + self.bias)
        return float(1.0 / (1.0 + np.exp(-logit)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "candidate_target_linear_selector_model",
            "weight": self.weight.tolist(),
            "bias": float(self.bias),
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "feature_names": list(self.feature_names),
            "decision_threshold": float(self.decision_threshold),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateTargetLinearSelectorModel":
        return cls(
            weight=np.asarray(payload.get("weight", []), dtype=np.float32),
            bias=float(payload.get("bias", 0.0)),
            feature_mean=np.asarray(payload.get("feature_mean", []), dtype=np.float32),
            feature_std=np.asarray(payload.get("feature_std", []), dtype=np.float32),
            feature_names=tuple(str(value) for value in payload.get("feature_names", [])),
            decision_threshold=float(payload.get("decision_threshold", 0.5)),
        )


@dataclass(slots=True)
class CandidateSafeRouterModel:
    safe_model: CandidateSafeLinearSelectorModel
    candidate_tokens: tuple[str, ...]
    fallback_candidate: str | None
    group_size: int = 32
    payload_layout_k: str = "group_major"
    payload_layout_v: str = "group_major"
    escape_dtype: str = "float16"
    prompt_family_thresholds: dict[str, float] = field(default_factory=dict)

    def predict_row(self, row: dict[str, Any]) -> str | None:
        supported: list[tuple[dict[str, Any], float]] = []
        for candidate_token in self.candidate_tokens:
            candidate_row = build_runtime_selector_candidate_row(
                row,
                candidate_token=candidate_token,
                group_size=self.group_size,
                payload_layout_k=self.payload_layout_k,
                payload_layout_v=self.payload_layout_v,
                escape_dtype=self.escape_dtype,
            )
            if candidate_row is None:
                continue
            probability = self.safe_model.predict_probability_for_row(candidate_row)
            supported.append((candidate_row, probability))
        if not supported:
            return self.fallback_candidate

        normalized_family = _normalize_categorical_token(row.get("prompt_family"))
        threshold = float(self.prompt_family_thresholds.get(normalized_family or "", self.safe_model.decision_threshold))
        predicted_safe = [item for item in supported if item[1] >= threshold]
        if predicted_safe:
            predicted_safe.sort(
                key=lambda item: (
                    int(item[0]["candidate_total_bytes"]),
                    -float(item[1]),
                    str(item[0]["candidate"]),
                )
            )
            return str(predicted_safe[0][0]["candidate"])

        if self.fallback_candidate is not None:
            return str(self.fallback_candidate)
        supported.sort(
            key=lambda item: (
                -float(item[1]),
                int(item[0]["candidate_total_bytes"]),
                str(item[0]["candidate"]),
            )
        )
        return str(supported[0][0]["candidate"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "candidate_safe_router_model",
            "safe_model": self.safe_model.to_dict(),
            "candidate_tokens": list(self.candidate_tokens),
            "fallback_candidate": self.fallback_candidate,
            "group_size": int(self.group_size),
            "payload_layout_k": str(self.payload_layout_k),
            "payload_layout_v": str(self.payload_layout_v),
            "escape_dtype": str(self.escape_dtype),
            "prompt_family_thresholds": {str(key): float(value) for key, value in sorted(self.prompt_family_thresholds.items())},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateSafeRouterModel":
        safe_model_payload = dict(payload.get("safe_model", {}))
        return cls(
            safe_model=CandidateSafeLinearSelectorModel.from_dict(safe_model_payload),
            candidate_tokens=tuple(str(value) for value in payload.get("candidate_tokens", [])),
            fallback_candidate=(
                None if payload.get("fallback_candidate") in (None, "") else str(payload.get("fallback_candidate"))
            ),
            group_size=int(payload.get("group_size", 32)),
            payload_layout_k=str(payload.get("payload_layout_k", "group_major")),
            payload_layout_v=str(payload.get("payload_layout_v", "group_major")),
            escape_dtype=str(payload.get("escape_dtype", "float16")),
            prompt_family_thresholds={
                str(key): float(value) for key, value in dict(payload.get("prompt_family_thresholds", {})).items()
            },
        )


@dataclass(slots=True)
class CandidateTargetRouterModel:
    target_model: CandidateTargetLinearSelectorModel
    candidate_tokens: tuple[str, ...]
    fallback_candidate: str | None
    group_size: int = 32
    payload_layout_k: str = "group_major"
    payload_layout_v: str = "group_major"
    escape_dtype: str = "float16"
    prompt_family_thresholds: dict[str, float] = field(default_factory=dict)
    candidate_logit_offsets: dict[str, float] = field(default_factory=dict)

    def predict_row(self, row: dict[str, Any]) -> str | None:
        supported: list[tuple[dict[str, Any], float]] = []
        for candidate_token in self.candidate_tokens:
            candidate_row = build_runtime_selector_candidate_row(
                row,
                candidate_token=candidate_token,
                group_size=self.group_size,
                payload_layout_k=self.payload_layout_k,
                payload_layout_v=self.payload_layout_v,
                escape_dtype=self.escape_dtype,
            )
            if candidate_row is None:
                continue
            probability = self.target_model.predict_probability_for_row(candidate_row)
            probability = _apply_candidate_logit_offset(
                probability,
                self.candidate_logit_offsets.get(str(candidate_row.get("candidate")), 0.0),
            )
            supported.append((candidate_row, probability))
        if not supported:
            return self.fallback_candidate

        normalized_family = _normalize_categorical_token(row.get("prompt_family"))
        threshold = float(self.prompt_family_thresholds.get(normalized_family or "", self.target_model.decision_threshold))
        predicted_target = [item for item in supported if item[1] >= threshold]
        if predicted_target:
            predicted_target.sort(
                key=lambda item: (
                    -float(item[1]),
                    int(item[0]["candidate_total_bytes"]),
                    str(item[0]["candidate"]),
                )
            )
            return str(predicted_target[0][0]["candidate"])

        if self.fallback_candidate is not None:
            return str(self.fallback_candidate)
        supported.sort(
            key=lambda item: (
                -float(item[1]),
                int(item[0]["candidate_total_bytes"]),
                str(item[0]["candidate"]),
            )
        )
        return str(supported[0][0]["candidate"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "candidate_target_router_model",
            "target_model": self.target_model.to_dict(),
            "candidate_tokens": list(self.candidate_tokens),
            "fallback_candidate": self.fallback_candidate,
            "group_size": int(self.group_size),
            "payload_layout_k": str(self.payload_layout_k),
            "payload_layout_v": str(self.payload_layout_v),
            "escape_dtype": str(self.escape_dtype),
            "prompt_family_thresholds": {str(key): float(value) for key, value in sorted(self.prompt_family_thresholds.items())},
            "candidate_logit_offsets": {str(key): float(value) for key, value in sorted(self.candidate_logit_offsets.items())},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateTargetRouterModel":
        target_model_payload = dict(payload.get("target_model", {}))
        return cls(
            target_model=CandidateTargetLinearSelectorModel.from_dict(target_model_payload),
            candidate_tokens=tuple(str(value) for value in payload.get("candidate_tokens", [])),
            fallback_candidate=(
                None if payload.get("fallback_candidate") in (None, "") else str(payload.get("fallback_candidate"))
            ),
            group_size=int(payload.get("group_size", 32)),
            payload_layout_k=str(payload.get("payload_layout_k", "group_major")),
            payload_layout_v=str(payload.get("payload_layout_v", "group_major")),
            escape_dtype=str(payload.get("escape_dtype", "float16")),
            prompt_family_thresholds={
                str(key): float(value) for key, value in dict(payload.get("prompt_family_thresholds", {})).items()
            },
            candidate_logit_offsets={
                str(key): float(value) for key, value in dict(payload.get("candidate_logit_offsets", {})).items()
            },
        )


def load_selector_examples(
    *,
    labels_path: str | Path,
    selector_dataset_path: str | Path,
) -> list[SelectorExample]:
    labels_by_trace: dict[str, dict[str, Any]] = {}
    with Path(labels_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            labels_by_trace[str(payload["trace_path"])] = payload

    examples: list[SelectorExample] = []
    with Path(selector_dataset_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            trace_path = str(row["trace_path"])
            label = labels_by_trace.get(trace_path)
            if label is None:
                raise ValueError(f"selector row is missing matching label: {trace_path}")
            candidate_map = {
                str(candidate["candidate"]): dict(candidate)
                for candidate in label.get("candidate_labels", [])
            }
            examples.append(
                SelectorExample(
                    trace_path=trace_path,
                    row=row,
                    label=label,
                    candidate_map=candidate_map,
                )
            )
    return examples


def save_page_selector_artifact(
    model: LinearSelectorModel | CandidateSafeRouterModel | CandidateTargetRouterModel,
    path: str | Path,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def load_page_selector_artifact(path: str | Path) -> LinearSelectorModel | CandidateSafeRouterModel | CandidateTargetRouterModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    artifact_type = str(payload.get("artifact_type", "linear_selector_model"))
    if artifact_type == "linear_selector_model":
        return LinearSelectorModel.from_dict(payload)
    if artifact_type == "candidate_safe_router_model":
        return CandidateSafeRouterModel.from_dict(payload)
    if artifact_type == "candidate_target_router_model":
        return CandidateTargetRouterModel.from_dict(payload)
    raise ValueError(f"unsupported page selector artifact_type: {artifact_type}")


def save_linear_selector_model(model: LinearSelectorModel, path: str | Path) -> None:
    save_page_selector_artifact(model, path)


def load_linear_selector_model(path: str | Path) -> LinearSelectorModel:
    artifact = load_page_selector_artifact(path)
    if not isinstance(artifact, LinearSelectorModel):
        raise ValueError("page selector artifact is not a linear selector model")
    return artifact


def adjust_linear_selector_model_logits(
    model: LinearSelectorModel,
    *,
    candidate_logit_offsets: dict[str, float],
) -> LinearSelectorModel:
    if not candidate_logit_offsets:
        return LinearSelectorModel(
            classes=tuple(model.classes),
            weight=np.array(model.weight, copy=True),
            bias=np.array(model.bias, copy=True),
            feature_mean=np.array(model.feature_mean, copy=True),
            feature_std=np.array(model.feature_std, copy=True),
            feature_names=tuple(model.feature_names),
        )
    updated_bias = np.array(model.bias, copy=True)
    classes = tuple(model.classes)
    for candidate, offset in candidate_logit_offsets.items():
        try:
            class_index = classes.index(str(candidate))
        except ValueError as exc:
            raise ValueError(f"selector model does not contain candidate: {candidate}") from exc
        updated_bias[class_index] = np.float32(updated_bias[class_index] + float(offset))
    return LinearSelectorModel(
        classes=classes,
        weight=np.array(model.weight, copy=True),
        bias=updated_bias,
        feature_mean=np.array(model.feature_mean, copy=True),
        feature_std=np.array(model.feature_std, copy=True),
        feature_names=tuple(model.feature_names),
    )


def build_selector_example_weights(
    examples: Sequence[SelectorExample],
    *,
    classes: Sequence[str] | None = None,
    class_balance: float = 0.0,
    safe_bytes_weight: float = 0.0,
    reference_candidate: str = "M3/affine/4/float16",
    trace_weight_multipliers: dict[str, float] | None = None,
) -> np.ndarray:
    target_examples = [example for example in examples if example.target_present and example.target_candidate is not None]
    if not target_examples:
        return np.zeros((0,), dtype=np.float32)

    resolved_classes = (
        tuple(str(candidate) for candidate in classes)
        if classes is not None
        else tuple(sorted({str(example.target_candidate) for example in target_examples}))
    )
    class_counts = Counter(str(example.target_candidate) for example in target_examples)
    total_count = max(len(target_examples), 1)
    class_count = max(len(resolved_classes), 1)

    weights: list[float] = []
    for example in target_examples:
        weight = 1.0
        if float(class_balance) > 0.0:
            candidate = str(example.target_candidate)
            balanced = float(total_count) / float(class_count * max(class_counts.get(candidate, 0), 1))
            weight *= balanced ** float(class_balance)
        if float(safe_bytes_weight) > 0.0:
            weight *= 1.0 + float(safe_bytes_weight) * _compression_gain_ratio(
                example,
                reference_candidate=str(reference_candidate),
            )
        if trace_weight_multipliers is not None:
            weight *= float(trace_weight_multipliers.get(example.trace_path, 1.0))
        weights.append(weight)
    return np.asarray(weights, dtype=np.float32)


def selector_feature_vector_from_row(
    row: dict[str, Any],
    *,
    feature_names: Sequence[str],
) -> np.ndarray:
    values = _selector_base_feature_values_from_row(row)
    prompt_family = _normalize_categorical_token(row.get("prompt_family"))
    prompt_variant = _normalize_categorical_token(row.get("prompt_variant"))
    return np.asarray(
        [_resolve_feature_value(values, name, prompt_family=prompt_family, prompt_variant=prompt_variant) for name in feature_names],
        dtype=np.float32,
    )


def selector_candidate_feature_vector_from_row(
    row: dict[str, Any],
    *,
    feature_names: Sequence[str],
) -> np.ndarray:
    stage_decode = 1.0 if str(row.get("stage", "")) == "decode" else 0.0
    candidate_mode = str(row.get("candidate_mode", ""))
    values = {
        **_selector_base_feature_values_from_row(row),
        "candidate_mode_m0": 1.0 if candidate_mode == "M0" else 0.0,
        "candidate_mode_m1": 1.0 if candidate_mode == "M1" else 0.0,
        "candidate_mode_m2": 1.0 if candidate_mode == "M2" else 0.0,
        "candidate_mode_m3": 1.0 if candidate_mode == "M3" else 0.0,
        "candidate_mode_m4": 1.0 if candidate_mode == "M4" else 0.0,
        "candidate_mode_t3": 1.0 if candidate_mode == "T3" else 0.0,
        "decode_candidate_mode_m0": stage_decode * (1.0 if candidate_mode == "M0" else 0.0),
        "decode_candidate_mode_m1": stage_decode * (1.0 if candidate_mode == "M1" else 0.0),
        "decode_candidate_mode_m2": stage_decode * (1.0 if candidate_mode == "M2" else 0.0),
        "decode_candidate_mode_m3": stage_decode * (1.0 if candidate_mode == "M3" else 0.0),
        "decode_candidate_mode_m4": stage_decode * (1.0 if candidate_mode == "M4" else 0.0),
        "decode_candidate_mode_t3": stage_decode * (1.0 if candidate_mode == "T3" else 0.0),
        "candidate_bits": float(row.get("candidate_bits", 0.0)),
        "candidate_scheme_affine": 1.0 if str(row.get("candidate_quant_scheme", "")) == "affine" else 0.0,
        "candidate_scheme_lut": 1.0 if str(row.get("candidate_quant_scheme", "")) == "lut" else 0.0,
        "candidate_scheme_sketch": 1.0 if str(row.get("candidate_quant_scheme", "")) == "sketch" else 0.0,
        "candidate_scheme_project": 1.0 if str(row.get("candidate_quant_scheme", "")) == "project" else 0.0,
        "candidate_scheme_turbo3": 1.0 if str(row.get("candidate_quant_scheme", "")) == "turbo3" else 0.0,
        "log_candidate_total_bytes": float(np.log1p(float(row.get("candidate_total_bytes", 0.0)))),
        "log_candidate_payload_bytes": float(np.log1p(float(row.get("candidate_payload_bytes", 0.0)))),
        "log_candidate_metadata_bytes": float(np.log1p(float(row.get("candidate_metadata_bytes", 0.0)))),
        "candidate_has_escape_dtype": 1.0 if bool(row.get("candidate_has_escape_dtype", False)) else 0.0,
        "candidate_bytes_over_best_safe": float(row.get("candidate_bytes_over_best_safe", 0.0)),
    }
    prompt_family = _normalize_categorical_token(row.get("prompt_family"))
    prompt_variant = _normalize_categorical_token(row.get("prompt_variant"))
    return np.asarray(
        [_resolve_feature_value(values, name, prompt_family=prompt_family, prompt_variant=prompt_variant) for name in feature_names],
        dtype=np.float32,
    )


def load_selector_candidate_examples(
    *,
    selector_candidate_dataset_path: str | Path,
) -> list[SelectorCandidateExample]:
    examples: list[SelectorCandidateExample] = []
    with Path(selector_candidate_dataset_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                SelectorCandidateExample(
                    trace_path=str(row["trace_path"]),
                    row=row,
                )
            )
    return examples


def load_selector_split_examples(
    *,
    split_dir: str | Path,
) -> dict[str, Any]:
    root = Path(split_dir)
    train_dir = root / "train"
    test_dir = root / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise ValueError(f"split_dir must contain train/ and test/ subdirectories: {root}")

    train_examples = load_selector_examples(
        labels_path=train_dir / "labels.jsonl",
        selector_dataset_path=train_dir / "selector_dataset.jsonl",
    )
    test_examples = load_selector_examples(
        labels_path=test_dir / "labels.jsonl",
        selector_dataset_path=test_dir / "selector_dataset.jsonl",
    )

    train_candidate_path = train_dir / "selector_candidate_dataset.jsonl"
    test_candidate_path = test_dir / "selector_candidate_dataset.jsonl"
    train_candidate_examples = (
        []
        if not train_candidate_path.exists()
        else load_selector_candidate_examples(selector_candidate_dataset_path=train_candidate_path)
    )
    test_candidate_examples = (
        []
        if not test_candidate_path.exists()
        else load_selector_candidate_examples(selector_candidate_dataset_path=test_candidate_path)
    )

    summary_path = root / "split_summary.json"
    split_summary = None if not summary_path.exists() else json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "split_dir": str(root),
        "split_summary": split_summary,
        "train_examples": train_examples,
        "test_examples": test_examples,
        "train_candidate_examples": train_candidate_examples,
        "test_candidate_examples": test_candidate_examples,
    }


def discover_selector_split_dirs(split_root: str | Path) -> list[Path]:
    root = Path(split_root)
    if not root.exists():
        raise ValueError(f"split_root does not exist: {root}")
    discovered: list[Path] = []
    if (root / "train").is_dir() and (root / "test").is_dir():
        discovered.append(root)
    for candidate in sorted(path for path in root.iterdir() if path.is_dir()):
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            discovered.append(candidate)
    return discovered


def split_selector_examples(
    examples: Sequence[SelectorExample],
    *,
    test_fraction: float = 0.25,
    seed: int = 0,
) -> SelectorSplit:
    if not 0.0 < float(test_fraction) < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    key_specs = (
        lambda example: (example.stage, example.kind, example.target_candidate),
        lambda example: (example.stage, example.target_candidate),
        lambda example: (example.target_candidate,),
    )
    for key_fn in key_specs:
        split = _stratified_split_with_key(examples, test_fraction=test_fraction, seed=seed, key_fn=key_fn)
        if split.train_indices and split.test_indices:
            return split
    return _random_split(examples, test_fraction=test_fraction, seed=seed)


def train_static_rule_selector(examples: Sequence[SelectorExample]) -> StaticRuleSelectorModel:
    target_examples = [example for example in examples if example.target_present and example.target_candidate is not None]
    global_candidate = _majority_target(target_examples)

    key_with_age: dict[tuple[str, str, int, int, bool], str] = {}
    key_without_age: dict[tuple[str, str, int, bool], str] = {}
    key_stage_kind: dict[tuple[str, str, bool], str] = {}

    grouped_with_age: dict[tuple[str, str, int, int, bool], list[SelectorExample]] = defaultdict(list)
    grouped_without_age: dict[tuple[str, str, int, bool], list[SelectorExample]] = defaultdict(list)
    grouped_stage_kind: dict[tuple[str, str, bool], list[SelectorExample]] = defaultdict(list)

    for example in target_examples:
        grouped_with_age[(example.stage, example.kind, example.layer_id, _age_bucket(example.token_age), example.query_present)].append(example)
        grouped_without_age[(example.stage, example.kind, example.layer_id, example.query_present)].append(example)
        grouped_stage_kind[(example.stage, example.kind, example.query_present)].append(example)

    for key, values in grouped_with_age.items():
        key_with_age[key] = _majority_target(values)
    for key, values in grouped_without_age.items():
        key_without_age[key] = _majority_target(values)
    for key, values in grouped_stage_kind.items():
        key_stage_kind[key] = _majority_target(values)

    return StaticRuleSelectorModel(
        global_candidate=global_candidate,
        key_with_age=key_with_age,
        key_without_age=key_without_age,
        key_stage_kind=key_stage_kind,
    )


def train_linear_selector(
    examples: Sequence[SelectorExample],
    *,
    steps: int = 400,
    learning_rate: float = 0.2,
    l2: float = 1e-3,
    feature_names: Sequence[str] | None = None,
    class_balance: float = 0.0,
    safe_bytes_weight: float = 0.0,
    unsafe_error_weight: float = 0.0,
    reference_candidate: str = "M3/affine/4/float16",
    trace_weight_multipliers: dict[str, float] | None = None,
) -> LinearSelectorModel:
    target_examples = [example for example in examples if example.target_present and example.target_candidate is not None]
    classes = tuple(sorted({str(example.target_candidate) for example in target_examples}))
    resolved_feature_names = tuple(feature_names) if feature_names is not None else _selector_feature_names_from_examples(target_examples)
    if not target_examples or not classes:
        feature_dim = len(resolved_feature_names)
        return LinearSelectorModel(
            classes=(),
            weight=np.zeros((feature_dim, 0), dtype=np.float32),
            bias=np.zeros((0,), dtype=np.float32),
            feature_mean=np.zeros((feature_dim,), dtype=np.float32),
            feature_std=np.ones((feature_dim,), dtype=np.float32),
            feature_names=resolved_feature_names,
        )

    class_to_index = {candidate: index for index, candidate in enumerate(classes)}
    x = np.stack([_feature_vector(example, feature_names=resolved_feature_names) for example in target_examples], axis=0).astype(np.float32)
    y = np.array([class_to_index[str(example.target_candidate)] for example in target_examples], dtype=np.int32)
    example_weights = build_selector_example_weights(
        target_examples,
        classes=classes,
        class_balance=class_balance,
        safe_bytes_weight=safe_bytes_weight,
        reference_candidate=reference_candidate,
        trace_weight_multipliers=trace_weight_multipliers,
    )
    class_error_weights = build_selector_class_error_weights(
        target_examples,
        classes=classes,
        unsafe_error_weight=unsafe_error_weight,
    )
    example_weight_sum = float(np.sum(example_weights, dtype=np.float32))
    if example_weight_sum <= 0.0:
        example_weights = np.ones((len(target_examples),), dtype=np.float32)
        example_weight_sum = float(len(target_examples))
    feature_mean = np.mean(x, axis=0, dtype=np.float32)
    feature_std = np.std(x, axis=0, dtype=np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)
    x_std = (x - feature_mean) / feature_std

    weight = np.zeros((x_std.shape[1], len(classes)), dtype=np.float32)
    bias = np.zeros((len(classes),), dtype=np.float32)
    targets = np.eye(len(classes), dtype=np.float32)[y]

    for _ in range(int(steps)):
        logits = x_std @ weight + bias
        probs = _softmax(logits)
        error = probs - targets
        weighted_error = error * example_weights[:, None] * class_error_weights
        grad_weight = (x_std.T @ weighted_error) / max(example_weight_sum, 1.0) + float(l2) * weight
        grad_bias = np.sum(weighted_error, axis=0, dtype=np.float32) / max(example_weight_sum, 1.0)
        weight -= float(learning_rate) * grad_weight.astype(np.float32)
        bias -= float(learning_rate) * grad_bias.astype(np.float32)

    return LinearSelectorModel(
        classes=classes,
        weight=weight,
        bias=bias,
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_names=resolved_feature_names,
    )


def train_runtime_linear_selector(
    examples: Sequence[SelectorExample],
    *,
    steps: int = 400,
    learning_rate: float = 0.2,
    l2: float = 1e-3,
    class_balance: float = 0.0,
    safe_bytes_weight: float = 0.0,
    unsafe_error_weight: float = 0.0,
    reference_candidate: str = "M3/affine/4/float16",
    trace_weight_multipliers: dict[str, float] | None = None,
) -> LinearSelectorModel:
    return train_linear_selector(
        examples,
        steps=steps,
        learning_rate=learning_rate,
        l2=l2,
        feature_names=_runtime_selector_feature_names_from_examples(examples),
        class_balance=class_balance,
        safe_bytes_weight=safe_bytes_weight,
        unsafe_error_weight=unsafe_error_weight,
        reference_candidate=reference_candidate,
        trace_weight_multipliers=trace_weight_multipliers,
    )


def selector_prompt_length_from_row(
    row: dict[str, Any],
    *,
    trace_path: str | None = None,
) -> int | None:
    value = row.get("prompt_length")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    for candidate in (
        trace_path,
        row.get("trace_path"),
        row.get("source"),
    ):
        if candidate in (None, ""):
            continue
        match = _PROMPT_LENGTH_RE.search(str(candidate))
        if match is None:
            continue
        try:
            return int(match.group("prompt_length"))
        except (TypeError, ValueError):
            continue
    return None


def train_candidate_safe_linear_selector(
    examples: Sequence[SelectorCandidateExample],
    *,
    steps: int = 400,
    learning_rate: float = 0.2,
    l2: float = 1e-3,
    feature_names: Sequence[str] | None = None,
) -> CandidateSafeLinearSelectorModel:
    resolved_feature_names = (
        tuple(str(value) for value in feature_names)
        if feature_names is not None
        else _candidate_feature_names_from_examples(examples)
    )
    feature_dim = len(resolved_feature_names)
    if not examples:
        return CandidateSafeLinearSelectorModel(
            weight=np.zeros((feature_dim,), dtype=np.float32),
            bias=0.0,
            feature_mean=np.zeros((feature_dim,), dtype=np.float32),
            feature_std=np.ones((feature_dim,), dtype=np.float32),
            feature_names=resolved_feature_names,
        )

    x = np.stack([_candidate_feature_vector(example, feature_names=resolved_feature_names) for example in examples], axis=0).astype(np.float32)
    y = np.asarray([1.0 if example.candidate_safe else 0.0 for example in examples], dtype=np.float32)
    feature_mean = np.mean(x, axis=0, dtype=np.float32)
    feature_std = np.std(x, axis=0, dtype=np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)
    x_std = (x - feature_mean) / feature_std

    weight = np.zeros((x_std.shape[1],), dtype=np.float32)
    bias = 0.0
    for _ in range(int(steps)):
        logits = x_std @ weight + bias
        probs = 1.0 / (1.0 + np.exp(-logits))
        error = probs - y
        grad_weight = (x_std.T @ error) / max(x_std.shape[0], 1) + float(l2) * weight
        grad_bias = float(np.mean(error, dtype=np.float32))
        weight -= float(learning_rate) * grad_weight.astype(np.float32)
        bias -= float(learning_rate) * grad_bias

    return CandidateSafeLinearSelectorModel(
        weight=weight,
        bias=float(bias),
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_names=resolved_feature_names,
    )


def train_candidate_safe_router(
    examples: Sequence[SelectorCandidateExample],
    *,
    steps: int = 400,
    learning_rate: float = 0.2,
    l2: float = 1e-3,
    group_size: int = 32,
    payload_layout_k: str = "group_major",
    payload_layout_v: str = "group_major",
    escape_dtype: str = "float16",
    candidate_tokens: Sequence[str] | None = None,
    fallback_candidate: str | None = None,
    feature_names: Sequence[str] | None = None,
    prompt_family_thresholds: dict[str, float] | None = None,
    decision_threshold: float = 0.5,
) -> CandidateSafeRouterModel:
    safe_model = train_candidate_safe_linear_selector(
        examples,
        steps=steps,
        learning_rate=learning_rate,
        l2=l2,
        feature_names=feature_names,
    )
    safe_model.decision_threshold = float(decision_threshold)
    resolved_candidate_tokens = (
        tuple(str(token) for token in candidate_tokens)
        if candidate_tokens is not None
        else tuple(
            sorted(
                {
                    str(example.oracle_target_candidate)
                    for example in examples
                    if example.oracle_target_candidate is not None
                }
            )
        )
    )
    resolved_fallback_candidate = fallback_candidate
    if resolved_fallback_candidate is None:
        resolved_fallback_candidate = (
            "M3/affine/4/float16"
            if "M3/affine/4/float16" in resolved_candidate_tokens
            else (resolved_candidate_tokens[-1] if resolved_candidate_tokens else None)
        )
    return CandidateSafeRouterModel(
        safe_model=safe_model,
        candidate_tokens=resolved_candidate_tokens,
        fallback_candidate=resolved_fallback_candidate,
        group_size=group_size,
        payload_layout_k=payload_layout_k,
        payload_layout_v=payload_layout_v,
        escape_dtype=escape_dtype,
        prompt_family_thresholds=(
            {}
            if prompt_family_thresholds is None
            else {str(key): float(value) for key, value in prompt_family_thresholds.items()}
        ),
    )


def train_candidate_target_linear_selector(
    examples: Sequence[SelectorCandidateExample],
    *,
    steps: int = 400,
    learning_rate: float = 0.2,
    l2: float = 1e-3,
    feature_names: Sequence[str] | None = None,
    loss_kind: str = "binary",
    class_balance: float = 0.0,
    reference_candidate: str = "M3/affine/4/float16",
    non_reference_target_weight: float = 0.0,
    compression_target_weight: float = 0.0,
    reference_false_positive_weight: float = 0.0,
) -> CandidateTargetLinearSelectorModel:
    resolved_feature_names = (
        tuple(str(value) for value in feature_names)
        if feature_names is not None
        else _candidate_feature_names_from_examples(examples)
    )
    feature_dim = len(resolved_feature_names)
    if not examples:
        return CandidateTargetLinearSelectorModel(
            weight=np.zeros((feature_dim,), dtype=np.float32),
            bias=0.0,
            feature_mean=np.zeros((feature_dim,), dtype=np.float32),
            feature_std=np.ones((feature_dim,), dtype=np.float32),
            feature_names=resolved_feature_names,
        )

    x = np.stack([_candidate_feature_vector(example, feature_names=resolved_feature_names) for example in examples], axis=0).astype(np.float32)
    y = np.asarray([1.0 if bool(example.row.get("candidate_is_target", False)) else 0.0 for example in examples], dtype=np.float32)
    feature_mean = np.mean(x, axis=0, dtype=np.float32)
    feature_std = np.std(x, axis=0, dtype=np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)
    x_std = (x - feature_mean) / feature_std

    resolved_loss_kind = str(loss_kind).strip().lower()
    weight = np.zeros((x_std.shape[1],), dtype=np.float32)
    bias = 0.0
    if resolved_loss_kind == "binary":
        for _ in range(int(steps)):
            logits = x_std @ weight + bias
            probs = 1.0 / (1.0 + np.exp(-logits))
            error = probs - y
            grad_weight = (x_std.T @ error) / max(x_std.shape[0], 1) + float(l2) * weight
            grad_bias = float(np.mean(error, dtype=np.float32))
            weight -= float(learning_rate) * grad_weight.astype(np.float32)
            bias -= float(learning_rate) * grad_bias
    elif resolved_loss_kind == "trace_softmax":
        grouped_indices: dict[str, list[int]] = defaultdict(list)
        for index, example in enumerate(examples):
            grouped_indices[example.trace_path].append(index)

        target_candidate_by_trace: dict[str, str] = {}
        grouped_training_rows: list[tuple[np.ndarray, int, str, float]] = []
        for trace_path, trace_indices in grouped_indices.items():
            target_positions = [position for position, index in enumerate(trace_indices) if bool(y[index] >= 0.5)]
            if len(target_positions) != 1:
                continue
            target_position = int(target_positions[0])
            target_index = int(trace_indices[target_position])
            target_example = examples[target_index]
            target_candidate = str(target_example.candidate)
            target_candidate_by_trace[trace_path] = target_candidate
            grouped_training_rows.append(
                (
                    np.asarray(trace_indices, dtype=np.int64),
                    target_position,
                    target_candidate,
                    _candidate_target_trace_weight(
                        trace_indices=trace_indices,
                        target_example=target_example,
                        examples=examples,
                        class_counts=None,
                        total_group_count=0,
                        class_count=0,
                        class_balance=float(class_balance),
                        reference_candidate=str(reference_candidate),
                        non_reference_target_weight=float(non_reference_target_weight),
                        compression_target_weight=float(compression_target_weight),
                    ),
                )
            )

        class_counts = Counter(target_candidate_by_trace.values())
        total_group_count = max(len(grouped_training_rows), 1)
        class_count = max(len(class_counts), 1)
        grouped_training_rows = [
            (
                trace_indices,
                target_position,
                target_candidate,
                _candidate_target_trace_weight(
                    trace_indices=trace_indices.tolist(),
                    target_example=examples[int(trace_indices[target_position])],
                    examples=examples,
                    class_counts=class_counts,
                    total_group_count=total_group_count,
                    class_count=class_count,
                    class_balance=float(class_balance),
                    reference_candidate=str(reference_candidate),
                    non_reference_target_weight=float(non_reference_target_weight),
                    compression_target_weight=float(compression_target_weight),
                ),
            )
            for trace_indices, target_position, target_candidate, _weight in grouped_training_rows
        ]
        group_weight_sum = max(sum(weight for _, _, _, weight in grouped_training_rows), 1.0)

        for _ in range(int(steps)):
            grad_weight = np.zeros_like(weight)
            grad_bias = 0.0
            for trace_indices, target_position, target_candidate, trace_weight in grouped_training_rows:
                group_x = x_std[trace_indices]
                logits = group_x @ weight + bias
                probs = _softmax_rows(logits.reshape(1, -1)).reshape(-1)
                error = np.array(probs, copy=True)
                error[target_position] -= 1.0
                if (
                    float(reference_false_positive_weight) > 0.0
                    and target_candidate != str(reference_candidate)
                ):
                    reference_position = next(
                        (
                            position
                            for position, index in enumerate(trace_indices.tolist())
                            if str(examples[int(index)].candidate) == str(reference_candidate)
                        ),
                        None,
                    )
                    if reference_position is not None:
                        reference_probability = float(probs[reference_position])
                        penalty_gradient = (-reference_probability * probs).astype(np.float32, copy=False)
                        penalty_gradient[reference_position] += reference_probability
                        error += float(reference_false_positive_weight) * penalty_gradient
                grad_weight += float(trace_weight) * (group_x.T @ error).astype(np.float32, copy=False)
                grad_bias += float(trace_weight) * float(np.sum(error, dtype=np.float32))
            grad_weight = grad_weight / group_weight_sum + float(l2) * weight
            grad_bias = grad_bias / group_weight_sum
            weight -= float(learning_rate) * grad_weight.astype(np.float32)
            bias -= float(learning_rate) * float(grad_bias)
    else:
        raise ValueError(f"unsupported candidate target loss_kind: {loss_kind}")

    return CandidateTargetLinearSelectorModel(
        weight=weight,
        bias=float(bias),
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_names=resolved_feature_names,
    )


def train_candidate_target_router(
    examples: Sequence[SelectorCandidateExample],
    *,
    steps: int = 400,
    learning_rate: float = 0.2,
    l2: float = 1e-3,
    group_size: int = 32,
    payload_layout_k: str = "group_major",
    payload_layout_v: str = "group_major",
    escape_dtype: str = "float16",
    candidate_tokens: Sequence[str] | None = None,
    fallback_candidate: str | None = None,
    feature_names: Sequence[str] | None = None,
    prompt_family_thresholds: dict[str, float] | None = None,
    decision_threshold: float = 0.5,
    loss_kind: str = "binary",
    class_balance: float = 0.0,
    reference_candidate: str = "M3/affine/4/float16",
    non_reference_target_weight: float = 0.0,
    compression_target_weight: float = 0.0,
    reference_false_positive_weight: float = 0.0,
    candidate_logit_offsets: dict[str, float] | None = None,
) -> CandidateTargetRouterModel:
    target_model = train_candidate_target_linear_selector(
        examples,
        steps=steps,
        learning_rate=learning_rate,
        l2=l2,
        feature_names=feature_names,
        loss_kind=loss_kind,
        class_balance=class_balance,
        reference_candidate=reference_candidate,
        non_reference_target_weight=non_reference_target_weight,
        compression_target_weight=compression_target_weight,
        reference_false_positive_weight=reference_false_positive_weight,
    )
    target_model.decision_threshold = float(decision_threshold)
    resolved_candidate_tokens = (
        tuple(str(token) for token in candidate_tokens)
        if candidate_tokens is not None
        else tuple(
            sorted(
                {
                    str(example.oracle_target_candidate)
                    for example in examples
                    if example.oracle_target_candidate is not None
                }
            )
        )
    )
    resolved_fallback_candidate = fallback_candidate
    if resolved_fallback_candidate is None:
        resolved_fallback_candidate = (
            "M3/affine/4/float16"
            if "M3/affine/4/float16" in resolved_candidate_tokens
            else (resolved_candidate_tokens[-1] if resolved_candidate_tokens else None)
        )
    return CandidateTargetRouterModel(
        target_model=target_model,
        candidate_tokens=resolved_candidate_tokens,
        fallback_candidate=resolved_fallback_candidate,
        group_size=group_size,
        payload_layout_k=payload_layout_k,
        payload_layout_v=payload_layout_v,
        escape_dtype=escape_dtype,
        prompt_family_thresholds=(
            {}
            if prompt_family_thresholds is None
            else {str(key): float(value) for key, value in prompt_family_thresholds.items()}
        ),
        candidate_logit_offsets=(
            {}
            if candidate_logit_offsets is None
            else {str(key): float(value) for key, value in candidate_logit_offsets.items()}
        ),
    )


def evaluate_selector_model(
    model: StaticRuleSelectorModel | LinearSelectorModel,
    examples: Sequence[SelectorExample],
) -> SelectorEvaluationSummary:
    predictions: list[SelectorPrediction] = []
    correct_target_count = 0
    targetable_count = 0
    safe_prediction_count = 0
    predicted_histogram: Counter[str] = Counter()
    oracle_histogram: Counter[str] = Counter()
    safe_regrets: list[int] = []
    predicted_total_bytes_values: list[int] = []
    stage_counts: Counter[str] = Counter()
    stage_correct: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    kind_correct: Counter[str] = Counter()

    for example in examples:
        predicted_candidate = model.predict(example)
        if predicted_candidate is not None:
            predicted_histogram[predicted_candidate] += 1
        if example.target_candidate is not None:
            oracle_histogram[str(example.target_candidate)] += 1
            targetable_count += 1
        candidate_payload = None if predicted_candidate is None else example.candidate_map.get(predicted_candidate)
        predicted_safe = bool(candidate_payload is not None and candidate_payload.get("safe", False))
        predicted_total_bytes = None if candidate_payload is None else int(candidate_payload["total_bytes"])
        if predicted_total_bytes is not None:
            predicted_total_bytes_values.append(predicted_total_bytes)
        safe_bytes_regret = None
        if predicted_safe and predicted_total_bytes is not None and example.best_safe_total_bytes is not None:
            safe_bytes_regret = int(predicted_total_bytes - example.best_safe_total_bytes)
            safe_regrets.append(safe_bytes_regret)
            safe_prediction_count += 1
        correct_target = bool(predicted_candidate is not None and predicted_candidate == example.target_candidate)
        if correct_target:
            correct_target_count += 1
            stage_correct[example.stage] += 1
            kind_correct[example.kind] += 1
        stage_counts[example.stage] += 1
        kind_counts[example.kind] += 1
        predictions.append(
            SelectorPrediction(
                trace_path=example.trace_path,
                predicted_candidate=predicted_candidate,
                oracle_target_candidate=example.target_candidate,
                correct_target=correct_target,
                predicted_safe=predicted_safe,
                predicted_total_bytes=predicted_total_bytes,
                best_safe_total_bytes=example.best_safe_total_bytes,
                safe_bytes_regret=safe_bytes_regret,
                stage=example.stage,
                kind=example.kind,
                layer_id=example.layer_id,
            )
        )

    mean_safe_bytes_regret = None
    p95_safe_bytes_regret = None
    max_safe_bytes_regret = None
    if safe_regrets:
        mean_safe_bytes_regret = float(np.mean(np.asarray(safe_regrets, dtype=np.float32)))
        p95_safe_bytes_regret = float(np.percentile(np.asarray(safe_regrets, dtype=np.float32), 95))
        max_safe_bytes_regret = int(max(safe_regrets))
    mean_predicted_total_bytes = None
    if predicted_total_bytes_values:
        mean_predicted_total_bytes = float(np.mean(np.asarray(predicted_total_bytes_values, dtype=np.float32)))

    return SelectorEvaluationSummary(
        example_count=len(examples),
        targetable_count=targetable_count,
        target_accuracy=float(correct_target_count / max(targetable_count, 1)),
        safe_prediction_rate=float(safe_prediction_count / max(len(examples), 1)),
        unsafe_prediction_rate=float(1.0 - (safe_prediction_count / max(len(examples), 1))),
        mean_safe_bytes_regret=mean_safe_bytes_regret,
        p95_safe_bytes_regret=p95_safe_bytes_regret,
        max_safe_bytes_regret=max_safe_bytes_regret,
        mean_predicted_total_bytes=mean_predicted_total_bytes,
        predicted_candidate_histogram=dict(sorted(predicted_histogram.items())),
        oracle_target_histogram=dict(sorted(oracle_histogram.items())),
        per_stage_accuracy={
            stage: float(stage_correct.get(stage, 0) / max(count, 1))
            for stage, count in sorted(stage_counts.items())
        },
        per_kind_accuracy={
            kind: float(kind_correct.get(kind, 0) / max(count, 1))
            for kind, count in sorted(kind_counts.items())
        },
        predictions=predictions,
    )


def evaluate_candidate_selector_model(
    model: CandidateSafeLinearSelectorModel,
    examples: Sequence[SelectorCandidateExample],
) -> SelectorEvaluationSummary:
    grouped_examples: dict[str, list[SelectorCandidateExample]] = defaultdict(list)
    for example in examples:
        grouped_examples[example.trace_path].append(example)

    collapsed_examples: list[SelectorExample] = []
    predicted_by_trace: dict[str, str | None] = {}
    for trace_path, trace_examples in grouped_examples.items():
        ordered = sorted(trace_examples, key=lambda item: (item.candidate_total_bytes, item.candidate))
        scored = []
        for example in ordered:
            probability = model.predict_probability(example)
            scored.append((example, probability))
        predicted_safe = [item for item in scored if item[1] >= model.decision_threshold]
        if predicted_safe:
            predicted_safe.sort(key=lambda item: (item[0].candidate_total_bytes, -item[1], item[0].candidate))
            chosen = predicted_safe[0][0]
        else:
            scored.sort(key=lambda item: (-item[1], item[0].candidate_total_bytes, item[0].candidate))
            chosen = scored[0][0] if scored else None
        predicted_by_trace[trace_path] = None if chosen is None else chosen.candidate
        first = ordered[0]
        candidate_map = {
            str(candidate_example.candidate): {
                "candidate": candidate_example.candidate,
                "safe": candidate_example.candidate_safe,
                "total_bytes": candidate_example.candidate_total_bytes,
            }
            for candidate_example in ordered
        }
        label = {
            "safe_candidates": [candidate_example.candidate for candidate_example in ordered if candidate_example.candidate_safe],
        }
        row = {
            key: value
            for key, value in first.row.items()
            if not key.startswith("candidate_")
        }
        collapsed_examples.append(
            SelectorExample(
                trace_path=trace_path,
                row=row,
                label=label,
                candidate_map=candidate_map,
            )
        )

    return evaluate_selector_model(_PredictedSelectorModel(predicted_by_trace), collapsed_examples)


def evaluate_candidate_safe_router_model(
    model: CandidateSafeRouterModel,
    examples: Sequence[SelectorCandidateExample],
) -> SelectorEvaluationSummary:
    grouped_examples: dict[str, list[SelectorCandidateExample]] = defaultdict(list)
    for example in examples:
        grouped_examples[example.trace_path].append(example)

    collapsed_examples: list[SelectorExample] = []
    predicted_by_trace: dict[str, str | None] = {}
    for trace_path, trace_examples in grouped_examples.items():
        ordered = sorted(trace_examples, key=lambda item: (item.candidate_total_bytes, item.candidate))
        first = ordered[0]
        predicted_by_trace[trace_path] = model.predict_row(
            {
                key: value
                for key, value in first.row.items()
                if not key.startswith("candidate_")
            }
        )
        candidate_map = {
            str(candidate_example.candidate): {
                "candidate": candidate_example.candidate,
                "safe": candidate_example.candidate_safe,
                "total_bytes": candidate_example.candidate_total_bytes,
            }
            for candidate_example in ordered
        }
        label = {
            "safe_candidates": [candidate_example.candidate for candidate_example in ordered if candidate_example.candidate_safe],
        }
        row = {
            key: value
            for key, value in first.row.items()
            if not key.startswith("candidate_")
        }
        collapsed_examples.append(
            SelectorExample(
                trace_path=trace_path,
                row=row,
                label=label,
                candidate_map=candidate_map,
            )
        )

    return evaluate_selector_model(_PredictedSelectorModel(predicted_by_trace), collapsed_examples)


def evaluate_candidate_target_router_model(
    model: CandidateTargetRouterModel,
    examples: Sequence[SelectorCandidateExample],
) -> SelectorEvaluationSummary:
    grouped_examples: dict[str, list[SelectorCandidateExample]] = defaultdict(list)
    for example in examples:
        grouped_examples[example.trace_path].append(example)

    collapsed_examples: list[SelectorExample] = []
    predicted_by_trace: dict[str, str | None] = {}
    for trace_path, trace_examples in grouped_examples.items():
        ordered = sorted(trace_examples, key=lambda item: (item.candidate_total_bytes, item.candidate))
        first = ordered[0]
        predicted_by_trace[trace_path] = model.predict_row(
            {
                key: value
                for key, value in first.row.items()
                if not key.startswith("candidate_")
            }
        )
        candidate_map = {
            str(candidate_example.candidate): {
                "candidate": candidate_example.candidate,
                "safe": candidate_example.candidate_safe,
                "total_bytes": candidate_example.candidate_total_bytes,
            }
            for candidate_example in ordered
        }
        label = {
            "safe_candidates": [candidate_example.candidate for candidate_example in ordered if candidate_example.candidate_safe],
        }
        row = {
            key: value
            for key, value in first.row.items()
            if not key.startswith("candidate_")
        }
        collapsed_examples.append(
            SelectorExample(
                trace_path=trace_path,
                row=row,
                label=label,
                candidate_map=candidate_map,
            )
        )

    return evaluate_selector_model(_PredictedSelectorModel(predicted_by_trace), collapsed_examples)


def calibrate_selector_logit_offset(
    model: LinearSelectorModel,
    examples: Sequence[SelectorExample],
    *,
    target_candidate: str,
    offsets: Sequence[float],
    min_target_accuracy: float | None = None,
    min_safe_prediction_rate: float = 1.0,
) -> dict[str, Any]:
    if not offsets:
        raise ValueError("offsets must be non-empty")
    evaluations: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    for offset in (float(value) for value in offsets):
        adjusted_model = adjust_linear_selector_model_logits(
            model,
            candidate_logit_offsets={str(target_candidate): offset},
        )
        summary = evaluate_selector_model(adjusted_model, examples)
        evaluation = {
            "target_candidate": str(target_candidate),
            "logit_offset": float(offset),
            "target_accuracy": float(summary.target_accuracy),
            "safe_prediction_rate": float(summary.safe_prediction_rate),
            "mean_safe_bytes_regret": summary.mean_safe_bytes_regret,
            "mean_predicted_total_bytes": summary.mean_predicted_total_bytes,
            "predicted_candidate_histogram": dict(summary.predicted_candidate_histogram),
        }
        evaluations.append(evaluation)
        meets_accuracy = min_target_accuracy is None or float(summary.target_accuracy) >= float(min_target_accuracy)
        meets_safety = float(summary.safe_prediction_rate) >= float(min_safe_prediction_rate)
        if meets_accuracy and meets_safety:
            feasible.append(evaluation)

    candidate_rows = feasible if feasible else evaluations
    best = min(
        candidate_rows,
        key=lambda row: (
            float("inf") if row["mean_predicted_total_bytes"] is None else float(row["mean_predicted_total_bytes"]),
            -float(row["target_accuracy"]),
            -float(row["safe_prediction_rate"]),
            float(row["logit_offset"]),
        ),
    )
    return {
        "target_candidate": str(target_candidate),
        "min_target_accuracy": None if min_target_accuracy is None else float(min_target_accuracy),
        "min_safe_prediction_rate": float(min_safe_prediction_rate),
        "calibration_objective": "constraint",
        "used_feasible_subset": bool(feasible),
        "best": dict(best),
        "evaluations": evaluations,
    }


def calibrate_selector_logit_offset_tradeoff(
    model: LinearSelectorModel,
    examples: Sequence[SelectorExample],
    *,
    target_candidate: str,
    offsets: Sequence[float],
    correctness_weight: float = 1.0,
    bytes_weight: float = 1.0,
) -> dict[str, Any]:
    if not offsets:
        raise ValueError("offsets must be non-empty")
    normalized_correctness_weight = max(float(correctness_weight), 0.0)
    normalized_bytes_weight = max(float(bytes_weight), 0.0)
    if normalized_correctness_weight <= 0.0 and normalized_bytes_weight <= 0.0:
        raise ValueError("at least one tradeoff weight must be positive")
    weight_sum = normalized_correctness_weight + normalized_bytes_weight
    normalized_correctness_weight /= weight_sum
    normalized_bytes_weight /= weight_sum

    evaluations: list[dict[str, Any]] = []
    for offset in (float(value) for value in offsets):
        adjusted_model = adjust_linear_selector_model_logits(
            model,
            candidate_logit_offsets={str(target_candidate): offset},
        )
        summary = evaluate_selector_model(adjusted_model, examples)
        evaluations.append(
            {
                "target_candidate": str(target_candidate),
                "logit_offset": float(offset),
                "target_accuracy": float(summary.target_accuracy),
                "safe_prediction_rate": float(summary.safe_prediction_rate),
                "correctness_score": float((float(summary.target_accuracy) + float(summary.safe_prediction_rate)) / 2.0),
                "mean_safe_bytes_regret": summary.mean_safe_bytes_regret,
                "mean_predicted_total_bytes": summary.mean_predicted_total_bytes,
                "predicted_candidate_histogram": dict(summary.predicted_candidate_histogram),
            }
        )

    byte_values = [
        float(row["mean_predicted_total_bytes"])
        for row in evaluations
        if row["mean_predicted_total_bytes"] is not None
    ]
    if byte_values:
        min_bytes = min(byte_values)
        max_bytes = max(byte_values)
        byte_span = max(max_bytes - min_bytes, 1e-6)
    else:
        min_bytes = 0.0
        max_bytes = 0.0
        byte_span = 1.0

    for evaluation in evaluations:
        mean_bytes = evaluation["mean_predicted_total_bytes"]
        if mean_bytes is None:
            byte_score = 0.0
        elif max_bytes - min_bytes <= 1e-6:
            byte_score = 1.0
        else:
            byte_score = float((max_bytes - float(mean_bytes)) / byte_span)
        evaluation["byte_score"] = float(byte_score)
        evaluation["tradeoff_score"] = float(
            normalized_correctness_weight * float(evaluation["correctness_score"])
            + normalized_bytes_weight * float(byte_score)
        )

    best = max(
        evaluations,
        key=lambda row: (
            float(row["tradeoff_score"]),
            float(row["correctness_score"]),
            -float("inf") if row["mean_predicted_total_bytes"] is None else -float(row["mean_predicted_total_bytes"]),
            -abs(float(row["logit_offset"])),
        ),
    )
    return {
        "target_candidate": str(target_candidate),
        "calibration_objective": "equal_tradeoff",
        "correctness_weight": float(normalized_correctness_weight),
        "bytes_weight": float(normalized_bytes_weight),
        "best": dict(best),
        "evaluations": evaluations,
    }


def train_calibrated_runtime_linear_selector(
    train_examples: Sequence[SelectorExample],
    *,
    steps: int = 400,
    learning_rate: float = 0.2,
    l2: float = 1e-3,
    class_balance: float = 0.0,
    safe_bytes_weight: float = 0.0,
    unsafe_error_weight: float = 0.0,
    reference_candidate: str = "M3/affine/4/float16",
    calibration_fraction: float = 0.25,
    calibration_seed: int = 0,
    calibration_target_candidate: str | None = None,
    calibration_offsets: Sequence[float] | None = None,
    calibration_min_target_accuracy: float | None = None,
    calibration_min_safe_prediction_rate: float = 1.0,
    calibration_objective: str = "constraint",
    calibration_correctness_weight: float = 1.0,
    calibration_bytes_weight: float = 1.0,
) -> tuple[LinearSelectorModel | None, dict[str, Any] | None]:
    return _train_calibrated_linear_selector(
        train_examples=train_examples,
        steps=steps,
        learning_rate=learning_rate,
        l2=l2,
        class_balance=class_balance,
        safe_bytes_weight=safe_bytes_weight,
        unsafe_error_weight=unsafe_error_weight,
        reference_candidate=reference_candidate,
        calibration_fraction=calibration_fraction,
        calibration_seed=calibration_seed,
        calibration_target_candidate=calibration_target_candidate,
        calibration_offsets=calibration_offsets,
        calibration_min_target_accuracy=calibration_min_target_accuracy,
        calibration_min_safe_prediction_rate=calibration_min_safe_prediction_rate,
        calibration_objective=calibration_objective,
        calibration_correctness_weight=calibration_correctness_weight,
        calibration_bytes_weight=calibration_bytes_weight,
    )


def render_selector_bakeoff_markdown(results: dict[str, SelectorEvaluationSummary]) -> str:
    header = "| baseline | examples | target_accuracy | safe_prediction_rate | mean_safe_bytes_regret | mean_predicted_total_bytes | p95_safe_bytes_regret |"
    separator = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    rows = [header, separator]
    for baseline_name, summary in results.items():
        rows.append(
            "| "
            + " | ".join(
                [
                    baseline_name,
                    str(summary.example_count),
                    f"{summary.target_accuracy:.3f}",
                    f"{summary.safe_prediction_rate:.3f}",
                    "n/a" if summary.mean_safe_bytes_regret is None else f"{summary.mean_safe_bytes_regret:.1f}",
                    "n/a" if summary.mean_predicted_total_bytes is None else f"{summary.mean_predicted_total_bytes:.1f}",
                    "n/a" if summary.p95_safe_bytes_regret is None else f"{summary.p95_safe_bytes_regret:.1f}",
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def render_selector_aggregate_markdown(results: dict[str, dict[str, Any]]) -> str:
    header = "| baseline | folds | mean_target_accuracy | std_target_accuracy | mean_safe_prediction_rate | mean_safe_bytes_regret | mean_predicted_total_bytes |"
    separator = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    rows = [header, separator]
    for baseline_name, summary in results.items():
        rows.append(
            "| "
            + " | ".join(
                [
                    baseline_name,
                    str(int(summary["fold_count"])),
                    f"{float(summary['mean_target_accuracy']):.3f}",
                    f"{float(summary['std_target_accuracy']):.3f}",
                    f"{float(summary['mean_safe_prediction_rate']):.3f}",
                    "n/a" if summary["mean_safe_bytes_regret"] is None else f"{float(summary['mean_safe_bytes_regret']):.1f}",
                    "n/a" if summary["mean_predicted_total_bytes"] is None else f"{float(summary['mean_predicted_total_bytes']):.1f}",
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def render_selector_fixed_split_batch_markdown(split_payloads: Sequence[dict[str, Any]]) -> str:
    header = "| split | baseline | test_examples | target_accuracy | safe_prediction_rate | mean_safe_bytes_regret | mean_predicted_total_bytes |"
    separator = "| --- | --- | ---: | ---: | ---: | ---: | ---: |"
    rows = [header, separator]
    for split_payload in split_payloads:
        split_name = str(split_payload["split_name"])
        test_count = int(split_payload["split"]["test_count"])
        for baseline_name, summary in sorted(dict(split_payload["results"]).items()):
            rows.append(
                "| "
                + " | ".join(
                    [
                        split_name,
                        baseline_name,
                        str(test_count),
                        f"{float(summary['target_accuracy']):.3f}",
                        f"{float(summary['safe_prediction_rate']):.3f}",
                        "n/a" if summary["mean_safe_bytes_regret"] is None else f"{float(summary['mean_safe_bytes_regret']):.1f}",
                        "n/a" if summary["mean_predicted_total_bytes"] is None else f"{float(summary['mean_predicted_total_bytes']):.1f}",
                    ]
                )
                + " |"
            )
    return "\n".join(rows)


def run_selector_baseline_bakeoff(
    examples: Sequence[SelectorExample],
    *,
    candidate_examples: Sequence[SelectorCandidateExample] | None = None,
    test_fraction: float = 0.25,
    seed: int = 0,
    linear_steps: int = 400,
    linear_learning_rate: float = 0.2,
    linear_l2: float = 1e-3,
) -> dict[str, Any]:
    split = split_selector_examples(examples, test_fraction=test_fraction, seed=seed)
    train_examples = [examples[index] for index in split.train_indices]
    test_examples = [examples[index] for index in split.test_indices]
    results = _evaluate_selector_split(
        examples,
        split,
        candidate_examples=candidate_examples,
        linear_steps=linear_steps,
        linear_learning_rate=linear_learning_rate,
        linear_l2=linear_l2,
    )
    return {
        "split": {
            "train_count": len(train_examples),
            "test_count": len(test_examples),
            "train_indices": list(split.train_indices),
            "test_indices": list(split.test_indices),
            "test_fraction": float(test_fraction),
            "seed": int(seed),
        },
        "results": {name: summary.to_dict() for name, summary in results.items()},
        "summary_markdown": render_selector_bakeoff_markdown(results),
    }


def run_selector_fixed_split_bakeoff(
    *,
    train_examples: Sequence[SelectorExample],
    test_examples: Sequence[SelectorExample],
    train_candidate_examples: Sequence[SelectorCandidateExample] | None = None,
    test_candidate_examples: Sequence[SelectorCandidateExample] | None = None,
    linear_steps: int = 400,
    linear_learning_rate: float = 0.2,
    linear_l2: float = 1e-3,
    weighted_selector_config: dict[str, Any] | None = None,
    split_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results = _evaluate_selector_train_test_examples(
        train_examples=train_examples,
        test_examples=test_examples,
        train_candidate_examples=train_candidate_examples,
        test_candidate_examples=test_candidate_examples,
        linear_steps=linear_steps,
        linear_learning_rate=linear_learning_rate,
        linear_l2=linear_l2,
        weighted_selector_config=weighted_selector_config,
    )
    return {
        "split": {
            "split_type": "fixed",
            "train_count": len(train_examples),
            "test_count": len(test_examples),
            "split_metadata": {} if split_metadata is None else dict(split_metadata),
        },
        "results": {name: summary.to_dict() for name, summary in results.items()},
        "summary_markdown": render_selector_bakeoff_markdown(results),
    }


def run_selector_fixed_split_batch_bakeoff(
    *,
    split_dirs: Sequence[str | Path],
    linear_steps: int = 400,
    linear_learning_rate: float = 0.2,
    linear_l2: float = 1e-3,
    weighted_selector_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    split_payloads: list[dict[str, Any]] = []
    for split_dir in split_dirs:
        split_examples = load_selector_split_examples(split_dir=split_dir)
        split_summary = split_examples["split_summary"] or {}
        split_name = str(split_summary.get("split_name") or Path(split_examples["split_dir"]).name)
        payload = run_selector_fixed_split_bakeoff(
            train_examples=split_examples["train_examples"],
            test_examples=split_examples["test_examples"],
            train_candidate_examples=split_examples["train_candidate_examples"],
            test_candidate_examples=split_examples["test_candidate_examples"],
            linear_steps=linear_steps,
            linear_learning_rate=linear_learning_rate,
            linear_l2=linear_l2,
            weighted_selector_config=weighted_selector_config,
            split_metadata=split_summary,
        )
        split_payloads.append(
            {
                "split_name": split_name,
                "split_dir": str(split_examples["split_dir"]),
                "split": payload["split"],
                "results": payload["results"],
                "summary_markdown": payload["summary_markdown"],
            }
        )
    aggregate_results = _aggregate_bakeoff_results([split_payload["results"] for split_payload in split_payloads])
    return {
        "split_count": len(split_payloads),
        "splits": split_payloads,
        "aggregate_results": aggregate_results,
        "summary_markdown": render_selector_fixed_split_batch_markdown(split_payloads),
        "aggregate_markdown": render_selector_aggregate_markdown(aggregate_results),
    }


def run_selector_multiseed_bakeoff(
    examples: Sequence[SelectorExample],
    *,
    candidate_examples: Sequence[SelectorCandidateExample] | None = None,
    seeds: Sequence[int],
    test_fraction: float = 0.25,
    linear_steps: int = 400,
    linear_learning_rate: float = 0.2,
    linear_l2: float = 1e-3,
) -> dict[str, Any]:
    resolved_seeds = [int(seed) for seed in seeds]
    folds: list[dict[str, Any]] = []
    for seed in resolved_seeds:
        payload = run_selector_baseline_bakeoff(
            examples,
            candidate_examples=candidate_examples,
            test_fraction=test_fraction,
            seed=seed,
            linear_steps=linear_steps,
            linear_learning_rate=linear_learning_rate,
            linear_l2=linear_l2,
        )
        folds.append({"fold_name": f"seed_{seed}", **payload})
    aggregate_results = _aggregate_bakeoff_results([fold["results"] for fold in folds])
    return {
        "evaluation_mode": "multiseed",
        "seeds": resolved_seeds,
        "test_fraction": float(test_fraction),
        "folds": folds,
        "aggregate_results": aggregate_results,
        "summary_markdown": render_selector_aggregate_markdown(aggregate_results),
    }


def run_selector_leave_layer_out_bakeoff(
    examples: Sequence[SelectorExample],
    *,
    candidate_examples: Sequence[SelectorCandidateExample] | None = None,
    linear_steps: int = 400,
    linear_learning_rate: float = 0.2,
    linear_l2: float = 1e-3,
) -> dict[str, Any]:
    return _run_selector_group_holdout_bakeoff(
        examples,
        candidate_examples=candidate_examples,
        group_values=sorted({int(example.layer_id) for example in examples}),
        group_key=lambda example: int(example.layer_id),
        group_label="layer",
        group_values_label="held_out_layers",
        evaluation_mode="leave_layer_out",
        linear_steps=linear_steps,
        linear_learning_rate=linear_learning_rate,
        linear_l2=linear_l2,
    )


def run_selector_leave_prompt_family_out_bakeoff(
    examples: Sequence[SelectorExample],
    *,
    candidate_examples: Sequence[SelectorCandidateExample] | None = None,
    linear_steps: int = 400,
    linear_learning_rate: float = 0.2,
    linear_l2: float = 1e-3,
) -> dict[str, Any]:
    normalized_families = sorted({_group_token(example.prompt_family) for example in examples})
    return _run_selector_group_holdout_bakeoff(
        examples,
        candidate_examples=candidate_examples,
        group_values=normalized_families,
        group_key=lambda example: _group_token(example.prompt_family),
        group_label="prompt_family",
        group_values_label="held_out_prompt_families",
        evaluation_mode="leave_prompt_family_out",
        linear_steps=linear_steps,
        linear_learning_rate=linear_learning_rate,
        linear_l2=linear_l2,
    )


def run_selector_leave_prompt_variant_out_bakeoff(
    examples: Sequence[SelectorExample],
    *,
    candidate_examples: Sequence[SelectorCandidateExample] | None = None,
    linear_steps: int = 400,
    linear_learning_rate: float = 0.2,
    linear_l2: float = 1e-3,
) -> dict[str, Any]:
    normalized_variants = sorted({_group_token(example.prompt_variant) for example in examples})
    return _run_selector_group_holdout_bakeoff(
        examples,
        candidate_examples=candidate_examples,
        group_values=normalized_variants,
        group_key=lambda example: _group_token(example.prompt_variant),
        group_label="prompt_variant",
        group_values_label="held_out_prompt_variants",
        evaluation_mode="leave_prompt_variant_out",
        linear_steps=linear_steps,
        linear_learning_rate=linear_learning_rate,
        linear_l2=linear_l2,
    )


def run_selector_leave_prompt_family_layer_out_bakeoff(
    examples: Sequence[SelectorExample],
    *,
    candidate_examples: Sequence[SelectorCandidateExample] | None = None,
    linear_steps: int = 400,
    linear_learning_rate: float = 0.2,
    linear_l2: float = 1e-3,
) -> dict[str, Any]:
    grouped_values = sorted(
        {
            (_group_token(example.prompt_family), int(example.layer_id))
            for example in examples
        }
    )
    return _run_selector_group_holdout_bakeoff(
        examples,
        candidate_examples=candidate_examples,
        group_values=grouped_values,
        group_key=lambda example: (_group_token(example.prompt_family), int(example.layer_id)),
        group_label="prompt_family_layer",
        group_values_label="held_out_prompt_family_layers",
        evaluation_mode="leave_prompt_family_layer_out",
        fold_name_fn=lambda value: f"prompt_family_{value[0]}_layer_{value[1]}",
        fold_metadata_builder=lambda value: {
            "held_out_prompt_family": value[0],
            "held_out_layer": int(value[1]),
        },
        linear_steps=linear_steps,
        linear_learning_rate=linear_learning_rate,
        linear_l2=linear_l2,
    )


def _run_selector_group_holdout_bakeoff(
    examples: Sequence[SelectorExample],
    *,
    candidate_examples: Sequence[SelectorCandidateExample] | None,
    group_values: Sequence[Any],
    group_key,
    group_label: str,
    group_values_label: str,
    evaluation_mode: str,
    fold_name_fn=None,
    fold_metadata_builder=None,
    linear_steps: int,
    linear_learning_rate: float,
    linear_l2: float,
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for held_out_group in group_values:
        train_indices = tuple(index for index, example in enumerate(examples) if group_key(example) != held_out_group)
        test_indices = tuple(index for index, example in enumerate(examples) if group_key(example) == held_out_group)
        if not train_indices or not test_indices:
            continue
        split = SelectorSplit(train_indices=train_indices, test_indices=test_indices)
        results = _evaluate_selector_split(
            examples,
            split,
            candidate_examples=candidate_examples,
            linear_steps=linear_steps,
            linear_learning_rate=linear_learning_rate,
            linear_l2=linear_l2,
        )
        fold_name = f"{group_label}_{held_out_group}" if fold_name_fn is None else str(fold_name_fn(held_out_group))
        fold_metadata = (
            {f"held_out_{group_label}": held_out_group}
            if fold_metadata_builder is None
            else dict(fold_metadata_builder(held_out_group))
        )
        folds.append(
            {
                "fold_name": fold_name,
                **fold_metadata,
                "split": {
                    "train_count": len(train_indices),
                    "test_count": len(test_indices),
                    "train_indices": list(train_indices),
                    "test_indices": list(test_indices),
                },
                "results": {name: summary.to_dict() for name, summary in results.items()},
                "summary_markdown": render_selector_bakeoff_markdown(results),
            }
        )
    aggregate_results = _aggregate_bakeoff_results([fold["results"] for fold in folds])
    held_out_groups = (
        [
            {
                key: value
                for key, value in fold.items()
                if key.startswith("held_out_")
            }
            for fold in folds
        ]
        if fold_metadata_builder is not None
        else [fold[f"held_out_{group_label}"] for fold in folds]
    )
    return {
        "evaluation_mode": evaluation_mode,
        group_values_label: held_out_groups,
        "folds": folds,
        "aggregate_results": aggregate_results,
        "summary_markdown": render_selector_aggregate_markdown(aggregate_results),
    }


def _majority_target(examples: Sequence[SelectorExample]) -> str | None:
    counter = Counter(str(example.target_candidate) for example in examples if example.target_candidate is not None)
    if not counter:
        return None
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _age_bucket(token_age: int) -> int:
    return int(np.floor(np.log2(max(int(token_age), 0) + 1)))


def _evaluate_selector_split(
    examples: Sequence[SelectorExample],
    split: SelectorSplit,
    *,
    candidate_examples: Sequence[SelectorCandidateExample] | None,
    linear_steps: int,
    linear_learning_rate: float,
    linear_l2: float,
) -> dict[str, SelectorEvaluationSummary]:
    train_examples = [examples[index] for index in split.train_indices]
    test_examples = [examples[index] for index in split.test_indices]
    train_candidate_examples = None
    test_candidate_examples = None
    if candidate_examples is not None:
        train_trace_paths = {examples[index].trace_path for index in split.train_indices}
        test_trace_paths = {examples[index].trace_path for index in split.test_indices}
        train_candidate_examples = [example for example in candidate_examples if example.trace_path in train_trace_paths]
        test_candidate_examples = [example for example in candidate_examples if example.trace_path in test_trace_paths]

    return _evaluate_selector_train_test_examples(
        train_examples=train_examples,
        test_examples=test_examples,
        train_candidate_examples=train_candidate_examples,
        test_candidate_examples=test_candidate_examples,
        linear_steps=linear_steps,
        linear_learning_rate=linear_learning_rate,
        linear_l2=linear_l2,
    )


def _evaluate_selector_train_test_examples(
    *,
    train_examples: Sequence[SelectorExample],
    test_examples: Sequence[SelectorExample],
    train_candidate_examples: Sequence[SelectorCandidateExample] | None,
    test_candidate_examples: Sequence[SelectorCandidateExample] | None,
    linear_steps: int,
    linear_learning_rate: float,
    linear_l2: float,
    weighted_selector_config: dict[str, Any] | None = None,
) -> dict[str, SelectorEvaluationSummary]:
    static_model = train_static_rule_selector(train_examples)
    linear_model = train_runtime_linear_selector(
        train_examples,
        steps=linear_steps,
        learning_rate=linear_learning_rate,
        l2=linear_l2,
    )
    results: dict[str, SelectorEvaluationSummary] = {
        "static_rule": evaluate_selector_model(static_model, test_examples),
        "linear_softmax": evaluate_selector_model(linear_model, test_examples),
    }
    if weighted_selector_config is not None:
        weighted_class_balance = float(weighted_selector_config.get("class_balance", 0.0))
        weighted_safe_bytes_weight = float(weighted_selector_config.get("safe_bytes_weight", 0.0))
        weighted_unsafe_error_weight = float(weighted_selector_config.get("unsafe_error_weight", 0.0))
        weighted_reference_candidate = str(
            weighted_selector_config.get("reference_candidate", "M3/affine/4/float16")
        )
        weighted_model = train_runtime_linear_selector(
            train_examples,
            steps=linear_steps,
            learning_rate=linear_learning_rate,
            l2=linear_l2,
            class_balance=weighted_class_balance,
            safe_bytes_weight=weighted_safe_bytes_weight,
            unsafe_error_weight=weighted_unsafe_error_weight,
            reference_candidate=weighted_reference_candidate,
        )
        results["linear_softmax_compression_weighted"] = evaluate_selector_model(weighted_model, test_examples)

        calibrated_model, calibration = _train_calibrated_linear_selector(
            train_examples=train_examples,
            steps=linear_steps,
            learning_rate=linear_learning_rate,
            l2=linear_l2,
            class_balance=weighted_class_balance,
            safe_bytes_weight=weighted_safe_bytes_weight,
            unsafe_error_weight=weighted_unsafe_error_weight,
            reference_candidate=weighted_reference_candidate,
            calibration_fraction=float(weighted_selector_config.get("calibration_fraction", 0.25)),
            calibration_seed=int(weighted_selector_config.get("calibration_seed", 0)),
            calibration_target_candidate=weighted_selector_config.get("calibration_target_candidate"),
            calibration_offsets=weighted_selector_config.get("calibration_offsets"),
            calibration_min_target_accuracy=weighted_selector_config.get("calibration_min_target_accuracy"),
            calibration_min_safe_prediction_rate=float(
                weighted_selector_config.get("calibration_min_safe_prediction_rate", 1.0)
            ),
            calibration_objective=str(weighted_selector_config.get("calibration_objective", "constraint")),
            calibration_correctness_weight=float(weighted_selector_config.get("calibration_correctness_weight", 1.0)),
            calibration_bytes_weight=float(weighted_selector_config.get("calibration_bytes_weight", 1.0)),
        )
        if calibrated_model is not None and calibration is not None:
            calibrated_summary = evaluate_selector_model(calibrated_model, test_examples)
            results["linear_softmax_compression_calibrated"] = calibrated_summary
    if train_candidate_examples is not None and test_candidate_examples is not None:
        candidate_model = train_candidate_safe_linear_selector(
            train_candidate_examples,
            steps=linear_steps,
            learning_rate=linear_learning_rate,
            l2=linear_l2,
        )
        results["candidate_linear_safe"] = evaluate_candidate_selector_model(candidate_model, test_candidate_examples)
        candidate_router_model = train_candidate_safe_router(
            train_candidate_examples,
            steps=linear_steps,
            learning_rate=linear_learning_rate,
            l2=linear_l2,
        )
        results["candidate_safe_router"] = evaluate_candidate_safe_router_model(candidate_router_model, test_candidate_examples)
    return results


def _train_calibrated_linear_selector(
    *,
    train_examples: Sequence[SelectorExample],
    steps: int,
    learning_rate: float,
    l2: float,
    class_balance: float,
    safe_bytes_weight: float,
    unsafe_error_weight: float,
    reference_candidate: str,
    calibration_fraction: float,
    calibration_seed: int,
    calibration_target_candidate: str | None,
    calibration_offsets: Sequence[float] | None,
    calibration_min_target_accuracy: float | None,
    calibration_min_safe_prediction_rate: float,
    calibration_objective: str,
    calibration_correctness_weight: float,
    calibration_bytes_weight: float,
) -> tuple[LinearSelectorModel | None, dict[str, Any] | None]:
    offsets = [] if calibration_offsets is None else [float(value) for value in calibration_offsets]
    if len(train_examples) < 2 or not offsets or float(calibration_fraction) <= 0.0:
        return None, None

    split = split_selector_examples(train_examples, test_fraction=float(calibration_fraction), seed=int(calibration_seed))
    if not split.train_indices or not split.test_indices:
        return None, None

    calibration_train_examples = [train_examples[index] for index in split.train_indices]
    calibration_examples = [train_examples[index] for index in split.test_indices]
    calibration_probe_model = train_runtime_linear_selector(
        calibration_train_examples,
        steps=steps,
        learning_rate=learning_rate,
        l2=l2,
        class_balance=class_balance,
        safe_bytes_weight=safe_bytes_weight,
        unsafe_error_weight=unsafe_error_weight,
        reference_candidate=reference_candidate,
    )
    resolved_target_candidate = _resolve_calibration_target_candidate(
        classes=calibration_probe_model.classes,
        preferred_candidate=calibration_target_candidate,
    )
    if resolved_target_candidate is None:
        return None, None

    resolved_calibration_objective = str(calibration_objective).strip().lower()
    if resolved_calibration_objective == "constraint":
        calibration = calibrate_selector_logit_offset(
            calibration_probe_model,
            calibration_examples,
            target_candidate=resolved_target_candidate,
            offsets=offsets,
            min_target_accuracy=calibration_min_target_accuracy,
            min_safe_prediction_rate=calibration_min_safe_prediction_rate,
        )
    elif resolved_calibration_objective == "equal_tradeoff":
        calibration = calibrate_selector_logit_offset_tradeoff(
            calibration_probe_model,
            calibration_examples,
            target_candidate=resolved_target_candidate,
            offsets=offsets,
            correctness_weight=calibration_correctness_weight,
            bytes_weight=calibration_bytes_weight,
        )
    else:
        raise ValueError(f"unsupported calibration_objective: {calibration_objective}")
    best_offset = float(calibration["best"]["logit_offset"])
    full_train_model = train_runtime_linear_selector(
        train_examples,
        steps=steps,
        learning_rate=learning_rate,
        l2=l2,
        class_balance=class_balance,
        safe_bytes_weight=safe_bytes_weight,
        unsafe_error_weight=unsafe_error_weight,
        reference_candidate=reference_candidate,
    )
    if resolved_target_candidate not in full_train_model.classes:
        return None, None
    return (
        adjust_linear_selector_model_logits(
            full_train_model,
            candidate_logit_offsets={resolved_target_candidate: best_offset},
        ),
        calibration,
    )


def _resolve_calibration_target_candidate(
    *,
    classes: Sequence[str],
    preferred_candidate: str | None,
) -> str | None:
    class_set = {str(candidate) for candidate in classes}
    if preferred_candidate is not None and str(preferred_candidate) in class_set:
        return str(preferred_candidate)
    for candidate in sorted(class_set):
        if candidate.startswith("M3/"):
            return candidate
    return None


def _aggregate_bakeoff_results(results_payloads: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    baseline_names = sorted({baseline_name for payload in results_payloads for baseline_name in payload.keys()})
    aggregate_results: dict[str, dict[str, Any]] = {}
    for baseline_name in baseline_names:
        rows = [payload[baseline_name] for payload in results_payloads if baseline_name in payload]
        if not rows:
            continue
        target_accuracies = np.asarray([float(row["target_accuracy"]) for row in rows], dtype=np.float32)
        safe_prediction_rates = np.asarray([float(row["safe_prediction_rate"]) for row in rows], dtype=np.float32)
        safe_regrets = [row["mean_safe_bytes_regret"] for row in rows if row.get("mean_safe_bytes_regret") is not None]
        predicted_total_bytes = [row["mean_predicted_total_bytes"] for row in rows if row.get("mean_predicted_total_bytes") is not None]
        aggregate_results[baseline_name] = {
            "fold_count": len(rows),
            "mean_target_accuracy": float(np.mean(target_accuracies)),
            "std_target_accuracy": float(np.std(target_accuracies)),
            "mean_safe_prediction_rate": float(np.mean(safe_prediction_rates)),
            "std_safe_prediction_rate": float(np.std(safe_prediction_rates)),
            "mean_safe_bytes_regret": None if not safe_regrets else float(np.mean(np.asarray(safe_regrets, dtype=np.float32))),
            "std_safe_bytes_regret": None if not safe_regrets else float(np.std(np.asarray(safe_regrets, dtype=np.float32))),
            "mean_predicted_total_bytes": None if not predicted_total_bytes else float(np.mean(np.asarray(predicted_total_bytes, dtype=np.float32))),
            "std_predicted_total_bytes": None if not predicted_total_bytes else float(np.std(np.asarray(predicted_total_bytes, dtype=np.float32))),
        }
    return aggregate_results


def _stratified_split_with_key(
    examples: Sequence[SelectorExample],
    *,
    test_fraction: float,
    seed: int,
    key_fn,
) -> SelectorSplit:
    rng = np.random.default_rng(int(seed))
    grouped_indices: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        key = key_fn(example)
        grouped_indices[tuple(key) if isinstance(key, tuple) else (key,)].append(index)

    train_indices: list[int] = []
    test_indices: list[int] = []
    for key in sorted(grouped_indices):
        indices = list(grouped_indices[key])
        if len(indices) > 1:
            order = rng.permutation(len(indices)).tolist()
            indices = [indices[position] for position in order]
        test_count = int(round(len(indices) * float(test_fraction)))
        if test_count <= 0 and len(indices) > 1:
            test_count = 1
        if test_count >= len(indices):
            test_count = max(len(indices) - 1, 0)
        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])

    train_indices.sort()
    test_indices.sort()
    return SelectorSplit(train_indices=tuple(train_indices), test_indices=tuple(test_indices))


def _random_split(
    examples: Sequence[SelectorExample],
    *,
    test_fraction: float,
    seed: int,
) -> SelectorSplit:
    if len(examples) <= 1:
        return SelectorSplit(train_indices=tuple(range(len(examples))), test_indices=())
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(examples)).tolist()
    test_count = int(round(len(examples) * float(test_fraction)))
    test_count = min(max(test_count, 1), len(examples) - 1)
    test_indices = tuple(sorted(order[:test_count]))
    train_indices = tuple(sorted(order[test_count:]))
    return SelectorSplit(train_indices=train_indices, test_indices=test_indices)


def _feature_vector(example: SelectorExample, *, feature_names: Sequence[str]) -> np.ndarray:
    return selector_feature_vector_from_row(example.row, feature_names=feature_names)


def _candidate_feature_vector(example: SelectorCandidateExample, *, feature_names: Sequence[str]) -> np.ndarray:
    return selector_candidate_feature_vector_from_row(example.row, feature_names=feature_names)


def normalize_selector_categorical_token(value: Any) -> str | None:
    return _normalize_categorical_token(value)


def selector_feature_names_from_examples(
    examples: Sequence[SelectorExample],
    *,
    feature_set_id: str = "research_extended",
) -> tuple[str, ...]:
    resolved_feature_set_id = str(feature_set_id)
    if resolved_feature_set_id == "runtime_safe":
        return _runtime_selector_feature_names_from_examples(examples)
    if resolved_feature_set_id == "research_extended":
        return _selector_feature_names_from_examples(examples)
    raise ValueError(f"unsupported selector feature_set_id: {feature_set_id}")


def candidate_feature_names_from_examples(
    examples: Sequence[SelectorCandidateExample],
    *,
    feature_set_id: str = "research_extended",
) -> tuple[str, ...]:
    resolved_feature_set_id = str(feature_set_id)
    if resolved_feature_set_id == "runtime_safe":
        return _runtime_candidate_feature_names_from_examples(examples)
    if resolved_feature_set_id == "research_extended":
        return _candidate_feature_names_from_examples(examples)
    raise ValueError(f"unsupported candidate feature_set_id: {feature_set_id}")


def _selector_feature_names_from_examples(examples: Sequence[SelectorExample]) -> tuple[str, ...]:
    prompt_families = sorted(
        {
            normalized
            for normalized in (_normalize_categorical_token(example.row.get("prompt_family")) for example in examples)
            if normalized is not None
        }
    )
    prompt_variants = sorted(
        {
            normalized
            for normalized in (_normalize_categorical_token(example.row.get("prompt_variant")) for example in examples)
            if normalized is not None
        }
    )
    return (
        *tuple(_BASE_SELECTOR_FEATURE_NAMES),
        *tuple(_RESEARCH_SELECTOR_EXTRA_FEATURE_NAMES),
        *tuple(f"family_{family}" for family in prompt_families),
        *tuple(f"variant_{variant}" for variant in prompt_variants),
    )


def _runtime_selector_feature_names_from_examples(examples: Sequence[SelectorExample]) -> tuple[str, ...]:
    prompt_families = sorted(
        {
            normalized
            for normalized in (_normalize_categorical_token(example.row.get("prompt_family")) for example in examples)
            if normalized is not None
        }
    )
    prompt_variants = sorted(
        {
            normalized
            for normalized in (_normalize_categorical_token(example.row.get("prompt_variant")) for example in examples)
            if normalized is not None
        }
    )
    return (
        *tuple(RUNTIME_SELECTOR_FEATURE_NAMES),
        *tuple(f"family_{family}" for family in prompt_families),
        *tuple(f"variant_{variant}" for variant in prompt_variants),
    )


def _candidate_feature_names_from_examples(examples: Sequence[SelectorCandidateExample]) -> tuple[str, ...]:
    prompt_families = sorted(
        {
            normalized
            for normalized in (_normalize_categorical_token(example.row.get("prompt_family")) for example in examples)
            if normalized is not None
        }
    )
    prompt_variants = sorted(
        {
            normalized
            for normalized in (_normalize_categorical_token(example.row.get("prompt_variant")) for example in examples)
            if normalized is not None
        }
    )
    return (
        *tuple(_BASE_CANDIDATE_FEATURE_NAMES),
        *tuple(_RESEARCH_CANDIDATE_EXTRA_FEATURE_NAMES),
        *tuple(f"family_{family}" for family in prompt_families),
        *tuple(f"variant_{variant}" for variant in prompt_variants),
    )


def _runtime_candidate_feature_names_from_examples(examples: Sequence[SelectorCandidateExample]) -> tuple[str, ...]:
    prompt_families = sorted(
        {
            normalized
            for normalized in (_normalize_categorical_token(example.row.get("prompt_family")) for example in examples)
            if normalized is not None
        }
    )
    prompt_variants = sorted(
        {
            normalized
            for normalized in (_normalize_categorical_token(example.row.get("prompt_variant")) for example in examples)
            if normalized is not None
        }
    )
    return (
        *tuple(_RUNTIME_CANDIDATE_FEATURE_NAMES),
        *tuple(f"family_{family}" for family in prompt_families),
        *tuple(f"variant_{variant}" for variant in prompt_variants),
    )


def _normalize_categorical_token(value: Any) -> str | None:
    if value in (None, ""):
        return None
    normalized = "".join(character if str(character).isalnum() else "_" for character in str(value).strip().lower())
    normalized = normalized.strip("_")
    return normalized or None


def _group_token(value: Any) -> str:
    normalized = _normalize_categorical_token(value)
    return "__none__" if normalized is None else normalized


def _compression_gain_ratio(
    example: SelectorExample,
    *,
    reference_candidate: str,
) -> float:
    if example.best_safe_total_bytes is None:
        return 0.0
    reference_payload = example.candidate_map.get(str(reference_candidate))
    if reference_payload is not None:
        reference_bytes = int(reference_payload.get("total_bytes", 0))
    else:
        reference_bytes = max(
            (int(payload.get("total_bytes", 0)) for payload in example.candidate_map.values()),
            default=0,
        )
    if reference_bytes <= 0:
        return 0.0
    return max(float(reference_bytes - int(example.best_safe_total_bytes)) / float(reference_bytes), 0.0)


def _candidate_target_trace_weight(
    *,
    trace_indices: Sequence[int],
    target_example: SelectorCandidateExample,
    examples: Sequence[SelectorCandidateExample],
    class_counts: Counter[str] | None,
    total_group_count: int,
    class_count: int,
    class_balance: float,
    reference_candidate: str,
    non_reference_target_weight: float,
    compression_target_weight: float,
) -> float:
    weight = 1.0
    target_candidate = str(target_example.candidate)
    if float(class_balance) > 0.0 and class_counts is not None and total_group_count > 0 and class_count > 0:
        balanced = float(total_group_count) / float(class_count * max(class_counts.get(target_candidate, 0), 1))
        weight *= balanced ** float(class_balance)
    if target_candidate != str(reference_candidate):
        weight *= 1.0 + max(float(non_reference_target_weight), 0.0)
    if float(compression_target_weight) > 0.0:
        reference_bytes = None
        for index in trace_indices:
            example = examples[int(index)]
            if str(example.candidate) == str(reference_candidate):
                reference_bytes = int(example.candidate_total_bytes)
                break
        if reference_bytes is None:
            reference_bytes = max((int(examples[int(index)].candidate_total_bytes) for index in trace_indices), default=0)
        if reference_bytes > 0:
            gain = max(float(reference_bytes - int(target_example.candidate_total_bytes)) / float(reference_bytes), 0.0)
            weight *= 1.0 + float(compression_target_weight) * gain
    return float(weight)


def _apply_candidate_logit_offset(probability: float, logit_offset: float) -> float:
    resolved_probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    resolved_offset = float(logit_offset)
    if abs(resolved_offset) < 1e-9:
        return resolved_probability
    logit = float(np.log(resolved_probability) - np.log1p(-resolved_probability))
    adjusted_logit = logit + resolved_offset
    return float(1.0 / (1.0 + np.exp(-adjusted_logit)))


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    stabilized = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(stabilized).astype(np.float32, copy=False)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def _resolve_feature_value(
    base_values: dict[str, float],
    feature_name: str,
    *,
    prompt_family: str | None,
    prompt_variant: str | None,
) -> float:
    if feature_name in base_values:
        return float(base_values[feature_name])
    if feature_name.startswith("family_"):
        return 1.0 if prompt_family == feature_name.removeprefix("family_") else 0.0
    if feature_name.startswith("variant_"):
        return 1.0 if prompt_variant == feature_name.removeprefix("variant_") else 0.0
    raise KeyError(f"unknown feature name: {feature_name}")


def _selector_base_feature_values_from_row(row: dict[str, Any]) -> dict[str, float]:
    stage_decode = 1.0 if str(row.get("stage", "")) == "decode" else 0.0
    kind_key = 1.0 if str(row.get("kind", "")) == "K" else 0.0
    query_present = 1.0 if bool(row.get("query_present", False)) else 0.0
    token_start = float(row.get("token_start", 0.0))
    token_age = float(row.get("token_age", 0.0))
    token_count = max(float(row.get("token_count", 0.0)), 0.0)
    sequence_length = max(token_start + token_count + token_age, token_count, 1.0)
    page_distance = token_age / max(token_count, 1.0)
    token_end_fraction = min(max((token_start + token_count) / sequence_length, 0.0), 1.0)
    token_age_fraction = min(max(token_age / sequence_length, 0.0), 1.0)
    old_page_indicator = 1.0 if token_age >= max(token_count, 1.0) else 0.0
    best_safe_total_bytes = max(float(row.get("best_safe_total_bytes", 0.0)), 0.0)
    reference_candidate_total_bytes = max(float(row.get("reference_candidate_total_bytes", 0.0)), 0.0)
    if reference_candidate_total_bytes <= 0.0:
        reference_candidate_total_bytes = max(float(row.get("candidate_total_bytes", 0.0)), 0.0)
    compression_gain_vs_m3 = 0.0
    if best_safe_total_bytes > 0.0:
        reference_total = reference_candidate_total_bytes if reference_candidate_total_bytes > 0.0 else best_safe_total_bytes
        compression_gain_vs_m3 = max(float(reference_total - best_safe_total_bytes) / max(reference_total, 1.0), 0.0)
    return {
        "stage_decode": stage_decode,
        "kind_key": kind_key,
        "query_present": query_present,
        "layer_fraction": float(row.get("layer_fraction", 0.0)),
        "kv_head_fraction": float(row.get("kv_head_fraction", 0.0)),
        "log_sequence_length": float(np.log1p(sequence_length)),
        "log_token_start": float(np.log1p(token_start)),
        "log_token_age": float(np.log1p(token_age)),
        "token_count": token_count,
        "head_dim": float(row.get("head_dim", 0.0)),
        "safe_candidate_count": float(row.get("safe_candidate_count", 0.0)),
        "log_best_safe_total_bytes": float(np.log1p(best_safe_total_bytes)),
        "trace_rms": float(row.get("trace_rms", 0.0)),
        "log_trace_abs_max": float(np.log1p(float(row.get("trace_abs_max", 0.0)))),
        "trace_channel_range_mean": float(row.get("trace_channel_range_mean", 0.0)),
        "trace_outlier_fraction": float(row.get("trace_outlier_fraction", 0.0)),
        "age_per_token": float(row.get("age_per_token", 0.0)),
        "page_distance": page_distance,
        "log_page_distance": float(np.log1p(page_distance)),
        "page_distance_ge_2": 1.0 if page_distance >= 2.0 else 0.0,
        "page_distance_ge_4": 1.0 if page_distance >= 4.0 else 0.0,
        "page_distance_ge_8": 1.0 if page_distance >= 8.0 else 0.0,
        "token_end_fraction": token_end_fraction,
        "token_age_fraction": token_age_fraction,
        "age_bucket_ge_64": 1.0 if token_age >= 64.0 else 0.0,
        "age_bucket_ge_256": 1.0 if token_age >= 256.0 else 0.0,
        "age_bucket_ge_1024": 1.0 if token_age >= 1024.0 else 0.0,
        "sequence_length_ge_512": 1.0 if sequence_length >= 512.0 else 0.0,
        "sequence_length_ge_1024": 1.0 if sequence_length >= 1024.0 else 0.0,
        "sequence_length_ge_2048": 1.0 if sequence_length >= 2048.0 else 0.0,
        "decode_old_page_indicator": stage_decode * old_page_indicator,
        "decode_long_context_indicator": stage_decode * (1.0 if sequence_length >= 1024.0 else 0.0),
        "decode_key_indicator": stage_decode * kind_key,
        "compression_gain_vs_m3": compression_gain_vs_m3,
    }


def estimate_runtime_candidate_storage(
    row: dict[str, Any],
    *,
    candidate_token: str,
    group_size: int = 32,
    payload_layout_k: str = "group_major",
    payload_layout_v: str = "group_major",
    escape_dtype: str = "float16",
) -> dict[str, Any] | None:
    candidate = parse_page_mode_token(candidate_token)
    mode = str(candidate.mode)
    if mode not in {"M0", "M3"}:
        return None

    head_dim = int(row.get("head_dim", 0))
    token_count = int(row.get("token_count", 0))
    kind = str(row.get("kind", "K"))
    layer_id = int(row.get("layer_id", 0))
    kv_head_id = int(row.get("kv_head_id", 0))
    token_start = int(row.get("token_start", 0))
    num_groups = max(ceil(head_dim / max(int(group_size), 1)), 1)
    padded_head_dim = num_groups * max(int(group_size), 1)
    bits = int(candidate.bits)
    scheme = str(candidate.quant_scheme)
    resolved_escape_dtype = str(candidate.escape_dtype or escape_dtype)
    layout = str(payload_layout_k if kind == "K" else payload_layout_v)
    header = PageHeader(
        layer_id=layer_id,
        kv_head_id=kv_head_id,
        kind="K" if kind == "K" else "V",
        token_start=token_start,
        token_count=token_count,
        head_dim=head_dim,
        padded_head_dim=padded_head_dim,
        group_size=int(group_size),
        num_groups=num_groups,
        bits=bits,
        words_per_group=0 if mode == "M3" else words_per_group(int(group_size), bits),
        mode_default=mode,
        layout=layout,
        quant_scheme=scheme,
        escape_dtype=resolved_escape_dtype,
    )
    metadata_bytes = len(header.to_json().encode("utf-8"))
    payload_bytes = 0
    if mode == "M3":
        payload_dtype_size = int(np.dtype(resolved_escape_dtype).itemsize)
        payload_bytes = int(token_count * head_dim * payload_dtype_size)
        if resolved_escape_dtype == "int8":
            metadata_bytes += int(token_count * np.dtype(np.float16).itemsize)
    elif mode == "M0":
        payload_bytes = int(token_count * num_groups * words_per_group(int(group_size), bits) * np.dtype(np.uint32).itemsize)
        scale_bytes = int(token_count * num_groups * np.dtype(np.float16).itemsize)
        metadata_bytes += scale_bytes
        if scheme == "affine":
            metadata_bytes += scale_bytes
    return {
        "candidate": candidate_token,
        "candidate_mode": mode,
        "candidate_bits": bits,
        "candidate_quant_scheme": scheme,
        "candidate_total_bytes": int(payload_bytes + metadata_bytes),
        "candidate_payload_bytes": int(payload_bytes),
        "candidate_metadata_bytes": int(metadata_bytes),
        "candidate_has_escape_dtype": bool(candidate.escape_dtype is not None),
    }


def build_runtime_selector_candidate_row(
    row: dict[str, Any],
    *,
    candidate_token: str,
    group_size: int = 32,
    payload_layout_k: str = "group_major",
    payload_layout_v: str = "group_major",
    escape_dtype: str = "float16",
) -> dict[str, Any] | None:
    candidate_storage = estimate_runtime_candidate_storage(
        row,
        candidate_token=candidate_token,
        group_size=group_size,
        payload_layout_k=payload_layout_k,
        payload_layout_v=payload_layout_v,
        escape_dtype=escape_dtype,
    )
    if candidate_storage is None:
        return None
    candidate_row = dict(row)
    candidate_row.update(candidate_storage)
    return candidate_row


def build_selector_class_error_weights(
    examples: Sequence[SelectorExample],
    *,
    classes: Sequence[str],
    unsafe_error_weight: float = 0.0,
) -> np.ndarray:
    target_examples = [example for example in examples if example.target_present and example.target_candidate is not None]
    if not target_examples:
        return np.zeros((0, len(tuple(classes))), dtype=np.float32)

    resolved_classes = tuple(str(candidate) for candidate in classes)
    weights = np.ones((len(target_examples), len(resolved_classes)), dtype=np.float32)
    if float(unsafe_error_weight) <= 0.0:
        return weights

    unsafe_multiplier = 1.0 + float(unsafe_error_weight)
    for row_index, example in enumerate(target_examples):
        target_candidate = str(example.target_candidate)
        for class_index, candidate in enumerate(resolved_classes):
            if candidate == target_candidate:
                continue
            candidate_payload = example.candidate_map.get(candidate)
            candidate_safe = bool(candidate_payload is not None and candidate_payload.get("safe", False))
            if not candidate_safe:
                weights[row_index, class_index] = unsafe_multiplier
    return weights


@dataclass(slots=True)
class _PredictedSelectorModel:
    predictions: dict[str, str | None]

    def predict(self, example: SelectorExample) -> str | None:
        return self.predictions.get(example.trace_path)


def _softmax(logits: np.ndarray) -> np.ndarray:
    stabilized = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(stabilized).astype(np.float32, copy=False)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
