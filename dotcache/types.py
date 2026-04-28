from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

Kind = Literal["K", "V"]
Layout = Literal["group_major", "token_major"]
Mode = Literal["M0", "M1", "M2", "M3", "M4", "T3"]
QuantScheme = Literal["affine", "symmetric", "lut", "sketch", "project", "turbo3"]


@dataclass(slots=True)
class PageHeader:
    layer_id: int
    kv_head_id: int
    kind: Kind
    token_start: int
    token_count: int
    head_dim: int
    padded_head_dim: int
    group_size: int
    num_groups: int
    bits: int
    words_per_group: int
    mode_default: Mode
    layout: Layout
    quant_scheme: QuantScheme
    policy_id: str = "exact_baseline"
    sensitivity_tier: str = "exact"
    fallback_reason: str = ""
    age_bucket: str = "aged"
    escape_dtype: str = "float16"
    project_basis: str = "hadamard"

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, int | str]) -> "PageHeader":
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "PageHeader":
        return cls.from_dict(json.loads(payload))


@dataclass(slots=True)
class EncodedPage:
    header: PageHeader
    payload: np.ndarray | None = None
    scales: np.ndarray | None = None
    bias: np.ndarray | None = None
    codebooks: np.ndarray | None = None
    m2_sketch: np.ndarray | None = None
    m2_basis: np.ndarray | None = None
    m2_mean: np.ndarray | None = None
    lut_segment_count: int = 1
    escape_payload: np.ndarray | None = None
    escape_scales: np.ndarray | None = None
    requested_mode: str | None = None
    trial_quant_error: float | None = None
    trial_token_p95_error: float | None = None
    runtime_page_mean: np.ndarray | None = None
    runtime_page_sketch: np.ndarray | None = None
    runtime_page_min: np.ndarray | None = None
    runtime_page_max: np.ndarray | None = None
    full_page_decode_calls: int = 0
    decode_group_calls: int = 0

    @property
    def payload_nbytes(self) -> int:
        total = 0
        if self.payload is not None:
            total += int(self.payload.nbytes)
        if self.escape_payload is not None:
            total += int(self.escape_payload.nbytes)
        return total

    @property
    def metadata_nbytes(self) -> int:
        total = len(self.header.to_json().encode("utf-8"))
        if self.scales is not None:
            total += int(self.scales.nbytes)
        if self.bias is not None:
            total += int(self.bias.nbytes)
        if self.codebooks is not None:
            total += int(self.codebooks.nbytes)
        if self.m2_sketch is not None:
            total += int(self.m2_sketch.nbytes)
        if self.m2_basis is not None:
            total += int(self.m2_basis.nbytes)
        if self.m2_mean is not None:
            total += int(self.m2_mean.nbytes)
        if self.escape_scales is not None:
            total += int(self.escape_scales.nbytes)
        return total

    @property
    def total_nbytes(self) -> int:
        return self.payload_nbytes + self.metadata_nbytes

    def record_full_decode(self) -> None:
        self.full_page_decode_calls += 1

    def record_group_decode(self) -> None:
        self.decode_group_calls += 1
