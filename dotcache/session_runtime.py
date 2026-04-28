from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Literal, Sequence

import numpy as np
from .attention_runtime import (
    BackendName,
    decode_step_with_page_logits,
    mix_page,
    prepare_pages,
    score_page,
    score_pages,
)
from .modes.m0_affine import dequantize_group
from .modes.m1_lut import dequantize_group_lut
from .modes.m2_key_sketch import reconstruct_group_m2
from .modes.m4_key_project import reconstruct_group_m4
from .modes.m3_escape import decode_escape_payload
from .modes.turbo3 import dequantize_group_turbo3
from .page_cache import PreparedPageCache
from .page_format import load_group_words
from .packing import unpack_bits
from .tracing import ExecutionTrace
from .types import EncodedPage
from .backends import PreparedPageTorch

PageLike = EncodedPage | PreparedPageTorch
RelevanceMode = Literal["sketch", "envelope"]


def _decode_page_dense(page: PageLike) -> np.ndarray:
    source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
    header = source_page.header

    if header.mode_default == "M3":
        if source_page.escape_payload is None:
            raise ValueError("escape payload is missing")
        return np.asarray(
            decode_escape_payload(
                source_page.escape_payload,
                head_dim=header.head_dim,
                scales=source_page.escape_scales,
            ),
            dtype=np.float32,
        )

    if header.mode_default == "M2":
        if source_page.m2_sketch is None or source_page.m2_basis is None:
            raise ValueError("M2 page is missing sketch payload")
        dense = np.zeros((header.token_count, header.padded_head_dim), dtype=np.float32)
        for group_index in range(header.num_groups):
            start = group_index * header.group_size
            end = start + header.group_size
            dense[:, start:end] = reconstruct_group_m2(
                source_page.m2_sketch[:, group_index, :],
                basis=source_page.m2_basis[group_index],
                mean=None if source_page.m2_mean is None else source_page.m2_mean[group_index],
            )
        return dense[:, : header.head_dim]

    if header.mode_default == "M4":
        if source_page.m2_sketch is None or source_page.m2_mean is None:
            raise ValueError("M4 page is missing projected payload")
        dense = np.zeros((header.token_count, header.padded_head_dim), dtype=np.float32)
        for group_index in range(header.num_groups):
            start = group_index * header.group_size
            end = start + header.group_size
            dense[:, start:end] = reconstruct_group_m4(
                source_page.m2_sketch[:, group_index, :],
                mean=source_page.m2_mean[group_index],
                group_size=header.group_size,
                basis_family=header.project_basis,
                basis=None if source_page.m2_basis is None else source_page.m2_basis[group_index],
            )
        return dense[:, : header.head_dim]

    if source_page.payload is None:
        raise ValueError(f"{header.mode_default} page is missing payload")

    dense = np.zeros((header.token_count, header.padded_head_dim), dtype=np.float32)
    for group_index in range(header.num_groups):
        words = load_group_words(source_page, group_index)
        codes = unpack_bits(words, header.bits, header.group_size)
        if header.mode_default == "M1":
            if source_page.codebooks is None:
                raise ValueError("M1 page is missing codebooks")
            group_values = dequantize_group_lut(
                codes,
                codebook=np.asarray(source_page.codebooks[group_index], dtype=np.float32),
            )
        elif header.mode_default == "T3":
            if source_page.scales is None or source_page.codebooks is None:
                raise ValueError("T3 page is missing correction metadata")
            group_values = dequantize_group_turbo3(
                codes,
                correction=source_page.scales[:, group_index].astype(np.float32),
                centroids=np.asarray(source_page.codebooks, dtype=np.float32),
            )
        else:
            if source_page.scales is None:
                raise ValueError("M0 page is missing scales")
            scales = source_page.scales[:, group_index].astype(np.float32)[:, None]
            bias = None
            if source_page.bias is not None:
                bias = source_page.bias[:, group_index].astype(np.float32)[:, None]
            group_values = dequantize_group(
                codes,
                scales=scales,
                bias=bias,
                bits=header.bits,
                scheme=header.quant_scheme,
            )
        start = group_index * header.group_size
        end = start + header.group_size
        dense[:, start:end] = group_values

    return dense[:, : header.head_dim]


def sketch_key_page(page: PageLike, *, sketch_size: int = 1) -> np.ndarray:
    if sketch_size <= 0:
        raise ValueError("sketch_size must be positive")
    source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
    if source_page.runtime_page_sketch is not None:
        stored = np.asarray(source_page.runtime_page_sketch, dtype=np.float32)
        if sketch_size == 1 and source_page.runtime_page_mean is not None:
            return np.asarray(source_page.runtime_page_mean, dtype=np.float32)[None, :]
        if stored.shape[0] == sketch_size:
            return stored
        if stored.shape[0] > sketch_size and sketch_size > 1:
            chunks = np.array_split(stored, sketch_size, axis=0)
            return np.stack([chunk.mean(axis=0) for chunk in chunks], axis=0).astype(np.float32, copy=False)
    dense = _decode_page_dense(page)
    if sketch_size == 1:
        return dense.mean(axis=0, keepdims=True)
    chunks = np.array_split(dense, min(sketch_size, dense.shape[0]), axis=0)
    return np.stack([chunk.mean(axis=0) for chunk in chunks], axis=0).astype(np.float32, copy=False)


def summarize_key_page(page: PageLike) -> np.ndarray:
    return sketch_key_page(page, sketch_size=1)[0]


def summarize_value_page(page: PageLike) -> np.ndarray:
    source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
    if source_page.runtime_page_mean is not None:
        return np.asarray(source_page.runtime_page_mean, dtype=np.float32)
    return _decode_page_dense(page).mean(axis=0)


def envelope_key_page(page: PageLike) -> tuple[np.ndarray, np.ndarray]:
    source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
    if source_page.runtime_page_min is not None and source_page.runtime_page_max is not None:
        return (
            np.asarray(source_page.runtime_page_min, dtype=np.float32),
            np.asarray(source_page.runtime_page_max, dtype=np.float32),
        )
    dense = _decode_page_dense(page)
    return (
        dense.min(axis=0).astype(np.float32, copy=False),
        dense.max(axis=0).astype(np.float32, copy=False),
    )


def score_page_relevance(
    query_slice: np.ndarray,
    *,
    relevance_mode: RelevanceMode,
    page_sketch: np.ndarray | None = None,
    page_min: np.ndarray | None = None,
    page_max: np.ndarray | None = None,
) -> float:
    query = np.asarray(query_slice, dtype=np.float32)
    if relevance_mode == "sketch":
        if page_sketch is None:
            raise ValueError("sketch relevance requires page_sketch")
        return float(np.max(np.asarray(page_sketch, dtype=np.float32) @ query))
    if relevance_mode == "envelope":
        if page_min is None or page_max is None:
            raise ValueError("envelope relevance requires page_min and page_max")
        positive_query = np.maximum(query, 0.0)
        negative_query = np.minimum(query, 0.0)
        return float(
            np.asarray(page_max, dtype=np.float32) @ positive_query
            + np.asarray(page_min, dtype=np.float32) @ negative_query
        )
    raise ValueError(f"unsupported relevance_mode: {relevance_mode}")


def select_window_page_indices(
    key_pages: Sequence[PageLike],
    *,
    recent_window_tokens: int | None = None,
    sink_window_tokens: int = 0,
) -> list[int]:
    if not key_pages:
        return []

    context_end = max(page.header.token_start + page.header.token_count for page in key_pages)
    sink_end = max(0, sink_window_tokens)
    recent_start = context_end
    if recent_window_tokens is not None and recent_window_tokens > 0:
        recent_start = max(0, context_end - recent_window_tokens)

    selected_indices: set[int] = set()
    for index, page in enumerate(key_pages):
        page_start = page.header.token_start
        page_end = page_start + page.header.token_count
        in_sink = sink_end > 0 and page_start < sink_end and page_end > 0
        in_recent = recent_window_tokens is not None and recent_window_tokens > 0 and page_end > recent_start
        if in_sink or in_recent:
            selected_indices.add(index)

    return sorted(selected_indices)


def select_execution_page_indices(
    key_pages: Sequence[PageLike],
    *,
    recent_window_tokens: int | None = None,
    sink_window_tokens: int = 0,
    query_slice: np.ndarray | None = None,
    key_page_sketches: Sequence[np.ndarray] | None = None,
    key_page_sketch_matrix: np.ndarray | None = None,
    tail_page_sketch: np.ndarray | None = None,
    key_page_minima: Sequence[np.ndarray] | None = None,
    key_page_minima_matrix: np.ndarray | None = None,
    tail_page_minimum: np.ndarray | None = None,
    key_page_maxima: Sequence[np.ndarray] | None = None,
    key_page_maxima_matrix: np.ndarray | None = None,
    tail_page_maximum: np.ndarray | None = None,
    relevance_top_k: int = 0,
    relevance_mode: RelevanceMode = "sketch",
    stage_recorder: Callable[[str, float], None] | None = None,
    score_all_pages_with_matrices: bool = False,
    score_all_pages_min_candidate_fraction: float = 0.0,
    selector_stats_recorder: Callable[[dict[str, int | float | bool]], None] | None = None,
) -> list[int]:
    def _record_stage(stage: str, started_at: float | None) -> None:
        if stage_recorder is None or started_at is None:
            return
        stage_recorder(stage, (perf_counter() - started_at) * 1000.0)

    def _materialize_candidate_rows(matrix: np.ndarray, direct_candidate_indices: Sequence[int]) -> np.ndarray:
        if not direct_candidate_indices:
            return np.empty((0,) + tuple(matrix.shape[1:]), dtype=np.float32)
        first_index = int(direct_candidate_indices[0])
        last_index = int(direct_candidate_indices[-1])
        if last_index - first_index + 1 == len(direct_candidate_indices):
            return np.ascontiguousarray(matrix[first_index : last_index + 1], dtype=np.float32)
        return np.take(matrix, direct_candidate_indices, axis=0).astype(np.float32, copy=False)

    if not key_pages:
        return []
    selected_indices = set(
        select_window_page_indices(
            key_pages,
            recent_window_tokens=recent_window_tokens,
            sink_window_tokens=sink_window_tokens,
        )
    )

    if relevance_top_k > 0:
        if query_slice is None:
            raise ValueError("relevance gating requires query_slice")
        candidate_index_build_started_at = perf_counter() if stage_recorder is not None else None
        candidate_indices = [index for index in range(len(key_pages)) if index not in selected_indices]
        _record_stage("shortlist_candidate_builtin_candidate_index_build", candidate_index_build_started_at)
        if candidate_indices:
            candidate_fraction = float(len(candidate_indices)) / float(len(key_pages))
            use_score_all_pages = bool(
                score_all_pages_with_matrices
                and candidate_fraction >= max(0.0, float(score_all_pages_min_candidate_fraction))
            )
            if selector_stats_recorder is not None:
                selector_stats_recorder(
                    {
                        "candidate_pages": int(len(candidate_indices)),
                        "total_pages": int(len(key_pages)),
                        "candidate_fraction": float(candidate_fraction),
                        "used_score_all_pages": bool(use_score_all_pages),
                    }
                )
            query = np.asarray(query_slice, dtype=np.float32)
            if relevance_mode == "sketch":
                if key_page_sketch_matrix is not None:
                    expected_sketch_rows = len(key_pages) - 1 if tail_page_sketch is not None else len(key_pages)
                    if int(key_page_sketch_matrix.shape[0]) != expected_sketch_rows:
                        raise ValueError("key_page_sketch_matrix must align with key_pages")
                    if use_score_all_pages:
                        score_compute_started_at = perf_counter() if stage_recorder is not None else None
                        all_scores = np.max(key_page_sketch_matrix @ query, axis=1).astype(np.float32, copy=False)
                        if tail_page_sketch is not None:
                            tail_score = float(np.max(np.asarray(tail_page_sketch, dtype=np.float32) @ query))
                            all_scores = np.concatenate(
                                [all_scores, np.asarray([tail_score], dtype=np.float32)],
                                axis=0,
                            )
                        scores = np.asarray(all_scores[candidate_indices], dtype=np.float32)
                        _record_stage("shortlist_candidate_builtin_score_compute", score_compute_started_at)
                    else:
                        direct_candidate_indices = [index for index in candidate_indices if index < key_page_sketch_matrix.shape[0]]
                        tail_candidate_selected = (
                            tail_page_sketch is not None and len(candidate_indices) > len(direct_candidate_indices)
                        )
                        sidecar_stack_started_at = perf_counter() if stage_recorder is not None else None
                        candidate_sketches = _materialize_candidate_rows(
                            key_page_sketch_matrix,
                            direct_candidate_indices,
                        )
                        _record_stage("shortlist_candidate_builtin_sidecar_stack", sidecar_stack_started_at)
                        score_compute_started_at = perf_counter() if stage_recorder is not None else None
                        direct_scores = np.max(candidate_sketches @ query, axis=1).astype(np.float32, copy=False)
                        if tail_candidate_selected:
                            tail_score = float(np.max(np.asarray(tail_page_sketch, dtype=np.float32) @ query))
                            scores = np.concatenate(
                                [direct_scores, np.asarray([tail_score], dtype=np.float32)],
                                axis=0,
                            )
                        else:
                            scores = direct_scores
                        _record_stage("shortlist_candidate_builtin_score_compute", score_compute_started_at)
                else:
                    if key_page_sketches is None:
                        raise ValueError("sketch relevance gating requires key_page_sketches")
                    if len(key_page_sketches) != len(key_pages):
                        raise ValueError("key_page_sketches must align with key_pages")
                    sidecar_stack_started_at = perf_counter() if stage_recorder is not None else None
                    candidate_sketches = np.stack(
                        [np.asarray(key_page_sketches[index], dtype=np.float32) for index in candidate_indices],
                        axis=0,
                    )
                    _record_stage("shortlist_candidate_builtin_sidecar_stack", sidecar_stack_started_at)
                    score_compute_started_at = perf_counter() if stage_recorder is not None else None
                    scores = np.max(candidate_sketches @ query, axis=1).astype(np.float32, copy=False)
                    _record_stage("shortlist_candidate_builtin_score_compute", score_compute_started_at)
            elif relevance_mode == "envelope":
                positive_query = np.maximum(query, 0.0)
                negative_query = np.minimum(query, 0.0)
                if key_page_minima_matrix is not None and key_page_maxima_matrix is not None:
                    expected_envelope_rows = (
                        len(key_pages) - 1
                        if tail_page_minimum is not None and tail_page_maximum is not None
                        else len(key_pages)
                    )
                    if (
                        int(key_page_minima_matrix.shape[0]) != expected_envelope_rows
                        or int(key_page_maxima_matrix.shape[0]) != expected_envelope_rows
                    ):
                        raise ValueError("page minima and maxima matrices must align with key_pages")
                    if use_score_all_pages:
                        score_compute_started_at = perf_counter() if stage_recorder is not None else None
                        all_scores = (
                            key_page_maxima_matrix @ positive_query + key_page_minima_matrix @ negative_query
                        ).astype(np.float32, copy=False)
                        if tail_page_minimum is not None and tail_page_maximum is not None:
                            tail_score = float(
                                np.asarray(tail_page_maximum, dtype=np.float32) @ positive_query
                                + np.asarray(tail_page_minimum, dtype=np.float32) @ negative_query
                            )
                            all_scores = np.concatenate(
                                [all_scores, np.asarray([tail_score], dtype=np.float32)],
                                axis=0,
                            )
                        scores = np.asarray(all_scores[candidate_indices], dtype=np.float32)
                        _record_stage("shortlist_candidate_builtin_score_compute", score_compute_started_at)
                    else:
                        direct_candidate_indices = [index for index in candidate_indices if index < key_page_minima_matrix.shape[0]]
                        tail_candidate_selected = (
                            tail_page_minimum is not None
                            and tail_page_maximum is not None
                            and len(candidate_indices) > len(direct_candidate_indices)
                        )
                        sidecar_stack_started_at = perf_counter() if stage_recorder is not None else None
                        candidate_minima = _materialize_candidate_rows(
                            key_page_minima_matrix,
                            direct_candidate_indices,
                        )
                        candidate_maxima = _materialize_candidate_rows(
                            key_page_maxima_matrix,
                            direct_candidate_indices,
                        )
                        _record_stage("shortlist_candidate_builtin_sidecar_stack", sidecar_stack_started_at)
                        score_compute_started_at = perf_counter() if stage_recorder is not None else None
                        direct_scores = (
                            candidate_maxima @ positive_query + candidate_minima @ negative_query
                        ).astype(np.float32, copy=False)
                        if tail_candidate_selected:
                            tail_score = float(
                                np.asarray(tail_page_maximum, dtype=np.float32) @ positive_query
                                + np.asarray(tail_page_minimum, dtype=np.float32) @ negative_query
                            )
                            scores = np.concatenate(
                                [direct_scores, np.asarray([tail_score], dtype=np.float32)],
                                axis=0,
                            )
                        else:
                            scores = direct_scores
                        _record_stage("shortlist_candidate_builtin_score_compute", score_compute_started_at)
                else:
                    if key_page_minima is None or key_page_maxima is None:
                        raise ValueError("envelope relevance gating requires page minima and maxima")
                    if len(key_page_minima) != len(key_pages) or len(key_page_maxima) != len(key_pages):
                        raise ValueError("page minima and maxima must align with key_pages")
                    sidecar_stack_started_at = perf_counter() if stage_recorder is not None else None
                    candidate_minima = np.stack(
                        [np.asarray(key_page_minima[index], dtype=np.float32) for index in candidate_indices],
                        axis=0,
                    )
                    candidate_maxima = np.stack(
                        [np.asarray(key_page_maxima[index], dtype=np.float32) for index in candidate_indices],
                        axis=0,
                    )
                    _record_stage("shortlist_candidate_builtin_sidecar_stack", sidecar_stack_started_at)
                    score_compute_started_at = perf_counter() if stage_recorder is not None else None
                    scores = (candidate_maxima @ positive_query + candidate_minima @ negative_query).astype(
                        np.float32,
                        copy=False,
                    )
                    _record_stage("shortlist_candidate_builtin_score_compute", score_compute_started_at)
            else:
                raise ValueError(f"unsupported relevance_mode: {relevance_mode}")
            ranking_started_at = perf_counter() if stage_recorder is not None else None
            ranked_candidates = [
                index
                for _, index in sorted(
                    zip(scores.tolist(), candidate_indices, strict=True),
                    key=lambda item: item[0],
                    reverse=True,
                )
            ]
            _record_stage("shortlist_candidate_builtin_ranking", ranking_started_at)
            selected_indices.update(ranked_candidates[:relevance_top_k])

    if not selected_indices:
        return list(range(len(key_pages)))
    return sorted(selected_indices)


def select_execution_page_pairs(
    key_pages: Sequence[PageLike],
    value_pages: Sequence[PageLike],
    *,
    recent_window_tokens: int | None = None,
    sink_window_tokens: int = 0,
    query_slice: np.ndarray | None = None,
    key_page_sketches: Sequence[np.ndarray] | None = None,
    key_page_minima: Sequence[np.ndarray] | None = None,
    key_page_maxima: Sequence[np.ndarray] | None = None,
    relevance_top_k: int = 0,
    relevance_mode: RelevanceMode = "sketch",
) -> tuple[list[PageLike], list[PageLike]]:
    if len(key_pages) != len(value_pages):
        raise ValueError("key_pages and value_pages must contain the same number of pages")
    if not key_pages:
        return [], []
    if (
        (recent_window_tokens is None or recent_window_tokens <= 0)
        and sink_window_tokens <= 0
        and relevance_top_k <= 0
    ):
        return list(key_pages), list(value_pages)
    selected_indices = select_execution_page_indices(
        key_pages,
        recent_window_tokens=recent_window_tokens,
        sink_window_tokens=sink_window_tokens,
        query_slice=query_slice,
        key_page_sketches=key_page_sketches,
        key_page_minima=key_page_minima,
        key_page_maxima=key_page_maxima,
        relevance_top_k=relevance_top_k,
        relevance_mode=relevance_mode,
    )
    return (
        [key_pages[index] for index in selected_indices],
        [value_pages[index] for index in selected_indices],
    )


@dataclass(slots=True)
class PagedDecodeSession:
    backend: BackendName = "auto"
    cache: PreparedPageCache | None = None
    recent_window_tokens: int | None = None
    sink_window_tokens: int = 0
    relevance_top_k: int = 0
    relevance_sketch_size: int = 1
    relevance_mode: RelevanceMode = "sketch"
    exact_refine_top_k: int = 0
    approximate_old_pages: bool = False
    key_pages: list[PageLike] = field(default_factory=list)
    value_pages: list[PageLike] = field(default_factory=list)
    key_page_sketches: list[np.ndarray] = field(default_factory=list)
    key_page_minima: list[np.ndarray] = field(default_factory=list)
    key_page_maxima: list[np.ndarray] = field(default_factory=list)
    value_page_summaries: list[np.ndarray] = field(default_factory=list)
    last_selected_indices: list[int] = field(default_factory=list)

    def clear(self) -> None:
        self.key_pages.clear()
        self.value_pages.clear()
        self.key_page_sketches.clear()
        self.key_page_minima.clear()
        self.key_page_maxima.clear()
        self.value_page_summaries.clear()
        self.last_selected_indices.clear()
        if self.cache is not None:
            self.cache.clear()

    @property
    def page_count(self) -> int:
        return len(self.key_pages)

    @property
    def active_page_count(self) -> int:
        return len(self.execution_pages()[0])

    @property
    def active_token_count(self) -> int:
        return sum(page.header.token_count for page in self.execution_pages()[0])

    def preload(
        self,
        key_pages: Sequence[PageLike],
        value_pages: Sequence[PageLike],
        *,
        prepare: bool = True,
        trace: ExecutionTrace | None = None,
    ) -> None:
        self.clear()
        self.append(key_pages, value_pages, prepare=prepare, trace=trace)

    def append(
        self,
        key_pages: Sequence[PageLike],
        value_pages: Sequence[PageLike],
        *,
        prepare: bool = True,
        trace: ExecutionTrace | None = None,
    ) -> None:
        if len(key_pages) != len(value_pages):
            raise ValueError("key_pages and value_pages must contain the same number of pages")
        if prepare:
            prepared_key_pages = prepare_pages(key_pages, backend=self.backend, cache=self.cache, trace=trace)
            prepared_value_pages = prepare_pages(value_pages, backend=self.backend, cache=self.cache, trace=trace)
        else:
            prepared_key_pages = list(key_pages)
            prepared_value_pages = list(value_pages)
        self.key_pages.extend(prepared_key_pages)
        self.value_pages.extend(prepared_value_pages)
        self.key_page_sketches.extend(
            sketch_key_page(page, sketch_size=self.relevance_sketch_size) for page in prepared_key_pages
        )
        for page in prepared_key_pages:
            page_min, page_max = envelope_key_page(page)
            self.key_page_minima.append(page_min)
            self.key_page_maxima.append(page_max)
        self.value_page_summaries.extend(summarize_value_page(page) for page in prepared_value_pages)

    def execution_pages(self, query_slice: np.ndarray | None = None) -> tuple[list[PageLike], list[PageLike]]:
        return select_execution_page_pairs(
            self.key_pages,
            self.value_pages,
            recent_window_tokens=self.recent_window_tokens,
            sink_window_tokens=self.sink_window_tokens,
            query_slice=query_slice,
            key_page_sketches=self.key_page_sketches,
            key_page_minima=self.key_page_minima,
            key_page_maxima=self.key_page_maxima,
            relevance_top_k=self.relevance_top_k,
            relevance_mode=self.relevance_mode,
        )

    def execution_indices(
        self,
        query_slice: np.ndarray | None = None,
        *,
        trace: ExecutionTrace | None = None,
    ) -> list[int]:
        return self._execution_plan(query_slice, trace=trace)[0]

    def _execution_plan(
        self,
        query_slice: np.ndarray | None = None,
        *,
        trace: ExecutionTrace | None = None,
    ) -> tuple[list[int], dict[int, np.ndarray]]:
        stage1_indices = select_execution_page_indices(
            self.key_pages,
            recent_window_tokens=self.recent_window_tokens,
            sink_window_tokens=self.sink_window_tokens,
            query_slice=query_slice,
            key_page_sketches=self.key_page_sketches,
            key_page_minima=self.key_page_minima,
            key_page_maxima=self.key_page_maxima,
            relevance_top_k=self.relevance_top_k,
            relevance_mode=self.relevance_mode,
        )
        if query_slice is None or self.exact_refine_top_k <= 0 or self.relevance_top_k <= 0:
            return stage1_indices, {}
        if not stage1_indices:
            return stage1_indices, {}

        base_indices = set(
            select_window_page_indices(
                self.key_pages,
                recent_window_tokens=self.recent_window_tokens,
                sink_window_tokens=self.sink_window_tokens,
            )
        )
        candidate_indices = [index for index in stage1_indices if index not in base_indices]
        if not candidate_indices or self.exact_refine_top_k >= len(candidate_indices):
            return stage1_indices, {}

        candidate_logits = score_pages(
            query_slice,
            [self.key_pages[index] for index in candidate_indices],
            backend=self.backend,
            trace=trace,
        )
        exact_scores = []
        for index, logits in zip(candidate_indices, candidate_logits, strict=True):
            exact_scores.append((float(np.max(logits)), index))
        chosen = [
            index
            for _, index in sorted(
                exact_scores,
                key=lambda item: item[0],
                reverse=True,
            )[: self.exact_refine_top_k]
        ]
        chosen_set = set(chosen)
        chosen_logits = {
            index: np.asarray(logits, dtype=np.float32)
            for index, logits in zip(candidate_indices, candidate_logits, strict=True)
            if index in chosen_set
        }
        return sorted(base_indices.union(chosen)), chosen_logits

    def decode(
        self,
        query_slice: np.ndarray,
        *,
        trace: ExecutionTrace | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.key_pages or not self.value_pages:
            raise ValueError("PagedDecodeSession requires preloaded pages before decode")
        selected_indices, selected_logits = self._execution_plan(query_slice, trace=trace)
        self.last_selected_indices = list(selected_indices)
        key_pages = [self.key_pages[index] for index in selected_indices]
        value_pages = [self.value_pages[index] for index in selected_indices]
        if not self.approximate_old_pages or len(selected_indices) == len(self.key_pages):
            precomputed_page_logits = [selected_logits.get(index) for index in selected_indices]
            return decode_step_with_page_logits(
                query_slice,
                key_pages,
                value_pages,
                page_logits=precomputed_page_logits,
                backend=self.backend,
                trace=trace,
            )
        return self._decode_with_old_page_fallback(query_slice, selected_indices, trace=trace)

    def _decode_with_old_page_fallback(
        self,
        query_slice: np.ndarray,
        selected_indices: Sequence[int],
        *,
        trace: ExecutionTrace | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        query = np.asarray(query_slice, dtype=np.float32)
        exact_index_set = set(selected_indices)
        all_logits: list[np.ndarray] = []

        max_logit = -np.inf
        for index, page in enumerate(self.key_pages):
            if index in exact_index_set:
                logits = score_page(query, page, backend=self.backend, trace=trace).astype(np.float32, copy=False)
                all_logits.append(logits)
                max_logit = max(max_logit, float(np.max(logits)))
                continue
            page_score = score_page_relevance(
                query,
                relevance_mode=self.relevance_mode,
                page_sketch=self.key_page_sketches[index],
                page_min=self.key_page_minima[index],
                page_max=self.key_page_maxima[index],
            )
            logits = np.full(page.header.token_count, page_score, dtype=np.float32)
            all_logits.append(logits)
            max_logit = max(max_logit, page_score)

        if not np.isfinite(max_logit):
            raise ValueError("failed to compute logits for session decode")

        output = np.zeros(self.value_pages[0].header.head_dim, dtype=np.float32)
        all_weights: list[np.ndarray] = []
        denom = 0.0

        for index, page in enumerate(self.key_pages):
            logits = all_logits[index]
            weights = np.exp(logits - max_logit).astype(np.float32, copy=False)
            all_weights.append(weights)
            denom += float(np.sum(weights))
            if index in exact_index_set:
                output = mix_page(
                    weights,
                    self.value_pages[index],
                    out_acc=output,
                    backend=self.backend,
                    trace=trace,
                )
            else:
                output += float(np.sum(weights)) * self.value_page_summaries[index]

        if denom <= 0.0:
            raise ValueError("invalid normalization denominator in session fallback decode")

        logits = np.concatenate(all_logits).astype(np.float32, copy=False)
        weights = np.concatenate(all_weights).astype(np.float32, copy=False) / np.float32(denom)
        return logits, weights, output / np.float32(denom)
