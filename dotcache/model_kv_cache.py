from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np

from .attention_runtime import BackendName, decode_multi_query_step, prepare_pages, score_pages
from .backends import (
    PreparedPageTorch,
    clear_prepared_chunk_cache,
    cuda_available,
    decode_grouped_multiquery_step_prepared_torch_tensor,
    decode_multi_query_step_torch_tensor,
    mps_available,
    prepare_m0_affine_pages_from_tensor_torch,
    prepared_chunk_cache_resident_bytes,
    set_prepared_chunk_cache_budget_override,
)
from .config import DotCacheConfig
from .decode_reference import decode_page
from .encode import encode_page
from .planner import PageModeSpec, choose_page_mode, observe_page, parse_page_mode_token
from .page_cache import PreparedPageCache
from .packing import words_per_group
from .selector_baselines import (
    CandidateSafeRouterModel,
    CandidateTargetRouterModel,
    LinearSelectorModel,
    adjust_linear_selector_model_logits,
    load_page_selector_artifact,
)
from .session_runtime import PagedDecodeSession, score_page_relevance, select_execution_page_indices, select_window_page_indices
from .tracing import ExecutionTrace
from .types import EncodedPage, PageHeader
from .modes.m2_key_sketch import segment_ids_for_token_count
from .modes.m4_key_project import fit_shared_project_basis

PageLike = EncodedPage | PreparedPageTorch


_DECODE_STAGE_TIMING_STAGES = (
    "prepare_pages_with_tail",
    "prepare_layout_build",
    "m2_prefilter",
    "query_export",
    "shortlist_selection",
    "shortlist_base_window",
    "shortlist_candidate_scoring",
    "shortlist_candidate_approx_scoring",
    "shortlist_candidate_ranking",
    "shortlist_candidate_secondary_scoring",
    "shortlist_candidate_neighbor_rescue",
    "shortlist_candidate_builtin_selection",
    "shortlist_candidate_builtin_candidate_index_build",
    "shortlist_candidate_builtin_sidecar_stack",
    "shortlist_candidate_builtin_score_compute",
    "shortlist_candidate_builtin_ranking",
    "shortlist_exact_selection",
    "shortlist_union_rescue",
    "shortlist_materialization",
    "grouping_validation",
    "chunk_budget_sync",
    "backend_call_wall",
    "backend_call_non_backend",
)


def _empty_decode_stage_timing_totals() -> dict[str, float]:
    return {stage: 0.0 for stage in _DECODE_STAGE_TIMING_STAGES}


def _decode_stage_summary_key(stage: str) -> str:
    return f"execution_decode_{stage}_ms_total"


def _backend_trace_ms_total(trace: ExecutionTrace | None) -> float:
    if trace is None:
        return 0.0
    return float(
        trace.prepare_ms_total
        + trace.score_ms_total
        + trace.mix_ms_total
        + trace.softmax_ms_total
        + trace.unpack_ms_total
        + trace.fwht_ms_total
        + trace.chunk_assembly_ms_total
    )


def default_q_head_to_kv_head(num_attention_heads: int, num_key_value_heads: int) -> np.ndarray:
    if num_attention_heads <= 0:
        raise ValueError("num_attention_heads must be positive")
    if num_key_value_heads <= 0:
        raise ValueError("num_key_value_heads must be positive")
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads for the Llama path")
    return (np.arange(num_attention_heads, dtype=np.int64) // (num_attention_heads // num_key_value_heads)).astype(
        np.int64,
        copy=False,
    )


def _group_query_heads(mapping: np.ndarray, *, num_key_value_heads: int) -> tuple[tuple[int, ...], ...]:
    grouped: list[list[int]] = [[] for _ in range(num_key_value_heads)]
    for q_head_id, kv_head_id in enumerate(mapping.tolist()):
        if kv_head_id < 0 or kv_head_id >= num_key_value_heads:
            raise ValueError("q_head_to_kv_head contains an invalid KV head id")
        grouped[kv_head_id].append(q_head_id)
    return tuple(tuple(group) for group in grouped)


def _page_header(page: PageLike) -> PageHeader:
    return page.header if not isinstance(page, PreparedPageTorch) else page.header


def _page_token_range(page: PageLike) -> dict[str, int]:
    header = _page_header(page)
    return {
        "token_start": int(header.token_start),
        "token_end": int(header.token_start + header.token_count),
        "token_count": int(header.token_count),
    }


def _page_age_bucket(page: PageLike, *, context_length: int) -> str:
    header = _page_header(page)
    if context_length <= 0:
        return "recent"
    age_fraction = max(0.0, min(1.0, 1.0 - (float(header.token_start + header.token_count) / float(context_length))))
    if age_fraction < 0.25:
        return "recent"
    if age_fraction < 0.75:
        return "middle"
    return "old"


def _recent_old_bonus_weight(
    page: PageLike,
    *,
    recent_start: int,
    bonus_window: int,
) -> float:
    if bonus_window <= 0:
        return 0.0
    header = _page_header(page)
    page_end = int(header.token_start + header.token_count)
    if page_end > recent_start:
        return 0.0
    distance = max(0, int(recent_start) - page_end)
    if distance >= int(bonus_window):
        return 0.0
    return float(1.0 - (float(distance) / float(max(int(bonus_window), 1))))


def _rank_correlation(lhs: Sequence[float], rhs: Sequence[float]) -> float | None:
    if len(lhs) != len(rhs):
        raise ValueError("rank correlation inputs must have matching lengths")
    if len(lhs) < 2:
        return None
    lhs_array = np.asarray(lhs, dtype=np.float32)
    rhs_array = np.asarray(rhs, dtype=np.float32)
    lhs_std = float(np.std(lhs_array))
    rhs_std = float(np.std(rhs_array))
    if lhs_std <= 0.0 or rhs_std <= 0.0:
        return None
    return float(np.corrcoef(lhs_array, rhs_array)[0, 1])


def _score_page_relevance_for_mode(
    query_slice: np.ndarray,
    page: PageLike,
    *,
    relevance_mode: str,
) -> float | None:
    source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
    if relevance_mode == "sketch" and source_page.runtime_page_sketch is None:
        return None
    if relevance_mode == "envelope" and (source_page.runtime_page_min is None or source_page.runtime_page_max is None):
        return None
    return float(
        score_page_relevance(
            np.asarray(query_slice, dtype=np.float32),
            relevance_mode=relevance_mode,
            page_sketch=None
            if source_page.runtime_page_sketch is None
            else np.asarray(source_page.runtime_page_sketch, dtype=np.float32),
            page_min=None if source_page.runtime_page_min is None else np.asarray(source_page.runtime_page_min, dtype=np.float32),
            page_max=None if source_page.runtime_page_max is None else np.asarray(source_page.runtime_page_max, dtype=np.float32),
        )
    )


def _page_has_m2_sidecar(page: PageLike) -> bool:
    if isinstance(page, PreparedPageTorch):
        return page.m2_sketch is not None and page.m2_basis is not None and page.m2_mean is not None
    return page.m2_sketch is not None and page.m2_basis is not None and page.m2_mean is not None


def _page_m2_prefilter_score_numpy(queries: np.ndarray, page: PageLike) -> float:
    source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
    if source_page.m2_sketch is None or source_page.m2_basis is None or source_page.m2_mean is None:
        raise ValueError("page is missing M2 sidecar payload")
    query_groups = queries.reshape(queries.shape[0], source_page.header.num_groups, source_page.header.group_size)
    logits = np.zeros((queries.shape[0], source_page.header.token_count), dtype=np.float32)
    for group_index in range(source_page.header.num_groups):
        group_basis = source_page.m2_basis[group_index].astype(np.float32)
        group_mean = source_page.m2_mean[group_index].astype(np.float32)
        if group_basis.ndim == 2:
            q_proj = query_groups[:, group_index, :] @ group_basis.T
            logits += np.einsum("tr,qr->qt", source_page.m2_sketch[:, group_index, :].astype(np.float32), q_proj)
            logits += np.einsum("g,qg->q", group_mean, query_groups[:, group_index, :])[:, None]
            continue
        segment_ids = segment_ids_for_token_count(source_page.header.token_count, int(group_basis.shape[0]))
        q_proj = np.einsum("srg,qg->qsr", group_basis, query_groups[:, group_index, :])
        logits += np.einsum(
            "tr,qtr->qt",
            source_page.m2_sketch[:, group_index, :].astype(np.float32),
            q_proj[:, segment_ids, :],
        )
        logits += np.einsum("tg,qg->qt", group_mean[segment_ids], query_groups[:, group_index, :])
    return float(np.max(logits))


def _page_m2_prefilter_score_torch(queries, page: PageLike) -> float:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for torch-side M2 prefiltering") from exc
    if not torch.is_tensor(queries):
        raise TypeError("queries must be a torch.Tensor")
    prepared = page if isinstance(page, PreparedPageTorch) else None
    if prepared is None or prepared.m2_sketch is None or prepared.m2_basis is None or prepared.m2_mean is None:
        return _page_m2_prefilter_score_numpy(queries.detach().cpu().numpy().astype(np.float32, copy=False), page)
    query_groups = queries.reshape(int(queries.shape[0]), prepared.header.num_groups, prepared.header.group_size)
    logits = torch.zeros((int(queries.shape[0]), prepared.header.token_count), dtype=torch.float32, device=queries.device)
    for group_index in range(prepared.header.num_groups):
        group_basis = prepared.m2_basis[group_index]
        group_mean = prepared.m2_mean[group_index]
        group_sketch = prepared.m2_sketch[:, group_index, :]
        work_dtype = torch.promote_types(query_groups.dtype, group_basis.dtype)
        work_dtype = torch.promote_types(work_dtype, group_sketch.dtype)
        work_dtype = torch.promote_types(work_dtype, group_mean.dtype)
        qg = query_groups[:, group_index, :].to(dtype=work_dtype)
        group_basis = group_basis.to(dtype=work_dtype)
        group_mean = group_mean.to(dtype=work_dtype)
        group_sketch = group_sketch.to(dtype=work_dtype)
        if group_basis.dim() == 2:
            q_proj = torch.einsum("qg,rg->qr", qg, group_basis)
            logits += torch.einsum("tr,qr->qt", group_sketch, q_proj)
            logits += torch.einsum("g,qg->q", group_mean, qg)[:, None]
            continue
        segment_ids = torch.from_numpy(
            segment_ids_for_token_count(prepared.header.token_count, int(group_basis.shape[0]))
        ).to(device=queries.device)
        q_proj = torch.einsum("srg,qg->qsr", group_basis, qg)
        logits += torch.einsum("tr,qtr->qt", group_sketch, q_proj[:, segment_ids, :])
        logits += torch.einsum("tg,qg->qt", group_mean[segment_ids], qg)
    return float(torch.max(logits).item())


def _pages_can_batch_m2_prefilter(pages: Sequence[PageLike]) -> bool:
    if not pages:
        return False
    first = pages[0]
    first_source = first.source_page if isinstance(first, PreparedPageTorch) else first
    if first_source.m2_sketch is None or first_source.m2_basis is None or first_source.m2_mean is None:
        return False
    token_count = int(first_source.header.token_count)
    num_groups = int(first_source.header.num_groups)
    group_size = int(first_source.header.group_size)
    sketch_dim = int(first_source.m2_sketch.shape[-1])
    segment_count = int(first_source.m2_basis.shape[1]) if first_source.m2_basis.ndim == 4 else 1
    prepared_device = first.device_type if isinstance(first, PreparedPageTorch) else None
    for page in pages[1:]:
        source = page.source_page if isinstance(page, PreparedPageTorch) else page
        if source.m2_sketch is None or source.m2_basis is None or source.m2_mean is None:
            return False
        if int(source.header.token_count) != token_count:
            return False
        if int(source.header.num_groups) != num_groups or int(source.header.group_size) != group_size:
            return False
        if int(source.m2_sketch.shape[-1]) != sketch_dim:
            return False
        if (int(source.m2_basis.shape[1]) if source.m2_basis.ndim == 4 else 1) != segment_count:
            return False
        if prepared_device is not None:
            if not isinstance(page, PreparedPageTorch) or page.device_type != prepared_device:
                return False
    return True


def _page_m2_prefilter_scores_numpy(queries: np.ndarray, pages: Sequence[PageLike]) -> np.ndarray:
    source_pages = [page.source_page if isinstance(page, PreparedPageTorch) else page for page in pages]
    first = source_pages[0]
    query_groups = queries.reshape(queries.shape[0], first.header.num_groups, first.header.group_size)
    sketch = np.stack([page.m2_sketch for page in source_pages], axis=0).astype(np.float32, copy=False)
    basis = np.stack([page.m2_basis for page in source_pages], axis=0).astype(np.float32, copy=False)
    mean = np.stack([page.m2_mean for page in source_pages], axis=0).astype(np.float32, copy=False)
    if basis.ndim == 4:
        q_proj = np.einsum("pgrd,qgd->qpgr", basis, query_groups)
        logits = np.einsum("ptgr,qpgr->qpt", sketch, q_proj)
        logits += np.einsum("pgd,qgd->qp", mean, query_groups)[:, :, None]
        return np.max(logits, axis=(0, 2)).astype(np.float32, copy=False)
    segment_ids = segment_ids_for_token_count(first.header.token_count, int(basis.shape[2]))
    logits = np.zeros((queries.shape[0], len(source_pages), first.header.token_count), dtype=np.float32)
    for group_index in range(first.header.num_groups):
        q_proj = np.einsum("psrd,qd->qpsr", basis[:, group_index], query_groups[:, group_index, :])
        logits += np.einsum("ptr,qptr->qpt", sketch[:, :, group_index, :], q_proj[:, :, segment_ids, :])
        logits += np.einsum("ptg,qg->qpt", mean[:, group_index, segment_ids, :], query_groups[:, group_index, :])
    return np.max(logits, axis=(0, 2)).astype(np.float32, copy=False)


def _page_m2_prefilter_scores_torch(queries, pages: Sequence[PageLike]) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for torch-side M2 prefiltering") from exc
    if not torch.is_tensor(queries):
        raise TypeError("queries must be a torch.Tensor")
    if not all(isinstance(page, PreparedPageTorch) for page in pages):
        return _page_m2_prefilter_scores_numpy(queries.detach().cpu().numpy().astype(np.float32, copy=False), pages)
    first = pages[0]
    query_groups = queries.reshape(int(queries.shape[0]), first.header.num_groups, first.header.group_size)
    sketch = torch.stack([page.m2_sketch for page in pages], dim=0)
    basis = torch.stack([page.m2_basis for page in pages], dim=0)
    mean = torch.stack([page.m2_mean for page in pages], dim=0)
    if basis.dim() == 4:
        q_proj = torch.einsum("pgrd,qgd->qpgr", basis, query_groups)
        logits = torch.einsum("ptgr,qpgr->qpt", sketch, q_proj)
        logits += torch.einsum("pgd,qgd->qp", mean, query_groups)[:, :, None]
        return torch.amax(logits, dim=(0, 2)).detach().cpu().numpy().astype(np.float32, copy=False)
    segment_ids = torch.from_numpy(segment_ids_for_token_count(first.header.token_count, int(basis.shape[2]))).to(device=queries.device)
    logits = torch.zeros((int(queries.shape[0]), len(pages), first.header.token_count), dtype=torch.float32, device=queries.device)
    for group_index in range(first.header.num_groups):
        group_basis = basis[:, group_index]
        group_sketch = sketch[:, :, group_index, :]
        group_mean = mean[:, group_index, segment_ids, :]
        work_dtype = torch.promote_types(query_groups.dtype, group_basis.dtype)
        work_dtype = torch.promote_types(work_dtype, group_sketch.dtype)
        work_dtype = torch.promote_types(work_dtype, group_mean.dtype)
        qg = query_groups[:, group_index, :].to(dtype=work_dtype)
        group_basis = group_basis.to(dtype=work_dtype)
        group_sketch = group_sketch.to(dtype=work_dtype)
        group_mean = group_mean.to(dtype=work_dtype)
        q_proj = torch.einsum("psrd,qg->qpsr", group_basis, qg)
        logits += torch.einsum("ptr,qptr->qpt", group_sketch, q_proj[:, :, segment_ids, :])
        logits += torch.einsum("ptg,qg->qpt", group_mean, qg)
    return torch.amax(logits, dim=(0, 2)).detach().cpu().numpy().astype(np.float32, copy=False)


def _grouped_pages_can_batch(
    key_pages_by_group: Sequence[Sequence[PageLike]],
    value_pages_by_group: Sequence[Sequence[PageLike]],
    query_groups: Sequence[Any],
) -> bool:
    return _grouped_pages_batch_rejection_reason(key_pages_by_group, value_pages_by_group, query_groups) is None


def _grouped_pages_batch_rejection_reason(
    key_pages_by_group: Sequence[Sequence[PageLike]],
    value_pages_by_group: Sequence[Sequence[PageLike]],
    query_groups: Sequence[Any],
) -> str | None:
    def _page_batch_signature(page: PreparedPageTorch) -> tuple[int | str, ...]:
        sketch = page.m2_sketch
        basis = page.m2_basis
        sketch_dim = int(sketch.shape[-1]) if sketch is not None else 0
        segment_count = int(basis.shape[1]) if basis is not None and int(basis.dim()) == 4 else 1
        centered = int(page.m2_mean is not None)
        header = page.header
        return (
            page.device_type,
            header.kind,
            header.mode_default,
            header.escape_dtype if header.mode_default == "M3" else "",
            header.token_count,
            header.head_dim,
            header.padded_head_dim,
            header.group_size,
            header.num_groups,
            header.bits,
            header.words_per_group,
            header.layout,
            header.quant_scheme,
            sketch_dim,
            segment_count,
            centered,
        )

    if not key_pages_by_group:
        return "no_key_groups"
    if len(key_pages_by_group) != len(value_pages_by_group):
        return "group_count_mismatch"
    group_count = len(key_pages_by_group)
    if len(query_groups) != group_count:
        return "query_group_count_mismatch"
    try:
        query_count = int(query_groups[0].shape[0])
    except Exception:
        return "query_shape_invalid"
    page_count = len(key_pages_by_group[0])
    if page_count == 0:
        return "page_count_zero"
    for group_index in range(group_count):
        if len(key_pages_by_group[group_index]) != page_count or len(value_pages_by_group[group_index]) != page_count:
            return "page_count_mismatch"
        if int(query_groups[group_index].shape[0]) != query_count:
            return "query_count_mismatch"
        if not all(isinstance(page, PreparedPageTorch) for page in key_pages_by_group[group_index]):
            return "key_page_not_prepared"
        if not all(isinstance(page, PreparedPageTorch) for page in value_pages_by_group[group_index]):
            return "value_page_not_prepared"
        if any(page.device_type != key_pages_by_group[0][0].device_type for page in key_pages_by_group[group_index]):
            return "key_device_mismatch"
        if any(page.device_type != value_pages_by_group[0][0].device_type for page in value_pages_by_group[group_index]):
            return "value_device_mismatch"
    return None


@dataclass(slots=True)
class _PreparedDecodeViewLayout:
    grouped_batch_signature: tuple[tuple[tuple[Any, ...], tuple[Any, ...]], ...]
    key_chunk_lengths: tuple[int, ...]
    value_chunk_lengths: tuple[int, ...]


@dataclass(slots=True)
class _ExecutionBuiltinSelectorCache:
    page_signature: tuple[tuple[int, int, int], ...] = ()
    sketch_matrix: np.ndarray | None = None
    minima_matrix: np.ndarray | None = None
    maxima_matrix: np.ndarray | None = None

    def resident_bytes(self) -> int:
        total = 0
        if self.sketch_matrix is not None:
            total += int(self.sketch_matrix.nbytes)
        if self.minima_matrix is not None:
            total += int(self.minima_matrix.nbytes)
        if self.maxima_matrix is not None:
            total += int(self.maxima_matrix.nbytes)
        return int(total)


def _prepared_page_group_signature(page: PreparedPageTorch) -> tuple[Any, ...]:
    basis = getattr(page, "m2_basis", None)
    if basis is None:
        segment_count = 0
    else:
        ndim = int(basis.ndim) if hasattr(basis, "ndim") else int(basis.dim())
        segment_count = int(basis.shape[1]) if ndim == 4 else 1
    return (
        page.device_type,
        page.header.mode_default,
        page.header.token_count,
        page.header.head_dim,
        page.header.padded_head_dim,
        page.header.group_size,
        page.header.num_groups,
        page.header.bits,
        page.header.words_per_group,
        page.header.layout,
        page.header.quant_scheme,
        int(page.m2_sketch.shape[-1]) if page.m2_sketch is not None else 0,
        segment_count,
    )


def _prepared_page_chunk_lengths(pages: Sequence[PreparedPageTorch]) -> tuple[int, ...]:
    if not pages:
        return ()
    lengths: list[int] = []
    current_signature: tuple[Any, ...] | None = None
    current_length = 0
    for page in pages:
        signature = _prepared_page_group_signature(page)
        if current_signature is None or signature == current_signature:
            current_signature = signature
            current_length += 1
            continue
        lengths.append(current_length)
        current_signature = signature
        current_length = 1
    if current_length > 0:
        lengths.append(current_length)
    return tuple(lengths)


def _prepared_page_aligned_chunk_lengths(
    key_pages: Sequence[PreparedPageTorch],
    value_pages: Sequence[PreparedPageTorch],
) -> tuple[int, ...]:
    if len(key_pages) != len(value_pages):
        return ()
    if not key_pages:
        return ()
    lengths: list[int] = []
    current_length = 0
    current_key_signature: tuple[Any, ...] | None = None
    current_value_signature: tuple[Any, ...] | None = None
    for key_page, value_page in zip(key_pages, value_pages, strict=True):
        key_signature = _prepared_page_group_signature(key_page)
        value_signature = _prepared_page_group_signature(value_page)
        if (
            current_length > 0
            and (key_signature != current_key_signature or value_signature != current_value_signature)
        ):
            lengths.append(current_length)
            current_length = 0
        if current_length == 0:
            current_key_signature = key_signature
            current_value_signature = value_signature
        current_length += 1
    if current_length > 0:
        lengths.append(current_length)
    return tuple(lengths)


def _build_prepared_decode_view_layout(
    key_pages: Sequence[PageLike],
    value_pages: Sequence[PageLike],
) -> _PreparedDecodeViewLayout | None:
    if len(key_pages) != len(value_pages) or not key_pages:
        return None
    if not all(isinstance(page, PreparedPageTorch) for page in key_pages):
        return None
    if not all(isinstance(page, PreparedPageTorch) for page in value_pages):
        return None
    prepared_key_pages = tuple(key_pages)
    prepared_value_pages = tuple(value_pages)
    return _PreparedDecodeViewLayout(
        grouped_batch_signature=tuple(
            (_prepared_page_group_signature(key_page), _prepared_page_group_signature(value_page))
            for key_page, value_page in zip(prepared_key_pages, prepared_value_pages, strict=True)
        ),
        key_chunk_lengths=_prepared_page_chunk_lengths(prepared_key_pages),
        value_chunk_lengths=_prepared_page_chunk_lengths(prepared_value_pages),
    )


def _grouped_layouts_can_batch(
    layouts: Sequence[_PreparedDecodeViewLayout | None],
    query_groups: Sequence[Any],
) -> bool:
    return _grouped_layout_batch_rejection_reason(layouts, query_groups) is None


def _grouped_layout_batch_rejection_reason(
    layouts: Sequence[_PreparedDecodeViewLayout | None],
    query_groups: Sequence[Any],
) -> str | None:
    if not layouts or any(layout is None for layout in layouts):
        return "layout_missing"
    try:
        query_count = int(query_groups[0].shape[0])
    except Exception:
        return "query_shape_invalid"
    first_layout = layouts[0]
    assert first_layout is not None
    for group_index in range(1, len(layouts)):
        layout = layouts[group_index]
        assert layout is not None
        if layout.grouped_batch_signature != first_layout.grouped_batch_signature:
            return "layout_signature_mismatch"
        if int(query_groups[group_index].shape[0]) != query_count:
            return "query_count_mismatch"
    return None


def _normalize_prefill_tensor(
    values: np.ndarray,
    *,
    num_key_value_heads: int,
    head_dim: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"{name} batch dimension must be 1 for the Phase 5 Llama path")
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [kv_heads, seq_len, head_dim] or [1, kv_heads, seq_len, head_dim]")
    if array.shape[0] != num_key_value_heads:
        raise ValueError(f"{name} must contain {num_key_value_heads} KV heads")
    if array.shape[2] != head_dim:
        raise ValueError(f"{name} head_dim must equal {head_dim}")
    return array


def _normalize_step_tensor(
    values: np.ndarray,
    *,
    num_key_value_heads: int,
    head_dim: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"{name} batch dimension must be 1 for the Phase 5 Llama path")
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [kv_heads, token_count, head_dim]")
    if array.shape[0] != num_key_value_heads:
        raise ValueError(f"{name} must contain {num_key_value_heads} KV heads")
    if array.shape[2] != head_dim:
        raise ValueError(f"{name} head_dim must equal {head_dim}")
    return array


def _normalize_query_step(query_step: np.ndarray, *, num_attention_heads: int, head_dim: int) -> np.ndarray:
    queries = np.asarray(query_step, dtype=np.float32)
    if queries.ndim == 4:
        if queries.shape[0] != 1 or queries.shape[2] != 1:
            raise ValueError("query_step must have shape [q_heads, head_dim] or [1, q_heads, 1, head_dim]")
        queries = queries[0, :, 0, :]
    if queries.ndim != 2:
        raise ValueError("query_step must have shape [q_heads, head_dim]")
    if queries.shape[0] != num_attention_heads:
        raise ValueError(f"query_step must contain {num_attention_heads} query heads")
    if queries.shape[1] != head_dim:
        raise ValueError(f"query_step head_dim must equal {head_dim}")
    return queries


def _normalize_prefill_tensor_torch(values, *, num_key_value_heads: int, head_dim: int, name: str):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for torch-native prefill ingest") from exc
    if not torch.is_tensor(values):
        raise TypeError(f"{name} must be a torch.Tensor")
    array = values.detach().to(dtype=torch.float32)
    if array.ndim == 4:
        if int(array.shape[0]) != 1:
            raise ValueError(f"{name} batch dimension must be 1 for the Phase 5 Llama path")
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [kv_heads, seq_len, head_dim] or [1, kv_heads, seq_len, head_dim]")
    if int(array.shape[0]) != num_key_value_heads:
        raise ValueError(f"{name} must contain {num_key_value_heads} KV heads")
    if int(array.shape[2]) != head_dim:
        raise ValueError(f"{name} head_dim must equal {head_dim}")
    return array


@dataclass(slots=True)
class _TailPageBuilder:
    config: DotCacheConfig
    layer_id: int
    kv_head_id: int
    select_page_mode: Callable[..., PageModeSpec | None] | None = None
    token_start: int | None = None
    key_rows: list[np.ndarray] = field(default_factory=list)
    value_rows: list[np.ndarray] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        return len(self.key_rows)

    def clear(self) -> None:
        self.token_start = None
        self.key_rows.clear()
        self.value_rows.clear()

    def _should_build_execution_runtime_metadata(self, *, kind: str) -> bool:
        if kind != "K":
            return False
        return self.config.execution_shortlist_enabled()

    def load_prefill_remainder(
        self,
        key_rows: np.ndarray,
        value_rows: np.ndarray,
        *,
        token_start: int,
    ) -> None:
        self.clear()
        if key_rows.shape[0] != value_rows.shape[0]:
            raise ValueError("prefill remainder key/value rows must align")
        if key_rows.shape[0] == 0:
            return
        self.token_start = int(token_start)
        self.key_rows.extend(np.asarray(key_rows, dtype=np.float32))
        self.value_rows.extend(np.asarray(value_rows, dtype=np.float32))

    def append_step_rows(
        self,
        key_rows: np.ndarray,
        value_rows: np.ndarray,
        *,
        token_start: int,
        sequence_length: int | None = None,
    ) -> tuple[list[EncodedPage], list[EncodedPage]]:
        if key_rows.shape != value_rows.shape:
            raise ValueError("step key/value rows must align")
        if key_rows.ndim != 2:
            raise ValueError("step rows must have shape [token_count, head_dim]")
        if key_rows.shape[0] == 0:
            return [], []

        finalized_key_pages: list[EncodedPage] = []
        finalized_value_pages: list[EncodedPage] = []
        expected_token = self.next_token_index
        if expected_token is not None and token_start != expected_token:
            raise ValueError(f"tail-page append expected token_index {expected_token}, received {token_start}")
        if self.token_start is None:
            self.token_start = int(token_start)

        for offset in range(key_rows.shape[0]):
            self.key_rows.append(np.asarray(key_rows[offset], dtype=np.float32))
            self.value_rows.append(np.asarray(value_rows[offset], dtype=np.float32))
            if len(self.key_rows) < self.config.tokens_per_page:
                continue

            if self.token_start is None:
                raise RuntimeError("tail-page token_start is missing while finalizing a page")
            dense_keys = np.stack(self.key_rows, axis=0).astype(np.float32, copy=False)
            dense_values = np.stack(self.value_rows, axis=0).astype(np.float32, copy=False)
            current_sequence_length = int(sequence_length if sequence_length is not None else (token_start + key_rows.shape[0]))
            key_page_mode = self.select_page_mode(
                dense_keys,
                kind="K",
                layer_id=self.layer_id,
                kv_head_id=self.kv_head_id,
                token_start=self.token_start,
                sequence_length=current_sequence_length,
                stage="decode",
            )
            key_mode = None if key_page_mode is not None else self.config.resolve_page_mode(kind="K", layer_id=self.layer_id, kv_head_id=self.kv_head_id)
            value_page_mode = self.select_page_mode(
                dense_values,
                kind="V",
                layer_id=self.layer_id,
                kv_head_id=self.kv_head_id,
                token_start=self.token_start,
                sequence_length=current_sequence_length,
                stage="decode",
            )
            value_mode = None if value_page_mode is not None else self.config.resolve_page_mode(kind="V", layer_id=self.layer_id, kv_head_id=self.kv_head_id)
            finalized_key_pages.append(
                encode_page(
                    dense_keys,
                    self.config,
                    kind="K",
                    layer_id=self.layer_id,
                    kv_head_id=self.kv_head_id,
                    token_start=self.token_start,
                    mode=key_mode,
                    page_mode=key_page_mode,
                )
            )
            finalized_value_pages.append(
                encode_page(
                    dense_values,
                    self.config,
                    kind="V",
                    layer_id=self.layer_id,
                    kv_head_id=self.kv_head_id,
                    token_start=self.token_start,
                    mode=value_mode,
                    page_mode=value_page_mode,
                )
            )
            self.key_rows.clear()
            self.value_rows.clear()
            self.token_start += self.config.tokens_per_page
        if self.token_count == 0:
            self.token_start = None
        return finalized_key_pages, finalized_value_pages

    @property
    def next_token_index(self) -> int | None:
        if self.token_start is None:
            return None
        return self.token_start + self.token_count

    def build_temp_pages(self) -> tuple[EncodedPage, EncodedPage] | None:
        if self.token_count == 0:
            return None
        if self.token_start is None:
            raise RuntimeError("tail-page token_start is missing")
        dense_keys = np.stack(self.key_rows, axis=0).astype(np.float32, copy=False)
        dense_values = np.stack(self.value_rows, axis=0).astype(np.float32, copy=False)
        return (
            encode_page(
                dense_keys,
                self.config,
                kind="K",
                layer_id=self.layer_id,
                kv_head_id=self.kv_head_id,
                token_start=self.token_start,
                mode="M3",
                build_runtime_metadata=self._should_build_execution_runtime_metadata(kind="K"),
            ),
            encode_page(
                dense_values,
                self.config,
                kind="V",
                layer_id=self.layer_id,
                kv_head_id=self.kv_head_id,
                token_start=self.token_start,
                mode="M3",
                build_runtime_metadata=False,
            ),
        )


def _tail_escape_dtype_numpy(dtype_name: str) -> np.dtype:
    if dtype_name == "float16":
        return np.float16
    if dtype_name == "float32":
        return np.float32
    if dtype_name == "int8":
        return np.int8
    raise ValueError(f"unsupported tail escape dtype: {dtype_name}")


def _quantize_tail_rows_numpy(rows: np.ndarray, dtype_name: str) -> tuple[np.ndarray, np.ndarray | None]:
    values = np.asarray(rows, dtype=np.float32)
    if dtype_name in {"float16", "float32"}:
        return values.astype(_tail_escape_dtype_numpy(dtype_name), copy=False), None
    if dtype_name == "int8":
        row_absmax = np.max(np.abs(values), axis=1)
        scales = np.maximum(row_absmax / 127.0, 1e-8).astype(np.float16, copy=False)
        quantized = np.clip(np.rint(values / scales[:, None]), -127.0, 127.0).astype(np.int8, copy=False)
        return quantized, scales
    raise ValueError(f"unsupported tail escape dtype: {dtype_name}")


@dataclass(slots=True)
class _PersistentTailPage:
    config: DotCacheConfig
    layer_id: int
    kv_head_id: int
    kind: str
    device_type: str
    source_page: EncodedPage | None = None
    prepared_page: PreparedPageTorch | None = None
    host_buffer: np.ndarray | None = None
    host_scales: np.ndarray | None = None
    token_count: int = 0
    resident_nbytes: int = 0

    def clear(self) -> None:
        self.token_count = 0
        if self.source_page is not None:
            self.source_page.header.token_count = 0
            self.source_page.escape_payload = None if self.host_buffer is None else self.host_buffer[:0]
            self.source_page.escape_scales = None if self.host_scales is None else self.host_scales[:0]
            self.source_page.runtime_page_mean = None
            self.source_page.runtime_page_sketch = None
            self.source_page.runtime_page_min = None
            self.source_page.runtime_page_max = None

    def _should_build_execution_runtime_metadata(self) -> bool:
        if self.kind != "K":
            return False
        return self.config.execution_shortlist_enabled()

    def _refresh_runtime_metadata(self) -> None:
        if self.source_page is None or not self._should_build_execution_runtime_metadata():
            return
        if self.token_count <= 0:
            self.source_page.runtime_page_mean = None
            self.source_page.runtime_page_sketch = None
            self.source_page.runtime_page_min = None
            self.source_page.runtime_page_max = None
            return
        dense = self.materialize_rows()
        self.source_page.runtime_page_mean = dense.mean(axis=0).astype(np.float32, copy=False)
        self.source_page.runtime_page_sketch = self.source_page.runtime_page_mean[None, :]
        self.source_page.runtime_page_min = dense.min(axis=0).astype(np.float32, copy=False)
        self.source_page.runtime_page_max = dense.max(axis=0).astype(np.float32, copy=False)

    def _ensure_allocated(self, *, token_start: int) -> bool:
        if self.source_page is not None and self.prepared_page is not None and self.host_buffer is not None:
            self.source_page.header.token_start = int(token_start)
            self.prepared_page.header.token_start = int(token_start)
            return False

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is required only for the MPS tail path
            raise RuntimeError("torch is required for the persistent torch tail path") from exc

        dtype_name = self.config.escape_dtype
        np_dtype = _tail_escape_dtype_numpy(dtype_name)
        torch_dtype = getattr(torch, dtype_name)
        host_buffer = np.zeros((self.config.tokens_per_page, self.config.head_dim), dtype=np_dtype)
        host_scales = None if dtype_name != "int8" else np.zeros((self.config.tokens_per_page,), dtype=np.float16)
        header = PageHeader(
            layer_id=self.layer_id,
            kv_head_id=self.kv_head_id,
            kind=self.kind,
            token_start=int(token_start),
            token_count=0,
            head_dim=self.config.head_dim,
            padded_head_dim=self.config.padded_head_dim,
            group_size=self.config.group_size,
            num_groups=self.config.num_groups,
            bits=self.config.bits_k if self.kind == "K" else self.config.bits_v,
            words_per_group=words_per_group(self.config.group_size, self.config.bits_k if self.kind == "K" else self.config.bits_v),
            mode_default="M3",
            layout=self.config.payload_layout_k if self.kind == "K" else self.config.payload_layout_v,
            quant_scheme=self.config.quant_scheme_k if self.kind == "K" else self.config.quant_scheme_v,
            escape_dtype=dtype_name,
        )
        source_page = EncodedPage(
            header=header,
            escape_payload=host_buffer[:0],
            escape_scales=None if host_scales is None else host_scales[:0],
        )
        # These buffers are mutated incrementally during serving. If they are
        # created under torch.inference_mode(), later appends fail with
        # "Inplace update to inference tensor outside InferenceMode".
        with torch.inference_mode(False):
            device_payload = torch.zeros(
                (self.config.tokens_per_page, self.config.head_dim),
                dtype=torch_dtype,
                device=self.device_type,
            )
            device_scales = None
            if dtype_name == "int8":
                scale_dtype = torch.float32 if self.device_type == "mps" else torch.float16
                device_scales = torch.zeros((self.config.tokens_per_page,), dtype=scale_dtype, device=self.device_type)
        prepared_page = PreparedPageTorch(
            device_type=self.device_type,
            source_page=source_page,
            header=header,
            escape_payload=device_payload,
            escape_scales=device_scales,
            host_to_device_nbytes=int(device_payload.numel() * device_payload.element_size())
            + (0 if device_scales is None else int(device_scales.numel() * device_scales.element_size())),
        )
        self.source_page = source_page
        self.prepared_page = prepared_page
        self.host_buffer = host_buffer
        self.host_scales = host_scales
        self.resident_nbytes = int(device_payload.numel() * device_payload.element_size()) + (
            0 if device_scales is None else int(device_scales.numel() * device_scales.element_size())
        )
        return True

    def load_rows(
        self,
        rows: np.ndarray,
        *,
        token_start: int,
        trace: ExecutionTrace | None = None,
    ) -> None:
        values = np.asarray(rows, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.config.head_dim:
            raise ValueError("tail rows must have shape [token_count, head_dim]")
        self.clear()
        if values.shape[0] == 0:
            return
        self._ensure_allocated(token_start=token_start)
        self.append_rows(values, token_start=token_start, trace=trace)

    def prepare_append_span(self, *, token_start: int, row_count: int) -> tuple[int, int]:
        if row_count < 0:
            raise ValueError("row_count must be non-negative")
        self._ensure_allocated(token_start=token_start if self.token_count == 0 else self.source_page.header.token_start)
        if self.source_page is None or self.prepared_page is None or self.host_buffer is None:
            raise RuntimeError("persistent tail page is not initialized")
        expected_token = self.source_page.header.token_start + self.token_count
        if token_start != expected_token:
            raise ValueError(f"persistent tail expected token_start {expected_token}, received {token_start}")
        end = self.token_count + row_count
        if end > self.config.tokens_per_page:
            raise ValueError("persistent tail cannot exceed tokens_per_page")
        start = self.token_count
        self.source_page.header.token_count = end
        self.prepared_page.header.token_count = end
        self.source_page.escape_payload = self.host_buffer[:end]
        self.source_page.escape_scales = None if self.host_scales is None else self.host_scales[:end]
        self.token_count = end
        return start, end

    def append_rows_from_device(
        self,
        *,
        rows: np.ndarray,
        device_rows: Any,
        token_start: int,
    ) -> None:
        import torch

        values = np.asarray(rows, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.config.head_dim:
            raise ValueError("tail rows must have shape [token_count, head_dim]")
        if values.shape[0] == 0:
            return
        if self.host_buffer is None or self.prepared_page is None:
            raise RuntimeError("persistent tail page is not initialized")
        start, end = self.prepare_append_span(token_start=token_start, row_count=values.shape[0])
        converted, scales = _quantize_tail_rows_numpy(values, self.config.escape_dtype)
        self.host_buffer[start:end] = converted
        if self.host_scales is not None and scales is not None:
            self.host_scales[start:end] = scales
        if self.prepared_page.escape_payload.dtype == device_rows.dtype:
            self.prepared_page.escape_payload[start:end, : self.config.head_dim] = device_rows
        else:
            if self.config.escape_dtype != "int8":
                self.prepared_page.escape_payload[start:end, : self.config.head_dim] = device_rows.to(
                    dtype=self.prepared_page.escape_payload.dtype
                )
            else:
                row_scales = torch.clamp(device_rows.abs().amax(dim=-1) / 127.0, min=1e-8).to(
                    dtype=self.prepared_page.escape_scales.dtype
                )
                quantized = torch.clamp(torch.round(device_rows / row_scales[:, None]), -127.0, 127.0).to(dtype=torch.int8)
                self.prepared_page.escape_payload[start:end, : self.config.head_dim] = quantized
                if self.prepared_page.escape_scales is None:
                    raise RuntimeError("int8 persistent tail is missing escape scales")
                self.prepared_page.escape_scales[start:end] = row_scales
                self._refresh_runtime_metadata()
                return
        if self.prepared_page.escape_scales is not None and scales is not None:
            self.prepared_page.escape_scales[start:end] = torch.from_numpy(np.ascontiguousarray(scales)).to(
                device=self.device_type,
                dtype=self.prepared_page.escape_scales.dtype,
            )
        self._refresh_runtime_metadata()

    def append_device_rows(
        self,
        device_rows,
        *,
        token_start: int,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required for the persistent torch tail path") from exc
        if not torch.is_tensor(device_rows):
            raise TypeError("append_device_rows requires a torch.Tensor")
        if device_rows.ndim != 2 or int(device_rows.shape[1]) != self.config.head_dim:
            raise ValueError("tail rows must have shape [token_count, head_dim]")
        if int(device_rows.shape[0]) == 0:
            return
        if self.prepared_page is None:
            self._ensure_allocated(token_start=token_start if self.token_count == 0 else self.source_page.header.token_start)
        if self.prepared_page is None:
            raise RuntimeError("persistent tail page is not initialized")
        start, end = self.prepare_append_span(token_start=token_start, row_count=int(device_rows.shape[0]))
        if self.config.escape_dtype != "int8":
            self.prepared_page.escape_payload[start:end, : self.config.head_dim] = device_rows.to(
                dtype=self.prepared_page.escape_payload.dtype
            )
            self._refresh_runtime_metadata()
            return
        row_scales = torch.clamp(device_rows.abs().amax(dim=-1) / 127.0, min=1e-8).to(
            dtype=self.prepared_page.escape_scales.dtype
        )
        quantized = torch.clamp(torch.round(device_rows / row_scales[:, None]), -127.0, 127.0).to(dtype=torch.int8)
        self.prepared_page.escape_payload[start:end, : self.config.head_dim] = quantized
        if self.prepared_page.escape_scales is None:
            raise RuntimeError("int8 persistent tail is missing escape scales")
        self.prepared_page.escape_scales[start:end] = row_scales
        self._refresh_runtime_metadata()

    def materialize_rows(self) -> np.ndarray:
        if self.prepared_page is None or self.token_count <= 0:
            return np.zeros((0, self.config.head_dim), dtype=np.float32)
        payload = self.prepared_page.escape_payload[: self.token_count, : self.config.head_dim].detach().cpu().numpy()
        if self.prepared_page.header.escape_dtype == "int8":
            if self.prepared_page.escape_scales is None:
                raise RuntimeError("int8 persistent tail is missing escape scales")
            scales = self.prepared_page.escape_scales[: self.token_count].detach().cpu().numpy()
            return payload.astype(np.float32, copy=False) * scales.astype(np.float32, copy=False)[:, None]
        return payload.astype(np.float32, copy=False)

    def append_rows(
        self,
        rows: np.ndarray,
        *,
        token_start: int,
        trace: ExecutionTrace | None = None,
    ) -> None:
        values = np.asarray(rows, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.config.head_dim:
            raise ValueError("tail rows must have shape [token_count, head_dim]")
        if values.shape[0] == 0:
            return
        self._ensure_allocated(token_start=token_start if self.token_count == 0 else self.source_page.header.token_start)
        if self.source_page is None or self.prepared_page is None or self.host_buffer is None:
            raise RuntimeError("persistent tail page is not initialized")
        converted, scales = _quantize_tail_rows_numpy(values, self.config.escape_dtype)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required for the persistent torch tail path") from exc
        row_tensor = torch.from_numpy(np.ascontiguousarray(converted)).to(device=self.device_type)
        start, end = self.prepare_append_span(token_start=token_start, row_count=values.shape[0])
        self.host_buffer[start:end] = converted
        if self.host_scales is not None and scales is not None:
            self.host_scales[start:end] = scales
        self.prepared_page.escape_payload[start:end, : self.config.head_dim] = row_tensor
        if trace is not None:
            trace.record_host_to_device(int(row_tensor.numel() * row_tensor.element_size()))
        if self.prepared_page.escape_scales is not None and scales is not None:
            scale_tensor = torch.from_numpy(np.ascontiguousarray(scales)).to(
                device=self.device_type,
                dtype=self.prepared_page.escape_scales.dtype,
            )
            self.prepared_page.escape_scales[start:end] = scale_tensor
            if trace is not None:
                trace.record_host_to_device(int(scale_tensor.numel() * scale_tensor.element_size()))
        self._refresh_runtime_metadata()

    @property
    def active_page(self) -> PreparedPageTorch | None:
        if self.token_count <= 0:
            return None
        return self.prepared_page


@dataclass(slots=True)
class _HeadSessionState:
    session: PagedDecodeSession
    tail: _TailPageBuilder
    persistent_key_tail: _PersistentTailPage | None = None
    persistent_value_tail: _PersistentTailPage | None = None
    decode_key_pages_with_tail: list[PageLike] | None = None
    decode_value_pages_with_tail: list[PageLike] | None = None
    decode_view_layout: _PreparedDecodeViewLayout | None = None
    execution_builtin_selector_cache: _ExecutionBuiltinSelectorCache | None = None
    sequence_length: int = 0
    tracked_direct_prepared_pages: dict[int, int] = field(default_factory=dict)
    tracked_tail_resident_bytes: int = 0

    def invalidate_decode_views(self) -> None:
        self.decode_key_pages_with_tail = None
        self.decode_value_pages_with_tail = None
        self.decode_view_layout = None
        self.execution_builtin_selector_cache = None

    def clear(self, *, clear_prepared_cache: bool) -> None:
        self.session.key_pages.clear()
        self.session.value_pages.clear()
        self.session.key_page_sketches.clear()
        self.session.key_page_minima.clear()
        self.session.key_page_maxima.clear()
        self.session.value_page_summaries.clear()
        self.session.last_selected_indices.clear()
        if clear_prepared_cache and self.session.cache is not None:
            self.session.cache.clear()
        self.tail.clear()
        if self.persistent_key_tail is not None:
            self.persistent_key_tail.clear()
        if self.persistent_value_tail is not None:
            self.persistent_value_tail.clear()
        self.invalidate_decode_views()
        self.sequence_length = 0


class ModelPagedKVCache:
    def __init__(
        self,
        *,
        config: DotCacheConfig,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        backend: BackendName = "auto",
        cache: PreparedPageCache | None = None,
    ) -> None:
        self.config = config
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.backend = backend
        self.cache = cache if cache is not None else PreparedPageCache()
        self.default_q_head_to_kv_head = default_q_head_to_kv_head(self.num_attention_heads, self.num_key_value_heads)
        self.default_grouped_query_heads = _group_query_heads(
            self.default_q_head_to_kv_head,
            num_key_value_heads=self.num_key_value_heads,
        )
        self._states: dict[tuple[int, int], _HeadSessionState] = {}
        self._m2_prefilter_invocations = 0
        self._m2_prefilter_candidate_pages = 0
        self._m2_prefilter_selected_pages = 0
        self._decode_path_counts: dict[str, int] = {
            "grouped_batched": 0,
            "per_kv_fallback": 0,
        }
        self._decode_path_counts_by_layer: dict[int, dict[str, int]] = {}
        self._execution_shortlist_invocations = 0
        self._execution_shortlist_applied = 0
        self._execution_shortlist_group_union_applied = 0
        self._execution_shortlist_grouping_rejections = 0
        self._execution_shortlist_grouping_rejection_reason_counts: dict[str, int] = {}
        self._execution_shortlist_grouping_rejection_reason_counts_by_layer: dict[int, dict[str, int]] = {}
        self._execution_shortlist_total_pages = 0
        self._execution_shortlist_selected_pages = 0
        self._execution_shortlist_invocations_by_layer: dict[int, int] = {}
        self._execution_shortlist_applied_by_layer: dict[int, int] = {}
        self._execution_shortlist_group_union_applied_by_layer: dict[int, int] = {}
        self._execution_shortlist_grouping_rejections_by_layer: dict[int, int] = {}
        self._execution_shortlist_total_pages_by_layer: dict[int, int] = {}
        self._execution_shortlist_selected_pages_by_layer: dict[int, int] = {}
        self._execution_shortlist_trace_records: list[dict[str, object]] = []
        self._execution_exact_refine_invocations = 0
        self._execution_exact_refine_candidate_pages = 0
        self._execution_exact_refine_selected_pages = 0
        self._execution_exact_refine_invocations_by_layer: dict[int, int] = {}
        self._execution_exact_refine_candidate_pages_by_layer: dict[int, int] = {}
        self._execution_exact_refine_selected_pages_by_layer: dict[int, int] = {}
        self._decode_grouped_batch_rejection_reason_counts: dict[str, int] = {}
        self._decode_grouped_batch_rejection_reason_counts_by_layer: dict[int, dict[str, int]] = {}
        self._decode_stage_timings = _empty_decode_stage_timing_totals()
        self._decode_stage_timings_by_layer: dict[int, dict[str, float]] = {}
        self._direct_prepared_page_resident_bytes = 0
        self._direct_prepared_page_refcounts: dict[int, int] = {}
        self._direct_prepared_page_sizes: dict[int, int] = {}
        self._tail_resident_bytes = 0
        self._chunk_budget_dirty_marks = 0
        self._chunk_budget_dirty_transitions = 0
        self._chunk_budget_dirty_reason_counts: dict[str, int] = {}
        self._chunk_budget_sync_invocations = 0
        self._chunk_budget_sync_clean_skips = 0
        self._chunk_budget_sync_dirty_invocations = 0
        self._chunk_budget_override_calls = 0
        self._chunk_budget_override_budget_change_calls = 0
        self._chunk_budget_override_same_budget_calls = 0
        self._chunk_budget_freeze_override_calls = 0
        self._builtin_selector_score_all_pages_calls = 0
        self._builtin_selector_candidate_only_calls = 0
        self._builtin_selector_candidate_pages = 0
        self._builtin_selector_total_pages = 0
        self._builtin_selector_candidate_fraction_sum = 0.0
        self._builtin_selector_candidate_fraction_max = 0.0
        self._builtin_selector_cache_hits = 0
        self._builtin_selector_cache_builds = 0
        self._builtin_selector_cache_build_bytes = 0
        self._builtin_selector_cache_build_bytes_max = 0
        self._execution_value_escape_cache: dict[tuple[int, int, int, str, str], PageLike] = {}
        self._execution_value_escape_source_pages: dict[tuple[int, str], EncodedPage] = {}
        self._execution_value_escape_cache_hits = 0
        self._execution_value_escape_source_registrations = 0
        self._execution_value_escape_prepared_page_builds = 0
        self._execution_value_escape_prewarm_invocations = 0
        self._execution_value_escape_prewarm_pages = 0
        self._execution_value_escape_prewarm_ms_total = 0.0
        self._execution_value_escape_builds = 0
        self._execution_value_escape_applied_pages = 0
        self._prepared_chunk_cache_frozen_budget_bytes: int | None = None
        self._prepared_chunk_cache_applied_budget_bytes: int | None = None
        self._prepared_chunk_cache_budget_dirty = True
        self._learned_page_selector_model: LinearSelectorModel | CandidateSafeRouterModel | CandidateTargetRouterModel | None = None
        self._learned_page_selector_invocations = 0
        self._learned_page_selector_predictions: dict[str, int] = {}
        self._learned_page_selector_fallbacks = 0
        self._learned_page_selector_ms_total = 0.0
        self._learned_page_selector_invocations_by_stage: dict[str, int] = {}
        self._learned_page_selector_fallbacks_by_stage: dict[str, int] = {}
        self._learned_page_selector_ms_total_by_stage: dict[str, float] = {}
        self._learned_page_selector_predictions_by_stage: dict[str, dict[str, int]] = {}
        if self.config.learned_page_selector_enabled():
            self._learned_page_selector_model = load_page_selector_artifact(str(self.config.learned_page_selector_path))
            if (
                isinstance(self._learned_page_selector_model, LinearSelectorModel)
                and float(self.config.learned_page_selector_logit_offset) != 0.0
            ):
                self._learned_page_selector_model = adjust_linear_selector_model_logits(
                    self._learned_page_selector_model,
                    candidate_logit_offsets={
                        str(self.config.learned_page_selector_target_candidate): float(
                            self.config.learned_page_selector_logit_offset
                        )
                    },
                )

    @property
    def resident_bytes(self) -> int:
        return self.resident_byte_summary()["resident_bytes"]

    @staticmethod
    def _prepared_page_resident_bytes(page: PreparedPageTorch) -> int:
        resident_nbytes = int(page.resident_nbytes)
        return resident_nbytes if resident_nbytes > 0 else int(page.host_to_device_nbytes)

    def _collect_state_direct_prepared_pages(self, state: _HeadSessionState) -> dict[int, int]:
        direct_pages: dict[int, int] = {}
        for page in state.session.key_pages:
            if isinstance(page, PreparedPageTorch) and not self.cache.owns_prepared_page(page):
                direct_pages[id(page)] = self._prepared_page_resident_bytes(page)
        for page in state.session.value_pages:
            if isinstance(page, PreparedPageTorch) and not self.cache.owns_prepared_page(page):
                direct_pages[id(page)] = self._prepared_page_resident_bytes(page)
        return direct_pages

    @staticmethod
    def _collect_state_tail_resident_bytes(state: _HeadSessionState) -> int:
        total = 0
        if state.persistent_key_tail is not None:
            total += int(state.persistent_key_tail.resident_nbytes)
        if state.persistent_value_tail is not None:
            total += int(state.persistent_value_tail.resident_nbytes)
        return total

    def _reset_resident_accounting(self) -> None:
        self._direct_prepared_page_resident_bytes = 0
        self._direct_prepared_page_refcounts.clear()
        self._direct_prepared_page_sizes.clear()
        self._tail_resident_bytes = 0
        for state in self._states.values():
            state.tracked_direct_prepared_pages.clear()
            state.tracked_tail_resident_bytes = 0

    def _refresh_state_resident_accounting(self, state: _HeadSessionState) -> bool:
        resident_bytes_changed = False
        new_direct_pages = self._collect_state_direct_prepared_pages(state)
        old_direct_pages = state.tracked_direct_prepared_pages

        removed_page_ids = set(old_direct_pages) - set(new_direct_pages)
        for page_id in removed_page_ids:
            refcount = int(self._direct_prepared_page_refcounts.get(page_id, 0)) - 1
            if refcount <= 0:
                self._direct_prepared_page_refcounts.pop(page_id, None)
                removed_size = int(self._direct_prepared_page_sizes.pop(page_id, old_direct_pages[page_id]))
                self._direct_prepared_page_resident_bytes = max(
                    0,
                    int(self._direct_prepared_page_resident_bytes) - removed_size,
                )
            else:
                self._direct_prepared_page_refcounts[page_id] = refcount
            resident_bytes_changed = True

        added_page_ids = set(new_direct_pages) - set(old_direct_pages)
        for page_id in added_page_ids:
            page_size = int(new_direct_pages[page_id])
            refcount = int(self._direct_prepared_page_refcounts.get(page_id, 0))
            if refcount <= 0:
                self._direct_prepared_page_sizes[page_id] = page_size
                self._direct_prepared_page_resident_bytes += page_size
                resident_bytes_changed = True
            self._direct_prepared_page_refcounts[page_id] = refcount + 1

        for page_id, page_size in new_direct_pages.items():
            if page_id not in old_direct_pages:
                continue
            previous_size = int(self._direct_prepared_page_sizes.get(page_id, old_direct_pages[page_id]))
            if previous_size == int(page_size):
                continue
            self._direct_prepared_page_sizes[page_id] = int(page_size)
            self._direct_prepared_page_resident_bytes += int(page_size) - previous_size
            resident_bytes_changed = True

        new_tail_resident_bytes = self._collect_state_tail_resident_bytes(state)
        if new_tail_resident_bytes != int(state.tracked_tail_resident_bytes):
            self._tail_resident_bytes += int(new_tail_resident_bytes) - int(state.tracked_tail_resident_bytes)
            state.tracked_tail_resident_bytes = int(new_tail_resident_bytes)
            resident_bytes_changed = True

        state.tracked_direct_prepared_pages = new_direct_pages
        return resident_bytes_changed

    def _rebuild_resident_accounting(self) -> None:
        self._reset_resident_accounting()
        for state in self._states.values():
            self._refresh_state_resident_accounting(state)

    def _reset_chunk_budget_tracking(self) -> None:
        self._chunk_budget_dirty_marks = 0
        self._chunk_budget_dirty_transitions = 0
        self._chunk_budget_dirty_reason_counts = {}
        self._chunk_budget_sync_invocations = 0
        self._chunk_budget_sync_clean_skips = 0
        self._chunk_budget_sync_dirty_invocations = 0
        self._chunk_budget_override_calls = 0
        self._chunk_budget_override_budget_change_calls = 0
        self._chunk_budget_override_same_budget_calls = 0
        self._chunk_budget_freeze_override_calls = 0

    def _reset_builtin_selector_tracking(self) -> None:
        self._builtin_selector_score_all_pages_calls = 0
        self._builtin_selector_candidate_only_calls = 0
        self._builtin_selector_candidate_pages = 0
        self._builtin_selector_total_pages = 0
        self._builtin_selector_candidate_fraction_sum = 0.0
        self._builtin_selector_candidate_fraction_max = 0.0
        self._builtin_selector_cache_hits = 0
        self._builtin_selector_cache_builds = 0
        self._builtin_selector_cache_build_bytes = 0
        self._builtin_selector_cache_build_bytes_max = 0

    def _reset_execution_value_escape_tracking(self) -> None:
        self._execution_value_escape_cache.clear()
        self._execution_value_escape_source_pages.clear()
        self._execution_value_escape_cache_hits = 0
        self._execution_value_escape_source_registrations = 0
        self._execution_value_escape_prepared_page_builds = 0
        self._execution_value_escape_prewarm_invocations = 0
        self._execution_value_escape_prewarm_pages = 0
        self._execution_value_escape_prewarm_ms_total = 0.0
        self._execution_value_escape_builds = 0
        self._execution_value_escape_applied_pages = 0

    def _kv_resident_byte_summary(self) -> dict[str, int]:
        static_resident_bytes = int(self._direct_prepared_page_resident_bytes)
        tail_resident_bytes = int(self._tail_resident_bytes)
        kv_resident_bytes = int(self.cache.resident_bytes) + static_resident_bytes + tail_resident_bytes
        return {
            "prepared_page_cache_resident_bytes": int(self.cache.resident_bytes),
            "direct_page_resident_bytes": int(static_resident_bytes),
            "tail_resident_bytes": int(tail_resident_bytes),
            "kv_resident_bytes": int(kv_resident_bytes),
        }

    def _prepared_chunk_cache_budget_bytes(self, *, kv_resident_bytes: int | None = None) -> int:
        if self._torch_device_type is None:
            return 0
        budget_ratio = float(self.config.prepared_chunk_cache_budget_ratio)
        if budget_ratio <= 0.0 or int(self.config.prepared_chunk_cache_max_bytes) <= 0:
            return 0
        if kv_resident_bytes is None:
            kv_resident_bytes = int(self._kv_resident_byte_summary()["kv_resident_bytes"])
        adaptive_budget = max(
            int(self.config.prepared_chunk_cache_min_bytes),
            int(kv_resident_bytes * budget_ratio),
        )
        return min(int(self.config.prepared_chunk_cache_max_bytes), adaptive_budget)

    def _sync_prepared_chunk_cache_budget(self, *, freeze_during_decode: bool = False) -> None:
        if self._torch_device_type is None:
            return
        self._chunk_budget_sync_invocations += 1
        if not self._prepared_chunk_cache_budget_dirty:
            self._chunk_budget_sync_clean_skips += 1
            return
        self._chunk_budget_sync_dirty_invocations += 1
        if bool(freeze_during_decode and self.config.execution_freeze_chunk_budget_during_decode):
            if self._prepared_chunk_cache_frozen_budget_bytes is None:
                self._prepared_chunk_cache_frozen_budget_bytes = int(self._prepared_chunk_cache_budget_bytes())
            if self._prepared_chunk_cache_applied_budget_bytes != int(self._prepared_chunk_cache_frozen_budget_bytes):
                applied_budget_bytes = self._prepared_chunk_cache_applied_budget_bytes
                set_prepared_chunk_cache_budget_override(
                    max_resident_bytes=self._prepared_chunk_cache_frozen_budget_bytes,
                )
                self._chunk_budget_override_calls += 1
                self._chunk_budget_freeze_override_calls += 1
                if applied_budget_bytes == int(self._prepared_chunk_cache_frozen_budget_bytes):
                    self._chunk_budget_override_same_budget_calls += 1
                else:
                    self._chunk_budget_override_budget_change_calls += 1
                self._prepared_chunk_cache_applied_budget_bytes = int(self._prepared_chunk_cache_frozen_budget_bytes)
            self._prepared_chunk_cache_budget_dirty = False
            return
        self._prepared_chunk_cache_frozen_budget_bytes = None
        budget_bytes = int(self._prepared_chunk_cache_budget_bytes())
        applied_budget_bytes = self._prepared_chunk_cache_applied_budget_bytes
        set_prepared_chunk_cache_budget_override(
            max_resident_bytes=budget_bytes,
        )
        self._chunk_budget_override_calls += 1
        if applied_budget_bytes == budget_bytes:
            self._chunk_budget_override_same_budget_calls += 1
        else:
            self._chunk_budget_override_budget_change_calls += 1
        self._prepared_chunk_cache_applied_budget_bytes = budget_bytes
        self._prepared_chunk_cache_budget_dirty = False

    def _mark_prepared_chunk_cache_budget_dirty(self, *, reason: str) -> None:
        if self._torch_device_type is None:
            return
        self._chunk_budget_dirty_marks += 1
        self._chunk_budget_dirty_reason_counts[str(reason)] = (
            int(self._chunk_budget_dirty_reason_counts.get(str(reason), 0)) + 1
        )
        if not self._prepared_chunk_cache_budget_dirty:
            self._chunk_budget_dirty_transitions += 1
        self._prepared_chunk_cache_budget_dirty = True

    def chunk_budget_summary(self) -> dict[str, object]:
        return {
            "execution_chunk_budget_dirty_marks": int(self._chunk_budget_dirty_marks),
            "execution_chunk_budget_dirty_transitions": int(self._chunk_budget_dirty_transitions),
            "execution_chunk_budget_dirty_reason_counts": {
                reason: int(count) for reason, count in sorted(self._chunk_budget_dirty_reason_counts.items())
            },
            "execution_chunk_budget_sync_invocations": int(self._chunk_budget_sync_invocations),
            "execution_chunk_budget_sync_clean_skips": int(self._chunk_budget_sync_clean_skips),
            "execution_chunk_budget_sync_dirty_invocations": int(self._chunk_budget_sync_dirty_invocations),
            "execution_chunk_budget_override_calls": int(self._chunk_budget_override_calls),
            "execution_chunk_budget_override_budget_change_calls": int(
                self._chunk_budget_override_budget_change_calls
            ),
            "execution_chunk_budget_override_same_budget_calls": int(
                self._chunk_budget_override_same_budget_calls
            ),
            "execution_chunk_budget_freeze_override_calls": int(self._chunk_budget_freeze_override_calls),
        }

    def builtin_selector_summary(self) -> dict[str, int | float]:
        return {
            "execution_builtin_selector_score_all_pages_calls": int(self._builtin_selector_score_all_pages_calls),
            "execution_builtin_selector_candidate_only_calls": int(self._builtin_selector_candidate_only_calls),
            "execution_builtin_selector_candidate_pages": int(self._builtin_selector_candidate_pages),
            "execution_builtin_selector_total_pages": int(self._builtin_selector_total_pages),
            "execution_builtin_selector_candidate_fraction_sum": float(
                self._builtin_selector_candidate_fraction_sum
            ),
            "execution_builtin_selector_candidate_fraction_max": float(
                self._builtin_selector_candidate_fraction_max
            ),
            "execution_builtin_selector_cache_hits": int(self._builtin_selector_cache_hits),
            "execution_builtin_selector_cache_builds": int(self._builtin_selector_cache_builds),
            "execution_builtin_selector_cache_build_bytes": int(self._builtin_selector_cache_build_bytes),
            "execution_builtin_selector_cache_build_bytes_max": int(self._builtin_selector_cache_build_bytes_max),
        }

    def execution_value_escape_summary(self) -> dict[str, int | str | list[int]]:
        return {
            "execution_value_escape_layers": [int(layer_id) for layer_id in self.config.execution_value_escape_layers],
            "execution_value_escape_mode": str(self.config.execution_value_escape_mode),
            "execution_value_escape_old_only": bool(self.config.execution_value_escape_old_only),
            "execution_value_escape_top_k": int(self.config.execution_value_escape_top_k),
            "execution_value_escape_prewarm": bool(self.config.execution_value_escape_prewarm),
            "execution_value_escape_prewarm_min_context": int(self.config.execution_value_escape_prewarm_min_context),
            "execution_value_escape_cache_hits": int(self._execution_value_escape_cache_hits),
            "execution_value_escape_source_registrations": int(self._execution_value_escape_source_registrations),
            "execution_value_escape_prepared_page_builds": int(self._execution_value_escape_prepared_page_builds),
            "execution_value_escape_prewarm_invocations": int(self._execution_value_escape_prewarm_invocations),
            "execution_value_escape_prewarm_pages": int(self._execution_value_escape_prewarm_pages),
            "execution_value_escape_prewarm_ms_total": float(self._execution_value_escape_prewarm_ms_total),
            "execution_value_escape_builds": int(self._execution_value_escape_builds),
            "execution_value_escape_applied_pages": int(self._execution_value_escape_applied_pages),
        }

    def _execution_value_escape_cache_key(self, page: PageLike, *, escape_mode: str) -> tuple[int, int, int, str, str]:
        source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
        header = source_page.header
        return (
            id(source_page),
            int(header.token_start),
            int(header.token_count),
            str(escape_mode),
            str(self._torch_device_type or "cpu_ref"),
        )

    def _execution_value_escape_source_key(self, page: PageLike, *, escape_mode: str) -> tuple[int, str]:
        source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
        return (id(source_page), str(escape_mode))

    def _maybe_register_execution_value_escape_source(
        self,
        source_page: EncodedPage,
        *,
        dense_values: np.ndarray,
        escape_mode: str,
    ) -> None:
        if not self.config.execution_value_escape_enabled_for_layer(layer_id=int(source_page.header.layer_id)):
            return
        if str(source_page.header.kind) != "V":
            return
        if str(source_page.header.mode_default) == str(escape_mode):
            return
        source_key = self._execution_value_escape_source_key(source_page, escape_mode=escape_mode)
        if source_key in self._execution_value_escape_source_pages:
            return
        self._execution_value_escape_source_pages[source_key] = encode_page(
            np.asarray(dense_values, dtype=np.float32, copy=False),
            self.config,
            kind="V",
            layer_id=int(source_page.header.layer_id),
            kv_head_id=int(source_page.header.kv_head_id),
            token_start=int(source_page.header.token_start),
            mode=str(escape_mode),
            build_runtime_metadata=False,
        )
        self._execution_value_escape_source_registrations += 1
        self._execution_value_escape_builds += 1

    def _prepare_execution_value_escape_page(
        self,
        page: PageLike,
        *,
        escape_mode: str,
        trace: ExecutionTrace | None = None,
    ) -> PageLike:
        source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
        if str(source_page.header.kind) != "V":
            raise ValueError("execution value escape requires V pages")
        if str(source_page.header.mode_default) == str(escape_mode):
            return page
        source_key = self._execution_value_escape_source_key(page, escape_mode=escape_mode)
        exact_source_page = self._execution_value_escape_source_pages.get(source_key)
        if exact_source_page is not None:
            exact_cache_key = self._execution_value_escape_cache_key(exact_source_page, escape_mode=escape_mode)
            cached_exact_page = self._execution_value_escape_cache.get(exact_cache_key)
            if cached_exact_page is not None:
                self._execution_value_escape_cache_hits += 1
                return cached_exact_page
            prepared_exact_page = prepare_pages(
                [exact_source_page],
                backend=self.backend,
                cache=self.cache,
                trace=trace,
            )[0]
            self._execution_value_escape_cache[exact_cache_key] = prepared_exact_page
            self._execution_value_escape_prepared_page_builds += 1
            self._execution_value_escape_builds += 1
            return prepared_exact_page
        cache_key = self._execution_value_escape_cache_key(page, escape_mode=escape_mode)
        cached_page = self._execution_value_escape_cache.get(cache_key)
        if cached_page is not None:
            self._execution_value_escape_cache_hits += 1
            return cached_page
        dense_values = decode_page(source_page).astype(np.float32, copy=False)
        escaped_page = encode_page(
            dense_values,
            self.config,
            kind="V",
            layer_id=int(source_page.header.layer_id),
            kv_head_id=int(source_page.header.kv_head_id),
            token_start=int(source_page.header.token_start),
            mode=str(escape_mode),
            build_runtime_metadata=False,
        )
        prepared_escape_page = prepare_pages(
            [escaped_page],
            backend=self.backend,
            cache=self.cache,
            trace=trace,
        )[0]
        self._execution_value_escape_cache[cache_key] = prepared_escape_page
        self._execution_value_escape_prepared_page_builds += 1
        self._execution_value_escape_builds += 1
        return prepared_escape_page

    def _maybe_prewarm_execution_value_escape_pages(
        self,
        state: _HeadSessionState,
        *,
        trace: ExecutionTrace | None = None,
    ) -> None:
        if not bool(self.config.execution_value_escape_prewarm):
            return
        layer_id = int(state.tail.layer_id)
        if not self.config.execution_value_escape_enabled_for_layer(layer_id=layer_id):
            return
        min_context = max(0, int(self.config.execution_value_escape_prewarm_min_context))
        if min_context > 0 and int(state.sequence_length) < min_context:
            return
        if bool(self.config.execution_value_escape_old_only) or int(self.config.execution_value_escape_top_k) > 0:
            return
        if not state.session.value_pages:
            return
        started_at = perf_counter()
        prewarmed_pages = 0
        escape_mode = str(self.config.execution_value_escape_mode)
        for page in state.session.value_pages:
            source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
            if str(source_page.header.kind) != "V":
                continue
            if str(source_page.header.mode_default) == escape_mode:
                continue
            prepared_page = self._prepare_execution_value_escape_page(page, escape_mode=escape_mode, trace=trace)
            if prepared_page is not page:
                prewarmed_pages += 1
        if prewarmed_pages <= 0:
            return
        self._execution_value_escape_prewarm_invocations += 1
        self._execution_value_escape_prewarm_pages += int(prewarmed_pages)
        self._execution_value_escape_prewarm_ms_total += float((perf_counter() - started_at) * 1000.0)

    def _apply_execution_value_escape(
        self,
        *,
        layer_id: int,
        key_pages_by_group: Sequence[Sequence[PageLike]],
        value_pages_by_group: Sequence[Sequence[PageLike]],
        context_lengths_by_group: Sequence[int] | None = None,
        representative_queries_by_group: Sequence[np.ndarray] | None = None,
        trace: ExecutionTrace | None = None,
    ) -> tuple[list[Sequence[PageLike]], bool]:
        if not self.config.execution_value_escape_enabled_for_layer(layer_id=layer_id):
            return [list(pages) for pages in value_pages_by_group], False
        escape_mode = str(self.config.execution_value_escape_mode)
        escape_top_k = max(0, int(self.config.execution_value_escape_top_k))
        escaped_groups: list[Sequence[PageLike]] = []
        any_applied = False
        for group_index, (key_pages, value_pages) in enumerate(zip(key_pages_by_group, value_pages_by_group, strict=True)):
            escaped_pages: list[PageLike] = []
            group_applied = False
            eligible_indices: set[int] | None = None
            if bool(self.config.execution_value_escape_old_only):
                context_length = None
                if context_lengths_by_group is not None and group_index < len(context_lengths_by_group):
                    context_length = int(context_lengths_by_group[group_index])
                layer_recent_window = self.config.resolve_execution_recent_window_for_context(
                    layer_id=layer_id,
                    context_length=context_length,
                )
                eligible_indices = set(range(len(key_pages))) - set(
                    select_window_page_indices(
                        key_pages,
                        recent_window_tokens=layer_recent_window if layer_recent_window > 0 else None,
                        sink_window_tokens=int(self.config.execution_sink_window),
                    )
                )
            if escape_top_k > 0:
                query_slice = None
                if representative_queries_by_group is not None and group_index < len(representative_queries_by_group):
                    query_slice = np.asarray(representative_queries_by_group[group_index], dtype=np.float32)
                ranked_candidate_indices: list[int] = []
                if query_slice is not None:
                    scored_indices: list[tuple[float, int]] = []
                    candidate_pool = range(len(key_pages)) if eligible_indices is None else sorted(eligible_indices)
                    for page_index in candidate_pool:
                        score = _score_page_relevance_for_mode(
                            query_slice,
                            key_pages[page_index],
                            relevance_mode=self.config.execution_relevance_mode,
                        )
                        if score is not None:
                            scored_indices.append((float(score), int(page_index)))
                    if scored_indices:
                        scored_indices.sort(key=lambda item: item[0], reverse=True)
                        ranked_candidate_indices = [index for _, index in scored_indices[:escape_top_k]]
                if ranked_candidate_indices:
                    eligible_indices = set(ranked_candidate_indices)
                elif eligible_indices is None:
                    eligible_indices = set(range(min(len(key_pages), escape_top_k)))
            for page_index, page in enumerate(value_pages):
                if eligible_indices is not None and page_index not in eligible_indices:
                    escaped_pages.append(page)
                    continue
                source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
                if str(source_page.header.mode_default) == escape_mode:
                    escaped_pages.append(page)
                    continue
                escaped_page = self._prepare_execution_value_escape_page(
                    page,
                    escape_mode=escape_mode,
                    trace=trace,
                )
                escaped_pages.append(escaped_page)
                if escaped_page is not page:
                    group_applied = True
                    self._execution_value_escape_applied_pages += 1
            escaped_groups.append(escaped_pages if group_applied else list(value_pages))
            any_applied = any_applied or group_applied
        return escaped_groups, any_applied

    def resident_byte_summary(self) -> dict[str, int]:
        summary = self._kv_resident_byte_summary()
        chunk_resident_bytes = prepared_chunk_cache_resident_bytes() if self._torch_device_type is not None else 0
        budget_bytes = self._prepared_chunk_cache_budget_bytes(kv_resident_bytes=int(summary["kv_resident_bytes"]))
        return {
            **summary,
            "prepared_chunk_cache_budget_bytes": int(budget_bytes),
            "prepared_chunk_resident_bytes": int(chunk_resident_bytes),
            "resident_bytes": int(summary["kv_resident_bytes"] + chunk_resident_bytes),
        }

    def _record_decode_path(self, layer_id: int, path_name: str) -> None:
        if path_name not in self._decode_path_counts:
            raise ValueError(f"unknown decode path: {path_name}")
        self._decode_path_counts[path_name] += 1
        layer_counts = self._decode_path_counts_by_layer.setdefault(int(layer_id), {})
        layer_counts[path_name] = layer_counts.get(path_name, 0) + 1

    def decode_path_summary(self) -> dict[str, object]:
        return {
            "decode_path_counts": dict(sorted(self._decode_path_counts.items())),
            "decode_path_counts_by_layer": {
                str(layer_id): dict(sorted(counts.items()))
                for layer_id, counts in sorted(self._decode_path_counts_by_layer.items())
            },
            "decode_grouped_batch_rejection_reason_counts": dict(
                sorted(self._decode_grouped_batch_rejection_reason_counts.items())
            ),
            "decode_grouped_batch_rejection_reason_counts_by_layer": {
                str(layer_id): dict(sorted(counts.items()))
                for layer_id, counts in sorted(self._decode_grouped_batch_rejection_reason_counts_by_layer.items())
            },
        }

    def _record_execution_shortlist(
        self,
        *,
        layer_id: int,
        total_pages: int,
        selected_pages: int,
        applied: bool,
        group_union_applied: bool = False,
        grouping_rejected: bool = False,
        grouping_rejection_reason: str | None = None,
    ) -> None:
        self._execution_shortlist_invocations += 1
        self._execution_shortlist_total_pages += int(total_pages)
        self._execution_shortlist_selected_pages += int(selected_pages)
        self._execution_shortlist_invocations_by_layer[int(layer_id)] = (
            self._execution_shortlist_invocations_by_layer.get(int(layer_id), 0) + 1
        )
        self._execution_shortlist_total_pages_by_layer[int(layer_id)] = (
            self._execution_shortlist_total_pages_by_layer.get(int(layer_id), 0) + int(total_pages)
        )
        self._execution_shortlist_selected_pages_by_layer[int(layer_id)] = (
            self._execution_shortlist_selected_pages_by_layer.get(int(layer_id), 0) + int(selected_pages)
        )
        if applied:
            self._execution_shortlist_applied += 1
            self._execution_shortlist_applied_by_layer[int(layer_id)] = (
                self._execution_shortlist_applied_by_layer.get(int(layer_id), 0) + 1
            )
        if group_union_applied:
            self._execution_shortlist_group_union_applied += 1
            self._execution_shortlist_group_union_applied_by_layer[int(layer_id)] = (
                self._execution_shortlist_group_union_applied_by_layer.get(int(layer_id), 0) + 1
            )
        if grouping_rejected:
            self._execution_shortlist_grouping_rejections += 1
            self._execution_shortlist_grouping_rejections_by_layer[int(layer_id)] = (
                self._execution_shortlist_grouping_rejections_by_layer.get(int(layer_id), 0) + 1
            )
            if grouping_rejection_reason:
                self._execution_shortlist_grouping_rejection_reason_counts[grouping_rejection_reason] = (
                    self._execution_shortlist_grouping_rejection_reason_counts.get(grouping_rejection_reason, 0) + 1
                )
                layer_reason_counts = self._execution_shortlist_grouping_rejection_reason_counts_by_layer.setdefault(
                    int(layer_id), {}
                )
                layer_reason_counts[grouping_rejection_reason] = (
                    layer_reason_counts.get(grouping_rejection_reason, 0) + 1
                )

    def _record_decode_grouped_batch_rejection(self, *, layer_id: int, reason: str) -> None:
        self._decode_grouped_batch_rejection_reason_counts[reason] = (
            self._decode_grouped_batch_rejection_reason_counts.get(reason, 0) + 1
        )
        layer_reason_counts = self._decode_grouped_batch_rejection_reason_counts_by_layer.setdefault(int(layer_id), {})
        layer_reason_counts[reason] = layer_reason_counts.get(reason, 0) + 1

    def _record_execution_exact_refine(
        self,
        *,
        layer_id: int,
        candidate_pages: int,
        selected_pages: int,
    ) -> None:
        self._execution_exact_refine_invocations += 1
        self._execution_exact_refine_candidate_pages += int(candidate_pages)
        self._execution_exact_refine_selected_pages += int(selected_pages)
        self._execution_exact_refine_invocations_by_layer[int(layer_id)] = (
            self._execution_exact_refine_invocations_by_layer.get(int(layer_id), 0) + 1
        )
        self._execution_exact_refine_candidate_pages_by_layer[int(layer_id)] = (
            self._execution_exact_refine_candidate_pages_by_layer.get(int(layer_id), 0) + int(candidate_pages)
        )
        self._execution_exact_refine_selected_pages_by_layer[int(layer_id)] = (
            self._execution_exact_refine_selected_pages_by_layer.get(int(layer_id), 0) + int(selected_pages)
        )

    def _record_decode_stage_timing(self, *, layer_id: int, stage: str, ms: float) -> None:
        if stage not in _DECODE_STAGE_TIMING_STAGES:
            raise ValueError(f"unknown decode stage timing: {stage}")
        self._decode_stage_timings[stage] += float(ms)
        layer_timings = self._decode_stage_timings_by_layer.setdefault(int(layer_id), {})
        layer_timings[stage] = float(layer_timings.get(stage, 0.0) + float(ms))

    def _record_builtin_selector_stats(
        self,
        *,
        candidate_pages: int,
        total_pages: int,
        candidate_fraction: float,
        used_score_all_pages: bool,
    ) -> None:
        if used_score_all_pages:
            self._builtin_selector_score_all_pages_calls += 1
        else:
            self._builtin_selector_candidate_only_calls += 1
        self._builtin_selector_candidate_pages += int(candidate_pages)
        self._builtin_selector_total_pages += int(total_pages)
        self._builtin_selector_candidate_fraction_sum += float(candidate_fraction)
        self._builtin_selector_candidate_fraction_max = max(
            float(self._builtin_selector_candidate_fraction_max),
            float(candidate_fraction),
        )

    def decode_stage_runtime_totals(self) -> dict[str, float]:
        return {
            _decode_stage_summary_key(stage): float(self._decode_stage_timings.get(stage, 0.0))
            for stage in _DECODE_STAGE_TIMING_STAGES
        }

    def _execution_builtin_selector_cache_for_state(
        self,
        state: _HeadSessionState,
        *,
        relevance_mode: str,
    ) -> _ExecutionBuiltinSelectorCache | None:
        direct_key_pages = state.session.key_pages
        if not direct_key_pages:
            return None
        page_signature = tuple(
            (
                id(page.source_page if isinstance(page, PreparedPageTorch) else page),
                int(_page_header(page).token_start),
                int(_page_header(page).token_count),
            )
            for page in direct_key_pages
        )
        cache = state.execution_builtin_selector_cache
        if cache is not None and cache.page_signature == page_signature:
            if relevance_mode == "sketch" and cache.sketch_matrix is not None:
                self._builtin_selector_cache_hits += 1
                return cache
            if relevance_mode == "envelope" and cache.minima_matrix is not None and cache.maxima_matrix is not None:
                self._builtin_selector_cache_hits += 1
                return cache
        if relevance_mode == "sketch":
            sketches = []
            for page in direct_key_pages:
                source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
                if source_page.runtime_page_sketch is None:
                    return None
                sketches.append(np.asarray(source_page.runtime_page_sketch, dtype=np.float32))
            cache = _ExecutionBuiltinSelectorCache(
                page_signature=page_signature,
                sketch_matrix=np.stack(sketches, axis=0),
            )
        elif relevance_mode == "envelope":
            minima = []
            maxima = []
            for page in direct_key_pages:
                source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
                if source_page.runtime_page_min is None or source_page.runtime_page_max is None:
                    return None
                minima.append(np.asarray(source_page.runtime_page_min, dtype=np.float32))
                maxima.append(np.asarray(source_page.runtime_page_max, dtype=np.float32))
            cache = _ExecutionBuiltinSelectorCache(
                page_signature=page_signature,
                minima_matrix=np.stack(minima, axis=0),
                maxima_matrix=np.stack(maxima, axis=0),
            )
        else:
            return None
        build_bytes = int(cache.resident_bytes())
        self._builtin_selector_cache_builds += 1
        self._builtin_selector_cache_build_bytes += int(build_bytes)
        self._builtin_selector_cache_build_bytes_max = max(
            int(self._builtin_selector_cache_build_bytes_max),
            int(build_bytes),
        )
        state.execution_builtin_selector_cache = cache
        return cache

    def _should_prewarm_execution_builtin_selector_cache(self) -> bool:
        return bool(
            self.config.execution_builtin_selector_cache
            and (
                int(self.config.execution_relevance_top_k) > 0
                or bool(self.config.execution_relevance_top_k_overrides)
                or bool(self.config.execution_relevance_top_k_context_overrides)
            )
        )

    def _maybe_prewarm_execution_builtin_selector_cache(self, state: _HeadSessionState) -> None:
        if not self._should_prewarm_execution_builtin_selector_cache():
            return
        if not state.session.key_pages:
            return
        if not all(isinstance(page, PreparedPageTorch) for page in state.session.key_pages):
            return
        self._execution_builtin_selector_cache_for_state(
            state,
            relevance_mode=self.config.execution_relevance_mode,
        )

    def _execution_builtin_selector_matrices(
        self,
        *,
        layer_id: int,
        kv_head_id: int | None,
        key_pages: Sequence[PageLike],
        relevance_mode: str,
        score_all_pages_with_matrices: bool,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if not self.config.execution_builtin_selector_cache or kv_head_id is None:
            return None, None, None, None, None, None
        state = self._state(layer_id, kv_head_id)
        cache = self._execution_builtin_selector_cache_for_state(state, relevance_mode=relevance_mode)
        if cache is None:
            return None, None, None, None, None, None
        direct_key_pages = state.session.key_pages
        direct_count = len(direct_key_pages)
        if len(key_pages) not in (direct_count, direct_count + 1):
            return None, None, None, None, None, None
        if any(key_pages[index] is not direct_key_pages[index] for index in range(direct_count)):
            return None, None, None, None, None, None

        def _tail_source_page() -> EncodedPage | None:
            if len(key_pages) != direct_count + 1:
                return None
            tail_page = key_pages[-1]
            return tail_page.source_page if isinstance(tail_page, PreparedPageTorch) else tail_page

        tail_source_page = _tail_source_page()
        if relevance_mode == "sketch":
            if cache.sketch_matrix is None:
                return None, None, None, None, None, None
            if tail_source_page is None:
                return cache.sketch_matrix, None, None, None, None, None
            if tail_source_page.runtime_page_sketch is None:
                return None, None, None, None, None, None
            if score_all_pages_with_matrices:
                return (
                    cache.sketch_matrix,
                    None,
                    None,
                    np.asarray(tail_source_page.runtime_page_sketch, dtype=np.float32),
                    None,
                    None,
                )
            return (
                np.concatenate(
                    [
                        cache.sketch_matrix,
                        np.asarray(tail_source_page.runtime_page_sketch, dtype=np.float32)[None, ...],
                    ],
                    axis=0,
                ),
                None,
                None,
                None,
                None,
                None,
            )
        if cache.minima_matrix is None or cache.maxima_matrix is None:
            return None, None, None, None, None, None
        if tail_source_page is None:
            return None, cache.minima_matrix, cache.maxima_matrix, None, None, None
        if tail_source_page.runtime_page_min is None or tail_source_page.runtime_page_max is None:
            return None, None, None, None, None, None
        if score_all_pages_with_matrices:
            return (
                None,
                cache.minima_matrix,
                cache.maxima_matrix,
                None,
                np.asarray(tail_source_page.runtime_page_min, dtype=np.float32),
                np.asarray(tail_source_page.runtime_page_max, dtype=np.float32),
            )
        return (
            None,
            np.concatenate(
                [
                    cache.minima_matrix,
                    np.asarray(tail_source_page.runtime_page_min, dtype=np.float32)[None, :],
                ],
                axis=0,
            ),
            np.concatenate(
                [
                    cache.maxima_matrix,
                    np.asarray(tail_source_page.runtime_page_max, dtype=np.float32)[None, :],
                ],
                axis=0,
            ),
            None,
            None,
            None,
        )

    def decode_stage_summary(self) -> dict[str, object]:
        summary: dict[str, object] = self.decode_stage_runtime_totals()
        for stage in _DECODE_STAGE_TIMING_STAGES:
            summary[f"{_decode_stage_summary_key(stage)}_by_layer"] = {
                str(layer_id): float(layer_timings.get(stage, 0.0))
                for layer_id, layer_timings in sorted(self._decode_stage_timings_by_layer.items())
                if float(layer_timings.get(stage, 0.0)) != 0.0
            }
        return summary

    def execution_shortlist_summary(self) -> dict[str, object]:
        return {
            "execution_shortlist_invocations": int(self._execution_shortlist_invocations),
            "execution_shortlist_applied": int(self._execution_shortlist_applied),
            "execution_shortlist_group_union_applied": int(self._execution_shortlist_group_union_applied),
            "execution_shortlist_grouping_rejections": int(self._execution_shortlist_grouping_rejections),
            "execution_shortlist_total_pages": int(self._execution_shortlist_total_pages),
            "execution_shortlist_selected_pages": int(self._execution_shortlist_selected_pages),
            "execution_shortlist_invocations_by_layer": {
                str(layer_id): int(count)
                for layer_id, count in sorted(self._execution_shortlist_invocations_by_layer.items())
            },
            "execution_shortlist_applied_by_layer": {
                str(layer_id): int(count)
                for layer_id, count in sorted(self._execution_shortlist_applied_by_layer.items())
            },
            "execution_shortlist_group_union_applied_by_layer": {
                str(layer_id): int(count)
                for layer_id, count in sorted(self._execution_shortlist_group_union_applied_by_layer.items())
            },
            "execution_shortlist_grouping_rejection_reason_counts": dict(
                sorted(self._execution_shortlist_grouping_rejection_reason_counts.items())
            ),
            "execution_shortlist_grouping_rejection_reason_counts_by_layer": {
                str(layer_id): dict(sorted(counts.items()))
                for layer_id, counts in sorted(self._execution_shortlist_grouping_rejection_reason_counts_by_layer.items())
            },
            "execution_shortlist_grouping_rejections_by_layer": {
                str(layer_id): int(count)
                for layer_id, count in sorted(self._execution_shortlist_grouping_rejections_by_layer.items())
            },
            "execution_shortlist_total_pages_by_layer": {
                str(layer_id): int(count)
                for layer_id, count in sorted(self._execution_shortlist_total_pages_by_layer.items())
            },
            "execution_shortlist_selected_pages_by_layer": {
                str(layer_id): int(count)
                for layer_id, count in sorted(self._execution_shortlist_selected_pages_by_layer.items())
            },
            "execution_shortlist_trace_records": list(self._execution_shortlist_trace_records),
            "execution_exact_refine_invocations": int(self._execution_exact_refine_invocations),
            "execution_exact_refine_candidate_pages": int(self._execution_exact_refine_candidate_pages),
            "execution_exact_refine_selected_pages": int(self._execution_exact_refine_selected_pages),
            "execution_exact_refine_invocations_by_layer": {
                str(layer_id): int(count)
                for layer_id, count in sorted(self._execution_exact_refine_invocations_by_layer.items())
            },
            "execution_exact_refine_candidate_pages_by_layer": {
                str(layer_id): int(count)
                for layer_id, count in sorted(self._execution_exact_refine_candidate_pages_by_layer.items())
            },
            "execution_exact_refine_selected_pages_by_layer": {
                str(layer_id): int(count)
                for layer_id, count in sorted(self._execution_exact_refine_selected_pages_by_layer.items())
            },
        }

    def _execution_exact_refine_enabled(self, *, layer_id: int) -> bool:
        if self.config.execution_exact_refine_top_k <= 0:
            return False
        if not self.config.execution_exact_refine_layers:
            return False
        return int(layer_id) in {int(value) for value in self.config.execution_exact_refine_layers}

    def _execution_exact_promote_enabled(self, *, layer_id: int, context_length: int | None = None) -> bool:
        enabled, _ = self._execution_exact_promote_policy_status(layer_id=layer_id, context_length=context_length)
        return enabled

    def _execution_exact_promote_policy_status(
        self,
        *,
        layer_id: int,
        context_length: int | None = None,
    ) -> tuple[bool, str | None]:
        if self.config.execution_exact_promote_top_k <= 0:
            return False, "top_k_disabled"
        if not self.config.execution_exact_promote_layers:
            return False, "no_layers_configured"
        if int(layer_id) not in {int(value) for value in self.config.execution_exact_promote_layers}:
            return False, "layer_not_selected"
        if (
            int(self.config.execution_exact_promote_max_context) > 0
            and context_length is not None
            and int(context_length) > int(self.config.execution_exact_promote_max_context)
        ):
            return False, "context_exceeds_max_context"
        return True, None

    def _execution_exact_promote_status(
        self,
        *,
        layer_id: int,
        context_length: int | None = None,
        boundary_margin_normalized: float | None = None,
    ) -> tuple[bool, str | None]:
        enabled, reason = self._execution_exact_promote_policy_status(
            layer_id=layer_id,
            context_length=context_length,
        )
        if not enabled:
            return False, reason
        if (
            float(self.config.execution_exact_promote_min_margin_threshold) > 0.0
            and (
                boundary_margin_normalized is None
                or boundary_margin_normalized < float(self.config.execution_exact_promote_min_margin_threshold)
            )
        ):
            return False, "below_min_margin_threshold"
        if (
            float(self.config.execution_exact_promote_margin_threshold) > 0.0
            and boundary_margin_normalized is not None
            and boundary_margin_normalized > float(self.config.execution_exact_promote_margin_threshold)
        ):
            return False, "above_max_margin_threshold"
        return True, None

    def _execution_secondary_relevance_enabled(self, *, layer_id: int) -> bool:
        return self.config.execution_secondary_relevance_enabled_for_layer(layer_id=layer_id)

    def _execution_recent_neighbor_rescue_enabled(self, *, layer_id: int) -> bool:
        return self.config.execution_recent_neighbor_rescue_enabled_for_layer(layer_id=layer_id)

    def _execution_exact_promote_union_rescue_enabled(self, *, layer_id: int) -> bool:
        return (
            int(self.config.execution_exact_promote_union_rescue_top_k) > 0
            and int(layer_id) in {int(value) for value in self.config.execution_exact_promote_layers}
        )

    def _apply_execution_exact_promote_union_rescue(
        self,
        *,
        layer_id: int,
        selected_indices_by_group: Sequence[list[int] | None],
        key_pages_by_group: Sequence[Sequence[PageLike]],
        representative_queries: Sequence[np.ndarray],
        shortlist_traces_by_group: Sequence[dict[str, object] | None],
        trace: ExecutionTrace | None = None,
    ) -> tuple[list[list[int] | None], list[dict[str, object]]]:
        if not self._execution_exact_promote_union_rescue_enabled(layer_id=layer_id):
            return [None if indices is None else list(indices) for indices in selected_indices_by_group], []

        adjusted_indices_by_group = [None if indices is None else list(indices) for indices in selected_indices_by_group]
        baseline_selected_index_sets = [
            set() if indices is None else {int(index) for index in indices}
            for indices in selected_indices_by_group
        ]
        rescue_records: list[dict[str, object]] = []
        union_rescue_top_k = int(self.config.execution_exact_promote_union_rescue_top_k)
        eligible_group_records: list[dict[str, object]] = []

        for group_index, (indices, key_pages, query_slice, shortlist_trace) in enumerate(
            zip(
                adjusted_indices_by_group,
                key_pages_by_group,
                representative_queries,
                shortlist_traces_by_group,
                strict=True,
            )
        ):
            if indices is None:
                rescue_records.append(
                    {
                        "record_type": "union_rescue",
                        "layer_id": int(layer_id),
                        "group_index": int(group_index),
                        "applied": False,
                        "disable_reason": "no_shortlist",
                    }
                )
                continue
            if shortlist_trace is None:
                rescue_records.append(
                    {
                        "record_type": "union_rescue",
                        "layer_id": int(layer_id),
                        "group_index": int(group_index),
                        "applied": False,
                        "disable_reason": "missing_shortlist_trace",
                    }
                )
                continue
            context_length = shortlist_trace.get("context_length")
            policy_enabled, policy_disable_reason = self._execution_exact_promote_policy_status(
                layer_id=layer_id,
                context_length=None if context_length is None else int(context_length),
            )
            if not policy_enabled:
                rescue_records.append(
                    {
                        "record_type": "union_rescue",
                        "layer_id": int(layer_id),
                        "group_index": int(group_index),
                        "applied": False,
                        "disable_reason": policy_disable_reason,
                    }
                )
                continue

            base_indices = {int(index) for index in shortlist_trace.get("base_indices", [])}
            selected_index_set = baseline_selected_index_sets[group_index]

            other_selected_indices = set().union(
                *[
                    selected_index_set
                    for other_group_index, selected_index_set in enumerate(baseline_selected_index_sets)
                    if other_group_index != group_index
                ]
            )
            novel_candidate_indices = [
                int(index)
                for index in range(len(key_pages))
                if int(index) not in base_indices
                and int(index) not in selected_index_set
                and int(index) not in other_selected_indices
            ]
            if not novel_candidate_indices:
                rescue_records.append(
                    {
                        "record_type": "union_rescue",
                        "layer_id": int(layer_id),
                        "group_index": int(group_index),
                        "applied": False,
                        "disable_reason": "no_novel_candidates",
                        "base_count": int(len(base_indices)),
                        "selected_count": int(len(selected_index_set)),
                    }
                )
                continue

            eligible_group_records.append(
                {
                    "group_index": int(group_index),
                    "kv_head_id": shortlist_trace.get("kv_head_id"),
                    "indices": indices,
                    "key_pages": key_pages,
                    "query_slice": np.asarray(query_slice, dtype=np.float32),
                    "base_indices": base_indices,
                    "selected_index_set": selected_index_set,
                    "novel_candidate_indices": novel_candidate_indices,
                }
            )

        if not eligible_group_records:
            return adjusted_indices_by_group, rescue_records

        scored_union_candidates: list[tuple[float, int, int]] = []
        for eligible_group_record in eligible_group_records:
            novel_candidate_indices = list(eligible_group_record["novel_candidate_indices"])
            novel_candidate_logits = score_pages(
                eligible_group_record["query_slice"],
                [eligible_group_record["key_pages"][index] for index in novel_candidate_indices],
                backend=self.backend,
                trace=trace,
            )
            for index, logits in zip(novel_candidate_indices, novel_candidate_logits, strict=True):
                scored_union_candidates.append(
                    (
                        float(np.max(np.asarray(logits, dtype=np.float32))),
                        int(eligible_group_record["group_index"]),
                        int(index),
                    )
                )

        selected_by_group: dict[int, list[int]] = {}
        seen_indices: set[int] = set()
        for _, group_index, index in sorted(scored_union_candidates, key=lambda item: item[0], reverse=True):
            if int(index) in seen_indices:
                continue
            selected_by_group.setdefault(int(group_index), []).append(int(index))
            seen_indices.add(int(index))
            if len(seen_indices) >= union_rescue_top_k:
                break

        for eligible_group_record in eligible_group_records:
            group_index = int(eligible_group_record["group_index"])
            chosen_novel_indices = selected_by_group.get(group_index, [])
            if not chosen_novel_indices:
                rescue_records.append(
                    {
                        "record_type": "union_rescue",
                        "layer_id": int(layer_id),
                        "group_index": int(group_index),
                        "applied": False,
                        "disable_reason": "novel_candidates_not_selected",
                        "base_count": int(len(eligible_group_record["base_indices"])),
                        "selected_count": int(len(eligible_group_record["selected_index_set"])),
                        "novel_candidate_count": int(len(eligible_group_record["novel_candidate_indices"])),
                    }
                )
                continue

            group_indices = eligible_group_record["indices"]
            group_key_pages = eligible_group_record["key_pages"]
            adjusted_indices_by_group[group_index] = sorted(
                set(group_indices).union(chosen_novel_indices)
            )
            rescue_records.append(
                {
                    "record_type": "union_rescue",
                    "layer_id": int(layer_id),
                    "group_index": int(group_index),
                    "kv_head_id": eligible_group_record["kv_head_id"],
                    "applied": True,
                    "disable_reason": None,
                    "base_count": int(len(eligible_group_record["base_indices"])),
                    "selected_count": int(len(eligible_group_record["selected_index_set"])),
                    "novel_candidate_count": int(len(eligible_group_record["novel_candidate_indices"])),
                    "selected_novel_count": int(len(chosen_novel_indices)),
                    "selected_novel_indices": [int(index) for index in chosen_novel_indices],
                    "selected_novel_page_ranges": [
                        f"{int(index)}:{int(_page_header(group_key_pages[index]).token_start)}-{int(_page_header(group_key_pages[index]).token_start + _page_header(group_key_pages[index]).token_count)}"
                        for index in chosen_novel_indices
                    ],
                }
            )

        return adjusted_indices_by_group, rescue_records

    def _execution_shortlist_page_indices(
        self,
        key_pages: Sequence[PageLike],
        *,
        layer_id: int,
        kv_head_id: int | None = None,
        query_slice: np.ndarray,
        context_length_override: int | None = None,
        trace: ExecutionTrace | None = None,
    ) -> list[int] | None:
        capture_stage_timings = bool(trace is not None and trace.capture_timings)

        def _stage_start() -> float | None:
            return perf_counter() if capture_stage_timings else None

        def _stage_finish(stage: str, started_at: float | None) -> None:
            if started_at is None:
                return
            self._record_decode_stage_timing(
                layer_id=int(layer_id),
                stage=stage,
                ms=(perf_counter() - started_at) * 1000.0,
            )

        if self.config.execution_shortlist_disabled_for_layer(layer_id=layer_id):
            return None
        context_length = int(context_length_override) if context_length_override is not None else None
        if context_length is None and key_pages:
            context_length = max(
                int((page.source_page if isinstance(page, PreparedPageTorch) else page).header.token_start)
                + int((page.source_page if isinstance(page, PreparedPageTorch) else page).header.token_count)
                for page in key_pages
            )
        layer_recent_window = int(
            self.config.resolve_execution_recent_window_for_context(
                layer_id=layer_id,
                context_length=context_length,
            )
        )
        layer_relevance_top_k = int(
            self.config.resolve_execution_relevance_top_k_for_context(
                layer_id=layer_id,
                context_length=context_length,
            )
        )
        if (
            layer_recent_window <= 0
            and self.config.execution_sink_window <= 0
            and layer_relevance_top_k <= 0
        ):
            return None
        promote_candidate_expansion_enabled, promote_candidate_expansion_disable_reason = (
            self._execution_exact_promote_policy_status(
                layer_id=layer_id,
                context_length=context_length,
            )
        )

        def _page_range_labels(indices: Sequence[int]) -> list[str]:
            labels: list[str] = []
            for index in indices:
                header = _page_header(key_pages[int(index)])
                labels.append(
                    f"{int(index)}:{int(header.token_start)}-{int(header.token_start + header.token_count)}"
                )
            return labels

        def _record_shortlist_trace(
            *,
            base_index_set: set[int],
            stage1_selected_indices: Sequence[int],
            final_selected_indices: Sequence[int],
            promote_enabled: bool,
            promote_disable_reason: str | None,
            promote_candidate_indices: Sequence[int] | None = None,
            promote_selected_indices: Sequence[int] | None = None,
            promote_target_old_page_count: int | None = None,
        ) -> None:
            stage1_old_indices = [int(index) for index in stage1_selected_indices if int(index) not in base_index_set]
            final_old_indices = [int(index) for index in final_selected_indices if int(index) not in base_index_set]
            promote_candidate_indices = (
                [] if promote_candidate_indices is None else [int(index) for index in promote_candidate_indices]
            )
            promote_selected_indices = (
                [] if promote_selected_indices is None else [int(index) for index in promote_selected_indices]
            )
            self._execution_shortlist_trace_records.append(
                {
                    "record_type": "shortlist_group",
                    "layer_id": int(layer_id),
                    "kv_head_id": None if kv_head_id is None else int(kv_head_id),
                    "context_length": None if context_length is None else int(context_length),
                    "layer_recent_window": int(layer_recent_window),
                    "layer_relevance_top_k": int(layer_relevance_top_k),
                    "candidate_relevance_top_k": int(candidate_relevance_top_k),
                    "base_count": int(len(base_index_set)),
                    "stage1_count": int(len(stage1_selected_indices)),
                    "final_count": int(len(final_selected_indices)),
                    "stage1_old_count": int(len(stage1_old_indices)),
                    "final_old_count": int(len(final_old_indices)),
                    "base_indices": [int(index) for index in sorted(base_index_set)],
                    "stage1_indices": [int(index) for index in stage1_selected_indices],
                    "final_indices": [int(index) for index in final_selected_indices],
                    "base_page_ranges": _page_range_labels(sorted(base_index_set)),
                    "stage1_old_page_ranges": _page_range_labels(stage1_old_indices),
                    "final_old_page_ranges": _page_range_labels(final_old_indices),
                    "exact_promote_candidate_expansion_enabled": bool(promote_candidate_expansion_enabled),
                    "exact_promote_candidate_expansion_disable_reason": promote_candidate_expansion_disable_reason,
                    "exact_promote_enabled": bool(promote_enabled),
                    "exact_promote_disable_reason": promote_disable_reason,
                    "promote_candidate_count": int(len(promote_candidate_indices)),
                    "promote_selected_count": int(len(promote_selected_indices)),
                    "promote_target_old_page_count": (
                        None if promote_target_old_page_count is None else int(promote_target_old_page_count)
                    ),
                    "promote_candidate_indices": [int(index) for index in promote_candidate_indices],
                    "promote_selected_indices": [int(index) for index in promote_selected_indices],
                    "promote_candidate_page_ranges": _page_range_labels(promote_candidate_indices),
                    "promote_selected_page_ranges": _page_range_labels(promote_selected_indices),
                    "boundary_margin_normalized": (
                        None if boundary_margin_normalized is None else float(boundary_margin_normalized)
                    ),
                }
            )
        key_page_sketches: list[np.ndarray] = []
        key_page_minima: list[np.ndarray] = []
        key_page_maxima: list[np.ndarray] = []
        for page in key_pages:
            source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
            sketch = source_page.runtime_page_sketch
            page_min = source_page.runtime_page_min
            page_max = source_page.runtime_page_max
            if self.config.execution_relevance_mode == "sketch":
                if sketch is None:
                    return None
                key_page_sketches.append(np.asarray(sketch, dtype=np.float32))
            else:
                if page_min is None or page_max is None:
                    return None
                key_page_minima.append(np.asarray(page_min, dtype=np.float32))
                key_page_maxima.append(np.asarray(page_max, dtype=np.float32))
        candidate_relevance_top_k = int(layer_relevance_top_k)
        if promote_candidate_expansion_enabled:
            candidate_relevance_top_k = max(
                candidate_relevance_top_k,
                int(layer_relevance_top_k) + int(self.config.execution_exact_promote_top_k) * 2,
            )
        if self._execution_exact_refine_enabled(layer_id=layer_id):
            candidate_relevance_top_k = max(
                candidate_relevance_top_k,
                int(self.config.execution_exact_refine_top_k) * 2,
            )
        base_window_started_at = _stage_start()
        base_indices = set(
            select_window_page_indices(
                key_pages,
                recent_window_tokens=layer_recent_window if layer_recent_window > 0 else None,
                sink_window_tokens=int(self.config.execution_sink_window),
            )
        )
        _stage_finish("shortlist_base_window", base_window_started_at)
        use_recent_old_bonus = self.config.execution_recent_old_bonus_enabled_for_layer(layer_id=layer_id)
        use_secondary_relevance_rescue = self._execution_secondary_relevance_enabled(layer_id=layer_id)
        use_recent_neighbor_rescue = self._execution_recent_neighbor_rescue_enabled(layer_id=layer_id)
        use_confidence_gated_exact_promote = (
            promote_candidate_expansion_enabled
            and float(self.config.execution_exact_promote_margin_threshold) > 0.0
        )
        boundary_margin_normalized = None
        shortlist_candidate_scoring_started_at = _stage_start()
        if (
            use_recent_old_bonus
            or use_secondary_relevance_rescue
            or use_recent_neighbor_rescue
            or use_confidence_gated_exact_promote
        ):
            if candidate_relevance_top_k > 0:
                candidate_indices = [index for index in range(len(key_pages)) if index not in base_indices]
                if candidate_indices:
                    approx_scores: list[float] = []
                    shortlist_candidate_approx_scoring_started_at = _stage_start()
                    for index in candidate_indices:
                        approx_score = _score_page_relevance_for_mode(
                            np.asarray(query_slice, dtype=np.float32),
                            key_pages[index],
                            relevance_mode=self.config.execution_relevance_mode,
                        )
                        if approx_score is None:
                            return None
                        approx_scores.append(float(approx_score))
                    _stage_finish("shortlist_candidate_approx_scoring", shortlist_candidate_approx_scoring_started_at)
                    shortlist_candidate_ranking_started_at = _stage_start()
                    score_scale = max(float(np.std(np.asarray(approx_scores, dtype=np.float32))), 1e-6)
                    recent_start = int(context_length) - int(layer_recent_window) if layer_recent_window > 0 and context_length is not None else int(context_length or 0)
                    adjusted_scores = [
                        score
                        + (
                            float(self.config.execution_recent_old_bonus_strength)
                            * score_scale
                            * _recent_old_bonus_weight(
                                key_pages[index],
                                recent_start=int(recent_start),
                                bonus_window=int(self.config.execution_recent_old_bonus_window),
                            )
                        )
                        for index, score in zip(candidate_indices, approx_scores, strict=True)
                    ]
                    if len(adjusted_scores) > candidate_relevance_top_k and candidate_relevance_top_k > 0:
                        sorted_scores = sorted(adjusted_scores, reverse=True)
                        boundary_margin_normalized = float(
                            (sorted_scores[candidate_relevance_top_k - 1] - sorted_scores[candidate_relevance_top_k])
                            / max(float(np.std(np.asarray(adjusted_scores, dtype=np.float32))), 1e-6)
                        )
                    ranked_candidates = [
                        index
                        for _, index in sorted(
                            zip(adjusted_scores, candidate_indices, strict=True),
                            key=lambda item: item[0],
                            reverse=True,
                        )
                    ]
                    stage1_ranked_candidates = ranked_candidates[:candidate_relevance_top_k]
                    stage1_indices = sorted(base_indices.union(stage1_ranked_candidates))
                    _stage_finish("shortlist_candidate_ranking", shortlist_candidate_ranking_started_at)
                    if use_recent_neighbor_rescue and layer_recent_window > 0 and context_length is not None:
                        shortlist_candidate_neighbor_rescue_started_at = _stage_start()
                        recent_start = int(context_length) - int(layer_recent_window)
                        primary_top_indices = stage1_ranked_candidates[:layer_relevance_top_k]
                        anchor_pages = [
                            index
                            for index in primary_top_indices
                            if int(_page_header(key_pages[index]).token_start + _page_header(key_pages[index]).token_count)
                            <= int(self.config.execution_recent_neighbor_rescue_anchor_window)
                        ]
                        recent_old_indices = [
                            index
                            for index in primary_top_indices
                            if (
                                int(_page_header(key_pages[index]).token_start + _page_header(key_pages[index]).token_count)
                                <= int(recent_start)
                                and int(_page_header(key_pages[index]).token_start + _page_header(key_pages[index]).token_count)
                                > int(recent_start - layer_recent_window)
                            )
                        ]
                        if (
                            len(anchor_pages) >= int(self.config.execution_recent_neighbor_rescue_min_anchor_pages)
                            and recent_old_indices
                        ):
                            rescue_indices: list[int] = []
                            probe_index = min(recent_old_indices) - 1
                            stage1_index_set = set(stage1_indices)
                            while probe_index >= 0 and len(rescue_indices) < int(self.config.execution_recent_neighbor_rescue_top_k):
                                page_end = int(
                                    _page_header(key_pages[probe_index]).token_start
                                    + _page_header(key_pages[probe_index]).token_count
                                )
                                if page_end <= int(recent_start - layer_recent_window):
                                    break
                                if probe_index not in base_indices and probe_index not in stage1_index_set:
                                    rescue_indices.append(int(probe_index))
                                probe_index -= 1
                            if rescue_indices:
                                stage1_indices = sorted(stage1_index_set.union(rescue_indices))
                        _stage_finish("shortlist_candidate_neighbor_rescue", shortlist_candidate_neighbor_rescue_started_at)
                    if use_secondary_relevance_rescue and layer_relevance_top_k > 0:
                        shortlist_candidate_secondary_scoring_started_at = _stage_start()
                        secondary_scores: list[float] = []
                        for index in candidate_indices:
                            secondary_score = _score_page_relevance_for_mode(
                                np.asarray(query_slice, dtype=np.float32),
                                key_pages[index],
                                relevance_mode=self.config.execution_secondary_relevance_mode,
                            )
                            if secondary_score is None:
                                secondary_scores = []
                                break
                            secondary_scores.append(float(secondary_score))
                        if secondary_scores:
                            secondary_ranked_candidates = [
                                index
                                for _, index in sorted(
                                    zip(secondary_scores, candidate_indices, strict=True),
                                    key=lambda item: item[0],
                                    reverse=True,
                                )
                            ]
                            primary_top_indices = stage1_ranked_candidates[:layer_relevance_top_k]
                            secondary_top_indices = secondary_ranked_candidates[:layer_relevance_top_k]
                            overlap_budget = min(len(primary_top_indices), len(secondary_top_indices))
                            overlap_ratio = (
                                float(len(set(primary_top_indices) & set(secondary_top_indices))) / float(overlap_budget)
                                if overlap_budget > 0
                                else 1.0
                            )
                            if overlap_ratio < float(self.config.execution_secondary_relevance_min_overlap):
                                rescue_indices: list[int] = []
                                for index in secondary_ranked_candidates:
                                    if index in stage1_indices:
                                        continue
                                    rescue_indices.append(int(index))
                                    if len(rescue_indices) >= int(self.config.execution_secondary_relevance_top_k):
                                        break
                                if rescue_indices:
                                    stage1_indices = sorted(set(stage1_indices).union(rescue_indices))
                        _stage_finish("shortlist_candidate_secondary_scoring", shortlist_candidate_secondary_scoring_started_at)
                else:
                    stage1_indices = sorted(base_indices)
            else:
                stage1_indices = sorted(base_indices)
        else:
            builtin_sketch_matrix = None
            builtin_minima_matrix = None
            builtin_maxima_matrix = None
            builtin_tail_sketch = None
            builtin_tail_minimum = None
            builtin_tail_maximum = None
            builtin_matrix_prepare_started_at = _stage_start()
            if self.config.execution_builtin_selector_cache:
                (
                    builtin_sketch_matrix,
                    builtin_minima_matrix,
                    builtin_maxima_matrix,
                    builtin_tail_sketch,
                    builtin_tail_minimum,
                    builtin_tail_maximum,
                ) = self._execution_builtin_selector_matrices(
                    layer_id=int(layer_id),
                    kv_head_id=kv_head_id,
                    key_pages=key_pages,
                    relevance_mode=self.config.execution_relevance_mode,
                    score_all_pages_with_matrices=(
                        self.config.execution_builtin_selector_score_all_pages
                        and not self.config.execution_builtin_selector_candidate_only
                    ),
                )
            if (
                builtin_sketch_matrix is not None
                or builtin_minima_matrix is not None
                or builtin_maxima_matrix is not None
            ):
                _stage_finish("shortlist_candidate_builtin_sidecar_stack", builtin_matrix_prepare_started_at)
            shortlist_candidate_builtin_selection_started_at = _stage_start()
            stage1_indices = select_execution_page_indices(
                key_pages,
                recent_window_tokens=layer_recent_window if layer_recent_window > 0 else None,
                sink_window_tokens=int(self.config.execution_sink_window),
                query_slice=np.asarray(query_slice, dtype=np.float32),
                key_page_sketches=key_page_sketches if key_page_sketches else None,
                key_page_sketch_matrix=builtin_sketch_matrix,
                tail_page_sketch=builtin_tail_sketch,
                key_page_minima=key_page_minima if key_page_minima else None,
                key_page_minima_matrix=builtin_minima_matrix,
                tail_page_minimum=builtin_tail_minimum,
                key_page_maxima=key_page_maxima if key_page_maxima else None,
                key_page_maxima_matrix=builtin_maxima_matrix,
                tail_page_maximum=builtin_tail_maximum,
                relevance_top_k=candidate_relevance_top_k,
                relevance_mode=self.config.execution_relevance_mode,
                score_all_pages_with_matrices=(
                    self.config.execution_builtin_selector_score_all_pages
                    and not self.config.execution_builtin_selector_candidate_only
                ),
                score_all_pages_min_candidate_fraction=self.config.execution_builtin_selector_score_all_pages_min_candidate_fraction,
                selector_stats_recorder=lambda stats: self._record_builtin_selector_stats(
                    candidate_pages=int(stats["candidate_pages"]),
                    total_pages=int(stats["total_pages"]),
                    candidate_fraction=float(stats["candidate_fraction"]),
                    used_score_all_pages=bool(stats["used_score_all_pages"]),
                ),
                stage_recorder=lambda stage, ms: self._record_decode_stage_timing(
                    layer_id=int(layer_id),
                    stage=stage,
                    ms=float(ms),
                ),
            )
            _stage_finish("shortlist_candidate_builtin_selection", shortlist_candidate_builtin_selection_started_at)
        _stage_finish("shortlist_candidate_scoring", shortlist_candidate_scoring_started_at)
        if not self._execution_exact_refine_enabled(layer_id=layer_id):
            promote_enabled, promote_disable_reason = self._execution_exact_promote_status(
                layer_id=layer_id,
                context_length=context_length,
                boundary_margin_normalized=boundary_margin_normalized,
            )
            if not promote_enabled:
                _record_shortlist_trace(
                    base_index_set=base_indices,
                    stage1_selected_indices=stage1_indices,
                    final_selected_indices=stage1_indices,
                    promote_enabled=False,
                    promote_disable_reason=promote_disable_reason,
                )
                return stage1_indices
            candidate_indices = [index for index in stage1_indices if index not in base_indices]
            if not candidate_indices:
                _record_shortlist_trace(
                    base_index_set=base_indices,
                    stage1_selected_indices=stage1_indices,
                    final_selected_indices=stage1_indices,
                    promote_enabled=True,
                    promote_disable_reason=None,
                    promote_candidate_indices=[],
                    promote_selected_indices=[],
                    promote_target_old_page_count=0,
                )
                return stage1_indices
            target_old_page_count = min(
                len(candidate_indices),
                int(layer_relevance_top_k) + int(self.config.execution_exact_promote_top_k),
            )
            if target_old_page_count >= len(candidate_indices):
                _record_shortlist_trace(
                    base_index_set=base_indices,
                    stage1_selected_indices=stage1_indices,
                    final_selected_indices=stage1_indices,
                    promote_enabled=True,
                    promote_disable_reason=None,
                    promote_candidate_indices=candidate_indices,
                    promote_selected_indices=candidate_indices,
                    promote_target_old_page_count=target_old_page_count,
                )
                return stage1_indices
            shortlist_exact_selection_started_at = _stage_start()
            candidate_logits = score_pages(
                np.asarray(query_slice, dtype=np.float32),
                [key_pages[index] for index in candidate_indices],
                backend=self.backend,
                trace=trace,
            )
            chosen = [
                index
                for _, index in sorted(
                    (
                        (float(np.max(np.asarray(logits, dtype=np.float32))), index)
                        for index, logits in zip(candidate_indices, candidate_logits, strict=True)
                    ),
                    key=lambda item: item[0],
                    reverse=True,
                )[:target_old_page_count]
            ]
            final_indices = sorted(base_indices.union(chosen))
            _record_shortlist_trace(
                base_index_set=base_indices,
                stage1_selected_indices=stage1_indices,
                final_selected_indices=final_indices,
                promote_enabled=True,
                promote_disable_reason=None,
                promote_candidate_indices=candidate_indices,
                promote_selected_indices=chosen,
                promote_target_old_page_count=target_old_page_count,
            )
            _stage_finish("shortlist_exact_selection", shortlist_exact_selection_started_at)
            return final_indices
        base_indices = set(
            select_window_page_indices(
                key_pages,
                recent_window_tokens=layer_recent_window if layer_recent_window > 0 else None,
                sink_window_tokens=int(self.config.execution_sink_window),
            )
        )
        candidate_indices = [index for index in stage1_indices if index not in base_indices]
        if not candidate_indices:
            self._record_execution_exact_refine(layer_id=layer_id, candidate_pages=0, selected_pages=0)
            _record_shortlist_trace(
                base_index_set=base_indices,
                stage1_selected_indices=stage1_indices,
                final_selected_indices=stage1_indices,
                promote_enabled=False,
                promote_disable_reason="exact_refine_enabled",
            )
            return stage1_indices
        top_k = min(int(self.config.execution_exact_refine_top_k), len(candidate_indices))
        if top_k >= len(candidate_indices):
            self._record_execution_exact_refine(
                layer_id=layer_id,
                candidate_pages=len(candidate_indices),
                selected_pages=len(candidate_indices),
            )
            _record_shortlist_trace(
                base_index_set=base_indices,
                stage1_selected_indices=stage1_indices,
                final_selected_indices=stage1_indices,
                promote_enabled=False,
                promote_disable_reason="exact_refine_enabled",
            )
            return stage1_indices
        shortlist_exact_selection_started_at = _stage_start()
        candidate_logits = score_pages(
            np.asarray(query_slice, dtype=np.float32),
            [key_pages[index] for index in candidate_indices],
            backend=self.backend,
            trace=trace,
        )
        chosen = [
            index
            for _, index in sorted(
                (
                    (float(np.max(np.asarray(logits, dtype=np.float32))), index)
                    for index, logits in zip(candidate_indices, candidate_logits, strict=True)
                ),
                key=lambda item: item[0],
                reverse=True,
            )[:top_k]
        ]
        self._record_execution_exact_refine(
            layer_id=layer_id,
            candidate_pages=len(candidate_indices),
            selected_pages=len(chosen),
        )
        final_indices = sorted(base_indices.union(chosen))
        _record_shortlist_trace(
            base_index_set=base_indices,
            stage1_selected_indices=stage1_indices,
            final_selected_indices=final_indices,
            promote_enabled=False,
            promote_disable_reason="exact_refine_enabled",
        )
        _stage_finish("shortlist_exact_selection", shortlist_exact_selection_started_at)
        return final_indices

    def _should_build_execution_runtime_metadata(self, *, kind: str) -> bool:
        if kind != "K":
            return False
        return self.config.execution_shortlist_enabled()

    def clear(self) -> None:
        for state in self._states.values():
            state.clear(clear_prepared_cache=False)
        self.cache.clear()
        clear_prepared_chunk_cache()
        self._m2_prefilter_invocations = 0
        self._m2_prefilter_candidate_pages = 0
        self._m2_prefilter_selected_pages = 0
        self._decode_path_counts = {
            "grouped_batched": 0,
            "per_kv_fallback": 0,
        }
        self._decode_path_counts_by_layer = {}
        self._execution_shortlist_invocations = 0
        self._execution_shortlist_applied = 0
        self._execution_shortlist_group_union_applied = 0
        self._execution_shortlist_grouping_rejections = 0
        self._execution_shortlist_grouping_rejection_reason_counts = {}
        self._execution_shortlist_grouping_rejection_reason_counts_by_layer = {}
        self._execution_shortlist_total_pages = 0
        self._execution_shortlist_selected_pages = 0
        self._execution_shortlist_invocations_by_layer = {}
        self._execution_shortlist_applied_by_layer = {}
        self._execution_shortlist_group_union_applied_by_layer = {}
        self._execution_shortlist_grouping_rejections_by_layer = {}
        self._execution_shortlist_total_pages_by_layer = {}
        self._execution_shortlist_selected_pages_by_layer = {}
        self._execution_shortlist_trace_records = []
        self._execution_exact_refine_invocations = 0
        self._execution_exact_refine_candidate_pages = 0
        self._execution_exact_refine_selected_pages = 0
        self._execution_exact_refine_invocations_by_layer = {}
        self._execution_exact_refine_candidate_pages_by_layer = {}
        self._execution_exact_refine_selected_pages_by_layer = {}
        self._decode_grouped_batch_rejection_reason_counts = {}
        self._decode_grouped_batch_rejection_reason_counts_by_layer = {}
        self._decode_stage_timings = _empty_decode_stage_timing_totals()
        self._decode_stage_timings_by_layer = {}
        self._reset_resident_accounting()
        self._reset_chunk_budget_tracking()
        self._reset_builtin_selector_tracking()
        self._reset_execution_value_escape_tracking()
        self._prepared_chunk_cache_frozen_budget_bytes = None
        self._prepared_chunk_cache_applied_budget_bytes = None
        self._prepared_chunk_cache_budget_dirty = True

    def clear_layer(self, layer_id: int) -> None:
        self._validate_layer_id(layer_id)
        layer_keys = [key for key in self._states if key[0] == layer_id]
        if not layer_keys:
            return
        for key in layer_keys:
            self._states[key].clear(clear_prepared_cache=False)
        self.cache.clear()
        clear_prepared_chunk_cache()
        self._rebuild_resident_accounting()
        self._decode_path_counts_by_layer.pop(int(layer_id), None)
        self._decode_path_counts = {
            "grouped_batched": 0,
            "per_kv_fallback": 0,
        }
        for counts in self._decode_path_counts_by_layer.values():
            for path_name, count in counts.items():
                if path_name in self._decode_path_counts:
                    self._decode_path_counts[path_name] += int(count)
        self._execution_shortlist_invocations_by_layer.pop(int(layer_id), None)
        self._execution_shortlist_applied_by_layer.pop(int(layer_id), None)
        self._execution_shortlist_group_union_applied_by_layer.pop(int(layer_id), None)
        self._execution_shortlist_grouping_rejections_by_layer.pop(int(layer_id), None)
        self._execution_shortlist_grouping_rejection_reason_counts_by_layer.pop(int(layer_id), None)
        self._execution_shortlist_total_pages_by_layer.pop(int(layer_id), None)
        self._execution_shortlist_selected_pages_by_layer.pop(int(layer_id), None)
        self._execution_exact_refine_invocations_by_layer.pop(int(layer_id), None)
        self._execution_exact_refine_candidate_pages_by_layer.pop(int(layer_id), None)
        self._execution_exact_refine_selected_pages_by_layer.pop(int(layer_id), None)
        self._decode_grouped_batch_rejection_reason_counts_by_layer.pop(int(layer_id), None)
        self._prepared_chunk_cache_frozen_budget_bytes = None
        self._prepared_chunk_cache_applied_budget_bytes = None
        self._prepared_chunk_cache_budget_dirty = True
        self._decode_stage_timings_by_layer.pop(int(layer_id), None)
        self._execution_shortlist_invocations = sum(self._execution_shortlist_invocations_by_layer.values())
        self._execution_shortlist_applied = sum(self._execution_shortlist_applied_by_layer.values())
        self._execution_shortlist_group_union_applied = sum(self._execution_shortlist_group_union_applied_by_layer.values())
        self._execution_shortlist_grouping_rejections = sum(self._execution_shortlist_grouping_rejections_by_layer.values())
        self._execution_shortlist_grouping_rejection_reason_counts = {}
        for layer_reason_counts in self._execution_shortlist_grouping_rejection_reason_counts_by_layer.values():
            for reason, count in layer_reason_counts.items():
                self._execution_shortlist_grouping_rejection_reason_counts[reason] = (
                    self._execution_shortlist_grouping_rejection_reason_counts.get(reason, 0) + int(count)
                )
        self._execution_shortlist_total_pages = sum(self._execution_shortlist_total_pages_by_layer.values())
        self._execution_shortlist_selected_pages = sum(self._execution_shortlist_selected_pages_by_layer.values())
        self._execution_exact_refine_invocations = sum(self._execution_exact_refine_invocations_by_layer.values())
        self._execution_exact_refine_candidate_pages = sum(self._execution_exact_refine_candidate_pages_by_layer.values())
        self._execution_exact_refine_selected_pages = sum(self._execution_exact_refine_selected_pages_by_layer.values())
        self._decode_grouped_batch_rejection_reason_counts = {}
        for layer_reason_counts in self._decode_grouped_batch_rejection_reason_counts_by_layer.values():
            for reason, count in layer_reason_counts.items():
                self._decode_grouped_batch_rejection_reason_counts[reason] = (
                    self._decode_grouped_batch_rejection_reason_counts.get(reason, 0) + int(count)
                )
        self._decode_stage_timings = _empty_decode_stage_timing_totals()
        for layer_timings in self._decode_stage_timings_by_layer.values():
            for stage, value in layer_timings.items():
                if stage in self._decode_stage_timings:
                    self._decode_stage_timings[stage] += float(value)
        self._reset_builtin_selector_tracking()
        self._reset_execution_value_escape_tracking()
        self._prepared_chunk_cache_frozen_budget_bytes = None
        self._prepared_chunk_cache_budget_dirty = True

    def _grouped_query_heads_for_mapping(self, q_head_to_kv_head: Sequence[int] | np.ndarray) -> tuple[tuple[int, ...], ...]:
        mapping = np.asarray(q_head_to_kv_head, dtype=np.int64)
        if mapping.shape != (self.num_attention_heads,):
            raise ValueError("q_head_to_kv_head must have shape [num_attention_heads]")
        if np.array_equal(mapping, self.default_q_head_to_kv_head):
            return self.default_grouped_query_heads
        return _group_query_heads(mapping, num_key_value_heads=self.num_key_value_heads)

    def _encode_full_prefill_pages(
        self,
        layer_id: int,
        keys: np.ndarray,
        values: np.ndarray,
        *,
        token_start: int,
        sequence_length: int,
        full_tokens: int,
    ) -> tuple[list[list[EncodedPage]], list[list[EncodedPage]]]:
        key_pages_by_head: list[list[EncodedPage]] = [[] for _ in range(self.num_key_value_heads)]
        value_pages_by_head: list[list[EncodedPage]] = [[] for _ in range(self.num_key_value_heads)]
        if full_tokens <= 0:
            return key_pages_by_head, value_pages_by_head

        page_size = self.config.tokens_per_page
        full_page_count = full_tokens // page_size
        build_key_sidecar = (
            self.config.m2_prefilter_top_k > 0
            and full_page_count >= int(self.config.m2_prefilter_min_pages)
        )
        full_keys = np.ascontiguousarray(keys[:, :full_tokens], dtype=np.float32)
        full_values = np.ascontiguousarray(values[:, :full_tokens], dtype=np.float32)
        shared_m4_basis_by_head: list[np.ndarray | None] = [None] * self.num_key_value_heads
        for kv_head_id in range(self.num_key_value_heads):
            resolved_basis = self.config.resolve_m4_project_basis_k(layer_id=layer_id)
            if resolved_basis != "svd_shared":
                continue
            key_page_mode = self._select_page_mode(
                full_keys[kv_head_id, :page_size],
                kind="K",
                layer_id=layer_id,
                kv_head_id=kv_head_id,
                token_start=token_start,
                sequence_length=sequence_length,
                stage="prefill",
            )
            key_mode_name = (
                key_page_mode.mode
                if key_page_mode is not None
                else self.config.resolve_page_mode(kind="K", layer_id=layer_id, kv_head_id=kv_head_id)
            )
            if key_mode_name != "M4":
                continue
            shared_m4_basis_by_head[kv_head_id] = fit_shared_project_basis(
                full_keys[kv_head_id, :full_tokens],
                group_size=self.config.group_size,
                project_dim=self.config.resolve_m4_project_dim_k(layer_id=layer_id),
                page_size=page_size,
            ).astype(np.float16, copy=False)

        for page_start in range(0, full_tokens, page_size):
            page_end = page_start + page_size
            absolute_page_start = int(token_start + page_start)
            for kv_head_id in range(self.num_key_value_heads):
                key_page_mode = self._select_page_mode(
                    full_keys[kv_head_id, page_start:page_end],
                    kind="K",
                    layer_id=layer_id,
                    kv_head_id=kv_head_id,
                    token_start=absolute_page_start,
                    sequence_length=sequence_length,
                    stage="prefill",
                )
                key_pages_by_head[kv_head_id].append(
                    encode_page(
                        full_keys[kv_head_id, page_start:page_end],
                        self.config,
                        kind="K",
                        layer_id=layer_id,
                        kv_head_id=kv_head_id,
                        token_start=absolute_page_start,
                        page_mode=key_page_mode,
                        build_runtime_metadata=self._should_build_execution_runtime_metadata(kind="K"),
                        build_m2_sidecar=build_key_sidecar,
                        m4_basis_override=(
                            shared_m4_basis_by_head[kv_head_id]
                            if (
                                (
                                    key_page_mode.mode
                                    if key_page_mode is not None
                                    else self.config.resolve_page_mode(kind="K", layer_id=layer_id, kv_head_id=kv_head_id)
                                )
                                == "M4"
                                and shared_m4_basis_by_head[kv_head_id] is not None
                            )
                            else None
                        ),
                    )
                )
                dense_value_page = full_values[kv_head_id, page_start:page_end]
                value_page = encode_page(
                    dense_value_page,
                    self.config,
                    kind="V",
                    layer_id=layer_id,
                    kv_head_id=kv_head_id,
                    token_start=absolute_page_start,
                    page_mode=self._select_page_mode(
                        dense_value_page,
                        kind="V",
                        layer_id=layer_id,
                        kv_head_id=kv_head_id,
                        token_start=absolute_page_start,
                        sequence_length=sequence_length,
                        stage="prefill",
                    ),
                    build_runtime_metadata=False,
                )
                self._maybe_register_execution_value_escape_source(
                    value_page,
                    dense_values=dense_value_page,
                    escape_mode=str(self.config.execution_value_escape_mode),
                )
                value_pages_by_head[kv_head_id].append(value_page)
        return key_pages_by_head, value_pages_by_head

    def _can_direct_prepare_full_prefill_pages_torch(self) -> bool:
        if not self._use_persistent_torch_tail:
            return False
        if self.config.learned_page_selector_enabled():
            return False
        if int(self.config.m2_prefilter_top_k) > 0:
            return False
        if self.config.has_mode_overrides() or self.config.has_policy_overrides():
            return False
        if self.config.default_mode_k != "M0" or self.config.default_mode_v != "M0":
            return False
        if self.config.quant_scheme_k != "affine" or self.config.quant_scheme_v != "affine":
            return False
        if self.config.payload_layout_k != "group_major" or self.config.payload_layout_v != "group_major":
            return False
        return True

    def _select_page_mode(
        self,
        values: np.ndarray,
        *,
        kind: str,
        layer_id: int,
        kv_head_id: int,
        token_start: int,
        sequence_length: int,
        stage: str = "unknown",
    ) -> PageModeSpec | None:
        learned_page_mode = self._select_page_mode_with_learned_selector(
            values,
            kind=kind,
            layer_id=layer_id,
            kv_head_id=kv_head_id,
            token_start=token_start,
            sequence_length=sequence_length,
            stage=stage,
        )
        if learned_page_mode is not None:
            return learned_page_mode
        if not self.config.has_policy_overrides(kind=kind) and not self.config.has_mode_overrides(kind=kind):
            return None
        layer_policy = self.config.resolve_layer_policy(kind=kind, layer_id=layer_id, kv_head_id=kv_head_id)
        page_stats = observe_page(values)
        token_age = max(0, int(sequence_length) - int(token_start) - 1)
        return choose_page_mode(
            int(layer_id),
            kind,
            token_age,
            page_stats,
            layer_policy=layer_policy,
        )

    def _select_page_mode_with_learned_selector(
        self,
        values: np.ndarray,
        *,
        kind: str,
        layer_id: int,
        kv_head_id: int,
        token_start: int,
        sequence_length: int,
        stage: str,
    ) -> PageModeSpec | None:
        model = self._learned_page_selector_model
        if model is None:
            return None
        if not self.config.learned_page_selector_applies_to_kind(kind=str(kind)):
            return None
        stage_name = str(stage)
        started_at = perf_counter()
        page_stats = observe_page(values)
        row = {
            "stage": stage_name,
            "kind": str(kind),
            "prompt_family": self.config.learned_page_selector_prompt_family,
            "prompt_variant": self.config.learned_page_selector_prompt_variant,
            "query_present": bool(stage_name == "decode"),
            "layer_fraction": float(int(layer_id) / max(self.num_hidden_layers - 1, 1)),
            "kv_head_fraction": float(int(kv_head_id) / max(self.num_key_value_heads - 1, 1)),
            "token_start": int(token_start),
            "token_age": max(0, int(sequence_length) - int(token_start) - int(values.shape[0])),
            "token_count": int(values.shape[0]),
            "head_dim": int(values.shape[1]),
            "safe_candidate_count": 0.0,
            "trace_rms": float(page_stats.rms),
            "trace_abs_max": float(page_stats.abs_max),
            "trace_channel_range_mean": float(page_stats.channel_range_mean),
            "trace_outlier_fraction": float(page_stats.outlier_fraction),
            "age_per_token": float(max(0, int(sequence_length) - int(token_start) - int(values.shape[0])) / max(int(values.shape[0]), 1)),
        }
        predicted = model.predict_row(row)
        elapsed_ms = float((perf_counter() - started_at) * 1000.0)
        self._learned_page_selector_ms_total += elapsed_ms
        self._learned_page_selector_invocations += 1
        self._learned_page_selector_invocations_by_stage[stage_name] = (
            self._learned_page_selector_invocations_by_stage.get(stage_name, 0) + 1
        )
        self._learned_page_selector_ms_total_by_stage[stage_name] = (
            float(self._learned_page_selector_ms_total_by_stage.get(stage_name, 0.0)) + elapsed_ms
        )
        if predicted is None:
            self._learned_page_selector_fallbacks += 1
            self._learned_page_selector_fallbacks_by_stage[stage_name] = (
                self._learned_page_selector_fallbacks_by_stage.get(stage_name, 0) + 1
            )
            return None
        try:
            page_mode = parse_page_mode_token(predicted)
        except ValueError:
            self._learned_page_selector_fallbacks += 1
            self._learned_page_selector_fallbacks_by_stage[stage_name] = (
                self._learned_page_selector_fallbacks_by_stage.get(stage_name, 0) + 1
            )
            return None
        token = f"{page_mode.mode}/{page_mode.quant_scheme}/{page_mode.bits}" + (
            "" if page_mode.escape_dtype is None else f"/{page_mode.escape_dtype}"
        )
        self._learned_page_selector_predictions[token] = self._learned_page_selector_predictions.get(token, 0) + 1
        stage_predictions = self._learned_page_selector_predictions_by_stage.setdefault(stage_name, {})
        stage_predictions[token] = stage_predictions.get(token, 0) + 1
        return page_mode

    def prepare_static_pages(self, *, trace: ExecutionTrace | None = None) -> None:
        if self._torch_device_type is None:
            return
        key_refs: list[tuple[_HeadSessionState, int]] = []
        key_pages: list[PageLike] = []
        value_refs: list[tuple[_HeadSessionState, int]] = []
        value_pages: list[PageLike] = []
        touched_states: dict[int, _HeadSessionState] = {}

        for state in self._states.values():
            for index, page in enumerate(state.session.key_pages):
                if isinstance(page, PreparedPageTorch):
                    continue
                key_refs.append((state, index))
                key_pages.append(page)
                touched_states[id(state)] = state
            for index, page in enumerate(state.session.value_pages):
                if isinstance(page, PreparedPageTorch):
                    continue
                value_refs.append((state, index))
                value_pages.append(page)
                touched_states[id(state)] = state

        if key_pages:
            prepared_keys = prepare_pages(key_pages, backend=self.backend, cache=self.cache, trace=trace)
            for (state, index), prepared in zip(key_refs, prepared_keys, strict=True):
                state.session.key_pages[index] = prepared
                state.invalidate_decode_views()
        if value_pages:
            prepared_values = prepare_pages(value_pages, backend=self.backend, cache=self.cache, trace=trace)
            for (state, index), prepared in zip(value_refs, prepared_values, strict=True):
                state.session.value_pages[index] = prepared
                state.invalidate_decode_views()
        if key_pages or value_pages:
            for state in touched_states.values():
                self._refresh_state_resident_accounting(state)
            self._mark_prepared_chunk_cache_budget_dirty(reason="prepare_static_pages")
        for state in self._states.values():
            self._maybe_prewarm_execution_builtin_selector_cache(state)
            self._maybe_prewarm_execution_value_escape_pages(state, trace=trace)

    def _ensure_prepared_static_pages(
        self,
        state: _HeadSessionState,
        *,
        trace: ExecutionTrace | None = None,
    ) -> None:
        if self._torch_device_type is None:
            return
        state_changed = False
        if state.session.key_pages and not all(isinstance(page, PreparedPageTorch) for page in state.session.key_pages):
            state.session.key_pages = prepare_pages(
                state.session.key_pages,
                backend=self.backend,
                cache=self.cache,
                trace=trace,
            )
            state.invalidate_decode_views()
            state_changed = True
        if state.session.value_pages and not all(isinstance(page, PreparedPageTorch) for page in state.session.value_pages):
            state.session.value_pages = prepare_pages(
                state.session.value_pages,
                backend=self.backend,
                cache=self.cache,
                trace=trace,
            )
            state.invalidate_decode_views()
            state_changed = True
        if state_changed:
            self._refresh_state_resident_accounting(state)
            self._mark_prepared_chunk_cache_budget_dirty(reason="ensure_prepared_static_pages")
        self._maybe_prewarm_execution_builtin_selector_cache(state)
        self._maybe_prewarm_execution_value_escape_pages(state, trace=trace)

    def _validate_layer_id(self, layer_id: int) -> None:
        if layer_id < 0 or layer_id >= self.num_hidden_layers:
            raise ValueError(f"layer_id must be in [0, {self.num_hidden_layers})")

    def _state(self, layer_id: int, kv_head_id: int) -> _HeadSessionState:
        self._validate_layer_id(layer_id)
        if kv_head_id < 0 or kv_head_id >= self.num_key_value_heads:
            raise ValueError(f"kv_head_id must be in [0, {self.num_key_value_heads})")
        key = (layer_id, kv_head_id)
        state = self._states.get(key)
        if state is None:
            torch_device_type = self._torch_device_type
            state = _HeadSessionState(
                session=PagedDecodeSession(backend=self.backend, cache=self.cache),
                tail=_TailPageBuilder(
                    self.config,
                    layer_id=layer_id,
                    kv_head_id=kv_head_id,
                    select_page_mode=self._select_page_mode,
                ),
                persistent_key_tail=_PersistentTailPage(
                    self.config,
                    layer_id=layer_id,
                    kv_head_id=kv_head_id,
                    kind="K",
                    device_type=torch_device_type,
                )
                if torch_device_type is not None
                else None,
                persistent_value_tail=_PersistentTailPage(
                    self.config,
                    layer_id=layer_id,
                    kv_head_id=kv_head_id,
                    kind="V",
                    device_type=torch_device_type,
                )
                if torch_device_type is not None
                else None,
            )
            self._states[key] = state
        return state

    @property
    def _torch_device_type(self) -> str | None:
        if self.backend == "torch_mps":
            return "mps" if mps_available() else None
        if self.backend == "torch_cuda":
            return "cuda" if cuda_available() else None
        if self.backend == "auto":
            if cuda_available():
                return "cuda"
            if mps_available():
                return "mps"
        return None

    @property
    def _use_persistent_torch_tail(self) -> bool:
        return self._torch_device_type is not None

    def layer_sequence_length(self, layer_id: int) -> int:
        self._validate_layer_id(layer_id)
        lengths = {self._state(layer_id, kv_head_id).sequence_length for kv_head_id in range(self.num_key_value_heads)}
        if len(lengths) > 1:
            raise RuntimeError(f"layer {layer_id} KV heads disagree on sequence length")
        return next(iter(lengths), 0)

    def _m2_prefilter_pages_numpy(
        self,
        queries: np.ndarray,
        key_pages: Sequence[PageLike],
        value_pages: Sequence[PageLike],
    ) -> tuple[list[PageLike], list[PageLike]]:
        top_k = int(self.config.m2_prefilter_top_k)
        if top_k <= 0 or len(key_pages) <= top_k:
            return list(key_pages), list(value_pages)

        always_keep: list[int] = []
        candidate_indices: list[int] = []
        candidate_scores: list[float] = []
        for page_index, page in enumerate(key_pages):
            if not _page_has_m2_sidecar(page):
                always_keep.append(page_index)
                continue
            candidate_indices.append(page_index)
        candidate_pages = [key_pages[index] for index in candidate_indices]
        if (
            len(candidate_indices) <= top_k
            or len(candidate_indices) < int(self.config.m2_prefilter_min_pages)
        ):
            return list(key_pages), list(value_pages)
        if _pages_can_batch_m2_prefilter(candidate_pages):
            score_array = _page_m2_prefilter_scores_numpy(queries, candidate_pages)
        else:
            for page in candidate_pages:
                candidate_scores.append(_page_m2_prefilter_score_numpy(queries, page))
            score_array = np.asarray(candidate_scores, dtype=np.float32)
        selected_order = np.argpartition(score_array, -top_k)[-top_k:]
        selected_indices = sorted(always_keep + [candidate_indices[index] for index in selected_order.tolist()])
        self._m2_prefilter_invocations += 1
        self._m2_prefilter_candidate_pages += len(candidate_indices)
        self._m2_prefilter_selected_pages += len(selected_indices)
        return [key_pages[index] for index in selected_indices], [value_pages[index] for index in selected_indices]

    def _m2_prefilter_pages_torch(
        self,
        queries,
        key_pages: Sequence[PageLike],
        value_pages: Sequence[PageLike],
    ) -> tuple[list[PageLike], list[PageLike]]:
        top_k = int(self.config.m2_prefilter_top_k)
        if top_k <= 0 or len(key_pages) <= top_k:
            return list(key_pages), list(value_pages)

        always_keep: list[int] = []
        candidate_indices: list[int] = []
        candidate_scores: list[float] = []
        for page_index, page in enumerate(key_pages):
            if not _page_has_m2_sidecar(page):
                always_keep.append(page_index)
                continue
            candidate_indices.append(page_index)
        candidate_pages = [key_pages[index] for index in candidate_indices]
        if (
            len(candidate_indices) <= top_k
            or len(candidate_indices) < int(self.config.m2_prefilter_min_pages)
        ):
            return list(key_pages), list(value_pages)
        if _pages_can_batch_m2_prefilter(candidate_pages):
            score_array = _page_m2_prefilter_scores_torch(queries, candidate_pages)
        else:
            for page in candidate_pages:
                candidate_scores.append(_page_m2_prefilter_score_torch(queries, page))
            score_array = np.asarray(candidate_scores, dtype=np.float32)
        selected_order = np.argpartition(score_array, -top_k)[-top_k:]
        selected_indices = sorted(always_keep + [candidate_indices[index] for index in selected_order.tolist()])
        self._m2_prefilter_invocations += 1
        self._m2_prefilter_candidate_pages += len(candidate_indices)
        self._m2_prefilter_selected_pages += len(selected_indices)
        return [key_pages[index] for index in selected_indices], [value_pages[index] for index in selected_indices]

    def page_mode_summary(self) -> dict[str, object]:
        counts: dict[str, int] = {
            "total_static_pages": 0,
            "m0_pages": 0,
            "m1_pages": 0,
            "m2_pages": 0,
            "m4_pages": 0,
            "m3_pages": 0,
            "m2_sidecar_pages": 0,
            "requested_m1_pages": 0,
            "m1_fallback_pages": 0,
            "active_tail_pages": 0,
            "k_total_static_pages": 0,
            "v_total_static_pages": 0,
            "k_m0_pages": 0,
            "k_m1_pages": 0,
            "k_m2_pages": 0,
            "k_m4_pages": 0,
            "k_m3_pages": 0,
            "k_m2_sidecar_pages": 0,
            "v_m0_pages": 0,
            "v_m1_pages": 0,
            "v_m2_pages": 0,
            "v_m4_pages": 0,
            "v_m3_pages": 0,
            "v_m2_sidecar_pages": 0,
            "k_requested_m1_pages": 0,
            "v_requested_m1_pages": 0,
            "k_m1_fallback_pages": 0,
            "v_m1_fallback_pages": 0,
        }
        m1_trial_errors: list[float] = []
        m1_trial_token_p95_errors: list[float] = []
        k_m1_trial_errors: list[float] = []
        k_m1_trial_token_p95_errors: list[float] = []
        v_m1_trial_errors: list[float] = []
        v_m1_trial_token_p95_errors: list[float] = []
        policy_tier_counts: dict[str, int] = {}
        fallback_reason_counts: dict[str, int] = {}
        signature_counts: dict[str, int] = {}
        layer_kind_mode_counts: dict[str, int] = {}

        def visit_page(page: PageLike) -> None:
            source = page.source_page if isinstance(page, PreparedPageTorch) else page
            counts["total_static_pages"] += 1
            kind_prefix = str(source.header.kind).lower()
            counts[f"{kind_prefix}_total_static_pages"] += 1
            mode_name = str(source.header.mode_default)
            key = f"{mode_name.lower()}_pages"
            if key in counts:
                counts[key] += 1
            kind_key = f"{kind_prefix}_{mode_name.lower()}_pages"
            if kind_key in counts:
                counts[kind_key] += 1
            policy_tier_counts[source.header.sensitivity_tier] = policy_tier_counts.get(source.header.sensitivity_tier, 0) + 1
            fallback_key = source.header.fallback_reason or "none"
            fallback_reason_counts[fallback_key] = fallback_reason_counts.get(fallback_key, 0) + 1
            signature = f"{source.header.kind}:{source.header.mode_default}:{source.header.quant_scheme}:{source.header.bits}"
            if source.header.mode_default == "M3":
                signature = f"{signature}:{source.header.escape_dtype}"
            signature_counts[signature] = signature_counts.get(signature, 0) + 1
            layer_mode_key = f"layer:{source.header.layer_id}:{source.header.kind}:{source.header.mode_default}:{source.header.bits}"
            if source.header.mode_default == "M3":
                layer_mode_key = f"{layer_mode_key}:{source.header.escape_dtype}"
            layer_kind_mode_counts[layer_mode_key] = layer_kind_mode_counts.get(layer_mode_key, 0) + 1
            if source.m2_sketch is not None and source.m2_basis is not None and source.header.mode_default != "M2":
                counts["m2_sidecar_pages"] += 1
                counts[f"{kind_prefix}_m2_sidecar_pages"] += 1
            if source.requested_mode == "M1":
                counts["requested_m1_pages"] += 1
                counts[f"{kind_prefix}_requested_m1_pages"] += 1
                if source.header.mode_default != "M1":
                    counts["m1_fallback_pages"] += 1
                    counts[f"{kind_prefix}_m1_fallback_pages"] += 1
                if source.trial_quant_error is not None:
                    error_value = float(source.trial_quant_error)
                    m1_trial_errors.append(error_value)
                    if kind_prefix == "k":
                        k_m1_trial_errors.append(error_value)
                    else:
                        v_m1_trial_errors.append(error_value)
                if source.trial_token_p95_error is not None:
                    token_error_value = float(source.trial_token_p95_error)
                    m1_trial_token_p95_errors.append(token_error_value)
                    if kind_prefix == "k":
                        k_m1_trial_token_p95_errors.append(token_error_value)
                    else:
                        v_m1_trial_token_p95_errors.append(token_error_value)

        for state in self._states.values():
            for page in state.session.key_pages:
                visit_page(page)
            for page in state.session.value_pages:
                visit_page(page)
            if state.tail.token_count > 0:
                counts["active_tail_pages"] += 2

        summary: dict[str, float | int] = dict(counts)
        summary["policy_tier_counts"] = dict(sorted(policy_tier_counts.items()))
        summary["fallback_reason_counts"] = dict(sorted(fallback_reason_counts.items()))
        summary["mode_signature_counts"] = dict(sorted(signature_counts.items()))
        summary["layer_kind_mode_counts"] = dict(sorted(layer_kind_mode_counts.items()))
        total_buckets = len(signature_counts)
        single_page_buckets = sum(1 for count in signature_counts.values() if count == 1)
        total_pages = int(counts["total_static_pages"])
        summary["fragmentation_total_buckets"] = total_buckets
        summary["fragmentation_single_page_buckets"] = single_page_buckets
        summary["fragmentation_avg_pages_per_bucket"] = (
            float(total_pages / total_buckets) if total_buckets > 0 else 0.0
        )
        summary["fragmentation_single_page_bucket_fraction"] = (
            float(single_page_buckets / total_buckets) if total_buckets > 0 else 0.0
        )
        if m1_trial_errors:
            errors = np.asarray(m1_trial_errors, dtype=np.float32)
            summary["m1_trial_error_mean"] = float(np.mean(errors))
            summary["m1_trial_error_max"] = float(np.max(errors))
            summary["m1_trial_error_p95"] = float(np.percentile(errors, 95))
        else:
            summary["m1_trial_error_mean"] = 0.0
            summary["m1_trial_error_max"] = 0.0
            summary["m1_trial_error_p95"] = 0.0
        if m1_trial_token_p95_errors:
            errors = np.asarray(m1_trial_token_p95_errors, dtype=np.float32)
            summary["m1_trial_token_p95_error_mean"] = float(np.mean(errors))
            summary["m1_trial_token_p95_error_max"] = float(np.max(errors))
            summary["m1_trial_token_p95_error_p95"] = float(np.percentile(errors, 95))
        else:
            summary["m1_trial_token_p95_error_mean"] = 0.0
            summary["m1_trial_token_p95_error_max"] = 0.0
            summary["m1_trial_token_p95_error_p95"] = 0.0
        for prefix, error_values in (("k", k_m1_trial_errors), ("v", v_m1_trial_errors)):
            if error_values:
                errors = np.asarray(error_values, dtype=np.float32)
                summary[f"{prefix}_m1_trial_error_mean"] = float(np.mean(errors))
                summary[f"{prefix}_m1_trial_error_max"] = float(np.max(errors))
                summary[f"{prefix}_m1_trial_error_p95"] = float(np.percentile(errors, 95))
            else:
                summary[f"{prefix}_m1_trial_error_mean"] = 0.0
                summary[f"{prefix}_m1_trial_error_max"] = 0.0
                summary[f"{prefix}_m1_trial_error_p95"] = 0.0
        for prefix, error_values in (("k", k_m1_trial_token_p95_errors), ("v", v_m1_trial_token_p95_errors)):
            if error_values:
                errors = np.asarray(error_values, dtype=np.float32)
                summary[f"{prefix}_m1_trial_token_p95_error_mean"] = float(np.mean(errors))
                summary[f"{prefix}_m1_trial_token_p95_error_max"] = float(np.max(errors))
                summary[f"{prefix}_m1_trial_token_p95_error_p95"] = float(np.percentile(errors, 95))
            else:
                summary[f"{prefix}_m1_trial_token_p95_error_mean"] = 0.0
                summary[f"{prefix}_m1_trial_token_p95_error_max"] = 0.0
                summary[f"{prefix}_m1_trial_token_p95_error_p95"] = 0.0
        summary["m2_prefilter_top_k"] = int(self.config.m2_prefilter_top_k)
        summary["m2_prefilter_min_pages"] = int(self.config.m2_prefilter_min_pages)
        summary["m2_prefilter_invocations"] = int(self._m2_prefilter_invocations)
        summary["m2_prefilter_candidate_pages"] = int(self._m2_prefilter_candidate_pages)
        summary["m2_prefilter_selected_pages"] = int(self._m2_prefilter_selected_pages)
        summary["learned_page_selector_enabled"] = bool(self._learned_page_selector_model is not None)
        summary["learned_page_selector_path"] = (
            None
            if self.config.learned_page_selector_path is None
            else str(self.config.learned_page_selector_path)
        )
        summary["learned_page_selector_prompt_family"] = (
            None
            if self.config.learned_page_selector_prompt_family is None
            else str(self.config.learned_page_selector_prompt_family)
        )
        summary["learned_page_selector_prompt_variant"] = (
            None
            if self.config.learned_page_selector_prompt_variant is None
            else str(self.config.learned_page_selector_prompt_variant)
        )
        summary["learned_page_selector_profile"] = str(self.config.learned_page_selector_profile)
        summary["learned_page_selector_scope"] = str(self.config.learned_page_selector_scope)
        summary["learned_page_selector_target_candidate"] = str(self.config.learned_page_selector_target_candidate)
        summary["learned_page_selector_logit_offset"] = float(self.config.learned_page_selector_logit_offset)
        summary["learned_page_selector_invocations"] = int(self._learned_page_selector_invocations)
        summary["learned_page_selector_fallbacks"] = int(self._learned_page_selector_fallbacks)
        summary["learned_page_selector_ms_total"] = float(self._learned_page_selector_ms_total)
        summary["learned_page_selector_invocations_by_stage"] = {
            stage: int(count)
            for stage, count in sorted(self._learned_page_selector_invocations_by_stage.items())
        }
        summary["learned_page_selector_fallbacks_by_stage"] = {
            stage: int(count)
            for stage, count in sorted(self._learned_page_selector_fallbacks_by_stage.items())
        }
        summary["learned_page_selector_ms_total_by_stage"] = {
            stage: float(ms)
            for stage, ms in sorted(self._learned_page_selector_ms_total_by_stage.items())
        }
        summary["learned_page_selector_prediction_counts"] = {
            token: int(count)
            for token, count in sorted(self._learned_page_selector_predictions.items())
        }
        summary["learned_page_selector_prediction_counts_by_stage"] = {
            stage: {token: int(count) for token, count in sorted(stage_counts.items())}
            for stage, stage_counts in sorted(self._learned_page_selector_predictions_by_stage.items())
        }
        return summary

    def _batch_upload_persistent_tail_rows(
        self,
        tails: Sequence[_PersistentTailPage | None],
        rows_by_head: np.ndarray,
        *,
        token_start: int,
        trace: ExecutionTrace | None = None,
    ) -> None:
        active_pairs = [(tail, rows_by_head[index]) for index, tail in enumerate(tails) if tail is not None]
        if not active_pairs:
            return
        non_empty_pairs = [(tail, rows) for tail, rows in active_pairs if rows.shape[0] > 0]
        if not non_empty_pairs:
            return
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required for the persistent torch tail path") from exc
        contiguous_rows = np.ascontiguousarray(np.stack([rows.astype(np.float16, copy=False) for _, rows in non_empty_pairs], axis=0))
        device_rows = torch.from_numpy(contiguous_rows).to(device=non_empty_pairs[0][0].device_type)
        if trace is not None:
            trace.record_host_to_device(int(device_rows.numel() * device_rows.element_size()))
        for batch_index, (tail, rows) in enumerate(non_empty_pairs):
            if tail is None:
                continue
            if tail.host_buffer is None or tail.prepared_page is None:
                tail._ensure_allocated(token_start=token_start if tail.token_count == 0 else tail.source_page.header.token_start)
            tail.append_rows_from_device(
                rows=rows,
                device_rows=device_rows[batch_index],
                token_start=token_start,
            )

    def _batch_append_persistent_tail_tensors(
        self,
        tails: Sequence[_PersistentTailPage | None],
        rows_by_head,
        *,
        token_start: int,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required for the persistent torch tail path") from exc
        if not torch.is_tensor(rows_by_head):
            raise TypeError("rows_by_head must be a torch.Tensor")
        if rows_by_head.ndim != 3:
            raise ValueError("rows_by_head must have shape [kv_heads, token_count, head_dim]")
        for index, tail in enumerate(tails):
            if tail is None or int(rows_by_head[index].shape[0]) == 0:
                continue
            if tail.prepared_page is None:
                tail._ensure_allocated(token_start=token_start if tail.token_count == 0 else tail.source_page.header.token_start)
            tail.append_device_rows(rows_by_head[index], token_start=token_start)

    def ingest_prefill_cache(
        self,
        layer_id: int,
        layer_k: np.ndarray,
        layer_v: np.ndarray,
        *,
        context_length: int | None = None,
        trace: ExecutionTrace | None = None,
    ) -> None:
        keys = _normalize_prefill_tensor(
            layer_k,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.config.head_dim,
            name="layer_k",
        )
        values = _normalize_prefill_tensor(
            layer_v,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.config.head_dim,
            name="layer_v",
        )
        if keys.shape[1] != values.shape[1]:
            raise ValueError("layer_k and layer_v sequence lengths must match")

        seq_len = int(keys.shape[1])
        absolute_context_length = seq_len if context_length is None else int(context_length)
        if absolute_context_length < seq_len:
            raise ValueError("context_length must be at least the stored prefill cache length")
        token_offset = absolute_context_length - seq_len
        full_page_count = seq_len // self.config.tokens_per_page
        full_tokens = full_page_count * self.config.tokens_per_page
        preload_key_pages_by_head, preload_value_pages_by_head = self._encode_full_prefill_pages(
            layer_id,
            keys,
            values,
            token_start=token_offset,
            sequence_length=absolute_context_length,
            full_tokens=full_tokens,
        )
        for kv_head_id in range(self.num_key_value_heads):
            state = self._state(layer_id, kv_head_id)
            state.clear(clear_prepared_cache=False)
            preload_key_pages = preload_key_pages_by_head[kv_head_id]
            preload_value_pages = preload_value_pages_by_head[kv_head_id]
            if preload_key_pages:
                state.session.append(preload_key_pages, preload_value_pages, prepare=False, trace=trace)
                state.invalidate_decode_views()
            remainder_keys = keys[kv_head_id, full_tokens:]
            remainder_values = values[kv_head_id, full_tokens:]
            state.tail.load_prefill_remainder(
                remainder_keys,
                remainder_values,
                token_start=token_offset + full_tokens,
            )
            state.sequence_length = absolute_context_length
        if self._use_persistent_torch_tail:
            key_tails = [self._state(layer_id, kv_head_id).persistent_key_tail for kv_head_id in range(self.num_key_value_heads)]
            value_tails = [self._state(layer_id, kv_head_id).persistent_value_tail for kv_head_id in range(self.num_key_value_heads)]
            for kv_head_id in range(self.num_key_value_heads):
                tail = key_tails[kv_head_id]
                if tail is not None:
                    tail.clear()
                    if keys[kv_head_id, full_tokens:].shape[0] > 0:
                        tail._ensure_allocated(token_start=token_offset + full_tokens)
                tail = value_tails[kv_head_id]
                if tail is not None:
                    tail.clear()
                    if values[kv_head_id, full_tokens:].shape[0] > 0:
                        tail._ensure_allocated(token_start=token_offset + full_tokens)
            self._batch_upload_persistent_tail_rows(
                key_tails,
                keys[:, full_tokens:],
                token_start=token_offset + full_tokens,
                trace=trace,
            )
            self._batch_upload_persistent_tail_rows(
                value_tails,
                values[:, full_tokens:],
                token_start=token_offset + full_tokens,
                trace=trace,
            )
        self._rebuild_resident_accounting()
        self._mark_prepared_chunk_cache_budget_dirty(reason="ingest_prefill_cache")

    def ingest_prefill_cache_torch(
        self,
        layer_id: int,
        layer_k,
        layer_v,
        *,
        context_length: int | None = None,
        trace: ExecutionTrace | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required for ingest_prefill_cache_torch") from exc
        if self._torch_device_type is None:
            raise RuntimeError("ingest_prefill_cache_torch is only available for a torch accelerator backend")
        keys = _normalize_prefill_tensor_torch(
            layer_k,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.config.head_dim,
            name="layer_k",
        )
        values = _normalize_prefill_tensor_torch(
            layer_v,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.config.head_dim,
            name="layer_v",
        )
        if tuple(keys.shape) != tuple(values.shape):
            raise ValueError("layer_k and layer_v sequence lengths must match")

        seq_len = int(keys.shape[1])
        absolute_context_length = seq_len if context_length is None else int(context_length)
        if absolute_context_length < seq_len:
            raise ValueError("context_length must be at least the stored prefill cache length")
        token_offset = absolute_context_length - seq_len
        full_page_count = seq_len // self.config.tokens_per_page
        full_tokens = full_page_count * self.config.tokens_per_page
        direct_prepare_full_pages = (
            self._can_direct_prepare_full_prefill_pages_torch()
            and full_tokens > 0
            and full_tokens == seq_len
        )
        if direct_prepare_full_pages:
            page_size = int(self.config.tokens_per_page)
            full_key_pages_by_head = [
                prepare_m0_affine_pages_from_tensor_torch(
                    keys[kv_head_id, :full_tokens].reshape(full_page_count, page_size, self.config.head_dim),
                    config=self.config,
                    kind="K",
                    layer_id=layer_id,
                    kv_head_id=kv_head_id,
                    token_start=token_offset,
                    device_type=self._torch_device_type,
                    build_runtime_metadata=self._should_build_execution_runtime_metadata(kind="K"),
                )
                for kv_head_id in range(self.num_key_value_heads)
            ]
            full_value_pages_by_head = [
                prepare_m0_affine_pages_from_tensor_torch(
                    values[kv_head_id, :full_tokens].reshape(full_page_count, page_size, self.config.head_dim),
                    config=self.config,
                    kind="V",
                    layer_id=layer_id,
                    kv_head_id=kv_head_id,
                    token_start=token_offset,
                    device_type=self._torch_device_type,
                )
                for kv_head_id in range(self.num_key_value_heads)
            ]
        elif full_tokens > 0:
            full_keys_cpu = keys[:, :full_tokens].detach().cpu().numpy()
            full_values_cpu = values[:, :full_tokens].detach().cpu().numpy()
            preload_key_pages_by_head, preload_value_pages_by_head = self._encode_full_prefill_pages(
                layer_id,
                full_keys_cpu,
                full_values_cpu,
                token_start=token_offset,
                sequence_length=absolute_context_length,
                full_tokens=full_tokens,
            )
        else:
            preload_key_pages_by_head = [[] for _ in range(self.num_key_value_heads)]
            preload_value_pages_by_head = [[] for _ in range(self.num_key_value_heads)]
        if direct_prepare_full_pages:
            preload_key_pages_by_head = full_key_pages_by_head
            preload_value_pages_by_head = full_value_pages_by_head
        if not self._use_persistent_torch_tail:
            remainder_keys_cpu = keys[:, full_tokens:].detach().cpu().numpy()
            remainder_values_cpu = values[:, full_tokens:].detach().cpu().numpy()

        for kv_head_id in range(self.num_key_value_heads):
            state = self._state(layer_id, kv_head_id)
            state.clear(clear_prepared_cache=False)
            preload_key_pages = preload_key_pages_by_head[kv_head_id]
            preload_value_pages = preload_value_pages_by_head[kv_head_id]
            if preload_key_pages:
                state.session.append(preload_key_pages, preload_value_pages, prepare=False, trace=trace)
                state.invalidate_decode_views()
            if self._use_persistent_torch_tail:
                state.tail.clear()
            else:
                remainder_keys = remainder_keys_cpu[kv_head_id]
                remainder_values = remainder_values_cpu[kv_head_id]
                state.tail.load_prefill_remainder(
                    remainder_keys,
                    remainder_values,
                    token_start=token_offset + full_tokens,
                )
            state.sequence_length = absolute_context_length

        if self._use_persistent_torch_tail:
            key_tails = [self._state(layer_id, kv_head_id).persistent_key_tail for kv_head_id in range(self.num_key_value_heads)]
            value_tails = [self._state(layer_id, kv_head_id).persistent_value_tail for kv_head_id in range(self.num_key_value_heads)]
            for kv_head_id in range(self.num_key_value_heads):
                tail = key_tails[kv_head_id]
                if tail is not None:
                    tail.clear()
                    if int(keys[kv_head_id, full_tokens:].shape[0]) > 0:
                        tail._ensure_allocated(token_start=token_offset + full_tokens)
                tail = value_tails[kv_head_id]
                if tail is not None:
                    tail.clear()
                    if int(values[kv_head_id, full_tokens:].shape[0]) > 0:
                        tail._ensure_allocated(token_start=token_offset + full_tokens)
            self._batch_append_persistent_tail_tensors(
                key_tails,
                keys[:, full_tokens:],
                token_start=token_offset + full_tokens,
            )
            self._batch_append_persistent_tail_tensors(
                value_tails,
                values[:, full_tokens:],
                token_start=token_offset + full_tokens,
            )
        self._rebuild_resident_accounting()
        self._mark_prepared_chunk_cache_budget_dirty(reason="ingest_prefill_cache_torch")

    def append_step(
        self,
        layer_id: int,
        key_step: np.ndarray,
        value_step: np.ndarray,
        token_index: int,
        *,
        trace: ExecutionTrace | None = None,
    ) -> None:
        keys = _normalize_step_tensor(
            key_step,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.config.head_dim,
            name="key_step",
        )
        values = _normalize_step_tensor(
            value_step,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.config.head_dim,
            name="value_step",
        )
        if keys.shape[1] != values.shape[1]:
            raise ValueError("key_step and value_step token counts must match")
        token_count = int(keys.shape[1])

        resident_bytes_changed = False
        if self._use_persistent_torch_tail:
            key_tails = [self._state(layer_id, kv_head_id).persistent_key_tail for kv_head_id in range(self.num_key_value_heads)]
            value_tails = [self._state(layer_id, kv_head_id).persistent_value_tail for kv_head_id in range(self.num_key_value_heads)]
            for kv_head_id in range(self.num_key_value_heads):
                key_tail = key_tails[kv_head_id]
                value_tail = value_tails[kv_head_id]
                if key_tail is not None:
                    resident_bytes_changed = (
                        key_tail._ensure_allocated(
                            token_start=token_index if key_tail.token_count == 0 else key_tail.source_page.header.token_start
                        )
                        or resident_bytes_changed
                    )
                if value_tail is not None:
                    resident_bytes_changed = (
                        value_tail._ensure_allocated(
                            token_start=token_index if value_tail.token_count == 0 else value_tail.source_page.header.token_start
                        )
                        or resident_bytes_changed
                    )
            self._batch_upload_persistent_tail_rows(key_tails, keys, token_start=token_index, trace=trace)
            self._batch_upload_persistent_tail_rows(value_tails, values, token_start=token_index, trace=trace)

        for kv_head_id in range(self.num_key_value_heads):
            state = self._state(layer_id, kv_head_id)
            if state.sequence_length != token_index:
                raise ValueError(
                    f"layer {layer_id} kv_head {kv_head_id} expected token_index {state.sequence_length}, received {token_index}"
                )
            finalized_key_pages, finalized_value_pages = state.tail.append_step_rows(
                keys[kv_head_id],
                values[kv_head_id],
                token_start=token_index,
                sequence_length=token_index + token_count,
            )
            if finalized_key_pages:
                state.session.append(finalized_key_pages, finalized_value_pages, trace=trace)
                state.invalidate_decode_views()
                if state.persistent_key_tail is not None and state.persistent_value_tail is not None:
                    state.persistent_key_tail.clear()
                    state.persistent_value_tail.clear()
            state.sequence_length += token_count
        if resident_bytes_changed:
            for kv_head_id in range(self.num_key_value_heads):
                self._refresh_state_resident_accounting(self._state(layer_id, kv_head_id))
            self._mark_prepared_chunk_cache_budget_dirty(reason="append_step_tail_alloc")
        return

    def append_step_torch(
        self,
        layer_id: int,
        key_step,
        value_step,
        token_index: int,
        *,
        trace: ExecutionTrace | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required for append_step_torch") from exc
        if not torch.is_tensor(key_step) or not torch.is_tensor(value_step):
            raise TypeError("append_step_torch requires torch.Tensor inputs")
        if self._torch_device_type is None:
            raise RuntimeError("append_step_torch is only available for a torch accelerator backend")

        keys = key_step.detach().to(dtype=torch.float32)
        values = value_step.detach().to(dtype=torch.float32)
        if keys.ndim == 4:
            if int(keys.shape[0]) != 1:
                raise ValueError("key_step batch dimension must be 1 for the Phase 5 Llama path")
            keys = keys[0]
        if values.ndim == 4:
            if int(values.shape[0]) != 1:
                raise ValueError("value_step batch dimension must be 1 for the Phase 5 Llama path")
            values = values[0]
        if keys.ndim != 3 or values.ndim != 3:
            raise ValueError("key_step and value_step must have shape [kv_heads, token_count, head_dim]")
        if int(keys.shape[0]) != self.num_key_value_heads or int(values.shape[0]) != self.num_key_value_heads:
            raise ValueError(f"append steps must contain {self.num_key_value_heads} KV heads")
        if int(keys.shape[2]) != self.config.head_dim or int(values.shape[2]) != self.config.head_dim:
            raise ValueError(f"append steps head_dim must equal {self.config.head_dim}")
        if tuple(keys.shape) != tuple(values.shape):
            raise ValueError("key_step and value_step token counts must match")
        token_count = int(keys.shape[1])

        if not self._use_persistent_torch_tail:
            self.append_step(
                layer_id,
                keys.cpu().numpy(),
                values.cpu().numpy(),
                token_index,
                trace=trace,
            )
            return

        key_tails = [self._state(layer_id, kv_head_id).persistent_key_tail for kv_head_id in range(self.num_key_value_heads)]
        value_tails = [self._state(layer_id, kv_head_id).persistent_value_tail for kv_head_id in range(self.num_key_value_heads)]
        resident_bytes_changed = False
        for kv_head_id in range(self.num_key_value_heads):
            state = self._state(layer_id, kv_head_id)
            if state.sequence_length != token_index:
                raise ValueError(
                    f"layer {layer_id} kv_head {kv_head_id} expected token_index {state.sequence_length}, received {token_index}"
                )
            if key_tails[kv_head_id] is not None:
                resident_bytes_changed = (
                    key_tails[kv_head_id]._ensure_allocated(
                        token_start=token_index
                        if key_tails[kv_head_id].token_count == 0
                        else key_tails[kv_head_id].source_page.header.token_start
                    )
                    or resident_bytes_changed
                )
            if value_tails[kv_head_id] is not None:
                resident_bytes_changed = (
                    value_tails[kv_head_id]._ensure_allocated(
                        token_start=token_index
                        if value_tails[kv_head_id].token_count == 0
                        else value_tails[kv_head_id].source_page.header.token_start
                    )
                    or resident_bytes_changed
                )

        self._batch_append_persistent_tail_tensors(key_tails, keys, token_start=token_index)
        self._batch_append_persistent_tail_tensors(value_tails, values, token_start=token_index)

        for kv_head_id in range(self.num_key_value_heads):
            state = self._state(layer_id, kv_head_id)
            if state.persistent_key_tail is None or state.persistent_value_tail is None:
                raise RuntimeError("persistent torch tail path requires allocated key/value tails")
            if state.tail.token_count > 0:
                state.tail.clear()
            if state.persistent_key_tail.token_count >= self.config.tokens_per_page:
                token_start_full = state.persistent_key_tail.source_page.header.token_start
                dense_keys = state.persistent_key_tail.materialize_rows()
                dense_values = state.persistent_value_tail.materialize_rows()
                finalized_key_page = encode_page(
                    dense_keys,
                    self.config,
                    kind="K",
                    layer_id=layer_id,
                    kv_head_id=kv_head_id,
                    token_start=token_start_full,
                    mode=None,
                    page_mode=(
                        self._select_page_mode(
                            dense_keys,
                            kind="K",
                            layer_id=layer_id,
                            kv_head_id=kv_head_id,
                            token_start=token_start_full,
                            sequence_length=int(token_index + token_count),
                            stage="decode",
                        )
                    ),
                    build_runtime_metadata=self._should_build_execution_runtime_metadata(kind="K"),
                    build_m2_sidecar=(
                        self.config.m2_prefilter_top_k > 0
                        and (len(state.session.key_pages) + 1) >= int(self.config.m2_prefilter_min_pages)
                    ),
                )
                finalized_value_page = encode_page(
                    dense_values,
                    self.config,
                    kind="V",
                    layer_id=layer_id,
                    kv_head_id=kv_head_id,
                    token_start=token_start_full,
                    mode=None,
                    page_mode=(
                        self._select_page_mode(
                            dense_values,
                            kind="V",
                            layer_id=layer_id,
                            kv_head_id=kv_head_id,
                            token_start=token_start_full,
                            sequence_length=int(token_index + token_count),
                            stage="decode",
                        )
                    ),
                    build_runtime_metadata=False,
                )
                state.session.append([finalized_key_page], [finalized_value_page], trace=trace)
                state.invalidate_decode_views()
                state.persistent_key_tail.clear()
                state.persistent_value_tail.clear()
            state.sequence_length += token_count
        if resident_bytes_changed:
            for kv_head_id in range(self.num_key_value_heads):
                self._refresh_state_resident_accounting(self._state(layer_id, kv_head_id))
            self._mark_prepared_chunk_cache_budget_dirty(reason="append_step_torch_tail_alloc")

    def _prepared_pages_with_tail(
        self,
        layer_id: int,
        kv_head_id: int,
        *,
        trace: ExecutionTrace | None = None,
    ) -> tuple[list[PageLike], list[PageLike], _PreparedDecodeViewLayout | None]:
        capture_stage_timings = bool(trace is not None and trace.capture_timings)

        def _record_layout_build_timing(started_at: float | None) -> None:
            if started_at is None:
                return
            self._record_decode_stage_timing(
                layer_id=int(layer_id),
                stage="prepare_layout_build",
                ms=(perf_counter() - started_at) * 1000.0,
            )

        state = self._state(layer_id, kv_head_id)
        self._ensure_prepared_static_pages(state, trace=trace)
        if state.persistent_key_tail is not None and state.persistent_value_tail is not None:
            prepared_key_tail = state.persistent_key_tail.active_page
            prepared_value_tail = state.persistent_value_tail.active_page
            if prepared_key_tail is not None and prepared_value_tail is not None:
                cached_key_pages = state.decode_key_pages_with_tail
                cached_value_pages = state.decode_value_pages_with_tail
                if (
                    cached_key_pages is not None
                    and cached_value_pages is not None
                    and cached_key_pages
                    and cached_value_pages
                    and cached_key_pages[-1] is prepared_key_tail
                    and cached_value_pages[-1] is prepared_value_tail
                    and len(cached_key_pages) == len(state.session.key_pages) + 1
                    and len(cached_value_pages) == len(state.session.value_pages) + 1
                ):
                    return cached_key_pages, cached_value_pages, state.decode_view_layout
                key_pages = list(state.session.key_pages)
                value_pages = list(state.session.value_pages)
                key_pages.append(prepared_key_tail)
                value_pages.append(prepared_value_tail)
                state.decode_key_pages_with_tail = key_pages
                state.decode_value_pages_with_tail = value_pages
                layout_started_at = perf_counter() if capture_stage_timings else None
                state.decode_view_layout = _build_prepared_decode_view_layout(key_pages, value_pages)
                _record_layout_build_timing(layout_started_at)
                return key_pages, value_pages, state.decode_view_layout
            state.invalidate_decode_views()
            layout_started_at = perf_counter() if capture_stage_timings else None
            state.decode_view_layout = _build_prepared_decode_view_layout(state.session.key_pages, state.session.value_pages)
            _record_layout_build_timing(layout_started_at)
            return state.session.key_pages, state.session.value_pages, state.decode_view_layout
        temp_pages = state.tail.build_temp_pages()
        if temp_pages is None:
            layout_started_at = perf_counter() if capture_stage_timings else None
            state.decode_view_layout = _build_prepared_decode_view_layout(state.session.key_pages, state.session.value_pages)
            _record_layout_build_timing(layout_started_at)
            return state.session.key_pages, state.session.value_pages, state.decode_view_layout
        temp_key_page, temp_value_page = temp_pages
        # Temporary live-tail pages are rebuilt on demand and should not go
        # through the shared prepared-page cache keyed by object id.
        prepared_temp_key_page = prepare_pages([temp_key_page], backend=self.backend, cache=None, trace=trace)[0]
        prepared_temp_value_page = prepare_pages([temp_value_page], backend=self.backend, cache=None, trace=trace)[0]
        key_pages = list(state.session.key_pages)
        value_pages = list(state.session.value_pages)
        key_pages.append(prepared_temp_key_page)
        value_pages.append(prepared_temp_value_page)
        layout_started_at = perf_counter() if capture_stage_timings else None
        layout = _build_prepared_decode_view_layout(key_pages, value_pages)
        _record_layout_build_timing(layout_started_at)
        return key_pages, value_pages, layout

    def decode_layer(
        self,
        layer_id: int,
        query_step: np.ndarray,
        q_head_to_kv_head: Sequence[int] | np.ndarray,
        *,
        query_scale: float = 1.0,
        trace: ExecutionTrace | None = None,
    ) -> np.ndarray:
        queries = _normalize_query_step(
            query_step,
            num_attention_heads=self.num_attention_heads,
            head_dim=self.config.head_dim,
        )
        scaled_queries = queries * np.float32(query_scale)
        grouped_query_heads = self._grouped_query_heads_for_mapping(q_head_to_kv_head)

        outputs = np.zeros((self.num_attention_heads, self.config.head_dim), dtype=np.float32)
        for kv_head_id, q_head_ids in enumerate(grouped_query_heads):
            if not q_head_ids:
                continue
            key_pages, value_pages, _ = self._prepared_pages_with_tail(layer_id, kv_head_id, trace=trace)
            if not key_pages:
                raise ValueError(f"layer {layer_id} kv_head {kv_head_id} has no cached tokens to decode against")
            kv_queries = scaled_queries[list(q_head_ids)]
            key_pages, value_pages = self._m2_prefilter_pages_numpy(kv_queries, key_pages, value_pages)
            self._sync_prepared_chunk_cache_budget()
            _, _, kv_outputs = decode_multi_query_step(
                kv_queries,
                key_pages,
                value_pages,
                backend=self.backend,
                trace=trace,
            )
            outputs[list(q_head_ids)] = kv_outputs
        return outputs

    def analyze_execution_shortlist_layer(
        self,
        layer_id: int,
        query_step: np.ndarray,
        q_head_to_kv_head: Sequence[int] | np.ndarray,
        *,
        query_scale: float = 1.0,
        prefer_grouped_batching: bool = True,
        trace: ExecutionTrace | None = None,
    ) -> dict[str, object]:
        queries = _normalize_query_step(
            query_step,
            num_attention_heads=self.num_attention_heads,
            head_dim=self.config.head_dim,
        )
        scaled_queries = queries * np.float32(query_scale)
        grouped_query_heads = self._grouped_query_heads_for_mapping(q_head_to_kv_head)
        layer_prefer_grouped_batching = (
            bool(prefer_grouped_batching)
            and not self.config.execution_grouped_batching_disabled_for_layer(layer_id=layer_id)
        )

        shortlist_enabled = self.config.execution_shortlist_enabled()
        group_entries: list[dict[str, object]] = []
        raw_selected_indices_by_group: list[list[int] | None] = []
        full_selected_indices_by_group: list[list[int]] = []
        key_pages_by_group: list[list[PageLike]] = []
        window_index_sets_by_group: list[set[int]] = []
        representative_queries: list[np.ndarray] = []
        shortlist_trace_records_by_group: list[dict[str, object] | None] = []

        for kv_head_id, q_head_ids in enumerate(grouped_query_heads):
            if not q_head_ids:
                continue
            key_pages, value_pages, _ = self._prepared_pages_with_tail(layer_id, kv_head_id, trace=trace)
            if not key_pages:
                raise ValueError(f"layer {layer_id} kv_head {kv_head_id} has no cached tokens to decode against")
            state = self._state(layer_id, kv_head_id)
            kv_queries = scaled_queries[list(q_head_ids)]
            key_pages, _ = self._m2_prefilter_pages_numpy(kv_queries, key_pages, value_pages)
            representative_query = kv_queries.mean(axis=0).astype(np.float32, copy=False)
            raw_selected_indices = None
            page_max_context_length = (
                max(_page_header(page).token_start + _page_header(page).token_count for page in key_pages)
                if key_pages
                else 0
            )
            context_length = int(state.sequence_length) if int(state.sequence_length) > 0 else int(page_max_context_length)
            layer_recent_window = int(
                self.config.resolve_execution_recent_window_for_context(
                    layer_id=layer_id,
                    context_length=context_length,
                )
            )
            if shortlist_enabled:
                trace_record_count = len(self._execution_shortlist_trace_records)
                raw_selected_indices = self._execution_shortlist_page_indices(
                    key_pages,
                    layer_id=layer_id,
                    kv_head_id=int(kv_head_id),
                    query_slice=representative_query,
                    context_length_override=int(context_length) if int(context_length) > 0 else None,
                    trace=trace,
                )
                shortlist_trace_record = (
                    dict(self._execution_shortlist_trace_records[-1])
                    if len(self._execution_shortlist_trace_records) > trace_record_count
                    else None
                )
            else:
                shortlist_trace_record = None
            selected_indices = (
                list(range(len(key_pages))) if raw_selected_indices is None else [int(index) for index in raw_selected_indices]
            )
            window_index_set = set(
                select_window_page_indices(
                    key_pages,
                    recent_window_tokens=layer_recent_window if layer_recent_window > 0 else None,
                    sink_window_tokens=int(self.config.execution_sink_window),
                )
            )
            group_entries.append(
                {
                    "kv_head_id": int(kv_head_id),
                    "query_head_ids": list(q_head_ids),
                    "layer_recent_window": int(layer_recent_window),
                    "context_length_page_max": int(page_max_context_length),
                    "context_length_effective": int(context_length),
                    "context_length_override_applied": bool(int(state.sequence_length) > 0),
                }
            )
            raw_selected_indices_by_group.append(raw_selected_indices)
            full_selected_indices_by_group.append(selected_indices)
            key_pages_by_group.append(key_pages)
            window_index_sets_by_group.append(window_index_set)
            representative_queries.append(representative_query)
            shortlist_trace_records_by_group.append(shortlist_trace_record)

        shortlist_attempted = any(indices is not None for indices in raw_selected_indices_by_group)
        union_rescue_records: list[dict[str, object]] = []
        if shortlist_attempted and len(group_entries) > 1 and layer_prefer_grouped_batching:
            adjusted_selected_indices_by_group, union_rescue_records = self._apply_execution_exact_promote_union_rescue(
                layer_id=layer_id,
                selected_indices_by_group=full_selected_indices_by_group,
                key_pages_by_group=key_pages_by_group,
                representative_queries=representative_queries,
                shortlist_traces_by_group=shortlist_trace_records_by_group,
                trace=trace,
            )
            full_selected_indices_by_group = [
                list(range(len(key_pages))) if indices is None else [int(index) for index in indices]
                for key_pages, indices in zip(key_pages_by_group, adjusted_selected_indices_by_group, strict=True)
            ]
            raw_selected_indices_by_group = [
                None if indices is None else [int(index) for index in indices]
                for indices in adjusted_selected_indices_by_group
            ]
            self._execution_shortlist_trace_records.extend(union_rescue_records)
        union_indices = sorted(
            {
                index
                for indices in raw_selected_indices_by_group
                if indices is not None
                for index in indices
            }
        )
        union_active = bool(union_indices) and len(group_entries) > 1 and layer_prefer_grouped_batching

        layer_exact_top_budget_total = 0
        layer_exact_top_overlap_total = 0
        layer_union_added_pages_total = 0
        layer_missed_age_buckets = {"recent": 0, "middle": 0, "old": 0}
        layer_recalls: list[float] = []
        layer_first_missed_ranks: list[int] = []

        for group_index, entry in enumerate(group_entries):
            key_pages = key_pages_by_group[group_index]
            selected_indices = full_selected_indices_by_group[group_index]
            union_rescue_record = next(
                (
                    record
                    for record in union_rescue_records
                    if int(record.get("group_index", -1)) == int(group_index)
                ),
                None,
            )
            if union_active:
                final_indices = list(union_indices)
                union_added_indices = sorted(set(union_indices) - set(selected_indices))
            else:
                final_indices = list(selected_indices)
                union_added_indices = []
            window_index_set = window_index_sets_by_group[group_index]
            representative_query = representative_queries[group_index]
            old_candidate_indices = [index for index in range(len(key_pages)) if index not in window_index_set]
            selected_old_indices = [index for index in final_indices if index not in window_index_set]
            exact_top_budget = min(len(selected_old_indices), len(old_candidate_indices))
            approx_top_budget = min(
                int(
                    self.config.resolve_execution_relevance_top_k_for_context(
                        layer_id=layer_id,
                        context_length=context_length,
                    )
                ),
                len(old_candidate_indices),
            )

            exact_rank_by_index: dict[int, int] = {}
            exact_top_indices: list[int] = []
            approx_rank_by_index: dict[int, int] = {}
            approx_top_indices: list[int] = []
            approx_scores_by_index: dict[int, float] = {}
            exact_scores_by_index: dict[int, float] = {}
            approx_boundary_margin = None
            approx_boundary_margin_normalized = None
            secondary_rank_by_index: dict[int, int] = {}
            secondary_top_indices: list[int] = []
            secondary_scores_by_index: dict[int, float] = {}
            recent_neighbor_anchor_pages = 0
            recent_neighbor_recent_old_pages = 0
            if old_candidate_indices:
                for index in old_candidate_indices:
                    approx_score = _score_page_relevance_for_mode(
                        representative_query,
                        key_pages[index],
                        relevance_mode=self.config.execution_relevance_mode,
                    )
                    if approx_score is None:
                        raise ValueError(
                            f"missing {self.config.execution_relevance_mode} relevance sidecars for layer {layer_id}"
                        )
                    approx_scores_by_index[int(index)] = float(approx_score)
                approx_ranked_pairs = sorted(
                    ((score, int(index)) for index, score in approx_scores_by_index.items()),
                    key=lambda item: item[0],
                    reverse=True,
                )
                if len(approx_ranked_pairs) > approx_top_budget and approx_top_budget > 0:
                    approx_boundary_margin = float(
                        approx_ranked_pairs[approx_top_budget - 1][0] - approx_ranked_pairs[approx_top_budget][0]
                    )
                    approx_std = max(
                        float(np.std(np.asarray(list(approx_scores_by_index.values()), dtype=np.float32))),
                        1e-6,
                    )
                    approx_boundary_margin_normalized = float(approx_boundary_margin / approx_std)
                approx_rank_by_index = {int(index): rank for rank, (_, index) in enumerate(approx_ranked_pairs, start=1)}
                approx_top_indices = [int(index) for _, index in approx_ranked_pairs[:approx_top_budget]]
                if self._execution_secondary_relevance_enabled(layer_id=layer_id):
                    secondary_scores_missing = False
                    for index in old_candidate_indices:
                        secondary_score = _score_page_relevance_for_mode(
                            representative_query,
                            key_pages[index],
                            relevance_mode=self.config.execution_secondary_relevance_mode,
                        )
                        if secondary_score is None:
                            secondary_scores_missing = True
                            secondary_scores_by_index = {}
                            secondary_rank_by_index = {}
                            secondary_top_indices = []
                            break
                        secondary_scores_by_index[int(index)] = float(secondary_score)
                    if not secondary_scores_missing:
                        secondary_ranked_pairs = sorted(
                            ((score, int(index)) for index, score in secondary_scores_by_index.items()),
                            key=lambda item: item[0],
                            reverse=True,
                        )
                        secondary_rank_by_index = {
                            int(index): rank for rank, (_, index) in enumerate(secondary_ranked_pairs, start=1)
                        }
                        secondary_top_indices = [int(index) for _, index in secondary_ranked_pairs[:approx_top_budget]]
                if self._execution_recent_neighbor_rescue_enabled(layer_id=layer_id) and layer_recent_window > 0:
                    recent_start = int(context_length) - int(layer_recent_window)
                    recent_neighbor_anchor_pages = sum(
                        1
                        for index in approx_top_indices
                        if int(_page_header(key_pages[index]).token_start + _page_header(key_pages[index]).token_count)
                        <= int(self.config.execution_recent_neighbor_rescue_anchor_window)
                    )
                    recent_neighbor_recent_old_pages = sum(
                        1
                        for index in approx_top_indices
                        if (
                            int(_page_header(key_pages[index]).token_start + _page_header(key_pages[index]).token_count)
                            <= int(recent_start)
                            and int(_page_header(key_pages[index]).token_start + _page_header(key_pages[index]).token_count)
                            > int(recent_start - layer_recent_window)
                        )
                    )
                candidate_logits = score_pages(
                    representative_query,
                    [key_pages[index] for index in old_candidate_indices],
                    backend=self.backend,
                    trace=trace,
                )
                ranked_pairs = sorted(
                    (
                        (
                            float(np.max(np.asarray(logits, dtype=np.float32))),
                            int(index),
                        )
                        for index, logits in zip(old_candidate_indices, candidate_logits, strict=True)
                    ),
                    key=lambda item: item[0],
                    reverse=True,
                )
                exact_scores_by_index = {
                    int(index): float(score)
                    for score, index in ranked_pairs
                }
                exact_rank_by_index = {int(index): rank for rank, (_, index) in enumerate(ranked_pairs, start=1)}
                exact_top_indices = [int(index) for _, index in ranked_pairs[:exact_top_budget]]

            exact_top_index_set = set(exact_top_indices)
            selected_old_index_set = set(selected_old_indices)
            exact_top_overlap = len(selected_old_index_set & exact_top_index_set)
            approx_top_index_set = set(approx_top_indices)
            approx_exact_top_overlap = len(approx_top_index_set & exact_top_index_set)
            secondary_top_index_set = set(secondary_top_indices)
            secondary_primary_top_overlap = len(secondary_top_index_set & approx_top_index_set)
            secondary_exact_top_overlap = len(secondary_top_index_set & exact_top_index_set)
            if exact_top_budget > 0:
                exact_top_recall = float(exact_top_overlap) / float(exact_top_budget)
                layer_recalls.append(exact_top_recall)
            else:
                exact_top_recall = 1.0
            if exact_top_budget > 0:
                approx_exact_top_recall = float(approx_exact_top_overlap) / float(exact_top_budget)
                secondary_exact_top_recall = float(secondary_exact_top_overlap) / float(exact_top_budget)
            else:
                approx_exact_top_recall = 1.0
                secondary_exact_top_recall = 1.0
            secondary_primary_top_recall = (
                float(secondary_primary_top_overlap) / float(max(min(len(secondary_top_indices), len(approx_top_indices)), 1))
                if secondary_top_indices and approx_top_indices
                else (1.0 if not secondary_top_indices else 0.0)
            )
            secondary_triggered = bool(
                secondary_top_indices
                and secondary_primary_top_recall < float(self.config.execution_secondary_relevance_min_overlap)
            )
            exact_promote_candidate_expansion_enabled, exact_promote_candidate_expansion_reason = (
                self._execution_exact_promote_policy_status(layer_id=layer_id, context_length=context_length)
            )
            exact_promote_enabled, exact_promote_disable_reason = self._execution_exact_promote_status(
                layer_id=layer_id,
                context_length=context_length,
                boundary_margin_normalized=approx_boundary_margin_normalized,
            )
            recent_neighbor_rescue_triggered = bool(
                self._execution_recent_neighbor_rescue_enabled(layer_id=layer_id)
                and recent_neighbor_anchor_pages >= int(self.config.execution_recent_neighbor_rescue_min_anchor_pages)
                and recent_neighbor_recent_old_pages > 0
            )
            missed_exact_indices = [index for index in exact_top_indices if index not in selected_old_index_set]
            first_missed_exact_rank = (
                int(exact_rank_by_index[missed_exact_indices[0]]) if missed_exact_indices else None
            )
            if first_missed_exact_rank is not None:
                layer_first_missed_ranks.append(first_missed_exact_rank)
            scorer_missed_exact_indices = [index for index in exact_top_indices if index not in approx_top_index_set]
            first_scorer_missed_exact_rank = (
                int(exact_rank_by_index[scorer_missed_exact_indices[0]]) if scorer_missed_exact_indices else None
            )
            missed_exact_age_buckets = {"recent": 0, "middle": 0, "old": 0}
            for index in missed_exact_indices:
                age_bucket = _page_age_bucket(key_pages[index], context_length=int(context_length))
                missed_exact_age_buckets[age_bucket] += 1
                layer_missed_age_buckets[age_bucket] += 1
            scorer_missed_exact_age_buckets = {"recent": 0, "middle": 0, "old": 0}
            for index in scorer_missed_exact_indices:
                age_bucket = _page_age_bucket(key_pages[index], context_length=int(context_length))
                scorer_missed_exact_age_buckets[age_bucket] += 1

            exact_top1_approx_rank = None
            approx_top1_exact_rank = None
            secondary_top1_exact_rank = None
            primary_top1_secondary_rank = None
            score_rank_correlation = None
            score_value_correlation = None
            mean_abs_rank_error = None
            if exact_top_indices:
                exact_top1_approx_rank = approx_rank_by_index.get(int(exact_top_indices[0]))
            if approx_top_indices:
                approx_top1_exact_rank = exact_rank_by_index.get(int(approx_top_indices[0]))
                primary_top1_secondary_rank = secondary_rank_by_index.get(int(approx_top_indices[0]))
            if secondary_top_indices:
                secondary_top1_exact_rank = exact_rank_by_index.get(int(secondary_top_indices[0]))
            if exact_rank_by_index and approx_rank_by_index:
                shared_indices = [index for index in old_candidate_indices if index in exact_rank_by_index and index in approx_rank_by_index]
                if shared_indices:
                    exact_ranks = [float(exact_rank_by_index[index]) for index in shared_indices]
                    approx_ranks = [float(approx_rank_by_index[index]) for index in shared_indices]
                    score_rank_correlation = _rank_correlation(approx_ranks, exact_ranks)
                    mean_abs_rank_error = float(
                        np.mean(
                            np.abs(
                                np.asarray(approx_ranks, dtype=np.float32) - np.asarray(exact_ranks, dtype=np.float32)
                            )
                        )
                    )
                    exact_scores = [float(exact_scores_by_index[index]) for index in shared_indices]
                    approx_scores = [float(approx_scores_by_index[index]) for index in shared_indices]
                    score_value_correlation = _rank_correlation(approx_scores, exact_scores)

            union_added_old_indices = [index for index in union_added_indices if index not in window_index_set]
            union_added_ranks = [
                int(exact_rank_by_index[index])
                for index in union_added_old_indices
                if index in exact_rank_by_index
            ]
            union_added_mean_exact_rank = (
                float(sum(union_added_ranks) / len(union_added_ranks)) if union_added_ranks else None
            )

            layer_exact_top_budget_total += int(exact_top_budget)
            layer_exact_top_overlap_total += int(exact_top_overlap)
            layer_union_added_pages_total += int(len(union_added_indices))

            entry.update(
                {
                    "context_length": int(context_length),
                    "total_pages": int(len(key_pages)),
                    "window_pages": int(len(window_index_set)),
                    "old_candidate_pages": int(len(old_candidate_indices)),
                    "selected_pages": int(len(final_indices)),
                    "selected_old_pages": int(len(selected_old_indices)),
                    "exact_top_budget": int(exact_top_budget),
                    "exact_top_overlap": int(exact_top_overlap),
                    "exact_top_recall": float(exact_top_recall),
                    "approx_top_budget": int(approx_top_budget),
                    "approx_exact_top_overlap": int(approx_exact_top_overlap),
                    "approx_exact_top_recall": float(approx_exact_top_recall),
                    "secondary_relevance_mode": (
                        self.config.execution_secondary_relevance_mode
                        if self._execution_secondary_relevance_enabled(layer_id=layer_id)
                        else None
                    ),
                    "secondary_primary_top_overlap": int(secondary_primary_top_overlap),
                    "secondary_primary_top_recall": float(secondary_primary_top_recall),
                    "secondary_exact_top_overlap": int(secondary_exact_top_overlap),
                    "secondary_exact_top_recall": float(secondary_exact_top_recall),
                    "secondary_triggered": bool(secondary_triggered),
                    "exact_promote_candidate_expansion_enabled": bool(exact_promote_candidate_expansion_enabled),
                    "exact_promote_candidate_expansion_disable_reason": exact_promote_candidate_expansion_reason,
                    "exact_promote_enabled": bool(exact_promote_enabled),
                    "exact_promote_disable_reason": exact_promote_disable_reason,
                    "union_exact_promote_rescue_applied": bool(
                        union_rescue_record is not None and bool(union_rescue_record.get("applied", False))
                    ),
                    "union_exact_promote_rescue_disable_reason": (
                        None if union_rescue_record is None else union_rescue_record.get("disable_reason")
                    ),
                    "union_exact_promote_rescue_selected_novel_count": int(
                        0 if union_rescue_record is None else union_rescue_record.get("selected_novel_count", 0)
                    ),
                    "union_exact_promote_rescue_selected_novel_page_ranges": (
                        [] if union_rescue_record is None else list(union_rescue_record.get("selected_novel_page_ranges", []))
                    ),
                    "recent_neighbor_anchor_pages": int(recent_neighbor_anchor_pages),
                    "recent_neighbor_recent_old_pages": int(recent_neighbor_recent_old_pages),
                    "recent_neighbor_rescue_triggered": bool(recent_neighbor_rescue_triggered),
                    "approx_boundary_margin": approx_boundary_margin,
                    "approx_boundary_margin_normalized": approx_boundary_margin_normalized,
                    "exact_top1_approx_rank": exact_top1_approx_rank,
                    "approx_top1_exact_rank": approx_top1_exact_rank,
                    "secondary_top1_exact_rank": secondary_top1_exact_rank,
                    "primary_top1_secondary_rank": primary_top1_secondary_rank,
                    "first_scorer_missed_exact_rank": first_scorer_missed_exact_rank,
                    "score_rank_correlation": score_rank_correlation,
                    "score_value_correlation": score_value_correlation,
                    "mean_abs_rank_error": mean_abs_rank_error,
                    "first_missed_exact_rank": first_missed_exact_rank,
                    "union_active": bool(union_active),
                    "union_added_pages": int(len(union_added_indices)),
                    "union_added_exact_top_hits": int(sum(1 for index in union_added_old_indices if index in exact_top_index_set)),
                    "union_added_mean_exact_rank": union_added_mean_exact_rank,
                    "selected_old_page_ranges": [_page_token_range(key_pages[index]) for index in selected_old_indices],
                    "top_approx_page_ranges": [_page_token_range(key_pages[index]) for index in approx_top_indices],
                    "top_secondary_page_ranges": [_page_token_range(key_pages[index]) for index in secondary_top_indices],
                    "top_exact_page_ranges": [_page_token_range(key_pages[index]) for index in exact_top_indices],
                    "missed_exact_page_ranges": [_page_token_range(key_pages[index]) for index in missed_exact_indices],
                    "missed_exact_age_buckets": missed_exact_age_buckets,
                    "scorer_missed_exact_page_ranges": [_page_token_range(key_pages[index]) for index in scorer_missed_exact_indices],
                    "scorer_missed_exact_age_buckets": scorer_missed_exact_age_buckets,
                }
            )

        first_missed_exact_rank_min = min(layer_first_missed_ranks) if layer_first_missed_ranks else None
        return {
            "layer_id": int(layer_id),
            "shortlist_enabled": bool(shortlist_enabled),
            "shortlist_attempted": bool(shortlist_attempted),
            "grouped_batching_enabled": bool(layer_prefer_grouped_batching),
            "union_active": bool(union_active),
            "group_count": int(len(group_entries)),
            "exact_top_budget_total": int(layer_exact_top_budget_total),
            "exact_top_overlap_total": int(layer_exact_top_overlap_total),
            "exact_top_recall_mean": (
                float(sum(layer_recalls) / len(layer_recalls)) if layer_recalls else 1.0
            ),
            "exact_top_recall_min": float(min(layer_recalls)) if layer_recalls else 1.0,
            "first_missed_exact_rank_min": first_missed_exact_rank_min,
            "union_added_pages_total": int(layer_union_added_pages_total),
            "missed_exact_age_buckets": dict(layer_missed_age_buckets),
            "groups": group_entries,
        }

    def decode_layer_torch(
        self,
        layer_id: int,
        query_step,
        q_head_to_kv_head: Sequence[int] | np.ndarray,
        *,
        query_scale: float = 1.0,
        prefer_grouped_batching: bool = True,
        trace: ExecutionTrace | None = None,
    ):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required for decode_layer_torch") from exc
        if not torch.is_tensor(query_step):
            raise TypeError("decode_layer_torch requires a torch.Tensor query_step")
        if self._torch_device_type is None:
            raise RuntimeError("decode_layer_torch is only available for a torch accelerator backend")
        if query_step.ndim == 4:
            if tuple(query_step.shape[:1] + query_step.shape[2:3]) != (1, 1):
                raise ValueError("query_step must have shape [q_heads, head_dim] or [1, q_heads, 1, head_dim]")
            queries = query_step[0, :, 0, :]
        elif query_step.ndim == 2:
            queries = query_step
        else:
            raise ValueError("query_step must have shape [q_heads, head_dim]")
        if int(queries.shape[0]) != self.num_attention_heads:
            raise ValueError(f"query_step must contain {self.num_attention_heads} query heads")
        if int(queries.shape[1]) != self.config.head_dim:
            raise ValueError(f"query_step head_dim must equal {self.config.head_dim}")

        scaled_queries = queries.to(dtype=torch.float32) * float(query_scale)
        grouped_query_heads = self._grouped_query_heads_for_mapping(q_head_to_kv_head)
        layer_prefer_grouped_batching = (
            bool(prefer_grouped_batching)
            and not self.config.execution_grouped_batching_disabled_for_layer(layer_id=layer_id)
        )
        capture_stage_timings = bool(trace is not None and trace.capture_timings)

        def _stage_start() -> float | None:
            return perf_counter() if capture_stage_timings else None

        def _stage_finish(stage: str, started_at: float | None) -> None:
            if started_at is None:
                return
            self._record_decode_stage_timing(
                layer_id=int(layer_id),
                stage=stage,
                ms=(perf_counter() - started_at) * 1000.0,
            )

        outputs = torch.zeros(
            (self.num_attention_heads, self.config.head_dim),
            dtype=torch.float32,
            device=scaled_queries.device,
        )
        active_q_head_ids: list[tuple[int, ...]] = []
        active_queries: list[Any] = []
        active_key_pages: list[Sequence[PageLike]] = []
        active_value_pages: list[Sequence[PageLike]] = []
        active_layouts: list[_PreparedDecodeViewLayout | None] = []
        active_context_lengths: list[int] = []
        active_representative_queries: list[np.ndarray] = []
        original_key_pages_by_group: list[Sequence[PageLike]] = []
        original_value_pages_by_group: list[Sequence[PageLike]] = []
        original_layouts: list[_PreparedDecodeViewLayout | None] = []
        shortlist_selected_indices_by_group: list[list[int] | None] = []
        shortlist_trace_records_by_group: list[dict[str, object] | None] = []
        for kv_head_id, q_head_ids in enumerate(grouped_query_heads):
            if not q_head_ids:
                continue
            prepare_started_at = _stage_start()
            key_pages, value_pages, decode_layout = self._prepared_pages_with_tail(layer_id, kv_head_id, trace=trace)
            _stage_finish("prepare_pages_with_tail", prepare_started_at)
            if not key_pages:
                raise ValueError(f"layer {layer_id} kv_head {kv_head_id} has no cached tokens to decode against")
            state = self._state(layer_id, kv_head_id)
            kv_queries = scaled_queries[list(q_head_ids)]
            m2_prefilter_started_at = _stage_start()
            key_pages, value_pages = self._m2_prefilter_pages_torch(kv_queries, key_pages, value_pages)
            _stage_finish("m2_prefilter", m2_prefilter_started_at)
            selected_indices = None
            shortlist_trace_record = None
            representative_query = kv_queries.mean(dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
            if self.config.execution_shortlist_enabled():
                query_export_started_at = _stage_start()
                _stage_finish("query_export", query_export_started_at)
                trace_record_count = len(self._execution_shortlist_trace_records)
                shortlist_started_at = _stage_start()
                selected_indices = self._execution_shortlist_page_indices(
                    key_pages,
                    layer_id=layer_id,
                    kv_head_id=int(kv_head_id),
                    query_slice=representative_query,
                    context_length_override=int(state.sequence_length) if int(state.sequence_length) > 0 else None,
                    trace=trace,
                )
                if len(self._execution_shortlist_trace_records) > trace_record_count:
                    shortlist_trace_record = dict(self._execution_shortlist_trace_records[-1])
                _stage_finish("shortlist_selection", shortlist_started_at)
            active_q_head_ids.append(q_head_ids)
            active_queries.append(kv_queries)
            original_key_pages_by_group.append(key_pages)
            original_value_pages_by_group.append(value_pages)
            original_layouts.append(decode_layout)
            active_key_pages.append(key_pages)
            active_value_pages.append(value_pages)
            active_layouts.append(decode_layout)
            active_context_lengths.append(int(state.sequence_length))
            active_representative_queries.append(representative_query)
            shortlist_selected_indices_by_group.append(selected_indices)
            shortlist_trace_records_by_group.append(shortlist_trace_record)

        shortlist_group_union_applied = False
        shortlist_attempted = any(indices is not None for indices in shortlist_selected_indices_by_group)
        shortlist_applied = False
        shortlist_selected_pages = 0
        shortlist_total_pages = 0
        if shortlist_attempted:
            total_pages_per_group = [len(pages) for pages in original_key_pages_by_group]
            shortlist_total_pages = int(sum(total_pages_per_group))
            if len(active_queries) > 1 and layer_prefer_grouped_batching:
                query_export_started_at = _stage_start()
                representative_queries = [
                    query.mean(dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
                    for query in active_queries
                ]
                _stage_finish("query_export", query_export_started_at)
                union_rescue_started_at = _stage_start()
                shortlist_selected_indices_by_group, union_rescue_records = self._apply_execution_exact_promote_union_rescue(
                    layer_id=layer_id,
                    selected_indices_by_group=shortlist_selected_indices_by_group,
                    key_pages_by_group=original_key_pages_by_group,
                    representative_queries=representative_queries,
                    shortlist_traces_by_group=shortlist_trace_records_by_group,
                    trace=trace,
                )
                self._execution_shortlist_trace_records.extend(union_rescue_records)
                _stage_finish("shortlist_union_rescue", union_rescue_started_at)
            shortlist_materialization_started_at = _stage_start()
            union_indices = sorted(
                {
                    index
                    for indices in shortlist_selected_indices_by_group
                    if indices is not None
                    for index in indices
                }
            )
            use_union_indices = bool(union_indices) and len(active_queries) > 1 and layer_prefer_grouped_batching
            if use_union_indices:
                shortlist_group_union_applied = True
                shortlist_selected_pages = int(len(union_indices) * len(active_queries))
                if len(union_indices) < total_pages_per_group[0]:
                    shortlist_applied = True
                active_key_pages = [[pages[index] for index in union_indices] for pages in original_key_pages_by_group]
                active_value_pages = [[pages[index] for index in union_indices] for pages in original_value_pages_by_group]
                active_layouts = [None] * len(active_key_pages)
            else:
                shortlisted_key_pages: list[Sequence[PageLike]] = []
                shortlisted_value_pages: list[Sequence[PageLike]] = []
                shortlisted_layouts: list[_PreparedDecodeViewLayout | None] = []
                for key_pages, value_pages, decode_layout, selected_indices in zip(
                    original_key_pages_by_group,
                    original_value_pages_by_group,
                    original_layouts,
                    shortlist_selected_indices_by_group,
                    strict=True,
                ):
                    if selected_indices is None:
                        shortlisted_key_pages.append(key_pages)
                        shortlisted_value_pages.append(value_pages)
                        shortlisted_layouts.append(decode_layout)
                        shortlist_selected_pages += int(len(key_pages))
                        continue
                    shortlist_selected_pages += int(len(selected_indices))
                    if len(selected_indices) < len(key_pages):
                        shortlist_applied = True
                        shortlisted_key_pages.append([key_pages[index] for index in selected_indices])
                        shortlisted_value_pages.append([value_pages[index] for index in selected_indices])
                        shortlisted_layouts.append(None)
                    else:
                        shortlisted_key_pages.append(key_pages)
                        shortlisted_value_pages.append(value_pages)
                        shortlisted_layouts.append(decode_layout)
                active_key_pages = shortlisted_key_pages
                active_value_pages = shortlisted_value_pages
                active_layouts = shortlisted_layouts
            _stage_finish("shortlist_materialization", shortlist_materialization_started_at)

        value_escape_applied = False
        if self.config.execution_value_escape_enabled_for_layer(layer_id=layer_id):
            active_value_pages, value_escape_applied = self._apply_execution_value_escape(
                layer_id=layer_id,
                key_pages_by_group=active_key_pages,
                value_pages_by_group=active_value_pages,
                context_lengths_by_group=active_context_lengths,
                representative_queries_by_group=active_representative_queries,
                trace=trace,
            )
            if value_escape_applied:
                active_layouts = [None] * len(active_layouts)

        chunk_budget_sync_started_at = _stage_start()
        self._sync_prepared_chunk_cache_budget(
            freeze_during_decode=bool(self.config.execution_freeze_chunk_budget_during_decode)
        )
        _stage_finish("chunk_budget_sync", chunk_budget_sync_started_at)

        grouping_validation_started_at = _stage_start()
        if shortlist_attempted and shortlist_applied and layer_prefer_grouped_batching and len(active_queries) > 1:
            shortlist_grouping_rejection_reason = _grouped_pages_batch_rejection_reason(
                active_key_pages,
                active_value_pages,
                active_queries,
            )
            shortlisted_can_batch = shortlist_grouping_rejection_reason is None
            if not shortlisted_can_batch:
                self._record_execution_shortlist(
                    layer_id=layer_id,
                    total_pages=shortlist_total_pages,
                    selected_pages=shortlist_selected_pages,
                    applied=False,
                    group_union_applied=shortlist_group_union_applied,
                    grouping_rejected=True,
                    grouping_rejection_reason=shortlist_grouping_rejection_reason,
                )
                active_key_pages = list(original_key_pages_by_group)
                active_value_pages = list(original_value_pages_by_group)
                active_layouts = list(original_layouts)
                shortlist_applied = False
            else:
                self._record_execution_shortlist(
                    layer_id=layer_id,
                    total_pages=shortlist_total_pages,
                    selected_pages=shortlist_selected_pages,
                    applied=True,
                    group_union_applied=shortlist_group_union_applied,
                )
        elif shortlist_attempted:
            self._record_execution_shortlist(
                layer_id=layer_id,
                total_pages=shortlist_total_pages,
                selected_pages=shortlist_selected_pages,
                applied=shortlist_applied,
                group_union_applied=shortlist_group_union_applied,
            )
        grouped_layout_rejection_reason = None
        grouped_page_rejection_reason = None
        if layer_prefer_grouped_batching:
            grouped_layout_rejection_reason = _grouped_layout_batch_rejection_reason(active_layouts, active_queries)
            grouped_page_rejection_reason = _grouped_pages_batch_rejection_reason(
                active_key_pages,
                active_value_pages,
                active_queries,
            )
        cached_group_layout = layer_prefer_grouped_batching and grouped_layout_rejection_reason is None
        grouped_path_ready = cached_group_layout or (layer_prefer_grouped_batching and grouped_page_rejection_reason is None)
        _stage_finish("grouping_validation", grouping_validation_started_at)

        if grouped_path_ready:
            self._record_decode_path(layer_id, "grouped_batched")
            key_chunk_lengths = active_layouts[0].key_chunk_lengths if cached_group_layout and active_layouts[0] is not None else None
            value_chunk_lengths = active_layouts[0].value_chunk_lengths if cached_group_layout and active_layouts[0] is not None else None
            backend_started_at = _stage_start()
            backend_trace_before = _backend_trace_ms_total(trace) if capture_stage_timings else 0.0
            _, _, grouped_outputs = decode_grouped_multiquery_step_prepared_torch_tensor(
                active_queries,
                active_key_pages,
                active_value_pages,
                key_chunk_lengths=key_chunk_lengths,
                value_chunk_lengths=value_chunk_lengths,
                compact_grouped_chunk=bool(self.config.execution_grouped_decode_compact),
                compact_grouped_mix_chunk=bool(self.config.execution_grouped_mix_compact),
                disable_packed_grouped_cuda_mix=bool(self.config.execution_grouped_mix_disable_packed_cuda),
                trace=trace,
            )
            backend_call_ms = 0.0
            if backend_started_at is not None:
                backend_call_ms = (perf_counter() - backend_started_at) * 1000.0
                self._record_decode_stage_timing(
                    layer_id=int(layer_id),
                    stage="backend_call_wall",
                    ms=backend_call_ms,
                )
                backend_trace_after = _backend_trace_ms_total(trace)
                self._record_decode_stage_timing(
                    layer_id=int(layer_id),
                    stage="backend_call_non_backend",
                    ms=backend_call_ms - float(backend_trace_after - backend_trace_before),
                )
            for q_head_ids, kv_outputs in zip(active_q_head_ids, grouped_outputs, strict=True):
                outputs[list(q_head_ids)] = kv_outputs
            return outputs

        if layer_prefer_grouped_batching:
            grouped_batch_rejection_reason = grouped_page_rejection_reason
            if grouped_batch_rejection_reason is None and grouped_layout_rejection_reason is not None:
                grouped_batch_rejection_reason = f"layout_{grouped_layout_rejection_reason}"
            if grouped_batch_rejection_reason is None and len(active_queries) <= 1:
                grouped_batch_rejection_reason = "single_query_group"
            if grouped_batch_rejection_reason is None:
                grouped_batch_rejection_reason = "unknown"
            self._record_decode_grouped_batch_rejection(
                layer_id=int(layer_id),
                reason=grouped_batch_rejection_reason,
            )
        self._record_decode_path(layer_id, "per_kv_fallback")
        for q_head_ids, kv_queries, key_pages, value_pages in zip(
            active_q_head_ids,
            active_queries,
            active_key_pages,
            active_value_pages,
            strict=True,
        ):
            backend_started_at = _stage_start()
            backend_trace_before = _backend_trace_ms_total(trace) if capture_stage_timings else 0.0
            _, _, kv_outputs = decode_multi_query_step_torch_tensor(
                kv_queries,
                key_pages,
                value_pages,
                device_type=self._torch_device_type,
                trace=trace,
            )
            if backend_started_at is not None:
                backend_call_ms = (perf_counter() - backend_started_at) * 1000.0
                self._record_decode_stage_timing(
                    layer_id=int(layer_id),
                    stage="backend_call_wall",
                    ms=backend_call_ms,
                )
                backend_trace_after = _backend_trace_ms_total(trace)
                self._record_decode_stage_timing(
                    layer_id=int(layer_id),
                    stage="backend_call_non_backend",
                    ms=backend_call_ms - float(backend_trace_after - backend_trace_before),
                )
            outputs[list(q_head_ids)] = kv_outputs
        return outputs
