from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from .attention_reference import softmax
from .backends import (
    PreparedPageTorch,
    cuda_available,
    decode_multi_query_step_cuda,
    decode_step_cuda,
    decode_multi_query_step_mps,
    decode_step_mps,
    mix_page_cpu_ref,
    mix_page_cuda,
    mix_page_mps,
    mps_available,
    page_supported_cuda,
    page_supported_mps,
    prepare_page_cuda,
    prepare_page_mps,
    prepare_pages_cuda,
    prepare_pages_mps,
    score_pages_cuda,
    score_pages_mps,
    score_page_cpu_ref,
    score_page_cuda,
    score_page_mps,
)
from .page_cache import PreparedPageCache
from .tracing import ExecutionTrace
from .types import EncodedPage

BackendName = Literal["cpu_ref", "torch_mps", "torch_cuda", "auto"]
PageLike = EncodedPage | PreparedPageTorch


def _resolve_backend(backend: BackendName, page: PageLike) -> Literal["cpu_ref", "torch_mps", "torch_cuda"]:
    if backend == "cpu_ref":
        return "cpu_ref"
    if backend == "torch_mps":
        if not mps_available():
            raise RuntimeError("torch_mps is unavailable on this machine")
        if not page_supported_mps(page):
            raise ValueError("page is unsupported by torch_mps in this phase")
        return "torch_mps"
    if backend == "torch_cuda":
        if not cuda_available():
            raise RuntimeError("torch_cuda is unavailable on this machine")
        if not page_supported_cuda(page):
            raise ValueError("page is unsupported by torch_cuda in this phase")
        return "torch_cuda"
    if isinstance(page, PreparedPageTorch):
        return "torch_cuda" if page.device_type == "cuda" else "torch_mps"
    if cuda_available() and page_supported_cuda(page):
        return "torch_cuda"
    if mps_available() and page_supported_mps(page):
        return "torch_mps"
    return "cpu_ref"


def _prepared_pages_backend(pages: Sequence[PageLike]) -> Literal["torch_mps", "torch_cuda"] | None:
    if not pages or not all(isinstance(page, PreparedPageTorch) for page in pages):
        return None
    device_type = pages[0].device_type
    if any(page.device_type != device_type for page in pages):
        raise ValueError("prepared torch pages must all target the same device")
    return "torch_cuda" if device_type == "cuda" else "torch_mps"


def prepare_page(
    page: PageLike,
    *,
    backend: BackendName = "auto",
    cache: PreparedPageCache | None = None,
    trace: ExecutionTrace | None = None,
) -> PageLike:
    resolved_backend = _resolve_backend(backend, page)
    if resolved_backend == "torch_mps":
        if cache is not None:
            return cache.prepare_page(page, backend="torch_mps", trace=trace)
        return prepare_page_mps(page, trace=trace)
    if resolved_backend == "torch_cuda":
        if cache is not None:
            return cache.prepare_page(page, backend="torch_cuda", trace=trace)
        return prepare_page_cuda(page, trace=trace)
    return page.source_page if isinstance(page, PreparedPageTorch) else page


def prepare_pages(
    pages: Sequence[PageLike],
    *,
    backend: BackendName = "auto",
    cache: PreparedPageCache | None = None,
    trace: ExecutionTrace | None = None,
) -> list[PageLike]:
    if pages:
        resolved_backend = _resolve_backend(backend, pages[0])
        if resolved_backend == "torch_mps":
            if cache is not None:
                return cache.prepare_pages(list(pages), backend="torch_mps", trace=trace)
            return prepare_pages_mps(pages, trace=trace)
        if resolved_backend == "torch_cuda":
            if cache is not None:
                return cache.prepare_pages(list(pages), backend="torch_cuda", trace=trace)
            return prepare_pages_cuda(pages, trace=trace)
    return [prepare_page(page, backend=backend, cache=cache, trace=trace) for page in pages]


def score_page(
    query_slice: np.ndarray,
    page: PageLike,
    *,
    backend: BackendName = "auto",
    trace: ExecutionTrace | None = None,
) -> np.ndarray:
    resolved_backend = _resolve_backend(backend, page)
    if resolved_backend == "torch_mps":
        return score_page_mps(query_slice, page, trace=trace)
    if resolved_backend == "torch_cuda":
        return score_page_cuda(query_slice, page, trace=trace)
    return score_page_cpu_ref(query_slice, page, trace=trace)


def score_pages(
    query_slice: np.ndarray,
    pages: Sequence[PageLike],
    *,
    backend: BackendName = "auto",
    cache: PreparedPageCache | None = None,
    trace: ExecutionTrace | None = None,
) -> list[np.ndarray]:
    if not pages:
        return []

    prepared_pages = prepare_pages(pages, backend=backend, cache=cache, trace=trace)
    prepared_backend = _prepared_pages_backend(prepared_pages)
    if prepared_backend == "torch_mps":
        return score_pages_mps(query_slice, prepared_pages, trace=trace)
    if prepared_backend == "torch_cuda":
        return score_pages_cuda(query_slice, prepared_pages, trace=trace)
    return [score_page(query_slice, page, backend=backend, trace=trace) for page in prepared_pages]


def mix_page(
    attn_weights: np.ndarray,
    page: PageLike,
    *,
    out_acc: np.ndarray | None = None,
    backend: BackendName = "auto",
    trace: ExecutionTrace | None = None,
) -> np.ndarray:
    resolved_backend = _resolve_backend(backend, page)
    if resolved_backend == "torch_mps":
        return mix_page_mps(attn_weights, page, out_acc=out_acc, trace=trace)
    if resolved_backend == "torch_cuda":
        return mix_page_cuda(attn_weights, page, out_acc=out_acc, trace=trace)
    return mix_page_cpu_ref(attn_weights, page, out_acc=out_acc, trace=trace)


def attention_step(
    query_slice: np.ndarray,
    key_page: PageLike,
    value_page: PageLike,
    *,
    backend: BackendName = "cpu_ref",
    cache: PreparedPageCache | None = None,
    trace: ExecutionTrace | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prepared_key_page = prepare_page(key_page, backend=backend, cache=cache, trace=trace)
    prepared_value_page = prepare_page(value_page, backend=backend, cache=cache, trace=trace)
    logits = score_page(query_slice, prepared_key_page, backend=backend, trace=trace)
    weights = softmax(logits)
    output = mix_page(weights, prepared_value_page, backend=backend, trace=trace)
    return logits, weights, output


def decode_step(
    query_slice: np.ndarray,
    key_pages: Sequence[PageLike],
    value_pages: Sequence[PageLike],
    *,
    backend: BackendName = "auto",
    cache: PreparedPageCache | None = None,
    trace: ExecutionTrace | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(key_pages) != len(value_pages):
        raise ValueError("key_pages and value_pages must contain the same number of pages")
    if not key_pages:
        raise ValueError("decode_step requires at least one page")

    return decode_step_with_page_logits(
        query_slice,
        key_pages,
        value_pages,
        backend=backend,
        cache=cache,
        trace=trace,
    )


def decode_multi_query_step(
    query_slices: np.ndarray,
    key_pages: Sequence[PageLike],
    value_pages: Sequence[PageLike],
    *,
    backend: BackendName = "auto",
    cache: PreparedPageCache | None = None,
    trace: ExecutionTrace | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    queries = np.asarray(query_slices, dtype=np.float32)
    if queries.ndim != 2:
        raise ValueError("query_slices must have shape [query_count, head_dim]")
    if len(key_pages) != len(value_pages):
        raise ValueError("key_pages and value_pages must contain the same number of pages")
    if not key_pages:
        raise ValueError("decode_multi_query_step requires at least one page")

    prepared_key_pages = prepare_pages(key_pages, backend=backend, cache=cache, trace=trace)
    prepared_value_pages = prepare_pages(value_pages, backend=backend, cache=cache, trace=trace)

    prepared_backend = _prepared_pages_backend(prepared_key_pages)
    if prepared_backend is not None and prepared_backend == _prepared_pages_backend(prepared_value_pages):
        if prepared_backend == "torch_cuda":
            return decode_multi_query_step_cuda(
                queries,
                prepared_key_pages,
                prepared_value_pages,
                trace=trace,
            )
        return decode_multi_query_step_mps(
            queries,
            prepared_key_pages,
            prepared_value_pages,
            trace=trace,
        )

    logits_list = []
    weights_list = []
    output_list = []
    for query_slice in queries:
        logits, weights, output = decode_step(
            query_slice,
            prepared_key_pages,
            prepared_value_pages,
            backend=backend,
            trace=trace,
        )
        logits_list.append(logits)
        weights_list.append(weights)
        output_list.append(output)
    return (
        np.stack(logits_list, axis=0).astype(np.float32, copy=False),
        np.stack(weights_list, axis=0).astype(np.float32, copy=False),
        np.stack(output_list, axis=0).astype(np.float32, copy=False),
    )


def decode_step_with_page_logits(
    query_slice: np.ndarray,
    key_pages: Sequence[PageLike],
    value_pages: Sequence[PageLike],
    *,
    page_logits: Sequence[np.ndarray | None] | None = None,
    backend: BackendName = "auto",
    cache: PreparedPageCache | None = None,
    trace: ExecutionTrace | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(key_pages) != len(value_pages):
        raise ValueError("key_pages and value_pages must contain the same number of pages")
    if not key_pages:
        raise ValueError("decode_step requires at least one page")
    if page_logits is not None and len(page_logits) != len(key_pages):
        raise ValueError("page_logits must align with key_pages")

    prepared_key_pages = prepare_pages(key_pages, backend=backend, cache=cache, trace=trace)
    prepared_value_pages = prepare_pages(value_pages, backend=backend, cache=cache, trace=trace)

    prepared_backend = _prepared_pages_backend(prepared_key_pages)
    if prepared_backend is not None and prepared_backend == _prepared_pages_backend(prepared_value_pages):
        if prepared_backend == "torch_cuda":
            return decode_step_cuda(
                query_slice,
                prepared_key_pages,
                prepared_value_pages,
                precomputed_page_logits=page_logits,
                trace=trace,
            )
        return decode_step_mps(
            query_slice,
            prepared_key_pages,
            prepared_value_pages,
            precomputed_page_logits=page_logits,
            trace=trace,
        )

    resolved_page_logits = []
    for index, page in enumerate(prepared_key_pages):
        cached_logits = None if page_logits is None else page_logits[index]
        if cached_logits is None:
            cached_logits = score_page(query_slice, page, backend=backend, trace=trace)
        resolved_page_logits.append(np.asarray(cached_logits, dtype=np.float32))
    logits = np.concatenate(resolved_page_logits).astype(np.float32, copy=False)
    weights = softmax(logits)

    output = np.zeros(prepared_key_pages[0].header.head_dim, dtype=np.float32)
    offset = 0
    for value_page in prepared_value_pages:
        token_count = value_page.header.token_count
        page_weights = weights[offset : offset + token_count]
        output = mix_page(page_weights, value_page, out_acc=output, backend=backend, trace=trace)
        offset += token_count

    return logits, weights, output
