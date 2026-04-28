from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..tracing import ExecutionTrace
from ..types import EncodedPage
from .torch_mps import (
    PreparedPageTorch,
    decode_grouped_multiquery_step_prepared_torch_tensor,
    decode_grouped_multiquery_step_prepared_torch_tensor_output_only,
    decode_grouped_multiquery_step_torch_tensor,
    decode_multi_query_step_torch,
    decode_multi_query_step_torch_tensor,
    decode_step_torch,
    page_supported_torch,
    prepare_page_torch,
    prepare_pages_torch,
    score_page_torch,
    score_pages_torch,
    mix_page_torch,
    torch_device_available,
)

PreparedPageCUDA = PreparedPageTorch


def cuda_available() -> bool:
    return torch_device_available("cuda")


def page_supported_cuda(page: EncodedPage | PreparedPageTorch) -> bool:
    return page_supported_torch(page)


def prepare_pages_cuda(
    pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    trace: ExecutionTrace | None = None,
) -> list[PreparedPageTorch]:
    return prepare_pages_torch(pages, device_type="cuda", trace=trace)


def prepare_page_cuda(
    page: EncodedPage | PreparedPageTorch,
    *,
    trace: ExecutionTrace | None = None,
) -> PreparedPageTorch:
    return prepare_page_torch(page, device_type="cuda", trace=trace)


def score_page_cuda(
    query_slice: np.ndarray | Any,
    page: EncodedPage | PreparedPageTorch,
    *,
    trace: ExecutionTrace | None = None,
) -> np.ndarray:
    return score_page_torch(query_slice, page, device_type="cuda", trace=trace)


def score_pages_cuda(
    query_slice: np.ndarray | Any,
    pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    trace: ExecutionTrace | None = None,
) -> list[np.ndarray]:
    return score_pages_torch(query_slice, pages, device_type="cuda", trace=trace)


def mix_page_cuda(
    attn_weights: np.ndarray | Any,
    page: EncodedPage | PreparedPageTorch,
    *,
    out_acc: np.ndarray | None = None,
    trace: ExecutionTrace | None = None,
) -> np.ndarray:
    return mix_page_torch(attn_weights, page, device_type="cuda", out_acc=out_acc, trace=trace)


def decode_step_cuda(
    query_slice: np.ndarray | Any,
    key_pages: Sequence[EncodedPage | PreparedPageTorch],
    value_pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    precomputed_page_logits=None,
    trace: ExecutionTrace | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return decode_step_torch(
        query_slice,
        key_pages,
        value_pages,
        device_type="cuda",
        precomputed_page_logits=precomputed_page_logits,
        trace=trace,
    )


def decode_multi_query_step_cuda_tensor(
    query_slices: np.ndarray | Any,
    key_pages: Sequence[EncodedPage | PreparedPageTorch],
    value_pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    trace: ExecutionTrace | None = None,
):
    return decode_multi_query_step_torch_tensor(
        query_slices,
        key_pages,
        value_pages,
        device_type="cuda",
        trace=trace,
    )


def decode_grouped_multiquery_step_cuda_tensor(
    query_groups,
    key_pages_by_group: Sequence[Sequence[EncodedPage | PreparedPageTorch]],
    value_pages_by_group: Sequence[Sequence[EncodedPage | PreparedPageTorch]],
    *,
    trace: ExecutionTrace | None = None,
):
    return decode_grouped_multiquery_step_torch_tensor(
        query_groups,
        key_pages_by_group,
        value_pages_by_group,
        device_type="cuda",
        trace=trace,
    )


def decode_grouped_multiquery_step_prepared_cuda_tensor(
    query_groups,
    key_pages_by_group: Sequence[Sequence[PreparedPageTorch]],
    value_pages_by_group: Sequence[Sequence[PreparedPageTorch]],
    *,
    trace: ExecutionTrace | None = None,
):
    return decode_grouped_multiquery_step_prepared_torch_tensor(
        query_groups,
        key_pages_by_group,
        value_pages_by_group,
        trace=trace,
    )


def decode_grouped_multiquery_step_prepared_cuda_tensor_output_only(
    query_groups,
    key_pages_by_group: Sequence[Sequence[PreparedPageTorch]],
    value_pages_by_group: Sequence[Sequence[PreparedPageTorch]],
    *,
    trace: ExecutionTrace | None = None,
):
    return decode_grouped_multiquery_step_prepared_torch_tensor_output_only(
        query_groups,
        key_pages_by_group,
        value_pages_by_group,
        trace=trace,
    )


def decode_multi_query_step_cuda(
    query_slices: np.ndarray,
    key_pages: Sequence[EncodedPage | PreparedPageTorch],
    value_pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    trace: ExecutionTrace | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return decode_multi_query_step_torch(query_slices, key_pages, value_pages, device_type="cuda", trace=trace)
