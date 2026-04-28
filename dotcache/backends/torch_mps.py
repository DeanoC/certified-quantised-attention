from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import time
from typing import Any, Literal, Sequence

import numpy as np
from ..tracing import ExecutionTrace
from ..types import EncodedPage, PageHeader
from ..modes.m2_key_sketch import segment_ids_for_token_count
from ..modes.m4_key_project import fixed_project_basis
from ..modes.turbo3 import TURBO3_CENTROIDS, fwht_last_dim
from ..packing import words_per_group

TorchDevice = Literal["mps", "cuda"]
PreparedDevice = Literal["torch_mps", "torch_cuda"]

_UNPACK_METADATA: dict[tuple[TorchDevice, int], tuple[Any, Any]] = {}
_SPILL_UNPACK_METADATA: dict[tuple[TorchDevice, int, int], tuple[Any, Any, Any, Any, Any, Any]] = {}
_TURBO3_CENTROID_TENSORS: dict[TorchDevice, Any] = {}
_FWHT_MATRICES: dict[tuple[TorchDevice, int], Any] = {}
_M4_BASIS_TENSORS: dict[tuple[TorchDevice, int, int, str], Any] = {}
_SEGMENT_ID_TENSORS: dict[tuple[TorchDevice, int, int], Any] = {}
_MAX_PREPARE_PAGES_PER_CHUNK = 128
_MPS_M0_KEY_PREPARE_PAGES_PER_CHUNK = 256
_MAX_PREPARED_CHUNK_CACHE_ENTRIES = 64
_MAX_PREPARED_CHUNK_CACHE_RESIDENT_BYTES = 64 * 1024 * 1024
_PREPARED_CHUNK_CACHE_BUDGET_OVERRIDE_BYTES: int | None = None
_MIN_PREPARED_CHUNK_CACHE_PAGE_COUNT = 4
_PREPARED_CHUNK_CACHE_KINDS = frozenset({"K", "V"})
_PREPARED_CHUNK_CACHE: "OrderedDict[tuple[tuple[int, int], ...], PreparedChunkMPS]" = OrderedDict()
_PREPARED_CHUNK_CACHE_RESIDENT_BYTES = 0
_PREPARED_GROUPED_CHUNK_CACHE: "OrderedDict[tuple[tuple[tuple[int, int], ...], ...], PreparedGroupedChunkMPS]" = OrderedDict()
_PREPARED_GROUPED_CHUNK_CACHE_RESIDENT_BYTES = 0
_PREPARED_PAGE_UID = 1
_PREPARED_CHUNK_CACHE_TOUCH_ID = 0
def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in environments without torch
        raise RuntimeError("torch is required for the torch accelerator backends") from exc
    return torch

def _backend_name(device_type: TorchDevice) -> PreparedDevice:
    return "torch_cuda" if device_type == "cuda" else "torch_mps"


def torch_device_available(device_type: TorchDevice) -> bool:
    try:
        import torch
    except ImportError:
        return False
    if device_type == "mps":
        return bool(torch.backends.mps.is_available())
    return bool(torch.cuda.is_available())


def mps_available() -> bool:
    return torch_device_available("mps")


def _device_tensor(array: np.ndarray, *, device: TorchDevice):
    torch = _load_torch()
    return torch.from_numpy(np.ascontiguousarray(array)).to(device=device)


def _torch_pack_codes(codes, *, bits: int, layout: str):
    torch = _load_torch()
    if int(codes.ndim) != 4:
        raise ValueError("codes must have shape [page_count, token_count, num_groups, group_size]")
    page_count, token_count, num_groups, group_size = map(int, codes.shape)
    if 32 % bits == 0:
        symbols_per_word = 32 // bits
        if group_size % symbols_per_word != 0:
            raise ValueError("torch-side code packing requires group_size divisible by symbols_per_word")
        word_count = group_size // symbols_per_word
        grouped = codes.to(dtype=torch.int32).reshape(page_count, token_count, num_groups, word_count, symbols_per_word)
        shifts = (torch.arange(symbols_per_word, dtype=torch.int32, device=codes.device) * int(bits)).reshape(1, 1, 1, 1, -1)
        packed = torch.bitwise_left_shift(grouped, shifts).sum(dim=-1).to(dtype=torch.int32)
    else:
        word_count = words_per_group(group_size, bits)
        packed = torch.zeros((page_count, token_count, num_groups, word_count), dtype=torch.int32, device=codes.device)
        codes_i64 = codes.to(dtype=torch.int64)
        for symbol_index in range(group_size):
            bit_offset = symbol_index * int(bits)
            word_index = bit_offset // 32
            bit_index = bit_offset % 32
            value = codes_i64[..., symbol_index]
            packed[..., word_index] = torch.bitwise_or(
                packed[..., word_index],
                torch.bitwise_left_shift(value, bit_index).to(dtype=torch.int32),
            )
            spill = bit_index + int(bits) - 32
            if spill > 0:
                packed[..., word_index + 1] = torch.bitwise_or(
                    packed[..., word_index + 1],
                    torch.bitwise_right_shift(value, int(bits) - spill).to(dtype=torch.int32),
                )
    if layout == "group_major":
        return packed.permute(0, 2, 1, 3).contiguous()
    if layout == "token_major":
        return packed.contiguous()
    raise ValueError("layout must be group_major or token_major")


@dataclass(slots=True)
class PreparedPageTorch:
    device_type: TorchDevice
    source_page: EncodedPage
    header: PageHeader
    payload: Any | None = None
    scales: Any | None = None
    bias: Any | None = None
    codebooks: Any | None = None
    m2_sketch: Any | None = None
    m2_basis: Any | None = None
    m2_mean: Any | None = None
    escape_payload: Any | None = None
    escape_scales: Any | None = None
    unpack_shifts: Any | None = None
    unpack_mask: Any | None = None
    host_to_device_nbytes: int = 0
    resident_nbytes: int = 0
    cache_uid: int = 0

    @property
    def payload_nbytes(self) -> int:
        return self.source_page.payload_nbytes

    @property
    def metadata_nbytes(self) -> int:
        return self.source_page.metadata_nbytes


PreparedPageMPS = PreparedPageTorch


@dataclass(slots=True)
class PreparedChunkMPS:
    header: PageHeader
    payload_groups: tuple[Any, ...]
    codes_groups: tuple[Any, ...] | None
    scales_groups: tuple[Any, ...] | None
    bias_groups: tuple[Any, ...] | None
    escape_payload_batch: Any | None = None
    escape_scales_batch: Any | None = None
    fused_scaled_codes: Any | None = None
    m2_sketch_groups: tuple[Any, ...] | None = None
    m2_basis_groups: tuple[Any, ...] | None = None
    m2_mean_groups: tuple[Any, ...] | None = None
    m2_segment_ids: Any | None = None
    resident_nbytes: int = 0
    touch_id: int = 0


@dataclass(slots=True)
class PreparedGroupedChunkMPS:
    header: PageHeader
    payload_groups: tuple[Any, ...]
    codes_groups: tuple[Any, ...] | None
    scales_groups: tuple[Any, ...] | None
    bias_groups: tuple[Any, ...] | None
    fused_scaled_codes: Any | None = None
    m2_sketch_groups: tuple[Any, ...] | None = None
    m2_basis_groups: tuple[Any, ...] | None = None
    m2_mean_groups: tuple[Any, ...] | None = None
    m2_segment_ids: Any | None = None
    m2_sketch_tensor: Any | None = None
    m2_basis_tensor: Any | None = None
    m2_mean_tensor: Any | None = None
    resident_nbytes: int = 0
    payload_groups_tensor: Any | None = None
    scales_groups_tensor: Any | None = None
    bias_groups_tensor: Any | None = None
    touch_id: int = 0


def _supports_fused_two_group64(header: PageHeader) -> bool:
    return bool(
        header.head_dim == 64
        and header.padded_head_dim == 64
        and header.group_size == 32
        and header.num_groups == 2
    )

def _supports_fused_m0_3bit(header: PageHeader, *, device_type: TorchDevice) -> bool:
    return bool(
        device_type == "mps"
        and header.mode_default == "M0"
        and header.bits == 3
        and header.quant_scheme == "affine"
    )


def _supports_grouped_fused_only_cache(header: PageHeader, *, device_type: TorchDevice) -> bool:
    if _supports_fused_m0_3bit(header, device_type=device_type):
        return True
    return _supports_fused_two_group64(header) and device_type in {"cuda", "mps"}


def _supports_packed_four_group128_cuda(header: PageHeader, *, device_type: TorchDevice) -> bool:
    return bool(
        device_type == "cuda"
        and header.mode_default == "M0"
        and header.quant_scheme == "affine"
        and header.layout == "group_major"
        and header.bits == 4
        and header.group_size == 32
        and header.num_groups == 4
        and header.padded_head_dim == 128
    )


def _fused_two_group64_cache_dtype(*, device_type: TorchDevice):
    torch = _load_torch()
    if device_type == "cuda":
        return torch.float16
    return torch.float32


def _m0_affine_metadata_dtype(*, device_type: TorchDevice):
    return _load_torch().float32


def _escape_scale_dtype(*, device_type: TorchDevice):
    torch = _load_torch()
    if device_type == "mps":
        return torch.float32
    return torch.float16
def prepared_chunk_cache_resident_bytes() -> int:
    return int(_PREPARED_CHUNK_CACHE_RESIDENT_BYTES + _PREPARED_GROUPED_CHUNK_CACHE_RESIDENT_BYTES)


def _effective_max_prepared_chunk_cache_resident_bytes() -> int:
    if _PREPARED_CHUNK_CACHE_BUDGET_OVERRIDE_BYTES is None:
        return int(_MAX_PREPARED_CHUNK_CACHE_RESIDENT_BYTES)
    return int(min(_MAX_PREPARED_CHUNK_CACHE_RESIDENT_BYTES, _PREPARED_CHUNK_CACHE_BUDGET_OVERRIDE_BYTES))


def _next_prepared_chunk_cache_touch_id() -> int:
    global _PREPARED_CHUNK_CACHE_TOUCH_ID
    _PREPARED_CHUNK_CACHE_TOUCH_ID += 1
    return _PREPARED_CHUNK_CACHE_TOUCH_ID


def _touch_prepared_chunk(chunk: PreparedChunkMPS | PreparedGroupedChunkMPS) -> None:
    chunk.touch_id = _next_prepared_chunk_cache_touch_id()


def _evict_oldest_prepared_chunk_cache_entry() -> bool:
    global _PREPARED_CHUNK_CACHE_RESIDENT_BYTES
    global _PREPARED_GROUPED_CHUNK_CACHE_RESIDENT_BYTES
    oldest_single = next(iter(_PREPARED_CHUNK_CACHE.items()), None)
    oldest_grouped = next(iter(_PREPARED_GROUPED_CHUNK_CACHE.items()), None)
    if oldest_single is None and oldest_grouped is None:
        return False
    if oldest_grouped is None or (
        oldest_single is not None and oldest_single[1].touch_id <= oldest_grouped[1].touch_id
    ):
        _, evicted_chunk = _PREPARED_CHUNK_CACHE.popitem(last=False)
        _PREPARED_CHUNK_CACHE_RESIDENT_BYTES = max(
            0,
            _PREPARED_CHUNK_CACHE_RESIDENT_BYTES - evicted_chunk.resident_nbytes,
        )
        return True
    _, evicted_chunk = _PREPARED_GROUPED_CHUNK_CACHE.popitem(last=False)
    _PREPARED_GROUPED_CHUNK_CACHE_RESIDENT_BYTES = max(
        0,
        _PREPARED_GROUPED_CHUNK_CACHE_RESIDENT_BYTES - evicted_chunk.resident_nbytes,
    )
    return True


def _trim_prepared_chunk_cache() -> None:
    effective_max_resident_bytes = _effective_max_prepared_chunk_cache_resident_bytes()
    while (
        len(_PREPARED_CHUNK_CACHE) + len(_PREPARED_GROUPED_CHUNK_CACHE) > _MAX_PREPARED_CHUNK_CACHE_ENTRIES
        or prepared_chunk_cache_resident_bytes() > effective_max_resident_bytes
    ):
        if not _evict_oldest_prepared_chunk_cache_entry():
            break


def configure_prepared_chunk_cache(
    *,
    max_entries: int | None = None,
    max_resident_bytes: int | None = None,
    min_page_count: int | None = None,
    cached_kinds: Sequence[str] | None = None,
    clear: bool = True,
) -> None:
    global _MAX_PREPARED_CHUNK_CACHE_ENTRIES
    global _MAX_PREPARED_CHUNK_CACHE_RESIDENT_BYTES
    global _MIN_PREPARED_CHUNK_CACHE_PAGE_COUNT
    global _PREPARED_CHUNK_CACHE_KINDS
    if max_entries is not None:
        _MAX_PREPARED_CHUNK_CACHE_ENTRIES = max(0, int(max_entries))
    if max_resident_bytes is not None:
        _MAX_PREPARED_CHUNK_CACHE_RESIDENT_BYTES = max(0, int(max_resident_bytes))
    if min_page_count is not None:
        _MIN_PREPARED_CHUNK_CACHE_PAGE_COUNT = max(1, int(min_page_count))
    if cached_kinds is not None:
        _PREPARED_CHUNK_CACHE_KINDS = frozenset(str(kind) for kind in cached_kinds)
    if clear:
        clear_prepared_chunk_cache()
        return
    _trim_prepared_chunk_cache()


def set_prepared_chunk_cache_budget_override(*, max_resident_bytes: int | None) -> None:
    global _PREPARED_CHUNK_CACHE_BUDGET_OVERRIDE_BYTES
    _PREPARED_CHUNK_CACHE_BUDGET_OVERRIDE_BYTES = None if max_resident_bytes is None else max(0, int(max_resident_bytes))
    _trim_prepared_chunk_cache()


def clear_prepared_chunk_cache() -> None:
    global _PREPARED_CHUNK_CACHE_RESIDENT_BYTES
    global _PREPARED_GROUPED_CHUNK_CACHE_RESIDENT_BYTES
    _PREPARED_CHUNK_CACHE.clear()
    _PREPARED_CHUNK_CACHE_RESIDENT_BYTES = 0
    _PREPARED_GROUPED_CHUNK_CACHE.clear()
    _PREPARED_GROUPED_CHUNK_CACHE_RESIDENT_BYTES = 0


def _next_prepared_page_uid() -> int:
    global _PREPARED_PAGE_UID
    page_uid = _PREPARED_PAGE_UID
    _PREPARED_PAGE_UID += 1
    return page_uid


def _prepare_signature(page: EncodedPage | PreparedPageTorch) -> tuple[int | str, ...]:
    source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
    header = source_page.header
    sketch_dim = int(source_page.m2_sketch.shape[-1]) if source_page.m2_sketch is not None else 0
    segment_count = int(source_page.m2_basis.shape[1]) if source_page.m2_basis is not None and source_page.m2_basis.ndim == 4 else 1
    centered = int(source_page.m2_mean is not None)
    return (
        header.kind,
        header.mode_default,
        header.token_count,
        header.head_dim,
        header.padded_head_dim,
        header.group_size,
        header.num_groups,
        header.bits,
        header.words_per_group,
        header.layout,
        header.quant_scheme,
        header.escape_dtype,
        sketch_dim,
        segment_count,
        centered,
    )


def _max_prepare_pages_for_source_page(page: EncodedPage | PreparedPageTorch, *, device_type: TorchDevice) -> int:
    source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
    if device_type == "mps" and source_page.header.mode_default == "M0" and source_page.header.kind == "K":
        return _MPS_M0_KEY_PREPARE_PAGES_PER_CHUNK
    return _MAX_PREPARE_PAGES_PER_CHUNK


def _batched_signature(page: PreparedPageTorch) -> tuple[int | str, ...]:
    header = page.header
    sketch_dim = int(page.m2_sketch.shape[-1]) if page.m2_sketch is not None else 0
    segment_count = int(page.m2_basis.shape[1]) if page.m2_basis is not None and page.m2_basis.dim() == 4 else 1
    centered = int(page.m2_mean is not None)
    return (
        page.device_type,
        header.kind,
        header.mode_default,
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


def _chunk_compatible_source_pages(
    pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    device_type: TorchDevice,
) -> list[list[EncodedPage | PreparedPageTorch]]:
    chunks: list[list[EncodedPage | PreparedPageTorch]] = []
    current_chunk: list[EncodedPage | PreparedPageTorch] = []
    current_signature: tuple[int | str, ...] | None = None
    current_limit = _MAX_PREPARE_PAGES_PER_CHUNK
    for page in pages:
        signature = _prepare_signature(page)
        if current_chunk and (
            signature != current_signature or len(current_chunk) >= current_limit
        ):
            chunks.append(current_chunk)
            current_chunk = [page]
            current_signature = signature
            current_limit = _max_prepare_pages_for_source_page(page, device_type=device_type)
            continue
        if not current_chunk:
            current_signature = signature
            current_limit = _max_prepare_pages_for_source_page(page, device_type=device_type)
        current_chunk.append(page)
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def _chunk_compatible_pages(pages: Sequence[PreparedPageTorch]) -> list[list[PreparedPageTorch]]:
    chunks: list[list[PreparedPageTorch]] = []
    current_chunk: list[PreparedPageTorch] = []
    current_signature: tuple[int | str, ...] | None = None
    for page in pages:
        signature = _batched_signature(page)
        if current_chunk and signature != current_signature:
            chunks.append(current_chunk)
            current_chunk = [page]
            current_signature = signature
            continue
        if not current_chunk:
            current_signature = signature
        current_chunk.append(page)
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def _chunk_lengths_for_pages(pages: Sequence[PreparedPageTorch]) -> tuple[int, ...]:
    return tuple(len(chunk) for chunk in _chunk_compatible_pages(pages))


def _merged_chunk_lengths_for_page_groups(
    pages_by_group: Sequence[Sequence[PreparedPageTorch]],
) -> tuple[int, ...]:
    if not pages_by_group:
        return ()
    page_count = len(pages_by_group[0])
    if page_count == 0:
        return ()
    boundaries = {0, page_count}
    for group_pages in pages_by_group:
        if len(group_pages) != page_count:
            raise ValueError("all grouped page streams must have the same page count")
        offset = 0
        for chunk_length in _chunk_lengths_for_pages(group_pages):
            offset += int(chunk_length)
            boundaries.add(offset)
    sorted_boundaries = sorted(boundaries)
    return tuple(
        int(sorted_boundaries[index + 1] - sorted_boundaries[index])
        for index in range(len(sorted_boundaries) - 1)
    )


def _aligned_chunk_lengths_for_page_pairs(
    key_pages: Sequence[PreparedPageTorch],
    value_pages: Sequence[PreparedPageTorch],
) -> tuple[int, ...]:
    if len(key_pages) != len(value_pages):
        raise ValueError("key/value page streams must have matching page counts")
    if not key_pages:
        return ()
    lengths: list[int] = []
    current_length = 0
    current_key_signature: tuple[int | str, ...] | None = None
    current_value_signature: tuple[int | str, ...] | None = None
    for key_page, value_page in zip(key_pages, value_pages, strict=True):
        key_signature = _batched_signature(key_page)
        value_signature = _batched_signature(value_page)
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


def _signature_buckets_for_page_chunk(
    pages_by_group: Sequence[Sequence[PreparedPageTorch]],
) -> tuple[tuple[int, ...], ...]:
    buckets: dict[tuple[tuple[int | str, ...], ...], list[int]] = {}
    for group_index, group_pages in enumerate(pages_by_group):
        signature = tuple(_batched_signature(page) for page in group_pages)
        buckets.setdefault(signature, []).append(int(group_index))
    return tuple(tuple(indices) for indices in buckets.values())


def _prepared_chunk_cache_key(pages: Sequence[PreparedPageTorch]) -> tuple[tuple[int, int], ...] | None:
    if not pages:
        return None
    if len(_chunk_compatible_pages(pages)) != 1:
        return None
    if pages[0].header.mode_default not in ("M0", "M2", "M3", "M4", "T3"):
        return None
    return tuple((int(page.cache_uid), int(page.header.token_count)) for page in pages)


def _segment_ids_tensor(token_count: int, segment_count: int, *, device_type: TorchDevice):
    cache_key = (device_type, int(token_count), int(segment_count))
    cached = _SEGMENT_ID_TENSORS.get(cache_key)
    if cached is not None:
        return cached
    tensor = _load_torch().from_numpy(
        segment_ids_for_token_count(int(token_count), int(segment_count))
    ).to(device=device_type)
    _SEGMENT_ID_TENSORS[cache_key] = tensor
    return tensor


def _build_prepared_chunk_mps(pages: Sequence[PreparedPageTorch]) -> PreparedChunkMPS:
    torch = _load_torch()
    if not pages:
        raise ValueError("pages must be non-empty")
    header = pages[0].header
    device_type = pages[0].device_type
    if header.mode_default not in ("M0", "M2", "M3", "M4", "T3"):
        raise ValueError("prepared chunk cache currently supports only M0, M2, M3, M4, and T3 pages")
    if header.mode_default == "M3":
        escape_payload_batch = torch.stack(
            [page.escape_payload[: header.token_count, : header.head_dim] for page in pages],
            dim=0,
        ).contiguous()
        escape_scales_batch = None
        resident_nbytes = int(escape_payload_batch.numel() * escape_payload_batch.element_size())
        if header.escape_dtype == "int8":
            escape_scales_batch = torch.stack(
                [page.escape_scales[: header.token_count] for page in pages],
                dim=0,
            ).contiguous()
            escape_scales_batch = escape_scales_batch.to(dtype=_escape_scale_dtype(device_type=device_type))
            resident_nbytes += int(escape_scales_batch.numel() * escape_scales_batch.element_size())
        return PreparedChunkMPS(
            header=header,
            payload_groups=(),
            codes_groups=None,
            scales_groups=None,
            bias_groups=None,
            escape_payload_batch=escape_payload_batch,
            escape_scales_batch=escape_scales_batch,
            fused_scaled_codes=None,
            resident_nbytes=resident_nbytes,
        )
    if header.mode_default == "M2":
        m2_sketch_groups = tuple(
            torch.stack([page.m2_sketch[:, group_index, :] for page in pages], dim=0).contiguous()
            for group_index in range(header.num_groups)
        )
        m2_basis_groups = tuple(
            torch.stack([page.m2_basis[group_index] for page in pages], dim=0).contiguous()
            for group_index in range(header.num_groups)
        )
        m2_mean_groups = tuple(
            torch.stack([page.m2_mean[group_index] for page in pages], dim=0).contiguous()
            for group_index in range(header.num_groups)
        )
        m2_segment_ids = None
        if m2_basis_groups and int(m2_basis_groups[0].dim()) == 4:
            m2_segment_ids = _segment_ids_tensor(
                header.token_count,
                int(m2_basis_groups[0].shape[1]),
                device_type=device_type,
            )
        resident_nbytes = sum(int(tensor.numel() * tensor.element_size()) for tensor in m2_sketch_groups)
        resident_nbytes += sum(int(tensor.numel() * tensor.element_size()) for tensor in m2_basis_groups)
        resident_nbytes += sum(int(tensor.numel() * tensor.element_size()) for tensor in m2_mean_groups)
        return PreparedChunkMPS(
            header=header,
            payload_groups=(),
            codes_groups=None,
            scales_groups=None,
            bias_groups=None,
            m2_sketch_groups=m2_sketch_groups,
            m2_basis_groups=m2_basis_groups,
            m2_mean_groups=m2_mean_groups,
            m2_segment_ids=m2_segment_ids,
            fused_scaled_codes=None,
            resident_nbytes=resident_nbytes,
        )
    if header.mode_default == "M4":
        shared_source_basis = pages[0].source_page.m2_basis
        shared_basis = (
            pages[0].header.project_basis == "svd_shared"
            and pages[0].m2_basis is None
            and shared_source_basis is not None
            and all(page.source_page.m2_basis is shared_source_basis for page in pages)
        )
        m2_sketch_groups = tuple(
            torch.stack([page.m2_sketch[:, group_index, :] for page in pages], dim=0).contiguous()
            for group_index in range(header.num_groups)
        )
        m2_basis_groups = (
            tuple(
                (
                    _device_tensor(np.asarray(pages[0].source_page.m2_basis[group_index]), device=device_type).contiguous()
                    if shared_basis
                    else torch.stack([page.m2_basis[group_index] for page in pages], dim=0).contiguous()
                )
                for group_index in range(header.num_groups)
            )
            if (shared_basis or pages[0].m2_basis is not None)
            else None
        )
        m2_mean_groups = tuple(
            torch.stack([page.m2_mean[group_index] for page in pages], dim=0).contiguous()
            for group_index in range(header.num_groups)
        )
        resident_nbytes = sum(int(tensor.numel() * tensor.element_size()) for tensor in m2_sketch_groups)
        if m2_basis_groups is not None:
            resident_nbytes += sum(int(tensor.numel() * tensor.element_size()) for tensor in m2_basis_groups)
        resident_nbytes += sum(int(tensor.numel() * tensor.element_size()) for tensor in m2_mean_groups)
        return PreparedChunkMPS(
            header=header,
            payload_groups=(),
            codes_groups=None,
            scales_groups=None,
            bias_groups=None,
            m2_sketch_groups=m2_sketch_groups,
            m2_basis_groups=m2_basis_groups,
            m2_mean_groups=m2_mean_groups,
            m2_segment_ids=None,
            fused_scaled_codes=None,
            resident_nbytes=resident_nbytes,
        )
    payload_groups = tuple(torch.stack([page.payload[group_index] for page in pages], dim=0) for group_index in range(header.num_groups))
    codes_groups = tuple(
        _unpack_bits_torch(
            payload_groups[group_index].reshape(-1, header.words_per_group),
            pages[0].unpack_shifts,
            pages[0].unpack_mask,
            header.group_size,
        ).reshape(len(pages), header.token_count, header.group_size)
        for group_index in range(header.num_groups)
    )
    scales_groups = tuple(
        torch.stack([page.scales[:, group_index].to(torch.float32) for page in pages], dim=0)
        for group_index in range(header.num_groups)
    )
    bias_groups = (
        tuple(
            torch.stack([page.bias[:, group_index].to(torch.float32) for page in pages], dim=0)
            for group_index in range(header.num_groups)
        )
        if header.mode_default == "M0"
        else None
    )
    fused_scaled_codes = None
    if header.mode_default == "M0" and (
        _supports_fused_two_group64(header) or _supports_fused_m0_3bit(header, device_type=pages[0].device_type)
    ):
        fused_dtype = _fused_two_group64_cache_dtype(device_type=device_type)
        fused_scaled_codes = torch.cat(
            [
                codes_groups[group_index].to(dtype=fused_dtype) * scales_groups[group_index].to(dtype=fused_dtype)[..., None]
                for group_index in range(header.num_groups)
            ],
            dim=-1,
        ).contiguous()
        bias_groups = tuple(bias.to(dtype=fused_dtype) for bias in bias_groups)
    if fused_scaled_codes is not None:
        # For the fused grouped-64 path, retain only the pre-scaled fused tensor and
        # the affine bias terms. Keeping payload/codes/scales as well doubles memory
        # for data the hot path no longer reads.
        payload_groups = ()
        codes_groups = None
        scales_groups = None
        resident_nbytes = int(fused_scaled_codes.numel() * fused_scaled_codes.element_size())
    else:
        resident_nbytes = sum(int(tensor.numel() * tensor.element_size()) for tensor in payload_groups)
        resident_nbytes += sum(int(tensor.numel() * tensor.element_size()) for tensor in codes_groups)
        resident_nbytes += sum(int(tensor.numel() * tensor.element_size()) for tensor in scales_groups)
    if bias_groups is not None:
        resident_nbytes += sum(int(tensor.numel() * tensor.element_size()) for tensor in bias_groups)
    return PreparedChunkMPS(
        header=header,
        payload_groups=payload_groups,
        codes_groups=codes_groups,
        scales_groups=scales_groups,
        bias_groups=bias_groups,
        m2_sketch_groups=None,
        m2_basis_groups=None,
        m2_mean_groups=None,
        m2_segment_ids=None,
        fused_scaled_codes=fused_scaled_codes,
        resident_nbytes=resident_nbytes,
    )


def _get_prepared_chunk_mps(pages: Sequence[PreparedPageTorch]) -> PreparedChunkMPS | None:
    global _PREPARED_CHUNK_CACHE_RESIDENT_BYTES
    cache_key = _prepared_chunk_cache_key(pages)
    if cache_key is None:
        return None
    if pages[0].header.kind not in _PREPARED_CHUNK_CACHE_KINDS:
        return None
    if len(pages) < _MIN_PREPARED_CHUNK_CACHE_PAGE_COUNT:
        return None
    cached_chunk = _PREPARED_CHUNK_CACHE.get(cache_key)
    if cached_chunk is not None:
        _touch_prepared_chunk(cached_chunk)
        _PREPARED_CHUNK_CACHE.move_to_end(cache_key)
        return cached_chunk
    prepared_chunk = _build_prepared_chunk_mps(pages)
    effective_max_resident_bytes = _effective_max_prepared_chunk_cache_resident_bytes()
    if (
        _MAX_PREPARED_CHUNK_CACHE_ENTRIES <= 0
        or effective_max_resident_bytes <= 0
        or prepared_chunk.resident_nbytes > effective_max_resident_bytes
    ):
        return prepared_chunk
    _touch_prepared_chunk(prepared_chunk)
    _PREPARED_CHUNK_CACHE[cache_key] = prepared_chunk
    _PREPARED_CHUNK_CACHE_RESIDENT_BYTES += prepared_chunk.resident_nbytes
    _trim_prepared_chunk_cache()
    return prepared_chunk


def _grouped_prepared_chunk_cache_key(
    pages_by_group: Sequence[Sequence[PreparedPageTorch]],
) -> tuple[tuple[tuple[int, int], ...], ...] | None:
    if not pages_by_group or not pages_by_group[0]:
        return None
    if pages_by_group[0][0].header.mode_default not in {"M0", "M2", "M4"}:
        return None
    page_count = len(pages_by_group[0])
    cache_key: list[tuple[tuple[int, int], ...]] = []
    for group_pages in pages_by_group:
        if len(group_pages) != page_count:
            return None
        group_key = _prepared_chunk_cache_key(group_pages)
        if group_key is None:
            return None
        cache_key.append(group_key)
    return tuple(cache_key)


def _build_grouped_prepared_chunk_mps(
    pages_by_group: Sequence[Sequence[PreparedPageTorch]],
) -> PreparedGroupedChunkMPS | None:
    if not pages_by_group or not pages_by_group[0]:
        return None
    torch = _load_torch()

    def _get_or_build_group_chunk(group_pages: Sequence[PreparedPageTorch]) -> PreparedChunkMPS | None:
        prepared_chunk = _get_prepared_chunk_mps(group_pages)
        if prepared_chunk is not None:
            return prepared_chunk
        cache_key = _prepared_chunk_cache_key(group_pages)
        if cache_key is None:
            return None
        return _build_prepared_chunk_mps(group_pages)
    header = pages_by_group[0][0].header
    device_type = pages_by_group[0][0].device_type
    if header.mode_default == "M2":
        prepared_chunks = [_get_or_build_group_chunk(group_pages) for group_pages in pages_by_group]
        if any(chunk is None for chunk in prepared_chunks):
            return None
        m2_sketch_groups = tuple(
            torch.stack([chunk.m2_sketch_groups[group_index] for chunk in prepared_chunks], dim=0).contiguous()
            for group_index in range(header.num_groups)
        )
        m2_basis_groups = tuple(
            torch.stack([chunk.m2_basis_groups[group_index] for chunk in prepared_chunks], dim=0).contiguous()
            for group_index in range(header.num_groups)
        )
        m2_mean_groups = tuple(
            torch.stack([chunk.m2_mean_groups[group_index] for chunk in prepared_chunks], dim=0).contiguous()
            for group_index in range(header.num_groups)
        )
        m2_sketch_tensor = torch.stack(m2_sketch_groups, dim=3).contiguous()
        m2_basis_tensor = torch.stack(m2_basis_groups, dim=2).contiguous()
        m2_mean_tensor = torch.stack(m2_mean_groups, dim=2).contiguous()
        resident_nbytes = sum(int(tensor.numel() * tensor.element_size()) for tensor in m2_sketch_groups)
        resident_nbytes += sum(int(tensor.numel() * tensor.element_size()) for tensor in m2_basis_groups)
        resident_nbytes += sum(int(tensor.numel() * tensor.element_size()) for tensor in m2_mean_groups)
        resident_nbytes += int(m2_sketch_tensor.numel() * m2_sketch_tensor.element_size())
        resident_nbytes += int(m2_basis_tensor.numel() * m2_basis_tensor.element_size())
        resident_nbytes += int(m2_mean_tensor.numel() * m2_mean_tensor.element_size())
        return PreparedGroupedChunkMPS(
            header=header,
            payload_groups=(),
            codes_groups=None,
            scales_groups=None,
            bias_groups=None,
            m2_sketch_groups=m2_sketch_groups,
            m2_basis_groups=m2_basis_groups,
            m2_mean_groups=m2_mean_groups,
            m2_segment_ids=prepared_chunks[0].m2_segment_ids,
            m2_sketch_tensor=m2_sketch_tensor,
            m2_basis_tensor=m2_basis_tensor,
            m2_mean_tensor=m2_mean_tensor,
            payload_groups_tensor=None,
            scales_groups_tensor=None,
            bias_groups_tensor=None,
            fused_scaled_codes=None,
            resident_nbytes=resident_nbytes,
        )
    if header.mode_default == "M4":
        prepared_chunks = [_get_or_build_group_chunk(group_pages) for group_pages in pages_by_group]
        if any(chunk is None for chunk in prepared_chunks):
            return None
        shared_basis = prepared_chunks[0].m2_basis_groups is not None and int(prepared_chunks[0].m2_basis_groups[0].dim()) == 2
        m2_basis_groups = (
            tuple(
                torch.stack([chunk.m2_basis_groups[group_index] for chunk in prepared_chunks], dim=0).contiguous()
                for group_index in range(header.num_groups)
            )
            if prepared_chunks[0].m2_basis_groups is not None
            else None
        )
        m2_sketch_groups = tuple(
            torch.stack([chunk.m2_sketch_groups[group_index] for chunk in prepared_chunks], dim=0).contiguous()
            for group_index in range(header.num_groups)
        )
        m2_mean_groups = tuple(
            torch.stack([chunk.m2_mean_groups[group_index] for chunk in prepared_chunks], dim=0).contiguous()
            for group_index in range(header.num_groups)
        )
        m2_sketch_tensor = torch.stack(m2_sketch_groups, dim=3).contiguous()
        m2_basis_tensor = (
            torch.stack(m2_basis_groups, dim=1).contiguous()
            if m2_basis_groups is not None and shared_basis
            else (
                torch.stack(m2_basis_groups, dim=2).contiguous()
                if m2_basis_groups is not None
                else None
            )
        )
        m2_mean_tensor = torch.stack(m2_mean_groups, dim=2).contiguous()
        resident_nbytes = sum(int(tensor.numel() * tensor.element_size()) for tensor in m2_sketch_groups)
        if m2_basis_groups is not None:
            resident_nbytes += sum(int(tensor.numel() * tensor.element_size()) for tensor in m2_basis_groups)
        resident_nbytes += sum(int(tensor.numel() * tensor.element_size()) for tensor in m2_mean_groups)
        resident_nbytes += int(m2_sketch_tensor.numel() * m2_sketch_tensor.element_size())
        if m2_basis_tensor is not None:
            resident_nbytes += int(m2_basis_tensor.numel() * m2_basis_tensor.element_size())
        resident_nbytes += int(m2_mean_tensor.numel() * m2_mean_tensor.element_size())
        return PreparedGroupedChunkMPS(
            header=header,
            payload_groups=(),
            codes_groups=None,
            scales_groups=None,
            bias_groups=None,
            m2_sketch_groups=m2_sketch_groups,
            m2_basis_groups=m2_basis_groups,
            m2_mean_groups=m2_mean_groups,
            m2_segment_ids=None,
            m2_sketch_tensor=m2_sketch_tensor,
            m2_basis_tensor=m2_basis_tensor,
            m2_mean_tensor=m2_mean_tensor,
            resident_nbytes=resident_nbytes,
            payload_groups_tensor=None,
            scales_groups_tensor=None,
            bias_groups_tensor=None,
            fused_scaled_codes=None,
        )
    if _supports_packed_four_group128_cuda(header, device_type=device_type):
        metadata_dtype = _m0_affine_metadata_dtype(device_type=device_type)
        payload_groups_tensor = torch.stack(
            [
                torch.stack(
                    [
                        torch.stack([page.payload[group_index] for page in group_pages], dim=0)
                        for group_pages in pages_by_group
                    ],
                    dim=0,
                )
                for group_index in range(header.num_groups)
            ],
            dim=1,
        ).contiguous()
        scales_groups_tensor = torch.stack(
            [
                torch.stack(
                    [
                        torch.stack([page.scales[:, group_index].to(dtype=metadata_dtype) for page in group_pages], dim=0)
                        for group_pages in pages_by_group
                    ],
                    dim=0,
                )
                for group_index in range(header.num_groups)
            ],
            dim=1,
        ).contiguous()
        bias_groups_tensor = torch.stack(
            [
                torch.stack(
                    [
                        torch.stack([page.bias[:, group_index].to(dtype=metadata_dtype) for page in group_pages], dim=0)
                        for group_pages in pages_by_group
                    ],
                    dim=0,
                )
                for group_index in range(header.num_groups)
            ],
            dim=1,
        ).contiguous()
        payload_groups = tuple(payload_groups_tensor[:, group_index] for group_index in range(header.num_groups))
        scales_groups = tuple(scales_groups_tensor[:, group_index] for group_index in range(header.num_groups))
        bias_groups = tuple(bias_groups_tensor[:, group_index] for group_index in range(header.num_groups))
        resident_nbytes = int(payload_groups_tensor.numel() * payload_groups_tensor.element_size())
        resident_nbytes += int(scales_groups_tensor.numel() * scales_groups_tensor.element_size())
        resident_nbytes += int(bias_groups_tensor.numel() * bias_groups_tensor.element_size())
        return PreparedGroupedChunkMPS(
            header=header,
            payload_groups=payload_groups,
            codes_groups=None,
            scales_groups=scales_groups,
            bias_groups=bias_groups,
            m2_sketch_groups=None,
            m2_basis_groups=None,
            m2_mean_groups=None,
            m2_segment_ids=None,
            m2_sketch_tensor=None,
            m2_basis_tensor=None,
            m2_mean_tensor=None,
            payload_groups_tensor=payload_groups_tensor,
            scales_groups_tensor=scales_groups_tensor,
            bias_groups_tensor=bias_groups_tensor,
            fused_scaled_codes=None,
            resident_nbytes=resident_nbytes,
        )
    prepared_chunks = [_get_or_build_group_chunk(group_pages) for group_pages in pages_by_group]
    if any(chunk is None for chunk in prepared_chunks):
        return None
    if (
        _supports_grouped_fused_only_cache(header, device_type=device_type)
        and all(chunk.fused_scaled_codes is not None for chunk in prepared_chunks)
        and all(chunk.bias_groups is not None for chunk in prepared_chunks)
    ):
        fused_scaled_codes = torch.stack([chunk.fused_scaled_codes for chunk in prepared_chunks], dim=0).contiguous()
        bias_groups = tuple(
            torch.stack([chunk.bias_groups[group_index] for chunk in prepared_chunks], dim=0).contiguous()
            for group_index in range(header.num_groups)
        )
        resident_nbytes = int(fused_scaled_codes.numel() * fused_scaled_codes.element_size())
        resident_nbytes += sum(int(tensor.numel() * tensor.element_size()) for tensor in bias_groups)
        return PreparedGroupedChunkMPS(
            header=header,
            payload_groups=(),
            codes_groups=None,
            scales_groups=None,
            bias_groups=bias_groups,
            m2_sketch_groups=None,
            m2_basis_groups=None,
            m2_mean_groups=None,
            m2_segment_ids=None,
            m2_sketch_tensor=None,
            m2_basis_tensor=None,
            m2_mean_tensor=None,
            payload_groups_tensor=None,
            scales_groups_tensor=None,
            bias_groups_tensor=None,
            fused_scaled_codes=fused_scaled_codes,
            resident_nbytes=resident_nbytes,
        )
    # Grouped decode uses unpacked codes/scales/bias directly when no fused-only cache
    # is available, so duplicating stacked payload tensors here only burns memory
    # without helping the hot path.
    payload_groups: tuple[Any, ...] = ()
    codes_groups = tuple(
        torch.stack([chunk.codes_groups[group_index] for chunk in prepared_chunks], dim=0)
        for group_index in range(header.num_groups)
    )
    scales_groups = tuple(
        torch.stack([chunk.scales_groups[group_index] for chunk in prepared_chunks], dim=0)
        for group_index in range(header.num_groups)
    )
    bias_groups = tuple(
        torch.stack([chunk.bias_groups[group_index] for chunk in prepared_chunks], dim=0)
        for group_index in range(header.num_groups)
    )
    fused_scaled_codes = None
    if all(chunk.fused_scaled_codes is not None for chunk in prepared_chunks):
        fused_scaled_codes = torch.stack([chunk.fused_scaled_codes for chunk in prepared_chunks], dim=0)
    resident_nbytes = sum(int(tensor.numel() * tensor.element_size()) for tensor in payload_groups)
    resident_nbytes += sum(int(tensor.numel() * tensor.element_size()) for tensor in codes_groups)
    resident_nbytes += sum(int(tensor.numel() * tensor.element_size()) for tensor in scales_groups)
    resident_nbytes += sum(int(tensor.numel() * tensor.element_size()) for tensor in bias_groups)
    if fused_scaled_codes is not None:
        resident_nbytes += int(fused_scaled_codes.numel() * fused_scaled_codes.element_size())
    return PreparedGroupedChunkMPS(
        header=header,
        payload_groups=payload_groups,
        codes_groups=codes_groups,
        scales_groups=scales_groups,
        bias_groups=bias_groups,
        m2_sketch_groups=None,
        m2_basis_groups=None,
        m2_mean_groups=None,
        m2_segment_ids=None,
        m2_sketch_tensor=None,
        m2_basis_tensor=None,
        m2_mean_tensor=None,
        payload_groups_tensor=None,
        scales_groups_tensor=None,
        bias_groups_tensor=None,
        fused_scaled_codes=fused_scaled_codes,
        resident_nbytes=resident_nbytes,
    )


def _assemble_grouped_fused_two_group64_components(
    prepared_chunks: Sequence[PreparedChunkMPS],
    *,
    trace: ExecutionTrace | None,
    device_type: TorchDevice,
):
    fused_scaled_codes = _trace_timed_call(
        trace,
        "chunk_assembly",
        device_type=device_type,
        fn=lambda: _load_torch().stack([chunk.fused_scaled_codes for chunk in prepared_chunks], dim=0),
    )
    bias_groups = tuple(
        _trace_timed_call(
            trace,
            "chunk_assembly",
            device_type=device_type,
            fn=lambda group_index=group_index: _load_torch().stack([chunk.bias_groups[group_index] for chunk in prepared_chunks], dim=0),
        )
        for group_index in range(2)
    )
    if trace is not None:
        trace.record_temporary(int(fused_scaled_codes.numel() * fused_scaled_codes.element_size()))
        trace.record_temporary(sum(int(tensor.numel() * tensor.element_size()) for tensor in bias_groups))
    return fused_scaled_codes, bias_groups


def _get_grouped_prepared_chunk_mps(
    pages_by_group: Sequence[Sequence[PreparedPageTorch]],
) -> PreparedGroupedChunkMPS | None:
    global _PREPARED_GROUPED_CHUNK_CACHE_RESIDENT_BYTES
    cache_key = _grouped_prepared_chunk_cache_key(pages_by_group)
    if cache_key is None:
        return None
    if pages_by_group[0][0].header.kind not in _PREPARED_CHUNK_CACHE_KINDS:
        return None
    total_page_count = sum(len(group_pages) for group_pages in pages_by_group)
    if total_page_count < _MIN_PREPARED_CHUNK_CACHE_PAGE_COUNT:
        return None
    cached_chunk = _PREPARED_GROUPED_CHUNK_CACHE.get(cache_key)
    if cached_chunk is not None:
        _touch_prepared_chunk(cached_chunk)
        _PREPARED_GROUPED_CHUNK_CACHE.move_to_end(cache_key)
        return cached_chunk
    prepared_chunk = _build_grouped_prepared_chunk_mps(pages_by_group)
    if prepared_chunk is None:
        return None
    effective_max_resident_bytes = _effective_max_prepared_chunk_cache_resident_bytes()
    if (
        _MAX_PREPARED_CHUNK_CACHE_ENTRIES <= 0
        or effective_max_resident_bytes <= 0
        or prepared_chunk.resident_nbytes > effective_max_resident_bytes
    ):
        return prepared_chunk
    _touch_prepared_chunk(prepared_chunk)
    _PREPARED_GROUPED_CHUNK_CACHE[cache_key] = prepared_chunk
    _PREPARED_GROUPED_CHUNK_CACHE_RESIDENT_BYTES += prepared_chunk.resident_nbytes
    _trim_prepared_chunk_cache()
    return prepared_chunk


def page_supported_torch(page: EncodedPage | PreparedPageTorch) -> bool:
    source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
    header = source_page.header
    if header.layout != "group_major":
        return False
    if int(header.group_size) <= 0 or int(header.num_groups) <= 0:
        return False
    if int(header.group_size) * int(header.num_groups) != int(header.padded_head_dim):
        return False
    if header.mode_default == "M3":
        if source_page.escape_payload is None:
            return False
        if header.escape_dtype == "int8":
            return source_page.escape_scales is not None
        return True
    if header.mode_default == "M2":
        return (
            header.kind == "K"
            and header.quant_scheme == "sketch"
            and source_page.m2_sketch is not None
            and source_page.m2_basis is not None
            and source_page.m2_mean is not None
        )
    if header.mode_default == "M4":
        return (
            header.kind == "K"
            and header.quant_scheme == "project"
            and source_page.m2_sketch is not None
            and source_page.m2_mean is not None
        )
    if header.mode_default == "T3":
        return (
            header.quant_scheme == "turbo3"
            and header.bits == 3
            and header.group_size in (32, 64)
            and source_page.payload is not None
            and source_page.scales is not None
            and source_page.codebooks is not None
        )
    return (
        source_page.payload is not None
        and (
            (
                header.mode_default == "M0"
                and header.bits in (2, 3, 4, 8)
                and header.quant_scheme == "affine"
                and source_page.scales is not None
                and source_page.bias is not None
            )
            or (
                header.mode_default == "M1"
                and header.bits in (2, 4)
                and header.quant_scheme == "lut"
                and source_page.codebooks is not None
            )
        )
    )


def page_supported_mps(page: EncodedPage | PreparedPageTorch) -> bool:
    return page_supported_torch(page)


def _unpack_metadata(bits: int, *, device_type: TorchDevice):
    cache_key = (device_type, bits)
    cached = _UNPACK_METADATA.get(cache_key)
    if cached is not None:
        return cached
    torch = _load_torch()
    symbols_per_word = 32 // bits
    shifts = torch.arange(symbols_per_word, dtype=torch.int32, device=device_type) * bits
    mask = torch.tensor((1 << bits) - 1, dtype=torch.int32, device=device_type)
    _UNPACK_METADATA[cache_key] = (shifts, mask)
    return shifts, mask


def _spill_unpack_metadata(bits: int, group_size: int, *, device_type: TorchDevice):
    cache_key = (device_type, bits, group_size)
    cached = _SPILL_UNPACK_METADATA.get(cache_key)
    if cached is not None:
        return cached
    torch = _load_torch()
    bit_offsets = np.arange(group_size, dtype=np.int64) * int(bits)
    word_count = words_per_group(group_size, bits)
    word_indices = torch.as_tensor(bit_offsets // 32, dtype=torch.int64, device=device_type)
    next_word_indices = torch.as_tensor(
        np.minimum(bit_offsets // 32 + 1, word_count - 1),
        dtype=torch.int64,
        device=device_type,
    )
    bit_indices = torch.as_tensor(bit_offsets % 32, dtype=torch.int64, device=device_type)
    spill_width = np.maximum((bit_offsets % 32) + int(bits) - 32, 0).astype(np.int64)
    spill_mask = torch.as_tensor((1 << spill_width) - 1, dtype=torch.int64, device=device_type)
    shift_back = torch.as_tensor(int(bits) - spill_width, dtype=torch.int64, device=device_type)
    spill_flags = torch.as_tensor(spill_width > 0, dtype=torch.bool, device=device_type)
    _SPILL_UNPACK_METADATA[cache_key] = (
        word_indices,
        next_word_indices,
        bit_indices,
        spill_mask,
        shift_back,
        spill_flags,
    )
    return _SPILL_UNPACK_METADATA[cache_key]


def _turbo3_centroids_torch(*, device_type: TorchDevice):
    cached = _TURBO3_CENTROID_TENSORS.get(device_type)
    if cached is not None:
        return cached
    tensor = _device_tensor(TURBO3_CENTROIDS.astype(np.float32, copy=False), device=device_type)
    tensor = tensor.to(dtype=_load_torch().float32)
    _TURBO3_CENTROID_TENSORS[device_type] = tensor
    return tensor


def _fwht_matrix_torch(width: int, *, device_type: TorchDevice):
    cache_key = (device_type, int(width))
    cached = _FWHT_MATRICES.get(cache_key)
    if cached is not None:
        return cached
    if width <= 0 or (width & (width - 1)):
        raise ValueError("FWHT requires the last dimension to be a power of two")
    basis = np.eye(width, dtype=np.float32)
    transformed = fwht_last_dim(basis)
    tensor = _device_tensor(transformed, device=device_type).to(dtype=_load_torch().float32)
    _FWHT_MATRICES[cache_key] = tensor
    return tensor


def _m4_basis_torch(group_size: int, rank: int, *, basis_family: str, device_type: TorchDevice):
    cache_key = (device_type, int(group_size), int(rank), basis_family)
    cached = _M4_BASIS_TENSORS.get(cache_key)
    if cached is not None:
        return cached
    basis = _device_tensor(
        fixed_project_basis(int(group_size), int(rank), basis_family),
        device=device_type,
    ).to(dtype=_load_torch().float32)
    _M4_BASIS_TENSORS[cache_key] = basis
    return basis


def _synchronize_torch_device(device_type: TorchDevice) -> None:
    torch = _load_torch()
    if device_type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
        return
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _trace_timed_call(
    trace: ExecutionTrace | None,
    section: str,
    *,
    device_type: TorchDevice,
    fn,
    synchronize: bool = True,
):
    if trace is None or not trace.capture_timings:
        return fn()
    if synchronize:
        _synchronize_torch_device(device_type)
    start = time.perf_counter()
    result = fn()
    if synchronize:
        _synchronize_torch_device(device_type)
    trace.record_timing(section, (time.perf_counter() - start) * 1000.0)
    return result


def _prepared_page_host_nbytes(page: EncodedPage) -> int:
    total = 0
    if page.payload is not None:
        total += int(page.payload.nbytes)
    if page.scales is not None:
        total += int(page.scales.nbytes)
    if page.bias is not None:
        total += int(page.bias.nbytes)
    if page.codebooks is not None:
        total += int(page.codebooks.nbytes)
    if page.m2_sketch is not None:
        total += int(page.m2_sketch.nbytes)
    if page.m2_basis is not None:
        total += int(page.m2_basis.nbytes)
    if page.m2_mean is not None:
        total += int(page.m2_mean.nbytes)
    if page.escape_payload is not None:
        total += int(page.escape_payload.nbytes)
    if page.escape_scales is not None:
        total += int(page.escape_scales.nbytes)
    return total


def _decode_escape_batch_torch(
    pages: Sequence[PreparedPageTorch],
    *,
    token_count: int,
    head_dim: int,
    promote_float32: bool = True,
):
    torch = _load_torch()
    target_dtype = torch.float32 if promote_float32 else None
    prepared_chunk = _get_prepared_chunk_mps(pages)
    if prepared_chunk is not None and prepared_chunk.escape_payload_batch is not None:
        payload = prepared_chunk.escape_payload_batch[:, :token_count, :head_dim]
        if pages[0].header.escape_dtype == "int8":
            scales = prepared_chunk.escape_scales_batch[:, :token_count]
            result = payload if target_dtype is None else payload.to(dtype=target_dtype)
            scale_values = scales if target_dtype is None else scales.to(dtype=target_dtype)
            return result * scale_values[..., None]
        return payload if target_dtype is None else payload.to(dtype=target_dtype)
    if len(pages) == 1:
        payload = pages[0].escape_payload[:token_count, :head_dim]
        if pages[0].header.escape_dtype == "int8":
            scales = pages[0].escape_scales[:token_count]
            result = payload if target_dtype is None else payload.to(dtype=target_dtype)
            scale_values = scales if target_dtype is None else scales.to(dtype=target_dtype)
            return (result * scale_values[:, None]).unsqueeze(0)
        return (payload if target_dtype is None else payload.to(dtype=target_dtype)).unsqueeze(0)
    payload = torch.stack([page.escape_payload[:token_count, :head_dim] for page in pages], dim=0)
    if pages[0].header.escape_dtype == "int8":
        scales = torch.stack([page.escape_scales[:token_count] for page in pages], dim=0)
        result = payload if target_dtype is None else payload.to(dtype=target_dtype)
        scale_values = scales if target_dtype is None else scales.to(dtype=target_dtype)
        return result * scale_values[..., None]
    return payload if target_dtype is None else payload.to(dtype=target_dtype)


def _m3_native_compute_enabled(pages: Sequence[PreparedPageTorch]) -> bool:
    if not pages:
        return False
    header = pages[0].header
    if pages[0].device_type != "cuda":
        return False
    return header.mode_default == "M3" and header.escape_dtype == "float16"


def _optional_m2_sidecar_batches(
    pages: Sequence[EncodedPage],
    *,
    device_type: TorchDevice,
) -> tuple[Any | None, Any | None, Any | None, int, int]:
    if not pages or not all(page.m2_sketch is not None and page.m2_basis is not None and page.m2_mean is not None for page in pages):
        return None, None, None, 0, 0
    sketch_array = np.stack([np.asarray(page.m2_sketch) for page in pages], axis=0)
    basis_array = np.stack([np.asarray(page.m2_basis) for page in pages], axis=0)
    mean_array = np.stack([np.asarray(page.m2_mean) for page in pages], axis=0)
    sketch_batch = _device_tensor(sketch_array, device=device_type)
    basis_batch = _device_tensor(basis_array, device=device_type)
    mean_batch = _device_tensor(mean_array, device=device_type)
    if device_type == "mps":
        sketch_batch = sketch_batch.to(dtype=_load_torch().float32)
        basis_batch = basis_batch.to(dtype=_load_torch().float32)
        mean_batch = mean_batch.to(dtype=_load_torch().float32)
    return sketch_batch, basis_batch, mean_batch, int(sketch_array.nbytes + basis_array.nbytes + mean_array.nbytes), int(
        sketch_batch.numel() * sketch_batch.element_size()
        + basis_batch.numel() * basis_batch.element_size()
        + mean_batch.numel() * mean_batch.element_size()
    )


def _prepare_page_chunk_torch(
    pages: Sequence[EncodedPage],
    *,
    device_type: TorchDevice,
    trace: ExecutionTrace | None = None,
) -> list[PreparedPageTorch]:
    if not pages:
        return []
    header = pages[0].header
    total_host_to_device_nbytes = 0

    if header.mode_default == "M3":
        escape_array = np.stack([np.asarray(page.escape_payload) for page in pages], axis=0)
        escape_batch = _device_tensor(escape_array, device=device_type)
        total_host_to_device_nbytes += int(escape_batch.numel() * escape_batch.element_size())
        escape_scale_batch = None
        if header.escape_dtype == "int8":
            escape_scale_array = np.stack([np.asarray(page.escape_scales) for page in pages], axis=0)
            escape_scale_batch = _device_tensor(escape_scale_array, device=device_type)
            escape_scale_batch = escape_scale_batch.to(dtype=_escape_scale_dtype(device_type=device_type))
            total_host_to_device_nbytes += int(escape_scale_batch.numel() * escape_scale_batch.element_size())
        prepared_pages = [
            PreparedPageTorch(
                device_type=device_type,
                source_page=page,
                header=page.header,
                escape_payload=escape_batch[index],
                escape_scales=None if escape_scale_batch is None else escape_scale_batch[index],
                host_to_device_nbytes=_prepared_page_host_nbytes(page),
                resident_nbytes=int(escape_batch[index].numel() * escape_batch[index].element_size())
                + (
                    0
                    if escape_scale_batch is None
                    else int(escape_scale_batch[index].numel() * escape_scale_batch[index].element_size())
                ),
                cache_uid=_next_prepared_page_uid(),
            )
            for index, page in enumerate(pages)
        ]
        if trace is not None:
            trace.record_host_to_device(total_host_to_device_nbytes)
        return prepared_pages

    if header.mode_default == "M2":
        sketch_array = np.stack([np.asarray(page.m2_sketch) for page in pages], axis=0)
        basis_array = np.stack([np.asarray(page.m2_basis) for page in pages], axis=0)
        mean_array = np.stack([np.asarray(page.m2_mean) for page in pages], axis=0)
        sketch_batch = _device_tensor(sketch_array, device=device_type)
        basis_batch = _device_tensor(basis_array, device=device_type)
        mean_batch = _device_tensor(mean_array, device=device_type)
        total_host_to_device_nbytes += int(sketch_array.nbytes)
        total_host_to_device_nbytes += int(basis_array.nbytes)
        total_host_to_device_nbytes += int(mean_array.nbytes)
        if device_type == "mps":
            sketch_batch = sketch_batch.to(dtype=_load_torch().float32)
            basis_batch = basis_batch.to(dtype=_load_torch().float32)
            mean_batch = mean_batch.to(dtype=_load_torch().float32)
        prepared_pages = [
            PreparedPageTorch(
                device_type=device_type,
                source_page=page,
                header=page.header,
                m2_sketch=sketch_batch[index],
                m2_basis=basis_batch[index],
                m2_mean=mean_batch[index],
                host_to_device_nbytes=_prepared_page_host_nbytes(page),
                resident_nbytes=(
                    int(sketch_batch[index].numel() * sketch_batch[index].element_size())
                    + int(basis_batch[index].numel() * basis_batch[index].element_size())
                    + int(mean_batch[index].numel() * mean_batch[index].element_size())
                ),
                cache_uid=_next_prepared_page_uid(),
            )
            for index, page in enumerate(pages)
        ]
        if trace is not None:
            trace.record_host_to_device(total_host_to_device_nbytes)
        return prepared_pages

    if header.mode_default == "M4":
        sketch_array = np.stack([np.asarray(page.m2_sketch) for page in pages], axis=0)
        shared_basis = pages[0].header.project_basis == "svd_shared"
        basis_array = None
        if not shared_basis and pages[0].m2_basis is not None:
            basis_array = np.stack([np.asarray(page.m2_basis) for page in pages], axis=0)
        mean_array = np.stack([np.asarray(page.m2_mean) for page in pages], axis=0)
        sketch_batch = _device_tensor(sketch_array, device=device_type)
        basis_batch = None if basis_array is None else _device_tensor(basis_array, device=device_type)
        mean_batch = _device_tensor(mean_array, device=device_type)
        total_host_to_device_nbytes += int(sketch_array.nbytes)
        if basis_array is not None:
            total_host_to_device_nbytes += int(basis_array.nbytes)
        total_host_to_device_nbytes += int(mean_array.nbytes)
        if device_type == "mps":
            sketch_batch = sketch_batch.to(dtype=_load_torch().float32)
            if basis_batch is not None:
                basis_batch = basis_batch.to(dtype=_load_torch().float32)
            mean_batch = mean_batch.to(dtype=_load_torch().float32)
        prepared_pages = [
            PreparedPageTorch(
                device_type=device_type,
                source_page=page,
                header=page.header,
                m2_sketch=sketch_batch[index],
                m2_basis=None if basis_batch is None else basis_batch[index],
                m2_mean=mean_batch[index],
                host_to_device_nbytes=_prepared_page_host_nbytes(page),
                resident_nbytes=(
                    int(sketch_batch[index].numel() * sketch_batch[index].element_size())
                    + (0 if basis_batch is None else int(basis_batch[index].numel() * basis_batch[index].element_size()))
                    + int(mean_batch[index].numel() * mean_batch[index].element_size())
                ),
                cache_uid=_next_prepared_page_uid(),
            )
            for index, page in enumerate(pages)
        ]
        if trace is not None:
            trace.record_host_to_device(total_host_to_device_nbytes)
        return prepared_pages

    if header.mode_default == "M1":
        payload_array = np.stack([np.asarray(page.payload, dtype=np.int32) for page in pages], axis=0)
        codebooks_array = np.stack([np.asarray(page.codebooks) for page in pages], axis=0)
        payload_batch = _device_tensor(payload_array, device=device_type)
        codebooks_batch = _device_tensor(codebooks_array, device=device_type)
        sidecar_sketch_batch, sidecar_basis_batch, sidecar_mean_batch, sidecar_h2d_nbytes, _ = _optional_m2_sidecar_batches(
            pages,
            device_type=device_type,
        )
        total_host_to_device_nbytes += int(payload_array.nbytes)
        total_host_to_device_nbytes += int(codebooks_array.nbytes)
        total_host_to_device_nbytes += sidecar_h2d_nbytes
        if device_type == "mps":
            codebooks_batch = codebooks_batch.to(dtype=_load_torch().float32)
        unpack_shifts, unpack_mask = _unpack_metadata(header.bits, device_type=device_type)
        prepared_pages = [
            PreparedPageTorch(
                device_type=device_type,
                source_page=page,
                header=page.header,
                payload=payload_batch[index],
                codebooks=codebooks_batch[index],
                m2_sketch=None if sidecar_sketch_batch is None else sidecar_sketch_batch[index],
                m2_basis=None if sidecar_basis_batch is None else sidecar_basis_batch[index],
                m2_mean=None if sidecar_mean_batch is None else sidecar_mean_batch[index],
                unpack_shifts=unpack_shifts,
                unpack_mask=unpack_mask,
                host_to_device_nbytes=_prepared_page_host_nbytes(page),
                resident_nbytes=(
                    int(payload_batch[index].numel() * payload_batch[index].element_size())
                    + int(codebooks_batch[index].numel() * codebooks_batch[index].element_size())
                    + (
                        0
                        if sidecar_sketch_batch is None or sidecar_basis_batch is None or sidecar_mean_batch is None
                        else int(sidecar_sketch_batch[index].numel() * sidecar_sketch_batch[index].element_size())
                        + int(sidecar_basis_batch[index].numel() * sidecar_basis_batch[index].element_size())
                        + int(sidecar_mean_batch[index].numel() * sidecar_mean_batch[index].element_size())
                    )
                ),
                cache_uid=_next_prepared_page_uid(),
            )
            for index, page in enumerate(pages)
        ]
        if trace is not None:
            trace.record_host_to_device(total_host_to_device_nbytes)
        return prepared_pages

    if header.mode_default == "T3":
        payload_array = np.stack([np.asarray(page.payload, dtype=np.int32) for page in pages], axis=0)
        scales_array = np.stack([np.asarray(page.scales) for page in pages], axis=0)
        payload_batch = _device_tensor(payload_array, device=device_type)
        scales_batch = _device_tensor(scales_array, device=device_type)
        sidecar_sketch_batch, sidecar_basis_batch, sidecar_mean_batch, sidecar_h2d_nbytes, _ = _optional_m2_sidecar_batches(
            pages,
            device_type=device_type,
        )
        total_host_to_device_nbytes += int(payload_array.nbytes)
        total_host_to_device_nbytes += int(scales_array.nbytes)
        total_host_to_device_nbytes += sidecar_h2d_nbytes
        if device_type == "mps":
            scales_batch = scales_batch.to(dtype=_load_torch().float32)
        codebooks_tensor = _turbo3_centroids_torch(device_type=device_type)
        unpack_shifts, unpack_mask = _unpack_metadata(header.bits, device_type=device_type)
        prepared_pages = [
            PreparedPageTorch(
                device_type=device_type,
                source_page=page,
                header=page.header,
                payload=payload_batch[index],
                scales=scales_batch[index],
                codebooks=codebooks_tensor,
                m2_sketch=None if sidecar_sketch_batch is None else sidecar_sketch_batch[index],
                m2_basis=None if sidecar_basis_batch is None else sidecar_basis_batch[index],
                m2_mean=None if sidecar_mean_batch is None else sidecar_mean_batch[index],
                unpack_shifts=unpack_shifts,
                unpack_mask=unpack_mask,
                host_to_device_nbytes=_prepared_page_host_nbytes(page),
                resident_nbytes=(
                    int(payload_batch[index].numel() * payload_batch[index].element_size())
                    + int(scales_batch[index].numel() * scales_batch[index].element_size())
                    + (
                        0
                        if sidecar_sketch_batch is None or sidecar_basis_batch is None or sidecar_mean_batch is None
                        else int(sidecar_sketch_batch[index].numel() * sidecar_sketch_batch[index].element_size())
                        + int(sidecar_basis_batch[index].numel() * sidecar_basis_batch[index].element_size())
                        + int(sidecar_mean_batch[index].numel() * sidecar_mean_batch[index].element_size())
                    )
                ),
                cache_uid=_next_prepared_page_uid(),
            )
            for index, page in enumerate(pages)
        ]
        if trace is not None:
            trace.record_host_to_device(total_host_to_device_nbytes)
        return prepared_pages

    payload_array = np.stack([np.asarray(page.payload, dtype=np.int32) for page in pages], axis=0)
    scales_array = np.stack([np.asarray(page.scales) for page in pages], axis=0)
    bias_array = np.stack([np.asarray(page.bias) for page in pages], axis=0)
    payload_batch = _device_tensor(payload_array, device=device_type)
    scales_batch = _device_tensor(scales_array, device=device_type)
    bias_batch = _device_tensor(bias_array, device=device_type)
    sidecar_sketch_batch, sidecar_basis_batch, sidecar_mean_batch, sidecar_h2d_nbytes, _ = _optional_m2_sidecar_batches(
        pages,
        device_type=device_type,
    )
    total_host_to_device_nbytes += int(payload_array.nbytes)
    total_host_to_device_nbytes += int(scales_array.nbytes)
    total_host_to_device_nbytes += int(bias_array.nbytes)
    total_host_to_device_nbytes += sidecar_h2d_nbytes
    metadata_dtype = _m0_affine_metadata_dtype(device_type=device_type)
    scales_batch = scales_batch.to(dtype=metadata_dtype)
    bias_batch = bias_batch.to(dtype=metadata_dtype)
    unpack_shifts, unpack_mask = _unpack_metadata(header.bits, device_type=device_type)

    prepared_pages = [
        PreparedPageTorch(
            device_type=device_type,
            source_page=page,
            header=page.header,
            payload=payload_batch[index],
            scales=scales_batch[index],
            bias=bias_batch[index],
            m2_sketch=None if sidecar_sketch_batch is None else sidecar_sketch_batch[index],
            m2_basis=None if sidecar_basis_batch is None else sidecar_basis_batch[index],
            m2_mean=None if sidecar_mean_batch is None else sidecar_mean_batch[index],
            unpack_shifts=unpack_shifts,
            unpack_mask=unpack_mask,
            host_to_device_nbytes=_prepared_page_host_nbytes(page),
            resident_nbytes=(
                int(payload_batch[index].numel() * payload_batch[index].element_size())
                + int(scales_batch[index].numel() * scales_batch[index].element_size())
                + int(bias_batch[index].numel() * bias_batch[index].element_size())
                + (
                    0
                    if sidecar_sketch_batch is None or sidecar_basis_batch is None or sidecar_mean_batch is None
                    else int(sidecar_sketch_batch[index].numel() * sidecar_sketch_batch[index].element_size())
                    + int(sidecar_basis_batch[index].numel() * sidecar_basis_batch[index].element_size())
                    + int(sidecar_mean_batch[index].numel() * sidecar_mean_batch[index].element_size())
                )
            ),
            cache_uid=_next_prepared_page_uid(),
        )
        for index, page in enumerate(pages)
    ]
    if trace is not None:
        trace.record_host_to_device(total_host_to_device_nbytes)
    return prepared_pages


def prepare_pages_torch(
    pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    device_type: TorchDevice,
    trace: ExecutionTrace | None = None,
) -> list[PreparedPageTorch]:
    backend_name = _backend_name(device_type)
    if not torch_device_available(device_type):
        raise RuntimeError(f"{backend_name} is unavailable on this machine")
    prepared_pages: list[PreparedPageTorch] = []
    for page_chunk in _chunk_compatible_source_pages(pages, device_type=device_type):
        if all(isinstance(page, PreparedPageTorch) and page.device_type == device_type for page in page_chunk):
            prepared_pages.extend(page_chunk)  # type: ignore[arg-type]
            continue
        source_pages = []
        for page in page_chunk:
            source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
            if not page_supported_torch(source_page):
                raise ValueError(f"page is unsupported by {backend_name} in this phase")
            source_pages.append(source_page)
        if source_pages:
            prepared_pages.extend(
                _trace_timed_call(
                    trace,
                    "prepare",
                    device_type=device_type,
                    fn=lambda source_pages=source_pages: _prepare_page_chunk_torch(
                        source_pages,
                        device_type=device_type,
                        trace=trace,
                    ),
                )
            )
    return prepared_pages


def prepare_page_torch(
    page: EncodedPage | PreparedPageTorch,
    *,
    device_type: TorchDevice,
    trace: ExecutionTrace | None = None,
) -> PreparedPageTorch:
    if isinstance(page, PreparedPageTorch) and page.device_type == device_type:
        return page
    source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
    return prepare_pages_torch([source_page], device_type=device_type, trace=trace)[0]


def prepare_m0_affine_pages_from_tensor_torch(
    values,
    *,
    config,
    kind: str,
    layer_id: int,
    kv_head_id: int,
    token_start: int,
    device_type: TorchDevice,
    build_runtime_metadata: bool = False,
):
    torch = _load_torch()
    if not torch.is_tensor(values):
        raise TypeError("values must be a torch.Tensor")
    if int(values.ndim) != 3:
        raise ValueError("values must have shape [page_count, token_count, head_dim]")
    if int(values.shape[2]) != int(config.head_dim):
        raise ValueError("values head_dim must match config.head_dim")

    bits = config.bits_k if kind == "K" else config.bits_v
    default_mode = config.default_mode_k if kind == "K" else config.default_mode_v
    quant_scheme = config.quant_scheme_k if kind == "K" else config.quant_scheme_v
    layout = config.payload_layout_k if kind == "K" else config.payload_layout_v
    if default_mode != "M0" or quant_scheme != "affine" or layout != "group_major":
        raise ValueError("direct torch preparation only supports exact M0 affine group_major pages")

    page_count = int(values.shape[0])
    token_count = int(values.shape[1])
    num_groups = int(config.num_groups)
    group_size = int(config.group_size)
    padded_head_dim = int(config.padded_head_dim)
    qmax = float((1 << int(bits)) - 1)
    eps = 1e-8

    values_device = values.to(device=device_type)
    work_dtype = values_device.dtype if values_device.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
    values_work = values_device.to(dtype=work_dtype)
    if padded_head_dim > int(config.head_dim):
        padded = torch.nn.functional.pad(values_work, (0, padded_head_dim - int(config.head_dim)))
    else:
        padded = values_work
    grouped = padded.reshape(page_count, token_count, num_groups, group_size)
    x_min, x_max = torch.aminmax(grouped, dim=-1)
    scales = torch.clamp(((x_max - x_min).to(dtype=torch.float32) / max(qmax, 1.0)), min=eps).to(dtype=grouped.dtype)
    shifted = (grouped - x_min.unsqueeze(-1)) / scales.unsqueeze(-1)
    codes = torch.clamp(torch.round(shifted), 0.0, qmax).to(dtype=torch.int32)
    payload = _torch_pack_codes(codes, bits=int(bits), layout=layout)
    metadata_dtype = _m0_affine_metadata_dtype(device_type=device_type)
    scales_device = scales.to(dtype=metadata_dtype)
    bias_device = x_min.to(dtype=metadata_dtype)
    unpack_shifts, unpack_mask = _unpack_metadata(int(bits), device_type=device_type)

    prepared_pages: list[PreparedPageTorch] = []
    word_count = int(payload.shape[-1])
    for page_index in range(page_count):
        page_token_start = int(token_start + page_index * token_count)
        header = PageHeader(
            layer_id=layer_id,
            kv_head_id=kv_head_id,
            kind=kind,
            token_start=page_token_start,
            token_count=token_count,
            head_dim=int(config.head_dim),
            padded_head_dim=padded_head_dim,
            group_size=group_size,
            num_groups=num_groups,
            bits=int(bits),
            words_per_group=word_count,
            mode_default="M0",
            layout=layout,
            quant_scheme="affine",
            escape_dtype=config.escape_dtype,
        )
        source_page = EncodedPage(
            header=header,
            payload=np.zeros((num_groups, token_count, word_count), dtype=np.uint32),
            scales=np.zeros((token_count, num_groups), dtype=np.float16),
            bias=np.zeros((token_count, num_groups), dtype=np.float16),
            requested_mode="M0",
        )
        if build_runtime_metadata:
            page_values = values_work[page_index, :, : int(config.head_dim)].detach().cpu().numpy().astype(np.float32, copy=False)
            runtime_page_mean = page_values.mean(axis=0).astype(np.float32, copy=False)
            source_page.runtime_page_mean = runtime_page_mean
            source_page.runtime_page_sketch = runtime_page_mean[None, :]
            source_page.runtime_page_min = page_values.min(axis=0).astype(np.float32, copy=False)
            source_page.runtime_page_max = page_values.max(axis=0).astype(np.float32, copy=False)
        payload_page = payload[page_index]
        scales_page = scales_device[page_index]
        bias_page = bias_device[page_index]
        prepared_pages.append(
            PreparedPageTorch(
                device_type=device_type,
                source_page=source_page,
                header=header,
                payload=payload_page,
                scales=scales_page,
                bias=bias_page,
                unpack_shifts=unpack_shifts,
                unpack_mask=unpack_mask,
                host_to_device_nbytes=0,
                resident_nbytes=(
                    int(payload_page.numel() * payload_page.element_size())
                    + int(scales_page.numel() * scales_page.element_size())
                    + int(bias_page.numel() * bias_page.element_size())
                ),
                cache_uid=_next_prepared_page_uid(),
            )
        )
    return prepared_pages


def _pad_query(query_slice: np.ndarray | Any, padded_head_dim: int, *, device_type: TorchDevice):
    torch = _load_torch()
    if torch.is_tensor(query_slice):
        query = query_slice.to(dtype=torch.float32, device=device_type)
    else:
        query = torch.as_tensor(query_slice, dtype=torch.float32, device=device_type)
    if query.ndim != 1:
        raise ValueError("query_slice must have shape [head_dim]")
    if int(query.shape[0]) > padded_head_dim:
        raise ValueError("query head_dim exceeds padded_head_dim")
    if int(query.shape[0]) == padded_head_dim:
        return query
    padded = torch.zeros(padded_head_dim, dtype=torch.float32, device=device_type)
    padded[: query.shape[0]] = query
    return padded


def _pad_queries(query_slices: np.ndarray | Any, padded_head_dim: int, *, device_type: TorchDevice):
    torch = _load_torch()
    if torch.is_tensor(query_slices):
        queries = query_slices.to(dtype=torch.float32, device=device_type)
    else:
        queries = torch.as_tensor(query_slices, dtype=torch.float32, device=device_type)
    if queries.ndim != 2:
        raise ValueError("query_slices must have shape [query_count, head_dim]")
    if int(queries.shape[1]) > padded_head_dim:
        raise ValueError("query head_dim exceeds padded_head_dim")
    if int(queries.shape[1]) == padded_head_dim:
        return queries
    padded = torch.zeros((queries.shape[0], padded_head_dim), dtype=torch.float32, device=device_type)
    padded[:, : queries.shape[1]] = queries
    return padded


def _coerce_m2_operands(query_groups, group_sketch, group_basis, group_mean):
    torch = _load_torch()
    work_dtype = torch.promote_types(query_groups.dtype, group_sketch.dtype)
    work_dtype = torch.promote_types(work_dtype, group_basis.dtype)
    work_dtype = torch.promote_types(work_dtype, group_mean.dtype)
    if query_groups.dtype != work_dtype:
        query_groups = query_groups.to(dtype=work_dtype)
    if group_sketch.dtype != work_dtype:
        group_sketch = group_sketch.to(dtype=work_dtype)
    if group_basis.dtype != work_dtype:
        group_basis = group_basis.to(dtype=work_dtype)
    if group_mean.dtype != work_dtype:
        group_mean = group_mean.to(dtype=work_dtype)
    return query_groups, group_sketch, group_basis, group_mean


def _coerce_m2_grouped_operands(query_groups, sketch_tensor, basis_tensor, mean_tensor):
    torch = _load_torch()
    work_dtype = torch.promote_types(query_groups.dtype, sketch_tensor.dtype)
    work_dtype = torch.promote_types(work_dtype, basis_tensor.dtype)
    work_dtype = torch.promote_types(work_dtype, mean_tensor.dtype)
    if query_groups.dtype != work_dtype:
        query_groups = query_groups.to(dtype=work_dtype)
    if sketch_tensor.dtype != work_dtype:
        sketch_tensor = sketch_tensor.to(dtype=work_dtype)
    if basis_tensor.dtype != work_dtype:
        basis_tensor = basis_tensor.to(dtype=work_dtype)
    if mean_tensor.dtype != work_dtype:
        mean_tensor = mean_tensor.to(dtype=work_dtype)
    return query_groups, sketch_tensor, basis_tensor, mean_tensor


def _prepare_output_accumulator(out_acc: np.ndarray | None, head_dim: int, padded_head_dim: int, *, device_type: TorchDevice):
    torch = _load_torch()
    output = torch.zeros(padded_head_dim, dtype=torch.float32, device=device_type)
    if out_acc is None:
        return output
    values = torch.as_tensor(out_acc, dtype=torch.float32, device=device_type)
    if values.shape != (head_dim,):
        raise ValueError("out_acc must have shape [head_dim]")
    output[:head_dim] = values
    return output


def _prepare_output_accumulator_tensor(out_acc, head_dim: int, padded_head_dim: int, *, device_type: TorchDevice):
    torch = _load_torch()
    if out_acc is None:
        return torch.zeros(padded_head_dim, dtype=torch.float32, device=device_type)
    if isinstance(out_acc, np.ndarray):
        return _prepare_output_accumulator(out_acc, head_dim, padded_head_dim, device_type=device_type)
    if tuple(out_acc.shape) != (padded_head_dim,):
        raise ValueError("out_acc tensor must have shape [padded_head_dim]")
    return out_acc.to(dtype=torch.float32, device=device_type)


def _prepare_grouped_output_accumulator_tensor(
    out_acc,
    batch_size: int,
    query_count: int,
    head_dim: int,
    padded_head_dim: int,
    *,
    device_type: TorchDevice,
):
    torch = _load_torch()
    expected_shape = (batch_size, query_count, padded_head_dim)
    if out_acc is None:
        return torch.zeros(expected_shape, dtype=torch.float32, device=device_type)
    values = out_acc.to(dtype=torch.float32, device=device_type) if torch.is_tensor(out_acc) else torch.as_tensor(
        out_acc,
        dtype=torch.float32,
        device=device_type,
    )
    if tuple(values.shape) == expected_shape:
        return values
    if tuple(values.shape) != (batch_size, query_count, head_dim):
        raise ValueError("out_acc must have shape [batch_size, query_count, head_dim] or [batch_size, query_count, padded_head_dim]")
    output = torch.zeros(expected_shape, dtype=torch.float32, device=device_type)
    output[:, :, :head_dim] = values
    return output


def _unpack_bits_torch(words, shifts, mask, group_size: int, *, trace: ExecutionTrace | None = None):
    torch = _load_torch()
    if words.ndim != 2:
        raise ValueError("words must have shape [token_count, words_per_group]")
    if shifts is None or mask is None:
        raise ValueError("prepared torch pages require unpack metadata")
    device_type = str(words.device.type)

    def _impl():
        words_u64 = torch.bitwise_and(words.to(dtype=torch.int64), 0xFFFFFFFF)
        mask_i64 = torch.as_tensor(mask, dtype=torch.int64, device=words.device)
        bits = int(mask_i64.item()).bit_length()
        if 32 % bits == 0:
            shifts_i64 = shifts.to(dtype=torch.int64)
            expanded = torch.bitwise_and(torch.bitwise_right_shift(words_u64[..., None], shifts_i64), mask_i64)
            return expanded.reshape(words.shape[0], -1)[:, :group_size].to(torch.float32)
        if words.device.type == "mps":
            word_indices, next_word_indices, bit_indices, spill_mask, shift_back, spill_flags = _spill_unpack_metadata(
                bits,
                group_size,
                device_type=words.device.type,
            )
            bit_indices_2d = bit_indices.unsqueeze(0).expand(words_u64.shape[0], -1)
            spill_mask_2d = spill_mask.unsqueeze(0).expand(words_u64.shape[0], -1)
            shift_back_2d = shift_back.unsqueeze(0).expand(words_u64.shape[0], -1)
            current_words = words_u64[:, word_indices]
            next_words = words_u64[:, next_word_indices]
            values = torch.bitwise_right_shift(current_words, bit_indices_2d)
            spilled = torch.bitwise_left_shift(torch.bitwise_and(next_words, spill_mask_2d), shift_back_2d)
            values = torch.where(spill_flags.unsqueeze(0), torch.bitwise_or(values, spilled), values)
            return torch.bitwise_and(values, mask_i64).to(torch.float32)
        word_indices, next_word_indices, bit_indices, spill_mask, shift_back, spill_flags = _spill_unpack_metadata(
            bits,
            group_size,
            device_type=words.device.type,
        )
        gather_index = word_indices.unsqueeze(0).expand(words_u64.shape[0], -1)
        bit_indices_2d = bit_indices.unsqueeze(0).expand(words_u64.shape[0], -1)
        values = torch.bitwise_right_shift(torch.gather(words_u64, 1, gather_index), bit_indices_2d)
        if bool(spill_flags.any()):
            next_gather_index = next_word_indices.unsqueeze(0).expand(words_u64.shape[0], -1)
            spill_mask_2d = spill_mask.unsqueeze(0).expand(words_u64.shape[0], -1)
            shift_back_2d = shift_back.unsqueeze(0).expand(words_u64.shape[0], -1)
            spilled = torch.bitwise_left_shift(
                torch.bitwise_and(torch.gather(words_u64, 1, next_gather_index), spill_mask_2d),
                shift_back_2d,
            )
            values = torch.where(spill_flags.unsqueeze(0), torch.bitwise_or(values, spilled), values)
        return torch.bitwise_and(values, mask_i64).to(torch.float32)

    return _trace_timed_call(trace, "unpack", device_type=device_type, fn=_impl, synchronize=False)


def _score_m0_logits_flat_torch(codes, queries, scales, bias, query_group_sums):
    torch = _load_torch()
    matmul_dtype = codes.dtype if torch.is_floating_point(codes) else torch.float32
    codes_mm = codes.to(dtype=matmul_dtype)
    queries_mm = queries.to(dtype=matmul_dtype)
    if codes.ndim == 3:
        code_dim = int(codes.shape[-1])
        codes_flat = codes_mm.reshape(-1, code_dim)
        logits = torch.matmul(codes_flat, queries_mm.transpose(0, 1)).transpose(0, 1).to(torch.float32)
        return (
            logits * scales.reshape(1, -1)
            + query_group_sums.reshape(-1, 1) * bias.reshape(1, -1)
        )
    if codes.ndim == 4:
        batch_size = int(codes.shape[0])
        code_dim = int(codes.shape[-1])
        codes_flat = codes_mm.reshape(batch_size, -1, code_dim)
        logits = torch.bmm(codes_flat, queries_mm.transpose(1, 2)).transpose(1, 2).to(torch.float32)
        return (
            logits * scales.reshape(batch_size, 1, -1)
            + query_group_sums.reshape(batch_size, -1, 1) * bias.reshape(batch_size, 1, -1)
        )
    raise ValueError("codes must have shape [page_count, token_count, group_size] or [batch_size, page_count, token_count, group_size]")


def _unpack_packed_word_slice_torch(words, shifts, mask):
    torch = _load_torch()
    words_u64 = torch.bitwise_and(words.to(dtype=torch.int64), 0xFFFFFFFF)
    shifts_i64 = shifts.to(dtype=torch.int64)
    mask_i64 = torch.as_tensor(mask, dtype=torch.int64, device=words.device)
    return torch.bitwise_and(torch.bitwise_right_shift(words_u64[..., None], shifts_i64), mask_i64).to(torch.float32)


def _unpack_packed_group32_torch(payload_words, *, unpack_shifts, unpack_mask, trace=None):
    if payload_words.ndim != 4:
        raise ValueError("payload_words must have shape [batch_size, page_count, token_count, words_per_group]")
    unpacked = _trace_timed_call(
        trace,
        "unpack",
        device_type=str(payload_words.device.type),
        fn=lambda: _unpack_packed_word_slice_torch(payload_words, unpack_shifts, unpack_mask),
        synchronize=False,
    )
    if trace is not None:
        trace.record_temporary(int(unpacked.numel() * unpacked.element_size()))
    return unpacked.reshape(int(payload_words.shape[0]), int(payload_words.shape[1]), int(payload_words.shape[2]), -1)


def _score_m0_logits_packed32_torch(payload_words, queries, scales, bias, query_group_sums, *, unpack_shifts, unpack_mask, trace=None):
    if int(queries.shape[-1]) != int(payload_words.shape[-1]) * int(unpack_shifts.numel()):
        raise ValueError("queries must align with the packed payload layout")
    codes = _unpack_packed_group32_torch(
        payload_words,
        unpack_shifts=unpack_shifts,
        unpack_mask=unpack_mask,
        trace=trace,
    )
    return _score_m0_logits_flat_torch(codes, queries, scales, bias, query_group_sums)


def _score_m0_logits_packed32_grouped_torch(
    payload_words,
    queries,
    scales,
    bias,
    query_group_sums,
    *,
    unpack_shifts,
    unpack_mask,
    trace=None,
):
    torch = _load_torch()
    if payload_words.ndim != 5:
        raise ValueError("payload_words must have shape [batch_size, num_groups, page_count, token_count, words_per_group]")
    if queries.ndim != 4:
        raise ValueError("queries must have shape [batch_size, query_count, num_groups, group_size]")
    batch_size, num_groups, page_count, token_count, words_per_group = map(int, payload_words.shape)
    query_count = int(queries.shape[1])
    group_size = int(queries.shape[-1])
    if int(queries.shape[2]) != num_groups:
        raise ValueError("queries must align with payload_words group count")
    if tuple(scales.shape) != (batch_size, num_groups, page_count, token_count):
        raise ValueError("scales must align with payload_words shape")
    if tuple(bias.shape) != (batch_size, num_groups, page_count, token_count):
        raise ValueError("bias must align with payload_words shape")
    if tuple(query_group_sums.shape) != (batch_size, query_count, num_groups):
        raise ValueError("query_group_sums must align with queries shape")
    if group_size != words_per_group * int(unpack_shifts.numel()):
        raise ValueError("queries must align with the packed payload layout")
    payload_words_flat = payload_words.reshape(batch_size * num_groups, page_count, token_count, words_per_group)
    codes = _unpack_packed_group32_torch(
        payload_words_flat,
        unpack_shifts=unpack_shifts,
        unpack_mask=unpack_mask,
        trace=trace,
    )
    matmul_dtype = codes.dtype if torch.is_floating_point(codes) else torch.float32
    codes_flat = codes.reshape(batch_size * num_groups, -1, group_size).to(dtype=matmul_dtype)
    queries_flat = queries.transpose(1, 2).contiguous().reshape(batch_size * num_groups, query_count, group_size).to(dtype=matmul_dtype)
    logits = torch.bmm(codes_flat, queries_flat.transpose(1, 2)).transpose(1, 2).to(torch.float32)
    scales_flat = scales.reshape(batch_size * num_groups, 1, -1)
    bias_flat = bias.reshape(batch_size * num_groups, 1, -1)
    query_sums_flat = query_group_sums.transpose(1, 2).contiguous().reshape(batch_size * num_groups, query_count, 1)
    logits = logits * scales_flat + query_sums_flat * bias_flat
    return logits.reshape(batch_size, num_groups, query_count, page_count * token_count).sum(dim=1)


def _mix_m0_contribution_torch(weights, codes, scales, bias):
    torch = _load_torch()
    matmul_dtype = codes.dtype if torch.is_floating_point(codes) else torch.float32
    codes_mm = codes.to(dtype=matmul_dtype)
    if codes.ndim == 3:
        code_dim = int(codes.shape[-1])
        codes_flat = codes_mm.reshape(-1, code_dim)
        weights_flat = weights.reshape(weights.shape[0], -1)
        weighted_scales_flat = (weights_flat * scales.reshape(1, -1)).to(dtype=matmul_dtype)
        contribution = torch.matmul(weighted_scales_flat, codes_flat).to(torch.float32)
        bias_term = (weights_flat * bias.reshape(1, -1)).sum(dim=-1, keepdim=True)
        return contribution + bias_term
    if codes.ndim == 4:
        batch_size = int(codes.shape[0])
        code_dim = int(codes.shape[-1])
        codes_flat = codes_mm.reshape(batch_size, -1, code_dim)
        weights_flat = weights.reshape(batch_size, weights.shape[1], -1)
        weighted_scales_flat = (weights_flat * scales.reshape(batch_size, 1, -1)).to(dtype=matmul_dtype)
        contribution = torch.bmm(weighted_scales_flat, codes_flat).to(torch.float32)
        bias_term = (weights_flat * bias.reshape(batch_size, 1, -1)).sum(dim=-1, keepdim=True)
        return contribution + bias_term
    raise ValueError("codes must have shape [page_count, token_count, group_size] or [batch_size, page_count, token_count, group_size]")


def _mix_m0_contribution_packed32_torch(weights, payload_words, scales, bias, *, unpack_shifts, unpack_mask, trace=None):
    codes = _unpack_packed_group32_torch(
        payload_words,
        unpack_shifts=unpack_shifts,
        unpack_mask=unpack_mask,
        trace=trace,
    )
    return _mix_m0_contribution_torch(weights, codes, scales, bias)


def _mix_m0_contribution_packed32_grouped_torch(
    weights,
    payload_words,
    scales,
    bias,
    *,
    unpack_shifts,
    unpack_mask,
    trace=None,
):
    torch = _load_torch()
    if payload_words.ndim != 5:
        raise ValueError("payload_words must have shape [batch_size, num_groups, page_count, token_count, words_per_group]")
    if weights.ndim != 4:
        raise ValueError("weights must have shape [batch_size, query_count, page_count, token_count]")
    batch_size, num_groups, page_count, token_count, words_per_group = map(int, payload_words.shape)
    query_count = int(weights.shape[1])
    if tuple(weights.shape[2:]) != (page_count, token_count):
        raise ValueError("weights must align with payload_words shape")
    if tuple(scales.shape) != (batch_size, num_groups, page_count, token_count):
        raise ValueError("scales must align with payload_words shape")
    if tuple(bias.shape) != (batch_size, num_groups, page_count, token_count):
        raise ValueError("bias must align with payload_words shape")
    payload_words_flat = payload_words.reshape(batch_size * num_groups, page_count, token_count, words_per_group)
    codes = _unpack_packed_group32_torch(
        payload_words_flat,
        unpack_shifts=unpack_shifts,
        unpack_mask=unpack_mask,
        trace=trace,
    )
    group_size = int(codes.shape[-1])
    matmul_dtype = codes.dtype if torch.is_floating_point(codes) else torch.float32
    codes_flat = codes.reshape(batch_size * num_groups, -1, group_size).to(dtype=matmul_dtype)
    weights_flat = weights.reshape(batch_size, query_count, -1)
    weighted_scales = (
        weights_flat[:, :, None, :] * scales.reshape(batch_size, 1, num_groups, -1)
    ).permute(0, 2, 1, 3).contiguous().reshape(batch_size * num_groups, query_count, -1).to(dtype=matmul_dtype)
    contribution = torch.bmm(weighted_scales, codes_flat).to(torch.float32)
    contribution = contribution.reshape(batch_size, num_groups, query_count, group_size).permute(0, 2, 1, 3)
    bias_term = (
        weights_flat[:, :, None, :] * bias.reshape(batch_size, 1, num_groups, -1)
    ).permute(0, 2, 1, 3).sum(dim=-1).permute(0, 2, 1)
    return contribution + bias_term[..., None]


def _score_m0_logits_two_group64_torch(fused_scaled_codes, fused_queries, bias_groups, query_group_sums):
    torch = _load_torch()
    if isinstance(bias_groups, torch.Tensor):
        if bias_groups.ndim == 2:
            if int(bias_groups.shape[0]) == 2:
                bias0 = bias_groups[0]
                bias1 = bias_groups[1]
            elif int(bias_groups.shape[1]) == 2:
                bias0 = bias_groups[:, 0]
                bias1 = bias_groups[:, 1]
            else:
                raise ValueError("tensor bias_groups must have a two-group dimension")
        elif bias_groups.ndim == 3:
            if int(bias_groups.shape[1]) != 2:
                raise ValueError("batched tensor bias_groups must have shape [batch, 2, token_count]")
            bias0 = bias_groups[:, 0, :]
            bias1 = bias_groups[:, 1, :]
        else:
            raise ValueError("tensor bias_groups must have shape [2, token_count], [token_count, 2], or [batch, 2, token_count]")
    else:
        bias0, bias1 = bias_groups
    matmul_dtype = fused_scaled_codes.dtype if torch.is_floating_point(fused_scaled_codes) else torch.float32
    if fused_scaled_codes.ndim == 3:
        fused_queries_mm = fused_queries.to(dtype=matmul_dtype)
        logits = torch.matmul(
            fused_scaled_codes.reshape(-1, int(fused_scaled_codes.shape[-1])),
            fused_queries_mm.transpose(0, 1),
        ).transpose(0, 1).to(torch.float32)
        bias_term = (
            query_group_sums[:, 0:1] * bias0.reshape(1, -1)
            + query_group_sums[:, 1:2] * bias1.reshape(1, -1)
        )
        return logits + bias_term
    if fused_scaled_codes.ndim == 4:
        batch_size = int(fused_scaled_codes.shape[0])
        squeeze_batch = False
        fused_queries_mm = fused_queries.to(dtype=matmul_dtype)
        query_group_sums_mm = query_group_sums
        if fused_queries_mm.ndim == 2:
            fused_queries_mm = fused_queries_mm.unsqueeze(0)
            query_group_sums_mm = query_group_sums_mm.unsqueeze(0)
            squeeze_batch = True
        logits = torch.bmm(
            fused_scaled_codes.reshape(batch_size, -1, int(fused_scaled_codes.shape[-1])),
            fused_queries_mm.transpose(1, 2),
        ).transpose(1, 2).to(torch.float32)
        bias_term = (
            query_group_sums_mm[:, :, 0:1] * bias0.reshape(batch_size, 1, -1)
            + query_group_sums_mm[:, :, 1:2] * bias1.reshape(batch_size, 1, -1)
        )
        output = logits + bias_term
        return output.squeeze(0) if squeeze_batch else output
    raise ValueError("fused_scaled_codes must have shape [page_count, token_count, 64] or [batch_size, page_count, token_count, 64]")


def _score_m0_logits_fused_torch(fused_scaled_codes, fused_queries, bias_groups, query_group_sums):
    torch = _load_torch()
    num_groups = int(query_group_sums.shape[-1])
    bias_tensor = None
    if isinstance(bias_groups, torch.Tensor):
        if bias_groups.ndim == 2:
            if int(bias_groups.shape[0]) == num_groups:
                bias_tensor = bias_groups
            elif int(bias_groups.shape[1]) == num_groups:
                bias_tensor = bias_groups.transpose(0, 1)
            else:
                raise ValueError("tensor bias_groups must align with the query group count")
        elif bias_groups.ndim == 3:
            if int(bias_groups.shape[1]) == num_groups:
                bias_tensor = bias_groups
            elif int(bias_groups.shape[2]) == num_groups:
                bias_tensor = bias_groups.transpose(1, 2)
            else:
                raise ValueError("batched tensor bias_groups must align with the query group count")
        elif bias_groups.ndim == 4:
            if int(bias_groups.shape[1]) == num_groups:
                bias_tensor = bias_groups.reshape(int(bias_groups.shape[0]), num_groups, -1)
            elif int(bias_groups.shape[-1]) == num_groups:
                bias_tensor = bias_groups.permute(0, 3, 1, 2).reshape(int(bias_groups.shape[0]), num_groups, -1)
            else:
                raise ValueError("4D tensor bias_groups must align with the query group count")
        else:
            raise ValueError(
                "tensor bias_groups must have shape [num_groups, token_count], [token_count, num_groups], "
                "[batch, num_groups, token_count], or [batch, num_groups, page_count, token_count]"
            )
    elif fused_scaled_codes.ndim == 3:
        bias_tensor = torch.stack(tuple(bias_group.reshape(-1) for bias_group in bias_groups), dim=0)
    elif fused_scaled_codes.ndim == 4:
        bias_tensor = torch.stack(tuple(bias_group.reshape(int(bias_group.shape[0]), -1) for bias_group in bias_groups), dim=1)
    else:
        raise ValueError(
            "fused_scaled_codes must have shape [page_count, token_count, padded_head_dim] "
            "or [batch_size, page_count, token_count, padded_head_dim]"
        )
    matmul_dtype = fused_scaled_codes.dtype if torch.is_floating_point(fused_scaled_codes) else torch.float32
    if fused_scaled_codes.ndim == 3:
        if bias_tensor.ndim == 3:
            if int(bias_tensor.shape[0]) != 1:
                raise ValueError("single-batch fused M0 scoring expects bias tensor batch dimension of 1")
            bias_tensor = bias_tensor.squeeze(0)
        logits = torch.matmul(
            fused_scaled_codes.reshape(-1, int(fused_scaled_codes.shape[-1])),
            fused_queries.to(dtype=matmul_dtype).transpose(0, 1),
        ).transpose(0, 1).to(torch.float32)
        bias_term = torch.matmul(
            query_group_sums.to(dtype=torch.float32),
            bias_tensor.to(dtype=torch.float32),
        )
        return logits + bias_term
    if fused_scaled_codes.ndim == 4:
        batch_size = int(fused_scaled_codes.shape[0])
        squeeze_batch = False
        fused_queries_mm = fused_queries
        query_group_sums_mm = query_group_sums
        if fused_queries_mm.ndim == 2:
            fused_queries_mm = fused_queries_mm.unsqueeze(0)
            query_group_sums_mm = query_group_sums_mm.unsqueeze(0)
            squeeze_batch = True
        if bias_tensor.ndim == 2:
            bias_tensor = bias_tensor.unsqueeze(0)
        logits = torch.bmm(
            fused_scaled_codes.reshape(batch_size, -1, int(fused_scaled_codes.shape[-1])),
            fused_queries_mm.to(dtype=matmul_dtype).transpose(1, 2),
        ).transpose(1, 2).to(torch.float32)
        bias_term = torch.bmm(
            query_group_sums_mm.to(dtype=torch.float32),
            bias_tensor.to(dtype=torch.float32),
        )
        output = logits + bias_term
        return output.squeeze(0) if squeeze_batch else output


def _score_m0_logits_fused_with_bias_torch(fused_with_bias_codes, fused_queries, query_group_sums):
    torch = _load_torch()
    matmul_dtype = fused_with_bias_codes.dtype if torch.is_floating_point(fused_with_bias_codes) else torch.float32
    if fused_with_bias_codes.ndim == 3:
        if fused_queries.ndim != 2:
            raise ValueError("single-batch fused-with-bias scoring expects fused_queries with shape [query_count, padded_head_dim]")
        combined_queries = torch.cat(
            [
                fused_queries.to(dtype=matmul_dtype),
                query_group_sums.to(dtype=matmul_dtype),
            ],
            dim=-1,
        )
        codes_flat = fused_with_bias_codes.reshape(-1, int(fused_with_bias_codes.shape[-1])).to(dtype=matmul_dtype)
        return torch.matmul(
            combined_queries,
            codes_flat.transpose(0, 1),
        ).to(torch.float32)
    if fused_with_bias_codes.ndim == 4:
        batch_size = int(fused_with_bias_codes.shape[0])
        squeeze_batch = False
        fused_queries_mm = fused_queries
        query_group_sums_mm = query_group_sums
        if fused_queries_mm.ndim == 2:
            fused_queries_mm = fused_queries_mm.unsqueeze(0)
            query_group_sums_mm = query_group_sums_mm.unsqueeze(0)
            squeeze_batch = True
        combined_queries = torch.cat(
            [
                fused_queries_mm.to(dtype=matmul_dtype),
                query_group_sums_mm.to(dtype=matmul_dtype),
            ],
            dim=-1,
        )
        codes_flat = fused_with_bias_codes.reshape(batch_size, -1, int(fused_with_bias_codes.shape[-1])).to(dtype=matmul_dtype)
        output = torch.bmm(
            combined_queries,
            codes_flat.transpose(1, 2),
        ).to(torch.float32)
        return output.squeeze(0) if squeeze_batch else output
    raise ValueError(
        "fused_with_bias_codes must have shape [page_count, token_count, padded_head_dim + num_groups] "
        "or [batch_size, page_count, token_count, padded_head_dim + num_groups]"
    )


def _score_m0_logits_fused_transposed_torch(fused_scaled_codes_transposed, fused_queries, bias_groups, query_group_sums):
    torch = _load_torch()
    num_groups = int(query_group_sums.shape[-1])
    bias_tensor = None
    if isinstance(bias_groups, torch.Tensor):
        if bias_groups.ndim == 2:
            if int(bias_groups.shape[0]) == num_groups:
                bias_tensor = bias_groups
            elif int(bias_groups.shape[1]) == num_groups:
                bias_tensor = bias_groups.transpose(0, 1)
            else:
                raise ValueError("tensor bias_groups must align with the query group count")
        elif bias_groups.ndim == 3:
            if int(bias_groups.shape[1]) == num_groups:
                bias_tensor = bias_groups
            elif int(bias_groups.shape[2]) == num_groups:
                bias_tensor = bias_groups.transpose(1, 2)
            else:
                raise ValueError("batched tensor bias_groups must align with the query group count")
        elif bias_groups.ndim == 4:
            if int(bias_groups.shape[1]) == num_groups:
                bias_tensor = bias_groups.reshape(int(bias_groups.shape[0]), num_groups, -1)
            elif int(bias_groups.shape[-1]) == num_groups:
                bias_tensor = bias_groups.permute(0, 3, 1, 2).reshape(int(bias_groups.shape[0]), num_groups, -1)
            else:
                raise ValueError("4D tensor bias_groups must align with the query group count")
        else:
            raise ValueError(
                "tensor bias_groups must have shape [num_groups, token_count], [token_count, num_groups], "
                "[batch, num_groups, token_count], or [batch, num_groups, page_count, token_count]"
            )
    elif fused_scaled_codes_transposed.ndim == 2:
        bias_tensor = torch.stack(tuple(bias_group.reshape(-1) for bias_group in bias_groups), dim=0)
    elif fused_scaled_codes_transposed.ndim == 3:
        bias_tensor = torch.stack(tuple(bias_group.reshape(int(bias_group.shape[0]), -1) for bias_group in bias_groups), dim=1)
    else:
        raise ValueError(
            "fused_scaled_codes_transposed must have shape [padded_head_dim, token_count] "
            "or [batch_size, padded_head_dim, token_count]"
        )
    matmul_dtype = fused_scaled_codes_transposed.dtype if torch.is_floating_point(fused_scaled_codes_transposed) else torch.float32
    if fused_scaled_codes_transposed.ndim == 2:
        logits = torch.matmul(
            fused_queries.to(dtype=matmul_dtype),
            fused_scaled_codes_transposed.to(dtype=matmul_dtype),
        ).to(torch.float32)
        bias_term = torch.matmul(
            query_group_sums.to(dtype=torch.float32),
            bias_tensor.to(dtype=torch.float32),
        )
        return logits + bias_term
    if fused_scaled_codes_transposed.ndim == 3:
        batch_size = int(fused_scaled_codes_transposed.shape[0])
        squeeze_batch = False
        fused_queries_mm = fused_queries
        query_group_sums_mm = query_group_sums
        if fused_queries_mm.ndim == 2:
            fused_queries_mm = fused_queries_mm.unsqueeze(0)
            query_group_sums_mm = query_group_sums_mm.unsqueeze(0)
            squeeze_batch = True
        if bias_tensor.ndim == 2:
            bias_tensor = bias_tensor.unsqueeze(0)
        logits = torch.bmm(
            fused_queries_mm.to(dtype=matmul_dtype),
            fused_scaled_codes_transposed.to(dtype=matmul_dtype),
        ).to(torch.float32)
        bias_term = torch.bmm(
            query_group_sums_mm.to(dtype=torch.float32),
            bias_tensor.to(dtype=torch.float32),
        )
        output = logits + bias_term
        return output.squeeze(0) if squeeze_batch else output
    raise ValueError(
        "fused_scaled_codes_transposed must have shape [padded_head_dim, token_count] "
        "or [batch_size, padded_head_dim, token_count]"
    )


def _score_exact_logits_paged_torch(keys, queries):
    torch = _load_torch()
    squeeze_batch = False
    if keys.ndim == 3:
        keys = keys.unsqueeze(0)
        squeeze_batch = True
    if queries.ndim == 2:
        queries = queries.unsqueeze(0)
        squeeze_batch = True
    if keys.ndim != 4:
        raise ValueError("keys must have shape [batch_size, page_count, token_count, head_dim]")
    if queries.ndim != 3:
        raise ValueError("queries must have shape [batch_size, query_count, head_dim]")
    batch_size, _page_count, _token_count, head_dim = map(int, keys.shape)
    if int(queries.shape[0]) != batch_size or int(queries.shape[-1]) != head_dim:
        raise ValueError("queries batch/head_dim must align with keys")
    key_flat = keys.reshape(batch_size, -1, head_dim)
    output = torch.bmm(key_flat, queries.transpose(1, 2)).transpose(1, 2).to(torch.float32)
    return output.squeeze(0) if squeeze_batch else output


def _score_exact_logits_flat_torch(keys_flat, queries):
    torch = _load_torch()
    squeeze_batch = False
    if keys_flat.ndim == 2:
        keys_flat = keys_flat.unsqueeze(0)
        squeeze_batch = True
    if queries.ndim == 2:
        queries = queries.unsqueeze(0)
        squeeze_batch = True
    if keys_flat.ndim != 3:
        raise ValueError("keys_flat must have shape [batch_size, token_count, head_dim]")
    if queries.ndim != 3:
        raise ValueError("queries must have shape [batch_size, query_count, head_dim]")
    batch_size, _token_count, head_dim = map(int, keys_flat.shape)
    if int(queries.shape[0]) != batch_size or int(queries.shape[-1]) != head_dim:
        raise ValueError("queries batch/head_dim must align with keys_flat")
    output = torch.bmm(keys_flat, queries.transpose(1, 2)).transpose(1, 2).to(torch.float32)
    return output.squeeze(0) if squeeze_batch else output


def _score_exact_logits_transposed_torch(keys_transposed, queries):
    torch = _load_torch()
    squeeze_batch = False
    if keys_transposed.ndim == 2:
        keys_transposed = keys_transposed.unsqueeze(0)
        squeeze_batch = True
    if queries.ndim == 2:
        queries = queries.unsqueeze(0)
        squeeze_batch = True
    if keys_transposed.ndim != 3:
        raise ValueError("keys_transposed must have shape [batch_size, head_dim, token_count]")
    if queries.ndim != 3:
        raise ValueError("queries must have shape [batch_size, query_count, head_dim]")
    batch_size, head_dim, _token_count = map(int, keys_transposed.shape)
    if int(queries.shape[0]) != batch_size or int(queries.shape[-1]) != head_dim:
        raise ValueError("queries batch/head_dim must align with keys_transposed")
    output = torch.bmm(queries, keys_transposed).to(torch.float32)
    return output.squeeze(0) if squeeze_batch else output


def _mix_m0_contribution_two_group64_torch(weights, fused_scaled_codes, bias_groups):
    torch = _load_torch()
    bias0, bias1 = bias_groups
    matmul_dtype = fused_scaled_codes.dtype if torch.is_floating_point(fused_scaled_codes) else torch.float32
    if fused_scaled_codes.ndim == 3:
        weights_flat = weights.reshape(weights.shape[0], -1)
        weights_mm = weights_flat.to(dtype=matmul_dtype)
        contribution = torch.matmul(
            weights_mm,
            fused_scaled_codes.reshape(-1, int(fused_scaled_codes.shape[-1])),
        ).to(torch.float32)
        bias0_term = (weights_flat * bias0.reshape(1, -1)).sum(dim=-1, keepdim=True)
        bias1_term = (weights_flat * bias1.reshape(1, -1)).sum(dim=-1, keepdim=True)
        return torch.cat([contribution[:, :32] + bias0_term, contribution[:, 32:] + bias1_term], dim=-1)
    if fused_scaled_codes.ndim == 4:
        batch_size = int(fused_scaled_codes.shape[0])
        weights_flat = weights.reshape(batch_size, weights.shape[1], -1)
        weights_mm = weights_flat.to(dtype=matmul_dtype)
        contribution = torch.bmm(
            weights_mm,
            fused_scaled_codes.reshape(batch_size, -1, int(fused_scaled_codes.shape[-1])),
        ).to(torch.float32)
        bias0_term = (weights_flat * bias0.reshape(batch_size, 1, -1)).sum(dim=-1, keepdim=True)
        bias1_term = (weights_flat * bias1.reshape(batch_size, 1, -1)).sum(dim=-1, keepdim=True)
        return torch.cat([contribution[:, :, :32] + bias0_term, contribution[:, :, 32:] + bias1_term], dim=-1)
    raise ValueError("fused_scaled_codes must have shape [page_count, token_count, 64] or [batch_size, page_count, token_count, 64]")


def _mix_m0_contribution_fused_torch(weights, fused_scaled_codes, bias_groups, *, group_size: int):
    torch = _load_torch()
    if fused_scaled_codes.ndim == 3:
        weights_flat = weights.reshape(weights.shape[0], -1)
        output = torch.matmul(
            weights_flat,
            fused_scaled_codes.reshape(-1, int(fused_scaled_codes.shape[-1])),
        ).to(torch.float32)
        for group_index, bias_group in enumerate(bias_groups):
            bias_term = (weights_flat * bias_group.reshape(1, -1)).sum(dim=-1, keepdim=True)
            start = group_index * group_size
            end = start + group_size
            output[:, start:end] += bias_term
        return output
    if fused_scaled_codes.ndim == 4:
        batch_size = int(fused_scaled_codes.shape[0])
        weights_flat = weights.reshape(batch_size, weights.shape[1], -1)
        output = torch.bmm(
            weights_flat,
            fused_scaled_codes.reshape(batch_size, -1, int(fused_scaled_codes.shape[-1])),
        ).to(torch.float32)
        for group_index, bias_group in enumerate(bias_groups):
            bias_term = (weights_flat * bias_group.reshape(batch_size, 1, -1)).sum(dim=-1, keepdim=True)
            start = group_index * group_size
            end = start + group_size
            output[:, :, start:end] += bias_term
        return output
    raise ValueError(
        "fused_scaled_codes must have shape [page_count, token_count, padded_head_dim] "
        "or [batch_size, page_count, token_count, padded_head_dim]"
    )


def _lookup_lut_group_torch(codebooks, codes):
    torch = _load_torch()
    lut = codebooks.to(dtype=torch.float32)
    code_indices = codes.to(dtype=torch.int64)
    if lut.ndim == 1 and code_indices.ndim == 1:
        return lut[code_indices]
    if lut.ndim == 2 and code_indices.ndim == 1 and lut.shape[0] == code_indices.shape[0]:
        page_index = torch.arange(lut.shape[0], device=lut.device)
        return lut[page_index, code_indices]
    if lut.ndim == 2 and code_indices.ndim == 2:
        token_count = int(code_indices.shape[0])
        segment_count = int(lut.shape[0])
        if segment_count == 1:
            return lut[0][code_indices]
        segment_ids = (torch.arange(token_count, device=lut.device, dtype=torch.int64) * segment_count) // max(token_count, 1)
        return lut[segment_ids[:, None], code_indices]
    if lut.ndim == 3 and code_indices.ndim == 3:
        page_count = int(code_indices.shape[0])
        token_count = int(code_indices.shape[1])
        segment_count = int(lut.shape[1])
        if segment_count == 1:
            page_index = torch.arange(page_count, device=lut.device)[:, None, None]
            return lut[page_index, torch.zeros(1, device=lut.device, dtype=torch.int64), code_indices]
        page_index = torch.arange(page_count, device=lut.device)[:, None, None]
        segment_ids = (torch.arange(token_count, device=lut.device, dtype=torch.int64) * segment_count) // max(token_count, 1)
        return lut[page_index, segment_ids[None, :, None], code_indices]
    if lut.ndim == 4 and code_indices.ndim == 4:
        batch_size = int(code_indices.shape[0])
        page_count = int(code_indices.shape[1])
        token_count = int(code_indices.shape[2])
        segment_count = int(lut.shape[2])
        batch_index = torch.arange(batch_size, device=lut.device)[:, None, None, None]
        page_index = torch.arange(page_count, device=lut.device)[None, :, None, None]
        if segment_count == 1:
            return lut[batch_index, page_index, torch.zeros(1, device=lut.device, dtype=torch.int64), code_indices]
        segment_ids = (torch.arange(token_count, device=lut.device, dtype=torch.int64) * segment_count) // max(token_count, 1)
        return lut[batch_index, page_index, segment_ids[None, None, :, None], code_indices]
    raise ValueError("unsupported LUT rank")


def _lookup_turbo_group_torch(codebooks, codes):
    torch = _load_torch()
    lut = codebooks.to(dtype=torch.float32)
    code_indices = codes.to(dtype=torch.int64)
    if lut.ndim == 1 and code_indices.ndim == 2:
        return lut[code_indices]
    if lut.ndim == 1 and code_indices.ndim == 3:
        return lut[code_indices]
    if lut.ndim == 1 and code_indices.ndim == 4:
        return lut[code_indices]
    if lut.ndim == 2 and code_indices.ndim == 3 and lut.shape[0] == code_indices.shape[0]:
        page_index = torch.arange(lut.shape[0], device=lut.device)[:, None, None]
        return lut[page_index, code_indices]
    if lut.ndim == 3 and code_indices.ndim == 4:
        batch_index = torch.arange(lut.shape[0], device=lut.device)[:, None, None, None]
        page_index = torch.arange(lut.shape[1], device=lut.device)[None, :, None, None]
        return lut[batch_index, page_index, code_indices]
    raise ValueError("unsupported turbo3 LUT rank")


def _fwht_last_dim_torch(values, *, trace: ExecutionTrace | None = None):
    torch = _load_torch()
    width = int(values.shape[-1])
    if width <= 0:
        return values
    if width & (width - 1):
        raise ValueError("FWHT requires the last dimension to be a power of two")
    device_type = str(values.device.type)

    def _impl():
        original_shape = tuple(values.shape)
        matrix = _fwht_matrix_torch(width, device_type=device_type)
        transformed = values.to(dtype=torch.float32).reshape(-1, width)
        return torch.matmul(transformed, matrix.T).reshape(original_shape)

    return _trace_timed_call(trace, "fwht", device_type=device_type, fn=_impl, synchronize=False)


def _score_page_chunk_torch(query_slice: np.ndarray | Any, pages: Sequence[PreparedPageTorch], *, trace: ExecutionTrace | None = None):
    torch = _load_torch()
    if not pages:
        raise ValueError("pages must be non-empty")
    header = pages[0].header
    device_type = pages[0].device_type
    if trace is not None:
        trace.record_page_read(
            sum(page.payload_nbytes for page in pages),
            sum(page.metadata_nbytes for page in pages),
        )

    if header.mode_default == "M3":
        use_native_dtype = _m3_native_compute_enabled(pages)
        dense = _decode_escape_batch_torch(
            pages,
            token_count=header.token_count,
            head_dim=header.head_dim,
            promote_float32=not use_native_dtype,
        )
        query = _pad_query(query_slice, header.head_dim, device_type=device_type)
        if use_native_dtype:
            return torch.matmul(dense, query.to(dtype=dense.dtype)).reshape(-1).to(dtype=torch.float32)
        return torch.matmul(dense, query).reshape(-1)

    if header.mode_default == "M2":
        query = _pad_query(query_slice, header.padded_head_dim, device_type=device_type)
        query_groups = query.reshape(header.num_groups, header.group_size)
        page_count = len(pages)
        logits = torch.zeros((page_count, header.token_count), dtype=torch.float32, device=device_type)
        prepared_chunk = _get_prepared_chunk_mps(pages)
        for group_index in range(header.num_groups):
            group_sketch = (
                prepared_chunk.m2_sketch_groups[group_index]
                if prepared_chunk is not None and prepared_chunk.m2_sketch_groups is not None
                else torch.stack([page.m2_sketch[:, group_index, :] for page in pages], dim=0)
            )
            group_basis = (
                prepared_chunk.m2_basis_groups[group_index]
                if prepared_chunk is not None and prepared_chunk.m2_basis_groups is not None
                else torch.stack([page.m2_basis[group_index] for page in pages], dim=0)
            )
            group_mean = (
                prepared_chunk.m2_mean_groups[group_index]
                if prepared_chunk is not None and prepared_chunk.m2_mean_groups is not None
                else torch.stack([page.m2_mean[group_index] for page in pages], dim=0)
            )
            qg, group_sketch, group_basis, group_mean = _coerce_m2_operands(
                query_groups[group_index],
                group_sketch,
                group_basis,
                group_mean,
            )
            if group_basis.dim() == 3:
                q_proj = torch.einsum("prg,g->pr", group_basis, qg)
                logits += torch.einsum("ptd,pd->pt", group_sketch, q_proj)
                logits += torch.einsum("pg,g->p", group_mean, qg)[:, None]
                continue
            segment_ids = (
                prepared_chunk.m2_segment_ids
                if prepared_chunk is not None and prepared_chunk.m2_segment_ids is not None
                else _segment_ids_tensor(header.token_count, int(group_basis.shape[1]), device_type=device_type)
            )
            q_proj = torch.einsum("psrg,g->psr", group_basis, qg)
            logits += torch.einsum("ptr,ptr->pt", group_sketch, q_proj[:, segment_ids, :])
            logits += torch.einsum("ptg,g->pt", group_mean[:, segment_ids, :], qg)
        return logits.reshape(-1)

    if header.mode_default == "M4":
        query = _pad_query(query_slice, header.padded_head_dim, device_type=device_type)
        query_groups = query.reshape(header.num_groups, header.group_size)
        page_count = len(pages)
        logits = torch.zeros((page_count, header.token_count), dtype=torch.float32, device=device_type)
        prepared_chunk = _get_prepared_chunk_mps(pages)
        for group_index in range(header.num_groups):
            group_sketch = (
                prepared_chunk.m2_sketch_groups[group_index]
                if prepared_chunk is not None and prepared_chunk.m2_sketch_groups is not None
                else torch.stack([page.m2_sketch[:, group_index, :] for page in pages], dim=0)
            )
            group_basis = (
                prepared_chunk.m2_basis_groups[group_index]
                if prepared_chunk is not None and prepared_chunk.m2_basis_groups is not None
                else (
                    torch.stack([page.m2_basis[group_index] for page in pages], dim=0)
                    if pages[0].m2_basis is not None
                    else (
                        _device_tensor(np.asarray(pages[0].source_page.m2_basis[group_index]), device=device_type).contiguous()
                        if (
                            pages[0].header.project_basis == "svd_shared"
                            and pages[0].source_page.m2_basis is not None
                            and all(page.source_page.m2_basis is pages[0].source_page.m2_basis for page in pages)
                        )
                        else None
                    )
                )
            )
            group_mean = (
                prepared_chunk.m2_mean_groups[group_index]
                if prepared_chunk is not None and prepared_chunk.m2_mean_groups is not None
                else torch.stack([page.m2_mean[group_index] for page in pages], dim=0)
            )
            qg = query_groups[group_index]
            if group_basis is not None:
                qg, group_sketch, group_basis, group_mean = _coerce_m2_operands(qg, group_sketch, group_basis, group_mean)
                if int(group_basis.dim()) == 2:
                    q_proj = torch.einsum("rg,g->r", group_basis, qg)
                    logits += torch.einsum("ptr,r->pt", group_sketch, q_proj)
                else:
                    q_proj = torch.einsum("prg,g->pr", group_basis, qg)
                    logits += torch.einsum("ptr,pr->pt", group_sketch, q_proj)
            else:
                basis = _m4_basis_torch(
                    header.group_size,
                    int(pages[0].m2_sketch.shape[-1]),
                    basis_family=header.project_basis,
                    device_type=device_type,
                )
                qg, group_sketch, _, group_mean = _coerce_m2_operands(qg, group_sketch, basis, group_mean)
                q_proj = torch.matmul(qg, basis.to(dtype=qg.dtype).transpose(0, 1))
                logits += torch.einsum("ptr,r->pt", group_sketch, q_proj)
            logits += torch.einsum("pg,g->p", group_mean, qg)[:, None]
        return logits.reshape(-1)

    if header.mode_default == "M1":
        query = _pad_query(query_slice, header.padded_head_dim, device_type=device_type)
        query_groups = query.reshape(header.num_groups, header.group_size)
        page_count = len(pages)
        logits = torch.zeros((page_count, header.token_count), dtype=torch.float32, device=device_type)
        for group_index in range(header.num_groups):
            group_words = torch.stack([page.payload[group_index] for page in pages], dim=0)
            codes = _unpack_bits_torch(
                group_words.reshape(-1, header.words_per_group),
                pages[0].unpack_shifts,
                pages[0].unpack_mask,
                header.group_size,
            ).reshape(page_count, header.token_count, header.group_size)
            if trace is not None:
                trace.record_temporary(int(codes.numel() * codes.element_size()))
            group = _lookup_lut_group_torch(
                torch.stack([page.codebooks[group_index] for page in pages], dim=0),
                codes,
            )
            logits += torch.matmul(group, query_groups[group_index])
        return logits.reshape(-1)

    if header.mode_default == "T3":
        query = _pad_query(query_slice, header.padded_head_dim, device_type=device_type)
        rotated_query_groups = _fwht_last_dim_torch(query.reshape(header.num_groups, header.group_size))
        page_count = len(pages)
        logits = torch.zeros((page_count, header.token_count), dtype=torch.float32, device=device_type)
        codebooks = pages[0].codebooks if pages[0].codebooks is not None else _turbo3_centroids_torch(device_type=device_type)
        prepared_chunk = _get_prepared_chunk_mps(pages)
        for group_index in range(header.num_groups):
            group_words = (
                prepared_chunk.payload_groups[group_index]
                if prepared_chunk is not None
                else torch.stack([page.payload[group_index] for page in pages], dim=0)
            )
            codes = _unpack_bits_torch(
                group_words.reshape(-1, header.words_per_group),
                pages[0].unpack_shifts,
                pages[0].unpack_mask,
                header.group_size,
            ).reshape(page_count, header.token_count, header.group_size)
            if trace is not None:
                trace.record_temporary(int(codes.numel() * codes.element_size()))
            scales = (
                prepared_chunk.scales_groups[group_index]
                if prepared_chunk is not None and prepared_chunk.scales_groups is not None
                else torch.stack([page.scales[:, group_index] for page in pages], dim=0)
            )
            corrected = _lookup_turbo_group_torch(codebooks, codes) * scales[..., None]
            logits += torch.matmul(corrected, rotated_query_groups[group_index])
        return logits.reshape(-1)

    query = _pad_query(query_slice, header.padded_head_dim, device_type=device_type)
    query_groups = query.reshape(header.num_groups, header.group_size)
    query_group_sums = query_groups.sum(dim=-1)
    page_count = len(pages)
    logits = torch.zeros((page_count, header.token_count), dtype=torch.float32, device=device_type)

    prepared_chunk = _get_prepared_chunk_mps(pages)
    if (
        prepared_chunk is not None
        and prepared_chunk.fused_scaled_codes is not None
        and prepared_chunk.bias_groups is not None
    ):
        if trace is not None:
            trace.record_per_kv_kernel_variant(section="score", variant="fused_generic")
        fused_query = query.reshape(1, header.padded_head_dim).contiguous()
        fused_query_group_sums = query_group_sums.reshape(1, header.num_groups).contiguous()
        return _score_m0_logits_fused_torch(
            prepared_chunk.fused_scaled_codes,
            fused_query,
            prepared_chunk.bias_groups,
            fused_query_group_sums,
        ).reshape(-1)
    if trace is not None:
        trace.record_per_kv_kernel_variant(section="score", variant="generic")
    for group_index in range(header.num_groups):
        cached_codes = prepared_chunk is not None and prepared_chunk.codes_groups is not None
        if cached_codes:
            codes = prepared_chunk.codes_groups[group_index]
        else:
            codes = _trace_timed_call(
                trace,
                "unpack",
                device_type=device_type,
                fn=lambda group_index=group_index: _unpack_bits_torch(
                    torch.stack([page.payload[group_index] for page in pages], dim=0).reshape(-1, header.words_per_group),
                    pages[0].unpack_shifts,
                    pages[0].unpack_mask,
                    header.group_size,
                ).reshape(page_count, header.token_count, header.group_size),
            )
        if trace is not None and not cached_codes:
            trace.record_temporary(int(codes.numel() * codes.element_size()))
        qg = query_groups[group_index]
        int_dot = torch.matmul(codes, qg)
        scales = (
            prepared_chunk.scales_groups[group_index]
            if prepared_chunk is not None and prepared_chunk.scales_groups is not None
            else torch.stack([page.scales[:, group_index].to(torch.float32) for page in pages], dim=0)
        )
        bias = (
            prepared_chunk.bias_groups[group_index]
            if prepared_chunk is not None and prepared_chunk.bias_groups is not None
            else torch.stack([page.bias[:, group_index].to(torch.float32) for page in pages], dim=0)
        )
        logits += scales * int_dot + bias * query_group_sums[group_index]

    return logits.reshape(-1)


def _mix_page_chunk_torch(
    attn_weights,
    pages: Sequence[PreparedPageTorch],
    *,
    out_acc=None,
    trace: ExecutionTrace | None = None,
):
    torch = _load_torch()
    if not pages:
        raise ValueError("pages must be non-empty")
    header = pages[0].header
    device_type = pages[0].device_type
    page_count = len(pages)
    token_count = header.token_count
    output = _prepare_output_accumulator_tensor(out_acc, header.head_dim, header.padded_head_dim, device_type=device_type)

    if trace is not None:
        trace.record_page_read(
            sum(page.payload_nbytes for page in pages),
            sum(page.metadata_nbytes for page in pages),
        )

    if not isinstance(attn_weights, torch.Tensor):
        weights = torch.as_tensor(attn_weights, dtype=torch.float32, device=device_type)
    else:
        weights = attn_weights.to(dtype=torch.float32, device=device_type)
    expected_shape = (page_count, token_count)
    if tuple(weights.shape) != expected_shape:
        raise ValueError("attn_weights chunk must have shape [page_count, token_count]")

    if header.mode_default == "M3":
        use_native_dtype = _m3_native_compute_enabled(pages)
        dense = _decode_escape_batch_torch(
            pages,
            token_count=header.token_count,
            head_dim=header.head_dim,
            promote_float32=not use_native_dtype,
        )
        if use_native_dtype:
            output[: header.head_dim] += torch.sum(weights.to(dtype=dense.dtype)[..., None] * dense, dim=(0, 1)).to(
                dtype=torch.float32
            )
        else:
            output[: header.head_dim] += torch.sum(weights[..., None] * dense, dim=(0, 1))
        return output

    if header.mode_default in {"M2", "M4"}:
        raise ValueError(f"{header.mode_default} is only supported for key scoring in this phase")

    if header.mode_default == "M1":
        for group_index in range(header.num_groups):
            group_words = torch.stack([page.payload[group_index] for page in pages], dim=0)
            codes = _unpack_bits_torch(
                group_words.reshape(-1, header.words_per_group),
                pages[0].unpack_shifts,
                pages[0].unpack_mask,
                header.group_size,
            ).reshape(page_count, token_count, header.group_size)
            if trace is not None:
                trace.record_temporary(int(codes.numel() * codes.element_size()))
            group = _lookup_lut_group_torch(
                torch.stack([page.codebooks[group_index] for page in pages], dim=0),
                codes,
            )
            start = group_index * header.group_size
            end = start + header.group_size
            output[start:end] += torch.einsum("pt,ptg->g", weights, group)
        return output

    if header.mode_default == "T3":
        codebooks = pages[0].codebooks if pages[0].codebooks is not None else _turbo3_centroids_torch(device_type=device_type)
        prepared_chunk = _get_prepared_chunk_mps(pages)
        for group_index in range(header.num_groups):
            group_words = (
                prepared_chunk.payload_groups[group_index]
                if prepared_chunk is not None
                else torch.stack([page.payload[group_index] for page in pages], dim=0)
            )
            codes = _unpack_bits_torch(
                group_words.reshape(-1, header.words_per_group),
                pages[0].unpack_shifts,
                pages[0].unpack_mask,
                header.group_size,
            ).reshape(page_count, token_count, header.group_size)
            if trace is not None:
                trace.record_temporary(int(codes.numel() * codes.element_size()))
            scales = (
                prepared_chunk.scales_groups[group_index]
                if prepared_chunk is not None and prepared_chunk.scales_groups is not None
                else torch.stack([page.scales[:, group_index] for page in pages], dim=0)
            )
            rotated_group = _lookup_turbo_group_torch(codebooks, codes) * scales[..., None]
            group = _fwht_last_dim_torch(rotated_group)
            start = group_index * header.group_size
            end = start + header.group_size
            output[start:end] += torch.einsum("pt,ptg->g", weights, group)
        return output

    prepared_chunk = _get_prepared_chunk_mps(pages)
    if (
        prepared_chunk is not None
        and prepared_chunk.fused_scaled_codes is not None
        and prepared_chunk.bias_groups is not None
    ):
        if trace is not None:
            trace.record_per_kv_kernel_variant(section="mix", variant="fused_generic")
        output[: header.padded_head_dim] += _mix_m0_contribution_fused_torch(
            weights.reshape(1, page_count, token_count),
            prepared_chunk.fused_scaled_codes,
            prepared_chunk.bias_groups,
            group_size=header.group_size,
        ).reshape(-1)
        return output
    if trace is not None:
        trace.record_per_kv_kernel_variant(section="mix", variant="generic")
    for group_index in range(header.num_groups):
        cached_codes = prepared_chunk is not None and prepared_chunk.codes_groups is not None
        if cached_codes:
            codes = prepared_chunk.codes_groups[group_index]
        else:
            codes = _trace_timed_call(
                trace,
                "unpack",
                device_type=device_type,
                fn=lambda group_index=group_index: _unpack_bits_torch(
                    torch.stack([page.payload[group_index] for page in pages], dim=0).reshape(-1, header.words_per_group),
                    pages[0].unpack_shifts,
                    pages[0].unpack_mask,
                    header.group_size,
                ).reshape(page_count, token_count, header.group_size),
            )
        if trace is not None and not cached_codes:
            trace.record_temporary(int(codes.numel() * codes.element_size()))
        scales = (
            prepared_chunk.scales_groups[group_index]
            if prepared_chunk is not None and prepared_chunk.scales_groups is not None
            else torch.stack([page.scales[:, group_index].to(torch.float32) for page in pages], dim=0)
        )
        bias = (
            prepared_chunk.bias_groups[group_index]
            if prepared_chunk is not None and prepared_chunk.bias_groups is not None
            else torch.stack([page.bias[:, group_index].to(torch.float32) for page in pages], dim=0)
        )
        weighted_scales = weights * scales
        contribution = torch.sum(weighted_scales[..., None] * codes, dim=(0, 1))
        bias_term = torch.sum(weights * bias)
        start = group_index * header.group_size
        end = start + header.group_size
        output[start:end] += contribution + bias_term

    return output


def score_page_torch(
    query_slice: np.ndarray | Any,
    page: EncodedPage | PreparedPageTorch,
    *,
    device_type: TorchDevice,
    trace: ExecutionTrace | None = None,
) -> np.ndarray:
    prepared = prepare_page_torch(page, device_type=device_type, trace=trace)
    return _score_page_chunk_torch(query_slice, [prepared], trace=trace).detach().cpu().numpy()


def score_pages_torch(
    query_slice: np.ndarray | Any,
    pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    device_type: TorchDevice,
    trace: ExecutionTrace | None = None,
) -> list[np.ndarray]:
    prepared_pages = prepare_pages_torch(pages, device_type=device_type, trace=trace)
    if not prepared_pages:
        return []
    page_logits: list[np.ndarray] = []
    for page_chunk in _chunk_compatible_pages(prepared_pages):
        chunk_logits = _score_page_chunk_torch(query_slice, page_chunk, trace=trace)
        chunk_logits = chunk_logits.reshape(len(page_chunk), page_chunk[0].header.token_count)
        page_logits.extend(chunk_logits[index].detach().cpu().numpy() for index in range(len(page_chunk)))
    return page_logits


def mix_page_torch(
    attn_weights: np.ndarray | Any,
    page: EncodedPage | PreparedPageTorch,
    *,
    device_type: TorchDevice,
    out_acc: np.ndarray | None = None,
    trace: ExecutionTrace | None = None,
) -> np.ndarray:
    prepared = prepare_page_torch(page, device_type=device_type, trace=trace)
    header = prepared.header
    output = _mix_page_chunk_torch(
        np.asarray(attn_weights, dtype=np.float32)[None, :],
        [prepared],
        out_acc=None if out_acc is None else _prepare_output_accumulator(out_acc, header.head_dim, header.padded_head_dim, device_type=device_type),
        trace=trace,
    )
    return output[: header.head_dim].detach().cpu().numpy()


def _page_logits_tensor(page_logits, token_count: int, *, device_type: TorchDevice):
    torch = _load_torch()
    logits = torch.as_tensor(page_logits, dtype=torch.float32, device=device_type)
    if tuple(logits.shape) != (token_count,):
        raise ValueError("precomputed page logits must have shape [token_count]")
    return logits


def decode_step_torch(
    query_slice: np.ndarray | Any,
    key_pages: Sequence[EncodedPage | PreparedPageTorch],
    value_pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    device_type: TorchDevice,
    precomputed_page_logits: Sequence[np.ndarray | Any | None] | None = None,
    trace: ExecutionTrace | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torch = _load_torch()
    prepared_key_pages = prepare_pages_torch(key_pages, device_type=device_type, trace=trace)
    prepared_value_pages = prepare_pages_torch(value_pages, device_type=device_type, trace=trace)
    if not prepared_key_pages:
        raise ValueError(f"decode_step_{device_type} requires at least one page")
    if precomputed_page_logits is not None and len(precomputed_page_logits) != len(prepared_key_pages):
        raise ValueError("precomputed_page_logits must align with key_pages")

    logits_parts = []
    score_run: list[PreparedPageTorch] = []

    def flush_score_run() -> None:
        nonlocal score_run
        if not score_run:
            return
        chunk_logits = _score_page_chunk_torch(query_slice, score_run, trace=trace)
        chunk_logits = chunk_logits.reshape(len(score_run), score_run[0].header.token_count)
        logits_parts.extend(chunk_logits[index] for index in range(len(score_run)))
        score_run = []

    for index, page in enumerate(prepared_key_pages):
        cached_logits = None if precomputed_page_logits is None else precomputed_page_logits[index]
        if cached_logits is not None:
            flush_score_run()
            logits_parts.append(_page_logits_tensor(cached_logits, page.header.token_count, device_type=device_type))
            continue
        if score_run and _batched_signature(score_run[-1]) != _batched_signature(page):
            flush_score_run()
        score_run.append(page)
    flush_score_run()

    logits = torch.cat(logits_parts, dim=0)
    weights = torch.softmax(logits, dim=0)

    output = torch.zeros(
        prepared_value_pages[0].header.padded_head_dim,
        dtype=torch.float32,
        device=device_type,
    )
    offset = 0
    for page_chunk in _chunk_compatible_pages(prepared_value_pages):
        chunk_token_count = page_chunk[0].header.token_count * len(page_chunk)
        chunk_weights = weights[offset : offset + chunk_token_count].reshape(len(page_chunk), page_chunk[0].header.token_count)
        output = _mix_page_chunk_torch(chunk_weights, page_chunk, out_acc=output, trace=trace)
        offset += chunk_token_count

    head_dim = prepared_value_pages[0].header.head_dim
    return (
        logits.detach().cpu().numpy(),
        weights.detach().cpu().numpy(),
        output[:head_dim].detach().cpu().numpy(),
    )


def _score_page_chunk_multiquery_torch(
    query_slices: np.ndarray | Any,
    pages: Sequence[PreparedPageTorch],
    *,
    trace: ExecutionTrace | None = None,
):
    torch = _load_torch()
    if not pages:
        raise ValueError("pages must be non-empty")
    header = pages[0].header
    device_type = pages[0].device_type
    if torch.is_tensor(query_slices):
        query_count = int(query_slices.shape[0])
    else:
        query_count = int(np.asarray(query_slices).shape[0])
    if trace is not None:
        trace.record_page_read(
            sum(page.payload_nbytes for page in pages),
            sum(page.metadata_nbytes for page in pages),
        )

    if header.mode_default == "M3":
        use_native_dtype = _m3_native_compute_enabled(pages)
        dense = _decode_escape_batch_torch(
            pages,
            token_count=header.token_count,
            head_dim=header.head_dim,
            promote_float32=not use_native_dtype,
        )
        queries = _pad_queries(query_slices, header.head_dim, device_type=device_type)
        if use_native_dtype:
            return torch.einsum("pth,qh->qpt", dense, queries.to(dtype=dense.dtype)).reshape(query_count, -1).to(
                dtype=torch.float32
            )
        return torch.einsum("pth,qh->qpt", dense, queries).reshape(query_count, -1)

    if header.mode_default == "M2":
        queries = _pad_queries(query_slices, header.padded_head_dim, device_type=device_type)
        query_groups = queries.reshape(query_count, header.num_groups, header.group_size)
        page_count = len(pages)
        logits = torch.zeros((query_count, page_count, header.token_count), dtype=torch.float32, device=device_type)
        prepared_chunk = _get_prepared_chunk_mps(pages)
        for group_index in range(header.num_groups):
            group_sketch = (
                prepared_chunk.m2_sketch_groups[group_index]
                if prepared_chunk is not None and prepared_chunk.m2_sketch_groups is not None
                else torch.stack([page.m2_sketch[:, group_index, :] for page in pages], dim=0)
            )
            group_basis = (
                prepared_chunk.m2_basis_groups[group_index]
                if prepared_chunk is not None and prepared_chunk.m2_basis_groups is not None
                else torch.stack([page.m2_basis[group_index] for page in pages], dim=0)
            )
            group_mean = (
                prepared_chunk.m2_mean_groups[group_index]
                if prepared_chunk is not None and prepared_chunk.m2_mean_groups is not None
                else torch.stack([page.m2_mean[group_index] for page in pages], dim=0)
            )
            qg, group_sketch, group_basis, group_mean = _coerce_m2_operands(
                query_groups[:, group_index, :],
                group_sketch,
                group_basis,
                group_mean,
            )
            if group_basis.dim() == 3:
                q_proj = torch.einsum("prg,qg->qpr", group_basis, qg)
                logits += torch.einsum("ptd,qpd->qpt", group_sketch, q_proj)
                logits += torch.einsum("pg,qg->qp", group_mean, qg)[:, :, None]
                continue
            segment_ids = (
                prepared_chunk.m2_segment_ids
                if prepared_chunk is not None and prepared_chunk.m2_segment_ids is not None
                else _segment_ids_tensor(header.token_count, int(group_basis.shape[1]), device_type=device_type)
            )
            q_proj = torch.einsum("psrg,qg->qpsr", group_basis, qg)
            logits += torch.einsum("ptr,qptr->qpt", group_sketch, q_proj[:, :, segment_ids, :])
            logits += torch.einsum("ptg,qg->qpt", group_mean[:, segment_ids, :], qg)
        return logits.reshape(query_count, -1)

    if header.mode_default == "M4":
        queries = _pad_queries(query_slices, header.padded_head_dim, device_type=device_type)
        query_groups = queries.reshape(query_count, header.num_groups, header.group_size)
        page_count = len(pages)
        logits = torch.zeros((query_count, page_count, header.token_count), dtype=torch.float32, device=device_type)
        prepared_chunk = _get_prepared_chunk_mps(pages)
        for group_index in range(header.num_groups):
            group_sketch = (
                prepared_chunk.m2_sketch_groups[group_index]
                if prepared_chunk is not None and prepared_chunk.m2_sketch_groups is not None
                else torch.stack([page.m2_sketch[:, group_index, :] for page in pages], dim=0)
            )
            group_basis = (
                prepared_chunk.m2_basis_groups[group_index]
                if prepared_chunk is not None and prepared_chunk.m2_basis_groups is not None
                else (
                    torch.stack([page.m2_basis[group_index] for page in pages], dim=0)
                    if pages[0].m2_basis is not None
                    else (
                        _device_tensor(np.asarray(pages[0].source_page.m2_basis[group_index]), device=device_type).contiguous()
                        if (
                            pages[0].header.project_basis == "svd_shared"
                            and pages[0].source_page.m2_basis is not None
                            and all(page.source_page.m2_basis is pages[0].source_page.m2_basis for page in pages)
                        )
                        else None
                    )
                )
            )
            group_mean = (
                prepared_chunk.m2_mean_groups[group_index]
                if prepared_chunk is not None and prepared_chunk.m2_mean_groups is not None
                else torch.stack([page.m2_mean[group_index] for page in pages], dim=0)
            )
            qg = query_groups[:, group_index, :]
            if group_basis is not None:
                qg, group_sketch, group_basis, group_mean = _coerce_m2_operands(qg, group_sketch, group_basis, group_mean)
                if int(group_basis.dim()) == 2:
                    q_proj = torch.einsum("rg,qg->qr", group_basis, qg)
                    logits += torch.einsum("ptr,qr->qpt", group_sketch, q_proj)
                else:
                    q_proj = torch.einsum("prg,qg->qpr", group_basis, qg)
                    logits += torch.einsum("ptr,qpr->qpt", group_sketch, q_proj)
            else:
                basis = _m4_basis_torch(
                    header.group_size,
                    int(pages[0].m2_sketch.shape[-1]),
                    basis_family=header.project_basis,
                    device_type=device_type,
                )
                qg, group_sketch, _, group_mean = _coerce_m2_operands(qg, group_sketch, basis, group_mean)
                q_proj = torch.matmul(qg, basis.to(dtype=qg.dtype).transpose(0, 1))
                logits += torch.einsum("ptr,qr->qpt", group_sketch, q_proj)
            logits += torch.einsum("pg,qg->qp", group_mean, qg)[:, :, None]
        return logits.reshape(query_count, -1)

    if header.mode_default == "M1":
        queries = _pad_queries(query_slices, header.padded_head_dim, device_type=device_type)
        query_groups = queries.reshape(query_count, header.num_groups, header.group_size)
        page_count = len(pages)
        logits = torch.zeros((query_count, page_count, header.token_count), dtype=torch.float32, device=device_type)
        for group_index in range(header.num_groups):
            group_words = torch.stack([page.payload[group_index] for page in pages], dim=0)
            codes = _unpack_bits_torch(
                group_words.reshape(-1, header.words_per_group),
                pages[0].unpack_shifts,
                pages[0].unpack_mask,
                header.group_size,
            ).reshape(page_count, header.token_count, header.group_size)
            if trace is not None:
                trace.record_temporary(int(codes.numel() * codes.element_size()))
            group = _lookup_lut_group_torch(
                torch.stack([page.codebooks[group_index] for page in pages], dim=0),
                codes,
            )
            logits += torch.einsum("ptg,qg->qpt", group, query_groups[:, group_index, :])
        return logits.reshape(query_count, -1)

    if header.mode_default == "T3":
        queries = _pad_queries(query_slices, header.padded_head_dim, device_type=device_type)
        rotated_query_groups = _fwht_last_dim_torch(queries.reshape(query_count, header.num_groups, header.group_size))
        page_count = len(pages)
        logits = torch.zeros((query_count, page_count, header.token_count), dtype=torch.float32, device=device_type)
        codebooks = pages[0].codebooks if pages[0].codebooks is not None else _turbo3_centroids_torch(device_type=device_type)
        prepared_chunk = _get_prepared_chunk_mps(pages)
        for group_index in range(header.num_groups):
            group_words = (
                prepared_chunk.payload_groups[group_index]
                if prepared_chunk is not None
                else torch.stack([page.payload[group_index] for page in pages], dim=0)
            )
            codes = _unpack_bits_torch(
                group_words.reshape(-1, header.words_per_group),
                pages[0].unpack_shifts,
                pages[0].unpack_mask,
                header.group_size,
            ).reshape(page_count, header.token_count, header.group_size)
            if trace is not None:
                trace.record_temporary(int(codes.numel() * codes.element_size()))
            scales = (
                prepared_chunk.scales_groups[group_index]
                if prepared_chunk is not None and prepared_chunk.scales_groups is not None
                else torch.stack([page.scales[:, group_index] for page in pages], dim=0)
            )
            corrected = _lookup_turbo_group_torch(codebooks, codes) * scales[..., None]
            logits += torch.einsum("ptg,qg->qpt", corrected, rotated_query_groups[:, group_index, :])
        return logits.reshape(query_count, -1)

    queries = _pad_queries(query_slices, header.padded_head_dim, device_type=device_type)
    query_groups = queries.reshape(query_count, header.num_groups, header.group_size)
    query_group_sums = query_groups.sum(dim=-1)
    page_count = len(pages)
    logits = torch.zeros((query_count, page_count * header.token_count), dtype=torch.float32, device=device_type)

    prepared_chunk = _get_prepared_chunk_mps(pages)
    if (
        prepared_chunk is not None
        and prepared_chunk.fused_scaled_codes is not None
        and prepared_chunk.bias_groups is not None
        and _supports_fused_two_group64(header)
    ):
        if trace is not None:
            trace.record_per_kv_kernel_variant(section="score", variant="fused_two_group64")
        fused_queries = query_groups.reshape(query_count, header.padded_head_dim).contiguous()
        return _score_m0_logits_two_group64_torch(
            prepared_chunk.fused_scaled_codes,
            fused_queries,
            prepared_chunk.bias_groups,
            query_group_sums,
        )
    if trace is not None:
        trace.record_per_kv_kernel_variant(section="score", variant="generic")
    for group_index in range(header.num_groups):
        cached_codes = prepared_chunk is not None and prepared_chunk.codes_groups is not None
        if cached_codes:
            codes = prepared_chunk.codes_groups[group_index]
        else:
            codes = _trace_timed_call(
                trace,
                "unpack",
                device_type=device_type,
                fn=lambda group_index=group_index: _unpack_bits_torch(
                    torch.stack([page.payload[group_index] for page in pages], dim=0).reshape(-1, header.words_per_group),
                    pages[0].unpack_shifts,
                    pages[0].unpack_mask,
                    header.group_size,
                ).reshape(page_count, header.token_count, header.group_size),
            )
        if trace is not None and not cached_codes:
            trace.record_temporary(int(codes.numel() * codes.element_size()))
        qg = query_groups[:, group_index, :]
        scales = (
            prepared_chunk.scales_groups[group_index]
            if prepared_chunk is not None and prepared_chunk.scales_groups is not None
            else torch.stack([page.scales[:, group_index].to(torch.float32) for page in pages], dim=0)
        )
        bias = (
            prepared_chunk.bias_groups[group_index]
            if prepared_chunk is not None and prepared_chunk.bias_groups is not None
            else torch.stack([page.bias[:, group_index].to(torch.float32) for page in pages], dim=0)
        )
        logits += _score_m0_logits_flat_torch(
            codes,
            qg,
            scales,
            bias,
            query_group_sums[:, group_index],
        )

    return logits


def _mix_page_chunk_multiquery_torch(
    attn_weights,
    pages: Sequence[PreparedPageTorch],
    *,
    trace: ExecutionTrace | None = None,
):
    torch = _load_torch()
    if not pages:
        raise ValueError("pages must be non-empty")
    header = pages[0].header
    device_type = pages[0].device_type
    page_count = len(pages)
    token_count = header.token_count

    if trace is not None:
        trace.record_page_read(
            sum(page.payload_nbytes for page in pages),
            sum(page.metadata_nbytes for page in pages),
        )

    if not isinstance(attn_weights, torch.Tensor):
        weights = torch.as_tensor(attn_weights, dtype=torch.float32, device=device_type)
    else:
        weights = attn_weights.to(dtype=torch.float32, device=device_type)
    if weights.ndim != 3 or tuple(weights.shape[1:]) != (page_count, token_count):
        raise ValueError("attn_weights chunk must have shape [query_count, page_count, token_count]")

    query_count = int(weights.shape[0])
    output = torch.zeros((query_count, header.padded_head_dim), dtype=torch.float32, device=device_type)

    if header.mode_default == "M3":
        use_native_dtype = _m3_native_compute_enabled(pages)
        dense = _decode_escape_batch_torch(
            pages,
            token_count=header.token_count,
            head_dim=header.head_dim,
            promote_float32=not use_native_dtype,
        )
        if use_native_dtype:
            output[:, : header.head_dim] += torch.einsum(
                "qpt,pth->qh",
                weights.to(dtype=dense.dtype),
                dense,
            ).to(dtype=torch.float32)
        else:
            output[:, : header.head_dim] += torch.einsum("qpt,pth->qh", weights, dense)
        return output

    if header.mode_default in {"M2", "M4"}:
        raise ValueError(f"{header.mode_default} is only supported for key scoring in this phase")

    if header.mode_default == "M1":
        for group_index in range(header.num_groups):
            group_words = torch.stack([page.payload[group_index] for page in pages], dim=0)
            codes = _unpack_bits_torch(
                group_words.reshape(-1, header.words_per_group),
                pages[0].unpack_shifts,
                pages[0].unpack_mask,
                header.group_size,
            ).reshape(page_count, token_count, header.group_size)
            if trace is not None:
                trace.record_temporary(int(codes.numel() * codes.element_size()))
            group = _lookup_lut_group_torch(
                torch.stack([page.codebooks[group_index] for page in pages], dim=0),
                codes,
            )
            start = group_index * header.group_size
            end = start + header.group_size
            output[:, start:end] += torch.einsum("qpt,ptg->qg", weights, group)
        return output

    if header.mode_default == "T3":
        codebooks = pages[0].codebooks if pages[0].codebooks is not None else _turbo3_centroids_torch(device_type=device_type)
        prepared_chunk = _get_prepared_chunk_mps(pages)
        for group_index in range(header.num_groups):
            group_words = (
                prepared_chunk.payload_groups[group_index]
                if prepared_chunk is not None
                else torch.stack([page.payload[group_index] for page in pages], dim=0)
            )
            codes = _unpack_bits_torch(
                group_words.reshape(-1, header.words_per_group),
                pages[0].unpack_shifts,
                pages[0].unpack_mask,
                header.group_size,
            ).reshape(page_count, token_count, header.group_size)
            if trace is not None:
                trace.record_temporary(int(codes.numel() * codes.element_size()))
            scales = (
                prepared_chunk.scales_groups[group_index]
                if prepared_chunk is not None and prepared_chunk.scales_groups is not None
                else torch.stack([page.scales[:, group_index] for page in pages], dim=0)
            )
            rotated_group = _lookup_turbo_group_torch(codebooks, codes) * scales[..., None]
            group = _fwht_last_dim_torch(rotated_group)
            start = group_index * header.group_size
            end = start + header.group_size
            output[:, start:end] += torch.einsum("qpt,ptg->qg", weights, group)
        return output

    prepared_chunk = _get_prepared_chunk_mps(pages)
    if (
        prepared_chunk is not None
        and prepared_chunk.fused_scaled_codes is not None
        and prepared_chunk.bias_groups is not None
        and _supports_fused_two_group64(header)
    ):
        if trace is not None:
            trace.record_per_kv_kernel_variant(section="mix", variant="fused_two_group64")
        output[:, : header.padded_head_dim] += _mix_m0_contribution_two_group64_torch(
            weights,
            prepared_chunk.fused_scaled_codes,
            prepared_chunk.bias_groups,
        )
        return output
    if trace is not None:
        trace.record_per_kv_kernel_variant(section="mix", variant="generic")
    for group_index in range(header.num_groups):
        cached_codes = prepared_chunk is not None and prepared_chunk.codes_groups is not None
        if cached_codes:
            codes = prepared_chunk.codes_groups[group_index]
        else:
            codes = _trace_timed_call(
                trace,
                "unpack",
                device_type=device_type,
                fn=lambda group_index=group_index: _unpack_bits_torch(
                    torch.stack([page.payload[group_index] for page in pages], dim=0).reshape(-1, header.words_per_group),
                    pages[0].unpack_shifts,
                    pages[0].unpack_mask,
                    header.group_size,
                ).reshape(page_count, token_count, header.group_size),
            )
        if trace is not None and not cached_codes:
            trace.record_temporary(int(codes.numel() * codes.element_size()))
        scales = (
            prepared_chunk.scales_groups[group_index]
            if prepared_chunk is not None and prepared_chunk.scales_groups is not None
            else torch.stack([page.scales[:, group_index].to(torch.float32) for page in pages], dim=0)
        )
        bias = (
            prepared_chunk.bias_groups[group_index]
            if prepared_chunk is not None and prepared_chunk.bias_groups is not None
            else torch.stack([page.bias[:, group_index].to(torch.float32) for page in pages], dim=0)
        )
        start = group_index * header.group_size
        end = start + header.group_size
        output[:, start:end] += _mix_m0_contribution_torch(weights, codes, scales, bias)

    return output


def _score_page_chunk_grouped_multiquery_torch(
    query_groups,
    pages_by_group: Sequence[Sequence[PreparedPageTorch]],
    *,
    prepared_query_groups_tensor=None,
    query_group_sums=None,
    compact_grouped_chunk: bool = False,
    trace: ExecutionTrace | None = None,
):
    torch = _load_torch()
    if not pages_by_group or not pages_by_group[0]:
        raise ValueError("pages_by_group must be non-empty")
    batch_size = len(pages_by_group)
    page_count = len(pages_by_group[0])
    header = pages_by_group[0][0].header
    device_type = pages_by_group[0][0].device_type

    for group_pages in pages_by_group:
        if len(group_pages) != page_count:
            raise ValueError("all page groups must have the same page count")

    if torch.is_tensor(query_groups):
        queries = query_groups.to(dtype=torch.float32, device=device_type)
    else:
        queries = torch.stack(
            [
                group.to(dtype=torch.float32, device=device_type)
                if torch.is_tensor(group)
                else torch.as_tensor(group, dtype=torch.float32, device=device_type)
                for group in query_groups
            ],
            dim=0,
        )
    if queries.ndim != 3:
        raise ValueError("query_groups must have shape [batch_size, query_count, head_dim]")
    query_count = int(queries.shape[1])
    if int(queries.shape[0]) != batch_size:
        raise ValueError("query_groups batch size must align with pages_by_group")

    signature_buckets = _signature_buckets_for_page_chunk(pages_by_group)
    if len(signature_buckets) > 1:
        bucket_logits = torch.zeros(
            (batch_size, query_count, page_count * header.token_count),
            dtype=torch.float32,
            device=device_type,
        )
        for group_indices in signature_buckets:
            group_index_list = list(group_indices)
            sub_prepared_queries = (
                None
                if prepared_query_groups_tensor is None
                else prepared_query_groups_tensor[group_index_list]
            )
            sub_query_sums = None if query_group_sums is None else query_group_sums[group_index_list]
            bucket_logits[group_index_list] = _score_page_chunk_grouped_multiquery_torch(
                queries[group_index_list],
                [pages_by_group[group_index] for group_index in group_index_list],
                prepared_query_groups_tensor=sub_prepared_queries,
                query_group_sums=sub_query_sums,
                compact_grouped_chunk=compact_grouped_chunk,
                trace=trace,
            )
        return bucket_logits

    if trace is not None:
        trace.record_page_read(
            sum(page.payload_nbytes for group_pages in pages_by_group for page in group_pages),
            sum(page.metadata_nbytes for group_pages in pages_by_group for page in group_pages),
        )

    if header.mode_default == "M3":
        use_native_dtype = _m3_native_compute_enabled(pages_by_group[0])
        dense = torch.stack(
            [
                _decode_escape_batch_torch(
                    group_pages,
                    token_count=header.token_count,
                    head_dim=header.head_dim,
                    promote_float32=not use_native_dtype,
                )
                for group_pages in pages_by_group
            ],
            dim=0,
        )
        if use_native_dtype:
            return torch.einsum("bpth,bqh->bqpt", dense, queries.to(dtype=dense.dtype)).reshape(
                batch_size, query_count, -1
            ).to(dtype=torch.float32)
        return torch.einsum("bpth,bqh->bqpt", dense, queries).reshape(batch_size, query_count, -1)

    if header.mode_default == "M2":
        query_groups_tensor = prepared_query_groups_tensor
        if query_groups_tensor is None:
            padded_queries = _pad_queries(
                queries.reshape(batch_size * query_count, header.head_dim),
                header.padded_head_dim,
                device_type=device_type,
            ).reshape(batch_size, query_count, header.padded_head_dim)
            query_groups_tensor = padded_queries.reshape(batch_size, query_count, header.num_groups, header.group_size)
        logits = torch.zeros((batch_size, query_count, page_count, header.token_count), dtype=torch.float32, device=device_type)
        grouped_prepared_chunk = (
            _build_grouped_prepared_chunk_mps(pages_by_group)
            if compact_grouped_chunk
            else _get_grouped_prepared_chunk_mps(pages_by_group)
        )
        if (
            grouped_prepared_chunk is not None
            and grouped_prepared_chunk.m2_sketch_tensor is not None
            and grouped_prepared_chunk.m2_basis_tensor is not None
            and grouped_prepared_chunk.m2_mean_tensor is not None
        ):
            qg, sketch_tensor, basis_tensor, mean_tensor = _coerce_m2_grouped_operands(
                query_groups_tensor,
                grouped_prepared_chunk.m2_sketch_tensor,
                grouped_prepared_chunk.m2_basis_tensor,
                grouped_prepared_chunk.m2_mean_tensor,
            )
            if int(basis_tensor.shape[3]) == 1:
                squeezed_basis = basis_tensor.squeeze(3)
                squeezed_mean = mean_tensor.squeeze(3)
                q_proj = torch.einsum("bpgrc,bqgc->bqpgr", squeezed_basis, qg)
                logits += torch.einsum("bptgr,bqpgr->bqpt", sketch_tensor, q_proj)
                logits += torch.einsum("bpgc,bqgc->bqp", squeezed_mean, qg)[:, :, :, None]
                return logits.reshape(batch_size, query_count, -1)
            segment_ids = (
                grouped_prepared_chunk.m2_segment_ids
                if grouped_prepared_chunk.m2_segment_ids is not None
                else _segment_ids_tensor(header.token_count, int(basis_tensor.shape[3]), device_type=device_type)
            )
            q_proj = torch.einsum("bpgsrc,bqgc->bqpgsr", basis_tensor, qg)
            logits += torch.einsum("bptgr,bqpgtr->bqpt", sketch_tensor, q_proj[:, :, :, :, segment_ids, :])
            logits += torch.einsum("bpgtc,bqgc->bqpt", mean_tensor[:, :, :, segment_ids, :], qg)
            return logits.reshape(batch_size, query_count, -1)
        prepared_chunks = None if grouped_prepared_chunk is not None else [
            (
                _build_prepared_chunk_mps(group_pages)
                if compact_grouped_chunk
                else _get_prepared_chunk_mps(group_pages) or (
                    _build_prepared_chunk_mps(group_pages) if _prepared_chunk_cache_key(group_pages) is not None else None
                )
            )
            for group_pages in pages_by_group
        ]
        for group_index in range(header.num_groups):
            if grouped_prepared_chunk is not None and grouped_prepared_chunk.m2_sketch_groups is not None:
                group_sketch = grouped_prepared_chunk.m2_sketch_groups[group_index]
                group_basis = grouped_prepared_chunk.m2_basis_groups[group_index]
                group_mean = grouped_prepared_chunk.m2_mean_groups[group_index]
            elif prepared_chunks is not None and all(
                chunk is not None and chunk.m2_sketch_groups is not None for chunk in prepared_chunks
            ):
                group_sketch = torch.stack([chunk.m2_sketch_groups[group_index] for chunk in prepared_chunks], dim=0)
                group_basis = torch.stack([chunk.m2_basis_groups[group_index] for chunk in prepared_chunks], dim=0)
                group_mean = torch.stack([chunk.m2_mean_groups[group_index] for chunk in prepared_chunks], dim=0)
            else:
                group_sketch = torch.stack(
                    [torch.stack([page.m2_sketch[:, group_index, :] for page in group_pages], dim=0) for group_pages in pages_by_group],
                    dim=0,
                )
                group_basis = torch.stack(
                    [torch.stack([page.m2_basis[group_index] for page in group_pages], dim=0) for group_pages in pages_by_group],
                    dim=0,
                )
                group_mean = torch.stack(
                    [torch.stack([page.m2_mean[group_index] for page in group_pages], dim=0) for group_pages in pages_by_group],
                    dim=0,
                )
            qg, group_sketch, group_basis, group_mean = _coerce_m2_operands(
                query_groups_tensor[:, :, group_index, :],
                group_sketch,
                group_basis,
                group_mean,
            )
            if group_basis.dim() == 4:
                q_proj = torch.einsum("bprg,bqg->bqpr", group_basis, qg)
                logits += torch.einsum("bptd,bqpd->bqpt", group_sketch, q_proj)
                logits += torch.einsum("bpg,bqg->bqp", group_mean, qg)[:, :, :, None]
                continue
            segment_ids = (
                grouped_prepared_chunk.m2_segment_ids
                if grouped_prepared_chunk is not None and grouped_prepared_chunk.m2_segment_ids is not None
                else (
                    prepared_chunks[0].m2_segment_ids
                    if prepared_chunks is not None and prepared_chunks[0] is not None and prepared_chunks[0].m2_segment_ids is not None
                    else _segment_ids_tensor(header.token_count, int(group_basis.shape[2]), device_type=device_type)
                )
            )
            q_proj = torch.einsum("bpsrg,bqg->bqpsr", group_basis, qg)
            logits += torch.einsum("bptr,bqptr->bqpt", group_sketch, q_proj[:, :, :, segment_ids, :])
            logits += torch.einsum("bptg,bqg->bqpt", group_mean[:, :, segment_ids, :], qg)
        return logits.reshape(batch_size, query_count, -1)

    if header.mode_default == "M4":
        query_groups_tensor = prepared_query_groups_tensor
        if query_groups_tensor is None:
            padded_queries = _pad_queries(
                queries.reshape(batch_size * query_count, header.head_dim),
                header.padded_head_dim,
                device_type=device_type,
            ).reshape(batch_size, query_count, header.padded_head_dim)
            query_groups_tensor = padded_queries.reshape(batch_size, query_count, header.num_groups, header.group_size)
        logits = torch.zeros((batch_size, query_count, page_count, header.token_count), dtype=torch.float32, device=device_type)
        grouped_prepared_chunk = (
            _build_grouped_prepared_chunk_mps(pages_by_group)
            if compact_grouped_chunk
            else _get_grouped_prepared_chunk_mps(pages_by_group)
        )
        if (
            grouped_prepared_chunk is not None
            and grouped_prepared_chunk.m2_sketch_tensor is not None
            and grouped_prepared_chunk.m2_mean_tensor is not None
        ):
            if grouped_prepared_chunk.m2_basis_tensor is not None:
                qg, sketch_tensor, basis_tensor, mean_tensor = _coerce_m2_grouped_operands(
                    query_groups_tensor,
                    grouped_prepared_chunk.m2_sketch_tensor,
                    grouped_prepared_chunk.m2_basis_tensor,
                    grouped_prepared_chunk.m2_mean_tensor,
                )
                if int(basis_tensor.dim()) == 4:
                    q_proj = torch.einsum("bgrc,bqgc->bqgr", basis_tensor, qg)
                    logits += torch.einsum("bptgr,bqgr->bqpt", sketch_tensor, q_proj)
                else:
                    q_proj = torch.einsum("bpgrc,bqgc->bqpgr", basis_tensor, qg)
                    logits += torch.einsum("bptgr,bqpgr->bqpt", sketch_tensor, q_proj)
            else:
                basis = _m4_basis_torch(
                    header.group_size,
                    int(pages_by_group[0][0].m2_sketch.shape[-1]),
                    basis_family=header.project_basis,
                    device_type=device_type,
                )
                qg, sketch_tensor, _, mean_tensor = _coerce_m2_grouped_operands(
                    query_groups_tensor,
                    grouped_prepared_chunk.m2_sketch_tensor,
                    basis.view(1, 1, 1, *basis.shape),
                    grouped_prepared_chunk.m2_mean_tensor,
                )
                basis_work = basis.to(dtype=qg.dtype)
                q_proj = torch.einsum("rc,bqgc->bqgr", basis_work, qg)
                logits += torch.einsum("bptgr,bqgr->bqpt", sketch_tensor, q_proj)
            logits += torch.einsum("bpgc,bqgc->bqp", mean_tensor, qg)[:, :, :, None]
            return logits.reshape(batch_size, query_count, -1)
        prepared_chunks = None if grouped_prepared_chunk is not None else [
            (
                _build_prepared_chunk_mps(group_pages)
                if compact_grouped_chunk
                else _get_prepared_chunk_mps(group_pages) or (
                    _build_prepared_chunk_mps(group_pages) if _prepared_chunk_cache_key(group_pages) is not None else None
                )
            )
            for group_pages in pages_by_group
        ]
        for group_index in range(header.num_groups):
            if grouped_prepared_chunk is not None and grouped_prepared_chunk.m2_sketch_groups is not None:
                group_sketch = grouped_prepared_chunk.m2_sketch_groups[group_index]
                group_basis = grouped_prepared_chunk.m2_basis_groups[group_index]
                group_mean = grouped_prepared_chunk.m2_mean_groups[group_index]
            elif prepared_chunks is not None and all(
                chunk is not None and chunk.m2_sketch_groups is not None for chunk in prepared_chunks
            ):
                group_sketch = torch.stack([chunk.m2_sketch_groups[group_index] for chunk in prepared_chunks], dim=0)
                group_basis = (
                    torch.stack([chunk.m2_basis_groups[group_index] for chunk in prepared_chunks], dim=0)
                    if prepared_chunks[0].m2_basis_groups is not None
                    else None
                )
                group_mean = torch.stack([chunk.m2_mean_groups[group_index] for chunk in prepared_chunks], dim=0)
            else:
                group_sketch = torch.stack(
                    [torch.stack([page.m2_sketch[:, group_index, :] for page in group_pages], dim=0) for group_pages in pages_by_group],
                    dim=0,
                )
                group_basis = (
                    torch.stack(
                        [torch.stack([page.m2_basis[group_index] for page in group_pages], dim=0) for group_pages in pages_by_group],
                        dim=0,
                    )
                    if pages_by_group[0][0].m2_basis is not None
                    else None
                )
                group_mean = torch.stack(
                    [torch.stack([page.m2_mean[group_index] for page in group_pages], dim=0) for group_pages in pages_by_group],
                    dim=0,
                )
            qg = query_groups_tensor[:, :, group_index, :]
            if group_basis is not None:
                qg, group_sketch, group_basis, group_mean = _coerce_m2_operands(qg, group_sketch, group_basis, group_mean)
                if int(group_basis.dim()) == 2:
                    q_proj = torch.einsum("rg,bqg->bqr", group_basis, qg)
                    logits += torch.einsum("bptr,bqr->bqpt", group_sketch, q_proj)
                elif int(group_basis.dim()) == 3:
                    q_proj = torch.einsum("brg,bqg->bqr", group_basis, qg)
                    logits += torch.einsum("bptr,bqr->bqpt", group_sketch, q_proj)
                else:
                    q_proj = torch.einsum("bprg,bqg->bqpr", group_basis, qg)
                    logits += torch.einsum("bptr,bqpr->bqpt", group_sketch, q_proj)
            else:
                basis = _m4_basis_torch(
                    header.group_size,
                    int(pages_by_group[0][0].m2_sketch.shape[-1]),
                    basis_family=header.project_basis,
                    device_type=device_type,
                )
                qg, group_sketch, _, group_mean = _coerce_m2_operands(qg, group_sketch, basis, group_mean)
                basis_work = basis.to(dtype=qg.dtype)
                q_proj = torch.einsum("rc,bqc->bqr", basis_work, qg)
                logits += torch.einsum("bptr,bqr->bqpt", group_sketch, q_proj)
            logits += torch.einsum("bpg,bqg->bqp", group_mean, qg)[:, :, :, None]
        return logits.reshape(batch_size, query_count, -1)

    if header.mode_default == "M1":
        query_groups_tensor = prepared_query_groups_tensor
        if query_groups_tensor is None:
            padded_queries = _pad_queries(
                queries.reshape(batch_size * query_count, header.head_dim),
                header.padded_head_dim,
                device_type=device_type,
            ).reshape(batch_size, query_count, header.padded_head_dim)
            query_groups_tensor = padded_queries.reshape(batch_size, query_count, header.num_groups, header.group_size)
        logits = torch.zeros((batch_size, query_count, page_count, header.token_count), dtype=torch.float32, device=device_type)
        for group_index in range(header.num_groups):
            group_words = torch.stack(
                [torch.stack([page.payload[group_index] for page in group_pages], dim=0) for group_pages in pages_by_group],
                dim=0,
            )
            codes = _unpack_bits_torch(
                group_words.reshape(-1, header.words_per_group),
                pages_by_group[0][0].unpack_shifts,
                pages_by_group[0][0].unpack_mask,
                header.group_size,
                trace=trace,
            ).reshape(batch_size, page_count, header.token_count, header.group_size)
            if trace is not None:
                trace.record_temporary(int(codes.numel() * codes.element_size()))
            group = _lookup_lut_group_torch(
                torch.stack(
                    [
                        torch.stack([page.codebooks[group_index] for page in group_pages], dim=0)
                        for group_pages in pages_by_group
                    ],
                    dim=0,
                ),
                codes,
            )
            logits += torch.einsum("bptg,bqg->bqpt", group, query_groups_tensor[:, :, group_index, :])
        return logits.reshape(batch_size, query_count, -1)

    if header.mode_default == "T3":
        query_groups_tensor = prepared_query_groups_tensor
        if query_groups_tensor is None:
            padded_queries = _pad_queries(
                queries.reshape(batch_size * query_count, header.head_dim),
                header.padded_head_dim,
                device_type=device_type,
            ).reshape(batch_size, query_count, header.padded_head_dim)
            query_groups_tensor = padded_queries.reshape(batch_size, query_count, header.num_groups, header.group_size)
        rotated_query_groups = _fwht_last_dim_torch(
            query_groups_tensor,
            trace=trace,
        )
        logits = torch.zeros((batch_size, query_count, page_count, header.token_count), dtype=torch.float32, device=device_type)
        codebooks = pages_by_group[0][0].codebooks if pages_by_group[0][0].codebooks is not None else _turbo3_centroids_torch(device_type=device_type)
        prepared_chunks = [_get_prepared_chunk_mps(group_pages) for group_pages in pages_by_group]
        for group_index in range(header.num_groups):
            group_words = torch.stack(
                [
                    prepared_chunks[group_id].payload_groups[group_index]
                    if prepared_chunks[group_id] is not None
                    else torch.stack([page.payload[group_index] for page in group_pages], dim=0)
                    for group_id, group_pages in enumerate(pages_by_group)
                ],
                dim=0,
            )
            codes = _unpack_bits_torch(
                group_words.reshape(-1, header.words_per_group),
                pages_by_group[0][0].unpack_shifts,
                pages_by_group[0][0].unpack_mask,
                header.group_size,
                trace=trace,
            ).reshape(batch_size, page_count, header.token_count, header.group_size)
            if trace is not None:
                trace.record_temporary(int(codes.numel() * codes.element_size()))
            scales = torch.stack(
                [
                    prepared_chunks[group_id].scales_groups[group_index]
                    if prepared_chunks[group_id] is not None and prepared_chunks[group_id].scales_groups is not None
                    else torch.stack([page.scales[:, group_index] for page in group_pages], dim=0)
                    for group_id, group_pages in enumerate(pages_by_group)
                ],
                dim=0,
            )
            corrected = _lookup_turbo_group_torch(codebooks, codes) * scales[..., None]
            logits += torch.einsum("bptg,bqg->bqpt", corrected, rotated_query_groups[:, :, group_index, :])
        return logits.reshape(batch_size, query_count, -1)

    query_groups_tensor = prepared_query_groups_tensor
    if query_groups_tensor is None:
        padded_queries = _pad_queries(
            queries.reshape(batch_size * query_count, header.head_dim),
            header.padded_head_dim,
            device_type=device_type,
        ).reshape(batch_size, query_count, header.padded_head_dim)
        query_groups_tensor = padded_queries.reshape(batch_size, query_count, header.num_groups, header.group_size)
    query_group_sums_tensor = query_group_sums if query_group_sums is not None else query_groups_tensor.sum(dim=-1)
    logits = torch.zeros((batch_size, query_count, page_count * header.token_count), dtype=torch.float32, device=device_type)
    grouped_prepared_chunk = (
        _build_grouped_prepared_chunk_mps(pages_by_group)
        if compact_grouped_chunk
        else _get_grouped_prepared_chunk_mps(pages_by_group)
    )
    prepared_chunks = None if grouped_prepared_chunk is not None else [
        (
            _build_prepared_chunk_mps(group_pages)
            if compact_grouped_chunk
            else _get_prepared_chunk_mps(group_pages) or (
                _build_prepared_chunk_mps(group_pages) if _prepared_chunk_cache_key(group_pages) is not None else None
            )
        )
        for group_pages in pages_by_group
    ]
    if (
        grouped_prepared_chunk is not None
        and grouped_prepared_chunk.payload_groups
        and grouped_prepared_chunk.scales_groups is not None
        and grouped_prepared_chunk.bias_groups is not None
        and grouped_prepared_chunk.payload_groups_tensor is not None
        and grouped_prepared_chunk.scales_groups_tensor is not None
        and grouped_prepared_chunk.bias_groups_tensor is not None
        and _supports_packed_four_group128_cuda(header, device_type=device_type)
    ):
        unpack_shifts = pages_by_group[0][0].unpack_shifts
        unpack_mask = pages_by_group[0][0].unpack_mask
        if unpack_shifts is None or unpack_mask is None:
            raise ValueError("packed grouped CUDA path requires unpack metadata")
        if trace is not None:
            trace.record_grouped_kernel_variant(section="score", variant="packed_cuda")
        for group_start in range(0, header.num_groups, 2):
            group_end = min(header.num_groups, group_start + 2)
            logits += _score_m0_logits_packed32_grouped_torch(
                grouped_prepared_chunk.payload_groups_tensor[:, group_start:group_end],
                query_groups_tensor[:, :, group_start:group_end, :],
                grouped_prepared_chunk.scales_groups_tensor[:, group_start:group_end],
                grouped_prepared_chunk.bias_groups_tensor[:, group_start:group_end],
                query_group_sums_tensor[:, :, group_start:group_end],
                unpack_shifts=unpack_shifts,
                unpack_mask=unpack_mask,
                trace=trace,
            )
        return logits
    if (
        grouped_prepared_chunk is not None
        and grouped_prepared_chunk.fused_scaled_codes is not None
        and grouped_prepared_chunk.bias_groups is not None
        and _supports_fused_two_group64(header)
    ):
        if trace is not None:
            trace.record_grouped_kernel_variant(section="score", variant="fused_two_group64")
        fused_queries = query_groups_tensor.reshape(batch_size, query_count, header.padded_head_dim).contiguous()
        return _score_m0_logits_two_group64_torch(
            grouped_prepared_chunk.fused_scaled_codes,
            fused_queries,
            grouped_prepared_chunk.bias_groups,
            query_group_sums_tensor,
        )
    if (
        grouped_prepared_chunk is not None
        and grouped_prepared_chunk.fused_scaled_codes is not None
        and grouped_prepared_chunk.bias_groups is not None
    ):
        if trace is not None:
            trace.record_grouped_kernel_variant(section="score", variant="fused_generic")
        fused_queries = query_groups_tensor.reshape(batch_size, query_count, header.padded_head_dim).contiguous()
        return _score_m0_logits_fused_torch(
            grouped_prepared_chunk.fused_scaled_codes,
            fused_queries,
            grouped_prepared_chunk.bias_groups,
            query_group_sums_tensor,
        )
    if (
        grouped_prepared_chunk is None
        and prepared_chunks is not None
        and _supports_fused_two_group64(header)
        and all(chunk is not None and chunk.fused_scaled_codes is not None and chunk.bias_groups is not None for chunk in prepared_chunks)
    ):
        if trace is not None:
            trace.record_grouped_kernel_variant(section="score", variant="fused_two_group64")
        fused_scaled_codes, bias_groups = _assemble_grouped_fused_two_group64_components(
            prepared_chunks,
            trace=trace,
            device_type=device_type,
        )
        fused_queries = query_groups_tensor.reshape(batch_size, query_count, header.padded_head_dim).contiguous()
        return _score_m0_logits_two_group64_torch(
            fused_scaled_codes,
            fused_queries,
            bias_groups,
            query_group_sums_tensor,
        )
    if (
        grouped_prepared_chunk is None
        and prepared_chunks is not None
        and all(chunk is not None and chunk.fused_scaled_codes is not None and chunk.bias_groups is not None for chunk in prepared_chunks)
    ):
        if trace is not None:
            trace.record_grouped_kernel_variant(section="score", variant="fused_generic")
        fused_scaled_codes = _trace_timed_call(
            trace,
            "chunk_assembly",
            device_type=device_type,
            fn=lambda: _load_torch().stack([chunk.fused_scaled_codes for chunk in prepared_chunks], dim=0),
        )
        bias_groups = tuple(
            _trace_timed_call(
                trace,
                "chunk_assembly",
                device_type=device_type,
                fn=lambda group_index=group_index: _load_torch().stack([chunk.bias_groups[group_index] for chunk in prepared_chunks], dim=0),
            )
            for group_index in range(header.num_groups)
        )
        if trace is not None:
            trace.record_temporary(int(fused_scaled_codes.numel() * fused_scaled_codes.element_size()))
            trace.record_temporary(sum(int(tensor.numel() * tensor.element_size()) for tensor in bias_groups))
        fused_queries = query_groups_tensor.reshape(batch_size, query_count, header.padded_head_dim).contiguous()
        return _score_m0_logits_fused_torch(
            fused_scaled_codes,
            fused_queries,
            bias_groups,
            query_group_sums_tensor,
        )
    if trace is not None:
        trace.record_grouped_kernel_variant(section="score", variant="generic")
    for group_index in range(header.num_groups):
        cached_codes = grouped_prepared_chunk is not None and grouped_prepared_chunk.codes_groups is not None
        if cached_codes:
            codes = grouped_prepared_chunk.codes_groups[group_index]
        else:
            def _build_codes(group_index: int = group_index):
                return torch.stack(
                    [
                        prepared_chunks[group_id].codes_groups[group_index]
                        if prepared_chunks is not None and prepared_chunks[group_id] is not None and prepared_chunks[group_id].codes_groups is not None
                        else _trace_timed_call(
                            trace,
                            "unpack",
                            device_type=device_type,
                            fn=lambda group_pages=group_pages, group_index=group_index: _unpack_bits_torch(
                                torch.stack([page.payload[group_index] for page in group_pages], dim=0).reshape(-1, header.words_per_group),
                                pages_by_group[0][0].unpack_shifts,
                                pages_by_group[0][0].unpack_mask,
                                header.group_size,
                            ).reshape(page_count, header.token_count, header.group_size),
                        )
                        for group_id, group_pages in enumerate(pages_by_group)
                    ],
                    dim=0,
                )

            codes = _trace_timed_call(
                trace,
                "chunk_assembly",
                device_type=device_type,
                fn=_build_codes,
            )
        if trace is not None and not cached_codes:
            trace.record_temporary(int(codes.numel() * codes.element_size()))
        qg = query_groups_tensor[:, :, group_index, :]
        scales = (
            grouped_prepared_chunk.scales_groups[group_index]
            if grouped_prepared_chunk is not None and grouped_prepared_chunk.scales_groups is not None
            else torch.stack(
                [
                    prepared_chunks[group_id].scales_groups[group_index]
                    if prepared_chunks is not None and prepared_chunks[group_id] is not None and prepared_chunks[group_id].scales_groups is not None
                    else torch.stack([page.scales[:, group_index].to(torch.float32) for page in group_pages], dim=0)
                    for group_id, group_pages in enumerate(pages_by_group)
                ],
                dim=0,
            )
        )
        bias = (
            grouped_prepared_chunk.bias_groups[group_index]
            if grouped_prepared_chunk is not None and grouped_prepared_chunk.bias_groups is not None
            else torch.stack(
                [
                    prepared_chunks[group_id].bias_groups[group_index]
                    if prepared_chunks is not None and prepared_chunks[group_id] is not None and prepared_chunks[group_id].bias_groups is not None
                    else torch.stack([page.bias[:, group_index].to(torch.float32) for page in group_pages], dim=0)
                    for group_id, group_pages in enumerate(pages_by_group)
                ],
                dim=0,
            )
        )
        logits += _score_m0_logits_flat_torch(
            codes,
            qg,
            scales,
            bias,
            query_group_sums_tensor[:, :, group_index],
        )

    return logits


def _mix_page_chunk_grouped_multiquery_torch(
    attn_weights,
    pages_by_group: Sequence[Sequence[PreparedPageTorch]],
    *,
    out_acc=None,
    compact_grouped_chunk: bool = False,
    disable_packed_grouped_cuda: bool = False,
    trace: ExecutionTrace | None = None,
):
    torch = _load_torch()
    if not pages_by_group or not pages_by_group[0]:
        raise ValueError("pages_by_group must be non-empty")
    batch_size = len(pages_by_group)
    page_count = len(pages_by_group[0])
    header = pages_by_group[0][0].header
    device_type = pages_by_group[0][0].device_type
    token_count = header.token_count

    weights = attn_weights if isinstance(attn_weights, torch.Tensor) else torch.as_tensor(attn_weights, dtype=torch.float32, device=device_type)
    weights = weights.to(dtype=torch.float32, device=device_type)
    if weights.ndim != 4 or tuple(weights.shape[2:]) != (page_count, token_count):
        raise ValueError("grouped attn_weights chunk must have shape [batch_size, query_count, page_count, token_count]")

    query_count = int(weights.shape[1])
    output = _prepare_grouped_output_accumulator_tensor(
        out_acc,
        batch_size,
        query_count,
        header.head_dim,
        header.padded_head_dim,
        device_type=device_type,
    )

    signature_buckets = _signature_buckets_for_page_chunk(pages_by_group)
    if len(signature_buckets) > 1:
        for group_indices in signature_buckets:
            group_index_list = list(group_indices)
            output[group_index_list] = _mix_page_chunk_grouped_multiquery_torch(
                weights[group_index_list],
                [pages_by_group[group_index] for group_index in group_index_list],
                out_acc=output[group_index_list],
                compact_grouped_chunk=compact_grouped_chunk,
                disable_packed_grouped_cuda=disable_packed_grouped_cuda,
                trace=trace,
            )
        return output

    if trace is not None:
        trace.record_page_read(
            sum(page.payload_nbytes for group_pages in pages_by_group for page in group_pages),
            sum(page.metadata_nbytes for group_pages in pages_by_group for page in group_pages),
        )

    if header.mode_default == "M3":
        use_native_dtype = _m3_native_compute_enabled(pages_by_group[0])
        dense = torch.stack(
            [
                _decode_escape_batch_torch(
                    group_pages,
                    token_count=header.token_count,
                    head_dim=header.head_dim,
                    promote_float32=not use_native_dtype,
                )
                for group_pages in pages_by_group
            ],
            dim=0,
        )
        if use_native_dtype:
            output[:, :, : header.head_dim] += torch.einsum(
                "bqpt,bpth->bqh",
                weights.to(dtype=dense.dtype),
                dense,
            ).to(dtype=torch.float32)
        else:
            output[:, :, : header.head_dim] += torch.einsum("bqpt,bpth->bqh", weights, dense)
        return output

    if header.mode_default in {"M2", "M4"}:
        raise ValueError(f"{header.mode_default} is only supported for key scoring in this phase")

    if header.mode_default == "M1":
        for group_index in range(header.num_groups):
            group_words = torch.stack(
                [torch.stack([page.payload[group_index] for page in group_pages], dim=0) for group_pages in pages_by_group],
                dim=0,
            )
            codes = _unpack_bits_torch(
                group_words.reshape(-1, header.words_per_group),
                pages_by_group[0][0].unpack_shifts,
                pages_by_group[0][0].unpack_mask,
                header.group_size,
                trace=trace,
            ).reshape(batch_size, page_count, token_count, header.group_size)
            if trace is not None:
                trace.record_temporary(int(codes.numel() * codes.element_size()))
            group = _lookup_lut_group_torch(
                torch.stack(
                    [
                        torch.stack([page.codebooks[group_index] for page in group_pages], dim=0)
                        for group_pages in pages_by_group
                    ],
                    dim=0,
                ),
                codes,
            )
            start = group_index * header.group_size
            end = start + header.group_size
            output[:, :, start:end] += torch.einsum("bqpt,bptg->bqg", weights, group)
        return output

    if header.mode_default == "T3":
        codebooks = pages_by_group[0][0].codebooks if pages_by_group[0][0].codebooks is not None else _turbo3_centroids_torch(device_type=device_type)
        prepared_chunks = [_get_prepared_chunk_mps(group_pages) for group_pages in pages_by_group]
        for group_index in range(header.num_groups):
            group_words = torch.stack(
                [
                    prepared_chunks[group_id].payload_groups[group_index]
                    if prepared_chunks[group_id] is not None
                    else torch.stack([page.payload[group_index] for page in group_pages], dim=0)
                    for group_id, group_pages in enumerate(pages_by_group)
                ],
                dim=0,
            )
            codes = _unpack_bits_torch(
                group_words.reshape(-1, header.words_per_group),
                pages_by_group[0][0].unpack_shifts,
                pages_by_group[0][0].unpack_mask,
                header.group_size,
                trace=trace,
            ).reshape(batch_size, page_count, token_count, header.group_size)
            if trace is not None:
                trace.record_temporary(int(codes.numel() * codes.element_size()))
            scales = torch.stack(
                [
                    prepared_chunks[group_id].scales_groups[group_index]
                    if prepared_chunks[group_id] is not None and prepared_chunks[group_id].scales_groups is not None
                    else torch.stack([page.scales[:, group_index] for page in group_pages], dim=0)
                    for group_id, group_pages in enumerate(pages_by_group)
                ],
                dim=0,
            )
            rotated_group = _lookup_turbo_group_torch(codebooks, codes) * scales[..., None]
            group = _fwht_last_dim_torch(rotated_group, trace=trace)
            start = group_index * header.group_size
            end = start + header.group_size
            output[:, :, start:end] += torch.einsum("bqpt,bptg->bqg", weights, group)
        return output

    grouped_prepared_chunk = (
        _build_grouped_prepared_chunk_mps(pages_by_group)
        if compact_grouped_chunk
        else _get_grouped_prepared_chunk_mps(pages_by_group)
    )
    prepared_chunks = None if grouped_prepared_chunk is not None else [
        (
            _build_prepared_chunk_mps(group_pages)
            if compact_grouped_chunk
            else _get_prepared_chunk_mps(group_pages) or (
                _build_prepared_chunk_mps(group_pages) if _prepared_chunk_cache_key(group_pages) is not None else None
            )
        )
        for group_pages in pages_by_group
    ]
    if (
        grouped_prepared_chunk is not None
        and grouped_prepared_chunk.payload_groups
        and grouped_prepared_chunk.scales_groups is not None
        and grouped_prepared_chunk.bias_groups is not None
        and grouped_prepared_chunk.payload_groups_tensor is not None
        and grouped_prepared_chunk.scales_groups_tensor is not None
        and grouped_prepared_chunk.bias_groups_tensor is not None
        and _supports_packed_four_group128_cuda(header, device_type=device_type)
        and not disable_packed_grouped_cuda
    ):
        unpack_shifts = pages_by_group[0][0].unpack_shifts
        unpack_mask = pages_by_group[0][0].unpack_mask
        if unpack_shifts is None or unpack_mask is None:
            raise ValueError("packed grouped CUDA path requires unpack metadata")
        if trace is not None:
            trace.record_grouped_kernel_variant(section="mix", variant="packed_cuda")
        for group_start in range(0, header.num_groups, 2):
            group_end = min(header.num_groups, group_start + 2)
            start = group_start * header.group_size
            end = group_end * header.group_size
            output[:, :, start:end] += _mix_m0_contribution_packed32_grouped_torch(
                weights,
                grouped_prepared_chunk.payload_groups_tensor[:, group_start:group_end],
                grouped_prepared_chunk.scales_groups_tensor[:, group_start:group_end],
                grouped_prepared_chunk.bias_groups_tensor[:, group_start:group_end],
                unpack_shifts=unpack_shifts,
                unpack_mask=unpack_mask,
                trace=trace,
            ).reshape(batch_size, query_count, end - start)
        return output
    if (
        grouped_prepared_chunk is not None
        and grouped_prepared_chunk.fused_scaled_codes is not None
        and grouped_prepared_chunk.bias_groups is not None
        and _supports_fused_two_group64(header)
    ):
        if trace is not None:
            trace.record_grouped_kernel_variant(section="mix", variant="fused_two_group64")
        output[:, :, : header.padded_head_dim] += _mix_m0_contribution_two_group64_torch(
            weights,
            grouped_prepared_chunk.fused_scaled_codes,
            grouped_prepared_chunk.bias_groups,
        )
        return output
    if (
        grouped_prepared_chunk is not None
        and grouped_prepared_chunk.fused_scaled_codes is not None
        and grouped_prepared_chunk.bias_groups is not None
    ):
        if trace is not None:
            trace.record_grouped_kernel_variant(section="mix", variant="fused_generic")
        output[:, :, : header.padded_head_dim] += _mix_m0_contribution_fused_torch(
            weights,
            grouped_prepared_chunk.fused_scaled_codes,
            grouped_prepared_chunk.bias_groups,
            group_size=header.group_size,
        )
        return output
    if (
        grouped_prepared_chunk is None
        and prepared_chunks is not None
        and _supports_fused_two_group64(header)
        and all(chunk is not None and chunk.fused_scaled_codes is not None and chunk.bias_groups is not None for chunk in prepared_chunks)
    ):
        if trace is not None:
            trace.record_grouped_kernel_variant(section="mix", variant="fused_two_group64")
        fused_scaled_codes, bias_groups = _assemble_grouped_fused_two_group64_components(
            prepared_chunks,
            trace=trace,
            device_type=device_type,
        )
        output[:, :, : header.padded_head_dim] += _mix_m0_contribution_two_group64_torch(
            weights,
            fused_scaled_codes,
            bias_groups,
        )
        return output
    if (
        grouped_prepared_chunk is None
        and prepared_chunks is not None
        and all(chunk is not None and chunk.fused_scaled_codes is not None and chunk.bias_groups is not None for chunk in prepared_chunks)
    ):
        if trace is not None:
            trace.record_grouped_kernel_variant(section="mix", variant="fused_generic")
        fused_scaled_codes = _trace_timed_call(
            trace,
            "chunk_assembly",
            device_type=device_type,
            fn=lambda: _load_torch().stack([chunk.fused_scaled_codes for chunk in prepared_chunks], dim=0),
        )
        bias_groups = tuple(
            _trace_timed_call(
                trace,
                "chunk_assembly",
                device_type=device_type,
                fn=lambda group_index=group_index: _load_torch().stack([chunk.bias_groups[group_index] for chunk in prepared_chunks], dim=0),
            )
            for group_index in range(header.num_groups)
        )
        if trace is not None:
            trace.record_temporary(int(fused_scaled_codes.numel() * fused_scaled_codes.element_size()))
            trace.record_temporary(sum(int(tensor.numel() * tensor.element_size()) for tensor in bias_groups))
        output[:, :, : header.padded_head_dim] += _mix_m0_contribution_fused_torch(
            weights,
            fused_scaled_codes,
            bias_groups,
            group_size=header.group_size,
        )
        return output
    if trace is not None:
        trace.record_grouped_kernel_variant(section="mix", variant="generic")
    for group_index in range(header.num_groups):
        cached_codes = grouped_prepared_chunk is not None and grouped_prepared_chunk.codes_groups is not None
        if cached_codes:
            codes = grouped_prepared_chunk.codes_groups[group_index]
        else:
            def _build_codes(group_index: int = group_index):
                return torch.stack(
                    [
                        prepared_chunks[group_id].codes_groups[group_index]
                        if prepared_chunks is not None and prepared_chunks[group_id] is not None and prepared_chunks[group_id].codes_groups is not None
                        else _trace_timed_call(
                            trace,
                            "unpack",
                            device_type=device_type,
                            fn=lambda group_pages=group_pages, group_index=group_index: _unpack_bits_torch(
                                torch.stack([page.payload[group_index] for page in group_pages], dim=0).reshape(-1, header.words_per_group),
                                pages_by_group[0][0].unpack_shifts,
                                pages_by_group[0][0].unpack_mask,
                                header.group_size,
                            ).reshape(page_count, token_count, header.group_size),
                        )
                        for group_id, group_pages in enumerate(pages_by_group)
                    ],
                    dim=0,
                )

            codes = _trace_timed_call(
                trace,
                "chunk_assembly",
                device_type=device_type,
                fn=_build_codes,
            )
        if trace is not None and not cached_codes:
            trace.record_temporary(int(codes.numel() * codes.element_size()))
        scales = (
            grouped_prepared_chunk.scales_groups[group_index]
            if grouped_prepared_chunk is not None and grouped_prepared_chunk.scales_groups is not None
            else torch.stack(
                [
                    prepared_chunks[group_id].scales_groups[group_index]
                    if prepared_chunks is not None and prepared_chunks[group_id] is not None and prepared_chunks[group_id].scales_groups is not None
                    else torch.stack([page.scales[:, group_index].to(torch.float32) for page in group_pages], dim=0)
                    for group_id, group_pages in enumerate(pages_by_group)
                ],
                dim=0,
            )
        )
        bias = (
            grouped_prepared_chunk.bias_groups[group_index]
            if grouped_prepared_chunk is not None and grouped_prepared_chunk.bias_groups is not None
            else torch.stack(
                [
                    prepared_chunks[group_id].bias_groups[group_index]
                    if prepared_chunks is not None and prepared_chunks[group_id] is not None and prepared_chunks[group_id].bias_groups is not None
                    else torch.stack([page.bias[:, group_index].to(torch.float32) for page in group_pages], dim=0)
                    for group_id, group_pages in enumerate(pages_by_group)
                ],
                dim=0,
            )
        )
        start = group_index * header.group_size
        end = start + header.group_size
        output[:, :, start:end] += _mix_m0_contribution_torch(weights, codes, scales, bias)

    return output


def decode_multi_query_step_torch_tensor(
    query_slices: np.ndarray | Any,
    key_pages: Sequence[EncodedPage | PreparedPageTorch],
    value_pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    device_type: TorchDevice,
    trace: ExecutionTrace | None = None,
):
    torch = _load_torch()
    prepared_key_pages = prepare_pages_torch(key_pages, device_type=device_type, trace=trace)
    prepared_value_pages = prepare_pages_torch(value_pages, device_type=device_type, trace=trace)
    if not prepared_key_pages:
        raise ValueError(f"decode_multi_query_step_{device_type} requires at least one page")
    if trace is not None:
        trace.record_per_kv_decode_call()

    logits_parts = []
    if torch.is_tensor(query_slices):
        query_count = int(query_slices.shape[0])
    else:
        query_count = int(np.asarray(query_slices).shape[0])
    for page_chunk in _chunk_compatible_pages(prepared_key_pages):
        if trace is not None:
            trace.record_per_kv_score_chunk(
                query_count=query_count,
                page_count=len(page_chunk),
                token_count=page_chunk[0].header.token_count,
            )
        logits_parts.append(
            _trace_timed_call(
                trace,
                "score",
                device_type=device_type,
                fn=lambda page_chunk=page_chunk: _score_page_chunk_multiquery_torch(query_slices, page_chunk, trace=trace),
            )
        )
    logits = torch.cat(logits_parts, dim=1)
    weights = _trace_timed_call(
        trace,
        "softmax",
        device_type=device_type,
        fn=lambda: torch.softmax(logits, dim=1),
    )
    output = torch.zeros(
        (query_count, prepared_value_pages[0].header.padded_head_dim),
        dtype=torch.float32,
        device=device_type,
    )
    offset = 0
    for page_chunk in _chunk_compatible_pages(prepared_value_pages):
        chunk_token_count = page_chunk[0].header.token_count * len(page_chunk)
        chunk_weights = _trace_timed_call(
            trace,
            "chunk_assembly",
            device_type=device_type,
            fn=lambda page_chunk=page_chunk, offset=offset, chunk_token_count=chunk_token_count: weights[
                :,
                offset : offset + chunk_token_count,
            ].reshape(
                weights.shape[0],
                len(page_chunk),
                page_chunk[0].header.token_count,
            ),
        )
        if trace is not None:
            trace.record_per_kv_mix_chunk(
                query_count=query_count,
                page_count=len(page_chunk),
                token_count=page_chunk[0].header.token_count,
                head_dim=prepared_value_pages[0].header.padded_head_dim,
            )
        output += _trace_timed_call(
            trace,
            "mix",
            device_type=device_type,
            fn=lambda chunk_weights=chunk_weights, page_chunk=page_chunk: _mix_page_chunk_multiquery_torch(
                chunk_weights,
                page_chunk,
                trace=trace,
            ),
        )
        offset += chunk_token_count

    head_dim = prepared_value_pages[0].header.head_dim
    return logits, weights, output[:, :head_dim]


def decode_grouped_multiquery_step_torch_tensor(
    query_groups,
    key_pages_by_group: Sequence[Sequence[EncodedPage | PreparedPageTorch]],
    value_pages_by_group: Sequence[Sequence[EncodedPage | PreparedPageTorch]],
    *,
    device_type: TorchDevice,
    trace: ExecutionTrace | None = None,
):
    if not key_pages_by_group or not value_pages_by_group:
        raise ValueError("grouped decode requires non-empty key/value page groups")
    if len(key_pages_by_group) != len(value_pages_by_group):
        raise ValueError("key_pages_by_group and value_pages_by_group must have the same group count")
    group_count = len(key_pages_by_group)
    if len(query_groups) != group_count:
        raise ValueError("query_groups must align with key/value group count")

    prepared_key_groups = [prepare_pages_torch(group_pages, device_type=device_type, trace=trace) for group_pages in key_pages_by_group]
    prepared_value_groups = [prepare_pages_torch(group_pages, device_type=device_type, trace=trace) for group_pages in value_pages_by_group]
    if not prepared_key_groups[0]:
        raise ValueError("grouped decode requires at least one key page per group")

    return decode_grouped_multiquery_step_prepared_torch_tensor(
        query_groups,
        prepared_key_groups,
        prepared_value_groups,
        trace=trace,
    )


def decode_grouped_multiquery_step_prepared_torch_tensor(
    query_groups,
    key_pages_by_group: Sequence[Sequence[PreparedPageTorch]],
    value_pages_by_group: Sequence[Sequence[PreparedPageTorch]],
    *,
    key_chunk_lengths: Sequence[int] | None = None,
    value_chunk_lengths: Sequence[int] | None = None,
    compact_grouped_chunk: bool = False,
    compact_grouped_mix_chunk: bool = False,
    disable_packed_grouped_cuda_mix: bool = False,
    trace: ExecutionTrace | None = None,
):
    torch = _load_torch()
    if not key_pages_by_group or not value_pages_by_group:
        raise ValueError("grouped decode requires non-empty key/value page groups")
    if len(key_pages_by_group) != len(value_pages_by_group):
        raise ValueError("key_pages_by_group and value_pages_by_group must have the same group count")
    group_count = len(key_pages_by_group)
    if len(query_groups) != group_count:
        raise ValueError("query_groups must align with key/value group count")
    if not key_pages_by_group[0]:
        raise ValueError("grouped decode requires at least one key page per group")
    device_type = key_pages_by_group[0][0].device_type

    query_tensors = [
        group.to(dtype=torch.float32, device=device_type)
        if torch.is_tensor(group)
        else torch.as_tensor(group, dtype=torch.float32, device=device_type)
        for group in query_groups
    ]
    query_count = int(query_tensors[0].shape[0])
    for group_query in query_tensors[1:]:
        if int(group_query.shape[0]) != query_count:
            raise ValueError("all query groups must have the same query count for batched grouped decode")
    stacked_queries = torch.stack(query_tensors, dim=0)
    if trace is not None:
        trace.record_grouped_decode_call(output_only=False)

    first_key_group = key_pages_by_group[0]
    first_value_group = value_pages_by_group[0]
    first_header = first_key_group[0].header
    prepared_query_groups_tensor = _trace_timed_call(
        trace,
        "chunk_assembly",
        device_type=device_type,
        fn=lambda: _pad_queries(
            stacked_queries.reshape(group_count * query_count, first_header.head_dim),
            first_header.padded_head_dim,
            device_type=device_type,
        ).reshape(group_count, query_count, first_header.num_groups, first_header.group_size),
    )
    query_group_sums = prepared_query_groups_tensor.sum(dim=-1)
    if key_chunk_lengths is None:
        key_chunk_lengths = _merged_chunk_lengths_for_page_groups(key_pages_by_group)
    else:
        key_chunk_lengths = tuple(int(length) for length in key_chunk_lengths)
    if value_chunk_lengths is None:
        value_chunk_lengths = _merged_chunk_lengths_for_page_groups(value_pages_by_group)
    else:
        value_chunk_lengths = tuple(int(length) for length in value_chunk_lengths)
    if sum(key_chunk_lengths) != len(first_key_group) or sum(value_chunk_lengths) != len(first_value_group):
        raise ValueError("grouped decode chunk lengths must cover all key/value pages exactly")
    # Score and mix can use different chunk schedules as long as both cover the
    # same flattened token stream in order.

    logits_parts = []
    key_offset = 0
    for chunk_length in key_chunk_lengths:
        chunk_pages = _trace_timed_call(
            trace,
            "chunk_assembly",
            device_type=device_type,
            fn=lambda key_offset=key_offset, chunk_length=chunk_length: [
                group_pages[key_offset : key_offset + chunk_length] for group_pages in key_pages_by_group
            ],
        )
        if trace is not None:
            trace.record_grouped_score_chunk(
                batch_size=group_count,
                query_count=query_count,
                page_count=chunk_length,
                token_count=first_key_group[0].header.token_count,
            )
        logits_parts.append(
            _trace_timed_call(
                trace,
                "score",
                device_type=device_type,
                fn=lambda chunk_pages=chunk_pages: _score_page_chunk_grouped_multiquery_torch(
                    stacked_queries,
                    chunk_pages,
                    prepared_query_groups_tensor=prepared_query_groups_tensor,
                    query_group_sums=query_group_sums,
                    compact_grouped_chunk=compact_grouped_chunk,
                    trace=trace,
                ),
            )
        )
        key_offset += chunk_length
    logits = torch.cat(logits_parts, dim=2)
    weights = _trace_timed_call(
        trace,
        "softmax",
        device_type=device_type,
        fn=lambda: torch.softmax(logits, dim=2),
    )

    head_dim = first_value_group[0].header.head_dim
    padded_head_dim = first_value_group[0].header.padded_head_dim
    output = torch.zeros((group_count, query_count, padded_head_dim), dtype=torch.float32, device=device_type)
    offset = 0
    value_offset = 0
    for chunk_index, chunk_length in enumerate(value_chunk_lengths):
        chunk_token_count = first_value_group[value_offset].header.token_count * chunk_length
        chunk_weights = _trace_timed_call(
            trace,
            "chunk_assembly",
            device_type=device_type,
            fn=lambda offset=offset, chunk_token_count=chunk_token_count, chunk_length=chunk_length, token_count=first_value_group[value_offset].header.token_count: weights[
                :,
                :,
                offset : offset + chunk_token_count,
            ].reshape(
                group_count,
                query_count,
                chunk_length,
                token_count,
            ),
        )
        chunk_pages = _trace_timed_call(
            trace,
            "chunk_assembly",
            device_type=device_type,
            fn=lambda value_offset=value_offset, chunk_length=chunk_length: [
                group_pages[value_offset : value_offset + chunk_length] for group_pages in value_pages_by_group
            ],
        )
        if trace is not None:
            trace.record_grouped_mix_chunk(
                batch_size=group_count,
                query_count=query_count,
                page_count=chunk_length,
                token_count=first_value_group[value_offset].header.token_count,
                head_dim=padded_head_dim,
            )
        output = _trace_timed_call(
            trace,
            "mix",
            device_type=device_type,
            fn=lambda chunk_weights=chunk_weights, chunk_pages=chunk_pages, output=output: _mix_page_chunk_grouped_multiquery_torch(
                chunk_weights,
                chunk_pages,
                out_acc=output,
                compact_grouped_chunk=(compact_grouped_chunk or compact_grouped_mix_chunk),
                disable_packed_grouped_cuda=disable_packed_grouped_cuda_mix,
                trace=trace,
            ),
        )
        offset += chunk_token_count
        value_offset += chunk_length

    return logits, weights, output[:, :, :head_dim]


def decode_grouped_multiquery_step_prepared_torch_tensor_output_only(
    query_groups,
    key_pages_by_group: Sequence[Sequence[PreparedPageTorch]],
    value_pages_by_group: Sequence[Sequence[PreparedPageTorch]],
    *,
    compact_grouped_chunk: bool = False,
    compact_grouped_mix_chunk: bool = False,
    disable_packed_grouped_cuda_mix: bool = False,
    trace: ExecutionTrace | None = None,
):
    torch = _load_torch()
    if not key_pages_by_group or not value_pages_by_group:
        raise ValueError("grouped decode requires non-empty key/value page groups")
    if len(key_pages_by_group) != len(value_pages_by_group):
        raise ValueError("key_pages_by_group and value_pages_by_group must have the same group count")
    group_count = len(key_pages_by_group)
    if len(query_groups) != group_count:
        raise ValueError("query_groups must align with key/value group count")
    if not key_pages_by_group[0]:
        raise ValueError("grouped decode requires at least one key page per group")
    device_type = key_pages_by_group[0][0].device_type

    query_tensors = [
        group.to(dtype=torch.float32, device=device_type)
        if torch.is_tensor(group)
        else torch.as_tensor(group, dtype=torch.float32, device=device_type)
        for group in query_groups
    ]
    query_count = int(query_tensors[0].shape[0])
    for group_query in query_tensors[1:]:
        if int(group_query.shape[0]) != query_count:
            raise ValueError("all query groups must have the same query count for batched grouped decode")
    stacked_queries = torch.stack(query_tensors, dim=0)
    if trace is not None:
        trace.record_grouped_decode_call(output_only=True)

    first_key_group = key_pages_by_group[0]
    first_value_group = value_pages_by_group[0]
    first_header = first_key_group[0].header
    prepared_query_groups_tensor = _trace_timed_call(
        trace,
        "chunk_assembly",
        device_type=device_type,
        fn=lambda: _pad_queries(
            stacked_queries.reshape(group_count * query_count, first_header.head_dim),
            first_header.padded_head_dim,
            device_type=device_type,
        ).reshape(group_count, query_count, first_header.num_groups, first_header.group_size),
    )
    query_group_sums = prepared_query_groups_tensor.sum(dim=-1)
    shared_chunk_lengths = _aligned_chunk_lengths_for_page_pairs(first_key_group, first_value_group)

    head_dim = first_value_group[0].header.head_dim
    padded_head_dim = first_value_group[0].header.padded_head_dim
    output = torch.zeros((group_count, query_count, padded_head_dim), dtype=torch.float32, device=device_type)
    running_max = torch.full((group_count, query_count), float("-inf"), dtype=torch.float32, device=device_type)
    running_denom = torch.zeros((group_count, query_count), dtype=torch.float32, device=device_type)

    key_offset = 0
    value_offset = 0
    for chunk_index, chunk_length in enumerate(shared_chunk_lengths):
        key_chunk_pages = _trace_timed_call(
            trace,
            "chunk_assembly",
            device_type=device_type,
            fn=lambda key_offset=key_offset, chunk_length=chunk_length: [
                group_pages[key_offset : key_offset + chunk_length] for group_pages in key_pages_by_group
            ],
        )
        logits_chunk = _trace_timed_call(
            trace,
            "score",
            device_type=device_type,
            fn=lambda key_chunk_pages=key_chunk_pages: _score_page_chunk_grouped_multiquery_torch(
                stacked_queries,
                key_chunk_pages,
                prepared_query_groups_tensor=prepared_query_groups_tensor,
                query_group_sums=query_group_sums,
                compact_grouped_chunk=compact_grouped_chunk,
                trace=trace,
            ),
        )
        if trace is not None:
            trace.record_grouped_score_chunk(
                batch_size=group_count,
                query_count=query_count,
                page_count=chunk_length,
                token_count=first_key_group[0].header.token_count,
            )
        value_chunk_pages = _trace_timed_call(
            trace,
            "chunk_assembly",
            device_type=device_type,
            fn=lambda value_offset=value_offset, chunk_length=chunk_length: [
                group_pages[value_offset : value_offset + chunk_length] for group_pages in value_pages_by_group
            ],
        )
        chunk_template = value_chunk_pages[0]
        chunk_token_count = chunk_template[0].header.token_count * chunk_length

        prev_max = running_max
        chunk_max = torch.amax(logits_chunk, dim=2)

        def _normalize_chunk():
            new_max = torch.maximum(prev_max, chunk_max)
            prev_scale = torch.where(torch.isfinite(prev_max), torch.exp(prev_max - new_max), torch.zeros_like(new_max))
            chunk_exp = torch.exp(logits_chunk - new_max[:, :, None])
            return new_max, prev_scale, chunk_exp

        new_max, prev_scale, chunk_exp = _trace_timed_call(
            trace,
            "softmax",
            device_type=device_type,
            fn=_normalize_chunk,
        )
        output *= prev_scale[:, :, None]
        running_denom = (running_denom * prev_scale) + torch.sum(chunk_exp, dim=2)
        chunk_weights = _trace_timed_call(
            trace,
            "chunk_assembly",
            device_type=device_type,
            fn=lambda chunk_exp=chunk_exp, chunk_length=chunk_length, chunk_token_count=chunk_template[0].header.token_count: chunk_exp.reshape(
                group_count,
                query_count,
                chunk_length,
                chunk_token_count,
            ),
        )
        output = _trace_timed_call(
            trace,
            "mix",
            device_type=device_type,
            fn=lambda chunk_weights=chunk_weights, value_chunk_pages=value_chunk_pages, output=output: _mix_page_chunk_grouped_multiquery_torch(
                chunk_weights,
                value_chunk_pages,
                out_acc=output,
                compact_grouped_chunk=(compact_grouped_chunk or compact_grouped_mix_chunk),
                disable_packed_grouped_cuda=disable_packed_grouped_cuda_mix,
                trace=trace,
            ),
        )
        if trace is not None:
            trace.record_grouped_mix_chunk(
                batch_size=group_count,
                query_count=query_count,
                page_count=chunk_length,
                token_count=chunk_template[0].header.token_count,
                head_dim=padded_head_dim,
            )
        running_max = new_max
        key_offset += chunk_length
        value_offset += chunk_length

    output = _trace_timed_call(
        trace,
        "softmax",
        device_type=device_type,
        fn=lambda: output / torch.clamp(running_denom[:, :, None], min=1e-12),
    )
    return output[:, :, :head_dim]


def decode_multi_query_step_torch(
    query_slices: np.ndarray,
    key_pages: Sequence[EncodedPage | PreparedPageTorch],
    value_pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    device_type: TorchDevice,
    trace: ExecutionTrace | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits, weights, output = decode_multi_query_step_torch_tensor(
        query_slices,
        key_pages,
        value_pages,
        device_type=device_type,
        trace=trace,
    )
    return (
        logits.detach().cpu().numpy(),
        weights.detach().cpu().numpy(),
        output.detach().cpu().numpy(),
    )


def prepare_pages_mps(
    pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    trace: ExecutionTrace | None = None,
) -> list[PreparedPageTorch]:
    return prepare_pages_torch(pages, device_type="mps", trace=trace)


def prepare_page_mps(page: EncodedPage | PreparedPageTorch, *, trace: ExecutionTrace | None = None) -> PreparedPageTorch:
    return prepare_page_torch(page, device_type="mps", trace=trace)


def score_page_mps(
    query_slice: np.ndarray | Any,
    page: EncodedPage | PreparedPageTorch,
    *,
    trace: ExecutionTrace | None = None,
) -> np.ndarray:
    return score_page_torch(query_slice, page, device_type="mps", trace=trace)


def score_pages_mps(
    query_slice: np.ndarray | Any,
    pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    trace: ExecutionTrace | None = None,
) -> list[np.ndarray]:
    return score_pages_torch(query_slice, pages, device_type="mps", trace=trace)


def mix_page_mps(
    attn_weights: np.ndarray | Any,
    page: EncodedPage | PreparedPageTorch,
    *,
    out_acc: np.ndarray | None = None,
    trace: ExecutionTrace | None = None,
) -> np.ndarray:
    return mix_page_torch(attn_weights, page, device_type="mps", out_acc=out_acc, trace=trace)


def decode_step_mps(
    query_slice: np.ndarray | Any,
    key_pages: Sequence[EncodedPage | PreparedPageTorch],
    value_pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    precomputed_page_logits: Sequence[np.ndarray | Any | None] | None = None,
    trace: ExecutionTrace | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return decode_step_torch(
        query_slice,
        key_pages,
        value_pages,
        device_type="mps",
        precomputed_page_logits=precomputed_page_logits,
        trace=trace,
    )


def decode_multi_query_step_mps_tensor(
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
        device_type="mps",
        trace=trace,
    )


def decode_grouped_multiquery_step_mps_tensor(
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
        device_type="mps",
        trace=trace,
    )


def decode_grouped_multiquery_step_prepared_mps_tensor(
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


def decode_multi_query_step_mps(
    query_slices: np.ndarray,
    key_pages: Sequence[EncodedPage | PreparedPageTorch],
    value_pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    trace: ExecutionTrace | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return decode_multi_query_step_torch(query_slices, key_pages, value_pages, device_type="mps", trace=trace)
