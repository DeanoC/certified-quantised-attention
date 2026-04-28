from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Literal

from .backends import (
    PreparedPageTorch,
    cuda_available,
    mps_available,
    prepare_page_cuda,
    prepare_page_mps,
    prepare_pages_cuda,
    prepare_pages_mps,
)
from .tracing import ExecutionTrace
from .types import EncodedPage

CachePolicy = Literal["fifo", "lru", "pinned_recent_fifo"]
CacheBackend = Literal["auto", "torch_mps", "torch_cuda", "cpu_ref"]


@dataclass(slots=True)
class PreparedPageCache:
    max_resident_bytes: int | None = None
    policy: CachePolicy = "fifo"
    pinned_recent_pages: int = 0
    _prepared_pages: dict[tuple[str, int], PreparedPageTorch] = field(default_factory=dict)
    _prepared_page_ids: set[int] = field(default_factory=set)
    _resident_bytes: int = 0
    _order: OrderedDict[tuple[str, int], None] = field(default_factory=OrderedDict)

    @property
    def resident_bytes(self) -> int:
        return self._resident_bytes

    @property
    def size(self) -> int:
        return len(self._prepared_pages)

    def clear(self) -> None:
        self._prepared_pages.clear()
        self._prepared_page_ids.clear()
        self._resident_bytes = 0
        self._order.clear()

    def owns_prepared_page(self, page: PreparedPageTorch) -> bool:
        return id(page) in self._prepared_page_ids

    def _page_nbytes(self, page: PreparedPageTorch) -> int:
        resident_nbytes = int(page.resident_nbytes)
        if resident_nbytes > 0:
            return resident_nbytes
        return int(page.host_to_device_nbytes)

    def _pinned_keys(self) -> set[int]:
        if self.policy != "pinned_recent_fifo" or self.pinned_recent_pages <= 0:
            return set()
        keys = list(self._order.keys())
        if not keys:
            return set()
        return set(keys[-self.pinned_recent_pages :])

    def _touch_cached_page(self, cache_key: tuple[str, int]) -> None:
        if self.policy == "lru":
            self._order.move_to_end(cache_key)

    def _evict_one(self, *, trace: ExecutionTrace | None = None) -> bool:
        pinned_keys = self._pinned_keys()
        fallback_key: tuple[str, int] | None = None
        while self._order:
            cache_key = next(iter(self._order))
            self._order.pop(cache_key, None)
            if cache_key in pinned_keys:
                if fallback_key is None:
                    fallback_key = cache_key
                self._order[cache_key] = None
                if len(pinned_keys) >= len(self._order):
                    break
                continue
            cached_page = self._prepared_pages.pop(cache_key, None)
            if cached_page is None:
                continue
            self._prepared_page_ids.discard(id(cached_page))
            evicted_bytes = self._page_nbytes(cached_page)
            self._resident_bytes = max(0, self._resident_bytes - evicted_bytes)
            if trace is not None:
                trace.record_cache_eviction(evicted_bytes)
                trace.observe_cache_resident_bytes(self._resident_bytes)
            return True
        if fallback_key is not None:
            self._order.pop(fallback_key, None)
            cached_page = self._prepared_pages.pop(fallback_key, None)
            if cached_page is None:
                return False
            self._prepared_page_ids.discard(id(cached_page))
            evicted_bytes = self._page_nbytes(cached_page)
            self._resident_bytes = max(0, self._resident_bytes - evicted_bytes)
            if trace is not None:
                trace.record_cache_eviction(evicted_bytes)
                trace.observe_cache_resident_bytes(self._resident_bytes)
            return True
        return False

    def _ensure_capacity(self, incoming_nbytes: int, *, trace: ExecutionTrace | None = None) -> None:
        if self.max_resident_bytes is None:
            return
        while self._resident_bytes + incoming_nbytes > self.max_resident_bytes and self._prepared_pages:
            if not self._evict_one(trace=trace):
                break

    def _resolve_backend(self, backend: CacheBackend) -> CacheBackend:
        if backend != "auto":
            return backend
        if cuda_available():
            return "torch_cuda"
        if mps_available():
            return "torch_mps"
        return "cpu_ref"

    def append_page(
        self,
        page: EncodedPage | PreparedPageTorch,
        *,
        backend: CacheBackend = "auto",
        trace: ExecutionTrace | None = None,
    ) -> EncodedPage | PreparedPageTorch:
        return self.prepare_page(page, backend=backend, trace=trace)

    def append_pages(
        self,
        pages: list[EncodedPage | PreparedPageTorch],
        *,
        backend: CacheBackend = "auto",
        trace: ExecutionTrace | None = None,
    ) -> list[EncodedPage | PreparedPageTorch]:
        return self.prepare_pages(pages, backend=backend, trace=trace)

    def prepare_page(
        self,
        page: EncodedPage | PreparedPageTorch,
        *,
        backend: CacheBackend = "auto",
        trace: ExecutionTrace | None = None,
    ) -> EncodedPage | PreparedPageTorch:
        resolved_backend = self._resolve_backend(backend)
        if resolved_backend == "cpu_ref":
            return page.source_page if isinstance(page, PreparedPageTorch) else page
        if isinstance(page, PreparedPageTorch):
            if trace is not None:
                trace.record_cache_hit()
                trace.observe_cache_resident_bytes(self._resident_bytes)
            return page

        cache_key = ("cuda" if resolved_backend == "torch_cuda" else "mps", id(page))
        cached_page = self._prepared_pages.get(cache_key)
        if cached_page is not None:
            self._touch_cached_page(cache_key)
            if trace is not None:
                trace.record_cache_hit()
                trace.observe_cache_resident_bytes(self._resident_bytes)
            return cached_page

        prepared_page = prepare_page_cuda(page, trace=trace) if resolved_backend == "torch_cuda" else prepare_page_mps(page, trace=trace)
        self._ensure_capacity(self._page_nbytes(prepared_page), trace=trace)
        self._prepared_pages[cache_key] = prepared_page
        self._prepared_page_ids.add(id(prepared_page))
        self._order[cache_key] = None
        self._resident_bytes += self._page_nbytes(prepared_page)
        if trace is not None:
            trace.record_cache_miss()
            trace.observe_cache_resident_bytes(self._resident_bytes)
        return prepared_page

    def prepare_pages(
        self,
        pages: list[EncodedPage | PreparedPageTorch],
        *,
        backend: CacheBackend = "auto",
        trace: ExecutionTrace | None = None,
    ) -> list[EncodedPage | PreparedPageTorch]:
        resolved_backend = self._resolve_backend(backend)
        if resolved_backend == "cpu_ref":
            return [page.source_page if isinstance(page, PreparedPageTorch) else page for page in pages]

        prepared_pages: list[EncodedPage | PreparedPageTorch | None] = [None] * len(pages)
        miss_indices: list[int] = []
        miss_pages: list[EncodedPage] = []

        for index, page in enumerate(pages):
            if isinstance(page, PreparedPageTorch):
                prepared_pages[index] = page
                if trace is not None:
                    trace.record_cache_hit()
                    trace.observe_cache_resident_bytes(self._resident_bytes)
                continue

            cache_key = ("cuda" if resolved_backend == "torch_cuda" else "mps", id(page))
            cached_page = self._prepared_pages.get(cache_key)
            if cached_page is not None:
                self._touch_cached_page(cache_key)
                prepared_pages[index] = cached_page
                if trace is not None:
                    trace.record_cache_hit()
                    trace.observe_cache_resident_bytes(self._resident_bytes)
                continue

            miss_indices.append(index)
            miss_pages.append(page)

        if miss_pages:
            new_prepared_pages = prepare_pages_cuda(miss_pages, trace=trace) if resolved_backend == "torch_cuda" else prepare_pages_mps(miss_pages, trace=trace)
            for index, source_page, prepared_page in zip(miss_indices, miss_pages, new_prepared_pages, strict=True):
                self._ensure_capacity(self._page_nbytes(prepared_page), trace=trace)
                cache_key = (prepared_page.device_type, id(source_page))
                self._prepared_pages[cache_key] = prepared_page
                self._prepared_page_ids.add(id(prepared_page))
                self._order[cache_key] = None
                self._resident_bytes += self._page_nbytes(prepared_page)
                prepared_pages[index] = prepared_page
                if trace is not None:
                    trace.record_cache_miss()
                    trace.observe_cache_resident_bytes(self._resident_bytes)

        if any(page is None for page in prepared_pages):
            raise RuntimeError("prepared page cache failed to populate all requested pages")
        return [page for page in prepared_pages if page is not None]
