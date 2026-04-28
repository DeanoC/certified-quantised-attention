from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ExecutionTrace:
    capture_timings: bool = False
    m0_full_page_materializations: int = 0
    payload_bytes_read: int = 0
    metadata_bytes_read: int = 0
    host_to_device_bytes: int = 0
    max_temporary_bytes: int = 0
    prepared_page_cache_hits: int = 0
    prepared_page_cache_misses: int = 0
    cache_resident_bytes: int = 0
    prepared_page_cache_evictions: int = 0
    cache_evicted_bytes: int = 0
    prepare_ms_total: float = 0.0
    prepare_calls: int = 0
    score_ms_total: float = 0.0
    score_calls: int = 0
    mix_ms_total: float = 0.0
    mix_calls: int = 0
    softmax_ms_total: float = 0.0
    softmax_calls: int = 0
    unpack_ms_total: float = 0.0
    unpack_calls: int = 0
    fwht_ms_total: float = 0.0
    fwht_calls: int = 0
    chunk_assembly_ms_total: float = 0.0
    chunk_assembly_calls: int = 0
    grouped_decode_calls: int = 0
    grouped_decode_output_only_calls: int = 0
    grouped_score_chunk_count: int = 0
    grouped_mix_chunk_count: int = 0
    grouped_score_chunk_pages_total: int = 0
    grouped_mix_chunk_pages_total: int = 0
    grouped_score_chunk_pages_max: int = 0
    grouped_mix_chunk_pages_max: int = 0
    grouped_logits_elements_total: int = 0
    grouped_weights_elements_total: int = 0
    grouped_output_elements_total: int = 0
    grouped_score_packed_cuda_calls: int = 0
    grouped_score_fused_two_group64_calls: int = 0
    grouped_score_fused_generic_calls: int = 0
    grouped_score_generic_calls: int = 0
    grouped_mix_packed_cuda_calls: int = 0
    grouped_mix_fused_two_group64_calls: int = 0
    grouped_mix_fused_generic_calls: int = 0
    grouped_mix_generic_calls: int = 0
    per_kv_decode_calls: int = 0
    per_kv_score_chunk_count: int = 0
    per_kv_mix_chunk_count: int = 0
    per_kv_score_chunk_pages_total: int = 0
    per_kv_mix_chunk_pages_total: int = 0
    per_kv_score_chunk_pages_max: int = 0
    per_kv_mix_chunk_pages_max: int = 0
    per_kv_logits_elements_total: int = 0
    per_kv_weights_elements_total: int = 0
    per_kv_output_elements_total: int = 0
    per_kv_score_fused_two_group64_calls: int = 0
    per_kv_score_fused_generic_calls: int = 0
    per_kv_score_generic_calls: int = 0
    per_kv_mix_fused_two_group64_calls: int = 0
    per_kv_mix_fused_generic_calls: int = 0
    per_kv_mix_generic_calls: int = 0

    def record_page_read(self, payload_bytes: int, metadata_bytes: int) -> None:
        self.payload_bytes_read += int(payload_bytes)
        self.metadata_bytes_read += int(metadata_bytes)

    def record_host_to_device(self, nbytes: int) -> None:
        self.host_to_device_bytes += int(nbytes)

    def record_temporary(self, nbytes: int) -> None:
        self.max_temporary_bytes = max(self.max_temporary_bytes, int(nbytes))

    def record_m0_full_page_materialization(self, count: int = 1) -> None:
        self.m0_full_page_materializations += int(count)

    def record_cache_hit(self, count: int = 1) -> None:
        self.prepared_page_cache_hits += int(count)

    def record_cache_miss(self, count: int = 1) -> None:
        self.prepared_page_cache_misses += int(count)

    def observe_cache_resident_bytes(self, nbytes: int) -> None:
        self.cache_resident_bytes = max(self.cache_resident_bytes, int(nbytes))

    def record_cache_eviction(self, nbytes: int, count: int = 1) -> None:
        self.prepared_page_cache_evictions += int(count)
        self.cache_evicted_bytes += int(nbytes)

    def record_grouped_decode_call(self, *, output_only: bool) -> None:
        if output_only:
            self.grouped_decode_output_only_calls += 1
            return
        self.grouped_decode_calls += 1

    def record_per_kv_decode_call(self) -> None:
        self.per_kv_decode_calls += 1

    def record_grouped_score_chunk(
        self,
        *,
        batch_size: int,
        query_count: int,
        page_count: int,
        token_count: int,
    ) -> None:
        self.grouped_score_chunk_count += 1
        self.grouped_score_chunk_pages_total += int(page_count)
        self.grouped_score_chunk_pages_max = max(self.grouped_score_chunk_pages_max, int(page_count))
        self.grouped_logits_elements_total += int(batch_size) * int(query_count) * int(page_count) * int(token_count)

    def record_grouped_mix_chunk(
        self,
        *,
        batch_size: int,
        query_count: int,
        page_count: int,
        token_count: int,
        head_dim: int,
    ) -> None:
        self.grouped_mix_chunk_count += 1
        self.grouped_mix_chunk_pages_total += int(page_count)
        self.grouped_mix_chunk_pages_max = max(self.grouped_mix_chunk_pages_max, int(page_count))
        self.grouped_weights_elements_total += int(batch_size) * int(query_count) * int(page_count) * int(token_count)
        self.grouped_output_elements_total += int(batch_size) * int(query_count) * int(head_dim)

    def record_per_kv_score_chunk(
        self,
        *,
        query_count: int,
        page_count: int,
        token_count: int,
    ) -> None:
        self.per_kv_score_chunk_count += 1
        self.per_kv_score_chunk_pages_total += int(page_count)
        self.per_kv_score_chunk_pages_max = max(self.per_kv_score_chunk_pages_max, int(page_count))
        self.per_kv_logits_elements_total += int(query_count) * int(page_count) * int(token_count)

    def record_per_kv_mix_chunk(
        self,
        *,
        query_count: int,
        page_count: int,
        token_count: int,
        head_dim: int,
    ) -> None:
        self.per_kv_mix_chunk_count += 1
        self.per_kv_mix_chunk_pages_total += int(page_count)
        self.per_kv_mix_chunk_pages_max = max(self.per_kv_mix_chunk_pages_max, int(page_count))
        self.per_kv_weights_elements_total += int(query_count) * int(page_count) * int(token_count)
        self.per_kv_output_elements_total += int(query_count) * int(head_dim)

    def record_grouped_kernel_variant(self, *, section: str, variant: str) -> None:
        if section == "score":
            if variant == "packed_cuda":
                self.grouped_score_packed_cuda_calls += 1
                return
            if variant == "fused_two_group64":
                self.grouped_score_fused_two_group64_calls += 1
                return
            if variant == "fused_generic":
                self.grouped_score_fused_generic_calls += 1
                return
            if variant == "generic":
                self.grouped_score_generic_calls += 1
                return
        if section == "mix":
            if variant == "packed_cuda":
                self.grouped_mix_packed_cuda_calls += 1
                return
            if variant == "fused_two_group64":
                self.grouped_mix_fused_two_group64_calls += 1
                return
            if variant == "fused_generic":
                self.grouped_mix_fused_generic_calls += 1
                return
            if variant == "generic":
                self.grouped_mix_generic_calls += 1
                return
        raise ValueError(f"unknown grouped kernel variant: {section}/{variant}")

    def record_per_kv_kernel_variant(self, *, section: str, variant: str) -> None:
        if section == "score":
            if variant == "fused_two_group64":
                self.per_kv_score_fused_two_group64_calls += 1
                return
            if variant == "fused_generic":
                self.per_kv_score_fused_generic_calls += 1
                return
            if variant == "generic":
                self.per_kv_score_generic_calls += 1
                return
        if section == "mix":
            if variant == "fused_two_group64":
                self.per_kv_mix_fused_two_group64_calls += 1
                return
            if variant == "fused_generic":
                self.per_kv_mix_fused_generic_calls += 1
                return
            if variant == "generic":
                self.per_kv_mix_generic_calls += 1
                return
        raise ValueError(f"unknown per_kv kernel variant: {section}/{variant}")

    def record_timing(self, section: str, ms: float, count: int = 1) -> None:
        if section == "prepare":
            self.prepare_ms_total += float(ms)
            self.prepare_calls += int(count)
            return
        if section == "score":
            self.score_ms_total += float(ms)
            self.score_calls += int(count)
            return
        if section == "mix":
            self.mix_ms_total += float(ms)
            self.mix_calls += int(count)
            return
        if section == "softmax":
            self.softmax_ms_total += float(ms)
            self.softmax_calls += int(count)
            return
        if section == "unpack":
            self.unpack_ms_total += float(ms)
            self.unpack_calls += int(count)
            return
        if section == "fwht":
            self.fwht_ms_total += float(ms)
            self.fwht_calls += int(count)
            return
        if section == "chunk_assembly":
            self.chunk_assembly_ms_total += float(ms)
            self.chunk_assembly_calls += int(count)
            return
        raise ValueError(f"unknown timing section: {section}")

    def merge(self, other: "ExecutionTrace") -> None:
        self.m0_full_page_materializations += other.m0_full_page_materializations
        self.payload_bytes_read += other.payload_bytes_read
        self.metadata_bytes_read += other.metadata_bytes_read
        self.host_to_device_bytes += other.host_to_device_bytes
        self.max_temporary_bytes = max(self.max_temporary_bytes, other.max_temporary_bytes)
        self.prepared_page_cache_hits += other.prepared_page_cache_hits
        self.prepared_page_cache_misses += other.prepared_page_cache_misses
        self.cache_resident_bytes = max(self.cache_resident_bytes, other.cache_resident_bytes)
        self.prepared_page_cache_evictions += other.prepared_page_cache_evictions
        self.cache_evicted_bytes += other.cache_evicted_bytes
        self.prepare_ms_total += other.prepare_ms_total
        self.prepare_calls += other.prepare_calls
        self.score_ms_total += other.score_ms_total
        self.score_calls += other.score_calls
        self.mix_ms_total += other.mix_ms_total
        self.mix_calls += other.mix_calls
        self.softmax_ms_total += other.softmax_ms_total
        self.softmax_calls += other.softmax_calls
        self.unpack_ms_total += other.unpack_ms_total
        self.unpack_calls += other.unpack_calls
        self.fwht_ms_total += other.fwht_ms_total
        self.fwht_calls += other.fwht_calls
        self.chunk_assembly_ms_total += other.chunk_assembly_ms_total
        self.chunk_assembly_calls += other.chunk_assembly_calls
        self.grouped_decode_calls += other.grouped_decode_calls
        self.grouped_decode_output_only_calls += other.grouped_decode_output_only_calls
        self.grouped_score_chunk_count += other.grouped_score_chunk_count
        self.grouped_mix_chunk_count += other.grouped_mix_chunk_count
        self.grouped_score_chunk_pages_total += other.grouped_score_chunk_pages_total
        self.grouped_mix_chunk_pages_total += other.grouped_mix_chunk_pages_total
        self.grouped_score_chunk_pages_max = max(self.grouped_score_chunk_pages_max, other.grouped_score_chunk_pages_max)
        self.grouped_mix_chunk_pages_max = max(self.grouped_mix_chunk_pages_max, other.grouped_mix_chunk_pages_max)
        self.grouped_logits_elements_total += other.grouped_logits_elements_total
        self.grouped_weights_elements_total += other.grouped_weights_elements_total
        self.grouped_output_elements_total += other.grouped_output_elements_total
        self.grouped_score_packed_cuda_calls += other.grouped_score_packed_cuda_calls
        self.grouped_score_fused_two_group64_calls += other.grouped_score_fused_two_group64_calls
        self.grouped_score_fused_generic_calls += other.grouped_score_fused_generic_calls
        self.grouped_score_generic_calls += other.grouped_score_generic_calls
        self.grouped_mix_packed_cuda_calls += other.grouped_mix_packed_cuda_calls
        self.grouped_mix_fused_two_group64_calls += other.grouped_mix_fused_two_group64_calls
        self.grouped_mix_fused_generic_calls += other.grouped_mix_fused_generic_calls
        self.grouped_mix_generic_calls += other.grouped_mix_generic_calls
        self.per_kv_decode_calls += other.per_kv_decode_calls
        self.per_kv_score_chunk_count += other.per_kv_score_chunk_count
        self.per_kv_mix_chunk_count += other.per_kv_mix_chunk_count
        self.per_kv_score_chunk_pages_total += other.per_kv_score_chunk_pages_total
        self.per_kv_mix_chunk_pages_total += other.per_kv_mix_chunk_pages_total
        self.per_kv_score_chunk_pages_max = max(self.per_kv_score_chunk_pages_max, other.per_kv_score_chunk_pages_max)
        self.per_kv_mix_chunk_pages_max = max(self.per_kv_mix_chunk_pages_max, other.per_kv_mix_chunk_pages_max)
        self.per_kv_logits_elements_total += other.per_kv_logits_elements_total
        self.per_kv_weights_elements_total += other.per_kv_weights_elements_total
        self.per_kv_output_elements_total += other.per_kv_output_elements_total
        self.per_kv_score_fused_two_group64_calls += other.per_kv_score_fused_two_group64_calls
        self.per_kv_score_fused_generic_calls += other.per_kv_score_fused_generic_calls
        self.per_kv_score_generic_calls += other.per_kv_score_generic_calls
        self.per_kv_mix_fused_two_group64_calls += other.per_kv_mix_fused_two_group64_calls
        self.per_kv_mix_fused_generic_calls += other.per_kv_mix_fused_generic_calls
        self.per_kv_mix_generic_calls += other.per_kv_mix_generic_calls

    def to_dict(self) -> dict[str, int | float]:
        return {
            "m0_full_page_materializations": self.m0_full_page_materializations,
            "payload_bytes_read": self.payload_bytes_read,
            "metadata_bytes_read": self.metadata_bytes_read,
            "host_to_device_bytes": self.host_to_device_bytes,
            "max_temporary_bytes": self.max_temporary_bytes,
            "prepared_page_cache_hits": self.prepared_page_cache_hits,
            "prepared_page_cache_misses": self.prepared_page_cache_misses,
            "cache_resident_bytes": self.cache_resident_bytes,
            "prepared_page_cache_evictions": self.prepared_page_cache_evictions,
            "cache_evicted_bytes": self.cache_evicted_bytes,
            "prepare_ms_total": self.prepare_ms_total,
            "prepare_calls": self.prepare_calls,
            "score_ms_total": self.score_ms_total,
            "score_calls": self.score_calls,
            "mix_ms_total": self.mix_ms_total,
            "mix_calls": self.mix_calls,
            "softmax_ms_total": self.softmax_ms_total,
            "softmax_calls": self.softmax_calls,
            "unpack_ms_total": self.unpack_ms_total,
            "unpack_calls": self.unpack_calls,
            "fwht_ms_total": self.fwht_ms_total,
            "fwht_calls": self.fwht_calls,
            "chunk_assembly_ms_total": self.chunk_assembly_ms_total,
            "chunk_assembly_calls": self.chunk_assembly_calls,
            "grouped_decode_calls": self.grouped_decode_calls,
            "grouped_decode_output_only_calls": self.grouped_decode_output_only_calls,
            "grouped_score_chunk_count": self.grouped_score_chunk_count,
            "grouped_mix_chunk_count": self.grouped_mix_chunk_count,
            "grouped_score_chunk_pages_total": self.grouped_score_chunk_pages_total,
            "grouped_mix_chunk_pages_total": self.grouped_mix_chunk_pages_total,
            "grouped_score_chunk_pages_max": self.grouped_score_chunk_pages_max,
            "grouped_mix_chunk_pages_max": self.grouped_mix_chunk_pages_max,
            "grouped_logits_elements_total": self.grouped_logits_elements_total,
            "grouped_weights_elements_total": self.grouped_weights_elements_total,
            "grouped_output_elements_total": self.grouped_output_elements_total,
            "grouped_score_packed_cuda_calls": self.grouped_score_packed_cuda_calls,
            "grouped_score_fused_two_group64_calls": self.grouped_score_fused_two_group64_calls,
            "grouped_score_fused_generic_calls": self.grouped_score_fused_generic_calls,
            "grouped_score_generic_calls": self.grouped_score_generic_calls,
            "grouped_mix_packed_cuda_calls": self.grouped_mix_packed_cuda_calls,
            "grouped_mix_fused_two_group64_calls": self.grouped_mix_fused_two_group64_calls,
            "grouped_mix_fused_generic_calls": self.grouped_mix_fused_generic_calls,
            "grouped_mix_generic_calls": self.grouped_mix_generic_calls,
            "per_kv_decode_calls": self.per_kv_decode_calls,
            "per_kv_score_chunk_count": self.per_kv_score_chunk_count,
            "per_kv_mix_chunk_count": self.per_kv_mix_chunk_count,
            "per_kv_score_chunk_pages_total": self.per_kv_score_chunk_pages_total,
            "per_kv_mix_chunk_pages_total": self.per_kv_mix_chunk_pages_total,
            "per_kv_score_chunk_pages_max": self.per_kv_score_chunk_pages_max,
            "per_kv_mix_chunk_pages_max": self.per_kv_mix_chunk_pages_max,
            "per_kv_logits_elements_total": self.per_kv_logits_elements_total,
            "per_kv_weights_elements_total": self.per_kv_weights_elements_total,
            "per_kv_output_elements_total": self.per_kv_output_elements_total,
            "per_kv_score_fused_two_group64_calls": self.per_kv_score_fused_two_group64_calls,
            "per_kv_score_fused_generic_calls": self.per_kv_score_fused_generic_calls,
            "per_kv_score_generic_calls": self.per_kv_score_generic_calls,
            "per_kv_mix_fused_two_group64_calls": self.per_kv_mix_fused_two_group64_calls,
            "per_kv_mix_fused_generic_calls": self.per_kv_mix_fused_generic_calls,
            "per_kv_mix_generic_calls": self.per_kv_mix_generic_calls,
        }
