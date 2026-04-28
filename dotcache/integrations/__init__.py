"""LLaMA integration retained for the paper benchmark path."""

from .llama import (
    CertifiedAttentionState,
    LlamaDotCacheHarness,
    LlamaDotCacheModelAdapter,
    LlamaReplayRecord,
    build_llama_page_trace_records,
    build_llama_prefill_page_trace_records,
    export_llama_page_traces,
    run_llama_generation_harness,
    run_llama_page_trace_capture_harness,
    run_llama_replay_harness,
    transformers_available,
)

__all__ = [
    "CertifiedAttentionState",
    "LlamaDotCacheHarness",
    "LlamaDotCacheModelAdapter",
    "LlamaReplayRecord",
    "build_llama_page_trace_records",
    "build_llama_prefill_page_trace_records",
    "export_llama_page_traces",
    "run_llama_generation_harness",
    "run_llama_page_trace_capture_harness",
    "run_llama_replay_harness",
    "transformers_available",
]
