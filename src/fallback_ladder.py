"""Fallback ladder helpers for Rungs 2-4.

Rung 1 expansion is implemented by `compute_adaptive_topk_mask` and the
`certified_attention_layer` orchestration. This module exposes the remaining
paper-facing fallback and monitoring helpers.
"""

from __future__ import annotations

from certified_attention import (
    compute_value_error_bound,
    decide_v_format,
    decide_v_format_tight,
    detect_ranking_disagreement,
    recompute_heads_dense_fp16,
    score_consistency_violations,
    sdpa_attend_with_skip,
)

__all__ = [
    "compute_value_error_bound",
    "decide_v_format",
    "decide_v_format_tight",
    "detect_ranking_disagreement",
    "recompute_heads_dense_fp16",
    "score_consistency_violations",
    "sdpa_attend_with_skip",
]
