"""Adaptive top-K*, coverage, and Rung-1 selection helpers.

The implementation lives in `certified_attention.py` because the selector is
used directly inside the fused per-layer attention path. This module gives
reviewers the paper-facing entrypoint named in the repository layout.
"""

from __future__ import annotations

from certified_attention import (
    DEFAULT_K_MAX,
    DEFAULT_K_MIN,
    DEFAULT_RUNG1_MULTIPLIER,
    DEFAULT_RUNG1_THRESHOLD,
    DEFAULT_TAU_COV,
    augment_mask_with_exploration,
    compute_adaptive_topk_mask,
    compute_delta_bound,
    compute_fp16_block_scores,
    compute_tier2_residual_mass,
)

__all__ = [
    "DEFAULT_K_MAX",
    "DEFAULT_K_MIN",
    "DEFAULT_RUNG1_MULTIPLIER",
    "DEFAULT_RUNG1_THRESHOLD",
    "DEFAULT_TAU_COV",
    "augment_mask_with_exploration",
    "compute_adaptive_topk_mask",
    "compute_delta_bound",
    "compute_fp16_block_scores",
    "compute_tier2_residual_mass",
]
