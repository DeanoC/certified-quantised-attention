"""Minimal configuration object required by the paper benchmark scripts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DotCacheConfig:
    """Small subset of the experimental DotCache config used by benchmarks."""

    head_dim: int
    group_size: int = 32
    bits_k: int = 4
    bits_v: int = 4
    tokens_per_page: int = 64

    def __post_init__(self) -> None:
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be positive")

