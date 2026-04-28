from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch


_EXTENSION = None
_PROFILE = {
    "calls": 0,
    "partial_ms_total": 0.0,
    "reduce_ms_total": 0.0,
}


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def certified_blackwell_available() -> bool:
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:
        return False
    if not (torch.cuda.is_available() and CUDA_HOME):
        return False
    if torch.cuda.get_device_capability()[0] < 12:
        return False
    try:
        _load_extension()
    except Exception:
        return False
    return True


def reset_native_profile() -> None:
    _PROFILE["calls"] = 0
    _PROFILE["partial_ms_total"] = 0.0
    _PROFILE["reduce_ms_total"] = 0.0


def native_profile_summary() -> dict[str, float | int]:
    calls = int(_PROFILE["calls"])
    partial = float(_PROFILE["partial_ms_total"])
    reduce = float(_PROFILE["reduce_ms_total"])
    total = partial + reduce
    return {
        "calls": calls,
        "partial_ms_total": partial,
        "reduce_ms_total": reduce,
        "total_ms": total,
        "partial_ms_per_call": partial / calls if calls else 0.0,
        "reduce_ms_per_call": reduce / calls if calls else 0.0,
    }


def score_blocks_cuda(
    *,
    keys_int8: Any,
    keys_scale: Any,
    keys_zero_points: Any,
    q_all: Any,
    gqa_group: int,
    block_size: int = 16,
    q_scale: float = 1.0,
    blocks_per_chunk: int | None = None,
) -> tuple[Any, Any]:
    ext = _load_extension()
    if blocks_per_chunk is None:
        blocks_per_chunk = int(os.environ.get("DOTCACHE_NATIVE_SCORE_BLOCKS_PER_CHUNK", "16"))
    return ext.score_blocks_cuda(
        keys_int8.contiguous(),
        keys_scale.contiguous(),
        keys_zero_points.contiguous(),
        q_all.contiguous(),
        int(gqa_group),
        int(block_size),
        float(q_scale),
        int(blocks_per_chunk),
    )


def adaptive_topk_cuda(
    *,
    m_b: Any,
    s_b: Any,
    tau_cov: float,
    k_min: int,
    k_max: int,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    ext = _load_extension()
    return tuple(
        ext.adaptive_topk_cuda(
            m_b.contiguous(),
            s_b.contiguous(),
            float(tau_cov),
            int(k_min),
            int(k_max),
        )
    )


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    import torch
    from torch.utils.cpp_extension import load

    base = Path(__file__).resolve().parent / "cuda_kernels"
    cpp = base / "certified_blackwell.cpp"
    cu = base / "certified_blackwell_kernel.cu"
    digest = hashlib.sha1((cpp.read_text() + cu.read_text()).encode("utf-8")).hexdigest()[:12]
    build_dir = base / ".build"
    build_dir.mkdir(parents=True, exist_ok=True)
    _EXTENSION = load(
        name=f"dotcache_certified_blackwell_{digest}",
        sources=[str(cpp), str(cu)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
        ],
        build_directory=str(build_dir),
        verbose=_env_flag("DOTCACHE_NATIVE_VERBOSE", default=False),
    )
    return _EXTENSION


def hybrid_mixedv_split_k_cuda(
    *,
    keys_int8: Any,
    keys_scale: Any,
    keys_zero_points: Any,
    keys_fp16: Any,
    key_block_slots: Any | None = None,
    topk_mask: Any,
    values_int4_packed: Any,
    values_int4_scales: Any,
    values_int4_zeros: Any,
    values_fp16_scratch: Any,
    value_fp16_mask: Any,
    value_block_slots: Any,
    q_all: Any,
    gqa_group: int,
    block_size: int = 16,
    group_size: int = 16,
    q_scale: float = 1.0,
    last_block_valid: int | None = None,
    num_splits: int | None = None,
    int8_token_scores: Any | None = None,
    workspace: Any | None = None,
    **_unused: Any,
) -> Any:
    ext = _load_extension()
    if num_splits is None:
        num_blocks = int(keys_int8.shape[1]) // int(block_size)
        target_env = os.environ.get("DOTCACHE_NATIVE_MIXEDV_BLOCKS_PER_SPLIT")
        if target_env is not None:
            target = int(target_env)
        elif num_blocks >= 4096:
            target = 16
        elif num_blocks >= 2048:
            target = 24
        else:
            target = 32
        ns = max(1, (num_blocks + target - 1) // target)
        num_splits = 1
        while num_splits < ns:
            num_splits *= 2
        num_splits = min(num_splits, num_blocks)
    lbv = int(block_size if last_block_valid is None else last_block_valid)
    if key_block_slots is None:
        num_blocks = int(keys_scale.shape[1])
        key_block_slots = torch.arange(num_blocks, dtype=torch.int32, device=keys_int8.device)
    if topk_mask.dtype != torch.int32:
        topk_mask = topk_mask.to(torch.int32)
    if value_fp16_mask.dtype != torch.int32:
        value_fp16_mask = value_fp16_mask.to(torch.int32)
    if value_block_slots.dtype != torch.int32:
        value_block_slots = value_block_slots.to(torch.int32)
    if key_block_slots.dtype != torch.int32:
        key_block_slots = key_block_slots.to(torch.int32)
    use_score_cache = int8_token_scores is not None and int(int8_token_scores.numel()) > 0
    if int8_token_scores is None:
        int8_token_scores = torch.empty(0, dtype=torch.float32, device=keys_int8.device)
    args = (
        keys_int8.contiguous(),
        keys_scale.contiguous(),
        keys_zero_points.contiguous(),
        keys_fp16.contiguous(),
        key_block_slots.contiguous(),
        topk_mask.contiguous(),
        values_int4_packed.contiguous(),
        values_int4_scales.contiguous(),
        values_int4_zeros.contiguous(),
        values_fp16_scratch.contiguous(),
        value_fp16_mask.contiguous(),
        value_block_slots.contiguous(),
        q_all.contiguous(),
        int8_token_scores.contiguous(),
        bool(use_score_cache),
        int(gqa_group),
        int(block_size),
        int(group_size),
        float(q_scale),
        int(lbv),
        int(num_splits),
    )
    if _env_flag("DOTCACHE_NATIVE_PROFILE", default=False):
        out, timing = ext.hybrid_mixedv_split_k_cuda_profile(*args)
        _PROFILE["calls"] += 1
        _PROFILE["partial_ms_total"] += float(timing[0].item())
        _PROFILE["reduce_ms_total"] += float(timing[1].item())
        return out
    return ext.hybrid_mixedv_split_k_cuda(*args)
