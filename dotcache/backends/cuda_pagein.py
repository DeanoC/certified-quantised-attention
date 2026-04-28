from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any


_EXTENSION = None


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def cuda_pagein_enabled() -> bool:
    return _env_flag("DOTCACHE_NATIVE_PAGEIN", default=True)


def cuda_pagein_available() -> bool:
    if not cuda_pagein_enabled():
        return False
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:
        return False
    if not (torch.cuda.is_available() and CUDA_HOME):
        return False
    try:
        _load_extension()
    except Exception:
        return False
    return True


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    import torch
    from torch.utils.cpp_extension import load

    base = Path(__file__).resolve().parent / "cuda_kernels"
    cpp = base / "cuda_pagein.cpp"
    cu = base / "cuda_pagein_kernel.cu"
    digest = hashlib.sha1((cpp.read_text() + cu.read_text()).encode("utf-8")).hexdigest()[:12]
    build_dir = base / ".build"
    build_dir.mkdir(parents=True, exist_ok=True)
    _EXTENSION = load(
        name=f"dotcache_cuda_pagein_{digest}",
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


def page_in_fp16_blocks_cuda(
    *,
    src_cpu: Any,
    dst_gpu: Any,
    loaded_blocks_cpu: Any,
    loaded_slots_cpu: Any,
    evicted_blocks_cpu: Any,
    loaded_blocks_gpu: Any,
    loaded_slots_gpu: Any,
    evicted_blocks_gpu: Any,
    slot_table_gpu: Any,
    block_size: int,
    active_tokens: int,
) -> int:
    ext = _load_extension()
    return int(ext.page_in_fp16_blocks_cuda(
        src_cpu.contiguous(),
        dst_gpu.contiguous(),
        loaded_blocks_cpu.contiguous(),
        loaded_slots_cpu.contiguous(),
        evicted_blocks_cpu.contiguous(),
        loaded_blocks_gpu.contiguous(),
        loaded_slots_gpu.contiguous(),
        evicted_blocks_gpu.contiguous(),
        slot_table_gpu.contiguous(),
        int(block_size),
        int(active_tokens),
    ))


def page_in_fp16_blocks_packed_cuda(
    *,
    src_cpu: Any,
    dst_gpu: Any,
    stage_cpu: Any,
    stage_gpu: Any,
    loaded_blocks_cpu: Any,
    loaded_slots_cpu: Any,
    evicted_blocks_cpu: Any,
    loaded_blocks_gpu: Any,
    loaded_slots_gpu: Any,
    evicted_blocks_gpu: Any,
    slot_table_gpu: Any,
    block_size: int,
    active_tokens: int,
) -> int:
    ext = _load_extension()
    return int(ext.page_in_fp16_blocks_packed_cuda(
        src_cpu.contiguous(),
        dst_gpu.contiguous(),
        stage_cpu.contiguous(),
        stage_gpu.contiguous(),
        loaded_blocks_cpu.contiguous(),
        loaded_slots_cpu.contiguous(),
        evicted_blocks_cpu.contiguous(),
        loaded_blocks_gpu.contiguous(),
        loaded_slots_gpu.contiguous(),
        evicted_blocks_gpu.contiguous(),
        slot_table_gpu.contiguous(),
        int(block_size),
        int(active_tokens),
    ))


def page_in_fp16_blocks_by_slots_cuda(
    *,
    src_cpu: Any,
    dst_gpu: Any,
    block_slots_gpu: Any,
    block_size: int,
    active_tokens: int,
    n_blocks: int,
) -> None:
    ext = _load_extension()
    ext.page_in_fp16_blocks_by_slots_cuda(
        src_cpu.contiguous(),
        dst_gpu.contiguous(),
        block_slots_gpu.contiguous(),
        int(block_size),
        int(active_tokens),
        int(n_blocks),
    )
