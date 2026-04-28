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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cutlass_root() -> Path:
    return _repo_root() / "third_party" / "cutlass"


def cutlass_sm120_available() -> bool:
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:
        return False
    if not (torch.cuda.is_available() and CUDA_HOME):
        return False
    if torch.cuda.get_device_capability()[0] < 12:
        return False
    if not (cutlass_root() / "include" / "cutlass" / "version.h").exists():
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
    cpp = base / "cutlass_sm120.cpp"
    cu = base / "cutlass_sm120_kernel.cu"
    cutlass = cutlass_root()
    version_file = cutlass / "include" / "cutlass" / "version.h"
    if not version_file.exists():
        raise FileNotFoundError(
            f"CUTLASS headers not found at {version_file}; run "
            "`git submodule update --init third_party/cutlass`"
        )

    digest_src = cpp.read_text() + cu.read_text() + version_file.read_text()
    digest = hashlib.sha1(digest_src.encode("utf-8")).hexdigest()[:12]
    build_dir = base / ".build_cutlass_sm120"
    build_dir.mkdir(parents=True, exist_ok=True)

    include_paths = [
        str(cutlass / "include"),
        str(cutlass / "tools" / "util" / "include"),
    ]
    _EXTENSION = load(
        name=f"dotcache_cutlass_sm120_{digest}",
        sources=[str(cpp), str(cu)],
        extra_include_paths=include_paths,
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "-gencode=arch=compute_120,code=sm_120",
        ],
        build_directory=str(build_dir),
        verbose=_env_flag("DOTCACHE_CUTLASS_VERBOSE", default=False),
    )
    return _EXTENSION


def cutlass_sm120_metadata() -> dict[str, Any]:
    ext = _load_extension()
    meta = ext.cutlass_sm120_metadata()
    return {
        "backend": "cutlass_sm120",
        "metadata": str(meta),
        "cutlass_root": str(cutlass_root()),
    }


def cutlass_sm120_probe(x: Any) -> Any:
    """Tiny CUDA ABI probe used by tests and the performance gate.

    This is intentionally not a certified-attention implementation. It proves
    the vendored CUTLASS headers and SM120 extension toolchain are usable
    before the tensor-core kernels are wired into the paper path.
    """
    return _load_extension().cutlass_sm120_probe(x)


def dequant_keys_to_fp16_t(
    keys_int8: Any,
    keys_scale: Any,
    keys_zero_points: Any,
    *,
    block_size: int = 16,
) -> Any:
    """Dequantize asymmetric INT8 keys into tensor-core-friendly layout.

    Returns `[kv_heads, head_dim, tokens]` FP16, matching the B operand layout
    used by the score-phase batched GEMM feasibility bound.
    """
    return _load_extension().dequant_keys_to_fp16_t(
        keys_int8.contiguous(),
        keys_scale.contiguous(),
        keys_zero_points.contiguous(),
        int(block_size),
    )


def hybrid_mixedv_split_k_cutlass(**kwargs: Any) -> Any:
    """Future tensor-core mixed-value attention entrypoint.

    Keep the call signature identical to the Triton/native mixed-value
    backends so certified_attention can switch backends without changing paper
    semantics. Until the real CUTLASS kernels pass the performance gates, the
    public wrapper deliberately raises so callers can fall back exactly.
    """
    raise NotImplementedError(
        "cutlass_sm120 mixed-value attention is not implemented yet; "
        "use the Triton fallback until the tensor-core kernels pass gates"
    )


def score_certify_cutlass(**kwargs: Any) -> Any:
    """Future tensor-core score/certify entrypoint.

    This v0 implementation is an SM120 extension ABI/correctness prototype,
    not the final tensor-core kernel. It remains opt-in behind
    DOTCACHE_CUTLASS_SM120_ENABLE_SCORE and must not become the default unless
    the score gate shows a real speedup.
    """
    ext = _load_extension()
    m_b, s_b, skip_i32 = ext.score_certify_sm120(
        kwargs["K_int8_packed"].contiguous(),
        kwargs["K_scale"].contiguous(),
        kwargs["K_zero_points"].contiguous(),
        kwargs["q_all"].contiguous(),
        kwargs["correction"].contiguous(),
        int(kwargs["gqa_group"]),
        int(kwargs.get("block_size", 16)),
        float(kwargs.get("q_scale", 1.0)),
        float(kwargs.get("block_epsilon", 0.001)),
    )
    return m_b, s_b, skip_i32.bool()
