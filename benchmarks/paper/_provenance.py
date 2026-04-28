"""Shared cache-config provenance helper for paper benches.

Every retained paper-bench output JSON (PG-19, NIAH, and RULER) embeds the
block returned by ``cache_config_dict()`` so that a downstream auditor can
prove which quantisation config produced the numbers without re-reading
the code at the time of the run. The required ``--v-tolerance`` flag keeps
the paper operating point explicit and prevents silent config drift.

The block is intentionally small and focused on the load-bearing fields
from paper §7. Hardware / git-sha / paper-tex-sha live in the run-level
manifest (benchmarks/_manifest.py), not in every cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from typing import Any


# Direct paper-bench defaults use the measured 64K bounded-cache knee from the
# RTX PRO 6000 sweep. The v2 orchestrator overrides these with context-scaled
# bounded values, so shorter contexts do not allocate unnecessary scratch.
DEFAULT_FP16_KEY_CACHE_BLOCKS = 3584
DEFAULT_FP16_VALUE_CACHE_BLOCKS = 1536

PAPER_TAU_COV = 0.995
PAPER_K_MIN = 2
PAPER_K_MAX = 128
PAPER_RANKING_R = 1
PAPER_EPS_GUARD = 0.01
PAPER_EXPLORATION_RATE = 0.02
PAPER_RUNG1_THRESHOLD = 0.02
PAPER_RUNG1_MULTIPLIER = 2.0


def configure_paper_runtime_defaults() -> dict[str, str]:
    """Set safe performance defaults for direct paper-benchmark invocations.

    Orchestrators and developers can still override these through the
    environment. On Blackwell, the native mixed-value attention backend has a
    correctness gate and is materially faster than the Triton path for 32K+.
    Non-Blackwell devices stay on Triton.
    """
    configured: dict[str, str] = {}
    if "DOTCACHE_CERTIFIED_BACKEND" not in os.environ:
        backend = "triton"
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12:
                backend = "native_blackwell"
        except Exception:
            backend = "triton"
        os.environ["DOTCACHE_CERTIFIED_BACKEND"] = backend
        configured["DOTCACHE_CERTIFIED_BACKEND"] = backend
    return configured


def parse_cache_blocks(value: str) -> int | str:
    """argparse type for bounded-cache capacity flags.

    Integer values mean bounded scratch/cache capacity in blocks. ``full`` is
    kept as an explicit opt-in for legacy/debug comparisons; paper runs should
    use bounded values so the VRAM accounting remains meaningful.
    """
    text = str(value).strip().lower()
    if text in {"full", "mirror", "none"}:
        return "full"
    try:
        parsed = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cache block count must be >= 0 or 'full'") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("cache block count must be >= 0 or 'full'")
    return parsed


def resolve_fp16_value_cache_blocks(
    value: int | str | None,
    env_value: str | None,
) -> int | None:
    """Resolve paper/default FP16 value scratch capacity.

    ``None`` means no CLI override was supplied by a direct Python caller, so
    honor the environment and then fall back to the paper-exact bounded default.
    The string ``"full"`` is an explicit legacy full-mirror request.
    """
    if isinstance(value, str) and value.strip().lower() in {"full", "mirror", "none"}:
        return None
    if value is not None:
        return int(value)
    if env_value is not None and env_value != "":
        env_parsed = parse_cache_blocks(env_value)
        if isinstance(env_parsed, str):
            return None
        return int(env_parsed)
    return DEFAULT_FP16_VALUE_CACHE_BLOCKS


def resolve_fp16_key_cache_blocks(
    value: int | str | None,
    env_value: str | None,
) -> int | None:
    if isinstance(value, str) and value.strip().lower() in {"full", "mirror", "none"}:
        return None
    if value is not None:
        return int(value)
    if env_value is not None and env_value != "":
        env_parsed = parse_cache_blocks(env_value)
        if isinstance(env_parsed, str):
            return None
        return int(env_parsed)
    return DEFAULT_FP16_KEY_CACHE_BLOCKS


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _quantisation_mode(use_int4_values: bool, group_size: int) -> str:
    keys = "asymmetric_int8_keys"  # set by Step 1; will be the only path
    values = f"int4_g{group_size}_values" if use_int4_values else "fp16_values"
    return f"{keys}+{values}"


def cache_config_dict(args: argparse.Namespace) -> dict[str, Any]:
    """Build the cache-config provenance block from parsed CLI args.

    All paper benches use a common subset of CLI flags
    (--v-tolerance, --use-int4-values, --group-size, plus the §7 knobs);
    this helper consumes that namespace and returns a dict suitable for
    embedding into the cell's output JSON.
    """
    use_int4 = bool(getattr(args, "use_int4_values", False))
    group_size = int(getattr(args, "group_size", 16))
    config = {
        "v_tolerance": float(args.v_tolerance),
        "quantization_mode": _quantisation_mode(use_int4, group_size),
        "asymmetric_keys": True,  # Step 1 makes this the only path
        "use_int4_values": use_int4,
        "group_size": group_size if use_int4 else None,
        "score_consistency_check": bool(getattr(args, "score_consistency_check", False)),
        "tau_cov": float(getattr(args, "tau_cov", PAPER_TAU_COV)) or None,
        "k_min": int(getattr(args, "k_min", PAPER_K_MIN)),
        "k_max": getattr(args, "k_max", PAPER_K_MAX),
        "ranking_fallback": bool(getattr(args, "ranking_fallback", True)),
        "ranking_r": int(getattr(args, "ranking_r", PAPER_RANKING_R)),
        "ranking_fallback_mode": getattr(args, "ranking_fallback_mode", "full"),
        "fp16_key_cache_blocks": getattr(args, "fp16_key_cache_blocks", None),
        "fp16_value_cache_blocks": getattr(args, "fp16_value_cache_blocks", None),
        "eps_guard": float(getattr(args, "eps_guard", PAPER_EPS_GUARD)),
        "exploration_rate": float(getattr(args, "exploration_rate", PAPER_EXPLORATION_RATE)),
        "rung1_threshold": float(getattr(args, "rung1_threshold", PAPER_RUNG1_THRESHOLD)),
        "rung1_multiplier": float(getattr(args, "rung1_multiplier", PAPER_RUNG1_MULTIPLIER)),
        "attention_backend": os.environ.get("DOTCACHE_CERTIFIED_BACKEND", "triton"),
        "static_resident_cache": os.environ.get("DOTCACHE_STATIC_RESIDENT_CACHE", "1"),
        "splitk_blocks_per_split": os.environ.get(
            "DOTCACHE_NATIVE_MIXEDV_BLOCKS_PER_SPLIT",
            os.environ.get(
            "DOTCACHE_MIXEDV_SPLITK_BLOCKS_PER_SPLIT",
            os.environ.get("DOTCACHE_SPLITK_BLOCKS_PER_SPLIT", "auto_native_32_24_16"),
            ),
        ),
        "cuda_graph_decode": bool(getattr(args, "cuda_graph_decode", False)),
        "runtime_profile": os.environ.get("DOTCACHE_RUNTIME_PROFILE", "0"),
        "code_sha": _git_sha(),
    }
    config["dotcache_config_hash"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    return config


def add_paper_cache_args(parser: argparse.ArgumentParser) -> None:
    """Attach the paper cache-format flags to a bench's argparse.

    Used by PG-19, NIAH, and RULER so the flag spelling is identical across
    benches. v_tolerance is REQUIRED; there is no silent default.
    """
    parser.add_argument(
        "--v-tolerance", type=float, required=True,
        help="INT4-vs-FP16 value-format threshold (paper §7: 0.05). REQUIRED — "
             "no silent default. The kernel raises if this isn't carried through.",
    )
    parser.add_argument(
        "--use-int4-values", action="store_true",
        help="Use INT4 per-group values (paper §3.1/§7). Without this flag, "
             "values stay FP16 (legacy ad-hoc-bench behaviour).",
    )
    parser.add_argument(
        "--group-size", type=int, default=16,
        help="INT4 value group size (paper §7: 16). Ignored unless "
             "--use-int4-values is set.",
    )
    parser.add_argument(
        "--fp16-key-cache-blocks", type=parse_cache_blocks,
        default=DEFAULT_FP16_KEY_CACHE_BLOCKS,
        help="Bounded GPU FP16 key scratch capacity in blocks "
             f"(paper default: {DEFAULT_FP16_KEY_CACHE_BLOCKS}). Use 'full' "
             "only for legacy/debug full mirror.",
    )
    parser.add_argument(
        "--fp16-value-cache-blocks", type=parse_cache_blocks,
        default=DEFAULT_FP16_VALUE_CACHE_BLOCKS,
        help="Bounded GPU FP16 value fallback scratch capacity in blocks "
             f"(paper default: {DEFAULT_FP16_VALUE_CACHE_BLOCKS}). Use 0 for "
             "one-step page-in only, or 'full' only for legacy/debug full mirror.",
    )


def add_paper_section7_args(parser: argparse.ArgumentParser) -> None:
    """Attach the paper certified-config knobs with paper defaults."""
    parser.add_argument("--tau-cov", type=float, default=PAPER_TAU_COV,
                        help=f"Adaptive K* cumulative-mass threshold (paper default: {PAPER_TAU_COV})")
    parser.add_argument("--k-min", type=int, default=PAPER_K_MIN)
    parser.add_argument("--k-max", type=int, default=PAPER_K_MAX,
                        help=f"Adaptive K* upper clamp (paper default: {PAPER_K_MAX})")
    parser.add_argument("--ranking-fallback", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Rung-3 ranking-consistency fallback (paper default: enabled)")
    parser.add_argument("--ranking-r", type=int, default=PAPER_RANKING_R)
    parser.add_argument("--ranking-fallback-mode", default="full",
                        choices=["full", "measure"])
    parser.add_argument("--score-consistency-check", action="store_true")
    parser.add_argument("--eps-guard", type=float, default=PAPER_EPS_GUARD)
    parser.add_argument("--exploration-rate", type=float, default=PAPER_EXPLORATION_RATE)
    parser.add_argument("--rung1-threshold", type=float, default=PAPER_RUNG1_THRESHOLD)
    parser.add_argument("--rung1-multiplier", type=float, default=PAPER_RUNG1_MULTIPLIER)
