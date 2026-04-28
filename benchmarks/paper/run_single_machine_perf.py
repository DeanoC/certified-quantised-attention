"""Single-machine performance runner for the paper performance section.

Outputs live under runs/perf_single_machine by default. The certified flags are
imported from the same helper used by run_distributed_quality_slice.py; timing
runs default to the same full-bounded cache mode used by the distributed quality
slice runner.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks"))

from run_experiment_v2_sweep import MODEL, _common_cert_args  # noqa: E402

CONTEXTS = [8192, 32768, 65536, 131072]
CACHE_SWEEP_64K = [0, 64, 256, 512, 1024, 2048, 3072, 3584, 4096, "full_mirror"]
CACHE_SWEEP_128K = [0, 256, 1024, 2048, 4096, 5120, 6144, 7168, 8192, "full_mirror"]
CAP2048 = 2048
CONFIG = {
    "k_max": 128,
    "tau_cov": 0.995,
    "group_size": 16,
    "v_tol": 0.05,
    "use_int4_values": True,
    "use_asymmetric_keys": True,
    "fp64_accumulators": True,
    "block_size": 16,
    "score_consistency_check": False,
    "attention_backend": "native_blackwell",
}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _hardware() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return socket.gethostname()


def _envelope(sub: str, results: Any) -> dict[str, Any]:
    return {
        "machine": "perf",
        "benchmark": "performance",
        "sub_benchmark": sub,
        "timestamp": _now(),
        "config": dict(CONFIG),
        "model": MODEL,
        "hardware": _hardware(),
        "results": results,
    }


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        f.write("cmd: " + " ".join(shlex.quote(c) for c in cmd) + "\n---\n")
        f.flush()
        rc = subprocess.call(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        raise SystemExit(f"command failed rc={rc}: {' '.join(shlex.quote(c) for c in cmd)}; log={log}")


def _cert_args(context: int, *, key_cap: str | int, value_cap: str | int) -> list[str]:
    args = _common_cert_args(context)
    out: list[str] = []
    skip = False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg == "--fp16-key-cache-blocks":
            out.extend([arg, str(key_cap)])
            skip = True
        elif arg == "--fp16-value-cache-blocks":
            out.extend([arg, str(value_cap)])
            skip = True
        else:
            out.append(arg)
    return out


def _context_blocks(context: int, *, block_size: int = 16) -> int:
    return (int(context) + block_size - 1) // block_size


def _quality_full_bounded_blocks(context: int) -> int:
    return _context_blocks(context) + 1024


def _compare_cmd(
    *,
    context: int,
    steps: int,
    warmup: int,
    out_json: Path,
    key_cap: str | int,
    value_cap: str | int,
    collect_step_stats: bool = False,
    phase_profile: bool = False,
    native_profile: bool = False,
    mode: str = "both",
) -> list[str]:
    # Match pg19_perplexity.py quality semantics: --context is the full chunk
    # length and certified decode starts at eval_start=0.5.
    prefix_len = context // 2
    cache_decode_budget = context - prefix_len - 1 + 16
    return [
        sys.executable,
        str(REPO / "benchmarks" / "paper" / "compare_decode_speed.py"),
        "--mode",
        mode,
        "--context",
        str(context),
        "--prefix-len",
        str(prefix_len),
        "--chunk-tokens",
        str(context),
        "--warmup-steps",
        str(warmup),
        "--measure-steps",
        str(steps),
        "--cache-decode-budget",
        str(cache_decode_budget),
        "--output",
        str(out_json),
        *_cert_args(context, key_cap=key_cap, value_cap=value_cap),
        *(["--collect-step-stats"] if collect_step_stats else []),
        *(["--phase-profile"] if phase_profile else []),
        *(["--native-profile"] if native_profile else []),
    ]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ms_mean(raw: dict[str, Any], side: str) -> float:
    stats = raw[side].get("decode_step_ms_stats") or {}
    return float(stats.get("mean") or raw[side]["decode_ms_per_token"])


def _ms_std(raw: dict[str, Any], side: str) -> float:
    stats = raw[side].get("decode_step_ms_stats") or {}
    return float(stats.get("std") or 0.0)


def _cert_ms_mean(raw: dict[str, Any]) -> float:
    return _ms_mean(raw, "certified")


def _cert_ms_std(raw: dict[str, Any]) -> float:
    return _ms_std(raw, "certified")


def _capacity_arg(context: int, capacity: int | str) -> int:
    if capacity == "full_mirror":
        return _quality_full_bounded_blocks(context)
    return int(capacity)


def _telemetry(raw: dict[str, Any]) -> dict[str, Any]:
    cert = raw.get("certified") or {}
    return cert.get("telemetry") or {}


def _cache_hit_rate(tel: dict[str, Any], *, context: int, cap: int) -> float | None:
    hit_rate = tel.get("fp16_cache_hit_rate")
    if hit_rate is not None:
        return float(hit_rate)
    if float(tel.get("h2d_bytes_per_step") or 0.0) == 0.0 and cap >= _context_blocks(context):
        return 1.0
    return None


def run_context_scaling(args: argparse.Namespace) -> None:
    raw_dir = args.output_dir / "raw" / "context_scaling"
    results: dict[str, Any] = {}
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    for ctx in CONTEXTS:
        full_bounded = _quality_full_bounded_blocks(ctx)
        raw = raw_dir / f"context_{ctx}_quality_window_timing.json"
        tele_raw = raw_dir / f"context_{ctx}_quality_window_telemetry.json"
        if not raw.exists() or args.force:
            _run(
                _compare_cmd(
                    context=ctx,
                    steps=128,
                    warmup=8,
                    out_json=raw,
                    key_cap=full_bounded,
                    value_cap=full_bounded,
                    collect_step_stats=False,
                ),
                cwd=REPO,
                env=env,
                log=raw.with_suffix(".log"),
            )
        if not tele_raw.exists() or args.force:
            _run(
                _compare_cmd(
                    context=ctx,
                    steps=128,
                    warmup=8,
                    out_json=tele_raw,
                    key_cap=full_bounded,
                    value_cap=full_bounded,
                    collect_step_stats=True,
                ),
                cwd=REPO,
                env=env,
                log=tele_raw.with_suffix(".log"),
            )
        data = _load(raw)
        tele_data = _load(tele_raw)
        cert_tel = tele_data["certified"].get("telemetry", {})
        dense_ms = _ms_mean(data, "dense")
        cert_ms = _ms_mean(data, "certified")
        results[str(ctx)] = {
            "dense_ms_mean": dense_ms,
            "dense_ms_std": _ms_std(data, "dense"),
            "cert_ms_mean": cert_ms,
            "cert_ms_std": _ms_std(data, "certified"),
            "ratio": cert_ms / max(dense_ms, 1e-9),
            "int8_tail_fraction": cert_tel.get("int8_tail_fraction"),
            "k_star_mean": cert_tel.get("k_star_mean"),
            "rung1_expansion_rate": cert_tel.get("rung1_expansion_rate"),
            "fp16_key_cache_blocks": full_bounded,
            "fp16_value_cache_blocks": full_bounded,
            "cache_mode": "full-bounded",
            "raw": str(raw),
            "telemetry_raw": str(tele_raw),
        }
    out = args.output_dir / "perf_context_scaling.json"
    out.write_text(json.dumps(_envelope("context_scaling", results), indent=2), encoding="utf-8")
    print(f"wrote {out}")


def run_phase_breakdown(args: argparse.Namespace) -> None:
    raw = args.output_dir / "raw" / "phase_breakdown_64k.json"
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    full_bounded = _quality_full_bounded_blocks(65536)
    if not raw.exists() or args.force:
        _run(
            _compare_cmd(
                context=65536,
                steps=500,
                warmup=8,
                out_json=raw,
                key_cap=full_bounded,
                value_cap=full_bounded,
                collect_step_stats=True,
                phase_profile=True,
                native_profile=True,
            ),
            cwd=REPO,
            env=env,
            log=raw.with_suffix(".log"),
        )
    data = _load(raw)
    phase = data["certified"].get("phase_timings_ms", {})
    dense_ms = _ms_mean(data, "dense")
    cert_ms = _ms_mean(data, "certified")
    denom = cert_ms * float(data.get("measure_steps", 500))
    attention_total = sum(float(v) for v in phase.values())
    denom = denom or attention_total or 1.0
    results = {
        "phase1_int8_scoring_pct": 100.0 * float(phase.get("phase1_int8_scoring_ms", 0.0)) / denom,
        "phase2_fp16_attend_pct": 100.0 * float(phase.get("phase2_fused_attend_ms", 0.0)) / denom,
        "adaptive_selection_pct": 100.0 * float(phase.get("adaptive_selection_ms", 0.0)) / denom,
        "value_decompression_pct": 100.0 * float(phase.get("value_check_ms", 0.0)) / denom,
        "ranking_check_pct": 100.0 * float(phase.get("ranking_check_ms", 0.0)) / denom,
        "h2d_pagein_pct": 100.0 * float(phase.get("h2d_pagein_ms", 0.0)) / denom,
        "rung3_dense_recompute_pct": 100.0 * float(phase.get("rung3_dense_recompute_ms", 0.0)) / denom,
        "non_attention_pct": 100.0 * max(denom - attention_total, 0.0) / denom,
        "phase_timings_ms": phase,
        "native_profile": data["certified"].get("native_profile"),
        "total_cert_ms_mean": cert_ms,
        "total_dense_ms_mean": dense_ms,
        "overhead_ms": cert_ms - dense_ms,
        "fp16_key_cache_blocks": full_bounded,
        "fp16_value_cache_blocks": full_bounded,
        "cache_mode": "full-bounded",
        "raw": str(raw),
    }
    out = args.output_dir / "perf_phase_breakdown_64k.json"
    out.write_text(json.dumps(_envelope("phase_breakdown", results), indent=2), encoding="utf-8")
    print(f"wrote {out}")


def run_optimisation(args: argparse.Namespace) -> None:
    raw = args.output_dir / "raw" / "context_scaling" / "context_65536_quality_window_timing.json"
    if not raw.exists() or args.force:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        cap = _quality_full_bounded_blocks(65536)
        _run(
            _compare_cmd(
                context=65536,
                steps=128,
                warmup=8,
                out_json=raw,
                key_cap=cap,
                value_cap=cap,
            ),
            cwd=REPO,
            env=env,
            log=raw.with_suffix(".log"),
        )
    data = _load(raw)
    dense_ms = _ms_mean(data, "dense")
    cert_ms = _ms_mean(data, "certified")
    result = {
        "configurations": [
            {
                "name": "final_optimised_quality_matched",
                "description": (
                    "Native Blackwell backend with asymmetric INT8 keys, INT4 g16 values, "
                    "adaptive K*, ranking fallback, static resident full-bounded FP16 scratch caches."
                ),
                "cert_ms_mean": cert_ms,
                "dense_ms_mean": dense_ms,
                "ratio": cert_ms / max(dense_ms, 1e-9),
                "active_optimisations": [
                    "native_blackwell",
                    "asymmetric_int8_keys",
                    "int4_g16_values",
                    "adaptive_k_tau_cov_0.995",
                    "ranking_fallback_full_r1",
                    "static_resident_full_bounded_cache",
                ],
                "raw": str(raw),
            }
        ],
        "note": "This branch does not expose independent toggles for the rewritten pipeline stages.",
    }
    out = args.output_dir / "perf_optimisation.json"
    out.write_text(json.dumps(_envelope("optimisation", result), indent=2), encoding="utf-8")
    print(f"wrote {out}")


def run_cache_sweep(args: argparse.Namespace, *, context: int, capacities: list[int | str], sub: str, out_name: str) -> None:
    raw_dir = args.output_dir / "raw" / sub
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    rows: list[dict[str, Any]] = []
    for cap in capacities:
        cap_arg = _capacity_arg(context, cap)
        raw = raw_dir / f"context_{context}_cap_{cap}.json"
        if not raw.exists() or args.force:
            _run(
                _compare_cmd(
                    context=context,
                    steps=64,
                    warmup=8,
                    out_json=raw,
                    key_cap=cap_arg,
                    value_cap=cap_arg,
                    collect_step_stats=True,
                    mode="certified",
                ),
                cwd=REPO,
                env=env,
                log=raw.with_suffix(".log"),
            )
        data = _load(raw)
        tel = _telemetry(data)
        rows.append(
            {
                "capacity_blocks": cap,
                "fp16_key_cache_blocks": cap_arg,
                "fp16_value_cache_blocks": cap_arg,
                "cert_ms_mean": _cert_ms_mean(data),
                "cert_ms_std": _cert_ms_std(data),
                "cache_hit_rate": 1.0 if cap == "full_mirror" else tel.get("fp16_cache_hit_rate"),
                "value_cache_hit_rate": 1.0 if cap == "full_mirror" else tel.get("fp16_value_cache_hit_rate"),
                "h2d_bytes_per_step": tel.get("h2d_bytes_per_step"),
                "int8_tail_fraction": tel.get("int8_tail_fraction"),
                "raw": str(raw),
            }
        )
    out = args.output_dir / out_name
    out.write_text(json.dumps(_envelope(sub, rows), indent=2), encoding="utf-8")
    print(f"wrote {out}")


def _telemetry_result(
    data: dict[str, Any],
    *,
    raw: Path,
    cache_blocks: int,
    source: str,
    cache_mode: str = "full-bounded",
) -> dict[str, Any]:
    tel = _telemetry(data)
    return {
        "source": source,
        "rung1_trigger_rate": tel.get("rung1_trigger_rate"),
        "rung2_trigger_rate": tel.get("rung2_trigger_rate"),
        "rung3_trigger_rate": tel.get("rung3_trigger_rate"),
        "rung4_trigger_rate": tel.get("rung4_trigger_rate"),
        "ranking_consistency_fire_rate": tel.get("ranking_consistency_fire_rate"),
        "e_key_max": tel.get("e_key_max"),
        "e_key_mean": tel.get("e_key_mean"),
        "e_val_max": tel.get("e_val_max"),
        "e_val_mean": tel.get("e_val_mean"),
        "boundary_check_triggers": tel.get("boundary_check_triggers"),
        "score_consistency_violations": tel.get("score_consistency_violations"),
        "k_star_mean": tel.get("k_star_mean"),
        "k_star_max": tel.get("k_star_max"),
        "int8_tail_fraction_mean": tel.get("int8_tail_fraction"),
        "h2d_bytes_per_step": tel.get("h2d_bytes_per_step"),
        "cache_hit_rate": _cache_hit_rate(tel, context=65536, cap=cache_blocks),
        "fp16_key_cache_blocks": cache_blocks,
        "fp16_value_cache_blocks": cache_blocks,
        "cache_mode": cache_mode,
        "raw": str(raw),
    }


def run_pg19_telemetry(args: argparse.Namespace) -> None:
    context = 65536
    cap = _quality_full_bounded_blocks(context)
    raw = args.output_dir / "raw" / "telemetry_64k_pg19.json"
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if not raw.exists() or args.force:
        _run(
            _compare_cmd(
                context=context,
                steps=1000,
                warmup=8,
                out_json=raw,
                key_cap=cap,
                value_cap=cap,
                collect_step_stats=True,
                mode="certified",
            ),
            cwd=REPO,
            env=env,
            log=raw.with_suffix(".log"),
        )
    out = args.output_dir / "perf_telemetry_64k.json"
    out.write_text(
        json.dumps(_envelope("telemetry_64k_pg19", _telemetry_result(_load(raw), raw=raw, cache_blocks=cap, source="pg19")), indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out}")


def run_memory(args: argparse.Namespace) -> None:
    raw = args.output_dir / "raw" / "perf_memory_raw.json"
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if not raw.exists() or args.force:
        _run(
            [
                sys.executable,
                str(REPO / "benchmarks" / "paper" / "measure_perf_memory.py"),
                "--contexts",
                *[str(c) for c in CONTEXTS],
                "--output",
                str(raw),
                *_cert_args(65536, key_cap=_quality_full_bounded_blocks(65536), value_cap=_quality_full_bounded_blocks(65536)),
            ],
            cwd=REPO,
            env=env,
            log=raw.with_suffix(".log"),
        )
    out = args.output_dir / "perf_memory.json"
    out.write_text(json.dumps(_envelope("memory", _load(raw)), indent=2), encoding="utf-8")
    print(f"wrote {out}")


def run_context_scaling_cap2048(args: argparse.Namespace) -> None:
    raw_dir = args.output_dir / "raw" / "context_scaling_cap2048"
    results: dict[str, Any] = {}
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    for ctx in CONTEXTS:
        raw = raw_dir / f"context_{ctx}_cap2048.json"
        if not raw.exists() or args.force:
            _run(
                _compare_cmd(
                    context=ctx,
                    steps=500,
                    warmup=8,
                    out_json=raw,
                    key_cap=CAP2048,
                    value_cap=CAP2048,
                    collect_step_stats=True,
                ),
                cwd=REPO,
                env=env,
                log=raw.with_suffix(".log"),
            )
        data = _load(raw)
        tel = _telemetry(data)
        dense_ms = _ms_mean(data, "dense")
        cert_ms = _ms_mean(data, "certified")
        results[str(ctx)] = {
            "dense_ms_mean": dense_ms,
            "dense_ms_std": _ms_std(data, "dense"),
            "cert_ms_mean": cert_ms,
            "cert_ms_std": _ms_std(data, "certified"),
            "ratio": cert_ms / max(dense_ms, 1e-9),
            "int8_tail_fraction": tel.get("int8_tail_fraction"),
            "k_star_mean": tel.get("k_star_mean"),
            "rung1_expansion_rate": tel.get("rung1_expansion_rate"),
            "cache_hit_rate": _cache_hit_rate(tel, context=ctx, cap=CAP2048),
            "h2d_bytes_per_step": tel.get("h2d_bytes_per_step"),
            "fp16_key_cache_blocks": CAP2048,
            "fp16_value_cache_blocks": CAP2048,
            "cache_mode": "capped-2048",
            "raw": str(raw),
        }
    out = args.output_dir / "perf_context_scaling_cap2048.json"
    out.write_text(json.dumps(_envelope("context_scaling_cap2048", results), indent=2), encoding="utf-8")
    print(f"wrote {out}")


def run_phase_breakdown_cap2048(args: argparse.Namespace) -> None:
    raw = args.output_dir / "raw" / "phase_breakdown_64k_cap2048.json"
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if not raw.exists() or args.force:
        _run(
            _compare_cmd(
                context=65536,
                steps=500,
                warmup=8,
                out_json=raw,
                key_cap=CAP2048,
                value_cap=CAP2048,
                collect_step_stats=True,
                phase_profile=True,
                native_profile=True,
            ),
            cwd=REPO,
            env=env,
            log=raw.with_suffix(".log"),
        )
    data = _load(raw)
    phase = data["certified"].get("phase_timings_ms", {})
    tel = _telemetry(data)
    dense_ms = _ms_mean(data, "dense")
    cert_ms = _ms_mean(data, "certified")
    denom = cert_ms * float(data.get("measure_steps", 500))
    attention_total = sum(float(v) for v in phase.values())
    denom = denom or attention_total or 1.0
    results = {
        "phase1_int8_scoring_pct": 100.0 * float(phase.get("phase1_int8_scoring_ms", 0.0)) / denom,
        "phase2_fp16_attend_pct": 100.0 * float(phase.get("phase2_fused_attend_ms", 0.0)) / denom,
        "adaptive_selection_pct": 100.0 * float(phase.get("adaptive_selection_ms", 0.0)) / denom,
        "value_decompression_pct": 100.0 * float(phase.get("value_check_ms", 0.0)) / denom,
        "ranking_check_pct": 100.0 * float(phase.get("ranking_check_ms", 0.0)) / denom,
        "h2d_pagein_pct": 100.0 * float(phase.get("h2d_pagein_ms", 0.0)) / denom,
        "rung3_dense_recompute_pct": 100.0 * float(phase.get("rung3_dense_recompute_ms", 0.0)) / denom,
        "non_attention_pct": 100.0 * max(denom - attention_total, 0.0) / denom,
        "phase_timings_ms": phase,
        "native_profile": data["certified"].get("native_profile"),
        "total_cert_ms_mean": cert_ms,
        "total_dense_ms_mean": dense_ms,
        "overhead_ms": cert_ms - dense_ms,
        "cache_hit_rate": _cache_hit_rate(tel, context=65536, cap=CAP2048),
        "h2d_bytes_per_step": tel.get("h2d_bytes_per_step"),
        "fp16_key_cache_blocks": CAP2048,
        "fp16_value_cache_blocks": CAP2048,
        "cache_mode": "capped-2048",
        "raw": str(raw),
    }
    out = args.output_dir / "perf_phase_breakdown_64k_cap2048.json"
    out.write_text(json.dumps(_envelope("phase_breakdown_64k_cap2048", results), indent=2), encoding="utf-8")
    print(f"wrote {out}")


def run_memory_cap2048(args: argparse.Namespace) -> None:
    raw = args.output_dir / "raw" / "perf_memory_cap2048_raw.json"
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if not raw.exists() or args.force:
        _run(
            [
                sys.executable,
                str(REPO / "benchmarks" / "paper" / "measure_perf_memory.py"),
                "--contexts",
                *[str(c) for c in CONTEXTS],
                "--output",
                str(raw),
                *_cert_args(65536, key_cap=CAP2048, value_cap=CAP2048),
            ],
            cwd=REPO,
            env=env,
            log=raw.with_suffix(".log"),
        )
    out = args.output_dir / "perf_memory_cap2048.json"
    out.write_text(json.dumps(_envelope("memory_cap2048", _load(raw)), indent=2), encoding="utf-8")
    print(f"wrote {out}")


def run_pg19_telemetry_cap2048(args: argparse.Namespace) -> None:
    context = 65536
    raw = args.output_dir / "raw" / "telemetry_64k_pg19_cap2048.json"
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if not raw.exists() or args.force:
        _run(
            _compare_cmd(
                context=context,
                steps=1000,
                warmup=8,
                out_json=raw,
                key_cap=CAP2048,
                value_cap=CAP2048,
                collect_step_stats=True,
                mode="certified",
            ),
            cwd=REPO,
            env=env,
            log=raw.with_suffix(".log"),
        )
    out = args.output_dir / "perf_telemetry_64k_cap2048.json"
    out.write_text(
        json.dumps(
            _envelope(
                "telemetry_64k_pg19_cap2048",
                _telemetry_result(
                    _load(raw),
                    raw=raw,
                    cache_blocks=CAP2048,
                    source="pg19",
                    cache_mode="capped-2048",
                ),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


def _summarize_niah_cap2048(raw: Path, page: Path) -> dict[str, Any]:
    niah = _load(raw)
    page_data = _load(page)
    steps = page_data.get("per_step", [])
    summary = page_data.get("summary", {})
    cert = (niah.get("results", {}).get("certified") or [{}])[0]
    dense = (niah.get("results", {}).get("dense") or [{}])[0]
    heads = sum(int(s.get("ranking_heads_total", 0) or 0) for s in steps)
    rank = sum(int(s.get("ranking_fallback_triggered", 0) or 0) for s in steps)

    def vals(key: str) -> list[float]:
        return [float(s[key]) for s in steps if s.get(key) is not None]

    def mean(xs: list[float]) -> float | None:
        return float(sum(xs) / len(xs)) if xs else None

    e_key_maxes = vals("e_key_step_max")
    e_key_means = vals("e_key_step_mean")
    e_val_maxes = vals("e_val_max")
    e_val_means = vals("e_val_mean")
    skip_rates = vals("skip_rate")
    boundary = sum(int(s.get("boundary_check_triggered_heads_total", 0) or 0) for s in steps)
    violations = sum(int(s.get("score_consistency_violation_heads_total", 0) or 0) for s in steps)
    return {
        "source": "niah",
        "trial_count": 1,
        "dense_correct": bool(dense.get("correct")),
        "certified_correct": bool(cert.get("correct")),
        "rung1_trigger_rate": summary.get("rung1_rate"),
        "rung2_trigger_rate": summary.get("rung2_rate"),
        "rung3_trigger_rate": summary.get("rung3_rate"),
        "rung4_trigger_rate": summary.get("rung4_rate"),
        "ranking_consistency_fire_rate": (rank / heads) if heads else None,
        "e_key_max": max(e_key_maxes) if e_key_maxes else cert.get("e_key_step_max"),
        "e_key_mean": mean(e_key_means) if e_key_means else cert.get("e_key_step_mean"),
        "e_val_max": max(e_val_maxes) if e_val_maxes else cert.get("e_val_max"),
        "e_val_mean": mean(e_val_means) if e_val_means else cert.get("e_val_mean"),
        "boundary_check_triggers": boundary or cert.get("boundary_check_triggered_heads_total"),
        "score_consistency_violations": violations,
        "k_star_mean": summary.get("k_star_mean") if summary.get("k_star_mean") is not None else cert.get("k_star_mean"),
        "k_star_max": summary.get("k_star_max") if summary.get("k_star_max") is not None else cert.get("k_star_max"),
        "int8_tail_fraction_mean": mean(skip_rates),
        "h2d_bytes_per_step": summary.get("h2d_total_bytes_mean"),
        "cache_hit_rate": summary.get("fp16_cache_hit_rate"),
        "fp16_key_cache_blocks": CAP2048,
        "fp16_value_cache_blocks": CAP2048,
        "cache_mode": "capped-2048",
        "raw": str(raw),
        "pagein_raw": str(page),
    }


def run_niah_telemetry_cap2048(args: argparse.Namespace) -> None:
    raw = args.output_dir / "raw" / "telemetry_64k_niah_cap2048.json"
    page = args.output_dir / "raw" / "telemetry_64k_niah_cap2048.pagein.json"
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if not raw.exists() or args.force:
        _run(
            [
                sys.executable,
                str(REPO / "benchmarks" / "paper" / "niah.py"),
                "--contexts",
                "65536",
                "--needles",
                "10",
                "--trial-start",
                "0",
                "--trial-count",
                "1",
                "--output",
                str(raw),
                "--pagein-telemetry",
                "--telemetry-output",
                str(page),
                *_cert_args(65536, key_cap=CAP2048, value_cap=CAP2048),
            ],
            cwd=REPO,
            env=env,
            log=raw.with_suffix(".log"),
        )
    out = args.output_dir / "perf_telemetry_64k_niah_cap2048.json"
    out.write_text(
        json.dumps(_envelope("telemetry_64k_niah_cap2048", _summarize_niah_cap2048(raw, page)), indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "which",
        choices=[
            "p1", "p2", "p3", "p4", "p5", "p6", "p7_pg19",
            "r1_cap2048", "r2_cap2048", "r3_cap2048",
            "r4_pg19_cap2048", "r4_niah_cap2048",
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/perf_single_machine"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.which == "p1":
        run_context_scaling(args)
    elif args.which == "p2":
        run_phase_breakdown(args)
    elif args.which == "p3":
        run_optimisation(args)
    elif args.which == "p4":
        run_cache_sweep(
            args,
            context=65536,
            capacities=CACHE_SWEEP_64K,
            sub="cache_sweep_64k",
            out_name="perf_cache_sweep_64k.json",
        )
    elif args.which == "p5":
        run_cache_sweep(
            args,
            context=131072,
            capacities=CACHE_SWEEP_128K,
            sub="cache_sweep_128k",
            out_name="perf_cache_sweep_128k.json",
        )
    elif args.which == "p7_pg19":
        run_pg19_telemetry(args)
    elif args.which == "p6":
        run_memory(args)
    elif args.which == "r1_cap2048":
        run_context_scaling_cap2048(args)
    elif args.which == "r2_cap2048":
        run_phase_breakdown_cap2048(args)
    elif args.which == "r3_cap2048":
        run_memory_cap2048(args)
    elif args.which == "r4_pg19_cap2048":
        run_pg19_telemetry_cap2048(args)
    elif args.which == "r4_niah_cap2048":
        run_niah_telemetry_cap2048(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
