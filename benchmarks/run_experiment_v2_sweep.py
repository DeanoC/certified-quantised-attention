"""Experiment Specification v2 clean-rerun orchestrator.

Runs the corrected-code quality matrix for the paper:
PG-19, NIAH, and RULER at 8K/32K/64K plus selected 128K cells.

This runner writes one wrapped JSON per cell under
benchmarks/results/paper_v2_20260425/. The wrapped JSON has a stable
paper-integration shape: benchmark, context_length, config, model, hardware,
results, telemetry, native.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "benchmarks" / "results" / "paper_v2_20260425"
BRANCH = "port-to-paper-20260424"
MODEL = "NousResearch/Meta-Llama-3.1-8B"
sys.path.insert(0, str(REPO))

CONTEXTS_MAIN = [8192, 32768, 65536]
BLOCK_SIZE = 16

CERT_FLAGS: dict[str, str] = {
    "v_tolerance": "0.05",
    "tau_cov": "0.995",
    "k_min": "2",
    "k_max": "128",
    "ranking_r": "1",
    "eps_guard": "0.01",
    "exploration_rate": "0.02",
    "rung1_threshold": "0.02",
    "rung1_multiplier": "2.0",
}


def recommended_fp16_cache_blocks(context_length: int, *, block_size: int = BLOCK_SIZE) -> tuple[int, int]:
    """Return bounded FP16 key/value scratch sizes for paper v2 runs.

    The 64K sweep on this RTX PRO 6000 found the practical knee at
    key/value = 3584/1536 blocks. Expressing that as fractions of the active
    4096 blocks keeps the setting bounded and scales consistently across
    contexts without silently switching to a full FP16 mirror.
    """
    n_blocks = (int(context_length) + int(block_size) - 1) // int(block_size)
    key_blocks = (n_blocks * 7) // 8
    value_blocks = (n_blocks * 3) // 8
    return max(1, key_blocks), max(1, value_blocks)

CELL_ESTIMATE_HOURS: dict[tuple[str, int], float] = {
    # PG-19 estimates are calibrated on this RTX PRO 6000 host from the
    # corrected certified path on 2026-04-25:
    # 8K: 512-step probe at ~5.4 tok/s; 32K/64K: split-K mixed-value probes
    # at ~1.9/~1.8 tok/s. These replace the paper's pre-fix estimates.
    ("pg19", 8192): 4.4,
    ("pg19", 32768): 48.0,
    ("pg19", 65536): 100.0,
    ("pg19", 131072): 65.0,
    ("niah", 8192): 2.0,
    ("niah", 32768): 6.0,
    ("niah", 65536): 14.0,
    ("niah", 131072): 15.0,  # reduced 50-trial optional cell.
    ("ruler", 8192): 3.0,
    ("ruler", 32768): 8.0,
    ("ruler", 65536): 18.0,
    ("ruler", 131072): 16.0,  # reduced 20-sample optional cell.
}

NON_CELL_ESTIMATE_HOURS: dict[str, dict[str, Any]] = {
    "performance_profiling": {"tier": 1, "hours": 2.0},
    "value_group_size_sweep": {"tier": 2, "hours": 1.0},
    "niah_precision_ablation": {"tier": 2, "hours": 4.0},
    "tau_cov_sweep_8k": {"tier": 3, "hours": 6.0},
    "tau_cov_sweep_32k": {"tier": 4, "hours": 18.0},
}


def build_cells(*, include_tier2: bool = True, include_tier3: bool = False,
                include_tier4: bool = False) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    idx = 0

    def add(bench: str, ctx: int, tier: int, **kwargs: Any) -> None:
        nonlocal idx
        idx += 1
        cells.append({"idx": f"{idx:02d}", "bench": bench, "ctx": ctx, "tier": tier, **kwargs})

    # Tier 1: must-have quality matrix, excluding 128K.
    for ctx in CONTEXTS_MAIN:
        add("pg19", ctx, 1, chunks=20)
        add("niah", ctx, 1, needles=10)
        add("ruler", ctx, 1, samples=50)

    # Tier 2: strong-to-have long-context PG-19 point estimate.
    if include_tier2:
        add("pg19", 131072, 2, chunks=5)

    # Tier 3: optional reduced 128K NIAH, 50 paired trials = 5 needles x 10 depths.
    if include_tier3:
        add("niah", 131072, 3, needles=5)

    # Tier 4: optional reduced 128K RULER, 20 samples x 7 subtasks.
    if include_tier4:
        add("ruler", 131072, 4, samples=20)

    return cells


def estimate_cell_hours(cell: dict[str, Any]) -> float:
    return float(CELL_ESTIMATE_HOURS.get((str(cell["bench"]), int(cell["ctx"])), 0.0))


def estimate_non_cell_hours(max_tier: int) -> float:
    return float(
        sum(float(v["hours"]) for v in NON_CELL_ESTIMATE_HOURS.values() if int(v["tier"]) <= max_tier)
    )


def schedule_cells(cells: list[dict[str, Any]], machines: int) -> list[dict[str, Any]]:
    machines = max(1, int(machines))
    slots = [{"machine": i + 1, "hours": 0.0, "cells": []} for i in range(machines)]
    for cell in sorted(cells, key=estimate_cell_hours, reverse=True):
        slot = min(slots, key=lambda s: float(s["hours"]))
        hours = estimate_cell_hours(cell)
        slot["cells"].append(cell)
        slot["hours"] = float(slot["hours"]) + hours
    return slots


def format_hours(hours: float) -> str:
    return f"{hours:.1f}h ({hours / 24.0:.1f}d)"


def _common_cert_args(context_length: int, group_size: int = 16) -> list[str]:
    key_cache_blocks, value_cache_blocks = recommended_fp16_cache_blocks(context_length)
    args = [
        "--model", MODEL,
        "--v-tolerance", CERT_FLAGS["v_tolerance"],
        "--use-int4-values",
        "--group-size", str(group_size),
        "--tau-cov", CERT_FLAGS["tau_cov"],
        "--k-min", CERT_FLAGS["k_min"],
        "--k-max", CERT_FLAGS["k_max"],
        "--ranking-fallback",
        "--ranking-r", CERT_FLAGS["ranking_r"],
        "--ranking-fallback-mode", "full",
        "--eps-guard", CERT_FLAGS["eps_guard"],
        "--exploration-rate", CERT_FLAGS["exploration_rate"],
        "--rung1-threshold", CERT_FLAGS["rung1_threshold"],
        "--rung1-multiplier", CERT_FLAGS["rung1_multiplier"],
    ]
    args.extend(["--fp16-key-cache-blocks", str(key_cache_blocks)])
    args.extend(["--fp16-value-cache-blocks", str(value_cache_blocks)])
    return args


def _cli_for_cell(cell: dict[str, Any], out_json: Path, *, smoke: bool) -> list[str]:
    bench = str(cell["bench"])
    ctx = int(cell["ctx"])
    if bench == "pg19":
        chunks = 1 if smoke else int(cell.get("chunks", 20))
        return [
            sys.executable, str(REPO / "benchmarks" / "paper" / "pg19_perplexity.py"),
            "--context", str(ctx),
            "--num-chunks", str(chunks),
            "--telemetry-mode", "summary",
            "--certified-warmup-steps", "4" if smoke else "128",
            "--output", str(out_json),
            *_common_cert_args(ctx),
        ]
    if bench == "niah":
        needles = 1 if smoke else int(cell.get("needles", 10))
        return [
            sys.executable, str(REPO / "benchmarks" / "paper" / "niah.py"),
            "--contexts", str(ctx),
            "--needles", str(needles),
            "--output", str(out_json),
            *_common_cert_args(ctx),
        ]
    if bench == "ruler":
        samples = 1 if smoke else int(cell.get("samples", 50))
        return [
            sys.executable, str(REPO / "benchmarks" / "paper" / "ruler.py"),
            "--contexts", str(ctx),
            "--num-samples", str(samples),
            "--output", str(out_json),
            *_common_cert_args(ctx),
        ]
    raise ValueError(f"unknown bench: {bench}")


def _runner_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _git(*args: str) -> tuple[int, str]:
    res = subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO)
    return res.returncode, (res.stdout + res.stderr)


def git_sha() -> str:
    rc, out = _git("rev-parse", "--short", "HEAD")
    return out.strip() if rc == 0 else "unknown"


def current_branch() -> str:
    rc, out = _git("branch", "--show-current")
    return out.strip() if rc == 0 else "unknown"


def _hw_tag() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return f"{p.name} sm_{p.major}{p.minor}"
    except Exception:
        pass
    return "unknown"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"_parse_error": str(exc)}


def _paper_config(native: dict[str, Any] | None = None, *, context_length: int | None = None) -> dict[str, Any]:
    cache_config = native.get("cache_config", {}) if isinstance(native, dict) else {}
    fallback_key_blocks, fallback_value_blocks = (
        recommended_fp16_cache_blocks(context_length)
        if context_length is not None else (None, None)
    )
    return {
        "k_max": int(CERT_FLAGS["k_max"]),
        "tau_cov": float(CERT_FLAGS["tau_cov"]),
        "group_size": int(cache_config.get("group_size") or 16),
        "v_tol": float(CERT_FLAGS["v_tolerance"]),
        "use_int4_values": True,
        "use_asymmetric_keys": True,
        "fp64_accumulators": True,
        "block_size": 16,
        "ranking_r": int(CERT_FLAGS["ranking_r"]),
        "ranking_fallback_mode": "full",
        "fp16_key_cache_blocks": cache_config.get("fp16_key_cache_blocks", fallback_key_blocks),
        "fp16_value_cache_blocks": cache_config.get("fp16_value_cache_blocks", fallback_value_blocks),
    }


def _sum_keys(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, int]:
    return {key: int(sum(int(r.get(key, 0) or 0) for r in rows)) for key in keys}


def _max_key(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    return float(max(vals)) if vals else None


def _native_telemetry(bench: str, native: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(native, dict):
        return {}
    if bench == "pg19":
        cert = native.get("certified") or {}
        telem = cert.get("telemetry") or {}
        return {
            "int8_tail_fraction": cert.get("skip_rate"),
            "rung_triggers": {
                "rung1": telem.get("rung1_fired_steps"),
                "rung2": telem.get("rung2_fired_steps"),
                "rung3": telem.get("rung3_fired_steps"),
                "rung4": telem.get("rung4_fired_steps"),
            },
            "e_key_max": telem.get("e_key_step_max"),
            "e_val_max": telem.get("e_val_step_max"),
            "boundary_check_triggers": telem.get("boundary_check_fired_steps"),
            "score_consistency_violations": telem.get("score_consistency_violation_heads_total"),
            "h2d_key_bytes": telem.get("h2d_key_bytes_total"),
            "h2d_value_bytes": telem.get("h2d_value_bytes_total"),
            "vram_fp16_value_cache_bytes": telem.get("vram_fp16_value_cache_bytes_max"),
        }

    if bench == "niah":
        rows = (native.get("results") or {}).get("certified") or []
    elif bench == "ruler":
        rows = [r.get("cert_stats") or {} for r in native.get("results") or []]
    else:
        rows = []
    sums = _sum_keys(rows, [
        "rung1_fired", "rung2_fired", "rung3_fired", "rung4_fired",
        "boundary_check_fired", "score_consistency_violation_heads_total",
    ])
    return {
        "int8_tail_fraction": None,
        "rung_triggers": {
            "rung1": sums["rung1_fired"],
            "rung2": sums["rung2_fired"],
            "rung3": sums["rung3_fired"],
            "rung4": sums["rung4_fired"],
        },
        "e_key_max": _max_key(rows, "e_key_step_max"),
        "e_val_max": _max_key(rows, "e_val_max"),
        "boundary_check_triggers": sums["boundary_check_fired"],
        "score_consistency_violations": sums["score_consistency_violation_heads_total"],
    }


def _results_summary(bench: str, native: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(native, dict):
        return {"_missing": True}
    if bench == "pg19":
        dense = native.get("dense", {})
        cert = native.get("certified", {})
        return {
            "dense_ppl": dense.get("perplexity"),
            "certified_ppl": cert.get("perplexity"),
            "delta_ppl": native.get("delta"),
            "ratio": native.get("ratio"),
            "paired_delta_stats": cert.get("paired_delta_stats"),
        }
    if bench == "niah":
        return {
            "dense_accuracy": native.get("dense_accuracy"),
            "certified_accuracy": native.get("certified_accuracy"),
            "delta": (
                native.get("certified_accuracy") - native.get("dense_accuracy")
                if native.get("dense_accuracy") is not None and native.get("certified_accuracy") is not None
                else None
            ),
            "paired_stats": native.get("paired_stats"),
            "paired_stats_by_context": native.get("paired_stats_by_context"),
            "paired_stats_by_needle_group": native.get("paired_stats_by_needle_group"),
            "critical_failures": native.get("critical_failures"),
        }
    if bench == "ruler":
        return {
            "dense_accuracy": native.get("overall_dense"),
            "certified_accuracy": native.get("overall_cert"),
            "delta": (
                native.get("overall_cert") - native.get("overall_dense")
                if native.get("overall_dense") is not None and native.get("overall_cert") is not None
                else None
            ),
            "paired_stats": native.get("paired_stats"),
            "paired_stats_by_context": native.get("paired_stats_by_context"),
            "paired_stats_by_task_context": native.get("paired_stats_by_task_context"),
            "critical_failures": native.get("critical_failures"),
        }
    return {}


def commit_and_push(paths: list[Path], message: str) -> None:
    if current_branch() != BRANCH:
        raise RuntimeError(f"Refusing to push from branch {current_branch()!r}; expected {BRANCH!r}")
    for path in paths:
        _git("add", str(path.relative_to(REPO)))
    rc, out = _git("commit", "-m", message)
    if rc != 0 and "nothing to commit" in out.lower():
        return
    if rc != 0:
        raise RuntimeError(out)
    rc, out = _git("push", "origin", BRANCH)
    if rc != 0:
        raise RuntimeError(out)


def run_cell(cell: dict[str, Any], *, smoke: bool, dry_run: bool) -> dict[str, Any]:
    bench = str(cell["bench"])
    ctx = int(cell["ctx"])
    ctx_k = ctx // 1024
    suffix = f"{cell['idx']}_{bench}_{ctx_k}K"
    cell_json = OUT_DIR / f"{suffix}.json"
    cell_log = OUT_DIR / f"{suffix}.log"
    native_json = OUT_DIR / f"{suffix}.native.json"
    cli = _cli_for_cell(cell, native_json, smoke=smoke)
    plan = {
        "idx": cell["idx"],
        "tier": cell["tier"],
        "benchmark": bench,
        "context_length": ctx,
        "eta_hours": estimate_cell_hours(cell),
        "cmd": " ".join(shlex.quote(c) for c in cli),
        "out": str(cell_json),
    }
    if dry_run:
        return plan

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    t0 = time.perf_counter()
    print(f"\n[{cell['idx']}] tier={cell['tier']} {bench} {ctx_k}K starting")
    print(f"  cmd: {plan['cmd']}")
    print(f"  log: {cell_log}")
    with cell_log.open("w") as log_f:
        log_f.write(f"cmd: {plan['cmd']}\nstarted: {started}\n---\n")
        log_f.flush()
        rc = subprocess.call(cli, stdout=log_f, stderr=subprocess.STDOUT, cwd=REPO, env=_runner_env())
    wall = time.perf_counter() - t0
    ended = dt.datetime.now(dt.timezone.utc).isoformat()
    native = _load_json(native_json)
    wrapped = {
        "benchmark": bench,
        "context_length": ctx,
        "config": _paper_config(native, context_length=ctx),
        "model": MODEL,
        "model_quant": "int8-bitsandbytes",
        "hardware": _hw_tag(),
        "results": _results_summary(bench, native),
        "telemetry": _native_telemetry(bench, native),
        "native": native,
        "meta": {
            "experiment_spec": "v2_clean_rerun",
            "tier": int(cell["tier"]),
            "smoke": bool(smoke),
            "timestamp": ended,
            "started": started,
            "wall_seconds": wall,
            "eta_hours": estimate_cell_hours(cell),
            "git_sha": git_sha(),
            "branch": current_branch(),
            "exit_code": rc,
        },
    }
    cell_json.write_text(json.dumps(wrapped, indent=2))
    print(f"[{cell['idx']}] {bench} {ctx_k}K exit={rc} {wall/60:.1f} min")
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--tier", choices=["1", "2", "3", "4", "all"], default="2",
                        help="Default '2' runs Tier 1 + Tier 2.")
    parser.add_argument("--only", default=None,
                        help="Comma filter, e.g. 'pg19,65536' or '05' or 'ruler'.")
    parser.add_argument("--from", dest="start_from", default=None,
                        help="Resume from cell index, inclusive.")
    parser.add_argument("--machines", type=int, default=1,
                        help="Print a greedy ETA split across this many machines/GPUs.")
    args = parser.parse_args()

    include_tier2 = args.tier in {"2", "3", "4", "all"}
    include_tier3 = args.tier in {"3", "4", "all"}
    include_tier4 = args.tier in {"4", "all"}
    cells = build_cells(
        include_tier2=include_tier2,
        include_tier3=include_tier3,
        include_tier4=include_tier4,
    )
    max_tier = 99 if args.tier == "all" else int(args.tier)
    cells = [c for c in cells if int(c["tier"]) <= max_tier]

    if args.only:
        toks = [t.strip().lower() for t in args.only.split(",") if t.strip()]

        def match(c: dict[str, Any]) -> bool:
            bag = {str(c["idx"]).lower(), str(c["bench"]).lower(), str(c["ctx"]).lower(), f"{c['ctx']//1024}k"}
            return all(t in bag or any(t in v for v in bag) for t in toks)

        cells = [c for c in cells if match(c)]
    if args.start_from:
        cells = [c for c in cells if int(c["idx"]) >= int(args.start_from)]

    selected_quality_hours = sum(estimate_cell_hours(c) for c in cells)
    max_tier_for_estimate = 4 if args.tier == "all" else int(args.tier)
    add_on_hours = estimate_non_cell_hours(max_tier_for_estimate)
    print(f"Plan: {len(cells)} cell(s), tier<={args.tier}, smoke={args.smoke}")
    for cell in cells:
        detail = ""
        if "chunks" in cell:
            detail = f"chunks={cell['chunks']}"
        elif "needles" in cell:
            detail = f"needles={cell['needles']}"
        elif "samples" in cell:
            detail = f"samples={cell['samples']}"
        print(
            f"  {cell['idx']}  T{cell['tier']} {cell['bench']:<5} "
            f"{cell['ctx']//1024:>3}K  {detail:<12} eta={estimate_cell_hours(cell):>4.1f}h"
        )

    print(f"Quality-cell ETA, one GPU: {format_hours(selected_quality_hours)}")
    if add_on_hours:
        print(
            f"Spec add-on ETA through tier {args.tier}: +{format_hours(add_on_hours)} "
            "(performance/ablations not run by this quality runner)"
        )
        print(f"Recommended-plan ETA incl. add-ons: {format_hours(selected_quality_hours + add_on_hours)}")
    if int(args.machines) > 1 and cells:
        slots = schedule_cells(cells, int(args.machines))
        print(f"Greedy quality-cell split across {int(args.machines)} machines:")
        for slot in slots:
            labels = ", ".join(
                f"{c['idx']}:{c['bench']}{c['ctx']//1024}K" for c in slot["cells"]
            )
            print(f"  machine {slot['machine']}: {format_hours(float(slot['hours']))}  {labels}")
        print(
            f"Greedy quality-cell wall ETA: "
            f"{format_hours(max(float(s['hours']) for s in slots))}"
        )

    if args.dry_run:
        for cell in cells:
            print(run_cell(cell, smoke=args.smoke, dry_run=True)["cmd"])
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        run_cell(cell, smoke=args.smoke, dry_run=False)
        if not args.no_push:
            commit_and_push([OUT_DIR], f"bench: paper_v2 cell {cell['idx']} {cell['bench']} {cell['ctx']//1024}K")
        else:
            print("[no-push] skipped commit/push for completed cell")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
