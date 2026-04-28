"""Run one distributed paper-quality slice.

The slice id is interpreted consistently across benchmarks:
  * PG-19: global held-out chunk index.
  * NIAH: paired trial index in depth-major (depth, needle) order.
  * RULER: deterministic sample index.

Modes:
  * context: run selected benchmarks for one context at the slice id.
  * line: run selected benchmarks for every requested context at the slice id.
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
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "benchmarks" / "paper"
sys.path.insert(0, str(REPO / "benchmarks"))

from run_experiment_v2_sweep import MODEL, _common_cert_args, recommended_fp16_cache_blocks  # noqa: E402

DEFAULT_CONTEXTS = [8192, 32768, 65536, 131072]
DEFAULT_BENCHES = ["pg19", "niah", "ruler"]
NIAH_DEPTHS = [i / 10 for i in range(10)]
NIAH_NEEDLES = 10


def _context_blocks(context: int, *, block_size: int = 16) -> int:
    return (int(context) + block_size - 1) // block_size


def _cert_args(context: int, *, cache_mode: str) -> list[str]:
    args = _common_cert_args(context)
    if cache_mode == "v2-bounded":
        return args

    if cache_mode == "full-bounded":
        # Covers the full active context plus decode growth while still being
        # represented as bounded scratch/cache rather than an untracked mirror.
        blocks = _context_blocks(context) + 1024
        out: list[str] = []
        skip_next = False
        for idx, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if arg in {"--fp16-key-cache-blocks", "--fp16-value-cache-blocks"}:
                out.extend([arg, str(blocks)])
                skip_next = True
            else:
                out.append(arg)
        return out

    if cache_mode == "full-mirror":
        out = []
        skip_next = False
        for arg in args:
            if skip_next:
                skip_next = False
                continue
            if arg in {"--fp16-key-cache-blocks", "--fp16-value-cache-blocks"}:
                out.extend([arg, "full"])
                skip_next = True
            else:
                out.append(arg)
        return out

    raise ValueError(f"unknown cache mode: {cache_mode}")


def build_jobs(
    *,
    slice_id: int,
    mode: str,
    contexts: list[int],
    benches: list[str],
    output_dir: Path,
    cache_mode: str,
    niah_trials_per_slice: int = 1,
) -> list[dict[str, Any]]:
    if mode == "context" and len(contexts) != 1:
        raise ValueError("--mode context requires exactly one --context/--contexts value")
    if niah_trials_per_slice < 1:
        raise ValueError("--niah-trials-per-slice must be >= 1")
    if "niah" in benches:
        niah_trials = len(NIAH_DEPTHS) * NIAH_NEEDLES
        niah_trial_start = slice_id * niah_trials_per_slice
        if slice_id < 0 or niah_trial_start >= niah_trials:
            raise ValueError(
                f"NIAH slice id must map into [0, {niah_trials - 1}] for "
                f"{len(NIAH_DEPTHS)} depths x {NIAH_NEEDLES} needles with "
                f"--niah-trials-per-slice={niah_trials_per_slice}"
            )
    selected_contexts = contexts if mode == "line" else [contexts[0]]

    jobs: list[dict[str, Any]] = []
    for context in selected_contexts:
        ctx_k = int(context) // 1024
        ctx_dir = output_dir / f"slice_{slice_id:04d}" / f"{ctx_k}K"
        for bench in benches:
            out_json = ctx_dir / f"{bench}_slice_{slice_id:04d}_{ctx_k}K.json"
            log_path = ctx_dir / f"{bench}_slice_{slice_id:04d}_{ctx_k}K.log"
            common = _cert_args(context, cache_mode=cache_mode)
            if bench == "pg19":
                cmd = [
                    sys.executable, str(PAPER / "pg19_perplexity.py"),
                    "--context", str(context),
                    "--chunk-index", str(slice_id),
                    "--telemetry-mode", "summary",
                    "--certified-warmup-steps", "128",
                    "--output", str(out_json),
                    *common,
                ]
            elif bench == "niah":
                niah_trial_start = slice_id * niah_trials_per_slice
                cmd = [
                    sys.executable, str(PAPER / "niah.py"),
                    "--contexts", str(context),
                    "--needles", str(NIAH_NEEDLES),
                    "--trial-start", str(niah_trial_start),
                    "--trial-count", str(niah_trials_per_slice),
                    "--output", str(out_json),
                    *common,
                ]
            elif bench == "ruler":
                cmd = [
                    sys.executable, str(PAPER / "ruler.py"),
                    "--contexts", str(context),
                    "--sample-index", str(slice_id),
                    "--output", str(out_json),
                    *common,
                ]
            else:
                raise ValueError(f"unknown benchmark: {bench}")

            jobs.append({
                "slice_id": int(slice_id),
                "benchmark": bench,
                "context_length": int(context),
                "output": str(out_json),
                "log": str(log_path),
                "command": cmd,
                "command_string": " ".join(shlex.quote(c) for c in cmd),
                "cache_mode": cache_mode,
                "recommended_v2_cache_blocks": recommended_fp16_cache_blocks(context),
            })
    return jobs


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _manifest_provenance() -> dict[str, Any]:
    return {
        "host": socket.gethostname(),
        "cwd": str(REPO),
        "python": sys.executable,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
    }


def _looks_complete_json(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        return False


def run_jobs(jobs: list[dict[str, Any]], *, fail_fast: bool, resume: bool) -> list[dict[str, Any]]:
    completed = []
    for job in jobs:
        out = Path(job["output"])
        log = Path(job["log"])
        out.parent.mkdir(parents=True, exist_ok=True)
        if resume and _looks_complete_json(out):
            record = {
                **{k: v for k, v in job.items() if k != "command"},
                "started": None,
                "ended": dt.datetime.now(dt.timezone.utc).isoformat(),
                "wall_seconds": 0.0,
                "exit_code": 0,
                "skipped_existing": True,
            }
            completed.append(record)
            print(f"[skip] {job['benchmark']} {job['context_length']//1024}K existing={out}")
            continue
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        t0 = time.perf_counter()
        print(f"[start] {job['benchmark']} {job['context_length']//1024}K slice={job['slice_id']}")
        print(f"  {job['command_string']}")
        with log.open("w", encoding="utf-8") as f:
            f.write(f"started: {started}\ncmd: {job['command_string']}\n---\n")
            f.flush()
            rc = subprocess.call(job["command"], cwd=REPO, env=_env(), stdout=f, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - t0
        record = {
            **{k: v for k, v in job.items() if k != "command"},
            "started": started,
            "ended": dt.datetime.now(dt.timezone.utc).isoformat(),
            "wall_seconds": elapsed,
            "exit_code": int(rc),
        }
        completed.append(record)
        print(f"[done] {job['benchmark']} {job['context_length']//1024}K exit={rc} wall={elapsed/60:.1f}m")
        if rc != 0 and fail_fast:
            break
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-id", type=int, required=True,
                        help="Distributed shard id: PG-19 chunk, NIAH trial, RULER sample.")
    parser.add_argument("--mode", choices=["context", "line"], default="context")
    parser.add_argument("--context", type=int, default=None,
                        help="Single context for --mode context.")
    parser.add_argument("--contexts", type=int, nargs="+", default=None,
                        help="Contexts for --mode line, or a single context for --mode context.")
    parser.add_argument("--benches", nargs="+", choices=DEFAULT_BENCHES, default=DEFAULT_BENCHES)
    parser.add_argument("--cache-mode", choices=["full-bounded", "v2-bounded", "full-mirror"],
                        default="full-bounded")
    parser.add_argument("--niah-trials-per-slice", type=int, default=1,
                        help="Number of paired NIAH trials to run for each distributed slice.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/paper_v2_distributed"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Skip jobs whose output JSON already exists and parses.")
    args = parser.parse_args()

    if args.context is not None and args.contexts is not None:
        raise SystemExit("Use either --context or --contexts, not both.")
    contexts = args.contexts if args.contexts is not None else ([args.context] if args.context else DEFAULT_CONTEXTS)
    if args.mode == "context" and len(contexts) != 1:
        raise SystemExit("--mode context requires --context CTX or --contexts CTX.")

    try:
        jobs = build_jobs(
            slice_id=args.slice_id,
            mode=args.mode,
            contexts=[int(c) for c in contexts],
            benches=list(args.benches),
            output_dir=args.output_dir,
            cache_mode=args.cache_mode,
            niah_trials_per_slice=args.niah_trials_per_slice,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    manifest_path = args.output_dir / f"slice_{args.slice_id:04d}" / "manifest.json"
    manifest = {
        "slice_id": int(args.slice_id),
        "mode": args.mode,
        "model": MODEL,
        "cache_mode": args.cache_mode,
        "niah_trials_per_slice": int(args.niah_trials_per_slice),
        "created": dt.datetime.now(dt.timezone.utc).isoformat(),
        "provenance": _manifest_provenance(),
        "jobs": [{k: v for k, v in j.items() if k != "command"} for j in jobs],
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        for job in jobs:
            print(job["command_string"])
        return 0

    completed = run_jobs(jobs, fail_fast=not args.keep_going, resume=args.resume)
    manifest["completed"] = completed
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest -> {manifest_path}")
    return 0 if all(j["exit_code"] == 0 for j in completed) and len(completed) == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
