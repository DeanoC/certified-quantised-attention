"""Per-step page-in telemetry collector for the arXiv v1 paper harnesses.

Usage inside a decode loop:

    tc = PageinTelemetry(adapter, enabled=args.pagein_telemetry,
                         host_sample_every=100)
    tc.start()
    for _ in range(max_new_tokens):
        out = model(...)
        tc.record_step()
    tc.finish()
    tc.write_json(output_path)

When `enabled=False` every method is a no-op, so the kernel-side stats
overhead is avoided entirely in untagged runs.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class PageinTelemetry:
    def __init__(
        self,
        adapter,
        enabled: bool = False,
        host_sample_every: int = 100,
    ):
        self.adapter = adapter
        self.enabled = bool(enabled)
        self.host_sample_every = int(host_sample_every)
        self.per_step: list[dict] = []
        self.host_samples: list[dict] = []
        self._step_idx = 0
        self._t0 = time.perf_counter()
        self._started = False
        # Index into certified_state.step_stats of the first entry belonging
        # to the NEXT decode step — updated after each record_step. Lets us
        # slice per-step without clearing, so the harness's cell-level
        # aggregate (e.g. ranking_fallback_summary) still sees the data.
        self._step_stats_cursor = 0
        # Last-seen clear_seq on the CertifiedAttentionState. When the host
        # harness drains+clears step_stats between our record_step calls
        # (pg19_perplexity.py does this), the cursor becomes stale; we detect
        # the change here and reset before slicing.
        self._last_clear_seq = 0

    def start(self):
        if not self.enabled:
            return
        self._started = True
        self._t0 = time.perf_counter()
        self._sample_host()

    def record_step(self):
        if not (self.enabled and self._started):
            return
        cs = getattr(self.adapter, "certified_state", None)
        if cs is None:
            return
        try:
            # Aggregate only the layer entries appended since the previous
            # record_step. Do NOT clear — other callers (e.g. niah.py's
            # end-of-cell ranking_fallback_summary) still need the full
            # accumulator at cell boundaries.
            #
            # Three cursor-invalidation cases to handle:
            #   (a) step_stats shrank below cursor → reset to 0 (list cleared
            #       and not yet refilled by a model forward).
            #   (b) _clear_seq advanced since last call → the host harness
            #       (pg19_perplexity.py) drained+cleared step_stats between
            #       our calls; the refill is new data and cursor at len is
            #       stale. Reset cursor to 0 so we capture the refill.
            #   (c) Otherwise (cursor <= len, seq unchanged): normal
            #       slice-from-cursor operation.
            current_seq = int(getattr(cs, "_clear_seq", 0))
            if self._step_stats_cursor > len(cs.step_stats):
                self._step_stats_cursor = 0
            elif current_seq != self._last_clear_seq:
                self._step_stats_cursor = 0
            self._last_clear_seq = current_seq
            agg = cs.aggregate_step_stats(since=self._step_stats_cursor)
            self._step_stats_cursor = len(cs.step_stats)
        except Exception:
            agg = {}
        if agg:
            agg["_step"] = self._step_idx
            agg["_elapsed_s"] = time.perf_counter() - self._t0
            self.per_step.append(agg)
        self._step_idx += 1
        if self.host_sample_every > 0 and self._step_idx % self.host_sample_every == 0:
            self._sample_host()

    def reset_cursor(self):
        """Call at the boundary between benchmark cells if the harness
        clears step_stats between cells (e.g. between niah needles). Keeps
        the per-step slicing aligned with the freshly-reset accumulator.
        """
        self._step_stats_cursor = 0

    def _sample_host(self):
        info = {"_step": self._step_idx, "_elapsed_s": time.perf_counter() - self._t0}
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        info["rss_kb"] = int(line.split()[1])
                        break
        except Exception:
            pass
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("Cached:"):
                        info["meminfo_cached_kb"] = int(line.split()[1])
                        break
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu_mem_allocated_mb"] = float(torch.cuda.memory_allocated() / (1024 ** 2))
                info["gpu_mem_reserved_mb"] = float(torch.cuda.memory_reserved() / (1024 ** 2))
                info["gpu_mem_peak_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        except Exception:
            pass
        self.host_samples.append(info)

    def finish(self):
        if not (self.enabled and self._started):
            return
        self._sample_host()

    def summary(self) -> dict:
        if not self.enabled or not self.per_step:
            return {"enabled": self.enabled, "n_steps": 0}
        n = len(self.per_step)

        def pct(key, p):
            vals = sorted(s.get(key, 0) for s in self.per_step)
            return vals[min(n - 1, int(n * p))]

        h2d_totals = [s.get("h2d_total_bytes", 0) for s in self.per_step]
        h2d_keys = [s.get("h2d_key_bytes", 0) for s in self.per_step]
        h2d_values = [s.get("h2d_value_bytes", 0) for s in self.per_step]
        k_stars = [s.get("k_star_mean") for s in self.per_step if s.get("k_star_mean") is not None]
        rates: dict[str, float] = {}
        for rk in ("rung1_fired", "rung2_fired", "rung3_fired", "rung4_fired"):
            fires = sum(1 for s in self.per_step if s.get(rk))
            rates[rk.replace("_fired", "_rate")] = fires / n
        rss_peak = max((h.get("rss_kb", 0) for h in self.host_samples), default=0)
        cached_peak = max((h.get("meminfo_cached_kb", 0) for h in self.host_samples), default=0)
        cached_min = min((h.get("meminfo_cached_kb", 0) for h in self.host_samples if "meminfo_cached_kb" in h), default=0)
        gpu_mem_peak = max((h.get("gpu_mem_peak_mb", 0.0) for h in self.host_samples), default=0.0)
        # FP16 cache rollup (only present when bounded cache is active).
        cache_hits = sum(s.get("fp16_cache_hits", 0) for s in self.per_step)
        cache_misses = sum(s.get("fp16_cache_misses", 0) for s in self.per_step)
        cache_access = cache_hits + cache_misses
        cache_capacity = int(self.per_step[0].get("fp16_cache_capacity_blocks", 0))
        cache_evictions = sum(s.get("fp16_cache_evictions", 0) for s in self.per_step)
        return {
            "enabled": True,
            "n_steps": n,
            "h2d_total_bytes_sum": int(sum(h2d_totals)),
            "h2d_total_bytes_mean": float(sum(h2d_totals) / n),
            "h2d_total_bytes_p50": int(pct("h2d_total_bytes", 0.50)),
            "h2d_total_bytes_p95": int(pct("h2d_total_bytes", 0.95)),
            "h2d_total_bytes_max": int(max(h2d_totals) if h2d_totals else 0),
            "h2d_key_bytes_sum": int(sum(h2d_keys)),
            "h2d_value_bytes_sum": int(sum(h2d_values)),
            "pct_steps_zero_pagein": sum(1 for b in h2d_totals if b == 0) / n,
            "vram_fp16_key_cache_bytes": int(self.per_step[0].get("vram_fp16_key_cache_bytes", 0)),
            "vram_fp16_value_cache_bytes": int(self.per_step[0].get("vram_fp16_value_cache_bytes", 0)),
            "k_star_mean": float(sum(k_stars) / len(k_stars)) if k_stars else None,
            "k_star_max": int(max((s.get("k_star_max", 0) for s in self.per_step), default=0)),
            **rates,
            "host_rss_peak_kb": rss_peak,
            "meminfo_cached_peak_kb": cached_peak,
            "meminfo_cached_min_kb": cached_min,
            "meminfo_cached_delta_kb": cached_peak - cached_min if cached_min else 0,
            "gpu_mem_peak_mb": gpu_mem_peak,
            "fp16_cache_capacity_blocks": cache_capacity,
            "fp16_cache_total_hits": int(cache_hits),
            "fp16_cache_total_misses": int(cache_misses),
            "fp16_cache_hit_rate": float(cache_hits / cache_access) if cache_access else 0.0,
            "fp16_cache_total_evictions": int(cache_evictions),
            "fp16_cache_avg_misses_per_step": float(cache_misses / n) if n else 0.0,
        }

    def write_json(self, path: str | Path):
        if not self.enabled:
            return
        data = {
            "summary": self.summary(),
            "per_step": self.per_step,
            "host_samples": self.host_samples,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2))
