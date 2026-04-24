# Certified Quantised Attention

Minimal review repository for the paper `Certified_Quantised_Attention.tex`.

This repository is being extracted from the experimental [`DotCache`](https://github.com/DeanoC/DotCache) research repo . 
This repo has only the paper, implementation code, benchmark drivers, and JSON
results required for reviewers to inspect and reproduce the reported claims.

## Layout

```text
paper/
  Certified_Quantised_Attention.tex
src/
  certified_attention.py
  tiered_cache.py
  adaptive_selector.py
  fallback_ladder.py
  quantisation.py
benchmarks/
  run_pg19.py
  run_niah.py
  run_ruler.py
  run_throughput.py
results/
  README.md
```

## Extraction Status

- `paper/`: contains `Certified_Quantised_Attention.tex`. The current file is
  a normalised short TeX placeholder; replace it with the full submission source
  before release.
- `src/`: contains the extracted certified attention implementation and helper
  kernels from `DotCache/dotcache/kernels`.
- `benchmarks/`: contains the extracted PG-19, NIAH, RULER, and throughput
  entrypoints, now rewritten to use local benchmark support modules.
- `results/`: contains the JSON result files currently identified as backing
  the paper tables, plus raw April 2026 sweep outputs for traceability.

## Intended Reviewer Workflow

1. Install dependencies with `pip install -r requirements.txt`.
2. Inspect the implementation in `src/`.
3. Run the relevant benchmark script from `benchmarks/`.
4. Compare generated outputs with the checked-in JSON files in `results/`.

## Current Extraction Boundary

The reviewer-facing implementation is concentrated in:

- `src/certified_attention.py`
- `src/tiered_cache.py`
- `src/adaptive_selector.py`
- `src/fallback_ladder.py`
- `src/quantisation.py`

Additional helper files in `src/` are direct dependencies of the fused Triton
path and are intentionally kept visible:

- `src/fused_score_certify.py`
- `src/score_phase_triton.py`
- `src/selective_attend_triton.py`
- `src/llama_integration.py`
- `src/config.py`
- `src/calibrated_profile.py`
