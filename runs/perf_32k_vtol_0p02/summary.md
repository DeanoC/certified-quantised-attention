# 32K v_tol=0.02 latency characterization

- Model: `NousResearch/Meta-Llama-3.1-8B`
- Context: `32768`
- Timed decode steps: `500`
- Cache mode: `capped-2048` with `2048` key / `2048` value FP16 scratch blocks
- v_tolerance: `0.02`

| Metric | Value |
| --- | ---: |
| Dense ms/step | 62.937 |
| Certified ms/step | 252.684 |
| Cert/dense ratio | 4.015x |
| K* mean | 217.292 |
| K* max | 256 |
| Cache hit | 99.740% |
| H2D MB/step | 434.697 |
| Rung 2 trigger rate | 100.000% |

Raw JSON: `runs/perf_32k_vtol_0p02/raw/context_32768_vtol_0p02_cap2048.json`
Log: `runs/perf_32k_vtol_0p02/raw/context_32768_vtol_0p02_cap2048.log`
