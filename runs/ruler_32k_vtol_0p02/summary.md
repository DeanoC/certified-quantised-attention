# RULER 32K v_tol=0.02 Sweep

Generated on branch `port-to-paper-20260424` with RULER only.

## Run

- Context: `32768`
- Samples: `0-4`
- Sample count: `5`
- Output: `runs/ruler_32k_vtol_0p02/ruler_32k_vtol_0p02_samples_0_4.json`
- Log: `runs/ruler_32k_vtol_0p02/ruler_32k_vtol_0p02_samples_0_4.log`
- v_tolerance: `0.02`
- FP16 key cache blocks: `3072`
- FP16 value cache blocks: `3072`

## Results

| Task | Dense | Certified | Delta | Critical |
|---|---:|---:|---:|---:|
| cwe_32K | 0.800 | 0.800 | +0.000 | 0 |
| fwe_32K | 0.467 | 0.467 | +0.000 | 0 |
| niah_multikey_32K | 1.000 | 1.000 | +0.000 | 0 |
| niah_multiquery_32K | 1.000 | 1.000 | +0.000 | 0 |
| niah_multivalue_32K | 1.000 | 1.000 | +0.000 | 0 |
| niah_single_32K | 1.000 | 1.000 | +0.000 | 0 |
| vt_32K | 0.950 | 0.950 | +0.000 | 0 |

## Aggregate

- Overall dense: `0.888095`
- Overall certified: `0.888095`
- Delta: `+0.000000`
- Critical failures: `0/35`
- Paired delta CI: `[+0.000, +0.000]`
- Wall time: `19.5m`

## Validation

- Output JSON parsed successfully.
- Result rows: `35` (`7` subtasks x `5` samples).
- Sample indexes present: `0, 1, 2, 3, 4`.
- No critical rows.
