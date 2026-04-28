# RULER 128K Extra Samples: 10-13

Generated on branch `port-to-paper-20260424` with RULER only.

## Run

- Context: `131072`
- Samples: `10-13`
- Sample count: `4`
- Output: `runs/ruler_128k_extra/ruler_128k_samples_10_13.json`
- Log: `runs/ruler_128k_extra/ruler_128k_samples_10_13.log`
- FP16 key cache blocks: `5120`
- FP16 value cache blocks: `5120`

## Results

| Task | Dense | Certified | Delta | Critical |
|---|---:|---:|---:|---:|
| cwe_128K | 0.750 | 0.750 | +0.000 | 0 |
| fwe_128K | 0.250 | 0.250 | +0.000 | 0 |
| niah_multikey_128K | 1.000 | 0.750 | -0.250 | 1 |
| niah_multiquery_128K | 1.000 | 1.000 | +0.000 | 0 |
| niah_multivalue_128K | 0.875 | 0.938 | +0.062 | 0 |
| niah_single_128K | 1.000 | 1.000 | +0.000 | 0 |
| vt_128K | 0.500 | 0.312 | -0.188 | 1 |

## Aggregate

- Overall dense: `0.767857`
- Overall certified: `0.714286`
- Delta: `-0.053571`
- Critical failures: `2/28`
- Paired delta CI: `[-0.152, +0.018]`
- Wall time: `52.7m`

## Validation

- Output JSON parsed successfully.
- Result rows: `28` (`7` subtasks x `4` samples).
- Sample indexes present: `10, 11, 12, 13`.
- Critical rows:
  - `niah_multikey`, sample `12`, dense `1.0`, certified `0.0`
  - `vt`, sample `13`, dense `1.0`, certified `0.25`
