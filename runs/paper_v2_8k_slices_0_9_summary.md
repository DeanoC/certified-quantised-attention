# Paper v2 8K Slice Results: 0-9

Generated on branch `port-to-paper-20260424` with the paper-quality
`full-bounded` cache mode.

## Slice Summary

| Slice | Source dir | Total wall | PG-19 dense ppl | PG-19 cert ppl | Ratio | Delta | NIAH dense/cert | NIAH critical | RULER critical |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `runs/paper_v2_distributed_8k_slices_0_9` | 56.4m | 6.4048 | 6.4122 | 1.001148 | +0.0074 | 5/6 | 0 | 2 |
| 1 | `runs/paper_v2_distributed_8k_slices_0_9` | 22.2m | 6.7636 | 6.7545 | 0.998657 | -0.0091 | 5/3 | 3 | 2 |
| 2 | `runs/paper_v2_distributed_8k_slices_0_9` | 22.3m | 6.7576 | 6.7627 | 1.000753 | +0.0051 | 3/1 | 3 | 1 |
| 3 | `runs/paper_v2_distributed_8k_slices_0_9` | 22.1m | 6.8740 | 6.8782 | 1.000615 | +0.0042 | 3/5 | 0 | 2 |
| 4 | `runs/paper_v2_distributed_8k_slices_0_9` | 22.1m | 6.4482 | 6.4561 | 1.001215 | +0.0078 | 3/3 | 1 | 2 |
| 5 | `runs/paper_v2_distributed_8k_slices_0_9` | 22.0m | 7.6310 | 7.6236 | 0.999039 | -0.0073 | 4/2 | 3 | 0 |
| 6 | `runs/paper_v2_distributed_8k_slices_0_9` | 22.0m | 17.4572 | 17.4143 | 0.997542 | -0.0429 | 4/4 | 3 | 0 |
| 7 | `runs/paper_v2_distributed_8k_slices_0_9` | 22.4m | 18.5206 | 18.5000 | 0.998886 | -0.0206 | 4/5 | 1 | 1 |
| 8 | `runs/paper_v2_distributed_8k_slices_0_9` | 22.1m | 18.4808 | 18.5047 | 1.001293 | +0.0239 | 4/6 | 1 | 0 |
| 9 | `runs/paper_v2_distributed_8k_slices_0_9` | 22.4m | 17.9219 | 17.9627 | 1.002274 | +0.0407 | 4/4 | 1 | 0 |

## Aggregate Checks

- PG-19 ratio mean for slices 0-9: `1.000142`.
- PG-19 ratio range for slices 0-9: `0.997542` to `1.002274`.
- PG-19 delta mean for slices 0-9: `+0.00092` ppl.
- NIAH trials per slice: `10`.
- NIAH critical failures for slices 0-9: `16`.
- RULER critical failures for slices 0-9: `10`.
- Average full slice time: `25.6m`.
- Total machine time: `255.9m`.

## Notes

- All completed benchmark jobs in slice manifests have `exit_code: 0`.
- Source directory: `runs/paper_v2_distributed_8k_slices_0_9`.
