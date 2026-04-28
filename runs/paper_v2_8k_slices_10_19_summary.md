# Paper v2 8K Slice Results: 10-19

Generated on branch `port-to-paper-20260424` with the paper-quality
`full-bounded` cache mode.

NIAH was not run for this batch because `--niah-trials-per-slice=10` exhausts
the 100-trial NIAH grid across slices 0-9.

## Slice Summary

| Slice | Source dir | Total wall | PG-19 dense ppl | PG-19 cert ppl | Ratio | Delta | NIAH dense/cert | NIAH critical | RULER critical |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | `runs/paper_v2_distributed_8k_slices_10_19` | 18.6m | 16.2802 | 16.2736 | 0.999597 | -0.0066 | n/a | n/a | 1 |
| 11 | `runs/paper_v2_distributed_8k_slices_10_19` | 18.4m | 18.3654 | 18.3486 | 0.999085 | -0.0168 | n/a | n/a | 0 |
| 12 | `runs/paper_v2_distributed_8k_slices_10_19` | 18.6m | 17.4568 | 17.4635 | 1.000386 | +0.0067 | n/a | n/a | 1 |
| 13 | `runs/paper_v2_distributed_8k_slices_10_19` | 18.6m | 14.3763 | 14.3546 | 0.998494 | -0.0217 | n/a | n/a | 0 |
| 14 | `runs/paper_v2_distributed_8k_slices_10_19` | 18.4m | 15.7071 | 15.7338 | 1.001702 | +0.0267 | n/a | n/a | 1 |
| 15 | `runs/paper_v2_distributed_8k_slices_10_19` | 18.3m | 10.6774 | 10.6575 | 0.998135 | -0.0199 | n/a | n/a | 2 |
| 16 | `runs/paper_v2_distributed_8k_slices_10_19` | 18.4m | 9.8961 | 9.8867 | 0.999049 | -0.0094 | n/a | n/a | 1 |
| 17 | `runs/paper_v2_distributed_8k_slices_10_19` | 18.5m | 9.8752 | 9.8778 | 1.000264 | +0.0026 | n/a | n/a | 2 |
| 18 | `runs/paper_v2_distributed_8k_slices_10_19` | 18.6m | 9.6539 | 9.6524 | 0.999844 | -0.0015 | n/a | n/a | 0 |
| 19 | `runs/paper_v2_distributed_8k_slices_10_19` | 18.4m | 10.7450 | 10.7401 | 0.999541 | -0.0049 | n/a | n/a | 1 |

## Aggregate Checks

- PG-19 ratio mean for slices 10-19: `0.999610`.
- PG-19 ratio range for slices 10-19: `0.998135` to `1.001702`.
- PG-19 delta mean for slices 10-19: `-0.00447` ppl.
- NIAH critical failures for slices 10-19: `n/a`.
- RULER critical failures for slices 10-19: `9`.
- Average full slice time: `18.5m`.
- Total machine time: `185.0m`.

## Notes

- All completed benchmark jobs in slice manifests have `exit_code: 0`.
- Source directory: `runs/paper_v2_distributed_8k_slices_10_19`.
