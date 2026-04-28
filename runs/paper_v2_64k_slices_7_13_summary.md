# Paper v2 64K Slice Results: 7-13

Generated on branch `port-to-paper-20260424` with the paper-quality
`full-bounded` cache mode.

## Slice Summary

| Slice | Source dir | Total wall | PG-19 dense ppl | PG-19 cert ppl | Ratio | Delta | NIAH dense/cert | NIAH critical | RULER critical |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | `runs/paper_v2_distributed_64k_machineA` | 146.1m | 8.6120 | 8.6214 | 1.001089 | +0.0094 | 1/1 | 0 | 0 |
| 8 | `runs/paper_v2_distributed_64k_machineA` | 138.5m | 7.9224 | 7.9221 | 0.999962 | -0.0003 | 1/1 | 0 | 0 |
| 9 | `runs/paper_v2_distributed_64k_machineA` | 139.3m | 6.7304 | 6.7317 | 1.000200 | +0.0013 | 0/0 | 0 | 0 |
| 10 | `runs/paper_v2_distributed_64k_machineA` | 138.3m | 8.2153 | 8.2109 | 0.999471 | -0.0043 | 1/0 | 1 | 0 |
| 11 | `runs/paper_v2_distributed_64k_machineA` | 139.6m | 8.9826 | 8.9770 | 0.999376 | -0.0056 | 1/1 | 0 | 0 |
| 12 | `runs/paper_v2_distributed_64k_machineA` | 147.6m | 7.9732 | 7.9779 | 1.000587 | +0.0047 | 0/0 | 0 | 0 |
| 13 | `runs/paper_v2_distributed_64k_machineA` | 138.1m | 7.9302 | 7.9339 | 1.000472 | +0.0037 | 0/0 | 0 | 0 |

## Aggregate Checks

- PG-19 ratio mean for slices 7-13: `1.000165`.
- PG-19 ratio range for slices 7-13: `0.999376` to `1.001089`.
- PG-19 delta mean for slices 7-13: `+0.00127` ppl.
- NIAH critical failures for slices 7-13: `1`.
- RULER critical failures for slices 7-13: `0`.
- Average full slice time: `141.1m`.

## Notes

- All completed benchmark jobs in slice manifests have `exit_code: 0`.
- Source directory: `runs/paper_v2_distributed_64k_machineA`.
