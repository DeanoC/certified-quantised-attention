# Paper v2 64K Slice Results: 0-6

Generated on branch `port-to-paper-20260424` with the paper-quality
`full-bounded` cache mode.

## Slice Summary

| Slice | Source dir | Total wall | PG-19 dense ppl | PG-19 cert ppl | Ratio | Delta | NIAH dense/cert | NIAH critical | RULER critical |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `runs/paper_v2_distributed_context64k_trial` | 103.0m | 15.5685 | 15.5785 | 1.000644 | +0.0100 | 1/1 | 0 | 0 |
| 1 | `runs/paper_v2_distributed_64k_overnight` | 96.7m | 9.5107 | 9.5109 | 1.000021 | +0.0002 | 1/1 | 0 | 1 |
| 2 | `runs/paper_v2_distributed_64k_overnight` | 96.1m | 14.7801 | 14.7709 | 0.999377 | -0.0092 | 0/0 | 0 | 0 |
| 3 | `runs/paper_v2_distributed_64k_overnight` | 95.1m | 6.9563 | 6.9529 | 0.999522 | -0.0033 | 0/0 | 0 | 0 |
| 4 | `runs/paper_v2_distributed_64k_overnight` | 98.0m | 9.4991 | 9.5091 | 1.001045 | +0.0099 | 1/1 | 0 | 0 |
| 5 | `runs/paper_v2_distributed_64k_overnight` | 98.0m | 9.4340 | 9.4337 | 0.999977 | -0.0002 | 1/1 | 0 | 0 |
| 6 | `runs/paper_v2_distributed_64k_overnight` | 101.3m | 7.8747 | 7.8742 | 0.999934 | -0.0005 | 1/1 | 0 | 0 |

## Aggregate Checks

- PG-19 ratio mean for overnight slices 1-6: `0.999979`.
- PG-19 ratio range for overnight slices 1-6: `0.999377` to `1.001045`.
- PG-19 delta mean for overnight slices 1-6: `-0.00052` ppl.
- NIAH critical failures for overnight slices 1-6: `0`.
- RULER critical failures for overnight slices 1-6: `1`.
- Average overnight full slice time: `97.5m`.

## Notes

- Slice 1 has one RULER `vt_64K` dense-pass/certified-fail critical case.
- NIAH slices 2 and 3 are dense-fail/certified-fail, not certified regressions.
- Remaining 64K PG-19 20-chunk CI slices after this batch: `7..19`.

