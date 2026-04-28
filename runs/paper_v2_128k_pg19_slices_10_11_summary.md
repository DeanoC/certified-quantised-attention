# Paper v2 128K PG-19 Slice Results: 10-11

Generated on branch `port-to-paper-20260424` with the paper-quality
`full-bounded` cache mode.

## Slice Summary

| Slice | Source dir | Wall | Dense ppl | Certified ppl | Ratio | Delta |
|---:|---|---:|---:|---:|---:|---:|
| 10 | `runs/paper_v2_distributed_128k_pg19_slices_10_11` | 373.1m | 6.9788 | 6.9798 | 1.000141 | +0.0010 |
| 11 | `runs/paper_v2_distributed_128k_pg19_slices_10_11` | 402.2m | 8.8397 | 8.8440 | 1.000484 | +0.0043 |

## Aggregate Checks

- PG-19 ratio mean for slices 10-11: `1.000313`.
- PG-19 delta mean for slices 10-11: `+0.00263` ppl.
- Average slice time: `387.7m`.
- Total machine time: `775.3m`.

## Notes

- Both completed benchmark jobs in slice manifests have `exit_code: 0`.
- Context: `131072`.
- FP16 key/value cache blocks: `9216` / `9216`.
- Source directory: `runs/paper_v2_distributed_128k_pg19_slices_10_11`.
