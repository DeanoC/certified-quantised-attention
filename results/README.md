# Results

JSON files that back the paper tables live here.

Top-level files:

- `pg19_ctx4096.json`
- `pg19_ctx8192.json`
- `pg19_ctx16384.json`
- `pg19_ctx32768.json`
- `ruler_ctx4096.json`
- `ruler_ctx8192.json`
- `throughput_context_scaling.json`

Subdirectories:

- `raw_arxiv_v1/`: dense/certified raw sweep outputs for PG-19, NIAH, and
  RULER at 4K, 8K, 16K, and 32K.
- `niah_8k_tau_sweep/`: 100-trial 8K tau sweep results.
- `performance/`: cache sweep, phase breakdown, and 64K performance JSONs.
