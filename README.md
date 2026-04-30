# Certified Quantised Attention

Reviewer-facing artifact for the paper in
`paper/runtime_certified_bounded_error_quantised_attention.tex` (PDF alongside).

This repository is a cleaned extraction from the DotCache paper branch. It
keeps the implementation, benchmark drivers, and retained JSON artifacts needed
to inspect or rerun the paper results. DotCache experiments unrelated to the
paper path have been removed. Released under the MIT licence (see `LICENSE`);
the version corresponding to the paper is tagged `arxiv-v1`.

## Layout

```text
paper/
  runtime_certified_bounded_error_quantised_attention.tex
  runtime_certified_bounded_error_quantised_attention.pdf
dotcache/
  integrations/llama.py
  kernels/
  backends/
benchmarks/
  paper/
  run_experiment_v2_sweep.py
runs/
  paper_v2_*/                  # certified system, main quality results
  naive_int8k_int4v_*/         # naive INT8K/INT4V baseline (Table tab:naive)
  perf_single_machine/         # performance and memory telemetry
  niah_64k_remaining/          # NIAH 64K follow-up trials
  ruler_*                      # RULER follow-up ablations
tests/
  paper CLI, provenance, CI helper, and CUDA smoke tests
```

## Reviewer Workflow

### Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=...  # required for gated model access when applicable
```

Full benchmark reproduction requires a CUDA GPU with enough VRAM for
LLaMA 3.1-8B, a CUDA-enabled PyTorch install, and model download access. The
native Blackwell backend additionally requires NVCC/CUDA_HOME and an SM 12.x
GPU; otherwise the benchmark helpers fall back to the Triton backend.

### Quick Validation

```bash
pytest -q
python benchmarks/run_experiment_v2_sweep.py --dry-run --tier 2 --no-push
python benchmarks/run_experiment_v2_sweep.py --smoke --tier 1 --only pg19,8192 --no-push
```

The smoke command loads the model and runs a reduced PG-19 cell. It is intended
to validate the runtime path before launching multi-hour cells.

### Direct Paper Benchmark Defaults

The direct benchmark CLIs default to the paper operating point for the certified
path: `tau_cov=0.995`, `k_min=2`, `k_max=128`, ranking fallback enabled with
`r=1`, `eps_guard=0.01`, `exploration_rate=0.02`, and Rung-1 expansion
`threshold=0.02`, `multiplier=2.0`. Only the cache/value format remains
explicit on the command line so the output provenance records it.

Example single-cell reruns:

```bash
python benchmarks/paper/pg19_perplexity.py \
  --context 8192 --chunk-index 0 \
  --v-tolerance 0.05 --use-int4-values --group-size 16 \
  --output runs/repro_single/pg19_8k_slice0.json

python benchmarks/paper/niah.py \
  --contexts 8192 --trial-start 0 --trial-count 5 --needles 10 \
  --v-tolerance 0.05 --use-int4-values --group-size 16 \
  --output runs/repro_single/niah_8k_trials0_4.json

python benchmarks/paper/ruler.py \
  --contexts 8192 --sample-index 0 \
  --v-tolerance 0.05 --use-int4-values --group-size 16 \
  --output runs/repro_single/ruler_8k_sample0.json
```

Use `--no-ranking-fallback`, `--tau-cov 0`, or alternative `--k-max` values
only for ablations; those are not the paper default.

### Distributed Quality Runs

`benchmarks/paper/run_distributed_quality_slice.py` is the reviewer-facing
driver for the retained quality artifacts. It records the exact command,
cache mode, git revision, and per-cell JSON path in each slice manifest.

```bash
# One representative slice across all three benchmarks at 8K.
python benchmarks/paper/run_distributed_quality_slice.py \
  --slice-id 0 --mode context --context 8192 \
  --benches pg19 niah ruler --niah-trials-per-slice 5 \
  --cache-mode full-bounded --output-dir runs/repro_quality --resume

# PG-19: 20 chunks per context.
for ctx in 8192 32768 65536; do
  for slice in $(seq 0 19); do
    python benchmarks/paper/run_distributed_quality_slice.py \
      --slice-id "$slice" --mode context --context "$ctx" \
      --benches pg19 --cache-mode full-bounded \
      --output-dir runs/repro_quality --resume
  done
done

# NIAH: 100 paired trials per context, packed as 20 slices x 5 trials.
for ctx in 8192 32768 65536; do
  for slice in $(seq 0 19); do
    python benchmarks/paper/run_distributed_quality_slice.py \
      --slice-id "$slice" --mode context --context "$ctx" \
      --benches niah --niah-trials-per-slice 5 \
      --cache-mode full-bounded --output-dir runs/repro_quality --resume
  done
done

# RULER: 50 samples per context.
for ctx in 8192 32768 65536; do
  for slice in $(seq 0 49); do
    python benchmarks/paper/run_distributed_quality_slice.py \
      --slice-id "$slice" --mode context --context "$ctx" \
      --benches ruler --cache-mode full-bounded \
      --output-dir runs/repro_quality --resume
  done
done
```

For selected 128K follow-ups, use `--context 131072` with the same slice
driver and the reduced slice counts described in the paper.

### Performance Runs

Single-machine performance and memory artifacts are produced by
`benchmarks/paper/run_single_machine_perf.py`:

```bash
for part in p1 p2 p3 p4 p5 p6 p7_pg19; do
  python benchmarks/paper/run_single_machine_perf.py "$part" \
    --output-dir runs/repro_perf
done
```

The `r*_cap2048` modes reproduce the capped-cache follow-up artifacts:

```bash
for part in r1_cap2048 r2_cap2048 r3_cap2048 r4_pg19_cap2048 r4_niah_cap2048; do
  python benchmarks/paper/run_single_machine_perf.py "$part" \
    --output-dir runs/repro_perf_cap2048
done
```

### Artifact Checks

Every benchmark JSON embeds a `cache_config` block with the certified operating
point, backend, cache sizes, git SHA, and a deterministic config hash. The
checked-in `runs/` tree is the retained artifact set used for paper review;
new reruns should be written under a separate `runs/repro_*` directory and
compared against the corresponding checked-in JSON/summary files.
