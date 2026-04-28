"""Aggregate RULER paper-sweep JSONs into a single paper-ready table.

Reads eps0_4k.json, calibrated_4k.json, eps0_8k.json, calibrated_8k.json
from a results directory and emits:

1. Per-subtask accuracy table (dense vs cert) across both regimes × both contexts.
2. Criticals breakdown (samples where dense passed and cert failed).
3. Per-subtask marginal pass-rate 95% CIs (Wilson) for dense and cert.
4. Paired 95% CI on the cert-minus-dense pass-rate difference (Wald on
   McNemar discordant counts) — exploits that every sample is scored
   under both policies, giving a tighter interval than independent rates.
5. Markdown-ready output (stdout) and CSV (--csv path).

Usage:
  python benchmarks/paper/aggregate_ruler.py \
    --dir benchmarks/results/ruler_paper_20260417 \
    --csv benchmarks/results/ruler_paper_20260417/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


CONFIGS = [
    ("eps0_4k", "ε=0 no-skip", 4096),
    ("calibrated_4k", "calibrated", 4096),
    ("eps0_8k", "ε=0 no-skip", 8192),
    ("calibrated_8k", "calibrated", 8192),
]

SUBTASKS = [
    "niah_single", "niah_multikey", "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe",
]

# Score threshold above which a sample is treated as a "pass". Matches
# the per-sample score convention in ruler.py (dense_score == 1.0 means
# full match; 0.999 tolerates fp rounding if scorers ever emit non-exact 1s).
PASS_THRESHOLD = 0.999


def wilson_ci(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score CI for a single Bernoulli rate.

    Returns (lower, upper) for p = k/n. Preferred over Wald because it
    stays inside [0, 1] even for k = 0 or k = n, which we hit on
    easy/impossible-for-cert subtasks.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfwidth = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - halfwidth), min(1.0, centre + halfwidth))


def paired_diff_ci(b: int, c: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Paired 95% CI for proportion difference p_cert − p_dense.

    Given the 2×2 paired table of pass/fail outcomes on the same n
    samples:

        b = # samples where dense PASS, cert FAIL  (critical failures)
        c = # samples where dense FAIL, cert PASS  (cert-only wins)

    the point estimate is (c − b) / n and the large-sample variance is
    Var = (b + c − (b − c)^2 / n) / n^2 (Fleiss et al., §13). Returns
    (point, lower, upper). The (a + d = n − b − c) concordant cells
    don't enter the CI — that's the whole point of the paired design.

    Edge case: if b + c == 0 (dense and cert agree on every sample),
    Var == 0 and we return a point interval. Technically the Wald CI
    is undefined there; Newcombe's method would give a nontrivial
    interval but we keep Wald for simplicity and return (0, 0, 0).
    """
    if n == 0:
        return (0.0, -1.0, 1.0)
    point = (c - b) / n
    discordant = b + c
    if discordant == 0:
        return (point, point, point)
    var = (discordant - (c - b) ** 2 / n) / (n ** 2)
    if var < 0:  # numerical guard near the edge
        var = 0.0
    half = z * math.sqrt(var)
    return (point, max(-1.0, point - half), min(1.0, point + half))


def load_config(results_dir: Path, tag: str) -> dict | None:
    path = results_dir / f"{tag}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def summarise_config(data: dict) -> dict:
    """Aggregate per-subtask. Returns {subtask: {dense, cert, delta, crit, n, ...ci}}.

    Tracks the full 2×2 paired table (a, b, c, d) per subtask so we can
    compute both marginal Wilson CIs and the paired McNemar-style CI on
    the pass-rate difference. Scores are continuous in [0,1] but the
    pass/fail binarisation (via PASS_THRESHOLD) is what RULER's paper
    tables report.
    """
    by_subtask = {
        s: {
            "dense_sum": 0.0, "cert_sum": 0.0, "crit": 0, "n": 0,
            "dense_pass": 0, "cert_pass": 0,
            # Paired 2×2 cells.
            "both_pass": 0,       # a
            "dense_only_pass": 0, # b  (cert missed)
            "cert_only_pass": 0,  # c  (dense missed, cert caught)
            "both_fail": 0,       # d
        }
        for s in SUBTASKS
    }
    for r in data["results"]:
        s = r["subtask"]
        if s not in by_subtask:
            continue
        bkt = by_subtask[s]
        bkt["dense_sum"] += r["dense_score"]
        bkt["cert_sum"] += r["cert_score"]
        bkt["n"] += 1
        if r.get("critical"):
            bkt["crit"] += 1
        dense_pass = r["dense_score"] >= PASS_THRESHOLD
        cert_pass = r["cert_score"] >= PASS_THRESHOLD
        if dense_pass:
            bkt["dense_pass"] += 1
        if cert_pass:
            bkt["cert_pass"] += 1
        if dense_pass and cert_pass:
            bkt["both_pass"] += 1
        elif dense_pass and not cert_pass:
            bkt["dense_only_pass"] += 1
        elif not dense_pass and cert_pass:
            bkt["cert_only_pass"] += 1
        else:
            bkt["both_fail"] += 1

    out = {}
    for s, bkt in by_subtask.items():
        n = bkt["n"]
        if n == 0:
            continue
        dense = bkt["dense_sum"] / n
        cert = bkt["cert_sum"] / n
        dense_pass = bkt["dense_pass"]
        cert_pass = bkt["cert_pass"]
        d_lo, d_hi = wilson_ci(dense_pass, n)
        c_lo, c_hi = wilson_ci(cert_pass, n)
        diff, diff_lo, diff_hi = paired_diff_ci(
            b=bkt["dense_only_pass"], c=bkt["cert_only_pass"], n=n,
        )
        out[s] = {
            "dense": dense, "cert": cert, "delta": cert - dense,
            "crit": bkt["crit"], "n": n,
            "dense_pass_rate": dense_pass / n,
            "cert_pass_rate": cert_pass / n,
            "dense_ci_lo": d_lo, "dense_ci_hi": d_hi,
            "cert_ci_lo": c_lo, "cert_ci_hi": c_hi,
            "paired_diff": diff,
            "paired_diff_ci_lo": diff_lo,
            "paired_diff_ci_hi": diff_hi,
            "both_pass": bkt["both_pass"],
            "dense_only_pass": bkt["dense_only_pass"],
            "cert_only_pass": bkt["cert_only_pass"],
            "both_fail": bkt["both_fail"],
        }
    return out


def print_table(rows, headers):
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*r))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--csv", default=None, type=Path)
    args = parser.parse_args()

    all_configs = {}
    missing = []
    for tag, _, _ in CONFIGS:
        d = load_config(args.dir, tag)
        if d is None:
            missing.append(tag)
            continue
        all_configs[tag] = summarise_config(d)

    if missing:
        print(f"# Missing configs (not yet complete): {missing}\n")

    # Per-subtask table: rows=subtasks, cols=(dense, cert, Δ) × 4 configs
    print(f"\n## Per-subtask mean score (dir={args.dir.name})\n")
    headers = ["subtask"]
    for _, label, ctx in CONFIGS:
        headers += [f"{label} {ctx//1024}K dense", f"cert", "Δ", "crit"]
    rows = []
    for s in SUBTASKS:
        row = [s]
        for tag, _, _ in CONFIGS:
            if tag in all_configs and s in all_configs[tag]:
                x = all_configs[tag][s]
                row += [f"{x['dense']:.3f}", f"{x['cert']:.3f}",
                        f"{x['delta']:+.3f}", str(x['crit'])]
            else:
                row += ["-", "-", "-", "-"]
        rows.append(row)
    print_table(rows, headers)

    # Per-subtask pass-rate CIs (95% Wilson) and paired diff CI (Wald on
    # McNemar discordants). One table per config so the rows stay narrow.
    for tag, label, ctx in CONFIGS:
        if tag not in all_configs:
            continue
        print(f"\n## 95% CIs — {label} {ctx//1024}K (n per subtask)\n")
        ci_headers = [
            "subtask", "n", "dense%", "dense CI", "cert%", "cert CI",
            "Δ (c-d)", "Δ CI", "b(d>c)", "c(c>d)",
        ]
        ci_rows = []
        for s in SUBTASKS:
            if s not in all_configs[tag]:
                continue
            x = all_configs[tag][s]
            ci_rows.append([
                s,
                str(x["n"]),
                f"{x['dense_pass_rate']*100:.1f}",
                f"[{x['dense_ci_lo']*100:.1f}, {x['dense_ci_hi']*100:.1f}]",
                f"{x['cert_pass_rate']*100:.1f}",
                f"[{x['cert_ci_lo']*100:.1f}, {x['cert_ci_hi']*100:.1f}]",
                f"{x['paired_diff']*100:+.1f}",
                f"[{x['paired_diff_ci_lo']*100:+.1f}, {x['paired_diff_ci_hi']*100:+.1f}]",
                str(x["dense_only_pass"]),
                str(x["cert_only_pass"]),
            ])
        print_table(ci_rows, ci_headers)

    # Overall per-config: pool paired cells across subtasks for a
    # single config-level paired CI.
    print("\n## Overall (all subtasks pooled, with paired 95% CI)\n")
    rows = []
    for tag, label, ctx in CONFIGS:
        if tag not in all_configs:
            rows.append([tag, "-", "-", "-", "-", "-", "-", "-"])
            continue
        d = all_configs[tag]
        total_n = sum(x["n"] for x in d.values())
        dense = sum(x["dense"] * x["n"] for x in d.values()) / total_n
        cert = sum(x["cert"] * x["n"] for x in d.values()) / total_n
        crit = sum(x["crit"] for x in d.values())
        b = sum(x["dense_only_pass"] for x in d.values())
        c = sum(x["cert_only_pass"] for x in d.values())
        diff, diff_lo, diff_hi = paired_diff_ci(b, c, total_n)
        rows.append([
            tag, f"{dense:.3f}", f"{cert:.3f}", f"{cert - dense:+.3f}",
            f"{crit}/{total_n}",
            f"{diff*100:+.2f}",
            f"[{diff_lo*100:+.2f}, {diff_hi*100:+.2f}]",
            str(total_n),
        ])
    print_table(rows, [
        "config", "dense_mean", "cert_mean", "Δ_mean", "crit",
        "Δ pass% (c-d)", "paired 95% CI", "n",
    ])

    # CSV dump — now includes Wilson + paired CI columns plus the full
    # paired-cell counts so downstream scripts can recompute any interval.
    if args.csv is not None:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "config", "ctx", "subtask",
                "dense_mean", "cert_mean", "delta_mean", "crit", "n",
                "dense_pass_rate", "dense_ci_lo", "dense_ci_hi",
                "cert_pass_rate", "cert_ci_lo", "cert_ci_hi",
                "paired_diff", "paired_diff_ci_lo", "paired_diff_ci_hi",
                "both_pass", "dense_only_pass", "cert_only_pass", "both_fail",
            ])
            for tag, _, ctx in CONFIGS:
                if tag not in all_configs:
                    continue
                for s, x in all_configs[tag].items():
                    w.writerow([
                        tag, ctx, s,
                        f"{x['dense']:.4f}", f"{x['cert']:.4f}",
                        f"{x['delta']:+.4f}", x['crit'], x['n'],
                        f"{x['dense_pass_rate']:.4f}",
                        f"{x['dense_ci_lo']:.4f}", f"{x['dense_ci_hi']:.4f}",
                        f"{x['cert_pass_rate']:.4f}",
                        f"{x['cert_ci_lo']:.4f}", f"{x['cert_ci_hi']:.4f}",
                        f"{x['paired_diff']:+.4f}",
                        f"{x['paired_diff_ci_lo']:+.4f}",
                        f"{x['paired_diff_ci_hi']:+.4f}",
                        x['both_pass'], x['dense_only_pass'],
                        x['cert_only_pass'], x['both_fail'],
                    ])
        print(f"\nCSV -> {args.csv}")


if __name__ == "__main__":
    main()
