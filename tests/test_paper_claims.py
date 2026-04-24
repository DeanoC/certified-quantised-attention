from __future__ import annotations

import json
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "Certified_Quantised_Attention.tex"
RESULTS = ROOT / "results"


def test_paper_contract_and_default_operating_point_are_present() -> None:
    text = PAPER.read_text(encoding="utf-8")

    required_snippets = [
        "Certified Quantised Attention",
        "Runtime Precision Escalation",
        "unconditional fallback",
        "O_{\\mathrm{dense}}",
        "\\tau_{\\mathrm{cov}} = 0.995",
        "K_{\\max}{=}128",
        "group size $g{=}16$",
        "Rung 1",
        "Rung 2",
        "Rung 3",
        "Rung 4",
        "PG-19",
        "NIAH",
        "RULER",
    ]
    for snippet in required_snippets:
        assert snippet in text


def test_softmax_total_variation_bound_from_appendix() -> None:
    generator = torch.Generator().manual_seed(0)
    for delta in (0.01, 0.1, 0.5, 1.0):
        for _ in range(100):
            logits = torch.randn(32, generator=generator)
            perturb = (torch.rand(32, generator=generator) * 2.0 - 1.0) * delta
            p = torch.softmax(logits, dim=0)
            q = torch.softmax(logits + perturb, dim=0)
            tv = 0.5 * (p - q).abs().sum()
            assert tv <= math.tanh(delta) + 1e-6


def test_value_compression_bound_from_theorem_1() -> None:
    weights = torch.tensor([0.55, 0.25, 0.20], dtype=torch.float32)
    values = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]], dtype=torch.float32)
    errors = torch.tensor([[0.05, 0.00], [-0.03, 0.04], [0.00, -0.02]], dtype=torch.float32)
    eta = errors.norm(dim=-1).max()

    quantised_values = values + errors
    output_error = (weights @ quantised_values - weights @ values).norm()

    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert output_error <= eta + 1e-7


def test_storage_arithmetic_matches_paper_table() -> None:
    d = 128
    block_size = 16
    group_size = 16

    dense_bytes = d * 2 + d * 2
    key_int8 = d
    key_metadata = 2 * d * 4 / block_size
    value_int4 = d / 2
    value_metadata = 2 * (d / group_size) * 2
    tier1 = key_int8 + key_metadata + value_int4 + value_metadata

    assert dense_bytes == 512
    assert tier1 == 288
    assert tier1 / dense_bytes == 0.5625


def test_pg19_result_jsons_support_near_parity_claim() -> None:
    for context in (4096, 8192, 16384, 32768):
        data = json.loads((RESULTS / f"pg19_ctx{context}.json").read_text())
        dense = data["dense"]["perplexity"]
        certified = data["certified"]["perplexity"]
        delta = data["delta"]

        assert data["context_length"] == context
        assert math.isclose(certified - dense, delta, rel_tol=0.0, abs_tol=1e-9)
        assert abs(delta) < 0.012


def test_result_artifacts_cover_paper_benchmarks_and_tau_sweep() -> None:
    assert (RESULTS / "raw_arxiv_v1").is_dir()
    assert len(list((RESULTS / "raw_arxiv_v1").glob("*_pg19_*.json"))) >= 8
    assert len(list((RESULTS / "raw_arxiv_v1").glob("*_niah_*.json"))) >= 8
    assert len(list((RESULTS / "raw_arxiv_v1").glob("*_ruler_*.json"))) >= 8

    niah = json.loads((RESULTS / "niah_8k_tau_sweep" / "niah_8k_tau0995_n100.json").read_text())
    assert niah["dense_accuracy"] == 0.39
    assert niah["certified_accuracy"] == 0.37
    assert math.isclose(niah["certified_accuracy"] - niah["dense_accuracy"], -0.02)

    throughput = json.loads((RESULTS / "throughput_context_scaling.json").read_text())
    assert "dense" in throughput["per_config"]
    assert "certified" in throughput["per_config"]

