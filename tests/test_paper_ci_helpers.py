"""Tests for the RULER + PG-19 confidence-interval helpers (paper reviewer
item: per-subtask / per-book CIs from retained per-unit data)."""
from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# wilson_ci — marginal Bernoulli rate CI
# ---------------------------------------------------------------------------

class TestWilsonCI:
    def test_empty_sample_returns_full_interval(self):
        from benchmarks.paper.aggregate_ruler import wilson_ci
        lo, hi = wilson_ci(0, 0)
        assert lo == 0.0 and hi == 1.0

    def test_all_pass_upper_is_one(self):
        """k = n: Wilson still gives an upper of 1 (or very close) and a
        nontrivial lower. Distinguishes it from the Wald CI which would
        collapse to [1, 1]."""
        from benchmarks.paper.aggregate_ruler import wilson_ci
        lo, hi = wilson_ci(20, 20)
        assert hi == pytest.approx(1.0, abs=1e-9)
        assert 0.80 < lo < 1.0, f"all-pass lower should be in (0.80, 1); got {lo}"

    def test_all_fail_lower_is_zero(self):
        from benchmarks.paper.aggregate_ruler import wilson_ci
        lo, hi = wilson_ci(0, 20)
        assert lo == pytest.approx(0.0, abs=1e-9)
        assert 0.0 < hi < 0.20, f"all-fail upper should be in (0, 0.20); got {hi}"

    def test_half_pass_centred(self):
        from benchmarks.paper.aggregate_ruler import wilson_ci
        lo, hi = wilson_ci(50, 100)
        # ~0.5 rate with n=100: Wilson CI is roughly [0.40, 0.60].
        assert 0.39 < lo < 0.41
        assert 0.59 < hi < 0.61

    def test_narrows_with_larger_n(self):
        from benchmarks.paper.aggregate_ruler import wilson_ci
        lo_small, hi_small = wilson_ci(5, 10)
        lo_big, hi_big = wilson_ci(500, 1000)
        assert (hi_big - lo_big) < (hi_small - lo_small) / 5


# ---------------------------------------------------------------------------
# paired_diff_ci — 2×2 paired proportion difference
# ---------------------------------------------------------------------------

class TestPairedDiffCI:
    def test_all_agree_returns_point_interval(self):
        """b = c = 0 means dense and cert agree on every sample. Wald CI
        collapses to a point."""
        from benchmarks.paper.aggregate_ruler import paired_diff_ci
        point, lo, hi = paired_diff_ci(b=0, c=0, n=50)
        assert point == 0.0
        assert lo == 0.0 and hi == 0.0

    def test_empty_sample(self):
        from benchmarks.paper.aggregate_ruler import paired_diff_ci
        point, lo, hi = paired_diff_ci(b=0, c=0, n=0)
        assert point == 0.0
        assert lo == -1.0 and hi == 1.0  # uninformative

    def test_cert_strictly_better(self):
        """c > b: point estimate > 0, interval stays above zero for
        sufficiently many discordants."""
        from benchmarks.paper.aggregate_ruler import paired_diff_ci
        point, lo, hi = paired_diff_ci(b=1, c=19, n=100)  # cert beats dense 18 net
        assert 0.17 < point < 0.19
        assert lo > 0, f"lower bound should exclude zero, got {lo}"

    def test_symmetric_sign(self):
        """Swapping b ↔ c negates the point estimate and the interval."""
        from benchmarks.paper.aggregate_ruler import paired_diff_ci
        p1, l1, h1 = paired_diff_ci(b=3, c=15, n=50)
        p2, l2, h2 = paired_diff_ci(b=15, c=3, n=50)
        assert p1 == pytest.approx(-p2, abs=1e-12)
        assert l1 == pytest.approx(-h2, abs=1e-9)
        assert h1 == pytest.approx(-l2, abs=1e-9)

    def test_wider_than_zero_when_few_discordants(self):
        """A tiny absolute difference on few discordants should include
        zero in the CI (not falsely 'significant')."""
        from benchmarks.paper.aggregate_ruler import paired_diff_ci
        point, lo, hi = paired_diff_ci(b=1, c=2, n=100)
        assert lo < 0 < hi, f"small imbalance should straddle zero, got [{lo}, {hi}]"


# ---------------------------------------------------------------------------
# paired_niah_stats — NIAH bootstrap CI + exact McNemar
# ---------------------------------------------------------------------------

class TestNiahPairedStats:
    @staticmethod
    def _row(ctx: int, depth: float, needle: int, correct: bool) -> dict:
        return {
            "target_context": ctx,
            "depth": depth,
            "needle_idx": needle,
            "correct": correct,
        }

    def test_exact_mcnemar_uses_discordant_pairs_only(self):
        from benchmarks.paper.niah import paired_niah_stats

        dense = [
            self._row(8192, 0.0, 0, True),   # both correct
            self._row(8192, 0.1, 0, True),   # dense only
            self._row(8192, 0.2, 0, False),  # cert only
            self._row(8192, 0.3, 0, False),  # cert only
            self._row(8192, 0.4, 0, False),  # both wrong
        ]
        cert = [
            self._row(8192, 0.0, 0, True),
            self._row(8192, 0.1, 0, False),
            self._row(8192, 0.2, 0, True),
            self._row(8192, 0.3, 0, True),
            self._row(8192, 0.4, 0, False),
        ]

        stats = paired_niah_stats(dense, cert, bootstrap_iters=500, seed=0)
        assert stats["n"] == 5
        assert stats["dense_accuracy"] == pytest.approx(2 / 5)
        assert stats["certified_accuracy"] == pytest.approx(3 / 5)
        assert stats["delta_accuracy"] == pytest.approx(1 / 5)
        assert stats["delta_pp"] == pytest.approx(20.0)
        assert stats["paired_table"] == {
            "both_correct": 1,
            "dense_only": 1,
            "certified_only": 2,
            "both_wrong": 1,
        }
        # Discordants are 1 vs 2. Exact two-sided binomial p = 1.0.
        assert stats["mcnemar_p"] == pytest.approx(1.0)
        assert stats["bootstrap_ci_lo"] <= stats["delta_accuracy"] <= stats["bootstrap_ci_hi"]

    def test_mcnemar_detects_extreme_imbalance(self):
        from benchmarks.paper.niah import paired_niah_stats

        dense = [self._row(8192, i / 10, 0, True) for i in range(4)]
        cert = [self._row(8192, i / 10, 0, False) for i in range(4)]
        stats = paired_niah_stats(dense, cert, bootstrap_iters=0)

        assert stats["paired_table"]["dense_only"] == 4
        assert stats["paired_table"]["certified_only"] == 0
        assert stats["delta_pp"] == pytest.approx(-100.0)
        # Two-sided exact binomial with four discordants all in one direction:
        # 2 * (1 / 2^4) = 0.125.
        assert stats["mcnemar_p"] == pytest.approx(0.125)

    def test_by_context_keeps_paper_cells_separate(self):
        from benchmarks.paper.niah import paired_niah_stats_by_context

        dense = [
            self._row(4096, 0.0, 0, True),
            self._row(8192, 0.0, 0, False),
        ]
        cert = [
            self._row(4096, 0.0, 0, False),
            self._row(8192, 0.0, 0, True),
        ]
        stats = paired_niah_stats_by_context(dense, cert, bootstrap_iters=0)

        assert sorted(stats) == ["4K", "8K"]
        assert stats["4K"]["delta_pp"] == pytest.approx(-100.0)
        assert stats["8K"]["delta_pp"] == pytest.approx(100.0)

    def test_by_needle_group_splits_original_and_harder_followup(self):
        from benchmarks.paper.niah import paired_niah_stats_by_needle_group

        dense = [
            self._row(8192, 0.0, 0, True),
            self._row(8192, 0.0, 5, False),
        ]
        cert = [
            self._row(8192, 0.0, 0, False),
            self._row(8192, 0.0, 5, True),
        ]
        stats = paired_niah_stats_by_needle_group(dense, cert, bootstrap_iters=0)

        assert sorted(stats) == ["harder", "original"]
        assert stats["original"]["delta_pp"] == pytest.approx(-100.0)
        assert stats["harder"]["delta_pp"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# per_chunk_bpt_stats — PG-19 bootstrap bits-per-token CI
# ---------------------------------------------------------------------------

class TestPerChunkBptStats:
    def test_empty_returns_n_chunks_zero(self):
        from benchmarks.paper.pg19_perplexity import per_chunk_bpt_stats
        out = per_chunk_bpt_stats([])
        assert out == {"n_chunks": 0}

    def test_single_chunk_zero_ci_width(self):
        """One chunk → no sampling variance → CI is a point."""
        from benchmarks.paper.pg19_perplexity import per_chunk_bpt_stats
        per_chunk = [{"nll": math.log(2.0) * 100, "tokens": 100}]
        out = per_chunk_bpt_stats(per_chunk)
        assert out["n_chunks"] == 1
        assert out["bpt_mean"] == pytest.approx(1.0, abs=1e-9)
        assert out["bpt_ci_lo"] == out["bpt_ci_hi"] == out["bpt_mean"]

    def test_constant_chunks_zero_width(self):
        """All chunks at the same bpt → bootstrap CI collapses to the
        mean regardless of how many chunks."""
        from benchmarks.paper.pg19_perplexity import per_chunk_bpt_stats
        per_chunk = [{"nll": math.log(2.0) * 50, "tokens": 50} for _ in range(20)]
        out = per_chunk_bpt_stats(per_chunk, bootstrap_iters=500, seed=1)
        assert out["n_chunks"] == 20
        assert out["bpt_mean"] == pytest.approx(1.0, abs=1e-9)
        assert out["bpt_ci_lo"] == pytest.approx(1.0, abs=1e-9)
        assert out["bpt_ci_hi"] == pytest.approx(1.0, abs=1e-9)

    def test_ci_contains_mean(self):
        from benchmarks.paper.pg19_perplexity import per_chunk_bpt_stats
        # Build 10 chunks with bpt in {0.5, 1.0, 1.5} mix.
        import math as _m
        per_chunk = []
        for v in [0.5, 1.0, 1.5, 0.8, 1.2, 0.9, 1.1, 0.7, 1.3, 1.0]:
            per_chunk.append({"nll": v * _m.log(2.0) * 100, "tokens": 100})
        out = per_chunk_bpt_stats(per_chunk, bootstrap_iters=2000, seed=0)
        assert out["bpt_ci_lo"] <= out["bpt_mean"] <= out["bpt_ci_hi"]

    def test_narrows_with_more_chunks(self):
        """Bootstrap CI on noisier data should narrow as n grows."""
        from benchmarks.paper.pg19_perplexity import per_chunk_bpt_stats
        import math as _m
        import random as _r
        _r.seed(0)
        noisy_10 = [{"nll": (1.0 + _r.gauss(0, 0.2)) * _m.log(2.0) * 100, "tokens": 100}
                    for _ in range(10)]
        _r.seed(0)
        noisy_200 = [{"nll": (1.0 + _r.gauss(0, 0.2)) * _m.log(2.0) * 100, "tokens": 100}
                     for _ in range(200)]
        s10 = per_chunk_bpt_stats(noisy_10, bootstrap_iters=2000, seed=0)
        s200 = per_chunk_bpt_stats(noisy_200, bootstrap_iters=2000, seed=0)
        width_10 = s10["bpt_ci_hi"] - s10["bpt_ci_lo"]
        width_200 = s200["bpt_ci_hi"] - s200["bpt_ci_lo"]
        # 20× more chunks → roughly √20 ≈ 4.5× tighter CI.
        assert width_200 < width_10 / 3, (
            f"CI should narrow substantially with 20× more chunks: "
            f"width_10={width_10:.4f}  width_200={width_200:.4f}"
        )

    def test_skips_zero_token_chunks(self):
        """A chunk with tokens=0 should be dropped rather than blow up."""
        from benchmarks.paper.pg19_perplexity import per_chunk_bpt_stats
        per_chunk = [
            {"nll": math.log(2.0) * 100, "tokens": 100},
            {"nll": 5.0, "tokens": 0},  # should be skipped
            {"nll": math.log(2.0) * 50, "tokens": 50},
        ]
        out = per_chunk_bpt_stats(per_chunk, bootstrap_iters=500, seed=0)
        assert out["n_chunks"] == 2
        assert out["bpt_mean"] == pytest.approx(1.0, abs=1e-9)


class TestPg19DeltaStats:
    def test_per_chunk_delta_stats_reports_mean_ci_and_ratios(self):
        from benchmarks.paper.pg19_perplexity import per_chunk_ppl_delta_stats

        dense = [
            {"chunk_idx": 0, "nll": math.log(10.0) * 100, "tokens": 100},
            {"chunk_idx": 1, "nll": math.log(20.0) * 100, "tokens": 100},
        ]
        cert = [
            {"chunk_idx": 0, "nll": math.log(11.0) * 100, "tokens": 100},
            {"chunk_idx": 1, "nll": math.log(18.0) * 100, "tokens": 100},
        ]
        stats = per_chunk_ppl_delta_stats(dense, cert, bootstrap_iters=0)

        assert stats["n_chunks"] == 2
        assert stats["mean_delta_ppl"] == pytest.approx((1.0 - 2.0) / 2.0)
        assert stats["mean_ratio"] == pytest.approx(((11 / 10) + (18 / 20)) / 2)
        assert [r["delta_ppl"] for r in stats["per_chunk"]] == pytest.approx([1.0, -2.0])


class TestRulerPairedStats:
    def test_ruler_stats_bootstrap_paired_score_delta(self):
        from benchmarks.paper.ruler import paired_ruler_stats

        rows = [
            {"dense_score": 1.0, "cert_score": 1.0},
            {"dense_score": 1.0, "cert_score": 0.5},
            {"dense_score": 0.0, "cert_score": 1.0},
        ]
        stats = paired_ruler_stats(rows, bootstrap_iters=500, seed=0)

        assert stats["n"] == 3
        assert stats["dense_mean"] == pytest.approx(2 / 3)
        assert stats["certified_mean"] == pytest.approx(2.5 / 3)
        assert stats["delta"] == pytest.approx((0.0 - 0.5 + 1.0) / 3)
        assert stats["delta_ci_lo"] <= stats["delta"] <= stats["delta_ci_hi"]
