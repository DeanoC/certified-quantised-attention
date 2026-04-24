from __future__ import annotations

import math

import torch

from certified_attention import (
    compute_adaptive_topk_mask,
    compute_delta_bound,
    compute_value_error_bound,
    decide_v_format,
    decide_v_format_tight,
    detect_ranking_disagreement,
    score_consistency_violations,
)


def test_adaptive_topk_selects_smallest_prefix_meeting_coverage() -> None:
    # S_b=1 and m_b=log(mass) makes the internal log-mass exactly log(mass).
    mass = torch.tensor([[0.50, 0.30, 0.15, 0.05]], dtype=torch.float32)
    m_b = mass.log()
    S_b = torch.ones_like(m_b)

    mask, k_star, tail_mass, tau_actual = compute_adaptive_topk_mask(
        m_b,
        S_b,
        tau_cov=0.80,
        k_min=1,
        k_max=None,
    )

    assert k_star.tolist() == [2]
    assert mask.tolist() == [[True, True, False, False]]
    assert torch.allclose(tau_actual, torch.tensor([0.80]), atol=1e-6)
    assert torch.allclose(tail_mass, torch.tensor([0.20]), atol=1e-6)


def test_adaptive_topk_honours_k_min_and_k_max_clamps() -> None:
    mass = torch.tensor([[0.90, 0.05, 0.03, 0.02]], dtype=torch.float32)
    m_b = mass.log()
    S_b = torch.ones_like(m_b)

    _, k_min_star, _, _ = compute_adaptive_topk_mask(m_b, S_b, tau_cov=0.50, k_min=2)
    _, k_max_star, tail_mass, _ = compute_adaptive_topk_mask(m_b, S_b, tau_cov=0.99, k_min=1, k_max=2)

    assert k_min_star.tolist() == [2]
    assert k_max_star.tolist() == [2]
    assert torch.allclose(tail_mass, torch.tensor([0.05]), atol=1e-6)


def test_delta_bound_matches_paper_runtime_formula() -> None:
    q = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float32)
    key_scales = torch.tensor([[[0.10, 0.20, 0.30, 0.40], [0.05, 0.25, 0.10, 0.50]]])

    delta = compute_delta_bound(q, key_scales, gqa_group=1, q_scale=0.5)

    max_scales = torch.tensor([0.10, 0.25, 0.30, 0.50])
    expected = (q.abs()[0] * max_scales).sum() / (2.0 * math.sqrt(4))
    assert torch.allclose(delta, expected.unsqueeze(0))


def test_value_error_bound_uses_only_residual_int4_blocks() -> None:
    mass_frac = torch.tensor(
        [
            [0.50, 0.30, 0.20],
            [0.10, 0.70, 0.20],
        ],
        dtype=torch.float32,
    )
    topk_idx = torch.tensor([[0], [1]])
    skip_mask = torch.tensor(
        [
            [False, False, True],
            [False, False, False],
        ],
        dtype=torch.bool,
    )
    eta_per_block = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)

    e_val = compute_value_error_bound(mass_frac, topk_idx, skip_mask, eta_per_block, gqa_group=2)

    # Head 0: block 0 is top-K, block 2 is skipped, residual is block 1.
    # Head 1: block 1 is top-K, residual is blocks 0 and 2.
    assert torch.allclose(e_val, torch.tensor([0.30 * 2.0, 0.10 * 1.0 + 0.20 * 3.0]))


def test_value_format_decisions_switch_at_tolerance() -> None:
    assert decide_v_format(torch.tensor([0.1, 0.2]), eta_int4=1.0, tolerance=0.5) == "int4"
    assert decide_v_format(torch.tensor([0.1, 0.6]), eta_int4=1.0, tolerance=0.5) == "fp16"
    assert decide_v_format_tight(torch.tensor([0.2, 0.49]), tolerance=0.5) == "int4"
    assert decide_v_format_tight(torch.tensor([0.2, 0.50]), tolerance=0.5) == "fp16"


def test_ranking_and_score_consistency_monitors() -> None:
    int8_scores = torch.tensor([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]])
    fp16_same_top1 = torch.tensor([[2.9, 2.1, 1.0], [1.0, 2.0, 3.1]])

    disagreement = detect_ranking_disagreement(int8_scores, fp16_same_top1, r=1)
    assert disagreement.tolist() == [False, True]

    violations = score_consistency_violations(
        int8_scores,
        fp16_same_top1,
        delta_per_head=torch.tensor([0.2, 0.2]),
        eps_guard=0.05,
    )
    assert violations.tolist() == [False, True]

