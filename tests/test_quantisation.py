from __future__ import annotations

import torch

from quantisation import dequantise_int4_grouped, quantise_int4_grouped, quantise_int4_grouped_block


def test_int4_grouped_shapes_and_nibble_packing() -> None:
    values = torch.arange(16, dtype=torch.float32).reshape(2, 8)

    result = quantise_int4_grouped(values, group_size=4)

    assert result["data_packed"].shape == (2, 4)
    assert result["scales"].shape == (2, 2)
    assert result["zeros"].shape == (2, 2)
    assert result["data_packed"].dtype == torch.uint8

    packed = result["data_packed"]
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    assert torch.all((0 <= low) & (low <= 15))
    assert torch.all((0 <= high) & (high <= 15))


def test_int4_roundtrip_error_bound_matches_actual_max_token_error() -> None:
    values = torch.tensor(
        [
            [-1.0, -0.5, 0.25, 1.0, 2.0, 2.5, 3.5, 4.0],
            [0.0, 0.1, 0.3, 0.7, -2.0, -1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    result = quantise_int4_grouped(values, group_size=4)
    dequant = dequantise_int4_grouped(
        result["data_packed"],
        result["scales"],
        result["zeros"],
        group_size=4,
    ).float()
    per_token_error = (values - dequant).norm(dim=-1)

    assert torch.isclose(per_token_error.max(), torch.tensor(result["error_bound"]), atol=2e-3)
    assert result["error_bound"] < 0.1


def test_block_quantisation_returns_per_block_error_annotations() -> None:
    values = torch.linspace(-1.0, 1.0, steps=2 * 8 * 8, dtype=torch.float32).reshape(2, 8, 8)

    result = quantise_int4_grouped_block(values, block_size=4, group_size=4)

    assert result["data_packed"].shape == (2, 8, 4)
    assert result["scales"].shape == (2, 8, 2)
    assert result["zeros"].shape == (2, 8, 2)
    assert result["error_bounds"].shape == (2, 2)
    assert torch.all(result["error_bounds"] >= 0)

