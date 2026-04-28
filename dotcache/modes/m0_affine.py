from __future__ import annotations

from math import ceil

import numpy as np


def pad_last_dim(values: np.ndarray, padded_size: int) -> np.ndarray:
    pad_width = padded_size - values.shape[-1]
    if pad_width <= 0:
        return values
    return np.pad(values, ((0, 0), (0, pad_width)), mode="constant")


def quantize_tensor(
    values: np.ndarray,
    *,
    group_size: int,
    bits: int,
    scheme: str = "affine",
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must have shape [token_count, head_dim]")

    token_count, head_dim = array.shape
    num_groups = ceil(head_dim / group_size)
    padded_head_dim = num_groups * group_size
    padded = pad_last_dim(array, padded_head_dim)
    grouped = padded.reshape(token_count, num_groups, group_size)

    if scheme == "affine":
        qmin = 0
        qmax = (1 << bits) - 1
        x_min = grouped.min(axis=-1)
        x_max = grouped.max(axis=-1)
        scales = np.maximum((x_max - x_min) / max(qmax - qmin, 1), eps)
        shifted = (grouped - x_min[..., None]) / scales[..., None]
        codes = np.clip(np.round(shifted), qmin, qmax).astype(np.uint8)
        bias = x_min.astype(np.float32)
        return codes, scales.astype(np.float32), bias, padded_head_dim

    if scheme == "symmetric":
        qmax = (1 << (bits - 1)) - 1
        zero_point = qmax
        max_abs = np.max(np.abs(grouped), axis=-1)
        scales = np.maximum(max_abs / max(qmax, 1), eps)
        signed_codes = np.clip(np.round(grouped / scales[..., None]), -qmax, qmax).astype(np.int32)
        codes = np.clip(signed_codes + zero_point, 0, (1 << bits) - 1).astype(np.uint8)
        return codes, scales.astype(np.float32), None, padded_head_dim

    raise ValueError("scheme must be affine or symmetric")


def dequantize_group(
    codes: np.ndarray,
    *,
    scales: np.ndarray,
    bias: np.ndarray | None,
    bits: int,
    scheme: str,
) -> np.ndarray:
    code_array = np.asarray(codes, dtype=np.float32)
    scale_array = np.asarray(scales, dtype=np.float32)

    if scheme == "affine":
        if bias is None:
            raise ValueError("affine mode requires bias")
        bias_array = np.asarray(bias, dtype=np.float32)
        return scale_array * code_array + bias_array

    if scheme == "symmetric":
        zero_point = (1 << (bits - 1)) - 1
        return scale_array * (code_array - zero_point)

    raise ValueError("scheme must be affine or symmetric")


def dequantize_groups(
    codes: np.ndarray,
    *,
    scales: np.ndarray,
    bias: np.ndarray | None,
    bits: int,
    scheme: str,
) -> np.ndarray:
    expanded_scales = np.asarray(scales, dtype=np.float32)[..., None]
    expanded_bias = None if bias is None else np.asarray(bias, dtype=np.float32)[..., None]
    return dequantize_group(codes, scales=expanded_scales, bias=expanded_bias, bits=bits, scheme=scheme)

