from __future__ import annotations

from math import ceil

import numpy as np

from .m0_affine import pad_last_dim


def _quantize_lut_flat_values(
    flat_values: np.ndarray,
    *,
    levels: int,
    refine_steps: int,
    preconditioner: str,
    precondition_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    fit_values = flat_values.astype(np.float32, copy=False)
    restore_mean = np.float32(0.0)
    restore_scale = np.float32(1.0)
    if preconditioner == "tanh":
        restore_mean = np.float32(np.mean(flat_values, dtype=np.float64))
        centered = flat_values - restore_mean
        restore_scale = np.float32(np.std(centered, dtype=np.float64))
        if restore_scale < np.float32(1e-6):
            restore_scale = np.float32(1.0)
        fit_values = np.tanh(centered / (restore_scale * np.float32(precondition_strength))).astype(np.float32, copy=False)
    elif preconditioner != "none":
        raise ValueError("unsupported preconditioner")

    lut = np.quantile(fit_values, np.linspace(0.0, 1.0, num=levels, dtype=np.float32)).astype(np.float32)
    if levels > 1:
        for _ in range(refine_steps):
            boundaries = (lut[:-1] + lut[1:]) * np.float32(0.5)
            codes = np.searchsorted(boundaries, fit_values, side="left").astype(np.int32)
            counts = np.bincount(codes, minlength=levels)
            sums = np.bincount(codes, weights=fit_values.astype(np.float64, copy=False), minlength=levels)
            updated = lut.copy()
            valid = counts > 0
            updated[valid] = (sums[valid] / counts[valid]).astype(np.float32, copy=False)
            if np.allclose(updated, lut, atol=1e-6, rtol=0.0):
                lut = updated
                break
            lut = updated
        boundaries = (lut[:-1] + lut[1:]) * np.float32(0.5)
        codes = np.searchsorted(boundaries, fit_values, side="left").astype(np.uint8, copy=False)
    else:
        codes = np.zeros_like(fit_values, dtype=np.uint8)

    if preconditioner == "tanh":
        lut = np.clip(lut, -0.999, 0.999)
        lut = (
            np.arctanh(lut).astype(np.float32) * np.float32(restore_scale * precondition_strength)
            + np.float32(restore_mean)
        )
    return codes, lut


def _quantize_lut_segment_matrix(
    segment_values: np.ndarray,
    *,
    levels: int,
    refine_steps: int,
    preconditioner: str,
    precondition_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    group_count = int(segment_values.shape[0])
    token_count = int(segment_values.shape[1])
    group_size = int(segment_values.shape[2])
    codes = np.zeros((group_count, token_count * group_size), dtype=np.uint8)
    lut = np.zeros((group_count, levels), dtype=np.float32)
    flat_values = segment_values.reshape(group_count, token_count * group_size)
    for group_index in range(group_count):
        group_codes, group_lut = _quantize_lut_flat_values(
            flat_values[group_index],
            levels=levels,
            refine_steps=refine_steps,
            preconditioner=preconditioner,
            precondition_strength=precondition_strength,
        )
        codes[group_index] = group_codes
        lut[group_index] = group_lut
    return codes.reshape(segment_values.shape), lut


def quantize_tensor_lut(
    values: np.ndarray,
    *,
    group_size: int,
    bits: int,
    segment_count: int = 1,
    refine_steps: int = 6,
    preconditioner: str = "none",
    precondition_strength: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must have shape [token_count, head_dim]")

    token_count, head_dim = array.shape
    segment_count = max(1, min(int(segment_count), token_count))
    num_groups = ceil(head_dim / group_size)
    padded_head_dim = num_groups * group_size
    padded = pad_last_dim(array, padded_head_dim)
    grouped = padded.reshape(token_count, num_groups, group_size)
    levels = 1 << bits

    codebooks = np.zeros((num_groups, segment_count, levels), dtype=np.float32)
    codes = np.zeros((token_count, num_groups, group_size), dtype=np.uint8)

    segment_slices = np.array_split(np.arange(token_count, dtype=np.int32), segment_count)
    grouped_by_group = np.transpose(grouped, (1, 0, 2))

    for segment_index, token_indices in enumerate(segment_slices):
        segment_values = grouped_by_group[:, token_indices, :]
        segment_codes, segment_lut = _quantize_lut_segment_matrix(
            segment_values,
            levels=levels,
            refine_steps=refine_steps,
            preconditioner=preconditioner,
            precondition_strength=precondition_strength,
        )
        codes[token_indices] = np.transpose(np.clip(segment_codes, 0, levels - 1), (1, 0, 2))
        codebooks[:, segment_index] = segment_lut

    return codes, codebooks, padded_head_dim


def dequantize_group_lut(codes: np.ndarray, *, codebook: np.ndarray) -> np.ndarray:
    code_array = np.asarray(codes, dtype=np.int64)
    lut = np.asarray(codebook, dtype=np.float32)
    if lut.ndim == 1:
        return lut[code_array]
    if lut.ndim == 2 and code_array.ndim == 2:
        token_count = code_array.shape[0]
        segment_count = lut.shape[0]
        if segment_count == 1:
            return lut[0][code_array]
        segment_ids = (np.arange(token_count, dtype=np.int64) * segment_count) // max(token_count, 1)
        return lut[segment_ids[:, None], code_array]
    if lut.ndim == 2 and code_array.ndim == 1 and lut.shape[0] == code_array.shape[0]:
        return lut[np.arange(lut.shape[0]), code_array]
    raise ValueError("unsupported codebook shape for LUT decode")
