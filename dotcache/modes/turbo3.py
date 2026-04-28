from __future__ import annotations

from math import ceil

import numpy as np

from .m0_affine import pad_last_dim

TURBO3_CENTROIDS = np.asarray(
    [-1.863, -1.318, -0.912, -0.522, 0.185, 0.603, 1.016, 1.594],
    dtype=np.float32,
)


def fwht_last_dim(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape[-1] == 0:
        return array.copy()
    width = int(array.shape[-1])
    if width & (width - 1):
        raise ValueError("FWHT requires the last dimension to be a power of two")
    original_shape = array.shape
    transformed = array.reshape(-1, width).copy()
    step = 1
    norm = np.float32(np.sqrt(width))
    while step < width:
        block = step * 2
        reshaped = transformed.reshape(-1, width // block, block)
        left = reshaped[..., :step].copy()
        right = reshaped[..., step:block].copy()
        reshaped[..., :step] = left + right
        reshaped[..., step:block] = left - right
        transformed = reshaped.reshape(-1, width)
        step = block
    return (transformed / norm).reshape(original_shape)


def quantize_tensor_turbo3(
    values: np.ndarray,
    *,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must have shape [token_count, head_dim]")
    if group_size <= 0 or (group_size & (group_size - 1)):
        raise ValueError("turbo3 requires a power-of-two group_size")

    token_count, head_dim = array.shape
    num_groups = ceil(head_dim / group_size)
    padded_head_dim = num_groups * group_size
    padded = pad_last_dim(array, padded_head_dim)
    grouped = padded.reshape(token_count, num_groups, group_size)
    rotated = fwht_last_dim(grouped)
    group_norm = np.linalg.norm(rotated, axis=-1).astype(np.float32)
    normalized = rotated / np.maximum(group_norm[..., None], 1e-6)

    centroid_deltas = np.abs(normalized[..., None] - TURBO3_CENTROIDS.reshape(1, 1, 1, -1))
    codes = np.argmin(centroid_deltas, axis=-1).astype(np.uint8, copy=False)
    reconstructed = TURBO3_CENTROIDS[codes.astype(np.int64)]
    reconstructed_norm = np.linalg.norm(reconstructed, axis=-1).astype(np.float32)
    correction = group_norm / np.maximum(reconstructed_norm, 1e-6)

    return (
        codes,
        correction.astype(np.float16, copy=False),
        TURBO3_CENTROIDS.astype(np.float16, copy=False),
        padded_head_dim,
    )


def dequantize_group_turbo3(
    codes: np.ndarray,
    *,
    correction: np.ndarray,
    centroids: np.ndarray | None = None,
) -> np.ndarray:
    centroid_table = TURBO3_CENTROIDS if centroids is None else np.asarray(centroids, dtype=np.float32)
    code_array = np.asarray(codes, dtype=np.int64)
    corrected = centroid_table[code_array] * np.asarray(correction, dtype=np.float32)[:, None]
    return fwht_last_dim(corrected)
