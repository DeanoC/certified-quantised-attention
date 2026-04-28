from __future__ import annotations

from math import ceil

import numpy as np

from .m0_affine import pad_last_dim


def segment_ids_for_token_count(token_count: int, segment_count: int) -> np.ndarray:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if segment_count <= 0:
        raise ValueError("segment_count must be positive")
    if segment_count == 1:
        return np.zeros(token_count, dtype=np.int64)
    return ((np.arange(token_count, dtype=np.int64) * segment_count) // token_count).astype(np.int64, copy=False)


def quantize_tensor_m2(
    values: np.ndarray,
    *,
    group_size: int,
    sketch_dim: int,
    center: bool = True,
    segment_count: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must have shape [token_count, head_dim]")

    token_count, head_dim = array.shape
    num_groups = ceil(head_dim / group_size)
    padded_head_dim = num_groups * group_size
    padded = pad_last_dim(array, padded_head_dim)
    grouped = padded.reshape(token_count, num_groups, group_size)

    rank = max(1, min(int(sketch_dim), group_size, token_count))
    segment_count = max(1, min(int(segment_count), token_count))
    segment_ids = segment_ids_for_token_count(token_count, segment_count)

    coeffs = np.zeros((token_count, num_groups, rank), dtype=np.float32)
    basis = np.zeros((num_groups, segment_count, rank, group_size), dtype=np.float32)
    mean = np.zeros((num_groups, segment_count, group_size), dtype=np.float32)

    for group_index in range(num_groups):
        group_values = grouped[:, group_index, :]
        for segment_index in range(segment_count):
            mask = segment_ids == segment_index
            segment_values = group_values[mask]
            if segment_values.shape[0] == 0:
                continue
            segment_rank = max(1, min(rank, segment_values.shape[0], group_size))
            if center:
                group_mean = segment_values.mean(axis=0, dtype=np.float32)
                residual = segment_values - group_mean[None, :]
                mean[group_index, segment_index, :] = group_mean
            else:
                residual = segment_values
            u, s, vt = np.linalg.svd(residual, full_matrices=False)
            coeffs[mask, group_index, :segment_rank] = (u[:, :segment_rank] * s[:segment_rank]).astype(
                np.float32,
                copy=False,
            )
            basis[group_index, segment_index, :segment_rank, :] = vt[:segment_rank, :].astype(np.float32, copy=False)

    return (
        coeffs.astype(np.float16, copy=False),
        basis.astype(np.float16, copy=False),
        mean.astype(np.float16, copy=False),
        padded_head_dim,
    )


def reconstruct_group_m2(coefficients: np.ndarray, *, basis: np.ndarray, mean: np.ndarray | None = None) -> np.ndarray:
    coeff_array = np.asarray(coefficients, dtype=np.float32)
    basis_array = np.asarray(basis, dtype=np.float32)
    token_count = int(coeff_array.shape[0])
    if basis_array.ndim == 2:
        reconstructed = coeff_array @ basis_array
        if mean is not None:
            reconstructed = reconstructed + np.asarray(mean, dtype=np.float32)[None, :]
        return reconstructed
    if basis_array.ndim != 3:
        raise ValueError("basis must have shape [rank, group_size] or [segment_count, rank, group_size]")
    segment_count = int(basis_array.shape[0])
    segment_ids = segment_ids_for_token_count(token_count, segment_count)
    reconstructed = np.einsum("tr,trg->tg", coeff_array, basis_array[segment_ids])
    if mean is not None:
        reconstructed = reconstructed + np.asarray(mean, dtype=np.float32)[segment_ids]
    return reconstructed
