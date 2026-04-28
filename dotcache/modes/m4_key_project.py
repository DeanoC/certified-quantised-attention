from __future__ import annotations

from functools import lru_cache
from math import ceil

import numpy as np

from .m0_affine import pad_last_dim
from .turbo3 import fwht_last_dim

_VALID_M4_BASIS_FAMILIES = ("hadamard", "dct", "svd", "svd_shared")


def valid_m4_basis_families() -> tuple[str, ...]:
    return _VALID_M4_BASIS_FAMILIES


def _dct_basis(group_size: int) -> np.ndarray:
    positions = np.arange(group_size, dtype=np.float32)[None, :]
    frequencies = np.arange(group_size, dtype=np.float32)[:, None]
    basis = np.cos((np.pi / np.float32(group_size)) * (positions + np.float32(0.5)) * frequencies).astype(np.float32)
    basis[0] *= np.float32(np.sqrt(1.0 / group_size))
    if group_size > 1:
        basis[1:] *= np.float32(np.sqrt(2.0 / group_size))
    return basis


@lru_cache(maxsize=None)
def fixed_project_basis(group_size: int, rank: int, basis_family: str = "hadamard") -> np.ndarray:
    if group_size <= 0 or (group_size & (group_size - 1)):
        raise ValueError("M4 fixed-project requires a power-of-two group_size")
    if basis_family not in _VALID_M4_BASIS_FAMILIES:
        raise ValueError(f"M4 fixed-project basis_family must be one of {', '.join(_VALID_M4_BASIS_FAMILIES)}")
    usable_rank = max(1, min(int(rank), group_size - 1))
    if basis_family == "hadamard":
        basis = fwht_last_dim(np.eye(group_size, dtype=np.float32))
    else:
        basis = _dct_basis(group_size)
    # Skip the DC row because the page mean already captures the constant offset.
    return np.asarray(basis[1 : 1 + usable_rank], dtype=np.float32)


def quantize_tensor_m4(
    values: np.ndarray,
    *,
    group_size: int,
    project_dim: int,
    basis_family: str = "hadamard",
    basis_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, int]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must have shape [token_count, head_dim]")

    token_count, head_dim = array.shape
    num_groups = ceil(head_dim / group_size)
    padded_head_dim = num_groups * group_size
    padded = pad_last_dim(array, padded_head_dim)
    grouped = padded.reshape(token_count, num_groups, group_size)

    if basis_override is not None:
        stored_basis = np.asarray(basis_override)
        learned_basis = stored_basis.astype(np.float32, copy=False)
        if learned_basis.ndim != 3 or learned_basis.shape[0] != num_groups or learned_basis.shape[2] != group_size:
            raise ValueError("basis_override must have shape [num_groups, rank, group_size]")
        coeffs = np.zeros((token_count, num_groups, learned_basis.shape[1]), dtype=np.float32)
        mean = np.zeros((num_groups, group_size), dtype=np.float32)
        for group_index in range(num_groups):
            group_values = grouped[:, group_index, :]
            group_mean = group_values.mean(axis=0, dtype=np.float32)
            centered = group_values - group_mean[None, :]
            coeffs[:, group_index, :] = centered @ learned_basis[group_index].T
            mean[group_index, :] = group_mean
        return (
            coeffs.astype(np.float16, copy=False),
            stored_basis.astype(np.float16, copy=False),
            mean.astype(np.float16, copy=False),
            padded_head_dim,
        )

    if basis_family in {"svd", "svd_shared"}:
        usable_rank = max(1, min(int(project_dim), group_size, token_count))
        coeffs = np.zeros((token_count, num_groups, usable_rank), dtype=np.float32)
        learned_basis = np.zeros((num_groups, usable_rank, group_size), dtype=np.float32)
        mean = np.zeros((num_groups, group_size), dtype=np.float32)
        for group_index in range(num_groups):
            group_values = grouped[:, group_index, :]
            group_mean = group_values.mean(axis=0, dtype=np.float32)
            centered = group_values - group_mean[None, :]
            u, s, vt = np.linalg.svd(centered, full_matrices=False)
            group_rank = max(1, min(usable_rank, int(vt.shape[0]), int(u.shape[1])))
            coeffs[:, group_index, :group_rank] = (u[:, :group_rank] * s[:group_rank]).astype(np.float32, copy=False)
            learned_basis[group_index, :group_rank, :] = vt[:group_rank, :].astype(np.float32, copy=False)
            mean[group_index, :] = group_mean
        return (
            coeffs.astype(np.float16, copy=False),
            learned_basis.astype(np.float16, copy=False),
            mean.astype(np.float16, copy=False),
            padded_head_dim,
        )

    basis = fixed_project_basis(group_size, project_dim, basis_family)
    coeffs = np.zeros((token_count, num_groups, basis.shape[0]), dtype=np.float32)
    mean = np.zeros((num_groups, group_size), dtype=np.float32)

    for group_index in range(num_groups):
        group_values = grouped[:, group_index, :]
        group_mean = group_values.mean(axis=0, dtype=np.float32)
        centered = group_values - group_mean[None, :]
        coeffs[:, group_index, :] = centered @ basis.T
        mean[group_index, :] = group_mean

    return (
        coeffs.astype(np.float16, copy=False),
        None,
        mean.astype(np.float16, copy=False),
        padded_head_dim,
    )


def fit_shared_project_basis(
    values: np.ndarray,
    *,
    group_size: int,
    project_dim: int,
    page_size: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must have shape [token_count, head_dim]")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    token_count, head_dim = array.shape
    num_groups = ceil(head_dim / group_size)
    padded_head_dim = num_groups * group_size
    padded = pad_last_dim(array, padded_head_dim)
    grouped = padded.reshape(token_count, num_groups, group_size)
    usable_rank = max(1, min(int(project_dim), group_size, token_count))
    basis = np.zeros((num_groups, usable_rank, group_size), dtype=np.float32)

    for group_index in range(num_groups):
        group_values = grouped[:, group_index, :]
        residual_chunks: list[np.ndarray] = []
        for page_start in range(0, token_count, page_size):
            page_values = group_values[page_start : page_start + page_size]
            if page_values.shape[0] == 0:
                continue
            page_mean = page_values.mean(axis=0, dtype=np.float32)
            residual_chunks.append(page_values - page_mean[None, :])
        residual = np.concatenate(residual_chunks, axis=0) if residual_chunks else group_values
        u, s, vt = np.linalg.svd(residual, full_matrices=False)
        group_rank = max(1, min(usable_rank, int(vt.shape[0]), int(u.shape[1])))
        basis[group_index, :group_rank, :] = vt[:group_rank, :].astype(np.float32, copy=False)
    return basis


def reconstruct_group_m4(
    coefficients: np.ndarray,
    *,
    mean: np.ndarray,
    group_size: int,
    basis_family: str = "hadamard",
    basis: np.ndarray | None = None,
) -> np.ndarray:
    coeff_array = np.asarray(coefficients, dtype=np.float32)
    if basis is None:
        rank = int(coeff_array.shape[-1])
        basis_array = fixed_project_basis(int(group_size), rank, basis_family)
    else:
        basis_array = np.asarray(basis, dtype=np.float32)
    reconstructed = coeff_array @ basis_array
    return reconstructed + np.asarray(mean, dtype=np.float32)[None, :]
