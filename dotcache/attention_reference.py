from __future__ import annotations

import numpy as np

from .decode_reference import decode_page
from .modes.m1_lut import dequantize_group_lut
from .modes.m2_key_sketch import segment_ids_for_token_count
from .modes.m4_key_project import fixed_project_basis
from .modes.m3_escape import decode_escape_payload
from .modes.turbo3 import fwht_last_dim
from .page_format import load_group_words
from .packing import unpack_bits
from .types import EncodedPage


def _pad_query(query_slice: np.ndarray, padded_head_dim: int) -> np.ndarray:
    query = np.asarray(query_slice, dtype=np.float32)
    if query.ndim != 1:
        raise ValueError("query_slice must have shape [head_dim]")
    if query.shape[0] > padded_head_dim:
        raise ValueError("query head_dim exceeds padded_head_dim")
    if query.shape[0] == padded_head_dim:
        return query
    return np.pad(query, (0, padded_head_dim - query.shape[0]), mode="constant")


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def score_page_ref(query_slice: np.ndarray, page: EncodedPage) -> np.ndarray:
    header = page.header
    query = _pad_query(query_slice, header.padded_head_dim)

    if header.mode_default == "M3":
        if page.escape_payload is None:
            raise ValueError("escape payload is missing")
        dense = decode_escape_payload(page.escape_payload, head_dim=header.head_dim, scales=page.escape_scales)
        return dense @ query_slice.astype(np.float32)

    if header.mode_default == "M2":
        if page.m2_sketch is None or page.m2_basis is None:
            raise ValueError("M2 page is missing sketch payload")
        query_groups = query.reshape(header.num_groups, header.group_size)
        logits = np.zeros(header.token_count, dtype=np.float32)
        for group_index in range(header.num_groups):
            group_mean = None if page.m2_mean is None else page.m2_mean[group_index].astype(np.float32)
            group_basis = page.m2_basis[group_index].astype(np.float32)
            if group_basis.ndim == 2:
                q_proj = group_basis @ query_groups[group_index]
                logits += page.m2_sketch[:, group_index, :].astype(np.float32) @ q_proj.astype(np.float32)
                if group_mean is not None:
                    logits += np.dot(group_mean, query_groups[group_index]).astype(np.float32)
                continue
            segment_ids = segment_ids_for_token_count(header.token_count, int(group_basis.shape[0]))
            q_proj = np.einsum("srg,g->sr", group_basis, query_groups[group_index])
            logits += np.einsum("tr,tr->t", page.m2_sketch[:, group_index, :].astype(np.float32), q_proj[segment_ids])
            if group_mean is not None:
                logits += group_mean[segment_ids].astype(np.float32) @ query_groups[group_index]
        return logits

    if header.mode_default == "M4":
        if page.m2_sketch is None or page.m2_mean is None:
            raise ValueError("M4 page is missing projected payload")
        query_groups = query.reshape(header.num_groups, header.group_size)
        logits = np.zeros(header.token_count, dtype=np.float32)
        for group_index in range(header.num_groups):
            basis = (
                np.asarray(page.m2_basis[group_index], dtype=np.float32)
                if page.m2_basis is not None
                else fixed_project_basis(header.group_size, int(page.m2_sketch.shape[-1]), header.project_basis)
            )
            q_proj = basis @ query_groups[group_index]
            logits += page.m2_sketch[:, group_index, :].astype(np.float32) @ q_proj.astype(np.float32)
            logits += np.dot(page.m2_mean[group_index].astype(np.float32), query_groups[group_index]).astype(np.float32)
        return logits

    if header.mode_default == "T3":
        if page.payload is None or page.scales is None or page.codebooks is None:
            raise ValueError("T3 page is missing payload or correction metadata")
        rotated_query_groups = fwht_last_dim(query.reshape(header.num_groups, header.group_size))
        logits = np.zeros(header.token_count, dtype=np.float32)
        centroids = np.asarray(page.codebooks, dtype=np.float32)
        for group_index in range(header.num_groups):
            words = load_group_words(page, group_index)
            codes_u8 = unpack_bits(words, header.bits, header.group_size).astype(np.int64, copy=False)
            corrected = centroids[codes_u8] * page.scales[:, group_index].astype(np.float32)[:, None]
            logits += corrected @ rotated_query_groups[group_index]
        return logits

    if page.payload is None:
        raise ValueError(f"{header.mode_default} page is missing payload")

    query_groups = query.reshape(header.num_groups, header.group_size)
    query_group_sums = query_groups.sum(axis=-1)
    logits = np.zeros(header.token_count, dtype=np.float32)

    for group_index in range(header.num_groups):
        words = load_group_words(page, group_index)
        codes_u8 = unpack_bits(words, header.bits, header.group_size)
        qg = query_groups[group_index]
        if header.mode_default == "M1":
            if page.codebooks is None:
                raise ValueError("M1 page is missing codebooks")
            group = dequantize_group_lut(codes_u8, codebook=np.asarray(page.codebooks[group_index], dtype=np.float32))
            logits += group @ qg
            continue

        if page.scales is None:
            raise ValueError("M0 page is missing scales")
        codes = codes_u8.astype(np.float32)
        scales = page.scales[:, group_index].astype(np.float32)

        if header.quant_scheme == "affine":
            if page.bias is None:
                raise ValueError("affine pages require bias metadata")
            int_dot = codes @ qg
            bias = page.bias[:, group_index].astype(np.float32)
            logits += scales * int_dot + bias * query_group_sums[group_index]
            continue

        zero_point = (1 << (header.bits - 1)) - 1
        logits += scales * ((codes - zero_point) @ qg)

    return logits


def mix_page_ref(attn_weights: np.ndarray, page: EncodedPage, out_acc: np.ndarray | None = None) -> np.ndarray:
    header = page.header
    weights = np.asarray(attn_weights, dtype=np.float32)
    if weights.shape != (header.token_count,):
        raise ValueError("attn_weights must have shape [token_count]")

    output = np.zeros(header.padded_head_dim, dtype=np.float32) if out_acc is None else np.asarray(out_acc, dtype=np.float32)
    if output.shape != (header.padded_head_dim,):
        raise ValueError("out_acc must have shape [padded_head_dim]")

    if header.mode_default == "M3":
        if page.escape_payload is None:
            raise ValueError("escape payload is missing")
        dense = decode_escape_payload(page.escape_payload, head_dim=header.head_dim, scales=page.escape_scales)
        output[: header.head_dim] += weights @ dense
        return output[: header.head_dim].copy()

    if header.mode_default in {"M2", "M4"}:
        raise ValueError(f"{header.mode_default} is only supported for key scoring in this phase")

    if header.mode_default == "T3":
        if page.payload is None or page.scales is None or page.codebooks is None:
            raise ValueError("T3 page is missing payload or correction metadata")
        centroids = np.asarray(page.codebooks, dtype=np.float32)
        for group_index in range(header.num_groups):
            words = load_group_words(page, group_index)
            codes_u8 = unpack_bits(words, header.bits, header.group_size).astype(np.int64, copy=False)
            rotated_group = centroids[codes_u8] * page.scales[:, group_index].astype(np.float32)[:, None]
            group = fwht_last_dim(rotated_group)
            start = group_index * header.group_size
            end = start + header.group_size
            output[start:end] += weights @ group
        return output[: header.head_dim].copy()

    if page.payload is None:
        raise ValueError(f"{header.mode_default} page is missing payload")

    for group_index in range(header.num_groups):
        words = load_group_words(page, group_index)
        codes_u8 = unpack_bits(words, header.bits, header.group_size)

        if header.mode_default == "M1":
            if page.codebooks is None:
                raise ValueError("M1 page is missing codebooks")
            group = dequantize_group_lut(codes_u8, codebook=np.asarray(page.codebooks[group_index], dtype=np.float32))
        else:
            if page.scales is None:
                raise ValueError("M0 page is missing scales")
            codes = codes_u8.astype(np.float32)
            scales = page.scales[:, group_index].astype(np.float32)[:, None]

            if header.quant_scheme == "affine":
                if page.bias is None:
                    raise ValueError("affine pages require bias metadata")
                group = scales * codes + page.bias[:, group_index].astype(np.float32)[:, None]
            else:
                zero_point = (1 << (header.bits - 1)) - 1
                group = scales * (codes - zero_point)

        start = group_index * header.group_size
        end = start + header.group_size
        output[start:end] += weights @ group

    return output[: header.head_dim].copy()


def explicit_dequantized_score(query_slice: np.ndarray, page: EncodedPage) -> np.ndarray:
    dense = decode_page(page)
    query = np.asarray(query_slice, dtype=np.float32)
    return dense @ query


def explicit_dequantized_mix(attn_weights: np.ndarray, page: EncodedPage) -> np.ndarray:
    dense = decode_page(page)
    weights = np.asarray(attn_weights, dtype=np.float32)
    return weights @ dense


def run_attention_reference(query_slice: np.ndarray, key_page: EncodedPage, value_page: EncodedPage) -> tuple[np.ndarray, np.ndarray]:
    logits = score_page_ref(query_slice, key_page)
    weights = softmax(logits)
    output = mix_page_ref(weights, value_page)
    return logits, output


def explicit_dequantized_attention(query_slice: np.ndarray, key_page: EncodedPage, value_page: EncodedPage) -> tuple[np.ndarray, np.ndarray]:
    logits = explicit_dequantized_score(query_slice, key_page)
    weights = softmax(logits)
    output = explicit_dequantized_mix(weights, value_page)
    return logits, output
