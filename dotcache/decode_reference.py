from __future__ import annotations

import numpy as np

from .modes.m0_affine import dequantize_group
from .modes.m1_lut import dequantize_group_lut
from .modes.m2_key_sketch import reconstruct_group_m2
from .modes.m4_key_project import reconstruct_group_m4
from .modes.m3_escape import decode_escape_payload
from .modes.turbo3 import dequantize_group_turbo3
from .page_format import load_group_words
from .packing import unpack_bits
from .types import EncodedPage


def decode_group_ref(page: EncodedPage, group_index: int) -> np.ndarray:
    page.record_group_decode()
    header = page.header

    if header.mode_default == "M3":
        if page.escape_payload is None:
            raise ValueError("escape payload is missing")
        start = group_index * header.group_size
        end = start + header.group_size
        return decode_escape_payload(page.escape_payload, scales=page.escape_scales)[:, start:end]

    if header.mode_default == "M2":
        if page.m2_sketch is None or page.m2_basis is None:
            raise ValueError("M2 page is missing sketch payload")
        return reconstruct_group_m2(
            page.m2_sketch[:, group_index, :],
            basis=page.m2_basis[group_index],
            mean=None if page.m2_mean is None else page.m2_mean[group_index],
        )
    if header.mode_default == "M4":
        if page.m2_sketch is None or page.m2_mean is None:
            raise ValueError("M4 page is missing projected payload")
        return reconstruct_group_m4(
            page.m2_sketch[:, group_index, :],
            mean=page.m2_mean[group_index],
            group_size=header.group_size,
            basis_family=header.project_basis,
            basis=None if page.m2_basis is None else page.m2_basis[group_index],
        )

    words = load_group_words(page, group_index)
    codes = unpack_bits(words, header.bits, header.group_size)
    if header.mode_default == "M1":
        if page.codebooks is None:
            raise ValueError("M1 page is missing codebooks")
        codebook = np.asarray(page.codebooks[group_index], dtype=np.float32)
        return dequantize_group_lut(codes, codebook=codebook)
    if header.mode_default == "T3":
        if page.scales is None or page.codebooks is None:
            raise ValueError("T3 page is missing correction metadata")
        correction = page.scales[:, group_index].astype(np.float32)
        return dequantize_group_turbo3(
            codes,
            correction=correction,
            centroids=np.asarray(page.codebooks, dtype=np.float32),
        )
    if page.payload is None or page.scales is None:
        raise ValueError("M0 page is missing payload or scales")
    scales = page.scales[:, group_index].astype(np.float32)[:, None]
    bias = None
    if page.bias is not None:
        bias = page.bias[:, group_index].astype(np.float32)[:, None]
    return dequantize_group(
        codes,
        scales=scales,
        bias=bias,
        bits=header.bits,
        scheme=header.quant_scheme,
    )


def decode_page(page: EncodedPage) -> np.ndarray:
    page.record_full_decode()
    header = page.header

    if header.mode_default == "M3":
        if page.escape_payload is None:
            raise ValueError("escape payload is missing")
        return decode_escape_payload(page.escape_payload, head_dim=header.head_dim, scales=page.escape_scales)

    groups = [decode_group_ref(page, group_index) for group_index in range(header.num_groups)]
    full = np.concatenate(groups, axis=-1)
    return full[:, : header.head_dim]
