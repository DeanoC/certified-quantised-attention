from __future__ import annotations

from typing import cast

import numpy as np

from .packing import pack_bits
from .types import EncodedPage, Layout, PageHeader


def serialize_header(header: PageHeader) -> bytes:
    return header.to_json().encode("utf-8")


def deserialize_header(payload: bytes) -> PageHeader:
    return PageHeader.from_json(payload.decode("utf-8"))


def build_payload(codes: np.ndarray, bits: int, layout: Layout) -> np.ndarray:
    if layout == "group_major":
        return pack_bits(np.transpose(codes, (1, 0, 2)), bits)

    return pack_bits(codes, bits)


def load_group_words(page: EncodedPage, group_index: int) -> np.ndarray:
    if page.payload is None:
        raise ValueError("M3 pages do not have packed payload")
    if group_index < 0 or group_index >= page.header.num_groups:
        raise IndexError("group_index out of range")
    if page.header.layout == "group_major":
        return cast(np.ndarray, page.payload[group_index])
    return cast(np.ndarray, page.payload[:, group_index, :])
