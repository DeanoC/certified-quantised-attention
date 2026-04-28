from __future__ import annotations

import numpy as np


def encode_escape_payload(values: np.ndarray, dtype: str = "float16") -> np.ndarray:
    return np.asarray(values, dtype=np.dtype(dtype))


def encode_escape_storage(values: np.ndarray, dtype: str = "float16") -> tuple[np.ndarray, np.ndarray | None]:
    array = np.asarray(values, dtype=np.float32)
    if dtype in {"float16", "float32"}:
        return np.asarray(array, dtype=np.dtype(dtype)), None
    if dtype == "int8":
        row_absmax = np.max(np.abs(array), axis=1)
        scales = np.maximum(row_absmax / 127.0, 1e-8).astype(np.float16, copy=False)
        quantized = np.clip(np.rint(array / scales[:, None]), -127.0, 127.0).astype(np.int8, copy=False)
        return quantized, scales
    raise ValueError(f"unsupported escape dtype: {dtype}")


def decode_escape_payload(
    payload: np.ndarray,
    *,
    head_dim: int | None = None,
    scales: np.ndarray | None = None,
) -> np.ndarray:
    array = np.asarray(payload)
    if array.dtype == np.int8:
        if scales is None:
            raise ValueError("int8 escape payloads require scales")
        decoded = array.astype(np.float32) * np.asarray(scales, dtype=np.float32)[:, None]
    else:
        decoded = np.asarray(array, dtype=np.float32)
    if head_dim is None:
        return decoded
    return decoded[:, :head_dim]
