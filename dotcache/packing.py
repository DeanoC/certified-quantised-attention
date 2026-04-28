from __future__ import annotations

import numpy as np


def words_per_group(group_size: int, bits: int) -> int:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if bits <= 0 or bits > 16:
        raise ValueError("bits must be between 1 and 16")
    return (group_size * bits + 31) // 32


def pack_bits(codes: np.ndarray, bits: int) -> np.ndarray:
    values = np.asarray(codes, dtype=np.uint32)
    if values.ndim == 0:
        raise ValueError("codes must have at least one dimension")

    mask = (1 << bits) - 1
    if np.any(values > mask):
        raise ValueError("codes contain values that do not fit in the requested bit width")

    symbol_count = values.shape[-1]
    word_count = words_per_group(symbol_count, bits)
    flat = values.reshape(-1, symbol_count)
    if 32 % bits == 0:
        symbols_per_word = 32 // bits
        padded_symbol_count = word_count * symbols_per_word
        if padded_symbol_count != symbol_count:
            padded = np.zeros((flat.shape[0], padded_symbol_count), dtype=np.uint32)
            padded[:, :symbol_count] = flat
            flat = padded
        grouped = flat.reshape(flat.shape[0], word_count, symbols_per_word)
        shifts = (np.arange(symbols_per_word, dtype=np.uint32) * np.uint32(bits)).reshape(1, 1, symbols_per_word)
        packed = np.bitwise_or.reduce(grouped << shifts, axis=-1, dtype=np.uint32)
        return packed.reshape(*values.shape[:-1], word_count)

    packed = np.zeros((flat.shape[0], word_count), dtype=np.uint32)
    for symbol_index in range(symbol_count):
        bit_offset = symbol_index * bits
        word_index = bit_offset // 32
        bit_index = bit_offset % 32
        values_col = flat[:, symbol_index] & np.uint32(mask)
        packed[:, word_index] |= np.left_shift(values_col, np.uint32(bit_index), dtype=np.uint32)
        spill = bit_index + bits - 32
        if spill > 0:
            packed[:, word_index + 1] |= np.right_shift(values_col, np.uint32(bits - spill), dtype=np.uint32)
    return packed.reshape(*values.shape[:-1], word_count)


def unpack_bits(words: np.ndarray, bits: int, group_size: int) -> np.ndarray:
    packed = np.asarray(words, dtype=np.uint32)
    if packed.ndim == 0:
        raise ValueError("words must have at least one dimension")

    expected_words = words_per_group(group_size, bits)
    if packed.shape[-1] != expected_words:
        raise ValueError("word count does not match group_size and bits")

    flat = packed.reshape(-1, expected_words)
    mask = np.uint32((1 << bits) - 1)
    if 32 % bits == 0:
        symbols_per_word = 32 // bits
        shifts = (np.arange(symbols_per_word, dtype=np.uint32) * np.uint32(bits)).reshape(1, 1, symbols_per_word)
        expanded = ((flat[:, :, None] >> shifts) & mask).reshape(flat.shape[0], expected_words * symbols_per_word)
        unpacked = expanded[:, :group_size].astype(np.uint8, copy=False)
        return unpacked.reshape(*packed.shape[:-1], group_size)

    unpacked = np.zeros((flat.shape[0], group_size), dtype=np.uint8)
    mask_int = int(mask)
    for symbol_index in range(group_size):
        bit_offset = symbol_index * bits
        word_index = bit_offset // 32
        bit_index = bit_offset % 32
        values = np.right_shift(flat[:, word_index], np.uint32(bit_index)).astype(np.uint32, copy=False)
        spill = bit_index + bits - 32
        if spill > 0:
            spill_bits = np.bitwise_and(flat[:, word_index + 1], np.uint32((1 << spill) - 1))
            values = np.bitwise_or(values, np.left_shift(spill_bits, np.uint32(bits - spill), dtype=np.uint32))
        unpacked[:, symbol_index] = np.bitwise_and(values, np.uint32(mask_int)).astype(np.uint8, copy=False)
    return unpacked.reshape(*packed.shape[:-1], group_size)
