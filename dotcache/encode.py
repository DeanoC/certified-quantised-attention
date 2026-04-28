from __future__ import annotations

import numpy as np

from .config import DotCacheConfig
from .planner import PageModeSpec
from .modes.m0_affine import quantize_tensor
from .modes.m1_lut import quantize_tensor_lut
from .modes.m2_key_sketch import quantize_tensor_m2, reconstruct_group_m2
from .modes.m4_key_project import quantize_tensor_m4, reconstruct_group_m4
from .modes.m3_escape import encode_escape_storage
from .modes.turbo3 import quantize_tensor_turbo3
from .page_format import build_payload
from .packing import words_per_group
from .types import EncodedPage, Kind, PageHeader

DEFAULT_RUNTIME_SKETCH_ROWS = 4


def _reconstruct_lut_page(codes: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    token_count, num_groups, group_size = codes.shape
    dense = np.zeros((token_count, num_groups * group_size), dtype=np.float32)
    for group_index in range(num_groups):
        start = group_index * group_size
        end = start + group_size
        group_codebook = codebooks[group_index].astype(np.float32)
        if group_codebook.ndim == 1:
            dense[:, start:end] = group_codebook[codes[:, group_index].astype(np.int64)]
        else:
            segment_count = group_codebook.shape[0]
            segment_ids = (np.arange(token_count, dtype=np.int64) * segment_count) // max(token_count, 1)
            dense[:, start:end] = group_codebook[segment_ids[:, None], codes[:, group_index].astype(np.int64)]
    return dense


def _reconstruct_m2_page(coeffs: np.ndarray, basis: np.ndarray, mean: np.ndarray | None, *, group_size: int) -> np.ndarray:
    token_count, num_groups, _ = coeffs.shape
    dense = np.zeros((token_count, num_groups * group_size), dtype=np.float32)
    for group_index in range(num_groups):
        start = group_index * group_size
        end = start + group_size
        dense[:, start:end] = reconstruct_group_m2(
            coeffs[:, group_index, :],
            basis=basis[group_index],
            mean=None if mean is None else mean[group_index],
        )
    return dense


def _reconstruct_m4_page(coeffs: np.ndarray, mean: np.ndarray, *, group_size: int) -> np.ndarray:
    token_count, num_groups, _ = coeffs.shape
    dense = np.zeros((token_count, num_groups * group_size), dtype=np.float32)
    for group_index in range(num_groups):
        start = group_index * group_size
        end = start + group_size
        dense[:, start:end] = reconstruct_group_m4(
            coeffs[:, group_index, :],
            mean=mean[group_index],
            group_size=group_size,
        )
    return dense


def _build_runtime_page_sketch(values: np.ndarray, *, sketch_rows: int = DEFAULT_RUNTIME_SKETCH_ROWS) -> tuple[np.ndarray, np.ndarray]:
    rows = min(max(1, sketch_rows), values.shape[0])
    chunks = np.array_split(values, rows, axis=0)
    sketch = np.stack([chunk.mean(axis=0) for chunk in chunks], axis=0).astype(np.float16)
    page_mean = values.mean(axis=0).astype(np.float16)
    return page_mean, sketch


def _build_runtime_page_envelope(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    page_min = values.min(axis=0).astype(np.float16)
    page_max = values.max(axis=0).astype(np.float16)
    return page_min, page_max


def _candidate_m2_segment_counts(max_segment_count: int) -> list[int]:
    max_count = max(1, int(max_segment_count))
    counts = [1]
    candidate = 2
    while candidate < max_count:
        counts.append(candidate)
        candidate *= 2
    if max_count not in counts:
        counts.append(max_count)
    return counts


def _encode_m2_tensor(values: np.ndarray, config: DotCacheConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    best_coeffs, best_basis, best_mean, padded_head_dim = quantize_tensor_m2(
        values,
        group_size=config.group_size,
        sketch_dim=config.m2_sketch_dim_k,
        center=config.m2_center_k,
        segment_count=1 if config.m2_adaptive_segments_k else config.m2_segment_count_k,
    )
    if not config.m2_adaptive_segments_k or config.m2_segment_count_k <= 1:
        return best_coeffs, best_basis, best_mean, padded_head_dim

    baseline = _reconstruct_m2_page(best_coeffs, best_basis, best_mean, group_size=config.group_size)[:, : config.head_dim]
    rms = float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))
    best_error = float(np.mean(np.abs(values - baseline), dtype=np.float64) / max(rms, 1e-6))

    for segment_count in _candidate_m2_segment_counts(config.m2_segment_count_k)[1:]:
        coeffs, basis, mean, padded_head_dim = quantize_tensor_m2(
            values,
            group_size=config.group_size,
            sketch_dim=config.m2_sketch_dim_k,
            center=config.m2_center_k,
            segment_count=segment_count,
        )
        reconstructed = _reconstruct_m2_page(coeffs, basis, mean, group_size=config.group_size)[:, : config.head_dim]
        trial_error = float(np.mean(np.abs(values - reconstructed), dtype=np.float64) / max(rms, 1e-6))
        if (best_error - trial_error) / max(best_error, 1e-6) >= config.m2_adaptive_min_improvement_k:
            best_coeffs, best_basis, best_mean = coeffs, basis, mean
            best_error = trial_error

    return best_coeffs, best_basis, best_mean, padded_head_dim


def encode_page(
    tensor_slice: np.ndarray,
    config: DotCacheConfig,
    *,
    kind: Kind,
    layer_id: int = 0,
    kv_head_id: int = 0,
    token_start: int = 0,
    mode: str | None = None,
    page_mode: PageModeSpec | None = None,
    layout: str | None = None,
    quant_scheme: str | None = None,
    build_runtime_metadata: bool = True,
    build_m2_sidecar: bool | None = None,
    m4_basis_override: np.ndarray | None = None,
) -> EncodedPage:
    values = np.asarray(tensor_slice, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("tensor_slice must have shape [token_count, head_dim]")
    if values.shape[1] != config.head_dim:
        raise ValueError("tensor_slice head_dim must match config.head_dim")

    bits = config.bits_k if kind == "K" else config.bits_v
    default_mode = config.default_mode_k if kind == "K" else config.default_mode_v
    selected_page_mode = page_mode
    page_mode_name = selected_page_mode.mode if selected_page_mode is not None else (mode or default_mode)
    if selected_page_mode is not None:
        bits = int(selected_page_mode.bits)
    page_layout = layout or (config.payload_layout_k if kind == "K" else config.payload_layout_v)
    scheme = (
        selected_page_mode.quant_scheme
        if selected_page_mode is not None
        else quant_scheme or (config.quant_scheme_k if kind == "K" else config.quant_scheme_v)
    )
    token_count = values.shape[0]
    requested_mode = page_mode_name
    trial_quant_error = None
    runtime_page_mean = None
    runtime_page_sketch = None
    runtime_page_min = None
    runtime_page_max = None
    if build_runtime_metadata:
        runtime_page_mean, runtime_page_sketch = _build_runtime_page_sketch(values)
        runtime_page_min, runtime_page_max = _build_runtime_page_envelope(values)

    def _build_m2_sidecar() -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        sidecar_enabled = config.m2_prefilter_top_k > 0 if build_m2_sidecar is None else bool(build_m2_sidecar)
        if kind != "K" or not sidecar_enabled:
            return None, None, None
        coeffs, basis, mean, _ = _encode_m2_tensor(values, config)
        return (
            coeffs.astype(np.float16, copy=False),
            basis.astype(np.float16, copy=False),
            mean.astype(np.float16, copy=False),
        )

    header_kwargs = {
        "policy_id": selected_page_mode.policy_id if selected_page_mode is not None else "exact_baseline",
        "sensitivity_tier": selected_page_mode.sensitivity_tier if selected_page_mode is not None else "exact",
        "fallback_reason": selected_page_mode.fallback_reason if selected_page_mode is not None else "",
        "age_bucket": selected_page_mode.age_bucket if selected_page_mode is not None else "aged",
    }

    if page_mode_name == "M3":
        escape_dtype = (
            selected_page_mode.escape_dtype
            if selected_page_mode is not None and selected_page_mode.escape_dtype is not None
            else config.escape_dtype
        )
        header = PageHeader(
            layer_id=layer_id,
            kv_head_id=kv_head_id,
            kind=kind,
            token_start=token_start,
            token_count=token_count,
            head_dim=config.head_dim,
            padded_head_dim=config.padded_head_dim,
            group_size=config.group_size,
            num_groups=config.num_groups,
            bits=bits,
            words_per_group=words_per_group(config.group_size, bits),
            mode_default="M3",
            layout=page_layout,
            quant_scheme=scheme,
            **header_kwargs,
            escape_dtype=escape_dtype,
        )
        escape_payload, escape_scales = encode_escape_storage(values, dtype=escape_dtype)
        return EncodedPage(
            header=header,
            escape_payload=escape_payload,
            escape_scales=escape_scales,
            requested_mode=page_mode,
            runtime_page_mean=runtime_page_mean,
            runtime_page_sketch=runtime_page_sketch,
            runtime_page_min=runtime_page_min,
            runtime_page_max=runtime_page_max,
        )

    trial_token_p95_error = None

    if page_mode_name == "M2":
        if kind != "K":
            raise ValueError("M2 is only supported for K pages in this phase")
        coeffs, basis, mean, padded_head_dim = _encode_m2_tensor(values, config)
        header = PageHeader(
            layer_id=layer_id,
            kv_head_id=kv_head_id,
            kind=kind,
            token_start=token_start,
            token_count=token_count,
            head_dim=config.head_dim,
            padded_head_dim=padded_head_dim,
            group_size=config.group_size,
            num_groups=config.num_groups,
            bits=bits,
            words_per_group=0,
            mode_default="M2",
            layout=page_layout,
            quant_scheme="sketch",
            **header_kwargs,
            escape_dtype=config.escape_dtype,
        )
        return EncodedPage(
            header=header,
            m2_sketch=coeffs.astype(np.float16, copy=False),
            m2_basis=basis.astype(np.float16, copy=False),
            m2_mean=mean.astype(np.float16, copy=False),
            requested_mode=page_mode,
            runtime_page_mean=runtime_page_mean,
            runtime_page_sketch=runtime_page_sketch,
            runtime_page_min=runtime_page_min,
            runtime_page_max=runtime_page_max,
        )

    if page_mode_name == "M4":
        if kind != "K":
            raise ValueError("M4 is only supported for K pages in this phase")
        coeffs, basis, mean, padded_head_dim = quantize_tensor_m4(
            values,
            group_size=config.group_size,
            project_dim=config.resolve_m4_project_dim_k(layer_id=layer_id),
            basis_family=config.resolve_m4_project_basis_k(layer_id=layer_id),
            basis_override=m4_basis_override,
        )
        header = PageHeader(
            layer_id=layer_id,
            kv_head_id=kv_head_id,
            kind=kind,
            token_start=token_start,
            token_count=token_count,
            head_dim=config.head_dim,
            padded_head_dim=padded_head_dim,
            group_size=config.group_size,
            num_groups=config.num_groups,
            bits=bits,
            words_per_group=0,
            mode_default="M4",
            layout=page_layout,
            quant_scheme="project",
            project_basis=config.resolve_m4_project_basis_k(layer_id=layer_id),
            **header_kwargs,
            escape_dtype=config.escape_dtype,
        )
        return EncodedPage(
            header=header,
            m2_sketch=coeffs.astype(np.float16, copy=False),
            m2_basis=None if basis is None else basis.astype(np.float16, copy=False),
            m2_mean=mean.astype(np.float16, copy=False),
            requested_mode=page_mode,
            runtime_page_mean=runtime_page_mean,
            runtime_page_sketch=runtime_page_sketch,
            runtime_page_min=runtime_page_min,
            runtime_page_max=runtime_page_max,
        )

    if page_mode_name == "M1":
        codes, codebooks, padded_head_dim = quantize_tensor_lut(
            values,
            group_size=config.group_size,
            bits=bits,
            segment_count=config.m1_segment_count_k if kind == "K" else config.m1_segment_count_v,
            refine_steps=config.lut_refine_steps,
            preconditioner=config.preconditioner,
            precondition_strength=config.precondition_strength,
        )
        if config.m1_fallback_to_m0:
            reconstructed = _reconstruct_lut_page(codes, codebooks)[:, : config.head_dim]
            rms = float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))
            trial_quant_error = float(np.mean(np.abs(values - reconstructed), dtype=np.float64) / max(rms, 1e-6))
            token_norms = np.linalg.norm(values, axis=1)
            token_rel_error = np.linalg.norm(values - reconstructed, axis=1) / np.maximum(token_norms, 1e-6)
            trial_token_p95_error = float(np.percentile(token_rel_error, 95))
            if (
                trial_quant_error > config.m1_error_threshold
                or trial_token_p95_error > config.m1_token_p95_error_threshold
            ):
                page_mode_name = "M0"
                scheme = "affine"
        if page_mode_name == "M1":
            sidecar_sketch, sidecar_basis, sidecar_mean = _build_m2_sidecar()
            payload = build_payload(codes, bits, page_layout)
            header = PageHeader(
                layer_id=layer_id,
                kv_head_id=kv_head_id,
                kind=kind,
                token_start=token_start,
                token_count=token_count,
                head_dim=config.head_dim,
                padded_head_dim=padded_head_dim,
                group_size=config.group_size,
                num_groups=config.num_groups,
                bits=bits,
                words_per_group=words_per_group(config.group_size, bits),
                mode_default="M1",
                layout=page_layout,
                quant_scheme="lut",
                **header_kwargs,
                escape_dtype=config.escape_dtype,
            )
            return EncodedPage(
                header=header,
                payload=payload,
                codebooks=codebooks.astype(np.float16),
                m2_sketch=sidecar_sketch,
                m2_basis=sidecar_basis,
                m2_mean=sidecar_mean,
                lut_segment_count=int(codebooks.shape[1]) if codebooks.ndim == 3 else 1,
                requested_mode=requested_mode,
                trial_quant_error=trial_quant_error,
                trial_token_p95_error=trial_token_p95_error,
                runtime_page_mean=runtime_page_mean,
                runtime_page_sketch=runtime_page_sketch,
                runtime_page_min=runtime_page_min,
                runtime_page_max=runtime_page_max,
        )

    if page_mode_name == "T3":
        codes, correction, centroids, padded_head_dim = quantize_tensor_turbo3(
            values,
            group_size=config.group_size,
        )
        sidecar_sketch, sidecar_basis, sidecar_mean = _build_m2_sidecar()
        payload = build_payload(codes, 3, page_layout)
        header = PageHeader(
            layer_id=layer_id,
            kv_head_id=kv_head_id,
            kind=kind,
            token_start=token_start,
            token_count=token_count,
            head_dim=config.head_dim,
            padded_head_dim=padded_head_dim,
            group_size=config.group_size,
            num_groups=config.num_groups,
            bits=3,
            words_per_group=words_per_group(config.group_size, 3),
            mode_default="T3",
            layout=page_layout,
            quant_scheme="turbo3",
            **header_kwargs,
            escape_dtype=config.escape_dtype,
        )
        return EncodedPage(
            header=header,
            payload=payload,
            scales=correction,
            codebooks=centroids,
            m2_sketch=sidecar_sketch,
            m2_basis=sidecar_basis,
            m2_mean=sidecar_mean,
            requested_mode=requested_mode,
            runtime_page_mean=runtime_page_mean,
            runtime_page_sketch=runtime_page_sketch,
            runtime_page_min=runtime_page_min,
            runtime_page_max=runtime_page_max,
        )

    if page_mode_name != "M0":
        raise ValueError("only M0, M1, M2, M3, M4, and T3 are supported in this bootstrap")

    codes, scales, bias, padded_head_dim = quantize_tensor(
        values,
        group_size=config.group_size,
        bits=bits,
        scheme=scheme,
    )
    payload = build_payload(codes, bits, page_layout)
    header = PageHeader(
        layer_id=layer_id,
        kv_head_id=kv_head_id,
        kind=kind,
        token_start=token_start,
        token_count=token_count,
        head_dim=config.head_dim,
        padded_head_dim=padded_head_dim,
        group_size=config.group_size,
        num_groups=config.num_groups,
        bits=bits,
        words_per_group=words_per_group(config.group_size, bits),
        mode_default="M0",
        layout=page_layout,
        quant_scheme=scheme,
        **header_kwargs,
        escape_dtype=config.escape_dtype,
    )
    stored_scales = scales.astype(np.float16)
    stored_bias = None if bias is None else bias.astype(np.float16)
    sidecar_sketch, sidecar_basis, sidecar_mean = _build_m2_sidecar()
    return EncodedPage(
        header=header,
        payload=payload,
        scales=stored_scales,
        bias=stored_bias,
        m2_sketch=sidecar_sketch,
        m2_basis=sidecar_basis,
        m2_mean=sidecar_mean,
        requested_mode=requested_mode,
        trial_quant_error=trial_quant_error,
        trial_token_p95_error=trial_token_p95_error if "trial_token_p95_error" in locals() else None,
        runtime_page_mean=runtime_page_mean,
        runtime_page_sketch=runtime_page_sketch,
        runtime_page_min=runtime_page_min,
        runtime_page_max=runtime_page_max,
    )
