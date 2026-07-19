"""Compact runtime capture for pre-transfer KV-cache geometry.

The capture path is deliberately opt-in.  Projectors only perform the detached
reductions in this module after :func:`configure_projector_cache_geometry` has
enabled them, so the normal forward path keeps its existing arithmetic and RNG
behaviour.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

import torch
from torch import Tensor


CACHE_GEOMETRY_SCHEMA_VERSION = 1
_EPS = 1e-12


@dataclass(frozen=True)
class CacheGeometryRuntimeContext:
    """Metadata active for one model call and its optional batch samples."""

    metadata: Mapping[str, Any]
    sample_contexts: Optional[tuple[Mapping[str, Any], ...]] = None


_CACHE_GEOMETRY_RUNTIME: ContextVar[Optional[CacheGeometryRuntimeContext]] = (
    ContextVar("cache_geometry_runtime", default=None)
)


def _normalize_metadata(metadata: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("cache geometry context must be a mapping")
    return dict(metadata)


@contextmanager
def cache_geometry_runtime(
    context: Optional[Mapping[str, Any]] = None,
    *,
    sample_contexts: Optional[Sequence[Mapping[str, Any]]] = None,
    **metadata: Any,
) -> Iterator[CacheGeometryRuntimeContext]:
    """Attach JSON-safe identity metadata to geometry captured in this scope.

    Scopes are nestable.  An evaluator can wrap the whole model call with sample
    identity while a wrapper adds source/target layer metadata around an
    individual projector call.  Explicit sample metadata has final precedence.
    """

    parent = _CACHE_GEOMETRY_RUNTIME.get()
    merged_metadata = dict(parent.metadata) if parent is not None else {}
    merged_metadata.update(_normalize_metadata(context))
    merged_metadata.update(metadata)

    parent_samples = None if parent is None else parent.sample_contexts
    if sample_contexts is None:
        merged_samples = parent_samples
    else:
        normalized_samples = tuple(
            _normalize_metadata(sample_context) for sample_context in sample_contexts
        )
        if parent_samples is None:
            merged_samples = normalized_samples
        else:
            if len(parent_samples) != len(normalized_samples):
                raise ValueError(
                    "nested cache geometry sample_contexts must have matching lengths"
                )
            merged_samples = tuple(
                {**parent_sample, **sample}
                for parent_sample, sample in zip(parent_samples, normalized_samples)
            )

    runtime = CacheGeometryRuntimeContext(
        metadata=merged_metadata,
        sample_contexts=merged_samples,
    )
    token = _CACHE_GEOMETRY_RUNTIME.set(runtime)
    try:
        yield runtime
    finally:
        _CACHE_GEOMETRY_RUNTIME.reset(token)


def current_cache_geometry_runtime() -> Optional[CacheGeometryRuntimeContext]:
    """Return the active runtime context, if capture is inside one."""

    return _CACHE_GEOMETRY_RUNTIME.get()


def _projectors(model_or_projectors: Any) -> list[Any]:
    projector_list = getattr(model_or_projectors, "projector_list", None)
    if projector_list is not None:
        return list(projector_list or [])
    if isinstance(model_or_projectors, (list, tuple)):
        return list(model_or_projectors)
    if isinstance(model_or_projectors, Iterable) and not isinstance(
        model_or_projectors, (str, bytes, Mapping)
    ):
        return list(model_or_projectors)
    return [model_or_projectors]


def _target_layer_by_projector(model: Any, projector_count: int) -> list[int]:
    mapping = list(range(projector_count))
    projector_dict = getattr(model, "projector_dict", None)
    if not isinstance(projector_dict, Mapping):
        return mapping
    for target_sources in projector_dict.values():
        if not isinstance(target_sources, Mapping):
            continue
        for target_layers in target_sources.values():
            if not isinstance(target_layers, Mapping):
                continue
            for target_layer, entries in target_layers.items():
                try:
                    target_layer_index = int(target_layer)
                except (TypeError, ValueError):
                    continue
                if isinstance(entries, tuple):
                    entries = [entries]
                if not isinstance(entries, Sequence):
                    continue
                for entry in entries:
                    if not isinstance(entry, Sequence) or len(entry) != 2:
                        continue
                    try:
                        projector_index = int(entry[1])
                    except (TypeError, ValueError):
                        continue
                    if 0 <= projector_index < projector_count:
                        mapping[projector_index] = target_layer_index
    return mapping


def configure_projector_cache_geometry(model: Any, enabled: bool) -> dict[str, Any]:
    """Enable or disable compact geometry capture independently of gate diagnostics."""

    projectors = _projectors(model)
    target_layers = _target_layer_by_projector(model, len(projectors))
    for projector_index, (projector, target_layer) in enumerate(
        zip(projectors, target_layers)
    ):
        projector.capture_cache_geometry = bool(enabled)
        projector._cache_geometry_projector_index = projector_index
        projector._cache_geometry_target_layer = target_layer
        records = getattr(projector, "cache_geometry_records", None)
        if isinstance(records, list):
            records.clear()
        else:
            projector.cache_geometry_records = []
    return {
        "enabled": bool(enabled),
        "projector_count": len(projectors),
        "target_layer_by_projector": target_layers,
    }


def clear_projector_cache_geometry_records(model_or_projectors: Any) -> None:
    """Drop captured records so a skipped call cannot reuse a prior sample."""

    for projector in _projectors(model_or_projectors):
        records = getattr(projector, "cache_geometry_records", None)
        if isinstance(records, list):
            records.clear()


def consume_cache_geometry_records(model_or_projectors: Any) -> list[dict[str, Any]]:
    """Return all current records and clear their projector-owned buffers."""

    output: list[dict[str, Any]] = []
    for projector in _projectors(model_or_projectors):
        records = getattr(projector, "cache_geometry_records", None)
        if not isinstance(records, list):
            continue
        output.extend(records)
        records.clear()
    return output


def _flatten_per_sample(tensor: Tensor) -> Tensor:
    return tensor.detach().to(dtype=torch.float32).reshape(tensor.shape[0], -1)


def _norm_per_sample(tensor: Tensor) -> Tensor:
    return torch.linalg.vector_norm(_flatten_per_sample(tensor), dim=-1)


def _ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
    return numerator / denominator.clamp_min(_EPS)


def _cosine_per_sample(left: Tensor, right: Tensor) -> Tensor:
    left_flat = _flatten_per_sample(left)
    right_flat = _flatten_per_sample(right)
    numerator = (left_flat * right_flat).sum(dim=-1)
    denominator = torch.linalg.vector_norm(left_flat, dim=-1) * torch.linalg.vector_norm(
        right_flat, dim=-1
    )
    cosine = torch.where(
        denominator > _EPS,
        numerator / denominator.clamp_min(_EPS),
        torch.zeros_like(numerator),
    )
    return cosine.clamp(-1.0, 1.0)


def _distribution_stats(tensor: Tensor) -> dict[str, Tensor]:
    flat = _flatten_per_sample(tensor)
    return {
        "mean": flat.mean(dim=-1),
        "std": flat.std(dim=-1, unbiased=False),
        "min": flat.min(dim=-1).values,
        "max": flat.max(dim=-1).values,
    }


def _combined_stats(left: Tensor, right: Tensor) -> dict[str, Tensor]:
    flat = torch.cat([_flatten_per_sample(left), _flatten_per_sample(right)], dim=-1)
    return {
        "mean": flat.mean(dim=-1),
        "std": flat.std(dim=-1, unbiased=False),
    }


def _component_metrics(
    *,
    prefix: str,
    native: Tensor,
    raw_projected: Tensor,
    fused: Tensor,
    weight: Tensor,
    confidence: Tensor,
    effective_gate: Tensor,
) -> dict[str, Tensor]:
    residual = fused.detach() - native.detach()
    native_norm = _norm_per_sample(native)
    raw_projected_norm = _norm_per_sample(raw_projected)
    fused_norm = _norm_per_sample(fused)
    residual_norm = _norm_per_sample(residual)

    batch_size, num_heads = residual.shape[:2]
    residual_float = residual.to(dtype=torch.float32)
    native_float = native.detach().to(dtype=torch.float32)
    residual_head_norm = torch.linalg.vector_norm(
        residual_float.reshape(batch_size, num_heads, -1), dim=-1
    )
    native_head_norm = torch.linalg.vector_norm(
        native_float.reshape(batch_size, num_heads, -1), dim=-1
    )
    residual_head_ratio = _ratio(residual_head_norm, native_head_norm)
    residual_head_energy = residual_float.square().sum(dim=tuple(range(2, residual.dim())))
    total_residual_head_energy = residual_head_energy.sum(dim=-1)
    residual_head_energy_share = torch.where(
        total_residual_head_energy > 0,
        residual_head_energy
        / total_residual_head_energy.unsqueeze(-1).clamp_min(
            torch.finfo(residual_head_energy.dtype).tiny
        ),
        torch.zeros_like(residual_head_energy),
    )
    residual_head_energy_hhi = residual_head_energy_share.square().sum(dim=-1)
    residual_head_norm_mean = residual_head_norm.mean(dim=-1)
    residual_head_norm_std = residual_head_norm.std(dim=-1, unbiased=False)

    metrics: dict[str, Tensor] = {
        f"{prefix}_native_norm": native_norm,
        f"{prefix}_raw_projected_norm": raw_projected_norm,
        f"{prefix}_fused_norm": fused_norm,
        f"{prefix}_residual_norm": residual_norm,
        f"{prefix}_raw_projected_to_native_norm_ratio": _ratio(
            raw_projected_norm, native_norm
        ),
        f"{prefix}_fused_to_native_norm_ratio": _ratio(fused_norm, native_norm),
        f"{prefix}_residual_to_native_norm_ratio": _ratio(
            residual_norm, native_norm
        ),
        f"{prefix}_native_raw_projected_cosine": _cosine_per_sample(
            native, raw_projected
        ),
        f"{prefix}_native_fused_cosine": _cosine_per_sample(native, fused),
        f"{prefix}_raw_projected_fused_cosine": _cosine_per_sample(
            raw_projected, fused
        ),
        f"{prefix}_residual_head_norm_mean": residual_head_norm_mean,
        f"{prefix}_residual_head_norm_std": residual_head_norm_std,
        f"{prefix}_residual_head_norm_min": residual_head_norm.min(dim=-1).values,
        f"{prefix}_residual_head_norm_max": residual_head_norm.max(dim=-1).values,
        f"{prefix}_head_residual_cv": residual_head_norm_std
        / (residual_head_norm_mean + _EPS),
        f"{prefix}_residual_head_to_native_norm_ratio_mean": residual_head_ratio.mean(
            dim=-1
        ),
        f"{prefix}_residual_head_to_native_norm_ratio_std": residual_head_ratio.std(
            dim=-1, unbiased=False
        ),
        f"{prefix}_residual_head_to_native_norm_ratio_min": residual_head_ratio.min(
            dim=-1
        ).values,
        f"{prefix}_residual_head_to_native_norm_ratio_max": residual_head_ratio.max(
            dim=-1
        ).values,
        f"{prefix}_residual_head_energy_hhi": residual_head_energy_hhi,
    }
    for name, tensor in (
        ("weight", weight),
        ("confidence", confidence),
        ("effective_gate", effective_gate),
    ):
        for statistic, values in _distribution_stats(tensor).items():
            metrics[f"{prefix}_{name}_{statistic}"] = values
    return metrics


def _source_metrics(
    *,
    batch_size: int,
    source_weights: Optional[Tensor],
    source_confidence: Optional[Tensor],
    device: torch.device,
) -> dict[str, Optional[Tensor]]:
    """Summarize alignment inputs without retaining token/candidate tensors.

    ``valid_alignment_mass`` is the mean, over receiver-token rows, of the sum
    of nonnegative candidate weights. ``valid_alignment_coverage`` is the
    fraction of receiver-token rows with positive total candidate mass. They are
    intentionally equal when every covered row is normalized and non-fallback.
    """

    metrics: dict[str, Optional[Tensor]] = {
        "source_confidence_mean": None,
        "source_confidence_std": None,
        "source_confidence_min": None,
        "source_confidence_max": None,
        "source_weight_top1_mean": None,
        "source_weight_entropy_mean": None,
        "source_weight_hhi_mean": None,
        "valid_alignment_mass": None,
        "valid_alignment_coverage": None,
    }
    if source_confidence is not None:
        confidence = source_confidence.detach().to(device=device, dtype=torch.float32)
        if confidence.shape[0] != batch_size:
            raise ValueError("source_confidence batch size does not match projected KV")
        for statistic, values in _distribution_stats(confidence).items():
            metrics[f"source_confidence_{statistic}"] = values

    if source_weights is not None:
        weights = source_weights.detach().to(device=device, dtype=torch.float32)
        if weights.shape[0] != batch_size or weights.dim() < 2:
            raise ValueError("source_weights must preserve the projected KV batch axis")
        candidate_count = weights.shape[-1]
        rows = weights.reshape(batch_size, -1, candidate_count).clamp_min(0.0)
        row_mass = rows.sum(dim=-1)
        normalized = torch.where(
            row_mass.unsqueeze(-1) > _EPS,
            rows / row_mass.unsqueeze(-1).clamp_min(_EPS),
            torch.zeros_like(rows),
        )
        entropy = -(normalized * normalized.clamp_min(_EPS).log()).sum(dim=-1)
        if candidate_count > 1:
            entropy = entropy / math.log(candidate_count)
        else:
            entropy = torch.zeros_like(entropy)
        metrics["source_weight_top1_mean"] = normalized.max(dim=-1).values.mean(
            dim=-1
        )
        metrics["source_weight_entropy_mean"] = entropy.mean(dim=-1)
        metrics["source_weight_hhi_mean"] = normalized.square().sum(dim=-1).mean(
            dim=-1
        )
        metrics["valid_alignment_mass"] = row_mass.mean(dim=-1)
        metrics["valid_alignment_coverage"] = (row_mass > _EPS).float().mean(dim=-1)
    return metrics


def _sample_metadata(batch_index: int, batch_size: int) -> dict[str, Any]:
    runtime = _CACHE_GEOMETRY_RUNTIME.get()
    if runtime is None:
        return {}
    metadata = dict(runtime.metadata)
    if runtime.sample_contexts is not None:
        if len(runtime.sample_contexts) != batch_size:
            raise ValueError(
                "cache geometry sample_contexts length must match projector batch size"
            )
        metadata.update(runtime.sample_contexts[batch_index])
    return metadata


def _source_receiver_length_ratio(metadata: Mapping[str, Any]) -> Optional[float]:
    if metadata.get("source_receiver_length_ratio") is not None:
        return float(metadata["source_receiver_length_ratio"])
    source_length = metadata.get("source_length", metadata.get("source_token_count"))
    receiver_length = metadata.get(
        "receiver_length", metadata.get("receiver_token_count")
    )
    if source_length is None or receiver_length is None:
        return None
    receiver_length = float(receiver_length)
    if receiver_length <= 0:
        return None
    return float(source_length) / receiver_length


def capture_projector_cache_geometry(
    projector: Any,
    *,
    native_key: Tensor,
    native_value: Tensor,
    raw_projected_key: Tensor,
    raw_projected_value: Tensor,
    fused_key: Tensor,
    fused_value: Tensor,
    key_weight: Tensor,
    value_weight: Tensor,
    key_confidence: Tensor,
    value_confidence: Tensor,
    key_effective_gate: Tensor,
    value_effective_gate: Tensor,
    source_weights: Optional[Tensor] = None,
    source_confidence: Optional[Tensor] = None,
) -> None:
    """Reduce one final projector fusion to detached, per-sample scalar records."""

    if not getattr(projector, "capture_cache_geometry", False):
        return
    batch_size = int(fused_key.shape[0])
    if batch_size != int(fused_value.shape[0]):
        raise ValueError("fused key/value batch sizes must match")

    with torch.no_grad():
        tensor_metrics: dict[str, Optional[Tensor]] = {}
        tensor_metrics.update(
            _component_metrics(
                prefix="key",
                native=native_key,
                raw_projected=raw_projected_key,
                fused=fused_key,
                weight=key_weight,
                confidence=key_confidence,
                effective_gate=key_effective_gate,
            )
        )
        tensor_metrics.update(
            _component_metrics(
                prefix="value",
                native=native_value,
                raw_projected=raw_projected_value,
                fused=fused_value,
                weight=value_weight,
                confidence=value_confidence,
                effective_gate=value_effective_gate,
            )
        )
        for output_prefix, key_tensor, value_tensor in (
            ("learned_weight", key_weight, value_weight),
            ("alignment_confidence", key_confidence, value_confidence),
            ("effective_gate", key_effective_gate, value_effective_gate),
        ):
            for statistic, values in _combined_stats(key_tensor, value_tensor).items():
                tensor_metrics[f"{output_prefix}_{statistic}"] = values
        tensor_metrics.update(
            _source_metrics(
                batch_size=batch_size,
                source_weights=source_weights,
                source_confidence=source_confidence,
                device=fused_key.device,
            )
        )

        records = getattr(projector, "cache_geometry_records", None)
        if not isinstance(records, list):
            records = []
            projector.cache_geometry_records = records
        projector_index = int(getattr(projector, "_cache_geometry_projector_index", -1))
        target_layer = int(getattr(projector, "_cache_geometry_target_layer", -1))
        for batch_index in range(batch_size):
            metadata = _sample_metadata(batch_index, batch_size)
            record = dict(metadata)
            record.update(
                {
                    "cache_geometry_schema_version": CACHE_GEOMETRY_SCHEMA_VERSION,
                    "projector_index": projector_index,
                    "target_layer": target_layer,
                    "batch_index": batch_index,
                    "source_receiver_length_ratio": _source_receiver_length_ratio(
                        metadata
                    ),
                }
            )
            for name, values in tensor_metrics.items():
                record[name] = (
                    None
                    if values is None
                    else float(values[batch_index].detach().float().cpu().item())
                )
            records.append(record)
