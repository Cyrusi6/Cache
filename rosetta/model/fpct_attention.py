from __future__ import annotations

"""Candidate sidecar and ambiguous-only packed attention for FPCT."""

from dataclasses import dataclass, replace
from math import sqrt
from typing import Iterable, Optional, Sequence

import torch
from torch import Tensor
from torch import nn


FPCT_OPERATORS = frozenset({"c_pre", "c_post", "f"})


def normalize_fpct_operator(operator: Optional[str]) -> Optional[str]:
    if operator is None or operator == "":
        return None
    normalized = str(operator).lower()
    if normalized not in FPCT_OPERATORS:
        raise ValueError(f"fpct_operator must be one of {sorted(FPCT_OPERATORS)}, got {operator!r}")
    return normalized


@dataclass(frozen=True)
class FPCTSidecarSegment:
    parent_start: int
    key: Tensor  # [B,Hkv,N,K,D]
    value: Tensor  # [B,Hkv,N,K,D]
    prior: Tensor  # [B,N,K]
    valid: Tensor  # [B,N,K]
    max_slots_hint: Optional[int] = None
    source_length_hint: Optional[int] = None
    prior_sha256: Optional[str] = None
    certified: bool = False
    parent_force_native: Optional[Tensor] = None  # [B,N], semantic hard-gate identity
    parent_equivalent: Optional[Tensor] = None  # [B,N], exact final parent identity

    def validate(self) -> None:
        if self.parent_start < 0:
            raise ValueError("parent_start must be non-negative")
        if self.key.ndim != 5 or self.value.shape != self.key.shape:
            raise ValueError("sidecar key/value must have matching [B,Hkv,N,K,D] shapes")
        if self.prior.shape != self.valid.shape or self.prior.shape != (
            self.key.shape[0], self.key.shape[2], self.key.shape[3]
        ):
            raise ValueError("sidecar prior/valid shape mismatch")
        if self.certified and self.prior.dtype != torch.float32:
            raise ValueError("FPCT canonical prior must be float32")
        if self.max_slots_hint is not None and self.max_slots_hint < 0:
            raise ValueError("max_slots_hint must be non-negative")
        if self.source_length_hint is not None and self.source_length_hint < 0:
            raise ValueError("source_length_hint must be non-negative")
        if self.certified and not self.prior_sha256:
            raise ValueError("certified FPCT sidecars require a prior SHA256")
        if self.parent_force_native is not None:
            if self.parent_force_native.shape != (
                self.key.shape[0], self.key.shape[2]
            ):
                raise ValueError("sidecar parent_force_native must be [B,N]")
            if self.parent_force_native.dtype != torch.bool:
                raise ValueError("sidecar parent_force_native must be boolean")
            if self.parent_force_native.device != self.key.device:
                raise ValueError(
                    "sidecar parent_force_native and candidate tensors must share a device"
                )
        if self.parent_equivalent is not None:
            if self.parent_equivalent.shape != (
                self.key.shape[0], self.key.shape[2]
            ):
                raise ValueError("sidecar parent_equivalent must be [B,N]")
            if self.parent_equivalent.dtype != torch.bool:
                raise ValueError("sidecar parent_equivalent must be boolean")
            if self.parent_equivalent.device != self.key.device:
                raise ValueError(
                    "sidecar parent_equivalent and candidate tensors must share a device"
                )


@dataclass
class FPCTPackedMemory:
    key: Tensor
    value: Tensor
    attention_mask: Tensor
    active: Tensor
    parent_index: Tensor
    candidate_index: Tensor
    log_prior: Tensor
    row_offsets: Tensor
    expanded_slots: Tensor
    extra_slots: Tensor
    parent_equivalent: Tensor
    all_parent_equivalent: Tensor


@dataclass(frozen=True)
class FPCTPackedLayout:
    """Parameter-free structural packing map, reusable across model layers."""

    batch_size: int
    source_length: int
    max_slots: int
    top_k: int
    active: Tensor
    parent_index: Tensor
    safe_parent: Tensor
    candidate_index: Tensor
    candidate_flat_index: Tensor
    safe_candidate_flat_index: Tensor
    use_candidate: Tensor
    log_prior: Tensor
    row_offsets: Tensor
    slot_index: Tensor
    expanded_slots: Tensor
    extra_slots: Tensor
    has_extra_slots: bool
    segment_specs: tuple[tuple[int, int, int, int], ...]
    semantic_layer_indices: tuple[int, ...] = ()
    semantic_parent_equivalent_by_layer: Optional[Tensor] = None

    def validate_runtime(
        self,
        key: Tensor,
        segments: Sequence[FPCTSidecarSegment],
    ) -> None:
        if key.shape[0] != self.batch_size or key.shape[2] != self.source_length:
            raise ValueError("FPCT layout/cache batch or source length mismatch")
        if key.device != self.active.device:
            raise ValueError("FPCT layout and cache must share a device")
        specs = []
        flat_offset = 0
        for segment in segments:
            segment.validate()
            n, k = segment.key.shape[2:4]
            specs.append((segment.parent_start, n, k, flat_offset))
            flat_offset += n * k
        if tuple(specs) != self.segment_specs:
            raise ValueError("FPCT sidecar structure differs from frozen layout")

    def semantic_parent_equivalent(self, layer_index: int) -> Optional[Tensor]:
        if self.semantic_parent_equivalent_by_layer is None:
            return None
        try:
            position = self.semantic_layer_indices.index(int(layer_index))
        except ValueError as error:
            raise ValueError(
                f"FPCT layout has no semantic metadata for layer {layer_index}"
            ) from error
        return self.semantic_parent_equivalent_by_layer[position]


def bind_fpct_layout_layer_semantics(
    layout: FPCTPackedLayout,
    ordered_layer_segments: Sequence[tuple[int, Sequence[FPCTSidecarSegment]]],
) -> FPCTPackedLayout:
    """Attach current-forward semantic parent metadata once for all layers."""

    layer_indices: list[int] = []
    layer_maps: list[Tensor] = []
    for layer_index, segments in ordered_layer_segments:
        semantic = torch.ones(
            layout.batch_size,
            layout.source_length,
            device=layout.active.device,
            dtype=torch.bool,
        )
        for segment in segments:
            segment.validate()
            parent_end = segment.parent_start + segment.key.shape[2]
            semantic[:, segment.parent_start:parent_end] = False
            parent_equivalent = (
                segment.parent_equivalent
                if segment.parent_equivalent is not None
                else segment.parent_force_native
            )
            if parent_equivalent is None:
                continue
            semantic[:, segment.parent_start:parent_end] = (
                parent_equivalent
            )
        layer_indices.append(int(layer_index))
        layer_maps.append(semantic)
    stacked = (
        torch.stack(layer_maps, dim=0)
        if layer_maps
        else torch.empty(
            0,
            layout.batch_size,
            layout.source_length,
            device=layout.active.device,
            dtype=torch.bool,
        )
    )
    return replace(
        layout,
        semantic_layer_indices=tuple(layer_indices),
        semantic_parent_equivalent_by_layer=stacked,
    )


def canonical_log_prior(prior: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return canonical FP32 probability/log-probability and legal mask."""

    prior_fp32 = prior.to(dtype=torch.float32)
    legal = valid.to(torch.bool) & torch.isfinite(prior_fp32) & (prior_fp32 > 0)
    positive = torch.where(legal, prior_fp32, torch.zeros_like(prior_fp32))
    total = positive.sum(dim=-1, keepdim=True)
    normalized = torch.where(
        total > 0,
        positive / total.clamp_min(torch.finfo(torch.float32).tiny),
        torch.zeros_like(positive),
    )
    raw_log = torch.where(
        legal,
        torch.log(normalized.clamp_min(torch.finfo(torch.float32).tiny)),
        torch.full_like(normalized, -torch.inf),
    )
    has_support = legal.any(dim=-1, keepdim=True)
    log_normalizer = torch.logsumexp(raw_log, dim=-1, keepdim=True)
    log_prior = torch.where(
        legal & has_support,
        raw_log - log_normalizer,
        torch.full_like(raw_log, -torch.inf),
    )
    canonical = torch.where(legal, torch.exp(log_prior), torch.zeros_like(log_prior))
    return canonical, log_prior, legal


def normalize_prior(prior: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    normalized, _log_prior, legal = canonical_log_prior(prior, valid)
    return normalized, legal


def build_fpct_packed_layout(
    source_length: int,
    sidecars: Iterable[FPCTSidecarSegment],
    *,
    max_slots_hint: Optional[int] = None,
) -> FPCTPackedLayout:
    """Build the candidate structural map once, before per-layer attention.

    The single scalar transfer used to size the padded batch layout occurs here,
    outside the attention hot path.  All per-layer packing below is tensor-only.
    """

    if source_length < 0:
        raise ValueError("source_length must be non-negative")
    segments = tuple(sidecars)
    if not segments:
        raise ValueError("FPCT layout requires at least one sidecar segment")
    for segment in segments:
        segment.validate()
    batch_size = segments[0].key.shape[0]
    device = segments[0].key.device
    dtype = torch.float32
    top_k = segments[0].key.shape[3]
    prior_full = torch.zeros(
        batch_size, source_length, top_k, device=device, dtype=dtype
    )
    legal_full = torch.zeros(
        batch_size, source_length, top_k, device=device, dtype=torch.bool
    )
    candidate_flat_by_parent = torch.full(
        (source_length, top_k), -1, device=device, dtype=torch.long
    )
    segment_specs: list[tuple[int, int, int, int]] = []
    occupied: list[tuple[int, int]] = []
    flat_offset = 0
    for segment in segments:
        if segment.key.shape[0] != batch_size or segment.key.device != device:
            raise ValueError("FPCT sidecar batch/device mismatch")
        n, k = segment.key.shape[2:4]
        if k != top_k:
            raise ValueError("FPCT sidecar top-k mismatch")
        start, end = segment.parent_start, segment.parent_start + n
        if end > source_length:
            raise ValueError("FPCT sidecar segment exceeds cache source length")
        if any(start < prior_end and prior_start < end for prior_start, prior_end in occupied):
            raise ValueError("overlapping FPCT sidecar parent ranges")
        occupied.append((start, end))
        normalized, _segment_log_prior, legal = canonical_log_prior(
            segment.prior.to(device=device, dtype=torch.float32),
            segment.valid.to(device=device),
        )
        prior_full[:, start:end] = normalized
        legal_full[:, start:end] = legal
        local = torch.arange(n, device=device, dtype=torch.long)[:, None]
        candidate = torch.arange(top_k, device=device, dtype=torch.long)[None, :]
        candidate_flat_by_parent[start:end] = flat_offset + local * top_k + candidate
        segment_specs.append((start, n, top_k, flat_offset))
        flat_offset += n * top_k

    legal_count = legal_full.sum(dim=-1)
    ambiguous = legal_count >= 2
    candidate_axis = torch.arange(top_k, device=device)[None, None, :]
    emit = (ambiguous[..., None] & legal_full) | (
        (~ambiguous[..., None]) & (candidate_axis == 0)
    )
    slots_per_parent = emit.sum(dim=-1)
    row_offsets = torch.cat(
        (
            torch.zeros(batch_size, 1, device=device, dtype=torch.long),
            slots_per_parent.cumsum(dim=-1),
        ),
        dim=-1,
    )
    expanded_slots = row_offsets[:, -1]
    if max_slots_hint is None:
        hints = {segment.max_slots_hint for segment in segments if segment.max_slots_hint is not None}
        if len(hints) == 1:
            max_slots_hint = hints.pop()
    source_length_hints = {
        segment.source_length_hint
        for segment in segments
        if segment.source_length_hint is not None
    }
    if max_slots_hint is not None and len(source_length_hints) == 1:
        max_slots_hint = int(max_slots_hint) + max(
            0, source_length - int(next(iter(source_length_hints)))
        )
    if max_slots_hint is None:
        if device.type == "cuda":
            raise ValueError("CUDA FPCT layout requires CPU-certified max_slots_hint")
        max_slots = int(expanded_slots.max()) if batch_size else 0
    else:
        max_slots = int(max_slots_hint)
        if device.type != "cuda" and batch_size and max_slots != int(expanded_slots.max()):
            raise ValueError("max_slots_hint differs from computed CPU layout width")
    extra_slots = expanded_slots - source_length

    flat_emit = emit.reshape(batch_size, -1)
    rank = flat_emit.to(torch.long).cumsum(dim=-1) - 1
    storage_width = max_slots + 1
    sink = torch.full_like(rank, max_slots)
    destination = torch.where(flat_emit, rank, sink)
    parent_grid = (
        torch.arange(source_length, device=device, dtype=torch.long)[:, None]
        .expand(source_length, top_k)
        .reshape(1, -1)
        .expand(batch_size, -1)
    )
    candidate_grid = (
        torch.arange(top_k, device=device, dtype=torch.long)[None, :]
        .expand(source_length, top_k)
        .reshape(1, -1)
        .expand(batch_size, -1)
    )
    ambiguous_flat = ambiguous[..., None].expand_as(emit).reshape(batch_size, -1)
    use_candidate_flat = flat_emit & ambiguous_flat
    candidate_flat_grid = (
        candidate_flat_by_parent.reshape(1, -1).expand(batch_size, -1)
    )
    _canonical_prior, canonical_log, _canonical_legal = canonical_log_prior(
        prior_full, legal_full
    )
    log_prior_full = canonical_log.reshape(batch_size, -1)

    parent_storage = torch.full(
        (batch_size, storage_width), -1, device=device, dtype=torch.long
    )
    candidate_storage = torch.full_like(parent_storage, -1)
    candidate_flat_storage = torch.full_like(parent_storage, -1)
    active_storage = torch.zeros(
        batch_size, storage_width, device=device, dtype=torch.bool
    )
    use_candidate_storage = torch.zeros_like(active_storage)
    log_prior_storage = torch.zeros(
        batch_size, storage_width, device=device, dtype=dtype
    )
    parent_storage.scatter_(1, destination, parent_grid)
    candidate_storage.scatter_(
        1,
        destination,
        torch.where(use_candidate_flat, candidate_grid, torch.full_like(candidate_grid, -1)),
    )
    candidate_flat_storage.scatter_(
        1,
        destination,
        torch.where(
            use_candidate_flat,
            candidate_flat_grid,
            torch.full_like(candidate_flat_grid, -1),
        ),
    )
    active_storage.scatter_(1, destination, flat_emit)
    use_candidate_storage.scatter_(1, destination, use_candidate_flat)
    log_prior_storage.scatter_(
        1,
        destination,
        torch.where(use_candidate_flat, log_prior_full, torch.zeros_like(log_prior_full)),
    )
    return FPCTPackedLayout(
        batch_size=batch_size,
        source_length=source_length,
        max_slots=max_slots,
        top_k=top_k,
        active=active_storage[:, :max_slots],
        parent_index=parent_storage[:, :max_slots],
        safe_parent=parent_storage[:, :max_slots].clamp_min(0),
        candidate_index=candidate_storage[:, :max_slots],
        candidate_flat_index=candidate_flat_storage[:, :max_slots],
        safe_candidate_flat_index=candidate_flat_storage[:, :max_slots].clamp_min(0),
        use_candidate=use_candidate_storage[:, :max_slots],
        log_prior=log_prior_storage[:, :max_slots],
        row_offsets=row_offsets,
        slot_index=torch.arange(
            max_slots, device=device, dtype=torch.long
        )[None, :].expand(batch_size, -1),
        expanded_slots=expanded_slots,
        extra_slots=extra_slots,
        has_extra_slots=max_slots > source_length,
        segment_specs=tuple(segment_specs),
    )


def _canonical_attention_mask(
    attention_mask: Optional[Tensor],
    *,
    batch_size: int,
    query_length: int,
    source_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if attention_mask is None:
        return torch.zeros(batch_size, 1, query_length, source_length, dtype=torch.float32, device=device)
    if attention_mask.ndim != 4:
        raise ValueError("attention_mask must be [B,Hmask,Q,S]")
    if (
        attention_mask.shape[0] != batch_size
        or attention_mask.shape[-2] != query_length
        or attention_mask.shape[-1] < source_length
    ):
        raise ValueError("attention_mask batch/query/source shape mismatch")
    mask = attention_mask[..., :source_length].to(device=device)
    if mask.dtype == torch.bool:
        mask = torch.where(mask, torch.zeros((), dtype=torch.float32, device=device), torch.full((), -torch.inf, dtype=torch.float32, device=device))
    return mask.to(dtype=torch.float32)


def pack_fpct_memory(
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor],
    sidecars: Iterable[FPCTSidecarSegment],
    *,
    query_length: int,
    layout: Optional[FPCTPackedLayout] = None,
    semantic_parent_equivalent: Optional[Tensor] = None,
    replicated_collapse: bool = False,
    collapse_replicated_groups: bool = True,
) -> FPCTPackedMemory:
    """Tensor-only per-layer packing using a reusable structural layout."""
    if key.ndim != 4 or value.shape != key.shape:
        raise ValueError("key/value must be [B,Hkv,S,D]")
    b, hkv, source_length, d = key.shape
    mask = _canonical_attention_mask(
        attention_mask,
        batch_size=b,
        query_length=query_length,
        source_length=source_length,
        dtype=torch.float32,
        device=key.device,
    )
    segments = tuple(sidecars)
    if layout is None:
        layout = build_fpct_packed_layout(source_length, segments)
    layout.validate_runtime(key, segments)
    for segment in segments:
        if (
            segment.key.shape[0] != b
            or segment.key.shape[1] != hkv
            or segment.key.shape[-1] != d
        ):
            raise ValueError("sidecar/cache batch/head/dim mismatch")
    safe_parent = layout.safe_parent
    parent_gather = safe_parent[:, None, :, None].expand(
        b, hkv, layout.max_slots, d
    )
    parent_key = torch.gather(key, 2, parent_gather)
    parent_value = torch.gather(value, 2, parent_gather)
    packed_key = parent_key
    packed_value = parent_value
    if not replicated_collapse:
        candidate_keys = [
            (
                segment.key
                if segment.key.device == key.device and segment.key.dtype == key.dtype
                else segment.key.to(device=key.device, dtype=key.dtype)
            ).reshape(b, hkv, -1, d)
            for segment in segments
        ]
        candidate_values = [
            (
                segment.value
                if segment.value.device == value.device
                and segment.value.dtype == value.dtype
                else segment.value.to(device=value.device, dtype=value.dtype)
            ).reshape(b, hkv, -1, d)
            for segment in segments
        ]
        candidate_key = (
            candidate_keys[0]
            if len(candidate_keys) == 1
            else torch.cat(candidate_keys, dim=2)
        )
        candidate_value = (
            candidate_values[0]
            if len(candidate_values) == 1
            else torch.cat(candidate_values, dim=2)
        )
        safe_candidate = layout.safe_candidate_flat_index
        candidate_gather = safe_candidate[:, None, :, None].expand(
            b, hkv, layout.max_slots, d
        )
        gathered_candidate_key = torch.gather(candidate_key, 2, candidate_gather)
        gathered_candidate_value = torch.gather(
            candidate_value, 2, candidate_gather
        )
        use_candidate = layout.use_candidate[:, None, :, None]
        packed_key = torch.where(use_candidate, gathered_candidate_key, packed_key)
        packed_value = torch.where(
            use_candidate, gathered_candidate_value, packed_value
        )
    active = layout.active
    log_prior = layout.log_prior
    candidate_slots = layout.use_candidate & active
    if semantic_parent_equivalent is None:
        key_equal = (packed_key == parent_key).all(dim=(1, 3))
        value_equal = (packed_value == parent_value).all(dim=(1, 3))
        slot_equal = (~candidate_slots) | (key_equal & value_equal)
        group_equal = torch.ones(
            b,
            source_length,
            device=key.device,
            dtype=torch.long,
        )
        group_equal.scatter_reduce_(
            1,
            safe_parent,
            slot_equal.to(torch.long),
            reduce="amin",
            include_self=True,
        )
        semantic_parent_equivalent = torch.zeros(
            b,
            source_length,
            device=key.device,
            dtype=torch.bool,
        )
        for segment in segments:
            if segment.parent_force_native is None:
                continue
            parent_end = segment.parent_start + segment.key.shape[2]
            semantic_parent_equivalent[:, segment.parent_start:parent_end] = (
                segment.parent_force_native
            )
        parent_equivalent = group_equal.to(torch.bool) | semantic_parent_equivalent
    elif semantic_parent_equivalent.shape != (b, source_length):
        raise ValueError("semantic parent-equivalence map must be [B,S]")
    else:
        parent_equivalent = semantic_parent_equivalent
    if collapse_replicated_groups:
        replicated_group = candidate_slots & torch.gather(
            parent_equivalent, 1, safe_parent
        )
        first_slot = torch.gather(
            layout.row_offsets[:, :-1], 1, safe_parent
        )
        replicated_first = replicated_group & (layout.slot_index == first_slot)
        active = active & (~replicated_group | replicated_first)
        equivalent_slot = replicated_first[:, None, :, None]
        packed_key = torch.where(equivalent_slot, parent_key, packed_key)
        packed_value = torch.where(equivalent_slot, parent_value, packed_value)
        log_prior = torch.where(
            replicated_first,
            torch.zeros_like(log_prior),
            log_prior,
        )
    log_prior = torch.where(
        active,
        log_prior,
        torch.full_like(log_prior, -torch.inf),
    )
    mask_gather = safe_parent[:, None, None, :].expand(
        b, mask.shape[1], query_length, layout.max_slots
    )
    packed_mask = torch.gather(mask, -1, mask_gather)
    packed_mask = packed_mask.float() + log_prior.to(
        device=key.device, dtype=torch.float32
    )[:, None, None, :]
    packed_mask = torch.where(
        active[:, None, None, :],
        packed_mask,
        torch.full_like(packed_mask, -torch.inf),
    )
    return FPCTPackedMemory(
        key=packed_key,
        value=packed_value,
        attention_mask=packed_mask,
        active=active,
        parent_index=layout.parent_index,
        candidate_index=layout.candidate_index,
        log_prior=log_prior,
        row_offsets=layout.row_offsets,
        expanded_slots=layout.expanded_slots,
        extra_slots=layout.extra_slots,
        parent_equivalent=parent_equivalent,
        all_parent_equivalent=parent_equivalent.all(dim=-1),
    )


def fpct_replicated_probability_mass_delta(
    module: nn.Module,
    query: Tensor,
    packed: FPCTPackedMemory,
    parent_key: Tensor,
    parent_attention_mask: Optional[Tensor],
    *,
    scaling: float,
) -> tuple[Tensor, Tensor]:
    """Compare expanded replicated group mass with parent FP32 probability."""

    groups = int(module.num_key_value_groups)

    def repeat_kv(value: Tensor) -> Tensor:
        if groups == 1:
            return value
        return value[:, :, None, :, :].expand(
            value.shape[0], value.shape[1], groups, value.shape[2], value.shape[3]
        ).reshape(
            value.shape[0],
            value.shape[1] * groups,
            value.shape[2],
            value.shape[3],
        )

    expanded_key = repeat_kv(packed.key)
    repeated_parent_key = repeat_kv(parent_key)
    expanded_logits = (
        torch.matmul(query.float(), expanded_key.float().transpose(2, 3))
        * float(scaling)
    )
    expanded_logits = expanded_logits + packed.attention_mask.float()
    parent_logits = (
        torch.matmul(query.float(), repeated_parent_key.float().transpose(2, 3))
        * float(scaling)
    )
    if parent_attention_mask is not None:
        parent_logits = parent_logits + parent_attention_mask[
            :, :, :, : parent_key.shape[-2]
        ].float()
    expanded_probability = nn.functional.softmax(
        expanded_logits, dim=-1, dtype=torch.float32
    )
    parent_probability = nn.functional.softmax(
        parent_logits, dim=-1, dtype=torch.float32
    )
    parent_index = packed.parent_index.clamp_min(0)[:, None, None, :].expand(
        query.shape[0], query.shape[1], query.shape[2], -1
    )
    grouped_probability = torch.zeros_like(parent_probability)
    grouped_probability.scatter_add_(
        -1, parent_index, expanded_probability
    )
    return (
        (grouped_probability - parent_probability).abs().amax(),
        expanded_probability,
    )


def fpct_eager_attention(query: Tensor, packed: FPCTPackedMemory) -> tuple[Tensor, Tensor]:
    """Small pure-tensor attention used only by CPU correctness tests."""
    if query.ndim != 4:
        raise ValueError("query must be [B,Hq,Q,D]")
    b, hq, _q, d = query.shape
    hkv = packed.key.shape[1]
    if hq % hkv != 0:
        raise ValueError("Hq must be divisible by Hkv")
    key = packed.key.repeat_interleave(hq // hkv, dim=1)
    value = packed.value.repeat_interleave(hq // hkv, dim=1)
    accumulation_dtype = (
        torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    )
    logits = torch.einsum(
        "bhqd,bhmd->bhqm",
        query.to(accumulation_dtype),
        key.to(accumulation_dtype),
    ) / sqrt(d)
    logits = logits + torch.broadcast_to(
        packed.attention_mask.to(accumulation_dtype), logits.shape
    )
    active = packed.active[:, None, None, :] & torch.isfinite(torch.broadcast_to(packed.attention_mask, logits.shape))
    masked = torch.where(active, logits, torch.full_like(logits, -torch.inf))
    any_active = active.any(dim=-1, keepdim=True)
    probability = torch.softmax(torch.where(any_active, masked, torch.zeros_like(masked)), dim=-1)
    probability = torch.where(active, probability, torch.zeros_like(probability))
    probability = torch.where(
        probability.sum(dim=-1, keepdim=True) > 0,
        probability / probability.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probability.dtype).tiny),
        torch.zeros_like(probability),
    )
    output = torch.einsum(
        "bhqm,bhmd->bhqd", probability, value.to(accumulation_dtype)
    ).to(query.dtype)
    return output, probability.to(query.dtype)


def fpct_qwen_eager_attention_forward(
    module: nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor],
    scaling: float,
    dropout: float = 0.0,
    return_fp32_logits: bool = False,
    **_kwargs,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Shared C_post/F eager adapter with FP32 logits, mask and reduction."""

    groups = int(module.num_key_value_groups)
    if groups > 1:
        key = key[:, :, None, :, :].expand(
            key.shape[0], key.shape[1], groups, key.shape[2], key.shape[3]
        ).reshape(key.shape[0], key.shape[1] * groups, key.shape[2], key.shape[3])
        value = value[:, :, None, :, :].expand(
            value.shape[0], value.shape[1], groups, value.shape[2], value.shape[3]
        ).reshape(
            value.shape[0], value.shape[1] * groups, value.shape[2], value.shape[3]
        )
    logits = torch.matmul(query.float(), key.float().transpose(2, 3)) * float(scaling)
    if attention_mask is not None:
        logits = logits + attention_mask[:, :, :, : key.shape[-2]].float()
    probability = nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
    probability = nn.functional.dropout(
        probability, p=dropout, training=module.training
    )
    output = torch.matmul(probability, value.float()).to(dtype=query.dtype)
    result = output.transpose(1, 2).contiguous(), probability.to(dtype=query.dtype)
    if return_fp32_logits:
        return result[0], result[1], logits
    return result


def fpct_qwen_hierarchical_attention_forward(
    module: nn.Module,
    query: Tensor,
    packed: FPCTPackedMemory,
    parent_key: Tensor,
    parent_value: Tensor,
    parent_attention_mask: Optional[Tensor],
    scaling: float,
    dropout: float = 0.0,
    **_kwargs,
) -> tuple[Tensor, Tensor]:
    """Global-equivalent flat-atom attention with exact parent reuse."""

    # Preserve the exact C_post numerical call order for the parent branch.
    # R2f established that candidate and collapsed K/V can be bit-identical to
    # native while a parent matmul executed after the grouped atom kernels can
    # still accumulate a deep FP32 deviation.  Compute the shared adapter first;
    # the tensor-only final selection remains valid for mixed batches.
    parent_output, _parent_probability, parent_logits = (
        fpct_qwen_eager_attention_forward(
        module,
        query,
        parent_key,
        parent_value,
        parent_attention_mask,
        scaling=scaling,
        dropout=dropout,
        return_fp32_logits=True,
        )
    )

    groups = int(module.num_key_value_groups)

    def repeat_kv(value: Tensor) -> Tensor:
        if groups == 1:
            return value
        return value[:, :, None, :, :].expand(
            value.shape[0], value.shape[1], groups, value.shape[2], value.shape[3]
        ).reshape(
            value.shape[0],
            value.shape[1] * groups,
            value.shape[2],
            value.shape[3],
        )

    atom_key = repeat_kv(packed.key)
    atom_value = repeat_kv(packed.value).float()
    parent_index = packed.parent_index.clamp_min(0)[:, None, None, :].expand(
        query.shape[0], query.shape[1], query.shape[2], -1
    )
    equivalent_slot = (
        packed.active
        & torch.gather(
            packed.parent_equivalent,
            1,
            packed.parent_index.clamp_min(0),
        )
    )[:, None, None, :]
    # Equivalent parents reuse the already-computed parent logits. Their keys
    # are removed from the atom QK dataflow by replacing them with exact zero
    # before the candidate matmul; the gathered parent logit is inserted below.
    equivalent_key_slot = equivalent_slot[:, 0, 0, :][:, None, :, None]
    matmul_key = torch.where(
        equivalent_key_slot,
        torch.zeros((), device=atom_key.device, dtype=atom_key.dtype),
        atom_key,
    )
    atom_logits = (
        torch.matmul(query.float(), matmul_key.float().transpose(2, 3))
        * float(scaling)
    )
    atom_logits = atom_logits + packed.attention_mask.float()
    atom_logits = torch.where(
        equivalent_slot,
        torch.gather(parent_logits, 3, parent_index),
        atom_logits,
    )
    atom_active = (
        packed.active[:, None, None, :]
        & torch.isfinite(packed.attention_mask[:, :, :, : packed.key.shape[-2]])
    )
    masked_logits = torch.where(
        atom_active, atom_logits, torch.full_like(atom_logits, -torch.inf)
    )
    any_active = atom_active.any(dim=-1, keepdim=True)
    atom_probability = nn.functional.softmax(
        torch.where(any_active, masked_logits, torch.zeros_like(masked_logits)),
        dim=-1,
        dtype=torch.float32,
    )
    atom_probability = torch.where(
        atom_active, atom_probability, torch.zeros_like(atom_probability)
    )
    if module.training and dropout > 0:
        atom_probability = nn.functional.dropout(
            atom_probability, p=dropout, training=True
        )
    flat_output = torch.matmul(
        atom_probability, atom_value
    ).to(query.dtype).transpose(
        1, 2
    ).contiguous()
    output = torch.where(
        packed.all_parent_equivalent[:, None, None, None],
        parent_output,
        flat_output,
    )
    return output, atom_probability.to(query.dtype)


def fpct_mechanism_diagnostics(
    query: Tensor,
    packed: FPCTPackedMemory,
) -> dict[str, Tensor]:
    """Aggregate query-time mechanism diagnostics without retaining raw KV."""

    b, hq, q_length, d = query.shape
    hkv = packed.key.shape[1]
    if hq % hkv != 0:
        raise ValueError("Hq must be divisible by Hkv")
    source_length = packed.row_offsets.shape[1] - 1
    key = packed.key.repeat_interleave(hq // hkv, dim=1).float()
    logits = torch.einsum("bhqd,bhmd->bhqm", query.float(), key) / sqrt(d)
    broadcast_mask = torch.broadcast_to(
        packed.attention_mask.float(), logits.shape
    )
    logits = logits + broadcast_mask
    parent = packed.parent_index.clamp_min(0)
    parent_index = parent[:, None, None, :].expand(b, hq, q_length, -1)
    candidate = (
        (packed.candidate_index >= 0) & packed.active
    )[:, None, None, :]
    candidate = (
        candidate.expand(b, hq, q_length, -1)
        & torch.isfinite(logits)
        & torch.isfinite(broadcast_mask)
        & (broadcast_mask > torch.finfo(torch.float32).min / 2)
    )
    log_prior = packed.log_prior[:, None, None, :].float().expand_as(logits)
    raw_logits = logits - log_prior
    zeros = torch.zeros_like(logits)
    count = torch.zeros(
        b, hq, q_length, source_length, device=query.device, dtype=torch.float32
    )
    count.scatter_add_(3, parent_index, candidate.to(torch.float32))
    raw_sum = torch.zeros_like(count)
    safe_raw_logits = torch.where(candidate, raw_logits, zeros)
    raw_sum.scatter_add_(3, parent_index, safe_raw_logits)
    raw_square_sum = torch.zeros_like(count)
    raw_square_sum.scatter_add_(
        3, parent_index, safe_raw_logits.square()
    )
    denominator = count.clamp_min(1.0)
    raw_mean = raw_sum / denominator
    raw_variance = (raw_square_sum / denominator - raw_mean.square()).clamp_min(0)
    negative_inf = torch.full_like(count, -torch.inf)
    positive_inf = torch.full_like(count, torch.inf)
    raw_max = negative_inf.scatter_reduce(
        3,
        parent_index,
        torch.where(candidate, raw_logits, torch.full_like(raw_logits, -torch.inf)),
        reduce="amax",
        include_self=True,
    )
    raw_min = positive_inf.scatter_reduce(
        3,
        parent_index,
        torch.where(candidate, raw_logits, torch.full_like(raw_logits, torch.inf)),
        reduce="amin",
        include_self=True,
    )
    logit_range = raw_max - raw_min
    parent_valid = count >= 2

    group_max = negative_inf.scatter_reduce(
        3,
        parent_index,
        torch.where(candidate, logits, torch.full_like(logits, -torch.inf)),
        reduce="amax",
        include_self=True,
    )
    gathered_max = torch.gather(group_max, 3, parent_index)
    exp_shifted = torch.where(
        candidate,
        torch.exp(logits - gathered_max),
        torch.zeros_like(logits),
    )
    group_sum = torch.zeros_like(count)
    group_sum.scatter_add_(3, parent_index, exp_shifted)
    gathered_sum = torch.gather(group_sum, 3, parent_index).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    gamma = torch.where(candidate, exp_shifted / gathered_sum, torch.zeros_like(logits))
    prior = torch.where(candidate, torch.exp(log_prior), torch.zeros_like(logits))
    kl_atom = torch.where(
        candidate & (gamma > 0),
        gamma
        * (
            torch.log(gamma.clamp_min(torch.finfo(torch.float32).tiny))
            - log_prior
        ),
        torch.zeros_like(gamma),
    )
    kl_parent = torch.zeros_like(count)
    kl_parent.scatter_add_(3, parent_index, kl_atom)
    prior_raw_mean = torch.zeros_like(count)
    prior_raw_mean.scatter_add_(
        3, parent_index, torch.where(candidate, prior * raw_logits, zeros)
    )
    group_logsumexp = group_max + torch.log(
        group_sum.clamp_min(torch.finfo(torch.float32).tiny)
    )
    jensen_gap = (group_logsumexp - prior_raw_mean).clamp_min(0)
    gamma_query_variance = gamma.var(dim=2, unbiased=False)
    gamma_query_mask = (
        (packed.candidate_index >= 0) & packed.active
    )[:, None, :]

    candidate_mask_kv = (
        (packed.candidate_index >= 0) & packed.active
    )[:, None, :, None]
    atom_prior = torch.exp(packed.log_prior.float())[:, None, :, None]
    parent_kv = parent[:, None, :, None].expand(
        b, hkv, packed.key.shape[2], packed.key.shape[3]
    )
    key_sum = torch.zeros(
        b, hkv, source_length, packed.key.shape[3],
        device=query.device, dtype=torch.float32,
    )
    value_sum = torch.zeros_like(key_sum)
    key_sum.scatter_add_(
        2,
        parent_kv,
        torch.where(candidate_mask_kv, atom_prior * packed.key.float(), 0.0),
    )
    value_sum.scatter_add_(
        2,
        parent_kv,
        torch.where(candidate_mask_kv, atom_prior * packed.value.float(), 0.0),
    )
    gathered_key_mean = torch.gather(key_sum, 2, parent_kv)
    gathered_value_mean = torch.gather(value_sum, 2, parent_kv)
    key_dispersion_atom = torch.where(
        candidate_mask_kv,
        atom_prior * (packed.key.float() - gathered_key_mean).square(),
        0.0,
    )
    value_dispersion_atom = torch.where(
        candidate_mask_kv,
        atom_prior * (packed.value.float() - gathered_value_mean).square(),
        0.0,
    )
    key_dispersion = torch.zeros_like(key_sum)
    value_dispersion = torch.zeros_like(value_sum)
    key_dispersion.scatter_add_(2, parent_kv, key_dispersion_atom)
    value_dispersion.scatter_add_(2, parent_kv, value_dispersion_atom)
    parent_has_candidates = (
        torch.zeros(b, source_length, device=query.device, dtype=torch.float32)
        .scatter_add_(
            2 - 1,
            parent,
            ((packed.candidate_index >= 0) & packed.active).float(),
        )
        >= 2
    )

    def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
        expanded = torch.broadcast_to(mask, value.shape)
        return torch.where(expanded, value, torch.zeros_like(value)).sum() / expanded.sum().clamp_min(1)

    return {
        "gamma_kl_prior": masked_mean(kl_parent, parent_valid),
        "gamma_query_variance": masked_mean(
            gamma_query_variance, gamma_query_mask
        ),
        "candidate_logit_variance": masked_mean(raw_variance, parent_valid),
        "candidate_logit_range": masked_mean(logit_range, parent_valid),
        "jensen_gap": masked_mean(jensen_gap, parent_valid),
        "d_k": masked_mean(
            key_dispersion.mean(dim=-1), parent_has_candidates[:, None, :]
        ),
        "d_v": masked_mean(
            value_dispersion.mean(dim=-1), parent_has_candidates[:, None, :]
        ),
        "expanded_slot_ratio": packed.expanded_slots.float().mean()
        / float(source_length),
        "extra_slots": packed.extra_slots.float().mean(),
    }
