from __future__ import annotations

"""Candidate sidecar and ambiguous-only packed attention for FPCT."""

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Optional, Sequence

import torch
from torch import Tensor


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

    def validate(self) -> None:
        if self.parent_start < 0:
            raise ValueError("parent_start must be non-negative")
        if self.key.ndim != 5 or self.value.shape != self.key.shape:
            raise ValueError("sidecar key/value must have matching [B,Hkv,N,K,D] shapes")
        if self.prior.shape != self.valid.shape or self.prior.shape != (
            self.key.shape[0], self.key.shape[2], self.key.shape[3]
        ):
            raise ValueError("sidecar prior/valid shape mismatch")


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


@dataclass(frozen=True)
class FPCTPackedLayout:
    """Parameter-free structural packing map, reusable across model layers."""

    batch_size: int
    source_length: int
    max_slots: int
    top_k: int
    active: Tensor
    parent_index: Tensor
    candidate_index: Tensor
    candidate_flat_index: Tensor
    use_candidate: Tensor
    log_prior: Tensor
    row_offsets: Tensor
    expanded_slots: Tensor
    extra_slots: Tensor
    segment_specs: tuple[tuple[int, int, int, int], ...]

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


def normalize_prior(prior: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    legal = valid.to(torch.bool) & torch.isfinite(prior) & (prior > 0)
    positive = torch.where(legal, prior, torch.zeros_like(prior))
    total = positive.sum(dim=-1, keepdim=True)
    normalized = torch.where(
        total > 0,
        positive / total.clamp_min(torch.finfo(prior.dtype).tiny),
        torch.zeros_like(positive),
    )
    return normalized, legal


def build_fpct_packed_layout(
    source_length: int,
    sidecars: Iterable[FPCTSidecarSegment],
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
    dtype = segments[0].prior.dtype
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
        normalized, legal = normalize_prior(
            segment.prior.to(device=device, dtype=dtype),
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
    # One synchronization per structural layout, never per layer/attention call.
    max_slots = int(expanded_slots.max().detach().cpu()) if batch_size else 0
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
    log_prior_full = torch.where(
        legal_full,
        torch.log(prior_full.clamp_min(torch.finfo(dtype).tiny)),
        torch.zeros_like(prior_full),
    ).reshape(batch_size, -1)

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
        candidate_index=candidate_storage[:, :max_slots],
        candidate_flat_index=candidate_flat_storage[:, :max_slots],
        use_candidate=use_candidate_storage[:, :max_slots],
        log_prior=log_prior_storage[:, :max_slots],
        row_offsets=row_offsets,
        expanded_slots=expanded_slots,
        extra_slots=extra_slots,
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
        return torch.zeros(batch_size, 1, query_length, source_length, dtype=dtype, device=device)
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
        mask = torch.where(mask, torch.zeros((), dtype=dtype, device=device), torch.full((), -torch.inf, dtype=dtype, device=device))
    return mask.to(dtype=dtype)


def pack_fpct_memory(
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor],
    sidecars: Iterable[FPCTSidecarSegment],
    *,
    query_length: int,
    layout: Optional[FPCTPackedLayout] = None,
    replicated_collapse: bool = False,
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
        dtype=key.dtype,
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
    safe_parent = layout.parent_index.clamp_min(0)
    parent_gather = safe_parent[:, None, :, None].expand(
        b, hkv, layout.max_slots, d
    )
    packed_key = torch.gather(key, 2, parent_gather)
    packed_value = torch.gather(value, 2, parent_gather)
    if not replicated_collapse:
        candidate_key = torch.cat(
            [
                segment.key.to(device=key.device, dtype=key.dtype).reshape(
                    b, hkv, -1, d
                )
                for segment in segments
            ],
            dim=2,
        )
        candidate_value = torch.cat(
            [
                segment.value.to(device=value.device, dtype=value.dtype).reshape(
                    b, hkv, -1, d
                )
                for segment in segments
            ],
            dim=2,
        )
        safe_candidate = layout.candidate_flat_index.clamp_min(0)
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
    mask_gather = safe_parent[:, None, None, :].expand(
        b, mask.shape[1], query_length, layout.max_slots
    )
    packed_mask = torch.gather(mask, -1, mask_gather)
    packed_mask = packed_mask + layout.log_prior.to(
        device=key.device, dtype=key.dtype
    )[:, None, None, :]
    packed_mask = torch.where(
        layout.active[:, None, None, :],
        packed_mask,
        torch.full_like(packed_mask, -torch.inf),
    )
    return FPCTPackedMemory(
        key=packed_key,
        value=packed_value,
        attention_mask=packed_mask,
        active=layout.active,
        parent_index=layout.parent_index,
        candidate_index=layout.candidate_index,
        log_prior=layout.log_prior,
        row_offsets=layout.row_offsets,
        expanded_slots=layout.expanded_slots,
        extra_slots=layout.extra_slots,
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
    candidate = packed.candidate_index[:, None, None, :] >= 0
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
    gamma_query_mask = packed.candidate_index[:, None, :] >= 0

    candidate_mask_kv = packed.candidate_index[:, None, :, None] >= 0
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
        .scatter_add_(2 - 1, parent, (packed.candidate_index >= 0).float())
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
