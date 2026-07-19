from __future__ import annotations

"""Candidate sidecar and ambiguous-only packed attention for FPCT."""

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Optional

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
    expanded_slots: Tensor
    extra_slots: Tensor


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
) -> FPCTPackedMemory:
    """Replace only m>=2 parent placeholders with legal child atoms."""
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
    segments = list(sidecars)
    parent_map: dict[int, tuple[FPCTSidecarSegment, int, Tensor, Tensor]] = {}
    for segment in segments:
        segment.validate()
        if segment.key.shape[0] != b or segment.key.shape[1] != hkv or segment.key.shape[-1] != d:
            raise ValueError("sidecar/cache batch/head/dim mismatch")
        a, legal = normalize_prior(
            segment.prior.to(device=key.device, dtype=key.dtype),
            segment.valid.to(device=key.device),
        )
        n = segment.key.shape[2]
        if segment.parent_start + n > source_length:
            raise ValueError("sidecar segment exceeds current cache length")
        for local_index in range(n):
            parent = segment.parent_start + local_index
            if parent in parent_map:
                raise ValueError(f"overlapping FPCT sidecar parent: {parent}")
            parent_map[parent] = (segment, local_index, a[:, local_index], legal[:, local_index])

    batch_keys: list[list[Tensor]] = [[] for _ in range(b)]
    batch_values: list[list[Tensor]] = [[] for _ in range(b)]
    batch_masks: list[list[Tensor]] = [[] for _ in range(b)]
    batch_parents: list[list[int]] = [[] for _ in range(b)]
    batch_candidates: list[list[int]] = [[] for _ in range(b)]
    expanded_slots = torch.zeros(b, dtype=torch.long, device=key.device)
    extra_slots = torch.zeros_like(expanded_slots)

    for batch in range(b):
        for parent in range(source_length):
            mapping = parent_map.get(parent)
            legal_indices: list[int] = []
            if mapping is not None:
                _segment, _local, _a, legal = mapping
                legal_indices = torch.nonzero(legal[batch], as_tuple=False).flatten().tolist()
            if mapping is not None and len(legal_indices) >= 2:
                segment, local, a, _legal = mapping
                for candidate in legal_indices:
                    batch_keys[batch].append(segment.key[batch, :, local, candidate].to(key.device, key.dtype))
                    batch_values[batch].append(segment.value[batch, :, local, candidate].to(value.device, value.dtype))
                    batch_masks[batch].append(mask[batch, :, :, parent] + torch.log(a[batch, candidate]))
                    batch_parents[batch].append(parent)
                    batch_candidates[batch].append(candidate)
                extra_slots[batch] += len(legal_indices) - 1
            else:
                batch_keys[batch].append(key[batch, :, parent])
                batch_values[batch].append(value[batch, :, parent])
                batch_masks[batch].append(mask[batch, :, :, parent])
                batch_parents[batch].append(parent)
                batch_candidates[batch].append(legal_indices[0] if len(legal_indices) == 1 else -1)
        expanded_slots[batch] = len(batch_keys[batch])

    max_slots = int(expanded_slots.max().item()) if b else 0
    packed_key = key.new_zeros((b, hkv, max_slots, d))
    packed_value = value.new_zeros((b, hkv, max_slots, d))
    packed_mask = key.new_full((b, mask.shape[1], query_length, max_slots), -torch.inf)
    active = torch.zeros((b, max_slots), dtype=torch.bool, device=key.device)
    parent_index = torch.full((b, max_slots), -1, dtype=torch.long, device=key.device)
    candidate_index = torch.full_like(parent_index, -1)
    for batch in range(b):
        count = len(batch_keys[batch])
        if count == 0:
            continue
        packed_key[batch, :, :count] = torch.stack(batch_keys[batch], dim=1)
        packed_value[batch, :, :count] = torch.stack(batch_values[batch], dim=1)
        packed_mask[batch, :, :, :count] = torch.stack(batch_masks[batch], dim=-1)
        active[batch, :count] = True
        parent_index[batch, :count] = torch.tensor(batch_parents[batch], device=key.device)
        candidate_index[batch, :count] = torch.tensor(batch_candidates[batch], device=key.device)
    return FPCTPackedMemory(
        key=packed_key,
        value=packed_value,
        attention_mask=packed_mask,
        active=active,
        parent_index=parent_index,
        candidate_index=candidate_index,
        expanded_slots=expanded_slots,
        extra_slots=extra_slots,
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
    logits = torch.einsum("bhqd,bhmd->bhqm", query, key) / sqrt(d)
    logits = logits + torch.broadcast_to(packed.attention_mask, logits.shape)
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
    return torch.einsum("bhqm,bhmd->bhqd", probability, value), probability
