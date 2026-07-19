from __future__ import annotations

"""Pure-tensor FPCT reference operators.

This module deliberately has no Transformers or Rosetta model dependency.  It
defines the mathematical C_pre, C_post, F and replicated-collapse contracts for
small CPU tensors and exposes both flat-global and hierarchical F reductions.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Callable

import torch
from torch import Tensor


Fuser = Callable[[Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]]


@dataclass
class OperatorResult:
    output: Tensor
    native_probability: Tensor
    transported_probability: Tensor
    fused_key: Tensor
    fused_value: Tensor
    normalized_prior: Tensor
    legal_candidate: Tensor


def _require_shape(name: str, tensor: Tensor, ndim: int) -> None:
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {tuple(tensor.shape)}")


def validate_inputs(
    q: Tensor,
    native_k: Tensor,
    native_v: Tensor,
    source_k: Tensor,
    source_v: Tensor,
    prior: Tensor,
    valid: Tensor,
) -> None:
    _require_shape("q", q, 4)
    _require_shape("native_k", native_k, 4)
    _require_shape("native_v", native_v, 4)
    _require_shape("source_k", source_k, 5)
    _require_shape("source_v", source_v, 5)
    _require_shape("prior", prior, 3)
    _require_shape("valid", valid, 3)
    b, hq, _t, d = q.shape
    bk, hkv, n, dk = native_k.shape
    if (bk, dk) != (b, d) or native_v.shape != native_k.shape:
        raise ValueError("native K/V shape mismatch")
    if hq % hkv != 0:
        raise ValueError("Hq must be divisible by Hkv for GQA/MQA expansion")
    if source_k.shape[:1] != (b,) or source_k.shape[2] != n:
        raise ValueError("source candidate batch/parent shape mismatch")
    if source_v.shape != source_k.shape:
        raise ValueError("source candidate K/V shape mismatch")
    if prior.shape != valid.shape or prior.shape != (b, n, source_k.shape[3]):
        raise ValueError("A/valid shape mismatch")


def normalize_prior(prior: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    legal = valid.to(torch.bool) & torch.isfinite(prior) & (prior > 0)
    masked = torch.where(legal, prior, torch.zeros_like(prior))
    total = masked.sum(dim=-1, keepdim=True)
    normalized = torch.where(total > 0, masked / total.clamp_min(torch.finfo(prior.dtype).tiny), torch.zeros_like(masked))
    parent_has_support = legal.any(dim=-1)
    return normalized, legal, parent_has_support


def expand_kv_heads(tensor: Tensor, hq: int) -> Tensor:
    hkv = tensor.shape[1]
    if hq % hkv != 0:
        raise ValueError("Hq must be divisible by Hkv")
    return tensor.repeat_interleave(hq // hkv, dim=1)


def broadcast_parent_term(term: Tensor | None, q: Tensor, n: int) -> Tensor:
    b, hq, t, _d = q.shape
    if term is None:
        return q.new_zeros((b, hq, t, n))
    try:
        return torch.broadcast_to(term, (b, hq, t, n))
    except RuntimeError as exc:
        raise ValueError(f"parent term cannot broadcast to {(b, hq, t, n)}") from exc


def masked_softmax(logits: Tensor, active: Tensor, dim: int = -1) -> Tensor:
    active = torch.broadcast_to(active.to(torch.bool), logits.shape)
    masked = torch.where(active, logits, torch.full_like(logits, -torch.inf))
    any_active = active.any(dim=dim, keepdim=True)
    safe = torch.where(any_active, masked, torch.zeros_like(masked))
    probability = torch.softmax(safe, dim=dim)
    probability = torch.where(active, probability, torch.zeros_like(probability))
    normalizer = probability.sum(dim=dim, keepdim=True)
    return torch.where(normalizer > 0, probability / normalizer, torch.zeros_like(probability))


def _attention_logits(q: Tensor, key: Tensor) -> Tensor:
    # q: [B,Hq,T,D], key: [B,Hq,M,D]
    return torch.einsum("bhtd,bhmd->bhtm", q, key) / sqrt(q.shape[-1])


def _single_slot_attention(
    q: Tensor,
    slot_k: Tensor,
    slot_v: Tensor,
    active: Tensor,
    parent_bias: Tensor | None,
    parent_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    b, hq, _t, _d = q.shape
    n = slot_k.shape[2]
    k = expand_kv_heads(slot_k, hq)
    v = expand_kv_heads(slot_v, hq)
    logits = _attention_logits(q, k)
    logits = logits + broadcast_parent_term(parent_bias, q, n) + broadcast_parent_term(parent_mask, q, n)
    finite_parent = torch.isfinite(broadcast_parent_term(parent_mask, q, n))
    slot_active = active[:, None, None, :] & finite_parent
    probability = masked_softmax(logits, slot_active)
    output = torch.einsum("bhtn,bhnd->bhtd", probability, v)
    return output, probability


def fuse_candidate_specific(
    fuser: Fuser,
    native_k: Tensor,
    native_v: Tensor,
    source_k: Tensor,
    source_v: Tensor,
) -> tuple[Tensor, Tensor]:
    k = source_k.shape[3]
    base_k = native_k.unsqueeze(3).expand(-1, -1, -1, k, -1)
    base_v = native_v.unsqueeze(3).expand(-1, -1, -1, k, -1)
    fused_k, fused_v = fuser(base_k, base_v, source_k, source_v)
    expected = (*native_k.shape[:3], k, native_k.shape[-1])
    if fused_k.shape != expected or fused_v.shape != expected:
        raise ValueError(f"candidate-specific fuser must return {expected}")
    return fused_k, fused_v


def c_pre(
    q: Tensor,
    native_k: Tensor,
    native_v: Tensor,
    source_k: Tensor,
    source_v: Tensor,
    prior: Tensor,
    valid: Tensor,
    fuser: Fuser,
    parent_bias: Tensor | None = None,
    parent_mask: Tensor | None = None,
) -> OperatorResult:
    validate_inputs(q, native_k, native_v, source_k, source_v, prior, valid)
    a, legal, has_support = normalize_prior(prior, valid)
    source_weight = a[:, None, :, :, None]
    averaged_k = (source_k * source_weight).sum(dim=3)
    averaged_v = (source_v * source_weight).sum(dim=3)
    fused_k, fused_v = fuser(native_k, native_v, averaged_k, averaged_v)
    if fused_k.shape != native_k.shape or fused_v.shape != native_v.shape:
        raise ValueError("C_pre fuser output shape mismatch")
    slot_k = torch.where(has_support[:, None, :, None], fused_k, native_k)
    slot_v = torch.where(has_support[:, None, :, None], fused_v, native_v)
    active = torch.ones_like(has_support, dtype=torch.bool)
    output, probability = _single_slot_attention(q, slot_k, slot_v, active, parent_bias, parent_mask)
    return OperatorResult(output, probability, probability, fused_k, fused_v, a, legal)


def candidate_fused(
    native_k: Tensor,
    native_v: Tensor,
    source_k: Tensor,
    source_v: Tensor,
    prior: Tensor,
    valid: Tensor,
    fuser: Fuser,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    a, legal, has_support = normalize_prior(prior, valid)
    fused_k, fused_v = fuse_candidate_specific(fuser, native_k, native_v, source_k, source_v)
    return fused_k, fused_v, a, legal, has_support


def c_post(
    q: Tensor,
    native_k: Tensor,
    native_v: Tensor,
    source_k: Tensor,
    source_v: Tensor,
    prior: Tensor,
    valid: Tensor,
    fuser: Fuser,
    parent_bias: Tensor | None = None,
    parent_mask: Tensor | None = None,
) -> OperatorResult:
    validate_inputs(q, native_k, native_v, source_k, source_v, prior, valid)
    fused_k, fused_v, a, legal, has_support = candidate_fused(native_k, native_v, source_k, source_v, prior, valid, fuser)
    weight = a[:, None, :, :, None]
    post_k = (fused_k * weight).sum(dim=3)
    post_v = (fused_v * weight).sum(dim=3)
    slot_k = torch.where(has_support[:, None, :, None], post_k, native_k)
    slot_v = torch.where(has_support[:, None, :, None], post_v, native_v)
    active = torch.ones_like(has_support, dtype=torch.bool)
    output, probability = _single_slot_attention(q, slot_k, slot_v, active, parent_bias, parent_mask)
    return OperatorResult(output, probability, probability, fused_k, fused_v, a, legal)


def _flat_factorized_attention(
    q: Tensor,
    native_k: Tensor,
    native_v: Tensor,
    fused_k: Tensor,
    fused_v: Tensor,
    a: Tensor,
    legal: Tensor,
    parent_bias: Tensor | None,
    parent_mask: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    b, hq, t, d = q.shape
    n, k = a.shape[1:]
    has_support = legal.any(dim=-1)
    parent_bias_full = broadcast_parent_term(parent_bias, q, n)
    parent_mask_full = broadcast_parent_term(parent_mask, q, n)
    parent_finite = torch.isfinite(parent_mask_full)

    native_k_hq = expand_kv_heads(native_k, hq)
    native_v_hq = expand_kv_heads(native_v, hq)
    native_logits = _attention_logits(q, native_k_hq) + parent_bias_full + parent_mask_full
    native_active = (~has_support)[:, None, None, :] & parent_finite

    child_k_hq = expand_kv_heads(fused_k.reshape(b, fused_k.shape[1], n * k, d), hq).reshape(b, hq, n, k, d)
    child_v_hq = expand_kv_heads(fused_v.reshape(b, fused_v.shape[1], n * k, d), hq).reshape(b, hq, n, k, d)
    child_logits = torch.einsum("bhtd,bhnkd->bhtnk", q, child_k_hq) / sqrt(d)
    child_logits = child_logits + parent_bias_full.unsqueeze(-1) + parent_mask_full.unsqueeze(-1)
    safe_a = torch.where(legal, a, torch.ones_like(a))
    log_a = torch.where(legal, torch.log(safe_a), torch.full_like(a, -torch.inf))
    child_logits = child_logits + log_a[:, None, None, :, :]
    child_active = legal[:, None, None, :, :] & parent_finite.unsqueeze(-1)

    flat_logits = torch.cat((native_logits, child_logits.reshape(b, hq, t, n * k)), dim=-1)
    flat_active = torch.cat((native_active, child_active.reshape(b, hq, t, n * k)), dim=-1)
    flat_probability = masked_softmax(flat_logits, flat_active)
    native_probability = flat_probability[..., :n]
    child_probability = flat_probability[..., n:].reshape(b, hq, t, n, k)
    output = torch.einsum("bhtn,bhnd->bhtd", native_probability, native_v_hq)
    output = output + torch.einsum("bhtnk,bhnkd->bhtd", child_probability, child_v_hq)
    return output, native_probability, child_probability


def f_flat(
    q: Tensor,
    native_k: Tensor,
    native_v: Tensor,
    source_k: Tensor,
    source_v: Tensor,
    prior: Tensor,
    valid: Tensor,
    fuser: Fuser,
    parent_bias: Tensor | None = None,
    parent_mask: Tensor | None = None,
) -> OperatorResult:
    validate_inputs(q, native_k, native_v, source_k, source_v, prior, valid)
    fused_k, fused_v, a, legal, _has_support = candidate_fused(native_k, native_v, source_k, source_v, prior, valid, fuser)
    output, native_probability, child_probability = _flat_factorized_attention(
        q, native_k, native_v, fused_k, fused_v, a, legal, parent_bias, parent_mask
    )
    return OperatorResult(output, native_probability, child_probability, fused_k, fused_v, a, legal)


def f_hierarchical(
    q: Tensor,
    native_k: Tensor,
    native_v: Tensor,
    source_k: Tensor,
    source_v: Tensor,
    prior: Tensor,
    valid: Tensor,
    fuser: Fuser,
    parent_bias: Tensor | None = None,
    parent_mask: Tensor | None = None,
) -> OperatorResult:
    validate_inputs(q, native_k, native_v, source_k, source_v, prior, valid)
    fused_k, fused_v, a, legal, has_support = candidate_fused(native_k, native_v, source_k, source_v, prior, valid, fuser)
    b, hq, t, d = q.shape
    n, k = a.shape[1:]
    bias = broadcast_parent_term(parent_bias, q, n)
    mask = broadcast_parent_term(parent_mask, q, n)
    parent_finite = torch.isfinite(mask)

    native_k_hq = expand_kv_heads(native_k, hq)
    native_v_hq = expand_kv_heads(native_v, hq)
    native_logits = _attention_logits(q, native_k_hq) + bias + mask
    native_active = (~has_support)[:, None, None, :] & parent_finite

    child_k_hq = expand_kv_heads(fused_k.reshape(b, fused_k.shape[1], n * k, d), hq).reshape(b, hq, n, k, d)
    child_v_hq = expand_kv_heads(fused_v.reshape(b, fused_v.shape[1], n * k, d), hq).reshape(b, hq, n, k, d)
    raw_child = torch.einsum("bhtd,bhnkd->bhtnk", q, child_k_hq) / sqrt(d)
    safe_a = torch.where(legal, a, torch.ones_like(a))
    within_logits = raw_child + torch.where(legal, torch.log(safe_a), torch.full_like(a, -torch.inf))[:, None, None, :, :]
    child_active = legal[:, None, None, :, :] & parent_finite.unsqueeze(-1)
    gamma = masked_softmax(within_logits, child_active)
    safe_within = torch.where(child_active, within_logits, torch.full_like(within_logits, -torch.inf))
    any_child = child_active.any(dim=-1)
    parent_partition = torch.where(any_child, torch.logsumexp(safe_within, dim=-1), torch.full_like(raw_child[..., 0], -torch.inf))
    transport_parent_logits = parent_partition + bias + mask
    parent_logits = torch.cat((native_logits, transport_parent_logits), dim=-1)
    parent_active = torch.cat((native_active, has_support[:, None, None, :] & parent_finite), dim=-1)
    parent_probability = masked_softmax(parent_logits, parent_active)
    native_probability = parent_probability[..., :n]
    transport_parent_probability = parent_probability[..., n:]
    child_probability = transport_parent_probability.unsqueeze(-1) * gamma
    output = torch.einsum("bhtn,bhnd->bhtd", native_probability, native_v_hq)
    output = output + torch.einsum("bhtnk,bhnkd->bhtd", child_probability, child_v_hq)
    return OperatorResult(output, native_probability, child_probability, fused_k, fused_v, a, legal)


def replicated_collapse(
    q: Tensor,
    native_k: Tensor,
    native_v: Tensor,
    source_k: Tensor,
    source_v: Tensor,
    prior: Tensor,
    valid: Tensor,
    fuser: Fuser,
    parent_bias: Tensor | None = None,
    parent_mask: Tensor | None = None,
) -> OperatorResult:
    validate_inputs(q, native_k, native_v, source_k, source_v, prior, valid)
    fused_k, fused_v, a, legal, _has_support = candidate_fused(native_k, native_v, source_k, source_v, prior, valid, fuser)
    weight = a[:, None, :, :, None]
    post_k = (fused_k * weight).sum(dim=3)
    post_v = (fused_v * weight).sum(dim=3)
    replicated_k = post_k.unsqueeze(3).expand_as(fused_k)
    replicated_v = post_v.unsqueeze(3).expand_as(fused_v)
    output, native_probability, child_probability = _flat_factorized_attention(
        q, native_k, native_v, replicated_k, replicated_v, a, legal, parent_bias, parent_mask
    )
    return OperatorResult(output, native_probability, child_probability, replicated_k, replicated_v, a, legal)


def jensen_gap(logits: Tensor, prior: Tensor, valid: Tensor) -> Tensor:
    a, legal, _ = normalize_prior(prior, valid)
    safe_a = torch.where(legal, a, torch.ones_like(a))
    lhs = torch.logsumexp(torch.where(legal, logits + torch.log(safe_a), torch.full_like(logits, -torch.inf)), dim=-1)
    rhs = (a * torch.where(legal, logits, torch.zeros_like(logits))).sum(dim=-1)
    return lhs - rhs


def general_f_with_native_sibling(
    q: Tensor,
    native_k: Tensor,
    native_v: Tensor,
    fused_k: Tensor,
    fused_v: Tensor,
    prior: Tensor,
    valid: Tensor,
    g: Tensor,
    parent_bias: Tensor | None = None,
    parent_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Future-only general reference for the exact g=0 recovery invariant."""
    a, legal, _has_support = normalize_prior(prior, valid)
    b, hq, t, d = q.shape
    n, k = prior.shape[1:]
    bias = broadcast_parent_term(parent_bias, q, n)
    mask = broadcast_parent_term(parent_mask, q, n)
    finite = torch.isfinite(mask)
    g_full = torch.broadcast_to(g, (b, 1, 1, n)).to(q.dtype)

    native_k_hq = expand_kv_heads(native_k, hq)
    native_v_hq = expand_kv_heads(native_v, hq)
    native_logits = _attention_logits(q, native_k_hq) + bias + mask
    native_legal = (g_full < 1) & finite
    safe_one_minus = torch.where(native_legal, 1.0 - g_full, torch.ones_like(g_full))
    native_logits = native_logits + torch.where(native_legal, torch.log(safe_one_minus), torch.full_like(safe_one_minus, -torch.inf))

    child_k_hq = expand_kv_heads(fused_k.reshape(b, fused_k.shape[1], n * k, d), hq).reshape(b, hq, n, k, d)
    child_v_hq = expand_kv_heads(fused_v.reshape(b, fused_v.shape[1], n * k, d), hq).reshape(b, hq, n, k, d)
    child_logits = torch.einsum("bhtd,bhnkd->bhtnk", q, child_k_hq) / sqrt(d) + bias.unsqueeze(-1) + mask.unsqueeze(-1)
    child_legal = legal[:, None, None, :, :] & (g_full > 0).unsqueeze(-1) & finite.unsqueeze(-1)
    safe_a = torch.where(legal, a, torch.ones_like(a))
    safe_g = torch.where(g_full > 0, g_full, torch.ones_like(g_full))
    child_logits = child_logits + torch.log(safe_a)[:, None, None, :, :] + torch.log(safe_g).unsqueeze(-1)
    logits = torch.cat((native_logits, child_logits.reshape(b, hq, t, n * k)), dim=-1)
    active = torch.cat((native_legal.expand(b, hq, t, n), child_legal.reshape(b, hq, t, n * k)), dim=-1)
    p = masked_softmax(logits, active)
    p_native = p[..., :n]
    p_child = p[..., n:].reshape(b, hq, t, n, k)
    output = torch.einsum("bhtn,bhnd->bhtd", p_native, native_v_hq) + torch.einsum("bhtnk,bhnkd->bhtd", p_child, child_v_hq)
    return output, p_native, p_child
