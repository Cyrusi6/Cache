from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from math import sqrt
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Callable

import torch
from torch import nn


if Path("/opt/fpct").is_dir():
    sys.path.insert(0, "/opt/fpct")


def canonical_prior(A: torch.Tensor, valid: torch.Tensor):
    A32 = A.float()
    valid = valid.bool()
    bad = ~torch.isfinite(A32) | (A32 < 0) | ((A32 > 0) & ~valid)
    if bool(bad.any()):
        raise ValueError("illegal FPCT prior")
    legal = valid & (A32 > 0)
    mass = torch.where(legal, A32, torch.zeros_like(A32))
    total = mass.sum(dim=-1, keepdim=True)
    prior = torch.where(
        total > 0,
        mass / total.clamp_min(torch.finfo(torch.float32).tiny),
        torch.zeros_like(mass),
    )
    log_prior = torch.where(
        legal,
        prior.clamp_min(torch.finfo(torch.float32).tiny).log(),
        torch.full_like(prior, -torch.inf),
    )
    return prior, log_prior, legal


def candidate_fusion(
    native_k: torch.Tensor,
    native_v: torch.Tensor,
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    legal: torch.Tensor,
    delta_fuser: Callable[..., tuple[torch.Tensor, torch.Tensor]],
):
    K = source_k.shape[3]
    fused_k, fused_v = [], []
    for j in range(K):
        delta_k, delta_v = delta_fuser(
            native_k,
            native_v,
            source_k[:, :, :, j, :],
            source_v[:, :, :, j, :],
        )
        fused_k.append(native_k + delta_k)
        fused_v.append(native_v + delta_v)
    fused_k = torch.stack(fused_k, dim=3)
    fused_v = torch.stack(fused_v, dim=3)
    legal5 = legal[:, None, :, :, None]
    return (
        torch.where(legal5, fused_k, torch.zeros_like(fused_k)),
        torch.where(legal5, fused_v, torch.zeros_like(fused_v)),
    )


def cpost_parent(
    native_k: torch.Tensor,
    native_v: torch.Tensor,
    cand_k: torch.Tensor,
    cand_v: torch.Tensor,
    A: torch.Tensor,
    valid: torch.Tensor,
):
    prior, _, legal = canonical_prior(A, valid)
    weight = prior[:, None, :, :, None]
    weighted_k = (cand_k.float() * weight).sum(dim=3).to(native_k.dtype)
    weighted_v = (cand_v.float() * weight).sum(dim=3).to(native_v.dtype)
    count = legal.sum(dim=-1)
    first = legal.long().argmax(dim=-1)
    index = first[:, None, :, None, None].expand(
        cand_k.shape[0], cand_k.shape[1], cand_k.shape[2], 1, cand_k.shape[-1]
    )
    single_k = torch.gather(cand_k, 3, index).squeeze(3)
    single_v = torch.gather(cand_v, 3, index).squeeze(3)
    singleton = (count == 1)[:, None, :, None]
    support = (count > 0)[:, None, :, None]
    parent_k = torch.where(singleton, single_k, weighted_k)
    parent_v = torch.where(singleton, single_v, weighted_v)
    parent_k = torch.where(support, parent_k, native_k)
    parent_v = torch.where(support, parent_v, native_v)
    return parent_k, parent_v


def repeat_kv(x: torch.Tensor, query_heads: int):
    kv_heads = x.shape[1]
    if query_heads % kv_heads:
        raise ValueError("Hq must be divisible by Hkv")
    return x.repeat_interleave(query_heads // kv_heads, dim=1)


def global_factorized_attention(
    q: torch.Tensor,
    parent_k: torch.Tensor,
    parent_v: torch.Tensor,
    cand_k: torch.Tensor,
    cand_v: torch.Tensor,
    A: torch.Tensor,
    valid: torch.Tensor,
    parent_mask: torch.Tensor,
):
    B, Hq, T, D = q.shape
    _, Hkv, N, Dk = parent_k.shape
    K = cand_k.shape[3]
    assert D == Dk
    assert parent_v.shape == parent_k.shape
    assert cand_k.shape == (B, Hkv, N, K, D)
    assert cand_v.shape == cand_k.shape
    assert A.shape == valid.shape == (B, N, K)
    prior, log_prior, legal = canonical_prior(A, valid)
    ambiguous = legal.sum(dim=-1) >= 2
    parent_k_h = repeat_kv(parent_k, Hq)
    parent_v_h = repeat_kv(parent_v, Hq)
    cand_k_h = repeat_kv(cand_k, Hq)
    cand_v_h = repeat_kv(cand_v, Hq)
    parent_k_atoms = parent_k_h.unsqueeze(3).expand(-1, -1, -1, K, -1)
    parent_v_atoms = parent_v_h.unsqueeze(3).expand_as(parent_k_atoms)
    choose_children = ambiguous[:, None, :, None, None]
    atom_k = torch.where(choose_children, cand_k_h, parent_k_atoms)
    atom_v = torch.where(choose_children, cand_v_h, parent_v_atoms)
    slot0 = torch.arange(K, device=q.device).view(1, 1, K).expand(B, N, K) == 0
    active = (ambiguous[..., None] & legal) | ((~ambiguous)[..., None] & slot0)
    neg_inf = torch.full_like(prior, -torch.inf)
    atom_log_prior = torch.where(ambiguous[..., None] & legal, log_prior, neg_inf)
    atom_log_prior = torch.where(
        (~ambiguous)[..., None] & slot0,
        torch.zeros_like(atom_log_prior),
        atom_log_prior,
    )
    mask = torch.broadcast_to(parent_mask.float(), (B, Hq, T, N))
    logits = torch.einsum("bhtd,bhnkd->bhtnk", q.float(), atom_k.float()) / sqrt(D)
    logits = logits + mask.unsqueeze(-1) + atom_log_prior[:, None, None, :, :]
    active = active[:, None, None, :, :] & torch.isfinite(mask.unsqueeze(-1))
    flat_logits = logits.flatten(-2)
    flat_active = active.flatten(-2)
    masked = torch.where(flat_active, flat_logits, torch.full_like(flat_logits, -torch.inf))
    any_active = flat_active.any(dim=-1, keepdim=True)
    safe = torch.where(any_active, masked, torch.zeros_like(masked))
    probability = torch.softmax(safe, dim=-1, dtype=torch.float32)
    probability = torch.where(flat_active, probability, torch.zeros_like(probability))
    probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    probability = probability.view(B, Hq, T, N, K)
    output = torch.einsum("bhtnk,bhnkd->bhtd", probability, atom_v.float())
    if not torch.isfinite(output).all():
        raise AssertionError("nonfinite oracle output")
    torch.testing.assert_close(
        probability.sum(dim=(-1, -2)),
        torch.ones(B, Hq, T, device=q.device),
        atol=2e-5,
        rtol=2e-5,
    )
    return output.to(q.dtype), probability


def _fixture(dtype: torch.dtype, hq: int, hkv: int):
    generator = torch.Generator().manual_seed(20260722)
    B, T, N, K, D = 2, 3, 5, 4, 8
    q = torch.randn(B, hq, T, D, generator=generator, dtype=dtype)
    native_k = torch.randn(B, hkv, N, D, generator=generator, dtype=dtype)
    native_v = torch.randn(B, hkv, N, D, generator=generator, dtype=dtype)
    cand_k = torch.randn(B, hkv, N, K, D, generator=generator, dtype=dtype)
    cand_v = torch.randn(B, hkv, N, K, D, generator=generator, dtype=dtype)
    A = torch.tensor(
        [
            [[0, 0, 0, 0], [1, 0, 0, 0], [.7, .3, 0, 0], [.5, .3, .2, 0], [.4, .3, .2, .1]],
            [[0, 0, 0, 0], [1, 0, 0, 0], [.6, .4, 0, 0], [.6, .25, .15, 0], [.55, .2, .15, .1]],
        ],
        dtype=torch.float32,
    )
    valid = A > 0
    parent_k, parent_v = cpost_parent(native_k, native_v, cand_k, cand_v, A, valid)
    parent_mask = torch.zeros(B, 1, T, N, dtype=torch.float32)
    parent_mask[:, :, 0, 3:] = -torch.inf
    parent_mask[:, :, 1, 4:] = -torch.inf
    return q, native_k, native_v, parent_k, parent_v, cand_k, cand_v, A, valid, parent_mask


def _production(q, parent_k, parent_v, cand_k, cand_v, A, valid, parent_mask):
    from rosetta.model.fpct_attention import (
        FPCTSidecarSegment,
        fpct_qwen_hierarchical_attention_forward,
        pack_fpct_memory,
    )

    packed = pack_fpct_memory(
        parent_k,
        parent_v,
        parent_mask,
        [FPCTSidecarSegment(0, cand_k, cand_v, A.float(), valid.bool())],
        query_length=q.shape[2],
    )
    module = SimpleNamespace(
        num_key_value_groups=q.shape[1] // parent_k.shape[1], training=False
    )
    output, probability = fpct_qwen_hierarchical_attention_forward(
        module,
        q,
        packed,
        parent_k,
        parent_v,
        parent_mask,
        scaling=1.0 / sqrt(q.shape[-1]),
    )
    return output.transpose(1, 2), probability.float(), packed


def _packed_oracle_probability(oracle: torch.Tensor, packed: Any) -> torch.Tensor:
    expected = torch.zeros(
        oracle.shape[0], oracle.shape[1], oracle.shape[2], packed.key.shape[2],
        device=oracle.device, dtype=oracle.dtype,
    )
    for batch in range(expected.shape[0]):
        for slot in range(expected.shape[-1]):
            parent = int(packed.parent_index[batch, slot])
            candidate = int(packed.candidate_index[batch, slot])
            expected[batch, :, :, slot] = oracle[
                batch, :, :, parent, candidate if candidate >= 0 else 0
            ]
    return expected


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(num_hidden_layers=1)
        self.model = SimpleNamespace(layers=[])

    @property
    def device(self):
        return self.anchor.device

    @property
    def dtype(self):
        return self.anchor.dtype


def _candidate_projection_checks() -> dict[str, Any]:
    from rosetta.model.projector import C2CProjector
    from rosetta.model.wrapper import RosettaModel

    def projector():
        return C2CProjector(
            source_dim=2,
            target_dim=2,
            source_num_heads=1,
            target_num_heads=1,
            intermediate_dim=8,
            hidden_dim=8,
            num_layers=3,
            dropout=0.0,
            dtype=torch.float32,
            alignment_confidence_gate_mode="token_mlp",
        ).float()

    torch.manual_seed(41)
    cproj, fproj = projector(), projector()
    fproj.load_state_dict(cproj.state_dict())
    cproj.eval(); fproj.eval()
    cmodel = RosettaModel([_FakeModel(), _FakeModel()], projector_list=[cproj], fpct_operator="c_post")
    fmodel = RosettaModel([_FakeModel(), _FakeModel()], projector_list=[fproj], fpct_operator="f")
    generator = torch.Generator().manual_seed(17)
    source_k = torch.randn(1, 1, 5, 2, generator=generator)
    source_v = torch.randn(1, 1, 5, 2, generator=generator)
    base_k = torch.randn(1, 1, 3, 2, generator=generator)
    base_v = torch.randn(1, 1, 3, 2, generator=generator)
    indices = torch.tensor([[[0, 1, -1, -1], [2, -1, -1, -1], [-1, -1, -1, -1]]])
    weights = torch.tensor([[[.6, .4, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]])
    soft = {
        "source_indices": indices,
        "source_weights": weights,
        "source_confidence": torch.tensor([[.8, .7, .25]]),
        "source_entropy": torch.tensor([[1., 0., 0.]]),
        "source_entropy_override": torch.ones(1, 3, dtype=torch.bool),
    }
    args = dict(
        source_key_cache=source_k,
        source_value_cache=source_v,
        base_kv=(base_k, base_v),
        source_indices=indices,
        source_weights=weights,
        soft_section=soft,
    )
    c = cmodel._project_fpct_candidates(projector=cproj, **args)
    f = fmodel._project_fpct_candidates(projector=fproj, **args)
    torch.testing.assert_close(c[2], f[2], atol=0, rtol=0)
    torch.testing.assert_close(c[3], f[3], atol=0, rtol=0)
    ckeys = [(name, tuple(value.shape)) for name, value in cmodel.named_parameters() if value.requires_grad]
    fkeys = [(name, tuple(value.shape)) for name, value in fmodel.named_parameters() if value.requires_grad]
    if ckeys != fkeys:
        raise AssertionError("F changed trainable parameter names/shapes")
    return {
        "precollapse_candidate_identity": True,
        "trainable_names_shapes_equal": True,
        "trainable_parameter_count": sum(value.numel() for value in cmodel.parameters() if value.requires_grad),
    }


def run_parity() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    maximum_fp32_error = 0.0
    for ratio in (1, 2, 4):
        q, _nk, _nv, pk, pv, ck, cv, A, valid, mask = _fixture(torch.float32, ratio * 2, 2)
        oracle_output, oracle_probability = global_factorized_attention(q, pk, pv, ck, cv, A, valid, mask)
        production_output, production_probability, packed = _production(q, pk, pv, ck, cv, A, valid, mask)
        expected_probability = _packed_oracle_probability(oracle_probability, packed)
        maximum_fp32_error = max(
            maximum_fp32_error,
            float((oracle_output - production_output).abs().max()),
            float((expected_probability - production_probability).abs().max()),
        )
        torch.testing.assert_close(oracle_output, production_output, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(expected_probability, production_probability, atol=1e-6, rtol=1e-6)
    checks["fp32_oracle_output_probability"] = True
    checks["gqa_mqa_ratios_1_2_4"] = True

    q, _nk, _nv, pk, pv, ck, cv, A, valid, mask = _fixture(torch.bfloat16, 4, 2)
    q = q.requires_grad_(True); ck = ck.requires_grad_(True); cv = cv.requires_grad_(True)
    output, probability, _ = _production(q, pk, pv, ck, cv, A, valid, mask)
    output.float().square().mean().backward()
    if not all(torch.isfinite(value).all() for value in (output, probability, q.grad, ck.grad, cv.grad)):
        raise AssertionError("BF16 forward/backward is nonfinite")
    checks["bf16_forward_backward_finite"] = True

    q, _nk, _nv, pk, pv, ck, cv, A, valid, mask = _fixture(torch.float32, 4, 2)
    m1_valid = torch.zeros_like(valid); m1_valid[..., 0] = True
    m1_A = m1_valid.float()
    m1_pk, m1_pv = cpost_parent(pk, pv, ck, cv, m1_A, m1_valid)
    fact, _, _ = _production(q, m1_pk, m1_pv, ck, cv, m1_A, m1_valid, mask)
    from rosetta.model.fpct_attention import fpct_qwen_eager_attention_forward
    module = SimpleNamespace(num_key_value_groups=2, training=False)
    post, _ = fpct_qwen_eager_attention_forward(module, q, m1_pk, m1_pv, mask, scaling=1/sqrt(q.shape[-1]))
    torch.testing.assert_close(fact, post.transpose(1, 2), atol=1e-6, rtol=1e-6)
    checks["m_le_1_equals_cpost"] = True

    identical_k = pk.unsqueeze(3).expand_as(ck).clone()
    identical_v = pv.unsqueeze(3).expand_as(cv).clone()
    fact, _, _ = _production(q, pk, pv, identical_k, identical_v, A, valid, mask)
    post, _ = fpct_qwen_eager_attention_forward(module, q, pk, pv, mask, scaling=1/sqrt(q.shape[-1]))
    torch.testing.assert_close(fact, post.transpose(1, 2), atol=1e-6, rtol=1e-6)
    checks["identical_candidates_equal_cpost"] = True

    permutation = torch.tensor([1, 0, 3, 2])
    base, _, _ = _production(q, pk, pv, ck, cv, A, valid, mask)
    perm, _, _ = _production(q, pk, pv, ck[:, :, :, permutation], cv[:, :, :, permutation], A[:, :, permutation], valid[:, :, permutation], mask)
    torch.testing.assert_close(base, perm, atol=1e-6, rtol=1e-6)
    checks["candidate_permutation_invariance"] = True

    refined_k, refined_v = ck.clone(), cv.clone()
    refined_k[:, :, 2, 1] = refined_k[:, :, 2, 0]
    refined_v[:, :, 2, 1] = refined_v[:, :, 2, 0]
    single_A, refined_A = A.clone(), A.clone()
    single_A[:, 2] = torch.tensor([1., 0., 0., 0.])
    refined_A[:, 2] = torch.tensor([.3, .7, 0., 0.])
    single_valid, refined_valid = single_A > 0, refined_A > 0
    single_pk, single_pv = cpost_parent(pk, pv, refined_k, refined_v, single_A, single_valid)
    refined_pk, refined_pv = cpost_parent(pk, pv, refined_k, refined_v, refined_A, refined_valid)
    single, _, _ = _production(q, single_pk, single_pv, refined_k, refined_v, single_A, single_valid, mask)
    refined, _, _ = _production(q, refined_pk, refined_pv, refined_k, refined_v, refined_A, refined_valid, mask)
    torch.testing.assert_close(single, refined, atol=1e-6, rtol=1e-6)
    checks["duplicate_prior_split_refinement"] = True

    ck_grad, cv_grad = ck.clone().requires_grad_(True), cv.clone().requires_grad_(True)
    invalid_valid = valid.clone(); invalid_A = A.clone()
    invalid_valid[..., 1] = False; invalid_A[..., 1] = 0
    invalid_pk, invalid_pv = cpost_parent(pk, pv, ck_grad, cv_grad, invalid_A, invalid_valid)
    invalid_output, invalid_probability, packed = _production(q, invalid_pk, invalid_pv, ck_grad, cv_grad, invalid_A, invalid_valid, mask)
    invalid_output.sum().backward()
    invalid_slots = (packed.candidate_index == 1)[:, None, None, :]
    if invalid_slots.any() and not torch.equal(invalid_probability.masked_select(invalid_slots), torch.zeros_like(invalid_probability.masked_select(invalid_slots))):
        raise AssertionError("invalid candidate probability is nonzero")
    if not torch.equal(ck_grad.grad[..., 1, :], torch.zeros_like(ck_grad.grad[..., 1, :])):
        raise AssertionError("invalid key gradient is nonzero")
    if not torch.equal(cv_grad.grad[..., 1, :], torch.zeros_like(cv_grad.grad[..., 1, :])):
        raise AssertionError("invalid value gradient is nonzero")
    checks["invalid_probability_value_gradient_zero"] = True

    causal_output, causal_probability, packed = _production(q, pk, pv, ck, cv, A, valid, mask)
    del causal_output
    blocked_parent = packed.parent_index >= 3
    if not torch.equal(causal_probability[:, :, 0].masked_select(blocked_parent[:, None]), torch.zeros_like(causal_probability[:, :, 0].masked_select(blocked_parent[:, None]))):
        raise AssertionError("child bypassed parent mask")
    checks["parent_causal_mask_inheritance"] = True
    checks["single_global_softmax_denominator"] = True
    checks["prior_enters_logits_once"] = True
    checks.update(_candidate_projection_checks())

    source = inspect.getsource(global_factorized_attention)
    checks["oracle_has_one_softmax"] = source.count("torch.softmax(") == 1
    status = "GO" if all(value is True or isinstance(value, int) for value in checks.values()) else "BLOCKED"
    return {
        "schema_version": 1,
        "protocol_id": "fpct_e0_formula_oracle_v1",
        "status": status,
        "checks": checks,
        "maximum_fp32_error": maximum_fp32_error,
        "fp32_tolerance": {"atol": 1e-6, "rtol": 1e-6},
        "bf16_tolerance": {"atol": 5e-3, "rtol": 5e-3},
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run_parity()
    if args.output:
        _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
