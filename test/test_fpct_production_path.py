from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rosetta.model.fpct_attention import (
    FPCTSidecarSegment,
    fpct_eager_attention,
    normalize_fpct_operator,
    pack_fpct_memory,
)
from rosetta.model.projector import C2CProjector
from rosetta.model.wrapper import RosettaModel


ROOT = Path(__file__).resolve().parents[1]
REF_PATH = ROOT / "script/analysis/fpct_reference_operator.py"
SPEC = importlib.util.spec_from_file_location("fpct_reference_for_production", REF_PATH)
assert SPEC is not None and SPEC.loader is not None
reference = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reference
SPEC.loader.exec_module(reference)


class FakeModel(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=dtype), requires_grad=False)
        self.config = SimpleNamespace(num_hidden_layers=1)
        self.model = SimpleNamespace(layers=[])

    @property
    def device(self):
        return self.anchor.device

    @property
    def dtype(self):
        return self.anchor.dtype


def make_projector(dtype=torch.float64, *, dropout=0.0) -> C2CProjector:
    return C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=1,
        intermediate_dim=8,
        hidden_dim=8,
        num_layers=3,
        dropout=dropout,
        dtype=dtype,
        alignment_confidence_gate_mode="token_mlp",
    ).to(dtype=dtype)


def make_wrapper(projector: nn.Module, operator=None, dtype=torch.float64) -> RosettaModel:
    return RosettaModel(
        [FakeModel(dtype), FakeModel(dtype)],
        projector_list=[projector],
        fpct_operator=operator,
    )


def candidate_inputs(dtype=torch.float64, *, k=4):
    generator = torch.Generator().manual_seed(17)
    source_k = torch.randn(1, 1, 5, 2, generator=generator, dtype=dtype)
    source_v = torch.randn(1, 1, 5, 2, generator=generator, dtype=dtype)
    base_k = torch.randn(1, 1, 3, 2, generator=generator, dtype=dtype)
    base_v = torch.randn(1, 1, 3, 2, generator=generator, dtype=dtype)
    indices = torch.tensor([[[0, 1, -1, -1], [2, -1, -1, -1], [-1, -1, -1, -1]]])[:, :, :k]
    weights = torch.tensor([[[0.6, 0.4, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], dtype=dtype)[:, :, :k]
    soft = {
        "source_indices": indices,
        "source_weights": weights,
        "source_confidence": torch.tensor([[0.8, 0.7, 0.25]], dtype=dtype),
        "source_entropy": torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype),
        "source_entropy_override": torch.ones(1, 3, dtype=torch.bool),
    }
    return source_k, source_v, base_k, base_v, indices, weights, soft


def identity_fuser(base_k, base_v, source_k, source_v):
    return source_k, source_v


def test_operator_flag_validation_and_serialization() -> None:
    assert normalize_fpct_operator(None) is None
    assert normalize_fpct_operator("C_POST") == "c_post"
    with pytest.raises(ValueError):
        normalize_fpct_operator("route3")
    wrapper = make_wrapper(make_projector(), "f")
    assert wrapper.fpct_config_dict() == {
        "operator": "f",
        "position_mode": "legacy",
        "a": "1",
        "g": "1",
    }


def test_state_dict_has_no_operator_specific_parameters() -> None:
    torch.manual_seed(11)
    default = make_wrapper(make_projector(), None)
    torch.manual_seed(11)
    c_pre = make_wrapper(make_projector(), "c_pre")
    torch.manual_seed(11)
    factorized = make_wrapper(make_projector(), "f")
    assert default.state_dict().keys() == c_pre.state_dict().keys() == factorized.state_dict().keys()
    for key in default.state_dict():
        torch.testing.assert_close(default.state_dict()[key], c_pre.state_dict()[key], atol=0, rtol=0)
        torch.testing.assert_close(default.state_dict()[key], factorized.state_dict()[key], atol=0, rtol=0)


def test_legacy_weighted_gather_regression() -> None:
    sk, sv, _bk, _bv, indices, weights, _soft = candidate_inputs()
    key, value = RosettaModel._weighted_source_kv_from_indices(sk, sv, indices, weights)
    expected_k = 0.6 * sk[:, :, 0] + 0.4 * sk[:, :, 1]
    expected_v = 0.6 * sv[:, :, 0] + 0.4 * sv[:, :, 1]
    torch.testing.assert_close(key[:, :, 0], expected_k)
    torch.testing.assert_close(value[:, :, 0], expected_v)
    torch.testing.assert_close(key[:, :, 2], torch.zeros_like(key[:, :, 2]))


def test_candidate_projection_uses_shared_parent_gumbel_nuisance() -> None:
    projector = make_projector(dropout=0.0)
    projector.train()
    wrapper = make_wrapper(projector, "c_post")
    sk, sv, bk, bv, indices, weights, soft = candidate_inputs()
    # Identical candidates expose any accidental K-wise Gumbel sampling.
    sk[:, :, 1] = sk[:, :, 0]
    sv[:, :, 1] = sv[:, :, 0]
    torch.manual_seed(123)
    record = wrapper._project_fpct_candidates(
        projector=projector,
        source_key_cache=sk,
        source_value_cache=sv,
        base_kv=(bk, bv),
        source_indices=indices,
        source_weights=weights,
        soft_section=soft,
    )
    torch.testing.assert_close(record[2][:, :, 0, 0, :], record[2][:, :, 0, 1, :], atol=0, rtol=0)
    torch.testing.assert_close(record[3][:, :, 0, 0, :], record[3][:, :, 0, 1, :], atol=0, rtol=0)
    nuisance = record[7]
    assert nuisance["legacy_key_gate"].shape == (1, 1, 3, 1)
    assert nuisance["key_alignment_confidence"].shape == (1, 1, 3, 1)


def test_cpost_and_f_candidate_fuser_outputs_are_identical() -> None:
    torch.manual_seed(41)
    cpost_projector = make_projector(dropout=0.0)
    f_projector = make_projector(dropout=0.0)
    f_projector.load_state_dict(cpost_projector.state_dict())
    cpost_projector.eval()
    f_projector.eval()
    cpost_wrapper = make_wrapper(cpost_projector, "c_post")
    f_wrapper = make_wrapper(f_projector, "f")
    sk, sv, bk, bv, indices, weights, soft = candidate_inputs()
    cpost = cpost_wrapper._project_fpct_candidates(
        projector=cpost_projector,
        source_key_cache=sk,
        source_value_cache=sv,
        base_kv=(bk, bv),
        source_indices=indices,
        source_weights=weights,
        soft_section=soft,
    )
    factorized = f_wrapper._project_fpct_candidates(
        projector=f_projector,
        source_key_cache=sk,
        source_value_cache=sv,
        base_kv=(bk, bv),
        source_indices=indices,
        source_weights=weights,
        soft_section=soft,
    )
    torch.testing.assert_close(cpost[2], factorized[2], atol=0, rtol=0)
    torch.testing.assert_close(cpost[3], factorized[3], atol=0, rtol=0)


@pytest.mark.parametrize("training", [False, True])
def test_k1_candidate_projection_equals_parent_projection(training: bool) -> None:
    projector = make_projector(dropout=0.0)
    projector.train(training)
    wrapper = make_wrapper(projector, "c_post")
    sk, sv, bk, bv, indices, weights, soft = candidate_inputs(k=1)
    torch.manual_seed(77)
    record = wrapper._project_fpct_candidates(
        projector=projector,
        source_key_cache=sk,
        source_value_cache=sv,
        base_kv=(bk, bv),
        source_indices=indices,
        source_weights=weights,
        soft_section=soft,
    )
    supported = record[5].any(dim=-1)[:, None, :, None]
    torch.testing.assert_close(
        record[0].masked_select(supported),
        record[2][..., 0, :].masked_select(supported),
        atol=1e-10,
        rtol=1e-8,
    )
    torch.testing.assert_close(
        record[1].masked_select(supported),
        record[3][..., 0, :].masked_select(supported),
        atol=1e-10,
        rtol=1e-8,
    )


def make_attention_case(dtype=torch.float64, *, hq=2, hkv=1):
    generator = torch.Generator().manual_seed(99)
    q = torch.randn(1, hq, 2, 3, generator=generator, dtype=dtype)
    native_k = torch.randn(1, hkv, 3, 3, generator=generator, dtype=dtype)
    native_v = torch.randn(1, hkv, 3, 3, generator=generator, dtype=dtype)
    fused_k = torch.randn(1, hkv, 3, 4, 3, generator=generator, dtype=dtype)
    fused_v = torch.randn(1, hkv, 3, 4, 3, generator=generator, dtype=dtype)
    prior = torch.tensor([[[0.6, 0.4, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], dtype=dtype)
    valid = prior > 0
    collapsed_k = (fused_k * prior[:, None, :, :, None]).sum(dim=3)
    collapsed_v = (fused_v * prior[:, None, :, :, None]).sum(dim=3)
    has = valid.any(dim=-1)[:, None, :, None]
    cache_k = torch.where(has, collapsed_k, native_k)
    cache_v = torch.where(has, collapsed_v, native_v)
    parent_term = torch.tensor([[[[0.1, -0.2, -torch.inf], [0.0, 0.3, -0.1]]]], dtype=dtype)
    sidecar = FPCTSidecarSegment(0, fused_k, fused_v, prior, valid)
    return q, native_k, native_v, fused_k, fused_v, prior, valid, cache_k, cache_v, parent_term, sidecar


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float64, 1e-10, 1e-8), (torch.float32, 2e-5, 2e-5)])
def test_packed_production_matches_dense_reference(dtype, atol, rtol) -> None:
    q, nk, nv, fk, fv, prior, valid, ck, cv, term, sidecar = make_attention_case(dtype)
    packed = pack_fpct_memory(ck, cv, term, [sidecar], query_length=q.shape[2])
    production, _ = fpct_eager_attention(q, packed)
    dense = reference.f_flat(q, nk, nv, fk, fv, prior, valid, identity_fuser, term, None)
    torch.testing.assert_close(production, dense.output, atol=atol, rtol=rtol)
    assert packed.extra_slots.tolist() == [1]
    assert packed.expanded_slots.tolist() == [4]


def test_replicated_collapse_packed_equals_ordinary_cpost() -> None:
    q, nk, nv, fk, fv, prior, valid, ck, cv, term, _sidecar = make_attention_case()
    replicated_k = ck.unsqueeze(3).expand_as(fk)
    replicated_v = cv.unsqueeze(3).expand_as(fv)
    packed = pack_fpct_memory(
        ck, cv, term, [FPCTSidecarSegment(0, replicated_k, replicated_v, prior, valid)],
        query_length=q.shape[2],
    )
    expanded, _ = fpct_eager_attention(q, packed)
    post = reference.c_post(q, nk, nv, fk, fv, prior, valid, identity_fuser, term, None)
    torch.testing.assert_close(expanded, post.output, atol=1e-10, rtol=1e-8)


def test_pack_refinement_and_permutation_invariance() -> None:
    q, _nk, _nv, fk, fv, prior, valid, ck, cv, term, sidecar = make_attention_case()
    base, _ = fpct_eager_attention(q, pack_fpct_memory(ck, cv, term, [sidecar], query_length=2))
    permutation = torch.tensor([1, 0, 3, 2])
    permuted_sidecar = FPCTSidecarSegment(
        0, fk[:, :, :, permutation], fv[:, :, :, permutation],
        prior[:, :, permutation], valid[:, :, permutation],
    )
    permuted, _ = fpct_eager_attention(q, pack_fpct_memory(ck, cv, term, [permuted_sidecar], query_length=2))
    torch.testing.assert_close(base, permuted, atol=1e-10, rtol=1e-8)

    refined_k = fk.clone()
    refined_v = fv.clone()
    refined_k[:, :, 0, 1] = refined_k[:, :, 0, 0]
    refined_v[:, :, 0, 1] = refined_v[:, :, 0, 0]
    refined_prior = prior.clone()
    refined_prior[:, 0] = torch.tensor([0.3, 0.7, 0.0, 0.0], dtype=prior.dtype)
    single_prior = prior.clone()
    single_prior[:, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=prior.dtype)
    single = FPCTSidecarSegment(0, refined_k, refined_v, single_prior, single_prior > 0)
    refined = FPCTSidecarSegment(0, refined_k, refined_v, refined_prior, refined_prior > 0)
    refined_cache_k = ck.clone()
    refined_cache_v = cv.clone()
    refined_cache_k[:, :, 0] = refined_k[:, :, 0, 0]
    refined_cache_v[:, :, 0] = refined_v[:, :, 0, 0]
    out_single, _ = fpct_eager_attention(q, pack_fpct_memory(refined_cache_k, refined_cache_v, term, [single], query_length=2))
    out_refined, _ = fpct_eager_attention(q, pack_fpct_memory(refined_cache_k, refined_cache_v, term, [refined], query_length=2))
    torch.testing.assert_close(out_single, out_refined, atol=1e-10, rtol=1e-8)


def test_causal_padding_zero_support_and_all_invalid_stability() -> None:
    q, _nk, _nv, fk, fv, prior, valid, ck, cv, term, sidecar = make_attention_case()
    packed = pack_fpct_memory(ck, cv, term, [sidecar], query_length=2)
    _output, probability = fpct_eager_attention(q, packed)
    parent2 = packed.parent_index == 2
    assert torch.equal(probability[:, :, 0].masked_select(parent2[:, None]), torch.zeros_like(probability[:, :, 0].masked_select(parent2[:, None])))

    all_invalid = FPCTSidecarSegment(0, fk, fv, torch.zeros_like(prior), torch.zeros_like(valid))
    packed_invalid = pack_fpct_memory(ck, cv, term, [all_invalid], query_length=2)
    output, probability = fpct_eager_attention(q, packed_invalid)
    assert torch.isfinite(output).all()
    assert torch.isfinite(probability).all()
    assert packed_invalid.extra_slots.tolist() == [0]


@pytest.mark.parametrize("hq,hkv", [(4, 2), (4, 1), (2, 2)])
def test_packed_gqa_mqa_shapes(hq: int, hkv: int) -> None:
    case = make_attention_case(hq=hq, hkv=hkv)
    packed = pack_fpct_memory(case[7], case[8], case[9], [case[10]], query_length=2)
    output, probability = fpct_eager_attention(case[0], packed)
    assert output.shape == case[0].shape
    assert probability.shape[:3] == case[0].shape[:3]


def test_invalid_candidate_probability_and_gradient_exact_zero() -> None:
    q, _nk, _nv, fk, fv, prior, valid, ck, cv, term, _sidecar = make_attention_case()
    fk = fk.clone().requires_grad_(True)
    fv = fv.clone().requires_grad_(True)
    prior = prior.clone()
    valid = valid.clone()
    valid[..., 1] = False
    prior[:, 0, 2] = 0.4
    valid[:, 0, 2] = True
    sidecar = FPCTSidecarSegment(0, fk, fv, prior, valid)
    packed = pack_fpct_memory(ck, cv, term, [sidecar], query_length=2)
    output, probability = fpct_eager_attention(q, packed)
    output.sum().backward()
    assert torch.equal(fk.grad[..., 1, :], torch.zeros_like(fk.grad[..., 1, :]))
    assert torch.equal(fv.grad[..., 1, :], torch.zeros_like(fv.grad[..., 1, :]))
    assert torch.isfinite(probability).all()


def test_candidate_prior_is_used_once() -> None:
    q, _nk, _nv, fk, fv, prior, valid, ck, cv, term, sidecar = make_attention_case()
    output, _ = fpct_eager_attention(q, pack_fpct_memory(ck, cv, term, [sidecar], query_length=2))
    squared_prior = prior.square()
    wrong = FPCTSidecarSegment(0, fk, fv, squared_prior, valid)
    wrong_output, _ = fpct_eager_attention(q, pack_fpct_memory(ck, cv, term, [wrong], query_length=2))
    assert not torch.allclose(output, wrong_output)


def test_overlapping_sidecar_is_rejected() -> None:
    case = make_attention_case()
    with pytest.raises(ValueError, match="overlapping"):
        case_wrapper = make_wrapper(make_projector(), "f")
        case_wrapper._store_fpct_sidecar(0, 0, case[3], case[4], case[5], case[6])
        case_wrapper._store_fpct_sidecar(0, 1, case[3], case[4], case[5], case[6])


def test_nonfrozen_weight_calibration_and_route3_are_rejected() -> None:
    projector = make_projector()
    wrapper = make_wrapper(projector, "f")
    projector.alignment_weight_calibration_mode = "span_mlp"
    assert not wrapper._fpct_projector_is_supported(projector)
    projector.alignment_weight_calibration_mode = "none"
    projector.learned_alignment_mode = "kv_router"
    assert not wrapper._fpct_projector_is_supported(projector)
