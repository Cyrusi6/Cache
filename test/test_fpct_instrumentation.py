from __future__ import annotations

import json
import math

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache

from rosetta.model.fpct_attention import (
    FPCTSidecarSegment,
    fpct_eager_attention,
    fpct_mechanism_diagnostics,
    pack_fpct_memory,
)
from rosetta.model.fpct_instrumentation import (
    FPCTCaptureAccumulator,
    teacher_forced_query_mask,
)
from rosetta.model.wrapper import RosettaModel


def _packed(
    candidate_key: torch.Tensor,
    candidate_value: torch.Tensor,
    prior: torch.Tensor,
    *,
    query_length: int,
):
    batch, heads, parents, _top_k, dimension = candidate_key.shape
    cache_key = candidate_key[..., 0, :].clone()
    cache_value = candidate_value[..., 0, :].clone()
    sidecar = FPCTSidecarSegment(
        parent_start=0,
        key=candidate_key,
        value=candidate_value,
        prior=prior.float(),
        valid=prior > 0,
    )
    return pack_fpct_memory(
        cache_key,
        cache_value,
        torch.zeros(batch, 1, query_length, parents),
        [sidecar],
        query_length=query_length,
    )


def _diagnose(
    accumulator: FPCTCaptureAccumulator,
    layer: int,
    query: torch.Tensor,
    packed,
) -> dict[str, torch.Tensor]:
    metrics, payload = fpct_mechanism_diagnostics(
        query, packed, return_capture_payload=True
    )
    factorized, _ = fpct_eager_attention(query, packed)
    metrics["output_delta_l2"] = factorized.float().square().mean().sqrt()
    accumulator.update(layer, metrics, payload)
    return metrics


def test_identical_candidates_have_zero_posterior_diagnostics() -> None:
    key = torch.tensor([[[[[1.0, -0.5], [1.0, -0.5]]]]])
    value = torch.tensor([[[[[0.25, 2.0], [0.25, 2.0]]]]])
    prior = torch.tensor([[[0.3, 0.7]]])
    packed = _packed(key, value, prior, query_length=3)
    query = torch.tensor([[[[1.0, 0.0], [-1.0, 0.5], [0.5, 1.0]]]])
    metrics = fpct_mechanism_diagnostics(query, packed)
    assert metrics["gamma_kl_prior"].abs() < 1e-7
    assert metrics["gamma_tv_prior"].abs() < 1e-7
    assert metrics["jensen_gap"].abs() < 1e-7

    factorized, _ = fpct_eager_attention(query, packed)
    collapsed = value[..., 0, :].expand_as(factorized.transpose(1, 2))
    assert torch.allclose(factorized, collapsed.transpose(1, 2), atol=1e-7, rtol=0)


def test_cross_forward_welford_detects_query_change_and_top1_change() -> None:
    key = torch.tensor([[[[[2.0, 0.0], [-2.0, 0.0]]]]])
    value = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]])
    prior = torch.tensor([[[0.5, 0.5]]])
    packed = _packed(key, value, prior, query_length=1)
    capture = FPCTCaptureAccumulator(
        "teacher_forced_response",
        metadata={"sample": "synthetic"},
        query_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    _diagnose(capture, 0, torch.tensor([[[[1.0, 0.0]]]]), packed)
    _diagnose(capture, 0, torch.tensor([[[[-1.0, 0.0]]]]), packed)
    report = capture.finalize()

    assert report["forward_count"] == 2
    assert report["layers"]["0"]["query_count"] == 2
    assert report["metrics"]["gamma_query_variance"] > 0
    assert report["metrics"]["posterior_top1_any_change"] is True
    assert report["metrics"]["posterior_top1_change_rate"] > 0
    assert len(report["layers"]["0"]["query_summaries"]) == 2
    assert report["layers"]["0"]["parent_summaries"][0]["parent_position"] == 0
    assert report["layers"]["0"]["parent_summaries"][0]["candidate_count"] == 2
    assert len(
        report["layers"]["0"]["parent_summaries"][0][
            "candidate_gamma_moments"
        ]
    ) == 2
    assert report["stores_raw_kv"] is False


def test_candidate_permutation_and_refinement_are_invariant() -> None:
    query = torch.tensor([[[[0.7, -0.2], [-0.1, 0.9]]]])
    key = torch.tensor([[[[[1.0, 0.0], [-0.5, 0.5]]]]])
    value = torch.tensor([[[[[1.0, 2.0], [-1.0, 0.5]]]]])
    prior = torch.tensor([[[0.4, 0.6]]])
    base, _ = fpct_eager_attention(query, _packed(key, value, prior, query_length=2))
    permutation = torch.tensor([1, 0])
    permuted, _ = fpct_eager_attention(
        query,
        _packed(
            key[..., permutation, :],
            value[..., permutation, :],
            prior[..., permutation],
            query_length=2,
        ),
    )
    assert torch.allclose(base, permuted, atol=1e-7, rtol=1e-7)

    refined_key = torch.cat((key[..., :1, :], key), dim=-2)
    refined_value = torch.cat((value[..., :1, :], value), dim=-2)
    refined_prior = torch.tensor([[[0.2, 0.2, 0.6]]])
    refined, _ = fpct_eager_attention(
        query,
        _packed(refined_key, refined_value, refined_prior, query_length=2),
    )
    assert torch.allclose(base, refined, atol=1e-7, rtol=1e-7)


def test_invalid_candidate_has_exact_zero_gradient_and_lambda_zero_collapse() -> None:
    query = torch.tensor([[[[0.5, 1.0]]]], requires_grad=True)
    key = torch.tensor(
        [[[[[1.0, 0.0], [-1.0, 0.0], [100.0, 100.0]]]]],
        requires_grad=True,
    )
    value = torch.tensor(
        [[[[[1.0, 2.0], [-2.0, 1.0], [100.0, 100.0]]]]],
        requires_grad=True,
    )
    prior = torch.tensor([[[0.5, 0.5, 0.0]]])
    packed = _packed(key, value, prior, query_length=1)
    output, probability = fpct_eager_attention(query, packed)
    output.sum().backward()
    assert probability.shape[-1] == 2
    assert torch.equal(key.grad[..., 2, :], torch.zeros_like(key.grad[..., 2, :]))
    assert torch.equal(value.grad[..., 2, :], torch.zeros_like(value.grad[..., 2, :]))

    parent_key = torch.tensor([[[[0.3, -0.2]]]])
    parent_value = torch.tensor([[[[2.0, -1.0]]]])
    sidecar = FPCTSidecarSegment(
        0,
        key.detach(),
        value.detach(),
        prior.float(),
        prior > 0,
    )
    replicated = pack_fpct_memory(
        parent_key,
        parent_value,
        torch.zeros(1, 1, 1, 1),
        [sidecar],
        query_length=1,
        replicated_collapse=True,
    )
    collapsed_output, _ = fpct_eager_attention(query.detach(), replicated)
    assert torch.equal(collapsed_output, parent_value.transpose(1, 2))


def _qwen_model(state=None) -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        use_cache=True,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(1701)
    model = Qwen3ForCausalLM(config)
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    return model


def _wrapper_sidecar(wrapper: RosettaModel) -> None:
    generator = torch.Generator().manual_seed(719)
    key = torch.randn(1, 2, 4, 2, 8, generator=generator)
    value = torch.randn(1, 2, 4, 2, 8, generator=generator)
    prior = torch.tensor([[[0.6, 0.4], [1.0, 0.0], [0.7, 0.3], [1.0, 0.0]]])
    wrapper._store_fpct_sidecar(0, 0, key, value, prior, prior > 0)


def _qwen_forward(
    wrapper: RosettaModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    cache: DynamicCache | None = None,
    labels: torch.Tensor | None = None,
):
    return wrapper._base_model_forward_with_fpct(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache or DynamicCache(),
        labels=labels,
        use_cache=True,
        return_dict=True,
    )


def _assert_cache_equal(left: DynamicCache, right: DynamicCache) -> None:
    assert left.get_seq_length() == right.get_seq_length()
    for left_key, right_key, left_value, right_value in zip(
        left.key_cache, right.key_cache, left.value_cache, right.value_cache
    ):
        assert torch.equal(left_key, right_key)
        assert torch.equal(left_value, right_value)


def test_explicit_capture_is_bitwise_observational_and_accumulates_decode() -> None:
    state = _qwen_model().state_dict()
    baseline = RosettaModel([_qwen_model(state)], fpct_operator="f")
    captured = RosettaModel([_qwen_model(state)], fpct_operator="f")
    _wrapper_sidecar(baseline)
    _wrapper_sidecar(captured)
    ids = torch.tensor([[4, 5, 6, 7]])
    mask = torch.ones(1, 4, dtype=torch.long)
    labels = ids.clone()

    baseline_output = _qwen_forward(baseline, ids, mask, labels=labels)
    captured.begin_fpct_capture(
        "teacher_forced_response",
        metadata={"seed": 45, "task": "synthetic", "sample": "parity"},
        query_mask=teacher_forced_query_mask(labels),
    )
    captured_output = _qwen_forward(captured, ids, mask, labels=labels)
    assert torch.equal(baseline_output.logits, captured_output.logits)
    assert torch.equal(baseline_output.loss, captured_output.loss)
    _assert_cache_equal(
        baseline_output.past_key_values, captured_output.past_key_values
    )
    report = captured.end_fpct_capture()
    json.dumps(report, sort_keys=True)
    assert report["mode"] == "teacher_forced_response"
    assert report["metadata"]["sample"] == "parity"
    assert report["layers"]["0"]["query_count"] == 4
    assert report["layers"]["0"]["eligible_query_count"] == 3
    assert not captured.fpct_capture_active

    captured.begin_fpct_capture("greedy_decode")
    first = _qwen_forward(captured, ids, mask)
    second = _qwen_forward(
        captured,
        torch.tensor([[8]]),
        torch.ones(1, 5, dtype=torch.long),
        cache=first.past_key_values,
    )
    _qwen_forward(
        captured,
        torch.tensor([[9]]),
        torch.ones(1, 6, dtype=torch.long),
        cache=second.past_key_values,
    )
    decode_report = captured.end_fpct_capture()
    assert decode_report["forward_count"] == 3
    assert decode_report["layers"]["0"]["query_count"] == 6
    assert len(decode_report["layers"]["0"]["query_summaries"]) == 6 * 4
    assert math.isfinite(decode_report["metrics"]["gamma_query_variance"])


def test_capture_lifecycle_rejects_nesting_and_unmatched_end() -> None:
    wrapper = RosettaModel([_qwen_model()], fpct_operator="f")
    with pytest.raises(RuntimeError, match="no FPCT capture"):
        wrapper.end_fpct_capture()
    wrapper.begin_fpct_capture("prefill")
    with pytest.raises(RuntimeError, match="already active"):
        wrapper.begin_fpct_capture("greedy_decode")
    empty = wrapper.end_fpct_capture()
    assert empty["forward_count"] == 0
    with pytest.raises(ValueError, match="capture mode"):
        wrapper.begin_fpct_capture("unknown")


def test_teacher_forced_query_mask_uses_causal_shift_and_excludes_prompt() -> None:
    labels = torch.tensor([[-100, -100, 5, 6, -100]])
    mask = teacher_forced_query_mask(labels)
    assert mask.tolist() == [[False, True, True, False, False]]
    with pytest.raises(ValueError, match="requires an explicit"):
        FPCTCaptureAccumulator("teacher_forced_response")

    key = torch.tensor([[[[[2.0, 0.0], [-2.0, 0.0]]]]])
    value = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]])
    prior = torch.tensor([[[0.5, 0.5]]])
    packed = _packed(key, value, prior, query_length=5)
    queries = torch.tensor(
        [[[[100.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [100.0, 0.0], [100.0, 0.0]]]]
    )
    capture = FPCTCaptureAccumulator(
        "teacher_forced_response", query_mask=mask
    )
    _diagnose(capture, 0, queries, packed)
    report = capture.finalize()
    layer = report["layers"]["0"]
    assert layer["query_count"] == 5
    assert layer["eligible_query_count"] == 2
    assert {row["query_position"] for row in layer["query_summaries"]} == {1, 2}
    parent = layer["parent_summaries"][0]
    assert parent["query_count"] == 2
    assert parent["gamma_query_variance"] > 0
    assert parent["posterior_top1_any_change"] is True
