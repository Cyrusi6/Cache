from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache

from rosetta.model.fpct_instrumentation import (
    FPCTCaptureAccumulator,
    teacher_forced_query_mask,
)
from rosetta.model.wrapper import RosettaModel


GEOMETRY_KEYS = (
    "source_d_k",
    "source_d_v",
    "source_energy_k",
    "source_energy_v",
    "fused_d_k",
    "fused_d_v",
    "fused_energy_k",
    "fused_energy_v",
)

PRIMITIVE_KEYS = {
    "batch_index",
    "layer",
    "query_head",
    "kv_head",
    "query_position",
    "parent_position",
    "candidate_count",
    "prior",
    "runtime_source_indices",
    "candidate_valid_mask",
    "candidate_slot_weights",
    "gamma",
    *GEOMETRY_KEYS,
    "candidate_logit_range",
    "candidate_logit_variance",
    "jensen_gap",
    "parent_attention_mass",
    "output_delta_l2",
}


def _qwen(num_key_value_heads: int = 2) -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=num_key_value_heads,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        use_cache=True,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(7103)
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model


def _geometry(parents: int) -> dict[str, torch.Tensor]:
    return {
        name: torch.arange(1, parents + 1, dtype=torch.float32)[None, :]
        * (index + 1) / 10.0
        for index, name in enumerate(GEOMETRY_KEYS)
    }


def _store_fixture_sidecar(
    wrapper: RosettaModel, *, with_geometry: bool
) -> None:
    generator = torch.Generator().manual_seed(8101)
    key = torch.randn(1, 2, 4, 4, 8, generator=generator)
    value = torch.randn(1, 2, 4, 4, 8, generator=generator)
    prior = torch.tensor(
        [[[0.6, 0.4, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.25, 0.75, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]
    )
    source_indices = torch.tensor(
        [[[11, 12, -1, -1], [13, -1, -1, -1], [14, 15, -1, -1], [16, -1, -1, -1]]]
    )
    wrapper._store_fpct_sidecar(
        0,
        0,
        key,
        value,
        prior,
        prior > 0,
        source_indices=source_indices,
        capture_geometry=_geometry(4) if with_geometry else None,
    )


def _captured(operator: str) -> tuple[dict, torch.Tensor]:
    wrapper = RosettaModel([_qwen()], fpct_operator=operator)
    _store_fixture_sidecar(wrapper, with_geometry=True)
    input_ids = torch.tensor([[4, 5, 6, 7]])
    labels = input_ids.clone()
    mask = torch.ones_like(input_ids)
    wrapper.begin_fpct_capture(
        "teacher_forced_response",
        query_mask=teacher_forced_query_mask(labels),
    )
    output = wrapper._base_model_forward_with_fpct(
        input_ids=input_ids,
        attention_mask=mask,
        labels=labels,
        past_key_values=DynamicCache(),
        use_cache=True,
        return_dict=True,
    )
    capture = wrapper._fpct_capture
    assert capture is not None
    compact = capture.layers[0].primitive_chunks[0]
    assert compact["index"].shape[1] == 4
    assert compact["gamma"].ndim == compact["prior"].ndim == 2
    assert compact["gamma"].shape[-1] == 4
    assert compact["source_indices"].shape == compact["valid"].shape
    assert compact["source_indices_certified"] is True
    return wrapper.end_fpct_capture(), output.logits


class _CollectingPrimitiveSink:
    def __init__(self) -> None:
        self.chunks: list[dict] = []
        self.aborted = False
        self.finalized = False

    def write_primitive_chunk(self, chunk) -> None:
        assert not self.aborted
        assert not self.finalized
        self.chunks.append(chunk)

    def finalize(self) -> dict:
        assert not self.aborted
        self.finalized = True
        sizes = [int(chunk["index"].shape[0]) for chunk in self.chunks]
        return {
            "complete": True,
            "row_count": sum(sizes),
            "chunk_count": len(sizes),
            "max_chunk_rows": max(sizes),
            "sink_kind": "synthetic_collecting_sink",
        }

    def abort(self) -> None:
        self.aborted = True


@pytest.mark.parametrize("operator", ["c_post", "f"])
def test_capture_on_off_is_bitwise_observational_for_real_qwen(
    operator: str,
) -> None:
    reference_model = _qwen()
    state = reference_model.state_dict()
    baseline_model = _qwen()
    baseline_model.load_state_dict(state)
    captured_model = _qwen()
    captured_model.load_state_dict(state)
    baseline = RosettaModel([baseline_model], fpct_operator=operator)
    captured = RosettaModel([captured_model], fpct_operator=operator)
    _store_fixture_sidecar(baseline, with_geometry=False)
    _store_fixture_sidecar(captured, with_geometry=True)
    input_ids = torch.tensor([[4, 5, 6, 7]])
    labels = input_ids.clone()
    mask = torch.ones_like(input_ids)
    rng_before = torch.random.get_rng_state().clone()
    baseline_output = baseline._base_model_forward_with_fpct(
        input_ids=input_ids,
        attention_mask=mask,
        labels=labels,
        past_key_values=DynamicCache(),
        use_cache=True,
        return_dict=True,
    )
    rng_after_baseline = torch.random.get_rng_state().clone()
    assert torch.equal(rng_before, rng_after_baseline)
    captured.begin_fpct_capture(
        "teacher_forced_response",
        query_mask=teacher_forced_query_mask(labels),
    )
    captured_output = captured._base_model_forward_with_fpct(
        input_ids=input_ids,
        attention_mask=mask,
        labels=labels,
        past_key_values=DynamicCache(),
        use_cache=True,
        return_dict=True,
    )
    captured.end_fpct_capture()
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert torch.equal(baseline_output.logits, captured_output.logits)
    assert torch.equal(baseline_output.loss, captured_output.loss)
    for baseline_key, captured_key, baseline_value, captured_value in zip(
        baseline_output.past_key_values.key_cache,
        captured_output.past_key_values.key_cache,
        baseline_output.past_key_values.value_cache,
        captured_output.past_key_values.value_cache,
    ):
        assert torch.equal(baseline_key, captured_key)
        assert torch.equal(baseline_value, captured_value)


@pytest.mark.parametrize("operator", ["c_post", "f"])
def test_bounded_long_form_primitives_have_complete_contract(operator: str) -> None:
    report, logits = _captured(operator)
    assert torch.isfinite(logits).all()
    assert report["stores_raw_kv"] is False
    assert report["long_form_contract_version"] == 1
    assert report["long_form_incomplete_chunk_count"] == 0
    rows = report["long_form_primitives"]
    assert report["long_form_row_count"] == len(rows)
    assert report["long_form_row_count"] <= report["max_long_form_rows"]
    # Causal visibility: parent 0 is visible to q=0,1,2; parent 2 only to q=2.
    assert len(rows) == (3 + 1) * 4
    assert all(set(row) == PRIMITIVE_KEYS for row in rows)
    assert {row["query_position"] for row in rows} == {0, 1, 2}
    assert {row["parent_position"] for row in rows} == {0, 2}
    assert all(row["kv_head"] == row["query_head"] // 2 for row in rows)
    for row in rows:
        assert row["candidate_count"] == 2
        assert len(row["runtime_source_indices"]) == 4
        assert row["candidate_valid_mask"] == [True, True, False, False]
        assert row["runtime_source_indices"][2:] == [-1, -1]
        assert math.isclose(sum(row["prior"]), 1.0, abs_tol=1e-6)
        assert math.isclose(sum(row["gamma"]), 1.0, abs_tol=1e-6)
        assert 0.0 <= row["parent_attention_mass"] <= 1.0
        assert all(math.isfinite(float(row[name])) for name in GEOMETRY_KEYS)


def test_long_form_row_ceiling_fails_closed_before_cpu_materialization() -> None:
    wrapper = RosettaModel([_qwen()], fpct_operator="f")
    _store_fixture_sidecar(wrapper, with_geometry=True)
    input_ids = torch.tensor([[4, 5, 6, 7]])
    wrapper.begin_fpct_capture(
        "teacher_forced_response",
        query_mask=teacher_forced_query_mask(input_ids),
        max_long_form_rows=3,
    )
    with pytest.raises(RuntimeError, match="row ceiling exceeded"):
        wrapper._base_model_forward_with_fpct(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=input_ids,
            past_key_values=DynamicCache(),
            use_cache=True,
            return_dict=True,
        )
    capture = wrapper._fpct_capture
    assert capture is not None
    assert capture.long_form_row_count == 0
    assert capture.layers[0].primitive_chunks == []
    with pytest.raises(RuntimeError, match="failed closed"):
        wrapper.end_fpct_capture()


def test_exact_streaming_capture_uses_bounded_sink_without_report_row_list() -> None:
    wrapper = RosettaModel([_qwen()], fpct_operator="f")
    _store_fixture_sidecar(wrapper, with_geometry=True)
    input_ids = torch.tensor([[4, 5, 6, 7]])
    sink = _CollectingPrimitiveSink()
    wrapper.begin_fpct_capture(
        "teacher_forced_response",
        query_mask=teacher_forced_query_mask(input_ids),
        expected_long_form_rows=16,
        primitive_sink=sink,
        primitive_chunk_rows=3,
        detail_mode="aggregate_only",
    )
    output = wrapper._base_model_forward_with_fpct(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
        past_key_values=DynamicCache(),
        use_cache=True,
        return_dict=True,
    )
    capture = wrapper._fpct_capture
    assert capture is not None
    assert capture.layers[0].primitive_chunks == []
    assert capture.layers[0].query_chunks == []
    report = wrapper.end_fpct_capture()

    assert torch.isfinite(output.logits).all()
    assert [int(chunk["index"].shape[0]) for chunk in sink.chunks] == [3, 3, 3, 3, 3, 1]
    assert {chunk["layer"] for chunk in sink.chunks} == {0}
    indices = torch.cat([chunk["index"] for chunk in sink.chunks], dim=0)
    assert indices.tolist() == sorted(indices.tolist())
    assert sink.finalized is True
    assert sink.aborted is False
    assert report["long_form_contract_version"] == 2
    assert report["long_form_row_count"] == 16
    assert report["expected_long_form_rows"] == 16
    assert report["primitive_chunk_rows"] == 3
    assert report["detail_mode"] == "aggregate_only"
    assert report["long_form_stream"]["row_count"] == 16
    assert "long_form_primitives" not in report
    assert "query_summaries" not in report["layers"]["0"]
    assert "parent_summaries" not in report["layers"]["0"]


def test_exact_streaming_count_mismatch_and_overflow_fail_closed() -> None:
    ids = torch.tensor([[4, 5, 6, 7]])

    short_wrapper = RosettaModel([_qwen()], fpct_operator="f")
    _store_fixture_sidecar(short_wrapper, with_geometry=True)
    short_sink = _CollectingPrimitiveSink()
    short_wrapper.begin_fpct_capture(
        "teacher_forced_response",
        query_mask=teacher_forced_query_mask(ids),
        expected_long_form_rows=17,
        primitive_sink=short_sink,
        detail_mode="aggregate_only",
    )
    short_wrapper._base_model_forward_with_fpct(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        labels=ids,
        past_key_values=DynamicCache(),
        use_cache=True,
        return_dict=True,
    )
    with pytest.raises(RuntimeError, match="exact logical row count mismatch"):
        short_wrapper.end_fpct_capture()
    assert short_sink.aborted is True
    assert short_sink.finalized is False

    overflow_wrapper = RosettaModel([_qwen()], fpct_operator="f")
    _store_fixture_sidecar(overflow_wrapper, with_geometry=True)
    overflow_sink = _CollectingPrimitiveSink()
    overflow_wrapper.begin_fpct_capture(
        "teacher_forced_response",
        query_mask=teacher_forced_query_mask(ids),
        expected_long_form_rows=15,
        primitive_sink=overflow_sink,
        detail_mode="aggregate_only",
    )
    with pytest.raises(RuntimeError, match="row ceiling exceeded"):
        overflow_wrapper._base_model_forward_with_fpct(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            labels=ids,
            past_key_values=DynamicCache(),
            use_cache=True,
            return_dict=True,
        )
    assert overflow_sink.chunks == []
    assert overflow_sink.aborted is True


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_exact_streaming_contract_rejects_invalid_counts(invalid) -> None:
    wrapper = RosettaModel([_qwen()], fpct_operator="f")
    with pytest.raises(ValueError, match="expected_long_form_rows"):
        wrapper.begin_fpct_capture(
            "prefill",
            expected_long_form_rows=invalid,
            primitive_sink=_CollectingPrimitiveSink(),
            detail_mode="aggregate_only",
        )


def test_streaming_contract_rejects_missing_sink_and_oversized_chunk() -> None:
    wrapper = RosettaModel([_qwen()], fpct_operator="f")
    with pytest.raises(ValueError, match="requires a primitive sink"):
        wrapper.begin_fpct_capture(
            "prefill", expected_long_form_rows=1, detail_mode="aggregate_only"
        )
    with pytest.raises(ValueError, match=r"\[1, 4096\]"):
        wrapper.begin_fpct_capture(
            "prefill",
            expected_long_form_rows=1,
            primitive_sink=_CollectingPrimitiveSink(),
            primitive_chunk_rows=4097,
            detail_mode="aggregate_only",
        )


def test_streaming_exact_count_is_not_capped_by_the_legacy_memory_guard() -> None:
    sink = _CollectingPrimitiveSink()
    capture = FPCTCaptureAccumulator(
        "prefill",
        expected_long_form_rows=1_000_000,
        primitive_sink=sink,
        detail_mode="aggregate_only",
    )
    assert capture.expected_long_form_rows == 1_000_000
    assert capture.max_long_form_rows == 1_000_000
    capture._abort_sink()
    assert sink.aborted is True

    with pytest.raises(ValueError, match="legacy row ceiling"):
        FPCTCaptureAccumulator(
            "prefill",
            max_long_form_rows=1_000_000,
            expected_long_form_rows=1_000_000,
            primitive_sink=_CollectingPrimitiveSink(),
            detail_mode="aggregate_only",
        )


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_long_form_row_ceiling_rejects_non_positive_or_non_integer(invalid) -> None:
    wrapper = RosettaModel([_qwen()], fpct_operator="f")
    with pytest.raises(ValueError, match="positive integer"):
        wrapper.begin_fpct_capture("prefill", max_long_form_rows=invalid)


def test_cpost_capture_is_exact_expanded_null() -> None:
    report, _logits = _captured("c_post")
    for row in report["long_form_primitives"]:
        torch.testing.assert_close(
            torch.tensor(row["gamma"]),
            torch.tensor(row["prior"]),
            atol=1e-7,
            rtol=1e-7,
        )
        assert abs(row["candidate_logit_range"]) <= 1e-7
        assert abs(row["candidate_logit_variance"]) <= 1e-7
        assert abs(row["jensen_gap"]) <= 1e-7
        assert row["output_delta_l2"] == 0.0


def test_mqa_capture_maps_every_query_head_to_kv_head_zero() -> None:
    wrapper = RosettaModel([_qwen(num_key_value_heads=1)], fpct_operator="f")
    generator = torch.Generator().manual_seed(8121)
    key = torch.randn(1, 1, 4, 2, 8, generator=generator)
    value = torch.randn(1, 1, 4, 2, 8, generator=generator)
    prior = torch.tensor(
        [[[0.6, 0.4], [1.0, 0.0], [0.25, 0.75], [1.0, 0.0]]]
    )
    wrapper._store_fpct_sidecar(
        0, 0, key, value, prior, prior > 0, capture_geometry=_geometry(4)
    )
    ids = torch.tensor([[4, 5, 6, 7]])
    wrapper.begin_fpct_capture(
        "teacher_forced_response", query_mask=teacher_forced_query_mask(ids)
    )
    wrapper._base_model_forward_with_fpct(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        labels=ids,
        past_key_values=DynamicCache(),
        use_cache=True,
        return_dict=True,
    )
    rows = wrapper.end_fpct_capture()["long_form_primitives"]
    assert {row["query_head"] for row in rows} == {0, 1, 2, 3}
    assert {row["kv_head"] for row in rows} == {0}


def test_centered_lambda_is_f_only_and_preserves_default_identity() -> None:
    factorized = RosettaModel([_qwen()], fpct_operator="f")
    assert factorized.fpct_centered_lambda == 1.0
    assert "centered_lambda" not in factorized.fpct_config_dict()
    generator = torch.Generator().manual_seed(9109)
    key = torch.randn(1, 2, 3, 2, 8, generator=generator)
    value = torch.randn(1, 2, 3, 2, 8, generator=generator)
    prior = torch.tensor([[[0.25, 0.75], [0.5, 0.5], [0.8, 0.2]]])
    collapsed_key = (key * prior[:, None, :, :, None]).sum(dim=3)
    collapsed_value = (value * prior[:, None, :, :, None]).sum(dim=3)

    identity_key, identity_value = factorized._apply_fpct_centered_lambda(
        key, value, collapsed_key, collapsed_value
    )
    assert identity_key is key
    assert identity_value is value

    factorized.set_fpct_centered_lambda(0.0)
    zero_key, zero_value = factorized._apply_fpct_centered_lambda(
        key, value, collapsed_key, collapsed_value
    )
    assert torch.equal(zero_key, collapsed_key.unsqueeze(3).expand_as(key))
    assert torch.equal(zero_value, collapsed_value.unsqueeze(3).expand_as(value))

    factorized.set_fpct_centered_lambda(0.5)
    half_key, half_value = factorized._apply_fpct_centered_lambda(
        key, value, collapsed_key, collapsed_value
    )
    torch.testing.assert_close(
        (half_key * prior[:, None, :, :, None]).sum(dim=3),
        collapsed_key,
        atol=2e-7,
        rtol=2e-7,
    )
    torch.testing.assert_close(
        (half_value * prior[:, None, :, :, None]).sum(dim=3),
        collapsed_value,
        atol=2e-7,
        rtol=2e-7,
    )
    assert factorized.fpct_config_dict()["centered_lambda"] == 0.5

    collapsed = RosettaModel([_qwen()], fpct_operator="c_post")
    with pytest.raises(ValueError, match="F-only"):
        collapsed.set_fpct_centered_lambda(0.5)
    with pytest.raises(ValueError, match="must be one of"):
        factorized.set_fpct_centered_lambda(0.125)
    factorized.begin_fpct_capture("prefill")
    with pytest.raises(RuntimeError, match="active capture"):
        factorized.set_fpct_centered_lambda(1.0)
    factorized.end_fpct_capture()


class _Carrier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = type("Config", (), {"num_hidden_layers": 1})()
        self.model = type("Backbone", (), {"layers": []})()

    @property
    def device(self):
        return self.anchor.device

    @property
    def dtype(self):
        return self.anchor.dtype


class _AffineProjector(nn.Module):
    alignment_weight_calibration_mode = "none"
    learned_alignment_mode = "none"
    learned_alignment_injection_gate_mode = "none"
    learned_alignment_transfer_gate_mode = "none"

    def uses_internal_source_confidence(self) -> bool:
        return False

    def _compute_alignment_confidence(self):
        raise AssertionError("compatibility marker only")

    def forward(
        self,
        source_kv,
        target_kv,
        fpct_parent_nuisance=None,
        fpct_capture_parent_nuisance=False,
        **_kwargs,
    ):
        source_key, source_value = source_kv
        target_key, target_value = target_kv
        if fpct_capture_parent_nuisance:
            shape = (*target_key.shape[:-1], 1)
            one = torch.ones(shape, dtype=target_key.dtype)
            self._fpct_last_parent_nuisance = {
                "legacy_key_gate": one,
                "legacy_value_gate": one,
                "key_alignment_confidence": one,
                "value_alignment_confidence": one,
            }
        return target_key + source_key, target_value + source_value


def _projection_record(lambda_value: float):
    projector = _AffineProjector()
    wrapper = RosettaModel(
        [_Carrier(), _Carrier()],
        projector_list=[projector],
        fpct_operator="f",
        fpct_centered_lambda=lambda_value,
    )
    wrapper.begin_fpct_capture("prefill")
    source_key = torch.tensor([[[[1.0, 0.0], [-1.0, 2.0], [0.5, -0.5]]]])
    source_value = torch.tensor([[[[0.0, 1.0], [2.0, -1.0], [1.0, 0.5]]]])
    base_key = torch.zeros(1, 1, 2, 2)
    base_value = torch.zeros_like(base_key)
    indices = torch.tensor([[[0, 1], [2, -1]]])
    weights = torch.tensor([[[0.25, 0.75], [1.0, 0.0]]])
    record = wrapper._project_fpct_candidates(
        projector=projector,
        source_key_cache=source_key,
        source_value_cache=source_value,
        base_kv=(base_key, base_value),
        source_indices=indices,
        source_weights=weights,
        soft_section={
            "source_indices": indices,
            "source_weights": weights,
        },
    )
    wrapper.end_fpct_capture()
    return record


def test_lambda_zero_recomputes_parent_equivalence_after_transform() -> None:
    lambda_one = _projection_record(1.0)
    lambda_zero = _projection_record(0.0)
    assert torch.equal(
        lambda_zero[2], lambda_zero[0].unsqueeze(3).expand_as(lambda_zero[2])
    )
    assert torch.equal(
        lambda_zero[3], lambda_zero[1].unsqueeze(3).expand_as(lambda_zero[3])
    )
    assert bool(lambda_zero[9].all())
    # The contraction tap is before lambda, so lambda=0 cannot erase evidence
    # that candidate-specific fusion produced distinct atoms.
    assert lambda_zero[10]["fused_d_k"][0, 0] > 0
    for name in GEOMETRY_KEYS:
        torch.testing.assert_close(
            lambda_zero[10][name], lambda_one[10][name], atol=0, rtol=0
        )


def _mixed_centered_sidecar_from_native(
    wrapper: RosettaModel,
    native_key: torch.Tensor,
    native_value: torch.Tensor,
) -> None:
    """Install mixed m0/m1/m2 atoms whose prior center is native K/V."""

    batch, heads, parents, dim = native_key.shape
    assert batch == 2 and parents == 4
    prior = torch.tensor(
        [
            [[0.25, 0.75], [1.0, 0.0], [0.0, 0.0], [0.5, 0.5]],
            [[0.0, 0.0], [0.6, 0.4], [1.0, 0.0], [0.2, 0.8]],
        ],
        dtype=torch.float32,
    )
    legal = prior > 0
    candidate_key = native_key.unsqueeze(3).expand(
        batch, heads, parents, 2, dim
    ).clone()
    candidate_value = native_value.unsqueeze(3).expand_as(candidate_key).clone()
    generator = torch.Generator().manual_seed(9419 + heads)
    key_delta = torch.randn(native_key.shape, generator=generator) * 0.125
    value_delta = torch.randn(native_value.shape, generator=generator) * 0.125
    for batch_index in range(batch):
        for parent_index in range(parents):
            if int(legal[batch_index, parent_index].sum()) != 2:
                continue
            first = float(prior[batch_index, parent_index, 0])
            second = float(prior[batch_index, parent_index, 1])
            candidate_key[batch_index, :, parent_index, 0] += key_delta[
                batch_index, :, parent_index
            ]
            candidate_key[batch_index, :, parent_index, 1] -= (
                first / second
            ) * key_delta[batch_index, :, parent_index]
            candidate_value[batch_index, :, parent_index, 0] += value_delta[
                batch_index, :, parent_index
            ]
            candidate_value[batch_index, :, parent_index, 1] -= (
                first / second
            ) * value_delta[batch_index, :, parent_index]
    candidate_key, candidate_value = wrapper._apply_fpct_centered_lambda(
        candidate_key,
        candidate_value,
        native_key,
        native_value,
    )
    candidate_key_equal = (
        candidate_key == native_key.unsqueeze(3)
    ).all(dim=(1, 4))
    candidate_value_equal = (
        candidate_value == native_value.unsqueeze(3)
    ).all(dim=(1, 4))
    parent_equivalent = (
        (~legal) | (candidate_key_equal & candidate_value_equal)
    ).all(dim=-1)
    wrapper._store_fpct_sidecar(
        0,
        0,
        candidate_key,
        candidate_value,
        prior,
        legal,
        parent_equivalent=parent_equivalent,
    )


def _production_centered_forward(
    *,
    state: dict[str, torch.Tensor],
    num_key_value_heads: int,
    operator: str,
    lambda_value: float,
    suffix_ids: torch.Tensor | None = None,
):
    model = _qwen(num_key_value_heads=num_key_value_heads)
    model.load_state_dict(state)
    wrapper = RosettaModel(
        [model],
        fpct_operator=operator,
        fpct_centered_lambda=lambda_value,
    )
    prefix_ids = torch.tensor([[4, 5, 6, 7], [0, 0, 8, 9]])
    prefix_mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
    with torch.no_grad():
        prefix = wrapper._base_model_forward_with_fpct(
            input_ids=prefix_ids,
            attention_mask=prefix_mask,
            past_key_values=DynamicCache(),
            use_cache=True,
            return_dict=True,
        )
    native_key = prefix.past_key_values.key_cache[0][:, :, :4].clone()
    native_value = prefix.past_key_values.value_cache[0][:, :, :4].clone()
    _mixed_centered_sidecar_from_native(wrapper, native_key, native_value)
    with torch.no_grad():
        output = wrapper._base_model_forward_with_fpct(
            input_ids=(
                torch.tensor([[10, 11], [12, 13]])
                if suffix_ids is None
                else suffix_ids
            ),
            attention_mask=torch.tensor(
                [[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]]
            ),
            past_key_values=prefix.past_key_values,
            use_cache=True,
            return_dict=True,
        )
    return wrapper, output


@pytest.mark.parametrize("num_key_value_heads", [1, 2])
def test_centered_lambda_production_qwen_dynamic_cache_oracles(
    num_key_value_heads: int,
) -> None:
    reference = _qwen(num_key_value_heads=num_key_value_heads)
    state = reference.state_dict()

    cpost, cpost_output = _production_centered_forward(
        state=state,
        num_key_value_heads=num_key_value_heads,
        operator="c_post",
        lambda_value=1.0,
    )
    lambda_zero, lambda_zero_output = _production_centered_forward(
        state=state,
        num_key_value_heads=num_key_value_heads,
        operator="f",
        lambda_value=0.0,
    )
    assert torch.equal(cpost_output.logits, lambda_zero_output.logits)
    for cpost_key, lambda_key, cpost_value, lambda_value in zip(
        cpost_output.past_key_values.key_cache,
        lambda_zero_output.past_key_values.key_cache,
        cpost_output.past_key_values.value_cache,
        lambda_zero_output.past_key_values.value_cache,
    ):
        assert torch.equal(cpost_key, lambda_key)
        assert torch.equal(cpost_value, lambda_value)
    assert lambda_zero._fpct_packed_layout is not None
    assert bool((lambda_zero._fpct_packed_layout.extra_slots > 0).all())
    _, causal_control = _production_centered_forward(
        state=state,
        num_key_value_heads=num_key_value_heads,
        operator="f",
        lambda_value=0.0,
        suffix_ids=torch.tensor([[10, 21], [12, 22]]),
    )
    # Changing only the second suffix token cannot alter the first-query logits.
    assert torch.equal(
        lambda_zero_output.logits[:, 0], causal_control.logits[:, 0]
    )

    lambda_default, default_output = _production_centered_forward(
        state=state,
        num_key_value_heads=num_key_value_heads,
        operator="f",
        lambda_value=1.0,
    )
    lambda_explicit, explicit_output = _production_centered_forward(
        state=state,
        num_key_value_heads=num_key_value_heads,
        operator="f",
        lambda_value=1,
    )
    assert "centered_lambda" not in lambda_default.fpct_config_dict()
    assert "centered_lambda" not in lambda_explicit.fpct_config_dict()
    assert torch.equal(default_output.logits, explicit_output.logits)
    for default_key, explicit_key, default_value, explicit_value in zip(
        default_output.past_key_values.key_cache,
        explicit_output.past_key_values.key_cache,
        default_output.past_key_values.value_cache,
        explicit_output.past_key_values.value_cache,
    ):
        assert torch.equal(default_key, explicit_key)
        assert torch.equal(default_value, explicit_value)
