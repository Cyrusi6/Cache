from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rosetta.model.projector import C2CProjector
from rosetta.utils.cache_geometry import (
    cache_geometry_runtime,
    capture_projector_cache_geometry,
    configure_projector_cache_geometry,
    consume_cache_geometry_records,
)


def _projector() -> C2CProjector:
    torch.manual_seed(11)
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=2,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
    )
    projector.eval()
    with torch.no_grad():
        projector.key_gate_logit.fill_(1.0)
        projector.value_gate_logit.fill_(1.0)
    return projector


def _projector_inputs(batch_size: int = 1):
    source_key = torch.arange(
        batch_size * 1 * 3 * 2, dtype=torch.float32
    ).reshape(batch_size, 1, 3, 2)
    source_value = source_key + 0.5
    target_key = torch.linspace(
        0.25, 2.0, batch_size * 2 * 3 * 2, dtype=torch.float32
    ).reshape(batch_size, 2, 3, 2)
    target_value = target_key + 1.0
    return (source_key, source_value), (target_key, target_value)


def test_cache_geometry_off_is_bitwise_unchanged() -> None:
    projector = _projector()
    source_kv, target_kv = _projector_inputs()
    model = SimpleNamespace(projector_list=[projector])

    projector.train()
    torch.manual_seed(101)
    without_capture = projector(source_kv, target_kv)
    rng_without_capture = torch.random.get_rng_state()
    configure_projector_cache_geometry(model, enabled=False)
    torch.manual_seed(101)
    disabled = projector(source_kv, target_kv)
    rng_disabled = torch.random.get_rng_state()

    assert torch.equal(without_capture[0], disabled[0])
    assert torch.equal(without_capture[1], disabled[1])
    assert torch.equal(rng_without_capture, rng_disabled)
    assert projector.cache_geometry_records == []

    projector.eval()
    without_capture = projector(source_kv, target_kv)
    configure_projector_cache_geometry(model, enabled=True)
    with cache_geometry_runtime({"question_id": "q0"}):
        with_capture = projector(source_kv, target_kv)

    assert torch.equal(without_capture[0], with_capture[0])
    assert torch.equal(without_capture[1], with_capture[1])
    assert len(projector.cache_geometry_records) == 1

    configure_projector_cache_geometry(model, enabled=False)
    disabled_again = projector(source_kv, target_kv)
    assert torch.equal(without_capture[0], disabled_again[0])
    assert torch.equal(without_capture[1], disabled_again[1])
    assert projector.cache_geometry_records == []


def test_projector_capture_records_detached_scalar_samples_and_context() -> None:
    projector = _projector()
    source_kv, target_kv = _projector_inputs(batch_size=2)
    source_weights = torch.tensor(
        [
            [[1.0, 0.0], [0.5, 0.5], [0.0, 0.0]],
            [[0.8, 0.2], [1.0, 0.0], [0.25, 0.75]],
        ]
    )
    source_confidence = torch.tensor([[1.0, 0.5, 0.0], [0.8, 0.9, 0.7]])
    model = SimpleNamespace(
        projector_list=[projector],
        projector_dict={0: {1: {7: [(3, 0)]}}},
    )

    info = configure_projector_cache_geometry(model, enabled=True)
    with cache_geometry_runtime(
        {"pair": "teacher_to_receiver", "seed": 42},
        sample_contexts=[
            {"question_id": "q0", "source_length": 6, "receiver_length": 3},
            {"question_id": "q1", "source_length": 4, "receiver_length": 2},
        ],
    ):
        with cache_geometry_runtime(source_layer=3, target_layer=99):
            projector(
                source_kv,
                target_kv,
                source_weights=source_weights,
                source_confidence=source_confidence,
            )

    records = consume_cache_geometry_records(model)
    assert info["target_layer_by_projector"] == [7]
    assert len(records) == 2
    assert projector.cache_geometry_records == []
    assert [record["question_id"] for record in records] == ["q0", "q1"]
    assert all(record["pair"] == "teacher_to_receiver" for record in records)
    assert all(record["source_layer"] == 3 for record in records)
    assert all(record["target_layer"] == 7 for record in records)
    assert all(record["source_receiver_length_ratio"] == 2.0 for record in records)
    assert records[0]["valid_alignment_mass"] == pytest.approx(2.0 / 3.0)
    assert records[0]["valid_alignment_coverage"] == pytest.approx(2.0 / 3.0)
    for record in records:
        assert record["cache_geometry_schema_version"] == 1
        assert record["projector_index"] == 0
        assert isinstance(record["key_native_norm"], float)
        assert isinstance(record["learned_weight_mean"], float)
        assert isinstance(record["alignment_confidence_std"], float)
        assert isinstance(record["effective_gate_mean"], float)
        assert not any(isinstance(value, torch.Tensor) for value in record.values())


def test_geometry_metrics_cover_ratios_cosines_and_head_concentration() -> None:
    projector = SimpleNamespace(
        capture_cache_geometry=True,
        cache_geometry_records=[],
        _cache_geometry_projector_index=2,
        _cache_geometry_target_layer=5,
    )
    native_key = torch.tensor([[[[1.0]], [[1.0]]]])
    raw_projected_key = torch.tensor([[[[2.0]], [[0.0]]]])
    fused_key = torch.tensor([[[[2.0]], [[1.0]]]])
    native_value = torch.tensor([[[[2.0]], [[2.0]]]])
    raw_projected_value = torch.tensor([[[[1.0]], [[1.0]]]])
    fused_value = torch.tensor([[[[3.0]], [[3.0]]]])
    key_weight = torch.tensor([[[[0.25]], [[0.75]]]])
    value_weight = torch.tensor([[[[0.5]], [[0.5]]]])
    key_confidence = torch.tensor([[[[0.5]], [[1.0]]]])
    value_confidence = torch.ones_like(key_confidence)
    key_effective_gate = torch.tensor([[[[0.5]], [[0.0]]]])
    value_effective_gate = torch.full_like(key_effective_gate, 0.5)
    source_weights = torch.tensor([[[0.5, 0.5], [0.0, 0.0]]])

    capture_projector_cache_geometry(
        projector,
        native_key=native_key,
        native_value=native_value,
        raw_projected_key=raw_projected_key,
        raw_projected_value=raw_projected_value,
        fused_key=fused_key,
        fused_value=fused_value,
        key_weight=key_weight,
        value_weight=value_weight,
        key_confidence=key_confidence,
        value_confidence=value_confidence,
        key_effective_gate=key_effective_gate,
        value_effective_gate=value_effective_gate,
        source_weights=source_weights,
    )

    record = projector.cache_geometry_records[0]
    assert record["key_raw_projected_to_native_norm_ratio"] == pytest.approx(
        2.0 / (2.0**0.5)
    )
    assert record["key_residual_to_native_norm_ratio"] == pytest.approx(
        1.0 / (2.0**0.5)
    )
    assert record["key_native_fused_cosine"] == pytest.approx(3.0 / 10.0**0.5)
    assert record["key_residual_head_energy_hhi"] == pytest.approx(1.0)
    assert record["value_residual_head_energy_hhi"] == pytest.approx(0.5)
    assert record["key_head_residual_cv"] == pytest.approx(1.0)
    assert record["source_weight_top1_mean"] == pytest.approx(0.25)
    assert record["source_weight_entropy_mean"] == pytest.approx(0.5)
    assert record["source_weight_hhi_mean"] == pytest.approx(0.25)
    assert record["valid_alignment_mass"] == pytest.approx(0.5)
    assert record["valid_alignment_coverage"] == pytest.approx(0.5)


def test_nested_runtime_requires_matching_sample_context_lengths() -> None:
    with cache_geometry_runtime(sample_contexts=[{"question_id": "q0"}]):
        with pytest.raises(ValueError, match="matching lengths"):
            with cache_geometry_runtime(
                sample_contexts=[{"question_id": "q0"}, {"question_id": "q1"}]
            ):
                pass
