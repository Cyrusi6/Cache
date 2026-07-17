from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rosetta.model.projector import C2CProjector
from rosetta.utils.gate_diagnostics import (
    GateDiagnosticsAccumulator,
    RunningGateStats,
    configure_projector_gate_diagnostics,
    merge_gate_diagnostic_states,
)
from script.analysis import route1_confidence_gate_diagnostics as posthoc_diagnostics
from script.analysis.route1_confidence_gate_diagnostics import _move_batch_to_device


def _record(key_values, value_values):
    return {
        "key_confidence": torch.tensor(key_values, dtype=torch.float32).reshape(
            1, 2, 2, 1
        ),
        "value_confidence": torch.tensor(value_values, dtype=torch.float32).reshape(
            1, 2, 2, 1
        ),
    }


def test_running_gate_stats_reports_population_variance_and_saturation() -> None:
    stats = RunningGateStats()
    stats.update(
        torch.tensor([0.0, 0.25, 0.75, 1.0]),
        low_threshold=0.05,
        high_threshold=0.95,
    )

    summary = stats.finalize()

    assert summary["count"] == 4
    assert summary["mean"] == pytest.approx(0.5)
    assert summary["variance"] == pytest.approx(0.15625)
    assert summary["std"] == pytest.approx(0.15625**0.5)
    assert summary["minimum"] == pytest.approx(0.0)
    assert summary["maximum"] == pytest.approx(1.0)
    assert summary["saturation_low_rate"] == pytest.approx(0.25)
    assert summary["saturation_high_rate"] == pytest.approx(0.25)


def test_accumulator_reduces_layer_head_token_axes_and_clears_records() -> None:
    projectors = [
        SimpleNamespace(
            alignment_confidence_gate_mode="token_mlp",
            alignment_diagnostic_records=[
                _record([0.0, 0.2, 0.4, 1.0], [0.1, 0.3, 0.5, 0.9])
            ],
            last_alignment_key_confidence_tensor=torch.ones(1),
        ),
        SimpleNamespace(
            alignment_confidence_gate_mode="none",
            alignment_diagnostic_records=[],
        ),
        SimpleNamespace(
            alignment_confidence_gate_mode="token_mlp",
            alignment_diagnostic_records=[
                _record([0.6, 0.7, 0.8, 0.9], [0.5, 0.6, 0.7, 0.8])
            ],
        ),
    ]
    model = SimpleNamespace(
        projector_list=projectors,
        projector_dict={0: {1: {2: [(0, 0)], 0: [(1, 1)], 1: [(2, 2)]}}},
    )
    info = configure_projector_gate_diagnostics(model, enabled=True)
    projectors[0].alignment_diagnostic_records.append(
        _record([0.0, 0.2, 0.4, 1.0], [0.1, 0.3, 0.5, 0.9])
    )
    projectors[2].alignment_diagnostic_records.append(
        _record([0.6, 0.7, 0.8, 0.9], [0.5, 0.6, 0.7, 0.8])
    )
    accumulator = GateDiagnosticsAccumulator()
    accumulator.note_projectors(
        info["projector_count"],
        info["gate_projector_count"],
        info["target_layer_by_projector"],
    )

    example = accumulator.consume_projectors(projectors)
    artifact = accumulator.finalize()

    assert info["target_layer_by_projector"] == [2, 0, 1]
    assert example["gate_diagnostics_status"] == "ok"
    assert example["gate_record_count"] == 2
    assert example["gate_token_count"] == 8
    assert all(not projector.alignment_diagnostic_records for projector in projectors)
    assert projectors[0].last_alignment_key_confidence_tensor is None
    assert artifact["status"] == "ok"
    assert artifact["layer_axis"] == "target_model_layer"
    assert {row["layer"] for row in artifact["by_layer"]} == {1, 2}
    assert len(artifact["by_layer_head"]) == 4
    assert len(artifact["by_token_position"]) == 2
    assert len(artifact["by_relative_token_bin"]) == 2
    assert artifact["layer_groups"] == {
        "early": {"start": 0, "end": 0, "count": 1},
        "middle": {"start": 1, "end": 1, "count": 1},
        "late": {"start": 2, "end": 2, "count": 1},
    }


def test_accumulator_marks_non_gated_control_unavailable() -> None:
    projector = SimpleNamespace(
        alignment_confidence_gate_mode="none",
        alignment_diagnostic_records=[],
    )
    accumulator = GateDiagnosticsAccumulator()

    example = accumulator.consume_projectors([projector])
    artifact = accumulator.finalize()

    assert example["gate_diagnostics_status"] == "unavailable"
    assert artifact["status"] == "unavailable"
    assert artifact["unavailable_reason"] == "no_token_head_gate_records"


def test_compact_accumulator_uses_scalar_projector_values_without_raw_records() -> None:
    projectors = [
        SimpleNamespace(
            alignment_confidence_gate_mode="token_mlp",
            last_alignment_key_confidence=torch.tensor(0.25),
            last_alignment_value_confidence=torch.tensor(0.50),
            last_alignment_key_confidence_std=0.1,
            last_alignment_value_confidence_std=0.2,
        ),
        SimpleNamespace(
            alignment_confidence_gate_mode="token_mlp",
            last_alignment_key_confidence=torch.tensor(0.75),
            last_alignment_value_confidence=torch.tensor(1.00),
            last_alignment_key_confidence_std=0.1,
            last_alignment_value_confidence_std=0.2,
        ),
    ]
    accumulator = GateDiagnosticsAccumulator()

    example = accumulator.consume_compact_projectors(projectors)
    artifact = accumulator.finalize()

    assert example["gate_diagnostics_status"] == "compact"
    assert example["gate"] == pytest.approx(0.625)
    assert example["key_gate_mean"] == pytest.approx(0.5)
    assert example["value_gate_mean"] == pytest.approx(0.75)
    assert artifact["status"] == "compact"
    assert artifact["by_layer_head"] == []
    assert artifact["by_relative_token_bin"] == []
    assert projectors[0].last_alignment_key_confidence is None


def test_gate_capture_does_not_change_projector_forward_values() -> None:
    torch.manual_seed(7)
    projector = C2CProjector(
        source_dim=2,
        target_dim=2,
        source_num_heads=1,
        target_num_heads=2,
        hidden_dim=4,
        intermediate_dim=4,
        num_layers=3,
        dropout=0.0,
        alignment_confidence_gate_mode="token_mlp",
    )
    key_hidden = torch.randn(1, 3, 4)
    value_hidden = torch.randn(1, 3, 4)
    source_confidence = torch.tensor([[0.25, 0.5, 0.75]])
    source_weights = torch.tensor([[[1.0, 0.0], [0.5, 0.5], [1.0, 0.0]]])

    projector.capture_alignment_diagnostics = False
    without_capture = projector._compute_alignment_confidence(
        source_confidence=source_confidence,
        source_weights=source_weights,
        key_hidden=key_hidden,
        value_hidden=value_hidden,
        target_shape=(1, 2, 3, 2),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    projector.capture_alignment_diagnostics = True
    with_capture = projector._compute_alignment_confidence(
        source_confidence=source_confidence,
        source_weights=source_weights,
        key_hidden=key_hidden,
        value_hidden=value_hidden,
        target_shape=(1, 2, 3, 2),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert torch.equal(without_capture[0], with_capture[0])
    assert torch.equal(without_capture[1], with_capture[1])
    assert len(projector.alignment_diagnostic_records) == 1


def test_gate_states_merge_without_averaging_worker_means() -> None:
    first = GateDiagnosticsAccumulator()
    second = GateDiagnosticsAccumulator()
    first_projector = SimpleNamespace(
        alignment_confidence_gate_mode="token_mlp",
        alignment_diagnostic_records=[
            _record([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
        ],
    )
    second_projector = SimpleNamespace(
        alignment_confidence_gate_mode="token_mlp",
        alignment_diagnostic_records=[
            _record([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0])
        ],
    )
    first.consume_projectors([first_projector])
    second.consume_projectors([second_projector])

    merged = merge_gate_diagnostic_states(
        [first.state_dict(), second.state_dict()]
    ).finalize()

    assert merged["counts"]["examples_seen"] == 2
    assert merged["global"]["combined"]["mean"] == pytest.approx(0.5)
    assert merged["global"]["combined"]["variance"] == pytest.approx(0.25)


def test_posthoc_batch_move_preserves_entropy_counterfactual_overrides() -> None:
    batch = {
        "input_ids": [torch.tensor([[1]]), torch.tensor([[2]])],
        "attention_mask": [torch.ones(1, 1), torch.ones(1, 1)],
        "position_ids": torch.tensor([[0]]),
        "labels": torch.tensor([[1]]),
        "kv_cache_index": [torch.tensor([[[1, 0]]])],
        "soft_alignment": [
            {
                "source_indices": torch.tensor([[[0]]]),
                "source_weights": torch.tensor([[[1.0]]]),
                "source_confidence": torch.tensor([[0.5]]),
                "source_entropy": torch.tensor([[0.75]]),
                "source_entropy_override": torch.tensor([[True]]),
            }
        ],
    }

    moved = _move_batch_to_device(batch, torch.device("cpu"))

    section = moved["soft_alignment"][0]
    assert section["source_entropy"].item() == pytest.approx(0.75)
    assert section["source_entropy_override"].item() is True


def test_posthoc_teacher_tokenizer_uses_resolved_offline_model_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = SimpleNamespace(chat_template="existing")
    calls = []
    monkeypatch.setattr(
        posthoc_diagnostics,
        "resolve_model_path",
        lambda model_id: calls.append(("resolve", model_id)) or "/netdisk/models/qwen",
    )
    monkeypatch.setattr(
        posthoc_diagnostics.AutoTokenizer,
        "from_pretrained",
        lambda path: calls.append(("load", path)) or tokenizer,
    )
    monkeypatch.setattr(
        posthoc_diagnostics,
        "set_default_chat_template",
        lambda value, model_id: calls.append(("template", value, model_id)),
    )

    loaded = posthoc_diagnostics._load_teacher_tokenizer("Qwen/Qwen3-1.7B")

    assert loaded is tokenizer
    assert calls == [
        ("resolve", "Qwen/Qwen3-1.7B"),
        ("load", "/netdisk/models/qwen"),
        ("template", tokenizer, "Qwen/Qwen3-1.7B"),
    ]


def test_posthoc_dataset_removes_gold_answer_and_enables_generation_prompt() -> None:
    class FakeAligner:
        def __init__(self) -> None:
            self.call = None

        def align_chat_messages_soft(self, messages, **kwargs):
            self.call = (messages, kwargs)
            return {
                "slm_ids": [10, 11, 12],
                "llm_ids": [20, 21, 22],
                "message_mask": [False, True, False],
                "soft_alignment": {
                    "source_indices": [[-1], [1], [-1]],
                    "source_weights": [[0.0], [1.0], [0.0]],
                    "source_confidence": [1.0, 0.5, 1.0],
                    "source_entropy": [0.0, 0.0, 0.0],
                    "source_entropy_override": [False, False, False],
                },
            }

    aligner = FakeAligner()
    dataset = posthoc_diagnostics.InferencePromptAlignedDataset(
        [[
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "gold answer"},
        ]],
        aligner=aligner,
        max_length=16,
        soft_alignment_top_k=4,
    )

    item = dataset[0]

    assert item["messages"] == [{"role": "user", "content": "question"}]
    assert aligner.call is not None
    messages, kwargs = aligner.call
    assert messages == item["messages"]
    assert kwargs["add_generation_prompt"] is True
    assert kwargs["top_k"] == 4
    assert item["labels"] == [-100, -100, -100]
    assert item["kv_cache_index"].tolist() == [[-1, 0], [1, 0], [-1, 0]]
