from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

import script.evaluation.unified_evaluator as evaluator_module
from rosetta.model.wrapper import RosettaModel
from rosetta.utils.cache_geometry import current_cache_geometry_runtime
from script.evaluation.unified_evaluator import (
    ContentGroupFilter,
    UnifiedEvaluator,
    cache_geometry_output_record,
    content_group_hash_for_example,
)


def _example(question: str) -> dict:
    return {
        "question": question,
        "choices": ["alpha", "beta", "gamma", "delta"],
        "answer": 0,
    }


def _content_manifest(tmp_path, fit_example: dict, test_example: dict):
    path = tmp_path / "content-groups.json"
    path.write_text(
        json.dumps(
            {
                "role": "content_group_split_manifest",
                "created_without_outcome_fields": True,
                "groups": [
                    {
                        "content_hash": content_group_hash_for_example(
                            "mmlu-redux", fit_example
                        ),
                        "split": "fit",
                        "members": [
                            {
                                "task": "mmlu-redux",
                                "subject": "subject",
                                "question_id": "0",
                            }
                        ],
                    },
                    {
                        "content_hash": content_group_hash_for_example(
                            "mmlu-redux", test_example
                        ),
                        "split": "test",
                        "members": [
                            {
                                "task": "mmlu-redux",
                                "subject": "subject",
                                "question_id": "1",
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_content_group_filter_is_input_only_and_fit_scoped(tmp_path) -> None:
    fit_example = _example("  Which   answer? ")
    test_example = _example("Held-out question")
    manifest = _content_manifest(tmp_path, fit_example, test_example)
    content_filter = ContentGroupFilter(
        manifest,
        split="fit",
        dataset="mmlu-redux",
    )

    same_input_different_outcome = {
        **fit_example,
        "question": "Which answer?",
        "answer": 3,
        "is_correct": False,
    }
    assert content_filter.match(same_input_different_outcome) == (
        content_group_hash_for_example("mmlu-redux", fit_example)
    )
    assert content_filter.match(test_example) is None


def test_arc_content_hash_matches_prediction_csv_four_choice_contract() -> None:
    five_choice = {
        "question": "Which option?",
        "choices": {
            "label": ["A", "B", "C", "D", "E"],
            "text": ["a", "b", "c", "d", "ignored-e"],
        },
    }
    four_choice = {
        "question": "Which option?",
        "choices": {
            "label": ["A", "B", "C", "D"],
            "text": ["a", "b", "c", "d"],
        },
    }

    assert content_group_hash_for_example(
        "ai2-arc", five_choice
    ) == content_group_hash_for_example("ai2-arc", four_choice)


def test_content_group_filter_precedes_answer_parse_and_forward(
    tmp_path, monkeypatch
) -> None:
    allowed = _example("Allowed")
    excluded = _example("Excluded")
    manifest = _content_manifest(tmp_path, allowed, excluded)

    evaluator = object.__new__(UnifiedEvaluator)
    evaluator.dataset_name = "mmlu-redux"
    evaluator.dataset_config = {
        "dataset_name": "fixture",
        "test_split": "test",
        "subcategories": {},
        "categories": {},
    }
    evaluator.data_root = None
    evaluator.eval_config = {
        "answer_method": "logits",
        "sample_interval": 1,
        "use_cot": False,
        "use_template": False,
    }
    evaluator.model_config = {"model_name": "fixture"}
    evaluator.output_dir = tmp_path
    evaluator.debug_dump_bad_samples = False
    evaluator.content_group_filter = ContentGroupFilter(
        manifest,
        split="fit",
        dataset="mmlu-redux",
    )
    evaluator.cache_geometry_configured = False
    evaluator.cache_geometry_enabled = False
    evaluator.cache_geometry_identity = {
        "role": "geometry_off",
        "pair": "fixture",
        "seed": 0,
        "task": "mmlu-redux",
    }
    evaluator._cache_geometry_records = []
    evaluator._cache_geometry_output_records = []

    monkeypatch.setattr(evaluator_module, "load_c2c_dataset", lambda *args, **kwargs: [excluded])
    monkeypatch.setattr(evaluator_module, "get_option_token_ids", lambda *args, **kwargs: {})

    def _forbidden_parse(_example):
        raise AssertionError("parse_answer must not inspect a non-fit example")

    evaluator.parse_answer = _forbidden_parse
    result = evaluator.evaluate_subject(
        "subject",
        SimpleNamespace(),
        SimpleNamespace(),
        torch.device("cpu"),
    )

    assert result == (None, 0, None, [], [])


def test_cache_geometry_output_hash_uses_only_frozen_prediction_fields() -> None:
    identity = {
        "role": "geometry_on",
        "pair": "tinyllama",
        "seed": 42,
        "task": "mmlu-redux",
        "subject": "algebra",
        "question_id": "7",
        "content_hash": "a" * 64,
    }
    prediction = {
        "pred": "B",
        "cot_pred": "B",
        "cot_output": "reasoning",
        "cot_gen_length": 3,
        "extraction_method_used": None,
        "extracted_normalized": None,
        "true_answer": "A",
        "is_correct": False,
    }
    first = cache_geometry_output_record(identity=identity, prediction=prediction)
    second = cache_geometry_output_record(
        identity=identity,
        prediction={**prediction, "true_answer": "B", "is_correct": True},
    )
    changed = cache_geometry_output_record(
        identity=identity,
        prediction={**prediction, "pred": "C"},
    )

    assert first == second
    assert first["output_sha256"] != changed["output_sha256"]
    assert set(first) == {
        "schema_version",
        "role",
        "pair",
        "seed",
        "task",
        "subject",
        "question_id",
        "content_hash",
        "output_sha256",
    }


def test_expected_fit_rows_are_checked_after_worker_merge() -> None:
    evaluator = object.__new__(UnifiedEvaluator)
    evaluator.gate_diagnostics_enabled = False
    evaluator.content_group_filter = SimpleNamespace(expected_rows=3)
    worker = {
        "all_cors": [],
        "subject_cors": {},
        "subcat_cors": {},
        "cat_cors": {},
        "length_stats": [],
        "cot_logs": [],
        "gate_diagnostics_state": None,
    }

    evaluator.merge_results(
        {
            0: {**worker, "content_group_matched_count": 1},
            1: {**worker, "content_group_matched_count": 2},
        }
    )
    with pytest.raises(ValueError, match="Merged content-group filter row count mismatch"):
        evaluator.merge_results(
            {
                0: {**worker, "content_group_matched_count": 1},
                1: {**worker, "content_group_matched_count": 1},
            }
        )


class _FakeLM(nn.Module):
    def __init__(self, cache_value: float):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.cache_value = float(cache_value)
        self.config = SimpleNamespace(num_hidden_layers=1)

    @property
    def device(self):
        return self.anchor.device

    @property
    def dtype(self):
        return self.anchor.dtype

    def forward(self, input_ids, past_key_values=None, **kwargs):
        cache = past_key_values if past_key_values is not None else DynamicCache()
        batch_size, token_count = input_ids.shape
        key = torch.full(
            (batch_size, 1, token_count, 1),
            self.cache_value,
            device=input_ids.device,
        )
        cache.update(key, key.clone(), 0)
        return CausalLMOutputWithPast(
            logits=torch.zeros(batch_size, token_count, 4),
            past_key_values=cache,
        )


class _OverwriteProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.contexts = []

    def forward(self, source_kv, target_kv, **kwargs):
        runtime = current_cache_geometry_runtime()
        self.contexts.append(dict(runtime.metadata) if runtime is not None else {})
        return (
            torch.full_like(target_kv[0], 9.0),
            torch.full_like(target_kv[1], 11.0),
        )


def test_wrapper_sets_layer_alignment_length_context_and_preserves_overwrite() -> None:
    projector = _OverwriteProjector()
    model = RosettaModel(
        [_FakeLM(1.0), _FakeLM(3.0)],
        projector_list=[projector],
        multi_source_fusion_mode="parallel",
    )
    model.set_projector_config(
        source_model_idx=1,
        source_model_layer_idx=0,
        target_model_idx=0,
        target_model_layer_idx=0,
        projector_idx=0,
    )
    segments = [
        torch.tensor([[[1, 0]]], dtype=torch.long),
        torch.tensor([[[-1, 0]]], dtype=torch.long),
    ]

    output = model.forward(
        kv_cache_index=segments,
        input_ids=torch.tensor([[5, 6]], dtype=torch.long),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        use_cache=True,
    )

    assert output.past_key_values.key_cache[0][0, 0, 0, 0].item() == 9.0
    assert output.past_key_values.value_cache[0][0, 0, 0, 0].item() == 11.0
    assert projector.contexts == [
        {
            "source_model_index": 1,
            "target_model_index": 0,
            "source_layer": 0,
            "target_layer": 0,
            "source_length": 1,
            "receiver_length": 1,
            "alignment_mode": "position_slice",
        }
    ]


@pytest.mark.parametrize("enabled", [False, True])
def test_sidecars_are_independent_from_prediction_csv(tmp_path, enabled) -> None:
    evaluator = object.__new__(UnifiedEvaluator)
    evaluator.model_config = {"model_name": "fixture/model"}
    evaluator.dataset_name = "mmlu-redux"
    evaluator.eval_config = {"answer_method": "generate"}
    evaluator.dataset_config = {"subcategories": {}, "categories": {}}
    evaluator.output_dir = tmp_path / "predictions"
    evaluator.output_dir.mkdir()
    evaluator.eval_intervention = None
    evaluator.cache_geometry_configured = True
    evaluator.cache_geometry_enabled = enabled
    evaluator.cache_geometry_identity = {
        "role": "geometry_on" if enabled else "geometry_off",
        "pair": "tinyllama",
        "seed": 42,
        "task": "mmlu-redux",
    }
    evaluator.cache_geometry_output_dir = tmp_path / "geometry"
    evaluator.cache_geometry_output_dir.mkdir()

    identity = {
        **evaluator.cache_geometry_identity,
        "subject": "subject",
        "question_id": "0",
        "content_hash": "b" * 64,
    }
    prediction = {
        "pred": "A",
        "cot_pred": "A",
        "cot_output": "Answer: A",
        "cot_gen_length": 2,
        "extraction_method_used": None,
        "extracted_normalized": None,
    }
    cot_row = {
        "subject": "subject",
        "question_id": 0,
        "question": "question",
        "true_answer": "A",
        "is_correct": True,
        "answer_method": "generate",
        **prediction,
    }
    layer_record = {
        **identity,
        "cache_geometry_schema_version": 1,
        "projector_index": 0,
        "target_layer": 0,
        "batch_index": 0,
    }
    layer_shards = []
    shard_path = evaluator.cache_geometry_output_dir / ".rank0.jsonl"
    if enabled:
        shard_path.write_text(json.dumps(layer_record) + "\n", encoding="utf-8")
        layer_shards.append({"path": str(shard_path), "record_count": 1, "rank": 0})

    evaluator.save_results(
        [np.array([True])],
        {"subject": 1.0},
        {},
        {},
        [],
        [cot_row],
        None,
        [],
        [cache_geometry_output_record(identity=identity, prediction=prediction)],
        [
            {
                "rank": 0,
                "gpu_id": 0,
                "wall_seconds": 1.25,
                "max_memory_allocated_bytes": 10,
                "max_memory_reserved_bytes": 20,
            }
        ],
        layer_shards,
    )

    csv_path = next(evaluator.output_dir.glob("*_cot.csv"))
    with csv_path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert "content_hash" not in header
    assert "output_sha256" not in header
    assert "source_receiver_length_ratio" not in header

    summary_path = next(evaluator.output_dir.glob("*_summary.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifacts = summary["cache_geometry_artifacts"]
    assert artifacts["role"] == ("geometry_on" if enabled else "geometry_off")
    assert artifacts["samples_jsonl"] is not None
    assert (artifacts.get("layers_jsonl") is not None) is enabled
    if enabled:
        assert not shard_path.exists()
    assert summary["cache_geometry_instrumentation"]["runtime"] == {
        "wall_seconds": 1.25,
        "max_memory_allocated_bytes": 10,
        "max_memory_reserved_bytes": 20,
        "workers": [
            {
                "rank": 0,
                "gpu_id": 0,
                "wall_seconds": 1.25,
                "max_memory_allocated_bytes": 10,
                "max_memory_reserved_bytes": 20,
            }
        ],
    }
