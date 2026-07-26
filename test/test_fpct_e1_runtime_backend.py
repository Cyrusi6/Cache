from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from script.analysis.fpct_e1_mechanism_audit import validate_and_derive_row
from script.experiment.fpct_e1_runtime_backend import (
    GOLD_RESPONSE_TEMPLATE,
    assemble_sample_rows,
    canonical_content_sha256,
    canonical_sample_sha256,
    capture_primitives,
    teacher_forced_capture_bounded,
    topology_contract,
    verify_expected_long_form_row_volume,
)


GROUP = "a" * 64
SAMPLE = "b" * 64


def _primitive(**overrides):
    value = {
        "batch_index": 0,
        "layer": 3,
        "query_head": 2,
        "kv_head": 1,
        "query_position": 8,
        "parent_position": 4,
        "candidate_count": 2,
        "prior": [0.5, 0.5],
        "gamma": [0.7, 0.3],
        "source_d_k": 0.4,
        "source_d_v": 0.3,
        "source_energy_k": 2.0,
        "source_energy_v": 1.5,
        "fused_d_k": 0.1,
        "fused_d_v": 0.15,
        "fused_energy_k": 1.0,
        "fused_energy_v": 1.0,
        "candidate_logit_range": 0.2,
        "candidate_logit_variance": 0.01,
        "jensen_gap": 0.005,
        "parent_attention_mass": 0.25,
        "output_delta_l2": 0.01,
    }
    value.update(overrides)
    return value


def _descriptor():
    return {
        "task": "ai2-arc",
        "content_group_sha256": GROUP,
        "sample_sha256": [SAMPLE],
        "evaluation_subject": "SPLIT_0_OF_2",
        "evaluation_question_id": 7,
    }


def _shard(operator: str = "f"):
    return {
        "seed": 2026072201,
        "checkpoint_arm": "c_post_trained",
        "inference_operator": operator,
        "cell": "Y_CF" if operator == "f" else "Y_CC",
        "task": "ai2-arc",
        "lambda_value": 1.0 if operator == "f" else 0.0,
    }


def _provenance():
    return {
        "input_sha256": "c" * 64,
        "alignment_sha256": "d" * 64,
        "labels_sha256": "e" * 64,
        "gold_response_sha256": "f" * 64,
    }


def _gold(logp: float = -0.7):
    return {
        "batch_index": torch.tensor([0]),
        "query_position": torch.tensor([8]),
        "target_position": torch.tensor([9]),
        "target_token_id": torch.tensor([12]),
        "gold_logp": torch.tensor([logp]),
    }


def test_bounded_capture_contract_rejects_summary_only_and_raw_kv() -> None:
    with pytest.raises(RuntimeError, match="contract version"):
        capture_primitives({"stores_raw_kv": False, "layers": {}})
    with pytest.raises(RuntimeError, match="stores_raw_kv"):
        capture_primitives(
            {
                "stores_raw_kv": True,
                "long_form_contract_version": 1,
                "long_form_incomplete_chunk_count": 0,
                "long_form_row_count": 1,
                "max_long_form_rows": 8,
                "long_form_primitives": [_primitive()],
            }
        )


def test_teacher_forced_capture_passes_and_attests_frozen_item_ceiling() -> None:
    class FakeModel:
        def fpct_teacher_forced_query_mask(self, labels):
            mask = torch.zeros_like(labels, dtype=torch.bool)
            mask[:, :-1] = labels[:, 1:] != -100
            return mask

        def begin_fpct_capture(self, **kwargs):
            self.begin = kwargs

        def __call__(self, **kwargs):
            batch, length = kwargs["labels"].shape
            return SimpleNamespace(logits=torch.zeros(batch, length, 32), loss=None)

        def end_fpct_capture(self):
            ceiling = self.begin["max_long_form_rows"]
            return {
                "max_long_form_rows": ceiling,
                "long_form_row_count": ceiling,
                "stores_raw_kv": False,
            }

    model = FakeModel()
    labels = torch.tensor([[-100, 3, 4]])
    observation = teacher_forced_capture_bounded(
        model,
        {},
        labels,
        metadata={"sample": "tiny"},
        max_long_form_rows=17,
    )
    assert model.begin["max_long_form_rows"] == 17
    assert observation.capture_report["long_form_row_count"] == 17
    report = {
        "stores_raw_kv": False,
        "long_form_contract_version": 1,
        "long_form_incomplete_chunk_count": 0,
        "long_form_row_count": 1,
        "max_long_form_rows": 8,
        "long_form_primitives": [_primitive()],
    }
    assert capture_primitives(report) == [_primitive()]

    with pytest.raises(RuntimeError, match="differs"):
        capture_primitives({**report, "long_form_row_count": 2})
    with pytest.raises(RuntimeError, match="exceeds"):
        capture_primitives(
            {
                **report,
                "long_form_primitives": [_primitive(), _primitive()],
                "long_form_row_count": 2,
                "max_long_form_rows": 1,
            }
        )


def test_backend_recomputes_exact_task_row_volume_before_model_load() -> None:
    from script.experiment.fpct_e1_prepare_input_lock import expected_long_form_row_volume

    items = [
        {
            "expected_long_form_rows": value,
            "sample_sha256": f"{index:064x}",
            "content_group_sha256": f"{index + 10:064x}",
        }
        for index, value in enumerate([10, 20, 30, 40])
    ]
    observed = expected_long_form_row_volume(items)
    contract = {"expected_long_form_rows": observed}
    summary = {"count": observed["count"], "sum": observed["sum"]}
    assert verify_expected_long_form_row_volume(items, contract, summary) == observed
    with pytest.raises(ValueError, match="summary changed"):
        verify_expected_long_form_row_volume(
            items, contract, {**summary, "sum": summary["sum"] + 1}
        )


def test_topology_accepts_only_certified_prompt_m2() -> None:
    details = {
        "soft_alignment": {
            "source_indices": [[0, -1], [1, 2], [3, 4], [5, -1]],
            "source_weights": [[1.0, 0.0], [0.5, 0.5], [0.5, 0.5], [1.0, 0.0]],
            "fpct_certified_mask": [False, True, True, False],
        }
    }
    topology = topology_contract(details, instruction_end=2)
    assert topology[1]["topology"] == "partition_compositional"
    assert topology[1]["within_instruction"] is True
    assert 2 not in topology
    uncertified = {
        "soft_alignment": {
            "source_indices": [[1, 2]],
            "source_weights": [[0.5, 0.5]],
            "fpct_certified_mask": [False],
        }
    }
    with pytest.raises(ValueError, match="uncertified"):
        topology_contract(uncertified, instruction_end=1)


def test_assemble_rows_emits_current_outcome_and_executor_can_join_cpost() -> None:
    rows = assemble_sample_rows(
        primitives=[_primitive()],
        gold=_gold(-0.7),
        topology={
            4: {
                "candidate_count": 2,
                "prior": [0.5, 0.5],
                "topology": "partition_compositional",
                "within_instruction": True,
            }
        },
        descriptor=_descriptor(),
        shard=_shard("f"),
        provenance=_provenance(),
        end_task_correct=False,
    )
    assert len(rows) == 1
    assert not any(name.startswith("cpost_") for name in rows[0])
    row = validate_and_derive_row(
        {**rows[0], "cpost_gold_logp": -0.8, "cpost_end_task_correct": True}
    )
    assert row["delta_gold_logp"] == pytest.approx(0.1)
    assert row["end_task_accuracy_flip"] is True
    assert row["target_position"] == row["query_position"] + 1


def test_cpost_row_is_exact_self_baseline_and_bad_prior_fails() -> None:
    primitive = _primitive(
        gamma=[0.5, 0.5],
        candidate_logit_range=0.0,
        candidate_logit_variance=0.0,
        jensen_gap=0.0,
        output_delta_l2=0.0,
    )
    rows = assemble_sample_rows(
        primitives=[primitive],
        gold=_gold(-0.75),
        topology={
            4: {
                "candidate_count": 2,
                "prior": [0.5, 0.5],
                "topology": "partition_compositional",
                "within_instruction": True,
            }
        },
        descriptor=_descriptor(),
        shard=_shard("c_post"),
        provenance=_provenance(),
        end_task_correct=True,
    )
    assert validate_and_derive_row(
        {
            **rows[0],
            "cpost_gold_logp": rows[0]["gold_logp"],
            "cpost_end_task_correct": rows[0]["end_task_correct"],
        }
    )["delta_gold_logp"] == 0.0

    with pytest.raises(ValueError, match="frozen sanitized alignment"):
        assemble_sample_rows(
            primitives=[{**primitive, "prior": [0.6, 0.4]}],
            gold=_gold(),
            topology={
                4: {
                    "candidate_count": 2,
                    "prior": [0.5, 0.5],
                    "topology": "partition_compositional",
                    "within_instruction": True,
                }
            },
            descriptor=_descriptor(),
            shard=_shard("c_post"),
            provenance=_provenance(),
            end_task_correct=True,
        )


def test_explicit_f_lambda_zero_is_current_operator_exact_control() -> None:
    shard = _shard("f")
    shard["lambda_value"] = 0.0
    primitive = _primitive(
        gamma=[0.5, 0.5],
        candidate_logit_range=0.0,
        candidate_logit_variance=0.0,
        jensen_gap=0.0,
        output_delta_l2=0.0,
    )
    current = assemble_sample_rows(
        primitives=[primitive],
        gold=_gold(-0.75),
        topology={
            4: {
                "candidate_count": 2,
                "prior": [0.5, 0.5],
                "topology": "partition_compositional",
                "within_instruction": True,
            }
        },
        descriptor=_descriptor(),
        shard=shard,
        provenance=_provenance(),
        end_task_correct=False,
    )[0]
    final = {
        **current,
        "cpost_gold_logp": current["gold_logp"],
        "cpost_end_task_correct": False,
    }
    assert final["inference_operator"] == "f"
    assert validate_and_derive_row(final)["delta_gold_logp"] == 0.0


def test_hash_projection_and_gold_template_are_stable() -> None:
    assert GOLD_RESPONSE_TEMPLATE.format(answer="C") == "The correct answer is C."
    assert canonical_content_sha256("  Why  sky? ", [" blue ", "red"]) == (
        "0f46d747b3d640fb947616d9db64c46b48e625f6d2bae9da6c0ab40eaae8e24d"
    )
    assert canonical_sample_sha256("ai2-arc", "SPLIT_0_OF_1", "679") == (
        "485c6e7d8545f7329ca51e67a867ee03c3f961c0bf1fbc2e3fa41e2730645ce0"
    )
