from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from script.analysis.fpct_e1_mechanism_audit import validate_and_derive_row
from script.experiment.fpct_e1_runtime_backend import (
    GOLD_RESPONSE_TEMPLATE,
    PrimitiveChunkSpool,
    assemble_sample_rows,
    canonical_content_sha256,
    canonical_sample_sha256,
    capture_primitives,
    teacher_forced_capture_bounded,
    topology_contract,
    verify_expected_long_form_row_volume,
    _canonical_runtime_items,
    _RecomputedPrefixVerifier,
    _validate_capture_resume_cursor,
)
from script.experiment.fpct_e1_capture_runner import (
    CAPTURE_STAGING_PROTOCOL_ID,
    _audit_columns,
    _backend_projection_bytes,
    canonical_json_bytes,
    sha256_bytes,
)


GROUP = "a" * 64
SAMPLE = "b" * 64


def test_shuffled_sidecar_items_canonicalize_before_resume_cursor() -> None:
    items = [
        {"sample_sha256": "b" * 64, "expected_long_form_rows": 4096},
        {"sample_sha256": "a" * 64, "expected_long_form_rows": 4096},
    ]
    ordered = _canonical_runtime_items(items)
    assert [item["sample_sha256"] for item in ordered] == ["a" * 64, "b" * 64]
    payload = {
        "schema_version": 1,
        "protocol_id": CAPTURE_STAGING_PROTOCOL_ID,
        "plan_sha256": "c" * 64,
        "shard_id": "shard",
        "completed_logical_rows": 0,
        "completed_sample_count": 0,
        "completed_chunk_count": 0,
        "next_sample_sha256": "a" * 64,
        "next_row_ordinal": 0,
        "prefix_semantic_stream_sha256": "d" * 64,
        "completion_receipt_chain_sha256": "e" * 64,
        "partial_sample_backend_prefix_row_count": 0,
        "partial_sample_backend_prefix_sha256": hashlib.sha256().hexdigest(),
        "backend_projection_columns": list(_audit_columns()[0]),
        "scientific_prefix_reverified": True,
        "resume_only_from_first_incomplete_range": True,
    }
    cursor = {
        **payload,
        "cursor_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }
    request = {
        "plan_sha256": payload["plan_sha256"],
        "shard": {"shard_id": "shard"},
        "resume_cursor": cursor,
    }
    assert _validate_capture_resume_cursor(request, ordered) == cursor
    with pytest.raises(ValueError, match="unique sample-SHA order"):
        _validate_capture_resume_cursor(request, list(reversed(ordered)))


def test_runtime_recomputed_prefix_digest_go_and_changed_metric_fails() -> None:
    columns = _audit_columns()[0]
    row = {name: 0 for name in columns}
    row["candidate_indices"] = [1, 2, -1, -1]
    row["candidate_valid_mask"] = [True, True, False, False]
    row["candidate_slot_weights"] = [0.5, 0.5, 0.0, 0.0]
    row["prior"] = [0.5, 0.5]
    row["gamma"] = [0.5, 0.5]
    expected = hashlib.sha256(_backend_projection_bytes(row, columns)).hexdigest()
    cursor = {
        "partial_sample_backend_prefix_row_count": 1,
        "partial_sample_backend_prefix_sha256": expected,
    }
    channel = {"status": "PENDING_RECOMPUTATION"}
    verifier = _RecomputedPrefixVerifier(cursor, channel)
    verifier.observe(row)
    verifier.finalize()
    assert channel == {
        "status": "GO_EXACT_PREFIX_MATCH",
        "observed_row_count": 1,
        "observed_backend_projection_sha256": expected,
        "exact_match": True,
    }

    changed = dict(row)
    changed["gold_logp"] = 1.0
    failed_channel = {"status": "PENDING_RECOMPUTATION"}
    failed = _RecomputedPrefixVerifier(cursor, failed_channel)
    failed.observe(changed)
    with pytest.raises(RuntimeError, match="differs from immutable prefix"):
        failed.finalize()
    assert failed_channel["status"] == "FAILED_PREFIX_MISMATCH"
    assert failed_channel["exact_match"] is False


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
        "runtime_source_indices": [11, 12, -1, -1],
        "candidate_valid_mask": [True, True, False, False],
        "candidate_slot_weights": [0.5, 0.5, 0.0, 0.0],
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


def _topology_record() -> dict:
    return {
        "candidate_count": 2,
        "prior": [0.5, 0.5],
        "candidate_indices": [11, 12, -1, -1],
        "candidate_valid_mask": [True, True, False, False],
        "candidate_slot_weights": [0.5, 0.5, 0.0, 0.0],
        "statistical_weight": 1.0,
        "topology": "partition_compositional",
        "within_instruction": True,
    }


def _spool_chunk() -> dict:
    index = torch.tensor([[0, 0, 8, 4], [0, 1, 8, 4]], dtype=torch.long)
    valid = torch.tensor(
        [[True, True, False, False], [True, True, False, False]]
    )
    scalar = torch.tensor([0.1, 0.2])
    return {
        "layer": 0,
        "index": index,
        "gamma": torch.tensor(
            [[0.7, 0.3, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0]]
        ),
        "prior": torch.tensor(
            [[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]
        ),
        "valid": valid,
        "source_indices": torch.tensor(
            [[11, 12, -1, -1], [11, 12, -1, -1]], dtype=torch.long
        ),
        "source_indices_certified": True,
        "parent_metrics": {
            name: scalar.clone()
            for name in (
                "candidate_logit_range",
                "candidate_logit_variance",
                "jensen_gap",
                "parent_attention_mass",
            )
        },
        "parent_geometry": {
            name: scalar.clone()
            for name in (
                "source_d_k",
                "source_d_v",
                "source_energy_k",
                "source_energy_v",
                "fused_d_k",
                "fused_d_v",
                "fused_energy_k",
                "fused_energy_v",
            )
        },
        "output_delta_l2": scalar.clone(),
        "num_key_value_heads": 1,
    }


def test_primitive_spool_is_bounded_ordered_and_replayable(tmp_path: Path) -> None:
    spool = PrimitiveChunkSpool(
        expected_rows=2,
        num_query_heads=2,
        num_key_value_heads=1,
        root=tmp_path,
    )
    spool.write_primitive_chunk(_spool_chunk())
    assert spool.finalize() == {
        "complete": True,
        "row_count": 2,
        "chunk_count": 1,
        "max_chunk_rows": 2,
        "physical_chunk_rows": 4096,
        "stores_raw_kv": False,
    }
    rows = list(spool.iter_primitives())
    assert [(row["query_head"], row["kv_head"]) for row in rows] == [(0, 0), (1, 0)]
    assert all(row["candidate_count"] == 2 for row in rows)
    spool.cleanup()


def test_primitive_spool_rejects_over_emission_before_publish(tmp_path: Path) -> None:
    spool = PrimitiveChunkSpool(
        expected_rows=1,
        num_query_heads=2,
        num_key_value_heads=1,
        root=tmp_path,
    )
    with pytest.raises(RuntimeError, match="more than expected"):
        spool.write_primitive_chunk(_spool_chunk())
    assert spool.row_count == 0
    assert not tuple(spool.root.glob("*.pt"))
    spool.abort()


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


def test_teacher_forced_capture_passes_and_attests_exact_stream_count() -> None:
    class FakeSink:
        def finalize(self):
            return {
                "complete": True,
                "row_count": 17,
                "chunk_count": 1,
                "max_chunk_rows": 17,
            }

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
            expected = self.begin["expected_long_form_rows"]
            receipt = self.begin["primitive_sink"].finalize()
            return {
                "long_form_contract_version": 2,
                "expected_long_form_rows": expected,
                "long_form_row_count": expected,
                "long_form_stream": receipt,
                "stores_raw_kv": False,
            }

    model = FakeModel()
    labels = torch.tensor([[-100, 3, 4]])
    observation = teacher_forced_capture_bounded(
        model,
        {},
        labels,
        metadata={"sample": "tiny"},
        expected_long_form_rows=17,
        primitive_sink=FakeSink(),
    )
    assert model.begin["expected_long_form_rows"] == 17
    assert model.begin["detail_mode"] == "aggregate_only"
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
            "source_indices": [[0, -1, -1, -1], [1, 2, -1, -1], [3, 4, -1, -1], [5, -1, -1, -1]],
            "source_weights": [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            "fpct_certified_mask": [False, True, True, False],
        }
    }
    topology = topology_contract(details, instruction_end=2)
    assert topology[1]["topology"] == "partition_compositional"
    assert topology[1]["within_instruction"] is True
    assert 2 not in topology
    uncertified = {
        "soft_alignment": {
            "source_indices": [[1, 2, -1, -1]],
            "source_weights": [[0.5, 0.5, 0.0, 0.0]],
            "fpct_certified_mask": [False],
        }
    }
    with pytest.raises(ValueError, match="uncertified"):
        topology_contract(uncertified, instruction_end=1)


def test_assemble_rows_emits_current_outcome_and_executor_can_join_cpost() -> None:
    rows = assemble_sample_rows(
        primitives=[_primitive()],
        gold=_gold(-0.7),
        topology={4: _topology_record()},
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
        topology={4: _topology_record()},
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
            topology={4: _topology_record()},
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
        topology={4: _topology_record()},
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


def test_runtime_candidate_slot_identity_cannot_be_reconstructed_from_uniform_prior() -> None:
    with pytest.raises(ValueError, match="slot/mask identity"):
        assemble_sample_rows(
            primitives=[
                _primitive(runtime_source_indices=[21, 22, -1, -1])
            ],
            gold=_gold(),
            topology={4: _topology_record()},
            descriptor=_descriptor(),
            shard=_shard("f"),
            provenance=_provenance(),
            end_task_correct=True,
        )
    bad_chunk = _spool_chunk()
    bad_chunk["source_indices_certified"] = False
    spool = PrimitiveChunkSpool(
        expected_rows=2, num_query_heads=2, num_key_value_heads=1
    )
    with pytest.raises(ValueError, match="certified four-slot"):
        spool.write_primitive_chunk(bad_chunk)
    spool.abort()


def test_hash_projection_and_gold_template_are_stable() -> None:
    assert GOLD_RESPONSE_TEMPLATE.format(answer="C") == "The correct answer is C."
    assert canonical_content_sha256("  Why  sky? ", [" blue ", "red"]) == (
        "0f46d747b3d640fb947616d9db64c46b48e625f6d2bae9da6c0ab40eaae8e24d"
    )
    assert canonical_sample_sha256("ai2-arc", "SPLIT_0_OF_1", "679") == (
        "485c6e7d8545f7329ca51e67a867ee03c3f961c0bf1fbc2e3fa41e2730645ce0"
    )
