from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import script.experiment.fpct_e1_prepare_input_lock as prepare
import script.analysis.fpct_e1_a5_prompt_gate as a5_prompt_gate

from script.experiment.fpct_e1_prepare_input_lock import (
    HISTORICAL_MAX_LONG_FORM_ROWS_PER_SAMPLE,
    _generic_asset_tree,
    _publish_blocked_receipt,
    _verified_geometry_rows,
    _write_geometry_lock,
    _write_streaming_template_lock,
    answer_query_contract,
    certified_parent_contract,
    expected_long_form_row_volume,
    expected_long_form_rows,
    row_template_attestation,
    row_template_records,
    streaming_row_template_attestation,
    raw_topology_ledger,
    expanded_row_absence_proof,
    publish_bytes_no_overwrite,
    task_membership_sha256,
    validate_a5_execution_identity,
)
from script.experiment.fpct_e1_capture_runner import (
    ROW_TEMPLATE_COLUMNS,
    _KeyLedger,
    _key_bytes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _direct_prepare_command(tmp_path: Path) -> list[str]:
    execution_sha = "a" * 40
    run_root = tmp_path / f"fpct-e1-a5-{execution_sha[:8]}-v1"
    return [
        sys.executable,
        str(REPO_ROOT / "script/experiment/fpct_e1_prepare_input_lock.py"),
        "--repo-root",
        str(REPO_ROOT),
        "--e0-data-root",
        str(tmp_path / "must-not-read-data"),
        "--output-sidecar",
        str(run_root / "input_lock/sidecar.pt"),
        "--output-manifest",
        str(run_root / "input_lock/manifest.json"),
        "--execution-sha",
        execution_sha,
        "--source-snapshot-root",
        str(REPO_ROOT),
        "--source-snapshot-receipt",
        str(REPO_ROOT / prepare.SOURCE_SNAPSHOT_RECEIPT_NAME),
        "--run-uid",
        f"fpct-e1-a5-runtime-prompt-{execution_sha[:8]}-v1",
        "--run-root",
        str(run_root),
    ]


def _item() -> dict:
    return {
        "sample_sha256": "1" * 64,
        "content_group_sha256": "2" * 64,
        "provenance": {
            "input_sha256": "3" * 64,
            "alignment_sha256": "4" * 64,
            "labels_sha256": "5" * 64,
            "gold_response_sha256": "6" * 64,
        },
        "answer_queries": [
            {"query_position": 7, "target_position": 8, "target_token_id": 41},
            {"query_position": 8, "target_position": 9, "target_token_id": 42},
        ],
        "certified_parents": [
            {
                "parent_position": 2,
                "candidate_count": 2,
                "prior": [0.5, 0.5],
                "candidate_indices": [10, 11, -1, -1],
                "candidate_valid_mask": [True, True, False, False],
                "candidate_slot_weights": [0.5, 0.5, 0.0, 0.0],
                "statistical_weight": 1.0,
                "topology": "partition_compositional",
            },
            {
                "parent_position": 4,
                "candidate_count": 3,
                "prior": [1 / 3, 1 / 3, 1 / 3],
                "candidate_indices": [20, 21, 22, -1],
                "candidate_valid_mask": [True, True, True, False],
                "candidate_slot_weights": [1 / 3, 1 / 3, 1 / 3, 0.0],
                "statistical_weight": 1.0,
                "topology": "partition_compositional",
            },
        ],
    }


def test_answer_queries_apply_exact_causal_shift() -> None:
    rows = answer_query_contract([-100, -100, 11, 12])
    assert rows == [
        {"query_position": 1, "target_position": 2, "target_token_id": 11},
        {"query_position": 2, "target_position": 3, "target_token_id": 12},
    ]
    with pytest.raises(ValueError, match="no eligible"):
        answer_query_contract([-100, -100])


def test_certified_parent_contract_excludes_m1_and_response() -> None:
    topology = {
        0: {"candidate_count": 1, "prior": [1.0], "topology": "not_applicable", "within_instruction": True},
        2: {"candidate_count": 2, "prior": [0.5, 0.5], "topology": "partition_compositional", "within_instruction": True},
    }
    assert [row["parent_position"] for row in certified_parent_contract(topology)] == [2]
    topology[3] = {"candidate_count": 2, "prior": [0.5, 0.5], "topology": "partition_compositional", "within_instruction": False}
    with pytest.raises(ValueError, match="include_response=false"):
        certified_parent_contract(topology)


def test_certified_parent_contract_freezes_four_sanitized_slots() -> None:
    topology = {
        0: {
            "candidate_count": 2,
            "prior": [0.75, 0.25],
            "topology": "partition_compositional",
            "within_instruction": True,
        }
    }
    rows = certified_parent_contract(
        topology,
        source_indices=[[17, 19, -1, -1]],
        source_weights=[[0.75, 0.25, 0.0, 0.0]],
    )
    assert rows == [
        {
            "parent_position": 0,
            "candidate_count": 2,
            "prior": [0.75, 0.25],
            "topology": "partition_compositional",
            "statistical_weight": 1.0,
            "candidate_indices": [17, 19, -1, -1],
            "candidate_valid_mask": [True, True, False, False],
            "candidate_slot_weights": [0.75, 0.25, 0.0, 0.0],
        }
    ]
    with pytest.raises(ValueError, match="four slots"):
        certified_parent_contract(
            topology,
            source_indices=[[17, 19]],
            source_weights=[[0.75, 0.25]],
        )


def test_row_template_count_is_parents_times_queries_times_layers_heads() -> None:
    item = _item()
    rows = row_template_records(
        item, num_layers=3, num_query_heads=4, num_kv_heads=2
    )
    assert len(rows) == 2 * 2 * 3 * 4
    assert {row[7] for row in rows} == {0, 1, 2, 3}
    assert {row[8] for row in rows} == {0, 1}
    first = row_template_attestation(
        [item], num_layers=3, num_query_heads=4, num_kv_heads=2
    )
    second = row_template_attestation(
        [item], num_layers=3, num_query_heads=4, num_kv_heads=2
    )
    assert first == second
    assert first["count"] == 48
    assert len(first["sha256"]) == 64


def test_expected_long_form_rows_freezes_exact_product_without_historical_cap() -> None:
    assert expected_long_form_rows(
        answer_query_count=2,
        certified_parent_count=3,
        num_layers=28,
        num_query_heads=16,
    ) == 2 * 3 * 28 * 16
    assert expected_long_form_rows(
        answer_query_count=HISTORICAL_MAX_LONG_FORM_ROWS_PER_SAMPLE,
        certified_parent_count=2,
        num_layers=1,
        num_query_heads=1,
    ) == 524288


def test_task_row_volume_uses_exact_nearest_rank_and_deterministic_argmax() -> None:
    values = [100, 20, 50, 10, 80]
    items = [
        {
            "expected_long_form_rows": value,
            "sample_sha256": f"{index:064x}",
            "content_group_sha256": f"{index + 100:064x}",
        }
        for index, value in enumerate(values)
    ]
    assert expected_long_form_row_volume(items) == {
        "formula": "num_layers*num_query_heads*answer_query_count*certified_parent_count",
        "historical_cumulative_sample_ceiling": HISTORICAL_MAX_LONG_FORM_ROWS_PER_SAMPLE,
        "historical_ceiling_operative": False,
        "physical_chunk_rows": 4096,
        "chunk_count": 5,
        "count": 5,
        "sum": 260,
        "min": 10,
        "p50": 50,
        "p95": 100,
        "max": 100,
        "argmax": {
            "sample_sha256": f"{0:064x}",
            "content_group_sha256": f"{100:064x}",
        },
        "quantile_method": "nearest_rank",
        "quantile_definition": "sorted[ceil(p*n)-1]",
    }


def test_streaming_template_attestation_is_ordinal_and_nonmaterialized() -> None:
    item = {**_item(), "task": "ai2-arc"}
    value = streaming_row_template_attestation(
        [item],
        task="ai2-arc",
        num_layers=3,
        num_query_heads=4,
        num_kv_heads=2,
        schema_sha256="a" * 64,
    )
    assert value["count"] == 48
    assert value["sample_count"] == 1
    assert value["physical_chunk_rows"] == 4096
    assert value["ordinal_order"] == "sample_sha256,row_ordinal"
    assert len(value["semantic_stream_sha256"]) == 64


def test_template_digest_is_byte_identical_to_executor_ledger(tmp_path: Path) -> None:
    item = _item()
    expected = row_template_attestation(
        [item], num_layers=3, num_query_heads=4, num_kv_heads=2
    )
    ledger = _KeyLedger(tmp_path / "executor.sqlite")
    try:
        for values in row_template_records(
            item, num_layers=3, num_query_heads=4, num_kv_heads=2
        ):
            row = dict(zip(ROW_TEMPLATE_COLUMNS, values))
            ledger.add(_key_bytes(row, ROW_TEMPLATE_COLUMNS))
        observed = ledger.attest()
    finally:
        ledger.close()
    assert {"count": observed[0], "sha256": observed[1]} == expected


def test_membership_hash_binds_plan_groups_to_semantic_items() -> None:
    planned = [
        {"content_group_sha256": "2" * 64, "sample_sha256": ["1" * 64]}
    ]
    members = [
        {
            "content_group_sha256": "2" * 64,
            "sample_sha256": "1" * 64,
            "item_semantic_sha256": "3" * 64,
        }
    ]
    first = task_membership_sha256(planned, members)
    assert len(first) == 64
    changed = [{**members[0], "item_semantic_sha256": "4" * 64}]
    assert task_membership_sha256(planned, changed) != first
    with pytest.raises(ValueError, match="sample identities"):
        task_membership_sha256(
            planned, [{**members[0], "sample_sha256": "5" * 64}]
        )


def test_raw_topology_taxonomy_is_conservative_and_output_free() -> None:
    raw = {
        "message_mask": [True, True],
        "slm_ids": [101, 102],
        "llm_ids": [201, 202, 203, 203],
        "slm_offsets": [(0, 2), (2, 4)],
        "llm_offsets": [(0, 1), (1, 2), (2, 3), (2, 3)],
        "content_spans_slm": [(0, 4)],
        "content_spans_llm": [(0, 4)],
        "sections": [
            {
                "type": "message",
                "slm_range": (0, 2),
                "llm_range": (0, 4),
            }
        ],
        "soft_alignment": {
            "source_indices": [[0, 1], [2, 3]],
            "source_weights": [[0.5, 0.5], [0.5, 0.5]],
        },
    }
    sanitized = {
        "soft_alignment": {
            "source_indices": [[0, 1], [2, -1]],
            "source_weights": [[0.5, 0.5], [1.0, 0.0]],
            "fpct_certified_mask": [True, False],
            "fpct_certification_reason": [
                "certified_disjoint_partition",
                "exact_duplicate_source_offsets",
            ],
        }
    }
    rows = raw_topology_ledger(
        raw_details=raw,
        sanitized_details=sanitized,
        instruction_end=2,
        task="ai2-arc",
        sample_sha256="1" * 64,
        content_group_sha256="2" * 64,
        candidate_window=0,
    )
    assert [row["taxonomy"] for row in rows] == [
        "partition_compositional",
        "boundary_fallback",
    ]
    first = rows[0]
    assert first["receiver_token_id"] == 101
    assert first["receiver_span"] == [0, 2]
    assert first["raw_candidate_indices"] == [0, 1]
    assert first["runtime_candidate_indices"] == [0, 1]
    assert [candidate["source_token_id"] for candidate in first["candidates"]] == [
        201,
        202,
    ]
    assert [candidate["intersection"] for candidate in first["candidates"]] == [
        [0, 1],
        [1, 2],
    ]
    assert all(candidate["origin"] == "span_overlap" for candidate in first["candidates"])
    assert rows[1]["duplicate_or_overlap_alias"] is True
    assert rows[1]["runtime_functional_eligible"] is False
    assert rows[1]["functional_metrics_present"] is False
    assert all("model" not in key and "logit" not in key for row in rows for key in row)
    assert all(len(row["span_geometry_sha256"]) == 64 for row in rows)
    assert not any(row["taxonomy"] in {"competing_overlap", "neighbor_expansion"} for row in rows)
    with pytest.raises(ValueError, match="nonnegative"):
        raw_topology_ledger(
            raw_details=raw,
            sanitized_details=sanitized,
            instruction_end=2,
            task="ai2-arc",
            sample_sha256="1" * 64,
            content_group_sha256="2" * 64,
            candidate_window=-1,
        )


def test_prepare_source_has_no_automodel_or_checkpoint_load() -> None:
    path = REPO_ROOT / "script/experiment/fpct_e1_prepare_input_lock.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    assert "AutoModel" not in source
    assert "from_pretrained" in source  # tokenizer only
    assert "load_rosetta_model" not in source
    assert "checkpoint_path" not in source
    assert "cuda" not in names


def _geometry_item(task: str, ordinal: int) -> dict:
    sample = f"{ordinal + 1:064x}"
    content = f"{ordinal + 1000:064x}"
    parent = {
        "parent_position": 2,
        "candidate_count": 2,
        "prior": [0.5, 0.5],
        "candidate_indices": [3, 4, -1, -1],
        "candidate_valid_mask": [True, True, False, False],
        "candidate_slot_weights": [0.5, 0.5, 0.0, 0.0],
        "statistical_weight": 1.0,
        "topology": "partition_compositional",
    }
    raw = {
        "row_count": 1,
        "rows": [
            {
                "parent_position": 2,
                "raw_candidate_count": 2,
                "runtime_candidate_count": 2,
                "raw_candidate_indices": [3, 4],
                "runtime_candidate_indices": [3, 4],
                "raw_weights": [0.5, 0.5],
                "runtime_weights": [0.5, 0.5],
                "certified": True,
                "certification_reason": "certified_disjoint_partition",
                "taxonomy": "partition_compositional",
                "span_geometry_sha256": "8" * 64,
            }
        ],
        "compact_rows_sha256": "7" * 64,
        "full_ledger_sha256": "6" * 64,
        "contains_model_output": False,
    }
    raw["compact_rows_sha256"] = prepare.nested_sha256(raw["rows"])
    query = {"query_position": 1, "target_position": 2, "target_token_id": 9}
    item = {
        "task": task,
        "sample_sha256": sample,
        "content_group_sha256": content,
        "provenance": {
            "input_sha256": "3" * 64,
            "alignment_sha256": "4" * 64,
            "labels_sha256": "5" * 64,
            "gold_response_sha256": "6" * 64,
        },
        "answer_queries": [query],
        "certified_parents": [parent],
        "Q_s": 1,
        "P_s": 1,
        "N_s": 1,
        "answer_query_sequence_sha256": prepare.nested_sha256([query]),
        "parent_sequence_sha256": prepare.nested_sha256([parent]),
        "raw_topology_compact": raw,
        "raw_topology_compact_sha256": prepare.nested_sha256(raw),
        "expected_chunk_count": 1,
        "expected_long_form_rows": 1,
    }
    item["item_semantic_sha256"] = prepare.nested_sha256(item)
    return item


def test_immutable_publication_verifies_winner_bytes(tmp_path: Path) -> None:
    target = tmp_path / "lock.json"
    assert publish_bytes_no_overwrite(target, b"first\n") == prepare.hashlib.sha256(
        b"first\n"
    ).hexdigest()
    publish_bytes_no_overwrite(target, b"first\n")
    with pytest.raises(RuntimeError, match="winner bytes differ"):
        publish_bytes_no_overwrite(target, b"second\n")
    assert target.read_bytes() == b"first\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_asset_tree_digest_is_a_real_before_after_hash(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    root.mkdir()
    asset = root / "data.json"
    asset.write_text('{"v":1}\n', encoding="utf-8")
    before = _generic_asset_tree(root)
    asset.write_text('{"v":2}\n', encoding="utf-8")
    after = _generic_asset_tree(root)
    assert before["tree_sha256"] != after["tree_sha256"]
    assert before["files"][0]["sha256"] != after["files"][0]["sha256"]


def test_execution_identity_requires_new_canonical_empty_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution_sha = "a" * 40
    run_root = tmp_path / f"fpct-e1-a5-{execution_sha[:8]}-v1"
    snapshot = run_root / "source_snapshot"
    snapshot.mkdir(parents=True)
    receipt = snapshot / prepare.SOURCE_SNAPSHOT_RECEIPT_NAME
    receipt.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "script.experiment.fpct_e1_source_snapshot_lock.verify_source_snapshot_receipt",
        lambda receipt_path, root, sha: {
            "status": "GO_MOUNTED_SOURCE_SNAPSHOT",
            "execution_sha": sha,
            "mounted_tree_canonical_sha256": "b" * 64,
        },
    )
    output = run_root / "input_lock"
    identity = validate_a5_execution_identity(
        execution_sha=execution_sha,
        source_snapshot_root=snapshot,
        source_snapshot_receipt=receipt,
        run_uid=f"fpct-e1-a5-runtime-prompt-{execution_sha[:8]}-v1",
        run_root=run_root,
        output_root=output,
        sealed_prepare_execution={
            "pytest_verified_test_sentinel": True,
            "repo_root": str(snapshot.absolute()),
            "execution_sha": execution_sha,
            "production_eligible": False,
        },
        _test_only_run_parent=tmp_path,
    )
    assert identity["historical_artifact_reuse_allowed"] is False
    assert (output / prepare.RUN_IDENTITY_NAME).is_file()
    validate_a5_execution_identity(
        execution_sha=execution_sha,
        source_snapshot_root=snapshot,
        source_snapshot_receipt=receipt,
        run_uid=f"fpct-e1-a5-runtime-prompt-{execution_sha[:8]}-v1",
        run_root=run_root,
        output_root=output,
        sealed_prepare_execution={
            "pytest_verified_test_sentinel": True,
            "repo_root": str(snapshot.absolute()),
            "execution_sha": execution_sha,
            "production_eligible": False,
        },
        _test_only_run_parent=tmp_path,
    )
    with pytest.raises(ValueError, match="historical abandoned"):
        validate_a5_execution_identity(
            execution_sha="07755a40" + "0" * 32,
            source_snapshot_root=snapshot,
            source_snapshot_receipt=receipt,
            run_uid="fpct-e1-a5-runtime-prompt-07755a40-v1",
            run_root=run_root,
            output_root=output,
            sealed_prepare_execution={
                "pytest_verified_test_sentinel": True,
                "repo_root": str(snapshot.absolute()),
                "execution_sha": "07755a40" + "0" * 32,
                "production_eligible": False,
            },
            _test_only_run_parent=tmp_path,
        )


def test_compact_geometry_is_self_contained_and_pass_b_reads_only_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = {"ai2-arc": 1, "openbookqa": 1, "mmlu-redux": 1}
    monkeypatch.setattr(prepare, "TASK_GROUP_COUNTS", counts)
    # This test deliberately shrinks the frozen 326-row population; exact v6
    # schema validation is exercised separately on production-shaped receipts.
    monkeypatch.setattr(prepare, "validate_streaming_schema_artifact", lambda value: None)
    items = [_geometry_item(task, index) for index, task in enumerate(counts)]
    gate = {
        "estimated_physical_bytes_per_row": 4096,
        "checks": {
            name: True
            for name in (
                "row_key_reference_equivalence",
                "weights_reference_equivalence",
                "topology_reference_equivalence",
                "chunk_partition_semantic_equivalence",
                "aggregate_partition_equivalence",
                "bounded_peak_rss",
                "atomic_no_overwrite",
                "crash_resume_equivalence",
            )
        },
    }
    asset_state = {"aggregate_sha256": "d" * 64}
    identity = {"identity_sha256": "e" * 64}
    geometry = _write_geometry_lock(
        output_dir=tmp_path,
        items=items,
        schema_sha256="a" * 64,
        synthetic_gate=gate,
        input_assets_before=asset_state,
        input_assets_after=asset_state,
        execution_identity=identity,
    )
    rows = list(
        _verified_geometry_rows(
            samples_path=Path(geometry["samples"]["path"]),
            manifest_path=Path(geometry["manifest"]["path"]),
            schema_sha256="a" * 64,
        )
    )
    assert {row["item_semantic_sha256"] for row in rows} == {
        item["item_semantic_sha256"] for item in items
    }
    assert all(row["raw_topology_compact"]["row_count"] == 1 for row in rows)

    # A process death after Parquet publication but before JSON receipts is a
    # resumable crash, not an integrity failure or a reason to regenerate data.
    Path(geometry["receipt"]["path"]).unlink()
    Path(geometry["manifest"]["path"]).unlink()
    resumed = _write_geometry_lock(
        output_dir=tmp_path,
        items=items,
        schema_sha256="a" * 64,
        synthetic_gate=gate,
        input_assets_before=asset_state,
        input_assets_after=asset_state,
        execution_identity=identity,
    )
    assert resumed["samples"]["sha256"] == geometry["samples"]["sha256"]

    # Poison the original in-memory geometry after Pass A.  Pass B has no
    # items argument and therefore must reproduce the immutable Parquet rows.
    for item in items:
        item["certified_parents"][0]["parent_position"] = 999
    gate_path = tmp_path / "synthetic_gate.json"
    gate_path.write_text("{}\n", encoding="utf-8")
    streamed = _write_streaming_template_lock(
        output_dir=tmp_path,
        dimensions={
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
        },
        schema_sha256="a" * 64,
        geometry_lock=geometry,
        execution_identity=identity,
        input_assets_before_sha256=asset_state["aggregate_sha256"],
        input_assets_after_sha256=asset_state["aggregate_sha256"],
        synthetic_gate_path=gate_path,
        synthetic_gate=gate,
    )
    index = json.loads(Path(streamed["index"]["path"]).read_text())
    assert index["pass_b_geometry_source"]["sole_geometry_input"] is True
    assert index["expected_logical_rows"] == 3
    assert all(record["expected_logical_rows"] == 1 for record in index["records"])


def _completed_input_lock_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    """Build a small but physically real completed A5 lock for resume tests."""

    counts = {"ai2-arc": 1, "openbookqa": 1, "mmlu-redux": 1}
    monkeypatch.setattr(prepare, "TASK_GROUP_COUNTS", counts)
    monkeypatch.setattr(
        prepare, "validate_streaming_schema_artifact", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        a5_prompt_gate, "validate_a5_schema_artifact", lambda *args, **kwargs: None
    )
    repo_root = tmp_path / "source_snapshot"
    schema_path = repo_root / prepare.A4_STREAMING_SCHEMA_RELATIVE
    gate_path = repo_root / prepare.A5_SYNTHETIC_GATE_RELATIVE
    schema_path.parent.mkdir(parents=True)
    schema_path.write_text('{"synthetic":"schema"}\n', encoding="utf-8")
    gate_path.write_text('{"synthetic":"gate"}\n', encoding="utf-8")
    schema_sha256 = prepare.sha256_file(schema_path)
    gate = {
        "protocol_id": prepare.A5_PROTOCOL_ID,
        "status": "GO_PRE_NATURAL_SYNTHETIC_HARD_GATE",
        "natural_e0_design_accessed": False,
        "estimated_physical_bytes_per_row": 4096,
        "checks": {
            name: True
            for name in (
                "row_key_reference_equivalence",
                "weights_reference_equivalence",
                "topology_reference_equivalence",
                "chunk_partition_semantic_equivalence",
                "aggregate_partition_equivalence",
                "bounded_peak_rss",
                "atomic_no_overwrite",
                "crash_resume_equivalence",
            )
        },
        "streaming_contract_checks": {
            name: True
            for name in (
                "row_key_reference_equivalence",
                "weights_reference_equivalence",
                "topology_reference_equivalence",
                "chunk_partition_semantic_equivalence",
                "aggregate_partition_equivalence",
                "bounded_peak_rss",
                "atomic_no_overwrite",
                "crash_resume_equivalence",
            )
        },
    }
    monkeypatch.setattr(a5_prompt_gate, "verify_a5_gate", lambda *args, **kwargs: gate)
    monkeypatch.setattr(
        prepare, "load_e0_design_lock", lambda path: {"sha256": "1" * 64}
    )
    monkeypatch.setattr(
        prepare,
        "verify_e0_dev_anchor",
        lambda path, split: {"sha256": "2" * 64},
    )
    renderer_source_identity = {
        "renderer_source_identity_attested": True,
        "production_renderer_exactly_attested": False,
        "production_renderer_exact_attestation_pending": "fixture",
        "source_closure_sha256": {"fixture.py": "f" * 64},
    }
    prompt_config_identity = {"records_sha256": "e" * 64}
    monkeypatch.setattr(
        prepare, "attest_e0_renderer_identity", lambda root: renderer_source_identity
    )
    monkeypatch.setattr(prepare, "_load_a5_prompt_contract", lambda root: {})
    monkeypatch.setattr(
        prepare,
        "_verify_all_e0_prompt_configs",
        lambda root, contract: prompt_config_identity,
    )

    output_root = tmp_path / "input_lock"
    output_root.mkdir()
    source_receipt_path = repo_root / prepare.SOURCE_SNAPSHOT_RECEIPT_NAME
    source_receipt_path.write_text('{"source":"locked"}\n', encoding="utf-8")
    execution_payload = {
        "schema_version": 1,
        "protocol_id": "fpct_e1_a5_input_lock_execution_identity_v1",
        "execution_sha": "a" * 40,
        "execution_prefix": "a" * 8,
        "run_uid": "fpct-e1-a5-runtime-prompt-aaaaaaaa-v1",
        "run_root": str(tmp_path.absolute()),
        "source_snapshot_root": str(repo_root.absolute()),
        "source_snapshot_receipt": {
            "path": str(source_receipt_path.absolute()),
            "bytes": source_receipt_path.stat().st_size,
            "file_sha256": prepare.sha256_file(source_receipt_path),
            "verification": {
                "status": "GO_MOUNTED_SOURCE_SNAPSHOT",
                "mounted_tree_canonical_sha256": "7" * 64,
            },
        },
        "input_lock_root": str(output_root.absolute()),
        "empty_downstream_roots_verified": ["k8s", "locks", "raw", "runtime"],
        "historical_execution_resume_allowed": False,
        "historical_artifact_reuse_allowed": False,
    }
    identity_path = output_root / prepare.RUN_IDENTITY_NAME
    prepare.atomic_json(identity_path, execution_payload)
    identity = {
        **execution_payload,
        "identity_sha256": prepare.sha256_file(identity_path),
    }

    asset_state_holder = {
        "value": {
            "tracked_source_assets": [],
            "e0_data_assets": {"tree_sha256": "3" * 64},
            "runtime_assets": {},
            "source_snapshot_verification": {
                "status": "GO_MOUNTED_SOURCE_SNAPSHOT"
            },
            "aggregate_sha256": "d" * 64,
        }
    }
    monkeypatch.setattr(
        prepare,
        "input_asset_state",
        lambda **kwargs: dict(asset_state_holder["value"]),
    )

    runtime_roots = {
        "receiver": tmp_path / "receiver-assets",
        "sender": tmp_path / "sender-assets",
    }
    for path in runtime_roots.values():
        path.mkdir()

    def fake_runtime_tree(path: Path) -> dict:
        absolute = path.absolute()
        return {
            "requested_path": str(absolute),
            "resolved_path": str(absolute),
            "root_kind": "directory",
            "root_symlink_target": None,
            "asset_scope": "tokenizer_config_chat_template_only_no_weights",
            "weight_or_checkpoint_file_opened": False,
            "files": [{"relative_path": "config.json", "sha256": "4" * 64}],
            "file_count": 1,
            "bytes": 1,
            "tree_sha256": "5" * 64,
        }

    tokenizer_files = [{"path": "tokenizer.json", "bytes": 1, "sha256": "6" * 64}]
    monkeypatch.setattr(prepare, "tokenizer_runtime_asset_tree", fake_runtime_tree)
    monkeypatch.setattr(prepare, "_tokenizer_files", lambda path: tokenizer_files)
    runtime_assets = {
        role: {"model_id": role, **fake_runtime_tree(path)}
        for role, path in runtime_roots.items()
    }
    dimensions = {
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
    }
    items = [_geometry_item(task, index) for index, task in enumerate(counts)]
    census_records = []
    for index, item in enumerate(items):
        item["N_s"] = 28 * 16
        item["expected_long_form_rows"] = item["N_s"]
        item["expected_chunk_count"] = 1
        item.update(
            {
                "production_rendered_prompt_sha256": "8" * 64,
                "historical_rendered_prompt_sha256": "9" * 64,
                "production_alignment_sha256": "a" * 64,
                "historical_alignment_sha256": "b" * 64,
                "raw_full_row_sha256": f"{index + 10:064x}",
                "prompt_relation": "EXACT_HISTORICAL_AND_PRODUCTION_MATCH",
                "choice_difference_only": False,
            }
        )
        item["item_semantic_sha256"] = prepare.nested_sha256(
            {key: value for key, value in item.items() if key != "item_semantic_sha256"}
        )
        census_records.append(
            {
                "schema_version": 7,
                "protocol_id": prepare.A5_PROTOCOL_ID,
                "artifact_type": "a5_prompt_census_record",
                "task": item["task"],
                "content_group_sha256": item["content_group_sha256"],
                "sample_key_sha256": item["sample_sha256"],
                "source_row_id": str(index),
                "historical_choice_count": 4,
                "production_choice_count": 4,
                "raw_choice_labels": list("ABCD"),
                "gold_answer": "A",
                "historical_first4_question_choices_sha256": "c" * 64,
                "historical_rendered_prompt_sha256": "9" * 64,
                "historical_alignment_sha256": "b" * 64,
                "raw_full_row_sha256": item["raw_full_row_sha256"],
                "production_rendered_prompt_sha256": "8" * 64,
                "production_alignment_sha256": "a" * 64,
                "historical_prompt_token_count": 10,
                "production_prompt_token_count": 10,
                "production_certified_parent_count": 1,
                "production_logical_row_count": item["N_s"],
                "production_physical_chunk_count": 1,
                "prompt_relation": item["prompt_relation"],
                "choice_difference_only": False,
            }
        )
    census_records_path = output_root / prepare.PROMPT_CENSUS_RECORDS_NAME
    census_payload = b"".join(
        prepare.canonical_json_bytes(row) for row in census_records
    )
    census_records_path.write_bytes(census_payload)
    census_manifest = prepare.build_census_manifest(
        census_records,
        execution_sha="a" * 40,
        run_uid="fpct-e1-a5-runtime-prompt-aaaaaaaa-v1",
        record_artifact={
            "relative_path": prepare.PROMPT_CENSUS_RECORDS_NAME,
            "sha256": prepare.sha256_file(census_records_path),
            "bytes": census_records_path.stat().st_size,
            "row_count": len(census_records),
        },
        expected_task_counts=counts,
    )
    census_manifest_path = output_root / prepare.PROMPT_CENSUS_NAME
    prepare.atomic_json(census_manifest_path, census_manifest)
    geometry = _write_geometry_lock(
        output_dir=output_root,
        items=items,
        schema_sha256=schema_sha256,
        synthetic_gate=gate,
        input_assets_before=asset_state_holder["value"],
        input_assets_after=asset_state_holder["value"],
        execution_identity=identity,
    )
    streamed = _write_streaming_template_lock(
        output_dir=output_root,
        dimensions=dimensions,
        schema_sha256=schema_sha256,
        geometry_lock=geometry,
        execution_identity=identity,
        input_assets_before_sha256=asset_state_holder["value"]["aggregate_sha256"],
        input_assets_after_sha256=asset_state_holder["value"]["aggregate_sha256"],
        synthetic_gate_path=gate_path,
        synthetic_gate=gate,
    )
    sidecar_path = output_root / "e0_design_input_sidecar.pt"
    sidecar = {
        "schema_version": prepare.SCHEMA_VERSION,
        "protocol_id": prepare.PROTOCOL_ID,
        "status": "A5_INPUT_LOCK_GO_NO_MODEL_OUTPUT",
        "items": items,
        "dimensions": dimensions,
        "execution_identity": identity,
        "input_asset_state": asset_state_holder["value"],
        "source": {
            "split_sha256": "1" * 64,
            "dev_manifest_sha256": "2" * 64,
            "e0_data_root": str((tmp_path / "e0-data").absolute()),
        },
        "tokenizers": {
            role: {
                "name": role,
                "path": str(path.absolute()),
                "files": tokenizer_files,
            }
            for role, path in runtime_roots.items()
        },
        "runtime_assets": runtime_assets,
        "a5_prompt_provenance": {
            "renderer_identity": {
                **renderer_source_identity,
                "historical_oracle_mode": "EXACT_FROZEN_SOURCE_IDENTITY",
                "production_renderer_exactly_attested": True,
                "production_renderer_exact_attestation_pending": None,
                "full_row_replay_group_count": sum(counts.values()),
                "prechat_prompt_replay_equal": True,
                "receiver_sender_rendered_replay_equal": True,
                "production_alignment_replay_equal": True,
            },
            "prompt_config_identity": prompt_config_identity,
            "runtime_prompt_assets": {
                "materialized_e0_dev_data_tree_sha256": "3" * 64
            },
        },
        "e1_pilot_consumed": False,
        "model_or_checkpoint_loaded": False,
        "cuda_initialized": False,
        "firewall": {
            "e1_pilot_consumed": False,
            "e1_pilot_rendered_tokenized_aligned_run_or_read": False,
            "model_selection_consumed": False,
            "test_consumed": False,
            "confirmatory_consumed": False,
            "model_or_checkpoint_loaded": False,
            "gpu_or_cuda_used": False,
            "training": False,
        },
        "streaming_contract": {
            "protocol_id": prepare.A4_PROTOCOL_ID,
            "schema_sha256": schema_sha256,
            "physical_chunk_rows": prepare.PHYSICAL_CHUNK_ROWS,
            "synthetic_gate": {
                "path": str(gate_path.absolute()),
                "sha256": prepare.sha256_file(gate_path),
            },
            "geometry_lock": geometry,
            "streaming_template_lock": streamed,
        },
    }
    sidecar["expanded_row_absence_proof"] = expanded_row_absence_proof(sidecar)
    torch.save(sidecar, sidecar_path)
    manifest_path = output_root / "e0_design_input_manifest.json"
    hard_checks = {
        name: True
        for name in (
            "historical_projection_anchor_unchanged",
            "e0_design_membership_unchanged",
            "renderer_source_identity_attested",
            "production_renderer_exactly_attested",
            "production_data_tree_exactly_attested",
            "all_326_groups_resolved",
            "all_first4_choices_equal_historical",
            "all_gold_answers_in_A_B_C_D",
            "all_prompt_differences_classified",
            "historical_to_runtime_mapping_complete",
            "production_prompt_replay_equal",
            "production_alignment_replay_equal",
            "choice_order_preserved",
            "raw_full_row_hashes_complete",
            "logical_row_coverage_exact",
            "streaming_semantic_replay_equal",
            "bounded_peak_rss",
        )
    }
    completed = {
        "schema_version": 7,
        "protocol_id": prepare.A5_PROTOCOL_ID,
        "artifact_type": "a5_input_lock_manifest",
        "status": "A5_INPUT_LOCK_GO_NO_MODEL_OUTPUT",
        "split_role": "e0_design",
        "execution": {
            "execution_sha": "a" * 40,
            "run_uid": execution_payload["run_uid"],
            "run_root": execution_payload["run_root"],
            "source_snapshot_receipt_sha256": execution_payload[
                "source_snapshot_receipt"
            ]["file_sha256"],
            "source_snapshot_tree_sha256": "7" * 64,
        },
        "task_counts": counts,
        "provenance": {
            "renderer_source_identity_sha256": prepare.nested_sha256(
                sidecar["a5_prompt_provenance"]["renderer_identity"]
            ),
            "prompt_config_identity_sha256": prepare.nested_sha256(
                prompt_config_identity
            ),
            "runtime_prompt_assets_sha256": prepare.nested_sha256(
                sidecar["a5_prompt_provenance"]["runtime_prompt_assets"]
            ),
            "materialized_e0_dev_data_tree_sha256": "3" * 64,
            "input_assets_before_sha256": "d" * 64,
            "input_assets_after_sha256": "d" * 64,
            "input_assets_unchanged": True,
        },
        "census": {
            "manifest": {
                "path": str(census_manifest_path),
                "bytes": census_manifest_path.stat().st_size,
                "sha256": prepare.sha256_file(census_manifest_path),
            },
            "records": {
                "path": str(census_records_path),
                "bytes": census_records_path.stat().st_size,
                "sha256": prepare.sha256_file(census_records_path),
            },
            "record_count": len(census_records),
            "canonical_semantic_stream_sha256": census_manifest[
                "canonical_semantic_stream_sha256"
            ],
        },
        "sidecar": {
            "contract_version": prepare.SCHEMA_VERSION,
            "path": str(sidecar_path.absolute()),
            "bytes": sidecar_path.stat().st_size,
            "file_sha256": prepare.sha256_file(sidecar_path),
            "semantic_sha256": prepare.nested_sha256(sidecar),
            "item_count": len(items),
            "expanded_logical_rows_present": False,
        },
        "streaming": {
            "protocol_id": prepare.A4_PROTOCOL_ID,
            "schema_sha256": schema_sha256,
            "physical_chunk_rows": prepare.PHYSICAL_CHUNK_ROWS,
            "synthetic_gate_sha256": prepare.sha256_file(gate_path),
            "geometry_lock_receipt_sha256": geometry["receipt"]["sha256"],
            "streaming_template_lock_receipt_sha256": streamed["receipt"]["sha256"],
            "logical_row_coverage_exact": True,
            "streaming_semantic_replay_equal": True,
            "whole_table_materialization_detected": False,
        },
        "hard_gate_checks": hard_checks,
        "zero_counts": {
            "unexpected_prompt_difference_count": 0,
            "missing_rows": 0,
            "duplicate_rows": 0,
        },
        "firewall": {
            "old_execution_artifact_reused": False,
            "model_instantiated": False,
            "model_or_checkpoint_loaded": False,
            "model_forward_run": False,
            "gpu_or_cuda_used": False,
            "kubernetes_used": False,
            "training": False,
            "e1_pilot_consumed": False,
            "confirmatory_consumed": False,
        },
        "e1_2_or_e1_3_authorized": False,
    }
    prepare.atomic_json(manifest_path, completed)
    go_receipt = {
        "schema_version": 7,
        "protocol_id": prepare.A5_PROTOCOL_ID,
        "artifact_type": "a5_input_lock_receipt",
        "status": "A5_INPUT_LOCK_GO",
        "execution_sha": "a" * 40,
        "run_uid": execution_payload["run_uid"],
        "run_root": execution_payload["run_root"],
        "source_snapshot_receipt_sha256": execution_payload[
            "source_snapshot_receipt"
        ]["file_sha256"],
        "prompt_census_manifest_sha256": prepare.sha256_file(census_manifest_path),
        "checks": hard_checks,
        "firewall": {
            "old_execution_artifact_reused": False,
            "whole_table_materialization_detected": False,
            "model_instantiated": False,
            "model_or_checkpoint_loaded": False,
            "model_forward_run": False,
            "gpu_or_kubernetes_used": False,
            "e1_pilot_consumed": False,
            "confirmatory_consumed": False,
        },
        "zero_counts": completed["zero_counts"],
        "resume_from_group_one": True,
        "downstream_e1_2_authorized": False,
    }
    prepare.atomic_json(output_root / prepare.GO_RECEIPT_NAME, go_receipt)
    return {
        "repo_root": repo_root,
        "e0_data_root": tmp_path / "e0-data",
        "output_root": output_root,
        "sidecar_path": sidecar_path,
        "manifest_path": manifest_path,
        "identity": identity,
        "completed": completed,
        "gate_path": gate_path,
        "source_receipt_path": source_receipt_path,
        "asset_state_holder": asset_state_holder,
        "streamed": streamed,
    }


def _small_geometry_for_hostile_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, object], dict[str, object], dict[str, object]]:
    counts = {"ai2-arc": 1, "openbookqa": 1, "mmlu-redux": 1}
    monkeypatch.setattr(prepare, "TASK_GROUP_COUNTS", counts)
    monkeypatch.setattr(
        prepare, "validate_streaming_schema_artifact", lambda *args, **kwargs: None
    )
    output = tmp_path / "input_lock"
    output.mkdir()
    items = [_geometry_item(task, index) for index, task in enumerate(counts)]
    gate = {
        "estimated_physical_bytes_per_row": 4096,
        "checks": {
            name: True
            for name in (
                "row_key_reference_equivalence",
                "weights_reference_equivalence",
                "topology_reference_equivalence",
                "chunk_partition_semantic_equivalence",
                "aggregate_partition_equivalence",
                "bounded_peak_rss",
                "atomic_no_overwrite",
                "crash_resume_equivalence",
            )
        },
    }
    assets = {"aggregate_sha256": "d" * 64}
    identity = {"identity_sha256": "e" * 64}
    geometry = _write_geometry_lock(
        output_dir=output,
        items=items,
        schema_sha256="a" * 64,
        synthetic_gate=gate,
        input_assets_before=assets,
        input_assets_after=assets,
        execution_identity=identity,
    )
    return output, geometry, gate, identity


@pytest.mark.parametrize("hostile_path", ("row_root", "sample_root", "chunk", "index"))
def test_creation_preflight_rejects_alias_before_external_target_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hostile_path: str,
) -> None:
    output, geometry, gate, identity = _small_geometry_for_hostile_preflight(
        tmp_path, monkeypatch
    )
    external = tmp_path / "external-target"
    external.mkdir()
    marker = external / "marker.bin"
    marker.write_bytes(b"must remain byte-identical")
    row_root = output / "row_templates"
    first_sample = f"{1:064x}"
    if hostile_path == "row_root":
        row_root.symlink_to(external, target_is_directory=True)
    else:
        row_root.mkdir()
        if hostile_path == "sample_root":
            (row_root / first_sample).symlink_to(external, target_is_directory=True)
        elif hostile_path == "chunk":
            sample_root = row_root / first_sample
            sample_root.mkdir()
            (sample_root / "chunk_00000000.parquet").symlink_to(marker)
        else:
            (output / "input_row_template_chunk_index.json").symlink_to(marker)
    before = marker.read_bytes()
    before_names = sorted(
        path.relative_to(external).as_posix() for path in external.rglob("*")
    )
    gate_path = tmp_path / "synthetic-gate.json"
    gate_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises((RuntimeError, ValueError), match="canonical|non-symlink|regular"):
        _write_streaming_template_lock(
            output_dir=output,
            dimensions={
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
            },
            schema_sha256="a" * 64,
            geometry_lock=geometry,
            execution_identity=identity,
            input_assets_before_sha256="d" * 64,
            input_assets_after_sha256="d" * 64,
            synthetic_gate_path=gate_path,
            synthetic_gate=gate,
        )
    assert marker.read_bytes() == before
    assert sorted(
        path.relative_to(external).as_posix() for path in external.rglob("*")
    ) == before_names
    assert not tuple(external.glob("execution_binding.json"))


def test_geometry_recovery_preflight_rejects_alias_with_zero_external_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, geometry, gate, identity = _small_geometry_for_hostile_preflight(
        tmp_path, monkeypatch
    )
    samples = Path(geometry["samples"]["path"])
    external = tmp_path / "external-geometry.parquet"
    samples.rename(external)
    samples.symlink_to(external)
    before = external.read_bytes()
    items = [
        _geometry_item(task, index)
        for index, task in enumerate(prepare.TASK_GROUP_COUNTS)
    ]
    with pytest.raises(RuntimeError, match="canonical regular"):
        _write_geometry_lock(
            output_dir=output,
            items=items,
            schema_sha256="a" * 64,
            synthetic_gate=gate,
            input_assets_before={"aggregate_sha256": "d" * 64},
            input_assets_after={"aggregate_sha256": "d" * 64},
            execution_identity=identity,
        )
    assert external.read_bytes() == before


def test_completed_manifest_resume_deeply_replays_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _completed_input_lock_fixture(tmp_path, monkeypatch)
    before = {
        path.relative_to(fixture["output_root"]).as_posix(): prepare.sha256_file(path)
        for path in Path(fixture["output_root"]).rglob("*")
        if path.is_file()
    }
    resumed = prepare._prepare_input_lock_after_identity(
        repo_root=Path(fixture["repo_root"]),
        e0_data_root=Path(fixture["e0_data_root"]),
        output_sidecar=Path(fixture["sidecar_path"]),
        output_manifest=Path(fixture["manifest_path"]),
        execution_identity=fixture["identity"],
    )
    after = {
        path.relative_to(fixture["output_root"]).as_posix(): prepare.sha256_file(path)
        for path in Path(fixture["output_root"]).rglob("*")
        if path.is_file()
    }
    assert resumed == fixture["completed"]
    assert after == before


def test_completed_manifest_resume_cleans_only_owned_crash_debris(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _completed_input_lock_fixture(tmp_path, monkeypatch)
    output_root = Path(fixture["output_root"])
    owned_root = output_root / ".e0_design_input_manifest.json.abcdefgh.tmp"
    index = json.loads((output_root / "input_row_template_chunk_index.json").read_text())
    sample_manifest = output_root / index["records"][0]["manifest_relative_path"]
    owned_sample = sample_manifest.parent / ".chunk_manifest.json.tmp"
    owned_root.write_bytes(b"crash debris")
    owned_sample.write_bytes(b"crash debris")
    resumed = prepare._prepare_input_lock_after_identity(
        repo_root=Path(fixture["repo_root"]),
        e0_data_root=Path(fixture["e0_data_root"]),
        output_sidecar=Path(fixture["sidecar_path"]),
        output_manifest=Path(fixture["manifest_path"]),
        execution_identity=fixture["identity"],
    )
    assert resumed == fixture["completed"]
    assert not owned_root.exists()
    assert not owned_sample.exists()

    unknown = output_root / ".intruder.tmp"
    unknown.write_bytes(b"not producer-owned")
    with pytest.raises(RuntimeError, match="missing or unbound artifacts"):
        prepare._prepare_input_lock_after_identity(
            repo_root=Path(fixture["repo_root"]),
            e0_data_root=Path(fixture["e0_data_root"]),
            output_sidecar=Path(fixture["sidecar_path"]),
            output_manifest=Path(fixture["manifest_path"]),
            execution_identity=fixture["identity"],
        )
    assert unknown.is_file()


@pytest.mark.parametrize(
    "corruption",
    (
        "geometry_receipt",
        "sample_chunk",
        "missing_sample_chunk",
        "sample_binding",
        "global_index",
        "gate",
        "source_receipt",
        "asset_state",
    ),
)
def test_completed_manifest_resume_rejects_deep_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    fixture = _completed_input_lock_fixture(tmp_path, monkeypatch)
    output_root = Path(fixture["output_root"])
    missing_path: Path | None = None
    if corruption == "geometry_receipt":
        (output_root / "input_geometry_receipt.json").write_text("{}\n", encoding="utf-8")
    elif corruption in {"sample_chunk", "missing_sample_chunk"}:
        index = json.loads((output_root / "input_row_template_chunk_index.json").read_text())
        sample_manifest = output_root / index["records"][0]["manifest_relative_path"]
        physical = json.loads(sample_manifest.read_text())
        chunk_path = sample_manifest.parent / physical["chunks"][0]["relative_path"]
        if corruption == "sample_chunk":
            chunk_path.write_bytes(b"corrupt")
        else:
            chunk_path.unlink()
            missing_path = chunk_path
    elif corruption == "sample_binding":
        index = json.loads((output_root / "input_row_template_chunk_index.json").read_text())
        binding = output_root / index["records"][0]["execution_binding_relative_path"]
        binding.write_text("{}\n", encoding="utf-8")
    elif corruption == "global_index":
        (output_root / "input_row_template_chunk_index.json").write_text("{}\n", encoding="utf-8")
    elif corruption == "gate":
        Path(fixture["gate_path"]).write_text('{"changed":true}\n', encoding="utf-8")
    elif corruption == "source_receipt":
        Path(fixture["source_receipt_path"]).write_text('{"changed":true}\n', encoding="utf-8")
    else:
        fixture["asset_state_holder"]["value"] = {
            **fixture["asset_state_holder"]["value"],
            "aggregate_sha256": "f" * 64,
        }
    with pytest.raises((RuntimeError, ValueError, OSError)):
        prepare._prepare_input_lock_after_identity(
            repo_root=Path(fixture["repo_root"]),
            e0_data_root=Path(fixture["e0_data_root"]),
            output_sidecar=Path(fixture["sidecar_path"]),
            output_manifest=Path(fixture["manifest_path"]),
            execution_identity=fixture["identity"],
        )
    if missing_path is not None:
        assert not missing_path.exists()


@pytest.mark.parametrize(
    ("relative_path", "is_directory"),
    (
        ("row_templates", True),
        ("input_row_template_chunk_index.json", False),
        ("streaming_input_lock_receipt.json", False),
    ),
)
def test_completed_manifest_resume_rejects_symlink_artifact_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    is_directory: bool,
) -> None:
    fixture = _completed_input_lock_fixture(tmp_path, monkeypatch)
    output_root = Path(fixture["output_root"])
    canonical = output_root / relative_path
    external = tmp_path / f"external-{canonical.name}"
    canonical.rename(external)
    canonical.symlink_to(external, target_is_directory=is_directory)

    with pytest.raises(
        RuntimeError, match="non-symlink|canonical real|canonical regular"
    ):
        prepare._prepare_input_lock_after_identity(
            repo_root=Path(fixture["repo_root"]),
            e0_data_root=Path(fixture["e0_data_root"]),
            output_sidecar=Path(fixture["sidecar_path"]),
            output_manifest=Path(fixture["manifest_path"]),
            execution_identity=fixture["identity"],
        )


def test_geometry_lock_rejects_changed_asset_hash_without_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = {"ai2-arc": 1, "openbookqa": 1, "mmlu-redux": 1}
    monkeypatch.setattr(prepare, "TASK_GROUP_COUNTS", counts)
    items = [_geometry_item(task, index) for index, task in enumerate(counts)]
    with pytest.raises(RuntimeError, match="input assets changed"):
        _write_geometry_lock(
            output_dir=tmp_path,
            items=items,
            schema_sha256="a" * 64,
            synthetic_gate={"estimated_physical_bytes_per_row": 4096},
            input_assets_before={"aggregate_sha256": "1" * 64},
            input_assets_after={"aggregate_sha256": "2" * 64},
            execution_identity={"identity_sha256": "3" * 64},
        )
    assert not (tmp_path / "input_geometry_receipt.json").exists()


def test_sidecar_absence_proof_and_blocked_receipt(tmp_path: Path) -> None:
    item = _geometry_item("ai2-arc", 0)
    item["N_s"] = 28 * 16
    item["expected_long_form_rows"] = item["N_s"]
    payload = {"items": [item]}
    assert expanded_row_absence_proof(payload)["expanded_logical_rows_present"] is False
    payload["row_ordinal"] = 0
    with pytest.raises(ValueError, match="expanded logical-row"):
        expanded_row_absence_proof(payload)

    monkey_identity = {
        "execution_sha": "a" * 40,
        "run_uid": "fpct-e1-a5-runtime-prompt-aaaaaaaa-v1",
        "run_root": "/netdisk/lijunsi/fpct-e1/fpct-e1-a5-aaaaaaaa-v1",
        "source_snapshot_root": str(REPO_ROOT),
    }
    _publish_blocked_receipt(
        tmp_path, ValueError("disk preflight failed"), monkey_identity
    )
    blocked = json.loads((tmp_path / prepare.BLOCKED_RECEIPT_NAME).read_text())
    assert blocked["status"] == "A5_INPUT_LOCK_BLOCKED"
    assert blocked["resume_allowed"] is False
    assert blocked["artifact_reuse_allowed"] is False
    assert blocked["scientific_result"] is False
    assert blocked["downstream_e1_2_or_e1_3_authorized"] is False


def test_a5_bounded_rss_uses_streaming_stress_evidence_not_checks_map() -> None:
    synthetic_gate = {
        "checks": {},
        "streaming_stress": {"bounded_peak_rss": True},
    }
    assert prepare._a5_bounded_peak_rss_gate(
        {"bounded_peak_rss": True}, synthetic_gate
    )
    assert not prepare._a5_bounded_peak_rss_gate(
        {"bounded_peak_rss": False}, synthetic_gate
    )
    assert not prepare._a5_bounded_peak_rss_gate(
        {"bounded_peak_rss": True}, {"checks": {"bounded_peak_rss": True}}
    )


def test_a5_sidecar_synthetic_gate_binding_is_path_and_byte_exact(
    tmp_path: Path,
) -> None:
    gate_path = tmp_path / "gate.json"
    gate_path.write_text('{"status":"GO"}\n', encoding="utf-8")
    assert prepare._a5_synthetic_gate_binding(gate_path) == {
        "path": str(gate_path.absolute()),
        "sha256": prepare.sha256_file(gate_path),
    }


def test_prepare_cli_direct_invocation_fails_before_input_access(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "must-not-read-data"
    marker.write_text("protected input marker\n", encoding="utf-8")
    before = marker.read_bytes()
    result = subprocess.run(
        _direct_prepare_command(tmp_path),
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(REPO_ROOT),
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DATASETS_OFFLINE": "1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0
    assert "canonical python -I fpct_bootstrap.py" in result.stderr
    assert marker.read_bytes() == before
    assert not list(tmp_path.rglob("A5_INPUT_LOCK_BLOCKED.json"))


def test_prepare_cli_hostile_pythonpath_cannot_spoof_bootstrap(
    tmp_path: Path,
) -> None:
    fake = tmp_path / "hostile-pythonpath"
    fake.mkdir()
    (fake / "fpct_bootstrap.py").write_text(
        "def require_active(*, target=None):\n"
        "    return {'repo_root': 'spoofed', 'target': str(target)}\n",
        encoding="utf-8",
    )
    marker = tmp_path / "must-not-read-data"
    marker.write_text("protected input marker\n", encoding="utf-8")
    before = marker.read_bytes()
    result = subprocess.run(
        _direct_prepare_command(tmp_path),
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(fake), str(REPO_ROOT))),
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DATASETS_OFFLINE": "1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0
    assert "outside the snapshot" in result.stderr
    assert marker.read_bytes() == before
    assert not list(tmp_path.rglob("A5_INPUT_LOCK_BLOCKED.json"))


def test_public_lock_turns_integrity_failure_into_terminal_blocked_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    output = tmp_path / "input_lock"
    output.mkdir()
    monkeypatch.setattr(
        prepare,
        "validate_a5_execution_identity",
        lambda **kwargs: {
            "identity_sha256": "a" * 64,
            "execution_sha": "a" * 40,
            "run_uid": "fpct-e1-a5-runtime-prompt-aaaaaaaa-v1",
            "run_root": str(tmp_path),
            "source_snapshot_root": str(snapshot),
        },
    )
    monkeypatch.setattr(
        a5_prompt_gate, "validate_a5_schema_artifact", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        prepare,
        "_prepare_input_lock_after_identity",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("A5_PREFLIGHT_FAILED:disk_preflight")
        ),
    )
    with pytest.raises(RuntimeError, match="PREFLIGHT"):
        prepare.prepare_input_lock(
            repo_root=snapshot,
            e0_data_root=tmp_path / "data",
            output_sidecar=output / "sidecar.pt",
            output_manifest=output / "manifest.json",
            execution_sha="a" * 40,
            source_snapshot_root=snapshot,
            source_snapshot_receipt=snapshot / "receipt.json",
            run_uid="fpct-e1-a5-runtime-prompt-aaaaaaaa-v1",
            run_root=tmp_path,
            _test_only_sealed_execution=(
                prepare._verified_test_sealed_prepare_sentinel(
                    snapshot, "a" * 40
                )
            ),
        )
    blocked = json.loads((output / prepare.BLOCKED_RECEIPT_NAME).read_text())
    assert blocked["status"] == "A5_INPUT_LOCK_BLOCKED"
    assert blocked["failed_check"] == "a5_preflight_failed"
