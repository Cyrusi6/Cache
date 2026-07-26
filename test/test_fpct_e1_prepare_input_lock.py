from __future__ import annotations

import ast
from pathlib import Path

import pytest

from script.experiment.fpct_e1_prepare_input_lock import (
    MAX_LONG_FORM_ROWS_PER_SAMPLE,
    answer_query_contract,
    certified_parent_contract,
    expected_long_form_row_volume,
    expected_long_form_rows,
    row_template_attestation,
    row_template_records,
    raw_topology_ledger,
    task_membership_sha256,
)
from script.experiment.fpct_e1_capture_runner import (
    ROW_TEMPLATE_COLUMNS,
    _KeyLedger,
    _key_bytes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


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
                "topology": "partition_compositional",
            },
            {
                "parent_position": 4,
                "candidate_count": 3,
                "prior": [1 / 3, 1 / 3, 1 / 3],
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


def test_expected_long_form_rows_freezes_exact_product_and_hard_cap() -> None:
    assert expected_long_form_rows(
        answer_query_count=2,
        certified_parent_count=3,
        num_layers=28,
        num_query_heads=16,
    ) == 2 * 3 * 28 * 16
    with pytest.raises(ValueError, match="exceeds"):
        expected_long_form_rows(
            answer_query_count=MAX_LONG_FORM_ROWS_PER_SAMPLE,
            certified_parent_count=2,
            num_layers=1,
            num_query_heads=1,
        )


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
        "sample_ceiling": MAX_LONG_FORM_ROWS_PER_SAMPLE,
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
