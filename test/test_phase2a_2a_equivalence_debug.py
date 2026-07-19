from __future__ import annotations

import csv
import json
from pathlib import Path

from script.analysis import phase2a_2a_equivalence_debug as debug


def _write_outputs(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subject",
                "question_id",
                "pred",
                "cot_pred",
                "cot_output",
                "cot_gen_length",
                "is_correct",
                "true_answer",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_read_outputs_keeps_only_output_fields_and_fit_members(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    _write_outputs(
        path,
        [
            {
                "subject": "fit",
                "question_id": "1",
                "pred": "A",
                "cot_pred": "A",
                "cot_output": "The answer is A.",
                "cot_gen_length": "5",
                "is_correct": "SECRET_CORRECTNESS",
                "true_answer": "SECRET_LABEL",
            },
            {
                "subject": "sealed",
                "question_id": "2",
                "pred": "SECRET_PREDICTION",
                "cot_pred": "SECRET_PREDICTION",
                "cot_output": "SECRET_OUTPUT",
                "cot_gen_length": "999",
                "is_correct": "SECRET_CORRECTNESS",
                "true_answer": "SECRET_LABEL",
            },
        ],
    )
    debug.FIT_COUNTS["ai2-arc"] = 1
    try:
        rows = debug._read_outputs(
            path,
            dataset="ai2-arc",
            fit_members={("ai2-arc", "fit", "1")},
        )
    finally:
        debug.FIT_COUNTS["ai2-arc"] = 351
    serialized = json.dumps(list(rows.values()), sort_keys=True)
    assert "SECRET_CORRECTNESS" not in serialized
    assert "SECRET_LABEL" not in serialized
    assert "SECRET_PREDICTION" not in serialized
    assert rows[("ai2-arc", "fit", "1")]["cot_output"] == "The answer is A."


def test_primary_equivalence_uses_all_prediction_generation_and_length_fields() -> None:
    key = ("ai2-arc", "SPLIT_0_OF_1", "7")
    left = {
        key: {
            "pred": "A",
            "cot_pred": "A",
            "cot_output": "The answer is A.",
            "cot_gen_length": "5",
        }
    }
    cot_pred_only = {key: {**left[key], "cot_pred": "B"}}
    generation_changed = {key: {**left[key], "cot_output": "A."}}

    diagnostic_only = debug._compare_tables(
        left, cot_pred_only, left_name="left", right_name="right"
    )
    primary = debug._compare_tables(
        left, generation_changed, left_name="left", right_name="right"
    )

    assert diagnostic_only["exact"] is False
    assert diagnostic_only["any_mismatch_count"] == 1
    assert diagnostic_only["primary_mismatch_count"] == 1
    assert primary["exact"] is False
    assert primary["primary_mismatch_count"] == 1


def test_core_config_sha_ignores_only_instrumentation_and_output_path() -> None:
    base = {
        "eval": {"dataset": "ai2-arc", "gpu_ids": [0]},
        "model": {"generation_config": {"do_sample": False}},
        "output": {"output_dir": "/one"},
    }
    instrumented = json.loads(json.dumps(base))
    instrumented["output"]["output_dir"] = "/two"
    instrumented["eval"]["cache_geometry_instrumentation"] = {
        "enabled": True,
        "capture_mode": "capture",
    }
    changed_generation = json.loads(json.dumps(base))
    changed_generation["model"]["generation_config"]["do_sample"] = True

    assert debug._core_config_sha(base) == debug._core_config_sha(instrumented)
    assert debug._core_config_sha(base) != debug._core_config_sha(changed_generation)
