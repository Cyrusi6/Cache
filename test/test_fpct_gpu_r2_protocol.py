import json
import hashlib
from pathlib import Path

from script.analysis.fpct_gpu_r2_diagnostics import (
    CARDINALITIES,
    GROUPS_PER_CELL,
    OPERATOR_PANEL_IDS,
    PROFILE_IDS,
    TASKS,
)


ROOT = Path(__file__).resolve().parents[1]


def test_r2_manifest_freezes_hypotheses_and_old_execution() -> None:
    manifest = json.loads(
        (ROOT / "recipe/eval_recipe/fpct_gpu_r2/root_cause_manifest.json").read_text()
    )
    assert manifest["source_parent_commit"] == "6550b2a832107d25fc2e62a0cc8b260c2e160773"
    assert manifest["immutable_old_execution"]["resume_or_patch"] is False
    assert manifest["hypotheses"] == [
        "H1_GATE_NATIVE_NULL", "H2_BF16_PRIOR_DRIFT",
        "H3_BACKEND_AUTODISPATCH", "H4_PROFILER_SCOPE_CONFOUND",
    ]
    assert manifest["attention_backend"]["r2_required"] == "eager"
    assert manifest["operator_panel_ids"] == list(OPERATOR_PANEL_IDS)
    assert manifest["profile_ids"] == list(PROFILE_IDS)
    for source in manifest["normative_documents"]:
        assert hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest() == source["sha256"]
    null_spec = manifest["null_floor_spec"]
    assert hashlib.sha256((ROOT / null_spec["path"]).read_bytes()).hexdigest() == null_spec["sha256"]
    panel = manifest["diagnostic_panel"]
    assert hashlib.sha256((ROOT / panel["manifest"]).read_bytes()).hexdigest() == panel["manifest_sha256"]


def test_metric_null_floors_are_metric_specific() -> None:
    spec = json.loads(
        (ROOT / "recipe/eval_recipe/fpct_gpu_r2/metric_null_floor_spec.json").read_text()
    )
    assert spec["depths"] == [1, 2, 4, 8, 14, 28]
    assert spec["percentile"] == 99.9
    assert spec["safety_multiplier"] == 2.0
    assert len(spec["metrics"]) == 9
    assert len({value["unit"] for value in spec["metrics"].values()}) > 1
    assert 0.0390625 not in [value["absolute_floor"] for value in spec["metrics"].values()]


def test_diagnostic_panel_is_complete_label_free_and_hash_stratified() -> None:
    panel = json.loads(
        (ROOT / "recipe/eval_recipe/fpct_gpu_r2/diagnostic_panel_manifest.json").read_text()
    )
    rows = panel["rows"]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == panel["panel_rows_sha256"]
    assert panel["selection"]["labels_or_correctness_accessed"] is False
    assert len(rows) == len(TASKS) * len(CARDINALITIES) * GROUPS_PER_CELL
    cells = {(row["task"], row["cardinality"]) for row in rows}
    assert cells == {(task, cardinality) for task in TASKS for cardinality in CARDINALITIES}
    for task in TASKS:
        for cardinality in CARDINALITIES:
            cell = [row for row in rows if row["task"] == task and row["cardinality"] == cardinality]
            assert len(cell) == GROUPS_PER_CELL
            assert [row["content_group_sha256"] for row in cell] == sorted(
                row["content_group_sha256"] for row in cell
            )
            assert all(row["split"] in {"fit", "calibration"} for row in cell)
