from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace


from script.analysis import fpct_e1_streaming_synthetic_gate as gate
from script.analysis.fpct_e1_streaming_verify import semantic_row_bytes


def test_gate_source_and_test_closure_binds_attention_implementation() -> None:
    assert "rosetta/model/fpct_attention.py" in gate.TRACKED_SOURCE_FILES
    assert "script/analysis/fpct_reference_operator.py" in gate.TRACKED_SOURCE_FILES
    assert "test/test_fpct_production_path.py" in gate.TEST_FILES
    assert "test/test_fpct_reference_operator.py" in gate.TEST_FILES
    assert "script/runtime/fpct_bootstrap.py" in gate.TRACKED_SOURCE_FILES
    assert "test/test_fpct_sealed_import.py" in gate.TEST_FILES
    assert "script/experiment/fpct_e1_source_snapshot_lock.py" in gate.TRACKED_SOURCE_FILES
    assert "test/test_fpct_e1_source_snapshot_lock.py" in gate.TEST_FILES
    assert "script/analysis/fpct_e1_instrumentation_gate.py" in gate.TRACKED_SOURCE_FILES
    assert "test/test_fpct_e1_instrumentation_gate_a4.py" in gate.TEST_FILES
    for relative in (
        "recipe/eval_recipe/fpct_e1/e1_instrumentation_parity.json",
        "recipe/eval_recipe/fpct_e1/e1_synthetic_query_variance.json",
        "recipe/eval_recipe/fpct_e1/e1_instrumentation_hard_gate.json",
        "recipe/eval_recipe/fpct_e1/e1_instrumentation_parity_a4.json",
        "recipe/eval_recipe/fpct_e1/e1_synthetic_query_variance_a4.json",
        "recipe/eval_recipe/fpct_e1/e1_instrumentation_hard_gate_a4.json",
    ):
        assert relative in gate.TRACKED_SOURCE_FILES


def test_full_synthetic_row_covers_mechanism_schema_and_bound() -> None:
    schema_sha = gate.sha256_file(gate.STREAMING_SCHEMA)
    stream = gate._stream(3, schema_sha)
    required = set(
        json.loads(gate.MECHANISM_SCHEMA.read_text(encoding="utf-8"))["input_row"][
            "required_columns"
        ]
    )
    row = stream.row_at(stream.row_count - 1)
    assert required.issubset(row)
    assert len(semantic_row_bytes(row)) <= gate._canonical_full_schema_row_bound(
        schema_sha
    )
    assert stream.row_count == gate.LAYERS * gate.QUERY_HEADS * 3


def test_small_full_parquet_worker_is_exact_and_replayable() -> None:
    record = gate._worker(10)
    assert record["logical_rows"] == gate.LAYERS * gate.QUERY_HEADS * 10
    assert record["emitted_rows"] == record["logical_rows"]
    assert record["chunk_count"] == 2
    assert record["semantic_stream_sha256"] == record[
        "replay_semantic_stream_sha256"
    ]
    assert record["physical_file_bytes"] > 0
    assert record["peak_rss_bytes"] > 0


def test_fresh_worker_uses_repo_rooted_module_launch(monkeypatch) -> None:
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return SimpleNamespace(stdout='{"logical_rows": 448}')

    monkeypatch.setenv("PYTHONPATH", "/untrusted/import/root")
    monkeypatch.setattr(gate.subprocess, "run", fake_run)

    result = gate._fresh_worker("module-launch", 1)

    assert observed["command"] == [
        gate.sys.executable,
        "-s",
        "-m",
        gate.WORKER_MODULE,
        "--worker",
        "1",
    ]
    assert observed["cwd"] == gate.REPO_ROOT
    assert observed["env"]["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in observed["env"]
    assert observed["check"] is True
    assert observed["capture_output"] is True
    assert observed["text"] is True
    assert result == {
        "case_id": "module-launch",
        "logical_rows": 448,
    }


def test_gate_helpers_are_deterministic_and_forbidden_import_free() -> None:
    assert gate._next_power_of_two(1) == 1
    assert gate._next_power_of_two(4097) == 8192
    environment = {"python": "x", "pyarrow": "y"}
    assert gate._sha256_value(environment) == gate._sha256_value(environment)
    tree = ast.parse(Path(gate.__file__).read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(
        name.startswith(prefix)
        for name in imports
        for prefix in (
            "transformers",
            "datasets",
            "torch",
            "kubernetes",
            "rosetta.model.wrapper",
        )
    )
