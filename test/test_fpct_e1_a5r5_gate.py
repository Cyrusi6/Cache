from __future__ import annotations

import json
from pathlib import Path

import pytest

from script.analysis import fpct_e1_a5r5_gate as gate


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_a5r5_immutable_predecessors_and_forensic_root_are_exact() -> None:
    assert {path: gate.sha256_file(REPO_ROOT / path) for path in gate.IMMUTABLE} == gate.IMMUTABLE
    observed = gate.verify_old_forensic_root()
    assert observed["tree_sha256"] == gate.OLD_TREE_SHA256
    assert observed["file_count"] == 708


def test_build_gate_binds_tests_sources_and_zero_access(tmp_path: Path) -> None:
    output = tmp_path / "pytest.txt"
    output.write_text("135 passed in 1.0s\n", encoding="utf-8")
    value = gate.build_gate(REPO_ROOT, output)
    assert value["status"] == gate.STATUS
    assert value["test_count"] == 135
    assert value["checks"]["natural_successor_accessed"] is False
    assert value["checks"]["model_forward_run"] is False


def test_verify_gate_rejects_tampered_evidence(tmp_path: Path) -> None:
    output = tmp_path / "pytest.txt"
    output.write_text("135 passed in 1.0s\n", encoding="utf-8")
    value = gate.build_gate(REPO_ROOT, output)
    value["test_count"] = 136
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="evidence hash"):
        gate.verify_gate(path, REPO_ROOT)
