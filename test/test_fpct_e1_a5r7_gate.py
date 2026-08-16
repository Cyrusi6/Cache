from __future__ import annotations

import json
from pathlib import Path

import pytest

from script.analysis import fpct_e1_a5r7_gate as gate


REPO_ROOT = Path(__file__).resolve().parents[1]


def _passing_output() -> str:
    nodes = "\n".join(f"{node} PASSED [ 50%]" for node in gate.GATE_REQUIRED_TEST_NODES)
    return (
        f"{gate.GATE_EXPECTED_COLLECTION_SUMMARY}\n{nodes}\n"
        f"{gate.GATE_EXPECTED_TEST_COUNT} passed in 1.0s\n"
    )


def _stub_retained_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gate, "verify_old_forensic_root", gate.expected_forensic_closure
    )
    monkeypatch.setattr(
        gate, "verify_old_controller_evidence", gate.expected_old_controller_evidence
    )


def test_a5r7_immutable_schemas_are_byte_exact() -> None:
    assert {path: gate.sha256_file(REPO_ROOT / path) for path in gate.IMMUTABLE} == gate.IMMUTABLE


def test_a5r7_build_gate_binds_five_field_and_zero_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_retained_evidence(monkeypatch)
    output = tmp_path / "pytest.txt"
    output.write_text(_passing_output(), encoding="utf-8")
    value = gate.build_gate(REPO_ROOT, output)
    assert value["status"] == gate.STATUS
    assert value["test_count"] == gate.GATE_EXPECTED_TEST_COUNT
    assert value["checks"]["choice_binding_exact_five_field_projection"]
    assert value["checks"]["successor_natural_accessed"] is False
    assert value["test_contract"]["required_nodes"] == list(
        gate.GATE_REQUIRED_TEST_NODES
    )


def test_a5r7_verify_gate_rejects_tampered_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_retained_evidence(monkeypatch)
    output = tmp_path / "pytest.txt"
    output.write_text(_passing_output(), encoding="utf-8")
    value = gate.build_gate(REPO_ROOT, output)
    value["test_count"] = gate.GATE_EXPECTED_TEST_COUNT + 1
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity/status|evidence hash"):
        gate.verify_gate(path, REPO_ROOT)


def test_a5r7_gate_tracks_operative_e1_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_retained_evidence(monkeypatch)
    output = tmp_path / "pytest.txt"
    output.write_text(_passing_output(), encoding="utf-8")
    value = gate.build_gate(REPO_ROOT, output)
    path = "recipe/eval_recipe/fpct_e1/e1_manifest.json"
    assert path in gate.TRACKED
    assert value["tracked_files_sha256"][path] == gate.sha256_file(REPO_ROOT / path)


def test_a5r7_verify_gate_rejects_weakened_checks_even_with_recomputed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_retained_evidence(monkeypatch)
    output = tmp_path / "pytest.txt"
    output.write_text(_passing_output(), encoding="utf-8")
    value = gate.build_gate(REPO_ROOT, output)
    value["checks"]["controller_requires_current_gate_before_natural_access"] = False
    value["evidence_sha256"] = gate.hashlib.sha256(
        gate._canonical_without_evidence(value)
    ).hexdigest()
    path = tmp_path / "weakened.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity/status"):
        gate.verify_gate(path, REPO_ROOT)


def test_a5r7_gate_replays_old_controller_terminal_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        gate, "verify_old_forensic_root", gate.expected_forensic_closure
    )
    calls = 0

    def replay_controller():
        nonlocal calls
        calls += 1
        return gate.expected_old_controller_evidence()

    monkeypatch.setattr(gate, "verify_old_controller_evidence", replay_controller)
    output = tmp_path / "pytest.txt"
    output.write_text(_passing_output(), encoding="utf-8")
    value = gate.build_gate(REPO_ROOT, output)
    assert calls == 1
    assert value["failed_controller_terminal_evidence"] == (
        gate.expected_old_controller_evidence()
    )


def test_a5r7_gate_rejects_rehashed_continuity_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_retained_evidence(monkeypatch)
    output = tmp_path / "pytest.txt"
    output.write_text(_passing_output(), encoding="utf-8")
    value = gate.build_gate(REPO_ROOT, output)
    value["a5r4_continuity"]["unchanged_mode_function_count"] = 12
    value["evidence_sha256"] = gate.hashlib.sha256(
        gate._canonical_without_evidence(value)
    ).hexdigest()
    path = tmp_path / "continuity-tampered.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="continuity changed"):
        gate.verify_gate(path, REPO_ROOT)


def test_a5r7_gate_rejects_symlink_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_retained_evidence(monkeypatch)
    output = tmp_path / "pytest.txt"
    output.write_text(_passing_output(), encoding="utf-8")
    value = gate.build_gate(REPO_ROOT, output)
    target = tmp_path / "gate.json"
    target.write_text(json.dumps(value), encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(RuntimeError, match="absent or unsafe"):
        gate.verify_gate(alias, REPO_ROOT)
