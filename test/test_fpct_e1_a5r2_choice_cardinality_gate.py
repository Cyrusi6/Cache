from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from script.analysis import fpct_e1_a5r2_choice_cardinality_gate as gate


def test_v9_contract_and_predecessor_bytes_validate() -> None:
    contract = gate._load_contract(gate.REPO_ROOT)
    assert contract["approval"]["approved"] is True
    assert contract["approval"]["user_response_verbatim"] == "批准"
    assert contract["amendment_id"].endswith("_CHOICE_CARDINALITY_RECOVERY")
    assert gate._verify_predecessors(gate.REPO_ROOT) == gate.V8_OBJECT_SHA256


def test_synthetic_choice_cardinality_oracles() -> None:
    assert gate.run_choice_cardinality_oracles(gate.REPO_ROOT) == {
        "taxonomy_precedence_pass": True,
        "task_cardinality_cases_pass": True,
        "historical_min_four_formula_pass": True,
        "label_free_firewall_pass": True,
        "descriptor_mismatch_hard_stop_pass": True,
        "normalization_prompt_parity_gate_pass": True,
        "audit_go_and_blocked_mutations_pass": True,
        "v8_inheritance_boundary_pass": True,
        "fresh_uid_root_nonreuse_pass": True,
        "atomic_failure_publishes_no_go_pass": True,
    }


def test_inherited_v8_streaming_evidence_is_exact_not_remeasured() -> None:
    evidence = gate._inherited_v8_streaming_evidence(gate.REPO_ROOT)
    assert evidence["source_gate_sha256"] == gate.V8_OBJECT_SHA256[
        "recipe/eval_recipe/fpct_e1/e1_a5r1_hash_domain_synthetic_gate.json"
    ]
    assert evidence["estimated_physical_bytes_per_row"] == 4096
    assert evidence["streaming_stress"]["bounded_peak_rss"] is True
    assert all(evidence["streaming_contract_checks"].values())
    assert evidence["cross_resource_or_hash_domain_substitution_allowed"] is False


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("operative_scope", "model_forward"), True),
        (("operative_scope", "gpu_cuda_or_kubernetes"), True),
        (("operative_scope", "e1_2"), True),
        (("authorization_after_input_lock_go", "e1_2"), True),
        (("fresh_execution", "uid_pattern"), ".*"),
        (("fresh_execution", "root_pattern"), ".*"),
        (("immutable_predecessor", "resume_allowed"), True),
    ),
)
def test_contract_schema_rejects_authority_and_identity_drift(
    path: tuple[str, str], replacement: object
) -> None:
    contract = gate._load_contract(gate.REPO_ROOT)
    broken = copy.deepcopy(contract)
    broken[path[0]][path[1]] = replacement
    with pytest.raises(ValueError):
        gate.validate_a5r2_schema_artifact(broken, repo_root=gate.REPO_ROOT)


def test_gate_publication_is_atomic_and_no_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = {
        "schema_version": 9,
        "protocol_id": "fpct_e1_mechanism_audit_v9_a5r2_choice_cardinality",
        "artifact_type": "synthetic-sentinel",
    }
    monkeypatch.setattr(gate, "validate_a5r2_schema_artifact", lambda *_a, **_k: None)
    output = tmp_path / "gate.json"
    gate.publish_gate(value, output)
    assert json.loads(output.read_text(encoding="utf-8")) == value
    with pytest.raises(FileExistsError):
        gate.publish_gate(value, output)


def test_tracked_closure_contains_v9_and_immutable_v8_objects() -> None:
    tracked = set(gate.TRACKED_SOURCE_FILES)
    assert gate.AMENDMENT_RELATIVE.as_posix() in tracked
    assert gate.CONTRACT_RELATIVE.as_posix() in tracked
    assert gate.SCHEMA_RELATIVE.as_posix() in tracked
    assert gate.BLOCKED_CLOSURE_RELATIVE.as_posix() in tracked
    assert set(gate.V8_OBJECT_SHA256) <= tracked
    assert gate.GATE_RELATIVE.as_posix() not in tracked
