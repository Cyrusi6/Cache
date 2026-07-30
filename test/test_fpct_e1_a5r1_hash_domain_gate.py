from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from script.analysis import fpct_e1_a5r1_hash_domain_gate as gate


def test_v8_contract_and_predecessor_bytes_validate() -> None:
    contract = gate._load_contract(gate.REPO_ROOT)
    assert contract["approval"]["approved"] is True
    assert (
        contract["amendment_id"]
        == "APPROVED_PROSPECTIVE_AMENDMENT_E1_A5R1_HASH_DOMAINS"
    )
    assert gate._verify_v7_predecessors(gate.REPO_ROOT) == gate.V7_OBJECT_SHA256
    gate._verify_contract_hashes(contract, gate.REPO_ROOT)


def test_independent_hash_domain_oracles_and_negative_controls() -> None:
    assert gate.run_hash_domain_oracles() == {
        "independent_declared_reference_oracle_pass": True,
        "independent_generic_reference_oracle_pass": True,
        "deliberate_cross_domain_divergence_accepted": True,
        "swapped_domain_values_fail_closed_pass": True,
        "missing_or_unqualified_domain_value_fails_closed_pass": True,
        "tamper_matrix_pass": True,
        "generic_before_after_equality_pass": True,
        "completed_verifier_dual_recompute_fixture_pass": True,
        "v7_four_object_sha_immutability_pass": True,
        "blocked_execution_and_root_nonreuse_pass": True,
        "atomic_failure_publishes_no_go_pass": True,
    }


def test_atomic_failure_check_executes_producer_terminal_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import script.experiment.fpct_e1_prepare_input_lock as prepare

    calls = {"quarantine": 0, "blocked": 0}
    original_quarantine = prepare._quarantine_canonical_go_before_blocked
    original_blocked = prepare._publish_blocked_receipt

    def counted_quarantine(output_root: Path) -> Path | None:
        calls["quarantine"] += 1
        return original_quarantine(output_root)

    def counted_blocked(*args: object, **kwargs: object) -> None:
        calls["blocked"] += 1
        original_blocked(*args, **kwargs)

    monkeypatch.setattr(
        prepare, "_quarantine_canonical_go_before_blocked", counted_quarantine
    )
    monkeypatch.setattr(prepare, "_publish_blocked_receipt", counted_blocked)
    checks = gate.run_hash_domain_oracles()
    assert checks["atomic_failure_publishes_no_go_pass"] is True
    assert calls == {"quarantine": 1, "blocked": 1}


@pytest.mark.parametrize(
    "missing",
    (
        "e0_declared_tree_algorithm",
        "e0_declared_tree_sha256",
        "generic_asset_tree_algorithm",
        "generic_asset_tree_sha256",
    ),
)
def test_schema_rejects_missing_hash_domain_fields(missing: str) -> None:
    contract = gate._load_contract(gate.REPO_ROOT)
    broken = copy.deepcopy(contract)
    del broken["asset_identity"]["materialized_e0_dev_data_tree"][missing]
    with pytest.raises(ValueError):
        gate.validate_a5r1_schema_artifact(broken, repo_root=gate.REPO_ROOT)


def test_schema_rejects_swapped_domain_labels() -> None:
    contract = gate._load_contract(gate.REPO_ROOT)
    broken = copy.deepcopy(contract)
    identity = broken["asset_identity"]["materialized_e0_dev_data_tree"]
    identity["e0_declared_tree_algorithm"], identity["generic_asset_tree_algorithm"] = (
        identity["generic_asset_tree_algorithm"],
        identity["e0_declared_tree_algorithm"],
    )
    with pytest.raises(ValueError):
        gate.validate_a5r1_schema_artifact(broken, repo_root=gate.REPO_ROOT)


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("operative_scope", "model_forward"), True),
        (("operative_scope", "model_or_checkpoint_load"), True),
        (("operative_scope", "gpu_cuda_or_kubernetes"), True),
        (("operative_scope", "training"), True),
        (("operative_scope", "e1_2"), True),
        (("authorization_after_input_lock_go", "e1_2"), True),
        (("authorization_after_input_lock_go", "model_forward"), True),
        (("authorization_after_input_lock_go", "gpu_or_training"), True),
        (("base_scientific_contract", "path"), "alternate/base.json"),
        (("base_scientific_contract", "sha256"), "0" * 64),
        (("fresh_execution", "uid_pattern"), ".*"),
        (("fresh_execution", "root_pattern"), ".*"),
        (("fresh_execution", "ordered_stages"), []),
    ),
)
def test_contract_schema_rejects_authority_and_identity_drift(
    path: tuple[str, str], replacement: object
) -> None:
    contract = gate._load_contract(gate.REPO_ROOT)
    broken = copy.deepcopy(contract)
    broken[path[0]][path[1]] = replacement
    with pytest.raises(ValueError):
        gate.validate_a5r1_schema_artifact(broken, repo_root=gate.REPO_ROOT)


def test_contract_schema_rejects_empty_immutable_predecessor_map() -> None:
    contract = gate._load_contract(gate.REPO_ROOT)
    broken = copy.deepcopy(contract)
    broken["immutable_predecessors"]["v7_objects"] = {}
    with pytest.raises(ValueError):
        gate.validate_a5r1_schema_artifact(broken, repo_root=gate.REPO_ROOT)


def _blocked_receipt_fixture() -> dict[str, object]:
    return {
        "schema_version": 8,
        "protocol_id": "fpct_e1_mechanism_audit_v8_a5r1_hash_domains",
        "artifact_type": "a5r1_input_lock_blocked_receipt",
        "status": "A5_INPUT_LOCK_BLOCKED",
        "execution_sha": "a" * 40,
        "run_uid": "fpct-e1-a5r1-hash-domains-aaaaaaaa-v1",
        "run_root": "/netdisk/lijunsi/fpct-e1/fpct-e1-a5r1-aaaaaaaa-v1",
        "failed_check": "preflight_failed",
        "failure_detail_sha256": "b" * 64,
        "resume_allowed": False,
        "artifact_reuse_allowed": False,
        "scientific_result": False,
        "e1_2_or_e1_3_authorized": False,
    }


def test_blocked_receipt_hash_domains_are_all_or_none() -> None:
    precomputation_failure = _blocked_receipt_fixture()
    gate.validate_a5r1_schema_artifact(
        precomputation_failure, repo_root=gate.REPO_ROOT
    )

    full_binding = copy.deepcopy(precomputation_failure)
    full_binding.update(
        {
            "e0_declared_tree_algorithm": "relative_path_nul_file_sha256_bytes_v1",
            "e0_declared_tree_sha256": "c" * 64,
            "generic_asset_tree_algorithm": "canonical_json_file_manifest_v1",
            "generic_asset_tree_sha256": "d" * 64,
        }
    )
    gate.validate_a5r1_schema_artifact(full_binding, repo_root=gate.REPO_ROOT)

    for field in (
        "e0_declared_tree_algorithm",
        "e0_declared_tree_sha256",
        "generic_asset_tree_algorithm",
        "generic_asset_tree_sha256",
    ):
        partial = copy.deepcopy(full_binding)
        del partial[field]
        with pytest.raises(ValueError):
            gate.validate_a5r1_schema_artifact(partial, repo_root=gate.REPO_ROOT)


def test_gate_publication_is_atomic_and_no_overwrite(tmp_path: Path) -> None:
    value = {
        "schema_version": 8,
        "protocol_id": "fpct_e1_mechanism_audit_v8_a5r1_hash_domains",
        "artifact_type": "a5r1_hash_domain_synthetic_gate",
    }
    # Publication validates first; use a sentinel monkeypatch so this unit test
    # isolates the transactional no-overwrite behavior from the full gate.
    original = gate.validate_a5r1_schema_artifact
    try:
        gate.validate_a5r1_schema_artifact = lambda *_args, **_kwargs: None
        output = tmp_path / "gate.json"
        gate.publish_gate(value, output)
        assert json.loads(output.read_text(encoding="utf-8")) == value
        with pytest.raises(FileExistsError):
            gate.publish_gate(value, output)
    finally:
        gate.validate_a5r1_schema_artifact = original


def test_tracked_closure_contains_v8_and_immutable_v7_objects() -> None:
    tracked = set(gate.TRACKED_SOURCE_FILES)
    assert "FPCT_E1_A5R1_HASH_DOMAIN_AMENDMENT.md" in tracked
    assert gate.CONTRACT_RELATIVE.as_posix() in tracked
    assert gate.SCHEMA_RELATIVE.as_posix() in tracked
    assert set(gate.V7_OBJECT_SHA256) <= tracked
    assert gate.GATE_RELATIVE.as_posix() not in tracked
