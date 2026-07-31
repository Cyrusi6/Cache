from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from script.analysis import fpct_e1_a5r3_portable_publication_gate as gate


def test_v10_contract_binds_prospective_approval_and_v9_semantics() -> None:
    contract = gate._load_contract(gate.REPO_ROOT)
    assert contract["approval"]["approved"] is True
    assert contract["approval"]["user_response_verbatim"] == "可以"
    assert contract["immutable_predecessor"]["base_commit"] == gate.BASE_COMMIT
    assert contract["choice_semantics_version"] == 9
    assert contract["scope"]["sole_changed_dimension"] == (
        "CHOICE_AUDIT_DIRECTORY_PUBLICATION_PRIMITIVE"
    )


def test_a5r2_and_e765_closure_are_byte_immutable() -> None:
    assert gate._verify_immutable_predecessors(gate.REPO_ROOT) == (
        gate.A5R2_OBJECT_SHA256
    )


def test_scientific_semantics_bind_full_files_and_population_ast() -> None:
    evidence = gate.verify_scientific_semantics_unchanged(gate.REPO_ROOT)
    assert evidence["choice_semantics_version"] == 9
    assert evidence["scientific_file_sha256"] == gate.SCIENTIFIC_FILE_SHA256
    assert evidence["population_prepare_ast_sha256"] == dict(
        sorted(gate.SCIENTIFIC_AST_SHA256.items())
    )


def test_publication_static_closure_is_portable_and_no_replace() -> None:
    assert gate.verify_publication_primitive_static(gate.REPO_ROOT) == {
        "ordinary_rename_exactly_once": True,
        "renameat2_replace_copy_delete_absent": True,
        "exclusive_nofollow_owner_files": True,
        "runtime_inode_device_token_checked": True,
        "retained_claim_and_durable_receipt_supported": True,
    }


def test_target_filesystem_probe_is_synthetic_and_preserves_bytes(
    tmp_path: Path,
) -> None:
    observed = gate._probe_target_filesystem(tmp_path)
    assert observed["natural_data_accessed"] is False
    assert observed["ordinary_rename_succeeded"] is True
    assert observed["same_device"] is True
    assert observed["scratch_only"] is True
    assert len(observed["probe_evidence_sha256"]) == 64
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("choice_semantics_version",), 10),
        (("scope", "model_forward"), True),
        (("scope", "gpu_cuda_or_kubernetes"), True),
        (("scope", "e1_2"), True),
        (("immutable_predecessor", "resume_allowed"), True),
        (("unchanged_scientific_contract", "threshold_changed"), True),
        (("portable_publication", "renameat2_allowed"), True),
    ),
)
def test_schema_rejects_authority_semantics_and_primitive_drift(
    path: tuple[str, ...], replacement: object
) -> None:
    contract = gate._load_contract(gate.REPO_ROOT)
    broken = copy.deepcopy(contract)
    cursor = broken
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises(ValueError):
        gate.validate_a5r3_schema_artifact(broken, repo_root=gate.REPO_ROOT)


def test_synthetic_gate_source_closure_has_no_natural_or_later_stage_access() -> None:
    assert gate.verify_synthetic_only_gate_closure(gate.REPO_ROOT) == {
        "natural_e0_design_accessed": False,
        "tokenizer_or_alignment_accessed": False,
        "model_or_checkpoint_loaded": False,
        "model_forward_run": False,
        "gpu_cuda_or_kubernetes_used": False,
        "training": False,
        "e1_pilot_or_confirmatory_accessed": False,
        "e1_2_or_e1_3_authorized": False,
    }


def test_gate_publication_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = {
        "schema_version": 10,
        "choice_semantics_version": 9,
        "protocol_id": gate.PROTOCOL_ID,
        "artifact_type": "synthetic-test-gate",
    }
    monkeypatch.setattr(
        gate, "validate_a5r3_schema_artifact", lambda *_args, **_kwargs: None
    )
    output = tmp_path / "gate.json"
    gate.publish_gate(value, output)
    assert json.loads(output.read_text(encoding="utf-8")) == value
    with pytest.raises(FileExistsError):
        gate.publish_gate(value, output)


def test_gate_tracked_closure_contains_normative_and_publication_sources() -> None:
    tracked = set(gate.TRACKED_SOURCE_FILES)
    assert gate.AMENDMENT_RELATIVE.as_posix() in tracked
    assert gate.CONTRACT_RELATIVE.as_posix() in tracked
    assert gate.SCHEMA_RELATIVE.as_posix() in tracked
    assert gate.BLOCKED_CLOSURE_RELATIVE.as_posix() in tracked
    assert gate.PREPARE_RELATIVE.as_posix() in tracked
    assert set(gate.A5R2_OBJECT_SHA256) <= tracked
    assert set(gate.PUBLICATION_TEST_FILES) <= tracked
    assert gate.GATE_RELATIVE.as_posix() not in tracked
