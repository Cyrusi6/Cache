from __future__ import annotations

import copy
import inspect
import json
import os
import stat
from pathlib import Path

import pytest

from script.analysis import fpct_e1_a5r4_setgid_mode_gate as gate
from script.experiment import fpct_e1_prepare_input_lock as prepare


def _mkdir_private(path: Path) -> os.stat_result:
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path.lstat()


def test_v11_contract_binds_prospective_approval_and_v9_v10_boundaries() -> None:
    contract = gate._load_contract(gate.REPO_ROOT)
    assert contract["approval"]["user_response_verbatim"] == "可以"
    assert contract["immutable_predecessor"]["base_commit"] == gate.BASE_COMMIT
    assert contract["choice_semantics_version"] == 9
    assert contract["version_boundary"]["a5r3_publication_envelope_schema_version"] == 10
    assert contract["scope"]["sole_changed_dimension"] == (
        "OWNER_CREATED_STAGING_DIRECTORY_INHERITED_SETGID_MODE_PREDICATE"
    )


def test_immutable_a5r3_and_3263531_closure_are_byte_bound() -> None:
    observed = gate.verify_immutable_predecessors(gate.REPO_ROOT)
    assert observed["immutable_a5r3_objects"] == gate.A5R3_OBJECT_SHA256
    assert observed["immutable_a5r3_gate_evidence"] == gate.A5R3_GATE_EVIDENCE
    assert observed["blocked_failure_closure_sha256"] == gate.BLOCKED_CLOSURE_SHA256


def test_scientific_and_ast_closure_is_explicit() -> None:
    observed = gate.verify_ast_and_scientific_closure(gate.REPO_ROOT)
    assert observed["scientific"]["choice_semantics_version"] == 9
    assert observed["unchanged_a5r3_publication_ast_sha256"] == (
        gate.UNCHANGED_A5R3_PUBLICATION_AST_SHA256
    )
    assert set(observed["changed_mode_predicate_ast_sha256"]) == (
        gate.CHANGED_MODE_ENFORCEMENT_FUNCTIONS
    )


def test_exact_0700_directory_is_accepted(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent_metadata = _mkdir_private(parent)
    child = parent / "child"
    child.mkdir(mode=0o700)
    child.chmod(0o700)
    token = prepare._assert_a5r4_owner_created_directory_mode(
        child, role="synthetic 0700", parent_metadata=parent_metadata
    )
    assert stat.S_IMODE(child.lstat().st_mode) == 0o700
    assert token == prepare._publication_stat_identity_token(child.lstat())


def test_inherited_02700_and_rename_identity_are_accepted(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    _mkdir_private(parent)
    parent.chmod(0o2700)
    parent_metadata = parent.lstat()
    child = parent / "child"
    child.mkdir(mode=0o700)
    assert stat.S_IMODE(child.lstat().st_mode) == 0o2700
    token = prepare._assert_a5r4_owner_created_directory_mode(
        child, role="synthetic inherited setgid", parent_metadata=parent_metadata
    )
    final = parent / "final"
    os.rename(child, final)
    assert prepare._assert_a5r4_owner_created_directory_mode(
        final,
        role="synthetic renamed setgid",
        parent_metadata=parent_metadata,
        expected_token=token,
    ) == token


def test_02700_completed_publication_verifier_and_mode_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        prepare,
        "_validate_a5r3_publication_schema_artifact",
        lambda *_args, **_kwargs: None,
    )
    deployment = tmp_path / "deployment"
    _mkdir_private(deployment)
    deployment.chmod(0o2700)
    run_root = deployment / "run"
    run_root.mkdir(mode=0o700)
    assert stat.S_IMODE(run_root.lstat().st_mode) == 0o2700
    identity = {
        "execution_sha": "a" * 40,
        "run_uid": "fpct-e1-a5r2-choice-cardinality-aaaaaaaa-v1",
        "run_root": str(run_root),
        "source_snapshot_root": str(gate.REPO_ROOT),
    }
    staging = run_root / ".choice_audit.staging"
    final = run_root / "choice_audit"
    _claim, claim_payload, claim_token = (
        prepare._acquire_choice_audit_publication_claim(
            execution_identity=identity,
            staging_root=staging,
            audit_root=final,
        )
    )
    staging_token = prepare._create_choice_audit_staging_after_reduction(
        execution_identity=identity,
        staging_root=staging,
        audit_root=final,
        claim_payload=claim_payload,
        claim_owner_token=claim_token,
    )
    assert stat.S_IMODE(staging.lstat().st_mode) == 0o2700
    payloads = {
        prepare.CHOICE_AUDIT_LEDGER_NAME: prepare.canonical_json_bytes(
            {"synthetic_row": True}
        ),
        prepare.CHOICE_AUDIT_SUMMARY_NAME: prepare.canonical_json_bytes(
            {"synthetic_summary": True}
        ),
        prepare.CHOICE_AUDIT_LOCK_NAME: prepare.canonical_json_bytes(
            {"synthetic_lock": True}
        ),
    }
    for name, payload in payloads.items():
        (staging / name).write_bytes(payload)
    ledger = prepare._choice_audit_artifact_descriptor(
        staging / prepare.CHOICE_AUDIT_LEDGER_NAME, 326
    )
    summary = prepare._choice_audit_artifact_descriptor(
        staging / prepare.CHOICE_AUDIT_SUMMARY_NAME, 1
    )
    lock = prepare._choice_audit_artifact_descriptor(
        staging / prepare.CHOICE_AUDIT_LOCK_NAME, 1
    )
    receipt = prepare._publish_choice_audit_directory_with_claim(
        staging_root=staging,
        audit_root=final,
        execution_identity=identity,
        claim_payload=claim_payload,
        claim_owner_token=claim_token,
        staging_directory_token=staging_token,
        ledger_artifact=ledger,
        summary_artifact=summary,
        lock_artifact=lock,
    )
    assert receipt["status"] == "A5R3_CHOICE_AUDIT_PUBLICATION_GO"
    assert stat.S_IMODE(final.lstat().st_mode) == 0o2700
    assert prepare._verify_completed_choice_audit_publication(
        execution_identity=identity, audit_root=final
    ) == receipt

    final.chmod(0o3700)
    with pytest.raises(RuntimeError, match="forbidden special mode bit"):
        prepare._verify_completed_choice_audit_publication(
            execution_identity=identity, audit_root=final
        )


@pytest.mark.parametrize("unsafe_bit", (0o001, 0o002, 0o004, 0o010, 0o020, 0o040))
def test_every_group_or_other_permission_bit_is_rejected(
    tmp_path: Path, unsafe_bit: int
) -> None:
    parent = tmp_path / "parent"
    parent_metadata = _mkdir_private(parent)
    child = parent / "child"
    child.mkdir(mode=0o700)
    child.chmod(0o700 | unsafe_bit)
    with pytest.raises(RuntimeError, match="permissions are unsafe"):
        prepare._assert_a5r4_owner_created_directory_mode(
            child, role="unsafe permission", parent_metadata=parent_metadata
        )


@pytest.mark.parametrize("unsafe_mode", (0o600, 0o500, 0o4700, 0o1700, 0o6700, 0o3700))
def test_missing_owner_or_forbidden_special_mode_is_rejected(
    tmp_path: Path, unsafe_mode: int
) -> None:
    parent = tmp_path / "parent"
    parent_metadata = _mkdir_private(parent)
    child = parent / "child"
    child.mkdir(mode=0o700)
    child.chmod(unsafe_mode)
    message = "permissions are unsafe|forbidden special mode bit"
    with pytest.raises(RuntimeError, match=message):
        prepare._assert_a5r4_owner_created_directory_mode(
            child, role="unsafe special", parent_metadata=parent_metadata
        )


def test_setgid_without_setgid_parent_is_rejected(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent_metadata = _mkdir_private(parent)
    child = parent / "child"
    child.mkdir(mode=0o700)
    child.chmod(0o2700)
    with pytest.raises(RuntimeError, match="not inherited"):
        prepare._assert_a5r4_owner_created_directory_mode(
            child, role="forged setgid", parent_metadata=parent_metadata
        )


def test_parent_gid_or_device_mismatch_is_rejected(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    _mkdir_private(parent)
    parent.chmod(0o2700)
    parent_metadata = parent.lstat()
    child = parent / "child"
    child.mkdir(mode=0o700)
    child_metadata = child.lstat()

    wrong_gid_parent = os.stat_result(
        (
            parent_metadata.st_mode,
            parent_metadata.st_ino,
            parent_metadata.st_dev,
            parent_metadata.st_nlink,
            parent_metadata.st_uid,
            parent_metadata.st_gid + 1,
            parent_metadata.st_size,
            parent_metadata.st_atime,
            parent_metadata.st_mtime,
            parent_metadata.st_ctime,
        )
    )
    with pytest.raises(RuntimeError, match="not inherited"):
        prepare._assert_a5r4_owner_created_directory_mode(
            child, role="wrong parent gid", parent_metadata=wrong_gid_parent
        )

    wrong_device_parent = os.stat_result(
        (
            parent_metadata.st_mode,
            parent_metadata.st_ino,
            child_metadata.st_dev + 1,
            parent_metadata.st_nlink,
            parent_metadata.st_uid,
            parent_metadata.st_gid,
            parent_metadata.st_size,
            parent_metadata.st_atime,
            parent_metadata.st_mtime,
            parent_metadata.st_ctime,
        )
    )
    with pytest.raises(RuntimeError, match="device differs"):
        prepare._assert_a5r4_owner_created_directory_mode(
            child, role="wrong parent device", parent_metadata=wrong_device_parent
        )


def test_wrong_child_or_setgid_parent_owner_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    _mkdir_private(parent)
    parent.chmod(0o2700)
    parent_metadata = parent.lstat()
    child = parent / "child"
    child.mkdir(mode=0o700)
    child_metadata = child.lstat()
    original_lstat = Path.lstat

    wrong_child = os.stat_result(
        (
            child_metadata.st_mode,
            child_metadata.st_ino,
            child_metadata.st_dev,
            child_metadata.st_nlink,
            child_metadata.st_uid + 1,
            child_metadata.st_gid,
            child_metadata.st_size,
            child_metadata.st_atime,
            child_metadata.st_mtime,
            child_metadata.st_ctime,
        )
    )

    def fake_lstat(path: Path) -> os.stat_result:
        return wrong_child if path == child else original_lstat(path)

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    with pytest.raises(RuntimeError, match="owner differs"):
        prepare._assert_a5r4_owner_created_directory_mode(
            child, role="wrong child owner", parent_metadata=parent_metadata
        )
    monkeypatch.undo()

    wrong_parent_owner = os.stat_result(
        (
            parent_metadata.st_mode,
            parent_metadata.st_ino,
            parent_metadata.st_dev,
            parent_metadata.st_nlink,
            parent_metadata.st_uid + 1,
            parent_metadata.st_gid,
            parent_metadata.st_size,
            parent_metadata.st_atime,
            parent_metadata.st_mtime,
            parent_metadata.st_ctime,
        )
    )
    with pytest.raises(RuntimeError, match="not inherited"):
        prepare._assert_a5r4_owner_created_directory_mode(
            child,
            role="wrong setgid parent owner",
            parent_metadata=wrong_parent_owner,
        )


def test_symlink_regular_file_and_identity_substitution_are_rejected(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent_metadata = _mkdir_private(parent)
    real = parent / "real"
    real.mkdir(mode=0o700)
    link = parent / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="not a real directory"):
        prepare._assert_a5r4_owner_created_directory_mode(
            link, role="symlink", parent_metadata=parent_metadata
        )
    regular = parent / "regular"
    regular.write_text("x", encoding="utf-8")
    regular.chmod(0o700)
    with pytest.raises(RuntimeError, match="not a real directory"):
        prepare._assert_a5r4_owner_created_directory_mode(
            regular, role="regular", parent_metadata=parent_metadata
        )
    token = prepare._assert_a5r4_owner_created_directory_mode(
        real, role="real", parent_metadata=parent_metadata
    )
    wrong = list(token)
    wrong[1] += 1
    with pytest.raises(RuntimeError, match="identity token changed"):
        prepare._assert_a5r4_owner_created_directory_mode(
            real,
            role="substituted",
            parent_metadata=parent_metadata,
            expected_token=tuple(wrong),
        )


def test_a5r3_claim_receipt_envelope_and_publication_primitive_are_unchanged() -> None:
    assert prepare.CHOICE_AUDIT_PUBLICATION_CLAIM_PROTOCOL_ID == (
        "fpct_e1_mechanism_audit_v10_a5r3_portable_publication"
    )
    assert '"schema_version": 10' in inspect.getsource(
        prepare._choice_audit_publication_claim_payload
    )
    assert '"schema_version": 10' in inspect.getsource(
        prepare._choice_audit_publication_receipt_payload
    )
    assert gate.verify_mode_predicate_static(gate.REPO_ROOT) == {
        "exact_0700_accepted": True,
        "inherited_02700_accepted": True,
        "unsafe_mode_matrix_rejected": True,
        "parent_setgid_and_gid_binding_enforced": True,
        "rename_identity_and_mode_preserved": True,
        "completed_verifier_mode_predicate_pass": True,
        "chmod_mode_normalization_retry_and_fallback_absent": True,
    }


def test_active_loader_consumes_static_a5r4_and_current_a5r7_gate() -> None:
    source = inspect.getsource(prepare._load_active_a5_synthetic_gate)
    assert "verify_a5r4_gate" not in source
    assert "verify_a5r3_gate" not in source
    assert "validate_a5r4_schema_artifact" in source
    assert "verify_current_gate" in source
    assert "A5R4_SYNTHETIC_GATE_SHA256" in source


def test_synthetic_gate_source_has_zero_natural_or_later_stage_access() -> None:
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


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("choice_semantics_version",), 10),
        (("scope", "model_forward"), True),
        (("scope", "e1_2"), True),
        (("immutable_predecessor", "resume_allowed"), True),
        (("directory_mode_predicate", "setuid_allowed"), True),
        (("directory_mode_predicate", "accepted_complete_permission_modes_octal"), ["00700", "07700"]),
        (("unchanged_a5r3_publication", "rename_call_count"), 2),
    ),
)
def test_schema_rejects_authority_semantics_and_mode_drift(
    path: tuple[str, ...], replacement: object
) -> None:
    contract = gate._load_contract(gate.REPO_ROOT)
    broken = copy.deepcopy(contract)
    cursor = broken
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises(ValueError):
        gate.validate_a5r4_schema_artifact(broken, repo_root=gate.REPO_ROOT)


def test_gate_publication_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = {
        "schema_version": 11,
        "choice_semantics_version": 9,
        "protocol_id": gate.PROTOCOL_ID,
        "artifact_type": gate.ARTIFACT_TYPE,
    }
    monkeypatch.setattr(
        gate, "validate_a5r4_schema_artifact", lambda *_args, **_kwargs: None
    )
    output = tmp_path / "gate.json"
    gate.publish_gate(value, output)
    assert json.loads(output.read_text(encoding="utf-8")) == value
    with pytest.raises(FileExistsError):
        gate.publish_gate(value, output)


def test_synthetic_probe_cleanup_refuses_replaced_path(tmp_path: Path) -> None:
    probe = tmp_path / "probe"
    probe.mkdir(mode=0o700)
    token = gate._probe_directory_token(probe.lstat())
    gate._remove_owned_synthetic_probe(probe, token)
    assert not probe.exists()

    probe.mkdir(mode=0o700)
    stale_token = gate._probe_directory_token(probe.lstat())
    original = tmp_path / "original"
    probe.rename(original)
    probe.mkdir(mode=0o700)
    with pytest.raises(RuntimeError, match="replacement is retained"):
        gate._remove_owned_synthetic_probe(probe, stale_token)
    assert probe.is_dir()


def test_gate_tracked_closure_contains_all_normative_and_execution_sources() -> None:
    tracked = set(gate.TRACKED_SOURCE_FILES)
    assert gate.AMENDMENT_RELATIVE.as_posix() in tracked
    assert gate.CONTRACT_RELATIVE.as_posix() in tracked
    assert gate.SCHEMA_RELATIVE.as_posix() in tracked
    assert gate.BLOCKED_CLOSURE_RELATIVE.as_posix() in tracked
    assert gate.PREPARE_RELATIVE.as_posix() in tracked
    assert set(gate.A5R3_OBJECT_SHA256) <= tracked
    assert set(gate.PUBLICATION_TEST_FILES) <= tracked
    assert gate.GATE_RELATIVE.as_posix() not in tracked
