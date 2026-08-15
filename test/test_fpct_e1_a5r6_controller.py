from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from script.experiment import fpct_e1_a5r6_controller as controller


def test_execution_identity_is_fresh_and_deterministic() -> None:
    sha = "1" * 40
    value = controller.execution_identity(sha)
    assert value["run_uid"] == "fpct-e1-a5r2-choice-cardinality-11111111-v1"
    assert value["run_root"].endswith("/fpct-e1-a5r2-11111111-v1")
    assert value["state_root"].endswith("/.fpct-e1-a5r6-controller-11111111-v1")
    assert "a47d52f8" not in value["run_root"]
    with pytest.raises(ValueError):
        controller.execution_identity("1" * 8)
    for predecessor in controller.PREDECESSOR_EXECUTION_SHAS:
        with pytest.raises(ValueError, match="fresh successor"):
            controller.execution_identity(predecessor)


def test_materialize_requires_current_pre_natural_gate_before_creating_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sha = "2" * 40
    monkeypatch.setattr(controller, "RUN_PARENT", tmp_path)
    monkeypatch.setattr(controller, "verify_repo", lambda _repo: sha)
    called = False

    def reject_gate(_repo: Path, *, replay_retained_evidence: bool = False):
        nonlocal called
        called = True
        assert replay_retained_evidence is True
        raise RuntimeError("pre-natural gate rejected")

    monkeypatch.setattr(controller, "verify_pre_natural_gate", reject_gate)
    with pytest.raises(RuntimeError, match="pre-natural gate rejected"):
        controller.materialize(tmp_path)
    assert called is True
    assert not Path(controller.execution_identity(sha)["state_root"]).exists()


def test_pre_natural_gate_rejects_dropped_tracked_or_immutable_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracked_path = tmp_path / "tracked.txt"
    tracked_path.write_bytes(b"tracked")
    immutable_path = tmp_path / "immutable.txt"
    immutable_path.write_bytes(b"immutable")
    monkeypatch.setattr(controller, "GATE_RELATIVE_PATH", Path("gate.json"))
    monkeypatch.setattr(
        controller, "GATE_REQUIRED_TRACKED_PATHS", ("tracked.txt",)
    )
    immutable = {"immutable.txt": controller.sha256_file(immutable_path)}
    monkeypatch.setattr(controller, "GATE_REQUIRED_IMMUTABLE_SHA256", immutable)
    forensic = {"forensic": "retained"}
    old_controller = {"controller": "retained"}
    monkeypatch.setattr(controller, "expected_forensic_closure", lambda: forensic)
    monkeypatch.setattr(
        controller, "expected_old_controller_evidence", lambda: old_controller
    )
    test_contract = {
        "required_nodes": list(controller.GATE_REQUIRED_TEST_NODES),
        "required_nodes_sha256": hashlib.sha256(
            ("\n".join(controller.GATE_REQUIRED_TEST_NODES) + "\n").encode()
        ).hexdigest(),
        "historical_deselected_nodes": list(
            controller.GATE_HISTORICAL_DESELECTED_NODES
        ),
        "historical_deselected_nodes_sha256": hashlib.sha256(
            ("\n".join(controller.GATE_HISTORICAL_DESELECTED_NODES) + "\n").encode()
        ).hexdigest(),
        "collection_summary": controller.GATE_EXPECTED_COLLECTION_SUMMARY,
    }
    value = {
        "schema_version": 13,
        "protocol_id": controller.PROTOCOL_ID,
        "artifact_type": "a5r6_pre_natural_synthetic_gate",
        "status": controller.GATE_STATUS,
        "base_commit": controller.BASE_COMMIT,
        "test_count": controller.GATE_EXPECTED_TEST_COUNT,
        "test_output_sha256": "a" * 64,
        "test_contract": test_contract,
        "tracked_files_sha256": {
            "tracked.txt": controller.sha256_file(tracked_path)
        },
        "immutable_predecessor_sha256": immutable,
        "failed_root_portable_closure": forensic,
        "failed_controller_terminal_evidence": old_controller,
        "checks": dict(controller.GATE_REQUIRED_CHECKS),
    }
    value["evidence_sha256"] = hashlib.sha256(
        controller._canonical_without_evidence(value)
    ).hexdigest()
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(value), encoding="utf-8")
    controller.verify_pre_natural_gate(tmp_path)

    for field in ("tracked_files_sha256", "immutable_predecessor_sha256"):
        tampered = json.loads(json.dumps(value))
        tampered[field] = {}
        tampered["evidence_sha256"] = hashlib.sha256(
            controller._canonical_without_evidence(tampered)
        ).hexdigest()
        gate_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(RuntimeError, match="source closure"):
            controller.verify_pre_natural_gate(tmp_path)


def test_tree_fingerprint_binds_path_bytes_mode_and_content(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    first = root / "a"
    first.write_bytes(b"alpha")
    first.chmod(0o600)
    nested = root / "nested"
    nested.mkdir()
    second = nested / "b"
    second.write_bytes(b"beta")
    second.chmod(0o640)
    observed = controller.tree_fingerprint(root)
    rows = [
        {"path": "a", "bytes": 5, "sha256": hashlib.sha256(b"alpha").hexdigest(), "mode": "0o600"},
        {"path": "nested/b", "bytes": 4, "sha256": hashlib.sha256(b"beta").hexdigest(), "mode": "0o640"},
    ]
    payload = "".join(f"{x['path']}\0{x['bytes']}\0{x['sha256']}\0{x['mode']}\n" for x in rows).encode()
    assert observed == {"file_count": 2, "total_bytes": 9, "tree_sha256": hashlib.sha256(payload).hexdigest()}
    second.write_bytes(b"BETA")
    assert controller.tree_fingerprint(root) != observed


def test_tree_fingerprint_rejects_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "target"
    target.write_text("x", encoding="utf-8")
    (root / "alias").symlink_to(target)
    with pytest.raises(RuntimeError, match="symlink"):
        controller.tree_fingerprint(root)


def test_filesystem_inventory_binds_empty_directories(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    before = controller.filesystem_inventory_fingerprint(root)
    (root / "empty").mkdir(mode=0o700)
    assert controller.filesystem_inventory_fingerprint(root) != before


def test_portable_inventory_excludes_numeric_uid_and_gid(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    child = root / "payload"
    child.write_bytes(b"portable")
    child.chmod(0o600)
    value = controller.filesystem_inventory_fingerprint(root)
    content = hashlib.sha256(b"portable").hexdigest()
    rows = (
        "d\x00.\x000o700\x00-\x00-\n"
        f"f\x00payload\x000o600\x008\x00{content}\n"
    ).encode()
    assert value == {
        "algorithm": controller.PORTABLE_INVENTORY_ALGORITHM,
        "entry_count": 2,
        "inventory_sha256": hashlib.sha256(rows).hexdigest(),
        "numeric_uid_in_digest": False,
        "numeric_gid_in_digest": False,
    }
    source = Path(controller.__file__).read_text(encoding="utf-8")
    function = source[source.index("def filesystem_inventory_fingerprint"):source.index("def environment_local_owner_mode_safety")]
    assert "st_uid" not in function
    assert "st_gid" not in function


def test_portable_digest_is_invariant_to_synthetic_numeric_owner_remap() -> None:
    base = [{
        "kind": "f",
        "relative_path": "payload",
        "mode": "0o600",
        "size": "4",
        "content": hashlib.sha256(b"safe").hexdigest(),
        "numeric_uid": 20007,
        "numeric_gid": 31000,
    }]
    remapped = [{**base[0], "numeric_uid": 65534, "numeric_gid": 65534}]
    assert controller.portable_inventory_digest(base) == controller.portable_inventory_digest(remapped)


def test_filesystem_portable_digest_and_local_safety_accept_uniform_gid_namespace_remap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    (root / "payload").write_bytes(b"safe")
    before = controller.filesystem_inventory_fingerprint(root)
    real_lstat = Path.lstat
    synthetic_gid = 424242

    def remapped_lstat(path: Path):
        values = list(real_lstat(path))
        values[5] = synthetic_gid
        return os.stat_result(values)

    monkeypatch.setattr(Path, "lstat", remapped_lstat)
    monkeypatch.setattr(controller.os, "getegid", lambda: synthetic_gid)
    monkeypatch.setattr(controller.os, "getgroups", lambda: [synthetic_gid])
    assert controller.filesystem_inventory_fingerprint(root) == before
    assert controller.environment_local_owner_mode_safety(root)[
        "all_entries_use_root_local_group"
    ] is True


def test_environment_local_owner_mode_safety_allows_read_only_world_bits_but_not_world_write(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o775)
    root.chmod(0o775)
    child = root / "payload"
    child.write_bytes(b"safe")
    child.chmod(0o664)
    value = controller.environment_local_owner_mode_safety(root)
    assert value["all_entries_owned_by_effective_user"] is True
    assert value["all_entries_use_root_local_group"] is True
    assert value["numeric_gid_frozen_across_namespaces"] is False
    child.chmod(0o666)
    with pytest.raises(RuntimeError, match="owner/mode safety failed"):
        controller.environment_local_owner_mode_safety(root)


def test_environment_local_safety_rejects_owner_and_unauthorized_root_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    (root / "payload").write_bytes(b"safe")
    monkeypatch.setattr(controller.os, "geteuid", lambda: root.stat().st_uid + 1)
    with pytest.raises(RuntimeError, match="owner/mode safety failed"):
        controller.environment_local_owner_mode_safety(root)
    monkeypatch.undo()
    impossible_group = root.stat().st_gid + 1000000
    monkeypatch.setattr(controller.os, "getegid", lambda: impossible_group)
    monkeypatch.setattr(controller.os, "getgroups", lambda: [])
    with pytest.raises(RuntimeError, match="owner/mode safety failed"):
        controller.environment_local_owner_mode_safety(root)


def test_environment_local_safety_rejects_mixed_root_local_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    child = root / "payload"
    child.write_bytes(b"safe")
    real_lstat = Path.lstat
    root_gid = real_lstat(root).st_gid

    def mixed_lstat(path: Path):
        values = list(real_lstat(path))
        if path == child:
            values[5] = root_gid + 1
        return os.stat_result(values)

    monkeypatch.setattr(Path, "lstat", mixed_lstat)
    with pytest.raises(RuntimeError, match="owner/mode safety failed"):
        controller.environment_local_owner_mode_safety(root)


def test_environment_local_safety_rejects_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    (root / "payload").write_bytes(b"safe")
    (root / "alias").symlink_to("payload")
    with pytest.raises(RuntimeError, match="owner/mode safety failed"):
        controller.environment_local_owner_mode_safety(root)


def test_environment_local_safety_rejects_special_file(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    os.mkfifo(root / "fifo", mode=0o600)
    with pytest.raises(RuntimeError, match="owner/mode safety failed"):
        controller.environment_local_owner_mode_safety(root)


@pytest.mark.parametrize("unsafe_mode", (0o4600, 0o2600, 0o1600))
def test_environment_local_safety_rejects_unsafe_special_file_bits(
    tmp_path: Path, unsafe_mode: int
) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    child = root / "payload"
    child.write_bytes(b"safe")
    child.chmod(unsafe_mode)
    with pytest.raises(RuntimeError, match="owner/mode safety failed"):
        controller.environment_local_owner_mode_safety(root)


def test_prepare_command_is_cpu_offline_scientific_target() -> None:
    lock = {
        **controller.execution_identity("2" * 40),
        "source_snapshot_receipt": "/snapshot/receipt.json",
    }
    command = controller._prepare_command(lock)
    joined = " ".join(command)
    assert "fpct_e1_prepare_input_lock.py" in joined
    assert "e0_design_input_lock.pt" in joined
    assert "e0_design_input_lock_manifest.json" in joined
    assert "fpct-e1-a5r2-choice-cardinality-22222222-v1" in joined


def test_status_never_relaunches_interrupted_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sha = "3" * 40
    monkeypatch.setattr(controller, "RUN_PARENT", tmp_path)
    monkeypatch.setattr(controller, "_git", lambda *_args: sha)
    identity = controller.execution_identity(sha)
    state_root = Path(identity["state_root"])
    state_root.mkdir(parents=True)
    for name in ("materialization_claim.json", "controller_lock.json", "launch_claim.json", "launch_receipt.json"):
        (state_root / name).write_text("{}", encoding="utf-8")
    lock = {**identity}
    receipt = {"pid": 42, "proc_starttime_ticks": 101, "proc_cmdline_sha256": "a" * 64}
    monkeypatch.setattr(controller, "_validate_controller_lock", lambda *_args, **_kwargs: lock)
    monkeypatch.setattr(
        controller, "_validate_launch_chain",
        lambda *_args, **_kwargs: {"claim": {}, "claim_sha256": "b" * 64, "worker_command": ["worker"]},
    )
    monkeypatch.setattr(controller, "_validate_launch_receipt", lambda *_args, **_kwargs: receipt)
    monkeypatch.setattr(
        controller, "_process_identity",
        lambda _pid: {"pid": 42, "proc_starttime_ticks": 102, "proc_cmdline_sha256": "a" * 64},
    )
    assert controller.status(tmp_path)["status"] == "INTERRUPTED_TERMINAL_NO_RELAUNCH"


def test_status_materialization_claim_without_lock_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sha = "4" * 40
    monkeypatch.setattr(controller, "RUN_PARENT", tmp_path)
    monkeypatch.setattr(controller, "_git", lambda *_args: sha)
    state_root = Path(controller.execution_identity(sha)["state_root"])
    state_root.mkdir(parents=True)
    (state_root / "materialization_claim.json").write_text("{}", encoding="utf-8")
    assert controller.status(tmp_path)["status"] == "MATERIALIZATION_FAILED_TERMINAL_NO_RELAUNCH"


def test_worker_cannot_bypass_launch_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sha = "5" * 40
    identity = controller.execution_identity(sha)
    lock_path = tmp_path / "controller_lock.json"
    lock_path.write_text("{}", encoding="utf-8")
    lock = {**identity, "source_snapshot_receipt": "/receipt", "prepare_environment": {}}
    monkeypatch.setattr(controller, "_validate_controller_lock", lambda *_args, **_kwargs: lock)
    called = False

    def forbidden_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("prepare subprocess must not start")

    monkeypatch.setattr(controller.subprocess, "run", forbidden_run)
    with pytest.raises(RuntimeError, match="launch claim is absent"):
        controller.worker(lock_path)
    assert called is False


def test_worker_start_claim_is_single_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sha = "6" * 40
    identity = controller.execution_identity(sha)
    lock_path = tmp_path / "controller_lock.json"
    lock_path.write_text("{}", encoding="utf-8")
    lock = {**identity, "source_snapshot_receipt": "/receipt", "prepare_environment": {}}
    launch = {"claim_sha256": "b" * 64}
    monkeypatch.setattr(controller, "_validate_controller_lock", lambda *_args, **_kwargs: lock)
    monkeypatch.setattr(controller, "_validate_launch_chain", lambda *_args, **_kwargs: launch)
    monkeypatch.setattr(
        controller, "_await_bound_launch_receipt",
        lambda *_args, **_kwargs: {
            "pid": os.getpid(), "proc_starttime_ticks": 1,
            "proc_cmdline_sha256": "c" * 64,
        },
    )
    start_path = tmp_path / "worker_start_claim.json"
    start_path.write_text("claimed", encoding="utf-8")
    start_path.chmod(0o600)
    called = False

    def forbidden_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("prepare subprocess must not start twice")

    monkeypatch.setattr(controller.subprocess, "run", forbidden_run)
    with pytest.raises(FileExistsError):
        controller.worker(lock_path)
    assert called is False


def test_worker_rejects_process_different_from_launch_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = {**controller.execution_identity("7" * 40)}
    launch = {"worker_command": ["worker"], "claim_sha256": "b" * 64}
    receipt = {
        "pid": os.getpid() + 100000,
        "proc_starttime_ticks": 1,
        "proc_cmdline_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        controller, "_validate_launch_receipt", lambda *_args, **_kwargs: receipt
    )
    monkeypatch.setattr(
        controller, "_process_identity",
        lambda _pid: {
            "pid": os.getpid(), "proc_starttime_ticks": 1,
            "proc_cmdline_sha256": "c" * 64,
        },
    )
    with pytest.raises(RuntimeError, match="non-launched worker"):
        controller._await_bound_launch_receipt(
            lock, tmp_path / "controller_lock.json", launch, timeout_seconds=0
        )


def test_exclusive_publication_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("retain", encoding="utf-8")
    link = tmp_path / "claim.json"
    link.symlink_to(target)
    with pytest.raises(FileExistsError):
        controller.publish_json_no_overwrite(link, {"status": "x"})
    assert target.read_text(encoding="utf-8") == "retain"


def test_deep_verifier_receipt_is_strict_and_replayable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir(mode=0o700)
    input_root = tmp_path / "input"
    input_root.mkdir(mode=0o700)
    lock_path = state_root / "controller_lock.json"
    controller.atomic_json(lock_path, {"lock": True})
    worker_result_path = state_root / "worker_result.json"
    controller.atomic_json(worker_result_path, {"result": True})
    artifact_paths = {
        "sidecar": input_root / "e0_design_input_lock.pt",
        "manifest": input_root / "e0_design_input_lock_manifest.json",
        "go": input_root / "A5R2_INPUT_LOCK_GO.json",
    }
    for name, path in artifact_paths.items():
        path.write_bytes(name.encode())
    monkeypatch.setattr(
        controller, "COMPLETED_INPUT_ROOT_ENTRIES",
        frozenset(path.name for path in artifact_paths.values()),
    )
    execution_sha = "9" * 40
    lock = {
        "execution_sha": execution_sha,
        "run_uid": "fpct-e1-a5r2-choice-cardinality-99999999-v1",
        "input_root": str(input_root),
        "source_snapshot_receipt_sha256": "b" * 64,
        "pre_natural_gate": {"sha256": "c" * 64},
    }
    value = {
        "schema_version": 1,
        "protocol_id": controller.PROTOCOL_ID,
        "artifact_type": "a5r6_deep_verifier_receipt",
        "status": "A5R6_INPUT_LOCK_GO_VERIFIED",
        "execution_sha": execution_sha,
        "run_uid": lock["run_uid"],
        "controller_lock_sha256": controller.sha256_file(lock_path),
        "worker_result_sha256": controller.sha256_file(worker_result_path),
        "source_snapshot_receipt_sha256": lock["source_snapshot_receipt_sha256"],
        "pre_natural_gate": lock["pre_natural_gate"],
        "deep_verifier": {
            "status": "A5R6_DEEP_COMPLETED_VERIFIER_GO",
            "execution_sha": execution_sha,
            "run_uid": lock["run_uid"],
            "manifest_sha256": controller.sha256_file(artifact_paths["manifest"]),
            "sidecar_sha256": controller.sha256_file(artifact_paths["sidecar"]),
            "manifest_status": "A5R2_INPUT_LOCK_GO",
        },
        "artifacts": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": controller.sha256_file(path),
            }
            for name, path in artifact_paths.items()
        },
        "verified_at_utc": "2026-08-15T00:00:00+00:00",
    }
    receipt_path = state_root / "deep_verifier_receipt.json"
    controller.atomic_json(receipt_path, value)
    assert controller._validate_deep_verifier_receipt(lock, lock_path) == value
    artifact_paths["go"].write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="artifact changed"):
        controller._validate_deep_verifier_receipt(lock, lock_path)


def test_controller_source_freezes_detached_one_shot_and_offline_environment() -> None:
    source = Path(controller.__file__).read_text(encoding="utf-8")
    assert "start_new_session=True" in source
    assert "publish_json_no_overwrite(state_root / \"launch_claim.json\"" in source
    assert '"CUDA_VISIBLE_DEVICES": ""' in source
    assert "never relaunch" in source
    assert controller.OLD_TREE_SHA256 in source
    assert "proc_starttime_ticks" in source
    assert "worker_start_claim.json" in source
    assert "_verify_completed_input_lock" in source
    assert "verify_pre_natural_gate(repo, replay_retained_evidence=True)" in source
    assert '"pre_natural_gate": gate_binding' in source
    assert "deep_verifier_receipt.json" in source
    assert "os.kill(" not in source
