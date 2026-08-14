from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from script.experiment import fpct_e1_a5r5_controller as controller


def test_execution_identity_is_fresh_and_deterministic() -> None:
    sha = "1" * 40
    value = controller.execution_identity(sha)
    assert value["run_uid"] == "fpct-e1-a5r2-choice-cardinality-11111111-v1"
    assert value["run_root"].endswith("/fpct-e1-a5r2-11111111-v1")
    assert "a47d52f8" not in value["run_root"]
    with pytest.raises(ValueError):
        controller.execution_identity("1" * 8)


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
    assert "os.kill(" not in source
