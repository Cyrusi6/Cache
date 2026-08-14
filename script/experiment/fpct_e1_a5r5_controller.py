#!/usr/bin/env python3
"""One-shot detached controller for the FPCT-E1 A5R5 fresh-root recovery."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any, Mapping, Sequence


PROTOCOL_ID = "fpct_e1_mechanism_audit_v12_a5r5_fresh_root_recovery"
BRANCH = "research/fpct-e1-mechanism-audit"
OLD_ROOT = Path("/netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-a47d52f8-v1")
OLD_TREE_SHA256 = "302762365c964e93a3f3fe3ad063447998b51207f6f9ab57b0594c4132184187"
OLD_FILE_COUNT = 708
OLD_TOTAL_BYTES = 39278548
OLD_INVENTORY_ENTRY_COUNT = 812
OLD_INVENTORY_SHA256 = "c480ecfa6b508fa1cad028ae54dc18b64f18295e0ce4239a598386b36131ebdc"
RUN_PARENT = Path("/netdisk/lijunsi/fpct-e1")
PYTHON = Path("/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python3.10")
E0_DATA_ROOT = Path("/netdisk/lijunsi/fpct-e0/fpct-e0-20260722-v1/dev_data")
EXECUTION_RE = re.compile(r"^[0-9a-f]{40}$")


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def publish_json_no_overwrite(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise RuntimeError("exclusive controller artifact identity/mode is unsafe")
    except BaseException:
        os.close(descriptor)
        raise
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        parent_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        # Retain a partial exclusive claim as terminal forensic evidence.
        raise


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tree_fingerprint(root: Path) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"forensic root is not a real directory: {root}")
    rows: list[dict[str, Any]] = []
    total = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"forensic root contains a symlink: {path}")
        if not stat.S_ISREG(mode):
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        rows.append({
            "path": relative,
            "bytes": size,
            "sha256": sha256_file(path),
            "mode": oct(stat.S_IMODE(mode)),
        })
        total += size
    payload = "".join(
        f"{row['path']}\0{row['bytes']}\0{row['sha256']}\0{row['mode']}\n"
        for row in rows
    ).encode()
    return {
        "file_count": len(rows),
        "total_bytes": total,
        "tree_sha256": hashlib.sha256(payload).hexdigest(),
    }


def filesystem_inventory_fingerprint(root: Path) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"inventory root is not a real directory: {root}")
    rows: list[str] = []
    paths = [root, *sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())]
    for path in paths:
        metadata = path.lstat()
        relative = "." if path == root else path.relative_to(root).as_posix()
        if stat.S_ISDIR(metadata.st_mode):
            kind = "d"
            content = "-"
            size = "-"
        elif stat.S_ISREG(metadata.st_mode):
            kind = "f"
            content = sha256_file(path)
            size = str(metadata.st_size)
        else:
            raise RuntimeError(f"forensic inventory contains an unsupported entry: {path}")
        rows.append(
            f"{kind}\0{relative}\0{oct(stat.S_IMODE(metadata.st_mode))}\0"
            f"{metadata.st_uid}\0{metadata.st_gid}\0{size}\0{content}\n"
        )
    payload = "".join(rows).encode()
    return {"entry_count": len(rows), "inventory_sha256": hashlib.sha256(payload).hexdigest()}


def verify_old_forensic_root() -> dict[str, Any]:
    observed = tree_fingerprint(OLD_ROOT)
    expected = {
        "file_count": OLD_FILE_COUNT,
        "total_bytes": OLD_TOTAL_BYTES,
        "tree_sha256": OLD_TREE_SHA256,
    }
    if observed != expected:
        raise RuntimeError(f"immutable a47d52f8 forensic root changed: {observed}")
    inventory = filesystem_inventory_fingerprint(OLD_ROOT)
    if inventory != {
        "entry_count": OLD_INVENTORY_ENTRY_COUNT,
        "inventory_sha256": OLD_INVENTORY_SHA256,
    }:
        raise RuntimeError(f"immutable a47d52f8 directory inventory changed: {inventory}")
    input_root = OLD_ROOT / "input_lock"
    required = [
        OLD_ROOT / "choice_audit_publication_receipt.json",
        OLD_ROOT / "choice_audit/choice_cardinality_audit_lock.json",
        input_root / "input_geometry_receipt.json",
    ]
    if not all(path.is_file() and not path.is_symlink() for path in required):
        raise RuntimeError("a47d52f8 forensic closure lost a required retained artifact")
    forbidden = [
        input_root / "e0_design_input_lock.pt",
        input_root / "e0_design_input_lock_manifest.json",
        input_root / "A5R2_INPUT_LOCK_GO.json",
        input_root / "A5R2_INPUT_LOCK_BLOCKED.json",
    ]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("a47d52f8 forensic root acquired a terminal/input-lock artifact")
    return {**observed, **inventory}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def verify_repo(repo: Path) -> str:
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    head = _git(repo, "rev-parse", "HEAD")
    upstream = _git(repo, "rev-parse", "@{upstream}")
    dirty = _git(repo, "status", "--porcelain=v1")
    if branch != BRANCH or head != upstream or dirty or not EXECUTION_RE.fullmatch(head):
        raise RuntimeError(
            f"repository identity is unsafe: branch={branch} head={head} upstream={upstream} dirty={bool(dirty)}"
        )
    return head


def execution_identity(execution_sha: str) -> dict[str, str]:
    if not EXECUTION_RE.fullmatch(execution_sha):
        raise ValueError("execution SHA must be a full lowercase SHA1")
    prefix = execution_sha[:8]
    uid = f"fpct-e1-a5r2-choice-cardinality-{prefix}-v1"
    root = RUN_PARENT / f"fpct-e1-a5r2-{prefix}-v1"
    return {
        "execution_sha": execution_sha,
        "execution_prefix": prefix,
        "run_uid": uid,
        "run_root": str(root),
        "snapshot_root": str(root / "source_snapshot"),
        "input_root": str(root / "input_lock"),
        "state_root": str(RUN_PARENT / f".fpct-e1-a5r5-controller-{prefix}-v1"),
    }


def _read_canonical_controller_json(path: Path, role: str) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"{role} is absent") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or path.is_symlink()
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise RuntimeError(f"{role} identity/mode is unsafe")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{role} is not valid JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise RuntimeError(f"{role} is not canonical JSON")
    return value


def _prepare_environment_projection() -> dict[str, str]:
    allowed = (
        "HOME", "PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "CONDA_PREFIX",
        "C2C_MODEL_ROOT", "C2C_DATA_ROOT", "HF_HOME", "XDG_CACHE_HOME",
    )
    environment = {key: os.environ[key] for key in allowed if key in os.environ}
    environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TMPDIR": "/tmp",
    })
    return environment


def _worker_command(lock: Mapping[str, Any], lock_path: Path) -> list[str]:
    snapshot = Path(str(lock["snapshot_root"]))
    return [
        str(PYTHON), "-I", str(snapshot / "script/runtime/fpct_bootstrap.py"),
        "--repo-root", str(snapshot),
        "--target", str(snapshot / "script/experiment/fpct_e1_a5r5_controller.py"),
        "--", "worker", "--lock", str(lock_path),
    ]


def _validate_controller_lock(
    lock_path: Path, *, expected_execution_sha: str | None = None
) -> dict[str, Any]:
    lock = _read_canonical_controller_json(lock_path, "A5R5 controller lock")
    execution_sha = str(lock.get("execution_sha", ""))
    if expected_execution_sha is not None and execution_sha != expected_execution_sha:
        raise RuntimeError("A5R5 controller lock execution SHA differs from current HEAD")
    identity = execution_identity(execution_sha)
    expected_path = Path(identity["state_root"]) / "controller_lock.json"
    if lock_path.absolute() != expected_path.absolute():
        raise RuntimeError("A5R5 controller lock path is not canonical")
    if any(lock.get(key) != value for key, value in identity.items()):
        raise RuntimeError("A5R5 controller lock derived identity changed")
    required = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "a5r5_controller_lock",
        "status": "READY_TO_LAUNCH_ONCE",
        "launch_count": 0,
        "resume_allowed": False,
    }
    if any(lock.get(key) != value for key, value in required.items()):
        raise RuntimeError("A5R5 controller lock schema/status changed")
    if lock.get("forensic_closure") != verify_old_forensic_root():
        raise RuntimeError("A5R5 interrupted-root forensic closure changed")
    snapshot = Path(identity["snapshot_root"])
    receipt = snapshot / ".fpct_e1_source_snapshot_receipt.json"
    if (
        lock.get("source_snapshot_receipt") != str(receipt)
        or not receipt.is_file()
        or receipt.is_symlink()
        or lock.get("source_snapshot_receipt_sha256") != sha256_file(receipt)
    ):
        raise RuntimeError("A5R5 controller lock source snapshot receipt changed")
    claim_path = Path(identity["state_root"]) / "materialization_claim.json"
    claim = _read_canonical_controller_json(claim_path, "A5R5 materialization claim")
    if (
        claim.get("protocol_id") != PROTOCOL_ID
        or claim.get("artifact_type") != "a5r5_materialization_claim"
        or any(claim.get(key) != value for key, value in identity.items())
        or lock.get("materialization_claim_sha256") != sha256_file(claim_path)
    ):
        raise RuntimeError("A5R5 materialization claim binding changed")
    environment = lock.get("prepare_environment")
    if (
        not isinstance(environment, dict)
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in environment.items())
        or environment.get("CUDA_VISIBLE_DEVICES") != ""
        or environment.get("HF_HUB_OFFLINE") != "1"
        or environment.get("TRANSFORMERS_OFFLINE") != "1"
        or environment.get("HF_DATASETS_OFFLINE") != "1"
        or "PYTHONPATH" in environment
    ):
        raise RuntimeError("A5R5 frozen prepare environment changed")
    return lock


def _process_identity(pid: int) -> dict[str, Any]:
    stat_path = Path(f"/proc/{pid}/stat")
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    stat_text = stat_path.read_text(encoding="utf-8")
    close = stat_text.rfind(")")
    if close < 0:
        raise RuntimeError("worker /proc stat has invalid format")
    fields = stat_text[close + 2 :].split()
    if len(fields) <= 19:
        raise RuntimeError("worker /proc stat lacks starttime")
    cmdline = cmdline_path.read_bytes()
    return {
        "pid": pid,
        "proc_starttime_ticks": int(fields[19]),
        "proc_cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
    }


def _validate_launch_chain(lock: Mapping[str, Any], lock_path: Path) -> dict[str, Any]:
    state_root = lock_path.parent
    lock_sha = sha256_file(lock_path)
    claim_path = state_root / "launch_claim.json"
    claim = _read_canonical_controller_json(claim_path, "A5R5 launch claim")
    worker_command = _worker_command(lock, lock_path)
    worker_command_sha = hashlib.sha256("\0".join(worker_command).encode()).hexdigest()
    if (
        claim.get("protocol_id") != PROTOCOL_ID
        or claim.get("artifact_type") != "a5r5_launch_claim"
        or claim.get("execution_sha") != lock["execution_sha"]
        or claim.get("run_uid") != lock["run_uid"]
        or claim.get("controller_lock_sha256") != lock_sha
        or claim.get("command_sha256") != worker_command_sha
        or claim.get("launch_count") != 1
        or claim.get("resume_allowed") is not False
    ):
        raise RuntimeError("A5R5 launch claim binding changed")
    return {"claim": claim, "claim_sha256": sha256_file(claim_path), "worker_command": worker_command}


def _validate_launch_receipt(
    lock: Mapping[str, Any], lock_path: Path, launch: Mapping[str, Any]
) -> dict[str, Any]:
    receipt_path = lock_path.parent / "launch_receipt.json"
    receipt = _read_canonical_controller_json(receipt_path, "A5R5 launch receipt")
    required = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "a5r5_launch_receipt",
        "execution_sha": lock["execution_sha"],
        "run_uid": lock["run_uid"],
        "command_sha256": hashlib.sha256(
            "\0".join(launch["worker_command"]).encode()
        ).hexdigest(),
        "controller_lock_sha256": sha256_file(lock_path),
        "launch_claim_sha256": launch["claim_sha256"],
        "launch_count": 1,
        "resume_allowed": False,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise RuntimeError("A5R5 launch receipt binding changed")
    if (
        not isinstance(receipt.get("pid"), int)
        or receipt["pid"] <= 0
        or receipt.get("proc_starttime_ticks") is None
        or not isinstance(receipt.get("proc_starttime_ticks"), int)
        or not isinstance(receipt.get("proc_cmdline_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", receipt["proc_cmdline_sha256"]) is None
        or receipt.get("log_path") != str(lock_path.parent / "worker.log")
    ):
        raise RuntimeError("A5R5 launch receipt process identity is incomplete")
    return receipt


def _validate_worker_start(
    lock: Mapping[str, Any], lock_path: Path, launch: Mapping[str, Any],
    launch_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = lock_path.parent / "worker_start_claim.json"
    value = _read_canonical_controller_json(path, "A5R5 worker-start claim")
    command_sha = hashlib.sha256("\0".join(_prepare_command(lock)).encode()).hexdigest()
    required = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "a5r5_worker_start_claim",
        "execution_sha": lock["execution_sha"],
        "run_uid": lock["run_uid"],
        "controller_lock_sha256": sha256_file(lock_path),
        "launch_claim_sha256": launch["claim_sha256"],
        "prepare_command_sha256": command_sha,
        "resume_allowed": False,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise RuntimeError("A5R5 worker-start claim binding changed")
    receipt = (
        dict(launch_receipt)
        if launch_receipt is not None
        else _validate_launch_receipt(lock, lock_path, launch)
    )
    if any(
        value.get(key) != receipt.get(key)
        for key in ("pid", "proc_starttime_ticks", "proc_cmdline_sha256")
    ):
        raise RuntimeError("A5R5 worker-start process differs from launched process")
    return value


def _validate_worker_result(
    lock: Mapping[str, Any], lock_path: Path, launch: Mapping[str, Any],
    launch_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    start_path = lock_path.parent / "worker_start_claim.json"
    _validate_worker_start(lock, lock_path, launch, launch_receipt)
    path = lock_path.parent / "worker_result.json"
    value = _read_canonical_controller_json(path, "A5R5 worker result")
    command_sha = hashlib.sha256("\0".join(_prepare_command(lock)).encode()).hexdigest()
    expected_status = (
        "WORKER_EXIT_ZERO" if value.get("exit_code") == 0 else "WORKER_TERMINAL_FAILURE"
    )
    required = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "a5r5_worker_result",
        "execution_sha": lock["execution_sha"],
        "run_uid": lock["run_uid"],
        "command_sha256": command_sha,
        "controller_lock_sha256": sha256_file(lock_path),
        "launch_claim_sha256": launch["claim_sha256"],
        "worker_start_claim_sha256": sha256_file(start_path),
        "status": expected_status,
        "resume_allowed": False,
    }
    if (
        any(value.get(key) != expected for key, expected in required.items())
        or type(value.get("exit_code")) is not int
    ):
        raise RuntimeError("A5R5 worker-result binding changed")
    return value


def _await_bound_launch_receipt(
    lock: Mapping[str, Any], lock_path: Path, launch: Mapping[str, Any],
    *, timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            receipt = _validate_launch_receipt(lock, lock_path, launch)
        except RuntimeError as error:
            if "is absent" not in str(error) or time.monotonic() >= deadline:
                raise
            time.sleep(0.01)
            continue
        live = _process_identity(os.getpid())
        if any(
            live[key] != receipt[key]
            for key in ("pid", "proc_starttime_ticks", "proc_cmdline_sha256")
        ):
            raise RuntimeError("A5R5 direct/non-launched worker entry is forbidden")
        return receipt


def _safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r") as handle:
        destination_resolved = destination.resolve()
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if destination_resolved not in (target, *target.parents):
                raise RuntimeError("git archive contains a path traversal")
            if member.issym() or member.islnk():
                raise RuntimeError("A5R5 source snapshot forbids archive links")
        handle.extractall(destination)


def materialize(repo: Path) -> dict[str, Any]:
    forensic = verify_old_forensic_root()
    execution_sha = verify_repo(repo)
    identity = execution_identity(execution_sha)
    run_root = Path(identity["run_root"])
    snapshot = Path(identity["snapshot_root"])
    input_root = Path(identity["input_root"])
    state_root = Path(identity["state_root"])
    if run_root.exists() or state_root.exists():
        raise RuntimeError("A5R5 successor root/state already exists; never reuse it")
    state_root.mkdir(mode=0o700, parents=True)
    materialization_claim_path = state_root / "materialization_claim.json"
    materialization_claim = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "a5r5_materialization_claim",
        **identity,
        "claimed_at_utc": utc_now(),
        "resume_allowed": False,
    }
    publish_json_no_overwrite(materialization_claim_path, materialization_claim)
    run_root.mkdir(mode=0o700)
    snapshot.mkdir(mode=0o700)
    input_root.mkdir(mode=0o700)
    archive = Path("/tmp") / f"{identity['run_uid']}.tar"
    if archive.exists():
        raise RuntimeError("A5R5 source archive path already exists")
    subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", f"--output={archive}", execution_sha],
        check=True,
    )
    try:
        _safe_extract(archive, snapshot)
    finally:
        archive.unlink(missing_ok=True)
    receipt = snapshot / ".fpct_e1_source_snapshot_receipt.json"
    lock_script = repo / "script/experiment/fpct_e1_source_snapshot_lock.py"
    for mode in ("create", "verify"):
        command = [str(PYTHON), str(lock_script), mode]
        if mode == "create":
            command += ["--repo", str(repo), "--execution-sha", execution_sha, "--snapshot", str(snapshot), "--output", str(receipt)]
        else:
            command += ["--repo", str(repo), "--execution-sha", execution_sha, "--snapshot", str(snapshot), "--receipt", str(receipt)]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(
        [str(PYTHON), str(lock_script), "verify-mounted", "--execution-sha", execution_sha,
         "--snapshot", str(snapshot), "--receipt", str(receipt)],
        check=True, stdout=subprocess.DEVNULL,
    )
    lock = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "a5r5_controller_lock",
        "status": "READY_TO_LAUNCH_ONCE",
        **identity,
        "source_snapshot_receipt": str(receipt),
        "source_snapshot_receipt_sha256": sha256_file(receipt),
        "materialization_claim_sha256": sha256_file(materialization_claim_path),
        "forensic_closure": forensic,
        "prepare_environment": _prepare_environment_projection(),
        "launch_count": 0,
        "resume_allowed": False,
        "created_at_utc": utc_now(),
    }
    publish_json_no_overwrite(state_root / "controller_lock.json", lock)
    return lock


def _prepare_command(lock: Mapping[str, Any]) -> list[str]:
    snapshot = Path(str(lock["snapshot_root"]))
    input_root = Path(str(lock["input_root"]))
    return [
        str(PYTHON), "-I", str(snapshot / "script/runtime/fpct_bootstrap.py"),
        "--repo-root", str(snapshot),
        "--target", str(snapshot / "script/experiment/fpct_e1_prepare_input_lock.py"),
        "--", "--repo-root", str(snapshot), "--e0-data-root", str(E0_DATA_ROOT),
        "--output-sidecar", str(input_root / "e0_design_input_lock.pt"),
        "--output-manifest", str(input_root / "e0_design_input_lock_manifest.json"),
        "--execution-sha", str(lock["execution_sha"]),
        "--source-snapshot-root", str(snapshot),
        "--source-snapshot-receipt", str(lock["source_snapshot_receipt"]),
        "--run-uid", str(lock["run_uid"]), "--run-root", str(lock["run_root"]),
    ]


def worker(lock_path: Path) -> int:
    lock = _validate_controller_lock(lock_path)
    state_root = lock_path.parent
    launch = _validate_launch_chain(lock, lock_path)
    launch_receipt = _await_bound_launch_receipt(lock, lock_path, launch)
    command = _prepare_command(lock)
    command_sha = hashlib.sha256("\0".join(command).encode()).hexdigest()
    worker_start_path = state_root / "worker_start_claim.json"
    worker_start = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "a5r5_worker_start_claim",
        "execution_sha": lock["execution_sha"],
        "run_uid": lock["run_uid"],
        "controller_lock_sha256": sha256_file(lock_path),
        "launch_claim_sha256": launch["claim_sha256"],
        "prepare_command_sha256": command_sha,
        **{
            key: launch_receipt[key]
            for key in ("pid", "proc_starttime_ticks", "proc_cmdline_sha256")
        },
        "started_at_utc": utc_now(),
        "resume_allowed": False,
    }
    publish_json_no_overwrite(worker_start_path, worker_start)
    environment = dict(lock["prepare_environment"])
    result = subprocess.run(command, cwd=lock["snapshot_root"], env=environment, check=False)
    payload = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "artifact_type": "a5r5_worker_result", "execution_sha": lock["execution_sha"],
        "run_uid": lock["run_uid"], "command_sha256": command_sha,
        "controller_lock_sha256": sha256_file(lock_path),
        "launch_claim_sha256": launch["claim_sha256"],
        "worker_start_claim_sha256": sha256_file(worker_start_path),
        "exit_code": result.returncode,
        "status": "WORKER_EXIT_ZERO" if result.returncode == 0 else "WORKER_TERMINAL_FAILURE",
        "resume_allowed": False,
        "finished_at_utc": utc_now(),
    }
    publish_json_no_overwrite(state_root / "worker_result.json", payload)
    return result.returncode


def launch(repo: Path) -> dict[str, Any]:
    execution_sha = verify_repo(repo)
    identity = execution_identity(execution_sha)
    state_root = Path(identity["state_root"])
    lock_path = state_root / "controller_lock.json"
    lock = _validate_controller_lock(lock_path, expected_execution_sha=execution_sha)
    if any((state_root / name).exists() for name in (
        "launch_claim.json", "launch_receipt.json", "worker_start_claim.json", "worker_result.json"
    )):
        raise RuntimeError("A5R5 worker was already launched; never relaunch")
    snapshot = Path(lock["snapshot_root"])
    source_lock = repo / "script/experiment/fpct_e1_source_snapshot_lock.py"
    subprocess.run(
        [str(PYTHON), str(source_lock), "verify-mounted", "--execution-sha", execution_sha,
         "--snapshot", str(snapshot), "--receipt", str(lock["source_snapshot_receipt"])],
        check=True, stdout=subprocess.DEVNULL,
    )
    command = _worker_command(lock, lock_path)
    command_sha = hashlib.sha256("\0".join(command).encode()).hexdigest()
    claim = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "artifact_type": "a5r5_launch_claim", "execution_sha": execution_sha,
        "run_uid": lock["run_uid"], "command_sha256": command_sha,
        "controller_lock_sha256": sha256_file(lock_path),
        "claimed_at_utc": utc_now(), "launch_count": 1, "resume_allowed": False,
    }
    publish_json_no_overwrite(state_root / "launch_claim.json", claim)
    log_path = state_root / "worker.log"
    log_handle = log_path.open("xb", buffering=0)
    process = subprocess.Popen(
        command, cwd=snapshot, stdin=subprocess.DEVNULL, stdout=log_handle,
        stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
    )
    log_handle.close()
    process_identity = _process_identity(process.pid)
    receipt = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "artifact_type": "a5r5_launch_receipt", "execution_sha": execution_sha,
        "run_uid": lock["run_uid"], "pid": process.pid,
        "command_sha256": command_sha,
        "controller_lock_sha256": sha256_file(lock_path),
        "launch_claim_sha256": sha256_file(state_root / "launch_claim.json"),
        **process_identity,
        "log_path": str(log_path), "launch_count": 1, "resume_allowed": False,
        "launched_at_utc": utc_now(),
    }
    publish_json_no_overwrite(state_root / "launch_receipt.json", receipt)
    return receipt


def status(repo: Path) -> dict[str, Any]:
    execution_sha = _git(repo, "rev-parse", "HEAD")
    identity = execution_identity(execution_sha)
    state_root = Path(identity["state_root"])
    lock_path = state_root / "controller_lock.json"
    if not state_root.exists():
        return {"status": "NOT_MATERIALIZED", **identity}
    if not (state_root / "materialization_claim.json").exists() or not lock_path.exists():
        return {"status": "MATERIALIZATION_FAILED_TERMINAL_NO_RELAUNCH", **identity}
    lock = _validate_controller_lock(lock_path, expected_execution_sha=execution_sha)
    claim_path = state_root / "launch_claim.json"
    result_path = state_root / "worker_result.json"
    if not claim_path.exists():
        return {"status": "READY_TO_LAUNCH_ONCE", **identity}
    launch = _validate_launch_chain(lock, lock_path)
    if not (state_root / "launch_receipt.json").exists():
        return {"status": "LAUNCH_FAILED_TERMINAL_NO_RELAUNCH", **identity}
    launch_receipt = _validate_launch_receipt(lock, lock_path, launch)
    if result_path.exists():
        result = _validate_worker_result(lock, lock_path, launch, launch_receipt)
        state = result["status"]
    else:
        try:
            live_identity = _process_identity(int(launch_receipt["pid"]))
        except (FileNotFoundError, ProcessLookupError):
            state = "INTERRUPTED_TERMINAL_NO_RELAUNCH"
        else:
            expected_identity = {
                name: launch_receipt[name]
                for name in ("pid", "proc_starttime_ticks", "proc_cmdline_sha256")
            }
            state = (
                "RUNNING"
                if live_identity == expected_identity
                else "INTERRUPTED_TERMINAL_NO_RELAUNCH"
            )
        result = None
    return {"status": state, **identity, "launch": launch_receipt, "result": result}


def _deep_verify_completed(lock_path: Path) -> dict[str, Any]:
    """Replay the immutable producer verifier from the source snapshot."""

    lock = _validate_controller_lock(lock_path)
    launch = _validate_launch_chain(lock, lock_path)
    launch_receipt = _validate_launch_receipt(lock, lock_path, launch)
    result = _validate_worker_result(lock, lock_path, launch, launch_receipt)
    if result["status"] != "WORKER_EXIT_ZERO" or result["exit_code"] != 0:
        raise RuntimeError("A5R5 worker did not exit successfully")
    input_root = Path(lock["input_root"])
    expected_entries = {
        "a5r2_input_lock_execution_identity.json",
        "input_geometry_samples.parquet",
        "input_geometry_manifest.json",
        "input_geometry_receipt.json",
        "row_templates",
        "input_row_template_chunk_index.json",
        "streaming_input_lock_receipt.json",
        "a5_prompt_census_records.jsonl",
        "a5_prompt_census_manifest.json",
        "e0_design_input_lock.pt",
        "e0_design_input_lock_manifest.json",
        "A5R2_INPUT_LOCK_GO.json",
    }
    if (
        not input_root.is_dir()
        or input_root.is_symlink()
        or {path.name for path in input_root.iterdir()} != expected_entries
        or any(path.name.startswith(".") and path.name.endswith(".tmp") for path in input_root.rglob("*"))
    ):
        raise RuntimeError("A5R5 completed input root has missing/unbound/crash artifacts")
    identity_path = input_root / "a5r2_input_lock_execution_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["identity_sha256"] = sha256_file(identity_path)
    from script.experiment.fpct_e1_prepare_input_lock import _verify_completed_input_lock

    manifest_path = input_root / "e0_design_input_lock_manifest.json"
    sidecar_path = input_root / "e0_design_input_lock.pt"
    completed = _verify_completed_input_lock(
        repo_root=Path(lock["snapshot_root"]),
        e0_data_root=E0_DATA_ROOT,
        output_sidecar=sidecar_path,
        output_manifest=manifest_path,
        execution_identity=identity,
        expect_go_receipt=True,
    )
    return {
        "status": "A5R5_DEEP_COMPLETED_VERIFIER_GO",
        "execution_sha": lock["execution_sha"],
        "run_uid": lock["run_uid"],
        "manifest_sha256": sha256_file(manifest_path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "manifest_status": completed.get("status"),
    }


def verify_success(repo: Path) -> dict[str, Any]:
    execution_sha = verify_repo(repo)
    current = status(repo)
    if current["status"] != "WORKER_EXIT_ZERO":
        raise RuntimeError(f"A5R5 worker is not successful: {current['status']}")
    identity = execution_identity(current["execution_sha"])
    lock_path = Path(identity["state_root"]) / "controller_lock.json"
    lock = _validate_controller_lock(lock_path, expected_execution_sha=execution_sha)
    trusted_source_lock = repo / "script/experiment/fpct_e1_source_snapshot_lock.py"
    subprocess.run(
        [str(PYTHON), str(trusted_source_lock), "verify-mounted",
         "--execution-sha", execution_sha, "--snapshot", str(lock["snapshot_root"]),
         "--receipt", str(lock["source_snapshot_receipt"])],
        check=True, stdout=subprocess.DEVNULL,
    )
    command = [
        str(PYTHON), "-I", str(Path(lock["snapshot_root"]) / "script/runtime/fpct_bootstrap.py"),
        "--repo-root", str(lock["snapshot_root"]),
        "--target", str(Path(lock["snapshot_root"]) / "script/experiment/fpct_e1_a5r5_controller.py"),
        "--", "deep-verify", "--lock", str(lock_path),
    ]
    environment = dict(lock["prepare_environment"])
    verified = subprocess.run(
        command, cwd=lock["snapshot_root"], env=environment, check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    deep = json.loads(verified.stdout)
    if deep.get("status") != "A5R5_DEEP_COMPLETED_VERIFIER_GO":
        raise RuntimeError("A5R5 independent deep verifier did not return GO")
    input_root = Path(current["input_root"])
    required = {
        "sidecar": input_root / "e0_design_input_lock.pt",
        "manifest": input_root / "e0_design_input_lock_manifest.json",
        "go": input_root / "A5R2_INPUT_LOCK_GO.json",
    }
    if any(not path.is_file() or path.is_symlink() for path in required.values()):
        raise RuntimeError("A5R5 worker exit 0 lacks a canonical terminal artifact")
    if (input_root / "A5R2_INPUT_LOCK_BLOCKED.json").exists():
        raise RuntimeError("A5R5 root contains contradictory BLOCKED state")
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    go = json.loads(required["go"].read_text(encoding="utf-8"))
    if (
        manifest.get("execution", {}).get("execution_sha") != current["execution_sha"]
        or go.get("execution_sha") != current["execution_sha"]
        or go.get("run_uid") != current["run_uid"]
        or go.get("input_lock_manifest_sha256") != sha256_file(required["manifest"])
    ):
        raise RuntimeError("A5R5 terminal artifact cross-binding changed")
    return {
        "status": "A5R5_INPUT_LOCK_GO_VERIFIED",
        "execution_sha": current["execution_sha"], "run_uid": current["run_uid"],
        "deep_verifier": deep,
        "artifacts": {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                      for name, path in required.items()},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("materialize", "launch", "status", "verify"):
        item = sub.add_parser(name)
        item.add_argument("--repo", type=Path, required=True)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--lock", type=Path, required=True)
    deep_parser = sub.add_parser("deep-verify")
    deep_parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "materialize":
        output = materialize(args.repo.absolute())
    elif args.command == "launch":
        output = launch(args.repo.absolute())
    elif args.command == "status":
        output = status(args.repo.absolute())
    elif args.command == "verify":
        output = verify_success(args.repo.absolute())
    elif args.command == "worker":
        return worker(args.lock.absolute())
    else:
        output = _deep_verify_completed(args.lock.absolute())
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
