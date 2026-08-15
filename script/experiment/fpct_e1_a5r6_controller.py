#!/usr/bin/env python3
"""One-shot detached controller for the FPCT-E1 A5R6 fresh-root recovery."""

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


PROTOCOL_ID = "fpct_e1_mechanism_audit_v13_a5r6_schema_binding_recovery"
BRANCH = "research/fpct-e1-mechanism-audit"
BASE_COMMIT = "ab4052cd0f97c248e674ed4f7b2e0a4fa23d87f5"
PREDECESSOR_EXECUTION_SHAS = frozenset({
    "a47d52f8adf5ada79919d39ac42d7b9fe655595f",
    BASE_COMMIT,
})
GATE_RELATIVE_PATH = Path(
    "recipe/eval_recipe/fpct_e1/e1_a5r6_schema_binding_recovery_gate.json"
)
GATE_STATUS = "GO_PRE_NATURAL_A5R6_SCHEMA_BINDING_RECOVERY"
GATE_EXPECTED_TEST_COUNT = 318
GATE_EXPECTED_COLLECTION_SUMMARY = "collected 322 items / 4 deselected / 318 selected"
GATE_REQUIRED_TEST_NODES = (
    "test/test_fpct_e1_a5r2_prepare_integration.py::test_a5r3_publication_is_independently_verified_but_v9_manifest_binding_is_exact_five_fields",
    "test/test_fpct_e1_a5r6_controller.py::test_materialize_requires_current_pre_natural_gate_before_creating_state",
    "test/test_fpct_e1_a5r6_controller.py::test_filesystem_portable_digest_and_local_safety_accept_uniform_gid_namespace_remap",
    "test/test_fpct_e1_a5r6_controller.py::test_environment_local_safety_rejects_owner_and_unauthorized_root_group",
    "test/test_fpct_e1_a5r6_controller.py::test_pre_natural_gate_rejects_dropped_tracked_or_immutable_binding",
    "test/test_fpct_e1_a5r6_controller.py::test_deep_verifier_receipt_is_strict_and_replayable",
    "test/test_fpct_e1_a5r6_gate.py::test_gate_tracks_operative_e1_manifest",
    "test/test_fpct_e1_a5r6_gate.py::test_verify_gate_rejects_weakened_checks_even_with_recomputed_evidence",
    "test/test_fpct_e1_a5r6_gate.py::test_gate_replays_old_controller_terminal_bytes",
)
GATE_REQUIRED_TRACKED_PATHS = (
    "FPCT_E1_A5R6_SCHEMA_BINDING_RECOVERY_AMENDMENT.md",
    "recipe/eval_recipe/fpct_e1/e1_a5r6_schema_binding_recovery_manifest.json",
    "recipe/eval_recipe/fpct_e1/executions/ab4052cd/input_lock_schema_binding_failure_observation.json",
    "recipe/eval_recipe/fpct_e1/e1_manifest.json",
    "script/experiment/fpct_e1_prepare_input_lock.py",
    "script/experiment/fpct_e1_a5r6_controller.py",
    "script/analysis/fpct_e1_a5r6_gate.py",
    "test/test_fpct_e1_a5r6_controller.py",
    "test/test_fpct_e1_a5r6_gate.py",
    "test/test_fpct_e1_a5r2_prepare_integration.py",
)
GATE_REQUIRED_IMMUTABLE_SHA256 = {
    "math.md": "98d1b61f84d046548d5ba0070d6858c7080cb14fdef9169b08ad167461b809ad",
    "recipe/eval_recipe/fpct_e1/e1_a5r2_choice_cardinality_schema.json": "9cb387628e4b8fcf6c978e082b8c3380dbc62406c6aa460892c679872d656d7f",
    "recipe/eval_recipe/fpct_e1/e1_a5r3_portable_publication_schema.json": "b24d197481c01c917ad9d4063441f96ac23979bdb4b3db24855e551c7fb91460",
    "recipe/eval_recipe/fpct_e1/e1_a5r5_interruption_recovery_gate.json": "6190db2d26ad55c8a467c880ff0050c1e3c7bd0cff2bec83d8a320bc9afc21f9",
}
GATE_HISTORICAL_DESELECTED_NODES = (
    "test/test_fpct_e1_a5_prompt_gate.py::test_all_config_attestation_rejects_index_or_config_tamper[config_index]",
    "test/test_fpct_e1_a5_prompt_gate.py::test_all_config_attestation_rejects_index_or_config_tamper[rendered_config]",
    "test/test_fpct_e1_a5r1_hash_domain_gate.py::test_independent_hash_domain_oracles_and_negative_controls",
    "test/test_fpct_e1_a5r1_hash_domain_gate.py::test_atomic_failure_check_executes_producer_terminal_transition",
)
COMPLETED_INPUT_ROOT_ENTRIES = frozenset({
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
})
GATE_REQUIRED_CHECKS = {
    "v9_schema_byte_immutable": True,
    "a5r3_schema_byte_immutable": True,
    "choice_binding_exact_five_field_projection": True,
    "publication_independently_verified": True,
    "complete_manifest_publication_regression": True,
    "numeric_uid_gid_excluded_from_portable_digest": True,
    "environment_local_owner_mode_safety": True,
    "failed_root_no_resume_repair_reuse_cleanup": True,
    "failed_controller_terminal_evidence_replayed": True,
    "controller_requires_current_gate_before_natural_access": True,
    "successor_natural_accessed": False,
    "model_or_checkpoint_loaded": False,
    "model_forward_run": False,
    "gpu_cuda_or_kubernetes_used": False,
    "training": False,
    "e1_2_e1_3_or_pilot_accessed": False,
}
OLD_ROOT = Path("/netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-ab4052cd-v1")
OLD_TREE_SHA256 = "b8d39a970eb8f3d4bd6f2c5070f475cce4c7f96fd9a2cc6cae6a7e9cf7d26b4a"
OLD_FILE_COUNT = 8913
OLD_TOTAL_BYTES = 4426202802
OLD_INVENTORY_ENTRY_COUNT = 9343
OLD_INVENTORY_SHA256 = "67c572aa55d421338b9ee8725b8bd12944fa194cc8e34dabefac175504a32daf"
PORTABLE_INVENTORY_ALGORITHM = "kind_path_mode_size_content_v2_no_numeric_owner"
OLD_BLOCKED_SHA256 = "9fc9c3a6f533d19ea9611c89828302eb8a6fb18ba725b210382e8499a51c4101"
OLD_SIDECAR_SHA256 = "9c35e8c652360629eda1b6cef4618e5b227e51c688d3a95259cf191b1bad84aa"
OLD_STREAMING_RECEIPT_SHA256 = "ec543abb13448484ecd7368860f2e1cf9b4067c927b8ca6b646fe9d9968ed827"
OLD_CONTROLLER_ROOT = Path(
    "/netdisk/lijunsi/fpct-e1/.fpct-e1-a5r5-controller-ab4052cd-v1"
)
OLD_WORKER_RESULT_BYTES = 727
OLD_WORKER_RESULT_SHA256 = "ab34009378db74259da22c92f1c2eb3a6c4222a32ceec598fdc5732bc090e628"
OLD_WORKER_LOG_BYTES = 14889
OLD_WORKER_LOG_SHA256 = "c7cbe5812ba13189a3fe6ee1b088dd066c4895df23f12f43763831e2af1393b9"
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


def portable_inventory_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash only portable fields; any caller-local owner fields are ignored."""

    payload = "".join(
        f"{record['kind']}\0{record['relative_path']}\0{record['mode']}\0"
        f"{record['size']}\0{record['content']}\n"
        for record in records
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def filesystem_inventory_fingerprint(root: Path) -> dict[str, Any]:
    """Portable content inventory; numeric UID/GID are deliberately excluded."""
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"inventory root is not a real directory: {root}")
    rows: list[dict[str, Any]] = []
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
        rows.append({
            "kind": kind,
            "relative_path": relative,
            "mode": oct(stat.S_IMODE(metadata.st_mode)),
            "size": size,
            "content": content,
        })
    return {
        "algorithm": PORTABLE_INVENTORY_ALGORITHM,
        "entry_count": len(rows),
        "inventory_sha256": portable_inventory_digest(rows),
        "numeric_uid_in_digest": False,
        "numeric_gid_in_digest": False,
    }


def environment_local_owner_mode_safety(root: Path) -> dict[str, Any]:
    """Validate ownership/mode only in the current mount/user namespace."""

    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"environment-local root is unsafe: {root}")
    root_metadata = root.lstat()
    allowed_groups = {os.getegid(), *os.getgroups()}
    counts = {
        "entry_count": 0,
        "directory_count": 0,
        "regular_file_count": 0,
        "owner_mismatch_count": 0,
        "root_local_group_mismatch_count": 0,
        "unsupported_entry_count": 0,
        "world_writable_count": 0,
        "setuid_count": 0,
        "sticky_count": 0,
        "setgid_regular_file_count": 0,
        "owner_access_failure_count": 0,
    }
    paths = [root, *sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())]
    for path in paths:
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        counts["entry_count"] += 1
        if stat.S_ISDIR(metadata.st_mode):
            counts["directory_count"] += 1
            owner_access_ok = mode & 0o500 == 0o500
        elif stat.S_ISREG(metadata.st_mode):
            counts["regular_file_count"] += 1
            owner_access_ok = mode & 0o400 == 0o400
        else:
            counts["unsupported_entry_count"] += 1
            owner_access_ok = False
        counts["owner_mismatch_count"] += int(metadata.st_uid != os.geteuid())
        counts["root_local_group_mismatch_count"] += int(
            metadata.st_gid != root_metadata.st_gid
        )
        counts["world_writable_count"] += int(bool(mode & 0o002))
        counts["setuid_count"] += int(bool(mode & stat.S_ISUID))
        counts["sticky_count"] += int(bool(mode & stat.S_ISVTX))
        counts["setgid_regular_file_count"] += int(
            stat.S_ISREG(metadata.st_mode) and bool(mode & stat.S_ISGID)
        )
        counts["owner_access_failure_count"] += int(not owner_access_ok)
    checks = {
        "root_group_authorized_for_process": root_metadata.st_gid in allowed_groups,
        "all_entries_owned_by_effective_user": counts["owner_mismatch_count"] == 0,
        "all_entries_use_root_local_group": counts["root_local_group_mismatch_count"] == 0,
        "only_directories_and_regular_files": counts["unsupported_entry_count"] == 0,
        "no_world_writable_entries": counts["world_writable_count"] == 0,
        "no_setuid_entries": counts["setuid_count"] == 0,
        "no_sticky_entries": counts["sticky_count"] == 0,
        "no_setgid_regular_files": counts["setgid_regular_file_count"] == 0,
        "owner_access_sufficient": counts["owner_access_failure_count"] == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"environment-local owner/mode safety failed: {counts} {checks}")
    return {
        **counts,
        **checks,
        "numeric_uid_frozen_across_namespaces": False,
        "numeric_gid_frozen_across_namespaces": False,
    }


def verify_old_forensic_root() -> dict[str, Any]:
    observed = tree_fingerprint(OLD_ROOT)
    expected = {
        "file_count": OLD_FILE_COUNT,
        "total_bytes": OLD_TOTAL_BYTES,
        "tree_sha256": OLD_TREE_SHA256,
    }
    if observed != expected:
        raise RuntimeError(f"immutable ab4052cd forensic root changed: {observed}")
    inventory = filesystem_inventory_fingerprint(OLD_ROOT)
    if inventory != {
        "algorithm": PORTABLE_INVENTORY_ALGORITHM,
        "entry_count": OLD_INVENTORY_ENTRY_COUNT,
        "inventory_sha256": OLD_INVENTORY_SHA256,
        "numeric_uid_in_digest": False,
        "numeric_gid_in_digest": False,
    }:
        raise RuntimeError(f"immutable ab4052cd portable inventory changed: {inventory}")
    local_safety = environment_local_owner_mode_safety(OLD_ROOT)
    input_root = OLD_ROOT / "input_lock"
    required = [
        OLD_ROOT / "choice_audit_publication_receipt.json",
        OLD_ROOT / "choice_audit/choice_cardinality_audit_lock.json",
        input_root / "input_geometry_receipt.json",
        input_root / "streaming_input_lock_receipt.json",
        input_root / "e0_design_input_lock.pt",
        input_root / "A5R2_INPUT_LOCK_BLOCKED.json",
    ]
    if not all(path.is_file() and not path.is_symlink() for path in required):
        raise RuntimeError("ab4052cd forensic closure lost a required retained artifact")
    if (
        sha256_file(input_root / "A5R2_INPUT_LOCK_BLOCKED.json") != OLD_BLOCKED_SHA256
        or sha256_file(input_root / "e0_design_input_lock.pt") != OLD_SIDECAR_SHA256
        or sha256_file(input_root / "streaming_input_lock_receipt.json")
        != OLD_STREAMING_RECEIPT_SHA256
    ):
        raise RuntimeError("ab4052cd terminal retained artifact bytes changed")
    forbidden = [
        input_root / "e0_design_input_lock_manifest.json",
        input_root / "A5R2_INPUT_LOCK_GO.json",
    ]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("ab4052cd forensic root acquired a forbidden GO artifact")
    blocked = json.loads((input_root / "A5R2_INPUT_LOCK_BLOCKED.json").read_text(encoding="utf-8"))
    if (
        blocked.get("status") != "A5R2_INPUT_LOCK_BLOCKED"
        or blocked.get("failure_stage") != "V9_INPUT_LOCK"
        or blocked.get("failure_code") != "VALUEERROR"
        or blocked.get("resume_allowed") is not False
        or blocked.get("artifact_reuse_allowed") is not False
        or blocked.get("scientific_result") is not False
    ):
        raise RuntimeError("ab4052cd terminal blocked semantics changed")
    return {**observed, **inventory, "environment_local_safety": local_safety}


def expected_forensic_closure() -> dict[str, Any]:
    return {
        "file_count": OLD_FILE_COUNT,
        "total_bytes": OLD_TOTAL_BYTES,
        "tree_sha256": OLD_TREE_SHA256,
        "algorithm": PORTABLE_INVENTORY_ALGORITHM,
        "entry_count": OLD_INVENTORY_ENTRY_COUNT,
        "inventory_sha256": OLD_INVENTORY_SHA256,
        "numeric_uid_in_digest": False,
        "numeric_gid_in_digest": False,
        "environment_local_safety": {
            "entry_count": 9343,
            "directory_count": 430,
            "regular_file_count": 8913,
            "owner_mismatch_count": 0,
            "root_local_group_mismatch_count": 0,
            "unsupported_entry_count": 0,
            "world_writable_count": 0,
            "setuid_count": 0,
            "sticky_count": 0,
            "setgid_regular_file_count": 0,
            "owner_access_failure_count": 0,
            "root_group_authorized_for_process": True,
            "all_entries_owned_by_effective_user": True,
            "all_entries_use_root_local_group": True,
            "only_directories_and_regular_files": True,
            "no_world_writable_entries": True,
            "no_setuid_entries": True,
            "no_sticky_entries": True,
            "no_setgid_regular_files": True,
            "owner_access_sufficient": True,
            "numeric_uid_frozen_across_namespaces": False,
            "numeric_gid_frozen_across_namespaces": False,
        },
    }


def expected_old_controller_evidence() -> dict[str, Any]:
    return {
        "controller_root": str(OLD_CONTROLLER_ROOT),
        "worker_result": {
            "bytes": OLD_WORKER_RESULT_BYTES,
            "sha256": OLD_WORKER_RESULT_SHA256,
            "status": "WORKER_TERMINAL_FAILURE",
            "exit_code": 1,
        },
        "worker_log": {
            "bytes": OLD_WORKER_LOG_BYTES,
            "sha256": OLD_WORKER_LOG_SHA256,
        },
    }


def verify_old_controller_evidence() -> dict[str, Any]:
    result_path = OLD_CONTROLLER_ROOT / "worker_result.json"
    log_path = OLD_CONTROLLER_ROOT / "worker.log"
    for path in (result_path, log_path):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"A5R5 terminal controller evidence is unsafe: {path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    observed = {
        "controller_root": str(OLD_CONTROLLER_ROOT),
        "worker_result": {
            "bytes": result_path.stat().st_size,
            "sha256": sha256_file(result_path),
            "status": result.get("status"),
            "exit_code": result.get("exit_code"),
        },
        "worker_log": {
            "bytes": log_path.stat().st_size,
            "sha256": sha256_file(log_path),
        },
    }
    if observed != expected_old_controller_evidence():
        raise RuntimeError(f"A5R5 terminal controller evidence changed: {observed}")
    if (
        result.get("execution_sha") != BASE_COMMIT
        or result.get("run_uid")
        != "fpct-e1-a5r2-choice-cardinality-ab4052cd-v1"
        or result.get("resume_allowed") is not False
    ):
        raise RuntimeError("A5R5 terminal controller semantics changed")
    return observed


def _canonical_without_evidence(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(
        {key: item for key, item in value.items() if key != "evidence_sha256"}
    )


def verify_pre_natural_gate(
    repo_root: Path, *, replay_retained_evidence: bool = False,
) -> dict[str, Any]:
    """Verify the committed A5R6 gate before any successor root can exist."""

    gate_path = repo_root / GATE_RELATIVE_PATH
    if not gate_path.is_file() or gate_path.is_symlink():
        raise RuntimeError("A5R6 pre-natural gate is absent or unsafe")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        gate.get("schema_version") != 13
        or gate.get("protocol_id") != PROTOCOL_ID
        or gate.get("artifact_type") != "a5r6_pre_natural_synthetic_gate"
        or gate.get("status") != GATE_STATUS
        or gate.get("base_commit") != BASE_COMMIT
        or gate.get("test_count") != GATE_EXPECTED_TEST_COUNT
        or re.fullmatch(r"[0-9a-f]{64}", str(gate.get("test_output_sha256", ""))) is None
        or gate.get("checks") != GATE_REQUIRED_CHECKS
    ):
        raise RuntimeError("A5R6 pre-natural gate contract changed")
    expected_test_contract = {
        "required_nodes": list(GATE_REQUIRED_TEST_NODES),
        "required_nodes_sha256": hashlib.sha256(
            ("\n".join(GATE_REQUIRED_TEST_NODES) + "\n").encode()
        ).hexdigest(),
        "historical_deselected_nodes": list(GATE_HISTORICAL_DESELECTED_NODES),
        "historical_deselected_nodes_sha256": hashlib.sha256(
            ("\n".join(GATE_HISTORICAL_DESELECTED_NODES) + "\n").encode()
        ).hexdigest(),
        "collection_summary": GATE_EXPECTED_COLLECTION_SUMMARY,
    }
    if gate.get("test_contract") != expected_test_contract:
        raise RuntimeError("A5R6 pre-natural gate test contract changed")
    evidence = gate.get("evidence_sha256")
    if evidence != hashlib.sha256(_canonical_without_evidence(gate)).hexdigest():
        raise RuntimeError("A5R6 pre-natural gate evidence hash changed")
    tracked = gate.get("tracked_files_sha256")
    immutable = gate.get("immutable_predecessor_sha256")
    expected_tracked = {
        path: sha256_file(repo_root / path) for path in GATE_REQUIRED_TRACKED_PATHS
    }
    if (
        tracked != expected_tracked
        or immutable != GATE_REQUIRED_IMMUTABLE_SHA256
        or any(
            sha256_file(repo_root / path) != digest
            for path, digest in GATE_REQUIRED_IMMUTABLE_SHA256.items()
        )
    ):
        raise RuntimeError("A5R6 pre-natural gate source closure changed")
    if gate.get("failed_root_portable_closure") != expected_forensic_closure():
        raise RuntimeError("A5R6 pre-natural gate failed-root closure changed")
    if gate.get("failed_controller_terminal_evidence") != expected_old_controller_evidence():
        raise RuntimeError("A5R6 pre-natural gate controller evidence changed")
    if replay_retained_evidence:
        if verify_old_forensic_root() != expected_forensic_closure():
            raise RuntimeError("A5R6 retained root replay changed")
        if verify_old_controller_evidence() != expected_old_controller_evidence():
            raise RuntimeError("A5R6 retained controller replay changed")
    return {
        "path": str(GATE_RELATIVE_PATH),
        "sha256": sha256_file(gate_path),
        "evidence_sha256": evidence,
        "test_count": gate.get("test_count"),
        "test_output_sha256": gate.get("test_output_sha256"),
        "test_contract": gate.get("test_contract"),
        "tracked_files_sha256": tracked,
        "immutable_predecessor_sha256": immutable,
    }


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
    if execution_sha in PREDECESSOR_EXECUTION_SHAS:
        raise ValueError("A5R6 execution SHA must be a fresh successor commit")
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
        "state_root": str(RUN_PARENT / f".fpct-e1-a5r6-controller-{prefix}-v1"),
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
        "--target", str(snapshot / "script/experiment/fpct_e1_a5r6_controller.py"),
        "--", "worker", "--lock", str(lock_path),
    ]


def _validate_controller_lock(
    lock_path: Path, *, expected_execution_sha: str | None = None,
    verify_forensic_root: bool = False,
) -> dict[str, Any]:
    lock = _read_canonical_controller_json(lock_path, "A5R6 controller lock")
    execution_sha = str(lock.get("execution_sha", ""))
    if expected_execution_sha is not None and execution_sha != expected_execution_sha:
        raise RuntimeError("A5R6 controller lock execution SHA differs from current HEAD")
    identity = execution_identity(execution_sha)
    expected_path = Path(identity["state_root"]) / "controller_lock.json"
    if lock_path.absolute() != expected_path.absolute():
        raise RuntimeError("A5R6 controller lock path is not canonical")
    if any(lock.get(key) != value for key, value in identity.items()):
        raise RuntimeError("A5R6 controller lock derived identity changed")
    required = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "a5r6_controller_lock",
        "status": "READY_TO_LAUNCH_ONCE",
        "launch_count": 0,
        "resume_allowed": False,
    }
    if any(lock.get(key) != value for key, value in required.items()):
        raise RuntimeError("A5R6 controller lock schema/status changed")
    if lock.get("forensic_closure") != expected_forensic_closure():
        raise RuntimeError("A5R6 interrupted-root forensic closure changed")
    if lock.get("failed_controller_terminal_evidence") != expected_old_controller_evidence():
        raise RuntimeError("A5R6 interrupted-controller evidence changed")
    if verify_forensic_root and verify_old_forensic_root() != expected_forensic_closure():
        raise RuntimeError("A5R6 interrupted-root forensic replay changed")
    snapshot = Path(identity["snapshot_root"])
    gate_binding = verify_pre_natural_gate(
        snapshot, replay_retained_evidence=verify_forensic_root,
    )
    if lock.get("pre_natural_gate") != gate_binding:
        raise RuntimeError("A5R6 controller lock pre-natural gate binding changed")
    receipt = snapshot / ".fpct_e1_source_snapshot_receipt.json"
    if (
        lock.get("source_snapshot_receipt") != str(receipt)
        or not receipt.is_file()
        or receipt.is_symlink()
        or lock.get("source_snapshot_receipt_sha256") != sha256_file(receipt)
    ):
        raise RuntimeError("A5R6 controller lock source snapshot receipt changed")
    claim_path = Path(identity["state_root"]) / "materialization_claim.json"
    claim = _read_canonical_controller_json(claim_path, "A5R6 materialization claim")
    if (
        claim.get("protocol_id") != PROTOCOL_ID
        or claim.get("artifact_type") != "a5r6_materialization_claim"
        or any(claim.get(key) != value for key, value in identity.items())
        or lock.get("materialization_claim_sha256") != sha256_file(claim_path)
        or claim.get("pre_natural_gate") != gate_binding
    ):
        raise RuntimeError("A5R6 materialization claim binding changed")
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
        raise RuntimeError("A5R6 frozen prepare environment changed")
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
    claim = _read_canonical_controller_json(claim_path, "A5R6 launch claim")
    worker_command = _worker_command(lock, lock_path)
    worker_command_sha = hashlib.sha256("\0".join(worker_command).encode()).hexdigest()
    if (
        claim.get("protocol_id") != PROTOCOL_ID
        or claim.get("artifact_type") != "a5r6_launch_claim"
        or claim.get("execution_sha") != lock["execution_sha"]
        or claim.get("run_uid") != lock["run_uid"]
        or claim.get("controller_lock_sha256") != lock_sha
        or claim.get("command_sha256") != worker_command_sha
        or claim.get("launch_count") != 1
        or claim.get("resume_allowed") is not False
    ):
        raise RuntimeError("A5R6 launch claim binding changed")
    return {"claim": claim, "claim_sha256": sha256_file(claim_path), "worker_command": worker_command}


def _validate_launch_receipt(
    lock: Mapping[str, Any], lock_path: Path, launch: Mapping[str, Any]
) -> dict[str, Any]:
    receipt_path = lock_path.parent / "launch_receipt.json"
    receipt = _read_canonical_controller_json(receipt_path, "A5R6 launch receipt")
    required = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "a5r6_launch_receipt",
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
        raise RuntimeError("A5R6 launch receipt binding changed")
    if (
        not isinstance(receipt.get("pid"), int)
        or receipt["pid"] <= 0
        or receipt.get("proc_starttime_ticks") is None
        or not isinstance(receipt.get("proc_starttime_ticks"), int)
        or not isinstance(receipt.get("proc_cmdline_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", receipt["proc_cmdline_sha256"]) is None
        or receipt.get("log_path") != str(lock_path.parent / "worker.log")
    ):
        raise RuntimeError("A5R6 launch receipt process identity is incomplete")
    return receipt


def _validate_worker_start(
    lock: Mapping[str, Any], lock_path: Path, launch: Mapping[str, Any],
    launch_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = lock_path.parent / "worker_start_claim.json"
    value = _read_canonical_controller_json(path, "A5R6 worker-start claim")
    command_sha = hashlib.sha256("\0".join(_prepare_command(lock)).encode()).hexdigest()
    required = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "a5r6_worker_start_claim",
        "execution_sha": lock["execution_sha"],
        "run_uid": lock["run_uid"],
        "controller_lock_sha256": sha256_file(lock_path),
        "launch_claim_sha256": launch["claim_sha256"],
        "prepare_command_sha256": command_sha,
        "resume_allowed": False,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise RuntimeError("A5R6 worker-start claim binding changed")
    receipt = (
        dict(launch_receipt)
        if launch_receipt is not None
        else _validate_launch_receipt(lock, lock_path, launch)
    )
    if any(
        value.get(key) != receipt.get(key)
        for key in ("pid", "proc_starttime_ticks", "proc_cmdline_sha256")
    ):
        raise RuntimeError("A5R6 worker-start process differs from launched process")
    return value


def _validate_worker_result(
    lock: Mapping[str, Any], lock_path: Path, launch: Mapping[str, Any],
    launch_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    start_path = lock_path.parent / "worker_start_claim.json"
    _validate_worker_start(lock, lock_path, launch, launch_receipt)
    path = lock_path.parent / "worker_result.json"
    value = _read_canonical_controller_json(path, "A5R6 worker result")
    command_sha = hashlib.sha256("\0".join(_prepare_command(lock)).encode()).hexdigest()
    expected_status = (
        "WORKER_EXIT_ZERO" if value.get("exit_code") == 0 else "WORKER_TERMINAL_FAILURE"
    )
    required = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "a5r6_worker_result",
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
        raise RuntimeError("A5R6 worker-result binding changed")
    return value


def _validate_deep_verifier_receipt(
    lock: Mapping[str, Any], lock_path: Path,
) -> dict[str, Any]:
    path = lock_path.parent / "deep_verifier_receipt.json"
    value = _read_canonical_controller_json(path, "A5R6 deep-verifier receipt")
    worker_result_path = lock_path.parent / "worker_result.json"
    required = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "a5r6_deep_verifier_receipt",
        "status": "A5R6_INPUT_LOCK_GO_VERIFIED",
        "execution_sha": lock["execution_sha"],
        "run_uid": lock["run_uid"],
        "controller_lock_sha256": sha256_file(lock_path),
        "worker_result_sha256": sha256_file(worker_result_path),
        "source_snapshot_receipt_sha256": lock["source_snapshot_receipt_sha256"],
        "pre_natural_gate": lock["pre_natural_gate"],
    }
    expected_value_keys = {
        *required,
        "deep_verifier",
        "artifacts",
        "verified_at_utc",
    }
    if (
        set(value) != expected_value_keys
        or any(value.get(key) != expected for key, expected in required.items())
        or not isinstance(value.get("verified_at_utc"), str)
    ):
        raise RuntimeError("A5R6 deep-verifier receipt binding changed")
    deep = value.get("deep_verifier")
    if (
        not isinstance(deep, dict)
        or set(deep) != {
            "status", "execution_sha", "run_uid", "manifest_sha256",
            "sidecar_sha256", "manifest_status",
        }
        or deep.get("status") != "A5R6_DEEP_COMPLETED_VERIFIER_GO"
        or deep.get("execution_sha") != lock["execution_sha"]
        or deep.get("run_uid") != lock["run_uid"]
    ):
        raise RuntimeError("A5R6 deep-verifier receipt replay changed")
    input_root = Path(lock["input_root"])
    if (
        not input_root.is_dir()
        or input_root.is_symlink()
        or {path.name for path in input_root.iterdir()} != COMPLETED_INPUT_ROOT_ENTRIES
        or (input_root / "A5R2_INPUT_LOCK_BLOCKED.json").exists()
        or any(
            path.name.startswith(".") and path.name.endswith(".tmp")
            for path in input_root.rglob("*")
        )
    ):
        raise RuntimeError("A5R6 deep-verifier receipt input-root shape changed")
    expected_paths = {
        "sidecar": input_root / "e0_design_input_lock.pt",
        "manifest": input_root / "e0_design_input_lock_manifest.json",
        "go": input_root / "A5R2_INPUT_LOCK_GO.json",
    }
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected_paths):
        raise RuntimeError("A5R6 deep-verifier receipt artifact set changed")
    for name, artifact_path in expected_paths.items():
        binding = artifacts.get(name)
        if (
            not isinstance(binding, dict)
            or set(binding) != {"path", "bytes", "sha256"}
            or binding.get("path") != str(artifact_path)
            or not artifact_path.is_file()
            or artifact_path.is_symlink()
            or binding.get("bytes") != artifact_path.stat().st_size
            or binding.get("sha256") != sha256_file(artifact_path)
        ):
            raise RuntimeError(f"A5R6 deep-verifier artifact changed: {name}")
    if (
        deep.get("manifest_sha256") != artifacts["manifest"]["sha256"]
        or deep.get("sidecar_sha256") != artifacts["sidecar"]["sha256"]
        or not isinstance(deep.get("manifest_status"), str)
    ):
        raise RuntimeError("A5R6 deep-verifier receipt digest cross-binding changed")
    return value


def _deep_receipt_result(value: Mapping[str, Any], path: Path) -> dict[str, Any]:
    return {
        **dict(value),
        "deep_verifier_receipt": {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
    }


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
            raise RuntimeError("A5R6 direct/non-launched worker entry is forbidden")
        return receipt


def _safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r") as handle:
        destination_resolved = destination.resolve()
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if destination_resolved not in (target, *target.parents):
                raise RuntimeError("git archive contains a path traversal")
            if member.issym() or member.islnk():
                raise RuntimeError("A5R6 source snapshot forbids archive links")
        handle.extractall(destination)


def materialize(repo: Path) -> dict[str, Any]:
    execution_sha = verify_repo(repo)
    gate_binding = verify_pre_natural_gate(repo, replay_retained_evidence=True)
    forensic = expected_forensic_closure()
    old_controller_evidence = expected_old_controller_evidence()
    identity = execution_identity(execution_sha)
    run_root = Path(identity["run_root"])
    snapshot = Path(identity["snapshot_root"])
    input_root = Path(identity["input_root"])
    state_root = Path(identity["state_root"])
    if run_root.exists() or state_root.exists():
        raise RuntimeError("A5R6 successor root/state already exists; never reuse it")
    state_root.mkdir(mode=0o700, parents=True)
    materialization_claim_path = state_root / "materialization_claim.json"
    materialization_claim = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "a5r6_materialization_claim",
        **identity,
        "pre_natural_gate": gate_binding,
        "claimed_at_utc": utc_now(),
        "resume_allowed": False,
    }
    publish_json_no_overwrite(materialization_claim_path, materialization_claim)
    run_root.mkdir(mode=0o700)
    snapshot.mkdir(mode=0o700)
    input_root.mkdir(mode=0o700)
    archive = Path("/tmp") / f"{identity['run_uid']}.tar"
    if archive.exists():
        raise RuntimeError("A5R6 source archive path already exists")
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
    if verify_pre_natural_gate(snapshot) != gate_binding:
        raise RuntimeError("A5R6 source snapshot gate differs from trusted worktree gate")
    lock = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "a5r6_controller_lock",
        "status": "READY_TO_LAUNCH_ONCE",
        **identity,
        "source_snapshot_receipt": str(receipt),
        "source_snapshot_receipt_sha256": sha256_file(receipt),
        "materialization_claim_sha256": sha256_file(materialization_claim_path),
        "pre_natural_gate": gate_binding,
        "forensic_closure": forensic,
        "failed_controller_terminal_evidence": old_controller_evidence,
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
    lock = _validate_controller_lock(lock_path, verify_forensic_root=True)
    state_root = lock_path.parent
    launch = _validate_launch_chain(lock, lock_path)
    launch_receipt = _await_bound_launch_receipt(lock, lock_path, launch)
    command = _prepare_command(lock)
    command_sha = hashlib.sha256("\0".join(command).encode()).hexdigest()
    worker_start_path = state_root / "worker_start_claim.json"
    worker_start = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "a5r6_worker_start_claim",
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
        "artifact_type": "a5r6_worker_result", "execution_sha": lock["execution_sha"],
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
    lock = _validate_controller_lock(
        lock_path, expected_execution_sha=execution_sha,
        verify_forensic_root=True,
    )
    if any((state_root / name).exists() for name in (
        "launch_claim.json", "launch_receipt.json", "worker_start_claim.json", "worker_result.json"
    )):
        raise RuntimeError("A5R6 worker was already launched; never relaunch")
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
        "artifact_type": "a5r6_launch_claim", "execution_sha": execution_sha,
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
        "artifact_type": "a5r6_launch_receipt", "execution_sha": execution_sha,
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
        deep_receipt_path = state_root / "deep_verifier_receipt.json"
        if state == "WORKER_EXIT_ZERO" and deep_receipt_path.exists():
            deep_receipt = _validate_deep_verifier_receipt(lock, lock_path)
            state = deep_receipt["status"]
        else:
            deep_receipt = None
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
        deep_receipt = None
    return {
        "status": state, **identity, "launch": launch_receipt, "result": result,
        "deep_verifier_receipt": deep_receipt,
    }


def _deep_verify_completed(lock_path: Path) -> dict[str, Any]:
    """Replay the immutable producer verifier from the source snapshot."""

    lock = _validate_controller_lock(lock_path, verify_forensic_root=True)
    launch = _validate_launch_chain(lock, lock_path)
    launch_receipt = _validate_launch_receipt(lock, lock_path, launch)
    result = _validate_worker_result(lock, lock_path, launch, launch_receipt)
    if result["status"] != "WORKER_EXIT_ZERO" or result["exit_code"] != 0:
        raise RuntimeError("A5R6 worker did not exit successfully")
    input_root = Path(lock["input_root"])
    if (
        not input_root.is_dir()
        or input_root.is_symlink()
        or {path.name for path in input_root.iterdir()} != COMPLETED_INPUT_ROOT_ENTRIES
        or any(path.name.startswith(".") and path.name.endswith(".tmp") for path in input_root.rglob("*"))
    ):
        raise RuntimeError("A5R6 completed input root has missing/unbound/crash artifacts")
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
        "status": "A5R6_DEEP_COMPLETED_VERIFIER_GO",
        "execution_sha": lock["execution_sha"],
        "run_uid": lock["run_uid"],
        "manifest_sha256": sha256_file(manifest_path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "manifest_status": completed.get("status"),
    }


def verify_success(repo: Path) -> dict[str, Any]:
    execution_sha = verify_repo(repo)
    current = status(repo)
    if current["status"] not in (
        "WORKER_EXIT_ZERO", "A5R6_INPUT_LOCK_GO_VERIFIED",
    ):
        raise RuntimeError(f"A5R6 worker is not successful: {current['status']}")
    identity = execution_identity(current["execution_sha"])
    lock_path = Path(identity["state_root"]) / "controller_lock.json"
    lock = _validate_controller_lock(lock_path, expected_execution_sha=execution_sha)
    receipt_path = Path(identity["state_root"]) / "deep_verifier_receipt.json"
    if receipt_path.exists():
        return _deep_receipt_result(
            _validate_deep_verifier_receipt(lock, lock_path), receipt_path
        )
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
        "--target", str(Path(lock["snapshot_root"]) / "script/experiment/fpct_e1_a5r6_controller.py"),
        "--", "deep-verify", "--lock", str(lock_path),
    ]
    environment = dict(lock["prepare_environment"])
    verified = subprocess.run(
        command, cwd=lock["snapshot_root"], env=environment, check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    deep = json.loads(verified.stdout)
    if deep.get("status") != "A5R6_DEEP_COMPLETED_VERIFIER_GO":
        raise RuntimeError("A5R6 independent deep verifier did not return GO")
    input_root = Path(current["input_root"])
    required = {
        "sidecar": input_root / "e0_design_input_lock.pt",
        "manifest": input_root / "e0_design_input_lock_manifest.json",
        "go": input_root / "A5R2_INPUT_LOCK_GO.json",
    }
    if any(not path.is_file() or path.is_symlink() for path in required.values()):
        raise RuntimeError("A5R6 worker exit 0 lacks a canonical terminal artifact")
    if (input_root / "A5R2_INPUT_LOCK_BLOCKED.json").exists():
        raise RuntimeError("A5R6 root contains contradictory BLOCKED state")
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    go = json.loads(required["go"].read_text(encoding="utf-8"))
    if (
        manifest.get("execution", {}).get("execution_sha") != current["execution_sha"]
        or go.get("execution_sha") != current["execution_sha"]
        or go.get("run_uid") != current["run_uid"]
        or go.get("input_lock_manifest_sha256") != sha256_file(required["manifest"])
    ):
        raise RuntimeError("A5R6 terminal artifact cross-binding changed")
    receipt = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "a5r6_deep_verifier_receipt",
        "status": "A5R6_INPUT_LOCK_GO_VERIFIED",
        "execution_sha": current["execution_sha"], "run_uid": current["run_uid"],
        "controller_lock_sha256": sha256_file(lock_path),
        "worker_result_sha256": sha256_file(
            Path(identity["state_root"]) / "worker_result.json"
        ),
        "source_snapshot_receipt_sha256": lock["source_snapshot_receipt_sha256"],
        "pre_natural_gate": lock["pre_natural_gate"],
        "deep_verifier": deep,
        "artifacts": {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                      for name, path in required.items()},
        "verified_at_utc": utc_now(),
    }
    publish_json_no_overwrite(receipt_path, receipt)
    return _deep_receipt_result(
        _validate_deep_verifier_receipt(lock, lock_path), receipt_path
    )


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
