#!/usr/bin/env python3
"""Seal and independently verify an immutable FPCT-E1 Git source snapshot.

The snapshot itself must already have been materialized from ``git archive``.
This tool never copies source files.  It proves that a clean repository HEAD,
the Git tree, the archive byte manifest, and the mounted snapshot are identical,
then emits a self-hashed portable receipt without overwriting an existing lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
PROTOCOL_ID = "fpct_e1_source_snapshot_lock_v1"
DEFAULT_RECEIPT_NAME = ".fpct_e1_source_snapshot_receipt.json"
EXECUTION_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
GIT_OBJECT_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")
GIT_FILE_MODES = {"100644", "100755", "120000"}
ENTRY_COLUMNS = (
    "path",
    "mode",
    "object_type",
    "object_id",
    "bytes",
    "sha256",
)
MOUNTED_ENTRY_COLUMNS = ("path", "mode", "kind", "bytes", "sha256")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _safe_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("snapshot entry path is empty or malformed")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe snapshot entry path: {value!r}")
    if str(path) != value:
        raise ValueError(f"snapshot entry path is not canonical: {value!r}")
    return value


def _git(repo: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    environment = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    process = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        check=False,
    )
    if process.returncode:
        detail = process.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return process.stdout


def _repo_identity(repo: Path, execution_sha: str) -> dict[str, Any]:
    if not EXECUTION_SHA_PATTERN.fullmatch(execution_sha):
        raise ValueError("execution_sha must be one lowercase 40-character Git SHA")
    repo = repo.resolve(strict=True)
    top = Path(
        _git(repo, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    ).resolve(strict=True)
    if top != repo:
        raise ValueError("--repo must name the Git worktree root")
    head = _git(repo, "rev-parse", "HEAD").decode("ascii").strip()
    if head != execution_sha:
        raise ValueError(f"repo HEAD {head} differs from execution_sha {execution_sha}")
    status = _git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if status:
        raise ValueError("source repository worktree is dirty")
    tree_oid = _git(repo, "rev-parse", f"{execution_sha}^{{tree}}").decode(
        "ascii"
    ).strip()
    object_format = _git(repo, "rev-parse", "--show-object-format").decode(
        "ascii"
    ).strip()
    if not GIT_OBJECT_PATTERN.fullmatch(tree_oid):
        raise ValueError("Git tree object ID is malformed")
    return {
        "head": head,
        "dirty": False,
        "git_tree_oid": tree_oid,
        "object_format": object_format,
    }


def _ls_tree_rows(repo: Path, execution_sha: str) -> list[dict[str, Any]]:
    payload = _git(
        repo,
        "-c",
        "core.quotePath=false",
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        "--long",
        execution_sha,
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in payload.split(b"\x00"):
        if not raw:
            continue
        try:
            header, path_bytes = raw.split(b"\t", 1)
            mode, object_type, object_id, size_text = header.split()
        except ValueError as error:
            raise ValueError("git ls-tree emitted a malformed entry") from error
        path = _safe_path(path_bytes.decode("utf-8", "strict"))
        mode_text = mode.decode("ascii")
        type_text = object_type.decode("ascii")
        oid_text = object_id.decode("ascii")
        if path in seen:
            raise ValueError("git ls-tree emitted a duplicate path")
        seen.add(path)
        if mode_text not in GIT_FILE_MODES or type_text != "blob":
            raise ValueError(
                f"unsupported Git entry {path}: mode={mode_text}, type={type_text}"
            )
        if not GIT_OBJECT_PATTERN.fullmatch(oid_text):
            raise ValueError("git ls-tree emitted a malformed object ID")
        try:
            size = int(size_text)
        except ValueError as error:
            raise ValueError("git ls-tree blob size is unavailable") from error
        rows.append(
            {
                "path": path,
                "mode": mode_text,
                "object_type": type_text,
                "object_id": oid_text,
                "bytes": size,
            }
        )
    if not rows:
        raise ValueError("Git source tree is empty")
    return sorted(rows, key=lambda row: row["path"])


def _attach_blob_sha256(repo: Path, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    environment = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    process = subprocess.Popen(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    assert process.stdin is not None and process.stdout is not None
    output: list[dict[str, Any]] = []
    try:
        for row in rows:
            process.stdin.write((str(row["object_id"]) + "\n").encode("ascii"))
            process.stdin.flush()
            header = process.stdout.readline().rstrip(b"\n").split()
            if len(header) != 3:
                raise ValueError("git cat-file emitted a malformed header")
            oid, object_type, size_text = header
            size = int(size_text)
            digest = hashlib.sha256()
            remaining = size
            observed_size = 0
            while remaining:
                block = process.stdout.read(min(1024 * 1024, remaining))
                if not block:
                    break
                observed_size += len(block)
                remaining -= len(block)
                digest.update(block)
            delimiter = process.stdout.read(1)
            if (
                oid.decode("ascii") != row["object_id"]
                or object_type != b"blob"
                or size != row["bytes"]
                or observed_size != size
                or delimiter != b"\n"
            ):
                raise ValueError("git cat-file blob differs from git ls-tree")
            output.append({**dict(row), "sha256": digest.hexdigest()})
        process.stdin.close()
        return_code = process.wait()
        stderr = (process.stderr.read() if process.stderr is not None else b"").decode(
            "utf-8", "replace"
        )
        if return_code:
            raise RuntimeError(f"git cat-file failed: {stderr.strip()}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return output


def _archive_manifest(repo: Path, execution_sha: str) -> list[dict[str, Any]]:
    environment = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    process = subprocess.Popen(
        ["git", "-C", str(repo), "archive", "--format=tar", execution_sha],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    assert process.stdout is not None
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                name = member.name.rstrip("/")
                if member.isdir():
                    if name:
                        _safe_path(name)
                    continue
                path = _safe_path(name)
                if path in seen:
                    raise ValueError("git archive contains a duplicate path")
                seen.add(path)
                if member.isfile():
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError("git archive regular file has no content")
                    digest = hashlib.sha256()
                    size = 0
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        size += len(block)
                        digest.update(block)
                    mode = "100755" if member.mode & 0o111 else "100644"
                    kind = "file"
                elif member.issym():
                    target = member.linkname.encode("utf-8", "surrogateescape")
                    size = len(target)
                    digest = hashlib.sha256(target)
                    mode = "120000"
                    kind = "symlink"
                else:
                    raise ValueError(f"git archive contains unsupported entry: {path}")
                rows.append(
                    {
                        "path": path,
                        "mode": mode,
                        "kind": kind,
                        "bytes": size,
                        "sha256": digest.hexdigest(),
                    }
                )
        while process.stdout.read(1024 * 1024):
            pass
        return_code = process.wait()
        stderr = (process.stderr.read() if process.stderr is not None else b"").decode(
            "utf-8", "replace"
        )
        if return_code:
            raise RuntimeError(f"git archive failed: {stderr.strip()}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return sorted(rows, key=lambda row: row["path"])


def _expected_mounted_entries(
    git_entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "path": row["path"],
            "mode": row["mode"],
            "kind": "symlink" if row["mode"] == "120000" else "file",
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in git_entries
    ]


def _compare_archive_and_tree(
    git_entries: Sequence[Mapping[str, Any]],
    archive_entries: Sequence[Mapping[str, Any]],
) -> None:
    expected = _expected_mounted_entries(git_entries)
    if list(archive_entries) != expected:
        raise ValueError(
            "git archive content differs from the complete tracked Git tree; "
            "export-ignore/export-subst or unsupported attributes are forbidden"
        )


def _relative_receipt_path(
    receipt_path: Path, snapshot_root: Path
) -> str | None:
    snapshot = snapshot_root.resolve(strict=True)
    candidate = receipt_path.absolute()
    try:
        relative = candidate.relative_to(snapshot)
    except ValueError:
        return None
    return _safe_path(relative.as_posix())


def _symlink_target_bytes(path: Path) -> bytes:
    return os.fsencode(os.readlink(path))


def _assert_symlink_confined(path: Path, root: Path) -> None:
    target = os.readlink(path)
    if os.path.isabs(target):
        raise ValueError(f"snapshot symlink has an absolute target: {path}")
    resolved = (path.parent / target).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"snapshot symlink escapes mounted root: {path}") from error


def _snapshot_manifest(
    snapshot_root: Path,
    expected_entries: Sequence[Mapping[str, Any]],
    receipt_relative_path: str | None,
) -> dict[str, Any]:
    if snapshot_root.is_symlink():
        raise ValueError("snapshot root may not be a symlink")
    root = snapshot_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("snapshot root is not a directory")
    expected_by_path = {row["path"]: dict(row) for row in expected_entries}
    expected_directories: set[str] = set()
    for path in expected_by_path:
        parent = PurePosixPath(path).parent
        while str(parent) != ".":
            expected_directories.add(str(parent))
            parent = parent.parent

    actual: dict[str, dict[str, Any]] = {}
    actual_directories: set[str] = set()
    stack = [root]
    while stack:
        directory = stack.pop()
        for entry in os.scandir(directory):
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            _safe_path(relative)
            if receipt_relative_path is not None and relative == receipt_relative_path:
                if not entry.is_file(follow_symlinks=False) or entry.is_symlink():
                    raise ValueError("snapshot receipt path is not a regular file")
                continue
            if entry.is_symlink():
                _assert_symlink_confined(path, root)
                target = _symlink_target_bytes(path)
                actual[relative] = {
                    "path": relative,
                    "mode": "120000",
                    "kind": "symlink",
                    "bytes": len(target),
                    "sha256": sha256_bytes(target),
                }
            elif entry.is_dir(follow_symlinks=False):
                actual_directories.add(relative)
                stack.append(path)
            elif entry.is_file(follow_symlinks=False):
                metadata = entry.stat(follow_symlinks=False)
                if metadata.st_nlink != 1:
                    raise ValueError(f"snapshot regular file is hard-linked: {relative}")
                permissions = stat.S_IMODE(metadata.st_mode)
                if permissions & 0o7000:
                    raise ValueError(f"snapshot regular file has special mode bits: {relative}")
                mode = "100755" if permissions & 0o111 else "100644"
                size, digest = sha256_file(path)
                actual[relative] = {
                    "path": relative,
                    "mode": mode,
                    "kind": "file",
                    "bytes": size,
                    "sha256": digest,
                }
            else:
                raise ValueError(f"snapshot contains a special filesystem entry: {relative}")

    if actual_directories != expected_directories:
        missing = sorted(expected_directories - actual_directories)
        extra = sorted(actual_directories - expected_directories)
        raise ValueError(f"snapshot directory universe differs; missing={missing}, extra={extra}")
    if set(actual) != set(expected_by_path):
        missing = sorted(set(expected_by_path) - set(actual))
        extra = sorted(set(actual) - set(expected_by_path))
        raise ValueError(f"snapshot path universe differs; missing={missing}, extra={extra}")
    for path, observed in actual.items():
        if observed != expected_by_path[path]:
            raise ValueError(f"snapshot file/mode/bytes differs from Git archive: {path}")
    mounted_entries = [actual[path] for path in sorted(actual)]
    return {
        "mounted_tree_canonical_sha256": sha256_bytes(
            canonical_json_bytes(mounted_entries)
        ),
        "file_count": sum(row["kind"] == "file" for row in mounted_entries),
        "symlink_count": sum(row["kind"] == "symlink" for row in mounted_entries),
        "entry_count": len(mounted_entries),
        "total_bytes": sum(int(row["bytes"]) for row in mounted_entries),
    }


def _freeze_git_tree(repo: Path, execution_sha: str) -> dict[str, Any]:
    identity = _repo_identity(repo, execution_sha)
    entries = _attach_blob_sha256(repo, _ls_tree_rows(repo, execution_sha))
    archive = _archive_manifest(repo, execution_sha)
    _compare_archive_and_tree(entries, archive)
    return {
        "repo": {
            **identity,
            "ls_tree_entry_count": len(entries),
            "ls_tree_canonical_sha256": sha256_bytes(canonical_json_bytes(entries)),
        },
        "git_entries": entries,
        "git_entries_canonical_sha256": sha256_bytes(canonical_json_bytes(entries)),
    }


def receipt_sha256(receipt: Mapping[str, Any]) -> str:
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    return sha256_bytes(canonical_json_bytes(payload))


def _atomic_create(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(path) from None
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def create_source_snapshot_lock(
    *,
    repo: Path,
    execution_sha: str,
    snapshot_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate a clean Git archive snapshot and atomically seal its receipt."""

    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    snapshot = snapshot_root.resolve(strict=True)
    receipt_relative_path = _relative_receipt_path(output_path, snapshot)
    frozen = _freeze_git_tree(repo.resolve(strict=True), execution_sha)
    expected_mounted = _expected_mounted_entries(frozen["git_entries"])
    mounted = _snapshot_manifest(snapshot, expected_mounted, receipt_relative_path)
    # Repeat both mutable-state checks immediately before sealing the receipt.
    repeated = _freeze_git_tree(repo.resolve(strict=True), execution_sha)
    if repeated != frozen:
        raise ValueError("Git source tree changed while source snapshot was being locked")
    if _snapshot_manifest(snapshot, expected_mounted, receipt_relative_path) != mounted:
        raise ValueError("mounted source snapshot changed while it was being locked")
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "GO_IMMUTABLE_GIT_ARCHIVE",
        "execution_sha": execution_sha,
        "dirty": False,
        "portable_root": ".",
        "repo": frozen["repo"],
        "git_entries": frozen["git_entries"],
        "git_entries_canonical_sha256": frozen[
            "git_entries_canonical_sha256"
        ],
        "snapshot": {
            **mounted,
            "portable_root": ".",
            "receipt_relative_path": receipt_relative_path,
            "exact_git_archive_content": True,
            "no_extra_paths": True,
            "no_symlink_escape": True,
        },
        "construction": {
            "source": "git archive <execution_sha>",
            "git_ls_tree": "git ls-tree -r -z --full-tree --long <execution_sha>",
            "manual_copy_allowed": False,
            "worktree_clean_required": True,
        },
    }
    receipt["receipt_sha256"] = receipt_sha256(receipt)
    _atomic_create(output_path, canonical_json_bytes(receipt))
    return receipt


def _load_receipt(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("source snapshot receipt may not be a symlink")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("source snapshot receipt is not a JSON object")
    return value


def _validated_receipt_entries(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = receipt.get("git_entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("source snapshot receipt has no Git entries")
    output: list[dict[str, Any]] = []
    previous: str | None = None
    for raw in entries:
        if not isinstance(raw, Mapping) or set(raw) != set(ENTRY_COLUMNS):
            raise ValueError("source snapshot receipt Git entry schema differs")
        row = {name: raw[name] for name in ENTRY_COLUMNS}
        row["path"] = _safe_path(row["path"])
        if previous is not None and row["path"] <= previous:
            raise ValueError("source snapshot receipt Git entries are not sorted/unique")
        previous = row["path"]
        if row["mode"] not in GIT_FILE_MODES or row["object_type"] != "blob":
            raise ValueError("source snapshot receipt Git entry type/mode differs")
        if not GIT_OBJECT_PATTERN.fullmatch(str(row["object_id"])):
            raise ValueError("source snapshot receipt object ID is malformed")
        if (
            isinstance(row["bytes"], bool)
            or int(row["bytes"]) != row["bytes"]
            or int(row["bytes"]) < 0
        ):
            raise ValueError("source snapshot receipt byte size is malformed")
        row["bytes"] = int(row["bytes"])
        if not _is_sha256(row["sha256"]):
            raise ValueError("source snapshot receipt blob SHA256 is malformed")
        output.append(row)
    canonical = sha256_bytes(canonical_json_bytes(output))
    if canonical != receipt.get("git_entries_canonical_sha256"):
        raise ValueError("source snapshot receipt Git-entry canonical SHA differs")
    repo = receipt.get("repo")
    if not isinstance(repo, Mapping):
        raise ValueError("source snapshot receipt repo record is malformed")
    if (
        repo.get("ls_tree_entry_count") != len(output)
        or repo.get("ls_tree_canonical_sha256") != canonical
    ):
        raise ValueError("source snapshot receipt ls-tree attestation differs")
    return output


def verify_source_snapshot_receipt(
    receipt_path: Path,
    snapshot_root: Path,
    expected_execution_sha: str,
) -> dict[str, Any]:
    """Verify a mounted snapshot using only its sealed portable receipt."""

    receipt = _load_receipt(receipt_path)
    expected_top_level = {
        "schema_version",
        "protocol_id",
        "status",
        "execution_sha",
        "dirty",
        "portable_root",
        "repo",
        "git_entries",
        "git_entries_canonical_sha256",
        "snapshot",
        "construction",
        "receipt_sha256",
    }
    if set(receipt) != expected_top_level:
        raise ValueError("source snapshot receipt top-level schema differs")
    if (
        not EXECUTION_SHA_PATTERN.fullmatch(expected_execution_sha)
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("status") != "GO_IMMUTABLE_GIT_ARCHIVE"
        or receipt.get("execution_sha") != expected_execution_sha
        or receipt.get("dirty") is not False
        or receipt.get("portable_root") != "."
    ):
        raise ValueError("source snapshot receipt identity/status differs")
    if receipt.get("receipt_sha256") != receipt_sha256(receipt):
        raise ValueError("source snapshot receipt self-hash differs")
    entries = _validated_receipt_entries(receipt)
    snapshot_record = receipt.get("snapshot")
    expected_snapshot_fields = {
        "mounted_tree_canonical_sha256",
        "file_count",
        "symlink_count",
        "entry_count",
        "total_bytes",
        "portable_root",
        "receipt_relative_path",
        "exact_git_archive_content",
        "no_extra_paths",
        "no_symlink_escape",
    }
    if not isinstance(snapshot_record, Mapping) or set(snapshot_record) != expected_snapshot_fields:
        raise ValueError("source snapshot mounted-tree record is malformed")
    if (
        snapshot_record.get("portable_root") != "."
        or not _is_sha256(snapshot_record.get("mounted_tree_canonical_sha256"))
        or any(
            not isinstance(snapshot_record.get(name), int)
            or isinstance(snapshot_record.get(name), bool)
            or snapshot_record.get(name) < 0
            for name in (
                "file_count",
                "symlink_count",
                "entry_count",
                "total_bytes",
            )
        )
    ):
        raise ValueError("source snapshot mounted-tree counters/root are malformed")
    repo_record = receipt["repo"]
    if (
        not isinstance(repo_record, Mapping)
        or set(repo_record)
        != {
            "head",
            "dirty",
            "git_tree_oid",
            "object_format",
            "ls_tree_entry_count",
            "ls_tree_canonical_sha256",
        }
        or repo_record.get("head") != expected_execution_sha
        or repo_record.get("dirty") is not False
        or not GIT_OBJECT_PATTERN.fullmatch(str(repo_record.get("git_tree_oid")))
        or not _is_sha256(repo_record.get("ls_tree_canonical_sha256"))
    ):
        raise ValueError("source snapshot receipt repo/tree record is malformed")
    receipt_relative_path = _relative_receipt_path(receipt_path, snapshot_root)
    if receipt_relative_path != snapshot_record.get("receipt_relative_path"):
        raise ValueError("source snapshot receipt portable location differs")
    observed = _snapshot_manifest(
        snapshot_root,
        _expected_mounted_entries(entries),
        receipt_relative_path,
    )
    expected_mounted = {
        name: snapshot_record.get(name)
        for name in (
            "mounted_tree_canonical_sha256",
            "file_count",
            "symlink_count",
            "entry_count",
            "total_bytes",
        )
    }
    if observed != expected_mounted:
        raise ValueError("source snapshot mounted-tree attestation differs")
    if not all(
        snapshot_record.get(name) is True
        for name in (
            "exact_git_archive_content",
            "no_extra_paths",
            "no_symlink_escape",
        )
    ):
        raise ValueError("source snapshot safety attestation is absent")
    if receipt.get("construction") != {
        "source": "git archive <execution_sha>",
        "git_ls_tree": "git ls-tree -r -z --full-tree --long <execution_sha>",
        "manual_copy_allowed": False,
        "worktree_clean_required": True,
    }:
        raise ValueError("source snapshot construction contract differs")
    return {
        "status": "GO_MOUNTED_SOURCE_SNAPSHOT",
        "execution_sha": expected_execution_sha,
        "git_tree_oid": receipt["repo"]["git_tree_oid"],
        "git_entries_canonical_sha256": receipt[
            "git_entries_canonical_sha256"
        ],
        "mounted_tree_canonical_sha256": observed[
            "mounted_tree_canonical_sha256"
        ],
        "entry_count": observed["entry_count"],
        "total_bytes": observed["total_bytes"],
        "receipt_sha256": receipt["receipt_sha256"],
    }


def verify_source_snapshot_lock(
    *,
    repo: Path,
    execution_sha: str,
    snapshot_root: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Independently recompute both repository Git state and mounted snapshot."""

    mounted = verify_source_snapshot_receipt(
        receipt_path, snapshot_root, execution_sha
    )
    receipt = _load_receipt(receipt_path)
    frozen = _freeze_git_tree(repo.resolve(strict=True), execution_sha)
    if receipt.get("repo") != frozen["repo"]:
        raise ValueError("source snapshot receipt repo/tree record differs from Git")
    if receipt.get("git_entries") != frozen["git_entries"]:
        raise ValueError("source snapshot receipt Git entries differ from Git")
    if (
        receipt.get("git_entries_canonical_sha256")
        != frozen["git_entries_canonical_sha256"]
    ):
        raise ValueError("source snapshot receipt Git-entry SHA differs from Git")
    return {**mounted, "status": "GO_REPO_AND_MOUNTED_SOURCE_SNAPSHOT"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--repo", type=Path, required=True)
    create.add_argument("--execution-sha", required=True)
    create.add_argument("--snapshot", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--repo", type=Path, required=True)
    verify.add_argument("--execution-sha", required=True)
    verify.add_argument("--snapshot", type=Path, required=True)
    verify.add_argument("--receipt", type=Path, required=True)
    mounted = subparsers.add_parser("verify-mounted")
    mounted.add_argument("--execution-sha", required=True)
    mounted.add_argument("--snapshot", type=Path, required=True)
    mounted.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "create":
        result = create_source_snapshot_lock(
            repo=args.repo,
            execution_sha=args.execution_sha,
            snapshot_root=args.snapshot,
            output_path=args.output,
        )
    elif args.command == "verify":
        result = verify_source_snapshot_lock(
            repo=args.repo,
            execution_sha=args.execution_sha,
            snapshot_root=args.snapshot,
            receipt_path=args.receipt,
        )
    else:
        result = verify_source_snapshot_receipt(
            args.receipt, args.snapshot, args.execution_sha
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
