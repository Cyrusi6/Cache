from __future__ import annotations

import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from script.experiment.fpct_e1_source_snapshot_lock import (
    DEFAULT_RECEIPT_NAME,
    PROTOCOL_ID,
    create_source_snapshot_lock,
    main,
    receipt_sha256,
    verify_source_snapshot_lock,
    verify_source_snapshot_receipt,
)


def _run(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _repository(tmp_path: Path, *, escaping_symlink: bool = False) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _run(repo, "init", "-q")
    _run(repo, "config", "user.email", "fpct-test@example.invalid")
    _run(repo, "config", "user.name", "FPCT Test")
    (repo / "nested").mkdir()
    (repo / "plain.txt").write_bytes(b"plain\n")
    executable = repo / "run.sh"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    (repo / "nested" / "payload.bin").write_bytes(b"\x00\x01fpct\xff")
    target = "../../outside" if escaping_symlink else "nested/payload.bin"
    os.symlink(target, repo / "payload-link")
    _run(repo, "add", ".")
    _run(repo, "commit", "-q", "-m", "fixture")
    return repo, _run(repo, "rev-parse", "HEAD")


def _archive(repo: Path, execution_sha: str, destination: Path) -> Path:
    tar_path = destination.parent / f"{destination.name}.tar"
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "archive",
            "--format=tar",
            f"--output={tar_path}",
            execution_sha,
        ],
        check=True,
    )
    destination.mkdir()
    with tarfile.open(tar_path, "r:") as archive:
        archive.extractall(destination)
    return destination


def _sealed(tmp_path: Path) -> tuple[Path, str, Path, Path, dict]:
    repo, execution_sha = _repository(tmp_path)
    snapshot = _archive(repo, execution_sha, tmp_path / "snapshot")
    receipt = snapshot / DEFAULT_RECEIPT_NAME
    record = create_source_snapshot_lock(
        repo=repo,
        execution_sha=execution_sha,
        snapshot_root=snapshot,
        output_path=receipt,
    )
    return repo, execution_sha, snapshot, receipt, record


def test_create_and_both_verifiers_freeze_exact_archive(tmp_path: Path) -> None:
    repo, execution_sha, snapshot, receipt, record = _sealed(tmp_path)
    assert record["protocol_id"] == PROTOCOL_ID
    assert record["execution_sha"] == execution_sha
    assert record["dirty"] is False
    assert record["portable_root"] == "."
    assert record["snapshot"]["receipt_relative_path"] == DEFAULT_RECEIPT_NAME
    assert record["snapshot"]["entry_count"] == 4
    assert record["receipt_sha256"] == receipt_sha256(record)
    assert str(repo) not in receipt.read_text()
    assert str(snapshot) not in receipt.read_text()
    modes = {row["path"]: row["mode"] for row in record["git_entries"]}
    assert modes["plain.txt"] == "100644"
    assert modes["run.sh"] == "100755"
    assert modes["payload-link"] == "120000"

    mounted = verify_source_snapshot_receipt(receipt, snapshot, execution_sha)
    assert mounted["status"] == "GO_MOUNTED_SOURCE_SNAPSHOT"
    full = verify_source_snapshot_lock(
        repo=repo,
        execution_sha=execution_sha,
        snapshot_root=snapshot,
        receipt_path=receipt,
    )
    assert full["status"] == "GO_REPO_AND_MOUNTED_SOURCE_SNAPSHOT"


def test_create_rejects_dirty_wrong_head_manual_copy_and_overwrite(
    tmp_path: Path,
) -> None:
    repo, execution_sha = _repository(tmp_path)
    snapshot = _archive(repo, execution_sha, tmp_path / "snapshot")
    receipt = snapshot / DEFAULT_RECEIPT_NAME

    (repo / "plain.txt").write_text("dirty\n")
    with pytest.raises(ValueError, match="dirty"):
        create_source_snapshot_lock(
            repo=repo,
            execution_sha=execution_sha,
            snapshot_root=snapshot,
            output_path=receipt,
        )
    _run(repo, "restore", "plain.txt")
    with pytest.raises(ValueError, match="differs from execution_sha"):
        create_source_snapshot_lock(
            repo=repo,
            execution_sha="0" * 40,
            snapshot_root=snapshot,
            output_path=receipt,
        )

    (snapshot / "nested" / "payload.bin").unlink()
    with pytest.raises(ValueError, match="path universe differs"):
        create_source_snapshot_lock(
            repo=repo,
            execution_sha=execution_sha,
            snapshot_root=snapshot,
            output_path=receipt,
        )
    snapshot = _archive(repo, execution_sha, tmp_path / "snapshot-clean")
    receipt = snapshot / DEFAULT_RECEIPT_NAME
    first = create_source_snapshot_lock(
        repo=repo,
        execution_sha=execution_sha,
        snapshot_root=snapshot,
        output_path=receipt,
    )
    first_bytes = receipt.read_bytes()
    with pytest.raises(FileExistsError):
        create_source_snapshot_lock(
            repo=repo,
            execution_sha=execution_sha,
            snapshot_root=snapshot,
            output_path=receipt,
        )
    assert receipt.read_bytes() == first_bytes
    assert json.loads(first_bytes)["receipt_sha256"] == first["receipt_sha256"]


@pytest.mark.parametrize("mutation", ["bytes", "mode", "extra", "receipt"])
def test_mounted_verify_rejects_every_post_lock_mutation(
    tmp_path: Path, mutation: str
) -> None:
    _repo, execution_sha, snapshot, receipt, _record = _sealed(tmp_path)
    if mutation == "bytes":
        (snapshot / "plain.txt").write_bytes(b"tampered\n")
    elif mutation == "mode":
        (snapshot / "plain.txt").chmod(0o755)
    elif mutation == "extra":
        (snapshot / "extra.txt").write_text("extra\n")
    else:
        value = json.loads(receipt.read_text())
        value["status"] = "TAMPERED"
        receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        verify_source_snapshot_receipt(receipt, snapshot, execution_sha)


def test_symlink_escape_and_dirty_repo_after_seal_are_hard_failures(
    tmp_path: Path,
) -> None:
    escape_repo, escape_sha = _repository(
        tmp_path / "escape-case", escaping_symlink=True
    )
    escape_snapshot = _archive(
        escape_repo, escape_sha, tmp_path / "escape-case" / "snapshot"
    )
    with pytest.raises(ValueError, match="escapes mounted root"):
        create_source_snapshot_lock(
            repo=escape_repo,
            execution_sha=escape_sha,
            snapshot_root=escape_snapshot,
            output_path=escape_snapshot / DEFAULT_RECEIPT_NAME,
        )

    clean_root = tmp_path / "dirty-after"
    clean_root.mkdir()
    repo, execution_sha, snapshot, receipt, _record = _sealed(clean_root)
    (repo / "untracked.txt").write_text("dirty\n")
    with pytest.raises(ValueError, match="dirty"):
        verify_source_snapshot_lock(
            repo=repo,
            execution_sha=execution_sha,
            snapshot_root=snapshot,
            receipt_path=receipt,
        )


def test_git_archive_attributes_cannot_hide_tracked_tree_entries(
    tmp_path: Path,
) -> None:
    repo, _old_sha = _repository(tmp_path)
    (repo / ".gitattributes").write_text("plain.txt export-ignore\n")
    _run(repo, "add", ".gitattributes")
    _run(repo, "commit", "-q", "-m", "export-ignore")
    execution_sha = _run(repo, "rev-parse", "HEAD")
    snapshot = _archive(repo, execution_sha, tmp_path / "snapshot")
    with pytest.raises(ValueError, match="git archive content differs"):
        create_source_snapshot_lock(
            repo=repo,
            execution_sha=execution_sha,
            snapshot_root=snapshot,
            output_path=snapshot / DEFAULT_RECEIPT_NAME,
        )


def test_cli_create_verify_and_mounted_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo, execution_sha = _repository(tmp_path)
    snapshot = _archive(repo, execution_sha, tmp_path / "snapshot")
    receipt = snapshot / DEFAULT_RECEIPT_NAME
    assert main(
        [
            "create",
            "--repo",
            str(repo),
            "--execution-sha",
            execution_sha,
            "--snapshot",
            str(snapshot),
            "--output",
            str(receipt),
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["receipt_sha256"]
    assert main(
        [
            "verify",
            "--repo",
            str(repo),
            "--execution-sha",
            execution_sha,
            "--snapshot",
            str(snapshot),
            "--receipt",
            str(receipt),
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"].startswith("GO_REPO")
    assert main(
        [
            "verify-mounted",
            "--execution-sha",
            execution_sha,
            "--snapshot",
            str(snapshot),
            "--receipt",
            str(receipt),
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"].startswith("GO_MOUNTED")
