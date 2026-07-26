from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
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


def _minimal_bootstrap_snapshot(
    tmp_path: Path,
) -> tuple[Path, Path, Path, str]:
    repo = tmp_path / "bootstrap-repo"
    repo.mkdir()
    _run(repo, "init", "-q")
    _run(repo, "config", "user.email", "fpct-test@example.invalid")
    _run(repo, "config", "user.name", "FPCT Test")
    for relative in (
        "script/runtime",
        "script/analysis",
        "rosetta/model",
        "rosetta/train",
        "rosetta/utils",
    ):
        (repo / relative).mkdir(parents=True, exist_ok=True)
    source_bootstrap = (
        Path(__file__).resolve().parents[1] / "script/runtime/fpct_bootstrap.py"
    )
    shutil.copy2(source_bootstrap, repo / "script/runtime/fpct_bootstrap.py")
    for relative in (
        "rosetta/__init__.py",
        "rosetta/model/__init__.py",
        "rosetta/train/__init__.py",
        "rosetta/utils/__init__.py",
        "rosetta/train/dataset_adapters.py",
        "rosetta/utils/evaluate.py",
        "script/analysis/fpct_1b_structural_support_audit.py",
        "script/analysis/fpct_3_5_alignment_correctness.py",
        "script/analysis/fpct_3_7_certified_support_audit.py",
    ):
        (repo / relative).write_text("# sealed fixture\n", encoding="utf-8")
    (repo / "rosetta/model/aligner.py").write_text(
        "class AlignmentStrategy:\n"
        "    EXACT_IDENTITY = 'exact_identity'\n\n"
        "class TokenAligner:\n"
        "    def align_chat_messages_soft(self, apply_confidence_control=False):\n"
        "        return None\n"
        "    def sanitize_fpct_soft_alignment(self):\n"
        "        return None\n",
        encoding="utf-8",
    )
    target = repo / "script/runtime/target.py"
    target.write_text("print('target executed')\n", encoding="utf-8")
    _run(repo, "add", ".")
    _run(repo, "commit", "-q", "-m", "bootstrap fixture")
    execution_sha = _run(repo, "rev-parse", "HEAD")
    snapshot = _archive(repo, execution_sha, tmp_path / "bootstrap-snapshot")
    receipt = snapshot / DEFAULT_RECEIPT_NAME
    create_source_snapshot_lock(
        repo=repo,
        execution_sha=execution_sha,
        snapshot_root=snapshot,
        output_path=receipt,
    )
    return snapshot, snapshot / "script/runtime/target.py", receipt, execution_sha


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
    from script.runtime import fpct_bootstrap

    bootstrap_record = fpct_bootstrap._stdlib_verify_source_snapshot(snapshot)
    assert bootstrap_record["head"] == execution_sha
    assert bootstrap_record["source"] == "fpct_e1_source_snapshot_receipt"
    assert (
        bootstrap_record["tree_sha256"]
        == mounted["mounted_tree_canonical_sha256"]
    )
    full = verify_source_snapshot_lock(
        repo=repo,
        execution_sha=execution_sha,
        snapshot_root=snapshot,
        receipt_path=receipt,
    )
    assert full["status"] == "GO_REPO_AND_MOUNTED_SOURCE_SNAPSHOT"


def test_bootstrap_rejects_tampered_closure_before_top_level_executes(
    tmp_path: Path,
) -> None:
    snapshot, target, _receipt, _execution_sha = _minimal_bootstrap_snapshot(
        tmp_path
    )
    marker = tmp_path / "tampered-closure-executed.txt"
    aligner = snapshot / "rosetta/model/aligner.py"
    aligner.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('EXECUTED')\n"
        "raise RuntimeError('tampered closure imported')\n",
        encoding="utf-8",
    )
    bootstrap = snapshot / "script/runtime/fpct_bootstrap.py"
    result = subprocess.run(
        [
            str(Path(sys.executable).resolve()),
            "-I",
            str(bootstrap),
            "--repo-root",
            str(snapshot),
            "--target",
            str(target),
        ],
        cwd=snapshot,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0
    assert "snapshot tree differs" in result.stderr
    assert not marker.exists()
    assert "tampered closure imported" not in result.stderr


@pytest.mark.parametrize(
    "receipt_mutation",
    ["missing", "broken_symlink", "directory", "fifo"],
)
def test_bootstrap_rejects_noncanonical_or_missing_provenance_before_closure_exec(
    tmp_path: Path,
    receipt_mutation: str,
) -> None:
    snapshot, target, receipt, _execution_sha = _minimal_bootstrap_snapshot(
        tmp_path
    )
    marker = tmp_path / f"closure-executed-{receipt_mutation}.txt"
    (snapshot / "rosetta/model/aligner.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('EXECUTED')\n"
        "raise RuntimeError('unverified closure imported')\n",
        encoding="utf-8",
    )
    receipt.unlink()
    if receipt_mutation == "broken_symlink":
        receipt.symlink_to("missing-source-snapshot-receipt.json")
    elif receipt_mutation == "directory":
        receipt.mkdir()
    elif receipt_mutation == "fifo":
        os.mkfifo(receipt)

    bootstrap = snapshot / "script/runtime/fpct_bootstrap.py"
    result = subprocess.run(
        [
            str(Path(sys.executable).resolve()),
            "-I",
            str(bootstrap),
            "--repo-root",
            str(snapshot),
            "--target",
            str(target),
        ],
        cwd=snapshot,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0
    if receipt_mutation == "missing":
        assert "requires exactly one provenance marker" in result.stderr
    else:
        assert "is not a canonical regular file" in result.stderr
    assert not marker.exists()
    assert "unverified closure imported" not in result.stderr


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
