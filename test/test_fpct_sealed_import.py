from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python3.10")
BOOTSTRAP = REPO / "script/runtime/fpct_bootstrap.py"
PROBE = REPO / "script/runtime/fpct_probe_target.py"
PROTOCOL_VERIFY = REPO / "script/analysis/fpct_r1_protocol_verify.py"


def _environment(**updates: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DATASETS_OFFLINE": "1",
        }
    )
    env.update(updates)
    return env


def _command(
    tmp_path: Path,
    *,
    python: str | Path = PYTHON,
    repo: str | Path = REPO,
    bootstrap: str | Path = BOOTSTRAP,
    target: str | Path = PROBE,
    extra_bootstrap: tuple[str, ...] = (),
    extra_target: tuple[str, ...] = (),
) -> tuple[list[str], Path, Path]:
    output = tmp_path / "probe.json"
    attestation = tmp_path / "attestation.json"
    command = [
        str(python),
        "-I",
        str(bootstrap),
        "--repo-root",
        str(repo),
        "--target",
        str(target),
        "--attestation-out",
        str(attestation),
        *extra_bootstrap,
        "--",
        "--output",
        str(output),
        *extra_target,
    ]
    return command, output, attestation


def _run(
    command: list[str],
    *,
    cwd: Path = REPO,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env or _environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )


def _success(tmp_path: Path, **kwargs):
    command, output, attestation = _command(tmp_path, **kwargs)
    result = _run(command)
    assert result.returncode == 0, result.stderr
    return json.loads(output.read_text()), json.loads(attestation.read_text())


def test_canonical_isolated_bootstrap_succeeds(tmp_path: Path) -> None:
    probe, attestation = _success(tmp_path)
    assert probe["status"] == "SEALED_SYNTHETIC_PROBE_OK"
    assert attestation["python"]["flags"] == {
        "ignore_environment": 1,
        "isolated": 1,
        "no_user_site": 1,
    }


def test_canonical_synthetic_exact_identity_probe(tmp_path: Path) -> None:
    probe, _ = _success(tmp_path)
    assert probe["eligible_parents"] > 0
    assert probe["extra_slots"] == 0


def test_protocol_diff_verifier_runs_under_canonical_bootstrap(
    tmp_path: Path,
) -> None:
    output = tmp_path / "protocol-diff.json"
    attestation = tmp_path / "attestation.json"
    command = [
        str(PYTHON), "-I", str(BOOTSTRAP),
        "--repo-root", str(REPO),
        "--target", str(PROTOCOL_VERIFY),
        "--attestation-out", str(attestation),
        "--", "--output", str(output),
    ]
    result = _run(command)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["scientific_change_count"] == 0
    assert "formal_target" in json.loads(attestation.read_text())["mandatory_modules"]


def test_direct_target_invocation_hard_fails(tmp_path: Path) -> None:
    result = _run(
        [str(PYTHON), "-I", str(PROBE), "--output", str(tmp_path / "x.json")]
    )
    assert result.returncode != 0
    assert (
        "canonical bootstrap" in result.stderr
        or "No module named 'fpct_bootstrap'" in result.stderr
    )


def test_nonisolated_bootstrap_hard_fails(tmp_path: Path) -> None:
    command, _, _ = _command(tmp_path)
    command.remove("-I")
    result = _run(command)
    assert result.returncode != 0
    assert "python -I" in result.stderr or "isolated mode" in result.stderr


@pytest.mark.parametrize("field", ["interpreter", "repo", "target"])
def test_relative_canonical_components_hard_fail(tmp_path: Path, field: str) -> None:
    relative_python = os.path.relpath(PYTHON, REPO)
    relative_repo = "."
    relative_target = os.path.relpath(PROBE, REPO)
    kwargs = {}
    if field == "interpreter":
        kwargs["python"] = relative_python
    elif field == "repo":
        kwargs["repo"] = relative_repo
    else:
        kwargs["target"] = relative_target
    command, _, _ = _command(tmp_path, **kwargs)
    result = _run(command)
    assert result.returncode != 0


def test_wrong_cwd_hard_fails(tmp_path: Path) -> None:
    command, _, _ = _command(tmp_path)
    result = _run(command, cwd=tmp_path)
    assert result.returncode != 0
    assert "cwd" in result.stderr


def test_old_editable_metadata_cannot_select_old_source(tmp_path: Path) -> None:
    probe, attestation = _success(tmp_path)
    assert probe["aligner_origin"].startswith(str(REPO / "rosetta"))
    assert attestation["distribution"]["installed"] is True
    assert attestation["distribution"]["direct_url"] is not None
    for record in attestation["mandatory_modules"].values():
        assert record["file"].startswith(str(REPO))


def test_pythonpath_fake_rosetta_is_ignored(tmp_path: Path) -> None:
    fake = tmp_path / "fake"
    (fake / "rosetta").mkdir(parents=True)
    (fake / "rosetta/__init__.py").write_text("raise RuntimeError('polluted')\n")
    command, output, attestation_path = _command(tmp_path / "run")
    (tmp_path / "run").mkdir()
    result = _run(command, env=_environment(PYTHONPATH=str(fake)))
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["status"] == "SEALED_SYNTHETIC_PROBE_OK"
    assert str(fake) not in json.loads(attestation_path.read_text())["process"]["sys_path"]


def test_second_namespace_cannot_extend_rosetta_path(tmp_path: Path) -> None:
    fake = tmp_path / "namespace"
    (fake / "rosetta/foreign").mkdir(parents=True)
    command, _, attestation_path = _command(tmp_path / "run")
    (tmp_path / "run").mkdir()
    result = _run(command, env=_environment(PYTHONPATH=str(fake)))
    assert result.returncode == 0, result.stderr
    attestation = json.loads(attestation_path.read_text())
    rosetta = next(
        row for row in attestation["loaded_rosetta_modules"]
        if row["name"] == "rosetta"
    )
    assert rosetta["path"] == [str((REPO / "rosetta").resolve())]


def test_missing_regular_package_init_hard_fails(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "script/runtime").mkdir(parents=True)
    (repo / "rosetta").mkdir()
    bootstrap = repo / "script/runtime/fpct_bootstrap.py"
    target = repo / "script/runtime/target.py"
    shutil.copy2(BOOTSTRAP, bootstrap)
    target.write_text("raise AssertionError('target must not execute')\n")
    command, _, _ = _command(
        tmp_path / "run", repo=repo, bootstrap=bootstrap, target=target
    )
    (tmp_path / "run").mkdir()
    result = _run(command, cwd=repo)
    assert result.returncode != 0
    assert "rosetta/__init__.py is required" in result.stderr


def test_locked_module_sha_mismatch_hard_fails(tmp_path: Path) -> None:
    command, _, _ = _command(
        tmp_path,
        extra_bootstrap=("--expected-module-sha", f"aligner={'0' * 64}"),
    )
    result = _run(command)
    assert result.returncode != 0
    assert "sealed module SHA mismatch" in result.stderr


def test_post_target_method_signature_mismatch_hard_fails(tmp_path: Path) -> None:
    command, output, _ = _command(
        tmp_path, extra_target=("--monkeypatch-signature",)
    )
    result = _run(command)
    assert output.is_file()
    assert result.returncode != 0
    assert "signature mismatch" in result.stderr


def test_fpct1b_delayed_imports_remain_in_current_root(tmp_path: Path) -> None:
    probe, _ = _success(tmp_path, extra_target=("--delayed-import",))
    assert probe["delayed_import_origins"]
    for origin in probe["delayed_import_origins"].values():
        assert origin.startswith(str(REPO / "rosetta"))


def test_freeze_shard_fingerprint_mismatch_hard_fails(tmp_path: Path) -> None:
    expected = tmp_path / "expected.json"
    expected.write_text(
        json.dumps({"stable_fingerprint_sha256": "0" * 64}) + "\n"
    )
    command, _, _ = _command(
        tmp_path / "run",
        extra_bootstrap=("--expected-attestation", str(expected)),
    )
    (tmp_path / "run").mkdir()
    result = _run(command)
    assert result.returncode != 0
    assert "fingerprint mismatch" in result.stderr


def test_synthetic_probe_reads_no_protected_natural_data(tmp_path: Path) -> None:
    _, attestation = _success(tmp_path)
    assert attestation["protected_data_opens_before_target"] == []
    assert attestation["protected_data_opens_after_target"] == []


@pytest.mark.parametrize("rank", [0, 1])
def test_every_distributed_rank_attests(tmp_path: Path, rank: int) -> None:
    probe, attestation = _success(
        tmp_path, extra_target=("--rank", str(rank))
    )
    assert probe["rank"] == rank
    assert len(attestation["stable_fingerprint_sha256"]) == 64
    assert attestation["post_target_attestation"][
        "stable_fingerprint_sha256"
    ] == attestation["stable_fingerprint_sha256"]
