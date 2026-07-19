from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import pytest

import script.k8s.phase2a_2a_cache_geometry_jobs as jobs


COMMIT = "1" * 40


def _manifest(path: Path) -> None:
    runs = []
    for pair in ("tinyllama", "qwen25_0p5b", "llama32_1b"):
        for dataset in ("ai2-arc", "openbookqa", "mmlu-redux"):
            runs.append(
                {
                    "pair": pair,
                    "dataset": dataset,
                    "seed": 42,
                    "kind": "instrumented",
                    "training_forbidden": True,
                }
            )
    runs.append(
        {
            "pair": "tinyllama",
            "dataset": "ai2-arc",
            "seed": 42,
            "kind": "overhead_control",
            "training_forbidden": True,
        }
    )
    path.write_text(
        json.dumps(
            {
                "phase": "Phase 2A-2a",
                "code_commit": COMMIT,
                "constraints": {
                    "training_forbidden": True,
                    "allowed_seed": [42],
                    "allowed_split": "fit",
                },
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )


def test_build_jobs_isolated_nodes_and_no_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    _manifest(manifest)
    monkeypatch.setattr(jobs, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(jobs, "WORKSPACE_ROOT", PurePosixPath("/netdisk/test/Cache"))
    rendered, summary = jobs.build_jobs(
        git_commit=COMMIT,
        manifest_path=Path("manifest.json"),
        uid=1000,
        gid=1000,
    )
    assert summary["run_count"] == 10
    assert {job["spec"]["template"]["spec"]["nodeSelector"]["kubernetes.io/hostname"] for job in rendered} == {
        "4090-24gx8", "4090-24gx4"
    }
    serialized = jobs.serialize_jobs(rendered)
    assert "SFT_train.py" not in serialized
    assert "run-pair" not in serialized  # node runner owns the only eval commands
    assert "phase2a_2a_cache_geometry_jobs.py" in serialized


def test_manifest_rejects_non_fit_scope(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _manifest(manifest)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["constraints"]["allowed_split"] = "test"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(jobs.GeometryJobError, match="seed/split"):
        jobs.validate_manifest(manifest, COMMIT)

