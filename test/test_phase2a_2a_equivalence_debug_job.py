from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import pytest

from script.k8s import phase2a_2a_equivalence_debug_job as jobs


COMMIT = "1" * 40


def _manifest(path: Path, workspace: PurePosixPath) -> None:
    runs = []
    for condition in ("off_a", "off_b", "on_a", "on_b", "noop_a", "noop_b"):
        for dataset in ("ai2-arc", "openbookqa"):
            runs.append(
                {
                    "condition": condition,
                    "dataset": dataset,
                    "training_forbidden": True,
                }
            )
    path.write_text(
        json.dumps(
            {
                "phase": "Phase 2A-2a equivalence debug",
                "code_commit": COMMIT,
                "workspace_root": str(workspace),
                "constraints": {
                    "evaluation_only": True,
                    "training_forbidden": True,
                    "selector_forbidden": True,
                    "geometry_predictability_forbidden": True,
                    "mmlu_forbidden": True,
                    "sealed_test_forbidden": True,
                    "one_visible_physical_gpu": True,
                    "serial_execution": True,
                },
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )


def test_job_is_one_gpu_one_node_and_serial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment_root = PurePosixPath(str(tmp_path))
    workspace = experiment_root / "workspace/Cache"
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, workspace)
    monkeypatch.setattr(jobs, "EXPERIMENT_ROOT", experiment_root)
    monkeypatch.setattr(jobs, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(jobs, "RESULTS_ROOT", experiment_root / "results/diagnostic")

    job, summary = jobs.build_job(
        git_commit=COMMIT,
        manifest_path=manifest,
        uid=1000,
        gid=1000,
    )

    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert summary["run_count"] == 12
    assert pod["nodeSelector"] == {"kubernetes.io/hostname": "4090-24gx4"}
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    rendered = json.dumps(job)
    assert "phase2a_2a_equivalence_debug.py" in rendered
    assert "SFT_train.py" not in rendered
    assert "mmlu-redux" not in rendered


def test_manifest_rejects_mmlu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment_root = PurePosixPath(str(tmp_path))
    workspace = experiment_root / "workspace/Cache"
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, workspace)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["runs"][0]["dataset"] = "mmlu-redux"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(jobs.EquivalenceJobError, match="run matrix"):
        jobs.validate_manifest(manifest, COMMIT)
