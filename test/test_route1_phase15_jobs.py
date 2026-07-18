from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "script/k8s/route1_phase15_jobs.py"
SPEC = importlib.util.spec_from_file_location("route1_phase15_jobs", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
jobs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = jobs
SPEC.loader.exec_module(jobs)


def _manifest(path: Path) -> Path:
    runs = []
    index = 0
    for pair in sorted(jobs.EXPECTED_PAIRS):
        for seed in sorted(jobs.EXPECTED_SEEDS):
            for intervention in sorted(jobs.EXPECTED_INTERVENTIONS):
                run_id = f"{pair}__{intervention}__seed_{seed}"
                runs.append(
                    {
                        "id": run_id,
                        "pair": pair,
                        "seed": seed,
                        "intervention": {"id": intervention},
                        "checkpoint": {"same_checkpoint_no_training": True},
                        "eval_configs": {
                            dataset: f"local/tmp/eval/{run_id}/{dataset}.yaml"
                            for dataset in jobs.DATASETS
                        },
                        "output_dirs": {
                            dataset: f"/netdisk/results/{index}/{dataset}"
                            for dataset in jobs.DATASETS
                        },
                    }
                )
                index += 1
    path.write_text(
        json.dumps({"schema_version": 1, "runs": runs}), encoding="utf-8"
    )
    return path


def test_render_max7_uses_fixed_cross_node_two_gpu_layout(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    options = jobs.RenderOptions(
        git_commit="0" * 40,
        execution_manifest=manifest,
    )
    rendered, summary = jobs.build_jobs(options)

    assert len(rendered) == 7
    assert summary["shard_run_counts"] == [11, 11, 10, 10, 10, 10, 10]
    assert [
        item["spec"]["template"]["spec"]["nodeSelector"][
            "kubernetes.io/hostname"
        ]
        for item in rendered
    ] == [
        "4090-24gx4",
        "4090-24gx4",
        "4090-24gx8",
        "4090-24gx8",
        "4090-24gx8",
        "4090-24gx8",
        "4090-48gx2",
    ]
    for shard_index, item in enumerate(rendered):
        pod = item["spec"]["template"]["spec"]
        container = pod["containers"][0]
        assert container["resources"]["requests"]["nvidia.com/gpu"] == "2"
        assert container["resources"]["limits"]["nvidia.com/gpu"] == "2"
        assert "--shard-index" in container["args"]
        assert str(shard_index) in container["args"]
        assert container["args"].count("--num-shards") == 1
        assert "7" in container["args"]
        assert pod["volumes"][0]["hostPath"]["path"] == "/netdisk"
        assert f"shard_{shard_index:02d}" in " ".join(container["args"])
        init_script = pod["initContainers"][0]["command"][-1]
        assert ".git/HEAD" in init_script
        assert "FileNotFoundError" in init_script


def test_manifest_validation_rejects_duplicate_outputs(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["runs"][1]["output_dirs"]["ai2-arc"] = value["runs"][0][
        "output_dirs"
    ]["ai2-arc"]
    manifest.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(jobs.Phase15JobError, match="duplicate output dir"):
        jobs.validate_execution_manifest(manifest)


def test_server_dry_run_never_uses_persistent_apply(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    options = jobs.RenderOptions(git_commit="1" * 40, execution_manifest=manifest)
    rendered, _summary = jobs.build_jobs(options)
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    def preflight(_options, **_kwargs):
        return None

    jobs.server_dry_run(rendered, options, runner=runner, preflight=preflight)
    command = calls[-1][0]
    assert command[-4:] == ["apply", "--dry-run=server", "-f", "-"]
    assert calls[-1][1]["input"]


def test_run_shard_records_gpu_allocation_and_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    digest = jobs._sha256(manifest)
    state_dir = tmp_path / "state"
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(
                command, 0, stdout="GPU-a, 12\nGPU-b, 18\n", stderr=""
            )
        return subprocess.CompletedProcess(command, 0)

    return_code = jobs.run_shard(
        execution_manifest=manifest,
        expected_manifest_sha256=digest,
        shard_index=3,
        num_shards=7,
        state_dir=state_dir,
        max_startup_used_mib=4096,
        runner=runner,
    )

    assert return_code == 0
    completed = json.loads((state_dir / "completed.json").read_text())
    assert completed["selected_gpu_uuids"] == ["GPU-a", "GPU-b"]
    assert len(completed["run_ids"]) == 10
    child_env = calls[-1][1]["env"]
    assert child_env["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b"
