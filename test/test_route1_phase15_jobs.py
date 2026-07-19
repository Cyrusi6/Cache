from __future__ import annotations

import copy
import importlib.util
import json
import os
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
                            dataset: str(path.parent / "results" / str(index) / dataset)
                            for dataset in jobs.DATASETS
                        },
                    }
                )
                index += 1
    path.write_text(
        json.dumps({"schema_version": 1, "runs": runs}), encoding="utf-8"
    )
    return path


def test_render_uses_three_node_level_gpu_pools(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    options = jobs.RenderOptions(
        git_commit="0" * 40,
        execution_manifest=manifest,
    )
    rendered, summary = jobs.build_jobs(options)

    assert len(rendered) == 3
    assert summary["shard_run_counts"] == [11, 11, 10, 10, 10, 10, 10]
    assert [
        item["spec"]["template"]["spec"]["nodeSelector"][
            "kubernetes.io/hostname"
        ]
        for item in rendered
    ] == [
        "4090-24gx4",
        "4090-24gx8",
        "4090-48gx2",
    ]
    assert [
        item["spec"]["template"]["spec"]["containers"][0]["resources"]
        ["requests"]["nvidia.com/gpu"]
        for item in rendered
    ] == ["4", "8", "2"]
    expected = [((0, 1), 1), ((2, 3, 4, 5), 3), ((6,), 1)]
    for item, (shard_indices, max_parallel) in zip(rendered, expected):
        pod = item["spec"]["template"]["spec"]
        container = pod["containers"][0]
        assert container["resources"]["requests"]["nvidia.com/gpu"] == container[
            "resources"
        ]["limits"]["nvidia.com/gpu"]
        assert "run-node" in container["args"]
        assert "--shard-indices" in container["args"]
        assert ",".join(str(index) for index in shard_indices) in container["args"]
        assert "--max-parallel-shards" in container["args"]
        assert str(max_parallel) in container["args"]
        assert container["args"].count("--num-shards") == 1
        assert "7" in container["args"]
        assert pod["volumes"][0]["hostPath"]["path"] == "/netdisk"
        env = {item["name"]: item["value"] for item in container["env"]}
        assert env["C2C_PRESERVE_CUDA_VISIBLE_DEVICES"] == "1"
        assert env["C2C_PHASE15_WORKSPACE_ROOT"] == str(jobs.WORKSPACE_ROOT)
        assert env["PIP_CONSTRAINT"].endswith(
            "recipe/train_recipe/identifiability/runtime_constraints.txt"
        )
        init_script = pod["initContainers"][0]["command"][-1]
        assert ".git/HEAD" in init_script
        assert "FileNotFoundError" in init_script


def test_workspace_root_can_be_overridden_for_resume_checkout(tmp_path: Path) -> None:
    workspace = tmp_path / "Cache_phase15_fast"
    script = (
        "import runpy; "
        f"value = runpy.run_path({str(MODULE_PATH)!r}); "
        "print(value['WORKSPACE_ROOT'])"
    )
    environment = dict(os.environ)
    environment["C2C_PHASE15_WORKSPACE_ROOT"] = str(workspace)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == str(workspace)


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
    completed = json.loads((state_dir / "shard_03/completed.json").read_text())
    assert completed["selected_gpu_uuids"] == ["GPU-a", "GPU-b"]
    assert len(completed["run_ids"]) == 10
    child_env = calls[-1][1]["env"]
    assert child_env["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b"
    assert child_env["C2C_PRESERVE_CUDA_VISIBLE_DEVICES"] == "1"


def test_run_shard_opportunistic_selects_idle_pair_and_forwards_tail_mode(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    digest = jobs._sha256(manifest)
    state_dir = tmp_path / "x4-tail"
    calls = []

    def runner(command, **kwargs):
        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "GPU-a, 12\nGPU-hidden, 19456\n"
                    "GPU-c, 18\nGPU-d, 24\n"
                ),
                stderr="",
            )
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    assert jobs.run_shard_opportunistic(
        execution_manifest=manifest,
        expected_manifest_sha256=digest,
        shard_index=5,
        num_shards=7,
        state_dir=state_dir,
        max_startup_used_mib=4096,
        idle_poll_seconds=2.5,
        runner=runner,
    ) == 0

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert "run-shard-opportunistic" in command
    assert command[command.index("--shard-index") + 1] == "5"
    assert command[command.index("--idle-poll-seconds") + 1] == "2.5"
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-c"
    assert kwargs["env"]["C2C_PRESERVE_CUDA_VISIBLE_DEVICES"] == "1"
    completed = json.loads((state_dir / "shard_05/completed.json").read_text())
    assert completed["runner_mode"] == "opportunistic"
    assert completed["idle_poll_seconds"] == 2.5
    assert all(
        (tmp_path / "results" / str(index)).is_dir()
        for index in range(72)
        if index % 7 == 5
    )


def test_run_shard_opportunistic_rejects_unbounded_poll_before_gpu_query(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")

    def unexpected_runner(*_args, **_kwargs):
        raise AssertionError("invalid polling must fail before querying GPUs")

    with pytest.raises(jobs.Phase15JobError, match="at most 60"):
        jobs.run_shard_opportunistic(
            execution_manifest=manifest,
            expected_manifest_sha256=jobs._sha256(manifest),
            shard_index=5,
            num_shards=7,
            state_dir=tmp_path / "tail",
            max_startup_used_mib=4096,
            idle_poll_seconds=60.01,
            runner=unexpected_runner,
        )


def test_main_routes_opportunistic_recovery_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def opportunistic(**kwargs):
        captured.update(kwargs)
        return 17

    monkeypatch.setattr(jobs, "run_shard_opportunistic", opportunistic)
    assert jobs.main(
        [
            "run-shard-opportunistic",
            "--execution-manifest",
            str(tmp_path / "manifest.json"),
            "--expected-manifest-sha256",
            "a" * 64,
            "--shard-index",
            "5",
            "--num-shards",
            "7",
            "--state-dir",
            str(tmp_path / "tail-state"),
            "--max-startup-used-mib",
            "3072",
            "--idle-poll-seconds",
            "3.5",
        ]
    ) == 17
    assert captured == {
        "execution_manifest": tmp_path / "manifest.json",
        "expected_manifest_sha256": "a" * 64,
        "shard_index": 5,
        "num_shards": 7,
        "state_dir": tmp_path / "tail-state",
        "max_startup_used_mib": 3072,
        "idle_poll_seconds": 3.5,
    }


def test_run_node_x4_filters_hidden_busy_gpu_and_serializes_two_shards(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    digest = jobs._sha256(manifest)
    calls = []

    def runner(command, **kwargs):
        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "GPU-a, 20\nGPU-hidden, 19456\n"
                    "GPU-c, 32\nGPU-d, 48\n"
                ),
                stderr="",
            )
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    state_dir = tmp_path / "x4"
    assert jobs.run_node(
        execution_manifest=manifest,
        expected_manifest_sha256=digest,
        shard_indices=(0, 1),
        num_shards=7,
        state_dir=state_dir,
        max_startup_used_mib=4096,
        max_parallel_shards=1,
        runner=runner,
    ) == 0

    assert len(calls) == 2
    assert [
        int(command[command.index("--shard-index") + 1])
        for command, _kwargs in calls
    ] == [0, 1]
    assert {
        kwargs["env"]["CUDA_VISIBLE_DEVICES"] for _command, kwargs in calls
    } == {"GPU-a,GPU-c"}
    started = json.loads((state_dir / "started.json").read_text())
    assert started["selected_gpu_groups"][0]["shard_indices"] == [0, 1]
    hidden = next(
        item for item in started["gpu_inventory"] if item["uuid"] == "GPU-hidden"
    )
    assert hidden["idle_eligible"] is False
    assert all(
        (tmp_path / "results" / str(index)).is_dir()
        for index in range(72)
        if index % 7 in {0, 1}
    )


def test_run_node_x8_uses_three_pairs_and_queues_fourth_shard(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    digest = jobs._sha256(manifest)
    calls = []

    def runner(command, **kwargs):
        if command[0] == "nvidia-smi":
            rows = [
                "GPU-a, 10",
                "GPU-hidden, 22528",
                "GPU-c, 20",
                "GPU-d, 30",
                "GPU-e, 40",
                "GPU-f, 50",
                "GPU-g, 60",
                "GPU-h, 70",
            ]
            return subprocess.CompletedProcess(
                command, 0, stdout="\n".join(rows) + "\n", stderr=""
            )
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    state_dir = tmp_path / "x8"
    assert jobs.run_node(
        execution_manifest=manifest,
        expected_manifest_sha256=digest,
        shard_indices=(2, 3, 4, 5),
        num_shards=7,
        state_dir=state_dir,
        max_startup_used_mib=4096,
        max_parallel_shards=3,
        runner=runner,
    ) == 0

    assert len(calls) == 4
    assert {
        int(command[command.index("--shard-index") + 1])
        for command, _kwargs in calls
    } == {2, 3, 4, 5}
    masks = [kwargs["env"]["CUDA_VISIBLE_DEVICES"] for _command, kwargs in calls]
    assert set(masks) == {"GPU-a,GPU-c", "GPU-d,GPU-e", "GPU-f,GPU-g"}
    assert sorted(masks.count(mask) for mask in set(masks)) == [1, 1, 2]
    started = json.loads((state_dir / "started.json").read_text())
    assert len(started["selected_gpu_groups"]) == 3
    assert sorted(
        len(group["shard_indices"])
        for group in started["selected_gpu_groups"]
    ) == [1, 1, 2]


def test_run_node_skips_matching_completed_shard_state(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    digest = jobs._sha256(manifest)
    state_dir = tmp_path / "x48"
    completed = state_dir / "shard_06/completed.json"
    completed.parent.mkdir(parents=True)
    completed.write_text(
        json.dumps({"manifest_sha256": digest, "shard_index": 6, "return_code": 0}),
        encoding="utf-8",
    )

    def unexpected_runner(*_args, **_kwargs):
        raise AssertionError("a completed shard must not query or launch GPUs")

    assert jobs.run_node(
        execution_manifest=manifest,
        expected_manifest_sha256=digest,
        shard_indices=(6,),
        num_shards=7,
        state_dir=state_dir,
        max_startup_used_mib=4096,
        max_parallel_shards=1,
        runner=unexpected_runner,
    ) == 0
