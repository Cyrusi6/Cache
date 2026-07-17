from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from script.k8s import route1_identifiability_lanes as lanes


COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _options(**overrides: object) -> lanes.RenderOptions:
    values: dict[str, object] = {
        "git_commit": COMMIT,
        "image": "example.invalid/c2c:cu124",
        "reproduction_gate": "pass",
        "uid": 20007,
        "gid": 20007,
    }
    values.update(overrides)
    return lanes.RenderOptions(**values)


def _by_kind(resources: list[dict], kind: str) -> list[dict]:
    return [resource for resource in resources if resource["kind"] == kind]


def _jobs_by_component(resources: list[dict]) -> dict[str, dict]:
    return {
        job["metadata"]["labels"]["app.kubernetes.io/component"]: job
        for job in _by_kind(resources, "Job")
    }


def _pod_spec(job: dict) -> dict:
    return job["spec"]["template"]["spec"]


def _runner_result(
    command: list[str], *, stdout: str = "", stderr: str = "", code: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, code, stdout=stdout, stderr=stderr)


def test_render_builds_shared_nfs_stager_gate_and_three_four_gpu_lanes() -> None:
    resources = lanes.build_resources(_options())

    assert len(resources) == 5
    assert not _by_kind(resources, "PersistentVolumeClaim")

    configmap = _by_kind(resources, "ConfigMap")[0]
    assert json.loads(configmap["data"]["gates.json"]) == {
        "reproduction": "pass",
        "conditional": "pending",
    }

    jobs = _jobs_by_component(resources)
    assert set(jobs) == {"workspace-stager", "lane-a", "lane-b", "lane-c"}
    assert _pod_spec(jobs["lane-a"])["nodeSelector"] == {
        "kubernetes.io/hostname": "4090-24gx4"
    }
    for component in ("workspace-stager", "lane-b", "lane-c"):
        assert _pod_spec(jobs[component])["nodeSelector"] == {
            "kubernetes.io/hostname": "4090-24gx8"
        }

    for component in jobs:
        volumes = {
            volume["name"]: volume for volume in _pod_spec(jobs[component])["volumes"]
        }
        assert volumes["shared-workspace"]["hostPath"] == {
            "path": "/netdisk",
            "type": "Directory",
        }
        mounts = {
            mount["name"]: mount
            for container in (
                _pod_spec(jobs[component]).get("initContainers", [])
                + _pod_spec(jobs[component]).get("containers", [])
            )
            for mount in container.get("volumeMounts", [])
        }
        assert mounts["shared-workspace"]["mountPath"] == "/netdisk"

    lane_a_volumes = {
        volume["name"]: volume
        for volume in _pod_spec(jobs["lane-a"])["volumes"]
    }
    assert "lane-a-models" not in lane_a_volumes
    assert "lane-a-datasets" not in lane_a_volumes


def test_lane_jobs_have_exact_resources_mounts_and_independent_plans() -> None:
    options = _options(
        lane_a_plan="local/tmp/custom/a.json",
        lane_b_plan="local/tmp/custom/b.json",
        lane_c_plan="local/tmp/custom/c.json",
        state_dir="local/tmp/custom/shared-state",
        hf_cache_host_path="/srv/hf-cache",
    )
    jobs = _jobs_by_component(lanes.build_resources(options))

    for component, suffix in (
        ("lane-a", "a.json"),
        ("lane-b", "b.json"),
        ("lane-c", "c.json"),
    ):
        pod = _pod_spec(jobs[component])
        container = pod["containers"][0]
        assert container["resources"] == {
            "requests": {
                "cpu": "24",
                "memory": "96Gi",
                "nvidia.com/gpu": "4",
            },
            "limits": {
                "cpu": "32",
                "memory": "112Gi",
                "nvidia.com/gpu": "4",
            },
        }
        args = container["args"]
        assert "run-lane" in args
        plan_index = args.index("--plan")
        assert args[plan_index + 1].endswith(suffix)
        assert args[args.index("--gate-file") + 1] == "/etc/route1-gates/gates.json"
        assert args[args.index("--state-dir") + 1].endswith("custom/shared-state")
        assert "--reuse-complete" in args
        assert args[args.index("--dependency-timeout-seconds") + 1] == "259200"
        env = {item["name"]: item["value"] for item in container["env"]}
        assert env["HF_HUB_OFFLINE"] == "1"
        assert env["HF_DATASETS_OFFLINE"] == "1"
        assert env["DATASETS_OFFLINE"] == "1"
        assert env["TRANSFORMERS_OFFLINE"] == "1"
        assert env["C2C_MODEL_ROOT"] == (
            "/netdisk/lijunsi/c2c-route1-identifiability/models"
        )
        assert env["C2C_DATA_ROOT"] == (
            "/netdisk/lijunsi/c2c-route1-identifiability/data/c2c"
        )

        mounts = {mount["name"]: mount for mount in container["volumeMounts"]}
        assert mounts["stage-gates"] == {
            "name": "stage-gates",
            "mountPath": "/etc/route1-gates",
            "readOnly": True,
        }
        assert mounts["shm"]["mountPath"] == "/dev/shm"
        volumes = {volume["name"]: volume for volume in pod["volumes"]}
        assert volumes["shm"] == {
            "name": "shm",
            "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"},
        }
        assert volumes["huggingface-cache"]["hostPath"] == {
            "path": "/srv/hf-cache",
            "type": "DirectoryOrCreate",
        }
        assert [item["name"] for item in pod["initContainers"]] == [
            "wait-for-staging"
        ]


def test_stager_clones_public_main_at_explicit_commit_and_runs_suite_generate() -> None:
    jobs = _jobs_by_component(
        lanes.build_resources(
            _options(reuse_override=str(lanes.DEFAULT_REUSE_OVERRIDE))
        )
    )
    pod = _pod_spec(jobs["workspace-stager"])
    checkout = pod["initContainers"][0]["args"][0]

    assert "git clone --branch main --single-branch" in checkout
    assert lanes.DEFAULT_REPOSITORY in checkout
    assert f"merge-base --is-ancestor {COMMIT} origin/main" in checkout
    assert f"checkout --detach {COMMIT}" in checkout

    args = pod["containers"][0]["args"]
    assert "stage-plans" in args
    assert args[args.index("--template") + 1] == str(
        lanes.WORKSPACE_ROOT / lanes.DEFAULT_TEMPLATE
    )
    assert args[args.index("--reuse-override") + 1] == str(
        lanes.WORKSPACE_ROOT / lanes.DEFAULT_REUSE_OVERRIDE
    )
    assert args[args.index("--plan-a") + 1].endswith("lane_a.phase1.json")
    assert args[args.index("--plan-b") + 1].endswith("lane_b.phase1.json")
    assert args[args.index("--plan-c") + 1].endswith("lane_c.phase1.json")
    init_names = [container["name"] for container in pod["initContainers"]]
    assert init_names == ["checkout", "audit-hf-cache-stager"]
    audit = pod["initContainers"][1]
    assert "audit-hf-cache" in audit["args"]
    assert audit["args"][audit["args"].index("--requirements") + 1].endswith(
        "recipe/train_recipe/identifiability/hf_cache_requirements.json"
    )
    assert all(item["name"] != "HF_HUB_OFFLINE" for item in audit["env"])
    assert audit["args"][audit["args"].index("--shared-root") + 1] == str(
        lanes.SHARED_ROOT
    )


def test_failed_b6_reproduction_is_not_reused_by_default() -> None:
    jobs = _jobs_by_component(lanes.build_resources(_options()))
    args = _pod_spec(jobs["workspace-stager"])["containers"][0]["args"]

    assert "--reuse-override" not in args


def test_optional_prefetch_job_warms_host_hf_cache_before_lanes() -> None:
    options = _options(
        prefetch_models=("Qwen/Qwen3-0.6B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    )
    resources = lanes.build_resources(options)
    jobs = _jobs_by_component(resources)

    assert "model-prefetch" in jobs
    prefetch_args = _pod_spec(jobs["model-prefetch"])["containers"][0]["args"]
    assert "prefetch-models" in prefetch_args
    assert prefetch_args.count("--model") == 2
    stager_init = _pod_spec(jobs["workspace-stager"])["initContainers"]
    assert [container["name"] for container in stager_init] == [
        "checkout",
        "wait-for-model-prefetch",
        "audit-hf-cache-stager",
    ]
    prefetch_wait = _pod_spec(jobs["model-prefetch"])["initContainers"][0]
    assert prefetch_wait["name"] == "wait-for-checkout"
    assert prefetch_wait["args"][-1].startswith(
        "/netdisk/lijunsi/c2c-route1-identifiability/status/checkout-"
    )
    for component in ("lane-a", "lane-b", "lane-c"):
        init = _pod_spec(jobs[component])["initContainers"]
        assert len(init[0]["args"]) == 2
        assert init[0]["args"][-1].startswith(
            "/netdisk/lijunsi/c2c-route1-identifiability/status/workspace-"
        )
        assert len(init) == 1


def test_new_phase_reuses_nfs_root_but_gets_new_immutable_job_names() -> None:
    phase1 = lanes.build_resources(_options())
    conditional = lanes.build_resources(
        _options(
            phase="conditional",
            conditional_gate="pass",
            lane_a_plan="local/tmp/route1_identifiability_suite/lanes/"
            "lane_a.conditional.json",
            lane_b_plan="local/tmp/route1_identifiability_suite/lanes/"
            "lane_b.conditional.json",
            lane_c_plan="local/tmp/route1_identifiability_suite/lanes/"
            "lane_c.conditional.json",
        )
    )

    assert not _by_kind(phase1, "PersistentVolumeClaim")
    assert not _by_kind(conditional, "PersistentVolumeClaim")
    for resources in (phase1, conditional):
        for job in _by_kind(resources, "Job"):
            volumes = {item["name"]: item for item in _pod_spec(job)["volumes"]}
            assert volumes["shared-workspace"]["hostPath"]["path"] == "/netdisk"
    phase1_jobs = {job["metadata"]["name"] for job in _by_kind(phase1, "Job")}
    conditional_jobs = {job["metadata"]["name"] for job in _by_kind(conditional, "Job")}
    assert phase1_jobs.isdisjoint(conditional_jobs)


@pytest.mark.parametrize(
    "overrides",
    [
        {"git_commit": "main"},
        {"repository": "git@github.com:Cyrusi6/Cache.git"},
        {"lane_b_plan": "same.json", "lane_c_plan": "same.json"},
        {"lane_b_plan": "../../escape.json"},
        {"shared_host_path": "/netdisk/lijunsi/c2c-route1-identifiability"},
    ],
)
def test_unsafe_or_ambiguous_render_inputs_are_rejected(overrides: dict) -> None:
    with pytest.raises(lanes.LaneInfrastructureError):
        lanes.build_resources(_options(**overrides))


def _write_plan(path: Path, lane: str, runs: list[dict]) -> None:
    normalized_runs = [dict(run, gate_key="reproduction") for run in runs]
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite": "route1_v22_identifiability",
                "lane": lane,
                "phase": "phase1",
                "hardware": {
                    "node_profile": "24gx4" if lane == "lane_a" else "24gx8",
                    "requested_gpus": 4,
                },
                "state_dir": "local/tmp/state",
                "runs": normalized_runs,
            }
        ),
        encoding="utf-8",
    )


def _run(
    run_id: str,
    checkpoint: str,
    output_dir: str,
    depends_on_runs: list[str] | None = None,
) -> dict:
    return {
        "run_id": run_id,
        "depends_on_runs": depends_on_runs or [],
        "training": {"required": True, "selected_checkpoint": checkpoint},
        "evaluation": {"output_dirs": {"mmlu-redux": output_dir}},
    }


def test_validate_plans_accepts_disjoint_outputs(tmp_path: Path) -> None:
    plan_a = tmp_path / "a.json"
    plan_b = tmp_path / "b.json"
    plan_c = tmp_path / "c.json"
    _write_plan(
        plan_a,
        "lane_a",
        [_run("tiny__b1__42", "local/checkpoints/b1/final", "local/results/b1")],
    )
    _write_plan(
        plan_b,
        "lane_b",
        [_run("tiny__b2__42", "local/checkpoints/b2/final", "local/results/b2")],
    )
    _write_plan(
        plan_c,
        "lane_c",
        [
            _run(
                "tiny__b3__42",
                "local/checkpoints/b3/final",
                "local/results/b3",
                depends_on_runs=["tiny__b2__42"],
            )
        ],
    )

    summary = lanes.validate_plan_files(plan_a, plan_b, plan_c)

    assert summary["lane_a_runs"] == 1
    assert summary["lane_b_runs"] == 1
    assert summary["lane_c_runs"] == 1
    assert summary["validated_outputs"] == 6


@pytest.mark.parametrize("duplicate_run", [False, True])
def test_validate_plans_rejects_output_or_run_collision(
    tmp_path: Path, duplicate_run: bool
) -> None:
    plan_a = tmp_path / "a.json"
    plan_b = tmp_path / "b.json"
    plan_c = tmp_path / "c.json"
    _write_plan(
        plan_a,
        "lane_a",
        [_run("run-a", "local/checkpoints/a/final", "local/results/a")],
    )
    _write_plan(
        plan_b,
        "lane_b",
        [_run("run-b", "local/checkpoints/shared/final", "local/results/b")],
    )
    run_c = "run-b" if duplicate_run else "run-c"
    checkpoint_c = (
        "local/checkpoints/c/final"
        if duplicate_run
        else "local/checkpoints/shared/final/nested"
    )
    _write_plan(
        plan_c,
        "lane_c",
        [_run(run_c, checkpoint_c, "local/results/c")],
    )

    match = "duplicate run ids" if duplicate_run else "output directory conflicts"
    with pytest.raises(lanes.LaneInfrastructureError, match=match):
        lanes.validate_plan_files(plan_a, plan_b, plan_c)


def test_validate_plans_accepts_cross_node_dependency_via_shared_nfs(
    tmp_path: Path,
) -> None:
    plan_a = tmp_path / "a.json"
    plan_b = tmp_path / "b.json"
    plan_c = tmp_path / "c.json"
    _write_plan(
        plan_a,
        "lane_a",
        [
            _run(
                "reproduction-run-on-lane-a",
                "local/checkpoints/a/final",
                "local/results/a",
            )
        ],
    )
    _write_plan(
        plan_b,
        "lane_b",
        [
            _run(
                "run-b",
                "local/checkpoints/b/final",
                "local/results/b",
                depends_on_runs=["reproduction-run-on-lane-a"],
            )
        ],
    )
    _write_plan(
        plan_c,
        "lane_c",
        [_run("run-c", "local/checkpoints/c/final", "local/results/c")],
    )

    summary = lanes.validate_plan_files(plan_a, plan_b, plan_c)
    assert summary["lane_a_runs"] == 1


def test_validate_plans_rejects_swapped_lane_identity(tmp_path: Path) -> None:
    plan_a = tmp_path / "a.json"
    plan_b = tmp_path / "b.json"
    plan_c = tmp_path / "c.json"
    _write_plan(plan_a, "lane_a", [_run("a", "ckpt/a", "results/a")])
    _write_plan(plan_b, "lane_c", [_run("b", "ckpt/b", "results/b")])
    _write_plan(plan_c, "lane_b", [_run("c", "ckpt/c", "results/c")])

    with pytest.raises(lanes.LaneInfrastructureError, match="expected lane=lane_b"):
        lanes.validate_plan_files(plan_a, plan_b, plan_c)


def test_stage_plans_surfaces_missing_template_without_calling_suite(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def runner(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return _runner_result(command)

    with pytest.raises(lanes.LaneInfrastructureError, match="template is missing"):
        lanes.stage_plans(
            suite_path=tmp_path / "suite.py",
            template_path=tmp_path / "missing.json",
            output_root=tmp_path / "out",
            plan_a_path=tmp_path / "a.json",
            plan_b_path=tmp_path / "b.json",
            plan_c_path=tmp_path / "c.json",
            ready_marker=tmp_path / "ready.json",
            git_commit=COMMIT,
            runner=runner,
        )
    assert calls == []


def test_stage_plans_publishes_ready_only_after_generation_and_validation(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite.py"
    template = tmp_path / "template.json"
    output = tmp_path / "out"
    plan_a = output / "lanes/a.json"
    plan_b = output / "lanes/b.json"
    plan_c = output / "lanes/c.json"
    ready = tmp_path / "status/ready.json"
    suite.write_text("# suite\n", encoding="utf-8")
    template.write_text("{}\n", encoding="utf-8")

    def runner(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        plan_b.parent.mkdir(parents=True)
        _write_plan(plan_a, "lane_a", [_run("a", "ckpt/a", "results/a")])
        _write_plan(plan_b, "lane_b", [_run("b", "ckpt/b", "results/b")])
        _write_plan(plan_c, "lane_c", [_run("c", "ckpt/c", "results/c")])
        return _runner_result(command)

    summary = lanes.stage_plans(
        suite_path=suite,
        template_path=template,
        output_root=output,
        plan_a_path=plan_a,
        plan_b_path=plan_b,
        plan_c_path=plan_c,
        ready_marker=ready,
        git_commit=COMMIT,
        runner=runner,
    )

    assert summary["lane_a_runs"] == 1
    assert summary["lane_b_runs"] == 1
    assert json.loads(ready.read_text(encoding="utf-8"))["git_commit"] == COMMIT


def test_server_dry_run_performs_preflight_then_uses_api_server_validation() -> None:
    options = _options()
    resources = lanes.build_resources(options)
    calls: list[tuple[list[str], str | None]] = []

    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        input_text = kwargs.get("input")
        calls.append((command, input_text if isinstance(input_text, str) else None))
        if "node" in command and "-o" in command:
            return _runner_result(
                command,
                stdout=json.dumps(
                    {
                        "status": {
                            "conditions": [{"type": "Ready", "status": "True"}],
                            "allocatable": {"nvidia.com/gpu": "8"},
                        }
                    }
                ),
            )
        if "can-i" in command:
            return _runner_result(command, stdout="yes\n")
        return _runner_result(command, stdout="ok\n")

    lanes.kubectl_submit(
        resources,
        options,
        server_dry_run=True,
        which=lambda _name: "/usr/bin/kubectl",
        runner=runner,
    )

    apply_command, payload = calls[-1]
    assert apply_command[-4:] == ["apply", "--dry-run=server", "-f", "-"]
    assert "apply" in apply_command
    assert payload is not None
    assert json.loads(payload)["kind"] == "List"
    assert any("auth" in command and "jobs.batch" in command for command, _ in calls)
    assert not any("storageclass" in command for command, _ in calls)
    assert not any("persistentvolumeclaims" in command for command, _ in calls)
    node_queries = [
        command[command.index("node") + 1]
        for command, _ in calls
        if "node" in command and "-o" in command
    ]
    assert node_queries == ["4090-24gx4", "4090-24gx8"]


def test_default_cli_is_render_only_and_does_not_submit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unexpected_submit(*_args: object, **_kwargs: object) -> None:
        pytest.fail("render-only CLI must not contact Kubernetes")

    monkeypatch.setattr(lanes, "kubectl_submit", unexpected_submit)

    assert lanes.main(["render", "--git-commit", COMMIT, "--format", "json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["kind"] == "List"
    assert "kubectl_mode=render-only" in captured.err
    assert "components=infra,stager" in captured.err


def _write_hf_requirements(
    path: Path, *, revision: str = COMMIT, package_version: str = "5.0.0"
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "python_packages": {"datasets": package_version},
                "repositories": [
                    {
                        "key": "fixture_model",
                        "repo_type": "model",
                        "repo_id": "Acme/fixture-model",
                        "ref": "main",
                        "expected_revision": revision,
                        "required_files": ["config.json"],
                        "required_globs": [
                            {"pattern": "*.safetensors", "minimum_matches": 1}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_hf_fixture_cache(hf_home: Path, revision: str = COMMIT) -> Path:
    repo = hf_home / "hub/models--Acme--fixture-model"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs/main").write_text(revision + "\n", encoding="utf-8")
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"fixture")
    return snapshot


def test_tracked_hf_requirements_pin_all_stage1_refs_and_environment() -> None:
    path = Path(lanes.DEFAULT_HF_REQUIREMENTS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    revisions = {
        item["key"]: item["expected_revision"] for item in payload["repositories"]
    }
    assert revisions == {
        "mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
        "mmlu_redux": "372ea425445d51e1ba1188c56e5e893f8138621f",
        "openbookqa": "388097ea7776314e93a529163e0fea805b8a6454",
        "arc": "210d026faf9955653af8916fad021475a3f00453",
        "qwen3_0p6b": "c1899de289a04d12100db370d81485cdf75e47ca",
        "tinyllama_1p1b": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        "qwen25_0p5b": "7ae557604adf67be50417f59c2c2f167def9a775",
    }
    assert payload["python_packages"] == {
        "datasets": "5.0.0",
        "accelerate": "1.14.0",
        "wandb": "0.28.0",
        "peft": "0.19.1",
        "transformers": "4.52.4",
    }
    templates = {item["key"]: item for item in payload["repository_templates"]}
    qwen = templates["qwen3_1p7b"]
    assert qwen["file_sha256"]["config.json"] == (
        "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197"
    )
    assert qwen["file_sha256"]["model-00001-of-00002.safetensors"] == (
        "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5"
    )
    shared_qwen = payload["shared_models"][0]
    assert shared_qwen["directory"] == "models/Qwen3-1.7B"
    assert shared_qwen["file_sha256"] == qwen["file_sha256"]
    assert shared_qwen["directory_sha256"] == (
        "b1b46b15937e8eb22d2dc7f0c5a6de698b4873f106f4e7f8d080bc7192879a32"
    )
    assert {item["key"] for item in payload["shared_models"]} == {
        "qwen3_0p6b",
        "tinyllama_1p1b",
        "qwen25_0p5b",
        "qwen3_1p7b",
        "llama32_1b",
    }
    assert {item["key"] for item in payload["shared_datasets"]} == {
        "mmlu",
        "mmlu_redux",
        "openbookqa",
        "arc",
    }
    assert all(
        len(item["directory_sha256"]) == 64
        for item in payload["shared_datasets"]
    )

    inactive = lanes.load_hf_requirements(path)
    assert all(item["key"] != "qwen3_1p7b" for item in inactive)
    activated = lanes.load_hf_requirements(path, [f"qwen3_1p7b={COMMIT}"])
    activated_qwen = next(item for item in activated if item["key"] == "qwen3_1p7b")
    assert activated_qwen["expected_revision"] == COMMIT
    assert activated_qwen["file_sha256"] == qwen["file_sha256"]


def test_hf_cache_audit_writes_provenance_only_after_complete_validation(
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements.json"
    hf_home = tmp_path / "hf"
    ready = tmp_path / "status/provenance.json"
    _write_hf_requirements(requirements)
    _write_hf_fixture_cache(hf_home)

    payload = lanes.audit_hf_cache(
        hf_home=hf_home,
        requirements_path=requirements,
        ready_marker=ready,
        version_resolver=lambda package: {"datasets": "5.0.0"}[package],
    )

    persisted = json.loads(ready.read_text(encoding="utf-8"))
    assert persisted == payload
    assert payload["status"] == "ready"
    assert payload["python_packages"] == {"datasets": "5.0.0"}
    assert payload["repositories"][0]["resolved_revision"] == COMMIT
    assert payload["repositories"][0]["glob_counts"][0]["actual_matches"] == 1


@pytest.mark.parametrize("failure", ["revision", "file", "environment"])
def test_hf_cache_audit_blocks_mismatch_or_incomplete_cache(
    tmp_path: Path, failure: str
) -> None:
    requirements = tmp_path / "requirements.json"
    hf_home = tmp_path / "hf"
    ready = tmp_path / "status/provenance.json"
    _write_hf_requirements(requirements)
    snapshot = _write_hf_fixture_cache(hf_home)
    if failure == "revision":
        (hf_home / "hub/models--Acme--fixture-model/refs/main").write_text(
            "f" * 40 + "\n", encoding="utf-8"
        )
    elif failure == "file":
        (snapshot / "config.json").unlink()

    actual_version = "6.0.0" if failure == "environment" else "5.0.0"
    with pytest.raises(lanes.LaneInfrastructureError):
        lanes.audit_hf_cache(
            hf_home=hf_home,
            requirements_path=requirements,
            ready_marker=ready,
            version_resolver=lambda _package: actual_version,
        )
    assert not ready.exists()


def test_hf_cache_audit_enforces_tracked_file_sha256(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.json"
    hf_home = tmp_path / "hf"
    ready = tmp_path / "status/provenance.json"
    _write_hf_requirements(requirements)
    payload = json.loads(requirements.read_text(encoding="utf-8"))
    payload["repositories"][0]["file_sha256"] = {
        "config.json": hashlib.sha256(b"{}\n").hexdigest()
    }
    requirements.write_text(json.dumps(payload), encoding="utf-8")
    snapshot = _write_hf_fixture_cache(hf_home)

    audited = lanes.audit_hf_cache(
        hf_home=hf_home,
        requirements_path=requirements,
        ready_marker=ready,
        version_resolver=lambda _package: "5.0.0",
    )
    assert audited["repositories"][0]["critical_files"][0]["sha256"] == (
        hashlib.sha256(b"{}\n").hexdigest()
    )

    ready.unlink()
    (snapshot / "config.json").write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(lanes.LaneInfrastructureError, match="SHA256 mismatch"):
        lanes.audit_hf_cache(
            hf_home=hf_home,
            requirements_path=requirements,
            ready_marker=ready,
            version_resolver=lambda _package: "5.0.0",
        )
    assert not ready.exists()


def test_stager_audit_enforces_shared_model_hashes(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.json"
    hf_home = tmp_path / "hf"
    shared_root = tmp_path / "shared"
    ready = tmp_path / "status/provenance.json"
    _write_hf_requirements(requirements)
    payload = json.loads(requirements.read_text(encoding="utf-8"))
    model_bytes = b"shared model fixture"
    payload["shared_models"] = [
        {
            "key": "shared_fixture",
            "directory": "models/fixture",
            "required_files": ["model.safetensors"],
            "file_sha256": {
                "model.safetensors": hashlib.sha256(model_bytes).hexdigest()
            },
            "directory_sha256": "0" * 64,
        }
    ]
    requirements.write_text(json.dumps(payload), encoding="utf-8")
    _write_hf_fixture_cache(hf_home)
    model_path = shared_root / "models/fixture/model.safetensors"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(model_bytes)
    payload["shared_models"][0]["directory_sha256"] = lanes._directory_tree_sha256(
        model_path.parent
    )
    requirements.write_text(json.dumps(payload), encoding="utf-8")

    audited = lanes.audit_hf_cache(
        hf_home=hf_home,
        shared_root=shared_root,
        requirements_path=requirements,
        ready_marker=ready,
        version_resolver=lambda _package: "5.0.0",
    )
    assert audited["shared_models"][0]["key"] == "shared_fixture"
    assert audited["shared_models"][0]["critical_files"][0]["sha256"] == (
        hashlib.sha256(model_bytes).hexdigest()
    )
    assert audited["shared_models"][0]["directory_sha256"] == payload[
        "shared_models"
    ][0]["directory_sha256"]

    ready.unlink()
    model_path.write_bytes(b"corrupt")
    with pytest.raises(lanes.LaneInfrastructureError, match="SHA256 mismatch"):
        lanes.audit_hf_cache(
            hf_home=hf_home,
            shared_root=shared_root,
            requirements_path=requirements,
            ready_marker=ready,
            version_resolver=lambda _package: "5.0.0",
        )
    assert not ready.exists()


def test_stager_audit_enforces_shared_dataset_tree_hash(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.json"
    hf_home = tmp_path / "hf"
    shared_root = tmp_path / "shared"
    ready = tmp_path / "status/provenance.json"
    _write_hf_requirements(requirements)
    payload = json.loads(requirements.read_text(encoding="utf-8"))
    dataset_root = shared_root / "data/c2c/fixture"
    dataset_root.mkdir(parents=True)
    (dataset_root / "README.md").write_text("fixture\n", encoding="utf-8")
    (dataset_root / "data.bin").write_bytes(b"dataset bytes")
    payload["shared_datasets"] = [
        {
            "key": "shared_dataset_fixture",
            "directory": "data/c2c/fixture",
            "required_files": ["README.md", "data.bin"],
            "directory_sha256": lanes._directory_tree_sha256(dataset_root),
        }
    ]
    requirements.write_text(json.dumps(payload), encoding="utf-8")
    _write_hf_fixture_cache(hf_home)

    audited = lanes.audit_hf_cache(
        hf_home=hf_home,
        shared_root=shared_root,
        requirements_path=requirements,
        ready_marker=ready,
        version_resolver=lambda _package: "5.0.0",
    )
    assert audited["shared_datasets"][0]["key"] == "shared_dataset_fixture"
    assert audited["shared_datasets"][0]["directory_sha256"] == payload[
        "shared_datasets"
    ][0]["directory_sha256"]

    ready.unlink()
    (dataset_root / "data.bin").write_bytes(b"corrupt")
    with pytest.raises(lanes.LaneInfrastructureError, match="dataset SHA256 mismatch"):
        lanes.audit_hf_cache(
            hf_home=hf_home,
            shared_root=shared_root,
            requirements_path=requirements,
            ready_marker=ready,
            version_resolver=lambda _package: "5.0.0",
        )
    assert not ready.exists()


def test_prefetch_uses_fixed_ref_and_rejects_unexpected_revision(
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements.json"
    marker = tmp_path / "prefetch.json"
    _write_hf_requirements(requirements)
    calls: list[dict[str, str]] = []

    def downloader(**kwargs: str) -> str:
        calls.append(kwargs)
        return str(tmp_path / "snapshots" / COMMIT)

    lanes.prefetch_models(
        ["Acme/fixture-model"],
        marker,
        requirements_path=requirements,
        downloader=downloader,
    )
    assert calls == [
        {
            "repo_id": "Acme/fixture-model",
            "revision": "main",
            "repo_type": "model",
        }
    ]
    assert json.loads(marker.read_text(encoding="utf-8"))["models"] == [
        {"model": "Acme/fixture-model", "revision": COMMIT}
    ]

    marker.unlink()
    with pytest.raises(lanes.LaneInfrastructureError, match="revision mismatch"):
        lanes.prefetch_models(
            ["Acme/fixture-model"],
            marker,
            requirements_path=requirements,
            downloader=lambda **_kwargs: str(tmp_path / "snapshots" / ("f" * 40)),
        )
    assert not marker.exists()


def test_staged_components_and_gate_hard_blocks_prevent_early_lane_jobs() -> None:
    pending = _options(reproduction_gate="pending")
    stage_resources = lanes.build_resources(pending, ("stage",))
    assert [resource["kind"] for resource in stage_resources] == [
        "ConfigMap",
        "Job",
    ]
    assert (
        _by_kind(stage_resources, "Job")[0]["metadata"]["labels"][
            "app.kubernetes.io/component"
        ]
        == "workspace-stager"
    )

    with pytest.raises(lanes.LaneInfrastructureError, match="reproduction_gate"):
        lanes.build_resources(pending, ("lanes",))
    with pytest.raises(lanes.LaneInfrastructureError, match="conditional_gate"):
        lanes.build_resources(
            _options(
                phase="conditional",
                conditional_gate="pending",
                lane_a_plan="local/tmp/lanes/lane_a.conditional.json",
                lane_b_plan="local/tmp/lanes/lane_b.conditional.json",
                lane_c_plan="local/tmp/lanes/lane_c.conditional.json",
            ),
            ("lanes",),
        )
    lane_resources = lanes.build_resources(_options(), ("lanes",))
    assert set(_jobs_by_component(lane_resources)) == {
        "lane-a",
        "lane-b",
        "lane-c",
    }


def test_apply_rejects_combining_lane_jobs_with_earlier_stages(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unexpected_submit(*_args: object, **_kwargs: object) -> None:
        pytest.fail("unsafe combined apply must fail before contacting Kubernetes")

    monkeypatch.setattr(lanes, "kubectl_submit", unexpected_submit)
    result = lanes.main(
        [
            "render",
            "--git-commit",
            COMMIT,
            "--reproduction-gate",
            "pass",
            "--component",
            "stage",
            "--component",
            "lanes",
            "--apply",
        ]
    )
    assert result == 1
    assert "dedicated --component lanes" in capsys.readouterr().err
