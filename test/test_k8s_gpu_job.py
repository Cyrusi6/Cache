from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from script.k8s import container_entrypoint, gpu_job


def _manifest(gpus: int = 1, **overrides: object) -> dict:
    values = {
        "job_name": "route-1-20260716-120000",
        "base_name": "Route 1",
        "namespace": "c2c-research",
        "node": "4090-24gx4",
        "image": "example.invalid/c2c:cu124",
        "gpus": gpus,
        "command": ["python", "train.py", "--message", "value with spaces"],
        "uid": 20007,
        "gid": 20007,
    }
    values.update(overrides)
    return gpu_job.build_job_manifest(**values)


def _pod_spec(manifest: dict) -> dict:
    return manifest["spec"]["template"]["spec"]


def _container(manifest: dict) -> dict:
    return _pod_spec(manifest)["containers"][0]


def test_sanitize_name_and_build_name_are_dns_compatible() -> None:
    assert gpu_job.sanitize_name(" Route_1 / V2.3 !!! ") == "route-1-v2-3"
    assert gpu_job.sanitize_name("A---B") == "a-b"
    assert gpu_job.sanitize_name("x" * 80) == "x" * 40
    assert (
        gpu_job.build_job_name("Route 1", datetime(2026, 7, 16, 12, 34, 56, 123456))
        == "route-1-20260716-123456-123456"
    )

    with pytest.raises(ValueError, match="任务名"):
        gpu_job.sanitize_name("___")


@pytest.mark.parametrize(
    ("gpus", "cpu", "memory", "shm"),
    [(1, "8", "32Gi", "16Gi"), (4, "32", "128Gi", "32Gi")],
)
def test_manifest_maps_one_and_four_gpu_resources(
    gpus: int, cpu: str, memory: str, shm: str
) -> None:
    manifest = _manifest(gpus)
    container = _container(manifest)

    assert container["resources"] == {
        "requests": {
            "cpu": cpu,
            "memory": memory,
            "nvidia.com/gpu": str(gpus),
        },
        "limits": {"nvidia.com/gpu": str(gpus)},
    }
    assert _pod_spec(manifest)["volumes"][-1] == {
        "name": "shm",
        "emptyDir": {"medium": "Memory", "sizeLimit": shm},
    }


def test_manifest_preserves_command_tokens_without_shell_joining() -> None:
    command = [
        "python",
        "script/train/SFT_train.py",
        "--config",
        "recipe/a config.yaml",
        "--literal",
        "$(do-not-execute)",
    ]
    manifest = _manifest(command=["--", *command])
    container = _container(manifest)

    assert container["command"] == [
        "python",
        "/workspace/Cache/script/k8s/container_entrypoint.py",
    ]
    assert container["args"][-(len(command) + 1) :] == ["--", *command]


def test_manifest_contains_expected_mounts_uid_and_management_labels() -> None:
    manifest = _manifest()
    pod_spec = _pod_spec(manifest)
    container = _container(manifest)

    labels = manifest["metadata"]["labels"]
    assert labels[gpu_job.MANAGED_BY_LABEL] == gpu_job.MANAGED_BY_VALUE
    assert labels[gpu_job.APP_LABEL] == gpu_job.APP_VALUE
    assert labels["c2c.research/experiment"] == "route-1"
    assert manifest["metadata"]["annotations"] == {
        gpu_job.MANAGED_BY_ANNOTATION: gpu_job.MANAGED_BY_VALUE
    }
    assert manifest["spec"]["template"]["metadata"]["labels"] == labels
    assert pod_spec["securityContext"] == {
        "runAsUser": 20007,
        "runAsGroup": 20007,
        "fsGroup": 20007,
    }
    assert pod_spec["nodeSelector"] == {"kubernetes.io/hostname": "4090-24gx4"}
    assert {item["name"] for item in container["volumeMounts"]} == {
        "workspace",
        "runtime",
        "huggingface-cache",
        "huggingface-cache-host-view",
        "c2c-datasets",
        "pip-cache",
        "shm",
    }
    assert {item["name"] for item in pod_spec["volumes"]} == {
        "workspace",
        "runtime",
        "huggingface-cache",
        "huggingface-cache-host-view",
        "c2c-datasets",
        "pip-cache",
        "shm",
    }
    mounts = {item["name"]: item for item in container["volumeMounts"]}
    assert mounts["huggingface-cache"] == {
        "name": "huggingface-cache",
        "mountPath": "/cache/huggingface",
    }
    assert mounts["huggingface-cache-host-view"] == {
        "name": "huggingface-cache-host-view",
        "mountPath": str(gpu_job.HF_CACHE),
        "readOnly": True,
    }
    assert mounts["c2c-datasets"] == {
        "name": "c2c-datasets",
        "mountPath": "/datasets",
        "readOnly": True,
    }
    volumes = {item["name"]: item for item in pod_spec["volumes"]}
    assert volumes["c2c-datasets"]["hostPath"] == {
        "path": str(gpu_job.DATASETS_ROOT),
        "type": "Directory",
    }
    env = {item["name"]: item["value"] for item in container["env"]}
    assert env["HOME"] == "/runtime/home"
    assert env["XDG_CACHE_HOME"] == "/runtime/home/.cache"
    assert env["TORCH_HOME"] == "/runtime/home/.cache/torch"
    assert env["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert env["HF_HUB_DOWNLOAD_TIMEOUT"] == "600"
    assert env["C2C_DATA_ROOT"] == "/datasets/c2c"


def test_init_dataset_links_are_complete_idempotent_and_portable(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets_root = tmp_path / "datasets"
    c2c_root = datasets_root / "c2c"
    hf_cache = tmp_path / "huggingface"
    longbench = datasets_root / "kvcomm_selective" / "LongBench"
    longbench.mkdir(parents=True)

    repo_name = "datasets--cais--mmlu"
    revision = "abc123"
    snapshot = hf_cache / "hub" / repo_name / "snapshots" / revision
    snapshot.mkdir(parents=True)
    ref = hf_cache / "hub" / repo_name / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text(revision, encoding="utf-8")

    monkeypatch.setattr(gpu_job, "DATASETS_ROOT", datasets_root)
    monkeypatch.setattr(gpu_job, "C2C_DATASETS_ROOT", c2c_root)
    monkeypatch.setattr(gpu_job, "HF_CACHE", hf_cache)
    monkeypatch.setattr(
        gpu_job,
        "HF_DATASET_LINKS",
        {"mmlu": repo_name, "ceval-exam": "datasets--ceval--ceval-exam"},
    )

    first = gpu_job.ensure_c2c_dataset_links()
    second = gpu_job.ensure_c2c_dataset_links()

    assert first == second
    assert first["mmlu"] == snapshot
    assert first["ceval-exam"] is None
    assert (c2c_root / "mmlu").resolve() == snapshot
    assert (c2c_root / "LongBench").readlink() == Path("../kvcomm_selective/LongBench")
    assert (c2c_root / "LongBench").resolve() == longbench


def test_submit_requires_initialized_dataset_root(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gpu_job, "C2C_DATASETS_ROOT", tmp_path / "missing")

    with pytest.raises(gpu_job.GpuJobError, match="统一数据目录不存在"):
        gpu_job.require_dataset_root()


def test_no_bootstrap_executes_image_command_directly() -> None:
    container = _container(_manifest(bootstrap=False))

    assert container["command"] == ["python"]
    assert container["args"] == [
        "train.py",
        "--message",
        "value with spaces",
    ]


@pytest.mark.parametrize("gpus", [-1, 0, 5, 99])
def test_illegal_gpu_count_is_rejected(gpus: int) -> None:
    with pytest.raises(ValueError, match="--gpus"):
        gpu_job.resource_defaults(gpus)


def test_delete_command_refuses_unmanaged_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        gpu_job,
        "kubectl_json",
        lambda *_args, **_kwargs: {"metadata": {"labels": {}}},
    )
    monkeypatch.setattr(
        gpu_job,
        "run_process",
        lambda command, **_kwargs: calls.append(list(command)),
    )
    args = gpu_job.build_parser().parse_args(["delete", "foreign-job"])

    with pytest.raises(gpu_job.GpuJobError, match="拒绝删除"):
        gpu_job.delete_command(args)
    assert calls == []


def test_non_local_node_is_rejected() -> None:
    with pytest.raises(gpu_job.GpuJobError, match="hostPath"):
        gpu_job.require_local_node("4090-24gx8")


@pytest.mark.parametrize(
    ("value", "seconds"), [("30s", 30), ("20m", 1200), ("72h", 259200)]
)
def test_parse_timeout(value: str, seconds: int) -> None:
    assert gpu_job.parse_timeout(value) == seconds

    with pytest.raises(ValueError):
        gpu_job.parse_timeout("2d")


def test_wait_for_job_pod_returns_terminal_pod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_job,
        "kubectl_json",
        lambda *_args, **_kwargs: {
            "items": [
                {
                    "metadata": {"name": "job-pod"},
                    "status": {"phase": "Succeeded"},
                }
            ]
        },
    )

    assert gpu_job.wait_for_job_pod("default", "ns", "job", 1) == "job-pod"


def test_wait_for_job_pod_fails_when_job_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            {"items": []},
            {"status": {"failed": 1}},
        ]
    )
    monkeypatch.setattr(
        gpu_job, "kubectl_json", lambda *_args, **_kwargs: next(responses)
    )

    with pytest.raises(gpu_job.GpuJobError, match="产生可读日志前失败"):
        gpu_job.wait_for_job_pod("default", "ns", "job", 1)


@pytest.mark.parametrize(
    "normalizer",
    [gpu_job.normalize_command, container_entrypoint.normalize_command],
)
def test_command_normalization_removes_only_leading_delimiter(normalizer) -> None:
    assert normalizer(["--", "python", "x.py", "--", "literal"]) == [
        "python",
        "x.py",
        "--",
        "literal",
    ]
    with pytest.raises(ValueError):
        normalizer(["--"])


def test_container_runtime_fingerprint_tracks_project_and_image(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[project]\nname='c2c'\n", encoding="utf-8")
    monkeypatch.setenv("C2C_RUNTIME_IMAGE", "c2c:image-a")

    first = container_entrypoint.runtime_identity(tmp_path)
    assert first["image"] == "c2c:image-a"
    assert first["extras"] == "dev,training,evaluation"
    assert len(first["fingerprint"]) == 64
    assert container_entrypoint.runtime_identity(tmp_path) == first

    pyproject.write_text("[project]\nname='c2c-v2'\n", encoding="utf-8")
    assert (
        container_entrypoint.runtime_identity(tmp_path)["fingerprint"]
        != first["fingerprint"]
    )
    monkeypatch.setenv("C2C_RUNTIME_IMAGE", "c2c:image-b")
    assert (
        container_entrypoint.runtime_identity(tmp_path)["fingerprint"]
        != first["fingerprint"]
    )


def test_bootstrap_uses_immutable_fingerprint_directories(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='c2c'\n", encoding="utf-8")
    runtime = tmp_path / "runtime"

    def fake_create(_self, path) -> None:
        bin_dir = path / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "python").write_text("", encoding="utf-8")

    monkeypatch.setattr(container_entrypoint.venv.EnvBuilder, "create", fake_create)
    monkeypatch.setattr(
        container_entrypoint.subprocess,
        "run",
        lambda *_args, **_kwargs: None,
    )

    monkeypatch.setenv("C2C_RUNTIME_IMAGE", "image:a")
    first_venv = container_entrypoint.bootstrap_runtime(runtime, project)
    sentinel = first_venv / "sentinel"
    sentinel.write_text("in use", encoding="utf-8")

    monkeypatch.setenv("C2C_RUNTIME_IMAGE", "image:b")
    second_venv = container_entrypoint.bootstrap_runtime(runtime, project)

    assert first_venv != second_venv
    assert sentinel.read_text(encoding="utf-8") == "in use"
    assert first_venv.is_dir()
    assert second_venv.is_dir()


def test_cli_parser_preserves_submit_options_and_remainder() -> None:
    args = gpu_job.build_parser().parse_args(
        [
            "submit",
            "--name",
            "route-1",
            "--gpus",
            "4",
            "--cpu",
            "24",
            "--memory",
            "96Gi",
            "--no-bootstrap",
            "--follow",
            "--",
            "torchrun",
            "--nproc_per_node=4",
            "train.py",
        ]
    )

    assert args.handler is gpu_job.submit_command
    assert args.namespace == "c2c-research"
    assert args.node == "4090-24gx4"
    assert (args.name, args.gpus, args.cpu, args.memory) == (
        "route-1",
        4,
        "24",
        "96Gi",
    )
    assert args.no_bootstrap is True
    assert args.follow is True
    assert args.command == [
        "--",
        "torchrun",
        "--nproc_per_node=4",
        "train.py",
    ]


def test_entrypoint_cli_parser_accepts_runtime_and_no_bootstrap(tmp_path) -> None:
    args = container_entrypoint.build_parser().parse_args(
        [
            "--runtime-dir",
            str(tmp_path / "runtime"),
            "--project-root",
            str(tmp_path),
            "--no-bootstrap",
            "--",
            "python",
            "-V",
        ]
    )

    assert args.runtime_dir == tmp_path / "runtime"
    assert args.project_root == tmp_path
    assert args.no_bootstrap is True
    assert args.command == ["--", "python", "-V"]
