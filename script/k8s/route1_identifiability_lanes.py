#!/usr/bin/env python3
"""Render the cross-node Route-1 identifiability lane infrastructure.

The default command is intentionally render-only. Kubernetes is contacted only
when ``--server-dry-run`` or ``--apply`` is supplied explicitly. All three lanes
mount the same NFS root: lane A runs on the four-GPU node and lanes B/C share the
eight-GPU node.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

try:
    from script.k8s.gpu_job import DEFAULT_IMAGE
except ModuleNotFoundError:  # Direct ``python script/k8s/...py`` execution.
    from gpu_job import DEFAULT_IMAGE


DEFAULT_CONTEXT = "default"
DEFAULT_NAMESPACE = "c2c-research"
DEFAULT_NODE_A = "4090-24gx4"
DEFAULT_NODE_BC = "4090-24gx8"
DEFAULT_REPOSITORY = "https://github.com/Cyrusi6/Cache.git"
DEFAULT_NAME_PREFIX = "route1-id-v22"
DEFAULT_HF_CACHE = "/home/lijunsi/.cache/huggingface"
DEFAULT_SHARED_HOST_PATH = "/netdisk"

SHARED_MOUNT_ROOT = PurePosixPath("/netdisk")
SHARED_ROOT = SHARED_MOUNT_ROOT / "lijunsi/c2c-route1-identifiability"
WORKSPACE_ROOT = SHARED_ROOT / "workspace" / "Cache"
RUNTIME_ROOT = SHARED_ROOT / "runtime"
DEFAULT_TEMPLATE = PurePosixPath(
    "recipe/train_recipe/identifiability/route1_v22_base.json"
)
DEFAULT_REUSE_OVERRIDE = PurePosixPath(
    "recipe/train_recipe/identifiability/reuse_step1_b6.json"
)
DEFAULT_SUITE_OUTPUT = PurePosixPath("local/tmp/route1_identifiability_suite")
DEFAULT_LANE_A_PLAN = DEFAULT_SUITE_OUTPUT / "lanes/lane_a.phase1.json"
DEFAULT_LANE_B_PLAN = DEFAULT_SUITE_OUTPUT / "lanes/lane_b.phase1.json"
DEFAULT_LANE_C_PLAN = DEFAULT_SUITE_OUTPUT / "lanes/lane_c.phase1.json"
DEFAULT_STATE_DIR = DEFAULT_SUITE_OUTPUT / "lane_state"
DEFAULT_HF_REQUIREMENTS = PurePosixPath(
    "recipe/train_recipe/identifiability/hf_cache_requirements.json"
)
SUITE_SCRIPT = PurePosixPath("script/analysis/route1_identifiability_suite.py")
THIS_SCRIPT = PurePosixPath("script/k8s/route1_identifiability_lanes.py")
GATE_MOUNT = PurePosixPath("/etc/route1-gates")
GATE_FILE = GATE_MOUNT / "gates.json"

MANAGED_BY = "route1-identifiability-lanes"
GATE_STATUSES = ("pass", "pending", "fail")
PHASES = ("phase1", "conditional")
RESOURCE_COMPONENTS = ("infra", "stager", "prefetch", "lanes", "stage", "all")
FULL_GIT_SHA = re.compile(r"[0-9a-fA-F]{40}")
HF_REVISION = re.compile(r"[0-9a-fA-F]{7,40}")


class LaneInfrastructureError(RuntimeError):
    """A concise, user-facing configuration or Kubernetes error."""


@dataclass(frozen=True)
class RenderOptions:
    """Inputs that fully determine a rendered infrastructure bundle."""

    git_commit: str
    repository: str = DEFAULT_REPOSITORY
    context: str = DEFAULT_CONTEXT
    namespace: str = DEFAULT_NAMESPACE
    node_a: str = DEFAULT_NODE_A
    node_bc: str = DEFAULT_NODE_BC
    image: str = DEFAULT_IMAGE
    name_prefix: str = DEFAULT_NAME_PREFIX
    template: str = str(DEFAULT_TEMPLATE)
    reuse_override: str | None = None
    suite_output_root: str = str(DEFAULT_SUITE_OUTPUT)
    lane_a_plan: str = str(DEFAULT_LANE_A_PLAN)
    lane_b_plan: str = str(DEFAULT_LANE_B_PLAN)
    lane_c_plan: str = str(DEFAULT_LANE_C_PLAN)
    state_dir: str = str(DEFAULT_STATE_DIR)
    phase: str = "phase1"
    reproduction_gate: str = "pending"
    conditional_gate: str = "pending"
    dependency_timeout_seconds: int = 259_200
    shared_host_path: str = DEFAULT_SHARED_HOST_PATH
    hf_cache_host_path: str = DEFAULT_HF_CACHE
    hf_requirements: str = str(DEFAULT_HF_REQUIREMENTS)
    hf_expected_refs: tuple[str, ...] = field(default_factory=tuple)
    prefetch_models: tuple[str, ...] = field(default_factory=tuple)
    uid: int = field(default_factory=os.getuid)
    gid: int = field(default_factory=os.getgid)


def sanitize_name(value: str, max_length: int = 63) -> str:
    normalized = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    normalized = re.sub(r"-+", "-", normalized)
    if not normalized:
        raise LaneInfrastructureError("Kubernetes name must contain a letter or digit")
    return normalized[:max_length].rstrip("-")


def _validate_options(options: RenderOptions) -> None:
    if FULL_GIT_SHA.fullmatch(options.git_commit) is None:
        raise LaneInfrastructureError(
            "--git-commit must be an explicit 40-character Git commit SHA"
        )
    if not (
        options.repository.startswith("https://github.com/")
        and options.repository.endswith(".git")
    ):
        raise LaneInfrastructureError(
            "--repository must be a public HTTPS GitHub clone URL ending in .git"
        )
    if options.reproduction_gate not in GATE_STATUSES:
        raise LaneInfrastructureError("invalid reproduction gate status")
    if options.conditional_gate not in GATE_STATUSES:
        raise LaneInfrastructureError("invalid conditional gate status")
    if options.phase not in PHASES:
        raise LaneInfrastructureError(f"invalid identifiability phase: {options.phase}")
    if options.dependency_timeout_seconds <= 0:
        raise LaneInfrastructureError(
            "--dependency-timeout-seconds must be greater than zero"
        )
    for name, value in (
        ("--shared-host-path", options.shared_host_path),
        ("--hf-cache-host-path", options.hf_cache_host_path),
    ):
        if not Path(value).is_absolute():
            raise LaneInfrastructureError(f"{name} must be absolute")
    if Path(options.shared_host_path) != Path("/netdisk"):
        raise LaneInfrastructureError(
            "--shared-host-path must mount the autofs root /netdisk; mounting the "
            "deeper experiment directory directly is not supported"
        )
    _workspace_path(options.hf_requirements)
    requirements = load_hf_requirements(
        _local_workspace_file(options.hf_requirements), options.hf_expected_refs
    )
    model_ids = {
        item["repo_id"] for item in requirements if item.get("repo_type") == "model"
    }
    missing_prefetch_refs = sorted(set(options.prefetch_models) - model_ids)
    if missing_prefetch_refs:
        raise LaneInfrastructureError(
            "prefetch models require a fixed --hf-expected-ref: "
            + ", ".join(missing_prefetch_refs)
        )
    plans = [
        _workspace_path(options.lane_a_plan),
        _workspace_path(options.lane_b_plan),
        _workspace_path(options.lane_c_plan),
    ]
    if len(set(plans)) != 3:
        raise LaneInfrastructureError("lane A/B/C must use different plan paths")
    conditional_paths = ["conditional" in path.name for path in plans]
    if any(conditional_paths) != all(conditional_paths):
        raise LaneInfrastructureError(
            "lane A/B/C plan names disagree about the conditional phase"
        )
    if all(conditional_paths) != (options.phase == "conditional"):
        raise LaneInfrastructureError(
            f"--phase {options.phase} does not match lane plan filenames"
        )


def _workspace_path(value: str | PurePosixPath) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute():
        try:
            path.relative_to(WORKSPACE_ROOT)
        except ValueError as exc:
            raise LaneInfrastructureError(
                f"workspace path must be under {WORKSPACE_ROOT}: {path}"
            ) from exc
        return path
    if ".." in path.parts:
        raise LaneInfrastructureError(f"workspace path cannot contain '..': {path}")
    return WORKSPACE_ROOT / path


def _local_workspace_file(value: str | PurePosixPath) -> Path:
    path = Path(value)
    if path.is_absolute():
        try:
            relative = PurePosixPath(path.as_posix()).relative_to(WORKSPACE_ROOT)
        except ValueError as exc:
            raise LaneInfrastructureError(
                f"workspace path must be under {WORKSPACE_ROOT}: {path}"
            ) from exc
        return Path.cwd() / Path(*relative.parts)
    return Path.cwd() / path


def _resource_names(options: RenderOptions) -> dict[str, str]:
    base = sanitize_name(f"{options.name_prefix}-{options.git_commit[:8]}", 45)
    staging = _staging_digest(options)
    plan_a = _short_digest(options.lane_a_plan, staging)
    plan_b = _short_digest(options.lane_b_plan, staging)
    plan_c = _short_digest(options.lane_c_plan, staging)
    models = (
        _short_digest(staging, *options.prefetch_models)
        if options.prefetch_models
        else "none"
    )
    return {
        "gates": sanitize_name(f"{base}-gates"),
        "stager": sanitize_name(f"{base}-stager-{staging}"),
        "prefetch": sanitize_name(f"{base}-prefetch-{models}"),
        "lane_a": sanitize_name(f"{base}-lane-a-{plan_a}"),
        "lane_b": sanitize_name(f"{base}-lane-b-{plan_b}"),
        "lane_c": sanitize_name(f"{base}-lane-c-{plan_c}"),
    }


def _short_digest(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()[:8]


def _staging_digest(options: RenderOptions) -> str:
    return _short_digest(
        options.template,
        options.reuse_override or "",
        options.suite_output_root,
        options.lane_a_plan,
        options.lane_b_plan,
        options.lane_c_plan,
        options.hf_requirements,
        *options.hf_expected_refs,
    )


def _labels(component: str) -> dict[str, str]:
    return {
        "app.kubernetes.io/name": "c2c",
        "app.kubernetes.io/managed-by": MANAGED_BY,
        "app.kubernetes.io/component": component,
        "c2c.research/experiment": "route1-v22-identifiability",
    }


def _metadata(name: str, namespace: str, component: str) -> dict[str, Any]:
    return {
        "name": name,
        "namespace": namespace,
        "labels": _labels(component),
        "annotations": {"c2c.research/managed-by": MANAGED_BY},
    }


def _shared_volume(path: str) -> dict[str, Any]:
    return {
        "name": "shared-workspace",
        "hostPath": {"path": path, "type": "Directory"},
    }


def _hf_volume(path: str) -> dict[str, Any]:
    return {
        "name": "huggingface-cache",
        "hostPath": {"path": path, "type": "DirectoryOrCreate"},
    }


def _shared_mount() -> dict[str, str]:
    return {"name": "shared-workspace", "mountPath": str(SHARED_MOUNT_ROOT)}


def _hf_mount() -> dict[str, str]:
    return {"name": "huggingface-cache", "mountPath": "/cache/huggingface"}


def _common_env(image: str) -> list[dict[str, str]]:
    return [
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "TOKENIZERS_PARALLELISM", "value": "false"},
        {"name": "HOME", "value": str(RUNTIME_ROOT / "home")},
        {"name": "XDG_CACHE_HOME", "value": str(RUNTIME_ROOT / "home/.cache")},
        {"name": "TORCH_HOME", "value": str(RUNTIME_ROOT / "home/.cache/torch")},
        {"name": "HF_HOME", "value": "/cache/huggingface"},
        {"name": "HF_HUB_CACHE", "value": "/cache/huggingface/hub"},
        {"name": "HF_ENDPOINT", "value": "https://hf-mirror.com"},
        {"name": "HF_HUB_DOWNLOAD_TIMEOUT", "value": "600"},
        {"name": "C2C_SHARED_ROOT", "value": str(SHARED_ROOT)},
        {"name": "C2C_RUNTIME_IMAGE", "value": image},
    ]


def _lane_env(image: str, lane: str) -> list[dict[str, str]]:
    # Every lane must consume the exact same audited bytes.  Local node caches are
    # still mounted for the immutable HF provenance audit, but experiments load
    # models and datasets from the cross-node shared volume only.
    model_root = str(SHARED_ROOT / "models")
    data_root = str(SHARED_ROOT / "data/c2c")
    return [
        *_common_env(image),
        {"name": "C2C_MODEL_ROOT", "value": model_root},
        {"name": "C2C_DATA_ROOT", "value": data_root},
        {"name": "HF_HUB_OFFLINE", "value": "1"},
        {"name": "HF_DATASETS_OFFLINE", "value": "1"},
        {"name": "DATASETS_OFFLINE", "value": "1"},
        {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
    ]


def _pod_security_context(options: RenderOptions) -> dict[str, int]:
    return {
        "runAsUser": options.uid,
        "runAsGroup": options.gid,
        "fsGroup": options.gid,
    }


def build_gate_configmap(
    options: RenderOptions, names: Mapping[str, str]
) -> dict[str, Any]:
    gates = {
        "reproduction": options.reproduction_gate,
        "conditional": options.conditional_gate,
    }
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": _metadata(names["gates"], options.namespace, "stage-gates"),
        "data": {"gates.json": json.dumps(gates, indent=2) + "\n"},
    }


def _checkout_script(options: RenderOptions) -> str:
    repository = shlex.quote(options.repository)
    commit = shlex.quote(options.git_commit.lower())
    workspace = shlex.quote(str(WORKSPACE_ROOT))
    parent = shlex.quote(str(WORKSPACE_ROOT.parent))
    marker = shlex.quote(str(_checkout_marker(options)))
    pre_staged_marker = shlex.quote(str(_pre_staged_workspace_marker(options)))
    marker_payload = shlex.quote(
        json.dumps({"git_commit": options.git_commit.lower()}, separators=(",", ":"))
    )
    return "\n".join(
        [
            "set -euo pipefail",
            f"mkdir -p {parent}",
            f"if [ -f {pre_staged_marker} ]; then",
            f"  test -d {workspace}/.git",
            f"  IFS= read -r prestaged_head < {workspace}/.git/HEAD",
            f"  test \"$prestaged_head\" = {commit}",
            f"  mkdir -p {shlex.quote(str(_checkout_marker(options).parent))}",
            f"  printf '%s\\n' {marker_payload} > {marker}.tmp",
            f"  mv {marker}.tmp {marker}",
            (
                "  echo 'using pre-staged shared Git checkout at commit "
                f"{options.git_commit.lower()}'"
            ),
            "  exit 0",
            "fi",
            f"if [ -e {workspace} ] && [ ! -d {workspace}/.git ]; then",
            (
                "  echo 'workspace path exists but is not a Git checkout: "
                f"{workspace}' >&2"
            ),
            "  exit 20",
            "fi",
            f"if [ ! -d {workspace}/.git ]; then",
            f"  git clone --branch main --single-branch {repository} {workspace}",
            "fi",
            f"git -C {workspace} remote set-url origin {repository}",
            f"git -C {workspace} fetch origin main",
            f"git -C {workspace} cat-file -e {commit}^{{commit}}",
            f"git -C {workspace} merge-base --is-ancestor {commit} origin/main",
            f"git -C {workspace} checkout --detach {commit}",
            f'test "$(git -C {workspace} rev-parse HEAD)" = {commit}',
            f"git -C {workspace} status --short",
            f"git -C {workspace} diff --quiet",
            f"git -C {workspace} diff --cached --quiet",
            (
                "echo 'checked out public GitHub main commit "
                f"{options.git_commit.lower()}'"
            ),
            f"mkdir -p {shlex.quote(str(_checkout_marker(options).parent))}",
            f"printf '%s\\n' {marker_payload} > {marker}.tmp",
            f"mv {marker}.tmp {marker}",
        ]
    )


def _entrypoint_command() -> list[str]:
    return [
        "python",
        str(WORKSPACE_ROOT / "script/k8s/container_entrypoint.py"),
    ]


def _entrypoint_args(command: Sequence[str]) -> list[str]:
    return [
        "--runtime-dir",
        str(RUNTIME_ROOT),
        "--project-root",
        str(WORKSPACE_ROOT),
        "--",
        *command,
    ]


def _workspace_marker(options: RenderOptions) -> PurePosixPath:
    name = (
        f"workspace-{options.git_commit[:12].lower()}-"
        f"{_staging_digest(options)}.json"
    )
    return SHARED_ROOT / "status" / name


def _checkout_marker(options: RenderOptions) -> PurePosixPath:
    return SHARED_ROOT / "status" / (f"checkout-{options.git_commit[:12].lower()}.json")


def _pre_staged_workspace_marker(options: RenderOptions) -> PurePosixPath:
    return SHARED_ROOT / "status" / (
        f"source-prestaged-{options.git_commit.lower()}.ready"
    )


def _hf_provenance_marker(options: RenderOptions, scope: str) -> PurePosixPath:
    digest = _short_digest(
        options.git_commit.lower(),
        options.hf_requirements,
        *options.hf_expected_refs,
    )
    return SHARED_ROOT / "status" / f"hf-cache-{scope}-{digest}.json"


def _prefetch_marker(options: RenderOptions) -> PurePosixPath | None:
    if not options.prefetch_models:
        return None
    digest = hashlib.sha256(
        "\n".join(
            [
                options.git_commit.lower(),
                options.hf_requirements,
                *sorted(options.hf_expected_refs),
                *sorted(options.prefetch_models),
            ]
        ).encode("utf-8")
    ).hexdigest()[:12]
    return SHARED_ROOT / "status" / f"models-{digest}.json"


def build_stager_job(
    options: RenderOptions, names: Mapping[str, str]
) -> dict[str, Any]:
    template = _workspace_path(options.template)
    reuse_override = (
        _workspace_path(options.reuse_override)
        if options.reuse_override is not None
        else None
    )
    output_root = _workspace_path(options.suite_output_root)
    plan_a = _workspace_path(options.lane_a_plan)
    plan_b = _workspace_path(options.lane_b_plan)
    plan_c = _workspace_path(options.lane_c_plan)
    marker = _workspace_marker(options)
    requirements = _workspace_path(options.hf_requirements)
    stage_command = [
        "python",
        str(WORKSPACE_ROOT / THIS_SCRIPT),
        "stage-plans",
        "--suite",
        str(WORKSPACE_ROOT / SUITE_SCRIPT),
        "--template",
        str(template),
        "--output-root",
        str(output_root),
        "--plan-a",
        str(plan_a),
        "--plan-b",
        str(plan_b),
        "--plan-c",
        str(plan_c),
        "--ready-marker",
        str(marker),
        "--git-commit",
        options.git_commit.lower(),
    ]
    if reuse_override is not None:
        stage_command.extend(["--reuse-override", str(reuse_override)])
    init_containers: list[dict[str, Any]] = [
        {
            "name": "checkout",
            "image": options.image,
            "imagePullPolicy": "IfNotPresent",
            "command": ["/bin/bash", "-ceu"],
            "args": [_checkout_script(options)],
            "resources": {
                "requests": {"cpu": "1", "memory": "2Gi"},
                "limits": {"cpu": "2", "memory": "4Gi"},
            },
            "volumeMounts": [_shared_mount()],
        }
    ]
    prefetch_marker = _prefetch_marker(options)
    if prefetch_marker is not None:
        init_containers.append(
            _wait_for_markers_init_container(
                options,
                name="wait-for-model-prefetch",
                markers=[prefetch_marker],
            )
        )
    init_containers.append(
        _hf_audit_init_container(
            options,
            scope="stager",
            include_shared_assets=True,
            requirements=requirements,
        )
    )
    labels = _labels("workspace-stager")
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": _metadata(names["stager"], options.namespace, "workspace-stager"),
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 21_600,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "nodeSelector": {"kubernetes.io/hostname": options.node_bc},
                    "securityContext": _pod_security_context(options),
                    "initContainers": init_containers,
                    "containers": [
                        {
                            "name": "generate-plans",
                            "image": options.image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": _entrypoint_command(),
                            "args": _entrypoint_args(stage_command),
                            "env": _common_env(options.image),
                            "resources": {
                                "requests": {"cpu": "8", "memory": "24Gi"},
                                "limits": {"cpu": "12", "memory": "48Gi"},
                            },
                            "volumeMounts": [_shared_mount(), _hf_mount()],
                        }
                    ],
                    "volumes": [
                        _shared_volume(options.shared_host_path),
                        _hf_volume(options.hf_cache_host_path),
                    ],
                },
            },
        },
    }


WAIT_FOR_MARKERS_SCRIPT = """
import pathlib
import sys
import time

timeout = int(sys.argv[1])
markers = [pathlib.Path(value) for value in sys.argv[2:]]
deadline = time.monotonic() + timeout
last_report = 0.0
while True:
    missing = [str(path) for path in markers if not path.is_file()]
    if not missing:
        print("all prerequisite markers are ready", flush=True)
        raise SystemExit(0)
    now = time.monotonic()
    if now >= deadline:
        print("timed out waiting for: " + ", ".join(missing), file=sys.stderr)
        raise SystemExit(1)
    if now - last_report >= 30:
        print("waiting for: " + ", ".join(missing), flush=True)
        last_report = now
    time.sleep(5)
""".strip()


def _wait_init_container(options: RenderOptions) -> dict[str, Any]:
    markers = [_workspace_marker(options)]
    return _wait_for_markers_init_container(
        options, name="wait-for-staging", markers=markers
    )


def _wait_for_markers_init_container(
    options: RenderOptions,
    *,
    name: str,
    markers: Sequence[PurePosixPath],
) -> dict[str, Any]:
    return {
        "name": name,
        "image": options.image,
        "imagePullPolicy": "IfNotPresent",
        "command": ["python", "-c", WAIT_FOR_MARKERS_SCRIPT],
        "args": [str(options.dependency_timeout_seconds), *map(str, markers)],
        "resources": {
            "requests": {"cpu": "100m", "memory": "128Mi"},
            "limits": {"cpu": "500m", "memory": "512Mi"},
        },
        "volumeMounts": [_shared_mount()],
    }


def _hf_audit_init_container(
    options: RenderOptions,
    *,
    scope: str,
    include_shared_assets: bool,
    requirements: PurePosixPath | None = None,
) -> dict[str, Any]:
    requirements_path = requirements or _workspace_path(options.hf_requirements)
    command = [
        "python",
        str(WORKSPACE_ROOT / THIS_SCRIPT),
        "audit-hf-cache",
        "--hf-home",
        "/cache/huggingface",
        "--shared-root",
        str(SHARED_ROOT),
        "--requirements",
        str(requirements_path),
        "--ready-marker",
        str(_hf_provenance_marker(options, scope)),
    ]
    if not include_shared_assets:
        command.append("--skip-shared-assets")
    for override in options.hf_expected_refs:
        command.extend(["--expected-ref", override])
    return {
        "name": f"audit-hf-cache-{scope}",
        "image": options.image,
        "imagePullPolicy": "IfNotPresent",
        "command": _entrypoint_command(),
        "args": _entrypoint_args(command),
        "env": _common_env(options.image),
        "resources": {
            "requests": {"cpu": "2", "memory": "4Gi"},
            "limits": {"cpu": "4", "memory": "8Gi"},
        },
        "volumeMounts": [_shared_mount(), _hf_mount()],
    }


def build_lane_job(
    options: RenderOptions,
    names: Mapping[str, str],
    *,
    lane: str,
    plan: str,
) -> dict[str, Any]:
    if lane not in {"a", "b", "c"}:
        raise LaneInfrastructureError("lane must be 'a', 'b' or 'c'")
    plan_path = _workspace_path(plan)
    state_dir = _workspace_path(options.state_dir)
    command = [
        "python",
        str(WORKSPACE_ROOT / SUITE_SCRIPT),
        "run-lane",
        "--plan",
        str(plan_path),
        "--gate-file",
        str(GATE_FILE),
        "--state-dir",
        str(state_dir),
        "--reuse-complete",
        "--dependency-timeout-seconds",
        str(options.dependency_timeout_seconds),
    ]
    component = f"lane-{lane}"
    labels = _labels(component)
    node = options.node_a if lane == "a" else options.node_bc
    volume_mounts: list[dict[str, Any]] = [
        _shared_mount(),
        _hf_mount(),
        {
            "name": "stage-gates",
            "mountPath": str(GATE_MOUNT),
            "readOnly": True,
        },
        {"name": "shm", "mountPath": "/dev/shm"},
    ]
    volumes: list[dict[str, Any]] = [
        _shared_volume(options.shared_host_path),
        _hf_volume(options.hf_cache_host_path),
        {"name": "stage-gates", "configMap": {"name": names["gates"]}},
        {
            "name": "shm",
            "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"},
        },
    ]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": _metadata(names[f"lane_{lane}"], options.namespace, component),
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 7 * 24 * 3600,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "terminationGracePeriodSeconds": 60,
                    "nodeSelector": {"kubernetes.io/hostname": node},
                    "securityContext": _pod_security_context(options),
                    "initContainers": [_wait_init_container(options)],
                    "containers": [
                        {
                            "name": f"lane-{lane}",
                            "image": options.image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": _entrypoint_command(),
                            "args": _entrypoint_args(command),
                            "env": _lane_env(options.image, lane),
                            "resources": {
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
                            },
                            "volumeMounts": volume_mounts,
                        }
                    ],
                    "volumes": volumes,
                },
            },
        },
    }


def build_prefetch_job(
    options: RenderOptions, names: Mapping[str, str]
) -> dict[str, Any] | None:
    marker = _prefetch_marker(options)
    if marker is None:
        return None
    command = [
        "python",
        str(WORKSPACE_ROOT / THIS_SCRIPT),
        "prefetch-models",
        "--ready-marker",
        str(marker),
        "--requirements",
        str(_workspace_path(options.hf_requirements)),
    ]
    for override in options.hf_expected_refs:
        command.extend(["--expected-ref", override])
    for model in options.prefetch_models:
        command.extend(["--model", model])
    labels = _labels("model-prefetch")
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": _metadata(names["prefetch"], options.namespace, "model-prefetch"),
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 43_200,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "nodeSelector": {"kubernetes.io/hostname": options.node_bc},
                    "securityContext": _pod_security_context(options),
                    "initContainers": [_wait_init_container_for_workspace(options)],
                    "containers": [
                        {
                            "name": "prefetch-models",
                            "image": options.image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": _entrypoint_command(),
                            "args": _entrypoint_args(command),
                            "env": _common_env(options.image),
                            "resources": {
                                "requests": {"cpu": "8", "memory": "24Gi"},
                                "limits": {"cpu": "16", "memory": "48Gi"},
                            },
                            "volumeMounts": [_shared_mount(), _hf_mount()],
                        }
                    ],
                    "volumes": [
                        _shared_volume(options.shared_host_path),
                        _hf_volume(options.hf_cache_host_path),
                    ],
                },
            },
        },
    }


def _wait_init_container_for_workspace(options: RenderOptions) -> dict[str, Any]:
    return _wait_for_markers_init_container(
        options,
        name="wait-for-checkout",
        markers=[_checkout_marker(options)],
    )


def _normalized_components(components: Iterable[str]) -> set[str]:
    selected = set(components)
    unknown = selected - set(RESOURCE_COMPONENTS)
    if unknown:
        raise LaneInfrastructureError(
            "unknown resource components: " + ", ".join(sorted(unknown))
        )
    if "all" in selected:
        return {"infra", "stager", "prefetch", "lanes"}
    if "stage" in selected:
        selected.remove("stage")
        selected.update({"infra", "stager"})
    return selected


def _require_lane_gate(options: RenderOptions) -> None:
    gate_name = "reproduction" if options.phase == "phase1" else "conditional"
    status = (
        options.reproduction_gate
        if gate_name == "reproduction"
        else options.conditional_gate
    )
    if status != "pass":
        raise LaneInfrastructureError(
            f"refusing to create {options.phase} lane Jobs: "
            f"{gate_name}_gate={status!r}, expected 'pass'"
        )


def build_resources(
    options: RenderOptions, components: Iterable[str] = ("all",)
) -> list[dict[str, Any]]:
    """Build selected staged resources without ever contacting Kubernetes."""
    _validate_options(options)
    raw_components = tuple(components)
    selected = _normalized_components(raw_components)
    if not selected:
        raise LaneInfrastructureError("at least one resource component is required")
    if "lanes" in selected:
        _require_lane_gate(options)
    if (
        "prefetch" in selected
        and not options.prefetch_models
        and "prefetch" in raw_components
    ):
        raise LaneInfrastructureError(
            "prefetch component requested without any --prefetch-model"
        )
    names = _resource_names(options)
    resources: list[dict[str, Any]] = []
    if "infra" in selected:
        resources.append(build_gate_configmap(options, names))
    elif "lanes" in selected:
        # A dedicated lane-stage apply refreshes the gate ConfigMap atomically with
        # the Jobs, so an older pending ConfigMap cannot contradict the CLI gate.
        resources.append(build_gate_configmap(options, names))
    if "stager" in selected:
        resources.append(build_stager_job(options, names))
    if "prefetch" in selected and options.prefetch_models:
        prefetch = build_prefetch_job(options, names)
        assert prefetch is not None
        resources.append(prefetch)
    if "lanes" in selected:
        resources.extend(
            [
                build_lane_job(options, names, lane="a", plan=options.lane_a_plan),
                build_lane_job(options, names, lane="b", plan=options.lane_b_plan),
                build_lane_job(options, names, lane="c", plan=options.lane_c_plan),
            ]
        )
    return resources


def serialize_resources(resources: Sequence[Mapping[str, Any]], fmt: str) -> str:
    if fmt == "json":
        payload = {"apiVersion": "v1", "kind": "List", "items": list(resources)}
        return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if fmt == "yaml":
        return yaml.safe_dump_all(
            list(resources), sort_keys=False, allow_unicode=True, explicit_start=True
        )
    raise LaneInfrastructureError(f"unsupported output format: {fmt}")


def _run_command(
    command: Sequence[str],
    *,
    input_text: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    result = runner(
        list(command),
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        suffix = f"\n{detail}" if detail else ""
        raise LaneInfrastructureError(f"command failed: {shlex.join(command)}{suffix}")
    return result


def kubectl_preflight(
    options: RenderOptions,
    *,
    which: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Check cluster reachability, RBAC and both lane-node GPU capacities."""
    if which("kubectl") is None:
        raise LaneInfrastructureError("kubectl was not found in PATH")
    prefix = ["kubectl", "--context", options.context]
    _run_command(prefix + ["get", "namespace", options.namespace], runner=runner)
    for node_name, required_gpus in (
        (options.node_a, 4),
        (options.node_bc, 8),
    ):
        node_result = _run_command(
            prefix + ["get", "node", node_name, "-o", "json"], runner=runner
        )
        try:
            node = json.loads(node_result.stdout)
            ready = any(
                condition.get("type") == "Ready"
                and condition.get("status") == "True"
                for condition in node["status"]["conditions"]
            )
            gpu_count = int(node["status"]["allocatable"]["nvidia.com/gpu"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LaneInfrastructureError(
                f"node {node_name} returned invalid readiness/GPU data"
            ) from exc
        if not ready:
            raise LaneInfrastructureError(f"node {node_name} is not Ready")
        if gpu_count < required_gpus:
            raise LaneInfrastructureError(
                f"node {node_name} has {gpu_count} allocatable GPUs; "
                f"{required_gpus} are required"
            )
    for resource in ("configmaps", "jobs.batch"):
        result = _run_command(
            prefix
            + [
                "-n",
                options.namespace,
                "auth",
                "can-i",
                "create",
                resource,
            ],
            runner=runner,
        )
        if result.stdout.strip() != "yes":
            raise LaneInfrastructureError(
                f"current identity cannot create {resource} in {options.namespace}"
            )


def kubectl_submit(
    resources: Sequence[Mapping[str, Any]],
    options: RenderOptions,
    *,
    server_dry_run: bool,
    which: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Apply a bundle, or ask the API server to validate it without persistence."""
    kubectl_preflight(options, which=which, runner=runner)
    command = [
        "kubectl",
        "--context",
        options.context,
        "-n",
        options.namespace,
        "apply",
    ]
    if server_dry_run:
        command.append("--dry-run=server")
    command.extend(["-f", "-"])
    _run_command(
        command,
        input_text=serialize_resources(resources, "json"),
        runner=runner,
    )


def _read_plan(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LaneInfrastructureError(f"lane plan does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LaneInfrastructureError(f"lane plan is not valid JSON: {path}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
        raise LaneInfrastructureError(f"lane plan has no runs list: {path}")
    return data


def _plan_outputs(plan: Mapping[str, Any], source: Path) -> dict[PurePosixPath, str]:
    outputs: dict[PurePosixPath, str] = {}
    for index, raw_run in enumerate(plan.get("runs", [])):
        if not isinstance(raw_run, dict):
            raise LaneInfrastructureError(f"{source}: runs[{index}] is not an object")
        run_id = raw_run.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise LaneInfrastructureError(f"{source}: runs[{index}] has no run_id")
        training = raw_run.get("training", {})
        if isinstance(training, dict):
            selected = training.get("selected_checkpoint")
            if isinstance(selected, str) and selected:
                outputs[PurePosixPath(selected)] = f"{run_id}:checkpoint"
        evaluation = raw_run.get("evaluation", {})
        output_dirs = (
            evaluation.get("output_dirs", {}) if isinstance(evaluation, dict) else {}
        )
        if isinstance(output_dirs, dict):
            for dataset, output in output_dirs.items():
                if isinstance(output, str) and output:
                    outputs[PurePosixPath(output)] = f"{run_id}:eval:{dataset}"
    return outputs


def _paths_overlap(left: PurePosixPath, right: PurePosixPath) -> bool:
    if left == right:
        return True
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def validate_plan_files(
    plan_a_path: Path, plan_b_path: Path, plan_c_path: Path
) -> dict[str, Any]:
    """Validate three shared-NFS lane plans, dependencies and output isolation."""
    paths = (plan_a_path, plan_b_path, plan_c_path)
    if len({path.resolve() for path in paths}) != 3:
        raise LaneInfrastructureError("lane A/B/C plan paths must be distinct")
    plans = tuple(_read_plan(path) for path in paths)
    lane_specs = tuple(
        zip(
            ("lane_a", "lane_b", "lane_c"),
            paths,
            plans,
            ("24gx4", "24gx8", "24gx8"),
        )
    )
    for expected_lane, source, plan, node_profile in lane_specs:
        if plan.get("lane") != expected_lane:
            raise LaneInfrastructureError(
                f"{source}: expected lane={expected_lane}, got {plan.get('lane')!r}"
            )
        hardware = plan.get("hardware")
        if not isinstance(hardware, dict) or (
            hardware.get("node_profile") != node_profile
            or hardware.get("requested_gpus") != 4
        ):
            raise LaneInfrastructureError(
                f"{source}: {expected_lane} must request four GPUs on {node_profile}"
            )
    phases = {plan.get("phase") for plan in plans}
    if len(phases) != 1:
        raise LaneInfrastructureError("lane A/B/C plans must belong to the same phase")
    phase = plans[0].get("phase")
    expected_gate = {
        "phase1": "reproduction",
        "conditional": "conditional",
    }.get(phase)
    if expected_gate is None:
        raise LaneInfrastructureError(f"unsupported A/B/C lane phase: {phase!r}")
    for _lane, source, plan, _profile in lane_specs:
        for index, raw_run in enumerate(plan["runs"]):
            if not isinstance(raw_run, dict):
                continue
            gate_key = raw_run.get("gate_key")
            is_reproduction_run = (
                phase == "phase1"
                and gate_key is None
                and raw_run.get("stage") == "stage1_reproduce_b6"
            )
            if gate_key != expected_gate and not is_reproduction_run:
                raise LaneInfrastructureError(
                    f"{source}: runs[{index}] must use gate_key={expected_gate!r}"
                )
    run_sets = [
        {
            run.get("run_id")
            for run in plan["runs"]
            if isinstance(run, dict) and isinstance(run.get("run_id"), str)
        }
        for plan in plans
    ]
    all_run_ids = [run_id for run_set in run_sets for run_id in run_set]
    duplicate_runs = sorted(
        run_id for run_id in set(all_run_ids) if all_run_ids.count(run_id) > 1
    )
    if duplicate_runs:
        raise LaneInfrastructureError(
            "lane A/B/C contain duplicate run ids: " + ", ".join(duplicate_runs)
        )
    available_runs = set(all_run_ids)
    external_dependencies: list[str] = []
    for _lane, source, plan, _profile in lane_specs:
        for index, raw_run in enumerate(plan["runs"]):
            if not isinstance(raw_run, dict):
                continue
            run_id = raw_run.get("run_id", f"runs[{index}]")
            dependencies = raw_run.get("depends_on_runs", [])
            if not isinstance(dependencies, list) or not all(
                isinstance(dependency, str) for dependency in dependencies
            ):
                raise LaneInfrastructureError(
                    f"{source}: {run_id} depends_on_runs must be a string list"
                )
            for dependency in dependencies:
                if dependency not in available_runs:
                    external_dependencies.append(
                        f"{run_id} -> {dependency} ({source.name})"
                    )
    if external_dependencies:
        raise LaneInfrastructureError(
            "lane A/B/C contain dependencies on unknown runs: "
            + "; ".join(sorted(external_dependencies))
        )
    outputs = [
        _plan_outputs(plan, source)
        for _lane, source, plan, _profile in lane_specs
    ]
    conflicts: list[str] = []
    for left_index, left_outputs in enumerate(outputs):
        for right_outputs in outputs[left_index + 1:]:
            for left_path, left_owner in left_outputs.items():
                for right_path, right_owner in right_outputs.items():
                    if _paths_overlap(left_path, right_path):
                        conflicts.append(
                            f"{left_path} ({left_owner}) <-> "
                            f"{right_path} ({right_owner})"
                        )
    if conflicts:
        raise LaneInfrastructureError(
            "lane A/B/C output directory conflicts: " + "; ".join(conflicts)
        )
    return {
        "lane_a": str(plan_a_path),
        "lane_b": str(plan_b_path),
        "lane_c": str(plan_c_path),
        "lane_a_runs": len(run_sets[0]),
        "lane_b_runs": len(run_sets[1]),
        "lane_c_runs": len(run_sets[2]),
        "validated_outputs": sum(len(item) for item in outputs),
    }


def stage_plans(
    *,
    suite_path: Path,
    template_path: Path,
    output_root: Path,
    plan_a_path: Path,
    plan_b_path: Path,
    plan_c_path: Path,
    ready_marker: Path,
    git_commit: str,
    reuse_override_path: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Run suite generation verbatim, validate lane isolation, then publish ready."""
    if not template_path.is_file():
        raise LaneInfrastructureError(
            "v2.2 suite template is missing at "
            f"{template_path}; no fallback or template mutation was attempted"
        )
    if not suite_path.is_file():
        raise LaneInfrastructureError(f"suite generator is missing: {suite_path}")
    command = [
        sys.executable,
        str(suite_path),
        "generate",
        "--template",
        str(template_path),
        "--output-root",
        str(output_root),
    ]
    if reuse_override_path is not None:
        if not reuse_override_path.is_file():
            raise LaneInfrastructureError(
                f"B6 reuse override is missing: {reuse_override_path}"
            )
        command.extend(["--reuse-override", str(reuse_override_path)])
    result = runner(command, text=True, check=False)
    if result.returncode != 0:
        raise LaneInfrastructureError(
            f"suite generate failed with exit code {result.returncode}; "
            "inspect the stager Job log for the template/configuration error"
        )
    summary = validate_plan_files(plan_a_path, plan_b_path, plan_c_path)
    ready_marker.parent.mkdir(parents=True, exist_ok=True)
    ready_marker.write_text(
        json.dumps(
            {"git_commit": git_commit.lower(), "plans": summary},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def _validate_relative_cache_path(value: str, *, label: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise LaneInfrastructureError(f"{label} must be a safe relative path: {value}")


def _parse_expected_ref(value: str) -> tuple[str, str | None, str | None, str]:
    try:
        selector, revision = value.rsplit("=", 1)
    except ValueError as exc:
        raise LaneInfrastructureError(
            "--hf-expected-ref must be KEY=REVISION or model|dataset:REPO=REVISION"
        ) from exc
    revision = revision.lower()
    if HF_REVISION.fullmatch(revision) is None:
        raise LaneInfrastructureError(
            f"invalid Hugging Face revision in --hf-expected-ref: {revision!r}"
        )
    if ":" not in selector:
        if not selector:
            raise LaneInfrastructureError("empty Hugging Face requirement key")
        return selector, None, None, revision
    repo_type, repo_id = selector.split(":", 1)
    if repo_type not in {"model", "dataset"} or not repo_id:
        raise LaneInfrastructureError(
            "new --hf-expected-ref entries require model:REPO or dataset:REPO"
        )
    return selector, repo_type, repo_id, revision


def _normalize_requirement(raw: Mapping[str, Any], source: str) -> dict[str, Any]:
    required = ("key", "repo_type", "repo_id", "ref", "expected_revision")
    missing = [name for name in required if not isinstance(raw.get(name), str)]
    if missing:
        raise LaneInfrastructureError(
            f"{source} is missing string fields: {', '.join(missing)}"
        )
    item = dict(raw)
    if item["repo_type"] not in {"model", "dataset"}:
        raise LaneInfrastructureError(f"{source} has invalid repo_type")
    if not item["repo_id"] or item["repo_id"].startswith("/"):
        raise LaneInfrastructureError(f"{source} repo_id must be a Hugging Face id")
    _validate_relative_cache_path(item["ref"], label=f"{source}.ref")
    revision = item["expected_revision"].lower()
    if HF_REVISION.fullmatch(revision) is None:
        raise LaneInfrastructureError(f"{source} has invalid expected_revision")
    item["expected_revision"] = revision
    files = item.get("required_files", [])
    if not isinstance(files, list) or not all(isinstance(path, str) for path in files):
        raise LaneInfrastructureError(f"{source}.required_files must be a string list")
    for path in files:
        _validate_relative_cache_path(path, label=f"{source}.required_files")
    globs = item.get("required_globs", [])
    if not isinstance(globs, list):
        raise LaneInfrastructureError(f"{source}.required_globs must be a list")
    normalized_globs: list[dict[str, Any]] = []
    for index, glob in enumerate(globs):
        if not isinstance(glob, dict) or not isinstance(glob.get("pattern"), str):
            raise LaneInfrastructureError(
                f"{source}.required_globs[{index}] must contain pattern"
            )
        minimum = glob.get("minimum_matches")
        if not isinstance(minimum, int) or minimum <= 0:
            raise LaneInfrastructureError(
                f"{source}.required_globs[{index}].minimum_matches must be positive"
            )
        _validate_relative_cache_path(
            glob["pattern"], label=f"{source}.required_globs[{index}].pattern"
        )
        normalized_globs.append(
            {"pattern": glob["pattern"], "minimum_matches": minimum}
        )
    item["required_files"] = files
    file_sha256 = item.get("file_sha256", {})
    if not isinstance(file_sha256, dict) or not all(
        isinstance(path, str)
        and path in files
        and isinstance(digest, str)
        and re.fullmatch(r"[0-9a-fA-F]{64}", digest) is not None
        for path, digest in file_sha256.items()
    ):
        raise LaneInfrastructureError(
            f"{source}.file_sha256 must map required files to SHA256 digests"
        )
    item["file_sha256"] = {path: digest.lower() for path, digest in file_sha256.items()}
    item["required_globs"] = normalized_globs
    return item


def load_hf_requirements(
    requirements_path: Path, expected_ref_overrides: Iterable[str] = ()
) -> list[dict[str, Any]]:
    try:
        payload = json.loads(requirements_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LaneInfrastructureError(
            f"Hugging Face cache requirements are missing: {requirements_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise LaneInfrastructureError(
            f"Hugging Face cache requirements are invalid JSON: {requirements_path}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise LaneInfrastructureError(
            f"unsupported Hugging Face requirements schema: {requirements_path}"
        )
    repositories = payload.get("repositories")
    if not isinstance(repositories, list) or not repositories:
        raise LaneInfrastructureError(
            "Hugging Face requirements contain no repositories"
        )
    normalized = [
        _normalize_requirement(raw, f"repositories[{index}]")
        for index, raw in enumerate(repositories)
        if isinstance(raw, dict)
    ]
    if len(normalized) != len(repositories):
        raise LaneInfrastructureError("every Hugging Face repository must be an object")
    keys = [item["key"] for item in normalized]
    repos = [(item["repo_type"], item["repo_id"]) for item in normalized]
    if len(keys) != len(set(keys)) or len(repos) != len(set(repos)):
        raise LaneInfrastructureError(
            "Hugging Face requirement keys/repos must be unique"
        )

    by_key = {item["key"]: item for item in normalized}
    by_repo = {(item["repo_type"], item["repo_id"]): item for item in normalized}
    raw_templates = payload.get("repository_templates", [])
    if not isinstance(raw_templates, list) or not all(
        isinstance(template, dict) for template in raw_templates
    ):
        raise LaneInfrastructureError("repository_templates must be an object list")
    templates_by_key: dict[str, dict[str, Any]] = {}
    templates_by_repo: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw_template in enumerate(raw_templates):
        template = dict(raw_template)
        template["expected_revision"] = "0" * 40
        normalized_template = _normalize_requirement(
            template, f"repository_templates[{index}]"
        )
        normalized_template.pop("expected_revision")
        key = normalized_template["key"]
        repo = (
            normalized_template["repo_type"],
            normalized_template["repo_id"],
        )
        if (
            key in by_key
            or key in templates_by_key
            or repo in by_repo
            or repo in templates_by_repo
        ):
            raise LaneInfrastructureError(
                "Hugging Face repository templates must have unique keys/repos"
            )
        templates_by_key[key] = normalized_template
        templates_by_repo[repo] = normalized_template
    for raw_override in expected_ref_overrides:
        selector, repo_type, repo_id, revision = _parse_expected_ref(raw_override)
        if repo_type is None:
            if selector in by_key:
                by_key[selector]["expected_revision"] = revision
                continue
            template = templates_by_key.get(selector)
            if template is None:
                raise LaneInfrastructureError(
                    f"unknown Hugging Face requirement key: {selector}"
                )
            repo_type = template["repo_type"]
            repo_id = template["repo_id"]
        else:
            template = templates_by_repo.get((repo_type, repo_id))
        existing = by_repo.get((repo_type, repo_id))
        if existing is not None:
            existing["expected_revision"] = revision
            continue
        key = (
            template["key"]
            if template is not None
            else re.sub(r"[^a-z0-9]+", "_", repo_id.lower()).strip("_")
        )
        if key in by_key:
            raise LaneInfrastructureError(
                f"generated Hugging Face requirement key collides: {key}"
            )
        added_payload = (
            {**template, "expected_revision": revision}
            if template is not None
            else {
                "key": key,
                "repo_type": repo_type,
                "repo_id": repo_id,
                "ref": "main",
                "expected_revision": revision,
                "required_files": (
                    ["config.json", "tokenizer_config.json", "tokenizer.json"]
                    if repo_type == "model"
                    else ["README.md"]
                ),
                "required_globs": (
                    [{"pattern": "*.safetensors", "minimum_matches": 1}]
                    if repo_type == "model"
                    else []
                ),
            }
        )
        added = _normalize_requirement(
            added_payload,
            f"--hf-expected-ref {selector}",
        )
        normalized.append(added)
        by_key[key] = added
        by_repo[(repo_type, repo_id)] = added
    return normalized


def load_expected_python_packages(requirements_path: Path) -> dict[str, str]:
    try:
        payload = json.loads(requirements_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise LaneInfrastructureError(
            f"cannot load Python package provenance from {requirements_path}"
        ) from exc
    packages = payload.get("python_packages") if isinstance(payload, dict) else None
    if not isinstance(packages, dict) or not all(
        isinstance(name, str) and name and isinstance(version, str) and version
        for name, version in packages.items()
    ):
        raise LaneInfrastructureError(
            "Hugging Face requirements python_packages must be a string mapping"
        )
    return dict(packages)


def load_shared_model_requirements(requirements_path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(requirements_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise LaneInfrastructureError(
            f"cannot load shared-model provenance from {requirements_path}"
        ) from exc
    raw_models = payload.get("shared_models", []) if isinstance(payload, dict) else []
    if not isinstance(raw_models, list) or not all(
        isinstance(model, dict) for model in raw_models
    ):
        raise LaneInfrastructureError("shared_models must be an object list")
    normalized: list[dict[str, Any]] = []
    for index, raw_model in enumerate(raw_models):
        key = raw_model.get("key")
        directory = raw_model.get("directory")
        files = raw_model.get("required_files")
        hashes = raw_model.get("file_sha256")
        directory_sha256 = raw_model.get("directory_sha256")
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(directory, str)
            or not isinstance(files, list)
            or not all(isinstance(path, str) for path in files)
            or not isinstance(hashes, dict)
            or not isinstance(directory_sha256, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", directory_sha256) is None
        ):
            raise LaneInfrastructureError(f"shared_models[{index}] is malformed")
        _validate_relative_cache_path(
            directory, label=f"shared_models[{index}].directory"
        )
        for path in files:
            _validate_relative_cache_path(
                path, label=f"shared_models[{index}].required_files"
            )
        if set(hashes) != set(files) or not all(
            isinstance(digest, str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", digest) is not None
            for digest in hashes.values()
        ):
            raise LaneInfrastructureError(
                f"shared_models[{index}].file_sha256 must cover every required file"
            )
        normalized.append(
            {
                "key": key,
                "directory": directory,
                "required_files": files,
                "file_sha256": {
                    path: digest.lower() for path, digest in hashes.items()
                },
                "directory_sha256": directory_sha256.lower(),
            }
        )
    if len({item["key"] for item in normalized}) != len(normalized):
        raise LaneInfrastructureError("shared model keys must be unique")
    return normalized


def load_shared_dataset_requirements(
    requirements_path: Path,
) -> list[dict[str, Any]]:
    try:
        payload = json.loads(requirements_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise LaneInfrastructureError(
            f"cannot load shared-dataset provenance from {requirements_path}"
        ) from exc
    raw_datasets = (
        payload.get("shared_datasets", []) if isinstance(payload, dict) else []
    )
    if not isinstance(raw_datasets, list) or not all(
        isinstance(dataset, dict) for dataset in raw_datasets
    ):
        raise LaneInfrastructureError("shared_datasets must be an object list")
    normalized: list[dict[str, Any]] = []
    for index, raw_dataset in enumerate(raw_datasets):
        key = raw_dataset.get("key")
        directory = raw_dataset.get("directory")
        files = raw_dataset.get("required_files")
        directory_sha256 = raw_dataset.get("directory_sha256")
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(directory, str)
            or not isinstance(files, list)
            or not files
            or not all(isinstance(path, str) and path for path in files)
            or not isinstance(directory_sha256, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", directory_sha256) is None
        ):
            raise LaneInfrastructureError(f"shared_datasets[{index}] is malformed")
        _validate_relative_cache_path(
            directory, label=f"shared_datasets[{index}].directory"
        )
        for path in files:
            _validate_relative_cache_path(
                path, label=f"shared_datasets[{index}].required_files"
            )
        normalized.append(
            {
                "key": key,
                "directory": directory,
                "required_files": files,
                "directory_sha256": directory_sha256.lower(),
            }
        )
    if len({item["key"] for item in normalized}) != len(normalized):
        raise LaneInfrastructureError("shared dataset keys must be unique")
    return normalized


def _repo_cache_dir(hf_home: Path, requirement: Mapping[str, Any]) -> Path:
    prefix = "models" if requirement["repo_type"] == "model" else "datasets"
    return hf_home / "hub" / f"{prefix}--{requirement['repo_id'].replace('/', '--')}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_tree_sha256(path: Path) -> str:
    """Hash every regular file as relative-path NUL raw-bytes, in path order."""
    if not path.is_dir():
        raise LaneInfrastructureError(f"shared asset directory is missing: {path}")
    digest = hashlib.sha256()
    files = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    if not files:
        raise LaneInfrastructureError(f"shared asset directory is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _checked_cache_file(
    path: Path,
    repository_root: Path,
    label: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise LaneInfrastructureError(f"Hugging Face cache is missing {label}: {path}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(repository_root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise LaneInfrastructureError(
            f"Hugging Face cache file escapes/breaks its repository: {path}"
        ) from exc
    size = resolved.stat().st_size
    if size <= 0:
        raise LaneInfrastructureError(f"Hugging Face cache file is empty: {path}")
    result: dict[str, Any] = {"path": str(path), "size_bytes": size}
    if expected_sha256 is not None:
        actual_sha256 = _sha256_file(resolved)
        if actual_sha256 != expected_sha256:
            raise LaneInfrastructureError(
                f"Hugging Face cache SHA256 mismatch for {label}: "
                f"expected {expected_sha256}, found {actual_sha256}"
            )
        result["sha256"] = actual_sha256
    return result


def audit_hf_cache(
    *,
    hf_home: Path,
    shared_root: Path | None = None,
    requirements_path: Path,
    ready_marker: Path,
    expected_ref_overrides: Iterable[str] = (),
    include_shared_assets: bool = True,
    version_resolver: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, Any]:
    """Verify immutable refs and complete critical files before publishing ready."""
    requirements = load_hf_requirements(requirements_path, expected_ref_overrides)
    expected_packages = load_expected_python_packages(requirements_path)
    resolved_packages: dict[str, str] = {}
    for package, expected_version in expected_packages.items():
        try:
            actual_version = version_resolver(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise LaneInfrastructureError(
                f"required Python package is missing: {package}=={expected_version}"
            ) from exc
        if actual_version != expected_version:
            raise LaneInfrastructureError(
                f"Python package mismatch for {package}: expected {expected_version}, "
                f"found {actual_version}"
            )
        resolved_packages[package] = actual_version
    audited: list[dict[str, Any]] = []
    for item in requirements:
        repo_root = _repo_cache_dir(hf_home, item)
        ref_path = repo_root / "refs" / item["ref"]
        try:
            resolved_revision = ref_path.read_text(encoding="utf-8").strip().lower()
        except FileNotFoundError as exc:
            raise LaneInfrastructureError(
                f"Hugging Face ref is missing for {item['repo_id']}: {ref_path}"
            ) from exc
        expected_revision = item["expected_revision"]
        if not resolved_revision.startswith(expected_revision):
            raise LaneInfrastructureError(
                f"Hugging Face revision mismatch for {item['repo_id']}: "
                f"expected {expected_revision}, found {resolved_revision or '<empty>'}"
            )
        if FULL_GIT_SHA.fullmatch(resolved_revision) is None:
            raise LaneInfrastructureError(
                f"Hugging Face ref is not a full commit for {item['repo_id']}: "
                f"{resolved_revision!r}"
            )
        snapshot = repo_root / "snapshots" / resolved_revision
        if not snapshot.is_dir():
            raise LaneInfrastructureError(
                f"Hugging Face snapshot is missing for {item['repo_id']}: {snapshot}"
            )
        files = [
            _checked_cache_file(
                snapshot / relative,
                repo_root,
                f"{item['repo_id']} critical file {relative}",
                item["file_sha256"].get(relative),
            )
            for relative in item["required_files"]
        ]
        glob_counts: list[dict[str, Any]] = []
        for glob in item["required_globs"]:
            matches = sorted(snapshot.glob(glob["pattern"]))
            if len(matches) < glob["minimum_matches"]:
                raise LaneInfrastructureError(
                    f"Hugging Face snapshot {item['repo_id']} has {len(matches)} "
                    f"matches for {glob['pattern']}; {glob['minimum_matches']} required"
                )
            for match in matches:
                _checked_cache_file(
                    match, repo_root, f"{item['repo_id']} glob {glob['pattern']}"
                )
            glob_counts.append(
                {
                    "pattern": glob["pattern"],
                    "minimum_matches": glob["minimum_matches"],
                    "actual_matches": len(matches),
                }
            )
        audited.append(
            {
                "key": item["key"],
                "repo_type": item["repo_type"],
                "repo_id": item["repo_id"],
                "ref": item["ref"],
                "expected_revision": expected_revision,
                "resolved_revision": resolved_revision,
                "snapshot": str(snapshot),
                "critical_files": files,
                "glob_counts": glob_counts,
            }
        )
    audited_shared_models: list[dict[str, Any]] = []
    audited_shared_datasets: list[dict[str, Any]] = []
    if include_shared_assets:
        shared_models = load_shared_model_requirements(requirements_path)
        if shared_models and shared_root is None:
            raise LaneInfrastructureError(
                "shared_root is required when shared asset auditing is enabled"
            )
        assert shared_root is not None or not shared_models
        for model in shared_models:
            assert shared_root is not None
            model_root = shared_root / model["directory"]
            files = [
                _checked_cache_file(
                    model_root / relative,
                    shared_root,
                    f"shared model {model['key']} file {relative}",
                    model["file_sha256"][relative],
                )
                for relative in model["required_files"]
            ]
            actual_directory_sha256 = _directory_tree_sha256(model_root)
            if actual_directory_sha256 != model["directory_sha256"]:
                raise LaneInfrastructureError(
                    f"shared model directory SHA256 mismatch for {model['key']}: "
                    f"expected {model['directory_sha256']}, "
                    f"found {actual_directory_sha256}"
                )
            audited_shared_models.append(
                {
                    "key": model["key"],
                    "directory": str(model_root),
                    "directory_sha256": actual_directory_sha256,
                    "critical_files": files,
                }
            )
        shared_datasets = load_shared_dataset_requirements(requirements_path)
        if shared_datasets and shared_root is None:
            raise LaneInfrastructureError(
                "shared_root is required when shared dataset auditing is enabled"
            )
        assert shared_root is not None or not shared_datasets
        for dataset in shared_datasets:
            assert shared_root is not None
            dataset_root = shared_root / dataset["directory"]
            files = [
                _checked_cache_file(
                    dataset_root / relative,
                    shared_root,
                    f"shared dataset {dataset['key']} file {relative}",
                )
                for relative in dataset["required_files"]
            ]
            actual_directory_sha256 = _directory_tree_sha256(dataset_root)
            if actual_directory_sha256 != dataset["directory_sha256"]:
                raise LaneInfrastructureError(
                    f"shared dataset SHA256 mismatch for {dataset['key']}: "
                    f"expected {dataset['directory_sha256']}, "
                    f"found {actual_directory_sha256}"
                )
            audited_shared_datasets.append(
                {
                    "key": dataset["key"],
                    "directory": str(dataset_root),
                    "directory_sha256": actual_directory_sha256,
                    "critical_files": files,
                }
            )
    payload = {
        "schema_version": 1,
        "status": "ready",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requirements_sha256": hashlib.sha256(
            requirements_path.read_bytes()
        ).hexdigest(),
        "python_packages": resolved_packages,
        "repositories": audited,
        "shared_models": audited_shared_models,
        "shared_datasets": audited_shared_datasets,
    }
    ready_marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = ready_marker.with_suffix(ready_marker.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(ready_marker)
    return payload


def prefetch_models(
    models: Iterable[str],
    ready_marker: Path,
    *,
    requirements_path: Path,
    expected_ref_overrides: Iterable[str] = (),
    downloader: Callable[..., str] | None = None,
) -> None:
    unique_models = list(dict.fromkeys(model for model in models if model))
    if not unique_models:
        raise LaneInfrastructureError("prefetch-models requires at least one --model")
    requirements = load_hf_requirements(requirements_path, expected_ref_overrides)
    by_model = {
        item["repo_id"]: item for item in requirements if item["repo_type"] == "model"
    }
    missing = sorted(set(unique_models) - set(by_model))
    if missing:
        raise LaneInfrastructureError(
            "prefetch models lack a fixed expected revision: " + ", ".join(missing)
        )
    if downloader is None:
        from huggingface_hub import snapshot_download

        downloader = snapshot_download
    resolved: list[dict[str, str]] = []
    for model in unique_models:
        requirement = by_model[model]
        expected = requirement["expected_revision"]
        print(f"prefetching {model} at verified {requirement['ref']} ref", flush=True)
        snapshot_path = Path(
            downloader(repo_id=model, revision=requirement["ref"], repo_type="model")
        )
        revision = snapshot_path.name.lower()
        if (
            not revision.startswith(expected)
            or FULL_GIT_SHA.fullmatch(revision) is None
        ):
            raise LaneInfrastructureError(
                f"prefetched revision mismatch for {model}: expected {expected}, "
                f"found {revision}"
            )
        resolved.append({"model": model, "revision": revision})
    ready_marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = ready_marker.with_suffix(ready_marker.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"models": resolved}, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(ready_marker)


def _add_render_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--context", default=DEFAULT_CONTEXT)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--node-a", default=DEFAULT_NODE_A)
    parser.add_argument("--node-bc", default=DEFAULT_NODE_BC)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--reuse-override", default=None)
    parser.add_argument("--suite-output-root", default=str(DEFAULT_SUITE_OUTPUT))
    parser.add_argument("--lane-a-plan", default=str(DEFAULT_LANE_A_PLAN))
    parser.add_argument("--lane-b-plan", default=str(DEFAULT_LANE_B_PLAN))
    parser.add_argument("--lane-c-plan", default=str(DEFAULT_LANE_C_PLAN))
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--phase", choices=PHASES, default="phase1")
    parser.add_argument("--reproduction-gate", choices=GATE_STATUSES, default="pending")
    parser.add_argument("--conditional-gate", choices=GATE_STATUSES, default="pending")
    parser.add_argument("--dependency-timeout-seconds", type=int, default=259_200)
    parser.add_argument("--shared-host-path", default=DEFAULT_SHARED_HOST_PATH)
    parser.add_argument("--hf-cache-host-path", default=DEFAULT_HF_CACHE)
    parser.add_argument("--hf-requirements", default=str(DEFAULT_HF_REQUIREMENTS))
    parser.add_argument(
        "--hf-expected-ref",
        action="append",
        default=[],
        help=(
            "override KEY=REVISION or add model|dataset:ORG/REPO=REVISION; "
            "repeatable"
        ),
    )
    parser.add_argument("--prefetch-model", action="append", default=[])
    parser.add_argument("--uid", type=int, default=os.getuid())
    parser.add_argument("--gid", type=int, default=os.getgid())
    parser.add_argument("--format", choices=("yaml", "json"), default="yaml")
    parser.add_argument(
        "--component",
        action="append",
        choices=RESOURCE_COMPONENTS,
        default=[],
        help=(
            "render/apply a safe stage (default: stage=infra+stager); use a "
            "separate prefetch, then lanes invocation"
        ),
    )
    parser.add_argument("--output", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--server-dry-run", action="store_true")
    action.add_argument("--apply", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render shared-NFS Kubernetes infrastructure for Route-1 lanes A/B/C"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    render_parser = subparsers.add_parser(
        "render", help="render YAML/JSON; contact Kubernetes only with an explicit flag"
    )
    _add_render_arguments(render_parser)

    validate_parser = subparsers.add_parser(
        "validate-plans", help="verify A/B/C run and output isolation"
    )
    validate_parser.add_argument("--plan-a", type=Path, required=True)
    validate_parser.add_argument("--plan-b", type=Path, required=True)
    validate_parser.add_argument("--plan-c", type=Path, required=True)

    stage_parser = subparsers.add_parser(
        "stage-plans", help="internal stager entrypoint used by the Kubernetes Job"
    )
    stage_parser.add_argument("--suite", type=Path, required=True)
    stage_parser.add_argument("--template", type=Path, required=True)
    stage_parser.add_argument("--output-root", type=Path, required=True)
    stage_parser.add_argument("--plan-a", type=Path, required=True)
    stage_parser.add_argument("--plan-b", type=Path, required=True)
    stage_parser.add_argument("--plan-c", type=Path, required=True)
    stage_parser.add_argument("--ready-marker", type=Path, required=True)
    stage_parser.add_argument("--git-commit", required=True)
    stage_parser.add_argument("--reuse-override", type=Path, default=None)

    prefetch_parser = subparsers.add_parser(
        "prefetch-models", help="internal optional Hugging Face cache warmer"
    )
    prefetch_parser.add_argument("--model", action="append", required=True)
    prefetch_parser.add_argument("--ready-marker", type=Path, required=True)
    prefetch_parser.add_argument("--requirements", type=Path, required=True)
    prefetch_parser.add_argument("--expected-ref", action="append", default=[])

    audit_parser = subparsers.add_parser(
        "audit-hf-cache",
        help="verify fixed HF refs/files and atomically publish provenance",
    )
    audit_parser.add_argument("--hf-home", type=Path, required=True)
    audit_parser.add_argument("--shared-root", type=Path)
    audit_parser.add_argument("--requirements", type=Path, required=True)
    audit_parser.add_argument("--ready-marker", type=Path, required=True)
    audit_parser.add_argument("--expected-ref", action="append", default=[])
    audit_parser.add_argument("--skip-shared-assets", action="store_true")
    return parser


def _options_from_args(args: argparse.Namespace) -> RenderOptions:
    return RenderOptions(
        git_commit=args.git_commit,
        repository=args.repository,
        context=args.context,
        namespace=args.namespace,
        node_a=args.node_a,
        node_bc=args.node_bc,
        image=args.image,
        name_prefix=args.name_prefix,
        template=args.template,
        reuse_override=args.reuse_override,
        suite_output_root=args.suite_output_root,
        lane_a_plan=args.lane_a_plan,
        lane_b_plan=args.lane_b_plan,
        lane_c_plan=args.lane_c_plan,
        state_dir=args.state_dir,
        phase=args.phase,
        reproduction_gate=args.reproduction_gate,
        conditional_gate=args.conditional_gate,
        dependency_timeout_seconds=args.dependency_timeout_seconds,
        shared_host_path=args.shared_host_path,
        hf_cache_host_path=args.hf_cache_host_path,
        hf_requirements=args.hf_requirements,
        hf_expected_refs=tuple(args.hf_expected_ref),
        prefetch_models=tuple(args.prefetch_model),
        uid=args.uid,
        gid=args.gid,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-plans":
            summary = validate_plan_files(args.plan_a, args.plan_b, args.plan_c)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0
        if args.command == "stage-plans":
            summary = stage_plans(
                suite_path=args.suite,
                template_path=args.template,
                output_root=args.output_root,
                plan_a_path=args.plan_a,
                plan_b_path=args.plan_b,
                plan_c_path=args.plan_c,
                ready_marker=args.ready_marker,
                git_commit=args.git_commit,
                reuse_override_path=args.reuse_override,
            )
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0
        if args.command == "prefetch-models":
            prefetch_models(
                args.model,
                args.ready_marker,
                requirements_path=args.requirements,
                expected_ref_overrides=args.expected_ref,
            )
            return 0
        if args.command == "audit-hf-cache":
            payload = audit_hf_cache(
                hf_home=args.hf_home,
                shared_root=args.shared_root,
                requirements_path=args.requirements,
                ready_marker=args.ready_marker,
                expected_ref_overrides=args.expected_ref,
                include_shared_assets=not args.skip_shared_assets,
            )
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        options = _options_from_args(args)
        components = tuple(args.component) if args.component else ("stage",)
        normalized_components = _normalized_components(components)
        if not options.prefetch_models:
            normalized_components.discard("prefetch")
        if (
            args.apply
            and "lanes" in normalized_components
            and len(normalized_components) > 1
        ):
            raise LaneInfrastructureError(
                "--apply with lanes must be a dedicated --component lanes call; "
                "wait for stager/prefetch completion first"
            )
        resources = build_resources(options, components)
        rendered = serialize_resources(resources, args.format)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
            print(f"rendered={args.output}", file=sys.stderr)
        else:
            print(rendered, end="")
        if args.server_dry_run or args.apply:
            kubectl_submit(
                resources,
                options,
                server_dry_run=bool(args.server_dry_run),
            )
            mode = "server-dry-run" if args.server_dry_run else "apply"
            print(f"kubectl_mode={mode}", file=sys.stderr)
        else:
            print("kubectl_mode=render-only", file=sys.stderr)
        print(
            "components="
            + ",".join(sorted(normalized_components))
            + f" lane_a_node={options.node_a} lane_b_c_node={options.node_bc}",
            file=sys.stderr,
        )
        return 0
    except (LaneInfrastructureError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
