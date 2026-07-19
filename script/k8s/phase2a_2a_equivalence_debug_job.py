#!/usr/bin/env python3
"""Render the single-GPU serial Phase 2A-2a equivalence debug Job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml

try:
    from script.k8s.gpu_job import DEFAULT_IMAGE
except ModuleNotFoundError:
    from gpu_job import DEFAULT_IMAGE


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTEXT = "default"
DEFAULT_NAMESPACE = "c2c-research"
DEFAULT_NODE = "4090-24gx4"
EXPERIMENT_ROOT = PurePosixPath(
    "/netdisk/lijunsi/c2c-phase2a2-equivalence-debug"
)
WORKSPACE_ROOT = EXPERIMENT_ROOT / "workspace/Cache"
RESULTS_ROOT = EXPERIMENT_ROOT / "results/diagnostic"
RUNTIME_ROOT = PurePosixPath(
    "/netdisk/lijunsi/c2c-route1-identifiability/runtime"
)
# Match the HOME used by the successful Phase 2A-2a Gate-1 run.
PHASE2A2_HOME = PurePosixPath(
    "/netdisk/lijunsi/c2c-phase2a2-cache-geometry/runtime/home"
)
FULL_SHA = re.compile(r"[0-9a-f]{40}")


class EquivalenceJobError(RuntimeError):
    """Fail-closed Kubernetes renderer error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    command: Sequence[str], *, input_text: str | None = None
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command), input=input_text, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise EquivalenceJobError(f"command failed: {' '.join(command)}\n{detail}")
    return result


def validate_manifest(path: Path, expected_commit: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    constraints = value.get("constraints", {})
    if value.get("phase") != "Phase 2A-2a equivalence debug":
        raise EquivalenceJobError("unexpected manifest phase")
    if value.get("code_commit") != expected_commit:
        raise EquivalenceJobError("manifest code commit mismatch")
    required = {
        "evaluation_only": True,
        "training_forbidden": True,
        "selector_forbidden": True,
        "geometry_predictability_forbidden": True,
        "mmlu_forbidden": True,
        "sealed_test_forbidden": True,
        "one_visible_physical_gpu": True,
        "serial_execution": True,
    }
    if any(constraints.get(key) is not expected for key, expected in required.items()):
        raise EquivalenceJobError("manifest constraints changed")
    runs = value.get("runs", [])
    expected = {
        (condition, dataset)
        for condition in ("off_a", "off_b", "on_a", "on_b", "noop_a", "noop_b")
        for dataset in ("ai2-arc", "openbookqa")
    }
    actual = {(run.get("condition"), run.get("dataset")) for run in runs}
    if len(runs) != 12 or actual != expected:
        raise EquivalenceJobError("manifest run matrix changed")
    if any(run.get("training_forbidden") is not True for run in runs):
        raise EquivalenceJobError("run without explicit training prohibition")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "run_count": len(runs),
    }


WAIT_FOR_WORKSPACE = r"""
import hashlib
import pathlib
import subprocess
import sys
import time

workspace = pathlib.Path(sys.argv[1])
manifest = pathlib.Path(sys.argv[2])
expected_commit = sys.argv[3]
expected_manifest_sha = sys.argv[4]
deadline = time.monotonic() + int(sys.argv[5])
while True:
    try:
        try:
            head = subprocess.run(
                ["git", "-C", str(workspace), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.SubprocessError):
            head = (workspace / ".git/HEAD").read_text().strip()
        manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        script_ok = (
            workspace / "script/analysis/phase2a_2a_equivalence_debug.py"
        ).is_file()
        if head == expected_commit and manifest_sha == expected_manifest_sha and script_ok:
            print("exact Phase 2A-2a equivalence-debug workspace ready", flush=True)
            raise SystemExit(0)
    except Exception as exc:
        print(f"waiting for exact workspace: {exc}", flush=True)
    if time.monotonic() >= deadline:
        raise SystemExit("timed out waiting for exact equivalence-debug workspace")
    time.sleep(5)
""".strip()


def build_job(
    *, git_commit: str, manifest_path: Path, image: str = DEFAULT_IMAGE,
    namespace: str = DEFAULT_NAMESPACE, node: str = DEFAULT_NODE,
    uid: int = os.getuid(), gid: int = os.getgid(), supplemental_gid: int = 31000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if FULL_SHA.fullmatch(git_commit) is None:
        raise EquivalenceJobError("git commit must be a lowercase full SHA")
    summary = validate_manifest(manifest_path, git_commit)
    manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    pod_manifest = PurePosixPath(str(manifest_path))
    if PurePosixPath(manifest_value["workspace_root"]) != WORKSPACE_ROOT:
        raise EquivalenceJobError("manifest workspace root changed")
    if not str(pod_manifest).startswith(str(EXPERIMENT_ROOT) + "/"):
        raise EquivalenceJobError("manifest must live under the experiment root")
    name = re.sub(
        r"[^a-z0-9-]+",
        "-",
        f"p2a2-eq-debug-{git_commit[:8]}-{summary['sha256'][:8]}",
    ).strip("-")[:63].rstrip("-")
    labels = {
        "app.kubernetes.io/name": "c2c",
        "app.kubernetes.io/managed-by": "phase2a2-equivalence-debug-job",
        "c2c.research/experiment": "phase2a2-equivalence-debug",
    }
    init = {
        "name": "wait-for-exact-workspace",
        "image": image,
        "command": ["python", "-c", WAIT_FOR_WORKSPACE],
        "args": [
            str(WORKSPACE_ROOT),
            str(pod_manifest),
            git_commit,
            summary["sha256"],
            "3600",
        ],
        "resources": {
            "requests": {"cpu": "100m", "memory": "128Mi"},
            "limits": {"cpu": "500m", "memory": "512Mi"},
        },
        "volumeMounts": [{"name": "netdisk", "mountPath": "/netdisk"}],
    }
    command = [
        "python",
        str(WORKSPACE_ROOT / "script/analysis/phase2a_2a_equivalence_debug.py"),
        "execute",
        "--manifest",
        str(pod_manifest),
        "--output-root",
        str(RESULTS_ROOT),
    ]
    container = {
        "name": "equivalence-debug",
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        "command": [
            "python", str(WORKSPACE_ROOT / "script/k8s/container_entrypoint.py")
        ],
        "args": [
            "--runtime-dir",
            str(RUNTIME_ROOT),
            "--project-root",
            str(WORKSPACE_ROOT),
            "--",
            *command,
        ],
        "env": [
            {"name": "PYTHONUNBUFFERED", "value": "1"},
            {"name": "TOKENIZERS_PARALLELISM", "value": "false"},
            {"name": "HOME", "value": str(PHASE2A2_HOME)},
            {"name": "XDG_CACHE_HOME", "value": str(PHASE2A2_HOME / ".cache")},
            {"name": "TORCH_HOME", "value": str(PHASE2A2_HOME / ".cache/torch")},
            {"name": "HF_HOME", "value": "/cache/huggingface"},
            {"name": "HF_HUB_OFFLINE", "value": "1"},
            {"name": "HF_DATASETS_OFFLINE", "value": "1"},
            {"name": "DATASETS_OFFLINE", "value": "1"},
            {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
            {
                "name": "C2C_DATA_ROOT",
                "value": "/netdisk/lijunsi/c2c-route1-identifiability/data/c2c",
            },
            {"name": "C2C_RUNTIME_IMAGE", "value": image},
            {
                "name": "PIP_CONSTRAINT",
                "value": str(
                    WORKSPACE_ROOT
                    / "recipe/train_recipe/identifiability/runtime_constraints.txt"
                ),
            },
            {"name": "C2C_PRESERVE_CUDA_VISIBLE_DEVICES", "value": "1"},
            {
                "name": "C2C_NODE_NAME",
                "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}},
            },
            {
                "name": "C2C_POD_NAMESPACE",
                "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}},
            },
        ],
        "resources": {
            "requests": {"cpu": "12", "memory": "48Gi", "nvidia.com/gpu": "1"},
            "limits": {"cpu": "20", "memory": "64Gi", "nvidia.com/gpu": "1"},
        },
        "volumeMounts": [
            {"name": "netdisk", "mountPath": "/netdisk"},
            {"name": "hf-cache", "mountPath": "/cache/huggingface"},
            {"name": "shm", "mountPath": "/dev/shm"},
        ],
    }
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": labels,
            "annotations": {
                "c2c.research/git-commit": git_commit,
                "c2c.research/manifest-sha256": summary["sha256"],
                "c2c.research/run-order": "off-a,off-b,on-a,on-b,conditional-noop",
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 21600,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "terminationGracePeriodSeconds": 60,
                    "nodeSelector": {"kubernetes.io/hostname": node},
                    "securityContext": {
                        "runAsUser": uid,
                        "runAsGroup": gid,
                        "fsGroup": gid,
                        "supplementalGroups": [supplemental_gid],
                    },
                    "initContainers": [init],
                    "containers": [container],
                    "volumes": [
                        {
                            "name": "netdisk",
                            "hostPath": {"path": "/netdisk", "type": "Directory"},
                        },
                        {"name": "hf-cache", "emptyDir": {}},
                        {
                            "name": "shm",
                            "emptyDir": {"medium": "Memory", "sizeLimit": "24Gi"},
                        },
                    ],
                },
            },
        },
    }
    return job, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--context", default=DEFAULT_CONTEXT)
    parser.add_argument("--node", default=DEFAULT_NODE)
    parser.add_argument("--server-dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    job, summary = build_job(
        git_commit=args.git_commit,
        manifest_path=args.manifest,
        image=args.image,
        namespace=args.namespace,
        node=args.node,
    )
    rendered = yaml.safe_dump(job, sort_keys=False, explicit_start=True)
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if args.server_dry_run:
        if shutil.which("kubectl") is None:
            raise EquivalenceJobError("kubectl not found")
        _run(
            [
                "kubectl",
                "--context",
                args.context,
                "-n",
                args.namespace,
                "apply",
                "--dry-run=server",
                "-f",
                "-",
            ],
            input_text=rendered,
        )
    print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
