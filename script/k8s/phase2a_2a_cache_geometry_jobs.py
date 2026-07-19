#!/usr/bin/env python3
"""Render and run the two isolated Phase 2A-2a Kubernetes node jobs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import yaml

try:
    from script.k8s.gpu_job import DEFAULT_IMAGE
except ModuleNotFoundError:
    from gpu_job import DEFAULT_IMAGE


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTEXT = "default"
DEFAULT_NAMESPACE = "c2c-research"
DEFAULT_SHARED_HOST_PATH = "/netdisk"
EXPERIMENT_ROOT = PurePosixPath("/netdisk/lijunsi/c2c-phase2a2-cache-geometry")
WORKSPACE_ROOT = PurePosixPath(
    os.environ.get(
        "C2C_PHASE2A2_WORKSPACE_ROOT", str(EXPERIMENT_ROOT / "workspace/Cache")
    )
)
RUNTIME_ROOT = EXPERIMENT_ROOT / "runtime"
DEFAULT_MANIFEST = Path("local/tmp/phase2a_2a_cache_geometry/execution_manifest.json")
DEFAULT_STATE_DIR = EXPERIMENT_ROOT / "k8s_state"
PILOT_SCRIPT = WORKSPACE_ROOT / "script/analysis/phase2a_2a_cache_geometry_pilot.py"
RUNTIME_CONSTRAINTS = (
    WORKSPACE_ROOT / "recipe/train_recipe/identifiability/runtime_constraints.txt"
)
FULL_SHA = re.compile(r"[0-9a-f]{40}")


class GeometryJobError(RuntimeError):
    """Fail-closed renderer or worker error."""


@dataclass(frozen=True)
class NodePlan:
    node: str
    gpu_request: int
    pairs: tuple[str, ...]
    cpu_request: str
    cpu_limit: str
    memory_request: str
    memory_limit: str
    shm_size: str


NODE_PLANS = (
    NodePlan(
        "4090-24gx8", 8, ("tinyllama", "qwen25_0p5b"),
        "32", "48", "128Gi", "192Gi", "96Gi",
    ),
    NodePlan(
        "4090-24gx4", 4, ("llama32_1b",),
        "18", "28", "64Gi", "96Gi", "48Gi",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_command(
    command: Sequence[str], *, input_text: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    result = runner(
        list(command), input=input_text, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise GeometryJobError(f"command failed: {shlex.join(command)}\n{detail}")
    return result


def _manifest_paths(value: Path) -> tuple[Path, PurePosixPath]:
    if value.is_absolute():
        resolved = value.resolve()
        try:
            relative = resolved.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise GeometryJobError("manifest must be inside the isolated worktree") from exc
    else:
        if ".." in value.parts:
            raise GeometryJobError("manifest cannot contain '..'")
        resolved = (REPO_ROOT / value).resolve()
        relative = resolved.relative_to(REPO_ROOT.resolve())
    return resolved, WORKSPACE_ROOT / relative.as_posix()


def validate_manifest(path: Path, expected_commit: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    constraints = value.get("constraints", {})
    if value.get("phase") != "Phase 2A-2a":
        raise GeometryJobError("unexpected execution manifest phase")
    if value.get("code_commit") != expected_commit:
        raise GeometryJobError("execution manifest commit does not match render commit")
    if constraints.get("training_forbidden") is not True:
        raise GeometryJobError("manifest does not forbid training")
    if constraints.get("allowed_seed") != [42] or constraints.get("allowed_split") != "fit":
        raise GeometryJobError("manifest is outside the frozen seed/split scope")
    instrumented = [run for run in value.get("runs", []) if run.get("kind") == "instrumented"]
    combinations = {(run.get("pair"), run.get("dataset"), run.get("seed")) for run in instrumented}
    expected = {
        (pair, dataset, 42)
        for pair in ("tinyllama", "qwen25_0p5b", "llama32_1b")
        for dataset in ("ai2-arc", "openbookqa", "mmlu-redux")
    }
    if combinations != expected or len(instrumented) != 9:
        raise GeometryJobError("manifest is not the frozen 3-pair x 3-task pilot")
    if any(run.get("training_forbidden") is not True for run in value.get("runs", [])):
        raise GeometryJobError("a run does not explicitly forbid training")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "run_count": len(value.get("runs", [])),
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
            workspace / "script/analysis/phase2a_2a_cache_geometry_pilot.py"
        ).is_file()
        if head == expected_commit and manifest_sha == expected_manifest_sha and script_ok:
            print("exact Phase 2A-2a workspace ready", flush=True)
            raise SystemExit(0)
    except Exception as exc:
        print(f"waiting for exact workspace: {exc}", flush=True)
    if time.monotonic() >= deadline:
        raise SystemExit("timed out waiting for exact Phase 2A-2a workspace")
    time.sleep(5)
""".strip()


def _environment(plan: NodePlan, image: str) -> list[dict[str, str]]:
    return [
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "TOKENIZERS_PARALLELISM", "value": "false"},
        {"name": "HOME", "value": str(RUNTIME_ROOT / "home")},
        {"name": "XDG_CACHE_HOME", "value": str(RUNTIME_ROOT / "home/.cache")},
        {"name": "TORCH_HOME", "value": str(RUNTIME_ROOT / "home/.cache/torch")},
        {"name": "HF_HOME", "value": "/cache/huggingface"},
        {"name": "HF_HUB_OFFLINE", "value": "1"},
        {"name": "HF_DATASETS_OFFLINE", "value": "1"},
        {"name": "DATASETS_OFFLINE", "value": "1"},
        {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
        {"name": "C2C_DATA_ROOT", "value": "/netdisk/lijunsi/c2c-route1-identifiability/data/c2c"},
        {"name": "C2C_PHASE2A2_WORKSPACE_ROOT", "value": str(WORKSPACE_ROOT)},
        {"name": "C2C_RUNTIME_IMAGE", "value": image},
        {"name": "PIP_CONSTRAINT", "value": str(RUNTIME_CONSTRAINTS)},
        {"name": "C2C_PRESERVE_CUDA_VISIBLE_DEVICES", "value": "1"},
        {"name": "C2C_PHASE2A2_PAIRS", "value": ",".join(plan.pairs)},
    ]


def build_jobs(
    *, git_commit: str, manifest_path: Path, image: str = DEFAULT_IMAGE,
    namespace: str = DEFAULT_NAMESPACE, name_prefix: str = "p2a2-geometry",
    shared_host_path: str = DEFAULT_SHARED_HOST_PATH,
    max_startup_used_mib: int = 4096, dependency_timeout_seconds: int = 86400,
    uid: int = os.getuid(), gid: int = os.getgid(), supplemental_gid: int = 31000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if FULL_SHA.fullmatch(git_commit) is None:
        raise GeometryJobError("git commit must be a lowercase full SHA")
    if shared_host_path != "/netdisk":
        raise GeometryJobError("shared hostPath must be /netdisk")
    local_manifest, pod_manifest = _manifest_paths(manifest_path)
    summary = validate_manifest(local_manifest, git_commit)
    jobs = []
    for plan in NODE_PLANS:
        suffix = plan.node.replace("4090-", "").replace("gx", "g")
        name = re.sub(
            r"[^a-z0-9-]+", "-",
            f"{name_prefix}-{git_commit[:8]}-{summary['sha256'][:8]}-{suffix}".lower(),
        ).strip("-")[:63].rstrip("-")
        labels = {
            "app.kubernetes.io/name": "c2c",
            "app.kubernetes.io/managed-by": "phase2a2-cache-geometry-jobs",
            "c2c.research/experiment": "phase2a2-cache-geometry-pilot",
        }
        init = {
            "name": "wait-for-exact-workspace",
            "image": image,
            "command": ["python", "-c", WAIT_FOR_WORKSPACE],
            "args": [
                str(WORKSPACE_ROOT), str(pod_manifest), git_commit,
                summary["sha256"], str(dependency_timeout_seconds),
            ],
            "resources": {
                "requests": {"cpu": "100m", "memory": "128Mi"},
                "limits": {"cpu": "500m", "memory": "512Mi"},
            },
            "volumeMounts": [{"name": "netdisk", "mountPath": "/netdisk"}],
        }
        command = [
            "python", str(WORKSPACE_ROOT / "script/k8s/phase2a_2a_cache_geometry_jobs.py"),
            "run-node", "--manifest", str(pod_manifest),
            "--expected-manifest-sha256", summary["sha256"],
            "--pairs", ",".join(plan.pairs),
            "--state-dir", str(DEFAULT_STATE_DIR / plan.node),
            "--max-startup-used-mib", str(max_startup_used_mib),
        ]
        container = {
            "name": "phase2a2-cache-geometry",
            "image": image,
            "command": ["python", str(WORKSPACE_ROOT / "script/k8s/container_entrypoint.py")],
            "args": [
                "--runtime-dir", str(RUNTIME_ROOT),
                "--project-root", str(WORKSPACE_ROOT), "--", *command,
            ],
            "env": _environment(plan, image),
            "resources": {
                "requests": {
                    "cpu": plan.cpu_request, "memory": plan.memory_request,
                    "nvidia.com/gpu": str(plan.gpu_request),
                },
                "limits": {
                    "cpu": plan.cpu_limit, "memory": plan.memory_limit,
                    "nvidia.com/gpu": str(plan.gpu_request),
                },
            },
            "volumeMounts": [
                {"name": "netdisk", "mountPath": "/netdisk"},
                {"name": "hf-cache", "mountPath": "/cache/huggingface"},
                {"name": "shm", "mountPath": "/dev/shm"},
            ],
        }
        jobs.append(
            {
                "apiVersion": "batch/v1", "kind": "Job",
                "metadata": {
                    "name": name, "namespace": namespace, "labels": labels,
                    "annotations": {
                        "c2c.research/git-commit": git_commit,
                        "c2c.research/manifest-sha256": summary["sha256"],
                        "c2c.research/pairs": ",".join(plan.pairs),
                    },
                },
                "spec": {
                    "backoffLimit": 0, "activeDeadlineSeconds": 86400,
                    "template": {
                        "metadata": {"labels": labels},
                        "spec": {
                            "restartPolicy": "Never",
                            "terminationGracePeriodSeconds": 60,
                            "nodeSelector": {"kubernetes.io/hostname": plan.node},
                            "securityContext": {
                                "runAsUser": uid, "runAsGroup": gid, "fsGroup": gid,
                                "supplementalGroups": [supplemental_gid],
                            },
                            "initContainers": [init], "containers": [container],
                            "volumes": [
                                {"name": "netdisk", "hostPath": {"path": shared_host_path, "type": "Directory"}},
                                {"name": "hf-cache", "emptyDir": {}},
                                {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": plan.shm_size}},
                            ],
                        },
                    },
                },
            }
        )
    return jobs, summary


def _gpu_inventory(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[tuple[str, int]]:
    result = runner(
        ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        if line.strip():
            uuid, used = [part.strip() for part in line.split(",", 1)]
            rows.append((uuid, int(used)))
    if len({uuid for uuid, _used in rows}) != len(rows):
        raise GeometryJobError("duplicate GPU UUIDs")
    return rows


def _workspace_head(workspace: Path) -> str:
    try:
        return _run_command(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"]
        ).stdout.strip()
    except (FileNotFoundError, GeometryJobError):
        head = (workspace / ".git/HEAD").read_text(encoding="utf-8").strip()
        if FULL_SHA.fullmatch(head) is None:
            raise GeometryJobError("shared workspace is not detached at an exact commit")
        return head


def run_node(
    *, manifest: Path, expected_manifest_sha256: str, pairs: Sequence[str],
    state_dir: Path, max_startup_used_mib: int,
) -> int:
    if _sha256(manifest) != expected_manifest_sha256:
        raise GeometryJobError("manifest SHA changed after rendering")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    head = _workspace_head(Path(WORKSPACE_ROOT))
    if head != value["code_commit"]:
        raise GeometryJobError("workspace commit changed after freezing manifest")
    pairs = tuple(pairs)
    allowed = {"tinyllama", "qwen25_0p5b", "llama32_1b"}
    if not pairs or len(set(pairs)) != len(pairs) or not set(pairs) <= allowed:
        raise GeometryJobError("invalid or duplicate pair assignment")
    inventory = _gpu_inventory()
    idle = [item for item in inventory if item[1] <= max_startup_used_mib]
    if len(idle) < 2 * len(pairs):
        raise GeometryJobError(
            f"not enough physically idle GPUs for {pairs}: inventory={inventory}"
        )
    assignments = {
        pair: idle[index * 2 : index * 2 + 2]
        for index, pair in enumerate(pairs)
    }
    base = {
        "schema_version": 1,
        "manifest": str(manifest),
        "manifest_sha256": expected_manifest_sha256,
        "pairs": list(pairs),
        "inventory": [
            {"uuid": uuid, "used_mib": used, "eligible": used <= max_startup_used_mib}
            for uuid, used in inventory
        ],
        "assignments": {
            pair: [{"uuid": uuid, "used_mib": used} for uuid, used in rows]
            for pair, rows in assignments.items()
        },
    }
    _atomic_json(
        state_dir / "started.json",
        {**base, "started_at": datetime.now(timezone.utc).isoformat()},
    )

    def run_one(pair: str) -> tuple[str, int]:
        child_env = dict(os.environ)
        child_env["CUDA_VISIBLE_DEVICES"] = ",".join(
            uuid for uuid, _used in assignments[pair]
        )
        child_env["C2C_PRESERVE_CUDA_VISIBLE_DEVICES"] = "1"
        result = subprocess.run(
            [
                sys.executable, str(PILOT_SCRIPT), "run-pair",
                "--manifest", str(manifest), "--pair", pair,
            ],
            cwd=str(WORKSPACE_ROOT), env=child_env, check=False,
        )
        return pair, int(result.returncode)

    with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
        results = list(pool.map(run_one, pairs))
    failures = [{"pair": pair, "return_code": code} for pair, code in results if code]
    finished = datetime.now(timezone.utc).isoformat()
    if failures:
        _atomic_json(state_dir / "failed.json", {**base, "failed_at": finished, "failures": failures})
        return failures[0]["return_code"]
    _atomic_json(state_dir / "completed.json", {**base, "completed_at": finished, "results": dict(results)})
    return 0


def serialize_jobs(jobs: Sequence[Mapping[str, Any]]) -> str:
    return yaml.safe_dump_all(jobs, sort_keys=False, explicit_start=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render")
    render.add_argument("--git-commit", required=True)
    render.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    render.add_argument("--output", type=Path)
    render.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    render.add_argument("--context", default=DEFAULT_CONTEXT)
    render.add_argument("--image", default=DEFAULT_IMAGE)
    render.add_argument("--server-dry-run", action="store_true")
    render.add_argument("--max-startup-used-mib", type=int, default=4096)
    node = sub.add_parser("run-node")
    node.add_argument("--manifest", type=Path, required=True)
    node.add_argument("--expected-manifest-sha256", required=True)
    node.add_argument("--pairs", required=True)
    node.add_argument("--state-dir", type=Path, required=True)
    node.add_argument("--max-startup-used-mib", type=int, default=4096)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run-node":
        return run_node(
            manifest=args.manifest,
            expected_manifest_sha256=args.expected_manifest_sha256,
            pairs=tuple(item for item in args.pairs.split(",") if item),
            state_dir=args.state_dir,
            max_startup_used_mib=args.max_startup_used_mib,
        )
    jobs, summary = build_jobs(
        git_commit=args.git_commit, manifest_path=args.manifest,
        image=args.image, namespace=args.namespace,
        max_startup_used_mib=args.max_startup_used_mib,
    )
    rendered = serialize_jobs(jobs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if args.server_dry_run:
        if shutil.which("kubectl") is None:
            raise GeometryJobError("kubectl not found")
        _run_command(
            ["kubectl", "--context", args.context, "-n", args.namespace,
             "apply", "--dry-run=server", "-f", "-"],
            input_text=rendered,
        )
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
