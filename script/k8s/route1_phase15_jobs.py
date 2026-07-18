#!/usr/bin/env python3
"""Render and validate three node-level Phase-1.5 Kubernetes workers.

The renderer is deliberately non-submitting: it can render locally or ask the
Kubernetes API server to dry-run the Jobs.  Execution uses one immutable
intervention manifest and the resume semantics in
``route1_phase15_interventions.py``.  Each Job reserves its complete Kubernetes
GPU allocation, filters physically busy devices with ``nvidia-smi``, and runs
the seven logical two-GPU shards through idle UUID pairs.  This protects the
experiment from GPU users that are invisible to the Kubernetes scheduler.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
import time
from typing import Any, Callable, Mapping, Sequence

import yaml

try:
    from script.k8s.gpu_job import DEFAULT_IMAGE
except ModuleNotFoundError:  # Direct script execution.
    from gpu_job import DEFAULT_IMAGE


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTEXT = "default"
DEFAULT_NAMESPACE = "c2c-research"
DEFAULT_NAME_PREFIX = "r1-p15"
DEFAULT_SHARED_HOST_PATH = "/netdisk"
EXPERIMENT_ROOT = PurePosixPath("/netdisk/lijunsi/c2c-route1-identifiability")
WORKSPACE_ROOT = EXPERIMENT_ROOT / "workspace/Cache"
RUNTIME_ROOT = EXPERIMENT_ROOT / "runtime"
DEFAULT_MANIFEST = Path("local/tmp/phase1_5_causal_diagnostics/manifest.json")
DEFAULT_STATE_DIR = WORKSPACE_ROOT / "local/tmp/phase1_5_causal_diagnostics/k8s_state"
INTERVENTION_SCRIPT = WORKSPACE_ROOT / "script/analysis/route1_phase15_interventions.py"
THIS_SCRIPT = WORKSPACE_ROOT / "script/k8s/route1_phase15_jobs.py"
RUNTIME_CONSTRAINTS = (
    WORKSPACE_ROOT / "recipe/train_recipe/identifiability/runtime_constraints.txt"
)
FULL_GIT_SHA = re.compile(r"[0-9a-fA-F]{40}")
DATASETS = ("ai2-arc", "openbookqa", "mmlu-redux")
EXPECTED_PAIRS = {"tinyllama", "qwen3_1p7b", "qwen25_0p5b", "llama32_1b"}
EXPECTED_SEEDS = {42, 43, 44}
EXPECTED_INTERVENTIONS = {
    "b2_eval_k4",
    "b3_eval_k1",
    "b6_entropy_constant",
    "b6_entropy_shuffled",
    "b6_gate_static",
    "b6_gate_forced_on",
}


class Phase15JobError(RuntimeError):
    """Concise renderer, validation, or cluster-preflight error."""


@dataclass(frozen=True)
class NodeWorker:
    node: str
    profile: str
    shard_indices: tuple[int, ...]
    gpu_request: int
    max_parallel_shards: int
    cpu_request: str
    cpu_limit: str
    memory_request: str
    memory_limit: str
    shm_size: str


NODE_WORKERS = (
    # Reserve all four cards, but use one idle UUID pair at a time.  If one
    # physical card is occupied outside Kubernetes, both logical shards still
    # finish serially on the remaining idle pair.
    NodeWorker(
        "4090-24gx4", "24gx4-node", (0, 1), 4, 1,
        "20", "28", "72Gi", "96Gi", "32Gi",
    ),
    # At most three pairs are used even when all eight cards are idle.  This
    # leaves the fourth pair as slack and also tolerates one scheduler-invisible
    # busy GPU while keeping three shards concurrent.
    NodeWorker(
        "4090-24gx8", "24gx8-node", (2, 3, 4, 5), 8, 3,
        "36", "48", "144Gi", "192Gi", "96Gi",
    ),
    NodeWorker(
        "4090-48gx2", "48gx2-node", (6,), 2, 1,
        "24", "32", "96Gi", "112Gi", "32Gi",
    ),
)


@dataclass(frozen=True)
class RenderOptions:
    git_commit: str
    execution_manifest: Path = DEFAULT_MANIFEST
    context: str = DEFAULT_CONTEXT
    namespace: str = DEFAULT_NAMESPACE
    image: str = DEFAULT_IMAGE
    name_prefix: str = DEFAULT_NAME_PREFIX
    shared_host_path: str = DEFAULT_SHARED_HOST_PATH
    state_dir: PurePosixPath = DEFAULT_STATE_DIR
    dependency_timeout_seconds: int = 259_200
    max_startup_used_mib: int = 4096
    uid: int = field(default_factory=os.getuid)
    gid: int = field(default_factory=os.getgid)
    supplemental_gid: int = 31000


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


def _manifest_paths(value: Path) -> tuple[Path, PurePosixPath]:
    if not value.is_absolute():
        if ".." in value.parts:
            raise Phase15JobError("execution manifest cannot contain '..'")
        return (REPO_ROOT / value).resolve(), WORKSPACE_ROOT / value.as_posix()
    resolved = value.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        try:
            PurePosixPath(resolved.as_posix()).relative_to(WORKSPACE_ROOT)
        except ValueError as exc:
            raise Phase15JobError(
                f"execution manifest must be under {REPO_ROOT} or {WORKSPACE_ROOT}"
            ) from exc
        return resolved, PurePosixPath(resolved.as_posix())
    return resolved, WORKSPACE_ROOT / relative.as_posix()


def validate_execution_manifest(path: Path, num_shards: int = 7) -> dict[str, Any]:
    """Validate the complete 4-pair x 3-seed x 6-intervention eval matrix."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Phase15JobError(f"execution manifest is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise Phase15JobError(f"execution manifest is invalid JSON: {path}") from exc
    runs = manifest.get("runs") if isinstance(manifest, dict) else None
    if manifest.get("schema_version") != 1 or not isinstance(runs, list):
        raise Phase15JobError("execution manifest must be schema_version=1 with runs")
    if num_shards != 7:
        raise Phase15JobError("Phase 1.5 Max7 requires exactly seven shards")

    combinations: set[tuple[str, int, str]] = set()
    run_ids: set[str] = set()
    outputs: set[str] = set()
    shard_counts = [0] * num_shards
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise Phase15JobError(f"runs[{index}] is not an object")
        run_id = run.get("id")
        if not isinstance(run_id, str) or not run_id or run_id in run_ids:
            raise Phase15JobError(f"runs[{index}] has a missing or duplicate id")
        run_ids.add(run_id)
        pair = run.get("pair")
        seed = run.get("seed")
        intervention = run.get("intervention")
        intervention_id = (
            intervention.get("id") if isinstance(intervention, dict) else intervention
        )
        if pair not in EXPECTED_PAIRS or seed not in EXPECTED_SEEDS:
            raise Phase15JobError(f"{run_id} has an unexpected pair or seed")
        if intervention_id not in EXPECTED_INTERVENTIONS:
            raise Phase15JobError(f"{run_id} has an unexpected intervention")
        combinations.add((str(pair), int(seed), str(intervention_id)))
        checkpoint = run.get("checkpoint", {})
        if checkpoint.get("same_checkpoint_no_training") is not True:
            raise Phase15JobError(f"{run_id} is not a no-training intervention")
        eval_configs = run.get("eval_configs")
        output_dirs = run.get("output_dirs")
        if not isinstance(eval_configs, dict) or set(eval_configs) != set(DATASETS):
            raise Phase15JobError(f"{run_id} does not have the three eval configs")
        if not isinstance(output_dirs, dict) or set(output_dirs) != set(DATASETS):
            raise Phase15JobError(f"{run_id} does not have the three output dirs")
        for dataset, raw_output in output_dirs.items():
            output = str(raw_output)
            if output in outputs:
                raise Phase15JobError(f"duplicate output dir: {output}")
            outputs.add(output)
        shard_counts[index % num_shards] += 1

    expected = {
        (pair, seed, intervention)
        for pair in EXPECTED_PAIRS
        for seed in EXPECTED_SEEDS
        for intervention in EXPECTED_INTERVENTIONS
    }
    if combinations != expected or len(runs) != 72:
        missing = sorted(expected - combinations)
        extra = sorted(combinations - expected)
        raise Phase15JobError(
            f"execution matrix must contain exactly 72 runs; missing={missing}, extra={extra}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "run_count": len(runs),
        "shard_run_counts": shard_counts,
        "output_dir_count": len(outputs),
    }


def _sanitize_name(value: str) -> str:
    result = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    result = re.sub(r"-+", "-", result)
    if not result:
        raise Phase15JobError("invalid Kubernetes name")
    return result[:63].rstrip("-")


def _common_env(options: RenderOptions, worker: NodeWorker) -> list[dict[str, str]]:
    return [
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "TOKENIZERS_PARALLELISM", "value": "false"},
        {"name": "HOME", "value": str(RUNTIME_ROOT / "home")},
        {"name": "XDG_CACHE_HOME", "value": str(RUNTIME_ROOT / "home/.cache")},
        {"name": "TORCH_HOME", "value": str(RUNTIME_ROOT / "home/.cache/torch")},
        {"name": "HF_HOME", "value": "/cache/huggingface"},
        {"name": "HF_HUB_CACHE", "value": "/cache/huggingface/hub"},
        {"name": "HF_HUB_OFFLINE", "value": "1"},
        {"name": "HF_DATASETS_OFFLINE", "value": "1"},
        {"name": "DATASETS_OFFLINE", "value": "1"},
        {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
        {"name": "C2C_SHARED_ROOT", "value": str(EXPERIMENT_ROOT)},
        {"name": "C2C_MODEL_ROOT", "value": str(EXPERIMENT_ROOT / "models")},
        {"name": "C2C_DATA_ROOT", "value": str(EXPERIMENT_ROOT / "data/c2c")},
        {"name": "C2C_RUNTIME_IMAGE", "value": options.image},
        {"name": "PIP_CONSTRAINT", "value": str(RUNTIME_CONSTRAINTS)},
        {
            "name": "C2C_PHASE15_SHARDS",
            "value": ",".join(str(index) for index in worker.shard_indices),
        },
        # unified_evaluator historically removed CUDA_VISIBLE_DEVICES.  The
        # node launcher assigns UUID pairs explicitly, so preserving the mask
        # is required for isolation between concurrent two-GPU shards.
        {"name": "C2C_PRESERVE_CUDA_VISIBLE_DEVICES", "value": "1"},
    ]


WAIT_FOR_EXACT_WORKSPACE = r"""
import hashlib
import pathlib
import subprocess
import sys
import time

workspace = pathlib.Path(sys.argv[1])
manifest = pathlib.Path(sys.argv[2])
expected_commit = sys.argv[3]
expected_sha = sys.argv[4]
timeout = int(sys.argv[5])
deadline = time.monotonic() + timeout
last = 0.0
while True:
    reason = "workspace not ready"
    try:
        try:
            head = subprocess.run(
                ["git", "-C", str(workspace), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.SubprocessError):
            # The fixed PyTorch runtime image may not contain git.  The shared
            # workspace is deliberately detached, so a literal 40-byte HEAD is
            # an equivalent immutable revision check.
            head = (workspace / ".git/HEAD").read_text().strip()
            if len(head) != 40 or any(ch not in "0123456789abcdefABCDEF" for ch in head):
                raise RuntimeError("detached .git/HEAD is not an exact commit")
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        script_ok = (workspace / "script/k8s/route1_phase15_jobs.py").is_file()
        if head == expected_commit and digest == expected_sha and script_ok:
            print("exact Phase-1.5 workspace and manifest are ready", flush=True)
            raise SystemExit(0)
        reason = f"head={head} manifest_sha={digest} script_ok={script_ok}"
    except Exception as exc:
        reason = str(exc)
    now = time.monotonic()
    if now >= deadline:
        print("timed out waiting for exact workspace: " + reason, file=sys.stderr)
        raise SystemExit(1)
    if now - last >= 30:
        print("waiting for exact workspace: " + reason, flush=True)
        last = now
    time.sleep(5)
""".strip()


def build_jobs(options: RenderOptions) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if FULL_GIT_SHA.fullmatch(options.git_commit) is None:
        raise Phase15JobError("--git-commit must be an explicit 40-character SHA")
    if Path(options.shared_host_path) != Path("/netdisk"):
        raise Phase15JobError("shared hostPath must be the /netdisk autofs root")
    if options.dependency_timeout_seconds <= 0 or options.max_startup_used_mib < 0:
        raise Phase15JobError("timeouts and GPU startup threshold must be valid")
    local_manifest, pod_manifest = _manifest_paths(options.execution_manifest)
    summary = validate_execution_manifest(local_manifest)
    manifest_sha = summary["sha256"]
    jobs = []
    for worker in NODE_WORKERS:
        node_state = options.state_dir / worker.node
        suffix = worker.node.replace("4090-", "").replace("gx", "g")
        name = _sanitize_name(
            f"{options.name_prefix}-{options.git_commit[:8]}-{manifest_sha[:8]}-"
            f"{suffix}"
        )
        labels = {
            "app.kubernetes.io/name": "c2c",
            "app.kubernetes.io/managed-by": "route1-phase15-jobs",
            "app.kubernetes.io/component": f"phase15-node-{worker.node}",
            "c2c.research/experiment": "route1-phase15-causal-diagnostics",
        }
        metadata = {
            "name": name,
            "namespace": options.namespace,
            "labels": labels,
            "annotations": {
                "c2c.research/git-commit": options.git_commit.lower(),
                "c2c.research/execution-manifest-sha256": manifest_sha,
                "c2c.research/shard-indices": ",".join(
                    str(index) for index in worker.shard_indices
                ),
                "c2c.research/shard-run-counts": ",".join(
                    str(summary["shard_run_counts"][index])
                    for index in worker.shard_indices
                ),
                "c2c.research/hardware-profile": worker.profile,
                "c2c.research/max-parallel-shards": str(
                    worker.max_parallel_shards
                ),
            },
        }
        init = {
            "name": "wait-for-exact-workspace",
            "image": options.image,
            "imagePullPolicy": "IfNotPresent",
            "command": ["python", "-c", WAIT_FOR_EXACT_WORKSPACE],
            "args": [
                str(WORKSPACE_ROOT),
                str(pod_manifest),
                options.git_commit.lower(),
                manifest_sha,
                str(options.dependency_timeout_seconds),
            ],
            "resources": {
                "requests": {"cpu": "100m", "memory": "128Mi"},
                "limits": {"cpu": "500m", "memory": "512Mi"},
            },
            "volumeMounts": [{"name": "shared-workspace", "mountPath": "/netdisk"}],
        }
        command = [
            "python",
            str(THIS_SCRIPT),
            "run-node",
            "--execution-manifest",
            str(pod_manifest),
            "--expected-manifest-sha256",
            manifest_sha,
            "--shard-indices",
            ",".join(str(index) for index in worker.shard_indices),
            "--num-shards",
            "7",
            "--state-dir",
            str(node_state),
            "--max-startup-used-mib",
            str(options.max_startup_used_mib),
            "--max-parallel-shards",
            str(worker.max_parallel_shards),
        ]
        container = {
            "name": f"phase15-node-{worker.node}",
            "image": options.image,
            "imagePullPolicy": "IfNotPresent",
            "command": ["python", str(WORKSPACE_ROOT / "script/k8s/container_entrypoint.py")],
            "args": [
                "--runtime-dir", str(RUNTIME_ROOT),
                "--project-root", str(WORKSPACE_ROOT),
                "--", *command,
            ],
            "env": _common_env(options, worker),
            "resources": {
                "requests": {
                    "cpu": worker.cpu_request,
                    "memory": worker.memory_request,
                    "nvidia.com/gpu": str(worker.gpu_request),
                },
                "limits": {
                    "cpu": worker.cpu_limit,
                    "memory": worker.memory_limit,
                    "nvidia.com/gpu": str(worker.gpu_request),
                },
            },
            "volumeMounts": [
                {"name": "shared-workspace", "mountPath": "/netdisk"},
                {"name": "huggingface-cache", "mountPath": "/cache/huggingface"},
                {"name": "shm", "mountPath": "/dev/shm"},
            ],
        }
        jobs.append(
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": metadata,
                "spec": {
                    "backoffLimit": 0,
                    "activeDeadlineSeconds": 7 * 24 * 3600,
                    "template": {
                        "metadata": {"labels": labels},
                        "spec": {
                            "restartPolicy": "Never",
                            "terminationGracePeriodSeconds": 60,
                            "nodeSelector": {"kubernetes.io/hostname": worker.node},
                            "securityContext": {
                                "runAsUser": options.uid,
                                "runAsGroup": options.gid,
                                "fsGroup": options.gid,
                                "supplementalGroups": [options.supplemental_gid],
                            },
                            "initContainers": [init],
                            "containers": [container],
                            "volumes": [
                                {
                                    "name": "shared-workspace",
                                    "hostPath": {
                                        "path": options.shared_host_path,
                                        "type": "Directory",
                                    },
                                },
                                {"name": "huggingface-cache", "emptyDir": {}},
                                {
                                    "name": "shm",
                                    "emptyDir": {
                                        "medium": "Memory",
                                        "sizeLimit": worker.shm_size,
                                    },
                                },
                            ],
                        },
                    },
                },
            }
        )
    return jobs, summary


def serialize_jobs(jobs: Sequence[Mapping[str, Any]], fmt: str) -> str:
    if fmt == "json":
        return json.dumps({"apiVersion": "v1", "kind": "List", "items": jobs}, indent=2) + "\n"
    return yaml.safe_dump_all(jobs, sort_keys=False, explicit_start=True)


def _run_command(
    command: Sequence[str], *, input_text: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    result = runner(list(command), input=input_text, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise Phase15JobError(f"command failed: {shlex.join(command)}\n{detail}")
    return result


def kubectl_preflight(
    options: RenderOptions, *,
    which: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    if which("kubectl") is None:
        raise Phase15JobError("kubectl was not found in PATH")
    prefix = ["kubectl", "--context", options.context]
    _run_command(prefix + ["get", "namespace", options.namespace], runner=runner)
    required = {"4090-24gx4": 4, "4090-24gx8": 8, "4090-48gx2": 2}
    for node_name, required_gpus in required.items():
        result = _run_command(prefix + ["get", "node", node_name, "-o", "json"], runner=runner)
        node = json.loads(result.stdout)
        ready = any(
            item.get("type") == "Ready" and item.get("status") == "True"
            for item in node.get("status", {}).get("conditions", [])
        )
        gpus = int(node.get("status", {}).get("allocatable", {}).get("nvidia.com/gpu", 0))
        if not ready or gpus < required_gpus:
            raise Phase15JobError(
                f"node {node_name} is not Ready with {required_gpus} allocatable GPUs"
            )
    auth = _run_command(
        prefix + ["-n", options.namespace, "auth", "can-i", "create", "jobs.batch"],
        runner=runner,
    )
    if auth.stdout.strip() != "yes":
        raise Phase15JobError("current identity cannot create jobs.batch")


def server_dry_run(
    jobs: Sequence[Mapping[str, Any]], options: RenderOptions, *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    preflight: Callable[..., None] = kubectl_preflight,
) -> None:
    preflight(options, runner=runner)
    _run_command(
        [
            "kubectl", "--context", options.context, "-n", options.namespace,
            "apply", "--dry-run=server", "-f", "-",
        ],
        input_text=serialize_jobs(jobs, "json"),
        runner=runner,
    )


def _query_gpu_memory(
    *, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[tuple[str, int]]:
    """Return the physical GPUs visible to the node-level container."""
    result = runner(
        ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    rows: list[tuple[str, int]] = []
    for line in result.stdout.splitlines():
        if line.strip():
            try:
                uuid, used = (part.strip() for part in line.split(",", maxsplit=1))
                rows.append((uuid, int(used)))
            except (TypeError, ValueError) as exc:
                raise Phase15JobError(
                    f"invalid nvidia-smi GPU row: {line!r}"
                ) from exc
    if not rows:
        raise Phase15JobError("nvidia-smi did not report any visible GPUs")
    if len({uuid for uuid, _used in rows}) != len(rows):
        raise Phase15JobError("nvidia-smi reported duplicate GPU UUIDs")
    return rows


def _select_idle_gpu_groups(
    max_used_mib: int, *,
    max_groups: int,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[list[tuple[list[str], list[int]]], list[tuple[str, int]]]:
    if max_groups < 1:
        raise Phase15JobError("max_groups must be positive")
    rows = _query_gpu_memory(runner=runner)
    healthy = [item for item in rows if item[1] <= max_used_mib]
    group_count = min(len(healthy) // 2, max_groups)
    if len(rows) < 2 or group_count < 1:
        raise Phase15JobError(
            "fewer than two visible GPUs are idle enough: "
            f"rows={rows}, threshold={max_used_mib}"
        )
    groups = []
    for index in range(group_count):
        selected = healthy[index * 2 : index * 2 + 2]
        groups.append(
            ([item[0] for item in selected], [item[1] for item in selected])
        )
    return groups, rows


def _shard_run_ids(manifest: Mapping[str, Any], shard_index: int, num_shards: int) -> list[str]:
    return [
        run["id"]
        for index, run in enumerate(manifest["runs"])
        if index % num_shards == shard_index
    ]


def _completed_shard_state(
    state_dir: Path, *, manifest_sha256: str, shard_index: int,
) -> bool:
    path = state_dir / f"shard_{shard_index:02d}" / "completed.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return (
        value.get("manifest_sha256") == manifest_sha256
        and value.get("shard_index") == shard_index
        and value.get("return_code", 0) == 0
    )


def _resolve_output_dir(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def _precreate_assigned_output_dirs(
    manifest: Mapping[str, Any],
    *,
    shard_indices: Sequence[int],
    num_shards: int,
    attempts: int = 20,
) -> None:
    """Create assigned result trees serially with NFS/autofs retry semantics."""
    assigned = set(shard_indices)
    paths = sorted(
        {
            _resolve_output_dir(str(output_dir))
            for index, run in enumerate(manifest["runs"])
            if index % num_shards in assigned
            for output_dir in run["output_dirs"].values()
        },
        key=str,
    )
    for path in paths:
        last_error: OSError | None = None
        for attempt in range(attempts):
            try:
                path.mkdir(parents=True, exist_ok=True)
                if path.is_dir():
                    break
            except (FileExistsError, FileNotFoundError, OSError) as exc:
                last_error = exc
                if path.is_dir():
                    break
            time.sleep(min(0.1 * (attempt + 1), 1.0))
        else:
            raise Phase15JobError(
                f"failed to pre-create NFS output directory {path}: {last_error}"
            )


def _run_shard_on_gpu_pair(
    *, execution_manifest: Path, manifest: Mapping[str, Any],
    manifest_sha256: str, shard_index: int, num_shards: int,
    state_dir: Path, selected_uuids: Sequence[str], used_mib: Sequence[int],
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> int:
    if len(selected_uuids) != 2 or len(used_mib) != 2:
        raise Phase15JobError("each logical shard requires exactly one two-GPU pair")
    shard_state = state_dir / f"shard_{shard_index:02d}"
    run_ids = _shard_run_ids(manifest, shard_index, num_shards)
    child_env = dict(os.environ)
    child_env["CUDA_VISIBLE_DEVICES"] = ",".join(selected_uuids)
    child_env["C2C_PRESERVE_CUDA_VISIBLE_DEVICES"] = "1"
    child_env["C2C_PHASE15_GPU_UUIDS"] = ",".join(selected_uuids)
    base = {
        "schema_version": 2,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "manifest": str(execution_manifest.resolve()),
        "manifest_sha256": manifest_sha256,
        "run_ids": run_ids,
        "selected_gpu_uuids": list(selected_uuids),
        "selected_gpu_used_mib_at_start": list(used_mib),
    }
    _atomic_json(
        shard_state / "started.json",
        {**base, "started_at": datetime.now(timezone.utc).isoformat()},
    )
    command = [
        sys.executable, str(INTERVENTION_SCRIPT), "run-shard",
        "--manifest", str(execution_manifest.resolve()),
        "--shard-index", str(shard_index), "--num-shards", str(num_shards),
    ]
    result = runner(command, cwd=str(WORKSPACE_ROOT), env=child_env, check=False)
    finished = datetime.now(timezone.utc).isoformat()
    if result.returncode == 0:
        _atomic_json(
            shard_state / "completed.json",
            {**base, "completed_at": finished, "return_code": 0},
        )
    else:
        _atomic_json(
            shard_state / "failed.json",
            {**base, "failed_at": finished, "return_code": int(result.returncode)},
        )
    return int(result.returncode)


def run_shard(
    *, execution_manifest: Path, expected_manifest_sha256: str,
    shard_index: int, num_shards: int, state_dir: Path,
    max_startup_used_mib: int,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> int:
    if num_shards != 7 or not 0 <= shard_index < num_shards:
        raise Phase15JobError("require seven shards and shard-index in [0, 6]")
    actual_sha = _sha256(execution_manifest)
    if actual_sha != expected_manifest_sha256:
        raise Phase15JobError("execution manifest SHA changed after rendering")
    manifest = json.loads(execution_manifest.read_text(encoding="utf-8"))
    groups, _inventory = _select_idle_gpu_groups(
        max_startup_used_mib, max_groups=1, runner=runner
    )
    selected_uuids, used_mib = groups[0]
    # Preserve the old CLI for one-off recovery, but use the same per-shard
    # state layout and UUID-isolated implementation as run-node.
    return _run_shard_on_gpu_pair(
        execution_manifest=execution_manifest,
        manifest=manifest,
        manifest_sha256=actual_sha,
        shard_index=shard_index,
        num_shards=num_shards,
        state_dir=state_dir.parent if state_dir.name == f"shard_{shard_index:02d}" else state_dir,
        selected_uuids=selected_uuids,
        used_mib=used_mib,
        runner=runner,
    )


def run_node(
    *, execution_manifest: Path, expected_manifest_sha256: str,
    shard_indices: Sequence[int], num_shards: int, state_dir: Path,
    max_startup_used_mib: int, max_parallel_shards: int,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> int:
    """Run several modulo shards on disjoint, physically idle UUID pairs."""
    if num_shards != 7:
        raise Phase15JobError("require exactly seven logical shards")
    normalized = tuple(int(index) for index in shard_indices)
    if not normalized or len(set(normalized)) != len(normalized):
        raise Phase15JobError("shard-indices must be non-empty and unique")
    if any(index < 0 or index >= num_shards for index in normalized):
        raise Phase15JobError("shard-indices must be in [0, 6]")
    if max_parallel_shards < 1:
        raise Phase15JobError("max-parallel-shards must be positive")
    actual_sha = _sha256(execution_manifest)
    if actual_sha != expected_manifest_sha256:
        raise Phase15JobError("execution manifest SHA changed after rendering")
    manifest = json.loads(execution_manifest.read_text(encoding="utf-8"))
    pending = [
        index for index in normalized
        if not _completed_shard_state(
            state_dir, manifest_sha256=actual_sha, shard_index=index
        )
    ]
    if not pending:
        _atomic_json(
            state_dir / "completed.json",
            {
                "schema_version": 2,
                "manifest_sha256": actual_sha,
                "shard_indices": list(normalized),
                "skipped_completed_shards": list(normalized),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return 0

    # Do this once per node, before any evaluator subprocesses are launched.
    # Cross-node autofs/NFS directory caches can briefly report a parent as
    # missing while another node reports EEXIST for the same parent.
    _precreate_assigned_output_dirs(
        manifest,
        shard_indices=pending,
        num_shards=num_shards,
    )

    groups, inventory = _select_idle_gpu_groups(
        max_startup_used_mib,
        max_groups=min(max_parallel_shards, len(pending)),
        runner=runner,
    )
    schedules = [pending[index::len(groups)] for index in range(len(groups))]
    selected = [
        {
            "gpu_uuids": uuids,
            "used_mib_at_start": used,
            "shard_indices": schedules[index],
        }
        for index, (uuids, used) in enumerate(groups)
    ]
    node_base = {
        "schema_version": 2,
        "manifest": str(execution_manifest.resolve()),
        "manifest_sha256": actual_sha,
        "shard_indices": list(normalized),
        "pending_shard_indices": pending,
        "max_parallel_shards": max_parallel_shards,
        "max_startup_used_mib": max_startup_used_mib,
        "gpu_inventory": [
            {
                "uuid": uuid,
                "used_mib": used,
                "idle_eligible": used <= max_startup_used_mib,
            }
            for uuid, used in inventory
        ],
        "selected_gpu_groups": selected,
    }
    _atomic_json(
        state_dir / "started.json",
        {**node_base, "started_at": datetime.now(timezone.utc).isoformat()},
    )

    def run_group(group_index: int) -> list[tuple[int, int]]:
        uuids, used = groups[group_index]
        results = []
        for shard_index in schedules[group_index]:
            return_code = _run_shard_on_gpu_pair(
                execution_manifest=execution_manifest,
                manifest=manifest,
                manifest_sha256=actual_sha,
                shard_index=shard_index,
                num_shards=num_shards,
                state_dir=state_dir,
                selected_uuids=uuids,
                used_mib=used,
                runner=runner,
            )
            results.append((shard_index, return_code))
            if return_code != 0:
                break
        return results

    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        group_results = list(pool.map(run_group, range(len(groups))))
    flat_results = [item for group in group_results for item in group]
    failures = [
        {"shard_index": shard_index, "return_code": return_code}
        for shard_index, return_code in flat_results
        if return_code != 0
    ]
    finished = datetime.now(timezone.utc).isoformat()
    if failures:
        _atomic_json(
            state_dir / "failed.json",
            {**node_base, "failures": failures, "failed_at": finished},
        )
        return failures[0]["return_code"]
    _atomic_json(
        state_dir / "completed.json",
        {
            **node_base,
            "completed_shard_indices": [index for index, _code in flat_results],
            "completed_at": finished,
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render")
    render.add_argument("--git-commit", required=True)
    render.add_argument("--execution-manifest", type=Path, default=DEFAULT_MANIFEST)
    render.add_argument("--context", default=DEFAULT_CONTEXT)
    render.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    render.add_argument("--image", default=DEFAULT_IMAGE)
    render.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    render.add_argument("--shared-host-path", default=DEFAULT_SHARED_HOST_PATH)
    render.add_argument("--state-dir", type=PurePosixPath, default=DEFAULT_STATE_DIR)
    render.add_argument("--dependency-timeout-seconds", type=int, default=259_200)
    render.add_argument("--max-startup-used-mib", type=int, default=4096)
    render.add_argument("--uid", type=int, default=os.getuid())
    render.add_argument("--gid", type=int, default=os.getgid())
    render.add_argument("--supplemental-gid", type=int, default=31000)
    render.add_argument("--format", choices=("yaml", "json"), default="yaml")
    render.add_argument("--output", type=Path)
    render.add_argument("--server-dry-run", action="store_true")

    launch = subparsers.add_parser("run-shard")
    launch.add_argument("--execution-manifest", type=Path, required=True)
    launch.add_argument("--expected-manifest-sha256", required=True)
    launch.add_argument("--shard-index", type=int, required=True)
    launch.add_argument("--num-shards", type=int, required=True)
    launch.add_argument("--state-dir", type=Path, required=True)
    launch.add_argument("--max-startup-used-mib", type=int, default=4096)

    node = subparsers.add_parser("run-node")
    node.add_argument("--execution-manifest", type=Path, required=True)
    node.add_argument("--expected-manifest-sha256", required=True)
    node.add_argument(
        "--shard-indices",
        required=True,
        help="Comma-separated logical shard indices assigned to this node",
    )
    node.add_argument("--num-shards", type=int, required=True)
    node.add_argument("--state-dir", type=Path, required=True)
    node.add_argument("--max-startup-used-mib", type=int, default=4096)
    node.add_argument("--max-parallel-shards", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run-shard":
            return run_shard(
                execution_manifest=args.execution_manifest,
                expected_manifest_sha256=args.expected_manifest_sha256,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                state_dir=args.state_dir,
                max_startup_used_mib=args.max_startup_used_mib,
            )
        if args.command == "run-node":
            return run_node(
                execution_manifest=args.execution_manifest,
                expected_manifest_sha256=args.expected_manifest_sha256,
                shard_indices=[
                    int(value) for value in args.shard_indices.split(",")
                    if value.strip()
                ],
                num_shards=args.num_shards,
                state_dir=args.state_dir,
                max_startup_used_mib=args.max_startup_used_mib,
                max_parallel_shards=args.max_parallel_shards,
            )
        options = RenderOptions(
            git_commit=args.git_commit,
            execution_manifest=args.execution_manifest,
            context=args.context,
            namespace=args.namespace,
            image=args.image,
            name_prefix=args.name_prefix,
            shared_host_path=args.shared_host_path,
            state_dir=args.state_dir,
            dependency_timeout_seconds=args.dependency_timeout_seconds,
            max_startup_used_mib=args.max_startup_used_mib,
            uid=args.uid,
            gid=args.gid,
            supplemental_gid=args.supplemental_gid,
        )
        jobs, summary = build_jobs(options)
        rendered = serialize_jobs(jobs, args.format)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        if args.server_dry_run:
            server_dry_run(jobs, options)
        print(
            json.dumps(
                {
                    "mode": "server-dry-run" if args.server_dry_run else "render-only",
                    "job_count": len(jobs),
                    "nodes": [worker.node for worker in NODE_WORKERS],
                    "manifest": summary,
                    "output": str(args.output) if args.output else None,
                }
            ),
            file=sys.stderr,
        )
        return 0
    except (OSError, ValueError, Phase15JobError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
