#!/usr/bin/env python3
"""Submit and manage C2C GPU jobs on the local Kubernetes node."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, cast

DEFAULT_CONTEXT = "default"
DEFAULT_NAMESPACE = "c2c-research"
DEFAULT_NODE = "4090-24gx4"
DEFAULT_IMAGE = (
    "swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/"
    "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"
)
MANAGED_BY_LABEL = "app.kubernetes.io/managed-by"
MANAGED_BY_VALUE = "c2c-gpu-job"
MANAGED_BY_ANNOTATION = "c2c.research/managed-by"
APP_LABEL = "app.kubernetes.io/name"
APP_VALUE = "c2c"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_K8S_ROOT = PROJECT_ROOT / "local" / "k8s"
RUNTIME_ROOT = LOCAL_K8S_ROOT / "runtime"
MANIFEST_ROOT = LOCAL_K8S_ROOT / "manifests"
HF_CACHE = Path.home() / ".cache" / "huggingface"
PIP_CACHE = Path.home() / ".cache" / "pip"


class GpuJobError(RuntimeError):
    """User-facing GPU job error."""


def sanitize_name(value: str, max_length: int = 40) -> str:
    """Return a DNS-1123-compatible name fragment."""
    normalized = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    normalized = re.sub(r"-+", "-", normalized)
    if not normalized:
        raise ValueError("任务名必须至少包含一个字母或数字")
    return normalized[:max_length].rstrip("-")


def build_job_name(base_name: str, now: datetime | None = None) -> str:
    """Build a unique Kubernetes Job name."""
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S-%f")
    return f"{sanitize_name(base_name)}-{timestamp}"


def normalize_command(command: Sequence[str]) -> list[str]:
    """Normalize argparse.REMAINDER output."""
    normalized = list(command)
    if normalized and normalized[0] == "--":
        normalized = normalized[1:]
    if not normalized:
        raise ValueError("`submit` 必须在 `--` 后提供要运行的命令")
    return normalized


def resource_defaults(
    gpus: int, cpu: str | None = None, memory: str | None = None
) -> tuple[str, str, str]:
    """Resolve CPU, memory and shared-memory defaults."""
    if gpus not in {1, 2, 3, 4}:
        raise ValueError("--gpus 只能是 1、2、3 或 4")
    cpu_value = cpu or str(gpus * 8)
    memory_value = memory or f"{gpus * 32}Gi"
    shm_value = f"{max(16, gpus * 8)}Gi"
    return cpu_value, memory_value, shm_value


def ensure_local_paths() -> None:
    """Create ignored host directories before Kubernetes mounts them."""
    for path in (
        RUNTIME_ROOT,
        RUNTIME_ROOT / "home",
        MANIFEST_ROOT,
        HF_CACHE,
        PIP_CACHE,
    ):
        path.mkdir(parents=True, exist_ok=True)


def kubectl_prefix(context: str, namespace: str | None = None) -> list[str]:
    command = ["kubectl", "--context", context]
    if namespace:
        command.extend(["-n", namespace])
    return command


def run_process(
    command: Sequence[str],
    *,
    input_text: str | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and turn failures into concise user errors."""
    result = subprocess.run(
        list(command),
        input=input_text,
        text=True,
        capture_output=capture,
        check=False,
    )
    if result.returncode != 0:
        detail = ""
        if capture:
            detail = (result.stderr or result.stdout or "").strip()
        suffix = f"\n{detail}" if detail else ""
        raise GpuJobError(f"命令失败：{' '.join(command)}{suffix}")
    return result


def kubectl_json(
    context: str, namespace: str | None, arguments: Sequence[str]
) -> dict[str, Any]:
    result = run_process(
        kubectl_prefix(context, namespace) + list(arguments) + ["-o", "json"],
        capture=True,
    )
    data = json.loads(result.stdout)
    if not isinstance(data, dict):
        raise GpuJobError("kubectl 返回的 JSON 不是对象")
    return cast(dict[str, Any], data)


def namespace_exists(context: str, namespace: str) -> bool:
    result = subprocess.run(
        kubectl_prefix(context) + ["get", "namespace", namespace],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return True
    detail = f"{result.stderr}\n{result.stdout}".lower()
    if "notfound" in detail or "not found" in detail:
        return False
    raise GpuJobError(
        f"无法检查 namespace {namespace}：{(result.stderr or result.stdout).strip()}"
    )


def can_i(context: str, namespace: str | None, action: str, resource: str) -> bool:
    result = run_process(
        kubectl_prefix(context, namespace) + ["auth", "can-i", action, resource],
        capture=True,
    )
    return result.stdout.strip() == "yes"


def node_capacity(context: str, node: str) -> int:
    data = kubectl_json(context, None, ["get", "node", node])
    conditions = data.get("status", {}).get("conditions", [])
    ready = any(
        item.get("type") == "Ready" and item.get("status") == "True"
        for item in conditions
    )
    if not ready:
        raise GpuJobError(f"节点 {node} 当前不是 Ready")
    raw_gpu = data.get("status", {}).get("allocatable", {}).get("nvidia.com/gpu", 0)
    try:
        return int(raw_gpu)
    except (TypeError, ValueError) as exc:
        raise GpuJobError(f"节点 {node} 的 GPU 容量无效：{raw_gpu}") from exc


def require_local_node(node: str) -> None:
    if node != DEFAULT_NODE:
        raise GpuJobError(
            "当前实现使用本机 hostPath，仅支持节点 "
            f"{DEFAULT_NODE}；不能安全调度到 {node}"
        )


def build_job_manifest(
    *,
    job_name: str,
    base_name: str,
    namespace: str,
    node: str,
    image: str,
    gpus: int,
    command: Sequence[str],
    cpu: str | None = None,
    memory: str | None = None,
    timeout_hours: int = 72,
    bootstrap: bool = True,
    uid: int | None = None,
    gid: int | None = None,
) -> dict[str, Any]:
    """Build a Kubernetes Job manifest without executing shell text."""
    cpu_value, memory_value, shm_value = resource_defaults(gpus, cpu, memory)
    normalized_command = normalize_command(command)
    if timeout_hours <= 0:
        raise ValueError("--timeout-hours 必须大于 0")

    run_uid = os.getuid() if uid is None else uid
    run_gid = os.getgid() if gid is None else gid
    if bootstrap:
        container_command = [
            "python",
            "/workspace/Cache/script/k8s/container_entrypoint.py",
        ]
        container_args = [
            "--runtime-dir",
            "/runtime",
            "--project-root",
            "/workspace/Cache",
            "--",
            *normalized_command,
        ]
    else:
        container_command = [normalized_command[0]]
        container_args = normalized_command[1:]

    labels = {
        APP_LABEL: APP_VALUE,
        MANAGED_BY_LABEL: MANAGED_BY_VALUE,
        "c2c.research/experiment": sanitize_name(base_name, max_length=63),
    }
    gpu_quantity = str(gpus)
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": labels,
            "annotations": {MANAGED_BY_ANNOTATION: MANAGED_BY_VALUE},
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": timeout_hours * 3600,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "terminationGracePeriodSeconds": 30,
                    "nodeSelector": {"kubernetes.io/hostname": node},
                    "securityContext": {
                        "runAsUser": run_uid,
                        "runAsGroup": run_gid,
                        "fsGroup": run_gid,
                    },
                    "containers": [
                        {
                            "name": "runner",
                            "image": image,
                            "imagePullPolicy": "IfNotPresent",
                            "workingDir": "/workspace/Cache",
                            "command": container_command,
                            "args": container_args,
                            "env": [
                                {"name": "PYTHONUNBUFFERED", "value": "1"},
                                {"name": "HOME", "value": "/runtime/home"},
                                {
                                    "name": "XDG_CACHE_HOME",
                                    "value": "/runtime/home/.cache",
                                },
                                {
                                    "name": "TORCH_HOME",
                                    "value": "/runtime/home/.cache/torch",
                                },
                                {
                                    "name": "TOKENIZERS_PARALLELISM",
                                    "value": "false",
                                },
                                {
                                    "name": "HF_HOME",
                                    "value": "/cache/huggingface",
                                },
                                {
                                    "name": "HF_HUB_CACHE",
                                    "value": "/cache/huggingface/hub",
                                },
                                {
                                    "name": "HF_ENDPOINT",
                                    "value": "https://hf-mirror.com",
                                },
                                {
                                    "name": "HF_HUB_DOWNLOAD_TIMEOUT",
                                    "value": "600",
                                },
                                {"name": "PIP_CACHE_DIR", "value": "/cache/pip"},
                                {"name": "C2C_RUNTIME_IMAGE", "value": image},
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": cpu_value,
                                    "memory": memory_value,
                                    "nvidia.com/gpu": gpu_quantity,
                                },
                                "limits": {"nvidia.com/gpu": gpu_quantity},
                            },
                            "volumeMounts": [
                                {
                                    "name": "workspace",
                                    "mountPath": "/workspace/Cache",
                                },
                                {"name": "runtime", "mountPath": "/runtime"},
                                {
                                    "name": "huggingface-cache",
                                    "mountPath": "/cache/huggingface",
                                },
                                {
                                    "name": "pip-cache",
                                    "mountPath": "/cache/pip",
                                },
                                {"name": "shm", "mountPath": "/dev/shm"},
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "workspace",
                            "hostPath": {
                                "path": str(PROJECT_ROOT),
                                "type": "Directory",
                            },
                        },
                        {
                            "name": "runtime",
                            "hostPath": {
                                "path": str(RUNTIME_ROOT),
                                "type": "Directory",
                            },
                        },
                        {
                            "name": "huggingface-cache",
                            "hostPath": {
                                "path": str(HF_CACHE),
                                "type": "Directory",
                            },
                        },
                        {
                            "name": "pip-cache",
                            "hostPath": {
                                "path": str(PIP_CACHE),
                                "type": "Directory",
                            },
                        },
                        {
                            "name": "shm",
                            "emptyDir": {"medium": "Memory", "sizeLimit": shm_value},
                        },
                    ],
                },
            },
        },
    }


def is_managed_job(job: dict[str, Any]) -> bool:
    metadata = job.get("metadata")
    if not isinstance(metadata, dict):
        return False
    labels = metadata.get("labels")
    if not isinstance(labels, dict):
        return False
    annotations = metadata.get("annotations")
    if not isinstance(annotations, dict):
        return False
    return (
        labels.get(MANAGED_BY_LABEL) == MANAGED_BY_VALUE
        and annotations.get(MANAGED_BY_ANNOTATION) == MANAGED_BY_VALUE
    )


def require_managed_job(job: dict[str, Any], name: str) -> None:
    if not is_managed_job(job):
        raise GpuJobError(f"拒绝删除非 {MANAGED_BY_VALUE} 管理的 Job：{name}")
    manifest_path = MANIFEST_ROOT / f"{name}.json"
    if not manifest_path.is_file():
        raise GpuJobError(f"拒绝删除缺少本地 manifest 的 Job：{name}")


def save_manifest(manifest: dict[str, Any]) -> Path:
    ensure_local_paths()
    path = MANIFEST_ROOT / f"{manifest['metadata']['name']}.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def apply_manifest(
    manifest: dict[str, Any], context: str, namespace: str, dry_run: bool
) -> None:
    command = kubectl_prefix(context, namespace) + ["create"]
    if dry_run:
        command.append("--dry-run=server")
    command.extend(["-f", "-"])
    run_process(command, input_text=json.dumps(manifest))


def init_command(args: argparse.Namespace) -> None:
    ensure_local_paths()
    require_local_node(args.node)
    capacity = node_capacity(args.context, args.node)
    if capacity < 1:
        raise GpuJobError(f"节点 {args.node} 没有可调度 NVIDIA GPU")

    if not namespace_exists(args.context, args.namespace):
        if not can_i(args.context, None, "create", "namespaces"):
            raise GpuJobError("当前身份没有创建 namespace 的权限")
        namespace_manifest = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": args.namespace,
                "labels": {APP_LABEL: APP_VALUE, MANAGED_BY_LABEL: MANAGED_BY_VALUE},
            },
        }
        run_process(
            kubectl_prefix(args.context) + ["apply", "-f", "-"],
            input_text=json.dumps(namespace_manifest),
        )

    if not can_i(args.context, args.namespace, "create", "jobs.batch"):
        raise GpuJobError(f"当前身份不能在 {args.namespace} 创建 Job")

    print(f"namespace={args.namespace}")
    print(f"node={args.node} ready=true gpu={capacity}")
    print(f"runtime={RUNTIME_ROOT}")
    print(f"manifests={MANIFEST_ROOT}")


def submit_command(args: argparse.Namespace) -> None:
    ensure_local_paths()
    require_local_node(args.node)
    if not namespace_exists(args.context, args.namespace):
        raise GpuJobError(
            f"namespace {args.namespace} 不存在，请先运行 `gpu_job.sh init`"
        )
    capacity = node_capacity(args.context, args.node)
    if args.gpus > capacity:
        raise GpuJobError(
            f"请求 {args.gpus} 张 GPU，但节点 {args.node} 仅提供 {capacity} 张"
        )
    if not can_i(args.context, args.namespace, "create", "jobs.batch"):
        raise GpuJobError(f"当前身份不能在 {args.namespace} 创建 Job")

    command = normalize_command(args.command)
    job_name = build_job_name(args.name)
    manifest = build_job_manifest(
        job_name=job_name,
        base_name=args.name,
        namespace=args.namespace,
        node=args.node,
        image=args.image,
        gpus=args.gpus,
        command=command,
        cpu=args.cpu,
        memory=args.memory,
        timeout_hours=args.timeout_hours,
        bootstrap=not args.no_bootstrap,
    )
    path = save_manifest(manifest)
    apply_manifest(manifest, args.context, args.namespace, args.dry_run)
    print(f"job={job_name}", flush=True)
    print(f"manifest={path}", flush=True)
    if args.dry_run:
        print("server_dry_run=true", flush=True)
        return
    print(
        "logs="
        f"bash bash/k8s/gpu_job.sh logs {job_name} --follow "
        f"--namespace {args.namespace}",
        flush=True,
    )
    if args.follow:
        logs_command(
            argparse.Namespace(
                context=args.context,
                namespace=args.namespace,
                job=job_name,
                follow=True,
            )
        )
        wait_for_job(args.context, args.namespace, job_name, f"{args.timeout_hours}h")


def list_command(args: argparse.Namespace) -> None:
    run_process(
        kubectl_prefix(args.context, args.namespace)
        + [
            "get",
            "jobs,pods",
            "-l",
            f"{MANAGED_BY_LABEL}={MANAGED_BY_VALUE}",
            "-o",
            "wide",
        ]
    )


def wait_for_job_pod(
    context: str, namespace: str, job_name: str, timeout_seconds: int = 3600
) -> str:
    deadline = time.monotonic() + timeout_seconds
    while True:
        data = kubectl_json(
            context,
            namespace,
            ["get", "pods", "-l", f"job-name={job_name}"],
        )
        items = data.get("items")
        if isinstance(items, list) and items:
            pod = items[0]
            if isinstance(pod, dict):
                metadata = pod.get("metadata")
                status = pod.get("status")
                if isinstance(metadata, dict) and isinstance(status, dict):
                    name = metadata.get("name")
                    phase = status.get("phase")
                    if isinstance(name, str) and phase in {
                        "Running",
                        "Succeeded",
                        "Failed",
                    }:
                        return name
        job = kubectl_json(context, namespace, ["get", "job", job_name])
        status = job.get("status")
        if isinstance(status, dict) and int(status.get("failed", 0) or 0) >= 1:
            raise GpuJobError(f"Job {job_name} 在产生可读日志前失败，请运行 describe")
        if time.monotonic() >= deadline:
            raise GpuJobError(f"等待 Job {job_name} 的 Pod 启动超时")
        time.sleep(2)


def logs_command(args: argparse.Namespace) -> None:
    pod_name = wait_for_job_pod(args.context, args.namespace, args.job)
    command = kubectl_prefix(args.context, args.namespace) + ["logs"]
    if args.follow:
        command.append("-f")
    command.append(f"pod/{pod_name}")
    run_process(command)


def describe_command(args: argparse.Namespace) -> None:
    run_process(
        kubectl_prefix(args.context, args.namespace) + ["describe", f"job/{args.job}"]
    )


def wait_command(args: argparse.Namespace) -> None:
    wait_for_job(args.context, args.namespace, args.job, args.timeout)


def parse_timeout(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)([smh])", value)
    if not match:
        raise ValueError("timeout 格式必须类似 30s、20m 或 72h")
    amount = int(match.group(1))
    multiplier = {"s": 1, "m": 60, "h": 3600}[match.group(2)]
    return amount * multiplier


def wait_for_job(context: str, namespace: str, job_name: str, timeout: str) -> None:
    deadline = time.monotonic() + parse_timeout(timeout)
    while True:
        job = kubectl_json(context, namespace, ["get", "job", job_name])
        status = job.get("status")
        if isinstance(status, dict):
            if int(status.get("succeeded", 0) or 0) >= 1:
                print(f"job={job_name} status=Complete")
                return
            if int(status.get("failed", 0) or 0) >= 1:
                raise GpuJobError(f"Job {job_name} 已失败，请运行 describe 和 logs")
            conditions = status.get("conditions")
            if isinstance(conditions, list):
                for condition in conditions:
                    if not isinstance(condition, dict):
                        continue
                    if (
                        condition.get("type") == "Failed"
                        and condition.get("status") == "True"
                    ):
                        reason = condition.get("reason", "unknown")
                        raise GpuJobError(f"Job {job_name} 已失败：{reason}")
        if time.monotonic() >= deadline:
            raise GpuJobError(f"等待 Job {job_name} 超时：{timeout}")
        time.sleep(5)


def delete_command(args: argparse.Namespace) -> None:
    job = kubectl_json(args.context, args.namespace, ["get", "job", args.job])
    require_managed_job(job, args.job)
    run_process(
        kubectl_prefix(args.context, args.namespace) + ["delete", "job", args.job]
    )


def add_target_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--context", default=DEFAULT_CONTEXT)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在 Kubernetes 上提交和管理 C2C GPU Job"
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    init_parser = subparsers.add_parser("init", help="初始化 namespace 和本地目录")
    add_target_arguments(init_parser)
    init_parser.add_argument("--node", default=DEFAULT_NODE)
    init_parser.set_defaults(handler=init_command)

    submit_parser = subparsers.add_parser("submit", help="提交 GPU Job")
    add_target_arguments(submit_parser)
    submit_parser.add_argument("--node", default=DEFAULT_NODE)
    submit_parser.add_argument("--name", required=True)
    submit_parser.add_argument("--gpus", required=True, type=int)
    submit_parser.add_argument("--cpu")
    submit_parser.add_argument("--memory")
    submit_parser.add_argument("--image", default=DEFAULT_IMAGE)
    submit_parser.add_argument("--timeout-hours", type=int, default=72)
    submit_parser.add_argument("--dry-run", action="store_true")
    submit_parser.add_argument("--no-bootstrap", action="store_true")
    submit_parser.add_argument("--follow", action="store_true")
    submit_parser.add_argument("command", nargs=argparse.REMAINDER)
    submit_parser.set_defaults(handler=submit_command)

    list_parser = subparsers.add_parser("list", help="列出本工具创建的任务")
    add_target_arguments(list_parser)
    list_parser.set_defaults(handler=list_command)

    logs_parser = subparsers.add_parser("logs", help="查看 Job 日志")
    add_target_arguments(logs_parser)
    logs_parser.add_argument("job")
    logs_parser.add_argument("--follow", action="store_true")
    logs_parser.set_defaults(handler=logs_command)

    describe_parser = subparsers.add_parser("describe", help="诊断 Job")
    add_target_arguments(describe_parser)
    describe_parser.add_argument("job")
    describe_parser.set_defaults(handler=describe_command)

    wait_parser = subparsers.add_parser("wait", help="等待 Job 完成")
    add_target_arguments(wait_parser)
    wait_parser.add_argument("job")
    wait_parser.add_argument("--timeout", default="72h")
    wait_parser.set_defaults(handler=wait_command)

    delete_parser = subparsers.add_parser("delete", help="删除本工具创建的 Job")
    add_target_arguments(delete_parser)
    delete_parser.add_argument("job")
    delete_parser.set_defaults(handler=delete_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if shutil.which("kubectl") is None:
        print("错误：未找到 kubectl", file=sys.stderr)
        return 1
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (GpuJobError, ValueError, json.JSONDecodeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
