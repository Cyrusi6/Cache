#!/usr/bin/env python3
"""Deterministically render the model-output-free FPCT-E1 runtime probe Job.

This renderer is intentionally Kubernetes-free.  It consumes the frozen
render-only template, replaces exactly five placeholder identities, validates
the resulting Job contract, and atomically writes a rendered YAML plus a
self-hashed provenance manifest.  It never creates or applies Kubernetes
resources.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
PROTOCOL_ID = "fpct_e1_runtime_probe_render_v1"
STATUS = "RENDERED_NOT_SUBMITTED"
NAMESPACE = "c2c-research"
NODE_NAME = "4090-48gx2"
OUTPUT_FILENAME = "runtime_probe.json"
RENDERED_JOB_FILENAME = "runtime_probe_job.yaml"
MANIFEST_FILENAME = "runtime_probe_render_manifest.json"
TEMPLATE_RELATIVE = PurePosixPath("recipe/k8s/fpct_e1/runtime_probe_job.yaml")
PRODUCER_RELATIVE = PurePosixPath(
    "script/experiment/fpct_e1_runtime_probe_renderer.py"
)
SOURCE_RECEIPT_RELATIVE = PurePosixPath(".fpct_e1_source_snapshot_receipt.json")
RUNTIME_PROBE_RELATIVE = PurePosixPath(
    "script/experiment/fpct_e1_runtime_probe.py"
)
SOURCE_LOCK_RELATIVE = PurePosixPath(
    "script/experiment/fpct_e1_source_snapshot_lock.py"
)
EXECUTION_SHA = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
IMAGE_DIGEST = re.compile(
    r"^(?:[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?/)?"
    r"(?:[a-z0-9]+(?:[._-][a-z0-9]+)*/)*"
    r"[a-z0-9]+(?:[._-][a-z0-9]+)*@sha256:[0-9a-f]{64}$"
)
GENERIC_PLACEHOLDER = re.compile(r"__[A-Z0-9_]+__")
SAFE_ABSOLUTE_PATH = re.compile(r"^/[A-Za-z0-9._/-]+$")
PLACEHOLDER_COUNTS = {
    "__EXECUTION_SHA__": 4,
    "__IMMUTABLE_IMAGE_DIGEST__": 2,
    "__SOURCE_SNAPSHOT_HOST__": 1,
    "__SOURCE_SNAPSHOT_TREE_SHA__": 1,
    "__RUNTIME_PROBE_RUN_ROOT__": 3,
}
EXPECTED_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYTHONPATH": "/opt/fpct",
    "PYTHONDONTWRITEBYTECODE": "1",
    "HOME": "/tmp/home",
    "XDG_CACHE_HOME": "/tmp/xdg",
    "HF_HOME": "/tmp/hf",
    "HF_DATASETS_CACHE": "/tmp/hf-datasets",
    "TORCH_HOME": "/tmp/torch",
    "CUDA_CACHE_PATH": "/tmp/cuda",
    "WANDB_DIR": "/tmp/wandb",
    "TMPDIR": "/tmp",
}


def _snapshot_root_from_producer() -> Path:
    producer = Path(__file__).absolute()
    relative = Path(PRODUCER_RELATIVE.as_posix())
    if tuple(producer.parts[-len(relative.parts) :]) != relative.parts:
        raise RuntimeError(f"renderer producer is outside a source snapshot: {producer}")
    root = producer
    for _ in relative.parts:
        root = root.parent
    return root


def _activate_snapshot_import_root() -> Path:
    root = _snapshot_root_from_producer()
    root_text = str(root)
    if sys.path:
        sys.path[0] = root_text
    else:
        sys.path.append(root_text)
    return root


_INITIAL_SNAPSHOT_IMPORT_ROOT = _activate_snapshot_import_root()

import yaml


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_exact_runtime_probe_module():
    root = _activate_snapshot_import_root()
    runtime_path = (root / Path(RUNTIME_PROBE_RELATIVE.as_posix())).absolute()
    if not runtime_path.is_file() or runtime_path.is_symlink():
        raise RuntimeError(f"exact runtime-probe module is absent/symlinked: {runtime_path}")
    unique_name = (
        "_fpct_e1_exact_runtime_probe_"
        + hashlib.sha256(str(runtime_path).encode()).hexdigest()[:16]
    )
    specification = importlib.util.spec_from_file_location(unique_name, runtime_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load exact runtime-probe module: {runtime_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[unique_name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(unique_name, None)
        raise
    observed = Path(str(getattr(module, "__file__", ""))).absolute()
    expected_source_lock = (
        root / Path(SOURCE_LOCK_RELATIVE.as_posix())
    ).absolute()
    if observed != runtime_path or observed.is_symlink():
        raise RuntimeError("runtime-probe module provenance mismatch")
    provenance = getattr(module, "LOCAL_IMPORT_PROVENANCE", {})
    if provenance != {"source_snapshot_lock": str(expected_source_lock)}:
        raise RuntimeError("runtime-probe local import provenance mismatch")
    return module, {
        "runtime_probe": str(runtime_path),
        "source_snapshot_lock": str(expected_source_lock),
    }


def _validated_absolute_path(value: str, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not SAFE_ABSOLUTE_PATH.fullmatch(value):
        raise ValueError(f"{name} must be one conservative absolute POSIX path")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or path == PurePosixPath("/")
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or str(path) != value
    ):
        raise ValueError(f"{name} is not canonical")
    return path


def _is_within(path: PurePosixPath, parent: PurePosixPath) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _overlap(first: PurePosixPath, second: PurePosixPath) -> bool:
    return _is_within(first, second) or _is_within(second, first)


def _existing_directory_without_symlink(path: Path, name: str) -> Path:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{name} contains a symlink component: {current}")
        if not current.exists():
            raise FileNotFoundError(f"{name} directory is absent: {current}")
    if not path.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {path}")
    return path.resolve(strict=True)


def _existing_file_without_symlink(path: Path, name: str) -> Path:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{name} contains a symlink component: {current}")
        if not current.exists():
            raise FileNotFoundError(f"{name} file is absent: {current}")
    if not path.is_file():
        raise FileNotFoundError(f"{name} is not a regular file: {path}")
    return path.resolve(strict=True)


def _verify_source_template_and_receipt(
    *,
    template_path: Path,
    source_snapshot_host: str,
    execution_sha: str,
    source_snapshot_tree_sha: str,
) -> dict[str, Any]:
    source = Path(source_snapshot_host)
    expected_template = source / Path(TEMPLATE_RELATIVE.as_posix())
    supplied = _validated_absolute_path(str(template_path), "template_path")
    if Path(str(supplied)) != expected_template:
        raise ValueError(
            "template_path must exactly equal the runtime template inside source_snapshot_host"
        )
    template_real = _existing_file_without_symlink(
        expected_template, "runtime-probe template"
    )
    if template_real != expected_template:
        raise ValueError("runtime-probe template realpath escaped the source snapshot")
    expected_producer = source / Path(PRODUCER_RELATIVE.as_posix())
    producer_real = _existing_file_without_symlink(
        expected_producer, "runtime-probe renderer producer"
    )
    invoked_producer = _existing_file_without_symlink(
        Path(__file__).absolute(), "invoked runtime-probe renderer producer"
    )
    if producer_real != expected_producer or invoked_producer != producer_real:
        raise ValueError(
            "runtime-probe renderer must execute from the exact source snapshot producer"
        )
    receipt_path = source / Path(SOURCE_RECEIPT_RELATIVE.as_posix())
    receipt_real = _existing_file_without_symlink(
        receipt_path, "source snapshot receipt"
    )
    if receipt_real != receipt_path:
        raise ValueError("source snapshot receipt realpath escaped the source snapshot")
    runtime_probe_module, local_imports = _load_exact_runtime_probe_module()
    verification = runtime_probe_module.verify_mounted_source_snapshot(
        execution_sha=execution_sha,
        source_snapshot_tree_sha=source_snapshot_tree_sha,
        source_snapshot_root=source,
        source_snapshot_receipt=receipt_path,
    )
    if (
        verification.get("status") != "GO_MOUNTED_SOURCE_SNAPSHOT"
        or verification.get("execution_sha") != execution_sha
        or verification.get("mounted_tree_canonical_sha256")
        != source_snapshot_tree_sha
        or not SHA256.fullmatch(str(verification.get("receipt_sha256", "")))
    ):
        raise ValueError("source snapshot receipt verification differs from render inputs")
    return {
        "template_path": str(template_real),
        "producer_path": str(producer_real),
        "receipt_path": str(receipt_real),
        "verification": verification,
        "local_imports": local_imports,
    }


def validate_inputs(
    *,
    execution_sha: str,
    image_digest: str,
    source_snapshot_host: str,
    source_snapshot_tree_sha: str,
    runtime_probe_run_root: str,
    output_dir: Path,
) -> dict[str, str]:
    if not EXECUTION_SHA.fullmatch(execution_sha):
        raise ValueError("execution_sha must be one lowercase 40-character Git SHA")
    if not IMAGE_DIGEST.fullmatch(image_digest):
        raise ValueError("image_digest must be an immutable sha256 digest")
    if not SHA256.fullmatch(source_snapshot_tree_sha):
        raise ValueError("source_snapshot_tree_sha must be 64 lowercase hex characters")
    source = _validated_absolute_path(source_snapshot_host, "source_snapshot_host")
    runtime = _validated_absolute_path(runtime_probe_run_root, "runtime_probe_run_root")
    output = _validated_absolute_path(str(output_dir), "output_dir")
    if len(runtime.parts) < 3:
        raise ValueError("runtime_probe_run_root must be a dedicated child directory")
    if runtime.name != execution_sha:
        raise ValueError("runtime_probe_run_root must end with the full execution SHA")
    if _overlap(runtime, source):
        raise ValueError("runtime output root and read-only source snapshot must be disjoint")
    if _overlap(output, source) or _overlap(output, runtime):
        raise ValueError("renderer output_dir must be disjoint from source and runtime roots")

    source_path = Path(str(source))
    runtime_path = Path(str(runtime))
    output_path = Path(str(output))
    source_real = _existing_directory_without_symlink(
        source_path, "source_snapshot_host"
    )
    runtime_parent_real = _existing_directory_without_symlink(
        runtime_path.parent, "runtime_probe_run_root parent"
    )
    output_parent_real = _existing_directory_without_symlink(
        output_path.parent, "output_dir parent"
    )
    if runtime_path.is_symlink() or runtime_path.exists():
        raise FileExistsError(f"runtime_probe_run_root already exists: {runtime}")
    if output_path.is_symlink() or output_path.exists():
        raise FileExistsError(f"output_dir already exists: {output}")
    runtime_real = runtime_parent_real / runtime_path.name
    output_real = output_parent_real / output_path.name
    source_real_posix = PurePosixPath(str(source_real))
    runtime_real_posix = PurePosixPath(str(runtime_real))
    output_real_posix = PurePosixPath(str(output_real))
    if _overlap(runtime_real_posix, source_real_posix):
        raise ValueError("runtime and source realpaths are not physically disjoint")
    if _overlap(output_real_posix, source_real_posix) or _overlap(
        output_real_posix, runtime_real_posix
    ):
        raise ValueError("output_dir realpath is not physically disjoint")
    return {
        "__EXECUTION_SHA__": execution_sha,
        "__IMMUTABLE_IMAGE_DIGEST__": image_digest,
        "__SOURCE_SNAPSHOT_HOST__": str(source),
        "__SOURCE_SNAPSHOT_TREE_SHA__": source_snapshot_tree_sha,
        "__RUNTIME_PROBE_RUN_ROOT__": str(runtime),
    }


def _validate_template(text: str) -> None:
    observed = set(GENERIC_PLACEHOLDER.findall(text))
    if observed != set(PLACEHOLDER_COUNTS):
        raise ValueError(
            "runtime-probe template placeholder universe changed: "
            f"expected={sorted(PLACEHOLDER_COUNTS)}, observed={sorted(observed)}"
        )
    counts = {placeholder: text.count(placeholder) for placeholder in observed}
    if counts != PLACEHOLDER_COUNTS:
        raise ValueError(
            "runtime-probe template placeholder occurrence counts changed: "
            f"expected={PLACEHOLDER_COUNTS}, observed={counts}"
        )


def _argument_map(arguments: Sequence[Any]) -> dict[str, str]:
    if len(arguments) % 2:
        raise ValueError("runtime-probe Job arguments are not flag/value pairs")
    result: dict[str, str] = {}
    for index in range(0, len(arguments), 2):
        flag, value = arguments[index : index + 2]
        if not isinstance(flag, str) or not flag.startswith("--") or flag in result:
            raise ValueError("runtime-probe Job arguments contain an invalid/duplicate flag")
        if not isinstance(value, str):
            raise ValueError("runtime-probe Job argument value is not text")
        result[flag] = value
    return result


def _validate_rendered_job(
    raw: bytes, replacements: Mapping[str, str]
) -> dict[str, Any]:
    if GENERIC_PLACEHOLDER.search(raw.decode("utf-8")):
        raise ValueError("rendered runtime-probe Job still has an unresolved placeholder")
    job = yaml.safe_load(raw)
    if not isinstance(job, Mapping):
        raise ValueError("rendered runtime-probe YAML is not one object")
    execution_sha = replacements["__EXECUTION_SHA__"]
    image_digest = replacements["__IMMUTABLE_IMAGE_DIGEST__"]
    source_host = replacements["__SOURCE_SNAPSHOT_HOST__"]
    source_tree = replacements["__SOURCE_SNAPSHOT_TREE_SHA__"]
    run_root = replacements["__RUNTIME_PROBE_RUN_ROOT__"]
    expected_job_name = f"fpct-e1-runtime-probe-{execution_sha}"
    if (
        job.get("apiVersion") != "batch/v1"
        or job.get("kind") != "Job"
        or job.get("metadata", {}).get("name") != expected_job_name
        or job.get("metadata", {}).get("namespace") != NAMESPACE
        or job.get("spec", {}).get("backoffLimit") != 0
    ):
        raise ValueError("rendered runtime-probe Job identity changed")
    pod = job.get("spec", {}).get("template", {}).get("spec", {})
    containers = pod.get("containers")
    if (
        pod.get("nodeName") != NODE_NAME
        or pod.get("restartPolicy") != "Never"
        or not isinstance(containers, list)
        or len(containers) != 1
    ):
        raise ValueError("rendered runtime-probe Pod contract changed")
    container = containers[0]
    if (
        container.get("name") != "runtime-probe"
        or container.get("image") != image_digest
        or container.get("command")
        != [
            "/opt/conda/bin/python3.11",
            "/opt/fpct/script/experiment/fpct_e1_runtime_probe.py",
        ]
        or container.get("securityContext", {}).get("readOnlyRootFilesystem") is not True
        or container.get("securityContext", {}).get("allowPrivilegeEscalation") is not False
    ):
        raise ValueError("rendered runtime-probe container contract changed")
    arguments = _argument_map(container.get("args", []))
    if arguments != {
        "--output": f"{run_root}/{OUTPUT_FILENAME}",
        "--execution-sha": execution_sha,
        "--image-digest": image_digest,
        "--source-snapshot-tree-sha": source_tree,
        "--source-snapshot-root": "/opt/fpct",
        "--source-snapshot-receipt": "/opt/fpct/.fpct_e1_source_snapshot_receipt.json",
    }:
        raise ValueError("rendered runtime-probe arguments changed")
    environment = {
        record.get("name"): record.get("value")
        for record in container.get("env", [])
        if isinstance(record, Mapping)
    }
    if environment != EXPECTED_ENVIRONMENT:
        raise ValueError("runtime-probe offline/rootfs environment changed")
    resources = container.get("resources", {})
    if (
        resources.get("limits", {}).get("nvidia.com/gpu") != "1"
        or resources.get("requests", {}).get("nvidia.com/gpu") != "1"
    ):
        raise ValueError("runtime-probe GPU request changed")
    mounts = {
        record.get("name"): record
        for record in container.get("volumeMounts", [])
        if isinstance(record, Mapping)
    }
    if mounts.get("source-snapshot") != {
        "name": "source-snapshot",
        "mountPath": "/opt/fpct",
        "readOnly": True,
    }:
        raise ValueError("runtime-probe source mount is not frozen read-only")
    if mounts.get("runtime-output") != {
        "name": "runtime-output",
        "mountPath": run_root,
    }:
        raise ValueError("runtime-probe output mount is not the unique run root")
    if mounts.get("tmp") != {"name": "tmp", "mountPath": "/tmp"}:
        raise ValueError("runtime-probe writable tmp mount changed")
    volumes = {
        record.get("name"): record
        for record in pod.get("volumes", [])
        if isinstance(record, Mapping)
    }
    if volumes.get("source-snapshot") != {
        "name": "source-snapshot",
        "hostPath": {"path": source_host, "type": "Directory"},
    }:
        raise ValueError("runtime-probe source hostPath changed")
    if volumes.get("runtime-output") != {
        "name": "runtime-output",
        "hostPath": {"path": run_root, "type": "DirectoryOrCreate"},
    }:
        raise ValueError("runtime-probe host/container output roots are not identical")
    if volumes.get("tmp") != {"name": "tmp", "emptyDir": {}}:
        raise ValueError("runtime-probe tmp volume changed")
    return {
        "name": expected_job_name,
        "namespace": NAMESPACE,
        "node_name": NODE_NAME,
        "gpu_count": 1,
        "runtime_probe_output": f"{run_root}/{OUTPUT_FILENAME}",
    }


def render_runtime_probe_job(
    *,
    template_path: Path,
    output_dir: Path,
    execution_sha: str,
    image_digest: str,
    source_snapshot_host: str,
    source_snapshot_tree_sha: str,
    runtime_probe_run_root: str,
) -> dict[str, Any]:
    replacements = validate_inputs(
        execution_sha=execution_sha,
        image_digest=image_digest,
        source_snapshot_host=source_snapshot_host,
        source_snapshot_tree_sha=source_snapshot_tree_sha,
        runtime_probe_run_root=runtime_probe_run_root,
        output_dir=output_dir,
    )
    source_contract = _verify_source_template_and_receipt(
        template_path=template_path,
        source_snapshot_host=replacements["__SOURCE_SNAPSHOT_HOST__"],
        execution_sha=execution_sha,
        source_snapshot_tree_sha=source_snapshot_tree_sha,
    )
    template_raw = template_path.read_bytes()
    try:
        template_text = template_raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("runtime-probe template is not UTF-8") from error
    _validate_template(template_text)
    rendered_text = template_text
    for placeholder in sorted(replacements):
        rendered_text = rendered_text.replace(placeholder, replacements[placeholder])
    rendered_raw = rendered_text.encode("utf-8")
    job_identity = _validate_rendered_job(rendered_raw, replacements)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": STATUS,
        "execution_sha": execution_sha,
        "image_digest": image_digest,
        "source_snapshot": {
            "host_path": source_snapshot_host,
            "canonical_tree_sha256": source_snapshot_tree_sha,
            "read_only": True,
            "receipt_path": source_contract["receipt_path"],
            "verification": source_contract["verification"],
        },
        "runtime_directory": {
            "host_root": runtime_probe_run_root,
            "container_root": runtime_probe_run_root,
            "same_absolute_path": True,
            "execution_sha_terminal_component": True,
            "output_path": job_identity["runtime_probe_output"],
            "writable_scope": "this execution's runtime-probe directory only",
        },
        "source_template": {
            "path": source_contract["template_path"],
            "bytes": len(template_raw),
            "sha256": sha256_bytes(template_raw),
            "placeholder_counts": dict(sorted(PLACEHOLDER_COUNTS.items())),
        },
        "replacements": dict(sorted(replacements.items())),
        "local_import_provenance": source_contract["local_imports"],
        "producer": {
            "path": source_contract["producer_path"],
            "logical_path": f"source://{PRODUCER_RELATIVE.as_posix()}",
            "bytes": Path(source_contract["producer_path"]).stat().st_size,
            "sha256": sha256_file(Path(source_contract["producer_path"])),
            "exact_snapshot_producer": True,
        },
        "rendered_job": {
            "path": RENDERED_JOB_FILENAME,
            "bytes": len(rendered_raw),
            "sha256": sha256_bytes(rendered_raw),
            **job_identity,
        },
        "firewall": {
            "kubectl_invoked": False,
            "network_accessed": False,
            "model_loaded": False,
            "tokenizer_loaded": False,
            "checkpoint_loaded": False,
            "model_output_accessed": False,
        },
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        (temporary / RENDERED_JOB_FILENAME).write_bytes(rendered_raw)
        (temporary / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))
        if output_dir.exists():
            raise FileExistsError(output_dir)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--source-snapshot-host", required=True)
    parser.add_argument("--source-snapshot-tree-sha", required=True)
    parser.add_argument("--runtime-probe-run-root", required=True)
    args = parser.parse_args(argv)
    result = render_runtime_probe_job(
        template_path=args.template,
        output_dir=args.output_dir,
        execution_sha=args.execution_sha,
        image_digest=args.image_digest,
        source_snapshot_host=args.source_snapshot_host,
        source_snapshot_tree_sha=args.source_snapshot_tree_sha,
        runtime_probe_run_root=args.runtime_probe_run_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
