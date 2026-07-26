#!/usr/bin/env python3
"""Write the model-output-free FPCT-E1 immutable-image runtime probe.

This probe is intended to run after Commit A inside the same immutable image
and on the same Kubernetes node class as the E1 capture shards.  It inspects
only interpreter/package/platform/CUDA metadata.  It neither imports model or
tokenizer factories nor reads a model, tokenizer, projector, or checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
PROTOCOL_ID = "fpct_e1_runtime_probe_v1"
STATUS = "FROZEN_MODEL_OUTPUT_FREE_RUNTIME_PROBE"
EXECUTION_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IMAGE_DIGEST_PATTERN = re.compile(r"^(?:[^@\s]+@)?sha256:[0-9a-f]{64}$")
PACKAGE_DISTRIBUTIONS = ("torch", "transformers", "tokenizers", "pyarrow")
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required runtime distribution is absent: {distribution}") from exc


def _device_record(properties: Any, index: int) -> dict[str, Any]:
    uuid = getattr(properties, "uuid", None)
    return {
        "index": index,
        "name": str(properties.name),
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability_major": int(properties.major),
        "compute_capability_minor": int(properties.minor),
        "multi_processor_count": int(properties.multi_processor_count),
        "uuid": None if uuid is None else str(uuid),
    }


def build_runtime_probe(
    *,
    execution_sha: str,
    image_digest: str,
    source_snapshot_tree_sha: str,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Collect runtime-only metadata under an enforced offline environment."""

    if not EXECUTION_SHA_PATTERN.fullmatch(execution_sha):
        raise ValueError("execution_sha must be exactly 40 lowercase hex characters")
    if not IMAGE_DIGEST_PATTERN.fullmatch(image_digest):
        raise ValueError("image_digest must be an immutable sha256 image digest")
    if not SHA256_PATTERN.fullmatch(source_snapshot_tree_sha):
        raise ValueError("source_snapshot_tree_sha must be exactly 64 lowercase hex characters")

    for name, value in OFFLINE_ENVIRONMENT.items():
        os.environ[name] = value

    if torch_module is None:
        import torch as torch_module  # runtime metadata only; no model factory import

    cuda = torch_module.cuda
    initialized_before = bool(cuda.is_initialized())
    available = bool(cuda.is_available())
    device_count = int(cuda.device_count()) if available else 0
    devices = [
        _device_record(cuda.get_device_properties(index), index)
        for index in range(device_count)
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": STATUS,
        "execution_sha": execution_sha,
        "image_digest": image_digest,
        "source_snapshot_tree_sha": source_snapshot_tree_sha,
        "runtime": {
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "executable": sys.executable,
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "platform": platform.platform(),
            },
            "packages": {
                name: _package_version(name) for name in PACKAGE_DISTRIBUTIONS
            },
            "torch_cuda_build": getattr(torch_module.version, "cuda", None),
            "cuda": {
                "available": available,
                "initialized_before_probe": initialized_before,
                "initialized_after_probe": bool(cuda.is_initialized()),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "device_count": device_count,
                "devices": devices,
            },
        },
        "offline": {
            **OFFLINE_ENVIRONMENT,
            "enforced_before_torch_import": True,
            "network_download_attempted": False,
        },
        "firewall": {
            "model_loaded": False,
            "tokenizer_loaded": False,
            "checkpoint_loaded": False,
            "model_output_accessed": False,
        },
    }
    validate_runtime_probe(payload)
    return payload


def validate_runtime_probe(payload: Mapping[str, Any]) -> None:
    """Fail closed on the identity/schema consumed by the execution plan."""

    if set(payload) != {
        "schema_version",
        "protocol_id",
        "status",
        "execution_sha",
        "image_digest",
        "source_snapshot_tree_sha",
        "runtime",
        "offline",
        "firewall",
    }:
        raise ValueError("runtime probe top-level schema changed")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != STATUS
    ):
        raise ValueError("runtime probe identity changed")
    if not EXECUTION_SHA_PATTERN.fullmatch(str(payload.get("execution_sha", ""))):
        raise ValueError("runtime probe execution SHA is invalid")
    if not IMAGE_DIGEST_PATTERN.fullmatch(str(payload.get("image_digest", ""))):
        raise ValueError("runtime probe image digest is invalid")
    if not SHA256_PATTERN.fullmatch(str(payload.get("source_snapshot_tree_sha", ""))):
        raise ValueError("runtime probe source snapshot tree SHA is invalid")
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "python", "platform", "packages", "torch_cuda_build", "cuda"
    }:
        raise ValueError("runtime probe runtime schema changed")
    python_record = runtime.get("python")
    platform_record = runtime.get("platform")
    if not isinstance(python_record, Mapping) or set(python_record) != {
        "version", "implementation", "executable"
    }:
        raise ValueError("runtime probe Python schema changed")
    if not isinstance(platform_record, Mapping) or set(platform_record) != {
        "system", "release", "version", "machine", "platform"
    }:
        raise ValueError("runtime probe platform schema changed")
    if any(not isinstance(value, str) or not value for value in python_record.values()):
        raise ValueError("runtime probe Python values are invalid")
    if any(not isinstance(value, str) for value in platform_record.values()):
        raise ValueError("runtime probe platform values are invalid")
    packages = runtime.get("packages")
    if not isinstance(packages, Mapping) or set(packages) != set(
        PACKAGE_DISTRIBUTIONS
    ):
        raise ValueError("runtime probe package-version schema changed")
    if any(not isinstance(value, str) or not value for value in packages.values()):
        raise ValueError("runtime probe package versions are invalid")
    if runtime.get("torch_cuda_build") is not None and not isinstance(
        runtime.get("torch_cuda_build"), str
    ):
        raise ValueError("runtime probe torch CUDA build is invalid")
    cuda = runtime.get("cuda")
    if not isinstance(cuda, Mapping) or set(cuda) != {
        "available",
        "initialized_before_probe",
        "initialized_after_probe",
        "cuda_visible_devices",
        "device_count",
        "devices",
    }:
        raise ValueError("runtime probe CUDA record is absent")
    if any(
        not isinstance(cuda.get(name), bool)
        for name in ("available", "initialized_before_probe", "initialized_after_probe")
    ):
        raise ValueError("runtime probe CUDA boolean flags are invalid")
    if cuda.get("cuda_visible_devices") is not None and not isinstance(
        cuda.get("cuda_visible_devices"), str
    ):
        raise ValueError("runtime probe CUDA_VISIBLE_DEVICES record is invalid")
    devices = cuda.get("devices")
    count = cuda.get("device_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("runtime probe CUDA device count is invalid")
    if not isinstance(devices, list) or len(devices) != count:
        raise ValueError("runtime probe CUDA device records do not match device_count")
    if not bool(cuda.get("available")) and count != 0:
        raise ValueError("unavailable CUDA runtime cannot expose devices")
    device_keys = {
        "index", "name", "total_memory_bytes", "compute_capability_major",
        "compute_capability_minor", "multi_processor_count", "uuid",
    }
    for index, device in enumerate(devices):
        if not isinstance(device, Mapping) or set(device) != device_keys:
            raise ValueError("runtime probe CUDA device schema changed")
        if device.get("index") != index or not isinstance(device.get("name"), str):
            raise ValueError("runtime probe CUDA device identity is invalid")
        for name in (
            "total_memory_bytes", "compute_capability_major",
            "compute_capability_minor", "multi_processor_count",
        ):
            if isinstance(device.get(name), bool) or not isinstance(device.get(name), int) or device[name] < 0:
                raise ValueError("runtime probe CUDA device numeric property is invalid")
        if device.get("uuid") is not None and not isinstance(device.get("uuid"), str):
            raise ValueError("runtime probe CUDA UUID is invalid")
    if payload.get("offline") != {
        **OFFLINE_ENVIRONMENT,
        "enforced_before_torch_import": True,
        "network_download_attempted": False,
    }:
        raise ValueError("runtime probe offline firewall changed")
    if payload.get("firewall") != {
        "model_loaded": False,
        "tokenizer_loaded": False,
        "checkpoint_loaded": False,
        "model_output_accessed": False,
    }:
        raise ValueError("runtime probe model-output firewall changed")


def atomic_write_probe(path: Path, payload: Mapping[str, Any]) -> None:
    """Create one immutable probe atomically; never overwrite an existing lock."""

    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--source-snapshot-tree-sha", required=True)
    args = parser.parse_args(argv)
    payload = build_runtime_probe(
        execution_sha=args.execution_sha,
        image_digest=args.image_digest,
        source_snapshot_tree_sha=args.source_snapshot_tree_sha,
    )
    atomic_write_probe(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
