#!/usr/bin/env python3
"""Write the model-output-free FPCT-E1 immutable-image runtime probe.

This probe is intended to run after Commit A inside the same immutable image
and on the same Kubernetes node class as the E1 capture shards.  It inspects
only interpreter/package/platform/CUDA metadata.  It neither imports model or
tokenizer factories nor reads a model, tokenizer, projector, or checkpoint.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence



_PRODUCER_RELATIVE = Path("script/experiment/fpct_e1_runtime_probe.py")
_SOURCE_LOCK_RELATIVE = Path("script/experiment/fpct_e1_source_snapshot_lock.py")


def _snapshot_root_from_file(path: Path, relative: Path) -> Path:
    absolute = path.absolute()
    if tuple(absolute.parts[-len(relative.parts) :]) != relative.parts:
        raise RuntimeError(f"local module path is outside the source snapshot: {absolute}")
    root = absolute
    for _ in relative.parts:
        root = root.parent
    return root


def _activate_snapshot_import_root() -> Path:
    root = _snapshot_root_from_file(Path(__file__), _PRODUCER_RELATIVE)
    root_text = str(root)
    if sys.path:
        sys.path[0] = root_text
    else:
        sys.path.append(root_text)
    return root


def _load_exact_local_module(module_label: str, relative: Path):
    root = _activate_snapshot_import_root()
    expected = (root / relative).absolute()
    if not expected.is_file() or expected.is_symlink():
        raise RuntimeError(f"exact local module is absent/symlinked: {expected}")
    unique_name = (
        f"_fpct_e1_exact_{module_label}_"
        f"{hashlib.sha256(str(expected).encode()).hexdigest()[:16]}"
    )
    specification = importlib.util.spec_from_file_location(unique_name, expected)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load exact local module: {expected}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[unique_name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(unique_name, None)
        raise
    observed = Path(str(getattr(module, "__file__", ""))).absolute()
    if observed != expected or observed.is_symlink():
        raise RuntimeError(f"local module provenance mismatch: {observed} != {expected}")
    return module


_SNAPSHOT_IMPORT_ROOT = _activate_snapshot_import_root()
_SOURCE_LOCK_MODULE = _load_exact_local_module(
    "source_snapshot_lock", _SOURCE_LOCK_RELATIVE
)
verify_source_snapshot_receipt = _SOURCE_LOCK_MODULE.verify_source_snapshot_receipt
LOCAL_IMPORT_PROVENANCE = {
    "source_snapshot_lock": str(
        (_SNAPSHOT_IMPORT_ROOT / _SOURCE_LOCK_RELATIVE).absolute()
    )
}


SCHEMA_VERSION = 1
PROTOCOL_ID = "fpct_e1_runtime_probe_v1"
STATUS = "FROZEN_MODEL_OUTPUT_FREE_RUNTIME_PROBE"
EXECUTION_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IMAGE_DIGEST_PATTERN = re.compile(
    r"^(?:[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?/)?"
    r"(?:[a-z0-9]+(?:[._-][a-z0-9]+)*/)*"
    r"[a-z0-9]+(?:[._-][a-z0-9]+)*@sha256:[0-9a-f]{64}$"
)
PACKAGE_DISTRIBUTIONS = ("torch", "transformers", "tokenizers", "pyarrow")
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
SOURCE_VERIFICATION_FIELDS = {
    "status",
    "execution_sha",
    "git_tree_oid",
    "git_entries_canonical_sha256",
    "mounted_tree_canonical_sha256",
    "entry_count",
    "total_bytes",
    "receipt_sha256",
    "receipt_file_sha256",
    "receipt_bytes",
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


def _validated_source_verification(
    verification: Mapping[str, Any],
    *,
    execution_sha: str,
    source_snapshot_tree_sha: str,
) -> dict[str, Any]:
    if not isinstance(verification, Mapping) or set(verification) != SOURCE_VERIFICATION_FIELDS:
        raise ValueError("runtime probe source-snapshot verification schema changed")
    if (
        verification.get("status") != "GO_MOUNTED_SOURCE_SNAPSHOT"
        or verification.get("execution_sha") != execution_sha
        or verification.get("mounted_tree_canonical_sha256")
        != source_snapshot_tree_sha
        or not re.fullmatch(
            r"[0-9a-f]{40,64}", str(verification.get("git_tree_oid", ""))
        )
        or not SHA256_PATTERN.fullmatch(
            str(verification.get("git_entries_canonical_sha256", ""))
        )
        or not SHA256_PATTERN.fullmatch(str(verification.get("receipt_sha256", "")))
        or not SHA256_PATTERN.fullmatch(
            str(verification.get("receipt_file_sha256", ""))
        )
    ):
        raise ValueError("runtime probe source-snapshot verification identity changed")
    for name in ("entry_count", "total_bytes", "receipt_bytes"):
        value = verification.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("runtime probe source-snapshot counters are invalid")
    return dict(verification)


def verify_mounted_source_snapshot(
    *,
    execution_sha: str,
    source_snapshot_tree_sha: str,
    source_snapshot_root: Path,
    source_snapshot_receipt: Path,
) -> dict[str, Any]:
    receipt_raw = source_snapshot_receipt.read_bytes()
    receipt_value = json.loads(receipt_raw.decode("utf-8"))
    if canonical_json_bytes(receipt_value) != receipt_raw:
        raise ValueError("source snapshot receipt bytes are not exact canonical JSON")
    verification = verify_source_snapshot_receipt(
        source_snapshot_receipt, source_snapshot_root, execution_sha
    )
    return _validated_source_verification(
        {
            **verification,
            "receipt_file_sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "receipt_bytes": len(receipt_raw),
        },
        execution_sha=execution_sha,
        source_snapshot_tree_sha=source_snapshot_tree_sha,
    )


def build_runtime_probe(
    *,
    execution_sha: str,
    image_digest: str,
    source_snapshot_tree_sha: str,
    source_snapshot_verification: Mapping[str, Any],
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Collect runtime-only metadata under an enforced offline environment."""

    if not EXECUTION_SHA_PATTERN.fullmatch(execution_sha):
        raise ValueError("execution_sha must be exactly 40 lowercase hex characters")
    if not IMAGE_DIGEST_PATTERN.fullmatch(image_digest):
        raise ValueError("image_digest must be an immutable sha256 image digest")
    if not SHA256_PATTERN.fullmatch(source_snapshot_tree_sha):
        raise ValueError("source_snapshot_tree_sha must be exactly 64 lowercase hex characters")
    source_verification = _validated_source_verification(
        source_snapshot_verification,
        execution_sha=execution_sha,
        source_snapshot_tree_sha=source_snapshot_tree_sha,
    )

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
        "source_snapshot_verification": source_verification,
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
        "source_snapshot_verification",
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
    _validated_source_verification(
        payload.get("source_snapshot_verification", {}),
        execution_sha=str(payload.get("execution_sha", "")),
        source_snapshot_tree_sha=str(payload.get("source_snapshot_tree_sha", "")),
    )
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
    parser.add_argument("--source-snapshot-root", type=Path, required=True)
    parser.add_argument("--source-snapshot-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    source_verification = verify_mounted_source_snapshot(
        execution_sha=args.execution_sha,
        source_snapshot_tree_sha=args.source_snapshot_tree_sha,
        source_snapshot_root=args.source_snapshot_root,
        source_snapshot_receipt=args.source_snapshot_receipt,
    )
    payload = build_runtime_probe(
        execution_sha=args.execution_sha,
        image_digest=args.image_digest,
        source_snapshot_tree_sha=args.source_snapshot_tree_sha,
        source_snapshot_verification=source_verification,
    )
    atomic_write_probe(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
