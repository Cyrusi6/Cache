#!/usr/bin/env python3
"""Build and verify immutable ConfigMaps for the frozen FPCT-E1 K8s run.

The module never calls Kubernetes or the network. ``build`` validates the
frozen plan, instrumentation gate, and runtime probe before emitting two
render-only immutable ConfigMap YAML files. ``verify`` consumes JSON exported
by kubectl and re-hashes the exact UTF-8 bytes Kubernetes would mount per key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from script.experiment.fpct_e1_runtime_probe import validate_runtime_probe


SCHEMA_VERSION = 1
PROTOCOL_ID = "fpct_e1_k8s_lock_bundle_v1"
FINALIZED_PROTOCOL_ID = "fpct_e1_k8s_finalized_lock_bundle_v1"
PLAN_PROTOCOL_ID = "fpct_e1_e0_design_capture_v2"
GATE_ID = "fpct_e1_instrumentation_hard_gate_v1"
REQUIRED_GATE_CHECKS = {
    "formula_oracles",
    "instrumentation_on_off_equivalent",
    "synthetic_query_variance_positive",
    "multistep_accumulation_preserved",
    "invalid_probability_gradient_exact_zero",
    "no_nan_inf",
}
CONFIGMAP_LIMIT_BYTES = 1024 * 1024
SHA256 = re.compile(r"^[0-9a-f]{64}$")
EXECUTION_SHA = re.compile(r"^[0-9a-f]{40}$")
IMAGE_DIGEST = re.compile(
    r"^(?:[a-z0-9.-]+(?::[0-9]+)?/)?[a-z0-9._-]+"
    r"(?:/[a-z0-9._-]+)*@sha256:[0-9a-f]{64}$"
)
PLAN_KEY = "e1_capture_plan.json"
GATE_KEY = "instrumentation_gate.json"
RUNTIME_KEY = "runtime_probe.json"
SOURCE_SNAPSHOT_RECEIPT_KEY = "source_snapshot_receipt.json"
SOURCE_SNAPSHOT_RECEIPT_PROTOCOL_ID = "fpct_e1_source_snapshot_lock_v1"
FINALIZED_RECEIPT_KEY = "e1_2_finalized_receipt.json"
STAGE_MANIFEST_KEY = "stage_artifact_manifest.json"
FINALIZED_STAGE = "e1_2_finalized"
SOURCE_STAGE = "e1_2_endpoints"


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json_bytes(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return raw, value


def _plan_hash(plan: Mapping[str, Any]) -> str:
    projected = dict(plan)
    projected.pop("plan_sha256", None)
    return sha256_bytes(canonical_json_bytes(projected))


def _source_record(path: Path, raw: bytes, mounted_key: str) -> dict[str, Any]:
    return {
        "path": str(path.absolute()),
        "mounted_key": mounted_key,
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }


def validate_frozen_sources(
    *,
    plan_path: Path,
    instrumentation_gate_path: Path,
    runtime_probe_path: Path,
    source_snapshot_receipt_path: Path,
) -> dict[str, Any]:
    """Validate identities and exact byte SHAs before ConfigMap serialization."""

    plan_raw, plan = _read_json_bytes(plan_path)
    gate_raw, gate = _read_json_bytes(instrumentation_gate_path)
    runtime_raw, runtime = _read_json_bytes(runtime_probe_path)
    source_receipt_raw, source_receipt = _read_json_bytes(source_snapshot_receipt_path)
    if (
        plan.get("schema_version") != 2
        or plan.get("protocol_id") != PLAN_PROTOCOL_ID
        or plan.get("status") != "PREPARED_NO_MODEL_LOAD"
        or plan.get("plan_sha256") != _plan_hash(plan)
    ):
        raise ValueError("frozen capture plan identity/self-hash mismatch")
    runtime_lock = plan.get("runtime_lock")
    source_snapshot = plan.get("source_snapshot")
    gate_lock = plan.get("instrumentation_gate")
    if not all(isinstance(value, Mapping) for value in (runtime_lock, source_snapshot, gate_lock)):
        raise ValueError("frozen plan lacks runtime/source/gate locks")
    execution_sha = str(runtime_lock.get("execution_sha", ""))
    image_digest = str(runtime_lock.get("image_digest", ""))
    tree_sha = str(source_snapshot.get("canonical_tree_sha256", ""))
    if not EXECUTION_SHA.fullmatch(execution_sha) or not IMAGE_DIGEST.fullmatch(image_digest):
        raise ValueError("frozen plan execution/image identity is invalid")
    if not SHA256.fullmatch(tree_sha) or source_snapshot.get("execution_sha") != execution_sha:
        raise ValueError("frozen plan source snapshot identity is invalid")
    receipt_lock = source_snapshot.get("git_receipt")
    if (
        not isinstance(receipt_lock, Mapping)
        or receipt_lock.get("sha256") != sha256_bytes(source_receipt_raw)
        or receipt_lock.get("bytes") != len(source_receipt_raw)
    ):
        raise ValueError("source snapshot receipt byte SHA/size differs from plan")
    receipt_without_hash = dict(source_receipt)
    receipt_without_hash.pop("receipt_sha256", None)
    receipt_self_hash = sha256_bytes(canonical_json_bytes(receipt_without_hash))
    if (
        source_receipt.get("schema_version") != 1
        or source_receipt.get("protocol_id") != SOURCE_SNAPSHOT_RECEIPT_PROTOCOL_ID
        or source_receipt.get("status") != "GO_IMMUTABLE_GIT_ARCHIVE"
        or source_receipt.get("execution_sha") != execution_sha
        or source_receipt.get("dirty") is not False
        or source_receipt.get("portable_root") != "."
        or source_receipt.get("receipt_sha256") != receipt_self_hash
    ):
        raise ValueError("source snapshot receipt identity/self-hash is invalid")
    repo_record = source_receipt.get("repo")
    snapshot_record = source_receipt.get("snapshot")
    if (
        not isinstance(repo_record, Mapping)
        or repo_record.get("head") != execution_sha
        or repo_record.get("dirty") is not False
        or not isinstance(snapshot_record, Mapping)
    ):
        raise ValueError("source snapshot receipt repo/snapshot identity is invalid")
    if (
        not re.fullmatch(r"[0-9a-f]{40,64}", str(repo_record.get("git_tree_oid", "")))
        or not SHA256.fullmatch(
            str(source_receipt.get("git_entries_canonical_sha256", ""))
        )
        or not SHA256.fullmatch(
            str(snapshot_record.get("mounted_tree_canonical_sha256", ""))
        )
        or any(
            isinstance(snapshot_record.get(name), bool)
            or not isinstance(snapshot_record.get(name), int)
            or snapshot_record[name] < 0
            for name in ("entry_count", "total_bytes")
        )
        or tree_sha != snapshot_record.get("mounted_tree_canonical_sha256")
    ):
        raise ValueError("source snapshot receipt canonical tree identity is invalid")
    expected_receipt_verification = {
        "status": "GO_MOUNTED_SOURCE_SNAPSHOT",
        "execution_sha": execution_sha,
        "git_tree_oid": repo_record.get("git_tree_oid"),
        "git_entries_canonical_sha256": source_receipt.get(
            "git_entries_canonical_sha256"
        ),
        "mounted_tree_canonical_sha256": snapshot_record.get(
            "mounted_tree_canonical_sha256"
        ),
        "entry_count": snapshot_record.get("entry_count"),
        "total_bytes": snapshot_record.get("total_bytes"),
        "receipt_sha256": receipt_self_hash,
    }
    if receipt_lock.get("verification") != expected_receipt_verification:
        raise ValueError("source snapshot receipt verification identity differs from plan")
    if gate_lock.get("sha256") != sha256_bytes(gate_raw):
        raise ValueError("instrumentation gate byte SHA differs from plan")
    runtime_record = runtime_lock.get("runtime_provenance")
    if not isinstance(runtime_record, Mapping) or runtime_record.get("sha256") != sha256_bytes(runtime_raw):
        raise ValueError("runtime probe byte SHA differs from plan")

    checks = gate.get("checks")
    if (
        gate.get("gate_id") != GATE_ID
        or gate.get("status") != "GO"
        or not isinstance(checks, Mapping)
        or set(checks) != REQUIRED_GATE_CHECKS
        or any(checks[name] is not True for name in REQUIRED_GATE_CHECKS)
    ):
        raise ValueError("instrumentation gate identity/check set is invalid")
    if (
        gate_lock.get("gate_id") != GATE_ID
        or gate_lock.get("status") != "GO"
        or gate_lock.get("checks") != checks
    ):
        raise ValueError("instrumentation gate contents differ from plan lock")
    evidence = gate.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("instrumentation gate lacks hashed evidence")
    if any(
        not isinstance(record, Mapping)
        or not record.get("logical_path")
        or not SHA256.fullmatch(str(record.get("sha256", "")))
        for record in evidence
    ):
        raise ValueError("instrumentation gate evidence is not fully hashed")

    validate_runtime_probe(runtime)
    expected_runtime_source_verification = {
        **expected_receipt_verification,
        "receipt_file_sha256": sha256_bytes(source_receipt_raw),
        "receipt_bytes": len(source_receipt_raw),
    }
    expected_runtime_identity = {
        "schema_version": runtime["schema_version"],
        "protocol_id": runtime["protocol_id"],
        "status": runtime["status"],
        "execution_sha": execution_sha,
        "image_digest": image_digest,
        "source_snapshot_tree_sha": tree_sha,
        "source_snapshot_verification": expected_runtime_source_verification,
    }
    if (
        runtime.get("execution_sha") != execution_sha
        or runtime.get("image_digest") != image_digest
        or runtime.get("source_snapshot_tree_sha") != tree_sha
        or runtime.get("source_snapshot_verification")
        != expected_runtime_source_verification
        or runtime_record.get("identity") != expected_runtime_identity
    ):
        raise ValueError("runtime probe/source-receipt identity differs from frozen plan")
    return {
        "plan": plan,
        "gate": gate,
        "runtime_probe": runtime,
        "execution_sha": execution_sha,
        "image_digest": image_digest,
        "source_snapshot_tree_sha": tree_sha,
        "sources": {
            "plan": _source_record(plan_path, plan_raw, PLAN_KEY),
            "instrumentation_gate": _source_record(
                instrumentation_gate_path, gate_raw, GATE_KEY
            ),
            "runtime_probe": _source_record(runtime_probe_path, runtime_raw, RUNTIME_KEY),
            "source_snapshot_receipt": _source_record(
                source_snapshot_receipt_path,
                source_receipt_raw,
                SOURCE_SNAPSHOT_RECEIPT_KEY,
            ),
        },
        "source_bytes": {
            PLAN_KEY: plan_raw,
            GATE_KEY: gate_raw,
            RUNTIME_KEY: runtime_raw,
            SOURCE_SNAPSHOT_RECEIPT_KEY: source_receipt_raw,
        },
    }


def _configmap_yaml(
    *,
    name: str,
    namespace: str,
    execution_sha: str,
    plan_sha: str,
    values: Mapping[str, bytes],
) -> tuple[bytes, dict[str, Any]]:
    if any(not isinstance(raw, bytes) for raw in values.values()):
        raise TypeError("ConfigMap sources must be exact bytes")
    try:
        data = {key: raw.decode("utf-8") for key, raw in sorted(values.items())}
    except UnicodeDecodeError as exc:
        raise ValueError("ConfigMap source is not UTF-8 JSON") from exc
    obj = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {
                "project": "fpct",
                "study": "fpct-e1-mechanism-audit",
                "execution-sha": execution_sha,
            },
            "annotations": {
                "fpct.openai.com/rendered-lock-bundle": "true",
                "fpct.openai.com/plan-sha256": plan_sha,
            },
        },
        "immutable": True,
        "data": data,
    }
    data_payload = canonical_json_bytes(data)
    serialized_object = canonical_json_bytes(obj)
    yaml_bytes = yaml.safe_dump(
        obj,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
        width=4096,
    ).encode("utf-8")
    roundtrip = yaml.safe_load(yaml_bytes)
    if roundtrip != obj:
        raise RuntimeError("ConfigMap YAML serialization is not byte-semantics preserving")
    sizes = {
        "mounted_value_bytes": sum(len(value) for value in values.values()),
        "data_payload_bytes": len(data_payload),
        "serialized_object_bytes": len(serialized_object),
        "yaml_bytes": len(yaml_bytes),
    }
    oversized = {key: value for key, value in sizes.items() if value >= CONFIGMAP_LIMIT_BYTES}
    if oversized:
        raise ValueError(f"ConfigMap {name} reaches/exceeds the 1MiB hard limit: {oversized}")
    return yaml_bytes, {
        "name": name,
        "namespace": namespace,
        "immutable": True,
        "keys": {
            key: {"bytes": len(raw), "sha256": sha256_bytes(raw)}
            for key, raw in sorted(values.items())
        },
        **sizes,
        "yaml_sha256": sha256_bytes(yaml_bytes),
    }


def _atomic_write(path: Path, raw: bytes) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_lock_bundle(
    *,
    plan_path: Path,
    instrumentation_gate_path: Path,
    runtime_probe_path: Path,
    source_snapshot_receipt_path: Path,
    output_dir: Path,
    namespace: str = "c2c-research",
) -> dict[str, Any]:
    """Build two immutable ConfigMap YAML objects and their provenance manifest."""

    if not namespace or not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", namespace):
        raise ValueError("invalid Kubernetes namespace")
    frozen = validate_frozen_sources(
        plan_path=plan_path,
        instrumentation_gate_path=instrumentation_gate_path,
        runtime_probe_path=runtime_probe_path,
        source_snapshot_receipt_path=source_snapshot_receipt_path,
    )
    execution_sha = frozen["execution_sha"]
    plan_sha = frozen["plan"]["plan_sha256"]
    suffix = f"{execution_sha[:8]}-{plan_sha[:12]}"
    specifications = {
        "plan_gate": (
            f"fpct-e1-lock-{suffix}",
            {
                key: frozen["source_bytes"][key]
                for key in (PLAN_KEY, GATE_KEY, SOURCE_SNAPSHOT_RECEIPT_KEY)
            },
        ),
        "runtime_probe": (
            f"fpct-e1-runtime-{suffix}",
            {RUNTIME_KEY: frozen["source_bytes"][RUNTIME_KEY]},
        ),
    }
    objects: dict[str, Any] = {}
    rendered: list[tuple[Path, bytes]] = []
    for role, (name, values) in specifications.items():
        raw, record = _configmap_yaml(
            name=name,
            namespace=namespace,
            execution_sha=execution_sha,
            plan_sha=plan_sha,
            values=values,
        )
        filename = f"{name}.yaml"
        record["yaml_path"] = str((output_dir / filename).absolute())
        objects[role] = record
        rendered.append((output_dir / filename, raw))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_NOT_APPLIED",
        "execution_sha": execution_sha,
        "plan_sha256": plan_sha,
        "image_digest": frozen["image_digest"],
        "source_snapshot_tree_sha": frozen["source_snapshot_tree_sha"],
        "configmap_limit_bytes_exclusive": CONFIGMAP_LIMIT_BYTES,
        "sources": frozen["sources"],
        "configmaps": objects,
        "network_accessed": False,
        "kubectl_invoked": False,
    }
    manifest_path = output_dir / "fpct_e1_k8s_lock_bundle_manifest.json"
    collisions = [str(path) for path, _ in rendered if path.exists()]
    if manifest_path.exists():
        collisions.append(str(manifest_path))
    if collisions:
        raise FileExistsError(f"lock-bundle outputs already exist: {collisions}")
    for path, raw in rendered:
        _atomic_write(path, raw)
    _atomic_write(manifest_path, canonical_json_bytes(manifest))
    return manifest


def _artifact_record(
    record: Any, *, relative_path: str, require_row_count: bool = False
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"finalized artifact record is absent: {relative_path}")
    digest = str(record.get("sha256", ""))
    size = record.get("bytes")
    if not SHA256.fullmatch(digest) or isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"finalized artifact hash/size is invalid: {relative_path}")
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts or relative_path in {"", "."}:
        raise ValueError(f"finalized artifact path is not portable: {relative_path}")
    mode = record.get("mode")
    if not isinstance(mode, str) or re.fullmatch(r"0o[0-7]{3}", mode) is None:
        raise ValueError(f"finalized artifact mode is invalid: {relative_path}")
    result = {
        "relative_path": relative_path,
        "sha256": digest,
        "bytes": size,
        "mode": mode,
    }
    if require_row_count:
        row_count = record.get("row_count")
        if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
            raise ValueError("finalized merged capture row_count is invalid")
        result["row_count"] = row_count
    return result


def validate_finalized_sources(
    *, plan_path: Path, finalized_receipt_path: Path, stage_manifest_path: Path
) -> dict[str, Any]:
    """Validate plan/closure/artifact-tree identity without reading artifacts."""

    plan_raw, plan = _read_json_bytes(plan_path)
    receipt_raw, receipt = _read_json_bytes(finalized_receipt_path)
    stage_raw, stage_manifest = _read_json_bytes(stage_manifest_path)
    if (
        plan.get("schema_version") != 2
        or plan.get("protocol_id") != PLAN_PROTOCOL_ID
        or plan.get("status") != "PREPARED_NO_MODEL_LOAD"
        or plan.get("plan_sha256") != _plan_hash(plan)
    ):
        raise ValueError("frozen capture plan identity/self-hash mismatch")
    plan_sha = plan["plan_sha256"]
    runtime_lock = plan.get("runtime_lock")
    if not isinstance(runtime_lock, Mapping) or not EXECUTION_SHA.fullmatch(
        str(runtime_lock.get("execution_sha", ""))
    ):
        raise ValueError("frozen capture plan execution identity is invalid")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != PLAN_PROTOCOL_ID
        or receipt.get("status") != "GO"
        or receipt.get("stage") != FINALIZED_STAGE
        or receipt.get("source_stage") != SOURCE_STAGE
        or receipt.get("plan_sha256") != plan_sha
        or receipt.get("e1_pilot_consumed") is not False
        or receipt.get("confirmatory_consumed") is not False
    ):
        raise ValueError("finalized receipt identity/plan binding is invalid")
    if receipt.get("raw_topology_lock") != plan.get("raw_topology_lock"):
        raise ValueError("finalized receipt raw-topology identity differs from plan")
    if not isinstance(receipt.get("output_dir"), str) or not receipt["output_dir"].startswith(
        "output://"
    ):
        raise ValueError("finalized receipt output path is not portable")
    closure = str(receipt.get("closure_sha256", ""))
    if not SHA256.fullmatch(closure):
        raise ValueError("finalized receipt closure SHA is invalid")
    frozen_stage = receipt.get("stage_artifact_manifest")
    if not isinstance(frozen_stage, Mapping) or frozen_stage != {
        "sha256": sha256_bytes(stage_raw), "bytes": len(stage_raw)
    }:
        raise ValueError("stage manifest byte identity differs from finalized receipt")
    if (
        stage_manifest.get("schema_version") != 1
        or stage_manifest.get("protocol_id") != PLAN_PROTOCOL_ID
        or stage_manifest.get("status") != "GO"
        or stage_manifest.get("stage") != SOURCE_STAGE
        or stage_manifest.get("plan_sha256") != plan_sha
        or stage_manifest.get("closure_sha256") != closure
        or stage_manifest.get("e1_pilot_consumed") is not False
        or stage_manifest.get("confirmatory_consumed") is not False
    ):
        raise ValueError("stage manifest plan/closure identity is invalid")
    for name in ("analyzer_verification", "merged_capture", "analyzer_artifacts"):
        if stage_manifest.get(name) != receipt.get(name):
            raise ValueError(f"finalized receipt/stage manifest differ: {name}")

    merged = stage_manifest.get("merged_capture")
    if not isinstance(merged, Mapping):
        raise ValueError("finalized merged capture identity is absent")
    merged_path = str(merged.get("relative_path", ""))
    artifact_tree = [
        {
            "role": "merged_capture",
            **_artifact_record(
                merged, relative_path=merged_path, require_row_count=True
            ),
        }
    ]
    analyzer = stage_manifest.get("analyzer_artifacts")
    if not isinstance(analyzer, Mapping) or not analyzer:
        raise ValueError("finalized analyzer artifact tree is absent")
    for filename, record in sorted(analyzer.items()):
        artifact_tree.append(
            {"role": "analyzer", **_artifact_record(record, relative_path=str(filename))}
        )
    artifact_tree_sha = sha256_bytes(canonical_json_bytes(artifact_tree))
    for source_name, source in (("receipt", receipt), ("stage manifest", stage_manifest)):
        declared = source.get("artifact_tree_sha256")
        if declared is not None and declared != artifact_tree_sha:
            raise ValueError(f"{source_name} artifact tree SHA differs from recomputed identity")
    return {
        "plan": plan,
        "receipt": receipt,
        "stage_manifest": stage_manifest,
        "execution_sha": runtime_lock["execution_sha"],
        "plan_sha256": plan_sha,
        "receipt_sha256": sha256_bytes(receipt_raw),
        "closure_sha256": closure,
        "artifact_tree": artifact_tree,
        "artifact_tree_sha256": artifact_tree_sha,
        "sources": {
            "plan": _source_record(plan_path, plan_raw, PLAN_KEY),
            "finalized_receipt": _source_record(
                finalized_receipt_path, receipt_raw, FINALIZED_RECEIPT_KEY
            ),
            "stage_manifest": _source_record(
                stage_manifest_path, stage_raw, STAGE_MANIFEST_KEY
            ),
        },
        "source_bytes": {
            FINALIZED_RECEIPT_KEY: receipt_raw,
            STAGE_MANIFEST_KEY: stage_raw,
        },
    }


def build_finalized_lock_bundle(
    *,
    plan_path: Path,
    finalized_receipt_path: Path,
    stage_manifest_path: Path,
    output_dir: Path,
    namespace: str = "c2c-research",
) -> dict[str, Any]:
    """Build the immutable E1-2 finalized-receipt ConfigMap."""

    if not namespace or not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", namespace):
        raise ValueError("invalid Kubernetes namespace")
    frozen = validate_finalized_sources(
        plan_path=plan_path,
        finalized_receipt_path=finalized_receipt_path,
        stage_manifest_path=stage_manifest_path,
    )
    name = (
        f"fpct-e1-finalized-{frozen['plan_sha256'][:12]}-"
        f"{frozen['receipt_sha256'][:12]}"
    )
    yaml_raw, configmap = _configmap_yaml(
        name=name,
        namespace=namespace,
        execution_sha=frozen["execution_sha"],
        plan_sha=frozen["plan_sha256"],
        values=frozen["source_bytes"],
    )
    yaml_path = output_dir / f"{name}.yaml"
    configmap["yaml_path"] = str(yaml_path.absolute())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": FINALIZED_PROTOCOL_ID,
        "status": "FROZEN_FINALIZED_NOT_APPLIED",
        "execution_sha": frozen["execution_sha"],
        "plan_sha256": frozen["plan_sha256"],
        "receipt_sha256": frozen["receipt_sha256"],
        "closure_sha256": frozen["closure_sha256"],
        "artifact_tree_sha256": frozen["artifact_tree_sha256"],
        "artifact_tree": frozen["artifact_tree"],
        "configmap_limit_bytes_exclusive": CONFIGMAP_LIMIT_BYTES,
        "sources": frozen["sources"],
        "configmaps": {"finalized": configmap},
        "network_accessed": False,
        "kubectl_invoked": False,
    }
    manifest_path = output_dir / "fpct_e1_k8s_finalized_lock_bundle_manifest.json"
    collisions = [path for path in (yaml_path, manifest_path) if path.exists()]
    if collisions:
        raise FileExistsError(f"finalized lock-bundle outputs already exist: {collisions}")
    _atomic_write(yaml_path, yaml_raw)
    _atomic_write(manifest_path, canonical_json_bytes(manifest))
    return manifest


def _configmap_objects(paths: Iterable[Path]) -> Iterable[Mapping[str, Any]]:
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, Mapping) and value.get("kind") == "List":
            items = value.get("items")
            if not isinstance(items, list):
                raise ValueError("kubectl ConfigMap List has no items")
            yield from items
        elif isinstance(value, Mapping):
            yield value
        else:
            raise ValueError("kubectl export must be a ConfigMap object or List")


def _verify_mounted_configmap_keys(
    manifest: Mapping[str, Any], configmap_json_paths: Sequence[Path]
) -> dict[str, Any]:
    expected = {
        record["name"]: record for record in manifest.get("configmaps", {}).values()
    }
    observed: dict[str, Any] = {}
    for obj in _configmap_objects(configmap_json_paths):
        name = obj.get("metadata", {}).get("name")
        if name not in expected or name in observed:
            raise ValueError(f"unexpected/duplicate exported ConfigMap: {name}")
        if obj.get("apiVersion") != "v1" or obj.get("kind") != "ConfigMap" or obj.get("immutable") is not True:
            raise ValueError(f"exported ConfigMap identity/immutability changed: {name}")
        if obj.get("metadata", {}).get("namespace") != expected[name]["namespace"]:
            raise ValueError(f"exported ConfigMap namespace changed: {name}")
        if obj.get("binaryData") not in (None, {}):
            raise ValueError(f"exported ConfigMap unexpectedly contains binaryData: {name}")
        data = obj.get("data")
        if not isinstance(data, Mapping) or set(data) != set(expected[name]["keys"]):
            raise ValueError(f"exported ConfigMap key universe changed: {name}")
        key_records = {}
        for key, locked in expected[name]["keys"].items():
            value = data[key]
            if not isinstance(value, str):
                raise ValueError(f"exported ConfigMap value is not text: {name}/{key}")
            raw = value.encode("utf-8")
            record = {"bytes": len(raw), "sha256": sha256_bytes(raw)}
            if record != {"bytes": locked["bytes"], "sha256": locked["sha256"]}:
                raise ValueError(f"mounted ConfigMap key bytes changed: {name}/{key}")
            key_records[key] = record
        observed[name] = {"immutable": True, "keys": key_records}
    if set(observed) != set(expected):
        raise ValueError("not all frozen ConfigMaps were exported for verification")
    return observed


def verify_exported_configmaps(
    *, bundle_manifest_path: Path, configmap_json_paths: Sequence[Path]
) -> dict[str, Any]:
    """Verify the exact mounted bytes from kubectl-exported ConfigMap JSON."""

    manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status") != "FROZEN_NOT_APPLIED"
    ):
        raise ValueError("lock-bundle manifest identity mismatch")
    observed = _verify_mounted_configmap_keys(manifest, configmap_json_paths)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "VERIFIED_MOUNTED_KEY_BYTES",
        "execution_sha": manifest["execution_sha"],
        "plan_sha256": manifest["plan_sha256"],
        "bundle_manifest_sha256": sha256_bytes(bundle_manifest_path.read_bytes()),
        "configmaps": observed,
        "network_accessed": False,
        "kubectl_invoked_by_verifier": False,
    }


def verify_finalized_exported_configmap(
    *, bundle_manifest_path: Path, configmap_json_paths: Sequence[Path]
) -> dict[str, Any]:
    """Verify mounted finalized receipt/stage-manifest bytes from kubectl JSON."""

    manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("protocol_id") != FINALIZED_PROTOCOL_ID
        or manifest.get("status") != "FROZEN_FINALIZED_NOT_APPLIED"
        or set(manifest.get("configmaps", {})) != {"finalized"}
        or not SHA256.fullmatch(str(manifest.get("receipt_sha256", "")))
        or not SHA256.fullmatch(str(manifest.get("closure_sha256", "")))
        or not SHA256.fullmatch(str(manifest.get("artifact_tree_sha256", "")))
    ):
        raise ValueError("finalized lock-bundle manifest identity mismatch")
    observed = _verify_mounted_configmap_keys(manifest, configmap_json_paths)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": FINALIZED_PROTOCOL_ID,
        "status": "VERIFIED_FINALIZED_MOUNTED_KEY_BYTES",
        "execution_sha": manifest["execution_sha"],
        "plan_sha256": manifest["plan_sha256"],
        "bundle_manifest_sha256": sha256_bytes(bundle_manifest_path.read_bytes()),
        "receipt_sha256": manifest["receipt_sha256"],
        "closure_sha256": manifest["closure_sha256"],
        "artifact_tree_sha256": manifest["artifact_tree_sha256"],
        "configmaps": observed,
        "network_accessed": False,
        "kubectl_invoked_by_verifier": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--plan", type=Path, required=True)
    build.add_argument("--instrumentation-gate", type=Path, required=True)
    build.add_argument("--runtime-probe", type=Path, required=True)
    build.add_argument("--source-snapshot-receipt", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--namespace", default="c2c-research")
    build_finalized = subparsers.add_parser("build-finalized")
    build_finalized.add_argument("--plan", type=Path, required=True)
    build_finalized.add_argument("--finalized-receipt", type=Path, required=True)
    build_finalized.add_argument("--stage-manifest", type=Path, required=True)
    build_finalized.add_argument("--output-dir", type=Path, required=True)
    build_finalized.add_argument("--namespace", default="c2c-research")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--bundle-manifest", type=Path, required=True)
    verify.add_argument("--configmap-json", type=Path, action="append", required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify_finalized = subparsers.add_parser("verify-finalized")
    verify_finalized.add_argument("--bundle-manifest", type=Path, required=True)
    verify_finalized.add_argument(
        "--configmap-json", type=Path, action="append", required=True
    )
    verify_finalized.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_lock_bundle(
            plan_path=args.plan,
            instrumentation_gate_path=args.instrumentation_gate,
            runtime_probe_path=args.runtime_probe,
            source_snapshot_receipt_path=args.source_snapshot_receipt,
            output_dir=args.output_dir,
            namespace=args.namespace,
        )
    elif args.command == "build-finalized":
        result = build_finalized_lock_bundle(
            plan_path=args.plan,
            finalized_receipt_path=args.finalized_receipt,
            stage_manifest_path=args.stage_manifest,
            output_dir=args.output_dir,
            namespace=args.namespace,
        )
    elif args.command == "verify":
        result = verify_exported_configmaps(
            bundle_manifest_path=args.bundle_manifest,
            configmap_json_paths=args.configmap_json,
        )
        _atomic_write(args.output, canonical_json_bytes(result))
    else:
        result = verify_finalized_exported_configmap(
            bundle_manifest_path=args.bundle_manifest,
            configmap_json_paths=args.configmap_json,
        )
        _atomic_write(args.output, canonical_json_bytes(result))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
