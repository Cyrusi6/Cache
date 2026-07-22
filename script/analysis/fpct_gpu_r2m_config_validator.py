#!/usr/bin/env python3
"""Fail-closed config-closure validator for FPCT GPU R2m."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any


EXPECTED_TASKS = ("ai2-arc", "mmlu-redux", "openbookqa")
EXPECTED_SOURCE_SHA256 = "6d5f6221be15db5af030f9c1d7702b6d1e814bfe4e423e065353f3167c914284"
EXPECTED_GEOMETRY_SHA256 = "221c5164c60ec4d27abe714a68cec2f6a1a630d031195e0f0e63818133a5c6a2"


class ConfigClosureError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ConfigClosureError(f"non-finite JSON constant in {path}: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ConfigClosureError(f"value is not finite canonical JSON: {error}") from error


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def json_pointer(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ConfigClosureError(f"invalid JSON pointer: {pointer}")
    current = value
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or token not in current:
            raise ConfigClosureError(f"missing pointer: {pointer}")
        current = current[token]
    return current


def _is_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    if expected == "null":
        return value is None
    raise ConfigClosureError(f"unsupported schema type: {expected}")


def validate_schema(value: Any, schema: dict[str, Any], path: str = "") -> None:
    location = path or "/"
    if "const" in schema and value != schema["const"]:
        raise ConfigClosureError(f"schema const mismatch at {location}")
    if "enum" in schema and value not in schema["enum"]:
        raise ConfigClosureError(f"schema enum mismatch at {location}")
    expected_type = schema.get("type")
    if expected_type is not None and not _is_type(value, expected_type):
        raise ConfigClosureError(f"schema type mismatch at {location}: expected {expected_type}")
    if isinstance(value, str) and "pattern" in schema:
        if re.search(str(schema["pattern"]), value) is None:
            raise ConfigClosureError(f"schema pattern mismatch at {location}")
    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise ConfigClosureError(f"missing pointer: {path}/{key}" if path else f"missing pointer: /{key}")
        if "minProperties" in schema and len(value) < int(schema["minProperties"]):
            raise ConfigClosureError(f"too few properties at {location}")
        if "maxProperties" in schema and len(value) > int(schema["maxProperties"]):
            raise ConfigClosureError(f"too many properties at {location}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                raise ConfigClosureError(f"additional properties at {location}: {extras}")
        for key, child_schema in properties.items():
            if key in value:
                child_path = f"{path}/{key}" if path else f"/{key}"
                validate_schema(value[key], child_schema, child_path)


def geometry_projection(value: dict[str, Any]) -> dict[str, Any]:
    if "resource_geometry" in value:
        geometry = json_pointer(value, "/resource_geometry/tinyllama_all_splits")
    elif "tinyllama_all_splits" in value:
        geometry = value["tinyllama_all_splits"]
    else:
        geometry = value
    if not isinstance(geometry, dict):
        raise ConfigClosureError("canonical geometry must be an object")
    return geometry


def validate_geometry(value: dict[str, Any], canonical: dict[str, Any]) -> dict[str, Any]:
    geometry = geometry_projection(value)
    expected = geometry_projection(canonical)
    if geometry.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise ConfigClosureError("geometry source SHA256 mismatch")
    tasks = geometry.get("tasks")
    if not isinstance(tasks, dict) or tuple(sorted(tasks)) != EXPECTED_TASKS:
        raise ConfigClosureError(f"geometry task set mismatch: {sorted(tasks) if isinstance(tasks, dict) else tasks}")
    for task in EXPECTED_TASKS:
        row = tasks[task]
        if not isinstance(row, dict) or set(row) != {"mean", "p95", "max"}:
            raise ConfigClosureError(f"geometry row schema mismatch: {task}")
        for field in ("mean", "p95", "max"):
            if not _is_type(row[field], "number"):
                raise ConfigClosureError(f"non-finite geometry value: {task}/{field}")
    projection_sha = sha256_bytes(canonical_bytes(geometry))
    if projection_sha != EXPECTED_GEOMETRY_SHA256:
        raise ConfigClosureError(f"geometry projection SHA mismatch: {projection_sha}")
    if geometry != expected:
        raise ConfigClosureError("geometry differs from canonical R2k object")
    return {
        "source_sha256": geometry["source_sha256"],
        "task_names": sorted(tasks),
        "row_count": len(tasks),
        "geometry_projection_sha256": projection_sha,
    }


def canonical_parsed_sha(lock: dict[str, Any]) -> str:
    projected = copy.deepcopy(lock)
    closure = projected.get("config_closure")
    if not isinstance(closure, dict):
        raise ConfigClosureError("missing pointer: /config_closure")
    closure.pop("canonical_parsed_sha256", None)
    return sha256_bytes(canonical_bytes(projected))


def _aggregate_consumer_nodes(path: Path) -> list[ast.AST]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "aggregate_pretrained"
    )
    selected: list[ast.AST] = []
    checks_value = None
    for node in function.body:
        if isinstance(node, ast.Assign):
            names = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            if names & {"geometry", "geometry_rows"}:
                selected.append(node)
            if "checks" in names and isinstance(node.value, ast.Dict):
                checks_value = node.value
    if checks_value is None or len(selected) != 2:
        raise ConfigClosureError("frozen aggregate geometry consumer AST not found")
    for key, value in zip(checks_value.keys, checks_value.values):
        if isinstance(key, ast.Constant) and key.value in {"expansion_mean", "expansion_p95"}:
            selected.append(ast.Expr(value=value))
    if len(selected) != 4:
        raise ConfigClosureError("frozen expansion check AST not found")
    return selected


def aggregate_consumer_ast_sha(path: Path) -> str:
    payload = "\n".join(ast.dump(node, include_attributes=False) for node in _aggregate_consumer_nodes(path))
    return sha256_bytes(payload.encode("utf-8"))


def exercise_original_geometry_consumer(
    lock: dict[str, Any], canonical: dict[str, Any], runner_path: Path
) -> dict[str, Any]:
    nodes = _aggregate_consumer_nodes(runner_path)
    environment: dict[str, Any] = {"lock": lock}
    for node in nodes[:2]:
        module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(module, str(runner_path), "exec"), {}, environment)
    results = []
    for node in nodes[2:]:
        expression = ast.fix_missing_locations(ast.Expression(body=node.value))
        results.append(bool(eval(compile(expression, str(runner_path), "eval"), {}, environment)))
    expected = geometry_projection(canonical)
    if environment["geometry"] != expected:
        raise ConfigClosureError("original aggregate consumer did not receive canonical geometry")
    if results != [True, True]:
        raise ConfigClosureError(f"original aggregate expansion fixture failed: {results}")
    return {
        "certified_geometry": environment["geometry"],
        "expansion_mean": results[0],
        "expansion_p95": results[1],
    }


def validate_consumer_manifest(repo: Path, manifest: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    checked = []
    for consumer in manifest.get("consumers", []):
        path = repo / consumer["path"]
        if not path.is_file():
            raise ConfigClosureError(f"consumer missing: {consumer['path']}")
        actual_sha = sha256_file(path)
        if actual_sha != consumer["sha256"]:
            raise ConfigClosureError(f"consumer blob mismatch: {consumer['path']}")
        source = path.read_text(encoding="utf-8")
        for function_name in str(consumer.get("function", "")).split("/"):
            if function_name and f"def {function_name}(" not in source:
                raise ConfigClosureError(f"consumer function missing: {consumer['path']}::{function_name}")
        for row in consumer.get("run_lock_pointers", []):
            try:
                pointer_value = json_pointer(lock, row["pointer"])
            except ConfigClosureError:
                if row.get("required"):
                    raise
                continue
            if not _is_type(pointer_value, row["type"]):
                raise ConfigClosureError(f"consumer pointer type mismatch: {row['pointer']}")
        checked.append(consumer["path"])
    runner = repo / "script/experiment/fpct_gpu_r2_runner.py"
    actual_ast = aggregate_consumer_ast_sha(runner)
    if actual_ast != manifest.get("aggregate_geometry_consumer_ast_sha256"):
        raise ConfigClosureError("aggregate geometry consumer AST mismatch")
    return {"consumer_count": len(checked), "consumer_paths": checked, "consumer_ast_sha256": actual_ast}


def validate_production_blobs(repo: Path, allowlist: dict[str, Any]) -> dict[str, str]:
    actual = {}
    for relative, expected in allowlist["forbidden_production_sha256"].items():
        digest = sha256_file(repo / relative)
        if digest != expected:
            raise ConfigClosureError(f"production blob changed: {relative}")
        actual[relative] = digest
    math_digest = sha256_file(repo / "math.md")
    if math_digest != allowlist["math_md_sha256"]:
        raise ConfigClosureError("math.md changed")
    return actual


def validate_changed_paths(repo: Path, allowlist: dict[str, Any]) -> list[str]:
    baseline = allowlist["baseline_commit"]
    output = subprocess.run(
        ["git", "diff", "--name-only", baseline, "--"], cwd=repo,
        check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.splitlines()
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=repo,
        check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.splitlines()
    changed = sorted(set(output + untracked))
    allowed_existing = set(allowlist["allowed_existing_non_scientific_files"])
    prefixes = tuple(allowlist["allowed_new_prefixes"])
    unexpected = [path for path in changed if path not in allowed_existing and not path.startswith(prefixes)]
    if unexpected:
        raise ConfigClosureError(f"paths outside R2m allowlist: {unexpected}")
    return changed


def validate_exact_bytes(
    git_lock: Path,
    *,
    configmap_lock: Path | None = None,
    mounted_lock: Path | None = None,
    container_lock: Path | None = None,
    expected_raw_sha256: str | None = None,
) -> dict[str, Any]:
    paths = [git_lock] + [path for path in (configmap_lock, mounted_lock, container_lock) if path is not None]
    blobs = [path.read_bytes() for path in paths]
    if any(blob != blobs[0] for blob in blobs[1:]):
        raise ConfigClosureError("Git/ConfigMap/Pod/container lock bytes differ")
    digest = sha256_bytes(blobs[0])
    if expected_raw_sha256 is not None and digest != expected_raw_sha256:
        raise ConfigClosureError("raw run-lock SHA256 mismatch")
    return {"raw_sha256": digest, "byte_count": len(blobs[0]), "path_count": len(paths)}


def validate_identity(
    lock: dict[str, Any],
    *,
    expected_run_uid: str | None = None,
    expected_root: str | None = None,
    expected_commit: str | None = None,
) -> None:
    if lock["scientific_code_commit"] != lock["scientific_code_upstream"]:
        raise ConfigClosureError("scientific commit/upstream mismatch")
    if expected_run_uid is not None and lock["run_uid"] != expected_run_uid:
        raise ConfigClosureError("stale run UID")
    if expected_root is not None and lock["storage"]["shared_run_root"] != expected_root:
        raise ConfigClosureError("stale run root")
    if expected_commit is not None and lock["scientific_code_commit"] != expected_commit:
        raise ConfigClosureError("stale scientific commit")
    if not lock["storage"]["shared_run_root"].endswith(lock["run_uid"]):
        raise ConfigClosureError("run root and UID mismatch")


def validate_configmap_object(configmap_path: Path, git_lock: Path) -> dict[str, Any]:
    configmap = strict_load(configmap_path)
    if configmap.get("kind") != "ConfigMap" or configmap.get("immutable") is not True:
        raise ConfigClosureError("ConfigMap is not immutable")
    data = configmap.get("data", {})
    mounted_name = "immutable_v1_run_lock.json"
    if mounted_name not in data:
        raise ConfigClosureError("ConfigMap is missing immutable_v1_run_lock.json")
    configmap_bytes = data[mounted_name].encode("utf-8")
    if configmap_bytes != git_lock.read_bytes():
        raise ConfigClosureError("ConfigMap and Git lock bytes differ")
    return {
        "configmap_name": configmap.get("metadata", {}).get("name"),
        "immutable": True,
        "raw_sha256": sha256_bytes(configmap_bytes),
        "byte_count": len(configmap_bytes),
    }


def validate_run_lock(
    repo: Path,
    lock_path: Path,
    schema_path: Path,
    canonical_path: Path,
    consumer_path: Path,
    allowlist_path: Path,
    *,
    expected_run_uid: str | None = None,
    expected_root: str | None = None,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    lock = strict_load(lock_path)
    schema = strict_load(schema_path)
    canonical = strict_load(canonical_path)
    consumers = strict_load(consumer_path)
    allowlist = strict_load(allowlist_path)
    validate_schema(lock, schema)
    geometry = validate_geometry(lock, canonical)
    closure = lock["config_closure"]
    expected_hashes = {
        "geometry_projection_sha256": geometry["geometry_projection_sha256"],
        "schema_sha256": sha256_file(schema_path),
        "consumer_pointer_manifest_sha256": sha256_file(consumer_path),
        "implementation_allowlist_sha256": sha256_file(allowlist_path),
        "consumer_ast_sha256": aggregate_consumer_ast_sha(repo / "script/experiment/fpct_gpu_r2_runner.py"),
    }
    for key, expected in expected_hashes.items():
        if closure.get(key) != expected:
            raise ConfigClosureError(f"closure hash mismatch: /config_closure/{key}")
    parsed_sha = canonical_parsed_sha(lock)
    if closure.get("canonical_parsed_sha256") != parsed_sha:
        raise ConfigClosureError("closure hash mismatch: /config_closure/canonical_parsed_sha256")
    consumer_result = validate_consumer_manifest(repo, consumers, lock)
    production = validate_production_blobs(repo, allowlist)
    changed_paths = validate_changed_paths(repo, allowlist)
    validate_identity(
        lock,
        expected_run_uid=expected_run_uid,
        expected_root=expected_root,
        expected_commit=expected_commit,
    )
    fixture = exercise_original_geometry_consumer(
        lock, canonical, repo / "script/experiment/fpct_gpu_r2_runner.py"
    )
    return {
        "status": "GO",
        "schema_status": "GO",
        "geometry": geometry,
        "consumer_closure": consumer_result,
        "production_blob_count": len(production),
        "changed_paths": changed_paths,
        "aggregate_fixture": fixture,
        "canonical_parsed_sha256": parsed_sha,
        "scientific_output": False,
        "training_authorized": False,
    }


def config_preflight(
    repo: Path,
    lock_path: Path,
    schema_path: Path,
    canonical_path: Path,
    consumer_path: Path,
    allowlist_path: Path,
    *,
    expected_run_uid: str,
    expected_root: str,
    expected_commit: str,
    expected_raw_sha256: str,
    configmap_object: Path,
    mounted_lock: Path,
    container_lock: Path,
) -> dict[str, Any]:
    validation = validate_run_lock(
        repo, lock_path, schema_path, canonical_path, consumer_path, allowlist_path,
        expected_run_uid=expected_run_uid,
        expected_root=expected_root,
        expected_commit=expected_commit,
    )
    configmap = validate_configmap_object(configmap_object, lock_path)
    byte_identity = validate_exact_bytes(
        lock_path,
        configmap_lock=lock_path,
        mounted_lock=mounted_lock,
        container_lock=container_lock,
        expected_raw_sha256=expected_raw_sha256,
    )
    return {
        "schema_version": 1,
        "protocol_id": "fpct_gpu_r2m_config_preflight_result_v1",
        "status": "CONFIG_PREFLIGHT_GO",
        "hashes": {
            "git_raw_sha256": byte_identity["raw_sha256"],
            "configmap_raw_sha256": configmap["raw_sha256"],
            "mounted_raw_sha256": sha256_file(mounted_lock),
            "container_raw_sha256": sha256_file(container_lock),
            "canonical_parsed_sha256": validation["canonical_parsed_sha256"],
            "geometry_projection_sha256": validation["geometry"]["geometry_projection_sha256"],
        },
        "schema_status": validation["schema_status"],
        "task_names": validation["geometry"]["task_names"],
        "row_count": validation["geometry"]["row_count"],
        "consumer_closure_status": "GO",
        "scientific_output": False,
        "training_authorized": False,
    }


def extract_geometry(source_lock: Path, output: Path) -> dict[str, Any]:
    source = strict_load(source_lock)
    geometry = geometry_projection(source)
    projection_sha = sha256_bytes(canonical_bytes(geometry))
    if projection_sha != EXPECTED_GEOMETRY_SHA256:
        raise ConfigClosureError("source geometry projection changed")
    repo = source_lock.resolve().parents[3]
    blob = subprocess.run(
        ["git", "hash-object", str(source_lock.resolve())], cwd=repo,
        check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.strip()
    payload = {
        "schema_version": 1,
        "protocol_id": "fpct_gpu_r2m_canonical_resource_geometry_v1",
        "source": {
            "path": str(source_lock),
            "file_sha256": sha256_file(source_lock),
            "git_blob": blob,
            "json_pointer": "/resource_geometry/tinyllama_all_splits",
        },
        "row_count": len(geometry["tasks"]),
        "geometry_projection_sha256": projection_sha,
        "tinyllama_all_splits": geometry,
        "label_or_correctness_accessed": False,
    }
    atomic_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract-geometry")
    extract.add_argument("--source-lock", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    fingerprint = commands.add_parser("consumer-fingerprint")
    fingerprint.add_argument("--runner", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--repo", type=Path, required=True)
    validate.add_argument("--lock", type=Path, required=True)
    validate.add_argument("--schema", type=Path, required=True)
    validate.add_argument("--canonical-geometry", type=Path, required=True)
    validate.add_argument("--consumer-manifest", type=Path, required=True)
    validate.add_argument("--allowlist", type=Path, required=True)
    validate.add_argument("--expected-run-uid")
    validate.add_argument("--expected-root")
    validate.add_argument("--expected-commit")
    validate.add_argument("--output", type=Path)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--repo", type=Path, required=True)
    preflight.add_argument("--lock", type=Path, required=True)
    preflight.add_argument("--schema", type=Path, required=True)
    preflight.add_argument("--canonical-geometry", type=Path, required=True)
    preflight.add_argument("--consumer-manifest", type=Path, required=True)
    preflight.add_argument("--allowlist", type=Path, required=True)
    preflight.add_argument("--expected-run-uid", required=True)
    preflight.add_argument("--expected-root", required=True)
    preflight.add_argument("--expected-commit", required=True)
    preflight.add_argument("--expected-raw-sha256", required=True)
    preflight.add_argument("--configmap-object", type=Path, required=True)
    preflight.add_argument("--mounted-lock", type=Path, required=True)
    preflight.add_argument("--container-lock", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "extract-geometry":
        payload = extract_geometry(args.source_lock, args.output)
    elif args.command == "consumer-fingerprint":
        payload = {"consumer_ast_sha256": aggregate_consumer_ast_sha(args.runner)}
    elif args.command == "validate":
        payload = validate_run_lock(
            args.repo.resolve(), args.lock, args.schema, args.canonical_geometry,
            args.consumer_manifest, args.allowlist,
            expected_run_uid=args.expected_run_uid,
            expected_root=args.expected_root,
            expected_commit=args.expected_commit,
        )
        if args.output is not None:
            atomic_json(args.output, payload)
    else:
        payload = config_preflight(
            args.repo.resolve(), args.lock, args.schema, args.canonical_geometry,
            args.consumer_manifest, args.allowlist,
            expected_run_uid=args.expected_run_uid,
            expected_root=args.expected_root,
            expected_commit=args.expected_commit,
            expected_raw_sha256=args.expected_raw_sha256,
            configmap_object=args.configmap_object,
            mounted_lock=args.mounted_lock,
            container_lock=args.container_lock,
        )
        atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
