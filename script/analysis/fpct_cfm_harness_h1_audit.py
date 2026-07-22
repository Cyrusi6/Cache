#!/usr/bin/env python3
"""Config-only producer/consumer audit for FPCT confirmatory harness H1."""

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
from typing import Any

import yaml


MANIFEST_SHA = "5e04fe7ffa2f5df6ed0159b67be88a1c9547f0cd11886d461166e1ce7ba455e4"
R2M_LOCK_SHA = "db67428e6778c07a4991d9a9675ace6883de3a785c1f317b15b2aac7bfa0deff"
R2M_FINAL_SHA = "6c1ce4322cf7773e57abb6a0e1604c75947fb3de2927ffd4f8420aea60a9b306"
R2M_IMAGE = "sha256:0ea40657a38dbe8778ef492474f28ef2360156f4c9b0f8287e2bb0988d695851"
R2M_SCIENCE = "80fb295542ad298fae4cddb1273517b401bbcd17"
AUDIT_UID = "h1-config-only-not-an-execution-uid"
AUDIT_ROOT = "H1_CONFIG_ONLY_NO_EXECUTION_ROOT"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise AuditError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def strict_loads(text: str) -> Any:
    def constant(value: str) -> Any:
        raise AuditError(f"nonfinite JSON constant: {value}")
    return json.loads(text, object_pairs_hook=_strict_pairs, parse_constant=constant)


def strict_load(path: Path) -> Any:
    return strict_loads(path.read_text(encoding="utf-8"))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_schema(value: Any, schema: dict[str, Any], pointer: str = "") -> None:
    expected = schema.get("type")
    if expected == "object" and not isinstance(value, dict):
        raise AuditError(f"{pointer or '/'} expected object")
    if expected == "array" and not isinstance(value, list):
        raise AuditError(f"{pointer or '/'} expected array")
    if expected == "string" and not isinstance(value, str):
        raise AuditError(f"{pointer or '/'} expected string")
    if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        raise AuditError(f"{pointer or '/'} expected integer")
    if expected == "number" and not _is_number(value):
        raise AuditError(f"{pointer or '/'} expected finite number")
    if "const" in schema and value != schema["const"]:
        raise AuditError(f"{pointer or '/'} const mismatch")
    if "enum" in schema and value not in schema["enum"]:
        raise AuditError(f"{pointer or '/'} enum mismatch")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise AuditError(f"{pointer or '/'} string too short")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise AuditError(f"{pointer or '/'} pattern mismatch")
    if _is_number(value):
        if "minimum" in schema and value < schema["minimum"]:
            raise AuditError(f"{pointer or '/'} below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise AuditError(f"{pointer or '/'} above maximum")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise AuditError(f"{pointer or '/'} too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise AuditError(f"{pointer or '/'} too many items")
        if "items" in schema:
            for index, item in enumerate(value):
                validate_schema(item, schema["items"], f"{pointer}/{index}")
    if isinstance(value, dict):
        required = set(schema.get("required", []))
        missing = sorted(required - set(value))
        if missing:
            raise AuditError(f"{pointer or '/'} missing {missing}")
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            raise AuditError(f"{pointer or '/'} too few properties")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        if additional is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise AuditError(f"{pointer or '/'} extra properties {extra}")
        for key, item in value.items():
            child = properties.get(key)
            if child is None and isinstance(additional, dict):
                child = additional
            if child is not None:
                validate_schema(item, child, f"{pointer}/{key}")


def _constant_string(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


class PointerVisitor(ast.NodeVisitor):
    def __init__(self, roots: set[str]) -> None:
        self.aliases: dict[str, tuple[str, ...]] = {name: () for name in roots}
        self.sets: dict[str, set[str]] = {}
        self.strings: dict[str, str] = {}
        self.pointers: set[str] = set()
        self.unresolved: list[str] = []

    def key(self, node: ast.AST) -> str | None:
        direct = _constant_string(node)
        if direct is not None:
            return direct
        if isinstance(node, ast.Name):
            return self.strings.get(node.id)
        return None

    def path(self, node: ast.AST) -> tuple[str, ...] | None:
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id)
        if isinstance(node, ast.Subscript):
            parent = self.path(node.value)
            key = self.key(node.slice)
            if parent is not None and key is None:
                self.unresolved.append(ast.unparse(node))
            return parent + (key,) if parent is not None and key is not None else None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
            parent = self.path(node.func.value)
            key = self.key(node.args[0])
            if parent is not None and key is None:
                self.unresolved.append(ast.unparse(node))
            return parent + (key,) if parent is not None and key is not None else None
        return None

    def add(self, path: tuple[str, ...] | None) -> None:
        if path:
            self.pointers.add("/" + "/".join(path))

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            path = self.path(node.value)
            if path is not None:
                self.aliases[name] = path
            if isinstance(node.value, (ast.Set, ast.Tuple, ast.List)):
                values = {_constant_string(item) for item in node.value.elts}
                if None not in values:
                    self.sets[name] = {str(item) for item in values}
            scalar = _constant_string(node.value)
            if scalar is not None:
                self.strings[name] = scalar
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.add(self.path(node))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.add(self.path(node))
        function_name = None
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            function_name = node.func.attr
        if function_name == "json_pointer" and len(node.args) >= 2:
            pointer = self.key(node.args[1])
            if pointer is not None and pointer.startswith("/"):
                self.pointers.add(pointer)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "issubset" and node.args:
            if isinstance(node.func.value, ast.Name) and node.func.value.id in self.sets:
                target = self.path(node.args[0])
                if target == ():
                    for key in self.sets[node.func.value.id]:
                        self.add((key,))
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)):
            key = _constant_string(node.left)
            parent = self.path(node.comparators[0])
            if key is not None and parent is not None:
                self.add(parent + (key,))
        self.generic_visit(node)


def function_node(path: Path, symbol: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol:
            return node
    raise AuditError(f"missing function {symbol} in {path}")


def discover_function(path: Path, symbol: str, roots: set[str]) -> list[str]:
    return discover_function_detail(path, symbol, roots)["pointers"]


def discover_function_detail(path: Path, symbol: str, roots: set[str]) -> dict[str, Any]:
    node = function_node(path, symbol)
    visitor = PointerVisitor(roots)
    visitor.visit(node)
    return {"pointers": sorted(visitor.pointers), "unresolved": sorted(set(visitor.unresolved))}


def local_string_set(path: Path, symbol: str, variable: str) -> list[str]:
    node = function_node(path, symbol)
    for child in ast.walk(node):
        if isinstance(child, ast.Assign) and any(isinstance(target, ast.Name) and target.id == variable for target in child.targets):
            return sorted(ast.literal_eval(child.value))
    raise AuditError(f"missing local set {variable}: {path}:{symbol}")


def assigned_constant(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AuditError(f"missing constant {name}: {path}")


def dict_keys_in_function(path: Path, symbol: str) -> set[str]:
    node = function_node(path, symbol)
    keys: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Dict):
            for key in child.keys:
                if key is None:
                    continue
                value = _constant_string(key)
                if value is not None:
                    keys.add(str(value))
    return keys


def argparse_contract(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    subcommands: set[str] = set()
    arguments: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "add_parser" and node.args:
            name = _constant_string(node.args[0])
            if name is not None:
                subcommands.add(name)
        if node.func.attr != "add_argument":
            continue
        flags = [value for item in node.args if (value := _constant_string(item)) is not None]
        keywords: dict[str, Any] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            try:
                keywords[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                keywords[keyword.arg] = ast.unparse(keyword.value)
        arguments.append({"flags": flags, "keywords": keywords})
    arguments.sort(key=lambda row: tuple(row["flags"]))
    return {"subcommands": sorted(subcommands), "arguments": arguments}


def pointer_get(payload: Any, pointer: str) -> Any:
    value = payload
    for part in pointer.strip("/").split("/") if pointer != "/" else []:
        value = value[part]
    return value


def pointer_parent(payload: Any, pointer: str) -> tuple[Any, str]:
    parts = pointer.strip("/").split("/")
    parent = payload
    for part in parts[:-1]:
        parent = parent[part]
    return parent, parts[-1]


def compile_candidate(repo: Path) -> dict[str, Any]:
    r2m = strict_load(repo / "recipe/eval_recipe/fpct_gpu_r2m/immutable_v1_run_lock.json")
    manifest = strict_load(repo / "recipe/eval_recipe/fpct_confirmatory/confirmatory_manifest.json")
    arm_order = manifest["formal_training"]["arm_order"]
    candidate = {
        "schema_version": 1,
        "protocol_id": "fpct_cfm_harness_h1_candidate_lock_projection_v1",
        "status": "H1_CONFIG_ONLY_CANDIDATE",
        "run_uid": AUDIT_UID,
        "scientific_code_commit": r2m["scientific_code_commit"],
        "image": {
            "reference": r2m["image"]["reference"],
            "source_tree_sha256": r2m["image"]["source_tree_sha256"],
        },
        "manifest_sha256": MANIFEST_SHA,
        "nested_confirmatory_manifest_sha256": r2m["normative"]["confirmatory_manifest"],
        "source_lock_sha256": sha256_file(repo / "recipe/eval_recipe/fpct_gpu_r2m/immutable_v1_run_lock.json"),
        "shared_run_root": AUDIT_ROOT,
        "assets": {
            "receiver_sha256": r2m["assets"]["receiver"]["model_sha256"],
            "sender_sha256": r2m["assets"]["sender"]["model_sha256"],
            "sidecar_sha256": r2m["assets"]["training_alignment_sidecar_2048"]["sha256"],
        },
        "training": {
            "seeds": manifest["formal_training"]["seeds"],
            "arm_order": arm_order,
            "examples": manifest["formal_training"]["training_examples"],
            "optimizer_steps": manifest["formal_training"]["optimizer_steps"],
            "world_size": manifest["formal_training"]["processes"],
        },
        "kubernetes": {
            "namespace": r2m["kubernetes"]["namespace"],
            "config_map": "fpct-h1-dry-run-not-applied",
            "node_name": r2m["kubernetes"]["node_name"],
            "gpu_per_seed_pod": r2m["kubernetes"]["gpu_per_seed_pod"],
        },
        "scientific_output": False,
        "training_authorized": False,
    }
    return candidate


def validate_candidate(candidate: dict[str, Any], repo: Path) -> None:
    schema = strict_load(repo / "recipe/eval_recipe/fpct_cfm_harness_h1/schemas/execution_lock.schema.json")
    validate_schema(candidate, schema)
    actual_manifest = sha256_file(repo / "recipe/eval_recipe/fpct_confirmatory/confirmatory_manifest.json")
    if {candidate["manifest_sha256"], candidate["nested_confirmatory_manifest_sha256"], actual_manifest} != {MANIFEST_SHA}:
        raise AuditError("manifest top-level/nested/raw mismatch")
    if candidate["source_lock_sha256"] != R2M_LOCK_SHA:
        raise AuditError("source lock identity mismatch")
    if candidate["scientific_code_commit"] != R2M_SCIENCE:
        raise AuditError("scientific commit mismatch")
    if candidate["image"]["reference"].split("@", 1)[-1] != R2M_IMAGE:
        raise AuditError("image digest mismatch")
    if candidate["run_uid"] != AUDIT_UID or candidate["shared_run_root"] != AUDIT_ROOT:
        raise AuditError("H1 sentinel identity mismatch")
    r2m = strict_load(repo / "recipe/eval_recipe/fpct_gpu_r2m/immutable_v1_run_lock.json")
    expected_assets = {
        "receiver_sha256": r2m["assets"]["receiver"]["model_sha256"],
        "sender_sha256": r2m["assets"]["sender"]["model_sha256"],
        "sidecar_sha256": r2m["assets"]["training_alignment_sidecar_2048"]["sha256"],
    }
    if candidate["assets"] != expected_assets:
        raise AuditError("model/sidecar asset identity mismatch")
    manifest = strict_load(repo / "recipe/eval_recipe/fpct_confirmatory/confirmatory_manifest.json")
    if candidate["training"]["arm_order"] != manifest["formal_training"]["arm_order"]:
        raise AuditError("arm order mismatch")
    if candidate["training"]["seeds"] != manifest["formal_training"]["seeds"]:
        raise AuditError("seed mismatch")


def validate_receipt(receipt: dict[str, Any], *, lock_sha: str) -> None:
    if receipt["run_uid"] != AUDIT_UID or receipt["lock_sha256"] != lock_sha:
        raise AuditError("receipt lock identity mismatch")
    if receipt["image_digest"] != R2M_IMAGE or receipt["artifact_sha256"] != R2M_FINAL_SHA:
        raise AuditError("receipt prerequisite identity mismatch")
    if receipt["classification"] != "H1_CONFIG_ONLY_NO_AUTHORITY":
        raise AuditError("receipt classification mismatch")
    if receipt["scientific_output"] or receipt["training_authorized"]:
        raise AuditError("receipt authority boundary violation")


def validate_job_projection(job: dict[str, Any], candidate: dict[str, Any], kind: str) -> None:
    text = json.dumps(job, sort_keys=True)
    if re.search(r"__[A-Z0-9_]+__", text):
        raise AuditError("unresolved K8s placeholder")
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    annotations = job["metadata"].get("annotations", {})
    if annotations.get("fpct.cache/scientific-output") != "false" or annotations.get("fpct.cache/training-authorized") != "false":
        raise AuditError("K8s H1 authority annotations missing")
    if container["image"] != candidate["image"]["reference"]:
        raise AuditError("K8s image mismatch")
    if pod["nodeName"] != candidate["kubernetes"]["node_name"]:
        raise AuditError("K8s node mismatch")
    if pod["volumes"][0]["configMap"]["name"] != candidate["kubernetes"]["config_map"]:
        raise AuditError("K8s ConfigMap mismatch")
    expected_gpu = 2 if kind == "formal" else 0
    actual_gpu = int(container["resources"]["limits"].get("nvidia.com/gpu", 0))
    if actual_gpu != expected_gpu:
        raise AuditError("K8s GPU count mismatch")


def compile_outputs(repo: Path, output: Path) -> dict[str, Any]:
    candidate = compile_candidate(repo)
    validate_candidate(candidate, repo)
    candidate_path = output / "candidate_lock_projection.json"
    atomic_json(candidate_path, candidate)
    raw = candidate_path.read_text(encoding="utf-8")
    lock_sha = sha256_file(candidate_path)
    authority_annotations = {
        "fpct.cache/scientific-output": "false",
        "fpct.cache/training-authorized": "false",
    }
    configmap = {
        "apiVersion": "v1", "kind": "ConfigMap",
        "metadata": {"name": candidate["kubernetes"]["config_map"], "namespace": candidate["kubernetes"]["namespace"], "labels": {"project": "fpct", "study": "h1-config-only"}, "annotations": authority_annotations},
        "immutable": True, "data": {"candidate_lock_projection.json": raw},
    }
    configmap_path = output / "candidate_configmap.json"
    atomic_json(configmap_path, configmap)
    base_container = {
        "name": "h1-dry-run", "image": candidate["image"]["reference"],
        "command": ["/opt/conda/bin/python3.11", "-I", "/opt/fpct/script/runtime/fpct_bootstrap.py"],
        "env": [{"name": "CUDA_VISIBLE_DEVICES", "value": ""}, {"name": "HF_HUB_OFFLINE", "value": "1"}, {"name": "TRANSFORMERS_OFFLINE", "value": "1"}],
        "volumeMounts": [{"name": "lock", "mountPath": "/opt/fpct-h1", "readOnly": True}],
    }
    jobs = {}
    for kind, args, gpu in (
        ("smoke", ["--repo-root", "/opt/fpct", "--target", "/opt/fpct/script/experiment/fpct_confirmatory_runner.py", "--", "probe", "--run-lock", "/opt/fpct-h1/candidate_lock_projection.json"], 0),
        ("formal", ["--repo-root", "/opt/fpct", "--target", "/opt/fpct/script/experiment/fpct_confirmatory_runner.py", "--", "train-triplet", "--run-lock", "/opt/fpct-h1/candidate_lock_projection.json", "--seed", "45", "--output-root", "/h1-must-not-exist"], 2),
    ):
        container = copy.deepcopy(base_container)
        container["args"] = args
        container["resources"] = {"requests": {"cpu": "1", "memory": "1Gi"}, "limits": {}}
        if gpu:
            container["resources"]["requests"]["nvidia.com/gpu"] = str(gpu)
            container["resources"]["limits"]["nvidia.com/gpu"] = str(gpu)
        job = {
            "apiVersion": "batch/v1", "kind": "Job",
            "metadata": {"name": f"fpct-h1-{kind}-dry-run", "namespace": candidate["kubernetes"]["namespace"], "labels": {"project": "fpct", "study": "h1-config-only"}, "annotations": authority_annotations},
            "spec": {"backoffLimit": 0, "template": {"metadata": {"labels": {"project": "fpct", "study": "h1-config-only"}, "annotations": authority_annotations}, "spec": {"restartPolicy": "Never", "nodeName": candidate["kubernetes"]["node_name"], "containers": [container], "volumes": [{"name": "lock", "configMap": {"name": candidate["kubernetes"]["config_map"]}}]}}},
        }
        path = output / f"candidate_{kind}_job.json"
        atomic_json(path, job)
        validate_job_projection(job, candidate, kind)
        jobs[kind] = {"path": str(path), "sha256": sha256_file(path)}
    receipt = {
        "schema_version": 1, "protocol_id": "fpct_h1_prerequisite_receipt_v1",
        "run_uid": AUDIT_UID, "lock_sha256": lock_sha, "image_digest": R2M_IMAGE,
        "artifact_sha256": R2M_FINAL_SHA, "classification": "H1_CONFIG_ONLY_NO_AUTHORITY",
        "scientific_output": False, "training_authorized": False,
    }
    receipt_path = output / "candidate_authorization_receipt.json"
    atomic_json(receipt_path, receipt)
    validate_schema(receipt, strict_load(repo / "recipe/eval_recipe/fpct_cfm_harness_h1/schemas/prerequisite_receipt.schema.json"))
    validate_receipt(receipt, lock_sha=lock_sha)
    return {
        "candidate_lock_path": str(candidate_path), "candidate_lock_sha256": lock_sha,
        "configmap_path": str(configmap_path), "configmap_sha256": sha256_file(configmap_path),
        "jobs": jobs, "receipt_path": str(receipt_path), "receipt_sha256": sha256_file(receipt_path),
    }


def schema_examples(candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    one = "1" * 64
    two = "2" * 64
    image = R2M_IMAGE
    return {
        "execution_lock.schema.json": candidate,
        "prerequisite_receipt.schema.json": {
            "schema_version": 1, "protocol_id": "fpct_h1_prerequisite_receipt_v1",
            "run_uid": AUDIT_UID, "lock_sha256": one, "image_digest": image,
            "artifact_sha256": two, "classification": "H1_CONFIG_ONLY",
            "scientific_output": False, "training_authorized": False,
        },
        "smoke_result.schema.json": {
            "schema_version": 1, "protocol_id": "fpct_h1_smoke_result_v1",
            "run_uid": AUDIT_UID, "lock_sha256": one, "image_digest": image,
            "prerequisite_sha256": two, "seed": 104729,
            "arms": ["c_pre", "c_post", "f"], "status": "GO",
        },
        "training_config.schema.json": {
            "schema_version": 1, "run_uid": AUDIT_UID, "seed": 45, "arm": "c_pre",
            "examples": 2048, "optimizer_steps": 64, "world_size": 2,
            "per_device_batch_size": 1, "gradient_accumulation_steps": 16,
            "learning_rate": 0.0001, "weight_decay": 0.01, "max_length": 1024,
            "precision": "bf16", "sidecar_sha256": one,
        },
        "arm_result.schema.json": {
            "schema_version": 1, "run_uid": AUDIT_UID, "lock_sha256": one,
            "image_digest": image, "seed": 45, "arm": "c_pre",
            "config_sha256": one, "checkpoint_sha256": two,
            "optimizer_steps": 64, "status": "COMPLETE",
        },
        "triplet_manifest.schema.json": {
            "schema_version": 1, "run_uid": AUDIT_UID, "lock_sha256": one,
            "image_digest": image, "seed": 45,
            "arm_order": ["c_pre", "c_post", "f"],
            "arm_result_sha256": {"c_pre": one, "c_post": one, "f": one},
            "status": "COMPLETE",
        },
        "controller_state.schema.json": {
            "schema_version": 1, "run_uid": AUDIT_UID, "lock_sha256": one,
            "stage": "RUN_LOCKED", "completed_triplets": [],
            "model_selection_release_count": 0, "held_out_release_count": 0,
        },
        "formal_completion.schema.json": {
            "schema_version": 1, "run_uid": AUDIT_UID, "lock_sha256": one,
            "image_digest": image,
            "triplet_sha256": {str(seed): one for seed in range(45, 57)},
            "status": "COMPLETE",
        },
        "release_receipt.schema.json": {
            "schema_version": 1, "run_uid": AUDIT_UID, "lock_sha256": one,
            "release": "model-selection", "release_count": 1,
            "input_artifact_sha256": two,
        },
        "terminal_result.schema.json": {
            "schema_version": 1, "run_uid": AUDIT_UID, "lock_sha256": one,
            "held_out_sha256": two, "statistics_code_sha256": one,
            "classification": "INCONCLUSIVE",
        },
    }


def schema_mutation_coverage(candidate: dict[str, Any], repo: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    examples = schema_examples(candidate)
    schema_root = repo / "recipe/eval_recipe/fpct_cfm_harness_h1/schemas"
    for schema_name, example in examples.items():
        schema = strict_load(schema_root / schema_name)
        validate_schema(example, schema)
        for key in schema["required"]:
            for mutation in ("delete", "null", "wrong_type"):
                value = copy.deepcopy(example)
                if mutation == "delete":
                    del value[key]
                elif mutation == "null":
                    value[key] = None
                else:
                    current = value[key]
                    if isinstance(current, bool): value[key] = "wrong"
                    elif isinstance(current, str): value[key] = []
                    elif isinstance(current, int): value[key] = "wrong"
                    elif isinstance(current, list): value[key] = {}
                    elif isinstance(current, dict): value[key] = []
                    else: value[key] = "wrong"
                try:
                    validate_schema(value, schema); failed, reason = False, ""
                except Exception as error:
                    failed, reason = True, str(error)
                rows.append({"artifact_schema": schema_name, "pointer": f"/{key}", "mutation": mutation, "fail_closed": failed, "reason": reason})
        value = copy.deepcopy(example); value["unexpected"] = 1
        try:
            validate_schema(value, schema); failed, reason = False, ""
        except Exception as error:
            failed, reason = True, str(error)
        rows.append({"artifact_schema": schema_name, "pointer": "/", "mutation": "extra_field", "fail_closed": failed, "reason": reason})
    return rows


def mutation_coverage(candidate: dict[str, Any], repo: Path) -> list[dict[str, Any]]:
    pointers = [
        "/run_uid", "/scientific_code_commit", "/image/reference", "/image/source_tree_sha256",
        "/manifest_sha256", "/nested_confirmatory_manifest_sha256", "/source_lock_sha256",
        "/shared_run_root", "/assets/receiver_sha256", "/assets/sender_sha256", "/assets/sidecar_sha256",
        "/training/seeds", "/training/arm_order", "/training/examples", "/training/optimizer_steps", "/training/world_size",
        "/kubernetes/namespace", "/kubernetes/config_map", "/kubernetes/node_name", "/kubernetes/gpu_per_seed_pod",
        "/scientific_output", "/training_authorized",
    ]
    rows: list[dict[str, Any]] = []
    for pointer in pointers:
        for mutation in ("delete", "null", "empty", "wrong_type"):
            value = copy.deepcopy(candidate)
            parent, key = pointer_parent(value, pointer)
            if mutation == "delete":
                del parent[key]
            elif mutation == "null":
                parent[key] = None
            elif mutation == "empty":
                current = parent[key]
                parent[key] = [] if isinstance(current, list) else {} if isinstance(current, dict) else ""
            else:
                current = parent[key]
                parent[key] = True if isinstance(current, int) and not isinstance(current, bool) else ["wrong"]
            failed = False
            reason = ""
            try:
                validate_candidate(value, repo)
            except Exception as error:
                failed, reason = True, str(error)
            rows.append({"pointer": pointer, "mutation": mutation, "fail_closed": failed, "reason": reason})
    special = {
        "stale_uid": ("/run_uid", "stale"), "stale_root": ("/shared_run_root", "/tmp/stale"),
        "stale_commit": ("/scientific_code_commit", "0" * 40),
        "stale_image": ("/image/reference", "x@sha256:" + "0" * 64),
        "manifest_mismatch": ("/nested_confirmatory_manifest_sha256", "0" * 64),
        "sidecar_mismatch": ("/assets/sidecar_sha256", "0" * 64),
        "model_mismatch": ("/assets/receiver_sha256", "0" * 64),
        "wrong_seed": ("/training/seeds", list(range(44, 56))),
        "wrong_arm_order": ("/training/arm_order", {str(seed): ["f", "c_post", "c_pre"] for seed in range(45, 57)}),
        "wrong_world_size": ("/training/world_size", 1),
        "wrong_gpu_count": ("/kubernetes/gpu_per_seed_pod", 1),
    }
    for name, (pointer, replacement) in special.items():
        value = copy.deepcopy(candidate); parent, key = pointer_parent(value, pointer); parent[key] = replacement
        try:
            validate_candidate(value, repo); failed, reason = False, ""
        except Exception as error:
            failed, reason = True, str(error)
        rows.append({"pointer": pointer, "mutation": name, "fail_closed": failed, "reason": reason})
    value = copy.deepcopy(candidate); value["extra"] = 1
    try:
        validate_candidate(value, repo); failed, reason = False, ""
    except Exception as error:
        failed, reason = True, str(error)
    rows.append({"pointer": "/", "mutation": "extra_field", "fail_closed": failed, "reason": reason})
    duplicate = '{"schema_version":1,"schema_version":1}'
    try:
        strict_loads(duplicate); failed, reason = False, ""
    except Exception as error:
        failed, reason = True, str(error)
    rows.append({"pointer": "/schema_version", "mutation": "duplicate_key", "fail_closed": failed, "reason": reason})
    rows.extend(schema_mutation_coverage(candidate, repo))
    lock_sha = hashlib.sha256((json.dumps(candidate, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()).hexdigest()
    receipt = schema_examples(candidate)["prerequisite_receipt.schema.json"]
    receipt["lock_sha256"] = lock_sha
    receipt["artifact_sha256"] = R2M_FINAL_SHA
    receipt["classification"] = "H1_CONFIG_ONLY_NO_AUTHORITY"
    for name, key, replacement in (
        ("wrong_prerequisite_sha", "artifact_sha256", "0" * 64),
        ("wrong_prerequisite_classification", "classification", "R2M_IMMUTABLE_GO"),
        ("wrong_receipt_lock_sha", "lock_sha256", "0" * 64),
    ):
        value = copy.deepcopy(receipt); value[key] = replacement
        try:
            validate_receipt(value, lock_sha=lock_sha); failed, reason = False, ""
        except Exception as error:
            failed, reason = True, str(error)
        rows.append({"artifact_schema": "prerequisite_receipt.schema.json", "pointer": f"/{key}", "mutation": name, "fail_closed": failed, "reason": reason})
    triplet = schema_examples(candidate)["triplet_manifest.schema.json"]
    del triplet["arm_result_sha256"]["c_post"]
    try:
        validate_schema(triplet, strict_load(repo / "recipe/eval_recipe/fpct_cfm_harness_h1/schemas/triplet_manifest.schema.json")); failed, reason = False, ""
    except Exception as error:
        failed, reason = True, str(error)
    rows.append({"artifact_schema": "triplet_manifest.schema.json", "pointer": "/arm_result_sha256/c_post", "mutation": "selective_arm_retry_or_missing_arm", "fail_closed": failed, "reason": reason})
    return rows


def stage_details(
    repo: Path,
    discovery: dict[str, Any],
    yaml_results: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    runner = "script/experiment/fpct_confirmatory_runner.py"
    controller = "script/experiment/fpct_gpu_r2_controller.py"
    statistics = "script/analysis/fpct_confirmatory_statistics.py"
    entries = [
        ("manifest", "recipe/eval_recipe/fpct_confirmatory/confirmatory_manifest.json", [], "confirmatory_manifest", [], "none"),
        ("execution_lock", "deterministic:H1 compiler", ["confirmatory_manifest", "source_blobs", "asset_hashes"], "execution_lock.schema.json", ["write_config_artifact"], "config_only"),
        ("config_preflight", "script/analysis/fpct_cfm_harness_h1_audit.py", ["execution_lock"], "prerequisite_receipt.schema.json", ["write_hash_receipt"], "no_scientific_output"),
        ("immutable_final", "script/analysis/fpct_gpu_r2m_finalize.py", ["original_result", "semantic_result", "active_aggregate"], "historical_r2m_final", ["historical_read_only"], "historical_only"),
        ("matched_smoke", runner, ["execution_lock", "immutable_final"], "smoke_result.schema.json", ["training_subprocess"], "blocked_in_h1"),
        ("formal_training", runner, ["execution_lock", "smoke_result"], "triplet_manifest.schema.json", ["training_subprocess", "checkpoint_write"], "blocked_in_h1"),
        ("formal_completion", controller, ["12_triplet_manifests"], "formal_completion.schema.json", ["write_completion_receipt"], "blocked_in_h1"),
        ("model_selection", controller, ["formal_completion"], "release_receipt.schema.json", ["performance_release"], "blocked_in_h1"),
        ("held_out", controller, ["model_selection_complete"], "release_receipt.schema.json", ["held_out_release"], "blocked_in_h1"),
        ("statistics", statistics, ["held_out_cells"], "terminal_result.schema.json", ["performance_analysis"], "blocked_in_h1"),
        ("terminal_result", statistics, ["statistics"], "terminal_result.schema.json", ["write_report"], "blocked_in_h1"),
    ]
    stage_pointer_keys = {
        "execution_lock": "candidate_compiler",
        "config_preflight": "candidate_compiler",
        "immutable_final": "r2m_finalizer",
        "matched_smoke": "matched_smoke",
        "formal_training": "formal_triplet",
        "formal_completion": "r2_controller",
        "model_selection": "r2_controller",
        "held_out": "r2_controller",
        "statistics": "statistics",
        "terminal_result": "statistics",
    }
    cli_sources = {
        "execution_lock": repo / "script/analysis/fpct_cfm_harness_h1_audit.py",
        "config_preflight": repo / "script/analysis/fpct_cfm_harness_h1_audit.py",
        "immutable_final": repo / "script/analysis/fpct_gpu_r2m_finalize.py",
        "matched_smoke": repo / runner,
        "formal_training": repo / runner,
        "formal_completion": repo / controller,
        "model_selection": repo / controller,
        "held_out": repo / controller,
        "statistics": repo / statistics,
        "terminal_result": repo / statistics,
    }
    yaml_by_stage = {
        "matched_smoke": "matched_smoke_immutable_v1_job.yaml",
        "formal_training": "seed_triplet_immutable_v1_job.yaml",
    }
    details = []
    for stage, producer, inputs, schema, side_effects, authorization in entries:
        path = repo / producer
        producer_sha = sha256_file(path) if path.is_file() else None
        cli_path = cli_sources.get(stage)
        yaml_name = yaml_by_stage.get(stage)
        details.append({
            "stage": stage, "producer": producer, "producer_sha256": producer_sha,
            "input_artifacts": inputs,
            "discovered_json_pointers": discovery.get(stage_pointer_keys.get(stage, stage), []),
            "cli_env_path_volume_image_identity_required": stage not in {"manifest", "terminal_result"},
            "cli_contract": argparse_contract(cli_path) if cli_path else None,
            "k8s_contract": yaml_results.get(yaml_name) if yaml_name else None,
            "identity_contract": {
                "run_uid": candidate["run_uid"],
                "shared_run_root": candidate["shared_run_root"],
                "image_reference": candidate["image"]["reference"],
                "lock_sha256": hashlib.sha256(canonical_bytes(candidate)).hexdigest(),
                "candidate_only": True,
            },
            "output_schema": schema, "prerequisites": inputs,
            "allowed_side_effects": side_effects,
            "forbidden_side_effects_in_h1": ["model_load", "dataset_load", "cuda", "optimizer", "checkpoint", "accuracy_read"],
            "next_stage_authorization": authorization,
        })
    return details


def yaml_contract(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    strings: list[str] = []
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values(): walk(item)
        elif isinstance(value, list):
            for item in value: walk(item)
        elif isinstance(value, str): strings.append(value)
    walk(payload)
    text = "\n".join(strings)
    pointers: set[str] = set()
    for match in re.finditer(r'\b[A-Za-z_]\w*((?:\["[^"]+"\])+)', text):
        keys = re.findall(r'\["([^"]+)"\]', match.group(1))
        if keys:
            pointers.add("/" + "/".join(keys))
    placeholders = sorted(set(re.findall(r"__[A-Z0-9_]+__", text)))
    pod = payload.get("spec", {}).get("template", {}).get("spec", {})
    containers = pod.get("containers", [])
    container = containers[0] if containers else {}
    return {
        "pointers": sorted(pointers),
        "shell_guards": sorted(line.strip() for line in text.splitlines() if "assert " in line),
        "placeholders": placeholders,
        "sha256": sha256_file(path),
        "metadata": {
            "name": payload.get("metadata", {}).get("name"),
            "namespace": payload.get("metadata", {}).get("namespace"),
            "labels": payload.get("metadata", {}).get("labels", {}),
        },
        "image": container.get("image"),
        "command": container.get("command", []),
        "args": container.get("args", []),
        "env": {row.get("name"): row.get("value") for row in container.get("env", [])},
        "resources": container.get("resources", {}),
        "volume_mounts": container.get("volumeMounts", []),
        "volumes": pod.get("volumes", []),
        "node_name": pod.get("nodeName"),
        "restart_policy": pod.get("restartPolicy"),
        "backoff_limit": payload.get("spec", {}).get("backoffLimit"),
    }


def graph_audit(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = {node["id"] for node in graph["nodes"]}
    edges = [tuple(edge) for edge in graph["edges"]]
    if any(left not in nodes or right not in nodes for left, right in edges):
        raise AuditError("stage graph references unknown node")
    visiting: set[str] = set(); visited: set[str] = set(); adjacency = {node: [] for node in nodes}
    for left, right in edges: adjacency[left].append(right)
    def visit(node: str) -> None:
        if node in visiting: raise AuditError("stage graph cycle")
        if node in visited: return
        visiting.add(node)
        for child in adjacency[node]: visit(child)
        visiting.remove(node); visited.add(node)
    for node in nodes: visit(node)
    return {"node_count": len(nodes), "edge_count": len(edges), "acyclic": True}


def audit(repo: Path, output: Path, image_result: Path | None = None, k8s_result: Path | None = None) -> dict[str, Any]:
    cfg = repo / "recipe/eval_recipe/fpct_cfm_harness_h1"
    compile_result = compile_outputs(repo, output)
    candidate = strict_load(Path(compile_result["candidate_lock_path"]))
    runner = repo / "script/experiment/fpct_confirmatory_runner.py"
    r2_controller = repo / "script/experiment/fpct_gpu_r2_controller.py"
    k8s_renderer = repo / "script/experiment/fpct_gpu_r2_k8s.py"
    source_details = {
        "candidate_compiler": discover_function_detail(Path(__file__), "compile_candidate", {"r2m", "manifest"}),
        "legacy_load_lock": discover_function_detail(runner, "load_lock", {"payload"}),
        "train_config_builder": discover_function_detail(runner, "training_config", {"lock"}),
        "matched_smoke_direct": discover_function_detail(runner, "matched_smoke", {"lock"}),
        "train_triplet_direct": discover_function_detail(runner, "train_triplet", {"lock"}),
        "r2_controller_initialize": discover_function_detail(r2_controller, "initialize", {"lock"}),
        "r2_controller_evidence": discover_function_detail(r2_controller, "_validate_evidence", {"payload"}),
        "r2_controller_triplet": discover_function_detail(r2_controller, "record_triplet", {"payload"}),
        "r2_k8s_renderer": discover_function_detail(k8s_renderer, "render", {"lock"}),
        "r2m_finalizer": discover_function_detail(
            repo / "script/analysis/fpct_gpu_r2m_finalize.py", "finalize", {"original", "semantic", "active"}
        ),
    }
    discovery = {name: row["pointers"] for name, row in source_details.items()}
    discovery["matched_smoke"] = sorted(set(discovery["legacy_load_lock"] + discovery["matched_smoke_direct"]))
    discovery["formal_triplet"] = sorted(set(discovery["legacy_load_lock"] + discovery["train_triplet_direct"]))
    discovery["r2_controller"] = sorted(set(
        discovery["r2_controller_initialize"]
        + discovery["r2_controller_evidence"]
        + discovery["r2_controller_triplet"]
    ))
    discovery["statistics_required_columns"] = local_string_set(
        repo / "script/analysis/fpct_confirmatory_statistics.py", "read_group_cells", "required"
    )
    trainer_text = (repo / "script/train/SFT_train.py").read_text(encoding="utf-8")
    discovery["trainer_config_get_keys"] = sorted(set(
        f"/{mapping}/{key}"
        for mapping, key in re.findall(r'\b(training_config|model_config)\.get\(["\']([^"\']+)["\']', trainer_text)
    ))
    discovery["trainer"] = discovery["trainer_config_get_keys"]
    discovery["statistics"] = [f"/columns/{name}" for name in discovery["statistics_required_columns"]]
    unresolved = {name: row["unresolved"] for name, row in source_details.items() if row["unresolved"]}
    if discovery["legacy_load_lock"] != ["/image", "/manifest_sha256", "/run_uid", "/scientific_code_commit"]:
        raise AuditError(f"legacy load_lock discovery mismatch: {discovery['legacy_load_lock']}")
    registry = strict_load(cfg / "consumer_registry.json")
    declared = {row["id"]: sorted(row["declared_required_inputs"]) for row in registry["consumers"]}
    closure = {}
    for key in (
        "legacy_load_lock", "train_config_builder", "matched_smoke", "formal_triplet",
        "r2_controller", "r2_k8s_renderer", "r2m_finalizer", "trainer", "statistics",
    ):
        actual = discovery[key]
        expected = declared.get(key, [])
        closure[key] = {"declared": expected, "discovered": actual, "equal": expected == actual}
    r2m = strict_load(repo / "recipe/eval_recipe/fpct_gpu_r2m/immutable_v1_run_lock.json")
    r2l = strict_load(repo / "recipe/eval_recipe/fpct_gpu_r2l/immutable_v1_run_lock.json")
    old_manifest = strict_load(repo / "recipe/eval_recipe/fpct_gpu_r2m/consumer_pointer_manifest.json")
    old_declared = json.dumps(old_manifest, sort_keys=True)
    reds = {
        "r2m_manifest_missing": "manifest_sha256" not in r2m,
        "r2l_geometry_missing": "resource_geometry" not in r2l,
        "old_consumer_manifest_incomplete": "manifest_sha256" not in old_declared,
        "actual_manifest_sha": sha256_file(repo / "recipe/eval_recipe/fpct_confirmatory/confirmatory_manifest.json") == MANIFEST_SHA,
        "last_training_lock_manifest": strict_load(repo / "recipe/eval_recipe/fpct_gpu_r2/fpct_gpu_r2j_run_lock.json").get("manifest_sha256") == MANIFEST_SHA,
    }
    arm_order = assigned_constant(runner, "ARM_ORDER")
    manifest_order = {int(seed): tuple(arms) for seed, arms in strict_load(repo / "recipe/eval_recipe/fpct_confirmatory/confirmatory_manifest.json")["formal_training"]["arm_order"].items()}
    arm_order_equal = arm_order == manifest_order
    output_bindings = {
        "r2m_finalizer": {"present": dict_keys_in_function(repo / "script/analysis/fpct_gpu_r2m_finalize.py", "finalize")},
        "matched_smoke": {"present": dict_keys_in_function(runner, "matched_smoke")},
        "formal_triplet": {"present": dict_keys_in_function(runner, "train_triplet")},
        "controller_state": {"present": dict_keys_in_function(r2_controller, "initialize")},
    }
    required_binding = {"run_uid", "lock_sha256", "image_digest", "prerequisite_sha256"}
    for row in output_bindings.values():
        semantic_present = set(row["present"])
        if "run_lock_sha256" in semantic_present:
            semantic_present.add("lock_sha256")
        row["required"] = required_binding
        row["semantic_present"] = semantic_present
        row["missing"] = required_binding - semantic_present
        row["complete"] = not row["missing"]
    graph = graph_audit(strict_load(cfg / "stage_graph.json"))
    schema_results = {}
    examples = schema_examples(candidate)
    for path in sorted((cfg / "schemas").glob("*.schema.json")):
        schema = strict_load(path)
        if schema.get("additionalProperties") is not False:
            raise AuditError(f"schema not strict: {path}")
        validate_schema(examples[path.name], schema)
        schema_results[path.name] = {"sha256": sha256_file(path), "strict_top_level": True}
    mutations = mutation_coverage(candidate, repo)
    mutation_ok = bool(mutations) and all(row["fail_closed"] for row in mutations)
    yaml_results = {
        path.name: yaml_contract(path)
        for path in sorted((repo / "recipe/k8s/fpct_gpu_r2m").glob("*.yaml"))
    }
    image_payload = strict_load(image_result) if image_result and image_result.is_file() else None
    k8s_payload = strict_load(k8s_result) if k8s_result and k8s_result.is_file() else None
    if image_payload:
        for name, passed in image_payload.get("controller_negative_checks", {}).items():
            mutations.append({"artifact_schema": "controller_state.schema.json", "pointer": "/stage", "mutation": name, "fail_closed": bool(passed), "reason": "exact-image controller simulation"})
    formal_job = strict_load(Path(compile_result["jobs"]["formal"]["path"]))
    placeholder_job = copy.deepcopy(formal_job)
    placeholder_job["spec"]["template"]["spec"]["containers"][0]["image"] = "__IMAGE_DIGEST__"
    try:
        validate_job_projection(placeholder_job, candidate, "formal"); failed, reason = False, ""
    except Exception as error:
        failed, reason = True, str(error)
    mutations.append({"artifact_schema": "k8s_critical_projection", "pointer": "/container/image", "mutation": "leftover_template_placeholder", "fail_closed": failed, "reason": reason})
    mutation_ok = bool(mutations) and all(row["fail_closed"] for row in mutations)
    image_changes = sorted(name for name, row in output_bindings.items() if not row["complete"])
    checks = {
        "historical_reds": all(reds.values()),
        "load_lock_discovery": True,
        "arm_order_exact": arm_order_equal,
        "graph_acyclic": graph["acyclic"],
        "schemas_strict": len(schema_results) == 10,
        "mutation_coverage": mutation_ok,
        "compiler_valid": True,
        "no_unannotated_dynamic_access": not unresolved,
        "training_config_matrix_complete": bool(image_payload and image_payload.get("training_configs_schema_validated") == 39),
        "controller_negative_tests": bool(image_payload and all(image_payload.get("controller_negative_checks", {}).values())),
        "image_dry_run_complete": bool(image_payload and image_payload.get("status") == "GO"),
        "k8s_server_dry_run_complete": bool(k8s_payload and k8s_payload.get("status") == "GO"),
        "k8s_authority_annotations": bool(k8s_payload and k8s_payload.get("authority_annotations_exact")),
        "exact_byte_identity": bool(image_payload and image_payload.get("exact_byte_all_equal")),
        "no_cluster_resources_created": bool(k8s_payload and k8s_payload.get("no_resources_created")),
    }
    audit_complete = all(checks.values())
    if not audit_complete:
        classification = "H1_AUDIT_BLOCKED"
    elif image_changes:
        classification = "H1_REQUIRES_NEW_IMAGE_QUALIFICATION"
    elif all(row["equal"] for row in closure.values()):
        classification = "H1_AUDIT_GO_NO_EXECUTION_AUTHORITY"
    else:
        classification = "H1_AUDIT_BLOCKED"
    stages = stage_details(repo, discovery, yaml_results, candidate)
    payload = {
        "schema_version": 1, "protocol_id": "fpct_cfm_harness_h1_audit_result_v1",
        "classification": classification, "status": "COMPLETE" if audit_complete else "BLOCKED",
        "checks": checks, "historical_red_fixtures": reds, "discovery": discovery,
        "unresolved_dynamic_accesses": unresolved,
        "declared_discovered_closure": closure, "arm_order_exact": arm_order_equal,
        "stage_graph": graph, "stage_details": stages, "schema_results": schema_results,
        "mutation_summary": {"row_count": len(mutations), "all_fail_closed": mutation_ok},
        "yaml_contracts": yaml_results,
        "image_internal_binding_gaps": {name: {key: sorted(value) if isinstance(value, set) else value for key, value in row.items()} for name, row in output_bindings.items()},
        "requires_image_changes": image_changes,
        "compiler": compile_result,
        "image_dry_run": image_payload,
        "k8s_server_dry_run": k8s_payload,
        "scientific_output": False, "training_authorized": False,
    }
    atomic_json(output / "discovered_consumer_registry.json", {"schema_version": 1, "contracts": source_details, "derived": discovery, "unresolved_dynamic_accesses": unresolved, "scientific_output": False, "training_authorized": False})
    atomic_json(output / "discovered_stage_graph.json", {"schema_version": 1, "stages": stages, "edges": strict_load(cfg / "stage_graph.json")["edges"], "scientific_output": False, "training_authorized": False})
    atomic_json(output / "mutation_coverage_matrix.json", {"schema_version": 1, "rows": mutations, "scientific_output": False, "training_authorized": False})
    atomic_json(output / "audit_result.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    compile_parser = sub.add_parser("compile")
    compile_parser.add_argument("--repo", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--repo", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    audit_parser.add_argument("--image-result", type=Path)
    audit_parser.add_argument("--k8s-result", type=Path)
    args = parser.parse_args()
    if args.command == "compile":
        payload = compile_outputs(args.repo.resolve(), args.output.resolve())
    else:
        payload = audit(args.repo.resolve(), args.output.resolve(), args.image_result, args.k8s_result)
    print(json.dumps(payload, indent=2, sort_keys=True, default=lambda value: sorted(value) if isinstance(value, set) else str(value)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
