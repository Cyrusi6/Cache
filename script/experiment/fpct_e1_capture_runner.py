#!/usr/bin/env python3
"""Sealed E0-design executor for FPCT-E1 endpoints and centered-lambda sweep.

The executor is deliberately split from the model backend.  ``prepare`` binds
the immutable E0 step-64/final projector artifacts, E1 data membership,
instrumentation hard gate, source/runtime provenance, and the two-stage shard
graph without importing torch, transformers, tokenizers, or model code.

Backends emit only operator-local columns.  This executor constructs every
``cpost_*`` field from the completed C_post capture artifact, verifies an exact
row-key bijection, and only then writes rows accepted by the offline audit.
Consequently an F/sweep backend cannot report its own baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence


SCHEMA_VERSION = 6
PROTOCOL_ID = "fpct_e1_e0_design_capture_v6_representation_preserving_streaming"
A4_PROTOCOL_ID = "fpct_e1_mechanism_audit_v6_representation_preserving_streaming"
INPUT_LOCK_SCHEMA_VERSION = 2
INPUT_LOCK_PROTOCOL_ID = "fpct_e1_e0_design_input_lock_v2_streaming"
INPUT_LOCK_STATUS = "GO_STREAMING_CPU_INPUT_LOCK_NO_MODEL_OUTPUT"
K8S_LOCK_PROTOCOL_ID = "fpct_e1_k8s_lock_bundle_v2_a4_streaming"
K8S_FINALIZED_LOCK_PROTOCOL_ID = "fpct_e1_k8s_finalized_lock_bundle_v2_a4_streaming"
K8S_LOCK_SCHEMA_VERSION = 2
CAPTURE_BACKEND_CONTRACT_VERSION = 3
CAPTURE_STAGING_PROTOCOL_ID = "fpct_e1_capture_staging_v1_immutable_chunks"
GOLD_RESPONSE_TEMPLATE = "The correct answer is {answer}."
CORRECTNESS_SEMANTICS = (
    "current operator deterministic E0-compatible generated-answer correctness "
    "repeated on rows; not teacher-forced token accuracy; C_post baseline is "
    "executor-owned"
)
CONSOLIDATED_GATE_ID = "fpct_e1_instrumentation_hard_gate_v1"
ALLOWED_SPLIT_ROLE = "e0_design"
STAGE_E1_2 = "e1_2_endpoints"
STAGE_E1_3 = "e1_3_centered_lambda"
PHASE_E1_2_BASELINES = "e1_2_cpost_baselines"
PHASE_E1_2_FACTORIZED = "e1_2_factorized_endpoints"
FINALIZED_E1_2 = "e1_2_finalized"
SEEDS = (2026072201, 2026072202, 2026072203)
TASKS = ("ai2-arc", "openbookqa", "mmlu-redux")
TASK_GROUP_COUNTS = {"ai2-arc": 128, "openbookqa": 70, "mmlu-redux": 128}
CHECKPOINT_ARMS = ("c_post_trained", "f_trained")
TRAINED_DIRECTORY = {"c_post_trained": "c_post", "f_trained": "f"}
CELL_MAP = {
    ("c_post_trained", "c_post"): "Y_CC",
    ("c_post_trained", "f"): "Y_CF",
    ("f_trained", "c_post"): "Y_FC",
    ("f_trained", "f"): "Y_FF",
}
CELL_SPEC = {cell: pair for pair, cell in CELL_MAP.items()}
BASELINE_CELL = {"Y_CF": "Y_CC", "Y_FF": "Y_FC"}
ENDPOINT_LAMBDAS = {"c_post": 0.0, "f": 1.0}
SWEEP_ADDITIONS = (0.0, 0.25, 0.5, 2.0)
FULL_LAMBDA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
EXPECTED_ENDPOINT_SHARDS = len(SEEDS) * 4 * len(TASKS)
EXPECTED_SWEEP_SHARDS = len(SEEDS) * len(CHECKPOINT_ARMS) * len(TASKS) * len(SWEEP_ADDITIONS)
EXPECTED_SHARD_COUNT = EXPECTED_ENDPOINT_SHARDS + EXPECTED_SWEEP_SHARDS
PARQUET_BATCH_ROWS = 4096

SPLIT_MANIFEST_RELATIVE = Path("recipe/eval_recipe/fpct_e1/e1_data_split_manifest.json")
E1_MANIFEST_RELATIVE = Path("recipe/eval_recipe/fpct_e1/e1_manifest.json")
STREAMING_CONTRACT_RELATIVE = Path("recipe/eval_recipe/fpct_e1/e1_streaming_contract.json")
STREAMING_SCHEMA_RELATIVE = Path("recipe/eval_recipe/fpct_e1/e1_streaming_schema.json")
STREAMING_SYNTHETIC_GATE_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_streaming_synthetic_gate.json"
)
MECHANISM_SCHEMA_RELATIVE = Path("recipe/eval_recipe/fpct_e1/e1_mechanism_schema.json")
E0_DEV_MANIFEST_RELATIVE = Path("recipe/eval_recipe/fpct_e0/exploratory_dev_manifest.json")
E0_CONFIG_INDEX_RELATIVE = Path("recipe/eval_recipe/fpct_e0/rendered/config_index.json")
E0_CONFIG_ROOT_RELATIVE = Path("recipe/eval_recipe/fpct_e0/rendered")
EXECUTOR_RELATIVE = Path("script/experiment/fpct_e1_capture_runner.py")
ANALYZER_RELATIVE = Path("script/analysis/fpct_e1_mechanism_audit.py")
K8S_CAPTURE_TEMPLATE_RELATIVE = Path("recipe/k8s/fpct_e1/e0_design_capture_job.yaml")
RUNTIME_PROBE_RENDERER_RELATIVE = Path(
    "script/experiment/fpct_e1_runtime_probe_renderer.py"
)

REQUIRED_GATE_CHECKS = (
    "formula_oracles",
    "instrumentation_on_off_equivalent",
    "synthetic_query_variance_positive",
    "multistep_accumulation_preserved",
    "invalid_probability_gradient_exact_zero",
    "no_nan_inf",
)
BASE_ROW_KEY_CANDIDATES = (
    "seed", "checkpoint_arm", "task", "sample_sha256", "content_group_sha256",
    "input_sha256", "alignment_sha256", "labels_sha256", "gold_response_sha256",
    "layer", "head", "query_head", "kv_head", "query_position", "target_position",
    "target_token_id", "parent_position", "candidate_count", "topology",
)
# This is the pre-A4 scientific identity.  In particular, row_ordinal,
# logical_row_id, endpoint_id and endpoint_row_id are deliberately absent:
# endpoint-specific transport identities may never alter the matched C_post
# baseline join.
ROW_TEMPLATE_COLUMNS = (
    "sample_sha256", "content_group_sha256", "input_sha256", "alignment_sha256",
    "labels_sha256", "gold_response_sha256", "layer", "query_head", "kv_head",
    "query_position", "target_position", "target_token_id", "parent_position",
    "candidate_count", "topology",
)
PLACEHOLDERS = (
    "__EXECUTION_SHA__", "__IMMUTABLE_IMAGE_DIGEST__",
    "__FROZEN_RUNTIME_BACKEND__", "__BACKEND_SOURCE__",
    "__BACKEND_SOURCE_SHA256__", "__RUNTIME_PROVENANCE__",
    "__RUNTIME_PROVENANCE_SHA256__", "__INSTRUMENTATION_GATE_SHA256__",
    "__GATE_HOST_ROOT__", "__RUNTIME_HOST_ROOT__", "__IMMUTABLE_ATTEMPT__",
    "__FINAL_TREE_SHA256__", "__STEP64_TREE_SHA256__",
    "__PROJECTOR_SET_SHA256__", "__PLAN_CONTAINER_PATH__",
    "__DEV_DATA_TREE_SHA256__", "__DEV_DATA_MANIFEST_SHA256__",
    "__DEV_DATA_DECLARED_TREE_SHA256__",
    "__PREPARED_INPUT_LOCK_MANIFEST__", "__PREPARED_INPUT_LOCK_MANIFEST_SHA256__",
    "__PREPARED_INPUT_LOCK_SIDECAR__", "__PREPARED_INPUT_LOCK_SIDECAR_SHA256__",
    "__EXPECTED_ROW_TEMPLATE_SHA256__", "__INPUT_LOCK_HOST_ROOT__",
    "__INSTRUMENTATION_GATE_CONTAINER_PATH__", "__OUTPUT_CONTAINER_ROOT__",
    "__SHARD_ID__", "__SHARD_SLUG__", "__LAMBDA_TAG__", "__CLAIM_ID__",
    "__PHASE__", "__REQUIRES_PHASE__",
    "__SOURCE_SNAPSHOT_HOST__",
    "__RAW_TOPOLOGY_HOST_ROOT__",
    "__RAW_TOPOLOGY_MANIFEST__", "__RAW_TOPOLOGY_MANIFEST_SHA256__",
    "__FINALIZED_LOCK_SHA256__", "__FINALIZED_OUTPUT_HOST_ROOT__",
    "__FINALIZED_OUTPUT_CONTAINER_ROOT__",
    "__PLAN_GATE_CONFIGMAP_NAME__", "__RUNTIME_CONFIGMAP_NAME__",
    "__EXPECTED_PLAN_SHA256__",
    "__FINALIZED_CONFIGMAP_NAME__", "__FINALIZED_RECEIPT_CONTAINER_PATH__",
    "__REPO_CONTAINER_ROOT__", "__E0_HOST_ROOT__", "__E0_CONTAINER_ROOT__",
    "__OUTPUT_HOST_ROOT__", "__OUTPUT_CONTAINER_ROOT_MOUNT__",
    "__INPUT_LOCK_HOST_ROOT_RENDER__", "__INPUT_LOCK_CONTAINER_ROOT__",
    "__RAW_TOPOLOGY_CONTAINER_ROOT__", "__GATE_CONTAINER_ROOT__",
    "__RUNTIME_CONTAINER_ROOT__", "__MODELS_HOST_ROOT__", "__MODELS_CONTAINER_ROOT__",
    "__SOURCE_SNAPSHOT_RECEIPT__", "__SOURCE_SNAPSHOT_RECEIPT_SHA256__",
    "__SOURCE_SNAPSHOT_CANONICAL_TREE_SHA256__",
    "__FINALIZED_ARG_BLOCK__", "__FINALIZED_VOLUME_MOUNT_BLOCK__",
    "__FINALIZED_VOLUME_BLOCK__", "__RUNNER_CONTAINER_PATH__",
)
UNRESOLVED = re.compile("|".join(re.escape(value) for value in PLACEHOLDERS))
IMAGE_DIGEST = re.compile(
    r"^(?:[a-z0-9.-]+(?::[0-9]+)?/)?[a-z0-9._-]+"
    r"(?:/[a-z0-9._-]+)*@sha256:[0-9a-f]{64}$"
)
EXECUTION_SHA = re.compile(r"^[0-9a-f]{40}$")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def ordered_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize a schema-ordered row without canonical key reordering."""

    return (json.dumps(value, sort_keys=False, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _contains_unresolved(value: Any) -> bool:
    if isinstance(value, str):
        return UNRESOLVED.search(value) is not None
    if isinstance(value, Mapping):
        return any(_contains_unresolved(key) or _contains_unresolved(child) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_unresolved(child) for child in value)
    return False


def _reject_sensitive_payload(value: Any, path: str = "root") -> None:
    sensitive = ("label", "answer", "prediction", "correctness", "outcome")
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized == "label_free":
                if child is not True:
                    raise ValueError(f"label-free firewall missing at {path}.{key}")
                continue
            if any(token in normalized for token in sensitive) and child not in (False, None, 0, "", [], {}):
                raise ValueError(f"outcome firewall violation at {path}.{key}")
            _reject_sensitive_payload(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_sensitive_payload(child, f"{path}[{index}]")


def _task_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {task: 0 for task in TASKS}
    for row in rows:
        task = row.get("task")
        if task not in counts:
            raise ValueError(f"unknown task: {task}")
        counts[task] += 1
    return counts


def load_e0_design_lock(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    _reject_sensitive_payload(manifest)
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError("split manifest has no rows")
    roles = {row.get("role") for row in rows}
    if not roles.issubset({"e0_design", "e1_pilot"}):
        raise ValueError(f"sealed role in split manifest: {roles}")
    e0_rows = [row for row in rows if row.get("role") == ALLOWED_SPLIT_ROLE]
    pilot_rows = [row for row in rows if row.get("role") == "e1_pilot"]
    if _task_counts(e0_rows) != TASK_GROUP_COUNTS or _task_counts(pilot_rows) != TASK_GROUP_COUNTS:
        raise ValueError("split counts differ from the frozen 128/70/128 contract")
    e0_groups: dict[str, dict[str, Any]] = {}
    pilot_groups: set[str] = set()
    for row in rows:
        group = row.get("content_group_sha256")
        samples = row.get("sample_key_sha256")
        if not _is_sha256(group) or not isinstance(samples, list) or not samples or not all(_is_sha256(v) for v in samples):
            raise ValueError("invalid group/sample SHA in split lock")
        if row["role"] == ALLOWED_SPLIT_ROLE:
            if group in e0_groups:
                raise ValueError("duplicate E0-design group")
            e0_groups[group] = {"task": row["task"], "sample_sha256": tuple(samples)}
        else:
            pilot_groups.add(group)
    if set(e0_groups) & pilot_groups:
        raise ValueError("E0-design and E1-pilot overlap")
    integrity = manifest.get("integrity", {})
    for name in ("labels_answers_predictions_correctness_accessed", "e1_pilot_forward_or_result_accessed", "confirmatory_outcomes_accessed"):
        if integrity.get(name) is not False:
            raise ValueError(f"split lock lacks false seal: {name}")
    return {"sha256": sha256_file(path), "e0_groups": e0_groups, "e1_pilot_groups": pilot_groups}


def verify_e0_dev_anchor(path: Path, split: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _read_json(path)
    _reject_sensitive_payload(manifest)
    rows = manifest.get("rows")
    if not isinstance(rows, list) or _task_counts(rows) != TASK_GROUP_COUNTS:
        raise ValueError("E0 development manifest count mismatch")
    observed = set()
    runtime_records: dict[str, dict[str, Any]] = {}
    for row in rows:
        group, sample = row.get("content_group_sha256"), row.get("sample_key_sha256")
        if group not in split["e0_groups"] or sample not in split["e0_groups"][group]["sample_sha256"]:
            raise ValueError("E0 development row outside frozen membership")
        if row.get("task") != split["e0_groups"][group]["task"]:
            raise ValueError("E0 task/group mapping changed")
        required = (
            "source_row_id", "subject", "evaluation_subject",
            "evaluation_question_id", "rendered_prompt_sha256",
            "alignment_sha256", "candidate_count", "eligibility",
        )
        missing = [name for name in required if name not in row]
        if missing:
            raise ValueError(f"E0 development row lacks runtime provenance: {missing}")
        if not _is_sha256(row["rendered_prompt_sha256"]) or not _is_sha256(row["alignment_sha256"]):
            raise ValueError("E0 prompt/alignment SHA provenance is malformed")
        if group in runtime_records:
            raise ValueError("E0 mechanism population requires one sample per content group")
        runtime_records[group] = {
            "task": row["task"], "source_row_id": row["source_row_id"],
            "subject": row["subject"], "evaluation_subject": row["evaluation_subject"],
            "evaluation_question_id": row["evaluation_question_id"],
            "rendered_prompt_sha256": row["rendered_prompt_sha256"],
            "prompt_alignment_sha256": row["alignment_sha256"],
            "certified_candidate_count": row["candidate_count"],
            "eligibility": row["eligibility"], "sample_sha256": sample,
        }
        observed.add(group)
    if observed != set(split["e0_groups"]):
        raise ValueError("E0 development and E1 lock differ")
    return {"sha256": sha256_file(path), "records": runtime_records}


def _expected_eval_records() -> dict[tuple[int, str, str], str]:
    return {(seed, cell, task): f"eval_{seed}_{cell}_{task}.yaml" for seed in SEEDS for cell in CELL_SPEC for task in TASKS}


def verify_e0_configs(repo_root: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    index_path = repo_root / E0_CONFIG_INDEX_RELATIVE
    records = _read_json(index_path).get("records")
    if not isinstance(records, list):
        raise ValueError("E0 config index has no records")
    expected, found = _expected_eval_records(), {}
    for record in records:
        if record.get("kind") != "evaluation":
            continue
        key = (int(record.get("seed", -1)), record.get("cell"), record.get("task"))
        if key not in expected or record.get("filename") != expected[key] or not _is_sha256(record.get("sha256")):
            raise ValueError(f"unexpected E0 evaluation config: {key}")
        path = repo_root / E0_CONFIG_ROOT_RELATIVE / record["filename"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"E0 config SHA mismatch: {path}")
        found[key] = {**record, "relative": str(path.relative_to(repo_root))}
    if set(found) != set(expected):
        raise ValueError("E0 evaluation config index incomplete")
    import yaml
    for (seed, cell, task), record in found.items():
        config = yaml.safe_load((repo_root / record["relative"]).read_text(encoding="utf-8"))
        arm, operator = CELL_SPEC[cell]
        rosetta = config["model"]["rosetta_config"]
        suffix = f"/seeds/{seed}/active/{TRAINED_DIRECTORY[arm]}/final"
        if not str(rosetta.get("checkpoints_dir", "")).endswith(suffix):
            raise ValueError("E0 checkpoint mapping changed")
        if rosetta.get("fpct_operator") != operator or rosetta.get("include_response") is not False or rosetta.get("attn_implementation") != "eager":
            raise ValueError("E0 operator/include_response/backend mapping changed")
        if config.get("eval", {}).get("dataset") != task:
            raise ValueError("E0 task config mismatch")
    return found


def _logical(root: str, relative: str | Path) -> str:
    return f"{root}://{PurePosixPath(str(relative))}"


def _parse_logical(value: str) -> tuple[str, PurePosixPath]:
    if "://" not in value:
        raise ValueError(f"not a logical path: {value}")
    root, relative = value.split("://", 1)
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe logical path: {value}")
    return root, pure


def resolve_logical_path(plan: Mapping[str, Any], logical: str, view: str = "host") -> Path:
    root, relative = _parse_logical(logical)
    roots = plan.get("path_roots", {})
    if root not in roots or view not in roots[root]:
        raise ValueError(f"logical root/view unavailable: {root}/{view}")
    return Path(roots[root][view]).joinpath(*relative.parts)


def _is_within_or_equal(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _validate_output_root_isolation(path_roots: Mapping[str, Any]) -> None:
    """Keep the sole writable output mount disjoint from every frozen RO root."""

    required = {
        "repo", "e0", "output", "gate", "runtime", "input_lock",
        "raw_topology", "models",
    }
    if not isinstance(path_roots, Mapping) or set(path_roots) != required:
        raise ValueError("execution path-root universe changed")
    output_record = path_roots.get("output")
    if not isinstance(output_record, Mapping):
        raise ValueError("output path root is absent")
    output_host_raw = str(output_record.get("host", ""))
    output_container_raw = str(output_record.get("container", ""))
    if _contains_unresolved(output_host_raw) or not Path(output_host_raw).is_absolute():
        raise ValueError("output host root must be a concrete absolute path")
    output_host = Path(output_host_raw).resolve(strict=False)
    for role in ("repo", "e0", "models", "input_lock", "raw_topology"):
        record = path_roots.get(role)
        if not isinstance(record, Mapping):
            raise ValueError(f"read-only path root is absent: {role}")
        raw = str(record.get("host", ""))
        if _contains_unresolved(raw):
            continue
        path = Path(raw)
        if not path.is_absolute():
            raise ValueError(f"read-only host root must be absolute: {role}")
        physical = path.resolve(strict=False)
        if _is_within_or_equal(output_host, physical) or _is_within_or_equal(
            physical, output_host
        ):
            raise ValueError(
                f"writable output host root overlaps read-only {role} root"
            )

    def container_path(role: str) -> PurePosixPath | None:
        record = path_roots.get(role)
        if not isinstance(record, Mapping):
            raise ValueError(f"container path root is absent: {role}")
        raw = str(record.get("container", ""))
        if _contains_unresolved(raw):
            return None
        path = PurePosixPath(raw)
        if not path.is_absolute() or ".." in path.parts or str(path) != raw.rstrip("/"):
            raise ValueError(f"container root must be canonical absolute POSIX: {role}")
        return path

    output_container = container_path("output")
    if output_container is None:
        raise ValueError("output container root must be concrete")
    for role in ("repo", "e0", "models", "input_lock", "raw_topology", "gate", "runtime"):
        read_only = container_path(role)
        if read_only is None:
            continue
        if _is_within_or_equal(output_container, read_only) or _is_within_or_equal(
            read_only, output_container
        ):
            raise ValueError(
                f"writable output container root overlaps read-only {role} root"
            )


def _validate_finalized_overlay(
    plan: Mapping[str, Any], finalized_output_host: Path,
) -> None:
    output_root = Path(plan["path_roots"]["output"]["host"]).resolve(strict=False)
    finalized_root = finalized_output_host.resolve(strict=False)
    if finalized_root == output_root or _is_within_or_equal(output_root, finalized_root):
        raise ValueError("finalized read-only overlay may not contain the writable output root")
    if not _is_within_or_equal(finalized_root, output_root):
        return
    for shard in plan["shards"]:
        if shard["stage"] != STAGE_E1_3:
            continue
        destination = (output_root / shard["output_relative"]).resolve(strict=False)
        if _is_within_or_equal(destination, finalized_root):
            raise ValueError("E1-3 shard output falls inside finalized read-only overlay")


def _tree_manifest(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = []
    for file_path in sorted(path.rglob("*")):
        if file_path.is_symlink():
            raise ValueError(f"checkpoint artifact contains symlink: {file_path}")
        if file_path.is_file():
            files.append({"relative": file_path.relative_to(path).as_posix(), "bytes": file_path.stat().st_size, "sha256": sha256_file(file_path)})
    if not files:
        raise ValueError(f"empty checkpoint artifact tree: {path}")
    return {"file_count": len(files), "bytes": sum(r["bytes"] for r in files), "files": files, "tree_sha256": sha256_bytes(canonical_json_bytes(files))}


def _e0_declared_tree_sha256(path: Path) -> str:
    """Reproduce fpct_e0_runner.tree_sha256 exactly.

    This is intentionally distinct from ``_tree_manifest``: E0 hashes
    ``relative-path + NUL + raw file-digest bytes`` while the executor also
    freezes a richer canonical file manifest.
    """

    digest = hashlib.sha256()
    files = sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: item.relative_to(path).as_posix())
    if not files:
        raise ValueError(f"empty E0 declared tree: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def _resolve_checkpoint(e0_root: Path, seed: int, checkpoint_arm: str) -> dict[str, Any]:
    arm = TRAINED_DIRECTORY[checkpoint_arm]
    active = e0_root / "seeds" / str(seed) / "active"
    if not active.is_symlink():
        raise ValueError(f"E0 active pointer is not a symlink: {active}")
    final = (active / arm / "final").resolve(strict=True)
    step64 = (active / arm / "checkpoint-64").resolve(strict=True)
    root = e0_root.resolve(strict=True)
    try:
        final_rel, step_rel = final.relative_to(root), step64.relative_to(root)
    except ValueError as error:
        raise ValueError("checkpoint resolves outside E0 root") from error
    if not re.fullmatch(r"attempt_[0-9]+", final_rel.parts[2]) or final_rel.parts[-2:] != (arm, "final"):
        raise ValueError(f"active/final did not resolve to immutable attempt path: {final_rel}")
    if step_rel.parts[:4] != final_rel.parts[:4] or step_rel.parts[-1] != "checkpoint-64":
        raise ValueError("step64 and final do not resolve to the same immutable attempt/arm")
    final_tree, step_tree = _tree_manifest(final), _tree_manifest(step64)
    final_projectors = {r["relative"]: r["sha256"] for r in final_tree["files"] if r["relative"].startswith("projector")}
    step_projectors = {r["relative"]: r["sha256"] for r in step_tree["files"] if r["relative"].startswith("projector")}
    if not final_projectors or final_projectors != step_projectors:
        raise ValueError("final projector artifacts are not identical to tracked step64 artifacts")
    return {
        "checkpoint_id": f"{seed}-{checkpoint_arm}", "seed": seed, "checkpoint_arm": checkpoint_arm,
        "active_pointer": _logical("e0", active.relative_to(e0_root)),
        "immutable_attempt": final_rel.parts[2], "final": {"logical_path": _logical("e0", final_rel), **final_tree},
        "step64": {"logical_path": _logical("e0", step_rel), **step_tree},
        "projector_artifact_set_sha256": sha256_bytes(canonical_json_bytes(final_projectors)), "read_only_reuse": True,
    }


def _missing_checkpoint_record(seed: int, checkpoint_arm: str) -> dict[str, Any]:
    arm = TRAINED_DIRECTORY[checkpoint_arm]
    return {
        "checkpoint_id": f"{seed}-{checkpoint_arm}", "seed": seed, "checkpoint_arm": checkpoint_arm,
        "active_pointer": _logical("e0", f"seeds/{seed}/active"), "immutable_attempt": "__IMMUTABLE_ATTEMPT__",
        "final": {"logical_path": _logical("e0", f"seeds/{seed}/__IMMUTABLE_ATTEMPT__/{arm}/final"), "tree_sha256": "__FINAL_TREE_SHA256__", "files": []},
        "step64": {"logical_path": _logical("e0", f"seeds/{seed}/__IMMUTABLE_ATTEMPT__/{arm}/checkpoint-64"), "tree_sha256": "__STEP64_TREE_SHA256__", "files": []},
        "projector_artifact_set_sha256": "__PROJECTOR_SET_SHA256__", "read_only_reuse": True,
    }


def verify_instrumentation_gate(
    path: Path,
    expected_sha256: str | None = None,
    *,
    evidence_resolver: Callable[[str], Path] | None = None,
) -> dict[str, Any]:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise RuntimeError("instrumentation gate exact SHA mismatch")
    gate = _read_json(path)
    if gate.get("gate_id") != CONSOLIDATED_GATE_ID or gate.get("status") != "GO":
        raise RuntimeError("consolidated E1 instrumentation hard gate is not GO")
    checks = gate.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != set(REQUIRED_GATE_CHECKS):
        raise RuntimeError("instrumentation gate must contain the exact consolidated check set")
    failed = [name for name in REQUIRED_GATE_CHECKS if checks.get(name) is not True]
    if failed:
        raise RuntimeError(f"instrumentation hard gate failed: {failed}")
    evidence = gate.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise RuntimeError("instrumentation gate has no hashed evidence")
    for record in evidence:
        if not isinstance(record, Mapping) or not _is_sha256(record.get("sha256")) or not record.get("logical_path"):
            raise RuntimeError("instrumentation gate evidence is not fully hashed")
        if evidence_resolver is None:
            root, relative = _parse_logical(str(record["logical_path"]))
            if root != "gate":
                raise RuntimeError("instrumentation evidence requires an explicit logical-path resolver")
            evidence_path = path.parent.joinpath(*relative.parts)
        else:
            evidence_path = evidence_resolver(str(record["logical_path"]))
        if not evidence_path.is_file() or sha256_file(evidence_path) != record["sha256"]:
            raise RuntimeError(f"instrumentation gate evidence SHA mismatch: {record['logical_path']}")
    return {
        "logical_path": gate.get("logical_path", "external://consolidated_gate"),
        "sha256": sha256_file(path), "gate_id": CONSOLIDATED_GATE_ID,
        "status": "GO", "checks": dict(checks),
        "evidence": [dict(record) for record in evidence],
    }


def _plan_hash(plan: Mapping[str, Any]) -> str:
    projected = dict(plan)
    projected.pop("plan_sha256", None)
    return sha256_bytes(canonical_json_bytes(projected))


def _expected_claim_id(plan_sha256: str, shard_id: str) -> str:
    return sha256_bytes(f"{plan_sha256}:{shard_id}".encode())


def _lambda_tag(value: float) -> str:
    return f"lambda-{value:.2f}".replace(".", "p")


def _shard_id(stage: str, seed: int, cell: str, task: str, value: float) -> str:
    return f"{stage}__{seed}__{cell}__{task}__{_lambda_tag(value)}"


def _membership_sha(groups: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes(list(groups)))


def _semantic_python(value: Any) -> Any:
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except ImportError:
        pass
    if isinstance(value, Mapping):
        return {str(key): _semantic_python(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_semantic_python(child) for child in value]
    return value


def _nested_semantic_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(_semantic_python(value)))


def _expected_long_form_rows_contract(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from script.experiment.fpct_e1_prepare_input_lock import (
        expected_long_form_row_volume,
    )

    return expected_long_form_row_volume(items)


def _a4_operative_source_lock(repo_root: Path) -> dict[str, Any]:
    """Bind the prospective A4 approval and pre-natural synthetic hard gate.

    A v2 input lock is necessary but not sufficient to make the historical DAG
    operative.  Every successor plan must be constructed from a source snapshot
    that contains the approved v6 manifest/contract/schema and a GO synthetic
    receipt.  Older plans therefore cannot be relabelled or resumed in place.
    """

    paths = {
        "manifest": repo_root / E1_MANIFEST_RELATIVE,
        "contract": repo_root / STREAMING_CONTRACT_RELATIVE,
        "schema": repo_root / STREAMING_SCHEMA_RELATIVE,
        "synthetic_gate": repo_root / STREAMING_SYNTHETIC_GATE_RELATIVE,
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("A4 operative source lock is incomplete")
    manifest = _read_json(paths["manifest"])
    contract = _read_json(paths["contract"])
    from script.analysis.fpct_e1_streaming_synthetic_gate import verify_tracked_gate

    synthetic = verify_tracked_gate(paths["synthetic_gate"], repo_root)
    if (
        manifest.get("schema_version") != 6
        or manifest.get("protocol_id") != A4_PROTOCOL_ID
    ):
        raise ValueError("E1 manifest is not the operative v6 A4 protocol")
    authorization = manifest.get("execution_authorization", {})
    expected_authorization = {
        "a4_successor_input_lock_operative": True,
        "a4_successor_dag_operative": True,
        "requires_streaming_input_lock_go": True,
        "historical_attempt_resume_allowed": False,
        "e1_pilot_forward_or_outcome": False,
        "training": False,
    }
    if authorization != expected_authorization:
        raise ValueError("E1 manifest lacks the exact prospective A4 execution authorization")
    approval = contract.get("approval", {})
    if (
        contract.get("schema_version") != 6
        or contract.get("protocol_id") != A4_PROTOCOL_ID
        or approval.get("approval_id")
        != "APPROVED_PROSPECTIVE_AMENDMENT_E1_A4_STREAMING"
        or approval.get("prospective") is not True
        or contract.get("historical_execution", {}).get("resume_allowed") is not False
        or contract.get("historical_execution", {}).get("artifact_reuse_allowed") is not False
    ):
        raise ValueError("A4 prospective streaming approval is absent or changed")
    schema_sha256 = sha256_file(paths["schema"])
    if (
        synthetic.get("schema_version") != 6
        or synthetic.get("protocol_id") != A4_PROTOCOL_ID
        or synthetic.get("status") != "GO_PRE_NATURAL_SYNTHETIC_HARD_GATE"
        or synthetic.get("natural_data_accessed") is not False
        or synthetic.get("physical_chunk_rows") != PARQUET_BATCH_ROWS
        or synthetic.get("streaming_schema_sha256") != schema_sha256
    ):
        raise ValueError("A4 synthetic streaming hard gate is absent or changed")
    return {
        "status": "GO_NEW_A4_SUCCESSOR_DAG_ONLY",
        "protocol_id": A4_PROTOCOL_ID,
        "historical_execution_resume_allowed": False,
        "historical_artifact_reuse_allowed": False,
        "execution_authorization": expected_authorization,
        "source": {
            name: {
                "logical_path": _logical("repo", path.relative_to(repo_root)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in sorted(paths.items())
        },
    }


def _prepared_input_lock(
    manifest_path: Path | None, sidecar_path: Path | None,
    group_contract: Mapping[str, Sequence[Mapping[str, Any]]],
    *, repo_root: Path,
) -> dict[str, Any]:
    if manifest_path is None or sidecar_path is None:
        return {
            "manifest": {"logical_path": "input_lock://__PREPARED_INPUT_LOCK_MANIFEST__", "sha256": "__PREPARED_INPUT_LOCK_MANIFEST_SHA256__"},
            "sidecar": {"logical_path": "input_lock://__PREPARED_INPUT_LOCK_SIDECAR__", "sha256": "__PREPARED_INPUT_LOCK_SIDECAR_SHA256__", "bytes": -1},
            "dimensions": {}, "task_contract": {}, "status": "UNRESOLVED",
        }
    if manifest_path.absolute().parent != sidecar_path.absolute().parent:
        raise ValueError("prepared input-lock manifest and sidecar must share one portable root")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != INPUT_LOCK_SCHEMA_VERSION
        or manifest.get("protocol_id") != INPUT_LOCK_PROTOCOL_ID
        or manifest.get("status") != INPUT_LOCK_STATUS
        or manifest.get("split_role") != ALLOWED_SPLIT_ROLE
    ):
        raise ValueError("prepared input-lock manifest identity/status mismatch")
    dimensions = manifest.get("dimensions")
    if dimensions != {"num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8}:
        raise ValueError("prepared input-lock receiver dimensions changed")
    firewall = manifest.get("firewall", {})
    if not firewall or any(value is not False for value in firewall.values()):
        raise ValueError("prepared input-lock firewall is not fully false")
    streaming = manifest.get("streaming_contract", {})
    operative = _a4_operative_source_lock(repo_root)
    frozen_schema = operative["source"]["schema"]
    frozen_gate = operative["source"]["synthetic_gate"]
    if (
        streaming.get("protocol_id") != A4_PROTOCOL_ID
        or streaming.get("schema_sha256") != frozen_schema["sha256"]
        or streaming.get("physical_chunk_rows") != PARQUET_BATCH_ROWS
        or streaming.get("historical_cumulative_ceiling_operative") is not False
        or streaming.get("synthetic_gate", {}).get("sha256") != frozen_gate["sha256"]
        or not isinstance(streaming.get("geometry_lock"), Mapping)
        or not isinstance(streaming.get("streaming_template_lock"), Mapping)
    ):
        raise ValueError("prepared input-lock lacks the operative A4 streaming lock")
    portable_geometry: dict[str, Any] = {}
    for name in ("manifest", "samples", "receipt"):
        record = streaming["geometry_lock"].get(name, {})
        source = Path(str(record.get("path", "")))
        if (
            not source.is_file()
            or source.absolute().parent != manifest_path.absolute().parent
            or record.get("sha256") != sha256_file(source)
            or record.get("bytes") != source.stat().st_size
        ):
            raise ValueError(f"prepared compact-geometry artifact changed: {name}")
        portable_geometry[name] = {
            "logical_path": _logical("input_lock", source.name),
            "sha256": record["sha256"],
            "bytes": record["bytes"],
        }
    geometry_receipt = _read_json(
        manifest_path.absolute().parent / Path(
            portable_geometry["receipt"]["logical_path"].split("://", 1)[1]
        )
    )
    if (
        geometry_receipt.get("schema_version") != 6
        or geometry_receipt.get("protocol_id") != A4_PROTOCOL_ID
        or geometry_receipt.get("status") != "GO"
        or geometry_receipt.get("expanded_logical_rows_materialized") is not False
        or geometry_receipt.get("e1_pilot_consumed") is not False
        or geometry_receipt.get("model_or_checkpoint_loaded") is not False
        or geometry_receipt.get("gpu_or_kubernetes_used") is not False
    ):
        raise ValueError("compact geometry receipt is not a model-output-free GO")
    portable_template: dict[str, Any] = {}
    for name in ("index", "receipt"):
        record = streaming["streaming_template_lock"].get(name, {})
        source = Path(str(record.get("path", "")))
        if (
            not source.is_file()
            or source.absolute().parent != manifest_path.absolute().parent
            or record.get("sha256") != sha256_file(source)
            or record.get("bytes") != source.stat().st_size
        ):
            raise ValueError(f"prepared streaming-template artifact changed: {name}")
        portable_template[name] = {
            "logical_path": _logical("input_lock", source.name),
            "sha256": record["sha256"],
            "bytes": record["bytes"],
        }
    template_receipt_path = manifest_path.absolute().parent / Path(
        portable_template["receipt"]["logical_path"].split("://", 1)[1]
    )
    template_receipt = _read_json(template_receipt_path)
    required_receipt_true = (
        "protocol_and_schema_versioned",
        "expected_logical_rows_eq_emitted",
        "ordinal_first_eq_zero",
        "ordinal_last_eq_expected_minus_one",
        "ordinal_ranges_contiguous",
        "row_key_reference_equivalence",
        "weights_reference_equivalence",
        "topology_reference_equivalence",
        "semantic_stream_replay_equal",
        "chunk_partition_semantic_equivalence",
        "aggregate_partition_equivalence",
        "bounded_peak_rss",
        "atomic_no_overwrite",
        "crash_resume_equivalence",
        "raw_topology_complete",
        "task_counts_exact_128_70_128",
        "input_assets_unchanged_during_lock",
    )
    if (
        template_receipt.get("schema_version") != 6
        or template_receipt.get("protocol_id") != A4_PROTOCOL_ID
        or template_receipt.get("status") != "GO"
        or any(template_receipt.get(name) is not True for name in required_receipt_true)
        or template_receipt.get("old_attempt_artifact_reused") is not False
        or template_receipt.get("e1_pilot_consumed") is not False
        or template_receipt.get("model_or_checkpoint_loaded") is not False
        or template_receipt.get("gpu_or_kubernetes_used") is not False
        or template_receipt.get("whole_table_materialization_detected") is not False
        or any(template_receipt.get(name) != 0 for name in ("missing_rows", "duplicate_rows", "overlapping_chunks"))
        or template_receipt.get("schema_sha256") != frozen_schema["sha256"]
        or template_receipt.get("rss_lock_sha256") != frozen_gate["sha256"]
        or template_receipt.get("geometry_manifest_sha256")
        != portable_geometry["manifest"]["sha256"]
    ):
        raise ValueError("streaming input-lock receipt is not an operative A4 GO")
    sidecar = manifest.get("sidecar", {})
    if not sidecar_path.is_file() or sidecar.get("bytes") != sidecar_path.stat().st_size or sidecar.get("sha256") != sha256_file(sidecar_path):
        raise ValueError("prepared input-lock sidecar SHA/size mismatch")
    import torch
    payload = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    payload_firewall = payload.get("firewall", {}) if isinstance(payload, Mapping) else {}
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != INPUT_LOCK_SCHEMA_VERSION
        or payload.get("protocol_id") != manifest["protocol_id"]
        or payload.get("status") != INPUT_LOCK_STATUS
        or payload.get("split_role") != ALLOWED_SPLIT_ROLE
        or payload.get("dimensions") != dimensions
        or payload.get("model_or_checkpoint_loaded") is not False
        or payload.get("cuda_initialized") is not False
        or any(
            payload_firewall.get(name) is not False
            for name in (
                "e1_pilot_consumed", "model_selection_consumed",
                "test_consumed", "model_or_checkpoint_loaded", "gpu_or_cuda_used",
            )
        )
    ):
        raise ValueError("prepared input-lock sidecar identity/dimensions/firewall mismatch")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("prepared input-lock sidecar has no semantic items")
    execution_identity = manifest.get("execution_identity")
    if (
        not isinstance(execution_identity, Mapping)
        or payload.get("execution_identity") != execution_identity
        or execution_identity.get("protocol_id")
        != "fpct_e1_a4_input_lock_execution_identity_v1"
        or not EXECUTION_SHA.fullmatch(str(execution_identity.get("execution_sha", "")))
        or execution_identity.get("execution_prefix")
        != execution_identity["execution_sha"][:8]
        or execution_identity.get("historical_execution_resume_allowed") is not False
        or execution_identity.get("historical_artifact_reuse_allowed") is not False
        or not _is_sha256(execution_identity.get("identity_sha256"))
        or Path(str(execution_identity.get("source_snapshot_root", ""))).absolute()
        != repo_root.absolute()
        or Path(str(execution_identity.get("input_lock_root", ""))).absolute()
        != manifest_path.absolute().parent
    ):
        raise ValueError("prepared input lock is not bound to one new A4 execution identity")
    contracts = manifest.get("task_contract")
    if not isinstance(contracts, Mapping) or set(contracts) != set(TASKS):
        raise ValueError("prepared input-lock task universe changed")
    runtime_assets = manifest.get("runtime_assets")
    if not isinstance(runtime_assets, Mapping) or set(runtime_assets) != {"receiver", "sender"}:
        raise ValueError("prepared input-lock runtime asset universe changed")
    for role, asset in runtime_assets.items():
        if not isinstance(asset, Mapping) or not asset.get("model_id") or not _is_sha256(asset.get("tree_sha256")):
            raise ValueError(f"prepared input-lock runtime asset record invalid: {role}")
        files = asset.get("files")
        if not isinstance(files, list) or not files or asset.get("file_count") != len(files):
            raise ValueError(f"prepared input-lock runtime asset file manifest invalid: {role}")
        if any(not _is_sha256(row.get("sha256")) or row.get("kind") not in {"file", "symlink_file"} for row in files):
            raise ValueError(f"prepared input-lock runtime asset file record invalid: {role}")
    for task in TASKS:
        record, template = contracts[task], contracts[task].get("row_template", {})
        if record.get("group_count") != len(group_contract[task]) or not _is_sha256(record.get("membership_sha256")):
            raise ValueError(f"prepared input-lock membership invalid: {task}")
        if (
            not isinstance(template.get("count"), int)
            or template["count"] <= 0
            or not _is_sha256(template.get("semantic_stream_sha256"))
            or template.get("schema_sha256") != frozen_schema["sha256"]
            or template.get("physical_chunk_rows") != PARQUET_BATCH_ROWS
            or template.get("ordinal_order") != "sample_sha256,row_ordinal"
        ):
            raise ValueError(f"prepared input-lock row template invalid: {task}")
        members = sorted((item for item in items if item.get("task") == task), key=lambda item: item.get("sample_sha256", ""))
        frozen_groups = {row["content_group_sha256"]: row for row in group_contract[task]}
        if len(members) != len(frozen_groups) or {item.get("content_group_sha256") for item in members} != set(frozen_groups):
            raise ValueError(f"prepared input-lock sidecar/group universe mismatch: {task}")
        semantic_members = []
        for item in members:
            group = frozen_groups[item["content_group_sha256"]]
            if item.get("sample_sha256") not in group["sample_sha256"]:
                raise ValueError(f"prepared input-lock sample/group mapping mismatch: {task}")
            descriptor = item.get("descriptor")
            if not isinstance(descriptor, Mapping):
                raise ValueError("prepared input-lock item lacks prompt-only descriptor provenance")
            if (
                descriptor.get("rendered_prompt_sha256") != group["rendered_prompt_sha256"]
                or descriptor.get("prompt_alignment_sha256") != group["prompt_alignment_sha256"]
                or item.get("rendered_prompt_sha256") != group["rendered_prompt_sha256"]
            ):
                raise ValueError("prompt-only provenance differs between group contract and input-lock sidecar")
            semantic = _nested_semantic_sha256({key: value for key, value in item.items() if key not in {"feature", "item_semantic_sha256"}})
            if semantic != item.get("item_semantic_sha256"):
                raise ValueError("prepared input-lock item semantic SHA does not recompute")
            semantic_members.append({
                "sample_sha256": item["sample_sha256"],
                "content_group_sha256": item["content_group_sha256"],
                "item_semantic_sha256": semantic,
            })
            from script.experiment.fpct_e1_prepare_input_lock import expected_long_form_rows
            expected_rows = expected_long_form_rows(
                answer_query_count=len(item.get("answer_queries", [])),
                certified_parent_count=len(item.get("certified_parents", [])),
                num_layers=dimensions["num_hidden_layers"],
                num_query_heads=dimensions["num_attention_heads"],
            )
            if (
                item.get("Q_s") != len(item.get("answer_queries", []))
                or item.get("P_s") != len(item.get("certified_parents", []))
                or item.get("N_s") != expected_rows
                or item.get("expected_long_form_rows") != expected_rows
                or item.get("expected_chunk_count")
                != (expected_rows + PARQUET_BATCH_ROWS - 1) // PARQUET_BATCH_ROWS
            ):
                raise ValueError("prepared input-lock compact geometry changed")
        if record.get("members") != semantic_members:
            raise ValueError(f"prepared input-lock manifest members differ from sidecar semantics: {task}")
        expected_long_form = _expected_long_form_rows_contract(members)
        if record.get("expected_long_form_rows") != expected_long_form:
            raise ValueError(f"prepared input-lock expected-long-form contract differs: {task}")
        from script.experiment.fpct_e1_prepare_input_lock import task_membership_sha256
        recomputed_membership = task_membership_sha256(group_contract[task], semantic_members)
        if recomputed_membership != record["membership_sha256"]:
            raise ValueError(f"prepared input-lock task membership SHA does not recompute: {task}")
    template_index_path = manifest_path.absolute().parent / Path(
        portable_template["index"]["logical_path"].split("://", 1)[1]
    )
    template_index = _read_json(template_index_path)
    template_records = template_index.get("records")
    if (
        template_index.get("schema_version") != 6
        or template_index.get("protocol_id") != A4_PROTOCOL_ID
        or template_index.get("artifact_type") != "chunk_manifest_index"
        or template_index.get("physical_chunk_rows") != PARQUET_BATCH_ROWS
        or template_index.get("schema_sha256") != frozen_schema["sha256"]
        or not isinstance(template_records, list)
        or len(template_records) != len(items)
        or template_receipt.get("chunk_manifest_index_sha256")
        != portable_template["index"]["sha256"]
        or template_receipt.get("semantic_stream_sha256")
        != template_index.get("semantic_stream_sha256")
        or streaming["streaming_template_lock"].get("expected_logical_rows")
        != template_index.get("expected_logical_rows")
        or streaming["streaming_template_lock"].get("emitted_logical_rows")
        != template_index.get("emitted_logical_rows")
        or streaming["streaming_template_lock"].get("semantic_stream_sha256")
        != template_index.get("semantic_stream_sha256")
    ):
        raise ValueError("streaming row-template index identity changed")
    template_by_sample = {record.get("sample_sha256"): record for record in template_records}
    if len(template_by_sample) != len(items):
        raise ValueError("streaming row-template index has duplicate samples")
    from script.analysis.fpct_e1_streaming_verify import (
        SampleRowStream,
        canonical_endpoint_id,
        iter_verified_parquet_rows,
        semantic_row_bytes,
    )
    replay_total = 0
    replay_semantic = hashlib.sha256()
    for item in sorted(items, key=lambda value: (value["sample_sha256"], value["content_group_sha256"])):
        record = template_by_sample.get(item["sample_sha256"])
        if (
            not isinstance(record, Mapping)
            or record.get("task") != item["task"]
            or record.get("content_group_sha256") != item["content_group_sha256"]
            or record.get("expected_logical_rows") != item["N_s"]
            or record.get("emitted_logical_rows") != item["N_s"]
            or record.get("chunk_count") != item["expected_chunk_count"]
            or record.get("item_semantic_sha256") != item["item_semantic_sha256"]
            or record.get("raw_topology_compact_sha256")
            != item["raw_topology_compact_sha256"]
            or not _is_sha256(record.get("manifest_sha256"))
            or not isinstance(record.get("manifest_bytes"), int)
            or record["manifest_bytes"] <= 0
            or not _is_sha256(record.get("semantic_stream_sha256"))
        ):
            raise ValueError("streaming row-template sample/task universe changed")
        relative = Path(str(record.get("manifest_relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("streaming row-template manifest path escapes its input root")
        sample_manifest_path = manifest_path.absolute().parent / relative
        if (
            not sample_manifest_path.is_file()
            or sha256_file(sample_manifest_path) != record["manifest_sha256"]
            or sample_manifest_path.stat().st_size != record["manifest_bytes"]
        ):
            raise ValueError("streaming row-template nested manifest changed")
        binding_relative = Path(str(record.get("execution_binding_relative_path", "")))
        if (
            binding_relative.is_absolute()
            or ".." in binding_relative.parts
            or not binding_relative.parts
        ):
            raise ValueError("streaming row-template execution binding path escapes input root")
        binding_path = manifest_path.absolute().parent / binding_relative
        expected_binding = {
            "schema_version": 1,
            "protocol_id": "fpct_e1_a4_sample_stream_execution_binding_v1",
            "execution_identity_sha256": execution_identity["identity_sha256"],
            "sample_sha256": item["sample_sha256"],
            "item_semantic_sha256": item["item_semantic_sha256"],
            "historical_artifact_reuse_allowed": False,
        }
        if (
            not binding_path.is_file()
            or sha256_file(binding_path) != record.get("execution_binding_sha256")
            or _read_json(binding_path) != expected_binding
        ):
            raise ValueError("streaming row-template execution binding changed")
        stream = SampleRowStream(
            item,
            num_layers=dimensions["num_hidden_layers"],
            num_query_heads=dimensions["num_attention_heads"],
            num_kv_heads=dimensions["num_key_value_heads"],
            endpoint_id=canonical_endpoint_id(
                0, "input_geometry", "input_geometry", "INPUT_LOCK", item["task"], 0.0,
            ),
            schema_sha256=frozen_schema["sha256"],
        )
        sample_count = 0
        sample_digest = hashlib.sha256()
        for row in iter_verified_parquet_rows(sample_manifest_path, stream):
            semantic = semantic_row_bytes(row)
            sample_digest.update(semantic)
            replay_semantic.update(semantic)
            sample_count += 1
        nested_manifest = _read_json(sample_manifest_path)
        expected_files = {
            sample_manifest_path.name,
            binding_path.name,
            *(Path(str(chunk["relative_path"])).name for chunk in nested_manifest.get("chunks", [])),
        }
        observed_files = {
            child.name for child in sample_manifest_path.parent.iterdir() if child.is_file()
        }
        if (
            sample_count != stream.row_count
            or sample_count != record["expected_logical_rows"]
            or sample_count != record["emitted_logical_rows"]
            or sample_digest.hexdigest() != record["semantic_stream_sha256"]
            or len(nested_manifest.get("chunks", [])) != record["chunk_count"]
            or observed_files != expected_files
        ):
            raise ValueError("streaming row-template nested replay receipt changed")
        replay_total += sample_count
    if (
        replay_total != template_index.get("expected_logical_rows")
        or replay_total != template_index.get("emitted_logical_rows")
        or replay_total != template_receipt.get("expected_logical_rows", replay_total)
        or replay_semantic.hexdigest() != template_index.get("semantic_stream_sha256")
        or replay_semantic.hexdigest() != template_receipt.get("semantic_stream_sha256")
    ):
        raise ValueError("streaming row-template global row count changed")
    expected_by_task = {
        task: {
            "count": contracts[task]["expected_long_form_rows"]["count"],
            "sum": contracts[task]["expected_long_form_rows"]["sum"],
        }
        for task in TASKS
    }
    if manifest.get("expected_long_form_rows_by_task") != expected_by_task:
        raise ValueError("prepared input-lock top-level row-volume contract differs")
    return {
        "manifest": {"logical_path": _logical("input_lock", manifest_path.name), "sha256": sha256_file(manifest_path)},
        "sidecar": {"logical_path": _logical("input_lock", sidecar_path.name), "sha256": sha256_file(sidecar_path), "bytes": sidecar_path.stat().st_size},
        "dimensions": dimensions, "task_contract": contracts,
        "runtime_assets": runtime_assets,
        "execution_identity": dict(execution_identity),
        "expected_long_form_rows_by_task": expected_by_task,
        "streaming_contract": {
            "protocol_id": A4_PROTOCOL_ID,
            "schema_sha256": frozen_schema["sha256"],
            "physical_chunk_rows": PARQUET_BATCH_ROWS,
            "geometry_lock": portable_geometry,
            "streaming_template_lock": {
                **portable_template,
                "expected_logical_rows": streaming["streaming_template_lock"].get("expected_logical_rows"),
                "emitted_logical_rows": streaming["streaming_template_lock"].get("emitted_logical_rows"),
                "semantic_stream_sha256": streaming["streaming_template_lock"].get("semantic_stream_sha256"),
            },
            "synthetic_gate_sha256": frozen_gate["sha256"],
        },
        "status": INPUT_LOCK_STATUS,
    }


def _prepared_raw_topology(
    manifest_path: Path | None,
    artifact_root: Path | None,
    input_lock_sidecar: Path | None,
) -> dict[str, Any]:
    """Bind the model-output-free raw topology export before any shard exists."""

    if manifest_path is None or artifact_root is None or input_lock_sidecar is None:
        return {
            "status": "UNRESOLVED",
            "root": "raw_topology://.",
            "manifest": {
                "logical_path": "raw_topology://__RAW_TOPOLOGY_MANIFEST__",
                "sha256": "__RAW_TOPOLOGY_MANIFEST_SHA256__",
                "bytes": -1,
            },
            "input_lock_sidecar_sha256": "__PREPARED_INPUT_LOCK_SIDECAR_SHA256__",
            "row_count": -1,
            "artifacts": {},
        }
    artifact_root = artifact_root.absolute()
    manifest_path = manifest_path.absolute()
    if manifest_path.parent != artifact_root:
        raise ValueError("raw-topology manifest must live directly in its artifact root")
    from script.analysis.fpct_e1_mechanism_audit import (
        RAW_TOPOLOGY_OUTPUT_NAMES,
        verify_raw_topology_artifacts,
    )

    expected_name = RAW_TOPOLOGY_OUTPUT_NAMES["manifest"]
    if manifest_path.name != expected_name:
        raise ValueError("raw-topology manifest filename differs from the frozen analyzer contract")
    verified = verify_raw_topology_artifacts(input_lock_sidecar, artifact_root)
    manifest = _read_json(manifest_path)
    if (
        verified.get("status") != "GO_MODEL_OUTPUT_FREE"
        or manifest.get("protocol_id") != "fpct_e1_raw_to_runtime_topology_audit_v1"
        or manifest.get("status") != "GO_MODEL_OUTPUT_FREE"
        or manifest.get("split_role") != ALLOWED_SPLIT_ROLE
        or manifest.get("functional_metrics_present") is not False
        or manifest.get("uncertified_functional_metrics_available") is not False
        or manifest.get("e1_pilot_consumed") is not False
        or manifest.get("confirmatory_consumed") is not False
        or manifest.get("input_lock_sidecar", {}).get("sha256") != sha256_file(input_lock_sidecar)
    ):
        raise ValueError("raw-topology manifest identity/firewall mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        RAW_TOPOLOGY_OUTPUT_NAMES[name]
        for name in ("jsonl", "parquet", "aggregate_csv", "aggregate_json")
    }:
        raise ValueError("raw-topology artifact universe changed")
    return {
        "status": "GO_MODEL_OUTPUT_FREE",
        "root": "raw_topology://.",
        "manifest": {
            "logical_path": _logical("raw_topology", manifest_path.name),
            "sha256": sha256_file(manifest_path),
            "bytes": manifest_path.stat().st_size,
        },
        "input_lock_sidecar_sha256": sha256_file(input_lock_sidecar),
        "row_count": int(verified["row_count"]),
        "artifacts": {name: dict(record) for name, record in sorted(artifacts.items())},
    }


def build_execution_plan(
    repo_root: Path, e0_root: Path, output_root: Path, *, require_checkpoints: bool = True,
    instrumentation_gate_path: Path | None = None, execution_sha: str = "__EXECUTION_SHA__",
    image_digest: str = "__IMMUTABLE_IMAGE_DIGEST__", backend_spec: str = "__FROZEN_RUNTIME_BACKEND__",
    backend_source: Path | None = None, runtime_provenance: Path | None = None,
    input_lock_manifest: Path | None = None, input_lock_sidecar: Path | None = None,
    raw_topology_manifest: Path | None = None, raw_topology_artifact_root: Path | None = None,
    source_snapshot_receipt: Path | None = None,
    repo_container: str = "/opt/fpct", e0_container: str = "/fpct-e0", output_container: str = "/fpct-e1/e0_design_capture",
    input_lock_container: str = "/fpct-e1/input_lock",
    raw_topology_container: str = "/fpct-e1/raw_topology",
    models_host_root: str = "/netdisk/lijunsi/c2c-route1-identifiability/models",
    models_container: str = "/models/c2c",
) -> dict[str, Any]:
    repo_root, e0_root, output_root = repo_root.resolve(), e0_root.absolute(), output_root.absolute()
    split = load_e0_design_lock(repo_root / SPLIT_MANIFEST_RELATIVE)
    dev = verify_e0_dev_anchor(repo_root / E0_DEV_MANIFEST_RELATIVE, split)
    configs = verify_e0_configs(repo_root)
    group_contract: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}
    for group, record in sorted(split["e0_groups"].items()):
        runtime = dev["records"].get(group)
        if runtime is None:
            raise ValueError("E0-design group lacks frozen runtime provenance")
        group_contract[record["task"]].append({
            "content_group_sha256": group,
            "task": record["task"],
            "sample_sha256": list(record["sample_sha256"]),
            "source_row_id": runtime["source_row_id"],
            "subject": runtime["subject"],
            "evaluation_subject": runtime["evaluation_subject"],
            "evaluation_question_id": runtime["evaluation_question_id"],
            "rendered_prompt_sha256": runtime["rendered_prompt_sha256"],
            "prompt_alignment_sha256": runtime["prompt_alignment_sha256"],
            "certified_candidate_count": runtime["certified_candidate_count"],
            "eligibility": runtime["eligibility"],
        })

    checkpoints = []
    for seed in SEEDS:
        for arm in CHECKPOINT_ARMS:
            if require_checkpoints:
                checkpoints.append(_resolve_checkpoint(e0_root, seed, arm))
            else:
                checkpoints.append(_missing_checkpoint_record(seed, arm))
    checkpoint_by_id = {r["checkpoint_id"]: r for r in checkpoints}

    if require_checkpoints:
        data_directory = e0_root / "dev_data"
        data_manifest_path = e0_root / "dev_data_manifest.json"
        data_manifest = _read_json(data_manifest_path)
        if data_manifest.get("row_counts") != TASK_GROUP_COUNTS or not _is_sha256(data_manifest.get("tree_sha256")):
            raise ValueError("E0 dev-data manifest differs from the frozen task/count contract")
        declared_computed = _e0_declared_tree_sha256(data_directory)
        if declared_computed != data_manifest["tree_sha256"]:
            raise ValueError("E0 dev-data declared tree SHA does not reproduce with its frozen algorithm")
        input_lock = {
            "data_root": _logical("e0", "dev_data"),
            "tree": _tree_manifest(data_directory),
            "manifest": {
                "logical_path": _logical("e0", "dev_data_manifest.json"),
                "sha256": sha256_file(data_manifest_path),
                "declared_tree_sha256": data_manifest["tree_sha256"],
                "declared_tree_algorithm": "relative_path_nul_file_sha256_bytes_v1",
                "recomputed_declared_tree_sha256": declared_computed,
            },
        }
    else:
        input_lock = {
            "data_root": _logical("e0", "dev_data"),
            "tree": {"tree_sha256": "__DEV_DATA_TREE_SHA256__", "files": []},
            "manifest": {"logical_path": _logical("e0", "dev_data_manifest.json"), "sha256": "__DEV_DATA_MANIFEST_SHA256__", "declared_tree_sha256": "__DEV_DATA_DECLARED_TREE_SHA256__", "declared_tree_algorithm": "relative_path_nul_file_sha256_bytes_v1", "recomputed_declared_tree_sha256": "__DEV_DATA_DECLARED_TREE_SHA256__"},
        }
    input_lock["prepared"] = _prepared_input_lock(
        input_lock_manifest, input_lock_sidecar, group_contract,
        repo_root=repo_root,
    )
    a4_operative_lock = _a4_operative_source_lock(repo_root)
    raw_topology = _prepared_raw_topology(
        raw_topology_manifest, raw_topology_artifact_root, input_lock_sidecar,
    )
    frozen_row_volume = input_lock["prepared"].get(
        "expected_long_form_rows_by_task", {}
    )
    e1_2_rows_by_task = {
        task: 12 * int(frozen_row_volume.get(task, {}).get("sum", -1))
        for task in TASKS
    }
    e1_3_addition_rows_by_task = {
        task: 24 * int(frozen_row_volume.get(task, {}).get("sum", -1))
        for task in TASKS
    }
    full_grid_rows_by_task = {
        task: 30 * int(frozen_row_volume.get(task, {}).get("sum", -1))
        for task in TASKS
    }
    final_108_rows_by_task = {
        task: 36 * int(frozen_row_volume.get(task, {}).get("sum", -1))
        for task in TASKS
    }

    if instrumentation_gate_path is not None:
        def resolve_gate_evidence(logical: str) -> Path:
            root, relative = _parse_logical(logical)
            roots = {"repo": repo_root, "gate": instrumentation_gate_path.parent.absolute()}
            if root not in roots:
                raise RuntimeError(f"instrumentation evidence logical root is not allowed: {root}")
            return roots[root].joinpath(*relative.parts)

        gate = verify_instrumentation_gate(
            instrumentation_gate_path,
            evidence_resolver=resolve_gate_evidence,
        )
        try:
            gate_rel = instrumentation_gate_path.resolve().relative_to(repo_root)
            gate["logical_path"] = _logical("repo", gate_rel)
        except ValueError:
            gate["logical_path"] = _logical("gate", instrumentation_gate_path.name)
            gate_container = f"/opt/fpct-e1-lock/{instrumentation_gate_path.name}"
    else:
        gate = {"logical_path": _logical("gate", "consolidated_gate.json"), "sha256": "__INSTRUMENTATION_GATE_SHA256__", "gate_id": CONSOLIDATED_GATE_ID, "status": "UNRESOLVED", "checks": {name: False for name in REQUIRED_GATE_CHECKS}}
        gate_container = "/opt/fpct-e1-lock/consolidated_gate.json"

    source_paths = [
        EXECUTOR_RELATIVE,
        Path("script/experiment/fpct_e1_prepare_input_lock.py"),
        Path("script/experiment/fpct_e1_runtime_backend.py"),
        Path("script/experiment/fpct_e1_runtime_probe.py"),
        RUNTIME_PROBE_RENDERER_RELATIVE,
        Path("script/experiment/fpct_e1_source_snapshot_lock.py"),
        Path("script/experiment/fpct_e1_k8s_lock_bundle.py"),
        ANALYZER_RELATIVE,
        Path("rosetta/model/wrapper.py"),
        Path("rosetta/model/fpct_attention.py"),
        Path("rosetta/model/fpct_instrumentation.py"),
        Path("rosetta/model/aligner.py"),
        Path("rosetta/train/dataset_adapters.py"),
        Path("rosetta/utils/evaluate.py"),
        Path("rosetta/utils/model_loading.py"),
        Path("script/evaluation/unified_evaluator.py"),
        Path("test/test_fpct_e1_capture_runner.py"),
        Path("test/test_fpct_e1_prepare_input_lock.py"),
        Path("test/test_fpct_e1_runtime_backend.py"),
        Path("test/test_fpct_e1_runtime_probe.py"),
        Path("test/test_fpct_e1_runtime_probe_k8s.py"),
        Path("test/test_fpct_e1_source_snapshot_lock.py"),
        Path("test/test_fpct_e1_k8s_lock_bundle.py"),
        Path("recipe/k8s/fpct_e1/runtime_probe_job.yaml"),
        K8S_CAPTURE_TEMPLATE_RELATIVE,
        Path("test/test_fpct_e1_runtime_capture_primitives.py"),
        Path("test/test_fpct_instrumentation.py"),
        Path("test/test_fpct_e1_mechanism_audit.py"),
        E1_MANIFEST_RELATIVE,
        STREAMING_CONTRACT_RELATIVE,
        STREAMING_SCHEMA_RELATIVE,
        STREAMING_SYNTHETIC_GATE_RELATIVE,
        Path("script/analysis/fpct_e1_streaming_verify.py"),
        Path("test/test_fpct_e1_streaming.py"),
        MECHANISM_SCHEMA_RELATIVE,
    ]
    source_files = [{"logical_path": _logical("repo", p), "sha256": sha256_file(repo_root / p)} for p in source_paths]
    k8s_template_lock = {
        "logical_path": _logical("repo", K8S_CAPTURE_TEMPLATE_RELATIVE),
        "sha256": sha256_file(repo_root / K8S_CAPTURE_TEMPLATE_RELATIVE),
        "bytes": (repo_root / K8S_CAPTURE_TEMPLATE_RELATIVE).stat().st_size,
    }
    if source_snapshot_receipt is None:
        source_receipt = {
            "logical_path": "repo://__SOURCE_SNAPSHOT_RECEIPT__",
            "sha256": "__SOURCE_SNAPSHOT_RECEIPT_SHA256__",
            "bytes": -1,
            "verification": {"status": "UNRESOLVED"},
        }
    else:
        try:
            receipt_relative = source_snapshot_receipt.resolve().relative_to(repo_root)
        except ValueError as error:
            raise ValueError("source snapshot receipt must live inside the archive root") from error
        from script.experiment.fpct_e1_source_snapshot_lock import (
            verify_source_snapshot_receipt,
        )

        receipt_verification = verify_source_snapshot_receipt(
            source_snapshot_receipt, repo_root, execution_sha,
        )
        source_receipt = {
            "logical_path": _logical("repo", receipt_relative),
            "sha256": sha256_file(source_snapshot_receipt),
            "bytes": source_snapshot_receipt.stat().st_size,
            "verification": receipt_verification,
        }
    source_snapshot = {
        "logical_path": "repo://.",
        "host_path": str(repo_root),
        "container_path": repo_container,
        "tree": _tree_manifest(repo_root),
        "execution_sha": execution_sha,
        "construction": "clean_git_archive",
        "read_only_required": True,
        "worktree_mount_forbidden": True,
        "git_receipt": source_receipt,
        "canonical_tree_sha256": source_receipt.get("verification", {}).get(
            "mounted_tree_canonical_sha256", "__SOURCE_SNAPSHOT_CANONICAL_TREE_SHA256__"
        ),
    }
    prepared_identity = input_lock["prepared"].get("execution_identity")
    if input_lock["prepared"].get("status") == INPUT_LOCK_STATUS:
        frozen_receipt = prepared_identity.get("source_snapshot_receipt", {}) if isinstance(prepared_identity, Mapping) else {}
        if (
            not isinstance(prepared_identity, Mapping)
            or prepared_identity.get("execution_sha") != execution_sha
            or prepared_identity.get("source_snapshot_root") != str(repo_root.absolute())
            or frozen_receipt.get("file_sha256") != source_receipt.get("sha256")
            or frozen_receipt.get("bytes") != source_receipt.get("bytes")
            or prepared_identity.get("historical_execution_resume_allowed") is not False
            or prepared_identity.get("historical_artifact_reuse_allowed") is not False
        ):
            raise ValueError("A4 input lock execution identity differs from the new source snapshot")
    backend = {"spec": backend_spec, "source": {"logical_path": _logical("repo", backend_source.relative_to(repo_root)) if backend_source else "repo://__BACKEND_SOURCE__", "sha256": sha256_file(backend_source) if backend_source else "__BACKEND_SOURCE_SHA256__"}}
    runtime = {"logical_path": _logical("runtime", runtime_provenance.name) if runtime_provenance else "runtime://__RUNTIME_PROVENANCE__", "sha256": sha256_file(runtime_provenance) if runtime_provenance else "__RUNTIME_PROVENANCE_SHA256__"}
    if runtime_provenance is not None:
        from script.experiment.fpct_e1_runtime_probe import validate_runtime_probe

        runtime_payload = _read_json(runtime_provenance)
        validate_runtime_probe(runtime_payload)
        if (
            runtime_payload["execution_sha"] != execution_sha
            or runtime_payload["image_digest"] != image_digest
            or runtime_payload["source_snapshot_tree_sha"] != source_snapshot["canonical_tree_sha256"]
        ):
            raise ValueError("runtime probe differs from execution/image/source snapshot lock")
        expected_source_verification = {
            **source_receipt["verification"],
            "receipt_file_sha256": source_receipt["sha256"],
            "receipt_bytes": source_receipt["bytes"],
        }
        if runtime_payload.get("source_snapshot_verification") != expected_source_verification:
            raise ValueError("runtime probe source receipt evidence differs from plan input")
        runtime["identity"] = {
            "schema_version": runtime_payload["schema_version"],
            "protocol_id": runtime_payload["protocol_id"],
            "status": runtime_payload["status"],
            "execution_sha": runtime_payload["execution_sha"],
            "image_digest": runtime_payload["image_digest"],
            "source_snapshot_tree_sha": runtime_payload["source_snapshot_tree_sha"],
            "source_snapshot_verification": expected_source_verification,
        }

    def shard(stage: str, seed: int, checkpoint_arm: str, operator: str, task: str, value: float) -> dict[str, Any]:
        cell = CELL_MAP[(checkpoint_arm, operator)]
        checkpoint_id = f"{seed}-{checkpoint_arm}"
        baseline_cell = "Y_CC" if checkpoint_arm == "c_post_trained" else "Y_FC"
        baseline_id = _shard_id(STAGE_E1_2, seed, baseline_cell, task, 0.0)
        sid = _shard_id(stage, seed, cell, task, value)
        return {
            "shard_id": sid, "stage": stage,
            "phase": (
                PHASE_E1_2_BASELINES if stage == STAGE_E1_2 and operator == "c_post"
                else PHASE_E1_2_FACTORIZED if stage == STAGE_E1_2
                else STAGE_E1_3
            ),
            "split_role": ALLOWED_SPLIT_ROLE, "seed": seed,
            "checkpoint_arm": checkpoint_arm, "checkpoint_id": checkpoint_id,
            "checkpoint_logical_path": checkpoint_by_id[checkpoint_id]["final"]["logical_path"],
            "checkpoint_tree_sha256": checkpoint_by_id[checkpoint_id]["final"]["tree_sha256"],
            "inference_operator": operator, "cell": cell, "task": task, "lambda_value": value,
            "lambda_tag": _lambda_tag(value), "expected_content_group_count": TASK_GROUP_COUNTS[task],
            "membership_sha256": input_lock["prepared"].get("task_contract", {}).get(task, {}).get("membership_sha256", _membership_sha(group_contract[task])),
            "expected_row_template": input_lock["prepared"].get("task_contract", {}).get(task, {}).get("row_template", {"count": -1, "sha256": "__EXPECTED_ROW_TEMPLATE_SHA256__"}),
            "e0_config_logical_path": _logical("repo", configs[(seed, cell, task)]["relative"]),
            "e0_config_sha256": configs[(seed, cell, task)]["sha256"],
            "depends_on_shard": None if operator == "c_post" else baseline_id,
            "requires_phase_completion": (
                None if stage == STAGE_E1_2 and operator == "c_post"
                else PHASE_E1_2_BASELINES if stage == STAGE_E1_2
                else FINALIZED_E1_2
            ),
            "output_relative": f"{stage}/{seed}/{cell}/{task}/{_lambda_tag(value)}",
        }

    shards = []
    for seed in SEEDS:
        for arm in CHECKPOINT_ARMS:
            for operator in ("c_post", "f"):
                for task in TASKS:
                    shards.append(shard(STAGE_E1_2, seed, arm, operator, task, ENDPOINT_LAMBDAS[operator]))
    for seed in SEEDS:
        for arm in CHECKPOINT_ARMS:
            for task in TASKS:
                for value in SWEEP_ADDITIONS:
                    shards.append(shard(STAGE_E1_3, seed, arm, "f", task, value))

    path_roots = {
        "repo": {"host": str(repo_root), "container": repo_container},
        "e0": {"host": str(e0_root), "container": e0_container},
        "output": {"host": str(output_root), "container": output_container},
        "gate": {"host": str(instrumentation_gate_path.parent.absolute()) if instrumentation_gate_path else "__GATE_HOST_ROOT__", "container": locals().get("gate_container", "/opt/fpct-e1-lock/consolidated_gate.json").rsplit("/", 1)[0]},
        "runtime": {"host": str(runtime_provenance.parent.absolute()) if runtime_provenance else "__RUNTIME_HOST_ROOT__", "container": "/opt/fpct-e1-runtime"},
        "input_lock": {"host": str(input_lock_manifest.parent.absolute()) if input_lock_manifest else "__INPUT_LOCK_HOST_ROOT__", "container": input_lock_container},
        "raw_topology": {"host": str(raw_topology_artifact_root.absolute()) if raw_topology_artifact_root else "__RAW_TOPOLOGY_HOST_ROOT__", "container": raw_topology_container},
        "models": {"host": models_host_root, "container": models_container},
    }
    _validate_output_root_isolation(path_roots)
    plan_status = (
        "PREPARED_NO_MODEL_LOAD"
        if input_lock["prepared"].get("status") == INPUT_LOCK_STATUS
        else "DRAFT_A4_INPUT_LOCK_REQUIRED"
    )
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "protocol_id": PROTOCOL_ID, "status": plan_status,
        "a4_operative_lock": a4_operative_lock,
        "split_role": ALLOWED_SPLIT_ROLE, "path_roots": path_roots,
        "source_sha256": {_logical("repo", SPLIT_MANIFEST_RELATIVE): split["sha256"], _logical("repo", E0_DEV_MANIFEST_RELATIVE): dev["sha256"], _logical("repo", E0_CONFIG_INDEX_RELATIVE): sha256_file(repo_root / E0_CONFIG_INDEX_RELATIVE)},
        "source_files": source_files, "source_snapshot": source_snapshot, "instrumentation_gate": gate,
        "k8s_template": k8s_template_lock,
        "runtime_lock": {"execution_sha": execution_sha, "image_digest": image_digest, "backend": backend, "runtime_provenance": runtime},
        "checkpoint_count": len(checkpoints), "checkpoints": checkpoints, "input_lock": input_lock,
        "raw_topology_lock": raw_topology,
        "stages": {
            STAGE_E1_2: {
                "physical_shards": EXPECTED_ENDPOINT_SHARDS,
                "lambda_values": [0.0, 1.0],
                "phases": [
                    {"phase": PHASE_E1_2_BASELINES, "physical_shards": 18, "depends_on": None},
                    {"phase": PHASE_E1_2_FACTORIZED, "physical_shards": 18, "depends_on": PHASE_E1_2_BASELINES},
                ],
                "expected_rows_by_task": e1_2_rows_by_task,
                "expected_total_rows": sum(e1_2_rows_by_task.values()),
                "bytes": "OBSERVED_AFTER_CAPTURE_NO_POSTHOC_GATE",
            },
            STAGE_E1_3: {
                "physical_shards": EXPECTED_SWEEP_SHARDS,
                "lambda_values": list(SWEEP_ADDITIONS),
                "depends_on": FINALIZED_E1_2,
                "lambda_zero_role": "F-runtime exact-collapse control; not substituted by C_post",
                "full_grid_after_closure": list(FULL_LAMBDA_GRID),
                "addition_expected_rows_by_task": e1_3_addition_rows_by_task,
                "addition_expected_total_rows": sum(e1_3_addition_rows_by_task.values()),
                "full_grid_expected_rows_by_task": full_grid_rows_by_task,
                "full_grid_expected_total_rows": sum(full_grid_rows_by_task.values()),
                "final_108_shard_expected_rows_by_task": final_108_rows_by_task,
                "final_108_shard_expected_total_rows": sum(final_108_rows_by_task.values()),
                "bytes": "OBSERVED_AFTER_CAPTURE_NO_POSTHOC_GATE",
            },
        },
        "shard_count": len(shards), "shards": shards, "group_contract": group_contract,
        "teacher_forcing": {"capture_mode": "teacher_forced_response", "eligible_query": "labels[:,t+1]!=-100", "causal_shift": "logits[:,t,:] score labels[:,t+1]", "include_response": False, "gold_response_template": "The correct answer is {answer}."},
        "schema_boundary": {
            "backend_contract_version": CAPTURE_BACKEND_CONTRACT_VERSION,
            "backend_forbidden_prefix": "cpost_",
            "final_columns": "dynamic audit.INPUT_COLUMNS",
            "baseline_join": "all cpost_* values are mechanically joined from the immutable C_post capture artifact using the original scientific row key",
            "baseline_join_excludes": [
                "row_ordinal", "logical_row_id", "endpoint_id", "endpoint_row_id",
            ],
            "physical_chunk_rows": PARQUET_BATCH_ROWS,
            "stream_order": "sample_sha256,row_ordinal",
            "stores_raw_kv": False,
        },
        "firewall": {"e0_design": "OPEN", "e1_pilot": "SEALED_NOT_RUN_NOT_READ", "model_selection": "SEALED", "test": "SEALED", "confirmatory": "SEALED"},
        "model_or_tokenizer_loaded_during_prepare": False, "natural_output_accessed_during_prepare": False,
    }
    plan["plan_sha256"] = _plan_hash(plan)
    validate_execution_plan(plan, require_checkpoints=require_checkpoints, for_execution=False)
    return plan


def _audit_columns() -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    from script.analysis.fpct_e1_mechanism_audit import INPUT_COLUMNS
    final = tuple(INPUT_COLUMNS)
    baseline = tuple(name for name in final if name.startswith("cpost_"))
    backend = tuple(name for name in final if not name.startswith("cpost_"))
    if not baseline:
        raise RuntimeError("audit schema has no mechanically joined cpost_* columns")
    for name in baseline:
        if name.removeprefix("cpost_") not in backend:
            raise RuntimeError(f"baseline source column absent from audit schema: {name}")
    return backend, final, baseline


def _row_key_columns(final_columns: Sequence[str]) -> tuple[str, ...]:
    columns = tuple(name for name in BASE_ROW_KEY_CANDIDATES if name in final_columns)
    required = {"seed", "checkpoint_arm", "task", "sample_sha256", "content_group_sha256", "layer", "query_position", "target_position", "parent_position", "candidate_count"}
    if not required.issubset(columns) or not ({"head", "query_head"} & set(columns)):
        raise RuntimeError("audit schema cannot form the frozen row-key universe")
    forbidden = {"row_ordinal", "logical_row_id", "endpoint_id", "endpoint_row_id"}
    if forbidden.intersection(columns):
        raise RuntimeError("endpoint/physical identity leaked into the C_post baseline join")
    return columns


def validate_execution_plan(plan: Mapping[str, Any], *, require_checkpoints: bool = False, for_execution: bool = False) -> None:
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("protocol_id") != PROTOCOL_ID or plan.get("plan_sha256") != _plan_hash(plan):
        raise ValueError("capture plan identity/hash mismatch")
    if plan.get("split_role") != ALLOWED_SPLIT_ROLE:
        raise ValueError("capture plan is not E0-design")
    expected_status = (
        "PREPARED_NO_MODEL_LOAD"
        if plan.get("input_lock", {}).get("prepared", {}).get("status") == INPUT_LOCK_STATUS
        else "DRAFT_A4_INPUT_LOCK_REQUIRED"
    )
    if plan.get("status") != expected_status:
        raise ValueError("capture plan status differs from its A4 input-lock state")
    if plan.get("firewall", {}).get("e1_pilot") != "SEALED_NOT_RUN_NOT_READ":
        raise ValueError("E1-pilot firewall is open")
    operative = plan.get("a4_operative_lock", {})
    if (
        operative.get("status") != "GO_NEW_A4_SUCCESSOR_DAG_ONLY"
        or operative.get("protocol_id") != A4_PROTOCOL_ID
        or operative.get("historical_execution_resume_allowed") is not False
        or operative.get("historical_artifact_reuse_allowed") is not False
        or operative.get("execution_authorization") != {
            "a4_successor_input_lock_operative": True,
            "a4_successor_dag_operative": True,
            "requires_streaming_input_lock_go": True,
            "historical_attempt_resume_allowed": False,
            "e1_pilot_forward_or_outcome": False,
            "training": False,
        }
        or set(operative.get("source", {}))
        != {"manifest", "contract", "schema", "synthetic_gate"}
        or any(
            not _is_sha256(record.get("sha256"))
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] <= 0
            for record in operative.get("source", {}).values()
        )
    ):
        raise ValueError("capture plan lacks the explicit operative A4 successor lock")
    _validate_output_root_isolation(plan.get("path_roots", {}))
    checkpoints = plan.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 6:
        raise ValueError("plan must bind six checkpoints")
    if {(r.get("seed"), r.get("checkpoint_arm")) for r in checkpoints} != {(s, a) for s in SEEDS for a in CHECKPOINT_ARMS}:
        raise ValueError("checkpoint universe changed")
    shards = plan.get("shards")
    if not isinstance(shards, list) or len(shards) != EXPECTED_SHARD_COUNT:
        raise ValueError(f"plan must contain {EXPECTED_SHARD_COUNT} physical shards")
    if len({r.get("shard_id") for r in shards}) != EXPECTED_SHARD_COUNT or any("lambda-" not in str(r.get("shard_id")) for r in shards):
        raise ValueError("shard IDs are not unique/lambda-bound")
    endpoint = [r for r in shards if r.get("stage") == STAGE_E1_2]
    sweep = [r for r in shards if r.get("stage") == STAGE_E1_3]
    if len(endpoint) != EXPECTED_ENDPOINT_SHARDS or len(sweep) != EXPECTED_SWEEP_SHARDS:
        raise ValueError("two-stage shard graph count mismatch")
    baselines = [r for r in endpoint if r.get("phase") == PHASE_E1_2_BASELINES]
    factorized = [r for r in endpoint if r.get("phase") == PHASE_E1_2_FACTORIZED]
    if len(baselines) != 18 or len(factorized) != 18:
        raise ValueError("E1-2 baseline/factorized phase split must be 18+18")
    ids = {r["shard_id"] for r in shards}
    checkpoint_ids = {r["checkpoint_id"] for r in checkpoints}
    for shard in shards:
        if shard.get("split_role") != ALLOWED_SPLIT_ROLE or shard.get("checkpoint_id") not in checkpoint_ids:
            raise ValueError("shard firewall/checkpoint mismatch")
        arm, operator = CELL_SPEC[shard["cell"]]
        if (shard["checkpoint_arm"], shard["inference_operator"]) != (arm, operator):
            raise ValueError("cell mapping mismatch")
        if shard["stage"] == STAGE_E1_2 and shard["lambda_value"] != ENDPOINT_LAMBDAS[operator]:
            raise ValueError("endpoint lambda mismatch")
        if shard["stage"] == STAGE_E1_3 and (operator != "f" or shard["lambda_value"] not in SWEEP_ADDITIONS or shard.get("requires_phase_completion") != FINALIZED_E1_2):
            raise ValueError("sweep contract mismatch")
        if shard["stage"] == STAGE_E1_2:
            expected_phase = PHASE_E1_2_BASELINES if operator == "c_post" else PHASE_E1_2_FACTORIZED
            expected_gate = None if operator == "c_post" else PHASE_E1_2_BASELINES
            if shard.get("phase") != expected_phase or shard.get("requires_phase_completion") != expected_gate:
                raise ValueError("E1-2 baseline/factorized phase dependency mismatch")
        dependency = shard.get("depends_on_shard")
        if operator == "c_post":
            if dependency is not None:
                raise ValueError("C_post baseline cannot depend on itself")
        elif dependency not in ids:
            raise ValueError("baseline shard dependency missing")
    for task in TASKS:
        groups = plan.get("group_contract", {}).get(task)
        if not isinstance(groups, list) or len(groups) != TASK_GROUP_COUNTS[task]:
            raise ValueError("group contract changed")
        runtime_fields = (
            "source_row_id", "subject", "evaluation_subject",
            "evaluation_question_id", "rendered_prompt_sha256",
            "prompt_alignment_sha256", "certified_candidate_count", "eligibility",
        )
        if any(any(name not in row for name in runtime_fields) for row in groups):
            raise ValueError("group contract lacks frozen runtime provenance")
        if any(
            not _is_sha256(row["rendered_prompt_sha256"])
            or not _is_sha256(row["prompt_alignment_sha256"])
            for row in groups
        ):
            raise ValueError("group runtime provenance SHA is malformed")
    prepared_contract = plan.get("input_lock", {}).get("prepared", {})
    if prepared_contract.get("status") == INPUT_LOCK_STATUS:
        streaming = prepared_contract.get("streaming_contract", {})
        if (
            streaming.get("protocol_id") != A4_PROTOCOL_ID
            or streaming.get("physical_chunk_rows") != PARQUET_BATCH_ROWS
            or streaming.get("schema_sha256")
            != operative["source"]["schema"]["sha256"]
            or streaming.get("synthetic_gate_sha256")
            != operative["source"]["synthetic_gate"]["sha256"]
        ):
            raise ValueError("plan input lock differs from the operative A4 lock")
        expected_by_task = {
            task: {
                "count": prepared_contract["task_contract"][task]["expected_long_form_rows"]["count"],
                "sum": prepared_contract["task_contract"][task]["expected_long_form_rows"]["sum"],
            }
            for task in TASKS
        }
        if prepared_contract.get("expected_long_form_rows_by_task") != expected_by_task:
            raise ValueError("plan row-volume task contract changed")
        task_sums = {task: record["sum"] for task, record in expected_by_task.items()}
        e1_2 = plan.get("stages", {}).get(STAGE_E1_2, {})
        e1_3 = plan.get("stages", {}).get(STAGE_E1_3, {})
        expected_e1_2 = {task: 12 * task_sums[task] for task in TASKS}
        expected_additions = {task: 24 * task_sums[task] for task in TASKS}
        expected_full = {task: 30 * task_sums[task] for task in TASKS}
        expected_final_108 = {task: 36 * task_sums[task] for task in TASKS}
        if (
            e1_2.get("expected_rows_by_task") != expected_e1_2
            or e1_2.get("expected_total_rows") != sum(expected_e1_2.values())
            or e1_3.get("addition_expected_rows_by_task") != expected_additions
            or e1_3.get("addition_expected_total_rows") != sum(expected_additions.values())
            or e1_3.get("full_grid_expected_rows_by_task") != expected_full
            or e1_3.get("full_grid_expected_total_rows") != sum(expected_full.values())
            or e1_3.get("final_108_shard_expected_rows_by_task") != expected_final_108
            or e1_3.get("final_108_shard_expected_total_rows") != sum(expected_final_108.values())
        ):
            raise ValueError("plan stage row-volume arithmetic changed")
    if require_checkpoints:
        for record in checkpoints:
            if _contains_unresolved(record):
                raise RuntimeError("checkpoint lock unresolved")
    if for_execution:
        if _contains_unresolved(plan):
            raise RuntimeError("execution plan contains unresolved render placeholders")
        lock = plan.get("runtime_lock", {})
        if not EXECUTION_SHA.fullmatch(str(lock.get("execution_sha", ""))) or not IMAGE_DIGEST.fullmatch(str(lock.get("image_digest", ""))):
            raise RuntimeError("execution SHA/image digest is not immutable")
        gate = plan.get("instrumentation_gate", {})
        if gate.get("status") != "GO" or not _is_sha256(gate.get("sha256")):
            raise RuntimeError("instrumentation gate is not exactly frozen")
        prepared = plan.get("input_lock", {}).get("prepared", {})
        if prepared.get("status") != INPUT_LOCK_STATUS:
            raise RuntimeError("prepared CPU input lock is not frozen")
        raw_topology = plan.get("raw_topology_lock", {})
        if (
            raw_topology.get("status") != "GO_MODEL_OUTPUT_FREE"
            or not _is_sha256(raw_topology.get("manifest", {}).get("sha256"))
            or raw_topology.get("input_lock_sidecar_sha256") != prepared.get("sidecar", {}).get("sha256")
            or not isinstance(raw_topology.get("row_count"), int)
            or raw_topology.get("row_count") < 0
        ):
            raise RuntimeError("raw-topology pre-shard lock is not frozen")
        snapshot = plan.get("source_snapshot", {})
        if snapshot.get("execution_sha") != lock["execution_sha"] or snapshot.get("construction") != "clean_git_archive" or snapshot.get("read_only_required") is not True or snapshot.get("worktree_mount_forbidden") is not True:
            raise RuntimeError("clean read-only source snapshot is not bound to execution SHA")
        receipt = snapshot.get("git_receipt", {})
        verification = receipt.get("verification", {}) if isinstance(receipt, Mapping) else {}
        if (
            not _is_sha256(receipt.get("sha256"))
            or verification.get("status") != "GO_MOUNTED_SOURCE_SNAPSHOT"
            or verification.get("execution_sha") != lock["execution_sha"]
            or snapshot.get("canonical_tree_sha256") != verification.get("mounted_tree_canonical_sha256")
        ):
            raise RuntimeError("source snapshot lacks a verified Git-archive receipt")


def _verify_tree(path: Path, frozen: Mapping[str, Any]) -> None:
    observed = _tree_manifest(path)
    if observed["tree_sha256"] != frozen.get("tree_sha256") or observed["files"] != frozen.get("files"):
        raise ValueError(f"checkpoint tree provenance changed: {path}")


def _verify_raw_topology_sources(plan: Mapping[str, Any], view: str) -> None:
    prepared = plan["input_lock"]["prepared"]
    raw_topology = plan["raw_topology_lock"]
    raw_manifest_path = resolve_logical_path(
        plan, raw_topology["manifest"]["logical_path"], view
    )
    if (
        not raw_manifest_path.is_file()
        or sha256_file(raw_manifest_path) != raw_topology["manifest"]["sha256"]
        or raw_manifest_path.stat().st_size != raw_topology["manifest"]["bytes"]
    ):
        raise ValueError("raw-topology manifest provenance changed")
    raw_manifest = _read_json(raw_manifest_path)
    if raw_manifest.get("artifacts") != raw_topology["artifacts"]:
        raise ValueError("raw-topology artifact contract changed")
    raw_root = resolve_logical_path(plan, raw_topology["root"], view)
    for filename, record in raw_topology["artifacts"].items():
        path = raw_root / filename
        if (
            not path.is_file()
            or sha256_file(path) != record.get("sha256")
            or path.stat().st_size != record.get("bytes")
        ):
            raise ValueError(f"raw-topology artifact provenance changed: {filename}")
    from script.analysis.fpct_e1_mechanism_audit import verify_raw_topology_artifacts
    verified_raw = verify_raw_topology_artifacts(
        resolve_logical_path(plan, prepared["sidecar"]["logical_path"], view),
        raw_root,
    )
    if (
        verified_raw.get("status") != "GO_MODEL_OUTPUT_FREE"
        or verified_raw.get("row_count") != raw_topology["row_count"]
    ):
        raise ValueError("raw-topology semantic verification changed")


def verify_plan_sources(plan: Mapping[str, Any], view: str = "host", checkpoint_id: str | None = None) -> None:
    snapshot = plan["source_snapshot"]
    snapshot_root = resolve_logical_path(plan, snapshot["logical_path"], view)
    _verify_tree(snapshot_root, snapshot["tree"])
    receipt_record = snapshot["git_receipt"]
    receipt_path = resolve_logical_path(plan, receipt_record["logical_path"], view)
    if (
        not receipt_path.is_file()
        or sha256_file(receipt_path) != receipt_record["sha256"]
        or receipt_path.stat().st_size != receipt_record["bytes"]
    ):
        raise ValueError("source snapshot Git receipt bytes changed")
    from script.experiment.fpct_e1_source_snapshot_lock import (
        verify_source_snapshot_receipt,
    )
    observed_receipt = verify_source_snapshot_receipt(
        receipt_path, snapshot_root, snapshot["execution_sha"],
    )
    if observed_receipt != receipt_record["verification"]:
        raise ValueError("source snapshot Git receipt verification changed")
    for logical, expected in plan.get("source_sha256", {}).items():
        path = resolve_logical_path(plan, logical, view)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"source provenance changed: {logical}")
    for record in plan.get("source_files", []):
        path = resolve_logical_path(plan, record["logical_path"], view)
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"source-code provenance changed: {record['logical_path']}")
    operative = plan["a4_operative_lock"]
    for name, record in operative["source"].items():
        path = resolve_logical_path(plan, record["logical_path"], view)
        if (
            not path.is_file()
            or sha256_file(path) != record["sha256"]
            or path.stat().st_size != record["bytes"]
        ):
            raise ValueError(f"operative A4 source lock changed: {name}")
    for shard in plan["shards"]:
        path = resolve_logical_path(plan, shard["e0_config_logical_path"], view)
        if not path.is_file() or sha256_file(path) != shard["e0_config_sha256"]:
            raise ValueError("E0 config provenance changed")
    records = plan["checkpoints"] if checkpoint_id is None else [r for r in plan["checkpoints"] if r["checkpoint_id"] == checkpoint_id]
    for record in records:
        _verify_tree(resolve_logical_path(plan, record["final"]["logical_path"], view), record["final"])
        _verify_tree(resolve_logical_path(plan, record["step64"]["logical_path"], view), record["step64"])
    input_lock = plan["input_lock"]
    _verify_tree(resolve_logical_path(plan, input_lock["data_root"], view), input_lock["tree"])
    manifest_path = resolve_logical_path(plan, input_lock["manifest"]["logical_path"], view)
    if sha256_file(manifest_path) != input_lock["manifest"]["sha256"]:
        raise ValueError("E0 dev-data manifest SHA changed")
    declared = _e0_declared_tree_sha256(resolve_logical_path(plan, input_lock["data_root"], view))
    if declared != input_lock["manifest"]["declared_tree_sha256"] or declared != input_lock["manifest"]["recomputed_declared_tree_sha256"]:
        raise ValueError("E0 dev-data declared tree SHA no longer closes")
    prepared = input_lock["prepared"]
    for name in ("manifest", "sidecar"):
        record = prepared[name]
        path = resolve_logical_path(plan, record["logical_path"], view)
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"prepared input-lock {name} SHA changed")
        if name == "sidecar" and path.stat().st_size != record["bytes"]:
            raise ValueError("prepared input-lock sidecar size changed")
    for name, record in prepared["streaming_contract"]["geometry_lock"].items():
        path = resolve_logical_path(plan, record["logical_path"], view)
        if (
            not path.is_file()
            or sha256_file(path) != record["sha256"]
            or path.stat().st_size != record["bytes"]
        ):
            raise ValueError(f"prepared compact-geometry artifact changed: {name}")
    for name in ("index", "receipt"):
        record = prepared["streaming_contract"]["streaming_template_lock"][name]
        path = resolve_logical_path(plan, record["logical_path"], view)
        if (
            not path.is_file()
            or sha256_file(path) != record["sha256"]
            or path.stat().st_size != record["bytes"]
        ):
            raise ValueError(f"prepared streaming-template artifact changed: {name}")
    # The top-level index and receipt are only a directory entry point.  Every
    # run-shard invocation must replay the complete immutable template tree
    # before the backend module is imported (and therefore before a model or
    # checkpoint can be loaded).  This deliberately repeats the input-lock
    # publication check in the mounted execution view; trusting only the two
    # top-level JSON files would leave nested manifests/chunks replaceable.
    _verify_streaming_template_tree(plan, view)
    _verify_raw_topology_sources(plan, view)
    backend = plan["runtime_lock"]["backend"]["source"]
    path = resolve_logical_path(plan, backend["logical_path"], view)
    if sha256_file(path) != backend["sha256"]:
        raise ValueError("backend source SHA changed")
    runtime = plan["runtime_lock"]["runtime_provenance"]
    path = resolve_logical_path(plan, runtime["logical_path"], view)
    if sha256_file(path) != runtime["sha256"]:
        raise ValueError("runtime provenance SHA changed")
    from script.experiment.fpct_e1_runtime_probe import validate_runtime_probe

    runtime_payload = _read_json(path)
    validate_runtime_probe(runtime_payload)
    expected_runtime_identity = {
        "schema_version": runtime_payload["schema_version"],
        "protocol_id": runtime_payload["protocol_id"],
        "status": runtime_payload["status"],
        "execution_sha": plan["runtime_lock"]["execution_sha"],
        "image_digest": plan["runtime_lock"]["image_digest"],
        "source_snapshot_tree_sha": plan["source_snapshot"]["canonical_tree_sha256"],
        "source_snapshot_verification": {
            **observed_receipt,
            "receipt_file_sha256": sha256_file(receipt_path),
            "receipt_bytes": receipt_path.stat().st_size,
        },
    }
    if runtime.get("identity") != expected_runtime_identity or any(
        runtime_payload[name] != expected_runtime_identity[name]
        for name in (
            "execution_sha", "image_digest", "source_snapshot_tree_sha",
            "source_snapshot_verification",
        )
    ):
        raise ValueError("runtime probe identity changed")


def _load_backend(specification: str) -> Callable[[Mapping[str, Any]], Any]:
    if ":" not in specification:
        raise ValueError("backend must be module:callable")
    module, attribute = specification.split(":", 1)
    callback = getattr(importlib.import_module(module), attribute)
    if not callable(callback):
        raise TypeError("backend is not callable")
    return callback


@dataclass(frozen=True)
class TeacherForcedObservation:
    capture_report: Mapping[str, Any]
    logits: Any
    loss: Any
    gold: Mapping[str, Any]
    teacher_forcing: Mapping[str, Any]


def teacher_forced_capture(model: Any, forward_inputs: Mapping[str, Any], labels: Any, *, metadata: Mapping[str, Any]) -> TeacherForcedObservation:
    import torch
    if labels.ndim != 2:
        raise ValueError("labels must be [B,T]")
    expected = torch.zeros_like(labels, dtype=torch.bool)
    expected[:, :-1] = labels[:, 1:] != -100
    for name in ("begin_fpct_capture", "end_fpct_capture", "fpct_teacher_forced_query_mask"):
        if not callable(getattr(model, name, None)):
            raise RuntimeError(f"model lacks explicit FPCT capture API: {name}")
    model_mask = model.fpct_teacher_forced_query_mask(labels)
    if not torch.equal(model_mask.detach().cpu(), expected.detach().cpu()):
        raise RuntimeError("teacher-forced query mask violates causal shift")
    model.begin_fpct_capture(mode="teacher_forced_response", metadata=dict(metadata), query_mask=model_mask)
    outputs = model(**dict(forward_inputs), labels=labels)
    report = model.end_fpct_capture()
    if report.get("stores_raw_kv") is not False:
        raise RuntimeError("capture must attest stores_raw_kv=false")
    logits = outputs.logits
    shifted, eligible = labels[:, 1:], labels[:, 1:] != -100
    batch, query = torch.where(eligible)
    target, token = query + 1, shifted[batch, query]
    gold_logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)[batch, query, token]
    return TeacherForcedObservation(report, logits, getattr(outputs, "loss", None), {"batch_index": batch, "query_position": query, "target_position": target, "target_token_id": token, "gold_logp": gold_logp}, {"mode": "teacher_forced_response", "causal_shift_verified": True, "eligible_query_count": int(eligible.sum()), "query_mask_sha256": sha256_bytes(expected.detach().cpu().numpy().tobytes()), "stores_raw_kv": False})


def capture_report_long_rows(report: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    if report.get("stores_raw_kv") is not False:
        raise RuntimeError("stores_raw_kv must be false")
    if report.get("long_form_contract_version") != 1:
        raise RuntimeError("capture API long-form contract version mismatch")
    if report.get("long_form_incomplete_chunk_count") != 0:
        raise RuntimeError("capture API has incomplete long-form chunks")
    primitives = report.get("long_form_primitives")
    if not isinstance(primitives, list):
        raise RuntimeError("capture API must provide bounded long_form_primitives")
    return primitives


def _validate_stable_backend_attestation(
    attestation: Mapping[str, Any], plan: Mapping[str, Any],
    shard: Mapping[str, Any],
) -> dict[str, Any]:
    """Fully validate persisted backend provenance before backend/model load."""

    if not isinstance(attestation, Mapping):
        raise ValueError("stable backend attestation is not an object")
    try:
        canonical_json_bytes(attestation)
    except (TypeError, ValueError) as error:
        raise ValueError("stable backend attestation is not canonical finite JSON") from error
    expected_keys = {
        "split_role", "capture_mode", "causal_shift_verified", "stores_raw_kv",
        "long_form_contract_version", "plan_sha256", "shard_id",
        "execution_sha", "image_digest", "checkpoint_tree_sha256",
        "membership_sha256", "gold_response_template",
        "gold_response_template_sha256", "correctness_semantics",
        "backend_supplied_cpost_fields", "runtime_assets", "projector_load",
        "expected_long_form_rows", "e1_pilot_consumed",
        "model_selection_consumed", "test_consumed",
    }
    if set(attestation) != expected_keys:
        raise ValueError("stable backend attestation schema/unknown keys changed")
    forbidden_resume = {
        "resume_cursor_sha256", "resume_completed_logical_rows",
        "resume_next_sample_sha256", "resume_next_row_ordinal",
    }
    if forbidden_resume.intersection(attestation):
        raise ValueError("stable backend attestation contains attempt-local resume fields")
    exact = {
        "split_role": ALLOWED_SPLIT_ROLE,
        "capture_mode": "teacher_forced_response",
        "causal_shift_verified": True,
        "stores_raw_kv": False,
        "plan_sha256": plan["plan_sha256"],
        "shard_id": shard["shard_id"],
        "execution_sha": plan["runtime_lock"]["execution_sha"],
        "image_digest": plan["runtime_lock"]["image_digest"],
        "checkpoint_tree_sha256": shard["checkpoint_tree_sha256"],
        "membership_sha256": shard["membership_sha256"],
        "long_form_contract_version": 2,
        "gold_response_template": GOLD_RESPONSE_TEMPLATE,
        "gold_response_template_sha256": sha256_bytes(
            GOLD_RESPONSE_TEMPLATE.encode("utf-8")
        ),
        "correctness_semantics": CORRECTNESS_SEMANTICS,
        "backend_supplied_cpost_fields": False,
        "e1_pilot_consumed": False,
        "model_selection_consumed": False,
        "test_consumed": False,
    }
    for name, expected in exact.items():
        if attestation.get(name) != expected:
            raise ValueError(f"stable backend attestation mismatch: {name}")
    frozen_assets = plan["input_lock"]["prepared"].get("runtime_assets")
    runtime_assets = attestation.get("runtime_assets")
    if (
        not isinstance(frozen_assets, Mapping)
        or not isinstance(runtime_assets, Mapping)
        or set(runtime_assets) != set(frozen_assets)
    ):
        raise ValueError("stable backend runtime-asset schema changed")
    for role, frozen in frozen_assets.items():
        observed = runtime_assets.get(role)
        if (
            not isinstance(observed, Mapping)
            or set(observed) != {
                "model_id", "runtime_path", "tree_sha256", "file_count",
                "bytes", "verified_before_model_or_tokenizer_load",
            }
            or observed.get("model_id") != frozen.get("model_id")
            or observed.get("tree_sha256") != frozen.get("tree_sha256")
            or observed.get("file_count") != frozen.get("file_count")
            or observed.get("bytes") != frozen.get("bytes")
            or not isinstance(observed.get("runtime_path"), str)
            or not observed["runtime_path"]
            or observed.get("verified_before_model_or_tokenizer_load") is not True
        ):
            raise ValueError(f"stable backend runtime asset changed: {role}")

    projector = attestation.get("projector_load")
    if (
        not isinstance(projector, Mapping)
        or set(projector) != {"mode", "record_count", "records"}
        or projector.get("mode") != "strict_attested"
        or not isinstance(projector.get("records"), list)
        or not projector["records"]
        or projector.get("record_count") != len(projector["records"])
    ):
        raise ValueError("stable backend projector-load schema changed")
    checkpoint = next(
        (
            record for record in plan.get("checkpoints", [])
            if record.get("checkpoint_id") == shard.get("checkpoint_id")
        ),
        None,
    )
    frozen_files = (
        checkpoint.get("final", {}).get("files", [])
        if isinstance(checkpoint, Mapping) else []
    )
    frozen_file_shas = {record.get("sha256") for record in frozen_files}
    seen_projectors = set()
    for record in projector["records"]:
        if (
            not isinstance(record, Mapping)
            or set(record) != {
                "source_model_index", "projector_index", "json_path",
                "json_sha256", "state_path", "state_sha256", "strict",
                "missing_keys", "unexpected_keys", "state_key_count",
            }
            or record.get("strict") is not True
            or record.get("missing_keys") != []
            or record.get("unexpected_keys") != []
            or not isinstance(record.get("state_key_count"), int)
            or record["state_key_count"] <= 0
            or record.get("json_sha256") not in frozen_file_shas
            or record.get("state_sha256") not in frozen_file_shas
            or not str(record.get("json_path", "")).endswith(
                f"projector_{record.get('projector_index')}.json"
            )
            or not str(record.get("state_path", "")).endswith(
                f"projector_{record.get('projector_index')}.pt"
            )
        ):
            raise ValueError("stable backend projector-load record changed")
        identity = (record["source_model_index"], record["projector_index"])
        if identity in seen_projectors:
            raise ValueError("stable backend projector-load record duplicated")
        seen_projectors.add(identity)

    task_rows = plan["input_lock"]["prepared"].get("task_contract", {}).get(
        shard["task"], {}
    ).get("expected_long_form_rows")
    expected_rows = attestation.get("expected_long_form_rows")
    expected_row_attestation = {
        **dict(task_rows or {}),
        "source": "frozen_cpu_input_lock",
        "passed_explicitly_per_item": True,
        "exact_row_count_required": True,
        "verified_before_model_load": True,
    }
    if expected_rows != expected_row_attestation:
        raise ValueError("stable backend expected-long-form attestation changed")
    return dict(attestation)


def _normalize_backend_result(
    value: Any,
) -> tuple[Iterable[Mapping[str, Any]], dict[str, Any], MutableMapping[str, Any]]:
    if not isinstance(value, Mapping) or value.get("contract_version") != CAPTURE_BACKEND_CONTRACT_VERSION:
        raise ValueError("backend contract version mismatch")
    rows, attestation = value.get("rows"), value.get("attestation")
    resume_verification = value.get("resume_prefix_verification")
    if (
        rows is None
        or not isinstance(attestation, dict)
        or not isinstance(resume_verification, MutableMapping)
    ):
        raise ValueError("backend returned no rows/attestation/resume verification")
    required = {"split_role": ALLOWED_SPLIT_ROLE, "capture_mode": "teacher_forced_response", "causal_shift_verified": True, "stores_raw_kv": False}
    for name, expected in required.items():
        if attestation.get(name) != expected:
            raise ValueError(f"backend attestation mismatch: {name}")
    return rows, attestation, resume_verification


def _allowed_members(plan: Mapping[str, Any], task: str) -> dict[str, set[str]]:
    return {r["content_group_sha256"]: set(r["sample_sha256"]) for r in plan["group_contract"][task]}


def _group_records(plan: Mapping[str, Any], task: str) -> dict[str, Mapping[str, Any]]:
    return {r["content_group_sha256"]: r for r in plan["group_contract"][task]}


def _key_bytes(row: Mapping[str, Any], columns: Sequence[str]) -> bytes:
    return canonical_json_bytes([row[name] for name in columns]).rstrip(b"\n")


class _KeyLedger:
    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE keys (key BLOB PRIMARY KEY)")
    def add(self, key: bytes) -> None:
        try:
            self.connection.execute("INSERT INTO keys VALUES (?)", (key,))
        except sqlite3.IntegrityError as error:
            raise ValueError("duplicate final row key") from error
    def attest(self) -> tuple[int, str]:
        self.connection.commit()
        digest, count = hashlib.sha256(), 0
        for (key,) in self.connection.execute("SELECT key FROM keys ORDER BY key"):
            digest.update(bytes(key) + b"\n"); count += 1
        return count, digest.hexdigest()
    def close(self) -> None:
        self.connection.close()


def _capture_sample_streams(
    plan: Mapping[str, Any], shard: Mapping[str, Any], path_view: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild compact expected streams from the immutable v2 input sidecar."""

    import torch
    from script.analysis.fpct_e1_streaming_verify import (
        SampleRowStream,
        canonical_endpoint_id,
    )

    prepared = plan["input_lock"]["prepared"]
    sidecar_path = resolve_logical_path(
        plan, prepared["sidecar"]["logical_path"], path_view,
    )
    if (
        not sidecar_path.is_file()
        or sha256_file(sidecar_path) != prepared["sidecar"]["sha256"]
        or sidecar_path.stat().st_size != prepared["sidecar"]["bytes"]
    ):
        raise ValueError("prepared input-lock sidecar changed before stream replay")
    payload = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    payload_firewall = payload.get("firewall", {}) if isinstance(payload, Mapping) else {}
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != INPUT_LOCK_SCHEMA_VERSION
        or payload.get("protocol_id") != INPUT_LOCK_PROTOCOL_ID
        or payload.get("status") != INPUT_LOCK_STATUS
        or payload.get("split_role") != ALLOWED_SPLIT_ROLE
        or payload.get("model_or_checkpoint_loaded") is not False
        or payload.get("cuda_initialized") is not False
        or any(
            payload_firewall.get(name) is not False
            for name in (
                "e1_pilot_consumed", "model_selection_consumed",
                "test_consumed", "model_or_checkpoint_loaded", "gpu_or_cuda_used",
            )
        )
    ):
        raise ValueError("prepared input-lock sidecar identity/firewall changed")
    dimensions = prepared["dimensions"]
    endpoint = canonical_endpoint_id(
        shard["seed"], shard["checkpoint_arm"], shard["inference_operator"],
        shard["cell"], shard["task"], shard["lambda_value"],
    )
    template_endpoint = canonical_endpoint_id(
        0, "input_geometry", "input_geometry", "INPUT_LOCK", shard["task"], 0.0,
    )
    schema_sha = prepared["streaming_contract"]["schema_sha256"]
    items = sorted(
        (
            item for item in payload.get("items", [])
            if item.get("task") == shard["task"]
        ),
        key=lambda item: (item["sample_sha256"], item["content_group_sha256"]),
    )
    if len(items) != TASK_GROUP_COUNTS[shard["task"]]:
        raise ValueError("prepared compact sample universe differs from the plan")
    runtime: dict[str, Any] = {}
    template: dict[str, Any] = {}
    for item in items:
        sample = item["sample_sha256"]
        if sample in runtime:
            raise ValueError("prepared input lock contains a duplicate sample")
        arguments = {
            "num_layers": dimensions["num_hidden_layers"],
            "num_query_heads": dimensions["num_attention_heads"],
            "num_kv_heads": dimensions["num_key_value_heads"],
            "schema_sha256": schema_sha,
        }
        runtime[sample] = SampleRowStream(item, endpoint_id=endpoint, **arguments)
        template[sample] = SampleRowStream(
            item, endpoint_id=template_endpoint, **arguments,
        )
        if runtime[sample].row_count != int(item["N_s"]):
            raise ValueError("prepared sample N_s differs from ordinal geometry")
    return runtime, template


def _safe_artifact_relative(value: Any, *, label: str) -> Path:
    relative = Path(str(value or ""))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"{label} path escapes its immutable root")
    return relative


def _verify_streaming_template_tree(
    plan: Mapping[str, Any], path_view: str = "host",
) -> dict[str, Any]:
    """Recursively close the mounted A4 template tree before model import.

    Verification includes every index record, execution binding, nested
    manifest, physical Parquet chunk, exact per-sample directory membership,
    exact sample-directory membership, and the global canonical stream.  No
    cached success receipt is accepted in lieu of replaying the mounted bytes.
    """

    prepared = plan["input_lock"]["prepared"]
    lock = prepared["streaming_contract"]["streaming_template_lock"]
    index_path = resolve_logical_path(
        plan, lock["index"]["logical_path"], path_view,
    )
    receipt_path = resolve_logical_path(
        plan, lock["receipt"]["logical_path"], path_view,
    )
    input_root = resolve_logical_path(
        plan, prepared["manifest"]["logical_path"], path_view,
    ).parent
    if index_path.parent != input_root or receipt_path.parent != input_root:
        raise ValueError("streaming-template top-level artifacts changed roots")
    index = _read_json(index_path)
    receipt = _read_json(receipt_path)
    records = index.get("records")
    if not isinstance(records, list):
        raise ValueError("streaming-template index has no record list")

    template_streams: dict[str, Any] = {}
    sample_tasks: dict[str, str] = {}
    for task in TASKS:
        representative = next(
            (row for row in plan["shards"] if row["task"] == task), None,
        )
        if representative is None:
            raise ValueError(f"streaming-template task has no shard: {task}")
        _, task_streams = _capture_sample_streams(
            plan, representative, path_view,
        )
        for sample, stream in task_streams.items():
            if sample in template_streams:
                raise ValueError("streaming-template sample occurs in two tasks")
            template_streams[sample] = stream
            sample_tasks[sample] = task

    expected_order = sorted(template_streams)
    if [record.get("sample_sha256") for record in records] != expected_order:
        raise ValueError("streaming-template index sample order/universe changed")

    global_semantic_sha = index.get("semantic_stream_sha256")
    if (
        not _is_sha256(global_semantic_sha)
        or global_semantic_sha != receipt.get("semantic_stream_sha256")
        or global_semantic_sha != lock.get("semantic_stream_sha256")
    ):
        raise ValueError("streaming-template global semantic binding changed")
    global_count = 0
    sample_directories: set[Path] = set()
    template_root: Path | None = None
    execution_identity = prepared["execution_identity"]
    for record in records:
        sample = str(record["sample_sha256"])
        stream = template_streams[sample]
        manifest_relative = _safe_artifact_relative(
            record.get("manifest_relative_path"),
            label="streaming-template sample manifest",
        )
        binding_relative = _safe_artifact_relative(
            record.get("execution_binding_relative_path"),
            label="streaming-template execution binding",
        )
        manifest_path = input_root / manifest_relative
        binding_path = input_root / binding_relative
        if manifest_path.parent != binding_path.parent:
            raise ValueError("streaming-template manifest/binding roots differ")
        if manifest_path.name != "chunk_manifest.json":
            raise ValueError("streaming-template nested manifest name changed")
        if manifest_path.parent.name != sample:
            raise ValueError("streaming-template sample directory identity changed")
        candidate_root = manifest_path.parent.parent
        if template_root is None:
            template_root = candidate_root
        elif template_root != candidate_root:
            raise ValueError("streaming-template records span multiple roots")
        sample_directories.add(manifest_path.parent)
        if (
            record.get("task") != sample_tasks[sample]
            or record.get("expected_logical_rows") != stream.row_count
            or record.get("emitted_logical_rows") != stream.row_count
            or not manifest_path.is_file()
            or sha256_file(manifest_path) != record.get("manifest_sha256")
            or manifest_path.stat().st_size != record.get("manifest_bytes")
        ):
            raise ValueError("streaming-template nested manifest binding changed")
        expected_binding = {
            "schema_version": 1,
            "protocol_id": "fpct_e1_a4_sample_stream_execution_binding_v1",
            "execution_identity_sha256": execution_identity["identity_sha256"],
            "sample_sha256": sample,
            "item_semantic_sha256": record.get("item_semantic_sha256"),
            "historical_artifact_reuse_allowed": False,
        }
        if (
            not binding_path.is_file()
            or sha256_file(binding_path)
            != record.get("execution_binding_sha256")
            or _read_json(binding_path) != expected_binding
        ):
            raise ValueError("streaming-template execution binding changed")

        nested = _read_json(manifest_path)
        chunks = nested.get("chunks")
        if not isinstance(chunks, list):
            raise ValueError("streaming-template nested chunk list changed")
        expected_files = {manifest_path.name, binding_path.name}
        for chunk in chunks:
            chunk_relative = _safe_artifact_relative(
                chunk.get("relative_path"), label="streaming-template chunk",
            )
            if len(chunk_relative.parts) != 1:
                raise ValueError("streaming-template chunk is not sample-local")
            expected_files.add(chunk_relative.name)
        observed_entries = {child.name for child in manifest_path.parent.iterdir()}
        if observed_entries != expected_files or any(
            not child.is_file() or child.is_symlink()
            for child in manifest_path.parent.iterdir()
        ):
            raise ValueError(
                "streaming-template sample tree has corruption or orphans"
            )

        previous_end = 0
        for expected_index, chunk in enumerate(chunks):
            first = int(chunk.get("first_row_ordinal", -1))
            end = int(chunk.get("end_row_ordinal_exclusive", -1))
            relative_path = _safe_artifact_relative(
                chunk.get("relative_path"), label="streaming-template chunk",
            )
            chunk_path = manifest_path.parent / relative_path
            if (
                chunk.get("chunk_index") != expected_index
                or first != previous_end
                or end <= first
                or end > stream.row_count
                or chunk.get("row_count") != end - first
                or chunk.get("sample_sha256") != sample
                or chunk.get("first_logical_row_id")
                != stream.row_at(first)["logical_row_id"]
                or chunk.get("last_logical_row_id")
                != stream.row_at(end - 1)["logical_row_id"]
                or not chunk_path.is_file()
                or chunk_path.is_symlink()
                or sha256_file(chunk_path)
                != chunk.get("physical_file_sha256")
                or chunk_path.stat().st_size
                != chunk.get("physical_file_bytes")
            ):
                raise ValueError("streaming-template physical chunk binding changed")
            previous_end = end
        sample_count = previous_end
        if (
            sample_count != stream.row_count
            or sample_count != record.get("expected_logical_rows")
            or sample_count != record.get("emitted_logical_rows")
            or nested.get("semantic_stream_sha256")
            != record.get("semantic_stream_sha256")
            or len(chunks) != record.get("chunk_count")
        ):
            raise ValueError("streaming-template nested semantic replay changed")
        global_count += sample_count

    if template_root is None or not template_root.is_dir():
        raise ValueError("streaming-template root is absent")
    observed_sample_directories = {
        child for child in template_root.iterdir() if child.is_dir()
    }
    if (
        observed_sample_directories != sample_directories
        or any(child.is_symlink() or not child.is_dir() for child in template_root.iterdir())
    ):
        raise ValueError("streaming-template root has corruption or orphan samples")
    if (
        global_count != index.get("expected_logical_rows")
        or global_count != index.get("emitted_logical_rows")
        or global_count != lock.get("expected_logical_rows")
        or global_count != lock.get("emitted_logical_rows")
    ):
        raise ValueError("streaming-template global semantic replay changed")
    return {
        "status": "GO_RECURSIVE_TEMPLATE_TREE",
        "sample_count": len(records),
        "logical_row_count": global_count,
        "semantic_stream_sha256": global_semantic_sha,
    }


def _iter_capture_parquet_rows(output_dir: Path) -> Iterator[dict[str, Any]]:
    """Read a capture artifact only in manifest sample/chunk ordinal order."""

    import pyarrow.parquet as pq

    manifest = _read_json(output_dir / "capture_manifest.json")
    for sample_record in manifest.get("sample_streams", []):
        sample_manifest_path = output_dir / sample_record["manifest_relative_path"]
        sample_manifest = _read_json(sample_manifest_path)
        for chunk in sample_manifest.get("chunks", []):
            path = sample_manifest_path.parent / chunk["relative_path"]
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=PARQUET_BATCH_ROWS,
            ):
                yield from batch.to_pylist()


class _BaselineIndex:
    def __init__(self, baseline_dir: Path, key_columns: Sequence[str], baseline_columns: Sequence[str], database: Path) -> None:
        self.connection = sqlite3.connect(database)
        self.connection.execute("CREATE TABLE baseline (key BLOB PRIMARY KEY, payload TEXT NOT NULL, used INTEGER NOT NULL DEFAULT 0)")
        count = 0
        for row in _iter_capture_parquet_rows(baseline_dir):
            payload = {name: row[name.removeprefix("cpost_")] for name in baseline_columns}
            try:
                self.connection.execute("INSERT INTO baseline(key,payload) VALUES (?,?)", (_key_bytes(row, key_columns), json.dumps(payload, sort_keys=True, allow_nan=False)))
            except sqlite3.IntegrityError as error:
                raise ValueError("duplicate C_post baseline row key") from error
            count += 1
        self.connection.commit(); self.count = count
    def join(self, key: bytes) -> dict[str, Any]:
        record = self.connection.execute("SELECT payload,used FROM baseline WHERE key=?", (key,)).fetchone()
        if record is None: raise ValueError("operator row has no C_post baseline row")
        if record[1]: raise ValueError("operator row reuses a C_post baseline row")
        self.connection.execute("UPDATE baseline SET used=1 WHERE key=?", (key,))
        return json.loads(record[0])
    def assert_bijection(self) -> None:
        self.connection.commit()
        used = self.connection.execute("SELECT COUNT(*) FROM baseline WHERE used=1").fetchone()[0]
        if used != self.count: raise ValueError(f"row-key universe differs from C_post baseline: used={used}, baseline={self.count}")
    def close(self) -> None: self.connection.close()


def _validate_backend_row(
    raw: Mapping[str, Any], shard: Mapping[str, Any],
    allowed: Mapping[str, set[str]], backend_columns: Sequence[str],
) -> dict[str, Any]:
    forbidden = [name for name in raw if name.startswith("cpost_")]
    if forbidden: raise ValueError(f"backend may not provide mechanically joined baseline fields: {forbidden}")
    missing = [name for name in backend_columns if name not in raw]
    if missing: raise ValueError(f"backend row missing fields: {missing}")
    row = {name: raw[name] for name in backend_columns}
    expected = {"schema_version": SCHEMA_VERSION, "split_role": ALLOWED_SPLIT_ROLE, "seed": shard["seed"], "checkpoint_arm": shard["checkpoint_arm"], "inference_operator": shard["inference_operator"], "cell": shard["cell"], "task": shard["task"], "lambda_value": shard["lambda_value"]}
    for name, value in expected.items():
        if row.get(name) != value: raise ValueError(f"row/shard mismatch: {name}")
    group = row.get("content_group_sha256")
    if group not in allowed or row.get("sample_sha256") not in allowed[group]:
        raise ValueError("capture row outside E0-design membership")
    if row.get("target_position") != row.get("query_position") + 1:
        raise ValueError("teacher-forced causal shift mismatch")
    from script.analysis.fpct_e1_streaming_verify import (
        canonical_endpoint_id,
        endpoint_row_id,
        logical_row_id,
    )
    endpoint = canonical_endpoint_id(
        shard["seed"], shard["checkpoint_arm"], shard["inference_operator"],
        shard["cell"], shard["task"], shard["lambda_value"],
    )
    if row.get("endpoint_id") != endpoint:
        raise ValueError("backend endpoint_id differs from the frozen shard endpoint")
    logical = logical_row_id(row)
    if row.get("logical_row_id") != logical:
        raise ValueError("backend logical_row_id differs from the original scientific row key")
    if row.get("endpoint_row_id") != endpoint_row_id(endpoint, logical):
        raise ValueError("backend endpoint_row_id differs from endpoint/logical identity")
    ordinal = row.get("row_ordinal")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise ValueError("backend row_ordinal must be a nonnegative integer")
    return row


def _finalize_row(row: dict[str, Any], baseline: Mapping[str, Any], final_columns: Sequence[str]) -> dict[str, Any]:
    final = {**row, **baseline}
    from script.analysis.fpct_e1_mechanism_audit import validate_and_derive_row
    validate_and_derive_row(final)
    return {name: final[name] for name in final_columns}


def _phase_marker(output_root: Path, stage: str) -> Path:
    return output_root / "locks" / f"{stage}_complete.json"


def _finalized_lock_path(output_root: Path) -> Path:
    return output_root / "locks" / f"{FINALIZED_E1_2}.json"


def _closure_records(plan: Mapping[str, Any], output_root: Path, stage: str, *, deep: bool) -> list[dict[str, Any]]:
    records = []
    for shard in _selected_shards(plan, stage):
        output_dir = output_root / shard["output_relative"]
        capture_manifest_path = output_dir / "capture_manifest.json"
        capture_manifest = _read_json(capture_manifest_path)
        if (
            capture_manifest.get("schema_version") != SCHEMA_VERSION
            or capture_manifest.get("protocol_id") != PROTOCOL_ID
            or capture_manifest.get("status") != "GO"
            or capture_manifest.get("plan_sha256") != plan["plan_sha256"]
            or capture_manifest.get("shard") != shard
            or capture_manifest.get("e1_pilot_consumed") is not False
            or capture_manifest.get("model_selection_consumed") is not False
            or capture_manifest.get("test_consumed") is not False
        ):
            raise RuntimeError(f"capture manifest identity changed: {shard['shard_id']}")
        verified = verify_capture_artifacts(output_dir, plan) if deep else {
            "row_key_sha256": capture_manifest.get("row_key_attestation", {}).get("sha256"),
            "row_count": capture_manifest.get("row_count"),
            "semantic_stream_sha256": capture_manifest.get("streaming_receipt", {}).get("semantic_stream_sha256"),
            "sample_manifest_tree_sha256": capture_manifest.get("streaming_receipt", {}).get("sample_manifest_tree_sha256"),
            "parquet_chunk_count": capture_manifest.get("parquet_chunk_count"),
        }
        record = {
            "shard_id": shard["shard_id"], "lambda_value": shard["lambda_value"],
            "semantic_stream_sha256": verified["semantic_stream_sha256"],
            "sample_manifest_tree_sha256": verified["sample_manifest_tree_sha256"],
            "parquet_chunk_count": verified["parquet_chunk_count"],
            "row_key_sha256": verified["row_key_sha256"], "row_count": verified["row_count"],
            "capture_manifest_sha256": sha256_file(capture_manifest_path),
        }
        if (
            not _is_sha256(record["row_key_sha256"])
            or not _is_sha256(record["semantic_stream_sha256"])
            or not _is_sha256(record["sample_manifest_tree_sha256"])
            or not isinstance(record["parquet_chunk_count"], int)
            or record["parquet_chunk_count"] <= 0
            or not isinstance(record["row_count"], int)
            or record["row_count"] <= 0
        ):
            raise RuntimeError(f"capture manifest row attestation invalid: {shard['shard_id']}")
        records.append(record)
    return records


def _verify_phase_marker(plan: Mapping[str, Any], output_root: Path, stage: str) -> dict[str, Any]:
    marker = _read_json(_phase_marker(output_root, stage))
    expected_count = {PHASE_E1_2_BASELINES: 18, PHASE_E1_2_FACTORIZED: 18, STAGE_E1_2: 36, STAGE_E1_3: 108}.get(stage)
    if marker.get("status") != "GO" or marker.get("stage") != stage or marker.get("plan_sha256") != plan["plan_sha256"] or marker.get("shard_count") != expected_count or not _is_sha256(marker.get("closure_sha256")):
        raise RuntimeError(f"required phase completion is absent/invalid: {stage}")
    observed = _closure_records(plan, output_root, stage, deep=False)
    if marker.get("ordered_shards") != observed or marker["closure_sha256"] != sha256_bytes(canonical_json_bytes(observed)):
        raise RuntimeError(f"phase completion artifacts/closure changed: {stage}")
    return marker


def _analyzer_artifact_records(output_dir: Path) -> dict[str, dict[str, Any]]:
    from script.analysis.fpct_e1_mechanism_audit import OUTPUT_NAMES

    records: dict[str, dict[str, Any]] = {}
    for filename in sorted(OUTPUT_NAMES.values()):
        path = output_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        records[filename] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "mode": oct(path.stat().st_mode & 0o777),
        }
    return records


def _write_finalized_e1_2_lock(
    plan: Mapping[str, Any], output_root: Path, output_dir: Path,
    stage_manifest: Mapping[str, Any], closure_marker: Mapping[str, Any],
) -> dict[str, Any]:
    stage_manifest_path = output_dir / "stage_artifact_manifest.json"
    record = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "GO",
        "stage": FINALIZED_E1_2,
        "source_stage": STAGE_E1_2,
        "plan_sha256": plan["plan_sha256"],
        "closure_sha256": closure_marker["closure_sha256"],
        "raw_topology_lock": plan["raw_topology_lock"],
        "output_dir": _logical(
            "output",
            output_dir.absolute().relative_to(
                Path(plan["path_roots"]["output"]["host"]).absolute()
            ),
        ),
        "stage_artifact_manifest": {
            "sha256": sha256_file(stage_manifest_path),
            "bytes": stage_manifest_path.stat().st_size,
        },
        "analyzer_verification": stage_manifest["analyzer_verification"],
        "merged_capture": stage_manifest["merged_capture"],
        "analyzer_artifacts": stage_manifest["analyzer_artifacts"],
        "e1_pilot_consumed": False,
        "confirmatory_consumed": False,
    }
    path = _finalized_lock_path(output_root)
    if path.exists():
        if _read_json(path) != record:
            raise RuntimeError("existing E1-2 finalized lock differs")
    else:
        atomic_write(path, canonical_json_bytes(record))
    return record


def _verify_finalized_e1_2_lock(
    plan: Mapping[str, Any], output_root: Path, *, path_view: str = "host",
    expected_lock_sha256: str | None = None, deep: bool = False,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    lock_path = receipt_path if receipt_path is not None else _finalized_lock_path(output_root)
    if expected_lock_sha256 is not None and sha256_file(lock_path) != expected_lock_sha256:
        raise RuntimeError("finalized E1-2 receipt SHA changed")
    lock = _read_json(lock_path)
    if (
        lock.get("status") != "GO"
        or lock.get("stage") != FINALIZED_E1_2
        or lock.get("source_stage") != STAGE_E1_2
        or lock.get("plan_sha256") != plan["plan_sha256"]
        or lock.get("raw_topology_lock") != plan["raw_topology_lock"]
        or lock.get("e1_pilot_consumed") is not False
        or lock.get("confirmatory_consumed") is not False
    ):
        raise RuntimeError("required finalized E1-2 lock is absent/invalid")
    _verify_raw_topology_sources(plan, path_view)
    endpoint = _verify_phase_marker(plan, output_root, STAGE_E1_2)
    if lock.get("closure_sha256") != endpoint["closure_sha256"]:
        raise RuntimeError("finalized E1-2 closure differs from endpoint closure")
    output_logical = lock.get("output_dir")
    if not isinstance(output_logical, str) or not output_logical.startswith("output://"):
        raise RuntimeError("finalized E1-2 output path is not portable")
    output_dir = resolve_logical_path(plan, output_logical, path_view)
    stage_manifest_path = output_dir / "stage_artifact_manifest.json"
    frozen_manifest = lock.get("stage_artifact_manifest", {})
    if (
        not stage_manifest_path.is_file()
        or sha256_file(stage_manifest_path) != frozen_manifest.get("sha256")
        or stage_manifest_path.stat().st_size != frozen_manifest.get("bytes")
    ):
        raise RuntimeError("finalized E1-2 stage manifest changed")
    stage_manifest = _read_json(stage_manifest_path)
    if (
        stage_manifest.get("plan_sha256") != plan["plan_sha256"]
        or stage_manifest.get("stage") != STAGE_E1_2
        or stage_manifest.get("closure_sha256") != endpoint["closure_sha256"]
        or stage_manifest.get("analyzer_verification") != lock.get("analyzer_verification")
        or stage_manifest.get("merged_capture") != lock.get("merged_capture")
        or stage_manifest.get("analyzer_artifacts") != lock.get("analyzer_artifacts")
    ):
        raise RuntimeError("finalized E1-2 artifact identity changed")
    merged = stage_manifest["merged_capture"]
    merged_path = output_dir / str(merged.get("relative_path", ""))
    analyzer_artifacts = stage_manifest["analyzer_artifacts"]
    if not isinstance(analyzer_artifacts, Mapping):
        raise RuntimeError("finalized E1-2 analyzer artifact receipt is absent")
    paths = [(merged_path, merged)] + [
        (output_dir / filename, record)
        for filename, record in sorted(analyzer_artifacts.items())
    ]
    for artifact_path, record in paths:
        if (
            not artifact_path.is_file()
            or artifact_path.stat().st_size != record.get("bytes")
            or (
                "mode" in record
                and oct(artifact_path.stat().st_mode & 0o777) != record.get("mode")
            )
            or (deep and sha256_file(artifact_path) != record.get("sha256"))
        ):
            raise RuntimeError(f"finalized E1-2 artifact changed: {artifact_path.name}")
    if deep:
        from script.analysis.fpct_e1_mechanism_audit import verify_artifacts

        verified = verify_artifacts(output_dir)
        if verified != stage_manifest["analyzer_verification"]:
            raise RuntimeError("finalized E1-2 streaming analyzer verification changed")
    return lock


def _claim_path(output_root: Path, shard: Mapping[str, Any]) -> Path:
    return output_root / "claims" / f"{shard['shard_id']}.json"


@dataclass
class _ClaimLease:
    path: Path
    handle: Any
    record: dict[str, Any]

    def transition(self, status: str, *, error_class: str | None = None, artifact_sha256: str | None = None) -> dict[str, Any]:
        if status not in {"FAILED", "SUCCEEDED"}:
            raise ValueError("claim terminal status must be FAILED or SUCCEEDED")
        updated = {**self.record, "status": status, "error_class": error_class, "artifact_manifest_sha256": artifact_sha256}
        atomic_write(self.path, canonical_json_bytes(updated))
        self.record = updated
        return updated

    def close(self) -> None:
        import fcntl
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def _acquire_claim(path: Path, plan: Mapping[str, Any], shard: Mapping[str, Any], claim_id: str) -> _ClaimLease:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    handle = lock_path.open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"shard claim lease is currently active: {path}") from error
    try:
        previous = _read_json(path) if path.exists() else None
        if previous is not None:
            identity = (previous.get("plan_sha256"), previous.get("shard_id"), previous.get("claim_id"))
            expected = (plan["plan_sha256"], shard["shard_id"], claim_id)
            if identity != expected:
                raise RuntimeError("existing shard claim belongs to another plan/shard/claim-id")
            if previous.get("status") == "SUCCEEDED":
                raise RuntimeError("successful claim exists but completed artifact is unavailable")
            if previous.get("status") not in {"ACTIVE", "FAILED"}:
                raise RuntimeError("existing shard claim has an invalid state")
            attempt = int(previous.get("attempt", 0)) + 1
        else:
            attempt = 1
        record = {
            "schema_version": 1, "plan_sha256": plan["plan_sha256"],
            "shard_id": shard["shard_id"], "claim_id": claim_id,
            "attempt": attempt, "status": "ACTIVE", "error_class": None,
            "artifact_manifest_sha256": None,
        }
        atomic_write(path, canonical_json_bytes(record))
        return _ClaimLease(path, handle, record)
    except Exception:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        raise


def _reconcile_completed_claim(path: Path, plan: Mapping[str, Any], shard: Mapping[str, Any], claim_id: str, artifact_manifest: Path) -> None:
    expected = (plan["plan_sha256"], shard["shard_id"], claim_id)
    if path.exists():
        record = _read_json(path)
        identity = (record.get("plan_sha256"), record.get("shard_id"), record.get("claim_id"))
        if identity != expected:
            raise RuntimeError("completed shard claim belongs to another plan/shard/claim-id")
        if record.get("status") == "SUCCEEDED":
            if record.get("artifact_manifest_sha256") != sha256_file(artifact_manifest):
                raise RuntimeError("completed shard claim artifact SHA mismatch")
            return
    lease = _acquire_claim(path, plan, shard, claim_id)
    try:
        lease.transition("SUCCEEDED", artifact_sha256=sha256_file(artifact_manifest))
    finally:
        lease.close()


def _capture_staging_path(output_dir: Path) -> Path:
    return output_dir.with_name(f".{output_dir.name}.staging")


def _write_once(
    path: Path, payload: bytes, *, temporary_root: Path,
    failure_injector: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> None:
    """Publish immutable metadata without replacing an existing completion."""

    final_parent = path.parent.absolute()
    temporary_parent = temporary_root.absolute()
    if (
        not final_parent.is_dir()
        or not temporary_parent.is_dir()
        or final_parent.is_symlink()
        or temporary_parent.is_symlink()
        or final_parent.resolve(strict=True) != final_parent
        or temporary_parent.resolve(strict=True) != temporary_parent
    ):
        raise ValueError("immutable metadata roots must be canonical non-symlink directories")
    if _is_within_or_equal(temporary_parent, final_parent):
        raise ValueError("immutable metadata temporary root is inside final tree")

    def existing_regular_file() -> bool:
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(mode) or path.is_symlink():
            raise ValueError("immutable metadata destination is not a regular file")
        return True

    if existing_regular_file():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable capture metadata changed: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=temporary_root,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not existing_regular_file():
                raise ValueError("immutable metadata destination changed during publish")
            if path.read_bytes() != payload:
                raise ValueError(f"immutable capture metadata changed: {path}")
        if failure_injector is not None:
            failure_injector("after_metadata_hardlink_before_temp_cleanup", {
                "artifact_path": str(path),
                "temporary_path": str(temporary),
            })
    finally:
        if temporary.exists():
            temporary.unlink()


def _capture_staging_identity(
    plan: Mapping[str, Any], shard: Mapping[str, Any],
) -> dict[str, Any]:
    expected = shard["expected_row_template"]
    return {
        "schema_version": 1,
        "protocol_id": CAPTURE_STAGING_PROTOCOL_ID,
        "plan_sha256": plan["plan_sha256"],
        "shard_id": shard["shard_id"],
        "shard_sha256": sha256_bytes(canonical_json_bytes(shard)),
        "output_relative": shard["output_relative"],
        "schema_sha256": plan["input_lock"]["prepared"]["streaming_contract"]["schema_sha256"],
        "physical_chunk_rows": PARQUET_BATCH_ROWS,
        "expected_logical_rows": expected["count"],
        "expected_template_semantic_sha256": expected["semantic_stream_sha256"],
        "immutable_completed_chunks": True,
        "resume_only_from_first_incomplete_range": True,
    }


def _chunk_completion_receipt(
    plan: Mapping[str, Any], shard: Mapping[str, Any],
    record: Mapping[str, Any], stable_backend_attestation_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": CAPTURE_STAGING_PROTOCOL_ID,
        "artifact_type": "capture_chunk_completion_receipt",
        "completion_state": "COMPLETE_IMMUTABLE",
        "plan_sha256": plan["plan_sha256"],
        "shard_id": shard["shard_id"],
        "stable_backend_attestation_sha256": stable_backend_attestation_sha256,
        "chunk": dict(record),
    }


def _chunk_receipt_name(chunk_index: int) -> str:
    return f"chunk-{chunk_index:06d}.receipt.json"


def _capture_sample_manifest_payload(
    sample: str, endpoint_id: str, expected_count: int,
    chunk_records: Sequence[Mapping[str, Any]], semantic_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "capture_sample_chunk_manifest",
        "sample_sha256": sample,
        "endpoint_id": endpoint_id,
        "expected_logical_rows": expected_count,
        "emitted_logical_rows": expected_count,
        "physical_chunk_rows": PARQUET_BATCH_ROWS,
        "chunks": [dict(record) for record in chunk_records],
        "semantic_stream_sha256": semantic_sha256,
        "missing_rows": 0,
        "duplicate_rows": 0,
        "overlapping_chunks": 0,
        "ordinal_ranges_contiguous": True,
    }


def _validate_persisted_capture_row(
    row: Mapping[str, Any], expected: Mapping[str, Any],
    final_columns: Sequence[str],
) -> None:
    if tuple(row) != tuple(final_columns):
        raise ValueError("staged capture row schema/order changed")
    from script.analysis.fpct_e1_mechanism_audit import validate_and_derive_row
    validate_and_derive_row(row)
    identity_columns = (
        *ROW_TEMPLATE_COLUMNS, "row_ordinal", "logical_row_id",
        "endpoint_id", "endpoint_row_id",
    )
    if any(row[name] != expected[name] for name in identity_columns):
        raise ValueError("staged capture row differs from frozen ordinal geometry")
    if (
        len(row["prior"]) != len(expected["prior"])
        or any(
            abs(float(left) - float(right)) > 2e-5
            for left, right in zip(row["prior"], expected["prior"])
        )
        or row["candidate_indices"] != expected["candidate_indices"]
        or row["candidate_valid_mask"] != expected["candidate_valid_mask"]
        or len(row["candidate_slot_weights"])
        != len(expected["candidate_slot_weights"])
        or any(
            abs(float(left) - float(right)) > 2e-5
            for left, right in zip(
                row["candidate_slot_weights"],
                expected["candidate_slot_weights"],
            )
        )
        or abs(
            float(row["statistical_weight"])
            - float(expected["statistical_weight"])
        ) > 1e-12
    ):
        raise ValueError("staged capture candidate geometry/weight changed")


def _backend_projection_bytes(
    row: Mapping[str, Any], backend_columns: Sequence[str] | None = None,
) -> bytes:
    if backend_columns is None:
        backend_columns = _audit_columns()[0]
    return canonical_json_bytes([row[name] for name in backend_columns])


def _validate_first_uncommitted_chunk(
    path: Path, stream: Any, chunk_index: int,
    final_columns: Sequence[str],
) -> None:
    """Prove an orphan is exactly one complete next chunk before removal."""

    import pyarrow.parquet as pq

    first = chunk_index * PARQUET_BATCH_ROWS
    end = min(first + PARQUET_BATCH_ROWS, stream.row_count)
    parquet_file = pq.ParquetFile(path)
    if (
        tuple(parquet_file.schema_arrow.names) != tuple(final_columns)
        or parquet_file.num_row_groups != 1
        or parquet_file.metadata.num_rows != end - first
    ):
        raise ValueError("uncommitted capture chunk is corrupt; refusing cleanup")
    count = 0
    for batch in parquet_file.iter_batches(batch_size=PARQUET_BATCH_ROWS):
        names = batch.schema.names
        for row_index in range(batch.num_rows):
            row = {
                name: batch.column(column_index)[row_index].as_py()
                for column_index, name in enumerate(names)
            }
            if first + count >= end:
                raise ValueError("uncommitted capture chunk exceeds its range")
            _validate_persisted_capture_row(
                row, stream.row_at(first + count), final_columns,
            )
            count += 1
    if count != end - first:
        raise ValueError("uncommitted capture chunk row count changed")


def _verify_resume_prefix_channel(
    channel: Mapping[str, Any], cursor: Mapping[str, Any],
    *, require_complete: bool,
) -> None:
    allowed_status = {"PENDING_RECOMPUTATION", "GO_EXACT_PREFIX_MATCH"}
    if (
        channel.get("schema_version") != 1
        or channel.get("protocol_id") != CAPTURE_STAGING_PROTOCOL_ID
        or channel.get("cursor_sha256") != cursor["cursor_sha256"]
        or channel.get("expected_row_count")
        != cursor["partial_sample_backend_prefix_row_count"]
        or channel.get("expected_backend_projection_sha256")
        != cursor["partial_sample_backend_prefix_sha256"]
        or channel.get("status") not in allowed_status
    ):
        raise ValueError("backend resume-prefix verification channel changed")
    if require_complete and (
        channel.get("status") != "GO_EXACT_PREFIX_MATCH"
        or channel.get("observed_row_count")
        != cursor["partial_sample_backend_prefix_row_count"]
        or channel.get("observed_backend_projection_sha256")
        != cursor["partial_sample_backend_prefix_sha256"]
        or channel.get("exact_match") is not True
    ):
        raise ValueError("backend recomputed prefix differs from immutable prefix")


def _inspect_capture_staging(
    output_dir: Path, plan: Mapping[str, Any], shard: Mapping[str, Any],
    *, path_view: str = "host", create: bool = False,
    failure_injector: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Verify a stable shard staging tree and derive its exact resume cursor."""

    staging = _capture_staging_path(output_dir)
    if output_dir.exists() and staging.exists():
        raise ValueError("completed capture has an orphan staging tree")
    expected_streams, _ = _capture_sample_streams(plan, shard, path_view)
    expected_samples = sorted(expected_streams)
    backend_columns, final_columns, _ = _audit_columns()
    identity = _capture_staging_identity(plan, shard)
    if not staging.exists():
        if create:
            staging.mkdir(parents=True, exist_ok=False)
            _write_once(
                staging / "capture_staging_identity.json",
                canonical_json_bytes(identity),
                temporary_root=staging.parent,
                failure_injector=failure_injector,
            )
        else:
            cursor_payload = {
                "schema_version": 1,
                "protocol_id": CAPTURE_STAGING_PROTOCOL_ID,
                "plan_sha256": plan["plan_sha256"],
                "shard_id": shard["shard_id"],
                "completed_logical_rows": 0,
                "completed_sample_count": 0,
                "completed_chunk_count": 0,
                "next_sample_sha256": expected_samples[0],
                "next_row_ordinal": 0,
                "prefix_semantic_stream_sha256": hashlib.sha256().hexdigest(),
                "completion_receipt_chain_sha256": hashlib.sha256().hexdigest(),
                "partial_sample_backend_prefix_row_count": 0,
                "partial_sample_backend_prefix_sha256": hashlib.sha256().hexdigest(),
                "backend_projection_columns": list(backend_columns),
                "scientific_prefix_reverified": True,
                "resume_only_from_first_incomplete_range": True,
            }
            return {
                "root": staging, "exists": False, "complete": False,
                "published_pending": False, "samples": {},
                "expected_streams": expected_streams,
                "cursor": {
                    **cursor_payload,
                    "cursor_sha256": sha256_bytes(canonical_json_bytes(cursor_payload)),
                },
            }
    if not staging.is_dir() or staging.is_symlink():
        raise ValueError("capture staging root is not a real directory")
    identity_path = staging / "capture_staging_identity.json"
    if (
        not identity_path.is_file()
        or identity_path.is_symlink()
        or _read_json(identity_path) != identity
    ):
        raise ValueError("capture staging identity changed")
    allowed_root = {"capture_staging_identity.json", "capture_chunks"}
    if (staging / "backend_attestation.json").exists():
        allowed_root.add("backend_attestation.json")
    if (staging / "capture_manifest.json").exists():
        allowed_root.add("capture_manifest.json")
    if {child.name for child in staging.iterdir()} - allowed_root:
        raise ValueError("capture staging root contains corruption or orphans")
    for child in staging.iterdir():
        if child.is_symlink():
            raise ValueError("capture staging tree contains a symlink")
        if child.name == "capture_chunks" and not child.is_dir():
            raise ValueError("capture staging chunk root is not a directory")
        if child.name != "capture_chunks" and not child.is_file():
            raise ValueError("capture staging metadata is not a file")
    attestation_path = staging / "backend_attestation.json"
    attestation_record: dict[str, Any] | None = None
    stable_backend_attestation_sha256: str | None = None
    attempt_local_attestation: Path | None = None
    if attestation_path.exists():
        attestation_record = _read_json(attestation_path)
        if (
            attestation_record.get("schema_version") != 1
            or attestation_record.get("protocol_id")
            != CAPTURE_STAGING_PROTOCOL_ID
            or attestation_record.get("artifact_type")
            != "stable_backend_attestation"
            or attestation_record.get("plan_sha256") != plan["plan_sha256"]
            or attestation_record.get("shard_id") != shard["shard_id"]
            or not isinstance(attestation_record.get("attestation"), Mapping)
        ):
            raise ValueError("capture staging backend attestation changed")

    import pyarrow.parquet as pq
    from script.analysis.fpct_e1_streaming_verify import semantic_row_bytes
    chunk_root = staging / "capture_chunks"
    observed_sample_dirs = set()
    if chunk_root.exists():
        observed_sample_dirs = {child.name for child in chunk_root.iterdir()}
        if any(child.is_symlink() or not child.is_dir() for child in chunk_root.iterdir()):
            raise ValueError("capture staging chunk root contains an orphan entry")
    if not observed_sample_dirs.issubset(set(expected_samples)):
        raise ValueError("capture staging contains an unknown sample directory")
    committed_receipts = list(chunk_root.glob("*/chunk-*.receipt.json")) if chunk_root.exists() else []
    if committed_receipts:
        if attestation_record is None:
            raise ValueError("capture staging chunks lack their backend attestation")
        stable_attestation = _validate_stable_backend_attestation(
            attestation_record["attestation"], plan, shard,
        )
        stable_backend_attestation_sha256 = sha256_bytes(
            canonical_json_bytes(stable_attestation)
        )
    elif attestation_record is not None:
        # No immutable chunk receipt binds this post-load attestation yet.  It
        # is attempt-local state and must not cross the next backend-load gate.
        if (staging / "capture_manifest.json").exists():
            raise ValueError("capture manifest exists without committed chunks")
        attempt_local_attestation = attestation_path

    prefix_digest = hashlib.sha256()
    receipt_digest = hashlib.sha256()
    completed_rows = 0
    completed_chunks = 0
    completed_samples = 0
    sample_states: dict[str, Any] = {}
    safe_uncommitted_orphans: list[Path] = []
    first_incomplete_seen = False
    arrow_schema = None
    for sample_index, sample in enumerate(expected_samples):
        stream = expected_streams[sample]
        sample_root = chunk_root / sample
        if not sample_root.exists():
            first_incomplete_seen = True
            if any(later in observed_sample_dirs for later in expected_samples[sample_index + 1:]):
                raise ValueError("capture staging sample prefix contains a gap")
            break
        if first_incomplete_seen:
            raise ValueError("capture staging has rows after the incomplete range")
        entries = {child.name: child for child in sample_root.iterdir()}
        if any(child.is_symlink() or not child.is_file() for child in entries.values()):
            raise ValueError("capture staging sample contains an orphan entry")
        parquet_names = sorted(
            name for name in entries if re.fullmatch(r"chunk-[0-9]{6}\.parquet", name)
        )
        receipt_names = sorted(
            name for name in entries if re.fullmatch(r"chunk-[0-9]{6}\.receipt\.json", name)
        )
        manifest_present = "chunk_manifest.json" in entries
        expected_chunk_total = (
            stream.row_count + PARQUET_BATCH_ROWS - 1
        ) // PARQUET_BATCH_ROWS
        # A process can die after the atomic Parquet hardlink but before its
        # completion receipt.  Only that one precisely located next chunk may
        # be cleaned up, and only after a full schema/range/row replay proves
        # it is complete and uncommitted.  Any other mismatch is corruption.
        paired_count = len(receipt_names)
        expected_paired_parquet = [
            f"chunk-{index:06d}.parquet" for index in range(paired_count)
        ]
        expected_paired_receipts = [
            _chunk_receipt_name(index) for index in range(paired_count)
        ]
        if (
            len(parquet_names) == paired_count + 1
            and parquet_names[:paired_count] == expected_paired_parquet
            and receipt_names == expected_paired_receipts
            and parquet_names[-1] == f"chunk-{paired_count:06d}.parquet"
            and paired_count < expected_chunk_total
            and not manifest_present
        ):
            orphan_path = entries[parquet_names[-1]]
            _validate_first_uncommitted_chunk(
                orphan_path, stream, paired_count, final_columns,
            )
            safe_uncommitted_orphans.append(orphan_path)
            entries.pop(parquet_names[-1])
            parquet_names = parquet_names[:-1]
        allowed_entries = set(parquet_names) | set(receipt_names)
        if manifest_present:
            allowed_entries.add("chunk_manifest.json")
        if set(entries) != allowed_entries:
            raise ValueError("capture staging sample contains corruption or orphans")
        completed_here = len(parquet_names)
        expected_parquet_names = [
            f"chunk-{index:06d}.parquet" for index in range(completed_here)
        ]
        expected_receipt_names = [
            _chunk_receipt_name(index) for index in range(completed_here)
        ]
        if (
            parquet_names != expected_parquet_names
            or receipt_names != expected_receipt_names
            or completed_here > expected_chunk_total
        ):
            raise ValueError("capture staging chunk/receipt prefix changed")
        chunk_records: list[dict[str, Any]] = []
        sample_digest = hashlib.sha256()
        sample_backend_digest = hashlib.sha256()
        sample_count = 0
        for index in range(completed_here):
            first = index * PARQUET_BATCH_ROWS
            end = min(first + PARQUET_BATCH_ROWS, stream.row_count)
            path = sample_root / f"chunk-{index:06d}.parquet"
            parquet_file = pq.ParquetFile(path)
            if arrow_schema is None:
                arrow_schema = parquet_file.schema_arrow
            if (
                parquet_file.schema_arrow != arrow_schema
                or tuple(parquet_file.schema_arrow.names) != final_columns
                or parquet_file.num_row_groups != 1
                or parquet_file.metadata.num_rows != end - first
            ):
                raise ValueError("capture staging Parquet schema/range changed")
            chunk_digest = hashlib.sha256()
            chunk_count = 0
            first_logical_id = None
            last_logical_id = None
            for batch in parquet_file.iter_batches(batch_size=PARQUET_BATCH_ROWS):
                names = batch.schema.names
                for row_index in range(batch.num_rows):
                    row = {
                        name: batch.column(column_index)[row_index].as_py()
                        for column_index, name in enumerate(names)
                    }
                    ordinal = first + chunk_count
                    if ordinal >= end:
                        raise ValueError("capture staging chunk exceeds its range")
                    _validate_persisted_capture_row(
                        row, stream.row_at(ordinal), final_columns,
                    )
                    payload = semantic_row_bytes(row)
                    chunk_digest.update(payload)
                    sample_digest.update(payload)
                    prefix_digest.update(payload)
                    sample_backend_digest.update(
                        _backend_projection_bytes(row, backend_columns)
                    )
                    if first_logical_id is None:
                        first_logical_id = row["logical_row_id"]
                    last_logical_id = row["logical_row_id"]
                    chunk_count += 1
            if chunk_count != end - first:
                raise ValueError("capture staging chunk row count changed")
            base_record = {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "sample_sha256": sample,
                "endpoint_id": stream.endpoint_id,
                "chunk_index": index,
                "first_row_ordinal": first,
                "end_row_ordinal_exclusive": end,
                "row_count": chunk_count,
                "first_logical_row_id": first_logical_id,
                "last_logical_row_id": last_logical_id,
                "canonical_row_stream_sha256": chunk_digest.hexdigest(),
                "relative_path": path.name,
                "physical_file_sha256": sha256_file(path),
                "physical_file_bytes": path.stat().st_size,
                "schema_sha256": plan["input_lock"]["prepared"]["streaming_contract"]["schema_sha256"],
            }
            receipt_path = sample_root / _chunk_receipt_name(index)
            expected_receipt = _chunk_completion_receipt(
                plan, shard, base_record,
                str(stable_backend_attestation_sha256),
            )
            if _read_json(receipt_path) != expected_receipt:
                raise ValueError("capture staging immutable chunk receipt changed")
            record = {
                **base_record,
                "completion_receipt_relative_path": receipt_path.name,
                "completion_receipt_sha256": sha256_file(receipt_path),
                "completion_receipt_bytes": receipt_path.stat().st_size,
            }
            receipt_digest.update(canonical_json_bytes(expected_receipt))
            chunk_records.append(record)
            sample_count += chunk_count
            completed_chunks += 1
        complete_sample = completed_here == expected_chunk_total
        if manifest_present and not complete_sample:
            raise ValueError("capture staging published an incomplete sample manifest")
        sample_manifest = _capture_sample_manifest_payload(
            sample, stream.endpoint_id, stream.row_count,
            chunk_records, sample_digest.hexdigest(),
        )
        if manifest_present and _read_json(entries["chunk_manifest.json"]) != sample_manifest:
            raise ValueError("capture staging sample manifest changed")
        sample_states[sample] = {
            "root": sample_root,
            "chunks": chunk_records,
            "semantic_stream_sha256": sample_digest.hexdigest(),
            "backend_projection_sha256": sample_backend_digest.hexdigest(),
            "row_count": sample_count,
            "complete": complete_sample,
            "manifest_present": manifest_present,
        }
        completed_rows += sample_count
        if complete_sample:
            completed_samples += 1
        else:
            first_incomplete_seen = True
            if any(later in observed_sample_dirs for later in expected_samples[sample_index + 1:]):
                raise ValueError("capture staging contains rows beyond incomplete sample")
            break

    if len(safe_uncommitted_orphans) > 1:
        raise ValueError("capture staging contains multiple uncommitted orphans")
    # Mutation is deliberately last: every root/sample entry, every committed
    # receipt, and the entire deterministic prefix has already been validated.
    for orphan_path in safe_uncommitted_orphans:
        orphan_path.unlink()
        directory_fd = os.open(
            orphan_path.parent, os.O_RDONLY | os.O_DIRECTORY,
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    if attempt_local_attestation is not None:
        attempt_local_attestation.unlink()
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    complete = completed_samples == len(expected_samples)
    if (staging / "capture_manifest.json").exists() and (
        not complete or not attestation_path.exists()
    ):
        raise ValueError("capture staging manifest was published before completion")
    if complete:
        next_sample = None
        next_ordinal = 0
        partial_backend_count = 0
        partial_backend_sha = hashlib.sha256().hexdigest()
    else:
        next_sample = expected_samples[completed_samples]
        next_ordinal = sample_states.get(next_sample, {}).get("row_count", 0)
        partial_backend_count = next_ordinal
        partial_backend_sha = sample_states.get(next_sample, {}).get(
            "backend_projection_sha256", hashlib.sha256().hexdigest(),
        )
    cursor_payload = {
        "schema_version": 1,
        "protocol_id": CAPTURE_STAGING_PROTOCOL_ID,
        "plan_sha256": plan["plan_sha256"],
        "shard_id": shard["shard_id"],
        "completed_logical_rows": completed_rows,
        "completed_sample_count": completed_samples,
        "completed_chunk_count": completed_chunks,
        "next_sample_sha256": next_sample,
        "next_row_ordinal": next_ordinal,
        "prefix_semantic_stream_sha256": prefix_digest.hexdigest(),
        "completion_receipt_chain_sha256": receipt_digest.hexdigest(),
        "partial_sample_backend_prefix_row_count": partial_backend_count,
        "partial_sample_backend_prefix_sha256": partial_backend_sha,
        "backend_projection_columns": list(backend_columns),
        "scientific_prefix_reverified": True,
        "resume_only_from_first_incomplete_range": True,
    }
    return {
        "root": staging,
        "exists": True,
        "complete": complete,
        "published_pending": (staging / "capture_manifest.json").exists(),
        "samples": sample_states,
        "expected_streams": expected_streams,
        "arrow_schema": arrow_schema,
        "cursor": {
            **cursor_payload,
            "cursor_sha256": sha256_bytes(canonical_json_bytes(cursor_payload)),
        },
    }


def write_capture_artifacts(
    rows: Iterable[Mapping[str, Any]], output_dir: Path,
    shard: Mapping[str, Any], plan: Mapping[str, Any],
    attestation: Mapping[str, Any], gate: Mapping[str, Any],
    claim: Mapping[str, Any], baseline_dir: Path | None,
    *, path_view: str = "host",
    expected_resume_cursor: Mapping[str, Any] | None = None,
    resume_prefix_verification: MutableMapping[str, Any] | None = None,
    failure_injector: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Resume one immutable shard stream at its first incomplete range.

    Completed Parquet chunks and their completion receipts live in one stable
    per-shard staging directory.  Failures preserve that verified prefix.  A
    retry must emit only the suffix named by the cryptographically bound resume
    cursor; replaying or skipping a scientific row is rejected.
    """

    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    state = _inspect_capture_staging(
        output_dir, plan, shard, path_view=path_view, create=True,
    )
    if state["published_pending"]:
        raise RuntimeError("completed staging manifest awaits recovery publication")
    cursor = state["cursor"]
    if expected_resume_cursor is not None and dict(expected_resume_cursor) != cursor:
        raise RuntimeError("capture staging changed after resume cursor publication")
    if resume_prefix_verification is None:
        raise ValueError("capture writer lacks a resume-prefix verification channel")
    _verify_resume_prefix_channel(
        resume_prefix_verification, cursor, require_complete=False,
    )
    staging = state["root"]
    working = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.work.", dir=output_dir.parent,
    ))
    backend_columns, final_columns, baseline_columns = _audit_columns()
    key_columns = _row_key_columns(final_columns)
    allowed = _allowed_members(plan, shard["task"])
    expected_streams, template_streams = _capture_sample_streams(
        plan, shard, path_view,
    )
    expected_samples = sorted(expected_streams)
    ledger = _KeyLedger(working / "row_keys.sqlite")
    baseline_index = _BaselineIndex(
        baseline_dir, key_columns, baseline_columns, working / "baseline.sqlite",
    ) if baseline_dir else None
    row_count, groups = 0, set()
    previous_order: tuple[str, int] | None = None
    current_sample: str | None = None
    current_chunk_index = 0
    current_sample_count = 0
    chunk_buffer: list[dict[str, Any]] = []
    chunk_records: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    sample_semantic = hashlib.sha256()
    runtime_semantic = hashlib.sha256()
    template_semantic = hashlib.sha256()
    arrow_schema = state.get("arrow_schema")
    sample_provenance: dict[str, tuple[str, str, str, str]] = {}

    from script.analysis.fpct_e1_streaming_verify import semantic_row_bytes

    def flush_chunk() -> None:
        nonlocal current_chunk_index, arrow_schema
        if not chunk_buffer:
            return
        assert current_sample is not None
        import pyarrow as pa
        import pyarrow.parquet as pq

        sample_root = staging / "capture_chunks" / current_sample
        sample_root.mkdir(parents=True, exist_ok=True)
        relative = f"chunk-{current_chunk_index:06d}.parquet"
        path = sample_root / relative
        if path.exists():
            raise FileExistsError(path)
        table = pa.Table.from_pylist(chunk_buffer).select(list(final_columns))
        if arrow_schema is None:
            arrow_schema = table.schema
        elif table.schema != arrow_schema:
            raise ValueError("capture chunk Arrow schema changed")
        temporary_chunk = working / (
            f"{current_sample}-{current_chunk_index:06d}.parquet"
        )
        pq.write_table(
            table, temporary_chunk, row_group_size=PARQUET_BATCH_ROWS,
            compression="zstd", use_dictionary=False,
        )
        with temporary_chunk.open("rb") as durable_chunk:
            os.fsync(durable_chunk.fileno())
        try:
            os.link(temporary_chunk, path)
        except FileExistsError as error:
            raise ValueError("immutable capture chunk already exists") from error
        finally:
            if temporary_chunk.exists():
                temporary_chunk.unlink()
        if failure_injector is not None:
            failure_injector("after_chunk_hardlink_before_receipt", {
                "sample_sha256": current_sample,
                "chunk_index": current_chunk_index,
                "path": str(path),
            })
        first = int(chunk_buffer[0]["row_ordinal"])
        end = int(chunk_buffer[-1]["row_ordinal"]) + 1
        chunk_digest = hashlib.sha256()
        for value in chunk_buffer:
            chunk_digest.update(semantic_row_bytes(value))
        base_record = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "sample_sha256": current_sample,
            "endpoint_id": chunk_buffer[0]["endpoint_id"],
            "chunk_index": current_chunk_index,
            "first_row_ordinal": first,
            "end_row_ordinal_exclusive": end,
            "row_count": len(chunk_buffer),
            "first_logical_row_id": chunk_buffer[0]["logical_row_id"],
            "last_logical_row_id": chunk_buffer[-1]["logical_row_id"],
            "canonical_row_stream_sha256": chunk_digest.hexdigest(),
            "relative_path": relative,
            "physical_file_sha256": sha256_file(path),
            "physical_file_bytes": path.stat().st_size,
            "schema_sha256": plan["input_lock"]["prepared"]["streaming_contract"]["schema_sha256"],
        }
        receipt_path = sample_root / _chunk_receipt_name(current_chunk_index)
        _write_once(
            receipt_path,
            canonical_json_bytes(_chunk_completion_receipt(
                plan, shard, base_record,
                stable_backend_attestation_sha256,
            )),
            temporary_root=working,
            failure_injector=failure_injector,
        )
        chunk_records.append({
            **base_record,
            "completion_receipt_relative_path": receipt_path.name,
            "completion_receipt_sha256": sha256_file(receipt_path),
            "completion_receipt_bytes": receipt_path.stat().st_size,
        })
        current_chunk_index += 1
        chunk_buffer.clear()

    def finish_sample() -> None:
        nonlocal current_sample_count, current_chunk_index, sample_semantic
        if current_sample is None:
            return
        flush_chunk()
        expected_count = expected_streams[current_sample].row_count
        if current_sample_count != expected_count:
            raise ValueError(
                "row-key universe/sample ordinal stream is incomplete: "
                f"{current_sample_count} != {expected_count}"
            )
        if not chunk_records:
            raise ValueError("sample emitted no physical chunks")
        previous_end = 0
        for index, record in enumerate(chunk_records):
            expected_end = min((index + 1) * PARQUET_BATCH_ROWS, expected_count)
            if (
                record["chunk_index"] != index
                or record["first_row_ordinal"] != previous_end
                or record["end_row_ordinal_exclusive"] != expected_end
                or record["row_count"] != expected_end - previous_end
                or record["row_count"] > PARQUET_BATCH_ROWS
            ):
                raise ValueError("sample chunk ranges are not the deterministic bounded partition")
            previous_end = expected_end
        sample_manifest = _capture_sample_manifest_payload(
            current_sample, chunk_records[0]["endpoint_id"], expected_count,
            chunk_records, sample_semantic.hexdigest(),
        )
        path = staging / "capture_chunks" / current_sample / "chunk_manifest.json"
        _write_once(
            path, canonical_json_bytes(sample_manifest),
            temporary_root=working, failure_injector=failure_injector,
        )
        sample_records.append({
            "sample_sha256": current_sample,
            "manifest_relative_path": str(path.relative_to(staging)),
            "manifest_sha256": sha256_file(path),
            "manifest_bytes": path.stat().st_size,
            "expected_logical_rows": expected_count,
            "semantic_stream_sha256": sample_manifest["semantic_stream_sha256"],
            "chunk_count": len(chunk_records),
        })
        current_sample_count = 0
        current_chunk_index = 0
        chunk_records.clear()
        sample_semantic = hashlib.sha256()

    def absorb_final_row(
        final: Mapping[str, Any], *, expected: Mapping[str, Any],
        mechanically_rejoin: bool,
    ) -> None:
        """Re-attest either an immutable prefix row or a new suffix row."""

        nonlocal row_count, previous_order, current_sample_count
        sample = str(final["sample_sha256"])
        ordinal = int(final["row_ordinal"])
        order = (sample, ordinal)
        if previous_order is not None and order <= previous_order:
            raise ValueError("capture rows are not a strict sample/ordinal stream")
        _validate_persisted_capture_row(final, expected, final_columns)
        provenance = tuple(
            str(final[name]) for name in (
                "input_sha256", "alignment_sha256", "labels_sha256",
                "gold_response_sha256",
            )
        )
        if sample in sample_provenance and sample_provenance[sample] != provenance:
            raise ValueError("input/alignment/labels/gold-response SHA changed within a sample")
        sample_provenance[sample] = provenance
        key = _key_bytes(final, key_columns)
        ledger.add(key)
        if mechanically_rejoin:
            local = {name: final[name] for name in backend_columns}
            if baseline_index is None:
                joined = {
                    name: local[name.removeprefix("cpost_")]
                    for name in baseline_columns
                }
            else:
                joined = baseline_index.join(key)
            rebuilt = _finalize_row(local, joined, final_columns)
            if rebuilt != dict(final):
                raise ValueError("immutable capture prefix no longer mechanically rejoins")
        semantic = semantic_row_bytes(final)
        runtime_semantic.update(semantic)
        sample_semantic.update(semantic)
        template_semantic.update(semantic_row_bytes(
            template_streams[sample].row_at(ordinal)
        ))
        groups.add(str(final["content_group_sha256"]))
        row_count += 1
        current_sample_count += 1
        previous_order = order

    # Replay every completed prefix row from its immutable chunk and rebuild
    # ledgers/joins/digests.  The resume cursor is never trusted by itself.
    import pyarrow.parquet as pq
    for sample in expected_samples:
        sample_state = state["samples"].get(sample)
        if sample_state is None:
            break
        current_sample = sample
        current_sample_count = 0
        current_chunk_index = len(sample_state["chunks"])
        chunk_records = list(sample_state["chunks"])
        sample_semantic = hashlib.sha256()
        for chunk in chunk_records:
            path = sample_state["root"] / chunk["relative_path"]
            first = int(chunk["first_row_ordinal"])
            offset = 0
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=PARQUET_BATCH_ROWS,
            ):
                names = batch.schema.names
                for row_index in range(batch.num_rows):
                    final = {
                        name: batch.column(column_index)[row_index].as_py()
                        for column_index, name in enumerate(names)
                    }
                    absorb_final_row(
                        final,
                        expected=expected_streams[sample].row_at(first + offset),
                        mechanically_rejoin=True,
                    )
                    offset += 1
        if sample_state["complete"]:
            finish_sample()
            current_sample = None
        else:
            if current_sample_count != cursor["next_row_ordinal"]:
                raise RuntimeError("verified prefix differs from its resume cursor")
            break

    stable_attestation = {
        key: value for key, value in attestation.items()
        if key not in {
            "resume_cursor_sha256", "resume_completed_logical_rows",
            "resume_next_sample_sha256", "resume_next_row_ordinal",
        }
    }
    stable_attestation = _validate_stable_backend_attestation(
        stable_attestation, plan, shard,
    )
    stable_backend_attestation_sha256 = sha256_bytes(
        canonical_json_bytes(stable_attestation)
    )
    attestation_record = {
        "schema_version": 1,
        "protocol_id": CAPTURE_STAGING_PROTOCOL_ID,
        "artifact_type": "stable_backend_attestation",
        "plan_sha256": plan["plan_sha256"],
        "shard_id": shard["shard_id"],
        "attestation": stable_attestation,
    }
    _write_once(
        staging / "backend_attestation.json",
        canonical_json_bytes(attestation_record),
        temporary_root=working,
        failure_injector=failure_injector,
    )
    if (
        attestation.get("resume_cursor_sha256") != cursor["cursor_sha256"]
        or attestation.get("resume_completed_logical_rows")
        != cursor["completed_logical_rows"]
        or attestation.get("resume_next_sample_sha256")
        != cursor["next_sample_sha256"]
        or attestation.get("resume_next_row_ordinal")
        != cursor["next_row_ordinal"]
    ):
        raise ValueError("backend did not attest the exact capture resume cursor")

    try:
        first_backend_row = True
        for raw in rows:
            local = _validate_backend_row(raw, shard, allowed, backend_columns)
            provenance = tuple(local[name] for name in ("input_sha256", "alignment_sha256", "labels_sha256", "gold_response_sha256"))
            sample = local["sample_sha256"]
            if sample in sample_provenance and sample_provenance[sample] != provenance:
                raise ValueError("input/alignment/labels/gold-response SHA changed within a sample")
            sample_provenance[sample] = provenance
            ordinal = int(local["row_ordinal"])
            order = (sample, ordinal)
            if first_backend_row:
                _verify_resume_prefix_channel(
                    resume_prefix_verification, cursor,
                    require_complete=True,
                )
                if order != (
                    cursor["next_sample_sha256"], cursor["next_row_ordinal"],
                ):
                    raise ValueError("backend did not resume at the first incomplete range")
                first_backend_row = False
            if sample not in expected_streams:
                raise ValueError("backend row has no frozen compact sample geometry")
            if current_sample != sample:
                finish_sample()
                current_sample = sample
                if ordinal != 0:
                    raise ValueError("each sample ordinal stream must start at zero")
            if ordinal != current_sample_count:
                raise ValueError("sample row_ordinal contains a gap or duplicate")
            expected = expected_streams[sample].row_at(ordinal)
            template_expected = template_streams[sample].row_at(ordinal)
            identity_columns = (*ROW_TEMPLATE_COLUMNS, "row_ordinal", "logical_row_id", "endpoint_id", "endpoint_row_id")
            if any(local[name] != expected[name] for name in identity_columns):
                raise ValueError("runtime row differs from frozen compact ordinal geometry")
            if (
                len(local["prior"]) != len(expected["prior"])
                or any(abs(float(left) - float(right)) > 2e-5 for left, right in zip(local["prior"], expected["prior"]))
                or local["candidate_indices"] != expected["candidate_indices"]
                or local["candidate_valid_mask"] != expected["candidate_valid_mask"]
                or len(local["candidate_slot_weights"])
                != len(expected["candidate_slot_weights"])
                or any(
                    abs(float(left) - float(right)) > 2e-5
                    for left, right in zip(
                        local["candidate_slot_weights"],
                        expected["candidate_slot_weights"],
                    )
                )
                or abs(float(local["statistical_weight"]) - float(expected["statistical_weight"])) > 1e-12
            ):
                raise ValueError("runtime candidate geometry/weight differs from frozen compact geometry")
            key = _key_bytes(local, key_columns)
            if baseline_index is None:
                joined = {name: local[name.removeprefix("cpost_")] for name in baseline_columns}
            else:
                joined = baseline_index.join(key)
            final = _finalize_row(local, joined, final_columns)
            absorb_final_row(final, expected=expected, mechanically_rejoin=False)
            chunk_buffer.append(final)
            if len(chunk_buffer) >= PARQUET_BATCH_ROWS:
                flush_chunk()
        if first_backend_row and not state["complete"]:
            raise ValueError("backend emitted no rows for the incomplete capture range")
        if not first_backend_row and state["complete"]:
            raise ValueError("backend emitted rows after the capture prefix was complete")
        _verify_resume_prefix_channel(
            resume_prefix_verification, cursor, require_complete=True,
        )
        finish_sample()
        if baseline_index is not None: baseline_index.assert_bijection()
        if row_count == 0 or groups != set(allowed) or [record["sample_sha256"] for record in sample_records] != expected_samples:
            raise ValueError("capture does not cover every frozen content group")
        key_count, key_sha = ledger.attest()
        expected_template = shard["expected_row_template"]
        if (
            row_count != expected_template.get("count")
            or template_semantic.hexdigest() != expected_template.get("semantic_stream_sha256")
        ):
            raise ValueError("runtime logical universe differs from frozen CPU input-lock stream")
        schema = {
            "backend_columns": list(backend_columns),
            "final_columns": list(final_columns),
            "baseline_columns": list(baseline_columns),
            "row_key_columns": list(key_columns),
            "baseline_join_excludes": ["row_ordinal", "logical_row_id", "endpoint_id", "endpoint_row_id"],
            "physical_chunk_rows": PARQUET_BATCH_ROWS,
            "stream_order": "sample_sha256,row_ordinal",
        }
        sample_manifest_tree_sha = sha256_bytes(canonical_json_bytes(sample_records))
        manifest = {
            "schema_version": SCHEMA_VERSION, "protocol_id": PROTOCOL_ID, "status": "GO", "shard": dict(shard),
            "plan_sha256": plan["plan_sha256"], "execution_sha": plan["runtime_lock"]["execution_sha"], "image_digest": plan["runtime_lock"]["image_digest"],
            "instrumentation_gate": dict(gate), "checkpoint_tree_sha256": shard["checkpoint_tree_sha256"], "attestation": stable_attestation, "claim": dict(claim),
            "schema_boundary": {**schema, "sha256": sha256_bytes(canonical_json_bytes(schema)), "backend_supplied_cpost_fields": False},
            "row_key_attestation": {"count": key_count, "sha256": key_sha, "exact_bijection_with_cpost": baseline_index is not None or shard["inference_operator"] == "c_post"},
            "input_lock_row_template": dict(expected_template),
            "streaming_receipt": {
                "order": "sample_sha256,row_ordinal",
                "physical_chunk_rows": PARQUET_BATCH_ROWS,
                "logical_row_count": row_count,
                "semantic_stream_sha256": runtime_semantic.hexdigest(),
                "sample_manifest_tree_sha256": sample_manifest_tree_sha,
                "sample_count": len(sample_records),
                "missing_rows": 0,
                "duplicate_rows": 0,
                "overlapping_chunks": 0,
                "resume_cursor_at_start": dict(cursor),
                "immutable_prefix_rows_reused": cursor["completed_logical_rows"],
            },
            "sample_streams": sample_records,
            "sample_provenance_sha256": sha256_bytes(canonical_json_bytes(sorted((sample, *values) for sample, values in sample_provenance.items()))),
            "row_count": row_count, "content_group_count": len(groups),
            "parquet_chunk_count": sum(record["chunk_count"] for record in sample_records),
            "artifacts": {"sample_chunk_manifest_tree": {"sha256": sample_manifest_tree_sha, "count": len(sample_records)}},
            "e1_pilot_consumed": False, "model_selection_consumed": False, "test_consumed": False,
        }
        _write_once(
            staging / "capture_manifest.json", canonical_json_bytes(manifest),
            temporary_root=working, failure_injector=failure_injector,
        )
        os.replace(staging, output_dir)
        return manifest
    finally:
        ledger.close()
        if baseline_index is not None:
            baseline_index.close()
        shutil.rmtree(working, ignore_errors=True)


def verify_capture_artifacts(
    output_dir: Path, plan: Mapping[str, Any], *, path_view: str = "host",
) -> dict[str, Any]:
    manifest = _read_json(output_dir / "capture_manifest.json")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status") != "GO"
        or manifest.get("plan_sha256") != plan["plan_sha256"]
        or manifest.get("e1_pilot_consumed") is not False
        or manifest.get("model_selection_consumed") is not False
        or manifest.get("test_consumed") is not False
    ):
        raise ValueError("capture artifact identity mismatch")
    shards = {r["shard_id"]: r for r in plan["shards"]}
    shard_id = manifest.get("shard", {}).get("shard_id")
    if shard_id not in shards or manifest["shard"] != shards[shard_id]:
        raise ValueError("capture shard attestation mismatch")
    backend, final_columns, _ = _audit_columns(); del backend
    key_columns = _row_key_columns(final_columns)
    boundary = manifest.get("schema_boundary", {})
    boundary_payload = {key: value for key, value in boundary.items() if key not in {"sha256", "backend_supplied_cpost_fields"}}
    if (
        boundary.get("sha256") != sha256_bytes(canonical_json_bytes(boundary_payload))
        or boundary.get("backend_supplied_cpost_fields") is not False
        or boundary_payload.get("row_key_columns") != list(key_columns)
        or boundary_payload.get("baseline_join_excludes")
        != ["row_ordinal", "logical_row_id", "endpoint_id", "endpoint_row_id"]
        or boundary_payload.get("physical_chunk_rows") != PARQUET_BATCH_ROWS
        or boundary_payload.get("stream_order") != "sample_sha256,row_ordinal"
    ):
        raise ValueError("capture schema/baseline-join boundary changed")
    shard = shards[shard_id]
    expected_root_entries = {
        "capture_staging_identity.json", "backend_attestation.json",
        "capture_manifest.json", "capture_chunks",
    }
    if (
        {child.name for child in output_dir.iterdir()} != expected_root_entries
        or any(child.is_symlink() for child in output_dir.iterdir())
        or not (output_dir / "capture_chunks").is_dir()
        or any(
            not (output_dir / name).is_file()
            for name in expected_root_entries - {"capture_chunks"}
        )
        or _read_json(output_dir / "capture_staging_identity.json")
        != _capture_staging_identity(plan, shard)
    ):
        raise ValueError("capture artifact root contains corruption or orphans")
    stable_attestation = _read_json(output_dir / "backend_attestation.json")
    if stable_attestation != {
        "schema_version": 1,
        "protocol_id": CAPTURE_STAGING_PROTOCOL_ID,
        "artifact_type": "stable_backend_attestation",
        "plan_sha256": plan["plan_sha256"],
        "shard_id": shard_id,
        "attestation": manifest.get("attestation"),
    }:
        raise ValueError("capture stable backend attestation changed")
    verified_stable_attestation = _validate_stable_backend_attestation(
        manifest["attestation"], plan, shard,
    )
    stable_backend_attestation_sha256 = sha256_bytes(
        canonical_json_bytes(verified_stable_attestation)
    )
    expected_streams, template_streams = _capture_sample_streams(
        plan, shard, path_view,
    )
    expected_samples = sorted(expected_streams)
    verification_root = Path(tempfile.mkdtemp(prefix="fpct-e1-verify-"))
    ledger = _KeyLedger(verification_root / "row-keys.sqlite")
    count, groups = 0, set()
    runtime_semantic = hashlib.sha256()
    template_semantic = hashlib.sha256()
    sample_records = manifest.get("sample_streams")
    if (
        not isinstance(sample_records, list)
        or [record.get("sample_sha256") for record in sample_records] != expected_samples
    ):
        raise ValueError("capture sample manifest order/universe changed")
    capture_chunks_root = output_dir / "capture_chunks"
    if (
        {child.name for child in capture_chunks_root.iterdir()}
        != set(expected_samples)
        or any(
            child.is_symlink() or not child.is_dir()
            for child in capture_chunks_root.iterdir()
        )
    ):
        raise ValueError("capture chunk root contains corruption or orphan samples")
    from script.analysis.fpct_e1_streaming_verify import semantic_row_bytes
    import pyarrow.parquet as pq
    try:
        for sample_record in sample_records:
            sample = sample_record["sample_sha256"]
            manifest_relative = Path(str(sample_record.get("manifest_relative_path", "")))
            if manifest_relative.is_absolute() or ".." in manifest_relative.parts or not manifest_relative.parts:
                raise ValueError("capture sample manifest path escapes its artifact root")
            sample_manifest_path = output_dir / manifest_relative
            if (
                not sample_manifest_path.is_file()
                or sha256_file(sample_manifest_path) != sample_record.get("manifest_sha256")
                or sample_manifest_path.stat().st_size != sample_record.get("manifest_bytes")
            ):
                raise ValueError("capture sample chunk manifest provenance mismatch")
            sample_manifest = _read_json(sample_manifest_path)
            expected_count = expected_streams[sample].row_count
            chunks = sample_manifest.get("chunks")
            if (
                sample_manifest.get("schema_version") != SCHEMA_VERSION
                or sample_manifest.get("protocol_id") != PROTOCOL_ID
                or sample_manifest.get("artifact_type") != "capture_sample_chunk_manifest"
                or sample_manifest.get("sample_sha256") != sample
                or sample_manifest.get("expected_logical_rows") != expected_count
                or sample_manifest.get("emitted_logical_rows") != expected_count
                or sample_manifest.get("physical_chunk_rows") != PARQUET_BATCH_ROWS
                or sample_manifest.get("endpoint_id")
                != expected_streams[sample].endpoint_id
                or any(sample_manifest.get(name) != 0 for name in ("missing_rows", "duplicate_rows", "overlapping_chunks"))
                or sample_manifest.get("ordinal_ranges_contiguous") is not True
                or not isinstance(chunks, list)
                or len(chunks) != (expected_count + PARQUET_BATCH_ROWS - 1) // PARQUET_BATCH_ROWS
            ):
                raise ValueError("capture sample chunk manifest identity/ranges changed")
            expected_sample_files = {"chunk_manifest.json"}
            for chunk in chunks:
                expected_sample_files.add(str(chunk.get("relative_path", "")))
                expected_sample_files.add(str(
                    chunk.get("completion_receipt_relative_path", "")
                ))
            if (
                {child.name for child in sample_manifest_path.parent.iterdir()}
                != expected_sample_files
                or any(
                    child.is_symlink() or not child.is_file()
                    for child in sample_manifest_path.parent.iterdir()
                )
            ):
                raise ValueError("capture sample tree contains corruption or orphans")
            sample_digest = hashlib.sha256()
            sample_count = 0
            relative_paths: set[str] = set()
            for index, chunk in enumerate(chunks):
                first = index * PARQUET_BATCH_ROWS
                end = min(first + PARQUET_BATCH_ROWS, expected_count)
                relative_path = str(chunk.get("relative_path", ""))
                portable = Path(relative_path)
                path = sample_manifest_path.parent / portable
                receipt_relative = str(
                    chunk.get("completion_receipt_relative_path", "")
                )
                receipt_portable = Path(receipt_relative)
                receipt_path = sample_manifest_path.parent / receipt_portable
                base_record = {
                    key: value for key, value in chunk.items()
                    if key not in {
                        "completion_receipt_relative_path",
                        "completion_receipt_sha256",
                        "completion_receipt_bytes",
                    }
                }
                if (
                    chunk.get("schema_version") != SCHEMA_VERSION
                    or chunk.get("protocol_id") != PROTOCOL_ID
                    or chunk.get("chunk_index") != index
                    or chunk.get("first_row_ordinal") != first
                    or chunk.get("end_row_ordinal_exclusive") != end
                    or chunk.get("row_count") != end - first
                    or chunk.get("sample_sha256") != sample
                    or chunk.get("endpoint_id") != expected_streams[sample].endpoint_id
                    or chunk.get("schema_sha256") != plan["input_lock"]["prepared"]["streaming_contract"]["schema_sha256"]
                    or portable.is_absolute()
                    or ".." in portable.parts
                    or not portable.parts
                    or relative_path in relative_paths
                    or receipt_portable.is_absolute()
                    or ".." in receipt_portable.parts
                    or len(receipt_portable.parts) != 1
                    or receipt_relative != _chunk_receipt_name(index)
                    or not path.is_file()
                    or sha256_file(path) != chunk.get("physical_file_sha256")
                    or path.stat().st_size != chunk.get("physical_file_bytes")
                    or not receipt_path.is_file()
                    or receipt_path.is_symlink()
                    or sha256_file(receipt_path)
                    != chunk.get("completion_receipt_sha256")
                    or receipt_path.stat().st_size
                    != chunk.get("completion_receipt_bytes")
                    or _read_json(receipt_path)
                    != _chunk_completion_receipt(
                        plan, shard, base_record,
                        stable_backend_attestation_sha256,
                    )
                ):
                    raise ValueError("capture provenance mismatch: chunk/range")
                relative_paths.add(relative_path)
                parquet_file = pq.ParquetFile(path)
                if tuple(parquet_file.schema_arrow.names) != final_columns:
                    raise ValueError("capture chunk parquet schema/order changed")
                if parquet_file.num_row_groups != 1 or parquet_file.metadata.num_rows != end - first:
                    raise ValueError("capture chunk is not one bounded physical row group")
                chunk_digest = hashlib.sha256()
                chunk_count = 0
                for batch in parquet_file.iter_batches(batch_size=PARQUET_BATCH_ROWS):
                    for row in batch.to_pylist():
                        ordinal = first + chunk_count
                        if ordinal >= end:
                            raise ValueError("capture chunk contains rows beyond its range")
                        from script.analysis.fpct_e1_mechanism_audit import validate_and_derive_row
                        validate_and_derive_row(row)
                        expected = expected_streams[sample].row_at(ordinal)
                        identity_columns = (*ROW_TEMPLATE_COLUMNS, "row_ordinal", "logical_row_id", "endpoint_id", "endpoint_row_id")
                        if any(row[name] != expected[name] for name in identity_columns):
                            raise ValueError("capture row differs from frozen compact ordinal geometry")
                        if (
                            len(row["prior"]) != len(expected["prior"])
                            or any(abs(float(left) - float(right)) > 2e-5 for left, right in zip(row["prior"], expected["prior"]))
                            or row["candidate_indices"] != expected["candidate_indices"]
                            or row["candidate_valid_mask"] != expected["candidate_valid_mask"]
                            or len(row["candidate_slot_weights"]) != len(expected["candidate_slot_weights"])
                            or any(abs(float(left) - float(right)) > 2e-5 for left, right in zip(row["candidate_slot_weights"], expected["candidate_slot_weights"]))
                            or abs(float(row["statistical_weight"]) - float(expected["statistical_weight"])) > 1e-12
                        ):
                            raise ValueError("capture candidate geometry/weight differs from frozen compact geometry")
                        semantic = semantic_row_bytes(row)
                        chunk_digest.update(semantic); sample_digest.update(semantic); runtime_semantic.update(semantic)
                        template_semantic.update(semantic_row_bytes(template_streams[sample].row_at(ordinal)))
                        ledger.add(_key_bytes(row, key_columns)); count += 1; sample_count += 1; chunk_count += 1
                        groups.add(row["content_group_sha256"])
                if (
                    chunk_count != end - first
                    or chunk_digest.hexdigest() != chunk.get("canonical_row_stream_sha256")
                    or chunk.get("first_logical_row_id")
                    != expected_streams[sample].row_at(first)["logical_row_id"]
                    or chunk.get("last_logical_row_id")
                    != expected_streams[sample].row_at(end - 1)["logical_row_id"]
                ):
                    raise ValueError("capture chunk row count/semantic SHA mismatch")
            if (
                sample_count != expected_count
                or sample_digest.hexdigest() != sample_manifest.get("semantic_stream_sha256")
                or sample_record.get("expected_logical_rows") != expected_count
                or sample_record.get("semantic_stream_sha256") != sample_digest.hexdigest()
                or sample_record.get("chunk_count") != len(chunks)
            ):
                raise ValueError("capture sample semantic stream mismatch")
        key_count, key_sha = ledger.attest()
    finally:
        ledger.close()
        shutil.rmtree(verification_root, ignore_errors=True)
    if count != manifest.get("row_count") or groups != set(_allowed_members(plan, shards[shard_id]["task"])):
        raise ValueError("row/group count mismatch")
    if manifest.get("row_key_attestation") != {"count": key_count, "sha256": key_sha, "exact_bijection_with_cpost": manifest["row_key_attestation"]["exact_bijection_with_cpost"]}:
        raise ValueError("row-key attestation mismatch")
    if (
        manifest.get("input_lock_row_template") != shard["expected_row_template"]
        or count != shard["expected_row_template"].get("count")
        or template_semantic.hexdigest()
        != shard["expected_row_template"].get("semantic_stream_sha256")
    ):
        raise ValueError("CPU input-lock row-template attestation mismatch")
    receipt = manifest.get("streaming_receipt", {})
    sample_tree_sha = sha256_bytes(canonical_json_bytes(sample_records))
    resume_cursor = receipt.get("resume_cursor_at_start", {})
    resume_cursor_payload = {
        key: value for key, value in resume_cursor.items()
        if key != "cursor_sha256"
    } if isinstance(resume_cursor, Mapping) else {}
    if (
        receipt.get("order") != "sample_sha256,row_ordinal"
        or receipt.get("physical_chunk_rows") != PARQUET_BATCH_ROWS
        or receipt.get("logical_row_count") != count
        or receipt.get("semantic_stream_sha256") != runtime_semantic.hexdigest()
        or receipt.get("sample_manifest_tree_sha256") != sample_tree_sha
        or receipt.get("sample_count") != len(expected_samples)
        or not isinstance(resume_cursor, Mapping)
        or resume_cursor.get("cursor_sha256")
        != sha256_bytes(canonical_json_bytes(resume_cursor_payload))
        or resume_cursor.get("protocol_id") != CAPTURE_STAGING_PROTOCOL_ID
        or resume_cursor.get("plan_sha256") != plan["plan_sha256"]
        or resume_cursor.get("shard_id") != shard_id
        or receipt.get("immutable_prefix_rows_reused")
        != resume_cursor.get("completed_logical_rows")
        or not isinstance(receipt.get("immutable_prefix_rows_reused"), int)
        or not 0 <= receipt["immutable_prefix_rows_reused"] <= count
        or any(receipt.get(name) != 0 for name in ("missing_rows", "duplicate_rows", "overlapping_chunks"))
    ):
        raise ValueError("capture streaming receipt mismatch")
    return {
        "status": "GO", "shard_id": shard_id, "row_count": count,
        "row_key_sha256": key_sha,
        "semantic_stream_sha256": runtime_semantic.hexdigest(),
        "sample_manifest_tree_sha256": sample_tree_sha,
        "parquet_chunk_count": manifest["parquet_chunk_count"],
        "e1_pilot_consumed": False,
    }


def _gate_path(plan: Mapping[str, Any], supplied: Path | None, view: str) -> Path:
    frozen = plan["instrumentation_gate"]
    path = supplied if supplied is not None else resolve_logical_path(plan, frozen["logical_path"], view)
    verified = verify_instrumentation_gate(
        path,
        frozen["sha256"],
        evidence_resolver=lambda logical: resolve_logical_path(plan, logical, view),
    )
    if verified.get("evidence") != frozen.get("evidence"):
        raise RuntimeError("instrumentation gate evidence closure changed")
    return path


def execute_shard(plan_path: Path, shard_id: str, gate_path: Path | None, output_root: Path, backend_spec: str | None, *, path_view: str = "host", expected_plan_sha256: str | None = None, claim_id: str = "__CLAIM_ID__", finalized_lock_sha256: str | None = None, finalized_receipt_path: Path | None = None, backend_loader: Callable[[str], Callable[[Mapping[str, Any]], Any]] = _load_backend) -> dict[str, Any]:
    plan = _read_json(plan_path)
    if (
        not _is_sha256(expected_plan_sha256)
        or expected_plan_sha256 != plan.get("plan_sha256")
        or expected_plan_sha256 != _plan_hash(plan)
    ):
        raise RuntimeError("mounted execution plan differs from rendered expected SHA")
    validate_execution_plan(plan, require_checkpoints=True, for_execution=True)
    shards = {r["shard_id"]: r for r in plan["shards"]}
    if shard_id not in shards: raise ValueError(f"unknown shard: {shard_id}")
    shard = shards[shard_id]
    if claim_id != _expected_claim_id(plan["plan_sha256"], shard_id):
        raise RuntimeError("run-shard claim ID differs from the rendered plan/shard identity")
    output_dir = output_root / shard["output_relative"]
    verify_plan_sources(plan, path_view, shard["checkpoint_id"])
    frozen_gate_path = _gate_path(plan, gate_path, path_view)
    required_phase = shard.get("requires_phase_completion")
    if required_phase is not None:
        if required_phase == FINALIZED_E1_2:
            if not _is_sha256(finalized_lock_sha256):
                raise RuntimeError("E1-3 shard lacks the rendered finalized-lock SHA")
            _verify_finalized_e1_2_lock(
                plan, output_root, path_view=path_view,
                expected_lock_sha256=finalized_lock_sha256, deep=False,
                receipt_path=finalized_receipt_path,
            )
        else:
            _verify_phase_marker(plan, output_root, required_phase)
    dependency = shard.get("depends_on_shard")
    baseline_dir = output_root / shards[dependency]["output_relative"] if dependency else None
    if baseline_dir is not None:
        verify_capture_artifacts(baseline_dir, plan, path_view=path_view)
    if output_dir.exists():
        verified = verify_capture_artifacts(output_dir, plan, path_view=path_view)
        _reconcile_completed_claim(
            _claim_path(output_root, shard), plan, shard, claim_id,
            output_dir / "capture_manifest.json",
        )
        return {**verified, "resumed_without_model_load": True}
    existing_staging = _capture_staging_path(output_dir)
    if existing_staging.exists():
        staged = _inspect_capture_staging(
            output_dir, plan, shard, path_view=path_view,
        )
        if staged["published_pending"]:
            verified = verify_capture_artifacts(
                existing_staging, plan, path_view=path_view,
            )
            os.replace(existing_staging, output_dir)
            _reconcile_completed_claim(
                _claim_path(output_root, shard), plan, shard, claim_id,
                output_dir / "capture_manifest.json",
            )
            return {**verified, "resumed_without_model_load": True}
    lease = _acquire_claim(_claim_path(output_root, shard), plan, shard, claim_id)
    try:
        resume_state = _inspect_capture_staging(
            output_dir, plan, shard, path_view=path_view, create=True,
        )
        resume_cursor = resume_state["cursor"]
        if resume_state["complete"]:
            attestation_record = _read_json(
                resume_state["root"] / "backend_attestation.json"
            )
            stable_attestation = dict(attestation_record["attestation"])
            resumed_attestation = {
                **stable_attestation,
                "resume_cursor_sha256": resume_cursor["cursor_sha256"],
                "resume_completed_logical_rows": resume_cursor["completed_logical_rows"],
                "resume_next_sample_sha256": resume_cursor["next_sample_sha256"],
                "resume_next_row_ordinal": resume_cursor["next_row_ordinal"],
            }
            completed_prefix_channel = {
                "schema_version": 1,
                "protocol_id": CAPTURE_STAGING_PROTOCOL_ID,
                "cursor_sha256": resume_cursor["cursor_sha256"],
                "expected_row_count": 0,
                "expected_backend_projection_sha256": hashlib.sha256().hexdigest(),
                "status": "GO_EXACT_PREFIX_MATCH",
                "observed_row_count": 0,
                "observed_backend_projection_sha256": hashlib.sha256().hexdigest(),
                "exact_match": True,
            }
            result = write_capture_artifacts(
                (), output_dir, shard, plan, resumed_attestation,
                plan["instrumentation_gate"], lease.record, baseline_dir,
                path_view=path_view, expected_resume_cursor=resume_cursor,
                resume_prefix_verification=completed_prefix_channel,
            )
            lease.transition(
                "SUCCEEDED",
                artifact_sha256=sha256_file(
                    output_dir / "capture_manifest.json"
                ),
            )
            return {**result, "resumed_without_model_load": True}
        spec = backend_spec or plan["runtime_lock"]["backend"]["spec"]
        if spec != plan["runtime_lock"]["backend"]["spec"]: raise RuntimeError("backend spec differs from frozen plan")
        backend = backend_loader(spec)
        request = {
        "contract_version": CAPTURE_BACKEND_CONTRACT_VERSION, "plan_sha256": plan["plan_sha256"], "execution_sha": plan["runtime_lock"]["execution_sha"],
        "image_digest": plan["runtime_lock"]["image_digest"], "split_role": ALLOWED_SPLIT_ROLE, "shard": dict(shard),
        "checkpoint_path": str(resolve_logical_path(plan, shard["checkpoint_logical_path"], path_view)),
        "e0_data_root": str(resolve_logical_path(plan, plan["input_lock"]["data_root"], path_view)),
        "input_lock_manifest_path": str(resolve_logical_path(plan, plan["input_lock"]["prepared"]["manifest"]["logical_path"], path_view)),
        "input_lock_manifest_sha256": plan["input_lock"]["prepared"]["manifest"]["sha256"],
        "input_lock_sidecar_path": str(resolve_logical_path(plan, plan["input_lock"]["prepared"]["sidecar"]["logical_path"], path_view)),
        "input_lock_sidecar_sha256": plan["input_lock"]["prepared"]["sidecar"]["sha256"],
        "expected_row_template": dict(shard["expected_row_template"]),
        "runtime_assets": plan["input_lock"]["prepared"]["runtime_assets"],
        "e0_config_path": str(resolve_logical_path(plan, shard["e0_config_logical_path"], path_view)),
        "group_contract": plan["group_contract"][shard["task"]], "membership_sha256": shard["membership_sha256"],
        "baseline_capture_dir": str(baseline_dir) if baseline_dir else None, "teacher_forcing": dict(plan["teacher_forcing"]),
        "schema_boundary": {"backend_columns": list(_audit_columns()[0]), "forbidden_prefix": "cpost_", "stores_raw_kv": False},
        "instrumentation_gate_path": str(frozen_gate_path),
        "resume_cursor": dict(resume_cursor),
        }
        rows, attestation, resume_prefix_verification = _normalize_backend_result(
            backend(request)
        )
        expected_attestation = {"plan_sha256": plan["plan_sha256"], "shard_id": shard_id, "execution_sha": plan["runtime_lock"]["execution_sha"], "image_digest": plan["runtime_lock"]["image_digest"], "checkpoint_tree_sha256": shard["checkpoint_tree_sha256"], "membership_sha256": shard["membership_sha256"]}
        for name, expected in expected_attestation.items():
            if attestation.get(name) != expected: raise ValueError(f"backend provenance attestation mismatch: {name}")
        result = write_capture_artifacts(
            rows, output_dir, shard, plan, attestation,
            plan["instrumentation_gate"], lease.record, baseline_dir,
            path_view=path_view, expected_resume_cursor=resume_cursor,
            resume_prefix_verification=resume_prefix_verification,
        )
        lease.transition("SUCCEEDED", artifact_sha256=sha256_file(output_dir / "capture_manifest.json"))
        return result
    except BaseException as error:
        try:
            lease.transition("FAILED", error_class=type(error).__name__)
        except Exception as claim_error:
            if hasattr(error, "add_note"):
                error.add_note(f"claim failure transition also failed: {type(claim_error).__name__}")
        raise
    finally:
        lease.close()


def _selected_shards(plan: Mapping[str, Any], stage: str) -> list[Mapping[str, Any]]:
    if stage == PHASE_E1_2_BASELINES:
        return sorted((r for r in plan["shards"] if r.get("phase") == PHASE_E1_2_BASELINES), key=lambda r: r["shard_id"])
    if stage == PHASE_E1_2_FACTORIZED:
        return sorted((r for r in plan["shards"] if r.get("phase") == PHASE_E1_2_FACTORIZED), key=lambda r: r["shard_id"])
    if stage == STAGE_E1_2: return sorted((r for r in plan["shards"] if r["stage"] == STAGE_E1_2), key=lambda r: r["shard_id"])
    if stage == STAGE_E1_3: return sorted(plan["shards"], key=lambda r: r["shard_id"])
    raise ValueError(f"unknown closure stage: {stage}")


def verify_all(plan_path: Path, output_root: Path, stage: str, merge_dir: Path | None = None, write_completion: bool = False) -> dict[str, Any]:
    plan = _read_json(plan_path); validate_execution_plan(plan, for_execution=True)
    shards = _selected_shards(plan, stage)
    records = _closure_records(plan, output_root, stage, deep=True)
    if stage == PHASE_E1_2_BASELINES and len(records) != 18: raise RuntimeError("18-shard C_post baseline closure failed")
    if stage == PHASE_E1_2_FACTORIZED and len(records) != 18: raise RuntimeError("18-shard F endpoint closure failed")
    if stage == STAGE_E1_2:
        _verify_phase_marker(plan, output_root, PHASE_E1_2_BASELINES)
        if len(records) != EXPECTED_ENDPOINT_SHARDS: raise RuntimeError("36-shard endpoint closure failed")
    if stage == STAGE_E1_3:
        _verify_finalized_e1_2_lock(plan, output_root, deep=True)
        observed = {(r["seed"], r["checkpoint_arm"], r["task"]): set() for r in plan["shards"]}
        for shard in shards: observed[(shard["seed"], shard["checkpoint_arm"], shard["task"])].add(shard["lambda_value"])
        if any(values != set(FULL_LAMBDA_GRID) for values in observed.values()): raise RuntimeError("full centered-lambda grid closure failed")
    closure_sha = sha256_bytes(canonical_json_bytes(records))
    merged = None
    if merge_dir is not None:
        merge_dir.mkdir(parents=True, exist_ok=True)
        merged_path = merge_dir / f"{stage}_rows.parquet"
        if merged_path.exists(): raise FileExistsError(merged_path)
        import pyarrow as pa
        import pyarrow.parquet as pq
        writer = None
        merged_row_count = 0
        buffer: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal writer
            if not buffer:
                return
            table = pa.Table.from_pylist(buffer).select(list(_audit_columns()[1]))
            if writer is None:
                writer = pq.ParquetWriter(
                    merged_path, table.schema, compression="zstd",
                    use_dictionary=False,
                )
            writer.write_table(table, row_group_size=PARQUET_BATCH_ROWS)
            buffer.clear()

        try:
            for shard in shards:
                for row in _iter_capture_parquet_rows(
                    output_root / shard["output_relative"]
                ):
                    buffer.append(row)
                    merged_row_count += 1
                    if len(buffer) >= PARQUET_BATCH_ROWS:
                        flush()
            flush()
        finally:
            if writer is not None:
                writer.close()
        expected_merged_rows = (
            plan["stages"][STAGE_E1_2]["expected_total_rows"]
            if stage == STAGE_E1_2
            else plan["stages"][STAGE_E1_3]["final_108_shard_expected_total_rows"]
            if stage == STAGE_E1_3
            else sum(int(record["row_count"]) for record in records)
        )
        if merged_row_count != expected_merged_rows:
            raise RuntimeError(
                f"merged stage row count differs from frozen budget: {merged_row_count} != {expected_merged_rows}"
            )
        merged = {
            "path": str(merged_path), "sha256": sha256_file(merged_path),
            "bytes": merged_path.stat().st_size, "row_count": merged_row_count,
        }
    marker = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "GO", "stage": stage, "plan_sha256": plan["plan_sha256"], "shard_count": len(records), "closure_sha256": closure_sha, "ordered_shards": records, "deterministic_merge": merged, "e1_pilot_consumed": False}
    if write_completion:
        path = _phase_marker(output_root, stage)
        if path.exists():
            if _read_json(path) != marker: raise RuntimeError("existing completion marker differs")
        else: atomic_write(path, canonical_json_bytes(marker))
    return marker


def finalize_stage(plan_path: Path, output_root: Path, stage: str, output_dir: Path) -> dict[str, Any]:
    """Mechanically merge a completed stage and run the frozen analyzer.

    This command is the only executor entry point that turns capture shards
    into the five formal E1 mechanism artifacts.  It requires a pre-existing
    verified phase marker and never overwrites a prior finalization.
    """

    if stage not in {STAGE_E1_2, STAGE_E1_3}:
        raise ValueError("only complete E1-2/E1-3 stages may be finalized")
    plan = _read_json(plan_path)
    validate_execution_plan(plan, for_execution=True)
    verify_plan_sources(plan, "host")
    closure_marker = _verify_phase_marker(plan, output_root, stage)
    if output_dir.exists():
        manifest = _read_json(output_dir / "stage_artifact_manifest.json")
        if manifest.get("plan_sha256") != plan["plan_sha256"] or manifest.get("stage") != stage or manifest.get("closure_sha256") != closure_marker["closure_sha256"]:
            raise RuntimeError("existing stage finalization has a different identity")
        merged_record = manifest.get("merged_capture", {})
        merged_path = output_dir / str(merged_record.get("relative_path", ""))
        if (
            not merged_path.is_file()
            or sha256_file(merged_path) != merged_record.get("sha256")
            or merged_path.stat().st_size != merged_record.get("bytes")
            or oct(merged_path.stat().st_mode & 0o777) != merged_record.get("mode")
        ):
            raise RuntimeError("finalized merged capture changed")
        import pyarrow.parquet as pq
        expected_stage_rows = (
            plan["stages"][STAGE_E1_2]["expected_total_rows"]
            if stage == STAGE_E1_2
            else plan["stages"][STAGE_E1_3]["final_108_shard_expected_total_rows"]
        )
        if (
            merged_record.get("row_count") != expected_stage_rows
            or pq.ParquetFile(merged_path).metadata.num_rows != expected_stage_rows
        ):
            raise RuntimeError("finalized merged capture row count changed")
        from script.analysis.fpct_e1_mechanism_audit import verify_artifacts
        verified = verify_artifacts(output_dir)
        artifact_records = _analyzer_artifact_records(output_dir)
        if (
            artifact_records != manifest.get("analyzer_artifacts")
            or verified != manifest.get("analyzer_verification")
        ):
            raise RuntimeError("finalized analyzer artifact receipt changed")
        if stage == STAGE_E1_2:
            _write_finalized_e1_2_lock(
                plan, output_root, output_dir, manifest, closure_marker,
            )
        return {**manifest, "resumed_without_reaggregation": True}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        closure = verify_all(plan_path, output_root, stage, merge_dir=temporary, write_completion=False)
        if closure["closure_sha256"] != closure_marker["closure_sha256"]:
            raise RuntimeError("stage closure changed after completion marker")
        merged = Path(closure["deterministic_merge"]["path"])
        from script.analysis.fpct_e1_mechanism_audit import read_input_rows, verify_artifacts, write_artifacts
        write_artifacts(read_input_rows(merged), temporary)
        verified = verify_artifacts(temporary)
        artifact_records = _analyzer_artifact_records(temporary)
        if verified.get("artifact_sha256") != {
            filename: record["sha256"]
            for filename, record in artifact_records.items()
        }:
            raise RuntimeError("analyzer verification/artifact receipt mismatch")
        expected_stage_rows = (
            plan["stages"][STAGE_E1_2]["expected_total_rows"]
            if stage == STAGE_E1_2
            else plan["stages"][STAGE_E1_3]["final_108_shard_expected_total_rows"]
        )
        if closure["deterministic_merge"].get("row_count") != expected_stage_rows:
            raise RuntimeError("finalizer row count differs from frozen stage budget")
        manifest = {
            "schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "GO",
            "stage": stage, "plan_sha256": plan["plan_sha256"],
            "closure_sha256": closure["closure_sha256"],
            "merged_capture": {
                "relative_path": merged.name,
                "sha256": closure["deterministic_merge"]["sha256"],
                "bytes": closure["deterministic_merge"]["bytes"],
                "row_count": closure["deterministic_merge"]["row_count"],
                "mode": oct(merged.stat().st_mode & 0o777),
            },
            "analyzer_verification": verified,
            "analyzer_artifacts": artifact_records,
            "e1_pilot_consumed": False, "confirmatory_consumed": False,
        }
        atomic_write(temporary / "stage_artifact_manifest.json", canonical_json_bytes(manifest))
        os.replace(temporary, output_dir)
        if stage == STAGE_E1_2:
            _write_finalized_e1_2_lock(
                plan, output_root, output_dir, manifest, closure_marker,
            )
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _verify_lock_bundle_manifest(
    plan_path: Path, plan: Mapping[str, Any], manifest_path: Path,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != K8S_LOCK_SCHEMA_VERSION
        or manifest.get("protocol_id") != K8S_LOCK_PROTOCOL_ID
        or manifest.get("status") != "FROZEN_NOT_APPLIED"
        or manifest.get("execution_sha") != plan["runtime_lock"]["execution_sha"]
        or manifest.get("plan_sha256") != plan["plan_sha256"]
        or manifest.get("image_digest") != plan["runtime_lock"]["image_digest"]
        or manifest.get("source_snapshot_tree_sha") != plan["source_snapshot"]["canonical_tree_sha256"]
        or manifest.get("network_accessed") is not False
        or manifest.get("kubectl_invoked") is not False
    ):
        raise RuntimeError("initial K8s lock-bundle identity changed")
    sources = manifest.get("sources", {})
    expected_sources = {
        "plan": (plan_path, sha256_file(plan_path)),
        "instrumentation_gate": (
            resolve_logical_path(plan, plan["instrumentation_gate"]["logical_path"], "host"),
            plan["instrumentation_gate"]["sha256"],
        ),
        "runtime_probe": (
            resolve_logical_path(plan, plan["runtime_lock"]["runtime_provenance"]["logical_path"], "host"),
            plan["runtime_lock"]["runtime_provenance"]["sha256"],
        ),
        "source_snapshot_receipt": (
            resolve_logical_path(plan, plan["source_snapshot"]["git_receipt"]["logical_path"], "host"),
            plan["source_snapshot"]["git_receipt"]["sha256"],
        ),
    }
    for name, (path, digest) in expected_sources.items():
        record = sources.get(name, {})
        if record.get("sha256") != digest or record.get("bytes") != path.stat().st_size:
            raise RuntimeError(f"initial K8s lock-bundle source changed: {name}")
    configmaps = manifest.get("configmaps", {})
    if set(configmaps) != {"plan_gate", "runtime_probe"}:
        raise RuntimeError("initial K8s lock-bundle ConfigMap universe changed")
    for role, record in configmaps.items():
        yaml_path = Path(str(record.get("yaml_path", "")))
        if (
            record.get("immutable") is not True
            or not yaml_path.is_file()
            or sha256_file(yaml_path) != record.get("yaml_sha256")
            or record.get("namespace") != "c2c-research"
        ):
            raise RuntimeError(f"initial K8s lock-bundle YAML changed: {role}")
    return manifest


def _verify_finalized_bundle_manifest(
    plan: Mapping[str, Any], manifest_path: Path, receipt_sha256: str,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    record = manifest.get("configmaps", {}).get("finalized", {})
    yaml_path = Path(str(record.get("yaml_path", "")))
    if (
        manifest.get("schema_version") != K8S_LOCK_SCHEMA_VERSION
        or manifest.get("protocol_id") != K8S_FINALIZED_LOCK_PROTOCOL_ID
        or manifest.get("status") != "FROZEN_FINALIZED_NOT_APPLIED"
        or manifest.get("execution_sha") != plan["runtime_lock"]["execution_sha"]
        or manifest.get("plan_sha256") != plan["plan_sha256"]
        or manifest.get("receipt_sha256") != receipt_sha256
        or set(manifest.get("configmaps", {})) != {"finalized"}
        or record.get("immutable") is not True
        or not yaml_path.is_file()
        or sha256_file(yaml_path) != record.get("yaml_sha256")
        or record.get("namespace") != "c2c-research"
        or manifest.get("network_accessed") is not False
        or manifest.get("kubectl_invoked") is not False
    ):
        raise RuntimeError("finalized K8s lock-bundle identity/YAML changed")
    return manifest


def _expected_mounted_configmaps(
    bundle_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        record["name"]: {
            "immutable": True,
            "keys": {
                key: {"bytes": locked["bytes"], "sha256": locked["sha256"]}
                for key, locked in sorted(record["keys"].items())
            },
        }
        for record in bundle_manifest["configmaps"].values()
    }


def _verify_mounted_bundle_receipt(
    *, bundle_manifest_path: Path, bundle_manifest: Mapping[str, Any],
    receipt_path: Path, finalized: bool,
) -> dict[str, Any]:
    receipt = _read_json(receipt_path)
    expected_protocol = (
        K8S_FINALIZED_LOCK_PROTOCOL_ID
        if finalized else K8S_LOCK_PROTOCOL_ID
    )
    expected_status = (
        "VERIFIED_FINALIZED_MOUNTED_KEY_BYTES"
        if finalized else "VERIFIED_MOUNTED_KEY_BYTES"
    )
    expected_keys = {
        "schema_version", "protocol_id", "status", "execution_sha",
        "plan_sha256", "bundle_manifest_sha256", "configmaps",
        "network_accessed", "kubectl_invoked_by_verifier",
    }
    if finalized:
        expected_keys |= {
            "receipt_sha256", "closure_sha256", "artifact_tree_sha256",
        }
    if (
        set(receipt) != expected_keys
        or receipt.get("schema_version") != K8S_LOCK_SCHEMA_VERSION
        or receipt.get("protocol_id") != expected_protocol
        or receipt.get("status") != expected_status
        or receipt.get("execution_sha") != bundle_manifest.get("execution_sha")
        or receipt.get("plan_sha256") != bundle_manifest.get("plan_sha256")
        or receipt.get("bundle_manifest_sha256") != sha256_file(bundle_manifest_path)
        or receipt.get("configmaps") != _expected_mounted_configmaps(bundle_manifest)
        or receipt.get("network_accessed") is not False
        or receipt.get("kubectl_invoked_by_verifier") is not False
    ):
        raise RuntimeError("ConfigMap mounted-byte verification receipt is absent/invalid")
    if finalized and any(
        receipt.get(name) != bundle_manifest.get(name)
        for name in ("receipt_sha256", "closure_sha256", "artifact_tree_sha256")
    ):
        raise RuntimeError("finalized ConfigMap verification receipt identity changed")
    return {
        "status": expected_status,
        "sha256": sha256_file(receipt_path),
        "bytes": receipt_path.stat().st_size,
        "bundle_manifest_sha256": receipt["bundle_manifest_sha256"],
        "configmaps": receipt["configmaps"],
    }


def render_k8s(
    plan_path: Path, template_path: Path, output_dir: Path, stage: str,
    lock_bundle_manifest_path: Path, lock_bundle_verification_receipt_path: Path,
    finalized_lock_bundle_manifest_path: Path | None = None,
    finalized_lock_bundle_verification_receipt_path: Path | None = None,
) -> dict[str, Any]:
    plan = _read_json(plan_path); validate_execution_plan(plan, for_execution=True)
    bundle = _verify_lock_bundle_manifest(plan_path, plan, lock_bundle_manifest_path)
    mounted_bundle_receipt = _verify_mounted_bundle_receipt(
        bundle_manifest_path=lock_bundle_manifest_path,
        bundle_manifest=bundle,
        receipt_path=lock_bundle_verification_receipt_path,
        finalized=False,
    )
    frozen_template_path = resolve_logical_path(
        plan, plan["k8s_template"]["logical_path"], "host"
    )
    if (
        template_path.resolve() != frozen_template_path.resolve()
        or sha256_file(template_path) != plan["k8s_template"]["sha256"]
        or template_path.stat().st_size != plan["k8s_template"]["bytes"]
    ):
        raise RuntimeError("capture K8s template differs from the plan-frozen source")
    if stage not in {PHASE_E1_2_BASELINES, PHASE_E1_2_FACTORIZED, STAGE_E1_3}:
        raise ValueError("K8s rendering must use an explicit dependency-safe phase")
    if stage == PHASE_E1_2_FACTORIZED:
        _verify_phase_marker(plan, Path(plan["path_roots"]["output"]["host"]), PHASE_E1_2_BASELINES)
    finalized_lock_sha256 = "none"
    finalized_receipt_container_path = "none"
    finalized_configmap_name = "none"
    finalized_mounted_receipt = None
    finalized_output_host = plan["path_roots"]["output"]["host"]
    finalized_output_container = str(
        Path(plan["path_roots"]["output"]["container"]) / "__no-finalized__"
    )
    if stage == STAGE_E1_3:
        capture_root = Path(plan["path_roots"]["output"]["host"])
        finalized = _verify_finalized_e1_2_lock(
            plan, capture_root, deep=True,
        )
        finalized_lock_sha256 = sha256_file(_finalized_lock_path(capture_root))
        if finalized_lock_bundle_manifest_path is None:
            raise RuntimeError("E1-3 render requires the immutable finalized lock bundle")
        if finalized_lock_bundle_verification_receipt_path is None:
            raise RuntimeError("E1-3 render requires finalized mounted-byte verification")
        finalized_bundle = _verify_finalized_bundle_manifest(
            plan, finalized_lock_bundle_manifest_path, finalized_lock_sha256,
        )
        finalized_mounted_receipt = _verify_mounted_bundle_receipt(
            bundle_manifest_path=finalized_lock_bundle_manifest_path,
            bundle_manifest=finalized_bundle,
            receipt_path=finalized_lock_bundle_verification_receipt_path,
            finalized=True,
        )
        finalized_configmap_name = finalized_bundle["configmaps"]["finalized"]["name"]
        finalized_receipt_container_path = "/opt/fpct-e1-finalized-lock/e1_2_finalized_receipt.json"
        finalized_output_host = str(
            resolve_logical_path(plan, finalized["output_dir"], "host")
        )
        _validate_finalized_overlay(plan, Path(finalized_output_host))
        finalized_output_container = str(
            resolve_logical_path(plan, finalized["output_dir"], "container")
        )
    elif (
        finalized_lock_bundle_manifest_path is not None
        or finalized_lock_bundle_verification_receipt_path is not None
    ):
        raise RuntimeError("pre-E1-3 render may not reference a future finalized bundle")
    template = template_path.read_text(encoding="utf-8")
    if not UNRESOLVED.search(template): raise ValueError("K8s source is not a render-only template")
    shards = (
        sorted((r for r in plan["shards"] if r["stage"] == STAGE_E1_3), key=lambda r: r["shard_id"])
        if stage == STAGE_E1_3
        else _selected_shards(plan, stage)
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    gate_container_root = plan["path_roots"]["gate"]["container"]
    runtime_container_root = plan["path_roots"]["runtime"]["container"]
    finalized_args = ""
    finalized_mounts = ""
    finalized_volumes = ""
    if stage == STAGE_E1_3:
        finalized_args = (
            "        - --finalized-lock-sha256\n"
            f"        - {finalized_lock_sha256}\n"
            "        - --finalized-receipt\n"
            f"        - {finalized_receipt_container_path}"
        )
        finalized_mounts = (
            f"        - {{name: finalized-e1-2, mountPath: {finalized_output_container}, readOnly: true}}\n"
            "        - {name: finalized-lock, mountPath: /opt/fpct-e1-finalized-lock, readOnly: true}"
        )
        finalized_volumes = (
            "      - name: finalized-e1-2\n"
            f"        hostPath: {{path: {finalized_output_host}, type: Directory}}\n"
            "      - name: finalized-lock\n"
            f"        configMap: {{name: {finalized_configmap_name}}}"
        )
    replacements_common = {
        "__EXECUTION_SHA__": plan["runtime_lock"]["execution_sha"], "__IMMUTABLE_IMAGE_DIGEST__": plan["runtime_lock"]["image_digest"],
        "__PLAN_CONTAINER_PATH__": str(Path(gate_container_root) / "e1_capture_plan.json"), "__INSTRUMENTATION_GATE_CONTAINER_PATH__": str(Path(gate_container_root) / "instrumentation_gate.json"),
        "__OUTPUT_CONTAINER_ROOT__": plan["path_roots"]["output"]["container"], "__FROZEN_RUNTIME_BACKEND__": plan["runtime_lock"]["backend"]["spec"],
        "__SOURCE_SNAPSHOT_HOST__": plan["path_roots"]["repo"]["host"],
        "__RAW_TOPOLOGY_HOST_ROOT__": plan["path_roots"]["raw_topology"]["host"],
        "__FINALIZED_LOCK_SHA256__": finalized_lock_sha256,
        "__FINALIZED_OUTPUT_HOST_ROOT__": finalized_output_host,
        "__FINALIZED_OUTPUT_CONTAINER_ROOT__": finalized_output_container,
        "__PLAN_GATE_CONFIGMAP_NAME__": bundle["configmaps"]["plan_gate"]["name"],
        "__EXPECTED_PLAN_SHA256__": plan["plan_sha256"],
        "__RUNTIME_CONFIGMAP_NAME__": bundle["configmaps"]["runtime_probe"]["name"],
        "__FINALIZED_CONFIGMAP_NAME__": finalized_configmap_name,
        "__FINALIZED_RECEIPT_CONTAINER_PATH__": finalized_receipt_container_path,
        "__REPO_CONTAINER_ROOT__": plan["path_roots"]["repo"]["container"],
        "__RUNNER_CONTAINER_PATH__": str(Path(plan["path_roots"]["repo"]["container"]) / EXECUTOR_RELATIVE),
        "__E0_HOST_ROOT__": plan["path_roots"]["e0"]["host"],
        "__E0_CONTAINER_ROOT__": plan["path_roots"]["e0"]["container"],
        "__OUTPUT_HOST_ROOT__": plan["path_roots"]["output"]["host"],
        "__OUTPUT_CONTAINER_ROOT_MOUNT__": plan["path_roots"]["output"]["container"],
        "__INPUT_LOCK_HOST_ROOT_RENDER__": plan["path_roots"]["input_lock"]["host"],
        "__INPUT_LOCK_CONTAINER_ROOT__": plan["path_roots"]["input_lock"]["container"],
        "__RAW_TOPOLOGY_CONTAINER_ROOT__": plan["path_roots"]["raw_topology"]["container"],
        "__GATE_CONTAINER_ROOT__": gate_container_root,
        "__RUNTIME_CONTAINER_ROOT__": runtime_container_root,
        "__MODELS_HOST_ROOT__": plan["path_roots"]["models"]["host"],
        "__MODELS_CONTAINER_ROOT__": plan["path_roots"]["models"]["container"],
        "__FINALIZED_ARG_BLOCK__": finalized_args,
        "__FINALIZED_VOLUME_MOUNT_BLOCK__": finalized_mounts,
        "__FINALIZED_VOLUME_BLOCK__": finalized_volumes,
        "__PHASE__": stage,
        "__REQUIRES_PHASE__": (
            "none" if stage == PHASE_E1_2_BASELINES
            else PHASE_E1_2_BASELINES if stage == PHASE_E1_2_FACTORIZED
            else FINALIZED_E1_2
        ),
    }
    for shard in sorted(shards, key=lambda r: r["shard_id"]):
        text = template
        raw_slug = re.sub(r"[^a-z0-9-]", "-", shard["shard_id"].lower()).strip("-")
        execution_prefix = plan["runtime_lock"]["execution_sha"][:8]
        slug_digest = sha256_bytes(
            f"{execution_prefix}:{shard['shard_id']}".encode()
        )[:10]
        slug = f"{execution_prefix}-{raw_slug[:29]}-{slug_digest}"
        values = {**replacements_common, "__SHARD_ID__": shard["shard_id"], "__SHARD_SLUG__": slug, "__LAMBDA_TAG__": shard["lambda_tag"], "__CLAIM_ID__": _expected_claim_id(plan["plan_sha256"], shard["shard_id"])}
        for old, new in values.items(): text = text.replace(old, new)
        if UNRESOLVED.search(text): raise RuntimeError("K8s render left unresolved placeholders")
        path = output_dir / f"{shard['shard_id']}.yaml"; atomic_write(path, text.encode())
        records.append({"shard_id": shard["shard_id"], "path": path.name, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": 1, "stage": stage,
        "requires_phase": (
            None if stage == PHASE_E1_2_BASELINES
            else PHASE_E1_2_BASELINES if stage == PHASE_E1_2_FACTORIZED
            else FINALIZED_E1_2
        ),
        "plan_sha256": plan["plan_sha256"],
        "run_identity": {
            "execution_sha": plan["runtime_lock"]["execution_sha"],
            "execution_prefix": plan["runtime_lock"]["execution_sha"][:8],
            "job_name_includes_execution_prefix": True,
        },
        "finalized_e1_2": {
            "lock_sha256": None if finalized_lock_sha256 == "none" else finalized_lock_sha256,
            "output_host": finalized_output_host,
            "output_container": finalized_output_container,
            "deep_verified_before_render": stage == STAGE_E1_3,
        },
        "mounted_byte_verification": {
            "initial": mounted_bundle_receipt,
            "finalized": finalized_mounted_receipt,
        },
        "resources": {
            "node_name": "4090-48gx2",
            "gpu_per_job": 1,
            "maximum_concurrent_jobs_on_node": 2,
            "scheduling": "remaining jobs stay Pending; no preemption",
        },
        "jobs": records,
    }
    atomic_write(output_dir / "render_manifest.json", canonical_json_bytes(manifest))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    for name in ("repo-root", "e0-root", "output-root", "output", "instrumentation-gate", "backend-source", "runtime-provenance", "input-lock-manifest", "input-lock-sidecar", "raw-topology-manifest", "raw-topology-artifact-root", "source-snapshot-receipt"): prepare.add_argument(f"--{name}", type=Path, required=True)
    prepare.add_argument("--execution-sha", required=True); prepare.add_argument("--image-digest", required=True); prepare.add_argument("--backend", required=True)
    prepare.add_argument("--repo-container", default="/opt/fpct"); prepare.add_argument("--e0-container", default="/fpct-e0"); prepare.add_argument("--output-container", default="/fpct-e1/e0_design_capture")
    prepare.add_argument("--input-lock-container", required=True, help="exact container mount corresponding to the host input-lock parent")
    prepare.add_argument("--raw-topology-container", required=True, help="read-only container mount corresponding to the raw-topology artifact root")
    prepare.add_argument("--models-host-root", required=True)
    prepare.add_argument("--models-container", required=True)
    run = sub.add_parser("run-shard"); run.add_argument("--plan", type=Path, required=True); run.add_argument("--expected-plan-sha256", required=True); run.add_argument("--shard-id", required=True); run.add_argument("--instrumentation-gate", type=Path); run.add_argument("--output-root", type=Path, required=True); run.add_argument("--backend"); run.add_argument("--path-view", choices=("host", "container"), default="host"); run.add_argument("--claim-id", required=True); run.add_argument("--finalized-lock-sha256"); run.add_argument("--finalized-receipt", type=Path)
    verify = sub.add_parser("verify-all"); verify.add_argument("--plan", type=Path, required=True); verify.add_argument("--output-root", type=Path, required=True); verify.add_argument("--stage", choices=(PHASE_E1_2_BASELINES, PHASE_E1_2_FACTORIZED, STAGE_E1_2, STAGE_E1_3), required=True); verify.add_argument("--merge-dir", type=Path); verify.add_argument("--write-completion", action="store_true")
    render = sub.add_parser("render-k8s"); render.add_argument("--plan", type=Path, required=True); render.add_argument("--template", type=Path, required=True); render.add_argument("--output-dir", type=Path, required=True); render.add_argument("--stage", choices=(PHASE_E1_2_BASELINES, PHASE_E1_2_FACTORIZED, STAGE_E1_3), required=True); render.add_argument("--lock-bundle-manifest", type=Path, required=True); render.add_argument("--lock-bundle-verification-receipt", type=Path, required=True); render.add_argument("--finalized-lock-bundle-manifest", type=Path); render.add_argument("--finalized-lock-bundle-verification-receipt", type=Path)
    finalize = sub.add_parser("finalize-stage"); finalize.add_argument("--plan", type=Path, required=True); finalize.add_argument("--output-root", type=Path, required=True); finalize.add_argument("--stage", choices=(STAGE_E1_2, STAGE_E1_3), required=True); finalize.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = build_execution_plan(args.repo_root, args.e0_root, args.output_root, instrumentation_gate_path=args.instrumentation_gate, execution_sha=args.execution_sha, image_digest=args.image_digest, backend_spec=args.backend, backend_source=args.backend_source, runtime_provenance=args.runtime_provenance, input_lock_manifest=args.input_lock_manifest, input_lock_sidecar=args.input_lock_sidecar, raw_topology_manifest=args.raw_topology_manifest, raw_topology_artifact_root=args.raw_topology_artifact_root, source_snapshot_receipt=args.source_snapshot_receipt, repo_container=args.repo_container, e0_container=args.e0_container, output_container=args.output_container, input_lock_container=args.input_lock_container, raw_topology_container=args.raw_topology_container, models_host_root=args.models_host_root, models_container=args.models_container)
        validate_execution_plan(result, require_checkpoints=True, for_execution=True)
        if args.output.exists(): raise FileExistsError(args.output)
        atomic_write(args.output, canonical_json_bytes(result)); result = {"status": "GO", "plan": str(args.output), "plan_sha256": result["plan_sha256"], "endpoint_shards": EXPECTED_ENDPOINT_SHARDS, "sweep_shards": EXPECTED_SWEEP_SHARDS, "model_or_tokenizer_loaded": False, "e1_pilot_consumed": False}
    elif args.command == "run-shard": result = execute_shard(args.plan, args.shard_id, args.instrumentation_gate, args.output_root, args.backend, path_view=args.path_view, expected_plan_sha256=args.expected_plan_sha256, claim_id=args.claim_id, finalized_lock_sha256=None if args.finalized_lock_sha256 in (None, "none") else args.finalized_lock_sha256, finalized_receipt_path=args.finalized_receipt)
    elif args.command == "verify-all": result = verify_all(args.plan, args.output_root, args.stage, args.merge_dir, args.write_completion)
    elif args.command == "render-k8s": result = render_k8s(args.plan, args.template, args.output_dir, args.stage, args.lock_bundle_manifest, args.lock_bundle_verification_receipt, args.finalized_lock_bundle_manifest, args.finalized_lock_bundle_verification_receipt)
    else: result = finalize_stage(args.plan, args.output_root, args.stage, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
