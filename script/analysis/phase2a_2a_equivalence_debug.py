#!/usr/bin/env python3
"""Run the Llama3.2 Phase 2A-2a Gate-1 equivalence diagnostic.

The diagnostic is deliberately evaluation-only and fail-closed.  It runs the
two frozen fit-only tasks serially on one visible physical GPU, compares only
sample identity and generated outputs, and never reads correctness fields.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path("/netdisk/lijunsi/c2c-route1-identifiability")
SOURCE_WORKSPACE = SOURCE_ROOT / "workspace/Cache"
SOURCE_CONFIG_ROOT = (
    SOURCE_WORKSPACE
    / "local/tmp/route1_identifiability_suite/eval/llama32_1b__b6__seed_42"
)
PHASE2A1_SPLIT = (
    REPO_ROOT / "recipe/eval_recipe/phase2a_1/content_group_split_manifest.json"
)
PHASE2A1_SPLIT_SHA256 = (
    "285b5b00cf3598bba075a97b1439b85031ef1cfffdc03b0e7e1775c6338701e0"
)
MODEL_PATHS = {
    "receiver": SOURCE_ROOT / "models/Qwen3-0.6B",
    "sender": SOURCE_ROOT / "models/Llama-3.2-1B-Instruct",
}
CHECKPOINT = (
    SOURCE_WORKSPACE
    / "local/checkpoints/route1_identifiability/rev_9b06d173eada/"
    "llama32_1b/b6/seed_42/final"
)
CHECKPOINT_SHA256 = (
    "ca789cc72884de477c5f02349a156c25774f0095f1d3a0f544bfba9929547cc5"
)
DATASETS = ("ai2-arc", "openbookqa")
FIT_COUNTS = {"ai2-arc": 351, "openbookqa": 158}
REFERENCE_PREDICTIONS = {
    "ai2-arc": (
        SOURCE_WORKSPACE
        / "local/final_results/route1_identifiability/rev_9b06d173eada/"
        "llama32_1b/b6/seed_42/ai2-arc/"
        "Rosetta_ai2-arc_generate_20260717_162247_cot.csv",
        "662ac29719856797a1ebd759564d7734680649325579a7bafe1705b8e402e9de",
    ),
    "openbookqa": (
        SOURCE_WORKSPACE
        / "local/final_results/route1_identifiability/rev_9b06d173eada/"
        "llama32_1b/b6/seed_42/openbookqa/"
        "Rosetta_openbookqa_generate_20260717_161658_cot.csv",
        "28b2ae3626aa2480c4bed8ee8a241debb0707d314e28798900472527bf7b6823",
    ),
}
CONDITION_MODES = {
    "off_a": "off",
    "off_b": "off",
    "on_a": "capture",
    "on_b": "capture",
    "noop_a": "noop",
    "noop_b": "noop",
}
RUN_ORDER = tuple(CONDITION_MODES)
OUTPUT_COLUMNS = ("pred", "cot_pred", "cot_output", "cot_gen_length")
PRIMARY_OUTPUT_COLUMNS = OUTPUT_COLUMNS


class EquivalenceDebugError(RuntimeError):
    """Fail-closed equivalence diagnostic contract violation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    if not path.is_dir():
        raise EquivalenceDebugError(f"missing checkpoint directory: {path}")
    digest = hashlib.sha256()
    files = sorted(
        (item for item in path.iterdir() if item.is_file()), key=lambda item: item.name
    )
    if not files:
        raise EquivalenceDebugError(f"empty checkpoint directory: {path}")
    for item in files:
        digest.update(item.name.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def _workspace_head(workspace: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        head = (workspace / ".git/HEAD").read_text(encoding="utf-8").strip()
        if len(head) != 40 or any(ch not in "0123456789abcdef" for ch in head):
            raise EquivalenceDebugError(
                "shared workspace is not detached at an exact commit"
            )
        return head


def _fit_members() -> tuple[set[tuple[str, str, str]], str]:
    if _sha256(PHASE2A1_SPLIT) != PHASE2A1_SPLIT_SHA256:
        raise EquivalenceDebugError("Phase 2A frozen split manifest SHA mismatch")
    value = json.loads(PHASE2A1_SPLIT.read_text(encoding="utf-8"))
    members: set[tuple[str, str, str]] = set()
    content_hashes: list[str] = []
    counts = {dataset: 0 for dataset in DATASETS}
    for group in value.get("groups", []):
        if group.get("split") != "fit":
            continue
        group_in_scope = False
        for member in group.get("members", []):
            task = str(member["task"])
            if task not in counts:
                continue
            group_in_scope = True
            key = (task, str(member["subject"]), str(member["question_id"]))
            if key in members:
                raise EquivalenceDebugError(f"duplicate frozen fit member: {key}")
            members.add(key)
            counts[task] += 1
        if group_in_scope:
            content_hashes.append(str(group["content_hash"]))
    if counts != FIT_COUNTS or len(members) != sum(FIT_COUNTS.values()):
        raise EquivalenceDebugError(
            f"unexpected frozen fit scope: counts={counts}, rows={len(members)}"
        )
    fit_digest = hashlib.sha256(
        "\n".join(sorted(content_hashes)).encode("utf-8")
    ).hexdigest()
    return members, fit_digest


def _validate_assets() -> dict[str, Any]:
    _members, fit_digest = _fit_members()
    checkpoint_sha = _directory_sha256(CHECKPOINT)
    if checkpoint_sha != CHECKPOINT_SHA256:
        raise EquivalenceDebugError(
            f"checkpoint SHA mismatch: {checkpoint_sha} != {CHECKPOINT_SHA256}"
        )
    assets: dict[str, Any] = {
        "fit_split": {
            "path": str(PHASE2A1_SPLIT),
            "sha256": PHASE2A1_SPLIT_SHA256,
            "fit_content_hash_digest": fit_digest,
            "row_counts": FIT_COUNTS,
        },
        "checkpoint": {
            "path": str(CHECKPOINT),
            "sha256": checkpoint_sha,
        },
        "models": {name: str(path) for name, path in MODEL_PATHS.items()},
        "source_configs": {},
        "references": {},
    }
    for dataset in DATASETS:
        source_config = SOURCE_CONFIG_ROOT / f"{dataset}.yaml"
        if not source_config.is_file():
            raise EquivalenceDebugError(f"missing source config: {source_config}")
        assets["source_configs"][dataset] = {
            "path": str(source_config),
            "sha256": _sha256(source_config),
        }
        reference, expected_sha = REFERENCE_PREDICTIONS[dataset]
        actual_sha = _sha256(reference)
        if actual_sha != expected_sha:
            raise EquivalenceDebugError(
                f"frozen reference SHA mismatch for {dataset}: {actual_sha}"
            )
        assets["references"][dataset] = {
            "path": str(reference),
            "sha256": actual_sha,
        }
    return assets


def _base_eval_config(
    *, dataset: str, workspace_root: Path, output_dir: Path
) -> dict[str, Any]:
    source = SOURCE_CONFIG_ROOT / f"{dataset}.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    rosetta = config["model"]["rosetta_config"]
    rosetta["base_model"] = str(MODEL_PATHS["receiver"])
    rosetta["teacher_model"] = str(MODEL_PATHS["sender"])
    rosetta["checkpoints_dir"] = str(CHECKPOINT)
    eval_config = config["eval"]
    eval_config["gpu_ids"] = [0]
    # Match the Phase 2A-2a Gate-1 run rather than introducing gate logging.
    eval_config["gate_diagnostics"] = False
    eval_config["content_group_filter"] = {
        "manifest": str(
            workspace_root
            / "recipe/eval_recipe/phase2a_1/content_group_split_manifest.json"
        ),
        "manifest_sha256": PHASE2A1_SPLIT_SHA256,
        "split": "fit",
        "expected_rows": FIT_COUNTS[dataset],
    }
    eval_config.pop("cache_geometry_instrumentation", None)
    config.pop("cache_geometry_instrumentation", None)
    config["output"]["output_dir"] = str(output_dir)
    return config


def _core_config_sha(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["output"]["output_dir"] = "<RUN_OUTPUT>"
    value["eval"].pop("cache_geometry_instrumentation", None)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def prepare(
    *, output_root: Path, results_root: Path, workspace_root: Path,
    code_commit: str,
) -> dict[str, Any]:
    if len(code_commit) != 40 or any(ch not in "0123456789abcdef" for ch in code_commit):
        raise EquivalenceDebugError("--code-commit must be a lowercase full SHA")
    assets = _validate_assets()
    assets["fit_split"]["path"] = str(
        workspace_root
        / "recipe/eval_recipe/phase2a_1/content_group_split_manifest.json"
    )
    output_root = output_root.resolve()
    results_root = results_root.resolve()
    runs = []
    core_shas: dict[str, set[str]] = {dataset: set() for dataset in DATASETS}
    for condition in RUN_ORDER:
        mode = CONDITION_MODES[condition]
        for dataset in DATASETS:
            run_root = results_root / "runs" / condition / dataset
            config = _base_eval_config(
                dataset=dataset,
                workspace_root=workspace_root,
                output_dir=run_root / "predictions",
            )
            if mode in {"capture", "noop"}:
                config["eval"]["cache_geometry_instrumentation"] = {
                    "enabled": True,
                    "capture_mode": mode,
                    "role": "geometry_on" if mode == "capture" else "geometry_noop",
                    "pair": "llama32_1b",
                    "seed": 42,
                    "output_dir": str(run_root / "geometry"),
                    "include_per_layer_diagnostics": mode == "capture",
                    "primary_features_from_per_layer": False,
                }
            target = output_root / "configs" / condition / f"{dataset}.yaml"
            _write_yaml(target, config)
            core_sha = _core_config_sha(config)
            core_shas[dataset].add(core_sha)
            runs.append(
                {
                    "id": f"llama32_1b__seed42__{dataset}__{condition}",
                    "condition": condition,
                    "capture_mode": mode,
                    "dataset": dataset,
                    "seed": 42,
                    "config": str(target),
                    "config_sha256": _sha256(target),
                    "core_config_sha256": core_sha,
                    "output_dir": str(run_root / "predictions"),
                    "geometry_output_dir": (
                        str(run_root / "geometry") if mode != "off" else None
                    ),
                    "checkpoint_sha256": CHECKPOINT_SHA256,
                    "training_forbidden": True,
                }
            )
    if any(len(values) != 1 for values in core_shas.values()):
        raise EquivalenceDebugError(f"condition core configs differ: {core_shas}")
    manifest = {
        "schema_version": 1,
        "phase": "Phase 2A-2a equivalence debug",
        "role": "llama32_gate1_equivalence_diagnostic",
        "created_at": _utc_now(),
        "base_commit": "00db4c7eeffc57a852c67fd1aedad9fd823ca528",
        "code_commit": code_commit,
        "branch": "research/phase2a2-equivalence-debug",
        "workspace_root": str(workspace_root),
        "results_root": str(results_root),
        "constraints": {
            "evaluation_only": True,
            "training_forbidden": True,
            "selector_forbidden": True,
            "geometry_predictability_forbidden": True,
            "mmlu_forbidden": True,
            "sealed_test_forbidden": True,
            "allowed_pair": "llama32_1b",
            "allowed_seed": 42,
            "allowed_tasks": list(DATASETS),
            "allowed_split": "fit",
            "one_visible_physical_gpu": True,
            "serial_execution": True,
        },
        "assets": assets,
        "core_config_sha256": {
            dataset: next(iter(values)) for dataset, values in core_shas.items()
        },
        "run_order": list(RUN_ORDER),
        "conditional_noop": True,
        "comparison_columns": list(OUTPUT_COLUMNS),
        "runs": runs,
    }
    manifest_path = output_root / "execution_manifest.json"
    _atomic_json(manifest_path, manifest)
    summary = {
        **manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    }
    _atomic_json(output_root / "prepare_summary.json", summary)
    return summary


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    constraints = value.get("constraints", {})
    if value.get("phase") != "Phase 2A-2a equivalence debug":
        raise EquivalenceDebugError("unexpected manifest phase")
    required_true = (
        "evaluation_only",
        "training_forbidden",
        "selector_forbidden",
        "geometry_predictability_forbidden",
        "mmlu_forbidden",
        "sealed_test_forbidden",
        "one_visible_physical_gpu",
        "serial_execution",
    )
    if any(constraints.get(name) is not True for name in required_true):
        raise EquivalenceDebugError("manifest does not preserve diagnostic constraints")
    if constraints.get("allowed_tasks") != list(DATASETS):
        raise EquivalenceDebugError("manifest task scope changed")
    return value


def _run_map(manifest: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    output = {
        (str(run["condition"]), str(run["dataset"])): run
        for run in manifest.get("runs", [])
    }
    expected = {(condition, dataset) for condition in RUN_ORDER for dataset in DATASETS}
    if set(output) != expected:
        raise EquivalenceDebugError("manifest run matrix is incomplete")
    return output


def _visible_gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,serial,pci.bus_id,driver_version,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 8:
            raise EquivalenceDebugError(f"unexpected nvidia-smi row: {line}")
        rows.append(
            {
                "index": fields[0],
                "uuid": fields[1],
                "name": fields[2],
                "serial": fields[3],
                "pci_bus_id": fields[4],
                "driver_version": fields[5],
                "memory_total_mib": int(fields[6]),
                "memory_used_mib": int(fields[7]),
            }
        )
    return rows


def _run_text(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            list(command), text=True, capture_output=True, check=False
        )
    except FileNotFoundError as exc:
        return f"unavailable: {exc}"
    return (result.stdout + result.stderr).strip()


def collect_provenance(*, output_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = _visible_gpu_inventory()
    if len(inventory) != 1:
        raise EquivalenceDebugError(
            f"diagnostic requires exactly one visible physical GPU: {inventory}"
        )
    nvidia_q = _run_text(["nvidia-smi", "-q"])
    nvidia_q_path = output_dir / "nvidia_smi_q.txt"
    nvidia_q_path.write_text(nvidia_q + "\n", encoding="utf-8")
    pip_freeze = _run_text([sys.executable, "-m", "pip", "freeze", "--all"])
    pip_path = output_dir / "pip_freeze.txt"
    pip_path.write_text(pip_freeze + "\n", encoding="utf-8")
    runtime_identity_path = Path(sys.prefix).parent / "identity.json"
    runtime_identity = None
    if runtime_identity_path.is_file():
        runtime_identity = json.loads(runtime_identity_path.read_text(encoding="utf-8"))
    provenance = {
        "schema_version": 1,
        "captured_at": _utc_now(),
        "code_commit": manifest["code_commit"],
        "workspace_head": _workspace_head(Path(str(manifest["workspace_root"]))),
        "hostname": platform.node(),
        "pod_name": os.environ.get("HOSTNAME"),
        "kubernetes_node": os.environ.get("C2C_NODE_NAME"),
        "namespace": os.environ.get("C2C_POD_NAMESPACE"),
        "python_executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "gpu_inventory": inventory,
        "runtime_identity_path": (
            str(runtime_identity_path) if runtime_identity_path.is_file() else None
        ),
        "runtime_identity": runtime_identity,
        "container_image": os.environ.get("C2C_RUNTIME_IMAGE"),
        "determinism_environment": {
            name: os.environ.get(name)
            for name in (
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTHONHASHSEED",
                "NVIDIA_TF32_OVERRIDE",
                "TOKENIZERS_PARALLELISM",
                "HF_HUB_OFFLINE",
                "HF_DATASETS_OFFLINE",
                "DATASETS_OFFLINE",
                "TRANSFORMERS_OFFLINE",
            )
        },
        "nvidia_smi_q": {"path": str(nvidia_q_path), "sha256": _sha256(nvidia_q_path)},
        "pip_freeze": {"path": str(pip_path), "sha256": _sha256(pip_path)},
        "checkpoint": manifest["assets"]["checkpoint"],
        "core_config_sha256": manifest["core_config_sha256"],
    }
    if provenance["workspace_head"] != manifest["code_commit"]:
        raise EquivalenceDebugError("workspace commit differs from execution manifest")
    _atomic_json(output_dir / "environment_gpu_provenance.json", provenance)
    return provenance


def _validate_run(run: Mapping[str, Any]) -> None:
    config_path = Path(str(run["config"]))
    if _sha256(config_path) != run["config_sha256"]:
        raise EquivalenceDebugError(f"config changed after freeze: {config_path}")
    if _directory_sha256(CHECKPOINT) != run["checkpoint_sha256"]:
        raise EquivalenceDebugError("checkpoint changed after freeze")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if _core_config_sha(config) != run["core_config_sha256"]:
        raise EquivalenceDebugError(f"core config changed: {config_path}")
    if config["eval"]["dataset"] not in DATASETS:
        raise EquivalenceDebugError("MMLU or another task entered the manifest")


def _prediction_csv(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob("*_cot.csv"))
    if len(candidates) != 1:
        raise EquivalenceDebugError(
            f"expected one prediction CSV in {output_dir}: {candidates}"
        )
    return candidates[0]


def _read_outputs(
    path: Path, *, dataset: str, fit_members: set[tuple[str, str, str]],
) -> dict[tuple[str, str, str], dict[str, str]]:
    output: dict[tuple[str, str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"subject", "question_id", *OUTPUT_COLUMNS}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise EquivalenceDebugError(f"missing output fields in {path}")
        for raw in reader:
            key = (dataset, str(raw["subject"]), str(raw["question_id"]))
            if key not in fit_members:
                continue
            if key in output:
                raise EquivalenceDebugError(f"duplicate sample in {path}: {key}")
            # Intentionally do not access is_correct, true_answer, raw text, or labels.
            output[key] = {name: str(raw.get(name, "")) for name in OUTPUT_COLUMNS}
    expected = FIT_COUNTS[dataset]
    if len(output) != expected:
        raise EquivalenceDebugError(
            f"unexpected fit rows in {path}: {len(output)} != {expected}"
        )
    return output


def _compare_tables(
    left: Mapping[tuple[str, str, str], Mapping[str, str]],
    right: Mapping[tuple[str, str, str], Mapping[str, str]],
    *, left_name: str, right_name: str,
) -> dict[str, Any]:
    left_keys = set(left)
    right_keys = set(right)
    common = sorted(left_keys & right_keys)
    column_mismatches = {name: 0 for name in OUTPUT_COLUMNS}
    primary_mismatch_count = 0
    any_mismatch_count = 0
    mismatch_samples = []
    for key in common:
        differing = [
            name for name in OUTPUT_COLUMNS if left[key][name] != right[key][name]
        ]
        primary_differing = [name for name in PRIMARY_OUTPUT_COLUMNS if name in differing]
        if differing:
            any_mismatch_count += 1
            for name in differing:
                column_mismatches[name] += 1
        if primary_differing:
            primary_mismatch_count += 1
            if len(mismatch_samples) < 50:
                mismatch_samples.append(
                    {
                        "sample_id": ":".join(key),
                        "columns": primary_differing,
                    }
                )
    keys_exact = left_keys == right_keys
    return {
        "left": left_name,
        "right": right_name,
        "left_rows": len(left),
        "right_rows": len(right),
        "keys_exact": keys_exact,
        "missing_from_left": len(right_keys - left_keys),
        "missing_from_right": len(left_keys - right_keys),
        "primary_columns": list(PRIMARY_OUTPUT_COLUMNS),
        "diagnostic_columns": list(OUTPUT_COLUMNS),
        "primary_mismatch_count": primary_mismatch_count,
        "any_mismatch_count": any_mismatch_count,
        "column_mismatch_count": column_mismatches,
        "exact": keys_exact and primary_mismatch_count == 0,
        "first_mismatch_samples": mismatch_samples,
    }


def _combine_comparisons(
    comparisons: Iterable[Mapping[str, Any]], *, left: str, right: str,
) -> dict[str, Any]:
    rows = list(comparisons)
    return {
        "left": left,
        "right": right,
        "tasks": rows,
        "rows": sum(int(row["left_rows"]) for row in rows),
        "primary_mismatch_count": sum(
            int(row["primary_mismatch_count"]) for row in rows
        ),
        "any_mismatch_count": sum(int(row["any_mismatch_count"]) for row in rows),
        "exact": all(bool(row["exact"]) for row in rows),
    }


def _load_condition(
    *, condition: str, run_map: Mapping[tuple[str, str], Mapping[str, Any]],
    fit_members: set[tuple[str, str, str]],
) -> tuple[dict[str, dict[tuple[str, str, str], dict[str, str]]], dict[str, Any]]:
    tables = {}
    files = {}
    for dataset in DATASETS:
        run = run_map[(condition, dataset)]
        path = _prediction_csv(Path(str(run["output_dir"])))
        tables[dataset] = _read_outputs(
            path, dataset=dataset, fit_members=fit_members
        )
        files[dataset] = {"path": str(path), "sha256": _sha256(path)}
    return tables, files


def _load_references(
    manifest: Mapping[str, Any], fit_members: set[tuple[str, str, str]],
) -> dict[str, dict[tuple[str, str, str], dict[str, str]]]:
    output = {}
    for dataset in DATASETS:
        record = manifest["assets"]["references"][dataset]
        path = Path(str(record["path"]))
        if _sha256(path) != record["sha256"]:
            raise EquivalenceDebugError(f"reference changed: {path}")
        output[dataset] = _read_outputs(path, dataset=dataset, fit_members=fit_members)
    return output


def _compare_conditions(
    left_name: str, right_name: str,
    tables: Mapping[str, Mapping[str, Mapping[tuple[str, str, str], Mapping[str, str]]]],
) -> dict[str, Any]:
    per_task = []
    for dataset in DATASETS:
        comparison = _compare_tables(
            tables[left_name][dataset],
            tables[right_name][dataset],
            left_name=f"{left_name}/{dataset}",
            right_name=f"{right_name}/{dataset}",
        )
        comparison["task"] = dataset
        per_task.append(comparison)
    return _combine_comparisons(per_task, left=left_name, right=right_name)


def _write_per_example(
    *, output: Path,
    tables: Mapping[str, Mapping[str, Mapping[tuple[str, str, str], Mapping[str, str]]]],
) -> dict[str, Any]:
    names = list(tables)
    keys = sorted(next(iter(tables.values()))[DATASETS[0]]) + sorted(
        next(iter(tables.values()))[DATASETS[1]]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for key in keys:
            dataset = key[0]
            record: dict[str, Any] = {
                "sample_id": ":".join(key),
                "task": dataset,
                "subject": key[1],
                "question_id": key[2],
                "outputs": {},
            }
            for name in names:
                record["outputs"][name] = dict(tables[name][dataset][key])
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    temporary.replace(output)
    return {"path": str(output), "sha256": _sha256(output), "rows": len(keys)}


def _snapshot_gpu_uuid(expected_uuid: str) -> dict[str, Any]:
    inventory = _visible_gpu_inventory()
    if len(inventory) != 1 or inventory[0]["uuid"] != expected_uuid:
        raise EquivalenceDebugError(
            f"visible physical GPU changed: expected={expected_uuid}, got={inventory}"
        )
    return inventory[0]


def _run_eval(
    *, run: Mapping[str, Any], workspace: Path, expected_gpu_uuid: str,
) -> dict[str, Any]:
    _validate_run(run)
    output_dir = Path(str(run["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir.parent / "run_state.json"
    before = _snapshot_gpu_uuid(expected_gpu_uuid)
    command = [
        sys.executable,
        str(workspace / "script/evaluation/unified_evaluator.py"),
        "--config",
        str(run["config"]),
    ]
    if any("train" in token.lower() or "mmlu" in token.lower() for token in command):
        raise EquivalenceDebugError(f"forbidden command token: {command}")
    log_path = output_dir.parent / "evaluator.log"
    started_at = _utc_now()
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        result = subprocess.run(
            command,
            cwd=workspace,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    after = _snapshot_gpu_uuid(expected_gpu_uuid)
    record = {
        "schema_version": 1,
        "run_id": run["id"],
        "condition": run["condition"],
        "capture_mode": run["capture_mode"],
        "dataset": run["dataset"],
        "status": "complete" if result.returncode == 0 else "failed",
        "return_code": int(result.returncode),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "wall_seconds": time.monotonic() - started,
        "gpu_before": before,
        "gpu_after": after,
        "config": run["config"],
        "config_sha256": run["config_sha256"],
        "core_config_sha256": run["core_config_sha256"],
        "checkpoint_sha256": run["checkpoint_sha256"],
        "log": {"path": str(log_path), "sha256": _sha256(log_path)},
        "command": command,
    }
    if result.returncode == 0:
        prediction = _prediction_csv(output_dir)
        record["prediction"] = {
            "path": str(prediction),
            "sha256": _sha256(prediction),
        }
    _atomic_json(state_path, record)
    if result.returncode != 0:
        raise EquivalenceDebugError(f"evaluation failed: {run['id']}")
    return record


def _determinism_record(
    *, manifest: Mapping[str, Any], provenance: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]], classification: str,
    stop_reason: str, executed_conditions: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "classification": classification,
        "stop_reason": stop_reason,
        "one_pod": True,
        "one_visible_physical_gpu": len(provenance["gpu_inventory"]) == 1,
        "gpu_uuid": provenance["gpu_inventory"][0]["uuid"],
        "serial_execution": True,
        "run_order_executed": list(executed_conditions),
        "same_checkpoint_sha256": manifest["assets"]["checkpoint"]["sha256"],
        "same_core_config_by_task": manifest["core_config_sha256"],
        "comparison_columns": list(OUTPUT_COLUMNS),
        "correctness_read_or_used": False,
        "mmlu_run": False,
        "selector_run": False,
        "geometry_predictability_evaluated": False,
        "comparisons": list(comparisons),
    }


def execute(*, manifest_path: Path, output_root: Path) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    workspace = Path(str(manifest["workspace_root"]))
    if _workspace_head(workspace) != manifest["code_commit"]:
        raise EquivalenceDebugError("workspace commit differs from manifest")
    fit_members, _fit_digest = _fit_members()
    run_map = _run_map(manifest)
    output_root = output_root.resolve()
    provenance = collect_provenance(
        output_dir=output_root / "provenance", manifest=manifest
    )
    expected_gpu_uuid = str(provenance["gpu_inventory"][0]["uuid"])
    references = _load_references(manifest, fit_members)
    tables: dict[
        str, dict[str, dict[tuple[str, str, str], dict[str, str]]]
    ] = {"reference": references}
    files: dict[str, Any] = {"reference": manifest["assets"]["references"]}
    run_states: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    executed: list[str] = []

    def run_condition(condition: str) -> None:
        for dataset in DATASETS:
            run_states.append(
                _run_eval(
                    run=run_map[(condition, dataset)],
                    workspace=workspace,
                    expected_gpu_uuid=expected_gpu_uuid,
                )
            )
        tables[condition], files[condition] = _load_condition(
            condition=condition, run_map=run_map, fit_members=fit_members
        )
        executed.append(condition)

    run_condition("off_a")
    run_condition("off_b")
    reference_off_a = _compare_conditions("reference", "off_a", tables)
    reference_off_b = _compare_conditions("reference", "off_b", tables)
    off_repeat = _compare_conditions("off_a", "off_b", tables)
    comparisons.extend([reference_off_a, reference_off_b, off_repeat])

    if not off_repeat["exact"]:
        classification = "baseline_runtime_numerical_nondeterminism"
        stop_reason = "OFF-A and OFF-B are not exactly reproducible"
    else:
        run_condition("on_a")
        run_condition("on_b")
        on_repeat = _compare_conditions("on_a", "on_b", tables)
        off_on_a = _compare_conditions("off_a", "on_a", tables)
        off_on_b = _compare_conditions("off_a", "on_b", tables)
        comparisons.extend([on_repeat, off_on_a, off_on_b])
        on_differs = not (
            on_repeat["exact"] and off_on_a["exact"] and off_on_b["exact"]
        )
        if on_differs:
            run_condition("noop_a")
            run_condition("noop_b")
            noop_repeat = _compare_conditions("noop_a", "noop_b", tables)
            off_noop_a = _compare_conditions("off_a", "noop_a", tables)
            off_noop_b = _compare_conditions("off_a", "noop_b", tables)
            comparisons.extend([noop_repeat, off_noop_a, off_noop_b])
            if not (
                noop_repeat["exact"]
                and off_noop_a["exact"]
                and off_noop_b["exact"]
            ):
                classification = "instrumentation_control_flow_or_runtime_perturbation"
                stop_reason = "OFF is stable, but NOOP is unstable or differs from OFF"
            else:
                classification = "geometry_reduction_or_synchronization_observer_effect"
                stop_reason = (
                    "OFF and NOOP are stable and exact, while ON is unstable or "
                    "differs from OFF"
                )
        elif not (reference_off_a["exact"] and reference_off_b["exact"]):
            classification = "historical_environment_or_reference_drift"
            stop_reason = (
                "current OFF repeats are stable and current ON is identical, but "
                "the frozen historical reference differs"
            )
        else:
            classification = "off_and_on_exact_no_noop_triggered"
            stop_reason = (
                "OFF and ON repeats are mutually exact; conditional NOOP was not "
                "required by the frozen decision tree"
            )

    per_example = _write_per_example(
        output=output_root / "per_example_outputs.jsonl", tables=tables
    )
    determinism = _determinism_record(
        manifest=manifest,
        provenance=provenance,
        comparisons=comparisons,
        classification=classification,
        stop_reason=stop_reason,
        executed_conditions=executed,
    )
    determinism_path = output_root / "determinism_rerun_checks.json"
    _atomic_json(determinism_path, determinism)
    aggregate = {
        "schema_version": 1,
        "phase": "Phase 2A-2a equivalence debug",
        "completed_at": _utc_now(),
        "classification": classification,
        "stop_reason": stop_reason,
        "decision": "STOP_FOR_REVIEW",
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "code_commit": manifest["code_commit"],
        "base_commit": manifest["base_commit"],
        "scope": {
            "pair": "llama32_1b",
            "seed": 42,
            "tasks": list(DATASETS),
            "fit_rows": sum(FIT_COUNTS.values()),
            "mmlu": False,
            "sealed_test": False,
        },
        "provenance": {
            "path": str(output_root / "provenance/environment_gpu_provenance.json"),
            "sha256": _sha256(
                output_root / "provenance/environment_gpu_provenance.json"
            ),
            "gpu_uuid": expected_gpu_uuid,
            "node": provenance.get("kubernetes_node"),
            "pod": provenance.get("pod_name"),
        },
        "run_states": run_states,
        "prediction_files": files,
        "per_example_outputs": per_example,
        "determinism_checks": {
            "path": str(determinism_path),
            "sha256": _sha256(determinism_path),
        },
        "comparisons": comparisons,
    }
    _atomic_json(output_root / "aggregate.json", aggregate)
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--results-root", type=Path, required=True)
    prepare_parser.add_argument("--workspace-root", type=Path, required=True)
    prepare_parser.add_argument("--code-commit", required=True)
    execute_parser = sub.add_parser("execute")
    execute_parser.add_argument("--manifest", type=Path, required=True)
    execute_parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            output_root=args.output_root,
            results_root=args.results_root,
            workspace_root=args.workspace_root,
            code_commit=args.code_commit,
        )
    else:
        result = execute(manifest_path=args.manifest, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
