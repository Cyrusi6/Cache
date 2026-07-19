#!/usr/bin/env python3
"""Prepare and execute the frozen Phase 2A-2a cache-geometry pilot.

This runner is intentionally evaluation-only.  It refuses incomplete or
byte-mismatched checkpoints, materializes configs from the completed Phase-1
B6 seed-42 evaluations, and never imports a training entry point.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE2A1_SPLIT = (
    REPO_ROOT / "recipe/eval_recipe/phase2a_1/content_group_split_manifest.json"
)
PHASE2A1_SPLIT_SHA256 = (
    "285b5b00cf3598bba075a97b1439b85031ef1cfffdc03b0e7e1775c6338701e0"
)
SOURCE_ROOT = Path("/netdisk/lijunsi/c2c-route1-identifiability")
SOURCE_WORKSPACE = SOURCE_ROOT / "workspace/Cache"
SOURCE_EVAL_ROOT = SOURCE_WORKSPACE / "local/tmp/route1_identifiability_suite/eval"
DEFAULT_OUTPUT_ROOT = Path("local/tmp/phase2a_2a_cache_geometry")
DEFAULT_RESULTS_ROOT = Path(
    "/netdisk/lijunsi/c2c-phase2a2-cache-geometry/results"
)
DATASETS = ("ai2-arc", "openbookqa", "mmlu-redux")
GPU_LAYOUT = {"ai2-arc": [0], "openbookqa": [1], "mmlu-redux": [0, 1]}
PAIR_ORDER = ("tinyllama", "qwen25_0p5b", "llama32_1b")
MODEL_PATHS = {
    "receiver": SOURCE_ROOT / "models/Qwen3-0.6B",
    "tinyllama": SOURCE_ROOT / "models/TinyLlama-1.1B-Chat-v1.0",
    "qwen25_0p5b": SOURCE_ROOT / "models/Qwen2.5-0.5B-Instruct",
    "llama32_1b": SOURCE_ROOT / "models/Llama-3.2-1B-Instruct",
}
CHECKPOINTS = {
    "tinyllama": SOURCE_ROOT / "checkpoints/b6_seed42/final",
    "qwen25_0p5b": SOURCE_WORKSPACE
    / "local/checkpoints/route1_identifiability/rev_9b06d173eada/"
    "qwen25_0p5b/b6/seed_42/final",
    "llama32_1b": SOURCE_WORKSPACE
    / "local/checkpoints/route1_identifiability/rev_9b06d173eada/"
    "llama32_1b/b6/seed_42/final",
}
CHECKPOINT_SHA256 = {
    "tinyllama": "a66bd9c0b2682dc204ff0efe9a8c0a68c78fe4fa537223e18ecbba396f1c1404",
    "qwen25_0p5b": "cb71b94299bd8ce5134f64a985f34f0f295db9adf4b77f85d9f05b7e66963471",
    "llama32_1b": "ca789cc72884de477c5f02349a156c25774f0095f1d3a0f544bfba9929547cc5",
}
REFERENCE_PREDICTIONS = {
    ("tinyllama", "ai2-arc"): (
        SOURCE_WORKSPACE
        / "local/final_results/route1_identifiability/rev_9b06d173eada/"
        "tinyllama/b6/seed_42/ai2-arc/"
        "Rosetta_ai2-arc_generate_20260717_123650_cot.csv",
        "e5ee920a4189d650b43c1c10a8d09528bec071a7ee24e1c315d2b83eb28a87b1",
    ),
    ("tinyllama", "openbookqa"): (
        SOURCE_WORKSPACE
        / "local/final_results/route1_identifiability/rev_9b06d173eada/"
        "tinyllama/b6/seed_42/openbookqa/"
        "Rosetta_openbookqa_generate_20260717_123212_cot.csv",
        "13d4c074a2f0c7b018527fbaa8bdd4e3a0f007d4c90836b8e018df59c57b0106",
    ),
    ("tinyllama", "mmlu-redux"): (
        SOURCE_WORKSPACE
        / "local/final_results/route1_identifiability/rev_9b06d173eada/"
        "tinyllama/b6/seed_42/mmlu-redux/"
        "Rosetta_mmlu-redux_generate_20260717_125041_cot.csv",
        "12669c3e937996e41b3c854a8a22d72bb96c21a49b402663c6967a2ca1b68240",
    ),
    ("qwen25_0p5b", "ai2-arc"): (
        SOURCE_WORKSPACE
        / "local/final_results/route1_identifiability/rev_9b06d173eada/"
        "qwen25_0p5b/b6/seed_42/ai2-arc/"
        "Rosetta_ai2-arc_generate_20260717_190810_cot.csv",
        "36b9d740d9e0add8bc2694432df957289f65a166ad55561dd4f746627a85a914",
    ),
    ("qwen25_0p5b", "openbookqa"): (
        SOURCE_WORKSPACE
        / "local/final_results/route1_identifiability/rev_9b06d173eada/"
        "qwen25_0p5b/b6/seed_42/openbookqa/"
        "Rosetta_openbookqa_generate_20260717_190138_cot.csv",
        "255be908205a000183bf50d6d5292c0e2b74ddca26c091ac3ba24eb3bb1dbf68",
    ),
    ("qwen25_0p5b", "mmlu-redux"): (
        SOURCE_WORKSPACE
        / "local/final_results/route1_identifiability/rev_9b06d173eada/"
        "qwen25_0p5b/b6/seed_42/mmlu-redux/"
        "Rosetta_mmlu-redux_generate_20260717_192843_cot.csv",
        "735ffd11b5046782bb1069075b9cfb27ca6ae968d47f59a9b7c4c5502b3e3a2e",
    ),
    ("llama32_1b", "ai2-arc"): (
        SOURCE_WORKSPACE
        / "local/final_results/route1_identifiability/rev_9b06d173eada/"
        "llama32_1b/b6/seed_42/ai2-arc/"
        "Rosetta_ai2-arc_generate_20260717_162247_cot.csv",
        "662ac29719856797a1ebd759564d7734680649325579a7bafe1705b8e402e9de",
    ),
    ("llama32_1b", "openbookqa"): (
        SOURCE_WORKSPACE
        / "local/final_results/route1_identifiability/rev_9b06d173eada/"
        "llama32_1b/b6/seed_42/openbookqa/"
        "Rosetta_openbookqa_generate_20260717_161658_cot.csv",
        "28b2ae3626aa2480c4bed8ee8a241debb0707d314e28798900472527bf7b6823",
    ),
    ("llama32_1b", "mmlu-redux"): (
        SOURCE_WORKSPACE
        / "local/final_results/route1_identifiability/rev_9b06d173eada/"
        "llama32_1b/b6/seed_42/mmlu-redux/"
        "Rosetta_mmlu-redux_generate_20260717_164148_cot.csv",
        "fbbf88c55493adffa59a06704675191c8cae1aa2ab9c887536c8d56622eb62f9",
    ),
}
RECEIVER_PREDICTIONS = {
    "ai2-arc": (
        SOURCE_WORKSPACE
        / "local/final_results/route1_identifiability/rev_9b06d173eada/"
        "receiver/b0/seed_42/ai2-arc/"
        "Qwen3-0.6B_ai2-arc_generate_20260717_125217_cot.csv",
        "94147453992282217327bb920c4ca6964ba505c874cb5e2bca15503ef24a68f0",
    ),
    "openbookqa": (
        SOURCE_WORKSPACE
        / "local/final_results/route1_identifiability/rev_9b06d173eada/"
        "receiver/b0/seed_42/openbookqa/"
        "Qwen3-0.6B_openbookqa_generate_20260717_124806_cot.csv",
        "21bda044dbe1074ec281376bfb2ce1ad5ba27606bedeebd0d4528ec38d2d2893",
    ),
    "mmlu-redux": (
        SOURCE_WORKSPACE
        / "local/final_results/route1_identifiability/rev_9b06d173eada/"
        "receiver/b0/seed_42/mmlu-redux/"
        "Qwen3-0.6B_mmlu-redux_generate_20260717_130306_cot.csv",
        "2006f1aaaa178e0877f2389bf36efcdf8192b4bf45c188a8f6e579c33af65003",
    ),
}
EXACT_COLUMNS = (
    "pred",
    "is_correct",
    "cot_pred",
    "cot_output",
    "cot_gen_length",
)


class PilotError(RuntimeError):
    """Fail-closed pilot contract violation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    """Canonical Phase-1 checkpoint SHA: sorted top-level filename NUL bytes."""
    if not path.is_dir():
        raise PilotError(f"missing checkpoint directory: {path}")
    digest = hashlib.sha256()
    files = sorted(
        (item for item in path.iterdir() if item.is_file()),
        key=lambda item: item.name,
    )
    if not files:
        raise PilotError(f"empty checkpoint directory: {path}")
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
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fit_members() -> tuple[set[tuple[str, str, str]], dict[str, int], str]:
    if _sha256(PHASE2A1_SPLIT) != PHASE2A1_SPLIT_SHA256:
        raise PilotError("Phase 2A frozen split manifest SHA mismatch")
    value = json.loads(PHASE2A1_SPLIT.read_text(encoding="utf-8"))
    members: set[tuple[str, str, str]] = set()
    counts = {dataset: 0 for dataset in DATASETS}
    content_hashes = []
    for group in value.get("groups", []):
        if group.get("split") != "fit":
            continue
        content_hashes.append(str(group["content_hash"]))
        for member in group.get("members", []):
            key = (
                str(member["task"]),
                str(member["subject"]),
                str(member["question_id"]),
            )
            if key in members:
                raise PilotError(f"duplicate frozen fit member: {key}")
            members.add(key)
            counts[key[0]] += 1
    digest = hashlib.sha256("\n".join(sorted(content_hashes)).encode()).hexdigest()
    expected = {"ai2-arc": 351, "openbookqa": 158, "mmlu-redux": 1658}
    if counts != expected or len(members) != 2167:
        raise PilotError(f"unexpected frozen fit counts: {counts}, rows={len(members)}")
    return members, counts, digest


def _fit_member_hashes() -> dict[tuple[str, str, str], str]:
    value = json.loads(PHASE2A1_SPLIT.read_text(encoding="utf-8"))
    output: dict[tuple[str, str, str], str] = {}
    for group in value.get("groups", []):
        if group.get("split") != "fit":
            continue
        content_hash = str(group["content_hash"])
        for member in group.get("members", []):
            key = (
                str(member["task"]),
                str(member["subject"]),
                str(member["question_id"]),
            )
            if key in output:
                raise PilotError(f"duplicate frozen fit member: {key}")
            output[key] = content_hash
    if len(output) != 2167:
        raise PilotError(f"unexpected frozen fit member count: {len(output)}")
    return output


def _source_config(pair: str, dataset: str) -> Path:
    return SOURCE_EVAL_ROOT / f"{pair}__b6__seed_42" / f"{dataset}.yaml"


def _validate_assets() -> dict[str, Any]:
    _members, fit_counts, fit_digest = _fit_members()
    assets: dict[str, Any] = {
        "split": {
            "path": str(PHASE2A1_SPLIT),
            "sha256": PHASE2A1_SPLIT_SHA256,
            "allowed_split": "fit",
            "row_counts": fit_counts,
            "content_hash_digest": fit_digest,
        },
        "receiver_predictions": {},
        "pairs": {},
    }
    for dataset, (path, expected_sha) in RECEIVER_PREDICTIONS.items():
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            raise PilotError(f"receiver prediction SHA mismatch: {path}")
        assets["receiver_predictions"][dataset] = {
            "path": str(path), "sha256": actual_sha
        }
    for pair in PAIR_ORDER:
        checkpoint_sha = _directory_sha256(CHECKPOINTS[pair])
        if checkpoint_sha != CHECKPOINT_SHA256[pair]:
            raise PilotError(
                f"checkpoint byte SHA mismatch for {pair}: {checkpoint_sha}"
            )
        pair_assets: dict[str, Any] = {
            "seed": 42,
            "checkpoint": str(CHECKPOINTS[pair]),
            "checkpoint_sha256": checkpoint_sha,
            "teacher_model": str(MODEL_PATHS[pair]),
            "source_configs": {},
            "reference_predictions": {},
        }
        for dataset in DATASETS:
            source = _source_config(pair, dataset)
            reference, expected_sha = REFERENCE_PREDICTIONS[(pair, dataset)]
            if not source.is_file() or not reference.is_file():
                raise PilotError(f"missing source artifact for {pair}/{dataset}")
            actual_sha = _sha256(reference)
            if actual_sha != expected_sha:
                raise PilotError(f"reference prediction SHA mismatch: {reference}")
            pair_assets["source_configs"][dataset] = {
                "path": str(source), "sha256": _sha256(source)
            }
            pair_assets["reference_predictions"][dataset] = {
                "path": str(reference), "sha256": actual_sha
            }
        assets["pairs"][pair] = pair_assets
    return assets


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def prepare(
    *, output_root: Path, results_root: Path, code_commit: str,
    workspace_root: Path,
) -> dict[str, Any]:
    if len(code_commit) != 40 or any(ch not in "0123456789abcdef" for ch in code_commit):
        raise PilotError("--code-commit must be a lowercase full Git SHA")
    assets = _validate_assets()
    assets["split"]["path"] = str(
        workspace_root
        / "recipe/eval_recipe/phase2a_1/content_group_split_manifest.json"
    )
    output_root = output_root if output_root.is_absolute() else REPO_ROOT / output_root
    results_root = results_root.resolve()
    configs_root = output_root / "eval_configs"
    runs = []
    local_config_paths: dict[tuple[str, str], Path] = {}
    for pair in PAIR_ORDER:
        for dataset in DATASETS:
            source = Path(assets["pairs"][pair]["source_configs"][dataset]["path"])
            config = yaml.safe_load(source.read_text(encoding="utf-8"))
            rosetta = config["model"]["rosetta_config"]
            rosetta["base_model"] = str(MODEL_PATHS["receiver"])
            rosetta["teacher_model"] = str(MODEL_PATHS[pair])
            rosetta["checkpoints_dir"] = str(CHECKPOINTS[pair])
            eval_config = config["eval"]
            eval_config["gpu_ids"] = list(GPU_LAYOUT[dataset])
            eval_config["gate_diagnostics"] = False
            eval_config["content_group_filter"] = {
                "manifest": str(
                    workspace_root
                    / "recipe/eval_recipe/phase2a_1/content_group_split_manifest.json"
                ),
                "manifest_sha256": PHASE2A1_SPLIT_SHA256,
                "split": "fit",
                "expected_rows": assets["split"]["row_counts"][dataset],
            }
            geometry_dir = results_root / pair / dataset / "geometry"
            eval_config["cache_geometry_instrumentation"] = {
                "enabled": True,
                "mode": "compact",
                "role": "geometry_on",
                "pair": pair,
                "seed": 42,
                "output_dir": str(geometry_dir),
                "include_per_layer_diagnostics": True,
                "primary_features_from_per_layer": False,
            }
            result_dir = results_root / pair / dataset / "instrumented"
            config["output"]["output_dir"] = str(result_dir)
            target = configs_root / pair / f"{dataset}.yaml"
            _write_yaml(target, config)
            local_config_paths[(pair, dataset)] = target
            pod_config = workspace_root / target.relative_to(REPO_ROOT)
            reference = assets["pairs"][pair]["reference_predictions"][dataset]
            runs.append(
                {
                    "id": f"{pair}__seed42__{dataset}__instrumented",
                    "pair": pair,
                    "seed": 42,
                    "dataset": dataset,
                    "kind": "instrumented",
                    "config": str(pod_config),
                    "config_sha256": _sha256(target),
                    "output_dir": str(result_dir),
                    "geometry_output_dir": str(geometry_dir),
                    "reference_prediction": reference,
                    "checkpoint_sha256": CHECKPOINT_SHA256[pair],
                    "training_forbidden": True,
                }
            )

    # One matched no-instrumentation run estimates debug collection overhead.
    source_run = next(
        run for run in runs
        if run["pair"] == "tinyllama" and run["dataset"] == "ai2-arc"
    )
    control_config = yaml.safe_load(
        local_config_paths[("tinyllama", "ai2-arc")].read_text(encoding="utf-8")
    )
    # Keep the original one-GPU virtual subject name (SPLIT_0_OF_1) so the
    # control has the exact frozen sample identity as the instrumented run.
    control_config["eval"]["gpu_ids"] = [0]
    control_config["eval"]["cache_geometry_instrumentation"] = {
        "enabled": False,
        "mode": "compact",
        "role": "geometry_off",
        "pair": "tinyllama",
        "seed": 42,
    }
    control_dir = results_root / "overhead_control" / "tinyllama" / "ai2-arc"
    control_config["output"]["output_dir"] = str(control_dir)
    control_target = configs_root / "overhead_control" / "tinyllama_ai2-arc.yaml"
    _write_yaml(control_target, control_config)
    pod_control_target = workspace_root / control_target.relative_to(REPO_ROOT)
    runs.append(
        {
            "id": "tinyllama__seed42__ai2-arc__instrumentation_off",
            "pair": "tinyllama",
            "seed": 42,
            "dataset": "ai2-arc",
            "kind": "overhead_control",
            "config": str(pod_control_target),
            "config_sha256": _sha256(control_target),
            "output_dir": str(control_dir),
            "reference_prediction": assets["pairs"]["tinyllama"]
            ["reference_predictions"]["ai2-arc"],
            "checkpoint_sha256": CHECKPOINT_SHA256["tinyllama"],
            "training_forbidden": True,
        }
    )
    manifest = {
        "schema_version": 1,
        "phase": "Phase 2A-2a",
        "role": "pre_transfer_cache_geometry_pilot_execution_manifest",
        "created_at": _utc_now(),
        "source_commit": "a320777ee3d8e2c5fbf988ad6cd840b560aab28b",
        "code_commit": code_commit,
        "branch": "research/phase2a2-cache-geometry",
        "workspace_root": str(workspace_root),
        "results_root": str(results_root),
        "constraints": {
            "evaluation_only": True,
            "checkpoint_modification_forbidden": True,
            "training_forbidden": True,
            "sealed_phase2a1_test_forbidden": True,
            "allowed_seed": [42],
            "allowed_pairs": list(PAIR_ORDER),
            "allowed_split": "fit",
        },
        "assets": assets,
        "runs": runs,
        "exact_equivalence_columns": list(EXACT_COLUMNS),
    }
    manifest_path = output_root / "execution_manifest.json"
    _atomic_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = _sha256(manifest_path)
    _atomic_json(output_root / "prepare_summary.json", manifest)
    return manifest


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("phase") != "Phase 2A-2a" or value.get("constraints", {}).get(
        "training_forbidden"
    ) is not True:
        raise PilotError("not a frozen Phase 2A-2a evaluation-only manifest")
    return value


def _validate_run(run: Mapping[str, Any]) -> None:
    config = Path(str(run["config"]))
    if _sha256(config) != run["config_sha256"]:
        raise PilotError(f"config changed after freeze: {config}")
    pair = str(run["pair"])
    if _directory_sha256(CHECKPOINTS[pair]) != run["checkpoint_sha256"]:
        raise PilotError(f"checkpoint changed after freeze: {pair}")
    config_value = yaml.safe_load(config.read_text(encoding="utf-8"))
    if Path(config_value["model"]["rosetta_config"]["checkpoints_dir"]) != CHECKPOINTS[pair]:
        raise PilotError(f"unexpected checkpoint path in {config}")


def _run_eval(run: Mapping[str, Any], workspace: Path) -> dict[str, Any]:
    _validate_run(run)
    output_dir = Path(str(run["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    state = output_dir / "phase2a2_run_state.json"
    if state.is_file():
        prior = json.loads(state.read_text(encoding="utf-8"))
        if prior.get("status") == "complete" and prior.get("config_sha256") == run["config_sha256"]:
            return prior
    command = [
        sys.executable,
        str(workspace / "script/evaluation/unified_evaluator.py"),
        "--config",
        str(run["config"]),
    ]
    if any("train" in token.lower() for token in command):
        raise PilotError("training-like command rejected")
    started = time.monotonic()
    started_at = _utc_now()
    result = subprocess.run(command, cwd=workspace, check=False)
    record = {
        "schema_version": 1,
        "run_id": run["id"],
        "status": "complete" if result.returncode == 0 else "failed",
        "return_code": int(result.returncode),
        "config": run["config"],
        "config_sha256": run["config_sha256"],
        "checkpoint_sha256": run["checkpoint_sha256"],
        "started_at": started_at,
        "finished_at": _utc_now(),
        "wall_seconds": time.monotonic() - started,
        "command": command,
    }
    _atomic_json(state, record)
    if result.returncode != 0:
        raise PilotError(f"evaluation failed: {run['id']}")
    return record


def run_pair(*, manifest_path: Path, pair: str) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    if pair not in PAIR_ORDER:
        raise PilotError(f"pair is outside frozen pilot: {pair}")
    workspace = Path(str(manifest["workspace_root"]))
    head = subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if head != manifest["code_commit"]:
        raise PilotError(f"workspace commit mismatch: {head}")
    pair_runs = {
        run["dataset"]: run
        for run in manifest["runs"]
        if run["pair"] == pair and run["kind"] == "instrumented"
    }
    if set(pair_runs) != set(DATASETS):
        raise PilotError(f"incomplete pair manifest: {pair}")
    with ThreadPoolExecutor(max_workers=2) as pool:
        short_results = list(
            pool.map(
                lambda dataset: _run_eval(pair_runs[dataset], workspace),
                ("ai2-arc", "openbookqa"),
            )
        )
    results = short_results + [_run_eval(pair_runs["mmlu-redux"], workspace)]
    if pair == "tinyllama":
        control = next(
            run for run in manifest["runs"]
            if run["kind"] == "overhead_control"
        )
        results.append(_run_eval(control, workspace))
    summary = {
        "schema_version": 1,
        "pair": pair,
        "seed": 42,
        "status": "complete",
        "completed_at": _utc_now(),
        "runs": results,
    }
    state_root = Path(str(manifest["results_root"])) / "k8s_state" / pair
    _atomic_json(state_root / "completed.json", summary)
    return summary


def _prediction_csv(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob("*_cot.csv"))
    if len(candidates) != 1:
        raise PilotError(f"expected one prediction CSV in {output_dir}: {candidates}")
    return candidates[0]


def _read_fit_csv(
    path: Path, *, dataset: str, fit_members: set[tuple[str, str, str]],
) -> dict[tuple[str, str, str], dict[str, str]]:
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (dataset, str(row["subject"]), str(row["question_id"]))
            if key not in fit_members:
                continue
            if key in rows:
                raise PilotError(f"duplicate prediction key in {path}: {key}")
            rows[key] = row
    return rows


def _same_cell(left: str | None, right: str | None) -> bool:
    return ("" if left is None else str(left)) == ("" if right is None else str(right))


def _strict_csv_bool(value: str, *, source: str) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return 1
    if normalized in {"false", "0"}:
        return 0
    raise PilotError(f"invalid correctness value in {source}: {value!r}")


def _correctness_by_fit_key(
    path: Path, *, dataset: str,
    fit_hashes: Mapping[tuple[str, str, str], str],
) -> dict[tuple[str, str, str], int]:
    output: dict[tuple[str, str, str], int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"subject", "question_id", "is_correct"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise PilotError(f"missing correctness fields in {path}")
        for row in reader:
            key = (dataset, str(row["subject"]), str(row["question_id"]))
            if key not in fit_hashes:
                continue
            if key in output:
                raise PilotError(f"duplicate correctness key in {path}: {key}")
            output[key] = _strict_csv_bool(row["is_correct"], source=str(path))
    return output


def build_outcomes(*, manifest_path: Path, output: Path) -> dict[str, Any]:
    """Build the local, text-free outcome table after instrumented eval completes."""
    manifest = _load_manifest(manifest_path)
    fit_hashes = _fit_member_hashes()
    receiver: dict[str, dict[tuple[str, str, str], int]] = {}
    for dataset, record in manifest["assets"]["receiver_predictions"].items():
        path = Path(str(record["path"]))
        if _sha256(path) != record["sha256"]:
            raise PilotError(f"receiver prediction changed: {path}")
        receiver[dataset] = _correctness_by_fit_key(
            path, dataset=dataset, fit_hashes=fit_hashes
        )

    rows: list[dict[str, Any]] = []
    for run in manifest["runs"]:
        if run["kind"] != "instrumented":
            continue
        dataset = str(run["dataset"])
        prediction_path = _prediction_csv(Path(str(run["output_dir"])))
        fused = _correctness_by_fit_key(
            prediction_path, dataset=dataset, fit_hashes=fit_hashes
        )
        expected_keys = {key for key in fit_hashes if key[0] == dataset}
        if set(receiver[dataset]) != expected_keys or set(fused) != expected_keys:
            raise PilotError(
                f"outcome key mismatch for {run['pair']}/{dataset}: "
                f"receiver={len(receiver[dataset])}, fused={len(fused)}, "
                f"expected={len(expected_keys)}"
            )
        for key in sorted(expected_keys):
            rows.append(
                {
                    "pair": run["pair"],
                    "seed": 42,
                    "task": dataset,
                    "subject": key[1],
                    "question_id": key[2],
                    "content_hash": fit_hashes[key],
                    "receiver_correct": receiver[dataset][key],
                    "fused_correct": fused[key],
                }
            )
    expected_rows = 3 * len(fit_hashes)
    if len(rows) != expected_rows:
        raise PilotError(f"unexpected outcome rows: {len(rows)} != {expected_rows}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    fieldnames = [
        "pair", "seed", "task", "subject", "question_id", "content_hash",
        "receiver_correct", "fused_correct",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    result = {
        "schema_version": 1,
        "role": "phase2a2a_text_free_outcomes",
        "rows": len(rows),
        "path": str(output.resolve()),
        "sha256": _sha256(output),
        "created_at": _utc_now(),
    }
    _atomic_json(output.with_suffix(output.suffix + ".manifest.json"), result)
    return result


def verify(*, manifest_path: Path, output: Path) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    fit_members, fit_counts, _digest = _fit_members()
    comparisons = []
    all_exact = True
    for run in manifest["runs"]:
        output_dir = Path(str(run["output_dir"]))
        actual_path = _prediction_csv(output_dir)
        reference_path = Path(str(run["reference_prediction"]["path"]))
        actual = _read_fit_csv(actual_path, dataset=run["dataset"], fit_members=fit_members)
        reference = _read_fit_csv(
            reference_path, dataset=run["dataset"], fit_members=fit_members
        )
        keys_exact = set(actual) == set(reference)
        mismatches = []
        for key in sorted(set(actual) & set(reference)):
            differing = [
                column for column in EXACT_COLUMNS
                if not _same_cell(actual[key].get(column), reference[key].get(column))
            ]
            if differing:
                mismatches.append({"key": list(key), "columns": differing})
                if len(mismatches) >= 20:
                    break
        exact = keys_exact and not mismatches and len(actual) == fit_counts[run["dataset"]]
        all_exact = all_exact and exact
        geometry_files = sorted(
            Path(str(run.get("geometry_output_dir", output_dir))).glob(
                "*cache_geometry*"
            )
        )
        comparisons.append(
            {
                "run_id": run["id"],
                "kind": run["kind"],
                "rows": len(actual),
                "expected_rows": fit_counts[run["dataset"]],
                "keys_exact": keys_exact,
                "exact_columns": list(EXACT_COLUMNS),
                "mismatch_count_capped": len(mismatches),
                "first_mismatches": mismatches,
                "exact": exact,
                "prediction_path": str(actual_path),
                "prediction_sha256": _sha256(actual_path),
                "geometry_artifacts": [
                    {"path": str(path), "sha256": _sha256(path)}
                    for path in geometry_files if path.is_file()
                ],
            }
        )
    result = {
        "schema_version": 1,
        "phase": "Phase 2A-2a",
        "verified_at": _utc_now(),
        "instrumentation_output_exact": all_exact,
        "fit_only": True,
        "comparisons": comparisons,
    }
    _atomic_json(output, result)
    if not all_exact:
        raise PilotError("instrumentation equivalence gate failed")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    prepare_parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    prepare_parser.add_argument("--code-commit", required=True)
    prepare_parser.add_argument("--workspace-root", type=Path, required=True)
    run_parser = subparsers.add_parser("run-pair")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--pair", choices=PAIR_ORDER, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument("--output", type=Path, required=True)
    outcomes_parser = subparsers.add_parser("build-outcomes")
    outcomes_parser.add_argument("--manifest", type=Path, required=True)
    outcomes_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        value = prepare(
            output_root=args.output_root,
            results_root=args.results_root,
            code_commit=args.code_commit,
            workspace_root=args.workspace_root,
        )
    elif args.command == "run-pair":
        value = run_pair(manifest_path=args.manifest, pair=args.pair)
    elif args.command == "verify":
        value = verify(manifest_path=args.manifest, output=args.output)
    elif args.command == "build-outcomes":
        value = build_outcomes(manifest_path=args.manifest, output=args.output)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
