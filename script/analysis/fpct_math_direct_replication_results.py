#!/usr/bin/env python3
"""Reduce the frozen three-seed math-direct replication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from statistics import mean, median
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.analysis.fpct_math_direct_results import _as_bool, _content_hash


SEEDS = (2026082101, 2026082102, 2026082103)
TASKS = ("ai2-arc", "openbookqa", "mmlu-redux")
CELLS = ("Y_CC", "Y_CF", "Y_FC", "Y_FF")
EXPECTED_GROUPS = {"ai2-arc": 128, "openbookqa": 70, "mmlu-redux": 128}
MATCHED_FIELDS = (
    "step0_trainable_sha256",
    "trainable_keys_sha256",
    "data_order_sha256",
    "training_examples",
    "rng_state_before_training_sha256",
    "optimizer_class",
    "optimizer_group_count",
    "optimizer_learning_rates",
    "optimizer_weight_decay",
    "scheduler_class",
    "scheduler_initial_state_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cot_file(path: Path) -> Path:
    files = sorted(path.glob("*_cot.csv"))
    if len(files) != 1:
        raise RuntimeError(f"expected one CoT CSV in {path}, found {len(files)}")
    return files[0]


def _matched(seed_root: Path) -> dict[str, Any]:
    records = {
        arm: json.loads((seed_root / arm / "fpct_formal_integrity.json").read_text())
        for arm in ("c_post", "f")
    }
    mismatch = [
        field
        for field in MATCHED_FIELDS
        if json.dumps(records["c_post"].get(field), sort_keys=True)
        != json.dumps(records["f"].get(field), sort_keys=True)
    ]
    if mismatch:
        raise RuntimeError(f"matched-integrity mismatch: {mismatch}")
    for arm, record in records.items():
        if (
            record.get("operator") != arm
            or record.get("optimizer_steps") != 64
            or record.get("checkpoint_reload_equal") is not True
        ):
            raise RuntimeError(f"formal integrity failed for {seed_root.name}/{arm}")
    return {
        "status": "GO",
        "matched_fields": {field: records["c_post"].get(field) for field in MATCHED_FIELDS},
        "checkpoint_sha256": {
            arm: record["official_checkpoint_sha256"] for arm, record in records.items()
        },
        "formal_integrity_sha256": {
            arm: sha256_file(seed_root / arm / "fpct_formal_integrity.json")
            for arm in records
        },
    }


def _estimands(y: dict[str, float]) -> dict[str, float]:
    d_c = y["Y_CF"] - y["Y_CC"]
    d_f = y["Y_FF"] - y["Y_FC"]
    return {
        "T": y["Y_FF"] - y["Y_CC"],
        "D_C": d_c,
        "D_F": d_f,
        "O": (d_c + d_f) / 2.0,
        "I": d_f - d_c,
    }


def accuracy_seed(seed: int, run_root: Path, dev_manifest: dict[str, Any]) -> dict[str, Any]:
    seed_root = run_root / "seeds" / str(seed)
    manifest_map = {
        (row["task"], row["evaluation_subject"], int(row["evaluation_question_id"])): row
        for row in dev_manifest["rows"]
    }
    task_cells: dict[str, dict[str, float]] = {cell: {} for cell in CELLS}
    artifact_sha: dict[str, str] = {}
    for cell in CELLS:
        for task in TASKS:
            csv_path = _cot_file(seed_root / "eval" / cell / task)
            observed: dict[str, list[bool]] = {}
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                key = (task, row["subject"], int(row["question_id"]))
                registered = manifest_map.get(key)
                if registered is None:
                    raise RuntimeError(f"unregistered evaluation row: {key}")
                content = _content_hash(
                    row.get("question", ""), [row.get(letter, "") for letter in "ABCD"]
                )
                if content != registered["content_group_sha256"]:
                    raise RuntimeError(f"evaluation content mismatch: {key}")
                observed.setdefault(content, []).append(_as_bool(row["is_correct"]))
            if len(observed) != EXPECTED_GROUPS[task]:
                raise RuntimeError(f"wrong group count for {seed}/{cell}/{task}")
            task_cells[cell][task] = mean(mean(values) for values in observed.values())
            artifact_sha[f"{cell}/{task}"] = sha256_file(csv_path)
    y = {cell: mean(task_cells[cell].values()) for cell in CELLS}
    estimands = _estimands(y)
    return {
        "seed": seed,
        "Y_pp": {key: value * 100.0 for key, value in y.items()},
        "task_cells_pp": {
            cell: {task: value * 100.0 for task, value in values.items()}
            for cell, values in task_cells.items()
        },
        "estimands_pp": {key: value * 100.0 for key, value in estimands.items()},
        "task_T_pp": {
            task: (task_cells["Y_FF"][task] - task_cells["Y_CC"][task]) * 100.0
            for task in TASKS
        },
        "matched_integrity": _matched(seed_root),
        "evaluation_artifact_sha256": artifact_sha,
    }


def teacher_forced_seed(seed: int, root: Path) -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = {
        cell: {task: [] for task in TASKS} for cell in CELLS
    }
    artifact_sha: dict[str, str] = {}
    for arm in ("c_post", "f"):
        for task in TASKS:
            shard = root / str(seed) / arm / task
            receipt = json.loads((shard / "receipt.json").read_text())
            rows_path = shard / "rows.jsonl"
            if (
                receipt.get("status") != "COMPLETE"
                or sha256_file(rows_path) != receipt.get("rows_sha256")
                or receipt.get("sample_count") != EXPECTED_GROUPS[task]
                or receipt.get("row_count") != 2 * EXPECTED_GROUPS[task]
            ):
                raise RuntimeError(f"teacher-forced receipt mismatch: {seed}/{arm}/{task}")
            groups: dict[tuple[str, str], float] = {}
            with rows_path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    key = (row["cell"], row["content_group_sha256"])
                    if key in groups:
                        raise RuntimeError("duplicate teacher-forced group/cell")
                    groups[key] = float(row["gold_logp_mean"])
            for cell in CELLS:
                selected = [value for (observed_cell, _group), value in groups.items() if observed_cell == cell]
                if selected:
                    if len(selected) != EXPECTED_GROUPS[task]:
                        raise RuntimeError("teacher-forced group count mismatch")
                    values[cell][task].extend(selected)
            artifact_sha[f"{arm}/{task}"] = sha256_file(rows_path)
    task_cells = {
        cell: {task: mean(values[cell][task]) for task in TASKS} for cell in CELLS
    }
    y = {cell: mean(task_cells[cell].values()) for cell in CELLS}
    return {
        "seed": seed,
        "Y_logp": y,
        "task_cells_logp": task_cells,
        "estimands_logp": _estimands(y),
        "artifact_sha256": artifact_sha,
    }


def decide(accuracy: list[dict[str, Any]]) -> dict[str, Any]:
    if [row["seed"] for row in accuracy] != list(SEEDS):
        raise ValueError("replication requires all three frozen seeds in order")
    mean_t = mean(row["estimands_pp"]["T"] for row in accuracy)
    mean_o = mean(row["estimands_pp"]["O"] for row in accuracy)
    positive_t = sum(row["estimands_pp"]["T"] > 0 for row in accuracy)
    positive_o = sum(row["estimands_pp"]["O"] > 0 for row in accuracy)
    task_t = {
        task: mean(row["task_T_pp"][task] for row in accuracy) for task in TASKS
    }
    integrity = all(row["matched_integrity"]["status"] == "GO" for row in accuracy)
    system_gate = mean_t >= 1.0 and positive_t >= 2 and all(value >= -2.0 for value in task_t.values()) and integrity
    mechanism_gate = mean_o > 0.0 and positive_o >= 2 and integrity
    classification = (
        "QUERY_TIME_FPCT_CONTINUE"
        if system_gate and mechanism_gate
        else "FACTORIZED_TRAINING_COLLAPSED_INFERENCE"
        if system_gate
        else "STOP_CURRENT_OPERATOR"
    )
    return {
        "classification": classification,
        "system_gate": system_gate,
        "mechanism_gate": mechanism_gate,
        "mean_T_pp": mean_t,
        "positive_T_seeds": positive_t,
        "mean_O_pp": mean_o,
        "positive_O_seeds": positive_o,
        "task_mean_T_pp": task_t,
        "integrity_all_go": integrity,
    }


def reduce_all(
    run_roots: dict[int, Path],
    teacher_root: Path,
    dev_manifest_path: Path,
    gradient_path: Path,
) -> dict[str, Any]:
    if set(run_roots) != set(SEEDS):
        raise ValueError("exactly three frozen seed roots are required")
    dev = json.loads(dev_manifest_path.read_text())
    if dev.get("group_counts") != EXPECTED_GROUPS:
        raise ValueError("development manifest group counts mismatch")
    accuracy = [accuracy_seed(seed, run_roots[seed], dev) for seed in SEEDS]
    teacher = [teacher_forced_seed(seed, teacher_root) for seed in SEEDS]
    gradient = json.loads(gradient_path.read_text())
    if gradient.get("status") != "GO" or gradient.get("optimizer_step_performed") is not False:
        raise RuntimeError("step-0 gradient diagnostic failed")
    return {
        "schema_version": 1,
        "protocol_id": "fpct_math_direct_replication_v1",
        "decision": decide(accuracy),
        "accuracy_seeds": accuracy,
        "teacher_forced_seeds": teacher,
        "teacher_forced_summary": {
            name: {
                "mean": mean(row["estimands_logp"][name] for row in teacher),
                "median": median(row["estimands_logp"][name] for row in teacher),
                "positive_seeds": sum(row["estimands_logp"][name] > 0 for row in teacher),
            }
            for name in ("T", "D_C", "D_F", "O", "I")
        },
        "step0_gradient": gradient,
        "firewall": {
            "e1_pilot": "SEALED_NOT_READ",
            "model_selection": "SEALED_NOT_READ",
            "test": "SEALED_NOT_READ",
            "confirmatory": "NOT_RUN",
        },
    }


def _seed_root(value: str) -> tuple[int, Path]:
    seed, raw_path = value.split("=", 1)
    return int(seed), Path(raw_path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-run", action="append", required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--gradient", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_roots = dict(_seed_root(value) for value in args.seed_run)
    result = reduce_all(
        run_roots,
        args.teacher_root.resolve(),
        args.dev_manifest.resolve(),
        args.gradient.resolve(),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
