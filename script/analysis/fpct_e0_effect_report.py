from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


SEEDS = (2026072201, 2026072202, 2026072203)
TASKS = ("ai2-arc", "openbookqa", "mmlu-redux")
CELLS = ("Y_CC", "Y_CF", "Y_FC", "Y_FF")


def normalize(value: Any) -> str:
    return " ".join(str(value).strip().split())


def content_hash(question: str, choices: list[str]) -> str:
    padded = [normalize(choices[index]) if index < min(4, len(choices)) else "" for index in range(10)]
    payload = {"question": normalize(question), "choices": padded}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _bool(value: str) -> bool:
    if value.lower() in {"true", "1"}:
        return True
    if value.lower() in {"false", "0"}:
        return False
    raise ValueError(value)


def _cot_file(path: Path) -> Path:
    files = list(path.glob("*_cot.csv"))
    if len(files) != 1:
        raise RuntimeError(f"expected one cot CSV in {path}, found {len(files)}")
    return files[0]


def seed_report(root: Path, dev_manifest: dict[str, Any], seed: int) -> dict[str, Any]:
    active = root / "seeds" / str(seed) / "active"
    attempt = active.resolve(strict=True)
    complete = json.loads((attempt / "seed_complete.json").read_text())
    if complete.get("status") != "COMPLETE":
        raise RuntimeError(f"seed {seed} is incomplete")
    manifest_map = {
        (row["task"], row["evaluation_subject"], int(row["evaluation_question_id"])): row
        for row in dev_manifest["rows"]
    }
    group_rows: list[dict[str, Any]] = []
    task_cells: dict[str, dict[str, float]] = {cell: {} for cell in CELLS}
    artifact_hashes: dict[str, str] = {}
    for cell in CELLS:
        for task in TASKS:
            csv_path = _cot_file(attempt / "eval" / cell / task)
            artifact_hashes[f"{cell}/{task}"] = sha256_file(csv_path)
            observed: dict[str, list[bool]] = {}
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                key = (task, row["subject"], int(row["question_id"]))
                if key not in manifest_map:
                    raise RuntimeError(f"unregistered evaluation row: {key}")
                registered = manifest_map[key]
                choices = [row.get(letter, "") for letter in "ABCD"]
                actual_content = content_hash(row.get("question", ""), choices)
                if actual_content != registered["content_group_sha256"]:
                    raise RuntimeError(f"prompt/content mismatch: {key}")
                observed.setdefault(actual_content, []).append(_bool(row["is_correct"]))
            expected_groups = {
                row["content_group_sha256"] for row in dev_manifest["rows"] if row["task"] == task
            }
            if set(observed) != expected_groups:
                raise RuntimeError(f"missing/extra groups in {seed}/{cell}/{task}")
            group_accuracy = {group: mean(values) for group, values in observed.items()}
            task_cells[cell][task] = mean(group_accuracy.values())
            for group, value in sorted(group_accuracy.items()):
                group_rows.append({"seed": seed, "cell": cell, "task": task, "content_group_sha256": group, "correctness": value})
    Y = {cell: mean(task_cells[cell].values()) for cell in CELLS}
    T = Y["Y_FF"] - Y["Y_CC"]
    D_C = Y["Y_CF"] - Y["Y_CC"]
    D_F = Y["Y_FF"] - Y["Y_FC"]
    O = (D_C + D_F) / 2
    I = D_F - D_C
    task_deltas = {task: task_cells["Y_FF"][task] - task_cells["Y_CC"][task] for task in TASKS}
    mechanism = json.loads((attempt / "mechanism_diagnostics.json").read_text())
    integrity = json.loads((attempt / "matched_integrity.json").read_text())
    return {
        "schema_version": 1,
        "seed": seed,
        "attempt": attempt.name,
        "Y": Y,
        "Y_pp": {key: value * 100 for key, value in Y.items()},
        "task_cells": task_cells,
        "task_cells_pp": {cell: {task: value * 100 for task, value in values.items()} for cell, values in task_cells.items()},
        "estimands": {"T": T, "D_C": D_C, "D_F": D_F, "O": O, "I": I},
        "estimands_pp": {"T": T * 100, "D_C": D_C * 100, "D_F": D_F * 100, "O": O * 100, "I": I * 100},
        "task_T": task_deltas,
        "task_T_pp": {task: value * 100 for task, value in task_deltas.items()},
        "mechanism": mechanism,
        "matched_integrity_status": integrity["status"],
        "artifact_sha256": artifact_hashes,
        "group_rows": group_rows,
    }


def aggregate(root: Path, dev_manifest_path: Path, allow_partial: bool = False) -> dict[str, Any]:
    dev = json.loads(dev_manifest_path.read_text())
    reports = []
    for seed in SEEDS:
        try:
            report = seed_report(root, dev, seed)
            atomic_json(root / "seeds" / str(seed) / "active" / "seed_effect.json", report)
            reports.append(report)
        except (FileNotFoundError, RuntimeError):
            if not allow_partial:
                raise
    if not reports:
        raise RuntimeError("no complete E0 seeds")
    values = {name: [row["estimands_pp"][name] for row in reports] for name in ("T", "D_C", "D_F", "O", "I")}
    task_values = {task: [row["task_T_pp"][task] for row in reports] for task in TASKS}
    mechanism_nonzero = all(row["mechanism"]["nonzero_activation"] for row in reports)
    complete = len(reports) == 3
    positive_t = sum(value > 0 for value in values["T"])
    positive_o = sum(value > 0 for value in values["O"])
    go = (
        complete
        and mean(values["T"]) >= 1.0
        and positive_t >= 2
        and all(mean(task_values[task]) >= -2.0 for task in TASKS)
        and mechanism_nonzero
        and all(row["matched_integrity_status"] == "GO" for row in reports)
    )
    classification = "E0_GO_EXPLORATORY_SIGNAL" if go else (
        "E0_NO_GO_FOR_FURTHER_SPEND" if complete else "E0_ENGINEERING_OR_EXECUTION_BLOCKED"
    )
    payload = {
        "schema_version": 1,
        "protocol_id": "fpct_e0_effect_report_v1",
        "classification": classification,
        "complete_seed_count": len(reports),
        "seeds": reports,
        "summary_pp": {
            name: {"mean": mean(data), "median": median(data), "min": min(data), "max": max(data)}
            for name, data in values.items()
        },
        "task_T_pp": {
            task: {"mean": mean(data), "values": data} for task, data in task_values.items()
        },
        "positive_T_seeds": positive_t,
        "mechanism_signal": "positive" if complete and mean(values["O"]) > 0 and positive_o >= 2 else "unresolved",
        "mechanism_nonzero_all_seeds": mechanism_nonzero,
        "claim_boundary": "exploratory TinyLlama-to-Qwen3 signal at 2048 examples/64 steps; n=3, no significance claim and no confirmatory data",
    }
    atomic_json(root / "e0_effect_report.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    payload = aggregate(args.root.resolve(), args.dev_manifest.resolve(), args.allow_partial)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
