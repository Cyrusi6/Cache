from __future__ import annotations

"""Reduce the frozen single-seed FPCT-MATH-DIRECT four-cell evaluation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any


SEED = 2026082101
TASKS = ("ai2-arc", "openbookqa", "mmlu-redux")
CELLS = ("Y_CC", "Y_CF", "Y_FC", "Y_FF")
MATCHED_FIELDS = (
    "step0_trainable_sha256",
    "trainable_keys_sha256",
    "data_order_sha256",
    "rng_state_before_training_sha256",
    "optimizer_class",
    "optimizer_group_count",
    "scheduler_initial_state_sha256",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(value: Any) -> str:
    return " ".join(str(value).strip().split())


def _content_hash(question: str, choices: list[str]) -> str:
    padded = [
        _normalize(choices[index]) if index < min(4, len(choices)) else ""
        for index in range(10)
    ]
    payload = {"question": _normalize(question), "choices": padded}
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _as_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1"}:
        return True
    if lowered in {"false", "0"}:
        return False
    raise ValueError(f"invalid correctness value: {value!r}")


def _cot_file(path: Path) -> Path:
    files = sorted(path.glob("*_cot.csv"))
    if len(files) != 1:
        raise RuntimeError(f"expected one cot CSV in {path}, found {len(files)}")
    return files[0]


def _matched_integrity(run_root: Path) -> dict[str, Any]:
    records = {
        arm: json.loads(
            (run_root / "seeds" / str(SEED) / arm / "fpct_formal_integrity.json").read_text()
        )
        for arm in ("c_post", "f")
    }
    mismatches = {
        field: {arm: record.get(field) for arm, record in records.items()}
        for field in MATCHED_FIELDS
        if len({record.get(field) for record in records.values()}) != 1
    }
    for arm, record in records.items():
        if record.get("operator") != arm:
            raise RuntimeError(f"formal integrity operator mismatch for {arm}")
        if record.get("optimizer_steps") != 64:
            raise RuntimeError(f"formal optimizer-step mismatch for {arm}")
        if record.get("checkpoint_reload_equal") is not True:
            raise RuntimeError(f"checkpoint reload failed for {arm}")
    if mismatches:
        raise RuntimeError(f"matched-integrity mismatch: {mismatches}")
    return {
        "status": "GO",
        "matched_fields": {field: records["c_post"].get(field) for field in MATCHED_FIELDS},
        "checkpoint_sha256": {
            arm: record["official_checkpoint_sha256"] for arm, record in records.items()
        },
        "formal_integrity_sha256": {
            arm: _sha256_file(
                run_root / "seeds" / str(SEED) / arm / "fpct_formal_integrity.json"
            )
            for arm in records
        },
    }


def reduce_results(run_root: Path, dev_manifest_path: Path) -> dict[str, Any]:
    dev = json.loads(dev_manifest_path.read_text())
    manifest_map = {
        (row["task"], row["evaluation_subject"], int(row["evaluation_question_id"])): row
        for row in dev["rows"]
    }
    expected_counts = {"ai2-arc": 128, "openbookqa": 70, "mmlu-redux": 128}
    if dev.get("group_counts") != expected_counts:
        raise RuntimeError("E0-design group counts differ from the frozen protocol")

    task_cells: dict[str, dict[str, float]] = {cell: {} for cell in CELLS}
    artifacts: dict[str, dict[str, Any]] = {}
    for cell in CELLS:
        for task in TASKS:
            csv_path = _cot_file(
                run_root / "seeds" / str(SEED) / "eval" / cell / task
            )
            observed: dict[str, list[bool]] = {}
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                key = (task, row["subject"], int(row["question_id"]))
                registered = manifest_map.get(key)
                if registered is None:
                    raise RuntimeError(f"unregistered evaluation row: {key}")
                actual_content = _content_hash(
                    row.get("question", ""), [row.get(letter, "") for letter in "ABCD"]
                )
                if actual_content != registered["content_group_sha256"]:
                    raise RuntimeError(f"content mismatch: {key}")
                observed.setdefault(actual_content, []).append(_as_bool(row["is_correct"]))
            expected_groups = {
                row["content_group_sha256"] for row in dev["rows"] if row["task"] == task
            }
            if set(observed) != expected_groups:
                raise RuntimeError(f"missing or extra groups for {cell}/{task}")
            task_cells[cell][task] = mean(mean(values) for values in observed.values())
            artifacts[f"{cell}/{task}"] = {
                "path": str(csv_path.resolve()),
                "sha256": _sha256_file(csv_path),
                "bytes": csv_path.stat().st_size,
                "row_count": len(rows),
                "group_count": len(observed),
            }

    y = {cell: mean(task_cells[cell].values()) for cell in CELLS}
    d_c = y["Y_CF"] - y["Y_CC"]
    d_f = y["Y_FF"] - y["Y_FC"]
    estimands = {
        "T": y["Y_FF"] - y["Y_CC"],
        "D_C": d_c,
        "D_F": d_f,
        "O": (d_c + d_f) / 2.0,
        "I": d_f - d_c,
    }
    return {
        "schema_version": 1,
        "protocol_id": "fpct_math_direct_single_seed_result_v1",
        "classification": (
            "SINGLE_SEED_POSITIVE" if estimands["T"] > 0 and estimands["O"] > 0
            else "SINGLE_SEED_NOT_POSITIVE"
        ),
        "seed": SEED,
        "matched_integrity": _matched_integrity(run_root),
        "Y": y,
        "Y_pp": {key: value * 100.0 for key, value in y.items()},
        "task_cells": task_cells,
        "task_cells_pp": {
            cell: {task: value * 100.0 for task, value in values.items()}
            for cell, values in task_cells.items()
        },
        "estimands": estimands,
        "estimands_pp": {key: value * 100.0 for key, value in estimands.items()},
        "additional_seed_condition": estimands["T"] > 0 and estimands["O"] > 0,
        "artifacts": artifacts,
        "claim_boundary": (
            "Exploratory one-seed TinyLlama-to-Qwen3 result on the already-open "
            "326-group E0-design set; no significance, confirmatory, cross-pair, or "
            "universal claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reduce_results(args.run_root.resolve(), args.dev_manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
