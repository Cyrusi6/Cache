from __future__ import annotations

"""Prospective, label-free diagnostics helpers for FPCT-GPU-R2."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


TASKS = ("ai2-arc", "openbookqa", "mmlu-redux")
CARDINALITIES = (2, 3, 4)
GROUPS_PER_CELL = 2
ALLOWED_SPLITS = frozenset({"fit", "calibration"})
OPERATOR_PANEL_IDS = (
    "OP01_CPOST_NATIVE", "OP02_F_NATIVE", "OP03_F_REP_NATIVE",
    "OP04_F_FORCED", "OP05_F_REP_FORCED", "OP06_F_BYPASS",
    "OP07_M1_CPOST", "OP08_M1_F",
)
PROFILE_IDS = (
    "P0_GPU_NEGATIVE", "P1_ITEM_POSITIVE", "P2_CPOST_OFF", "P3_F_OFF",
    "P4_F_REPLICATED", "P5_F_ON", "P6_DECODE4",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _eligible_rows(path: Path, task: str, cardinality: int) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["task"] != task or row["split"] not in ALLOWED_SPLITS:
                continue
            if row["certified"] != "1" or int(row["sanitized_m"]) != cardinality:
                continue
            yield row


def freeze_panel(source_root: Path, output: Path) -> dict[str, Any]:
    sources: dict[str, dict[str, str]] = {}
    panel: list[dict[str, Any]] = []
    for task in TASKS:
        path = source_root / f"tinyllama__{task}" / "r1_parent_geometry.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        sources[task] = {"path": str(path.resolve()), "sha256": sha256(path)}
        for cardinality in CARDINALITIES:
            candidates = sorted(
                (
                    row["content_group_sha256"], row["sample_key_sha256"],
                    int(row["parent_index"]), row["split"],
                )
                for row in _eligible_rows(path, task, cardinality)
            )
            selected = []
            seen_groups: set[str] = set()
            for group_sha, sample_sha, parent_index, split in candidates:
                if group_sha in seen_groups:
                    continue
                seen_groups.add(group_sha)
                selected.append({
                    "panel_id": f"{task}__m{cardinality}__{len(selected):02d}",
                    "task": task, "cardinality": cardinality, "split": split,
                    "content_group_sha256": group_sha,
                    "sample_key_sha256": sample_sha,
                    "parent_index": parent_index,
                })
                if len(selected) == GROUPS_PER_CELL:
                    break
            if len(selected) != GROUPS_PER_CELL:
                raise RuntimeError(f"insufficient panel support for {task} m={cardinality}")
            panel.extend(selected)
    payload = {
        "schema_version": 1,
        "protocol_id": "fpct_gpu_r2_label_free_panel_v1",
        "status": "FROZEN_BEFORE_PRETRAINED_OUTPUT",
        "selection": {
            "pair": "tinyllama", "splits": sorted(ALLOWED_SPLITS),
            "tasks": list(TASKS), "cardinalities": list(CARDINALITIES),
            "groups_per_cell": GROUPS_PER_CELL,
            "sort": "content_group_sha256_then_sample_sha_then_parent_index",
            "labels_or_correctness_accessed": False,
        },
        "sources": sources,
        "rows": panel,
    }
    encoded_rows = json.dumps(panel, sort_keys=True, separators=(",", ":")).encode()
    payload["panel_rows_sha256"] = hashlib.sha256(encoded_rows).hexdigest()
    atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    panel = sub.add_parser("freeze-panel")
    panel.add_argument("--source-root", type=Path, required=True)
    panel.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze-panel":
        payload = freeze_panel(args.source_root, args.output)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
