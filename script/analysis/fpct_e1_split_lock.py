#!/usr/bin/env python3
"""Materialize the prospective FPCT-E1 design/pilot split without outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


TASKS = ("ai2-arc", "openbookqa", "mmlu-redux")
LIMITS = {"ai2-arc": 128, "openbookqa": 70, "mmlu-redux": 128}
LEDGER_NAMES = {
    "ai2-arc": "tinyllama__ai2-arc/group_support.csv",
    "openbookqa": "tinyllama__openbookqa/group_support.csv",
    "mmlu-redux": "tinyllama__mmlu-redux/group_support.csv",
}
EXPECTED_LEDGER_SHA256 = {
    "ai2-arc": "3147dcfc437dc22629665a91d6079199eed56fc4e477363995d8b8729a05b39e",
    "openbookqa": "e06ccc5661d770945a16b830b694c07f4414b4752be1c093d7944d00cfb4904f",
    "mmlu-redux": "e785b2ce6647835599ba5462d9cff7c16c87e177bfc3b46442f97f20db3cab9e",
}
EXPECTED_SPLIT_SHA256 = "aa40b696aa91cebb5c0c77774db170d4450a8d6d712087731d9b28cf23557050"
EXPECTED_E0_MANIFEST_SHA256 = "25fe8c4dceeaa1174e1433a02ec86f312d909d7d58c3c9a8c7f2caa9d908216a"
DOMAIN = b"fpct-e1-pilot-v1\0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def rank_sha(task: str, group_sha: str) -> str:
    return hashlib.sha256(DOMAIN + task.encode() + b"\0" + group_sha.encode()).hexdigest()


def distinct_groups(rows: Iterable[dict[str, str]], split: str) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["split"] == split:
            result[row["task"]].add(row["content_group_sha256"])
    return result


def build_manifest(
    split_path: Path,
    e0_manifest_path: Path,
    certified_root: Path,
    base_commit: str,
) -> dict[str, Any]:
    if sha256_file(split_path) != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("frozen split manifest SHA256 mismatch")
    if sha256_file(e0_manifest_path) != EXPECTED_E0_MANIFEST_SHA256:
        raise RuntimeError("E0 development manifest SHA256 mismatch")

    split_rows = read_csv(split_path)
    sample_members: dict[tuple[str, str], list[str]] = defaultdict(list)
    group_split: dict[tuple[str, str], str] = {}
    for row in split_rows:
        key = (row["task"], row["content_group_sha256"])
        if key in group_split and group_split[key] != row["split"]:
            raise RuntimeError(f"content group crosses frozen splits: {key}")
        group_split[key] = row["split"]
        sample_members[key].append(row["sample_key_sha256"])

    e0 = json.loads(e0_manifest_path.read_text())
    if e0.get("confirmatory_model_selection_or_test_accessed") is not False:
        raise RuntimeError("E0 manifest does not preserve the confirmatory seal")
    e0_groups: dict[str, set[str]] = defaultdict(set)
    for row in e0["rows"]:
        task = row["task"]
        group = row["content_group_sha256"]
        if group_split[(task, group)] != "calibration":
            raise RuntimeError("E0-design group is not in frozen calibration")
        e0_groups[task].add(group)
    if {task: len(e0_groups[task]) for task in TASKS} != LIMITS:
        raise RuntimeError("unexpected E0-design counts")

    ledgers: dict[str, list[dict[str, str]]] = {}
    ledger_sha: dict[str, str] = {}
    for task in TASKS:
        path = certified_root / LEDGER_NAMES[task]
        observed = sha256_file(path)
        if observed != EXPECTED_LEDGER_SHA256[task]:
            raise RuntimeError(f"certified ledger SHA256 mismatch for {task}")
        ledger_sha[task] = observed
        ledgers[task] = read_csv(path)

    pilot_groups: dict[str, list[tuple[str, str, int]]] = {}
    for task in TASKS:
        candidates: list[tuple[str, str, int]] = []
        for row in ledgers[task]:
            group = row["content_group_sha256"]
            if (
                row["split"] == "fit"
                and row["has_certified_m2"] == "1"
                and row["member_consistent"] == "1"
                and group not in e0_groups[task]
            ):
                if group_split[(task, group)] != "fit":
                    raise RuntimeError("certified ledger and frozen split disagree")
                members = sorted(set(sample_members[(task, group)]))
                if int(row["group_member_count"]) != len(members):
                    raise RuntimeError("group member count mismatch")
                candidates.append((rank_sha(task, group), group, len(members)))
        candidates.sort(key=lambda value: (value[0], value[1]))
        if len(candidates) < LIMITS[task]:
            raise RuntimeError(f"insufficient fit-only groups for {task}")
        pilot_groups[task] = candidates[: LIMITS[task]]

    output_rows: list[dict[str, Any]] = []
    for task in TASKS:
        for group in sorted(e0_groups[task]):
            output_rows.append(
                {
                    "role": "e0_design",
                    "task": task,
                    "source_partition": "calibration",
                    "content_group_sha256": group,
                    "selection_rank_sha256": None,
                    "group_member_count": len(set(sample_members[(task, group)])),
                    "sample_key_sha256": sorted(set(sample_members[(task, group)])),
                }
            )
        for rank, group, member_count in pilot_groups[task]:
            output_rows.append(
                {
                    "role": "e1_pilot",
                    "task": task,
                    "source_partition": "fit",
                    "content_group_sha256": group,
                    "selection_rank_sha256": rank,
                    "group_member_count": member_count,
                    "sample_key_sha256": sorted(set(sample_members[(task, group)])),
                }
            )

    pilot_sets = {task: {row[1] for row in pilot_groups[task]} for task in TASKS}
    model_selection = distinct_groups(split_rows, "model-selection")
    test = distinct_groups(split_rows, "test")
    for task in TASKS:
        if pilot_sets[task] & e0_groups[task]:
            raise RuntimeError("E1-pilot overlaps E0-design")
        if pilot_sets[task] & model_selection[task]:
            raise RuntimeError("E1-pilot overlaps model-selection")
        if pilot_sets[task] & test[task]:
            raise RuntimeError("E1-pilot overlaps test")

    return {
        "schema_version": 1,
        "protocol_id": "fpct_e1_data_split_v1",
        "base_commit": base_commit,
        "human_approval": {
            "approval_id": "APPROVED_PROSPECTIVE_AMENDMENT",
            "source": "support-fit-only certified groups",
            "counts": LIMITS,
            "selection": "domain-separated SHA256 ordering",
            "role": "exploratory mechanism pilot",
            "confirmatory_eligibility": False,
        },
        "selection_contract": {
            "domain_hex": DOMAIN.hex(),
            "rank_formula": "SHA256(domain || utf8(task) || NUL || utf8(content_group_sha256))",
            "ordering": ["selection_rank_sha256", "content_group_sha256"],
            "eligibility": [
                "split == fit",
                "has_certified_m2 == 1",
                "member_consistent == 1",
                "content_group_sha256 not in E0-design",
            ],
        },
        "source_sha256": {
            "content_group_split_manifest.csv": EXPECTED_SPLIT_SHA256,
            "exploratory_dev_manifest.json": EXPECTED_E0_MANIFEST_SHA256,
            "certified_group_support": ledger_sha,
        },
        "counts": {
            "e0_design": LIMITS,
            "e1_pilot": LIMITS,
            "rows_total": len(output_rows),
        },
        "integrity": {
            "e0_design_subset_of_calibration": True,
            "e1_pilot_subset_of_fit": True,
            "e0_design_e1_pilot_intersection": 0,
            "e1_pilot_model_selection_intersection": 0,
            "e1_pilot_test_intersection": 0,
            "labels_answers_predictions_correctness_accessed": False,
            "e1_pilot_forward_or_result_accessed": False,
            "confirmatory_outcomes_accessed": False,
        },
        "confirmatory_seal": {
            "model_selection_distinct_group_counts": {
                task: len(model_selection[task]) for task in TASKS
            },
            "test_distinct_group_counts": {task: len(test[task]) for task in TASKS},
            "meaning": "No raw content, labels, model outputs, predictions, correctness, or outcome aggregates are accessed.",
        },
        "rows": output_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--e0-manifest", type=Path, required=True)
    parser.add_argument("--certified-root", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-sha", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(
        args.split_manifest,
        args.e0_manifest,
        args.certified_root,
        args.base_commit,
    )
    manifest_bytes = canonical_bytes(manifest)
    atomic_write(args.output_manifest, manifest_bytes)
    digest_payload = {
        "schema_version": 1,
        "protocol_id": "fpct_e1_data_split_hash_v1",
        "path": str(args.output_manifest),
        "bytes": len(manifest_bytes),
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "row_count": len(manifest["rows"]),
        "e1_pilot_executed_or_read": False,
        "confirmatory_outcomes_accessed": False,
    }
    atomic_write(args.output_sha, canonical_bytes(digest_payload))
    print(json.dumps(digest_payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
