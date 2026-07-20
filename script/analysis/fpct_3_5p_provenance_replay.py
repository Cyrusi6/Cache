from __future__ import annotations

"""Sealed deterministic replay of the immutable FPCT-3.5 forensic."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from fpct_bootstrap import loaded_module, require_active


REPO_ROOT = Path(__file__).resolve().parents[2]
require_active(target=Path(__file__))

TASK_ORDER = ("ai2-arc", "openbookqa", "mmlu-redux")
RESULT_BASE = Path(
    "local/final_results/fpct_factorized_transport/"
    "fpct_3_5p_provenance_replay"
)
OLD_EXECUTION_SHA = "0398d26b63e96263b813730368275ee66e313f66"
OLD_ROOT = REPO_ROOT / (
    "local/final_results/fpct_factorized_transport/"
    f"fpct_3_5_alignment_correctness/rev_{OLD_EXECUTION_SHA}"
)
PROTOCOL = REPO_ROOT / "FPCT_3_5P_PROVENANCE_REPLAY_PROTOCOL.md"
MANIFEST = REPO_ROOT / (
    "recipe/eval_recipe/fpct_3_5p/provenance_replay_manifest.json"
)
DIFF = REPO_ROOT / "recipe/eval_recipe/fpct_3_7r/protocol_diff.json"
CONTEXT_RADIUS = 4


class ReplayError(RuntimeError):
    pass


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReplayError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance((value := row.get(column)), (dict, list, tuple))
                    else value
                    for column in columns
                }
            )
            count += 1
    temp.replace(path)
    return count


def _root(execution_sha: str) -> Path:
    return REPO_ROOT / RESULT_BASE / f"rev_{execution_sha}"


def _configured_old() -> Any:
    old = loaded_module("fpct_3_5_audit")
    old.load_fpct1b_module = lambda: loaded_module("fpct_1b_audit")
    return old


def _tracked() -> dict[str, Path]:
    return {
        "protocol": PROTOCOL,
        "manifest": MANIFEST,
        "protocol_diff": DIFF,
        "analysis": Path(__file__).resolve(),
        "bootstrap": REPO_ROOT / "script/runtime/fpct_bootstrap.py",
        "probe": REPO_ROOT / "script/runtime/fpct_probe_target.py",
        "regular_package_init": REPO_ROOT / "rosetta/__init__.py",
        "hostile_tests": REPO_ROOT / "test/test_fpct_sealed_import.py",
        "old_protocol": REPO_ROOT / "FPCT_3_5_ALIGNMENT_CORRECTNESS_PROTOCOL.md",
        "old_manifest": REPO_ROOT / "recipe/eval_recipe/fpct_3_5/alignment_correctness_manifest.json",
        "old_analysis": REPO_ROOT / "script/analysis/fpct_3_5_alignment_correctness.py",
        "fpct_1b_analysis": REPO_ROOT / "script/analysis/fpct_1b_structural_support_audit.py",
        "fpct_1b_lock": REPO_ROOT / "recipe/eval_recipe/fpct_1b/pre_audit_lock.json",
        "fpct_1b_split": REPO_ROOT / "recipe/eval_recipe/fpct_1b/content_group_split_manifest.csv",
        "aligner": REPO_ROOT / "rosetta/model/aligner.py",
        "prompt_builder": REPO_ROOT / "rosetta/utils/evaluate.py",
    }


def freeze() -> None:
    if _git("status", "--short"):
        raise ReplayError("freeze requires a clean worktree")
    execution_sha = _git("rev-parse", "HEAD")
    if _git("rev-parse", "@{upstream}") != execution_sha:
        raise ReplayError("freeze requires local/upstream identity")
    attestation = require_active(target=Path(__file__))
    lock = {
        "schema_version": 1,
        "stage": "FPCT-3.5P",
        "status": "PRE_DATA_FROZEN",
        "execution_sha": execution_sha,
        "historical_execution_sha": OLD_EXECUTION_SHA,
        "stable_attestation_sha256": attestation[
            "stable_fingerprint_sha256"
        ],
        "sealed_modules": attestation["mandatory_modules"],
        "tracked": {
            key: {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256(path),
            }
            for key, path in _tracked().items()
        },
        "natural_forensic_started": False,
        "gpu_authorized": False,
    }
    root = _root(execution_sha)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "pre_data_lock.json"
    if lock_path.exists() and _read_json(lock_path) != lock:
        raise ReplayError("existing pre-data lock differs")
    if not lock_path.exists():
        _write_json(lock_path, lock)
    _write_json(
        root / "controller_state.json",
        {
            "schema_version": 1,
            "stage": "FPCT-3.5P",
            "execution_sha": execution_sha,
            "state": "PRE_DATA_FROZEN",
            "completed_tasks": [],
            "held_out_test_released": False,
        },
    )
    print(json.dumps({"status": "FROZEN", "execution_sha": execution_sha}))


def _verify_lock(execution_sha: str) -> tuple[Path, dict[str, Any]]:
    if _git("rev-parse", "HEAD") != execution_sha:
        raise ReplayError("HEAD differs from replay execution SHA")
    if _git("rev-parse", "@{upstream}") != execution_sha:
        raise ReplayError("upstream differs from replay execution SHA")
    if _git("status", "--short"):
        raise ReplayError("replay requires a clean worktree")
    root = _root(execution_sha)
    lock = _read_json(root / "pre_data_lock.json")
    if lock.get("execution_sha") != execution_sha:
        raise ReplayError("replay lock execution SHA mismatch")
    current = require_active(target=Path(__file__))
    if current["stable_fingerprint_sha256"] != lock[
        "stable_attestation_sha256"
    ]:
        raise ReplayError("replay stable attestation differs from lock")
    for record in lock["tracked"].values():
        if _sha256(REPO_ROOT / record["path"]) != record["sha256"]:
            raise ReplayError(f"frozen replay input changed: {record['path']}")
    return root, lock


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def _geometry_flags(row: Mapping[str, str]) -> dict[str, int]:
    receiver = tuple(json.loads(row["receiver_relative_interval"]))
    candidates = [tuple(item) for item in json.loads(row["candidate_relative_intervals"])]
    candidate_indices = [int(item) for item in json.loads(row["candidate_indices"])]
    intersections = [
        (max(receiver[0], item[0]), min(receiver[1], item[1]))
        for item in candidates
        if max(receiver[0], item[0]) < min(receiver[1], item[1])
    ]
    ordered_intersections = sorted(intersections)
    coverage_gap = False
    if receiver[1] > receiver[0]:
        cursor = receiver[0]
        for start, end in ordered_intersections:
            if start > cursor:
                coverage_gap = True
            cursor = max(cursor, end)
        coverage_gap = coverage_gap or cursor < receiver[1]
    return {
        "receiver_zero_length": int(receiver[1] <= receiver[0]),
        "receiver_overlap": int(
            row["anomaly_category"] == "duplicate_or_overlap_receiver_offsets"
        ),
        "source_zero_length": int(any(end <= start for start, end in candidates)),
        "source_duplicate": int(len(candidates) != len(set(candidates))),
        "source_overlap": int(
            any(
                _overlap(left, right)
                for index, left in enumerate(candidates)
                for right in candidates[index + 1 :]
            )
        ),
        "coverage_gap": int(coverage_gap),
        "non_monotonic": int(
            candidate_indices
            != [
                item[1]
                for item in sorted(
                    zip(candidates, candidate_indices), key=lambda value: value[0]
                )
            ]
        ),
        "topk_non_exhaustive": int(
            "exhaust" in row["certification_reason"].lower()
        ),
        "truncation_loss": 0,
        "illegal_slot0": int(not candidate_indices or candidate_indices[0] < 0),
    }


def _enrich_geometry(task: str, execution_sha: str) -> None:
    root, lock = _verify_lock(execution_sha)
    directory = root / "forensic" / task
    ledger_path = directory / "qwen3_forensic_ledger.csv"
    rows = _read_csv(ledger_path)
    output_rows = []
    cooccurrence: Counter[tuple[str, str]] = Counter()
    clusters: set[tuple[str, tuple[int, ...]]] = set()
    flag_counts: Counter[str] = Counter()
    for row in rows:
        flags = _geometry_flags(row)
        indices = tuple(int(item) for item in json.loads(row["candidate_indices"]))
        clusters.add((row["sample_key_sha256"], indices))
        for flag, value in flags.items():
            flag_counts[flag] += value
            if value:
                cooccurrence[(row["anomaly_category"], flag)] += 1
        output_rows.append(
            {
                "schema_version": 1,
                "task": task,
                "split": row["split"],
                "sample_key_sha256": row["sample_key_sha256"],
                "content_group_sha256": row["content_group_sha256"],
                "parent_index": int(row["parent_index"]),
                "primary_reason": row["anomaly_category"],
                **flags,
            }
        )
    columns = tuple(output_rows[0]) if output_rows else (
        "schema_version", "task", "split", "sample_key_sha256",
        "content_group_sha256", "parent_index", "primary_reason",
    )
    flags_path = directory / "geometry_flags.csv"
    _write_csv(flags_path, columns, output_rows)
    summary = {
        "schema_version": 1,
        "execution_sha": execution_sha,
        "task": task,
        "stable_attestation_sha256": lock["stable_attestation_sha256"],
        "rows": len(rows),
        "overlap_clusters": len(clusters),
        "flag_counts": dict(sorted(flag_counts.items())),
        "primary_reason_x_secondary_flag": [
            {"primary_reason": key[0], "secondary_flag": key[1], "count": count}
            for key, count in sorted(cooccurrence.items())
        ],
        "artifact": {
            "path": str(flags_path),
            "rows": len(output_rows),
            "sha256": _sha256(flags_path),
            "bytes": flags_path.stat().st_size,
        },
    }
    _write_json(directory / "geometry_summary.json", summary)
    _write_json(
        directory / "sealed_shard_manifest.json",
        {
            "schema_version": 1,
            "stage": "FPCT-3.5P",
            "execution_sha": execution_sha,
            "task": task,
            "stable_attestation_sha256": lock["stable_attestation_sha256"],
            "delegate_manifest_sha256": _sha256(directory / "manifest.json"),
            "geometry_summary_sha256": _sha256(
                directory / "geometry_summary.json"
            ),
        },
    )


def run_task(task: str, execution_sha: str) -> None:
    root, lock = _verify_lock(execution_sha)
    directory = root / "forensic" / task
    sealed = directory / "sealed_shard_manifest.json"
    if sealed.exists():
        existing = _read_json(sealed)
        if (
            existing.get("execution_sha") != execution_sha
            or existing.get("stable_attestation_sha256")
            != lock["stable_attestation_sha256"]
            or _sha256(directory / "manifest.json")
            != existing.get("delegate_manifest_sha256")
            or _sha256(directory / "geometry_summary.json")
            != existing.get("geometry_summary_sha256")
        ):
            raise ReplayError("completed sealed forensic shard changed")
        print(json.dumps({"status": "SEALED_RESUME_SKIP", "task": task}))
        return
    old = _configured_old()
    old.run_forensic_task(task, execution_sha, RESULT_BASE)
    _enrich_geometry(task, execution_sha)


def _normalized_ledger(row: Mapping[str, str]) -> dict[str, Any]:
    excluded = {"sender_directory", "sender_name_or_path"}
    return {key: value for key, value in row.items() if key not in excluded}


def _candidate_atoms(row: Mapping[str, str]) -> tuple[Any, ...]:
    return tuple(
        zip(
            json.loads(row["candidate_indices"]),
            json.loads(row["candidate_token_ids"]),
            map(tuple, json.loads(row["candidate_offsets"])),
            json.loads(row["candidate_weights"]),
        )
    )


def _context_projection(keys: set[tuple[str, int]]) -> dict[tuple[str, int], Any]:
    audit = loaded_module("fpct_1b_audit")
    old = _configured_old()
    aligner_module = loaded_module("aligner")
    lock = _read_json(REPO_ROOT / "recipe/eval_recipe/fpct_1b/pre_audit_lock.json")
    receiver, senders = audit.load_tokenizers(lock)
    qwen3 = senders["qwen3_1p7b"]
    aligner = aligner_module.TokenAligner(
        receiver,
        qwen3,
        strategy="soft_span_overlap_v2",
        soft_alignment_score_mode="uniform",
        soft_alignment_boundary_bonus=0.5,
        soft_alignment_boundary_tolerance=1,
        soft_alignment_min_weight=0.0,
        soft_alignment_confidence_mode="none",
        soft_alignment_reweight_mode="none",
        soft_alignment_candidate_window=0,
        verbose=False,
    )
    wanted_samples = {sample for sample, _parent in keys}
    samples = audit.load_projected_samples(audit.DEFAULT_SHARED_ROOT)
    projection: dict[tuple[str, int], Any] = {}
    for sample in samples:
        if sample.sample_key_sha256 not in wanted_samples:
            continue
        messages = [{"role": "user", "content": audit.prompt_for_sample(sample)}]
        snapshot = old.identity_snapshot(aligner, qwen3, messages)
        for key in sorted(item for item in keys if item[0] == sample.sample_key_sha256):
            parent = key[1]
            start = max(0, parent - CONTEXT_RADIUS)
            end = min(len(snapshot.input_ids), parent + CONTEXT_RADIUS + 1)
            projection[key] = {
                "window_start": start,
                "window_end": end,
                "input_ids": list(snapshot.input_ids[start:end]),
                "offsets": [list(item) for item in snapshot.offsets[start:end]],
            }
    if set(projection) != keys:
        raise ReplayError("fixed context projection did not cover every replay row")
    return projection


def compare(execution_sha: str) -> dict[str, Any]:
    root, lock = _verify_lock(execution_sha)
    old_summary = _read_json(OLD_ROOT / "forensic_summary.json")
    new_summary = _read_json(root / "forensic_summary.json")
    mismatches: list[str] = []
    old_rows: list[dict[str, str]] = []
    new_rows: list[dict[str, str]] = []
    old_comparison: list[dict[str, str]] = []
    new_comparison: list[dict[str, str]] = []
    clusters: set[tuple[str, tuple[int, ...]]] = set()
    groups: set[str] = set()
    cooccurrence: Counter[tuple[str, str]] = Counter()
    for task in TASK_ORDER:
        old_task = OLD_ROOT / "forensic" / task
        new_task = root / "forensic" / task
        task_old = _read_csv(old_task / "qwen3_forensic_ledger.csv")
        task_new = _read_csv(new_task / "qwen3_forensic_ledger.csv")
        old_rows.extend(task_old)
        new_rows.extend(task_new)
        old_comparison.extend(
            _read_csv(old_task / "qwen3_qwen25_comparison_rows.csv")
        )
        new_comparison.extend(
            _read_csv(new_task / "qwen3_qwen25_comparison_rows.csv")
        )
        geometry = _read_json(new_task / "geometry_summary.json")
        for row in geometry["primary_reason_x_secondary_flag"]:
            cooccurrence[(row["primary_reason"], row["secondary_flag"])] += int(
                row["count"]
            )
    old_normalized = [_normalized_ledger(row) for row in old_rows]
    new_normalized = [_normalized_ledger(row) for row in new_rows]
    if old_normalized != new_normalized:
        mismatches.append("ordered normalized forensic ledger")
    if Counter(map(_stable_sha, old_normalized)) != Counter(
        map(_stable_sha, new_normalized)
    ):
        mismatches.append("multiset normalized forensic ledger")
    old_atoms = [_candidate_atoms(row) for row in old_rows]
    new_atoms = [_candidate_atoms(row) for row in new_rows]
    if old_atoms != new_atoms:
        mismatches.append("ordered candidate atom signatures")
    if Counter(map(str, old_atoms)) != Counter(map(str, new_atoms)):
        mismatches.append("multiset candidate atom signatures")
    if old_comparison != new_comparison:
        mismatches.append("Qwen3/Qwen2.5 row projection")
    keys = {
        (row["sample_key_sha256"], int(row["parent_index"]))
        for row in old_rows + new_rows
    }
    contexts = _context_projection(keys)
    old_contexts = [
        contexts[(row["sample_key_sha256"], int(row["parent_index"]))]
        for row in old_rows
    ]
    new_contexts = [
        contexts[(row["sample_key_sha256"], int(row["parent_index"]))]
        for row in new_rows
    ]
    if old_contexts != new_contexts:
        mismatches.append("fixed context token windows")
    for row, context in zip(new_rows, new_contexts):
        center = int(row["parent_index"]) - int(context["window_start"])
        if int(row["receiver_token_id"]) != int(context["input_ids"][center]):
            mismatches.append("receiver token does not match context center")
            break
        indices = tuple(int(item) for item in json.loads(row["candidate_indices"]))
        clusters.add((row["sample_key_sha256"], indices))
        groups.add(row["content_group_sha256"])
    paired_rows = sum(
        int(row["qwen3_positive"]) == 1
        and int(row["qwen25_positive"]) == 1
        and row["qwen3_offset_signature"] == row["qwen25_offset_signature"]
        and row["qwen3_candidate_ids"] == row["qwen25_candidate_ids"]
        for row in new_comparison
    )
    expectations = {
        "canonical_samples": new_summary["canonical_sample_count"] == 7265,
        "identity_samples": new_summary["identity_sample_count"] == 7265,
        "raw_m2_parents": len(new_rows) == 802,
        "raw_positive_groups": len(groups) == 104,
        "fit_calibration_m2_parents": new_summary[
            "fit_calibration_qwen3_m2_parent_count"
        ] == 410,
        "fit_calibration_positive_groups": new_summary[
            "fit_calibration_qwen3_positive_group_count"
        ] == 56,
        "overlap_clusters": len(clusters) == 401,
        "paired_row_equality": paired_rows == 802,
        "original_summary_counts": all(
            old_summary[key] == new_summary[key]
            for key in (
                "canonical_sample_count",
                "identity_sample_count",
                "qwen3_raw_m_ge_2_parent_count_all_splits",
                "fit_calibration_qwen3_positive_group_count",
                "fit_calibration_qwen3_m2_parent_count",
                "category_counts",
                "unicode_category_counts",
                "mmlu_subject_counts",
            )
        ),
    }
    mismatches.extend(key for key, passed in expectations.items() if not passed)
    result = {
        "schema_version": 1,
        "stage": "FPCT-3.5P",
        "status": (
            "PROVENANCE_CONFIRMED" if not mismatches
            else "FORENSIC_REPLAY_MISMATCH"
        ),
        "execution_sha": execution_sha,
        "historical_execution_sha": OLD_EXECUTION_SHA,
        "stable_attestation_sha256": lock["stable_attestation_sha256"],
        "counts": {
            "canonical_samples": new_summary["canonical_sample_count"],
            "identity_samples": new_summary["identity_sample_count"],
            "raw_m2_parents": len(new_rows),
            "raw_positive_groups": len(groups),
            "fit_calibration_m2_parents": new_summary[
                "fit_calibration_qwen3_m2_parent_count"
            ],
            "fit_calibration_positive_groups": new_summary[
                "fit_calibration_qwen3_positive_group_count"
            ],
            "overlap_clusters": len(clusters),
            "paired_equal_rows": paired_rows,
        },
        "equality": {
            "ordered_rows": old_normalized == new_normalized,
            "multiset_rows": Counter(map(_stable_sha, old_normalized))
            == Counter(map(_stable_sha, new_normalized)),
            "ordered_candidate_atoms": old_atoms == new_atoms,
            "multiset_candidate_atoms": Counter(map(str, old_atoms))
            == Counter(map(str, new_atoms)),
            "fixed_context_windows": old_contexts == new_contexts,
            "qwen3_qwen25_rows": old_comparison == new_comparison,
        },
        "mismatches": sorted(set(mismatches)),
        "primary_reason_x_secondary_flag": [
            {"primary_reason": key[0], "secondary_flag": key[1], "count": value}
            for key, value in sorted(cooccurrence.items())
        ],
        "normalized_projection_sha256": _stable_sha(new_normalized),
        "fixed_context_projection_sha256": _stable_sha(new_contexts),
    }
    _write_json(root / "replay_comparison.json", result)
    state = _read_json(root / "controller_state.json")
    state["state"] = result["status"]
    state["completed_tasks"] = list(TASK_ORDER)
    _write_json(root / "controller_state.json", state)
    if mismatches:
        raise ReplayError("sealed forensic replay mismatch: " + ", ".join(mismatches))
    return result


def finalize(execution_sha: str) -> None:
    _verify_lock(execution_sha)
    old = _configured_old()
    old.finalize_forensic(execution_sha, RESULT_BASE)
    result = compare(execution_sha)
    print(json.dumps({"status": result["status"], "execution_sha": execution_sha}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "forensic", "finalize", "compare"))
    parser.add_argument("--execution-sha")
    parser.add_argument("--task", choices=TASK_ORDER + ("all",), default="all")
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze()
        return 0
    execution_sha = args.execution_sha or _git("rev-parse", "HEAD")
    if args.mode == "forensic":
        tasks = TASK_ORDER if args.task == "all" else (args.task,)
        for task in tasks:
            run_task(task, execution_sha)
        return 0
    if args.mode == "finalize":
        finalize(execution_sha)
        return 0
    compare(execution_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
