from __future__ import annotations

"""Phase 2A-0 zero-GPU opportunity audit for calibrated no-transfer.

The script consumes frozen receiver-only and B6-native per-example CSVs.  It
never loads a model or imports torch.  Its primary outputs are a compact CSV of
event/accuracy/headroom aggregates and a JSON file containing the same rows,
source provenance, schema/field coverage, and bootstrap metadata.

The bootstrap is paired at the example level.  Canonical examples are sampled
synchronously across pair/seed cells, while pair and seed clusters are sampled
hierarchically for aggregate rows.  This preserves the repeated receiver-only
vector and shared sample difficulty instead of treating 12 copies of each
question as independent observations.
"""

import argparse
import csv
import glob
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


EVENT_NAMES: Tuple[str, ...] = (
    "receiver_correct_fused_correct_rate",
    "beneficial_transfer_rate",
    "harmful_transfer_rate",
    "receiver_wrong_fused_wrong_rate",
)

METRIC_NAMES: Tuple[str, ...] = (
    *EVENT_NAMES,
    "receiver_accuracy",
    "fused_accuracy",
    "oracle_accuracy",
    "best_fixed_accuracy",
    "oracle_headroom_over_fused",
    "oracle_headroom_over_receiver",
    "oracle_headroom_over_best_fixed",
    "fused_minus_receiver",
    "mean_transfer_utility",
)

CHOICE_FIELDS: Tuple[str, ...] = tuple(chr(code) for code in range(65, 75))
PRETRANSFER_NUMERIC_FIELDS: Tuple[str, ...] = (
    "cot_input_length",
    "candidate_count",
    "candidate_count_max",
    "one_to_many_rate",
    "alignment_entropy",
    "boundary_mismatch",
    "confidence",
    "fallback_rate",
)

FIELD_AUDIT: Mapping[str, Sequence[Mapping[str, Any]]] = {
    "A_pre_transfer": (
        {
            "fields": ["question", *CHOICE_FIELDS],
            "use": "deployment input; excluded from the primary scalar selector",
            "reason": "Available before transfer, but raw benchmark text creates memorization/lookup risk.",
        },
        {
            "fields": ["subject", "task", "pair"],
            "use": "group metadata only unless an explicitly task/pair-aware policy is preregistered",
            "reason": "Known before transfer but can learn base rates and weaken leave-one-group generalization.",
        },
        {
            "fields": ["cot_input_length"],
            "use": "eligible primary structural feature",
            "reason": "Receiver-tokenized prompt length is exactly recomputable before model forward.",
        },
        {
            "fields": [
                "alignment_bucket",
                "candidate_count",
                "candidate_count_max",
                "one_to_many_rate",
                "alignment_entropy",
                "boundary_mismatch",
                "confidence",
                "fallback_rate",
            ],
            "use": "eligible primary alignment features after redundancy filtering",
            "reason": "Computed during tokenizer/alignment input preparation before the fused forward.",
        },
    ),
    "B_post_fused_forward": (
        {
            "fields": ["pred", "cot_pred", "cot_gen_length", "cot_output"],
            "use": "diagnostic only for a no-transfer selector",
            "reason": "Requires completing B6 generation.",
        },
        {
            "fields": [
                "gate_diagnostics_status",
                "gate_record_count",
                "gate_token_count",
                "gate",
                "key_gate_mean",
                "key_gate_std",
                "key_gate_saturation_low_rate",
                "key_gate_saturation_high_rate",
                "value_gate_mean",
                "value_gate_std",
                "value_gate_saturation_low_rate",
                "value_gate_saturation_high_rate",
            ],
            "use": "diagnostic only",
            "reason": "Summarized from projector records after the fused call.",
        },
        {
            "fields": ["extraction_method_used", "extracted_normalized"],
            "use": "diagnostic only",
            "reason": "Derived from the generated fused answer when populated.",
        },
    ),
    "C_receiver_dual_pass_upper_bound": (
        {
            "fields": [
                "receiver.pred",
                "receiver.cot_pred",
                "receiver.cot_gen_length",
                "receiver.cot_output",
                "receiver.extraction_method_used",
                "receiver.extracted_normalized",
            ],
            "use": "dual-pass upper-bound analysis only",
            "reason": "Requires a complete receiver-only forward; current artifacts do not contain logits, margins, or calibrated confidence.",
        },
    ),
    "D_forbidden_or_leaky": (
        {
            "fields": [
                "true_answer",
                "is_correct",
                "ground_truth_normalized",
                "receiver_is_correct",
                "fused_is_correct",
                "four_way_event",
                "utility",
                "oracle_choice",
            ],
            "use": "outcome construction only; forbidden selector input",
            "reason": "Contains the label or a direct label-derived target.",
        },
        {
            "fields": ["question_id", "row_index", "filename", "timestamp", "checkpoint_id", "seed", "method"],
            "use": "join/split/provenance only; forbidden selector input",
            "reason": "Identity or run metadata can memorize benchmark examples or experimental conditions.",
        },
        {
            "fields": ["answer_latency_ms"],
            "use": "forbidden confirmatory selector input",
            "reason": "Post-forward and confounded by hardware, node load, and run ordering.",
        },
        {
            "fields": ["answer_method"],
            "use": "configuration provenance only",
            "reason": "Constant run metadata, not a sample feature.",
        },
    ),
}


@dataclass(frozen=True)
class TaskSpec:
    task: str
    expected_rows: int


@dataclass(frozen=True)
class PairSpec:
    pair: str
    label: str
    pair_type: str
    heterogeneous: bool


@dataclass
class LoadedCsv:
    path: Path
    sha256: str
    fieldnames: Tuple[str, ...]
    correctness: Dict[Tuple[str, str, str], bool]
    content_hashes: Dict[Tuple[str, str, str], str]
    labels: Dict[Tuple[str, str, str], str]
    nonempty_counts: Dict[str, int]
    numeric_summary: Dict[str, Dict[str, float]]
    redundancy: Dict[str, float]


@dataclass
class AuditData:
    pairs: Tuple[PairSpec, ...]
    seeds: Tuple[int, ...]
    tasks: Tuple[TaskSpec, ...]
    event_codes: Dict[str, np.ndarray]
    sample_keys: Dict[str, Tuple[Tuple[str, str, str], ...]]
    source_artifacts: List[Dict[str, Any]]
    schema: Tuple[str, ...]
    field_coverage: Dict[str, Dict[str, int]]
    pretransfer_summary: Dict[str, Dict[str, Any]]
    unique_sample_count: int
    unique_content_group_count: int
    duplicate_content_group_count: int
    duplicate_sample_count: int
    cross_task_duplicate_group_count: int
    task_content_sha256: Dict[str, str]
    dataset_content_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _strict_bool(value: str, *, path: Path, key: Tuple[str, str, str]) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid is_correct={value!r} for {key} in {path}")


def _normalized_text(value: str) -> str:
    return " ".join(value.strip().split())


def _content_hash(row: Mapping[str, str]) -> str:
    payload = {
        "question": _normalized_text(row.get("question", "")),
        "choices": [_normalized_text(row.get(name, "")) for name in CHOICE_FIELDS],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_key(task: str, row: Mapping[str, str], path: Path) -> Tuple[str, str, str]:
    subject = row.get("subject", "").strip()
    question_id = row.get("question_id", "").strip()
    if not subject or not question_id:
        raise ValueError(f"Missing subject/question_id in {path}")
    return task, subject, question_id


def _update_numeric(
    target: Dict[str, Dict[str, float]], field: str, raw_value: str
) -> None:
    if not raw_value.strip():
        return
    try:
        value = float(raw_value)
    except ValueError:
        return
    if not math.isfinite(value):
        return
    summary = target.setdefault(
        field,
        {"count": 0.0, "nonzero_count": 0.0, "min": value, "max": value, "sum": 0.0},
    )
    summary["count"] += 1.0
    summary["nonzero_count"] += float(value != 0.0)
    summary["min"] = min(summary["min"], value)
    summary["max"] = max(summary["max"], value)
    summary["sum"] += value


def _load_csv(path: Path, task: str) -> LoadedCsv:
    correctness: Dict[Tuple[str, str, str], bool] = {}
    content_hashes: Dict[Tuple[str, str, str], str] = {}
    labels: Dict[Tuple[str, str, str], str] = {}
    nonempty_counts: Dict[str, int] = {}
    numeric_summary: Dict[str, Dict[str, float]] = {}
    max_entropy_one_to_many_error = 0.0
    max_confidence_identity_error = 0.0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        fieldnames = tuple(reader.fieldnames)
        required = {"subject", "question_id", "question", "true_answer", "is_correct"}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(f"Missing required columns {sorted(missing)} in {path}")
        for row in reader:
            key = _sample_key(task, row, path)
            if key in correctness:
                raise ValueError(f"Duplicate sample key {key} in {path}")
            correctness[key] = _strict_bool(row.get("is_correct", ""), path=path, key=key)
            content_hashes[key] = _content_hash(row)
            labels[key] = _normalized_text(row.get("true_answer", "")).upper()
            for field in fieldnames:
                value = row.get(field, "")
                if value is not None and value.strip():
                    nonempty_counts[field] = nonempty_counts.get(field, 0) + 1
            for field in PRETRANSFER_NUMERIC_FIELDS:
                _update_numeric(numeric_summary, field, row.get(field, ""))
            try:
                entropy = float(row.get("alignment_entropy", ""))
                one_to_many = float(row.get("one_to_many_rate", ""))
                confidence = float(row.get("confidence", ""))
            except ValueError:
                continue
            max_entropy_one_to_many_error = max(
                max_entropy_one_to_many_error, abs(entropy - one_to_many)
            )
            max_confidence_identity_error = max(
                max_confidence_identity_error,
                abs(confidence - (1.0 - 0.5 * entropy)),
            )
    return LoadedCsv(
        path=path,
        sha256=_sha256(path),
        fieldnames=fieldnames,
        correctness=correctness,
        content_hashes=content_hashes,
        labels=labels,
        nonempty_counts=nonempty_counts,
        numeric_summary=numeric_summary,
        redundancy={
            "max_abs_alignment_entropy_minus_one_to_many_rate": max_entropy_one_to_many_error,
            "max_abs_confidence_minus_one_minus_half_entropy": max_confidence_identity_error,
        },
    )


def _resolve_one(pattern: str, artifact_root: Path) -> Path:
    raw = Path(pattern)
    candidate = raw if raw.is_absolute() else artifact_root / raw
    matches = sorted(Path(value).resolve() for value in glob.glob(str(candidate)))
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one artifact for {pattern!r} under {artifact_root}; "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def _declared_source_file(
    source: Mapping[str, Any],
    artifact_root: Path,
    *,
    path_key: str,
    sha_key: str,
) -> Tuple[Path, str] | None:
    path_value = str(source.get(path_key, "")).strip()
    sha_value = str(source.get(sha_key, "")).strip().lower()
    if not path_value and not sha_value:
        return None
    if not path_value or not sha_value:
        raise ValueError(f"Source must declare both {path_key} and {sha_key}")
    path = Path(path_value)
    path = path if path.is_absolute() else artifact_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != sha_value:
        raise ValueError(f"SHA mismatch for {path}: {actual} != {sha_value}")
    return path, actual


def _phase15_crosscheck(
    config: Mapping[str, Any], artifact_root_override: Path | None
) -> Dict[str, Any]:
    source = config.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Audit manifest needs a source object")
    artifact_root = (
        artifact_root_override.resolve()
        if artifact_root_override is not None
        else Path(str(source["artifact_root"])).resolve()
    )
    suite_file = _declared_source_file(
        source,
        artifact_root,
        path_key="phase1_suite_manifest",
        sha_key="phase1_suite_manifest_sha256",
    )
    execution_file = _declared_source_file(
        source,
        artifact_root,
        path_key="phase1_5_execution_manifest",
        sha_key="phase1_5_execution_manifest_sha256",
    )
    oracle_file = _declared_source_file(
        source,
        artifact_root,
        path_key="phase1_5_oracle_csv",
        sha_key="phase1_5_oracle_csv_sha256",
    )
    if suite_file is None and execution_file is None and oracle_file is None:
        return {"status": "not_declared"}
    if suite_file is None or execution_file is None or oracle_file is None:
        raise ValueError(
            "Formal provenance crosscheck requires Phase1 suite, Phase1.5 execution, "
            "and Phase1.5 oracle files"
        )
    suite_path, suite_sha = suite_file
    suite = _read_json(suite_path)
    artifact_commit = str(source.get("artifact_commit", "")).strip()
    suite_commit = str(suite.get("git_commit", "")).strip()
    if artifact_commit and suite_commit != artifact_commit:
        raise ValueError(
            f"Phase1 artifact commit mismatch: {suite_commit} != {artifact_commit}"
        )
    execution_path, execution_sha = execution_file
    oracle_path, oracle_sha = oracle_file
    with oracle_path.open(newline="", encoding="utf-8") as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if row.get("method") == "b6_native"
            and row.get("pair") == "__all__"
            and row.get("seed") == "all"
            and row.get("task") == "__pooled__"
            and row.get("aggregation_level") == "across_pairs"
        ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one Phase1.5 B6-native across-pair oracle row, found {len(matches)}"
        )
    old = matches[0]
    return {
        "status": "verified",
        "phase1_suite_manifest": str(suite_path),
        "phase1_suite_manifest_sha256": suite_sha,
        "phase1_artifact_commit": suite_commit,
        "phase1_5_execution_manifest": str(execution_path),
        "phase1_5_execution_manifest_sha256": execution_sha,
        "phase1_5_oracle_csv": str(oracle_path),
        "phase1_5_oracle_csv_sha256": oracle_sha,
        "old_estimand": "oracle_headroom_over_fused",
        "old_point": float(old["oracle_headroom_over_fused"]),
        "old_ci_low": float(old["bootstrap_ci_low"]),
        "old_ci_high": float(old["bootstrap_ci_high"]),
        "old_oracle_over_best_fixed_field": old.get(
            "oracle_headroom_over_best_fixed", ""
        ),
    }


def _dataset_prediction_path(
    run: Mapping[str, Any], task: str, artifact_root: Path
) -> Path:
    datasets = run.get("datasets")
    if not isinstance(datasets, Mapping) or task not in datasets:
        raise ValueError(f"Run {run.get('run_id')} has no dataset {task}")
    artifact = datasets[task]
    if not isinstance(artifact, Mapping):
        raise ValueError(f"Invalid dataset entry for {run.get('run_id')}/{task}")
    direct = artifact.get("csv", artifact.get("per_example_csv"))
    if direct:
        path = Path(str(direct))
        path = path if path.is_absolute() else artifact_root / path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    pattern = artifact.get("prediction_glob")
    if not pattern:
        raise ValueError(f"Missing prediction_glob for {run.get('run_id')}/{task}")
    return _resolve_one(str(pattern), artifact_root)


def _merge_field_counts(
    target: Dict[str, int], source: Mapping[str, int]
) -> None:
    for field, count in source.items():
        target[field] = target.get(field, 0) + int(count)


def _feature_file_summary(loaded: LoadedCsv) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for field, values in loaded.numeric_summary.items():
        count = int(values["count"])
        output[field] = {
            "count": count,
            "nonzero_count": int(values["nonzero_count"]),
            "min": values["min"],
            "max": values["max"],
            "mean": values["sum"] / count if count else None,
        }
    output["redundancy"] = dict(loaded.redundancy)
    return output


def _parse_specs(config: Mapping[str, Any]) -> Tuple[Tuple[PairSpec, ...], Tuple[int, ...], Tuple[TaskSpec, ...]]:
    raw_pairs = config.get("pairs")
    raw_seeds = config.get("seeds")
    raw_tasks = config.get("tasks")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("Audit manifest needs a non-empty pairs list")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("Audit manifest needs a non-empty seeds list")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("Audit manifest needs a non-empty tasks list")
    pairs = tuple(
        PairSpec(
            pair=str(item["id"]),
            label=str(item.get("label", item["id"])),
            pair_type=str(item.get("pair_type", "unspecified")),
            heterogeneous=bool(item.get("heterogeneous", False)),
        )
        for item in raw_pairs
    )
    seeds = tuple(int(seed) for seed in raw_seeds)
    tasks = tuple(
        TaskSpec(str(item["id"]), int(item["expected_rows"])) for item in raw_tasks
    )
    if len({item.pair for item in pairs}) != len(pairs):
        raise ValueError("Duplicate pair id in audit manifest")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seed in audit manifest")
    if len({item.task for item in tasks}) != len(tasks):
        raise ValueError("Duplicate task id in audit manifest")
    return pairs, seeds, tasks


def _load_audit_data(
    config: Mapping[str, Any], artifact_root_override: Path | None = None
) -> AuditData:
    pairs, seeds, tasks = _parse_specs(config)
    source = config.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Audit manifest needs a source object")
    artifact_root = (
        artifact_root_override.resolve()
        if artifact_root_override is not None
        else Path(str(source["artifact_root"])).resolve()
    )
    analysis_value = Path(str(source["phase1_analysis_manifest"]))
    analysis_path = (
        analysis_value if analysis_value.is_absolute() else artifact_root / analysis_value
    ).resolve()
    expected_manifest_sha = str(source.get("phase1_analysis_manifest_sha256", "")).strip()
    actual_manifest_sha = _sha256(analysis_path)
    if expected_manifest_sha and actual_manifest_sha != expected_manifest_sha:
        raise ValueError(
            f"Phase1 analysis manifest SHA mismatch: {actual_manifest_sha} != "
            f"{expected_manifest_sha}"
        )
    analysis = _read_json(analysis_path)
    raw_runs = analysis.get("runs")
    if not isinstance(raw_runs, list):
        raise ValueError(f"Phase1 analysis manifest has no runs list: {analysis_path}")
    runs: Dict[Tuple[str, str, int], Mapping[str, Any]] = {}
    runs_by_id: Dict[str, Mapping[str, Any]] = {}
    for run in raw_runs:
        if not isinstance(run, Mapping):
            continue
        run_id = str(run.get("run_id", ""))
        if run_id:
            runs_by_id[run_id] = run
        try:
            key = (str(run["pair"]), str(run["variant"]).lower(), int(run["seed"]))
        except (KeyError, TypeError, ValueError):
            continue
        if key in runs:
            raise ValueError(f"Duplicate Phase1 run key: {key}")
        runs[key] = run

    receiver_run_id = str(source.get("receiver_run_id", analysis.get("receiver_baseline_run_id", "")))
    receiver_run = runs_by_id.get(receiver_run_id)
    if receiver_run is None:
        receiver_key = (
            str(source.get("receiver_pair", "receiver")),
            str(source.get("receiver_variant", "b0")).lower(),
            int(source.get("receiver_seed", 42)),
        )
        receiver_run = runs.get(receiver_key)
    if receiver_run is None:
        raise ValueError("Cannot resolve receiver-only run")

    receiver_by_task: Dict[str, LoadedCsv] = {}
    source_artifacts: List[Dict[str, Any]] = []
    field_coverage: Dict[str, Dict[str, int]] = {"receiver_only": {}, "b6_native": {}}
    schema: Tuple[str, ...] | None = None
    for task_spec in tasks:
        path = _dataset_prediction_path(receiver_run, task_spec.task, artifact_root)
        loaded = _load_csv(path, task_spec.task)
        if len(loaded.correctness) != task_spec.expected_rows:
            raise ValueError(
                f"Unexpected receiver row count for {task_spec.task}: "
                f"{len(loaded.correctness)} != {task_spec.expected_rows}"
            )
        schema = loaded.fieldnames if schema is None else schema
        if loaded.fieldnames != schema:
            raise ValueError(f"Receiver schema mismatch in {path}")
        receiver_by_task[task_spec.task] = loaded
        _merge_field_counts(field_coverage["receiver_only"], loaded.nonempty_counts)
        source_artifacts.append(
            {
                "role": "receiver_only",
                "pair": "receiver",
                "seed": int(receiver_run.get("seed", 42)),
                "task": task_spec.task,
                "path": str(path),
                "sha256": loaded.sha256,
                "rows": len(loaded.correctness),
            }
        )

    if schema is None:
        raise ValueError("No receiver artifacts loaded")
    content_groups: Dict[str, List[Tuple[str, str, str]]] = {}
    task_content_sha256: Dict[str, str] = {}
    for receiver in receiver_by_task.values():
        for key, content_hash in receiver.content_hashes.items():
            content_groups.setdefault(content_hash, []).append(key)
    for task_spec in tasks:
        receiver = receiver_by_task[task_spec.task]
        payload = "\n".join(
            "\t".join((*key, receiver.content_hashes[key]))
            for key in sorted(receiver.content_hashes)
        ).encode("utf-8")
        task_content_sha256[task_spec.task] = hashlib.sha256(payload).hexdigest()
    dataset_content_sha256 = hashlib.sha256(
        json.dumps(
            task_content_sha256,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    expected_dataset_sha = str(source.get("dataset_content_sha256", "")).strip()
    if expected_dataset_sha and dataset_content_sha256 != expected_dataset_sha:
        raise ValueError(
            f"Evaluation dataset content SHA mismatch: {dataset_content_sha256} != "
            f"{expected_dataset_sha}"
        )
    duplicate_groups = [values for values in content_groups.values() if len(values) > 1]
    event_codes: Dict[str, np.ndarray] = {}
    sample_keys: Dict[str, Tuple[Tuple[str, str, str], ...]] = {}
    pretransfer_summary: Dict[str, Dict[str, Any]] = {}
    fused_variant = str(source.get("fused_variant", "b6")).lower()
    for task_spec in tasks:
        receiver = receiver_by_task[task_spec.task]
        ordered_keys = tuple(sorted(receiver.correctness))
        sample_keys[task_spec.task] = ordered_keys
        receiver_values = np.asarray(
            [receiver.correctness[key] for key in ordered_keys], dtype=np.bool_
        )
        task_events = np.empty(
            (len(pairs), len(seeds), task_spec.expected_rows), dtype=np.uint8
        )
        for pair_index, pair_spec in enumerate(pairs):
            pair_summary = pretransfer_summary.setdefault(
                pair_spec.pair,
                {
                    "seed_42_or_first_available": {},
                    "max_abs_alignment_entropy_minus_one_to_many_rate": 0.0,
                    "max_abs_confidence_minus_one_minus_half_entropy": 0.0,
                },
            )
            for seed_index, seed in enumerate(seeds):
                run = runs.get((pair_spec.pair, fused_variant, seed))
                if run is None:
                    raise ValueError(
                        f"Missing Phase1 B6-native run for {pair_spec.pair}/seed_{seed}"
                    )
                path = _dataset_prediction_path(run, task_spec.task, artifact_root)
                loaded = _load_csv(path, task_spec.task)
                if loaded.fieldnames != schema:
                    raise ValueError(f"B6 schema mismatch in {path}")
                if set(loaded.correctness) != set(ordered_keys):
                    raise ValueError(f"Sample-key mismatch between receiver and {path}")
                for key in ordered_keys:
                    if loaded.content_hashes[key] != receiver.content_hashes[key]:
                        raise ValueError(f"Input-content mismatch for {key} in {path}")
                    if loaded.labels[key] != receiver.labels[key]:
                        raise ValueError(f"Label mismatch for {key} in {path}")
                fused_values = np.asarray(
                    [loaded.correctness[key] for key in ordered_keys], dtype=np.bool_
                )
                codes = np.full(task_spec.expected_rows, 3, dtype=np.uint8)
                codes[np.logical_and(receiver_values, fused_values)] = 0
                codes[np.logical_and(~receiver_values, fused_values)] = 1
                codes[np.logical_and(receiver_values, ~fused_values)] = 2
                task_events[pair_index, seed_index] = codes
                _merge_field_counts(field_coverage["b6_native"], loaded.nonempty_counts)
                source_artifacts.append(
                    {
                        "role": "b6_native",
                        "pair": pair_spec.pair,
                        "seed": seed,
                        "task": task_spec.task,
                        "path": str(path),
                        "sha256": loaded.sha256,
                        "rows": len(loaded.correctness),
                    }
                )
                pair_summary[
                    "max_abs_alignment_entropy_minus_one_to_many_rate"
                ] = max(
                    pair_summary[
                        "max_abs_alignment_entropy_minus_one_to_many_rate"
                    ],
                    loaded.redundancy[
                        "max_abs_alignment_entropy_minus_one_to_many_rate"
                    ],
                )
                pair_summary[
                    "max_abs_confidence_minus_one_minus_half_entropy"
                ] = max(
                    pair_summary[
                        "max_abs_confidence_minus_one_minus_half_entropy"
                    ],
                    loaded.redundancy[
                        "max_abs_confidence_minus_one_minus_half_entropy"
                    ],
                )
                if seed == 42 or (42 not in seeds and seed == seeds[0]):
                    pair_summary["seed_42_or_first_available"][task_spec.task] = (
                        _feature_file_summary(loaded)
                    )
        event_codes[task_spec.task] = task_events

    source_artifacts.sort(
        key=lambda row: (row["role"], row["pair"], int(row["seed"]), row["task"])
    )
    return AuditData(
        pairs=pairs,
        seeds=seeds,
        tasks=tasks,
        event_codes=event_codes,
        sample_keys=sample_keys,
        source_artifacts=source_artifacts,
        schema=schema,
        field_coverage=field_coverage,
        pretransfer_summary=pretransfer_summary,
        unique_sample_count=sum(item.expected_rows for item in tasks),
        unique_content_group_count=len(content_groups),
        duplicate_content_group_count=len(duplicate_groups),
        duplicate_sample_count=sum(len(values) for values in duplicate_groups),
        cross_task_duplicate_group_count=sum(
            len({key[0] for key in values}) > 1 for values in duplicate_groups
        ),
        task_content_sha256=task_content_sha256,
        dataset_content_sha256=dataset_content_sha256,
    )


def _event_rates(codes: np.ndarray) -> np.ndarray:
    flat = codes.reshape(-1)
    counts = np.bincount(flat, minlength=4).astype(np.float64)
    return counts / counts.sum()


def _point_task_rates(data: AuditData) -> Dict[str, np.ndarray]:
    output: Dict[str, np.ndarray] = {}
    for task_spec in data.tasks:
        codes = data.event_codes[task_spec.task]
        rates = np.empty((*codes.shape[:2], 4), dtype=np.float64)
        for pair_index in range(codes.shape[0]):
            for seed_index in range(codes.shape[1]):
                rates[pair_index, seed_index] = _event_rates(
                    codes[pair_index, seed_index]
                )
        output[task_spec.task] = rates
    return output


def _bootstrap_task_rates(
    data: AuditData,
    *,
    samples: int,
    seed: int,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if batch_size <= 0:
        raise ValueError("bootstrap batch_size must be positive")
    output: Dict[str, np.ndarray] = {}
    cell_count = len(data.pairs) * len(data.seeds)
    for task_spec in data.tasks:
        codes = data.event_codes[task_spec.task]
        observations = np.moveaxis(codes, -1, 0).reshape(task_spec.expected_rows, cell_count)
        patterns, counts = np.unique(observations, axis=0, return_counts=True)
        indicator = np.zeros((len(patterns), cell_count * 4), dtype=np.uint8)
        row_indices = np.arange(len(patterns))[:, None]
        column_indices = np.arange(cell_count)[None, :] * 4 + patterns
        indicator[row_indices, column_indices] = 1
        probabilities = counts.astype(np.float64) / task_spec.expected_rows
        rates = np.empty(
            (samples, len(data.pairs), len(data.seeds), 4), dtype=np.float64
        )
        rng = np.random.default_rng(_stable_seed(seed, f"samples:{task_spec.task}"))
        for start in range(0, samples, batch_size):
            end = min(samples, start + batch_size)
            draws = rng.multinomial(
                task_spec.expected_rows, probabilities, size=end - start
            )
            event_counts = draws @ indicator
            rates[start:end] = event_counts.reshape(
                end - start, len(data.pairs), len(data.seeds), 4
            ) / task_spec.expected_rows
        output[task_spec.task] = rates
    return output


def _metrics_from_event_rates(event_rates: np.ndarray) -> Dict[str, np.ndarray]:
    both_correct = event_rates[..., 0]
    beneficial = event_rates[..., 1]
    harmful = event_rates[..., 2]
    both_wrong = event_rates[..., 3]
    receiver = both_correct + harmful
    fused = both_correct + beneficial
    oracle = both_correct + beneficial + harmful
    best_fixed = np.maximum(receiver, fused)
    return {
        "receiver_correct_fused_correct_rate": both_correct,
        "beneficial_transfer_rate": beneficial,
        "harmful_transfer_rate": harmful,
        "receiver_wrong_fused_wrong_rate": both_wrong,
        "receiver_accuracy": receiver,
        "fused_accuracy": fused,
        "oracle_accuracy": oracle,
        "best_fixed_accuracy": best_fixed,
        "oracle_headroom_over_fused": oracle - fused,
        "oracle_headroom_over_receiver": oracle - receiver,
        "oracle_headroom_over_best_fixed": oracle - best_fixed,
        "fused_minus_receiver": fused - receiver,
        "mean_transfer_utility": beneficial - harmful,
    }


def _task_weights(data: AuditData, task_ids: Sequence[str], weighting: str) -> np.ndarray:
    if weighting == "single_task":
        if len(task_ids) != 1:
            raise ValueError("single_task weighting requires one task")
        return np.ones(1, dtype=np.float64)
    if weighting == "task_macro":
        return np.full(len(task_ids), 1.0 / len(task_ids), dtype=np.float64)
    if weighting == "sample_weighted":
        sizes = {
            item.task: item.expected_rows for item in data.tasks
        }
        counts = np.asarray([sizes[task] for task in task_ids], dtype=np.float64)
        return counts / counts.sum()
    raise ValueError(f"Unknown weighting: {weighting}")


def _aggregate_tasks(
    task_values: Mapping[str, np.ndarray],
    data: AuditData,
    task_ids: Sequence[str],
    weighting: str,
) -> np.ndarray:
    weights = _task_weights(data, task_ids, weighting)
    result: np.ndarray | None = None
    for weight, task in zip(weights, task_ids):
        value = task_values[task] * weight
        result = value if result is None else result + value
    if result is None:
        raise ValueError("Cannot aggregate an empty task list")
    return result


def _point_scope_events(
    point_task_rates: Mapping[str, np.ndarray],
    data: AuditData,
    *,
    pair_indices: Sequence[int],
    seed_index: int | None,
    task_ids: Sequence[str],
    weighting: str,
) -> np.ndarray:
    task_values: Dict[str, np.ndarray] = {}
    for task in task_ids:
        values = point_task_rates[task][np.asarray(pair_indices)]
        if seed_index is not None:
            values = values[:, seed_index]
        else:
            values = values.mean(axis=1)
        task_values[task] = values.mean(axis=0)
    return _aggregate_tasks(task_values, data, task_ids, weighting)


def _bootstrap_scope_events(
    boot_task_rates: Mapping[str, np.ndarray],
    data: AuditData,
    *,
    pair_indices: Sequence[int],
    seed_index: int | None,
    task_ids: Sequence[str],
    weighting: str,
    resample_pairs: bool,
    resample_seeds: bool,
    base_seed: int,
    label: str,
) -> np.ndarray:
    sample_count = next(iter(boot_task_rates.values())).shape[0]
    pair_indices_array = np.asarray(pair_indices, dtype=np.int64)
    if not len(pair_indices_array):
        raise ValueError("Scope has no pairs")
    rng = np.random.default_rng(_stable_seed(base_seed, f"clusters:{label}"))
    rows = np.arange(sample_count)
    pair_slots = len(pair_indices_array)
    seed_slots = len(data.seeds)
    if resample_pairs:
        selected_pair_offsets = rng.integers(
            0, pair_slots, size=(sample_count, pair_slots)
        )
        selected_pairs = pair_indices_array[selected_pair_offsets]
    else:
        selected_pairs = np.broadcast_to(
            pair_indices_array, (sample_count, pair_slots)
        )
    if seed_index is not None:
        selected_seeds = np.full(
            (sample_count, pair_slots, 1), seed_index, dtype=np.int64
        )
    elif resample_seeds:
        selected_seeds = rng.integers(
            0, seed_slots, size=(sample_count, pair_slots, seed_slots)
        )
    else:
        selected_seeds = np.broadcast_to(
            np.arange(seed_slots, dtype=np.int64),
            (sample_count, pair_slots, seed_slots),
        )

    task_values: Dict[str, np.ndarray] = {}
    for task in task_ids:
        rates = boot_task_rates[task]
        aggregate = np.zeros((sample_count, 4), dtype=np.float64)
        for pair_slot in range(pair_slots):
            pair_values = np.zeros((sample_count, 4), dtype=np.float64)
            for seed_slot in range(selected_seeds.shape[2]):
                pair_values += rates[
                    rows,
                    selected_pairs[:, pair_slot],
                    selected_seeds[:, pair_slot, seed_slot],
                ]
            pair_values /= selected_seeds.shape[2]
            aggregate += pair_values
        task_values[task] = aggregate / pair_slots
    return _aggregate_tasks(task_values, data, task_ids, weighting)


def _quantile_bounds(values: np.ndarray, confidence: float) -> Tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    try:
        result = np.quantile(values, [alpha, 1.0 - alpha], method="linear")
    except TypeError:  # pragma: no cover - numpy < 1.22.
        result = np.quantile(values, [alpha, 1.0 - alpha], interpolation="linear")
    return float(result[0]), float(result[1])


def _row_from_scope(
    *,
    data: AuditData,
    point_task_rates: Mapping[str, np.ndarray],
    boot_task_rates: Mapping[str, np.ndarray],
    aggregation_level: str,
    pair_label: str,
    seed_label: str | int,
    task_label: str,
    pair_indices: Sequence[int],
    seed_index: int | None,
    task_ids: Sequence[str],
    weighting: str,
    resample_pairs: bool,
    resample_seeds: bool,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    label = f"{aggregation_level}:{pair_label}:{seed_label}:{task_label}:{weighting}"
    point_events = _point_scope_events(
        point_task_rates,
        data,
        pair_indices=pair_indices,
        seed_index=seed_index,
        task_ids=task_ids,
        weighting=weighting,
    )
    boot_events = _bootstrap_scope_events(
        boot_task_rates,
        data,
        pair_indices=pair_indices,
        seed_index=seed_index,
        task_ids=task_ids,
        weighting=weighting,
        resample_pairs=resample_pairs,
        resample_seeds=resample_seeds,
        base_seed=bootstrap_seed,
        label=label,
    )
    point_metrics = _metrics_from_event_rates(point_events)
    boot_metrics = _metrics_from_event_rates(boot_events)
    unique_samples = sum(
        next(item.expected_rows for item in data.tasks if item.task == task)
        for task in task_ids
    )
    observation_seeds = 1 if seed_index is not None else len(data.seeds)
    row: Dict[str, Any] = {
        "aggregation_level": aggregation_level,
        "weighting": weighting,
        "pair": pair_label,
        "seed": seed_label,
        "task": task_label,
        "n_pairs": len(pair_indices),
        "n_seeds": observation_seeds,
        "n_tasks": len(task_ids),
        "n_unique_samples": unique_samples,
        "n_repeated_observations": unique_samples * len(pair_indices) * observation_seeds,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_confidence": bootstrap_confidence,
        "bootstrap_level": (
            "pairs_then_seeds_with_synchronous_task_stratified_paired_examples"
            if resample_pairs and resample_seeds
            else "pairs_with_synchronous_task_stratified_paired_examples"
            if resample_pairs
            else "seeds_with_synchronous_task_stratified_paired_examples"
            if resample_seeds
            else "synchronous_task_stratified_paired_examples"
        ),
    }
    for metric in METRIC_NAMES:
        point_value = float(np.asarray(point_metrics[metric]))
        low, high = _quantile_bounds(np.asarray(boot_metrics[metric]), bootstrap_confidence)
        row[metric] = point_value
        row[f"{metric}_ci_low"] = low
        row[f"{metric}_ci_high"] = high
    receiver = row["receiver_accuracy"]
    fused = row["fused_accuracy"]
    row["best_fixed_policy"] = "fused" if fused > receiver else "receiver"
    return row


def _build_rows(
    data: AuditData,
    *,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
    bootstrap_batch_size: int,
) -> List[Dict[str, Any]]:
    point_task_rates = _point_task_rates(data)
    boot_task_rates = _bootstrap_task_rates(
        data,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        batch_size=bootstrap_batch_size,
    )
    rows: List[Dict[str, Any]] = []
    task_ids = [item.task for item in data.tasks]
    all_pairs = list(range(len(data.pairs)))
    hetero_pairs = [
        index for index, item in enumerate(data.pairs) if item.heterogeneous
    ]

    for pair_index, pair_spec in enumerate(data.pairs):
        for seed_index, seed in enumerate(data.seeds):
            for task in task_ids:
                rows.append(
                    _row_from_scope(
                        data=data,
                        point_task_rates=point_task_rates,
                        boot_task_rates=boot_task_rates,
                        aggregation_level="pair_seed_task",
                        pair_label=pair_spec.pair,
                        seed_label=seed,
                        task_label=task,
                        pair_indices=[pair_index],
                        seed_index=seed_index,
                        task_ids=[task],
                        weighting="single_task",
                        resample_pairs=False,
                        resample_seeds=False,
                        bootstrap_samples=bootstrap_samples,
                        bootstrap_confidence=bootstrap_confidence,
                        bootstrap_seed=bootstrap_seed,
                    )
                )
            for weighting in ("task_macro", "sample_weighted"):
                rows.append(
                    _row_from_scope(
                        data=data,
                        point_task_rates=point_task_rates,
                        boot_task_rates=boot_task_rates,
                        aggregation_level="pair_seed",
                        pair_label=pair_spec.pair,
                        seed_label=seed,
                        task_label="__all__",
                        pair_indices=[pair_index],
                        seed_index=seed_index,
                        task_ids=task_ids,
                        weighting=weighting,
                        resample_pairs=False,
                        resample_seeds=False,
                        bootstrap_samples=bootstrap_samples,
                        bootstrap_confidence=bootstrap_confidence,
                        bootstrap_seed=bootstrap_seed,
                    )
                )
        for weighting in ("task_macro", "sample_weighted"):
            rows.append(
                _row_from_scope(
                    data=data,
                    point_task_rates=point_task_rates,
                    boot_task_rates=boot_task_rates,
                    aggregation_level="pair_across_seeds",
                    pair_label=pair_spec.pair,
                    seed_label="all",
                    task_label="__all__",
                    pair_indices=[pair_index],
                    seed_index=None,
                    task_ids=task_ids,
                    weighting=weighting,
                    resample_pairs=False,
                    resample_seeds=True,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_confidence=bootstrap_confidence,
                    bootstrap_seed=bootstrap_seed,
                )
            )

    for seed_index, seed in enumerate(data.seeds):
        for weighting in ("task_macro", "sample_weighted"):
            rows.append(
                _row_from_scope(
                    data=data,
                    point_task_rates=point_task_rates,
                    boot_task_rates=boot_task_rates,
                    aggregation_level="seed_pair_balanced",
                    pair_label="__all__",
                    seed_label=seed,
                    task_label="__all__",
                    pair_indices=all_pairs,
                    seed_index=seed_index,
                    task_ids=task_ids,
                    weighting=weighting,
                    resample_pairs=True,
                    resample_seeds=False,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_confidence=bootstrap_confidence,
                    bootstrap_seed=bootstrap_seed,
                )
            )

    for task in task_ids:
        rows.append(
            _row_from_scope(
                data=data,
                point_task_rates=point_task_rates,
                boot_task_rates=boot_task_rates,
                aggregation_level="task_pair_balanced",
                pair_label="__all__",
                seed_label="all",
                task_label=task,
                pair_indices=all_pairs,
                seed_index=None,
                task_ids=[task],
                weighting="single_task",
                resample_pairs=True,
                resample_seeds=True,
                bootstrap_samples=bootstrap_samples,
                bootstrap_confidence=bootstrap_confidence,
                bootstrap_seed=bootstrap_seed,
            )
        )

    for pair_label, pair_indices in (
        ("__all__", all_pairs),
        ("__heterogeneous__", hetero_pairs),
    ):
        if not pair_indices:
            continue
        for weighting in ("task_macro", "sample_weighted"):
            rows.append(
                _row_from_scope(
                    data=data,
                    point_task_rates=point_task_rates,
                    boot_task_rates=boot_task_rates,
                    aggregation_level="pair_balanced",
                    pair_label=pair_label,
                    seed_label="all",
                    task_label="__all__",
                    pair_indices=pair_indices,
                    seed_index=None,
                    task_ids=task_ids,
                    weighting=weighting,
                    resample_pairs=True,
                    resample_seeds=True,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_confidence=bootstrap_confidence,
                    bootstrap_seed=bootstrap_seed,
                )
            )
    return rows


def _aggregate_pretransfer_summary(data: AuditData) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for pair_spec in data.pairs:
        raw = data.pretransfer_summary[pair_spec.pair]
        tasks = raw["seed_42_or_first_available"]
        pair_result: Dict[str, Any] = {
            "pair_type": pair_spec.pair_type,
            "heterogeneous": pair_spec.heterogeneous,
            "n_unique_samples": sum(
                int(task_values.get("candidate_count", {}).get("count", 0))
                for task_values in tasks.values()
            ),
            "max_abs_alignment_entropy_minus_one_to_many_rate": raw[
                "max_abs_alignment_entropy_minus_one_to_many_rate"
            ],
            "max_abs_confidence_minus_one_minus_half_entropy": raw[
                "max_abs_confidence_minus_one_minus_half_entropy"
            ],
        }
        for field in PRETRANSFER_NUMERIC_FIELDS:
            field_values = [
                task_values[field]
                for task_values in tasks.values()
                if field in task_values
            ]
            count = sum(int(item["count"]) for item in field_values)
            nonzero = sum(int(item["nonzero_count"]) for item in field_values)
            if count:
                pair_result[field] = {
                    "count": count,
                    "nonzero_count": nonzero,
                    "nonzero_rate": nonzero / count,
                    "min": min(float(item["min"]) for item in field_values),
                    "max": max(float(item["max"]) for item in field_values),
                    "mean": sum(float(item["mean"]) * int(item["count"]) for item in field_values)
                    / count,
                }
        output[pair_spec.pair] = pair_result
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("No aggregate rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run_audit(
    manifest_path: Path,
    output_csv: Path,
    output_json: Path,
    *,
    artifact_root_override: Path | None = None,
    bootstrap_samples_override: int | None = None,
) -> Dict[str, Any]:
    manifest_path = manifest_path.resolve()
    config = _read_json(manifest_path)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported Phase2A-0 audit manifest schema")
    constraints = config.get("constraints", {})
    if not isinstance(constraints, Mapping) or not all(
        constraints.get(name) is False
        for name in ("gpu", "training", "checkpoint_mutation", "selector_training")
    ):
        raise ValueError("Manifest must explicitly freeze the zero-GPU/no-training constraints")
    bootstrap = config.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ValueError("Audit manifest needs bootstrap settings")
    bootstrap_samples = int(
        bootstrap_samples_override
        if bootstrap_samples_override is not None
        else bootstrap.get("samples", 10000)
    )
    bootstrap_confidence = float(bootstrap.get("confidence", 0.95))
    bootstrap_seed = int(bootstrap.get("seed", 20260719))
    bootstrap_batch_size = int(bootstrap.get("batch_size", 250))
    if not 0.0 < bootstrap_confidence < 1.0:
        raise ValueError("bootstrap confidence must be between 0 and 1")
    phase15_crosscheck = _phase15_crosscheck(config, artifact_root_override)
    data = _load_audit_data(config, artifact_root_override)
    rows = _build_rows(
        data,
        bootstrap_samples=bootstrap_samples,
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_seed=bootstrap_seed,
        bootstrap_batch_size=bootstrap_batch_size,
    )
    pair_balanced = [
        row
        for row in rows
        if row["aggregation_level"] == "pair_balanced"
        and row["pair"] == "__all__"
    ]
    sample_weighted = next(
        row for row in pair_balanced if row["weighting"] == "sample_weighted"
    )
    if phase15_crosscheck.get("status") == "verified" and not math.isclose(
        sample_weighted["oracle_headroom_over_best_fixed"],
        float(phase15_crosscheck["old_point"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            "New oracle-over-best-fixed point does not match the frozen Phase1.5 "
            "oracle-over-fused point despite fused being the audited best fixed policy"
        )
    result: Dict[str, Any] = {
        "schema_version": 1,
        "phase": "2A-0",
        "status": "complete_zero_gpu_audit",
        "audit_manifest": str(manifest_path),
        "audit_manifest_sha256": _sha256(manifest_path),
        "constraints": dict(constraints),
        "source_commit": config.get("source_commit"),
        "source_artifact_commit": config.get("source", {}).get("artifact_commit"),
        "phase1_5_crosscheck": phase15_crosscheck,
        "source_artifacts": data.source_artifacts,
        "integrity": {
            "source_file_count": len(data.source_artifacts),
            "receiver_file_count": sum(
                row["role"] == "receiver_only" for row in data.source_artifacts
            ),
            "b6_native_file_count": sum(
                row["role"] == "b6_native" for row in data.source_artifacts
            ),
            "unique_sample_count": data.unique_sample_count,
            "unique_content_group_count": data.unique_content_group_count,
            "duplicate_content_group_count": data.duplicate_content_group_count,
            "samples_in_duplicate_content_groups": data.duplicate_sample_count,
            "cross_task_duplicate_content_group_count": data.cross_task_duplicate_group_count,
            "task_content_sha256": data.task_content_sha256,
            "dataset_content_sha256": data.dataset_content_sha256,
            "repeated_pair_seed_observation_count": data.unique_sample_count
            * len(data.pairs)
            * len(data.seeds),
            "canonical_sample_key": ["task", "subject", "question_id"],
            "future_split_group_key": "normalized question+choices SHA256; identical-content rows share a split",
            "content_hash_validated_across_all_files": True,
            "label_validated_across_all_files": True,
            "row_order_used_for_pairing": False,
            "schema": list(data.schema),
        },
        "bootstrap": {
            "samples": bootstrap_samples,
            "confidence": bootstrap_confidence,
            "seed": bootstrap_seed,
            "batch_size": bootstrap_batch_size,
            "paired": True,
            "sample_resampling": "canonical examples sampled synchronously across every pair/seed cell, stratified by task",
            "cluster_resampling": "pairs, then seeds within selected pair",
            "nonlinear_best_fixed": "max(receiver,fused) recomputed inside every draw",
        },
        "field_audit": {
            "classification": FIELD_AUDIT,
            "nonempty_counts_by_role": data.field_coverage,
            "pretransfer_feature_summary_seed42": _aggregate_pretransfer_summary(data),
            "known_missing_fields": [
                "receiver option logits/probabilities/margin/entropy",
                "fused option logits/probabilities/margin/entropy",
                "sender token length and sender/receiver token-length ratio",
                "per-example full layer/head/token gate distributions",
            ],
        },
        "pair_balanced_summary": pair_balanced,
        "aggregate_rows": rows,
        "outputs": {
            "csv": str(output_csv.resolve()),
            "json": str(output_json.resolve()),
        },
    }
    _write_csv(output_csv.resolve(), rows)
    _write_json(output_json.resolve(), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Phase 2A-0 receiver/B6 opportunity audit on CPU only"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Optional override for the shared Phase1 artifact checkout root",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=None,
        help="Override manifest bootstrap draws (useful only for tests/smoke audits)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_audit(
        args.manifest,
        args.output_csv,
        args.output_json,
        artifact_root_override=args.artifact_root,
        bootstrap_samples_override=args.bootstrap_samples,
    )
    for row in result["pair_balanced_summary"]:
        print(
            f"{row['weighting']}: receiver={row['receiver_accuracy']:.6f}, "
            f"fused={row['fused_accuracy']:.6f}, oracle={row['oracle_accuracy']:.6f}, "
            f"oracle-best-fixed={row['oracle_headroom_over_best_fixed']:.6f} "
            f"[{row['oracle_headroom_over_best_fixed_ci_low']:.6f}, "
            f"{row['oracle_headroom_over_best_fixed_ci_high']:.6f}]"
        )
    print(f"Wrote {len(result['aggregate_rows'])} aggregate rows")
    print(f"CSV: {Path(args.output_csv).resolve()}")
    print(f"JSON: {Path(args.output_json).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
