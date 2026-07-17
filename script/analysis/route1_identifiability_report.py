from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - torch environments provide numpy.
    np = None  # type: ignore[assignment]


Row = Dict[str, str]
SampleKey = Tuple[str, ...]

DIAGNOSTIC_FIELDS = (
    "alignment_bucket",
    "candidate_count",
    "alignment_entropy",
    "boundary_mismatch",
    "confidence",
    "gate",
)
NUMERIC_DIAGNOSTIC_FIELDS = (
    "candidate_count",
    "alignment_entropy",
    "boundary_mismatch",
    "confidence",
    "gate",
)

DEFAULT_COMPARISONS = (
    ("c2c_longest_vs_receiver", "B0", "B1"),
    ("hard_span_vs_receiver", "B0", "B2"),
    ("hard_span_vs_longest", "B1", "B2"),
    ("soft_candidates", "B2", "B3"),
    ("static_entropy", "B3", "B4"),
    ("gate_capacity", "B2-constant", "B5"),
    ("full_over_hard_span", "B2", "B6"),
    ("full_over_static_entropy", "B4", "B6"),
    ("full_over_gate_only", "B5", "B6"),
    ("entropy_values", "B6-constant", "B6"),
    ("entropy_position", "B6-shuffle", "B6"),
)

DEFAULT_FINAL_GATE_SEEDS = (42, 43, 44)
CANONICAL_TASK_EXPECTED_ROWS = {
    "ai2-arc": 1150,
    "openbookqa": 500,
    "mmlu-redux": 5615,
}
PAIRED_BUCKET_COMPARISONS = frozenset(
    {
        "soft_candidates",
        "static_entropy",
        "gate_capacity",
        "gate_capacity_confounded",
        "full_over_static_entropy",
        "full_over_gate_only",
        "entropy_values",
        "entropy_position",
    }
)

COMPONENT_ALIASES = {
    "soft_gate_interaction": "full_over_static_entropy",
    "gate_soft_interaction": "full_over_gate_only",
    "entropy_constant_counterfactual": "entropy_values",
    "entropy_shuffle_counterfactual": "entropy_position",
    "gate_capacity_static_scale_confounded": "gate_capacity_confounded",
}


@dataclass(frozen=True)
class RunSpec:
    method: str
    pair: str
    seed: int
    task: str
    csv_path: Path
    receiver_csv_path: Path | None = None
    receiver_method: str | None = None
    gate_diagnostics_posthoc_path: Path | None = None


@dataclass
class Sample:
    key: SampleKey
    correct: bool
    receiver_correct: bool | None
    diagnostics: Dict[str, str]


@dataclass
class RunData:
    spec: RunSpec
    samples: Dict[SampleKey, Sample]
    fieldnames: set[str]


@dataclass(frozen=True)
class ComparisonSpec:
    name: str
    baseline: str
    candidate: str


@dataclass
class ReceiverAttachment:
    values: Dict[SampleKey, bool]
    source_kind: str
    source_pair: str | None = None
    source_method: str | None = None
    source_seed: int | None = None
    source_csv: str | None = None


@dataclass(frozen=True)
class PairedCluster:
    comparison: str
    pair: str
    seed: int
    baseline_method: str
    candidate_method: str
    differences: Tuple[int, ...]


def _read_csv(path: Path) -> Tuple[List[Row], set[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader), set(reader.fieldnames)


def _first_nonempty(row: Mapping[str, str], names: Sequence[str]) -> str:
    for name in names:
        value = row.get(name, "").strip()
        if value:
            return value
    return ""


def _parse_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in {"1", "true", "yes", "y", "correct"}:
        return True
    if normalized in {"0", "false", "no", "n", "incorrect"}:
        return False
    return None


def _normalized_answer(value: str) -> str:
    return " ".join(value.strip().upper().split())


def _infer_correct(row: Mapping[str, str], prefix: str = "") -> bool:
    correct_names = (
        f"{prefix}is_correct",
        f"{prefix}correct",
    )
    for name in correct_names:
        if name not in row:
            continue
        parsed = _parse_bool(row.get(name, ""))
        if parsed is not None:
            return parsed

    pred_names = (
        f"{prefix}extracted_normalized",
        f"{prefix}pred",
        f"{prefix}prediction",
        f"{prefix}cot_pred",
    )
    truth_names = (
        "ground_truth_normalized",
        "true_answer",
        "answer",
        "label",
        "target",
    )
    pred = _first_nonempty(row, pred_names)
    truth = _first_nonempty(row, truth_names)
    if pred and truth:
        return _normalized_answer(pred) == _normalized_answer(truth)
    raise ValueError(
        "Cannot infer correctness: expected is_correct/correct or prediction and truth "
        f"columns; available columns={sorted(row)}"
    )


def _infer_receiver_correct(row: Mapping[str, str]) -> bool | None:
    for name in (
        "receiver_is_correct",
        "receiver_correct",
        "baseline_is_correct",
        "baseline_correct",
    ):
        if name in row:
            parsed = _parse_bool(row.get(name, ""))
            if parsed is not None:
                return parsed

    receiver_pred = _first_nonempty(
        row,
        (
            "receiver_extracted_normalized",
            "receiver_pred",
            "receiver_prediction",
            "baseline_pred",
            "baseline_prediction",
        ),
    )
    truth = _first_nonempty(
        row,
        (
            "ground_truth_normalized",
            "true_answer",
            "answer",
            "label",
            "target",
        ),
    )
    if receiver_pred and truth:
        return _normalized_answer(receiver_pred) == _normalized_answer(truth)
    return None


def _sample_key(row: Mapping[str, str], row_index: int) -> SampleKey:
    subject = _first_nonempty(row, ("subject", "category", "subset"))
    for name in ("sample_id", "example_id", "id", "question_id", "index"):
        value = row.get(name, "").strip()
        if value:
            return (name, subject, value) if subject else (name, value)
    return ("row_index", str(row_index))


def _load_samples(path: Path) -> Tuple[Dict[SampleKey, Sample], set[str]]:
    rows, fieldnames = _read_csv(path)
    samples: Dict[SampleKey, Sample] = {}
    for index, row in enumerate(rows):
        key = _sample_key(row, index)
        if key in samples:
            raise ValueError(f"Duplicate sample key {key!r} in {path}")
        diagnostics = {name: row.get(name, "").strip() for name in DIAGNOSTIC_FIELDS}
        samples[key] = Sample(
            key=key,
            correct=_infer_correct(row),
            receiver_correct=_infer_receiver_correct(row),
            diagnostics=diagnostics,
        )
    return samples, fieldnames


def _manifest_entries(data: Any) -> Tuple[List[Mapping[str, Any]], Mapping[str, Any]]:
    if isinstance(data, list):
        return data, {}
    if not isinstance(data, dict):
        raise ValueError("Manifest JSON must be a list or an object containing 'runs'")
    entries = data.get("runs", data.get("experiments"))
    if not isinstance(entries, list):
        raise ValueError("Manifest object must contain a list under 'runs'")
    return entries, data


def _resolve_prediction_glob(pattern: str, manifest_dir: Path) -> Path:
    raw = Path(pattern)
    candidates = (
        [raw]
        if raw.is_absolute()
        else [
            manifest_dir / raw,
            Path.cwd() / raw,
            Path(__file__).resolve().parents[2] / raw,
        ]
    )
    matches: set[Path] = set()
    for candidate in candidates:
        for match in glob.glob(str(candidate)):
            path = Path(match).resolve()
            if path.is_file():
                matches.add(path)
    ordered = sorted(matches)
    if not ordered:
        raise FileNotFoundError(f"No prediction artifact matches glob: {pattern}")
    if len(ordered) > 1:
        raise ValueError(
            f"Prediction glob must match exactly one file: {pattern}; matches={ordered}"
        )
    return ordered[0]


def _expand_suite_runs(
    entries: Sequence[Mapping[str, Any]], manifest_dir: Path
) -> List[Mapping[str, Any]]:
    expanded: List[Mapping[str, Any]] = []
    for entry in entries:
        datasets = entry.get("datasets")
        if not isinstance(datasets, dict):
            expanded.append(entry)
            continue
        method = entry.get("method", entry.get("variant"))
        for task, artifact in sorted(datasets.items()):
            if not isinstance(artifact, dict):
                raise ValueError(
                    f"Suite dataset artifact must be an object: {entry.get('run_id')}/{task}"
                )
            direct = artifact.get("csv", artifact.get("per_example_csv"))
            if direct:
                csv_path = _resolve_path(direct, manifest_dir, "dataset csv")
            else:
                pattern = artifact.get("prediction_glob")
                if not pattern:
                    raise ValueError(
                        f"Suite dataset artifact needs prediction_glob: {entry.get('run_id')}/{task}"
                    )
                csv_path = _resolve_prediction_glob(str(pattern), manifest_dir)
            expanded.append(
                {
                    "method": method,
                    "pair": entry.get("pair"),
                    "seed": entry.get("seed"),
                    "task": task,
                    "csv": str(csv_path),
                    "gate_diagnostics_posthoc": (
                        entry.get("posthoc_gate_diagnostics", {}).get("artifact")
                        if isinstance(entry.get("posthoc_gate_diagnostics"), Mapping)
                        else None
                    ),
                }
            )
    return expanded


def _resolve_path(value: Any, manifest_dir: Path, field: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"Manifest entry is missing {field}")
    path = Path(str(value))
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def _load_manifest(
    manifest_path: Path,
) -> Tuple[List[RunSpec], List[ComparisonSpec], str, Mapping[str, Any]]:
    if manifest_path.suffix.lower() == ".csv":
        entries, _ = _read_csv(manifest_path)
        metadata: Mapping[str, Any] = {}
    else:
        with manifest_path.open(encoding="utf-8") as handle:
            entries, metadata = _manifest_entries(json.load(handle))

    default_receiver_method = str(metadata.get("receiver_method", "B0"))
    manifest_dir = manifest_path.parent
    entries = _expand_suite_runs(entries, manifest_dir)
    specs: List[RunSpec] = []
    seen: set[Tuple[str, str, int, str]] = set()
    for entry in entries:
        method = str(entry.get("method", "")).strip()
        pair = str(entry.get("pair", "")).strip()
        task = str(entry.get("task", entry.get("dataset", ""))).strip()
        if not method or not pair or not task:
            raise ValueError("Each manifest entry needs method, pair, and task")
        try:
            seed = int(entry.get("seed"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid seed for {method}/{pair}/{task}") from exc

        csv_value = entry.get(
            "csv",
            entry.get("per_example_csv", entry.get("path", entry.get("predictions"))),
        )
        csv_path = _resolve_path(csv_value, manifest_dir, "csv/per_example_csv")
        receiver_value = entry.get("receiver_csv", entry.get("baseline_csv"))
        receiver_csv_path = (
            _resolve_path(receiver_value, manifest_dir, "receiver_csv")
            if receiver_value is not None and str(receiver_value).strip()
            else None
        )
        receiver_method_value = entry.get("receiver_method")
        receiver_method = (
            str(receiver_method_value).strip() if receiver_method_value else None
        )
        posthoc_value = entry.get("gate_diagnostics_posthoc")
        gate_diagnostics_posthoc_path = (
            _resolve_path(
                posthoc_value,
                manifest_dir,
                "gate_diagnostics_posthoc",
            )
            if posthoc_value is not None and str(posthoc_value).strip()
            else None
        )
        key = (pair, method, seed, task)
        if key in seen:
            raise ValueError(f"Duplicate manifest run: {key}")
        seen.add(key)
        specs.append(
            RunSpec(
                method=method,
                pair=pair,
                seed=seed,
                task=task,
                csv_path=csv_path,
                receiver_csv_path=receiver_csv_path,
                receiver_method=receiver_method,
                gate_diagnostics_posthoc_path=gate_diagnostics_posthoc_path,
            )
        )

    comparisons: List[ComparisonSpec] = []
    raw_comparisons = metadata.get(
        "comparisons", metadata.get("component_comparisons", [])
    )
    if raw_comparisons:
        if not isinstance(raw_comparisons, list):
            raise ValueError("Manifest 'comparisons' must be a list")
        for item in raw_comparisons:
            if not isinstance(item, dict):
                raise ValueError("Each comparison must be an object")
            baseline = str(
                item.get(
                    "baseline",
                    item.get("baseline_method", item.get("control", "")),
                )
            ).strip()
            candidate = str(
                item.get("candidate", item.get("candidate_method", ""))
            ).strip()
            name = str(
                item.get("name", item.get("question", f"{candidate}_vs_{baseline}"))
            ).strip()
            if not baseline or not candidate:
                raise ValueError("Each comparison needs baseline and candidate")
            comparisons.append(ComparisonSpec(name, baseline, candidate))
    else:
        comparisons = [ComparisonSpec(*values) for values in DEFAULT_COMPARISONS]

    available_codes = {_method_code(spec.method) for spec in specs}
    cleaned_comparisons: List[ComparisonSpec] = []
    for item in comparisons:
        if item.name == "gate_capacity":
            if "B2-constant" in available_codes:
                item = ComparisonSpec("gate_capacity", "B2-constant", "B5")
            else:
                item = ComparisonSpec("gate_capacity_confounded", "B2", "B5")
        elif item.name == "gate_capacity_static_scale_confounded":
            item = ComparisonSpec("gate_capacity_confounded", "B2", "B5")
        cleaned_comparisons.append(item)
    comparisons = []
    seen_comparisons: set[Tuple[str, str, str]] = set()
    for item in cleaned_comparisons:
        key = (
            item.name,
            _method_code(item.baseline),
            _method_code(item.candidate),
        )
        if key not in seen_comparisons:
            comparisons.append(item)
            seen_comparisons.add(key)

    required_gate_comparisons = (
        ComparisonSpec("full_over_hard_span", "B2", "B6"),
        ComparisonSpec("full_over_gate_only", "B5", "B6"),
    )
    existing_method_pairs = {
        (_method_code(item.baseline), _method_code(item.candidate))
        for item in comparisons
    }
    for item in required_gate_comparisons:
        method_pair = (_method_code(item.baseline), _method_code(item.candidate))
        if method_pair not in existing_method_pairs:
            comparisons.append(item)
            existing_method_pairs.add(method_pair)

    specs.sort(key=lambda item: (item.pair, item.method, item.seed, item.task))
    return specs, comparisons, default_receiver_method, metadata


def _method_code(method: str) -> str:
    normalized = method.strip().lower().replace("_", "-")
    if normalized.startswith("b2-constant") or normalized.startswith("b2 constant"):
        return "B2-constant"
    if normalized.startswith("b6-constant") or normalized.startswith("b6 constant"):
        return "B6-constant"
    if normalized.startswith("b6-shuffle") or normalized.startswith("b6 shuffle"):
        return "B6-shuffle"
    match = re.match(r"b([0-6])(?:$|[^0-9])", normalized)
    if match:
        return f"B{match.group(1)}"
    if normalized in {"receiver", "receiver-only", "receiver only"}:
        return "B0"
    return method


def _method_matches(actual: str, requested: str) -> bool:
    return actual.casefold() == requested.casefold() or _method_code(
        actual
    ).casefold() == (_method_code(requested).casefold())


def _load_runs(specs: Sequence[RunSpec]) -> List[RunData]:
    cache: Dict[Path, Tuple[Dict[SampleKey, Sample], set[str]]] = {}
    runs: List[RunData] = []
    for spec in specs:
        if not spec.csv_path.is_file():
            raise FileNotFoundError(spec.csv_path)
        if spec.csv_path not in cache:
            cache[spec.csv_path] = _load_samples(spec.csv_path)
        samples, fieldnames = cache[spec.csv_path]
        runs.append(RunData(spec=spec, samples=samples, fieldnames=fieldnames))
    return runs


def _receiver_samples_from_csv(path: Path) -> Dict[SampleKey, Sample]:
    if not path.is_file():
        raise FileNotFoundError(path)
    samples, _ = _load_samples(path)
    return samples


def _find_receiver_run(
    run: RunData,
    runs: Sequence[RunData],
    default_receiver_method: str,
) -> RunData | None:
    requested = run.spec.receiver_method or default_receiver_method
    candidates = [
        candidate
        for candidate in runs
        if candidate.spec.task == run.spec.task
        and _method_matches(candidate.spec.method, requested)
    ]
    if not candidates:
        return None

    def rank(candidate: RunData) -> Tuple[Any, ...]:
        same_pair = candidate.spec.pair == run.spec.pair
        same_seed = candidate.spec.seed == run.spec.seed
        if same_pair and same_seed:
            reuse_rank = 0
        elif same_pair and candidate.spec.seed == 42:
            reuse_rank = 1
        elif same_pair:
            reuse_rank = 2
        elif same_seed:
            reuse_rank = 3
        elif candidate.spec.seed == 42:
            reuse_rank = 4
        else:
            reuse_rank = 5
        return (
            reuse_rank,
            abs(candidate.spec.seed - run.spec.seed),
            candidate.spec.seed,
            candidate.spec.pair,
            candidate.spec.method,
            str(candidate.spec.csv_path),
        )

    return min(candidates, key=rank)


def _attach_receiver_correctness(
    runs: Sequence[RunData], default_receiver_method: str
) -> Dict[Tuple[str, str, int, str], ReceiverAttachment]:
    receiver_cache: Dict[Path, Dict[SampleKey, Sample]] = {}
    attached: Dict[Tuple[str, str, int, str], ReceiverAttachment] = {}
    for run in runs:
        receiver_values: Dict[SampleKey, bool] = {
            key: sample.receiver_correct
            for key, sample in run.samples.items()
            if sample.receiver_correct is not None
        }
        source_kind = "row_columns" if receiver_values else "missing"
        source_pair: str | None = None
        source_method: str | None = None
        source_seed: int | None = None
        source_csv: str | None = None
        receiver_source: Dict[SampleKey, Sample] | None = None
        if run.spec.receiver_csv_path is not None:
            path = run.spec.receiver_csv_path
            if path not in receiver_cache:
                receiver_cache[path] = _receiver_samples_from_csv(path)
            receiver_source = receiver_cache[path]
            source_kind = (
                "row_columns+receiver_csv" if receiver_values else "receiver_csv"
            )
            source_csv = str(path)
        else:
            receiver_run = _find_receiver_run(run, runs, default_receiver_method)
            if receiver_run is not None:
                receiver_source = receiver_run.samples
                source_kind = (
                    "row_columns+manifest_run" if receiver_values else "manifest_run"
                )
                source_pair = receiver_run.spec.pair
                source_method = receiver_run.spec.method
                source_seed = receiver_run.spec.seed
                source_csv = str(receiver_run.spec.csv_path)
        if receiver_source is not None:
            for key in set(run.samples) & set(receiver_source):
                receiver_values.setdefault(key, receiver_source[key].correct)
        attached[(run.spec.pair, run.spec.method, run.spec.seed, run.spec.task)] = (
            ReceiverAttachment(
                values=receiver_values,
                source_kind=source_kind,
                source_pair=source_pair,
                source_method=source_method,
                source_seed=source_seed,
                source_csv=source_csv,
            )
        )
    return attached


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _task_metric_row(run: RunData, receiver: ReceiverAttachment) -> Dict[str, Any]:
    receiver_values = receiver.values
    samples = list(run.samples.values())
    total = len(samples)
    correct = sum(sample.correct for sample in samples)
    paired_keys = sorted(set(run.samples) & set(receiver_values))
    receiver_correct_count = sum(receiver_values[key] for key in paired_keys)
    receiver_wrong_count = len(paired_keys) - receiver_correct_count
    positive = sum(
        (not receiver_values[key]) and run.samples[key].correct for key in paired_keys
    )
    negative = sum(
        receiver_values[key] and (not run.samples[key].correct) for key in paired_keys
    )
    return {
        "pair": run.spec.pair,
        "method": run.spec.method,
        "method_code": _method_code(run.spec.method),
        "seed": run.spec.seed,
        "task": run.spec.task,
        "n": total,
        "correct": correct,
        "accuracy": _safe_rate(correct, total),
        "receiver_paired_n": len(paired_keys),
        "receiver_correct": receiver_correct_count,
        "receiver_accuracy": _safe_rate(receiver_correct_count, len(paired_keys)),
        "receiver_wrong": receiver_wrong_count,
        "positive_transfer_count": positive,
        "positive_transfer_rate": _safe_rate(positive, receiver_wrong_count),
        "positive_transfer_total_rate": _safe_rate(positive, len(paired_keys)),
        "negative_transfer_count": negative,
        "negative_transfer_rate": _safe_rate(negative, receiver_correct_count),
        "negative_transfer_total_rate": _safe_rate(negative, len(paired_keys)),
        "transfer_status": "ok" if len(paired_keys) == total else "partial_or_missing",
        "receiver_source_kind": receiver.source_kind,
        "receiver_source_pair": receiver.source_pair,
        "receiver_source_method": receiver.source_method,
        "receiver_source_seed": receiver.source_seed,
        "receiver_seed_reused": (
            receiver.source_seed is not None and receiver.source_seed != run.spec.seed
        ),
        "receiver_source_csv": receiver.source_csv,
        "csv": str(run.spec.csv_path),
    }


def _aggregate_metric_rows(
    task_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, int], List[Mapping[str, Any]]] = {}
    for row in task_rows:
        key = (row["pair"], row["method"], row["method_code"], int(row["seed"]))
        grouped.setdefault(key, []).append(row)

    output: List[Dict[str, Any]] = []
    for (pair, method, method_code, seed), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda item: str(item["task"]))
        total = sum(int(row["n"]) for row in rows)
        correct = sum(int(row["correct"]) for row in rows)
        paired = sum(int(row["receiver_paired_n"]) for row in rows)
        receiver_correct = sum(int(row["receiver_correct"]) for row in rows)
        receiver_wrong = sum(int(row["receiver_wrong"]) for row in rows)
        positive = sum(int(row["positive_transfer_count"]) for row in rows)
        negative = sum(int(row["negative_transfer_count"]) for row in rows)
        accuracies = [
            float(row["accuracy"]) for row in rows if row["accuracy"] is not None
        ]
        output.append(
            {
                "pair": pair,
                "method": method,
                "method_code": method_code,
                "seed": seed,
                "task_count": len(rows),
                "tasks": ",".join(str(row["task"]) for row in rows),
                "n": total,
                "correct": correct,
                "macro_mean": statistics.fmean(accuracies) if accuracies else None,
                "weighted_mean": _safe_rate(correct, total),
                "receiver_paired_n": paired,
                "receiver_accuracy": _safe_rate(receiver_correct, paired),
                "positive_transfer_count": positive,
                "positive_transfer_rate": _safe_rate(positive, receiver_wrong),
                "positive_transfer_total_rate": _safe_rate(positive, paired),
                "negative_transfer_count": negative,
                "negative_transfer_rate": _safe_rate(negative, receiver_correct),
                "negative_transfer_total_rate": _safe_rate(negative, paired),
            }
        )
    return output


def _sample_std(values: Sequence[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _seed_summary_rows(
    task_rows: Sequence[Mapping[str, Any]],
    aggregate_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    values: Dict[Tuple[str, str, str, str, str], List[Tuple[int, float]]] = {}
    for row in task_rows:
        if row["accuracy"] is not None:
            key = (
                str(row["pair"]),
                str(row["method"]),
                str(row["method_code"]),
                str(row["task"]),
                "accuracy",
            )
            values.setdefault(key, []).append(
                (int(row["seed"]), float(row["accuracy"]))
            )
    for row in aggregate_rows:
        for metric in (
            "macro_mean",
            "weighted_mean",
            "positive_transfer_rate",
            "negative_transfer_rate",
        ):
            if row[metric] is None:
                continue
            key = (
                str(row["pair"]),
                str(row["method"]),
                str(row["method_code"]),
                "__aggregate__",
                metric,
            )
            values.setdefault(key, []).append((int(row["seed"]), float(row[metric])))

    output: List[Dict[str, Any]] = []
    for (pair, method, method_code, task, metric), seed_values in sorted(
        values.items()
    ):
        seed_values.sort()
        numbers = [value for _, value in seed_values]
        output.append(
            {
                "pair": pair,
                "method": method,
                "method_code": method_code,
                "task": task,
                "metric": metric,
                "n_seeds": len(numbers),
                "mean": statistics.fmean(numbers),
                "sample_std": _sample_std(numbers),
                "seeds": ",".join(str(seed) for seed, _ in seed_values),
                "values_json": json.dumps(numbers, separators=(",", ":")),
            }
        )
    return output


def _stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a quantile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_bootstrap_ci(
    differences: Sequence[int],
    samples: int,
    confidence: float,
    seed: int,
) -> Tuple[float | None, float | None]:
    if not differences or samples <= 0:
        return None, None
    n = len(differences)
    negative = sum(value < 0 for value in differences)
    zero = sum(value == 0 for value in differences)
    positive = n - negative - zero
    probabilities = [negative / n, zero / n, positive / n]
    if np is not None:
        rng = np.random.default_rng(seed)
        counts = rng.multinomial(n, probabilities, size=samples)
        boot = (counts[:, 2] - counts[:, 0]) / n
        alpha = (1.0 - confidence) / 2.0
        try:
            low, high = np.quantile(boot, [alpha, 1.0 - alpha], method="linear")
        except TypeError:  # pragma: no cover - numpy < 1.22.
            low, high = np.quantile(boot, [alpha, 1.0 - alpha], interpolation="linear")
        return float(low), float(high)

    rng = random.Random(seed)
    boot = [statistics.fmean(rng.choices(differences, k=n)) for _ in range(samples)]
    alpha = (1.0 - confidence) / 2.0
    return _quantile(boot, alpha), _quantile(boot, 1.0 - alpha)


def _mcnemar_exact_p(improvements: int, regressions: int) -> float:
    discordant = improvements + regressions
    if discordant == 0:
        return 1.0
    lower = min(improvements, regressions)
    log_terms = [
        math.lgamma(discordant + 1)
        - math.lgamma(k + 1)
        - math.lgamma(discordant - k + 1)
        - discordant * math.log(2.0)
        for k in range(lower + 1)
    ]
    maximum = max(log_terms)
    log_tail = maximum + math.log(sum(math.exp(value - maximum) for value in log_terms))
    return min(1.0, 2.0 * math.exp(log_tail))


def _comparison_row(
    name: str,
    pair: str,
    baseline: RunData,
    candidate: RunData,
    task: str,
    baseline_samples: Mapping[SampleKey, Sample],
    candidate_samples: Mapping[SampleKey, Sample],
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
    expected_n: int | None = None,
) -> Dict[str, Any]:
    baseline_keys = set(baseline_samples)
    candidate_keys = set(candidate_samples)
    common = sorted(baseline_keys & candidate_keys)
    exact_keys = baseline_keys == candidate_keys
    expected_n_ok = expected_n is None or (
        len(baseline_samples) == len(candidate_samples) == expected_n
    )
    aggregation_eligible = exact_keys and expected_n_ok and bool(common)
    differences = (
        [
            int(candidate_samples[key].correct)
            - int(baseline_samples[key].correct)
            for key in common
        ]
        if aggregation_eligible
        else []
    )
    improvements = sum(value == 1 for value in differences)
    regressions = sum(value == -1 for value in differences)
    label = (
        f"{name}:{pair}:{baseline.spec.method}:{candidate.spec.method}:"
        f"{baseline.spec.seed}:{task}"
    )
    low, high = _paired_bootstrap_ci(
        differences,
        samples=bootstrap_samples,
        confidence=bootstrap_confidence,
        seed=_stable_seed(bootstrap_seed, label),
    )
    return {
        "comparison": name,
        "pair": pair,
        "baseline_method": baseline.spec.method,
        "baseline_code": _method_code(baseline.spec.method),
        "candidate_method": candidate.spec.method,
        "candidate_code": _method_code(candidate.spec.method),
        "seed": baseline.spec.seed,
        "task": task,
        "n_paired": len(common),
        "expected_n": expected_n,
        "missing_in_baseline": len(candidate_keys - baseline_keys),
        "missing_in_candidate": len(baseline_keys - candidate_keys),
        "baseline_accuracy": _safe_rate(
            sum(baseline_samples[key].correct for key in common), len(common)
        ),
        "candidate_accuracy": _safe_rate(
            sum(candidate_samples[key].correct for key in common), len(common)
        ),
        "delta_accuracy": statistics.fmean(differences) if differences else None,
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "bootstrap_confidence": bootstrap_confidence,
        "bootstrap_samples": bootstrap_samples,
        "improvements": improvements,
        "regressions": regressions,
        "discordant": improvements + regressions,
        "mcnemar_exact_p": (
            _mcnemar_exact_p(improvements, regressions)
            if aggregation_eligible
            else None
        ),
        "aggregation_eligible": aggregation_eligible,
        "pairing_status": (
            "ok"
            if aggregation_eligible
            else ("sample_keys_mismatch" if not exact_keys else "unexpected_n")
        ),
    }


def _paired_differences(
    baseline_samples: Mapping[SampleKey, Sample],
    candidate_samples: Mapping[SampleKey, Sample],
) -> Tuple[int, ...]:
    common = sorted(set(baseline_samples) & set(candidate_samples))
    return tuple(
        int(candidate_samples[key].correct) - int(baseline_samples[key].correct)
        for key in common
    )


def _resolve_method_run(runs: Sequence[RunData], requested: str) -> RunData | None:
    matches = [run for run in runs if _method_matches(run.spec.method, requested)]
    if len(matches) > 1:
        exact = [
            run for run in matches if run.spec.method.casefold() == requested.casefold()
        ]
        return exact[0] if len(exact) == 1 else None
    return matches[0] if matches else None


def _paired_comparison_rows(
    runs: Sequence[RunData],
    comparisons: Sequence[ComparisonSpec],
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
    expected_task_rows: Mapping[str, int] | None = None,
) -> Tuple[List[Dict[str, Any]], List[PairedCluster]]:
    expected_task_rows = dict(expected_task_rows or {})
    grouped: Dict[Tuple[str, int], List[RunData]] = {}
    for run in runs:
        grouped.setdefault((run.spec.pair, run.spec.seed), []).append(run)

    output: List[Dict[str, Any]] = []
    clusters: List[PairedCluster] = []
    for (pair, _seed), pair_runs in sorted(grouped.items()):
        by_task: Dict[str, List[RunData]] = {}
        for run in pair_runs:
            by_task.setdefault(run.spec.task, []).append(run)
        for comparison in comparisons:
            pooled_baseline: Dict[SampleKey, Sample] = {}
            pooled_candidate: Dict[SampleKey, Sample] = {}
            baseline_for_pooled: RunData | None = None
            candidate_for_pooled: RunData | None = None
            baseline_tasks: set[str] = set()
            candidate_tasks: set[str] = set()
            task_rows: list[Dict[str, Any]] = []
            for task, task_runs in sorted(by_task.items()):
                baseline = _resolve_method_run(task_runs, comparison.baseline)
                candidate = _resolve_method_run(task_runs, comparison.candidate)
                if baseline is not None:
                    baseline_tasks.add(task)
                if candidate is not None:
                    candidate_tasks.add(task)
                if baseline is None or candidate is None:
                    continue
                comparison_row = _comparison_row(
                    comparison.name,
                    pair,
                    baseline,
                    candidate,
                    task,
                    baseline.samples,
                    candidate.samples,
                    bootstrap_samples,
                    bootstrap_confidence,
                    bootstrap_seed,
                    expected_task_rows.get(task),
                )
                output.append(comparison_row)
                task_rows.append(comparison_row)
                baseline_for_pooled = baseline
                candidate_for_pooled = candidate
                if comparison_row["aggregation_eligible"]:
                    for key, sample in baseline.samples.items():
                        pooled_baseline[(task, *key)] = sample
                    for key, sample in candidate.samples.items():
                        pooled_candidate[(task, *key)] = sample

            required_tasks = (
                set(expected_task_rows)
                if expected_task_rows
                else baseline_tasks | candidate_tasks
            )
            tasks_complete = (
                bool(required_tasks)
                and baseline_tasks == required_tasks
                and candidate_tasks == required_tasks
                and len(task_rows) == len(required_tasks)
                and all(bool(row["aggregation_eligible"]) for row in task_rows)
            )
            if (
                tasks_complete
                and baseline_for_pooled is not None
                and candidate_for_pooled is not None
            ):
                output.append(
                    _comparison_row(
                        comparison.name,
                        pair,
                        baseline_for_pooled,
                        candidate_for_pooled,
                        "__pooled__",
                        pooled_baseline,
                        pooled_candidate,
                        bootstrap_samples,
                        bootstrap_confidence,
                        bootstrap_seed,
                        sum(expected_task_rows.values())
                        if expected_task_rows
                        else len(pooled_baseline),
                    )
                )
                clusters.append(
                    PairedCluster(
                        comparison=comparison.name,
                        pair=pair,
                        seed=baseline_for_pooled.spec.seed,
                        baseline_method=baseline_for_pooled.spec.method,
                        candidate_method=candidate_for_pooled.spec.method,
                        differences=_paired_differences(
                            pooled_baseline, pooled_candidate
                        ),
                    )
                )
    output.sort(
        key=lambda row: (
            str(row["pair"]),
            str(row["comparison"]),
            int(row["seed"]),
            str(row["task"]),
        )
    )
    clusters.sort(key=lambda item: (item.comparison, item.pair, item.seed))
    return output, clusters


def _cluster_bootstrap_ci(
    clusters: Mapping[str, Sequence[int]],
    samples: int,
    confidence: float,
    seed: int,
) -> Tuple[float | None, float | None, float | None]:
    clean = [
        (name, tuple(values)) for name, values in sorted(clusters.items()) if values
    ]
    if not clean:
        return None, None, None
    point = statistics.fmean(statistics.fmean(values) for _, values in clean)
    cluster_count = len(clean)
    alpha = (1.0 - confidence) / 2.0

    if np is not None:
        rng = np.random.default_rng(seed)
        selected = rng.integers(0, cluster_count, size=(samples, cluster_count))
        bootstrap_sums = np.zeros(samples, dtype=float)
        for slot in range(cluster_count):
            selected_for_slot = selected[:, slot]
            for cluster_index, (_name, values) in enumerate(clean):
                mask = selected_for_slot == cluster_index
                count = int(mask.sum())
                if count == 0:
                    continue
                n = len(values)
                negative = sum(value < 0 for value in values)
                zero = sum(value == 0 for value in values)
                positive = n - negative - zero
                draws = rng.multinomial(
                    n,
                    [negative / n, zero / n, positive / n],
                    size=count,
                )
                bootstrap_sums[mask] += (draws[:, 2] - draws[:, 0]) / n
        boot = bootstrap_sums / cluster_count
        try:
            low, high = np.quantile(boot, [alpha, 1.0 - alpha], method="linear")
        except TypeError:  # pragma: no cover - numpy < 1.22.
            low, high = np.quantile(boot, [alpha, 1.0 - alpha], interpolation="linear")
        return point, float(low), float(high)

    rng = random.Random(seed)
    boot: List[float] = []
    for _ in range(samples):
        selected_clusters = rng.choices(clean, k=cluster_count)
        cluster_means = [
            statistics.fmean(rng.choices(values, k=len(values)))
            for _name, values in selected_clusters
        ]
        boot.append(statistics.fmean(cluster_means))
    return point, _quantile(boot, alpha), _quantile(boot, 1.0 - alpha)


def _hierarchical_pair_seed_bootstrap_ci(
    source_clusters: Sequence[PairedCluster],
    samples: int,
    confidence: float,
    seed: int,
) -> Tuple[float | None, float | None, float | None]:
    by_pair: Dict[str, List[Tuple[int, ...]]] = {}
    for item in source_clusters:
        if item.differences:
            by_pair.setdefault(item.pair, []).append(tuple(item.differences))
    clean = [
        (pair, tuple(seed_clusters))
        for pair, seed_clusters in sorted(by_pair.items())
        if seed_clusters
    ]
    if not clean:
        return None, None, None
    point = statistics.fmean(
        statistics.fmean(statistics.fmean(values) for values in seed_clusters)
        for _pair, seed_clusters in clean
    )
    alpha = (1.0 - confidence) / 2.0
    pair_count = len(clean)

    def draw_example_mean(values: Sequence[int], rng: Any) -> float:
        n = len(values)
        negative = sum(value < 0 for value in values)
        zero = sum(value == 0 for value in values)
        positive = n - negative - zero
        if np is not None:
            counts = rng.multinomial(n, [negative / n, zero / n, positive / n])
            return float(counts[2] - counts[0]) / n
        return statistics.fmean(rng.choices(values, k=n))

    if np is not None:
        rng: Any = np.random.default_rng(seed)
        boot: List[float] = []
        for _ in range(samples):
            selected_pairs = rng.integers(0, pair_count, size=pair_count)
            pair_means: List[float] = []
            for pair_index in selected_pairs:
                seed_clusters = clean[int(pair_index)][1]
                seed_count = len(seed_clusters)
                selected_seeds = rng.integers(0, seed_count, size=seed_count)
                pair_means.append(
                    statistics.fmean(
                        draw_example_mean(seed_clusters[int(seed_index)], rng)
                        for seed_index in selected_seeds
                    )
                )
            boot.append(statistics.fmean(pair_means))
    else:  # pragma: no cover - production/test environments provide numpy.
        rng = random.Random(seed)
        boot = []
        for _ in range(samples):
            selected_pairs = rng.choices(clean, k=pair_count)
            pair_means = []
            for _pair, seed_clusters in selected_pairs:
                selected_seeds = rng.choices(seed_clusters, k=len(seed_clusters))
                pair_means.append(
                    statistics.fmean(
                        draw_example_mean(seed_cluster, rng)
                        for seed_cluster in selected_seeds
                    )
                )
            boot.append(statistics.fmean(pair_means))
    return point, _quantile(boot, alpha), _quantile(boot, 1.0 - alpha)


def _aggregate_comparison_row(
    *,
    comparison: str,
    aggregation_level: str,
    cluster_unit: str,
    pair: str,
    clusters: Mapping[str, Sequence[int]],
    source_clusters: Sequence[PairedCluster],
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    label = f"cluster:{comparison}:{aggregation_level}:{cluster_unit}:{pair}"
    hierarchical = aggregation_level == "across_pairs" and cluster_unit == "pair"
    if hierarchical:
        point, low, high = _hierarchical_pair_seed_bootstrap_ci(
            source_clusters,
            samples=bootstrap_samples,
            confidence=bootstrap_confidence,
            seed=_stable_seed(bootstrap_seed, label),
        )
    else:
        point, low, high = _cluster_bootstrap_ci(
            clusters,
            samples=bootstrap_samples,
            confidence=bootstrap_confidence,
            seed=_stable_seed(bootstrap_seed, label),
        )
    pooled = [value for values in clusters.values() for value in values]
    improvements = sum(value == 1 for value in pooled)
    regressions = sum(value == -1 for value in pooled)
    pair_seed_means: Dict[str, List[float]] = {}
    for item in source_clusters:
        if item.differences:
            pair_seed_means.setdefault(item.pair, []).append(
                statistics.fmean(item.differences)
            )
    pair_deltas = {
        pair_name: statistics.fmean(values)
        for pair_name, values in sorted(pair_seed_means.items())
    }
    baseline_methods = sorted({item.baseline_method for item in source_clusters})
    candidate_methods = sorted({item.candidate_method for item in source_clusters})
    return {
        "comparison": comparison,
        "aggregation_level": aggregation_level,
        "cluster_unit": cluster_unit,
        "delta_estimand": "equal_weighted_cluster_mean_accuracy_delta",
        "pair": pair,
        "seed": "all",
        "task": "__pooled__",
        "baseline_method": ",".join(baseline_methods),
        "baseline_code": (
            _method_code(baseline_methods[0]) if baseline_methods else None
        ),
        "candidate_method": ",".join(candidate_methods),
        "candidate_code": (
            _method_code(candidate_methods[0]) if candidate_methods else None
        ),
        "n_clusters": len(clusters),
        "cluster_ids_json": json.dumps(sorted(clusters), separators=(",", ":")),
        "n_pairs": len(pair_deltas),
        "n_seeds": len({item.seed for item in source_clusters}),
        "n_pair_seed_runs": len(source_clusters),
        "n_paired": len(pooled),
        "delta_accuracy": point,
        "pooled_accuracy_delta": statistics.fmean(pooled) if pooled else None,
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "bootstrap_confidence": bootstrap_confidence,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_level": (
            "pairs_then_seeds_then_paired_examples"
            if hierarchical
            else "clusters_then_paired_examples"
        ),
        "bootstrap_status": "ok" if len(clusters) >= 2 else "single_cluster",
        "ci_excludes_zero_positive": low is not None and low > 0.0,
        "ci_excludes_zero_negative": high is not None and high < 0.0,
        "improvements": improvements,
        "regressions": regressions,
        "discordant": improvements + regressions,
        "aggregate_mcnemar_exact_p": _mcnemar_exact_p(improvements, regressions),
        "aggregate_mcnemar_scope": "pooled paired predictions; not cluster-adjusted",
        "positive_pair_count": sum(value > 0.0 for value in pair_deltas.values()),
        "pair_deltas_json": json.dumps(
            pair_deltas, sort_keys=True, separators=(",", ":")
        ),
    }


def _clustered_comparison_rows(
    paired_clusters: Sequence[PairedCluster],
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> List[Dict[str, Any]]:
    by_comparison: Dict[str, List[PairedCluster]] = {}
    for item in paired_clusters:
        by_comparison.setdefault(item.comparison, []).append(item)

    output: List[Dict[str, Any]] = []
    for comparison, comparison_clusters in sorted(by_comparison.items()):
        by_pair: Dict[str, List[PairedCluster]] = {}
        for item in comparison_clusters:
            by_pair.setdefault(item.pair, []).append(item)
        for pair, pair_clusters in sorted(by_pair.items()):
            clusters = {
                f"{item.pair}/seed_{item.seed}": item.differences
                for item in pair_clusters
            }
            output.append(
                _aggregate_comparison_row(
                    comparison=comparison,
                    aggregation_level="across_seeds_within_pair",
                    cluster_unit="pair_seed",
                    pair=pair,
                    clusters=clusters,
                    source_clusters=pair_clusters,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_confidence=bootstrap_confidence,
                    bootstrap_seed=bootstrap_seed,
                )
            )

        pair_seed_clusters = {
            f"{item.pair}/seed_{item.seed}": item.differences
            for item in comparison_clusters
        }
        output.append(
            _aggregate_comparison_row(
                comparison=comparison,
                aggregation_level="across_pairs_and_seeds",
                cluster_unit="pair_seed",
                pair="__all__",
                clusters=pair_seed_clusters,
                source_clusters=comparison_clusters,
                bootstrap_samples=bootstrap_samples,
                bootstrap_confidence=bootstrap_confidence,
                bootstrap_seed=bootstrap_seed,
            )
        )
        pair_clusters = {
            pair: tuple(
                value
                for item in sorted(items, key=lambda cluster: cluster.seed)
                for value in item.differences
            )
            for pair, items in sorted(by_pair.items())
        }
        output.append(
            _aggregate_comparison_row(
                comparison=comparison,
                aggregation_level="across_pairs",
                cluster_unit="pair",
                pair="__all__",
                clusters=pair_clusters,
                source_clusters=comparison_clusters,
                bootstrap_samples=bootstrap_samples,
                bootstrap_confidence=bootstrap_confidence,
                bootstrap_seed=bootstrap_seed,
            )
        )
    output.sort(
        key=lambda row: (
            str(row["comparison"]),
            str(row["aggregation_level"]),
            str(row["cluster_unit"]),
            str(row["pair"]),
        )
    )
    return output


def _report_contract(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    raw = metadata.get("report_contract", {})
    if not isinstance(raw, Mapping):
        raw = {}
    expected_raw = raw.get("expected_task_rows", {})
    expected_task_rows: Dict[str, int] = {}
    if isinstance(expected_raw, Mapping):
        for task, value in expected_raw.items():
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            if count > 0:
                expected_task_rows[str(task)] = count
    pairs_raw = raw.get("required_pairs", [])
    seeds_raw = raw.get("required_seeds", DEFAULT_FINAL_GATE_SEEDS)
    required_pairs = (
        tuple(str(item) for item in pairs_raw if str(item).strip())
        if isinstance(pairs_raw, Sequence) and not isinstance(pairs_raw, (str, bytes))
        else ()
    )
    required_seeds = (
        tuple(int(item) for item in seeds_raw)
        if isinstance(seeds_raw, Sequence) and not isinstance(seeds_raw, (str, bytes))
        else DEFAULT_FINAL_GATE_SEEDS
    )
    conditional_complete = bool(
        raw.get("conditional_complete", metadata.get("conditional_complete", False))
    )
    if conditional_complete and not expected_task_rows:
        expected_task_rows = dict(CANONICAL_TASK_EXPECTED_ROWS)
    return {
        "expected_task_rows": expected_task_rows,
        "required_pairs": required_pairs,
        "required_seeds": required_seeds,
        "conditional_complete": conditional_complete,
    }


def _final_gate_rows(
    clustered_rows: Sequence[Mapping[str, Any]],
    paired_clusters: Sequence[PairedCluster],
    contract: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    targets = (("B2", "B6"), ("B5", "B6"))
    required_seeds = tuple(
        int(item) for item in contract.get("required_seeds", DEFAULT_FINAL_GATE_SEEDS)
    )
    configured_pairs = tuple(str(item) for item in contract.get("required_pairs", ()))
    conditional_complete = bool(contract.get("conditional_complete", False))
    target_clusters = [
        item
        for item in paired_clusters
        if (
            _method_code(item.baseline_method),
            _method_code(item.candidate_method),
        )
        in targets
    ]
    inferred_pairs = tuple(sorted({item.pair for item in target_clusters}))
    required_pairs = configured_pairs or (
        inferred_pairs if conditional_complete else ()
    )
    coverage_registered = (
        len(required_pairs) == 4
        and len(set(required_pairs)) == 4
        and required_seeds == DEFAULT_FINAL_GATE_SEEDS
        and len(contract.get("expected_task_rows", {})) == 3
        and (
            bool(contract.get("expected_task_rows"))
            or conditional_complete
        )
    )
    output: List[Dict[str, Any]] = []
    for baseline_code, candidate_code in targets:
        matches = [
            row
            for row in clustered_rows
            if row["aggregation_level"] == "across_pairs"
            and row["cluster_unit"] == "pair"
            and row["baseline_code"] == baseline_code
            and row["candidate_code"] == candidate_code
        ]
        row = matches[0] if len(matches) == 1 else None
        matching_clusters = [
            item
            for item in target_clusters
            if _method_code(item.baseline_method) == baseline_code
            and _method_code(item.candidate_method) == candidate_code
        ]
        available_pair_seeds = {(item.pair, item.seed) for item in matching_clusters}
        required_pair_seeds = {
            (pair, seed) for pair in required_pairs for seed in required_seeds
        }
        complete_pairs = {
            pair
            for pair in required_pairs
            if all((pair, seed) in available_pair_seeds for seed in required_seeds)
        }
        n_pairs = len(complete_pairs)
        positive_pairs = int(row["positive_pair_count"]) if row is not None else 0
        coverage_complete = (
            coverage_registered and available_pair_seeds == required_pair_seeds
        )
        positive_pairs_pass = coverage_complete and positive_pairs >= 3
        ci_pass = bool(row is not None and row["ci_excludes_zero_positive"])
        gate_pass = coverage_complete and positive_pairs_pass and ci_pass
        status = (
            "pass" if gate_pass else ("fail" if coverage_complete else "incomplete")
        )
        output.append(
            {
                "contrast": f"{candidate_code}_vs_{baseline_code}",
                "candidate_code": candidate_code,
                "baseline_code": baseline_code,
                "required_pair_count": 4,
                "required_pairs_json": json.dumps(required_pairs, separators=(",", ":")),
                "required_seeds_json": json.dumps(required_seeds, separators=(",", ":")),
                "required_positive_pair_count": 3,
                "available_pair_count": n_pairs,
                "available_pair_seed_count": len(available_pair_seeds),
                "required_pair_seed_count": len(required_pair_seeds),
                "missing_pair_seeds_json": json.dumps(
                    sorted(required_pair_seeds - available_pair_seeds),
                    separators=(",", ":"),
                ),
                "unexpected_pair_seeds_json": json.dumps(
                    sorted(available_pair_seeds - required_pair_seeds),
                    separators=(",", ":"),
                ),
                "coverage_registered": coverage_registered,
                "conditional_complete": conditional_complete,
                "positive_pair_count": positive_pairs,
                "coverage_complete": coverage_complete,
                "positive_pairs_pass": positive_pairs_pass,
                "aggregate_pair_cluster_ci_low": (
                    row["bootstrap_ci_low"] if row is not None else None
                ),
                "aggregate_pair_cluster_ci_high": (
                    row["bootstrap_ci_high"] if row is not None else None
                ),
                "aggregate_ci_positive": ci_pass,
                "aggregate_mcnemar_exact_p": (
                    row["aggregate_mcnemar_exact_p"] if row is not None else None
                ),
                "gate_pass": gate_pass,
                "status": status,
            }
        )
    component_statuses = [str(row["status"]) for row in output]
    combined_status = (
        "pass"
        if all(status == "pass" for status in component_statuses)
        else (
            "incomplete"
            if any(status == "incomplete" for status in component_statuses)
            else "fail"
        )
    )
    output.append(
        {
            "contrast": "combined_B6_vs_B2_and_B5",
            "candidate_code": "B6",
            "baseline_code": "B2,B5",
            "required_pair_count": 4,
            "required_pairs_json": json.dumps(required_pairs, separators=(",", ":")),
            "required_seeds_json": json.dumps(required_seeds, separators=(",", ":")),
            "required_positive_pair_count": 3,
            "available_pair_count": min(
                int(row["available_pair_count"]) for row in output
            ),
            "positive_pair_count": None,
            "available_pair_seed_count": min(
                int(row["available_pair_seed_count"]) for row in output
            ),
            "required_pair_seed_count": len(required_pairs) * len(required_seeds),
            "missing_pair_seeds_json": None,
            "unexpected_pair_seeds_json": None,
            "coverage_registered": coverage_registered,
            "conditional_complete": conditional_complete,
            "coverage_complete": all(bool(row["coverage_complete"]) for row in output),
            "positive_pairs_pass": all(
                bool(row["positive_pairs_pass"]) for row in output
            ),
            "aggregate_pair_cluster_ci_low": None,
            "aggregate_pair_cluster_ci_high": None,
            "aggregate_ci_positive": all(
                bool(row["aggregate_ci_positive"]) for row in output
            ),
            "aggregate_mcnemar_exact_p": None,
            "gate_pass": combined_status == "pass",
            "status": combined_status,
        }
    )
    return output


def _parse_number(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _bucket_numeric(field: str, value: float) -> str:
    if field == "candidate_count":
        if value <= 0:
            return "0"
        if value < 1.5:
            return "1"
        if value < 2.5:
            return "2"
        if value < 3.5:
            return "3"
        return "4+"
    if field == "alignment_entropy":
        if abs(value) <= 1e-12:
            return "0"
        if value <= 0.25:
            return "(0,0.25]"
        if value <= 0.5:
            return "(0.25,0.5]"
        if value <= 0.75:
            return "(0.5,0.75]"
        return ">0.75"
    if field == "boundary_mismatch":
        if abs(value) <= 1e-12:
            return "0"
        if value <= 1.0:
            return "(0,1]"
        if value <= 2.0:
            return "(1,2]"
        return ">2"
    if field in {"confidence", "gate"}:
        if value < 0.0:
            return "<0"
        if value < 0.25:
            return "[0,0.25)"
        if value < 0.5:
            return "[0.25,0.5)"
        if value < 0.75:
            return "[0.5,0.75)"
        if value < 1.0:
            return "[0.75,1)"
        if abs(value - 1.0) <= 1e-12:
            return "1"
        return ">1"
    return str(value)


def _diagnostic_bucket(
    sample: Sample, field: str, fieldnames: set[str]
) -> Tuple[str, str]:
    if field == "alignment_bucket":
        value = sample.diagnostics[field]
        if value:
            return value, "observed"
        if "candidate_count" in fieldnames:
            count = _parse_number(sample.diagnostics["candidate_count"])
            if count is not None:
                return ("1-to-1" if count <= 1 else "one-to-many"), "derived"
        return "missing", "missing"
    if field not in fieldnames:
        return "missing", "missing"
    value = _parse_number(sample.diagnostics[field])
    if value is None:
        return "missing", "observed_with_missing_values"
    return _bucket_numeric(field, value), "observed"


def _bucket_metric_rows(
    runs: Sequence[RunData],
    receiver_by_run: Mapping[Tuple[str, str, int, str], ReceiverAttachment],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for run in runs:
        run_key = (run.spec.pair, run.spec.method, run.spec.seed, run.spec.task)
        receivers = receiver_by_run[run_key].values
        for field in DIAGNOSTIC_FIELDS:
            grouped: Dict[Tuple[str, str], List[Sample]] = {}
            for sample in run.samples.values():
                bucket, status = _diagnostic_bucket(sample, field, run.fieldnames)
                grouped.setdefault((bucket, status), []).append(sample)
            if not grouped:
                grouped[("missing", "missing")] = []
            for (bucket, status), samples in sorted(grouped.items()):
                keys = [sample.key for sample in samples if sample.key in receivers]
                receiver_correct = sum(receivers[key] for key in keys)
                receiver_wrong = len(keys) - receiver_correct
                positive = sum(
                    (not receivers[key]) and run.samples[key].correct for key in keys
                )
                negative = sum(
                    receivers[key] and (not run.samples[key].correct) for key in keys
                )
                output.append(
                    {
                        "pair": run.spec.pair,
                        "method": run.spec.method,
                        "method_code": _method_code(run.spec.method),
                        "seed": run.spec.seed,
                        "task": run.spec.task,
                        "field": field,
                        "bucket": bucket,
                        "status": status,
                        "n": len(samples),
                        "correct": sum(sample.correct for sample in samples),
                        "accuracy": _safe_rate(
                            sum(sample.correct for sample in samples), len(samples)
                        ),
                        "receiver_paired_n": len(keys),
                        "positive_transfer_count": positive,
                        "positive_transfer_rate": _safe_rate(positive, receiver_wrong),
                        "negative_transfer_count": negative,
                        "negative_transfer_rate": _safe_rate(
                            negative, receiver_correct
                        ),
                    }
                )
    output.sort(
        key=lambda row: (
            str(row["pair"]),
            str(row["method"]),
            int(row["seed"]),
            str(row["task"]),
            str(row["field"]),
            str(row["bucket"]),
        )
    )
    return output


def _paired_bucket_gain_rows_for_samples(
    *,
    comparison: ComparisonSpec,
    pair: str,
    seed: int,
    task: str,
    baseline: RunData,
    candidate: RunData,
    baseline_samples: Mapping[SampleKey, Sample],
    candidate_samples: Mapping[SampleKey, Sample],
    candidate_fieldnames: set[str],
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> List[Dict[str, Any]]:
    keys = sorted(candidate_samples)
    output: List[Dict[str, Any]] = []
    for field in DIAGNOSTIC_FIELDS:
        grouped: Dict[Tuple[str, str], List[SampleKey]] = {}
        for key in keys:
            bucket, status = _diagnostic_bucket(
                candidate_samples[key], field, candidate_fieldnames
            )
            grouped.setdefault((bucket, status), []).append(key)
        for (bucket, diagnostic_status), bucket_keys in sorted(grouped.items()):
            differences = [
                int(candidate_samples[key].correct)
                - int(baseline_samples[key].correct)
                for key in bucket_keys
            ]
            improvements = sum(value == 1 for value in differences)
            regressions = sum(value == -1 for value in differences)
            label = (
                f"bucket:{comparison.name}:{pair}:{seed}:{task}:{field}:{bucket}"
            )
            low, high = _paired_bootstrap_ci(
                differences,
                samples=bootstrap_samples,
                confidence=bootstrap_confidence,
                seed=_stable_seed(bootstrap_seed, label),
            )
            output.append(
                {
                    "comparison": comparison.name,
                    "pair": pair,
                    "seed": seed,
                    "task": task,
                    "baseline_method": baseline.spec.method,
                    "baseline_code": _method_code(baseline.spec.method),
                    "candidate_method": candidate.spec.method,
                    "candidate_code": _method_code(candidate.spec.method),
                    "bucket_source_method": candidate.spec.method,
                    "bucket_source": "candidate_fixed_sample_keys",
                    "field": field,
                    "bucket": bucket,
                    "diagnostic_status": diagnostic_status,
                    "n_paired": len(bucket_keys),
                    "baseline_accuracy": _safe_rate(
                        sum(baseline_samples[key].correct for key in bucket_keys),
                        len(bucket_keys),
                    ),
                    "candidate_accuracy": _safe_rate(
                        sum(candidate_samples[key].correct for key in bucket_keys),
                        len(bucket_keys),
                    ),
                    "delta_accuracy": statistics.fmean(differences),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "bootstrap_confidence": bootstrap_confidence,
                    "bootstrap_samples": bootstrap_samples,
                    "improvements": improvements,
                    "regressions": regressions,
                    "mcnemar_exact_p": _mcnemar_exact_p(
                        improvements, regressions
                    ),
                }
            )
    return output


def _paired_bucket_gain_rows(
    runs: Sequence[RunData],
    comparisons: Sequence[ComparisonSpec],
    expected_task_rows: Mapping[str, int],
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[RunData]] = {}
    for run in runs:
        grouped.setdefault((run.spec.pair, run.spec.seed), []).append(run)

    output: List[Dict[str, Any]] = []
    for (pair, seed), pair_runs in sorted(grouped.items()):
        by_task: Dict[str, List[RunData]] = {}
        for run in pair_runs:
            by_task.setdefault(run.spec.task, []).append(run)
        for comparison in comparisons:
            if comparison.name not in PAIRED_BUCKET_COMPARISONS:
                continue
            baseline_tasks: set[str] = set()
            candidate_tasks: set[str] = set()
            valid_tasks: list[
                tuple[str, RunData, RunData, Mapping[SampleKey, Sample], Mapping[SampleKey, Sample]]
            ] = []
            for task, task_runs in sorted(by_task.items()):
                baseline = _resolve_method_run(task_runs, comparison.baseline)
                candidate = _resolve_method_run(task_runs, comparison.candidate)
                if baseline is not None:
                    baseline_tasks.add(task)
                if candidate is not None:
                    candidate_tasks.add(task)
                if baseline is None or candidate is None:
                    continue
                exact_keys = set(baseline.samples) == set(candidate.samples)
                expected_n = expected_task_rows.get(task)
                expected_n_ok = expected_n is None or (
                    len(baseline.samples) == len(candidate.samples) == expected_n
                )
                if not exact_keys or not expected_n_ok or not baseline.samples:
                    continue
                valid_tasks.append(
                    (task, baseline, candidate, baseline.samples, candidate.samples)
                )
                output.extend(
                    _paired_bucket_gain_rows_for_samples(
                        comparison=comparison,
                        pair=pair,
                        seed=seed,
                        task=task,
                        baseline=baseline,
                        candidate=candidate,
                        baseline_samples=baseline.samples,
                        candidate_samples=candidate.samples,
                        candidate_fieldnames=candidate.fieldnames,
                        bootstrap_samples=bootstrap_samples,
                        bootstrap_confidence=bootstrap_confidence,
                        bootstrap_seed=bootstrap_seed,
                    )
                )

            required_tasks = (
                set(expected_task_rows)
                if expected_task_rows
                else baseline_tasks | candidate_tasks
            )
            if not (
                required_tasks
                and baseline_tasks == required_tasks
                and candidate_tasks == required_tasks
                and {task for task, *_rest in valid_tasks} == required_tasks
            ):
                continue
            pooled_baseline: Dict[SampleKey, Sample] = {}
            pooled_candidate: Dict[SampleKey, Sample] = {}
            candidate_fieldnames: set[str] = set()
            for task, _baseline, candidate, baseline_samples, candidate_samples in valid_tasks:
                candidate_fieldnames.update(candidate.fieldnames)
                for key, sample in baseline_samples.items():
                    pooled_baseline[(task, *key)] = sample
                for key, sample in candidate_samples.items():
                    pooled_candidate[(task, *key)] = sample
            first_task = valid_tasks[0]
            output.extend(
                _paired_bucket_gain_rows_for_samples(
                    comparison=comparison,
                    pair=pair,
                    seed=seed,
                    task="__pooled__",
                    baseline=first_task[1],
                    candidate=first_task[2],
                    baseline_samples=pooled_baseline,
                    candidate_samples=pooled_candidate,
                    candidate_fieldnames=candidate_fieldnames,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_confidence=bootstrap_confidence,
                    bootstrap_seed=bootstrap_seed,
                )
            )
    output.sort(
        key=lambda row: (
            str(row["pair"]),
            str(row["comparison"]),
            int(row["seed"]),
            str(row["task"]),
            str(row["field"]),
            str(row["bucket"]),
        )
    )
    return output


def _pearson(
    values: Sequence[float], outcomes: Sequence[float]
) -> Tuple[float | None, str]:
    if len(values) < 3:
        return None, "insufficient_n"
    mean_x = statistics.fmean(values)
    mean_y = statistics.fmean(outcomes)
    centered_x = [value - mean_x for value in values]
    centered_y = [value - mean_y for value in outcomes]
    sum_x2 = sum(value * value for value in centered_x)
    sum_y2 = sum(value * value for value in centered_y)
    if sum_x2 <= 0:
        return None, "constant_field"
    if sum_y2 <= 0:
        return None, "constant_outcome"
    coefficient = sum(x * y for x, y in zip(centered_x, centered_y)) / math.sqrt(
        sum_x2 * sum_y2
    )
    return max(-1.0, min(1.0, coefficient)), "ok"


def _correlation_rows(
    runs: Sequence[RunData],
    receiver_by_run: Mapping[Tuple[str, str, int, str], ReceiverAttachment],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for run in runs:
        run_key = (run.spec.pair, run.spec.method, run.spec.seed, run.spec.task)
        receivers = receiver_by_run[run_key].values
        for field in NUMERIC_DIAGNOSTIC_FIELDS:
            parsed = {
                key: _parse_number(sample.diagnostics[field])
                for key, sample in run.samples.items()
            }
            observed = {
                key: value for key, value in parsed.items() if value is not None
            }
            outcomes: Dict[str, List[Tuple[float, float]]] = {
                "fused_correct": [
                    (value, float(run.samples[key].correct))
                    for key, value in observed.items()
                ],
                "positive_transfer": [
                    (value, float(run.samples[key].correct))
                    for key, value in observed.items()
                    if key in receivers and not receivers[key]
                ],
                "negative_transfer": [
                    (value, float(not run.samples[key].correct))
                    for key, value in observed.items()
                    if key in receivers and receivers[key]
                ],
            }
            for outcome, pairs in outcomes.items():
                values = [value for value, _ in pairs]
                labels = [label for _, label in pairs]
                if field not in run.fieldnames:
                    coefficient, status = None, "missing"
                else:
                    coefficient, status = _pearson(values, labels)
                output.append(
                    {
                        "pair": run.spec.pair,
                        "method": run.spec.method,
                        "method_code": _method_code(run.spec.method),
                        "seed": run.spec.seed,
                        "task": run.spec.task,
                        "field": field,
                        "outcome": outcome,
                        "n": len(pairs),
                        "pearson_r": coefficient,
                        "status": status,
                    }
                )
    output.sort(
        key=lambda row: (
            str(row["pair"]),
            str(row["method"]),
            int(row["seed"]),
            str(row["task"]),
            str(row["field"]),
            str(row["outcome"]),
        )
    )
    return output


def _gate_statistic_rows(runs: Sequence[RunData]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for run in runs:
        values = [
            value
            for value in (
                _parse_number(sample.diagnostics["gate"])
                for sample in run.samples.values()
            )
            if value is not None
        ]
        output.append(
            {
                "pair": run.spec.pair,
                "method": run.spec.method,
                "method_code": _method_code(run.spec.method),
                "seed": run.spec.seed,
                "task": run.spec.task,
                "status": (
                    "ok"
                    if values
                    else (
                        "missing"
                        if "gate" not in run.fieldnames
                        else "no_numeric_values"
                    )
                ),
                "n": len(values),
                "mean": statistics.fmean(values) if values else None,
                "sample_std": _sample_std(values),
                "minimum": min(values) if values else None,
                "maximum": max(values) if values else None,
                "saturation_low_rate": _safe_rate(
                    sum(value <= 0.05 for value in values), len(values)
                ),
                "saturation_high_rate": _safe_rate(
                    sum(value >= 0.95 for value in values), len(values)
                ),
            }
        )
    return output


def _posthoc_gate_rows(
    runs: Sequence[RunData],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    coverage: List[Dict[str, Any]] = []
    statistics_rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, int, Path]] = set()
    for run in runs:
        path = run.spec.gate_diagnostics_posthoc_path
        if path is None:
            continue
        dedupe_key = (run.spec.pair, run.spec.method, run.spec.seed, path)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        base = {
            "pair": run.spec.pair,
            "method": run.spec.method,
            "method_code": _method_code(run.spec.method),
            "seed": run.spec.seed,
            "artifact": str(path),
        }
        if not path.is_file():
            coverage.append({**base, "status": "missing"})
            continue
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            coverage.append({**base, "status": "invalid_json"})
            continue
        counts = artifact.get("counts", {})
        metadata = artifact.get("metadata", {})
        coverage.append(
            {
                **base,
                "status": artifact.get("status"),
                "processed_samples": (
                    metadata.get("processed_samples")
                    if isinstance(metadata, Mapping)
                    else None
                ),
                "examples_seen": (
                    counts.get("examples_seen") if isinstance(counts, Mapping) else None
                ),
                "examples_with_gate": (
                    counts.get("examples_with_gate")
                    if isinstance(counts, Mapping)
                    else None
                ),
                "token_head_gate_projectors": (
                    counts.get("token_head_gate_projectors")
                    if isinstance(counts, Mapping)
                    else None
                ),
                "layer_count": len(artifact.get("by_layer", [])),
                "stage_count": len(artifact.get("by_stage", [])),
                "layer_head_count": len(artifact.get("by_layer_head", [])),
                "relative_token_bin_count": len(
                    artifact.get("by_relative_token_bin", [])
                ),
            }
        )

        scopes: list[tuple[str, Sequence[Mapping[str, Any]]]] = []
        global_stats = artifact.get("global", {})
        if isinstance(global_stats, Mapping):
            scopes.append(("global", ({"scope": "global", **global_stats},)))
        for scope, key in (
            ("layer", "by_layer"),
            ("stage", "by_stage"),
            ("layer_head", "by_layer_head"),
            ("relative_token_bin", "by_relative_token_bin"),
        ):
            values = artifact.get(key, [])
            if isinstance(values, list):
                scopes.append((scope, [row for row in values if isinstance(row, Mapping)]))

        for scope, rows in scopes:
            for row in rows:
                for kv in ("combined", "key", "value"):
                    stats = row.get(kv)
                    if not isinstance(stats, Mapping):
                        continue
                    statistics_rows.append(
                        {
                            **base,
                            "scope": scope,
                            "kv": kv,
                            "layer": row.get("layer"),
                            "stage": row.get("stage"),
                            "head": row.get("head"),
                            "relative_token_bin": row.get("relative_token_bin"),
                            "count": stats.get("count"),
                            "mean": stats.get("mean"),
                            "variance": stats.get("variance"),
                            "std": stats.get("std"),
                            "minimum": stats.get("minimum"),
                            "maximum": stats.get("maximum"),
                            "saturation_low_rate": stats.get(
                                "saturation_low_rate"
                            ),
                            "saturation_high_rate": stats.get(
                                "saturation_high_rate"
                            ),
                        }
                    )
    coverage.sort(
        key=lambda row: (str(row["pair"]), str(row["method"]), int(row["seed"]))
    )
    statistics_rows.sort(
        key=lambda row: (
            str(row["pair"]),
            str(row["method"]),
            int(row["seed"]),
            str(row["scope"]),
            str(row["layer"]),
            str(row["stage"]),
            str(row["head"]),
            str(row["relative_token_bin"]),
            str(row["kv"]),
        )
    )
    return coverage, statistics_rows


def _posthoc_gate_conclusions(
    coverage: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> List[str]:
    conclusions: List[str] = []
    for item in coverage:
        identity = f"{item['pair']}/{item['method']}/seed_{item['seed']}"
        if item.get("status") != "ok":
            conclusions.append(f"[{identity}] post-hoc gate diagnostics are {item.get('status')}.")
            continue
        selected = [
            row
            for row in rows
            if row["pair"] == item["pair"]
            and row["method"] == item["method"]
            and row["seed"] == item["seed"]
        ]
        stages = {
            (str(row["stage"]), str(row["kv"])): row
            for row in selected
            if row["scope"] == "stage" and row["stage"] is not None
        }
        stage_text = ", ".join(
            f"{stage}/{kv} mean={_fmt(row['mean'])}, sat-low={_fmt(row['saturation_low_rate'])}, sat-high={_fmt(row['saturation_high_rate'])}"
            for (stage, kv), row in sorted(stages.items())
        )
        head_count = sum(row["scope"] == "layer_head" for row in selected)
        token_count = sum(row["scope"] == "relative_token_bin" for row in selected)
        conclusions.append(
            f"[{identity}] K/V stage statistics: {stage_text or 'missing'}; "
            f"head rows={head_count}, relative-token rows={token_count}."
        )
    return conclusions


def _component_contribution_rows(
    comparison_rows: Sequence[Mapping[str, Any]],
    clustered_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    pooled = [
        row
        for row in comparison_rows
        if row["task"] == "__pooled__" and row["delta_accuracy"] is not None
    ]
    output: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str, str, str], List[Mapping[str, Any]]] = {}
    within_pair_clustered = {
        (str(row["comparison"]), str(row["pair"])): row
        for row in clustered_rows
        if row["aggregation_level"] == "across_seeds_within_pair"
        and row["cluster_unit"] == "pair_seed"
    }
    for row in pooled:
        key = (
            str(row["pair"]),
            str(row["comparison"]),
            str(row["baseline_method"]),
            str(row["candidate_method"]),
        )
        grouped.setdefault(key, []).append(row)
        output.append(
            {
                "pair": row["pair"],
                "component": row["comparison"],
                "baseline_method": row["baseline_method"],
                "candidate_method": row["candidate_method"],
                "seed": row["seed"],
                "n_seeds": 1,
                "n_paired": row["n_paired"],
                "delta_accuracy_mean": row["delta_accuracy"],
                "delta_accuracy_sample_std": None,
                "bootstrap_ci_low": row["bootstrap_ci_low"],
                "bootstrap_ci_high": row["bootstrap_ci_high"],
                "mcnemar_exact_p": row["mcnemar_exact_p"],
                "positive_seed_count": int(float(row["delta_accuracy"]) > 0),
            }
        )
    for (pair, name, baseline, candidate), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["seed"]))
        values = [float(row["delta_accuracy"]) for row in rows]
        clustered = within_pair_clustered.get((name, pair))
        output.append(
            {
                "pair": pair,
                "component": name,
                "baseline_method": baseline,
                "candidate_method": candidate,
                "seed": "all",
                "n_seeds": len(values),
                "n_paired": sum(int(row["n_paired"]) for row in rows),
                "delta_accuracy_mean": statistics.fmean(values),
                "delta_accuracy_sample_std": _sample_std(values),
                "bootstrap_ci_low": (
                    clustered["bootstrap_ci_low"] if clustered is not None else None
                ),
                "bootstrap_ci_high": (
                    clustered["bootstrap_ci_high"] if clustered is not None else None
                ),
                "bootstrap_level": (
                    "pair_seed_cluster_within_pair" if clustered is not None else None
                ),
                "mcnemar_exact_p": (
                    clustered["aggregate_mcnemar_exact_p"]
                    if clustered is not None
                    else None
                ),
                "positive_seed_count": sum(value > 0 for value in values),
            }
        )
    for row in clustered_rows:
        if not (
            row["aggregation_level"] == "across_pairs" and row["cluster_unit"] == "pair"
        ):
            continue
        output.append(
            {
                "pair": "__all__",
                "component": row["comparison"],
                "baseline_method": row["baseline_method"],
                "candidate_method": row["candidate_method"],
                "seed": "all",
                "n_seeds": row["n_seeds"],
                "n_paired": row["n_paired"],
                "delta_accuracy_mean": row["delta_accuracy"],
                "delta_accuracy_sample_std": None,
                "bootstrap_ci_low": row["bootstrap_ci_low"],
                "bootstrap_ci_high": row["bootstrap_ci_high"],
                "bootstrap_level": "pair_cluster_across_pairs",
                "mcnemar_exact_p": row["aggregate_mcnemar_exact_p"],
                "positive_seed_count": None,
                "positive_pair_count": row["positive_pair_count"],
            }
        )
    output.sort(
        key=lambda row: (str(row["pair"]), str(row["component"]), str(row["seed"]))
    )
    return output


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    output = [
        "| " + " | ".join(clean(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend(
        "| " + " | ".join(clean(value) for value in row) + " |" for row in rows
    )
    return "\n".join(output)


def _conclusion_for_component(
    component: str,
    value: float,
    std: float | None,
    ci_low: float | None,
    ci_high: float | None,
    equivalence_margin: float,
) -> str:
    uncertainty = f" (seed sample std={std:.4f})" if std is not None else ""
    ci_text = (
        f", cluster bootstrap CI=[{ci_low:+.4f}, {ci_high:+.4f}]"
        if ci_low is not None and ci_high is not None
        else ", cluster bootstrap CI=missing"
    )
    positive = ci_low is not None and ci_low > 0.0
    negative = ci_high is not None and ci_high < 0.0
    if component == "soft_candidates":
        if positive:
            verdict = "supports a contribution from multiple source candidates."
        elif negative:
            verdict = "shows a significant disadvantage for multiple source candidates."
        else:
            verdict = (
                "is inconclusive because the corresponding CI crosses or touches zero."
            )
        return f"B3-B2={value:+.4f}{uncertainty}{ci_text}: {verdict}"
    if component == "static_entropy":
        if positive:
            verdict = "supports an independent static entropy-confidence contribution."
        elif negative:
            verdict = "shows a significant disadvantage from static entropy confidence."
        else:
            verdict = (
                "is inconclusive because the corresponding CI crosses or touches zero."
            )
        return f"B4-B3={value:+.4f}{uncertainty}{ci_text}: {verdict}"
    if component == "gate_capacity":
        if positive:
            verdict = "supports a gate-capacity contribution under matched constant confidence."
        elif negative:
            verdict = (
                "shows a significant gate-only disadvantage under matched confidence."
            )
        else:
            verdict = (
                "is inconclusive because the corresponding CI crosses or touches zero."
            )
        return f"B5-B2-constant={value:+.4f}{uncertainty}{ci_text}: {verdict}"
    if component == "gate_capacity_confounded":
        if positive:
            verdict = (
                "is positive, but the B2-to-B5 contrast is confounded by confidence "
                "mismatch and cannot isolate gate capacity."
            )
        elif negative:
            verdict = (
                "is negative, but the B2-to-B5 contrast remains confounded by "
                "confidence mismatch."
            )
        else:
            verdict = (
                "is inconclusive and additionally confounded by confidence mismatch; "
                "B2-constant is required for a clean gate claim."
            )
        return f"B5-B2={value:+.4f}{uncertainty}{ci_text}: {verdict}"
    if component in {"entropy_values", "entropy_position"}:
        label = (
            "constant-confidence"
            if component == "entropy_values"
            else "shuffled-entropy"
        )
        if (
            ci_low is not None
            and ci_high is not None
            and ci_low >= -equivalence_margin
            and ci_high <= equivalence_margin
        ):
            return (
                f"B6-{label}={value:+.4f}{uncertainty}{ci_text}: the full CI lies "
                f"inside ±{equivalence_margin:.4f} and meets the configured practical-"
                "equivalence interval criterion."
            )
        if positive:
            verdict = "supports useful entropy information in this counterfactual."
        elif negative:
            verdict = "significantly favors the counterfactual over native entropy."
        else:
            verdict = (
                "is inconclusive: the CI is neither outside zero nor contained in the "
                "configured equivalence margin."
            )
        return f"B6-{label}={value:+.4f}{uncertainty}{ci_text}: {verdict}"
    if positive:
        verdict = "supports a positive contribution."
    elif negative:
        verdict = "shows a significant negative contribution."
    else:
        verdict = (
            "is inconclusive because the corresponding CI crosses or touches zero."
        )
    return f"{component}: delta={value:+.4f}{uncertainty}{ci_text}; {verdict}"


def _mechanism_conclusions(
    component_rows: Sequence[Mapping[str, Any]], equivalence_margin: float
) -> List[str]:
    aggregate = [row for row in component_rows if row["seed"] == "all"]
    by_pair: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for row in aggregate:
        original = str(row["component"])
        semantic = COMPONENT_ALIASES.get(original, original)
        pair_components = by_pair.setdefault(str(row["pair"]), {})
        pair_components[semantic] = row

    conclusions: List[str] = []
    for pair, components in sorted(by_pair.items()):
        conclusions.append(f"[{pair}]")
        for name in (
            "soft_candidates",
            "static_entropy",
            "gate_capacity",
            "gate_capacity_confounded",
            "entropy_values",
            "entropy_position",
        ):
            row = components.get(name)
            if row is not None:
                conclusions.append(
                    _conclusion_for_component(
                        name,
                        float(row["delta_accuracy_mean"]),
                        (
                            float(row["delta_accuracy_sample_std"])
                            if row["delta_accuracy_sample_std"] is not None
                            else None
                        ),
                        (
                            float(row["bootstrap_ci_low"])
                            if row["bootstrap_ci_low"] is not None
                            else None
                        ),
                        (
                            float(row["bootstrap_ci_high"])
                            if row["bootstrap_ci_high"] is not None
                            else None
                        ),
                        equivalence_margin,
                    )
                )
        full_static = components.get("full_over_static_entropy")
        full_gate = components.get("full_over_gate_only")
        if full_static is not None and full_gate is not None:
            static_delta = float(full_static["delta_accuracy_mean"])
            gate_delta = float(full_gate["delta_accuracy_mean"])
            static_low = full_static["bootstrap_ci_low"]
            gate_low = full_gate["bootstrap_ci_low"]
            if (
                static_low is not None
                and float(static_low) > 0.0
                and gate_low is not None
                and float(gate_low) > 0.0
            ):
                conclusions.append(
                    f"B6 exceeds both B4 ({static_delta:+.4f}) and B5 "
                    f"({gate_delta:+.4f}) with both cluster CIs above zero, supporting complementarity."
                )
            else:
                conclusions.append(
                    f"Complementarity is inconclusive: B6-B4={static_delta:+.4f} "
                    f"and B6-B5={gate_delta:+.4f}, but both corresponding CIs are not strictly above zero."
                )
    if not conclusions:
        conclusions.append("No complete default component contrast is available yet.")
    return conclusions


def _build_markdown(
    task_rows: Sequence[Mapping[str, Any]],
    aggregate_rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
    paired_bucket_rows: Sequence[Mapping[str, Any]],
    clustered_rows: Sequence[Mapping[str, Any]],
    final_gate_rows: Sequence[Mapping[str, Any]],
    gate_posthoc_coverage: Sequence[Mapping[str, Any]],
    gate_posthoc_rows: Sequence[Mapping[str, Any]],
    conclusions: Sequence[str],
    missing_diagnostics: Sequence[Mapping[str, Any]],
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> str:
    lines = ["# Route-1 v2.2 identifiability report", ""]
    lines.extend(
        [
            "## Accuracy and transfer by task",
            "",
            _markdown_table(
                (
                    "pair",
                    "method",
                    "seed",
                    "task",
                    "n",
                    "accuracy",
                    "positive transfer",
                    "negative transfer",
                ),
                [
                    (
                        row["pair"],
                        row["method"],
                        row["seed"],
                        row["task"],
                        row["n"],
                        _fmt(row["accuracy"]),
                        _fmt(row["positive_transfer_rate"]),
                        _fmt(row["negative_transfer_rate"]),
                    )
                    for row in task_rows
                ],
            ),
            "",
            "Positive transfer is conditioned on receiver-wrong examples; negative transfer is conditioned on receiver-correct examples.",
            "",
            "## Macro and sample-weighted means",
            "",
            _markdown_table(
                (
                    "pair",
                    "method",
                    "seed",
                    "tasks",
                    "macro mean",
                    "weighted mean",
                ),
                [
                    (
                        row["pair"],
                        row["method"],
                        row["seed"],
                        row["task_count"],
                        _fmt(row["macro_mean"]),
                        _fmt(row["weighted_mean"]),
                    )
                    for row in aggregate_rows
                ],
            ),
            "",
            "## Component contributions",
            "",
        ]
    )
    aggregate_components = [row for row in component_rows if row["seed"] == "all"]
    if aggregate_components:
        lines.extend(
            [
                _markdown_table(
                    (
                        "pair",
                        "component contrast",
                        "candidate - baseline",
                        "seeds",
                        "sample std",
                        "CI low",
                        "CI high",
                        "bootstrap level",
                        "positive seeds",
                    ),
                    [
                        (
                            row["pair"],
                            row["component"],
                            _fmt(row["delta_accuracy_mean"]),
                            row["n_seeds"],
                            _fmt(row["delta_accuracy_sample_std"]),
                            _fmt(row["bootstrap_ci_low"]),
                            _fmt(row["bootstrap_ci_high"]),
                            row.get("bootstrap_level"),
                            row["positive_seed_count"],
                        )
                        for row in aggregate_components
                    ],
                ),
                "",
            ]
        )
    else:
        lines.extend(["No complete component comparison is available.", ""])

    pooled_bucket_rows = [
        row
        for row in paired_bucket_rows
        if row["task"] == "__pooled__"
        and row["diagnostic_status"] != "missing"
        and row["field"]
        in {
            "alignment_bucket",
            "candidate_count",
            "alignment_entropy",
            "boundary_mismatch",
        }
    ]
    lines.extend(["## Paired mechanism gains in fixed candidate buckets", ""])
    if pooled_bucket_rows:
        lines.extend(
            [
                _markdown_table(
                    (
                        "pair",
                        "contrast",
                        "seed",
                        "candidate-defined field",
                        "bucket",
                        "n",
                        "delta",
                        "CI low",
                        "CI high",
                    ),
                    [
                        (
                            row["pair"],
                            row["comparison"],
                            row["seed"],
                            row["field"],
                            row["bucket"],
                            row["n_paired"],
                            _fmt(row["delta_accuracy"]),
                            _fmt(row["bootstrap_ci_low"]),
                            _fmt(row["bootstrap_ci_high"]),
                        )
                        for row in pooled_bucket_rows
                    ],
                ),
                "",
                "Buckets are assigned once from the candidate/aligner diagnostics and the same sample keys are used for both methods in each paired delta.",
                "",
            ]
        )
    else:
        lines.extend(["No complete paired mechanism bucket contrast is available.", ""])

    lines.extend(["## Cross-seed and cross-pair clustered inference", ""])
    pair_clustered = [
        row
        for row in clustered_rows
        if row["aggregation_level"] == "across_pairs" and row["cluster_unit"] == "pair"
    ]
    if pair_clustered:
        lines.extend(
            [
                _markdown_table(
                    (
                        "contrast",
                        "pairs",
                        "positive pairs",
                        "delta",
                        "cluster CI low",
                        "cluster CI high",
                        "aggregate McNemar p",
                    ),
                    [
                        (
                            row["comparison"],
                            row["n_pairs"],
                            row["positive_pair_count"],
                            _fmt(row["delta_accuracy"]),
                            _fmt(row["bootstrap_ci_low"]),
                            _fmt(row["bootstrap_ci_high"]),
                            _fmt(row["aggregate_mcnemar_exact_p"]),
                        )
                        for row in pair_clustered
                    ],
                ),
                "",
            ]
        )
    else:
        lines.extend(["No cross-pair contrast is available.", ""])

    lines.extend(
        [
            "## Final B6 gate",
            "",
            _markdown_table(
                (
                    "contrast",
                    "available pairs",
                    "positive pairs",
                    "aggregate CI positive",
                    "status",
                ),
                [
                    (
                        row["contrast"],
                        row["available_pair_count"],
                        row["positive_pair_count"],
                        row["aggregate_ci_positive"],
                        row["status"],
                    )
                    for row in final_gate_rows
                ],
            ),
            "",
        ]
    )

    lines.extend(["## Post-hoc token/head gate diagnostics", ""])
    if gate_posthoc_coverage:
        lines.extend(
            [
                _markdown_table(
                    (
                        "pair",
                        "method",
                        "seed",
                        "status",
                        "processed",
                        "gate examples",
                        "layers",
                        "heads rows",
                        "token bins",
                    ),
                    [
                        (
                            row["pair"],
                            row["method"],
                            row["seed"],
                            row["status"],
                            row.get("processed_samples"),
                            row.get("examples_with_gate"),
                            row.get("layer_count"),
                            row.get("layer_head_count"),
                            row.get("relative_token_bin_count"),
                        )
                        for row in gate_posthoc_coverage
                    ],
                ),
                "",
            ]
        )
        stage_rows = [row for row in gate_posthoc_rows if row["scope"] == "stage"]
        if stage_rows:
            lines.extend(
                [
                    _markdown_table(
                        (
                            "pair",
                            "method",
                            "seed",
                            "stage",
                            "K/V",
                            "mean",
                            "std",
                            "sat low",
                            "sat high",
                        ),
                        [
                            (
                                row["pair"],
                                row["method"],
                                row["seed"],
                                row["stage"],
                                row["kv"],
                                _fmt(row["mean"]),
                                _fmt(row["std"]),
                                _fmt(row["saturation_low_rate"]),
                                _fmt(row["saturation_high_rate"]),
                            )
                            for row in stage_rows
                        ],
                    ),
                    "",
                    "Full layer, layer/head, and relative-token-bin K/V statistics are in gate_posthoc_statistics.csv.",
                    "",
                ]
            )
    else:
        lines.extend(["No post-hoc gate diagnostic artifact is attached.", ""])

    lines.extend(["## Mechanism conclusions", ""])
    lines.extend(f"- {item}" for item in conclusions)
    lines.extend(["", "## Diagnostic coverage", ""])
    if missing_diagnostics:
        lines.append(
            _markdown_table(
                ("pair", "method", "seed", "task", "missing field"),
                [
                    (
                        row["pair"],
                        row["method"],
                        row["seed"],
                        row["task"],
                        row["field"],
                    )
                    for row in missing_diagnostics
                ],
            )
        )
    else:
        lines.append(
            "All requested diagnostic fields were present (alignment_bucket may be derived from candidate_count)."
        )
    lines.extend(
        [
            "",
            "## Statistical notes",
            "",
            f"Paired bootstrap uses {bootstrap_samples} resamples, confidence={bootstrap_confidence:.3f}, and deterministic base seed {bootstrap_seed}.",
            "Within-pair inference resamples pair/seed clusters then paired examples. Across-pair inference is hierarchical: pairs, then seeds within pair, then paired examples; pairs and seeds are equally weighted.",
            "McNemar p-values use the exact two-sided binomial test over discordant pairs.",
            "Aggregate McNemar pools paired predictions and is not cluster-adjusted; cluster uncertainty is represented by the bootstrap CI.",
            "Receiver reuse is deterministic: same pair+seed, same pair seed 42, same pair nearest seed, any pair same seed, any pair seed 42, then a lexically stable nearest fallback.",
            "Three-seed dispersion is the sample standard deviation (ddof=1); it is reported as missing when only one seed is available.",
            "A lower eval loss is not used in any mechanism conclusion.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_report(
    manifest_path: Path,
    output_dir: Path,
    *,
    bootstrap_samples: int = 5000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 20260717,
    equivalence_margin: float = 0.0025,
) -> Dict[str, Any]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < bootstrap_confidence < 1.0:
        raise ValueError("bootstrap_confidence must be between 0 and 1")
    if equivalence_margin < 0:
        raise ValueError("equivalence_margin cannot be negative")

    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
    specs, comparisons, default_receiver_method, metadata = _load_manifest(
        manifest_path
    )
    contract = _report_contract(metadata)
    runs = _load_runs(specs)
    receiver_by_run = _attach_receiver_correctness(runs, default_receiver_method)

    task_rows = [
        _task_metric_row(
            run,
            receiver_by_run[
                (run.spec.pair, run.spec.method, run.spec.seed, run.spec.task)
            ],
        )
        for run in runs
    ]
    aggregate_rows = _aggregate_metric_rows(task_rows)
    seed_rows = _seed_summary_rows(task_rows, aggregate_rows)
    comparison_rows, paired_clusters = _paired_comparison_rows(
        runs,
        comparisons,
        bootstrap_samples,
        bootstrap_confidence,
        bootstrap_seed,
        contract["expected_task_rows"],
    )
    clustered_rows = _clustered_comparison_rows(
        paired_clusters,
        bootstrap_samples,
        bootstrap_confidence,
        bootstrap_seed,
    )
    final_gate_rows = _final_gate_rows(clustered_rows, paired_clusters, contract)
    component_rows = _component_contribution_rows(comparison_rows, clustered_rows)
    bucket_rows = _bucket_metric_rows(runs, receiver_by_run)
    paired_bucket_rows = _paired_bucket_gain_rows(
        runs,
        comparisons,
        contract["expected_task_rows"],
        bootstrap_samples,
        bootstrap_confidence,
        bootstrap_seed,
    )
    correlation_rows = _correlation_rows(runs, receiver_by_run)
    gate_rows = _gate_statistic_rows(runs)
    gate_posthoc_coverage, gate_posthoc_rows = _posthoc_gate_rows(runs)
    conclusions = _mechanism_conclusions(component_rows, equivalence_margin)
    conclusions.extend(
        _posthoc_gate_conclusions(gate_posthoc_coverage, gate_posthoc_rows)
    )
    missing_diagnostics = [
        {
            "pair": row["pair"],
            "method": row["method"],
            "seed": row["seed"],
            "task": row["task"],
            "field": row["field"],
        }
        for row in bucket_rows
        if row["status"] == "missing" and row["bucket"] == "missing"
    ]

    summary: Dict[str, Any] = {
        "schema_version": 2,
        "manifest": str(manifest_path),
        "report_contract": contract,
        "statistics": {
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_confidence": bootstrap_confidence,
            "bootstrap_seed": bootstrap_seed,
            "cluster_bootstrap": (
                "within-pair: pair/seed clusters then paired examples; across-pair: "
                "pairs then seeds within pair then paired examples; equal pair and seed weighting"
            ),
            "equivalence_margin": equivalence_margin,
            "positive_transfer_denominator": "receiver_wrong",
            "negative_transfer_denominator": "receiver_correct",
            "receiver_reuse_order": (
                "same pair+seed, same pair seed42, same pair nearest seed, "
                "any pair same seed, any pair seed42, deterministic nearest fallback"
            ),
            "seed_std_ddof": 1,
        },
        "task_metrics": task_rows,
        "aggregate_metrics": aggregate_rows,
        "seed_summary": seed_rows,
        "paired_comparisons": comparison_rows,
        "clustered_comparisons": clustered_rows,
        "final_gate": final_gate_rows,
        "component_contributions": component_rows,
        "bucket_metrics": bucket_rows,
        "paired_bucket_gains": paired_bucket_rows,
        "correlations": correlation_rows,
        "gate_statistics": gate_rows,
        "gate_posthoc_coverage": gate_posthoc_coverage,
        "gate_posthoc_statistics": gate_posthoc_rows,
        "mechanism_conclusions": conclusions,
        "missing_diagnostics": missing_diagnostics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "task_metrics.csv", task_rows)
    _write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    _write_csv(output_dir / "seed_summary.csv", seed_rows)
    _write_csv(output_dir / "paired_comparisons.csv", comparison_rows)
    _write_csv(output_dir / "clustered_comparisons.csv", clustered_rows)
    _write_csv(output_dir / "final_gate.csv", final_gate_rows)
    _write_csv(output_dir / "component_contributions.csv", component_rows)
    _write_csv(output_dir / "bucket_metrics.csv", bucket_rows)
    _write_csv(output_dir / "paired_bucket_gains.csv", paired_bucket_rows)
    _write_csv(output_dir / "correlations.csv", correlation_rows)
    _write_csv(output_dir / "gate_statistics.csv", gate_rows)
    _write_csv(output_dir / "gate_posthoc_coverage.csv", gate_posthoc_coverage)
    _write_csv(output_dir / "gate_posthoc_statistics.csv", gate_posthoc_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        _build_markdown(
            task_rows,
            aggregate_rows,
            component_rows,
            paired_bucket_rows,
            clustered_rows,
            final_gate_rows,
            gate_posthoc_coverage,
            gate_posthoc_rows,
            conclusions,
            missing_diagnostics,
            bootstrap_samples,
            bootstrap_confidence,
            bootstrap_seed,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate route-1 identifiability per-example predictions into accuracy, "
            "transfer, paired significance, diagnostic buckets, and mechanism reports."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help=(
            "JSON/CSV manifest with method, pair, seed, task, and "
            "csv/per_example_csv per run"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for Markdown, JSON, and CSV report artifacts",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=20260717)
    parser.add_argument(
        "--equivalence-margin",
        type=float,
        default=0.0025,
        help=(
            "Point-estimate margin used only to flag provisional similarity for the "
            "constant/shuffle counterfactuals; it is not a formal equivalence test."
        ),
    )
    args = parser.parse_args()
    try:
        generate_report(
            args.manifest,
            args.output_dir,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_confidence=args.bootstrap_confidence,
            bootstrap_seed=args.bootstrap_seed,
            equivalence_margin=args.equivalence_margin,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"report={args.output_dir.resolve() / 'report.md'}")
    print(f"summary={args.output_dir.resolve() / 'summary.json'}")


if __name__ == "__main__":
    main()
