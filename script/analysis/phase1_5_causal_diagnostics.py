from __future__ import annotations

"""Phase 1.5 paired causal-intervention and oracle-abstention statistics.

The input is a JSON manifest using the same ``runs`` layout as
``route1_identifiability_report.py``.  Intervention method names are matched
exactly (rather than collapsed to B2/B3/B6 codes), which permits several
inference-time variants of the same checkpoint in one manifest.

Example additions to the Phase 1 manifest schema::

    {
      "receiver_method": "receiver_only",
      "comparisons": [
        {
          "name": "b2_train_k1_eval_k4_vs_k1",
          "baseline": "b2_train_k1_eval_k1",
          "candidate": "b2_train_k1_eval_k4",
          "ambiguity_source": "b3_native"
        }
      ],
      "ambiguity": {
        "score_fields": [
          "alignment_entropy", "one_to_many_rate", "boundary_mismatch"
        ],
        "quantile": 0.75
      },
      "runs": [
        {
          "method": "b3_native", "pair": "tinyllama", "seed": 42,
          "task": "mmlu-redux", "csv": "predictions.csv",
          "ambiguity_csv": "token_alignment_diagnostics.csv"
        }
      ]
    }

``ambiguity_csv`` is optional; when absent, diagnostic columns are read from
the prediction CSV.  Repeated sample IDs in a token-level sidecar are reduced
to sample-level candidate-count maxima and mean entropy/mismatch rates.
"""

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - production environments provide numpy.
    np = None  # type: ignore[assignment]

try:
    from script.analysis import route1_identifiability_report as phase1
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback.
    import route1_identifiability_report as phase1  # type: ignore[no-redef]


SampleKey = Tuple[str, ...]

DEFAULT_SCORE_FIELDS = (
    "alignment_entropy",
    "one_to_many_rate",
    "boundary_mismatch",
)
NUMERIC_ALIASES = {
    "candidate_count_max": (
        "candidate_count_max",
        "max_candidate_count",
        "candidate_count",
    ),
    "alignment_entropy": (
        "alignment_entropy",
        "alignment_entropy_mean",
        "entropy",
    ),
    "one_to_many_rate": (
        "one_to_many_rate",
        "one_to_many_fraction",
    ),
    "boundary_mismatch": (
        "boundary_mismatch",
        "boundary_mismatch_rate",
    ),
}

MAIN_EXECUTION_SUITE = "route1_phase1_5_same_checkpoint_interventions"
QWEN25_SEED44_ANOMALY_SUITE = "route1_phase1_5_qwen25_seed44_gate_anomaly"


def _main_execution_comparisons() -> List[Dict[str, str]]:
    """Return the preregistered eight main-matrix contrasts in fixed order."""

    return [
        {
            "name": "b2_train_k1_eval_k4_vs_k1",
            "baseline": "b2_native",
            "candidate": "b2_eval_k4",
            "ambiguity_source": "b3_native",
        },
        {
            "name": "b3_train_k4_eval_k4_vs_k1",
            "baseline": "b3_eval_k1",
            "candidate": "b3_native",
            "ambiguity_source": "b3_native",
        },
        {
            "name": "train_k4_vs_k1_at_eval_k1",
            "baseline": "b2_native",
            "candidate": "b3_eval_k1",
            "ambiguity_source": "b3_native",
        },
        {
            "name": "train_k4_vs_k1_at_eval_k4",
            "baseline": "b2_eval_k4",
            "candidate": "b3_native",
            "ambiguity_source": "b3_native",
        },
        {
            "name": "b6_native_entropy_vs_constant",
            "baseline": "b6_entropy_constant",
            "candidate": "b6_native",
            "ambiguity_source": "b6_native",
        },
        {
            "name": "b6_native_entropy_vs_shuffled",
            "baseline": "b6_entropy_shuffled",
            "candidate": "b6_native",
            "ambiguity_source": "b6_native",
        },
        {
            "name": "b6_learned_gate_vs_static",
            "baseline": "b6_gate_static",
            "candidate": "b6_native",
            "ambiguity_source": "b6_native",
        },
        {
            "name": "b6_learned_gate_vs_forced_on",
            "baseline": "b6_gate_forced_on",
            "candidate": "b6_native",
            "ambiguity_source": "b6_native",
        },
    ]


def _qwen25_seed44_anomaly_comparisons() -> List[Dict[str, str]]:
    """Return the two anomaly-only same-checkpoint gate contrasts."""

    return [
        {
            "name": "qwen25_seed44_alignment_forced_on_vs_native",
            "baseline": "b6_native",
            "candidate": "b6_gate_alignment_forced_on",
            "ambiguity_source": "b6_native",
        },
        {
            "name": "qwen25_seed44_legacy_forced_on_vs_native",
            "baseline": "b6_native",
            "candidate": "b6_gate_legacy_forced_on",
            "ambiguity_source": "b6_native",
        },
    ]


@dataclass(frozen=True)
class InterventionComparison:
    name: str
    baseline: str
    candidate: str
    ambiguity_source: str | None


@dataclass(frozen=True)
class LoadedRun:
    data: phase1.RunData
    ambiguity_path: Path
    ambiguity: Mapping[SampleKey, Mapping[str, float]]


@dataclass(frozen=True)
class ComparisonCluster:
    comparison: str
    pair: str
    seed: int
    baseline_method: str
    candidate_method: str
    differences: Tuple[int, ...]


@dataclass(frozen=True)
class AmbiguityCluster:
    comparison: str
    pair: str
    seed: int
    scheme: str
    high_differences: Tuple[int, ...]
    low_differences: Tuple[int, ...]


def _resolve_path(value: Any, manifest_dir: Path, field: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"Manifest entry is missing {field}")
    path = Path(str(value))
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def _prediction_path(entry: Mapping[str, Any], manifest_dir: Path) -> Path:
    direct = entry.get(
        "csv",
        entry.get("per_example_csv", entry.get("path", entry.get("predictions"))),
    )
    if direct:
        return _resolve_path(direct, manifest_dir, "csv/per_example_csv")
    pattern = entry.get("prediction_glob")
    if pattern:
        return phase1._resolve_prediction_glob(str(pattern), manifest_dir)
    raise ValueError("Manifest run needs csv/per_example_csv or prediction_glob")


def _ambiguity_path(
    entry: Mapping[str, Any], prediction_path: Path, manifest_dir: Path
) -> Path:
    direct = entry.get(
        "ambiguity_csv",
        entry.get("alignment_diagnostics_csv", entry.get("token_diagnostics_csv")),
    )
    if direct:
        return _resolve_path(direct, manifest_dir, "ambiguity_csv")
    pattern = entry.get("ambiguity_glob", entry.get("alignment_diagnostics_glob"))
    if pattern:
        return phase1._resolve_prediction_glob(str(pattern), manifest_dir)
    return prediction_path


def _expand_entries(
    entries: Sequence[Mapping[str, Any]], manifest_dir: Path
) -> List[Mapping[str, Any]]:
    expanded: List[Mapping[str, Any]] = []
    for entry in entries:
        datasets = entry.get("datasets")
        if not isinstance(datasets, Mapping):
            expanded.append(entry)
            continue
        method = entry.get("method", entry.get("variant"))
        for task, artifact in sorted(datasets.items()):
            if not isinstance(artifact, Mapping):
                raise ValueError(f"Dataset artifact must be an object: {task}")
            merged = dict(entry)
            merged.pop("datasets", None)
            merged.update(artifact)
            merged["method"] = method
            merged["task"] = task
            expanded.append(merged)
    return expanded


def _parse_comparisons(metadata: Mapping[str, Any]) -> List[InterventionComparison]:
    raw = metadata.get(
        "comparisons",
        metadata.get(
            "intervention_comparisons", metadata.get("component_comparisons", [])
        ),
    )
    if not isinstance(raw, list) or not raw:
        raise ValueError("Manifest needs a non-empty comparisons list")
    source_map = metadata.get("ambiguity", {}).get("bucket_sources", {})
    if not isinstance(source_map, Mapping):
        source_map = {}
    output: List[InterventionComparison] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("Each comparison must be an object")
        baseline = str(
            item.get("baseline", item.get("baseline_method", item.get("control", "")))
        ).strip()
        candidate = str(item.get("candidate", item.get("candidate_method", ""))).strip()
        name = str(item.get("name", f"{candidate}_vs_{baseline}")).strip()
        source = item.get(
            "ambiguity_source",
            item.get("bucket_source", source_map.get(name)),
        )
        source_name = str(source).strip() if source is not None else None
        if not name or not baseline or not candidate:
            raise ValueError("Each comparison needs name, baseline, and candidate")
        if name in seen:
            raise ValueError(f"Duplicate comparison name: {name}")
        seen.add(name)
        output.append(
            InterventionComparison(name, baseline, candidate, source_name or None)
        )
    return output


def _resolve_existing_path(value: Any, manifest_dir: Path, field: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"Manifest entry is missing {field}")
    raw = Path(str(value))
    candidates = (
        [raw]
        if raw.is_absolute()
        else [
            manifest_dir / raw,
            Path.cwd() / raw,
            Path(__file__).resolve().parents[2] / raw,
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return candidates[0].resolve()


def _single_prediction_in_output(output_dir: Path) -> Path:
    matches = sorted(output_dir.glob("*_cot.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *_cot.csv in {output_dir}, found {len(matches)}"
        )
    return matches[0].resolve()


def _materialize_execution_manifest(
    data: Mapping[str, Any], manifest_path: Path
) -> Dict[str, Any]:
    """Convert the Phase1.5 scheduler manifest into the analysis run schema."""

    source = data.get("source", {})
    if not isinstance(source, Mapping):
        raise ValueError("Phase1.5 execution manifest is missing source metadata")
    phase1_analysis_path = _resolve_existing_path(
        source.get("phase1_analysis_manifest"),
        manifest_path.parent,
        "source.phase1_analysis_manifest",
    )
    phase1_artifact_root = _resolve_existing_path(
        source.get("phase1_artifact_root"),
        manifest_path.parent,
        "source.phase1_artifact_root",
    )
    with phase1_analysis_path.open(encoding="utf-8") as handle:
        phase1_analysis = json.load(handle)

    materialized: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}

    def add_run(entry: Mapping[str, Any]) -> None:
        key = (
            str(entry["pair"]),
            str(entry["method"]),
            int(entry["seed"]),
            str(entry["task"]),
        )
        candidate = dict(entry)
        previous = materialized.get(key)
        if previous is not None:
            if (
                Path(str(previous["csv"])).resolve()
                != Path(str(candidate["csv"])).resolve()
            ):
                raise ValueError(f"Conflicting materialized run for {key}")
            return
        materialized[key] = candidate

    phase1_runs = phase1_analysis.get("runs", [])
    if not isinstance(phase1_runs, list):
        raise ValueError("Phase1 analysis manifest has invalid runs")
    receiver_runs = [
        run
        for run in phase1_runs
        if str(run.get("variant", run.get("method", ""))).casefold() == "b0"
    ]
    if not receiver_runs:
        raise ValueError("Phase1 analysis manifest has no B0 receiver run")
    for receiver_run in receiver_runs:
        datasets = receiver_run.get("datasets", {})
        if not isinstance(datasets, Mapping):
            continue
        for task, artifact in datasets.items():
            if not isinstance(artifact, Mapping):
                continue
            direct = artifact.get("csv", artifact.get("per_example_csv"))
            if direct:
                prediction = _resolve_existing_path(
                    direct, phase1_artifact_root, "B0 prediction csv"
                )
            else:
                pattern = artifact.get("prediction_glob")
                if not pattern:
                    continue
                matches = sorted(phase1_artifact_root.glob(str(pattern)))
                if len(matches) != 1:
                    raise FileNotFoundError(
                        f"Expected one B0 prediction for {pattern}, found {len(matches)}"
                    )
                prediction = matches[0].resolve()
            add_run(
                {
                    "method": "receiver_only",
                    "pair": str(receiver_run.get("pair", "receiver")),
                    "seed": int(receiver_run.get("seed", 42)),
                    "task": str(task),
                    "csv": str(prediction),
                }
            )

    execution_runs = data.get("runs", [])
    if not isinstance(execution_runs, list):
        raise ValueError("Phase1.5 execution manifest has invalid runs")
    for run in execution_runs:
        if not isinstance(run, Mapping):
            raise ValueError("Each Phase1.5 execution run must be an object")
        pair = str(run["pair"])
        seed = int(run["seed"])
        intervention = run.get("intervention", {})
        if not isinstance(intervention, Mapping):
            raise ValueError(f"Execution run has invalid intervention: {run.get('id')}")
        intervention_method = str(intervention.get("id", "")).strip()
        native = run.get("native_comparator", {})
        ambiguity = run.get("ambiguity_source", {})
        if (
            not intervention_method
            or not isinstance(native, Mapping)
            or not isinstance(ambiguity, Mapping)
        ):
            raise ValueError(
                f"Execution run lacks intervention/native/source: {run.get('id')}"
            )
        native_method = f"{str(native['variant']).casefold()}_native"
        ambiguity_method = f"{str(ambiguity['variant']).casefold()}_native"
        output_dirs = run.get("output_dirs", {})
        native_csvs = native.get("prediction_csv", {})
        ambiguity_csvs = ambiguity.get("prediction_csv", {})
        if not all(
            isinstance(value, Mapping)
            for value in (output_dirs, native_csvs, ambiguity_csvs)
        ):
            raise ValueError(
                f"Execution run has invalid dataset mappings: {run.get('id')}"
            )
        for task, output_value in output_dirs.items():
            output_dir = _resolve_existing_path(
                output_value, manifest_path.parent, f"{run.get('id')} output_dir"
            )
            intervention_csv = _single_prediction_in_output(output_dir)
            native_csv = _resolve_existing_path(
                native_csvs.get(task), manifest_path.parent, "native prediction csv"
            )
            ambiguity_csv = _resolve_existing_path(
                ambiguity_csvs.get(task),
                manifest_path.parent,
                "ambiguity prediction csv",
            )
            add_run(
                {
                    "method": intervention_method,
                    "pair": pair,
                    "seed": seed,
                    "task": str(task),
                    "csv": str(intervention_csv),
                }
            )
            add_run(
                {
                    "method": native_method,
                    "pair": pair,
                    "seed": seed,
                    "task": str(task),
                    "csv": str(native_csv),
                }
            )
            add_run(
                {
                    "method": ambiguity_method,
                    "pair": pair,
                    "seed": seed,
                    "task": str(task),
                    "csv": str(ambiguity_csv),
                    "ambiguity_csv": str(ambiguity_csv),
                }
            )

    return {
        "schema_version": 1,
        "source_execution_manifest": str(manifest_path),
        "receiver_method": "receiver_only",
        "oracle_methods": sorted(
            {key[1] for key in materialized if key[1] != "receiver_only"}
        ),
        "report_contract": phase1_analysis.get("report_contract", {}),
        "ambiguity": {
            "score_fields": list(DEFAULT_SCORE_FIELDS),
            "quantile": 0.75,
        },
        "pair_types": {
            "tinyllama": "heterogeneous",
            "qwen3_1p7b": "same_tokenizer",
            "qwen25_0p5b": "heterogeneous",
            "llama32_1b": "heterogeneous",
        },
        "comparisons": _main_execution_comparisons(),
        "runs": list(materialized.values()),
    }


def _analysis_run_key(entry: Mapping[str, Any]) -> Tuple[str, str, int, str]:
    return (
        str(entry["pair"]),
        str(entry["method"]),
        int(entry["seed"]),
        str(entry["task"]),
    )


def _validate_qwen25_seed44_anomaly_manifest(data: Mapping[str, Any]) -> None:
    if data.get("suite") != QWEN25_SEED44_ANOMALY_SUITE:
        raise ValueError(
            "--anomaly-manifest must use suite "
            f"{QWEN25_SEED44_ANOMALY_SUITE!r}"
        )
    runs = data.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Qwen2.5 seed-44 anomaly manifest has invalid runs")
    expected = {
        "b6_gate_alignment_forced_on",
        "b6_gate_legacy_forced_on",
    }
    observed: set[str] = set()
    for run in runs:
        if not isinstance(run, Mapping):
            raise ValueError("Each Qwen2.5 seed-44 anomaly run must be an object")
        intervention = run.get("intervention", {})
        if not isinstance(intervention, Mapping):
            raise ValueError("Qwen2.5 seed-44 anomaly run has invalid intervention")
        intervention_id = str(intervention.get("id", "")).strip()
        if (
            str(run.get("pair")) != "qwen25_0p5b"
            or int(run.get("seed", -1)) != 44
            or str(run.get("trained_variant", "")).casefold() != "b6"
        ):
            raise ValueError(
                "Qwen2.5 anomaly runs must be qwen25_0p5b/B6/seed 44"
            )
        observed.add(intervention_id)
    if observed != expected or len(runs) != len(expected):
        raise ValueError(
            "Qwen2.5 seed-44 anomaly manifest must contain exactly "
            f"{sorted(expected)}; observed {sorted(observed)}"
        )


def _merge_qwen25_seed44_anomaly(
    primary: Mapping[str, Any],
    anomaly: Mapping[str, Any],
    anomaly_manifest_path: Path,
) -> Dict[str, Any]:
    """Merge anomaly-only runs without changing the registered main contrasts."""

    primary_runs = primary.get("runs")
    anomaly_runs = anomaly.get("runs")
    if not isinstance(primary_runs, list) or not isinstance(anomaly_runs, list):
        raise ValueError("Materialized Phase1.5 manifests have invalid runs")

    merged_runs: List[Dict[str, Any]] = [dict(run) for run in primary_runs]
    by_key = {_analysis_run_key(run): run for run in merged_runs}
    for run in anomaly_runs:
        candidate = dict(run)
        key = _analysis_run_key(candidate)
        previous = by_key.get(key)
        if previous is not None:
            previous_csv = Path(str(previous["csv"])).resolve()
            candidate_csv = Path(str(candidate["csv"])).resolve()
            if previous_csv != candidate_csv:
                raise ValueError(
                    "Conflicting primary/anomaly prediction CSV for "
                    f"{key}: {previous_csv} != {candidate_csv}"
                )
            continue
        by_key[key] = candidate
        merged_runs.append(candidate)

    main_comparisons = primary.get("comparisons")
    if not isinstance(main_comparisons, list):
        raise ValueError("Materialized primary Phase1.5 manifest has no comparisons")
    expected_main_names = [item["name"] for item in _main_execution_comparisons()]
    observed_main_names = [
        str(item.get("name", ""))
        for item in main_comparisons
        if isinstance(item, Mapping)
    ]
    if observed_main_names != expected_main_names:
        raise ValueError(
            "Optional anomaly merge requires the unchanged eight registered main "
            "Phase1.5 comparisons"
        )

    anomaly_methods = {
        "b6_gate_alignment_forced_on",
        "b6_gate_legacy_forced_on",
    }
    present_anomaly_methods = {
        key[1]
        for key in by_key
        if key[0] == "qwen25_0p5b" and key[2] == 44
    }
    missing = anomaly_methods - present_anomaly_methods
    if missing:
        raise ValueError(
            f"Materialized anomaly manifest is missing methods: {sorted(missing)}"
        )

    oracle_methods = primary.get("oracle_methods", [])
    if not isinstance(oracle_methods, list):
        raise ValueError("Materialized primary oracle_methods must be a list")
    merged = dict(primary)
    merged["source_anomaly_execution_manifest"] = str(anomaly_manifest_path)
    merged["comparisons"] = [
        *[dict(item) for item in main_comparisons],
        *_qwen25_seed44_anomaly_comparisons(),
    ]
    merged["oracle_methods"] = sorted(set(map(str, oracle_methods)) | anomaly_methods)
    merged["runs"] = merged_runs
    return merged


def _numeric_values(value: Any) -> List[float]:
    text = str(value).strip()
    if not text:
        return []
    normalized = text.casefold()
    if normalized in {"true", "yes", "y"}:
        return [1.0]
    if normalized in {"false", "no", "n"}:
        return [0.0]
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        parsed = text
    if isinstance(parsed, list):
        output: List[float] = []
        for item in parsed:
            output.extend(_numeric_values(item))
        return output
    if isinstance(parsed, (int, float)) and math.isfinite(float(parsed)):
        return [float(parsed)]
    try:
        number = float(str(parsed))
    except ValueError:
        return []
    return [number] if math.isfinite(number) else []


def _field_values(
    rows: Sequence[Mapping[str, str]], names: Sequence[str]
) -> List[float]:
    output: List[float] = []
    for row in rows:
        for name in names:
            if name in row and str(row.get(name, "")).strip():
                output.extend(_numeric_values(row.get(name, "")))
                break
    return output


def _load_ambiguity(
    path: Path, custom_fields: Sequence[str]
) -> Dict[SampleKey, Mapping[str, float]]:
    rows, _fieldnames = phase1._read_csv(path)
    grouped: Dict[SampleKey, List[Mapping[str, str]]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(phase1._sample_key(row, index), []).append(row)

    output: Dict[SampleKey, Mapping[str, float]] = {}
    for key, sample_rows in grouped.items():
        candidate_values = _field_values(
            sample_rows, NUMERIC_ALIASES["candidate_count_max"]
        )
        entropy_values = _field_values(
            sample_rows, NUMERIC_ALIASES["alignment_entropy"]
        )
        one_to_many_values = _field_values(
            sample_rows, NUMERIC_ALIASES["one_to_many_rate"]
        )
        mismatch_values = _field_values(
            sample_rows, NUMERIC_ALIASES["boundary_mismatch"]
        )
        values: Dict[str, float] = {}
        if candidate_values:
            values["candidate_count_max"] = max(candidate_values)
        if entropy_values:
            values["alignment_entropy"] = statistics.fmean(entropy_values)
        if one_to_many_values:
            values["one_to_many_rate"] = statistics.fmean(one_to_many_values)
        elif candidate_values:
            values["one_to_many_rate"] = statistics.fmean(
                float(value > 1.0) for value in candidate_values
            )
        if mismatch_values:
            values["boundary_mismatch"] = statistics.fmean(mismatch_values)
        for field in custom_fields:
            field_values = _field_values(sample_rows, (field,))
            if field_values:
                values[field] = statistics.fmean(field_values)
        output[key] = values
    return output


def _load_manifest(
    manifest_path: Path,
    anomaly_manifest_path: Path | None = None,
) -> Tuple[
    List[LoadedRun],
    List[InterventionComparison],
    str,
    Mapping[str, Any],
]:
    with manifest_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    source_suite = data.get("suite") if isinstance(data, Mapping) else None
    if (
        isinstance(data, Mapping)
        and source_suite == MAIN_EXECUTION_SUITE
    ):
        data = _materialize_execution_manifest(data, manifest_path)
    if anomaly_manifest_path is not None:
        if source_suite != MAIN_EXECUTION_SUITE:
            raise ValueError(
                "--anomaly-manifest is only valid with the Phase1.5 same-checkpoint "
                "execution manifest"
            )
        with anomaly_manifest_path.open(encoding="utf-8") as handle:
            anomaly_source = json.load(handle)
        if not isinstance(anomaly_source, Mapping):
            raise ValueError("Qwen2.5 seed-44 anomaly manifest must be an object")
        _validate_qwen25_seed44_anomaly_manifest(anomaly_source)
        anomaly_data = _materialize_execution_manifest(
            anomaly_source, anomaly_manifest_path
        )
        if not isinstance(data, Mapping):  # Defensive: source suite was validated.
            raise ValueError("Materialized primary Phase1.5 manifest must be an object")
        data = _merge_qwen25_seed44_anomaly(
            data, anomaly_data, anomaly_manifest_path
        )
    entries, metadata = phase1._manifest_entries(data)
    comparisons = _parse_comparisons(metadata)
    ambiguity_config = metadata.get("ambiguity", {})
    if not isinstance(ambiguity_config, Mapping):
        ambiguity_config = {}
    score_fields_raw = ambiguity_config.get("score_fields", DEFAULT_SCORE_FIELDS)
    if isinstance(score_fields_raw, str):
        score_fields = (score_fields_raw,)
    elif isinstance(score_fields_raw, Sequence):
        score_fields = tuple(str(value) for value in score_fields_raw)
    else:
        raise ValueError("ambiguity.score_fields must be a string or list")

    manifest_dir = manifest_path.parent
    specs: List[phase1.RunSpec] = []
    ambiguity_paths: Dict[Tuple[str, str, int, str], Path] = {}
    seen: set[Tuple[str, str, int, str]] = set()
    for entry in _expand_entries(entries, manifest_dir):
        method = str(entry.get("method", entry.get("variant", ""))).strip()
        pair = str(entry.get("pair", "")).strip()
        task = str(entry.get("task", entry.get("dataset", ""))).strip()
        try:
            seed = int(entry.get("seed"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid seed for {method}/{pair}/{task}") from error
        if not method or not pair or not task:
            raise ValueError("Each run needs method, pair, seed, and task")
        key = (pair, method, seed, task)
        if key in seen:
            raise ValueError(f"Duplicate manifest run: {key}")
        seen.add(key)
        prediction_path = _prediction_path(entry, manifest_dir)
        ambiguity_paths[key] = _ambiguity_path(entry, prediction_path, manifest_dir)
        receiver_value = entry.get("receiver_csv", entry.get("baseline_csv"))
        specs.append(
            phase1.RunSpec(
                method=method,
                pair=pair,
                seed=seed,
                task=task,
                csv_path=prediction_path,
                receiver_csv_path=(
                    _resolve_path(receiver_value, manifest_dir, "receiver_csv")
                    if receiver_value is not None and str(receiver_value).strip()
                    else None
                ),
                receiver_method=(
                    str(entry["receiver_method"]).strip()
                    if entry.get("receiver_method")
                    else None
                ),
            )
        )
    run_data = phase1._load_runs(specs)
    ambiguity_cache: Dict[Path, Dict[SampleKey, Mapping[str, float]]] = {}
    loaded: List[LoadedRun] = []
    for run in run_data:
        key = (run.spec.pair, run.spec.method, run.spec.seed, run.spec.task)
        path = ambiguity_paths[key]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path not in ambiguity_cache:
            ambiguity_cache[path] = _load_ambiguity(path, score_fields)
        loaded.append(LoadedRun(run, path, ambiguity_cache[path]))
    return (
        loaded,
        comparisons,
        str(metadata.get("receiver_method", "B0")),
        metadata,
    )


def _exact_run(runs: Sequence[LoadedRun], method: str) -> LoadedRun | None:
    matches = [run for run in runs if run.data.spec.method == method]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous exact method in pair/seed/task: {method}")
    return matches[0] if matches else None


def _paired_row(
    comparison: InterventionComparison,
    pair: str,
    seed: int,
    task: str,
    baseline: Mapping[SampleKey, phase1.Sample],
    candidate: Mapping[SampleKey, phase1.Sample],
    *,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
    expected_n: int | None,
) -> Tuple[Dict[str, Any], Tuple[int, ...]]:
    baseline_keys = set(baseline)
    candidate_keys = set(candidate)
    common = sorted(baseline_keys & candidate_keys)
    exact_keys = baseline_keys == candidate_keys
    expected_ok = expected_n is None or (len(baseline) == len(candidate) == expected_n)
    eligible = bool(common) and exact_keys and expected_ok
    differences = (
        tuple(
            int(candidate[key].correct) - int(baseline[key].correct) for key in common
        )
        if eligible
        else ()
    )
    improvements = sum(value == 1 for value in differences)
    regressions = sum(value == -1 for value in differences)
    label = f"phase1.5:{comparison.name}:{pair}:{seed}:{task}"
    low, high = phase1._paired_bootstrap_ci(
        differences,
        samples=bootstrap_samples,
        confidence=bootstrap_confidence,
        seed=phase1._stable_seed(bootstrap_seed, label),
    )
    if not exact_keys:
        status = "sample_keys_mismatch"
    elif not expected_ok:
        status = "unexpected_n"
    elif not common:
        status = "empty"
    else:
        status = "ok"
    return (
        {
            "comparison": comparison.name,
            "pair": pair,
            "seed": seed,
            "task": task,
            "baseline_method": comparison.baseline,
            "candidate_method": comparison.candidate,
            "n_paired": len(common),
            "expected_n": expected_n,
            "baseline_accuracy": phase1._safe_rate(
                sum(baseline[key].correct for key in common), len(common)
            ),
            "candidate_accuracy": phase1._safe_rate(
                sum(candidate[key].correct for key in common), len(common)
            ),
            "delta_accuracy": statistics.fmean(differences) if differences else None,
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
            "improvements": improvements,
            "regressions": regressions,
            "mcnemar_exact_p": phase1._mcnemar_exact_p(improvements, regressions),
            "pairing_status": status,
            "aggregation_eligible": eligible,
        },
        differences,
    )


def _comparison_statistics(
    runs: Sequence[LoadedRun],
    comparisons: Sequence[InterventionComparison],
    expected_task_rows: Mapping[str, int],
    *,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> Tuple[List[Dict[str, Any]], List[ComparisonCluster]]:
    grouped: Dict[Tuple[str, int], List[LoadedRun]] = {}
    for run in runs:
        grouped.setdefault((run.data.spec.pair, run.data.spec.seed), []).append(run)
    rows: List[Dict[str, Any]] = []
    clusters: List[ComparisonCluster] = []
    for (pair, seed), pair_runs in sorted(grouped.items()):
        by_task: Dict[str, List[LoadedRun]] = {}
        for run in pair_runs:
            by_task.setdefault(run.data.spec.task, []).append(run)
        for comparison in comparisons:
            pooled_baseline: Dict[SampleKey, phase1.Sample] = {}
            pooled_candidate: Dict[SampleKey, phase1.Sample] = {}
            available_tasks: set[str] = set()
            task_eligible = True
            for task, task_runs in sorted(by_task.items()):
                baseline = _exact_run(task_runs, comparison.baseline)
                candidate = _exact_run(task_runs, comparison.candidate)
                if baseline is None or candidate is None:
                    continue
                available_tasks.add(task)
                row, _differences = _paired_row(
                    comparison,
                    pair,
                    seed,
                    task,
                    baseline.data.samples,
                    candidate.data.samples,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_confidence=bootstrap_confidence,
                    bootstrap_seed=bootstrap_seed,
                    expected_n=expected_task_rows.get(task),
                )
                rows.append(row)
                task_eligible &= bool(row["aggregation_eligible"])
                if row["aggregation_eligible"]:
                    for key, sample in baseline.data.samples.items():
                        pooled_baseline[(task, *key)] = sample
                    for key, sample in candidate.data.samples.items():
                        pooled_candidate[(task, *key)] = sample
            required_tasks = (
                set(expected_task_rows) if expected_task_rows else available_tasks
            )
            tasks_complete = (
                bool(required_tasks)
                and available_tasks == required_tasks
                and task_eligible
            )
            if not tasks_complete:
                continue
            pooled_row, differences = _paired_row(
                comparison,
                pair,
                seed,
                "__pooled__",
                pooled_baseline,
                pooled_candidate,
                bootstrap_samples=bootstrap_samples,
                bootstrap_confidence=bootstrap_confidence,
                bootstrap_seed=bootstrap_seed,
                expected_n=(
                    sum(expected_task_rows.values()) if expected_task_rows else None
                ),
            )
            rows.append(pooled_row)
            clusters.append(
                ComparisonCluster(
                    comparison.name,
                    pair,
                    seed,
                    comparison.baseline,
                    comparison.candidate,
                    differences,
                )
            )
    return rows, clusters


def _hierarchical_rows(
    clusters: Sequence[ComparisonCluster],
    pair_types: Mapping[str, str],
    *,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> List[Dict[str, Any]]:
    by_comparison: Dict[str, List[ComparisonCluster]] = {}
    for cluster in clusters:
        by_comparison.setdefault(cluster.comparison, []).append(cluster)
    rows: List[Dict[str, Any]] = []
    for comparison, comparison_clusters in sorted(by_comparison.items()):
        by_pair: Dict[str, List[ComparisonCluster]] = {}
        for cluster in comparison_clusters:
            by_pair.setdefault(cluster.pair, []).append(cluster)
        for pair, pair_clusters in sorted(by_pair.items()):
            cluster_map = {
                f"{item.pair}/seed_{item.seed}": item.differences
                for item in pair_clusters
            }
            point, low, high = phase1._cluster_bootstrap_ci(
                cluster_map,
                samples=bootstrap_samples,
                confidence=bootstrap_confidence,
                seed=phase1._stable_seed(
                    bootstrap_seed, f"phase1.5:{comparison}:{pair}:seeds"
                ),
            )
            seed_deltas = [statistics.fmean(item.differences) for item in pair_clusters]
            rows.append(
                {
                    "comparison": comparison,
                    "aggregation_level": "across_seeds_within_pair",
                    "pair": pair,
                    "n_pairs": 1,
                    "n_seeds": len(seed_deltas),
                    "n_pair_seed_runs": len(pair_clusters),
                    "delta_accuracy": point,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "bootstrap_level": "seeds_then_paired_examples",
                    "positive_pair_count": int(point is not None and point > 0.0),
                    "pair_deltas_json": json.dumps({pair: point}),
                    "seed_sample_std": phase1._sample_std(seed_deltas),
                }
            )
        source = [
            phase1.PairedCluster(
                item.comparison,
                item.pair,
                item.seed,
                item.baseline_method,
                item.candidate_method,
                item.differences,
            )
            for item in comparison_clusters
        ]
        point, low, high = phase1._hierarchical_pair_seed_bootstrap_ci(
            source,
            samples=bootstrap_samples,
            confidence=bootstrap_confidence,
            seed=phase1._stable_seed(
                bootstrap_seed, f"phase1.5:{comparison}:all_pairs"
            ),
        )
        pair_deltas = {
            pair: statistics.fmean(
                statistics.fmean(item.differences) for item in pair_clusters
            )
            for pair, pair_clusters in sorted(by_pair.items())
        }
        rows.append(
            {
                "comparison": comparison,
                "aggregation_level": "across_pairs",
                "pair": "__all__",
                "n_pairs": len(pair_deltas),
                "n_seeds": len({item.seed for item in comparison_clusters}),
                "n_pair_seed_runs": len(comparison_clusters),
                "delta_accuracy": point,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "bootstrap_level": "pairs_then_seeds_then_paired_examples",
                "positive_pair_count": sum(
                    value > 0.0 for value in pair_deltas.values()
                ),
                "heterogeneous_pair_count": sum(
                    pair_types.get(pair) == "heterogeneous" for pair in pair_deltas
                ),
                "positive_heterogeneous_pair_count": sum(
                    value > 0.0 and pair_types.get(pair) == "heterogeneous"
                    for pair, value in pair_deltas.items()
                ),
                "pair_deltas_json": json.dumps(
                    pair_deltas, sort_keys=True, separators=(",", ":")
                ),
                "seed_sample_std": None,
            }
        )
    return rows


def _seed_variance_rows(
    clusters: Sequence[ComparisonCluster],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[ComparisonCluster]] = {}
    by_comparison_seed: Dict[Tuple[str, int], List[ComparisonCluster]] = {}
    for cluster in clusters:
        grouped.setdefault((cluster.comparison, cluster.pair), []).append(cluster)
        by_comparison_seed.setdefault((cluster.comparison, cluster.seed), []).append(
            cluster
        )
    rows: List[Dict[str, Any]] = []
    for (comparison, pair), items in sorted(grouped.items()):
        values = sorted(
            (item.seed, statistics.fmean(item.differences)) for item in items
        )
        numbers = [value for _seed, value in values]
        rows.append(
            {
                "comparison": comparison,
                "pair": pair,
                "n_seeds": len(values),
                "mean_delta": statistics.fmean(numbers),
                "seed_sample_std": phase1._sample_std(numbers),
                "positive_seed_count": sum(value > 0.0 for value in numbers),
                "seed_deltas_json": json.dumps(dict(values), separators=(",", ":")),
            }
        )
    comparisons = sorted({cluster.comparison for cluster in clusters})
    for comparison in comparisons:
        seed_values: List[Tuple[int, float]] = []
        for (name, seed), items in sorted(by_comparison_seed.items()):
            if name != comparison:
                continue
            by_pair: Dict[str, List[float]] = {}
            for item in items:
                by_pair.setdefault(item.pair, []).append(
                    statistics.fmean(item.differences)
                )
            seed_values.append(
                (seed, statistics.fmean(statistics.fmean(v) for v in by_pair.values()))
            )
        numbers = [value for _seed, value in seed_values]
        rows.append(
            {
                "comparison": comparison,
                "pair": "__all__",
                "n_seeds": len(seed_values),
                "mean_delta": statistics.fmean(numbers) if numbers else None,
                "seed_sample_std": phase1._sample_std(numbers),
                "positive_seed_count": sum(value > 0.0 for value in numbers),
                "seed_deltas_json": json.dumps(
                    dict(seed_values), separators=(",", ":")
                ),
            }
        )
    return rows


def _oracle_task_row(
    run: LoadedRun,
    receiver: phase1.ReceiverAttachment,
    task: str,
) -> Tuple[Dict[str, Any], Tuple[int, ...]]:
    common = sorted(set(run.data.samples) & set(receiver.values))
    fused = [run.data.samples[key].correct for key in common]
    receiver_correct = [receiver.values[key] for key in common]
    oracle = [left or right for left, right in zip(fused, receiver_correct)]
    negative = sum(
        base and not candidate for base, candidate in zip(receiver_correct, fused)
    )
    positive = sum(
        not base and candidate for base, candidate in zip(receiver_correct, fused)
    )
    differences = tuple(
        int(best) - int(candidate) for best, candidate in zip(oracle, fused)
    )
    n = len(common)
    fused_accuracy = phase1._safe_rate(sum(fused), n)
    receiver_accuracy = phase1._safe_rate(sum(receiver_correct), n)
    oracle_accuracy = phase1._safe_rate(sum(oracle), n)
    return (
        {
            "method": run.data.spec.method,
            "pair": run.data.spec.pair,
            "seed": run.data.spec.seed,
            "task": task,
            "aggregation_level": (
                "pair_seed_task" if task != "__pooled__" else "pair_seed_pooled"
            ),
            "n_paired": n,
            "receiver_accuracy": receiver_accuracy,
            "fused_accuracy": fused_accuracy,
            "oracle_accuracy": oracle_accuracy,
            "oracle_headroom_over_fused": (
                oracle_accuracy - fused_accuracy
                if oracle_accuracy is not None and fused_accuracy is not None
                else None
            ),
            "oracle_headroom_over_receiver": (
                oracle_accuracy - receiver_accuracy
                if oracle_accuracy is not None and receiver_accuracy is not None
                else None
            ),
            "oracle_headroom_over_best_fixed": (
                oracle_accuracy - max(receiver_accuracy, fused_accuracy)
                if oracle_accuracy is not None
                and receiver_accuracy is not None
                and fused_accuracy is not None
                else None
            ),
            "ideal_abstain_count": negative,
            "ideal_abstain_rate": phase1._safe_rate(negative, n),
            "beneficial_transfer_count": positive,
            "beneficial_transfer_rate": phase1._safe_rate(positive, n),
            "decision_relevant_rate": phase1._safe_rate(positive + negative, n),
            "receiver_source_kind": receiver.source_kind,
            "receiver_source_seed": receiver.source_seed,
            "pairing_status": (
                "ok" if n == len(run.data.samples) else "partial_receiver_pairing"
            ),
        },
        differences,
    )


def _oracle_statistics(
    runs: Sequence[LoadedRun],
    receiver_method: str,
    metadata: Mapping[str, Any],
    *,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> List[Dict[str, Any]]:
    report_runs = [run.data for run in runs]
    attached = phase1._attach_receiver_correctness(report_runs, receiver_method)
    configured = metadata.get("oracle_methods")
    if configured is None:
        oracle_methods = {
            run.data.spec.method
            for run in runs
            if not phase1._method_matches(run.data.spec.method, receiver_method)
        }
    elif isinstance(configured, Sequence) and not isinstance(configured, str):
        oracle_methods = {str(value) for value in configured}
    else:
        raise ValueError("oracle_methods must be a list")

    rows: List[Dict[str, Any]] = []
    clusters: Dict[str, List[ComparisonCluster]] = {}
    grouped: Dict[Tuple[str, str, int], List[LoadedRun]] = {}
    for run in runs:
        if run.data.spec.method in oracle_methods:
            grouped.setdefault(
                (run.data.spec.method, run.data.spec.pair, run.data.spec.seed), []
            ).append(run)
    for (method, pair, seed), items in sorted(grouped.items()):
        pooled_samples: Dict[SampleKey, phase1.Sample] = {}
        pooled_receiver: Dict[SampleKey, bool] = {}
        source: phase1.ReceiverAttachment | None = None
        for run in sorted(items, key=lambda item: item.data.spec.task):
            attachment = attached[(pair, method, seed, run.data.spec.task)]
            row, differences = _oracle_task_row(run, attachment, run.data.spec.task)
            low, high = phase1._paired_bootstrap_ci(
                differences,
                samples=bootstrap_samples,
                confidence=bootstrap_confidence,
                seed=phase1._stable_seed(
                    bootstrap_seed,
                    f"phase1.5:oracle:{method}:{pair}:{seed}:{run.data.spec.task}",
                ),
            )
            row["bootstrap_ci_low"] = low
            row["bootstrap_ci_high"] = high
            rows.append(row)
            source = attachment
            for key, sample in run.data.samples.items():
                pooled_samples[(run.data.spec.task, *key)] = sample
            for key, value in attachment.values.items():
                pooled_receiver[(run.data.spec.task, *key)] = value
        if source is None:
            continue
        pooled_spec = phase1.RunSpec(
            method=method,
            pair=pair,
            seed=seed,
            task="__pooled__",
            csv_path=Path("__pooled__"),
        )
        pooled_run = LoadedRun(
            phase1.RunData(pooled_spec, pooled_samples, set()),
            Path("__pooled__"),
            {},
        )
        pooled_attachment = phase1.ReceiverAttachment(
            pooled_receiver,
            source.source_kind,
            source.source_pair,
            source.source_method,
            source.source_seed,
            source.source_csv,
        )
        row, differences = _oracle_task_row(pooled_run, pooled_attachment, "__pooled__")
        low, high = phase1._paired_bootstrap_ci(
            differences,
            samples=bootstrap_samples,
            confidence=bootstrap_confidence,
            seed=phase1._stable_seed(
                bootstrap_seed, f"phase1.5:oracle:{method}:{pair}:{seed}:pooled"
            ),
        )
        row["bootstrap_ci_low"] = low
        row["bootstrap_ci_high"] = high
        rows.append(row)
        clusters.setdefault(method, []).append(
            ComparisonCluster(
                f"oracle::{method}",
                pair,
                seed,
                method,
                f"oracle({method},receiver)",
                differences,
            )
        )

    for method, method_clusters in sorted(clusters.items()):
        by_pair: Dict[str, List[ComparisonCluster]] = {}
        for cluster in method_clusters:
            by_pair.setdefault(cluster.pair, []).append(cluster)
        for pair, pair_clusters in sorted(by_pair.items()):
            cluster_map = {
                f"seed_{item.seed}": item.differences for item in pair_clusters
            }
            point, low, high = phase1._cluster_bootstrap_ci(
                cluster_map,
                samples=bootstrap_samples,
                confidence=bootstrap_confidence,
                seed=phase1._stable_seed(
                    bootstrap_seed, f"phase1.5:oracle:{method}:{pair}:seeds"
                ),
            )
            rows.append(
                {
                    "method": method,
                    "pair": pair,
                    "seed": "all",
                    "task": "__pooled__",
                    "aggregation_level": "across_seeds_within_pair",
                    "n_paired": sum(len(item.differences) for item in pair_clusters),
                    "oracle_headroom_over_fused": point,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "n_pairs": 1,
                    "n_seeds": len(pair_clusters),
                    "bootstrap_level": "seeds_then_paired_examples",
                }
            )
        source = [
            phase1.PairedCluster(
                item.comparison,
                item.pair,
                item.seed,
                item.baseline_method,
                item.candidate_method,
                item.differences,
            )
            for item in method_clusters
        ]
        point, low, high = phase1._hierarchical_pair_seed_bootstrap_ci(
            source,
            samples=bootstrap_samples,
            confidence=bootstrap_confidence,
            seed=phase1._stable_seed(
                bootstrap_seed, f"phase1.5:oracle:{method}:all_pairs"
            ),
        )
        pair_deltas = {
            pair: statistics.fmean(statistics.fmean(item.differences) for item in items)
            for pair, items in sorted(by_pair.items())
        }
        rows.append(
            {
                "method": method,
                "pair": "__all__",
                "seed": "all",
                "task": "__pooled__",
                "aggregation_level": "across_pairs",
                "n_paired": sum(len(item.differences) for item in method_clusters),
                "oracle_headroom_over_fused": point,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "n_pairs": len(by_pair),
                "n_seeds": len({item.seed for item in method_clusters}),
                "bootstrap_level": "pairs_then_seeds_then_paired_examples",
                "positive_pair_count": sum(
                    value > 0.0 for value in pair_deltas.values()
                ),
                "pair_headroom_json": json.dumps(
                    pair_deltas, sort_keys=True, separators=(",", ":")
                ),
            }
        )
    return rows


def _ambiguity_masks(
    ambiguity: Mapping[SampleKey, Mapping[str, float]],
    keys: Sequence[SampleKey],
    score_fields: Sequence[str],
    quantile: float,
) -> Tuple[Mapping[str, set[SampleKey]], Mapping[str, Any]]:
    available = [key for key in keys if ambiguity.get(key)]
    absolute = {
        key
        for key in available
        if ambiguity[key].get("candidate_count_max", 0.0) > 1.0
        and (
            ambiguity[key].get("alignment_entropy", 0.0) > 0.0
            or ambiguity[key].get("one_to_many_rate", 0.0) > 0.0
            or ambiguity[key].get("boundary_mismatch", 0.0) > 0.0
        )
    }
    normalized: Dict[str, Dict[SampleKey, float]] = {}
    used_fields: List[str] = []
    for field in score_fields:
        field_values = {
            key: ambiguity[key][field] for key in available if field in ambiguity[key]
        }
        if not field_values:
            continue
        low = min(field_values.values())
        high = max(field_values.values())
        if high <= low:
            continue
        used_fields.append(field)
        normalized[field] = {
            key: (value - low) / (high - low) for key, value in field_values.items()
        }
    scores = {
        key: statistics.fmean(
            normalized[field][key] for field in used_fields if key in normalized[field]
        )
        for key in available
        if any(key in normalized[field] for field in used_fields)
    }
    threshold = phase1._quantile(list(scores.values()), quantile) if scores else None
    top_quantile = {
        key
        for key, value in scores.items()
        if threshold is not None and value >= threshold
    }
    return (
        {"absolute": absolute, f"q{int(round(quantile * 100))}": top_quantile},
        {
            "diagnostic_key_count": len(available),
            "score_fields": ",".join(used_fields),
            "score_threshold": threshold,
        },
    )


def _stratified_bootstrap_ci(
    high: Sequence[int],
    low: Sequence[int],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> Tuple[float | None, float | None]:
    if not high or not low:
        return None, None
    if np is None:  # pragma: no cover - production/test environments provide numpy.
        import random

        rng = random.Random(seed)
        boot = [
            statistics.fmean(rng.choices(high, k=len(high)))
            - statistics.fmean(rng.choices(low, k=len(low)))
            for _ in range(samples)
        ]
    else:
        rng = np.random.default_rng(seed)

        def draws(values: Sequence[int]) -> Any:
            n = len(values)
            negative = sum(value < 0 for value in values)
            zero = sum(value == 0 for value in values)
            positive = n - negative - zero
            counts = rng.multinomial(
                n, [negative / n, zero / n, positive / n], size=samples
            )
            return (counts[:, 2] - counts[:, 0]) / n

        boot = draws(high) - draws(low)
    alpha = (1.0 - confidence) / 2.0
    if np is not None:
        try:
            bounds = np.quantile(boot, [alpha, 1.0 - alpha], method="linear")
        except TypeError:  # pragma: no cover - numpy < 1.22.
            bounds = np.quantile(boot, [alpha, 1.0 - alpha], interpolation="linear")
        return float(bounds[0]), float(bounds[1])
    return phase1._quantile(boot, alpha), phase1._quantile(boot, 1.0 - alpha)


def _draw_example_mean(values: Sequence[int], rng: Any) -> float:
    n = len(values)
    negative = sum(value < 0 for value in values)
    zero = sum(value == 0 for value in values)
    positive = n - negative - zero
    if np is not None:
        counts = rng.multinomial(n, [negative / n, zero / n, positive / n])
        return float(counts[2] - counts[0]) / n
    return statistics.fmean(rng.choices(values, k=n))


def _hierarchical_interaction_ci(
    clusters: Sequence[AmbiguityCluster],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> Tuple[float | None, float | None, float | None]:
    by_pair: Dict[str, List[Tuple[Tuple[int, ...], Tuple[int, ...]]]] = {}
    for cluster in clusters:
        if cluster.high_differences and cluster.low_differences:
            by_pair.setdefault(cluster.pair, []).append(
                (cluster.high_differences, cluster.low_differences)
            )
    clean = [(pair, values) for pair, values in sorted(by_pair.items()) if values]
    if not clean:
        return None, None, None
    point = statistics.fmean(
        statistics.fmean(
            statistics.fmean(high) - statistics.fmean(low) for high, low in seeds
        )
        for _pair, seeds in clean
    )
    if np is None:  # pragma: no cover - production/test environments provide numpy.
        import random

        rng: Any = random.Random(seed)
        select_pairs = lambda n: rng.choices(range(n), k=n)
        select_seeds = lambda n: rng.choices(range(n), k=n)
    else:
        rng = np.random.default_rng(seed)
        select_pairs = lambda n: rng.integers(0, n, size=n)
        select_seeds = lambda n: rng.integers(0, n, size=n)
    boot: List[float] = []
    for _ in range(samples):
        pair_means: List[float] = []
        for pair_index in select_pairs(len(clean)):
            seed_clusters = clean[int(pair_index)][1]
            seed_means = []
            for seed_index in select_seeds(len(seed_clusters)):
                high, low = seed_clusters[int(seed_index)]
                seed_means.append(
                    _draw_example_mean(high, rng) - _draw_example_mean(low, rng)
                )
            pair_means.append(statistics.fmean(seed_means))
        boot.append(statistics.fmean(pair_means))
    alpha = (1.0 - confidence) / 2.0
    return point, phase1._quantile(boot, alpha), phase1._quantile(boot, 1.0 - alpha)


def _ambiguity_statistics(
    runs: Sequence[LoadedRun],
    comparisons: Sequence[InterventionComparison],
    metadata: Mapping[str, Any],
    *,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> List[Dict[str, Any]]:
    config = metadata.get("ambiguity", {})
    if not isinstance(config, Mapping):
        config = {}
    pair_types = metadata.get("pair_types", {})
    if not isinstance(pair_types, Mapping):
        pair_types = {}
    raw_fields = config.get("score_fields", DEFAULT_SCORE_FIELDS)
    score_fields = (raw_fields,) if isinstance(raw_fields, str) else tuple(raw_fields)
    quantile = float(config.get("quantile", 0.75))
    if not 0.0 < quantile < 1.0:
        raise ValueError("ambiguity.quantile must be between 0 and 1")

    grouped: Dict[Tuple[str, int, str], List[LoadedRun]] = {}
    for run in runs:
        grouped.setdefault(
            (run.data.spec.pair, run.data.spec.seed, run.data.spec.task), []
        ).append(run)
    rows: List[Dict[str, Any]] = []
    pooled: Dict[Tuple[str, str, int], Dict[str, List[int]]] = {}
    for (pair, seed, task), task_runs in sorted(grouped.items()):
        for comparison in comparisons:
            if not comparison.ambiguity_source:
                continue
            baseline = _exact_run(task_runs, comparison.baseline)
            candidate = _exact_run(task_runs, comparison.candidate)
            source = _exact_run(task_runs, comparison.ambiguity_source)
            if baseline is None or candidate is None or source is None:
                continue
            common = sorted(set(baseline.data.samples) & set(candidate.data.samples))
            if set(baseline.data.samples) != set(candidate.data.samples):
                continue
            differences = {
                key: int(candidate.data.samples[key].correct)
                - int(baseline.data.samples[key].correct)
                for key in common
            }
            masks, coverage = _ambiguity_masks(
                source.ambiguity, common, score_fields, quantile
            )
            for scheme, high_keys in masks.items():
                diagnostic_keys = set(source.ambiguity) & set(common)
                low_keys = diagnostic_keys - high_keys
                high = tuple(differences[key] for key in sorted(high_keys))
                low = tuple(differences[key] for key in sorted(low_keys))
                high_low, high_high = phase1._paired_bootstrap_ci(
                    high,
                    samples=bootstrap_samples,
                    confidence=bootstrap_confidence,
                    seed=phase1._stable_seed(
                        bootstrap_seed,
                        f"phase1.5:ambiguity:{comparison.name}:{pair}:{seed}:{task}:{scheme}:high",
                    ),
                )
                interaction_low, interaction_high = _stratified_bootstrap_ci(
                    high,
                    low,
                    samples=bootstrap_samples,
                    confidence=bootstrap_confidence,
                    seed=phase1._stable_seed(
                        bootstrap_seed,
                        f"phase1.5:ambiguity:{comparison.name}:{pair}:{seed}:{task}:{scheme}:interaction",
                    ),
                )
                high_delta = statistics.fmean(high) if high else None
                low_delta = statistics.fmean(low) if low else None
                rows.append(
                    {
                        "comparison": comparison.name,
                        "pair": pair,
                        "seed": seed,
                        "task": task,
                        "aggregation_level": "pair_seed_task",
                        "scheme": scheme,
                        "ambiguity_source_method": comparison.ambiguity_source,
                        "ambiguity_source_csv": str(source.ambiguity_path),
                        "n_paired": len(common),
                        "diagnostic_key_count": coverage["diagnostic_key_count"],
                        "high_n": len(high),
                        "low_n": len(low),
                        "high_delta_accuracy": high_delta,
                        "high_bootstrap_ci_low": high_low,
                        "high_bootstrap_ci_high": high_high,
                        "low_delta_accuracy": low_delta,
                        "ambiguity_interaction": (
                            high_delta - low_delta
                            if high_delta is not None and low_delta is not None
                            else None
                        ),
                        "interaction_ci_low": interaction_low,
                        "interaction_ci_high": interaction_high,
                        "score_fields": coverage["score_fields"],
                        "score_threshold": coverage["score_threshold"],
                        "status": "ok" if high and low else "empty_high_or_low_bucket",
                    }
                )
                target = pooled.setdefault(
                    (comparison.name, scheme, seed, pair), {"high": [], "low": []}
                )
                target["high"].extend(high)
                target["low"].extend(low)

    clusters_by_key: Dict[Tuple[str, str], List[AmbiguityCluster]] = {}
    for (comparison, scheme, seed, pair), values in sorted(pooled.items()):
        cluster = AmbiguityCluster(
            comparison,
            pair,
            seed,
            scheme,
            tuple(values["high"]),
            tuple(values["low"]),
        )
        clusters_by_key.setdefault((comparison, scheme), []).append(cluster)
        high_delta = (
            statistics.fmean(cluster.high_differences)
            if cluster.high_differences
            else None
        )
        low_delta = (
            statistics.fmean(cluster.low_differences)
            if cluster.low_differences
            else None
        )
        interaction_low, interaction_high = _stratified_bootstrap_ci(
            cluster.high_differences,
            cluster.low_differences,
            samples=bootstrap_samples,
            confidence=bootstrap_confidence,
            seed=phase1._stable_seed(
                bootstrap_seed,
                f"phase1.5:ambiguity:{comparison}:{pair}:{seed}:pooled:{scheme}",
            ),
        )
        rows.append(
            {
                "comparison": comparison,
                "pair": pair,
                "seed": seed,
                "task": "__pooled__",
                "aggregation_level": "pair_seed_pooled",
                "scheme": scheme,
                "high_n": len(cluster.high_differences),
                "low_n": len(cluster.low_differences),
                "high_delta_accuracy": high_delta,
                "low_delta_accuracy": low_delta,
                "ambiguity_interaction": (
                    high_delta - low_delta
                    if high_delta is not None and low_delta is not None
                    else None
                ),
                "interaction_ci_low": interaction_low,
                "interaction_ci_high": interaction_high,
                "status": (
                    "ok"
                    if cluster.high_differences and cluster.low_differences
                    else "empty_high_or_low_bucket"
                ),
            }
        )

    for (comparison, scheme), clusters in sorted(clusters_by_key.items()):
        by_pair: Dict[str, List[AmbiguityCluster]] = {}
        for cluster in clusters:
            by_pair.setdefault(cluster.pair, []).append(cluster)
        for pair, pair_clusters in [*sorted(by_pair.items()), ("__all__", clusters)]:
            if pair == "__all__":
                point, low, high = _hierarchical_interaction_ci(
                    pair_clusters,
                    samples=bootstrap_samples,
                    confidence=bootstrap_confidence,
                    seed=phase1._stable_seed(
                        bootstrap_seed,
                        f"phase1.5:ambiguity:{comparison}:{scheme}:all_pairs",
                    ),
                )
                aggregation_level = "across_pairs"
                bootstrap_level = "pairs_then_seeds_then_stratified_paired_examples"
            else:
                point, low, high = _hierarchical_interaction_ci(
                    pair_clusters,
                    samples=bootstrap_samples,
                    confidence=bootstrap_confidence,
                    seed=phase1._stable_seed(
                        bootstrap_seed,
                        f"phase1.5:ambiguity:{comparison}:{scheme}:{pair}:seeds",
                    ),
                )
                aggregation_level = "across_seeds_within_pair"
                bootstrap_level = "seeds_then_stratified_paired_examples"
            high_source = [
                phase1.PairedCluster(
                    cluster.comparison,
                    cluster.pair,
                    cluster.seed,
                    "baseline",
                    "candidate",
                    cluster.high_differences,
                )
                for cluster in pair_clusters
                if cluster.high_differences
            ]
            high_point, high_low, high_high = (
                phase1._hierarchical_pair_seed_bootstrap_ci(
                    high_source,
                    samples=bootstrap_samples,
                    confidence=bootstrap_confidence,
                    seed=phase1._stable_seed(
                        bootstrap_seed,
                        f"phase1.5:ambiguity:{comparison}:{scheme}:{pair}:high",
                    ),
                )
            )
            pair_interactions = {
                pair_name: statistics.fmean(
                    statistics.fmean(item.high_differences)
                    - statistics.fmean(item.low_differences)
                    for item in items
                    if item.high_differences and item.low_differences
                )
                for pair_name, items in sorted(by_pair.items())
                if any(item.high_differences and item.low_differences for item in items)
            }
            rows.append(
                {
                    "comparison": comparison,
                    "pair": pair,
                    "seed": "all",
                    "task": "__pooled__",
                    "aggregation_level": aggregation_level,
                    "scheme": scheme,
                    "n_pairs": len(by_pair) if pair == "__all__" else 1,
                    "n_seeds": len({item.seed for item in pair_clusters}),
                    "n_pair_seed_runs": len(pair_clusters),
                    "high_n": sum(len(item.high_differences) for item in pair_clusters),
                    "low_n": sum(len(item.low_differences) for item in pair_clusters),
                    "high_delta_accuracy": high_point,
                    "high_bootstrap_ci_low": high_low,
                    "high_bootstrap_ci_high": high_high,
                    "ambiguity_interaction": point,
                    "interaction_ci_low": low,
                    "interaction_ci_high": high,
                    "bootstrap_level": bootstrap_level,
                    "positive_pair_count": sum(
                        value > 0.0 for value in pair_interactions.values()
                    ),
                    "heterogeneous_pair_count": sum(
                        pair_types.get(pair_name) == "heterogeneous"
                        for pair_name in pair_interactions
                    ),
                    "positive_heterogeneous_pair_count": sum(
                        value > 0.0 and pair_types.get(pair_name) == "heterogeneous"
                        for pair_name, value in pair_interactions.items()
                    ),
                    "pair_interactions_json": json.dumps(
                        pair_interactions, sort_keys=True, separators=(",", ":")
                    ),
                    "status": "ok" if point is not None else "incomplete",
                }
            )
    return rows


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


def generate_diagnostics(
    manifest_path: Path,
    output_dir: Path,
    *,
    anomaly_manifest_path: Path | None = None,
    bootstrap_samples: int = 5000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 20260718,
) -> Dict[str, Any]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < bootstrap_confidence < 1.0:
        raise ValueError("bootstrap_confidence must be between 0 and 1")
    manifest_path = manifest_path.resolve()
    if anomaly_manifest_path is not None:
        anomaly_manifest_path = anomaly_manifest_path.resolve()
    output_dir = output_dir.resolve()
    runs, comparisons, receiver_method, metadata = _load_manifest(
        manifest_path, anomaly_manifest_path
    )
    contract = metadata.get("report_contract", {})
    expected_task_rows = (
        contract.get("expected_task_rows", {}) if isinstance(contract, Mapping) else {}
    )
    if not isinstance(expected_task_rows, Mapping):
        raise ValueError("report_contract.expected_task_rows must be an object")
    expected_task_rows = {
        str(task): int(count) for task, count in expected_task_rows.items()
    }
    paired_rows, clusters = _comparison_statistics(
        runs,
        comparisons,
        expected_task_rows,
        bootstrap_samples=bootstrap_samples,
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_seed=bootstrap_seed,
    )
    hierarchical_rows = _hierarchical_rows(
        clusters,
        (
            metadata.get("pair_types", {})
            if isinstance(metadata.get("pair_types", {}), Mapping)
            else {}
        ),
        bootstrap_samples=bootstrap_samples,
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_seed=bootstrap_seed,
    )
    seed_rows = _seed_variance_rows(clusters)
    oracle_rows = _oracle_statistics(
        runs,
        receiver_method,
        metadata,
        bootstrap_samples=bootstrap_samples,
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_seed=bootstrap_seed,
    )
    ambiguity_rows = _ambiguity_statistics(
        runs,
        comparisons,
        metadata,
        bootstrap_samples=bootstrap_samples,
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_seed=bootstrap_seed,
    )
    summary: Dict[str, Any] = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "anomaly_manifest": (
            str(anomaly_manifest_path) if anomaly_manifest_path is not None else None
        ),
        "statistics": {
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_confidence": bootstrap_confidence,
            "bootstrap_seed": bootstrap_seed,
            "ordinary_bootstrap": (
                "Phase 1 implementation: pairs, then seeds within pair, then paired examples; "
                "equal pair and seed weighting"
            ),
            "ambiguity_interaction_bootstrap": (
                "pairs, then seeds within pair, then paired examples independently within "
                "fixed native-source high/low ambiguity strata"
            ),
            "seed_std_ddof": 1,
        },
        "paired_interventions": paired_rows,
        "hierarchical_interventions": hierarchical_rows,
        "seed_variance": seed_rows,
        "oracle_abstention": oracle_rows,
        "ambiguity_interactions": ambiguity_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("paired_interventions", paired_rows),
        ("hierarchical_interventions", hierarchical_rows),
        ("seed_variance", seed_rows),
        ("oracle_abstention", oracle_rows),
        ("ambiguity_interactions", ambiguity_rows),
    ):
        _write_csv(output_dir / f"{name}.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Phase 1.5 intervention contrasts, pair-balanced hierarchical "
            "bootstrap CIs, fixed-source ambiguity interactions, seed variance, and "
            "receiver/fused oracle-abstention headroom."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--anomaly-manifest",
        type=Path,
        help=(
            "Optional route1_phase1_5_qwen25_seed44_gate_anomaly execution "
            "manifest; adds two gate-component contrasts and their oracle rows"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    args = parser.parse_args()
    try:
        generate_diagnostics(
            args.manifest,
            args.output_dir,
            anomaly_manifest_path=args.anomaly_manifest,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_confidence=args.bootstrap_confidence,
            bootstrap_seed=args.bootstrap_seed,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"summary={args.output_dir.resolve() / 'summary.json'}")


if __name__ == "__main__":
    main()
