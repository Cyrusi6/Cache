from __future__ import annotations

"""Phase 2A-2A fit-only cache-geometry cross-fitted statistics.

The command has a hard stage boundary:

* ``freeze-join`` validates outcome-free geometry/output sidecars and reads only
  identity columns from the separate outcome CSV before freezing file hashes and
  the exact join key set.
* ``analyze`` verifies that freeze and only then parses receiver/fused
  correctness, runs the preregistered leave-one-pair/content-fold cross-fit, and
  evaluates the nine conjunctive GO gates.

The script is CPU-only and never imports torch or launches an experiment.
"""

import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
import csv
import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_PROTOCOL = REPO_ROOT / "recipe/eval_recipe/phase2a_2a/protocol_manifest.json"
DEFAULT_FEATURES = REPO_ROOT / "recipe/eval_recipe/phase2a_2a/feature_manifest.json"
DEFAULT_CANDIDATES = REPO_ROOT / "recipe/eval_recipe/phase2a_2a/candidate_manifest.json"
DEFAULT_SCHEMA = (
    REPO_ROOT / "recipe/eval_recipe/phase2a_2a/cache_geometry_artifact_schema.json"
)

JOIN_FIELDS: Tuple[str, ...] = (
    "pair",
    "seed",
    "task",
    "subject",
    "question_id",
    "content_hash",
)
OUTCOME_FIELDS: Tuple[str, ...] = (*JOIN_FIELDS, "receiver_correct", "fused_correct")
CLASS_ORDER = np.asarray([-1, 0, 1], dtype=np.int64)
SUMMARY_SUFFIXES: Tuple[str, ...] = (
    "all_mean",
    "all_std",
    "all_max",
    "early_mean",
    "middle_mean",
    "late_mean",
)
SCALAR_FEATURES: Tuple[str, ...] = (
    "source_receiver_length_ratio",
    "valid_alignment_mass",
    "valid_alignment_coverage",
)
FORBIDDEN_GEOMETRY_FIELDS = frozenset(
    {
        "true_answer",
        "is_correct",
        "receiver_correct",
        "fused_correct",
        "utility",
        "event",
        "label",
        "oracle_choice",
    }
)
_FROZEN_WEIGHT_WARNING_PREFIX = (
    "Since FrozenEstimator does not appear to accept sample_weight, sample weights "
    "will only be used for the calibration itself."
)

IdentityKey = Tuple[str, int, str, str, str, str]


@dataclass(frozen=True)
class GeometrySample:
    key: IdentityKey
    features: Tuple[float, ...]
    within_key_variation: bool
    within_value_variation: bool

    @property
    def pair(self) -> str:
        return self.key[0]

    @property
    def seed(self) -> int:
        return self.key[1]

    @property
    def task(self) -> str:
        return self.key[2]

    @property
    def subject(self) -> str:
        return self.key[3]

    @property
    def question_id(self) -> str:
        return self.key[4]

    @property
    def content_hash(self) -> str:
        return self.key[5]


@dataclass(frozen=True)
class Observation:
    geometry: GeometrySample
    receiver_correct: int
    fused_correct: int
    fold: int

    @property
    def key(self) -> IdentityKey:
        return self.geometry.key

    @property
    def pair(self) -> str:
        return self.geometry.pair

    @property
    def seed(self) -> int:
        return self.geometry.seed

    @property
    def task(self) -> str:
        return self.geometry.task

    @property
    def content_hash(self) -> str:
        return self.geometry.content_hash

    @property
    def features(self) -> Tuple[float, ...]:
        return self.geometry.features

    @property
    def utility(self) -> int:
        return self.fused_correct - self.receiver_correct


@dataclass(frozen=True)
class CrossfitPrediction:
    observation: Observation
    probabilities: Tuple[float, float, float]
    prior_probabilities: Tuple[float, float, float]
    score: float
    transfer: bool
    candidate_id: str

    @property
    def selector_correct(self) -> int:
        return (
            self.observation.fused_correct
            if self.transfer
            else self.observation.receiver_correct
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _path_record(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _verify_path_record(record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"])).resolve()
    expected = str(record["sha256"])
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected}")
    return path


def _identity(record: Mapping[str, Any], *, source: str) -> IdentityKey:
    missing = [field for field in JOIN_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Missing identity fields {missing} in {source}")
    content_hash = str(record["content_hash"]).lower()
    if len(content_hash) != 64 or any(ch not in "0123456789abcdef" for ch in content_hash):
        raise ValueError(f"Invalid content_hash in {source}: {content_hash!r}")
    return (
        str(record["pair"]),
        int(record["seed"]),
        str(record["task"]),
        str(record["subject"]),
        str(record["question_id"]),
        content_hash,
    )


def _key_digest(keys: Iterable[IdentityKey]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update("\t".join(map(str, key)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _iter_jsonl(paths: Sequence[Path]) -> Iterable[Tuple[Dict[str, Any], str]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}")
                yield value, f"{path}:{line_number}"


def _tolerance(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Variation audit received empty/non-finite values")
    return 1e-8 * max(1.0, float(np.max(np.abs(values))))


def _is_variable(values: Sequence[float]) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return float(np.max(array) - np.min(array)) > _tolerance(array)


def _resolve_summary_artifacts(
    summary_paths: Sequence[Path],
) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
    summaries: List[Path] = []
    layers: List[Path] = []
    samples_on: List[Path] = []
    samples_off: List[Path] = []
    for summary_path in summary_paths:
        summary_path = summary_path.resolve()
        summary = _read_json(summary_path)
        artifacts = summary.get("cache_geometry_artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError(f"Missing cache_geometry_artifacts in {summary_path}")
        role = str(artifacts.get("role"))
        if role not in {"geometry_on", "geometry_off"}:
            raise ValueError(f"Invalid cache geometry role in {summary_path}: {role}")

        def resolve_pointer(name: str, *, required: bool) -> Path | None:
            raw = artifacts.get(name)
            if raw is None:
                if required:
                    raise ValueError(f"Missing {name} for {role} in {summary_path}")
                return None
            path = Path(str(raw))
            if not path.is_absolute():
                path = summary_path.parent / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            return path

        sample_path = resolve_pointer("samples_jsonl", required=True)
        assert sample_path is not None
        if role == "geometry_on":
            layer_path = resolve_pointer("layers_jsonl", required=True)
            assert layer_path is not None
            layers.append(layer_path)
            samples_on.append(sample_path)
        else:
            if artifacts.get("layers_jsonl") not in {None, ""}:
                raise ValueError(f"geometry_off summary unexpectedly has layers in {summary_path}")
            samples_off.append(sample_path)
        summaries.append(summary_path)
    if not layers or not samples_on or not samples_off:
        raise ValueError("Need at least one geometry_on and one geometry_off summary")
    for values, label in (
        (summaries, "summaries"),
        (layers, "layer sidecars"),
        (samples_on, "on sample sidecars"),
        (samples_off, "off sample sidecars"),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {label}")
    return summaries, layers, samples_on, samples_off


def _load_output_fingerprints(
    paths: Sequence[Path], role: str
) -> Dict[IdentityKey, str]:
    output: Dict[IdentityKey, str] = {}
    expected_fields = {*JOIN_FIELDS, "schema_version", "role", "output_sha256"}
    for record, source in _iter_jsonl(paths):
        if set(record) != expected_fields:
            raise ValueError(
                f"Output fingerprint fields differ from frozen schema in {source}: "
                f"{sorted(set(record) ^ expected_fields)}"
            )
        if int(record["schema_version"]) != 1 or str(record["role"]) != role:
            raise ValueError(f"Output fingerprint role/schema mismatch in {source}")
        if FORBIDDEN_GEOMETRY_FIELDS.intersection(record):
            raise ValueError(f"Outcome field present in output sidecar: {source}")
        key = _identity(record, source=source)
        fingerprint = str(record["output_sha256"]).lower()
        if len(fingerprint) != 64 or any(
            ch not in "0123456789abcdef" for ch in fingerprint
        ):
            raise ValueError(f"Invalid output_sha256 in {source}")
        if key in output:
            raise ValueError(f"Duplicate output fingerprint key {key}")
        output[key] = fingerprint
    if not output:
        raise ValueError(f"No {role} output fingerprints")
    return output


def _summary_values(values: np.ndarray) -> Dict[str, float]:
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("Each sample needs at least three finite layer values")
    thirds = np.array_split(values, 3)
    return {
        "all_mean": float(np.mean(values)),
        "all_std": float(np.std(values, ddof=0)),
        "all_max": float(np.max(values)),
        "early_mean": float(np.mean(thirds[0])),
        "middle_mean": float(np.mean(thirds[1])),
        "late_mean": float(np.mean(thirds[2])),
    }


def _aggregate_geometry(
    paths: Sequence[Path], feature_manifest: Mapping[str, Any]
) -> Dict[IdentityKey, GeometrySample]:
    source_mapping = feature_manifest.get("per_layer_source_mapping")
    feature_order = tuple(map(str, feature_manifest.get("feature_order", [])))
    if not isinstance(source_mapping, Mapping) or not feature_order:
        raise ValueError("Invalid feature manifest")
    if int(feature_manifest.get("feature_count", -1)) != len(feature_order):
        raise ValueError("Feature manifest count mismatch")
    if len(feature_order) != len(set(feature_order)) or any(
        "*" in name for name in feature_order
    ):
        raise ValueError("Primary feature whitelist is duplicate or wildcarded")
    layer_bases = [
        str(name)
        for name in source_mapping
        if str(name) not in SCALAR_FEATURES
    ]
    scalar_bases = [name for name in SCALAR_FEATURES if name in source_mapping]
    expected_features = {
        f"{base}__{suffix}" for base in layer_bases for suffix in SUMMARY_SUFFIXES
    } | set(scalar_bases)
    if set(feature_order) != expected_features:
        raise ValueError("feature_order does not exactly match source mapping expansion")

    grouped: Dict[IdentityKey, List[Dict[str, Any]]] = {}
    seen_units: set[Tuple[IdentityKey, int, int]] = set()
    for record, source in _iter_jsonl(paths):
        if int(record.get("cache_geometry_schema_version", -1)) != 1:
            raise ValueError(f"Geometry schema mismatch in {source}")
        if str(record.get("role")) != "geometry_on":
            raise ValueError(f"Layer sidecar is not geometry_on in {source}")
        leaked = FORBIDDEN_GEOMETRY_FIELDS.intersection(record)
        if leaked:
            raise ValueError(f"Outcome fields {sorted(leaked)} present in {source}")
        key = _identity(record, source=source)
        projector_index = int(record.get("projector_index", -1))
        target_layer = int(record.get("target_layer", -1))
        batch_index = int(record.get("batch_index", -1))
        if projector_index < 0 or target_layer < 0 or batch_index < 0:
            raise ValueError(f"Invalid layer identity in {source}")
        unit = (key, projector_index, target_layer)
        if unit in seen_units:
            raise ValueError(f"Duplicate projector/layer record for {key}")
        seen_units.add(unit)
        grouped.setdefault(key, []).append(dict(record))
    if not grouped:
        raise ValueError("No geometry layer records")

    output: Dict[IdentityKey, GeometrySample] = {}
    for key, records in grouped.items():
        records.sort(
            key=lambda row: (
                int(row["target_layer"]),
                int(row["projector_index"]),
                int(row["batch_index"]),
            )
        )
        if len(records) < 3:
            raise ValueError(f"Sample {key} has fewer than three layer records")
        base_values: Dict[str, np.ndarray] = {}
        for base in layer_bases:
            source_name = str(source_mapping[base])
            values: List[float] = []
            if source_name.startswith("derived:"):
                if base != "residual_imbalance":
                    raise ValueError(f"Unsupported derived primary metric {base}")
                for row in records:
                    key_ratio = float(row["key_residual_to_native_norm_ratio"])
                    value_ratio = float(row["value_residual_to_native_norm_ratio"])
                    if key_ratio < 0.0 or value_ratio < 0.0:
                        raise ValueError(f"Negative residual ratio for {key}")
                    values.append(
                        abs(math.log((key_ratio + 1e-12) / (value_ratio + 1e-12)))
                    )
            else:
                for row in records:
                    raw = row.get(source_name)
                    if raw is None:
                        raise ValueError(f"Missing primary raw field {source_name} for {key}")
                    values.append(float(raw))
            array = np.asarray(values, dtype=np.float64)
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite primary raw field {base} for {key}")
            base_values[base] = array

        feature_values: Dict[str, float] = {}
        for base, values in base_values.items():
            for suffix, value in _summary_values(values).items():
                feature_values[f"{base}__{suffix}"] = value
        for name in scalar_bases:
            source_name = str(source_mapping[name])
            values = np.asarray(
                [float(row[source_name]) for row in records], dtype=np.float64
            )
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite scalar primary feature {name} for {key}")
            if float(np.max(values) - np.min(values)) > _tolerance(values):
                raise ValueError(f"Scalar primary feature {name} varies by layer for {key}")
            feature_values[name] = float(np.mean(values))
        if not 0.0 <= feature_values["valid_alignment_coverage"] <= 1.0:
            raise ValueError(f"valid_alignment_coverage out of range for {key}")
        if feature_values["source_receiver_length_ratio"] <= 0.0:
            raise ValueError(f"source_receiver_length_ratio must be positive for {key}")

        key_raw = [
            values
            for base, values in base_values.items()
            if base.startswith("key_")
        ]
        value_raw = [
            values
            for base, values in base_values.items()
            if base.startswith("value_")
        ]
        output[key] = GeometrySample(
            key=key,
            features=tuple(feature_values[name] for name in feature_order),
            within_key_variation=any(_is_variable(values) for values in key_raw),
            within_value_variation=any(_is_variable(values) for values in value_raw),
        )
    return output


def _geometry_audit(
    geometry: Mapping[IdentityKey, GeometrySample],
    feature_order: Sequence[str],
    primary_pairs: Sequence[str],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"pairs": {}}
    for pair in primary_pairs:
        samples = [item for item in geometry.values() if item.pair == pair]
        if not samples:
            raise ValueError(f"No geometry samples for pair {pair}")
        matrix = np.asarray([item.features for item in samples], dtype=np.float64)
        key_features = [
            name
            for index, name in enumerate(feature_order)
            if name.startswith("key_") and _is_variable(matrix[:, index])
        ]
        value_features = [
            name
            for index, name in enumerate(feature_order)
            if name.startswith("value_") and _is_variable(matrix[:, index])
        ]
        within_key = any(item.within_key_variation for item in samples)
        within_value = any(item.within_value_variation for item in samples)
        passed = bool(within_key and within_value and key_features and value_features)
        result["pairs"][pair] = {
            "n_samples": len(samples),
            "within_sample_key_variation": within_key,
            "within_sample_value_variation": within_value,
            "nonconstant_key_feature_count": len(key_features),
            "nonconstant_value_feature_count": len(value_features),
            "nonconstant_key_features": key_features,
            "nonconstant_value_features": value_features,
            "passed": passed,
        }
    result["all_pairs_passed"] = all(
        bool(value["passed"]) for value in result["pairs"].values()
    )
    return result


def _output_parity(
    on: Mapping[IdentityKey, str],
    off: Mapping[IdentityKey, str],
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    on_keys = set(on)
    off_keys = set(off)
    scope = protocol["canonical_output_fingerprint"][
        "matched_runtime_control_scope"
    ]
    expected_off = {
        key
        for key in on_keys
        if key[0] == str(scope["pair"])
        and key[1] == int(scope["seed"])
        and key[2] == str(scope["task"])
    }
    missing_off = sorted(expected_off - off_keys)
    unexpected_off = sorted(off_keys - expected_off)
    mismatched = sorted(key for key in expected_off & off_keys if on[key] != off[key])
    expected_count = int(scope["expected_rows"])
    return {
        "on_count": len(on),
        "off_count": len(off),
        "expected_off_count": expected_count,
        "missing_off_count": len(missing_off),
        "unexpected_off_count": len(unexpected_off),
        "mismatched_output_count": len(mismatched),
        "missing_off_examples": [list(key) for key in missing_off[:10]],
        "unexpected_off_examples": [list(key) for key in unexpected_off[:10]],
        "mismatched_examples": [list(key) for key in mismatched[:10]],
        "control_scope": dict(scope),
        "exact": (
            len(expected_off) == expected_count
            and len(off_keys) == expected_count
            and not missing_off
            and not unexpected_off
            and not mismatched
        ),
    }


def _load_fit_content(
    split_manifest_path: Path, protocol: Mapping[str, Any]
) -> Dict[str, set[str]]:
    expected = str(protocol["source"]["phase2a1_content_group_split_manifest_sha256"])
    if _sha256(split_manifest_path) != expected:
        raise ValueError("Phase2A-1 content-group split manifest SHA mismatch")
    manifest = _read_json(split_manifest_path)
    allowed: Dict[str, set[str]] = {}
    for group in manifest.get("groups", []):
        if str(group.get("split")) != str(protocol["source"]["permitted_split"]):
            continue
        content_hash = str(group["content_hash"])
        tasks = {str(member["task"]) for member in group.get("members", [])}
        allowed.setdefault(content_hash, set()).update(tasks)
    if not allowed:
        raise ValueError("No allowed Phase2A-1 fit content groups")
    return allowed


def _validate_scope(
    keys: Iterable[IdentityKey],
    protocol: Mapping[str, Any],
    allowed_content: Mapping[str, set[str]],
) -> None:
    keys = set(keys)
    expected_pairs = set(map(str, protocol["scope"]["primary_pairs"]))
    expected_seeds = set(map(int, protocol["scope"]["seeds"]))
    expected_tasks = set(map(str, protocol["scope"]["tasks"]))
    if {key[0] for key in keys} != expected_pairs:
        raise ValueError("Observed pair set differs from frozen primary pairs")
    if {key[1] for key in keys} != expected_seeds:
        raise ValueError("Observed seed set differs from frozen seeds")
    if {key[2] for key in keys} != expected_tasks:
        raise ValueError("Observed task set differs from frozen tasks")
    for key in keys:
        if key[5] not in allowed_content or key[2] not in allowed_content[key[5]]:
            raise ValueError(f"Non-fit or task-mismatched content group in input: {key}")
    expected_rows = protocol["source"].get("expected_fit_rows_by_task")
    if isinstance(expected_rows, Mapping):
        for pair in expected_pairs:
            for seed in expected_seeds:
                for task in expected_tasks:
                    actual = sum(
                        key[0] == pair and key[1] == seed and key[2] == task
                        for key in keys
                    )
                    expected = int(expected_rows[task])
                    if actual != expected:
                        raise ValueError(
                            f"Row count mismatch for {pair}/{seed}/{task}: "
                            f"{actual} != {expected}"
                        )


def _strict_bool(value: Any, *, source: str) -> int:
    if isinstance(value, bool):
        return int(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return 1
    if normalized in {"false", "0", "no"}:
        return 0
    raise ValueError(f"Invalid boolean {value!r} in {source}")


def _load_outcomes(
    path: Path, *, parse_values: bool
) -> Dict[IdentityKey, Tuple[int, int] | None]:
    output: Dict[IdentityKey, Tuple[int, int] | None] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != set(OUTCOME_FIELDS):
            raise ValueError(
                f"Outcome CSV fields must exactly equal {list(OUTCOME_FIELDS)}"
            )
        for row_number, row in enumerate(reader, start=2):
            source = f"{path}:{row_number}"
            key = _identity(row, source=source)
            if key in output:
                raise ValueError(f"Duplicate outcome key {key}")
            if parse_values:
                output[key] = (
                    _strict_bool(row["receiver_correct"], source=source),
                    _strict_bool(row["fused_correct"], source=source),
                )
            else:
                output[key] = None
    if not output:
        raise ValueError("No outcome rows")
    return output


def _content_fold(content_hash: str, protocol: Mapping[str, Any]) -> int:
    prefix = str(protocol["fold"]["outer_prefix"])
    dataset_hash = str(protocol["source"]["dataset_content_sha256"])
    digest = hashlib.sha256(
        f"{prefix}|{dataset_hash}|{content_hash}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % int(
        protocol["fold"]["count"]
    )


def _development_role(content_hash: str, protocol: Mapping[str, Any]) -> str:
    config = protocol["fold"]["development"]
    prefix = str(config["prefix"])
    dataset_hash = str(protocol["source"]["dataset_content_sha256"])
    digest = hashlib.sha256(
        f"{prefix}|{dataset_hash}|{content_hash}".encode("utf-8")
    ).digest()
    fraction = int.from_bytes(digest[:8], "big", signed=False) / 2**64
    if fraction < float(config["fit_interval"][1]):
        return "fit"
    if fraction < float(config["calibration_interval"][1]):
        return "calibration"
    return "model_selection"


def _verify_design_bundle(
    protocol_path: Path,
    feature_path: Path,
    candidate_path: Path,
    schema_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    protocol = _read_json(protocol_path)
    features = _read_json(feature_path)
    candidates = _read_json(candidate_path)
    if protocol.get("phase") != "2A-2A":
        raise ValueError("Wrong protocol phase")
    prereg_record = protocol["design_inputs"]["preregistration"]
    prereg_path = REPO_ROOT / str(prereg_record["path"])
    if str(prereg_record["sha256"]) != _sha256(prereg_path):
        raise ValueError("Frozen preregistration SHA mismatch")
    expected_paths = {
        "feature_manifest": feature_path,
        "candidate_manifest": candidate_path,
        "artifact_schema": schema_path,
    }
    for name, path in expected_paths.items():
        record = protocol["design_inputs"][name]
        if str(record["sha256"]) != _sha256(path):
            raise ValueError(f"Frozen {name} SHA mismatch")
    feature_order = list(map(str, features["feature_order"]))
    if int(features["feature_count"]) != len(feature_order):
        raise ValueError("Feature count mismatch")
    blocks = candidates["candidates"]
    for family in (
        "single_feature_stumps",
        "l2_multinomial_logistic",
        "depth2_trees",
    ):
        if list(map(str, blocks[family]["explicit_features"])) != feature_order:
            raise ValueError(f"Candidate family {family} feature freeze mismatch")
    expected_candidate_count = (
        len(feature_order)
        + len(blocks["l2_multinomial_logistic"]["C"])
        + len(blocks["depth2_trees"]["min_weight_fraction_leaf"])
    )
    if int(candidates["candidate_count"]) != expected_candidate_count:
        raise ValueError("Candidate count mismatch")
    return protocol, features, candidates


def _validate_equivalence_contract(
    execution_manifest_path: Path,
    equivalence_report_path: Path,
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    execution = _read_json(execution_manifest_path)
    report = _read_json(equivalence_report_path)
    constraints = execution.get("constraints", {})
    if (
        execution.get("role")
        != protocol["reference_equivalence"]["execution_manifest_role"]
        or constraints.get("allowed_seed") != [42]
        or constraints.get("allowed_split") != "fit"
    ):
        raise ValueError("Execution manifest is outside the frozen pilot scope")
    runs = execution.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("Execution manifest runs are invalid")
    by_id: Dict[str, Mapping[str, Any]] = {}
    for run in runs:
        run_id = str(run["id"])
        if run_id in by_id:
            raise ValueError(f"Duplicate execution run id {run_id}")
        by_id[run_id] = run
    expected_pairs = set(map(str, protocol["scope"]["primary_pairs"]))
    expected_tasks = set(map(str, protocol["scope"]["tasks"]))
    instrumented = [run for run in runs if run.get("kind") == "instrumented"]
    instrumented_scope = {
        (str(run["pair"]), int(run["seed"]), str(run["dataset"]))
        for run in instrumented
    }
    expected_scope = {
        (pair, 42, task) for pair in expected_pairs for task in expected_tasks
    }
    controls = [run for run in runs if run.get("kind") == "overhead_control"]
    control_scope = protocol["canonical_output_fingerprint"][
        "matched_runtime_control_scope"
    ]
    if instrumented_scope != expected_scope or len(instrumented) != int(
        protocol["reference_equivalence"]["instrumented_cell_count"]
    ):
        raise ValueError("Execution manifest does not contain the frozen nine cells")
    if len(controls) != 1 or (
        str(controls[0]["pair"]), int(controls[0]["seed"]), str(controls[0]["dataset"])
    ) != (
        str(control_scope["pair"]),
        int(control_scope["seed"]),
        str(control_scope["task"]),
    ):
        raise ValueError("Execution manifest matched off control is invalid")
    for run in runs:
        reference = run.get("reference_prediction")
        if not isinstance(reference, Mapping):
            raise ValueError(f"Missing frozen reference for {run['id']}")
        _verify_path_record(reference)

    comparisons = report.get("comparisons")
    if (
        report.get("phase") != protocol["reference_equivalence"]["verify_report_phase"]
        or report.get("instrumentation_output_exact") is not True
        or not isinstance(comparisons, list)
        or len(comparisons) != len(runs)
    ):
        raise ValueError("Equivalence report is incomplete or failed")
    expected_columns = list(protocol["reference_equivalence"]["exact_columns"])
    seen: set[str] = set()
    expected_rows = protocol["source"]["expected_fit_rows_by_task"]
    for comparison in comparisons:
        run_id = str(comparison["run_id"])
        if run_id in seen or run_id not in by_id:
            raise ValueError(f"Invalid equivalence comparison id {run_id}")
        seen.add(run_id)
        run = by_id[run_id]
        task = str(run["dataset"])
        if (
            comparison.get("kind") != run.get("kind")
            or comparison.get("exact") is not True
            or comparison.get("keys_exact") is not True
            or list(comparison.get("exact_columns", [])) != expected_columns
            or int(comparison.get("rows", -1)) != int(expected_rows[task])
            or int(comparison.get("expected_rows", -1)) != int(expected_rows[task])
            or int(comparison.get("mismatch_count_capped", -1)) != 0
        ):
            raise ValueError(f"Equivalence comparison failed contract for {run_id}")
        prediction_path = Path(str(comparison["prediction_path"])).resolve()
        if _sha256(prediction_path) != str(comparison["prediction_sha256"]):
            raise ValueError(f"Instrumented prediction changed after verify: {run_id}")
    if seen != set(by_id):
        raise ValueError("Equivalence report run set differs from execution manifest")
    return {
        "instrumented_comparison_count": len(instrumented),
        "matched_off_control_count": len(controls),
        "comparison_count": len(comparisons),
        "exact_columns": expected_columns,
        "all_exact": True,
    }


def freeze_join(
    *,
    summary_paths: Sequence[Path],
    execution_manifest_path: Path,
    equivalence_report_path: Path,
    outcomes_path: Path,
    output_path: Path,
    protocol_path: Path = DEFAULT_PROTOCOL,
    feature_path: Path = DEFAULT_FEATURES,
    candidate_path: Path = DEFAULT_CANDIDATES,
    schema_path: Path = DEFAULT_SCHEMA,
) -> Dict[str, Any]:
    protocol, features, _candidates = _verify_design_bundle(
        protocol_path, feature_path, candidate_path, schema_path
    )
    summaries, layer_paths, on_paths, off_paths = _resolve_summary_artifacts(
        summary_paths
    )
    equivalence = _validate_equivalence_contract(
        execution_manifest_path, equivalence_report_path, protocol
    )
    geometry = _aggregate_geometry(layer_paths, features)
    on_outputs = _load_output_fingerprints(on_paths, "geometry_on")
    off_outputs = _load_output_fingerprints(off_paths, "geometry_off")
    parity = _output_parity(on_outputs, off_outputs, protocol)
    outcome_identities = _load_outcomes(outcomes_path, parse_values=False)
    key_sets = {
        "geometry": set(geometry),
        "geometry_on_outputs": set(on_outputs),
        "outcomes": set(outcome_identities),
    }
    reference = key_sets["geometry"]
    for name, keys in key_sets.items():
        if keys != reference:
            raise ValueError(
                f"Frozen join key-set mismatch for {name}: "
                f"missing={len(reference - keys)} extra={len(keys - reference)}"
            )
    split_path = REPO_ROOT / str(
        protocol["source"]["phase2a1_content_group_split_manifest"]
    )
    allowed_content = _load_fit_content(split_path, protocol)
    _validate_scope(reference, protocol, allowed_content)
    geometry_audit = _geometry_audit(
        geometry, features["feature_order"], protocol["scope"]["primary_pairs"]
    )
    fold_counts = {str(index): 0 for index in range(5)}
    for key in reference:
        fold_counts[str(_content_fold(key[5], protocol))] += 1
    manifest = {
        "schema_version": 1,
        "phase": "2A-2A",
        "role": "frozen_geometry_outcome_join",
        "created_without_outcome_parse": True,
        "join_key": list(JOIN_FIELDS),
        "key_count": len(reference),
        "key_digest_sha256": _key_digest(reference),
        "inputs": {
            "evaluator_summaries": [_path_record(path) for path in summaries],
            "execution_manifest": _path_record(execution_manifest_path),
            "equivalence_report": _path_record(equivalence_report_path),
            "geometry_layers": [_path_record(path) for path in layer_paths],
            "geometry_on_outputs": [_path_record(path) for path in on_paths],
            "geometry_off_outputs": [_path_record(path) for path in off_paths],
            "outcomes": _path_record(outcomes_path),
            "phase2a1_split_manifest": _path_record(split_path),
        },
        "design": {
            "protocol": _path_record(protocol_path),
            "features": _path_record(feature_path),
            "candidates": _path_record(candidate_path),
            "schema": _path_record(schema_path),
            "implementation": _path_record(SCRIPT_PATH),
        },
        "outcome_access_audit": {
            "identity_rows_read": len(outcome_identities),
            "correctness_rows_parsed": 0,
        },
        "reference_equivalence": equivalence,
        "output_parity": parity,
        "geometry_variation_audit": geometry_audit,
        "fold_counts": fold_counts,
    }
    _write_json_once(output_path, manifest)
    return manifest


def _pair_task_weights(observations: Sequence[Observation]) -> np.ndarray:
    if not observations:
        raise ValueError("Cannot weight empty observations")
    counts: Dict[Tuple[str, str], int] = {}
    for item in observations:
        cell = (item.pair, item.task)
        counts[cell] = counts.get(cell, 0) + 1
    expected = {
        (pair, task)
        for pair in {item.pair for item in observations}
        for task in {item.task for item in observations}
    }
    if set(counts) != expected:
        raise ValueError("Active pair/task cells are not a full Cartesian product")
    total = len(observations)
    weights = np.asarray(
        [total / (len(counts) * counts[(item.pair, item.task)]) for item in observations],
        dtype=np.float64,
    )
    if not math.isclose(float(np.mean(weights)), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("Pair-task weights must have mean one")
    masses: Dict[Tuple[str, str], float] = {}
    for item, weight in zip(observations, weights):
        cell = (item.pair, item.task)
        masses[cell] = masses.get(cell, 0.0) + float(weight)
    if max(masses.values()) - min(masses.values()) > 1e-9:
        raise AssertionError("Pair-task cells do not have equal mass")
    return weights


def _arrays(observations: Sequence[Observation]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray([item.features for item in observations], dtype=np.float64)
    y = np.asarray([item.utility for item in observations], dtype=np.int64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Invalid feature matrix")
    return x, y


def _expand_candidates(
    manifest: Mapping[str, Any], feature_order: Sequence[str]
) -> List[Dict[str, Any]]:
    blocks = manifest["candidates"]
    candidates: List[Dict[str, Any]] = []
    ordinal = 0
    stump = blocks["single_feature_stumps"]
    for feature in stump["explicit_features"]:
        candidates.append(
            {
                "id": f"stump__{feature}",
                "ordinal": ordinal,
                "family": "single_feature_stump",
                "features": [str(feature)],
                "params": dict(stump["params"]),
            }
        )
        ordinal += 1
    logistic = blocks["l2_multinomial_logistic"]
    for c_value in logistic["C"]:
        params = dict(logistic["params"])
        params["C"] = float(c_value)
        candidates.append(
            {
                "id": f"logreg_l2_c{float(c_value):g}",
                "ordinal": ordinal,
                "family": "l2_multinomial_logistic",
                "features": list(map(str, logistic["explicit_features"])),
                "params": params,
            }
        )
        ordinal += 1
    trees = blocks["depth2_trees"]
    for leaf in trees["min_weight_fraction_leaf"]:
        params = dict(trees["params"])
        params["min_weight_fraction_leaf"] = float(leaf)
        candidates.append(
            {
                "id": f"tree_depth2_leaf{float(leaf):g}",
                "ordinal": ordinal,
                "family": "shallow_decision_tree",
                "features": list(map(str, trees["explicit_features"])),
                "params": params,
            }
        )
        ordinal += 1
    if len(candidates) != int(manifest["candidate_count"]):
        raise ValueError("Expanded candidate count differs from frozen manifest")
    if set(feature_order) != set(blocks["single_feature_stumps"]["explicit_features"]):
        raise ValueError("Candidate feature set differs from feature manifest")
    return candidates


def _base_estimator(
    candidate: Mapping[str, Any], feature_order: Sequence[str]
) -> Pipeline:
    feature_index = {name: index for index, name in enumerate(feature_order)}
    indices = [feature_index[str(name)] for name in candidate["features"]]
    select = ColumnTransformer(
        [("selected", "passthrough", indices)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    params = candidate["params"]
    if candidate["family"] == "l2_multinomial_logistic":
        classifier = LogisticRegression(
            C=float(params["C"]),
            penalty=str(params["penalty"]),
            solver=str(params["solver"]),
            tol=float(params["tol"]),
            max_iter=int(params["max_iter"]),
            fit_intercept=bool(params["fit_intercept"]),
            class_weight=params["class_weight"],
            random_state=int(params["random_state"]),
        )
        return Pipeline(
            [("select", select), ("scale", StandardScaler()), ("clf", classifier)]
        )
    classifier = DecisionTreeClassifier(
        criterion=str(params["criterion"]),
        splitter=str(params["splitter"]),
        max_depth=int(params["max_depth"]),
        min_weight_fraction_leaf=float(params["min_weight_fraction_leaf"]),
        class_weight=params["class_weight"],
        random_state=int(params["random_state"]),
    )
    return Pipeline([("select", select), ("clf", classifier)])


def _fit_model(
    candidate: Mapping[str, Any],
    feature_order: Sequence[str],
    fit_observations: Sequence[Observation],
    calibration_observations: Sequence[Observation],
) -> CalibratedClassifierCV:
    x_fit, y_fit = _arrays(fit_observations)
    x_cal, y_cal = _arrays(calibration_observations)
    if not np.array_equal(np.unique(y_fit), CLASS_ORDER):
        raise ValueError(f"Fit role lacks all classes for {candidate['id']}")
    if not np.array_equal(np.unique(y_cal), CLASS_ORDER):
        raise ValueError(f"Calibration role lacks all classes for {candidate['id']}")
    fit_weights = _pair_task_weights(fit_observations)
    calibration_weights = _pair_task_weights(calibration_observations)
    base = _base_estimator(candidate, feature_order)
    fit_params: Dict[str, Any] = {"clf__sample_weight": fit_weights}
    if candidate["family"] == "l2_multinomial_logistic":
        fit_params["scale__sample_weight"] = fit_weights
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        base.fit(x_fit, y_fit, **fit_params)
    convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    if convergence:
        raise ValueError(f"ConvergenceWarning for {candidate['id']}")
    if [item for item in caught if not issubclass(item.category, ConvergenceWarning)]:
        raise ValueError(f"Unexpected fit warning for {candidate['id']}")
    if not np.array_equal(np.asarray(base.classes_, dtype=np.int64), CLASS_ORDER):
        raise ValueError("Base estimator class order mismatch")
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(base), method="sigmoid", cv=None, ensemble=False, n_jobs=1
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calibrated.fit(x_cal, y_cal, sample_weight=calibration_weights)
    unexpected = [
        str(item.message)
        for item in caught
        if not str(item.message).startswith(_FROZEN_WEIGHT_WARNING_PREFIX)
    ]
    if unexpected:
        raise ValueError(f"Unexpected calibration warnings: {unexpected}")
    if not np.array_equal(np.asarray(calibrated.classes_, dtype=np.int64), CLASS_ORDER):
        raise ValueError("Calibrated estimator class order mismatch")
    return calibrated


def _predict_probabilities(model: Any, x: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=np.int64)
    if not np.array_equal(classes, CLASS_ORDER):
        positions = {int(value): index for index, value in enumerate(classes)}
        if set(positions) != {-1, 0, 1}:
            raise ValueError(f"Invalid probability class order: {classes}")
        probabilities = probabilities[:, [positions[-1], positions[0], positions[1]]]
    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != 3
        or not np.isfinite(probabilities).all()
        or (probabilities < 0.0).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-10)
    ):
        raise ValueError("Invalid calibrated probabilities")
    return probabilities


def _score(probabilities: np.ndarray) -> np.ndarray:
    return probabilities[:, 2] - probabilities[:, 0]


def _policy_accuracy(
    observations: Sequence[Observation], actions: np.ndarray, weights: np.ndarray
) -> float:
    receiver = np.asarray([item.receiver_correct for item in observations], dtype=float)
    utility = np.asarray([item.utility for item in observations], dtype=float)
    return float(np.average(receiver + actions.astype(float) * utility, weights=weights))


def _select_threshold(
    observations: Sequence[Observation],
    probabilities: np.ndarray,
    tolerance: float,
) -> Tuple[float, float]:
    scores = _score(probabilities)
    thresholds = np.concatenate(([-math.inf], np.unique(scores), [math.inf]))
    weights = _pair_task_weights(observations)
    best_threshold = -math.inf
    best_accuracy = -math.inf
    for threshold in thresholds:
        accuracy = _policy_accuracy(observations, scores > threshold, weights)
        if accuracy > best_accuracy + tolerance or (
            abs(accuracy - best_accuracy) <= tolerance and threshold > best_threshold
        ):
            best_threshold = float(threshold)
            best_accuracy = accuracy
    return best_threshold, best_accuracy


def _serialize_threshold(value: float) -> Dict[str, Any]:
    if value == -math.inf:
        return {"kind": "always_fused"}
    if value == math.inf:
        return {"kind": "always_receiver"}
    return {"kind": "finite", "float_hex": float(value).hex()}


def _fit_select_candidate(
    candidates: Sequence[Mapping[str, Any]],
    feature_order: Sequence[str],
    fit_observations: Sequence[Observation],
    calibration_observations: Sequence[Observation],
    selection_observations: Sequence[Observation],
    tolerance: float,
) -> Tuple[Mapping[str, Any], CalibratedClassifierCV, float, List[Dict[str, Any]]]:
    x_cal, _ = _arrays(calibration_observations)
    x_select, _ = _arrays(selection_observations)
    selection_weights = _pair_task_weights(selection_observations)
    fitted: List[Tuple[Mapping[str, Any], CalibratedClassifierCV, float, float]] = []
    summaries: List[Dict[str, Any]] = []
    for candidate in candidates:
        model = _fit_model(
            candidate, feature_order, fit_observations, calibration_observations
        )
        threshold, calibration_accuracy = _select_threshold(
            calibration_observations,
            _predict_probabilities(model, x_cal),
            tolerance,
        )
        selection_probabilities = _predict_probabilities(model, x_select)
        selection_accuracy = _policy_accuracy(
            selection_observations,
            _score(selection_probabilities) > threshold,
            selection_weights,
        )
        fitted.append((candidate, model, threshold, selection_accuracy))
        summaries.append(
            {
                "candidate_id": candidate["id"],
                "ordinal": int(candidate["ordinal"]),
                "threshold": _serialize_threshold(threshold),
                "calibration_accuracy": calibration_accuracy,
                "model_selection_accuracy": selection_accuracy,
            }
        )
    best = 0
    for index in range(1, len(fitted)):
        current = fitted[index]
        incumbent = fitted[best]
        if current[3] > incumbent[3] + tolerance or (
            abs(current[3] - incumbent[3]) <= tolerance
            and int(current[0]["ordinal"]) < int(incumbent[0]["ordinal"])
        ):
            best = index
    candidate, model, threshold, _accuracy = fitted[best]
    return candidate, model, threshold, summaries


def _fit_prior(observations: Sequence[Observation]) -> np.ndarray:
    weights = _pair_task_weights(observations)
    utility = np.asarray([item.utility for item in observations], dtype=np.int64)
    prior = np.asarray(
        [float(np.average(utility == target, weights=weights)) for target in CLASS_ORDER],
        dtype=np.float64,
    )
    if not np.isfinite(prior).all() or not math.isclose(
        float(prior.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Invalid cross-fitted constant prior")
    return prior


def _crossfit(
    observations: Sequence[Observation],
    feature_order: Sequence[str],
    candidate_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> Tuple[List[CrossfitPrediction], List[Dict[str, Any]]]:
    candidates = _expand_candidates(candidate_manifest, feature_order)
    primary_pairs = list(map(str, protocol["scope"]["primary_pairs"]))
    fold_count = int(protocol["fold"]["count"])
    tolerance = float(protocol["calibration_and_selection"]["threshold_tolerance"])
    predictions: List[CrossfitPrediction] = []
    audits: List[Dict[str, Any]] = []
    for held_pair in primary_pairs:
        for evaluation_fold in range(fold_count):
            evaluation = [
                item
                for item in observations
                if item.pair == held_pair and item.fold == evaluation_fold
            ]
            development = [
                item
                for item in observations
                if item.pair != held_pair and item.fold != evaluation_fold
            ]
            fit = [
                item
                for item in development
                if _development_role(item.content_hash, protocol) == "fit"
            ]
            calibration = [
                item
                for item in development
                if _development_role(item.content_hash, protocol) == "calibration"
            ]
            selection = [
                item
                for item in development
                if _development_role(item.content_hash, protocol) == "model_selection"
            ]
            if not evaluation or not fit or not calibration or not selection:
                raise ValueError(f"Empty crossfit role for {held_pair}/fold {evaluation_fold}")
            evaluation_content = {item.content_hash for item in evaluation}
            development_content = {
                item.content_hash for item in (*fit, *calibration, *selection)
            }
            overlap = evaluation_content & development_content
            if overlap:
                raise AssertionError("Evaluation content leaked into development")
            if any(item.pair == held_pair for item in (*fit, *calibration, *selection)):
                raise AssertionError("Held pair leaked into development")
            candidate, model, threshold, candidate_summaries = _fit_select_candidate(
                candidates,
                feature_order,
                fit,
                calibration,
                selection,
                tolerance,
            )
            x_evaluation, _ = _arrays(evaluation)
            probabilities = _predict_probabilities(model, x_evaluation)
            scores = _score(probabilities)
            actions = scores > threshold
            prior = _fit_prior(fit)
            for item, probability, score, action in zip(
                evaluation, probabilities, scores, actions
            ):
                predictions.append(
                    CrossfitPrediction(
                        observation=item,
                        probabilities=tuple(map(float, probability)),
                        prior_probabilities=tuple(map(float, prior)),
                        score=float(score),
                        transfer=bool(action),
                        candidate_id=str(candidate["id"]),
                    )
                )
            audits.append(
                {
                    "held_out_pair": held_pair,
                    "evaluation_fold": evaluation_fold,
                    "outer_excluded_fold": evaluation_fold,
                    "development_hash_split": {
                        "fit": [0.0, 0.6],
                        "calibration": [0.6, 0.8],
                        "model_selection": [0.8, 1.0],
                    },
                    "fit_pairs": sorted({item.pair for item in fit}),
                    "fit_rows": len(fit),
                    "calibration_rows": len(calibration),
                    "model_selection_rows": len(selection),
                    "evaluation_rows": len(evaluation),
                    "content_overlap_count": len(overlap),
                    "selected_candidate_id": candidate["id"],
                    "selected_threshold": _serialize_threshold(threshold),
                    "candidate_summaries": candidate_summaries,
                }
            )
    if len(predictions) != len(observations):
        raise AssertionError(
            f"Crossfit coverage mismatch: {len(predictions)} != {len(observations)}"
        )
    keys = [item.observation.key for item in predictions]
    if len(keys) != len(set(keys)) or set(keys) != {item.key for item in observations}:
        raise AssertionError("Crossfit predictions are duplicate or incomplete")
    predictions.sort(key=lambda item: item.observation.key)
    return predictions, audits


def _safe_auprc(
    target: np.ndarray, score: np.ndarray, weights: np.ndarray
) -> float | None:
    if float(weights[target == 1].sum()) <= 0.0 or float(
        weights[target == 0].sum()
    ) <= 0.0:
        return None
    return float(average_precision_score(target, score, sample_weight=weights))


def _prediction_metrics(
    predictions: Sequence[CrossfitPrediction],
    pair: str | None = None,
    task: str | None = None,
) -> Dict[str, Any]:
    selected = [
        item
        for item in predictions
        if (pair is None or item.observation.pair == pair)
        and (task is None or item.observation.task == task)
    ]
    observations = [item.observation for item in selected]
    if not observations:
        raise ValueError(f"No predictions for pair={pair}, task={task}")
    weights = _pair_task_weights(observations)
    utility = np.asarray([item.utility for item in observations], dtype=np.int64)
    receiver = np.asarray([item.receiver_correct for item in observations], dtype=float)
    fused = np.asarray([item.fused_correct for item in observations], dtype=float)
    action = np.asarray([item.transfer for item in selected], dtype=bool)
    selector = receiver + action.astype(float) * utility
    probabilities = np.asarray([item.probabilities for item in selected], dtype=float)
    prior = np.asarray([item.prior_probabilities for item in selected], dtype=float)
    one_hot = np.column_stack([utility == target for target in CLASS_ORDER]).astype(float)
    harm = utility == -1
    benefit = utility == 1
    harm_rate = float(np.average(harm, weights=weights))
    benefit_rate = float(np.average(benefit, weights=weights))
    accepted_harm = float(np.average(action & harm, weights=weights))
    accepted_benefit = float(np.average(action & benefit, weights=weights))
    harm_auprc = _safe_auprc(
        harm.astype(np.int8), probabilities[:, 0], weights
    )
    benefit_auprc = _safe_auprc(
        benefit.astype(np.int8), probabilities[:, 2], weights
    )
    return {
        "pair": pair or "__all__",
        "task": task or "__all__",
        "n_rows": len(selected),
        "n_content_groups": len({item.content_hash for item in observations}),
        "harm_prevalence": harm_rate,
        "harm_auprc": harm_auprc,
        "benefit_prevalence": benefit_rate,
        "benefit_auprc": benefit_auprc,
        "selector_accuracy": float(np.average(selector, weights=weights)),
        "fused_accuracy": float(np.average(fused, weights=weights)),
        "receiver_accuracy": float(np.average(receiver, weights=weights)),
        "selector_minus_fused": float(np.average(selector - fused, weights=weights)),
        "transfer_rate": float(np.average(action, weights=weights)),
        "harmful_reduction": (
            None if harm_rate <= 0.0 else 1.0 - accepted_harm / harm_rate
        ),
        "beneficial_retention": (
            None if benefit_rate <= 0.0 else accepted_benefit / benefit_rate
        ),
        "multiclass_brier": float(
            np.average(np.square(probabilities - one_hot).sum(axis=1), weights=weights)
        ),
        "crossfit_prior_brier": float(
            np.average(np.square(prior - one_hot).sum(axis=1), weights=weights)
        ),
        "binary_harm_brier": float(
            np.average(np.square(probabilities[:, 0] - harm.astype(float)), weights=weights)
        ),
        "crossfit_harm_prior_brier": float(
            np.average(np.square(prior[:, 0] - harm.astype(float)), weights=weights)
        ),
    }


def evaluate_go_gates(
    *,
    pooled_metrics: Mapping[str, Any],
    pair_metrics: Mapping[str, Mapping[str, Any]],
    reference_equivalence: Mapping[str, Any],
    output_parity: Mapping[str, Any],
    geometry_audit: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    config = protocol["go_gate"]
    pooled_auprc = pooled_metrics.get("harm_auprc")
    harm_reduction = pooled_metrics.get("harmful_reduction")
    benefit_retention = pooled_metrics.get("beneficial_retention")
    pair_auprc_passes = sum(
        metric.get("harm_auprc") is not None
        and float(metric["harm_auprc"]) > float(metric["harm_prevalence"])
        for metric in pair_metrics.values()
    )
    every_pair_delta = min(
        float(metric["selector_minus_fused"]) for metric in pair_metrics.values()
    )
    gates = {
        "on_off_output_exact": bool(reference_equivalence.get("all_exact"))
        and bool(output_parity.get("exact")),
        "real_geometry_variation_nonconstant": bool(
            geometry_audit.get("all_pairs_passed")
        ),
        "pooled_harm_auprc_margin": (
            pooled_auprc is not None
            and float(pooled_auprc)
            >= float(pooled_metrics["harm_prevalence"])
            + float(config["pooled_harm_auprc_minimum_margin_over_prevalence"])
        ),
        "held_out_pair_harm_auprc": pair_auprc_passes
        >= int(config["minimum_pairs_harm_auprc_above_own_prevalence"]),
        "selector_accuracy_gain": float(pooled_metrics["selector_minus_fused"])
        >= float(config["minimum_selector_minus_always_fused"]),
        "harmful_reduction": harm_reduction is not None
        and float(harm_reduction) >= float(config["minimum_harmful_reduction"]),
        "beneficial_retention": benefit_retention is not None
        and float(benefit_retention) >= float(config["minimum_beneficial_retention"]),
        "every_pair_noninferiority": every_pair_delta
        >= float(config["minimum_each_pair_selector_minus_fused"]),
        "brier_beats_crossfit_prior": float(pooled_metrics["multiclass_brier"])
        < float(pooled_metrics["crossfit_prior_brier"]),
    }
    expected_order = list(config["gates_in_order"])
    if list(gates) != expected_order or len(gates) != 9:
        raise AssertionError("GO gate implementation differs from frozen nine-gate order")
    return {
        "all_conjunctive": True,
        "decision": "GO" if all(gates.values()) else "NO_GO",
        "gates": gates,
        "values": {
            "pooled_harm_auprc": pooled_auprc,
            "pooled_harm_prevalence": pooled_metrics["harm_prevalence"],
            "pair_harm_auprc_above_prevalence_count": pair_auprc_passes,
            "selector_minus_fused": pooled_metrics["selector_minus_fused"],
            "harmful_reduction": harm_reduction,
            "beneficial_retention": benefit_retention,
            "minimum_pair_selector_minus_fused": every_pair_delta,
            "multiclass_brier": pooled_metrics["multiclass_brier"],
            "crossfit_prior_brier": pooled_metrics["crossfit_prior_brier"],
        },
    }


def analyze(
    *, join_manifest_path: Path, output_path: Path
) -> Dict[str, Any]:
    join_manifest = _read_json(join_manifest_path)
    if (
        join_manifest.get("phase") != "2A-2A"
        or join_manifest.get("role") != "frozen_geometry_outcome_join"
        or join_manifest.get("created_without_outcome_parse") is not True
        or tuple(join_manifest.get("join_key", [])) != JOIN_FIELDS
    ):
        raise ValueError("Invalid frozen join manifest")
    inputs = join_manifest["inputs"]
    design = join_manifest["design"]
    protocol_path = _verify_path_record(design["protocol"])
    feature_path = _verify_path_record(design["features"])
    candidate_path = _verify_path_record(design["candidates"])
    schema_path = _verify_path_record(design["schema"])
    _verify_path_record(design["implementation"])
    protocol, features, candidates = _verify_design_bundle(
        protocol_path, feature_path, candidate_path, schema_path
    )
    summary_paths = [_verify_path_record(item) for item in inputs["evaluator_summaries"]]
    execution_manifest_path = _verify_path_record(inputs["execution_manifest"])
    equivalence_report_path = _verify_path_record(inputs["equivalence_report"])
    layer_paths = [_verify_path_record(item) for item in inputs["geometry_layers"]]
    on_paths = [_verify_path_record(item) for item in inputs["geometry_on_outputs"]]
    off_paths = [_verify_path_record(item) for item in inputs["geometry_off_outputs"]]
    outcomes_path = _verify_path_record(inputs["outcomes"])
    split_path = _verify_path_record(inputs["phase2a1_split_manifest"])
    discovered = _resolve_summary_artifacts(summary_paths)
    if (
        set(discovered[1]) != set(layer_paths)
        or set(discovered[2]) != set(on_paths)
        or set(discovered[3]) != set(off_paths)
    ):
        raise ValueError("Evaluator summary pointers changed after join freeze")
    geometry = _aggregate_geometry(layer_paths, features)
    equivalence = _validate_equivalence_contract(
        execution_manifest_path, equivalence_report_path, protocol
    )
    on_outputs = _load_output_fingerprints(on_paths, "geometry_on")
    off_outputs = _load_output_fingerprints(off_paths, "geometry_off")
    parity = _output_parity(on_outputs, off_outputs, protocol)
    outcomes = _load_outcomes(outcomes_path, parse_values=True)
    keys = set(geometry)
    if any(set(value) != keys for value in (on_outputs, outcomes)):
        raise ValueError("Join key set changed after freeze")
    if len(keys) != int(join_manifest["key_count"]) or _key_digest(keys) != str(
        join_manifest["key_digest_sha256"]
    ):
        raise ValueError("Frozen join key digest mismatch")
    allowed_content = _load_fit_content(split_path, protocol)
    _validate_scope(keys, protocol, allowed_content)
    geometry_audit = _geometry_audit(
        geometry, features["feature_order"], protocol["scope"]["primary_pairs"]
    )
    observations = [
        Observation(
            geometry=geometry[key],
            receiver_correct=int(outcomes[key][0]),  # type: ignore[index]
            fused_correct=int(outcomes[key][1]),  # type: ignore[index]
            fold=_content_fold(key[5], protocol),
        )
        for key in sorted(keys)
    ]
    predictions, crossfit_audit = _crossfit(
        observations, features["feature_order"], candidates, protocol
    )
    pooled = _prediction_metrics(predictions)
    pair_metrics = {
        pair: _prediction_metrics(predictions, pair)
        for pair in protocol["scope"]["primary_pairs"]
    }
    task_metrics = {
        task: _prediction_metrics(predictions, task=task)
        for task in protocol["scope"]["tasks"]
    }
    pair_task_metrics = {
        f"{pair}::{task}": _prediction_metrics(
            predictions, pair=pair, task=task
        )
        for pair in protocol["scope"]["primary_pairs"]
        for task in protocol["scope"]["tasks"]
    }
    go_gate = evaluate_go_gates(
        pooled_metrics=pooled,
        pair_metrics=pair_metrics,
        reference_equivalence=equivalence,
        output_parity=parity,
        geometry_audit=geometry_audit,
        protocol=protocol,
    )
    result = {
        "schema_version": 1,
        "phase": "2A-2A",
        "role": "fit_only_cache_geometry_crossfit_statistics",
        "join_manifest": _path_record(join_manifest_path),
        "outcome_access_audit": {
            "identity_rows_read_before_freeze": int(
                join_manifest["outcome_access_audit"]["identity_rows_read"]
            ),
            "correctness_rows_parsed_before_freeze": 0,
            "correctness_rows_parsed_after_freeze": len(outcomes),
        },
        "output_parity": parity,
        "reference_equivalence": equivalence,
        "geometry_variation_audit": geometry_audit,
        "pooled_metrics": pooled,
        "pair_metrics": pair_metrics,
        "task_metrics_secondary": task_metrics,
        "pair_task_metrics_secondary": pair_task_metrics,
        "crossfit_folds": crossfit_audit,
        "go_gate": go_gate,
    }
    _write_json_once(output_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-join")
    freeze.add_argument("--run-summary", action="append", required=True, type=Path)
    freeze.add_argument("--execution-manifest", required=True, type=Path)
    freeze.add_argument("--equivalence-report", required=True, type=Path)
    freeze.add_argument("--outcomes", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    freeze.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    freeze.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    freeze.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    freeze.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--join-manifest", required=True, type=Path)
    analyze_parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze-join":
        result = freeze_join(
            summary_paths=args.run_summary,
            execution_manifest_path=args.execution_manifest,
            equivalence_report_path=args.equivalence_report,
            outcomes_path=args.outcomes,
            output_path=args.output,
            protocol_path=args.protocol,
            feature_path=args.features,
            candidate_path=args.candidates,
            schema_path=args.schema,
        )
    else:
        result = analyze(
            join_manifest_path=args.join_manifest,
            output_path=args.output,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
