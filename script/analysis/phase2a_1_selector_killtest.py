from __future__ import annotations

"""Phase 2A-1 CPU-only selector predictability kill-test.

The command surface is deliberately stage-separated:

* ``prepare-split`` reads input text/choices only and freezes content groups.
* ``fit-select`` may read fit/calibration/model-selection outcomes, but its
  stage-aware loader skips test outcomes before correctness parsing.
* ``validate-test`` checks the sealed test preconditions without labels.
* ``consume-test`` creates an outcome-free attempt receipt which must be
  committed and pushed as the sole child of the selection-artifact commit.
* ``evaluate-test`` atomically claims a never-reused remote Git tag before
  reading any outcome and has no overwrite/resume/force path.

No command imports torch, launches Kubernetes, or mutates model checkpoints.
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
import glob
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sklearn.set_config(enable_metadata_routing=False)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
CLASS_ORDER = np.asarray([-1, 0, 1], dtype=np.int64)
CHOICE_FIELDS: Tuple[str, ...] = tuple(chr(code) for code in range(65, 75))
SPLITS: Tuple[str, ...] = ("fit", "calibration", "model_selection", "test")
PRIMARY_FEATURES: Tuple[str, ...] = (
    "cot_input_length",
    "candidate_count",
    "candidate_count_max",
    "one_to_many_rate",
    "boundary_mismatch",
)
HARD_FORBIDDEN_PRIMARY_FIELDS = frozenset(
    {
        "question",
        "raw_question",
        "options",
        "choices",
        "subject",
        "task",
        "pair",
        "seed",
        "question_id",
        "true_answer",
        "predicted_answer",
        "is_correct",
        "receiver_correct",
        "fused_correct",
        "utility",
        "label",
        "alignment_entropy",
        "source_confidence",
        "confidence",
        "fallback_rate",
    }
)
FORMAL_OUTPUTS = {
    "aggregate_csv": REPO_ROOT / "PHASE2A_1_SELECTOR_AGGREGATES.csv",
    "aggregate_json": REPO_ROOT / "PHASE2A_1_SELECTOR_AGGREGATES.json",
    "report": REPO_ROOT / "PHASE2A_1_SELECTOR_KILLTEST_REPORT.md",
    "summary_zh": REPO_ROOT / "PHASE2A_1_SELECTOR_KILLTEST_SUMMARY_ZH.md",
    "result_manifest": REPO_ROOT
    / "recipe/eval_recipe/phase2a_1/phase2a_1_result_manifest.json",
    "test_attempt": REPO_ROOT / "PHASE2A_1_TEST_ATTEMPT.json",
    "test_complete": REPO_ROOT / "PHASE2A_1_TEST_COMPLETE.json",
}


@dataclass(frozen=True)
class PairSpec:
    pair: str
    label: str
    heterogeneous: bool


@dataclass(frozen=True)
class TaskSpec:
    task: str
    expected_rows: int


@dataclass(frozen=True)
class SourceLayout:
    artifact_root: Path
    pairs: Tuple[PairSpec, ...]
    seeds: Tuple[int, ...]
    tasks: Tuple[TaskSpec, ...]
    receiver_paths: Mapping[str, Path]
    b6_paths: Mapping[Tuple[str, int, str], Path]
    source_artifacts: Tuple[Mapping[str, Any], ...]
    dataset_content_sha256: str


@dataclass(frozen=True)
class Observation:
    pair: str
    seed: int
    task: str
    subject: str
    question_id: str
    content_hash: str
    split: str
    features: Tuple[float, ...]
    receiver_correct: int
    fused_correct: int

    @property
    def utility(self) -> int:
        return self.fused_correct - self.receiver_correct

    @property
    def key(self) -> Tuple[str, str, str, int, str, str]:
        return (
            self.pair,
            self.task,
            self.subject,
            self.seed,
            self.question_id,
            self.content_hash,
        )


@dataclass(frozen=True)
class Prediction:
    observation: Observation
    probabilities: Tuple[float, float, float]
    score: float
    transfer: bool

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


def _stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    """Create a durable JSON artifact and fail if it already exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_csv_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_head() -> str:
    return _git("rev-parse", "HEAD")


def _require_clean_worktree() -> None:
    status = _git("status", "--porcelain")
    if status:
        raise ValueError(f"Formal stage requires a clean worktree:\n{status}")


def _verify_single_parent_commit_diff(
    commit: str, expected_parent: str, expected_paths: Sequence[Path]
) -> None:
    parents = _git("rev-list", "--parents", "-n", "1", commit).split()
    if len(parents) != 2 or parents[0] != commit or parents[1] != expected_parent:
        raise ValueError(
            f"Commit {commit} must have the single parent {expected_parent}; got {parents}"
        )
    changed = set(
        filter(None, _git("diff", "--name-only", expected_parent, commit).splitlines())
    )
    expected = {str(path.resolve().relative_to(REPO_ROOT)) for path in expected_paths}
    if changed != expected:
        raise ValueError(
            f"Commit {commit} changed unexpected paths: changed={sorted(changed)}, "
            f"expected={sorted(expected)}"
        )


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


def _parse_correctness(raw: str, *, path: Path, key: Tuple[str, str, str]) -> int:
    normalized = raw.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return 1
    if normalized in {"false", "0", "no"}:
        return 0
    raise ValueError(f"Invalid is_correct={raw!r} for {key} in {path}")


def _split_fraction(split_version: str, dataset_sha: str, content_hash: str) -> float:
    digest = hashlib.sha256(
        f"{split_version}{dataset_sha}{content_hash}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False) / float(2**64)


def _split_name(fraction: float, intervals: Mapping[str, Sequence[float]]) -> str:
    for name in SPLITS:
        low, high = map(float, intervals[name])
        if low <= fraction < high or (name == "test" and fraction == high == 1.0):
            return name
    raise ValueError(f"Split fraction {fraction} is outside frozen intervals")


def _resolve_one(pattern: str, artifact_root: Path) -> Path:
    raw = Path(pattern)
    candidate = raw if raw.is_absolute() else artifact_root / raw
    matches = sorted(Path(value).resolve() for value in glob.glob(str(candidate)))
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one source artifact for {pattern!r}; found {matches}"
        )
    return matches[0]


def _dataset_prediction_path(
    run: Mapping[str, Any], task: str, artifact_root: Path
) -> Path:
    datasets = run.get("datasets")
    if not isinstance(datasets, Mapping) or task not in datasets:
        raise ValueError(f"Run {run.get('run_id')} has no task {task}")
    artifact = datasets[task]
    if not isinstance(artifact, Mapping):
        raise ValueError(f"Invalid artifact entry for {run.get('run_id')}/{task}")
    pattern = artifact.get("prediction_glob")
    if not pattern:
        raise ValueError(f"Missing prediction_glob for {run.get('run_id')}/{task}")
    return _resolve_one(str(pattern), artifact_root)


def _load_protocol_bundle(
    protocol_path: Path,
    candidate_path: Path,
    feature_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    protocol = _read_json(protocol_path)
    candidates = _read_json(candidate_path)
    features = _read_json(feature_path)
    if protocol.get("phase") != "2A-1":
        raise ValueError("Protocol phase must be 2A-1")
    if candidates.get("selection_eligible_candidate_count") != len(
        candidates.get("candidates", [])
    ):
        raise ValueError("Candidate count contract mismatch")
    if candidates.get("phase") != "2A-1" or features.get("phase") != "2A-1":
        raise ValueError("Candidate and feature manifests must be Phase 2A-1")
    feature_order = tuple(map(str, features.get("feature_order", [])))
    if feature_order != PRIMARY_FEATURES:
        raise ValueError(
            f"Primary feature whitelist must exactly equal {PRIMARY_FEATURES}, got {feature_order}"
        )
    allowed = set(feature_order)
    ordinals: List[int] = []
    ids: set[str] = set()
    expected_logistic_c = [0.01, 0.1, 1.0, 10.0, 100.0]
    seen_logistic_c: List[float] = []
    stump_features: List[str] = []
    depth2_leaf_fractions: List[float] = []
    for candidate in candidates["candidates"]:
        candidate_id = str(candidate["id"])
        if candidate_id in ids:
            raise ValueError(f"Duplicate candidate id {candidate_id}")
        ids.add(candidate_id)
        ordinals.append(int(candidate["ordinal"]))
        used = set(map(str, candidate.get("features", [])))
        if not used or not used.issubset(allowed):
            raise ValueError(f"Candidate {candidate_id} violates feature whitelist")
        family = str(candidate.get("family"))
        params = candidate.get("params")
        if not isinstance(params, Mapping):
            raise ValueError(f"Candidate {candidate_id} has no parameter object")
        if params.get("class_weight") is not None:
            raise ValueError(f"Candidate {candidate_id} may not use class weights")
        if family == "single_feature_stump":
            if len(used) != 1 or int(params.get("max_depth", -1)) != 1:
                raise ValueError(f"Candidate {candidate_id} is not a depth-one stump")
            stump_features.extend(used)
        elif family == "l2_multinomial_logistic":
            if list(candidate.get("features", [])) != list(PRIMARY_FEATURES):
                raise ValueError(f"Logistic candidate {candidate_id} must use all features")
            if (
                str(params.get("penalty")) != "l2"
                or str(params.get("solver")) != "lbfgs"
                or int(params.get("max_iter", 0)) != 5000
            ):
                raise ValueError(f"Logistic candidate {candidate_id} violates frozen family")
            seen_logistic_c.append(float(params["C"]))
        elif family == "shallow_decision_tree":
            if list(candidate.get("features", [])) != list(PRIMARY_FEATURES):
                raise ValueError(f"Tree candidate {candidate_id} must use all features")
            if int(params.get("max_depth", -1)) != 2:
                raise ValueError(f"Candidate {candidate_id} exceeds frozen depth two")
            depth2_leaf_fractions.append(float(params["min_weight_fraction_leaf"]))
        else:
            raise ValueError(f"Candidate {candidate_id} has forbidden family {family}")
    if ordinals != list(range(len(ordinals))):
        raise ValueError("Candidate ordinals must be contiguous manifest order")
    forbidden = set(map(str, features.get("forbidden_primary_features", [])))
    if not HARD_FORBIDDEN_PRIMARY_FIELDS.issubset(forbidden):
        missing = sorted(HARD_FORBIDDEN_PRIMARY_FIELDS - forbidden)
        raise ValueError(f"Feature manifest denylist is incomplete: {missing}")
    if allowed & (forbidden | HARD_FORBIDDEN_PRIMARY_FIELDS):
        raise ValueError("Allowed and forbidden feature sets overlap")
    if sorted(stump_features) != sorted(PRIMARY_FEATURES):
        raise ValueError("Exactly one stump per primary feature is required")
    if seen_logistic_c != expected_logistic_c:
        raise ValueError(f"Frozen logistic C grid mismatch: {seen_logistic_c}")
    if depth2_leaf_fractions != [0.05, 0.01]:
        raise ValueError(f"Frozen depth-two tree grid mismatch: {depth2_leaf_fractions}")
    if len(ordinals) != 12:
        raise ValueError("Exactly twelve trainable candidates are permitted")
    expected_ids = [
        *(f"stump_{name}" for name in PRIMARY_FEATURES),
        "logreg_l2_c001",
        "logreg_l2_c01",
        "logreg_l2_c1",
        "logreg_l2_c10",
        "logreg_l2_c100",
        "tree_depth2_leaf005",
        "tree_depth2_leaf001",
    ]
    if [str(item["id"]) for item in candidates["candidates"]] != expected_ids:
        raise ValueError("Frozen candidate ids/order mismatch")
    for candidate in candidates["candidates"]:
        params = candidate["params"]
        if int(params.get("random_state", -1)) != 20260721:
            raise ValueError(f"Candidate {candidate['id']} random state mismatch")
        if candidate["family"] == "l2_multinomial_logistic":
            if (
                float(params.get("tol", 0.0)) != 1e-10
                or params.get("fit_intercept") is not True
            ):
                raise ValueError(f"Candidate {candidate['id']} logistic params mismatch")
        else:
            if (
                str(params.get("criterion")) != "gini"
                or str(params.get("splitter")) != "best"
            ):
                raise ValueError(f"Candidate {candidate['id']} tree params mismatch")
    expected_baselines = ["always_receiver", "always_fused", "same_rate_random"]
    references = candidates.get("reference_baselines")
    if not isinstance(references, list) or [
        str(item.get("id")) for item in references
    ] != expected_baselines or any(item.get("selection_eligible") is not False for item in references):
        raise ValueError("Frozen reference baseline contract mismatch")
    return protocol, candidates, features


def _validate_runtime(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    actual = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "sklearn_metadata_routing": bool(
            sklearn.get_config()["enable_metadata_routing"]
        ),
    }
    expected = protocol.get("runtime", {})
    for key, value in actual.items():
        if expected.get(key) != value:
            raise ValueError(f"Runtime mismatch for {key}: {value} != {expected.get(key)}")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if os.environ.get(name) != "1":
            raise ValueError(f"{name} must equal 1")
    return actual


def _load_source_layout(protocol: Mapping[str, Any]) -> SourceLayout:
    source = protocol.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Protocol source must be an object")
    phase0_manifest_path = _resolve_repo_path(str(source["phase2a_0_manifest"]))
    expected_phase0_sha = str(source["phase2a_0_manifest_sha256"])
    if _sha256(phase0_manifest_path) != expected_phase0_sha:
        raise ValueError("Phase2A-0 manifest SHA mismatch")
    aggregate_path = _resolve_repo_path(str(source["phase2a_0_aggregate_json"]))
    if _sha256(aggregate_path) != str(source["phase2a_0_aggregate_json_sha256"]):
        raise ValueError("Phase2A-0 aggregate provenance SHA mismatch")
    phase0_aggregate = _read_json(aggregate_path)
    frozen_source_artifacts = phase0_aggregate.get("source_artifacts")
    if not isinstance(frozen_source_artifacts, list):
        raise ValueError("Phase2A-0 aggregate has no frozen source artifact list")
    phase0 = _read_json(phase0_manifest_path)
    phase0_source = phase0["source"]
    artifact_root = Path(str(phase0_source["artifact_root"])).resolve()
    analysis_path = artifact_root / str(phase0_source["phase1_analysis_manifest"])
    if _sha256(analysis_path) != str(phase0_source["phase1_analysis_manifest_sha256"]):
        raise ValueError("Phase1 analysis manifest SHA mismatch")
    analysis = _read_json(analysis_path)
    raw_runs = analysis.get("runs")
    if not isinstance(raw_runs, list):
        raise ValueError("Phase1 analysis manifest has no runs")
    runs: Dict[Tuple[str, str, int], Mapping[str, Any]] = {}
    runs_by_id: Dict[str, Mapping[str, Any]] = {}
    for run in raw_runs:
        if not isinstance(run, Mapping):
            continue
        run_id = str(run.get("run_id", ""))
        if run_id:
            runs_by_id[run_id] = run
        if all(key in run for key in ("pair", "variant", "seed")):
            key = (str(run["pair"]), str(run["variant"]).lower(), int(run["seed"]))
            if key in runs:
                raise ValueError(f"Duplicate source run {key}")
            runs[key] = run
    pair_specs = tuple(
        PairSpec(
            pair=str(item["id"]),
            label=str(item.get("label", item["id"])),
            heterogeneous=bool(item.get("heterogeneous", False)),
        )
        for item in phase0["pairs"]
    )
    seeds = tuple(map(int, phase0["seeds"]))
    tasks = tuple(
        TaskSpec(str(item["id"]), int(item["expected_rows"]))
        for item in phase0["tasks"]
    )
    receiver_id = str(phase0_source["receiver_run_id"])
    receiver_run = runs_by_id.get(receiver_id)
    if receiver_run is None:
        raise ValueError(f"Missing receiver run {receiver_id}")
    receiver_paths: Dict[str, Path] = {}
    b6_paths: Dict[Tuple[str, int, str], Path] = {}
    source_artifacts: List[Mapping[str, Any]] = []
    for task in tasks:
        path = _dataset_prediction_path(receiver_run, task.task, artifact_root)
        receiver_paths[task.task] = path
        source_artifacts.append(
            {
                "role": "receiver_only",
                "pair": "receiver",
                "seed": int(receiver_run["seed"]),
                "task": task.task,
                "path": str(path),
                "rows": task.expected_rows,
                "sha256": _sha256(path),
            }
        )
    for pair in pair_specs:
        for seed in seeds:
            run = runs.get((pair.pair, str(phase0_source["fused_variant"]).lower(), seed))
            if run is None:
                raise ValueError(f"Missing B6-native source {pair.pair}/seed_{seed}")
            for task in tasks:
                path = _dataset_prediction_path(run, task.task, artifact_root)
                b6_paths[(pair.pair, seed, task.task)] = path
                source_artifacts.append(
                    {
                        "role": "b6_native",
                        "pair": pair.pair,
                        "seed": seed,
                        "task": task.task,
                        "path": str(path),
                        "rows": task.expected_rows,
                        "sha256": _sha256(path),
                    }
                )
    normalized_keys = ("role", "pair", "seed", "task", "path", "rows", "sha256")
    current_frozen_view = sorted(
        ({key: item[key] for key in normalized_keys} for item in source_artifacts),
        key=lambda item: (
            str(item["role"]),
            str(item["pair"]),
            int(item["seed"]),
            str(item["task"]),
        ),
    )
    expected_frozen_view = sorted(
        ({key: item[key] for key in normalized_keys} for item in frozen_source_artifacts),
        key=lambda item: (
            str(item["role"]),
            str(item["pair"]),
            int(item["seed"]),
            str(item["task"]),
        ),
    )
    if current_frozen_view != expected_frozen_view:
        raise ValueError(
            "Current receiver/B6 artifacts differ from the Phase2A-0 frozen provenance"
        )
    return SourceLayout(
        artifact_root=artifact_root,
        pairs=pair_specs,
        seeds=seeds,
        tasks=tasks,
        receiver_paths=receiver_paths,
        b6_paths=b6_paths,
        source_artifacts=tuple(current_frozen_view),
        dataset_content_sha256=str(source["dataset_content_sha256"]),
    )


def _read_input_hashes(path: Path, task: str) -> Dict[Tuple[str, str, str], str]:
    """Read only deployment inputs needed for grouping; never retain outcomes."""

    rows: Dict[Tuple[str, str, str], str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        required = {"subject", "question_id", "question", *CHOICE_FIELDS}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing input fields {sorted(missing)} in {path}")
        for raw in reader:
            key = _sample_key(task, raw, path)
            if key in rows:
                raise ValueError(f"Duplicate sample key {key} in {path}")
            rows[key] = _content_hash(raw)
    return rows


def prepare_split_manifest(
    protocol_path: Path,
    candidate_path: Path,
    feature_path: Path,
    output_path: Path,
    sha_output_path: Path,
) -> Dict[str, Any]:
    if output_path.exists() or sha_output_path.exists():
        raise FileExistsError("Content-group split freeze is write-once")
    protocol, _candidates, _features = _load_protocol_bundle(
        protocol_path, candidate_path, feature_path
    )
    _validate_runtime(protocol)
    layout = _load_source_layout(protocol)
    split_config = protocol["split"]
    groups: Dict[str, List[Tuple[str, str, str]]] = {}
    task_hash_payloads: Dict[str, List[str]] = {}
    expected_by_task = {item.task: item.expected_rows for item in layout.tasks}
    for task in layout.tasks:
        rows = _read_input_hashes(layout.receiver_paths[task.task], task.task)
        if len(rows) != task.expected_rows:
            raise ValueError(
                f"Unexpected receiver rows for {task.task}: {len(rows)} != {task.expected_rows}"
            )
        payloads: List[str] = []
        for key, content_hash in rows.items():
            groups.setdefault(content_hash, []).append(key)
            payloads.append("\t".join((*key, content_hash)))
        task_hash_payloads[task.task] = sorted(payloads)
    task_content_sha = {
        task: hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()
        for task, values in sorted(task_hash_payloads.items())
    }
    dataset_sha = hashlib.sha256(
        json.dumps(task_content_sha, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if dataset_sha != layout.dataset_content_sha256:
        raise ValueError(
            f"Dataset content SHA mismatch: {dataset_sha} != {layout.dataset_content_sha256}"
        )
    split_rows: Dict[str, Dict[str, int]] = {
        task.task: {name: 0 for name in SPLITS} for task in layout.tasks
    }
    split_groups: Dict[str, Dict[str, int]] = {
        task.task: {name: 0 for name in SPLITS} for task in layout.tasks
    }
    group_entries: List[Dict[str, Any]] = []
    for content_hash, members in sorted(groups.items()):
        fraction = _split_fraction(
            str(split_config["version"]), dataset_sha, content_hash
        )
        split = _split_name(fraction, split_config["intervals"])
        members_sorted = sorted(members)
        tasks_in_group = sorted({member[0] for member in members_sorted})
        if len(tasks_in_group) != 1:
            raise ValueError(
                f"Current protocol expects no cross-task duplicate group: {content_hash}"
            )
        task = tasks_in_group[0]
        split_rows[task][split] += len(members_sorted)
        split_groups[task][split] += 1
        group_entries.append(
            {
                "content_hash": content_hash,
                "split": split,
                "hash_fraction_hex": float(fraction).hex(),
                "member_count": len(members_sorted),
                "members": [
                    {
                        "task": member[0],
                        "subject": member[1],
                        "question_id": member[2],
                    }
                    for member in members_sorted
                ],
            }
        )
    for task, expected in expected_by_task.items():
        if sum(split_rows[task].values()) != expected:
            raise ValueError(f"Split row count mismatch for {task}")
    manifest = {
        "schema_version": 1,
        "phase": "2A-1",
        "role": "content_group_split_manifest",
        "created_without_outcome_fields": True,
        "source_commit": protocol["source_commit"],
        "dataset_content_sha256": dataset_sha,
        "task_content_sha256": task_content_sha,
        "split_version": split_config["version"],
        "split_algorithm": {
            "content_hash": "SHA256(canonical JSON of normalized question and choices A-J)",
            "group_hash": split_config["hash"],
            "fraction": split_config["hash_fraction"],
            "intervals": split_config["intervals"],
        },
        "counts": {
            "unique_rows": sum(expected_by_task.values()),
            "unique_content_groups": len(group_entries),
            "duplicate_content_group_count": sum(
                int(entry["member_count"]) > 1 for entry in group_entries
            ),
            "rows_by_task_split": split_rows,
            "groups_by_task_split": split_groups,
        },
        "groups": group_entries,
    }
    _write_json_once(output_path, manifest)
    digest = _sha256(output_path)
    sha_output_path.parent.mkdir(parents=True, exist_ok=True)
    with sha_output_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{digest}  {output_path.name}\n")
    return manifest


def _load_split_manifest(
    path: Path, sha_path: Path, protocol: Mapping[str, Any]
) -> Dict[str, Any]:
    expected_line = sha_path.read_text(encoding="utf-8").strip().split()
    if not expected_line:
        raise ValueError(f"Empty split SHA file: {sha_path}")
    actual = _sha256(path)
    if actual != expected_line[0]:
        raise ValueError(f"Split manifest SHA mismatch: {actual} != {expected_line[0]}")
    manifest = _read_json(path)
    if manifest.get("created_without_outcome_fields") is not True:
        raise ValueError("Split manifest outcome-free contract is missing")
    split_config = protocol["split"]
    if manifest.get("dataset_content_sha256") != protocol["source"][
        "dataset_content_sha256"
    ]:
        raise ValueError("Split manifest dataset SHA differs from protocol")
    if manifest.get("split_version") != split_config["version"]:
        raise ValueError("Split version differs from protocol")
    if manifest.get("split_algorithm", {}).get("intervals") != split_config[
        "intervals"
    ]:
        raise ValueError("Split intervals differ from protocol")
    seen: set[str] = set()
    member_total = 0
    for entry in manifest.get("groups", []):
        content_hash = str(entry["content_hash"])
        if content_hash in seen:
            raise ValueError(f"Duplicate content group in split manifest: {content_hash}")
        seen.add(content_hash)
        expected = _split_name(
            _split_fraction(
                str(split_config["version"]),
                str(manifest["dataset_content_sha256"]),
                content_hash,
            ),
            split_config["intervals"],
        )
        if str(entry["split"]) != expected:
            raise ValueError(f"Deterministic split mismatch for {content_hash}")
        members = entry.get("members")
        if not isinstance(members, list) or int(entry.get("member_count", -1)) != len(
            members
        ):
            raise ValueError(f"Invalid members for split group {content_hash}")
        member_total += len(members)
    counts = manifest.get("counts", {})
    if len(seen) != int(counts.get("unique_content_groups", -1)):
        raise ValueError("Split content-group count mismatch")
    if member_total != int(counts.get("unique_rows", -1)):
        raise ValueError("Split member count mismatch")
    return manifest


def _split_lookup(manifest: Mapping[str, Any]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for entry in manifest.get("groups", []):
        content_hash = str(entry["content_hash"])
        split = str(entry["split"])
        if content_hash in lookup or split not in SPLITS:
            raise ValueError(f"Invalid split group entry {content_hash}/{split}")
        lookup[content_hash] = split
    return lookup


def _parse_feature_vector(
    row: Mapping[str, str], feature_order: Sequence[str], path: Path
) -> Tuple[float, ...]:
    values: List[float] = []
    for field in feature_order:
        raw = row.get(field, "")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Missing/non-numeric feature {field} in {path}") from exc
        if not math.isfinite(value):
            raise ValueError(f"Non-finite feature {field}={value} in {path}")
        values.append(value)
    return tuple(values)


def load_observations(
    protocol: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    *,
    stage: str,
    include_pairs: Sequence[str] | None = None,
    include_seeds: Sequence[int] | None = None,
    include_tasks: Sequence[str] | None = None,
    test_authorization: Mapping[str, Any] | None = None,
    source_layout: SourceLayout | None = None,
) -> Tuple[List[Observation], Dict[str, Any], SourceLayout]:
    if stage not in {"develop", "test"}:
        raise ValueError(f"Unknown observation stage {stage}")
    if stage == "test":
        _validate_test_authorization(test_authorization)
    layout = source_layout or _load_source_layout(protocol)
    allowed_pairs = set(include_pairs or (item.pair for item in layout.pairs))
    allowed_seeds = set(map(int, include_seeds or layout.seeds))
    allowed_tasks = set(include_tasks or (item.task for item in layout.tasks))
    if not allowed_pairs or not allowed_seeds or not allowed_tasks:
        raise ValueError("Observation scope cannot be empty")
    if not allowed_pairs.issubset({item.pair for item in layout.pairs}):
        raise ValueError("Unknown pair in observation scope")
    if not allowed_seeds.issubset(set(layout.seeds)):
        raise ValueError("Unknown seed in observation scope")
    if not allowed_tasks.issubset({item.task for item in layout.tasks}):
        raise ValueError("Unknown task in observation scope")
    split_by_hash = _split_lookup(split_manifest)
    feature_order = tuple(map(str, feature_manifest["feature_order"]))
    receiver_meta: Dict[
        Tuple[str, str, str], Tuple[str, str, int | None]
    ] = {}
    audit = {
        "stage": stage,
        "receiver_outcome_rows_parsed_by_split": {name: 0 for name in SPLITS},
        "fused_outcome_rows_parsed_by_split": {name: 0 for name in SPLITS},
        "rows_skipped_before_outcome_parse_by_split": {name: 0 for name in SPLITS},
        "included_pairs": sorted(allowed_pairs),
        "included_seeds": sorted(allowed_seeds),
        "included_tasks": sorted(allowed_tasks),
    }
    for task in layout.tasks:
        if task.task not in allowed_tasks:
            continue
        path = layout.receiver_paths[task.task]
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = _sample_key(task.task, row, path)
                content_hash = _content_hash(row)
                split = split_by_hash.get(content_hash)
                if split is None:
                    raise ValueError(f"Missing split for content {content_hash}")
                receiver_correct: int | None = None
                if stage == "test" or split != "test":
                    receiver_correct = _parse_correctness(
                        row.get("is_correct", ""), path=path, key=key
                    )
                    audit["receiver_outcome_rows_parsed_by_split"][split] += 1
                else:
                    audit["rows_skipped_before_outcome_parse_by_split"][split] += 1
                receiver_meta[key] = (content_hash, split, receiver_correct)
    observations: List[Observation] = []
    for pair in layout.pairs:
        if pair.pair not in allowed_pairs:
            continue
        for seed in layout.seeds:
            if seed not in allowed_seeds:
                continue
            for task in layout.tasks:
                if task.task not in allowed_tasks:
                    continue
                path = layout.b6_paths[(pair.pair, seed, task.task)]
                seen: set[Tuple[str, str, str]] = set()
                with path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    if reader.fieldnames is None:
                        raise ValueError(f"CSV has no header: {path}")
                    if not set(feature_order).issubset(reader.fieldnames):
                        raise ValueError(f"B6 file lacks primary features: {path}")
                    for row in reader:
                        key = _sample_key(task.task, row, path)
                        if key in seen:
                            raise ValueError(f"Duplicate B6 key {key} in {path}")
                        seen.add(key)
                        receiver = receiver_meta.get(key)
                        if receiver is None:
                            raise ValueError(f"B6 key absent from receiver: {key}")
                        content_hash, split, receiver_correct = receiver
                        if _content_hash(row) != content_hash:
                            raise ValueError(f"Content mismatch for {key} in {path}")
                        if stage == "develop" and split == "test":
                            audit["rows_skipped_before_outcome_parse_by_split"][split] += 1
                            continue
                        features = _parse_feature_vector(row, feature_order, path)
                        fused_correct = _parse_correctness(
                            row.get("is_correct", ""), path=path, key=key
                        )
                        audit["fused_outcome_rows_parsed_by_split"][split] += 1
                        if receiver_correct is None:
                            raise ValueError(f"Receiver outcome unavailable for included row {key}")
                        observations.append(
                            Observation(
                                pair=pair.pair,
                                seed=seed,
                                task=task.task,
                                subject=key[1],
                                question_id=key[2],
                                content_hash=content_hash,
                                split=split,
                                features=features,
                                receiver_correct=receiver_correct,
                                fused_correct=fused_correct,
                            )
                        )
                if seen != {
                    receiver_key
                    for receiver_key in receiver_meta
                    if receiver_key[0] == task.task
                }:
                    raise ValueError(f"B6/receiver key-set mismatch for {pair.pair}/{seed}/{task.task}")
    if stage == "develop":
        if audit["receiver_outcome_rows_parsed_by_split"]["test"] != 0:
            raise AssertionError("Development parsed receiver test outcomes")
        if audit["fused_outcome_rows_parsed_by_split"]["test"] != 0:
            raise AssertionError("Development parsed fused test outcomes")
        if any(item.split == "test" for item in observations):
            raise AssertionError("Development observations contain test rows")
    return observations, audit, layout


def _balanced_weights(observations: Sequence[Observation]) -> np.ndarray:
    if not observations:
        raise ValueError("Cannot weight empty observations")
    counts: Dict[Tuple[str, int, str], int] = {}
    for item in observations:
        key = (item.pair, item.seed, item.task)
        counts[key] = counts.get(key, 0) + 1
    expected_cells = {
        (pair, seed, task)
        for pair in {item.pair for item in observations}
        for seed in {item.seed for item in observations}
        for task in {item.task for item in observations}
    }
    if set(counts) != expected_cells:
        raise ValueError("Active pair/seed/task cells are not a full Cartesian product")
    cell_count = len(counts)
    total = len(observations)
    weights = np.asarray(
        [total / (cell_count * counts[(item.pair, item.seed, item.task)]) for item in observations],
        dtype=np.float64,
    )
    if not math.isclose(float(weights.mean()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("Balanced training weights must have mean one")
    cell_mass: Dict[Tuple[str, int, str], float] = {}
    for item, weight in zip(observations, weights):
        key = (item.pair, item.seed, item.task)
        cell_mass[key] = cell_mass.get(key, 0.0) + float(weight)
    masses = list(cell_mass.values())
    if max(masses) - min(masses) > 1e-9:
        raise AssertionError("Pair/seed/task cell masses are not equal")
    return weights


def _arrays(
    observations: Sequence[Observation],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not observations:
        raise ValueError("No observations")
    x = np.asarray([item.features for item in observations], dtype=np.float64)
    y = np.asarray([item.utility for item in observations], dtype=np.int64)
    receiver = np.asarray([item.receiver_correct for item in observations], dtype=np.int8)
    fused = np.asarray([item.fused_correct for item in observations], dtype=np.int8)
    if not np.isfinite(x).all():
        raise ValueError("Feature matrix contains non-finite values")
    return x, y, receiver, fused


def _feature_indices(
    candidate: Mapping[str, Any], feature_order: Sequence[str]
) -> List[int]:
    index = {name: position for position, name in enumerate(feature_order)}
    return [index[str(name)] for name in candidate["features"]]


def _base_estimator(
    candidate: Mapping[str, Any], feature_order: Sequence[str]
) -> Pipeline:
    indices = _feature_indices(candidate, feature_order)
    selector = ColumnTransformer(
        [("selected", "passthrough", indices)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    family = str(candidate["family"])
    params = dict(candidate["params"])
    if family == "l2_multinomial_logistic":
        classifier = LogisticRegression(
            penalty=str(params["penalty"]),
            C=float(params["C"]),
            solver=str(params["solver"]),
            tol=float(params["tol"]),
            max_iter=int(params["max_iter"]),
            fit_intercept=bool(params["fit_intercept"]),
            class_weight=params["class_weight"],
            random_state=int(params["random_state"]),
        )
        return Pipeline(
            [("select", selector), ("scale", StandardScaler()), ("clf", classifier)]
        )
    if family in {"single_feature_stump", "shallow_decision_tree"}:
        classifier = DecisionTreeClassifier(
            criterion=str(params["criterion"]),
            splitter=str(params["splitter"]),
            max_depth=int(params["max_depth"]),
            min_weight_fraction_leaf=float(params["min_weight_fraction_leaf"]),
            class_weight=params["class_weight"],
            random_state=int(params["random_state"]),
        )
        return Pipeline([("select", selector), ("clf", classifier)])
    raise ValueError(f"Unsupported candidate family {family}")


def _fit_base(
    candidate: Mapping[str, Any],
    feature_order: Sequence[str],
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> Pipeline:
    if not np.array_equal(np.unique(y), CLASS_ORDER):
        raise ValueError(
            f"Fit split lacks target classes for {candidate['id']}: {np.unique(y)}"
        )
    estimator = _base_estimator(candidate, feature_order)
    family = str(candidate["family"])
    fit_params: Dict[str, Any] = {"clf__sample_weight": weights}
    if family == "l2_multinomial_logistic":
        fit_params["scale__sample_weight"] = weights
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(x, y, **fit_params)
    convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    other = [item for item in caught if not issubclass(item.category, ConvergenceWarning)]
    if convergence:
        raise ValueError(
            f"Candidate {candidate['id']} emitted ConvergenceWarning: "
            + "; ".join(str(item.message) for item in convergence)
        )
    if other:
        raise ValueError(
            f"Candidate {candidate['id']} emitted unexpected warnings: "
            + "; ".join(str(item.message) for item in other)
        )
    classes = np.asarray(estimator.classes_, dtype=np.int64)
    if not np.array_equal(classes, CLASS_ORDER):
        raise ValueError(f"Base classes are not [-1,0,1]: {classes}")
    return estimator


_FROZEN_WEIGHT_WARNING_PREFIX = (
    "Since FrozenEstimator does not appear to accept sample_weight, sample weights "
    "will only be used for the calibration itself."
)


def _fit_calibrator(
    base: Pipeline,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> Tuple[CalibratedClassifierCV, List[str]]:
    if not np.array_equal(np.unique(y), CLASS_ORDER):
        raise ValueError(f"Calibration split lacks target classes: {np.unique(y)}")
    calibrator = CalibratedClassifierCV(
        FrozenEstimator(base),
        method="sigmoid",
        cv=None,
        ensemble=False,
        n_jobs=1,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calibrator.fit(x, y, sample_weight=weights)
    warning_messages = [str(item.message) for item in caught]
    unexpected = [
        message
        for message in warning_messages
        if not message.startswith(_FROZEN_WEIGHT_WARNING_PREFIX)
    ]
    if unexpected:
        raise ValueError(f"Unexpected calibration warnings: {unexpected}")
    classes = np.asarray(calibrator.classes_, dtype=np.int64)
    if not np.array_equal(classes, CLASS_ORDER):
        raise ValueError(f"Calibrated classes are not [-1,0,1]: {classes}")
    if len(calibrator.calibrated_classifiers_) != 1:
        raise ValueError("Frozen calibration must produce exactly one calibrated model")
    probabilities = np.asarray(calibrator.predict_proba(x), dtype=np.float64)
    _validate_probabilities(probabilities)
    return calibrator, warning_messages


def _validate_probabilities(probabilities: np.ndarray) -> None:
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError(f"Expected Nx3 probabilities, got {probabilities.shape}")
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError("Probabilities are non-finite or negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("Probability rows do not sum to one")


def _predict_probabilities(model: Any, x: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=np.int64)
    if not np.array_equal(classes, CLASS_ORDER):
        positions = {int(value): index for index, value in enumerate(classes)}
        if set(positions) != {-1, 0, 1}:
            raise ValueError(f"Model probability classes are invalid: {classes}")
        probabilities = probabilities[:, [positions[-1], positions[0], positions[1]]]
    _validate_probabilities(probabilities)
    return probabilities


def _score(probabilities: np.ndarray) -> np.ndarray:
    return probabilities[:, 2] - probabilities[:, 0]


def _policy_accuracy(
    observations: Sequence[Observation], actions: np.ndarray, weights: np.ndarray
) -> float:
    receiver = np.asarray([item.receiver_correct for item in observations], dtype=np.float64)
    utility = np.asarray([item.utility for item in observations], dtype=np.float64)
    correctness = receiver + actions.astype(np.float64) * utility
    return float(np.average(correctness, weights=weights))


def _serialize_threshold(value: float) -> Dict[str, Any]:
    if value == -math.inf:
        return {"kind": "always_fused"}
    if value == math.inf:
        return {"kind": "always_receiver"}
    return {"kind": "finite", "decimal": repr(float(value)), "float_hex": float(value).hex()}


def _threshold_value(value: Mapping[str, Any]) -> float:
    kind = str(value["kind"])
    if kind == "always_fused":
        return -math.inf
    if kind == "always_receiver":
        return math.inf
    if kind == "finite":
        return float.fromhex(str(value["float_hex"]))
    raise ValueError(f"Unknown threshold kind {kind}")


def _select_threshold(
    observations: Sequence[Observation],
    scores: np.ndarray,
    weights: np.ndarray,
    tolerance: float,
) -> Dict[str, Any]:
    finite = np.unique(scores[np.isfinite(scores)])
    thresholds = np.concatenate(([-math.inf], finite, [math.inf]))
    best_threshold = -math.inf
    best_accuracy = -math.inf
    best_transfer = math.inf
    for threshold in thresholds:
        actions = scores > threshold
        accuracy = _policy_accuracy(observations, actions, weights)
        transfer = float(np.average(actions.astype(np.float64), weights=weights))
        if accuracy > best_accuracy + tolerance or (
            abs(accuracy - best_accuracy) <= tolerance and threshold > best_threshold
        ):
            best_threshold = float(threshold)
            best_accuracy = accuracy
            best_transfer = transfer
    return {
        "threshold": _serialize_threshold(best_threshold),
        "calibration_selector_accuracy": best_accuracy,
        "calibration_transfer_rate": best_transfer,
        "threshold_candidate_count": int(len(thresholds)),
    }


def _select_comparator(
    observations: Sequence[Observation], weights: np.ndarray
) -> Dict[str, Any]:
    receiver = float(
        np.average(
            np.asarray([item.receiver_correct for item in observations], dtype=np.float64),
            weights=weights,
        )
    )
    fused = float(
        np.average(
            np.asarray([item.fused_correct for item in observations], dtype=np.float64),
            weights=weights,
        )
    )
    comparator = "always_fused" if fused > receiver else "always_receiver"
    return {
        "policy": comparator,
        "receiver_accuracy": receiver,
        "fused_accuracy": fused,
        "selected_accuracy": fused if comparator == "always_fused" else receiver,
        "tie_rule": "receiver",
    }


def _prediction_checksum(
    observations: Sequence[Observation], probabilities: np.ndarray, actions: np.ndarray
) -> str:
    digest = hashlib.sha256()
    for item, probability, action in zip(observations, probabilities, actions):
        digest.update("|".join(map(str, item.key)).encode("utf-8"))
        digest.update(np.asarray(probability, dtype="<f8").tobytes())
        digest.update(bytes([int(action)]))
    return digest.hexdigest()


def _estimator_summary(base: Pipeline, calibrated: CalibratedClassifierCV) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "classes": list(map(int, calibrated.classes_)),
        "calibrators": [],
    }
    calibrated_item = calibrated.calibrated_classifiers_[0]
    for item in calibrated_item.calibrators:
        summary["calibrators"].append(
            {"a": float(item.a_), "b": float(item.b_)}
        )
    if "scale" in base.named_steps:
        scaler = base.named_steps["scale"]
        summary["scaler"] = {
            "mean": [float(value) for value in scaler.mean_],
            "var": [float(value) for value in scaler.var_],
            "scale": [float(value) for value in scaler.scale_],
            "n_samples_seen": float(scaler.n_samples_seen_),
        }
    classifier = base.named_steps["clf"]
    if isinstance(classifier, LogisticRegression):
        summary["logistic"] = {
            "coef": np.asarray(classifier.coef_, dtype=float).tolist(),
            "intercept": np.asarray(classifier.intercept_, dtype=float).tolist(),
            "n_iter": np.asarray(classifier.n_iter_, dtype=int).tolist(),
        }
    elif isinstance(classifier, DecisionTreeClassifier):
        tree = classifier.tree_
        summary["tree"] = {
            "node_count": int(tree.node_count),
            "max_depth": int(tree.max_depth),
            "children_left": tree.children_left.astype(int).tolist(),
            "children_right": tree.children_right.astype(int).tolist(),
            "feature": tree.feature.astype(int).tolist(),
            "threshold": tree.threshold.astype(float).tolist(),
            "weighted_n_node_samples": tree.weighted_n_node_samples.astype(float).tolist(),
            "value": tree.value.astype(float).tolist(),
        }
    return summary


def _fit_candidate(
    candidate: Mapping[str, Any],
    feature_order: Sequence[str],
    fit_observations: Sequence[Observation],
    calibration_observations: Sequence[Observation],
    selection_observations: Sequence[Observation],
    threshold_tolerance: float,
) -> Tuple[Dict[str, Any], CalibratedClassifierCV, Pipeline]:
    x_fit, y_fit, _receiver_fit, _fused_fit = _arrays(fit_observations)
    x_cal, y_cal, _receiver_cal, _fused_cal = _arrays(calibration_observations)
    x_select, _y_select, _receiver_select, _fused_select = _arrays(
        selection_observations
    )
    w_fit = _balanced_weights(fit_observations)
    w_cal = _balanced_weights(calibration_observations)
    w_select = _balanced_weights(selection_observations)
    base = _fit_base(candidate, feature_order, x_fit, y_fit, w_fit)
    calibrated, calibration_warnings = _fit_calibrator(base, x_cal, y_cal, w_cal)
    cal_probabilities = _predict_probabilities(calibrated, x_cal)
    threshold = _select_threshold(
        calibration_observations,
        _score(cal_probabilities),
        w_cal,
        threshold_tolerance,
    )
    select_probabilities = _predict_probabilities(calibrated, x_select)
    select_scores = _score(select_probabilities)
    threshold_float = _threshold_value(threshold["threshold"])
    select_actions = select_scores > threshold_float
    selection_accuracy = _policy_accuracy(
        selection_observations, select_actions, w_select
    )
    selection_transfer = float(
        np.average(select_actions.astype(np.float64), weights=w_select)
    )
    summary = {
        "candidate_id": candidate["id"],
        "ordinal": int(candidate["ordinal"]),
        "family": candidate["family"],
        "features": list(candidate["features"]),
        **threshold,
        "model_selection_selector_accuracy": selection_accuracy,
        "model_selection_transfer_rate": selection_transfer,
        "calibration_warning_messages": calibration_warnings,
        "model_selection_prediction_checksum": _prediction_checksum(
            selection_observations, select_probabilities, select_actions
        ),
        "state_summary": _estimator_summary(base, calibrated),
    }
    return summary, calibrated, base


def _scope_selection(
    observations: Sequence[Observation],
    candidates: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    scope_id: str,
    model_path: Path,
) -> Dict[str, Any]:
    by_split = {
        split: [item for item in observations if item.split == split]
        for split in ("fit", "calibration", "model_selection")
    }
    if any(not by_split[name] for name in by_split):
        raise ValueError(f"Scope {scope_id} has an empty development split")
    comparator = _select_comparator(
        by_split["calibration"], _balanced_weights(by_split["calibration"])
    )
    threshold_tolerance = float(protocol["threshold"]["tie_tolerance"])
    fitted: List[Tuple[Dict[str, Any], CalibratedClassifierCV, Pipeline]] = []
    feature_order = tuple(map(str, feature_manifest["feature_order"]))
    for candidate in candidates["candidates"]:
        fitted.append(
            _fit_candidate(
                candidate,
                feature_order,
                by_split["fit"],
                by_split["calibration"],
                by_split["model_selection"],
                threshold_tolerance,
            )
        )
    selection_tolerance = float(protocol["model_selection"]["tie_tolerance"])
    best_index = 0
    for index in range(1, len(fitted)):
        current = fitted[index][0]
        best = fitted[best_index][0]
        current_accuracy = float(current["model_selection_selector_accuracy"])
        best_accuracy = float(best["model_selection_selector_accuracy"])
        if current_accuracy > best_accuracy + selection_tolerance or (
            abs(current_accuracy - best_accuracy) <= selection_tolerance
            and int(current["ordinal"]) < int(best["ordinal"])
        ):
            best_index = index
    selected_summary, selected_model, selected_base = fitted[best_index]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen model: {model_path}")
    temporary_model = model_path.with_name(f".{model_path.name}.tmp-{os.getpid()}")
    if temporary_model.exists():
        raise FileExistsError(f"Unexpected model temporary exists: {temporary_model}")
    joblib.dump(
        {
            "schema_version": 1,
            "scope_id": scope_id,
            "candidate_id": selected_summary["candidate_id"],
            "feature_order": list(feature_order),
            "model": selected_model,
        },
        temporary_model,
        compress=0,
        protocol=5,
    )
    loaded_bundle = joblib.load(temporary_model)
    if (
        loaded_bundle.get("scope_id") != scope_id
        or loaded_bundle.get("candidate_id") != selected_summary["candidate_id"]
        or tuple(loaded_bundle.get("feature_order", [])) != feature_order
    ):
        temporary_model.unlink(missing_ok=True)
        raise ValueError(f"Frozen model roundtrip metadata mismatch for {scope_id}")
    x_verify, _y_verify, _r_verify, _f_verify = _arrays(
        by_split["model_selection"]
    )
    roundtrip_probabilities = _predict_probabilities(
        loaded_bundle["model"], x_verify
    )
    original_probabilities = _predict_probabilities(selected_model, x_verify)
    if not np.array_equal(roundtrip_probabilities, original_probabilities):
        temporary_model.unlink(missing_ok=True)
        raise ValueError(f"Frozen model roundtrip changed probabilities for {scope_id}")
    try:
        os.link(temporary_model, model_path)
    finally:
        temporary_model.unlink(missing_ok=True)
    model_sha = _sha256(model_path)
    ranking = sorted(
        (item[0] for item in fitted),
        key=lambda row: (
            -float(row["model_selection_selector_accuracy"]),
            int(row["ordinal"]),
        ),
    )
    return {
        "scope_id": scope_id,
        "development_counts": {name: len(values) for name, values in by_split.items()},
        "active_pairs": sorted({item.pair for item in observations}),
        "active_seeds": sorted({item.seed for item in observations}),
        "active_tasks": sorted({item.task for item in observations}),
        "comparator": comparator,
        "selected_candidate": selected_summary,
        "candidate_ranking": ranking,
        "model_path": str(model_path.resolve()),
        "model_sha256": model_sha,
        "selected_base_state_summary": _estimator_summary(selected_base, selected_model),
    }


def _manifest_hash_record(path: Path) -> Dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _verify_hash_record(record: Mapping[str, Any], expected_path: Path | None = None) -> Path:
    path = Path(str(record["path"])).resolve()
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"Frozen path mismatch: {path} != {expected_path.resolve()}")
    actual = _sha256(path)
    if actual != str(record["sha256"]):
        raise ValueError(f"Frozen SHA mismatch for {path}: {actual} != {record['sha256']}")
    return path


def _require_main_at_origin(expected_head: str) -> None:
    _require_clean_worktree()
    branch = _git("symbolic-ref", "--short", "HEAD")
    if branch != "main":
        raise ValueError(f"Formal stage requires branch main, found {branch}")
    head = _git_head()
    origin = _git("rev-parse", "origin/main")
    if head != expected_head or origin != expected_head:
        raise ValueError(
            f"Formal stage requires HEAD==origin/main=={expected_head}; "
            f"found HEAD={head}, origin/main={origin}"
        )


def _source_artifact_digest(layout: SourceLayout) -> str:
    return hashlib.sha256(
        json.dumps(
            list(layout.source_artifacts), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def freeze_design(
    protocol_path: Path,
    candidate_path: Path,
    feature_path: Path,
    split_path: Path,
    split_sha_path: Path,
    output_path: Path,
    expected_implementation_commit: str,
) -> Dict[str, Any]:
    """Freeze committed code/design inputs before any outcome-driven fitting."""

    _require_main_at_origin(expected_implementation_commit)
    if output_path.exists():
        raise FileExistsError(f"Design freeze already exists: {output_path}")
    protocol, _candidates, _features = _load_protocol_bundle(
        protocol_path, candidate_path, feature_path
    )
    source_commit = str(protocol["source_commit"])
    ancestor_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
    )
    if ancestor_check.returncode != 0:
        raise ValueError(f"Frozen source commit {source_commit} is not an ancestor of HEAD")
    runtime = _validate_runtime(protocol)
    split_manifest = _load_split_manifest(split_path, split_sha_path, protocol)
    layout = _load_source_layout(protocol)
    freeze = {
        "schema_version": 1,
        "phase": "2A-1",
        "role": "pre_outcome_design_and_code_freeze",
        "created_without_selector_outcome_access": True,
        "implementation_commit": expected_implementation_commit,
        "implementation_script": _manifest_hash_record(SCRIPT_PATH),
        "inputs": {
            "protocol": _manifest_hash_record(protocol_path),
            "candidates": _manifest_hash_record(candidate_path),
            "features": _manifest_hash_record(feature_path),
            "split_manifest": _manifest_hash_record(split_path),
            "split_sha_file": _manifest_hash_record(split_sha_path),
            "source_artifact_digest": _source_artifact_digest(layout),
            "source_artifacts": list(layout.source_artifacts),
        },
        "split_summary": split_manifest["counts"],
        "runtime": runtime,
        "thread_environment": {
            name: os.environ[name]
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    _write_json_once(output_path, freeze)
    return freeze


def _validate_design_freeze(
    design_freeze_path: Path,
    *,
    protocol_path: Path,
    candidate_path: Path,
    feature_path: Path,
    split_path: Path,
    split_sha_path: Path,
    expected_design_commit: str | None = None,
) -> Dict[str, Any]:
    freeze = _read_json(design_freeze_path)
    if (
        freeze.get("role") != "pre_outcome_design_and_code_freeze"
        or freeze.get("created_without_selector_outcome_access") is not True
    ):
        raise ValueError("Invalid Phase2A-1 design freeze")
    inputs = freeze["inputs"]
    for name, path in (
        ("protocol", protocol_path),
        ("candidates", candidate_path),
        ("features", feature_path),
        ("split_manifest", split_path),
        ("split_sha_file", split_sha_path),
    ):
        _verify_hash_record(inputs[name], path)
    _verify_hash_record(freeze["implementation_script"], SCRIPT_PATH)
    implementation_commit = str(freeze["implementation_commit"])
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation_commit, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("Frozen implementation commit is not an ancestor of HEAD")
    if expected_design_commit is not None:
        _verify_single_parent_commit_diff(
            expected_design_commit,
            implementation_commit,
            [design_freeze_path],
        )
    protected = [
        str(SCRIPT_PATH.relative_to(REPO_ROOT)),
        str(protocol_path.resolve().relative_to(REPO_ROOT)),
        str(candidate_path.resolve().relative_to(REPO_ROOT)),
        str(feature_path.resolve().relative_to(REPO_ROOT)),
        str(split_path.resolve().relative_to(REPO_ROOT)),
        str(split_sha_path.resolve().relative_to(REPO_ROOT)),
    ]
    changed = _git("diff", "--name-only", implementation_commit, "HEAD", "--", *protected)
    if changed:
        raise ValueError(f"Frozen executable/design changed after implementation commit:\n{changed}")
    return freeze


def fit_and_select(
    protocol_path: Path,
    candidate_path: Path,
    feature_path: Path,
    split_path: Path,
    split_sha_path: Path,
    output_path: Path,
    model_dir: Path,
    design_freeze_path: Path,
    expected_design_commit: str,
) -> Dict[str, Any]:
    _require_main_at_origin(expected_design_commit)
    if output_path.exists() or model_dir.exists():
        raise FileExistsError("Selection lock/model directory already exists; fit-select is write-once")
    protocol, candidates, feature_manifest = _load_protocol_bundle(
        protocol_path, candidate_path, feature_path
    )
    runtime = _validate_runtime(protocol)
    design_freeze = _validate_design_freeze(
        design_freeze_path,
        protocol_path=protocol_path,
        candidate_path=candidate_path,
        feature_path=feature_path,
        split_path=split_path,
        split_sha_path=split_sha_path,
        expected_design_commit=expected_design_commit,
    )
    split_manifest = _load_split_manifest(split_path, split_sha_path, protocol)
    layout = _load_source_layout(protocol)
    if _source_artifact_digest(layout) != design_freeze["inputs"][
        "source_artifact_digest"
    ]:
        raise ValueError("Source artifact digest changed after design freeze")

    all_pairs = [item.pair for item in layout.pairs]
    all_seeds = list(layout.seeds)
    all_tasks = [item.task for item in layout.tasks]
    scope_specs: List[Tuple[str, List[str], List[int], List[str], Dict[str, Any]]] = [
        ("global", all_pairs, all_seeds, all_tasks, {"type": "global"})
    ]
    for seed in layout.seeds:
        scope_specs.append(
            (
                f"leave_one_seed_{seed}",
                all_pairs,
                [value for value in all_seeds if value != seed],
                all_tasks,
                {"type": "leave_one_seed_out", "held_out_seed": seed},
            )
        )
    for task in layout.tasks:
        scope_specs.append(
            (
                f"leave_one_task_{task.task}",
                all_pairs,
                all_seeds,
                [value for value in all_tasks if value != task.task],
                {"type": "leave_one_task_out", "held_out_task": task.task},
            )
        )
    for pair in layout.pairs:
        scope_specs.append(
            (
                f"leave_one_pair_{pair.pair}",
                [value for value in all_pairs if value != pair.pair],
                all_seeds,
                all_tasks,
                {"type": "leave_one_pair_out", "held_out_pair": pair.pair},
            )
        )

    scope_results: Dict[str, Any] = {}
    loader_audits: Dict[str, Any] = {}
    for scope_id, scope_pairs, scope_seeds, scope_tasks, evaluation in scope_specs:
        scoped_observations, scope_audit, _scope_layout = load_observations(
            protocol,
            feature_manifest,
            split_manifest,
            stage="develop",
            include_pairs=scope_pairs,
            include_seeds=scope_seeds,
            include_tasks=scope_tasks,
            source_layout=layout,
        )
        if scope_audit["receiver_outcome_rows_parsed_by_split"]["test"] != 0 or scope_audit[
            "fused_outcome_rows_parsed_by_split"
        ]["test"] != 0:
            raise AssertionError(f"Development outcome-access audit failed for {scope_id}")
        loader_audits[scope_id] = scope_audit
        print(f"Fitting frozen candidate matrix for {scope_id} ...", flush=True)
        model_path = model_dir / f"{scope_id}.joblib"
        result = _scope_selection(
            scoped_observations,
            candidates,
            feature_manifest,
            protocol,
            scope_id=scope_id,
            model_path=model_path,
        )
        result["evaluation"] = evaluation
        scope_results[scope_id] = result

    source_artifact_digest = _source_artifact_digest(layout)
    lock = {
        "schema_version": 1,
        "phase": "2A-1",
        "role": "development_selection_lock",
        "implementation_commit": design_freeze["implementation_commit"],
        "development_execution_commit": expected_design_commit,
        "design_freeze": _manifest_hash_record(design_freeze_path),
        "implementation_script": _manifest_hash_record(SCRIPT_PATH),
        "created_after_test_outcomes": False,
        "test_outcome_rows_parsed": 0,
        "loader_audit": loader_audits,
        "runtime": runtime,
        "thread_environment": {
            name: os.environ[name]
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "inputs": {
            "protocol": _manifest_hash_record(protocol_path),
            "candidates": _manifest_hash_record(candidate_path),
            "features": _manifest_hash_record(feature_path),
            "split_manifest": _manifest_hash_record(split_path),
            "split_sha_file": _manifest_hash_record(split_sha_path),
            "source_artifact_digest": source_artifact_digest,
            "source_artifacts": list(layout.source_artifacts),
        },
        "global_selected_candidate_id": scope_results["global"]["selected_candidate"][
            "candidate_id"
        ],
        "global_threshold": scope_results["global"]["selected_candidate"]["threshold"],
        "global_comparator": scope_results["global"]["comparator"],
        "scopes": scope_results,
        "sealed_test_contract": protocol["sealed_test"],
    }
    _write_json_once(output_path, lock)
    return lock


def _evaluation_weights(
    observations: Sequence[Observation], weighting: str
) -> np.ndarray:
    if weighting in {"task_macro", "single_task"}:
        return _balanced_weights(observations)
    if weighting != "sample_weighted":
        raise ValueError(f"Unknown evaluation weighting {weighting}")
    counts: Dict[Tuple[str, int], int] = {}
    for item in observations:
        key = (item.pair, item.seed)
        counts[key] = counts.get(key, 0) + 1
    expected = {
        (pair, seed)
        for pair in {item.pair for item in observations}
        for seed in {item.seed for item in observations}
    }
    if set(counts) != expected:
        raise ValueError("Sample-weighted pair/seed cells are incomplete")
    total = len(observations)
    weights = np.asarray(
        [total / (len(counts) * counts[(item.pair, item.seed)]) for item in observations],
        dtype=np.float64,
    )
    if not math.isclose(float(weights.mean()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("Sample-weighted evaluation weights must have mean one")
    return weights


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(np.asarray(values, dtype=np.float64), weights=weights))


def _safe_average_precision(
    target: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> float | None:
    target = np.asarray(target, dtype=np.int8)
    positive_weight = float(weights[target == 1].sum())
    negative_weight = float(weights[target == 0].sum())
    if positive_weight <= 0.0 or negative_weight <= 0.0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return float(average_precision_score(target, scores, sample_weight=weights))


def _binary_ece(
    target: np.ndarray, probability: np.ndarray, weights: np.ndarray, bins: int = 15
) -> float:
    target = np.asarray(target, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    indices = np.minimum((probability * bins).astype(np.int64), bins - 1)
    total_weight = float(weights.sum())
    value = 0.0
    for index in range(bins):
        mask = indices == index
        if not mask.any():
            continue
        bin_weight = float(weights[mask].sum())
        observed = float(np.average(target[mask], weights=weights[mask]))
        predicted = float(np.average(probability[mask], weights=weights[mask]))
        value += bin_weight / total_weight * abs(observed - predicted)
    return value


def _probability_metrics(
    observations: Sequence[Observation],
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, Any]:
    _validate_probabilities(probabilities)
    utility = np.asarray([item.utility for item in observations], dtype=np.int64)
    one_hot = np.zeros_like(probabilities)
    for index, target_class in enumerate(CLASS_ORDER):
        one_hot[:, index] = utility == target_class
    brier = _weighted_mean(np.square(probabilities - one_hot).sum(axis=1), weights)
    class_ece = [
        _binary_ece(one_hot[:, index], probabilities[:, index], weights)
        for index in range(3)
    ]
    benefit_target = (utility == 1).astype(np.int8)
    harm_target = (utility == -1).astype(np.int8)
    output: Dict[str, Any] = {
        "benefit_auprc_pooled": _safe_average_precision(
            benefit_target, probabilities[:, 2], weights
        ),
        "harm_auprc_pooled": _safe_average_precision(
            harm_target, probabilities[:, 0], weights
        ),
        "multiclass_brier": brier,
        "ece_multiclass_macro": float(np.mean(class_ece)),
        "ece_harm": class_ece[0],
        "ece_neutral": class_ece[1],
        "ece_benefit": class_ece[2],
    }
    cell_values: Dict[str, List[float]] = {"benefit": [], "harm": []}
    undefined = {"benefit": 0, "harm": 0}
    cells = sorted({(item.pair, item.seed, item.task) for item in observations})
    for cell in cells:
        mask = np.asarray(
            [(item.pair, item.seed, item.task) == cell for item in observations],
            dtype=bool,
        )
        cell_weights = np.ones(int(mask.sum()), dtype=np.float64)
        for name, target, score in (
            ("benefit", benefit_target[mask], probabilities[mask, 2]),
            ("harm", harm_target[mask], probabilities[mask, 0]),
        ):
            value = _safe_average_precision(target, score, cell_weights)
            if value is None:
                undefined[name] += 1
            else:
                cell_values[name].append(value)
    for name in ("benefit", "harm"):
        output[f"{name}_auprc_cell_macro"] = (
            None
            if undefined[name]
            else float(np.mean(cell_values[name]))
        )
        output[f"{name}_auprc_undefined_cell_count"] = undefined[name]
    output["benefit_prevalence"] = _weighted_mean(benefit_target, weights)
    output["harm_prevalence"] = _weighted_mean(harm_target, weights)
    return output


def _metric_values(
    observations: Sequence[Observation],
    actions: np.ndarray,
    comparator_policy: str,
    weighting: str,
    probabilities: np.ndarray | None,
) -> Dict[str, Any]:
    if len(observations) != len(actions):
        raise ValueError("Observation/action length mismatch")
    weights = _evaluation_weights(observations, weighting)
    receiver = np.asarray(
        [item.receiver_correct for item in observations], dtype=np.float64
    )
    fused = np.asarray([item.fused_correct for item in observations], dtype=np.float64)
    utility = fused - receiver
    action = np.asarray(actions, dtype=bool)
    selector = receiver + action.astype(np.float64) * utility
    if comparator_policy == "always_receiver":
        comparator = receiver
    elif comparator_policy == "always_fused":
        comparator = fused
    else:
        raise ValueError(f"Unknown comparator {comparator_policy}")
    oracle = np.maximum(receiver, fused)
    receiver_accuracy = _weighted_mean(receiver, weights)
    fused_accuracy = _weighted_mean(fused, weights)
    selector_accuracy = _weighted_mean(selector, weights)
    comparator_accuracy = _weighted_mean(comparator, weights)
    oracle_accuracy = _weighted_mean(oracle, weights)
    paired_delta = _weighted_mean(selector - comparator, weights)
    retrospective_best = max(receiver_accuracy, fused_accuracy)
    harm = utility == -1
    benefit = utility == 1
    harm_rate = _weighted_mean(harm, weights)
    benefit_rate = _weighted_mean(benefit, weights)
    accepted_harm_rate = _weighted_mean(action & harm, weights)
    accepted_benefit_rate = _weighted_mean(action & benefit, weights)
    harm_reduction = (
        None if harm_rate <= 0.0 else 1.0 - accepted_harm_rate / harm_rate
    )
    benefit_retention = (
        None if benefit_rate <= 0.0 else accepted_benefit_rate / benefit_rate
    )
    recovery_denominator = oracle_accuracy - retrospective_best
    recovery = (
        None
        if recovery_denominator <= 0.0
        else (selector_accuracy - retrospective_best) / recovery_denominator
    )
    output: Dict[str, Any] = {
        "n_rows": len(observations),
        "n_content_groups": len({item.content_hash for item in observations}),
        "receiver_accuracy": receiver_accuracy,
        "fused_accuracy": fused_accuracy,
        "selector_accuracy": selector_accuracy,
        "comparator_accuracy": comparator_accuracy,
        "selector_minus_comparator": paired_delta,
        "oracle_accuracy": oracle_accuracy,
        "retrospective_best_fixed_accuracy": retrospective_best,
        "oracle_headroom_over_best_fixed": oracle_accuracy - retrospective_best,
        "selector_headroom_below_oracle": oracle_accuracy - selector_accuracy,
        "oracle_headroom_recovery": recovery,
        "transfer_rate": _weighted_mean(action, weights),
        "abstention_rate": 1.0 - _weighted_mean(action, weights),
        "harmful_event_rate": harm_rate,
        "accepted_harmful_rate": accepted_harm_rate,
        "harmful_reduction": harm_reduction,
        "beneficial_event_rate": benefit_rate,
        "accepted_beneficial_rate": accepted_benefit_rate,
        "beneficial_retention": benefit_retention,
    }
    if probabilities is not None:
        output.update(_probability_metrics(observations, probabilities, weights))
    else:
        for name in (
            "benefit_auprc_pooled",
            "harm_auprc_pooled",
            "benefit_auprc_cell_macro",
            "harm_auprc_cell_macro",
            "multiclass_brier",
            "ece_multiclass_macro",
            "ece_harm",
            "ece_neutral",
            "ece_benefit",
        ):
            output[name] = None
    return output


def _load_scope_model(
    selection_lock: Mapping[str, Any],
    scope_id: str,
    feature_order: Sequence[str],
    candidate_manifest: Mapping[str, Any] | None = None,
) -> Any:
    scope = selection_lock["scopes"][scope_id]
    path = Path(str(scope["model_path"])).resolve()
    if _sha256(path) != str(scope["model_sha256"]):
        raise ValueError(f"Model SHA mismatch for {scope_id}")
    bundle = joblib.load(path)
    expected_candidate = scope["selected_candidate"]["candidate_id"]
    if (
        bundle.get("scope_id") != scope_id
        or bundle.get("candidate_id") != expected_candidate
        or tuple(bundle.get("feature_order", [])) != tuple(feature_order)
    ):
        raise ValueError(f"Model metadata mismatch for {scope_id}")
    model = bundle.get("model")
    classes = np.asarray(model.classes_, dtype=np.int64)
    if not np.array_equal(classes, CLASS_ORDER):
        raise ValueError(f"Model class order mismatch for {scope_id}: {classes}")
    if not isinstance(model, CalibratedClassifierCV) or not isinstance(
        model.estimator, FrozenEstimator
    ):
        raise ValueError(f"Model wrapper mismatch for {scope_id}")
    base = model.estimator.estimator
    if not isinstance(base, Pipeline):
        raise ValueError(f"Base estimator is not a Pipeline for {scope_id}")
    if _estimator_summary(base, model) != scope["selected_candidate"]["state_summary"]:
        raise ValueError(f"Model state summary mismatch for {scope_id}")
    if candidate_manifest is not None:
        candidate_by_id = {
            str(item["id"]): item for item in candidate_manifest["candidates"]
        }
        candidate = candidate_by_id.get(str(expected_candidate))
        if candidate is None:
            raise ValueError(f"Unknown selected candidate for {scope_id}")
        summary = scope["selected_candidate"]
        if (
            summary["family"] != candidate["family"]
            or summary["features"] != candidate["features"]
            or int(summary["ordinal"]) != int(candidate["ordinal"])
        ):
            raise ValueError(f"Selected candidate manifest mismatch for {scope_id}")
        expected_steps = (
            ("select", "scale", "clf")
            if candidate["family"] == "l2_multinomial_logistic"
            else ("select", "clf")
        )
        if tuple(base.named_steps) != expected_steps:
            raise ValueError(f"Pipeline steps mismatch for {scope_id}")
        selector = base.named_steps["select"]
        selected_indices = list(selector.transformers_[0][2])
        if selected_indices != _feature_indices(candidate, feature_order):
            raise ValueError(f"Selected feature indices mismatch for {scope_id}")
        classifier = base.named_steps["clf"]
        params = candidate["params"]
        if candidate["family"] == "l2_multinomial_logistic":
            if not isinstance(classifier, LogisticRegression):
                raise ValueError(f"Classifier family mismatch for {scope_id}")
            expected_params = {
                "C": float(params["C"]),
                "penalty": params["penalty"],
                "solver": params["solver"],
                "tol": float(params["tol"]),
                "max_iter": int(params["max_iter"]),
                "fit_intercept": bool(params["fit_intercept"]),
                "class_weight": params["class_weight"],
                "random_state": int(params["random_state"]),
            }
        else:
            if not isinstance(classifier, DecisionTreeClassifier):
                raise ValueError(f"Classifier family mismatch for {scope_id}")
            expected_params = {
                "criterion": params["criterion"],
                "splitter": params["splitter"],
                "max_depth": int(params["max_depth"]),
                "min_weight_fraction_leaf": float(
                    params["min_weight_fraction_leaf"]
                ),
                "class_weight": params["class_weight"],
                "random_state": int(params["random_state"]),
            }
        actual_params = classifier.get_params(deep=False)
        for name, expected in expected_params.items():
            if actual_params.get(name) != expected:
                raise ValueError(
                    f"Classifier parameter mismatch for {scope_id}/{name}: "
                    f"{actual_params.get(name)} != {expected}"
                )
    return model


def _predict_actions(
    observations: Sequence[Observation], model: Any, threshold_record: Mapping[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, _y, _receiver, _fused = _arrays(observations)
    probabilities = _predict_probabilities(model, x)
    scores = _score(probabilities)
    actions = scores > _threshold_value(threshold_record)
    return probabilities, scores, actions


def _same_rate_random_actions(
    observations: Sequence[Observation],
    selector_actions: np.ndarray,
    protocol: Mapping[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if len(observations) != len(selector_actions):
        raise ValueError("Random baseline action length mismatch")
    seed = int(protocol["random_baseline"]["seed"])
    actions = np.zeros(len(observations), dtype=bool)
    audit: Dict[str, Any] = {}
    strata = sorted({(item.pair, item.task) for item in observations})
    for pair, task in strata:
        indices = [
            index
            for index, item in enumerate(observations)
            if item.pair == pair and item.task == task
        ]
        target = int(np.asarray(selector_actions, dtype=np.int64)[indices].sum())
        groups: Dict[str, List[int]] = {}
        for index in indices:
            groups.setdefault(observations[index].content_hash, []).append(index)
        ranked = sorted(
            groups,
            key=lambda content_hash: hashlib.sha256(
                (
                    f"phase2a1-random-v1|{seed}|{pair}|{task}|{content_hash}"
                ).encode("utf-8")
            ).digest(),
        )
        cumulative = [0]
        for content_hash in ranked:
            cumulative.append(cumulative[-1] + len(groups[content_hash]))
        prefix = min(
            range(len(cumulative)),
            key=lambda value: (abs(cumulative[value] - target), cumulative[value], value),
        )
        selected_groups = set(ranked[:prefix])
        for index in indices:
            actions[index] = observations[index].content_hash in selected_groups
        achieved = int(actions[indices].sum())
        audit[f"{pair}|{task}"] = {
            "target_transferred_rows": target,
            "achieved_transferred_rows": achieved,
            "total_rows": len(indices),
            "absolute_rate_gap": abs(achieved - target) / len(indices),
            "selected_content_groups": prefix,
            "total_content_groups": len(ranked),
        }
        for content_hash, group_indices in groups.items():
            if len({bool(actions[index]) for index in group_indices}) != 1:
                raise AssertionError(f"Random action differs within group {content_hash}")
    return actions, audit


COMPONENT_INDEX = {
    "receiver": 0,
    "fused": 1,
    "oracle": 2,
    "comparator": 3,
    "harm": 4,
    "benefit": 5,
    "selected_correct": 6,
    "selected_action": 7,
    "selected_accepted_harm": 8,
    "selected_accepted_benefit": 9,
    "random_correct": 10,
    "random_action": 11,
    "random_accepted_harm": 12,
    "random_accepted_benefit": 13,
    "selected_minus_comparator": 14,
    "random_minus_comparator": 15,
}
COMPONENT_COUNT = len(COMPONENT_INDEX)


def _component_vector(
    item: Observation,
    selected_action: bool,
    random_action: bool,
    comparator_policy: str,
) -> np.ndarray:
    receiver = float(item.receiver_correct)
    fused = float(item.fused_correct)
    utility = item.utility
    harm = float(utility == -1)
    benefit = float(utility == 1)
    comparator = receiver if comparator_policy == "always_receiver" else fused
    if comparator_policy not in {"always_receiver", "always_fused"}:
        raise ValueError(f"Unknown comparator {comparator_policy}")
    values = np.zeros(COMPONENT_COUNT, dtype=np.float64)
    values[COMPONENT_INDEX["receiver"]] = receiver
    values[COMPONENT_INDEX["fused"]] = fused
    values[COMPONENT_INDEX["oracle"]] = max(receiver, fused)
    values[COMPONENT_INDEX["comparator"]] = comparator
    values[COMPONENT_INDEX["harm"]] = harm
    values[COMPONENT_INDEX["benefit"]] = benefit
    values[COMPONENT_INDEX["selected_correct"]] = (
        fused if selected_action else receiver
    )
    values[COMPONENT_INDEX["selected_action"]] = float(selected_action)
    values[COMPONENT_INDEX["selected_accepted_harm"]] = float(
        selected_action and utility == -1
    )
    values[COMPONENT_INDEX["selected_accepted_benefit"]] = float(
        selected_action and utility == 1
    )
    values[COMPONENT_INDEX["random_correct"]] = fused if random_action else receiver
    values[COMPONENT_INDEX["random_action"]] = float(random_action)
    values[COMPONENT_INDEX["random_accepted_harm"]] = float(
        random_action and utility == -1
    )
    values[COMPONENT_INDEX["random_accepted_benefit"]] = float(
        random_action and utility == 1
    )
    values[COMPONENT_INDEX["selected_minus_comparator"]] = (
        values[COMPONENT_INDEX["selected_correct"]] - comparator
    )
    values[COMPONENT_INDEX["random_minus_comparator"]] = (
        values[COMPONENT_INDEX["random_correct"]] - comparator
    )
    return values


def _build_bootstrap_tensors(
    observations: Sequence[Observation],
    selected_actions: np.ndarray,
    random_actions: np.ndarray,
    comparator_policy: str,
    *,
    pair_order: Sequence[str],
    seed_order: Sequence[int],
    task_order: Sequence[str],
    samples: int,
    base_seed: int,
    batch_size: int,
    context: str,
) -> Dict[str, Any]:
    if samples <= 0 or batch_size <= 0:
        raise ValueError("Bootstrap samples and batch size must be positive")
    if not (len(observations) == len(selected_actions) == len(random_actions)):
        raise ValueError("Bootstrap observation/action length mismatch")
    pairs = tuple(pair_order)
    seeds = tuple(map(int, seed_order))
    tasks = tuple(task_order)
    pair_index = {value: index for index, value in enumerate(pairs)}
    seed_index = {value: index for index, value in enumerate(seeds)}
    tensors: Dict[str, Any] = {}
    for task in tasks:
        task_indices = [
            index for index, item in enumerate(observations) if item.task == task
        ]
        if not task_indices:
            raise ValueError(f"Bootstrap task {task} has no rows")
        groups = sorted({observations[index].content_hash for index in task_indices})
        group_index = {value: index for index, value in enumerate(groups)}
        group_sums = np.zeros(
            (len(groups), len(pairs), len(seeds), COMPONENT_COUNT),
            dtype=np.float64,
        )
        group_counts = np.zeros(
            (len(groups), len(pairs), len(seeds)), dtype=np.float64
        )
        for index in task_indices:
            item = observations[index]
            try:
                p_index = pair_index[item.pair]
                s_index = seed_index[item.seed]
            except KeyError as exc:
                raise ValueError(f"Observation outside bootstrap scope: {item.key}") from exc
            g_index = group_index[item.content_hash]
            group_sums[g_index, p_index, s_index] += _component_vector(
                item,
                bool(selected_actions[index]),
                bool(random_actions[index]),
                comparator_policy,
            )
            group_counts[g_index, p_index, s_index] += 1.0
        reference_counts = group_counts[:, 0, 0]
        if (reference_counts <= 0.0).any():
            raise ValueError(f"Bootstrap group missing reference cell for {task}")
        if not np.array_equal(
            group_counts,
            np.broadcast_to(
                reference_counts[:, None, None], group_counts.shape
            ),
        ):
            raise ValueError(
                f"Content-group membership differs across pair/seed cells for {task}"
            )
        point_sums = group_sums.sum(axis=0)
        point_counts = group_counts.sum(axis=0)
        boot_sums = np.empty(
            (samples, len(pairs), len(seeds), COMPONENT_COUNT), dtype=np.float64
        )
        boot_counts = np.empty(
            (samples, len(pairs), len(seeds)), dtype=np.float64
        )
        rng = np.random.default_rng(
            _stable_seed(base_seed, f"content-groups:{context}:{task}")
        )
        probabilities = np.full(len(groups), 1.0 / len(groups), dtype=np.float64)
        for start in range(0, samples, batch_size):
            end = min(samples, start + batch_size)
            draws = rng.multinomial(len(groups), probabilities, size=end - start)
            boot_sums[start:end] = np.einsum(
                "bg,gpsk->bpsk", draws, group_sums, optimize=True
            )
            boot_counts[start:end] = np.einsum(
                "bg,gps->bps", draws, group_counts, optimize=True
            )
        tensors[task] = {
            "groups": groups,
            "point_sums": point_sums,
            "point_counts": point_counts,
            "boot_sums": boot_sums,
            "boot_counts": boot_counts,
        }
    return {
        "pairs": pairs,
        "seeds": seeds,
        "tasks": tasks,
        "samples": samples,
        "task_tensors": tensors,
    }


def _select_cluster_tensors(
    sums: np.ndarray,
    counts: np.ndarray,
    selected_pairs: np.ndarray,
    selected_seeds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    sample_count, pair_slots = selected_pairs.shape
    seed_slots = selected_seeds.shape[2]
    rows = np.arange(sample_count)
    output_sums = np.empty(
        (sample_count, pair_slots, seed_slots, sums.shape[-1]), dtype=np.float64
    )
    output_counts = np.empty(
        (sample_count, pair_slots, seed_slots), dtype=np.float64
    )
    for pair_slot in range(pair_slots):
        for seed_slot in range(seed_slots):
            output_sums[:, pair_slot, seed_slot] = sums[
                rows,
                selected_pairs[:, pair_slot],
                selected_seeds[:, pair_slot, seed_slot],
            ]
            output_counts[:, pair_slot, seed_slot] = counts[
                rows,
                selected_pairs[:, pair_slot],
                selected_seeds[:, pair_slot, seed_slot],
            ]
    return output_sums, output_counts


def _aggregate_tensor_scope(
    tensors: Mapping[str, Any],
    *,
    pairs: Sequence[str],
    seeds: Sequence[int],
    tasks: Sequence[str],
    weighting: str,
    resample_pairs: bool,
    resample_seeds: bool,
    base_seed: int,
    label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    all_pairs = tuple(tensors["pairs"])
    all_seeds = tuple(tensors["seeds"])
    pair_indices = np.asarray([all_pairs.index(value) for value in pairs], dtype=np.int64)
    seed_indices = np.asarray([all_seeds.index(int(value)) for value in seeds], dtype=np.int64)
    if not len(pair_indices) or not len(seed_indices) or not tasks:
        raise ValueError("Bootstrap scope is empty")

    point_task_rates: List[np.ndarray] = []
    point_task_sums: List[np.ndarray] = []
    point_task_counts: List[float] = []
    for task in tasks:
        tensor = tensors["task_tensors"][task]
        sums = tensor["point_sums"][np.ix_(pair_indices, seed_indices)]
        counts = tensor["point_counts"][np.ix_(pair_indices, seed_indices)]
        cell_rates = sums / counts[..., None]
        point_task_rates.append(cell_rates.mean(axis=(0, 1)))
        point_task_sums.append(sums)
        point_task_counts.append(float(counts.sum()))
    if weighting in {"task_macro", "single_task"}:
        point = np.mean(point_task_rates, axis=0)
    elif weighting == "sample_weighted":
        cell_sums = np.zeros(
            (len(pair_indices), len(seed_indices), COMPONENT_COUNT), dtype=np.float64
        )
        cell_counts = np.zeros((len(pair_indices), len(seed_indices)), dtype=np.float64)
        for task in tasks:
            tensor = tensors["task_tensors"][task]
            cell_sums += tensor["point_sums"][np.ix_(pair_indices, seed_indices)]
            cell_counts += tensor["point_counts"][np.ix_(pair_indices, seed_indices)]
        point = (cell_sums / cell_counts[..., None]).mean(axis=(0, 1))
    else:
        raise ValueError(f"Unknown bootstrap weighting {weighting}")

    sample_count = int(tensors["samples"])
    rng = np.random.default_rng(_stable_seed(base_seed, f"clusters:{label}"))
    if resample_pairs:
        selected_pair_offsets = rng.integers(
            0, len(pair_indices), size=(sample_count, len(pair_indices))
        )
        selected_pairs = pair_indices[selected_pair_offsets]
    else:
        selected_pairs = np.broadcast_to(
            pair_indices, (sample_count, len(pair_indices))
        )
    if resample_seeds:
        selected_seed_offsets = rng.integers(
            0,
            len(seed_indices),
            size=(sample_count, len(pair_indices), len(seed_indices)),
        )
        selected_seeds = seed_indices[selected_seed_offsets]
    else:
        selected_seeds = np.broadcast_to(
            seed_indices[None, None, :],
            (sample_count, len(pair_indices), len(seed_indices)),
        )

    boot_task_rates: List[np.ndarray] = []
    selected_by_task: List[Tuple[np.ndarray, np.ndarray]] = []
    for task in tasks:
        tensor = tensors["task_tensors"][task]
        selected_sums, selected_counts = _select_cluster_tensors(
            tensor["boot_sums"],
            tensor["boot_counts"],
            selected_pairs,
            selected_seeds,
        )
        selected_by_task.append((selected_sums, selected_counts))
        boot_task_rates.append(
            (selected_sums / selected_counts[..., None]).mean(axis=(1, 2))
        )
    if weighting in {"task_macro", "single_task"}:
        bootstrap = np.mean(boot_task_rates, axis=0)
    else:
        aggregate_sums = np.zeros_like(selected_by_task[0][0])
        aggregate_counts = np.zeros_like(selected_by_task[0][1])
        for selected_sums, selected_counts in selected_by_task:
            aggregate_sums += selected_sums
            aggregate_counts += selected_counts
        bootstrap = (
            aggregate_sums / aggregate_counts[..., None]
        ).mean(axis=(1, 2))
    if not np.isfinite(point).all() or not np.isfinite(bootstrap).all():
        raise ValueError(f"Non-finite bootstrap components for {label}")
    return point, bootstrap


def _safe_ratio_array(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    output = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    return np.divide(numerator, denominator, out=output, where=denominator > 0.0)


def _metrics_from_component_rates(
    rates: np.ndarray, policy: str
) -> Dict[str, np.ndarray]:
    receiver = rates[..., COMPONENT_INDEX["receiver"]]
    fused = rates[..., COMPONENT_INDEX["fused"]]
    oracle = rates[..., COMPONENT_INDEX["oracle"]]
    comparator = rates[..., COMPONENT_INDEX["comparator"]]
    harm = rates[..., COMPONENT_INDEX["harm"]]
    benefit = rates[..., COMPONENT_INDEX["benefit"]]
    if policy == "selected":
        selected = rates[..., COMPONENT_INDEX["selected_correct"]]
        action = rates[..., COMPONENT_INDEX["selected_action"]]
        accepted_harm = rates[..., COMPONENT_INDEX["selected_accepted_harm"]]
        accepted_benefit = rates[..., COMPONENT_INDEX["selected_accepted_benefit"]]
        paired_delta = rates[..., COMPONENT_INDEX["selected_minus_comparator"]]
    elif policy == "same_rate_random":
        selected = rates[..., COMPONENT_INDEX["random_correct"]]
        action = rates[..., COMPONENT_INDEX["random_action"]]
        accepted_harm = rates[..., COMPONENT_INDEX["random_accepted_harm"]]
        accepted_benefit = rates[..., COMPONENT_INDEX["random_accepted_benefit"]]
        paired_delta = rates[..., COMPONENT_INDEX["random_minus_comparator"]]
    else:
        raise ValueError(f"Unknown component policy {policy}")
    retrospective = np.maximum(receiver, fused)
    recovery = _safe_ratio_array(selected - retrospective, oracle - retrospective)
    harm_reduction = 1.0 - _safe_ratio_array(accepted_harm, harm)
    benefit_retention = _safe_ratio_array(accepted_benefit, benefit)
    return {
        "receiver_accuracy": receiver,
        "fused_accuracy": fused,
        "selector_accuracy": selected,
        "comparator_accuracy": comparator,
        "selector_minus_comparator": paired_delta,
        "oracle_accuracy": oracle,
        "retrospective_best_fixed_accuracy": retrospective,
        "oracle_headroom_over_best_fixed": oracle - retrospective,
        "selector_headroom_below_oracle": oracle - selected,
        "oracle_headroom_recovery": recovery,
        "transfer_rate": action,
        "abstention_rate": 1.0 - action,
        "harmful_event_rate": harm,
        "accepted_harmful_rate": accepted_harm,
        "harmful_reduction": harm_reduction,
        "beneficial_event_rate": benefit,
        "accepted_beneficial_rate": accepted_benefit,
        "beneficial_retention": benefit_retention,
    }


def _quantile_bounds(values: np.ndarray, confidence: float) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Bootstrap confidence interval contains undefined draws")
    alpha = (1.0 - confidence) / 2.0
    result = np.quantile(values, [alpha, 1.0 - alpha], method="linear")
    return float(result[0]), float(result[1])


def _scope_specs(layout: SourceLayout) -> List[Dict[str, Any]]:
    pairs = [item.pair for item in layout.pairs]
    seeds = list(layout.seeds)
    tasks = [item.task for item in layout.tasks]
    specs: List[Dict[str, Any]] = []
    for pair in pairs:
        for seed in seeds:
            for task in tasks:
                specs.append(
                    {
                        "aggregation_level": "pair_seed_task",
                        "pair": pair,
                        "seed": seed,
                        "task": task,
                        "pairs": [pair],
                        "seeds": [seed],
                        "tasks": [task],
                        "weighting": "single_task",
                        "resample_pairs": False,
                        "resample_seeds": False,
                    }
                )
    for pair in pairs:
        for weighting in ("task_macro", "sample_weighted"):
            specs.append(
                {
                    "aggregation_level": "pair_across_seeds",
                    "pair": pair,
                    "seed": "all",
                    "task": "__all__",
                    "pairs": [pair],
                    "seeds": seeds,
                    "tasks": tasks,
                    "weighting": weighting,
                    "resample_pairs": False,
                    "resample_seeds": True,
                }
            )
    for seed in seeds:
        for weighting in ("task_macro", "sample_weighted"):
            specs.append(
                {
                    "aggregation_level": "seed_pair_balanced",
                    "pair": "__all__",
                    "seed": seed,
                    "task": "__all__",
                    "pairs": pairs,
                    "seeds": [seed],
                    "tasks": tasks,
                    "weighting": weighting,
                    "resample_pairs": True,
                    "resample_seeds": False,
                }
            )
    for task in tasks:
        specs.append(
            {
                "aggregation_level": "task_pair_balanced",
                "pair": "__all__",
                "seed": "all",
                "task": task,
                "pairs": pairs,
                "seeds": seeds,
                "tasks": [task],
                "weighting": "single_task",
                "resample_pairs": True,
                "resample_seeds": True,
            }
        )
    hetero = [item.pair for item in layout.pairs if item.heterogeneous]
    sensitivity = {
        "__all__": pairs,
        "__heterogeneous__": hetero,
        "__strict_cross_family__": ["tinyllama", "llama32_1b"],
        "__same_tokenizer__": ["qwen3_1p7b"],
    }
    for label, selected_pairs in sensitivity.items():
        if not set(selected_pairs).issubset(set(pairs)):
            raise ValueError(f"Unknown frozen sensitivity pair set {label}")
        for weighting in ("task_macro", "sample_weighted"):
            specs.append(
                {
                    "aggregation_level": "pair_balanced",
                    "pair": label,
                    "seed": "all",
                    "task": "__all__",
                    "pairs": selected_pairs,
                    "seeds": seeds,
                    "tasks": tasks,
                    "weighting": weighting,
                    "resample_pairs": len(selected_pairs) > 1,
                    "resample_seeds": True,
                }
            )
    return specs


def _filter_scope_indices(
    observations: Sequence[Observation], spec: Mapping[str, Any]
) -> np.ndarray:
    pairs = set(spec["pairs"])
    seeds = set(map(int, spec["seeds"]))
    tasks = set(spec["tasks"])
    return np.asarray(
        [
            index
            for index, item in enumerate(observations)
            if item.pair in pairs and item.seed in seeds and item.task in tasks
        ],
        dtype=np.int64,
    )


def _add_bootstrap_intervals(
    row: MutableMapping[str, Any],
    boot_metrics: Mapping[str, np.ndarray],
    confidence: float,
    *,
    require_primary_finite: bool,
) -> None:
    interval_metrics = (
        "receiver_accuracy",
        "fused_accuracy",
        "selector_accuracy",
        "comparator_accuracy",
        "selector_minus_comparator",
        "oracle_accuracy",
        "oracle_headroom_recovery",
        "transfer_rate",
        "harmful_reduction",
        "beneficial_retention",
    )
    for metric in interval_metrics:
        values = np.asarray(boot_metrics[metric], dtype=np.float64)
        finite = np.isfinite(values)
        row[f"{metric}_bootstrap_undefined"] = int((~finite).sum())
        if require_primary_finite and metric == "selector_minus_comparator" and not finite.all():
            raise ValueError("Primary bootstrap delta contains undefined draws")
        if not finite.all():
            row[f"{metric}_ci_low"] = None
            row[f"{metric}_ci_high"] = None
        else:
            low, high = _quantile_bounds(values, confidence)
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high


def _evaluate_specs(
    observations: Sequence[Observation],
    probabilities: np.ndarray,
    selected_actions: np.ndarray,
    random_actions: np.ndarray,
    comparator_policy: str,
    tensors: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    selected_candidate_id: str,
    threshold_record: Mapping[str, Any],
    include_random: bool,
    scope_prefix: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    confidence = float(protocol["bootstrap"]["confidence"])
    base_seed = int(protocol["bootstrap"]["seed"])
    for spec in specs:
        label = (
            f"{scope_prefix}:{spec['aggregation_level']}:{spec['pair']}:"
            f"{spec['seed']}:{spec['task']}:{spec['weighting']}"
        )
        point_rates, bootstrap_rates = _aggregate_tensor_scope(
            tensors,
            pairs=spec["pairs"],
            seeds=spec["seeds"],
            tasks=spec["tasks"],
            weighting=str(spec["weighting"]),
            resample_pairs=bool(spec["resample_pairs"]),
            resample_seeds=bool(spec["resample_seeds"]),
            base_seed=base_seed,
            label=label,
        )
        indices = _filter_scope_indices(observations, spec)
        scoped_observations = [observations[index] for index in indices]
        policies = ["selected"] + (["same_rate_random"] if include_random else [])
        for policy in policies:
            actions = selected_actions if policy == "selected" else random_actions
            policy_probabilities = probabilities[indices] if policy == "selected" else None
            point_metrics = _metric_values(
                scoped_observations,
                actions[indices],
                comparator_policy,
                str(spec["weighting"]),
                policy_probabilities,
            )
            component_point = _metrics_from_component_rates(point_rates, policy)
            for metric in (
                "receiver_accuracy",
                "fused_accuracy",
                "selector_accuracy",
                "comparator_accuracy",
                "selector_minus_comparator",
                "oracle_accuracy",
                "retrospective_best_fixed_accuracy",
                "oracle_headroom_over_best_fixed",
                "selector_headroom_below_oracle",
                "oracle_headroom_recovery",
                "transfer_rate",
                "accepted_harmful_rate",
                "harmful_reduction",
                "accepted_beneficial_rate",
                "beneficial_retention",
            ):
                expected = point_metrics[metric]
                actual = float(np.asarray(component_point[metric]))
                if expected is None:
                    if math.isfinite(actual):
                        raise AssertionError(f"Point component mismatch for {label}/{metric}")
                elif not math.isclose(
                    float(expected), actual, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise AssertionError(
                        f"Point component mismatch for {label}/{policy}/{metric}: "
                        f"{expected} != {actual}"
                    )
            row: Dict[str, Any] = {
                "scope": scope_prefix,
                "policy": policy,
                "candidate_id": selected_candidate_id if policy == "selected" else "same_rate_random",
                "threshold_kind": threshold_record["kind"] if policy == "selected" else None,
                "threshold_float_hex": threshold_record.get("float_hex") if policy == "selected" else None,
                "comparator_policy": comparator_policy,
                "aggregation_level": spec["aggregation_level"],
                "weighting": spec["weighting"],
                "pair": spec["pair"],
                "seed": spec["seed"],
                "task": spec["task"],
                "n_pairs": len(spec["pairs"]),
                "n_seeds": len(spec["seeds"]),
                "n_tasks": len(spec["tasks"]),
                "bootstrap_samples": int(protocol["bootstrap"]["samples"]),
                "bootstrap_confidence": confidence,
                **point_metrics,
            }
            boot_metrics = _metrics_from_component_rates(bootstrap_rates, policy)
            _add_bootstrap_intervals(
                row,
                boot_metrics,
                confidence,
                require_primary_finite=(
                    policy == "selected"
                    and spec["aggregation_level"] == "pair_balanced"
                    and spec["pair"] == "__all__"
                    and spec["weighting"] == "task_macro"
                ),
            )
            rows.append(row)
    return rows


def _validate_inputs_without_outcomes(
    layout: SourceLayout,
    feature_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate deployment-visible input/features without parsing correctness."""

    split_by_hash = _split_lookup(split_manifest)
    feature_order = tuple(map(str, feature_manifest["feature_order"]))
    receiver_keys: Dict[str, Dict[Tuple[str, str, str], str]] = {}
    audit: Dict[str, Any] = {
        "correctness_fields_interpreted_or_indexed": 0,
        "csv_row_materialization_note": "DictReader materializes rows containing outcome columns, but preflight never indexes, parses, branches on, or retains those fields",
        "receiver_rows": 0,
        "b6_rows": 0,
        "rows_by_split": {name: 0 for name in SPLITS},
        "cross_seed_feature_mismatches": 0,
    }
    feature_reference: Dict[
        Tuple[str, str, str, str, str], Tuple[float, ...]
    ] = {}
    for task in layout.tasks:
        hashes = _read_input_hashes(layout.receiver_paths[task.task], task.task)
        if len(hashes) != task.expected_rows:
            raise ValueError(f"Input-only receiver row mismatch for {task.task}")
        for content_hash in hashes.values():
            if content_hash not in split_by_hash:
                raise ValueError(f"Input-only validation missing split {content_hash}")
        receiver_keys[task.task] = hashes
        audit["receiver_rows"] += len(hashes)
    for pair in layout.pairs:
        for seed in layout.seeds:
            for task in layout.tasks:
                path = layout.b6_paths[(pair.pair, seed, task.task)]
                seen: set[Tuple[str, str, str]] = set()
                with path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    if reader.fieldnames is None or not set(feature_order).issubset(
                        reader.fieldnames
                    ):
                        raise ValueError(f"Input-only feature schema mismatch: {path}")
                    for row in reader:
                        key = _sample_key(task.task, row, path)
                        if key in seen:
                            raise ValueError(f"Input-only duplicate B6 key {key}")
                        seen.add(key)
                        expected_hash = receiver_keys[task.task].get(key)
                        if expected_hash is None or _content_hash(row) != expected_hash:
                            raise ValueError(f"Input-only content mismatch for {key}")
                        features = _parse_feature_vector(row, feature_order, path)
                        invariant_key = (
                            pair.pair,
                            task.task,
                            key[1],
                            key[2],
                            expected_hash,
                        )
                        previous = feature_reference.setdefault(invariant_key, features)
                        if previous != features:
                            audit["cross_seed_feature_mismatches"] += 1
                            raise ValueError(
                                f"Pre-transfer feature drift across seeds for {invariant_key}"
                            )
                        split = split_by_hash[expected_hash]
                        audit["rows_by_split"][split] += 1
                        audit["b6_rows"] += 1
                if seen != set(receiver_keys[task.task]):
                    raise ValueError(
                        f"Input-only key-set mismatch for {pair.pair}/{seed}/{task.task}"
                    )
    return audit


def _assert_seed_invariant_predictions(
    observations: Sequence[Observation],
    probabilities: np.ndarray,
    scores: np.ndarray,
    actions: np.ndarray,
) -> None:
    reference: Dict[
        Tuple[str, str, str, str, str], Tuple[Tuple[float, ...], np.ndarray, float, bool]
    ] = {}
    for item, probability, score, action in zip(
        observations, probabilities, scores, actions
    ):
        key = (
            item.pair,
            item.task,
            item.subject,
            item.question_id,
            item.content_hash,
        )
        if key not in reference:
            reference[key] = (
                item.features,
                np.asarray(probability, dtype=np.float64).copy(),
                float(score),
                bool(action),
            )
            continue
        previous_features, previous_probability, previous_score, previous_action = reference[key]
        if item.features != previous_features:
            raise ValueError(f"Feature drift across seeds for {key}")
        if not np.array_equal(np.asarray(probability, dtype=np.float64), previous_probability):
            raise ValueError(f"Calibrated probability drift across seeds for {key}")
        if float(score) != previous_score or bool(action) != previous_action:
            raise ValueError(f"Selector score/action drift across seeds for {key}")


def _expected_development_scopes(layout: SourceLayout) -> Dict[str, Dict[str, Any]]:
    pairs = [item.pair for item in layout.pairs]
    seeds = list(layout.seeds)
    tasks = [item.task for item in layout.tasks]
    output: Dict[str, Dict[str, Any]] = {
        "global": {
            "active_pairs": pairs,
            "active_seeds": seeds,
            "active_tasks": tasks,
            "evaluation": {"type": "global"},
        }
    }
    for seed in seeds:
        output[f"leave_one_seed_{seed}"] = {
            "active_pairs": pairs,
            "active_seeds": [value for value in seeds if value != seed],
            "active_tasks": tasks,
            "evaluation": {"type": "leave_one_seed_out", "held_out_seed": seed},
        }
    for task in tasks:
        output[f"leave_one_task_{task}"] = {
            "active_pairs": pairs,
            "active_seeds": seeds,
            "active_tasks": [value for value in tasks if value != task],
            "evaluation": {"type": "leave_one_task_out", "held_out_task": task},
        }
    for pair in pairs:
        output[f"leave_one_pair_{pair}"] = {
            "active_pairs": [value for value in pairs if value != pair],
            "active_seeds": seeds,
            "active_tasks": tasks,
            "evaluation": {"type": "leave_one_pair_out", "held_out_pair": pair},
        }
    return output


def _validate_selection_scope_contract(
    selection_lock: Mapping[str, Any],
    layout: SourceLayout,
    candidate_manifest: Mapping[str, Any],
) -> None:
    expected = _expected_development_scopes(layout)
    scopes = selection_lock.get("scopes")
    audits = selection_lock.get("loader_audit")
    if not isinstance(scopes, Mapping) or set(scopes) != set(expected):
        raise ValueError("Selection lock scope set differs from frozen 11-scope contract")
    if not isinstance(audits, Mapping) or set(audits) != set(expected):
        raise ValueError("Selection lock loader-audit scope set mismatch")
    candidate_ids = {str(item["id"]) for item in candidate_manifest["candidates"]}
    for scope_id, contract in expected.items():
        scope = scopes[scope_id]
        audit = audits[scope_id]
        if scope.get("evaluation") != contract["evaluation"]:
            raise ValueError(f"LOO evaluation contract mismatch for {scope_id}")
        for field in ("active_pairs", "active_seeds", "active_tasks"):
            if sorted(scope.get(field, [])) != sorted(contract[field]):
                raise ValueError(f"{field} mismatch for {scope_id}")
        if (
            list(audit.get("included_pairs", [])) != sorted(contract["active_pairs"])
            or list(audit.get("included_seeds", [])) != sorted(contract["active_seeds"])
            or list(audit.get("included_tasks", [])) != sorted(contract["active_tasks"])
        ):
            raise ValueError(f"Outcome-access scope audit mismatch for {scope_id}")
        if audit["receiver_outcome_rows_parsed_by_split"]["test"] != 0 or audit[
            "fused_outcome_rows_parsed_by_split"
        ]["test"] != 0:
            raise ValueError(f"Development test outcome access recorded for {scope_id}")
        if any(int(value) <= 0 for value in scope["development_counts"].values()):
            raise ValueError(f"Empty development split recorded for {scope_id}")
        candidate_id = str(scope["selected_candidate"]["candidate_id"])
        if candidate_id not in candidate_ids:
            raise ValueError(f"Unknown selected candidate {candidate_id} for {scope_id}")
        if scope["comparator"]["policy"] not in {"always_receiver", "always_fused"}:
            raise ValueError(f"Invalid comparator for {scope_id}")
        _threshold_value(scope["selected_candidate"]["threshold"])


def _replay_development_lock(
    selection_lock: Mapping[str, Any],
    protocol: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    layout: SourceLayout,
    candidate_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    replay: Dict[str, Any] = {}
    feature_order = tuple(map(str, feature_manifest["feature_order"]))
    for scope_id, contract in _expected_development_scopes(layout).items():
        observations, audit, _ = load_observations(
            protocol,
            feature_manifest,
            split_manifest,
            stage="develop",
            include_pairs=contract["active_pairs"],
            include_seeds=contract["active_seeds"],
            include_tasks=contract["active_tasks"],
            source_layout=layout,
        )
        scope = selection_lock["scopes"][scope_id]
        model = _load_scope_model(
            selection_lock,
            scope_id,
            feature_order,
            candidate_manifest,
        )
        by_split = {
            name: [item for item in observations if item.split == name]
            for name in ("fit", "calibration", "model_selection")
        }
        cal_x, _cal_y, _cal_r, _cal_f = _arrays(by_split["calibration"])
        cal_probabilities = _predict_probabilities(model, cal_x)
        threshold = _select_threshold(
            by_split["calibration"],
            _score(cal_probabilities),
            _balanced_weights(by_split["calibration"]),
            float(protocol["threshold"]["tie_tolerance"]),
        )
        frozen = scope["selected_candidate"]
        for key in (
            "threshold",
            "calibration_selector_accuracy",
            "calibration_transfer_rate",
            "threshold_candidate_count",
        ):
            if threshold[key] != frozen[key]:
                if isinstance(threshold[key], float) and math.isclose(
                    threshold[key], float(frozen[key]), rel_tol=0.0, abs_tol=1e-15
                ):
                    continue
                raise ValueError(f"Calibration replay mismatch for {scope_id}/{key}")
        comparator = _select_comparator(
            by_split["calibration"], _balanced_weights(by_split["calibration"])
        )
        if comparator != scope["comparator"]:
            raise ValueError(f"Comparator replay mismatch for {scope_id}")
        select_x, _select_y, _select_r, _select_f = _arrays(
            by_split["model_selection"]
        )
        select_probabilities = _predict_probabilities(model, select_x)
        select_actions = _score(select_probabilities) > _threshold_value(
            frozen["threshold"]
        )
        checksum = _prediction_checksum(
            by_split["model_selection"], select_probabilities, select_actions
        )
        if checksum != frozen["model_selection_prediction_checksum"]:
            raise ValueError(f"Development prediction replay mismatch for {scope_id}")
        selection_weights = _balanced_weights(by_split["model_selection"])
        accuracy = _policy_accuracy(
            by_split["model_selection"], select_actions, selection_weights
        )
        transfer = float(
            np.average(select_actions.astype(np.float64), weights=selection_weights)
        )
        if not math.isclose(
            accuracy,
            float(frozen["model_selection_selector_accuracy"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ) or not math.isclose(
            transfer,
            float(frozen["model_selection_transfer_rate"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"Development metric replay mismatch for {scope_id}")
        replay[scope_id] = {
            "test_outcomes_parsed": 0,
            "prediction_checksum": checksum,
            "loader_audit": audit,
        }
    return replay


def _selection_lock_preflight(
    *,
    expected_head: str,
    selection_artifact_commit: str,
    protocol_path: Path,
    candidate_path: Path,
    feature_path: Path,
    split_path: Path,
    split_sha_path: Path,
    design_freeze_path: Path,
    selection_lock_path: Path,
    require_outputs_absent: bool,
) -> Dict[str, Any]:
    _require_main_at_origin(expected_head)
    for tracked_path in (design_freeze_path, selection_lock_path):
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(tracked_path.relative_to(REPO_ROOT))],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ValueError(f"Frozen lock is not tracked by Git: {tracked_path}")
    protocol, candidates, feature_manifest = _load_protocol_bundle(
        protocol_path, candidate_path, feature_path
    )
    runtime = _validate_runtime(protocol)
    selection_lock = _read_json(selection_lock_path)
    if selection_lock.get("role") != "development_selection_lock":
        raise ValueError("Invalid development selection lock")
    if selection_lock.get("test_outcome_rows_parsed") != 0:
        raise ValueError("Selection lock reports test outcome access")
    development_commit = str(selection_lock["development_execution_commit"])
    design_freeze = _validate_design_freeze(
        design_freeze_path,
        protocol_path=protocol_path,
        candidate_path=candidate_path,
        feature_path=feature_path,
        split_path=split_path,
        split_sha_path=split_sha_path,
        expected_design_commit=development_commit,
    )
    _verify_hash_record(selection_lock["design_freeze"], design_freeze_path)
    _verify_hash_record(selection_lock["implementation_script"], SCRIPT_PATH)
    if selection_lock.get("implementation_commit") != design_freeze.get(
        "implementation_commit"
    ):
        raise ValueError("Selection/design implementation commits differ")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", selection_artifact_commit, expected_head],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("Selection artifact commit is not an ancestor of formal HEAD")
    for name, path in (
        ("protocol", protocol_path),
        ("candidates", candidate_path),
        ("features", feature_path),
        ("split_manifest", split_path),
        ("split_sha_file", split_sha_path),
    ):
        _verify_hash_record(selection_lock["inputs"][name], path)
    split_manifest = _load_split_manifest(split_path, split_sha_path, protocol)
    layout = _load_source_layout(protocol)
    digest = _source_artifact_digest(layout)
    if digest != selection_lock["inputs"]["source_artifact_digest"]:
        raise ValueError("Selection lock source artifact digest mismatch")
    if list(layout.source_artifacts) != selection_lock["inputs"]["source_artifacts"]:
        raise ValueError("Selection lock source artifact list mismatch")
    if runtime != selection_lock["runtime"]:
        raise ValueError("Selection lock runtime mismatch")
    _validate_selection_scope_contract(selection_lock, layout, candidates)
    feature_order = tuple(map(str, feature_manifest["feature_order"]))
    model_paths: List[Path] = []
    for scope_id, scope in selection_lock["scopes"].items():
        path = Path(str(scope["model_path"])).resolve()
        model_paths.append(path)
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(REPO_ROOT))],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ValueError(f"Frozen model is not tracked by Git: {path}")
        _load_scope_model(selection_lock, scope_id, feature_order, candidates)
    _verify_single_parent_commit_diff(
        selection_artifact_commit,
        development_commit,
        [selection_lock_path, *model_paths],
    )
    global_scope = selection_lock["scopes"]["global"]
    if selection_lock["global_selected_candidate_id"] != global_scope[
        "selected_candidate"
    ]["candidate_id"]:
        raise ValueError("Global candidate lock mismatch")
    if selection_lock["global_threshold"] != global_scope["selected_candidate"][
        "threshold"
    ]:
        raise ValueError("Global threshold lock mismatch")
    if selection_lock["global_comparator"] != global_scope["comparator"]:
        raise ValueError("Global comparator lock mismatch")
    if require_outputs_absent:
        forbidden_paths = list(FORMAL_OUTPUTS.values()) + [
            _resolve_repo_path(protocol["sealed_test"]["attempt_marker"]),
            _resolve_repo_path(protocol["sealed_test"]["completion_marker"]),
            _resolve_repo_path(protocol["sealed_test"]["per_example_output"]),
        ]
        existing = [str(path) for path in forbidden_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Sealed test has an existing attempt/output and cannot be run: "
                + ", ".join(existing)
            )
    input_audit = _validate_inputs_without_outcomes(
        layout, feature_manifest, split_manifest
    )
    if input_audit["correctness_fields_interpreted_or_indexed"] != 0:
        raise AssertionError("Preflight parsed correctness")
    development_replay = _replay_development_lock(
        selection_lock,
        protocol,
        feature_manifest,
        split_manifest,
        layout,
        candidates,
    )
    return {
        "protocol": protocol,
        "candidates": candidates,
        "features": feature_manifest,
        "split_manifest": split_manifest,
        "layout": layout,
        "design_freeze": design_freeze,
        "selection_lock": selection_lock,
        "runtime": runtime,
        "input_audit": input_audit,
        "development_replay": development_replay,
        "selection_artifact_commit": selection_artifact_commit,
        "selection_lock_sha256": _sha256(selection_lock_path),
    }


def validate_test(
    *,
    expected_selection_commit: str,
    protocol_path: Path,
    candidate_path: Path,
    feature_path: Path,
    split_path: Path,
    split_sha_path: Path,
    design_freeze_path: Path,
    selection_lock_path: Path,
) -> Dict[str, Any]:
    preflight = _selection_lock_preflight(
        expected_head=expected_selection_commit,
        selection_artifact_commit=expected_selection_commit,
        protocol_path=protocol_path,
        candidate_path=candidate_path,
        feature_path=feature_path,
        split_path=split_path,
        split_sha_path=split_sha_path,
        design_freeze_path=design_freeze_path,
        selection_lock_path=selection_lock_path,
        require_outputs_absent=True,
    )
    tag = str(preflight["protocol"]["sealed_test"]["remote_consumption_tag"])
    existing_tag = _remote_tag_target(tag)
    if existing_tag is not None:
        raise FileExistsError(f"Historical remote test-consumption tag exists: {tag}")
    return {
        "status": "READY_FOR_SINGLE_SEALED_TEST",
        "head": _git_head(),
        "selection_lock_sha256": preflight["selection_lock_sha256"],
        "input_audit": preflight["input_audit"],
        "test_outcomes_parsed": 0,
    }


def consume_test_attempt(
    *,
    expected_selection_commit: str,
    protocol_path: Path,
    candidate_path: Path,
    feature_path: Path,
    split_path: Path,
    split_sha_path: Path,
    design_freeze_path: Path,
    selection_lock_path: Path,
) -> Dict[str, Any]:
    """Arm the single test attempt without reading any test outcome."""

    preflight = _selection_lock_preflight(
        expected_head=expected_selection_commit,
        selection_artifact_commit=expected_selection_commit,
        protocol_path=protocol_path,
        candidate_path=candidate_path,
        feature_path=feature_path,
        split_path=split_path,
        split_sha_path=split_sha_path,
        design_freeze_path=design_freeze_path,
        selection_lock_path=selection_lock_path,
        require_outputs_absent=True,
    )
    tag = str(preflight["protocol"]["sealed_test"]["remote_consumption_tag"])
    existing_tag = _remote_tag_target(tag)
    if existing_tag is not None:
        raise FileExistsError(f"Historical remote test-consumption tag exists: {tag}")
    selection_lock = preflight["selection_lock"]
    models = {
        scope_id: {
            "path": scope["model_path"],
            "sha256": scope["model_sha256"],
            "candidate_id": scope["selected_candidate"]["candidate_id"],
        }
        for scope_id, scope in sorted(selection_lock["scopes"].items())
    }
    receipt = {
        "schema_version": 1,
        "phase": "2A-1",
        "role": "durable_single_test_attempt_receipt",
        "status": "ARMED_NOT_EVALUATED",
        "armed_at_utc": _utc_timestamp(),
        "selection_artifact_commit": expected_selection_commit,
        "selection_lock": _manifest_hash_record(selection_lock_path),
        "design_freeze": _manifest_hash_record(design_freeze_path),
        "implementation_commit": selection_lock["implementation_commit"],
        "models": models,
        "source_artifact_digest": selection_lock["inputs"]["source_artifact_digest"],
        "protocol": _manifest_hash_record(protocol_path),
        "remote_consumption_tag": preflight["protocol"]["sealed_test"][
            "remote_consumption_tag"
        ],
        "input_only_preflight": preflight["input_audit"],
        "development_replay_checksums": {
            scope_id: values["prediction_checksum"]
            for scope_id, values in preflight["development_replay"].items()
        },
        "test_outcomes_parsed": 0,
        "next_step": "commit and push this receipt as the only child of the selection-artifact commit, then run evaluate-test once",
    }
    _write_json_once(FORMAL_OUTPUTS["test_attempt"], receipt)
    return {
        "status": "TEST_ATTEMPT_ARMED_OUTCOMES_UNREAD",
        "selection_artifact_commit": expected_selection_commit,
        "attempt_receipt": _manifest_hash_record(FORMAL_OUTPUTS["test_attempt"]),
        "test_outcomes_parsed": 0,
    }


def _remote_tag_target(tag: str) -> str | None:
    result = subprocess.run(
        ["git", "ls-remote", "--refs", "--tags", "origin", f"refs/tags/{tag}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Unable to query remote consumption tag {tag}: {result.stderr}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    if len(lines) != 1:
        raise ValueError(f"Unexpected remote tag response for {tag}: {lines}")
    return lines[0].split()[0]


def _claim_remote_test_consumption(tag: str, expected_commit: str) -> Dict[str, Any]:
    existing = _remote_tag_target(tag)
    if existing is not None:
        raise FileExistsError(
            f"Remote test-consumption tag {tag} already exists at {existing}; "
            "the sealed test is terminal and cannot be rerun"
        )
    local_ref = f"refs/tags/{tag}"
    local = subprocess.run(
        ["git", "rev-parse", "--verify", local_ref],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if local.returncode == 0:
        if local.stdout.strip() != expected_commit:
            raise ValueError(f"Local consumption tag {tag} points to wrong commit")
    else:
        subprocess.run(
            ["git", "tag", tag, expected_commit],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    pushed = subprocess.run(
        ["git", "push", "origin", local_ref],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if pushed.returncode != 0:
        raise RuntimeError(
            f"Failed to create remote consumption ledger; outcomes remain unread: {pushed.stderr}"
        )
    target = _remote_tag_target(tag)
    if target != expected_commit:
        raise ValueError(f"Remote consumption tag verification failed: {target}")
    return {
        "tag": tag,
        "target_commit": target,
        "push_stdout": pushed.stdout.strip(),
        "claimed_at_utc": _utc_timestamp(),
    }


def _validate_committed_attempt(
    *,
    expected_attempt_commit: str,
    protocol_path: Path,
    candidate_path: Path,
    feature_path: Path,
    split_path: Path,
    split_sha_path: Path,
    design_freeze_path: Path,
    selection_lock_path: Path,
) -> Dict[str, Any]:
    _require_main_at_origin(expected_attempt_commit)
    receipt_path = FORMAL_OUTPUTS["test_attempt"]
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(receipt_path.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        raise ValueError("Durable test attempt receipt is not committed")
    receipt = _read_json(receipt_path)
    if (
        receipt.get("role") != "durable_single_test_attempt_receipt"
        or receipt.get("status") != "ARMED_NOT_EVALUATED"
        or receipt.get("test_outcomes_parsed") != 0
    ):
        raise ValueError("Invalid durable test attempt receipt")
    selection_commit = str(receipt["selection_artifact_commit"])
    _verify_single_parent_commit_diff(
        expected_attempt_commit, selection_commit, [receipt_path]
    )
    preflight = _selection_lock_preflight(
        expected_head=expected_attempt_commit,
        selection_artifact_commit=selection_commit,
        protocol_path=protocol_path,
        candidate_path=candidate_path,
        feature_path=feature_path,
        split_path=split_path,
        split_sha_path=split_sha_path,
        design_freeze_path=design_freeze_path,
        selection_lock_path=selection_lock_path,
        require_outputs_absent=False,
    )
    _verify_hash_record(receipt["selection_lock"], selection_lock_path)
    _verify_hash_record(receipt["design_freeze"], design_freeze_path)
    _verify_hash_record(receipt["protocol"], protocol_path)
    if receipt["source_artifact_digest"] != preflight["selection_lock"]["inputs"][
        "source_artifact_digest"
    ]:
        raise ValueError("Attempt receipt source digest mismatch")
    expected_models = {
        scope_id: {
            "path": scope["model_path"],
            "sha256": scope["model_sha256"],
            "candidate_id": scope["selected_candidate"]["candidate_id"],
        }
        for scope_id, scope in sorted(preflight["selection_lock"]["scopes"].items())
    }
    if receipt["models"] != expected_models:
        raise ValueError("Attempt receipt model lock mismatch")
    forbidden_paths = [
        path
        for name, path in FORMAL_OUTPUTS.items()
        if name not in {"test_attempt"}
    ] + [
        _resolve_repo_path(preflight["protocol"]["sealed_test"]["attempt_marker"]),
        _resolve_repo_path(preflight["protocol"]["sealed_test"]["completion_marker"]),
        _resolve_repo_path(preflight["protocol"]["sealed_test"]["per_example_output"]),
    ]
    existing = [str(path) for path in forbidden_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Sealed test already has local/results/completion artifacts: "
            + ", ".join(existing)
        )
    return {**preflight, "attempt_receipt": receipt, "attempt_commit": expected_attempt_commit}


def _validate_test_authorization(
    authorization: Mapping[str, Any] | None,
) -> None:
    if not isinstance(authorization, Mapping):
        raise PermissionError("Test outcomes require a committed, remotely consumed attempt")
    expected_commit = str(authorization.get("attempt_commit", ""))
    receipt_path = Path(str(authorization.get("receipt_path", ""))).resolve()
    expected_sha = str(authorization.get("receipt_sha256", ""))
    tag = str(authorization.get("remote_consumption_tag", ""))
    if receipt_path != FORMAL_OUTPUTS["test_attempt"].resolve():
        raise PermissionError("Test authorization receipt path mismatch")
    _require_main_at_origin(expected_commit)
    if _sha256(receipt_path) != expected_sha:
        raise PermissionError("Test authorization receipt SHA mismatch")
    receipt = _read_json(receipt_path)
    _verify_single_parent_commit_diff(
        expected_commit,
        str(receipt["selection_artifact_commit"]),
        [receipt_path],
    )
    if tag != str(receipt["remote_consumption_tag"]):
        raise PermissionError("Remote consumption tag mismatch")
    if _remote_tag_target(tag) != expected_commit:
        raise PermissionError("Remote consumption ledger is absent or points elsewhere")


def _prediction_rows(
    scope: str,
    observations: Sequence[Observation],
    probabilities: np.ndarray,
    scores: np.ndarray,
    actions: np.ndarray,
    comparator_policy: str,
    candidate_id: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item, probability, score, action in zip(
        observations, probabilities, scores, actions
    ):
        selector_correct = item.fused_correct if action else item.receiver_correct
        comparator_correct = (
            item.receiver_correct
            if comparator_policy == "always_receiver"
            else item.fused_correct
        )
        rows.append(
            {
                "scope": scope,
                "candidate_id": candidate_id,
                "pair": item.pair,
                "seed": item.seed,
                "task": item.task,
                "subject": item.subject,
                "question_id": item.question_id,
                "content_hash": item.content_hash,
                "split": item.split,
                "cot_input_length": item.features[0],
                "candidate_count": item.features[1],
                "candidate_count_max": item.features[2],
                "one_to_many_rate": item.features[3],
                "boundary_mismatch": item.features[4],
                "receiver_correct": item.receiver_correct,
                "fused_correct": item.fused_correct,
                "utility": item.utility,
                "probability_harm": float(probability[0]),
                "probability_neutral": float(probability[1]),
                "probability_benefit": float(probability[2]),
                "score": float(score),
                "transfer": int(action),
                "selector_correct": selector_correct,
                "comparator_policy": comparator_policy,
                "comparator_correct": comparator_correct,
            }
        )
    return rows


def _find_row(
    rows: Sequence[Mapping[str, Any]], **criteria: Any
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one aggregate row for {criteria}, found {len(matches)}")
    return matches[0]


def _go_decision(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> Dict[str, Any]:
    primary = _find_row(
        rows,
        scope="global_test",
        policy="selected",
        aggregation_level="pair_balanced",
        weighting="task_macro",
        pair="__all__",
        seed="all",
        task="__all__",
    )
    hetero_pairs = list(protocol["sensitivity"]["heterogeneous_pairs"])
    pair_deltas: Dict[str, float] = {}
    for pair in hetero_pairs:
        row = _find_row(
            rows,
            scope="global_test",
            policy="selected",
            aggregation_level="pair_across_seeds",
            weighting="task_macro",
            pair=pair,
            seed="all",
            task="__all__",
        )
        pair_deltas[pair] = float(row["selector_minus_comparator"])
    config = protocol["go_gate"]
    primary_delta = float(primary["selector_minus_comparator"])
    ci_low = float(primary["selector_minus_comparator_ci_low"])
    recovery = primary.get("oracle_headroom_recovery")
    harmful_reduction = primary.get("harmful_reduction")
    beneficial_retention = primary.get("beneficial_retention")
    positive_count = sum(value > 0.0 for value in pair_deltas.values())
    gates = {
        "primary_delta_at_least_0p5pp": primary_delta
        >= float(config["minimum_primary_delta"]),
        "primary_ci_lower_above_zero": ci_low
        > float(config["primary_ci_low_strictly_greater_than"]),
        "heterogeneous_pair_sign_rule": positive_count
        >= int(config["minimum_positive_heterogeneous_pairs"])
        and min(pair_deltas.values())
        >= float(config["remaining_heterogeneous_pair_minimum_delta"]),
        "oracle_headroom_recovery_at_least_15pct": recovery is not None
        and math.isfinite(float(recovery))
        and float(recovery) >= float(config["minimum_true_headroom_recovery"]),
        "harmful_reduction_at_least_25pct": harmful_reduction is not None
        and math.isfinite(float(harmful_reduction))
        and float(harmful_reduction) >= float(config["minimum_harmful_reduction"]),
        "beneficial_retention_at_least_80pct": beneficial_retention is not None
        and math.isfinite(float(beneficial_retention))
        and float(beneficial_retention) >= float(config["minimum_beneficial_retention"]),
    }
    return {
        "decision": "GO_INTERNAL_PREDICTABILITY_ONLY" if all(gates.values()) else "NO_GO",
        "all_conjunctive": True,
        "gates": gates,
        "values": {
            "primary_delta": primary_delta,
            "primary_ci_low": ci_low,
            "primary_ci_high": float(primary["selector_minus_comparator_ci_high"]),
            "heterogeneous_pair_deltas": pair_deltas,
            "positive_heterogeneous_pair_count": positive_count,
            "oracle_headroom_recovery": recovery,
            "harmful_reduction": harmful_reduction,
            "beneficial_retention": beneficial_retention,
        },
    }


def _format_percent(value: Any, digits: int = 2) -> str:
    if value is None:
        return "undefined"
    return f"{100.0 * float(value):.{digits}f}%"


def _format_pp(value: Any, digits: int = 2) -> str:
    if value is None:
        return "undefined"
    return f"{100.0 * float(value):+.{digits}f} pp"


def _write_text_once(path: Path, text_value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text_value.rstrip() + "\n")


def _render_report(
    rows: Sequence[Mapping[str, Any]],
    go: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
    random_audit: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    primary = _find_row(
        rows,
        scope="global_test",
        policy="selected",
        aggregation_level="pair_balanced",
        weighting="task_macro",
        pair="__all__",
        seed="all",
        task="__all__",
    )
    secondary = _find_row(
        rows,
        scope="global_test",
        policy="selected",
        aggregation_level="pair_balanced",
        weighting="sample_weighted",
        pair="__all__",
        seed="all",
        task="__all__",
    )
    random_primary = _find_row(
        rows,
        scope="global_test",
        policy="same_rate_random",
        aggregation_level="pair_balanced",
        weighting="task_macro",
        pair="__all__",
        seed="all",
        task="__all__",
    )
    strict = _find_row(
        rows,
        scope="global_test",
        policy="selected",
        aggregation_level="pair_balanced",
        weighting="task_macro",
        pair="__strict_cross_family__",
        seed="all",
        task="__all__",
    )
    same_tokenizer = _find_row(
        rows,
        scope="global_test",
        policy="selected",
        aggregation_level="pair_balanced",
        weighting="task_macro",
        pair="__same_tokenizer__",
        seed="all",
        task="__all__",
    )
    scope = selection_lock["scopes"]["global"]
    gate_lines = "\n".join(
        f"- {'PASS' if passed else 'FAIL'} — `{name}`"
        for name, passed in go["gates"].items()
    )
    pair_lines = []
    for pair in ["tinyllama", "qwen25_0p5b", "llama32_1b", "qwen3_1p7b"]:
        row = _find_row(
            rows,
            scope="global_test",
            policy="selected",
            aggregation_level="pair_across_seeds",
            weighting="task_macro",
            pair=pair,
            seed="all",
            task="__all__",
        )
        pair_lines.append(
            f"| {pair} | {_format_percent(row['selector_accuracy'])} | "
            f"{_format_pp(row['selector_minus_comparator'])} | "
            f"[{_format_pp(row['selector_minus_comparator_ci_low'])}, "
            f"{_format_pp(row['selector_minus_comparator_ci_high'])}] | "
            f"{_format_percent(row['transfer_rate'])} |"
        )
    loo_lines = []
    for row in rows:
        if str(row.get("scope", "")).startswith("leave_one_") and row.get(
            "weighting"
        ) == "task_macro":
            loo_lines.append(
                f"| {row['scope']} | {row['candidate_id']} | "
                f"{_format_pp(row['selector_minus_comparator'])} | "
                f"[{_format_pp(row['selector_minus_comparator_ci_low'])}, "
                f"{_format_pp(row['selector_minus_comparator_ci_high'])}] |"
            )
    interpretation = (
        "The primary A-tier selector passed only an internal predictability gate. "
        "It still requires external validation on a new benchmark before any general "
        "deployable no-transfer claim."
        if go["decision"] == "GO_INTERNAL_PREDICTABILITY_ONLY"
        else "The conjunctive gate failed. Per preregistration, Phase 2A-1 is a NO-GO: "
        "do not train a more complex or neural selector. Stop and wait for explicit "
        "authorization before any new pre-transfer cache-geometry instrumentation."
    )
    return f"""# Phase 2A-1 Existing A-tier Selector Predictability Kill-Test

## Decision

**{go['decision']}**

{interpretation}

## Frozen design and execution

- Source baseline commit: `9fa1f0ac3bedefd282961a853278ab88fb376fa2`
- Implementation commit: `{selection_lock['implementation_commit']}`
- Development execution commit: `{selection_lock['development_execution_commit']}`
- Sealed test commit: `{provenance['test_commit']}`
- Selected candidate: `{scope['selected_candidate']['candidate_id']}`
- Candidate family: `{scope['selected_candidate']['family']}`
- Features: `{', '.join(scope['selected_candidate']['features'])}`
- Frozen threshold: `{json.dumps(scope['selected_candidate']['threshold'], sort_keys=True)}`
- Calibration-selected comparator: `{scope['comparator']['policy']}`
- Test execution count: one; existing attempt is terminal and no rerun path exists.

## Conjunctive GO gate

{gate_lines}

Primary pair-balanced task-macro delta is {_format_pp(primary['selector_minus_comparator'])},
95% hierarchical paired-bootstrap CI
[{_format_pp(primary['selector_minus_comparator_ci_low'])},
{_format_pp(primary['selector_minus_comparator_ci_high'])}].

## Primary and secondary results

| Estimand | Selector | Comparator | Delta | Transfer | Harm reduction | Benefit retention | Oracle recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pair-balanced task-macro | {_format_percent(primary['selector_accuracy'])} | {_format_percent(primary['comparator_accuracy'])} | {_format_pp(primary['selector_minus_comparator'])} | {_format_percent(primary['transfer_rate'])} | {_format_percent(primary['harmful_reduction'])} | {_format_percent(primary['beneficial_retention'])} | {_format_percent(primary['oracle_headroom_recovery'])} |
| Pair-balanced sample-weighted | {_format_percent(secondary['selector_accuracy'])} | {_format_percent(secondary['comparator_accuracy'])} | {_format_pp(secondary['selector_minus_comparator'])} | {_format_percent(secondary['transfer_rate'])} | {_format_percent(secondary['harmful_reduction'])} | {_format_percent(secondary['beneficial_retention'])} | {_format_percent(secondary['oracle_headroom_recovery'])} |
| Same-rate random (task-macro) | {_format_percent(random_primary['selector_accuracy'])} | {_format_percent(random_primary['comparator_accuracy'])} | {_format_pp(random_primary['selector_minus_comparator'])} | {_format_percent(random_primary['transfer_rate'])} | {_format_percent(random_primary['harmful_reduction'])} | {_format_percent(random_primary['beneficial_retention'])} | {_format_percent(random_primary['oracle_headroom_recovery'])} |

## Pair results

| Pair | Selector accuracy | Delta vs frozen comparator | 95% CI | Transfer rate |
|---|---:|---:|---:|---:|
{chr(10).join(pair_lines)}

Strict cross-family (TinyLlama + Llama3.2) delta:
{_format_pp(strict['selector_minus_comparator'])}. Same-tokenizer Qwen3 control delta:
{_format_pp(same_tokenizer['selector_minus_comparator'])}.

## Predictive diagnostics

- Benefit AUPRC (balanced pooled): {primary['benefit_auprc_pooled']}
- Harm AUPRC (balanced pooled): {primary['harm_auprc_pooled']}
- Multiclass Brier: {primary['multiclass_brier']}
- 15-bin class-macro ECE: {primary['ece_multiclass_macro']}
- Random baseline is one deterministic, outcome-blind realization. Per-stratum achieved-rate gaps are frozen in the result JSON; max absolute gap is {max(float(value['absolute_rate_gap']) for value in random_audit.values())}.

## Leave-one-out sensitivity

| Fold | Selected candidate | Delta | 95% CI |
|---|---|---:|---:|
{chr(10).join(loo_lines)}

Leave-one-out rows are diagnostics only and do not feed back into the global candidate,
threshold, comparator, or GO decision.

## Interpretation boundary

- Primary features are exactly the five preregistered A-tier fields. No raw text,
  task/pair/seed metadata, IDs, labels, correctness, entropy/confidence duplicate,
  or constant fallback feature entered a model.
- Candidate selection occurred only on model-selection data after fit-only training and
  calibration-only probability/threshold selection. Test evaluated one frozen global
  candidate exactly once.
- A positive result is internal predictability evidence only. A failure terminates the
  complex-selector path under the preregistration.

## Reproducibility

The frozen split, candidate, feature, protocol, code, selection lock, model SHA files,
aggregate CSV/JSON, and result manifest live in the repository. Full per-example
predictions remain under `local/` and are intentionally not committed.
"""


def _render_summary_zh(
    rows: Sequence[Mapping[str, Any]],
    go: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
) -> str:
    primary = _find_row(
        rows,
        scope="global_test",
        policy="selected",
        aggregation_level="pair_balanced",
        weighting="task_macro",
        pair="__all__",
        seed="all",
        task="__all__",
    )
    scope = selection_lock["scopes"]["global"]
    failures = [name for name, passed in go["gates"].items() if not passed]
    if go["decision"] == "GO_INTERNAL_PREDICTABILITY_ONLY":
        conclusion = (
            "A-tier 特征通过了内部可预测性门槛，但这还不是可泛化 selector 的证据；"
            "必须在全新 benchmark 上做外部验证。"
        )
    else:
        conclusion = (
            "六项联合门槛至少一项失败，因此 Phase 2A-1 为 NO-GO。按预注册停止复杂/"
            "神经 selector，不自动进入下一阶段；如要继续，只能等待是否授权一次新的"
            " pre-transfer cache-geometry instrumentation。"
        )
    return f"""# Phase 2A-1 Selector Kill-Test 中文摘要

## 结论

**{go['decision']}**

{conclusion}

冻结后的唯一候选是 `{scope['selected_candidate']['candidate_id']}`，比较对象是在
calibration split 选定的 `{scope['comparator']['policy']}`。主指标
（pair-balanced task-macro）差值为 {_format_pp(primary['selector_minus_comparator'])}，
95% 分层 paired bootstrap CI 为
[{_format_pp(primary['selector_minus_comparator_ci_low'])},
{_format_pp(primary['selector_minus_comparator_ci_high'])}]。

- transfer rate：{_format_percent(primary['transfer_rate'])}
- harmful transfer reduction：{_format_percent(primary['harmful_reduction'])}
- beneficial transfer retention：{_format_percent(primary['beneficial_retention'])}
- oracle-over-best-fixed headroom recovery：{_format_percent(primary['oracle_headroom_recovery'])}
- 未通过门槛：{', '.join(failures) if failures else '无'}

本轮严格只用了五项 A-tier 特征，未使用题目文本、task/pair/seed、ID、标签、
正确性字段、entropy/confidence 冗余字段或恒零 fallback。Test 在候选、阈值、
comparator、代码和模型 SHA 全部冻结后只执行了一次。逐例文件保留在 `local/`，
仓库只提交小型聚合结果与审计 manifest。
"""


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _loo_spec(
    scope_id: str,
    evaluation: Mapping[str, Any],
    layout: SourceLayout,
    weighting: str,
) -> Dict[str, Any]:
    pairs = [item.pair for item in layout.pairs]
    seeds = list(layout.seeds)
    tasks = [item.task for item in layout.tasks]
    fold_type = evaluation["type"]
    if fold_type == "leave_one_seed_out":
        held = int(evaluation["held_out_seed"])
        return {
            "aggregation_level": "leave_one_seed_out",
            "pair": "__all__",
            "seed": held,
            "task": "__all__",
            "pairs": pairs,
            "seeds": [held],
            "tasks": tasks,
            "weighting": weighting,
            "resample_pairs": True,
            "resample_seeds": False,
        }
    if fold_type == "leave_one_pair_out":
        held = str(evaluation["held_out_pair"])
        return {
            "aggregation_level": "leave_one_pair_out",
            "pair": held,
            "seed": "all",
            "task": "__all__",
            "pairs": [held],
            "seeds": seeds,
            "tasks": tasks,
            "weighting": weighting,
            "resample_pairs": False,
            "resample_seeds": True,
        }
    if fold_type == "leave_one_task_out":
        held = str(evaluation["held_out_task"])
        return {
            "aggregation_level": "leave_one_task_out",
            "pair": "__all__",
            "seed": "all",
            "task": held,
            "pairs": pairs,
            "seeds": seeds,
            "tasks": [held],
            "weighting": weighting,
            "resample_pairs": True,
            "resample_seeds": True,
        }
    raise ValueError(f"Unknown leave-one-out fold type for {scope_id}: {fold_type}")


def _loo_observations(
    all_observations: Sequence[Observation], evaluation: Mapping[str, Any]
) -> List[Observation]:
    fold_type = evaluation["type"]
    if fold_type == "leave_one_seed_out":
        held = int(evaluation["held_out_seed"])
        return [
            item
            for item in all_observations
            if item.seed == held and item.split == "test"
        ]
    if fold_type == "leave_one_pair_out":
        held = str(evaluation["held_out_pair"])
        return [
            item
            for item in all_observations
            if item.pair == held and item.split == "test"
        ]
    if fold_type == "leave_one_task_out":
        held = str(evaluation["held_out_task"])
        return [item for item in all_observations if item.task == held]
    raise ValueError(f"Unknown LOO evaluation {evaluation}")


def evaluate_test(
    *,
    expected_attempt_commit: str,
    protocol_path: Path,
    candidate_path: Path,
    feature_path: Path,
    split_path: Path,
    split_sha_path: Path,
    design_freeze_path: Path,
    selection_lock_path: Path,
) -> Dict[str, Any]:
    preflight = _validate_committed_attempt(
        expected_attempt_commit=expected_attempt_commit,
        protocol_path=protocol_path,
        candidate_path=candidate_path,
        feature_path=feature_path,
        split_path=split_path,
        split_sha_path=split_sha_path,
        design_freeze_path=design_freeze_path,
        selection_lock_path=selection_lock_path,
    )
    protocol = preflight["protocol"]
    feature_manifest = preflight["features"]
    split_manifest = preflight["split_manifest"]
    layout: SourceLayout = preflight["layout"]
    selection_lock = preflight["selection_lock"]
    start = _utc_timestamp()
    receipt = preflight["attempt_receipt"]
    # Close the preflight/claim TOCTOU window as far as a shared filesystem permits.
    _require_main_at_origin(expected_attempt_commit)
    _verify_hash_record(receipt["selection_lock"], selection_lock_path)
    _verify_hash_record(receipt["design_freeze"], design_freeze_path)
    _verify_hash_record(receipt["protocol"], protocol_path)
    for scope_id, frozen_model in receipt["models"].items():
        model_path = Path(str(frozen_model["path"])).resolve()
        if _sha256(model_path) != str(frozen_model["sha256"]):
            raise ValueError(f"Last-moment model SHA mismatch for {scope_id}")
    preclaim_layout = _load_source_layout(protocol)
    if _source_artifact_digest(preclaim_layout) != receipt["source_artifact_digest"]:
        raise ValueError("Last-moment source artifact digest mismatch")
    remote_claim = _claim_remote_test_consumption(
        str(receipt["remote_consumption_tag"]), expected_attempt_commit
    )
    local_attempt_path = _resolve_repo_path(protocol["sealed_test"]["attempt_marker"])
    local_attempt = {
        "schema_version": 1,
        "phase": "2A-1",
        "role": "local_evaluation_started_marker",
        "started_at_utc": start,
        "attempt_commit": expected_attempt_commit,
        "attempt_receipt": _manifest_hash_record(FORMAL_OUTPUTS["test_attempt"]),
        "remote_consumption": remote_claim,
    }
    _write_json_once(local_attempt_path, local_attempt)
    authorization = {
        "attempt_commit": expected_attempt_commit,
        "receipt_path": str(FORMAL_OUTPUTS["test_attempt"].resolve()),
        "receipt_sha256": _sha256(FORMAL_OUTPUTS["test_attempt"]),
        "remote_consumption_tag": receipt["remote_consumption_tag"],
    }

    all_observations, outcome_audit, _layout = load_observations(
        protocol,
        feature_manifest,
        split_manifest,
        stage="test",
        test_authorization=authorization,
        source_layout=layout,
    )
    _require_main_at_origin(expected_attempt_commit)
    postread_layout = _load_source_layout(protocol)
    if _source_artifact_digest(postread_layout) != selection_lock["inputs"][
        "source_artifact_digest"
    ]:
        raise ValueError("Source artifacts changed during sealed outcome read")
    global_observations = [item for item in all_observations if item.split == "test"]
    feature_order = tuple(map(str, feature_manifest["feature_order"]))
    global_scope = selection_lock["scopes"]["global"]
    global_model = _load_scope_model(
        selection_lock, "global", feature_order, preflight["candidates"]
    )
    global_probabilities, global_scores, global_actions = _predict_actions(
        global_observations,
        global_model,
        global_scope["selected_candidate"]["threshold"],
    )
    _assert_seed_invariant_predictions(
        global_observations,
        global_probabilities,
        global_scores,
        global_actions,
    )
    global_comparator = str(global_scope["comparator"]["policy"])
    random_actions, random_audit = _same_rate_random_actions(
        global_observations, global_actions, protocol
    )
    bootstrap = protocol["bootstrap"]
    pair_order = [item.pair for item in layout.pairs]
    seed_order = list(layout.seeds)
    task_order = [item.task for item in layout.tasks]
    global_tensors = _build_bootstrap_tensors(
        global_observations,
        global_actions,
        random_actions,
        global_comparator,
        pair_order=pair_order,
        seed_order=seed_order,
        task_order=task_order,
        samples=int(bootstrap["samples"]),
        base_seed=int(bootstrap["seed"]),
        batch_size=int(bootstrap["batch_size"]),
        context="global_test",
    )
    aggregate_rows = _evaluate_specs(
        global_observations,
        global_probabilities,
        global_actions,
        random_actions,
        global_comparator,
        global_tensors,
        _scope_specs(layout),
        protocol,
        selected_candidate_id=str(global_scope["selected_candidate"]["candidate_id"]),
        threshold_record=global_scope["selected_candidate"]["threshold"],
        include_random=True,
        scope_prefix="global_test",
    )
    per_example_rows = _prediction_rows(
        "global_test",
        global_observations,
        global_probabilities,
        global_scores,
        global_actions,
        global_comparator,
        str(global_scope["selected_candidate"]["candidate_id"]),
    )

    for scope_id, scope in selection_lock["scopes"].items():
        if scope_id == "global":
            continue
        evaluation = scope["evaluation"]
        fold_observations = _loo_observations(all_observations, evaluation)
        fold_model = _load_scope_model(
            selection_lock, scope_id, feature_order, preflight["candidates"]
        )
        probabilities, scores, actions = _predict_actions(
            fold_observations,
            fold_model,
            scope["selected_candidate"]["threshold"],
        )
        if len({item.seed for item in fold_observations}) > 1:
            _assert_seed_invariant_predictions(
                fold_observations, probabilities, scores, actions
            )
        comparator = str(scope["comparator"]["policy"])
        zero_random = np.zeros(len(fold_observations), dtype=bool)
        fold_pairs = sorted(
            {item.pair for item in fold_observations}, key=pair_order.index
        )
        fold_seeds = sorted({item.seed for item in fold_observations})
        fold_tasks = sorted(
            {item.task for item in fold_observations}, key=task_order.index
        )
        fold_tensors = _build_bootstrap_tensors(
            fold_observations,
            actions,
            zero_random,
            comparator,
            pair_order=fold_pairs,
            seed_order=fold_seeds,
            task_order=fold_tasks,
            samples=int(bootstrap["samples"]),
            base_seed=int(bootstrap["seed"]),
            batch_size=int(bootstrap["batch_size"]),
            context=scope_id,
        )
        specs = [
            _loo_spec(scope_id, evaluation, layout, weighting)
            for weighting in ("task_macro", "sample_weighted")
        ]
        # For a held-out task, task-macro and sample-weighted are identical; retain
        # both labels for the preregistered reporting table.
        aggregate_rows.extend(
            _evaluate_specs(
                fold_observations,
                probabilities,
                actions,
                zero_random,
                comparator,
                fold_tensors,
                specs,
                protocol,
                selected_candidate_id=str(scope["selected_candidate"]["candidate_id"]),
                threshold_record=scope["selected_candidate"]["threshold"],
                include_random=False,
                scope_prefix=scope_id,
            )
        )
        per_example_rows.extend(
            _prediction_rows(
                scope_id,
                fold_observations,
                probabilities,
                scores,
                actions,
                comparator,
                str(scope["selected_candidate"]["candidate_id"]),
            )
        )

    go = _go_decision(aggregate_rows, protocol)
    provenance = {
        "test_commit": expected_attempt_commit,
        "selection_artifact_commit": receipt["selection_artifact_commit"],
        "remote_consumption": remote_claim,
        "selection_lock_sha256": preflight["selection_lock_sha256"],
        "attempt_receipt_sha256": _sha256(FORMAL_OUTPUTS["test_attempt"]),
        "outcome_access_audit": outcome_audit,
    }
    aggregate_payload = {
        "schema_version": 1,
        "phase": "2A-1",
        "status": "SEALED_TEST_COMPLETE",
        "decision": go,
        "provenance": provenance,
        "selected_candidate": global_scope["selected_candidate"],
        "global_comparator": global_scope["comparator"],
        "random_baseline_audit": random_audit,
        "aggregate_rows": aggregate_rows,
    }
    report = _render_report(
        aggregate_rows, go, selection_lock, random_audit, provenance
    )
    summary_zh = _render_summary_zh(aggregate_rows, go, selection_lock)
    per_example_path = _resolve_repo_path(
        protocol["sealed_test"]["per_example_output"]
    )
    _write_csv_once(per_example_path, per_example_rows)
    _write_csv_once(FORMAL_OUTPUTS["aggregate_csv"], aggregate_rows)
    _write_json_once(FORMAL_OUTPUTS["aggregate_json"], aggregate_payload)
    _write_text_once(FORMAL_OUTPUTS["report"], report)
    _write_text_once(FORMAL_OUTPUTS["summary_zh"], summary_zh)
    result_manifest = {
        "schema_version": 1,
        "phase": "2A-1",
        "role": "sealed_test_result_manifest",
        "decision": go["decision"],
        "test_commit": expected_attempt_commit,
        "selection_lock": _manifest_hash_record(selection_lock_path),
        "attempt_receipt": _manifest_hash_record(FORMAL_OUTPUTS["test_attempt"]),
        "outputs": {
            "aggregate_csv": _manifest_hash_record(FORMAL_OUTPUTS["aggregate_csv"]),
            "aggregate_json": _manifest_hash_record(FORMAL_OUTPUTS["aggregate_json"]),
            "report": _manifest_hash_record(FORMAL_OUTPUTS["report"]),
            "summary_zh": _manifest_hash_record(FORMAL_OUTPUTS["summary_zh"]),
            "per_example_local": _manifest_hash_record(per_example_path),
        },
        "outcome_access_audit": outcome_audit,
        "go_gate": go,
    }
    _write_json_once(FORMAL_OUTPUTS["result_manifest"], result_manifest)
    completion = {
        "schema_version": 1,
        "phase": "2A-1",
        "role": "durable_single_test_completion_receipt",
        "status": "COMPLETE",
        "started_at_utc": start,
        "completed_at_utc": _utc_timestamp(),
        "decision": go["decision"],
        "test_commit": expected_attempt_commit,
        "result_manifest": _manifest_hash_record(FORMAL_OUTPUTS["result_manifest"]),
    }
    _write_json_once(FORMAL_OUTPUTS["test_complete"], completion)
    local_completion_path = _resolve_repo_path(
        protocol["sealed_test"]["completion_marker"]
    )
    _write_json_once(local_completion_path, completion)
    return {
        "status": "SEALED_TEST_COMPLETE",
        "decision": go["decision"],
        "primary": go["values"],
        "result_manifest": str(FORMAL_OUTPUTS["result_manifest"]),
    }


def _default_paths() -> Dict[str, Path]:
    base = REPO_ROOT / "recipe/eval_recipe/phase2a_1"
    return {
        "protocol": base / "protocol_manifest.json",
        "candidates": base / "candidate_manifest.json",
        "features": base / "feature_whitelist.json",
        "split": base / "content_group_split_manifest.json",
        "split_sha": base / "content_group_split_manifest.sha256",
        "design_freeze": base / "code_and_design_freeze.json",
        "selection_lock": base / "selection_lock.json",
        "model_dir": base / "locked_models",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 2A-1 CPU-only frozen selector kill-test"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "prepare-split",
        help="Create the outcome-free content-group split manifest exactly once",
    )
    freeze_parser = subparsers.add_parser(
        "freeze-design",
        help="Freeze committed code/design hashes before selector fitting",
    )
    freeze_parser.add_argument("--expected-implementation-commit", required=True)
    fit_parser = subparsers.add_parser(
        "fit-select",
        help="Run fit/calibration/model-selection only; test outcomes stay sealed",
    )
    fit_parser.add_argument("--expected-design-commit", required=True)
    validate_parser = subparsers.add_parser(
        "validate-test",
        help="Outcome-free preflight for the one sealed test attempt",
    )
    validate_parser.add_argument("--expected-selection-commit", required=True)
    consume_parser = subparsers.add_parser(
        "consume-test",
        help="Create the durable outcome-free attempt receipt for a later commit",
    )
    consume_parser.add_argument("--expected-selection-commit", required=True)
    evaluate_parser = subparsers.add_parser(
        "evaluate-test",
        help="Run the only permitted sealed test attempt; no force/resume option exists",
    )
    evaluate_parser.add_argument("--expected-attempt-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    paths = _default_paths()
    if arguments.command == "prepare-split":
        result = prepare_split_manifest(
            paths["protocol"],
            paths["candidates"],
            paths["features"],
            paths["split"],
            paths["split_sha"],
        )
        summary = {
            "status": "CONTENT_SPLIT_FROZEN_WITHOUT_OUTCOMES",
            "counts": result["counts"],
            "manifest": _manifest_hash_record(paths["split"]),
        }
    elif arguments.command == "freeze-design":
        result = freeze_design(
            paths["protocol"],
            paths["candidates"],
            paths["features"],
            paths["split"],
            paths["split_sha"],
            paths["design_freeze"],
            str(arguments.expected_implementation_commit),
        )
        summary = {
            "status": "CODE_AND_DESIGN_FROZEN",
            "implementation_commit": result["implementation_commit"],
            "freeze": _manifest_hash_record(paths["design_freeze"]),
        }
    elif arguments.command == "fit-select":
        result = fit_and_select(
            paths["protocol"],
            paths["candidates"],
            paths["features"],
            paths["split"],
            paths["split_sha"],
            paths["selection_lock"],
            paths["model_dir"],
            paths["design_freeze"],
            str(arguments.expected_design_commit),
        )
        summary = {
            "status": "DEVELOPMENT_SELECTION_FROZEN_TEST_UNREAD",
            "selected_candidate": result["global_selected_candidate_id"],
            "threshold": result["global_threshold"],
            "comparator": result["global_comparator"],
            "test_outcome_rows_parsed": result["test_outcome_rows_parsed"],
            "selection_lock": _manifest_hash_record(paths["selection_lock"]),
        }
    elif arguments.command == "validate-test":
        summary = validate_test(
            expected_selection_commit=str(arguments.expected_selection_commit),
            protocol_path=paths["protocol"],
            candidate_path=paths["candidates"],
            feature_path=paths["features"],
            split_path=paths["split"],
            split_sha_path=paths["split_sha"],
            design_freeze_path=paths["design_freeze"],
            selection_lock_path=paths["selection_lock"],
        )
    elif arguments.command == "consume-test":
        summary = consume_test_attempt(
            expected_selection_commit=str(arguments.expected_selection_commit),
            protocol_path=paths["protocol"],
            candidate_path=paths["candidates"],
            feature_path=paths["features"],
            split_path=paths["split"],
            split_sha_path=paths["split_sha"],
            design_freeze_path=paths["design_freeze"],
            selection_lock_path=paths["selection_lock"],
        )
    elif arguments.command == "evaluate-test":
        summary = evaluate_test(
            expected_attempt_commit=str(arguments.expected_attempt_commit),
            protocol_path=paths["protocol"],
            candidate_path=paths["candidates"],
            feature_path=paths["features"],
            split_path=paths["split"],
            split_sha_path=paths["split_sha"],
            design_freeze_path=paths["design_freeze"],
            selection_lock_path=paths["selection_lock"],
        )
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(arguments.command)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
