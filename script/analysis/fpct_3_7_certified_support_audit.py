from __future__ import annotations

"""CPU-only certified FPCT structural-support reaudit.

This audit executes the production ``exact_identity`` strategy and the common
``certified_slot0_v1`` sanitizer.  It is task/pair sharded, atomic and resumable.
No labels, answers, predictions or correctness fields enter its outputs.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PAIR_ORDER = ("tinyllama", "qwen25_0p5b", "llama32_1b", "qwen3_1p7b")
TASK_ORDER = ("ai2-arc", "openbookqa", "mmlu-redux")
SPLIT_ORDER = ("fit", "calibration", "model-selection", "test", "all")
TOP_K = 4
MAX_LENGTH = 1024
EXPECTED_BRANCH = "research/fpct-factorized-transport"
FPCT1B_LOCK = REPO_ROOT / "recipe/eval_recipe/fpct_1b/pre_audit_lock.json"
FPCT1B_ANALYSIS = REPO_ROOT / "script/analysis/fpct_1b_structural_support_audit.py"
FPCT1B_SPLITS = REPO_ROOT / "recipe/eval_recipe/fpct_1b/content_group_split_manifest.csv"
PROMPT_SOURCE = REPO_ROOT / "rosetta/utils/evaluate.py"
RECEIVER_CONFIG_PATH = Path(
    "/netdisk/lijunsi/c2c-route1-identifiability/models/Qwen3-0.6B/config.json"
)
MANIFEST_PATH = REPO_ROOT / "recipe/eval_recipe/fpct_3_7/certified_support_manifest.json"
SCRIPT_PATH = Path(__file__).resolve()
TEST_PATH = REPO_ROOT / "test/test_fpct_certified_support_audit.py"
SANITIZER_TEST_PATH = REPO_ROOT / "test/test_fpct_alignment_sanitizer.py"
DEFAULT_RESULT_BASE = Path(
    "local/final_results/fpct_factorized_transport/"
    "fpct_3_7_certified_support"
)
_VERIFIED_EXTERNAL_LOCKS: set[str] = set()


class CertifiedAuditError(RuntimeError):
    pass


def ensure_cpu_only() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise CertifiedAuditError('CUDA_VISIBLE_DEVICES must be explicitly set to ""')


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CertifiedAuditError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return format(value, ".17g")
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def atomic_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: csv_value(row.get(column)) for column in columns})
            count += 1
    temp.replace(path)
    return count


def load_module() -> Any:
    path = REPO_ROOT / "script/analysis/fpct_1b_structural_support_audit.py"
    spec = importlib.util.spec_from_file_location("fpct1b_for_certified", path)
    if spec is None or spec.loader is None:
        raise CertifiedAuditError("cannot load FPCT-1B module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_frozen_splits(samples: Sequence[Any]) -> dict[str, int]:
    rows = read_csv(FPCT1B_SPLITS)
    expected_columns = [
        "schema_version",
        "task",
        "sample_key_sha256",
        "content_group_sha256",
        "split",
    ]
    if not rows or list(rows[0]) != expected_columns:
        raise CertifiedAuditError("frozen split manifest schema mismatch")
    if len(rows) != 7265:
        raise CertifiedAuditError("frozen split manifest row count mismatch")
    by_sample: dict[str, tuple[str, str, str]] = {}
    group_splits: dict[str, str] = {}
    for row in rows:
        key = row["sample_key_sha256"]
        record = (
            row["task"],
            row["content_group_sha256"],
            row["split"],
        )
        if key in by_sample:
            raise CertifiedAuditError("duplicate sample in frozen split manifest")
        by_sample[key] = record
        prior = group_splits.setdefault(record[1], record[2])
        if prior != record[2]:
            raise CertifiedAuditError("content group spans frozen splits")
    counts: Counter[str] = Counter()
    for sample in samples:
        expected = by_sample.get(sample.sample_key_sha256)
        actual = (sample.task, sample.content_group_sha256, sample.split)
        if expected != actual:
            raise CertifiedAuditError(
                f"sample differs from frozen split manifest: {sample.sample_key_sha256}"
            )
        counts[f"{sample.task}/{sample.split}"] += 1
    if len(by_sample) != len(samples):
        raise CertifiedAuditError("frozen split/sample membership mismatch")
    return dict(sorted(counts.items()))


def result_root(execution_sha: str, result_base: Path) -> Path:
    return REPO_ROOT / result_base / f"rev_{execution_sha}"


def freeze(result_base: Path) -> None:
    ensure_cpu_only()
    if git("status", "--short"):
        raise CertifiedAuditError("freeze requires clean worktree")
    if git("branch", "--show-current") != EXPECTED_BRANCH:
        raise CertifiedAuditError("certified audit branch mismatch")
    execution_sha = git("rev-parse", "HEAD")
    if git("rev-parse", "@{upstream}") != execution_sha:
        raise CertifiedAuditError("certified audit local/upstream mismatch")
    manifest = read_json(MANIFEST_PATH)
    for record in manifest["tracked_artifacts"].values():
        path = REPO_ROOT / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise CertifiedAuditError(
                f"manifest tracked-artifact SHA mismatch: {record['path']}"
            )
    audit = load_module()
    fpct1b_lock = read_json(FPCT1B_LOCK)
    current_assets = audit.resolve_assets(audit.DEFAULT_SHARED_ROOT)
    if current_assets != fpct1b_lock["assets"]:
        raise CertifiedAuditError("tokenizer/dataset assets differ from FPCT-1B lock")
    samples = audit.load_projected_samples(audit.DEFAULT_SHARED_ROOT)
    canonical_input = audit.validate_canonical_samples(samples)
    if canonical_input != fpct1b_lock["canonical_input"]:
        raise CertifiedAuditError("canonical input differs from FPCT-1B lock")
    if sha256_file(FPCT1B_SPLITS) != fpct1b_lock["split_manifest"]["sha256"]:
        raise CertifiedAuditError("content-group split manifest changed")
    split_counts = validate_frozen_splits(samples)
    tracked_paths = {
        "manifest": MANIFEST_PATH,
        "analysis": SCRIPT_PATH,
        "tests": TEST_PATH,
        "sanitizer_tests": SANITIZER_TEST_PATH,
        "aligner": REPO_ROOT / "rosetta/model/aligner.py",
        "dataset_adapter": REPO_ROOT / "rosetta/train/dataset_adapters.py",
        "training_entry": REPO_ROOT / "script/train/SFT_train.py",
        "evaluation_entry": REPO_ROOT / "script/evaluation/unified_evaluator.py",
        "fpct_1b_lock": FPCT1B_LOCK,
        "fpct_1b_analysis": FPCT1B_ANALYSIS,
        "fpct_1b_splits": FPCT1B_SPLITS,
        "prompt_source": PROMPT_SOURCE,
        "fpct_3_5_protocol": REPO_ROOT / "FPCT_3_5_ALIGNMENT_CORRECTNESS_PROTOCOL.md",
        "fpct_3_5_result": REPO_ROOT / "recipe/eval_recipe/fpct_3_5/alignment_correctness_result_manifest.json",
    }
    lock = {
        "schema_version": 1,
        "stage": "FPCT-3.7",
        "status": "FROZEN",
        "execution_sha": execution_sha,
        "protocol_id": manifest["protocol_id"],
        "tracked": {
            name: {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}
            for name, path in tracked_paths.items()
        },
        "thresholds": manifest["readiness"],
        "max_length": MAX_LENGTH,
        "canonical_input": canonical_input,
        "split_counts": split_counts,
        "assets": current_assets,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "datasets", "pyarrow")
            },
        },
        "natural_audit_started": False,
        "gpu_authorized": False,
    }
    root = result_root(execution_sha, result_base)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "pre_audit_lock.json"
    if path.exists() and read_json(path) != lock:
        raise CertifiedAuditError("existing certified pre-audit lock differs")
    if not path.exists():
        atomic_json(path, lock)
    state_path = root / "controller_state.json"
    initial_state = {
        "schema_version": 1,
        "execution_sha": execution_sha,
        "state": "FROZEN",
        "completed_shards": [],
        "held_out_test_released": False,
    }
    if state_path.exists():
        existing_state = read_json(state_path)
        if existing_state.get("execution_sha") != execution_sha:
            raise CertifiedAuditError("existing controller state SHA mismatch")
    else:
        atomic_json(state_path, initial_state)
    print(json.dumps({"status": "FROZEN", "execution_sha": execution_sha, "lock_sha256": sha256_file(path)}, sort_keys=True))


def verify_lock(execution_sha: str, result_base: Path) -> tuple[Path, dict[str, Any]]:
    if git("rev-parse", "HEAD") != execution_sha:
        raise CertifiedAuditError("current HEAD differs from certified execution SHA")
    if git("branch", "--show-current") != EXPECTED_BRANCH:
        raise CertifiedAuditError("current branch differs from certified branch")
    if git("rev-parse", "@{upstream}") != execution_sha:
        raise CertifiedAuditError("current upstream differs from certified execution SHA")
    if git("status", "--short"):
        raise CertifiedAuditError("certified audit requires a clean worktree")
    root = result_root(execution_sha, result_base)
    lock = read_json(root / "pre_audit_lock.json")
    if lock.get("execution_sha") != execution_sha:
        raise CertifiedAuditError("certified audit execution SHA mismatch")
    for record in lock["tracked"].values():
        if sha256_file(REPO_ROOT / record["path"]) != record["sha256"]:
            raise CertifiedAuditError(f"frozen certified file changed: {record['path']}")
    if execution_sha not in _VERIFIED_EXTERNAL_LOCKS:
        audit = load_module()
        if audit.resolve_assets(audit.DEFAULT_SHARED_ROOT) != lock["assets"]:
            raise CertifiedAuditError("frozen tokenizer/dataset assets changed")
        runtime = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "datasets", "pyarrow")
            },
        }
        if runtime != lock["runtime"]:
            raise CertifiedAuditError("certified audit runtime changed")
        _VERIFIED_EXTERNAL_LOCKS.add(execution_sha)
    return root, lock


SAMPLE_COLUMNS = (
    "schema_version", "pair", "pair_type", "task", "split",
    "sample_key_sha256", "content_group_sha256", "eligible_parent_count",
    "raw_m0", "raw_m1", "raw_m2", "raw_m3", "raw_m4",
    "certified_m0", "certified_m1", "certified_m2", "certified_m3",
    "certified_m4", "offset_uncertified_parent_count",
    "offset_uncertified_sample",
    "has_raw_m2", "has_certified_m2", "has_raw_m3", "has_certified_m3",
    "has_raw_m4", "has_certified_m4", "exact_control",
    "receiver_native_slots", "raw_extra_slots", "certified_extra_slots",
    "raw_expansion_ratio", "certified_expansion_ratio",
)

GROUP_COLUMNS = (
    "schema_version", "pair", "pair_type", "task", "split",
    "content_group_sha256", "group_member_count", "member_consistent",
    "has_raw_m2", "has_certified_m2", "has_raw_m3", "has_certified_m3",
    "has_raw_m4", "has_certified_m4", "offset_uncertified",
)

EXCEPTION_COLUMNS = (
    "schema_version", "pair", "task", "split", "sample_key_sha256",
    "content_group_sha256", "parent_index", "raw_m", "sanitized_m",
    "certified", "offset_uncertified", "reason",
)

PAIR_TASK_COLUMNS = (
    "schema_version", "pair", "task", "split", "sample_count",
    "group_count", "raw_positive_groups", "certified_positive_groups",
    "raw_minus_certified_positive_groups", "raw_positive_samples",
    "certified_positive_samples", "raw_minus_certified_positive_samples",
    "raw_m3_positive_groups", "certified_m3_positive_groups",
    "raw_m4_positive_groups", "certified_m4_positive_groups",
    "raw_support_rate", "certified_support_rate",
    "raw_minus_certified_support_rate", "raw_wilson95_low",
    "raw_wilson95_high", "certified_wilson95_low",
    "certified_wilson95_high", "offset_uncertified_groups",
    "offset_uncertified_samples", "offset_uncertified_parents",
    "raw_m0", "raw_m1", "raw_m2", "raw_m3", "raw_m4",
    "certified_m0", "certified_m1", "certified_m2",
    "certified_m3", "certified_m4",
)

COMPACT_COLUMNS = (
    "schema_version", "pair", "task", "positive_groups",
    "raw_positive_groups", "raw_minus_certified_positive_groups",
    "raw_m3_positive_groups", "certified_m3_positive_groups",
    "raw_m4_positive_groups", "certified_m4_positive_groups",
    "total_groups", "raw_support_rate", "support_rate",
    "raw_minus_certified_support_rate", "raw_wilson95_low",
    "raw_wilson95_high", "wilson95_low", "wilson95_high",
    "bonferroni9_wilson_lcb_sensitivity",
    "raw_m0", "raw_m1", "raw_m2", "raw_m3", "raw_m4",
    "certified_m0", "certified_m1", "certified_m2",
    "certified_m3", "certified_m4",
)


def legal_count(
    indices: Sequence[int],
    weights: Sequence[float],
    source_length: int,
    *,
    original_source_length: int | None = None,
) -> int:
    original_source_length = (
        source_length if original_source_length is None else original_source_length
    )
    seen: set[int] = set()
    original_weights: list[float] = []
    retained_count = 0
    for index, weight in zip(indices, weights):
        index = int(index)
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise CertifiedAuditError("nonfinite/negative alignment mass")
        if weight <= 0:
            continue
        if index < 0 or index >= original_source_length:
            if weight > 0:
                raise CertifiedAuditError("positive mass on invalid source index")
        if index in seen:
            raise CertifiedAuditError("duplicate legal source index")
        seen.add(index)
        original_weights.append(weight)
        if index < source_length:
            retained_count += 1
    if len(seen) > TOP_K:
        raise CertifiedAuditError("candidate count exceeds top-k")
    if original_weights:
        if abs(sum(original_weights) - 1.0) > 1e-8:
            raise CertifiedAuditError("alignment mass is not L1 normalized")
        expected = 1.0 / len(original_weights)
        if max(abs(weight - expected) for weight in original_weights) > 1e-8:
            raise CertifiedAuditError("uniform alignment invariant failed")
    return retained_count


def make_aligner(receiver: Any, sender: Any, strategy: str) -> Any:
    from rosetta.model.aligner import TokenAligner
    return TokenAligner(
        receiver, sender, strategy=strategy,
        soft_alignment_score_mode="uniform",
        soft_alignment_boundary_bonus=0.5,
        soft_alignment_boundary_tolerance=1,
        soft_alignment_min_weight=0.0,
        soft_alignment_confidence_mode="none",
        soft_alignment_reweight_mode="none",
        soft_alignment_candidate_window=0,
        verbose=False,
    )


def shard_dir(root: Path, pair: str, task: str) -> Path:
    return root / "shards" / f"{pair}__{task}"


def run_shard(pair: str, task: str, execution_sha: str, result_base: Path) -> dict[str, Any]:
    ensure_cpu_only()
    root, _lock = verify_lock(execution_sha, result_base)
    directory = shard_dir(root, pair, task)
    staging = directory.with_name(directory.name + ".incomplete")
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("status") == "complete" and manifest.get("execution_sha") == execution_sha:
            expected_columns = {
                "sample": SAMPLE_COLUMNS,
                "group": GROUP_COLUMNS,
                "exception": EXCEPTION_COLUMNS,
            }
            for kind, artifact in manifest["artifacts"].items():
                path = Path(artifact["path"])
                if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                    raise CertifiedAuditError(
                        f"completed shard artifact mismatch: {path}"
                    )
                columns, rows = read_csv_table(path)
                if columns != list(expected_columns[kind]):
                    raise CertifiedAuditError(
                        f"completed shard schema mismatch: {path}"
                    )
                if len(rows) != int(artifact["rows"]):
                    raise CertifiedAuditError(
                        f"completed shard row-count mismatch: {path}"
                    )
            print(json.dumps({"status": "RESUME_SKIP", "pair": pair, "task": task}, sort_keys=True))
            return manifest
        raise CertifiedAuditError(f"partial certified shard: {manifest_path}")
    if directory.exists():
        raise CertifiedAuditError(f"partial certified shard: {directory}")
    if staging.exists():
        shutil.rmtree(staging)

    audit = load_module()
    lock = read_json(FPCT1B_LOCK)
    receiver, senders = audit.load_tokenizers(lock)
    raw_aligner = make_aligner(receiver, senders[pair], "soft_span_overlap_v2")
    exact_aligner = (
        make_aligner(receiver, senders[pair], "exact_identity")
        if pair == "qwen3_1p7b"
        else None
    )
    samples_all = audit.load_projected_samples(audit.DEFAULT_SHARED_ROOT)
    audit.validate_canonical_samples(samples_all)
    validate_frozen_splits(samples_all)
    samples = [sample for sample in samples_all if sample.task == task]
    pair_type = "same_tokenizer_control" if pair == "qwen3_1p7b" else "heterogeneous"
    sample_rows: list[dict[str, Any]] = []
    exception_rows: list[dict[str, Any]] = []

    for ordinal, sample in enumerate(samples, 1):
        messages = [{"role": "user", "content": audit.prompt_for_sample(sample)}]
        raw = raw_aligner.align_chat_messages_soft(
            messages, add_generation_prompt=True, enable_thinking=False,
            top_k=TOP_K, return_details=True, apply_confidence_control=False,
        )
        target_length = min(len(raw["slm_ids"]), MAX_LENGTH)
        source_length = min(len(raw["llm_ids"]), MAX_LENGTH)
        raw_counts: Counter[int] = Counter()
        for parent_index in range(target_length):
            if not raw["message_mask"][parent_index]:
                continue
            raw_counts[legal_count(
                raw["soft_alignment"]["source_indices"][parent_index],
                raw["soft_alignment"]["source_weights"][parent_index],
                source_length,
                original_source_length=len(raw["llm_ids"]),
            )] += 1

        if exact_aligner is not None:
            corrected = exact_aligner.align_chat_messages_soft(
                messages, add_generation_prompt=True, enable_thinking=False,
                top_k=TOP_K, return_details=True, apply_confidence_control=False,
            )
            target_length = min(len(corrected["slm_ids"]), MAX_LENGTH)
            source_length = min(len(corrected["llm_ids"]), MAX_LENGTH)
            certified_mask = [False] * len(corrected["slm_ids"])
            uncertified_mask = [False] * len(corrected["slm_ids"])
            reasons = ["exact_identity"] * len(corrected["slm_ids"])
            if corrected["soft_alignment"].get("fpct_extra_slots") != 0:
                raise CertifiedAuditError("exact identity created an FPCT extra slot")
            for parent_index in range(target_length):
                if not corrected["message_mask"][parent_index]:
                    continue
                if (
                    corrected["soft_alignment"]["source_indices"][parent_index]
                    != [parent_index, -1, -1, -1]
                    or corrected["soft_alignment"]["source_weights"][parent_index]
                    != [1.0, 0.0, 0.0, 0.0]
                    or corrected["soft_alignment"]["fallback_mask"][parent_index]
                ):
                    raise CertifiedAuditError(
                        "exact identity row invariant failed"
                    )
        else:
            corrected = raw_aligner.sanitize_fpct_soft_alignment(
                raw, target_length=target_length, source_length=source_length
            )
            certified_mask = corrected["soft_alignment"]["fpct_certified_mask"]
            uncertified_mask = corrected["soft_alignment"]["fpct_offset_uncertified_mask"]
            reasons = corrected["soft_alignment"]["fpct_certification_reason"]

        certified_counts: Counter[int] = Counter()
        uncertified_count = 0
        eligible_count = 0
        raw_extra = 0
        certified_extra = 0
        for parent_index in range(target_length):
            if not corrected["message_mask"][parent_index]:
                continue
            eligible_count += 1
            corrected_m = legal_count(
                corrected["soft_alignment"]["source_indices"][parent_index],
                corrected["soft_alignment"]["source_weights"][parent_index],
                source_length,
                original_source_length=len(corrected["llm_ids"]),
            )
            certified_counts[corrected_m] += 1
            raw_m = legal_count(
                raw["soft_alignment"]["source_indices"][parent_index],
                raw["soft_alignment"]["source_weights"][parent_index],
                min(len(raw["llm_ids"]), MAX_LENGTH),
                original_source_length=len(raw["llm_ids"]),
            )
            raw_extra += max(raw_m - 1, 0)
            certified_extra += max(corrected_m - 1, 0)
            uncertified = bool(uncertified_mask[parent_index])
            uncertified_count += int(uncertified)
            if raw_m >= 2 or uncertified or certified_mask[parent_index]:
                exception_rows.append({
                    "schema_version": 1, "pair": pair, "task": task,
                    "split": sample.split,
                    "sample_key_sha256": sample.sample_key_sha256,
                    "content_group_sha256": sample.content_group_sha256,
                    "parent_index": parent_index, "raw_m": raw_m,
                    "sanitized_m": corrected_m,
                    "certified": int(bool(certified_mask[parent_index])),
                    "offset_uncertified": int(uncertified),
                    "reason": reasons[parent_index],
                })
        native_slots = target_length
        sample_rows.append({
            "schema_version": 1, "pair": pair, "pair_type": pair_type,
            "task": task, "split": sample.split,
            "sample_key_sha256": sample.sample_key_sha256,
            "content_group_sha256": sample.content_group_sha256,
            "eligible_parent_count": eligible_count,
            **{f"raw_m{m}": raw_counts[m] for m in range(5)},
            **{f"certified_m{m}": certified_counts[m] for m in range(5)},
            "offset_uncertified_parent_count": uncertified_count,
            "offset_uncertified_sample": int(uncertified_count > 0),
            "has_raw_m2": int(sum(raw_counts[m] for m in (2, 3, 4)) > 0),
            "has_certified_m2": int(sum(certified_counts[m] for m in (2, 3, 4)) > 0),
            "has_raw_m3": int(sum(raw_counts[m] for m in (3, 4)) > 0),
            "has_certified_m3": int(sum(certified_counts[m] for m in (3, 4)) > 0),
            "has_raw_m4": int(raw_counts[4] > 0),
            "has_certified_m4": int(certified_counts[4] > 0),
            "exact_control": int(pair == "qwen3_1p7b"),
            "receiver_native_slots": native_slots,
            "raw_extra_slots": raw_extra,
            "certified_extra_slots": certified_extra,
            "raw_expansion_ratio": (native_slots + raw_extra) / native_slots,
            "certified_expansion_ratio": (native_slots + certified_extra) / native_slots,
        })
        if ordinal % 100 == 0 or ordinal == len(samples):
            print(json.dumps({"status": "PROGRESS", "pair": pair, "task": task, "done": ordinal, "total": len(samples)}, sort_keys=True), flush=True)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        groups[row["content_group_sha256"]].append(row)
    group_rows = []
    for group_hash, members in sorted(groups.items()):
        signatures = {
            (
                member["has_raw_m2"], member["has_certified_m2"],
                member["has_raw_m3"], member["has_certified_m3"],
                member["has_raw_m4"], member["has_certified_m4"],
                int(member["offset_uncertified_parent_count"] > 0),
                member["split"],
            )
            for member in members
        }
        if len(signatures) != 1:
            raise CertifiedAuditError(f"content-group member inconsistency: {group_hash}")
        first = members[0]
        group_rows.append({
            "schema_version": 1, "pair": pair, "pair_type": pair_type,
            "task": task, "split": first["split"],
            "content_group_sha256": group_hash,
            "group_member_count": len(members), "member_consistent": 1,
            "has_raw_m2": first["has_raw_m2"],
            "has_certified_m2": first["has_certified_m2"],
            "has_raw_m3": first["has_raw_m3"],
            "has_certified_m3": first["has_certified_m3"],
            "has_raw_m4": first["has_raw_m4"],
            "has_certified_m4": first["has_certified_m4"],
            "offset_uncertified": int(first["offset_uncertified_parent_count"] > 0),
        })

    staging.mkdir(parents=True, exist_ok=False)
    staged_sample_path = staging / "sample_support.csv"
    staged_group_path = staging / "group_support.csv"
    staged_exception_path = staging / "exception_parents.csv"
    sample_count = atomic_csv(staged_sample_path, SAMPLE_COLUMNS, sample_rows)
    group_count = atomic_csv(staged_group_path, GROUP_COLUMNS, group_rows)
    exception_count = atomic_csv(
        staged_exception_path, EXCEPTION_COLUMNS, exception_rows
    )
    sample_path = directory / "sample_support.csv"
    group_path = directory / "group_support.csv"
    exception_path = directory / "exception_parents.csv"
    manifest = {
        "schema_version": 1, "status": "complete", "execution_sha": execution_sha,
        "pair": pair, "task": task,
        "artifacts": {
            "sample": {"path": str(sample_path), "rows": sample_count, "sha256": sha256_file(staged_sample_path), "bytes": staged_sample_path.stat().st_size},
            "group": {"path": str(group_path), "rows": group_count, "sha256": sha256_file(staged_group_path), "bytes": staged_group_path.stat().st_size},
            "exception": {"path": str(exception_path), "rows": exception_count, "sha256": sha256_file(staged_exception_path), "bytes": staged_exception_path.stat().st_size},
        },
    }
    atomic_json(staging / "manifest.json", manifest)
    staging.replace(directory)
    return manifest


def wilson(successes: int, total: int) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def one_sided_wilson_lcb(
    successes: int, total: int, error: float = 0.05 / 9.0
) -> float | None:
    if total <= 0:
        return None
    z = statistics.NormalDist().inv_cdf(1.0 - error)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, center - radius)


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    def q(probability: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * probability
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {"mean": statistics.fmean(values), "p50": q(0.5), "p90": q(0.9), "p95": q(0.95), "max": ordered[-1]}


def derive_readiness(
    compact_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str, list[str], str | None]:
    rows: list[dict[str, Any]] = []
    ready: list[dict[str, Any]] = []
    for pair in PAIR_ORDER:
        pair_rows = [row for row in compact_rows if row["pair"] == pair]
        counts = {row["task"]: int(row["positive_groups"]) for row in pair_rows}
        rates = {row["task"]: float(row["support_rate"]) for row in pair_rows}
        pooled = sum(counts.values())
        minimum = min(counts.values())
        macro = statistics.fmean(rates.values())
        eligible = (
            pair != "qwen3_1p7b" and minimum >= 30 and pooled >= 100
        )
        record = {
            "pair": pair,
            "task_counts": counts,
            "pooled": pooled,
            "minimum_task_count": minimum,
            "task_macro_support_rate": macro,
            "ready": eligible,
        }
        rows.append(record)
        if eligible:
            ready.append(record)
    ready.sort(
        key=lambda row: (
            -row["minimum_task_count"],
            -row["task_macro_support_rate"],
            -row["pooled"],
            row["pair"],
        )
    )
    ranking = [row["pair"] for row in ready]
    heterogeneous_positive = sum(
        int(row["positive_groups"])
        for row in compact_rows
        if row["pair"] != "qwen3_1p7b"
    )
    if not ranking:
        status = "NO_SUPPORT" if heterogeneous_positive == 0 else "DIAGNOSTIC_ONLY"
        return rows, status, [], None
    status = "SINGLE_PAIR_PILOT_READY" if len(ranking) == 1 else "CROSS_PAIR_PILOT_READY"
    return rows, status, ranking, ranking[0]


def read_csv_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def read_csv(path: Path) -> list[dict[str, str]]:
    return read_csv_table(path)[1]


def finalize(execution_sha: str, result_base: Path) -> dict[str, Any]:
    root, _lock = verify_lock(execution_sha, result_base)
    summary_path = root / "certified_support_summary.json"
    if summary_path.exists():
        verify(execution_sha, result_base, require_terminal_state=False)
        summary = read_json(summary_path)
        state = read_json(root / "controller_state.json")
        state.update({
            "state": "CERTIFIED_SUPPORT_COMPLETE",
            "completed_shards": [
                f"{pair}/{task}" for pair in PAIR_ORDER for task in TASK_ORDER
            ],
            "support_gate_passed": bool(summary["support_gate_passed"]),
        })
        atomic_json(root / "controller_state.json", state)
        verify(execution_sha, result_base)
        return summary
    samples: list[dict[str, str]] = []
    groups: list[dict[str, str]] = []
    artifacts: dict[str, Any] = {}
    for pair in PAIR_ORDER:
        for task in TASK_ORDER:
            directory = shard_dir(root, pair, task)
            manifest = read_json(directory / "manifest.json")
            if manifest.get("status") != "complete" or manifest.get("execution_sha") != execution_sha:
                raise CertifiedAuditError(f"incomplete shard: {pair}/{task}")
            for kind, collection in (("sample", samples), ("group", groups)):
                path = Path(manifest["artifacts"][kind]["path"])
                if sha256_file(path) != manifest["artifacts"][kind]["sha256"]:
                    raise CertifiedAuditError(f"shard artifact hash mismatch: {path}")
                collection.extend(read_csv(path))
            exception_path = Path(manifest["artifacts"]["exception"]["path"])
            if (
                sha256_file(exception_path)
                != manifest["artifacts"]["exception"]["sha256"]
                or len(read_csv(exception_path))
                != int(manifest["artifacts"]["exception"]["rows"])
            ):
                raise CertifiedAuditError(
                    f"shard exception artifact mismatch: {exception_path}"
                )
            artifacts[f"{pair}/{task}"] = manifest["artifacts"]

    pair_task_rows: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    for pair in PAIR_ORDER:
        for task in TASK_ORDER:
            for split_name in SPLIT_ORDER:
                selected_groups = [
                    row for row in groups
                    if row["pair"] == pair and row["task"] == task
                    and (split_name == "all" or row["split"] == split_name)
                ]
                raw_positive = sum(int(row["has_raw_m2"]) for row in selected_groups)
                certified_positive = sum(int(row["has_certified_m2"]) for row in selected_groups)
                raw_m3_positive = sum(int(row["has_raw_m3"]) for row in selected_groups)
                certified_m3_positive = sum(int(row["has_certified_m3"]) for row in selected_groups)
                raw_m4_positive = sum(int(row["has_raw_m4"]) for row in selected_groups)
                certified_m4_positive = sum(int(row["has_certified_m4"]) for row in selected_groups)
                uncertified_groups = sum(int(row["offset_uncertified"]) for row in selected_groups)
                total_groups = len(selected_groups)
                raw_low, raw_high = wilson(raw_positive, total_groups)
                certified_low, certified_high = wilson(
                    certified_positive, total_groups
                )
                selected_samples = [
                    row for row in samples
                    if row["pair"] == pair and row["task"] == task
                    and (split_name == "all" or row["split"] == split_name)
                ]
                raw_parent = Counter()
                certified_parent = Counter()
                uncertified_parents = 0
                uncertified_samples = 0
                for row in selected_samples:
                    for m in range(5):
                        raw_parent[m] += int(row[f"raw_m{m}"])
                        certified_parent[m] += int(row[f"certified_m{m}"])
                    uncertified_parents += int(row["offset_uncertified_parent_count"])
                    uncertified_samples += int(row["offset_uncertified_sample"])
                raw_positive_samples = sum(
                    int(row["has_raw_m2"]) for row in selected_samples
                )
                certified_positive_samples = sum(
                    int(row["has_certified_m2"]) for row in selected_samples
                )
                record = {
                    "schema_version": 1, "pair": pair, "task": task,
                    "split": split_name, "sample_count": len(selected_samples),
                    "group_count": total_groups,
                    "raw_positive_groups": raw_positive,
                    "certified_positive_groups": certified_positive,
                    "raw_minus_certified_positive_groups": (
                        raw_positive - certified_positive
                    ),
                    "raw_positive_samples": raw_positive_samples,
                    "certified_positive_samples": certified_positive_samples,
                    "raw_minus_certified_positive_samples": (
                        raw_positive_samples - certified_positive_samples
                    ),
                    "raw_m3_positive_groups": raw_m3_positive,
                    "certified_m3_positive_groups": certified_m3_positive,
                    "raw_m4_positive_groups": raw_m4_positive,
                    "certified_m4_positive_groups": certified_m4_positive,
                    "raw_support_rate": raw_positive / total_groups if total_groups else None,
                    "certified_support_rate": certified_positive / total_groups if total_groups else None,
                    "raw_minus_certified_support_rate": (
                        (raw_positive - certified_positive) / total_groups
                        if total_groups else None
                    ),
                    "raw_wilson95_low": raw_low,
                    "raw_wilson95_high": raw_high,
                    "certified_wilson95_low": certified_low,
                    "certified_wilson95_high": certified_high,
                    "offset_uncertified_groups": uncertified_groups,
                    "offset_uncertified_samples": uncertified_samples,
                    "offset_uncertified_parents": uncertified_parents,
                    **{f"raw_m{m}": raw_parent[m] for m in range(5)},
                    **{f"certified_m{m}": certified_parent[m] for m in range(5)},
                }
                pair_task_rows.append(record)
                if split_name == "all":
                    raw_expansion = [float(row["raw_expansion_ratio"]) for row in selected_samples]
                    certified_expansion = [float(row["certified_expansion_ratio"]) for row in selected_samples]
                    record["raw_expansion"] = quantiles(raw_expansion)
                    record["certified_expansion"] = quantiles(certified_expansion)
            fit_cal_groups = [
                row for row in groups
                if row["pair"] == pair and row["task"] == task
                and row["split"] in {"fit", "calibration"}
            ]
            positive = sum(int(row["has_certified_m2"]) for row in fit_cal_groups)
            raw_positive = sum(int(row["has_raw_m2"]) for row in fit_cal_groups)
            raw_low, raw_high = wilson(raw_positive, len(fit_cal_groups))
            low, high = wilson(positive, len(fit_cal_groups))
            fit_cal_samples = [
                row for row in samples
                if row["pair"] == pair and row["task"] == task
                and row["split"] in {"fit", "calibration"}
            ]
            raw_parent = Counter()
            certified_parent = Counter()
            for row in fit_cal_samples:
                for m in range(5):
                    raw_parent[m] += int(row[f"raw_m{m}"])
                    certified_parent[m] += int(row[f"certified_m{m}"])
            compact_rows.append({
                "schema_version": 1, "pair": pair, "task": task,
                "positive_groups": positive,
                "raw_positive_groups": raw_positive,
                "raw_minus_certified_positive_groups": raw_positive - positive,
                "raw_m3_positive_groups": sum(
                    int(row["has_raw_m3"]) for row in fit_cal_groups
                ),
                "certified_m3_positive_groups": sum(
                    int(row["has_certified_m3"]) for row in fit_cal_groups
                ),
                "raw_m4_positive_groups": sum(
                    int(row["has_raw_m4"]) for row in fit_cal_groups
                ),
                "certified_m4_positive_groups": sum(
                    int(row["has_certified_m4"]) for row in fit_cal_groups
                ),
                "total_groups": len(fit_cal_groups),
                "raw_support_rate": raw_positive / len(fit_cal_groups),
                "support_rate": positive / len(fit_cal_groups),
                "raw_minus_certified_support_rate": (
                    raw_positive - positive
                ) / len(fit_cal_groups),
                "raw_wilson95_low": raw_low,
                "raw_wilson95_high": raw_high,
                "wilson95_low": low, "wilson95_high": high,
                "bonferroni9_wilson_lcb_sensitivity": (
                    one_sided_wilson_lcb(positive, len(fit_cal_groups))
                    if pair != "qwen3_1p7b" else None
                ),
                **{f"raw_m{m}": raw_parent[m] for m in range(5)},
                **{f"certified_m{m}": certified_parent[m] for m in range(5)},
            })

    readiness_rows, readiness, ready_pairs, selected = derive_readiness(
        compact_rows
    )
    tinyllama_record = next(
        row for row in readiness_rows if row["pair"] == "tinyllama"
    )
    tinyllama_ready = bool(tinyllama_record["ready"])

    qwen_samples = [row for row in samples if row["pair"] == "qwen3_1p7b"]
    if any(
        int(row["certified_m0"]) != 0
        or int(row["certified_m2"]) != 0
        or int(row["certified_m3"]) != 0
        or int(row["certified_m4"]) != 0
        or int(row["certified_m1"]) != int(row["eligible_parent_count"])
        or int(row["certified_extra_slots"]) != 0
        for row in qwen_samples
    ):
        raise CertifiedAuditError("Qwen exact identity non-m1 stratum detected")

    pair_task_path = root / "pair_task_support.csv"
    compact_path = root / "certified_support_aggregates.csv"
    atomic_csv(pair_task_path, PAIR_TASK_COLUMNS, pair_task_rows)
    atomic_csv(compact_path, COMPACT_COLUMNS, compact_rows)

    receiver_config_path = RECEIVER_CONFIG_PATH
    receiver_config = json.loads(receiver_config_path.read_text(encoding="utf-8"))
    layers = int(receiver_config["num_hidden_layers"])
    hq = int(receiver_config["num_attention_heads"])
    hkv = int(receiver_config["num_key_value_heads"])
    head_dim = int(
        receiver_config.get("head_dim", receiver_config["hidden_size"] // hq)
    )
    kv_atom_bytes = 2 * hkv * head_dim * 2
    resource_rows = []
    for pair in PAIR_ORDER:
        for task in TASK_ORDER:
            selected_samples = [
                row for row in samples
                if row["pair"] == pair and row["task"] == task
            ]
            raw_ratios = [float(row["raw_expansion_ratio"]) for row in selected_samples]
            certified_ratios = [
                float(row["certified_expansion_ratio"]) for row in selected_samples
            ]
            raw_sidecar_atoms = [
                sum(m * int(row[f"raw_m{m}"]) for m in (2, 3, 4))
                for row in selected_samples
            ]
            certified_sidecar_atoms = [
                sum(m * int(row[f"certified_m{m}"]) for m in (2, 3, 4))
                for row in selected_samples
            ]
            raw_extra = [int(row["raw_extra_slots"]) for row in selected_samples]
            certified_extra = [
                int(row["certified_extra_slots"]) for row in selected_samples
            ]
            dense_topk4_ratios = [
                (
                    int(row["receiver_native_slots"])
                    + 3 * int(row["eligible_parent_count"])
                ) / int(row["receiver_native_slots"])
                for row in selected_samples
            ]
            raw_sidecar_per_layer = [
                atoms * kv_atom_bytes for atoms in raw_sidecar_atoms
            ]
            certified_sidecar_per_layer = [
                atoms * kv_atom_bytes for atoms in certified_sidecar_atoms
            ]
            dense_sidecar_per_layer = [
                4 * int(row["eligible_parent_count"]) * kv_atom_bytes
                for row in selected_samples
            ]
            resource_rows.append({
                "pair": pair,
                "task": task,
                "sample_count": len(selected_samples),
                "raw_expansion": quantiles(raw_ratios),
                "certified_expansion": quantiles(certified_ratios),
                "raw_attention_score_flop_ratio": quantiles(raw_ratios),
                "certified_attention_score_flop_ratio": quantiles(certified_ratios),
                "dense_topk4_attention_score_flop_ratio": quantiles(
                    dense_topk4_ratios
                ),
                "raw_sidecar_bytes_per_layer": quantiles(raw_sidecar_per_layer),
                "certified_sidecar_bytes_per_layer": quantiles(
                    certified_sidecar_per_layer
                ),
                "dense_topk4_sidecar_bytes_per_layer": quantiles(
                    dense_sidecar_per_layer
                ),
                "raw_sidecar_cache_bytes_all_layers": quantiles(
                    [value * layers for value in raw_sidecar_per_layer]
                ),
                "certified_sidecar_cache_bytes_all_layers": quantiles(
                    [value * layers for value in certified_sidecar_per_layer]
                ),
                "dense_topk4_sidecar_cache_bytes_all_layers": quantiles(
                    [value * layers for value in dense_sidecar_per_layer]
                ),
                "raw_packed_extra_kv_bytes_all_layers": quantiles(
                    [value * kv_atom_bytes * layers for value in raw_extra]
                ),
                "certified_packed_extra_kv_bytes_all_layers": quantiles(
                    [value * kv_atom_bytes * layers for value in certified_extra]
                ),
            })
    resource = {
        "schema_version": 1,
        "execution_sha": execution_sha,
        "receiver_config": {
            "path": str(receiver_config_path),
            "sha256": sha256_file(receiver_config_path),
            "num_hidden_layers": layers,
            "num_attention_heads": hq,
            "num_key_value_heads": hkv,
            "head_dim": head_dim,
            "kv_atom_bytes_per_layer": kv_atom_bytes,
        },
        "rows": resource_rows,
    }
    resource_path = root / "resource_estimates.json"
    atomic_json(resource_path, resource)
    result = {
        "schema_version": 1, "stage": "FPCT-3.7", "status": "COMPLETE",
        "execution_sha": execution_sha, "global_readiness": readiness,
        "selected_pair_by_frozen_ranking": selected,
        "ready_pairs": ready_pairs,
        "readiness_rows": readiness_rows,
        "confirmatory_pair": "tinyllama",
        "confirmatory_pair_status": (
            "SINGLE_PAIR_PILOT_READY" if tinyllama_ready else "NO_GO_GPU"
        ),
        "artifacts": {
            "pair_task": {"path": str(pair_task_path), "rows": len(pair_task_rows), "sha256": sha256_file(pair_task_path), "bytes": pair_task_path.stat().st_size},
            "compact": {"path": str(compact_path), "rows": len(compact_rows), "sha256": sha256_file(compact_path), "bytes": compact_path.stat().st_size},
            "resource": {"path": str(resource_path), "rows": len(resource_rows), "sha256": sha256_file(resource_path), "bytes": resource_path.stat().st_size},
            "shards": artifacts,
        },
        "support_gate_passed": tinyllama_ready,
        "next_stage_authorized": tinyllama_ready,
        "gpu_authorized": False,
        "claim_boundary": "certified structural support only",
    }
    atomic_json(summary_path, result)
    verify(execution_sha, result_base, require_terminal_state=False)
    state = read_json(root / "controller_state.json")
    state.update({
        "state": "CERTIFIED_SUPPORT_COMPLETE",
        "completed_shards": [
            f"{pair}/{task}" for pair in PAIR_ORDER for task in TASK_ORDER
        ],
        "support_gate_passed": tinyllama_ready,
    })
    atomic_json(root / "controller_state.json", state)
    verify(execution_sha, result_base)
    print(json.dumps({"status": result["confirmatory_pair_status"], "global_readiness": readiness, "selected_pair_by_frozen_ranking": selected, "summary_sha256": sha256_file(summary_path)}, sort_keys=True))
    return result


def _assert_csv_value(actual: str, expected: Any, label: str) -> None:
    if expected is None:
        if actual != "":
            raise CertifiedAuditError(f"{label}: expected empty, found {actual}")
        return
    if isinstance(expected, (int, bool)):
        if int(actual) != int(expected):
            raise CertifiedAuditError(f"{label}: integer mismatch")
        return
    if isinstance(expected, float):
        value = float(actual)
        if not math.isfinite(value) or not math.isclose(
            value, expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise CertifiedAuditError(f"{label}: float mismatch")
        return
    if actual != str(expected):
        raise CertifiedAuditError(f"{label}: value mismatch")


def _assert_json_close(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise CertifiedAuditError(f"{label}: JSON object schema mismatch")
        for key in expected:
            _assert_json_close(actual[key], expected[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise CertifiedAuditError(f"{label}: JSON list mismatch")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_json_close(left, right, f"{label}[{index}]")
        return
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isfinite(float(actual)):
            raise CertifiedAuditError(f"{label}: nonfinite JSON value")
        if not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12):
            raise CertifiedAuditError(f"{label}: JSON float mismatch")
        return
    if actual != expected:
        raise CertifiedAuditError(f"{label}: JSON value mismatch")


def verify(
    execution_sha: str,
    result_base: Path,
    *,
    require_terminal_state: bool = True,
) -> None:
    root, lock = verify_lock(execution_sha, result_base)
    summary_path = root / "certified_support_summary.json"
    summary = read_json(summary_path)
    if summary.get("execution_sha") != execution_sha:
        raise CertifiedAuditError("certified summary execution SHA mismatch")

    samples: list[dict[str, str]] = []
    groups: list[dict[str, str]] = []
    expected_schema = {
        "sample": SAMPLE_COLUMNS,
        "group": GROUP_COLUMNS,
        "exception": EXCEPTION_COLUMNS,
    }
    for pair in PAIR_ORDER:
        for task in TASK_ORDER:
            manifest_path = shard_dir(root, pair, task) / "manifest.json"
            manifest = read_json(manifest_path)
            if (
                manifest.get("status") != "complete"
                or manifest.get("execution_sha") != execution_sha
                or manifest.get("pair") != pair
                or manifest.get("task") != task
            ):
                raise CertifiedAuditError(f"invalid shard manifest: {pair}/{task}")
            for kind, artifact in manifest["artifacts"].items():
                path = Path(artifact["path"])
                columns, rows = read_csv_table(path)
                if columns != list(expected_schema[kind]):
                    raise CertifiedAuditError(f"shard schema mismatch: {path}")
                if (
                    sha256_file(path) != artifact["sha256"]
                    or path.stat().st_size != int(artifact["bytes"])
                    or len(rows) != int(artifact["rows"])
                ):
                    raise CertifiedAuditError(f"shard artifact mismatch: {path}")
                if kind == "sample":
                    samples.extend(rows)
                elif kind == "group":
                    groups.extend(rows)

    expected_sample_rows = int(lock["canonical_input"]["total_rows"]) * len(
        PAIR_ORDER
    )
    expected_group_rows = int(
        lock["canonical_input"]["total_distinct_content_groups"]
    ) * len(PAIR_ORDER)
    if len(samples) != expected_sample_rows or len(groups) != expected_group_rows:
        raise CertifiedAuditError("certified shard total row count mismatch")
    qwen_samples = [row for row in samples if row["pair"] == "qwen3_1p7b"]
    if len(qwen_samples) != int(lock["canonical_input"]["total_rows"]) or any(
        int(row["certified_m0"]) != 0
        or int(row["certified_m1"]) != int(row["eligible_parent_count"])
        or any(int(row[f"certified_m{m}"]) != 0 for m in (2, 3, 4))
        or int(row["certified_extra_slots"]) != 0
        for row in qwen_samples
    ):
        raise CertifiedAuditError("all-split Qwen exact-identity invariant failed")

    pair_task_artifact = summary["artifacts"]["pair_task"]
    pair_task_path = Path(pair_task_artifact["path"])
    pair_task_columns, pair_task_rows = read_csv_table(pair_task_path)
    expected_pair_task_rows = len(PAIR_ORDER) * len(TASK_ORDER) * len(SPLIT_ORDER)
    if (
        pair_task_columns != list(PAIR_TASK_COLUMNS)
        or len(pair_task_rows) != expected_pair_task_rows
    ):
        raise CertifiedAuditError("pair-task aggregate schema/count mismatch")
    if (
        sha256_file(pair_task_path) != pair_task_artifact["sha256"]
        or pair_task_path.stat().st_size != int(pair_task_artifact["bytes"])
        or int(pair_task_artifact["rows"]) != expected_pair_task_rows
    ):
        raise CertifiedAuditError("pair-task aggregate artifact mismatch")
    pair_task_map = {
        (row["pair"], row["task"], row["split"]): row
        for row in pair_task_rows
    }
    if len(pair_task_map) != expected_pair_task_rows:
        raise CertifiedAuditError("duplicate pair-task aggregate key")
    expected_pair_task_order = [
        (pair, task, split_name)
        for pair in PAIR_ORDER
        for task in TASK_ORDER
        for split_name in SPLIT_ORDER
    ]
    if [
        (row["pair"], row["task"], row["split"])
        for row in pair_task_rows
    ] != expected_pair_task_order:
        raise CertifiedAuditError("pair-task aggregate order mismatch")

    for pair in PAIR_ORDER:
        for task in TASK_ORDER:
            for split_name in SPLIT_ORDER:
                selected_groups = [
                    row for row in groups
                    if row["pair"] == pair and row["task"] == task
                    and (split_name == "all" or row["split"] == split_name)
                ]
                selected_samples = [
                    row for row in samples
                    if row["pair"] == pair and row["task"] == task
                    and (split_name == "all" or row["split"] == split_name)
                ]
                raw_positive = sum(int(row["has_raw_m2"]) for row in selected_groups)
                certified_positive = sum(
                    int(row["has_certified_m2"]) for row in selected_groups
                )
                raw_low, raw_high = wilson(raw_positive, len(selected_groups))
                cert_low, cert_high = wilson(
                    certified_positive, len(selected_groups)
                )
                raw_parent = Counter()
                certified_parent = Counter()
                for sample in selected_samples:
                    for m in range(5):
                        raw_parent[m] += int(sample[f"raw_m{m}"])
                        certified_parent[m] += int(sample[f"certified_m{m}"])
                expected = {
                    "schema_version": 1,
                    "pair": pair,
                    "task": task,
                    "split": split_name,
                    "sample_count": len(selected_samples),
                    "group_count": len(selected_groups),
                    "raw_positive_groups": raw_positive,
                    "certified_positive_groups": certified_positive,
                    "raw_minus_certified_positive_groups": raw_positive - certified_positive,
                    "raw_positive_samples": sum(int(row["has_raw_m2"]) for row in selected_samples),
                    "certified_positive_samples": sum(int(row["has_certified_m2"]) for row in selected_samples),
                    "raw_minus_certified_positive_samples": sum(int(row["has_raw_m2"]) - int(row["has_certified_m2"]) for row in selected_samples),
                    "raw_m3_positive_groups": sum(int(row["has_raw_m3"]) for row in selected_groups),
                    "certified_m3_positive_groups": sum(int(row["has_certified_m3"]) for row in selected_groups),
                    "raw_m4_positive_groups": sum(int(row["has_raw_m4"]) for row in selected_groups),
                    "certified_m4_positive_groups": sum(int(row["has_certified_m4"]) for row in selected_groups),
                    "raw_support_rate": raw_positive / len(selected_groups),
                    "certified_support_rate": certified_positive / len(selected_groups),
                    "raw_minus_certified_support_rate": (raw_positive - certified_positive) / len(selected_groups),
                    "raw_wilson95_low": raw_low,
                    "raw_wilson95_high": raw_high,
                    "certified_wilson95_low": cert_low,
                    "certified_wilson95_high": cert_high,
                    "offset_uncertified_groups": sum(int(row["offset_uncertified"]) for row in selected_groups),
                    "offset_uncertified_samples": sum(int(row["offset_uncertified_sample"]) for row in selected_samples),
                    "offset_uncertified_parents": sum(int(row["offset_uncertified_parent_count"]) for row in selected_samples),
                    **{f"raw_m{m}": raw_parent[m] for m in range(5)},
                    **{f"certified_m{m}": certified_parent[m] for m in range(5)},
                }
                actual = pair_task_map[(pair, task, split_name)]
                for key, value in expected.items():
                    _assert_csv_value(actual[key], value, f"pair_task/{pair}/{task}/{split_name}/{key}")

    compact_artifact = summary["artifacts"]["compact"]
    compact_path = Path(compact_artifact["path"])
    compact_columns, compact_rows = read_csv_table(compact_path)
    expected_compact_rows = len(PAIR_ORDER) * len(TASK_ORDER)
    if (
        compact_columns != list(COMPACT_COLUMNS)
        or len(compact_rows) != expected_compact_rows
    ):
        raise CertifiedAuditError("compact aggregate schema/count mismatch")
    expected_compact_order = [
        (pair, task) for pair in PAIR_ORDER for task in TASK_ORDER
    ]
    if [(row["pair"], row["task"]) for row in compact_rows] != expected_compact_order:
        raise CertifiedAuditError("compact aggregate key/order mismatch")
    if (
        sha256_file(compact_path) != compact_artifact["sha256"]
        or compact_path.stat().st_size != int(compact_artifact["bytes"])
        or int(compact_artifact["rows"]) != expected_compact_rows
    ):
        raise CertifiedAuditError("compact aggregate artifact mismatch")
    compact_numeric: list[dict[str, Any]] = []
    for row in compact_rows:
        pair = row["pair"]
        task = row["task"]
        selected_groups = [
            group for group in groups
            if group["pair"] == pair and group["task"] == task
            and group["split"] in {"fit", "calibration"}
        ]
        selected_samples = [
            sample for sample in samples
            if sample["pair"] == pair and sample["task"] == task
            and sample["split"] in {"fit", "calibration"}
        ]
        positive = sum(int(group["has_certified_m2"]) for group in selected_groups)
        raw_positive = sum(int(group["has_raw_m2"]) for group in selected_groups)
        expected_lcb = (
            one_sided_wilson_lcb(positive, len(selected_groups))
            if pair != "qwen3_1p7b" else None
        )
        raw_parent = Counter()
        certified_parent = Counter()
        for sample in selected_samples:
            for m in range(5):
                raw_parent[m] += int(sample[f"raw_m{m}"])
                certified_parent[m] += int(sample[f"certified_m{m}"])
        expected = {
            "schema_version": 1,
            "pair": pair,
            "task": task,
            "positive_groups": positive,
            "raw_positive_groups": raw_positive,
            "raw_minus_certified_positive_groups": raw_positive - positive,
            "raw_m3_positive_groups": sum(
                int(group["has_raw_m3"]) for group in selected_groups
            ),
            "certified_m3_positive_groups": sum(
                int(group["has_certified_m3"]) for group in selected_groups
            ),
            "raw_m4_positive_groups": sum(
                int(group["has_raw_m4"]) for group in selected_groups
            ),
            "certified_m4_positive_groups": sum(
                int(group["has_certified_m4"]) for group in selected_groups
            ),
            "total_groups": len(selected_groups),
            "raw_support_rate": raw_positive / len(selected_groups),
            "support_rate": positive / len(selected_groups),
            "raw_minus_certified_support_rate": (raw_positive - positive) / len(selected_groups),
            "bonferroni9_wilson_lcb_sensitivity": expected_lcb,
            **{f"raw_m{m}": raw_parent[m] for m in range(5)},
            **{f"certified_m{m}": certified_parent[m] for m in range(5)},
        }
        raw_low, raw_high = wilson(raw_positive, len(selected_groups))
        low, high = wilson(positive, len(selected_groups))
        expected.update({
            "raw_wilson95_low": raw_low,
            "raw_wilson95_high": raw_high,
            "wilson95_low": low,
            "wilson95_high": high,
        })
        if set(expected) != set(COMPACT_COLUMNS):
            raise CertifiedAuditError("independent compact schema is incomplete")
        for key in COMPACT_COLUMNS:
            value = expected[key]
            _assert_csv_value(row[key], value, f"compact/{pair}/{task}/{key}")
        compact_numeric.append({
            "pair": pair,
            "task": task,
            "positive_groups": int(row["positive_groups"]),
            "total_groups": int(row["total_groups"]),
            "support_rate": float(row["support_rate"]),
        })

    readiness_rows, global_readiness, ready_pairs, selected = derive_readiness(
        compact_numeric
    )
    _assert_json_close(summary["readiness_rows"], readiness_rows, "readiness")
    if (
        summary["global_readiness"] != global_readiness
        or summary["ready_pairs"] != ready_pairs
        or summary["selected_pair_by_frozen_ranking"] != selected
    ):
        raise CertifiedAuditError("readiness/ranking mismatch")
    tiny_ready = bool(
        next(row for row in readiness_rows if row["pair"] == "tinyllama")["ready"]
    )
    if (
        bool(summary["support_gate_passed"]) != tiny_ready
        or bool(summary["next_stage_authorized"]) != tiny_ready
        or bool(summary["gpu_authorized"])
    ):
        raise CertifiedAuditError("support/GPU gate mismatch")

    resource_artifact = summary["artifacts"]["resource"]
    resource_path = Path(resource_artifact["path"])
    resource = read_json(resource_path)
    if (
        sha256_file(resource_path) != resource_artifact["sha256"]
        or resource_path.stat().st_size != int(resource_artifact["bytes"])
        or len(resource.get("rows", [])) != expected_compact_rows
        or int(resource_artifact["rows"]) != expected_compact_rows
        or resource.get("execution_sha") != execution_sha
    ):
        raise CertifiedAuditError("resource artifact mismatch")
    config_path = Path(resource["receiver_config"]["path"])
    if sha256_file(config_path) != resource["receiver_config"]["sha256"]:
        raise CertifiedAuditError("receiver config provenance mismatch")
    kv_atom_bytes = int(resource["receiver_config"]["kv_atom_bytes_per_layer"])
    layers = int(resource["receiver_config"]["num_hidden_layers"])
    resource_map = {(row["pair"], row["task"]): row for row in resource["rows"]}
    if [(row["pair"], row["task"]) for row in resource["rows"]] != (
        expected_compact_order
    ):
        raise CertifiedAuditError("resource row order mismatch")
    for pair in PAIR_ORDER:
        for task in TASK_ORDER:
            selected_samples = [
                row for row in samples if row["pair"] == pair and row["task"] == task
            ]
            raw_atoms = [sum(m * int(row[f"raw_m{m}"]) for m in (2, 3, 4)) for row in selected_samples]
            certified_atoms = [sum(m * int(row[f"certified_m{m}"]) for m in (2, 3, 4)) for row in selected_samples]
            expected = {
                "pair": pair,
                "task": task,
                "sample_count": len(selected_samples),
                "raw_expansion": quantiles([float(row["raw_expansion_ratio"]) for row in selected_samples]),
                "certified_expansion": quantiles([float(row["certified_expansion_ratio"]) for row in selected_samples]),
                "raw_attention_score_flop_ratio": quantiles([float(row["raw_expansion_ratio"]) for row in selected_samples]),
                "certified_attention_score_flop_ratio": quantiles([float(row["certified_expansion_ratio"]) for row in selected_samples]),
                "dense_topk4_attention_score_flop_ratio": quantiles([(int(row["receiver_native_slots"]) + 3 * int(row["eligible_parent_count"])) / int(row["receiver_native_slots"]) for row in selected_samples]),
                "raw_sidecar_bytes_per_layer": quantiles([atoms * kv_atom_bytes for atoms in raw_atoms]),
                "certified_sidecar_bytes_per_layer": quantiles([atoms * kv_atom_bytes for atoms in certified_atoms]),
                "dense_topk4_sidecar_bytes_per_layer": quantiles([4 * int(row["eligible_parent_count"]) * kv_atom_bytes for row in selected_samples]),
                "raw_sidecar_cache_bytes_all_layers": quantiles([atoms * kv_atom_bytes * layers for atoms in raw_atoms]),
                "certified_sidecar_cache_bytes_all_layers": quantiles([atoms * kv_atom_bytes * layers for atoms in certified_atoms]),
                "dense_topk4_sidecar_cache_bytes_all_layers": quantiles([4 * int(row["eligible_parent_count"]) * kv_atom_bytes * layers for row in selected_samples]),
                "raw_packed_extra_kv_bytes_all_layers": quantiles([int(row["raw_extra_slots"]) * kv_atom_bytes * layers for row in selected_samples]),
                "certified_packed_extra_kv_bytes_all_layers": quantiles([int(row["certified_extra_slots"]) * kv_atom_bytes * layers for row in selected_samples]),
            }
            _assert_json_close(resource_map[(pair, task)], expected, f"resource/{pair}/{task}")

    state = read_json(root / "controller_state.json")
    if state.get("execution_sha") != execution_sha or bool(
        state.get("held_out_test_released")
    ):
        raise CertifiedAuditError("certified controller identity/firewall mismatch")
    if require_terminal_state and (
        state.get("state") != "CERTIFIED_SUPPORT_COMPLETE"
        or len(state.get("completed_shards", [])) != expected_compact_rows
        or bool(state.get("support_gate_passed"))
        != bool(summary.get("support_gate_passed"))
    ):
        raise CertifiedAuditError("certified controller terminal state mismatch")
    if lock.get("gpu_authorized") or summary.get("gpu_authorized"):
        raise CertifiedAuditError("GPU was authorized before hardening gates")
    print(json.dumps({"status": "VERIFIED", "pair_task_rows": expected_pair_task_rows, "compact_rows": expected_compact_rows, "sample_rows": len(samples), "group_rows": len(groups)}, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "audit", "finalize", "verify"))
    parser.add_argument("--execution-sha")
    parser.add_argument("--pair", choices=PAIR_ORDER + ("all",), default="all")
    parser.add_argument("--task", choices=TASK_ORDER + ("all",), default="all")
    parser.add_argument("--result-base", type=Path, default=DEFAULT_RESULT_BASE)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "freeze":
        freeze(args.result_base)
        return 0
    execution_sha = args.execution_sha or git("rev-parse", "HEAD")
    if args.mode == "audit":
        pairs = PAIR_ORDER if args.pair == "all" else (args.pair,)
        tasks = TASK_ORDER if args.task == "all" else (args.task,)
        for pair in pairs:
            for task in tasks:
                run_shard(pair, task, execution_sha, args.result_base)
        return 0
    if args.mode == "finalize":
        finalize(execution_sha, args.result_base)
        return 0
    verify(execution_sha, args.result_base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
