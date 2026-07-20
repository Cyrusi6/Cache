from __future__ import annotations

"""Sealed FPCT-3.7-R1 audit plus non-operative descriptive enrichment."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from fpct_bootstrap import loaded_module, require_active


REPO_ROOT = Path(__file__).resolve().parents[2]
require_active(target=Path(__file__))

PAIR_ORDER = ("tinyllama", "qwen25_0p5b", "llama32_1b", "qwen3_1p7b")
TASK_ORDER = ("ai2-arc", "openbookqa", "mmlu-redux")
SPLIT_ORDER = ("fit", "calibration", "model-selection", "test", "all")
STAGE_ORDER = ("raw_pre_truncation", "retained_after_truncation", "sanitized")
TOP_K = 4
MAX_LENGTH = 1024
RESULT_BASE = Path(
    "local/final_results/fpct_factorized_transport/"
    "fpct_3_7r_certified_support"
)
PROTOCOL = REPO_ROOT / "FPCT_3_7R_IMPORT_PROVENANCE_PROTOCOL.md"
MANIFEST = REPO_ROOT / "recipe/eval_recipe/fpct_3_7r/import_provenance_manifest.json"
DIFF = REPO_ROOT / "recipe/eval_recipe/fpct_3_7r/protocol_diff.json"
FPCT1B_LOCK = REPO_ROOT / "recipe/eval_recipe/fpct_1b/pre_audit_lock.json"
FPCT1B_SPLITS = REPO_ROOT / "recipe/eval_recipe/fpct_1b/content_group_split_manifest.csv"
REPLAY_ROOT_BASE = REPO_ROOT / (
    "local/final_results/fpct_factorized_transport/fpct_3_5p_provenance_replay"
)


class R1AuditError(RuntimeError):
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
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise R1AuditError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


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
    old = loaded_module("fpct_3_7_audit")
    old.load_module = lambda: loaded_module("fpct_1b_audit")
    return old


def _tracked(replay_comparison: Path) -> dict[str, Path]:
    return {
        "protocol": PROTOCOL,
        "manifest": MANIFEST,
        "protocol_diff": DIFF,
        "analysis": Path(__file__).resolve(),
        "bootstrap": REPO_ROOT / "script/runtime/fpct_bootstrap.py",
        "probe": REPO_ROOT / "script/runtime/fpct_probe_target.py",
        "regular_package_init": REPO_ROOT / "rosetta/__init__.py",
        "hostile_tests": REPO_ROOT / "test/test_fpct_sealed_import.py",
        "old_manifest": REPO_ROOT / "recipe/eval_recipe/fpct_3_7/certified_support_manifest.json",
        "old_analysis": REPO_ROOT / "script/analysis/fpct_3_7_certified_support_audit.py",
        "old_protocol": REPO_ROOT / "FPCT_3_5_ALIGNMENT_CORRECTNESS_PROTOCOL.md",
        "fpct_1b_lock": FPCT1B_LOCK,
        "fpct_1b_analysis": REPO_ROOT / "script/analysis/fpct_1b_structural_support_audit.py",
        "fpct_1b_splits": FPCT1B_SPLITS,
        "aligner": REPO_ROOT / "rosetta/model/aligner.py",
        "dataset_adapter": REPO_ROOT / "rosetta/train/dataset_adapters.py",
        "training_entry": REPO_ROOT / "script/train/SFT_train.py",
        "evaluation_entry": REPO_ROOT / "script/evaluation/unified_evaluator.py",
        "prompt_source": REPO_ROOT / "rosetta/utils/evaluate.py",
        "replay_comparison": replay_comparison,
    }


def _validate_splits(audit: Any, samples: Sequence[Any]) -> dict[str, int]:
    return _configured_old().validate_frozen_splits(samples)


def freeze() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise R1AuditError('CUDA_VISIBLE_DEVICES must be explicitly set to ""')
    if _git("status", "--short"):
        raise R1AuditError("freeze requires a clean worktree")
    execution_sha = _git("rev-parse", "HEAD")
    if _git("rev-parse", "@{upstream}") != execution_sha:
        raise R1AuditError("freeze requires local/upstream identity")
    replay_comparison = REPLAY_ROOT_BASE / f"rev_{execution_sha}" / "replay_comparison.json"
    if not replay_comparison.is_file():
        raise R1AuditError("sealed FPCT-3.5P replay comparison is missing")
    replay = _read_json(replay_comparison)
    if replay.get("status") != "PROVENANCE_CONFIRMED":
        raise R1AuditError("FPCT-3.5P is not provenance-confirmed")
    audit = loaded_module("fpct_1b_audit")
    fpct1b_lock = _read_json(FPCT1B_LOCK)
    assets = audit.resolve_assets(audit.DEFAULT_SHARED_ROOT)
    if assets != fpct1b_lock["assets"]:
        raise R1AuditError("tokenizer/dataset assets differ from FPCT-1B lock")
    samples = audit.load_projected_samples(audit.DEFAULT_SHARED_ROOT)
    canonical_input = audit.validate_canonical_samples(samples)
    if canonical_input != fpct1b_lock["canonical_input"]:
        raise R1AuditError("canonical input differs from FPCT-1B lock")
    split_counts = _validate_splits(audit, samples)
    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "datasets", "pyarrow")
        },
    }
    attestation = require_active(target=Path(__file__))
    lock = {
        "schema_version": 1,
        "stage": "FPCT-3.7-R1",
        "status": "FROZEN",
        "execution_sha": execution_sha,
        "protocol_id": _read_json(MANIFEST)["protocol_id"],
        "stable_attestation_sha256": attestation[
            "stable_fingerprint_sha256"
        ],
        "sealed_modules": attestation["mandatory_modules"],
        "tracked": {
            key: {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256(path),
            }
            for key, path in _tracked(replay_comparison).items()
        },
        "thresholds": _read_json(MANIFEST)["readiness"],
        "max_length": MAX_LENGTH,
        "canonical_input": canonical_input,
        "split_counts": split_counts,
        "assets": assets,
        "runtime": runtime,
        "natural_audit_started": False,
        "gpu_authorized": False,
    }
    root = _root(execution_sha)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "pre_audit_lock.json"
    if lock_path.exists() and _read_json(lock_path) != lock:
        raise R1AuditError("existing R1 pre-audit lock differs")
    if not lock_path.exists():
        _write_json(lock_path, lock)
    _write_json(
        root / "controller_state.json",
        {
            "schema_version": 1,
            "execution_sha": execution_sha,
            "state": "FROZEN",
            "completed_shards": [],
            "held_out_test_released": False,
        },
    )
    print(json.dumps({"status": "FROZEN", "execution_sha": execution_sha}))


def _verify_lock(execution_sha: str) -> tuple[Path, dict[str, Any]]:
    if _git("rev-parse", "HEAD") != execution_sha:
        raise R1AuditError("HEAD differs from FPCT-3.7-R1 execution SHA")
    if _git("rev-parse", "@{upstream}") != execution_sha:
        raise R1AuditError("upstream differs from FPCT-3.7-R1 execution SHA")
    if _git("status", "--short"):
        raise R1AuditError("FPCT-3.7-R1 requires a clean worktree")
    root = _root(execution_sha)
    lock = _read_json(root / "pre_audit_lock.json")
    if lock.get("execution_sha") != execution_sha:
        raise R1AuditError("R1 lock execution SHA mismatch")
    current = require_active(target=Path(__file__))
    if current["stable_fingerprint_sha256"] != lock[
        "stable_attestation_sha256"
    ]:
        raise R1AuditError("R1 stable attestation differs from lock")
    for record in lock["tracked"].values():
        if _sha256(REPO_ROOT / record["path"]) != record["sha256"]:
            raise R1AuditError(f"frozen R1 file changed: {record['path']}")
    return root, lock


def _legal(
    indices: Sequence[int], weights: Sequence[float], source_length: int,
    original_source_length: int,
) -> list[tuple[int, float]]:
    legal: list[tuple[int, float]] = []
    seen: set[int] = set()
    for raw_index, raw_weight in zip(indices, weights):
        index, weight = int(raw_index), float(raw_weight)
        if not math.isfinite(weight) or weight < 0:
            raise R1AuditError("nonfinite/negative prior mass")
        if weight <= 0:
            continue
        if index < 0 or index >= original_source_length:
            raise R1AuditError("positive mass on invalid source index")
        if index in seen:
            raise R1AuditError("duplicate legal source index")
        seen.add(index)
        if index < source_length:
            legal.append((index, weight))
    if len(seen) > TOP_K:
        raise R1AuditError("candidate count exceeds top-k")
    total = sum(weight for _index, weight in legal)
    return [(index, weight / total) for index, weight in legal] if total else []


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(float(value) for value in values)

    def q(probability: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * probability
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (
            position - lower
        )

    return {
        "mean": statistics.fmean(ordered),
        "p50": q(0.50),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": ordered[-1],
    }


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left, mean_right = statistics.fmean(left), statistics.fmean(right)
    numerator = sum(
        (x - mean_left) * (y - mean_right) for x, y in zip(left, right)
    )
    denom_left = sum((x - mean_left) ** 2 for x in left)
    denom_right = sum((y - mean_right) ** 2 for y in right)
    if denom_left == 0 or denom_right == 0:
        return None
    return numerator / math.sqrt(denom_left * denom_right)


def _unicode_flags(text: str) -> dict[str, Any]:
    categories = sorted({unicodedata.category(char) for char in text})
    return {
        "unicode_categories": categories,
        "byte_fallback_like": int(
            "<0x" in text or "�" in text or any(ord(char) >= 0x80 for char in text)
        ),
    }


def _enrich_shard(pair: str, task: str, execution_sha: str) -> dict[str, Any]:
    root, lock = _verify_lock(execution_sha)
    old = _configured_old()
    geometry = loaded_module("fpct_3_5_audit")
    audit = loaded_module("fpct_1b_audit")
    fpct1b_lock = _read_json(FPCT1B_LOCK)
    receiver, senders = audit.load_tokenizers(fpct1b_lock)
    raw_aligner = old.make_aligner(receiver, senders[pair], "soft_span_overlap_v2")
    exact_aligner = (
        old.make_aligner(receiver, senders[pair], "exact_identity")
        if pair == "qwen3_1p7b" else None
    )
    samples_all = audit.load_projected_samples(audit.DEFAULT_SHARED_ROOT)
    audit.validate_canonical_samples(samples_all)
    old.validate_frozen_splits(samples_all)
    samples = [sample for sample in samples_all if sample.task == task]

    parent_counts: Counter[tuple[str, str, int]] = Counter()
    sample_counts: Counter[tuple[str, str, int]] = Counter()
    group_sets: defaultdict[tuple[str, str, int], set[str]] = defaultdict(set)
    exposure: dict[str, dict[str, Any]] = {}
    for split in SPLIT_ORDER:
        exposure[split] = {
            "eligible_parents": 0,
            "certified_ambiguous_parents": 0,
            "groups": set(),
            "positive_groups": set(),
            "group_ambiguous_counts": Counter(),
            "a_max": [],
            "entropy": [],
            "n_eff": [],
            "secondary_mass": [],
            "removed_non_top1_mass": 0.0,
        }
    geometry_rows: list[dict[str, Any]] = []
    cluster_sets: defaultdict[str, set[tuple[str, tuple[int, ...]]]] = defaultdict(set)
    arm_digest = hashlib.sha256()
    raw_expansion: list[float] = []
    certified_expansion: list[float] = []
    raw_extra_slots: list[int] = []
    certified_extra_slots: list[int] = []
    density_values: list[float] = []
    sanitizer_checks = Counter()

    for ordinal, sample in enumerate(samples, 1):
        messages = [{"role": "user", "content": audit.prompt_for_sample(sample)}]
        raw = raw_aligner.align_chat_messages_soft(
            messages, add_generation_prompt=True, enable_thinking=False,
            top_k=TOP_K, return_details=True, apply_confidence_control=False,
        )
        target_length = min(len(raw["slm_ids"]), MAX_LENGTH)
        source_length = min(len(raw["llm_ids"]), MAX_LENGTH)
        if exact_aligner is not None:
            corrected = exact_aligner.align_chat_messages_soft(
                messages, add_generation_prompt=True, enable_thinking=False,
                top_k=TOP_K, return_details=True, apply_confidence_control=False,
            )
            certified_mask = [False] * len(corrected["slm_ids"])
            uncertified_mask = [False] * len(corrected["slm_ids"])
            reasons = ["exact_identity"] * len(corrected["slm_ids"])
            if corrected["soft_alignment"].get("fpct_extra_slots") != 0:
                raise R1AuditError("Qwen exact identity created an extra slot")
        else:
            corrected = raw_aligner.sanitize_fpct_soft_alignment(
                raw, target_length=target_length, source_length=source_length
            )
            certified_mask = corrected["soft_alignment"]["fpct_certified_mask"]
            uncertified_mask = corrected["soft_alignment"][
                "fpct_offset_uncertified_mask"
            ]
            reasons = corrected["soft_alignment"]["fpct_certification_reason"]

        stage_counts = {stage: Counter() for stage in STAGE_ORDER}
        sample_certified_ambiguous = 0
        sample_eligible = 0
        raw_extra = certified_extra = 0
        for parent in range(target_length):
            if not corrected["message_mask"][parent]:
                continue
            sample_eligible += 1
            raw_indices = raw["soft_alignment"]["source_indices"][parent]
            raw_weights = raw["soft_alignment"]["source_weights"][parent]
            pre = _legal(raw_indices, raw_weights, len(raw["llm_ids"]), len(raw["llm_ids"]))
            retained = _legal(raw_indices, raw_weights, source_length, len(raw["llm_ids"]))
            corrected_indices = corrected["soft_alignment"]["source_indices"][parent]
            corrected_weights = corrected["soft_alignment"]["source_weights"][parent]
            sanitized = _legal(
                corrected_indices, corrected_weights, source_length,
                len(corrected["llm_ids"]),
            )
            counts = {
                "raw_pre_truncation": len(pre),
                "retained_after_truncation": len(retained),
                "sanitized": len(sanitized),
            }
            if any(value > TOP_K for value in counts.values()):
                raise R1AuditError("m exceeds top-k")
            for stage, value in counts.items():
                stage_counts[stage][value] += 1
            raw_extra += max(len(retained) - 1, 0)
            certified_extra += max(len(sanitized) - 1, 0)
            row_sum = sum(weight for _index, weight in sanitized)
            if sanitized and abs(row_sum - 1.0) > 1e-10:
                raise R1AuditError("sanitized row sum differs from one")
            sanitizer_checks["row_sum_checked"] += 1
            if not sanitized:
                sanitizer_checks["m0_native"] += 1
            if len(retained) == 1 and sanitized != retained:
                raise R1AuditError("m1 changed under sanitizer")
            if len(retained) == 1:
                sanitizer_checks["m1_unchanged"] += 1
            if bool(uncertified_mask[parent]):
                if len(sanitized) != 1 or not retained or sanitized[0][0] != retained[0][0] or sanitized[0][1] != 1.0:
                    raise R1AuditError("uncertified row is not exact slot-0 one-hot")
                sanitizer_checks["uncertified_slot0_one_hot"] += 1
            if bool(certified_mask[parent]) and sanitized != retained:
                raise R1AuditError("certified row changed under sanitizer")
            if pair == "qwen3_1p7b":
                if sanitized != [(parent, 1.0)]:
                    raise R1AuditError("Qwen exact identity index/weight failure")
                if corrected["soft_alignment"]["fallback_mask"][parent]:
                    raise R1AuditError("Qwen exact identity fallback failure")
                sanitizer_checks["qwen_exact_identity_parent"] += 1

            arm_digest.update(
                json.dumps(
                    [sample.sample_key_sha256, parent, corrected_indices, corrected_weights],
                    sort_keys=True, separators=(",", ":"),
                ).encode("utf-8")
            )
            certified_ambiguous = bool(certified_mask[parent] and len(sanitized) >= 2)
            sample_certified_ambiguous += int(certified_ambiguous)
            for split in (sample.split, "all"):
                bucket = exposure[split]
                bucket["eligible_parents"] += 1
                bucket["groups"].add(sample.content_group_sha256)
                if certified_ambiguous:
                    bucket["certified_ambiguous_parents"] += 1
                    bucket["positive_groups"].add(sample.content_group_sha256)
                    bucket["group_ambiguous_counts"][sample.content_group_sha256] += 1
                    probs = [weight for _index, weight in sanitized]
                    a_max = max(probs)
                    entropy = -sum(value * math.log(value) for value in probs)
                    n_eff = 1.0 / sum(value * value for value in probs)
                    bucket["a_max"].append(a_max)
                    bucket["entropy"].append(entropy)
                    bucket["n_eff"].append(n_eff)
                    bucket["secondary_mass"].append(1.0 - a_max)
                if bool(uncertified_mask[parent]) and len(retained) >= 2:
                    bucket["removed_non_top1_mass"] += 1.0 - max(
                        weight for _index, weight in retained
                    )

            if len(retained) >= 2:
                certification = geometry.classify_parent_geometry(
                    raw, parent, retained, exact_control=(pair == "qwen3_1p7b")
                )
                candidate_indices = [index for index, _weight in retained]
                candidate_offsets = [raw["llm_offsets"][index] for index in candidate_indices]
                candidate_texts = [
                    raw["llm_text"][start:end] for start, end in candidate_offsets
                ]
                receiver_offset = raw["slm_offsets"][parent]
                receiver_text = raw["slm_text"][receiver_offset[0]:receiver_offset[1]]
                token_flags = _unicode_flags(receiver_text + "".join(candidate_texts))
                flags = {
                    "receiver_zero_length": int(certification.receiver_interval[1] <= certification.receiver_interval[0]),
                    "receiver_overlap": int(certification.category == "duplicate_or_overlap_receiver_offsets"),
                    "source_zero_length": int(any(end <= start for start, end in certification.candidate_intervals)),
                    "source_duplicate": int(len(set(certification.candidate_intervals)) != len(certification.candidate_intervals)),
                    "source_overlap": int(any(max(a[0], b[0]) < min(a[1], b[1]) for idx, a in enumerate(certification.candidate_intervals) for b in certification.candidate_intervals[idx + 1:])),
                    "coverage_gap": int(bool(certification.intersections) and (min(start for start, _end in certification.intersections) > certification.receiver_interval[0] or max(end for _start, end in certification.intersections) < certification.receiver_interval[1])),
                    "non_monotonic": int(candidate_indices != [value[1] for value in sorted(zip(certification.candidate_intervals, candidate_indices), key=lambda item: item[0])]),
                    "topk_non_exhaustive": int(int(raw["soft_alignment"]["positive_overlap_counts"][parent]) != len(pre)),
                    "truncation_loss": int(len(pre) != len(retained)),
                    "illegal_slot0": int(not retained or retained[0][0] >= source_length),
                }
                cluster_sets[sample.split].add((sample.sample_key_sha256, tuple(candidate_indices)))
                cluster_sets["all"].add((sample.sample_key_sha256, tuple(candidate_indices)))
                geometry_rows.append({
                    "schema_version": 1,
                    "pair": pair,
                    "task": task,
                    "split": sample.split,
                    "subject": sample.subject,
                    "sample_key_sha256": sample.sample_key_sha256,
                    "content_group_sha256": sample.content_group_sha256,
                    "parent_index": parent,
                    "raw_pre_truncation_m": len(pre),
                    "retained_m": len(retained),
                    "sanitized_m": len(sanitized),
                    "certified": int(bool(certified_mask[parent])),
                    "offset_uncertified": int(bool(uncertified_mask[parent])),
                    "primary_reason": reasons[parent],
                    "receiver_span_length": certification.receiver_interval[1] - certification.receiver_interval[0],
                    "candidate_cardinality": len(retained),
                    "source_intersection_lengths": [end - start for start, end in certification.intersections],
                    "coverage": sum(end - start for start, end in certification.intersections),
                    "candidate_indices": candidate_indices,
                    "candidate_token_ids": [int(raw["llm_ids"][index]) for index in candidate_indices],
                    "candidate_token_strings": candidate_texts,
                    "unicode_categories": token_flags["unicode_categories"],
                    "byte_fallback_like": token_flags["byte_fallback_like"],
                    **flags,
                })

        for split in (sample.split, "all"):
            for stage, counts in stage_counts.items():
                for m in range(5):
                    parent_counts[(split, stage, m)] += counts[m]
                    if counts[m] > 0:
                        sample_counts[(split, stage, m)] += 1
                        group_sets[(split, stage, m)].add(sample.content_group_sha256)
        native_slots = target_length
        raw_expansion.append((native_slots + raw_extra) / native_slots)
        certified_expansion.append((native_slots + certified_extra) / native_slots)
        raw_extra_slots.append(raw_extra)
        certified_extra_slots.append(certified_extra)
        density_values.append(sample_certified_ambiguous / sample_eligible if sample_eligible else 0.0)
        if ordinal % 100 == 0 or ordinal == len(samples):
            print(json.dumps({"status": "R1_DESCRIPTIVE_PROGRESS", "pair": pair, "task": task, "done": ordinal, "total": len(samples)}, sort_keys=True), flush=True)

    transition_rows = []
    for split in SPLIT_ORDER:
        for stage in STAGE_ORDER:
            for m in range(5):
                transition_rows.append({
                    "pair": pair,
                    "task": task,
                    "split": split,
                    "stage": stage,
                    "m": m,
                    "parent_count": parent_counts[(split, stage, m)],
                    "sample_count": sample_counts[(split, stage, m)],
                    "distinct_content_group_count": len(group_sets[(split, stage, m)]),
                })
    exposure_rows = []
    for split in SPLIT_ORDER:
        bucket = exposure[split]
        eligible = int(bucket["eligible_parents"])
        ambiguous = int(bucket["certified_ambiguous_parents"])
        group_counts = list(bucket["group_ambiguous_counts"].values())
        exposure_rows.append({
            "pair": pair,
            "task": task,
            "split": split,
            "group_count": len(bucket["groups"]),
            "certified_positive_group_count": len(bucket["positive_groups"]),
            "certified_positive_group_rate": len(bucket["positive_groups"]) / len(bucket["groups"]) if bucket["groups"] else 0.0,
            "eligible_parent_count": eligible,
            "certified_ambiguous_parent_count": ambiguous,
            "certified_ambiguous_parent_density": ambiguous / eligible if eligible else 0.0,
            "ambiguous_parents_per_positive_group": _quantiles(group_counts),
            "sum_secondary_mass": sum(bucket["secondary_mass"]),
            "a_max": _quantiles(bucket["a_max"]),
            "entropy": _quantiles(bucket["entropy"]),
            "n_eff": _quantiles(bucket["n_eff"]),
            "sanitizer_removed_non_top1_prior_mass": bucket["removed_non_top1_mass"],
            "event_count": ambiguous,
            "cluster_count": len(cluster_sets[split]),
        })

    receiver_config = json.loads(old.RECEIVER_CONFIG_PATH.read_text(encoding="utf-8"))
    layers = int(receiver_config["num_hidden_layers"])
    hq = int(receiver_config["num_attention_heads"])
    hkv = int(receiver_config["num_key_value_heads"])
    head_dim = int(receiver_config.get("head_dim", receiver_config["hidden_size"] // hq))
    kv_atom_bytes = 2 * hkv * head_dim * 2
    resource = {
        "raw_expansion": _quantiles(raw_expansion),
        "certified_expansion": _quantiles(certified_expansion),
        "raw_attention_flop_ratio": _quantiles(raw_expansion),
        "certified_attention_flop_ratio": _quantiles(certified_expansion),
        "raw_sidecar_bytes_per_layer": _quantiles([value * kv_atom_bytes for value in raw_extra_slots]),
        "certified_sidecar_bytes_per_layer": _quantiles([value * kv_atom_bytes for value in certified_extra_slots]),
        "raw_all_layer_cache_bytes": _quantiles([value * kv_atom_bytes * layers for value in raw_extra_slots]),
        "certified_all_layer_cache_bytes": _quantiles([value * kv_atom_bytes * layers for value in certified_extra_slots]),
        "raw_expansion_x_certified_density_pearson": _pearson(raw_expansion, density_values),
        "certified_expansion_x_certified_density_pearson": _pearson(certified_expansion, density_values),
        "receiver_config": {
            "path": str(old.RECEIVER_CONFIG_PATH),
            "sha256": _sha256(old.RECEIVER_CONFIG_PATH),
            "layers": layers,
            "hkv": hkv,
            "head_dim": head_dim,
            "kv_atom_bytes_per_layer": kv_atom_bytes,
        },
    }
    digest = arm_digest.hexdigest()
    result = {
        "schema_version": 1,
        "stage": "FPCT-3.7-R1",
        "execution_sha": execution_sha,
        "pair": pair,
        "task": task,
        "stable_attestation_sha256": lock["stable_attestation_sha256"],
        "transition_rows": transition_rows,
        "factorization_exposure": exposure_rows,
        "resource": resource,
        "sanitizer_integrity": {
            "checks": dict(sorted(sanitizer_checks.items())),
            "c_pre_alignment_input_sha256": digest,
            "c_post_alignment_input_sha256": digest,
            "f_alignment_input_sha256": digest,
            "three_arm_input_hash_equal": True,
        },
    }
    directory = root / "shards" / f"{pair}__{task}"
    geometry_path = directory / "r1_parent_geometry.csv"
    geometry_columns = tuple(geometry_rows[0]) if geometry_rows else (
        "schema_version", "pair", "task", "split", "subject",
        "sample_key_sha256", "content_group_sha256", "parent_index",
    )
    _write_csv(geometry_path, geometry_columns, geometry_rows)
    result["geometry_artifact"] = {
        "path": str(geometry_path),
        "rows": len(geometry_rows),
        "sha256": _sha256(geometry_path),
        "bytes": geometry_path.stat().st_size,
    }
    descriptive_path = directory / "r1_descriptive.json"
    _write_json(descriptive_path, result)
    _write_json(
        directory / "sealed_shard_manifest.json",
        {
            "schema_version": 1,
            "execution_sha": execution_sha,
            "pair": pair,
            "task": task,
            "stable_attestation_sha256": lock["stable_attestation_sha256"],
            "delegate_manifest_sha256": _sha256(directory / "manifest.json"),
            "descriptive_sha256": _sha256(descriptive_path),
            "geometry_sha256": _sha256(geometry_path),
        },
    )
    return result


def run_shard(pair: str, task: str, execution_sha: str) -> None:
    _verify_lock(execution_sha)
    old = _configured_old()
    old.run_shard(pair, task, execution_sha, RESULT_BASE)
    directory = _root(execution_sha) / "shards" / f"{pair}__{task}"
    sealed = directory / "sealed_shard_manifest.json"
    if sealed.exists():
        existing = _read_json(sealed)
        if existing.get("execution_sha") != execution_sha:
            raise R1AuditError("sealed shard manifest execution mismatch")
        print(json.dumps({"status": "R1_RESUME_SKIP", "pair": pair, "task": task}))
        return
    _enrich_shard(pair, task, execution_sha)


def finalize(execution_sha: str) -> dict[str, Any]:
    root, lock = _verify_lock(execution_sha)
    old = _configured_old()
    old.finalize(execution_sha, RESULT_BASE)
    old.verify(execution_sha, RESULT_BASE)
    summaries = []
    for pair in PAIR_ORDER:
        for task in TASK_ORDER:
            directory = root / "shards" / f"{pair}__{task}"
            sealed = _read_json(directory / "sealed_shard_manifest.json")
            if sealed["stable_attestation_sha256"] != lock[
                "stable_attestation_sha256"
            ]:
                raise R1AuditError("shard attestation differs from lock")
            descriptive_path = directory / "r1_descriptive.json"
            if _sha256(descriptive_path) != sealed["descriptive_sha256"]:
                raise R1AuditError("descriptive shard hash mismatch")
            summaries.append({
                "pair": pair,
                "task": task,
                "descriptive_path": str(descriptive_path),
                "descriptive_sha256": sealed["descriptive_sha256"],
                "geometry_sha256": sealed["geometry_sha256"],
                "stable_attestation_sha256": sealed["stable_attestation_sha256"],
            })
    primary = _read_json(root / "certified_support_summary.json")
    if primary.get("support_gate_passed"):
        status = "SINGLE_PAIR_PILOT_READY"
    else:
        status = "NO_GO_GPU_CURRENT_CERTIFIER"
    result = {
        "schema_version": 1,
        "stage": "FPCT-3.7-R1",
        "status": status,
        "execution_sha": execution_sha,
        "stable_attestation_sha256": lock["stable_attestation_sha256"],
        "primary_certified_support_summary": primary,
        "descriptive_shards": summaries,
        "gpu_authorized": False,
        "claim_boundary": (
            "Current conservative character-partition certifier support only; "
            "not a mathematical or universal aligner verdict."
        ),
    }
    _write_json(root / "r1_result.json", result)
    state = _read_json(root / "controller_state.json")
    state["state"] = status
    state["completed_shards"] = [
        f"{pair}/{task}" for pair in PAIR_ORDER for task in TASK_ORDER
    ]
    state["support_gate_passed"] = bool(primary.get("support_gate_passed"))
    _write_json(root / "controller_state.json", state)
    print(json.dumps({"status": status, "execution_sha": execution_sha}))
    return result


def verify(execution_sha: str) -> None:
    root, lock = _verify_lock(execution_sha)
    old = _configured_old()
    old.verify(execution_sha, RESULT_BASE)
    for pair in PAIR_ORDER:
        for task in TASK_ORDER:
            directory = root / "shards" / f"{pair}__{task}"
            sealed = _read_json(directory / "sealed_shard_manifest.json")
            if sealed["stable_attestation_sha256"] != lock[
                "stable_attestation_sha256"
            ]:
                raise R1AuditError("sealed shard attestation mismatch")
            for name, key in (
                ("r1_descriptive.json", "descriptive_sha256"),
                ("r1_parent_geometry.csv", "geometry_sha256"),
            ):
                if _sha256(directory / name) != sealed[key]:
                    raise R1AuditError(f"sealed shard artifact changed: {name}")
    print(json.dumps({"status": "VERIFIED", "execution_sha": execution_sha}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "audit", "finalize", "verify"))
    parser.add_argument("--execution-sha")
    parser.add_argument("--pair", choices=PAIR_ORDER + ("all",), default="all")
    parser.add_argument("--task", choices=TASK_ORDER + ("all",), default="all")
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze()
        return 0
    execution_sha = args.execution_sha or _git("rev-parse", "HEAD")
    if args.mode == "audit":
        pairs = PAIR_ORDER if args.pair == "all" else (args.pair,)
        tasks = TASK_ORDER if args.task == "all" else (args.task,)
        for pair in pairs:
            for task in tasks:
                run_shard(pair, task, execution_sha)
        return 0
    if args.mode == "finalize":
        finalize(execution_sha)
        return 0
    verify(execution_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
