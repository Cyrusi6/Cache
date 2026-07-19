from __future__ import annotations

"""Pre-registered CPU forensic for FPCT alignment correctness.

The ``freeze`` mode is run from the clean pre-data commit.  Natural tokenizer
execution is forbidden before that lock exists.  Per-parent ledgers are local
only and contain no labels, predictions or correctness fields.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


REPO_ROOT = Path(__file__).resolve().parents[2]
STARTING_HEAD = "d296a18be9cc3b0dce3c07f4c2d7244145f2e3ac"
TOP_K = 4
TASK_ORDER = ("ai2-arc", "openbookqa", "mmlu-redux")
PROTOCOL_PATH = REPO_ROOT / "FPCT_3_5_ALIGNMENT_CORRECTNESS_PROTOCOL.md"
MANIFEST_PATH = (
    REPO_ROOT
    / "recipe/eval_recipe/fpct_3_5/alignment_correctness_manifest.json"
)
TEST_PATH = REPO_ROOT / "test/test_fpct_3_5_alignment_correctness.py"
SCRIPT_PATH = Path(__file__).resolve()
FPCT1B_LOCK = REPO_ROOT / "recipe/eval_recipe/fpct_1b/pre_audit_lock.json"
FPCT1B_SPLIT = (
    REPO_ROOT / "recipe/eval_recipe/fpct_1b/content_group_split_manifest.csv"
)
DEFAULT_RESULT_BASE = Path(
    "local/final_results/fpct_factorized_transport/"
    "fpct_3_5_alignment_correctness"
)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)
ANOMALY_CATEGORIES = (
    "tokenizer_or_pair_path_mixup",
    "rendered_text_difference",
    "token_id_difference",
    "offset_difference",
    "zero_length_offset",
    "duplicate_or_overlap_receiver_offsets",
    "exact_duplicate_source_offsets",
    "partial_overlap_source_offsets",
    "candidate_missing_identity_index",
    "unexplained_other",
)


class AlignmentCorrectnessError(RuntimeError):
    pass


@dataclass(frozen=True)
class IdentitySnapshot:
    rendered_text: str
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    content_spans: tuple[tuple[int, int], ...]
    message_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Certification:
    certified: bool
    category: str
    reason: str
    receiver_interval: tuple[int, int]
    candidate_intervals: tuple[tuple[int, int], ...]
    intersections: tuple[tuple[int, int], ...]


def ensure_cpu_only() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise AlignmentCorrectnessError(
            'CUDA_VISIBLE_DEVICES must be explicitly set to ""'
        )


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AlignmentCorrectnessError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def atomic_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> int:
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
                    if isinstance((value := row.get(column)), (list, tuple, dict))
                    else value
                    for column in columns
                }
            )
            count += 1
    temp.replace(path)
    return count


def _stable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _stable(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_stable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def tokenizer_file_fingerprint(directory: Path) -> dict[str, Any]:
    files = {
        name: sha256_file(directory / name)
        for name in TOKENIZER_FILES
        if (directory / name).is_file()
    }
    if not files:
        raise AlignmentCorrectnessError(f"no tokenizer files in {directory}")
    return {"files": files, "sha256": sha256_json(files)}


def tokenizer_runtime_fingerprint(tokenizer: Any, directory: Path) -> dict[str, Any]:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not hasattr(backend, "to_str"):
        raise AlignmentCorrectnessError("exact identity requires a fast tokenizer backend")
    payload = {
        "tokenizer_class": tokenizer.__class__.__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "backend": json.loads(backend.to_str()),
        "vocab": sorted(
            (str(token), int(token_id))
            for token, token_id in tokenizer.get_vocab().items()
        ),
        "added_vocab": sorted(
            (str(token), int(token_id))
            for token, token_id in tokenizer.get_added_vocab().items()
        ),
        "special_tokens_map": _stable(tokenizer.special_tokens_map_extended),
        "all_special_tokens": list(tokenizer.all_special_tokens),
        "all_special_ids": [int(token_id) for token_id in tokenizer.all_special_ids],
        "chat_template": tokenizer.chat_template,
        "tokenizer_files": tokenizer_file_fingerprint(directory),
    }
    return {"payload": payload, "sha256": sha256_json(payload)}


def identity_snapshot(
    aligner: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> IdentitySnapshot:
    text, ids, offsets = aligner._apply_chat_template_to_ids(
        tokenizer,
        messages,
        add_generation_prompt,
        False,
        False,
    )
    if offsets is None:
        raise AlignmentCorrectnessError("exact identity requires offset mappings")
    spans = aligner._compute_content_spans(text, messages)
    expected_contents = [
        message["content"]
        for message in messages
        if isinstance(message.get("content"), str) and message["content"]
    ]
    if len(spans) != len(expected_contents):
        raise AlignmentCorrectnessError("content span count mismatch")
    for span, content in zip(spans, expected_contents):
        if text[span[0]:span[1]] != content:
            raise AlignmentCorrectnessError("content span does not recover message content")
    ranges = aligner._spans_to_token_ranges(offsets, spans)
    return IdentitySnapshot(
        rendered_text=text,
        input_ids=tuple(int(value) for value in ids),
        offsets=tuple((int(start), int(end)) for start, end in offsets),
        content_spans=tuple((int(start), int(end)) for start, end in spans),
        message_ranges=tuple((int(start), int(end)) for start, end in ranges),
    )


def assert_exact_runtime_identity(
    receiver: IdentitySnapshot,
    sender: IdentitySnapshot,
    receiver_fingerprint: Mapping[str, Any],
    sender_fingerprint: Mapping[str, Any],
) -> None:
    if receiver_fingerprint != sender_fingerprint:
        raise AlignmentCorrectnessError("tokenizer_or_pair_path_mixup")
    comparisons = (
        ("rendered_text_difference", receiver.rendered_text, sender.rendered_text),
        ("token_id_difference", receiver.input_ids, sender.input_ids),
        ("offset_difference", receiver.offsets, sender.offsets),
        ("content_span_difference", receiver.content_spans, sender.content_spans),
        ("message_range_difference", receiver.message_ranges, sender.message_ranges),
    )
    for name, left, right in comparisons:
        if left != right:
            raise AlignmentCorrectnessError(name)


def exact_identity_alignment(
    snapshot: IdentitySnapshot,
    *,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    message_mask = [False] * len(snapshot.input_ids)
    for start, end in snapshot.message_ranges:
        for index in range(start, end):
            message_mask[index] = True
    indices = [[-1] * top_k for _ in snapshot.input_ids]
    weights = [[0.0] * top_k for _ in snapshot.input_ids]
    fallback = [False] * len(snapshot.input_ids)
    for index, eligible in enumerate(message_mask):
        if eligible:
            indices[index][0] = index
            weights[index][0] = 1.0
    return {
        "source_indices": indices,
        "source_weights": weights,
        "fallback_mask": fallback,
        "message_mask": message_mask,
        "extra_slots": 0,
    }


def sanitize_candidate_row(
    indices: Sequence[int],
    weights: Sequence[float],
    *,
    certified: bool,
) -> tuple[list[int], list[float]]:
    legal = [
        (int(index), float(weight))
        for index, weight in zip(indices, weights)
        if int(index) >= 0 and math.isfinite(float(weight)) and float(weight) > 0
    ]
    if certified or len(legal) <= 1:
        return list(map(int, indices)), list(map(float, weights))
    if (
        not indices
        or int(indices[0]) < 0
        or not weights
        or not math.isfinite(float(weights[0]))
        or float(weights[0]) <= 0
    ):
        raise AlignmentCorrectnessError(
            "uncertified row has no legal positive-mass slot-0 anchor"
        )
    anchor = int(indices[0])
    return [anchor] + [-1] * (len(indices) - 1), [1.0] + [0.0] * (
        len(weights) - 1
    )


def relative_interval(
    raw_interval: tuple[int, int], content_span: tuple[int, int]
) -> tuple[int, int]:
    start = max(raw_interval[0], content_span[0]) - content_span[0]
    end = min(raw_interval[1], content_span[1]) - content_span[0]
    return int(start), int(end)


def interval_overlap(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def certify_one_to_many(
    *,
    receiver_interval: tuple[int, int],
    other_receiver_intervals: Sequence[tuple[int, int]],
    candidate_indices: Sequence[int],
    candidate_intervals: Sequence[tuple[int, int]],
    require_identity_index: int | None = None,
) -> Certification:
    receiver = tuple(map(int, receiver_interval))
    candidates = tuple(tuple(map(int, interval)) for interval in candidate_intervals)
    empty = Certification(False, "unexplained_other", "not evaluated", receiver, candidates, ())
    if receiver[1] <= receiver[0] or any(end <= start for start, end in candidates):
        return Certification(False, "zero_length_offset", "zero-length clipped interval", receiver, candidates, ())
    if any(interval_overlap(receiver, tuple(other)) > 0 for other in other_receiver_intervals):
        return Certification(False, "duplicate_or_overlap_receiver_offsets", "receiver interval overlaps another eligible receiver interval", receiver, candidates, ())
    if len(set(map(int, candidate_indices))) != len(candidate_indices):
        return Certification(False, "tokenizer_or_pair_path_mixup", "duplicate legal source index", receiver, candidates, ())
    if len(set(candidates)) != len(candidates):
        return Certification(False, "exact_duplicate_source_offsets", "candidate source intervals are exact duplicates", receiver, candidates, ())
    ordered_candidates = sorted(zip(candidates, map(int, candidate_indices)))
    for (left, _), (right, _) in zip(ordered_candidates, ordered_candidates[1:]):
        if interval_overlap(left, right) > 0:
            return Certification(False, "partial_overlap_source_offsets", "candidate source intervals overlap", receiver, candidates, ())
    if require_identity_index is not None and require_identity_index not in set(map(int, candidate_indices)):
        return Certification(False, "candidate_missing_identity_index", "identity source index absent", receiver, candidates, ())
    intersections = tuple(
        (max(receiver[0], start), min(receiver[1], end))
        for start, end in candidates
    )
    if any(end <= start for start, end in intersections):
        return Certification(False, "unexplained_other", "candidate does not intersect receiver interval", receiver, candidates, intersections)
    ordered_intersections = sorted(intersections)
    for left, right in zip(ordered_intersections, ordered_intersections[1:]):
        if interval_overlap(left, right) > 0:
            return Certification(False, "partial_overlap_source_offsets", "candidate intersections overlap", receiver, candidates, intersections)
    cursor = receiver[0]
    for start, end in ordered_intersections:
        if start != cursor:
            return Certification(False, "unexplained_other", "candidate intersections do not exactly cover receiver interval", receiver, candidates, intersections)
        cursor = end
    if cursor != receiver[1]:
        return Certification(False, "unexplained_other", "candidate intersections leave uncovered receiver span", receiver, candidates, intersections)
    source_index_order = [index for _interval, index in ordered_candidates]
    if source_index_order != sorted(source_index_order):
        return Certification(False, "unexplained_other", "source indices are not monotonic by span", receiver, candidates, intersections)
    if not candidates:
        return empty
    return Certification(True, "certified_one_to_many", "exact disjoint partition", receiver, candidates, intersections)


def escaped_text(value: str) -> str:
    return value.encode("unicode_escape").decode("ascii")


def unicode_records(value: str) -> list[dict[str, str]]:
    return [
        {
            "codepoint": f"U+{ord(char):04X}",
            "name": unicodedata.name(char, "UNNAMED"),
            "category": unicodedata.category(char),
        }
        for char in value
    ]


def legal_candidates(indices: Sequence[int], weights: Sequence[float]) -> list[tuple[int, float]]:
    legal: list[tuple[int, float]] = []
    for index, weight in zip(indices, weights):
        index = int(index)
        weight = float(weight)
        if index >= 0 and math.isfinite(weight) and weight > 0:
            legal.append((index, weight))
    return legal


def _message_section(details: Mapping[str, Any], parent_index: int) -> tuple[int, Mapping[str, Any]]:
    message_ordinal = 0
    for section in details["sections"]:
        if section["type"] != "message":
            continue
        start, end = section["slm_range"]
        if start <= parent_index < end:
            return message_ordinal, section
        message_ordinal += 1
    raise AlignmentCorrectnessError(f"eligible parent outside message section: {parent_index}")


def classify_parent_geometry(
    details: Mapping[str, Any],
    parent_index: int,
    legal: Sequence[tuple[int, float]],
    *,
    exact_control: bool,
) -> Certification:
    ordinal, section = _message_section(details, parent_index)
    receiver_span = tuple(details["content_spans_slm"][ordinal])
    source_span = tuple(details["content_spans_llm"][ordinal])
    receiver_range = tuple(section["slm_range"])
    receiver_interval = relative_interval(
        tuple(details["slm_offsets"][parent_index]), receiver_span
    )
    others = [
        relative_interval(tuple(details["slm_offsets"][index]), receiver_span)
        for index in range(receiver_range[0], receiver_range[1])
        if index != parent_index
        and details["slm_offsets"][index][1] > details["slm_offsets"][index][0]
    ]
    candidate_indices = [index for index, _weight in legal]
    candidate_intervals = [
        relative_interval(tuple(details["llm_offsets"][index]), source_span)
        for index in candidate_indices
    ]
    certification = certify_one_to_many(
        receiver_interval=receiver_interval,
        other_receiver_intervals=others,
        candidate_indices=candidate_indices,
        candidate_intervals=candidate_intervals,
        require_identity_index=parent_index if exact_control else None,
    )
    positive_overlap_count = int(
        details["soft_alignment"]["positive_overlap_counts"][parent_index]
    )
    if positive_overlap_count != len(legal):
        return Certification(
            False,
            "unexplained_other",
            "retained top-k candidates do not exhaust positive-overlap source tokens",
            certification.receiver_interval,
            certification.candidate_intervals,
            certification.intersections,
        )
    return certification


def set_metrics(left: set[Any], right: set[Any]) -> dict[str, Any]:
    intersection = left & right
    union = left | right
    return {
        "equal": left == right,
        "left_count": len(left),
        "right_count": len(right),
        "intersection_count": len(intersection),
        "left_only_count": len(left - right),
        "right_only_count": len(right - left),
        "union_count": len(union),
        "jaccard": 1.0 if not union else len(intersection) / len(union),
        "left_sha256": sha256_json(sorted(map(str, left))),
        "right_sha256": sha256_json(sorted(map(str, right))),
        "intersection_sha256": sha256_json(sorted(map(str, intersection))),
    }


def load_fpct1b_module() -> Any:
    path = REPO_ROOT / "script/analysis/fpct_1b_structural_support_audit.py"
    spec = importlib.util.spec_from_file_location("fpct_1b_for_fpct35", path)
    if spec is None or spec.loader is None:
        raise AlignmentCorrectnessError("cannot load FPCT-1B audit module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def result_root(execution_sha: str, result_base: Path) -> Path:
    return REPO_ROOT / result_base / f"rev_{execution_sha}"


def freeze(result_base: Path) -> None:
    ensure_cpu_only()
    if git("status", "--short"):
        raise AlignmentCorrectnessError("freeze requires a clean worktree")
    execution_sha = git("rev-parse", "HEAD")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", STARTING_HEAD, execution_sha],
        cwd=REPO_ROOT,
        check=True,
    )
    manifest = read_json(MANIFEST_PATH)
    tracked = {
        "protocol": PROTOCOL_PATH,
        "manifest": MANIFEST_PATH,
        "analysis": SCRIPT_PATH,
        "tests": TEST_PATH,
        "fpct_1b_lock": FPCT1B_LOCK,
        "fpct_1b_split": FPCT1B_SPLIT,
        "aligner": REPO_ROOT / "rosetta/model/aligner.py",
        "prompt_builder": REPO_ROOT / "rosetta/utils/evaluate.py",
    }
    lock = {
        "schema_version": 1,
        "stage": "FPCT-3.5",
        "status": "PRE_DATA_FROZEN",
        "execution_sha": execution_sha,
        "starting_head": STARTING_HEAD,
        "protocol_id": manifest["protocol_id"],
        "tracked": {
            name: {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
            }
            for name, path in tracked.items()
        },
        "natural_forensic_started": False,
        "gpu_authorized": False,
    }
    root = result_root(execution_sha, result_base)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "pre_data_lock.json"
    if path.exists():
        existing = read_json(path)
        if existing != lock:
            raise AlignmentCorrectnessError("pre-data lock already exists with different content")
    else:
        atomic_json(path, lock)
    atomic_json(root / "controller_state.json", {
        "schema_version": 1,
        "stage": "FPCT-3.5",
        "execution_sha": execution_sha,
        "state": "PRE_DATA_FROZEN",
        "completed_tasks": [],
        "held_out_test_released": False,
    })
    print(json.dumps({"status": "FROZEN", "execution_sha": execution_sha, "lock_sha256": sha256_file(path)}, sort_keys=True))


def verify_lock(execution_sha: str, result_base: Path) -> tuple[Path, dict[str, Any]]:
    root = result_root(execution_sha, result_base)
    lock = read_json(root / "pre_data_lock.json")
    if lock.get("execution_sha") != execution_sha:
        raise AlignmentCorrectnessError("execution SHA does not match pre-data lock")
    for record in lock["tracked"].values():
        path = REPO_ROOT / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise AlignmentCorrectnessError(f"frozen file changed: {record['path']}")
    return root, lock


LEDGER_COLUMNS = (
    "schema_version", "pair", "task", "split", "subject",
    "sample_key_sha256", "content_group_sha256", "parent_index",
    "receiver_token_id", "receiver_token_text", "receiver_offset",
    "receiver_relative_interval", "receiver_unicode", "candidate_indices",
    "candidate_token_ids", "candidate_token_texts", "candidate_offsets",
    "candidate_relative_intervals", "candidate_weights",
    "identity_candidate_present", "raw_m", "certified",
    "anomaly_category", "certification_reason", "sender_directory",
    "sender_name_or_path", "sender_file_fingerprint",
    "sender_runtime_fingerprint",
)

COMPARISON_COLUMNS = (
    "schema_version", "task", "split", "sample_key_sha256",
    "content_group_sha256", "parent_index", "qwen3_positive",
    "qwen25_positive", "qwen3_offset_signature",
    "qwen25_offset_signature", "qwen3_candidate_ids",
    "qwen25_candidate_ids",
)


def _row_signature(details: Mapping[str, Any], parent_index: int, legal: Sequence[tuple[int, float]]) -> tuple[Any, ...]:
    ordinal, _section = _message_section(details, parent_index)
    receiver_span = tuple(details["content_spans_slm"][ordinal])
    source_span = tuple(details["content_spans_llm"][ordinal])
    receiver = relative_interval(tuple(details["slm_offsets"][parent_index]), receiver_span)
    candidates = tuple(
        relative_interval(tuple(details["llm_offsets"][index]), source_span)
        for index, _weight in legal
    )
    return receiver, candidates


def _candidate_ids(details: Mapping[str, Any], legal: Sequence[tuple[int, float]]) -> tuple[int, ...]:
    return tuple(int(details["llm_ids"][index]) for index, _weight in legal)


def run_forensic_task(task: str, execution_sha: str, result_base: Path) -> dict[str, Any]:
    ensure_cpu_only()
    root, _lock = verify_lock(execution_sha, result_base)
    shard_dir = root / "forensic" / task
    manifest_path = shard_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("status") == "complete" and manifest.get("execution_sha") == execution_sha:
            print(json.dumps({"status": "RESUME_SKIP", "task": task}, sort_keys=True))
            return manifest
        raise AlignmentCorrectnessError(f"partial forensic shard exists: {manifest_path}")
    if shard_dir.exists() and any(shard_dir.iterdir()):
        raise AlignmentCorrectnessError(f"partial forensic shard exists: {shard_dir}")

    audit = load_fpct1b_module()
    lock = read_json(FPCT1B_LOCK)
    receiver, senders = audit.load_tokenizers(lock)
    qwen3 = senders["qwen3_1p7b"]
    qwen25 = senders["qwen25_0p5b"]
    specs = audit.pair_specs(audit.DEFAULT_SHARED_ROOT, lock)
    from rosetta.model.aligner import TokenAligner

    def make_aligner(sender: Any) -> Any:
        return TokenAligner(
            receiver, sender, strategy="soft_span_overlap_v2",
            soft_alignment_score_mode="uniform",
            soft_alignment_boundary_bonus=0.5,
            soft_alignment_boundary_tolerance=1,
            soft_alignment_min_weight=0.0,
            soft_alignment_confidence_mode="none",
            soft_alignment_reweight_mode="none",
            soft_alignment_candidate_window=0,
            verbose=False,
        )

    qwen3_aligner = make_aligner(qwen3)
    qwen25_aligner = make_aligner(qwen25)
    receiver_dir = Path(lock["assets"]["tokenizers"]["receiver_qwen3_0p6b"]["directory"])
    qwen3_dir = specs["qwen3_1p7b"].tokenizer_dir
    qwen25_dir = specs["qwen25_0p5b"].tokenizer_dir
    receiver_fp = tokenizer_runtime_fingerprint(receiver, receiver_dir)
    qwen3_fp = tokenizer_runtime_fingerprint(qwen3, qwen3_dir)
    qwen25_fp = tokenizer_runtime_fingerprint(qwen25, qwen25_dir)
    if receiver_fp != qwen3_fp:
        raise AlignmentCorrectnessError("tokenizer_or_pair_path_mixup")

    samples = [sample for sample in audit.load_projected_samples(audit.DEFAULT_SHARED_ROOT) if sample.task == task]
    audit.validate_canonical_samples(audit.load_projected_samples(audit.DEFAULT_SHARED_ROOT))
    ledger_rows: list[dict[str, Any]] = []
    comparison_by_parent: dict[tuple[str, int], dict[str, Any]] = {}
    qwen3_groups: set[str] = set()
    qwen25_groups: set[str] = set()
    qwen3_parents: set[tuple[str, int]] = set()
    qwen25_parents: set[tuple[str, int]] = set()
    qwen3_offsets: set[str] = set()
    qwen25_offsets: set[str] = set()
    qwen3_ids: set[str] = set()
    qwen25_ids: set[str] = set()
    category_counts: Counter[str] = Counter()
    unicode_categories: Counter[str] = Counter()
    subjects: Counter[str] = Counter()
    fit_cal_groups: set[str] = set()
    fit_cal_m2 = 0
    identity_samples = 0

    for ordinal, sample in enumerate(samples, 1):
        prompt = audit.prompt_for_sample(sample)
        messages = [{"role": "user", "content": prompt}]
        receiver_snapshot = identity_snapshot(qwen3_aligner, receiver, messages)
        sender_snapshot = identity_snapshot(qwen3_aligner, qwen3, messages)
        assert_exact_runtime_identity(receiver_snapshot, sender_snapshot, receiver_fp, qwen3_fp)
        identity_samples += 1
        qwen3_result = qwen3_aligner.align_chat_messages_soft(
            messages, add_generation_prompt=True, enable_thinking=False,
            top_k=TOP_K, return_details=True, apply_confidence_control=False,
        )
        qwen25_result = qwen25_aligner.align_chat_messages_soft(
            messages, add_generation_prompt=True, enable_thinking=False,
            top_k=TOP_K, return_details=True, apply_confidence_control=False,
        )

        for pair_name, result in (("qwen3", qwen3_result), ("qwen25", qwen25_result)):
            alignment = result["soft_alignment"]
            for parent_index, eligible in enumerate(result["message_mask"]):
                if not eligible or result["slm_padding_mask"][parent_index]:
                    continue
                legal = legal_candidates(
                    alignment["source_indices"][parent_index],
                    alignment["source_weights"][parent_index],
                )
                if len(legal) < 2:
                    continue
                key = (sample.sample_key_sha256, parent_index)
                signature = _row_signature(result, parent_index, legal)
                candidate_ids = _candidate_ids(result, legal)
                record = comparison_by_parent.setdefault(key, {
                    "schema_version": 1, "task": task, "split": sample.split,
                    "sample_key_sha256": sample.sample_key_sha256,
                    "content_group_sha256": sample.content_group_sha256,
                    "parent_index": parent_index,
                    "qwen3_positive": 0, "qwen25_positive": 0,
                    "qwen3_offset_signature": "", "qwen25_offset_signature": "",
                    "qwen3_candidate_ids": "", "qwen25_candidate_ids": "",
                })
                record[f"{pair_name}_positive"] = 1
                record[f"{pair_name}_offset_signature"] = json.dumps(signature)
                record[f"{pair_name}_candidate_ids"] = json.dumps(candidate_ids)
                signature_key = f"{sample.sample_key_sha256}:{parent_index}:{signature}"
                ids_key = f"{sample.sample_key_sha256}:{parent_index}:{candidate_ids}"
                if pair_name == "qwen3":
                    qwen3_groups.add(sample.content_group_sha256)
                    qwen3_parents.add(key)
                    qwen3_offsets.add(signature_key)
                    qwen3_ids.add(ids_key)
                else:
                    qwen25_groups.add(sample.content_group_sha256)
                    qwen25_parents.add(key)
                    qwen25_offsets.add(signature_key)
                    qwen25_ids.add(ids_key)

        alignment = qwen3_result["soft_alignment"]
        for parent_index, eligible in enumerate(qwen3_result["message_mask"]):
            if not eligible or qwen3_result["slm_padding_mask"][parent_index]:
                continue
            legal = legal_candidates(
                alignment["source_indices"][parent_index],
                alignment["source_weights"][parent_index],
            )
            if len(legal) < 2:
                continue
            certification = classify_parent_geometry(
                qwen3_result, parent_index, legal, exact_control=True
            )
            if certification.certified:
                raise AlignmentCorrectnessError(
                    "same-tokenizer raw m>=2 unexpectedly certified as one-to-many"
                )
            if certification.category not in ANOMALY_CATEGORIES:
                raise AlignmentCorrectnessError(
                    f"unregistered anomaly category: {certification.category}"
                )
            if certification.category in {
                "tokenizer_or_pair_path_mixup", "rendered_text_difference",
                "token_id_difference", "offset_difference", "unexplained_other",
            }:
                raise AlignmentCorrectnessError(
                    f"unresolved Qwen3 anomaly: {certification.category}"
                )
            ordinal_message, _section = _message_section(qwen3_result, parent_index)
            receiver_span = tuple(qwen3_result["content_spans_slm"][ordinal_message])
            source_span = tuple(qwen3_result["content_spans_llm"][ordinal_message])
            receiver_offset = tuple(qwen3_result["slm_offsets"][parent_index])
            receiver_text = qwen3_result["slm_text"][receiver_offset[0]:receiver_offset[1]]
            unicode_info = unicode_records(receiver_text)
            for item in unicode_info:
                unicode_categories[item["category"]] += 1
            category_counts[certification.category] += 1
            if task == "mmlu-redux":
                subjects[sample.subject] += 1
            if sample.split in {"fit", "calibration"}:
                fit_cal_groups.add(sample.content_group_sha256)
                if len(legal) == 2:
                    fit_cal_m2 += 1
            candidate_indices = [index for index, _weight in legal]
            candidate_ids = [int(qwen3_result["llm_ids"][index]) for index in candidate_indices]
            candidate_offsets = [tuple(qwen3_result["llm_offsets"][index]) for index in candidate_indices]
            candidate_texts = [
                escaped_text(qwen3_result["llm_text"][start:end])
                for start, end in candidate_offsets
            ]
            ledger_rows.append({
                "schema_version": 1, "pair": "qwen3_1p7b_raw_soft_span",
                "task": task, "split": sample.split, "subject": sample.subject,
                "sample_key_sha256": sample.sample_key_sha256,
                "content_group_sha256": sample.content_group_sha256,
                "parent_index": parent_index,
                "receiver_token_id": int(qwen3_result["slm_ids"][parent_index]),
                "receiver_token_text": escaped_text(receiver_text),
                "receiver_offset": receiver_offset,
                "receiver_relative_interval": certification.receiver_interval,
                "receiver_unicode": unicode_info,
                "candidate_indices": candidate_indices,
                "candidate_token_ids": candidate_ids,
                "candidate_token_texts": candidate_texts,
                "candidate_offsets": candidate_offsets,
                "candidate_relative_intervals": certification.candidate_intervals,
                "candidate_weights": [weight for _index, weight in legal],
                "identity_candidate_present": int(parent_index in candidate_indices),
                "raw_m": len(legal), "certified": 0,
                "anomaly_category": certification.category,
                "certification_reason": certification.reason,
                "sender_directory": str(qwen3_dir),
                "sender_name_or_path": str(getattr(qwen3, "name_or_path", "")),
                "sender_file_fingerprint": qwen3_fp["payload"]["tokenizer_files"]["sha256"],
                "sender_runtime_fingerprint": qwen3_fp["sha256"],
            })
        if ordinal % 100 == 0 or ordinal == len(samples):
            print(json.dumps({"status": "PROGRESS", "task": task, "done": ordinal, "total": len(samples)}, sort_keys=True), flush=True)

    shard_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = shard_dir / "qwen3_forensic_ledger.csv"
    comparison_path = shard_dir / "qwen3_qwen25_comparison_rows.csv"
    ledger_count = atomic_csv(ledger_path, LEDGER_COLUMNS, ledger_rows)
    comparison_count = atomic_csv(
        comparison_path,
        COMPARISON_COLUMNS,
        [comparison_by_parent[key] for key in sorted(comparison_by_parent)],
    )
    summary = {
        "schema_version": 1, "status": "complete", "execution_sha": execution_sha,
        "task": task, "sample_count": len(samples),
        "identity_samples": identity_samples,
        "runtime_identity": {
            "receiver": receiver_fp, "qwen3": qwen3_fp,
            "equal": receiver_fp == qwen3_fp,
        },
        "sender_provenance": {
            "qwen3": {"directory": str(qwen3_dir), "name_or_path": str(getattr(qwen3, "name_or_path", "")), "runtime_fingerprint": qwen3_fp["sha256"]},
            "qwen25": {"directory": str(qwen25_dir), "name_or_path": str(getattr(qwen25, "name_or_path", "")), "runtime_fingerprint": qwen25_fp["sha256"]},
        },
        "qwen3_raw_m_ge_2_parent_count": len(qwen3_parents),
        "qwen3_raw_positive_group_count": len(qwen3_groups),
        "fit_calibration_qwen3_m2_parent_count": fit_cal_m2,
        "fit_calibration_qwen3_positive_group_count": len(fit_cal_groups),
        "category_counts": dict(sorted(category_counts.items())),
        "unicode_category_counts": dict(sorted(unicode_categories.items())),
        "mmlu_subject_counts": dict(sorted(subjects.items())),
        "comparison": {
            "positive_groups": set_metrics(qwen3_groups, qwen25_groups),
            "positive_parents": set_metrics(qwen3_parents, qwen25_parents),
            "offset_signatures": set_metrics(qwen3_offsets, qwen25_offsets),
            "candidate_id_signatures": set_metrics(qwen3_ids, qwen25_ids),
        },
        "artifacts": {
            "ledger": {"path": str(ledger_path), "rows": ledger_count, "sha256": sha256_file(ledger_path), "bytes": ledger_path.stat().st_size},
            "comparison_rows": {"path": str(comparison_path), "rows": comparison_count, "sha256": sha256_file(comparison_path), "bytes": comparison_path.stat().st_size},
        },
    }
    atomic_json(shard_dir / "summary.json", summary)
    manifest = {
        "schema_version": 1, "status": "complete", "execution_sha": execution_sha,
        "task": task, "summary_sha256": sha256_file(shard_dir / "summary.json"),
        "ledger_sha256": sha256_file(ledger_path),
        "comparison_sha256": sha256_file(comparison_path),
    }
    atomic_json(manifest_path, manifest)
    return manifest


def finalize_forensic(execution_sha: str, result_base: Path) -> dict[str, Any]:
    root, _lock = verify_lock(execution_sha, result_base)
    summaries = []
    for task in TASK_ORDER:
        manifest_path = root / "forensic" / task / "manifest.json"
        if not manifest_path.exists():
            raise AlignmentCorrectnessError(f"missing forensic shard: {task}")
        manifest = read_json(manifest_path)
        if manifest.get("status") != "complete" or manifest.get("execution_sha") != execution_sha:
            raise AlignmentCorrectnessError(f"invalid forensic shard: {task}")
        summary_path = root / "forensic" / task / "summary.json"
        if sha256_file(summary_path) != manifest["summary_sha256"]:
            raise AlignmentCorrectnessError(f"forensic summary hash mismatch: {task}")
        summaries.append(read_json(summary_path))
    if not all(summary["runtime_identity"]["equal"] for summary in summaries):
        raise AlignmentCorrectnessError("Qwen3 runtime identity failed")
    qwen_parent_count = sum(summary["qwen3_raw_m_ge_2_parent_count"] for summary in summaries)
    fit_cal_m2 = sum(summary["fit_calibration_qwen3_m2_parent_count"] for summary in summaries)
    fit_cal_groups = sum(summary["fit_calibration_qwen3_positive_group_count"] for summary in summaries)
    if fit_cal_m2 != 410 or fit_cal_groups != 56:
        raise AlignmentCorrectnessError(
            f"historical Qwen3 consistency mismatch: groups={fit_cal_groups}, m2={fit_cal_m2}"
        )
    category_counts: Counter[str] = Counter()
    unicode_counts: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()
    for summary in summaries:
        category_counts.update(summary["category_counts"])
        unicode_counts.update(summary["unicode_category_counts"])
        subject_counts.update(summary["mmlu_subject_counts"])
    unresolved = sum(
        category_counts[category]
        for category in (
            "tokenizer_or_pair_path_mixup", "rendered_text_difference",
            "token_id_difference", "offset_difference", "unexplained_other",
        )
    )
    if unresolved:
        raise AlignmentCorrectnessError(f"unresolved Qwen3 anomalies: {unresolved}")
    final = {
        "schema_version": 1, "stage": "FPCT-3.5",
        "status": "IDENTITY_FORENSIC_GO_TO_CONDITIONAL_CORRECTION",
        "execution_sha": execution_sha,
        "canonical_sample_count": sum(summary["sample_count"] for summary in summaries),
        "identity_sample_count": sum(summary["identity_samples"] for summary in summaries),
        "qwen3_raw_m_ge_2_parent_count_all_splits": qwen_parent_count,
        "fit_calibration_qwen3_positive_group_count": fit_cal_groups,
        "fit_calibration_qwen3_m2_parent_count": fit_cal_m2,
        "category_counts": dict(sorted(category_counts.items())),
        "unicode_category_counts": dict(sorted(unicode_counts.items())),
        "mmlu_subject_counts": dict(sorted(subject_counts.items())),
        "task_summaries": [
            {"task": summary["task"], "summary_sha256": sha256_file(root / "forensic" / summary["task"] / "summary.json")}
            for summary in summaries
        ],
        "gpu_authorized": False,
    }
    path = root / "forensic_summary.json"
    if path.exists():
        existing = read_json(path)
        if existing != final:
            raise AlignmentCorrectnessError("refusing to overwrite different forensic summary")
    else:
        atomic_json(path, final)
    state_path = root / "controller_state.json"
    state = read_json(state_path)
    state.update({"state": "IDENTITY_FORENSIC_COMPLETE", "completed_tasks": list(TASK_ORDER)})
    atomic_json(state_path, state)
    print(json.dumps({"status": final["status"], "summary": str(path), "sha256": sha256_file(path)}, sort_keys=True))
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "forensic", "finalize-forensic"))
    parser.add_argument("--execution-sha")
    parser.add_argument("--task", choices=TASK_ORDER + ("all",), default="all")
    parser.add_argument("--result-base", type=Path, default=DEFAULT_RESULT_BASE)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "freeze":
        freeze(args.result_base)
        return 0
    execution_sha = args.execution_sha or git("rev-parse", "HEAD")
    if args.mode == "forensic":
        tasks = TASK_ORDER if args.task == "all" else (args.task,)
        for task in tasks:
            run_forensic_task(task, execution_sha, args.result_base)
        return 0
    finalize_forensic(execution_sha, args.result_base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
