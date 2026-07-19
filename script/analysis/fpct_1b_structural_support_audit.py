from __future__ import annotations

"""FPCT-1B prospective, CPU-only structural-support audit.

The six modes intentionally separate input locking from natural tokenizer and
alignment execution:

* prepare: provenance, label-eliding canonical projection, hashes and splits;
* freeze: seal the clean execution commit and every audit input;
* selection: run fit+calibration pair/task shards only;
* lock-selection: derive the immutable pilot decision from selection shards;
* reporting: run model-selection/test shards and materialize seven artifacts;
* verify: independently reduce frozen CSVs and verify the decision/provenance.

No mode imports or instantiates a Hugging Face model class.  Tokenizers are instantiated only
inside the natural-audit modes.  All benchmark answer fields are used, where
production requires it, only at the raw-row eligibility boundary and are never
retained in the projected Sample object, hashes, logs, or output artifacts.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = 2
PROTOCOL_ID = "fpct_1a_structural_support_v2"
PAIR_ORDER = ("tinyllama", "qwen25_0p5b", "llama32_1b", "qwen3_1p7b")
TASK_ORDER = ("ai2-arc", "openbookqa", "mmlu-redux")
SPLIT_ORDER = ("fit", "calibration", "fit_calibration", "model-selection", "test")
SELECTION_SPLITS = frozenset({"fit", "calibration"})
REPORTING_SPLITS = frozenset({"model-selection", "test"})
EXPECTED_TOTAL_ROWS = 7265
EXPECTED_TOTAL_GROUPS = 7233
DATASET_CONTENT_SHA256 = "0366be7e5b129710024543bc065774bef165b6b6bca92541a3d641aea2918114"
SPLIT_VERSION = "cache-phase2a-v1-29a96947"
TOP_K = 4
UNIFORM_TOLERANCE = 1e-12
NORMALIZATION_TOLERANCE = 1e-12
PAIR_TASK_ROWS = len(PAIR_ORDER) * len(TASK_ORDER) * len(SPLIT_ORDER)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHARED_ROOT = Path("/netdisk/lijunsi/c2c-route1-identifiability")
DEFAULT_RESULT_BASE = Path(
    "local/final_results/fpct_factorized_transport/fpct_1b_ambiguity_support"
)
EXECUTION_MANIFEST = REPO_ROOT / "recipe/eval_recipe/fpct_1b/fpct_1b_execution_manifest.json"
SPLIT_MANIFEST = REPO_ROOT / "recipe/eval_recipe/fpct_1b/content_group_split_manifest.csv"
PRE_AUDIT_LOCK = REPO_ROOT / "recipe/eval_recipe/fpct_1b/pre_audit_lock.json"
PILOT_LOCK = REPO_ROOT / "recipe/eval_recipe/fpct_1b/pilot_selection_lock.json"
RESULT_MANIFEST = REPO_ROOT / "recipe/eval_recipe/fpct_1b/fpct_1b_result_manifest.json"

NORMATIVE_FILES = {
    "v1_protocol": (
        "FPCT_1A_AMBIGUITY_PROTOCOL.md",
        "6dced3da6b8f82228666eb64250f51dd6db53e78203249904d47c56e26988ea4",
    ),
    "v1_manifest": (
        "recipe/eval_recipe/fpct_1a/ambiguity_protocol_manifest.json",
        "8cb562a6dc915c59275b652bc99deb83e3d81c291185c931a9ed8325f1cb27f4",
    ),
    "approval_addendum": (
        "FPCT_1A_APPROVAL_ADDENDUM.md",
        "7ead6ce446168b36f9f5d8937e1a9aff8d522317d11b3ce330506069ccd7b44c",
    ),
    "v2_manifest": (
        "recipe/eval_recipe/fpct_1a/ambiguity_protocol_manifest_v2.json",
        "f7c8bd7fbc456484d1a40ca88d32dc8da3104c422a5addd89f7d033b12c82511",
    ),
    "preregistration": (
        "FPCT_PREREGISTRATION.md",
        "391b68807f5c7241fc0776993c7e3e1a95fdcae29b5500e7574075793f428e8c",
    ),
    "math_reference": (
        "math.md",
        "98d1b61f84d046548d5ba0070d6858c7080cb14fdef9169b08ad167461b809ad",
    ),
}

TRACKED_SOURCES = {
    "aligner": "rosetta/model/aligner.py",
    "prompt_builder": "rosetta/utils/evaluate.py",
    "canonical_evaluator": "script/evaluation/unified_evaluator.py",
    "hf_requirements": "recipe/train_recipe/identifiability/hf_cache_requirements.json",
}

PARENT_COLUMNS = (
    "schema_version", "pair", "pair_type", "task", "split",
    "content_group_sha256", "sample_key_sha256", "parent_index",
    "legal_candidate_count", "candidate_count_stratum",
    "is_primary_structural_m2", "is_high_cardinality_m3", "is_strict_m4",
    "a_max", "secondary_mass", "n_eff", "fallback_flag",
    "expected_inactive_slot_count", "invalid_positive_mass_count",
    "negative_weight_count", "nonfinite_weight_count",
    "duplicate_source_index_count", "normalization_error",
    "uniform_identity_error",
)
SAMPLE_COLUMNS = (
    "schema_version", "pair", "pair_type", "task", "split",
    "content_group_sha256", "sample_key_sha256", "eligible_parent_count",
    "m0_parent_count", "m1_parent_count", "m2_parent_count",
    "m3_parent_count", "m4_parent_count", "has_primary_structural_m2",
    "has_high_cardinality_m3", "has_strict_m4",
    "exact_f_equals_c_post_control", "integrity_failure",
)
GROUP_COLUMNS = (
    "schema_version", "pair", "pair_type", "task", "split",
    "content_group_sha256", "group_member_count",
    "representative_sample_key_sha256", "member_support_consistent",
    "has_primary_structural_m2", "has_high_cardinality_m3",
    "has_strict_m4", "exact_f_equals_c_post_control", "integrity_failure",
)
PAIR_TASK_COLUMNS = (
    "schema_version", "pair", "pair_type", "task", "split",
    "observed_positive_group_count", "observed_total_group_count",
    "observed_support_rate", "wilson95_low", "wilson95_high",
    "bonferroni9_wilson_lcb_sensitivity",
    "direct_structural_positive_sample_count", "canonical_sample_count",
    "direct_structural_support_ceiling",
    "high_cardinality_positive_group_count", "high_cardinality_support_rate",
    "high_cardinality_positive_sample_count", "high_cardinality_support_ceiling",
    "strict_positive_group_count", "strict_support_rate",
    "strict_positive_sample_count", "strict_support_ceiling",
)
READINESS_COLUMNS = (
    "schema_version", "pair", "pair_type", "heterogeneous",
    "ai2_arc_positive_group_count", "openbookqa_positive_group_count",
    "mmlu_redux_positive_group_count", "minimum_task_positive_group_count",
    "pooled_positive_group_count", "task_macro_observed_support_rate",
    "each_task_floor_pass", "pooled_floor_pass", "pair_pilot_ready",
    "not_ready_reason", "pilot_rank", "pilot_selected",
)


class AuditFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class Sample:
    task: str
    subject: str
    question_id: str
    question: str
    choices: tuple[str, ...]
    content_group_sha256: str
    sample_key_sha256: str
    split: str


@dataclass(frozen=True)
class PairSpec:
    pair: str
    pair_type: str
    heterogeneous: bool
    tokenizer_dir: Path
    repo_id: str
    resolved_tokenizer_revision: str
    source_repo_revision: str | None
    revision_resolution: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    if not files:
        raise AuditFailure(f"empty directory: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def tokenizer_bundle_sha256(path: Path) -> str:
    allowed = (
        "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
        "special_tokens_map.json", "vocab.json", "merges.txt", "config.json",
        "generation_config.json",
    )
    files = [path / name for name in allowed if (path / name).is_file()]
    if not files:
        raise AuditFailure(f"tokenizer bundle has no files: {path}")
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda p: p.name):
        digest.update(item.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditFailure(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return proc.stdout.strip()


def ensure_cpu_only() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise AuditFailure('CUDA_VISIBLE_DEVICES must be explicitly set to ""')


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def content_hash(question: str, choices: Sequence[str]) -> str:
    padded = [normalize_text(choices[i]) if i < min(4, len(choices)) else "" for i in range(10)]
    payload = {"question": normalize_text(question), "choices": padded}
    return sha256_bytes(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def sample_key_hash(task: str, subject: str, question_id: str) -> str:
    payload = {"task": task, "subject": subject, "question_id": question_id}
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def split_for_content(content_group_sha256: str) -> str:
    digest = hashlib.sha256(
        f"{SPLIT_VERSION}{DATASET_CONTENT_SHA256}{content_group_sha256}".encode("utf-8")
    ).digest()
    u = int.from_bytes(digest, "big") / float(1 << 256)
    if u < 0.30:
        return "fit"
    if u < 0.45:
        return "calibration"
    if u < 0.60:
        return "model-selection"
    return "test"


def _mmlu_row_is_production_eligible(raw: Mapping[str, Any]) -> bool:
    """Mirror production parse_answer only at the raw projection boundary."""
    try:
        error_type = raw.get("error_type", "")
        if error_type in {"no_correct_answer", "expert"}:
            return False
        if error_type == "wrong_groundtruth" and raw.get("correct_answer") is not None:
            answer = raw["correct_answer"]
            if answer >= "0" and answer <= "3":
                int(answer)
            else:
                ord(answer) - ord("A")
        else:
            int(raw["answer"])
        return True
    except (KeyError, TypeError, ValueError):
        return False


def load_projected_samples(shared_root: Path) -> list[Sample]:
    """Load raw local data and immediately project away answer/label fields."""
    import pyarrow.ipc as ipc
    import pyarrow.parquet as parquet

    data_root = shared_root / "data/c2c"
    projected: list[tuple[str, str, str, str, tuple[str, ...]]] = []

    arc_path = data_root / "ai2_arc/ARC-Challenge/test-00000-of-00001.parquet"
    for index, raw in enumerate(parquet.read_table(arc_path).to_pylist()):
        if raw.get("answerKey") not in {"A", "B", "C", "D"}:
            continue
        choices = tuple(str(x) for x in raw["choices"]["text"][:4])
        projected.append(("ai2-arc", "SPLIT_0_OF_1", str(index), str(raw["question"]), choices))

    obqa_path = data_root / "openbookqa/main/test-00000-of-00001.parquet"
    for index, raw in enumerate(parquet.read_table(obqa_path).to_pylist()):
        if raw.get("answerKey") not in {"A", "B", "C", "D"}:
            continue
        choices = tuple(str(x) for x in raw["choices"]["text"][:4])
        projected.append(("openbookqa", "SPLIT_0_OF_1", str(index), str(raw["question_stem"]), choices))

    mmlu_root = data_root / "mmlu-redux-2.0"
    for arrow_path in sorted(mmlu_root.glob("*/data-00000-of-00001.arrow")):
        subject = arrow_path.parent.name
        with arrow_path.open("rb") as handle:
            rows = ipc.open_stream(handle).read_all().to_pylist()
        for index, raw in enumerate(rows):
            if not _mmlu_row_is_production_eligible(raw):
                continue
            choices = tuple(str(x) for x in raw["choices"][:4])
            projected.append(("mmlu-redux", subject, str(index), str(raw["question"]), choices))

    samples: list[Sample] = []
    for task, subject, question_id, question, choices in projected:
        group_hash = content_hash(question, choices)
        samples.append(
            Sample(
                task=task,
                subject=subject,
                question_id=question_id,
                question=question,
                choices=choices,
                content_group_sha256=group_hash,
                sample_key_sha256=sample_key_hash(task, subject, question_id),
                split=split_for_content(group_hash),
            )
        )
    return samples


def validate_canonical_samples(samples: Sequence[Sample]) -> dict[str, Any]:
    expected = {
        "ai2-arc": (1150, 1150, "365daa1a31374b5fccde3de93d7aa248b1e1080f326491b7f8aefba6274a0142"),
        "openbookqa": (500, 500, "803946f36a5cd5ca99b041e6f451b56c80a7b5c4c8f13a32d5c7efdc34a76994"),
        "mmlu-redux": (5615, 5583, "d9284f27cf799e5ba225c3441c2e2822e1ad53f35d6ed58792273d6c7f013291"),
    }
    task_hashes: dict[str, str] = {}
    task_summary: dict[str, Any] = {}
    for task in TASK_ORDER:
        rows = sorted((s for s in samples if s.task == task), key=lambda s: (s.task, s.subject, s.question_id))
        payload = "\n".join(
            "\t".join((s.task, s.subject, s.question_id, s.content_group_sha256)) for s in rows
        ).encode("utf-8")
        digest = sha256_bytes(payload)
        row_count = len(rows)
        group_count = len({s.content_group_sha256 for s in rows})
        exp_rows, exp_groups, exp_sha = expected[task]
        if (row_count, group_count, digest) != (exp_rows, exp_groups, exp_sha):
            raise AuditFailure(
                f"canonical input mismatch for {task}: "
                f"found rows={row_count}, groups={group_count}, sha={digest}"
            )
        task_hashes[task] = digest
        task_summary[task] = {"rows": row_count, "distinct_content_groups": group_count, "content_sha256": digest}
    dataset_sha = sha256_bytes(
        json.dumps(task_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    if dataset_sha != DATASET_CONTENT_SHA256:
        raise AuditFailure(f"dataset content SHA mismatch: {dataset_sha}")
    if len(samples) != EXPECTED_TOTAL_ROWS or len({s.content_group_sha256 for s in samples}) != EXPECTED_TOTAL_GROUPS:
        raise AuditFailure("global canonical row/group count mismatch")
    return {
        "tasks": task_summary,
        "total_rows": len(samples),
        "total_distinct_content_groups": len({s.content_group_sha256 for s in samples}),
        "dataset_content_sha256": dataset_sha,
    }


def verify_normative_files() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, (relative, expected) in NORMATIVE_FILES.items():
        path = REPO_ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise AuditFailure(f"normative SHA mismatch for {relative}: {actual} != {expected}")
        results[name] = {"path": relative, "sha256": actual, "normative": name != "math_reference"}
    v2 = read_json(REPO_ROOT / NORMATIVE_FILES["v2_manifest"][0])
    if v2.get("schema_version") != 2 or v2.get("protocol_id") != PROTOCOL_ID:
        raise AuditFailure("operative v2 manifest identity mismatch")
    return results


def _stager_record(shared_root: Path) -> tuple[Path, dict[str, Any]]:
    candidates = sorted((shared_root / "status").glob("hf-cache-stager-*.json"))
    if not candidates:
        raise AuditFailure("local HF cache stager record is missing")
    payloads = [(path, read_json(path)) for path in candidates]
    def evidence_payload(payload: Mapping[str, Any]) -> str:
        # Repeated successful staging audits may differ only in their wall-clock
        # generation time.  Asset paths, revisions, sizes and hashes must remain
        # byte-for-byte identical.
        stable = {key: value for key, value in payload.items() if key != "generated_at"}
        return json.dumps(stable, sort_keys=True, separators=(",", ":"))

    canonical = evidence_payload(payloads[0][1])
    for path, payload in payloads[1:]:
        if evidence_payload(payload) != canonical:
            raise AuditFailure(f"non-unique local HF cache stager records: {path}")
    return payloads[-1]


def _find_record(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    for section in ("repositories", "shared_models", "shared_datasets"):
        for item in payload.get(section, []):
            if item.get("key") == key:
                return item
    raise AuditFailure(f"missing stager record for {key}")


def resolve_assets(shared_root: Path) -> dict[str, Any]:
    stager_path, stager = _stager_record(shared_root)
    model_dirs = {
        "receiver_qwen3_0p6b": shared_root / "models/Qwen3-0.6B",
        "tinyllama": shared_root / "models/TinyLlama-1.1B-Chat-v1.0",
        "qwen25_0p5b": shared_root / "models/Qwen2.5-0.5B-Instruct",
        "llama32_1b": shared_root / "models/Llama-3.2-1B-Instruct",
        "qwen3_1p7b": shared_root / "models/Qwen3-1.7B",
    }
    expected_files = {
        "receiver_qwen3_0p6b": {
            "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
            "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        },
        "tinyllama": {
            "tokenizer_config.json": "7b41ba7d0eb91e77914ca3dafde559ea3e19878769b7e68409e89bed5222e77a",
            "tokenizer.json": "bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e",
            "tokenizer.model": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
        },
        "qwen25_0p5b": {
            "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
            "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        },
        "llama32_1b": {
            "tokenizer_config.json": "9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e",
            "tokenizer.json": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
        },
        "qwen3_1p7b": {
            "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
            "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        },
    }
    tokenizer_records: dict[str, Any] = {}
    for key, directory in model_dirs.items():
        if not directory.is_dir():
            raise AuditFailure(f"tokenizer directory missing: {directory}")
        files: dict[str, str] = {}
        for relative, expected in expected_files[key].items():
            actual = sha256_file(directory / relative)
            if actual != expected:
                raise AuditFailure(f"tokenizer SHA mismatch: {directory / relative}")
            files[relative] = actual
        tokenizer_records[key] = {
            "directory": str(directory),
            "tokenizer_bundle_sha256": tokenizer_bundle_sha256(directory),
            "files": files,
        }

    repo_revisions = {
        "receiver_qwen3_0p6b": _find_record(stager, "qwen3_0p6b").get("resolved_revision"),
        "tinyllama": _find_record(stager, "tinyllama_1p1b").get("resolved_revision"),
        "qwen25_0p5b": _find_record(stager, "qwen25_0p5b").get("resolved_revision"),
    }
    expected_revisions = {
        "receiver_qwen3_0p6b": "c1899de289a04d12100db370d81485cdf75e47ca",
        "tinyllama": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        "qwen25_0p5b": "7ae557604adf67be50417f59c2c2f167def9a775",
    }
    if repo_revisions != expected_revisions:
        raise AuditFailure(f"resolved tokenizer revision mismatch: {repo_revisions}")

    llama_meta_dir = model_dirs["llama32_1b"] / ".cache/huggingface/download"
    llama_revisions = {
        path.read_text(encoding="utf-8").splitlines()[0].strip()
        for path in llama_meta_dir.glob("*.metadata")
        if path.is_file() and path.read_text(encoding="utf-8").splitlines()
    }
    if llama_revisions != {"9213176726f574b556790deb65791e0c5aa438b6"}:
        raise AuditFailure(f"Llama tokenizer revision is not unique: {sorted(llama_revisions)}")

    # The shared Qwen3-1.7B copy lacks immutable repo metadata.  Its tokenizer
    # bytes are, however, exactly identical to the immutable receiver snapshot.
    # Record that byte-identity alias without claiming a Qwen3-1.7B repo commit.
    if expected_files["qwen3_1p7b"] != expected_files["receiver_qwen3_0p6b"]:
        raise AuditFailure("same-tokenizer byte identity was not established")
    tokenizer_records["receiver_qwen3_0p6b"].update({
        "source_repo_revision": expected_revisions["receiver_qwen3_0p6b"],
        "resolved_tokenizer_revision": expected_revisions["receiver_qwen3_0p6b"],
        "revision_resolution": "immutable_hf_snapshot",
    })
    tokenizer_records["tinyllama"].update({
        "source_repo_revision": expected_revisions["tinyllama"],
        "resolved_tokenizer_revision": expected_revisions["tinyllama"],
        "revision_resolution": "immutable_hf_snapshot",
    })
    tokenizer_records["qwen25_0p5b"].update({
        "source_repo_revision": expected_revisions["qwen25_0p5b"],
        "resolved_tokenizer_revision": expected_revisions["qwen25_0p5b"],
        "revision_resolution": "immutable_hf_snapshot",
    })
    tokenizer_records["llama32_1b"].update({
        "source_repo_revision": next(iter(llama_revisions)),
        "resolved_tokenizer_revision": next(iter(llama_revisions)),
        "revision_resolution": "consistent_local_download_metadata",
    })
    tokenizer_records["qwen3_1p7b"].update({
        "source_repo_revision": None,
        "resolved_tokenizer_revision": expected_revisions["receiver_qwen3_0p6b"],
        "revision_resolution": "byte_identical_alias_to_Qwen3-0.6B_immutable_snapshot",
        "source_repo_revision_claimed": False,
    })

    expected_dataset_dirs = {
        "ai2_arc": (shared_root / "data/c2c/ai2_arc", "57aff44115478c094aba1ce9ec3c13ee3556ee47e637eadb3ac728691ad70b51"),
        "openbookqa": (shared_root / "data/c2c/openbookqa", "5f903c0bd9272642eb983c4e30ec71a16e3922c578dc1f08a6cdfb33e99b4ecf"),
        "mmlu_redux": (shared_root / "data/c2c/mmlu-redux-2.0", "3630ceb45031facee39532ba1110760d49c71270be0bd5de96e6235971cdb402"),
    }
    dataset_records: dict[str, Any] = {}
    for key, (directory, expected) in expected_dataset_dirs.items():
        actual = directory_tree_sha256(directory)
        if actual != expected:
            raise AuditFailure(f"dataset directory SHA mismatch for {key}: {actual}")
        dataset_records[key] = {"directory": str(directory), "directory_sha256": actual}

    return {
        "stager_record": {"path": str(stager_path), "sha256": sha256_file(stager_path)},
        "tokenizers": tokenizer_records,
        "datasets": dataset_records,
    }


def materialize_split_manifest(samples: Sequence[Sample], path: Path) -> None:
    rows = sorted(samples, key=lambda s: (TASK_ORDER.index(s.task), s.subject, int(s.question_id)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("schema_version", "task", "sample_key_sha256", "content_group_sha256", "split"),
            lineterminator="\n",
        )
        writer.writeheader()
        for sample in rows:
            writer.writerow({
                "schema_version": SCHEMA_VERSION,
                "task": sample.task,
                "sample_key_sha256": sample.sample_key_sha256,
                "content_group_sha256": sample.content_group_sha256,
                "split": sample.split,
            })


def prepare(shared_root: Path) -> None:
    ensure_cpu_only()
    if "transformers" in sys.modules or "rosetta.model.aligner" in sys.modules:
        raise AuditFailure("prepare must run before any tokenizer/aligner import")
    normative = verify_normative_files()
    assets = resolve_assets(shared_root)
    samples = load_projected_samples(shared_root)
    canonical = validate_canonical_samples(samples)
    materialize_split_manifest(samples, SPLIT_MANIFEST)
    split_counts = Counter((s.task, s.split) for s in samples)
    group_split: dict[str, str] = {}
    for sample in samples:
        prior = group_split.setdefault(sample.content_group_sha256, sample.split)
        if prior != sample.split:
            raise AuditFailure("content group assigned to multiple splits")
    source_hashes = {name: {"path": rel, "sha256": sha256_file(REPO_ROOT / rel)} for name, rel in TRACKED_SOURCES.items()}
    manifest = {
        "schema_version": 1,
        "stage": "FPCT-1B",
        "mode": "prepare",
        "prepared_at": utc_now(),
        "protocol_id": PROTOCOL_ID,
        "prepare_guards": {
            "natural_tokenizer_execution": False,
            "chat_template_rendering": False,
            "tokenization": False,
            "alignment": False,
            "model_forward": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "normative": normative,
        "tracked_sources": source_hashes,
        "assets": assets,
        "canonical_input": canonical,
        "split_contract": {
            "version": SPLIT_VERSION,
            "formula": "SHA256(split_version + dataset_content_sha256 + content_hash)",
            "manifest": str(SPLIT_MANIFEST.relative_to(REPO_ROOT)),
            "manifest_sha256": sha256_file(SPLIT_MANIFEST),
            "rows": len(samples),
            "distinct_content_groups": len(group_split),
            "counts": {f"{task}/{split_name}": split_counts[(task, split_name)] for task in TASK_ORDER for split_name in ("fit", "calibration", "model-selection", "test")},
        },
        "alignment": {
            "production_api": "TokenAligner.align_chat_messages_soft",
            "strategy": "soft_span_overlap_v2", "top_k": TOP_K,
            "score_mode": "uniform", "boundary_bonus": 0.5,
            "boundary_tolerance": 1, "min_weight": 0.0,
            "candidate_window": 0, "reweight_mode": "none",
            "apply_confidence_control": False,
        },
        "thresholds": {
            "primary": "primary_structural_m2:m>=2",
            "sensitivities": ["high_cardinality_m3:m>=3", "strict_m4:m==4"],
            "readiness": {"minimum_positive_groups_each_task": 30, "minimum_pooled_positive_groups": 100},
        },
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "datasets", "pyarrow")},
        },
    }
    write_json(EXECUTION_MANIFEST, manifest)
    print(json.dumps({"status": "PREPARED", "rows": len(samples), "groups": len(group_split), "split_manifest_sha256": sha256_file(SPLIT_MANIFEST)}, sort_keys=True))


def current_execution_sha() -> str:
    sha = git("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise AuditFailure(f"invalid execution SHA: {sha}")
    return sha


def result_root(execution_sha: str, result_base: Path) -> Path:
    return REPO_ROOT / result_base / f"rev_{execution_sha}"


def freeze(shared_root: Path, result_base: Path) -> None:
    ensure_cpu_only()
    if git("status", "--short"):
        raise AuditFailure("freeze requires a clean execution commit")
    execution_sha = current_execution_sha()
    manifest = read_json(EXECUTION_MANIFEST)
    if sha256_file(SPLIT_MANIFEST) != manifest["split_contract"]["manifest_sha256"]:
        raise AuditFailure("split manifest changed after prepare")
    root = result_root(execution_sha, result_base)
    root.mkdir(parents=True, exist_ok=True)
    code_path = Path(__file__).resolve()
    tests_path = REPO_ROOT / "test/test_fpct_1b_structural_support_audit.py"
    lock = {
        "schema_version": 1, "stage": "FPCT-1B", "execution_sha": execution_sha,
        "created_at": utc_now(), "protocol_id": PROTOCOL_ID,
        "clean_execution_commit": True,
        "analysis_code": {"path": str(code_path.relative_to(REPO_ROOT)), "sha256": sha256_file(code_path)},
        "tests": {"path": str(tests_path.relative_to(REPO_ROOT)), "sha256": sha256_file(tests_path)},
        "execution_manifest": {"path": str(EXECUTION_MANIFEST.relative_to(REPO_ROOT)), "sha256": sha256_file(EXECUTION_MANIFEST)},
        "split_manifest": {"path": str(SPLIT_MANIFEST.relative_to(REPO_ROOT)), "sha256": sha256_file(SPLIT_MANIFEST)},
        "normative": verify_normative_files(),
        "assets": resolve_assets(shared_root),
        "tracked_sources": {name: {"path": rel, "sha256": sha256_file(REPO_ROOT / rel)} for name, rel in TRACKED_SOURCES.items()},
        "alignment": manifest["alignment"], "thresholds": manifest["thresholds"],
        "canonical_input": manifest["canonical_input"], "runtime": manifest["runtime"],
        "result_root": str(root),
    }
    write_json(PRE_AUDIT_LOCK, lock)
    local_lock = root / "pre_audit_lock.json"
    write_json(local_lock, lock)
    print(json.dumps({"status": "FROZEN", "execution_sha": execution_sha, "pre_audit_lock_sha256": sha256_file(PRE_AUDIT_LOCK)}, sort_keys=True))


def verify_frozen_state(execution_sha: str, shared_root: Path) -> dict[str, Any]:
    lock = read_json(PRE_AUDIT_LOCK)
    if lock.get("execution_sha") != execution_sha:
        raise AuditFailure("pre-audit lock execution SHA mismatch")
    checks = {
        Path(lock["analysis_code"]["path"]): lock["analysis_code"]["sha256"],
        Path(lock["tests"]["path"]): lock["tests"]["sha256"],
        Path(lock["execution_manifest"]["path"]): lock["execution_manifest"]["sha256"],
        Path(lock["split_manifest"]["path"]): lock["split_manifest"]["sha256"],
    }
    for relative, expected in checks.items():
        actual = sha256_file(REPO_ROOT / relative)
        if actual != expected:
            raise AuditFailure(f"frozen file changed: {relative}")
    verify_normative_files()
    current_assets = resolve_assets(shared_root)
    if current_assets != lock["assets"]:
        raise AuditFailure("local tokenizer/dataset assets changed after freeze")
    return lock


def pair_specs(shared_root: Path, lock: Mapping[str, Any]) -> dict[str, PairSpec]:
    tokens = lock["assets"]["tokenizers"]
    return {
        "tinyllama": PairSpec("tinyllama", "heterogeneous", True, Path(tokens["tinyllama"]["directory"]), "TinyLlama/TinyLlama-1.1B-Chat-v1.0", tokens["tinyllama"]["resolved_tokenizer_revision"], tokens["tinyllama"]["source_repo_revision"], tokens["tinyllama"]["revision_resolution"]),
        "qwen25_0p5b": PairSpec("qwen25_0p5b", "heterogeneous", True, Path(tokens["qwen25_0p5b"]["directory"]), "Qwen/Qwen2.5-0.5B-Instruct", tokens["qwen25_0p5b"]["resolved_tokenizer_revision"], tokens["qwen25_0p5b"]["source_repo_revision"], tokens["qwen25_0p5b"]["revision_resolution"]),
        "llama32_1b": PairSpec("llama32_1b", "heterogeneous", True, Path(tokens["llama32_1b"]["directory"]), "meta-llama/Llama-3.2-1B-Instruct", tokens["llama32_1b"]["resolved_tokenizer_revision"], tokens["llama32_1b"]["source_repo_revision"], tokens["llama32_1b"]["revision_resolution"]),
        "qwen3_1p7b": PairSpec("qwen3_1p7b", "same_tokenizer_control", False, Path(tokens["qwen3_1p7b"]["directory"]), "Qwen/Qwen3-1.7B", tokens["qwen3_1p7b"]["resolved_tokenizer_revision"], tokens["qwen3_1p7b"]["source_repo_revision"], tokens["qwen3_1p7b"]["revision_resolution"]),
    }


def prompt_for_sample(sample: Sample) -> str:
    from rosetta.utils.evaluate import build_prompt
    choices = "".join(f"{chr(65 + i)}. {choice}\n" for i, choice in enumerate(sample.choices))
    return build_prompt(dataset="mmlu-redux", locale="", question=sample.question, choices=choices, use_cot=False, use_template=True)


def classify_candidate_row(
    indices: Sequence[int], weights: Sequence[float], *, source_length: int,
    source_padding_mask: Sequence[bool], fallback_flag: bool,
) -> tuple[dict[str, Any], bool]:
    invalid_positive = negative = nonfinite = duplicates = 0
    legal: list[tuple[int, float]] = []
    for index, raw_weight in zip(indices, weights):
        weight = float(raw_weight)
        if not math.isfinite(weight):
            nonfinite += 1
            continue
        if weight < 0:
            negative += 1
            continue
        index_valid = 0 <= int(index) < source_length and not source_padding_mask[int(index)]
        if weight > 0 and not index_valid:
            invalid_positive += 1
            continue
        if weight > 0 and index_valid:
            legal.append((int(index), weight))
    legal_indices = [index for index, _ in legal]
    duplicates = len(legal_indices) - len(set(legal_indices))
    total = sum(weight for _, weight in legal)
    normalized = [(index, weight / total) for index, weight in legal] if total > 0 else []
    m = len(normalized)
    normalization_error = abs(sum(weight for _, weight in normalized) - 1.0) if normalized else 0.0
    if m:
        probs = [weight for _, weight in normalized]
        a_max = max(probs)
        secondary_mass = 1.0 - a_max
        n_eff = 1.0 / sum(weight * weight for weight in probs)
        uniform_identity_error = max(abs(weight - 1.0 / m) for weight in probs)
    else:
        a_max = secondary_mass = n_eff = None
        uniform_identity_error = 0.0
    inactive = sum(1 for index, weight in zip(indices, weights) if int(index) == -1 and float(weight) == 0.0)
    failure = bool(
        invalid_positive or negative or nonfinite or duplicates or m > TOP_K
        or normalization_error > NORMALIZATION_TOLERANCE
        or uniform_identity_error > UNIFORM_TOLERANCE
    )
    return {
        "legal_candidate_count": m,
        "candidate_count_stratum": f"m{m}_" + ("zero_support" if m == 0 else "one_to_one" if m == 1 else "low_cardinality" if m == 2 else "high_cardinality" if m == 3 else "strict"),
        "is_primary_structural_m2": int(m >= 2),
        "is_high_cardinality_m3": int(m >= 3),
        "is_strict_m4": int(m == 4),
        "a_max": a_max, "secondary_mass": secondary_mass, "n_eff": n_eff,
        "fallback_flag": int(fallback_flag),
        "expected_inactive_slot_count": inactive,
        "invalid_positive_mass_count": invalid_positive,
        "negative_weight_count": negative, "nonfinite_weight_count": nonfinite,
        "duplicate_source_index_count": duplicates,
        "normalization_error": normalization_error,
        "uniform_identity_error": uniform_identity_error,
    }, failure


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return format(value, ".17g")
    return value


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in columns})
            count += 1
    return count


def read_csv(path: Path, expected_columns: Sequence[str] | None = None) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if expected_columns is not None and tuple(reader.fieldnames or ()) != tuple(expected_columns):
            raise AuditFailure(f"CSV schema mismatch: {path}")
        return list(reader)


def shard_paths(root: Path, phase: str, pair: str, task: str) -> dict[str, Path]:
    base = root / "shards" / phase / f"{pair}__{task}"
    return {
        "parent": base.with_suffix(".parent.csv"),
        "sample": base.with_suffix(".sample.csv"),
        "group": base.with_suffix(".group.csv"),
        "manifest": base.with_suffix(".manifest.json"),
    }


def shard_is_complete(paths: Mapping[str, Path], execution_sha: str) -> bool:
    if not paths["manifest"].is_file():
        return False
    manifest = read_json(paths["manifest"])
    if manifest.get("execution_sha") != execution_sha or manifest.get("status") != "complete":
        return False
    for key in ("parent", "sample", "group"):
        if not paths[key].is_file() or sha256_file(paths[key]) != manifest["artifacts"][key]["sha256"]:
            return False
    return True


def run_shard(
    *, pair: PairSpec, task: str, splits: frozenset[str], phase: str,
    samples: Sequence[Sample], receiver_tokenizer: Any, sender_tokenizer: Any,
    execution_sha: str, root: Path,
) -> dict[str, Any]:
    from rosetta.model.aligner import TokenAligner

    paths = shard_paths(root, phase, pair.pair, task)
    if shard_is_complete(paths, execution_sha):
        print(json.dumps({"status": "RESUME_SKIP", "phase": phase, "pair": pair.pair, "task": task}, sort_keys=True), flush=True)
        return read_json(paths["manifest"])
    if any(path.exists() for path in paths.values()):
        raise AuditFailure(f"partial/existing shard must not be overwritten: {paths['manifest']}")

    aligner = TokenAligner(
        receiver_tokenizer, sender_tokenizer, strategy="soft_span_overlap_v2",
        soft_alignment_score_mode="uniform", soft_alignment_boundary_bonus=0.5,
        soft_alignment_boundary_tolerance=1, soft_alignment_min_weight=0.0,
        soft_alignment_confidence_mode="none", soft_alignment_reweight_mode="none",
        soft_alignment_candidate_window=0, verbose=False,
    )
    selected = [sample for sample in samples if sample.task == task and sample.split in splits]
    parent_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    integrity_messages: list[str] = []

    for ordinal, sample in enumerate(selected, 1):
        prompt = prompt_for_sample(sample)
        result = aligner.align_chat_messages_soft(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            enable_thinking=False, top_k=TOP_K, return_details=False,
            apply_confidence_control=False,
        )
        alignment = result["soft_alignment"]
        counts = Counter()
        sample_failure = False
        eligible_count = 0
        for parent_index, (is_message, is_padding) in enumerate(zip(result["message_mask"], result["slm_padding_mask"])):
            if not is_message or is_padding:
                continue
            eligible_count += 1
            classified, failure = classify_candidate_row(
                alignment["source_indices"][parent_index], alignment["source_weights"][parent_index],
                source_length=len(result["llm_ids"]), source_padding_mask=result["llm_padding_mask"],
                fallback_flag=alignment["fallback_mask"][parent_index],
            )
            m = classified["legal_candidate_count"]
            counts[m] += 1
            sample_failure = sample_failure or failure
            parent_rows.append({
                "schema_version": SCHEMA_VERSION, "pair": pair.pair,
                "pair_type": pair.pair_type, "task": task, "split": sample.split,
                "content_group_sha256": sample.content_group_sha256,
                "sample_key_sha256": sample.sample_key_sha256,
                "parent_index": parent_index, **classified,
            })
        if sample_failure:
            integrity_messages.append(sample.sample_key_sha256)
        sample_rows.append({
            "schema_version": SCHEMA_VERSION, "pair": pair.pair,
            "pair_type": pair.pair_type, "task": task, "split": sample.split,
            "content_group_sha256": sample.content_group_sha256,
            "sample_key_sha256": sample.sample_key_sha256,
            "eligible_parent_count": eligible_count,
            "m0_parent_count": counts[0], "m1_parent_count": counts[1],
            "m2_parent_count": counts[2], "m3_parent_count": counts[3],
            "m4_parent_count": counts[4],
            "has_primary_structural_m2": int(sum(counts[m] for m in (2, 3, 4)) > 0),
            "has_high_cardinality_m3": int(sum(counts[m] for m in (3, 4)) > 0),
            "has_strict_m4": int(counts[4] > 0),
            "exact_f_equals_c_post_control": int(sum(counts[m] for m in (2, 3, 4)) == 0),
            "integrity_failure": int(sample_failure),
        })
        if ordinal % 100 == 0 or ordinal == len(selected):
            print(json.dumps({"status": "PROGRESS", "phase": phase, "pair": pair.pair, "task": task, "done": ordinal, "total": len(selected)}, sort_keys=True), flush=True)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        groups[row["content_group_sha256"]].append(row)
    group_rows: list[dict[str, Any]] = []
    for group_hash, members in sorted(groups.items()):
        signatures = {
            (m["has_primary_structural_m2"], m["has_high_cardinality_m3"], m["has_strict_m4"], m["exact_f_equals_c_post_control"], m["integrity_failure"])
            for m in members
        }
        consistent = len(signatures) == 1
        if not consistent:
            integrity_messages.append(group_hash)
        rep = min(m["sample_key_sha256"] for m in members)
        first = members[0]
        group_rows.append({
            "schema_version": SCHEMA_VERSION, "pair": pair.pair,
            "pair_type": pair.pair_type, "task": task, "split": first["split"],
            "content_group_sha256": group_hash, "group_member_count": len(members),
            "representative_sample_key_sha256": rep,
            "member_support_consistent": int(consistent),
            "has_primary_structural_m2": first["has_primary_structural_m2"],
            "has_high_cardinality_m3": first["has_high_cardinality_m3"],
            "has_strict_m4": first["has_strict_m4"],
            "exact_f_equals_c_post_control": first["exact_f_equals_c_post_control"],
            "integrity_failure": int(any(int(m["integrity_failure"]) for m in members) or not consistent),
        })

    if integrity_messages:
        failure_path = root / "integrity_failure.json"
        if not failure_path.exists():
            write_json(failure_path, {"status": "INCONCLUSIVE", "phase": phase, "pair": pair.pair, "task": task, "failure_count": len(integrity_messages), "failure_hashes": sorted(integrity_messages)})
        raise AuditFailure(f"integrity failure in {phase}/{pair.pair}/{task}")

    parent_count = write_csv(paths["parent"], PARENT_COLUMNS, parent_rows)
    sample_count = write_csv(paths["sample"], SAMPLE_COLUMNS, sample_rows)
    group_count = write_csv(paths["group"], GROUP_COLUMNS, group_rows)
    manifest = {
        "schema_version": 1, "status": "complete", "execution_sha": execution_sha,
        "phase": phase, "pair": pair.pair, "task": task,
        "splits": sorted(splits), "completed_at": utc_now(),
        "artifacts": {
            key: {"path": str(paths[key]), "sha256": sha256_file(paths[key]), "rows": count, "bytes": paths[key].stat().st_size}
            for key, count in (("parent", parent_count), ("sample", sample_count), ("group", group_count))
        },
        "integrity_failure": False,
    }
    write_json(paths["manifest"], manifest)
    return manifest


def load_tokenizers(lock: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    from transformers import AutoTokenizer
    receiver_dir = Path(lock["assets"]["tokenizers"]["receiver_qwen3_0p6b"]["directory"])
    receiver = AutoTokenizer.from_pretrained(receiver_dir, local_files_only=True, use_fast=True)
    senders = {
        pair: AutoTokenizer.from_pretrained(spec.tokenizer_dir, local_files_only=True, use_fast=True)
        for pair, spec in pair_specs(DEFAULT_SHARED_ROOT, lock).items()
    }
    return receiver, senders


def natural_audit(phase: str, splits: frozenset[str], shared_root: Path, result_base: Path) -> None:
    ensure_cpu_only()
    execution_sha = current_execution_sha()
    lock = verify_frozen_state(execution_sha, shared_root)
    root = result_root(execution_sha, result_base)
    receiver, senders = load_tokenizers(lock)
    samples = load_projected_samples(shared_root)
    validate_canonical_samples(samples)
    specs = pair_specs(shared_root, lock)
    for pair_id in PAIR_ORDER:
        for task in TASK_ORDER:
            run_shard(
                pair=specs[pair_id], task=task, splits=splits, phase=phase,
                samples=samples, receiver_tokenizer=receiver,
                sender_tokenizer=senders[pair_id], execution_sha=execution_sha, root=root,
            )
    print(json.dumps({"status": f"{phase.upper()}_COMPLETE", "execution_sha": execution_sha}, sort_keys=True))


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def one_sided_wilson_lcb(successes: int, total: int, alpha: float) -> float | None:
    if total <= 0:
        return None
    z = NormalDist().inv_cdf(1.0 - alpha)
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return max(0.0, center - half)


def selection_group_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for pair in PAIR_ORDER:
        for task in TASK_ORDER:
            paths = shard_paths(root, "selection", pair, task)
            if not shard_is_complete(paths, root.name.removeprefix("rev_")):
                raise AuditFailure(f"selection shard incomplete: {pair}/{task}")
            rows.extend(read_csv(paths["group"], GROUP_COLUMNS))
    return rows


def derive_readiness(group_rows: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, Any]], str, list[str], str | None]:
    by_pair_task: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in group_rows:
        if row["split"] not in SELECTION_SPLITS:
            raise AuditFailure("selection rows contain non-selection split")
        if int(row["integrity_failure"]):
            raise AuditFailure("selection rows contain integrity failure")
        by_pair_task[(row["pair"], row["task"])].append(row)
    readiness: list[dict[str, Any]] = []
    for pair in PAIR_ORDER:
        counts: dict[str, int] = {}
        rates: dict[str, float] = {}
        for task in TASK_ORDER:
            rows = by_pair_task[(pair, task)]
            counts[task] = sum(int(r["has_primary_structural_m2"]) for r in rows)
            rates[task] = counts[task] / len(rows) if rows else 0.0
        minimum = min(counts.values())
        pooled = sum(counts.values())
        each_pass = all(value >= 30 for value in counts.values())
        pooled_pass = pooled >= 100
        heterogeneous = pair != "qwen3_1p7b"
        ready = heterogeneous and each_pass and pooled_pass
        reasons = []
        if not heterogeneous: reasons.append("same_tokenizer_control_excluded")
        if not each_pass: reasons.append("one_or_more_tasks_below_30_positive_groups")
        if not pooled_pass: reasons.append("pooled_positive_groups_below_100")
        readiness.append({
            "schema_version": SCHEMA_VERSION, "pair": pair,
            "pair_type": "same_tokenizer_control" if not heterogeneous else "heterogeneous",
            "heterogeneous": int(heterogeneous),
            "ai2_arc_positive_group_count": counts["ai2-arc"],
            "openbookqa_positive_group_count": counts["openbookqa"],
            "mmlu_redux_positive_group_count": counts["mmlu-redux"],
            "minimum_task_positive_group_count": minimum,
            "pooled_positive_group_count": pooled,
            "task_macro_observed_support_rate": sum(rates.values()) / len(TASK_ORDER),
            "each_task_floor_pass": int(each_pass), "pooled_floor_pass": int(pooled_pass),
            "pair_pilot_ready": int(ready), "not_ready_reason": ";".join(reasons),
            "pilot_rank": None, "pilot_selected": 0,
        })
    ready_rows = [row for row in readiness if row["pair_pilot_ready"]]
    ranked = sorted(ready_rows, key=lambda row: (-row["minimum_task_positive_group_count"], -row["task_macro_observed_support_rate"], -row["pooled_positive_group_count"], row["pair"]))
    for rank, row in enumerate(ranked, 1):
        row["pilot_rank"] = rank
        row["pilot_selected"] = int(rank == 1)
    total_positive = sum(row["pooled_positive_group_count"] for row in readiness if row["heterogeneous"])
    if total_positive == 0:
        status = "NO_SUPPORT"
    elif len(ranked) == 0:
        status = "DIAGNOSTIC_ONLY"
    elif len(ranked) == 1:
        status = "SINGLE_PAIR_PILOT_READY"
    else:
        status = "CROSS_PAIR_PILOT_READY"
    return readiness, status, [row["pair"] for row in ranked], ranked[0]["pair"] if ranked else None


def lock_selection(shared_root: Path, result_base: Path) -> None:
    ensure_cpu_only()
    execution_sha = current_execution_sha()
    verify_frozen_state(execution_sha, shared_root)
    root = result_root(execution_sha, result_base)
    rows = selection_group_rows(root)
    readiness, status, ranking, selected = derive_readiness(rows)
    lock = {
        "schema_version": 1, "stage": "FPCT-1B", "execution_sha": execution_sha,
        "created_at": utc_now(), "selection_splits": ["fit", "calibration"],
        "selection_group_rows": len(rows),
        "selection_group_rows_sha256": sha256_bytes(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")),
        "global_readiness_status": status, "ready_pair_ranking": ranking,
        "selected_pilot": selected, "pair_readiness": readiness,
        "same_tokenizer_control_ranked": False,
        "m3_m4_changed_selection": False,
        "claim_boundary": "label-free structural opportunity and engineering readiness only",
    }
    if PILOT_LOCK.exists():
        raise AuditFailure("pilot selection lock already exists; refusing overwrite")
    write_json(PILOT_LOCK, lock)
    write_json(root / "pilot_selection_lock.json", lock)
    print(json.dumps({"status": status, "ranking": ranking, "selected_pilot": selected, "pilot_lock_sha256": sha256_file(PILOT_LOCK)}, sort_keys=True))


def combine_rows(root: Path, kind: str, columns: Sequence[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for phase in ("selection", "reporting"):
        for pair in PAIR_ORDER:
            for task in TASK_ORDER:
                path = shard_paths(root, phase, pair, task)[kind]
                rows.extend(read_csv(path, columns))
    return rows


def _aggregate_pair_task(
    group_rows: Sequence[Mapping[str, str]], sample_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for pair in PAIR_ORDER:
        pair_type = "same_tokenizer_control" if pair == "qwen3_1p7b" else "heterogeneous"
        for task in TASK_ORDER:
            for split_name in SPLIT_ORDER:
                base_splits = SELECTION_SPLITS if split_name == "fit_calibration" else {split_name}
                groups = [r for r in group_rows if r["pair"] == pair and r["task"] == task and r["split"] in base_splits]
                samples = [r for r in sample_rows if r["pair"] == pair and r["task"] == task and r["split"] in base_splits]
                pos_g = sum(int(r["has_primary_structural_m2"]) for r in groups)
                total_g = len(groups)
                high_g = sum(int(r["has_high_cardinality_m3"]) for r in groups)
                strict_g = sum(int(r["has_strict_m4"]) for r in groups)
                pos_s = sum(int(r["has_primary_structural_m2"]) for r in samples)
                high_s = sum(int(r["has_high_cardinality_m3"]) for r in samples)
                strict_s = sum(int(r["has_strict_m4"]) for r in samples)
                total_s = len(samples)
                low, high = wilson_interval(pos_g, total_g)
                output.append({
                    "schema_version": SCHEMA_VERSION, "pair": pair, "pair_type": pair_type,
                    "task": task, "split": split_name,
                    "observed_positive_group_count": pos_g,
                    "observed_total_group_count": total_g,
                    "observed_support_rate": pos_g / total_g if total_g else None,
                    "wilson95_low": low, "wilson95_high": high,
                    "bonferroni9_wilson_lcb_sensitivity": one_sided_wilson_lcb(pos_g, total_g, 0.05 / 9.0),
                    "direct_structural_positive_sample_count": pos_s,
                    "canonical_sample_count": total_s,
                    "direct_structural_support_ceiling": pos_s / total_s if total_s else None,
                    "high_cardinality_positive_group_count": high_g,
                    "high_cardinality_support_rate": high_g / total_g if total_g else None,
                    "high_cardinality_positive_sample_count": high_s,
                    "high_cardinality_support_ceiling": high_s / total_s if total_s else None,
                    "strict_positive_group_count": strict_g,
                    "strict_support_rate": strict_g / total_g if total_g else None,
                    "strict_positive_sample_count": strict_s,
                    "strict_support_ceiling": strict_s / total_s if total_s else None,
                })
    return output


def artifact_record(path: Path, row_count: int | None = None) -> dict[str, Any]:
    value = {"path": str(path), "sha256": sha256_file(path), "byte_size": path.stat().st_size}
    if row_count is not None:
        value["row_count"] = row_count
    return value


def finalize_reporting(shared_root: Path, result_base: Path) -> None:
    execution_sha = current_execution_sha()
    lock = verify_frozen_state(execution_sha, shared_root)
    root = result_root(execution_sha, result_base)
    pilot_lock = read_json(PILOT_LOCK)
    if pilot_lock.get("execution_sha") != execution_sha:
        raise AuditFailure("pilot lock execution SHA mismatch")
    parent_rows = combine_rows(root, "parent", PARENT_COLUMNS)
    sample_rows = combine_rows(root, "sample", SAMPLE_COLUMNS)
    group_rows = combine_rows(root, "group", GROUP_COLUMNS)
    pair_task_rows = _aggregate_pair_task(group_rows, sample_rows)
    if len(pair_task_rows) != PAIR_TASK_ROWS:
        raise AuditFailure(f"expected {PAIR_TASK_ROWS} pair_task rows")
    readiness, status, ranking, selected = derive_readiness([r for r in group_rows if r["split"] in SELECTION_SPLITS])
    if (status, ranking, selected) != (pilot_lock["global_readiness_status"], pilot_lock["ready_pair_ranking"], pilot_lock["selected_pilot"]):
        raise AuditFailure("reporting data changed the frozen pilot decision")

    formal = {
        "parent_support.csv": (PARENT_COLUMNS, parent_rows),
        "sample_support.csv": (SAMPLE_COLUMNS, sample_rows),
        "content_group_support.csv": (GROUP_COLUMNS, group_rows),
        "pair_task_support.csv": (PAIR_TASK_COLUMNS, pair_task_rows),
        "pair_readiness.csv": (READINESS_COLUMNS, readiness),
    }
    artifact_counts: dict[str, int] = {}
    for name, (columns, rows) in formal.items():
        artifact_counts[name] = write_csv(root / name, columns, rows)

    provenance = {
        "source_commit": "7207aafffc7f72976473815bc11102f8b06dddc1",
        "execution_commit": execution_sha,
        "v1_protocol_sha256": NORMATIVE_FILES["v1_protocol"][1],
        "v1_manifest_sha256": NORMATIVE_FILES["v1_manifest"][1],
        "approval_addendum_sha256": NORMATIVE_FILES["approval_addendum"][1],
        "v2_manifest_sha256": NORMATIVE_FILES["v2_manifest"][1],
        "preregistration_sha256": NORMATIVE_FILES["preregistration"][1],
        "analysis_code_sha256": lock["analysis_code"]["sha256"],
        "tokenizer_sha256": {key: value["tokenizer_bundle_sha256"] for key, value in lock["assets"]["tokenizers"].items()},
        "resolved_tokenizer_revisions": {key: {"resolved_tokenizer_revision": value["resolved_tokenizer_revision"], "source_repo_revision": value["source_repo_revision"], "resolution": value["revision_resolution"]} for key, value in lock["assets"]["tokenizers"].items()},
        "alignment_config_sha256": sha256_bytes(json.dumps(lock["alignment"], sort_keys=True, separators=(",", ":")).encode("utf-8")),
        "input_sha256": DATASET_CONTENT_SHA256,
        "content_group_manifest_sha256": lock["split_manifest"]["sha256"],
        "split_manifest_sha256": lock["split_manifest"]["sha256"],
        "pre_audit_lock_sha256": sha256_file(PRE_AUDIT_LOCK),
        "pilot_selection_lock_sha256": sha256_file(PILOT_LOCK),
    }
    write_json(root / "provenance.json", provenance)
    summary = {
        "protocol_id": PROTOCOL_ID, "provenance": provenance,
        "integrity": {"failure": False, "duplicate_indices": 0, "invalid_positive_mass": 0, "nonfinite_or_negative_mass": 0, "member_inconsistency": 0},
        "pair_task_support": pair_task_rows, "pair_readiness": readiness,
        "global_readiness_status": status,
        "pilot_selection": {"ranking": ranking, "selected_pair": selected, "locked_before_reporting": True},
        "claim_boundary": "candidate count establishes structural opportunity only; it does not establish query-time separability, accuracy benefit, or mathematical validity",
    }
    write_json(root / "audit_summary.json", summary)

    artifacts = {name: artifact_record(root / name, artifact_counts[name]) for name in formal}
    artifacts["audit_summary.json"] = artifact_record(root / "audit_summary.json")
    artifacts["provenance.json"] = artifact_record(root / "provenance.json")
    local_manifest = {
        "schema_version": 1, "execution_sha": execution_sha,
        "global_readiness_status": status, "selected_pilot": selected,
        "formal_local_artifacts": artifacts,
    }
    write_json(root / "local_result_manifest.json", local_manifest)
    print(json.dumps({"status": "REPORTING_COMPLETE", "global_readiness_status": status, "selected_pilot": selected, "formal_artifacts": artifacts}, sort_keys=True))


def verify_outputs(shared_root: Path, result_base: Path) -> dict[str, Any]:
    ensure_cpu_only()
    execution_sha = current_execution_sha()
    lock = verify_frozen_state(execution_sha, shared_root)
    root = result_root(execution_sha, result_base)
    parent_rows = read_csv(root / "parent_support.csv", PARENT_COLUMNS)
    sample_rows = read_csv(root / "sample_support.csv", SAMPLE_COLUMNS)
    group_rows = read_csv(root / "content_group_support.csv", GROUP_COLUMNS)
    pair_task_rows = read_csv(root / "pair_task_support.csv", PAIR_TASK_COLUMNS)
    readiness_rows = read_csv(root / "pair_readiness.csv", READINESS_COLUMNS)
    if len(pair_task_rows) != PAIR_TASK_ROWS:
        raise AuditFailure(f"pair_task_support row count is {len(pair_task_rows)}, expected {PAIR_TASK_ROWS}")
    for rows in (parent_rows, sample_rows, group_rows, pair_task_rows, readiness_rows):
        for row in rows:
            for value in row.values():
                if value.strip().lower() in {"nan", "inf", "+inf", "-inf"}:
                    raise AuditFailure("NaN/Inf found in formal CSV")
    recomputed = _aggregate_pair_task(group_rows, sample_rows)
    for frozen, fresh in zip(pair_task_rows, recomputed):
        for field in PAIR_TASK_COLUMNS:
            expected = csv_value(fresh.get(field))
            if str(frozen[field]) != str(expected):
                raise AuditFailure(f"aggregate mismatch in {field}: {frozen[field]} != {expected}")
    readiness, status, ranking, selected = derive_readiness([r for r in group_rows if r["split"] in SELECTION_SPLITS])
    pilot = read_json(PILOT_LOCK)
    if (status, ranking, selected) != (pilot["global_readiness_status"], pilot["ready_pair_ranking"], pilot["selected_pilot"]):
        raise AuditFailure("independent decision reduction disagrees with pilot lock")
    if any(row["pair"] == "qwen3_1p7b" and int(row["pilot_rank"] or 0) for row in readiness_rows):
        raise AuditFailure("same-tokenizer control participated in ranking")
    # Deterministic reread/reduction.
    second = _aggregate_pair_task(read_csv(root / "content_group_support.csv", GROUP_COLUMNS), read_csv(root / "sample_support.csv", SAMPLE_COLUMNS))
    if json.dumps(recomputed, sort_keys=True) != json.dumps(second, sort_keys=True):
        raise AuditFailure("nondeterministic reread/reduction")
    if sha256_file(REPO_ROOT / "math.md") != NORMATIVE_FILES["math_reference"][1]:
        raise AuditFailure("math.md changed")
    result = {
        "status": "VERIFIED", "execution_sha": execution_sha,
        "global_readiness_status": status, "ranking": ranking,
        "selected_pilot": selected, "parent_rows": len(parent_rows),
        "sample_rows": len(sample_rows), "content_group_rows": len(group_rows),
        "pair_task_rows": len(pair_task_rows), "pre_audit_lock_sha256": sha256_file(PRE_AUDIT_LOCK),
        "deterministic_reread": True, "m3_m4_selection_use": False,
        "same_tokenizer_ranked": False, "reporting_changed_selection": False,
        "normative_source_paths_verified": True,
    }
    write_json(root / "verification.json", result)
    print(json.dumps(result, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "freeze", "selection", "lock-selection", "reporting", "verify"))
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_SHARED_ROOT)
    parser.add_argument("--result-base", type=Path, default=DEFAULT_RESULT_BASE)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.mode == "prepare":
            prepare(args.shared_root)
        elif args.mode == "freeze":
            freeze(args.shared_root, args.result_base)
        elif args.mode == "selection":
            natural_audit("selection", SELECTION_SPLITS, args.shared_root, args.result_base)
        elif args.mode == "lock-selection":
            lock_selection(args.shared_root, args.result_base)
        elif args.mode == "reporting":
            natural_audit("reporting", REPORTING_SPLITS, args.shared_root, args.result_base)
            finalize_reporting(args.shared_root, args.result_base)
        else:
            verify_outputs(args.shared_root, args.result_base)
    except AuditFailure as exc:
        print(json.dumps({"status": "INCONCLUSIVE", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
